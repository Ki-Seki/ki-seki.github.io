---
date: '2026-10-24T13:41:46+08:00'
title: '大规模文生图模型预训练中的数据检索系统'
author:
  - Elliot Song
summary: '在大规模文生图模型训练时候，为了数据分析与定向筛选，需要构建交互式图文检索系统......'
cover:
  image: "assets/semantic-islands-cover.png"
  caption: "语义岛屿：从海量图片中检索相似样本（由 [ChatGPT Images](https://chatgpt.com/) 生成）"
tags:
  - text-to-image
  - training-data
  - retrieval
math: false
---

## Introduction

从零训练大型文生图模型时，需要构建面向大规模训练数据的，交互式图文检索与类簇浏览工具，帮助研究与数据团队从样本与类簇中查找和检查数据，并查看其内容、标注与来源。它最主要应用场景包括三个：

- **预训练数据筛选、清理与配比。** 通过类簇搜索与代表样本浏览检查数据分布，按既定安全与质量策略审核和剔除 NSFW 等不合格样本或类簇，并调整不同类簇或数据源的采样权重（比如对高质量的风景，设计图进行上采样）与配额，提高或降低其在训练集中的占比。
- **后训练数据回捞与筛选。** 围绕模型薄弱项、目标样例和特定能力需求，从数据池中检索、回捞相关候选样本，经人工筛选、质量检查和必要标注后组织为数据集，为 SFT 与 RL 中的特定能力（比如文字渲染，人像细节图，特定风格图等）增强提供数据支持。
- **评测数据泄漏风险排查。** 将评测图像或构建评测提示词时使用的参考图片作为查询，单独或批量检索相同及近重复的训练样本，辅助排查训练与评测数据的重合。

构建这种大规模查询系统并不是一个 trivial 的任务，下面就针对构建 billion+图文样本的检索系统，介绍设计与实现思路。

## 检索系统的构建

billion 样本级别的检索系统不可能使用精确查找，所以一定是一个最近邻查询系统，简单来说就是构建 embeddings，把样本映射到向量空间，通过最近邻算法，比如我们用到的 FAISS[^douze_faiss][^johnson_faiss_gpu] 中的 IVF-PQ[^jegou_pq]，进行检索。

构建过程主要包括数据清单整理与样本映射、图像表征提取、聚类与向量量化、分片索引构建及完整性校验，如图 1 所示。

{{< media
src="assets/offline-build-workflow.svg"
caption="图 1：离线索引构建的阶段依赖与资源划分。实线表示必需产物依赖，灰色虚线表示可选能力依赖，橙色虚线表示人工确认；实际并发由资源预算与编排策略决定。"
>}}

面向文生图训练数据的检索系统通常要同时支持文本语义查询和图片语义查询，所以文和图应该能够被映射在同一向量空间，一般要采用通过图文对比学习训练出的模型[^radford_clip][^zhai_siglip]，这里我们采用 SigLIP2[^tschannen_siglip2] 提取图像表征。接下来，我们可以基于 FAISS，将倒排文件索引（Inverted File，IVF）[^sivic_video_google]与乘积量化（Product Quantization，PQ）[^jegou_pq]结合，构建压缩检索索引。这样可以在保证检索效率的同时，显著降低存储开销。下面依次介绍数据存储格式、分布式表征提取、聚类与压缩索引，以及全量索引的构建。

### 数据存储格式与管理

首先要明确我们的数据存储格式。逻辑上来看，它实际上就是以图片为键的一个巨大的表，图片最核心的 sidecar 数据是他的文本 caption 描述，毕竟我们在训练文生图模型。其他 sidecar 数据可能包括布局、标签、来源、美学打分、各种类别标签等等。

数据的物理布局上，我们可以考虑把图像数据单独存储成一个个分片的 main parquets[^parquet]，保存图像数据或引用，而各类其他关联信息可以通过和 main parquet 对齐的 sidecar parquets 存储，从而实现图像与其属性的高效管理与访问。样本通过 `(dataset_revision_id, source_file_id, source_row)` 三元组统一定位，分别对应数据集版本、源分片编号和分片内行号，便于统一访问与来源追溯。

{{< media
src="assets/data-storage-format.svg"
caption="图 2：数据存储的逻辑视图与物理布局。样本由数据集版本、源分片编号和分片内行号三元组定位。"
>}}

这样的数据管理方式能够尽可能同时满足数据团队和训练团队的需求：

- **数据团队需求。** 大规模、多阶段训练中的数据筛选、重采样与标注迭代可能形成数十种派生数据集或版本，增加数据管理和样本追溯的难度。
- **训练团队需求。** 训练时的尽量少的数据处理开销，能够通过一些简单的谓词过滤快速定位所需样本，减少对图像数据的重复读取和解码，提高训练效率。当然，有时候分片内部也可能存在不均匀的数据分布，数据团队可能需要对分片内的数据进行重采样或调整，以保证训练数据的均衡性。

### 分布式批量表征提取

我们根据待检索的数据范围整理输入清单，以 Parquet 分片为处理单位，记录源文件位置与行数。图像既可从分片内嵌的图片字节读取，也可从分片记录的 S3 路径获取，随后按固定配置完成解码、尺寸调整和像素归一化。

图像编码采用 SigLIP2 SO400M（patch14，384 × 384 输入），为有效图像生成 1,152 维向量，经 L2 归一化后以 FP16 存储。在线查询使用同一模型的图像或文本编码器，将参考图片和文本映射到共享表征空间，从而在同一图像索引上支持以图搜图和以文搜图。

对于十亿级图像数据，表征提取不仅涉及 GPU 推理，还包含对象存储读取、图像解码和结果写出。我们以 Parquet 分片为任务单元，按分片大小估计初始负载，将任务分配到多个计算节点，并由节点内的多个 GPU 工作进程并行处理，各分片独立写出结果。

不同图片格式、尺寸和存储读取延迟会造成任务耗时不均。我们从任务调度、数据供给和推理组批三个方面提高处理效率：

- **节点内动态任务调度。** 工作进程完成任务后继续领取尚未处理的分片，配合有限的提前领取窗口，避免少数进程积压慢任务而其他进程空闲，缓解负载不均与任务收尾时的长尾等待，缩短整批任务的完成时间。
- **流水线重叠与有界预取。** 在 GPU 计算当前批次时提前准备后续数据，使读取、解码与推理尽量重叠，减少 GPU 等待数据的时间；同时限制每个进程在途的分片数量，避免过量占用资源。
- **跨分片组批。** 在同一工作进程内，将不同 Parquet 分片中就绪且解码路径相同的图片合并为推理批次，减少分片边界或数据筛选造成的小批次推理，提高批次填充率。推理结果仍按原始分片和行号回写，保持样本对应关系。

{{< media
src="assets/embedding-extraction.svg"
caption="图 3：分布式表征提取示意。(a) 分片按估计负载分到各节点，节点内 GPU 工作进程空闲即领取下一个分片，相比固定分配可避免慢分片造成的长尾；(b) 单个工作进程内读取、解码、推理与写出重叠执行，不同分片中就绪的图片可拼成满批次，结果按原分片和行号写回。时长与行号仅作示意。"
>}}

为降低长时间任务中断后的重算成本，我们按分片内的数据块保存检查点。构建完成后，核对源分片与输出的覆盖范围和行数，汇总下载、解码等失败状态，并抽查样本标识的对应关系以及向量的维度、数据类型和 L2 范数。失败样本保留原始位置与状态，避免因直接丢弃记录造成行号错位；缺失或异常输出需处理并复核后再交付。

### 聚类与压缩索引

按每张图像对应一个 1,152 维 FP16 向量计算，十亿张图像仅向量数值就占 2.304 TB（十进制，不含图像、标注及文件格式开销），运行时内存太大，并且逐一比较也难以满足交实时检索需求。我们采用 IVF-PQ，将缩小搜索范围与向量压缩结合。

#### IVF 聚类

IVF 需要聚类得到簇中心，PQ 码本训练也要在各个子空间上分别聚类，而且要在 IVF 中心确定之后进行（见下文），所以两者都要先跑聚类算法。在全量的数据上跑聚类非常慢；通常我们会取一个较小的样本集，在抽样 N 条向量数据上训练。

下面以 1.4B 预训练库的参考构建配置为例，N取100M 条向量：按各输入 embedding 分片的行数比例分配配额，再在每个分片内按行区间分层抽样，按行区间读取。这种分层块抽样可利用对象存储的范围读取，避免为取少量样本下载整个向量文件；它并非逐行独立的全局均匀随机抽样。该样本集划分为 9900 万条 IVF 训练样本与 100 万条留出样本，后者只用于该阶段的聚类评估。

{{< media
src="assets/ivf-index-construction.svg"
caption="图 4：IVF 的抽样、中心训练与查询范围示意，以 1.4B 预训练库的参考构建配置为例；二维簇位置和查询探测范围仅作示意。"
>}}

IVF 的簇数预设为 `K=10,000`，训练根据上述 9900 万条向量学习中心位置。随后将全量向量分配到最接近的中心，并将同一簇的样本组织为倒排列表。查询时先找到与查询向量接近的若干中心，再扫描对应列表，避免遍历整个数据集。

#### PQ 压缩与编码

在压缩方面，我们对向量与所属聚类中心之差，即残差，进行乘积量化（PQ）。从同一份 100M 条样本资产中进行再采样（依然为了加速聚类速度，因此没有全部使用 100M 条样本），按样本分片均衡分配读取配额、取各分片的前缀，共选取 210 万条向量，其中 200 万条用于 PQ 训练，10 万条留出用于量化重构误差评估。

{{< media
src="assets/pq-encoding.svg"
caption="图 5：PQ64 的码本抽样训练与全量编码示意。码本在整个索引中共享；容量比较仅针对向量编码，不代表完整索引大小。"
>}}

PQ 的分段数与每段编码位数也由配置决定：这里采用 64 段、每段 8 位的 PQ64 配置，每段为 18 维，8 位编号对应每个子码本中的 256 个代表向量。先计算 PQ 训练样本相对于已确定的 IVF 中心的残差，再汇集各 IVF 簇中相同段位置的子向量，分别执行 256 类的 K-means，得到 64 套在整个索引中共享的码本。

编码时，每个残差子向量选择对应码本中最近的代表向量，仅保存其 0–255 的编号。由此，原本占 2,304 字节的 1,152 维 FP16 向量被表示为 64 字节的量化编码。

#### IVF-PQ 的查询

建好的 IVF-PQ 索引中，倒排列表的每个条目只保存样本标识和 64 字节的 PQ 编码，不保存原始向量。查询时也不需要把编码还原成向量，近似距离可以直接查表得到，过程分为四步，如图 6 所示。

{{< media
src="assets/ivf-pq-search.svg"
caption="图 6：IVF-PQ 查询时的近似距离计算。先选出最近的 nprobe 个倒排列表；对每个列表计算查询残差与各子码本代表向量的距离表，条目的 64 个编号查表累加即得近似距离，最后保留最近的 k 个。数值与样本编号仅作示意。"
>}}

1. **选择倒排列表。** 计算查询向量 q 与全部 10,000 个聚类中心的距离，只搜索最近的 `nprobe` 个簇对应的倒排列表，其余列表直接跳过。
2. **构建距离表。** PQ 编码的是向量相对所属聚类中心的残差，因此对每个被探测的列表，先计算查询残差 r = q − c，按与编码相同的方式切成 64 段、每段 18 维；再计算每一段与对应子码本中 256 个代表向量的距离，得到一张 64 × 256 的距离表。每个列表只需计算一次，列表内所有条目共用这张表。
3. **查表累加。** 条目的 64 个编号分别指向距离表中对应段的一个位置，取出这 64 个值相加，就是 q 与该样本的近似距离。每个条目只需 64 次查表与加法，无需解码向量，这正是 PQ 能快速扫描大量候选的原因。
4. **保留最近的 k 个。** 汇总所有被探测列表中的条目，按近似距离保留最小的 k 个作为候选。



### 全量索引构建

聚类中心和 PQ 码本确定后，全量簇分配与编码可分布式执行。随后将样本标识和量化编码按簇归并，组装为多个 FAISS 索引分片，避免单个索引文件太大导致无法高效加载与查询；同一索引的各分片共享中心与码本。索引的聚类数、PQ 编码长度与查询时探测的簇数（`nprobe`）共同影响资源开销与检索效果：增加探测簇数通常有助于提高召回，但需要扫描更多候选；PQ 则以量化误差换取更低的存储与扫描成本，因此基于该索引的召回属于近似检索。

以1.4B 样本库例子来看，构建出的索引的 specification：

|  |  |
|---|---|
| 样本数 | 1,386,418,079，来自 1,264 个 embedding 分片 |
| 表征 | SigLIP2 SO400M，1,152 维 FP16，原始向量约 3.19 TB |
| 距离 | 归一化向量上的平方 L2，与余弦相似度等价 |
| IVF | K = 10,000，球面 mini-batch k-means[^dhillon_spherical][^sculley_minibatch]；9,900 万条训练、100 万条留出；全量分配后 10,000 个列表均非空，单个列表 297–946,850 条 |
| PQ | 64 段 × 8 位，每段 18 维、256 个代表向量；200 万条训练、10 万条留出；每个向量 64 字节 |
| 索引分片 | 8 个，按簇编号连续切分，每片约 1,250 个簇、1.67–1.80 亿条；单片 12.1–13.0 GB，合计 100.2 GB，约为原始向量的 1/32 |
| 构建 | 全量分配与编码用 64 个并行任务；每个分片组装约 1.6–3.6 分钟；每个分片回读后用 nprobe = 32 做检索校验；从 1 亿抽样完成到校验通过约 7.3 小时 |
| 在线服务 | 8 张 GPU 各加载一个分片；默认 nprobe = 32，即每次查询只扫描 0.32% 的列表，按平均列表长度约 440 万条编码 |

## 在线检索系统

### 图文语义查询服务

在线检索接受参考图像、文本或二者的组合。系统使用与离线建库配套的 SigLIP2 编码器生成归一化查询向量；多图或图文输入可先汇聚为一个查询向量，也可分别召回后按相似度或排名融合[^cormack_rrf]结果。每路查询并行搜索各索引分片，服务端按样本标识合并去重，并按 PQ 近似距离排序保留候选。

{{< media
src="assets/online-query-flow.svg"
caption="图 7：一次在线查询的处理流程。查询可以是文本、图片或二者组合，编码为查询向量后并行搜索各索引分片，服务端按样本标识合并去重，并按距离排序保留候选。"
>}}

结果浏览将检索排名与样本内容、标注和来源关联起来。用户可从缩略图进入详情，查看描述文本、原始 JSON 及与标注对齐的展示图，以及数据集版本、源文件路径和行号等信息，如图 8 所示。

{{< media
src="assets/search-ui-example.png"
caption="图 8：检索系统界面示例。在 1.4B 样本库中以文本“A watercolor illustration of a red fox in a snowy pine forest.”发起查询，展示查询配置、多图融合方式、检索参数与前 20 个真实结果。"
>}}

### 类簇浏览与数据配比

预训练数据清理与配比，更多是在类簇层面完成的。我们以离线聚类得到的类簇为单位提供浏览工具：每个簇附有自动生成、可搜索的语义标签，可按美学、紧密度或多样性排序；簇内样本按与簇中心的余弦相似度排列，便于快速判断一个簇的内容与质量。簇中心经 UMAP[^mcinnes_umap] 投影到二维后可交互浏览，用于从整体上观察数据分布。

在此基础上，多名标注者可对低质量或不合规的类簇投票，并导出删除清单；配比工具则按「簇 × 数据集」求解采样配额，研究人员设定各数据源的目标占比后，即可导出相应的采样计划。

### 额外功能

除了基础的 FAISS ANN，借助 AI编程，系统还可以很方便地额外提供一些功能，下面是实践中我们常用的几种：

- **批量独立查询。** 针对一组模型坏例或评测样本，批量独立查询可逐条检索并对照结果，减少反复提交单次查询的操作。研究人员可按数据来源、尺寸等属性缩小候选范围，将人工选定的样本整理为跨查询保留的素材集，并导出包含源路径和行号的 JSONL 清单，衔接后续数据回捞、复核和重标注。查询参数与结果快照还可保存为分享链接，便于团队围绕同一批候选协作检查。
- **辅助结果核验。** 为辅助结果核验，系统对指定数量的前排候选回源读取图像，按当前编码配置重新生成完整向量，计算不经 PQ 近似的余弦相似度，并支持在这些候选内重排；同时计算 pHash[^zauner_phash] 汉明距离并比对文件 SHA-256，标出近重复与字节级完全相同的图片。
- **更灵活的查询组合。** 多张参考图有三种合并方式：取共同特征（向量平均）、像其中任意一张即可（各自检索后取最优）、多图共识（按排名融合，不易被个别离群图带偏）。图文同时给出时，可用滑杆调节两者权重，默认图片 80%、文本 20%。任一结果都可以一键「用此图搜」或加入参考图，也可以直接粘贴存储路径作为查询图片。
- 等等

## Conclusion

本文介绍了大规模图像训练数据的检索与管理方法，包括数据存储与管理、向量构建、聚类与压缩索引，以及在线检索与样本检查的流程与技术细节。

## Citation

{{< bibtex >}}

## References

[^douze_faiss]: **Douze, Matthijs, et al.** *The Faiss Library*. 2024. *arXiv*, https://arxiv.org/abs/2401.08281.

[^johnson_faiss_gpu]: **Johnson, Jeff, Matthijs Douze, and Hervé Jégou.** "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*, vol. 7, no. 3, 2021, pp. 535–547. *arXiv*, https://arxiv.org/abs/1702.08734.

[^jegou_pq]: **Jégou, Hervé, Matthijs Douze, and Cordelia Schmid.** "Product Quantization for Nearest Neighbor Search." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 1, 2011, pp. 117–128. *IEEE Xplore*, https://doi.org/10.1109/TPAMI.2010.57.

[^sivic_video_google]: **Sivic, Josef, and Andrew Zisserman.** "Video Google: A Text Retrieval Approach to Object Matching in Videos." *Proceedings of the Ninth IEEE International Conference on Computer Vision*, 2003. *IEEE Xplore*, https://doi.org/10.1109/ICCV.2003.1238663.

[^radford_clip]: **Radford, Alec, et al.** "Learning Transferable Visual Models From Natural Language Supervision." *Proceedings of the 38th International Conference on Machine Learning*, vol. 139, Proceedings of Machine Learning Research, 2021, pp. 8748–8763. *PMLR*, https://proceedings.mlr.press/v139/radford21a.html.

[^zhai_siglip]: **Zhai, Xiaohua, et al.** "Sigmoid Loss for Language Image Pre-Training." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2023, pp. 11975–11986. *arXiv*, https://arxiv.org/abs/2303.15343.

[^tschannen_siglip2]: **Tschannen, Michael, et al.** *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. 20 Feb. 2025. *arXiv*, https://arxiv.org/abs/2502.14786.

[^parquet]: **The Apache Software Foundation.** "Apache Parquet." https://parquet.apache.org/.

[^dhillon_spherical]: **Dhillon, Inderjit S., and Dharmendra S. Modha.** "Concept Decompositions for Large Sparse Text Data Using Clustering." *Machine Learning*, vol. 42, 2001, pp. 143–175. *Springer*, https://doi.org/10.1023/A:1007612920971.

[^sculley_minibatch]: **Sculley, D.** "Web-Scale K-Means Clustering." *Proceedings of the 19th International Conference on World Wide Web*, ACM, 2010, pp. 1177–1178. *ACM Digital Library*, https://doi.org/10.1145/1772690.1772862.

[^cormack_rrf]: **Cormack, Gordon V., Charles L. A. Clarke, and Stefan Büttcher.** "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval*, ACM, 2009, pp. 758–759. *ACM Digital Library*, https://doi.org/10.1145/1571941.1572114.

[^zauner_phash]: **Zauner, Christoph.** *Implementation and Benchmarking of Perceptual Image Hash Functions*. Master's thesis, Upper Austria University of Applied Sciences, 2010. https://www.phash.org/docs/pubs/thesis_zauner.pdf.

[^mcinnes_umap]: **McInnes, Leland, John Healy, and James Melville.** *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. 2018. *arXiv*, https://arxiv.org/abs/1802.03426.
