---
date: '2026-11-13T11:28:01+08:00'
title: 'BBox-SigLIP: A Better Way to Find Images by What and Where'
author:
  - Elliot Shichao Song
summary: ''
tags:
  - retrieval
  - siglip
  - contrastive-learning
  - spatial-reasoning
  - ANN
math: true
---

> **大纲。** 下面每节列出这一节要讲的内容，正文和图之后补。正文只讲 BBox-SigLIP，其他方案的对比放在附录。

- **可公开的资源（占位，待整理）：**
  - 模型：BBox-SigLIP 第 208,448 步
  - 实验记录：W&B [songsc-team/bbox-siglip](https://forge.coreweave.com/wandb/songsc-team/bbox-siglip)（9 个 run，待设为公开）
  - Demo：手机相册里用格子指定位置找图
  - 示例训练数据：100 对图像与 JSON，JSON 只保留训练时用到的字段
  - 评测集：949 张图、3,725 个区域查询，以及镜像、拼图对调的构造方式
  - 推理代码：JSON 查询怎么拼、怎么打分
  - 评测代码：内部四个测试，加上 SPEC 的「格子换成框」转换
  - 训练代码：采样和损失

## 引言

用 CLIP、SigLIP 这类模型搜图，查询是一句话，比如「一只狗」，返回的是内容最像的图。但有时候，人想要的不只是图里有什么，还有它在哪儿。比如翻手机相册的人可能想找到「狗在右下角的那张照片」（图 1）。普通的图文检索只匹配内容，同一只狗在左边还是在右边，分数几乎一样。

一些图库网站曾做过类似的功能，但都没有开源实现。比如 iStock 可以在 3×3 的格子里指定哪里要留白 [^istock_copyspace]，Shutterstock 试过把关键词拖到版面上的某个位置再搜 [^petapixel_shutterstock]，Adobe Stock 能在示例图里拖动一个物体再找相似的图 [^adobe_findsimilar]。

学术上，这个问题探索得还不多。已有的方法通常在图像一侧下功夫：要么给每张图存一张空间特征图 [^xu_conceptmap][^mai_spatialsemantic][^kilickaya_cal]，要么给每个候选框各存一个向量 [^xie_fgclip][^chen_cloc][^fu_objembed]。用这样的表示建近似最近邻（ANN）索引，索引要比一图一向量大一到两个数量级。

因此我们提出 BBox-SigLIP，一个按精细内容和精确位置一起找图的图文表征模型。用它建库，**每张图只需要存一个向量**，和普通的 SigLIP 检索完全一样，IVF-PQ 这类 ANN 索引照常能用。位置和描述都写在查询里，是一段 JSON：`bbox` 是框，坐标 $[x_1, y_1, x_2, y_2]$ 归一化到 0–1000；`description` 说框里应该有什么。图 1 是一个例子：查询给出狗的精确位置和一段细致的描述，只有狗确实在框里的那张图分高。

{{< media
src="assets/query-and-scores.svg"
caption="图 1：BBox-SigLIP（第 208,448 步）是一个双塔模型，包含文本和图像 encoders。查询把框和描述写成一段 JSON，进文本编码器得到一个向量 f(b, t)；图库里每张图进图像编码器，各得到一个向量 g(I)，可以离线算好。红色是查询给定的框：只有狗确实在框里的第一张图分高；同一张图左右镜像、狗在别处、框里是别的东西，分数都低。"
>}}

BBox-SigLIP 沿用 SigLIP2 的结构，没有加任何新模块：JSON 查询当作一段文本整个送进文本编码器，图像送进图像编码器，分数是两个向量的点积（图 1）。它和 SigLIP2 的区别全在训练：我们用带框的数据把两个编码器一起微调，让文本编码器学会读坐标，让图像编码器把「什么东西在哪儿」压进一个 1,152 维的向量。

<!-- TODO：到最后再润色贡献 -->
本文的贡献有四点：

1. **提出 BBox-SigLIP**，一个按内容和位置一起找图的图文表征模型，图库里每张图只存一个向量。它沿用 SigLIP2 的结构，不加新模块，只是把框写进查询，再用带框的数据微调。同一个模型也能用在另外两个方向上：给定图和描述找框，给定图和框找描述（图 2）。
2. **设计了一组专门测框有没有起作用的测试。** 区域检索的命中率高，不代表模型看了框：不看框也能拿到很高的命中率。镜像框、拼图对调这些测试每次只改一样东西，其余不变，分数的变化只能来自框和位置。
3. **发现一张图一个向量能学会位置，但要训很久。** 拼图对调把两张图拼在一起，查询问其中一半，只看内容的模型只能拿到 50%。BBox-SigLIP 训到第 208,448 步时，左右拼是 93.6%，上下拼是 94.1%；原版 SigLIP2 是 50.2% 和 50.1%。这个能力出来得很慢：第 2,500 步时只比 50% 高三四个点，第 1 万步时是 63.2% / 70.8%，第 124,745 步时才到 91.9% / 93.3%。
4. **在公开基准 SPEC 的绝对位置测试上，图找文 97.8%，文找图 98.2%**（九选一，随机是 11.1%）。SPEC 论文里的模型读的是写在句子里的位置，CLIP 是 12.6% / 12.3%，作者专门微调过的 CLIP 是 30.6% / 34.2%；我们把位置以框的形式交给模型，两边的输入方式不同，细节见评测。



## 问题定义 

我们把图像、区域和描述放在一起，看成一个三元组 $(I, b, t)$：图 $I$ 里，框 $b$ 框住的地方是 $t$。框通常可以用四元组表示 $b = [x_1, y_1, x_2, y_2]$ ，本文我们选择将坐标归一化到 0–1000；$t$ 是一段自由文本，可以是一个词，也可以是几句细致的描述。

记 $t$ 所说的东西的紧框为 $b^\star$，只要 $b^\star \subseteq b$，三元组就算匹配；图里没有这个东西，或者 $b \cap b^\star = \varnothing$，就算不匹配。只部分重叠的框介于两者之间属于部分匹配。我们希望模型要学会一个打分函数 $s(I, b, t)$：三元组匹配时分高，不匹配时分低，部分匹配时分居中。

现实世界中，如果三个量里已知两个、找第三个，恰好可以划分出来真实的三类任务（图 2）。

{{< media
src="assets/three-tasks.svg"
caption="图 2：同一个三元组（图 I、框 b、描述 t），已知其中两个、找第三个，就是三类任务。"
>}}

- 已知 $(b, t)$，找 $I \in \mathcal{I}$：按内容和位置找图，也就是本文的 what-and-where 图像检索；
- 已知 $(I, t)$，找 $b \in \mathcal{B}$：视觉 grounding 和开放词表检测，比如在 RefCOCO [^yu_refcoco] 上按指代表达找框；
- 已知 $(I, b)$，找 $t \in \mathcal{T}$：区域分类和区域描述，比如给定框判断类别，或者 dense captioning。

面对本文最早提出的问题，在大量候选图中，找到与给定框和描述匹配的图像，三种建模方式都可以解决，但显而易见第一种方式最优，本文的方法也限制在第一种。

{{% admonition type="tip" title="Proof" open=true %}}
简单证明如下。

以第三种方式为例，假设存在一个能完成 $f_3: (I, b) \mapsto t$ 的区域分类模型，那么当一个查询 $(b, t)$ 来找图时，我们可以取 $I^\star = \arg\max_{\color{red} I \in \mathcal{I}} \langle \color{red} f_3(I, b) \color{black},\ t \rangle$.

不过显然是第一种最佳，图库里的大量图离线编码好即固定，在线查询时只需要现算$(b, t)$ 这一侧，若 $f_1: (b, t) \mapsto I$，则可以直接取 $I^\star = \arg\max_{\color{red} I \in \mathcal{I}} \langle f_1(b, t),\ \color{red} g(I) \color{black} \rangle$.

两者的区别是，由于循环变量是图像，而第一种方式图库离线已经计算完毕，所以他的时间复杂度是 $O(|\mathcal{I}|)$，而第三种方式则会是 $O(|\mathcal{I}| \cdot \text{计算 } f_3(I, b) \text{ 的时间})$。

对于第二种方式也同理。
{{% /admonition %}}



以往的工作大多只看其中一两个方向，比如 UniRef [^zheng_uniref] 和 CapDet [^long_capdet] 把后两个方向放在一起训练。

放进同一个三元组里看，三类任务可以共用一个 $s$，区别只在于哪些量要提前算好。

检索时，图库里的几百万张图必须先编码好，查询来了只能现算查询这一侧，所以分数要写成两个向量的点积，一个只看查询，一个只看图库，和 CLIP [^radford_clip]、SigLIP [^zhai_siglip]、DPR [^karpukhin_dpr] 这类双塔模型一样。

$I$ 和 $t$ 各在一边，框 $b$ 跟谁放在一起，是各类方法最大的分歧，而这又取决于方法原本要做哪个方向：

- 框跟着图，$s = \langle f(t),\ g(I, b) \rangle$。已知 $(I, b)$ 找 $t$ 时，图和框一起给出，区域特征算一次，就能和任意多句描述比。FG-CLIP、CLOC 这类区域级图文对齐都这样拆。拿它来找图，查询来之前不知道框在哪儿，只能给每张图的每个候选框各存一个向量。
- 框跟着描述，$s = \langle f(b, t),\ g(I) \rangle$。已知 $(b, t)$ 找 $I$ 时，框和描述一起给出，图库那一侧与查询无关，建一次就行。Concept Map 这类按版式找图的方法属于这一种，但它们的 $g(I)$ 是一张空间特征图，位置摊在特征的空间布局里。

图 3 把这几类方法都放到找图的场景里对比。本文也是后一种，但要求 $g(I)$ 只是一个向量：

$$s(I, b, t) = \langle f(b, t),\ g(I) \rangle$$

其中 $g(I)$ 是每张图一个低维向量（BBox-SigLIP 里是 1,152 维），离线算好，直接放进 ANN 索引。代价是 $g(I)$ 要事先把「什么东西在哪儿」都压进这一个向量：它没有空间布局可读，位置只能靠训练学进去。

这样拆也能做另外两类任务。固定一张图 $I$，$g(I)$ 只算一次；换框或者换描述，只需要重新算 $f(b, t)$。给一组候选框，就能挑出描述最可能在的位置；给一组候选描述，就能挑出框里最可能是什么。图 2 的中栏和右栏就是这样得到的，右栏的候选描述是 COCO [^lin_coco] 的 80 个类别名。

后面的评测也沿用这个三元组：每个测试都只改一样东西，其余不变。只换图 $I$，是区域检索；只把框 $b$ 左右镜像过去，是镜像框；图里的东西不变、只换它们的位置，是拼图对调。

## 相关工作

按论文实际训练和评测的任务，把相关工作分进三类（下表）。第一类是本文的任务，列出我们找到的全部工作；后两类工作很多，按研究方向各举两篇代表，其中一篇是 2026 年的最新工作，完整的梳理见综述 [^xiao_vgsurvey] [^wu_ovsurvey]。只靠文字找图、查询里不带位置的检索，以及从一张图同时输出框和描述的 dense captioning，不在这三类里，不列。

| 任务 | 工作 |
|---|---|
| 已知 $(b, t)$<br>找 $I$ | Concept Map（SIGIR'10） [^xu_conceptmap]<br>Spatial-Semantic Image Search（CVPR'17） [^mai_spatialsemantic]<br>Shutterstock 构图检索专利（US Patent'19） [^ranzinger_patent]<br>CAL（WACV'21） [^kilickaya_cal]<br>Changpinyo 等人（ICCV'21，口述加鼠标轨迹） [^changpinyo_whatwhere]<br>**BBox-SigLIP（本文）** |
| 已知 $(I, t)$<br>找 $b$ | 视觉 grounding：MDETR（ICCV'21） [^kamath_mdetr]、Rex-Thinker（ICLR'26） [^jiang_rexthinker]<br>开放词表检测：GLIP（CVPR'22） [^li_glip]、ObjEmbed（ICML'26） [^fu_objembed] |
| 已知 $(I, b)$<br>找 $t$ | 区域级图文对齐：CLOC（ICML'25） [^chen_cloc]、FG-CLIP 2（ICML'26） [^xie_fgclip2]<br>区域描述：UniRef（EMNLP'22） [^zheng_uniref]、GAR（ICLR'26） [^wang_gar] |

- 按问题定义里的两种拆法归类（图 3）：
  - 框跟着描述：Concept Map、Spatial-Semantic Image Search、CAL 和我们，做的都是已知 $(b, t)$ 找 $I$；区别在图库一侧，前三者存空间特征图，我们存一个向量。Changpinyo 等人也属于这一种，图库同样每张图一个向量，区别在查询（见下文）；
  - 框跟着图：FG-CLIP、CLOC、ObjEmbed，主要做区域识别（已知 $(I, b)$ 找 $t$）和定位（已知 $(I, t)$ 找 $b$），按框从图像特征上取区域特征，或者每个候选框存一个向量。ObjEmbed 也能找图，但查询里没有位置。任务方向不同，不展开比较。

{{< media
src="assets/where-the-box-goes.svg"
caption="图 3：各类方法把框 b 放在哪一边。框和描述在查询一侧时，图库向量和查询无关，建一次就行；框在索引一侧时，要按框从图像特征上取区域特征，或者给每个候选框存一个向量。"
>}}

- 下面只展开和我们定位相同或相近的工作。

### 同一定位的工作：已知框和描述，找图

- Image Search by Concept Map（SIGIR 2010，MSRA）：
  - 交互：在空白画布上打关键词，每个关键词带一个可以拉伸的矩形，默认占画布的 1/9。
  - 查询：关键词先拿去文本搜图，取前 50 张，用 affinity propagation 聚类挖出 6 个「视觉实例」；位置变成一个二维高斯。
  - 图库：每张图切成 9×9 个格子，每格一个视觉词袋直方图（SIFT 6K 词表或 HSV 192 色）。
  - 打分：每个实例在每个格子上算出现程度，外观分乘空间分；多个概念取平均，再减去离散程度。
  - 评测：只在文本搜索结果上重排（每个关键词组合取前 500 张），42 个任务、33 个概念，NDCG，外加 20 人的用户研究。
- Spatial-Semantic Image Search by Visual Feature Synthesis（CVPR 2017，Adobe 和 Portland State）：最接近的前作。
  - 查询：画布变成 31×31×300 的网格，框里的格子填 Word2Vec 向量；三层卷积网络把它「合成」成 7×7×832 的视觉特征。
  - 图库：固定的 ImageNet GoogLeNet 卷积特征（摊平约 4 万维），不训练，所以换查询模型不用重算图库。
  - 损失：0.6 × 相似度 + 0.3 × 分类 + 0.1 × 三元组；只用单概念查询训练，多概念时逐个合成再取逐元素最大值。
  - 评测：COCO 加 Visual Genome 共 179,877 张图、269 个概念、28,699 个测试查询；相关度是同类框 IoU 的平均；NDCG、mAP、Spearman 都超过文本搜索、Concept Map 和直接用目标图真实特征的 oracle；合成一个查询 1 毫秒，500 万张图一秒内搜完。
- Structured Visual Search via Composition-aware Learning（WACV 2021）：
  - 查询：COCO 80 类的框组成的画布（32×32×80 的 one-hot）。
  - 图库：冻结的 ResNet-50 加三层卷积，输出 7×7×256 的特征图。
  - 损失：让两张图特征的点积去拟合它们版式的 IoU，版式变了，特征跟着变。
  - 评测：COCO 上 5,000 个查询、约 7 万张图的图库，mAP@1 81.2；另在 HICO-DET 上测。
- 和我们比：这三项都只认固定的类别词，图像一侧是空间特征图（81 个格子或 7×7 网格，几千到几万维），位置信息摊在特征的空间布局里，不用学。我们用自由文本描述，图像一侧是一个 1,152 维的整图向量，可以直接用 ANN，代价是位置要靠训练学进去。

### 相近的设定：口述加鼠标轨迹，找图

- Telling the What while Pointing to the Where（ICCV 2021，Google）：「what+where 检索」这个叫法出自这篇。
  - 查询：标注员一边口述整张图（Localized Narratives），一边用鼠标指着说到的东西。每段话说出时鼠标扫过的范围取成一个框，在时间和空间上各放宽一圈，再和文字一起送进文本编码器（早融合）。
  - 图库：ResNet-152 整图特征加 16 个 Faster R-CNN 区域特征，过 6 层 Transformer 后取平均，每张图一个向量，点积打分。
  - 训练：先在 Open Images 50.7 万张图的口述数据上预训练，再在 Flickr30k 的口述数据上微调；文字用 1 万个子词，随机初始化。
  - 评测：Flickr30k、COCO、ADE20K 的口述数据，加上轨迹后 R@1 比只用文字高 7 到 13 个点（Flickr30k 83.4 → 90.6）。
- 和我们比：图库一侧同样每张图一个向量，区别在查询。它的位置来自说话时鼠标恰好停在哪里，不是用户有意圈出的范围：轨迹和语音对不齐，要放宽一圈才能用，"we can see" 这类虚词也带着轨迹；而且查询是一整段描述全图的口语，用户得把整张图讲一遍。我们的查询是用户直接画的框加一句描述，找什么、在哪里都是明确的。

### 按版式找图的产品

- iStock CopySpace（2005）：一个 3×3 格子，每格选「放文字」「放图」或「都行」，可以和关键词、颜色、尺寸一起筛。所有图片都预先处理过，找出连续、平坦的留白区域。位置信息是每张图预先算好的，只回答「哪里空」，不回答「哪里有什么」。
- Shutterstock Composition Aware Search（2017 年在 Shutterstock Labs 上线测试，约 1.5 亿张图）：输入几个关键词，每个变成一个圆点，在版面框里拖到想要的位置，结果实时更新；另有一个按钮找有留白的图。技术上只说用了深度学习、已申请专利，没有公开细节。
- Adobe Stock：「Images with Copy Space」筛选只判断有没有留白；Find Similar 可以在一张示例图里选中一个物体，拖动、缩放后再搜，还能换关键词。同样没有公开技术细节。

## 方法：BBox-SigLIP

### 模型和查询格式

- 底座：SigLIP2 so400m-patch16-naflex，共约 11.4 亿参数：图像编码器 4.28 亿，文本编码器 4.13 亿，另有 2.95 亿的词表嵌入（25.6 万词 × 1,152 维）。不加任何新模块，两个编码器都微调。
- 查询：整个 JSON 字符串直接进文本编码器。写成紧凑格式，键的顺序固定为先 bbox 后 description，坐标取整，全部小写（SigLIP2 预训练时用的是小写文本）。超过 128 个 token 直接报错，不截断。
- 图像：NaFlex 按原图宽高比切 patch，最多 1,024 个（官方默认 256），训练和推理一致；每张图输出一个 1,152 维向量。两个编码器都微调过，所以图库向量要用 BBox-SigLIP 的图像编码器重新提一遍。

### 文本长度从 64 扩到 128

- 原生的文本上限是 64 个 token；为了放下完整描述加上框，扩到 128。
- 做法：前 20 个位置编码保留原值，其余 44 个线性插值拉伸成 108 个，只多出 7.4 万个参数。
- 新的位置表单独用 5e-6 的学习率，是其余参数（1e-6）的 5 倍。

### 训练数据

- 约 8,300 万张内部的通用图像，每个元素都有一个框和一段描述（来自图层检测加图像描述）。
- 每张图 4 个正样本查询：窗口有 1/3 的概率是紧框，其余是包含它的更大的窗口；文本是完整描述、第一句、关键词或类别名。

### 负样本与损失

- 同一张图、同一句描述、窗口不对：每张图 2 个，放在不碰任何相关框的位置。
- 不相关的图：和查询没有共同的词，也没有相关类别。
- 另一张图有同类东西，但不在窗口里：只对约 25 个常见类别才有。
- 一个 sigmoid 损失，权重 1 / 0.5 / 0.5 / 0.35；其余的对都忽略。
- 图 4：在一张训练图上画出每种正样本和负样本。

### 训练配置

- 2 个节点共 16 卡，每步 256 张图，学习率 1e-6（文本位置编码 5e-6），预热 1,000 步；FP32 参数、BF16 前向。
- 计划把 8,300 万张图完整过一遍（329,851 步），训练到 208,700 步中断，约完成计划的 63%。本文所有结果都用最后一个 checkpoint，第 208,448 步，这时看过约 5,340 万张图。

## 评测

### 怎么判断模型真的在看框

- 单看区域检索 hit@10，不看框也能拿高分。
- 每个测试固定一些东西，只改一样：
  - 区域检索：同一个查询，换不同的图（图库 949 张，3,725 个区域查询）；
  - 镜像框：同一张图、同一句描述，把框镜像过去（2,483 次）；
  - 镜像图：镜像的是图，不是框；
  - 拼图对调：两张图左右或上下拼在一起，查询问其中一半；50% 等于只看文字（14,900 次）。
- 一个坑：训练时记录的空间对指标在 2,500 步时就到了 86%，严重高估。以拼图对调测试为准。
- 图 5：四个测试的小示意图。

### 结果

- 第 208,448 步，和 2 万步对比用同一套 949 张图的评测：区域检索 hit@10 88.1，镜像框 83.3，镜像图 84.4，拼图对调 93.6 / 94.1（左右 / 上下）。官方 SigLIP2 的拼图对调是 50.2 / 50.1；2 万步的 JSON 基线是 73、69、70 和 74 / 78。
- 位置能力跟着步数走（拼图对调，左右 / 上下）：

  | 步数 | 2,500 | 5,000 | 10,000 | 124,745 | 208,448 |
  |---|---|---|---|---|---|
  | 拼图对调 | 53.7 / 54.3 | 57.6 / 63.4 | 63.2 / 70.8 | 91.9 / 93.3 | 93.6 / 94.1 |

- 我犯过的错：10 月 9 日，我对一个只训了 4,000 步的模型做了「看不看框」的诊断：对的框比错的框分数高的只占 43.5%，框只让查询向量变了约 2.4%。我据此下结论说一个向量学不会位置。长训练的模型推翻了这个结论，它只是需要更多训练。
- 图 6：上表画成曲线。图 7：真实查询的例子，需要换成公开图片。

### 公开基准：SPEC 绝对位置（CVPR 2024）

- 数据：合成图，每张图一个物体，放在 3×3 九个位置之一；500 组，每组 9 张图配 9 句话。
- 我们的用法：位置格子直接当框，描述只写物体名。
  - 文找图：一个查询，在同组 9 张图里选最高分的那张。
  - 图找文：固定一张图，给同一物体、9 个格子的框各打一个分，选最高的那个，也就是三元组里固定 $I$ 和 $t$、只改 $b$。
- 对比：原版 SigLIP2 用 SPEC 的原句；论文里的 CLIP 系列和 SPEC 作者微调的 CLIP。
- 另做一组：位置只写在句子里、框给整张图，看位置是从框里读的还是从字里读的。
- 结果表之后补。

## 分析（待定）

### 向量空间的变化：394 万张图的聚类对比

- 做法：一个 3,941,356 张图的图库，分别用原版 SigLIP2（patch14-384）和 BBox-SigLIP（第 208,448 步）提向量，各自做 K = 10,000 的球面 k-means（100 万张图训练、20 轮）。
- 两套分簇差别很大：ARI 0.125，NMI 0.285；在一个 2 万张图的固定池里，两个模型给同一张图找的前 20 个近邻平均只有 8% 重合（1,000 张查询图）。
- 簇和「图里有几个元素」的关联（调整互信息）从 0.035 升到 0.095，约 2.7 倍；和宽高比的关联从 0.065 降到 0.051。也就是说，新的向量空间更多按版式的繁简来组织图片。
- 要注明：两边的底座分辨率不同（固定 384 对 NaFlex），差别不全来自微调；分簇一致性也不等于检索质量。
- 图 8：两个模型的 t-SNE 簇图，加几组「旧簇 → 新簇」的拼图。

## Demo：在手机相册里用格子指定位置

- 把画面分成 3×3 格子，用户点选格子，比输入 0–1000 的坐标容易。选中格子的外接矩形就是框。
- 图库用 COCO val2017 的 5,000 张日常照片，只放 BBox-SigLIP（第 208,448 步）。
- 跟着框走的例子和没做好的例子各放几个。
- 图 9：demo 截图。

## 结论

- 四条结论，和引言里的四个结论一一对应。
- 局限，简单提：内部评测图库小（949 张）、只跑了一个种子、训练数据是内部数据；单个向量装不下一张图里所有小物体的信息。

## 附录：2 万步下的方案对比

### 六个方案

- 配方完全一样：批大小 128，8 卡，20,000 步，同样的数据和随机种子。
- 两种查询编码 × 三种训练目标：
  - 编码：JSON（正文的做法）或残差（文本和框分开编码，再用零初始化的残差融合）；
  - 目标：正文里的检索损失；裁图蒸馏，冻结的 SigLIP2 给窗口里的裁图打分，学生去拟合这些分数；成对蒸馏，在同一张图上，学生拟合老师对「对的窗口」和「挪开的窗口」两个分数的比较。
- 结果表：

  | 方案 | 拼图 左右 / 上下 | 镜像框 | Hit@10 |
  |---|---|---|---|
  | JSON + 检索损失 | 74 / 78 | 69 | 73 |
  | 残差 + 检索损失 | 59 / 66 | 65 | 78 |
  | JSON + 裁图蒸馏 | 54 / 59 | 52 | 82 |
  | 残差 + 裁图蒸馏 | 54 / 53 | 50 | 84 |
  | JSON + 成对蒸馏 | 60 / 77 | 76 | 15 |
  | 残差 + 成对蒸馏 | 52 / 57 | 59 | 5 |

### 为什么每种目标赢的测试不同

- 每个测试都是在固定某些东西的前提下比较两个分数。损失只塑造它比较过的那些分差。
- 这样的分差有三种：哪张图、哪个框、哪种摆法。
- 裁图蒸馏从来没见过「同一句话、同一张图、只挪框」，所以镜像框在整整 2 万步里一直停在 50%。
- 成对蒸馏比较的是同一张图上的两个窗口，内容那一项被抵消掉，检索随之崩掉：hit@10 在第 2,500 步就从 37 掉到 1。
- 检索损失三种分差都有，但位置信号又少又容易，所以学得慢。
- 图 A1：目标与三种分差的对应矩阵，加训练曲线。

### 怎样让位置学得更快

- 检索损失学位置为什么慢：
  - 每张图只有 2 个窗口不对的负样本，大多落在空白背景上；
  - 跨图的版式信号只对约 25 个类别才有；
  - 到 2 万步，窗口不对这一项是剩下最大的损失（0.317，不相关的图只有 0.064）。
- 多加同类数据不是关键：
  - 两次训练都没有重复用过一张图；
  - 每步图片数翻倍（256 对 128），同步数下只高 2–3 个点。
- 一个更清楚的损失（还没跑）：三道等权的选择题，哪张图、哪个框、哪种摆法。
- 图 A2：三问式损失和现在的损失并排对比。

## 引用

{{< bibtex >}}

## 参考文献

[^adobe_findsimilar]: **Adobe.** *Find assets similar to another asset in Adobe Stock*. Adobe Stock Help, https://helpx.adobe.com/stock/web/search-for-assets/find-similar-assets.html.

[^changpinyo_whatwhere]: **Changpinyo, Soravit, et al.** *Telling the What while Pointing to the Where: Multimodal Queries for Image Retrieval*. 2021. *ICCV*, https://arxiv.org/abs/2102.04980.

[^chen_cloc]: **Chen, Hong-You, et al.** *Contrastive Localized Language-Image Pre-Training*. 2025. *ICML*, https://arxiv.org/abs/2410.02746.

[^fu_objembed]: **Fu, Shenghao, et al.** *ObjEmbed: Towards Universal Multimodal Object Embeddings*. 2026. *ICML*, https://arxiv.org/abs/2602.01753.

[^istock_copyspace]: **CreativePro.** *iStockphoto Makes it Easy for Designers to Match Copy to Images with New CopySpace Feature*. 2005, https://creativepro.com/istockphoto-makes-it-easy-for-designers-to-match-copy-to-images-with-new-copyspace-feature/.

[^jiang_rexthinker]: **Jiang, Qing, et al.** *Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning*. 2026. *ICLR*, https://arxiv.org/abs/2506.04034.

[^kamath_mdetr]: **Kamath, Aishwarya, et al.** *MDETR – Modulated Detection for End-to-End Multi-Modal Understanding*. 2021. *ICCV*, https://arxiv.org/abs/2104.12763.

[^karpukhin_dpr]: **Karpukhin, Vladimir, et al.** *Dense Passage Retrieval for Open-Domain Question Answering*. 2020. *EMNLP*, https://arxiv.org/abs/2004.04906.

[^kilickaya_cal]: **Kilickaya, Mert, and Arnold W. M. Smeulders.** *Structured Visual Search via Composition-aware Learning*. 2021. *WACV*, https://openaccess.thecvf.com/content/WACV2021/html/Kilickaya_Structured_Visual_Search_via_Composition-Aware_Learning_WACV_2021_paper.html.

[^li_glip]: **Li, Liunian Harold, et al.** *Grounded Language-Image Pre-training*. 2022. *CVPR*, https://arxiv.org/abs/2112.03857.

[^lin_coco]: **Lin, Tsung-Yi, et al.** *Microsoft COCO: Common Objects in Context*. 2014. *ECCV*, https://arxiv.org/abs/1405.0312.

[^long_capdet]: **Long, Yanxin, et al.** *CapDet: Unifying Dense Captioning and Open-World Detection Pretraining*. 2023. *CVPR*, https://arxiv.org/abs/2303.02489.

[^mai_spatialsemantic]: **Mai, Long, et al.** *Spatial-Semantic Image Search by Visual Feature Synthesis*. 2017. *CVPR*, https://openaccess.thecvf.com/content_cvpr_2017/html/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.html.

[^petapixel_shutterstock]: **PetaPixel.** *Shutterstock's Composition Aware AI Photo Search Uses Deep Learning*. 2017, https://petapixel.com/2017/10/13/shutterstocks-composition-aware-ai-photo-search-uses-deep-learning/.

[^radford_clip]: **Radford, Alec, et al.** *Learning Transferable Visual Models From Natural Language Supervision*. 2021. *ICML*, https://arxiv.org/abs/2103.00020.

[^ranzinger_patent]: **Ranzinger, Mike, and Heath Hohwald.** *Composition Aware Image Querying*. 2019. *US Patent 10,503,775 B1*, https://patents.google.com/patent/US10503775B1/en.

[^wang_gar]: **Wang, Haochen, et al.** *Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs*. 2026. *ICLR*, https://arxiv.org/abs/2510.18876.

[^wu_ovsurvey]: **Wu, Jianzong, et al.** *Towards Open Vocabulary Learning: A Survey*. 2024. *TPAMI*, https://arxiv.org/abs/2306.15880.

[^xiao_vgsurvey]: **Xiao, Linhui, et al.** *Towards Visual Grounding: A Survey*. 2026. *TPAMI*, https://arxiv.org/abs/2412.20206.

[^xie_fgclip]: **Xie, Chunyu, et al.** *FG-CLIP: Fine-Grained Visual and Textual Alignment*. 2025. *ICML*, https://arxiv.org/abs/2505.05071.

[^xie_fgclip2]: **Xie, Chunyu, et al.** *FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model*. 2026. *ICML*, https://arxiv.org/abs/2510.10921.

[^xu_conceptmap]: **Xu, Hao, et al.** *Image Search by Concept Map*. 2010. *SIGIR*, https://www.microsoft.com/en-us/research/wp-content/uploads/2010/07/fp83-xu.pdf.

[^yu_refcoco]: **Yu, Licheng, et al.** *Modeling Context in Referring Expressions*. 2016. *ECCV*, https://arxiv.org/abs/1608.00272.

[^zhai_siglip]: **Zhai, Xiaohua, et al.** *Sigmoid Loss for Language Image Pre-Training*. 2023. *ICCV*, https://arxiv.org/abs/2303.15343.

[^zheng_uniref]: **Zheng, Duo, et al.** *Towards Unifying Reference Expression Generation and Comprehension*. 2022. *EMNLP*, https://arxiv.org/abs/2210.13076.

