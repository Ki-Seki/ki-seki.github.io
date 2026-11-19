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

- **可公开的资源（占位，待整理）：**
  - 模型：BBox-SigLIP 第 208,448 步
  - 实验记录：W&B [songsc-team/bbox-siglip](https://forge.coreweave.com/wandb/songsc-team/bbox-siglip)（9 个 run，待设为公开）
  - Demo：手机相册里用格子指定位置找图
  - 示例训练数据：100 对图像与 JSON，JSON 只保留训练时用到的字段
  - 评测集：949 张图、3,725 个区域查询，以及镜像、拼图对调的构造方式
  - 推理代码：JSON 查询怎么拼、怎么打分
  - 评测代码：内部四个测试，加上 SPEC 的「格子换成框」转换
  - 训练代码：采样和损失

TODO：引用尽量找权威的发表版本。

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
caption="图 2：同一个三元组（图 I、框 b、描述 t），已知其中两个、找第三个，就是三类任务。图中分数来自 BBox-SigLIP（第 208,448 步）。"
>}}

- 已知 $(b, t)$，找 $I \in \mathcal{I}$：按内容和位置找图，也就是本文的 what-and-where 图像检索；
- 已知 $(I, t)$，找 $b \in \mathcal{B}$：视觉 grounding 和开放词表检测，比如在 RefCOCO [^yu_refcoco] 上按指代表达找框；
- 已知 $(I, b)$，找 $t \in \mathcal{T}$：区域分类和区域描述，比如给定框判断类别，或者给定框生成描述。

三类任务都可以写成对未知量取 argmax。记未知量为 $u \in \{I, b, t\}$，它的候选集合为 $\mathcal{U}$（找图时是 $\mathcal{I}$，找框时是 $\mathcal{B}$，找描述时是 $\mathcal{T}$）：

$$u^\star = \arg\max_{u \in \mathcal{U}} s(I, b, t)$$

所以同一个 $s$ 能做全部三类任务，区别只在代价。如果能尽可能地离线编码未知量的候选，就能降低在线检索时的时延。能够实现这样检索方式的只能是类似 CLIP [^radford_clip]、SigLIP [^zhai_siglip]、DPR [^karpukhin_dpr]的双塔模型，即把查询和候选分开编码，候选可以提前离线计算。

显而易见，对于 what-and-where 图像检索任务，我们最好也用双塔模型，并且把 $(b, t)$ 放在查询一侧，$I$ 放在候选一侧，即：

$$s(I, b, t) := \langle f(b, t),\ g(I) \rangle$$

{{% admonition type="tip" title="证明：任务结构与模型结构匹配时时空复杂度最优" open=true %}}
双塔模型把三个量分成不重叠的两组 $X$ 和 $Y$，各进一个编码器，分数是两个向量的点积：

$$s(I, b, t) = \langle \phi(X),\ \psi(Y) \rangle$$

记 $u$ 所在的一组为 $Y$。若 $Y = \{u\}$，所有候选的 $\psi(u)$ 都能提前算好，查询时只需把已知量编码一次，再做 $|\mathcal{U}|$ 次点积；否则 $u$ 要和已知量一起进编码器，每个候选都得重新编码一次：

$$\mathrm{cost}(u) = \begin{cases} C_{\mathrm{enc}} + |\mathcal{U}| \, C_{\mathrm{dot}}, & Y = \{u\} \\ |\mathcal{U}| \, C_{\mathrm{enc}}, & Y \neq \{u\} \end{cases}$$

其中 $C_{\mathrm{enc}}$ 是一次编码，$C_{\mathrm{dot}}$ 是一次点积，并且 $C_{\mathrm{enc}} \gg C_{\mathrm{dot}}$。

以 what-and-where 任务为例，$|\mathcal{I}|$ 可以有几百万甚至数十亿，只有 $Y = \{I\}$ 付得起。$Y = \{I, b\}$ 时也可以给每张图预存 $K$ 个候选框的 $\psi(I, b)$，但存储变成 $K$ 倍，查询的框也只能就近对上某个候选框。
{{% /admonition %}}

## 相关工作

根据任务和模型结构我们可以非常清晰的分出来12 种方法。下表的列是三类任务；行是模型结构，也就是三个量怎样分组编码，竖线右边是单独成组的量：$(b, t) \mid I$ 是框和描述一起编码、图单独编码，以此类推；$(I, b, t)$ 是单塔，三个量在同一个网络里一起算，比如多模态大模型。并且我们知道只有对角线上的方法是在对应任务上高效的。

实际上第二三列的任务，detection，classfication，caption 都是研究很充分的领域，完整的梳理也有相关综述 [^xiao_vgsurvey] [^wu_ovsurvey]。而第一种目前还没有充分的研究。而据我们所知，第一列的 what-and-where 任务尚只有这列出的 6 份工作。

| | 已知 $(b, t)$ 找 $I$ | 已知 $(I, t)$ 找 $b$ | 已知 $(I, b)$ 找 $t$ |
|---|---|---|---|
| ${(b, t) \mid I}$ | <small>Spatial&#8209;Semantic&nbsp;Search&nbsp;<small>CVPR'17</small>&nbsp;[^mai_spatialsemantic]<br>FCN&#8209;based&nbsp;Retrieval&nbsp;<small>MMM'18</small>&nbsp;[^furuta_segretrieval]<br>Text+Trace&nbsp;Model&nbsp;<small>ICCV'21</small>&nbsp;[^changpinyo_whatwhere]<br>**BBox&#8209;SigLIP**&nbsp;<small>本文</small></small> | <small>暂无</small> | <small>暂无</small> |
| ${(I, t) \mid b}$ | <small>不可行</small> | <small>MDETR&nbsp;<small>ICCV'21</small>&nbsp;[^kamath_mdetr]<br>GLIP&nbsp;<small>CVPR'22</small>&nbsp;[^li_glip]<br>Grounding&nbsp;DINO&nbsp;<small>ECCV'24</small>&nbsp;[^liu_groundingdino]</small> | <small>暂无</small> |
| ${(I, b) \mid t}$ | <small>Concept&nbsp;Map&nbsp;<small>SIGIR'10</small>&nbsp;[^xu_conceptmap]<br>Shutterstock&nbsp;构图检索&nbsp;<small>US&nbsp;Patent'19</small>&nbsp;[^ranzinger_patent]</small> | <small>OWL&#8209;ViT&nbsp;<small>ECCV'22</small>&nbsp;[^minderer_owlvit]<br>RegionCLIP&nbsp;<small>CVPR'22</small>&nbsp;[^zhong_regionclip]<br>ObjEmbed&nbsp;<small>ICML'26</small>&nbsp;[^fu_objembed]</small> | <small>Alpha&#8209;CLIP&nbsp;<small>CVPR'24</small>&nbsp;[^sun_alphaclip]<br>CLOC&nbsp;<small>ICML'25</small>&nbsp;[^chen_cloc]<br>FG&#8209;CLIP&nbsp;2&nbsp;<small>ICML'26</small>&nbsp;[^xie_fgclip2]</small> |
| ${(I, b, t)}$ | <small>不可行</small> | <small>Kosmos&#8209;2&nbsp;<small>ICLR'24</small>&nbsp;[^peng_kosmos2]<br>Rex&#8209;Thinker&nbsp;<small>ICLR'26</small>&nbsp;[^jiang_rexthinker]</small> | <small>UniRef&nbsp;<small>EMNLP'22</small>&nbsp;[^zheng_uniref]<br>Ferret&nbsp;<small>ICLR'24</small>&nbsp;[^you_ferret]<br>GAR&nbsp;<small>ICLR'26</small>&nbsp;[^wang_gar]</small> |

第一列中以找图为目标的六个方法，差别在查询长什么样、框进了哪一侧、每张图要存什么（图 3）。和我们结构相同、都是 $(b, t) \mid I$ 的有三项。最接近的是 **Text+Trace Model** [^changpinyo_whatwhere]，「what+where 检索」这个叫法就出自这篇，图库一侧也和我们一样，每张图只存一个向量。但它的查询不是为找图设计的，而是直接搬用 Localized Narratives 的标注方式：标注员一边口述整张图，一边移动鼠标，每段话说出时鼠标扫过的范围取成一个框。位置因此只是说话时鼠标恰好停在哪里，不是用户有意圈出的范围：语音和鼠标对不齐，论文要在时间和空间上各放宽一圈才能用，"we can see" 这类虚词也带着轨迹。我们统计过它的评测数据，每条口述平均被切成 36 到 50 段，每段平均只有 1.1 个词，11% 到 15% 的段落里没有任何轨迹点。用户想找一只在右下角的狗，也得先把整张图用语音讲一遍，甚至不能通过文本搜索。另一项是 **Spatial-Semantic Search** [^mai_spatialsemantic]：它把画布上的词向量合成一张 7×7×832 的查询特征图，和图库里固定的 GoogLeNet 特征图整张比。结构对得上，但每张图存的是摊平后约 4 万维的空间特征图，是我们的 35 倍，查询也只能是单个词。**FCN-based Retrieval** [^furuta_segretrieval] 和它类似：用户在画布上直接画一张语义分割图，每块区域涂一个类别；每张图离线存一张 DeepLab 输出的 60 类、64×64 的概率图，摊平约 24.6 万维，再用乘积量化压缩。它的区域可以是任意形状，但查询同样只能用这 60 个固定类别。

{{< media
src="assets/where-the-box-goes.svg"
caption="图 3：找图的六个方法。左边是查询和查询编码器的输出，右边是每张图存的东西。上面四行框和描述一起进查询编码器（表中 (b, t) | I），下面两行框只在打分时作用在图的格子上（表中 (I, b) | t）。"
>}}

**Concept Map** [^xu_conceptmap] 和 **Shutterstock 构图检索** [^ranzinger_patent] 要找的是图，模型结构却是 $(I, b) \mid t$，任务和结构从一开始就对不上，注定低效：每张图要存一整张网格、每格一份特征；每来一个查询，都要拿关键词和每张图的每个格子比一遍，计算和存储都是格子数的倍数，框也只能对齐到格子边界。两者做法相同：离线把每张图切成网格，每格存一份特征；关键词单独编码；查询时先算关键词和每个格子的匹配度，得到一张热力图，再按用户给的位置读出分数。不同的只是各模块的实现：格子特征分别是视觉词袋直方图、CNN 向量，关键词分别表示成网页搜图挖出的视觉实例、神经语言模型的词向量，位置分别来自可拉伸的矩形、画布上的点。

以上几项都没有公开代码或者模型。评测方面，只有 Text+Trace Model 用的 Localized Narratives 是公开数据，但是由于他的方法设定过于奇怪，我们不打算在本文中采用。

## 方法：BBox-SigLIP

BBox-SigLIP 从 Google 的 SigLIP2 [^tschannen_siglip2] 初始化，不加任何新模块，两个编码器一起微调。实现有效的 BBo-SigLIP主要有赖于三处设计：训练数据哪里来，$(b, t)$在训练和推理中怎么编码，损失函数怎么设计，以及文本长度从 64 扩到 128 的方法。一些模型细节见附录 A（TODO：回头加上 anchor）。

### 合成训练数据

训练的基础数据来自品类多样的约 8,300 万张通用图像。我们需要图像的 region-wise 的 caption 信息，所以需要数据标注。内部实测发现 Gemini 模型在位置标注准确性较高。但鉴于成本过高，所以我们用了 Gemini 蒸馏出来的 4B 模型：用 Gemini 的标注把 Qwen3.5-4B 微调成看图写结构化描述的模型，再用它逐张标注训练图。每张图标出若干元素，每个元素有一个类别名、一个框和一段描述（prompt 和样例见附录）。描述里如果出现 left、right、top、center 这类方位词，这个元素就不用：位置只能从框里读，不能从字里读。

每张图随机取 4 个元素，各合成一条查询 $(b, t)$。框有 1/3 的概率直接用紧框 $b^\star$，其余时候随机放大 5%–90%，成为一个包住它的窗口：按问题定义，只要 $b^\star \subseteq b$ 就算匹配，用户画的框本来也很少恰好贴着物体。放大后的窗口如果超过全图的 85%，就退回紧框，免得查询退化成「图里有没有这个东西」。描述有四种写法：完整描述、第一句、2–3 个关键词、类别名，比例是 25 / 30 / 35 / 10%，让模型既读得了长描述，也接得住只打几个词的查询。

4 条查询里有 2 条各配一个「窗口不对」的负样本：描述不变，同样大小的窗口挪到不碰任何相关框的位置，即 $b \cap b^\star = \varnothing$。相关框包括这个元素自己、和它有共同概念词的元素，以及同类的东西，所以挪开后的窗口里确实没有查询说的东西。图 4 在一张训练图上画出了这几种窗口。

### 查询编码与损失函数

查询编码器 $f$ 的输入就是一段 JSON：框和描述原样写成文本，送进 SigLIP2 的文本编码器，不另设坐标编码器。图 1 的查询实际是：

```json
{"bbox":[500,612,765,898],"description":"a dark brown dog with short shiny fur bounds through the shallow water at the edge of a lake, front legs reaching forward and tail held up."}
```

SigLIP2 的分词器把数字逐位拆开，所以模型是一位一位地读坐标。这要求训练和推理拼出的字符串格式完全一致，具体规则见附录。

训练时，一批 256 张图里的每条查询都和每张图配成一对。我们按标注把这些对分成四类：

- **正样本**：查询和它来源的那张图；
- **窗口不对**：同一张图，配上挪开的窗口；
- **跨图窗口不对**：另一张图里有同类的东西，但全在窗口外。这一类只对人、狗、车、椅子等 30 个常见类别才能判断；
- **不相关**：另一张图和查询没有共同的概念词，也没有同类的东西。

其余的对不进损失。它们是标注没法判断的情况：另一张图里可能也有查询说的东西，或者东西和窗口只部分重叠，硬标成负样本会制造假负例。这和问题定义一一对应：包含算匹配，不相交算不匹配，部分重叠不监督。

损失沿用 SigLIP [^zhai_siglip] 的 sigmoid 损失。记 $z = \alpha \langle f(b, t), g(I) \rangle + \beta$，其中 $\alpha$、$\beta$ 是可学习的温度和偏置；正样本的损失是 $\log(1 + e^{-z})$，三类负样本是 $\log(1 + e^{z})$。四类各自在全批上取平均 $\bar\ell$，再加权求和：

$$\mathcal{L} = \bar\ell_{\mathrm{pos}} + 0.5\,\bar\ell_{\mathrm{unrel}} + 0.5\,\bar\ell_{\mathrm{own}} + 0.35\,\bar\ell_{\mathrm{cross}}$$

下标依次是正样本、不相关、窗口不对、跨图窗口不对。各自取平均，是为了不让数量决定权重：一批里不相关的对有几十万个，窗口不对只有几百个。

### 线性插值扩展上下文

SigLIP2 的文本编码器最多读 64 个 token。一条查询光框这一段就占约 21 个，而我们评测集里有 9% 的描述，不加框就已经超过 64 个 token。我们把上限扩到 128，做法沿用 Long-CLIP [^zhang_longclip]：前 20 个位置编码保留原值，其余 44 个线性插值拉伸成 108 个。Long-CLIP 的依据是，预训练文本大多很短，靠前的位置编码训练得最充分，应当原样保留，靠后的只需拉伸。恰好任何合法的框，坐标数字都落在这前 20 个位置上，模型读坐标用的仍是原来的位置编码。新的位置表只多出 7.4 万个参数，单独用 5e-6 的学习率，是其余参数的 5 倍。

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
- 图库用 COCO val2017 [^lin_coco] 的 5,000 张日常照片，只放 BBox-SigLIP（第 208,448 步）。
- 跟着框走的例子和没做好的例子各放几个。
- 图 9：demo 截图。

## 结论

- 四条结论，和引言里的四个结论一一对应。
- 局限，简单提：内部评测图库小（949 张）、只跑了一个种子、训练数据是内部数据；单个向量装不下一张图里所有小物体的信息。

## 附录 A：实现细节

### 训练数据的标注

用 Gemini 的标注微调过的 Qwen3.5-4B，贪心解码，不开思考模式。prompt 如下：

```text
[system]
You produce a complete structured design description from an image. Return only valid JSON in the requested full schema.

[user]
<image>
Generate the complete JSON containing elements. Every item must contain label, bbox in xyxy_1000 coordinates, and desc.
```

下面是一张训练图（来自 Pixabay，图 A1）的标注：

```json
[
  {"label":"<woman 1>","bbox":[0,0,584,1000],"desc":"A woman shown from the chest up on the left side of the frame, bathed in slatted sunlight and shadow."},
  {"label":"<face 1>","bbox":[172,0,517,483],"desc":"The face of the woman, tilted upward with closed eyes, covered in horizontal shadow lines from window blinds."},
  {"label":"<lips 1>","bbox":[377,249,488,368],"desc":"The woman's lips, coated in deep red lipstick and slightly parted."},
  {"label":"<hair 1>","bbox":[0,0,273,990],"desc":"Dark, straight hair falling along the left side of the woman's face and shoulder."},
  {"label":"<rose 1>","bbox":[447,345,755,1000],"desc":"A single red rose held in the lower right foreground, featuring a full bloom and green stem."},
  {"label":"<flower head 1>","bbox":[447,345,735,747],"desc":"The deep red bloom of the rose, with dense, velvety petals catching soft light."},
  {"label":"<stem 1>","bbox":[481,627,755,1000],"desc":"The green stem and leaves of the rose extending downward from the flower head."}
]
```

其中 `<woman 1>`、`<hair 1>`、`<rose 1>` 的描述带方位词（left、lower right），我们会用规则尽量去除，不用来合成查询，但仍留在这张图的元素列表里，判断哪些框、哪些图和查询相关时还会用到。其余 4 个元素各能合成一条查询。

{{< media
src="assets/training-annotation.svg"
caption="图 A1：上面那条标注画在原图上。实线的 4 个元素描述里没有方位词，各合成一条查询；虚线的 3 个元素描述里带方位词，不合成查询。图片来自 Pixabay。"
>}}

### 模型参数

| 项目 | 设置 |
|---|---|
| 底座 | SigLIP2 so400m-patch16-naflex |
| 总参数量 | 约 11.4 亿 |
| 图像编码器 | 4.28 亿参数 |
| 文本编码器 | 4.13 亿参数 |
| 词表嵌入 | 2.95 亿参数（25.6 万个词 × 1,152 维） |
| 图像切分 | NaFlex，按原图宽高比切 16×16 的 patch，最多 1,024 个（官方默认 256），训练和推理一致 |
| 文本长度 | 最多 128 个 token（原 64） |
| 输出向量 | 图和查询各一个 1,152 维向量 |
| 微调范围 | 两个编码器都微调 |

两个编码器都改过，所以图库要用 BBox-SigLIP 重新提向量。

### 训练配置

| 项目 | 设置 |
|---|---|
| 硬件 | 2 个节点共 16 卡 |
| 每步图片数 | 256 |
| 优化器 | AdamW，β = (0.9, 0.98)，ε = 1e-8 |
| 学习率 | 1e-6；文本位置表 5e-6 |
| 学习率调度 | 预热 1,000 步，之后余弦衰减，最低到峰值的 10% |
| weight decay | 0.01 |
| 梯度裁剪 | 1.0 |
| 精度 | FP32 参数，BF16 前向 |
| 计划步数 | 329,851 步，8,300 万张图完整过一遍 |
| 实际步数 | 第 208,800 步中断，约完成 63% |
| 本文结果用的 checkpoint | 第 208,448 步，看过约 5,340 万张图，学习率约为峰值的 37% |

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
- 图 A2：目标与三种分差的对应矩阵，加训练曲线。

### 怎样让位置学得更快

- 检索损失学位置为什么慢：
  - 每张图只有 2 个窗口不对的负样本，大多落在空白背景上；
  - 跨图的版式信号只对约 25 个类别才有；
  - 到 2 万步，窗口不对这一项是剩下最大的损失（0.317，不相关的图只有 0.064）。
- 多加同类数据不是关键：
  - 两次训练都没有重复用过一张图；
  - 每步图片数翻倍（256 对 128），同步数下只高 2–3 个点。
- 一个更清楚的损失（还没跑）：三道等权的选择题，哪张图、哪个框、哪种摆法。
- 图 A3：三问式损失和现在的损失并排对比。

## 引用

{{< bibtex >}}

## 参考文献

[^adobe_findsimilar]: **Adobe.** *Find assets similar to another asset in Adobe Stock*. Adobe Stock Help, https://helpx.adobe.com/stock/web/search-for-assets/find-similar-assets.html.

[^changpinyo_whatwhere]: **Changpinyo, Soravit, et al.** *Telling the What while Pointing to the Where: Multimodal Queries for Image Retrieval*. 2021. *ICCV*, https://arxiv.org/abs/2102.04980.

[^chen_cloc]: **Chen, Hong-You, et al.** *Contrastive Localized Language-Image Pre-Training*. 2025. *ICML*, https://arxiv.org/abs/2410.02746.

[^fu_objembed]: **Fu, Shenghao, et al.** *ObjEmbed: Towards Universal Multimodal Object Embeddings*. 2026. *ICML*, https://arxiv.org/abs/2602.01753.

[^furuta_segretrieval]: **Furuta, Ryosuke, Naoto Inoue, and Toshihiko Yamasaki.** *Efficient and Interactive Spatial-Semantic Image Retrieval*. 2018. *MMM*, https://doi.org/10.1007/978-3-319-73603-7_16.

[^istock_copyspace]: **CreativePro.** *iStockphoto Makes it Easy for Designers to Match Copy to Images with New CopySpace Feature*. 2005, https://creativepro.com/istockphoto-makes-it-easy-for-designers-to-match-copy-to-images-with-new-copyspace-feature/.

[^jiang_rexthinker]: **Jiang, Qing, et al.** *Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning*. 2026. *ICLR*, https://arxiv.org/abs/2506.04034.

[^kamath_mdetr]: **Kamath, Aishwarya, et al.** *MDETR – Modulated Detection for End-to-End Multi-Modal Understanding*. 2021. *ICCV*, https://arxiv.org/abs/2104.12763.

[^karpukhin_dpr]: **Karpukhin, Vladimir, et al.** *Dense Passage Retrieval for Open-Domain Question Answering*. 2020. *EMNLP*, https://arxiv.org/abs/2004.04906.

[^kilickaya_cal]: **Kilickaya, Mert, and Arnold W. M. Smeulders.** *Structured Visual Search via Composition-aware Learning*. 2021. *WACV*, https://openaccess.thecvf.com/content/WACV2021/html/Kilickaya_Structured_Visual_Search_via_Composition-Aware_Learning_WACV_2021_paper.html.

[^li_glip]: **Li, Liunian Harold, et al.** *Grounded Language-Image Pre-training*. 2022. *CVPR*, https://arxiv.org/abs/2112.03857.

[^lin_coco]: **Lin, Tsung-Yi, et al.** *Microsoft COCO: Common Objects in Context*. 2014. *ECCV*, https://arxiv.org/abs/1405.0312.

[^liu_groundingdino]: **Liu, Shilong, et al.** *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*. 2024. *ECCV*, https://arxiv.org/abs/2303.05499.

[^mai_spatialsemantic]: **Mai, Long, et al.** *Spatial-Semantic Image Search by Visual Feature Synthesis*. 2017. *CVPR*, https://openaccess.thecvf.com/content_cvpr_2017/html/Mai_Spatial-Semantic_Image_Search_CVPR_2017_paper.html.

[^minderer_owlvit]: **Minderer, Matthias, et al.** *Simple Open-Vocabulary Object Detection with Vision Transformers*. 2022. *ECCV*, https://arxiv.org/abs/2205.06230.

[^peng_kosmos2]: **Peng, Zhiliang, et al.** *Kosmos-2: Grounding Multimodal Large Language Models to the World*. 2024. *ICLR*, https://arxiv.org/abs/2306.14824.

[^petapixel_shutterstock]: **PetaPixel.** *Shutterstock's Composition Aware AI Photo Search Uses Deep Learning*. 2017, https://petapixel.com/2017/10/13/shutterstocks-composition-aware-ai-photo-search-uses-deep-learning/.

[^radford_clip]: **Radford, Alec, et al.** *Learning Transferable Visual Models From Natural Language Supervision*. 2021. *ICML*, https://arxiv.org/abs/2103.00020.

[^ranzinger_patent]: **Ranzinger, Mike, and Heath Hohwald.** *Composition Aware Image Querying*. 2019. *US Patent 10,503,775 B1*, https://patents.google.com/patent/US10503775B1/en.

[^sun_alphaclip]: **Sun, Zeyi, et al.** *Alpha-CLIP: A CLIP Model Focusing on Wherever You Want*. 2024. *CVPR*, https://arxiv.org/abs/2312.03818.

[^tschannen_siglip2]: **Tschannen, Michael, et al.** *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. 2025. *arXiv*, https://arxiv.org/abs/2502.14786.

[^wang_gar]: **Wang, Haochen, et al.** *Grasp Any Region: Towards Precise, Contextual Pixel Understanding for Multimodal LLMs*. 2026. *ICLR*, https://arxiv.org/abs/2510.18876.

[^wu_ovsurvey]: **Wu, Jianzong, et al.** *Towards Open Vocabulary Learning: A Survey*. 2024. *TPAMI*, https://arxiv.org/abs/2306.15880.

[^xiao_vgsurvey]: **Xiao, Linhui, et al.** *Towards Visual Grounding: A Survey*. 2026. *TPAMI*, https://arxiv.org/abs/2412.20206.

[^xie_fgclip]: **Xie, Chunyu, et al.** *FG-CLIP: Fine-Grained Visual and Textual Alignment*. 2025. *ICML*, https://arxiv.org/abs/2505.05071.

[^xie_fgclip2]: **Xie, Chunyu, et al.** *FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Model*. 2026. *ICML*, https://arxiv.org/abs/2510.10921.

[^xu_conceptmap]: **Xu, Hao, et al.** *Image Search by Concept Map*. 2010. *SIGIR*, https://www.microsoft.com/en-us/research/wp-content/uploads/2010/07/fp83-xu.pdf.

[^you_ferret]: **You, Haoxuan, et al.** *Ferret: Refer and Ground Anything Anywhere at Any Granularity*. 2024. *ICLR*, https://arxiv.org/abs/2310.07704.

[^yu_refcoco]: **Yu, Licheng, et al.** *Modeling Context in Referring Expressions*. 2016. *ECCV*, https://arxiv.org/abs/1608.00272.

[^zhai_siglip]: **Zhai, Xiaohua, et al.** *Sigmoid Loss for Language Image Pre-Training*. 2023. *ICCV*, https://arxiv.org/abs/2303.15343.

[^zhang_longclip]: **Zhang, Beichen, et al.** *Long-CLIP: Unlocking the Long-Text Capability of CLIP*. 2024. *ECCV*, https://arxiv.org/abs/2403.15378.

[^zheng_uniref]: **Zheng, Duo, et al.** *Towards Unifying Reference Expression Generation and Comprehension*. 2022. *EMNLP*, https://arxiv.org/abs/2210.13076.

[^zhong_regionclip]: **Zhong, Yiwu, et al.** *RegionCLIP: Region-based Language-Image Pretraining*. 2022. *CVPR*, https://arxiv.org/abs/2112.09106.
