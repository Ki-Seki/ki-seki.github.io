---
date: '2026-10-26T21:31:14+08:00'
title: 'CaptionExam：一个高难度的 captioner QA benchmark'
author:
  - Elliot Shichao Song
summary: '在 CaptionExam benchmark 上，目前 GPT 系列最强模型 GPT-6 Astra 的图像描述能力也才刚刚及格，得分 61.2%。'
tags:
  - image-captioning
  - benchmark
  - evaluation
  - text-to-image
  - vision-language-model
math: false
---

- **代码：** <https://github.com/Ki-Seki/CaptionExam>
- **数据：** [Ki-Seki/CaptionExam](https://huggingface.co/datasets/Ki-Seki/CaptionExam)（109 张图、3,099 道题）
- **评测结果：** [Ki-Seki/CaptionExam-Results](https://huggingface.co/datasets/Ki-Seki/CaptionExam-Results)（35 个 captioner 的 caption、判定和排行榜）

## 引言

越来越多的图像生成模型选择用详细丰富的 caption 来训练，比如说最近的开源模型 ideogram4， MOSS-Image 都在训练中加入了非常长的 cpation 的图文对。而且如今的这些 renderer s 已经能还原 caption 中的大部分信息。

renderer 还原得越好，captioner 就越决定上限：它漏掉的每个细节，生成模型都学不会控制；它编造的每个细节，都在教生成模型错误的东西。因此，一个好的 captioner，也就是负责标注训练数据的模型，和生成模型本身同样重要。https://heheyas.github.io/context-scaling/ 字节的 Context Scaling 更是在明确探索 caption 的 scaling law。

为了选出强大的 captioner，目前有两个重要因素要做好。一个从结构出发，大多都设计为 region-wise 的 json 结构，这非常方便扩展标注细致程度。但是这里面的信息量高低还是缺少需要一个足够难、能拉开强模型差距，又足够具体、能指出每个模型错在哪里的 benchmark。所以我们设计了CaptionExam ，一个非常具有挑战性的 captioner QA benchmark。

{{< media
src="assets/how-it-works.svg"
caption="图 1：被测 captioner 为图像写 caption；一个只读文字、看不到图的判官，仅凭 caption 回答关于这张图的选择题。"
>}}

benchmark 包括 109 张图，每张图都配有一组关于其细节的选择题。待评测的 captioner 写出 caption 后，判官模型读 caption，但看不到图，仅凭 caption 回答每道题。

显而易见，captioner 的描述越细致准确，判官就越容易根据 caption 回答正确，从而获得更高的评分。

## 相关工作

目前最接近的是 **CapProbe**[^huang_capprobe]，而且也是同时期的工作。它同样让一个只读文字的判官仅凭 caption 回答选择题，并用一个 *Uncertain* 选项区分 caption 漏掉的和说错的内容。它的题目按区域覆盖整个画面生成：346 张图、25,650 道题、10 个语义类别，用有效准确率和基于信息密度的指标打分。

CaptionExam 的不同在于题目的来源：我们的题是从真实 captioning 模型的失误中挖出来的，而不是为了覆盖画面而生成的，因此题量更小（109 张图、3,099 道题），对强 captioner 来说却很难。另外，CapProbe 的数据和代码尚未公开；CaptionExam 的代码和数据都已开源，判官在本地一张 GPU 上就能运行，可能更方便直接使用。

## CaptionExam Benchmark

### 数据构造方式

题目是从真实的proprietary captioning 模型（抱歉，这不能开源）的失误中挖出来的：

1. Gemini 3.1 Pro（preview）同时看一张图和某个模型为它写的 caption，找出这份 caption 最严重的事实性失误（说错的内容，以及漏掉的显著内容），并针对每个失误写一道选择题。
2. Gemini 3.1 Pro 和 GPT-5.6 Sol 分别在看图的情况下回答每道候选题。只有两者都答对的题才会保留，这样可以去掉有歧义的题目和答案有误的题目。
3. 每道保留下来的题在进入 benchmark 之前都经过人工审核，审核时可以修改题面和区域。

### 数据统计

| | |
|---|---|
| 图像 | 109 张：照片、插画、商品图、海报、书籍封面、文字密集的图形 |
| 题目 | 3,099 道，每张图 15–42 道（平均 28 道） |
| 选项 | 每题 3–6 个（多数为 5 个），其中一个是「未提及」 |
| 区域 | 120 道题针对整张图，其余题目各自指向一个区域，区域通常只占全图的 4% 左右 |
| 随机猜测 | 在「未提及」以外的选项中均匀随机选择，准确率为 26.6% |

### 题目问什么

题目可以涉及图中任何可见的方面：有哪些物体、有几个，姿态与朝向，空间关系与版式，颜色及其具体色调，材质与表面质感，光照，图中文字及其确切内容，重复出现的文字，字体、字重与大小写，图形风格与绘制手法，背景，以及被画面边缘截断的内容。

{{< media
src="assets/question-types.svg"
caption="图 2：3,099 道题按考察内容的分布。GPT-5.6 Luna 根据题面、选项和答案把每道题归入十类之一。"
>}}

近三成的题问的是有什么物体、是什么场景；颜色、位置与版式各占约 15%；问文字内容和字体的题合计约 13%。图 3 是一张示例图和它的四道题。

{{< media
src="assets/example-questions.webp"
caption="图 3：一张示例图和其中的四道题。"
>}}

### 判官模型打分

用最强的闭源模型当判官或许更准，但每次评测都要花钱、占用 API 配额，还依赖一个可能随时变化的模型。我们选择了 Phi-4：一个 14B 的模型。

在众多小规模模型中，Phi-4 和前沿模型的校准程度是非常高的。让 GPT-5.6 Sol 分别判 Qwen2.5-VL-72B、Qwen3-VL-235B-A22B 和 Qwen3.5-397B-A17B 的 caption。两个判官在 76.9% 的判定上一致（Cohen's κ = 0.61）。准确率几乎相同：Qwen2.5-VL-72B 都是 23.2%，Qwen3-VL-235B-A22B 是 41.5% 对 41.6%；Qwen3.5-397B-A17B 差了 3 个点（41.4% 对 44.5%），它和 Qwen3-VL-235B-A22B 的先后也因此换了位置，不过两者在全部题目上本来只差 0.5 个点。

## 一些评测结果

我们评价了 23 个 Qwen 视觉语言模型 和 12 个 OpenAI 模型。

- Qwen：Qwen2-VL（2B、7B、72B）、Qwen2.5-VL（3B、7B、32B、72B）、Qwen3-VL（2B、4B、8B、32B、30B-A3B、235B-A22B）、Qwen3.5（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B）和 Qwen3.6（27B、35B-A3B）
- OpenAI：GPT-5（nano，mini，base）、GPT-5.4（nano，mini，base）、GPT-5.6（Luna，Terra，Sol）、GPT-6（Luna，Sol，Astra）

所有模型都尽可能关闭思考模式，保持 greedy 解码，使用相同 prompt，要求尽可能详尽地描述图中一切可见内容。

### 准确率与长度

显而易见，说的越多，更容易面面俱到，但也可能带来冗余和不必要的 token，下面是各模型在 token 数与准确率上的表现。目前，token 数与准确率表现俱佳的大多是 OpenAI 的模型。

{{< media
src="assets/pareto.html"
caption="图 5：每个 captioner 为每张图平均输出的 token 数与准确率。左上绿色区域里的模型比中位模型 token 更少、准确率更高。"
>}}

一个很有意思的现象是，同一个系列的模型（比如 Qwen3.5 系列）他们的表现并不随模型规模的增加而提升。

### 冗余发生的地方

token 数相近的模型，准确率可以差得很远：GPT-5.6 Sol 和 Qwen2.5-VL-32B 都写 630 个 token 左右，准确率差了 27 个点。多写的 token 大致花在三个地方：

- **重复。** 小模型会陷入循环，把同一段话反复写到 4,096 个 token 的上限。Qwen3-VL-2B 的 109 份 caption 里有 9 份如此，Qwen3.5-0.8B 有 4 份，Qwen3-VL-4B 有 3 份。
- **格式和客套。** GPT-5.4 nano 平均每份 caption 有 6 个小标题、38 个列表项；20% 的 caption 带一段总结，把前文再说一遍；24% 在末尾主动提出 “If you want, I can also…”。这些 token 不带来任何新事实。
- **写得多，却不具体。** 长 caption 常把篇幅花在笼统的风格、质感和不确定性说明上，而题目问的是具体的东西：在哪里、什么颜色、被画面哪条边截断。

举个例子，图 6 是同一张木瓜插画的两份 caption。GPT-5.4 nano 写了 678 个词（980 个 token），38 道题只答对 5 道；GPT-6 Sol 只写了 201 个词（265 个 token），答对 16 道。

{{< media
src="assets/papaya-captions.webp"
caption="图 6：GPT-5.4 nano 和 GPT-6 Sol 为同一张木瓜插画写的完整 caption，按模型输出原样排版（## 和 ### 是 nano 自己写的 Markdown 标题）。右上的横条是这张图 38 道题的判定：绿色答对，灰色漏述，红色幻觉。"
>}}

nano 的 caption 一直在说「很多切开的木瓜」，却没有提到画面里还有完整的木瓜，也没有交代每个木瓜在哪里、哪些被边缘截断，于是这些题判官只能选「未提及」；问到画里的木瓜是什么形态时，判官据它选了「只有切半的和切块的」，算作幻觉。Sol 的 caption 只有它的三分之一长，却逐个交代了切开的和完整的木瓜各在哪里、哪些被截断。

### 从 caption 回到图像

Qwen 的三个 2B 尺寸模型Qwen2-VL-2B（准确率 11.4%）、Qwen3-VL-2B（准确率 26.0%）、Qwen3.5-2B（准确率 32.9%）表现恰好从低分到高分的趋势。我们可以把他们的 caption 拿给 MOSS-Image[^mosi_moss] 生成图像，看看是否生成的图像也更能还原原图，另放最强的 captioner GPT-6 Astra 作参照（图 7）。

非常直观，caption 在 CaptionExam 上得分越高，生成的图越接近原图：低分 caption 只够还原「一套石榴石首饰」「一间酒店客房」这样的大意，物体怎么摆、画面里写了什么字，都由生成模型自己补；高分 caption 把这些写了出来，构图也就跟着对了。脚部解剖图最明显：Qwen2-VL-2B 没有写出图中的标注文字，MOSS-Image 便自己编了一套中文标注。

{{< media
src="assets/caption-to-image.webp"
caption="图 7：每一行是一张图。左起第一列是原图，其余各列是 MOSS-Image 根据各个 captioner 的 caption 生成的图（seed 42，50 步，cfg 3.0），按 captioner 的总体准确率从低到高排列，最右一列 GPT-6 Astra 作参照。每张生成图下方是这份 caption 在该图题目上的准确率。"
>}}

这说明两件事。训练时，**captioner 漏掉或写错的细节，生成模型都学不到正确的对应**；使用时，用户随手写的 prompt 往往比低分 caption 还要简略，**扩写prompt为具体、准确的描述，同样决定了生成结果离想要的画面有多近**。

此外，题目从不提及 caption，因此也可以直接用来评估生成的图像。我们让 GPT-5.6 Sol 看着生成的图而不是原图，回答这六张图的全部 184 道题。看原图时它答对 93.5%；看由 caption 生成的图时，准确率依次是 Qwen2-VL-2B 43.5%、Qwen3-VL-2B 66.8%、Qwen3.5-2B 69.6%、GPT-6 Astra 85.3%，和这些 caption 本身在 CaptionExam 上的排序完全一致。

## 结论

CaptionExam 是一个包含 109 张图，3099 道题的高难度 captioner 评测 benchmark，能够支持最近的文生图，图生文模型的训练。

## 引用

{{< bibtex >}}

## 参考文献

[^abdin_phi4]: **Abdin, Marah, et al.** *Phi-4 Technical Report*. 2024. *arXiv*, https://arxiv.org/abs/2412.08905.

[^betker_dalle3]: **Betker, James, et al.** *Improving Image Generation with Better Captions*. OpenAI, 2023, https://cdn.openai.com/papers/dall-e-3.pdf.

[^chen_pixart]: **Chen, Junsong, et al.** "PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis." *The Twelfth International Conference on Learning Representations*, 2024. *arXiv*, https://arxiv.org/abs/2310.00426.

[^esser_sd3]: **Esser, Patrick, et al.** "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *Proceedings of the 41st International Conference on Machine Learning*, 2024. *arXiv*, https://arxiv.org/abs/2403.03206.

[^huang_capprobe]: **Huang, Mouxiao, et al.** *CapProbe: Evaluating Detailed Image Captions via Full-Scene Dense Question Answering*. 2026. *arXiv*, https://arxiv.org/abs/2608.11074.

[^liu_pgv3]: **Liu, Bingchen, et al.** *Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models*. 2024. *arXiv*, https://arxiv.org/abs/2409.10695.

[^lu_comprecap]: **Lu, Fan, et al.** "Benchmarking Large Vision-Language Models via Directed Scene Graph for Comprehensive Image Captioning." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2025. *arXiv*, https://arxiv.org/abs/2412.08614.

[^mosi_moss]: **MOSI.AI.** *MOSS-Image Technical Report*. 2026. *arXiv*, https://arxiv.org/abs/2610.23374.

[^tang_cccaption]: **Tang, Zhijiang, et al.** "CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2026. *arXiv*, https://arxiv.org/abs/2602.21655.

[^ye_decapbench]: **Ye, Qinghao, et al.** "Painting with Words: Elevating Detailed Image Captioning with Benchmark and Alignment Learning." *The Thirteenth International Conference on Learning Representations*, 2025. *arXiv*, https://arxiv.org/abs/2503.07906.
