---
date: '2026-10-26T21:31:14+08:00'
title: 'CaptionExam：一个高难度的 captioner QA benchmark'
author:
  - Elliot Shichao Song
summary: '文生图模型越来越依赖详细的 caption 来训练，写 caption 的 captioner 因此变得重要。CaptionExam 让一个只读文字的判官仅凭 caption 回答关于图像的细粒度问题，按答对多少来给 caption 打分，并区分 caption 是漏说了还是说错了。'
draft: false
tags:
  - image-captioning
  - benchmark
  - evaluation
  - text-to-image
  - vision-language-model
math: false
---

## 引言

越来越多的图像生成模型选择用详细丰富的 caption 来训练，而不是网络图片自带的简短 alt 文本。DALL·E 3 证明，用专门的 captioner 把训练集重新标注一遍，能让文生图模型明显更好地遵循 prompt[^betker_dalle3]；PixArt-α、Stable Diffusion 3 和 Playground v3 也都用密集的合成 caption 训练[^chen_pixart][^esser_sd3][^liu_pgv3]。如今的 renderer 已经能还原 caption 中的大部分信息，比如 MOSS-Image 就以渲染密集、准确的文字和连贯的版式为目标[^mosi_moss]。

renderer 还原得越好，captioner 就越决定上限：它漏掉的每个细节，生成模型都学不会控制；它编造的每个细节，都在教生成模型错误的东西。因此，一个好的 captioner，也就是负责标注训练数据的模型，和生成模型本身同样重要。

要选出这样的 captioner，需要一个足够难、能拉开强模型差距，又足够具体、能指出每个模型错在哪里的 benchmark。BLEU、CIDEr 这类基于参考答案的指标只看和一段简短参考文本的词语重合，面对长 caption 基本失效；让 LLM 给整段 caption 打分，又只能得到一个总分，指不出到底哪个事实错了。CaptionExam 换了个思路：让 caption 参加一场考试。

### 工作方式

{{< media
src="assets/how-it-works.svg"
caption="图 1：被测 captioner 为图像写 caption；一个只读文字、看不到图的判官，仅凭 caption 回答关于这张图的选择题。"
>}}

每张图都配有一组关于其细节的选择题。判官模型读 caption，但看不到图，仅凭 caption 回答每道题。每道题都多给一个选项 *"not mentioned / indeterminable"*（未提及），所以每次作答都落入以下三种结果之一：

- **答对**（correct）：判官选中了正确答案，说明 caption 说清了这个事实；
- **漏述**（omission）：判官选了「未提及」，说明 caption 没有说；
- **幻觉**（hallucination）：判官选了错误选项，说明 caption 说错了。

一道题长这样：

```json
{"id": "img_0001_q02", "image_id": "img_0001",
 "question": "What color is the body of the pickup truck parked under the carport?",
 "options": ["white or light silver", "dark blue or black", "red or maroon", "dark green",
             "not mentioned / indeterminable"],
 "answer": ["white or light silver"],
 "bbox": [453, 552, 622, 804]}
```

几个错误选项对这张图来说都说得通，所以如果没有从 caption 中得知卡车的颜色，判官猜不出答案。一份 caption 的**准确率**（accuracy）就是答对题目的比例，其余部分分别计为**漏述率**和**幻觉率**。

## 相关工作

与我们最接近的是 **CapProbe**[^huang_capprobe]。它同样让一个只读文字的判官仅凭 caption 回答选择题，并用一个 *Uncertain* 选项区分 caption 漏掉的和说错的内容。它的题目按区域覆盖整个画面生成：346 张图、25,650 道题、10 个语义类别，用有效准确率和基于信息密度的指标打分。CaptionExam 的不同在于题目的来源：我们的题是从真实 captioning 模型的失误中挖出来的，而不是为了覆盖画面而生成的，因此题量更小（109 张图、3,099 道题），对强 captioner 来说却很难。另外，CapProbe 的数据和代码尚未公开；CaptionExam 的代码和数据都已开源，判官在本地一张 GPU 上就能运行，可能更方便直接使用。

Playground v3 一同提出的 **CapsBench**[^liu_pgv3] 也是让模型仅凭 caption 回答问题：200 张图、17 个类别、2,471 道是非题。它的答案大多为「是」，而 CaptionExam 的错误选项都说得通，caption 中的错误陈述会直接表现为幻觉。**CCCaption**[^tang_cccaption] 在强化学习中用视觉问题奖励 caption 的完整性，同时用断言级的正确性奖励约束错误。另一些详细 caption 评测不出题，而是把 caption 与结构化参考比对：DeCapBench 用 DCScore 逐条检查 caption 拆出的原子事实[^ye_decapbench]，CompreCap 则将 caption 与人工标注的有向场景图比对[^lu_comprecap]。

## Benchmark 构成

| | |
|---|---|
| 图像 | 109 张：照片、插画、商品图、海报、书籍封面、文字密集的图形 |
| 题目 | 3,099 道，每张图 15–42 道（平均 28 道） |
| 选项 | 每题 3–6 个（多数为 5 个），其中一个是「未提及」 |
| 区域 | 120 道题针对整张图，其余题目各自指向一个区域，区域通常只占全图的 4% 左右 |
| 随机猜测 | 在「未提及」以外的选项中均匀随机选择，准确率为 26.6% |

**题目问什么。** 题目可以涉及图中任何可见的方面：有哪些物体、有几个，姿态与朝向，空间关系与版式，颜色及其具体色调，材质与表面质感，光照，图中文字及其确切内容，重复出现的文字，字体、字重与大小写，图形风格与绘制手法，背景，以及被画面边缘截断的内容。

{{< media
src="assets/question-types.svg"
caption="图 2：3,099 道题按考察内容的分布。GPT-5.6 Luna 根据题面、选项和答案把每道题归入十类之一。"
>}}

近三成的题问的是有什么物体、是什么场景；颜色、位置与版式各占约 15%；问文字内容和字体的题合计约 13%。图 3 是一张示例图和它的四道题。

{{< media
src="assets/example-questions.webp"
caption="图 3：一张示例图和它的四道题。黄框是每道题指向的区域，绿色是正确答案。"
>}}

**题目从哪里来。** 题目是从真实 captioning 模型的失误中挖出来的：

1. Gemini 3.1 Pro（preview）同时看一张图和某个模型为它写的 caption，找出这份 caption 最严重的事实性失误（说错的内容，以及漏掉的显著内容），并针对每个失误写一道选择题。
2. Gemini 3.1 Pro 和 GPT-5.6 Sol 分别在看图的情况下回答每道候选题。只有两者都答对的题才会保留，这样可以去掉有歧义的题目和答案有误的题目。
3. 每道保留下来的题在进入 benchmark 之前都经过人工审核，审核时可以修改题面和区域。

**两个设计取舍。**

- *每个错误选项都必须对这张图说得通。* 如果只有一个选项看起来合理，判官单凭常识或措辞就能选中答案，这道题就等于白送分。这是对题目唯一的硬性要求。
- *不单设「文字精确匹配」题型。* 图中的文字可能倾斜、使用艺术字，或者被设计成图形，文字的标准答案本身就不稳定，用字符级的严格判定去测不稳定的标准答案，测出来的主要是噪声。逐元素比对还会受 captioner 如何切分文字的影响。改成选择题之后，写错、漏写、多写和重复这几种失误，都统一体现在漏述率和幻觉率里。

**打分。** 判官是 Phi-4[^abdin_phi4]，用 vLLM 在本地部署，temperature 为 0，答案被约束为某个选项的原文。每条判定都记录了题目和 caption 的指纹；排行榜只在所有模型都有有效判定的公共题目上比较。

## 评测三年来的 Qwen VLM

所有 Qwen 模型都在一台 8 卡 RTX PRO 6000 节点上用 vLLM 部署、写 caption，再由 Phi-4 打分；OpenAI 的模型通过 API 写 caption，由同一个判官打分。35 个模型的完整排行榜、每一份 caption 和每一条判定都放在 [Hugging Face](https://huggingface.co/datasets/Ki-Seki/CaptionExam-Results) 上。最强的是 GPT-6 Astra（61.2%），最强的开源模型是 Qwen3.5-397B-A17B（42.9%），两者离满分都还很远：即使是 GPT-6 Astra，也有四分之一的题它的 caption 没有提到。

### 实验设置

我们用 2024 年 8 月至 2026 年 4 月发布的 23 个 Qwen 视觉语言模型为 benchmark 写 caption：Qwen2-VL（2B、7B、72B）、Qwen2.5-VL（3B、7B、32B、72B）、Qwen3-VL（2B、4B、8B、32B、30B-A3B、235B-A22B）、Qwen3.5（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B）和 Qwen3.6（27B、35B-A3B）。所有模型使用同一个 prompt，要求尽可能详尽地描述图中一切可见内容；解码使用 greedy，最多生成 4,096 个 token。Qwen3.5 和 Qwen3.6 关闭 thinking。除 Qwen3.5-397B-A17B 外，所有模型都以 BF16 运行；397B 的 BF16 权重放不进一台节点，因此使用官方的 FP8 版本。作为参照，我们还通过 API 测了 OpenAI 的 12 个模型：GPT-5、GPT-5.4、GPT-5.6 和 GPT-6 四个系列各三个，使用同一个 prompt，reasoning effort 取各模型允许的最低一档。

### 规模与代际

{{< media
src="assets/scaling.svg"
caption="图 4：23 个 Qwen 模型的准确率与总参数量（对数坐标），每一代一条线。空心圆是混合专家（MoE）模型，横坐标取它的总参数量。"
>}}

代际比规模重要得多。Qwen3.5-2B 的准确率（32.9%）比参数多 36 倍的 Qwen2.5-VL-72B（23.9%）还高，Qwen3-VL-2B（26.0%）也已经超过它。同一代之内，模型越大分数越高，但很快就饱和了：Qwen3.5 从 0.8B 的 25.3% 涨到 27B 的 42.8%，再大到 397B 只多了 0.1 个点。总参数相近时，激活参数少的混合专家模型不如稠密模型：Qwen3-VL-30B-A3B 比 Qwen3-VL-32B 低 6.4 个点，Qwen3.5-35B-A3B 也比 Qwen3.5-27B 低 1.6 个点。

{{< media
src="assets/release.svg"
caption="图 5：各 Qwen 模型的准确率与发布时间（以 Hugging Face 仓库的创建日期计）。虚线是截至当时的最高准确率。"
>}}

最大的两次跃升发生在 Qwen2.5-VL（2025 年 1 月，最高分从 15.5% 升到 23.9%）和 Qwen3-VL（2025 年 9 月，升到 42.4%）。此后最高分基本停在 43% 左右：Qwen3.5 只比 Qwen3-VL 高 0.5 个点，Qwen3.6 也没有超过 Qwen3.5。这之后的进步主要落在小模型上，2B 一档从 Qwen3-VL 的 26.0% 升到 Qwen3.5 的 32.9%。

### 漏述、幻觉与长度

{{< media
src="assets/verdicts.svg"
caption="图 6：35 个 captioner 的三类结果占比，按准确率从高到低排列；右侧数字是幻觉率。"
>}}

新一代模型的提升几乎都来自少漏。Qwen2-VL 平均只写 210 个词，近八成的题 caption 没有提到（漏述率 79.1%），说错的很少（幻觉率 7.4%）；到 Qwen3.5，caption 长到 526 个词，漏述率降到 44.7%，幻觉率却升到 17.9%。caption 说得越多，说错的也越多。OpenAI 的模型在相近的准确率下幻觉更少：GPT-6 Luna 和 Qwen3.5-9B 的准确率都在 37.5% 上下，幻觉率分别是 12.7% 和 18.4%；GPT-6 Astra 的准确率最高，幻觉率也只有 13.6%。

{{< media
src="assets/pareto.html"
caption="图 7：每个 captioner 为每张图平均输出的 token 数与准确率。虚线是帕累托前沿：前沿上的模型，没有别的模型用更少的 token 拿到更高的准确率。左上绿色区域里的模型比中位模型 token 更少、准确率更高。悬停圆点看具体数字；悬停或点击上方的家族按钮，会把同一家族的模型按尺寸从小到大连起来。token 数按各模型自己的 tokenizer 统计，Qwen 与 OpenAI 之间只能粗略比较。"
>}}

帕累托前沿上几乎全是 OpenAI 的模型：GPT-6 Sol 平均只写 294 个 token，准确率 35.9%；GPT-5.6 Sol 写 631 个，准确率 50.7%；GPT-6 Astra 写 879 个，准确率 61.2%。Qwen 只有最早、也最短的 Qwen2-VL-72B 和 Qwen2.5-VL-7B 落在前沿左下端。最强的 Qwen3.5-397B-A17B 写 695 个 token，准确率 42.9%，比 token 更少的 GPT-5.6 Terra（612 个，46.9%）还低。

token 数相近的模型，准确率可以差得很远：GPT-5.6 Sol 和 Qwen2.5-VL-32B 都写 630 个 token 左右，准确率差了 27 个点。多写的 token 大致花在三个地方：

- **重复。** 小模型会陷入循环，把同一段话反复写到 4,096 个 token 的上限。Qwen3-VL-2B 的 109 份 caption 里有 9 份如此，Qwen3.5-0.8B 有 4 份，Qwen3-VL-4B 有 3 份。
- **格式和客套。** GPT-5.4 nano 平均每份 caption 有 6 个小标题、38 个列表项；20% 的 caption 带一段总结，把前文再说一遍；24% 在末尾主动提出 “If you want, I can also…”。这些 token 不带来任何新事实。
- **写得多，却不具体。** 长 caption 常把篇幅花在笼统的风格、质感和不确定性说明上，而题目问的是具体的东西：在哪里、什么颜色、被画面哪条边截断。

下面是同一张木瓜插画的两份 caption。GPT-5.4 nano 写了 678 个词（980 个 token），38 道题只答对 5 道；GPT-6 Sol 只写了 201 个词（265 个 token），答对 16 道。

**GPT-5.4 nano**（节选；答对 5 道，漏述 30 道，幻觉 3 道）：

> The dominant visible objects are triangular-to-oval papaya half-slice segments (like cross-sections). … Because the pattern is tiled/scattered, the exact count in the single frame is hard to guarantee, but the visible frame contains many papaya slices—roughly around two dozen total fragments/slices—distributed across the canvas. … The “material” is purely the digital illustration texture: … In summary: a seamless illustrated pattern of papaya half-slice segments…

**GPT-6 Sol**（节选；答对 16 道，漏述 18 道，幻觉 4 道）：

> Five prominent papayas are cut lengthwise, showing orange-red flesh around pale inner cavities filled with many small, dark blue-gray and black seeds. They sit at the upper left (partly cropped), upper center-right (partly cropped), upper right-center, left-center, lower left, and lower center. … Three prominent whole papayas have elongated, tapered shapes and yellow skin shaded green and orange: one extends down from the top near the center-left, one lies diagonally at the right-center, and one enters from the bottom edge near the center-left.

nano 的 caption 一直在说「很多切开的木瓜」，却没有提到画面里还有完整的木瓜，也没有交代每个木瓜在哪里、哪些被边缘截断，于是这些题判官只能选「未提及」；问到画里的木瓜是什么形态时，判官据它选了「只有切半的和切块的」，算作幻觉。Sol 的 caption 只有它的三分之一长，却逐个交代了切开的和完整的木瓜各在哪里、哪些被截断。

### 从 caption 回到图像

我们把四代 Qwen 中最小的模型（Qwen2-VL-2B、Qwen2.5-VL-3B、Qwen3-VL-2B、Qwen3.5-2B）为同一张图写的 caption 原样交给 MOSS-Image[^mosi_moss] 生成图像，另放最强的 captioner GPT-6 Astra 作参照（图 8）。生成模型、随机种子和采样设置都相同，差别只来自 caption。caption 在 CaptionExam 上得分越高，生成的图越接近原图：低分 caption 只够还原「一套石榴石首饰」「一间酒店客房」这样的大意，物体怎么摆、画面里写了什么字，都由生成模型自己补；高分 caption 把这些写了出来，构图也就跟着对了。脚部解剖图最明显：Qwen2-VL-2B 没有写出图中的标注文字，MOSS-Image 便自己编了一套中文标注。

{{< media
src="assets/caption-to-image.webp"
caption="图 8：每一行是一张图。左起第一列是原图，其余各列是 MOSS-Image 根据各个 captioner 的 caption 生成的图（seed 42，50 步，cfg 3.0），按 captioner 的总体准确率从低到高排列，最右一列 GPT-6 Astra 作参照。每张生成图下方是这份 caption 在该图题目上的准确率。"
>}}

这说明两件事。训练时，captioner 漏掉或写错的细节，生成模型都学不到正确的对应；使用时，用户随手写的 prompt 往往比低分 caption 还要简略，能把它扩写成具体、准确描述的 prompt enhancer，同样决定了生成结果离想要的画面有多近。

题目从不提及 caption，因此也可以直接用来评估生成的图像。我们让 GPT-5.6 Sol 看着生成的图而不是原图，回答这六张图的全部 184 道题。看原图时它答对 93.5%；看由 caption 生成的图时，准确率依次是 Qwen2-VL-2B 43.5%、Qwen2.5-VL-3B 57.1%、Qwen3-VL-2B 66.8%、Qwen3.5-2B 69.6%、GPT-6 Astra 85.3%，和这些 caption 本身在 CaptionExam 上的排序完全一致。生成的图总会画出点什么，所以看图时答错几乎都是选了错误选项，而不是「未提及」：从 Qwen2-VL-2B 的 caption 生成的图上，38.6% 的题选错了。只有六张图，这些数字只能说明大致趋势。

## 小模型当判官够用吗？

用最强的闭源模型当判官或许更准，但每次评测都要花钱、占用 API 配额，还依赖一个可能随时变化的模型。我们选择了 Phi-4：一个 14B 的模型，一张 GPU 就能运行，不按调用计费。在此前对 11 个 captioner、约 3,100 道题的校准中，它与一个强大的闭源判官在 75.5% 的题目上答案一致（三类结果上的 Cohen's κ = 0.62），两个判官对 captioner 的排名也几乎相同（Spearman ρ = 0.89）。模型排名是稳定的，但单条判定并不等于真值。

这次我们在三个 Qwen 模型的 caption 上重复了这项比较：从 3,099 道题里随机抽 1,000 道，让 GPT-5.6 Sol 分别判 Qwen2.5-VL-72B、Qwen3-VL-235B-A22B 和 Qwen3.5-397B-A17B 的 caption。两个判官在 76.9% 的判定上一致（Cohen's κ = 0.61），与此前的校准相当。准确率几乎相同：Qwen2.5-VL-72B 都是 23.2%，Qwen3-VL-235B-A22B 是 41.5% 对 41.6%；Qwen3.5-397B-A17B 差了 3 个点（41.4% 对 44.5%），它和 Qwen3-VL-235B-A22B 的先后也因此换了位置，不过两者在全部题目上本来只差 0.5 个点。分歧有一个偏向：Phi-4 判为选错、而 GPT-5.6 Sol 判为「未提及」的有 161 条，反过来只有 55 条，所以 Phi-4 给出的幻觉率要高 3 到 7 个点。比较模型时，准确率和排名更可靠，幻觉率的绝对值要谨慎看待。

## 局限性

- **判官会出错。** 不同判官给出的模型排名高度一致，但单条判定并不一致；分数只有在同一个判官下才能比较。
- **图像数量少。** 只有 109 张图，模型之间一两个点的差距在噪声范围之内。
- **题目偏向某些细节。** 题目是从特定 captioner 的失误中挖出来的，因此会侧重这些 captioner 容易漏掉的细节。它衡量的是 caption 是否完整、是否忠实，而不是行文是否流畅、组织是否清晰。
- **只用了一个 prompt。** 每个 captioner 只用一个 prompt 测试（Qwen 用 greedy 解码，OpenAI 的模型用 API 的默认设置），换一个 prompt 可能会改变分数。

## 结论

CaptionExam 用一个本地的小判官，把 caption 的好坏拆成答对、漏述和幻觉三部分。在 35 个 captioner 上，结论相当清楚：代际比规模重要，Qwen3.5-2B 已经超过两代之前的 72B 模型，但 Qwen 的最高分从 Qwen3-VL 起就停在 43% 左右；caption 写得长不等于写得好，帕累托前沿上几乎都是 OpenAI 的模型，它们用更少或相近的 token 说出了更多具体的事实；即使是最强的 GPT-6 Astra，也有四分之一的题没有提到。把 caption 交给 MOSS-Image 生成图像，得分越高的 caption 生成的图越接近原图：captioner 的能力直接决定了生成模型能学到什么、画出什么。

CaptionExam 的代码见 [github.com/Ki-Seki/CaptionExam](https://github.com/Ki-Seki/CaptionExam)，数据见 [huggingface.co/datasets/Ki-Seki/CaptionExam](https://huggingface.co/datasets/Ki-Seki/CaptionExam)。

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
