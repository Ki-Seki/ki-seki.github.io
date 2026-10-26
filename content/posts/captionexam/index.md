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

*[TODO：题目类型分布图，以及一张示例图和它的几道题。]*

**题目从哪里来。** 题目是从真实 captioning 模型的失误中挖出来的：

1. 一个强大的视觉语言模型同时看一张图和某个模型为它写的 caption，找出这份 caption 最严重的事实性失误（说错的内容，以及漏掉的显著内容），并针对每个失误写一道选择题。
2. 另外两个强大的视觉语言模型在看图的情况下回答每道候选题。只有两者都答对的题才会保留，这样可以去掉有歧义的题目和答案有误的题目。
3. 每道保留下来的题在进入 benchmark 之前都经过人工审核，审核时可以修改题面和区域。

*[TODO：要不要写明出题和验题用的是哪些模型。]*

**两个设计取舍。**

- *每个错误选项都必须对这张图说得通。* 如果只有一个选项看起来合理，判官单凭常识或措辞就能选中答案，这道题就等于白送分。这是对题目唯一的硬性要求。
- *不单设「文字精确匹配」题型。* 图中的文字可能倾斜、使用艺术字，或者被设计成图形，文字的标准答案本身就不稳定，用字符级的严格判定去测不稳定的标准答案，测出来的主要是噪声。逐元素比对还会受 captioner 如何切分文字的影响。改成选择题之后，写错、漏写、多写和重复这几种失误，都统一体现在漏述率和幻觉率里。

**打分。** 判官是 Phi-4[^abdin_phi4]，用 vLLM 在本地部署，temperature 为 0，答案被约束为某个选项的原文。每条判定都记录了题目和 caption 的指纹；排行榜只在所有模型都有有效判定的公共题目上比较。

## 评测三年来的 Qwen VLM

*[TODO：结果，由一台 8 卡 RTX PRO 6000 节点跑出。]*

### 实验设置

我们用 2024 年 8 月至 2026 年 4 月发布的 23 个 Qwen 视觉语言模型为 benchmark 写 caption：Qwen2-VL（2B、7B、72B）、Qwen2.5-VL（3B、7B、32B、72B）、Qwen3-VL（2B、4B、8B、32B、30B-A3B、235B-A22B）、Qwen3.5（0.8B、2B、4B、9B、27B、35B-A3B、122B-A10B、397B-A17B）和 Qwen3.6（27B、35B-A3B）。所有模型使用同一个 prompt，要求尽可能详尽地描述图中一切可见内容；解码使用 greedy，最多生成 4,096 个 token。Qwen3.5 和 Qwen3.6 关闭 thinking。除 Qwen3.5-397B-A17B 外，所有模型都以 BF16 运行；397B 的 BF16 权重放不进一台节点，因此使用官方的 FP8 版本。

### 规模与代际

*[TODO：图 2，准确率随参数量的变化，每一代一条曲线；图 3，准确率随发布时间的变化。]*

### 漏述、幻觉与长度

*[TODO：图 4，各模型的漏述率与幻觉率；图 5，准确率与 caption 长度的关系。]*

### 从 caption 回到图像

*[TODO：用 MOSS-Image[^mosi_moss] 根据各个 captioner 写的 caption 生成图像，再看着生成的图像而不是原图回答同一套题。题目从不提及 caption，因此可以直接用来评估生成的图像。]*

## 小模型当判官够用吗？

用最强的闭源模型当判官或许更准，但每次评测都要花钱、占用 API 配额，还依赖一个可能随时变化的模型。我们选择了 Phi-4：一个 14B 的模型，一张 GPU 就能运行，不按调用计费。在此前对 11 个 captioner、约 3,100 道题的校准中，它与一个强大的闭源判官在 75.5% 的题目上答案一致（三类结果上的 Cohen's κ = 0.62），两个判官对 captioner 的排名也几乎相同（Spearman ρ = 0.89）。模型排名是稳定的，但单条判定并不等于真值。

*[TODO：在一部分 Qwen 的 caption 上重复这项比较。]*

## 局限性

- **判官会出错。** 不同判官给出的模型排名高度一致，但单条判定并不一致；分数只有在同一个判官下才能比较。
- **图像数量少。** 只有 109 张图，模型之间一两个点的差距在噪声范围之内。
- **题目偏向某些细节。** 题目是从特定 captioner 的失误中挖出来的，因此会侧重这些 captioner 容易漏掉的细节。它衡量的是 caption 是否完整、是否忠实，而不是行文是否流畅、组织是否清晰。
- **只用了一个 prompt。** 每个 captioner 只用一个 prompt 和 greedy 解码测试，换一个 prompt 可能会改变分数。

## 结论

*[TODO：等结果出来再写。]*

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
