---
date: '2026-11-17T14:00:00+08:00'
title: '文字笔画分割模型'
author:
  - Elliot Shichao Song
summary: ''
tags:
  - text-segmentation
  - benchmark
  - evaluation
math: false
---

## 引言

（待写）

## 方法

（待写）

## 评测

文字笔画分割的任务是：输入一张图，输出一张二值 mask，标出属于文字笔画的像素，字与字之间、笔画之间的背景都不算。

### 测试集

领域里最常用的 TextSeg [^xu_textseg] 和 Total-Text [^chng_totaltext] 需要向作者申请使用许可，我们没有拿到，所以没有用。最后选了六个公开测试集，加上我们自己标注的 Crello 和自建的 MLD-TS：

| 测试集 | 图片数 | 图片 | 文字 | 标注 |
|---|---:|---|---|---|
| HierText [^long_hiertext] | 1,634 | 自然场景和文档照片，字小而密 | 以英文为主 | 笔画级，由 Hi-SAM 作者补标 [^ye_hisam] |
| WAS-R [^xie_was] | 3,000 | 海报、卡片、封面、logo、商品、招牌上的艺术字 | 以英文为主 | 笔画级 |
| TextLogo3K [^wang_textlogo3k] | 300 | 中文电影海报 | 中文 | 只标了中文标题 |
| ICDAR 2013 FST [^karatzas_icdar13] | 233 | 对焦拍摄的场景文字 | 英文 | 笔画级 |
| COCO-TS [^bonechi_cocots] | 100 | COCO 日常照片里的文字 | 以英文为主 | 笔画级，EAFormer 作者重标 [^yu_eaformer] |
| MLT-S [^bonechi_mlts] | 100 | ICDAR 2017 MLT [^nayef_mlt17] 场景照片 | 拉丁字母为主，另有少量韩文、阿拉伯文、日文、中文、孟加拉文 | 笔画级，EAFormer 作者重标 |
| Crello [^yamaguchi_canvasvae] | 300 | 模板设计图 | 几乎全是英文 | 我们由设计图层渲染 |
| MLD-TS | 230 | 海报为主，另有广告、封面、传单等设计图 | 中文 110 张，日文、韩文、阿拉伯文、天城文各 30 张 | 笔画级，标全部文字；模型预标，人工逐张检查 |

- **HierText、WAS-R：** 用官方测试集的全部图片。
- **TextLogo3K：** 每张海报只标了中文标题，副标题、英文译名和日期都没标。所以只在标题框各边外扩 10% 的范围内计分。这个范围里如果有没标的英文或日期，模型把它们分出来会被算作错误，这是这个测试集的已知偏差。
- **ICDAR 2013 FST：** 来自 ICDAR 2013 Robust Reading Competition 的文字分割任务。分割标注里每个字符一种颜色，我们把所有非白色像素当作字；词框取自同一挑战的文字检测标注。它有两处标注习惯和别的测试集不同：个别图里整块 logo 被标成了字，空心涂鸦字的内部也被填满。这几张图像素恰好很多，对 fgIoU 的影响很大，所以 FST 上要结合每张图权重相同的 F-score 一起看（见下文指标）。
- **COCO-TS、MLT-S：** 原始标注是用弱监督自动生成的，比较粗。EAFormer 的作者人工重标了测试集，各公开了其中 100 张，我们只用这一部分。MLT-S 公开的部分缺 7 张原图，词框也没有放出；我们从 ICDAR 2017 MLT 官方补齐原图，并用它的词级标注作为词框（不计无法辨认的词）。
- **Crello：** CyberAgent 公开的模板设计数据集，每个设计都带完整的图层。我们按图层顺序重画每个设计，文字图层的字形就是真值，被上层元素挡住的部分扣掉；重画结果和官方预览图对不上的设计跳过。从 Crello 测试集随机抽了 300 张。它有两个限制：一是几乎全是英文；二是做成图片或矢量图形的字（比如 logo）不在文字图层里，真值里不算字，大约五分之一的图有这种字，我们会补标。另外，我们的模型初始化自 LayerD [^suzuki_layerd]，而 LayerD 在 Crello 训练集上训练过，所以在这个测试集上可能有主场优势。
- **MLD-TS：** 我们自建的多语言设计图测试集，涵盖中文、日文、韩文、阿拉伯文和天城文。每张图都按笔画级标注，GPT 6.1 Sol 和 GPT 6.2 Astra 先做两轮预标，再由人工逐张检查，确保标注质量。

### 指标

- **fgIoU（主指标）：** 文字像素的交并比。每个测试集把所有图片的交集像素和并集像素分别加总再相除，和领域里通行的算法一致。我们用 ST-SAM [^wei_stsam] 作者放出的权重在 FST 上跑出 72.6，和它论文里的 72.62 一致。这种算法下大图的权重大。
- **F-score：** 领域里通行的第二个指标，是文字像素的精确率（precision）和召回率（recall）的调和平均。我们按 TexRNet [^xu_textseg] 官方代码的算法：每张图各算精确率和召回率，对所有图取平均，再合成 F-score。每张图的权重相同，和 fgIoU 互补。不同论文的 F-score 算法并不一致，这里统一用这一种，所有方法都由我们重新计算。
- **精确率、召回率：** F-score 的两个分量。精确率低，说明把背景和装饰抠了进来；召回率低，说明有字没抠出来。
- **漏词率：** 上面几个像素指标对整个词漏掉都不敏感。一张图里漏掉一个小词，丢的像素很少，几乎不扣分，但对用户来说就是一个词没抠出来。所以我们按词框统计：框内真值文字像素不少于 20 个的词才计入，模型分出的文字像素不到真值的 20%，就算漏掉。
- **置信区间：** 按图片做 1,000 次配对自助抽样（paired bootstrap），所有方法用同一批抽样，报 95% 区间；两个方法之差的区间也用同一批抽样算。
- **速度：** 每张图的平均秒数。所有方法在同一张 RTX PRO 6000 Blackwell 上逐个单独跑；需要先找字的方法，时间里包括找字。

### 对比方法

我们只和放出了权重的方法比，在同样的测试集和规则下自己重跑：

- **Hi-SAM [^ye_hisam]：** 放出的四个权重，即 SAM-TS-H 分别在 TextSeg、HierText、Total-Text 上训练的三个版本，和在 HierText 上训练、按滑窗推理的 SAM-TS-L。另加一个组合：SAM-TS-H TextSeg 版和 SAM-TS-L 滑窗版的概率图逐像素取最大，它在验证集上比任何单个版本都好。
- **ST-SAM [^wei_stsam]：** 作者只放出了 ViT-B 的权重，我们用 TextSeg 版。FST 上另加作者在 FST 训练集上训练的版本，作为见过这个领域的参照。
- **ConTextV [^zhang_context]：** 基于视觉上下文学习，把文字去除和文字分割放在一个模型里做。
- **UPOCR [^peng_upocr]：** 统一的像素级 OCR 模型，用它的文字分割任务，按滑窗推理。
- **SAM 3 [^carion_sam3]：** 用文字概念作提示词，提示词和置信度阈值都在验证集上选。

没有放出权重的方法，比如 WASNet [^xie_was]、EAFormer [^yu_eaformer] 和 TexRNet [^xu_textseg]（权重链接已失效），只在测试集划分和我们一致时，引用它们论文里的分数。

## 结果

（待写）

## 结论

（待写）

## References

[^bonechi_cocots]: **Bonechi, Simone, et al.** *COCO_TS Dataset: Pixel-level Annotations Based on Weak Supervision for Scene Text Segmentation*. 2019. *ICANN*, https://arxiv.org/abs/1904.00818.

[^bonechi_mlts]: **Bonechi, Simone, et al.** *Weak Supervision for Generating Pixel-Level Annotations in Scene Text Segmentation*. 2020. *Pattern Recognition Letters*, https://arxiv.org/abs/1911.09026.

[^carion_sam3]: **Carion, Nicolas, et al.** *SAM 3: Segment Anything with Concepts*. 2025. *arXiv*, https://arxiv.org/abs/2511.16719.

[^chng_totaltext]: **Ch'ng, Chee Kheng, and Chee Seng Chan.** *Total-Text: A Comprehensive Dataset for Scene Text Detection and Recognition*. 2017. *ICDAR*, https://arxiv.org/abs/1710.10400.

[^karatzas_icdar13]: **Karatzas, Dimosthenis, et al.** *ICDAR 2013 Robust Reading Competition*. 2013. *ICDAR*, https://rrc.cvc.uab.es/?ch=2.

[^long_hiertext]: **Long, Shangbang, et al.** *Towards End-to-End Unified Scene Text Detection and Layout Analysis*. 2022. *CVPR*, https://arxiv.org/abs/2203.15143.

[^nayef_mlt17]: **Nayef, Nibal, et al.** *ICDAR2017 Robust Reading Challenge on Multi-Lingual Scene Text Detection and Script Identification – RRC-MLT*. 2017. *ICDAR*, https://rrc.cvc.uab.es/?ch=8.

[^peng_upocr]: **Peng, Dezhi, et al.** *UPOCR: Towards Unified Pixel-Level OCR Interface*. 2024. *ICML*, https://arxiv.org/abs/2312.02694.

[^suzuki_layerd]: **Suzuki, Tomoyuki, et al.** *LayerD: Decomposing Raster Graphic Designs into Layers*. 2025. *ICCV*, https://arxiv.org/abs/2509.25134.

[^wang_textlogo3k]: **Wang, Yizhi, et al.** *Aesthetic Text Logo Synthesis via Content-aware Layout Inferring*. 2022. *CVPR*, https://arxiv.org/abs/2204.02701.

[^wei_stsam]: **Wei, Jin, et al.** *ST-SAM: Multimodal Scene Text Segmentation with Dense Visual and Sparse Textual Prompts via SAM*. 2026. *AAAI*, https://ojs.aaai.org/index.php/AAAI/article/view/38025.

[^xie_was]: **Xie, Xudong, et al.** *WAS: Dataset and Methods for Artistic Text Segmentation*. 2024. *ECCV*, https://arxiv.org/abs/2408.00106.

[^xu_textseg]: **Xu, Xingqian, et al.** *Rethinking Text Segmentation: A Novel Dataset and A Text-Specific Refinement Approach*. 2021. *CVPR*, https://arxiv.org/abs/2011.14021.

[^yamaguchi_canvasvae]: **Yamaguchi, Kota.** *CanvasVAE: Learning to Generate Vector Graphic Documents*. 2021. *ICCV*, https://arxiv.org/abs/2108.01249.

[^ye_hisam]: **Ye, Maoyuan, et al.** *Hi-SAM: Marrying Segment Anything Model for Hierarchical Text Segmentation*. 2025. *IEEE TPAMI*, https://arxiv.org/abs/2401.17904.

[^yu_eaformer]: **Yu, Haiyang, et al.** *EAFormer: Scene Text Segmentation with Edge-Aware Transformers*. 2024. *ECCV*, https://arxiv.org/abs/2407.17020.

[^zhang_context]: **Zhang, Fei, et al.** *ConText: Driving In-context Learning for Text Removal and Segmentation*. 2025. *ICML*, https://arxiv.org/abs/2506.03799.
