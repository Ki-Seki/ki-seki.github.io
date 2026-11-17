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

文字笔画分割的任务是：输入一张图，输出一张二值 mask，标出属于文字笔画的像素，字与字之间、笔画之间的背景都不算。我们只评端到端的设定：每个方法只拿到原图，不给任何文字框，要自己找到字，再分出笔画。

### 测试集

我们希望测试集覆盖三类图：实拍场景、艺术字和设计图、英文以外的文字。领域里最常用的 TextSeg [^xu_textseg] 和 Total-Text [^chng_totaltext] 需要向作者申请使用许可，我们没有拿到，所以没有用。最后选了六个公开测试集，加上我们自己标注的 Crello，都只用各自的测试集：

| 测试集 | 图片数 | 图片 | 文字 | 标注 |
|---|---:|---|---|---|
| HierText [^long_hiertext] | 1,634 | 自然场景和文档照片，字小而密 | 以英文为主 | 笔画级，由 Hi-SAM 作者补标 [^ye_hisam] |
| WAS-R [^xie_was] | 3,000 | 海报、卡片、封面、logo、商品、招牌上的艺术字 | 以英文为主 | 笔画级 |
| TextLogo3K [^wang_textlogo3k] | 300 | 中文电影海报 | 中文 | 只标了中文标题 |
| ICDAR 2013 FST [^karatzas_icdar13] | 233 | 对焦拍摄的场景文字 | 英文 | 笔画级 |
| COCO-TS [^bonechi_cocots] | 100 | COCO 日常照片里的文字 | 以英文为主 | 笔画级，EAFormer 作者重标 [^yu_eaformer] |
| MLT-S [^bonechi_mlts] | 93 | ICDAR 2017 MLT 场景照片 | 拉丁字母为主，另有少量阿拉伯文、中文、日文、韩文、孟加拉文 | 笔画级，EAFormer 作者重标 |
| Crello [^yamaguchi_canvasvae] | 300 | 模板设计图 | 几乎全是英文 | 我们由设计图层渲染 |

- **HierText、WAS-R：** 用官方测试集的全部图片。
- **TextLogo3K：** 每张海报只标了中文标题，副标题、英文译名和日期都没标。所以只在标题框各边外扩 10% 的范围内计分。这个范围里如果有没标的英文或日期，模型把它们分出来会被算作错误，这是这个测试集的已知偏差。
- **ICDAR 2013 FST：** 来自 ICDAR 2013 Robust Reading Competition 的文字分割任务。分割标注里每个字符一种颜色，我们把所有非白色像素当作字；词框取自同一挑战的文字检测标注。它有两处标注习惯和别的测试集不同：个别图里整块 logo 被标成了字，空心涂鸦字的内部也被填满。这几张图像素恰好很多，对 fgIoU 的影响很大，所以 FST 上我们同时报逐张平均（见下文指标）。
- **COCO-TS、MLT-S：** 原始标注是用弱监督自动生成的，比较粗。EAFormer 的作者人工重标了测试集，公开了其中约 100 张，我们只用这一部分：COCO-TS 100 张；MLT-S 有 7 张只有标注、没有图片，剩 93 张。MLT-S 只有像素标注、没有词框，不报漏词率。
- **Crello：** CyberAgent 公开的模板设计数据集，每个设计都带完整的图层。我们按图层顺序重画每个设计，文字图层的字形就是真值，被上层元素挡住的部分扣掉；重画结果和官方预览图对不上的设计跳过。从 Crello 测试集随机抽了 300 张。它有两个限制：一是几乎全是英文；二是做成图片或矢量图形的字（比如 logo）不在文字图层里，真值里不算字，大约五分之一的图有这种字，我们会补标。另外，我们的模型初始化自 LayerD [^suzuki_layerd]，而 LayerD 在 Crello 训练集上训练过，所以在这个测试集上可能有主场优势。

### 规则

- **端到端：** 所有方法都只拿到原图，不给文字框。
- **只在验证集上调参：** 每个公开数据集和 Crello 都从各自的训练集或验证集里另取一份验证集（每份 150–300 张），和测试集按图片 id 和 sha256 比对，没有重叠。阈值、提示词、组合方式都只在验证集上选。
- **测试集只跑一次：** 方法定稿后才在测试集上跑。
- **对比方法也在验证集上选版本：** 比如 Hi-SAM 放出了多个权重，我们在验证集上选平均最好的一个；同时也报每个测试集上最好版本的分数，作为对比方法的上限。
- **训练数据去重：** 我们的训练图和所有测试图做了近似重复检测，重复的图从训练集里去掉。

### 指标

- **fgIoU（主指标）：** 文字像素的交并比。每个测试集把所有图片的交集像素和并集像素分别加总再相除，和领域里通行的算法一致。我们用 ST-SAM [^wei_stsam] 作者放出的权重在 FST 上跑出 72.6，和它论文里的 72.62 一致。这种算法下大图的权重大，所以我们也报逐张平均。
- **漏词率：** fgIoU 对整个词漏掉不敏感。一张图里漏掉一个小词，丢的像素很少，几乎不扣分，但对用户来说就是一个词没抠出来。所以我们按词框统计：框内真值文字像素不少于 20 个的词才计入，模型分出的文字像素不到真值的 20%，就算漏掉。
- **混入率：** 同样在词框内，模型分成字、但真值不是字的像素，占模型分成字的像素的比例，反映把背景和装饰抠进来的程度。
- **置信区间：** 按图片做 1,000 次配对自助抽样（paired bootstrap），所有方法用同一批抽样，报 95% 区间；两个方法之差的区间也用同一批抽样算。
- **速度：** 每张图的平均秒数。所有方法在同一张 RTX PRO 6000 Blackwell 上逐个单独跑；需要先找字的方法，时间里包括找字。

结果表里每个方法同时列 fgIoU、漏词率、混入率和速度，不只看 fgIoU。

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
