---
title: 'CaptionExam: A Hard QA Benchmark for Image Captioners'
author:
  - Elliot Shichao Song
summary: 'Text-to-image models now learn from detailed captions, so the captioner that writes them matters. CaptionExam grades a caption by how many fine-grained questions about the image a text-only judge can answer from it, and tells apart what the caption left out from what it got wrong.'
draft: true
tags:
  - image-captioning
  - benchmark
  - evaluation
  - text-to-image
  - vision-language-model
math: false
---

## Introduction

More and more image generation models are trained on long, detailed captions instead of the short alt text that comes with web images. DALL·E 3 showed that recaptioning the training set with a dedicated captioner makes a text-to-image model follow prompts much more closely[^betker_dalle3], and PixArt-α, Stable Diffusion 3 and Playground v3 all train on dense synthetic captions[^chen_pixart][^esser_sd3][^liu_pgv3]. Today's renderers can reproduce most of what such a caption says; MOSS-Image, for example, is built to render dense, accurate text and coherent layouts[^mosi_moss].

When the renderer is this faithful, the captioner sets the ceiling. Every detail it leaves out is one the generator never learns to control, and every detail it invents teaches the generator something false. A good captioner, the model that labels the training data, matters as much as the generator itself.

Choosing that captioner needs a benchmark that is hard enough to separate strong models and specific enough to say what each one gets wrong. Reference-based metrics such as BLEU and CIDEr reward word overlap with a short reference and fall apart on long captions. An LLM that grades a whole caption returns a single score and cannot point to the fact that is wrong. CaptionExam instead gives the caption an exam.

### How It Works

{{< media
src="assets/how-it-works.svg"
caption="Figure 1: The captioner under test describes the image. A text-only judge, which never sees the image, answers multiple-choice questions about the image from the caption alone."
>}}

Each image comes with a set of multiple-choice questions about its details. A judge model reads the caption, but never the image, and answers every question from the caption alone. Every question also offers the option *"not mentioned / indeterminable"*, so each answer falls into one of three outcomes:

- **correct**: the judge picked the right answer, so the caption states the fact;
- **omission**: the judge picked *not mentioned*, so the caption does not say;
- **hallucination**: the judge picked a wrong option, so the caption says something false.

A question looks like this:

```json
{"id": "img_0001_q02", "image_id": "img_0001",
 "question": "What color is the body of the pickup truck parked under the carport?",
 "options": ["white or light silver", "dark blue or black", "red or maroon", "dark green",
             "not mentioned / indeterminable"],
 "answer": ["white or light silver"],
 "bbox": [453, 552, 622, 804]}
```

The wrong options are all plausible for this image, so a judge that has not been told the truck's color cannot guess it. A caption's **accuracy** is the share of questions answered correctly; the **omission rate** and the **hallucination rate** account for the rest.

## Related Work

**CapProbe**[^huang_capprobe] is the closest work. It also has a text-only judge answer multiple-choice questions from a caption, with an *Uncertain* option that separates what a caption leaves out from what it gets wrong. Its questions are generated region by region over the whole scene: 25,650 questions on 346 images, in 10 semantic categories, scored with an effective accuracy and density-based metrics. CaptionExam differs in where its questions come from. They are mined from the actual mistakes of captioning models rather than generated to cover the scene, which makes a smaller set (3,099 questions on 109 images) hard for strong captioners. CapProbe's data and code have not been released yet; CaptionExam's are open, with a judge that runs on one local GPU.

**CapsBench**, introduced with Playground v3[^liu_pgv3], uses the same idea of answering questions from the caption alone: 2,471 yes/no questions over 200 images and 17 categories. Most of its answers are *yes*, whereas CaptionExam's plausible wrong options make a false statement in the caption show up as a hallucination. **CCCaption**[^tang_cccaption] uses visual questions to reward a caption's completeness during reinforcement learning, alongside a claim-level correctness reward. Other detailed-caption benchmarks grade captions against structured references instead of questions: DeCapBench scores the caption's atomic facts with DCScore[^ye_decapbench], and CompreCap compares it against a human-annotated directed scene graph[^lu_comprecap].

## The Benchmark

| | |
|---|---|
| Images | 109: photos, illustrations, product shots, posters, book covers, text-heavy graphics |
| Questions | 3,099, 15–42 per image (28 on average) |
| Options | 3–6 per question (most have 5), one of them *not mentioned / indeterminable* |
| Regions | 120 questions are about the whole image; the rest point at a region, typically about 4% of the image |
| Random guessing | 26.6% accuracy, choosing uniformly among the options other than *not mentioned* |

**What the questions ask.** A question can be about any visible aspect of the image: which objects are present and how many, pose and facing direction, spatial relations and layout, colors and their exact shade, materials and finish, lighting, rendered text and its exact wording, repeated text, typeface and casing, graphic style and rendering technique, the background, and things cropped at the frame.

*[TODO: a figure with the distribution of question types and an example image with a few of its questions.]*

**How the questions were made.** The questions were mined from the mistakes of real captioning models:

1. A strong vision-language model looked at an image together with one model's caption, found the caption's most severe factual failures (what it stated wrongly and what salient things it left out), and wrote one multiple-choice question about each.
2. Two other strong vision-language models answered every candidate question while looking at the image. A question was kept only if both answered it correctly, which removes ambiguous questions and wrong answer keys.
3. A person reviewed every kept question before it entered the benchmark, and could edit its wording and region.

*[TODO: name the models used for writing and verifying questions, if we want to.]*

**Two design choices.**

- *Every wrong option must be plausible for this particular image.* If only one option looks sensible, a judge can pick the answer from common sense or wording alone, and the question gives the point away. This is the one hard requirement on the questions.
- *There is no separate exact-text-match question type.* Text in images comes slanted, stylized, or turned into graphics, so the ground-truth transcription itself is unstable, and a character-level matcher on unstable ground truth mostly measures noise. Element-by-element matching also depends on how the captioner splits the text. Asked as multiple-choice questions instead, misspelled, missing, extra and repeated text all show up in the same omission and hallucination rates.

**Scoring.** The judge is Phi-4[^abdin_phi4] served locally with vLLM, at temperature 0, with its answer constrained to the exact text of one option. Every verdict is stored with fingerprints of the question and the caption it was made from, and the leaderboard ranks all models on the questions they all have current verdicts for.

## Evaluating Three Years of Qwen VLMs

*[TODO: results, from a run on a single node with 8 RTX PRO 6000 GPUs.]*

### Setup

We caption the benchmark with 23 Qwen vision-language models released from August 2024 to April 2026: Qwen2-VL (2B, 7B, 72B), Qwen2.5-VL (3B, 7B, 32B, 72B), Qwen3-VL (2B, 4B, 8B, 32B, 30B-A3B, 235B-A22B), Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B) and Qwen3.6 (27B, 35B-A3B). Every model gets the same prompt, which asks for an exhaustive description of everything visible, and decodes greedily with up to 4,096 new tokens. Qwen3.5 and Qwen3.6 run with thinking turned off. All models run in BF16 except Qwen3.5-397B-A17B, which runs in the official FP8 release because the BF16 weights do not fit on one node.

### Scale and Generations

*[TODO: Figure 2, accuracy against parameter count, one line per generation. Figure 3, accuracy against release date.]*

### Omissions, Hallucinations and Length

*[TODO: Figure 4, omission rate against hallucination rate per model. Figure 5, accuracy against caption length.]*

### From Caption Back to Image

*[TODO: generate an image from each captioner's caption with MOSS-Image[^mosi_moss], then answer the same questions while looking at the generated image instead of the original. The questions never mention a caption, so they grade the generated image directly.]*

## Is a Small Judge Enough?

A proprietary frontier model would make a stronger judge, but every evaluation would cost money and API quota and would depend on a model that can change without notice. We chose Phi-4, a 14B model that runs on one GPU at no per-call cost. In an earlier calibration over 11 captioners and about 3,100 questions, its answers agreed with a strong proprietary judge on 75.5% of the questions (Cohen's κ = 0.62 over the three outcomes), and the two judges ranked the captioners almost the same way (Spearman ρ = 0.89). Rankings are stable, but a single verdict is not ground truth.

*[TODO: repeat the comparison on a sample of the Qwen captions.]*

## Limitations

- **The judge makes mistakes.** Model rankings agree closely across judges, but individual verdicts do not, and scores are comparable only under the same judge.
- **The image set is small.** With 109 images, a difference of a point or two between models is within noise.
- **The questions favor certain details.** They were mined from what particular captioners got wrong, so they emphasize the details those captioners missed. They measure how complete and faithful a caption is, not how fluent or well organized it is.
- **One prompt.** Each captioner is tested with a single prompt and greedy decoding; another prompt can change its score.

## Conclusion

*[TODO: write once the results are in.]*

CaptionExam's code is at [github.com/Ki-Seki/CaptionExam](https://github.com/Ki-Seki/CaptionExam) and the data is at [huggingface.co/datasets/Ki-Seki/CaptionExam](https://huggingface.co/datasets/Ki-Seki/CaptionExam).

## Citation

{{< bibtex >}}

## References

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
