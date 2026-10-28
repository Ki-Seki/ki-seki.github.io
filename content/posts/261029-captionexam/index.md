---
date: '2026-10-29T00:07:13+08:00'
title: 'CaptionExam: A Challenging QA Benchmark for Captioners'
author:
  - Elliot Shichao Song
summary: 'On CaptionExam, even GPT-6 Astra only just passes at describing images, with a score of 61.2%.'
cover:
  image: "assets/captionexam-cover.png"
  relative: true
  caption: "A caption passes through the judge and splits three ways: correct, omission and hallucination (generated with [ChatGPT Images](https://chatgpt.com/))"
tags:
  - image-captioning
  - benchmark
  - evaluation
  - text-to-image
  - vision-language-model
math: false
---

- **Code:** <https://github.com/Ki-Seki/CaptionExam>
- **Data:** [Ki-Seki/CaptionExam](https://huggingface.co/datasets/Ki-Seki/CaptionExam) (109 images, 3,099 questions)
- **Results:** [Ki-Seki/CaptionExam-Results](https://huggingface.co/datasets/Ki-Seki/CaptionExam-Results) (captions, verdicts and leaderboard for 35 captioners)

## Introduction

More and more image generation models are trained on detailed, rich captions. Recent open models such as Ideogram 4 [^ideogram_4] and MOSS-Image [^mosi_moss] both add image–text pairs with very long captions to their training data. And today's renderers can already reproduce most of the information in a caption.

The better the renderer reproduces a caption, the more the captioner sets the ceiling: every detail it leaves out is one the generator never learns to control, and every detail it makes up teaches the generator something wrong. A good captioner, the model that labels the training data, therefore matters as much as the generator itself. ByteDance's Context Scaling [^chen_contextscaling] goes further and explicitly studies scaling laws for captions.

Getting a strong captioner takes two things. The first is structure: most captions today are designed as region-wise JSON, which makes it easy to scale up how detailed the annotation is. The second is information: how much a caption gets right, how much it leaves out and how much it makes up. Structure can be designed, but information can only be measured by evaluation, and what's still missing is a benchmark hard enough to separate strong models and specific enough to show where each model goes wrong. So we built CaptionExam, a very challenging QA benchmark for captioners.

{{< media
src="assets/how-it-works.svg"
caption="Figure 1: The captioner under test writes a caption for an image; a judge that reads only text and never sees the image answers multiple-choice questions about the image from the caption alone."
>}}

The benchmark has 109 images, each with a set of multiple-choice questions about its details. Once the captioner under test has written its caption, the judge model reads the caption, without seeing the image, and answers every question from the caption alone.

Clearly, the more detailed and accurate a captioner's description, the more questions the judge can answer correctly from it, and the higher the score.

## Related Work

The closest work is **CapProbe** [^huang_capprobe], which appeared at about the same time. It also has a text-only judge answer multiple-choice questions from the caption alone, and uses an *Uncertain* option to tell what a caption leaves out from what it gets wrong. Its questions are generated region by region to cover the whole image: 346 images, 25,650 questions and 10 semantic categories, scored with effective accuracy and an information-density metric.

CaptionExam differs in where its questions come from: ours are mined from the failures of real captioning models rather than generated to cover the image. So there are fewer of them (109 images, 3,099 questions), yet they are hard for strong captioners. Also, CapProbe's data and code are not yet public, while CaptionExam's code and data are both open source and its judge runs locally on one GPU, which may make it easier to use directly.

## CaptionExam Benchmark

### How the Data Is Built

The questions are mined from the failures of real proprietary captioning models (sorry, those can't be open-sourced):

1. Gemini 3.1 Pro (preview) looks at an image together with a caption some model wrote for it, finds the caption's most serious factual failures (things it gets wrong, and salient things it leaves out), and writes a multiple-choice question for each failure.
2. Gemini 3.1 Pro and GPT-5.6 Sol each answer every candidate question while looking at the image. Only questions both answer correctly are kept, which removes ambiguous questions and questions with wrong answers.
3. Every question that is kept is reviewed by a person before it enters the benchmark, and the reviewer can edit the question and its region.

### Statistics

| | |
|---|---|
| Images | 109: photos, illustrations, product shots, posters, book covers and text-heavy graphics |
| Questions | 3,099, 15–42 per image (28 on average) |
| Options | 3–6 per question (usually 5), one of which is “not mentioned” |
| Regions | 120 questions are about the whole image; each of the rest points to a region, usually about 4% of the image |
| Random guessing | Choosing uniformly at random among the options other than “not mentioned” gives 26.6% accuracy |

### What the Questions Ask

A question can be about any visible aspect of the image: which objects there are and how many, pose and orientation, spatial relations and layout, colors and their exact shades, materials and surface texture, lighting, text in the image and its exact content, repeated text, typeface, weight and case, graphic style and rendering technique, the background, and things cut off by the edge of the frame.

{{< media
src="assets/question-types.svg"
caption="Figure 2: The 3,099 questions by what they test. GPT-5.6 Luna sorted each question into one of ten types from its wording, options and answer."
>}}

Nearly 30% of the questions ask what objects or scene are present; color, and position and layout, each take about 15%; questions about text content and typography add up to about 13%. Figure 3 shows an example image and four of its questions.

{{< media
src="assets/example-questions.webp"
caption="Figure 3: An example image and four of its questions."
>}}

### Scoring with a Judge Model

The strongest closed model might make a more accurate judge, but every evaluation would cost money, use up API quota and depend on a model that can change at any time. We chose Phi-4 [^abdin_phi4], a 14B model.

Among the many small models, Phi-4 is very well calibrated against frontier models. We also had GPT-5.6 Sol judge the captions of Qwen2.5-VL-72B, Qwen3-VL-235B-A22B and Qwen3.5-397B-A17B on all 3,099 questions. The two judges agree on 77.4% of verdicts (Cohen's κ = 0.62). Accuracy under Phi-4 and under GPT-5.6 Sol is close: 23.9% against 24.1% for Qwen2.5-VL-72B, 42.4% against 43.4% for Qwen3-VL-235B-A22B, and 42.9% against 45.1% for Qwen3.5-397B-A17B, so the three models rank in the same order under both judges.

## Evaluation Results

We evaluated 23 Qwen vision-language models and 12 OpenAI models.

- Qwen: Qwen2-VL (2B, 7B, 72B), Qwen2.5-VL (3B, 7B, 32B, 72B), Qwen3-VL (2B, 4B, 8B, 32B, 30B-A3B, 235B-A22B), Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B) and Qwen3.6 (27B, 35B-A3B)
- OpenAI: GPT-5 (nano, mini, base), GPT-5.4 (nano, mini, base), GPT-5.6 (Luna, Terra, Sol) and GPT-6 (Luna, Sol, Astra)

All models run with thinking turned off and greedy decoding wherever possible, and all use the same prompt, which asks for as detailed a description of everything visible in the image as possible.

### Accuracy and Length

Clearly, saying more makes it easier to cover everything, but it can also bring redundancy and unnecessary tokens. Context Scaling [^chen_contextscaling] also finds that making natural-language captions longer barely improves the generator's training; what matters is how much image-grounded information the caption carries. Below is how each model does on token count and accuracy. For now, the models that do well on both are mostly OpenAI's.

{{< media
src="assets/pareto.html"
caption="Figure 4: Average output tokens per image and accuracy for each captioner. Models in the green area at the top left use fewer tokens than the median model and score higher."
>}}

Interestingly, within one series, such as Qwen3.5, a larger model sometimes does no better.

### Where the Extra Tokens Go

Models that write about the same number of tokens can be far apart in accuracy: GPT-5.6 Sol and Qwen2.5-VL-32B both write about 630 tokens, and their accuracy differs by 27 points. The extra tokens go to roughly three places:

- **Repetition.** Small models get stuck in loops, writing the same passage over and over until they hit the 4,096-token limit. This happens in 9 of Qwen3-VL-2B's 109 captions, 4 of Qwen3.5-0.8B's and 3 of Qwen3-VL-4B's.
- **Formatting and pleasantries.** GPT-5.4 nano's captions have 6 headings and 38 list items on average; 20% end with a summary that repeats what came before, and 24% close by offering “If you want, I can also…”. None of these tokens add a new fact.
- **Long but not specific.** Long captions often spend their length on general remarks about style, texture and uncertainty, while the questions ask about specifics: where something is, what color it is, which edge of the frame cuts it off.

For example, Figure 5 shows two captions of the same papaya illustration. GPT-5.4 nano wrote 678 words (980 tokens) and got only 5 of the 38 questions right; GPT-6 Sol wrote just 201 words (265 tokens) and got 16 right.

{{< media
src="assets/papaya-captions.webp"
caption="Figure 5: Captions of the same papaya illustration by GPT-5.4 nano and GPT-6 Sol."
>}}

Nano's caption keeps talking about “many cut papayas” but never mentions that the picture also has whole papayas, or says where each papaya is and which ones are cut off by the edge, so for those questions the judge can only choose “not mentioned”. Asked what forms the papayas take, the judge went by the caption and chose “Halved fruits and diced pieces only”, which counts as a hallucination. Sol's caption is a third as long, yet it says where each cut and each whole papaya is and which ones are cut off.

### From Captions Back to Images

Qwen's three 2B models, Qwen2-VL-2B (11.4% accuracy), Qwen3-VL-2B (26.0%) and Qwen3.5-2B (32.9%), happen to run from low to high scores. We can give their captions to MOSS-Image [^mosi_moss] to generate images and see whether the generated images also come closer to the original, with the strongest captioner, GPT-6 Astra, as a reference (Figure 6).

It is plain to see: the higher a caption scores on CaptionExam, the closer the generated image is to the original. Low-scoring captions carry only the gist, such as “a set of garnet jewelry” or “a hotel room”, and leave the generator to fill in how objects are arranged and what text the picture contains; high-scoring captions spell these out, and the composition follows. The foot anatomy chart shows it most clearly: Qwen2-VL-2B didn't write out the labels in the image, so MOSS-Image made up a set of Chinese labels of its own.

{{< media
src="assets/caption-to-image.webp"
caption="Figure 6: Each row is one image. The first column is the original; the other columns are the images MOSS-Image generated from each captioner’s caption (seed 42, 50 steps, cfg 3.0). Under each generated image is that caption’s accuracy on the image’s questions."
>}}

The questions never mention captions, so they can also be used directly to evaluate generated images. We had GPT-5.6 Sol answer all 184 questions for these six images while looking at the generated images instead of the originals. Looking at the originals, it gets 93.5% right; on the images generated from captions, its accuracy is 43.5% for Qwen2-VL-2B, 66.8% for Qwen3-VL-2B, 69.6% for Qwen3.5-2B and 85.3% for GPT-6 Astra, exactly the same order as the captions' own scores on CaptionExam.

This means two things. In training, **whatever a captioner leaves out or gets wrong, the generator can never learn the right correspondence for**; in use, the prompts people dash off are often even sketchier than low-scoring captions, so **expanding a prompt into a specific, accurate description likewise decides how close the result comes to the picture they want**.

## Conclusion

CaptionExam is a challenging captioner benchmark of 109 images and 3,099 questions that can support the development of recent text-to-image and image-to-text models. We also found that:

- **Even the strongest models are far from the ceiling.** The top scorer, GPT-6 Astra, gets only 61.2%, and the strongest open model, Qwen3.5-397B-A17B, gets 42.9%.
- **Longer is not better.** Models that write about the same number of tokens can differ by 27 points in accuracy; the extra tokens often go to repetition, formatting and pleasantries, and general description. The models that score high with few tokens are mostly OpenAI's.
- **Scale doesn't always help.** Within one series, a larger model is sometimes no stronger.
- **A caption's score shows in the images generated from it.** Given to MOSS-Image, higher-scoring captions produce images closer to the original, and answering the questions from the generated images puts them in the same order.

## Citation

{{< bibtex >}}

## References

[^abdin_phi4]: **Abdin, Marah, et al.** *Phi-4 Technical Report*. 2024. *arXiv*, https://arxiv.org/abs/2412.08905.

[^chen_contextscaling]: **Chen, Zilong, et al.** *Scaling Properties of Text Conditioning in Visual Generation*. 2026. *arXiv*, https://arxiv.org/abs/2607.29679.

[^huang_capprobe]: **Huang, Mouxiao, et al.** *CapProbe: Evaluating Detailed Image Captions via Full-Scene Dense Question Answering*. 2026. *arXiv*, https://arxiv.org/abs/2608.11074.

[^ideogram_4]: **Ideogram.** *Ideogram 4.0*. 2026, https://ideogram.ai/models/4.0/.

[^mosi_moss]: **MOSI.AI.** *MOSS-Image Technical Report*. 2026. *arXiv*, https://arxiv.org/abs/2610.23374.
