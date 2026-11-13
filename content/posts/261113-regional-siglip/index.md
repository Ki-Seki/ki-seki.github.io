---
date: '2026-11-13T11:28:01+08:00'
title: 'Regional SigLIP: Finding What Is Where with One Vector per Image'
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

> **Outline.** Each section below lists what it will say. Numbers come from the 20k-step ablation and the long run; the prose and figures come later.

## Introduction

- The query: "a logo in the top-left corner", written as `{"bbox": [x1, y1, x2, y2], "description": "..."}` with xyxy in 0–1000. Ordinary CLIP/SigLIP search matches the content and ignores where it is.
- Why: picking training data by layout, such as SFT data for design generation. This follows on from the billion-scale retrieval post, which searches by content only.
- The constraint: keep one vector per image, so the IVF-PQ index from that post still works. Position goes into the query, never into the index.
- The three findings, one line each:
  1. one vector per image can learn position, but slowly; position grows with training steps;
  2. a loss only teaches the score differences it compares;
  3. how to test whether a model actually reads the box.
- Figure 1: the JSON query goes through the text tower, the image through the image tower, and the score is one dot product.

## Related Work

- Region-aware CLIP models (RegionCLIP, Alpha-CLIP and others): they take the region as input on the image side, so the index has to be rebuilt for each region.
- One vector per object: ObjEmbed (ICML 2026) is the closest comparison; its index grows with the number of objects.
- Detection and grounding (GLIP, Grounding DINO): these answer "where" for a single image, but cannot search a billion images.
- Coordinates as text tokens: Pix2Seq, Kosmos-2, Shikra.
- Spatial and compositional failures of CLIP: COLA, What'sUp.
- How this work differs: the box goes into the text tower, the index stays one vector per image, and the training positives come from structured annotations.

## Method

### Query Format and Two Models

- The JSON model puts the whole JSON string into the SigLIP2 text tower.
- The residual model encodes the text and the box separately, then fuses them with a zero-initialized residual.
- The score is $s(q, I) = \langle f(\text{desc}, \text{box}), g(I) \rangle$, with sigmoid outputs.

### Training Data

- In-house design images, with a box and a description for every element (from layer detection plus captioning).
- 4 positive queries per image. The window is the tight box with probability 1/3; otherwise it is a larger window that contains the box. The text is the full description, its first sentence, keywords or the label.

### Negatives and the Loss

- Same image, same description, wrong window: 2 per image, placed where they touch no related box.
- Unrelated images: they share no words with the query and no related category.
- Another image that has the same category, but outside the window: this exists only for about 25 everyday categories.
- One sigmoid loss with weights 1 / 0.5 / 0.5 / 0.35; all other pairs are ignored.
- Figure 2: one design image with each kind of positive and negative drawn on it.

## How to Tell Whether a Model Reads the Box

- Region retrieval hit@10 alone can be passed without looking at the box.
- Each test holds something fixed and changes one thing:
  - region retrieval: same query, different images (949-image gallery, 3,725 region queries);
  - mirror box: same image and text, with the box mirrored (2,483 trials);
  - mirror image: the image mirrored instead of the box;
  - jigsaw swap: two images side by side or stacked, the query about one half; 50% means text only (14,900 trials).
- A pitfall: the spatial-pair metric logged during training already read 86% at 2,500 steps, far too optimistic. Trust the jigsaw test.
- Figure 3: the four tests as small schematics.

## Position Comes with Training Steps

- The long run reaches jigsaw 92 / 93 (horizontal / vertical) at 124,745 steps, after about 32M images.
- What I got wrong: at 4,000 steps I concluded that one vector cannot learn position. The deployed model's region queries proved otherwise; it simply needed more training.
- Figure 4: jigsaw accuracy against steps, for the long run and the 20k ablation.

## Six Models at 20k Steps

- Same recipe for all: batch 128, 8 GPUs, 20,000 steps, same data and seed.
- Three objectives, each in a JSON and a residual version:
  - the retrieval loss above;
  - crop distillation: a frozen SigLIP2 scores the crop at the window, and the student matches those scores;
  - pair distillation: the student matches the teacher on the right window against a moved window, on the same image.
- Table:

  | Model | Jigsaw H / V | Mirror box | Hit@10 |
  |---|---|---|---|
  | JSON baseline | 74 / 78 | 69 | 73 |
  | Residual | 59 / 66 | 65 | 78 |
  | Crop distillation, JSON | 54 / 59 | 52 | 82 |
  | Crop distillation, residual | 54 / 53 | 50 | 84 |
  | Pair distillation, JSON | 60 / 77 | 76 | 15 |
  | Pair distillation, residual | 52 / 57 | 59 | 5 |

- Each test has a different winner. Figure 5 shows real queries side by side; it needs public images.

## Why Each Loss Wins a Different Test

- Every test compares two scores with something held fixed. A loss shapes only the differences it compares.
- There are three such differences: which image, which box and which layout.
- Crop distillation never sees "same text, same image, box moved", so the mirror box test stays at 50% for all 20,000 steps.
- Pair distillation compares two windows on the same image, so the content term cancels and retrieval collapses: hit@10 drops from 37 to 1 by step 2,500.
- The retrieval loss covers all three differences, but its position signal is scarce and easy, so it learns slowly.
- Figure 6: a matrix of losses against the three differences, with training curves.

## What Would Make Position Faster

- Why the retrieval loss is slow at position:
  - only 2 wrong-window negatives per image, mostly on empty background;
  - the cross-image layout signal exists only for about 25 categories;
  - at 20k steps the wrong-window loss is the largest term left (0.317, against 0.064 for unrelated images).
- More of the same data is not the lever:
  - neither run ever repeated an image (2.56M images in the ablation; about 32M of 83M in the long run);
  - doubling the images per step (256 vs 128) gave only 2–3 points at matched steps.
- A clearer loss, not yet run: three equal-weight choices.
  - Which image: keep the current term.
  - Which box: the right box against several moved boxes, half of them on other elements.
  - Which layout: the source image against its flipped and shifted copies, with flipped positives to block shortcuts. No collages in training, so the jigsaw test stays held out.
- Denser data: images with several instances of the same kind, and denser boxes.
- Figure 7: the three-question loss beside the current one.

## Limitations

- A small gallery (949 images), one seed, no public benchmark, and in-house training data.

## Conclusion

- Three takeaways, matching the three findings in the introduction.

## Citation

{{< bibtex >}}

## References
