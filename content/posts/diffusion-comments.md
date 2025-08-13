---
math: true
---

本文致力于几乎零数学背景知识和零生成模型的情况下对《What are Diffusion Models?》进行完善的注释。

## Introduction

> GAN, VAE, and Flow-based models

![Generative Models](/images/Generative_Models.png)

- GAN 生成对抗网络：训练两个网络，一个用于生成图像，一个用于判别图像的真伪
- VAE 变分自编码器模型：通过编码器将输入图像压缩为潜在空间变量，再通过解码器重建图像
- Flow matching models：通过流动匹配的方法生成图像，是函数级别的过程组合起来生成图像的。

## What are Diffusion Models?

### Forward diffusion process

> 噪声

对数据的随机扰动

---

> $\mathbf{x}_0 \sim q(\mathbf{x})$

表示从真实数据分布 $q(\mathbf{x})$ 中采样得到的样本 $\mathbf{x}_0$，其中

- $\mathbf{x}_0$：表示一个真实数据样本，比如一张图像、一段语音或一个文本向量。
- $q(\mathbf{x})$：表示真实数据的分布，也叫经验分布，比如训练集中的图像分布。

---

> $\mathbf{x}_1, \dots, \mathbf{x}_T$

每一层加噪后的输出结果

---

> $\{\beta_t \in (0, 1)\}_{t=1}^T$

variance schedule.之所以叫做variance schedule，是因为它定义了在扩散过程中每个时间步的方差大小。和学习率调度是类似的。

---

> $$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$

---

> isotropic Gaussian distribution

各方向都均匀的高斯噪声，即张量中的所有值符合 $\mathcal{N}(0, \mathbf{I})$。
