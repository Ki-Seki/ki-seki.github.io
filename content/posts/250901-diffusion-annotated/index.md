---
date: '2025-08-15T01:41:53+08:00'
title: '《What are Diffusion Models?》注释'
author:
  - Shichao Song
tags: ["vision", "diffusion", "math"]
math: true
---

<!-- TODO: 翻译为英文 -->
<!-- TODO：标点符号用英文的 -->
<!-- TODO：更改post日期等yaml头 -->
<!-- TODO：记得email给lilian about this article -->

本文对Lilian Weng的《What are Diffusion Models?》 [^lilian_diffusion] 进行完善的注释导读。

笔者在写此文时，对图像生成模型和相关的数学背景知识了解均较少。如果你也有相似的背景，那么此文应该会适合你。当然，我可能也因此犯一些低级错误，敬请指正。

本文的结构和原文基本保持一致。每个小节中重点的公式，概念都会进行扩展的推导或解释。除此之外，文章开始附上了常见的符号解释；文末还附加上了为了看懂原文所需的一些背景知识。

### Notations

| Category / Symbol                                                                                                                                                       | Meaning                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **样本与分布**                                                                                                                                                          |                                                                                                                                                                                                                                                          |
| $\mathbf{x}_0$                                                                                                                                                          | 一个真实数据样本，比如图像、语音或文本。是一个向量（例如图像的像素向量、文本的嵌入向量等），维度可能是几百甚至几千.                                                                                                                                      |
| $\mathbf{x}_t, \, t = 1, 2, ..., T$                                                                                                                                     | 对数据样本 $\mathbf{x}_0$ 进行逐步的加噪之后的结果，最终我们得到的 $\mathbf{x}_T$ 是一个纯噪声样本                                                                                                                                                       |
| $q(\mathbf{x})$                                                                                                                                                         | 在Diffusion相关论文中，为了表示方便，$q(\mathbf{x})$ 既可以表示概率密度函数（PDF），也可以是该PDF对应的分布。这里，$q(\mathbf{x})$ 是真实数据的分布，也叫经验分布，比如训练集中的图像分布。                                                              |
| $\mathbf{x}_0 \sim q(\mathbf{x})$                                                                                                                                       | 从真实数据分布 $q(\mathbf{x})$ 中采样得到的样本 $\mathbf{x}_0$                                                                                                                                                                                           |
| $q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$                                            | 这里的$q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$ 是概率密度函数. $\mathbf{x}_t \sim \mathcal{N}(\sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$  的正态分布， $q(\mathbf{x}_t \vert \mathbf{x}_{t-1})=f(\mathbf{x}_t)$，$f(\cdot)$ 是概率密度函数 |
| **噪声超参数**                                                                                                                                                          |                                                                                                                                                                                                                                                          |
| $\beta_t$                                                                                                                                                               | Noise variance schedule parameter。超参数，他对应一个variance schedule，$\{\beta_t \in (0, 1)\}_{t=1}^T$，和学习率调度是类似的.                                                                                                                          |
| $\alpha_t$                                                                                                                                                              | $\alpha_t = 1 - \beta_t$,是为了公式书写方便而做的符号。                                                                                                                                                                                                  |
| $\bar{\alpha}_t$                                                                                                                                                        | $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，是为了公式书写方便而做的符号。                                                                                                                                                                                |
| **Diffusion 过程**                                                                                                                                                      |                                                                                                                                                                                                                                                          |
| $q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$                                            | **Forward diffusion process 递推式**。构造高斯马尔可夫链，逐步加噪，破坏数据。                                                                                                                                                                           |
| $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$                                     | **forward diffusion closed-form**，直接从数据一步得到噪声样本                                                                                                                                                                                            |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$ | **reverse diffusion 后验**，用于定义训练目标，约等于golden truth                                                                                                                                                                                         |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$   | **reverse diffusion 似然**，通过训练模型去拟合上面的后验                                                                                                                                                                                                 |
| $q(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; 0, I)$                                                                                                                     | **先验**，固定为高斯分布 $\mathcal{N}(0, I)$，推理时直接采样作为起点。                                                                                                                                                                                   |

## What are Diffusion Models?

Diffusion 模型的基本原理就是，前向扩散增加噪声，得到纯高斯分布的样本。训练模型似然逆向diffusion process，使其能从任意高斯噪声样本恢复为真实数据样本。

### Forward diffusion process

{{% admonition type="quote" title="前向扩散表达式" open=true %}}
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
{{% /admonition %}}

是前向扩散过程的两种表达形式，单步扩散过程和整体扩散过程。

单步扩散过程中，

- $\mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$ 表示 $\mathbf{x}_t$ 服从 均值为 $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$，方差为 $\beta_t\mathbf{I}$ 的高斯分布。
- $\beta_t$ 是Noise variance schedule parameter，他对应一个variance schedule，$\{\beta_t \in (0, 1)\}_{t=1}^T$，和学习率调度是类似的.
- $\beta_t$ 定义了在扩散过程中每个时间步的方差大小，一般来说$\beta_t$逐渐增大，因此和原始数据差异越来越大（$\sqrt{1 - \beta_t}$ ↓），数据变异性也逐渐变大（$\beta_t\mathbf{I}$ ↑），总体上逐渐使得每一步的噪声更多。
- $\beta_t\mathbf{I}$，是协方差矩阵，也是个对角矩阵，所有对角线元素都是 $\beta_t$. 每一维都加相同强度的噪声，不偏向任何方向。

整体扩散过程是，根据[马尔可夫性质](#markov-property)将单步扩散过程连乘起来的递推式。
整体扩散过程是我们需要的，因为他能帮助我们从真实数据分布中快速采样得到最后的纯噪声 $\mathbf{x}_T$；
然而它依赖于递推式，计算起来较慢。因此实践中会使用更简单的计算方式，即下面讲的封闭形式的公式。

{{< admonition type=quote title="前向扩散表达式的closed form形式" >}}
Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$
{{< /admonition >}}

Closed-form expression 指的是可以用有限的、明确的数学表达式直接写出来解，不需要迭代、数值近似或求解方程的公式 [^wiki_closed]。

根据单步扩散过程$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$，以及[重参数化技巧](#reparameterization-trick) $z = \mu + \sigma \cdot \epsilon$，我们可以重写单步扩散过程为：

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1}$$

这样就可以让我们来重写更详细的closed form的推导：

$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \mathcal{N}\left( \boldsymbol{\epsilon}_{t-2}; 0, \sqrt{\alpha_t (1 - \alpha_{t-1})} \right) + \mathcal{N}\left(\boldsymbol{\epsilon}_{t-1}; 0, \sqrt{1 - \alpha_t} \right)  \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \mathcal{N}\left( \bar{\boldsymbol{\epsilon}}_{t-2}; 0, \sqrt{1 - \alpha_t \alpha_{t-1}} \right)  & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

(*) Recall that when we merge two Gaussians with different variance, $\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$
 and $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$, the new distribution is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Here the merged standard deviation is $\sqrt{\alpha_t (1-\alpha_{t-1}) + (1 - \alpha_t)} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.

PS。这里还有一点指的注意的，$\mathbf{x}_t$ 是个被加噪声的中间样本，他服从两个事情，

一个是 $\mathbf{x}_t= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$，这个是从扩散的角度来说的，样本会怎么变化。

另外一个是它对应的概率 $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$，这是另外一个东西，是指得到这个样本的先验概率是怎样的，概率密度是多少。

这两个是相辅相成的关系。

{{< admonition type=quote title="Connection with stochastic gradient Langevin dynamics" >}}
$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
{{< /admonition >}}

Langevin dynamics（朗之万动力学）是物理学中用于模拟分子运动的统计方法。它描述了粒子在势能场中运动时受到的随机扰动（比如热噪声），因此常用于建模复杂系统的随机行为。

Stochastic Gradient Langevin Dynamics（SGLD，随机梯度朗之万动力学）是将 Langevin 动力学与机器学习中的随机梯度下降（SGD）结合起来的一种采样方法。
它的目标是从某个概率分布 \( p(x) \) 中采样，而不需要知道这个分布的具体形式，只需要知道它的梯度。

上面的采样公式是一个迭代式，他的含义是：“在梯度方向上前进一点，同时加入一些随机扰动，使得最终的样本分布逼近目标分布 \( p(x) \)。” 相关符号含义：

- \( \mathbf{x}_t \)：第 \( t \) 步的样本
- \( \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \): 漂移项，根据目标分布的梯度移动，类似受力牵引。也可以类比为扩散中的 $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$。
  - \( \delta / 2 \): 步长，控制每次更新的幅度
  - \( p(x) \)：目标分布的概率密度函数
  - \( \log p(x) \)：对数概率密度，便于计算和优化
  - \( \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \)：对数概率密度的梯度，也叫 score function，表示当前点的“上升方向”
- \( \sqrt{\delta} \boldsymbol{\epsilon}_t \): 扩散项，像布朗运动的分子碰撞。可以类比为扩散中的 $\sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}$。
  - \( \sqrt{\delta} \)：步长（step size），控制每次更新的幅度
  - \( \epsilon_t \sim \mathcal{N}(0, I) \)：标准正态分布的随机噪声，加入随机性以避免陷入局部最优

注意：这里提到的 $p(\cdot)$ 是一个通用的目标分布，可以是任何我们希望采样的分布。他和我们在diffusion中见到的$q(\cdot)$ 和 $p_\theta(\cdot)$ 是不同的。

对于diffusion场景，如果我们想要生成更真实的样本，则有

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其中，$q(\cdot)$ 衡量样本的真实性，每一轮迭代 $\mathbf{x}_t$ 都能比 $\mathbf{x}_{t-1}$ 更真实。同时还有 $\boldsymbol{\epsilon}_t$ 避免生成样本陷入局部最优。

### Reverse diffusion process

{{< admonition type=quote title="Reverse diffusion process也是高斯分布的" >}}
Note that if \(\beta_t\) is small enough, \(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)\) will also be Gaussian.
{{< /admonition >}}

让我们再来回顾下前向单步扩散公式：

$$
\begin{align}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\
\mathbf{x}_t &= \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}
\end{align}
$$

当 \(\beta_t\) 很小，意味着每一步加入的噪声很少，那么：

- \(\mathbf{x}_t\) 与 \(\mathbf{x}_{t-1}\) 的关系非常接近线性变换加微小扰动；
- 高斯分布线性变换仍然保持高斯形式。
- 这使得反向条件分布也可以近似为高斯分布，所以我们通常用高斯来建模反向过程。

{{< admonition type=quote title="建模似然" >}}
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
{{< /admonition >}}

上面两个公式对应了整体的，和单步的Reverse diffusion process的似然公式。即我们准备建立的神经网络的形式。

由于我们把reverse diffusion process建模为了高斯分布，
因此其可学习的参数就是高斯的均值和方差，$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$.

让我们看下diffusion模型训练推理中涉及到的四个重要的分布。
forward diffusion 生成噪声；
后验产生神经网络的训练目标，即每一步reverse diffusion神经网络要学什么；
reverse diffusion 似然就是真正用于拟合后验的。
根据先验，生成纯符合标准正态分布的噪声，交给拟合的reverse diffusion去噪，生成图像。

| 分布                                                  | 作用                                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| $q(\mathbf{x}_t \mid \mathbf{x}_0)$                   | **forward diffusion closed-form**，直接从数据加噪得到训练样本          |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ | **reverse diffusion 后验**，用于定义训练目标，约等于golden truth       |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$        | **reverse diffusion 似然**，通过训练模型去拟合上面的后验               |
| $q(\mathbf{x}_T)$                                     | **先验**，固定为高斯分布 $\mathcal{N}(0, I)$，推理时直接采样作为起点。 |

{{< admonition type=quote title="建模后验" >}}
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$
{{< /admonition >}}

仍然由于我们把 reverse diffusion process建模为了高斯分布，所以我们可以先定义后验的公式为上面的形式。
那么接下来问题就转换为了如何凑出这个 ${\tilde{\boldsymbol{\mu}}_t}(\mathbf{x}_t, \mathbf{x}_0)$ and $\tilde{\beta}_t$ 。
我们后面会用两个步骤将他们推导出来，他们的具体形式是：

$$
\begin{align}
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
\end{align}
$$

{{< admonition type=quote title="后验推导步骤一：按bayes公式和Gaussian公式展开" >}}
$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$
where $C(\mathbf{x}_t, \mathbf{x}_0)$ is some function not involving $\mathbf{x}_{t-1}$ and details are omitted...recall that $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
{{< /admonition >}}

我们首先可以利用贝叶斯公式把后验的计算变为先验计算，即forward diffusion的计算，这样我们可以利用之前推导出来的继续往下推；
其次，中间可以将概率展开为高斯概率密度函数间的计算，这是为了凑出新的高斯概率密度形式。
让我们根据这两个思路把这里的推理步骤写完善点：

$$
\begin{aligned}
&q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \\
%
&= \frac{ q(\mathbf{x}_{t-1}, \mathbf{x}_t \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
%
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
%
&= \mathcal{N}(\mathbf{x}_t; \sqrt{{\alpha}_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
\frac{ \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1 - \bar{\alpha}_{t-1})\mathbf{I}) }
{ \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) } \\
%
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big)
\quad\text{;where}\quad (*) \\
%
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
%
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 \color{blue}{- (\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\quad\text{;where}\quad (**)
\end{aligned}
$$

(*) 根据高斯概率密度函数我们可以对其展开，同时进行线性简化。
线性简化只是写法上的一个优化，因为不影响我们后续凑出来新的高斯分布。只要凑出来新的高斯分布形式上和当前长得差不多就没问题。

$$
\begin{align}
p(x)
& = \mathcal{N}(x; \mu, \sigma^2) \\
& = \frac{1}{\sqrt{2\pi\sigma^2}} \; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \\
& \propto \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\end{align}
$$

(**) $C(\mathbf{x}_t,\mathbf{x}_0)$ 里不包含 $\mathbf{x}_{t-1}$，
并且由于我们本身计算的就是$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$，其中$\mathbf{x}_{t}$ 和$\mathbf{x}_{0}$是已知的。
因此$C(\mathbf{x}_t,\mathbf{x}_0)$ 里不包含 $\mathbf{x}_{t-1}$只是一个常数项。
又因为它是在指数的位置，因此可以提出来单独作为系数。
最后，当我们把它拿出来计算最终的loss时，由于是在KL散度里的，它一定在log里，那么就可以提出来作为单独的常数项。
常数项求梯度是0，因此这里可以忽略掉。

{{< admonition type=quote title="后验推导步骤二：凑出新的高斯分布" >}}
$$\begin{aligned}
\tilde{\beta}_t
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$
{{< /admonition >}}

由于，

$$
\mathcal{N}(x; \mu, \sigma^2)
\propto \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
= \exp\!\left( -\frac{1}{2} (\color{red}{\frac{1}{\sigma^2}x^2} \color{blue}{- \frac{2\mu}{\sigma^2}x} \color{black}{+ \frac{\mu^2}{\sigma^2})} \right)
$$

再根据之前的计算：

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \propto \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 \color{blue}{- (\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
$$

我们可以有:

$$
\begin{aligned}
\frac{1}{\sigma^2}
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \;
\color{black}{\triangleq \tilde{\beta}_t} \\
%
\mu
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
&\triangleq \tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) \quad\text{or}\quad \tilde{\boldsymbol{\mu}}_t \\
\end{aligned}
$$

此时：

$$
\begin{align}
& \quad q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \\
%
& \propto \exp\Big( -\frac{1}{2} \big( \color{red}{\frac{1}{\tilde{\beta}_t}} \mathbf{x}_{t-1}^2 \color{blue}{- \frac{2\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t}} \mathbf{x}_{t-1} \color{black}{ + \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} + C(\mathbf{x}_t, \mathbf{x}_0) - \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} \big) \Big)} \\
%
& = \exp\Big( -\frac{1}{2} \big( \color{red}{\frac{1}{\tilde{\beta}_t}} \mathbf{x}_{t-1}^2 \color{blue}{- \frac{2\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t}} \mathbf{x}_{t-1} \color{black}{ + \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} \big) \Big)}
\cdot
\exp\Big( -\frac{1}{2} \big( C(\mathbf{x}_t, \mathbf{x}_0) - \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} \big) \Big) \\
%
& \propto \exp\Big( -\frac{1}{2} \big( \color{red}{\frac{1}{\tilde{\beta}_t}} \mathbf{x}_{t-1}^2 \color{blue}{- \frac{2\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t}} \mathbf{x}_{t-1} \color{black}{ + \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} \big) \Big)}
\quad\text{;where}\quad (*) \\
%
& = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}_t}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
\end{align}
$$

得证。

(*) 这里可以忽略系数项，就是因为前面所说的：系数项最终会转换为loss中的常数项，常数于求导无意义，因此可以忽略。

{{< admonition type=quote title="化简后验" >}}
Thanks to the nice property, we can represent $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$ and plug it into the above equation and obtain:

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$
{{< /admonition >}}

这一步的意义是，让计算完全依赖于噪声，而不依赖于真实数据，这样可以直接从任意噪声中恢复出真实数据。

其中提到的nice property就是closed form的前向扩散表达式：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$

让我们把化简步骤写的更完整些：

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
%
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
%
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_t / \alpha_t)}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_t / \alpha_t} (1-\alpha_t)}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
%
&= \frac{\alpha_t(1 - \bar{\alpha}_t / \alpha_t)}{\sqrt{\alpha_t} (1 - \bar{\alpha}_t)} \mathbf{x}_t + \frac{1-\alpha_t}{\sqrt{\alpha_t} (1 - \bar{\alpha}_t)} (\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
%
&= \frac{1}{\sqrt{\alpha_t}} \left( \frac{\alpha_t - \bar{\alpha}_t}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{1-\alpha_t}{1 - \bar{\alpha}_t} \mathbf{x}_t - \frac{1-\alpha_t}{1 - \bar{\alpha}_t} \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t \right) \\
%
&= \frac{1}{\sqrt{\alpha_t}} \left( \frac{\alpha_t - \bar{\alpha}_t + 1-\alpha_t}{1 - \bar{\alpha}_t} \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \right) \\
%
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$

得证。

{{< admonition type=quote title="varational lower bound：从零推导">}}
$$
\begin{aligned}
\mathord{-} \log p_\theta(\mathbf{x}_0)
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) & \small{\text{; KL is non-negative}}\\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
\end{aligned}
$$
{{< /admonition >}}

刚刚已经后验的公式都推证完了，这样我们就有了golden truth 了，即神经网络要模仿的对象。那么如何建模golden truth和模型之间的关联呢。

这就要用到变分推断（Variational Inference）了，或者叫做ELBO（Evidence Lower Bound）。可以把VLB理解为目标函数，Loss函数。总之就是我们期望VLB小一些，这样我们训练的模型就更容易产生真实的图像。

这里可以大致了解下 variational lower bound是什么：

1. 对数边际似然, $\log p_\theta(\mathbf{x}_0)$ 是我们期望模型的，即希望模型更有概率生成真实的图片
2. 然而他很难直接算，因为无法对潜变量分布中的所有情况都进行直接积分计算。
3. 所以要找替代的优化下界，优化该下界就相当于优化对数边际似然
4. 如果想要完全了解相关概念，强烈建议阅读 Lilian Weng 的另一篇文章 From Autoencoder to Beta-VAE [^lilian_ae] 中的 [章节 VAE: Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder)。

$$
\begin{aligned}
\mathord{-} \log p_\theta(\mathbf{x}_0)
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) & \small{\text{; KL is non-negative}}\\
&= - \log p_\theta(\mathbf{x}_0) + \sum_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0)} \Big] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= - \log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\end{aligned}
$$

到这里，我们已经将难以计算的对数边际似然转换为了含有 $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ 和 $p_\theta(\mathbf{x}_{0:T})$.前者通过我们先前推导的后验公式可以直接计算，而后者就是由神经网络定义的，可以分解为每一步的条件概率。

为了训练，我们不能只在一个图片 $\mathbf{x}_0$ 上跑模型，还需要通过蒙特卡洛的方式进行采样，因此我们有：

$$
\begin{align}- \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
&\leq - \mathbb{E}_{q(\mathbf{x}_0)} \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= - \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0), \, \mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= - \mathbb{E}_{\mathbf{x}_{0, 1, ..., T}\sim q(\mathbf{x}_0) q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&\triangleq L_\text{VLB}
\end{align}
$$

得证。

{{< admonition type=quote title="varational lower bound: 用Jensen不等式">}}
$$
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
$$
{{< /admonition >}}

这里推导主要用到了概率论中的边际化以及Jensen 不等式。了解这两个之后就易证了。

对$\mathbf{x}_{1:T}$边际化即:

$$
\begin{align}
p_\theta(\mathbf{x}_0)
&= \int \Big[ p_\theta(\mathbf{x}_0 | \mathbf{x}_{1:T}) p_\theta(\mathbf{x}_{1:T}) \Big] d\mathbf{x}_{1:T} \\
&= \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
\end{align}
$$

Jensen 不等式 [^wiki_jensen], 是指设 \( \phi(\cdot) \) 是一个concave function [^wiki_concave]，\( X \) 是一个可积的随机变量，则有不等式:

\[
\phi\left( \mathbb{E}[X] \right) \geq \mathbb{E}\left[ \phi(X) \right]
\]

例如，$-log(\cdot)$ is a concave function。

{{< admonition type="quote" title="VLB 展开" >}}
$$
\begin{aligned}
L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$
{{< /admonition>}}

为什么要把简洁的VLB展开为最后那个复杂的由多个KL散度组成的复杂公式呢?

回忆下我们之前推证得到的forward diffusion process的closed form表达式，以及建模reverse diffusion process时候得到的后验和似然：

$$
\begin{align}
%
q(\mathbf{x}_t \vert \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \\
%
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\
%
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
&= \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{align}
$$

而刚刚我们获得的 $L_\text{VLB} = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big]$ 并没有用到这些已有的公式，因此不能直接计算.
而且，我们也不希望神经网络拟合的时候直接一步到位，还是希望他能模拟逐步去噪的过程。
（ 当然后面也出现了可以一步到位的Consistency Models [^song_consistency]，后面会讲到。）

因此，展开VLB的目的是将训练目标从一个难以直接优化的对数似然函数，转化为一组可计算的 KL 散度项与重构项，从而指导神经网络学习如何从噪声中逐步恢复原始数据。

让我们把VLB的展开过程给写的更完整些：

$$
\begin{aligned}
& L_\text{VLB} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big]
\quad\text{; 利用马尔可夫性质展开联合概率分布为递推式} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big]
\quad\text{; } -\log p_\theta(\mathbf{x}_T) \text{是常数，因此可以单独提出来} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; } \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \text{进行了特殊的建模，后面会提到} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; 根据贝叶斯公式把} q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \text{转换为后验公式和前向closed form公式的组合} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; 根据log函数的计算规律进行拆分} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; 根据log函数的计算规律进行重组} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big]
\quad\text{; 根据log函数的计算规律进行重组} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} +
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} -
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \\
%
&= \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_T)\sim q(\mathbf{x}_0, \mathbf{x}_T)} \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} +
\sum_{t=2}^T \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_{t-1}, \mathbf{x}_t)\sim q(\mathbf{x}_0, \mathbf{x}_{t-1}, \mathbf{x}_t)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} -
\mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1)\sim q(\mathbf{x}_0, \mathbf{x}_1)} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; 按全期望公式简化} \\
%
&= \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0)} \left[ \mathbb{E}_{\mathbf{x}_T \sim q(\mathbf{x}_T | \mathbf{x}_0)} \log \frac{q(\mathbf{x}_T | \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} \right] +
\sum_{t=2}^T \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_t) \sim q(\mathbf{x}_0, \mathbf{x}_t)} \left[ \mathbb{E}_{\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)} \right] -
\mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1)\sim q(\mathbf{x}_0, \mathbf{x}_1)} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; 展开为条件期望形式} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ \mathbb{E}_{\mathbf{x}_T \sim q(\mathbf{x}_T | \mathbf{x}_0)} \log \frac{q(\mathbf{x}_T | \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} \right] +
\sum_{t=2}^T \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ \mathbb{E}_{\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)} \right] -
\mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; 根据全期望公式补齐} \\
%
&=
\underbrace{
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \right]
 }_{L_T, \, \text{Prior Matching Term}} +
\sum_{t=2}^T
\underbrace{
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)) \right]
}_{L_t, \, \text{Denoising Matching Term}}
\underbrace{ -
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
}_{L_0, \, \text{Reconstruction Term}}
\quad\text{; 改写为KL散度形式}
\end{aligned}
$$

{{% admonition type="quote" title="VLB中三项的参数化" open=true %}}
$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$

... $L_T$ is constant and can be ignored during training because $q$ has no learnable parameters and $\mathbf{x}_T$ is a Gaussian noise. [Ho et al. 2020](https://arxiv.org/abs/2006.11239) models $L_t$ using a separate discrete decoder derived from $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$.
{{% /admonition %}}

为了书写方便，实际上这里省略掉了期望的符号。并且原公式中第二项的取值范围是$2 \leq t \leq T$，而不是 $1 \leq t \leq T-1$. 因此我们改为严格书写应当是：

$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0, \, \text{where } \\
L_T &= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)) \text{ for }2 \leq t \leq T \\
L_0 &= - \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$

下面会分别讲下三项如何参与到神经网络的训练中：

⭐ $L_T = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))$

如原博文所述，这只是一个常数，反向传播梯度为0，因此可以忽略掉，不参与优化。

⭐ $L_t = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))$

这一项是最重要的，会在接下来细讲，这里暂时略过。

⭐ $L_0 = - \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)$

为了计算的方便，这一项被近似为 t=1 时的 denoising matching term：

\[
L_0 \approx \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_{\text{KL}}(q(\mathbf{x}_0 \vert \mathbf{x}_1) \parallel p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1))
\]

### Parameterization of $L_t$ for Training Loss

{{< admonition type="quote" title="Parameterization of $L_t$" open=true >}}
... We would like to train $\boldsymbol{\mu}_\theta$ to predict $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$. ...

$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$

The loss term $L_t$ is parameterized to minimize the difference from $\tilde{\boldsymbol{\mu}}$:

$$
\begin{aligned}
L_t
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
{{< /admonition >}}

推导时仍然要用到之前推证得到的forward diffusion process的closed form表达式，以及建模reverse diffusion process时候得到的后验和似然：

$$
\begin{align}
%
q(\mathbf{x}_t \vert \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \\
%
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\
%
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
&= \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{align}
$$

除此之外，我们还需要知道两个多元高斯分布的KL散度计算公式（read this post [^gupta_gaussian_kl] to know how to derive it）. 给定两个高斯分布，$\mathcal{N}(\boldsymbol{\mu_q},\,\Sigma_q)$ and $\mathcal{N}(\boldsymbol{\mu_p},\,\Sigma_p)$, 数据维度式是$k$维，则他们的KL散度为：

$$D_{KL}(q||p) = \frac{1}{2}\left[\log\frac{|\Sigma_p|}{|\Sigma_q|} - k + (\boldsymbol{\mu_q}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\boldsymbol{\mu_q}-\boldsymbol{\mu_p}) + tr\left\{\Sigma_p^{-1}\Sigma_q\right\}\right]$$

再者，我们要知道一个重要的假设，DDPM原论文[^ho_ddpm] 假定 $\Sigma_\theta$ 是常量超参数。假如$\Sigma_\theta$需要学习，则会导致

1. 训练时需要对方差做梯度更新，可能导致发散。
2. 对于每个时间步 $t$，$\Sigma_\theta(\mathbf{x}_t, t)$ 是高维的（例如图像像素维），训练量非常大。
3. 经验表明，如果只预测均值，模型已经能很好地学习反向过程，生成的样本质量也很高。

根据这些信息，让我们写一下完整的 $L_t$的推导：

$$
\begin{aligned}
& L_t \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)) \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL} \left(
  \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
  \parallel
  \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\right) \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL} \left(
  \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_{t}, \tilde{\beta}_t \mathbf{I})
  \parallel
  \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta,t}, \boldsymbol{\Sigma}_{\theta,t})
\right)
\quad\text{; 简写上式}\\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \left[
  \log \frac{|\boldsymbol{\Sigma}_{\theta,t}|}{|\tilde{\beta}_t \mathbf{I}|} -
  k +
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})^T \boldsymbol{\Sigma}_{\theta,t}^{-1} (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t}) +
  \text{tr}(\boldsymbol{\Sigma}_{\theta,t}^{-1} \tilde{\beta}_t \mathbf{I})
\right]
\quad\text{; 高斯分布的KL散度展开}\\
%
&\approx \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \left[
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})^T
  \boldsymbol{\Sigma}_{\theta,t}^{-1}
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})
\right]
\quad\text{; 忽略常量} \Sigma_\theta(\mathbf{x}_t, t), \tilde{\beta}_t\mathbf{I}, k \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \Big[
  \frac{1}{\| \boldsymbol{\Sigma}_{\theta,t} \|^2_2}
  \| \tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\theta,t} \|^2
\Big]
\quad\text{; }\boldsymbol{\Sigma}_{\theta,t}\text{; 是对角矩阵，所以可以单独提到前面} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_{\theta,t} \|^2_2} \| \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big) - \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big) \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_{0})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
&\propto \mathbb{E}_{t \sim [1, T], \mathbf{x}_0\sim q(\mathbf{x}_{0}), \boldsymbol{\epsilon}_t \sim\mathcal{N}(0, I) } \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
\end{aligned}
$$

其中，

- $t, \mathbf{x}_0, \boldsymbol{\epsilon}_t$ 都是通过蒙特卡洛采样来的，因此是模型的输入；
- $\alpha_t, \bar{\alpha}_t$ 则都依赖于 Noise variance schedule parameter $\beta_t$，这个值是超参数，是一个调度数列，是一个值得设计的地方；
- $\boldsymbol{\Sigma}_{\theta,t}$ 是模型在时间步 $t$ 的reverse diffusion distribution 的协方差，可以设计为constant 也可以被设计为learnable parameters。
- $\boldsymbol{\epsilon}_\theta$ 神经网络必然要学习的项，是learnable parameters，没什么好说的。

接下来会逐一讲解三个参数相关的设计策略：

1. $\beta_t$和$\boldsymbol{\Sigma}_{\theta,t}$为简单常数的 $L_t^\text{simple}$ 策略
2. $\beta_t$ 为non-trivial schedule的策略
3. $\boldsymbol{\Sigma}_{\theta,t}$为可学习的参数策略

{{% admonition type="quote" title="$L_t$的化简" open=true %}}
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
{{% /admonition %}}

这里主要解释了两件事情：

1. 训练时的蒙特卡洛采样，是对任意真实图片样本 $\mathbf{x}_0$, 任意difussion步骤 $t$ 以及任意噪声 $\boldsymbol{\epsilon}_t$ 进行采样。
2. 训练时忽略掉了含有$\boldsymbol{\Sigma}_{\theta,t}$的权重系数，因为在原论文 [^ho_ddpm] 中这个被设置为了常数。

PS. 同时这个简化的公式还给我们观察 $L_t$ 另外的一个视角，即他可以不是KL散度 loss，而是一个MSE loss。

{{% admonition type="quote" title="DDPM Algorithm的训练和采样" open=true %}}
{{< media
src="DDPM_Algo.png"
caption="The training and sampling algorithms in DDPM (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))"
>}}
{{% /admonition %}}

- Algorithm 1: Training（训练阶段）：训练就是教模型去“猜出某一时刻图像里的噪声”。这里要感谢前面推出来的各种close form的公式，我们无需逐步的进行计算。
- Algorithm 2: Sampling（采样阶段）：采样就是从随机噪声开始，逐步去噪生成新图像。所以他要走完全部的difussion process，因此生成速度会很慢。

PS. 在图像生成领域，采样指的就是拿训练好的模型进行推理。

{{% admonition type="quote" title="Connection with noise-conditioned score networks (NCSN)" open=true %}}

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) using gradients of the data distribution estimated with score matching.

...

Given a Gaussian distribution $\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 \mathbf{I})$, we can write the derivative of the logarithm of its density function as $\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$. Recall that $q(\mathbf{x}_t \vert \mathbf{x}_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$ and therefore,

$$
\mathbf{s}_\theta(\mathbf{x}_t, t)
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$
{{% /admonition %}}

让我们回顾下之前提到过Stochastic Gradient Langevin Dynamics采样公式：

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其中，$p(\cdot)$ 是用于衡量生成样本真实性的。如原文所示$p(\cdot)$被定义为

$$
\begin{align}
p(\mathbf{x}_t) &\triangleq q(\mathbf{x}_t \vert \mathbf{x}_0) \\
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{align}
$$

代入进去，我们则拥有更完整的采样公式为：

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

由于这个close form公式的存在，这意味着，我们能不断地迭代这个公式，来获得一个真实性更高的样本。所以现在问题的关键是看下这其中的各项是否能得到。

因此，我们只要做出神经网络拟合这个采样公式即可。在这个式子中，$\mathbf{x}_{t-1}$ 是我们输入的值；$\frac{\delta}{2}$ 是常量系数；$\sqrt{\delta} \boldsymbol{\epsilon}_t$ 是随机采样的；只有中间的 $\nabla_\mathbf{x} \log q(\mathbf{x}_{t-1})$ 是关键的神经网络需要拟合的。

---

在了解如何建模中间项之前，我们先推导一下derivative of the logarithm of Gaussian density function（这个是一个单独的推导，也可以先跳过）:

$$
\begin{align}
\nabla_{\mathbf{x}}\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})
&= \nabla_{\mathbf{x}}\log \Big[ \frac{1}{\sqrt{(2\pi)^D \sigma^2}} \cdot \exp\Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) \Big] \\
&= \nabla_{\mathbf{x}} \Big[ \log\frac{1}{\sqrt{(2\pi)^D \sigma^2}} + \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) \Big] \\
&= \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) \\
&= - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} \quad\text{; where } \mathbf{x} - \boldsymbol{\mu} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \\
&= - \frac{(\mathbf{x} - \boldsymbol{\mu}) / \sigma}{\sigma} \quad\text{; where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&\triangleq - \frac{\boldsymbol{\epsilon}}{\sigma}
\end{align}
$$

所以，结论即：

$$
\boxed{
  \nabla_{\mathbf{x}}\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})
  = - \frac{\boldsymbol{\epsilon}}{\sigma}
}
\quad\text{; where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

---

有了上面的这个公式，我们来推导下我们现在的golden truth：

$$
\begin{align}
\nabla_\mathbf{x} \log q(\mathbf{x}_{t-1})
&= \nabla_\mathbf{x} \log \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1 - \bar{\alpha}_{t-1})\mathbf{I}) \\
&= - \frac{\boldsymbol{\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0}}{\sqrt{1 - \bar{\alpha}_{t-1}}}
\end{align}
$$

这里面有两个变量，时间步 $t$ 和真实样本 $\mathbf{x}_0$。所以我们有许多 golden truth 组成的期望构成监督信号：

$$
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\mathbf{x}_t - \boldsymbol{\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0}}{1 - \bar{\alpha}_{t-1}} \right) =
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_{t-1}}} \right)
$$

我们据此可以定义神经网络中的权重即为：

$$
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_{t-1}}} \right)
$$

- 训练时，我们sample 很多组 $(t, \mathbf{x}_0)$ 来使得神经网络拟合golden truth
- 推理时，我们将已有的 $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) \approx \mathbf{s}_\theta(\mathbf{x}_t, t) = - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$ 代入到Stochastic Gradient Langevin Dynamics采样公式，得到：

  $$
  \mathbf{x}_t =
  \mathbf{x}_{t-1} -
  \frac{\delta \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{2\sqrt{1 - \bar{\alpha}_t}} +
  \sqrt{\delta} \boldsymbol{\epsilon}_t
  ,\quad\text{where }
  \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
  $$

  此时，便可以不依赖真实样本 $\mathbf{x}_0$ 来迭代式采样生成新图像。

### Parameterization of $\beta_t$

{{% admonition type="quote" title="从 trivial 到 non-trivial $\beta_t$ scheduling" open=true %}}
The forward variances are set to be a sequence of linearly increasing constants in [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), from $\beta_1=10^{-4}$ to $\beta_T=0.02$. They are relatively small compared to the normalized image pixel values between $[-1, 1]$. Diffusion models in their experiments showed high-quality samples but still could not achieve competitive model log-likelihood as other generative models.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed several improvement techniques to help diffusion models to obtain lower NLL. One of the improvements is to use a cosine-based variance schedule. The choice of the scheduling function can be arbitrary, as long as it provides a near-linear drop in the middle of the training process and subtle changes around $t=0$ and $t=T$.

$$ \beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2 $$

where the small offset $s$ is to prevent $\beta_t$ from being too small when close to $t=0$.

{{< media
src="Linear_and_Cosine_Scheduling.png"
caption="Comparison of linear and cosine-based scheduling of during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))"
>}}
{{% /admonition %}}

让我们把DDPM [^ho_ddpm] 原论文中的调度方法，和Improved DDPM [^nichol_improved_ddpm] 中的调度公式给完整写出来：

DDPM [^ho_ddpm]的 **linear variance schedule**是：

$$
\begin{align}
\beta_t &= \beta_{\text{min}} + \frac{t - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}}) \\
\alpha_t &= 1 - \beta_t = 1 - \left(\beta_{\text{min}} + \frac{t - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})\right) \\
\bar{\alpha}_t &= \prod_{k=1}^{t} \alpha_t = \prod_{k=1}^{t} \left(1 - \beta_{\text{min}} - \frac{k - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})\right)
\end{align}
$$

其中：

- \( \beta_{\text{min}} \) 和 \( \beta_{\text{max}} \) 是预设的最小和最大噪声值（例如 0.0001 和 0.02）
- \( T \) 是总的扩散步数（例如 1000）

Improved DDPM [^nichol_improved_ddpm] 的 **cosine-based variance schedule**是：

$$
\begin{align}
\bar{\alpha}_t &= \frac{f(t)}{f(0)} \quad \text{; where } f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2, s=0.008 \\
\alpha_t &= \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = \frac{f(t)}{f(t-1)} \\
\beta_t &= \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999)
\end{align}
$$

其中：

- 这个schedule先定义 $\bar{\alpha}_t$，再定义 $\alpha_t$ 和 $\beta_t$。
- \( s \) 是一个小的偏移量，用于防止 \( \beta_t \) 在接近 \( t=0 \) 时变得过小。Improved DDPM [^nichol_improved_ddpm]认为 having tiny amounts of noise at the beginning of the process made it hard for the network to predict accurately enough.
- clip 函数用于确保 \( \beta_t \) 不超过 0.999，这避免了数值不稳定性。Improved DDPM [^nichol_improved_ddpm]认为这能够prevent singularities at the end of the diffusion process near $t = T$.

点击下面的图的右下角可以进入到交互式界面直观的感受两种scheduler。

<iframe src="https://www.desmos.com/calculator/sxftdp4sib?embed" width="100%" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

### Parameterization of reverse process variance $\boldsymbol{\Sigma}_\theta$

{{% admonition type="quote" title="从 unlearnable 到 learnable $\boldsymbol{\Sigma}_\theta$" open=true %}}
[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) chose to fix $\beta_t$ as constants instead of making them learnable and set $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$ , where $\sigma_t$ is not learned but set to $\beta_t$ or $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$. Because they found that learning a diagonal variance $\boldsymbol{\Sigma}_\theta$ leads to unstable training and poorer sample quality.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed to learn $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ as an interpolation between $\beta_t$ and $\tilde{\beta}_t$ by model predicting a mixing vector $\mathbf{v}$ :

$$ \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t) $$
{{% /admonition %}}

回忆下我们之前的reverse diffsion 中的后验和似然

$$
\begin{align}
%
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\
%
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
&= \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{align}
$$

其中，$\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$, $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$.

为了让训练更易收敛，在DDPM [^ho_ddpm] 原论文中，作者根据两个式子的形式上的一致性，将方差部分建模为常数，$\boldsymbol{\Sigma}_\theta \triangleq \tilde{\beta}_t \text{ or } \beta_t$。而只把$\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0)$ close form 中的噪声 $\boldsymbol{\epsilon}_t$当作可学习的参数。

为什么还可以建模为 forward diffusion中 noise schedule，$\beta_t$呢？OpenAI的Nichol 在Improved DDPM [^nichol_improved_ddpm]给出了解释，因为他们实际上相差很小，尤其是在diffusion process的后程。

{{< media
src="Ratio_vs_Diffusion_Step.png"
caption="The ratio for every diffusion step for diffusion processes of different lengths. ([source](https://proceedings.mlr.press/v139/nichol21a.html))"
>}}

并且他们还发现，对这里进行改进，使其变成learnable interpolation between $\beta_t$ and $\tilde{\beta}_t$ 能带来更好的结果。

$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)\mathbf{I}
$$

其中，

- $\mathbf{I}$ （单位矩阵），我额外添加的，以确保最终的 $\boldsymbol{\Sigma}_\theta$ 是一个对角协方差矩阵。
- $\beta_t$ （scalar） forward noise schedule
- $\tilde{\beta}_t$ （scalar） posterior variance
- $\mathbf{v}$ （vector） 模型输出的 mixing coefficient。这是模型要学习的。
- $\boldsymbol{\Sigma}_\theta$ （vector） 每个维度的预测方差（对角协方差）

{{% admonition type="quote" title="Integrate learnable $\boldsymbol{\Sigma}_\theta$ into 最终的loss" open=true %}}
However, the simple objective $L_\text{simple}$ does not depend on $\boldsymbol{\Sigma}_\theta$ . To add the dependency, they constructed a hybrid objective $L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}$ where $\lambda=0.001$ is small and stop gradient on $\boldsymbol{\mu}_\theta$ in the $L_\text{VLB}$ term such that $L_\text{VLB}$ only guides the learning of $\boldsymbol{\Sigma}_\theta$. Empirically they observed that $L_\text{VLB}$ is pretty challenging to optimize likely due to noisy gradients, so they proposed to use a time-averaging smoothed version of $L_\text{VLB}$ with importance sampling.
{{% /admonition %}}

这段话整体意思比较简单，主要是希望把原来的$L_\text{simple}$损失和带有learnable parameter $\boldsymbol{\Sigma}_\theta$的损失结合起来，联合优化。不过其中出现两个概念可以展开来讲下：**“noisy gradient”**，**“time-averaging smoothed version of $L_\text{VLB}$ with importance sampling”**。

**noisy gradient**是出自openai的论文 An Empirical Model of Large-Batch Training [^mc_candlish_grad_noise] 提出的一个概念。

在随机梯度下降（SGD）中，我们不是用整个数据集计算梯度，而是用一个小批量（mini-batch）。这会引入噪声，因为不同批次的梯度可能差异很大。**Gradient Noise Scale**就能衡量这种梯度的波动性。它的核心思想是：如果梯度在不同批次之间变化很大（噪声高），我们需要更大的批量来获得更稳定的更新。在简化假设下（如 Hessian 是单位矩阵的倍数），Gradient Noise Scale 可以表示为：

\[
B_{\text{simple}} = \frac{\text{tr}(\Sigma)}{\|G\|_2^2}
\]

其中：

- \(\text{tr}(\Sigma)\)：梯度协方差矩阵的迹。即所有参数梯度的方差之和。表示梯度的“波动性”。
- \(\|G\|_2^2\)：梯度的平方范数（global gradient norm）。即所有参数梯度的平方和。表示梯度的“平均强度”。

\(B_{\text{simple}}\) 表示：**梯度的噪声强度相对于其平均强度的比例**

- 如果 \(B_{\text{simple}}\) 很大，说明梯度噪声很强，建议使用更大的 batch size。
- 如果它很小，说明梯度稳定，可以用较小的 batch size，加快训练。

---

这段话中，“time-averaging smoothed version of $L_\text{VLB}$ with importance sampling”则具体指的是这个公式，Improved DDPM [^nichol_improved_ddpm]提出的新的损失函数设计：

$$L_{\text{vlb}} = \mathbb{E}_{t \sim p_t} \left[ \frac{L_t}{p_t} \right], \text{ where } p_t \propto \sqrt{\mathbb{E}[L_t^2]} \text{ and } \sum p_t = 1$$

这个公式中提到的[重要性采样技巧](#importance-sampling-trick)是机器学习中一种常见的技术，旨在对采样的改进来提高期望的计算效率。

让我们回忆下原始的DDPM的损失函数：

$$
L_t^\text{simple}
= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
$$

可以看到它的时间步 $t$ 服从均匀分布，则其概率密度函数为 $p'(t)=\frac{1}{T}$.

然而 Improved DDPM [^nichol_improved_ddpm] 指出，前几个时间步贡献了绝大多损失值，因而均匀采样是低效的。

{{< media
src="VLB_vs_Diffusion_Step.png"
caption="Terms of the VLB vs diffusion step. The first few terms contribute most to NLL. ([source](https://proceedings.mlr.press/v139/nichol21a.html))"
>}}

因此，我们可以用另外一个分布来优化采样，作者提出了一个新的分布 $p_t \propto \sqrt{\mathbb{E}[L_t^2]} \text{ where } \sum p_t = 1$，这个分布和损失值成正比，意味着我们期望损失值高的区域（小时间步的区域）被更多的采样到。利用重要性采样公式调整期望为：

$$
\begin{align}
L_{\text{vlb}}
&= \mathbb{E}_{t \sim p'(t)}[L_t] \\
&= \int L_t p'(t) dt \\
&= \int L_t \frac{p'(t)}{p(t)} p(t) dt \\
&= \mathbb{E}_{t \sim p(t)}\left[ L_t \frac{p'(t)}{p(t)} \right] \\
&= \mathbb{E}_{t \sim p(t)}\left[ \frac{1}{T} \frac{L_t}{p(t)} \right] \\
&= \frac{1}{T} \mathbb{E}_{t \sim p(t)}\left[ \frac{L_t}{p(t)} \right] \\
&\propto \boxed{\mathbb{E}_{t \sim p(t)} \left[ \frac{L_t}{p(t)} \right]}, \text{ where } p(t) \propto \sqrt{\mathbb{E}[L_t^2]} \text{ and } \sum p(t) = 1
\end{align}
$$

## Conditioned Generation

### Classifier Guided Diffusion

{{% admonition type="quote" title="Classifier Guided Diffusion采样公式" open=true %}}
To explicit incorporate class information into the diffusion process, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) trained a classifier $f_\phi(y \vert \mathbf{x}_t, t)$ on noisy image $\mathbf{x}_t$ and use gradients $\nabla_\mathbf{x} \log f_\phi(y \vert \mathbf{x}_t)$ to guide the diffusion sampling process toward the conditioning information $y$ (e.g. a target class label) by altering the noise prediction. Recall that $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ and we can write the score function for the joint distribution $q(\mathbf{x}_t, y)$ as following,

$$ \begin{aligned} \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y) &= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\ &\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\ &= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)) \end{aligned} $$
Thus, a new classifier-guided predictor $\bar{\boldsymbol{\epsilon}}_\theta$ would take the form as following,

$$ \bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) $$
To control the strength of the classifier guidance, we can add a weight $w$ to the delta part,

$$ \bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) $$
{{% /admonition %}}

整体想法即拿一个pre-trained diffusion model 和一个 pre-trained image classifier model，组合两个模型来做条件生成。利用langevin dynamics的技巧，如果我们拥有classifier，便能拿到类别$l$的梯度信息，有梯度信息便可以通过langevin dynamics来进行采样，通过逐步迭代拿到符合类别 $l$ 的图片：

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_{t-1}, y) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**博文公式中的推导思路**就是第一步先通过条件概率转换为diffusion part的梯度，和分类器 part的梯度。第二步将理论公式转换为含learnable参数的形式。第三步进行了并非必要的化简。

### Classifier-Free Guidance

{{% admonition type="quote" title="Classifier-Free Guidance 采样公式" open=true %}}
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$
{{% /admonition %}}

classifer guided diffusion 的优势是可以直接利用两个已经训练好的模型，无需其他操作。另外一个优势是，通过前面提到的$w$ 权重，可以控制“conditioin”的强度。

而对于 claasifer-free guidance，最简单的就是直接把condition信息训练进diffusion模型即可，但是这就失去condition强度控制的这个feature了。

所以另外一种思路是，如原博文所示，在同一个network骨架上训练带condition和不带condition的两种情况，只在condition输入上做区分。这样就可以通过“相减”来实现condition强度的控制。

让我们把采样公式给写的完整些：

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) + \nabla_{\mathbf{x}_t} \log p(y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) &\quad\text{(按照贝叶斯公式展开)}\\
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big)  &\quad\text{(按照classifier guided 方式展开)}\\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

{{% admonition type="quote" title="FID和IS" open=true %}}
Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).
{{% /admonition %}}

FID和IS是生成模型中重要的评估指标。

---

FID（Fréchet Inception Distance）[^heusel_fid] 衡量的是生成图像与真实图像在特征空间中的分布差异。它使用 Inception 网络提取图像特征，然后计算两个高维高斯分布之间的 Fréchet 距离。

\[
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
\]
其中：

- \( \mu_r, \Sigma_r \)：真实图像的均值和协方差
- \( \mu_g, \Sigma_g \)：生成图像的均值和协方差

FID 越低，表示生成图像与真实图像越接近；

---

IS（Inception Score）[^salimans_improve_gan] 衡量的是生成图像的“清晰度”和“多样性”。它使用 Inception 网络预测图像类别分布，然后计算预测分布的 KL 散度。

\[
\text{IS} = \exp\left( \mathbb{E}_{x \sim p_g} \left[ D_{\text{KL}}(p(y|x) \| p(y)) \right] \right)
\]

其中：

- \( p(y|x) \)：Inception 网络对生成图像的预测分布
- \( p(y) \)：所有生成图像的平均预测分布

IS 越高，表示图像清晰（预测分布熵低）且多样性高（平均分布熵高）

{{% admonition type="quote" title="无分类器引导比 Classifier 引导效果更好" open=true %}}
The guided diffusion model, GLIDE ([Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741)), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.
{{% /admonition %}}

GLIDE 是一种引导式扩散模型（guided diffusion model），由 Nichol、Dhariwal 和 Ramesh 等人在 2022 年提出。它尝试了两种图像生成的引导策略：

1. **CLIP guidance（CLIP 引导）**：利用 CLIP 模型的图文匹配能力来引导图像生成过程。
2. **Classifier-free guidance（无分类器引导）**：不依赖外部分类器，而是通过训练一个模型同时学习有条件和无条件的图像生成，从而实现引导。

GLIDE 的实验发现，**无分类器引导比 CLIP 引导效果更好**，即：

- 无分类器引导：模型自己学会怎么生成图像，不依赖外部判断。
- CLIP 引导：模型依赖 CLIP 的评分，但可能会“作弊”去骗过 CLIP。
- GLIDE 更偏好前者，因为它更自然、更稳健。

## Speed up Diffusion Models

### Fewer Sampling Steps & Distillation

{{% admonition type="quote" title="Naive Strided Sampling" open=true %}}
One simple way is to run a strided sampling schedule (Nichol & Dhariwal, 2021) by taking the sampling update every $\lceil T/S \rceil$ steps to reduce the process from $T$ to $S$ steps. The new sampling schedule for generation is $\{\tau_1, \dots, \tau_S\}$ where $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$ and $S < T$.
{{% /admonition %}}

下面是标准的DDPM采样公式：

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
= \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \quad, t\in[1,T]
$$

这里的naive strided sampling将采样公式改为了：

$$
p_\theta(\mathbf{x}_{\tau_{k-1}} \vert \mathbf{x}_{\tau_k})
= \mathcal{N}(\mathbf{x}_{\tau_{k-1}}; \boldsymbol{\mu}_\theta(\mathbf{x}_{\tau_k}, \tau_k), \boldsymbol{\Sigma}_\theta(\mathbf{x}_{\tau_k}, \tau_k)) \quad, k\in[1,S]
$$

当然，跨多步时，真实的均值和方差应该通过多步高斯组合公式推导出来。因此这个strided sampling的策略只是一个粗糙的加速。

{{% admonition type="quote" title="denoising diffusion implicit model" open=true %}}
For another approach, let’s rewrite $q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ to be parameterized by a desired standard deviation $\sigma_t$ according to the [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice):

$$ \begin{aligned} \mathbf{x}_{t-1} &= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} & \\ &= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} & \\ &= \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t) + \sigma_t\boldsymbol{\epsilon} \\ q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I}) \end{aligned} $$
where the model $\epsilon^{(t)}_\theta(.)$ predicts the $\epsilon_t$ from $\mathbf{x}_t$.

Recall that in $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$, therefore we have:

$$ \tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t $$
Let $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$ such that we can adjust $\eta \in \mathbb{R}^+$ as a hyperparameter to control the sampling stochasticity. The special case of $\eta = 0$ makes the sampling process deterministic. Such a model is named the denoising diffusion implicit model (DDIM; [Song et al., 2020](https://arxiv.org/abs/2010.02502)). DDIM has the same marginal noise distribution but deterministically maps noise back to the original data samples.

During generation, we don’t have to follow the whole chain $t=1,\dots,T$, but rather a subset of steps. Let’s denote $s < t$ as two steps in this accelerated trajectory. The DDIM update step is:

$$ q_{\sigma, s < t}(\mathbf{x}_s \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_s; \sqrt{\bar{\alpha}_s} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_s - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I}) $$
While all the models are trained with $T=1000$ diffusion steps in the experiments, they observed that DDIM ($\eta=0$) can produce the best quality samples when $S$ is small, while DDPM ($\eta=1$) performs much worse on small $S$. DDPM does perform better when we can afford to run the full reverse Markov diffusion steps ($S=T=1000$). With DDIM, it is possible to train the diffusion model up to any arbitrary number of forward steps but only sample from a subset of steps in the generative process.
{{% /admonition %}}

Naive Strided Sampling 在数学上缺少严谨性，DDIM [^song_ddim]弥补了这个问题，同时DDIM将原来的随机采样过程转变为随机和确定性相结合的采样过程，更加灵活。

此外，DDIM和DDPM的主要区别在于如何建模似然，来预测后验。

$$
\begin{align}
\text{DDPM后验：} & q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\
\text{DDPM似然：} & p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \\
\text{DDIM似然：} & q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{align}
$$

其中，

- $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$
- $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$。当$\eta=0$时，方差为0，因此采样过程是确定性的（DDIM）；当$\eta=1$时，采样过程是随机的（DDPM）。

注意：DDPM和DDIM分属不同的论文，因此可能有符号上的冲突，这里遵循原文，并没有将他们做统一。

公式的推导的思路是，从forward diffusion process closed form公式出发，尝试分出去一个$\sigma_t^2$ 到方差位置，利用下面这样的技巧：

\[
\text{Var}(\boldsymbol{\epsilon}_{t-1}) = a^2 \cdot \text{Var}(\boldsymbol{\epsilon}_t) + b^2 \cdot \text{Var}(\boldsymbol{\epsilon}) = a^2 + b^2 \quad \text{; where }\epsilon_t \text{ and }\epsilon \text{ are independent}
\]

{{% admonition type="quote" title="Progressive Distillation" open=true %}}
{{< media
src="Progressive_Distillation_Algo.png"
caption="Comparison of Algorithm 1 (diffusion model training) and Algorithm 2 (progressive distillation) side-by-side, where the relative changes in progressive distillation are highlighted in green. (Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))"
>}}
{{% /admonition %}}

我们可以对右侧Progressive Distillation的算法进行一些更详细的解释。

首先是两层循环：外层循环控制训练 K 个不同的 student diffusion model，每个都能用更少的 step 采样生成图像；内层循环则是训练当前 student model。

在内层循环中：

1. $\text{Cat}[1,2,...,N]$指的是 Catgorical 分布，从1到N均匀采样一个整数 $t$，表示当前的时间步。
2. $t' = t - \frac{0.5}{N}, \quad t'' = t - \frac{1}{N}$ 指的是将时间步 $t$ 拆分成两个更小的步长，分别是 $t'$ 和 $t''$。
3. $z_{t'} = \alpha_{t'} \tilde{x}_\eta(z_t) + \frac{\sigma_{t'}}{\sigma_t} \Big( z_t - \alpha_t \tilde{x}_\eta(z_t) \Big)$ 是其中一个teacher DDIM采样的更新公式，表示从 $z_t$ 经过一步采样得到 $z_{t'}$。具体推导可以参见原文[^salimans_progressive_distillation].

{{% admonition type="quote" title="Consistency Model" open=true %}}
Given a trajectory $\{\mathbf{x}_t \vert t \in [\epsilon, T]\}$ , the consistency function $f$ is defined as $f: (\mathbf{x}_t, t) \mapsto \mathbf{x}_\epsilon$ and the equation $f(\mathbf{x}_t, t) = f(\mathbf{x}_{t’}, t’) = \mathbf{x}_\epsilon$ holds true for all $t, t’ \in [\epsilon, T]$. When $t=\epsilon$, $f$ is an identify function. The model can be parameterized as follows, where $c_\text{skip}(t)$ and $c_\text{out}(t)$ functions are designed in a way that $c_\text{skip}(\epsilon) = 1, c_\text{out}(\epsilon) = 0$:

$$ f_\theta(\mathbf{x}, t) = c_\text{skip}(t)\mathbf{x} + c_\text{out}(t) F_\theta(\mathbf{x}, t) $$
It is possible for the consistency model to generate samples in a single step, while still maintaining the flexibility of trading computation for better quality following a multi-step sampling process.
{{% /admonition %}}

Consistency Model（CM，一致性模型）的目标是学一个直接映射$f$，能把任意噪声等级 $t>0$ 的点 $\mathbf{x}_t$ 直接送回同一条生成轨迹的“源头”（更精确地说，是非常靠近 0 的一个小时间点 $\epsilon$ 上的样本 $\mathbf{x}_\epsilon$）。

---

为什么映到 $\mathbf{x}_\epsilon$ 而不是精确的 $\mathbf{x}_0$？

- $t=0$ 往往不够数值稳定（奇异/条件数很差），选一个很小的 $\epsilon>0$ 会**更好训练、更稳**。
- $\mathbf{x}_\epsilon$ 与 $\mathbf{x}_0$ 已经极其接近；需要时再从 $\epsilon\to 0$ 补一两步细化即可。

---

$t=\epsilon$ 时为什么是恒等映射？

论文把 $f_\theta$ 写成一个**带跳连的残差型参数化**：

$$
f_\theta(\mathbf{x},t) \;=\; c_{\text{skip}}(t)\,\mathbf{x} \;+\; c_{\text{out}}(t)\,F_\theta(\mathbf{x},t),
$$

并特意设计

$$
c_{\text{skip}}(\epsilon)=1,\qquad c_{\text{out}}(\epsilon)=0.
$$

于是

$$
f_\theta(\mathbf{x},\epsilon)=\mathbf{x},
$$

也就是恒等映射（identity）。

### Latent Variable Space

{{% admonition type="quote" title="Latent Diffusion Model（LDM）" open=true %}}
$$
\begin{aligned}
&\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V} \\
&\text{where }\mathbf{Q} = \mathbf{W}^{(i)}_Q \cdot \varphi_i(\mathbf{z}_i),\;
\mathbf{K} = \mathbf{W}^{(i)}_K \cdot \tau_\theta(y),\;
\mathbf{V} = \mathbf{W}^{(i)}_V \cdot \tau_\theta(y) \\
&\text{and }
\mathbf{W}^{(i)}_Q \in \mathbb{R}^{d \times d^i_\epsilon},\;
\mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V \in \mathbb{R}^{d \times d_\tau},\;
\varphi_i(\mathbf{z}_i) \in \mathbb{R}^{N \times d^i_\epsilon},\;
\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}
\end{aligned}
$$
{{% /admonition %}}

这个公式描述的是 **Latent Diffusion Model（LDM）** 中用于 **交叉注意力（cross-attention）机制** 的一个关键模块。它结合了 **Transformer 中的注意力机制** 与 **扩散模型中的条件控制机制**，是 LDM 实现 **文本到图像生成** 或其他条件生成任务的核心组件。

---

缩放点积注意力（scaled dot-product attention）:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right) \cdot \mathbf{V}
$$

- $\mathbf{Q}$：查询（Query），代表当前需要处理的信息。
- $\mathbf{K}$：键（Key），代表可被关注的信息。
- $\mathbf{V}$：值（Value），当 $\mathbf{Q}$ 和 $\mathbf{K}$ 匹配时，从中提取实际内容。
- 分母 $\sqrt{d}$ 是缩放因子，防止内积过大导致 softmax 梯度消失。
- softmax 沿着 Key 的维度归一化，得到注意力权重。
- 最终输出是一个加权和，表示“根据查询，从值中提取哪些部分更重要”。

---

LDM 中 Query、Key、Value 的来源：

- **Query 来自图像（潜在表示）**
- **Key 和 Value 来自文本/条件信息**

这意味着：**图像的每一个空间位置都在“关注”哪些文本词最相关**。例如，生成一张“一只红色的狗在草地上奔跑”的图片时，图像中“草地”区域会更多地关注文本中的“grass”这个词。

这种机制让模型能够精确地将语义条件与图像空间结构对齐。

| 符号                                                         | 含义                                                                                                                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{z}_i$                                               | 第 $i$ 层的潜在变量（latent feature），即扩散模型在 U-Net 中某一层的特征图，处于低维潜在空间中。                                                                                                              |
| $\varphi_i(\mathbf{z}_i)$                                    | 将潜在特征 $\mathbf{z}_i$ 投影或 reshape 成适合注意力计算的形式（例如展平为序列）。维度：$N \times d^i_\epsilon$，其中 $N$ 是空间位置数（如 H×W），$d^i_\epsilon$ 是该层特征维度。                            |
| $y$                                                          | 条件输入（如文本描述）。                                                                                                                                                                                      |
| $\tau_\theta(y)$                                             | 条件编码器（如 CLIP 或 BERT）对条件 $y$ 的编码结果。输出为一组 token embeddings（例如每个词一个向量）。维度：$M \times d_\tau$，其中 $M$ 是 token 数量（如 77 个文本 token），$d_\tau$ 是嵌入维度（如 768）。 |
| $\mathbf{W}^{(i)}_Q, \mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V$ | 可学习的投影矩阵（参数），用于将输入映射到注意力空间中的 Q/K/V。                                                                                                                                              |

## Scale up Generation Resolution and Quality

这段是在讲如何通过一系列技术手段，把扩散模型的图像生成质量和分辨率提升到更高水平。主要讲了Noise Conditioning Augmentation技巧，unCLIP模型结构，Imagen模型结构，以及其他一些改进。最重要的是要了解Noise Conditioning Augmentation和unCLIP模型结构。

{{% admonition type="quote" title="Noise conditioning augmentation" open=true %}}
_Noise conditioning augmentation_ between pipeline models is crucial to the final image quality, which is to apply strong data augmentation to the conditioning input $\mathbf{z}$ of each super-resolution model $p_\theta(\mathbf{x} \vert \mathbf{z})$. The conditioning noise helps reduce compounding error in the pipeline setup...

They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution. In addition, they also explored two forms of conditioning augmentation that require small modification to the training process. Note that conditioning noise is only applied to training but not at inference.

- Truncated conditioning augmentation stops the diffusion process early at step $t > 0$ for low resolution.
- Non-truncated conditioning augmentation runs the full low resolution reverse process until step 0 but then corrupt it by $\mathbf{z}_t \sim q(\mathbf{x}_t \vert \mathbf{x}_0)$ and then feeds the corrupted $\mathbf{z}_t$ s into the super-resolution model.
{{% /admonition %}}

用pipeline of multiple diffusion models时，每一级的输入是上一级的输出，因此就容易造成误差累积，导致最终图像的损坏，因此他们引入了 **噪声条件增强（Noise Conditioning Augmentation）**，用来让模型学会在“有点模糊或有点噪声”的条件下也能生成清晰图像，避免误差在多级模型中逐步放大。

两种噪声：

- **低分辨率阶段**：加入高斯噪声。
- **高分辨率阶段**：加入高斯模糊。
- 这些噪声只在训练时加入，推理阶段不使用。

两种训练方式：

1. **Truncated Conditioning Augmentation**：在低分辨率阶段提前终止扩散过程（比如只跑到第 t 步）。
2. **Non-Truncated Conditioning Augmentation**：完整跑完低分辨率扩散过程，然后再人为加入噪声，作为高分模型的输入。

{{% admonition type="quote" title="unCLIP" open=true %}}
The two-stage diffusion model unCLIP ([Ramesh et al. 2022](https://arxiv.org/abs/2204.06125)) heavily utilizes the CLIP text encoder to produce text-guided images at high quality. Given a pretrained CLIP model $\mathbf{c}$ and paired training data for the diffusion model, $(\mathbf{x}, y)$, where $x$ is an image and $y$ is the corresponding caption, we can compute the CLIP text and image embedding, $\mathbf{c}^t(y)$ and $\mathbf{c}^i(\mathbf{x})$, respectively. The unCLIP learns two models in parallel:

- A prior model $P(\mathbf{c}^i \vert y)$: outputs CLIP image embedding $\mathbf{c}^i$ given the text $y$.
- A decoder $P(\mathbf{x} \vert \mathbf{c}^i, [y])$: generates the image $\mathbf{x}$ given CLIP image embedding $\mathbf{c}^i$ and optionally the original text $y$.

These two models enable conditional generation, because

$$ \underbrace{P(\mathbf{x} \vert y) = P(\mathbf{x}, \mathbf{c}^i \vert y)}_{\mathbf{c}^i\text{ is deterministic given }\mathbf{x}} = P(\mathbf{x} \vert \mathbf{c}^i, y)P(\mathbf{c}^i \vert y) $$

{{< media
src="unCLIP_Structure.png"
caption="The architecture of unCLIP. (Image source: [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125))"
>}}

unCLIP follows a two-stage image generation process:

1. Given a text $y$, a CLIP model is first used to generate a text embedding $\mathbf{c}^t(y)$. Using CLIP latent space enables zero-shot image manipulation via text.
2. A diffusion or autoregressive prior $P(\mathbf{c}^i \vert y)$ processes this CLIP text embedding to construct an image prior and then a diffusion decoder $P(\mathbf{x} \vert \mathbf{c}^i, [y])$ generates an image, conditioned on the prior. This decoder can also generate image variations conditioned on an image input, preserving its style and semantics.
{{% /admonition %}}

unCLIP是一个两阶段的文本生成图像模型，核心是利用 CLIP 的文本和图像嵌入：

1. **Prior 模型**：输入文本 → 输出 CLIP 图像嵌入。
2. **Decoder 模型**：输入图像嵌入（和可选文本）→ 输出图像。

这种设计允许：

- 文本生成图像。
- 给定图像生成变体（保持风格和语义）。

和unCLIP类似，Imagen模型也是两阶段的文本生成图像模型，区别在于文本编码器的选择。unCLIP用的是CLIP，Imagen用的是大型语言模型T5-XXL。Imagen另外还对U-Net结构做了一些优化。

## Model Architecture

{{% admonition type="quote" title="U-Net，ControlNet，DiT的实现" open=true %}}
**U-Net** ([Ronneberger, et al. 2015](https://arxiv.org/abs/1505.04597)) consists of a downsampling stack and an upsampling stack...

**ControlNet** ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543)) introduces architectural changes via adding a “sandwiched” zero convolution layers of a trainable copy of the original model weights into each encoder layer of the U-Net...

**Diffusion Transformer** (**DiT**; [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) for diffusion modeling operates on latent patches, using the same design space of [LDM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#ldm) (Latent Diffusion Model)...
{{% /admonition %}}

Weng的blog已经对这三个模型进行了介绍，下面展示他们的精简 PyTorch 实现示例。这些示例旨在展示核心的架构思想，并非完整的可运行代码。这里需要对CNN网络有些了解，参考Wang的CNN Explainer文章[^zijie_cnn]以掌握一些基础。

**U-Net** 主要展示了下采样、上采样和跳跃连接的核心思想。它使用了一个简单的 `double_conv` 块，包含了两个卷积层、ReLU 和批量归一化。

{{< details "U-Net PyTorch Implementation" >}}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.down5 = nn.MaxPool2d(2)

        # Upsampling path
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)  # 128 from upconv + 128 from skip connection
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)   # 64 from upconv + 64 from skip connection
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down2(self.down1(x1))
        x3 = self.down4(self.down3(x2))

        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        logits = self.outc(x)
        return logits
```

{{< /details >}}

**ControlNet** 相当于是对U-Net的微调方法，他的核心思想是在冻结的 U-Net 主干网络上添加一个可训练的副本，并通过零卷积连接。

{{< details "ControlNet PyTorch Implementation" >}}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.conv.weight.data.fill_(0.)
        self.conv.bias.data.fill_(0.)

    def forward(self, x):
        return self.conv(x)

class ControlNetBlock(nn.Module):
    """A simplified example of a ControlNet block with a frozen backbone."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The frozen U-Net backbone block
        self.backbone_frozen = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # A trainable copy of the backbone block
        self.backbone_trainable = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Initialize trainable part with backbone's weights
        self.backbone_trainable.load_state_dict(self.backbone_frozen.state_dict())

        # Freeze the original backbone
        for param in self.backbone_frozen.parameters():
            param.requires_grad = False

        # Zero convolutions
        self.zero_conv_in = ZeroConv2d(in_channels, in_channels, kernel_size=1)
        self.zero_conv_out = ZeroConv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, conditioning_vector):
        # Apply the frozen backbone to the input
        output_frozen = self.backbone_frozen(x)

        # Apply the trainable copy to the sum of input and conditioning vector
        # The conditioning vector is first processed by a zero convolution
        conditioning_processed = self.zero_conv_in(conditioning_vector)
        trainable_input = x + conditioning_processed
        output_trainable = self.backbone_trainable(trainable_input)

        # Add the output of the trainable part (with zero convolution) to the frozen output
        final_output = output_frozen + self.zero_conv_out(output_trainable)
        return final_output
```

{{< /details >}}

**DiT** 的核心思想是使用 Transformer 来处理扩散模型的潜在表示。它将图像潜在表示“切片”成序列化的 patches，这里展示使用 Adaptive Layer Normalization (adaLN) 来注入时间步长和类别信息。

{{< details "DiT PyTorch Implementation" >}}

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization (adaLN) for DiT.
    Generates scale and shift parameters from a conditioning vector.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        # We need a linear layer to predict gamma and beta
        self.linear = nn.Linear(n_embd, 2 * n_embd)
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x, c):
        # c is the conditioning vector (sum of time and class embeddings)
        gamma, beta = self.linear(c).chunk(2, dim=-1)
        x = self.norm(x)
        # Apply scaling and shifting
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

class DiTBlock(nn.Module):
    """A simplified DiT block, combining Attention and MLP with adaLN."""
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.adaLN1 = AdaLN(n_embd)
        self.attention = nn.MultiheadAttention(n_embd, n_heads, batch_first=True)
        self.adaLN2 = AdaLN(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, c):
        # x is the sequence of patches, c is the conditioning vector
        x = x + self.attention(self.adaLN1(x, c), self.adaLN1(x, c), self.adaLN1(x, c))[0]
        x = x + self.mlp(self.adaLN2(x, c))
        return x

class SimpleDiT(nn.Module):
    """Simplified DiT model for latent diffusion."""
    def __init__(self, img_size, patch_size, n_embd, n_heads, n_layers, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding and positional embedding
        self.patch_embed = nn.Linear(3 * patch_size**2, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, n_embd))

        # Time and class embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.class_embed = nn.Embedding(num_classes, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([DiTBlock(n_embd, n_heads) for _ in range(n_layers)])

        # Final output layer
        self.final_layer = nn.Linear(n_embd, 3 * patch_size**2)

    def forward(self, x, t, c):
        # 1. Patchify and embed
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), -1, 3 * self.patch_size**2)
        x = self.patch_embed(x) + self.pos_embed

        # 2. Time and class conditioning
        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)
        cond = t_emb + c_emb

        # 3. Pass through DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # 4. Final layer to predict noise
        output = self.final_layer(x)
        return output
```

{{< /details >}}

## Quick Summary

{{% admonition type="quote" title="Pros and Cons" open=true %}}

- **Pros**: Tractability and flexibility are two conflicting objectives in generative modeling. Tractable models can be analytically evaluated and cheaply fit data (e.g. via a Gaussian or Laplace), but they cannot easily describe the structure in rich datasets. Flexible models can fit arbitrary structures in data, but evaluating, training, or sampling from these models is usually expensive. Diffusion models are both analytically tractable and flexible

- **Cons**: Diffusion models rely on a long Markov chain of diffusion steps to generate samples, so it can be quite expensive in terms of time and compute. New methods have been proposed to make the process much faster, but the sampling is still slower than GAN.
{{% /admonition %}}

在生成模型设计中，“可解析性”（tractability）与“灵活性”（flexibility）通常是矛盾的目标：

- **可解析性**：指模型的数学结构清晰，便于推理和优化。例如高斯分布、拉普拉斯分布等可以直接计算似然、采样、训练。
- **灵活性**：指模型可以拟合复杂、高维、非线性的数据结构，比如图像、音频、文本等。但这类模型往往难以训练、采样或评估。

传统模型如：

- **VAE**：可解析但牺牲了生成质量。
- **GAN**：灵活但训练不稳定，缺乏显式似然。
- **Flow-based models**：可逆但架构受限。

而扩散模型的独特之处在于：它们通过逐步添加噪声（forward process）和学习去噪（reverse process），在理论上保持了高维高斯建模的可解析性，同时通过深度神经网络学习复杂数据结构，达到了灵活性。

其缺点依然存在，但是仍然有各种新技术在不断涌现，试图解决采样速度慢的问题。

此外，今天，diffusion model已经是de facto的生成模型标准，尤其是在图像生成领域。它们在生成质量、样本多样性和训练稳定性方面表现出色，且易于扩展到更大规模和更复杂的任务。

## Appendix

### 相关深度学习知识点

#### GAN, VAE, and Flow-based models 是什么

{{< media
src="Generative_Models.png"
caption="Overview of different types of generative models. ([Source](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/))"
>}}

这几个都是最常见的几类图像生成模型，可以大致了解其原理：

- GAN 生成对抗网络：训练两个网络，一个用于生成图像，一个用于判别图像的真伪
- VAE 变分自编码器模型：通过编码器将输入图像压缩为潜在空间变量，再通过解码器重建图像
- Flow matching models：通过流动匹配的方法生成图像，是函数级别的过程组合起来生成图像的。

#### 从 AE 到 VAE 再到 VQ-VAE

- AE 是一个具有编码器和解码器的神经网络，目标是学习输入数据的压缩表示（潜在变量），主要用于降维和特征提取。
- VAE 在 AE 的基础上引入了概率建模和变分推断，使模型具备生成能力。
- VQ-VAE（向量量化变分自编码器）进一步引入了离散潜在空间，通过向量量化替代连续潜在分布，更适合离散结构建模（如语音、图像中的符号化特征），并为后续的生成模型（如 Transformer 解码器）提供离散 token 表示。

| 特性                 | 自编码器（AE）                  | 变分自编码器（VAE）                                            | 向量量化 VAE（VQ-VAE）                               |
| -------------------- | ------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------- |
| 编码器输出           | 一个确定性的向量 \( z = f(x) \) | 一个分布 \( q(z \vert x) = \mathcal{N}(\mu(x), \sigma^2(x)) \) | 最近邻查找得到的离散码本向量 \( z \in \mathcal{E} \) |
| 解码器输入           | 固定向量 \( z \)                | 从分布中采样的 \( z \sim q(z \vert x) \)                       | 离散码本向量 \( z \)                                 |
| 潜在空间类型         | 连续、确定性                    | 连续、概率分布                                                 | 离散、有限的码本（codebook）                         |
| 训练目标             | 最小化重构误差（如 MSE）        | 最大化变分下界（ELBO）                                         | 重构误差 + codebook 损失 + commitment 损失           |
| 是否为生成模型       | 否                              | 是                                                             | 是                                                   |
| 是否有概率建模       | 否                              | 有（对潜在变量分布建模）                                       | 无（但提供离散符号化潜在空间）                       |
| 是否可采样生成新数据 | 否                              | 可从先验 \( p(z) \) 采样生成数据                               | 可通过采样离散 token 并解码生成数据                  |
| 是否使用KL散度       | 否                              | 是（约束潜在分布接近先验）                                     | 否（改用向量量化和额外损失函数替代）                 |

#### Reparameterization Trick

定义：重参数化**将随机变量从不可导的采样操作中解耦出来**的方法，让采样操作可以参与梯度下降优化。

原理：他没有消除随机采样，只是将随机采样对梯度传播的影响降到了最低.

举例：如果你有一个随机变量 $z \sim \mathcal{N}(\mu, \sigma^2)$，直接从这个分布采样，梯度无法通过 $\mu, \sigma$ 传播。那就可以按照下式从随机采样 $z$ 转换为随机采样 $\epsilon$。

$$
\mathcal{N}(z; \mu, \sigma^2)
= \mu + \mathcal{N}(\epsilon'; 0, \sigma^2)
= \mu + \sigma \cdot \mathcal{N}(\epsilon; 0, 1)
$$

PyTorch 代码示例：

```python
import torch

# 假设编码器输出的均值和标准差
mu = torch.tensor([0.0, 1.0], requires_grad=True)
log_sigma = torch.tensor([0.0, 0.0], requires_grad=True)  # 通常输出 log(sigma) 避免负数
sigma = torch.exp(log_sigma)

# 重参数化采样
epsilon = torch.randn_like(mu)  # 从标准正态采样
z = mu + sigma * epsilon  # z 可导

# 假设一个简单的损失函数
loss = (z**2).sum()
loss.backward()

print("grad mu:", mu.grad)
print("grad log_sigma:", log_sigma.grad)
```

#### Importance Sampling Trick

假设你想计算某个函数 \( f(x) \) 关于概率分布 \( p(x) \) 的期望：

\[
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx
\]

但直接从 \( p(x) \) 采样困难，或者 \( f(x) \) 在 \( p(x) \) 下大多数样本贡献很小，只有少数区域贡献大（比如尾部事件），这时直接蒙特卡洛估计效率很低。

如果能**从另一个更容易采样的分布 \( q(x) \) 中采样，然后通过加权来纠正偏差**，那么估计起来就会更方便，这就是重要性采样。

重要性采样对原期望重写为：

\[
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[ f(x) \frac{p(x)}{q(x)} \right]
\]

其中，

- \( q(x) \)是**提议分布（proposal distribution）** ，  并且 \( q(x) > 0 \) 当 \( p(x) > 0 \)（即支撑集包含 \( p \) 的支撑集）
- \( w(x_i) = \frac{p(x_i)}{q(x_i)} \) 被称为**重要性权重（importance weight）**。

这时，我们就可以从 \( q(x) \) 中采样 \( x_i \sim q(x) \)，然后估计：

\[
\hat{\mu} = \frac{1}{N} \sum_{i=1}^N f(x_i) \cdot \frac{p(x_i)}{q(x_i)}
\]

让我们举个例子，假设：

- \( p(x) = \mathcal{N}(0, 1) \)
- \( f(x) = x^2 \cdot \mathbb{1}[x > 2] \)
- 直接从 \( p \) 采样，大部分样本 \( x < 2 \)，贡献为0，效率低。
- 如果改用 \( q(x) = \mathcal{N}(2.5, 1) \) 来采样，更可能采到 \( x > 2 \) 的区域。

代码示意（Python）：

```python
import numpy as np

N = 10000
# 从 q(x) ~ N(2.5, 1) 采样
x = np.random.normal(2.5, 1, N)

# 计算未归一化密度（忽略常数）
log_p = -0.5 * x**2           # log N(0,1)
log_q = -0.5 * (x - 2.5)**2   # log N(2.5,1)

# 重要性权重（未归一化）
w = np.exp(log_p - log_q)

# 函数值
f = x**2 * (x > 2)

# 归一化重要性采样估计
mu_hat = np.sum(f * w) / np.sum(w)
print("Importance Sampling estimate:", mu_hat)
```

#### 重要的diffusion相关的论文

这些论文均为 lilian weng写作diffusion博文的时候所引用的文章，同样也是diffusion领域最重要的一些文章。

| 论文                                                                                                                                                                   | 介绍                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [1] Jascha Sohl-Dickstein et al. “Deep Unsupervised Learning using Nonequilibrium Thermodynamics.” ICML 2015.                                                          | 最早提出基于扩散过程的生成建模思想，将前向噪声扩散与反向过程联系起来，奠定了 diffusion model 的基础。    |
| [2] Max Welling & Yee Whye Teh. “Bayesian learning via stochastic gradient langevin dynamics.” ICML 2011.                                                              | 提出 SGLD 方法，将梯度下降与 Langevin 动力学结合，用噪声驱动采样，启发了后续基于随机微分方程的生成模型。 |
| [3] Yang Song & Stefano Ermon. “Generative modeling by estimating gradients of the data distribution.” NeurIPS 2019.                                                   | 提出 **score matching + Langevin dynamics** 框架，通过估计数据分布梯度（score function）进行生成建模。   |
| [4] Yang Song & Stefano Ermon. “Improved techniques for training score-based generative models.” NeurIPS 2020.                                                         | 提出多噪声尺度训练和改进的采样技巧，使 score-based models 性能大幅提升。                                 |
| [5] Jonathan Ho et al. “Denoising diffusion probabilistic models.” arXiv 2020.                                                                                         | 提出 **DDPM**，将扩散模型与变分推断结合，首次在图像生成上取得接近 GAN 的效果。                           |
| [6] Jiaming Song et al. “Denoising diffusion implicit models.” arXiv 2020.                                                                                             | 提出 **DDIM**，提供确定性采样方式，大幅减少采样步骤并保持高质量生成。                                    |
| [7] Alex Nichol & Prafulla Dhariwal. “Improved denoising diffusion probabilistic models.” arXiv 2021.                                                                  | 提出改进训练方法（如余弦噪声调度、数据增强），显著提升 DDPM 性能。                                       |
| [8] Prafula Dhariwal & Alex Nichol. “Diffusion Models Beat GANs on Image Synthesis.” arXiv 2021.                                                                       | 展示扩散模型在图像生成质量上超越 GAN，推动扩散模型成为主流生成方法。                                     |
| [9] Jonathan Ho & Tim Salimans. “Classifier-Free Diffusion Guidance.” NeurIPS 2021 Workshop.                                                                           | 提出 **无分类器引导**，通过条件/无条件模型差值实现 controllable generation，成为主流控制方法。           |
| [10] Yang Song, et al. “Score-Based Generative Modeling through Stochastic Differential Equations.” ICLR 2021.                                                         | 将扩散与 SDE 统一，提出连续时间 score-based framework，连接 DDPM 和 SDE。                                |
| [11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. “GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.” ICML 2022. | 提出 **GLIDE**，结合扩散与 CLIP 文本引导，实现高质量文本到图像生成。                                     |
| [12] Jonathan Ho, et al. “Cascaded diffusion models.” JMLR 2022.                                                                                                       | 提出级联扩散模型，通过逐级提高分辨率生成高保真图像。                                                     |
| [13] Aditya Ramesh et al. “Hierarchical Text-Conditional Image Generation with CLIP Latents.” arXiv 2022.                                                              | 提出 **DALL·E 2** 的核心：利用 CLIP latent 作为扩散条件，实现语义一致的文本-图像生成。                   |
| [14] Chitwan Saharia & William Chan, et al. “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.” arXiv 2022.                              | 提出 **Imagen**，结合大规模语言模型与扩散，实现当时最优的文本到图像生成。                                |
| [15] Rombach & Blattmann, et al. “High-Resolution Image Synthesis with Latent Diffusion Models.” CVPR 2022.                                                            | 提出 **Latent Diffusion / Stable Diffusion**，在潜在空间而非像素空间扩散，大幅提高效率。                 |
| [16] Song et al. “Consistency Models.” arXiv 2023.                                                                                                                     | 提出一致性模型，支持一步/少步采样，提升生成速度。                                                        |
| [17] Salimans & Ho. “Progressive Distillation for Fast Sampling of Diffusion Models.” ICLR 2022.                                                                       | 提出蒸馏方法，将扩散模型加速至少量采样步数。                                                             |
| [18] Ronneberger, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” MICCAI 2015.                                                               | 提出 U-Net 架构，后来成为扩散模型去噪网络的标准 backbone。                                               |
| [19] Peebles & Xie. “Scalable diffusion models with transformers.” ICCV 2023.                                                                                          | 将扩散模型 backbone 换为 transformer，提升可扩展性与生成质量。                                           |
| [20] Zhang et al. “Adding Conditional Control to Text-to-Image Diffusion Models.” arXiv 2023.                                                                          | 提出 **ControlNet**，在已有扩散模型上添加条件控制层，实现可控图像生成。                                  |

### 概率论

#### Gaussian distribution

**一维高斯分布（Univariate Gaussian）** $x \sim \mathcal{N}(\mu, \sigma^2)$ 的PDF是

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- **均值 $\mu$**：分布的中心位置
- **方差 $\sigma^2$**（标准差 $\sigma$）：分布的宽度（不确定性）

**多维高斯分布（Multivariate Gaussian）** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的概率密度函数 (pdf)：

$$
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}}
\exp\!\left(-\tfrac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中：

- $\mathbf{x} \in \mathbb{R}^d$ 是 $d$ 维向量
- 均值向量 $\boldsymbol{\mu} \in \mathbb{R}^d$：分布的中心位置。
- 协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{d\times d}$：描述不同维度之间的方差和相关性。
- $|\boldsymbol{\Sigma}|$ = 协方差矩阵的行列式，代表“体积缩放”。
- $\boldsymbol{\Sigma}^{-1}$ = 协方差矩阵的逆，定义了“椭球形”的等密度曲线。

对 $\boldsymbol{\Sigma}$ 的分解能揭示分布的几何性质：

- 对角元素：每个维度的方差（数值大小 = 在该轴上的“宽度”）。
- 非对角元素：不同维度之间的相关性，决定分布是否是“旋转的椭圆/椭球”。

例子：

- 如果 $\boldsymbol{\Sigma} = \sigma^2 I$，就是一个各向同性的“圆形/球形”分布。
- 如果 $\boldsymbol{\Sigma}$ 不是对角阵，就有相关性，等密度线是“倾斜的椭圆/椭球”。

下面两个图示更直观的展示了 $\mu$ 和 $\Sigma$ 对PDF的形状的影响 [^saleem_gaussian]：

{{< media
src="Gaussian_Mean.gif"
caption="Changes to the mean vector act to translate the Gaussian’s main ‘bump’. ([source](https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b))"
>}}

{{< media
src="Gaussian_Covariance.gif"
caption="Changes to the covariance matrix act to change the shape of the Gaussian’s main ‘bump’. ([source](https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b))"
>}}

#### 联合分布，边缘分布和条件分布

- 联合分布 $P(A, B)$：全景地图（包含所有组合的概率）。
- 边缘分布 $P(B)$：全景地图投影到某一个轴。
- 条件分布 $P(A \vert B)$：全景地图切一条线（已知另一变量的值），看这条线上的概率分布。公式为：$P(A \vert B) = \frac{P(A, B)}{P(B)}$

#### 先验，似然，与后验

| 概念 | 类比解释                 | 在扩散模型中的类比                                     |
| ---- | ------------------------ | ------------------------------------------------------ |
| 先验 | 在观察任何数据之前对变量的假设分布             | $q(\mathbf{x}_t) \sim \mathcal{N}(0, I)$               |
| 似然 | 某假设/模型参数下，观测数据出现的概率 | $p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$               |
| 后验 | 结合先验与观测数据之后对潜在变量的更新分布     | $q(\mathbf{x}_{t-1}\mid \mathbf{x}_t, \mathbf{x}_0)$ |

三者之间的关系是：

\[
\text{后验} = \frac{\text{似然} \times \text{先验}}{\text{证据}}
\quad\leftrightarrow\quad
P(\text{参数} | \text{数据}) = \frac{P(\text{数据} | \text{参数}) \cdot P(\text{参数})}{P(\text{数据})}
\]

#### Bayes’ rule

贝叶斯公式（Bayes’ Rule）是概率论中的一个核心法则，用于在已知条件下更新事件的概率。它的基本形式是：

\[
P(A \vert B) = \frac{P(B \vert A) \cdot P(A)}{P(B)}
\]

其中：

- \(P(A)\)：事件 A 的先验概率（在观察 B 之前对 A 的信念）
- \(P(B \vert A)\)：在 A 发生的前提下，观察到 B 的可能性（似然）
- \(P(B)\)：事件 B 的边际概率（所有可能情况下 B 发生的概率）
- \(P(A \vert B)\)：在观察到 B 之后，事件 A 的后验概率（更新后的信念）

#### Law of Total Probability

全概率公式用于**计算一个事件的概率**，当这个事件的概率不容易直接计算时，可以通过将其**分解为若干互斥情形下的条件概率之和**来求解。

设事件 \( B_1, B_2, \dots, B_n \) 构成样本空间的一个**完备事件组**（即它们互斥且并集为全集），即：

- \( B_i \cap B_j = \emptyset \)（互斥）
- \( \bigcup_{i=1}^n B_i = \Omega \)（穷尽）

那么对于任意事件 \( A \)，有：

\[
P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)
\]

#### 边际化（Marginalization）

边际化是指：**从联合概率分布中，通过“对某些变量求和（或积分）”来得到另一个变量的概率分布**。

比如，已知两个随机变量 \( X \) 和 \( Y \) 的联合分布 \( P(X, Y) \)，我们想得到 \( X \) 的分布 \( P(X) \)，就需要对 \( Y \) 所有可能的取值进行“边缘化”：

$$
\begin{align}
&\text{离散情况: } P(X = x) = \sum_{y} P(X = x, Y = y) \\
&\text{连续情况: } P(X = x) = \int_{-\infty}^{\infty} P(X = x, Y = y) \, dy
\end{align}
$$

此外，我们还可以把全概率公式看作是边际化的应用：

$$P(X) = \sum_{y} P(X, Y = y) = \sum_{y} P(X | Y = y) P(Y = y)$$

#### 期望

期望的定义

- 离散随机变量：$\mathbb{E}_{x \sim p}[g(x)] = \sum_{x} g(x) \cdot p(x)$
- 连续随机变量：$\mathbb{E}_{x \sim p}[g(x)] = \int_{-\infty}^{\infty} g(x) \cdot p(x)  dx$

三个重要的性质

1. 线性性质：
  $\mathbb{E}_{x \sim p, y \sim q}[a \cdot f(x, y) + b \cdot g(x, y)] = a \cdot \mathbb{E}_{x \sim p, y \sim q}[f(x, y)] + b \cdot \mathbb{E}_{x \sim p, y \sim q}[g(x, y)]$
2. 期望的迭代法则（Law of Iterated Expectation / Tower Property）：
  $\mathbb{E}_{(x, y) \sim p(x, y)}[f(x, y)] = \mathbb{E}_{y \sim p(y)}\left[ \mathbb{E}_{x \sim p(x|y)}[f(x, y)] \right]$
3. 全期望公式
  $\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{y \sim p(y)}\left[ \mathbb{E}_{x \sim p(x|y)}[f(x)] \right]$

#### 随机过程

随机过程（Stochastic Process）是随时间（或空间）演化的随机变量族。一个随机过程可以写成：

$$
\{ X_t \}_{t \in T}
$$

- $t$：索引集，可以是 **离散的**（如整数时间点 $t=0,1,2,\dots$）或 **连续的**（如实数时间 $t \ge 0$）。
- $X_t$：在每个时间点 $t$ 上的一个随机变量。
- 整个过程就是一组随机变量组成的族，反映系统随 $t$ 演化时的随机性。

直观理解：

- 随机变量是“某个时刻的随机量”；
- 随机过程是“随时间变化的一串随机量”。

#### Markov Property

一个随机过程若满足

$$
P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)
$$

就说它具有马尔可夫性质，即未来只依赖于现在，而与过去无关。

#### Score Function

在概率论和统计学中，**score function** 原本的定义是：

$$
s_\theta(x) = \nabla_\theta \log p_\theta(x)
$$

- 这是对**参数 θ 的对数似然函数**的梯度。
- 在经典统计中，它用来做 **最大似然估计** 或 **Fisher 信息量计算**。
- 直觉上，它告诉你“如果想让观测数据 x 更可能，应该如何调整模型参数 θ”。

但是在 **diffusion 模型 / score-based generative model** 中，score function 被扩展为：

$$
s(x) = \nabla_x \log p(x)
$$

- 这里是对 **数据本身 x 的对数密度**求梯度。
- 它告诉你**数据分布的上升方向**，也就是“哪里数据更可能出现”。

### 信息论

下面用一个场景来讲解信息论中最重要的四个公式：你要发送一系列消息（比如字母），每个字母出现的概率不尽相同，为了节省带宽，你需要用最短的二进制编码来发送它们......

#### 信息量 $I(x) = -\log p(x)$

- 直觉：一个事件包含的信息多少，和它发生的意外程度有关。越不可能发生的事，一旦发生，带来的信息量就越大。
  - “太阳从东边升起” (概率~1)：几乎不带来信息，信息量极少。
  - “明天会下雨” (概率0.3)：带来一些信息。
  - “北京明天下了钻石雨” (概率几乎为0)：如果发生，信息量是爆炸性的。
- 为什么是 $-\log p(x)$？
  - 编码角度：为了最优地编码一个事件，我们应该给出现概率高的事件分配短的码字，给出现概率低的事件分配长的码字。这样平均编码长度最短。
  - 最优编码长度：一个事件 $x$ 的最优编码长度就是 $-\log p(x)$。概率 $p(x)$ 越小，编码长度 $-\log p(x)$ 就越大，这完美对应了“意外程度”。
  - 为什么用log：使用对数可以将概率的相乘关系（独立事件联合概率）转为编码长度的相加关系，非常方便。
- 结论：信息量 $I(x)$ 就是事件 $x$ 的最优编码长度。

#### 香农熵 $H(p) = -\sum_i p(x_i) \log p(x_i)$

- 直觉：对于一个概率分布 $p$，我们对其中的事件进行编码，平均每个事件需要的最短编码长度是多少？ 这个“平均最低成本”就是分布的不确定性。
  - 一个分布越均匀（比如公平骰子），你越难猜中下一个结果，它的不确定性就越高，平均编码长度也越长。
  - 一个分布越集中（比如偏袒的骰子），你很容易猜中下一个结果，它的不确定性就低，平均编码长度也短。
- 为什么是 $-\sum_i p(x_i) \log p(x_i)$？
  - 这就是信息量的期望值：$\mathbb{E}_{x \sim p}[I(x)] = \mathbb{E}_{x \sim p}[-\log p(x)]$。
  - 每个事件 $x_i$ 都有自己的最优编码长度 $-\log p(x_i)$。我们用该事件发生的概率 $p(x_i)$ 作为权重，对所有可能的编码长度求平均，就得到了整个分布的平均最优编码长度。
- 结论：香农熵 $H(p)$ 就是用分布 $p$ 本身的最优编码方案时，所需的平均编码长度。它衡量了分布 $p$ 固有的不确定性。

#### 交叉熵 $H(p, q) = -\sum_i p(x_i) \log q(x_i)$

- 直觉：现在情况变了。数据的真实分布是 $p$，但你误以为分布是 $q$，并采用了为 $q$ 设计的最优编码方案（即给事件 $x_i$ 分配了长度为 $-\log q(x_i)$ 的码字）。用这个错的方案去编码真实的数据，平均需要多长的码字？
- 为什么是 $-\sum_i p(x_i) \log q(x_i)$？
  - 真实数据中，事件 $x_i$ 出现的概率是 $p(x_i)$。
  - 你为它分配的码字长度是 $-\log q(x_i)$。
  - 所以，整体的平均编码长度就是 $\mathbb{E}_{x \sim p}[-\log q(x)]$。
- 结论：交叉熵 $H(p, q)$ 就是用为 $q$ 设计的最优编码去表示来自 $p$ 的数据时，所需的平均编码长度。

#### 相对熵 (KL散度) $D_{KL}(p\|q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$

- 直觉：承接交叉熵的概念。既然用错误的方案 q 编码会比用正确的方案 p 编码要长，那么平均每个样本，我们多浪费了多少编码长度？ 这个“多浪费的长度”就是两个分布之间的差异。
- 为什么是 $\sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$？
  - 对于事件 $x_i$，使用错误方案的长度是 $-\log q(x_i)$，使用正确方案的长度是 $-\log p(x_i)$。
  - 多浪费的长度就是：$(-\log q(x_i)) - (-\log p(x_i)) = \log \frac{p(x_i)}{q(x_i)}$。
  - 我们用真实概率 $p(x_i)$ 对这个“浪费”求平均，就得到了平均浪费的长度：$\mathbb{E}_{x \sim p}[\log \frac{p(x)}{q(x)}]$。
- 结论：KL散度 $D_{KL}(p\|q)$ 衡量了因为使用近似分布 $q$ 而不是真实分布 $p$ 所带来的额外的平均编码长度。它直接衡量了两个分布的差异。

#### 关系 $D_{KL}(p\|q) = H(p, q) - H(p) \geq 0$

让我们进行完整的推导：

$$
\begin{align}
D_{KL}(p\|q) &= H(p, q) - H(p) \\
&= \mathbb{E}_{x \sim p}[I_q(x)] - \mathbb{E}_{x \sim p}[I_p(x)] \\
&= \mathbb{E}_{x \sim p}[-\log q(x)] - \mathbb{E}_{x \sim p}[-\log p(x)] \\
&= \mathbb{E}_{x \sim p} \Big[\log \frac{p(x)}{q(x)} \Big] \\
&= -\mathbb{E}_{x \sim p} \Big[\log \frac{q(x)}{p(x)} \Big] \\
&\geq -\log \mathbb{E}_{x \sim p} \Big[ \frac{q(x)}{p(x)} \Big] \quad \text{; Jensen's Inequality} \\
&= -\log \sum_{x \sim p} \Big[p(x) \frac{q(x)}{p(x)} \Big] \\
&= -\log 1 \\
&= 0
\end{align}
$$

## Citation

{{< bibtex >}}

## References

[^ho_ddpm]: **Ho, Jonathan, Ajay Jain, and Pieter Abbeel.** "Denoising Diffusion Probabilistic Models." _Advances in Neural Information Processing Systems_, vol. 33, edited by H. Larochelle et al., Curran Associates, Inc., 2020, pp. 6840–6851. _NeurIPS_, https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

[^nichol_improved_ddpm]: **Nichol, Alexander Quinn, and Prafulla Dhariwal.** "Improved Denoising Diffusion Probabilistic Models." _Proceedings of the 38th International Conference on Machine Learning_, vol. 139, edited by Marina Meila and Tong Zhang, Proceedings of Machine Learning Research, 18–24 July 2021, pp. 8162–8171. _PMLR_, https://proceedings.mlr.press/v139/nichol21a.html.

[^song_consistency]: **Song, Yang, et al.** "Consistency Models." _International Conference on Machine Learning_, 2023. _ICML_, https://icml.cc/virtual/2023/poster/24593.

[^song_ddim]: **Song, Jiaming, Chenlin Meng, and Stefano Ermon.** "Denoising Diffusion Implicit Models." _International Conference on Learning Representations_, 2021. _OpenReview_, https://openreview.net/forum?id=St1giarCHLP.

[^salimans_progressive_distillation]: **Salimans, Tim, and Jonathan Ho.** "Progressive Distillation for Fast Sampling of Diffusion Models." _International Conference on Learning Representations_, 2022. _OpenReview_, https://openreview.net/forum?id=TIdIXIpzhoI.

[^salimans_improve_gan]: **Salimans, Tim, et al.** "Improved Techniques for Training GANs." _Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16)_, Curran Associates Inc., 2016, pp. 2234–2242. _ACM Digital Library_, https://dl.acm.org/doi/10.5555/3157096.3157346.

[^heusel_fid]: **Heusel, Martin, et al.** "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." _Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)_, Curran Associates Inc., 2017, pp. 6629–6640. _ACM Digital Library_, https://dl.acm.org/doi/10.5555/3295222.3295408.

[^mc_candlish_grad_noise]: **McCandlish, Sam, et al.** _An Empirical Model of Large-Batch Training_. 14 Dec. 2018. _arXiv_, https://arxiv.org/abs/1812.06162.

[^lilian_diffusion]: **Weng, Lilian.** "What Are Diffusion Models?" _Lil'Log_, 11 July 2021, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[^lilian_ae]: **Weng, Lilian.** "From Autoencoder to Beta-VAE." _Lil'Log_, 12 Aug. 2018, https://lilianweng.github.io/posts/2018-08-12-vae/.

[^saleem_gaussian]: **Saleem, Ameer.** "Unpacking the Multivariate Gaussian Distribution." _Medium_, 12 May 2025, https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b.

[^gupta_gaussian_kl]: **Gupta, Rishabh.** "KL Divergence between 2 Gaussian Distributions." _Mr. Easy_, 16 Apr. 2020, https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/.

[^zijie_cnn]: **Wang, Zijie J., et al.** "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization." _IEEE Transactions on Visualization and Computer Graphics (TVCG)_, IEEE, 2020. https://poloclub.github.io/cnn-explainer/.

[^wiki_closed]: Wikipedia contributors. "Closed-form expression." _Wikipedia, The Free Encyclopedia_, 26 July 2025, https://en.wikipedia.org/wiki/Closed-form_expression. Accessed 1 Sept. 2025.

[^wiki_jensen]: Wikipedia contributors. "Jensen's inequality." _Wikipedia, The Free Encyclopedia_, 12 June 2025, https://en.wikipedia.org/wiki/Jensen%27s_inequality. Accessed 23 Aug. 2025.

[^wiki_concave]: Wikipedia contributors. "Concave function." _Wikipedia, The Free Encyclopedia_, 17 July 2025, https://en.wikipedia.org/wiki/Concave_function. Accessed 23 Aug. 2025.
