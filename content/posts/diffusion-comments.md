---
date: '2025-08-15T01:41:53+08:00'
title: '《What are Diffusion Models?》注释'
author:
  - Shichao Song
summary: ''
tags: ["diffusion"]
math: true
---

<!-- TODO：记得email给lilian about this article -->

本文对Lilian Weng的《What are Diffusion Models?》 [^lilian_diffusion] 进行完善的注释导读。

笔者在写此文时，对图像生成模型了解和相关的数学背景知识了解均较少。如果你也有相似的背景，那么此文应该会适合你。当然，我可能也因此犯一些低级错误，敬请指正。

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

整体扩散过程是，根据马尔可夫性质(TODO: ref to appendix)将单步扩散过程连乘起来的递推式。
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

根据单步扩散过程$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$，以及重参数化技巧(TODO: ref to appendix) $z = \mu + \sigma \cdot \epsilon$，我们可以重写单步扩散过程为：

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
- \( \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \): 漂移项，根据目标分布的梯度移动，类似受力牵引。也可以类比为前向单步前向扩散中的 $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$。
  - \( \delta / 2 \): 步长，控制每次更新的幅度
  - \( p(x) \)：目标分布的概率密度函数
  - \( \log p(x) \)：对数概率密度，便于计算和优化
  - \( \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \)：对数概率密度的梯度，也叫 score function，表示当前点的“上升方向”
- \( \sqrt{\delta} \boldsymbol{\epsilon}_t \): 扩散项，像布朗运动的分子碰撞。可以类比为前向单步前向扩散中的 $\sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}$。
  - \( \sqrt{\delta} \)：步长（step size），控制每次更新的幅度
  - \( \epsilon_t \sim \mathcal{N}(0, I) \)：标准正态分布的随机噪声，加入随机性以避免陷入局部最优

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

刚刚已经后验的公式都推证完了，这样我们就有了golden truth 了，即神经网络要模仿的对象。那么如何建模golden truth和模型之间的关联呢。这就要用到变分推断（Variational Inference）了，或者叫做ELBO（Evidence Lower Bound）。

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

这里推导主要用到了概率论中的边际化以及Jensen 不等式。

对$\mathbf{x}_{1:T}$边际化即:

$$
\begin{align}
p_\theta(\mathbf{x}_0) 
&= \int \Big[ p_\theta(\mathbf{x}_0 | \mathbf{x}_{1:T}) p_\theta(\mathbf{x}_{1:T}) \Big] d\mathbf{x}_{1:T} \\
&= \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
\end{align}
$$

[Jensen 不等式](https://en.wikipedia.org/wiki/Jensen%27s_inequality), 是指设 \( \phi \) 是一个[concave function](https://en.wikipedia.org/wiki/Concave_function)，\( X \) 是一个可积的随机变量，则有不等式: 

\[
\phi\left( \mathbb{E}[X] \right) \geq \mathbb{E}\left[ \phi(X) \right]
\]


例如，$-log(\cdot)$ is a concave function。

































## Appendix

### 生成模型背景

#### GAN, VAE, and Flow-based models 是什么

![Generative Models](/images/Generative_Models.png)

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

#### reparameterization trick

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


#### 重要的diffusion相关的论文

这些论文均为 weng lilian写作diffusion博文的时候所引用的文章，同样也是diffusion领域最重要的一些文章。

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

高斯分布（Gaussian distribution）也被称为**正态分布**，$\mathcal{N}(\mu, \sigma)$，其概率密度函数（PDF, Probability Density Function）为：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

- $\mu$：均值（mean），决定分布的中心位置
- $\sigma$：标准差（standard deviation），决定分布的宽度
- $\sigma^2$：方差（variance）
- $\exp(\cdot)$：自然指数函数 $e^x$

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

#### Score Function

在概率论和统计学中，**score function** 原本的定义是：

$$
s_\theta(x) = \nabla_\theta \log p_\theta(x)
$$

* 这是对**参数 θ 的对数似然函数**的梯度。
* 在经典统计中，它用来做 **最大似然估计** 或 **Fisher 信息量计算**。
* 直觉上，它告诉你“如果想让观测数据 x 更可能，应该如何调整模型参数 θ”。

但是在 **diffusion 模型 / score-based generative model** 中，score function 被扩展为：

$$
s(x) = \nabla_x \log p(x)
$$

* 这里是对 **数据本身 x 的对数密度**求梯度。
* 它告诉你**数据分布的上升方向**，也就是“哪里数据更可能出现”。

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

### 随机过程

随机过程（Stochastic Process）是随时间（或空间）演化的随机变量族。一个随机过程可以写成：

$$
\{ X_t \}_{t \in T}
$$

* $t$：索引集，可以是 **离散的**（如整数时间点 $t=0,1,2,\dots$）或 **连续的**（如实数时间 $t \ge 0$）。
* $X_t$：在每个时间点 $t$ 上的一个随机变量。
* 整个过程就是一组随机变量组成的族，反映系统随 $t$ 演化时的随机性。

直观理解：

* 随机变量是“某个时刻的随机量”；
* 随机过程是“随时间变化的一串随机量”。

#### Markov Property

一个随机过程若满足

$$
P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)
$$

就说它具有马尔可夫性质，即未来只依赖于现在，而与过去无关。

## Citation

{{< bibtex >}}

## References

[^ho_ddpm]: **Ho, Jonathan, Ajay Jain, and Pieter Abbeel.** “Denoising Diffusion Probabilistic Models.” _Advances in Neural Information Processing Systems_, edited by H. Larochelle et al., vol. 33, Curran Associates, Inc., 2020, pp. 6840–6851. https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

[^nichol_improved_ddpm]: **Nichol, Alexander Quinn, and Prafulla Dhariwal.** “Improved Denoising Diffusion Probabilistic Models.” _Proceedings of the 38th International Conference on Machine Learning_, edited by Marina Meila and Tong Zhang, vol. 139, Proceedings of Machine Learning Research, 18–24 July 2021, pp. 8162–8171. PMLR. https://proceedings.mlr.press/v139/nichol21a.html.

[^mc_candlish_grad_noise]: **McCandlish, Sam, et al.** _An Empirical Model of Large-Batch Training_. arXiv, 14 Dec. 2018, https://arxiv.org/abs/1812.06162.

[^lilian_diffusion]: **Weng, Lilian.** “What Are Diffusion Models?” _Lil'Log_, 11 July 2021, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[^lilian_ae]: **Weng, Lilian.** “From Autoencoder to Beta-VAE.” _Lil'Log_, 12 Aug. 2018, https://lilianweng.github.io/posts/2018-08-12-vae/.

[^wiki_closed]: “Closed-form Expression.” _Wikipedia_, Wikimedia Foundation, https://en.wikipedia.org/wiki/Closed-form_expression.
