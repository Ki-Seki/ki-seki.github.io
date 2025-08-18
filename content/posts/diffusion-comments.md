---
date: '2025-08-15T01:41:53+08:00'
title: '《What are Diffusion Models?》注释'
author:
  - Shichao Song
summary: ''
tags: []
math: true
---

<!-- TODO：记得email给lilian about this article -->
<!-- TODO：看看可以考虑重新整理下所有内容，目前看起来有点乱。 -->

本文致力于在几乎零数学背景知识和零生成模型知识的情况下，对Lilian Weng的《What are Diffusion Models?》 [^lilian_diffusion] 进行完善的注释导读。

## What are Diffusion Models?

### Forward diffusion process

{{< admonition type=quote title="前向扩散表达式" >}}
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
{{< /admonition >}}

是前向扩散过程的两种表达形式，单步扩散过程和整体扩散过程。

单步扩散过程中，

- $\mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$ 表示 $\mathbf{x}_t$ 服从 均值为 $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$，方差为 $\beta_t\mathbf{I}$ 的高斯分布。
- $\beta_t$ 是Noise variance schedule parameter，他对应一个variance schedule，$\{\beta_t \in (0, 1)\}_{t=1}^T$，和学习率调度是类似的.
- $\beta_t$ 定义了在扩散过程中每个时间步的方差大小，一般来说$\beta_t$逐渐增大，因此和原始数据差异越来越大（$\sqrt{1 - \beta_t}$ ↓），数据变异性也逐渐变大（$\beta_t\mathbf{I}$ ↑），总体上逐渐使得每一步的噪声更多。
- $\beta_t\mathbf{I}$，是协方差矩阵，也是个对角矩阵，所有对角线元素都是 $\beta_t$. 每一维都加相同强度的噪声，不偏向任何方向。

整体扩散过程只是使用马尔可夫过程性质（每一步只依赖前一步）来连乘而已。实践中因为可以使用更简单的计算方式，该公式也不常用到。

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

根据单步扩散过程$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$，以及重参数化技巧 $z = \mu + \sigma \cdot \epsilon$，我们可以重写单步扩散过程为：

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1}$$

这样就可以让我们来重写更详细的closed form的推导：

$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_{t-1}} \boldsymbol{\epsilon}_{t-2} \right) + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t} \boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$

(*) Recall that when we merge two Gaussians with different variance, $\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$
 and $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$, the new distribution is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Here the merged standard deviation is $\sqrt{\alpha_t (1-\alpha_{t-1}) + (1 - \alpha_t)} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.

{{< admonition type=quote title="Connection with stochastic gradient Langevin dynamics" >}}
$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
{{< /admonition >}}

注意：这个联系其实仅学习Diffusion的话，用不到。只是扩展地展示和Langevin dynamics的关联。这里可以类比diffusion model中重参数化后的单步扩散过程。

Langevin dynamics（朗之万动力学）是物理学中用于模拟分子运动的统计方法。它描述了粒子在势能场中运动时受到的随机扰动（比如热噪声），因此常用于建模复杂系统的随机行为。

Stochastic Gradient Langevin Dynamics（SGLD，随机梯度朗之万动力学）是将 Langevin 动力学与机器学习中的随机梯度下降（SGD）结合起来的一种采样方法。它的目标是从某个概率分布 \( p(x) \) 中采样，而不需要知道这个分布的具体形式，只需要知道它的梯度。

上面的采样公式是一个迭代式，他的含义是：“在梯度方向上前进一点，同时加入一些随机扰动，使得最终的样本分布逼近目标分布 \( p(x) \)。” 相关符号含义：

- \( \mathbf{x}_t \)：第 \( t \) 步的样本
- \( \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \): 漂移项，根据目标分布的梯度移动，类似受力牵引。
  - \( p(x) \)：目标分布的概率密度函数
  - \( \log p(x) \)：对数概率密度，便于计算和优化
  - \( \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) \)：对数概率密度的梯度，也叫 score function，表示当前点的“上升方向”
- \( \sqrt{\delta} \boldsymbol{\epsilon}_t \): 扩散项，像布朗运动的分子碰撞
  - \( \sqrt{\delta} \)：步长（step size），控制每次更新的幅度
  - \( \epsilon_t \sim \mathcal{N}(0, I) \)：标准正态分布的随机噪声，加入随机性以避免陷入局部最优

扩散模型的反向过程（从噪声恢复数据）可以看作是一个马尔可夫链，每一步都在做“去噪 + 随机扰动”，这与 SGLD 的更新方式非常相似：

- 都使用了 **score function**（即梯度）
- 都在每一步加入了 **高斯噪声**
- 都是为了从一个复杂的分布中采样

因此，扩散模型的reverse diffusion process可以被理解为一种特殊形式的 Langevin dynamics。

### Reverse diffusion process

{{< admonition type=quote title="Reverse diffusion process也是高斯分布的" >}}
Note that if \(\beta_t\) is small enough, \(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)\) will also be Gaussian.
{{< /admonition >}}

以下仅为简单理解，非严格证明。当 \(\beta_t\) 很小，意味着每一步加入的噪声很少，那么：

- \(\mathbf{x}_t\) 与 \(\mathbf{x}_{t-1}\) 的关系非常接近线性变换加微小扰动；
- 高斯分布线性变换仍然保持高斯形式。
- 这使得反向条件分布也可以近似为高斯分布。

{{< admonition type=quote title="Reverse diffusion 表达式" >}}
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
{{< /admonition >}}

对应了整体的，和单步的Reverse diffusion process。

由于我们不可能知道后验的，单步reverse diffusion process的具体形式，因此需要通过神经网络来学习。

所以这里的高斯分布的两个参数是可学习的参数$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$. 其中 $\theta$ 是神经网络的学习参数。

{{< admonition type=quote title="后验closed form" >}}
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$
{{< /admonition >}}

这里是先放了个简单的结论， ${\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0)$ and $\tilde{\beta}_t$ 是什么下文会有解释。

让我们先了解下为什么需要这个公式。下面是diffusion模型训练推理中涉及到的三个重要的分布。

| 分布                                                  | 作用                                             |
| ----------------------------------------------------- | ------------------------------------------------ |
| $q(\mathbf{x}_t \mid \mathbf{x}_0)$                   | **前向 closed-form**，直接从数据加噪得到训练样本 |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ | **真实后验 closed-form**，用于定义训练目标       |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$        | **似然**，通过训练模型去拟合上面的真实后验       |

可以看到就是从前向diffusion closed-form来得到训练样本，然后通过真实后验closed-form来得到训练目标；反向是包括参数，需要训练的，是个似然，那么这个训练是为了拟合谁呢，答案就是真实后验，$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$. 如此一来，我们就得到了最终的可以用于推理的diffusion model。

---

{{< admonition type=quote title="后验closed form推导步骤一：按bayes公式和Gaussian公式展开" >}}
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

让我们把这里的推理步骤写完善点：

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

(*) 根据高斯概率密度函数可以进行线性简化

$$
\begin{align}
p(x) 
& = \mathcal{N}(x; \mu, \sigma^2) \\
& = \frac{1}{\sqrt{2\pi\sigma^2}} \; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \\
& \propto \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\end{align}
$$

(**) $C(\mathbf{x}_t,\mathbf{x}_0)$ 里压根不包含 $\mathbf{x}_{t-1}$，那么 $C(\mathbf{x}_t,\mathbf{x}_0)$ 对于服从高斯分布的$\mathbf{x}_{t-1}$ 来说就是常数项，后面就可以直接被忽略掉。稍后你就能看到为什么会被忽略掉。

{{< admonition type=quote title="后验closed form推导步骤二：凑出新的高斯分布" >}}
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

之所以这么计算，是根据高斯概率密度函数凑出来的。由于，

$$
\mathcal{N}(p(x); \mu, \sigma^2)
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
& \propto \exp\Big( -\frac{1}{2} \big( \color{red}{\frac{1}{\tilde{\beta}_t}} \mathbf{x}_{t-1}^2 \color{blue}{- \frac{2\tilde{\boldsymbol{\mu}}_t}{\tilde{\beta}_t}} \mathbf{x}_{t-1} \color{black}{ + \frac{\tilde{\boldsymbol{\mu}}_t^2}{\tilde{\beta}_t} \big) \Big)} \\
%
& = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}_t}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
\end{align}
$$

得证。

{{< admonition type=quote title="化简后验closed form" >}}
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
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \mathbf{x}_t - \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_t}} \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_t \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) + \beta_t \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_t - \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t} \sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \\
&\neq \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$

这和原文不符合，其实也和原始的DDPM论文[^ho_ddpm]中的计算也不符。我暂时认为我是对的。

<!-- TODO: 最后回来再看看有什么其他理解的办法没有 -->

{{< admonition type=quote title="从零推导 varational lower bound">}}
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

我们先大致了解下 variational lower bound是什么：

1. 对数边际似然, $\log p_\theta(\mathbf{x}_0)$ , 很难直接算，因为无法对潜变量分布中的所有情况都进行直接积分计算。所以要找替代的优化下界，优化该下界就相当于优化对数边际似然
2. 如果想要完全了解相关概念，强烈建议阅读 Lilian Weng 的另一篇文章 From Autoencoder to Beta-VAE [^lilian_ae] 中的 [章节 VAE: Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder)。

我们已经知道了对数边际似然, $\log p_\theta(\mathbf{x}_0)$ 无法计算，Lilian这里给出了从零推导出 varational lower bound 的过程。这里推导比较清晰，不再展开。

{{< admonition type=quote title="用Jensen不等式推导 varational lower bound">}}
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

这个是另外一个推导VLB的方式，直接用了 [Jensen 不等式](https://en.wikipedia.org/wiki/Jensen%27s_inequality)：

设 \( \phi \) 是一个[concave function](https://en.wikipedia.org/wiki/Concave_function)，\( X \) 是一个可积的随机变量，则有

\[
\phi\left( \mathbb{E}[X] \right) \geq \mathbb{E}\left[ \phi(X) \right]
\]

对应于ddpm，RHS即为variational lower bound / ELBO：

$$
\log p_\theta(x) 
\geq \mathbb{E}_{q(z \mid x)} \left[ \log \frac{p_\theta(x, z)}{q(z \mid x)} \right]
$$

其中，$log()$ is a concave function。

推导也非常直观，不过让我们对部分符号进行解释。 <!-- TODO -->

{{< admonition type="quote" title="展开" >}}
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

![alt text](/posts/image.png)

上面这个图来自DDPM论文，可以把相关内容加进去

这个是目标函数，而不是loss

这组推导是对扩散模型（Diffusion Models）中的变分下界（Variational Lower Bound, VLB）或证据下界（Evidence Lower Bound, ELBO）进行逐步展开与重构的过程。它的目的，是将训练目标从一个难以直接优化的对数似然函数，转化为一组可计算的 KL 散度项与重构项，从而指导神经网络学习如何从噪声中恢复原始数据。

---

为什么要推导这个公式？

扩散模型的训练目标是最大化数据的对数似然 \( \log p_\theta(\mathbf{x}_0) \)，但由于这个目标涉及对高维隐变量的积分，无法直接计算。因此我们引入一个近似分布 \( q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \)，并通过 Jensen 不等式构造一个下界：

\[
\log p_\theta(\mathbf{x}_0) \geq \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \right] = -L_\text{VLB}
\]

这个下界就是我们训练时要最小化的损失函数。

---

用了哪些技巧？

1. **马尔可夫链展开**
利用正向过程的马尔可夫性质：
\[
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
\]
以及逆向过程的建模：
\[
p_\theta(\mathbf{x}_{0:T}) = p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
\]

1. **KL 散度重构**
将对数比值转化为 KL 散度形式：
\[
D_\text{KL}(q \parallel p) = \mathbb{E}_q \left[ \log \frac{q}{p} \right]
\]
从而将损失函数拆解为三部分：
- \( L_T \): 终点分布匹配（高斯先验）
- \( L_{t-1} \): 每一步的去噪匹配
- \( L_0 \): 最终重构项

1. **后验重构技巧**
利用：
\[
\frac{q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)}{q(\mathbf{x}_t \vert \mathbf{x}_0)} = q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
\]
这个技巧是关键，它允许我们将不可直接采样的后验 \( q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \) 显式表达为高斯分布，从而计算 KL 散度。

---

🔍 推导的意义是什么？

这套推导的意义在于：

- **理论清晰化**：将训练目标从抽象的似然最大化，转化为具体的 KL 散度项与重构项。
- **可计算性**：每一项都可以通过 Monte Carlo 采样估计，适合梯度下降优化。
- **模型设计指导**：明确了神经网络要学习的是从 \( \mathbf{x}_t \) 预测 \( \mathbf{x}_0 \) 或噪声 \( \boldsymbol{\varepsilon}_t \)，从而构建 \( p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \)。

你说得非常对，KL 散度的定义确实是：

\[
D_{\text{KL}}(q(x) \parallel p(x)) = \mathbb{E}_{q(x)} \left[ \log \frac{q(x)}{p(x)} \right]
\]

也就是说，**log 比值外面必须乘上一个期望**，而不是直接写成 log 比值本身。你指出的这个问题，正是这类推导中最容易混淆的地方之一。

---

✅ 那么原推导为什么看起来“少乘了一个期望”？

其实没有少。我们来还原一下最后一行：

\[
L_\text{VLB} = \mathbb{E}_q \left[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \right]
\]

这其实是对每一项都在 **\( \mathbb{E}_{q(\mathbf{x}_{0:T})} \)** 下取期望的写法。也就是说：

- 第一项是 \( \mathbb{E}_{q(\mathbf{x}_T \vert \mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} \right] \)，即 KL 散度项 \( D_{\text{KL}}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \)
- 第二项是对每个 \( t \) 的 KL 散度项 \( D_{\text{KL}}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)) \)
- 第三项是重构项 \( -\mathbb{E}_{q(\mathbf{x}_1 \vert \mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \right] \)

所以虽然表面上看是 log 比值，实际上每一项都隐含了在对应的 \( q \) 分布下的期望。

---

🧠 为什么这么写？

这是为了简洁地表达整个损失函数的结构。在实际实现中，每一项都会通过采样 \( \mathbf{x}_t \sim q(\cdot \vert \mathbf{x}_0) \) 来估计期望值。比如：

- \( \mathbf{x}_T \sim q(\mathbf{x}_T \vert \mathbf{x}_0) \)，用于估计 prior matching KL
- \( \mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \)，用于估计 denoising KL
- \( \mathbf{x}_1 \sim q(\mathbf{x}_1 \vert \mathbf{x}_0) \)，用于估计重构项

🔍 VLB 的具体拆解（参考 Ho et al. 2020 和 Nichol & Dhariwal 2021）

VLB 被拆成三部分：

1. **Prior Matching Term**：
   \[
   D_{\text{KL}}(q(x_T | x_0) \| p(x_T))
   \]
   约束最终加噪结果接近标准高斯。

2. **Denoising Matching Term**：
   \[
   \sum_{t=2}^{T} D_{\text{KL}}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))
   \]
   约束每一步的逆向去噪分布拟合真实分布。

3. **Reconstruction Term**：
   \[
   -\mathbb{E}_{q(x_1 | x_0)}[\log p_\theta(x_0 | x_1)]
   \]
   约束最终生成结果与原始图像接近。


{{< admonition type="quote" >}}
$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$
{{< /admonition >}}

LT 和 L0 不用计算

你问得非常到位：**为什么 \( L_T \) 和 \( L_0 \) 在训练中可以不用计算？**这其实是扩散模型训练过程中的一个重要简化策略。我们来逐项分析：

---

🧊 为什么 \( L_T = D_{\text{KL}}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \) 可以不计算？

✅ 原因一：它是一个常数项
- \( q(\mathbf{x}_T \vert \mathbf{x}_0) \) 是一个高斯分布，其均值和方差是固定的（由噪声调度决定）。
- \( p_\theta(\mathbf{x}_T) \) 是标准高斯 \( \mathcal{N}(0, I) \)，也不依赖模型参数。
- 所以它们之间的 KL 散度是一个 **解析可计算的常数**，不影响梯度优化。

> 📌 结论：**不依赖模型参数 \( \theta \)**，所以可以在训练时忽略。

---

🎯 为什么 \( L_0 = -\log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \) 可以不计算？

✅ 原因一：它被近似为 t=1 时的 denoising loss
- 原始的重构项 \( L_0 \) 是从 \( \mathbf{x}_1 \) 预测 \( \mathbf{x}_0 \)，但这项在训练中会带来较高的方差。
- 所以很多实现（如 DDPM）将其近似为：
  \[
  L_0 \approx D_{\text{KL}}(q(\mathbf{x}_0 \vert \mathbf{x}_1) \parallel p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1))
  \]
  或直接用 t=1 的 denoising KL 来代替。

✅ 原因二：它可以合并进统一的 denoising loss 框架
- 在实际训练中，我们从 \( t \sim \text{Uniform}(1, T) \) 采样一个时间步，然后优化：
  \[
  \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
  \]
- 这个统一的损失已经隐含了对所有时间步（包括 t=1）的优化。

> 📌 结论：**它被包含在统一的训练损失中，不需要单独计算。**

---

🧠 总结表格

| 项目 | 是否依赖模型参数 | 是否需要训练时计算 | 原因 |
|------|------------------|---------------------|------|
| \( L_T \) | ❌ 否 | ❌ 不需要 | 是常数项 |
| \( L_0 \) | ✅ 是 | ❌ 不需要单独计算 | 被包含在 t=1 的 denoising loss 中 |

---

如果你想，我可以带你推导一下 \( L_T \) 的具体值，或者看看如何从 \( L_0 \) 推出噪声预测损失。要不要继续深入？

### Parameterization of $L_t$ for Training Loss

{{< admonition type="quote" >}}
$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$
{{< /admonition >}}

这是为了预测 \mu 的，定义下 \mu 这块的神经网络。

{{< admonition type="quote" title="Title" open=true >}}
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

这个公式的第一步，需要recall下KL散度中对两个分布的计算，相当于转换成了

🔍 KL 散度展开（两高斯分布）

假设两者都是高斯分布：

- 真实后验：\( q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mu_q, \Sigma_q) \)
- 模型估计：\( p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mu_\theta, \Sigma_q) \)

那么 KL 散度为：

\[
L_t = \frac{1}{2} \left[ \log \frac{|\Sigma_q|}{|\Sigma_q|} - d + \text{tr}(\Sigma_q^{-1} \Sigma_q) + (\mu_q - \mu_\theta)^T \Sigma_q^{-1} (\mu_q - \mu_\theta) \right]
= \frac{1}{2 \sigma_q^2} \|\mu_q - \mu_\theta\|^2
\]


展开后其实刚好还是典型的mse error：$\text{MSE}=?$

\epsilon_\theta 符合均值为 x_t 方差为t的gaossian 分布。

最后又调用了封闭形式的forward diffusion，完整的展示了如何从原始数据产生出一个loss。

{{% admonition type="quote" title="Title" open=true %}}
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
{{% /admonition %}}

注意下标处，这里是取三个变量的联合分布。

这里就非常接近真实训练模型时候的样子了。

{{% admonition type="quote" title="Title" open=true %}}
![DDPM Algorithm](/images/DDPM_Algo.png)
{{% /admonition %}}

左边是训练，右边是推理时的。

{{% admonition type="quote" title="Connection with noise-conditioned score networks (NCSN)" open=true %}}

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600) proposed a score-based generative modeling method where samples are produced via [Langevin dynamics](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) using gradients of the data distribution estimated with score matching.

...

$$
\mathbf{s}_\theta(\mathbf{x}_t, t) 
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$
{{% /admonition %}}

目的上来看，就是用求梯度的方式，来建模 $\mu_\theta$

你问的这段关于 Noise-Conditioned Score Networks (NCSN) 的内容确实挺密的，咱们来一步步拆解一下它的核心逻辑和数学含义：

---

🧠 背景：Score-Based Generative Modeling 是什么？

Score-based 模型的目标是估计数据分布的梯度（score），即：
\[
\nabla_x \log p(x)
\]
这个梯度可以用来通过 Langevin dynamics 采样出新的数据点：
\[
x_{t+1} = x_t + \frac{\delta}{2} \nabla_x \log p(x_t) + \sqrt{\delta} \cdot \epsilon_t
\]
其中 \(\epsilon_t \sim \mathcal{N}(0, I)\)，这是一种“带噪梯度上升”的方式。

---

🧊 问题：数据集中在低维流形上怎么办？

根据 manifold hypothesis，真实数据 \(x\) 虽然在高维空间中，但其实集中在一个低维子空间上。这导致：
- 在数据密度低的区域，score 估计不准。
- Langevin dynamics 可能会“走偏”，因为梯度估计不可靠。

---

💡 解决方案：加入噪声 + 多尺度训练

Song & Ermon 提出：
1. **加入不同强度的高斯噪声**：让数据分布变得更“满”，覆盖整个空间。
2. **训练一个 Noise-Conditioned Score Network**：记作 \(s_\theta(x, \sigma)\)，它能估计不同噪声水平下的 score：
   \[
   s_\theta(x, \sigma) \approx \nabla_x \log p_\sigma(x)
   \]
   其中 \(p_\sigma(x)\) 是加入噪声后的数据分布。

---

🔁 与扩散模型的联系：Forward Process 类似加噪过程

这个“逐步加噪”的过程和扩散模型中的 forward diffusion 是一样的：
\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
\]
最终 \(x_T\) 会变成一个纯高斯噪声。

---

📐 Score 与高斯分布的梯度关系

对于高斯分布：
\[
p(x) = \mathcal{N}(x; \mu, \sigma^2 I)
\]
其 score 是：
\[
\nabla_x \log p(x) = -\frac{1}{\sigma^2}(x - \mu)
\]
所以在扩散模型中，score network 其实是在估计：
\[
\nabla_x \log p_\sigma(x) = -\frac{x - \mu}{\sigma^2}
\]
而这个 \(\mu\) 可以是原始数据 \(x_0\)，也可以是神经网络预测的值。

---

🧪 总结一下这段话的意思：

- 原始 score-based 模型用 Langevin dynamics 来采样，但在高维空间中不稳定。
- 加入噪声后训练 score network 更稳定。
- 多尺度噪声训练（noise-conditioned）让模型能适应不同的噪声水平。
- 这个过程和扩散模型的 forward diffusion 是一致的。
- 最终，score network 估计的是加入噪声后的数据分布的梯度。

---

如果你想，我可以把这部分用图示或公式推导再展开一层，或者对比一下 DDPM 和 NCSN 的异同。你更想深入哪一块？

你的问题非常好，我们来逐个解释这些概念和公式，帮助你建立清晰的直觉：

---

🧭 什么是 \(\mathbb{R}^D\) 空间？

这是一个 **D维欧几里得空间**，也就是所有长度为 \(D\) 的实数向量组成的空间。比如：
- \(\mathbb{R}^2\)：二维平面
- \(\mathbb{R}^3\)：三维空间
- \(\mathbb{R}^{512}\)：比如图像的潜在表示空间

在扩散模型中，数据（如图像）被表示为高维向量，通常就在 \(\mathbb{R}^D\) 中。

---

🧮 为什么要计算梯度 \(\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)\)？

这是所谓的 **score function**，表示在某个点 \(\mathbf{x}_t\) 上，数据分布的对数密度的梯度。它的作用是：
- 指出“数据分布上升最快的方向”
- 可以用来进行 **Langevin dynamics** 采样
- 在扩散模型中，它帮助我们从噪声中“走回”真实数据分布

---

🔁 要不要算 \(\mathbf{x}_{t-1}\)？

是的，扩散模型的目标就是从一个高斯噪声 \(\mathbf{x}_T\) 开始，逐步去噪得到 \(\mathbf{x}_0\)。每一步都要估计：
\[
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
\]
这个分布通常建模为高斯分布，其均值由神经网络预测。

---

📦 什么是 \(q(\tilde{\mathbf{x}} \vert \mathbf{x})\)？

这是一个 **加噪过程的条件分布**，表示在原始数据 \(\mathbf{x}\) 上加噪后得到 \(\tilde{\mathbf{x}}\) 的概率。比如：
\[
q(\tilde{\mathbf{x}} \vert \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 I)
\]
在 NCSN 中，这个分布用于训练 score network 来估计加噪数据的 score。

---

📐 这个公式是 score function 的定义吗？

你写的这组公式：

\[
\mathbf{s}_\theta(\mathbf{x}_t, t) 
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
\]

是 **扩散模型中 score function 的近似表达式**，其中：
- \(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\) 是神经网络预测的噪声
- \(\bar{\alpha}_t = \prod_{s=1}^t \alpha_s\)，是前向过程的累计衰减因子
- 最后一行是因为我们用 \(\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}\) 来生成 \(\mathbf{x}_t\)，所以可以反推 score

---

🧪 那训练时怎么训练？真实的 score 有吗？

关键点是：**真实的 score 没有显式表达式**，但我们可以通过构造损失函数来间接训练 score network。

在 DDPM 或 NCSN 中，训练目标是：
\[
\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]
\]
也就是说：
- 我们知道 \(\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}\)
- 所以我们知道真实的 \(\boldsymbol{\epsilon}\)
- 训练目标就是让网络预测的 \(\boldsymbol{\epsilon}_\theta\) 尽量接近真实的 \(\boldsymbol{\epsilon}\)

这就间接地训练了 score function，因为：
\[
\mathbf{s}_\theta(\mathbf{x}_t, t) \approx - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
\]

---

如果你想，我可以把这个过程画成图，或者用代码形式展示训练过程。你更想看哪种形式？

### Parameterization of $\beta_t$

{{% admonition type="quote" title="Title" open=true %}}
Diffusion models in their experiments showed high-quality samples but still could not achieve competitive **model log-likelihood** as other generative models.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed several improvement techniques to help diffusion models to obtain lower **NLL**.
{{% /admonition %}}

这里提到的“model log-likelihood”（正方向目标），“NLL”（负方向目标）实际上就是我们想要优化的目标。

其计算就是靠刚刚我们见到的的L_VLB。然后计算和训练的时候的采样一样，同样需要多次进行采样，来估计NLL，大致如下：

当然，我们来逐项讲清楚扩散模型中三项 VLB 的 Monte Carlo 估计方式：

1️⃣ Prior Matching Term：KL(q(x_T | x₀) || p(x_T))

这是对最终加噪结果是否接近标准高斯的约束。

- **计算方式**：  
 由于 \( q(x_T | x_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_T} x_0, (1 - \bar{\alpha}_T) I) \)，而 \( p(x_T) \sim \mathcal{N}(0, I) \)，两个都是高斯分布，KL 散度可以解析计算：

  \[
  D_{\text{KL}} = \frac{1}{2} \left[ \text{tr}(\Sigma_p^{-1} \Sigma_q) + (\mu_p - \mu_q)^T \Sigma_p^{-1} (\mu_p - \mu_q) - d + \log \frac{|\Sigma_p|}{|\Sigma_q|} \right]
  \]

  实际中直接代入均值和方差即可，不需要采样。

2️⃣ Denoising Matching Term：KL(q(x_{t-1} | x_t, x₀) || p_θ(x_{t-1} | x_t))

这是最核心的一项，约束模型预测的去噪分布是否接近真实分布。

- **计算方式**：
  对每个时间步 \( t \)，我们采样：

  \[
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
  \]

  然后模型预测 \( \hat{\epsilon}_\theta(x_t, t) \)，我们用它恢复出模型的均值：

  \[
  \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon}_\theta \right)
  \]

  而真实均值是：

  \[
  \mu_q(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)
  \]

  两者之间的 KL 散度简化为均值差的平方除以方差：

  \[
  \mathcal{L}_t = \frac{1}{2 \sigma_t^2} \| \mu_q - \mu_\theta \|^2 \propto \| \epsilon - \hat{\epsilon}_\theta(x_t, t) \|^2
  \]

  所以训练时只需采样 \( x_t \)，计算模型预测的噪声与真实噪声之间的 MSE。

3️⃣ Reconstruction Term：−E_{q(x₁ | x₀)} [log p_θ(x₀ | x₁)]

这是对最终生成结果是否接近原图的约束。

- **计算方式**：
  采样 \( x_1 \sim q(x_1 | x_0) \)，然后模型预测 \( p_\theta(x_0 | x_1) \)，通常建模为高斯分布：

  \[
  p_\theta(x_0 | x_1) = \mathcal{N}(x_0; \mu_\theta(x_1), \sigma^2 I)
  \]

  然后计算 log-likelihood：

  \[
  \mathcal{L}_0 = -\log p_\theta(x_0 | x_1) \propto \| x_0 - \mu_\theta(x_1) \|^2
  \]

  实践中这项可以近似为 t=1 时的 MSE 损失。

---

如果你想我帮你写出 PyTorch 代码来估计这三项，或者推导某一项的 KL 散度公式，我可以继续展开。你对哪一项最感兴趣？

{{% admonition type="quote" title="Comparison of linear and cosine-based scheduling of $\beta_t$ during training" open=true %}}
![](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png)
{{% /admonition %}}

在 **linear variance schedule** 中，我们定义每一步的噪声强度为：

\[
\beta_t = \beta_{\text{min}} + \frac{t - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})
\]

其中：
- \( \beta_{\text{min}} \) 和 \( \beta_{\text{max}} \) 是预设的最小和最大噪声值（例如 0.0001 和 0.02）
- \( T \) 是总的扩散步数（例如 1000）

---

🔁 闭式形式的 \( \bar{\alpha}_t \)

我们定义：

\[
\alpha_t = 1 - \beta_t
\quad \text{and} \quad
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
\]

由于 \( \beta_t \) 是线性递增的，\( \alpha_t \) 是线性递减的，因此 \( \bar{\alpha}_t \) 是一连串乘积，虽然不能简化为一个完全闭式表达，但可以写成：

\[
\bar{\alpha}_t = \prod_{s=1}^t \left(1 - \beta_{\text{min}} - \frac{s - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})\right)
\]

这个表达式是 **显式的 closed-form**，但仍然是一个乘积形式。在实际实现中，通常会预先计算所有 \( \bar{\alpha}_t \) 并缓存下来。

---

🧠 进一步简化（近似）

如果你希望得到一个近似闭式表达，可以考虑将乘积转换为指数形式：

\[
\log \bar{\alpha}_t = \sum_{s=1}^t \log \alpha_s
\quad \Rightarrow \quad
\bar{\alpha}_t = \exp\left( \sum_{s=1}^t \log(1 - \beta_s) \right)
\]

这在数值计算中更稳定，也更容易处理。

---

如果你想我帮你用 Python 或 PyTorch 写出这个 linear schedule 的初始化代码，我可以直接给你模板。或者你想比较它和 cosine schedule 的图像，我也可以画出来。你想继续哪一方向？

### Parameterization of reverse process variance $\boldsymbol{\Sigma}_\theta$

{{% admonition type="quote" title="Title" open=true %}}
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
{{% /admonition %}}

recall 之前计算simplification of L_VLB的时候，DDPM原论文 [^ho_ddpm] 是把这个weight 系数丢掉了，这里，OpenAI的Nichol 的论文 [^nichol_improved_ddpm] 对这里再次改进，既不去掉这个，仍然参与优化。

{{% admonition type="quote" title="Title" open=true %}}
noisy gradients
{{% /admonition %}}

这是出自openai的论文 An Empirical Model of Large-Batch Training[^mccandlish_grad_noise] 提出的一个指标

Gradient Noise Scale（梯度噪声尺度）是一个用于衡量优化过程中梯度稳定性的统计指标，尤其在深度学习中用于估计**最优批量大小（optimal batch size）**。

---

🌪️ 定义与直觉

在随机梯度下降（SGD）中，我们不是用整个数据集计算梯度，而是用一个小批量（mini-batch）。这会引入噪声，因为不同批次的梯度可能差异很大。

**Gradient Noise Scale**衡量的就是这种梯度的波动性。它的核心思想是：

> 如果梯度在不同批次之间变化很大（噪声高），我们需要更大的批量来获得更稳定的更新。

---

📐 数学表达

在简化假设下（如 Hessian 是单位矩阵的倍数），Gradient Noise Scale 可以表示为：

\[
B_{\text{simple}} = \frac{\text{tr}(\Sigma)}{|G|^2}
\]

其中：

- \(\text{tr}(\Sigma)\)：梯度协方差矩阵的迹，即所有梯度分量的方差之和。
- \(|G|^2\)：梯度的平方范数（global gradient norm）。

这个比值表示：**梯度的噪声强度相对于其平均强度的比例**。

---

🧠 实际意义

- 如果 \(B_{\text{simple}}\) 很大，说明梯度噪声很强，建议使用更大的 batch size。
- 如果它很小，说明梯度稳定，可以用较小的 batch size，加快训练。

---

🔍 应用场景

- 自动调整 batch size（如在 Torch-Foresight 中使用）
- 分析数据集复杂度：高噪声可能意味着数据分布复杂或模型不稳定
- 优化训练效率：在资源受限时找到最合适的 batch size

{{% admonition type="quote" title="Title" open=true %}}
time-averaging smoothed version of $L_\text{VLB}$ with importance sampling.
{{% /admonition %}}

根据Improved DDPM [^nichol_improved_ddpm]，这里的公式是：


![alt text](/posts/image-1.png)

核心动机就是不同的t对应的L贡献度不同，想要消解掉magnitude的差异。

![alt text](/posts/image-2.png)

## Conditioned Generation

{{% admonition type="quote" title="Title" open=true %}}
While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.
{{% /admonition %}}

其实就是今天我们常说的，文生图任务，之前的叫法很有学术味儿。

### Classifier Guided Diffusion

## Appendix

这里汇总了要想更完整了解整个diffusion models的内容需要的小的基础知识点。

### Notations

- $\beta_t$ 是Noise variance schedule parameter，他对应一个variance schedule，$\{\beta_t \in (0, 1)\}_{t=1}^T$，和学习率调度是类似的.

### 重要的diffusion相关的论文

[1] Jascha Sohl-Dickstein et al. “Deep Unsupervised Learning using Nonequilibrium Thermodynamics.” ICML 2015.

[2] Max Welling & Yee Whye Teh. “Bayesian learning via stochastic gradient langevin dynamics.” ICML 2011.

[3] Yang Song & Stefano Ermon. “Generative modeling by estimating gradients of the data distribution.” NeurIPS 2019.

[4] Yang Song & Stefano Ermon. “Improved techniques for training score-based generative models.” NeuriPS 2020.

[5] Jonathan Ho et al. “Denoising diffusion probabilistic models.” arxiv Preprint arxiv:2006.11239 (2020). [code]

[6] Jiaming Song et al. “Denoising diffusion implicit models.” arxiv Preprint arxiv:2010.02502 (2020). [code]

[7] Alex Nichol & Prafulla Dhariwal. “Improved denoising diffusion probabilistic models” arxiv Preprint arxiv:2102.09672 (2021). [code]

[8] Prafula Dhariwal & Alex Nichol. “Diffusion Models Beat GANs on Image Synthesis.” arxiv Preprint arxiv:2105.05233 (2021). [code]

[9] Jonathan Ho & Tim Salimans. “Classifier-Free Diffusion Guidance.” NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

[10] Yang Song, et al. “Score-Based Generative Modeling through Stochastic Differential Equations.” ICLR 2021.

[11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. “GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.” ICML 2022.

[12] Jonathan Ho, et al. “Cascaded diffusion models for high fidelity image generation.” J. Mach. Learn. Res. 23 (2022): 47-1.

[13] Aditya Ramesh et al. “Hierarchical Text-Conditional Image Generation with CLIP Latents.” arxiv Preprint arxiv:2204.06125 (2022).

[14] Chitwan Saharia & William Chan, et al. “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.” arxiv Preprint arxiv:2205.11487 (2022).

[15] Rombach & Blattmann, et al. “High-Resolution Image Synthesis with Latent Diffusion Models.” CVPR 2022.code

[16] Song et al. “Consistency Models” arxiv Preprint arxiv:2303.01469 (2023)

[17] Salimans & Ho. “Progressive Distillation for Fast Sampling of Diffusion Models” ICLR 2022.

[18] Ronneberger, et al. “U-Net: Convolutional Networks for Biomedical Image Segmentation” MICCAI 2015.

[19] Peebles & Xie. “Scalable diffusion models with transformers.” ICCV 2023.

[20] Zhang et al. “Adding Conditional Control to Text-to-Image Diffusion Models.” arxiv Preprint arxiv:2302.05543 (2023).

### GAN, VAE, and Flow-based models 是什么

![Generative Models](/images/Generative_Models.png)

- GAN 生成对抗网络：训练两个网络，一个用于生成图像，一个用于判别图像的真伪
- VAE 变分自编码器模型：通过编码器将输入图像压缩为潜在空间变量，再通过解码器重建图像
- Flow matching models：通过流动匹配的方法生成图像，是函数级别的过程组合起来生成图像的。

### VAE 与 AE

VAE（变分自编码器）的前身是AE（自编码器），就是一个具有编码器和解码器的神经网络，目的是通过让解码器的值和原始值尽可能相似，而学习一个压缩了的潜在变量，用于表示学习和降维。

VAE和AE在结构上非常相似，但在理论基础和目标函数上有本质区别。

| 特性                 | 自编码器（AE）                  | 变分自编码器（VAE）                                            |
| -------------------- | ------------------------------- | -------------------------------------------------------------- |
| 编码器输出           | 一个确定性的向量 \( z = f(x) \) | 一个分布 \( q(z \vert x) = \mathcal{N}(\mu(x), \sigma^2(x)) \) |
| 解码器输入           | 固定向量 \( z \)                | 从分布中采样的 \( z \sim q(z \vert x) \)                       |
| 训练目标             | 最小化重构误差（如 MSE）        | 最大化变分下界（ELBO）                                         |
| 是否为生成模型       | 否                              | 是                                                             |
| 是否有概率建模       | 否                              | 有（对潜在变量建模）                                           |
| 是否可采样生成新数据 | 否                              | 可从先验 \( p(z) \) 采样生成数据                               |
| 是否使用KL散度       | 否                              | 用于正则化潜在分布                                             |

### Diffusion中 数据样本的记法

$\mathbf{x}_0 \sim q(\mathbf{x})$ 表示从真实数据分布 $q(\mathbf{x})$ 中采样得到的样本 $\mathbf{x}_0$，其中

- $\mathbf{x}_0$：表示一个真实数据样本，比如一张图像、一段语音或一个文本向量。是一个向量（例如图像的像素向量、文本的嵌入向量等），维度可能是几百甚至几千.
- $q(\mathbf{x})$：表示真实数据的分布，也叫经验分布，比如训练集中的图像分布。

### Gaussian distribution

高斯分布（Gaussian distribution）也被称为**正态分布**，$\mathcal{N}(\mu, \sigma)$，其概率密度函数（PDF, Probability Density Function）为：

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

* $\mu$：均值（mean），决定分布的中心位置
* $\sigma$：标准差（standard deviation），决定分布的宽度
* $\sigma^2$：方差（variance）
* $\exp(\cdot)$：自然指数函数 $e^x$

其累积分布函数（CDF, Cumulative Distribution Function）为：

$$
F(x) = P(X \le x) = \frac{1}{2} \left[ 1 + \operatorname{erf} \!\left( \frac{x - \mu}{\sigma\sqrt{2}} \right) \right]
$$

* $\operatorname{erf}(\cdot)$：误差函数（error function），是无法用初等函数表示的积分函数，定义为

$$
\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} \, dt
$$

此外，isotropic Gaussian distribution是指各方向都均匀的高斯分布，即向量中的每个分量都符合 $\mathcal{N}(0, \mathbf{I})$。

### 扩散模型中对分布的记法

扩散模型相关论文更倾向于这么写概率，样本，和分布间的关系：

$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

它等价于概率论中常见的表达形式：

$$
\begin{align}
\mathbf{x}_t &\sim \mathcal{N}(\sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= f(\mathbf{x}_t) 
\end{align}
$$

其中，$f(\cdot)$ 是 Gaussian Probability Density Function

### Closed-form expression

> In mathematics, an expression or formula (including equations and inequalities) is in closed form if it is formed with constants, variables, and a set of functions considered as basic and connected by arithmetic operations (+, −, ×, /, and integer powers) and function composition.
>
> — <cite>Wikipedia [^wiki_closed]</cite>

简单来说，就是可以用有限的、明确的数学表达式直接写出来解，不需要迭代、数值近似或求解方程。

### reparameterization trick

定义：重参数化**将随机变量从不可导的采样操作中解耦出来**的方法，让采样操作可以参与梯度下降优化。

原理：他没有消除随机采样，只是将随机采样对梯度传播的影响降到了最低.

举例：如果你有一个随机变量 $z \sim \mathcal{N}(\mu, \sigma^2)$，直接从这个分布采样，梯度无法通过 $\mu, \sigma$ 传播。那就可以按照下式从随机采样 $z$ 转换为随机采样 $\epsilon$。

$$
\mathcal{N}(z; \mu, \sigma^2)
= \mu + \mathcal{N}(\epsilon'; 0, \sigma^2)
= \mu + \sigma \cdot \mathcal{N}(\epsilon; 0, 1)
$$

{{< details "PyTorch代码示例">}}
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
{{< /details >}}

### 扩散模型前向和反向的记法

| 方向 | 概率密度                                        | 噪声 | 作用                             | 概率类型 |
| ---- | ----------------------------------------------- | ---- | -------------------------------- | -------- |
| 前向 | $q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$        | 加噪 | 构造高斯马尔可夫链，逐步破坏数据 | 真实分布 |
| 反向 | $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ | 去噪 | 恢复数据，从噪声生成样本         | 近似后验     |

### 先验，似然，与后验

| 概念 | 类比解释                 | 在扩散模型中的类比                                     |
| ---- | ------------------------ | ------------------------------------------------------ |
| 先验 | 在观察任何数据之前对变量的假设分布             | $q(\mathbf{x}_t) \sim \mathcal{N}(0, I)$               |
| 似然 | 某假设/模型参数下，观测数据出现的概率 | $p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$               |
| 后验 | 结合先验与观测数据之后对潜在变量的更新分布     | $q(\mathbf{x}_{t-1}\mid \mathbf{x}_t, \mathbf{x}_0)$ |

三者之间的关系是：

\[
\text{后验} = \frac{\text{似然} \times \text{先验}}{\text{证据}}
\quad\leftrightarrow\quad
p(\text{参数} | \text{数据}) = \frac{p(\text{数据} | \text{参数}) \cdot p(\text{参数})}{p(\text{数据})}
\]

### Bayes’ rule

贝叶斯公式（Bayes’ Rule）是概率论中的一个核心法则，用于在已知条件下更新事件的概率。它的基本形式是：

\[
P(A \vert B) = \frac{P(B \vert A) \cdot P(A)}{P(B)}
\]

其中：

- \(P(A)\)：事件 A 的先验概率（在观察 B 之前对 A 的信念）
- \(P(B \vert A)\)：在 A 发生的前提下，观察到 B 的可能性（似然）
- \(P(B)\)：事件 B 的边际概率（所有可能情况下 B 发生的概率）
- \(P(A \vert B)\)：在观察到 B 之后，事件 A 的后验概率（更新后的信念）

### 联合分布，边缘分布和条件分布

- 联合分布 $P(A, B)$：全景地图（包含所有组合的概率）。
- 边缘分布 $P(B)$：全景地图投影到某一个轴。
- 条件分布 $P(A \vert B)$：全景地图切一条线（已知另一变量的值），看这条线上的概率分布。公式为：$P(A \vert B) = \frac{P(A, B)}{P(B)}$

### KL 散度

KL 散度（Kullback–Leibler Divergence），也叫相对熵（Relative Entropy），它用来衡量 **两个概率分布之间差异** 的一种信息论度量。

$$
\begin{align}
\text{对于离散分布:}\quad & D_{\mathrm{KL}}(P \,\|\, Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \\
\text{对于连续分布:}\quad & D_{\mathrm{KL}}(P \,\|\, Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
\end{align}
$$

TODO：添加个转换为期望形式的表达方式

这里：

- $P$ 是“真实”分布（或目标分布）。
- $Q$ 是用来近似 $P$ 的分布。
- $\log$ 的底通常取自然对数（信息单位为 nats），取 2 则单位为 bits。

直观理解:

- 如果你用 $Q$ 作为编码策略去编码实际上来自 $P$ 的数据，那么平均每个样本会多花 $D_{\mathrm{KL}}(P\|Q)$ 个信息单位（nats/bits）。
- KL 散度表示使用分布 $Q$ 来替代 $P$ 时丢失的信息量。
- 公式里的 $\log \frac{P(x)}{Q(x)}$ 是 **对数概率比**，乘上 $P(x)$ 并取期望，就是平均的概率比差异。

性质:

- **非负性**: $D_{\mathrm{KL}}(P \,\|\, Q) \ge 0$
- **非对称性**: $D_{\mathrm{KL}}(P \,\|\, Q) \neq D_{\mathrm{KL}}(Q \,\|\, P)$
- **相对熵=交叉熵-香农熵**: $D_{\mathrm{KL}}(P \,\|\, Q) = H(P, Q) - H(P)$

### 几个重要的熵

| 熵类型     | 公式                                                      | 解释                                   |
| ---------- | --------------------------------------------------------- | -------------------------------------- |
| **香农熵** | $H(p) = -\sum_i p(x_i) \log p(x_i)$                       | 衡量分布 $p$ 的不确定性                |
| **交叉熵** | $H(p,q) = -\sum_i p(x_i) \log q(x_i)$                     | 衡量用 $q$ 表示 $p$ 的平均信息量       |
| **相对熵** | $D_{KL}(p\|q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$ | 衡量 $p$ 和 $q$ 的差异，多付出的信息量 |

## Citation

{{< bibtex >}}

## References

[^ho_ddpm]: **Ho, Jonathan, Ajay Jain, and Pieter Abbeel.** “Denoising Diffusion Probabilistic Models.” _Advances in Neural Information Processing Systems_, edited by H. Larochelle et al., vol. 33, Curran Associates, Inc., 2020, pp. 6840–6851. https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

[^nichol_improved_ddpm]: **Nichol, Alexander Quinn, and Prafulla Dhariwal.** “Improved Denoising Diffusion Probabilistic Models.” _Proceedings of the 38th International Conference on Machine Learning_, edited by Marina Meila and Tong Zhang, vol. 139, Proceedings of Machine Learning Research, 18–24 July 2021, pp. 8162–8171. PMLR. https://proceedings.mlr.press/v139/nichol21a.html.

[^mccandlish_grad_noise]: **McCandlish, Sam, et al.** _An Empirical Model of Large-Batch Training_. arXiv, 14 Dec. 2018, https://arxiv.org/abs/1812.06162.

[^lilian_diffusion]: **Weng, Lilian.** “What Are Diffusion Models?” _Lil'Log_, 11 July 2021, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[^lilian_ae]: **Weng, Lilian.** “From Autoencoder to Beta-VAE.” _Lil'Log_, 12 Aug. 2018, https://lilianweng.github.io/posts/2018-08-12-vae/.

[^wiki_closed]: “Closed-form Expression.” _Wikipedia_, Wikimedia Foundation, https://en.wikipedia.org/wiki/Closed-form_expression.
