
x





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
![Comparison](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png)
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
$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$
{{% /admonition %}}

recall 之前计算simplification of L_VLB的时候，DDPM原论文 [^ho_ddpm] 是把这个weight 系数丢掉了，这里，OpenAI的Nichol 的论文 [^nichol_improved_ddpm] 对这里再次改进，既不去掉这个，仍然参与优化。

你的问题非常好，咱们来逐步拆解这个公式：

🧠 公式回顾

你提到的是：

$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$

这是在 DDPM 的改进版本中（如 Nichol & Dhariwal 2021）用于建模反向过程的协方差矩阵（对角形式）的参数化方式。

---

❓ 为什么是 vector？

- 这里的 **$\mathbf{v}$ 是一个向量**，通常是模型输出的一个“mixing coefficient”，用于在 $\log \beta_t$ 和 $\log \tilde{\beta}_t$ 之间做逐元素插值。
- 也就是说，模型不是输出一个标量，而是输出一个和图像维度一样的向量（比如每个像素一个值），来控制每个维度上的方差大小。

这是一种 **learned per-dimension variance interpolation**，比直接用 $\beta_t$ 或 $\tilde{\beta}_t$ 更灵活。

---

🔍 为什么用 exp？

- 因为 $\log \beta_t$ 和 $\log \tilde{\beta}_t$ 是在 log-space 做插值，最后要回到原始空间（即方差空间），所以需要用 $\exp$。
- 这是一个常见的技巧：在 log-space 插值可以避免数值不稳定（比如 $\beta_t$ 很小），而 exp 可以恢复原始尺度。

---

📉 $\beta_t$ 是不是向量？

- **不是**，$\beta_t$ 和 $\tilde{\beta}_t$ 都是标量（每个时间步一个值）。
- 但由于 $\mathbf{v}$ 是一个向量，插值结果是一个向量，所以 $\boldsymbol{\Sigma}_\theta$ 最终也是一个向量（表示对角协方差矩阵的每个元素）。

---

🧩 总结一下：

| 项目 | 类型 | 含义 |
|------|------|------|
| $\beta_t$ | scalar | forward noise schedule |
| $\tilde{\beta}_t$ | scalar | posterior variance |
| $\mathbf{v}$ | vector | 模型输出的 mixing coefficient |
| $\boldsymbol{\Sigma}_\theta$ | vector | 每个维度的预测方差（对角协方差） |

这个设计的好处是：**既保留了 $\beta_t$ 和 $\tilde{\beta}_t$ 的物理意义，又允许模型在每个维度上灵活调整方差，从而提升生成质量。**

---

如果你想深入探讨 $\mathbf{v}$ 的训练方式、它的分布、或者它在 U-Net 中是怎么被预测出来的，我可以继续展开。你也可以问我它和 $\epsilon_\theta$ 的关系，或者它在 loss function 中是否有梯度传播。你想继续往哪个方向挖掘？

{{% admonition type="quote" title="Title" open=true %}}
noisy gradients
{{% /admonition %}}

这是出自openai的论文 An Empirical Model of Large-Batch Training[^mc_candlish_grad_noise] 提出的一个指标

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

{{% admonition type="quote" title="Title" open=true %}}
Comparison of negative log-likelihood of improved DDPM with other **likelihood-based generative models**
{{% /admonition %}}

你的理解已经非常接近了，我们来系统地梳理一下：

📚 什么是“基于似然的生成模型”？

所谓 **likelihood-based generative models**，指的是那些通过最大化数据的 **log-likelihood（对数似然）** 来训练的生成模型。也就是说，它们直接建模并优化：

$$
\log p_\theta(\mathbf{x})
$$

其中 $\mathbf{x}$ 是真实数据，$p_\theta$ 是模型定义的概率分布。

---

✅ 判断标准：是否优化 log-likelihood？

你问得很关键：是不是只要优化方向是 log probability 的深度学习模型就算是“基于似然”的？

答案是：

- **基本是的**，但要注意：
  - 模型必须显式地定义了一个概率分布 $p_\theta(\mathbf{x})$
  - 并且训练目标是最大化这个分布对真实数据的 log-likelihood

---

🔍 举例说明

| 模型类型 | 是否是 likelihood-based | 说明 |
|----------|--------------------------|------|
| **VAE** | ✅ 是 | 优化 ELBO，下界是 log-likelihood |
| **Flow-based models** | ✅ 是 | 显式建模 $p_\theta(\mathbf{x})$，可精确计算 log-likelihood |
| **Diffusion models** | ✅ 是 | 优化变分下界（VLB），近似最大化 log-likelihood |
| **GANs** | ❌ 否 | 没有定义 $p_\theta(\mathbf{x})$，优化的是判别器 loss，不是 log-likelihood |
| **Score-based models** | ✅ 是 | 虽然不直接建模 $p_\theta(\mathbf{x})$，但通过 score function $\nabla_x \log p(x)$ 间接建模 |

---

🧠 衍生理解：score-based 也是似然的一种

Score-based models（如 NCSN、SDE-based diffusion）虽然不直接建模 $p(x)$，但它们建模的是：

$$
s_\theta(x) \approx \nabla_x \log p(x)
$$

这其实是对 log-likelihood 的梯度建模，也属于一种“基于似然”的方式，只不过是通过 score matching 来实现。

---

🧩 总结定义

> **基于似然的生成模型**是指那些通过最大化或近似最大化数据的 log-likelihood 来训练的模型。它们通常显式或隐式地定义了一个概率分布，并以此为基础进行采样和优化。

---

如果你想进一步探讨：

- 为什么 GAN 不属于这类模型？
- 为什么 score function 能代表概率分布？
- 或者 VAE 的 ELBO 和真实 log-likelihood 有什么差异？

我可以继续展开。你想从哪个角度深入？

## Conditioned Generation

{{% admonition type="quote" title="Title" open=true %}}
While training generative models on images with conditioning information such as ImageNet dataset, it is common to generate samples conditioned on class labels or a piece of descriptive text.
{{% /admonition %}}

其实就是今天我们常说的，文生图任务，之前的叫法很有学术味儿。

### Classifier Guided Diffusion

{{% admonition type="quote" title="Title" open=true %}}
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y)
&= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t))
\end{aligned}
$$
{{% /admonition %}}

这个是优化目标，就相当于在原本的ddpm网络头上加了一个分类器。

第一行其实就是既要训练从噪声开始到真实图像的生成能力，还要加上从条件/类别到噪声的能力。

第二行把他们都转换为了含learnable参数的形式。

这个公式就是ADM-G。

### Classifier-Free Guidance

{{% admonition type="quote" title="Title" open=true %}}
Inner content...
{{% /admonition %}}

🧠 Classifier-Free Guidance 全面总结

1️⃣ 背景与动机

传统的扩散模型引导方法（如 Classifier Guidance）依赖一个额外的分类器 \( f_\phi(y|x_t) \)，通过其梯度来引导生成过程。但这种方法存在：

- 分类器容易被 adversarial prompt 误导；
- 增加训练和推理复杂度；
- 需要额外模型参数。

**Classifier-Free Guidance** 提供了一种无需独立分类器的替代方案。

---

2️⃣ 核心思想

使用一个统一的模型 \( \epsilon_\theta(x_t, t, y) \)，通过训练时随机丢弃条件 \( y \)，让模型同时学会：

- 有条件生成：输入 \( y \)
- 无条件生成：输入 \( y = \emptyset \)

然后在推理时通过两种 score 的差值来模拟分类器梯度：

\[
\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t)
\]

近似为：

\[
\nabla_{x_t} \log p(y | x_t) \approx -\frac{1}{1 - \bar{\alpha}_t} \left( \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t) \right)
\]

最终构造引导后的 score：

\[
\bar{\epsilon}_\theta(x_t, t, y) = (1 + w) \cdot \epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t)
\]

其中 \( w \) 是引导强度。

---

3️⃣ 贝叶斯公式推导细节

你指出的非常关键的一点：

\[
\log p(y | x_t) = \log p(x_t | y) + \log p(y) - \log p(x_t)
\]

对 \( x_t \) 求导后：

\[
\nabla_{x_t} \log p(y | x_t) = \nabla_{x_t} \log p(x_t | y) - \nabla_{x_t} \log p(x_t)
\]

其中 \( \nabla_{x_t} \log p(y) = 0 \)，因为 \( y \) 与 \( x_t \) 无关，是常数项。因此原文的推导是合理的。

---

4️⃣ 模型结构与参数共享

- ✅ 只使用一个模型（一个参数集）
- ✅ 条件信息 \( y \) 通过输入控制是否存在
- ✅ 无需保留两套参数
- ✅ 节省计算资源，简化部署

训练时的策略：

- 每个 batch 中，以一定概率将 \( y \) 替换为特殊 token（如空字符串或全零向量）
- 模型学会在 \( y \) 存在与缺失两种情况下都能预测噪声

---

5️⃣ 条件输入的处理方式

- \( y = \emptyset \) 并不是“随便输入点内容”，而是明确输入一个“空条件”标记；
- 在文本任务中可以是空字符串、特殊 token；
- 在图像任务中可以是全零 embedding；
- 模型内部 embedding 层会处理这种情况。

---

 6️⃣ 条件类型的多样性

你问到是否只能训练在一种 \( y \) 上，答案是：

- ❌ 不限于一种条件；
- ✅ 可以训练在多种类别标签、文本描述、语义图等；
- 只要训练数据覆盖充分，模型就能学会在整个 \( p(y) \) 分布上进行条件生成。

---

7️⃣ 实验验证与优势

- GLIDE 模型对比了 CLIP Guidance 与 Classifier-Free Guidance；
- 发现后者更稳定，图像质量与语义一致性更好；
- 原因是 CLIP Guidance 容易被 adversarial prompt 误导，而 Classifier-Free Guidance 是从数据分布中直接建模。

---

✅ 总结表格

| 项目 | Classifier-Free Guidance |
|------|---------------------------|
| 是否需要额外分类器 | ❌ 不需要 |
| 参数数量 | ✅ 一套共享参数 |
| 条件输入处理 | ✅ 随机丢弃条件训练 |
| 是否支持多种条件类型 | ✅ 支持 |
| 推理时引导方式 | ✅ 条件与无条件 score 差值 |
| 贝叶斯公式是否完整 | ✅ 忽略常数项后是合理的 |
| 实验效果 | ✅ FID 与 IS 平衡良好 |
| 实践模型 | GLIDE、Imagen 等均采用 |

{{% admonition type="quote" title="Title" open=true %}}
Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).
{{% /admonition %}}

📊 FID（Fréchet Inception Distance）【论文：https://arxiv.org/abs/1706.08500】
✅ 定义：
FID 衡量的是生成图像与真实图像在特征空间中的分布差异。它使用 Inception 网络提取图像特征，然后计算两个高维高斯分布之间的 Fréchet 距离。

✅ 公式：
\[
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
\]
其中：

- \( \mu_r, \Sigma_r \)：真实图像的均值和协方差
- \( \mu_g, \Sigma_g \)：生成图像的均值和协方差
✅ 解读：
- FID 越低，表示生成图像与真实图像越接近；
- 既考虑图像质量，也考虑分布一致性；
- 对图像模糊、失真、模式崩溃（mode collapse）都很敏感。

🌈 IS（Inception Score）【论文：https://arxiv.org/abs/1606.03498】

✅ 定义：
IS 衡量的是生成图像的“清晰度”和“多样性”。它使用 Inception 网络预测图像类别分布，然后计算预测分布的 KL 散度。

✅ 公式：
\[
\text{IS} = \exp\left( \mathbb{E}_{x \sim p_g} \left[ D_{\text{KL}}(p(y|x) \| p(y)) \right] \right)
\]
其中：

- \( p(y|x) \)：Inception 网络对生成图像的预测分布
- \( p(y) \)：所有生成图像的平均预测分布

 ✅ 解读：

- IS 越高，表示图像清晰（预测分布熵低）且多样性高（平均分布熵高）；
- 适合评估图像的“语义清晰度”和“类别覆盖度”；
- 对图像模糊或重复生成同一类别非常敏感。

🧪 在 Classifier-Free Guidance 中的作用

- 实验表明，**适当的 guidance scale \( w \)** 可以在 FID 和 IS 之间取得良好平衡；
- 太小的 \( w \)：图像多样性高但质量差（FID 高，IS 低）；
- 太大的 \( w \)：图像质量高但容易模式崩溃（FID 低，IS 下降）；
- 所以 Classifier-Free Guidance 的优势之一就是可以**灵活调节 \( w \)** 来控制这个 trade-off。

{{% admonition type="quote" title="Title" open=true %}}
The guided diffusion model, GLIDE ([Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741)), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.
{{% /admonition %}}

GLIDE 是一种引导式扩散模型（guided diffusion model），由 Nichol、Dhariwal 和 Ramesh 等人在 2022 年提出。它尝试了两种图像生成的引导策略：

1. **CLIP guidance（CLIP 引导）**：利用 CLIP 模型的图文匹配能力来引导图像生成过程。
2. **Classifier-free guidance（无分类器引导）**：不依赖外部分类器，而是通过训练一个模型同时学习有条件和无条件的图像生成，从而实现引导。

GLIDE 的实验发现，**无分类器引导比 CLIP 引导更受欢迎**。他们的解释是：CLIP 引导可能会让生成模型“过度迎合”CLIP 模型的判断标准，甚至生成一些对 CLIP 模型“看起来很好”但实际上并不真实或合理的图像（这类图像可以被视为对 CLIP 的“对抗样本”）。换句话说，CLIP guidance 更像是在“讨好”CLIP 模型，而不是在真正优化图像与文本之间的匹配质量。

🔍 简化理解：

- 无分类器引导：模型自己学会怎么生成图像，不依赖外部判断。
- CLIP 引导：模型依赖 CLIP 的评分，但可能会“作弊”去骗过 CLIP。
- GLIDE 更偏好前者，因为它更自然、更稳健。

## Speed up Diffusion Models

### Fewer Sampling Steps & Distillation

{{% admonition type="quote" title="Title" open=true %}}
One simple way is to run a strided sampling schedule (Nichol & Dhariwal, 2021) by taking the sampling update every $\lceil T/S \rceil$ steps to reduce the process from $T$ to $S$ steps. The new sampling schedule for generation is $\{\tau_1, \dots, \tau_S\}$ where $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$ and $S < T$.
{{% /admonition %}}

PS。一定记得，只要提到sampling，就是指将训练好的模型用于推理，本节 Speed up Diffusion Models 讲的也都是如何加速模型reasoning。

这里说的很简略，只是说用个新的子序列，没有讲具体怎么sample，下面是相关推导：

非常棒的问题！我们来系统推导一下在使用子序列 \( S = \{S_1, S_2, \dots, S_K\} \) 进行加速采样时，如何重定义扩散模型中的关键参数，尤其是：

- 累积噪声因子 \( \bar{\alpha}_{S_t} \)
- 反向采样的方差 \( \tilde{\beta}_{S_t} \)
- 均值项 \( \mu_{S_t} \)

---

🧮 1. 从完整扩散过程出发

在标准 DDPM 中，正向过程定义为：

\[
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I})
\]

其中：

- \( \alpha_t = 1 - \beta_t \)
- \( \bar{\alpha}_t = \prod_{i=1}^t \alpha_i \)

由此可以得到：

\[
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
\]

---

🧭 2. 子序列采样的目标

我们希望从 \( x_{S_t} \) 直接采样到 \( x_{S_{t-1}} \)，跳过中间的时间步。由于扩散过程是马尔科夫链，我们可以构造如下的后验分布：

\[
q(x_{S_{t-1}} | x_{S_t}, x_0) = \mathcal{N}(x_{S_{t-1}}; \mu_{S_t}, \tilde{\beta}_{S_t} \mathbf{I})
\]

🧠 3. 推导均值项 \( \mu_{S_t} \)

根据 Bayes 规则和高斯合并公式，我们可以得到：

\[
\mu_{S_t} = \frac{\sqrt{\alpha_{S_t}} (1 - \bar{\alpha}_{S_{t-1}})}{1 - \bar{\alpha}_{S_t}} x_{S_t} + \frac{\sqrt{\bar{\alpha}_{S_{t-1}}} (1 - \alpha_{S_t})}{1 - \bar{\alpha}_{S_t}} x_0
\]

这个公式的推导核心在于将两个高斯分布：

- \( q(x_{S_t} | x_{S_{t-1}}) \)
- \( q(x_{S_{t-1}} | x_0) \)

合并为一个条件分布 \( q(x_{S_{t-1}} | x_{S_t}, x_0) \)，并利用高斯乘积公式完成推导。

📉 4. 推导方差项 \( \tilde{\beta}_{S_t} \)

同样地，后验方差可以推导为：

\[
\tilde{\beta}_{S_t} = \frac{(1 - \alpha_{S_t})(1 - \bar{\alpha}_{S_{t-1}})}{1 - \bar{\alpha}_{S_t}}
\]

这个表达式确保我们在跳跃采样时仍然保持正确的噪声比例。

🧪 5. 如果用 DDIM（确定性采样）

DDIM 会将采样过程改写为：

\[
x_{S_{t-1}} = \sqrt{\bar{\alpha}_{S_{t-1}}} x_0 + \sqrt{1 - \bar{\alpha}_{S_{t-1}}} \cdot \epsilon
\]

其中 \( x_0 \) 是由模型预测的，通常通过：

\[
x_0 = \frac{x_{S_t} - \sqrt{1 - \bar{\alpha}_{S_t}} \cdot \epsilon_\theta(x_{S_t}, S_t)}{\sqrt{\bar{\alpha}_{S_t}}}
\]

然后代入上式即可完成一步跳跃采样。

✅ 总结表格

| 参数 | 定义 | 作用 |
|------|------|------|
| \( \bar{\alpha}_{S_t} \) | \( \prod_{i=1}^{S_t} \alpha_i \) | 累积噪声因子 |
| \( \mu_{S_t} \) | 见上式 | 后验均值，用于采样 |
| \( \tilde{\beta}_{S_t} \) | 见上式 | 后验方差，用于采样 |
| \( x_0 \) | 由模型预测 | 用于构造采样分布 |

{{% admonition type="quote" title="Title" open=true %}}
$$
\begin{aligned}
\mathbf{x}_{t-1} 
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} & \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} & \\
&= \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t) + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{aligned}
$$
{{% /admonition %}}

这个变换的关键在于理解扩散模型中的**反向过程近似**，特别是如何将 $\mathbf{x}_0$ 和 $\boldsymbol{\epsilon}_{t-1}$ 表达为关于 $\mathbf{x}_t$ 和预测噪声 $\epsilon_\theta^{(t)}(\mathbf{x}_t)$ 的函数。我们来逐步拆解第二步的变换：

🧩 第一步：原始形式

\[
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}
\]

这是标准的前向过程采样公式，表示从 $\mathbf{x}_0$ 生成 $\mathbf{x}_{t-1}$ 的方式。

🔄 第二步：引入两个噪声项的分解

\[
\boldsymbol{\epsilon}_{t-1} = \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma_t^2}{1 - \bar{\alpha}_{t-1}}} \boldsymbol{\epsilon}_t + \sqrt{\frac{\sigma_t^2}{1 - \bar{\alpha}_{t-1}}} \boldsymbol{\epsilon}
\]

这是一个**重新参数化技巧**，将原始的高斯噪声 $\boldsymbol{\epsilon}_{t-1}$ 分解为两个部分：

- 一个是与时间步 $t$ 的噪声 $\boldsymbol{\epsilon}_t$ 相关的部分；
- 一个是独立的高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$，用于控制采样的方差 $\sigma_t^2$。

代入后得到：

\[
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon}
\]

🔁 第三步：将 $\mathbf{x}_0$ 表达为 $\mathbf{x}_t$ 和预测噪声的函数

利用前向过程的公式：

\[
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_t
\quad \Rightarrow \quad
\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}
\]

在训练时，我们用模型预测的噪声 $\epsilon_\theta^{(t)}(\mathbf{x}_t)$ 来近似真实噪声 $\boldsymbol{\epsilon}_t$，于是：

\[
\mathbf{x}_0 \approx \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta^{(t)}(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}}
\]

将这个表达式代入上一步的 $\mathbf{x}_{t-1}$ 中，就得到了第三行的形式。

📌 总结

第二步的变换是通过以下两个关键操作实现的：

1. **重新参数化噪声项**：将 $\boldsymbol{\epsilon}_{t-1}$ 拆成 $\boldsymbol{\epsilon}_t$ 和独立噪声 $\boldsymbol{\epsilon}$；
2. **用模型预测的噪声近似真实噪声**，并将 $\mathbf{x}_0$ 表达为 $\mathbf{x}_t$ 和 $\epsilon_\theta^{(t)}(\mathbf{x}_t)$ 的函数。

这一步是构造反向过程分布 $q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ 的核心，使其可以用模型预测的噪声来进行采样。


> 关于为什么那么分解

这个分解其实是一个**高斯噪声的线性组合技巧**，目的是构造一个新的高斯变量 $\boldsymbol{\epsilon}_{t-1}$，它仍然是标准正态分布，但由两个独立的高斯变量 $\boldsymbol{\epsilon}_t$ 和 $\boldsymbol{\epsilon}$ 组合而成。我们来一步步拆解这个分解的逻辑：

🎯 目标：构造一个新的高斯变量

我们希望构造一个变量：

\[
\boldsymbol{\epsilon}_{t-1} = a \boldsymbol{\epsilon}_t + b \boldsymbol{\epsilon}
\]

其中：
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
- $\boldsymbol{\epsilon}_t$ 和 $\boldsymbol{\epsilon}$ 是**独立的**

我们希望这个组合仍然满足：

\[
\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})
\]

🧮 条件：保持方差为 1

由于 $\boldsymbol{\epsilon}_t$ 和 $\boldsymbol{\epsilon}$ 独立，线性组合的方差为：

\[
\text{Var}(\boldsymbol{\epsilon}_{t-1}) = a^2 \cdot \text{Var}(\boldsymbol{\epsilon}_t) + b^2 \cdot \text{Var}(\boldsymbol{\epsilon}) = a^2 + b^2
\]

为了让 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$，我们需要：

\[
a^2 + b^2 = 1
\]

🧠 现在回到你的公式：

\[
\boldsymbol{\epsilon}_{t-1} = \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma_t^2}{1 - \bar{\alpha}_{t-1}}} \boldsymbol{\epsilon}_t + \sqrt{\frac{\sigma_t^2}{1 - \bar{\alpha}_{t-1}}} \boldsymbol{\epsilon}
\]

我们可以设：

- $a = \sqrt{\frac{1 - \bar{\alpha}_{t-1} - \sigma_t^2}{1 - \bar{\alpha}_{t-1}}}$
- $b = \sqrt{\frac{\sigma_t^2}{1 - \bar{\alpha}_{t-1}}}$

验证一下：

\[
a^2 + b^2 = \frac{1 - \bar{\alpha}_{t-1} - \sigma_t^2 + \sigma_t^2}{1 - \bar{\alpha}_{t-1}} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t-1}} = 1
\]

✅ 所以这个分解是合法的，确保 $\boldsymbol{\epsilon}_{t-1}$ 仍然是标准正态分布。

---

🔍 为什么要这么分解？

因为我们希望构造一个**可控的采样过程**，其中：

- $\boldsymbol{\epsilon}_t$ 是模型预测的噪声（可被估计）
- $\boldsymbol{\epsilon}$ 是随机采样的噪声（用于引入随机性）
- $\sigma_t^2$ 控制采样的方差（越小越接近确定性）

这种分解方式是 **DDIM（Denoising Diffusion Implicit Models）** 和 **variance-preserving sampling** 的核心技巧之一。

你完全正确：满足 $a^2 + b^2 = 1$ 的组合确实有无穷多种可能。但在扩散模型中，选择这个特定分解并不是随意的，而是**有明确的动机和物理意义**。我们来拆解一下这个选择背后的逻辑。

🎯 这个分解的目的是什么？

我们要构造一个新的噪声项 $\boldsymbol{\epsilon}_{t-1}$，它是由：
- 模型预测的噪声 $\boldsymbol{\epsilon}_t$
- 随机采样的噪声 $\boldsymbol{\epsilon}$

组合而成，同时保持 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$。

🧠 为什么选这个特定的系数？

这个分解来源于 **variance-preserving diffusion sampling** 的推导，尤其是在 DDIM 或改进的 DDPM 中。我们来看看它的几个关键动因：

1. **匹配目标分布的方差结构**

在扩散模型中，$x_{t-1}$ 的采样公式通常是：

\[
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_{t-1}
\]

为了让这个采样过程保持一致性，我们需要 $\boldsymbol{\epsilon}_{t-1}$ 的方差结构与 $\boldsymbol{\epsilon}_t$ 和 $\boldsymbol{\epsilon}$ 的组合方式精确匹配。

所以我们设定：

\[
\text{Var}(\boldsymbol{\epsilon}_{t-1}) = \frac{1 - \bar{\alpha}_{t-1} - \sigma_t^2}{1 - \bar{\alpha}_{t-1}} + \frac{\sigma_t^2}{1 - \bar{\alpha}_{t-1}} = 1
\]

这不是任意选择，而是为了让整个采样链条的方差保持一致。

1. **可控的随机性注入**

$\sigma_t^2$ 是一个**可调参数**，控制采样过程中的随机性：

- 当 $\sigma_t^2 = 0$，采样是**确定性的**（DDIM）
- 当 $\sigma_t^2 = \beta_t$，采样是**完全随机的**（DDPM）

这个分解方式允许我们在两者之间平滑过渡，形成一个**统一的采样框架**。

1. **保持 $\boldsymbol{\epsilon}_t$ 的主导性**

我们希望 $\boldsymbol{\epsilon}_{t-1}$ 主要由模型预测的 $\boldsymbol{\epsilon}_t$ 决定，而不是完全由随机噪声 $\boldsymbol{\epsilon}$ 主导。这个分解方式确保：

- 当 $\sigma_t^2$ 较小时，$\boldsymbol{\epsilon}_t$ 的权重更大
- 当 $\sigma_t^2$ 较大时，$\boldsymbol{\epsilon}$ 的权重增加

这使得采样过程既能保持模型预测的结构，又能引入必要的随机性。

🧩 总结：为什么选这个分解？

| 原因 | 解释 |
|------|------|
| 方差匹配 | 保证 $\boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(0, \mathbf{I})$ |
| 采样一致性 | 与 $x_{t-1}$ 的采样公式结构匹配 |
| 可控随机性 | 通过 $\sigma_t^2$ 调节确定性与随机性 |
| 模型主导性 | 保留 $\boldsymbol{\epsilon}_t$ 的预测能力 |

---

如果你想更深入地推导这个分解的来源，我可以从 DDPM 的逆过程公式开始，一步步推导出这个结构。或者我们可以一起看看它在代码实现中的具体位置，比如在 `noise_schedule` 或 `sample_step` 中的体现。你想往哪个方向挖掘？

完全正确，Shichao，你已经抓住了 DDIM 的核心思想，而且表达得非常清晰👏。我们可以稍微扩展一下这个理解，让它更系统化：

🧠 DDIM 的本质：**去随机化的扩散采样**

DDIM（Denoising Diffusion Implicit Models）是对 DDPM 的一种改进，它的关键点在于：

- **不再强制加入随机噪声**，而是通过模型预测的噪声直接构造下一个时刻的样本
- 这使得整个采样过程变成了**确定性的轨迹**，只要初始点 $x_T$ 固定，整个生成序列就完全可复现

---

🔄 与 DDPM 的对比

| 特性 | DDPM | DDIM |
|------|------|------|
| 是否随机 | ✅ 有随机性（每一步加入 $\epsilon$） | ❌ 可设为完全确定性 |
| 采样速度 | 慢（通常需要 1000 步） | 快（可用 50 或更少步） |
| 可控性 | 难以复现同样的轨迹 | 完全可复现 |
| 采样公式 | $x_{t-1} = \mu_\theta + \sigma_t \epsilon$ | $x_{t-1} = f_\theta(x_t, \epsilon_\theta)$，无 $\epsilon$ |

---

🎯 为什么 DDIM 更快？

因为它不依赖随机采样，每一步都是模型预测的结果，所以我们可以：

- **跳步采样**：比如从 $t=1000$ 直接跳到 $t=980, 960, \dots$
- **减少采样步数**：甚至只用 25 或 50 步就能生成清晰图像
- **加速推理**：尤其在部署时，能显著减少计算成本

---

🔧 控制随机性的方式

DDIM 实际上提供了一个参数 $\eta$ 来控制采样的随机性：

- $\eta = 0$：完全确定性（经典 DDIM）
- $\eta > 0$：引入部分随机性，介于 DDIM 和 DDPM 之间

这就像一个“采样温度旋钮”，你可以决定要多随机、要多确定。

---

如果你想，我可以给你写一个 PyTorch 的 DDIM 采样伪代码，或者展示如何从 DDPM 的公式推导出 DDIM 的采样路径。也可以一起看看 DDIM 在 latent space 或 classifier-free guidance 中的应用。你想往哪个方向深入？

### Latent Variable Space
