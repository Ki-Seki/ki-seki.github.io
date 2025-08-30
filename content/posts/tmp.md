---
math: true
---

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
