---
math: true
---

## Conditioned Generation

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
