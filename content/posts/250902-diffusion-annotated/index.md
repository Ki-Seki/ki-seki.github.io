---
date: '2025-09-02T18:49:24+08:00'
title: 'Fully Annotated Guide to "What are Diffusion Models?"'
author:
  - Shichao Song
tags: ["vision", "diffusion", "math"]
math: true
---

This article provides a comprehensive annotated guide to Lilian Weng's "What are Diffusion Models?" [^lilian_diffusion].

When writing this article, I had limited knowledge of image generation models and related mathematical background. If you have a similar background, this article should be suitable for you. Of course, I may have made some elementary mistakes due to this. Your corrections are welcome.

The structure of this article is largely consistent with the original:

- Key formulas and concepts in each section will be expanded with derivations or explanations.
- In addition, common symbol explanations are provided at the beginning of the article.
- And some background knowledge required to understand the original text is appended at the end.

Note: If you only want to learn the basics of the Diffusion model, the first section, [*What are Diffusion Models?*]({{< relref "#what-are-diffusion-models" >}}) is sufficient.

### Notations

<table>
  <thead>
    <tr>
      <th>Category / Symbol</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2"><b>Samples and Distributions</b></td>
    </tr>
    <tr>
      <td>$\mathbf{x}_0$</td>
      <td>A real data sample, such as an image, speech, or text. It is a vector (e.g., a pixel vector of an image, an embedding vector of text, etc.), and its dimension can be hundreds or even thousands.</td>
    </tr>
    <tr>
      <td>$\mathbf{x}_t, \, t = 1, 2, ..., T$</td>
      <td>The result of gradually adding noise to the data sample $\mathbf{x}_0$. Eventually, $\mathbf{x}_T$ is a pure noise sample.</td>
    </tr>
    <tr>
      <td>$q(\mathbf{x})$</td>
      <td>For convenience, $q(\mathbf{x})$ can represent either a probability density function (PDF) or the distribution corresponding to this PDF. Here, $q(\mathbf{x})$ is the distribution of real data, also known as the empirical distribution, such as the image distribution in the training set.</td>
    </tr>
    <tr>
      <td>$\mathbf{x}_0 \sim q(\mathbf{x})$</td>
      <td>A sample $\mathbf{x}_0$ sampled from the real data distribution $q(\mathbf{x})$.</td>
    </tr>
    <tr>
      <td>$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$</td>
      <td>Here, $q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$ is a probability density function. $\mathbf{x}_t$ follows a normal distribution $\mathcal{N}(\sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$, and $q(\mathbf{x}_t \vert \mathbf{x}_{t-1})=f(\mathbf{x}_t)$, where $f(\cdot)$ is the probability density function.</td>
    </tr>
    <tr>
      <td colspan="2"><b>Noise Hyperparameters</b></td>
    </tr>
    <tr>
      <td>$\beta_t$</td>
      <td>Noise variance schedule parameter. It is a hyperparameter corresponding to a variance schedule $\{\beta_t \in (0, 1)\}_{t=1}^T$, similar to the learning rate schedule.</td>
    </tr>
    <tr>
      <td>$\alpha_t$</td>
      <td>$\alpha_t = 1 - \beta_t$, a symbol introduced for the convenience of writing formulas.</td>
    </tr>
    <tr>
      <td>$\bar{\alpha}_t$</td>
      <td>$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, a symbol introduced for the convenience of writing formulas.</td>
    </tr>
    <tr>
      <td colspan="2"><b>Diffusion Process</b></td>
    </tr>
    <tr>
      <td>$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$</td>
      <td><b>Forward diffusion process recurrence formula</b>. Construct a Gaussian Markov chain to gradually add noise and destroy the data.</td>
    </tr>
    <tr>
      <td>$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$</td>
      <td><b>Forward diffusion closed-form</b>. Obtain the final noisy sample directly from the data in one step.</td>
    </tr>
    <tr>
      <td>$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$</td>
      <td><b>Reverse diffusion posterior</b>. Used to define the training target.</td>
    </tr>
    <tr>
      <td>$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$</td>
      <td><b>Reverse diffusion likelihood</b>. Train a model to fit the above posterior.</td>
    </tr>
    <tr>
      <td>$q(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; 0, I)$</td>
      <td><b>Prior</b>. Fixed as a Gaussian distribution $\mathcal{N}(0, I)$, directly sampled as the starting point during inference.</td>
    </tr>
  </tbody>
</table>

## What are Diffusion Models?

The basic principle of diffusion models is to increase noise through forward diffusion to obtain samples following a pure Gaussian distribution. Train a model to mimic the reverse diffusion process so that it can recover real data samples from any Gaussian noise samples.

### Forward diffusion process

{{% admonition type="quote" title="Forward Diffusion Expressions" open=true %}}
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
{{% /admonition %}}

These are two expressions of the forward diffusion process: the single-step diffusion process and the overall diffusion process.

In **the single-step diffusion process**:

- $\mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$ indicates that $\mathbf{x}_t$ follows a Gaussian distribution with a mean of $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$ and a variance of $\beta_t\mathbf{I}$.
- $\beta_t$ is the noise variance schedule parameter. It corresponds to a variance schedule $\{\beta_t \in (0, 1)\}_{t=1}^T$, similar to the learning rate schedule.
- $\beta_t$ defines the variance at each time step in the diffusion process. Generally, $\beta_t$ increases gradually. As a result, the difference from the original data becomes larger ($\sqrt{1 - \beta_t}$ decreases), the data variability also increases ($\beta_t\mathbf{I}$ increases), and overall, more noise is added at each step.
- $\beta_t\mathbf{I}$ is the covariance matrix, which is also a diagonal matrix. All diagonal elements are $\beta_t$. The same intensity of noise is added to each dimension.

The **overall diffusion process** is a recurrence formula obtained by multiplying the single-step diffusion processes together according to the [Markov property]({{< relref "#markov-property" >}}).

The overall diffusion process is necessary because it helps us quickly sample the final pure noise $\mathbf{x}_T$ from the real data distribution. However, it relies on the recurrence formula and is computationally expensive. Therefore, in practice, a simpler calculation method, i.e., the closed-form formula described below, is used.

{{< admonition type=quote title="Closed-Form Forward Diffusion " >}}
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

A closed-form expression refers to a formula that can be directly written as a solution using a finite number of explicit mathematical expressions without iteration, numerical approximation, or solving equations [^wiki_closed].

Based on the single-step diffusion process $q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I})$ and the [reparameterization trick]({{< relref "#reparameterization-trick" >}}) $z = \mu + \sigma \cdot \epsilon$, we can rewrite the single-step diffusion process as:

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1}$$

This allows us to provide a more detailed derivation of the closed-form:

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

(*) Recall that when we merge two Gaussians with different variances, $\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$, the new distribution is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Here, the merged standard deviation is $\sqrt{\alpha_t (1-\alpha_{t-1}) + (1 - \alpha_t)} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.

---

PS. It's worth noting that $\mathbf{x}_t$ is an intermediate noisy sample that satisfies two conditions.

One is $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$, which describes how the sample changes from the perspective of diffusion.

The other is its corresponding probability $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$, which indicates its probability density.

These two aspects are complementary.

{{< admonition type=quote title="Connection with Stochastic Gradient Langevin Dynamics" >}}
$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
{{< /admonition >}}

Langevin dynamics is a statistical method in physics for simulating molecular motion. It describes the random disturbances (e.g., thermal noise) experienced by particles moving in a potential field and is often used to model the stochastic behavior of complex systems.

Stochastic Gradient Langevin Dynamics (SGLD) is a sampling method that combines Langevin dynamics with stochastic gradient descent (SGD) in machine learning. Its goal is to sample from a probability distribution $p(x)$ without knowing the specific form of this distribution, only requiring knowledge of its gradient.

The above sampling formula is an iterative formula. Its meaning is: "Move a little in the gradient direction and add some random disturbances to make the final sample distribution approximate the target distribution $p(x)$." The meanings of the relevant symbols are as follows:

- $\mathbf{x}_t$: The sample at the $t$-th step.
- $\frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1})$: The drift term. It moves according to the gradient of the target distribution, similar to being pulled by a force. It can also be analogized to $\sqrt{1 - \beta_t} \mathbf{x}_{t-1}$ in diffusion.
  - $\delta / 2$: The step size, controlling the magnitude of each update.
  - $p(x)$: The probability density function of the target distribution.
  - $\log p(x)$: The log-probability density, facilitating calculation and optimization.
  - $\nabla_\mathbf{x} \log p(\mathbf{x}_{t-1})$: The gradient of the log-probability density, also called the score function, representing the "uphill direction" at the current point.
- $\sqrt{\delta} \boldsymbol{\epsilon}_t$: The diffusion term, similar to the molecular collisions in Brownian motion. It can be analogized to $\sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}$ in diffusion.
  - $\sqrt{\delta}$: The step size, controlling the magnitude of each update.
  - $\epsilon_t \sim \mathcal{N}(0, I)$: Random noise following a standard normal distribution, adding randomness to avoid getting stuck in local optima.

---

Note: The $p(\cdot)$ mentioned here is a general target distribution and can be any distribution we want to sample from. It is different from the $q(\cdot)$ and $p_\theta(\cdot)$ we see in diffusion.

For the diffusion scenario, if we want to generate more realistic samples, we have:

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Here, $q(\cdot)$ measures the authenticity of the samples. In each iteration, $\mathbf{x}_t$ is more realistic than $\mathbf{x}_{t-1}$. Meanwhile, $\boldsymbol{\epsilon}_t$ prevents the generated samples from getting stuck in local optima.

### Reverse diffusion process

{{< admonition type=quote title="Reverse Diffusion Process is Also Gaussian" >}}
Note that if $\beta_t$ is small enough, $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ will also be Gaussian.
{{< /admonition >}}

Let's review the forward single-step diffusion formula:

$$
\begin{align}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \\
\mathbf{x}_t &= \sqrt{1-\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}
\end{align}
$$

When $\beta_t$ is small, it means that little noise is added at each step. Then:

- The relationship between $\mathbf{x}_t$ and $\mathbf{x}_{t-1}$ is very close to a linear transformation plus a small perturbation.
- A linear transformation of a Gaussian distribution still maintains a Gaussian form.
- This makes the reverse conditional distribution approximately Gaussian.

Therefore, we usually model the reverse process using a Gaussian distribution.

{{< admonition type=quote title="Modeling the Likelihood" >}}
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
{{< /admonition >}}

The above two formulas correspond to the likelihood formulas for the overall and single-step reverse diffusion processes, which are the form of the neural network we are going to build.

Since we model the reverse diffusion process as a Gaussian distribution, its learnable parameters are the mean and variance of the Gaussian distribution, $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ and $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$.

Let's take a look at the four important distributions involved in the training and inference of the diffusion model:

- The forward diffusion generates noise.
- The posterior generates the training target for the neural network, i.e., what the reverse diffusion neural network needs to learn at each step.
- The reverse diffusion likelihood is used to fit the posterior.
- According to the prior, we generate pure noise following a standard normal distribution and pass it to the fitted network for denoising to generate images.

| Distribution                                          | Role                                                                                                                      |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| $q(\mathbf{x}_t \mid \mathbf{x}_0)$                   | **Closed-form forward diffusion**. Directly add noise to the data to obtain training samples.                             |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ | **Reverse diffusion posterior**. Used to define the training target.                                                      |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$        | **Reverse diffusion likelihood**. Train a model to fit the above posterior.                                               |
| $q(\mathbf{x}_T)$                                     | **Prior**. Fixed as a Gaussian distribution $\mathcal{N}(0, I)$, directly sampled as the starting point during inference. |

{{< admonition type=quote title="Modeling the Posterior" >}}
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$
{{< /admonition >}}

Since we model the reverse diffusion process as a Gaussian distribution, we can first define the formula for the posterior as above. Then the problem is transformed into how to derive ${\tilde{\boldsymbol{\mu}}_t}(\mathbf{x}_t, \mathbf{x}_0)$ and $\tilde{\beta}_t$. We will derive them in two steps later, and their specific forms are:

$$
\begin{align}
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
\end{align}
$$

{{< admonition type=quote title="Posterior Derivation Step 1: Expand According to Bayes' Formula and Gaussian Formula" >}}
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

First, we can use Bayes' formula to transform the calculation of the posterior into the calculation of the prior, i.e., the calculation of the forward diffusion. This allows us to build on the previous derivations.

Second, we can expand the probabilities into calculations between Gaussian probability density functions to obtain a new Gaussian probability density form.

Let's elaborate on the reasoning steps based on these two ideas:

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

(*) According to the Gaussian probability density function, we can expand it and perform linear simplification. Linear simplification is just a notational optimization because it does not affect our ability to obtain a new Gaussian distribution later. As long as the new Gaussian distribution has a similar form to the current one, it is acceptable.

$$
\begin{align}
p(x)
& = \mathcal{N}(x; \mu, \sigma^2) \\
& = \frac{1}{\sqrt{2\pi\sigma^2}} \; \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \\
& \propto \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\end{align}
$$

(**) $C(\mathbf{x}_t,\mathbf{x}_0)$ does not contain $\mathbf{x}_{t-1}$. Since we are calculating $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$ and $\mathbf{x}_{t}$ and $\mathbf{x}_{0}$ are known, $C(\mathbf{x}_t,\mathbf{x}_0)$ is just a constant term. Moreover, since it is in the exponent, it can be factored out as a coefficient. Finally, when we use it to calculate the final loss, as it is inside the KL divergence and thus inside the logarithm, it can be factored out as a separate constant term. The gradient of a constant term is 0, so it can be ignored here.

{{< admonition type=quote title="Posterior Derivation Step 2: Obtain a New Gaussian Distribution" >}}
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

Since

$$
\mathcal{N}(x; \mu, \sigma^2)
\propto \exp\!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
= \exp\!\left( -\frac{1}{2} (\color{red}{\frac{1}{\sigma^2}x^2} \color{blue}{- \frac{2\mu}{\sigma^2}x} \color{black}{+ \frac{\mu^2}{\sigma^2})} \right)
$$

And according to the previous calculation:

$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \propto \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 \color{blue}{- (\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
$$

We can obtain:

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

At this point:

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
& = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}_t}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
\end{align}
$$

This completes the proof.

(*) We can ignore the coefficient term here because, as mentioned earlier, the coefficient term will eventually become a constant term in the loss. Since the derivative of a constant is 0, it can be ignored.

{{< admonition type=quote title="Posterior Simplification" >}}
Thanks to the nice property, we can represent $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$ and plug it into the above equation and obtain:

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$
{{< /admonition >}}

The significance of this step is to make the calculation rely solely on noise rather than real data, enabling the direct recovery of real data from any noise.

The nice property mentioned here is the closed-form forward diffusion: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$.

Let's provide a more detailed derivation:

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

This completes the proof.

{{< admonition type=quote title="VLB: Derivation from Scratch" >}}
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

We have now derived the formula for the posterior, which serves as the golden truth that the neural network needs to mimic. So how do we model the relationship between the golden truth and the model?

This is where variational inference (or ELBO, Evidence Lower Bound) comes in. The VLB can be understood as the objective function or the loss function. In short, we hope the VLB to be smaller so that the trained model can more easily generate realistic images.

Here is a brief introduction to the variational lower bound:

1. The log marginal likelihood, $\log p_\theta(\mathbf{x}_0)$, is what we expect from the model, i.e., we hope the model has a higher probability of generating real images.
2. However, it is difficult to calculate directly because we cannot directly integrate over all cases in the latent variable distribution.
3. Therefore, we need to find an alternative optimization lower bound. Optimizing this lower bound is equivalent to optimizing the log marginal likelihood.
4. If you want to fully understand the relevant concepts, I strongly recommend reading another article by Lilian Weng, "From Autoencoder to Beta-VAE" [^lilian_ae], specifically the [section VAE: Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder).

---

Let's re-derive the variational lower bound (VLB) with more details.

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

At this point, we have transformed the difficult-to-calculate log marginal likelihood into an expression involving $q(\mathbf{x}_{1:T} | \mathbf{x}_0)$ and $p_\theta(\mathbf{x}_{0:T})$. The former can be directly calculated using the posterior formula we derived earlier, and the latter is defined by the neural network and can be decomposed into conditional probabilities at each step.

For training, we cannot run the model on only one image $\mathbf{x}_0$. We also need to sample using the Monte Carlo method. Therefore, we have:

$$
\begin{align}- \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
&\leq - \mathbb{E}_{q(\mathbf{x}_0)} \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= - \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0), \, \mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= - \mathbb{E}_{\mathbf{x}_{0, 1, ..., T}\sim q(\mathbf{x}_0) q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T} \sim q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&\triangleq L_\text{VLB}
\end{align}
$$

This completes the proof.

{{< admonition type=quote title="VLB: Derivation Using Jensen's Inequality" >}}
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

The derivation mainly uses [marginalization in probability theory]({{< relref "#marginalization" >}}) and Jensen's inequality. It is easy to prove after understanding these two concepts.

Marginalizing over $\mathbf{x}_{1:T}$ gives:

$$
\begin{align}
p_\theta(\mathbf{x}_0)
&= \int \Big[ p_\theta(\mathbf{x}_0 | \mathbf{x}_{1:T}) p_\theta(\mathbf{x}_{1:T}) \Big] d\mathbf{x}_{1:T} \\
&= \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}
\end{align}
$$

Jensen's inequality [^wiki_jensen] states that if $\phi(\cdot)$ is a concave function [^wiki_concave] and $X$ is an integrable random variable, then the following inequality holds:

$$
\phi\left( \mathbb{E}[X] \right) \geq \mathbb{E}\left[ \phi(X) \right]
$$

For example, $\log(\cdot)$ is a concave function.

{{< admonition type="quote" title="VLB Expansion" >}}
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

Why expand the concise VLB into the final complex formula consisting of multiple KL divergences?

Recall the closed-form expression of the forward diffusion process we derived earlier, as well as the posterior and likelihood obtained when modeling the reverse diffusion process:

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

The $L_\text{VLB} = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big]$ we obtained just now does not use these existing formulas, so it cannot be calculated directly.
Moreover, we do not want the neural network to fit the result in one step. Instead, we hope it can simulate the step-by-step denoising process.
(Of course, later there emerged Consistency Models [^song_consistency] that can achieve this in one step, which will be discussed later.)

Therefore, the purpose of expanding VLB is to transform the training objective from a log-likelihood function that is difficult to optimize directly into a set of computable KL divergence terms and reconstruction terms, thereby guiding the neural network to learn how to gradually recover the original data from noise.

Let's write out the expansion process of VLB more completely:

$$
\begin{aligned}
& L_\text{VLB} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big]
\quad\text{; Expand the joint probability distribution into a recurrence formula using the Markov property} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big]
\quad\text{; } -\log p_\theta(\mathbf{x}_T) \text{ is a constant, so it can be taken out separately} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; } \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \text{ is specially modeled and will be mentioned later} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; Transform } q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \text{ into a combination of the posterior formula and the forward closed-form formula according to Bayes' theorem} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; Split according to the calculation rules of the log function} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]
\quad\text{; Recombine according to the calculation rules of the log function} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big]
\quad\text{; Recombine according to the calculation rules of the log function} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} +
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} -
  \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \\
%
&= \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_T)\sim q(\mathbf{x}_0, \mathbf{x}_T)} \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} +
\sum_{t=2}^T \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_{t-1}, \mathbf{x}_t)\sim q(\mathbf{x}_0, \mathbf{x}_{t-1}, \mathbf{x}_t)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} -
\mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1)\sim q(\mathbf{x}_0, \mathbf{x}_1)} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; Simplify according to the law of total expectation} \\
%
&= \mathbb{E}_{\mathbf{x}_0 \sim q(\mathbf{x}_0)} \left[ \mathbb{E}_{\mathbf{x}_T \sim q(\mathbf{x}_T | \mathbf{x}_0)} \log \frac{q(\mathbf{x}_T | \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} \right] +
\sum_{t=2}^T \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_t) \sim q(\mathbf{x}_0, \mathbf{x}_t)} \left[ \mathbb{E}_{\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)} \right] -
\mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1)\sim q(\mathbf{x}_0, \mathbf{x}_1)} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; Expand into the form of conditional expectation} \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ \mathbb{E}_{\mathbf{x}_T \sim q(\mathbf{x}_T | \mathbf{x}_0)} \log \frac{q(\mathbf{x}_T | \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} \right] +
\sum_{t=2}^T \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \left[ \mathbb{E}_{\mathbf{x}_{t-1} \sim q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)} \right] -
\mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\quad\text{; Complete according to the law of total expectation} \\
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
\quad\text{; Rewrite into the form of KL divergence}
\end{aligned}
$$

{{% admonition type="quote" title="VLB Parameterization" open=true %}}
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

For the sake of brevity, the expectation symbols are actually omitted here. Moreover, the range of the second term in the original formula is $2 \leq t \leq T$, not $1 \leq t \leq T-1$. Therefore, we write it more strictly as follows:

$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0, \, \text{where } \\
L_T &= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)) \text{ for }2 \leq t \leq T \\
L_0 &= - \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$

The following will explain how these three terms participate in the training of the neural network:

⭐ $L_T = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))$

As mentioned in the original blog post, this is just a constant with a backpropagation gradient of 0, so it can be ignored and does not participate in optimization.

⭐ $L_t = \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))$

This term is the most important and will be elaborated on later. It is temporarily skipped here.

⭐ $L_0 = - \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)$

For the convenience of calculation, this term is approximated as the denoising matching term at $t = 1$:

$$
L_0 \approx \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} D_{\text{KL}}(q(\mathbf{x}_0 \vert \mathbf{x}_1) \parallel p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1))
$$

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

The derivation still needs to use the closed-form expression of the forward diffusion process we derived earlier, as well as the posterior and likelihood obtained when modeling the reverse diffusion process:

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

---

In addition, we need to know the formula for calculating the KL divergence between two multivariate Gaussian distributions (read this post [^gupta_gaussian_kl] to learn how to derive it). Given two Gaussian distributions, $\mathcal{N}(\boldsymbol{\mu_q},\,\Sigma_q)$ and $\mathcal{N}(\boldsymbol{\mu_p},\,\Sigma_p)$, with a data dimension of $k$, their KL divergence is:

$$D_{KL}(q||p) = \frac{1}{2}\left[\log\frac{|\Sigma_p|}{|\Sigma_q|} - k + (\boldsymbol{\mu_q}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\boldsymbol{\mu_q}-\boldsymbol{\mu_p}) + tr\left\{\Sigma_p^{-1}\Sigma_q\right\}\right]$$

---

Furthermore, we need to know an important assumption. The original DDPM paper [^ho_ddpm] assumes that $\Sigma_\theta$ is a constant hyperparameter. If $\Sigma_\theta$ needs to be learned, it will lead to:

1. Gradient updates for the variance are required during training, which may cause divergence.
2. For each time step $t$, $\Sigma_\theta(\mathbf{x}_t, t)$ is high-dimensional (e.g., the image pixel dimension), resulting in a large training volume.
3. Experience shows that if only the mean is predicted, the model can already learn the reverse process well, and the quality of the generated samples is also high.

---

Based on these information, let's write out the complete derivation of $L_t$:

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
\quad\text{; Simplify the above formula}\\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \left[
  \log \frac{|\boldsymbol{\Sigma}_{\theta,t}|}{|\tilde{\beta}_t \mathbf{I}|} -
  k +
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})^T \boldsymbol{\Sigma}_{\theta,t}^{-1} (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t}) +
  \text{tr}(\boldsymbol{\Sigma}_{\theta,t}^{-1} \tilde{\beta}_t \mathbf{I})
\right]
\quad\text{; Expand the KL divergence of the Gaussian distributions}\\
%
&\approx \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \left[
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})^T
  \boldsymbol{\Sigma}_{\theta,t}^{-1}
  (\tilde{\boldsymbol{\mu}}_{t} - \boldsymbol{\mu}_{\theta,t})
\right]
\quad\text{; Ignore the constants } \Sigma_\theta(\mathbf{x}_t, t), \tilde{\beta}_t\mathbf{I}, k \\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})}
\frac{1}{2} \Big[
  \frac{1}{\| \boldsymbol{\Sigma}_{\theta,t} \|^2_2}
  \| \tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_{\theta,t} \|^2
\Big]
\quad\text{; }\boldsymbol{\Sigma}_{\theta,t}\text{ is a diagonal matrix, so it can be taken out separately}\\
%
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_{\theta,t} \|^2_2} \| \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big) - \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big) \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_{0:T}\sim q(\mathbf{x}_{0:T})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0\sim q(\mathbf{x}_{0})} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
&\propto \mathbb{E}_{t \sim [1, T], \mathbf{x}_0\sim q(\mathbf{x}_{0}), \boldsymbol{\epsilon}_t \sim\mathcal{N}(0, I) } \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_{\theta,t} \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] \\
\end{aligned}
$$

Herein:

- $t, \mathbf{x}_0, \boldsymbol{\epsilon}_t$ are all obtained through Monte Carlo sampling, so they are inputs to the model.
- $\alpha_t, \bar{\alpha}_t$ both depend on the Noise variance schedule parameter $\beta_t$, which is a hyperparameter and a scheduling sequence, a value worth designing.
- $\boldsymbol{\Sigma}_{\theta,t}$ is the covariance of the reverse diffusion distribution at time step $t$, which can be designed as a constant or learnable parameters.
- $\boldsymbol{\epsilon}_\theta$ is a term that the neural network must learn, which needs no further explanation.

Next, the design strategies related to these three parameters will be explained one by one:

1. The $L_t^\text{simple}$ strategy where $\beta_t$ and $\boldsymbol{\Sigma}_{\theta,t}$ are simple constants.
2. The strategy where $\beta_t$ is a non-trivial schedule.
3. The strategy where $\boldsymbol{\Sigma}_{\theta,t}$ is a learnable parameter.

{{% admonition type="quote" title="Simplification of $L_t$" open=true %}}
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
{{% /admonition %}}

This mainly explains two things:

1. During training, Monte Carlo sampling is performed on any real image sample $\mathbf{x}_0$, any diffusion step $t$, and any noise $\boldsymbol{\epsilon}_t$.
2. The weight coefficient containing $\boldsymbol{\Sigma}_{\theta,t}$ is ignored during training because it is set to a constant in the original paper [^ho_ddpm].

PS. This simplified formula also provides us with another perspective to observe $L_t$, i.e., it can also be regarded as an MSE loss instead of a KL divergence loss.

{{% admonition type="quote" title="Training and Sampling in the DDPM Algorithm" open=true %}}
{{< media
src="DDPM_Algo.png"
caption="The training and sampling algorithms in DDPM (Image source: [Ho et al. 2020](https://arxiv.org/abs/2006.11239))"
>}}
{{% /admonition %}}

- Algorithm 1: Training is to teach the model to "guess the noise in the image at a certain moment". Thanks to the various closed-form formulas derived earlier, we do not need to perform step-by-step calculations.
- Algorithm 2: Sampling is to start from random noise and gradually denoise to generate new images. Therefore, it has to go through the entire diffusion process, so the generation speed is slow.

PS. In the field of image generation, sampling refers to using the trained model for inference.

{{% admonition type="quote" title="Connection with Noise-Conditioned Score Networks (NCSN)" open=true %}}

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

Let's recall the Stochastic Gradient Langevin Dynamics sampling formula mentioned earlier:

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Here, $p(\cdot)$ is used to measure the authenticity of the generated samples. As stated in the original text, $p(\cdot)$ is defined as

$$
\begin{align}
p(\mathbf{x}_t) &\triangleq q(\mathbf{x}_t \vert \mathbf{x}_0) \\
&= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{align}
$$

Substituting this in, we obtain a more complete sampling formula:

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Due to the existence of this closed-form formula, it means that we can continuously iterate this formula to obtain a sample with higher authenticity. So now the key question is to see if each term in it can be obtained.

In this formula, $\mathbf{x}_{t-1}$ is our input value; $\frac{\delta}{2}$ is a constant coefficient; $\sqrt{\delta} \boldsymbol{\epsilon}_t$ is randomly sampled; only the middle $\nabla_\mathbf{x} \log q(\mathbf{x}_{t-1})$ is the key term that the neural network needs to fit.

---

Before understanding how to model the middle term, let's first derive the derivative of the logarithm of the Gaussian density function (this is a separate derivation and can be skipped first):

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

So, the conclusion is:

$$
\boxed{
  \nabla_{\mathbf{x}}\log \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2 \mathbf{I})
  = - \frac{\boldsymbol{\epsilon}}{\sigma}
}
\quad\text{; where } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

---

With the above formula, let's derive our current ground truth:

$$
\begin{align}
\nabla_\mathbf{x} \log q(\mathbf{x}_{t-1})
&= \nabla_\mathbf{x} \log \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0, (1 - \bar{\alpha}_{t-1})\mathbf{I}) \\
&= - \frac{\boldsymbol{\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0}}{\sqrt{1 - \bar{\alpha}_{t-1}}}
\end{align}
$$

There are two variables here, the time step $t$ and the real sample $\mathbf{x}_0$. So we have a supervision signal composed of the expectations of many ground truths:

$$
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\mathbf{x}_t - \boldsymbol{\sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0}}{1 - \bar{\alpha}_{t-1}} \right) =
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_{t-1}}} \right)
$$

Based on this, we can parameterize the neural network as:

$$
\mathbb{E}_{t \sim [1, .., T], \mathbf{x}_0 \sim q(\mathbf{x}_0)}
\left( - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_{t-1}}} \right)
$$

- During training, we sample many sets of $(t, \mathbf{x}_0)$ to make the neural network fit the ground truth.
- During inference, we substitute the existing $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) \approx \mathbf{s}_\theta(\mathbf{x}_t, t) = - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$ into the Stochastic Gradient Langevin Dynamics sampling formula to get:

  $$
  \mathbf{x}_t =
  \mathbf{x}_{t-1} -
  \frac{\delta \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{2\sqrt{1 - \bar{\alpha}_t}} +
  \sqrt{\delta} \boldsymbol{\epsilon}_t
  ,\quad\text{where }
  \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
  $$

  At this point, we can iteratively sample to generate new images without relying on the real sample $\mathbf{x}_0$.

### Parameterization of $\beta_t$

{{% admonition type="quote" title="From trivial to non-trivial $\beta_t$ scheduling" open=true %}}
The forward variances are set to be a sequence of linearly increasing constants in [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), from $\beta_1=10^{-4}$ to $\beta_T=0.02$. They are relatively small compared to the normalized image pixel values between $[-1, 1]$. Diffusion models in their experiments showed high-quality samples but still could not achieve competitive model log-likelihood as other generative models.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed several improvement techniques to help diffusion models to obtain lower NLL. One of the improvements is to use a cosine-based variance schedule. The choice of the scheduling function can be arbitrary, as long as it provides a near-linear drop in the middle of the training process and subtle changes around $t=0$ and $t=T$.

$$ \beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2 $$

where the small offset $s$ is to prevent $\beta_t$ from being too small when close to $t=0$.

{{< media
src="Linear_and_Cosine_Scheduling.png"
caption="Comparison of linear and cosine-based scheduling of during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))"
>}}
{{% /admonition %}}

Let's write out the scheduling methods from the original DDPM paper [^ho_ddpm] and the scheduling formulas from Improved DDPM [^nichol_improved_ddpm] in full:

The **linear variance schedule** in DDPM [^ho_ddpm] is:

$$
\begin{align}
\beta_t &= \beta_{\text{min}} + \frac{t - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}}) \\
\alpha_t &= 1 - \beta_t = 1 - \left(\beta_{\text{min}} + \frac{t - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})\right) \\
\bar{\alpha}_t &= \prod_{k=1}^{t} \alpha_k = \prod_{k=1}^{t} \left(1 - \beta_{\text{min}} - \frac{k - 1}{T - 1} (\beta_{\text{max}} - \beta_{\text{min}})\right)
\end{align}
$$

where:

- $\beta_{\text{min}}$ and $\beta_{\text{max}}$ are the preset minimum and maximum noise values (e.g., 0.0001 and 0.02).
- $T$ is the total number of diffusion steps (e.g., 1000).

The **cosine-based variance schedule** in Improved DDPM [^nichol_improved_ddpm] is:

$$
\begin{align}
\bar{\alpha}_t &= \frac{f(t)}{f(0)} \quad \text{; where } f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2, s = 0.008 \\
\alpha_t &= \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = \frac{f(t)}{f(t-1)} \\
\beta_t &= \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999)
\end{align}
$$

where:

- This schedule first defines $\bar{\alpha}_t$, then $\alpha_t$ and $\beta_t$.
- The small offset $s$ is used to prevent $\beta_t$ from becoming too small when close to $t = 0$. Improved DDPM [^nichol_improved_ddpm] argues that having tiny amounts of noise at the beginning of the process makes it difficult for the network to predict accurately enough.
- The clip function ensures that $\beta_t$ does not exceed 0.999, avoiding numerical instability. Improved DDPM [^nichol_improved_ddpm] suggests that this can prevent singularities at the end of the diffusion process near $t = T$.

Click the bottom-right corner of the following figure to enter an interactive interface and intuitively experience the two schedulers.

<iframe src="https://www.desmos.com/calculator/sxftdp4sib?embed" width="100%" height="500" style="border: 1px solid #ccc" frameborder=0></iframe>

### Parameterization of reverse process variance $\boldsymbol{\Sigma}_\theta$

{{% admonition type="quote" title="From unlearnable to learnable $\boldsymbol{\Sigma}_\theta$" open=true %}}
[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) chose to fix $\beta_t$ as constants instead of making them learnable and set $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$ , where $\sigma_t$ is not learned but set to $\beta_t$ or $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$. Because they found that learning a diagonal variance $\boldsymbol{\Sigma}_\theta$ leads to unstable training and poorer sample quality.

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672) proposed to learn $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ as an interpolation between $\beta_t$ and $\tilde{\beta}_t$ by model predicting a mixing vector $\mathbf{v}$ :

$$ \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t) $$
{{% /admonition %}}

Recall the posterior and likelihood in the reverse diffusion process we discussed earlier:

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

where $\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$ and $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$.

To make the training more convergent, in the original DDPM [^ho_ddpm], the authors modeled the variance part as a constant, $\boldsymbol{\Sigma}_\theta \triangleq \tilde{\beta}_t \text{ or } \beta_t$, based on the formal consistency of the two equations. They only treated the noise $\boldsymbol{\epsilon}_t$ in the closed form of $\tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0)$ as a learnable parameter.

---

Why can it also be modeled as the noise schedule $\beta_t$ in the forward diffusion? Nichol from OpenAI provided an explanation in Improved DDPM [^nichol_improved_ddpm]. They are actually very close, especially in the later stages of the diffusion process.

{{< media
src="Ratio_vs_Diffusion_Step.png"
caption="The ratio for every diffusion step for diffusion processes of different lengths. ([source](https://proceedings.mlr.press/v139/nichol21a.html))"
>}}

---

They also found that improving this part to make it a learnable interpolation between $\beta_t$ and $\tilde{\beta}_t$ can lead to better results.

$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)\mathbf{I}
$$

where:

- $\mathbf{I}$ (identity matrix), added here to ensure that the final $\boldsymbol{\Sigma}_\theta$ is a diagonal covariance matrix.
- $\beta_t$ (scalar) is the forward noise schedule.
- $\tilde{\beta}_t$ (scalar) is the posterior variance.
- $\mathbf{v}$ (vector) is the mixing coefficient output by the model, which is what the model needs to learn.
- $\boldsymbol{\Sigma}_\theta$ (vector) is the predicted variance for each dimension (diagonal covariance).

{{% admonition type="quote" title="Integrate learnable $\boldsymbol{\Sigma}_\theta$ into the final loss" open=true %}}
However, the simple objective $L_\text{simple}$ does not depend on $\boldsymbol{\Sigma}_\theta$ . To add the dependency, they constructed a hybrid objective $L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}$ where $\lambda=0.001$ is small and stop gradient on $\boldsymbol{\mu}_\theta$ in the $L_\text{VLB}$ term such that $L_\text{VLB}$ only guides the learning of $\boldsymbol{\Sigma}_\theta$. Empirically they observed that $L_\text{VLB}$ is pretty challenging to optimize likely due to noisy gradients, so they proposed to use a time-averaging smoothed version of $L_\text{VLB}$ with importance sampling.
{{% /admonition %}}

The overall meaning of this passage is relatively straightforward. The main goal is to combine the original $L_\text{simple}$ loss with the loss containing the learnable parameter $\boldsymbol{\Sigma}_\theta$ for joint optimization. However, two concepts mentioned here can be elaborated: **"noisy gradient"** and **"time-averaged smoothed version of $L_\text{VLB}$ with importance sampling"**.

---

**Noisy gradient** is a concept proposed in OpenAI's paper *An Empirical Model of Large-Batch Training* [^mc_candlish_grad_noise].

In Stochastic Gradient Descent (SGD), instead of computing gradients using the entire dataset, we use a mini-batch. This introduces noise because gradients from different batches can vary significantly. The **Gradient Noise Scale** can measure this gradient volatility.

Its core idea is that if gradients vary greatly across different batches (high noise), we need a larger batch size to obtain more stable updates. Under simplified assumptions (e.g., the Hessian is a multiple of the identity matrix), the Gradient Noise Scale can be expressed as:

$$
B_{\text{simple}} = \frac{\text{tr}(\Sigma)}{\|G\|_2^2}
$$

where:

- $\text{tr}(\Sigma)$: The trace of the gradient covariance matrix, which is the sum of the variances of all parameter gradients, representing the "volatility" of the gradients.
- $\|G\|_2^2$: The squared norm of the gradient (global gradient norm), which is the sum of the squares of all parameter gradients, representing the "average intensity" of the gradients.

$B_{\text{simple}}$ represents: **the ratio of the gradient noise intensity to its average intensity**

- If $B_{\text{simple}}$ is large, it indicates strong gradient noise, and a larger batch size is recommended.
- If it is small, it means the gradients are stable, and a smaller batch size can be used to speed up training.

---

In this passage, the "time-averaged smoothed version of $L_\text{VLB}$ with [importance sampling](#importance-sampling-trick)" specifically refers to the following formula, a new loss function design proposed in Improved DDPM [^nichol_improved_ddpm]:

$$L_{\text{vlb}} = \mathbb{E}_{t \sim p_t} \left[ \frac{L_t}{p_t} \right], \text{ where } p_t \propto \sqrt{\mathbb{E}[L_t^2]} \text{ and } \sum p_t = 1$$

Let's recall the original DDPM loss function:

$$
L_t^\text{simple}
= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
$$

We can see that the time step $t$ follows a uniform distribution, so its probability density function is $p'(t)=\frac{1}{T}$.

However, Improved DDPM [^nichol_improved_ddpm] points out that the first few time steps contribute most of the loss values, so uniform sampling is inefficient.

{{< media
src="VLB_vs_Diffusion_Step.png"
caption="Terms of the VLB vs diffusion step. The first few terms contribute most to NLL. ([source](https://proceedings.mlr.press/v139/nichol21a.html))"
>}}

Therefore, we can use another distribution to optimize sampling. The authors proposed a new distribution $p_t \propto \sqrt{\mathbb{E}[L_t^2]}$ where $\sum p_t = 1$. This distribution is proportional to the loss values, meaning we expect regions with high loss values (regions with small time steps) to be sampled more frequently. Using the importance sampling formula to adjust the expectation:

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

{{% admonition type="quote" title="Classifier Guided Diffusion Sampling Formula" open=true %}}
To explicit incorporate class information into the diffusion process, [Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233) trained a classifier $f_\phi(y \vert \mathbf{x}_t, t)$ on noisy image $\mathbf{x}_t$ and use gradients $\nabla_\mathbf{x} \log f_\phi(y \vert \mathbf{x}_t)$ to guide the diffusion sampling process toward the conditioning information $y$ (e.g. a target class label) by altering the noise prediction. Recall that $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ and we can write the score function for the joint distribution $q(\mathbf{x}_t, y)$ as following,

$$ \begin{aligned} \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y) &= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\ &\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\ &= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)) \end{aligned} $$
Thus, a new classifier-guided predictor $\bar{\boldsymbol{\epsilon}}_\theta$ would take the form as following,

$$ \bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) $$
To control the strength of the classifier guidance, we can add a weight $w$ to the delta part,

$$ \bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) $$
{{% /admonition %}}

The overall idea is to combine a pre-trained diffusion model and a pre-trained image classifier model for conditional generation. By leveraging the Langevin dynamics technique, if we have a classifier, we can obtain the gradient information of class $l$. With this gradient information, we can perform sampling via Langevin dynamics and iteratively generate images that match class $l$:

$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_{t-1}, y) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**The derivation ideas in the blog post are as follows**:

- First, we convert the conditional probability into the gradients of the diffusion part and the classifier part.
- Second, we transform the theoretical formula into a form containing learnable parameters.
- Third, we perform a maybe unnecessary simplification.

### Classifier-Free Guidance

{{% admonition type="quote" title="Classifier-Free Guidance Sampling Formula" open=true %}}
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

The advantage of classifier-guided diffusion is that it can directly utilize two pre-trained models without additional operations. Another advantage is that we can control the strength of the "condition" through the weight $w$ mentioned above.

For classifier-free guidance, the simplest approach is to directly incorporate the condition information into the training of the diffusion model. However, this approach loses the feature of controlling the condition strength.

Therefore, another approach, as shown in the original blog post, is to train both conditional and unconditional cases on the same network architecture, only distinguishing them by the condition input. In this way, we can control the condition strength through "subtraction".

Let's write out the sampling formula more comprehensively:

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) + \nabla_{\mathbf{x}_t} \log p(y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) &\quad\text{(Expanded according to Bayes' formula)}\\
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big)  &\quad\text{(Expanded in the classifier-guided manner)}\\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$

{{% admonition type="quote" title="FID and IS" open=true %}}
Their experiments showed that classifier-free guidance can achieve a good balance between FID (distinguish between synthetic and generated images) and IS (quality and diversity).
{{% /admonition %}}

FID and IS are important evaluation metrics for generative models.

---

FID (Fréchet Inception Distance) [^heusel_fid] measures the distribution difference between generated images and real images in the feature space. It uses the Inception network to extract image features and then calculates the Fréchet distance between two high-dimensional Gaussian distributions.

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

where:

- $\mu_r, \Sigma_r$: Mean and covariance of real images.
- $\mu_g, \Sigma_g$: Mean and covariance of generated images.

A lower FID indicates that the generated images are closer to the real images.

---

IS (Inception Score) [^salimans_improve_gan] measures the "sharpness" and "diversity" of generated images. It uses the Inception network to predict the image category distribution and then calculates the KL divergence of the predicted distribution.

$$
\text{IS} = \exp\left( \mathbb{E}_{x \sim p_g} \left[ D_{\text{KL}}(p(y|x) \| p(y)) \right] \right)
$$

where:

- $p(y|x)$: Predicted distribution of the Inception network for generated images.
- $p(y)$: Average predicted distribution of all generated images.

A higher IS indicates that the images are sharp (low entropy of the predicted distribution) and diverse (high entropy of the average distribution).

{{% admonition type="quote" title="Classifier-Free Guidance Outperforms Classifier Guidance" open=true %}}
The guided diffusion model, GLIDE ([Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741)), explored both guiding strategies, CLIP guidance and classifier-free guidance, and found that the latter is more preferred. They hypothesized that it is because CLIP guidance exploits the model with adversarial examples towards the CLIP model, rather than optimize the better matched images generation.
{{% /admonition %}}

GLIDE is a guided diffusion model. It tried two guiding strategies for image generation:

1. **CLIP guidance**: Leverages the image-text matching ability of the CLIP model to guide the image generation process.
2. **Classifier-free guidance**: Does not rely on an external classifier. Instead, it trains a single model to learn both conditional and unconditional image generation, achieving guidance in this way.

GLIDE's experiments showed that **classifier-free guidance outperforms CLIP guidance**, as follows:

- Classifier-free guidance: The model learns to generate images independently without relying on external judgment.
- CLIP guidance: The model relies on CLIP scores but may "cheat" to deceive the CLIP model.
- GLIDE prefers the former because it is more natural and robust.

## Speed up Diffusion Models

### Fewer Sampling Steps & Distillation

{{% admonition type="quote" title="Naive Strided Sampling" open=true %}}
One simple way is to run a strided sampling schedule (Nichol & Dhariwal, 2021) by taking the sampling update every $\lceil T/S \rceil$ steps to reduce the process from $T$ to $S$ steps. The new sampling schedule for generation is $\{\tau_1, \dots, \tau_S\}$ where $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$ and $S < T$.
{{% /admonition %}}

Here is the standard DDPM sampling formula:

$$
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)
= \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \quad, t\in[1,T]
$$

The naive strided sampling modifies the sampling formula as follows:

$$
p_\theta(\mathbf{x}_{\tau_{k-1}} \vert \mathbf{x}_{\tau_k})
= \mathcal{N}(\mathbf{x}_{\tau_{k-1}}; \boldsymbol{\mu}_\theta(\mathbf{x}_{\tau_k}, \tau_k), \boldsymbol{\Sigma}_\theta(\mathbf{x}_{\tau_k}, \tau_k)) \quad, k\in[1,S]
$$

Of course, when taking multiple steps at once, the true mean and variance should be derived from the multi-step Gaussian combination formula. Therefore, this strided sampling strategy is only a rough acceleration method.

{{% admonition type="quote" title="Denoising Diffusion Implicit Model" open=true %}}
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

Naive Strided Sampling lacks mathematical rigor, which DDIM [^song_ddim] addresses. Meanwhile, DDIM transforms the original random sampling process into a combination of random and deterministic sampling, making it more flexible.

Moreover, the main difference between DDIM and DDPM lies in how they model the likelihood to predict the posterior.

$$
\begin{align}
\text{DDPM Posterior:} & q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}) \\
\text{DDPM Likelihood:} & p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)) \\
\text{DDIM Likelihood:} & q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{align}
$$

where:

- $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$
- $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$. When $\eta = 0$, the variance is 0, making the sampling process deterministic (DDIM); when $\eta = 1$, the sampling process is random (DDPM).

Note: DDPM and DDIM are from different papers, so there may be math symbol conflicts. Here, we follow the original papers and do not unify them.

The derivation of the formula starts from the closed-form formula of the forward diffusion process. We attempt to separate a $\sigma_t^2$ term to the variance position using the following technique:

$$
\text{Var}(\boldsymbol{\epsilon}_{t-1}) = a^2 \cdot \text{Var}(\boldsymbol{\epsilon}_t) + b^2 \cdot \text{Var}(\boldsymbol{\epsilon}) = a^2 + b^2 \quad \text{; where }\epsilon_t \text{ and }\epsilon \text{ are independent}
$$

{{% admonition type="quote" title="Progressive Distillation" open=true %}}
{{< media
src="Progressive_Distillation_Algo.png"
caption="Comparison of Algorithm 1 (diffusion model training) and Algorithm 2 (progressive distillation) side-by-side, where the relative changes in progressive distillation are highlighted in green. (Image source: [Salimans & Ho, 2022](https://arxiv.org/abs/2202.00512))"
>}}
{{% /admonition %}}

We can provide a more detailed explanation of the Progressive Distillation algorithm on the right.

First, there are two nested loops: The outer loop controls the training of $K$ different student diffusion models, each capable of generating images with fewer sampling steps.

The inner loop trains the current student model:

1. $\text{Cat}[1,2,...,N]$ refers to the categorical distribution. We uniformly sample an integer $t$ from 1 to $N$, representing the current time step.
2. $t' = t - \frac{0.5}{N}, \quad t'' = t - \frac{1}{N}$ means splitting the time step $t$ into two smaller steps, $t'$ and $t''$.
3. $z_{t'} = \alpha_{t'} \tilde{x}_\eta(z_t) + \frac{\sigma_{t'}}{\sigma_t} \Big( z_t - \alpha_t \tilde{x}_\eta(z_t) \Big)$ is one of the update formulas for teacher DDIM sampling, indicating that $z_{t'}$ is obtained by taking one sampling step from $z_t$. For the specific derivation, please refer to the original paper [^salimans_progressive_distillation].

{{% admonition type="quote" title="Consistency Model" open=true %}}
Given a trajectory $\{\mathbf{x}_t \vert t \in [\epsilon, T]\}$ , the consistency function $f$ is defined as $f: (\mathbf{x}_t, t) \mapsto \mathbf{x}_\epsilon$ and the equation $f(\mathbf{x}_t, t) = f(\mathbf{x}_{t’}, t’) = \mathbf{x}_\epsilon$ holds true for all $t, t’ \in [\epsilon, T]$. When $t=\epsilon$, $f$ is an identify function. The model can be parameterized as follows, where $c_\text{skip}(t)$ and $c_\text{out}(t)$ functions are designed in a way that $c_\text{skip}(\epsilon) = 1, c_\text{out}(\epsilon) = 0$:

$$ f_\theta(\mathbf{x}, t) = c_\text{skip}(t)\mathbf{x} + c_\text{out}(t) F_\theta(\mathbf{x}, t) $$
It is possible for the consistency model to generate samples in a single step, while still maintaining the flexibility of trading computation for better quality following a multi-step sampling process.
{{% /admonition %}}

The goal of the Consistency Model (CM) is to learn a direct mapping $f$ that can map any point $\mathbf{x}_t$ with a noise level $t > 0$ back to the "source" of the same generation trajectory (more precisely, a sample $\mathbf{x}_\epsilon$ at a small time point $\epsilon$ very close to 0).

---

Why map to $\mathbf{x}_\epsilon$ instead of $\mathbf{x}_0$ exactly?

- $t = 0$ is often numerically unstable (singular or has a poor condition number). Choosing a small $\epsilon > 0$ makes the training **more stable and easier**.
- $\mathbf{x}_\epsilon$ is already extremely close to $\mathbf{x}_0$. If necessary, we can add one or two refinement steps from $\epsilon$ to 0.

---

Why is it an identity mapping when $t = \epsilon$?

The paper parameterizes $f_\theta$ as a **residual form with skip connections**:

$$
f_\theta(\mathbf{x},t) \;=\; c_{\text{skip}}(t)\,\mathbf{x} \;+\; c_{\text{out}}(t)\,F_\theta(\mathbf{x},t),
$$

and specifically designs

$$
c_{\text{skip}}(\epsilon)=1,\qquad c_{\text{out}}(\epsilon)=0.
$$

Thus,

$$
f_\theta(\mathbf{x},\epsilon)=\mathbf{x},
$$

which is an identity mapping.

### Latent Variable Space

{{% admonition type="quote" title="Latent Diffusion Model (LDM)" open=true %}}
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

This formula describes a key module for the **cross-attention mechanism** in the **Latent Diffusion Model (LDM)**. It combines the **attention mechanism in Transformer** with the **conditional control mechanism in diffusion models**, serving as the core component for LDM to achieve **text-to-image generation** or other conditional generation tasks.

---

Scaled dot-product attention:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right) \cdot \mathbf{V}
$$

- $\mathbf{Q}$: Query, representing the information to be processed currently.
- $\mathbf{K}$: Key, representing the information that can be attended to.
- $\mathbf{V}$: Value, from which the actual content is extracted when $\mathbf{Q}$ and $\mathbf{K}$ match.
- The denominator $\sqrt{d}$ is a scaling factor to prevent the inner product from being too large, which could cause the softmax gradient to vanish.
- Softmax normalizes along the Key dimension to obtain attention weights.
- The final output is a weighted sum, indicating "which parts of the value are more important to extract based on the query".

---

Sources of Query, Key, and Value in LDM:

- **Query comes from the image (latent representation)**
- **Key and Value come from the text/conditional information**

This means that **every spatial position in the image "attends" to which text words are most relevant**. For example, when generating an image of "a red dog running on the grass", the "grass" area in the image will pay more attention to the word "grass" in the text.

This mechanism allows the model to precisely align semantic conditions with the spatial structure of the image.

| Symbol                                                       | Meaning                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{z}_i$                                               | The latent variable (latent feature) at the $i$-th layer, i.e., the feature map of the diffusion model at a certain layer in the U-Net, located in the low-dimensional latent space.                                                                                                                          |
| $\varphi_i(\mathbf{z}_i)$                                    | Projects or reshapes the latent feature $\mathbf{z}_i$ into a form suitable for attention calculation (e.g., flattened into a sequence). Dimension: $N \times d^i_\epsilon$, where $N$ is the number of spatial positions (e.g., H×W), and $d^i_\epsilon$ is the feature dimension of this layer.             |
| $y$                                                          | Conditional input (e.g., text description).                                                                                                                                                                                                                                                                   |
| $\tau_\theta(y)$                                             | The encoding result of the conditional input $y$ by the conditional encoder (e.g., CLIP or BERT). Outputs a set of token embeddings (e.g., one vector per word). Dimension: $M \times d_\tau$, where $M$ is the number of tokens (e.g., 77 text tokens), and $d_\tau$ is the embedding dimension (e.g., 768). |
| $\mathbf{W}^{(i)}_Q, \mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V$ | Learnable projection matrices (parameters) used to map the inputs to Q/K/V in the attention space.                                                                                                                                                                                                            |

## Scale up Generation Resolution and Quality

This section discusses how to improve the image generation quality and resolution of diffusion models to a higher level through a series of techniques.

It mainly covers the Noise Conditioning Augmentation technique, the unCLIP model architecture, the Imagen model architecture, and some other improvements.

Here, we only pay attention to the Noise Conditioning Augmentation and the unCLIP model architecture.

{{% admonition type="quote" title="Noise conditioning augmentation" open=true %}}
*Noise conditioning augmentation* between pipeline models is crucial to the final image quality, which is to apply strong data augmentation to the conditioning input $\mathbf{z}$ of each super-resolution model $p_\theta(\mathbf{x} \vert \mathbf{z})$. The conditioning noise helps reduce compounding error in the pipeline setup...

They found the most effective noise is to apply Gaussian noise at low resolution and Gaussian blur at high resolution. In addition, they also explored two forms of conditioning augmentation that require small modification to the training process. Note that conditioning noise is only applied to training but not at inference.

- Truncated conditioning augmentation stops the diffusion process early at step $t > 0$ for low resolution.
- Non-truncated conditioning augmentation runs the full low resolution reverse process until step 0 but then corrupt it by $\mathbf{z}_t \sim q(\mathbf{x}_t \vert \mathbf{x}_0)$ and then feeds the corrupted $\mathbf{z}_t$ s into the super-resolution model.
{{% /admonition %}}

When using a pipeline of multiple diffusion models, the input of each stage is the output of the previous stage, which can easily lead to error accumulation and damage to the final image.

Therefore, they introduced **Noise Conditioning Augmentation** to enable the model to generate clear images even under "slightly blurred or noisy" conditions.

Two types of noise:

- **Low-resolution stage**: Add Gaussian noise.
- **High-resolution stage**: Add Gaussian blur.
- These noises are only added during training and not during inference.

Two training methods:

1. **Truncated Conditioning Augmentation**: Terminate the diffusion process early in the low-resolution stage (e.g., stop at step $t$).
2. **Non-Truncated Conditioning Augmentation**: Complete the low-resolution diffusion process, then artificially add noise, and use the result as the input for the high-resolution model.

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

unCLIP is a two-stage text-to-image generation model that corelly leverages the text and image embeddings of CLIP:

1. **Prior model**: Input text → Output CLIP image embedding.
2. **Decoder model**: Input image embedding (and optional text) → Output image.

This design allows:

- Text-to-image generation.
- Generating variations of a given image while preserving its style and semantics.

Similar to unCLIP, the Imagen model is also a two-stage text-to-image generation model. The difference lies in the choice of the text encoder. unCLIP uses CLIP, while Imagen uses the large language model T5-XXL. Imagen also makes some optimizations to the U-Net architecture.

## Model Architecture

{{% admonition type="quote" title="Implementations of U-Net, ControlNet, and DiT" open=true %}}
**U-Net** ([Ronneberger, et al. 2015](https://arxiv.org/abs/1505.04597)) consists of a downsampling stack and an upsampling stack...

**ControlNet** ([Zhang et al. 2023](https://arxiv.org/abs/2302.05543)) introduces architectural changes via adding a “sandwiched” zero convolution layers of a trainable copy of the original model weights into each encoder layer of the U-Net...

**Diffusion Transformer** (**DiT**; [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) for diffusion modeling operates on latent patches, using the same design space of [LDM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#ldm) (Latent Diffusion Model)...
{{% /admonition %}}

Lilian Weng's blog has already introduced these three models. Below are simplified PyTorch implementation examples. Some knowledge of CNN networks is required here. Refer to Wang's CNN Explainer article [^zijie_cnn] to grasp the basics.

**U-Net** mainly demonstrates the core ideas of downsampling, upsampling, and skip connections. It uses a simple `double_conv` block containing two convolutional layers, ReLU, and batch normalization.

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

**ControlNet** is essentially a fine-tuning method for U-Net. Its core idea is to add a trainable copy to the frozen U-Net backbone and connect them through zero convolutions.

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

The core idea of **DiT** is to use a Transformer to process the latent representations of diffusion models. It "slices" the latent image representations into sequential patches. Here, we demonstrate the use of Adaptive Layer Normalization (adaLN) to inject time step and class information.

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

In the design of generative models, "tractability" and "flexibility" are often conflicting goals:

- **Tractability**: This refers to models with clear mathematical structures that facilitate inference and optimization. For example, Gaussian distributions, Laplace distributions, etc., allow for direct likelihood calculation, sampling, and training.
- **Flexibility**: This refers to models that can fit complex, high-dimensional, and non-linear data structures, such as images, audio, and text. However, these models are often difficult to train, sample from, or evaluate.

Traditional models include:

{{< media
src="Generative_Models.png"
caption="Overview of different types of generative models. ([Source](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/))"
>}}

- **VAE (Variational Autoencoder)**:
  - Principal: The input image is compressed into latent space variables by an encoder and then reconstructed by a decoder.
  - Pro/Con: Tractable but sacrifices generation quality.
- **GAN (Generative Adversarial Network)**:
  - Principal: Two networks are trained, one for generating images and the other for distinguishing real from fake images.
  - Pro/Con: Flexible but suffers from unstable training and lacks an explicit likelihood.
- **Flow-based models**:
  - Principal: Images are generated through flow matching, which combines function-level processes.
  - Pro/Con: Reversible but have limited architectures.

The uniqueness of diffusion models lies in the fact that they maintain the tractability of high-dimensional Gaussian modeling in theory by gradually adding noise (forward process) and learning to denoise (reverse process). Meanwhile, they achieve flexibility by using deep neural networks to learn complex data structures.

Moreover, nowadays, diffusion models have become the de facto standard for generative models, especially in the field of image generation. They excel in generation quality, sample diversity, and training stability, and can be easily scaled to larger and more complex tasks.

## Appendix

### Relevant Deep Learning Concepts

#### From AE to VAE and Then to VQ-VAE

- **AE (Autoencoder)**: A neural network with an encoder and a decoder. Its goal is to learn a compressed representation (latent variables) of the input data, mainly used for dimensionality reduction and feature extraction.
- **VAE (Variational Autoencoder)**: Based on the AE, it introduces probabilistic modeling and variational inference, enabling the model to have generative capabilities.
- **VQ-VAE (Vector Quantized Variational Autoencoder)**: It further introduces a discrete latent space, replacing the continuous latent distribution with vector quantization. It is more suitable for modeling discrete structures (e.g., symbolic features in speech and images) and provides discrete token representations for subsequent generative models (e.g., Transformer decoders).

| Feature                          | Autoencoder (AE)                          | Variational Autoencoder (VAE)                                     | Vector Quantized VAE (VQ-VAE)                                                      |
| -------------------------------- | ----------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Encoder Output                   | A deterministic vector $z = f(x)$         | A distribution $q(z \vert x) = \mathcal{N}(\mu(x), \sigma^2(x))$  | A discrete codebook vector $z \in \mathcal{E}$ obtained by nearest-neighbor search |
| Latent Space Type                | Continuous and deterministic              | Continuous and probabilistic                                      | Discrete and finite codebook                                                       |
| Training Objective               | Minimize reconstruction error (e.g., MSE) | Maximize the variational lower bound (ELBO)                       | Reconstruction error + codebook loss + commitment loss                             |
| Generative Model?                | No                                        | Yes                                                               | Yes                                                                                |
| Probabilistic Modeling?          | No                                        | Yes (models the latent variable distribution)                     | No (but provides a discrete symbolic latent space)                                 |
| Can Sample to Generate New Data? | No                                        | Can generate data by sampling from the prior $p(z)$               | Can generate data by sampling discrete tokens and decoding                         |
| Uses KL Divergence?              | No                                        | Yes (constrains the latent distribution to be close to the prior) | No (replaced by vector quantization and additional loss functions)                 |

#### Reparameterization Trick

**Definition**: Reparameterization is a method that **decouples random variables from non-differentiable sampling operations**, allowing the sampling operations to participate in gradient descent optimization.

**Principle**: It does not eliminate random sampling but minimizes the impact of random sampling on gradient propagation.

**Example**: If you have a random variable $z \sim \mathcal{N}(\mu, \sigma^2)$, directly sampling from this distribution prevents gradients from propagating through $\mu$ and $\sigma$. You can convert the random sampling of $z$ to the random sampling of $\epsilon$ using the following formula:

$$
\mathcal{N}(z; \mu, \sigma^2)
= \mu + \mathcal{N}(\epsilon'; 0, \sigma^2)
= \mu + \sigma \cdot \mathcal{N}(\epsilon; 0, 1)
$$

PyTorch Code Example:

```python
import torch

# Assume the mean and standard deviation output by the encoder
mu = torch.tensor([0.0, 1.0], requires_grad=True)
log_sigma = torch.tensor([0.0, 0.0], requires_grad=True)  # Usually output log(sigma) to avoid negative values
sigma = torch.exp(log_sigma)

# Reparameterization sampling
epsilon = torch.randn_like(mu)  # Sample from the standard normal distribution
z = mu + sigma * epsilon  # z is differentiable

# Assume a simple loss function
loss = (z**2).sum()
loss.backward()

print("grad mu:", mu.grad)
print("grad log_sigma:", log_sigma.grad)
```

#### Importance Sampling Trick

Suppose you want to compute the expectation of a function $f(x)$ with respect to a probability distribution $p(x)$:

$$
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx
$$

However, it might be difficult to sample directly from $p(x)$, or most samples of $f(x)$ under $p(x)$ contribute little, with only a few regions making significant contributions (e.g., tail events). In such cases, direct Monte Carlo estimation can be inefficient.

If we can **sample from another distribution $q(x)$ that is easier to sample from and then correct the bias through weighting**, the estimation becomes more convenient. This is known as importance sampling.

Importance sampling rewrites the original expectation as:

$$
\mathbb{E}_{x \sim p}[f(x)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q}\left[ f(x) \frac{p(x)}{q(x)} \right]
$$

Where:

- $q(x)$ is the **proposal distribution**, and $q(x) > 0$ when $p(x) > 0$ (i.e., the support of $q$ contains the support of $p$).
- $w(x_i) = \frac{p(x_i)}{q(x_i)}$ is called the **importance weight**.

At this point, we can sample $x_i \sim q(x)$ from $q(x)$ and then estimate:

$$
\hat{\mu} = \frac{1}{N} \sum_{i=1}^N f(x_i) \cdot \frac{p(x_i)}{q(x_i)}
$$

Let's take an example. Assume:

- $p(x) = \mathcal{N}(0, 1)$
- $f(x) = x^2 \cdot \mathbb{1}[x > 2]$
- If we sample directly from $p$, most samples satisfy $x < 2$, contributing zero to the result, which is inefficient.
- If we switch to sampling from $q(x) = \mathcal{N}(2.5, 1)$, we are more likely to sample points in the region $x > 2$.

PyTorch Code Example:

```python
import numpy as np

N = 10000
# Sample from q(x) ~ N(2.5, 1)
x = np.random.normal(2.5, 1, N)

# Compute unnormalized densities (ignoring constants)
log_p = -0.5 * x**2           # log N(0,1)
log_q = -0.5 * (x - 2.5)**2   # log N(2.5,1)

# Importance weights (unnormalized)
w = np.exp(log_p - log_q)

# Function values
f = x**2 * (x > 2)

# Normalized importance sampling estimate
mu_hat = np.sum(f * w) / np.sum(w)
print("Importance Sampling estimate:", mu_hat)
```

### Probability Theory

#### Gaussian Distribution

**Univariate Gaussian Distribution** $x \sim \mathcal{N}(\mu, \sigma^2)$ has the following probability density function (PDF):

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- **Mean $\mu$**: The central location of the distribution.
- **Variance $\sigma^2$** (Standard Deviation $\sigma$): The width (uncertainty) of the distribution.

**Multivariate Gaussian Distribution** $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ has the following probability density function (pdf):

$$
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}}
\exp\!\left(-\tfrac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

Where:

- $\mathbf{x} \in \mathbb{R}^d$ is a $d$-dimensional vector.
- Mean vector $\boldsymbol{\mu} \in \mathbb{R}^d$: The central location of the distribution.
- Covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{d\times d}$: Describes the variance and correlation between different dimensions.
- $|\boldsymbol{\Sigma}|$ = The determinant of the covariance matrix, representing "volume scaling".
- $\boldsymbol{\Sigma}^{-1}$ = The inverse of the covariance matrix, defining the "ellipsoidal" isodensity curves.

Decomposing $\boldsymbol{\Sigma}$ can reveal the geometric properties of the distribution:

- Diagonal elements: The variance of each dimension (the numerical value equals the "width" along that axis).
- Off-diagonal elements: The correlation between different dimensions, determining whether the distribution is a "rotated ellipse/ellipsoid".

Examples:

- If $\boldsymbol{\Sigma} = \sigma^2 I$, it is an isotropic "circular/spherical" distribution.
- If $\boldsymbol{\Sigma}$ is not a diagonal matrix, there is correlation, and the isodensity lines are "tilted ellipses/ellipsoids".

The following two illustrations more intuitively show the effects of $\mu$ and $\Sigma$ on the shape of the PDF [^saleem_gaussian]:

{{< media
src="Gaussian_Mean.gif"
caption="Changes to the mean vector act to translate the Gaussian’s main ‘bump’. ([source](https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b))"
>}}

{{< media
src="Gaussian_Covariance.gif"
caption="Changes to the covariance matrix act to change the shape of the Gaussian’s main ‘bump’. ([source](https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b))"
>}}

#### Joint, Marginal, and Conditional Distributions

- Joint distribution $P(A, B)$: A panoramic map (containing the probabilities of all combinations).
- Marginal distribution $P(B)$: The projection of the panoramic map onto a single axis.
- Conditional distribution $P(A \vert B)$: Cutting a line from the panoramic map (given the value of another variable) and observing the probability distribution along this line. The formula is: $P(A \vert B) = \frac{P(A, B)}{P(B)}$.

#### Prior, Likelihood, and Posterior

| Concept    | Analogous Explanation                                                                    | Analogy in Diffusion Models                          |
| ---------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Prior      | The assumed distribution of a variable before observing any data                         | $q(\mathbf{x}_t) \sim \mathcal{N}(0, I)$             |
| Likelihood | The probability of observing the data under a certain hypothesis/model parameters        | $p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)$        |
| Posterior  | The updated distribution of latent variables after combining the prior and observed data | $q(\mathbf{x}_{t-1}\mid \mathbf{x}_t, \mathbf{x}_0)$ |

The relationship among them is:

$$
\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}
\quad\leftrightarrow\quad
P(\text{Parameters} | \text{Data}) = \frac{P(\text{Data} | \text{Parameters}) \cdot P(\text{Parameters})}{P(\text{Data})}
$$

#### Bayes’ Rule

Bayes’ Rule is a core principle in probability theory, used to update the probability of an event given certain conditions. Its basic form is:

$$
P(A \vert B) = \frac{P(B \vert A) \cdot P(A)}{P(B)}
$$

Where:

- $P(A)$: The prior probability of event A (the belief in A before observing B).
- $P(B \vert A)$: The likelihood of observing B given that A has occurred.
- $P(B)$: The marginal probability of event B (the probability of B occurring under all possible scenarios).
- $P(A \vert B)$: The posterior probability of event A after observing B (the updated belief).

#### Law of Total Probability

The Law of Total Probability is used to **calculate the probability of an event** when the probability of this event is not easy to calculate directly. It can be solved by **decomposing it into the sum of conditional probabilities under several mutually exclusive scenarios**.

Suppose events $B_1, B_2, \dots, B_n$ form a **complete set of events** in the sample space (i.e., they are mutually exclusive and their union is the entire space), that is:

- $B_i \cap B_j = \emptyset$ (Mutually exclusive).
- $\bigcup_{i=1}^n B_i = \Omega$ (Exhaustive).

Then for any event $A$, we have:

$$
P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)
$$

#### Marginalization

Marginalization refers to **obtaining the probability distribution of another variable from a joint probability distribution by "summing (or integrating) over certain variables"**.

For example, given the joint distribution $P(X, Y)$ of two random variables $X$ and $Y$, if we want to obtain the distribution $P(X)$ of $X$, we need to "marginalize" over all possible values of $Y$:

$$
\begin{align}
&\text{Discrete case: } P(X = x) = \sum_{y} P(X = x, Y = y) \\
&\text{Continuous case: } P(X = x) = \int_{-\infty}^{\infty} P(X = x, Y = y) \, dy
\end{align}
$$

Moreover, we can also view the Law of Total Probability as an application of marginalization:

$$P(X) = \sum_{y} P(X, Y = y) = \sum_{y} P(X | Y = y) P(Y = y)$$

#### Expectation

Definition of expectation:

- Discrete random variable: $\mathbb{E}_{x \sim p}[g(x)] = \sum_{x} g(x) \cdot p(x)$
- Continuous random variable: $\mathbb{E}_{x \sim p}[g(x)] = \int_{-\infty}^{\infty} g(x) \cdot p(x)  dx$

Three important properties:

1. Linearity property:
  $\mathbb{E}_{x \sim p, y \sim q}[a \cdot f(x, y) + b \cdot g(x, y)] = a \cdot \mathbb{E}_{x \sim p, y \sim q}[f(x, y)] + b \cdot \mathbb{E}_{x \sim p, y \sim q}[g(x, y)]$
2. Law of Iterated Expectation / Tower Property:
  $\mathbb{E}_{(x, y) \sim p(x, y)}[f(x, y)] = \mathbb{E}_{y \sim p(y)}\left[ \mathbb{E}_{x \sim p(x|y)}[f(x, y)] \right]$
3. Law of Total Expectation:
  $\mathbb{E}_{x \sim p(x)}[f(x)] = \mathbb{E}_{y \sim p(y)}\left[ \mathbb{E}_{x \sim p(x|y)}[f(x)] \right]$

#### Stochastic Process

A Stochastic Process is a family of random variables that evolve over time (or space). A stochastic process can be written as:

$$
\{ X_t \}_{t \in T}
$$

- $t$: The index set, which can be **discrete** (e.g., integer time points $t = 0, 1, 2, \dots$) or **continuous** (e.g., real time $t \geq 0$).
- $X_t$: A random variable at each time point $t$.
- The entire process is a family of random variables, reflecting the randomness of the system as it evolves with $t$.

Intuitive understanding:

- A random variable is a "random quantity at a certain moment".
- A stochastic process is a "sequence of random quantities that change over time".

#### Markov Property

A stochastic process satisfies the Markov property if:

$$
P(X_{t+1} \mid X_t, X_{t-1}, \dots, X_0) = P(X_{t+1} \mid X_t)
$$

That is, the future only depends on the present and is independent of the past.

#### Score Function

In probability theory and statistics, the original definition of the **score function** is:

$$
s_\theta(x) = \nabla_\theta \log p_\theta(x)
$$

- This is the gradient of the **log-likelihood function with respect to the parameter θ**.
- In classical statistics, it is used for **maximum likelihood estimation** or **Fisher information calculation**.
- Intuitively, it tells you "how to adjust the model parameter θ to make the observed data x more likely".

However, in **diffusion models / score-based generative models**, the score function is extended to:

$$
s(x) = \nabla_x \log p(x)
$$

- Here, it is the gradient of the **log-density of the data itself x**.
- It tells you **the ascending direction of the data distribution**, that is, "where the data is more likely to appear".

### Information Theory

Let's use a scenario to explain the four most important formulas in information theory: You need to send a series of messages (e.g., letters), and each letter has a different probability of occurrence. To save bandwidth, you need to send them using the shortest binary codes...

#### Information Content $I(x) = -\log p(x)$

- Intuition: The amount of information contained in an event is related to how surprising it is. The less likely an event is to occur, the more information it carries when it does happen.
  - "The sun rises in the east" (probability ~1): Carries almost no information, with extremely little information content.
  - "It will rain tomorrow" (probability 0.3): Carries some information.
  - "It rained diamonds in Beijing tomorrow" (probability almost 0): If it happens, the information content is explosive.
- Why is it $-\log p(x)$?
  - From the encoding perspective: To optimally encode an event, we should assign short codewords to events with high probabilities and long codewords to events with low probabilities. This minimizes the average code length.
  - Optimal code length: The optimal code length for an event $x$ is $-\log p(x)$. The smaller the probability $p(x)$, the larger the code length $-\log p(x)$, which perfectly corresponds to the "surprise factor".
  - Why use logarithm: Using logarithms can convert the multiplicative relationship of probabilities (joint probability of independent events) into an additive relationship of code lengths, which is very convenient.
- Conclusion: The information content $I(x)$ is the optimal code length for event $x$.

#### Shannon Entropy $H(p) = -\sum_i p(x_i) \log p(x_i)$

- Intuition: For a probability distribution $p$, when we encode the events in it, what is the shortest average code length required per event? This "average minimum cost" represents the uncertainty of the distribution.
  - The more uniform a distribution is (e.g., a fair die), the harder it is to predict the next outcome, the higher its uncertainty, and the longer the average code length.
  - The more concentrated a distribution is (e.g., a biased die), the easier it is to predict the next outcome, the lower its uncertainty, and the shorter the average code length.
- Why is it $-\sum_i p(x_i) \log p(x_i)$?
  - This is the expected value of the information content: $\mathbb{E}_{x \sim p}[I(x)] = \mathbb{E}_{x \sim p}[-\log p(x)]$.
  - Each event $x_i$ has its own optimal code length $-\log p(x_i)$. We use the probability $p(x_i)$ of this event as the weight and average all possible code lengths to obtain the average optimal code length for the entire distribution.
- Conclusion: The Shannon entropy $H(p)$ is the average code length required when using the optimal encoding scheme of the distribution $p$ itself. It measures the inherent uncertainty of the distribution $p$.

#### Cross-Entropy $H(p, q) = -\sum_i p(x_i) \log q(x_i)$

- Intuition: Now the situation has changed. The true distribution of the data is $p$, but you mistakenly assume the distribution is $q$ and adopt the optimal encoding scheme designed for $q$ (i.e., assign a codeword of length $-\log q(x_i)$ to event $x_i$). How long, on average, are the codewords needed to encode the real data using this incorrect scheme?
- Why is it $-\sum_i p(x_i) \log q(x_i)$?
  - In the real data, the probability of event $x_i$ occurring is $p(x_i)$.
  - The codeword length you assign to it is $-\log q(x_i)$.
  - Therefore, the overall average code length is $\mathbb{E}_{x \sim p}[-\log q(x)]$.
- Conclusion: The cross-entropy $H(p, q)$ is the average code length required when using the optimal encoding designed for $q$ to represent data from $p$.

#### Relative Entropy (KL Divergence) $D_{KL}(p\|q) = \sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$

- Intuition: Continuing from the concept of cross-entropy. Since encoding with the incorrect scheme $q$ is longer than encoding with the correct scheme $p$, on average, how much extra code length do we waste per sample? This "extra wasted length" represents the difference between the two distributions.
- Why is it $\sum_i p(x_i) \log \frac{p(x_i)}{q(x_i)}$?
  - For event $x_i$, the length using the incorrect scheme is $-\log q(x_i)$, and the length using the correct scheme is $-\log p(x_i)$.
  - The extra wasted length is: $(-\log q(x_i)) - (-\log p(x_i)) = \log \frac{p(x_i)}{q(x_i)}$.
  - We average this "wastage" using the true probability $p(x_i)$ to obtain the average wasted length: $\mathbb{E}_{x \sim p}[\log \frac{p(x)}{q(x)}]$.
- Conclusion: The KL divergence $D_{KL}(p\|q)$ measures the additional average code length incurred by using the approximate distribution $q$ instead of the true distribution $p$. It directly measures the difference between the two distributions.

#### Relationship $D_{KL}(p\|q) = H(p, q) - H(p) \geq 0$

Let's conduct a complete derivation:

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

[^ho_ddpm]: **Ho, Jonathan, Ajay Jain, and Pieter Abbeel.** "Denoising Diffusion Probabilistic Models." *Advances in Neural Information Processing Systems*, vol. 33, edited by H. Larochelle et al., Curran Associates, Inc., 2020, pp. 6840–6851. *NeurIPS*, https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

[^nichol_improved_ddpm]: **Nichol, Alexander Quinn, and Prafulla Dhariwal.** "Improved Denoising Diffusion Probabilistic Models." *Proceedings of the 38th International Conference on Machine Learning*, vol. 139, edited by Marina Meila and Tong Zhang, Proceedings of Machine Learning Research, 18–24 July 2021, pp. 8162–8171. *PMLR*, https://proceedings.mlr.press/v139/nichol21a.html.

[^song_consistency]: **Song, Yang, et al.** "Consistency Models." *International Conference on Machine Learning*, 2023. *ICML*, https://icml.cc/virtual/2023/poster/24593.

[^song_ddim]: **Song, Jiaming, Chenlin Meng, and Stefano Ermon.** "Denoising Diffusion Implicit Models." *International Conference on Learning Representations*, 2021. *OpenReview*, https://openreview.net/forum?id=St1giarCHLP.

[^salimans_progressive_distillation]: **Salimans, Tim, and Jonathan Ho.** "Progressive Distillation for Fast Sampling of Diffusion Models." *International Conference on Learning Representations*, 2022. *OpenReview*, https://openreview.net/forum?id=TIdIXIpzhoI.

[^salimans_improve_gan]: **Salimans, Tim, et al.** "Improved Techniques for Training GANs." *Proceedings of the 30th International Conference on Neural Information Processing Systems (NIPS'16)*, Curran Associates Inc., 2016, pp. 2234–2242. *ACM Digital Library*, https://dl.acm.org/doi/10.5555/3157096.3157346.

[^heusel_fid]: **Heusel, Martin, et al.** "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)*, Curran Associates Inc., 2017, pp. 6629–6640. *ACM Digital Library*, https://dl.acm.org/doi/10.5555/3295222.3295408.

[^mc_candlish_grad_noise]: **McCandlish, Sam, et al.** *An Empirical Model of Large-Batch Training*. 14 Dec. 2018. *arXiv*, https://arxiv.org/abs/1812.06162.

[^lilian_diffusion]: **Weng, Lilian.** "What Are Diffusion Models?" *Lil'Log*, 11 July 2021, https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.

[^lilian_ae]: **Weng, Lilian.** "From Autoencoder to Beta-VAE." *Lil'Log*, 12 Aug. 2018, https://lilianweng.github.io/posts/2018-08-12-vae/.

[^saleem_gaussian]: **Saleem, Ameer.** "Unpacking the Multivariate Gaussian Distribution." *Medium*, 12 May 2025, https://ameer-saleem.medium.com/why-the-multivariate-gaussian-distribution-isnt-as-scary-as-you-might-think-5c43433ca23b.

[^gupta_gaussian_kl]: **Gupta, Rishabh.** "KL Divergence between 2 Gaussian Distributions." *Mr. Easy*, 16 Apr. 2020, https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/.

[^zijie_cnn]: **Wang, Zijie J., et al.** "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization." *IEEE Transactions on Visualization and Computer Graphics (TVCG)*, IEEE, 2020. https://poloclub.github.io/cnn-explainer/.

[^wiki_closed]: Wikipedia contributors. "Closed-form expression." *Wikipedia, The Free Encyclopedia*, 26 July 2025, https://en.wikipedia.org/wiki/Closed-form_expression. Accessed 1 Sept. 2025.

[^wiki_jensen]: Wikipedia contributors. "Jensen's inequality." *Wikipedia, The Free Encyclopedia*, 12 June 2025, https://en.wikipedia.org/wiki/Jensen%27s_inequality. Accessed 23 Aug. 2025.

[^wiki_concave]: Wikipedia contributors. "Concave function." *Wikipedia, The Free Encyclopedia*, 17 July 2025, https://en.wikipedia.org/wiki/Concave_function. Accessed 23 Aug. 2025.
