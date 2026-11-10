---
title: "Sampling faster: DDIM and the probability-flow ODE"
slug: 058-fast-samplers
part: "Part 7 — Diffusion and flow matching"
summary: Sample a trained diffusion model in tens of steps instead of a thousand, with DDIM, the probability-flow ODE, and EDM's Heun sampler.
minutes: 40
gpu: false
objectives:
  - Map a sample between the DDPM, VP-ODE, EDM, and flow-matching time conventions, and say which end is data in each.
  - Build DDIM's deterministic and stochastic updates, and skip steps without retraining.
  - Write the probability-flow ODE and show that a DDIM step is an Euler step in EDM's coordinates.
  - Build EDM's noise levels and Heun sampler, and measure first- and second-order convergence.
prereqs:
  - 057-diffusion-from-scratch
exercise: 058-fast-samplers
---

# Sampling faster: DDIM and the probability-flow ODE

The DDPM from the previous chapter generates a sample in a thousand network evaluations. For a vocoder that must produce a second of audio in well under a second, that is hopeless. The good news is that the network does not need to change. It was trained to estimate the noise at any noise level, and that estimate defines a smooth path from noise to data. A *sampler* is a way of following that path, and a better sampler follows it in 10 to 50 steps.

This chapter builds three samplers that share one trained model: DDIM, the probability-flow ODE, and EDM's Heun method. Along the way you meet the *score* view, which explains why they all work. The [next chapter](/lessons/059-classifier-free-guidance) steers these samplers toward a condition.

## What you already know

- In DDPM, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, with data at $t = 0$ and $\bar\alpha_0 = 1$ (see [Diffusion from scratch](/lessons/057-diffusion-from-scratch)).
- A network trained on the noise-prediction loss estimates $\mathbb{E}[\epsilon \mid x_t]$, and $\hat x_0$ follows from it.
- The score is $-\hat\epsilon / \sqrt{1 - \bar\alpha_t}$.
- Euler's method steps an ODE $dx/dt = f(x, t)$ with $x \leftarrow x + h f(x, t)$, and its error shrinks in proportion to the step $h$.
- When the data is Gaussian, the ideal denoiser is known in closed form.

## Four clocks, one process

Every sampler in this chapter follows the same noising process, but papers measure its progress with different clocks. The following table lists the conventions you meet in Part 7.

| Family | Clock | Data at | Noise at | The noisy sample |
|---|---|---|---|---|
| DDPM, DDIM, DiffWave | step $t \in \{0, \ldots, T\}$ | $t = 0$ | $t = T$ | $\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ |
| VP SDE, Grad-TTS | time $t \in [0, 1]$ | $t = 0$ | $t = 1$ | $a(t)\, x_0 + \sigma(t)\, \epsilon$ |
| EDM | noise level $\sigma \in [0, 80]$ | $\sigma = 0$ | $\sigma = 80$ | $x_0 + \sigma\, n$ |
| Flow matching, Lipman convention | time $t \in [0, 1]$ | $t = 1$ | $t = 0$ | $t\, x_1 + (1 - t)\, x_0$, with noise $x_0$ |

The first three agree that data is at the start. Flow matching, which the [next part](/lessons/061-flow-matching) builds, puts *noise* at $t = 0$ and data at $t = 1$, as Lipman et al., Matcha-TTS, Voicebox, and F5-TTS do. Worse, it calls the noise $x_0$, the symbol DDPM uses for data.

The conversion you need most is short. A VP sample divided by $\sqrt{\bar\alpha_t}$ is an EDM sample, $x_0 + \sigma \epsilon$, at noise level $\sigma = \sqrt{(1 - \bar\alpha_t) / \bar\alpha_t}$.

> [!KEY]
> Before you use a formula from a paper, find which end of its clock is data and what its $x_0$ means. DDPM, DDIM, VP SDEs, and EDM put data at the start. Flow matching puts noise at the start. Every sign in the update depends on it.

## DDIM: the same model, fewer steps

Can you skip steps of a model trained for a thousand? DDIM says yes. It keeps DDPM's training and its marginals, but defines a reverse step that can jump from any step $t$ to any earlier step $t'$ (Song, Meng, and Ermon, 2021). It estimates the data, then rebuilds the sample at the lower noise level from that estimate and the same noise estimate:

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad
x_{t'} = \sqrt{\bar\alpha_{t'}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\, \hat\epsilon + \sigma z
$$

$$
\sigma(\eta) = \eta \sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}} \sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}
$$

Read it aloud: the sample at step $t'$ is the data estimate scaled for $t'$, plus the noise estimate with whatever variance is left after the fresh noise $\sigma z$. The knob $\eta$ sets how much fresh noise. At $\eta = 0$ the step is deterministic. At $\eta = 1$ and $t' = t - 1$, $\sigma^2$ equals DDPM's posterior variance $\tilde\beta_t$, and DDIM becomes DDPM.

> [!WARNING]
> The DDIM paper writes $\alpha_t$ for what DDPM calls $\bar\alpha_t$. Mixing the two notations is a common DDIM bug. This course writes $\bar\alpha$ throughout.

```python
x0_hat = (x_t - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
x_prev = np.sqrt(ab_prev) * x0_hat + np.sqrt(1 - ab_prev - sigma**2) * eps + sigma * z
```

### Work one step and one convergence run

Continue the last chapter's example: $\bar\alpha_2 = 0.72$, $\bar\alpha_1 = 0.9$, $x_2 = 1.113103$, and the perfect $\hat\epsilon = 0.5$. Then $\hat x_0 = 1$, and with $\eta = 0$, $x_1 = \sqrt{0.9} \times 1 + \sqrt{0.1} \times 0.5 = 1.106797$. With $\eta = 1$, $\sigma^2 = (0.1 / 0.28)(1 - 0.72 / 0.9) = 0.071429$, the posterior variance from the last chapter.

Next, take data $x_0 \sim \mathcal{N}(2, 0.5^2)$, the linear schedule with $T = 1000$, the exact noise predictor for that Gaussian, and one fixed start $x_{1000}$ at 1.3 standard deviations above its mean. The exact answer maps that start to the data quantile $2 + 0.5 \times 1.3 = 2.65$.

| Steps | 1 | 10 | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|
| $x_0$ | 2.002 | 2.481 | 2.561 | 2.613 | 2.631 | 2.648 |

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="DDIM output for one fixed start against the number of steps: 2.002 with one step, rising to 2.648 with 1000 steps, toward the exact 2.65">
<line x1="70" y1="230" x2="560" y2="230" class="stroke-muted"/><line x1="70" y1="230" x2="70" y2="30" class="stroke-muted"/>
<line x1="70" y1="207.77777777777777" x2="560" y2="207.77777777777777" class="stroke-grid"/><text x="62" y="211.77777777777777" class="fig-small" text-anchor="end">2.0</text>
<line x1="70" y1="163.33333333333326" x2="560" y2="163.33333333333326" class="stroke-grid"/><text x="62" y="167.33333333333326" class="fig-small" text-anchor="end">2.2</text>
<line x1="70" y1="118.88888888888889" x2="560" y2="118.88888888888889" class="stroke-grid"/><text x="62" y="122.88888888888889" class="fig-small" text-anchor="end">2.4</text>
<line x1="70" y1="74.4444444444444" x2="560" y2="74.4444444444444" class="stroke-grid"/><text x="62" y="78.4444444444444" class="fig-small" text-anchor="end">2.6</text>
<line x1="70" y1="63.333333333333314" x2="560" y2="63.333333333333314" class="stroke-bad dashed"/><text x="76" y="57.333333333333314" class="fig-small">exact answer 2.65</text>
<polyline points="70.0,207.3 233.3,100.9 282.5,83.1 347.5,71.6 396.7,67.6 560.0,63.8" class="stroke-b" fill="none" stroke-width="2"/>
<circle cx="70.0" cy="207.3" r="5" class="fill-b"/><text x="70.0" y="248" class="fig-small" text-anchor="middle">1</text>
<text x="78.0" y="225.3" class="fig-small" text-anchor="start">2.002</text>
<circle cx="233.3" cy="100.9" r="5" class="fill-b"/><text x="233.3" y="248" class="fig-small" text-anchor="middle">10</text>
<text x="241.3" y="118.9" class="fig-small" text-anchor="start">2.481</text>
<circle cx="282.5" cy="83.1" r="5" class="fill-b"/><text x="282.5" y="248" class="fig-small" text-anchor="middle">20</text>
<text x="290.5" y="101.1" class="fig-small" text-anchor="start">2.561</text>
<circle cx="347.5" cy="71.6" r="5" class="fill-b"/><text x="347.5" y="248" class="fig-small" text-anchor="middle">50</text>
<text x="355.5" y="89.6" class="fig-small" text-anchor="start">2.613</text>
<circle cx="396.7" cy="67.6" r="5" class="fill-b"/><text x="396.7" y="248" class="fig-small" text-anchor="middle">100</text>
<text x="404.7" y="85.6" class="fig-small" text-anchor="start">2.631</text>
<circle cx="560.0" cy="63.8" r="5" class="fill-b"/><text x="560.0" y="248" class="fig-small" text-anchor="middle">1000</text>
<text x="552.0" y="81.8" class="fig-small" text-anchor="end">2.648</text>
<text x="315.0" y="270" class="fig-small" text-anchor="middle">DDIM steps (log scale)</text>
<text x="70" y="20" class="fig-small">sampled x0 for one fixed starting noise</text>
</svg>
Figure 1. DDIM with eta = 0 returns the same start's data value for evenly spaced step counts. Notice that one step returns the mean, 2.002, because from pure noise the best guess of the data is its average, and that more steps recover more of the individual sample.
```

DDIM is exact only in the limit, so the next question is how to follow the path more accurately.

## The probability-flow ODE

Score-based models (Song et al., 2021) write diffusion as a stochastic differential equation, and the same marginal distributions at every $t$ come out of a deterministic ODE, the *probability-flow ODE*. For the variance-preserving (VP) process, with $\beta(t)$ rising from 0.1 to 20 over $t \in [0, 1]$ and data at $t = 0$:

$$
\frac{dx}{dt} = -\frac{1}{2} \beta(t) \left( x + \nabla_x \log p_t(x) \right)
$$

Read it aloud: the sample moves as the forward drift says, $-\frac12 \beta x$, corrected by half of $\beta$ times the score, the direction toward more likely data. Integrate it from $t = 1$ down to near 0 and noise becomes data. There is no randomness: each starting point maps to one output.

```python
beta = beta_min + t * (beta_max - beta_min)
x = x + (t_next - t) * (-0.5 * beta * (x + score(x, t)))
```

DDIM with $\eta = 0$ is a discretization of this ODE. In EDM's coordinates, $y = x_t / \sqrt{\bar\alpha_t}$ and $\sigma = \sqrt{(1 - \bar\alpha_t) / \bar\alpha_t}$, the DDIM update becomes exactly one Euler step, $y_{t'} = y_t + (\sigma_{t'} - \sigma_t)\, \hat\epsilon$. For $\bar\alpha = 0.72$, $x_2 = 1.113103$ becomes $y = 1.311804$ at $\sigma = 0.623610$.

> [!KEY]
> DDIM, the probability-flow ODE, and EDM's sampler all follow the same deterministic path from noise to data. They differ only in the clock they measure it with and the integrator they step it with.

## EDM: a better clock and a better integrator

EDM (Karras et al., 2022) strips the process to its simplest form. The noisy sample is the data plus noise of standard deviation $\sigma$, and the network is a *denoiser* $D(x; \sigma) \approx \mathbb{E}[x_0 \mid x]$. The ODE is:

$$
\frac{dx}{d\sigma} = \frac{x - D(x; \sigma)}{\sigma}
$$

Read it aloud: as the noise level falls, move the sample toward the denoiser's estimate, at a rate of one noise standard deviation per unit of $\sigma$. The score is $(D - x) / \sigma^2$, so this is the probability-flow ODE in EDM's clock.

EDM spends its steps where they matter, with more of them at low noise:

$$
\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N - 1} \left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right) \right)^{\rho}, \qquad i = 0, \ldots, N - 1, \qquad \sigma_N = 0
$$

Read it aloud: space the levels evenly in $\sigma^{1/\rho}$ between $\sigma_{\max} = 80$ and $\sigma_{\min} = 0.002$, then append 0. With $\rho = 7$ the spacing crowds toward small $\sigma$. For $N = 5$ the levels are $80$, $17.528$, $2.515$, $0.1698$, $0.002$, then 0.

EDM integrates with *Heun's method*: take an Euler step, measure the slope at the landing point, and redo the step with the average of the two slopes. That costs two denoiser calls per step, but the error shrinks with the square of the step size. The last step, to $\sigma = 0$, is plain Euler, because the slope there would divide by zero.

```python
d = (x - D(x, s)) / s
x_euler = x + (s_next - s) * d
d_next = (x_euler - D(x_euler, s_next)) / s_next
x = x + (s_next - s) * 0.5 * (d + d_next)
```

On the same Gaussian problem, with the exact answer 2.65, the **Run** measured the following errors.

| Levels $N$ | Euler error | Heun error |
|---|---|---|
| 16 | 0.112 | 0.047 |
| 32 | 0.057 | 0.010 |
| 64 | 0.029 | 0.0024 |

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Error against number of noise levels on log-log axes: Euler falls from 0.112 to 0.029 and Heun from 0.047 to 0.0024 as levels go from 16 to 64">
<line x1="80" y1="230" x2="560" y2="230" class="stroke-muted"/><line x1="80" y1="230" x2="80" y2="30" class="stroke-muted"/>
<line x1="80" y1="230.0" x2="560" y2="230.0" class="stroke-grid"/><text x="72" y="234.0" class="fig-small" text-anchor="end">0.001</text>
<line x1="80" y1="143.1" x2="560" y2="143.1" class="stroke-grid"/><text x="72" y="147.1" class="fig-small" text-anchor="end">0.01</text>
<line x1="80" y1="56.2" x2="560" y2="56.2" class="stroke-grid"/><text x="72" y="60.2" class="fig-small" text-anchor="end">0.1</text>
<text x="160.0" y="248" class="fig-small" text-anchor="middle">16</text>
<text x="320.0" y="248" class="fig-small" text-anchor="middle">32</text>
<text x="480.0" y="248" class="fig-small" text-anchor="middle">64</text>
<polyline points="160.0,51.9 320.0,77.4 480.0,102.9" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="160.0" cy="51.9" r="5" class="fill-b"/><text x="168.0" y="43.9" class="fig-small">0.112</text>
<circle cx="320.0" cy="77.4" r="5" class="fill-b"/><text x="328.0" y="69.4" class="fig-small">0.057</text>
<circle cx="480.0" cy="102.9" r="5" class="fill-b"/><text x="488.0" y="94.9" class="fig-small">0.029</text>
<text x="510.0" y="106.9" class="fig-small fig-bold">Euler</text>
<polyline points="160.0,84.7 320.0,143.1 480.0,197.0" class="stroke-a" fill="none" stroke-width="2.5"/>
<circle cx="160.0" cy="84.7" r="5" class="fill-a"/><text x="168.0" y="76.7" class="fig-small">0.047</text>
<circle cx="320.0" cy="143.1" r="5" class="fill-a"/><text x="328.0" y="135.1" class="fig-small">0.01</text>
<circle cx="480.0" cy="197.0" r="5" class="fill-a"/><text x="488.0" y="189.0" class="fig-small">0.0024</text>
<text x="510.0" y="201.0" class="fig-small fig-bold">Heun</text>
<text x="320.0" y="270" class="fig-small" text-anchor="middle">noise levels N (log scale)</text>
<text x="80" y="20" class="fig-small">error against the exact 2.65 (log scale)</text>
</svg>
Figure 2. The same errors on log-log axes. Notice that each doubling of N halves Euler's error and quarters Heun's, which is what first- and second-order convergence look like.
```

At $N = 4$ or 8, Heun is worse than Euler: the first step, from 80 to 17.5, is too long for the correction to help.

> [!RECAP]
> - DDPM, DDIM, VP ODEs, and EDM put data at the start of their clocks. Flow matching puts noise at $t = 0$.
> - DDIM rebuilds $x_{t'}$ from $\hat x_0$ and $\hat\epsilon$, with fresh noise $\sigma(\eta)$, and $\eta = 0$ is deterministic.
> - The probability-flow ODE is $dx/dt = -\frac12 \beta (x + \text{score})$, and DDIM is its Euler step in EDM's coordinates.
> - EDM's levels follow $\rho = 7$ from 80 to 0.002, then 0, and Heun averages two slopes on every step but the last.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ddim_sigma(ab_t, ab_prev, eta)` and `ddim_step(x_t, eps_hat, ab_t, ab_prev, eta, z)`.** Form $\hat x_0$, then $x_{t'}$. Common mistake: using $\bar\alpha_t$ where the formula needs $\bar\alpha_{t'}$.
2. **`ddim_sample(eps_fn, x_T, timesteps, alpha_bar, eta, rng)`.** Append 0 to the steps and loop over consecutive pairs. Common mistake: stopping at the last listed step, which leaves a little noise in every sample.
3. **`vp_to_edm(x_t, ab)`.** Two lines.
4. **`pf_ode_sample(score_fn, x1, n_steps, ...)`.** Euler over `np.linspace(1, t_end, n_steps + 1)`. Common mistake: a positive step size, which runs the ODE toward noise.
5. **`edm_sigmas(N, ...)`.** Evaluate the $\rho$ formula at `np.arange(N)` and append 0.
6. **`edm_sample(denoiser, x, sigmas, heun)`.** Loop over consecutive levels, with the Heun correction only when the next level is above 0.

## Cost and scaling

Every sampler's cost is its number of network evaluations, since one evaluation dwarfs the arithmetic around it. DDIM and Euler need $N$, and EDM Heun needs $2N - 1$.

## Where this breaks

**Stochastic sampling needs more steps.** With $\eta = 1$ and 100 steps on the Gaussian problem, the samples came out with a standard deviation of 0.469 instead of 0.5, and with 1000 steps 0.497. Fresh noise only averages out when the steps are small.

**The wrong convention.** A formula copied between the DDPM clock and the flow-matching clock runs without an error and gives worse samples. Test every sampler on a Gaussian, where the right answer is known.

## The exercise

You build `ddim_sigma`, `ddim_step`, `ddim_sample`, `vp_to_edm`, `pf_ode_sample`, `edm_sigmas`, and `edm_sample`; `vp_marginal` is provided. The checks reproduce $x_1 = 1.106797$, confirm that $\eta = 1$ gives DDPM's posterior variance, and run DDIM with the exact Gaussian denoiser at 1, 10, 50, and 1000 steps against the table. They check that a DDIM step is an Euler step in $\sigma$, and that your probability-flow ODE, Euler, and Heun converge at first, first, and second order.

Press **Run** to see DDIM converge toward 2.65 and to see Euler and Heun errors against network evaluations.
