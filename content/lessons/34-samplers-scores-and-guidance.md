---
title: Samplers, scores, and guidance
slug: 34-samplers-scores-and-guidance
part: "Part 7 — Diffusion and flow matching"
summary: Sample a trained diffusion model in tens of steps instead of a thousand, and steer it toward a condition with classifier-free guidance.
minutes: 70
gpu: false
objectives:
  - Map a sample between the DDPM, VP-ODE, EDM, and flow-matching time conventions, and say which end is data in each.
  - Build DDIM's deterministic and stochastic updates, and skip steps without retraining.
  - Write the probability-flow ODE and show that a DDIM step is an Euler step in EDM's coordinates.
  - Build EDM's noise levels and Heun sampler, and measure first- and second-order convergence.
  - Apply classifier-free guidance, convert between its two scale conventions, and predict what it does to a distribution.
prereqs:
  - 33-diffusion-from-scratch
exercise: 34-samplers-scores-and-guidance
---

# Samplers, scores, and guidance

The DDPM from the last lesson generates a sample in a thousand network
evaluations. For a vocoder that must produce a second of audio in well under
a second, that is hopeless. The good news is that the network doesn't need to
change. It was trained to estimate the noise at any noise level, and that
estimate defines a smooth path from noise to data. A *sampler* is a way of
following that path, and a better sampler follows it in 10 to 50 steps.

This lesson builds three samplers that share one trained model: DDIM, the
probability-flow ODE, and EDM's Heun method. Along the way you meet the
*score* view, which explains why they all work, and a table of the time
conventions that different papers use.

Then you steer. A model trained with and without its condition, such as a
text prompt or a speaker, can be pushed toward the condition at sampling
time. That trick, *classifier-free guidance*, is in nearly every text-to-audio
and TTS system that uses diffusion or flow matching.

## What you already know

- In DDPM, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, with data at $t = 0$ and $\bar\alpha_0 = 1$.
- A network trained on the noise-prediction loss estimates $\mathbb{E}[\epsilon \mid x_t]$, and $\hat x_0$ follows from it.
- The score is $-\hat\epsilon / \sqrt{1 - \bar\alpha_t}$.
- Euler's method steps an ODE $dx/dt = f(x, t)$ with $x \leftarrow x + h f(x, t)$, and its error shrinks in proportion to the step $h$.
- When the data is Gaussian, the ideal denoiser is known in closed form.

## Four clocks, one process

Every sampler in this lesson follows the same noising process, but papers
measure its progress with different clocks. The following table lists the
conventions you meet in Part 7.

| Family | Clock | Data at | Noise at | The noisy sample |
|---|---|---|---|---|
| DDPM, DDIM, DiffWave | step $t \in \{0, \ldots, T\}$ | $t = 0$ | $t = T$ | $\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ |
| VP SDE, Grad-TTS | time $t \in [0, 1]$ | $t = 0$ | $t = 1$ | $a(t)\, x_0 + \sigma(t)\, \epsilon$ |
| EDM | noise level $\sigma \in [0, 80]$ | $\sigma = 0$ | $\sigma = 80$ | $x_0 + \sigma\, n$ |
| Flow matching, Lipman convention | time $t \in [0, 1]$ | $t = 1$ | $t = 0$ | $t\, x_1 + (1 - t)\, x_0$, with noise $x_0$ |

The first three agree that data is at the start. Flow matching, which the
lesson on flow matching builds, puts *noise* at $t = 0$ and data at $t = 1$,
as Lipman et al., Matcha-TTS, Voicebox, and F5-TTS do. Worse, it calls the
noise $x_0$, the symbol DDPM uses for data.

The conversions are short. A VP sample divided by $\sqrt{\bar\alpha_t}$ is an
EDM sample, $x_0 + \sigma \epsilon$, at noise level
$\sigma = \sqrt{(1 - \bar\alpha_t) / \bar\alpha_t}$. A flow-matching time
$t_{\text{FM}}$ is the same moment as the "data at 0" time
$\tau = 1 - t_{\text{FM}}$, at which a rectified-flow sample
$(1 - \tau)\, x_{\text{data}} + \tau\, \epsilon$ corresponds to EDM's
$\sigma = \tau / (1 - \tau)$.

> [!KEY]
> Before you use a formula from a paper, find which end of its clock is data and what its $x_0$ means. DDPM, DDIM, VP SDEs, and EDM put data at the start; flow matching puts noise at the start. Every sign in the update depends on it.

## DDIM: the same model, fewer steps

DDIM keeps DDPM's training and its marginals, but defines a reverse step that
can jump from any step $t$ to any earlier step $t'$ (Song, Meng, and Ermon,
2021). It estimates the data, then rebuilds the sample at the lower noise
level from that estimate and the same noise estimate:

$$
\hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t}\, \hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad
x_{t'} = \sqrt{\bar\alpha_{t'}}\, \hat x_0 + \sqrt{1 - \bar\alpha_{t'} - \sigma^2}\, \hat\epsilon + \sigma z
$$

$$
\sigma(\eta) = \eta \sqrt{\frac{1 - \bar\alpha_{t'}}{1 - \bar\alpha_t}} \sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t'}}}
$$

Read it aloud: the sample at step $t'$ is the data estimate scaled for $t'$,
plus the noise estimate with whatever variance is left after the fresh noise
$\sigma z$. The knob $\eta$ sets how much fresh noise. At $\eta = 0$ the step
is deterministic. At $\eta = 1$ and $t' = t - 1$, $\sigma^2$ equals DDPM's
posterior variance $\tilde\beta_t$, and DDIM becomes DDPM.

> [!WARNING]
> The DDIM paper writes $\alpha_t$ for what DDPM calls $\bar\alpha_t$. Mixing the two notations is a common DDIM bug. This course writes $\bar\alpha$ throughout.

```python
x0_hat = (x_t - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
x_prev = np.sqrt(ab_prev) * x0_hat + np.sqrt(1 - ab_prev - sigma**2) * eps + sigma * z
```

### Work one step and one convergence run

Continue the last lesson's example: $\bar\alpha_2 = 0.72$, $\bar\alpha_1 = 0.9$,
$x_2 = 1.113103$, and the perfect $\hat\epsilon = 0.5$. Then $\hat x_0 = 1$, and
with $\eta = 0$,
$x_1 = \sqrt{0.9} \times 1 + \sqrt{0.1} \times 0.5 = 1.106797$. With
$\eta = 1$, $\sigma^2 = (0.1 / 0.28)(1 - 0.72 / 0.9) = 0.071429$, the
posterior variance from the last lesson.

Next, take data $x_0 \sim \mathcal{N}(2, 0.5^2)$, the linear schedule with
$T = 1000$, the exact noise predictor for that Gaussian, and one fixed start
$x_{1000}$ at 1.3 standard deviations above its mean. The exact answer maps
that start to the data quantile $2 + 0.5 \times 1.3 = 2.65$. The following
table shows what DDIM with $\eta = 0$ returns for evenly spaced steps.

| Steps | 1 | 10 | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|
| $x_0$ | 2.002 | 2.481 | 2.561 | 2.613 | 2.631 | 2.648 |

One step returns the mean, because from pure noise the best guess of the data
is its average. More steps recover more of the individual sample. DDIM is
exact only in the limit, so the next question is how to follow the path
more accurately with each step.

## The probability-flow ODE

Score-based models (Song et al., 2021) write diffusion as a stochastic
differential equation, with data at $t = 0$. For the variance-preserving (VP) process, with
$\beta(t)$ rising from 0.1 to 20 over $t \in [0, 1]$, noising follows
$dx = -\tfrac12 \beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw$. The same marginal
distributions at every $t$ come out of a deterministic ODE, the
*probability-flow ODE*:

$$
\frac{dx}{dt} = -\frac{1}{2} \beta(t) \left( x + \nabla_x \log p_t(x) \right)
$$

Read it aloud: the sample moves as the forward drift says, $-\frac12 \beta x$,
corrected by half of $\beta$ times the score, the direction toward more
likely data. Integrate it from $t = 1$ down to near 0 and noise becomes data.
There is no randomness: each starting point maps to one output.

```python
beta = beta_min + t * (beta_max - beta_min)
x = x + (t_next - t) * (-0.5 * beta * (x + score(x, t)))
```

DDIM with $\eta = 0$ is a discretization of this ODE. In EDM's coordinates,
$y = x_t / \sqrt{\bar\alpha_t}$ and $\sigma = \sqrt{(1 - \bar\alpha_t) / \bar\alpha_t}$,
the DDIM update becomes exactly one Euler step,
$y_{t'} = y_t + (\sigma_{t'} - \sigma_t)\, \hat\epsilon$. For
$\bar\alpha = 0.72$, $x_2 = 1.113103$ becomes $y = 1.311804$ at
$\sigma = 0.623610$.

> [!KEY]
> DDIM, the probability-flow ODE, and EDM's sampler all follow the same
> deterministic path from noise to data. They differ only in the clock they
> measure it with and the integrator they step it with.

## EDM: a better clock and a better integrator

EDM (Karras et al., 2022) strips the process to its simplest form. The noisy
sample is the data plus noise of standard deviation $\sigma$, and the network
is a *denoiser*
$D(x; \sigma) \approx \mathbb{E}[x_0 \mid x]$. The ODE is:

$$
\frac{dx}{d\sigma} = \frac{x - D(x; \sigma)}{\sigma}
$$

Read it aloud: as the noise level falls, move the sample toward the
denoiser's estimate, at a rate of one noise standard deviation per unit of
$\sigma$. The score is $(D - x) / \sigma^2$, so this is the probability-flow
ODE in EDM's clock. EDM wraps the raw network with scalings
$c_{\text{skip}}$, $c_{\text{out}}$, and $c_{\text{in}}$ that keep its inputs
and targets at unit variance, with $\sigma_{\text{data}} = 0.5$.

EDM spends its steps where they matter, with more of them at low noise:

$$
\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N - 1} \left( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \right) \right)^{\rho}, \qquad i = 0, \ldots, N - 1, \qquad \sigma_N = 0
$$

Read it aloud: space the levels evenly in $\sigma^{1/\rho}$ between
$\sigma_{\max} = 80$ and $\sigma_{\min} = 0.002$, then append 0. With
$\rho = 7$ the spacing crowds toward small $\sigma$. For $N = 5$ the levels
are $80$, $17.528$, $2.515$, $0.1698$, $0.002$, then 0.

EDM integrates with *Heun's method*: take an Euler step, measure the slope
at the landing point, and redo the step with the average of the two slopes.
That costs two denoiser calls per step, but the error shrinks with the
square of the step size. The last step, to $\sigma = 0$, is plain Euler,
because the slope there would divide by zero.

```python
d = (x - D(x, s)) / s
x_euler = x + (s_next - s) * d
d_next = (x_euler - D(x_euler, s_next)) / s_next
x = x + (s_next - s) * 0.5 * (d + d_next)
```

On the same Gaussian problem, with the exact answer 2.65, the **Run**
measured the following errors.

| Levels $N$ | Euler error | Heun error |
|---|---|---|
| 16 | 0.112 | 0.047 |
| 32 | 0.057 | 0.010 |
| 64 | 0.029 | 0.0024 |

Each doubling halves Euler's error and quarters Heun's. At $N = 4$ or 8, Heun
is worse than Euler: the first step, from 80 to 17.5, is too long for the
correction to help.

## Classifier-free guidance

A conditional model estimates $\epsilon(x, c)$ for a condition $c$, such as a
transcript. Train the same network to also work without the condition, by
replacing $c$ with a null token $\varnothing$ on a random 10 to 20 percent of
examples. At sampling time, extrapolate from the unconditional prediction
past the conditional one (Ho and Salimans, 2022):

$$
\tilde\epsilon(x, c) = (1 + w)\, \epsilon(x, c) - w\, \epsilon(x, \varnothing)
$$

Read it aloud: take the conditional prediction, and add $w$ times the
difference between the conditional and unconditional predictions. At
$w = 0$ there is no guidance. In score terms, the guided score is
$(1 + w) \nabla \log p(x \mid c) - w \nabla \log p(x)$, the score of
$p(x \mid c)\, p(c \mid x)^w$. Guidance favors samples that a classifier would
confidently assign to $c$.

```python
guided = (1 + w) * cond - w * uncond
```

The same combination works for denoisers, scores, and flow-matching
velocities. Two conventions exist for the scale. This lesson's $w$ has no
guidance at 0. Diffusers, AudioLDM, and Tango write
$\epsilon_\varnothing + s (\epsilon_c - \epsilon_\varnothing)$, with no
guidance at $s = 1$, so $s = 1 + w$. F5-TTS's default of 2 in the $w$
convention is $s = 3$, and CosyVoice 2's 0.7 is $s = 1.7$.

### Hear guidance narrow a distribution

The **Run** models pitches in semitones as two classes, low notes around 57
and high notes around 69, each with a standard deviation of 2. It samples
the high class with EDM's Heun sampler and exact denoisers:

- With $w = 0$: mean 68.82, standard deviation 2.02, close to the class itself.
- With $w = 1$: mean 70.05, standard deviation 1.43.
- With $w = 4$: mean 71.81, standard deviation 1.05.

Guidance pushes samples away from the other class and squeezes them together.
You hear it as twelve notes that wander at $w = 0$ and cluster high and close
at $w = 4$. In TTS the same effect gives clearer, more on-script speech with
less variety, and too much gives strained, over-bright audio.

> [!RECAP]
> - DDPM, DDIM, VP ODEs, and EDM put data at the start of their clocks; flow matching puts noise at $t = 0$.
> - DDIM rebuilds $x_{t'}$ from $\hat x_0$ and $\hat\epsilon$, with fresh noise $\sigma(\eta)$; $\eta = 0$ is deterministic.
> - The probability-flow ODE is $dx/dt = -\frac12 \beta (x + \text{score})$, and DDIM is its Euler step in EDM's coordinates.
> - EDM's levels follow $\rho = 7$ from 80 to 0.002, then 0, and Heun averages two slopes on every step but the last.
> - Guidance is $(1 + w) f_c - w f_\varnothing$, with $s = 1 + w$ in the other convention.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ddim_sigma(ab_t, ab_prev, eta)` and `ddim_step(x_t, eps_hat, ab_t, ab_prev, eta, z)`.** Form $\hat x_0$, then $x_{t'}$. Common mistake: using $\bar\alpha_t$ where the formula needs $\bar\alpha_{t'}$.
2. **`ddim_sample(eps_fn, x_T, timesteps, alpha_bar, eta, rng)`.** Append 0 to the steps and loop over consecutive pairs. Common mistake: stopping at the last listed step, which leaves a little noise in every sample.
3. **`vp_to_edm(x_t, ab)`.** Two lines.
4. **`pf_ode_sample(score_fn, x1, n_steps, ...)`.** Euler over `np.linspace(1, t_end, n_steps + 1)`. Common mistake: a positive step size, which runs the ODE toward noise.
5. **`edm_sigmas(N, ...)`.** Evaluate the $\rho$ formula at `np.arange(N)` and append 0.
6. **`edm_sample(denoiser, x, sigmas, heun)`.** Loop over consecutive levels, with the Heun correction only when the next level is above 0.
7. **`cfg_combine(cond, uncond, w)`.** One line. Common mistake: the $s$ convention, which is one unit weaker at the same number.

## Cost and scaling

Every sampler's cost is its number of network evaluations (NFE), since one
evaluation dwarfs the arithmetic around it. The following table compares
them.

| Sampler | NFE for $N$ steps | Error order | Random |
|---|---|---|---|
| DDPM ancestral | $N = T$ | needs every step | yes |
| DDIM, $\eta = 0$ | $N$ | 1 | no |
| Euler on the ODE | $N$ | 1 | no |
| EDM Heun | $2N - 1$ | 2 | no |
| Any of these with guidance | twice as many | unchanged | unchanged |

Guidance doubles the cost unless you run the conditional and unconditional
passes as one batch of two, which most implementations do. All of the
**Run**'s experiments, with exact denoisers, take well under a second.

## Where this breaks

**Stochastic sampling needs more steps.** With $\eta = 1$ and 100 steps on
the Gaussian problem, the samples came out with a standard deviation of 0.469
instead of 0.5. With all 1000 steps it was 0.497. Fresh noise at every step
only averages out when the steps are small.

**Heun with too few levels.** Below about 16 levels the first steps are too
long, and Heun's correction overshoots. Use Euler or more levels.

**Guidance too strong.** Large $w$ oversaturates and removes variety.
MaskGCT and others rescale the guided prediction back toward the conditional
prediction's standard deviation.

**A null condition that differs from training.** The unconditional branch
must see exactly the null input it was trained with: a zeroed embedding, a
learned null token, or dropped prompt audio.

**The wrong convention.** A DDIM step written with the flow-matching $t$, or
a guidance scale copied between the $w$ and $s$ conventions, runs without an
error and gives worse samples. Test every sampler on a Gaussian, where the
right answer is known.

## The exercise

You build `ddim_sigma`, `ddim_step`, `ddim_sample`, `vp_to_edm`,
`pf_ode_sample`, `edm_sigmas`, `edm_sample`, and `cfg_combine`; `vp_marginal`
is provided. The checks reproduce $x_1 = 1.106797$, confirm that $\eta = 1$
gives DDPM's posterior variance, and run DDIM with the exact Gaussian
denoiser at 1, 10, 50, and 1000 steps against the table. They check that
stochastic DDIM samples the right Gaussian, that a DDIM step is an Euler step
in $\sigma$, and that your probability-flow ODE, Euler, and Heun converge at
first, first, and second order. The last checks test the guided score against
$\nabla \log p(x \mid c) p(c \mid x)^w$ and that guidance moves and narrows a
two-class distribution.

Press **Run** to see DDIM converge toward 2.65, see Euler and Heun errors
against network evaluations, and hear twelve guided notes at $w = 0$ and
$w = 4$.
