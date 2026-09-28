---
title: Diffusion from scratch
slug: 33-diffusion-from-scratch
part: "Part 7 — Diffusion and flow matching"
summary: Destroy data with Gaussian noise one step at a time, learn to predict the noise, and run the process backward to generate.
minutes: 70
gpu: false
objectives:
  - "State the DDPM time convention: data at t = 0, noise at t = T, and index schedules so that entry t means step t."
  - Build linear and cosine noise schedules and the closed-form jump from data to any step.
  - Convert between noise, data, velocity, and score predictions.
  - Derive and compute the posterior mean and variance of one reverse step.
  - Train a small DDPM with the noise-prediction loss and sample from it by ancestral sampling.
prereqs:
  - 28-normalizing-flows-and-vits
exercise: 33-diffusion-from-scratch
---

# Diffusion from scratch

Take a clip of speech and add a little noise. Add a little more. After a
thousand small steps, nothing of the speech is left: the clip is white
noise. That direction is easy, and you can write it down exactly. A
*diffusion model* learns the other direction. It is trained to remove one
small step of noise at a time, so that starting from fresh noise and
repeating the removal a thousand times produces a clip that was never
recorded.

The trick that makes this trainable is that the noising is Gaussian, so you
can jump from clean data to any noise level in one line. Training becomes a
plain regression: show the network a noisy example and ask it for the noise
that was added.

This lesson builds the whole machine, the schedule, the forward jump, the
four equivalent things a network can predict, the exact reverse step, the
loss, and the sampler. You hear it first on a voice-like signal with a
denoiser you can compute exactly, and then train one on a toy distribution.
The next two lessons speed up sampling and apply the idea to vocoders.

## What you already know

- A normalizing flow generates by an exact invertible map, and trains by maximum likelihood.
- If $\epsilon \sim \mathcal{N}(0, 1)$, then $a + b\epsilon \sim \mathcal{N}(a, b^2)$.
- The product of two Gaussian densities in the same variable is another Gaussian, up to a constant.
- An SNR in decibels is $10 \log_{10}$ of a ratio of powers.
- Mean squared error regression predicts the conditional mean of the target.

## The time convention

Papers disagree about which end of time is data. This course picks one for
all of Part 7's diffusion lessons, and you must hold on to it.

> [!KEY]
> In this lesson and the next two, **$t = 0$ is the data and $t = T$ is noise**, as in DDPM, DiffWave, and Grad-TTS. Steps run $t = 1, \ldots, T$. Every schedule array has $T + 1$ entries, so `alpha_bar[t]` means step $t$, `betas[0] = 0`, and `alpha_bar[0] = 1`.

The lesson on flow matching uses the opposite convention, noise at $t = 0$,
as Lipman's flow matching and F5-TTS do. The next lesson shows how to move
between the two.

## The forward process

Each forward step shrinks the signal slightly and adds a little noise
(Ho et al., 2020):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\right)
$$

Read it aloud: to get $x_t$, scale $x_{t-1}$ by $\sqrt{1 - \beta_t}$ and add
Gaussian noise of variance $\beta_t$. The numbers $\beta_1, \ldots, \beta_T$
are the *noise schedule*. The shrink keeps the total variance from growing:
if $x_{t-1}$ has unit variance, so does $x_t$.

Because every step is Gaussian, many steps compose into one. Write
$\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t} \alpha_s$. Then:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Read it aloud: the noisy sample at step $t$ is the clean data scaled by the
square root of $\bar\alpha_t$, plus fresh standard noise scaled by the square
root of $1 - \bar\alpha_t$. As $t$ grows, $\bar\alpha_t$ falls from 1 toward
0, and the data fades into the noise.

```python
ab = alpha_bar[t].reshape(-1, *[1] * (x0.dim() - 1))   # one step per example
x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
```

### Schedules

DDPM's *linear* schedule raises $\beta_t$ from $10^{-4}$ to $0.02$ over
$T = 1000$ steps, which leaves $\bar\alpha_T = 4.04 \times 10^{-5}$. The
*cosine* schedule of Nichol and Dhariwal sets $\bar\alpha_t$ directly, so it
falls smoothly instead of collapsing early:

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \qquad s = 0.008
$$

Read it aloud: $\bar\alpha_t$ follows a squared cosine from 1 at $t = 0$ to
almost 0 at $t = T$. The small offset $s$ keeps the first steps from being
too tiny. Each $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ is clipped at
0.999, because the last step would otherwise be exactly 1.

### Work four steps by hand

Take $T = 4$ and $\beta = (0.1, 0.2, 0.3, 0.4)$, with signal-to-noise ratio
$\text{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$ for unit-variance data.

| $t$ | $\beta_t$ | $\alpha_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\sqrt{1 - \bar\alpha_t}$ | SNR |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 1 | 1 | 0 | $\infty$ |
| 1 | 0.1 | 0.9 | 0.9 | 0.948683 | 0.316228 | 9.0 |
| 2 | 0.2 | 0.8 | 0.72 | 0.848528 | 0.529150 | 2.571 |
| 3 | 0.3 | 0.7 | 0.504 | 0.709930 | 0.704273 | 1.016 |
| 4 | 0.4 | 0.6 | 0.3024 | 0.549909 | 0.835225 | 0.433 |

With $x_0 = 1$ and $\epsilon = 0.5$ at $t = 2$:
$x_2 = 0.848528 \times 1 + 0.529150 \times 0.5 = 1.113103$. The four-step
cosine schedule gives $\bar\alpha = (0.847012, 0.493844, 0.144272)$ for the
first three steps and needs the clip on the fourth, where $\beta_4 = 0.999$.

> [!WARNING]
> A speech waveform in $[-1, 1]$ has a standard deviation near 0.1, not 1. Its SNR at step $t$ is $0.01\,\bar\alpha_t / (1 - \bar\alpha_t)$, 20 dB lower than the table says. A schedule that looks half-noisy for unit-variance data is almost pure noise for audio.

## Four things a network can predict

Given $x_t$ and $t$, a network can predict any of four quantities, and each
determines the others. Write $a_t = \sqrt{\bar\alpha_t}$ and
$\sigma_t = \sqrt{1 - \bar\alpha_t}$, so $x_t = a_t x_0 + \sigma_t \epsilon$
and $a_t^2 + \sigma_t^2 = 1$.

- **Noise** $\hat\epsilon$, which DDPM and DiffWave predict.
- **Data** $\hat x_0 = (x_t - \sigma_t \hat\epsilon) / a_t$.
- **Velocity** $v = a_t \epsilon - \sigma_t x_0$, which Stable Audio uses. It converts back with $\hat x_0 = a_t x_t - \sigma_t \hat v$ and $\hat\epsilon = \sigma_t x_t + a_t \hat v$.
- **Score** $\nabla_{x_t} \log p(x_t) = -\hat\epsilon / \sigma_t$, the direction in which the density rises fastest.

Read the score formula aloud: the noise that was added points away from the
data, so the direction back toward likely data is minus the noise, divided
by the noise scale. For one fixed $x_0$, the density $q(x_t \mid x_0)$ is a
Gaussian with mean $a_t x_0$ and variance $\sigma_t^2$. Its log-density has
gradient $-(x_t - a_t x_0) / \sigma_t^2 = -\epsilon / \sigma_t$, which is
exactly this formula.

```python
x0_hat = (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
score = -eps_hat / (1 - ab).sqrt()
```

Continue the worked example at $t = 2$. The velocity is
$v = 0.848528 \times 0.5 - 0.529150 \times 1 = -0.104886$. Converting back,
$0.848528 \times 1.113103 - 0.529150 \times (-0.104886) = 1.000000$, which is
$x_0$. The score is $-0.5 / 0.529150 = -0.944911$.

## One step back

To reverse one step, you want $q(x_{t-1} \mid x_t)$. That distribution is
intractable, because it depends on all the data. But if you also know
$x_0$, Bayes' rule gives a Gaussian in closed form:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I), \qquad
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t, \qquad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t
$$

Read it aloud: the previous step's mean is a weighted blend of the clean data
and the current noisy sample, and its variance is a fraction of $\beta_t$.
The blend comes from multiplying two Gaussians in $x_{t-1}$: the prior
$q(x_{t-1} \mid x_0)$ and the likelihood $q(x_t \mid x_{t-1})$. At $t = 1$,
$\bar\alpha_0 = 1$, so the variance is 0 and the mean is $x_0$ itself.

At sampling time you don't know $x_0$, so you plug in the network's
estimate. Substituting $\hat x_0$ from $\hat\epsilon$ gives the DDPM reverse
step:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\epsilon_\theta(x_t, t) \right) + \sqrt{\tilde\beta_t}\, z, \qquad z \sim \mathcal{N}(0, I)
$$

Read it aloud: remove a scaled copy of the predicted noise, undo the shrink
by dividing by $\sqrt{\alpha_t}$, and add a little fresh noise, except at the
last step, where $z = 0$.

For the worked example with the perfect prediction $\hat\epsilon = 0.5$:
$\tilde\mu_2 = (1.113103 - 0.2 / 0.529150 \times 0.5) / \sqrt{0.8} = 1.033199$,
and the formula in terms of $x_0$ gives the same number. The variance is
$\tilde\beta_2 = (0.1 / 0.28) \times 0.2 = 0.071429$, a standard deviation of
0.267261.

## Train by predicting the noise

The DDPM training loss is a regression (Ho et al., 2020, equation 14):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \left\lVert \epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\ t\right) \right\rVert^2
$$

Read it aloud: pick a training example, a step $t$ uniformly from 1 to $T$,
and fresh noise $\epsilon$. Noise the example to step $t$ in one jump, and
penalize the squared difference between the true noise and the network's
guess. Because squared error targets the conditional mean, the ideal network
predicts $\mathbb{E}[\epsilon \mid x_t]$, which is exactly what the reverse
step needs.

```python
t = torch.randint(1, T + 1, (B,), generator=g)
eps = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
loss = ((model(q_sample(x0, t, eps, alpha_bar), t) - eps) ** 2).mean()
```

The loss is an ELBO, the bound that VAEs maximize, with its per-step weights
dropped. The dropped weights would emphasize the nearly clean steps. Leaving
them out trains the network more evenly across noise levels, and gave better
samples in the DDPM paper.

> [!KEY]
> Training is plain regression: jump the data to a random step in one line and
> predict the noise. Only sampling pays for all $T$ steps.

## Hear it with an exact denoiser

When the data itself is Gaussian, the ideal noise predictor is known in
closed form, so you can test a sampler with no training at all. The
**Run** uses a voice-like Gaussian signal: random noise whose power spectrum
has harmonics of 140 Hz under three formant peaks, with a standard deviation
of 0.1. For such a signal, $\mathbb{E}[\epsilon \mid x_t]$ is a linear filter.
In each frequency bin $k$ with data power $P_k$, it multiplies $x_t$ by
$\sigma_t / (\bar\alpha_t P_k + 1 - \bar\alpha_t)$. It is the Wiener filter
from signal processing, rearranged to predict the noise instead of the
signal.

With a linear schedule from $10^{-4}$ to 0.1 over 200 steps, the signal's SNR
is 20 dB at step 1, $-2$ dB at step 8, $-12$ dB at step 25, and $-65$ dB at
step 200. You hear $x_{200}$, $x_{25}$, $x_8$, and $x_0$ on the way back. The
voice emerges only in the last few steps, because a small-amplitude signal
spends most of the schedule buried.

The following demo noises a mixture of three Gaussians and walks it back
with the exact score, so it needs no network either.

```demo
{"id": "diffusion-1d", "caption": "Drag the time step from 0 to T and watch the three-bump mixture fade into noise, then follow the sampling paths back from t = T and notice where each one commits to a bump."}
```

> [!RECAP]
> - Data is at $t = 0$, and arrays of length $T + 1$ make `alpha_bar[t]` mean step $t$.
> - $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$, with $\bar\alpha_t$ indexed per example and broadcast.
> - Noise, data, velocity, and score predictions convert into one another with $a_t$ and $\sigma_t$.
> - The posterior $q(x_{t-1} \mid x_t, x_0)$ has mean $\tilde\mu_t$ and variance $\tilde\beta_t$, and the DDPM step plugs in $\hat\epsilon$.
> - Training regresses the noise at a uniformly random step, and sampling adds no noise on the last step.

## Build it step by step

Build the functions in the order the checks test them.

1. **`linear_betas(T, beta_1, beta_T)` and `alpha_bars(betas)`.** Concatenate a zero with `torch.linspace`, and take `torch.cumprod(1 - betas, 0)`. Common mistake: a length-$T$ array, which shifts every index by one.
2. **`cosine_betas(T, s)`.** Evaluate $f$ at `torch.arange(T + 1)`, divide by $f(0)$, take ratios of neighbors, and clamp at 0.999.
3. **`q_sample(x0, t, eps, alpha_bar)`.** Index `alpha_bar[t]` and reshape it to `(B, 1, ..., 1)`. Common mistake: broadcasting a `(B,)` tensor against `(B, C, L)`, which errors, or silently runs along the last axis when $L = B$. The check uses exactly that shape.
4. **`predict_x0`, `velocity`, `from_velocity`, and `score_from_eps`.** One line each.
5. **`posterior(x0, x_t, t, betas, alpha_bar)`.** The two coefficients and $\tilde\beta_t$. Common mistake: using $\bar\alpha_t$ where the formula needs $\bar\alpha_{t-1}$.
6. **`ddpm_loss(model, x0, alpha_bar, generator)`.** Draw $t$ from `torch.randint(1, T + 1, ...)`, not from 0, which is the clean data.
7. **`ddpm_sample(model, shape, betas, alpha_bar, generator, keep)`.** Loop `t` from $T$ down to 1, apply the reverse step, and add noise only while $t > 1$. Save a copy of $x$ at every step in `keep`.

## Cost and scaling

The following table shows how the cost grows with the number of steps $T$
and the cost $C$ of one network evaluation.

| Operation | Network evaluations | Notes |
|---|---|---|
| One training step | 1 per example | Any $t$ in one jump, so $T$ doesn't matter |
| Ancestral sampling | $T$ | Strictly sequential |
| Memory at sampling | One copy of $x$ | Plus the network's activations |

Training costs the same as any regression, which is why diffusion trains so
stably. Sampling is the problem. A thousand sequential network evaluations
per sample made DDPM far slower than a GAN vocoder's single pass. The next
lesson cuts that to 10 to 50 steps without retraining, and DiffWave's fast
schedule in the lesson after cuts it to 6. The whole **Run**, including
3,000 training steps and 4,000 samples on the 2-D ring at $T = 100$, takes
about 5 seconds on a laptop CPU.

## Where this breaks

**Off-by-one steps.** Papers index steps from 1, code often from 0. Mixing
them shifts every $\bar\alpha$ by one step, and samples come out slightly too
noisy or too smooth. The $T + 1$ convention removes the ambiguity.

**Noise on the last step.** Adding $\sqrt{\tilde\beta_1} z$ at $t = 1$ is
harmless, since $\tilde\beta_1 = 0$. Adding $\sqrt{\beta_1} z$ instead leaves
audible hiss on every sample.

**A schedule mismatched to the data's scale.** Small-amplitude audio and
large-offset log-mels both see a different SNR than the schedule's designer
assumed. Normalize the data, or pick the schedule by the SNR it gives your
data.

**Too few steps with a schedule built for many.** The linear schedule from
$10^{-4}$ to 0.02 over 100 steps ends at $\bar\alpha_{100} \approx 0.36$,
nowhere near noise. Scale $\beta$ up when you shrink $T$, or use the cosine
schedule, which the **Run** does for the ring.

**Dividing by a tiny $\sqrt{\bar\alpha_t}$.** Near $t = T$, $\hat x_0$
divides by almost zero and explodes. Clip $\hat x_0$ or avoid forming it at
the noisiest steps.

## The exercise

You build `linear_betas`, `cosine_betas`, `alpha_bars`, `q_sample`,
`predict_x0`, `velocity`, `from_velocity`, `score_from_eps`, `posterior`,
`ddpm_loss`, and `ddpm_sample`. The checks compare the schedules with the
worked tables and $\bar\alpha_T = 4.04 \times 10^{-5}$, test $x_2 = 1.113103$
and per-example broadcasting, and check the conversions and the score against
a finite difference. They check the posterior against the worked numbers and
against Bayes' rule, recover the noise your loss used, and run your sampler
with the exact denoiser for Gaussian data. The last check trains a small
network with your loss and sampler on a ring of eight Gaussians: the samples
must average radius 2.0 and sit near the modes.

Press **Run** to hear the voice-like signal come back from noise, see its
spectrogram, and see 4,000 samples from your ring DDPM, about 97 percent of
them within 0.4 of a mode.
