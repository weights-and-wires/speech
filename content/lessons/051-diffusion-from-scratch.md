---
title: Diffusion from scratch
slug: 051-diffusion-from-scratch
part: "Part 7 — Diffusion and flow matching"
summary: Destroy data with Gaussian noise one step at a time, learn to predict the noise, and run the process backward to generate.
minutes: 40
gpu: false
objectives:
  - "State the DDPM time convention: data at t = 0, noise at t = T, and index schedules so that entry t means step t."
  - Build linear and cosine noise schedules and the closed-form jump from data to any step.
  - Convert between noise, data, velocity, and score predictions.
  - Derive and compute the posterior mean and variance of one reverse step.
  - Train a small DDPM with the noise-prediction loss and sample from it by ancestral sampling.
prereqs:
  - 045-normalizing-flows-and-vits
exercise: 051-diffusion-from-scratch
---

# Diffusion from scratch

Take a clip of speech and add a little noise. Add a little more. After a thousand small steps, nothing of the speech is left: the clip is white noise. That direction is easy, and you can write it down exactly. A *diffusion model* learns the other direction. It is trained to remove one small step of noise at a time, so that starting from fresh noise and repeating the removal a thousand times produces a clip that was never recorded.

Why would removing noise teach a network to generate? Because noising is Gaussian, you can jump from clean data to any noise level in one line. Training becomes a plain regression: show the network a noisy example and ask it for the noise that was added. This chapter builds the schedule, the forward jump, the four equivalent things a network can predict, the reverse step, the loss, and the sampler. You hear it on a voice-like signal with a denoiser you can compute exactly, and then train one on a toy distribution.

## What you already know

- A normalizing flow generates by an exact invertible map, and trains by maximum likelihood (see [Normalizing flows and VITS](/lessons/045-normalizing-flows-and-vits)).
- If $\epsilon \sim \mathcal{N}(0, 1)$, then $a + b\epsilon \sim \mathcal{N}(a, b^2)$.
- The product of two Gaussian densities in the same variable is another Gaussian, up to a constant.
- An SNR in decibels is $10 \log_{10}$ of a ratio of powers.
- Mean squared error regression predicts the conditional mean of the target.

## The time convention

Papers disagree about which end of time is data, so hold on to this course's choice.

> [!KEY]
> In this chapter and the next two, **$t = 0$ is the data and $t = T$ is noise**, as in DDPM, DiffWave, and Grad-TTS. Steps run $t = 1, \ldots, T$. Every schedule array has $T + 1$ entries, so `alpha_bar[t]` means step $t$, `betas[0] = 0`, and `alpha_bar[0] = 1`.

The flow matching chapters use the opposite convention, noise at $t = 0$. The next chapter shows how to move between the two.

## The forward process

Each forward step shrinks the signal slightly and adds a little noise (Ho et al., 2020):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I\right)
$$

Read it aloud: to get $x_t$, scale $x_{t-1}$ by $\sqrt{1 - \beta_t}$ and add Gaussian noise of variance $\beta_t$. The numbers $\beta_1, \ldots, \beta_T$ are the *noise schedule*. The shrink keeps the total variance from growing: if $x_{t-1}$ has unit variance, so does $x_t$.

Do you have to run all thousand steps to train on step 500? No. Every step is Gaussian, so many steps compose into one. Write $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t} \alpha_s$. Then:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

Read it aloud: the noisy sample at step $t$ is the clean data scaled by the square root of $\bar\alpha_t$, plus fresh standard noise scaled by the square root of $1 - \bar\alpha_t$. As $t$ grows, $\bar\alpha_t$ falls from 1 toward 0, and the data fades into the noise.

```python
ab = alpha_bar[t].reshape(-1, *[1] * (x0.dim() - 1))   # one step per example
x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
```

### Schedules

DDPM's *linear* schedule raises $\beta_t$ from $10^{-4}$ to $0.02$ over $T = 1000$ steps, which leaves $\bar\alpha_T = 4.04 \times 10^{-5}$. The *cosine* schedule of Nichol and Dhariwal sets $\bar\alpha_t$ directly, so it falls smoothly instead of collapsing early:

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \qquad s = 0.008
$$

Read it aloud: $\bar\alpha_t$ follows a squared cosine from 1 at $t = 0$ to almost 0 at $t = T$. Each $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ is clipped at 0.999, because the last step would otherwise be exactly 1.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="The fraction of signal power alpha-bar against step for the linear and cosine schedules, T = 1000">
<line x1="70" y1="230" x2="560" y2="230" class="stroke-muted"/><line x1="70" y1="230" x2="70" y2="30" class="stroke-muted"/>
<line x1="70" y1="230" x2="560" y2="230" class="stroke-grid"/><text x="62" y="234" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="130.0" x2="560" y2="130.0" class="stroke-grid"/><text x="62" y="134.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70" y1="30" x2="560" y2="30" class="stroke-grid"/><text x="62" y="34" class="fig-small" text-anchor="end">1</text>
<text x="70.0" y="248" class="fig-small" text-anchor="middle">0</text>
<text x="192.5" y="248" class="fig-small" text-anchor="middle">250</text>
<text x="315.0" y="248" class="fig-small" text-anchor="middle">500</text>
<text x="437.5" y="248" class="fig-small" text-anchor="middle">750</text>
<text x="560.0" y="248" class="fig-small" text-anchor="middle">1000</text>
<polyline points="70.0,30.0 74.9,30.4 79.8,31.2 84.7,32.3 89.6,33.9 94.5,35.8 99.4,38.1 104.3,40.7 109.2,43.7 114.1,47.0 119.0,50.6 123.9,54.5 128.8,58.6 133.7,63.0 138.6,67.6 143.5,72.3 148.4,77.3 153.3,82.4 158.2,87.5 163.1,92.8 168.0,98.2 172.9,103.6 177.8,109.0 182.7,114.4 187.6,119.8 192.5,125.2 197.4,130.5 202.3,135.7 207.2,140.8 212.1,145.8 217.0,150.7 221.9,155.5 226.8,160.1 231.7,164.5 236.6,168.8 241.5,173.0 246.4,176.9 251.3,180.7 256.2,184.3 261.1,187.7 266.0,191.0 270.9,194.0 275.8,196.9 280.7,199.7 285.6,202.2 290.5,204.6 295.4,206.8 300.3,208.9 305.2,210.8 310.1,212.6 315.0,214.3 319.9,215.8 324.8,217.2 329.7,218.5 334.6,219.7 339.5,220.8 344.4,221.7 349.3,222.6 354.2,223.4 359.1,224.2 364.0,224.8 368.9,225.4 373.8,226.0 378.7,226.4 383.6,226.9 388.5,227.2 393.4,227.6 398.3,227.9 403.2,228.2 408.1,228.4 413.0,228.6 417.9,228.8 422.8,229.0 427.7,229.1 432.6,229.2 437.5,229.3 442.4,229.4 447.3,229.5 452.2,229.6 457.1,229.6 462.0,229.7 466.9,229.7 471.8,229.8 476.7,229.8 481.6,229.8 486.5,229.9 491.4,229.9 496.3,229.9 501.2,229.9 506.1,229.9 511.0,229.9 515.9,230.0 520.8,230.0 525.7,230.0 530.6,230.0 535.5,230.0 540.4,230.0 545.3,230.0 550.2,230.0 555.1,230.0 560.0,230.0" class="stroke-a" fill="none" stroke-width="2.5"/>
<polyline points="70.0,30.0 74.9,30.1 79.8,30.3 84.7,30.7 89.6,31.1 94.5,31.6 99.4,32.2 104.3,32.9 109.2,33.7 114.1,34.6 119.0,35.6 123.9,36.7 128.8,37.8 133.7,39.1 138.6,40.4 143.5,41.9 148.4,43.4 153.3,45.0 158.2,46.7 163.1,48.4 168.0,50.3 172.9,52.2 177.8,54.2 182.7,56.2 187.6,58.4 192.5,60.6 197.4,62.9 202.3,65.2 207.2,67.6 212.1,70.1 217.0,72.6 221.9,75.2 226.8,77.8 231.7,80.5 236.6,83.3 241.5,86.0 246.4,88.8 251.3,91.7 256.2,94.6 261.1,97.5 266.0,100.5 270.9,103.5 275.8,106.5 280.7,109.6 285.6,112.6 290.5,115.7 295.4,118.8 300.3,121.9 305.2,125.0 310.1,128.1 315.0,131.2 319.9,134.3 324.8,137.5 329.7,140.6 334.6,143.7 339.5,146.7 344.4,149.8 349.3,152.8 354.2,155.9 359.1,158.9 364.0,161.8 368.9,164.8 373.8,167.7 378.7,170.6 383.6,173.4 388.5,176.2 393.4,178.9 398.3,181.6 403.2,184.2 408.1,186.8 413.0,189.4 417.9,191.9 422.8,194.3 427.7,196.6 432.6,198.9 437.5,201.1 442.4,203.3 447.3,205.4 452.2,207.4 457.1,209.3 462.0,211.2 466.9,213.0 471.8,214.7 476.7,216.3 481.6,217.8 486.5,219.3 491.4,220.6 496.3,221.9 501.2,223.1 506.1,224.2 511.0,225.2 515.9,226.1 520.8,226.9 525.7,227.6 530.6,228.3 535.5,228.8 540.4,229.2 545.3,229.6 550.2,229.8 555.1,230.0 560.0,230.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<line x1="330" y1="60" x2="360" y2="60" class="stroke-a" stroke-width="2.5"/><text x="366" y="64" class="fig-small">linear: 0.524 at t = 250, 0.079 at 500</text>
<line x1="330" y1="82" x2="360" y2="82" class="stroke-b" stroke-width="2.5"/><text x="366" y="86" class="fig-small">cosine: 0.847 at t = 250, 0.494 at 500</text>
<text x="315.0" y="270" class="fig-small" text-anchor="middle">step t</text>
<text x="70" y="20" class="fig-small">alpha-bar: share of the original signal left</text>
</svg>
Figure 1. The share of the original signal left at each step. Notice that the linear schedule has destroyed 92 percent of the signal power by step 500, while the cosine schedule has destroyed about half.
```

### Work four steps by hand

Take $T = 4$ and $\beta = (0.1, 0.2, 0.3, 0.4)$, with signal-to-noise ratio $\text{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$ for unit-variance data.

| $t$ | $\beta_t$ | $\alpha_t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\sqrt{1 - \bar\alpha_t}$ | SNR |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 1 | 1 | 0 | $\infty$ |
| 1 | 0.1 | 0.9 | 0.9 | 0.948683 | 0.316228 | 9.0 |
| 2 | 0.2 | 0.8 | 0.72 | 0.848528 | 0.529150 | 2.571 |
| 3 | 0.3 | 0.7 | 0.504 | 0.709930 | 0.704273 | 1.016 |
| 4 | 0.4 | 0.6 | 0.3024 | 0.549909 | 0.835225 | 0.433 |

With $x_0 = 1$ and $\epsilon = 0.5$ at $t = 2$: $x_2 = 0.848528 \times 1 + 0.529150 \times 0.5 = 1.113103$. The four-step cosine schedule gives $\bar\alpha = (0.847012, 0.493844, 0.144272)$ for the first three steps and needs the clip on the fourth, where $\beta_4 = 0.999$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The four-step example on a quarter circle: step t sits at x = sqrt(alpha-bar t), y = sqrt(1 - alpha-bar t), so signal weight and noise weight always square-sum to 1">
<line x1="90" y1="240" x2="300.0" y2="240" class="stroke-muted"/><line x1="90" y1="240" x2="90" y2="30.0" class="stroke-muted"/>
<polyline points="290.0,240.0 289.9,234.7 289.7,229.4 289.4,224.0 288.9,218.7 288.2,213.5 287.5,208.2 286.5,202.9 285.5,197.7 284.3,192.5 283.0,187.4 281.5,182.3 279.9,177.2 278.1,172.2 276.3,167.2 274.3,162.2 272.1,157.4 269.9,152.5 267.5,147.8 265.0,143.1 262.3,138.5 259.5,133.9 256.7,129.4 253.7,125.0 250.5,120.7 247.3,116.5 244.0,112.4 240.5,108.3 237.0,104.3 233.3,100.5 229.5,96.7 225.7,93.0 221.7,89.5 217.6,86.0 213.5,82.7 209.3,79.5 205.0,76.3 200.6,73.3 196.1,70.5 191.5,67.7 186.9,65.0 182.2,62.5 177.5,60.1 172.6,57.9 167.8,55.7 162.8,53.7 157.8,51.9 152.8,50.1 147.7,48.5 142.6,47.0 137.5,45.7 132.3,44.5 127.1,43.5 121.8,42.5 116.5,41.8 111.3,41.1 106.0,40.6 100.6,40.3 95.3,40.1 90.0,40.0" class="stroke-grid" fill="none" stroke-width="2"/>
<circle cx="290.0" cy="240.0" r="5" class="fill-b"/>
<text x="300.0" y="234.0" class="fig-small">t = 0: (1.000, 0.000)</text>
<circle cx="279.7" cy="176.8" r="5" class="fill-b"/>
<text x="289.7" y="180.8" class="fig-small">t = 1: (0.949, 0.316)</text>
<circle cx="259.7" cy="134.2" r="5" class="fill-b"/>
<text x="269.7" y="138.2" class="fig-small">t = 2: (0.849, 0.529)</text>
<circle cx="232.0" cy="99.1" r="5" class="fill-b"/>
<text x="242.0" y="103.1" class="fig-small">t = 3: (0.710, 0.704)</text>
<circle cx="200.0" cy="73.0" r="5" class="fill-b"/>
<text x="210.0" y="77.0" class="fig-small">t = 4: (0.550, 0.835)</text>
<text x="190.0" y="262" class="fig-small" text-anchor="middle">signal weight sqrt(alpha-bar t)</text>
<text x="90" y="24.0" class="fig-small">noise weight sqrt(1 - alpha-bar t)</text>
</svg>
Figure 2. The two weights of x_t from the table, plotted against each other. Notice that every step lies on the unit circle, because the squares of the two weights always add to 1, and that diffusion slides along the arc from data to noise.
```

> [!WARNING]
> A speech waveform in $[-1, 1]$ has a standard deviation near 0.1, not 1. Its SNR at step $t$ is $0.01\,\bar\alpha_t / (1 - \bar\alpha_t)$, 20 dB lower than the table says. A schedule that looks half-noisy for unit-variance data is almost pure noise for audio.

## Four things a network can predict

Given $x_t$ and $t$, a network can predict any of four quantities, and each determines the others. Write $a_t = \sqrt{\bar\alpha_t}$ and $\sigma_t = \sqrt{1 - \bar\alpha_t}$, so $x_t = a_t x_0 + \sigma_t \epsilon$ and $a_t^2 + \sigma_t^2 = 1$.

- **Noise** $\hat\epsilon$, which DDPM and DiffWave predict.
- **Data** $\hat x_0 = (x_t - \sigma_t \hat\epsilon) / a_t$.
- **Velocity** $v = a_t \epsilon - \sigma_t x_0$, which Stable Audio uses. It converts back with $\hat x_0 = a_t x_t - \sigma_t \hat v$ and $\hat\epsilon = \sigma_t x_t + a_t \hat v$.
- **Score** $\nabla_{x_t} \log p(x_t) = -\hat\epsilon / \sigma_t$, the direction in which the density rises fastest.

Read the score formula aloud: the noise that was added points away from the data, so the direction back toward likely data is minus the noise, divided by the noise scale. For one fixed $x_0$, the density $q(x_t \mid x_0)$ is a Gaussian with mean $a_t x_0$ and variance $\sigma_t^2$. Its log-density has gradient $-(x_t - a_t x_0) / \sigma_t^2 = -\epsilon / \sigma_t$, which is exactly this formula.

```python
x0_hat = (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()
v = ab.sqrt() * eps - (1 - ab).sqrt() * x0
score = -eps_hat / (1 - ab).sqrt()
```

Continue the worked example at $t = 2$. The velocity is $v = 0.848528 \times 0.5 - 0.529150 \times 1 = -0.104886$. Converting back, $0.848528 \times 1.113103 - 0.529150 \times (-0.104886) = 1.000000$, which is $x_0$. The score is $-0.5 / 0.529150 = -0.944911$.

## One step back

To reverse one step, you want $q(x_{t-1} \mid x_t)$. That distribution is intractable, because it depends on all the data. But if you also know $x_0$, Bayes' rule gives a Gaussian in closed form:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t, \tilde\beta_t I), \qquad
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t, \qquad
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t
$$

Read it aloud: the previous step's mean is a weighted blend of the clean data and the current noisy sample, and its variance is a fraction of $\beta_t$. The blend comes from multiplying two Gaussians in $x_{t-1}$: the prior $q(x_{t-1} \mid x_0)$ and the likelihood $q(x_t \mid x_{t-1})$. At $t = 1$, $\bar\alpha_0 = 1$, so the variance is 0 and the mean is $x_0$ itself.

At sampling time you do not know $x_0$, so you plug in the network's estimate. Substituting $\hat x_0$ from $\hat\epsilon$ gives the DDPM reverse step:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \hat\epsilon_\theta(x_t, t) \right) + \sqrt{\tilde\beta_t}\, z, \qquad z \sim \mathcal{N}(0, I)
$$

Read it aloud: remove a scaled copy of the predicted noise, undo the shrink by dividing by $\sqrt{\alpha_t}$, and add a little fresh noise, except at the last step, where $z = 0$.

For the worked example with the perfect prediction $\hat\epsilon = 0.5$: $\tilde\mu_2 = (1.113103 - 0.2 / 0.529150 \times 0.5) / \sqrt{0.8} = 1.033199$, and the formula in terms of $x_0$ gives the same number. The variance is $\tilde\beta_2 = (0.1 / 0.28) \times 0.2 = 0.071429$, a standard deviation of 0.267261.

## Train by predicting the noise

The DDPM training loss is a regression (Ho et al., 2020, equation 14):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0,\, \epsilon,\, t} \left\lVert \epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\ t\right) \right\rVert^2
$$

Read it aloud: pick a training example, a step $t$ uniformly from 1 to $T$, and fresh noise $\epsilon$. Noise the example to step $t$ in one jump, and penalize the squared difference between the true noise and the network's guess. Because squared error targets the conditional mean, the ideal network predicts $\mathbb{E}[\epsilon \mid x_t]$, which is exactly what the reverse step needs.

```python
t = torch.randint(1, T + 1, (B,), generator=g)
eps = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
loss = ((model(q_sample(x0, t, eps, alpha_bar), t) - eps) ** 2).mean()
```

> [!KEY]
> Training is plain regression: jump the data to a random step in one line and predict the noise. Only sampling pays for all $T$ steps. A thousand sequential network evaluations per sample make DDPM far slower than a GAN vocoder's single pass, a problem the next chapter solves.

## Hear it with an exact denoiser

When the data itself is Gaussian, the ideal noise predictor is known in closed form, so you can test a sampler with no training at all. The **Run** uses a voice-like Gaussian signal: random noise whose power spectrum has harmonics of 140 Hz under three formant peaks, with a standard deviation of 0.1. For such a signal, $\mathbb{E}[\epsilon \mid x_t]$ is a linear filter. In each frequency bin $k$ with data power $P_k$, it multiplies $x_t$ by $\sigma_t / (\bar\alpha_t P_k + 1 - \bar\alpha_t)$. It is the Wiener filter from signal processing, rearranged to predict the noise instead of the signal.

With a linear schedule from $10^{-4}$ to 0.1 over 200 steps, the signal's SNR is 20 dB at step 1, $-2$ dB at step 8, $-12$ dB at step 25, and $-65$ dB at step 200. You hear $x_{200}$, $x_{25}$, $x_8$, and $x_0$ on the way back. The voice emerges only in the last few steps, because a small-amplitude signal spends most of the schedule buried.

The following demo noises a mixture of three Gaussians and walks it back with the exact score, so it needs no network either.

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

## Where this breaks

**Off-by-one steps.** Papers index steps from 1, code often from 0. Mixing them shifts every $\bar\alpha$ by one step. The $T + 1$ convention removes the ambiguity.

**Too few steps with a schedule built for many.** The linear schedule from $10^{-4}$ to 0.02 over 100 steps ends at $\bar\alpha_{100} \approx 0.36$, nowhere near noise. Scale $\beta$ up when you shrink $T$, or use the cosine schedule, which the **Run** does for the ring.

**Dividing by a tiny $\sqrt{\bar\alpha_t}$.** Near $t = T$, $\hat x_0$ divides by almost zero and explodes. Clip $\hat x_0$ or avoid forming it at the noisiest steps.

## The exercise

You build `linear_betas`, `cosine_betas`, `alpha_bars`, `q_sample`, `predict_x0`, `velocity`, `from_velocity`, `score_from_eps`, `posterior`, `ddpm_loss`, and `ddpm_sample`. The checks compare the schedules with the worked tables and $\bar\alpha_T = 4.04 \times 10^{-5}$, test $x_2 = 1.113103$ and per-example broadcasting, and check the conversions and the score against a finite difference. They check the posterior against the worked numbers and against Bayes' rule, recover the noise your loss used, and run your sampler with the exact denoiser for Gaussian data. The last check trains a small network with your loss and sampler on a ring of eight Gaussians: the samples must average radius 2.0 and sit near the modes.

Press **Run** to hear the voice-like signal come back from noise, see its spectrogram, and see 4,000 samples from your ring DDPM, about 97 percent of them within 0.4 of a mode.
