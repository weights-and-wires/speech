---
title: Few-step generation and audio metrics
slug: 060-few-step-generation-and-audio-metrics
part: "Part 7 — Diffusion and flow matching"
summary: Teach a flow to jump from noise to data in one step with shortcut models and MeanFlow, then score generators with FAD and CLAP.
minutes: 60
gpu: false
objectives:
  - Explain why one step of an ordinary flow returns the data mean, and what a one-step model must learn instead.
  - Build the shortcut-model loss, whose target is two half steps of the model itself.
  - Derive and compute the MeanFlow target for the average velocity with one Jacobian-vector product.
  - Compute the Fréchet audio distance from embeddings, and correct its sample-size bias with FAD-infinity.
  - Compute a CLAP score, and say what FAD and CLAP can't tell you.
prereqs:
  - 057-flow-matching
  - 058-straighter-paths-and-better-solvers
  - 059-flow-matching-for-speech
  - 054-fast-samplers
exercise: 060-few-step-generation-and-audio-metrics
---

# Few-step generation and audio metrics

A flow-matching TTS model such as F5-TTS spends 16 to 32 network evaluations on each utterance, and guidance doubles that. A voice agent that must answer in a fraction of a second wants one to eight. But the previous chapter showed that a single step of an ordinary flow returns a blur: the average of all the data. So what must a network learn to make one step land on a real sample?

The answer has a picture: a flow's velocity is the arrow at one instant, and a jump needs the arrow that covers the whole trip. This chapter builds two ways to learn it without a separate teacher, shortcut models and MeanFlow. Once a generator is fast, you need a number that says whether it still sounds right. The second half builds the Fréchet audio distance (FAD), which compares generated and real audio, and the CLAP score, which checks that sound matches its prompt. Both are used everywhere, and both have biases you must know before you trust a table. The time convention is unchanged: $t = 0$ is noise and $t = 1$ is data.

## What you already know

- A flow's velocity field $v(x, t)$ moves noise at $t = 0$ to data at $t = 1$, and Euler integrates it one step at a time.
- One Euler step from $t = 0$ lands on the data mean, because the field there is the average of every straight path through the noise sample.
- [Reflow](/lessons/058-straighter-paths-and-better-solvers) straightens a flow by retraining it on its own noise-and-sample pairs.
- A multivariate Gaussian is described by a mean vector and a covariance matrix.
- Cosine similarity measures the angle between two vectors, whatever their lengths.

## What a one-step model must learn

The velocity $v(x, t)$ is the direction the sample moves at that instant, the tangent of its path. To jump from time $s$ to a later time $e$ in one step, you need the *average* velocity over the whole interval, the chord of the trajectory rather than its tangent:

$$
u(x_s, s, e) = \frac{1}{e - s} \int_s^e v(x_\tau, \tau)\, d\tau,
\qquad x_e = x_s + (e - s)\, u(x_s, s, e)
$$

Read it aloud: the average velocity is the total displacement from $s$ to $e$ divided by the time it took, so one step with it lands exactly where the trajectory lands. When $e = s$, it is the ordinary velocity.

Take the Gaussian flow of [the flow-matching chapter](/lessons/057-flow-matching): data $\mathcal{N}(2, 0.5^2)$ and a noise sample $x_0 = 1.3$, whose trajectory ends at 2.65 (Figure 1). The velocity at the start is $v = 0.7$, so one Euler step stops at 2.0, the mean. The average velocity over $[0, 1]$ is $2.65 - 1.3 = 1.35$. The trajectory passes through 1.726722 at $t = 0.5$, so the two halves have average velocities $(1.726722 - 1.3) / 0.5 = 0.853444$ and $(2.65 - 1.726722) / 0.5 = 1.846556$. Their mean is exactly 1.35. Averages over consecutive halves compose into the average over the whole, and both methods below are built on that fact.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The curved trajectory of the Gaussian flow from 1.3 at t = 0 to 2.65 at t = 1, with one Euler step along the starting tangent ending at 2.0 and one step along the chord ending at 2.65">
<defs><marker id="c053-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="240.0" x2="540.0" y2="240.0" class="stroke-ink"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-ink"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-grid"/>
<text x="70.0" y="256.0" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="30.0" x2="187.5" y2="240.0" class="stroke-grid"/>
<text x="187.5" y="256.0" class="fig-small" text-anchor="middle">0.25</text>
<line x1="305.0" y1="30.0" x2="305.0" y2="240.0" class="stroke-grid"/>
<text x="305.0" y="256.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="422.5" y1="30.0" x2="422.5" y2="240.0" class="stroke-grid"/>
<text x="422.5" y="256.0" class="fig-small" text-anchor="middle">0.75</text>
<line x1="540.0" y1="30.0" x2="540.0" y2="240.0" class="stroke-grid"/>
<text x="540.0" y="256.0" class="fig-small" text-anchor="middle">1</text>
<line x1="70.0" y1="240.0" x2="540.0" y2="240.0" class="stroke-grid"/>
<text x="64.0" y="244.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="187.5" x2="540.0" y2="187.5" class="stroke-grid"/>
<text x="64.0" y="191.5" class="fig-small" text-anchor="end">1.5</text>
<line x1="70.0" y1="135.0" x2="540.0" y2="135.0" class="stroke-grid"/>
<text x="64.0" y="139.0" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="82.5" x2="540.0" y2="82.5" class="stroke-grid"/>
<text x="64.0" y="86.5" class="fig-small" text-anchor="end">2.5</text>
<line x1="70.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="64.0" y="34.0" class="fig-small" text-anchor="end">3</text>
<path d="M70.0 208.5 L75.9 207.6 L81.9 206.6 L87.8 205.7 L93.8 204.7 L99.7 203.8 L105.7 202.8 L111.6 201.8 L117.6 200.9 L123.5 199.9 L129.5 198.9 L135.4 197.9 L141.4 196.9 L147.3 195.9 L153.3 194.8 L159.2 193.8 L165.2 192.7 L171.1 191.7 L177.1 190.6 L183.0 189.5 L189.0 188.4 L194.9 187.3 L200.9 186.2 L206.8 185.1 L212.8 183.9 L218.7 182.8 L224.7 181.6 L230.6 180.4 L236.6 179.2 L242.5 178.0 L248.5 176.7 L254.4 175.4 L260.4 174.2 L266.3 172.8 L272.3 171.5 L278.2 170.1 L284.2 168.8 L290.1 167.3 L296.1 165.9 L302.0 164.4 L308.0 162.9 L313.9 161.4 L319.9 159.8 L325.8 158.2 L331.8 156.6 L337.7 154.9 L343.7 153.2 L349.6 151.5 L355.6 149.7 L361.5 147.8 L367.5 145.9 L373.4 144.0 L379.4 142.0 L385.3 140.0 L391.3 137.9 L397.2 135.7 L403.2 133.5 L409.1 131.3 L415.1 129.0 L421.0 126.6 L427.0 124.1 L432.9 121.7 L438.9 119.1 L444.8 116.5 L450.8 113.8 L456.7 111.1 L462.7 108.3 L468.6 105.4 L474.6 102.5 L480.5 99.5 L486.5 96.5 L492.4 93.4 L498.4 90.2 L504.3 87.0 L510.3 83.8 L516.2 80.5 L522.2 77.1 L528.1 73.7 L534.1 70.3 L540.0 66.8" class="stroke-a" fill="none"/>
<line x1="70.0" y1="208.5" x2="540.0" y2="135.0" class="stroke-bad dashed"/>
<circle cx="540.0" cy="135.0" r="5" class="fill-bad"/>
<text x="532.0" y="175.0" class="fig-small" text-anchor="end">Euler: 2.0 (the mean)</text>
<line x1="70.0" y1="208.5" x2="540.0" y2="66.8" class="stroke-b"/>
<circle cx="540.0" cy="66.8" r="5" class="fill-b"/>
<text x="532.0" y="56.8" class="fig-small" text-anchor="end">average velocity 1.35: 2.65</text>
<circle cx="70.0" cy="208.5" r="5" class="fill-ink"/>
<text x="78.0" y="226.5" class="fig-small" text-anchor="start">x0 = 1.3, v = 0.7</text>
<circle cx="305.0" cy="163.7" r="4" class="fill-a"/>
<text x="313.0" y="181.7" class="fig-small" text-anchor="start">1.7267 at t = 0.5</text>
<text x="305.0" y="282.0" class="fig-small" text-anchor="middle">time t (0 is noise, 1 is data)</text>
</svg>
Figure 1. The curve is the true path. The velocity at the start points along the dashed tangent, which misses the end by 0.65. The average velocity over the interval points along the solid chord, which lands exactly on 2.65.
```

> [!KEY]
> A one-step generator must learn the average velocity over an interval, not the velocity at a point. The average over a long interval is the mean of the averages over its halves.

The oldest route to few steps is a teacher: progressive distillation trains a student to match two teacher steps in one, and consistency models learn a function that maps every point of a trajectory to its end. CoMoSpeech and WaveFM use consistency distillation, and DMOSpeech uses distribution matching to bring an F5-style model to 4 steps. Shortcut models and MeanFlow need no teacher. They train from scratch and learn the jump alongside the flow.

## Shortcut models

A *shortcut model* (Frans et al., 2024) adds the step size $d$ as a third input, $s_\theta(x_t, t, d)$. At $d = 0$ it is an ordinary flow, trained on $x_1 - x_0$. For $d > 0$ it learns from itself: one step of size $2d$ must equal two steps of size $d$:

$$
s_\theta(x_t, t, 2d) \leftarrow \tfrac{1}{2}\, s_\theta(x_t, t, d) + \tfrac{1}{2}\, s_\theta(x'_{t+d}, t + d, d),
\qquad x'_{t+d} = x_t + d\, s_\theta(x_t, t, d)
$$

Read it aloud: take a step of size $d$, then another from where it lands, and teach the big step to point along the average of the two. The arrow means the right side is a fixed target, computed without gradient. Small steps are accurate because they are nearly flow matching, and each doubling passes that accuracy up, so the largest step, $d = 1$, becomes a one-step sampler. In torch:

```python
with torch.no_grad():
    first = model(x_t, t, d)
    second = model(x_t + d[:, None] * first, t + d, d)
target = 0.5 * (first + second)
```

The **Run** block trains a quarter of each batch on self-consistency, with $d$ from $1/32$ to $1/2$, and starts each of those rows at a multiple of $2d$ so that $t + 2d \le 1$. The paper uses 128 base steps, so $d$ takes 8 values from $1/128$ to 1.

## MeanFlow

*MeanFlow* (Geng et al., 2025) learns the average velocity $u_\theta(x, s, e)$ directly. There's no integral in the training data, so it uses an identity instead. Multiply the definition by $e - s$ and differentiate with respect to the start time $s$, holding $e$ fixed and letting $x_s$ move along the flow:

$$
u(x_s, s, e) = v(x_s, s) + (e - s)\,\frac{d}{ds} u(x_s, s, e),
\qquad \frac{d}{ds} u = v \cdot \nabla_x u + \partial_s u
$$

Read it aloud: the average velocity equals the velocity at the start, plus a correction for how the average changes as the start slides forward along the trajectory. The total derivative has two parts: $u$ changes because $x$ moves with velocity $v$, and because $s$ itself moves. In the Gaussian example, $u = 1.35$ and $v = 0.7$, so $du/ds$ must be 0.65, which a finite difference along the trajectory confirms.

Training plugs the conditional velocity $x_1 - x_0$ in for $v$, as flow matching does, and regresses $u_\theta$ onto the right side with the gradient stopped. One forward-mode *Jacobian-vector product* (JVP) returns $u_\theta$ and its derivative along the tangent $(v, 1, 0)$ for $(x, s, e)$:

```python
_, du_ds = torch.func.jvp(u_fn, (x, s, e), (v, torch.ones_like(s), torch.zeros_like(e)))
target = (v + (e - s)[:, None] * du_ds).detach()
```

To sample in one step, compute $x_1 = x_0 + u_\theta(x_0, 0, 1)$.

> [!WARNING]
> The MeanFlow paper puts data at $t = 0$ and writes $u = v - (t - r)\,\tfrac{d}{dt}u$. Its $t$ is $1 - s$ here, its $r$ is $1 - e$, and its velocities point the other way, which turns its minus sign into this chapter's plus. Copy the paper's formula into a noise-at-0 code base and the model learns to run backward.

On the eight-blob ring of the **Run** block (Figure 2), one step from a plain flow gives an FAD of 96.7 on note embeddings, against 0.38 for the shortcut model and 0.39 for MeanFlow. The plain flow needs 8 steps to reach 0.16.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Bars on a log scale of FAD for the note embeddings in the Run block: a plain flow in one step scores 96.7, a shortcut model in one step 0.38, MeanFlow in one step 0.39, and a plain flow in eight steps 0.16">
<defs><marker id="c053c-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="200.0" y1="30.0" x2="200.0" y2="170.0" class="stroke-grid"/>
<text x="200.0" y="186.0" class="fig-small" text-anchor="middle">0.1</text>
<line x1="309.1" y1="30.0" x2="309.1" y2="170.0" class="stroke-grid"/>
<text x="309.1" y="186.0" class="fig-small" text-anchor="middle">1</text>
<line x1="418.1" y1="30.0" x2="418.1" y2="170.0" class="stroke-grid"/>
<text x="418.1" y="186.0" class="fig-small" text-anchor="middle">10</text>
<line x1="527.2" y1="30.0" x2="527.2" y2="170.0" class="stroke-grid"/>
<text x="527.2" y="186.0" class="fig-small" text-anchor="middle">100</text>
<rect x="200.0" y="42.0" width="325.6" height="22.0" class="fill-bad"/>
<text x="190.0" y="58.0" class="fig-small" text-anchor="end">Plain flow, 1 step</text>
<text x="531.6" y="58.0" class="fig-small" text-anchor="start">96.7</text>
<rect x="200.0" y="74.0" width="63.2" height="22.0" class="fill-b"/>
<text x="190.0" y="90.0" class="fig-small" text-anchor="end">Shortcut, 1 step</text>
<text x="269.2" y="90.0" class="fig-small" text-anchor="start">0.38</text>
<rect x="200.0" y="106.0" width="64.5" height="22.0" class="fill-b"/>
<text x="190.0" y="122.0" class="fig-small" text-anchor="end">MeanFlow, 1 step</text>
<text x="270.5" y="122.0" class="fig-small" text-anchor="start">0.39</text>
<rect x="200.0" y="138.0" width="22.3" height="22.0" class="fill-a"/>
<text x="190.0" y="154.0" class="fig-small" text-anchor="end">Plain flow, 8 steps</text>
<text x="228.3" y="154.0" class="fig-small" text-anchor="start">0.16</text>
<line x1="200.0" y1="170.0" x2="560.0" y2="170.0" class="stroke-ink"/>
<text x="380.0" y="212.0" class="fig-small" text-anchor="middle">FAD on note embeddings (log scale, lower is better)</text>
</svg>
Figure 2. One step of a plain flow scores about 250 times worse than one step of either teacher-free method. Notice that the shortcut and MeanFlow bars are about as short as the eight-step flow's.
```

## Measure a generator with FAD

How do you tell that 0.38 is good? The *Fréchet audio distance* (Kilgour et al., 2019) runs real and generated clips through a fixed audio embedding model, fits a Gaussian to each cloud of embeddings, and measures how far apart the two Gaussians are:

$$
\text{FAD} = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Read it aloud: the squared distance between the two means, plus a term that is zero when the covariances match and grows as their shapes differ. In one dimension it is $(\mu_r - \mu_g)^2 + (\sigma_r - \sigma_g)^2$, so $\mathcal{N}(0, 1)$ against $\mathcal{N}(1, 2^2)$ gives $1 + 1 = 2$ (Figure 3). With diagonal covariances it is $\lVert \Delta\mu \rVert^2 + \sum_i (\sigma_{r,i} - \sigma_{g,i})^2$. In numpy:

```python
root = scipy.linalg.sqrtm(cov_r @ cov_g).real
fad = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2 * root)
```

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="The probability densities of N(0, 1) and N(1, 2 squared), whose means differ by 1 and whose standard deviations differ by 1, giving a Frechet distance of 2">
<defs><marker id="c053b-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="50.0" y1="200.0" x2="560.0" y2="200.0" class="stroke-ink"/>
<line x1="50.0" y1="30.0" x2="50.0" y2="200.0" class="stroke-ink"/>
<line x1="50.0" y1="30.0" x2="50.0" y2="200.0" class="stroke-grid"/>
<text x="50.0" y="216.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="128.5" y1="30.0" x2="128.5" y2="200.0" class="stroke-grid"/>
<text x="128.5" y="216.0" class="fig-small" text-anchor="middle">-4</text>
<line x1="206.9" y1="30.0" x2="206.9" y2="200.0" class="stroke-grid"/>
<text x="206.9" y="216.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="285.4" y1="30.0" x2="285.4" y2="200.0" class="stroke-grid"/>
<text x="285.4" y="216.0" class="fig-small" text-anchor="middle">0</text>
<line x1="363.8" y1="30.0" x2="363.8" y2="200.0" class="stroke-grid"/>
<text x="363.8" y="216.0" class="fig-small" text-anchor="middle">2</text>
<line x1="442.3" y1="30.0" x2="442.3" y2="200.0" class="stroke-grid"/>
<text x="442.3" y="216.0" class="fig-small" text-anchor="middle">4</text>
<line x1="520.8" y1="30.0" x2="520.8" y2="200.0" class="stroke-grid"/>
<text x="520.8" y="216.0" class="fig-small" text-anchor="middle">6</text>
<path d="M50.0 200.0 L52.6 200.0 L55.1 200.0 L57.7 200.0 L60.3 200.0 L62.8 200.0 L65.4 200.0 L67.9 200.0 L70.5 200.0 L73.1 200.0 L75.6 200.0 L78.2 200.0 L80.8 200.0 L83.3 200.0 L85.9 200.0 L88.4 200.0 L91.0 200.0 L93.6 200.0 L96.1 200.0 L98.7 200.0 L101.3 200.0 L103.8 200.0 L106.4 200.0 L108.9 200.0 L111.5 200.0 L114.1 200.0 L116.6 200.0 L119.2 200.0 L121.8 200.0 L124.3 200.0 L126.9 200.0 L129.4 199.9 L132.0 199.9 L134.6 199.9 L137.1 199.9 L139.7 199.8 L142.3 199.8 L144.8 199.8 L147.4 199.7 L149.9 199.6 L152.5 199.5 L155.1 199.4 L157.6 199.2 L160.2 199.1 L162.8 198.9 L165.3 198.6 L167.9 198.3 L170.5 197.9 L173.0 197.5 L175.6 197.0 L178.1 196.4 L180.7 195.7 L183.3 194.9 L185.8 194.0 L188.4 192.9 L191.0 191.7 L193.5 190.3 L196.1 188.7 L198.6 186.9 L201.2 184.9 L203.8 182.7 L206.3 180.2 L208.9 177.5 L211.5 174.5 L214.0 171.2 L216.6 167.6 L219.1 163.8 L221.7 159.6 L224.3 155.2 L226.8 150.5 L229.4 145.6 L232.0 140.4 L234.5 135.0 L237.1 129.4 L239.6 123.6 L242.2 117.7 L244.8 111.8 L247.3 105.8 L249.9 99.9 L252.5 94.0 L255.0 88.3 L257.6 82.7 L260.2 77.5 L262.7 72.5 L265.3 67.8 L267.8 63.6 L270.4 59.9 L273.0 56.7 L275.5 54.0 L278.1 51.9 L280.7 50.4 L283.2 49.5 L285.8 49.3 L288.3 49.7 L290.9 50.8 L293.5 52.5 L296.0 54.7 L298.6 57.6 L301.2 61.0 L303.7 64.9 L306.3 69.2 L308.8 74.0 L311.4 79.1 L314.0 84.4 L316.5 90.0 L319.1 95.8 L321.7 101.7 L324.2 107.7 L326.8 113.6 L329.3 119.6 L331.9 125.4 L334.5 131.1 L337.0 136.7 L339.6 142.0 L342.2 147.1 L344.7 152.0 L347.3 156.6 L349.8 160.9 L352.4 165.0 L355.0 168.7 L357.5 172.2 L360.1 175.4 L362.7 178.3 L365.2 181.0 L367.8 183.4 L370.4 185.6 L372.9 187.5 L375.5 189.2 L378.0 190.7 L380.6 192.1 L383.2 193.3 L385.7 194.3 L388.3 195.2 L390.9 195.9 L393.4 196.6 L396.0 197.2 L398.5 197.6 L401.1 198.1 L403.7 198.4 L406.2 198.7 L408.8 198.9 L411.4 199.1 L413.9 199.3 L416.5 199.4 L419.0 199.5 L421.6 199.6 L424.2 199.7 L426.7 199.8 L429.3 199.8 L431.9 199.9 L434.4 199.9 L437.0 199.9 L439.5 199.9 L442.1 199.9 L444.7 200.0 L447.2 200.0 L449.8 200.0 L452.4 200.0 L454.9 200.0 L457.5 200.0 L460.1 200.0 L462.6 200.0 L465.2 200.0 L467.7 200.0 L470.3 200.0 L472.9 200.0 L475.4 200.0 L478.0 200.0 L480.6 200.0 L483.1 200.0 L485.7 200.0 L488.2 200.0 L490.8 200.0 L493.4 200.0 L495.9 200.0 L498.5 200.0 L501.1 200.0 L503.6 200.0 L506.2 200.0 L508.7 200.0 L511.3 200.0 L513.9 200.0 L516.4 200.0 L519.0 200.0 L521.6 200.0 L524.1 200.0 L526.7 200.0 L529.2 200.0 L531.8 200.0 L534.4 200.0 L536.9 200.0 L539.5 200.0 L542.1 200.0 L544.6 200.0 L547.2 200.0 L549.7 200.0 L552.3 200.0 L554.9 200.0 L557.4 200.0 L560.0 200.0" class="stroke-a" fill="none"/>
<path d="M50.0 199.8 L52.6 199.8 L55.1 199.8 L57.7 199.8 L60.3 199.7 L62.8 199.7 L65.4 199.7 L67.9 199.6 L70.5 199.6 L73.1 199.6 L75.6 199.5 L78.2 199.5 L80.8 199.4 L83.3 199.3 L85.9 199.3 L88.4 199.2 L91.0 199.1 L93.6 199.0 L96.1 198.9 L98.7 198.8 L101.3 198.7 L103.8 198.6 L106.4 198.4 L108.9 198.3 L111.5 198.1 L114.1 197.9 L116.6 197.8 L119.2 197.6 L121.8 197.3 L124.3 197.1 L126.9 196.9 L129.4 196.6 L132.0 196.3 L134.6 196.0 L137.1 195.7 L139.7 195.3 L142.3 194.9 L144.8 194.5 L147.4 194.1 L149.9 193.7 L152.5 193.2 L155.1 192.7 L157.6 192.2 L160.2 191.6 L162.8 191.0 L165.3 190.4 L167.9 189.8 L170.5 189.1 L173.0 188.3 L175.6 187.6 L178.1 186.8 L180.7 186.0 L183.3 185.1 L185.8 184.2 L188.4 183.3 L191.0 182.3 L193.5 181.3 L196.1 180.3 L198.6 179.2 L201.2 178.1 L203.8 177.0 L206.3 175.8 L208.9 174.6 L211.5 173.4 L214.0 172.1 L216.6 170.8 L219.1 169.5 L221.7 168.1 L224.3 166.7 L226.8 165.3 L229.4 163.9 L232.0 162.5 L234.5 161.0 L237.1 159.6 L239.6 158.1 L242.2 156.6 L244.8 155.1 L247.3 153.6 L249.9 152.1 L252.5 150.6 L255.0 149.1 L257.6 147.7 L260.2 146.2 L262.7 144.8 L265.3 143.4 L267.8 142.0 L270.4 140.6 L273.0 139.3 L275.5 138.0 L278.1 136.8 L280.7 135.6 L283.2 134.4 L285.8 133.3 L288.3 132.3 L290.9 131.3 L293.5 130.4 L296.0 129.5 L298.6 128.7 L301.2 127.9 L303.7 127.3 L306.3 126.7 L308.8 126.2 L311.4 125.7 L314.0 125.3 L316.5 125.0 L319.1 124.8 L321.7 124.7 L324.2 124.6 L326.8 124.7 L329.3 124.8 L331.9 125.0 L334.5 125.2 L337.0 125.6 L339.6 126.0 L342.2 126.5 L344.7 127.1 L347.3 127.7 L349.8 128.4 L352.4 129.2 L355.0 130.1 L357.5 131.0 L360.1 132.0 L362.7 133.0 L365.2 134.1 L367.8 135.2 L370.4 136.4 L372.9 137.7 L375.5 138.9 L378.0 140.2 L380.6 141.6 L383.2 143.0 L385.7 144.4 L388.3 145.8 L390.9 147.2 L393.4 148.7 L396.0 150.2 L398.5 151.7 L401.1 153.1 L403.7 154.6 L406.2 156.1 L408.8 157.6 L411.4 159.1 L413.9 160.6 L416.5 162.0 L419.0 163.5 L421.6 164.9 L424.2 166.3 L426.7 167.7 L429.3 169.1 L431.9 170.4 L434.4 171.7 L437.0 173.0 L439.5 174.2 L442.1 175.4 L444.7 176.6 L447.2 177.8 L449.8 178.9 L452.4 180.0 L454.9 181.0 L457.5 182.0 L460.1 183.0 L462.6 184.0 L465.2 184.9 L467.7 185.7 L470.3 186.6 L472.9 187.4 L475.4 188.1 L478.0 188.8 L480.6 189.5 L483.1 190.2 L485.7 190.8 L488.2 191.4 L490.8 192.0 L493.4 192.5 L495.9 193.1 L498.5 193.5 L501.1 194.0 L503.6 194.4 L506.2 194.8 L508.7 195.2 L511.3 195.6 L513.9 195.9 L516.4 196.2 L519.0 196.5 L521.6 196.8 L524.1 197.0 L526.7 197.3 L529.2 197.5 L531.8 197.7 L534.4 197.9 L536.9 198.1 L539.5 198.2 L542.1 198.4 L544.6 198.5 L547.2 198.7 L549.7 198.8 L552.3 198.9 L554.9 199.0 L557.4 199.1 L560.0 199.2" class="stroke-b" fill="none"/>
<text x="222.6" y="67.8" class="fig-small" text-anchor="end">real N(0, 1)</text>
<text x="410.9" y="143.3" class="fig-small" text-anchor="start">generated N(1, 2^2)</text>
<line x1="285.4" y1="200.0" x2="285.4" y2="41.3" class="stroke-a dashed"/>
<line x1="324.6" y1="200.0" x2="324.6" y2="105.6" class="stroke-b dashed"/>
<text x="300.0" y="236.0" class="fig-small" text-anchor="middle">embedding value</text>
<text x="330.0" y="40.0" class="fig-bold" text-anchor="start">FAD = (0 - 1)^2 + (1 - 2)^2 = 1 + 1 = 2</text>
</svg>
Figure 3. In one dimension FAD adds the squared gap between the means and the squared gap between the standard deviations. Notice that a generator can match the mean and still score badly if its spread is wrong.
```

The original embedding is VGGish: 128 numbers per 0.96 s of 64-band log-mel at 16 kHz. Stable Audio reports it with OpenL3, and `fadtk` (Gui et al., 2024) supports CLAP, EnCodec, MERT, and others.

### What biases FAD

FAD looks like one objective number. It isn't, for the following reasons:

- **Sample size.** An estimated covariance is noisy, and the noise always adds distance, so FAD falls as the sets grow. In the **Run** block, two independent sets of 2,000 real notes score 0.030, and one generator scores 0.461 with 100 samples and 0.139 with 1,600. Compare only at equal sizes. *FAD-infinity*, from the same paper, computes FAD at several sizes $N$, fits a line against $1/N$, and reports its value at $1/N = 0$, which is 0.102 for that generator.
- **The embedding.** Change the embedding model and every number changes. On raw 2-D points instead of note embeddings, the one-step flow scores 3.13 rather than 96.7. A FAD without its embedding's name means nothing.
- **The Gaussian.** FAD sees only a mean and a covariance. Two sets with the same mean and covariance score 0 even when one clusters in a few tight modes and the other spreads smoothly.

> [!KEY]
> A FAD score means something only next to its embedding model, its reference set, and its sample size. Compare two generators only when all three match.

> [!TIP]
> `scipy.linalg.sqrtm` can return complex values from rounding. Take the real part, and check that the imaginary part you dropped is tiny.

## Check the prompt with a CLAP score

FAD says whether generated audio looks like real audio, not whether it matches its prompt. The *CLAP score* embeds each text prompt and its generated clip with a contrastive language-audio model and averages their cosine similarity:

$$
\text{CLAP} = \frac{1}{n}\sum_{i=1}^{n} \frac{a_i \cdot b_i}{\lVert a_i \rVert\, \lVert b_i \rVert}
$$

Read it aloud: for each prompt, the cosine of the angle between the text embedding $a_i$ and the audio embedding $b_i$, averaged over prompts. For example, text $(1, 0)$ with audio $(0.6, 0.8)$ scores 0.6, and text $(0, 1)$ with audio $(0, 2)$ scores 1, so the CLAP score is 0.8. Some papers clip at 0 or multiply by 100, and the score depends on the CLAP checkpoint, so report both. A CLAP score checks the words of a prompt, not the structure of a song or the correctness of a sentence. For speech, word error rate from an ASR model and speaker similarity fit better.

> [!RECAP]
> - A one-step model learns the average velocity $u(x, s, e)$, and averages over halves compose into the whole.
> - Shortcut models teach the step $2d$ to match two stop-gradient steps of size $d$.
> - MeanFlow regresses onto $v + (e - s)\,du/ds$, computed with one JVP with tangent $(v, 1, 0)$, in the noise-at-0 convention.
> - FAD is the Fréchet distance between Gaussians fitted to embeddings. It is biased upward at small $N$, and it depends on the embedding.
> - The CLAP score is the mean cosine similarity of matching text and audio embeddings.

## Build it step by step

Build the functions in the order the checks test them.

1. **`shortcut_target(model, x_t, t, d)`** returns the two-step average, shape `(B, D)`, under `torch.no_grad()`. Mistake: starting the second step at time `t` instead of `t + d`.
2. **`shortcut_loss(model, x0, x1, t, d)`** builds `x_t`, fills the target with `x1 - x0` where `d == 0` and with `shortcut_target` elsewhere, and asks the model for step `torch.where(d == 0, d, 2 * d)`. Return the mean squared error.
3. **`meanflow_target(u_fn, x, s, e, v)`** calls `torch.func.jvp` once and returns the detached target. At `e == s` it returns `v`.
4. **`frechet_distance(mu1, cov1, mu2, cov2)`** uses `scipy.linalg.sqrtm` and `np.trace`. Accept 1-D inputs with `np.atleast_1d` and `np.atleast_2d`.
5. **`fad(emb_a, emb_b)`** fits `mean(axis=0)` and `np.cov(emb, rowvar=False)`. Forgetting `rowvar=False` gives an `(n, n)` covariance of the samples, not of the features.
6. **`fad_infinity(emb_ref, emb_gen, sizes, seed)`** draws subsets with `rng.choice(..., replace=False)`, scores each, and returns the intercept of `np.polyfit(1 / N, scores, 1)`.
7. **`clap_score(text_emb, audio_emb)`** normalizes rows and averages the row-wise dot products.

## Where this breaks

- **The convention flips.** MeanFlow and Stable Audio put data at 0. Port their equations without converting and the model generates noise.
- **Few steps lose detail first.** One-step samples tend to lose fine texture and variety before they lose their overall shape. Listen, and compare FAD at equal sample sizes.
- **Self-consistency drifts.** A shortcut model's large steps learn from its own small steps, so an error at small $d$ spreads upward. Keep enough of each batch on plain flow matching.
- **The metric is gamed.** A generator tuned against one embedding can lower FAD without sounding better. Use thousands of clips, report more than one metric, and listen.

## The exercise

You build `shortcut_target`, `shortcut_loss`, `meanflow_target`, `frechet_distance`, `fad`, `fad_infinity`, and `clap_score`. The checks confirm the two-step target and its missing gradient, and the loss on flow rows and on self-consistency rows, including its gradient against a hand-built reference. They check that the exact average velocity of a Gaussian flow is a fixed point of your MeanFlow target, which a sign error breaks. They test FAD against its 1-D, diagonal, and commuting closed forms, its sample-size bias, FAD-infinity, and the CLAP score. The last check trains a small shortcut model and requires its one-step samples to land on the ring.

Press **Run** to train a plain flow, a shortcut model, and a MeanFlow model on the eight-blob ring and hear twelve one-step samples from each as notes, in tune only when they land on the data. The run prints FAD against the number of steps for all three, and shows FAD's sample-size bias and its dependence on the embedding.
