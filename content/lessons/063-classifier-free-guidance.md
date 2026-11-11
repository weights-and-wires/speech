---
title: Classifier-free guidance
slug: 063-classifier-free-guidance
part: "Part 7 — Diffusion and flow matching"
summary: Steer a diffusion or flow sampler toward a condition by extrapolating from an unconditional prediction past the conditional one.
minutes: 30
gpu: false
objectives:
  - Say why a conditional sampler can follow its condition loosely, and why guidance fixes that without a separate classifier.
  - Train with the condition dropped at random, so one network gives both predictions.
  - Combine a conditional and an unconditional prediction in the w and s conventions, and convert between them.
  - Show that guided scores are the score of the tilted distribution, which is narrower and further from the unconditional mean.
prereqs:
  - 061-diffusion-from-scratch
  - 062-fast-samplers
exercise: 063-classifier-free-guidance
---

# Classifier-free guidance

You ask a text-to-speech model for a sentence, and it says something close. A word is slurred, or a name is swapped for a more common one. The model was trained with the transcript, and the sampler uses it, but the sample is still just a typical draw from "speech that fits this transcript". Typical draws sometimes fit it poorly.

This chapter builds the fix that nearly every text-to-audio and TTS system based on diffusion or flow matching uses: *classifier-free guidance*. You make the model predict twice, once with the condition and once without it, and push the sample in the direction that separates the two. The cost is one extra prediction per step. The gain is clearer, more on-script audio, at the price of some variety.

## What you already know

- A diffusion model estimates a noise, a denoised value, or a score at each noise level (see [Diffusion from scratch](/lessons/061-diffusion-from-scratch)).
- A sampler such as EDM's Heun method turns those estimates into a sample ([Sampling faster](/lessons/062-fast-samplers)).
- The score is the gradient of the log density, so it points toward more likely samples.
- Bayes' rule says $p(x \mid c) \propto p(c \mid x)\, p(x)$, where $c$ is the condition, such as a transcript or a speaker.

## Two predictions and a gap

Suppose the network can make two predictions at the same point: $\epsilon(x, c)$ with the condition, and $\epsilon(x, \varnothing)$ with a *null* condition that says nothing. The null prediction describes what is likely in general. The conditional one describes what is likely given $c$. Their difference is exactly what the condition adds.

The obvious plan is to use the conditional prediction alone. That samples the conditional distribution faithfully, which means it also reproduces its weak spots: samples that barely depend on $c$ are common, because the model assigns them real probability. A second plan, *classifier guidance*, trains a separate classifier on noisy inputs and follows its gradient. That works, but it needs a second network that must handle every noise level, and its gradient can drift away from what the generator knows.

Classifier-free guidance (Ho and Salimans, 2022) takes a third route. If the gap tells you what the condition adds, move further along the gap. Starting at the unconditional prediction, travel past the conditional one:

$$
\tilde\epsilon(x, c) = (1 + w)\, \epsilon(x, c) - w\, \epsilon(x, \varnothing)
$$

Read it aloud: take the conditional prediction, then add $w$ more copies of the gap between the conditional and unconditional predictions. The number $w$ is the *guidance weight*. At $w = 0$ you get the conditional prediction back, so there is no guidance.

```python
guided = (1 + w) * cond - w * uncond
```

Take a single number to see it. Let the conditional prediction be 2 and the unconditional one 1, so the gap is 1. At $w = 1$ the guided value is $2 \times 2 - 1 \times 1 = 3$. At $w = 3$ it is $4 \times 2 - 3 \times 1 = 5$. Each unit of $w$ moves the value one more gap past the conditional prediction.

```figure
<svg viewBox="0 0 600 200" role="img" aria-label="A number line of prediction values: unconditional 1, conditional 2, and guided predictions 3 at w = 1 and 5 at w = 3">
<defs><marker id="c050-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-b"/></marker></defs>
<line x1="60" y1="130" x2="540" y2="130" class="stroke-muted"/>
<line x1="60" y1="126" x2="60" y2="134" class="stroke-muted"/><text x="60" y="150" class="fig-small" text-anchor="middle">0</text>
<line x1="140" y1="126" x2="140" y2="134" class="stroke-muted"/><text x="140" y="150" class="fig-small" text-anchor="middle">1</text>
<line x1="220" y1="126" x2="220" y2="134" class="stroke-muted"/><text x="220" y="150" class="fig-small" text-anchor="middle">2</text>
<line x1="300" y1="126" x2="300" y2="134" class="stroke-muted"/><text x="300" y="150" class="fig-small" text-anchor="middle">3</text>
<line x1="380" y1="126" x2="380" y2="134" class="stroke-muted"/><text x="380" y="150" class="fig-small" text-anchor="middle">4</text>
<line x1="460" y1="126" x2="460" y2="134" class="stroke-muted"/><text x="460" y="150" class="fig-small" text-anchor="middle">5</text>
<line x1="540" y1="126" x2="540" y2="134" class="stroke-muted"/><text x="540" y="150" class="fig-small" text-anchor="middle">6</text>
<line x1="146" y1="95" x2="452" y2="95" class="stroke-b" stroke-width="2.5" marker-end="url(#c050-arrow)"/>
<circle cx="140" cy="130" r="6" class="fill-d"/><text x="140" y="112" class="fig-small" text-anchor="middle">unconditional 1</text>
<circle cx="220" cy="130" r="6" class="fill-a"/><text x="220" y="172" class="fig-small" text-anchor="middle">conditional 2 (w = 0)</text>
<circle cx="300" cy="130" r="6" class="fill-b"/><text x="300" y="55" class="fig-small" text-anchor="middle">w = 1: 3</text>
<circle cx="460" cy="130" r="6" class="fill-b"/><text x="460" y="55" class="fig-small" text-anchor="middle">w = 3: 5</text>
<line x1="300" y1="62" x2="300" y2="124" class="stroke-grid dashed"/><line x1="460" y1="62" x2="460" y2="124" class="stroke-grid dashed"/>
<text x="300" y="20" class="fig-small" text-anchor="middle">prediction value (the guided value is (1 + w) * 2 - w * 1)</text>
</svg>
Figure 1. Guidance starts at the unconditional prediction, passes the conditional one, and keeps going in the same direction. Notice that each unit of w adds one more copy of the gap, 2 minus 1.
```

## One network, two predictions

You do not train two networks. During training, replace the condition with the null token on a random 10 to 20 percent of examples. The same network then learns both jobs, and you ask it for each by feeding a real condition or the null one. The null must be a fixed, learned or zeroed input, because at sampling time the unconditional branch has to see exactly what it saw in training.

```python
drop = rng.random(len(cond)) < p
cond = np.where(drop[:, None], null[None, :], cond)
```

At sampling time, run the network on both inputs at every step and combine the outputs. This doubles the number of evaluations unless you stack the two inputs into one batch, which most implementations do. The same combination works for noise predictions, denoiser outputs, scores, and flow-matching velocities, because each is a linear function of the same underlying quantity.

## Why the combination steers

Why should extrapolation help? Read the guided prediction as a score. With scores, the guided score is

$$
(1 + w)\, \nabla_x \log p(x \mid c) - w\, \nabla_x \log p(x)
$$

and Bayes' rule says $\nabla_x \log p(x \mid c) - \nabla_x \log p(x) = \nabla_x \log p(c \mid x)$. So the guided score is the score of $p(x \mid c)\, p(c \mid x)^w$. The sampler is pulled toward samples that a classifier would confidently assign to $c$, and you get that classifier without training one.

Work a tiny case where every density is Gaussian. Let the unconditional density be $\mathcal{N}(0, 4)$, with mean 0 and variance 4, and the conditional density $\mathcal{N}(3, 1)$. At $x = 2$ the scores are $s_c = -(2 - 3)/1 = 1$ and $s_\varnothing = -2/4 = -0.5$. With $w = 2$, the guided score is $3 \times 1 - 2 \times (-0.5) = 4$.

The tilted density has precision (one over the variance) $3 \times 1 - 2 \times \tfrac14 = 2.5$, so its variance is $0.4$. Its mean is $(3 \times 3 \times 1 - 2 \times 0)/2.5 = 3.6$. Its score at $x = 2$ is $-(2 - 3.6)/0.4 = 4$, which matches. Guidance with $w = 2$ turned $\mathcal{N}(3, 1)$ into $\mathcal{N}(3.6, 0.4)$.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Three densities over x: the unconditional N(0, 4), the conditional N(3, 1), and the guided N(3.6, 0.4) for w = 2, which is narrower and further from 0">
<line x1="60" y1="230" x2="540" y2="230" class="stroke-muted"/><line x1="60" y1="230" x2="60" y2="30" class="stroke-muted"/>
<line x1="60.0" y1="230" x2="60.0" y2="234" class="stroke-muted"/><text x="60.0" y="248" class="fig-small" text-anchor="middle">-6</text>
<line x1="128.6" y1="230" x2="128.6" y2="234" class="stroke-muted"/><text x="128.6" y="248" class="fig-small" text-anchor="middle">-4</text>
<line x1="197.1" y1="230" x2="197.1" y2="234" class="stroke-muted"/><text x="197.1" y="248" class="fig-small" text-anchor="middle">-2</text>
<line x1="265.7" y1="230" x2="265.7" y2="234" class="stroke-muted"/><text x="265.7" y="248" class="fig-small" text-anchor="middle">0</text>
<line x1="334.3" y1="230" x2="334.3" y2="234" class="stroke-muted"/><text x="334.3" y="248" class="fig-small" text-anchor="middle">2</text>
<line x1="402.9" y1="230" x2="402.9" y2="234" class="stroke-muted"/><text x="402.9" y="248" class="fig-small" text-anchor="middle">4</text>
<line x1="471.4" y1="230" x2="471.4" y2="234" class="stroke-muted"/><text x="471.4" y="248" class="fig-small" text-anchor="middle">6</text>
<line x1="540.0" y1="230" x2="540.0" y2="234" class="stroke-muted"/><text x="540.0" y="248" class="fig-small" text-anchor="middle">8</text>
<polyline points="60.0,229.4 68.6,229.1 77.1,228.7 85.7,228.1 94.3,227.4 102.9,226.5 111.4,225.4 120.0,223.9 128.6,222.1 137.1,219.9 145.7,217.4 154.3,214.4 162.9,211.1 171.4,207.3 180.0,203.3 188.6,199.0 197.1,194.6 205.7,190.2 214.3,186.0 222.9,182.0 231.4,178.5 240.0,175.7 248.6,173.5 257.1,172.1 265.7,171.7 274.3,172.1 282.9,173.5 291.4,175.7 300.0,178.5 308.6,182.0 317.1,186.0 325.7,190.2 334.3,194.6 342.9,199.0 351.4,203.3 360.0,207.3 368.6,211.1 377.1,214.4 385.7,217.4 394.3,219.9 402.9,222.1 411.4,223.9 420.0,225.4 428.6,226.5 437.1,227.4 445.7,228.1 454.3,228.7 462.9,229.1 471.4,229.4 480.0,229.6 488.6,229.7 497.1,229.8 505.7,229.9 514.3,229.9 522.9,229.9 531.4,230.0 540.0,230.0" class="stroke-d" fill="none" stroke-width="2.5"/>
<polyline points="60.0,230.0 68.6,230.0 77.1,230.0 85.7,230.0 94.3,230.0 102.9,230.0 111.4,230.0 120.0,230.0 128.6,230.0 137.1,230.0 145.7,230.0 154.3,230.0 162.9,230.0 171.4,230.0 180.0,230.0 188.6,230.0 197.1,230.0 205.7,230.0 214.3,230.0 222.9,230.0 231.4,230.0 240.0,229.9 248.6,229.7 257.1,229.4 265.7,228.7 274.3,227.3 282.9,224.9 291.4,220.7 300.0,214.2 308.6,204.8 317.1,192.1 325.7,176.6 334.3,159.3 342.9,142.0 351.4,127.1 360.0,117.0 368.6,113.4 377.1,117.0 385.7,127.1 394.3,142.0 402.9,159.3 411.4,176.6 420.0,192.1 428.6,204.8 437.1,214.2 445.7,220.7 454.3,224.9 462.9,227.3 471.4,228.7 480.0,229.4 488.6,229.7 497.1,229.9 505.7,230.0 514.3,230.0 522.9,230.0 531.4,230.0 540.0,230.0" class="stroke-a" fill="none" stroke-width="2.5"/>
<polyline points="60.0,230.0 68.6,230.0 77.1,230.0 85.7,230.0 94.3,230.0 102.9,230.0 111.4,230.0 120.0,230.0 128.6,230.0 137.1,230.0 145.7,230.0 154.3,230.0 162.9,230.0 171.4,230.0 180.0,230.0 188.6,230.0 197.1,230.0 205.7,230.0 214.3,230.0 222.9,230.0 231.4,230.0 240.0,230.0 248.6,230.0 257.1,230.0 265.7,230.0 274.3,230.0 282.9,230.0 291.4,230.0 300.0,230.0 308.6,229.8 317.1,229.3 325.7,227.4 334.3,222.5 342.9,211.1 351.4,189.4 360.0,155.3 368.6,112.4 377.1,71.8 385.7,47.9 394.3,50.7 402.9,79.0 411.4,121.3 420.0,163.0 428.6,194.7 437.1,214.1 445.7,223.9 454.3,228.0 462.9,229.4 471.4,229.9 480.0,230.0 488.6,230.0 497.1,230.0 505.7,230.0 514.3,230.0 522.9,230.0 531.4,230.0 540.0,230.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<text x="300" y="268" class="fig-small" text-anchor="middle">x (a pitch offset, for example in semitones)</text><text x="64" y="22" class="fig-small">density</text>
<text x="279.7" y="163.7" class="fig-small">p(x): N(0, 4)</text>
<text x="248.6" y="117.4" class="fig-small">p(x | c): N(3, 1)</text>
<text x="433.1" y="49.6" class="fig-small">guided, w = 2: N(3.6, 0.4)</text>
</svg>
Figure 2. Guiding with w = 2 gives a Gaussian that is both further from the unconditional mean and narrower than the conditional one. Notice that the peak rises from 0.40 to 0.63 because the same area is squeezed into a smaller width.
```

> [!KEY]
> Guidance does two things at once. It moves samples away from what is typical in general, and it narrows them around what the condition asks for. Both come from the same extrapolation, and you cannot get one without the other.

## Two conventions for one number

Papers disagree on what the scale means. Ho and Salimans, Voicebox, and F5-TTS use $w$, with $w = 0$ meaning no guidance. Diffusers, AudioLDM, and Tango write the prediction as $\epsilon_\varnothing + s\,(\epsilon_c - \epsilon_\varnothing)$, with $s = 1$ meaning no guidance. Expand the second form and you get the first with $s = 1 + w$. F5-TTS's default of 2 in the $w$ convention is $s = 3$.

```python
guided_s = uncond + s * (cond - uncond)   # equals cfg_combine(cond, uncond, s - 1)
```

> [!WARNING]
> Copying a guidance scale between the two conventions runs without an error and gives a guidance one unit too weak or too strong. Check which convention a codebase uses before you reuse its number.

## Hear guidance narrow a distribution

The **Run** models pitches in semitones as two classes, low notes around 57 and high notes around 69, each with a standard deviation of 2. It samples the high class with EDM's Heun sampler and exact denoisers, so you can see the effect without training anything:

- With $w = 0$: mean 68.82, standard deviation 2.02, close to the class itself.
- With $w = 1$: mean 70.05, standard deviation 1.43.
- With $w = 4$: mean 71.81, standard deviation 1.05.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Distribution of sampled pitches in semitones for guidance weights 0, 1, and 4: the mean moves from 68.82 to 71.81 and the standard deviation shrinks from 2.02 to 1.05">
<line x1="60" y1="230" x2="570" y2="230" class="stroke-muted"/><line x1="60" y1="230" x2="60" y2="30" class="stroke-muted"/>
<text x="60.0" y="248" class="fig-small" text-anchor="middle">60</text>
<text x="173.3" y="248" class="fig-small" text-anchor="middle">64</text>
<text x="286.7" y="248" class="fig-small" text-anchor="middle">68</text>
<text x="400.0" y="248" class="fig-small" text-anchor="middle">72</text>
<text x="513.3" y="248" class="fig-small" text-anchor="middle">76</text>
<polyline points="60.0,230.0 62.8,230.0 65.7,230.0 68.5,230.0 71.3,230.0 74.2,230.0 77.0,230.0 79.8,230.0 82.7,230.0 85.5,230.0 88.3,229.9 91.2,229.9 94.0,229.9 96.8,229.9 99.7,229.9 102.5,229.9 105.3,229.8 108.2,229.8 111.0,229.8 113.8,229.7 116.7,229.7 119.5,229.6 122.3,229.6 125.2,229.5 128.0,229.4 130.8,229.3 133.7,229.2 136.5,229.0 139.3,228.9 142.2,228.7 145.0,228.5 147.8,228.3 150.7,228.0 153.5,227.8 156.3,227.4 159.2,227.1 162.0,226.7 164.8,226.2 167.7,225.7 170.5,225.2 173.3,224.5 176.2,223.9 179.0,223.1 181.8,222.3 184.7,221.4 187.5,220.4 190.3,219.4 193.2,218.3 196.0,217.0 198.8,215.7 201.7,214.3 204.5,212.7 207.3,211.1 210.2,209.4 213.0,207.6 215.8,205.6 218.7,203.6 221.5,201.5 224.3,199.2 227.2,196.9 230.0,194.5 232.8,192.0 235.7,189.4 238.5,186.8 241.3,184.1 244.2,181.4 247.0,178.6 249.8,175.8 252.7,173.0 255.5,170.1 258.3,167.3 261.2,164.6 264.0,161.8 266.8,159.1 269.7,156.5 272.5,154.0 275.3,151.6 278.2,149.4 281.0,147.2 283.8,145.2 286.7,143.4 289.5,141.7 292.3,140.3 295.2,139.0 298.0,138.0 300.8,137.1 303.7,136.5 306.5,136.1 309.3,136.0 312.2,136.0 315.0,136.3 317.8,136.9 320.7,137.6 323.5,138.6 326.3,139.8 329.2,141.1 332.0,142.7 334.8,144.5 337.7,146.4 340.5,148.5 343.3,150.7 346.2,153.1 349.0,155.5 351.8,158.1 354.7,160.7 357.5,163.5 360.3,166.2 363.2,169.0 366.0,171.8 368.8,174.7 371.7,177.5 374.5,180.3 377.3,183.0 380.2,185.7 383.0,188.4 385.8,191.0 388.7,193.5 391.5,196.0 394.3,198.3 397.2,200.6 400.0,202.8 402.8,204.8 405.7,206.8 408.5,208.7 411.3,210.4 414.2,212.1 417.0,213.7 419.8,215.1 422.7,216.5 425.5,217.8 428.3,218.9 431.2,220.0 434.0,221.0 436.8,222.0 439.7,222.8 442.5,223.6 445.3,224.3 448.2,224.9 451.0,225.5 453.8,226.0 456.7,226.5 459.5,226.9 462.3,227.3 465.2,227.6 468.0,227.9 470.8,228.2 473.7,228.4 476.5,228.6 479.3,228.8 482.2,229.0 485.0,229.1 487.8,229.3 490.7,229.4 493.5,229.5 496.3,229.5 499.2,229.6 502.0,229.7 504.8,229.7 507.7,229.8 510.5,229.8 513.3,229.8 516.2,229.9 519.0,229.9 521.8,229.9 524.7,229.9 527.5,229.9 530.3,229.9 533.2,230.0 536.0,230.0 538.8,230.0 541.7,230.0 544.5,230.0 547.3,230.0 550.2,230.0 553.0,230.0 555.8,230.0 558.7,230.0 561.5,230.0 564.3,230.0 567.2,230.0 570.0,230.0" class="stroke-a" fill="none" stroke-width="2.5"/>
<polyline points="60.0,230.0 62.8,230.0 65.7,230.0 68.5,230.0 71.3,230.0 74.2,230.0 77.0,230.0 79.8,230.0 82.7,230.0 85.5,230.0 88.3,230.0 91.2,230.0 94.0,230.0 96.8,230.0 99.7,230.0 102.5,230.0 105.3,230.0 108.2,230.0 111.0,230.0 113.8,230.0 116.7,230.0 119.5,230.0 122.3,230.0 125.2,230.0 128.0,230.0 130.8,230.0 133.7,230.0 136.5,230.0 139.3,230.0 142.2,230.0 145.0,230.0 147.8,230.0 150.7,230.0 153.5,230.0 156.3,230.0 159.2,230.0 162.0,230.0 164.8,230.0 167.7,230.0 170.5,230.0 173.3,230.0 176.2,230.0 179.0,230.0 181.8,230.0 184.7,229.9 187.5,229.9 190.3,229.9 193.2,229.9 196.0,229.8 198.8,229.8 201.7,229.7 204.5,229.7 207.3,229.6 210.2,229.5 213.0,229.3 215.8,229.2 218.7,229.0 221.5,228.7 224.3,228.4 227.2,228.0 230.0,227.6 232.8,227.1 235.7,226.5 238.5,225.7 241.3,224.9 244.2,223.9 247.0,222.8 249.8,221.5 252.7,220.0 255.5,218.3 258.3,216.3 261.2,214.2 264.0,211.8 266.8,209.1 269.7,206.1 272.5,202.9 275.3,199.4 278.2,195.6 281.0,191.5 283.8,187.1 286.7,182.5 289.5,177.6 292.3,172.5 295.2,167.2 298.0,161.7 300.8,156.2 303.7,150.6 306.5,144.9 309.3,139.3 312.2,133.9 315.0,128.5 317.8,123.5 320.7,118.7 323.5,114.2 326.3,110.2 329.2,106.6 332.0,103.6 334.8,101.1 337.7,99.2 340.5,97.9 343.3,97.2 346.2,97.2 349.0,97.9 351.8,99.2 354.7,101.1 357.5,103.6 360.3,106.6 363.2,110.2 366.0,114.2 368.8,118.7 371.7,123.5 374.5,128.5 377.3,133.9 380.2,139.3 383.0,144.9 385.8,150.6 388.7,156.2 391.5,161.7 394.3,167.2 397.2,172.5 400.0,177.6 402.8,182.5 405.7,187.1 408.5,191.5 411.3,195.6 414.2,199.4 417.0,202.9 419.8,206.1 422.7,209.1 425.5,211.8 428.3,214.2 431.2,216.3 434.0,218.3 436.8,220.0 439.7,221.5 442.5,222.8 445.3,223.9 448.2,224.9 451.0,225.7 453.8,226.5 456.7,227.1 459.5,227.6 462.3,228.0 465.2,228.4 468.0,228.7 470.8,229.0 473.7,229.2 476.5,229.3 479.3,229.5 482.2,229.6 485.0,229.7 487.8,229.7 490.7,229.8 493.5,229.8 496.3,229.9 499.2,229.9 502.0,229.9 504.8,229.9 507.7,230.0 510.5,230.0 513.3,230.0 516.2,230.0 519.0,230.0 521.8,230.0 524.7,230.0 527.5,230.0 530.3,230.0 533.2,230.0 536.0,230.0 538.8,230.0 541.7,230.0 544.5,230.0 547.3,230.0 550.2,230.0 553.0,230.0 555.8,230.0 558.7,230.0 561.5,230.0 564.3,230.0 567.2,230.0 570.0,230.0" class="stroke-c" fill="none" stroke-width="2.5"/>
<polyline points="60.0,230.0 62.8,230.0 65.7,230.0 68.5,230.0 71.3,230.0 74.2,230.0 77.0,230.0 79.8,230.0 82.7,230.0 85.5,230.0 88.3,230.0 91.2,230.0 94.0,230.0 96.8,230.0 99.7,230.0 102.5,230.0 105.3,230.0 108.2,230.0 111.0,230.0 113.8,230.0 116.7,230.0 119.5,230.0 122.3,230.0 125.2,230.0 128.0,230.0 130.8,230.0 133.7,230.0 136.5,230.0 139.3,230.0 142.2,230.0 145.0,230.0 147.8,230.0 150.7,230.0 153.5,230.0 156.3,230.0 159.2,230.0 162.0,230.0 164.8,230.0 167.7,230.0 170.5,230.0 173.3,230.0 176.2,230.0 179.0,230.0 181.8,230.0 184.7,230.0 187.5,230.0 190.3,230.0 193.2,230.0 196.0,230.0 198.8,230.0 201.7,230.0 204.5,230.0 207.3,230.0 210.2,230.0 213.0,230.0 215.8,230.0 218.7,230.0 221.5,230.0 224.3,230.0 227.2,230.0 230.0,230.0 232.8,230.0 235.7,230.0 238.5,230.0 241.3,230.0 244.2,230.0 247.0,230.0 249.8,230.0 252.7,230.0 255.5,230.0 258.3,230.0 261.2,230.0 264.0,230.0 266.8,230.0 269.7,230.0 272.5,230.0 275.3,229.9 278.2,229.9 281.0,229.9 283.8,229.8 286.7,229.7 289.5,229.6 292.3,229.5 295.2,229.3 298.0,229.1 300.8,228.7 303.7,228.3 306.5,227.7 309.3,227.0 312.2,226.1 315.0,225.0 317.8,223.5 320.7,221.8 323.5,219.6 326.3,217.0 329.2,213.9 332.0,210.3 334.8,206.0 337.7,201.0 340.5,195.4 343.3,189.1 346.2,182.0 349.0,174.2 351.8,165.7 354.7,156.6 357.5,146.9 360.3,136.9 363.2,126.5 366.0,116.1 368.8,105.7 371.7,95.6 374.5,86.0 377.3,77.2 380.2,69.2 383.0,62.4 385.8,56.8 388.7,52.7 391.5,50.1 394.3,49.1 397.2,49.7 400.0,52.0 402.8,55.8 405.7,61.1 408.5,67.7 411.3,75.5 414.2,84.2 417.0,93.7 419.8,103.7 422.7,114.0 425.5,124.4 428.3,134.8 431.2,144.9 434.0,154.7 436.8,163.9 439.7,172.5 442.5,180.5 445.3,187.7 448.2,194.2 451.0,200.0 453.8,205.0 456.7,209.4 459.5,213.2 462.3,216.4 465.2,219.1 468.0,221.4 470.8,223.2 473.7,224.7 476.5,225.9 479.3,226.9 482.2,227.6 485.0,228.2 487.8,228.7 490.7,229.0 493.5,229.3 496.3,229.5 499.2,229.6 502.0,229.7 504.8,229.8 507.7,229.9 510.5,229.9 513.3,229.9 516.2,230.0 519.0,230.0 521.8,230.0 524.7,230.0 527.5,230.0 530.3,230.0 533.2,230.0 536.0,230.0 538.8,230.0 541.7,230.0 544.5,230.0 547.3,230.0 550.2,230.0 553.0,230.0 555.8,230.0 558.7,230.0 561.5,230.0 564.3,230.0 567.2,230.0 570.0,230.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<text x="64.0" y="48" class="fig-small">w = 0: mean 68.82, std 2.02</text><line x1="64.0" y1="56" x2="100.0" y2="56" class="stroke-a" stroke-width="2.5"/>
<text x="64.0" y="78" class="fig-small">w = 1: mean 70.05, std 1.43</text><line x1="64.0" y1="86" x2="100.0" y2="86" class="stroke-c" stroke-width="2.5"/>
<text x="64.0" y="108" class="fig-small">w = 4: mean 71.81, std 1.05</text><line x1="64.0" y1="116" x2="100.0" y2="116" class="stroke-b" stroke-width="2.5"/>
<text x="315.0" y="268" class="fig-small" text-anchor="middle">pitch in semitones (the low class sits near 57)</text>
</svg>
Figure 3. The sampled pitch distributions for three guidance weights. Notice that guidance moves the mean away from the other class and also narrows the spread.
```

You hear it as twelve notes that wander at $w = 0$ and cluster high and close at $w = 4$. In TTS the same effect gives clearer, more on-script speech with less variety, and too much gives strained, over-bright audio.

> [!RECAP]
> - A conditional sampler follows its condition loosely, because it reproduces every draw the model finds plausible.
> - Training with the condition dropped on 10 to 20 percent of examples gives one network both predictions.
> - Guidance is $(1 + w) f_c - w f_\varnothing$, which in the score view is the score of $p(x \mid c)\, p(c \mid x)^w$.
> - The $s$ convention is $s = 1 + w$, so $s = 1$ means no guidance.
> - Guidance moves the mean away from the unconditional mean and shrinks the spread.

## Build it step by step

The EDM sampler from the previous chapter is provided in full. You build four functions, in the order the checks test them.

1. **`drop_condition(cond, null, p, rng)`.** Draw one uniform number per row and replace the row by the null when the number is below $p$. Common mistake: drawing one number for the whole batch, which drops everything or nothing.
2. **`cfg_combine(cond, uncond, w)`.** One line. Common mistake: writing $\text{cond} + w\,\text{uncond}$ and forgetting the subtraction.
3. **`cfg_combine_s(cond, uncond, s)`.** Start at `uncond` and add `s` times the gap. Common mistake: passing $w$ where this function wants $s$.
4. **`guided_denoiser(D_cond, D_uncond, w)`.** Return a function of `(x, sigma)` that calls both denoisers at the same point and combines them with `cfg_combine`.

## Where this breaks

**Guidance too strong.** Large $w$ oversaturates and removes variety, because the guided distribution keeps narrowing. MaskGCT and others rescale the guided prediction back toward the conditional prediction's standard deviation.

**A null condition that differs from training.** The unconditional branch must see exactly the null input it was trained with: a zeroed embedding, a learned null token, or dropped prompt audio. Otherwise the gap measures the mismatch, not the condition.

**Cost.** Two predictions per step double the compute. Batch them together to keep the extra cost in arithmetic rather than in launches.

## The exercise

You build `drop_condition`, `cfg_combine`, `cfg_combine_s`, and `guided_denoiser`. The checks verify the null replacement rate, the values 2, 3, and 5 from the hand example, the equivalence of the two conventions, the tilted score from the Gaussian example, and that guiding a two-class mixture moves the mean away from the other class and narrows the samples.

Press **Run** to print the hand example, see the sampled pitch densities for three weights, and hear twelve notes at $w = 0$ and at $w = 4$.
