---
title: Diffusion for audio
slug: 055-diffusion-for-audio
part: "Part 7 — Diffusion and flow matching"
summary: Choose where to run diffusion on audio, fit its noise schedule to quiet waveforms, and sample in six steps with DiffWave.
minutes: 70
gpu: true
objectives:
  - Explain why audio forces three decisions on a diffusion model, namely where to diffuse, which schedule to use, and how few steps to take.
  - Build DiffWave's step embedding and its dilated, gated residual block, and compute the stack's receptive field.
  - Align a short hand-chosen noise schedule to fractional training steps, and sample with it in 6 steps.
  - Compute the mean and variance of Grad-TTS's forward process, which diffuses toward an encoder's mel instead of toward zero.
  - Train a small DiffWave-style vocoder on spoken digits and hear its samples.
prereqs:
  - 053-fast-samplers
  - 044-wavenet
  - 011-hearing-and-the-mel-scale
exercise: 055-diffusion-for-audio
---

# Diffusion for audio

You trained a diffusion model on two-dimensional points, and it worked. Now hand it one second of speech at 22,050 samples per second and keep everything else: the same noise schedule, the same thousand sampling steps. Three things go wrong. The network must keep 22,050 numbers coherent in phase. The schedule was built for data with a standard deviation near 1, and speech has a standard deviation near 0.1, so the noise hides the signal too early. And a thousand passes for one second of sound is far more than a listener will wait for.

It needs three decisions: where to run diffusion, which noise schedule matches how quiet the data is, and how to cover that schedule in a few steps. You build DiffWave's pieces, and on a GPU you train a small DiffWave on spoken digits and hear it.

## What you already know

- [DDPM](/lessons/052-diffusion-from-scratch) trains a network to predict the noise in $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, with data at $t = 0$ and $\bar\alpha_0 = 1$.
- The DDPM reverse step subtracts scaled predicted noise, divides by $\sqrt{\alpha_t}$, and adds posterior noise except on the last step.
- [WaveNet](/lessons/044-wavenet) stacks dilated convolutions with gated activations, $\tanh(\cdot) \odot \sigma(\cdot)$, and residual and skip connections.
- A vocoder turns a [mel spectrogram](/lessons/011-hearing-and-the-mel-scale) into a waveform.
- Signal-to-noise ratio (SNR) is signal power divided by noise power, and a quieter signal has a lower SNR at the same noise level.

## Three places to run diffusion

Every sampling step passes the whole sequence through the network, so a step costs time proportional to the sequence's length. The first decision is therefore how long a sequence to diffuse. Figure 1 shows the options for one second of audio.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Three sequence lengths per second of audio on a log scale: 22,050 waveform samples, 86 mel frames, 21.5 latent frames">
<defs><marker id="c050-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="200.0" y1="40.0" x2="200.0" y2="190.0" class="stroke-grid"/>
<text x="200.0" y="206.0" class="fig-small" text-anchor="middle">10</text>
<line x1="280.0" y1="40.0" x2="280.0" y2="190.0" class="stroke-grid"/>
<text x="280.0" y="206.0" class="fig-small" text-anchor="middle">100</text>
<line x1="360.0" y1="40.0" x2="360.0" y2="190.0" class="stroke-grid"/>
<text x="360.0" y="206.0" class="fig-small" text-anchor="middle">1,000</text>
<line x1="440.0" y1="40.0" x2="440.0" y2="190.0" class="stroke-grid"/>
<text x="440.0" y="206.0" class="fig-small" text-anchor="middle">10,000</text>
<line x1="520.0" y1="40.0" x2="520.0" y2="190.0" class="stroke-grid"/>
<text x="520.0" y="206.0" class="fig-small" text-anchor="middle">100,000</text>
<line x1="200.0" y1="190.0" x2="520.0" y2="190.0" class="stroke-ink"/>
<rect x="200.0" y="50.0" width="267.5" height="30.0" class="fill-a"/>
<text x="190.0" y="70.0" class="fig-small" text-anchor="end">DiffWave: waveform</text>
<text x="473.5" y="70.0" class="fig-small" text-anchor="start">22,050</text>
<rect x="200.0" y="98.0" width="74.8" height="30.0" class="fill-c"/>
<text x="190.0" y="118.0" class="fig-small" text-anchor="end">Grad-TTS: mel, hop 256</text>
<text x="280.8" y="118.0" class="fig-small" text-anchor="start">86.1</text>
<rect x="200.0" y="146.0" width="26.6" height="30.0" class="fill-d"/>
<text x="190.0" y="166.0" class="fig-small" text-anchor="end">Stable Audio: latent</text>
<text x="232.6" y="166.0" class="fig-small" text-anchor="start">21.5</text>
<text x="360.0" y="228.0" class="fig-small" text-anchor="middle">numbers per second of audio (log scale)</text>
</svg>
Figure 1. Each step of a diffusion model touches every number in the sequence, so moving from the waveform to a mel spectrogram to a latent cuts the work of one step by a factor of 256 and then 4. Notice that the bar for the waveform is more than a thousand times longer than the bar for the latent.
```

DiffWave diffuses the raw waveform, conditioned on a mel spectrogram (Kong et al., 2021). That is the expensive choice, but a vocoder's output is the waveform. The mel supplies the long-range structure, such as which phoneme comes next, so the network only fills in fine detail.

Grad-TTS diffuses the mel spectrogram itself (Popov et al., 2021). At a hop of 256 samples, a mel spectrogram has $22{,}050 / 256 \approx 86$ frames per second, 256 times fewer numbers than the waveform. A separate vocoder turns the result into sound.

AudioLDM and Stable Audio first train an autoencoder, then diffuse its *latents*, the short codes in its middle. Stable Audio Open's autoencoder downsamples a 44.1 kHz waveform by 2,048, so the latent has $44{,}100 / 2{,}048 \approx 21.5$ frames per second.

> [!KEY]
> Where you diffuse sets the price of every step. The waveform is the longest sequence, the mel is 256 times shorter, and a latent is shorter still. The rest of this chapter uses the waveform, because it shows the other two decisions most clearly.

## DiffWave's denoiser

DiffWave's network is WaveNet with two changes. Its convolutions are not causal, because it sees the whole noisy waveform at once and generates every sample in parallel. And every layer is told two things: the diffusion step, so one network can denoise at every noise level, and the mel spectrogram, so it knows which words to produce.

The paper's network has 30 residual layers with kernel size 3, and the dilation cycles through $1, 2, 4, \ldots, 512$ three times. The BASE model has $C = 64$ residual channels and 2.64M parameters. One layer computes the following:

$$
y = \operatorname{DilConv}\left(x + W_t\, e(t)\right) + W_c\, c, \qquad
h = \tanh(y_{\text{filter}}) \odot \sigma(y_{\text{gate}}), \qquad
(r, s) = W_o h, \qquad x' = \frac{x + r}{\sqrt 2}
$$

Read it aloud: add the projected step embedding $e(t)$ to the layer's input, run the dilated convolution to $2C$ channels, and add the projection of the upsampled mel $c$. Split the result in half. The tanh half proposes values, the sigmoid half decides how much of each passes, and the product is $h$. A $1 \times 1$ convolution splits $h$ into a residual $r$ and a skip output $s$. The residual is added back to the input and divided by $\sqrt 2$, because the sum of two unit-variance signals has variance 2, and the division keeps the variance steady through 30 layers. The skips of all layers are summed and projected to the noise estimate.

```python
y = self.dilated_conv(x + self.diffusion_projection(emb)[:, :, None])
y = y + self.conditioner_projection(cond)
gate, filt = torch.chunk(y, 2, dim=1)
residual, skip = torch.chunk(self.output_projection(torch.sigmoid(gate) * torch.tanh(filt)), 2, dim=1)
```

The step enters as a 128-dimensional sinusoid, $e(t) = [\sin(t \cdot 10^{4j/63}), \cos(t \cdot 10^{4j/63})]_{j=0}^{63}$, followed by two fully connected layers to 512 dimensions. It is defined for any real $t$, which the fast sampler below depends on. The exercise's `TinyDiffWave` upsamples the mel with one transposed 1-D convolution.

### Work out the receptive field

A kernel-3 convolution with dilation $d$ sees $d$ samples on each side, so the stack sees 1 plus twice the sum of its dilations. One cycle sums to 1023:

- DiffWave's 30 layers: $1 + 2 \times 3 \times 1023 = 6{,}139$ samples, 0.28 s at 22.05 kHz.

## A schedule that fits quiet audio

DiffWave trains with DDPM's unweighted noise loss, conditioned on the mel: $\lVert \epsilon - \epsilon_\theta(x_t, c, t) \rVert^2$. For the vocoder, $T = 50$ steps, and $\beta_t$ rises linearly from $10^{-4}$ to 0.05. That schedule ends at $\bar\alpha_{50} = 0.280$, so $\sqrt{\bar\alpha_{50}} = 0.529$ of the signal is still there. Whether that counts as noise depends on how loud the data is. The SNR at step 50 is the data's variance times $\bar\alpha_{50}$ over $1 - \bar\alpha_{50}$:

- For unit-variance data, $0.280 / 0.720 = 0.39$, about $-4$ dB. The signal still shows through, so starting from pure noise would be wrong.
- For a waveform with standard deviation 0.1, $0.01 \times 0.280 / 0.720 = 0.0039$, about $-24$ dB. The signal is buried, and pure noise is a fair starting point.

> [!WARNING]
> DiffWave's schedule relies on waveforms being quiet. Reuse it on normalized mel spectrograms, which have unit variance, and sampling starts from noise that the model never saw at step $T$. Pick a schedule by the SNR it gives your data at $t = T$.

## Sample in six steps

Fifty passes is still many. The obvious shortcut is to pick six noise variances $\eta_1, \ldots, \eta_6$ by hand, and run the DDPM step with them while telling the network "step 1" through "step 6". That fails, because the network does not know "noise level 3". It knows step numbers, and training step 6 means $\sqrt{\bar\alpha_6} = 0.992$, almost clean audio. DiffWave's BASE schedule $\eta = (0.0001, 0.001, 0.01, 0.05, 0.2, 0.5)$ has a sixth step with a signal level near 0.61, which the network would read as a nearly clean sample.

The fix is to ask the network about the right noise level instead of the right step number. Compute $\gamma_s = 1 - \eta_s$ and $\bar\gamma_s = \prod_{i \le s} \gamma_i$ the way DDPM computes $\bar\alpha$, then match each fast step to the fractional training step whose $\sqrt{\bar\alpha}$ equals $\sqrt{\bar\gamma_s}$ (Figure 2).

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The signal level sqrt(alpha_bar) falls from 1 to 0.53 over 50 training steps, and six fast steps sit on that curve at fractional steps 1.0, 1.9, 5.1, 11.5, 24.0, and 43.9">
<defs><marker id="c050b-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-grid"/>
<text x="70.0" y="256.0" class="fig-small" text-anchor="middle">0</text>
<line x1="170.0" y1="30.0" x2="170.0" y2="240.0" class="stroke-grid"/>
<text x="170.0" y="256.0" class="fig-small" text-anchor="middle">10</text>
<line x1="270.0" y1="30.0" x2="270.0" y2="240.0" class="stroke-grid"/>
<text x="270.0" y="256.0" class="fig-small" text-anchor="middle">20</text>
<line x1="370.0" y1="30.0" x2="370.0" y2="240.0" class="stroke-grid"/>
<text x="370.0" y="256.0" class="fig-small" text-anchor="middle">30</text>
<line x1="470.0" y1="30.0" x2="470.0" y2="240.0" class="stroke-grid"/>
<text x="470.0" y="256.0" class="fig-small" text-anchor="middle">40</text>
<line x1="570.0" y1="30.0" x2="570.0" y2="240.0" class="stroke-grid"/>
<text x="570.0" y="256.0" class="fig-small" text-anchor="middle">50</text>
<line x1="70.0" y1="210.0" x2="570.0" y2="210.0" class="stroke-grid"/>
<text x="64.0" y="214.0" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="150.0" x2="570.0" y2="150.0" class="stroke-grid"/>
<text x="64.0" y="154.0" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="90.0" x2="570.0" y2="90.0" class="stroke-grid"/>
<text x="64.0" y="94.0" class="fig-small" text-anchor="end">0.8</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="64.0" y="34.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-ink"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-ink"/>
<path d="M70.0 30.0 L80.0 30.0 L90.0 30.2 L100.0 30.5 L110.0 31.0 L120.0 31.6 L130.0 32.4 L140.0 33.3 L150.0 34.4 L160.0 35.6 L170.0 37.0 L180.0 38.5 L190.0 40.1 L200.0 41.9 L210.0 43.8 L220.0 45.9 L230.0 48.1 L240.0 50.4 L250.0 52.9 L260.0 55.4 L270.0 58.1 L280.0 60.9 L290.0 63.8 L300.0 66.8 L310.0 69.9 L320.0 73.2 L330.0 76.5 L340.0 79.8 L350.0 83.3 L360.0 86.9 L370.0 90.5 L380.0 94.2 L390.0 98.0 L400.0 101.8 L410.0 105.7 L420.0 109.6 L430.0 113.6 L440.0 117.6 L450.0 121.6 L460.0 125.7 L470.0 129.8 L480.0 134.0 L490.0 138.1 L500.0 142.3 L510.0 146.4 L520.0 150.6 L530.0 154.8 L540.0 158.9 L550.0 163.1 L560.0 167.2 L570.0 171.3" class="stroke-a" fill="none"/>
<line x1="70.0" y1="30.0" x2="80.0" y2="30.0" class="stroke-muted dashed"/>
<circle cx="80.0" cy="30.0" r="5" class="fill-b"/>
<line x1="70.0" y1="30.2" x2="88.9" y2="30.2" class="stroke-muted dashed"/>
<circle cx="88.9" cy="30.2" r="5" class="fill-b"/>
<line x1="70.0" y1="31.7" x2="120.9" y2="31.7" class="stroke-muted dashed"/>
<circle cx="120.9" cy="31.7" r="5" class="fill-b"/>
<line x1="70.0" y1="39.2" x2="184.5" y2="39.2" class="stroke-muted dashed"/>
<circle cx="184.5" cy="39.2" r="5" class="fill-b"/>
<text x="192.5" y="31.2" class="fig-small" text-anchor="start">s=4, t=11.5</text>
<line x1="70.0" y1="69.9" x2="309.9" y2="69.9" class="stroke-muted dashed"/>
<circle cx="309.9" cy="69.9" r="5" class="fill-b"/>
<text x="317.9" y="61.9" class="fig-small" text-anchor="start">s=5, t=24.0</text>
<line x1="70.0" y1="146.1" x2="509.2" y2="146.1" class="stroke-muted dashed"/>
<circle cx="509.2" cy="146.1" r="5" class="fill-b"/>
<text x="517.2" y="138.1" class="fig-small" text-anchor="start">s=6, t=43.9</text>
<text x="90.0" y="190.0" class="fig-small" text-anchor="start">The dots at the top left are s = 1, 2, 3, at t = 1.0, 1.9, 5.1</text>
<text x="320.0" y="282.0" class="fig-small" text-anchor="middle">training step t (0 is clean audio)</text>
<text x="70.0" y="20.0" class="fig-small" text-anchor="start">signal level sqrt(alpha_bar)</text>
</svg>
Figure 2. The blue curve is the signal level the network was trained on. Each orange dot is a fast step placed at the fractional training step with the same level, so the network is only asked about noise levels it has seen.
```

$$
t_s^{\text{align}} = t + \frac{\sqrt{\bar\alpha_t} - \sqrt{\bar\gamma_s}}{\sqrt{\bar\alpha_t} - \sqrt{\bar\alpha_{t+1}}}
\quad \text{where} \quad \sqrt{\bar\alpha_{t+1}} \le \sqrt{\bar\gamma_s} \le \sqrt{\bar\alpha_t}
$$

Read it aloud: find the two training steps whose signal levels bracket the fast step's level, and interpolate linearly in $\sqrt{\bar\alpha}$ between them. The embedding accepts any real $t$, so the network receives that fractional step directly.

### Work the alignment by hand

The BASE schedule's cumulative products are $\bar\gamma = (0.9999, 0.99890, 0.98891, 0.93947, 0.75157, 0.37579)$. The training schedule starts with $\beta_1 = 10^{-4}$ and $\beta_2 = 0.0011184$, so $\bar\alpha_1 = 0.9999$ and $\bar\alpha_2 = 0.998782$.

- Fast step 1: $\bar\gamma_1 = 0.9999 = \bar\alpha_1$, so $t^{\text{align}} = 1.000$.
- Fast step 2: $\sqrt{\bar\gamma_2} = 0.999450$ lies between $\sqrt{\bar\alpha_1} = 0.999950$ and $\sqrt{\bar\alpha_2} = 0.999391$. The fraction is $0.000500 / 0.000559 = 0.894$, so $t^{\text{align}} = 1.894$.
- The same computation for all six steps gives $(1.000, 1.894, 5.087, 11.452, 23.992, 43.919)$.

> [!KEY]
> A fast schedule reuses the trained network unchanged. Each fast step asks it about the fractional training step whose signal level matches, so the network only sees noise levels it was trained on.

Sampling then runs the DDPM step with the fast schedule's own coefficients, from $s = 6$ down to 1:

$$
x \leftarrow \frac{1}{\sqrt{\gamma_s}} \left( x - \frac{\eta_s}{\sqrt{1 - \bar\gamma_s}}\, \epsilon_\theta\left(x, c, t_s^{\text{align}}\right) \right) + \sigma_s z, \qquad
\sigma_s^2 = \frac{1 - \bar\gamma_{s-1}}{1 - \bar\gamma_s}\, \eta_s
$$

Read it aloud: this is the DDPM reverse step with $\eta$ in place of $\beta$ and $\bar\gamma$ in place of $\bar\alpha$, and the network asked about the aligned step. No noise is added at $s = 1$, and DiffWave's code clamps $x$ to $[-1, 1]$ after each step.

The last fast step stops at $\bar\gamma_6 = 0.376$, about step 44, not at step 50. The paper reports MOS 4.37 for BASE with this schedule, against 4.44 for LARGE with all 200 of its training steps.

## Grad-TTS: start near the answer

Plain diffusion ends at pure noise, so the network builds everything from nothing. A TTS system already has a guess: a Glow-TTS-style encoder and a duration model produce $\mu$, a blurry, frame-aligned mel for the text. Grad-TTS's forward process drifts toward $\mu$ instead of toward zero, so the decoder only adds detail. With data at $t = 0$ and $t \in [0, 1]$:

$$
dX_t = \tfrac12 (\mu - X_t)\, \beta_t\, dt + \sqrt{\beta_t}\, dW_t, \qquad
X_t \mid X_0 \sim \mathcal{N}\left(e^{-B_t / 2} X_0 + \left(1 - e^{-B_t / 2}\right) \mu,\ \left(1 - e^{-B_t}\right) I\right)
$$

Read it aloud: the sample is pulled toward $\mu$ at rate $\frac12 \beta_t$ while noise is added. With $B_t = \int_0^t \beta_s\, ds$, the mean blends the data and $\mu$, and the variance grows from 0 toward 1. Grad-TTS uses $\beta_t = 0.05 + (20 - 0.05)\, t$, so $B_t = 0.05 t + 9.975 t^2$. At $t = 0.5$, $B_t = 2.51875$, so the mean is $0.284\, X_0 + 0.716\, \mu$: the sample has already moved most of the way to $\mu$. The following table gives the numbers the exercise checks.

| $t$ | $B_t$ | Coefficient on $X_0$, $e^{-B_t/2}$ | Variance $1 - e^{-B_t}$ |
|---|---|---|---|
| 0.1 | 0.10475 | 0.949 | 0.0995 |
| 0.5 | 2.51875 | 0.284 | 0.9194 |
| 1.0 | 10.025 | 0.0067 | 0.99996 |

At $t = 1$ the sample is almost exactly $\mathcal{N}(\mu, I)$, and generation starts there.

> [!RECAP]
> - Diffusion for audio is the same recipe with three decisions: where to diffuse (waveform, mel, or latent), which schedule fits the data's loudness, and how to take few steps.
> - A DiffWave layer gates a dilated convolution of the input plus step embedding plus mel, and splits the result into a residual scaled by $1 / \sqrt 2$ and a skip.
> - Fast sampling aligns each $\bar\gamma_s$ to a fractional training step by linear interpolation in $\sqrt{\bar\alpha}$, then runs the DDPM step with $\eta_s$ and $\bar\gamma_s$.
> - Grad-TTS's forward process has mean $e^{-B_t/2} X_0 + (1 - e^{-B_t/2}) \mu$ and variance $1 - e^{-B_t}$.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`diffusion_embedding(t, dim)`.** Frequencies $10^{4j/(h-1)}$, an outer product with $t$, then sine and cosine concatenated. Mistake: rounding $t$, which breaks fractional steps.
2. **`ResidualBlock`.** Build the four named layers, then follow the formula. Mistake: `padding=1` on the dilated convolution, which shortens the output.
3. **`diffwave_alignment(train_betas, infer_betas)`.** Two cumulative products and a search for the bracketing training steps, counted from 1. Mistake: interpolating in $\bar\alpha$ instead of $\sqrt{\bar\alpha}$.
4. **`fast_sampling_coefficients(infer_betas)`.** Vectorized $c_1$, $c_2$, and $\sigma$, with $\sigma = 0$ at the first entry.
5. **`grad_tts_stats(x0, mu, t, beta0, beta1)`.** $B_t$, then the mean and variance.
6. **`diffwave_loss(model, audio, mel, alpha_bar, generator)`.** The DDPM loss with the mel passed through. Draw on the CPU and move to the audio's device, or the **Run** fails on a GPU.
7. **`fast_sample(model, mel, length, infer_betas, t_align, generator)`.** Loop $s$ from $S - 1$ down to 0, and clamp after each step.

## Where this breaks

**Too few steps with a model trained for many.** Without alignment, a 50-step model sampled at 6 arbitrary steps is asked about noise levels it never saw. Align, or condition on the noise level itself, as WaveGrad does. A schedule also breaks when the data's scale changes, as the warning above says.

## The exercise

You build `diffusion_embedding`, `ResidualBlock`, `diffwave_alignment`, `fast_sampling_coefficients`, `grad_tts_stats`, `diffwave_loss`, and `fast_sample`; the rest of `TinyDiffWave` and the log-mel front end are provided. The checks compare the embedding with the formula at whole and fractional steps, test the gated block against a hand computation, and confirm that an impulse moves its output only within the dilation. They compare your alignment with the DiffWave repository's loop for the BASE and LARGE schedules, check the coefficients, and match the Grad-TTS table and a simulation of its SDE. The last checks recover the noise your loss used and run your fast sampler against a reference loop.

Press **Run** on a GPU worker to train a DiffWave-style vocoder (20 layers of 64 channels, 1.96M parameters, batch 32, 7 minutes) on 500 recordings of one speaker from `fsdd`, then hear a held-out digit, its reconstruction from the mel in 6 fast steps, and in all 50 steps. Without a GPU the **Run** shrinks the model to 8 layers of 16 channels and trains for 40 seconds. Its loss falls from 1.00 to about 0.15, and the loudness envelope of its 6-step sample correlates about 0.87 with the original's, under a steady hiss.
