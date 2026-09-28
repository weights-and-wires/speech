---
title: Diffusion for audio
slug: 35-diffusion-for-audio
part: "Part 7 — Diffusion and flow matching"
summary: Build DiffWave's gated denoiser and its 6-step fast sampler, Grad-TTS's mel prior, and the latent recipe behind AudioLDM and Stable Audio.
minutes: 80
gpu: true
objectives:
  - Build DiffWave's step embedding and its dilated, gated residual block, and compute the stack's receptive field.
  - Align a short user-chosen noise schedule to fractional training steps, and sample with it in 6 steps.
  - Compute the mean and variance of Grad-TTS's forward process, which diffuses toward an encoder's mel instead of toward zero.
  - Explain latent diffusion as AudioLDM and Stable Audio use it, and compute how much shorter the latent sequence is.
  - Train a small DiffWave-style vocoder on spoken digits and hear its samples.
prereqs:
  - 34-samplers-scores-and-guidance
  - 26-wavenet-and-autoregressive-vocoders
  - 07-hearing-and-the-mel-scale
exercise: 35-diffusion-for-audio
---

# Diffusion for audio

The last two lessons built diffusion on toy data. Audio adds three problems.
A waveform is long: one second at 22.05 kHz is 22,050 numbers that must stay
coherent in phase. It is quiet: speech has a standard deviation near 0.1, so
a noise schedule designed for unit-variance images spends most of its steps
on pure noise. And listeners want it fast, so a thousand sampling steps is
out of the question.

Three systems answer these problems in three different places. DiffWave
denoises the waveform directly, conditioned on a mel spectrogram, with a
WaveNet-like network and a 6-step sampler [Kong et al. 2021]. Grad-TTS
diffuses mel spectrograms, and starts the reverse process from the text
encoder's rough mel instead of from zero [Popov et al. 2021]. AudioLDM and
Stable Audio diffuse in the latent space of an autoencoder, where a
10-second clip is a few hundred frames.

In this lesson you build DiffWave's pieces, train a small one on spoken
digits, and compute the numbers behind the other two.

## What you already know

- DDPM trains a network to predict the noise in $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, with data at $t = 0$ and $\bar\alpha_0 = 1$.
- The DDPM reverse step subtracts scaled predicted noise, divides by $\sqrt{\alpha_t}$, and adds posterior noise except on the last step.
- WaveNet stacks dilated convolutions with gated activations, $\tanh(\cdot) \odot \sigma(\cdot)$, and residual and skip connections.
- A vocoder turns a mel spectrogram into a waveform, and GAN vocoders such as HiFi-GAN do it in one forward pass.
- A small-amplitude signal has a much lower SNR at a given step than the schedule's unit-variance numbers suggest.

## DiffWave's denoiser

DiffWave's network looks like WaveNet with two changes. Its convolutions are
not causal, because it sees the whole noisy waveform at once and generates
every sample in parallel. And every layer is told the diffusion step and the
mel spectrogram.

The paper's network has 30 residual layers with kernel size 3, and the
dilation cycles through $1, 2, 4, \ldots, 512$ three times. The BASE model
has $C = 64$ residual channels and 2.64M parameters, and LARGE has
$C = 128$ and 6.91M. One layer computes the following:

$$
y = \operatorname{DilConv}\left(x + W_t\, e(t)\right) + W_c\, c, \qquad
h = \tanh(y_{\text{filter}}) \odot \sigma(y_{\text{gate}}), \qquad
(r, s) = W_o h, \qquad x' = \frac{x + r}{\sqrt 2}
$$

Read it aloud: add the projected step embedding $e(t)$ to the layer's input,
run the dilated convolution to $2C$ channels, and add the $2C$-channel
projection of the upsampled mel $c$. Split the result in half, pass one half
through tanh and the other through a sigmoid gate, and multiply. A
$1 \times 1$ convolution makes $2C$ channels again, split into a residual $r$,
added back to the input and scaled by $1 / \sqrt 2$ to keep the variance
steady, and a skip output $s$. The skips of all layers are summed and
projected to the noise estimate.

```python
y = self.dilated_conv(x + self.diffusion_projection(emb)[:, :, None])
y = y + self.conditioner_projection(cond)
gate, filt = torch.chunk(y, 2, dim=1)
residual, skip = torch.chunk(self.output_projection(torch.sigmoid(gate) * torch.tanh(filt)), 2, dim=1)
```

The step enters as a 128-dimensional sinusoid,
$e(t) = [\sin(t \cdot 10^{4j/63}), \cos(t \cdot 10^{4j/63})]_{j=0}^{63}$,
followed by two fully connected layers to 512 dimensions. The mel frames,
256 samples apart, are upsampled 256 times by two transposed 2-D
convolutions. The exercise's `TinyDiffWave` uses one transposed 1-D
convolution for the same job.

### Work out the receptive field

A kernel-3 convolution with dilation $d$ sees $d$ samples on each side. The
stack's receptive field is 1 plus twice the sum of its dilations. One cycle
sums to $1 + 2 + \cdots + 512 = 1023$, so:

- DiffWave's 30 layers: $1 + 2 \times 3 \times 1023 = 6{,}139$ samples, 0.28 s at 22.05 kHz.
- The **Run**'s GPU model, 20 layers: $1 + 2 \times 2 \times 1023 = 4{,}093$ samples, 0.51 s at 8 kHz.
- The **Run**'s CPU model, 8 layers with dilations up to 128: $1 + 2 \times 255 = 511$ samples, 64 ms.

The mel conditioning carries the long-range structure, so the receptive field
only has to cover a few pitch periods.

## Training, and a schedule that ends early

DiffWave trains with DDPM's unweighted noise loss, conditioned on the mel:
$\lVert \epsilon - \epsilon_\theta(x_t, c, t) \rVert^2$. For the vocoder with
$T = 50$ steps, $\beta_t$ rises linearly from $10^{-4}$ to 0.05. That schedule
ends at $\bar\alpha_{50} = 0.280$, so $\sqrt{\bar\alpha_{50}} = 0.529$ of the
signal is still there. For unit-variance data that would be nowhere near
noise, but for a waveform with standard deviation 0.1 the SNR at step 50 is
$0.01 \times 0.280 / 0.720$, about $-24$ dB.

> [!WARNING]
> DiffWave's schedule relies on waveforms being quiet. Reuse it on normalized mel spectrograms, which have unit variance, and sampling starts from noise that the model never saw at step $T$. Pick a schedule by the SNR it gives your data at $t = T$.

## Sample in six steps

DiffWave's fast sampler picks a short schedule of $S$ noise variances
$\eta_1, \ldots, \eta_S$ by hand, and computes $\gamma_s = 1 - \eta_s$ and
$\bar\gamma_s = \prod_{i \le s} \gamma_i$ the way DDPM computes $\bar\alpha$.
The BASE schedule is
$\eta = (0.0001, 0.001, 0.01, 0.05, 0.2, 0.5)$. But the network only knows the
50 training steps. So each fast step $s$ is matched to the fractional training
step whose $\sqrt{\bar\alpha}$ equals $\sqrt{\bar\gamma_s}$:

$$
t_s^{\text{align}} = t + \frac{\sqrt{\bar\alpha_t} - \sqrt{\bar\gamma_s}}{\sqrt{\bar\alpha_t} - \sqrt{\bar\alpha_{t+1}}}
\quad \text{where} \quad \sqrt{\bar\alpha_{t+1}} \le \sqrt{\bar\gamma_s} \le \sqrt{\bar\alpha_t}
$$

Read it aloud: find the two training steps whose signal levels bracket the
fast step's level, and interpolate linearly in $\sqrt{\bar\alpha}$ between
them. The network receives this fractional step through the sinusoidal
embedding, which is defined for any real $t$.

```python
for n in range(T - 1):
    if ab[n + 1] <= g <= ab[n]:
        t_align = n + 1 + (sqrt(ab[n]) - sqrt(g)) / (sqrt(ab[n]) - sqrt(ab[n + 1]))
```

Sampling then runs the DDPM step with the fast schedule's own coefficients,
from $s = S$ down to 1:

$$
x \leftarrow \frac{1}{\sqrt{\gamma_s}} \left( x - \frac{\eta_s}{\sqrt{1 - \bar\gamma_s}}\, \epsilon_\theta\left(x, c, t_s^{\text{align}}\right) \right) + \sigma_s z, \qquad
\sigma_s^2 = \frac{1 - \bar\gamma_{s-1}}{1 - \bar\gamma_s}\, \eta_s
$$

Read it aloud: this is the DDPM reverse step with $\eta$ in place of $\beta$
and $\bar\gamma$ in place of $\bar\alpha$, and the network is asked about the
aligned training step. No noise is added on the final step, $s = 1$.
DiffWave's code also clamps $x$ to $[-1, 1]$ after each step.

### Work the alignment by hand

The BASE schedule's cumulative products are
$\bar\gamma = (0.9999, 0.99890, 0.98891, 0.93947, 0.75157, 0.37579)$.
The training schedule starts with $\beta_1 = 10^{-4}$ and
$\beta_2 = 0.0011184$, so $\bar\alpha_1 = 0.9999$ and $\bar\alpha_2 = 0.998782$.

- Fast step 1: $\bar\gamma_1 = 0.9999 = \bar\alpha_1$, so $t^{\text{align}} = 1.000$.
- Fast step 2: $\sqrt{\bar\gamma_2} = 0.999450$ lies between $\sqrt{\bar\alpha_1} = 0.999950$ and $\sqrt{\bar\alpha_2} = 0.999391$. The fraction is $(0.999950 - 0.999450) / (0.999950 - 0.999391) = 0.894$, so $t^{\text{align}} = 1.894$.
- The same computation for all six steps gives $(1.000, 1.894, 5.087, 11.452, 23.992, 43.919)$.

The last fast step stops at $\bar\gamma_6 = 0.376$, about step 44, not at
step 50. Sampling starts from standard noise anyway, which for quiet audio
is close enough. The paper reports MOS 4.37 for BASE with this 6-step
schedule, 5.6 times faster than real time on a V100, against 4.44 for LARGE
with all 200 of its training steps.

## Grad-TTS: diffuse toward a rough mel

Grad-TTS works on mel spectrograms. A Glow-TTS-style encoder and a duration
model produce $\mu$, a blurry, frame-aligned mel for the text. The diffusion
decoder only has to add detail, so its forward process drifts toward $\mu$
instead of toward zero. With data at $t = 0$ and $t \in [0, 1]$:

$$
dX_t = \tfrac12 (\mu - X_t)\, \beta_t\, dt + \sqrt{\beta_t}\, dW_t, \qquad
X_t \mid X_0 \sim \mathcal{N}\left(e^{-B_t / 2} X_0 + \left(1 - e^{-B_t / 2}\right) \mu,\ \left(1 - e^{-B_t}\right) I\right)
$$

Read it aloud: the sample is pulled toward $\mu$ at rate $\frac12 \beta_t$
while noise is added. With $B_t = \int_0^t \beta_s\, ds$, the mean blends the
data and $\mu$, and the variance grows from 0 toward 1. Grad-TTS uses
$\beta_t = 0.05 + (20 - 0.05)\, t$, so $B_t = 0.05 t + 9.975 t^2$. At $t = 1$
the sample is almost exactly $\mathcal{N}(\mu, I)$, and generation starts
there.

The following table gives the numbers the exercise checks.

| $t$ | $B_t$ | Coefficient on $X_0$, $e^{-B_t/2}$ | Variance $1 - e^{-B_t}$ |
|---|---|---|---|
| 0.1 | 0.10475 | 0.949 | 0.0995 |
| 0.5 | 2.51875 | 0.284 | 0.9194 |
| 1.0 | 10.025 | 0.0067 | 0.99996 |

At synthesis, Grad-TTS samples the start from $\mathcal{N}(\mu, \tau^{-1} I)$
with temperature $\tau = 1.5$, and its code defaults to 50 ODE steps.

## Latent diffusion: AudioLDM and Stable Audio

For sound effects and music, the dominant recipe is to train an autoencoder
first, then run diffusion on its latents. The latent sequence is hundreds of
times shorter than the waveform, which is what makes minute-long generation
affordable.

- **AudioLDM** compresses mel spectrograms with a VAE and trains the latent diffusion model on CLAP *audio* embeddings, then samples it with CLAP *text* embeddings, which share the same space. Training needs no captions. A HiFi-GAN turns the decoded mel into 16 kHz audio.
- **Stable Audio Open** uses a VAE on 44.1 kHz stereo waveforms with a 64-channel latent at 21.5 Hz, and a DiT of about 1.06B parameters conditioned on T5 text features and timing. It trains with the velocity objective and samples with DPM-Solver++ for 100 steps at guidance scale 7.

The frame arithmetic for Stable Audio Open's latent: $44{,}100 / 21.5 \approx 2{,}051$
samples per latent frame. Its longest output, 47 seconds, is about 1,010
latent frames instead of 2.07 million samples per channel.

> [!KEY]
> Diffusion for audio happens in one of three places. DiffWave denoises the waveform under a mel, Grad-TTS denoises the mel starting near an encoder's guess, and latent models denoise an autoencoder's short sequence. The shorter the sequence, the cheaper each of the many steps.

> [!RECAP]
> - A DiffWave layer adds the step embedding, runs a dilated convolution, adds the mel projection, gates with $\tanh \odot \sigma$, and splits into a residual scaled by $1 / \sqrt 2$ and a skip.
> - The receptive field is $1 + 2 \sum d$, 6,139 samples for 30 layers.
> - Fast sampling aligns each $\bar\gamma_s$ to a fractional training step by linear interpolation in $\sqrt{\bar\alpha}$, then runs the DDPM step with $\eta_s$ and $\bar\gamma_s$.
> - Grad-TTS's forward process has mean $e^{-B_t/2} X_0 + (1 - e^{-B_t/2}) \mu$ and variance $1 - e^{-B_t}$.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`diffusion_embedding(t, dim)`.** Frequencies $10^{4j/(h-1)}$, an outer product with $t$, then sine and cosine concatenated. Common mistake: rounding $t$ to an integer, which breaks the fractional steps.
2. **`ResidualBlock`.** Build the four named layers, then follow the formula. Common mistake: `padding=1` on the dilated convolution, which shortens the output by $2(d - 1)$ samples.
3. **`diffwave_alignment(train_betas, infer_betas)`.** Two cumulative products and a search for the bracketing training steps. Count from 1. Common mistake: interpolating in $\bar\alpha$ instead of $\sqrt{\bar\alpha}$.
4. **`fast_sampling_coefficients(infer_betas)`.** Vectorized $c_1$, $c_2$, and $\sigma$, with $\sigma = 0$ at the first entry.
5. **`grad_tts_stats(x0, mu, t, beta0, beta1)`.** $B_t$, then the mean and variance.
6. **`diffwave_loss(model, audio, mel, alpha_bar, generator)`.** The DDPM loss with the mel passed through. Draw on the CPU and move to the audio's device, or the **Run** fails on a GPU.
7. **`fast_sample(model, mel, length, infer_betas, t_align, generator)`.** Loop $s$ from $S - 1$ down to 0, and clamp after each step.

## Cost and scaling

The following table compares the vocoders and their costs.

| Model | Network passes per clip | Parameters | Notes |
|---|---|---|---|
| HiFi-GAN V1 | 1 | about 14M | Adversarial training |
| DiffWave BASE, fast | 6 | 2.64M | MOS 4.37, 5.6 times faster than real time on a V100 |
| DiffWave LARGE | 200 | 6.91M | MOS 4.44 |
| The **Run**'s GPU model | 6 | 1.96M | 20 layers of 64 channels at 8 kHz |

Each pass costs time proportional to the clip's length times the layers
times $C^2$. Training cost doesn't depend on $T$, because each example uses
one random step.

The **Run** trains for a fixed wall-clock budget. On a laptop CPU, the small
version, 8 layers of 16 channels, trained 349 steps in 40 seconds and ran
43 seconds in all. Its loss fell from 1.00 to 0.17, and the loudness envelope
of its 6-step sample correlated 0.87 with the original's, under a steady hiss.

The GPU version trains 20 layers of 64 channels at batch 32 for 7 minutes.
The same configuration, trained for 15 minutes on an Apple M-series GPU,
reached about 1,470 steps, at 0.6 seconds a step. Its loss fell from 1.00 to
0.064, its envelope correlation reached 0.89, and the hiss in the silent
parts fell to an RMS of 0.09 against 0.19 inside the digit. An A100 is
roughly ten times faster, so the 7-minute budget there buys several thousand
steps. That is an estimate, not a measurement. Expect a recognizable digit
with audible hiss, not a clean recording. Published diffusion vocoders train
for hundreds of times longer.

## Where this breaks

**Too few steps with a model trained for many.** Without alignment, running
a 50-step model at 6 arbitrary steps asks it about noise levels it never saw
at those step numbers. Align, or condition on the noise level itself, as
WaveGrad does.

**A schedule mismatched to the data's scale.** DiffWave's schedule assumes
quiet waveforms, and Grad-TTS assumes mels normalized so that $\mathcal{N}(\mu, I)$
is a sensible start. Change the data's scale and both break.

**Hiss in the high frequencies.** Diffusion vocoders trained briefly leave
broadband noise that GAN vocoders don't, because the last steps' small
errors are white. More training, more steps at low noise, or a
frequency-aware variant such as PeriodWave or FreGrad helps.

**Samples that are too quiet.** A network that predicts the conditional mean
of the noise under-disperses when you take few steps. With the exact
denoiser for a Gaussian of standard deviation 0.1, the 6-step BASE schedule
returns samples with a standard deviation of 0.065.

**Slow sampling after all.** Six passes of a 30-layer network over every
sample is still six times a GAN's single pass. In production, GAN vocoders
remain the default, and diffusion or flow models are distilled to one step.

## The exercise

You build `diffusion_embedding`, `ResidualBlock`, `diffwave_alignment`,
`fast_sampling_coefficients`, `grad_tts_stats`, `diffwave_loss`, and
`fast_sample`; the rest of `TinyDiffWave` and the log-mel front end are
provided. The checks compare the embedding with the formula at whole and
fractional steps, test the gated block against a hand computation from its own
weights, and confirm that an impulse moves its output only within the
dilation. They compare your alignment with the DiffWave repository's loop for
the BASE and LARGE schedules, check $\bar\gamma_6 = 0.37579$ and the
coefficients, and match the Grad-TTS table and a simulation of its SDE. The
last checks recover the noise your loss used and run your fast sampler
against a reference loop with the same random numbers.

Press **Run** on a GPU worker to train a DiffWave-style vocoder on 500
recordings of one speaker from `fsdd`, then hear a held-out digit, its
reconstruction from the mel in 6 fast steps, and in all 50 steps. Without a
GPU the **Run** shrinks the model and trains for 40 seconds, and without the
`fsdd` asset it trains on synthetic gliding tones.
