---
title: GAN vocoders
slug: 27-gan-vocoders
part: "Part 5 — Speech synthesis"
summary: Build the losses, discriminator reshape, and activation of HiFi-GAN and BigVGAN, then train a vocoder on real speech and hear it.
minutes: 90
gpu: true
objectives:
  - Compute the multi-resolution STFT loss from spectral convergence and log-magnitude distance at three resolutions.
  - Reshape a waveform by period for the multi-period discriminator, and explain why its periods are 2, 3, 5, 7, and 11.
  - Write the least-squares GAN losses and the feature-matching loss, and say which tensor to detach in each step.
  - Implement the Snake activation and its gradient, and explain its periodic bias.
  - Train a HiFi-GAN-style vocoder from 80-band log-mel to waveform, and compare it with Griffin-Lim by ear and by mel L1.
prereqs:
  - 26-wavenet-and-autoregressive-vocoders
  - 05-phase-and-griffin-lim
  - 07-hearing-and-the-mel-scale
exercise: 27-gan-vocoders
---

# GAN vocoders

WaveNet sounds natural and runs one sample at a time, far slower than real
time. A vocoder that generates every sample at once is easy to build: a
stack of transposed convolutions that stretches each mel frame into 256
samples. The hard part is training it. Trained only to match the waveform or
the spectrum, it produces muffled, buzzy audio, because many waveforms fit
one mel spectrogram and a regression loss averages them.

A *generative adversarial network* (GAN) fixes that with a second network, a
*discriminator*, trained to tell real audio from generated audio. The
generator is trained to fool it. The discriminator punishes exactly the
artifacts that a spectral loss ignores, such as smeared phase and missing
fine structure, so the generator learns to produce audio that sounds real
rather than audio that is average.

HiFi-GAN, from 2020, made this approach standard. In its paper's listening
test, the V1 model scored higher than a WaveNet vocoder while running 168
times faster than real time on a V100 GPU. In this lesson you build its losses, its
multi-period discriminator's reshape, and the Snake activation from BigVGAN,
then train a small HiFi-GAN-style vocoder on one LibriSpeech speaker.

## What you already know

- A vocoder turns a mel spectrogram into samples, and WaveNet does it one sample at a time (lesson 26, *WaveNet and autoregressive vocoders*).
- Griffin-Lim recovers a waveform from a magnitude spectrogram by iterating phase estimates, and it sounds metallic (lesson 5, *Phase and Griffin-Lim*).
- An 80-band log-mel spectrogram at a hop of 256 samples is the input most vocoders take (lesson 7, *Hearing and the mel scale*).
- An STFT frame's magnitude says how much of each frequency it holds; its phase says where each sinusoid is in its cycle (lesson 4, *The short-time Fourier transform*).
- The Frobenius norm of a matrix is the square root of the sum of its squared entries.

## The generator

A HiFi-GAN generator is a stack of upsampling stages. A convolution first
maps the 80 mel bands of each frame to $h_u$ channels. Then four transposed
convolutions stretch time by 8, 8, 2, and 2, a total of 256, the hop length,
halving the channels each time. After each one, a *multi-receptive-field
fusion* (MRF) block averages the outputs of three residual blocks with kernel
sizes 3, 7, and 11 and dilations 1, 3, and 5, so every stage sees patterns at
several time scales. A last convolution and a tanh give one sample per time
step. The paper's variants are the following:

| Variant | $h_u$ | Parameters | MOS on LJSpeech |
|---|---|---|---|
| V1 | 512 | 13.92M | 4.36 |
| V2 | 128 | 0.92M | 4.23 |
| V3 | 256, three stages | 1.46M | 4.05 |

Recordings scored 4.45 in the same test. The transposed convolutions use a
kernel of twice the stride and padding of half the stride. A kernel that
isn't a multiple of the stride overlaps unevenly and leaves a checkerboard
pattern, which you hear as a tone at the frame rate.

## The multi-resolution STFT loss

Before any discriminator, a vocoder needs a reconstruction loss that says
"the spectrum is right." One STFT resolution isn't enough: a long window
checks harmonics and misses the timing of a click, and a short window does
the reverse. Parallel WaveGAN's *multi-resolution STFT loss* checks three
resolutions at once. At each, it combines two terms:

$$
L_{\text{sc}} = \frac{\big\lVert\, |S| - |\hat S| \,\big\rVert_F}{\big\lVert\, |S| \,\big\rVert_F}, \qquad
L_{\text{mag}} = \frac{1}{N} \big\lVert \log |S| - \log |\hat S| \big\rVert_1, \qquad
L_{\text{aux}} = \frac{1}{M} \sum_{m=1}^{M} \left( L_{\text{sc}}^{(m)} + L_{\text{mag}}^{(m)} \right).
$$

Read it aloud: $|S|$ and $|\hat S|$ are the STFT magnitudes of the real and
generated audio. *Spectral convergence* $L_{\text{sc}}$ is the size of the
magnitude error relative to the size of the real magnitude, so loud bins
dominate it. $L_{\text{mag}}$ is the mean absolute difference of log
magnitudes over all $N$ entries, so a quiet bin counts as much as a loud
one. The loss averages both over $M = 3$ resolutions. Parallel WaveGAN used
these, all with Hann windows:

| FFT size | Window | Hop |
|---|---|---|
| 1024 | 600 | 120 |
| 2048 | 1200 | 240 |
| 512 | 240 | 50 |

In torch, for one resolution:

```python
mag = torch.stft(x, n_fft, hop, win, window=torch.hann_window(win),
                 return_complex=True).abs().clamp(min=1e-7)
sc = torch.linalg.norm(mag - mag_hat) / torch.linalg.norm(mag)
```

Two small examples show the difference. If $|S|$ has Frobenius norm 10 and
the error $|S| - |\hat S|$ has norm 2, then $L_{\text{sc}} = 0.2$. If every
generated magnitude is $e^{0.1}$ times the real one, $L_{\text{mag}} = 0.1$,
whether the bins are loud or quiet. Both terms are unchanged when you scale
both signals by the same gain.

> [!KEY]
> The multi-resolution STFT loss compares magnitudes at a long, a medium, and a
> short window, and mixes a relative error that weighs loud bins with a log
> error that weighs every bin alike.

## The multi-period discriminator

Voiced speech is periodic: a 200 Hz vowel at 16 kHz repeats every 80
samples. HiFi-GAN's *multi-period discriminator* (MPD) is built to judge
that periodicity. Each of its five sub-discriminators takes a period $p$,
folds the waveform of length $T$ into a 2-D array of $\lceil T/p \rceil$ rows
and $p$ columns, and applies 2-D convolutions whose kernels are 1 wide along
the columns:

$$
X^{(p)}_{r, c} = x_{r p + c}, \qquad r = 0, \ldots, \lceil T/p \rceil - 1, \quad c = 0, \ldots, p - 1.
$$

Read it aloud: row $r$ of the folded array holds samples $rp$ to $rp + p - 1$,
so column $c$ holds every $p$-th sample starting at $c$. A kernel that is 1
wide sees one column at a time, so each column, a signal sampled at $1/p$ of
the rate, is judged separately. If $T$ isn't a multiple of $p$, the end is
padded by reflection first. In torch:

```python
x = F.pad(x, (0, (-T) % p), mode="reflect")
folded = x.reshape(B, C, -1, p)
```

For example, fold 10 samples, $0, 1, \ldots, 9$, with $p = 4$. Two samples
of padding reflect the end as $8, 7$, and the rows are $[0, 1, 2, 3]$,
$[4, 5, 6, 7]$, and $[8, 9, 8, 7]$. Column 1 holds samples 1, 5, and 9.

The periods are 2, 3, 5, 7, and 11. They are prime so that no two
sub-discriminators look at overlapping sets of samples: a period of 4 would
repeat much of what the period of 2 already sees. HiFi-GAN pairs the MPD
with MelGAN's *multi-scale discriminator* (MSD), three sub-discriminators on
the raw waveform and on versions average-pooled by 2 and 4, which judge
structure across time. Later vocoders replace the MSD with discriminators on
spectrograms: UnivNet's multi-resolution discriminator, used by BigVGAN and
Vocos, and EnCodec's complex STFT discriminator.

## Adversarial and feature-matching losses

HiFi-GAN uses the *least-squares GAN* (LSGAN) objective. The discriminator
$D$ is a regressor trained to output 1 on real audio $x$ and 0 on generated
audio $G(s)$ from the mel $s$. The generator is trained to make $D$ output 1
on its audio:

$$
\mathcal{L}_{\text{adv}}(D; G) = \mathbb{E}\left[(D(x) - 1)^2 + D(G(s))^2\right], \qquad
\mathcal{L}_{\text{adv}}(G; D) = \mathbb{E}\left[(D(G(s)) - 1)^2\right].
$$

Read it aloud: the discriminator pays the squared distance of its real score
from 1 and of its fake score from 0. The generator pays the squared distance
of the fake score from 1. Each sub-discriminator outputs a map of scores, the
expectation is the mean over it, and the losses add over sub-discriminators.
For one score, $D(x) = 0.9$ and $D(G(s)) = 0.3$ give
$\mathcal{L}_D = 0.01 + 0.09 = 0.10$ and $\mathcal{L}_G = 0.49$. The
generator's gradient with respect to that score is $2(0.3 - 1) = -1.4$: raise
it.

*Feature matching* adds a denser signal. Every layer of a discriminator
computes features, and the generator is asked to produce the same features
on its audio as on the real audio:

$$
\mathcal{L}_{\text{FM}}(G; D) = \mathbb{E}\left[\sum_{i} \frac{1}{N_i} \big\lVert D^i(x) - D^i(G(s)) \big\rVert_1\right].
$$

Read it aloud: at each layer $i$, take the mean absolute difference over its
$N_i$ features, and add over the layers. Real features $[1, 2]$ against
generated $[1.5, 1]$ give $(0.5 + 1)/2 = 0.75$ for that layer.

The generator's total loss adds the mel term, the adversarial term, and
feature matching, with weights $\lambda_{\text{fm}} = 2$ and
$\lambda_{\text{mel}} = 45$:

$$
\mathcal{L}_G = \sum_{k} \left[\mathcal{L}_{\text{adv}}(G; D_k) + 2\,\mathcal{L}_{\text{FM}}(G; D_k)\right] + 45\,\big\lVert \phi(x) - \phi(G(s)) \big\rVert_1.
$$

Here $\phi$ is the log-mel transform and $D_k$ the $k$-th sub-discriminator.
The exercise's run also adds the MR-STFT loss with weight 2.5.

> [!WARNING]
> Detach $G(s)$ in the discriminator's step, or the discriminator's loss
> trains the generator to be caught. Don't detach it in the generator's step,
> or the generator gets no adversarial gradient at all.

## The Snake activation

A leaky ReLU is piecewise linear. A generator built from it has to assemble
every oscillation from many kinks, and it tends to extrapolate badly to pitches
it didn't see in training. BigVGAN gives the generator a periodic activation,
*Snake*:

$$
f_\alpha(x) = x + \frac{1}{\alpha} \sin^2(\alpha x), \qquad
\frac{\partial f_\alpha}{\partial x} = 1 + \sin(2 \alpha x).
$$

Read it aloud: Snake is the identity plus a periodic bump whose frequency is
set by a learned $\alpha$, one per channel. Its slope, $1 + \sin(2\alpha x)$,
never goes below 0, so the function never decreases, and the linear part lets
gradients pass as in a residual network. For example, at $x = 0.5$ with
$\alpha = 1$, $f = 0.5 + \sin^2(0.5) = 0.5 + 0.2298 = 0.7298$. With
$\alpha = 5$, $f = 0.5 + 0.2 \sin^2(2.5) = 0.5 + 0.2 \times 0.3582 = 0.5716$.

```python
a = alpha.reshape(1, -1, 1)
y = x + torch.sin(a * x) ** 2 / a
```

A nonlinearity applied to a sampled signal creates harmonics above the
Nyquist frequency, which fold back as aliases (lesson 1). BigVGAN therefore
upsamples by 2 before each Snake and low-pass filters and downsamples after.
The exercise's generator skips that filtering to stay small. BigVGAN scaled
this design to 112 million parameters and trained a universal vocoder that
works on unseen speakers, languages, and even music.

## Two more designs

**iSTFTNet** and **Vocos** cut the generator's work by predicting an STFT
and letting an inverse STFT do the upsampling. Vocos keeps the whole network
at the frame rate, predicts a magnitude $M = e^m$ and a phase $p$ for every
bin, and outputs $\operatorname{iSTFT}(M e^{j p})$. With 13.5 million
parameters it ran about 13 times faster than HiFi-GAN in its paper. F5-TTS
uses it as its vocoder.

> [!RECAP]
> - The MR-STFT loss averages spectral convergence plus log-magnitude L1 over three STFT resolutions.
> - The MPD folds a waveform into $p$ columns, with reflect padding, for $p \in \{2, 3, 5, 7, 11\}$.
> - LSGAN trains $D$ toward 1 on real and 0 on fake, and $G$ toward 1 on fake, summed over sub-discriminators.
> - Feature matching sums the mean absolute feature difference over every layer of every sub-discriminator.
> - Snake is $x + \sin^2(\alpha x)/\alpha$ with one $\alpha$ per channel.

## Build it step by step

Build the functions in the order the checks test them.

1. **`snake(x, alpha)`** reshapes `alpha` to broadcast over the batch and
   time, and returns $x + \sin^2(\alpha x)/\alpha$. The common mistake is
   broadcasting `alpha` along time instead of channels.
2. **`period_reshape(x, period)`** pads the end with `F.pad(..., mode="reflect")`
   and reshapes to `(B, C, T / p, p)`. The common mistake is a reshape to
   `(B, C, p, T / p)`, which puts consecutive samples in a column.
3. **`stft_magnitude(x, n_fft, hop, win)`** calls `torch.stft` with a
   periodic Hann window of length `win`, `center=True`, and reflect padding,
   and clamps at $10^{-7}$ so the log is finite.
4. **`spectral_convergence(mag_true, mag_pred)`** and
   **`log_magnitude_loss(mag_true, mag_pred)`** are one line each. The common
   mistake is dividing by the norm of the prediction.
5. **`multi_resolution_stft_loss(x, x_hat, resolutions)`** loops over the
   resolutions and returns the mean, not the sum.
6. **`lsgan_discriminator_loss`**, **`lsgan_generator_loss`**, and
   **`feature_matching_loss`** add a `torch.mean` per sub-discriminator, and
   per layer for feature matching.

## Cost and scaling

The following table compares vocoders by the speeds their papers reported on
a GPU.

| Vocoder | Parameters | Speed | Sequential steps per second of audio |
|---|---|---|---|
| WaveNet (mixture of logistics) | 24.7M | 0.003 times real time | one per sample |
| WaveGlow (flow) | 87.7M | 22.8 times real time | none |
| HiFi-GAN V1 | 13.9M | 168 times real time | none |
| Vocos | 13.5M | about 13 times HiFi-GAN | none |

Inference is one parallel pass, so its cost grows linearly with the length
of the audio. Training is the expensive part. HiFi-GAN trained for 2.5
million steps, with batches of 16 segments of 8,192 samples, which takes days
on one GPU. The
exercise's GPU run trains a V2-sized generator for a few minutes, so it shows
early training, not a finished vocoder.

## Where this breaks

**The mel settings differ.** The mel inside the loss must match the mel the
generator is conditioned on, band for band. A vocoder trained on one mel
range and run on another produces noise.

**The discriminator wins.** If $D$ learns much faster than $G$, its scores
saturate and the generator gets no useful gradient. Warm up with the
reconstruction losses alone, as the run does, and watch that $D$'s loss
doesn't collapse to 0.

**Artifacts at the frame rate.** Checkerboard patterns from transposed
convolutions, and aliasing from nonlinearities, show up as tones at
multiples of the frame rate. Look for horizontal lines in the output's
spectrogram.

**The numbers disagree with your ears.** Mel L1 against the input mel is the
standard objective check, but it doesn't measure phase. Griffin-Lim inverts
the mel filterbank exactly and then only fixes phase, so its mel L1 is low
even though it sounds metallic. A GAN vocoder early in training scores worse
on mel L1 and can still sound more natural. Listen, and use a listening test
(lesson 29) to decide.

## The exercise

You implement `snake`, `period_reshape`, `stft_magnitude`,
`spectral_convergence`, `log_magnitude_loss`, `multi_resolution_stft_loss`,
`lsgan_discriminator_loss`, `lsgan_generator_loss`, and
`feature_matching_loss`. The checks run on the CPU with synthetic signals.
They confirm Snake's hand-worked values and its gradient, the reshape for
periods 2 to 11 with reflect padding, your STFT against a numpy one, the
hand-worked 0.2 and 0.1, and that the MR-STFT loss is zero for a copy,
unchanged by gain, and rising with noise. They check the LSGAN gradient
against finite differences, and that a period-5 sub-discriminator built on
your reshape changes only one column when one sample changes.

Press **Run** to train. On a GPU, the run trains a V2-sized generator with
Snake activations and a five-period MPD on four minutes of LibriSpeech
speaker 1272, for at most seven minutes. On a CPU it trains a tiny version
for 35 seconds, and says its output is still noisy. Then it vocodes a
held-out clip from another chapter, prints mel L1 and the MR-STFT loss for
your vocoder and for Griffin-Lim, and plays the original, your vocoder, and
Griffin-Lim side by side.
