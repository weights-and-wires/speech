# Research notes 04: Flows, GANs, diffusion, and flow matching for audio

These notes are source material for the part of the course that covers generative
modelling of speech, sound, and music. Later agents turn them into lessons and labs, so
every section records the intuition, the exact equations with conventions, a hand-worked
number where possible, the canonical references, common student mistakes, and at least
one auto-checkable lab idea.

Research date: 2026-09-28. Facts about papers come from the full text of the arXiv PDFs
(downloaded and searched during research) unless marked otherwise. Facts about model
cards, licenses, and repositories come from the Hugging Face or GitHub pages on the
research date. Items that the research couldn't confirm are collected in
[Unverified or uncertain facts](#unverified-or-uncertain-facts).

Labels used in lab ideas:

- **CPU-60s**: runs in the browser editor's CPU sandbox in under 60 seconds. The toy 2-D
  flow-matching and DDPM trainers in this document were timed at about 1.5 s each for
  3,000 Adam steps (MLP 3-128-128-2, batch 512, 2 CPU threads, PyTorch 2.11).
- **GPU-L4** or **GPU-A100**: needs a RunPod serverless GPU. The time and VRAM estimates
  are engineering estimates, not measurements, unless the text says otherwise.

## Contents

1. [Conventions you must track](#conventions-you-must-track)
2. [Generative modelling recap for audio](#generative-modelling-recap-for-audio)
3. [Normalizing flows](#normalizing-flows)
4. [GANs for audio](#gans-for-audio)
5. [Diffusion models: DDPM, DDIM, score SDEs, and guidance](#diffusion-models-ddpm-ddim-score-sdes-and-guidance)
6. [Audio diffusion models](#audio-diffusion-models)
7. [Flow matching](#flow-matching)
8. [Flow matching in speech and audio](#flow-matching-in-speech-and-audio)
9. [Few-step generation and distillation](#few-step-generation-and-distillation)
10. [Autoregressive plus continuous hybrids](#autoregressive-plus-continuous-hybrids)
11. [Music generation and evaluation metrics](#music-generation-and-evaluation-metrics)
12. [Pretrained checkpoints for serverless labs](#pretrained-checkpoints-for-serverless-labs)
13. [Proposed lesson ordering](#proposed-lesson-ordering)
14. [Unverified or uncertain facts](#unverified-or-uncertain-facts)
15. [Bibliography](#bibliography)

## Conventions you must track

The single largest source of student bugs in this topic is the time direction. Papers
disagree about whether the time variable at 0 means data or noise, and about what the
symbols x_0 and x_1 denote. The following table records the conventions as they appear in
each source paper:

| Source | t = 0 is | t = end is | Data symbol | Noise symbol | Network predicts |
|---|---|---|---|---|---|
| DDPM (Ho et al. 2020), DiffWave, WaveGrad, DiffSinger | data | noise (t = T, discrete) | x_0 | x_T, ε | ε |
| Score SDE (Song et al. 2021), Grad-TTS, NaturalSpeech 2, SGMSE+ | data | noise (t = T or 1) | x_0 or X_0 | x_T | score (Grad-TTS, SGMSE+), x_0 (NaturalSpeech 2) |
| EDM (Karras et al. 2022) | data (σ = 0) | noise (σ_max = 80) | y | n ~ N(0, σ²I) | denoised D(x; σ) |
| Flow matching (Lipman et al. 2022), Matcha-TTS, Voicebox, E2 TTS, F5-TTS, CosyVoice 2 | noise | data (t = 1) | x_1 | x_0 | velocity x_1 - (1 - σ_min) x_0 |
| Rectified flow (Liu et al. 2022) | π_0 (noise, in generation) | π_1 (data) | X_1 | X_0 | X_1 - X_0 |
| SD3-style rectified flow, Stable Audio Open Small (ARC), Stable Audio 3, MeanFlow | data | noise (t = 1) | x_0 or x | ε | ε - x_0 |
| Shortcut models (Frans et al. 2024) | noise | data | x_1 | x_0 | x_1 - x_0 |

Two consequences follow:

- The velocity sign flips between the "noise at 0" family and the "data at 0" family. In
  the Lipman family you integrate dx/dt = v from t = 0 to 1. In the SD3 family you
  integrate dx/dt = v from t = 1 down to 0, which Stable Audio Open Small writes as
  dx_t = -v_θ dt.
- The symbol x_0 means data in DDPM papers and noise in flow-matching papers. Lesson
  text must restate the convention at the top of every lab.

The course needs one house convention. We recommend the Lipman convention (t = 0 noise,
t = 1 data) for flow-matching lessons, because Matcha-TTS, Voicebox, E2 TTS, F5-TTS, and
CosyVoice all use it, and the DDPM convention for diffusion lessons, with an explicit
conversion lab between them.

Classifier-free guidance (CFG) also has two conventions. The following table shows how
common papers write the guidance scale:

| Convention | Formula | No guidance at | Used by |
|---|---|---|---|
| w-convention | (1 + w) · f_cond - w · f_uncond | w = 0 | Ho and Salimans 2022, Voicebox (α), F5-TTS (α, written as f_c + α(f_c - f_u)), CosyVoice 2 (β) |
| s-convention | f_uncond + s · (f_cond - f_uncond) | s = 1 | Tango (w), AudioLDM, diffusers, stable-audio-tools |

The two are related by s = 1 + w. F5-TTS's default α = 2 equals s = 3. CosyVoice 2's
β = 0.7 equals s = 1.7. Mixing these up silently changes guidance strength by one unit.

## Generative modelling recap for audio

### Intuition

A generative model learns p(x), or p(x | c) for a condition c such as text, a mel
spectrogram, or a speaker prompt, so that you can sample new audio. Audio is hard because
a waveform is a very long, high-rate signal with structure at every time scale: sample
periodicity (pitch, about 2-20 ms), phonemes (about 50-200 ms), words and prosody
(seconds), and musical form (minutes).

### Model families

- **Likelihood-based, exact**: autoregressive models (WaveNet, SampleRNN) and normalizing
  flows (WaveGlow, Glow-TTS). You can evaluate log p(x) exactly.
- **Likelihood-based, bound**: VAEs and diffusion models optimize an evidence lower bound
  (ELBO). DDPM's simplified ε-loss is a reweighted ELBO.
- **Implicit**: GANs define a sampler but no density. You train them with a
  discriminator.
- **ODE or SDE based**: score SDEs and flow matching define a continuous-time transport
  from noise to data. The probability-flow ODE also gives exact likelihoods through the
  instantaneous change of variables, but audio systems rarely compute them.

### Why raw waveform is hard

- **Sample rate**: 24 kHz audio has 240,000 samples in 10 s. A 44.1 kHz stereo track of
  3 minutes has about 15.9 million numbers.
- **Long-range dependence**: phase must stay coherent over thousands of samples, and a
  single-sample error is audible as a click.
- **Amplitude scale**: speech waveforms normalized to [-1, 1] typically have a standard
  deviation well below 0.2. A noise schedule that looks "almost pure noise" for unit-variance
  data can leave substantial signal-to-noise ratio (SNR) for waveforms. See the DiffWave
  gotcha later in this document.
- **Perception**: humans are sensitive to phase artifacts, metallic buzz from aliasing,
  and over-smoothed spectra. L2 losses on waveforms produce muffled audio, which is why
  vocoders use adversarial and multi-resolution spectral losses.

### Domains

The following table compares the three domains in which audio generators operate:

| Domain | Typical rate | Example | Pros | Cons |
|---|---|---|---|---|
| Waveform | 16-48 kHz | WaveGlow, DiffWave, WaveGrad, PeriodWave | No vocoder needed | Very long sequences, slow iterative sampling |
| Spectrogram (mel) | 50-100 frames per second, 80-128 bins | Grad-TTS, Matcha-TTS, Voicebox, F5-TTS (100-bin mel, 24 kHz, hop 256, about 94 frames per second) | Short sequences, perceptually shaped | Needs a vocoder (HiFi-GAN, Vocos, BigVGAN) |
| Latent (VAE or codec) | 7.5-86 Hz | NaturalSpeech 2, AudioLDM (mel VAE), Stable Audio Open (64-channel, 21.5 Hz), ACE-Step 1.5 (64-dim, 25 Hz), VibeVoice (7.5 Hz) | Shortest sequences, enables minute-long generation | Autoencoder bounds quality; latents need regularization |

A two-stage pipeline (acoustic model to mel, then vocoder) remains the norm for speech.
Latent diffusion over a VAE dominates sound effects and music.

### Gotchas

- Mel spectrograms are log-magnitude and have a large offset (for example, values
  around -11 for silence with natural log and a 1e-5 floor). Normalize them per channel,
  or the Gaussian prior is mismatched in scale.
- Frame-rate arithmetic: frames per second = sample_rate / hop_length. For F5-TTS,
  24000 / 256 = 93.75 frames per second.

## Normalizing flows

### Intuition

A normalizing flow is an invertible neural network f that maps data x to a latent z with
a simple density. Because f is invertible, you can compute the exact density of x from
the density of z and the change in volume, which is the Jacobian determinant.

### Change of variables

Let z = f(x) with f invertible and differentiable, and let p_Z be a standard Gaussian.
Then:

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|.
$$

For a composition f = f_K ∘ ... ∘ f_1 with h_0 = x and h_k = f_k(h_{k-1}), the log
determinants add:

$$
\log p_X(x) = \log p_Z(h_K) + \sum_{k=1}^{K} \log \left| \det \frac{\partial h_k}{\partial h_{k-1}} \right|.
$$

Sampling runs the inverse: draw z ~ p_Z, then compute x = f^{-1}(z).

Direction gotcha: papers name the directions differently. WaveGlow writes the generative
direction as x = f_0 ∘ ... ∘ f_k(z) and computes the loss with f^{-1}(x). Glow-TTS writes
the decoder as f_dec: z → x and its likelihood as
log P_X(x | c) = log P_Z(z | c) + log |det ∂f_dec^{-1}(x) / ∂x|. The sign of the
log-determinant term is always "plus the log-determinant of the data-to-latent map."

### Affine coupling (RealNVP, Glow)

Split the input into two halves, x = (x_a, x_b). The coupling layer is:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp(s(x_a)) + t(x_a),
$$

where s and t are arbitrary networks (they don't need to be invertible). The Jacobian is
triangular, so:

$$
\log \left| \det \frac{\partial y}{\partial x} \right| = \sum_i s(x_a)_i .
$$

The inverse is x_a = y_a and x_b = (y_b - t(y_a)) ⊙ exp(-s(y_a)). WaveGlow parameterizes
the scale directly as s and writes the log-determinant as log |s|. In practice, predict
log s and exponentiate, or use a sigmoid-based scale, to keep the scale positive.

An additive (volume-preserving) coupling sets s = 0, so the log-determinant is 0. VITS
uses this mean-only variant: "we design the normalizing flow to be a volume-preserving
transformation with the Jacobian determinant of one."

### Glow components

- **ActNorm**: y = s ⊙ x + b per channel, initialized from the first batch so outputs
  have zero mean and unit variance. For a tensor with T time steps and C channels, the
  log-determinant is T · Σ_c log |s_c|.
- **Invertible 1x1 convolution**: y_t = W x_t for every time step t, with W a C × C
  matrix. The log-determinant is T · log |det W|. Glow's LU parameterization
  W = P L (U + diag(s)) reduces the cost to Σ log |s|.
- **Squeeze**: reshape to trade time resolution for channels. WaveGlow groups 8 audio
  samples into an 8-dimensional vector. Glow-TTS halves the time steps and doubles the
  channels. Squeeze is a permutation, so its log-determinant is 0.
- **Multi-scale early outputs**: WaveGlow outputs 2 channels to the loss after every 4
  coupling layers.

### WaveGlow

- 12 coupling layers and 12 invertible 1x1 convolutions; each coupling network (WN) has
  8 layers of dilated convolutions with 512 residual channels and 256 skip channels.
- The mel spectrogram conditions the WN layers (added before the gated tanh).
- Loss (negative of Eq. 12 in the paper):

$$
\log p_\theta(x) = -\frac{z(x)^\top z(x)}{2\sigma^2} + \sum_{j} \log s_j(x, \text{mel}) + \sum_k \log |\det W_k| .
$$

- Training uses σ = √0.5; inference samples z with standard deviation 0.6 (a
  temperature that trades diversity for quality).
- DiffWave's comparison table lists WaveGlow at 87.88M parameters.

### Flows in TTS

- **Glow-TTS**: a text encoder predicts the prior mean μ and standard deviation σ for each
  token. Monotonic alignment search (MAS) finds the most likely monotonic alignment
  between tokens and frames under the flow likelihood. The decoder is a stack of flow
  blocks (ActNorm, invertible 1x1 convolution, affine coupling) with squeeze. The best
  MOS came from a prior temperature of 0.333.
- **VITS**: a conditional VAE with a flow prior. The flow f_θ maps the posterior latent z
  to the text-conditioned Gaussian, and the prior density is
  p_θ(z | c) = N(f_θ(z); μ_θ(c), σ_θ(c)) |det ∂f_θ(z) / ∂z|. With a volume-preserving
  flow the determinant is 1. The total loss is
  L_vae = L_recon + L_kl + L_dur + L_adv(G) + L_fm(G). The decoder is the HiFi-GAN V1
  generator, so VITS is also a GAN.

### Worked 2-D example

Let s(x_1) = 0.5 x_1 and t(x_1) = x_1, and take x = (1, 2):

- y_1 = 1.
- y_2 = 2 · exp(0.5) + 1 = 4.297443.
- log |det J| = s(x_1) = 0.5.
- log N(y; 0, I) = -log(2π) - (1² + 4.297443²) / 2 = -11.571883.
- log p_X(x) = -11.571883 + 0.5 = -11.071883.

Inverse check: x_2 = (4.297443 - 1) · exp(-0.5) = 2.0.

### Common mistakes

- Subtracting the log-determinant instead of adding it, or adding the log-determinant of
  the latent-to-data map instead of the data-to-latent map.
- Forgetting the T multiplier on ActNorm and 1x1-convolution log-determinants.
- Using `torch.det` on a large W every step; use `torch.linalg.slogdet` or the LU form.
- Reporting the loss in nats per utterance instead of nats (or bits) per dimension, which
  makes runs with different lengths incomparable.
- For discrete data (16-bit audio), forgetting dequantization noise, which lets the
  likelihood blow up.

### Lab ideas

- **CPU-60s**: implement an affine coupling layer with a check that
  `inverse(forward(x)) == x` to 1e-5 and that the analytic log-determinant matches
  `torch.autograd.functional.jacobian` followed by `slogdet` on random 2-D and 4-D
  inputs. Autocheck the worked example value -11.071883.
- **CPU-60s**: train a RealNVP with 4-6 coupling layers on the two-moons or 8-Gaussians
  dataset for about 2,000 steps; check that the mean test negative log-likelihood is
  below a threshold and that samples match data moments.
- **CPU-60s**: implement ActNorm and a LU-parameterized invertible 1x1 convolution on a
  (B, C, T) tensor; check log-determinants against brute force.

## GANs for audio

### Intuition

A GAN trains a generator G to fool a discriminator D that tries to tell real audio from
generated audio. For vocoders, the generator gets a strong conditioning signal (a mel
spectrogram), so most of the work is done by reconstruction losses, and the adversarial
term mainly removes over-smoothing and phase artifacts.

### Losses

Least-squares GAN (LSGAN), as used by HiFi-GAN:

$$
\mathcal{L}_{\text{adv}}(D; G) = \mathbb{E}_{(x,s)}\left[(D(x) - 1)^2 + D(G(s))^2\right],
\qquad
\mathcal{L}_{\text{adv}}(G; D) = \mathbb{E}_{s}\left[(D(G(s)) - 1)^2\right].
$$

Hinge loss, as used by MelGAN, EnCodec, and Vocos:

$$
\mathcal{L}_D = \frac{1}{K}\sum_k \max(0, 1 - D_k(x)) + \max(0, 1 + D_k(\hat{x})).
$$

MelGAN uses the generator loss -Σ_k D_k(G(s, z)). EnCodec and Vocos use
(1/K) Σ_k max(0, 1 - D_k(x̂)). (The MelGAN PDF prints `min(0, ...)` in the discriminator
loss; the intended function is the standard hinge with `max`.)

Feature matching (HiFi-GAN Eq. 4):

$$
\mathcal{L}_{FM}(G; D) = \mathbb{E}_{(x,s)}\left[\sum_{i=1}^{T} \frac{1}{N_i}\left\| D^i(x) - D^i(G(s)) \right\|_1\right],
$$

where D^i are the features of layer i with N_i elements. EnCodec uses a relative version
that divides each term by mean(‖D_k^l(x)‖_1).

HiFi-GAN totals:

$$
\mathcal{L}_G = \sum_{k=1}^{K}\left[\mathcal{L}_{\text{adv}}(G; D_k) + \lambda_{fm}\mathcal{L}_{FM}(G; D_k)\right] + \lambda_{mel}\mathcal{L}_{Mel}(G),
\quad \lambda_{fm} = 2,\ \lambda_{mel} = 45,
$$

with L_Mel the L1 distance between log-mel spectrograms.

### Discriminators

The following table summarizes the discriminators used by common vocoders and codecs:

| Discriminator | Input | Key settings | Used in |
|---|---|---|---|
| Multi-scale (MSD) | Raw waveform at 1x, 2x, 4x average-pooled | 3 sub-discriminators, strided grouped 1-D convs | MelGAN, HiFi-GAN |
| Multi-period (MPD) | Waveform reshaped to (T/p, p), 2-D convs with kernel width 1 | Periods [2, 3, 5, 7, 11] | HiFi-GAN, BigVGAN, DAC, Vocos, VITS |
| Multi-resolution (MRD) | Magnitude spectrograms at several STFT resolutions | 3 resolutions | UnivNet, BigVGAN, Vocos |
| Multi-scale STFT (MS-STFTD) | Complex STFT, real and imaginary parts concatenated | 5 window lengths [2048, 1024, 512, 256, 128], hop = window / 4 | EnCodec |
| Multi-band multi-scale STFT | Complex STFT split into bands [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] | Windows [2048, 1024, 512], hop = window / 4 | DAC |
| Multi-scale sub-band CQT | Constant-Q transform | Several scales | BigVGAN-v2 |

DAC also uses a multi-scale mel loss with windows [32, 64, 128, 256, 512, 1024, 2048] and
[5, 10, 20, 40, 80, 160, 320] mel bins.

### Generator ideas worth teaching

- **Snake activation** (BigVGAN, DAC): x + (1/α) sin²(αx), a periodic inductive bias.
  BigVGAN also applies anti-aliased up- and down-sampling around each Snake (the AMP
  module). BigVGAN scales to 112M parameters; BigVGAN-base has 14M.
- **Vocos**: predicts STFT magnitude and phase with ConvNeXt blocks and reconstructs with
  the inverse STFT instead of transposed convolutions. The paper reports it about 13 times
  faster than HiFi-GAN and nearly 70 times faster than BigVGAN.

### Training instabilities and tricks

- The discriminator overpowers the generator easily. EnCodec updates the discriminator
  with probability 2/3 at 24 kHz and 0.5 at 48 kHz.
- Warm up with reconstruction losses only, then add the adversarial loss.
- Weight normalization in discriminators (HiFi-GAN, MelGAN, EnCodec); spectral
  normalization is used in the first MSD scale of HiFi-GAN.
- BigVGAN reports that image-domain regularizers such as spectral normalization
  introduce phase-mismatch artifacts in audio. At 112M parameters, the HiFi-GAN learning
  rate of 2e-4 caused early collapse (discriminator losses go to zero), so BigVGAN halves
  it to 1e-4, doubles the batch from 16 to 32, and clips the global gradient norm to 10³.
- EnCodec's loss balancer normalizes each loss's gradient so a weight means "fraction of
  total gradient," which decouples weights from loss scale.

### Common mistakes

- Computing the mel loss on a different mel configuration than the one used to condition
  the generator.
- Forgetting to detach G(s) in the discriminator step, or to zero discriminator gradients
  before the generator step.
- Using BCE-with-logits losses and then plugging in LSGAN targets.
- Reshaping for MPD with `view` when T isn't divisible by p; pad with reflection first.

### Lab ideas

- **CPU-60s**: implement LSGAN and hinge losses, feature matching, and the MPD reshape.
  Autocheck output shapes for periods [2, 3, 5, 7, 11] on a 1-second 16 kHz input, and
  check loss values on fixed tensors.
- **CPU-60s**: implement a multi-resolution STFT loss (spectral convergence plus log-magnitude
  L1) and check that it is zero for identical signals and increases with added noise.
- **GPU-L4, about 15-20 minutes, under 8 GB**: train a tiny HiFi-GAN-style generator on
  LJSpeech mel segments for a few thousand steps and report mel L1 on held-out clips.
  Autocheck that mel L1 falls below a threshold. Quality will still be poor; the lab is
  about losses and discriminators, not quality.

## Diffusion models: DDPM, DDIM, score SDEs, and guidance

### Intuition

A diffusion model fixes a forward process that gradually adds Gaussian noise to data
until nothing is left, then trains a network to undo one small noising step at a time.
Because each forward step is Gaussian, you can jump to any noise level in closed form,
so training is a simple regression: "given a noisy input and its noise level, predict the
noise" (or the clean signal, or a mix called v).

### DDPM forward process

Convention for this section: x_0 is data, t ∈ {1, ..., T}, and x_T is approximately pure
noise (Ho et al. 2020).

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I\right),
\quad \alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s .
$$

Closed form for any t:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I\right),
\qquad x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon,\ \epsilon \sim \mathcal{N}(0, I).
$$

Posterior (tractable because x_0 is known during training):

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I),\quad
\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t,\quad
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t .
$$

### Training loss and reverse step

The simplified loss (Ho et al. 2020, Eq. 14; DiffWave Eq. 7):

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left\| \epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, t\right) \right\|^2 .
$$

The exact ancestral sampling step, for t = T, ..., 1:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z,
\quad z \sim \mathcal{N}(0, I) \text{ if } t > 1, \text{ else } z = 0,
$$

with σ_t² = β_t or σ_t² = β̃_t (both work; β̃_t is the posterior variance).

### Noise schedules

- **Linear** (DDPM): β_t linear from 1e-4 to 0.02 with T = 1000, which gives
  ᾱ_T ≈ 4.04e-5 (computed during research).
- **Cosine** (Nichol and Dhariwal 2021):

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)},\quad f(t) = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right),\quad s = 0.008,\quad \beta_t = \min\!\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999\right).
$$

- **Continuous VP** (score SDE and EDM's VP baseline): β(t) = β_min + t(β_max - β_min)
  with β_min = 0.1 and β_max = 20 for t ∈ [0, 1].

### Parameterizations and their relationships

Write the VP marginal with α_t = √ᾱ_t and σ_t = √(1 - ᾱ_t), so x_t = α_t x_0 + σ_t ε and
α_t² + σ_t² = 1. Define SNR(t) = α_t² / σ_t². The three common targets are:

- ε-prediction: ε̂.
- x_0-prediction: x̂_0 = (x_t - σ_t ε̂) / α_t.
- v-prediction (Salimans and Ho 2022): v = α_t ε - σ_t x_0.

Conversions from v:

$$
\hat{x}_0 = \alpha_t x_t - \sigma_t \hat{v}, \qquad \hat{\epsilon} = \sigma_t x_t + \alpha_t \hat{v}.
$$

The losses are the same regression with different weights:

$$
\|\epsilon - \hat{\epsilon}\|^2 = \text{SNR}(t)\,\|x_0 - \hat{x}_0\|^2, \qquad
\|v - \hat{v}\|^2 = (1 + \text{SNR}(t))\,\|x_0 - \hat{x}_0\|^2 .
$$

Score connection. The conditional score of the Gaussian kernel is
∇_{x_t} log q(x_t | x_0) = -ε / σ_t, so the learned score is

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}},
$$

and Tweedie's formula gives E[x_0 | x_t] = (x_t + σ_t² ∇ log p_t(x_t)) / α_t.

SNR weighting. Min-SNR-γ (Hang et al. 2023) weights the x_0-loss by min(SNR, γ) with
γ = 5. In ε-prediction this becomes the weight min(SNR, γ) / SNR = min(γ / SNR, 1), and in
v-prediction it becomes min(SNR, γ) / (SNR + 1).

### Worked 4-step example

Take T = 4 and β = (0.1, 0.2, 0.3, 0.4). All values were computed during research:

| t | β_t | α_t | ᾱ_t | √ᾱ_t | √(1 - ᾱ_t) | SNR |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.9 | 0.9 | 0.948683 | 0.316228 | 9.0 |
| 2 | 0.2 | 0.8 | 0.72 | 0.848528 | 0.529150 | 2.571429 |
| 3 | 0.3 | 0.7 | 0.504 | 0.709930 | 0.704273 | 1.016129 |
| 4 | 0.4 | 0.6 | 0.3024 | 0.549909 | 0.835225 | 0.433486 |

With x_0 = 1.0, ε = 0.5, and t = 2:

- x_2 = 0.848528 · 1 + 0.529150 · 0.5 = 1.113103.
- Posterior mean with a perfect ε̂ = 0.5: μ̃ = (1.113103 - 0.2 / 0.529150 · 0.5) / √0.8
  = 1.033199. The x_0-form of μ̃ gives the same number.
- β̃_2 = (0.1 / 0.28) · 0.2 = 0.071429, so σ_2 = 0.267261.
- DDIM (η = 0) step to t = 1: x_1 = √0.9 · 1 + √0.1 · 0.5 = 1.106797.
- v = 0.848528 · 0.5 - 0.529150 · 1 = -0.104886. Check: 0.848528 · 1.113103 - 0.529150 ·
  (-0.104886) = 1.0.
- Score = -0.5 / 0.529150 = -0.944911.

The same four-step cosine schedule gives ᾱ = (0.847012, 0.493844, 0.144272, ≈0) and
β = (0.152988, 0.416958, 0.707859, 0.999 after clipping). This shows why the clip matters
at small T.

### DDIM

DDIM (Song, Meng, and Ermon 2021) keeps the same marginals and training, but defines a
non-Markovian reverse process. Warning: the DDIM paper writes α_t for what DDPM calls
ᾱ_t. In DDPM notation, the update from t to an earlier index t' is:

$$
x_{t'} = \sqrt{\bar{\alpha}_{t'}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t'}-\sigma_t^2}\;\hat{\epsilon} + \sigma_t z,
\qquad \hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}},
$$

$$
\sigma_t(\eta) = \eta \sqrt{\frac{1-\bar{\alpha}_{t'}}{1-\bar{\alpha}_t}}\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t'}}} .
$$

η = 0 gives the deterministic DDIM sampler, and η = 1 recovers DDPM ancestral sampling
with σ² = β̃. Because the model is conditioned only on the noise level, you can skip
steps (t' < t - 1) without retraining.

DDIM with η = 0 is a first-order discretization of the probability-flow ODE. It's exact
for data that is a single point. For Gaussian data it converges as steps increase. A
research check with an oracle denoiser for x_0 ~ N(2, 0.5²), the linear T = 1000
schedule, and a fixed standardized latent 1.3 gave final x_0 = 2.015 (1 step), 2.486
(10 steps), 2.614 (50 steps), and 2.648 (1,000 steps) against the exact quantile-map
value 2.65.

### Score SDE view

Song et al. (2021) write the forward process as an SDE with t ∈ [0, T], data at t = 0:

$$
dx = f(x, t)\,dt + g(t)\,dw .
$$

Reverse-time SDE (Anderson 1982):

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{w},
$$

integrated from t = T to 0. The probability-flow ODE has the same marginals:

$$
\frac{dx}{dt} = f(x, t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x).
$$

- **VP SDE**: f = -½ β(t) x, g = √β(t). The kernel is
  N(x_0 e^{-½∫_0^t β}, (1 - e^{-∫_0^t β}) I). DDPM is its discretization.
- **VE SDE**: f = 0, g = √(dσ²(t)/dt) with σ(t) = σ_min (σ_max / σ_min)^t. NCSN (Song
  and Ermon 2019) is its discretization.

### EDM (Karras et al. 2022)

EDM writes the noisy data as x = y + n with n ~ N(0, σ²I), uses σ(t) = t and scaling
s(t) = 1, and wraps the raw network F_θ in a preconditioned denoiser:

$$
D_\theta(x; \sigma) = c_{\text{skip}}(\sigma)\,x + c_{\text{out}}(\sigma)\,F_\theta\!\left(c_{\text{in}}(\sigma)\,x;\ c_{\text{noise}}(\sigma)\right),
$$

$$
c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2},\quad
c_{\text{out}} = \frac{\sigma\,\sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}},\quad
c_{\text{in}} = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}},\quad
c_{\text{noise}} = \tfrac{1}{4}\ln\sigma .
$$

- Training: ln σ ~ N(P_mean = -1.2, P_std² = 1.2²); loss weight
  λ(σ) = (σ² + σ_data²) / (σ σ_data)²; σ_data = 0.5.
- Score: ∇_x log p(x; σ) = (D(x; σ) - x) / σ². The ODE is dx/dσ = (x - D(x; σ)) / σ.
- Sampling schedule with ρ = 7, σ_min = 0.002, and σ_max = 80:

$$
\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)\right)^{\rho}, \quad i = 0, \dots, N-1, \quad \sigma_N = 0 .
$$

  For N = 5: σ = (80, 17.528, 2.515, 0.1698, 0.002), then 0.
- Solver: second-order Heun, with a final Euler step to σ = 0, and optional stochastic
  "churn."

### Classifier-free guidance

Train one network with the condition c replaced by a null token ∅ with probability
p_uncond (Ho and Salimans compare 0.1, 0.2, and 0.5 and find the smaller values work
better; Voicebox and MusicGen use 0.2). At
sampling time (Ho and Salimans, Algorithm 2):

$$
\tilde{\epsilon}_\theta(z, c) = (1 + w)\,\epsilon_\theta(z, c) - w\,\epsilon_\theta(z, \varnothing).
$$

This corresponds to sampling from p(x | c) p(c | x)^w, because its score is
(1 + w) ∇ log p(x | c) - w ∇ log p(x). For flow matching, apply the same combination to
velocities. For Gaussian paths the velocity is an affine function of ε̂ with the same x
term in both branches, so guiding v is identical to guiding ε̂. See
[Conventions you must track](#conventions-you-must-track) for the s = 1 + w conversion.

Gotchas:

- Guidance doubles the number of function evaluations (NFE) unless you batch the
  conditional and unconditional passes together.
- Large guidance over-saturates and reduces diversity. MaskGCT and others add CFG
  rescale (factor 0.75 in MaskGCT).
- The null condition must match training exactly (zeroed embedding, a learned null
  token, or dropped prompt audio). F5-TTS drops the masked speech with rate 0.3, then
  drops speech and text together with rate 0.2.

### Consistency models (Song et al. 2023)

A consistency model learns f_θ(x_t, t) that maps any point on a probability-flow ODE
trajectory to the trajectory's origin, with the boundary condition f_θ(x, ε) = x. The
parameterization is f_θ(x, t) = c_skip(t) x + c_out(t) F_θ(x, t), with the EDM-derived

$$
c_{\text{skip}}(t) = \frac{\sigma_{\text{data}}^2}{(t-\epsilon)^2 + \sigma_{\text{data}}^2},\qquad
c_{\text{out}}(t) = \frac{\sigma_{\text{data}}(t-\epsilon)}{\sqrt{\sigma_{\text{data}}^2 + t^2}} .
$$

Consistency distillation minimizes d(f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}(x̂_{t_n}, t_n)),
where x̂_{t_n} is one ODE-solver step of a teacher from x_{t_{n+1}} and θ⁻ is an EMA of θ.
Consistency training replaces the teacher step with the same-noise sample at t_n.
CoMoSpeech applies consistency distillation to TTS and singing for one-step synthesis.
WaveFM uses a tailored consistency distillation for one-step vocoding.

### Common mistakes

- Broadcasting: `alpha_bar[t]` has shape (B,), but x has shape (B, C, T). Index and
  reshape to (B, 1, 1) before multiplying. A (B,) tensor broadcast against (B, C, T)
  either errors or, worse, broadcasts over the last axis when T == B.
- Off-by-one indexing: DDPM is 1-indexed in the paper and 0-indexed in code.
- Using σ_t² = β_t at t = 1 and still adding noise; the final step must add no noise.
- Mixing DDIM's α (meaning ᾱ) with DDPM's α.
- Feeding the integer step t to a continuous-time model that expects t ∈ [0, 1], or a
  noise level √ᾱ as WaveGrad does.
- Forgetting to clip ᾱ near zero before dividing by √ᾱ in x̂_0.
- Applying CFG to the output after converting to x̂_0 in one branch but not the other.

### Lab ideas

- **CPU-60s**: implement q(x_t | x_0), the posterior mean and variance, and the parameter
  conversions (ε, x_0, v, score). Autocheck against the worked 4-step table and values
  (x_2 = 1.113103, μ̃ = 1.033199, v = -0.104886).
- **CPU-60s**: train a 2-D toy DDPM (8-Gaussians ring, T = 100, MLP) for 3,000 steps in
  about 1.5 s; sample with ancestral DDPM and DDIM (η = 0, 20 steps). Autocheck that the
  sample radius mean is 2.0 ± 0.1 and that more than 90 percent of samples lie within 0.4
  of a mode.
- **CPU-60s**: implement DDIM with an oracle Gaussian denoiser and check convergence to
  the closed-form quantile map (use the numbers in the DDIM section).
- **CPU-60s**: implement the EDM σ schedule and a Heun sampler with an analytic Gaussian
  denoiser D(x; σ) = (σ_data² x + σ² μ) / (σ_data² + σ²) for data N(μ, σ_data²);
  check that the final samples have mean μ and standard deviation σ_data within 2 percent.
- **CPU-60s**: CFG on a 2-class 2-D Gaussian mixture; check that increasing w moves the
  sample mean toward the class center and reduces sample variance.

## Audio diffusion models

### DiffWave (Kong et al. 2021)

- **Intuition**: a non-autoregressive WaveNet-like denoiser turns white noise into a
  waveform in a fixed number of steps, conditioned on a mel spectrogram (vocoder) or on
  nothing (unconditional generation).
- **Architecture**: 30 residual layers, kernel size 3, dilation cycle [1, 2, ..., 512]
  (three cycles of 10), bidirectional dilated convolutions with gated tanh, residual
  channels C = 64 (BASE, 2.64M parameters) or 128 (LARGE, 6.91M parameters). The
  diffusion step goes through a sinusoidal embedding and fully connected layers and is
  added to each layer. The 80-band mel conditioner (FFT 1024, hop 256) is upsampled 256
  times by two transposed 2-D convolutions (stride 16 in time, filter [32, 3], leaky ReLU
  with slope 0.4), then projected by a per-layer 1x1 convolution to 2C channels and added
  as a bias before the gated tanh.
- **Schedules**: β linear in [1e-4, 0.05] for T ≤ 50, and in [1e-4, 0.02] for T = 200.
  The vocoder runs used T ∈ {20, 40, 50, 200}. Training uses the unweighted ε-loss.
- **Fast sampling** (Appendix B): pick a short user-defined variance schedule η_s with
  T_infer = 6, for example {0.0001, 0.001, 0.01, 0.05, 0.2, 0.5} for BASE and
  {0.0001, 0.001, 0.01, 0.05, 0.2, 0.7} for LARGE. Compute γ_s = 1 - η_s and
  γ̄_s = Π γ. Align each sampling step to a fractional training step:

$$
t^{\text{align}}_s = t + \frac{\sqrt{\bar{\alpha}_t} - \sqrt{\bar{\gamma}_s}}{\sqrt{\bar{\alpha}_t} - \sqrt{\bar{\alpha}_{t+1}}}
\quad \text{if } \sqrt{\bar{\gamma}_s} \in [\sqrt{\bar{\alpha}_{t+1}}, \sqrt{\bar{\alpha}_t}],
$$

$$
\mu^{\text{fast}}_\theta(x_s, s) = \frac{1}{\sqrt{\gamma_s}}\left(x_s - \frac{\eta_s}{\sqrt{1-\bar{\gamma}_s}}\,\epsilon_\theta(x_s, t^{\text{align}}_s)\right),
\qquad \sigma^{\text{fast}}_s = \tilde{\eta}_s^{1/2},\ \tilde{\eta}_s = \frac{1-\bar{\gamma}_{s-1}}{1-\bar{\gamma}_s}\eta_s .
$$

- **Results**: LARGE (T = 200) MOS 4.44 against WaveNet 4.43; BASE (Fast, T_infer = 6)
  MOS 4.37 and 5.6 times faster than real time on a V100.
- **Gotcha worth a lesson**: with β linear in [1e-4, 0.05] and T = 50, ᾱ_T ≈ 0.28
  (computed during research), so √ᾱ_T ≈ 0.53. The endpoint is far from unit Gaussian for
  unit-variance data, but waveforms have small amplitude, so the SNR at T is still low.
  The BASE fast schedule ends at γ̄ = 0.376. Students who reuse this schedule on
  normalized mel spectrograms get poor samples.

### WaveGrad (Chen et al. 2021)

- **Intuition**: condition the network on the continuous noise level √ᾱ instead of the
  integer step, so one trained model supports any inference schedule.
- **Training** (Algorithm 1): pick s ~ U{1, ..., S}, draw √ᾱ ~ U(l_{s-1}, l_s) from a
  predefined schedule of levels l, and minimize the L1 loss
  ‖ε - ε_θ(√ᾱ y_0 + √(1 - ᾱ) ε, x, √ᾱ)‖_1.
- **Sampling**: the standard DDPM step. A grid-searched 6-iteration schedule gives
  MOS above 4.4 and a real-time factor of 0.2 on a V100 (WaveGrad Base).
- **Architecture**: upsampling blocks (UBlock) and downsampling blocks from GAN-TTS, with
  feature-wise linear modulation (FiLM) conditioning.

### PriorGrad (Lee et al. 2022)

- **Intuition**: start from a data-dependent Gaussian prior N(μ, Σ) that already matches
  the rough energy or mean of the target, so the network has less to transport.
- **Training** (Algorithm 1): ε ~ N(0, Σ), x_t = √ᾱ_t (x_0 - μ) + √(1 - ᾱ_t) ε, loss
  ‖ε - ε_θ(x_t, c, t)‖²_{Σ⁻¹}. Sampling starts from x_T ~ N(0, Σ), uses Σ-scaled noise, and
  returns x_0 + μ.
- **Vocoder prior**: Σ_c is the frame-level energy of the mel spectrogram, normalized to
  (0, 1], used as a per-sample variance proxy. **Acoustic model prior**: per-phoneme
  80-dimensional diagonal mean and variance, upsampled by duration.
- Gotcha: the loss uses the Σ⁻¹-weighted norm; plain MSE changes the optimum weighting.

### Grad-TTS (Popov et al. 2021)

- **Intuition**: a Glow-TTS-style encoder predicts a frame-aligned mean μ (a blurry mel).
  A score-based decoder then diffuses from N(μ, I) instead of N(0, I), so it only has to
  add detail.
- **Forward SDE** (Eq. 2), data at t = 0, T = 1:

$$
dX_t = \tfrac{1}{2}\Sigma^{-1}(\mu - X_t)\,\beta_t\,dt + \sqrt{\beta_t}\,dW_t ,
$$

  with Σ = I in the paper. Its solution has mean
  (I - e^{-½∫β}) μ + e^{-½∫β} X_0 and, for Σ = I, variance λ_t I with
  λ_t = 1 - e^{-∫_0^t β_s ds}.
- **Reverse SDE and ODE** (Eqs. 8-9):

$$
dX_t = \left(\tfrac{1}{2}\Sigma^{-1}(\mu - X_t) - \nabla\log p_t(X_t)\right)\beta_t\,dt + \sqrt{\beta_t}\,d\tilde{W}_t,
\qquad
dX_t = \tfrac{1}{2}\left(\Sigma^{-1}(\mu - X_t) - \nabla\log p_t(X_t)\right)\beta_t\,dt .
$$

- **Loss** (Eq. 17): L_diff = E_{X_0, t}[λ_t E_ξ ‖s_θ(X_t, μ, t) + ξ / √λ_t‖²] with
  ξ ~ N(0, I). The encoder loss L_enc = -Σ_j log φ(y_j; μ̃_{A(j)}, I) is an MSE between
  aligned μ and the target mel. MAS provides alignment, and a duration predictor is
  trained with MSE on log durations.
- **Hyperparameters**: β_t = β_0 + (β_1 - β_0) t with β_0 = 0.05 and β_1 = 20; training
  on random 2-second mel segments; temperature τ = 1.5 at synthesis (sample the terminal
  condition from N(μ, τ⁻¹ I)). The official code defaults to 50 ODE steps; the paper
  evaluates N ∈ {4, 10, 100, 1000}.
- **Numbers for a lab** (computed during research): ∫_0^t β = 0.10475, 2.51875, and
  10.025 at t = 0.1, 0.5, and 1.0; λ_t = 0.0995, 0.9194, and 0.99996; the coefficient on
  X_0 is 0.949, 0.284, and 0.0067.

### DiffSinger (Liu et al. 2022)

- **Intuition**: "shallow diffusion." A simple decoder trained with L1 produces a blurry
  mel M̃. Diffusing M̃ and the ground truth M to a moderate step k makes them almost
  indistinguishable, so inference can start at step k from the noised M̃ rather than
  from pure noise at step T.
- **Inference** (Algorithm 2): M̃_k = √ᾱ_k M̃ + √(1 - ᾱ_k) ε, then run the standard
  DDPM reverse step from t = k down to 1.
- **Settings**: T = 100, β linear from 1e-4 to 0.06, a boundary predictor chooses k (the
  ablation table reports k = 54). The shallow mechanism gave +0.14 MOS and a 45.1 percent
  speedup. The TTS variant is DiffSpeech.
- **Lab tie-in**: shallow diffusion is also the idea behind SDEdit-style editing and
  AudioLDM's style transfer (start the reverse process from a partially noised source).

### NaturalSpeech 2 (Shen et al. 2023)

- **Intuition**: a neural codec with continuous latents plus a non-autoregressive latent
  diffusion model avoids the long multi-token sequences of codec language models.
- **Formulation**: VP SDE dz_t = -½ β_t z_t dt + √β_t dw_t on codec latents, t ∈ [0, 1],
  data at 0. The WaveNet-based denoiser predicts ẑ_0 directly instead of the score. The
  loss (Eq. 6) combines the data loss ‖ẑ_0 - z_0‖², a score loss using the score derived
  from ẑ_0, and an RVQ cross-entropy regularizer L_ce-rvq.
- **Prompting**: a speech prompt feeds the duration and pitch predictors and the
  diffusion model through attention.
- **Scale**: 400M parameters, 44K hours, 5K speakers. The codec uses RVQ with many
  quantizers so that continuous vectors can be stored as indices.

### NaturalSpeech 3 (Ju et al. 2024)

- **Intuition**: factorize speech into subspaces (content, prosody, acoustic details, and
  timbre) with a codec (FACodec) that uses factorized vector quantization, then generate
  each attribute with its own diffusion model in a fixed order.
- **Contrast with NaturalSpeech 2**: NaturalSpeech 3 uses **discrete** (masked) diffusion
  on tokens, not Gaussian diffusion on continuous latents. Order: duration, then prosody,
  then content, then acoustic details; timbre isn't generated but comes from the prompt.
- **Inference**: 4 iterations per diffusion process, CFG scale 1.0 except for duration,
  60 forward passes total.
- **Scale**: 500M parameters on 60K hours of LibriLight; scaled to 1B parameters and 200K
  hours in the paper's scaling study.

### AudioLDM and AudioLDM 2

- **AudioLDM** (Liu et al. 2023): a VAE compresses the mel spectrogram (compression level
  r = 4 by default), and a latent diffusion model generates the latent. The key trick is
  to train the latent diffusion model on **CLAP audio embeddings** and sample with **CLAP
  text embeddings**, which share a joint space, so training needs no captions. Audio is
  16 kHz mono, padded to 10 s. HiFi-GAN converts the mel to a waveform. AudioLDM-S has
  181M parameters and AudioLDM-L 739M. The paper studies DDIM steps (100 is close to 200)
  and CFG scale, and shows zero-shot inpainting, super-resolution (8 kHz to 16 kHz), and
  style transfer by starting from a partially noised latent.
- **AudioLDM 2** (Liu et al. 2024): introduces the "language of audio" (LOA), which is
  AudioMAE features of the target audio. A GPT-2 model translates conditions (CLAP and
  FLAN-T5 text embeddings, or phonemes) into LOA, and a latent diffusion model
  (Transformer-UNet) generates the VAE latent conditioned on LOA. One framework covers
  speech, music, and sound effects. The diffusers model card lists `audioldm2` (350M
  UNet, 1.1B total), `audioldm2-large` (750M, 1.5B), and `audioldm2-music` (350M, 1.1B),
  CC-BY-NC-SA-4.0, 16 kHz, 200 recommended inference steps.

### Make-An-Audio (Huang et al. 2023)

A spectrogram autoencoder plus latent diffusion with CLAP text conditioning. Its
distinctive idea is pseudo-prompt enhancement ("distill-then-reprogram"): generate
captions for unlabeled audio and compose concepts to create large numbers of text-audio
pairs. It also demonstrates X-to-audio conditioning.

### Tango and Tango 2

- **Tango** (Ghosal et al. 2023): a frozen instruction-tuned FLAN-T5-Large text encoder,
  the AudioLDM VAE and vocoder, and a Stable Diffusion U-Net latent diffusion model
  (866M parameters, 8 latent channels, cross-attention dimension 1024), trained only on
  AudioCaps (about 63 times less data than AudioLDM). It mixes training clips by pressure
  level for augmentation. Its CFG formula is ε̂ = w ε(z, τ) + (1 - w) ε(z), which is the
  s-convention.
- **Tango 2** (Majumder et al. 2024): fine-tunes Tango-full-ft with diffusion-DPO on
  Audio-Alpaca, a synthetic preference dataset of about 15K pairs whose "loser" outputs
  miss concepts or order them wrongly. Hugging Face: `declare-lab/tango2`,
  CC-BY-NC-SA-4.0, 16 kHz, 100 steps by default and 200 for best quality.

### Stable Audio and Stable Audio Open

- **Stable Audio 1.0** (Evans et al. 2024, "Fast timing-conditioned latent audio
  diffusion"): a VAE on 44.1 kHz stereo waveforms, a 907M-parameter U-Net diffusion model,
  CLAP text features from the next-to-last layer, and **timing conditioning**: learned
  per-second embeddings for `seconds_start` (where the training chunk starts in the file)
  and `seconds_total` (the file's total length). At inference, `seconds_total` sets the
  length of non-silent audio in a fixed window (95.1 s). Training uses the v-objective with
  a cosine schedule and continuous timesteps, with 10 percent condition dropout for CFG.
- **Stable Audio Open 1.0** (Evans et al. 2024): a 156M-parameter VAE (64-channel latent
  at 21.5 Hz for 44.1 kHz stereo), a 109M-parameter T5-base text encoder, and a
  1,057M-parameter DiT that receives text and timing through cross-attention and timing
  and timestep through prepended tokens. It generates up to 47 s. It uses the v-objective;
  inference uses DPM-Solver++ for 100 steps with CFG scale 7.0. It was trained only on
  Creative Commons audio. Hugging Face: `stabilityai/stable-audio-open-1.0`, gated,
  Stability AI Community License.
- **Stable Audio Open Small** (Novack et al. 2025, "Fast text-to-audio generation with
  adversarial post-training"): DiT reduced from 1.06B to 0.34B parameters (width 1536 to
  1024, 24 to 16 layers, QK-LayerNorm, no `seconds_start`), up to 11.89 s of 44.1 kHz
  stereo. The pre-training is rectified flow with x_t = (1 - t) x_0 + t ε (t = 0 is data)
  and target v = ε - x_0. ARC post-training (adversarial relativistic plus contrastive)
  turns it into a few-step generator without distillation or CFG, sampled with
  "ping-pong" sampling: denoise to x̂_0, re-noise to a lower level with fresh noise, and
  repeat. It generates about 12 s in about 75 ms on an H100 and about 7 s on a phone.
  Hugging Face: `stabilityai/stable-audio-open-small`, gated, Stability AI Community
  License, 8 steps with the `pingpong` sampler in `stable-audio-tools`. The model card
  lists 0.5B total parameters.
- **Stable Audio 2.5** (September 2025, enterprise-only): company announcement and press
  coverage describe ARC post-training, 8 inference steps against about 50 for Stable Audio
  2.0, and under 2 s on a GPU for tracks up to 3 minutes. No paper.
- **Stable Audio 3** (technical report arXiv 2605.17991, May 2026): small (459M DiT,
  up to 2 minutes, separate music and SFX variants), medium (1.4B DiT, up to 6 min 20 s),
  and large (2.7B DiT, enterprise only). A semantic-acoustic autoencoder with 4096 times
  downsampling. The training pipeline is flow-matching pre-training (minibatch
  optimal-transport coupling, truncated logit-normal timesteps), ODE warmup distillation,
  and adversarial post-training; inference uses 8 ping-pong steps. The model recovers
  x̂_0 = x_t - t v_θ(x_t, t, c) (t = 0 is data). Small and medium weights are on Hugging
  Face (`stabilityai/stable-audio-3-medium`, `stabilityai/stable-audio-3-small-sfx`,
  gated, Stability AI Community License).

### Speech enhancement with diffusion: SGMSE+ (Richter et al. 2023)

- **Intuition**: instead of diffusing clean speech toward pure noise, diffuse it toward
  the **noisy recording** plus Gaussian noise, so the reverse process starts from the
  observed noisy speech.
- **Forward SDE** on complex STFTs (an Ornstein-Uhlenbeck process with variance
  exploding noise):

$$
dx_t = \gamma(y - x_t)\,dt + g(t)\,dw,\qquad
g(t) = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t}\sqrt{2\log\frac{\sigma_{\max}}{\sigma_{\min}}},
$$

  with stiffness γ = 1.5, σ_min = 0.05, and σ_max = 0.5. The reverse SDE is
  dx_t = [γ(y - x_t) - g(t)² ∇ log p_t(x_t | y)] dt + g(t) dw̄.
- **Representation**: amplitude-compressed complex STFT c̃ = β |c|^α e^{i∠c} with α = 0.5
  and β = 0.15, real and imaginary parts as channels, an NCSN++ score network, and a
  minimum time t_ε to avoid division by σ(t)² ≈ 0.
- **Sampling**: predictor-corrector with 30 reverse steps and one annealed Langevin
  corrector step (r = 0.33), which gives 60 NFE.

### Diffusion vocoders compared with GAN vocoders

The following table summarizes the trade-off as of the research date:

| Aspect | GAN vocoders (HiFi-GAN, BigVGAN, Vocos) | Diffusion and flow vocoders (DiffWave, WaveGrad, PriorGrad, PeriodWave, WaveFM) |
|---|---|---|
| Inference | One forward pass | 6-50 steps, or 1 step after distillation (WaveFM) |
| Training | Unstable; needs discriminators and loss balancing | Stable regression loss |
| Quality | High; BigVGAN-v2 is a strong universal baseline | Competitive; early models lost high-frequency detail, which PeriodWave and FreGrad address |
| Practice | Default in production TTS (F5-TTS uses Vocos; IndexTTS2 uses BigVGAN-v2; CosyVoice uses HiFT) | Research and one-step distilled variants |

### Lab ideas

- **CPU-60s**: implement the DiffWave fast-sampling alignment t_s^align for a given
  training schedule and user schedule; autocheck values against a reference
  implementation, and verify γ̄ for the BASE schedule ends at 0.37579.
- **CPU-60s**: implement the Grad-TTS forward mean and variance and check them against
  the numbers in the Grad-TTS section; then check by Monte Carlo that
  X_1 ≈ N(μ, I) for any X_0.
- **CPU-60s**: shallow diffusion on 1-D toy signals: train a denoiser for sine waves,
  start the reverse process at k from a blurred sine, and check that MSE to the target is
  lower than starting from pure noise at the same NFE.
- **GPU-L4, about 10 minutes, under 8 GB**: train a small DiffWave (C = 64, 10-20
  layers, T = 50) on synthetic harmonic tones conditioned on their mel spectrograms.
  Autocheck multi-resolution STFT distance and pitch error of fast-sampled outputs.
- **GPU-A100, about 2-3 minutes, about 12 GB**: run Stable Audio Open Small with 8
  ping-pong steps and with more steps; compute CLAP score for each prompt and check that
  it exceeds a threshold. Requires accepting the gated license on Hugging Face.

## Flow matching

### Intuition

A continuous normalizing flow (CNF) moves samples from noise to data by integrating an
ODE dx/dt = v_θ(x, t). Flow matching trains v_θ without simulating the ODE: pick a simple
path between a noise sample and a data sample (a straight line, for example), and regress
the network onto that path's velocity at a random time. Averaged over all pairs, the
network learns the marginal velocity field that transports the whole noise distribution
onto the data distribution.

Convention for this section (Lipman et al. 2022): p_0 is noise N(0, I), p_1 ≈ q is data,
x_0 is noise, and x_1 is data.

### Continuous normalizing flows

A vector field v_t generates a flow φ_t through dφ_t(x)/dt = v_t(φ_t(x)) with φ_0(x) = x,
and pushes p_0 to p_t = [φ_t]_* p_0. The instantaneous change of variables (Chen et al.
2018) is d log p_t(x_t) / dt = -∇ · v_t(x_t), so exact likelihoods need a trace
estimator (FFJORD uses Hutchinson's estimator). Classic CNFs trained by maximum likelihood
simulate the ODE during training, which is slow. Flow matching removes that cost.

### Objectives

Flow matching (FM):

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim U[0,1],\, x \sim p_t}\left\| v_\theta(x, t) - u_t(x) \right\|^2 .
$$

u_t is intractable, so conditional flow matching (CFM) uses a per-sample path
p_t(x | x_1) and its field u_t(x | x_1):

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p_t(x \mid x_1)}\left\| v_\theta(x, t) - u_t(x \mid x_1) \right\|^2 ,
$$

which has the same gradient as L_FM (Lipman et al., Theorem 2).

Gaussian conditional paths p_t(x | x_1) = N(x | μ_t(x_1), σ_t(x_1)² I) with flow
ψ_t(x) = σ_t(x_1) x + μ_t(x_1) have the unique generating field (Theorem 3):

$$
u_t(x \mid x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}\left(x - \mu_t(x_1)\right) + \mu_t'(x_1).
$$

### The optimal-transport path (verified against the paper)

Lipman et al., Eqs. 20-23, with σ_min small (Matcha-TTS uses 1e-4; Voicebox cites
"typically 10⁻⁵"):

$$
\mu_t(x) = t\,x_1, \qquad \sigma_t(x) = 1 - (1-\sigma_{\min})t,
$$

$$
u_t(x \mid x_1) = \frac{x_1 - (1-\sigma_{\min})\,x}{1 - (1-\sigma_{\min})\,t}, \qquad
\psi_t(x_0) = \left(1 - (1-\sigma_{\min})t\right) x_0 + t\,x_1 ,
$$

$$
\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t,\, q(x_1),\, p(x_0)}\left\| v_\theta(\psi_t(x_0), t) - \left(x_1 - (1-\sigma_{\min})\,x_0\right) \right\|^2 .
$$

The target x_1 - (1 - σ_min) x_0 is constant along each conditional path. Setting
σ_min = 0 gives the rectified-flow target x_1 - x_0 (F5-TTS and CosyVoice 2 write it this
way).

Diffusion paths are special cases. The reversed VP path is μ_t = α_{1-t} x_1 and
σ_t = √(1 - α_{1-t}²) with α_t = e^{-½ T(t)}, T(t) = ∫_0^t β(s) ds (Lipman Eq. 18). Its
conditional field coincides with the probability-flow ODE field of the VP SDE.

### Worked numeric example

With σ_min = 0, x_0 = -1 (noise), x_1 = 2 (data), and t = 0.25:
x_t = 0.75 · (-1) + 0.25 · 2 = -0.25, and the target is u = 2 - (-1) = 3. With
σ_min = 1e-4, x_t = (1 - 0.9999 · 0.25)(-1) + 0.5 = -0.250025 and u = 2 + 0.9999 =
2.9999.

Closed-form marginal field for a lab. If data is N(μ, s²) in 1-D and the coupling is
independent with σ_min = 0, then x_t ~ N(tμ, (1 - t)² + t²s²) and the marginal field is:

$$
v(x, t) = \mu + \frac{t s^2 - (1-t)}{(1-t)^2 + t^2 s^2}\,(x - t\mu),
$$

and the exact ODE solution maps x_0 to x_1 = μ + s x_0. For μ = 2, s = 0.5, and
x_0 = 1.3 (exact answer 2.65), a research check gave Euler results of 2.0 (1 step),
2.26 (2), 2.443 (4), 2.539 (8), and 2.621 (32), and midpoint results of 2.520 (1 step,
2 NFE), 2.643 (2 steps), and 2.6478 (4 steps). One Euler step returns exactly the data
mean μ, which is why one-step sampling of an undistilled flow is blurry.

### Converting between velocity, noise, data, and score

For the linear path x_t = t x_1 + (1 - t) x_0 (σ_min = 0) and a velocity prediction v̂:

$$
\hat{x}_1 = x_t + (1-t)\,\hat{v}, \qquad \hat{x}_0 = x_t - t\,\hat{v}, \qquad
\nabla \log p_t(x_t) \approx -\frac{\hat{x}_0}{1-t} = -\frac{x_t - t\hat{v}}{1-t}.
$$

Relation to diffusion. Any Gaussian path x_t = a_t x_1 + b_t x_0 is a diffusion process
with a (generally non-variance-preserving) schedule, and velocity prediction is a
reweighted ε- or x-prediction. Kingma and Gao (2023) show that the objectives differ only
by a weighting over log-SNR, and F5-TTS states that the OT-CFM loss is equivalent to
v-prediction with a cosine schedule under that view. The "Diffusion meets flow matching"
note (Gao et al. 2024) shows that the DDIM sampler equals Euler integration of the
flow-matching ODE after a change of time variable, so DDIM and "flow matching with Euler"
are the same sampler for Gaussian paths.

### Rectified flow and reflow (Liu et al. 2022)

Rectified flow uses the straight interpolation X_t = t X_1 + (1 - t) X_0 between a
coupling (X_0, X_1) and minimizes E‖(X_1 - X_0) - v(X_t, t)‖². The optimum is
v(x, t) = E[X_1 - X_0 | X_t = x]. Because conditional paths cross, the marginal flow
"rewires" them and yields a new, deterministic coupling (Z_0, Z_1). **Reflow** trains a
new rectified flow on (Z_0, Z_1). Each reflow straightens trajectories, and 2-rectified
flow gives good samples with one Euler step. Reflow costs a full simulation pass to
build pairs, and errors compound.

### Stochastic interpolants

Albergo and Vanden-Eijnden (2023) and Albergo, Boffi, and Vanden-Eijnden (JMLR 2025)
generalize to x_t = α(t) x_0 + β(t) x_1 + γ(t) z with z ~ N(0, I), and learn both a
velocity and a denoiser, so you can sample either with an ODE or with an SDE whose
diffusion coefficient you choose after training. The variance-preserving interpolant
cos(½πt) x_0 + sin(½πt) x_1 is one example.

### Minibatch OT-CFM (Tong et al. 2024)

The following table (Tong et al., Table 1) summarizes common path choices, in the
Lipman time convention:

| Method | Coupling q(z) | μ_t | σ_t |
|---|---|---|---|
| FM (Lipman) | q(x_1) | t x_1 | tσ - t + 1 |
| Rectified flow | q(x_0) q(x_1) | t x_1 + (1 - t) x_0 | 0 |
| I-CFM | q(x_0) q(x_1) | t x_1 + (1 - t) x_0 | σ |
| OT-CFM | π(x_0, x_1), minibatch OT plan | t x_1 + (1 - t) x_0 | σ |
| Schrödinger bridge CFM | π_{2σ²}(x_0, x_1), entropic OT | t x_1 + (1 - t) x_0 | σ √(t(1 - t)) |

OT-CFM solves an exact OT assignment (Hungarian or `ot.emd`) between the noise and data
samples in each minibatch, then trains CFM on the matched pairs. This straightens the
marginal paths and lowers the NFE needed. The `torchcfm` package implements all of these.
Stable Audio 3 uses minibatch OT coupling in pre-training. Terminology gotcha: Matcha-TTS
and CosyVoice call their loss "OT-CFM," but they mean Lipman's OT *path* with independent
coupling, not Tong's minibatch OT *coupling*.

### Time sampling during training

- Uniform t ~ U[0, 1] (Lipman, F5-TTS, Matcha-TTS).
- Logit-normal t = sigmoid(z), z ~ N(0, 1) (SD3, DiffRhythm; Stable Audio 3 truncates at
  0.075 and rescales), which spends more training on intermediate noise levels.

### ODE solvers and NFE trade-offs

The following table lists the solvers used by audio flow models:

| Solver | NFE per step | Order | Used by |
|---|---|---|---|
| Euler | 1 | 1 | Matcha-TTS (2-10 steps), F5-TTS (16-32), CosyVoice 2 (10), DiffRhythm (32) |
| Midpoint | 2 | 2 | Voicebox (step 0.0625, NFE 32), E2 TTS (NFE 32) |
| Heun | 2 (1 on the last step) | 2 | EDM |
| RK4 | 4 | 4 | Rarely used for audio; a good reference solver in labs |
| Adaptive dopri5 | Variable | 5(4) | `torchdiffeq` default; SGMSE+ ODE baseline (RK45) |

With CFG, each evaluation costs two network passes, so "NFE 32 with CFG" is 64 passes
unless batched. Bespoke solvers (used by Audiobox) learn a solver for a fixed model and
report more than 25 times speedup.

### Common mistakes

- Using the Lipman convention in training and the SD3 convention in sampling (or the
  reverse), which integrates the wrong way and outputs noise.
- Sampling t with shape (B,) and multiplying it with (B, C, T) tensors without reshaping.
- Forgetting σ_min in the target when you use it in the interpolation (or the reverse).
- Integrating to t = 1 with a step count that doesn't land exactly on 1, for example by
  accumulating floating-point dt.
- Applying the network at t = 1 exactly when it was trained on t ∈ [0, 1) and never saw
  the endpoint; usually harmless, but it matters for σ_min = 0 and x̂ conversions that
  divide by (1 - t).
- Believing that straight conditional paths mean straight marginal paths. Only reflow,
  OT coupling, or distillation straighten the marginal field.

### Lab ideas

- **CPU-60s**: implement the conditional path and target for σ_min ∈ {0, 1e-4}; autocheck
  the worked values (-0.25 and 3; -0.250025 and 2.9999).
- **CPU-60s**: implement Euler, midpoint, and RK4 for a given v(x, t) and check them
  against the closed-form Gaussian flow (values in the worked example). Plot error against
  NFE and autocheck the empirical order of convergence (slopes near 1, 2, and 4).
- **CPU-60s**: train flow matching on the 8-Gaussians ring (timed at about 1.4 s for
  3,000 steps; 20 Euler steps give a radius mean of 1.98 and standard deviation 0.15).
  Then add minibatch OT coupling with `scipy.optimize.linear_sum_assignment` and check
  that the 2-step sample quality improves (for example, the mean distance to the nearest
  mode shrinks).
- **CPU-60s**: one round of reflow on the 2-D toy; autocheck that the path straightness
  E‖(Z_1 - Z_0) - v(Z_t, t)‖² decreases after reflow.
- **CPU-60s**: prove numerically that DDIM equals Euler on the flow-matching ODE after
  the time change, using an oracle Gaussian model.

## Flow matching in speech and audio

All models in this section use the Lipman convention (t = 0 noise, t = 1 data) unless
the text says otherwise.

### Voicebox (Le et al. 2023)

- **Task**: text-guided speech infilling. With a binary temporal mask m,
  x_mis = m ⊙ x and x_ctx = (1 - m) ⊙ x, the model learns p(x_mis | y, x_ctx). Zero-shot
  TTS, denoising, editing, style transfer, and cross-lingual generation are all infilling
  with different contexts ("in-context learning").
- **Audio model**: 80-dimensional log-mel at 100 Hz. The inputs x_t, x_ctx, and
  frame-aligned phone embeddings are concatenated per frame and projected; the flow step
  t is a sinusoidal embedding prepended as an extra token. A 24-layer Transformer
  (16 heads, 1024/4096, 330M parameters) with convolutional positional embedding,
  symmetric bidirectional ALiBi bias, and U-Net-style skip connections between symmetric
  layers.
- **Loss** (Eq. 6), computed only on masked frames:

$$
\mathcal{L}_{\text{audio-CFM-m}} = \mathbb{E}_{t, m, q(x,z), p_0(x_0)}\left\| m \odot \left(u_t(x_t \mid x) - v_t(x_t, x_{ctx}, z; \theta)\right)\right\|^2,
$$

  with x_t = (1 - (1 - σ_min) t) x_0 + t x and u_t = x - (1 - σ_min) x_0.
- **Duration model**: either a flow-matching model on phone durations or an L1
  regression model on masked phones (the regression model is the default).
- **Training details**: mask 70-100 percent of frames (audio) or 10-100 percent
  (duration); p_drop = 0.3 and 0.2; p_uncond = 0.2.
- **CFG** (Eq. 8): ṽ = (1 + α) v(w, x_ctx, z) - α v(w), with the conditioner dropped
  as a whole. α = 0.7 appears in the results tables.
- **Inference**: `torchdiffeq` midpoint solver with step 0.0625 (NFE 32). NFE 2 without
  CFG takes about 0.31 s for 10 s of audio. Vocoder: HiFi-GAN trained on the same 60K
  hours. Voicebox was never released publicly.

### Audiobox (Vyas et al. 2023)

A unified flow-matching model for speech and sound with description-based and
example-based prompting. It pre-trains Audiobox SSL with a self-supervised masked
infilling objective (following SpeechFlow) on unlabeled speech, music, and sound, then
fine-tunes for speech (Audiobox Speech) and sound (Audiobox Sound). Bespoke Solvers
reduce sampling cost by more than 25 times. Reported numbers: 0.745 speaker similarity on
LibriSpeech zero-shot TTS and 0.77 FAD on AudioCaps.

### Matcha-TTS (Mehta et al. 2024)

- **Architecture**: Glow-TTS or Grad-TTS text encoder and duration predictor with RoPE
  instead of relative position embeddings; MAS and the Grad-TTS prior loss L_enc for
  alignment. The encoder output μ, upsampled by predicted durations, conditions the
  decoder but, unlike Grad-TTS, isn't the mean of the initial noise.
- **Decoder**: a 1-D U-Net with 1-D convolutional ResNet blocks, each followed by a
  Transformer block with Snake-beta feedforward activations and no position embeddings.
  Two downsampling blocks, two mid blocks, and two upsampling blocks, each with one
  Transformer layer (hidden 256, 2 heads, attention dimension 64).
- **Loss** (Eq. 4), exactly:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, q(x_1), p_0(x_0)}\left\| u^{OT}_t\!\left(\phi^{OT}_t(x) \mid x_1\right) - v_t\!\left(\phi^{OT}_t(x) \mid \mu; \theta\right)\right\|^2,
$$

$$
\phi^{OT}_t(x) = (1 - (1-\sigma_{\min})t)\,x_0 + t\,x_1, \qquad u^{OT}_t = x_1 - (1-\sigma_{\min})\,x_0, \qquad \sigma_{\min} = 10^{-4}.
$$

  The total training loss adds L_enc and the duration loss.
- **Results on LJSpeech**: 18.2M parameters, 4.8 GiB training memory; with the Euler
  solver, NFE 10 gives WER 2.09 and MOS 3.84, NFE 2 gives WER 2.34. HiFi-GAN LJ_V1
  vocoder. Temperature 0.667.
- **Code**: `pip install matcha-tts`, MIT license; checkpoints download automatically
  through the CLI or come from a Google Drive folder. ONNX export is supported.

### E2 TTS (Eskimez et al. 2024)

- **Idea**: remove the phone aligner, the duration model, and grapheme-to-phoneme. Pad
  the character sequence with filler tokens ⟨F⟩ to the mel length and let the model learn
  alignment through the infilling task.
- Same Voicebox audio-model architecture and OT path. Evaluation used the midpoint solver
  with NFE 32 and CFG strength 1.0.
- The total duration is an input, so you can change speaking rate by scaling it.
- F5-TTS reports robustness (WER) issues for E2 TTS in zero-shot use.

### F5-TTS (Chen et al. 2024)

- **Input**: characters (full pinyin for Chinese) padded with ⟨F⟩ to N frames,
  z = (c_1, ..., c_M, ⟨F⟩, ..., ⟨F⟩). The text first goes through 4 ConvNeXt V2 blocks
  (512/1024) with absolute sinusoidal position embedding, then is concatenated in the
  feature dimension with the noisy mel (1 - t) x_0 + t x_1 and the masked mel
  (1 - m) ⊙ x_1.
- **Backbone**: DiT with adaLN-zero (t conditions adaLN), 22 layers, 16 heads, 1024/2048,
  RoPE, convolutional position embedding; 335.8M parameters in total. No U-Net skips.
- **Data**: about 95K hours of English and Chinese from Emilia; 100-bin log mel, 24 kHz,
  hop 256; 70-100 percent of frames masked.
- **Loss**: OT-CFM with σ_min = 0, E‖v_t((1 - t) x_0 + t x_1) - (x_1 - x_0)‖².
- **Duration at inference**: the ratio of character counts, N_gen ≈ N_ref · M_gen / M_ref.
- **CFG** (Eq. 4): v_CFG = v(ψ_t(x_0), c) + α (v(ψ_t(x_0), c) - v(ψ_t(x_0))), default
  α = 2 (s = 3 in the s-convention).
- **Sway sampling** (Eq. 7). Draw u ~ U[0, 1] (or a uniform grid) and map:

$$
f_{\text{sway}}(u; s) = u + s\left(\cos\!\left(\tfrac{\pi}{2}u\right) - 1 + u\right),\qquad s \in \left[-1, \tfrac{2}{\pi - 2}\right] \approx [-1, 1.752].
$$

  s < 0 concentrates steps near t = 0 (noise), where the model sketches alignment and
  contours; the default is s = -1. It's inference-only and applies to any CFM model
  without retraining.
- **Worked values** for s = -1 on a 9-point grid u = i/8 (computed during research):
  t = 0, 0.0192, 0.0761, 0.1685, 0.2929, 0.4444, 0.6173, 0.8049, 1.0.
- **Identity worth teaching**: with s = -1, f_sway(u; -1) = 1 - cos(πu/2), which is
  exactly the "cosine scheduler" t := 1 - cos(½tπ) that CosyVoice 2 uses at inference.
- **Results**: LibriSpeech-PC test-clean WER 2.42 at NFE 32 (RTF 0.31 on an RTX 3090) and
  WER 2.53 at NFE 16 (RTF 0.15). The Euler solver, EMA weights, and the Vocos vocoder.
- **Checkpoints**: `SWivid/F5-TTS` on Hugging Face (`F5TTS_v1_Base` and `F5TTS_Base`),
  CC-BY-NC-4.0 because of the Emilia training data; the code is MIT, `pip install f5-tts`.

A minimal F5-style sampler step, for lessons:

```python
# x: (B, N, F) noise at t=0; cond: masked reference mel; text: padded chars.
# Lipman convention: t=0 is noise, t=1 is data.
ts = sway(torch.linspace(0, 1, nfe + 1), s=-1.0)
for i in range(nfe):
    t, dt = ts[i], ts[i + 1] - ts[i]
    v_c = model(x, cond, text, t)
    v_u = model(x, torch.zeros_like(cond), torch.zeros_like(text), t)
    v = v_c + cfg_alpha * (v_c - v_u)  # w-convention, alpha=2
    x = x + dt * v                     # Euler
```

### CosyVoice 2 and CosyVoice 3

- **CosyVoice 2** (Du et al. 2024): a text-speech language model (Qwen2.5-0.5B
  initialized) emits 25 Hz FSQ speech tokens; a chunk-aware causal flow-matching model
  turns tokens into 50 Hz, 24 kHz mel; a HiFT vocoder makes the waveform.
  - Tokens are upsampled 2 times after a look-ahead convolution (right padding P, kernel
    P + 1) and chunk-aware causal Transformer blocks.
  - OT flow (Eqs. 3-6): ω_t = X_1 - X_0, φ_t = (1 - t) X_0 + t X_1, X_0 ~ N(0, I).
  - Network: a causal convolutional Transformer U-Net conditioned on upsampled tokens μ,
    masked mel X̃_1 (70-100 percent of final frames masked in training; the reference mel
    at inference), speaker embedding v, and t.
  - The paper writes the objective as an L1 loss (Eq. 8). t ~ U[0, 1] in training;
    inference uses t := 1 - cos(½tπ). CFG (Eq. 10) with β = 0.7 and NFE 10.
  - Streaming: the 10 unrolled flow steps are treated as one deep network. Four masks
    (non-causal, full-causal, chunk-M, chunk-2M) are sampled uniformly during training, so
    one model serves offline and streaming modes.
  - Hugging Face: `FunAudioLLM/CosyVoice2-0.5B`, Apache-2.0 (`llm.pt`, `flow.pt`,
    `hift.pt`).
- **CosyVoice 3** (Du et al. 2025): a supervised multi-task speech tokenizer derived from
  MinMo, DiffRO post-training (differentiable reward optimization), data scaled to 1M
  hours in 9 languages and 18 Chinese dialects, LM scaled from 0.5B to 1.5B, and the
  flow model switched to a DiT (100M to 300M) that drops the text encoder and length
  regulator and aligns frame rates by interpolation. Released checkpoint:
  `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (December 2025, Apache-2.0).

### Seed-TTS DiT variant (Anastassiou et al. 2024)

Seed-TTS_DiT is a fully diffusion-based, non-autoregressive variant that removes the
phoneme-duration model. It estimates only the **total** duration and learns local
alignment inside the diffusion transformer; in practice the total duration is given as an
input, which enables speaking-rate and content editing. The report gives few architecture
details.

### MaskGCT (Wang et al. 2024): a contrast, not a flow

MaskGCT is a masked generative codec transformer: two non-autoregressive stages
(text-to-semantic tokens from a VQ-VAE over w2v-BERT 2.0 features, then
semantic-to-acoustic RVQ tokens), both trained with mask-and-predict cross-entropy
L_mask = E[-Σ_i m_{t,i} log p_θ(x_i | X_t, C)] and decoded by iterative parallel
unmasking. It uses a separate flow-matching model only for total duration. Inference:
50 T2S steps, CFG 2.5 with rescale 0.75, top-k 20, temperature annealed from 1.5 to 0.
Sizes: T2S-Base 315M, T2S-Large 695M, S2A 353M. Contrast points for a lesson: discrete
tokens and categorical likelihood compared with continuous mel and velocity regression;
both are "iterative refinement from a fully corrupted sequence."

### Other flow-based and related TTS systems, 2025-2026

- **ZipVoice** (Zhu et al. 2025, Xiaomi): about 123M parameters, a Zipformer vector-field
  estimator, average-upsampling initial alignment, and flow distillation that removes the
  CFG cost. ZipVoice-Dialog extends it to two-speaker dialogue. Hugging Face: `k2-fsa/ZipVoice`.
- **IndexTTS2** (2025): autoregressive text-to-semantic model with explicit token-count
  duration control, then a flow-matching semantic-to-mel module, then BigVGAN-v2.
- **MegaTTS 3** (Jiang et al. 2025): a latent diffusion transformer with sparse alignment
  boundaries, multi-condition CFG to control accent intensity, and piecewise rectified
  flow (PeRFlow) to cut 25 steps to 8.
- **DMOSpeech** (Li et al. 2024): DMD-distills an F5-style teacher from 128 steps to 4
  and then optimizes CTC and speaker-verification losses end to end.
- **IntMeanFlow** (ByteDance, 2025): MeanFlow-style distillation without JVPs; 1-NFE
  token-to-mel for CosyVoice 2 and 3-NFE text-to-mel for F5-TTS.
- **DiFlow-TTS** (2025): discrete flow matching over factorized codec tokens.
- **dots.tts** (arXiv 2606.07080, 2026): a DiT acoustic model trained on 1.5M hours that
  exposes the DiT to its own off-trajectory errors.
- **OmniVoice** (arXiv 2604.00688, 2026): omnilingual zero-shot TTS with diffusion
  language models.
- **VoiceStar** (Peng et al. 2025) is **not** flow-based: an autoregressive
  encoder-decoder codec language model with progress-monitoring RoPE for duration control
  and extrapolation. It's a useful counterexample.

### Flow-matching vocoders

- **PeriodWave** (Lee et al. 2024): period-aware flow matching on waveforms with a
  period-conditional estimator batched over periods, a discrete wavelet transform for
  high frequencies, and FreeU to reduce high-frequency noise; 16 steps; outperforms
  BigVGAN on several LJSpeech and LibriTTS metrics in the paper.
- **RFWave** (Liu et al. 2024): multi-band rectified flow that predicts complex
  spectrograms frame-wise with a ConvNeXtV2 backbone; 10 sampling steps; energy-balanced,
  overlap, and STFT losses.
- **WaveFM** (Luo et al. 2025): reparameterized flow matching with a mel-conditioned
  prior instead of N(0, I), auxiliary multi-resolution STFT and gradient or Laplacian
  losses, and consistency distillation to one step; 6 Euler steps undistilled.

### Flow-based music and long-form audio

- **ACE-Step** (Gong et al. 2025): flow matching with a linear-attention transformer on a
  DCAE-compressed mel latent, REPA alignment to MERT and mHuBERT features; 3.5B
  parameters; up to 4 minutes of music in about 20 s on an A100.
- **ACE-Step 1.5** (arXiv 2602.00744, January 2026): an LM planner produces song
  blueprints (metadata, lyrics, captions) for a DiT of about 2B parameters on a 1-D VAE
  (48 kHz stereo into a 64-dimensional latent at 25 Hz, a 1920 times compression);
  distilled from 50 steps to 4-8; under 2 s per song on an A100 and under 4 GB VRAM. An
  XL series with a 4B DiT followed in April 2026. MIT license.
- **DiffRhythm** (Ning et al. 2025): end-to-end full songs (up to 4 min 45 s) with a
  1.1B DiT and conditional flow matching, logit-normal timesteps, sentence-level lyric
  alignment, a VAE compatible with the Stable Audio VAE latent space, Euler with 32 steps
  and CFG 4. Hugging Face: `ASLP-lab/DiffRhythm-full`.
- **DiffRhythm 2** (arXiv 2510.22950): semi-autoregressive block flow matching (blocks
  generated non-autoregressively with flow matching, autoregressive across blocks),
  stochastic block REPA, songs up to 210 s, Apache-2.0.
- **Stable Audio 2.5 and Stable Audio 3**: see [Stable Audio and Stable Audio
  Open](#stable-audio-and-stable-audio-open).

### Common mistakes in this section

- Reporting "CFG 2" without the convention. F5-TTS's α = 2 isn't the same as a diffusers
  scale of 2.
- Forgetting that the reference-audio frames are part of the generated tensor in
  infilling models. Discard x_ref frames after sampling, and compute the loss only on
  masked frames.
- Estimating F5-style duration in frames with the wrong frame rate (93.75 frames per
  second at 24 kHz and hop 256).
- Applying sway sampling to u values that aren't sorted, or to an SD3-convention model
  where "near noise" is t ≈ 1.

### Lab ideas

- **CPU-60s**: implement f_sway and check the 9-point values, monotonicity for
  s ∈ [-1, 1.75], and the identity f_sway(u; -1) = 1 - cos(πu/2).
- **CPU-60s**: implement an F5-style sampler (Euler, sway, CFG in the w-convention)
  against a mock analytic velocity field for a 2-class Gaussian mel toy; autocheck that
  α = 0 matches the conditional sampler and that the s-convention scale 1 + α gives
  identical outputs.
- **CPU-60s**: implement the Voicebox masked CFM loss with a random temporal mask and
  check that unmasked frames contribute zero gradient.
- **GPU-L4, about 10 minutes, about 6 GB**: train a tiny conditional flow-matching mel
  generator (1-D U-Net or 4-layer DiT, about 5-10M parameters) on synthetic tone data:
  random pitch and duration rendered to 80-bin mel, conditioned on a frame-level pitch
  track. Autocheck pitch accuracy of generated mels (argmax harmonic bin) and mel L1
  against targets at NFE 4 and 16.
- **GPU-L4, about 5 minutes, about 4 GB**: run pretrained F5-TTS (v1 base, 336M) for
  NFE ∈ {8, 16, 32} and s ∈ {0, -1}; compute Whisper WER and a speaker-similarity score;
  autocheck that sway with s = -1 at NFE 16 isn't worse than uniform at NFE 32 by more
  than a tolerance.
- **GPU-L4, about 3 minutes**: run Matcha-TTS with NFE ∈ {2, 4, 10} and plot RTF against
  NFE.

## Few-step generation and distillation

### Why it's needed

Undistilled flow and diffusion models need 10-100 NFE, doubled by CFG. For streaming TTS
and interactive music tools, the goal is 1-8 NFE. The main approaches, with conventions
restated:

### Progressive distillation (Salimans and Ho 2022)

Train a student to match two DDIM steps of the teacher in one step, then repeat, halving
the steps each round. This paper also introduced v-prediction, because ε-prediction is
unstable at very low SNR when few steps remain.

### Consistency models

See [Consistency models (Song et al. 2023)](#consistency-models-song-et-al-2023). Audio
uses: CoMoSpeech (TTS and singing, one step) and WaveFM (vocoder, one step).

### Shortcut models (Frans et al. 2024)

Condition the network on the step size d as well as t. Convention: x_t = (1 - t) x_0 +
t x_1 with x_0 noise (Lipman style). The self-consistency property is that one step of
size 2d equals two steps of size d:

$$
s(x_t, t, 2d) = \tfrac{1}{2}s(x_t, t, d) + \tfrac{1}{2}s(x'_{t+d}, t + d, d), \qquad x'_{t+d} = x_t + s(x_t, t, d)\,d .
$$

The loss combines flow matching at d = 0, ‖s_θ(x_t, t, 0) - (x_1 - x_0)‖², with a
self-consistency term for d > 0 against the stop-gradient two-step target. With 128 base
steps, d takes values in {1/128, ..., 1/2, 1} (8 sizes). The printed Eq. 5 of the paper
writes the second term's time as t; Eq. 4 and the algorithm use t + d.

### MeanFlow (Geng et al. 2025)

Convention: z_t = (1 - t) x + t ε, so t = 0 is data and t = 1 is noise, and v = ε - x.
MeanFlow learns the **average** velocity over an interval [r, t]:

$$
u(z_t, r, t) = \frac{1}{t - r}\int_r^t v(z_\tau, \tau)\,d\tau .
$$

Differentiating (t - r) u = ∫ v dτ with respect to t gives the MeanFlow identity:

$$
u(z_t, r, t) = v(z_t, t) - (t - r)\frac{d}{dt}u(z_t, r, t), \qquad
\frac{d}{dt}u = v\,\partial_z u + \partial_t u .
$$

Loss: ‖u_θ(z_t, r, t) - sg(u_tgt)‖² with u_tgt = v_t - (t - r)(v_t ∂_z u_θ + ∂_t u_θ),
computed with one Jacobian-vector product (`torch.func.jvp` with tangent (v, 0, 1)).
Sampling: z_r = z_t - (t - r) u_θ(z_t, r, t); one step is x = ε - u_θ(ε, 0, 1). When
r = t the method reduces to flow matching. Audio follow-ups include MeanAudio
(text-to-audio), MeanFlowSE (speech enhancement), IntMeanFlow (TTS distillation without
JVPs), and MeanFlow-TSE (target speaker extraction).

### Distribution matching distillation (DMD, DMD2)

A one-step generator G_θ is trained so that its output distribution matches the
teacher's. The gradient of the KL divergence is approximated with two score functions
evaluated on noised generator samples: a frozen "real" score from the teacher and a
"fake" score trained online on generator samples:

$$
\nabla_\theta D_{KL} \approx \mathbb{E}_{z, t}\left[w_t\left(s_{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t)\right)\frac{\partial G_\theta(z)}{\partial \theta}\right].
$$

DMD adds a regression loss on precomputed teacher noise-sample pairs. DMD2 removes the
regression loss, adds a GAN loss, and supports multi-step generators. DMOSpeech uses DMD
for TTS (4 steps) and then adds CTC and speaker-verification losses, which are only
practical because the student is fast and differentiable.

### Adversarial post-training (ARC)

ARC (Stable Audio Open Small, Stable Audio 2.5, and Stable Audio 3) fine-tunes the
pre-trained flow model into a few-step x̂_0 generator with a relativistic adversarial
loss plus a contrastive discriminator loss for prompt adherence. It needs no teacher
rollouts and no CFG, and uses ping-pong sampling. Stable Audio 3 adds an ODE warmup
distillation stage before the adversarial stage.

### Lab ideas

- **CPU-60s**: shortcut model on the 2-D toy with d ∈ {1/32, ..., 1}; autocheck that
  1-step samples have mode coverage above a threshold, compared with a plain flow model
  whose 1-step output collapses to the data mean (the point (0, 0) for the ring).
- **CPU-60s**: MeanFlow on the 2-D toy with `torch.func.jvp`; autocheck that the loss
  reduces to flow matching when r = t by comparing gradients.
- **CPU-60s**: ping-pong sampling with an oracle x̂_0 predictor on Gaussian data; check
  that the output has the correct mean and variance for any number of steps.

## Autoregressive plus continuous hybrids

### Idea

Autoregressive (AR) language models are strong at long-range structure and in-context
prompting but need discrete tokens for a softmax. Continuous-valued AR models keep the
causal transformer and replace the softmax head with a small per-token density model,
usually a diffusion or flow head, or a regression head with a sampling trick.

### MAR diffusion loss (Li et al. 2024)

The AR backbone emits a condition vector z per token; a small MLP denoiser models
p(x | z) with a standard ε-prediction diffusion loss L(z, x) = E‖ε - ε_θ(x_t | t, z)‖².
Sampling runs the DDPM reverse chain for that token. A temperature τ scales the noise in
the sampler, the continuous analog of softmax temperature.

### Next-token diffusion and VibeVoice

LatentLM (Sun et al. 2024) introduced "next-token diffusion" over σ-VAE latents.
VibeVoice (Peng et al. 2025) applies it to long-form multi-speaker speech:

- A causal acoustic tokenizer (σ-VAE) at 7.5 Hz (a 3200 times compression of 24 kHz),
  plus a separate semantic tokenizer. About two speech tokens per BPE text token.
- A Qwen2.5 LLM (1.5B or 7B) reads voice-prompt latents and the script; its hidden state
  h_i conditions a 4-layer diffusion head that predicts the next acoustic VAE latent with
  ε-prediction.
- Inference: DPM-Solver++ with 10 denoising steps and CFG scale 1.3. Up to 90 minutes and
  4 speakers in a 64K context.
- Availability: Microsoft removed the VibeVoice-TTS code from the repository after
  misuse. VibeVoice-Realtime-0.5B (released 2025-12-03) remains available at
  `microsoft/VibeVoice-Realtime-0.5B`.

### MELLE (Meng et al. 2024)

Autoregressive continuous mel-frame generation without vector quantization. The loss is
L = L_reg + λ L_KL + β L_flux + γ L_stop: L1 plus L2 regression on intermediate and final
predictions; a KL term from a latent sampling module (the model predicts a Gaussian per
frame and samples from it, which gives diversity); a spectrogram flux loss that
encourages frame-to-frame variation and prevents repetition; and a stop-token BCE. A
reduction factor lets it emit several frames per step.

### DiTAR (Jia et al. 2025)

Patch-based AR: an aggregation encoder compresses each patch of continuous tokens into
one vector for a causal LM (inter-patch), and a local diffusion transformer (LocDiT,
bidirectional within the patch, with historical patches as context) generates the next
patch (intra-patch). Temperature is defined as the time point at which noise is
introduced during the reverse ODE, which trades diversity against determinism.

### VoxCPM

VoxCPM (OpenBMB, arXiv 2509.24650) is a tokenizer-free, diffusion-autoregressive TTS built
on MiniCPM-4 with local DiT heads and FSQ-regularized hierarchical modelling. VoxCPM2 is a
2B model trained on over 2M hours, Apache-2.0 (from the project repository).

### Lab ideas

- **CPU-60s**: a tiny AR model over 2-D continuous tokens (a GRU plus an MLP diffusion
  head) on a synthetic sequence where each token is a noisy rotation of the last;
  autocheck that the sampled next-token distribution is bimodal when the data is, which a
  pure regression head can't reproduce.
- **CPU-60s**: implement diffusion-head temperature scaling and check that τ → 0 shrinks
  sample variance toward zero.

## Music generation and evaluation metrics

### Autoregressive and token-based music models

- **Jukebox** (Dhariwal et al. 2020): a 3-level VQ-VAE on 44 kHz audio with hop lengths
  8, 32, and 128 and codebook size 2048 per level; a 5B-parameter top-level prior
  (72 layers, width 4800, context 8192) conditioned on artist, genre, and lyrics;
  upsampler priors for the lower levels. Sampling takes hours per minute of audio.
- **MusicLM** (Agostinelli et al. 2023): AudioLM-style hierarchy (w2v-BERT semantic
  tokens, SoundStream acoustic tokens, 24 kHz) conditioned on MuLan joint music-text
  embeddings, which removes the need for captions during training.
- **MusicGen** (Copet et al. 2023): a single-stage transformer over EnCodec tokens
  (32 kHz, 50 Hz frame rate, 4 RVQ codebooks of size 2048). The **delay pattern** offsets
  codebook k by k steps, so at each step the model predicts codebook 1 of frame t,
  codebook 2 of frame t - 1, and so on; this keeps one forward pass per step while
  modelling inter-codebook dependence approximately. Sizes 300M, 1.5B, and 3.3B; 20K
  hours of licensed music; CFG 3.0 on logits with condition dropout 0.2; top-k 250,
  temperature 1.0.
- **YuE** (2025): LLaMA2-architecture LMs with track-decoupled next-token prediction
  (vocal and accompaniment tokens interleaved), structural progressive conditioning for
  lyrics, X-Codec tokens, a 1B stage-2 model for residual codebooks, and up to 5 minutes
  of music.

Diffusion and flow music models (Stable Audio, ACE-Step, DiffRhythm) are covered in
earlier sections.

### Fréchet Audio Distance (FAD)

Fit Gaussians to embeddings of a reference set, N(μ_r, Σ_r), and of the generated set,
N(μ_g, Σ_g). Kilgour et al. (2019), Eq. 1:

$$
\text{FAD} = \|\mu_r - \mu_g\|^2 + \operatorname{tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right).
$$

- The original embedding is VGGish (128-dimensional, 0.96 s frames of 64-bin log-mel).
  Stable Audio uses OpenL3 (FD_openl3). `fadtk` (Gui et al. 2024) supports CLAP-LAION,
  EnCodec, MERT, and others; it introduces FAD∞, which fits a line to FAD computed at
  several sample sizes and extrapolates to infinite size to remove sample-size bias, and
  per-song FAD for outlier detection.
- Worked checks: in 1-D, FAD = (μ_r - μ_g)² + (σ_r - σ_g)². For μ_r = 0, σ_r = 1,
  μ_g = 1, and σ_g = 2, FAD = 1 + 1 = 2. For diagonal covariances,
  FAD = ‖Δμ‖² + Σ_i (σ_{r,i} - σ_{g,i})².
- Gotchas: `scipy.linalg.sqrtm` can return complex values from numerical noise; take the
  real part and warn if the imaginary part is large. FAD is biased upward at small sample
  sizes, so compare only at equal sizes or use FAD∞. Embedding models expect specific
  sample rates (VGGish 16 kHz). Don't average frame embeddings per clip unless the
  protocol says to; `fadtk` uses frame-level embeddings.

### CLAP score

The mean cosine similarity between the CLAP text embedding of each prompt and the CLAP
audio embedding of its generated audio. Some papers clip at 0 or multiply by 100. The
CLAP checkpoint (for example, LAION-CLAP music or general) must be reported.

### KL on classifier outputs

Pass reference and generated audio for the same prompt through an audio tagger (PaSST in
Stable Audio's KL_passt, PANNs in AudioLDM's evaluation), then average the KL divergence
between the reference and generated label distributions over pairs. It measures semantic
correspondence rather than quality. Implementation details (softmax compared with
per-class sigmoid, and the KL direction) vary; always name the code you use.

### Lab ideas

- **CPU-60s**: implement FAD from embedding matrices; autocheck the 1-D value 2.0, the
  diagonal formula, and the full-covariance case against a closed-form commuting example;
  check that FAD of a set against itself is about 0 and increases with added noise.
- **CPU-60s**: implement FAD∞ extrapolation on synthetic Gaussian embeddings and check
  that it's closer to the analytic FAD than the finite-sample estimate.
- **GPU-L4, about 5 minutes**: compute FAD with `fadtk` (CLAP-LAION embeddings) between
  two small real sets and between real and generated sets from Stable Audio Open Small,
  and check the ordering.

## Pretrained checkpoints for serverless labs

The following table lists checkpoints that fit a serverless L4 (24 GB) or A100 (40-80 GB)
for inference labs. Licenses come from the model cards on the research date. Sizes are
parameter counts from papers or model cards; they're not download sizes unless stated.

| Model | Hugging Face or source | Size | License | Packages | Notes |
|---|---|---|---|---|---|
| Matcha-TTS | `pip install matcha-tts` (auto-download; Google Drive) | 18.2M | MIT | `matcha-tts`, `phonemizer`, espeak-ng | LJSpeech; runs on CPU; Euler NFE 2-10 |
| F5-TTS | `SWivid/F5-TTS` (`F5TTS_v1_Base`) | 336M | CC-BY-NC-4.0 (weights), MIT (code) | `f5-tts`, `vocos` | Needs about 3-4 GB VRAM for inference (estimate) |
| CosyVoice 2 | `FunAudioLLM/CosyVoice2-0.5B` | 0.5B LM plus flow plus HiFT | Apache-2.0 | CosyVoice repo, Python 3.10, sox | Streaming flow decoder |
| CosyVoice 3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 0.5B | Apache-2.0 | CosyVoice repo | DiT flow decoder, 10-step Euler |
| ZipVoice | `k2-fsa/ZipVoice` | about 123M | Not verified | ZipVoice repo | Distilled variant available |
| Stable Audio Open Small | `stabilityai/stable-audio-open-small` | 341M DiT, 0.5B total | Stability AI Community License, gated | `stable-audio-tools` | 8 ping-pong steps, 11 s |
| Stable Audio Open 1.0 | `stabilityai/stable-audio-open-1.0` | 1.06B DiT, about 1.3B total | Stability AI Community License, gated | `stable-audio-tools` or diffusers | 100 DPM-Solver++ steps, 47 s |
| Stable Audio 3 small and medium | `stabilityai/stable-audio-3-small-sfx`, `stabilityai/stable-audio-3-medium` | 459M and 1.4B DiT | Stability AI Community License, gated | `stable-audio-tools` | 8 steps |
| AudioLDM 2 | `cvssp/audioldm2` | 1.1B total | CC-BY-NC-SA-4.0 | diffusers `AudioLDM2Pipeline` | 16 kHz, 200 steps |
| Tango 2 | `declare-lab/tango2` | about 866M U-Net plus FLAN-T5-Large | CC-BY-NC-SA-4.0 | Tango repo | 16 kHz, 100-200 steps |
| DiffWave | `lmnt.com` checkpoint through `pip install diffwave` | 31 MB file | Apache-2.0 | `diffwave` | LJSpeech 22.05 kHz; `fast_sampling=True` |
| Vocos | `charactr/vocos-mel-24khz` | about 13.5M | MIT | `vocos` | 100-bin mel, 24 kHz; F5-TTS vocoder |
| BigVGAN-v2 | `nvidia/bigvgan_v2_24khz_100band_256x` | 112M | MIT | BigVGAN repo | Optional CUDA kernel |
| ACE-Step 1.5 | ACE-Step 1.5 GitHub and Hugging Face | about 2B DiT plus LM | MIT | ACE-Step 1.5 repo | Under 4 GB VRAM claimed |
| DiffRhythm | `ASLP-lab/DiffRhythm-full` | 1.1B | Not verified | DiffRhythm repo | Full songs, 32 Euler steps |
| VibeVoice Realtime | `microsoft/VibeVoice-Realtime-0.5B` | 0.5B | Not verified | VibeVoice repo | Streaming next-token diffusion |

We recommend Matcha-TTS (CPU-friendly, MIT) and Vocos for the first pretrained labs,
F5-TTS for sampler labs (sway and CFG) on an L4, and Stable Audio Open Small for
text-to-audio labs. The gated Stability AI models need each student, or the course
account, to accept the license; the course must check whether a shared token is allowed.

## Proposed lesson ordering

The following sequence covers this part of the course in eight lessons. Each lesson has
one primary lab:

1. **Generative modelling for audio and normalizing flows.** Lab: implement affine
   coupling, ActNorm, and LU 1x1 convolution with exact log-determinant checks, then train
   RealNVP on 2-D data. CPU-60s.
2. **GANs for vocoding.** Lab: implement LSGAN and hinge losses, feature matching, the MPD
   reshape, and a multi-resolution STFT loss with shape and value checks; optional
   tiny-HiFi-GAN run. CPU-60s (optional GPU-L4, about 15 minutes).
3. **DDPM from scratch.** Lab: forward process, posterior, parameter conversions against
   the worked 4-step table, then a 2-D toy DDPM with ancestral sampling and moment checks.
   CPU-60s.
4. **Samplers, score SDEs, and guidance.** Lab: DDIM (η = 0 and 1), EDM σ schedule and
   Heun with an oracle Gaussian denoiser, and CFG on a 2-class mixture, checked against
   closed-form answers. CPU-60s.
5. **Diffusion for speech and audio.** Lab: DiffWave fast-sampling alignment and Grad-TTS
   forward statistics (CPU-60s), plus a tiny DiffWave on synthetic tones. GPU-L4, about
   10 minutes.
6. **Flow matching and ODE solvers.** Lab: OT path and target, Euler, midpoint, and RK4
   convergence against the closed-form Gaussian flow, then 2-D flow matching with and
   without minibatch OT, and one reflow round. CPU-60s.
7. **Flow-matching TTS (Matcha, Voicebox, E2, F5, CosyVoice).** Lab: masked CFM loss, sway
   sampling, and an F5-style sampler with CFG against a mock field (CPU-60s), then a tiny
   conditional flow-matching mel generator on tone data. GPU-L4, about 10 minutes.
8. **Few-step generation, hybrids, and evaluation.** Lab: shortcut or MeanFlow on the 2-D
   toy and FAD with FAD∞ on synthetic embeddings (CPU-60s), then Stable Audio Open Small
   or F5-TTS at several NFE with CLAP score or WER. GPU-L4 or GPU-A100, about 5 minutes.

An optional ninth lesson on music generation (MusicGen delay pattern, latent diffusion
music, and long-form structure) can reuse the FAD and CLAP tooling. Lab: implement the
delay-pattern interleaving and its inverse with exact round-trip checks. CPU-60s.

## Unverified or uncertain facts

The research couldn't confirm the following items, or found conflicting information:

- **CosyVoice 2 loss type**: the paper writes an L1 loss (Eq. 8). The research didn't
  check whether the released code uses L1 or MSE.
- **Stable Audio 3 sample rate**: the technical report describes 44.1 kHz stereo; a
  summary of the `stable-audio-3-medium` model card suggested 48 kHz and "2B params,"
  which might include the autoencoder. Check the card before a lab.
- **Stable Audio 3 small-music repository name**: only `stable-audio-3-small-sfx` and
  `stable-audio-3-medium` were confirmed.
- **Stable Audio 2.5 details**: no paper; the facts come from the company announcement
  and press coverage.
- **Stable Audio Open 1.0 CFG convention**: the paper reports scale 7.0; the s-convention
  is assumed from `stable-audio-tools` and wasn't checked in source code.
- **Licenses not confirmed**: ZipVoice, DiffRhythm (v1), and VibeVoice-Realtime-0.5B.
- **VRAM figures**: all VRAM and runtime estimates for GPU labs are estimates, except the
  CPU toy timings and figures quoted from papers.
- **Authorship of "Diffusion meets flow matching"**: the page didn't show a byline in the
  fetched content; the attribution to Gao et al. (Google DeepMind, 2024) is from memory.
- **CoMoSpeech details**: the title was verified; its internals weren't read.
- **Seed-TTS DiT architecture**: the report gives few details beyond total-duration input.
- **MelGAN hinge typo**: the PDF text extraction shows `min(0, ...)`; this might be an
  extraction artifact rather than a typo in the paper.
- **ACE-Step 1.5 release month**: web sources say January 2026; the arXiv version read was
  v3 of February 2026.
- **Kingma and Gao equivalence claim**: taken as stated in F5-TTS; the research didn't
  re-derive the "v-prediction with cosine schedule" equivalence.

## Bibliography

### Foundations

- Dinh, Sohl-Dickstein, and Bengio. Density estimation using Real NVP. 2016. https://arxiv.org/abs/1605.08803
- Kingma and Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. 2018. https://arxiv.org/abs/1807.03039
- Papamakarios et al. Normalizing flows for probabilistic modeling and inference. 2019. https://arxiv.org/abs/1912.02762
- Chen et al. Neural ordinary differential equations. 2018. https://arxiv.org/abs/1806.07366
- Grathwohl et al. FFJORD. 2018. https://arxiv.org/abs/1810.01367
- Mao et al. Least squares generative adversarial networks. 2016. https://arxiv.org/abs/1611.04076
- Ho, Jain, and Abbeel. Denoising diffusion probabilistic models. 2020. https://arxiv.org/abs/2006.11239
- Nichol and Dhariwal. Improved denoising diffusion probabilistic models. 2021. https://arxiv.org/abs/2102.09672
- Song, Meng, and Ermon. Denoising diffusion implicit models. 2020. https://arxiv.org/abs/2010.02502
- Song and Ermon. Generative modeling by estimating gradients of the data distribution. 2019. https://arxiv.org/abs/1907.05600
- Song et al. Score-based generative modeling through stochastic differential equations. 2020. https://arxiv.org/abs/2011.13456
- Ho and Salimans. Classifier-free diffusion guidance. 2022. https://arxiv.org/abs/2207.12598
- Karras et al. Elucidating the design space of diffusion-based generative models (EDM). 2022. https://arxiv.org/abs/2206.00364
- Salimans and Ho. Progressive distillation for fast sampling of diffusion models. 2022. https://arxiv.org/abs/2202.00512
- Song et al. Consistency models. 2023. https://arxiv.org/abs/2303.01469
- Hang et al. Efficient diffusion training via Min-SNR weighting strategy. 2023. https://arxiv.org/abs/2303.09556
- Kingma and Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. 2023. https://arxiv.org/abs/2303.00848
- Lu et al. DPM-Solver. 2022. https://arxiv.org/abs/2206.00927
- Lu et al. DPM-Solver++. 2022. https://arxiv.org/abs/2211.01095
- Rombach et al. High-resolution image synthesis with latent diffusion models. 2021. https://arxiv.org/abs/2112.10752
- Peebles and Xie. Scalable diffusion models with transformers (DiT). 2022. https://arxiv.org/abs/2212.09748
- Esser et al. Scaling rectified flow transformers for high-resolution image synthesis (SD3). 2024. https://arxiv.org/abs/2403.03206

### Flow matching and few-step methods

- Lipman et al. Flow matching for generative modeling. 2022. https://arxiv.org/abs/2210.02747
- Liu, Gong, and Liu. Flow straight and fast: rectified flow. 2022. https://arxiv.org/abs/2209.03003
- Albergo and Vanden-Eijnden. Building normalizing flows with stochastic interpolants. 2022. https://arxiv.org/abs/2209.15571
- Albergo, Boffi, and Vanden-Eijnden. Stochastic interpolants: a unifying framework. JMLR 2025. https://arxiv.org/abs/2303.08797
- Tong et al. Improving and generalizing flow-based generative models with minibatch optimal transport. TMLR 2024. https://arxiv.org/abs/2302.00482
- Gao et al. Diffusion meets flow matching (blog). 2024. https://diffusionflow.github.io/
- Frans et al. One step diffusion via shortcut models. 2024. https://arxiv.org/abs/2410.12557
- Geng et al. Mean flows for one-step generative modeling. 2025. https://arxiv.org/abs/2505.13447
- Yin et al. One-step diffusion with distribution matching distillation. 2023. https://arxiv.org/abs/2311.18828
- Yin et al. Improved distribution matching distillation (DMD2). 2024. https://arxiv.org/abs/2405.14867
- Yan et al. PeRFlow: piecewise rectified flow. 2024. https://arxiv.org/abs/2405.07510

### Vocoders and GANs

- Prenger, Valle, and Catanzaro. WaveGlow. 2018. https://arxiv.org/abs/1811.00002
- Kumar et al. MelGAN. 2019. https://arxiv.org/abs/1910.06711
- Kong, Kim, and Bae. HiFi-GAN. 2020. https://arxiv.org/abs/2010.05646
- Jang et al. UnivNet. 2021. https://arxiv.org/abs/2106.07889
- Lee et al. BigVGAN. 2022. https://arxiv.org/abs/2206.04658
- Siuzdak. Vocos. 2023. https://arxiv.org/abs/2306.00814
- Défossez et al. EnCodec: high fidelity neural audio compression. 2022. https://arxiv.org/abs/2210.13438
- Kumar et al. High-fidelity audio compression with improved RVQGAN (DAC). 2023. https://arxiv.org/abs/2306.06546
- Kong et al. DiffWave. 2020. https://arxiv.org/abs/2009.09761
- Chen et al. WaveGrad. 2020. https://arxiv.org/abs/2009.00713
- Lee et al. PriorGrad. 2021. https://arxiv.org/abs/2106.06406
- Huang et al. FastDiff. 2022. https://arxiv.org/abs/2204.09934
- Lee et al. PeriodWave. 2024. https://arxiv.org/abs/2408.07547
- Liu et al. RFWave. 2024. https://arxiv.org/abs/2403.05010
- Luo, Miao, and Duan. WaveFM. 2025. https://arxiv.org/abs/2503.16689

### Speech generation

- Kim et al. Glow-TTS. 2020. https://arxiv.org/abs/2005.11129
- Kim, Kong, and Son. VITS. 2021. https://arxiv.org/abs/2106.06103
- Popov et al. Grad-TTS. 2021. https://arxiv.org/abs/2105.06337
- Liu et al. DiffSinger. 2021. https://arxiv.org/abs/2105.02446
- Ye et al. CoMoSpeech. 2023. https://arxiv.org/abs/2305.06908
- Shen et al. NaturalSpeech 2. 2023. https://arxiv.org/abs/2304.09116
- Ju et al. NaturalSpeech 3. 2024. https://arxiv.org/abs/2403.03100
- Le et al. Voicebox. 2023. https://arxiv.org/abs/2306.15687
- Vyas et al. Audiobox. 2023. https://arxiv.org/abs/2312.15821
- Mehta et al. Matcha-TTS. 2023. https://arxiv.org/abs/2309.03199
- Eskimez et al. E2 TTS. 2024. https://arxiv.org/abs/2406.18009
- Chen et al. F5-TTS. 2024. https://arxiv.org/abs/2410.06885
- Lee et al. DiTTo-TTS. 2024. https://arxiv.org/abs/2406.11427
- Anastassiou et al. Seed-TTS. 2024. https://arxiv.org/abs/2406.02430
- Wang et al. MaskGCT. 2024. https://arxiv.org/abs/2409.00750
- Du et al. CosyVoice 2. 2024. https://arxiv.org/abs/2412.10117
- Du et al. CosyVoice 3. 2025. https://arxiv.org/abs/2505.17589
- Li et al. DMOSpeech. 2024. https://arxiv.org/abs/2410.11097
- Jiang et al. MegaTTS 3. 2025. https://arxiv.org/abs/2502.18924
- Zhu et al. ZipVoice. 2025. https://arxiv.org/abs/2506.13053
- Zhu et al. ZipVoice-Dialog. 2025. https://arxiv.org/abs/2507.09318
- Zhou et al. IndexTTS2. 2025. https://arxiv.org/abs/2506.21619
- Peng et al. VoiceStar. 2025. https://arxiv.org/abs/2505.19462
- Wang et al. IntMeanFlow. 2025. https://arxiv.org/abs/2510.07979
- DiFlow-TTS. 2025. https://arxiv.org/abs/2509.09631
- VoxCPM. 2025. https://arxiv.org/abs/2509.24650
- OmniVoice. 2026. https://arxiv.org/abs/2604.00688
- dots.tts technical report. 2026. https://arxiv.org/abs/2606.07080
- Richter et al. Speech enhancement and dereverberation with diffusion-based generative models (SGMSE+). 2022. https://arxiv.org/abs/2208.05830

### Hybrids

- Li et al. Autoregressive image generation without vector quantization (MAR). 2024. https://arxiv.org/abs/2406.11838
- Sun et al. Multimodal latent language modeling with next-token diffusion (LatentLM). 2024. https://arxiv.org/abs/2412.08635
- Peng et al. VibeVoice technical report. 2025. https://arxiv.org/abs/2508.19205
- Meng et al. MELLE: autoregressive speech synthesis without vector quantization. 2024. https://arxiv.org/abs/2407.08551
- Jia et al. DiTAR. 2025. https://arxiv.org/abs/2502.03930

### Sound and music generation

- Liu et al. AudioLDM. 2023. https://arxiv.org/abs/2301.12503
- Liu et al. AudioLDM 2. 2023. https://arxiv.org/abs/2308.05734
- Huang et al. Make-An-Audio. 2023. https://arxiv.org/abs/2301.12661
- Ghosal et al. Tango: text-to-audio generation using instruction-tuned LLM and latent diffusion. 2023. https://arxiv.org/abs/2304.13731
- Majumder et al. Tango 2. 2024. https://arxiv.org/abs/2404.09956
- Kreuk et al. AudioGen. 2022. https://arxiv.org/abs/2209.15352
- Schneider et al. Moûsai. 2023. https://arxiv.org/abs/2301.11757
- Evans et al. Fast timing-conditioned latent audio diffusion (Stable Audio). 2024. https://arxiv.org/abs/2402.04825
- Evans et al. Stable Audio Open. 2024. https://arxiv.org/abs/2407.14358
- Novack et al. Fast text-to-audio generation with adversarial post-training (ARC, Stable Audio Open Small). 2025. https://arxiv.org/abs/2505.08175
- Evans et al. Stable Audio 3 technical report. 2026. https://arxiv.org/abs/2605.17991
- Stability AI. Stable Audio 2.5 announcement. 2025. https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale
- Li et al. MeanAudio. 2025. https://arxiv.org/abs/2508.06098
- Dhariwal et al. Jukebox. 2020. https://arxiv.org/abs/2005.00341
- Agostinelli et al. MusicLM. 2023. https://arxiv.org/abs/2301.11325
- Copet et al. Simple and controllable music generation (MusicGen). 2023. https://arxiv.org/abs/2306.05284
- Borsos et al. SoundStorm. 2023. https://arxiv.org/abs/2305.09636
- YuE: scaling open foundation models for long-form music generation. 2025. https://arxiv.org/abs/2503.08638
- Gong et al. ACE-Step. 2025. https://arxiv.org/abs/2506.00045
- Gong et al. ACE-Step 1.5. 2026. https://arxiv.org/abs/2602.00744
- Ning et al. DiffRhythm. 2025. https://arxiv.org/abs/2503.01183
- Jiang et al. DiffRhythm 2. 2025. https://arxiv.org/abs/2510.22950

### Evaluation

- Kilgour et al. Fréchet Audio Distance. 2018. https://arxiv.org/abs/1812.08466
- Gui et al. Adapting Fréchet Audio Distance for generative music evaluation (fadtk). 2023. https://arxiv.org/abs/2311.01616
- Wu et al. Large-scale contrastive language-audio pretraining (LAION-CLAP). 2022. https://arxiv.org/abs/2211.06687
- Koutini et al. Efficient training of audio transformers with Patchout (PaSST). 2021. https://arxiv.org/abs/2110.05069

### Model cards and repositories

- F5-TTS weights: https://huggingface.co/SWivid/F5-TTS
- Matcha-TTS code: https://github.com/shivammehta25/Matcha-TTS
- CosyVoice 2: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- CosyVoice 3: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- ZipVoice: https://huggingface.co/k2-fsa/ZipVoice
- Stable Audio Open Small: https://huggingface.co/stabilityai/stable-audio-open-small
- Stable Audio Open 1.0: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Stable Audio 3 medium: https://huggingface.co/stabilityai/stable-audio-3-medium
- AudioLDM 2: https://huggingface.co/cvssp/audioldm2
- Tango 2: https://huggingface.co/declare-lab/tango2
- DiffWave: https://github.com/lmnt-com/diffwave
- Vocos: https://huggingface.co/charactr/vocos-mel-24khz
- BigVGAN-v2: https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x
- ACE-Step 1.5: https://github.com/ace-step/ACE-Step-1.5
- DiffRhythm: https://huggingface.co/ASLP-lab/DiffRhythm-full
- VibeVoice: https://github.com/microsoft/VibeVoice
- VibeVoice Realtime: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B
- VoxCPM: https://github.com/OpenBMB/VoxCPM
- torchcfm: https://github.com/atong01/conditional-flow-matching
- fadtk: https://github.com/microsoft/fadtk
