---
title: Flow matching for speech
slug: 37-flow-matching-for-speech
part: "Part 7 — Diffusion and flow matching"
summary: Train a conditional flow-matching mel generator that says the digits, with Voicebox's masked loss, F5-TTS's sway sampling, and guidance.
minutes: 80
gpu: true
objectives:
  - Build the masked infilling loss that Voicebox, E2 TTS, and F5-TTS train with, and explain why text-to-speech is a special case of infilling.
  - Apply classifier-free guidance to a velocity, convert between the two guidance conventions, and batch both passes into one call.
  - Warp the sampling time grid with sway sampling and explain why the early steps near noise matter most.
  - Build CosyVoice 2's chunk-causal attention mask and explain how one model serves streaming and offline synthesis.
  - Turn generated log-mels into sound with a pseudo-inverse filterbank and Griffin-Lim.
prereqs:
  - 36-flow-matching
  - 34-samplers-scores-and-guidance
  - 07-hearing-and-the-mel-scale
  - 05-phase-and-griffin-lim
exercise: 37-flow-matching-for-speech
---

# Flow matching for speech

The flow from the last lesson carried 2-D points from noise to data. Replace
each point with a mel spectrogram and add a condition, such as a transcript or
a digit label, and you have the acoustic model of a text-to-speech (TTS)
system. Voicebox, E2 TTS, F5-TTS, Matcha-TTS, and the decoder of CosyVoice 2
all work this way. A separate vocoder then turns the mel spectrogram into a
waveform.

Four ideas turn the plain flow into a speech generator. A masked loss teaches
the model to fill in missing frames, which covers TTS, editing, and voice
cloning at once. Guidance makes the output follow its condition more strongly.
A warped time grid spends the sampling steps where they matter. A chunk mask
lets the same model stream. In this lesson you build all four, then train a
small model that says the digits zero to nine, and you hear it.

The convention stays as it was in the last lesson: $t = 0$ is noise and
$t = 1$ is data, and $\sigma_{\min} = 0$, as in F5-TTS.

## What you already know

- A flow-matching model regresses $v_\theta(x_t, t)$ onto $x_1 - x_0$ along
  the straight path $x_t = (1 - t)\,x_0 + t\,x_1$, and samples by Euler steps.
- A log-mel spectrogram is a matrix of bands by frames. The frame rate is the
  sample rate divided by the hop.
- Griffin-Lim recovers a waveform from a magnitude spectrogram by alternating
  between the STFT and its inverse.
- Classifier-free guidance combines a conditional and an unconditional
  prediction to follow the condition more closely.
- An attention mask sets which positions each query may look at.

## Speech as a picture to fill in

Picture a mel spectrogram as a painting with a strip cut out. Voicebox (Le et
al.) trains one model to repaint the strip, given the rest of the painting and
the text. A *mask* $m$ marks each frame 1 when the model must generate it and
0 when it can see it. The visible frames form the *context*
$x_{\text{ctx}} = (1 - m) \odot x_1$, which the network receives next to the
noisy input $x_t$. The loss counts only the frames the model had to generate:

$$
\mathcal{L} = \frac{\sum \left[ m \odot \left(v_\theta(x_t, x_{\text{ctx}}, c, t) - (x_1 - x_0)\right)^2 \right]}{F \sum m}
$$

Read it aloud: take the squared velocity error at every band and frame, keep
only the masked frames, and divide by the number of masked values, $F$ bands
times the number of masked frames. Frames the model could see add nothing,
whatever it outputs there. The symbol $c$ is the condition, the transcript in
Voicebox and the digit label in this lesson's run. In torch, with mels shaped
`(B, F, N)` and a mask shaped `(B, N)`:

```python
m = mask[:, None, :]
context = (1 - m) * x1
loss = (((v - (x1 - x0)) ** 2) * m).sum() / (m.sum() * v.shape[1])
```

Voicebox masks 70 to 100 percent of the frames of each training clip. At
inference, *everything* is infilling. For zero-shot TTS, the context is a few
seconds of a reference speaker, and the masked part is the new sentence in
that voice. For editing, the mask covers one word. F5-TTS goes further: it
drops the phone aligner and pads the character sequence with filler tokens to
the mel length, so the model also learns where each character goes.

### Work one masked loss by hand

Take one band and four frames: data $x_1 = (2, 1, -1, 0)$, noise
$x_0 = (0.5, -1, 1, 0.2)$, mask $m = (0, 1, 1, 0)$, and $t = 0.5$. Then:

- $x_t = 0.5\,x_0 + 0.5\,x_1 = (1.25, 0, 0, 0.1)$.
- The context is $(2, 0, 0, 0)$: frame 1 is visible, frames 2 and 3 are masked,
  and frame 4 is visible but happens to be 0.
- The target is $x_1 - x_0 = (1.5, 2, -2, -0.2)$.
- A prediction $v = (0, 1, -1, 0)$ has squared errors $(2.25, 1, 1, 0.04)$.
  The masked ones sum to 2, over $1 \times 2$ masked values, so the loss is 1.

> [!KEY]
> Masked infilling turns text-to-speech, editing, and voice cloning into one
> task: generate the masked frames, given the visible ones and the condition.

## Guidance for a flow

Classifier-free guidance works on velocities the same way it works on noise
predictions. During training, the condition is replaced by a *null* condition
some of the time, with probability 0.2 in Voicebox. At sampling time, the
network makes two predictions at each step, one with the condition and one
without, and extrapolates past the conditional one:

$$
\tilde v = v_c + \alpha\,(v_c - v_u)
$$

Read it aloud: start from the conditional velocity $v_c$ and push further in
the direction that separates it from the unconditional velocity $v_u$, by a
factor $\alpha$. With $v_c = 1.0$, $v_u = 0.4$, and $\alpha = 2$, the guided
velocity is $1.0 + 2 \times 0.6 = 2.2$. In code:

```python
both = model(torch.cat([x, x]), t, torch.cat([cond, null_cond]))
v = both[:B] + alpha * (both[:B] - both[B:])
```

Papers write the strength in two ways. This lesson uses the *w-convention* of
Voicebox and F5-TTS, where $\alpha = 0$ means no guidance. The *s-convention*
of diffusers and AudioLDM writes $v_u + s\,(v_c - v_u)$, where $s = 1$ means no
guidance. They are the same formula with $s = 1 + \alpha$. F5-TTS's default
$\alpha = 2$ is $s = 3$, and CosyVoice 2's $\beta = 0.7$ is $s = 1.7$.

> [!WARNING]
> "CFG 2" means twice as much extra push in the w-convention as in the
> s-convention. Always report which one you use. Also match the null
> condition to training exactly. F5-TTS drops the audio context with
> probability 0.3 and the audio and text together with 0.2, and its
> unguided pass drops both.

Guidance doubles the cost of each step. Stacking the two passes into one batch
of $2B$ rows, as the code does, keeps it to one network call per step.

## Spend the steps where they matter

Near $t = 0$ the input is almost pure noise, and the model decides the
large-scale layout: where each sound goes and roughly how loud it is. Near
$t = 1$ it only polishes detail. F5-TTS's *sway sampling* moves the step
boundaries toward the noise end:

$$
f_{\text{sway}}(u; s) = u + s\left(\cos\!\left(\tfrac{\pi}{2}u\right) - 1 + u\right)
$$

Read it aloud: take evenly spaced times $u$ and bend them by $s$ times a curve
that is 0 at both ends. For $s < 0$ the times crowd toward 0. At $s = -1$ the
formula simplifies to $1 - \cos(\pi u / 2)$, which is the cosine time scheduler
CosyVoice 2 uses. In numpy:

```python
u = np.linspace(0, 1, nfe + 1)
t = u + s * (np.cos(np.pi / 2 * u) - 1 + u)
```

For $s = -1$ and 8 steps, the grid $u = i / 8$ becomes the following times:
0, 0.0192, 0.0761, 0.1685, 0.2929, 0.4444, 0.6173, 0.8049, and 1. The first
step is 0.0192 long and the last is 0.1951, against 0.125 each for an even
grid. The map stays increasing for $s$ between $-1$ and
$2 / (\pi - 2) \approx 1.752$. Sway needs no retraining, and F5-TTS reports
2.53 percent word error rate on LibriSpeech-PC test-clean at NFE 16 with it,
against 2.42 at NFE 32.

## Stream with a chunk mask

A streaming TTS system must play the first words before it has generated the
last ones. CosyVoice 2 (Du et al.) runs a language model that emits speech
tokens at 25 Hz, upsamples them to 50 Hz, and uses a flow-matching decoder to
turn them into mel frames. Its decoder is a convolutional transformer U-Net,
and its attention mask decides what each frame may see. With *chunks* of $M$
frames, a frame may attend to every frame in its own chunk and in earlier
chunks, never later ones:

$$
\text{allowed}[i, j] = \left( \lfloor j / M \rfloor \le \lfloor i / M \rfloor \right)
$$

Read it aloud: query frame $i$ may look at key frame $j$ when the chunk of $j$
isn't later than the chunk of $i$. For six frames in chunks of two, frames 0
and 1 see frames 0 and 1, frames 2 and 3 see frames 0 to 3, and frames 4 and 5
see everything. Chunks of 1 give a fully causal mask, and one chunk as long as
the clip gives full attention. In numpy:

```python
chunk = np.arange(n) // M
allowed = chunk[None, :] <= chunk[:, None]
```

During training, CosyVoice 2 samples one of four masks for each batch:
non-causal, fully causal, chunks of $M$, and chunks of $2M$. One set of
weights then serves offline synthesis, which sees everything, and streaming,
which waits for only one chunk. The flow's ten sampling steps run on each
chunk as it arrives.

## From mel back to sound

The model generates 80-band log-mel frames. A real system hands them to a
neural vocoder such as HiFi-GAN, Vocos, or BigVGAN. This lesson uses the tools
you built earlier, so the chain is exact and inspectable.

First undo the mel filterbank. The filterbank $W$ maps 513 FFT bins to 80
bands, so many spectra share each mel frame. The pseudo-inverse $W^+$ picks
the one with the smallest total energy, which spreads each band's energy
smoothly across the bins it covers. Negative values can appear between
filters, so clip them to 0 and take the square root to get a magnitude:

$$
\hat{S} = \sqrt{\max\left(W^+ \exp(\text{log-mel}),\ 0\right)}
$$

Read it aloud: exponentiate the log-mel back to power, solve for a power
spectrum by least squares, clip, and take the square root. Then Griffin-Lim
invents a phase that fits the magnitude. The smoothing blurs the harmonics,
so the result sounds whispery and buzzy even for a real recording. The
**Run** block plays real mels through the same chain so you can hear that
ceiling, apart from the model's own errors.

> [!RECAP]
> - The masked loss averages the squared velocity error over masked frames
>   only, and the context is $(1 - m) \odot x_1$.
> - Guidance is $v_c + \alpha(v_c - v_u)$ in the w-convention, which equals
>   the s-convention with $s = 1 + \alpha$.
> - Sway sampling with $s = -1$ is $t = 1 - \cos(\pi u / 2)$, with small steps
>   near noise.
> - A chunk mask lets frame $i$ attend to frame $j$ when
>   $\lfloor j/M \rfloor \le \lfloor i/M \rfloor$.
> - The pseudo-inverse filterbank and Griffin-Lim turn a log-mel into sound.

## Build it step by step

1. **`sway(u, s)`** returns the warped times, same type as `u`. Pick
   `torch.cos` for a tensor and `np.cos` otherwise. Check it at `u = 0.5`,
   where $s = -1$ gives 0.2929.
2. **`infilling_batch(x0, x1, t, mask)`** returns `(x_t, context, target)`,
   each `(B, F, N)`. Reshape `t` to `(B, 1, 1)` and `mask` to `(B, 1, N)`.
   The common mistake is building the context from `x_t` instead of the clean
   `x1`.
3. **`masked_flow_loss(v_pred, target, mask)`** multiplies the squared error
   by the mask, sums, and divides by `mask.sum() * F`. Taking `.mean()` over
   all elements instead makes the loss shrink when fewer frames are masked.
4. **`cfg_velocity(v_cond, v_uncond, alpha)`** is one line. Check that
   `alpha = 0` returns `v_cond`.
5. **`sample(velocity, x0, cond, null_cond, nfe, sway_s, cfg_alpha)`** builds
   the grid with `sway(torch.linspace(0, 1, nfe + 1), sway_s)`, then loops
   over steps. With guidance, call `velocity` once on `torch.cat([x, x])` with
   `torch.cat([cond, null_cond])` and a `(2B,)` time; without it, call once on
   `x`. The mistake to avoid is a time grid built by adding a fixed `dt`,
   which ignores the warp.
6. **`chunk_causal_mask(n_frames, chunk_size)`** is one broadcast comparison
   of chunk indices.
7. **`mel_to_magnitude(mel_power, filterbank)`** uses `np.linalg.pinv`,
   `np.maximum`, and `np.sqrt`. Feed it power, not log-mel.

## Cost and scaling

Training costs one network pass per batch, as in lesson 36. Sampling costs
NFE passes, doubled by guidance unless you batch them. The following table
lists the pieces of this lesson's run, and real systems for scale:

| Piece | This lesson | Real systems |
|---|---|---|
| Frames per second | $16000 / 256 = 62.5$ | Voicebox 100, F5-TTS $24000 / 256 = 93.75$, CosyVoice 2 50 |
| Model | 0.74M parameters on CPU, 5.7M on a GPU | Matcha-TTS 18.2M, Voicebox 330M, F5-TTS 335.8M |
| Steps | NFE 16 with sway and guidance | Matcha-TTS 2 to 10, CosyVoice 2 10, F5-TTS 16 or 32, Voicebox 32 |
| Mask cost | $n^2$ entries for $n$ frames | Grows with the square of clip length, like attention itself |

The attention inside a transformer decoder costs time proportional to the
square of the number of frames. A 30-second clip at 93.75 frames per second
has 2,813 frames, so long-form models cut audio into chunks or turn to latent
representations with fewer frames per second.

## Where this breaks

- **The null condition drifts.** If training zeroes the label embedding and
  sampling uses a learned null token, the unconditional branch is garbage and
  guidance amplifies it. Use one null, in one place, for both.
- **Guidance over-saturates.** Large $\alpha$ makes every sample the most
  typical one: loud, flat, and all alike. The run's classifier accuracy rises
  with guidance while the variety falls. MaskGCT adds a guidance rescale of
  0.75 for this reason.
- **The reference frames are left in.** An infilling model regenerates the
  whole tensor. Discard the frames that belonged to the reference before you
  vocode, and compute the loss only on masked frames.
- **The duration is wrong.** F5-TTS estimates the output length from the
  ratio of character counts, $N_{\text{gen}} \approx N_{\text{ref}} M_{\text{gen}} / M_{\text{ref}}$.
  A wrong frame rate in that estimate squeezes or stretches the speech.
- **The vocoder sets the ceiling.** Griffin-Lim on a pseudo-inverted mel can't
  sound natural. Judge the model by its mels, or by the real-mel baseline
  through the same chain.

## The exercise

You build `sway`, `infilling_batch`, `masked_flow_loss`, `cfg_velocity`,
`sample`, `chunk_causal_mask`, and `mel_to_magnitude`. The checks confirm the
sway grid, its identity with $1 - \cos(\pi u / 2)$, and its range of
monotonicity. They verify the infilling inputs and that unmasked frames get
exactly zero gradient, the two guidance conventions, and that `sample` calls
the field once per step with $2B$ rows on the sway grid. On an exact two-class
Gaussian field, `sample` must reach the class and guidance must push past it.
The last checks compare your chunk mask with a brute-force rule, your mel
inverse with the pseudo-inverse, and train a tiny conditional flow with your
pieces.

Press **Run** to train on `fsdd`, resampled to 16 kHz and cut to 0.6 s, with
digit labels as the condition, a null label 20 percent of the time, and whole
or 70 to 100 percent masks. On a laptop CPU, the small version trained for
700 steps in about 30 s. A linear classifier that scores 97 percent on real
held-out digits read 89 percent of its unguided samples and 100 percent of its
guided ones as the right digit. A 5,000-step run with a 1.2M-parameter model
read 97 percent unguided, and its mels show formants and fricatives. The
network also adds a per-band gain on $x_t$ set by the time, the linear part of
the Gaussian field in lesson 36. Without it, the small run's loss after 700
steps was 0.93 instead of 0.50, and its mels stayed grainy. The run shows real
and generated mels for 0 to 9 and plays both through the same vocoder. On a
GPU, the same block trains a 5.7M-parameter model for 10,000 steps.
