---
title: Neural audio codecs
slug: 31-neural-audio-codecs
part: "Part 6 — Codecs and token models"
summary: Build the pieces of a neural codec, count its bits, and measure what each extra codebook buys on EnCodec and Mimi.
minutes: 75
gpu: true
objectives:
  - Compute a codec's frame rate, tokens per second, and bitrate from its strides, codebook count, and codebook size.
  - Build quantizer dropout and a batched residual quantizer, so one model serves every bitrate.
  - Assemble a toy convolutional codec whose encoder, quantizer, and decoder have exact length arithmetic.
  - "Score reconstructions with SI-SDR and multi-scale mel distance, and explain why the two can disagree."
  - Measure quality against bitrate for EnCodec and Mimi on real speech.
prereqs:
  - 30-vector-quantization
  - 27-gan-vocoders
  - 07-hearing-and-the-mel-scale
exercise: 31-neural-audio-codecs
---

# Neural audio codecs

Uncompressed audio is expensive: 16-bit PCM at 24 kHz costs 384 kilobits per
second (kbps). A neural codec sends the same speech at 1.5 to 6 kbps, and the listener still hears the speaker's voice. It does
this by learning what speech looks like, so it only has to send the part
that changes.

A neural codec is an autoencoder with a quantizer in the middle. The encoder
shrinks the waveform to a few dozen vectors a second. The residual quantizer
from lesson 30, *Vector quantization*, turns each vector into a handful of
integers. The decoder, which is a GAN vocoder in all but name, turns the
integers back into sound.

Codecs matter twice in this course: their integers are the vocabulary of
the next lesson's language models, and their decoders are those models'
vocoders. In this lesson you count a codec's bits, build the parts that let
one model serve many bitrates, and measure what each codebook buys.

## What you already know

- Residual vector quantization codes a vector in stages, and each stage costs $\log_2 K$ bits for a codebook of $K$ entries.
- The straight-through estimator passes the decoder's gradient through the quantizer unchanged.
- A strided convolution with stride $s$ produces one output for every $s$ input samples.
- A GAN vocoder trains its generator with an adversarial loss, feature matching, and a multi-resolution spectral loss.
- A log-mel spectrogram compresses each frame's spectrum onto a perceptual frequency scale.

## An autoencoder with a quantized middle

SoundStream set the pattern that EnCodec, DAC, and Mimi follow
(Zeghidour et al., 2021). The encoder is a stack of convolution blocks. Each
block ends with a strided convolution that divides the time axis by its
stride. SoundStream and EnCodec use strides $(2, 4, 5, 8)$, so the encoder
turns 320 samples into one latent vector. The number of channels doubles at
each downsampling, so the time axis shrinks while each frame grows richer.
EnCodec adds a two-layer LSTM and a final convolution to 128 dimensions. The
decoder mirrors the encoder with transposed convolutions in reverse order.

The number of samples per frame is the *hop*, the product of the strides.
The *frame rate* is the sample rate divided by the hop:

$$
f_{\text{frame}} = \frac{f_s}{\prod_j s_j}
$$

Read it aloud: the frame rate in frames per second is the sample rate $f_s$
in hertz divided by the product of the strides $s_j$. For EnCodec at 24 kHz,
$24{,}000 / 320 = 75$ frames per second, which is 13.3 ms of audio per frame.

Mimi, the codec behind Kyutai's Moshi, goes much further
(Défossez et al., 2024). Its convolutions use strides $(4, 5, 6, 8)$, a hop of
960, which gives 25 Hz. One more stride-2 layer after its Transformer halves
that to 12.5 frames per second: 80 ms per frame, a total hop of 1920. The
`config.json` of the `mimi` asset lists `upsampling_ratios` of
$[8, 6, 5, 4]$ and a `frame_rate` of 12.5, and the **Run** checks that the
two agree.

```python
fps = sample_rate / math.prod(strides)   # 24000 / 320 = 75.0
```

## Count the bits

Every frame, each of the $N_q$ codebooks sends one index of $\log_2 K$ bits.
The bitrate is therefore:

$$
\text{bitrate} = f_{\text{frame}} \times N_q \times \log_2 K
$$

Read it aloud: bits per second equal frames per second, times the number of
codebooks in use, times the bits per index. The *token rate*, which is what a
language model over the codes must predict, is $f_{\text{frame}} \times N_q$
tokens per second.

```python
bits_per_second = fps * n_codebooks * math.log2(codebook_size)
```

### Work the numbers by hand

EnCodec's 24 kHz model has 32 codebooks of 1024 entries, 10 bits each:

- At 6 kbps it uses 8 codebooks: $75 \times 8 \times 10 = 6{,}000$ bits per second, and $75 \times 8 = 600$ tokens per second.
- At 1.5 kbps it uses 2 codebooks: $75 \times 2 \times 10 = 1{,}500$ bits per second, 150 tokens per second.
- At 24 kbps it uses all 32: 24,000 bits per second, 2,400 tokens per second.
- 16-bit PCM at 24 kHz is $24{,}000 \times 16 = 384{,}000$ bits per second, so 6 kbps is a compression of 64 times.

Mimi has codebooks of 2048 entries, 11 bits each. With 8 codebooks, as Moshi
uses it, the bitrate is $12.5 \times 8 \times 11 = 1{,}100$ bits per second,
only 100 tokens per second. The `transformers` model carries all 32
codebooks, which give 4.4 kbps.

The following table lists the values the exercise checks, read from each
codec's configuration.

| Codec | $f_s$ | Hop | Frame rate | Codebooks $\times$ size | Bitrate |
|---|---|---|---|---|---|
| EnCodec 24 kHz | 24,000 Hz | 320 | 75 Hz | 2 to 32 $\times$ 1024 | 1.5 to 24 kbps |
| Mimi | 24,000 Hz | 1920 | 12.5 Hz | 1 to 32 $\times$ 2048 | 0.14 to 4.4 kbps |
| DAC 44 kHz | 44,100 Hz | 512 | 86.13 Hz | 9 $\times$ 1024 | 7.75 kbps |

The following demo turns the three knobs of the formula.

```demo
{"id": "codec-bitrate", "caption": "Start from EnCodec, raise the hop to lower the frame rate and add codebooks to hold the bitrate, and compare how many tokens per second each setting asks a language model to predict."}
```

> [!KEY]
> Bitrate is frame rate times codebooks times bits per index. For a codec language model, the token rate matters more than the bitrate, so low frame rates like Mimi's 12.5 Hz make the next lesson's models practical.

## One model for every bitrate

A codec trained always with 8 codebooks decodes badly with 2, because its
decoder has never seen such a coarse latent. *Quantizer dropout* fixes this.
For each training example, draw how many stages it uses, and quantize it with
only those. SoundStream draws the count uniformly from 1 to $N_q$. DAC found
that this hurt quality at full bitrate, so it applies dropout to each example
with probability $p = 0.5$ and uses every stage otherwise
(Kumar et al., 2023):

$$
n_q^{(b)} =
\begin{cases}
U\{1, \ldots, N_q\} & \text{with probability } p \\
N_q & \text{otherwise}
\end{cases}
$$

Read it aloud: example $b$ in the batch uses a count drawn uniformly from 1 to
$N_q$ with probability $p$, and all $N_q$ stages otherwise. With $p = 0.5$
and $N_q = 8$, an example uses all 8 stages with probability
$0.5 + 0.5 / 8 = 0.5625$, and each smaller count with probability
$0.5 / 8 = 0.0625$.

```python
counts = torch.randint(1, n_q + 1, (batch,), generator=g)
drop = torch.rand(batch, generator=g) < p
counts = torch.where(drop, counts, n_q)
```

Inside the batched quantizer, stage $i$ is *active* for example $b$ when
$n_q^{(b)} > i$. An inactive stage adds nothing to that example's latent,
costs nothing in its losses, and records the code $-1$. In the frame layout
$(B, D, T)$, the mask has shape $(B,)$ and broadcasts over frames and
channels. At evaluation, dropout is off, and you choose the count by hand.

> [!WARNING]
> Apply the straight-through estimator once, to the sum of all stages: $z + \operatorname{sg}[\hat z - z]$. If you pass the gradient through every stage's residual too, the encoder receives the gradient once per stage.

## What the codec is trained on

The losses come from the vocoder lessons, with one addition from the
quantizer. The following list gives EnCodec's recipe at 24 kHz
(Défossez et al., 2022):

- **Reconstruction.** An L1 loss on the waveform, weight 0.1, and an L1 plus L2 loss on 64-band mel spectrograms at seven window sizes from 32 to 2048, weight 1.
- **Adversarial and feature matching.** A multi-scale STFT discriminator on complex spectrograms, as in lesson 27, *GAN vocoders*, weights 3 and 3.
- **Commitment.** The RVQ commitment loss summed over stages. EMA updates move the codebooks.
- **A loss balancer.** Each loss's gradient on the output is rescaled so its weight sets its share of the total gradient, whatever its natural scale.

Mimi changes two things. Its first codebook is trained to match WavLM
features, so that token carries what was said more than how it sounded; the
next lesson calls this a *semantic* token. And its authors found that training
with only the adversarial and feature-matching losses sounded better in
listening tests, even though objective scores got worse. You measure the
effect of that choice in this lesson's **Run**.

## Measure what survives

Two numbers score a reconstruction $\hat s$ against the original $s$.

*SI-SDR*, the scale-invariant signal-to-distortion ratio, asks how much of
the estimate is a scaled copy of the reference. Subtract each signal's mean,
project the estimate onto the reference, and compare the projection with
what is left over:

$$
\alpha = \frac{\langle \hat s, s \rangle}{\langle s, s \rangle}, \qquad
\text{SI-SDR} = 10 \log_{10} \frac{\lVert \alpha s \rVert^2}{\lVert \hat s - \alpha s \rVert^2}
$$

Read it aloud: $\alpha$ is the gain that best matches the reference to the
estimate. The target $\alpha s$ is the part of the estimate that the
reference explains, and $\hat s - \alpha s$ is the distortion. SI-SDR is the
ratio of their energies in decibels. Because $\alpha$ absorbs any gain,
turning the volume up or down doesn't change the score. The score depends
only on the angle between the two zero-mean signals, so swapping the
estimate and the reference gives the same value.

```python
e, r = est - est.mean(), ref - ref.mean()
alpha = (e @ r) / (r @ r)
si_sdr = 10 * np.log10(np.sum((alpha * r) ** 2) / np.sum((e - alpha * r) ** 2))
```

*Mel distance* compares log-mel spectrograms, which ignore phase and weight
frequencies roughly as the ear does. DAC reports a multi-scale version. The
exercise takes the mean absolute difference of $\log_{10}$ mel magnitudes,
floored at $10^{-5}$, at four scales from 256-point frames with 20 bands to
2048-point frames with 160 bands, and averages over the scales.

### Work SI-SDR by hand

Take the reference $s = (1, -1, 1, -1)$ and the estimate
$\hat s = (2.1, -1.9, 1.9, -2.1)$, which is $2s$ plus the error
$(0.1, 0.1, -0.1, -0.1)$. Both already have mean zero.

- $\langle \hat s, s \rangle = 2.1 + 1.9 + 1.9 + 2.1 = 8$, and $\langle s, s \rangle = 4$, so $\alpha = 2$.
- The target is $2s$, with energy $4 \times 4 = 16$.
- The distortion is $(0.1, 0.1, -0.1, -0.1)$, with energy $0.04$.
- SI-SDR is $10 \log_{10}(16 / 0.04) = 10 \log_{10} 400 = 26.02$ dB.

The factor of 2 in the estimate costs nothing. Plain SNR against $s$ would
count it as error and give about 0 dB.

### Why the two scores disagree

SI-SDR compares waveforms sample by sample, so it punishes any change of
phase. A codec trained with adversarial losses is free to rebuild a sound
that has the right spectrum but a different phase, and the ear can't tell.
The following table shows what the **Run** measured on 16 clips of
LibriSpeech dev-clean, 6 seconds each.

| Codec, codebooks | Bitrate | SI-SDR | Mel distance |
|---|---|---|---|
| EnCodec, 2 | 1.5 kbps | 0.3 dB | 0.396 |
| EnCodec, 8 | 6.0 kbps | 6.6 dB | 0.325 |
| EnCodec, 32 | 24.0 kbps | 10.0 dB | 0.285 |
| Mimi, 1 (semantic only) | 0.14 kbps | $-43.6$ dB | 0.982 |
| Mimi, 8 | 1.1 kbps | 3.2 dB | 0.316 |
| Mimi, 32 | 4.4 kbps | 9.7 dB | 0.246 |

By mel distance, Mimi at 1.1 kbps matches EnCodec at 6 kbps, and Mimi at
4.4 kbps beats EnCodec at 24 kbps. By SI-SDR, Mimi at 1.1 kbps looks far
worse. With only its semantic codebook, Mimi's output shares almost nothing
with the waveform, yet it still carries speech: listen to it in the **Run**.

> [!KEY]
> SI-SDR measures waveform fidelity and mel distance measures spectral fidelity. A codec trained to sound right can score poorly on SI-SDR, so report both, and listen.

> [!RECAP]
> - The frame rate is $f_s / \prod_j s_j$, and the bitrate is frame rate $\times N_q \times \log_2 K$.
> - Quantizer dropout gives each example $U\{1..N_q\}$ stages with probability $p$, and all of them otherwise.
> - The batched quantizer masks inactive stages per example, marks their codes $-1$, and applies the straight-through estimator once.
> - SI-SDR projects the estimate onto the reference, and mel distance compares log-mel spectrograms at several scales.

## Build it step by step

Build the functions in the order the checks test them.

1. **`frame_rate(sample_rate, strides)` and `bitrate(frames_per_second, n_codebooks, codebook_size)`.** Use `math.prod` and `math.log2`. Common mistake: using the hop in milliseconds, or $K$ instead of $\log_2 K$.
2. **`sample_quantizer_counts(batch, n_q, p, generator)`.** Draw counts with `torch.randint(1, n_q + 1, ...)` and coins with `torch.rand`, both from `generator`, and combine with `torch.where`. Common mistake: `torch.randint(1, n_q, ...)`, whose upper bound is exclusive.
3. **`rvq_forward(z, codebooks, n_q)`.** Permute to `(B, T, D)`, loop over stages, compute distances on detached tensors, and gather `q = codebooks[i][idx]`. Mask each stage with `n_q > i`. Add the codebook loss with `sg[r]`, the commitment loss with `sg[q]`, and subtract `sg[q]` from the residual. Return `z + (hard - z).detach()` permuted back. Common mistake: averaging a stage's loss over all examples, including those for which it is inactive.
4. **`ToyCodec`.** Build `encoder`, `decoder`, and `codebooks` as the docstring lists. A convolution with kernel $s$ and stride $s$ divides the length by exactly $s$, and a transposed convolution with the same kernel and stride multiplies it back. `forward` chains `encode`, `rvq_forward`, and `decode`. `codes_to_audio` sums codewords and decodes. Common mistake: forgetting to permute the summed codewords from `(B, T, D)` to `(B, D, T)`.
5. **`si_sdr(estimate, reference)`.** Remove the means, project, and take the ratio in dB. Common mistake: computing $\alpha$ with $\langle \hat s, \hat s \rangle$ in the denominator while still scaling the reference, which mixes the two directions of the projection.
6. **`mel_distance(x, y, sample_rate, scales)`.** Trim to the shorter length, call the provided `log_mel` at each scale, and average the mean absolute differences.

## Cost and scaling

The following table compares the two pretrained codecs on the quantities
that set their cost.

| Quantity | EnCodec 24 kHz | Mimi |
|---|---|---|
| Parameters, as `transformers` loads them | 14.9M | 79.3M |
| Encoder hop | 320 samples | 1920 samples |
| Tokens per second at full depth | 2,400 | 400 |
| Tokens for a 10-second utterance at about 1 kbps | 1,500 at 1.5 kbps | 1,000 at 1.1 kbps |

The encoder and decoder cost grows linearly with the audio length. The
quantizer costs $O(T N_q K D)$ per clip, which is small next to the
convolutions. What grows fastest is downstream: a Transformer over the codes
pays attention cost proportional to the square of the token count, so a
codec's frame rate decides how long a clip a language model can handle.

The full **Run**, 16 clips of 6 seconds through both codecs at 11 settings,
takes well under a minute on an A100, mostly loading the two models. Without
a GPU it shrinks to 2 clips of 3 seconds.

## Where this breaks

**Mismatched sample rates.** EnCodec and Mimi expect 24 kHz. LibriSpeech is
16 kHz, so the **Run** resamples by $3/2$ first. A 16 kHz clip fed as if it
were 24 kHz plays back fast and high, and every metric is meaningless.

**Quantizer dropout left on at evaluation.** Evaluate each $N_q$ on its own,
with dropout off, or the scores mix bitrates.

**Trusting one metric.** SI-SDR punishes a phase change the ear ignores, and
mel distance misses clicks and buzz that a listener hears at once. Pair them
with listening, or with a learned quality predictor.

**The toy codec collapses.** With equal weights on reconstruction and the VQ
losses, the toy encoder shrinks its latents to a point, and the output goes
silent. The **Run** weights a normalized reconstruction error by 100 to
prevent it. Real codecs use the loss balancer instead.

## The exercise

You build `frame_rate`, `bitrate`, `sample_quantizer_counts`, `rvq_forward`,
the `ToyCodec` class, `si_sdr`, and `mel_distance`. The checks test the
codec bitrates, the statistics of quantizer dropout, the two-stage RVQ
example, the straight-through gradient, per-example stage masking, the toy
codec's exact lengths and code round trip, the SI-SDR example and its gain
invariance, and the mel distance against an independent implementation.

Press **Run** on a GPU worker to encode LibriSpeech clips with `encodec-24khz`
at each of its five bandwidths and `mimi` at 1 to 32 codebooks. You see
SI-SDR and mel distance plotted against bitrate, and hear the original, both
codecs at their lowest and highest bitrates, and Mimi's semantic codebook
alone. Without the assets, the **Run** trains your `ToyCodec` on synthetic
voices for 20 seconds instead.
