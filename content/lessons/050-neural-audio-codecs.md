---
title: Neural audio codecs
slug: 050-neural-audio-codecs
part: "Part 6 — Codecs and token models"
summary: Build the pieces of a neural codec, count its bits, and measure what each extra codebook buys on EnCodec and Mimi.
minutes: 40
gpu: true
objectives:
  - Compute a codec's frame rate, tokens per second, and bitrate from its strides, codebook count, and codebook size.
  - Build quantizer dropout and a batched residual quantizer, so one model serves every bitrate.
  - Assemble a toy convolutional codec whose encoder, quantizer, and decoder have exact length arithmetic.
  - "Score reconstructions with SI-SDR and multi-scale mel distance, and explain why the two can disagree."
  - Measure quality against bitrate for EnCodec and Mimi on real speech.
prereqs:
  - 049-residual-and-scalar-quantization
  - 045-gan-vocoders
  - 011-hearing-and-the-mel-scale
exercise: 050-neural-audio-codecs
---

# Neural audio codecs

Uncompressed audio is expensive. 16-bit PCM at 24 kHz costs 384 kilobits per second (kbps), and a voice call cannot afford that. A neural codec sends the same speech at 1.5 to 6 kbps, and the listener still recognizes the speaker. It can do this because it learned what speech looks like, so it only sends the part that changes.

How many bits does a codec send, and what does each extra bit buy? A codec is an autoencoder with a quantizer in the middle. The encoder shrinks the waveform to a few dozen vectors per second, the residual quantizer from the previous chapter turns each vector into a handful of integers, and the decoder, a GAN vocoder in all but name, turns the integers back into sound.

Their integers are the vocabulary of the [language models](/lessons/051-codec-language-models) in the next chapter.

## What you already know

- Residual vector quantization codes a vector in stages, and each stage costs $\log_2 K$ bits for a codebook of $K$ entries (see [Residual and scalar quantization](/lessons/049-residual-and-scalar-quantization)).
- The straight-through estimator passes the decoder's gradient through the quantizer unchanged.
- A strided convolution with stride $s$ produces one output for every $s$ input samples.
- A GAN vocoder trains its generator with an adversarial loss, feature matching, and a multi-resolution spectral loss (see [GAN vocoders](/lessons/045-gan-vocoders)).
- A log-mel spectrogram compresses each frame's spectrum onto a perceptual frequency scale.

## An autoencoder with a quantized middle

SoundStream set the pattern that EnCodec, DAC, and Mimi follow (Zeghidour et al., 2021). The encoder is a stack of convolution blocks, and each block ends with a strided convolution that divides the time axis by its stride. SoundStream and EnCodec use strides $(2, 4, 5, 8)$, so the encoder turns 320 samples into one latent vector. The decoder mirrors the encoder with transposed convolutions in reverse order.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A neural codec: audio at 24000 samples per second goes through an encoder that divides time by 320, a residual quantizer with 8 codebooks of 1024, and a decoder">
<defs><marker id="c046-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="10" y="40" width="100" height="70" rx="6" class="None stroke-muted"/><text x="60.0" y="63.0" class="fig-small fig-bold" text-anchor="middle">waveform</text><text x="60.0" y="79.0" class="fig-small" text-anchor="middle">24,000</text><text x="60.0" y="95.0" class="fig-small" text-anchor="middle">samples/s</text>
<rect x="150" y="40" width="110" height="70" rx="6" class="None stroke-muted"/><text x="205.0" y="63.0" class="fig-small fig-bold" text-anchor="middle">encoder</text><text x="205.0" y="79.0" class="fig-small" text-anchor="middle">strides</text><text x="205.0" y="95.0" class="fig-small" text-anchor="middle">2, 4, 5, 8</text>
<rect x="300" y="40" width="120" height="70" rx="6" class="None stroke-muted"/><text x="360.0" y="63.0" class="fig-small fig-bold" text-anchor="middle">RVQ</text><text x="360.0" y="79.0" class="fig-small" text-anchor="middle">8 codebooks</text><text x="360.0" y="95.0" class="fig-small" text-anchor="middle">of 1024</text>
<rect x="460" y="40" width="120" height="70" rx="6" class="None stroke-muted"/><text x="520.0" y="63.0" class="fig-small fig-bold" text-anchor="middle">decoder</text><text x="520.0" y="79.0" class="fig-small" text-anchor="middle">mirrors the</text><text x="520.0" y="95.0" class="fig-small" text-anchor="middle">encoder</text>
<line x1="110" y1="75" x2="148" y2="75" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<line x1="260" y1="75" x2="298" y2="75" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<line x1="420" y1="75" x2="458" y2="75" class="stroke-ink" marker-end="url(#c046-arrow)"/>
<text x="205" y="30" class="fig-small" text-anchor="middle">hop 320</text>
<text x="205" y="135" class="fig-small" text-anchor="middle">75 frames/s</text>
<text x="205" y="151" class="fig-small" text-anchor="middle">of 128 numbers</text>
<text x="360" y="135" class="fig-small" text-anchor="middle">8 integers per frame</text>
<text x="360" y="151" class="fig-small" text-anchor="middle">10 bits each</text>
<line x1="205" y1="112" x2="205" y2="124" class="stroke-muted"/><line x1="360" y1="112" x2="360" y2="124" class="stroke-muted"/>
<rect x="150" y="175" width="270" height="52" rx="6" class="fill-soft-b stroke-b"/>
<text x="285" y="197" class="fig-bold" text-anchor="middle">75 x 8 x 10 = 6,000 bits/s</text>
<text x="285" y="215" class="fig-small" text-anchor="middle">600 integers per second cross the wire</text>
<text x="535" y="30" class="fig-small" text-anchor="middle">24,000 samples/s</text>
</svg>
Figure 1. Audio goes in as 24,000 numbers a second and crosses the wire as 600 integers a second. Notice that the quantizer is the only place information is thrown away, so its bit budget sets the bitrate.
```

The number of samples per frame is the *hop*, the product of the strides. The *frame rate* is the sample rate divided by the hop:

$$
f_{\text{frame}} = \frac{f_s}{\prod_j s_j}
$$

Read it aloud: the frame rate in frames per second is the sample rate $f_s$ in hertz divided by the product of the strides $s_j$. For EnCodec at 24 kHz, $24{,}000 / 320 = 75$ frames per second, which is 13.3 ms of audio per frame.

Mimi, the codec behind Kyutai's Moshi, goes much further (Défossez et al., 2024). Its convolutions use strides $(4, 5, 6, 8)$, a hop of 960, which gives 25 Hz, and one more stride-2 layer halves that to 12.5 frames per second: 80 ms per frame, a hop of 1920. The `config.json` of the `mimi` asset lists `upsampling_ratios` of $[8, 6, 5, 4]$ and a `frame_rate` of 12.5, and the **Run** checks that the two agree.

```python
fps = sample_rate / math.prod(strides)   # 24000 / 320 = 75.0
```

## Count the bits

Every frame, each of the $N_q$ codebooks sends one index of $\log_2 K$ bits. So the bitrate is:

$$
\text{bitrate} = f_{\text{frame}} \times N_q \times \log_2 K
$$

Read it aloud: bits per second equal frames per second, times the number of codebooks in use, times the bits per index. The *token rate*, which is what a language model over the codes must predict, is $f_{\text{frame}} \times N_q$ tokens per second.

```python
bits_per_second = fps * n_codebooks * math.log2(codebook_size)
```

### Work the numbers by hand

EnCodec's 24 kHz model has 32 codebooks of 1024 entries, 10 bits each:

- At 6 kbps it uses 8 codebooks: $75 \times 8 \times 10 = 6{,}000$ bits per second, and $75 \times 8 = 600$ tokens per second.
- At 1.5 kbps it uses 2 codebooks, and at 24 kbps all 32: $75 \times 2 \times 10 = 1{,}500$ and $75 \times 32 \times 10 = 24{,}000$ bits per second, which are 150 and 2,400 tokens per second.
- 16-bit PCM at 24 kHz is $24{,}000 \times 16 = 384{,}000$ bits per second, so 6 kbps is a compression of 64 times.

Mimi has codebooks of 2048 entries, 11 bits each. With 8 codebooks, as Moshi uses it, the bitrate is $12.5 \times 8 \times 11 = 1{,}100$ bits per second, only 100 tokens per second. All 32 codebooks give 4.4 kbps.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Bitrates on a logarithmic axis: 384 kbps for 16-bit PCM, 24, 6, and 1.5 kbps for EnCodec, and 1.1 kbps for Mimi with 8 codebooks">
<line x1="190" y1="30" x2="190" y2="200" class="stroke-grid"/><text x="190" y="218" class="fig-small" text-anchor="middle">1</text>
<line x1="290" y1="30" x2="290" y2="200" class="stroke-grid"/><text x="290" y="218" class="fig-small" text-anchor="middle">10</text>
<line x1="390" y1="30" x2="390" y2="200" class="stroke-grid"/><text x="390" y="218" class="fig-small" text-anchor="middle">100</text>
<text x="340.0" y="240" class="fig-small" text-anchor="middle">bitrate in kbps, log scale</text>
<text x="180" y="53" class="fig-small" text-anchor="end">16-bit PCM</text>
<rect x="190" y="36" width="266.4" height="24" class="fill-muted"/>
<text x="462.4" y="53" class="fig-small">384 kbps</text>
<text x="180" y="85" class="fig-small" text-anchor="end">EnCodec, 32 codebooks</text>
<rect x="190" y="68" width="146.0" height="24" class="fill-a"/>
<text x="342.0" y="85" class="fig-small">24 kbps</text>
<text x="180" y="117" class="fig-small" text-anchor="end">EnCodec, 8 codebooks</text>
<rect x="190" y="100" width="85.8" height="24" class="fill-a"/>
<text x="281.8" y="117" class="fig-small">6 kbps</text>
<text x="180" y="149" class="fig-small" text-anchor="end">EnCodec, 2 codebooks</text>
<rect x="190" y="132" width="25.6" height="24" class="fill-a"/>
<text x="221.6" y="149" class="fig-small">1.5 kbps</text>
<text x="180" y="181" class="fig-small" text-anchor="end">Mimi, 8 codebooks</text>
<rect x="190" y="164" width="12.1" height="24" class="fill-b"/>
<text x="208.1" y="181" class="fig-small">1.1 kbps</text>
</svg>
Figure 2. Bitrates on a log axis. Mimi at 1.1 kbps sends one-fifth of the bits of EnCodec at 6 kbps, because its frames are six times longer.
```

DAC at 44.1 kHz has a hop of 512, which is 86.13 frames per second, and with 9 codebooks of 1024 it runs at 7.75 kbps.

The following demo turns the three knobs of the formula.

```demo
{"id": "codec-bitrate", "caption": "Start from EnCodec, raise the hop to lower the frame rate and add codebooks to hold the bitrate, and compare how many tokens per second each setting asks a language model to predict."}
```

> [!KEY]
> Bitrate is frame rate times codebooks times bits per index. For a codec language model, the token rate matters more than the bitrate, so low frame rates like Mimi's 12.5 Hz make the next chapter's models practical.

## One model for every bitrate

A codec trained always with 8 codebooks decodes badly with 2, because its decoder has never seen such a coarse latent. *Quantizer dropout* fixes this: for each training example, draw how many stages it uses, and quantize it with only those. SoundStream draws the count uniformly from 1 to $N_q$. DAC found that this hurt quality at full bitrate, so it applies dropout with probability $p = 0.5$ and uses every stage otherwise (Kumar et al., 2023):

$$
n_q^{(b)} =
\begin{cases}
U\{1, \ldots, N_q\} & \text{with probability } p \\
N_q & \text{otherwise}
\end{cases}
$$

Read it aloud: example $b$ in the batch uses a count drawn uniformly from 1 to $N_q$ with probability $p$, and all $N_q$ stages otherwise. With $p = 0.5$ and $N_q = 8$, an example uses all 8 stages with probability $0.5 + 0.5 / 8 = 0.5625$, and each smaller count with probability $0.5 / 8 = 0.0625$.

```python
counts = torch.randint(1, n_q + 1, (batch,), generator=g)
drop = torch.rand(batch, generator=g) < p
counts = torch.where(drop, counts, n_q)
```

Inside the batched quantizer, stage $i$ is *active* for example $b$ when $n_q^{(b)} > i$. An inactive stage adds nothing to that example's latent, costs nothing in its losses, and records the code $-1$. At evaluation, dropout is off, and you choose the count by hand.

> [!WARNING]
> Apply the straight-through estimator once, to the sum of all stages: $z + \operatorname{sg}[\hat z - z]$. If you pass the gradient through every stage's residual too, the encoder receives the gradient once per stage.

## What the codec is trained on

The losses come from the vocoder chapters, plus one addition from the quantizer. EnCodec at 24 kHz uses a waveform L1 loss, a multi-scale mel loss, and a multi-scale STFT discriminator with feature matching, and adds the RVQ commitment loss summed over stages. EMA updates move the codebooks (Défossez et al., 2022).

Mimi changes two things. Its first codebook is trained to match WavLM features, so that token carries what was said more than how it sounded. The next chapter calls this a *semantic* token. And its authors found that training with only the adversarial and feature-matching losses sounded better in listening tests, even though objective scores got worse.

## Measure what survives

Two numbers score a reconstruction $\hat s$ against the original $s$.

*SI-SDR*, the scale-invariant signal-to-distortion ratio, asks how much of the estimate is a scaled copy of the reference. Subtract each signal's mean, project the estimate onto the reference, and compare the projection with what is left over:

$$
\alpha = \frac{\langle \hat s, s \rangle}{\langle s, s \rangle}, \qquad
\text{SI-SDR} = 10 \log_{10} \frac{\lVert \alpha s \rVert^2}{\lVert \hat s - \alpha s \rVert^2}
$$

Read it aloud: $\alpha$ is the gain that best matches the reference to the estimate, the target $\alpha s$ is the part of the estimate the reference explains, and $\hat s - \alpha s$ is the distortion. Because $\alpha$ absorbs any gain, turning the volume up or down does not change the score.

```python
e, r = est - est.mean(), ref - ref.mean()
alpha = (e @ r) / (r @ r)
si_sdr = 10 * np.log10(np.sum((alpha * r) ** 2) / np.sum((e - alpha * r) ** 2))
```

*Mel distance* compares log-mel spectrograms, which ignore phase. The exercise takes the mean absolute difference of $\log_{10}$ mel magnitudes, floored at $10^{-5}$, at four scales from 256-point frames with 20 bands to 2048-point frames with 160 bands, and averages over the scales.

### Work SI-SDR by hand

Take the reference $s = (1, -1, 1, -1)$ and the estimate $\hat s = (2.1, -1.9, 1.9, -2.1)$, which is $2s$ plus the error $(0.1, 0.1, -0.1, -0.1)$. Both already have mean zero.

- $\langle \hat s, s \rangle = 2.1 + 1.9 + 1.9 + 2.1 = 8$, and $\langle s, s \rangle = 4$, so $\alpha = 2$.
- The target is $2s$, with energy $16$, and the distortion is $(0.1, 0.1, -0.1, -0.1)$, with energy $0.04$.
- SI-SDR is $10 \log_{10}(16 / 0.04) = 10 \log_{10} 400 = 26.02$ dB.

The factor of 2 in the estimate costs nothing. Plain SNR against $s$ would count it as error and give about 0 dB.

### Why the two scores disagree

SI-SDR compares waveforms sample by sample, so it punishes any change of phase. A codec trained with adversarial losses is free to rebuild a sound that has the right spectrum but a different phase, and the ear cannot tell. The following table shows what the **Run** measured on 16 clips of LibriSpeech dev-clean, 6 seconds each.

| Codec, codebooks | Bitrate | SI-SDR | Mel distance |
|---|---|---|---|
| EnCodec, 2 | 1.5 kbps | 0.3 dB | 0.396 |
| EnCodec, 8 | 6.0 kbps | 6.6 dB | 0.325 |
| EnCodec, 32 | 24.0 kbps | 10.0 dB | 0.285 |
| Mimi, 1 (semantic only) | 0.14 kbps | $-43.6$ dB | 0.982 |
| Mimi, 8 | 1.1 kbps | 3.2 dB | 0.316 |
| Mimi, 32 | 4.4 kbps | 9.7 dB | 0.246 |

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Mel distance against SI-SDR for EnCodec and Mimi: Mimi with 8 codebooks has lower SI-SDR than EnCodec with 8 but nearly the same mel distance; Mimi with one codebook, at SI-SDR -43.6 dB, lies off the chart">
<line x1="60" y1="230" x2="570" y2="230" class="stroke-muted"/><line x1="60" y1="230" x2="60" y2="40" class="stroke-muted"/>
<line x1="99.23076923076923" y1="230" x2="99.23076923076923" y2="235" class="stroke-muted"/><text x="99.23076923076923" y="250" class="fig-small" text-anchor="middle">0</text>
<line x1="177.69230769230768" y1="230" x2="177.69230769230768" y2="235" class="stroke-muted"/><text x="177.69230769230768" y="250" class="fig-small" text-anchor="middle">2</text>
<line x1="256.1538461538462" y1="230" x2="256.1538461538462" y2="235" class="stroke-muted"/><text x="256.1538461538462" y="250" class="fig-small" text-anchor="middle">4</text>
<line x1="334.6153846153846" y1="230" x2="334.6153846153846" y2="235" class="stroke-muted"/><text x="334.6153846153846" y="250" class="fig-small" text-anchor="middle">6</text>
<line x1="413.0769230769231" y1="230" x2="413.0769230769231" y2="235" class="stroke-muted"/><text x="413.0769230769231" y="250" class="fig-small" text-anchor="middle">8</text>
<line x1="491.53846153846155" y1="230" x2="491.53846153846155" y2="235" class="stroke-muted"/><text x="491.53846153846155" y="250" class="fig-small" text-anchor="middle">10</text>
<line x1="570.0" y1="230" x2="570.0" y2="235" class="stroke-muted"/><text x="570.0" y="250" class="fig-small" text-anchor="middle">12</text>
<line x1="55" y1="230.0" x2="60" y2="230.0" class="stroke-muted"/><text x="51" y="234.0" class="fig-small" text-anchor="end">0.2</text>
<line x1="55" y1="154.0" x2="60" y2="154.0" class="stroke-muted"/><text x="51" y="158.0" class="fig-small" text-anchor="end">0.3</text>
<line x1="55" y1="78.0" x2="60" y2="78.0" class="stroke-muted"/><text x="51" y="82.0" class="fig-small" text-anchor="end">0.4</text>
<circle cx="111.0" cy="81.0" r="6" class="fill-a"/>
<text x="121.0" y="85.0" class="fig-small" text-anchor="start">EnCodec 2</text>
<circle cx="358.2" cy="135.0" r="6" class="fill-a"/>
<text x="358.2" y="157.0" class="fig-small" text-anchor="middle">EnCodec 8</text>
<circle cx="491.5" cy="165.4" r="6" class="fill-a"/>
<text x="491.5" y="153.4" class="fig-small" text-anchor="middle">EnCodec 32</text>
<circle cx="224.8" cy="141.8" r="6" class="fill-b"/>
<text x="224.8" y="163.8" class="fig-small" text-anchor="middle">Mimi 8</text>
<circle cx="479.8" cy="195.0" r="6" class="fill-b"/>
<text x="479.8" y="217.0" class="fig-small" text-anchor="middle">Mimi 32</text>
<text x="68" y="54" class="fig-small">Mimi 1 (SI-SDR -43.6 dB, mel distance 0.982) is off the chart to the left.</text>
<text x="315.0" y="272" class="fig-small" text-anchor="middle">SI-SDR in dB (higher is better)</text>
<text x="60" y="20" class="fig-small">mel distance (lower is better)</text>
</svg>
Figure 3. Five of the table's six measurements, with SI-SDR on one axis and mel distance on the other. Notice that Mimi with 8 codebooks sits well left of EnCodec with 8 on SI-SDR but beside it on mel distance.
```

By mel distance, Mimi at 1.1 kbps matches EnCodec at 6 kbps, and Mimi at 4.4 kbps beats EnCodec at 24 kbps. By SI-SDR, Mimi at 1.1 kbps looks far worse. With only its semantic codebook, Mimi's output shares almost nothing with the waveform, yet it still carries speech: listen to it in the **Run**.

> [!KEY]
> SI-SDR measures waveform fidelity and mel distance measures spectral fidelity. A codec trained to sound right can score poorly on SI-SDR, so report both, and listen.

> [!RECAP]
> - The bitrate is $f_s / \prod_j s_j$ frames per second, times $N_q \times \log_2 K$.
> - Quantizer dropout lets one model serve every bitrate, and the straight-through estimator applies once, to the sum.
> - SI-SDR scores the waveform and mel distance scores the spectrum, so they can disagree.

## Build it step by step

Build the functions in the order the checks test them. The starter provides `mel_filterbank` and `log_mel`, which you built in the mel chapters.

1. **`frame_rate(sample_rate, strides)` and `bitrate(frames_per_second, n_codebooks, codebook_size)`.** Use `math.prod` and `math.log2`. Common mistake: $K$ instead of $\log_2 K$.
2. **`sample_quantizer_counts(batch, n_q, p, generator)`.** Draw counts with `torch.randint(1, n_q + 1, ...)` and coins with `torch.rand`, both from `generator`, and combine with `torch.where`. Common mistake: `torch.randint(1, n_q, ...)`, whose upper bound is exclusive.
3. **`rvq_forward(z, codebooks, n_q)`.** Permute to `(B, T, D)` and loop over stages. Compute distances on detached tensors, gather `q = codebooks[i][idx]`, and mask each stage with `n_q > i`. Add the codebook loss with `sg[r]` and the commitment loss with `sg[q]`, and subtract `sg[q]` from the residual. Return `z + (hard - z).detach()` permuted back. Common mistake: averaging a stage's loss over examples for which it is inactive.
4. **`ToyCodec`.** Build `encoder`, `decoder`, and `codebooks` as the docstring lists. A convolution with kernel $s$ and stride $s$ divides the length by exactly $s$, and a transposed convolution with the same kernel and stride multiplies it back. Common mistake: forgetting to permute the summed codewords from `(B, T, D)` to `(B, D, T)` in `codes_to_audio`.
5. **`si_sdr(estimate, reference)`.** Remove the means, project, and take the ratio in dB. Common mistake: dividing by $\langle \hat s, \hat s \rangle$ instead of $\langle s, s \rangle$.
6. **`mel_distance(x, y, sample_rate, scales)`.** Trim to the shorter length, call the provided `log_mel` at each scale, and average the mean absolute differences.

## Cost and scaling

The encoder and decoder cost grows linearly with the audio length, and the quantizer costs $O(T N_q K D)$ per clip, which is small next to the convolutions. What grows fastest is downstream. A Transformer over the codes pays attention cost proportional to the square of the token count, so a codec's frame rate decides how long a clip a language model can handle. EnCodec has 14.9M parameters and 2,400 tokens per second at full depth, and Mimi has 79.3M parameters and 400.

The full **Run**, 16 clips of 6 seconds through both codecs at 11 settings, takes well under a minute on an A100, mostly loading the two models. Without a GPU it shrinks to 2 clips of 3 seconds.

## Where this breaks

**Mismatched sample rates.** EnCodec and Mimi expect 24 kHz, and LibriSpeech is 16 kHz, so the **Run** resamples by $3/2$ first. A 16 kHz clip fed as if it were 24 kHz plays back fast and high.

**Quantizer dropout left on at evaluation.** Evaluate each $N_q$ on its own, with dropout off, or the scores mix bitrates.

**The toy codec collapses.** With equal weights on reconstruction and the VQ losses, the toy encoder shrinks its latents to a point, and the output goes silent. The **Run** weights a normalized reconstruction error by 100 to prevent it. Real codecs use the loss balancer instead.

## The exercise

You build `frame_rate`, `bitrate`, `sample_quantizer_counts`, `rvq_forward`, the `ToyCodec` class, `si_sdr`, and `mel_distance`. The checks test the codec bitrates, the statistics of quantizer dropout, the two-stage RVQ example, the straight-through gradient, per-example stage masking, the toy codec's exact lengths and code round trip, SI-SDR and its gain invariance, and the mel distance against an independent implementation.

Press **Run** on a GPU worker to encode LibriSpeech clips with `encodec-24khz` at each of its five bandwidths and `mimi` at 1 to 32 codebooks. You see SI-SDR and mel distance plotted against bitrate, and hear the original, both codecs at their lowest and highest bitrates, and Mimi's semantic codebook alone. Without the assets, the **Run** trains your `ToyCodec` on synthetic voices for 20 seconds instead.
