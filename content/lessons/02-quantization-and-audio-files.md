---
title: Quantization and audio files
slug: 02-quantization-and-audio-files
part: "Part 1 — Sound and signals"
summary: Round samples to N bits and measure the 6.02N + 1.76 dB rule, dither them, compand them with mu-law, and write a WAV file byte by byte.
minutes: 60
gpu: false
objectives:
  - Quantize a signal to N bits, and predict and measure its signal-to-noise ratio.
  - Explain what dither buys and what it costs, and add triangular dither.
  - Compand a signal with mu-law and map it to the 256 codes WaveNet predicted.
  - Write and parse a 16-bit PCM WAV file with struct, including chunks you skip.
prereqs:
  - 01-sound-and-sampling
exercise: 02-quantization-and-audio-files
---

# Quantization and audio files

In the lesson on sound and sampling, every sample was a real number. A file
can't store a real number. It stores an integer with a fixed number of bits,
so every sample gets rounded to the nearest of a few thousand levels. That
rounding adds noise, and how much noise depends on the number of bits and on
how loud the signal is.

This lesson measures that noise and the tricks that tame it. **Dither** trades
a little extra hiss for noise that no longer depends on the signal.
**Companding** spends more levels on quiet samples, which is how telephones
fit speech into 8 bits and how WaveNet turned audio generation into a
256-way classification. Then you write a WAV file by hand, so the bytes on
disk hold no surprises.

## What you already know

- A digital signal is a list of samples, taken $f_s$ times a second.
- Full scale is an amplitude of 1.0, and a level in dBFS is $20 \log_{10}$ of an amplitude.
- A power ratio in decibels is $10 \log_{10}$ of the ratio.
- The RMS of a sine of amplitude $A$ is $A / \sqrt{2}$, so its mean square, its power, is $A^2 / 2$.
- `np.round`, `np.clip`, and `astype` convert between floats and integers.

## Quantize to N bits

A **quantizer** rounds each sample to the nearest level on a fixed grid. With
$N$ bits there are $2^N$ levels. For signals in $[-1, 1)$, the levels are a
**step** $\Delta$ apart:

$$
\Delta = \frac{2}{2^N} = 2^{1-N}, \qquad
Q(x) = \Delta \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{\Delta}\right),\, -2^{N-1},\, 2^{N-1} - 1\right)
$$

Read it aloud: divide the sample by the step, round to the nearest whole
number, clip that number to the $2^N$ codes that exist, and multiply by the
step. The codes run from $-2^{N-1}$ to $2^{N-1} - 1$, so the lowest level is
$-1$ and the highest is one step below $+1$. In numpy:

```python
step = 2.0 ** (1 - bits)
codes = np.clip(np.round(x / step), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
y = codes * step
```

### Work a 3-bit example by hand

With 3 bits, $\Delta = 0.25$, and the eight levels are $-1, -0.75, \dots, 0.75$.
Take the samples 0.3, $-0.62$, 0.9, 0.05, and 0.99.

- Divided by the step: 1.2, $-2.48$, 3.6, 0.2, and 3.96.
- Rounded: 1, $-2$, 4, 0, and 4.
- Clipped to $[-4, 3]$: 1, $-2$, 3, 0, and 3.
- Times the step: 0.25, $-0.5$, 0.75, 0, and 0.75.

The errors are 0.05, $-0.12$, 0.15, 0.05, and 0.24. The first, second, and
fourth are rounding errors, never more than half a step. The third and fifth
are larger because those samples hit the top code and were clipped.

## The 6.02N + 1.76 dB rule

When a signal is loud and busy, the rounding error jumps around like noise
that is equally likely to fall anywhere in $[-\Delta/2, \Delta/2]$. A
uniform error on that interval has a mean square of $\Delta^2/12$. A sine of
amplitude $A$ has a power of $A^2/2$. The **signal-to-noise ratio** (SNR) is
the ratio of the two powers, in decibels:

$$
\text{SNR} = 10 \log_{10} \frac{A^2 / 2}{\Delta^2 / 12}
= 10 \log_{10}\!\left(1.5 \cdot 4^N\right) + 20 \log_{10} A
\approx 6.02 N + 1.76 + 20 \log_{10} A \ \text{dB}
$$

Read it aloud: at full scale, the SNR is ten times the log of 1.5 times four
to the power $N$. Each extra bit halves the step, which quarters the noise
power and adds 6.02 dB. A quieter sine loses exactly its level in dBFS.

```python
predicted = 10 * np.log10(1.5 * 4.0 ** bits) + 20 * np.log10(amplitude)
measured = 10 * np.log10(np.sum(x ** 2) / np.sum((quantize(x, bits) - x) ** 2))
```

At full scale, 8 bits gives 49.93 dB, 16 bits gives 98.09 dB, and 24 bits
gives 146.26 dB. A sine of amplitude 0.99 at 8 bits is predicted at 49.84 dB,
and quantizing a 1234.5 Hz one at 16 kHz measures 49.73 dB.

> [!KEY]
> Every bit is worth about 6 dB of SNR, but only for a signal near full scale.
> A signal 20 dB below full scale gets 20 dB less.

Speech is never a full-scale sine. A recording that peaks at $-20$ dBFS has an
RMS level near $-35$ dBFS, 32 dB below the $-3.01$ dBFS of a full-scale sine.
So at 16 bits its SNR is about 66 dB, not 98, and at 8 bits about 18 dB,
which sounds harsh. The rule also assumes
the error behaves like noise, and that fails for quiet or repetitive signals,
which is where dither comes in.

## Dither

Quantize a constant 0.3 steps above zero. It rounds to 0 every time, so the
signal is gone and the error is exactly $-0.3\Delta$ on every sample. A quiet
tone does something similar: its error follows the tone, so you hear it as
distortion at the tone's harmonics rather than as hiss. A fading tone even
cuts off abruptly once it falls below half a step.

**Dither** is a little random noise added before quantizing. It pushes each
sample over or under the rounding threshold at random, in proportion to how
close the sample was. The average output then follows the signal, even below
one step.

**Triangular (TPDF) dither** is the sum of two independent uniform draws on
$[-\Delta/2, \Delta/2]$. It has two useful properties. The mean of the error
is zero whatever the signal is, and its power is the same at every signal
level. The price is noise. The dither adds $\Delta^2/6$ to the $\Delta^2/12$
of rounding, for a total error power of $\Delta^2/4$, three times as much.
That is $10 \log_{10} 3 = 4.77$ dB less SNR.

```python
noise = rng.uniform(-step / 2, step / 2, x.shape) + rng.uniform(-step / 2, step / 2, x.shape)
y = quantize(x + noise, bits)
```

Measured on 200,000 samples at 8 bits, the constant 0.3 steps comes out with
a mean of 0.299 steps once it is dithered, against 0 without. The error power
is within 0.4% of $\Delta^2/4$ at 0, 0.25, and 0.5 steps. A single uniform
draw, called RPDF dither, fixes the mean but not the power: its error power
runs from 0 at a level of 0 to $\Delta^2/4$ at half a step, so the hiss
pumps up and down with the signal.

> [!KEY]
> Dither turns distortion that follows the signal into steady noise that
> doesn't. Triangular dither costs 4.77 dB of SNR and buys an error whose mean
> and power don't depend on the signal.

The following demo lets you lower the bit depth of a quiet tone and switch
dither on and off.

```demo
{"id": "quantization-and-dither", "caption": "Drop to a few bits, then turn on dither and listen for the distortion turning into a steady hiss."}
```

Kaldi's feature extractor adds dither by default, one integer step at 16 bits,
so that digital silence never reaches $\log 0$. torchaudio's Kaldi-compatible
functions turn it off by default, so the two disagree slightly on the same
file.

## Compand with mu-law

Most speech samples are near zero. A uniform quantizer spends as many levels
on the loud extremes as on the quiet middle, so quiet passages get few levels.
**Companding** compresses the signal with a log-like curve before a uniform
quantizer and expands it afterward. The error then scales with the size of
each sample, which keeps the SNR nearly constant as the level drops.

The **mu-law** curve, from the ITU-T G.711 telephone standard, with $\mu = 255$:

$$
F(x) = \operatorname{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \qquad
F^{-1}(y) = \operatorname{sgn}(y) \frac{(1 + \mu)^{|y|} - 1}{\mu}
$$

Read it aloud: $F$ keeps the sign of $x$ and replaces its size with a
logarithm of one plus $\mu$ times the size, scaled so that 1 maps to 1. The
inverse undoes it with a power. Both map $[-1, 1]$ onto $[-1, 1]$.

To get integer codes, map $F(x)$ from $[-1, 1]$ to $[0, \mu]$ and round, the
way `torchaudio.functional.mu_law_encode` does:

```python
y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
codes = np.floor((y + 1) / 2 * mu + 0.5).astype(np.int64)
```

The following table shows five inputs and their 8-bit codes.

| $x$ | $F(x)$ | Code |
|---|---|---|
| 0.001 | 0.0410 | 133 |
| 0.01 | 0.2285 | 157 |
| 0.1 | 0.5910 | 203 |
| 0.5 | 0.8757 | 239 |
| 1.0 | 1.0 | 255 |

The inputs from 0 to 0.01, one percent of the positive range, get codes 128
to 157, which is 12% of all 256 codes. Measured on a 1234.5 Hz sine, 8-bit
mu-law gives 37.7 dB at full scale, 37.6 dB at $-20$ dB, and 33.7 dB at
$-40$ dB. Linear 8-bit gives 49.7, 29.8, and 11.1 dB. On Laplacian samples
at about $-31$ dBFS, a rough stand-in for speech, mu-law wins by 15 dB.

WaveNet used exactly this. It predicts each sample with a softmax, and a
softmax over 65,536 16-bit values was impractical. Its authors compressed
with mu-law and quantized to 256 values, and they reported a much better
reconstruction than 256 linear levels gave. You build that model in the
lesson on WaveNet.

> [!TIP]
> Code 128 decodes to about 0.000086, not to 0, because 256 codes have no
> middle one. Don't test for exact silence after a mu-law round trip.

## Store samples in a WAV file

A **WAV** file stores integer samples, called PCM, for pulse-code modulation.
The common convention converts a float to 16 bits by multiplying by 32768,
rounding, and clipping to $[-32768, 32767]$, and converts back by dividing by
32768. Whisper's audio loader divides by exactly 32768.0.

A WAV file is a RIFF container: a sequence of **chunks**, each a four-letter
id, a 4-byte little-endian size, and that many bytes. A `fmt ` chunk describes
the samples and a `data` chunk holds them. The following table lays out the
44-byte header that the `wave` module and `soundfile` write.

| Bytes | Field | Type | Value for 16-bit PCM |
|---|---|---|---|
| 0 to 3 | Chunk id | 4 letters | `RIFF` |
| 4 to 7 | Size of everything after this field | uint32 | 36 plus the sample bytes |
| 8 to 11 | Format | 4 letters | `WAVE` |
| 12 to 15 | Chunk id | 4 letters | `fmt ` |
| 16 to 19 | Chunk size | uint32 | 16 |
| 20 to 21 | Format tag | uint16 | 1, integer PCM |
| 22 to 23 | Channels | uint16 | 1 or 2 |
| 24 to 27 | Sample rate | uint32 | 16000 |
| 28 to 31 | Bytes per second | uint32 | rate times channels times 2 |
| 32 to 33 | Bytes per frame | uint16 | channels times 2 |
| 34 to 35 | Bits per sample | uint16 | 16 |
| 36 to 39 | Chunk id | 4 letters | `data` |
| 40 to 43 | Sample bytes | uint32 | frames times channels times 2 |

The `struct` module packs these fields. `<` means little-endian, `H` is a
uint16, and `I` is a uint32:

```python
fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * 2 * channels, 2 * channels, 16)
header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
```

For three stereo frames at 16 kHz, the data is 12 bytes and the file is 56.
Bytes 4 to 7 are `30 00 00 00`, which is 48, and bytes 24 to 27 are
`80 3e 00 00`, which is 16,000.

Samples are **interleaved** by channel: left 0, right 0, left 1, right 1. A
numpy array shaped `(frames, channels)` is already in that order, so
`pcm.astype("<i2").tobytes()` writes it directly.

> [!WARNING]
> Files often carry extra chunks, such as `LIST` for metadata, before the
> data. Code that seeks to byte 44 reads metadata as audio. Walk the chunks,
> and remember that a chunk with an odd size is followed by one padding byte.

Other formats trade size for fidelity. **FLAC** is lossless: it predicts each
block with a linear predictor and stores the small residual. **MP3** and
**Opus** are lossy, and their decoders add a delay at the start, so a decoded
clip no longer lines up sample for sample with the original.

> [!RECAP]
> - `quantize` divides by $\Delta = 2^{1-N}$, rounds, clips to the $2^N$ codes, and multiplies back.
> - The predicted SNR of a sine is $10 \log_{10}(1.5 \cdot 4^N) + 20 \log_{10} A$ dB.
> - TPDF dither is two uniform draws on $[-\Delta/2, \Delta/2]$, added before quantizing.
> - mu-law codes are `np.floor((F(x) + 1) / 2 * mu + 0.5)`, and decoding inverts both steps.
> - A 16-bit WAV file is a 44-byte header of little-endian fields, then interleaved int16 samples.

## Build it step by step

The checks test the functions in this order.

1. **`quantize(x, bits)`** returns floats on the grid. Use `np.round` and
   `np.clip`. Common mistake: clipping to $2^{N-1}$ instead of $2^{N-1} - 1$,
   which invents a level at $+1.0$ that no integer format has.
2. **`predicted_snr_db(bits, amplitude)`** returns the formula. Common mistake:
   `6.02 * bits + 1.76`, which is off by up to 0.01 dB at 24 bits.
3. **`tpdf_dither(x, bits, rng)`** adds two `rng.uniform` draws and calls your
   `quantize`. Common mistake: one draw of width $2\Delta$, which is uniform,
   not triangular, and leaves the noise power depending on the signal.
4. **`mu_law_encode(x, mu)`** clips, compands with `np.log1p`, and maps to
   int64 codes. Common mistake: `np.round` instead of `np.floor(... + 0.5)`,
   which rounds 0.5 to the even neighbor and moves a few codes.
5. **`mu_law_decode(codes, mu)`** maps codes to $[-1, 1]$ and expands. Common
   mistake: forgetting to convert the codes to float before dividing.
6. **`float_to_pcm16(x)`** and **`pcm16_to_float(pcm)`** scale by 32768. Clip
   before `astype(np.int16)`. Common mistake: casting without clipping, which
   wraps 1.0 to $-32768$ and makes a loud click.
7. **`wav_bytes(pcm, sample_rate)`** packs the header with `struct.pack` and
   appends `pcm.astype("<i2").tobytes()`. Common mistake: writing 44 plus the
   data length in bytes 4 to 7 instead of 36 plus it.
8. **`read_wav_bytes(data)`** starts at byte 12 and loops over chunks with
   `struct.unpack("<I", ...)`. Use `np.frombuffer(body, dtype="<i2")` and
   reshape to `(-1, channels)`. Common mistake: forgetting the padding byte
   after an odd-sized chunk.

## Cost and scaling

Everything here is one pass over the samples, so time and memory grow as
$O(N)$ for $N$ samples. The bit depth sets the storage, as the following
table shows for one minute of mono audio at 16 kHz.

| Encoding | Bits per sample | Size of one minute | SNR of a full-scale sine |
|---|---|---|---|
| 8-bit mu-law, G.711 | 8 | 0.96 MB | about 38 dB, and nearly as much 20 dB lower |
| 16-bit PCM | 16 | 1.92 MB | 98.09 dB |
| 32-bit float | 32 | 3.84 MB | far beyond hearing |

G.711 sends 8-bit mu-law at 8 kHz, which is 64 kilobits per second. The
neural codecs later in the course send speech at a few kilobits per second.

## Where this breaks

**Clipping hides in the top code.** An amplitude of 0.999 at 8 bits divides to
127.87, which rounds to 128, which clips to 127. The SNR of that sine comes
out about 0.6 dB below the prediction. Leave headroom, or normalize to a peak below
$1 - \Delta/2$.

**Tones that divide the sample rate don't give noise.** A 1 kHz tone at 16 kHz
repeats every 16 samples, so its rounding error repeats too. It lands on
harmonics of 1 kHz instead of spreading out, and the measured SNR can miss the
prediction by more than 1 dB. Dither fixes this.

**Readers disagree on scaling.** `scipy.io.wavfile.read` returns raw integers,
and `soundfile.read` returns floats in $[-1, 1)$. A 24-bit file often reads as
left-justified `int32`, so its scale is $2^{31}$, not $2^{23}$. Check the dtype
of everything you load.

**Layouts disagree too.** `soundfile` returns `(frames, channels)`, while
torchaudio and `speechlab.play` use `(channels, frames)`. Downmix to mono with
the mean over channels, not the sum, or the result can clip.

**Lossy decoders shift the clip.** Don't compute sample-aligned metrics between
an MP3 decode and its source without aligning them first.

## The exercise

You build `quantize`, `predicted_snr_db`, `tpdf_dither`, `mu_law_encode`,
`mu_law_decode`, `float_to_pcm16`, `pcm16_to_float`, `wav_bytes`, and
`read_wav_bytes`. Press **Run** to hear a plucked note at 6 bits, gritty
without dither and cleanly hissing with it, and to see both errors as
spectrograms. Then compare 8-bit linear and mu-law on a quiet copy, and
write, read, and play a stereo WAV file.

The checks confirm the 3-bit example and the 16 levels of 4 bits, the
predicted SNR, and a measured SNR within 0.25 dB of it at 8, 12, and 16 bits.
They confirm that dither recovers a constant of 0.3 steps and keeps the error
power at $\Delta^2/4$ at three levels. They test the mu-law codes in the
table, a round-trip error under 4%, and a 10 dB win over linear on quiet
samples. The last checks compare your PCM scaling on every int16 value, your
file with the `wave` module's byte for byte, and your reader on a
`soundfile` stereo file and on a file with a padded `LIST` chunk.
