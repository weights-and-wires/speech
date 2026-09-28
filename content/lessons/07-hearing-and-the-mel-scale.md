---
title: Hearing and the mel scale
slug: 07-hearing-and-the-mel-scale
part: "Part 2 — Hearing and speech production"
summary: Build the mel scale, a triangular mel filterbank that matches librosa's, and Whisper's exact log-mel front end, from the ear up.
minutes: 65
gpu: false
objectives:
  - Explain how the cochlea analyzes frequency, and why speech features space their bands like it does.
  - Convert between hertz and mel on the HTK and Slaney scales, and say where the two disagree.
  - Build a triangular mel filterbank that matches librosa's to 1e-6, with and without Slaney normalization.
  - "Reproduce Whisper's log-mel spectrogram exactly: 400-point periodic Hann, hop 160, power, 80 Slaney mels, log10 clamp, a floor 8 decades under the maximum, and (x + 4) / 4."
prereqs:
  - 04-the-short-time-fourier-transform
exercise: 07-hearing-and-the-mel-scale
---

# Hearing and the mel scale

A spectrogram from the STFT spaces its frequency bins evenly: at 16 kHz with a
400-sample frame, one bin every 40 Hz, from 0 Hz to 8000 Hz. Your ear does not
work like that. A step from 100 Hz to 200 Hz sounds like a big jump in pitch. A
step from 4000 Hz to 4100 Hz, the same 100 Hz, is barely audible. Most of a
linear spectrogram's bins are spent on high frequencies where hearing is
coarse.

Speech systems fix this with a *mel filterbank*: a few dozen overlapping
filters, narrow at low frequencies and wide at high ones, spaced like the ear's
own frequency analysis. Their outputs, compressed with a logarithm, are the
*log-mel spectrogram*. It is the input to nearly every speech model in this
course, from the classical recognizers of Part 3 to Whisper.

In this lesson you build the mel scale, the filterbank, and Whisper's exact
front end. "Exact" matters: a model trained on one front end sees garbage when
you feed it features made a slightly different way.

## What you already know

- The STFT cuts a signal into overlapping windowed frames and takes the FFT of each; row `k` of a spectrogram is the frequency `k * sample_rate / n_fft`.
- With `center=True`, the signal is padded by `n_fft // 2` samples at each end by reflection, so a signal of $L$ samples has `1 + L // hop` frames.
- The periodic Hann window of length `N` is `0.5 - 0.5 * cos(2 pi n / N)`.
- A decibel (dB) is ten times the base-10 log of a power ratio, so every factor of 10 in power is 10 dB.
- Power is the square of magnitude: `np.abs(X) ** 2`.

## How the ear hears frequency

Sound enters the inner ear at the *cochlea*, a coiled tube. Inside it runs the
*basilar membrane*. The membrane is stiff and narrow at
the entrance and floppy and wide at the far end. A high frequency makes the
stiff end vibrate most; a low frequency travels further and peaks near the far
end. Each place along the membrane responds to a band of frequencies, so the
cochlea works as a bank of band-pass filters laid out by frequency.

Those filters are not evenly spaced. Their bandwidth is about 100 Hz below
500 Hz, and above that it grows roughly in proportion to the center frequency,
about 20% of it. Two tones inside the same band, called a *critical band*,
interact strongly: a loud one can hide a quieter one. This effect is called
*masking*. Around 24 critical bands cover the range of hearing, which runs from
about 20 Hz to 20 kHz.

Three more facts about hearing shape the features you build:

- Hearing is most sensitive around 3 kHz to 4 kHz, and much less sensitive at
  low frequencies.
- Perceived loudness grows roughly with the logarithm of power, not with power.
- The ear is fairly insensitive to the phase of steady sounds, so features
  keep power and throw phase away.

> [!KEY]
> A speech front end imitates the ear in three ways: bands that widen with
> frequency, a logarithm for loudness, and no phase.

## The mel scale

The *mel scale* is a pitch scale built from listening experiments in the 1930s:
listeners adjusted one tone until it sounded half as high as another. Equal
steps in mel sound like equal steps in pitch. Below about 1 kHz the scale is
close to linear in hertz; above it, close to logarithmic. Two formulas for it
dominate speech code, and they are not interchangeable.

### The HTK formula

The HTK formula, used by HTK, Kaldi, and `torchaudio`, is one logarithm:

$$
\begin{aligned}
m_{\text{HTK}}(f) &= 2595 \log_{10}\left(1 + \frac{f}{700}\right) \\
f &= 700\left(10^{m / 2595} - 1\right)
\end{aligned}
$$

Read it aloud: divide the frequency $f$ in hertz by 700, add one, take the
base-10 log, and scale by 2595. The second formula runs it backward. Kaldi
writes the same curve as $1127 \ln(1 + f / 700)$, because
$1127 \ln 10 \approx 2595$. The constants make 1000 Hz land at about 1000 mel.

```python
mel = 2595.0 * np.log10(1.0 + f / 700.0)
hz = 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
```

### The Slaney formula

The Slaney formula, from Malcolm Slaney's Auditory Toolbox, is the default in
librosa and therefore in Whisper. It is exactly linear below 1000 Hz and
exactly logarithmic above:

$$
m_{\text{Slaney}}(f) =
\begin{cases}
\dfrac{3f}{200}, & f < 1000 \\[2mm]
15 + \dfrac{27 \ln(f / 1000)}{\ln 6.4}, & f \ge 1000
\end{cases}
$$

Read it aloud: below 1 kHz, every 200/3 Hz (about 66.7 Hz) is one mel, so
1000 Hz is 15 mel. Above 1 kHz, every factor of 6.4 in frequency adds 27 mel.
The log step, $\ln(6.4)/27 = 0.068752$, is the natural-log change per mel.

```python
log_branch = 15.0 + np.log(np.maximum(f, 1000.0) / 1000.0) / (np.log(6.4) / 27.0)
mel = np.where(f >= 1000.0, log_branch, f / (200.0 / 3.0))
```

### Work the two scales by hand

Take $f = 440$ Hz. On the HTK scale, $1 + 440/700 = 1.628571$, its base-10 log
is $0.211807$, and $2595 \times 0.211807 = 549.64$ mel. On the Slaney scale,
440 Hz is below 1 kHz, so $3 \times 440 / 200 = 6.60$ mel. The following table
lists more values. You will check your code against it.

| Frequency (Hz) | HTK (mel) | Slaney (mel) |
|---|---|---|
| 100 | 150.49 | 1.50 |
| 440 | 549.64 | 6.60 |
| 1000 | 999.99 | 15.00 |
| 2000 | 1521.36 | 25.08 |
| 4000 | 2146.06 | 35.16 |
| 8000 | 2840.02 | 45.25 |

The absolute numbers differ by a factor of about 60, which does not matter: a
filterbank uses only equal spacing on the scale. The shape matters. The two
curves place filter centers differently below 1 kHz.

Back to the opening experiment: 100 Hz to 200 Hz is 1.5 mel to 3.0 mel on the
Slaney scale, a step of 1.5 mel. 4000 Hz to 4100 Hz is 35.16 to 35.52 mel, a
step of 0.36 mel. The scale agrees with your ear.

## A triangular mel filterbank

A *mel filterbank* is a matrix. Each row is one filter: a triangle of weights
over the FFT bins. Multiply it by a power spectrogram and each row sums the
power under its triangle, turning 201 bins into 80 bands.

To place the triangles, pick $n_{\text{mels}} + 2$ edge frequencies equally
spaced in mel from $f_{\min}$ to $f_{\max}$, and convert them back to hertz:
$f_0 < f_1 < \dots < f_{n+1}$. Filter $i$ starts at $f_i$, peaks at $f_{i+1}$,
and ends at $f_{i+2}$, so neighbors overlap by half. Its weight at the bin
frequency $f$ is the following:

$$
W_i(f) = \max\left(0,\ \min\left(\frac{f - f_i}{f_{i+1} - f_i},\ \frac{f_{i+2} - f}{f_{i+2} - f_{i+1}}\right)\right)
$$

Read it aloud: the first fraction is a ramp that rises from 0 at $f_i$ to 1 at
$f_{i+1}$. The second is a ramp that falls from 1 at $f_{i+1}$ to 0 at
$f_{i+2}$. The smaller of the two is the triangle, and the $\max$ with 0 cuts
off the negative parts outside it. The ramps are straight lines in hertz, not
in mel: the edges are mel-spaced, but librosa draws each triangle in hertz.

Two normalizations are common:

- `norm=None` leaves every peak at 1. This is the `torchaudio` default.
- `norm="slaney"` multiplies filter $i$ by $2 / (f_{i+2} - f_i)$, one over half
  its width, so every triangle has an area of 1 in hertz. Wide high filters no
  longer collect more power only because they are wide. This is the librosa
  default, and Whisper's.

With broadcasting, the whole bank is a few lines:

```python
lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
rising = (bin_hz[None, :] - lower) / (center - lower)
falling = (upper - bin_hz[None, :]) / (upper - center)
weights = np.maximum(0.0, np.minimum(rising, falling))
```

### Build a two-filter bank by hand

Take a sample rate of 8000 Hz, `n_fft=8`, and two Slaney mel filters from 0 Hz
to 4000 Hz. The bins sit at 0, 1000, 2000, 3000, and 4000 Hz.

1. The top edge is $m(4000) = 35.1638$ mel. Four edges equally spaced in mel
   are 0, 11.7213, 23.4425, and 35.1638.
2. Back to hertz: 11.7213 mel is below 15, so it is
   $11.7213 \times 200/3 = 781.42$ Hz. 23.4425 mel is above 15, so it is
   $1000\,e^{(23.4425 - 15) \times 0.068752} = 1786.82$ Hz. The edges are
   0, 781.42, 1786.82, and 4000 Hz.
3. Filter 0 runs from 0 Hz through 781.42 Hz to 1786.82 Hz. Only the bin at
   1000 Hz is inside, on the falling side:
   $(1786.82 - 1000) / (1786.82 - 781.42) = 0.7826$.
4. Filter 1 runs from 781.42 Hz through 1786.82 Hz to 4000 Hz. At 1000 Hz it is
   rising: $(1000 - 781.42) / 1005.40 = 0.2174$. At 2000 Hz and 3000 Hz it is
   falling: $2000 / 2213.18 = 0.9037$ and $1000 / 2213.18 = 0.4518$.
5. Slaney normalization multiplies filter 0 by $2 / 1786.82 = 0.0011193$ and
   filter 1 by $2 / 3218.58 = 0.00062139$.

The peak-1 bank is `[[0, 0.7826, 0, 0, 0], [0, 0.2174, 0.9037, 0.4518, 0]]`.
Notice that 0.7826 + 0.2174 = 1: between the first and last centers, peak-1
triangles always add up to one.

The following demo draws the filters over the FFT bins.

```demo
{"id": "mel-filterbank", "caption": "Raise the number of mel bands and switch the formula to HTK, and watch the lowest triangles narrow until the fewest bins in a band reads 1 and empty bands appear."}
```

At Whisper's settings, 16 kHz and `n_fft=400`, the bins are 40 Hz apart. The
lowest Slaney edges are only 37.2 Hz apart for 80 mels, and 23.4 Hz apart for
128 mels. So the lowest filters touch one bin each: they sample the spectrum
rather than average it. The top filter, near 8 kHz, spans 14 bins.

## Whisper's log-mel spectrogram

Whisper is OpenAI's speech recognizer. The following table compares its front
end with three others you will meet. A mismatch in any row changes the numbers.

| Setting | Whisper | librosa `melspectrogram` | `torchaudio` `MelSpectrogram` | Kaldi `fbank` |
|---|---|---|---|---|
| Frame and hop | 400 and 160 samples | 2048 and 512 | 400 and 200 | 400 (FFT of 512) and 160 |
| Window | periodic Hann | periodic Hann | periodic Hann | Povey |
| Padding | center, reflect | center, zeros | center, reflect | none |
| Mel scale and norm | Slaney, `"slaney"` | Slaney, `"slaney"` | HTK, none | HTK, none |
| Log | $\log_{10}$, clamp $10^{-10}$ | none | none | $\ln$ |

Whisper fixes the sample rate at 16 kHz. A frame of 400 samples is 25 ms, and
a hop of 160 samples is 10 ms, so there are 100 frames per second. Its pipeline
has five steps:

1. The STFT with a 400-point periodic Hann window, hop 160, and `center=True`
   reflect padding, as in *The short-time Fourier transform*.
2. Drop the last frame. A clip of $L$ samples has $1 + \lfloor L/160 \rfloor$
   frames; Whisper keeps $\lfloor L/160 \rfloor$.
3. Take the power $|X|^2$ and multiply by the 80-filter Slaney bank
   (128 filters for large-v3). Call the result $M$.
4. Compress it with a logarithm and a floor.
5. Rescale it.

Steps 4 and 5 are the following:

$$
\begin{aligned}
L &= \log_{10} \max\left(M, 10^{-10}\right) \\
L' &= \max\left(L,\ \max_{b,t} L_{b,t} - 8\right) \\
Y &= \frac{L' + 4}{4}
\end{aligned}
$$

Read it aloud: clamp every mel power to at least $10^{-10}$ and take its base-10
log. Find the largest value anywhere in the matrix, over every band $b$ and
frame $t$, and raise anything more than 8 below it up to that floor. Eight
decades of power is 80 dB. Then add 4 and divide by 4, so values land roughly
between $-1$ and $1.5$.

```python
log_spec = np.log10(np.maximum(mel_power, 1e-10))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
features = (log_spec + 4.0) / 4.0
```

### Work the normalization by hand

Take a mel power matrix of two bands and three frames,
`[[1, 1e-3, 0], [1e-5, 10, 1e-12]]`.

1. Clamp and take $\log_{10}$: `[[0, -3, -10], [-5, 1, -10]]`. The zero and
   the $10^{-12}$ both became $10^{-10}$ first.
2. The global maximum is 1, so the floor is $1 - 8 = -7$:
   `[[0, -3, -7], [-5, 1, -7]]`.
3. Add 4 and divide by 4: `[[1, 0.25, -0.75], [-0.25, 1.25, -0.75]]`.

Whisper reads audio in 30-second chunks: 480,000 samples, padded with zeros at
the end when the clip is shorter. That is 3001 STFT frames, minus the last,
3000. The encoder's first layers include a convolution with stride 2, so the
transformer sees 1500 positions, one every 20 ms.

> [!KEY]
> Whisper's features are the Slaney-normalized 80-band mel power of a 25 ms
> Hann STFT every 10 ms, in $\log_{10}$, floored 8 decades under the loudest
> value in the whole input, then mapped by $(x + 4) / 4$.

> [!RECAP]
> - The ear's filters widen with frequency; the mel scale spaces bands to match.
> - HTK mel is $2595 \log_{10}(1 + f/700)$. Slaney mel is $3f/200$ below 1 kHz and logarithmic above, with 1 kHz at 15 mel.
> - A filterbank has $n_{\text{mels}} + 2$ mel-spaced edges; each triangle is straight in hertz, and Slaney normalization gives it unit area.
> - Whisper drops the last STFT frame, uses power, and floors at the global maximum minus 8 before $(x + 4)/4$.

## Build it step by step

The checks test the functions in this order.

1. **`hz_to_mel(f, htk=False)`** takes a scalar or array in hertz and returns
   mel as a float array. Branch on `htk`. For Slaney, compute both branches
   and choose with `np.where`. The usual mistake is writing `f / 200 * 3` for
   the whole range and forgetting the log branch above 1 kHz.
2. **`mel_to_hz(m, htk=False)`** inverts it. The Slaney break point is at
   15 mel. Test it by round-tripping: `mel_to_hz(hz_to_mel(f))` must return `f`
   to 1e-9. The usual mistake is switching branches at 1000 mel instead of
   15.
3. **`mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, htk, norm)`**
   returns `(n_mels, n_fft // 2 + 1)`. Build the edges with `np.linspace` in
   mel, convert to hertz, and build every triangle by broadcasting. The
   usual mistake is drawing the ramps in mel instead of hertz, which gives
   curved triangles that miss librosa's peak-1 filters by up to 5e-3.
4. **`power_spectrogram(x, n_fft, hop)`** returns
   `(n_fft // 2 + 1, 1 + len(x) // hop)`. It is the centered, reflect-padded,
   periodic-Hann STFT from *The short-time Fourier transform*, squared. The
   usual mistake is `np.hanning(n_fft)`, which is the symmetric window, not
   the periodic one.
5. **`whisper_normalize(mel_power)`** applies the clamp, the $\log_{10}$, the
   global floor, and the rescale. The usual mistake is a per-frame floor,
   `log_spec.max(axis=0)`.
6. **`pad_or_trim(audio, length)`** zero-pads at the end with `np.pad` or
   slices to exactly `length` samples.
7. **`whisper_log_mel(audio, n_mels)`** returns `(n_mels, len(audio) // 160)`
   and chains your functions in the order of Whisper's five steps. The usual
   mistake is keeping 3001 frames for 30 seconds.

## Cost and scaling

For a clip of $L$ samples with hop $H$, FFT size $N$, and $B$ mel bands, the
cost of each step grows as the following table shows.

| Step | Time | Memory |
|---|---|---|
| The filterbank | $O(B N)$, once | $O(B N)$ |
| The STFT | $O((L/H)\, N \log N)$ | $O(L N / H)$ |
| The mel product | $O(B \cdot (N/2) \cdot L / H)$ | $O(B L / H)$ |
| The log and floor | $O(B L / H)$ | $O(B L / H)$ |

For one 30-second Whisper chunk, the STFT is 3001 FFTs of 400 points, and the
mel product is about 48 million multiply-adds. Both take milliseconds on a
laptop. The result is 240,000 values for 480,000 samples: the front end halves
the data and throws away phase.

## Where this breaks

**Magnitude instead of power.** Whisper and librosa square the magnitude.
Without the square, the features have about half the dynamic range, and the
filters weight the bins differently.

**The wrong logarithm.** $\ln$ and $\log_{10}$ differ by a factor of 2.303.
Kaldi uses $\ln$; Whisper uses $\log_{10}$.

**A per-frame floor.** Whisper floors against the loudest value in the whole
30-second chunk. A per-frame floor lifts silent frames into noise.

**Off-by-one frames.** A centered STFT of 480,000 samples has 3001 frames.
Whisper's encoder expects 3000.

**The wrong sample rate.** `librosa.load` resamples to 22,050 Hz unless you
pass `sr=None` or `sr=16000`. Whisper expects 16 kHz, and features from
another rate put every formant in the wrong band.

**Too many filters for the FFT.** At `n_fft=400`, 128 HTK filters leave
filters 0, 3, 6, and 13 with no bins at all, so those bands are always zero.
The Slaney scale spaces the lowest filters further apart and avoids this.
Use a larger FFT when you need many narrow low bands.

> [!WARNING]
> Kaldi feeds its front end integers on the 16-bit scale, from -32768 to
> 32767. Floats in $[-1, 1]$ shift every natural-log mel value by
> $\ln(32768^2) = 20.79$. Scale the input the way the model was trained.

## The exercise

You build `hz_to_mel`, `mel_to_hz`, `mel_filterbank`, `power_spectrogram`,
`whisper_normalize`, `pad_or_trim`, and `whisper_log_mel`. Press **Run** to
hear the same 100 Hz step low and high, see Whisper's 80 filters, and see
Whisper's log-mel of a linear sweep, which bends on the mel axis.

The checks verify the mel table and the exact inverse. They compare your
filterbank with numbers from `librosa.filters.mel` and with an independent
construction for both scales and both normalizations. They compare your power
spectrogram with `torch.stft`, work the normalization example, and match
`whisper_log_mel` to Whisper's own `log_mel_spectrogram` within 1e-4, for 80
and 128 mels and for a padded chunk of shape `(80, 3000)`.
