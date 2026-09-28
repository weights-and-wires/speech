---
title: MFCCs and normalization
slug: 08-mfccs-and-normalization
part: "Part 2 — Hearing and speech production"
summary: Turn log-mel energies into MFCCs with a DCT, add liftering and deltas, and normalize each utterance so the microphone cancels out.
minutes: 55
gpu: false
objectives:
  - Build the orthonormal DCT-II from its formula, and explain why it decorrelates log-mel energies.
  - Compute MFCCs that match librosa's, and say how they differ from HTK's and Kaldi's.
  - Apply HTK sinusoidal liftering and regression deltas with repeated edge frames.
  - Explain why a fixed channel adds a constant to every cepstral frame, and remove it with cepstral mean and variance normalization.
prereqs:
  - 07-hearing-and-the-mel-scale
exercise: 08-mfccs-and-normalization
---

# MFCCs and normalization

A log-mel spectrogram describes each 10 ms frame of speech with 40 or 80
numbers. Neighboring bands are strongly correlated: when band 12 is loud, band
13 almost always is too. For a neural network that does not matter much. For
the classical recognizers of Part 3, which model each frame with Gaussians
that assume independent dimensions, it matters a lot.

*Mel-frequency cepstral coefficients* (MFCCs) fix it with one more transform.
A discrete cosine transform of the log-mel vector turns 40 correlated numbers
into a handful of nearly independent ones, ordered from the broad shape of the
spectrum to its fine ripples. Keep the first 13, and you have the speech
feature that recognizers used from 1980 until deep learning.

MFCCs also expose a useful fact. A microphone, a telephone line, or a room adds
the same constant to every frame's cepstrum. Subtract each utterance's average
and the channel cancels. In this lesson you build the DCT, the MFCCs, the two
classical additions, liftering and deltas, and that normalization.

## What you already know

- A mel filterbank turns a power spectrogram of shape `(bins, frames)` into a mel power matrix of shape `(n_mels, frames)`.
- A decibel is $10 \log_{10}$ of a power ratio.
- Convolving a signal with a filter multiplies its spectrum by the filter's frequency response.
- A logarithm turns a product into a sum: $\log(ab) = \log a + \log b$.
- A matrix with orthonormal rows keeps the length of every vector it multiplies.

## The cepstrum idea

Look at the log-mel values of one voiced frame, plotted against band number.
You see a smooth curve with a few broad humps, the *spectral envelope*, which
the vocal tract shapes and which says which vowel is being spoken. On top of it,
in the low bands, sits a faster ripple from the individual harmonics of the
voice's pitch.

A second transform can separate the two. Treat the log-mel vector as a signal
and ask which "frequencies" it contains. The smooth envelope lives in the
slowest components, the ripple in faster ones. This spectrum of a log spectrum
is called a *cepstrum*, an anagram of "spectrum", and its axis is called
*quefrency*. MFCCs are a cepstrum computed on mel bands, with a cosine
transform instead of a Fourier transform.

## The discrete cosine transform

The *discrete cosine transform* (DCT) writes a vector as a sum of cosines of
increasing frequency. Its type-II form with orthonormal scaling is the one
every MFCC library uses. For log-mel energies $L_m$, with $m = 0, \dots, M-1$:

$$
c_n = \sqrt{\frac{2 - \delta_{n0}}{M}} \sum_{m=0}^{M-1} L_m \cos\left(\frac{\pi n (2m + 1)}{2M}\right),
\qquad n = 0, \dots, M - 1
$$

Read it aloud: coefficient $c_n$ is the correlation of the log-mel vector with a
cosine that makes $n$ half-turns across the $M$ bands. The cosine is sampled at
the band centers $m + \tfrac{1}{2}$. The scale factor is $\sqrt{1/M}$ for
$n = 0$ and $\sqrt{2/M}$ for every other $n$; $\delta_{n0}$ is 1 when $n = 0$
and 0 otherwise. That scale makes the transform *orthonormal*: the rows of the
cosine matrix are perpendicular unit vectors, so the DCT keeps the sum of
squares, and its transpose is its inverse.

As a matrix, the DCT is one line of broadcasting and one product:

```python
basis = np.cos(np.pi * n[:, None] * (2 * m[None, :] + 1) / (2 * M))
coeffs = (scale[:, None] * basis) @ log_mel      # (M, M) @ (M, frames)
```

This equals `scipy.fft.dct(log_mel, type=2, norm="ortho", axis=0)`.

### Work a four-band DCT by hand

Take $L = [4, 2, 2, 0]$, so $M = 4$.

- $c_0 = \sqrt{1/4}\,(4 + 2 + 2 + 0) = 0.5 \times 8 = 4$. Coefficient 0 is the
  mean level times $\sqrt{M}$.
- For $c_1$, the cosines at $n = 1$ are $\cos(\pi/8), \cos(3\pi/8), \cos(5\pi/8), \cos(7\pi/8)$
  $= [0.9239, 0.3827, -0.3827, -0.9239]$. The sum is
  $4(0.9239) + 2(0.3827) - 2(0.3827) - 0 = 3.6955$, and
  $c_1 = \sqrt{2/4} \times 3.6955 = 2.6131$. A positive $c_1$ means energy tilts
  toward the low bands.
- At $n = 2$ the cosines are $[0.7071, -0.7071, -0.7071, 0.7071]$, and
  $4 - 2 - 2 + 0 = 0$ times 0.7071 gives $c_2 = 0$.
- At $n = 3$ they are $[0.3827, -0.9239, 0.9239, -0.3827]$, which gives
  $c_3 = 0.7071 \times 1.5307 = 1.0824$.

The result is $[4, 2.6131, 0, 1.0824]$. Check the energy: $16 + 4 + 4 + 0 = 24$,
and $16 + 6.8284 + 0 + 1.1716 = 24$. Now keep only $c_0$ and $c_1$ and invert:
you get $[3.707, 2.707, 1.293, 0.293]$, a straight-ish slope through the
original. Truncating the cepstrum smooths the spectrum.

> [!KEY]
> The DCT of a log-mel vector orders it from broad shape to fine detail.
> Keeping the first 13 coefficients keeps the spectral envelope and drops the
> harmonics.

## MFCCs

With the DCT in hand, an MFCC frame takes three steps. From a mel power matrix:

1. Convert to decibels: $10 \log_{10} \max(M, 10^{-10})$, floored 80 dB below
   the loudest value, as `librosa.power_to_db` does.
2. Take the DCT-II down the mel axis.
3. Keep the first 13 rows.

```python
db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
db = np.maximum(db, db.max() - 80.0)
mfccs = dct2_ortho(db)[:13]
```

The numbers depend on conventions. The following table lists them for the
libraries you are most likely to compare against.

| Library | Log | Coefficients kept | DCT |
|---|---|---|---|
| librosa `feature.mfcc` | dB, $10 \log_{10}$, 80 dB floor | 20 | type II, orthonormal |
| `torchaudio` `MFCC` | dB, 80 dB floor | 40 | type II, orthonormal |
| HTK and Kaldi | $\ln$ | 13 | type II |

A dB value is $10 / \ln 10 = 4.343$ times a natural-log value, so librosa's
MFCCs are about 4.3 times Kaldi's, before the other differences. Coefficient
$c_0$ tracks overall loudness, and HTK and Kaldi often replace it with the
frame's log energy.

Why did this work so well for so long? Davis and Mermelstein showed in 1980
that MFCCs beat other features for word recognition. The DCT nearly
decorrelates the log-mel bands, which made it practical to model each frame
with Gaussians whose covariance matrices are diagonal: 13 variances instead of
91 covariance entries.

The following demo shows one frame at every stage of the pipeline.

```demo
{"id": "mfcc-pipeline", "caption": "Lower the number of coefficients kept and watch the red curve rebuilt from the cepstrum smooth away detail until only the vowel's broad formant shape is left."}
```

## Liftering

The higher MFCCs are much smaller than the low ones. For a distance between
frames, such as the template matching of Part 3, the low coefficients would
dominate. HTK *lifters* them, an anagram of "filter", by scaling coefficient
$n$ with a raised sine:

$$
\tilde c_n = \left(1 + \frac{L}{2} \sin\frac{\pi n}{L}\right) c_n
$$

Read it aloud: each coefficient is multiplied by one plus $L/2$ times a sine
that rises from 0 at $n = 0$ to its peak at $n = L/2$. With the usual $L = 22$,
$c_0$ keeps weight 1, $c_1$ gets $1 + 11 \sin(\pi/22) = 2.565$, and $c_{11}$
gets 12.

```python
weights = 1.0 + (L / 2.0) * np.sin(np.pi * np.arange(n_ceps) / L)
liftered = c * weights[:, None]
```

librosa's `lifter` argument uses $n = 1, \dots, N$ instead of $0, \dots, N-1$,
so it scales $c_0$ by 2.565 as well. Check which convention a reference uses.

## Deltas

A single frame says nothing about motion. Yet how the spectrum moves, such as a
formant sliding into a consonant, carries much of speech. *Delta* features add
the local slope of each coefficient over time, fitted by least squares over a
window of $\Theta$ frames on each side:

$$
d_t = \frac{\sum_{\theta=1}^{\Theta} \theta \left(c_{t+\theta} - c_{t-\theta}\right)}{2 \sum_{\theta=1}^{\Theta} \theta^2}
$$

Read it aloud: at frame $t$, take the difference between the frame $\theta$
ahead and the frame $\theta$ behind, weight it by $\theta$, add those up, and
divide by twice the sum of squares of the weights. For a straight line with
slope $s$, every difference is $2\theta s$, so the result is exactly $s$. Past
either end, the first or last frame repeats. *Delta-deltas* apply the same
formula to the deltas. Thirteen MFCCs, 13 deltas, and 13 delta-deltas are the
classic 39-dimensional feature.

```python
padded = np.pad(c, ((0, 0), (width, width)), mode="edge")
for theta in range(1, width + 1):
    num += theta * (padded[:, width + theta : width + theta + T] - padded[:, width - theta : width - theta + T])
```

### Work the deltas by hand

Take one coefficient over five frames, $c = [1, 2, 4, 7, 11]$, and $\Theta = 2$,
so the denominator is $2(1 + 4) = 10$. Padding repeats the ends:
$[1, 1, \mathbf{1, 2, 4, 7, 11}, 11, 11]$.

- $d_0 = [1(2 - 1) + 2(4 - 1)] / 10 = 0.7$
- $d_1 = [1(4 - 1) + 2(7 - 1)] / 10 = 1.5$
- $d_2 = [1(7 - 2) + 2(11 - 1)] / 10 = 2.5$
- $d_3 = [1(11 - 4) + 2(11 - 2)] / 10 = 2.5$
- $d_4 = [1(11 - 7) + 2(11 - 4)] / 10 = 1.8$

## Cepstral mean and variance normalization

Record the same sentence through two microphones and the MFCCs differ. The
difference is simple, though. A fixed channel with frequency response $H(f)$
multiplies the power spectrum by $|H(f)|^2$. If $H$ changes slowly across each
mel band, the mel power of band $b$ is multiplied by about $|H(f_b)|^2$, and the
log turns that product into a sum: $\log M'_{b,t} = \log M_{b,t} + \log |H(f_b)|^2$.
The DCT is linear, so every frame's cepstrum gains the same constant vector.

*Cepstral mean normalization* (CMN) subtracts each coefficient's average over
the utterance, and the constant goes with it. *Cepstral mean and variance
normalization* (CMVN) also divides by each coefficient's standard deviation:

$$
\begin{aligned}
\hat c_t &= \frac{c_t - \mu}{\sigma} \\
\mu &= \frac{1}{T} \sum_{t=1}^{T} c_t \\
\sigma^2 &= \frac{1}{T} \sum_{t=1}^{T} (c_t - \mu)^2
\end{aligned}
$$

Read it aloud: for each coefficient separately, subtract its mean over the $T$
frames of the utterance, and divide by its standard deviation over the same
frames. Afterward, every row has mean 0 and variance 1.

```python
normalized = (c - c.mean(axis=1, keepdims=True)) / np.maximum(c.std(axis=1, keepdims=True), eps)
```

For one row $[2, 4, 6, 8]$: the mean is 5, the variance is
$(9 + 1 + 1 + 9)/4 = 5$, and $\sigma = 2.2361$. The normalized row is
$[-1.342, -0.447, 0.447, 1.342]$.

Kaldi applies the mean part by default and the variance part as an option. It
normalizes per utterance, per speaker, or over a sliding window of 300 to
600 frames for live audio, where the whole utterance is not available yet.

> [!KEY]
> A fixed channel adds a constant to every frame's cepstrum. Subtracting the
> utterance mean removes it.

> [!RECAP]
> - The orthonormal DCT-II has rows $\sqrt{(2 - \delta_{n0})/M}\cos(\pi n (2m + 1)/2M)$ and keeps energy.
> - MFCCs are the first 13 DCT coefficients of log-mel energies in dB.
> - The HTK lifter scales row $n$ by $1 + (L/2)\sin(\pi n / L)$, with $n$ from 0.
> - Deltas are least-squares slopes over $\pm\Theta$ frames with repeated edges; a ramp gives its slope.
> - CMVN makes every row zero-mean and unit-variance over the utterance.

## Build it step by step

The checks test the functions in this order.

1. **`dct2_ortho(x)`** takes `(M,)` or `(M, T)` and returns the same shape.
   Build the `(M, M)` cosine matrix, scale its rows, and multiply. The usual
   mistake is scaling row 0 by $\sqrt{2/M}$ like the others, which breaks
   orthonormality.
2. **`power_to_db(S, amin, top_db)`** returns $10\log_{10}\max(S, a_{\min})$
   with the global floor. The usual mistake is $20\log_{10}$, which is for
   magnitudes, not powers.
3. **`mfcc(power_mel, n_mfcc, top_db)`** chains `power_to_db`, `dct2_ortho`,
   and a slice `[:n_mfcc]`. The usual mistake is taking the DCT along the time
   axis.
4. **`lifter(c, L)`** multiplies row $n$ by its weight, with $n$ starting at 0.
   The usual mistake is librosa's convention, starting at 1.
5. **`deltas(c, width)`** pads with `mode="edge"` and sums weighted
   differences of shifted slices. The usual mistake is zero padding, which
   makes the first and last deltas jump.
6. **`cmvn(c, variance, eps)`** normalizes each row with `keepdims=True`. The
   usual mistake is normalizing each frame across coefficients, axis 0,
   instead of each coefficient across time.
7. **`mfcc_features(x, sample_rate, n_mfcc)`** returns `(3 * n_mfcc, frames)`:
   liftered MFCCs from the provided `mel_power`, their deltas, and their
   delta-deltas, stacked and normalized with CMVN. The usual mistake is taking
   the delta-deltas from the MFCCs instead of from the deltas.

## Cost and scaling

For $T$ frames, $M$ mel bands, and $N_c$ kept coefficients, the steps cost the
following.

| Step | Time | Memory |
|---|---|---|
| DCT as a matrix product | $O(N_c M T)$ | $O(M^2)$ for the matrix |
| Liftering | $O(N_c T)$ | $O(N_c T)$ |
| Deltas with window $\Theta$ | $O(\Theta N_c T)$ | $O(N_c T)$ |
| CMVN | $O(N_c T)$, two passes | $O(N_c)$ statistics |

Everything here is cheap next to the STFT that produces the mel power. An hour
of audio at 100 frames per second is 360,000 frames of 39 numbers, about
110 MB as float64. Only live normalization changes the picture, because it
cannot see the future: it keeps running sums over a window instead.

## Where this breaks

**Short utterances.** The mean of one second of speech depends on which
sounds it holds, not only on the channel. CMN then removes real information.
Normalize per speaker or over a longer window.

**Liftering, then variance normalization.** CMVN divides each row by its own
standard deviation, which cancels any fixed per-row scale, lifter included.
Liftering matters only for raw distances or mean-only normalization.

**Additive noise.** CMN removes channels that multiply the spectrum. Noise
adds to the power, which is not a constant after the log, and CMN does not
remove it.

**Silence and the floor.** The log of a silent band is the floor value. If
one recording is quieter, more bands hit the floor, and the channel stops
being a constant shift.

**Mismatched deltas.** librosa's `feature.delta` fits a Savitzky-Golay filter
over 9 frames and interpolates at the edges. On a ramp it gives the slope
everywhere, where the HTK formula gives 0.5 and 0.8 at the first two frames.

> [!WARNING]
> Compute CMVN statistics per utterance at training time and the same way at
> test time. Normalizing training data per speaker and test data per
> utterance shifts every feature the model sees.

## The exercise

You build `dct2_ortho`, `power_to_db`, `mfcc`, `lifter`, `deltas`, `cmvn`,
and `mfcc_features`; `mel_power` is provided. Press **Run** to hear three
vowels clean and through a telephone band, see their MFCCs and their 39
normalized features, and see how much mean normalization shrinks the gap
between the two recordings.

The checks compare your DCT with `scipy.fft.dct` and confirm it is
orthonormal. They work the `power_to_db` floor, match your MFCCs to numbers
from `librosa.feature.mfcc`, and verify the HTK lifter weights. They compare
your deltas with a brute-force loop and the hand example, and check that
CMVN gives zero mean and unit variance and handles a constant row. The last
two checks convolve a voice with a short filter and require mean
normalization to remove at least 90% of the MFCC difference, and assemble the
39-dimensional features.
