---
title: Speech production and linear prediction
slug: 09-speech-production-and-lpc
part: "Part 2 — Hearing and speech production"
summary: Synthesize vowels from a pulse train and formant resonators, then recover the formants with LPC and the pitch with the cepstrum.
minutes: 70
gpu: false
objectives:
  - Explain the source-filter model of speech, and predict a neutral vowel's formants from a uniform tube.
  - Build a formant resonator from its frequency and bandwidth, and a vowel synthesizer from a cascade of them.
  - Estimate an all-pole vocal tract with autocorrelation LPC and the Levinson-Durbin recursion.
  - Read formants from the angles and radii of the LPC poles, and pitch from the peak of the real cepstrum.
prereqs:
  - 06-filters-convolution-and-reverb
  - 08-mfccs-and-normalization
exercise: 09-speech-production-and-lpc
---

# Speech production and linear prediction

When you say "ah", your vocal folds snap open and shut about 120 times a
second, and each snap sends a pulse of air up your throat. On its own that
buzz sounds like nothing in particular. Your throat, mouth, and lips shape it
into a vowel. Move your tongue, keep the buzz, and "ah" becomes "ee".

This split is the *source-filter model*. The source sets the pitch. The filter,
the shape of your vocal tract, sets which sound you make. The two are nearly
independent, which is why you can sing one vowel on many notes, and why TTS
systems predict pitch and spectrum separately.

*MFCCs and normalization* kept the smooth envelope of each spectrum and
threw the harmonics away. This lesson explains where each one comes from.
You build both halves: a synthesizer that makes vowels from pulses and
resonators, and the tools that take a recording apart again.
*Linear prediction* (LPC) recovers the filter; it compressed speech in digital
telephony for decades. The *cepstrum* recovers the pitch.

## What you already know

- An IIR filter computes each output from past outputs; `scipy.signal.lfilter(b, a, x)` runs its difference equation.
- A filter's *poles* are the roots of its denominator polynomial, and a pole near the unit circle makes a sharp peak in the frequency response.
- Convolving two signals multiplies their spectra.
- A logarithm turns a product into a sum.
- MFCCs are a cepstrum: a cosine transform of a log spectrum, whose low coefficients describe its smooth shape.

## The source-filter model

A speech signal is an excitation passed through a chain of filters. In
$z$-transform notation, where multiplying transforms is the same as
convolving signals:

$$
S(z) = E(z)\,G(z)\,V(z)\,R(z)
$$

Read it aloud: speech $S$ is the excitation $E$, shaped by the glottal pulse
$G$, then by the vocal tract $V$, then by the radiation at the lips $R$. For a
*voiced* sound such as a vowel, $E$ is a pulse train at the *fundamental
frequency* $F_0$, the vocal folds' rate. For an *unvoiced* sound such as "s",
$E$ is noise from air rushing through a narrow gap.

A pulse train with period $T_0 = 1/F_0$ has a spectrum made only of lines at
$F_0, 2F_0, 3F_0, \dots$, the *harmonics*. The filter does not move those lines;
it raises and lowers them. The peaks of the filter's response are the
*formants*, numbered $F_1, F_2, F_3$ from the bottom. The formants decide the
vowel. $F_0$ decides the pitch.

### A tube predicts a neutral vowel

Model the vocal tract as a uniform tube, closed at the vocal folds and open at
the lips. Such a tube resonates when its length is an odd number of quarter
wavelengths:

$$
F_n = \frac{(2n - 1)\,c}{4L}
$$

Read it aloud: the $n$-th resonance is an odd multiple, $2n - 1$, of the speed
of sound $c$ divided by four times the tube's length $L$. With $L = 17.5$ cm,
typical for an adult man, and $c = 350$ m/s in warm, moist air, the
resonances are 500, 1500, and 2500 Hz. That is close to the neutral vowel in
"sofa". Moving the tongue narrows the tube in one place and shifts each
formant. The following table lists average formants for three vowels spoken
by adult men, measured by Peterson and Barney in 1952.

| Vowel | Example | $F_1$ (Hz) | $F_2$ (Hz) | $F_3$ (Hz) |
|---|---|---|---|---|
| /i/ | heed | 270 | 2290 | 3010 |
| /ɑ/ | father | 730 | 1090 | 2440 |
| /u/ | who'd | 300 | 870 | 2240 |

$F_1$ falls as the tongue rises: /i/ and /u/ are high vowels, /ɑ/ is low.
$F_2$ rises as the tongue moves forward: /i/ is front, /u/ is back.

## A formant as a resonator

One formant is one pair of complex-conjugate poles. Put a pole at radius $r$
and angle $\theta$, and its mirror at $-\theta$. The angle sets the peak
frequency, and the radius sets how sharp it is: the closer $r$ is to 1, the
narrower the peak. For a formant at $F$ Hz with a *bandwidth* of $B$ Hz, the
width of the peak 3 dB below its top, at sample rate $f_s$:

$$
\begin{aligned}
r &= e^{-\pi B / f_s} \\
\theta &= \frac{2\pi F}{f_s} \\
y[n] &= b_0\,x[n] + 2r\cos\theta\; y[n-1] - r^2\, y[n-2]
\end{aligned}
$$

Read it aloud: the radius shrinks exponentially with bandwidth over sample
rate, and the angle is the formant's share of a full turn of the unit circle.
Each output sample is a scaled input plus two feedback terms. Dennis Klatt's
synthesizer sets $b_0 = 1 - 2r\cos\theta + r^2$, which makes the gain at 0 Hz
exactly 1, so a cascade of resonators does not change the overall level.

```python
a = np.array([1.0, -2 * r * np.cos(theta), r * r])
b = np.array([a.sum()])
y = scipy.signal.lfilter(b, a, x)
```

For $F = 500$ Hz and $B = 100$ Hz at 16 kHz: $r = e^{-\pi \cdot 100/16000} = 0.980557$
and $\theta = 0.19635$ rad, with $\cos\theta = 0.980785$. So
$a = [1, -1.923431, 0.961491]$ and $b_0 = 0.038060$. The gain is 1 at 0 Hz and
5.06, or 14.1 dB, at 500 Hz.

A vowel synthesizer is an impulse train through three of these in a row. The
following demo is that synthesizer.

```demo
{"id": "vowel-formants", "caption": "Drag on the chart to move the first two formants between vowels, then change the pitch of the source and hear the vowel stay the same as its harmonics spread apart."}
```

## Linear prediction

Now go the other way: given a recorded frame, find the filter. If the vocal
tract is all-pole, each sample is almost a fixed weighted sum of the previous
$p$ samples, and only the excitation is unpredictable. Linear prediction
finds those weights:

$$
\begin{aligned}
\hat x[n] &= \sum_{k=1}^{p} \alpha_k\, x[n-k] \\
e[n] &= x[n] - \hat x[n] \\
A(z) &= 1 - \sum_{k=1}^{p} \alpha_k z^{-k}
\end{aligned}
$$

Read it aloud: the prediction $\hat x[n]$ is a weighted sum of the $p$ samples
before it. The error $e[n]$ is what the prediction misses. Filtering the
signal by $A(z)$ produces that error, so $1/A(z)$ is the all-pole filter that
turns the error back into the signal.

The *autocorrelation method* windows the frame, treats it as zero outside its
$N$ samples, and picks the $\alpha_k$ that minimize $\sum_n e[n]^2$. Setting
each derivative to zero gives the *normal equations*:

$$
\begin{aligned}
\sum_{k=1}^{p} \alpha_k\, r\big[|i - k|\big] &= r[i],
\quad i = 1, \dots, p \\
r[k] &= \sum_{n=k}^{N-1} x[n]\, x[n-k]
\end{aligned}
$$

Read it aloud: $r[k]$ is the frame's *autocorrelation*, the sum of products of
the frame with itself shifted by $k$ samples. The equations say the weighted
autocorrelations must reproduce each $r[i]$. Their matrix has $r[|i - k|]$ in
row $i$, column $k$: constant along every diagonal, a *Toeplitz* matrix.

### The Levinson-Durbin recursion

A general solver takes $O(p^3)$ operations. The Levinson-Durbin recursion
uses the Toeplitz structure to solve order 1, then order 2 from order 1, up to
order $p$, in $O(p^2)$:

1. Start with $E_0 = r[0]$.
2. For $i = 1, \dots, p$:
   1. $k_i = \left(r[i] - \sum_{j=1}^{i-1} \alpha_j^{(i-1)}\, r[i-j]\right) / E_{i-1}$.
   2. $\alpha_i^{(i)} = k_i$.
   3. $\alpha_j^{(i)} = \alpha_j^{(i-1)} - k_i\, \alpha_{i-j}^{(i-1)}$ for $j = 1, \dots, i-1$.
   4. $E_i = (1 - k_i^2)\, E_{i-1}$.

Read it aloud: $k_i$ is what the order-$(i-1)$ predictor fails to explain at lag
$i$, scaled by its error. It becomes the last coefficient, and every older
coefficient is corrected by $k_i$ times its mirror image. The error shrinks by
$1 - k_i^2$. The $k_i$ are *reflection coefficients*. With the autocorrelation
method, every $|k_i| < 1$, which guarantees that $1/A(z)$ is stable.

```python
k_i = (r[i] - alpha[: i - 1] @ r[i - 1 : 0 : -1]) / E
alpha[: i - 1] = old - k_i * old[::-1]
```

### Work the recursion by hand

Take $r = [1, 0.5, 0.1]$ and $p = 2$.

1. $E_0 = 1$. Order 1: $k_1 = 0.5 / 1 = 0.5$, so $\alpha_1^{(1)} = 0.5$ and
   $E_1 = (1 - 0.25) \times 1 = 0.75$.
2. Order 2: $k_2 = (0.1 - 0.5 \times 0.5) / 0.75 = -0.2$. The last
   coefficient of order 2 is $\alpha_2 = -0.2$. The old one is corrected:
   $\alpha_1 = 0.5 - (-0.2)(0.5) = 0.6$. The error is
   $E_2 = 0.75 \times (1 - 0.04) = 0.72$.

So $\alpha = [0.6, -0.2]$ and $A(z) = 1 - 0.6z^{-1} + 0.2z^{-2}$.
`scipy.linalg.solve_toeplitz([1, 0.5], [0.5, 0.1])` returns the same
`[0.6, -0.2]`.

> [!KEY]
> LPC fits an all-pole filter by making each sample predictable from the
> previous $p$. Levinson-Durbin solves the Toeplitz normal equations in
> $O(p^2)$, and $|k_i| < 1$ keeps the filter stable.

## The envelope and the formants

With gain $G^2 = E_p$, the all-pole model's power spectrum is a smooth curve
through the frame's harmonics:

$$
\left|H(e^{j\omega})\right|^2 = \frac{E_p}{\left|1 - \sum_{k=1}^{p} \alpha_k e^{-j\omega k}\right|^2}
$$

Read it aloud: the envelope is the error energy divided by the squared size of
$A$ evaluated around the unit circle, at angular frequency $\omega$. Where $A$
is small, near a pole, the envelope peaks. The FFT of the coefficient vector
$[1, -\alpha_1, \dots, -\alpha_p]$ evaluates $A$ on the FFT grid:

```python
A = np.fft.rfft(np.concatenate([[1.0], -alpha]), n_fft)
envelope = E / np.abs(A) ** 2
```

Each formant is a root $z_i$ of $A(z)$ with a positive angle. Invert the
resonator formulas: $F = \frac{\angle z_i}{2\pi} f_s$ and
$B = -\frac{f_s}{\pi} \ln |z_i|$. Keep the roots with a narrow bandwidth, under
about 400 Hz. The broad ones shape the overall tilt, not a formant.

The worked $A(z)$ has roots $0.3 \pm 0.3317j$, with radius 0.4472. At 16 kHz
the positive one has angle 0.8355 rad, so $F = 2128$ Hz, and a bandwidth of
$-(16000/\pi)\ln 0.4472 = 4098$ Hz. That broad bump is no formant.

How many coefficients? Each formant needs two poles, and speech has about one
formant per kilohertz of bandwidth, so the rule of thumb is $p = f_s/1000 + 2$.
The extra two model the glottal pulse and the lips. At 16 kHz that is 18.

## The real cepstrum

LPC models the filter and discards the source. The cepstrum keeps both, side by
side. The *real cepstrum* is the inverse DFT of the log magnitude spectrum:

$$
c[n] = \operatorname{IDFT}\left(\ln\left|\operatorname{DFT}(x)\right|\right)[n]
$$

Read it aloud: take the spectrum, keep its magnitude, take the natural log,
and transform back. Its index $n$ is called *quefrency*, and it is measured in
samples. For speech, $x = e * h$ and $\ln|X| = \ln|E| + \ln|H|$: the log turns
the source-filter product into a sum. The vocal tract's smooth envelope lands
at low quefrencies, below about 2 ms. The harmonics' regular ripple, spaced
$F_0$ apart in frequency, lands as a peak at quefrency $f_s / F_0$ samples, the
pitch period.

```python
ceps = np.fft.irfft(np.log(np.abs(np.fft.rfft(frame)) + 1e-10), len(frame))
```

As a small example, take 32 samples with $x[0] = 1$ and an echo $x[4] = 0.5$.
Its cepstrum is 0.2507 at quefrency 4, $-0.0639$ at 8, and 0.0240 at 12, close
to the series $a/2 = 0.25$, $-a^2/4 = -0.0625$, and $a^3/6 = 0.0208$ for an echo
of strength $a = 0.5$. The small differences come from the 32-point FFT
wrapping around. A voice is an "echo" of each glottal pulse every period, so
for $F_0 = 120$ Hz at 16 kHz the peak sits near quefrency 133.

> [!KEY]
> In the cepstrum, the vocal tract sits at low quefrencies and the pitch
> period shows up as a peak at $f_s / F_0$ samples.

> [!RECAP]
> - Speech is a source (pulses or noise) through a vocal-tract filter; formants are the filter's peaks.
> - A formant at $F$ with bandwidth $B$ is a pole pair at radius $e^{-\pi B/f_s}$ and angle $2\pi F/f_s$.
> - LPC solves Toeplitz normal equations by Levinson-Durbin: $k_i$, the mirrored update, and $E_i = (1 - k_i^2)E_{i-1}$.
> - Formants are narrow-bandwidth roots of $A(z)$; the envelope is $E / |A|^2$.
> - The real cepstrum is $\operatorname{IDFT}(\ln|X|)$, with the pitch peak at $f_s/F_0$.

## Build it step by step

The checks test the functions in this order.

1. **`impulse_train(f0, sample_rate, n_samples)`** returns `n_samples` zeros
   with a one at the nearest sample to every multiple of the period
   `sample_rate / f0`. The usual mistake is truncating the period to an
   integer first, which drifts the pitch.
2. **`resonator(frequency, bandwidth, sample_rate)`** returns `(b, a)` from
   the formulas. The usual mistake is $r = e^{-2\pi B / f_s}$, which doubles
   the bandwidth.
3. **`synthesize_vowel(f0, formants, bandwidths, sample_rate, seconds)`**
   runs `lfilter` once per resonator and scales the peak to 0.5.
4. **`autocorrelation(frame, p)`** returns `r[0..p]` as dot products of the
   frame with shifted copies of itself. The usual mistake is normalizing by
   `N - k`, which can make the matrix indefinite and the filter unstable.
5. **`levinson_durbin(r, p)`** returns `(alpha, E, k)`. Copy the old
   coefficients before updating them. The usual mistake is updating
   `alpha` in place, so later terms read already-updated values.
6. **`lpc(frame, order)`** multiplies by `np.hamming(len(frame))`, then calls
   the two functions before it.
7. **`lpc_envelope(alpha, E, n_fft)`** evaluates `E / |A|^2` with
   `np.fft.rfft`. The usual mistake is forgetting the minus signs in
   `[1, -alpha]`.
8. **`formants(alpha, sample_rate, max_bandwidth, min_frequency)`** calls
   `np.roots`, keeps positive angles, converts, filters, and sorts.
9. **`real_cepstrum(frame, eps)`** returns `len(frame)` real values from the
   formula. The usual mistake is leaving out `n=len(frame)` in
   `np.fft.irfft`, which returns one sample short for an odd-length frame.
10. **`cepstral_pitch(frame, sample_rate, fmin, fmax)`** windows the frame,
    finds the largest cepstral value between `floor(fs / fmax)` and
    `ceil(fs / fmin)`, and returns `fs / q`.

## Cost and scaling

For a frame of $N$ samples and order $p$, the steps cost the following.

| Step | Time | Memory |
|---|---|---|
| Autocorrelation | $O(N p)$ | $O(p)$ |
| Levinson-Durbin | $O(p^2)$ | $O(p)$ |
| A general solve of the same system | $O(p^3)$ | $O(p^2)$ |
| Roots of $A(z)$ | $O(p^3)$, an eigenvalue problem | $O(p^2)$ |
| Cepstrum | $O(N \log N)$ | $O(N)$ |

At $N = 512$ and $p = 18$, all of it takes microseconds per frame. That is why
LPC ran on 1970s hardware: telephone codecs sent roughly a dozen coefficients
and a pitch per frame instead of the waveform.

## Where this breaks

**High voices.** At high $F_0$ the harmonics are far apart, and LPC fits the
harmonics instead of the envelope. For the synthetic /ɑ/ in this lesson,
order-18 LPC finds $F_1$ at 720 Hz when $F_0 = 100$ Hz and at 791 Hz when
$F_0 = 200$ Hz, 8% off the true 730 Hz. Children's speech is hardest.

**Nasals and zeros.** In "m" and "n", the nasal cavity adds antiresonances,
which are zeros. An all-pole model can only approximate them with extra poles.

**Spectral tilt.** Real glottal pulses fall off at about 12 dB per octave, so
high formants are weak and poorly fitted. Pre-emphasis with $\alpha = 0.97$,
from *Filters, convolution, and reverb*, flattens the tilt before LPC.

**Spurious poles.** Too high an order fits noise and harmonics with extra
narrow poles. Filter roots by bandwidth, and track formants across frames.

**Silence in the cepstrum.** $\ln 0$ is $-\infty$. Add a small epsilon to the
magnitude, and skip frames with no energy.

> [!WARNING]
> Sign conventions for LPC differ. Some texts and libraries write
> $A(z) = 1 + \sum a_k z^{-k}$, so $a_k = -\alpha_k$ and the reflection
> coefficients flip sign. Check which one a reference uses before you
> compare numbers.

## The exercise

You build `impulse_train`, `resonator`, `synthesize_vowel`, `autocorrelation`,
`levinson_durbin`, `lpc`, `lpc_envelope`, `formants`, `real_cepstrum`, and
`cepstral_pitch`. Press **Run** to hear /ɑ/, /i/, and /u/, see each frame's
spectrum under its LPC envelope, read the formants and pitch that your code
recovers, and hear one vowel at two pitches.

The checks verify the impulse positions and the resonator's coefficients,
0 Hz gain, and peak. They compare your autocorrelation with `np.correlate`,
work the Levinson-Durbin example, and compare the recursion with
`scipy.linalg.solve_toeplitz` on random stable processes, including $|k_i| < 1$
and $E$ as the true residual energy. They compare the envelope with
`scipy.signal.freqz`, require formants within 5% of 500, 1500, and 2500 Hz,
check the cepstrum and cepstral pitch within 2%, and confirm that your vowel
keeps its formants when its pitch changes.
