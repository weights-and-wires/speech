---
title: MFCCs
slug: 013-mfccs
part: "Part 2 — Hearing and speech production"
summary: Turn 40 correlated log-mel energies into 13 MFCCs with a cosine transform that sorts the spectrum from broad shape to fine ripple.
minutes: 40
gpu: false
objectives:
  - Explain why a cosine transform of a log-mel vector separates the vowel's broad shape from its fine ripple.
  - Build the orthonormal DCT-II from its formula, and check it by hand on a four-band example.
  - Compute MFCCs from mel power that match librosa's, and say how HTK and Kaldi differ.
  - Apply HTK sinusoidal liftering, and say what it changes.
prereqs:
  - 012-log-mel-spectrograms
exercise: 013-mfccs
---

# MFCCs

A log-mel spectrogram describes each 10 ms frame of speech with 40 or 80 numbers, and those numbers are not independent. When band 12 is loud, band 13 almost always is too, because a vowel's energy sits in broad humps that cover many neighboring bands. In the three-vowel clip you build in this chapter, neighboring bands correlate at 0.80 on average.

That redundancy costs something. Before neural networks, the recognizers of Part 3 described each frame with a Gaussian. A Gaussian that models every correlation among 13 numbers needs $13 \times 14 / 2 = 91$ covariance entries, and one that assumes independent dimensions needs 13 variances. The cheap model is only correct if the dimensions are close to independent. So the question is: what one transform turns 40 tangled numbers into a handful that carry the same story and do not repeat each other?

The answer is a cosine transform. It sorts a log-mel vector from its slow, broad shape to its fast, fine ripple. Keep the first 13, and you have the *mel-frequency cepstral coefficients*, or MFCCs, the speech feature that recognizers used from 1980 until deep learning.

## What you already know

- A log-mel spectrogram has one row per mel band and one column per frame.
- A decibel is $10 \log_{10}$ of a power ratio.
- A logarithm turns a product into a sum: $\log(ab) = \log a + \log b$.
- A cosine that makes $n$ half-turns across a vector is slow when $n$ is small and fast when $n$ is large.
- A matrix with orthonormal rows keeps the length of every vector it multiplies.

## Look at one frame

Plot the log-mel values of one voiced frame against band number. You see a smooth curve with a few broad humps. That curve is the *spectral envelope*: the vocal tract shapes it, and it tells you which vowel is being spoken. In Figure 1 the humps are the first three formants of an /ɑ/. On top of the envelope, in the low bands, sits a faster ripple from the individual harmonics of the voice's pitch. The [source-filter chapter](/lessons/015-the-source-filter-model) explains where each part comes from.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="One frame of the vowel /ɑ/ as 40 log-mel values, and the smooth curve rebuilt from its first 13 DCT coefficients">
<line x1="70.0" y1="260.0" x2="580.0" y2="260.0" class="stroke-grid"/>
<text x="64.0" y="264.0" class="fig-small" text-anchor="end">-60</text>
<line x1="70.0" y1="227.1" x2="580.0" y2="227.1" class="stroke-grid"/>
<text x="64.0" y="231.1" class="fig-small" text-anchor="end">-50</text>
<line x1="70.0" y1="194.3" x2="580.0" y2="194.3" class="stroke-grid"/>
<text x="64.0" y="198.3" class="fig-small" text-anchor="end">-40</text>
<line x1="70.0" y1="161.4" x2="580.0" y2="161.4" class="stroke-grid"/>
<text x="64.0" y="165.4" class="fig-small" text-anchor="end">-30</text>
<line x1="70.0" y1="128.6" x2="580.0" y2="128.6" class="stroke-grid"/>
<text x="64.0" y="132.6" class="fig-small" text-anchor="end">-20</text>
<line x1="70.0" y1="95.7" x2="580.0" y2="95.7" class="stroke-grid"/>
<text x="64.0" y="99.7" class="fig-small" text-anchor="end">-10</text>
<line x1="70.0" y1="62.9" x2="580.0" y2="62.9" class="stroke-grid"/>
<text x="64.0" y="66.9" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="30.0" x2="580.0" y2="30.0" class="stroke-grid"/>
<text x="64.0" y="34.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="260.0" x2="580.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted"/>
<text x="70.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="200.8" y1="260.0" x2="200.8" y2="265.0" class="stroke-muted"/>
<text x="200.8" y="279.0" class="fig-small" text-anchor="middle">10</text>
<line x1="331.5" y1="260.0" x2="331.5" y2="265.0" class="stroke-muted"/>
<text x="331.5" y="279.0" class="fig-small" text-anchor="middle">20</text>
<line x1="462.3" y1="260.0" x2="462.3" y2="265.0" class="stroke-muted"/>
<text x="462.3" y="279.0" class="fig-small" text-anchor="middle">30</text>
<line x1="580.0" y1="260.0" x2="580.0" y2="265.0" class="stroke-muted"/>
<text x="580.0" y="279.0" class="fig-small" text-anchor="middle">39</text>
<text x="325.0" y="298.0" text-anchor="middle">mel band (0 is the lowest, 39 the highest)</text>
<text x="8.0" y="20.0" text-anchor="start">dB</text>
<polyline points="70.0,133.3 83.1,130.4 96.2,124.9 109.2,126.4 122.3,116.7 135.4,111.0 148.5,105.1 161.5,86.1 174.6,64.3 187.7,51.0 200.8,69.1 213.8,74.8 226.9,73.3 240.0,69.4 253.1,66.3 266.2,92.5 279.2,116.8 292.3,131.3 305.4,145.1 318.5,156.4 331.5,165.0 344.6,173.2 357.7,173.7 370.8,168.8 383.8,157.8 396.9,170.3 410.0,196.5 423.1,212.2 436.2,214.1 449.2,221.9 462.3,214.2 475.4,224.9 488.5,225.2 501.5,219.9 514.6,220.8 527.7,208.1 540.8,212.9 553.8,223.9 566.9,221.2 580.0,225.4" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="133.3" r="2.8" class="fill-a"/>
<circle cx="83.1" cy="130.4" r="2.8" class="fill-a"/>
<circle cx="96.2" cy="124.9" r="2.8" class="fill-a"/>
<circle cx="109.2" cy="126.4" r="2.8" class="fill-a"/>
<circle cx="122.3" cy="116.7" r="2.8" class="fill-a"/>
<circle cx="135.4" cy="111.0" r="2.8" class="fill-a"/>
<circle cx="148.5" cy="105.1" r="2.8" class="fill-a"/>
<circle cx="161.5" cy="86.1" r="2.8" class="fill-a"/>
<circle cx="174.6" cy="64.3" r="2.8" class="fill-a"/>
<circle cx="187.7" cy="51.0" r="2.8" class="fill-a"/>
<circle cx="200.8" cy="69.1" r="2.8" class="fill-a"/>
<circle cx="213.8" cy="74.8" r="2.8" class="fill-a"/>
<circle cx="226.9" cy="73.3" r="2.8" class="fill-a"/>
<circle cx="240.0" cy="69.4" r="2.8" class="fill-a"/>
<circle cx="253.1" cy="66.3" r="2.8" class="fill-a"/>
<circle cx="266.2" cy="92.5" r="2.8" class="fill-a"/>
<circle cx="279.2" cy="116.8" r="2.8" class="fill-a"/>
<circle cx="292.3" cy="131.3" r="2.8" class="fill-a"/>
<circle cx="305.4" cy="145.1" r="2.8" class="fill-a"/>
<circle cx="318.5" cy="156.4" r="2.8" class="fill-a"/>
<circle cx="331.5" cy="165.0" r="2.8" class="fill-a"/>
<circle cx="344.6" cy="173.2" r="2.8" class="fill-a"/>
<circle cx="357.7" cy="173.7" r="2.8" class="fill-a"/>
<circle cx="370.8" cy="168.8" r="2.8" class="fill-a"/>
<circle cx="383.8" cy="157.8" r="2.8" class="fill-a"/>
<circle cx="396.9" cy="170.3" r="2.8" class="fill-a"/>
<circle cx="410.0" cy="196.5" r="2.8" class="fill-a"/>
<circle cx="423.1" cy="212.2" r="2.8" class="fill-a"/>
<circle cx="436.2" cy="214.1" r="2.8" class="fill-a"/>
<circle cx="449.2" cy="221.9" r="2.8" class="fill-a"/>
<circle cx="462.3" cy="214.2" r="2.8" class="fill-a"/>
<circle cx="475.4" cy="224.9" r="2.8" class="fill-a"/>
<circle cx="488.5" cy="225.2" r="2.8" class="fill-a"/>
<circle cx="501.5" cy="219.9" r="2.8" class="fill-a"/>
<circle cx="514.6" cy="220.8" r="2.8" class="fill-a"/>
<circle cx="527.7" cy="208.1" r="2.8" class="fill-a"/>
<circle cx="540.8" cy="212.9" r="2.8" class="fill-a"/>
<circle cx="553.8" cy="223.9" r="2.8" class="fill-a"/>
<circle cx="566.9" cy="221.2" r="2.8" class="fill-a"/>
<circle cx="580.0" cy="225.4" r="2.8" class="fill-a"/>
<polyline points="70.0,130.3 83.1,130.1 96.2,129.1 109.2,126.3 122.3,120.2 135.4,110.3 148.5,97.0 161.5,82.8 174.6,70.8 187.7,63.8 200.8,62.4 213.8,65.0 226.9,69.3 240.0,74.1 253.1,80.3 266.2,90.5 279.2,106.4 292.3,126.8 305.4,147.6 318.5,163.3 331.5,170.5 344.6,169.7 357.7,165.4 370.8,163.5 383.8,167.7 396.9,178.2 410.0,192.0 423.1,205.1 436.2,214.8 449.2,220.5 462.3,223.0 475.4,223.4 488.5,222.1 501.5,219.5 514.6,216.5 527.7,214.5 540.8,215.1 553.8,218.3 566.9,222.7 580.0,225.8" class="stroke-b" fill="none"/>
<line x1="380" y1="40.0" x2="404" y2="40.0" class="stroke-a"/>
<text x="410.0" y="44.0" class="fig-small" text-anchor="start">40 log-mel bands</text>
<line x1="380" y1="60.0" x2="404" y2="60.0" class="stroke-b"/>
<text x="410.0" y="64.0" class="fig-small" text-anchor="start">first 13 coefficients</text>
</svg>
Figure 1. One 10 ms frame of the synthetic /ɑ/ from the exercise. Thirteen numbers rebuild the 40 bands to within 1.8 dB RMS: the curve keeps the broad humps that identify the vowel and drops the band-to-band ripple.
```

A recognizer wants the humps and does not want the ripple, because the ripple changes with the speaker's pitch while the vowel stays the same. Notice that the humps are slow along the band axis and the ripple is fast. A fast-versus-slow split is what a Fourier-style transform makes.

## Treat the band axis as a signal

Ask which "frequencies" the log-mel vector contains, as if the 40 bands were 40 samples of a signal. The smooth envelope lives in the slowest components and the ripple in the faster ones. This is a spectrum of a log spectrum, called a *cepstrum*, an anagram of "spectrum". Its axis is called *quefrency*, and MFCCs are a cepstrum computed on mel bands.

The obvious tool is the DFT. It has a flaw here: it treats the vector as one period of a repeating signal, so the jump from the last band back to the first looks like a sharp edge, and an edge spreads energy across every coefficient. The *discrete cosine transform* (DCT) avoids that by treating the vector as if it were mirrored at both ends. No artificial edge appears, so the energy of a smooth vector piles up in the first few coefficients. That pile-up is what lets you throw the rest away.

## The discrete cosine transform

The type-II DCT with orthonormal scaling is the one every MFCC library uses. For log-mel values $L_m$, with $m = 0, \dots, M-1$:

$$
c_n = \sqrt{\frac{2 - \delta_{n0}}{M}} \sum_{m=0}^{M-1} L_m \cos\left(\frac{\pi n (2m + 1)}{2M}\right),
\qquad n = 0, \dots, M - 1
$$

Read it aloud: coefficient $c_n$ is the correlation of the log-mel vector with a cosine that makes $n$ half-turns across the $M$ bands. The cosine is sampled at the band centers $m + \tfrac{1}{2}$, which is the mirrored picture from the last section. The scale factor is $\sqrt{1/M}$ for $n = 0$ and $\sqrt{2/M}$ for every other $n$; $\delta_{n0}$ is 1 when $n = 0$ and 0 otherwise. That scale makes the transform *orthonormal*: the rows of the cosine matrix are perpendicular unit vectors, so the DCT keeps the sum of squares, and its transpose is its inverse.

As a matrix, the DCT is one line of broadcasting and one product:

```python
basis = np.cos(np.pi * n[:, None] * (2 * m[None, :] + 1) / (2 * M))
coeffs = (scale[:, None] * basis) @ log_mel      # (M, M) @ (M, frames)
```

This equals `scipy.fft.dct(log_mel, type=2, norm="ortho", axis=0)`.

### Work a four-band DCT by hand

Take $L = [4, 2, 2, 0]$, so $M = 4$.

- $c_0 = \sqrt{1/4}\,(4 + 2 + 2 + 0) = 0.5 \times 8 = 4$. Coefficient 0 is the mean level times $\sqrt{M}$.
- For $c_1$, the cosines at $n = 1$ are $\cos(\pi/8), \cos(3\pi/8), \cos(5\pi/8), \cos(7\pi/8) = [0.9239, 0.3827, -0.3827, -0.9239]$. The sum is $4(0.9239) + 2(0.3827) - 2(0.3827) - 0 = 3.6955$, and $c_1 = \sqrt{2/4} \times 3.6955 = 2.6131$. A positive $c_1$ means energy tilts toward the low bands.
- At $n = 2$ the cosines are $[0.7071, -0.7071, -0.7071, 0.7071]$, and $4 - 2 - 2 + 0 = 0$ times 0.7071 gives $c_2 = 0$.
- At $n = 3$ they are $[0.3827, -0.9239, 0.9239, -0.3827]$, which gives $c_3 = 0.7071 \times 1.5307 = 1.0824$.

The result is $[4, 2.6131, 0, 1.0824]$. Check the energy: $16 + 4 + 4 + 0 = 24$, and $16 + 6.8284 + 0 + 1.1716 = 24$. Now keep only $c_0$ and $c_1$ and invert with the transpose. You get $[3.707, 2.707, 1.293, 0.293]$, a smooth slope through the original. Truncating the cepstrum smooths the spectrum.

```figure
<svg viewBox="0 0 640 250" role="img" aria-label="Three bar charts: the four-band log-mel vector 4, 2, 2, 0, its DCT coefficients, and the vector rebuilt from the first two coefficients">
<text x="95.0" y="24.0" class="fig-small fig-bold" text-anchor="middle">log-mel L</text>
<line x1="20" y1="140" x2="170" y2="140" class="stroke-muted"/>
<rect x="30" y="60.0" width="26" height="80.0" class="fill-a"/>
<text x="43.0" y="54.0" class="fig-small" text-anchor="middle">4</text>
<text x="43.0" y="160.0" class="fig-small" text-anchor="middle">m=0</text>
<rect x="66" y="100.0" width="26" height="40.0" class="fill-a"/>
<text x="79.0" y="94.0" class="fig-small" text-anchor="middle">2</text>
<text x="79.0" y="160.0" class="fig-small" text-anchor="middle">m=1</text>
<rect x="102" y="100.0" width="26" height="40.0" class="fill-a"/>
<text x="115.0" y="94.0" class="fig-small" text-anchor="middle">2</text>
<text x="115.0" y="160.0" class="fig-small" text-anchor="middle">m=2</text>
<rect x="138" y="140.0" width="26" height="0.0" class="fill-a"/>
<text x="151.0" y="134.0" class="fig-small" text-anchor="middle">0</text>
<text x="151.0" y="160.0" class="fig-small" text-anchor="middle">m=3</text>
<text x="300.0" y="24.0" class="fig-small fig-bold" text-anchor="middle">DCT coefficients c</text>
<line x1="225" y1="140" x2="375" y2="140" class="stroke-muted"/>
<rect x="235" y="60.0" width="26" height="80.0" class="fill-d"/>
<text x="248.0" y="54.0" class="fig-small" text-anchor="middle">4</text>
<text x="248.0" y="160.0" class="fig-small" text-anchor="middle">n=0</text>
<rect x="271" y="87.7" width="26" height="52.3" class="fill-d"/>
<text x="284.0" y="81.7" class="fig-small" text-anchor="middle">2.613</text>
<text x="284.0" y="160.0" class="fig-small" text-anchor="middle">n=1</text>
<rect x="307" y="140.0" width="26" height="0.0" class="fill-d"/>
<text x="320.0" y="134.0" class="fig-small" text-anchor="middle">0</text>
<text x="320.0" y="160.0" class="fig-small" text-anchor="middle">n=2</text>
<rect x="343" y="118.4" width="26" height="21.6" class="fill-d"/>
<text x="356.0" y="112.4" class="fig-small" text-anchor="middle">1.082</text>
<text x="356.0" y="160.0" class="fig-small" text-anchor="middle">n=3</text>
<text x="505.0" y="24.0" class="fig-small fig-bold" text-anchor="middle">rebuilt from c0, c1</text>
<line x1="430" y1="140" x2="580" y2="140" class="stroke-muted"/>
<rect x="440" y="65.9" width="26" height="74.1" class="fill-b"/>
<text x="453.0" y="59.9" class="fig-small" text-anchor="middle">3.707</text>
<text x="453.0" y="160.0" class="fig-small" text-anchor="middle">m=0</text>
<rect x="476" y="85.9" width="26" height="54.1" class="fill-b"/>
<text x="489.0" y="79.9" class="fig-small" text-anchor="middle">2.707</text>
<text x="489.0" y="160.0" class="fig-small" text-anchor="middle">m=1</text>
<rect x="512" y="114.1" width="26" height="25.9" class="fill-b"/>
<text x="525.0" y="108.1" class="fig-small" text-anchor="middle">1.293</text>
<text x="525.0" y="160.0" class="fig-small" text-anchor="middle">m=2</text>
<rect x="548" y="134.1" width="26" height="5.9" class="fill-b"/>
<text x="561.0" y="128.1" class="fig-small" text-anchor="middle">0.293</text>
<text x="561.0" y="160.0" class="fig-small" text-anchor="middle">m=3</text>
<text x="20.0" y="215.0" class="fig-small" text-anchor="start">Left: four bands. Middle: c0 is the level (times √4), c1 the slope across bands, c2 and c3 the finer wiggles.</text>
<text x="20.0" y="234.0" class="fig-small" text-anchor="start">Right: zeroing c2 and c3 and inverting leaves a smooth slope close to L.</text>
</svg>
Figure 2. The worked four-band example. The coefficients 4, 2.613, 0, 1.082 carry the same energy as the bands, 24. Keeping only the first two rebuilds 3.707, 2.707, 1.293, 0.293, a smooth version of the input.
```

> [!KEY]
> The DCT of a log-mel vector orders it from broad shape to fine detail. Keeping the first 13 coefficients keeps the spectral envelope and drops the harmonics.

## From log-mel to MFCCs

With the DCT in hand, an MFCC frame takes three steps, starting from a mel power matrix:

1. Convert to decibels: $10 \log_{10} \max(M, 10^{-10})$, floored 80 dB below the loudest value, as `librosa.power_to_db` does.
2. Take the DCT-II down the mel axis.
3. Keep the first 13 rows.

```python
db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
db = np.maximum(db, db.max() - 80.0)
mfccs = dct2_ortho(db)[:13]
```

The numbers depend on conventions. The following table lists them for the libraries you are most likely to compare against.

| Library | Log | Coefficients kept | DCT |
|---|---|---|---|
| librosa `feature.mfcc` | dB, $10 \log_{10}$, 80 dB floor | 20 | type II, orthonormal |
| `torchaudio` `MFCC` | dB, 80 dB floor | 40 | type II, orthonormal |
| HTK and Kaldi | $\ln$ | 13 | type II |

A dB value is $10 / \ln 10 = 4.343$ times a natural-log value, so librosa's MFCCs are about 4.3 times Kaldi's, before the other differences. Coefficient $c_0$ tracks overall loudness, and HTK and Kaldi often replace it with the frame's log energy.

Why did this work so well for so long? Davis and Mermelstein showed in 1980 that MFCCs beat other features for word recognition. The DCT nearly decorrelates the log-mel bands, which made it practical to model each frame with Gaussians whose covariance matrices are diagonal: 13 variances instead of 91 covariance entries. Nearly is the word that matters. The coefficients are far less redundant than the bands, but they are not exactly independent.

The following demo shows one frame at every stage of the pipeline.

```demo
{"id": "mfcc-pipeline", "caption": "Lower the number of coefficients kept and watch the red curve rebuilt from the cepstrum smooth away detail until only the vowel's broad formant shape is left."}
```

## Liftering

The higher MFCCs are much smaller than the low ones. When you compare two frames by distance, as the template matching of Part 3 does, the large low coefficients would dominate. HTK *lifters* them, an anagram of "filter", by scaling coefficient $n$ with a raised sine:

$$
\tilde c_n = \left(1 + \frac{L}{2} \sin\frac{\pi n}{L}\right) c_n
$$

Read it aloud: each coefficient is multiplied by one plus $L/2$ times a sine that rises from 0 at $n = 0$ to its peak at $n = L/2$. With the usual $L = 22$, $c_0$ keeps weight 1, $c_1$ gets $1 + 11 \sin(\pi/22) = 2.565$, and $c_{11}$ gets 12. The middle coefficients are boosted most, so they count as much as the first.

```python
weights = 1.0 + (L / 2.0) * np.sin(np.pi * np.arange(n_ceps) / L)
liftered = c * weights[:, None]
```

librosa's `lifter` argument uses $n = 1, \dots, N$ instead of $0, \dots, N-1$, so it scales $c_0$ by 2.565 as well. Check which convention a reference uses.

> [!RECAP]
> - The orthonormal DCT-II has rows $\sqrt{(2 - \delta_{n0})/M}\cos(\pi n (2m + 1)/2M)$ and keeps energy.
> - MFCCs are the first 13 DCT coefficients of log-mel energies in dB.
> - Truncating the coefficients smooths the spectrum, because the dropped ones hold the fast ripple.
> - The HTK lifter scales row $n$ by $1 + (L/2)\sin(\pi n / L)$, with $n$ from 0.

## Build it step by step

The mel power function is provided in the starter. The checks test the four functions in this order.

1. **`dct2_ortho(x)`** takes `(M,)` or `(M, T)` and returns the same shape. Build the `(M, M)` cosine matrix, scale its rows, and multiply. The usual mistake is scaling row 0 by $\sqrt{2/M}$ like the others, which breaks orthonormality.
2. **`power_to_db(S, amin, top_db)`** returns $10\log_{10}\max(S, a_{\min})$ with the global floor. The usual mistake is $20\log_{10}$, which is for magnitudes, not powers.
3. **`mfcc(power_mel, n_mfcc, top_db)`** chains `power_to_db`, `dct2_ortho`, and a slice `[:n_mfcc]`. The usual mistake is taking the DCT along the time axis.
4. **`lifter(c, L)`** multiplies row $n$ by its weight, with $n$ starting at 0. The usual mistake is librosa's convention, starting at 1.

## Where this breaks

**Loudness hides in $c_0$.** Coefficient 0 is the mean level, so it changes with volume. Many systems drop it or replace it with the frame's log energy.

**Mismatched conventions.** Natural log against dB, 13 coefficients against 20 or 40, and the lifter's starting index all change the numbers. Compare against the library whose features your model was trained on.

**Neural networks often skip the DCT.** A network can learn its own combinations of correlated inputs, so Whisper uses the log-mel values directly. MFCCs still matter where a small, nearly independent feature is useful.

> [!WARNING]
> Keeping 13 of 40 coefficients throws away the harmonics, which means it throws away pitch. A task that needs the pitch, such as tracking a tune, cannot recover it from MFCCs.

## The exercise

You build `dct2_ortho`, `power_to_db`, `mfcc`, and `lifter`; `mel_power` is provided. Press **Run** to hear three vowels, see their first 13 MFCCs, and see one frame of log-mel against the curve rebuilt from its 13 coefficients.

The checks compare your DCT with `scipy.fft.dct` and confirm that it is orthonormal. They rebuild the four-band example from two coefficients, work the `power_to_db` floor, match your MFCCs to numbers from `librosa.feature.mfcc`, and verify the HTK lifter weights.
