---
title: Deltas and normalization
slug: 014-deltas-and-normalization
part: "Part 2 — Hearing and speech production"
summary: Add motion to MFCCs with regression deltas, cancel a fixed microphone with cepstral mean and variance normalization, and stack the 39-dimensional feature.
minutes: 40
gpu: false
objectives:
  - Compute delta features as least-squares slopes over a window of frames, with repeated edge frames, and work one by hand.
  - Explain why a fixed channel adds the same constant to every cepstral frame, and remove it with cepstral mean and variance normalization.
  - Stack MFCCs, deltas, and delta-deltas into the classic 39-dimensional feature and normalize it.
  - Say when per-utterance normalization misleads, and how streaming systems work around it.
prereqs:
  - 013-mfccs
exercise: 014-deltas-and-normalization
---

# Deltas and normalization

Say "ba" and "da" into a microphone. The two syllables hold nearly the same vowel, and what separates them is how the spectrum moves in the first 30 ms: a formant sliding up or sliding down. A single MFCC frame is a snapshot and cannot see that motion. Now say the same sentence into a second microphone, and every frame changes, because the new microphone colors the sound in its own way.

Both problems live on the time axis, the columns of the MFCC matrix. Motion shows up as a difference between frames. A microphone shows up as something the frames share. This chapter builds one operation for each, a *delta* feature for the motion and a *mean subtraction* for the shared part, and stacks the results into the classic 39-dimensional speech feature.

## What you already know

- An MFCC matrix has shape `(coefficients, frames)`, and time runs along axis 1.
- A slope is the change in a quantity per step, and a least-squares line is the straight line that minimizes the sum of squared errors.
- Convolving a signal with a filter multiplies its spectrum by the filter's frequency response.
- A logarithm turns a product into a sum, and the DCT is linear, so it keeps sums as sums.
- The mean of $T$ numbers is their sum divided by $T$, and the standard deviation is the root of the mean squared distance from the mean.

## Deltas: the slope of each coefficient

The obvious way to measure motion is the difference between neighboring frames, $c_{t+1} - c_{t}$. It is noisy, because noise in either of two frames goes straight into the answer. A steadier estimate uses more frames. Take a window of $\Theta$ frames on each side, fit a straight line through the $2\Theta + 1$ values of one coefficient by least squares, and read off its slope. Averaging over the window smooths out the noise.

The least-squares slope through points at offsets $-\Theta, \dots, \Theta$ is a weighted sum, and writing the left and right halves in pairs gives the following:

$$
d_t = \frac{\sum_{\theta=1}^{\Theta} \theta \left(c_{t+\theta} - c_{t-\theta}\right)}{2 \sum_{\theta=1}^{\Theta} \theta^2}
$$

Read it aloud: at frame $t$, take the difference between the frame $\theta$ ahead and the frame $\theta$ behind, weight it by $\theta$ so that distant frames count more, add those up, and divide by twice the sum of squares of the weights. For a straight line with slope $s$, every difference is $2\theta s$, so the result is exactly $s$. Frames before the start and after the end repeat the first and last frame. The unit of a delta is "coefficient units per frame".

```python
padded = np.pad(c, ((0, 0), (width, width)), mode="edge")
for theta in range(1, width + 1):
    num += theta * (padded[:, width + theta : width + theta + T] - padded[:, width - theta : width - theta + T])
```

*Delta-deltas* apply the same formula to the deltas, so they measure how the slope itself changes, the acceleration of the spectrum.

### Work the deltas by hand

Take one coefficient over five frames, $c = [1, 2, 4, 7, 11]$, and $\Theta = 2$, so the denominator is $2(1 + 4) = 10$. Padding repeats the ends: $[1, 1, \mathbf{1, 2, 4, 7, 11}, 11, 11]$.

- $d_0 = [1(2 - 1) + 2(4 - 1)] / 10 = 0.7$
- $d_1 = [1(4 - 1) + 2(7 - 1)] / 10 = 1.5$
- $d_2 = [1(7 - 2) + 2(11 - 1)] / 10 = 2.5$
- $d_3 = [1(11 - 4) + 2(11 - 2)] / 10 = 2.5$
- $d_4 = [1(11 - 7) + 2(11 - 4)] / 10 = 1.8$

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="A coefficient over five frames and its delta: the delta tracks the slope of the coefficient">
<line x1="70.0" y1="227.1" x2="570.0" y2="227.1" class="stroke-grid"/>
<text x="64.0" y="231.1" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="194.3" x2="570.0" y2="194.3" class="stroke-grid"/>
<text x="64.0" y="198.3" class="fig-small" text-anchor="end">2</text>
<line x1="70.0" y1="161.4" x2="570.0" y2="161.4" class="stroke-grid"/>
<text x="64.0" y="165.4" class="fig-small" text-anchor="end">4</text>
<line x1="70.0" y1="128.6" x2="570.0" y2="128.6" class="stroke-grid"/>
<text x="64.0" y="132.6" class="fig-small" text-anchor="end">6</text>
<line x1="70.0" y1="95.7" x2="570.0" y2="95.7" class="stroke-grid"/>
<text x="64.0" y="99.7" class="fig-small" text-anchor="end">8</text>
<line x1="70.0" y1="62.9" x2="570.0" y2="62.9" class="stroke-grid"/>
<text x="64.0" y="66.9" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="64.0" y="34.0" class="fig-small" text-anchor="end">12</text>
<line x1="70.0" y1="260.0" x2="570.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted"/>
<text x="70.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="260.0" x2="195.0" y2="265.0" class="stroke-muted"/>
<text x="195.0" y="279.0" class="fig-small" text-anchor="middle">1</text>
<line x1="320.0" y1="260.0" x2="320.0" y2="265.0" class="stroke-muted"/>
<text x="320.0" y="279.0" class="fig-small" text-anchor="middle">2</text>
<line x1="445.0" y1="260.0" x2="445.0" y2="265.0" class="stroke-muted"/>
<text x="445.0" y="279.0" class="fig-small" text-anchor="middle">3</text>
<line x1="570.0" y1="260.0" x2="570.0" y2="265.0" class="stroke-muted"/>
<text x="570.0" y="279.0" class="fig-small" text-anchor="middle">4</text>
<text x="320.0" y="298.0" text-anchor="middle">frame t</text>
<text x="8.0" y="20.0" text-anchor="start">value</text>
<polyline points="70.0,210.7 195.0,194.3 320.0,161.4 445.0,112.1 570.0,46.4" class="stroke-a" fill="none"/>
<polyline points="70.0,215.6 195.0,202.5 320.0,186.1 445.0,186.1 570.0,197.6" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="210.7" r="5" class="fill-a"/>
<circle cx="70.0" cy="215.6" r="5" class="fill-b"/>
<text x="85.0" y="197.6" class="fig-small" text-anchor="start">1</text>
<text x="85.0" y="235.4" class="fig-small" text-anchor="start">0.7</text>
<circle cx="195.0" cy="194.3" r="5" class="fill-a"/>
<circle cx="195.0" cy="202.5" r="5" class="fill-b"/>
<text x="195.0" y="181.1" class="fig-small" text-anchor="middle">2</text>
<text x="195.0" y="222.2" class="fig-small" text-anchor="middle">1.5</text>
<circle cx="320.0" cy="161.4" r="5" class="fill-a"/>
<circle cx="320.0" cy="186.1" r="5" class="fill-b"/>
<text x="320.0" y="148.3" class="fig-small" text-anchor="middle">4</text>
<text x="320.0" y="205.8" class="fig-small" text-anchor="middle">2.5</text>
<circle cx="445.0" cy="112.1" r="5" class="fill-a"/>
<circle cx="445.0" cy="186.1" r="5" class="fill-b"/>
<text x="445.0" y="99.0" class="fig-small" text-anchor="middle">7</text>
<text x="445.0" y="205.8" class="fig-small" text-anchor="middle">2.5</text>
<circle cx="570.0" cy="46.4" r="5" class="fill-a"/>
<circle cx="570.0" cy="197.6" r="5" class="fill-b"/>
<text x="570.0" y="33.3" class="fig-small" text-anchor="middle">11</text>
<text x="570.0" y="217.3" class="fig-small" text-anchor="middle">1.8</text>
<circle cx="95.0" cy="48.1" r="5" class="fill-a"/>
<text x="107.5" y="54.6" class="fig-small" text-anchor="start">c: one coefficient over five frames</text>
<circle cx="95.0" cy="71.1" r="5" class="fill-b"/>
<text x="107.5" y="77.6" class="fig-small" text-anchor="start">d: its delta, in units per frame</text>
</svg>
Figure 1. The coefficient c = [1, 2, 4, 7, 11] and its delta d = [0.7, 1.5, 2.5, 2.5, 1.8] with a window of 2 frames. The delta is small where c rises slowly, at frames 0 and 1, and large where it climbs steeply, at frames 2 and 3. The two ends, where frames are repeated, are pulled toward 0.
```

Each delta subtracts two frames, so a constant that sits in every frame cancels inside it. Keep that in mind: the other half of this chapter is about exactly such a constant.

## One microphone adds one constant

Record the same sentence through two microphones, and the MFCCs differ. The difference is simple, though. A fixed channel with frequency response $H(f)$ multiplies the power spectrum by $|H(f)|^2$. If $H$ changes slowly across each mel band, the mel power of band $b$ is multiplied by about $|H(f_b)|^2$, and the logarithm turns that product into a sum:

$$
\log M'_{b,t} = \log M_{b,t} + \log |H(f_b)|^2
$$

The added term depends on the band $b$ and not on the frame $t$. The DCT is linear, so every frame's cepstrum gains the same constant vector. The microphone is a fixed offset on every column.

Then there is an easy way to remove it: subtract each coefficient's average over the utterance. The average of the sum is the average of the speech plus the constant, so after subtracting it, the constant goes away. This is *cepstral mean normalization* (CMN). *Cepstral mean and variance normalization* (CMVN) also divides each row by its standard deviation, so every coefficient has the same spread:

$$
\hat c_t = \frac{c_t - \mu}{\sigma}, \qquad
\mu = \frac{1}{T} \sum_{t=1}^{T} c_t, \qquad
\sigma^2 = \frac{1}{T} \sum_{t=1}^{T} (c_t - \mu)^2
$$

Read it aloud: for each coefficient separately, subtract its mean over the $T$ frames of the utterance, and divide by its standard deviation over the same frames. Afterward, every row has mean 0 and variance 1. In numpy, with `keepdims=True` so the `(rows, 1)` statistics broadcast over time:

```python
normalized = (c - c.mean(axis=1, keepdims=True)) / np.maximum(c.std(axis=1, keepdims=True), eps)
```

For one row $[2, 4, 6, 8]$: the mean is 5, the variance is $(9 + 1 + 1 + 9)/4 = 5$, and $\sigma = 2.2361$. The normalized row is $[-1.342, -0.447, 0.447, 1.342]$. The `eps` guards a row that never changes, whose $\sigma$ is 0, so it comes out as zeros instead of NaN.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Coefficient 3 over 50 frames for a voice and for the same voice through a short filter: far apart before mean subtraction, overlapping after">
<line x1="60.0" y1="183.0" x2="290.0" y2="183.0" class="stroke-grid"/>
<text x="54.0" y="187.0" class="fig-small" text-anchor="end">10</text>
<line x1="60.0" y1="108.4" x2="290.0" y2="108.4" class="stroke-grid"/>
<text x="54.0" y="112.4" class="fig-small" text-anchor="end">20</text>
<line x1="60.0" y1="33.9" x2="290.0" y2="33.9" class="stroke-grid"/>
<text x="54.0" y="37.9" class="fig-small" text-anchor="end">30</text>
<line x1="60.0" y1="230.0" x2="290.0" y2="230.0" class="stroke-muted"/>
<line x1="60.0" y1="30.0" x2="60.0" y2="230.0" class="stroke-muted"/>
<line x1="60.0" y1="230.0" x2="60.0" y2="235.0" class="stroke-muted"/>
<text x="60.0" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="172.7" y1="230.0" x2="172.7" y2="235.0" class="stroke-muted"/>
<text x="172.7" y="249.0" class="fig-small" text-anchor="middle">24</text>
<line x1="290.0" y1="230.0" x2="290.0" y2="235.0" class="stroke-muted"/>
<text x="290.0" y="249.0" class="fig-small" text-anchor="middle">49</text>
<text x="175.0" y="268.0" text-anchor="middle">frame (10 ms)</text>
<text x="6.0" y="20.0" text-anchor="start">coefficient 3</text>
<polyline points="60.0,149.0 64.7,113.1 69.4,110.1 74.1,104.0 78.8,97.1 83.5,93.2 88.2,93.4 92.9,100.0 97.6,103.1 102.2,107.5 106.9,117.5 111.6,116.4 116.3,123.7 121.0,128.5 125.7,127.8 130.4,130.9 135.1,139.9 139.8,157.8 144.5,158.1 149.2,160.5 153.9,169.0 158.6,166.0 163.3,167.5 168.0,186.1 172.7,186.2 177.3,186.9 182.0,189.2 186.7,173.8 191.4,160.1 196.1,156.2 200.8,149.8 205.5,152.6 210.2,134.8 214.9,117.3 219.6,117.3 224.3,104.6 229.0,113.0 233.7,117.7 238.4,110.2 243.1,122.9 247.8,139.3 252.4,140.5 257.1,156.8 261.8,156.0 266.5,159.8 271.2,192.7 275.9,186.7 280.6,189.6 285.3,191.4 290.0,188.6" class="stroke-a" fill="none"/>
<polyline points="60.0,133.3 64.7,88.6 69.4,84.1 74.1,78.1 78.8,71.1 83.5,67.3 88.2,67.5 92.9,74.1 97.6,77.0 102.2,81.5 106.9,91.4 111.6,90.4 116.3,97.5 121.0,102.4 125.7,101.6 130.4,104.7 135.1,113.8 139.8,131.7 144.5,132.0 149.2,134.4 153.9,142.8 158.6,140.1 163.3,141.4 168.0,160.2 172.7,160.1 177.3,160.9 182.0,162.9 186.7,147.9 191.4,134.1 196.1,130.2 200.8,123.7 205.5,126.6 210.2,108.9 214.9,91.1 219.6,91.0 224.3,78.4 229.0,86.4 233.7,91.4 238.4,83.7 243.1,96.5 247.8,112.5 252.4,113.7 257.1,130.5 261.8,129.2 266.5,133.4 271.2,166.2 275.9,160.4 280.6,163.1 285.3,164.9 290.0,162.2" class="stroke-b" fill="none"/>
<text x="60.0" y="292.0" class="fig-small" text-anchor="start">blue: clean voice</text>
<text x="210.0" y="292.0" class="fig-small" text-anchor="start">orange: the same voice through the filter [1, 0.6, 0.3]</text>
<line x1="350.0" y1="220.8" x2="580.0" y2="220.8" class="stroke-grid"/>
<text x="344.0" y="224.8" class="fig-small" text-anchor="end">-10</text>
<line x1="350.0" y1="135.2" x2="580.0" y2="135.2" class="stroke-grid"/>
<text x="344.0" y="139.2" class="fig-small" text-anchor="end">0</text>
<line x1="350.0" y1="49.5" x2="580.0" y2="49.5" class="stroke-grid"/>
<text x="344.0" y="53.5" class="fig-small" text-anchor="end">10</text>
<line x1="350.0" y1="230.0" x2="580.0" y2="230.0" class="stroke-muted"/>
<line x1="350.0" y1="30.0" x2="350.0" y2="230.0" class="stroke-muted"/>
<line x1="350.0" y1="230.0" x2="350.0" y2="235.0" class="stroke-muted"/>
<text x="350.0" y="249.0" class="fig-small" text-anchor="middle">0</text>
<line x1="462.7" y1="230.0" x2="462.7" y2="235.0" class="stroke-muted"/>
<text x="462.7" y="249.0" class="fig-small" text-anchor="middle">24</text>
<line x1="580.0" y1="230.0" x2="580.0" y2="235.0" class="stroke-muted"/>
<text x="580.0" y="249.0" class="fig-small" text-anchor="middle">49</text>
<text x="465.0" y="268.0" text-anchor="middle">frame (10 ms)</text>
<text x="288.0" y="20.0" text-anchor="start">after subtracting each curve's mean</text>
<polyline points="350.0,136.9 354.7,95.7 359.4,92.3 364.1,85.2 368.8,77.3 373.5,72.8 378.2,73.0 382.9,80.7 387.6,84.2 392.2,89.3 396.9,100.8 401.6,99.5 406.3,107.8 411.0,113.4 415.7,112.6 420.4,116.2 425.1,126.5 429.8,147.1 434.5,147.4 439.2,150.1 443.9,159.9 448.6,156.5 453.3,158.2 458.0,179.6 462.7,179.7 467.3,180.5 472.0,183.1 476.7,165.4 481.4,149.7 486.1,145.2 490.8,137.8 495.5,141.1 500.2,120.7 504.9,100.6 509.6,100.5 514.3,86.0 519.0,95.6 523.7,101.0 528.4,92.4 533.1,107.0 537.8,125.8 542.4,127.2 547.1,145.9 551.8,144.9 556.5,149.4 561.2,187.2 565.9,180.3 570.6,183.6 575.3,185.6 580.0,182.4" class="stroke-a" fill="none"/>
<polyline points="350.0,148.8 354.7,97.4 359.4,92.3 364.1,85.4 368.8,77.4 373.5,72.9 378.2,73.2 382.9,80.8 387.6,84.1 392.2,89.2 396.9,100.7 401.6,99.5 406.3,107.7 411.0,113.3 415.7,112.4 420.4,116.0 425.1,126.4 429.8,146.9 434.5,147.2 439.2,150.1 443.9,159.7 448.6,156.6 453.3,158.1 458.0,179.7 462.7,179.6 467.3,180.5 472.0,182.8 476.7,165.5 481.4,149.8 486.1,145.2 490.8,137.8 495.5,141.1 500.2,120.7 504.9,100.3 509.6,100.2 514.3,85.8 519.0,94.9 523.7,100.7 528.4,91.9 533.1,106.5 537.8,124.9 542.4,126.3 547.1,145.6 551.8,144.1 556.5,148.9 561.2,186.6 565.9,180.0 570.6,183.0 575.3,185.2 580.0,182.0" class="stroke-b dashed" fill="none"/>
</svg>
Figure 2. Coefficient 3 of the exercise's voiced test signal, clean and convolved with the filter [1, 0.6, 0.3]. Left: the filter lifts every frame by about the same 3.5. Right: after each curve loses its own mean, the two lie almost on top of each other. Over all 13 coefficients, the mean difference falls from 4.29 to 0.11.
```

## The 39-dimensional feature

Thirteen MFCCs, 13 deltas, and 13 delta-deltas stack into 39 rows. You normalize after stacking, so the deltas get zero mean and unit variance too. Figure 3 shows the flow.

```figure
<svg viewBox="0 0 600 215" role="img" aria-label="Three blocks of 13 rows (MFCCs, deltas, delta-deltas) stacked into 39 rows and normalized">
<defs><marker id="c014-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<rect x="30" y="30" width="220" height="46" class="fill-soft-a stroke-muted"/>
<text x="140.0" y="58.0" class="fig-small fig-bold" text-anchor="middle">13 liftered MFCCs  c</text>
<rect x="30" y="80" width="220" height="46" class="fill-soft-b stroke-muted"/>
<text x="140.0" y="108.0" class="fig-small fig-bold" text-anchor="middle">13 deltas  d</text>
<rect x="30" y="130" width="220" height="46" class="fill-soft-c stroke-muted"/>
<text x="140.0" y="158.0" class="fig-small fig-bold" text-anchor="middle">13 delta-deltas  dd</text>
<text x="30.0" y="22.0" class="fig-small" text-anchor="start">rows of the feature matrix</text>
<line x1="256" y1="105" x2="330" y2="105" class="stroke-muted" marker-end="url(#c014-arrow)"/>
<text x="293.0" y="95.0" class="fig-small fig-bold" text-anchor="middle">CMVN</text>
<rect x="336" y="30" width="220" height="146" class="fill-sunk stroke-muted"/>
<text x="446.0" y="98.0" class="fig-small" text-anchor="middle">39 rows, each with</text>
<text x="446.0" y="118.0" class="fig-small" text-anchor="middle">mean 0 and variance 1</text>
<text x="30.0" y="200.0" class="fig-small" text-anchor="start">deltas of c give d; deltas of d give dd, so dd is the acceleration of the spectrum.</text>
</svg>
Figure 3. The classic 39-dimensional feature. The three blocks are stacked first and then normalized row by row, so every one of the 39 rows has mean 0 and variance 1 over the utterance.
```

> [!KEY]
> A delta is a slope, so it sees motion and ignores a constant. A fixed channel adds a constant to every frame's cepstrum, and subtracting each coefficient's utterance mean removes it.

> [!RECAP]
> - Deltas are least-squares slopes over $\pm\Theta$ frames with repeated edges, and a ramp gives its slope.
> - Delta-deltas are the deltas of the deltas.
> - A fixed channel adds a constant vector to every cepstral frame, because the log turns a product into a sum.
> - CMVN makes every row zero-mean and unit-variance over the utterance.
> - The classic feature stacks 13 MFCCs, 13 deltas, and 13 delta-deltas, then normalizes all 39 rows.

## Build it step by step

The front end from earlier chapters, `mel_power`, `dct2_ortho`, `power_to_db`, `mfcc`, and `lifter`, is provided in full in the starter. The checks test your three functions in this order.

1. **`deltas(c, width)`** pads with `mode="edge"` and sums weighted differences of shifted slices. The usual mistake is zero padding, which makes the first and last deltas jump.
2. **`cmvn(c, variance, eps)`** normalizes each row with `keepdims=True`. The usual mistake is normalizing each frame across coefficients, axis 0, instead of each coefficient across time, axis 1.
3. **`mfcc_features(x, sample_rate, n_mfcc)`** returns `(3 * n_mfcc, frames)`: liftered MFCCs from the provided `mel_power`, their deltas, and their delta-deltas, stacked and normalized with CMVN. The usual mistake is taking the delta-deltas from the MFCCs instead of from the deltas.

## Where this breaks

**Short utterances.** The mean of one second of speech depends on which sounds it holds, not only on the channel. CMN then removes real information. Normalize per speaker, or over a longer window.

**Streaming.** Kaldi normalizes per utterance, per speaker, or over a sliding window of 300 to 600 frames for live audio, where the whole utterance is not available yet.

**Liftering, then variance normalization.** CMVN divides each row by its own standard deviation, which cancels any fixed per-row scale, lifter included. Liftering matters only for raw distances or mean-only normalization.

**Additive noise.** CMN removes channels that multiply the spectrum. Noise adds to the power, which is not a constant after the logarithm, and CMN does not remove it.

**Silence and the floor.** The log of a silent band is the floor value. If one recording is quieter, more bands hit the floor, and the channel stops being a constant shift.

**Mismatched deltas.** librosa's `feature.delta` fits a Savitzky-Golay filter over 9 frames and interpolates at the edges. On a ramp it gives the slope everywhere, where the HTK formula gives 0.5 and 0.8 at the first two frames.

> [!WARNING]
> Compute CMVN statistics per utterance at training time and the same way at test time. Normalizing training data per speaker and test data per utterance shifts every feature the model sees.

## The exercise

You build `deltas`, `cmvn`, and `mfcc_features`. Press **Run** to hear three vowels clean and through a telephone band, see their 39 normalized features, and see how much mean normalization shrinks the gap between the two recordings.

The checks compare your deltas with the hand example, a ramp, and a brute-force loop at several widths. They check the CMVN example, zero mean and unit variance on every row, and a constant row that stays finite. The last two checks convolve a voice with a short filter and require mean normalization to remove at least 90% of the MFCC difference, and assemble the 39-dimensional features.
