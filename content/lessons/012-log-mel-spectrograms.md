---
title: Log-mel spectrograms
slug: 012-log-mel-spectrograms
part: "Part 2 — Hearing and speech production"
summary: Compress mel power with a logarithm, a floor, and a rescale to get Whisper's exact log-mel features, and see why each step is there.
minutes: 40
gpu: false
objectives:
  - Explain why a speech model sees the logarithm of mel power rather than the power itself.
  - "Apply Whisper's compression by hand: clamp, log10, floor at the global maximum minus 8, then (x + 4) / 4."
  - Say why the floor is global rather than per frame, and why Whisper drops the last STFT frame.
  - Reproduce Whisper's 80 by 3000 log-mel matrix for a 30-second chunk.
prereqs:
  - 007-the-short-time-fourier-transform
  - 011-hearing-and-the-mel-scale
exercise: 012-log-mel-spectrograms
---

# Log-mel spectrograms

Whisper's encoder never sees audio. It sees an 80 by 3000 grid of numbers: 80 mel bands, 3000 frames, 100 frames per second, for exactly 30 seconds. If you build that grid even slightly differently from the way the model's authors did, the model reads garbage, because every weight was trained on their numbers.

The mel filterbank from the last chapter gives you the bands. It leaves one question open: how should you write down the power in each band? Speech is loud and quiet in the same second. A vowel can carry a million times more power than the soft "f" next to it, and a model whose weights are of order 1 cannot read both a million and one-thousandth on the same page.

In this chapter you build the compression that fixes it, one step at a time, and you assemble Whisper's complete front end. Each step answers a failure of the step before it. Figure 1 shows the whole front end, with the shape of the data after each box. The first three boxes come from earlier chapters, and this chapter builds the last one.

```figure
<svg viewBox="0 0 640 190" role="img" aria-label="Whisper's front end as five boxes, from 480,000 audio samples to an 80 by 3000 feature matrix">
<defs><marker id="c012-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<rect x="10" y="40" width="108" height="70" rx="6" class="fill-soft-a stroke-muted"/>
<text x="64.0" y="66.0" class="fig-small fig-bold" text-anchor="middle">30 s of audio</text>
<text x="64.0" y="90.0" class="fig-small fig-mono" text-anchor="middle">480,000 samples</text>
<line x1="120" y1="75" x2="135" y2="75" class="stroke-muted" marker-end="url(#c012-arrow)"/>
<rect x="138" y="40" width="108" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="192.0" y="66.0" class="fig-small fig-bold" text-anchor="middle">power STFT</text>
<text x="192.0" y="90.0" class="fig-small fig-mono" text-anchor="middle">201 × 3001</text>
<line x1="248" y1="75" x2="263" y2="75" class="stroke-muted" marker-end="url(#c012-arrow)"/>
<rect x="266" y="40" width="108" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="320.0" y="66.0" class="fig-small fig-bold" text-anchor="middle">drop last frame</text>
<text x="320.0" y="90.0" class="fig-small fig-mono" text-anchor="middle">201 × 3000</text>
<line x1="376" y1="75" x2="391" y2="75" class="stroke-muted" marker-end="url(#c012-arrow)"/>
<rect x="394" y="40" width="108" height="70" rx="6" class="fill-sunk stroke-muted"/>
<text x="448.0" y="66.0" class="fig-small fig-bold" text-anchor="middle">80 mel filters</text>
<text x="448.0" y="90.0" class="fig-small fig-mono" text-anchor="middle">80 × 3000</text>
<line x1="504" y1="75" x2="519" y2="75" class="stroke-muted" marker-end="url(#c012-arrow)"/>
<rect x="522" y="40" width="108" height="70" rx="6" class="fill-soft-b stroke-muted"/>
<text x="576.0" y="66.0" class="fig-small fig-bold" text-anchor="middle">log, floor, rescale</text>
<text x="576.0" y="90.0" class="fig-small fig-mono" text-anchor="middle">80 × 3000</text>
<text x="12.0" y="150.0" class="fig-small" text-anchor="start">frame 400 samples (25 ms), hop 160 samples (10 ms), 16 kHz</text>
<text x="12.0" y="172.0" class="fig-small" text-anchor="start">201 = 400 / 2 + 1 frequency bins;  3001 = 1 + 480,000 / 160 frames;  3000 after the last frame is dropped</text>
</svg>
Figure 1. Whisper's front end, with the shape of the data after each step. The first three boxes reuse earlier chapters. This chapter builds the last box, the compression that turns mel power into the numbers the network sees.
```

## What you already know

- The STFT cuts a signal into overlapping windowed frames and takes the FFT of each. At 16 kHz, a 400-sample frame is 25 ms, and a hop of 160 samples is 10 ms.
- With `center=True`, a signal of $L$ samples has `1 + L // hop` frames.
- A mel filterbank of shape `(n_mels, n_fft // 2 + 1)` times a power spectrogram of shape `(bins, frames)` gives mel power of shape `(n_mels, frames)`.
- A decibel is ten times the base-10 logarithm of a power ratio, so every factor of 10 in power is 10 dB.
- A logarithm turns a product into a sum: $\log(ab) = \log a + \log b$.

## Why not feed in the mel power directly?

Suppose a loud vowel has a mel power of 1000 in one band and the soft consonant next to it has 0.001. A network sees those two numbers as inputs, and the ratio is a million to one. After the first layer, the vowel dominates every sum, and the consonant is rounded away. The consonant carries part of what separates one word from another, so losing it costs accuracy.

Your ear solves the same problem with a logarithm: perceived loudness grows roughly with the logarithm of power, not with power. So take the logarithm. The following table shows what it does to the numbers above.

| Mel power | $\log_{10}$ of it | Decibels |
|---|---|---|
| 1000 | 3 | 30 |
| 1 | 0 | 0 |
| 0.001 | −3 | −30 |

A million-to-one ratio becomes a gap of 6 units, and every factor of 10 in power is one unit. Both values now fit on the same page. The logarithm also makes volume additive. Halve the amplitude and every power drops by a factor of 4, so every $\log_{10}$ value drops by $\log_{10} 4 = 0.602$, the same amount in every band and frame. Scaling the recording shifts the picture. It does not reshape it.

## Clamp, so that zero does not become minus infinity

The logarithm has one hole. A band that holds exactly zero power, as digital silence does, gives $\log_{10} 0 = -\infty$, and one infinite entry poisons every sum the network computes. The fix is to clamp: replace every power below $10^{-10}$ with $10^{-10}$ before the logarithm. That is 100 dB below a power of 1, far under anything audible.

## Floor, so that the faint stays faint

A clamp at a fixed number is not enough. Real recordings contain faint noise, perhaps $10^{-7}$ in one band and $10^{-9}$ in another. The logarithm spreads these out into a wide range of values that carry no speech at all, and the network wastes capacity on them.

Whisper therefore raises everything more than 8 below the *largest* log value in the whole input up to that level. Eight decades of power is 80 dB, a wide range for speech. The floor is set by the loudest point of this clip, so a quiet recording and a loud one get the same dynamic range.

Why the whole input and not each frame? Take a frame of silence. Its own loudest value is tiny, so a per-frame floor 8 decades below it would let the noise in that frame show up at full contrast, and silence would look like speech. The global floor keeps silence flat, because "silent" means "far below the loudest thing said".

## Rescale, so that the values land near zero

After the floor, the values lie in a window of 8 units under the maximum. A network trains best on inputs near zero with a spread near 1, so Whisper maps the result by $(x + 4) / 4$. Dividing by 4 shrinks the 8-decade window to 2 units. For a clip whose loudest value is near 1, the window runs from $-7$ to 1, and adding 4 and dividing by 4 sends it to $-0.75$ through 1.25, close to zero. Whisper's values fall roughly between $-1$ and $1.5$.

The three steps together are the following:

$$
\begin{aligned}
L &= \log_{10} \max\left(M, 10^{-10}\right) \\
L' &= \max\left(L,\ \max_{b,t} L_{b,t} - 8\right) \\
Y &= \frac{L' + 4}{4}
\end{aligned}
$$

Read it aloud: $M$ is the mel power matrix. Clamp every entry to at least $10^{-10}$ and take its base-10 logarithm to get $L$. Find the largest entry over every band $b$ and frame $t$, and raise anything more than 8 below it to that floor, to get $L'$. Then add 4 and divide by 4. In numpy:

```python
log_spec = np.log10(np.maximum(mel_power, 1e-10))
log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
features = (log_spec + 4.0) / 4.0
```

### Work the compression by hand

Take a mel power matrix of two bands and three frames, `[[1, 1e-3, 0], [1e-5, 10, 1e-12]]`.

1. Clamp and take $\log_{10}$: `[[0, -3, -10], [-5, 1, -10]]`. The zero and the $10^{-12}$ both became $10^{-10}$ first.
2. The global maximum is 1, so the floor is $1 - 8 = -7$: `[[0, -3, -7], [-5, 1, -7]]`. The two clamped entries are lifted to the floor.
3. Add 4 and divide by 4: `[[1, 0.25, -0.75], [-0.25, 1.25, -0.75]]`.

```figure
<svg viewBox="0 0 640 160" role="img" aria-label="A two by three mel power matrix passing through log10 with a clamp, a floor, and the rescale (x + 4) / 4">
<defs><marker id="c012-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="80.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">mel power</text>
<rect x="14" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="36.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="58" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="80.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">1e-3</text>
<rect x="102" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="124.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="14" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="36.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">1e-5</text>
<rect x="58" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="80.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">10</text>
<rect x="102" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="124.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">1e-12</text>
<line x1="152" y1="75" x2="170" y2="75" class="stroke-muted" marker-end="url(#c012-arrow2)"/>
<text x="238.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">log10, clamped</text>
<rect x="172" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="194.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="216" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="238.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">-3</text>
<rect x="260" y="40" width="44" height="34" class="fill-soft-b stroke-muted"/>
<text x="282.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">-10</text>
<rect x="172" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="194.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-5</text>
<rect x="216" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="238.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="260" y="74" width="44" height="34" class="fill-soft-b stroke-muted"/>
<text x="282.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-10</text>
<line x1="310" y1="75" x2="328" y2="75" class="stroke-muted" marker-end="url(#c012-arrow2)"/>
<text x="396.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">floor at max − 8</text>
<rect x="330" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="352.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">0</text>
<rect x="374" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="396.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">-3</text>
<rect x="418" y="40" width="44" height="34" class="fill-soft-b stroke-muted"/>
<text x="440.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">-7</text>
<rect x="330" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="352.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-5</text>
<rect x="374" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="396.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="418" y="74" width="44" height="34" class="fill-soft-b stroke-muted"/>
<text x="440.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-7</text>
<line x1="468" y1="75" x2="486" y2="75" class="stroke-muted" marker-end="url(#c012-arrow2)"/>
<text x="554.0" y="28.0" class="fig-small fig-bold" text-anchor="middle">(x + 4) / 4</text>
<rect x="488" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="510.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">1</text>
<rect x="532" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="554.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">0.25</text>
<rect x="576" y="40" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="598.0" y="61.0" class="fig-small fig-mono" text-anchor="middle">-0.75</text>
<rect x="488" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="510.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-0.25</text>
<rect x="532" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="554.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">1.25</text>
<rect x="576" y="74" width="44" height="34" class="fill-sunk stroke-muted"/>
<text x="598.0" y="95.0" class="fig-small fig-mono" text-anchor="middle">-0.75</text>
<text x="14.0" y="128.0" class="fig-small" text-anchor="start">Shaded cells changed in that step: the zero and the 1e-12 were clamped to 1e-10 (log10 = −10),</text>
<text x="14.0" y="146.0" class="fig-small" text-anchor="start">then both were lifted to the floor, 8 decades under the largest value, which is 10 (log10 = 1).</text>
</svg>
Figure 2. The worked example. The floor is 1 − 8 = −7, so the two clamped cells land together at −7 and finish at −0.75. Notice that the quietest real value, 1e-5, finishes at −0.25 and stays above the floor.
```

Each decade of power moves the output by 0.25, until the floor. Below it, everything is the same.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="The compression curve: the feature value is flat below log10 power minus 7, then rises with slope one quarter">
<line x1="70.0" y1="237.0" x2="570.0" y2="237.0" class="stroke-grid"/>
<text x="64.0" y="241.0" class="fig-small" text-anchor="end">-0.75</text>
<line x1="70.0" y1="191.0" x2="570.0" y2="191.0" class="stroke-grid"/>
<text x="64.0" y="195.0" class="fig-small" text-anchor="end">-0.25</text>
<line x1="70.0" y1="145.0" x2="570.0" y2="145.0" class="stroke-grid"/>
<text x="64.0" y="149.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="99.0" x2="570.0" y2="99.0" class="stroke-grid"/>
<text x="64.0" y="103.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="53.0" x2="570.0" y2="53.0" class="stroke-grid"/>
<text x="64.0" y="57.0" class="fig-small" text-anchor="end">1.25</text>
<line x1="70.0" y1="260.0" x2="570.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted"/>
<text x="70.0" y="279.0" class="fig-small" text-anchor="middle">-12</text>
<line x1="141.4" y1="260.0" x2="141.4" y2="265.0" class="stroke-muted"/>
<text x="141.4" y="279.0" class="fig-small" text-anchor="middle">-10</text>
<line x1="212.9" y1="260.0" x2="212.9" y2="265.0" class="stroke-muted"/>
<text x="212.9" y="279.0" class="fig-small" text-anchor="middle">-8</text>
<line x1="284.3" y1="260.0" x2="284.3" y2="265.0" class="stroke-muted"/>
<text x="284.3" y="279.0" class="fig-small" text-anchor="middle">-6</text>
<line x1="355.7" y1="260.0" x2="355.7" y2="265.0" class="stroke-muted"/>
<text x="355.7" y="279.0" class="fig-small" text-anchor="middle">-4</text>
<line x1="427.1" y1="260.0" x2="427.1" y2="265.0" class="stroke-muted"/>
<text x="427.1" y="279.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="498.6" y1="260.0" x2="498.6" y2="265.0" class="stroke-muted"/>
<text x="498.6" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="570.0" y1="260.0" x2="570.0" y2="265.0" class="stroke-muted"/>
<text x="570.0" y="279.0" class="fig-small" text-anchor="middle">2</text>
<text x="320.0" y="298.0" text-anchor="middle">log10 of mel power (power of ten)</text>
<text x="8.0" y="20.0" text-anchor="start">feature value</text>
<polyline points="70.0,237.0 71.7,237.0 73.3,237.0 75.0,237.0 76.7,237.0 78.4,237.0 80.0,237.0 81.7,237.0 83.4,237.0 85.1,237.0 86.7,237.0 88.4,237.0 90.1,237.0 91.7,237.0 93.4,237.0 95.1,237.0 96.8,237.0 98.4,237.0 100.1,237.0 101.8,237.0 103.4,237.0 105.1,237.0 106.8,237.0 108.5,237.0 110.1,237.0 111.8,237.0 113.5,237.0 115.2,237.0 116.8,237.0 118.5,237.0 120.2,237.0 121.8,237.0 123.5,237.0 125.2,237.0 126.9,237.0 128.5,237.0 130.2,237.0 131.9,237.0 133.5,237.0 135.2,237.0 136.9,237.0 138.6,237.0 140.2,237.0 141.9,237.0 143.6,237.0 145.3,237.0 146.9,237.0 148.6,237.0 150.3,237.0 151.9,237.0 153.6,237.0 155.3,237.0 157.0,237.0 158.6,237.0 160.3,237.0 162.0,237.0 163.6,237.0 165.3,237.0 167.0,237.0 168.7,237.0 170.3,237.0 172.0,237.0 173.7,237.0 175.4,237.0 177.0,237.0 178.7,237.0 180.4,237.0 182.0,237.0 183.7,237.0 185.4,237.0 187.1,237.0 188.7,237.0 190.4,237.0 192.1,237.0 193.7,237.0 195.4,237.0 197.1,237.0 198.8,237.0 200.4,237.0 202.1,237.0 203.8,237.0 205.5,237.0 207.1,237.0 208.8,237.0 210.5,237.0 212.1,237.0 213.8,237.0 215.5,237.0 217.2,237.0 218.8,237.0 220.5,237.0 222.2,237.0 223.8,237.0 225.5,237.0 227.2,237.0 228.9,237.0 230.5,237.0 232.2,237.0 233.9,237.0 235.6,237.0 237.2,237.0 238.9,237.0 240.6,237.0 242.2,237.0 243.9,237.0 245.6,237.0 247.3,237.0 248.9,236.8 250.6,235.7 252.3,234.6 253.9,233.5 255.6,232.5 257.3,231.4 259.0,230.3 260.6,229.2 262.3,228.2 264.0,227.1 265.7,226.0 267.3,224.9 269.0,223.8 270.7,222.8 272.3,221.7 274.0,220.6 275.7,219.5 277.4,218.5 279.0,217.4 280.7,216.3 282.4,215.2 284.0,214.2 285.7,213.1 287.4,212.0 289.1,210.9 290.7,209.8 292.4,208.8 294.1,207.7 295.8,206.6 297.4,205.5 299.1,204.5 300.8,203.4 302.4,202.3 304.1,201.2 305.8,200.2 307.5,199.1 309.1,198.0 310.8,196.9 312.5,195.8 314.1,194.8 315.8,193.7 317.5,192.6 319.2,191.5 320.8,190.5 322.5,189.4 324.2,188.3 325.9,187.2 327.5,186.2 329.2,185.1 330.9,184.0 332.5,182.9 334.2,181.8 335.9,180.8 337.6,179.7 339.2,178.6 340.9,177.5 342.6,176.5 344.2,175.4 345.9,174.3 347.6,173.2 349.3,172.2 350.9,171.1 352.6,170.0 354.3,168.9 356.0,167.8 357.6,166.8 359.3,165.7 361.0,164.6 362.6,163.5 364.3,162.5 366.0,161.4 367.7,160.3 369.3,159.2 371.0,158.2 372.7,157.1 374.3,156.0 376.0,154.9 377.7,153.8 379.4,152.8 381.0,151.7 382.7,150.6 384.4,149.5 386.1,148.5 387.7,147.4 389.4,146.3 391.1,145.2 392.7,144.2 394.4,143.1 396.1,142.0 397.8,140.9 399.4,139.8 401.1,138.8 402.8,137.7 404.4,136.6 406.1,135.5 407.8,134.5 409.5,133.4 411.1,132.3 412.8,131.2 414.5,130.2 416.2,129.1 417.8,128.0 419.5,126.9 421.2,125.8 422.8,124.8 424.5,123.7 426.2,122.6 427.9,121.5 429.5,120.5 431.2,119.4 432.9,118.3 434.5,117.2 436.2,116.2 437.9,115.1 439.6,114.0 441.2,112.9 442.9,111.8 444.6,110.8 446.3,109.7 447.9,108.6 449.6,107.5 451.3,106.5 452.9,105.4 454.6,104.3 456.3,103.2 458.0,102.2 459.6,101.1 461.3,100.0 463.0,98.9 464.6,97.8 466.3,96.8 468.0,95.7 469.7,94.6 471.3,93.5 473.0,92.5 474.7,91.4 476.4,90.3 478.0,89.2 479.7,88.2 481.4,87.1 483.0,86.0 484.7,84.9 486.4,83.8 488.1,82.8 489.7,81.7 491.4,80.6 493.1,79.5 494.7,78.5 496.4,77.4 498.1,76.3 499.8,75.2 501.4,74.2 503.1,73.1 504.8,72.0 506.5,70.9 508.1,69.8 509.8,68.8 511.5,67.7 513.1,66.6 514.8,65.5 516.5,64.5 518.2,63.4 519.8,62.3 521.5,61.2 523.2,60.2 524.8,59.1 526.5,58.0 528.2,56.9 529.9,55.8 531.5,54.8 533.2,53.7 534.9,52.6 536.6,51.5 538.2,50.5 539.9,49.4 541.6,48.3 543.2,47.2 544.9,46.2 546.6,45.1 548.3,44.0 549.9,42.9 551.6,41.8 553.3,40.8 554.9,39.7 556.6,38.6 558.3,37.5 560.0,36.5 561.6,35.4 563.3,34.3 565.0,33.2 566.7,32.2 568.3,31.1 570.0,30.0" class="stroke-b" fill="none"/>
<line x1="248.6" y1="260.0" x2="248.6" y2="30.0" class="stroke-muted dashed"/>
<circle cx="70.0" cy="237.0" r="4.5" class="fill-a"/>
<circle cx="320.0" cy="191.0" r="4.5" class="fill-a"/>
<circle cx="391.4" cy="145.0" r="4.5" class="fill-a"/>
<circle cx="498.6" cy="76.0" r="4.5" class="fill-a"/>
<circle cx="534.3" cy="53.0" r="4.5" class="fill-a"/>
<text x="255.7" y="250.8" class="fig-small" text-anchor="start">floor: 1 − 8 = −7</text>
<text x="77.1" y="218.6" class="fig-small" text-anchor="start">flat: everything quieter</text>
<text x="77.1" y="229.6" class="fig-small" text-anchor="start">than the floor looks the same</text>
<text x="362.9" y="80.6" class="fig-small" text-anchor="end">slope 1/4 above the floor</text>
<text x="541.4" y="61.3" class="fig-small" text-anchor="start">10</text>
</svg>
Figure 3. How one mel power becomes one feature value in the worked example, whose loudest entry is 10. The five dots are its entries 1e-12, 1e-5, 1e-3, 1, and 10. Every decade of power adds 0.25, and everything below the floor collapses to −0.75. The entry 0 has no logarithm, so it is clamped first and lands on the flat part as well.
```

> [!KEY]
> The log makes loudness additive, the clamp keeps zero finite, the global floor keeps quiet noise flat, and $(x + 4)/4$ puts the values near zero. Dividing the amplitude by 10 lowers every output by exactly 0.5, as long as no entry reaches the clamp, because power drops by 100, which is 2 decades, and 2 / 4 = 0.5.

## Whisper's whole front end

Whisper fixes the sample rate at 16 kHz and chains the five steps of Figure 1. The first three reuse earlier chapters. The last two are the compression you just worked.

1. Take the STFT with a 400-point periodic Hann window, hop 160, and `center=True` reflect padding.
2. Drop the last frame. A clip of $L$ samples has $1 + \lfloor L/160 \rfloor$ frames, and Whisper keeps $\lfloor L/160 \rfloor$. For 30 seconds, 480,000 samples, that is 3001 frames, minus one, 3000.
3. Take the power $|X|^2$ and multiply by the 80-filter Slaney bank, which is 128 filters for large-v3. Call the result $M$.
4. Clamp, take $\log_{10}$, and floor.
5. Rescale.

Whisper reads audio in 30-second chunks and pads a shorter clip with zeros at the end. The encoder's first layers include a convolution with stride 2, so the transformer sees 1500 positions, one every 20 ms. The following table compares this front end with three you will meet, because a mismatch in any row changes the numbers.

| Setting | Whisper | librosa `melspectrogram` | `torchaudio` `MelSpectrogram` | Kaldi `fbank` |
|---|---|---|---|---|
| Frame and hop | 400 and 160 samples | 2048 and 512 | 400 and 200 | 400 (FFT of 512) and 160 |
| Window | periodic Hann | periodic Hann | periodic Hann | Povey |
| Padding | center, reflect | center, zeros | center, reflect | none |
| Mel scale and norm | Slaney, `"slaney"` | Slaney, `"slaney"` | HTK, none | HTK, none |
| Log | $\log_{10}$, clamp $10^{-10}$ | none | none | $\ln$ |

> [!RECAP]
> - Raw mel power spans a million to one in one second of speech; the logarithm turns that ratio into 6 units.
> - The clamp at $10^{-10}$ keeps $\log_{10}$ finite for digital silence.
> - The floor sits 8 below the loudest value in the whole input, so silence stays flat.
> - $(x + 4)/4$ maps the result near $-1$ to $1.5$.
> - Whisper drops the last STFT frame, so 30 seconds give 3000 frames.

## Build it step by step

The mel filterbank and the power spectrogram are provided in the starter in full. You build the last three pieces, and the checks test them in this order.

1. **`whisper_normalize(mel_power)`** applies the clamp, the $\log_{10}$, the global floor, and the rescale, and returns the same shape. The usual mistake is a per-frame floor, `log_spec.max(axis=0)`, which lifts silent frames into noise.
2. **`pad_or_trim(audio, length)`** zero-pads at the end with `np.pad` or slices to exactly `length` samples. The usual mistake is padding at the start, which shifts every frame.
3. **`whisper_log_mel(audio, n_mels)`** returns `(n_mels, len(audio) // 160)` and chains the provided power spectrogram, a slice `[:, :-1]`, the provided filterbank, and `whisper_normalize`. The usual mistake is keeping 3001 frames for 30 seconds.

## Where this breaks

**Magnitude instead of power.** Whisper and librosa square the magnitude. Without the square, the features have about half the dynamic range, and the filters weight the bins differently.

**The wrong logarithm.** $\ln$ and $\log_{10}$ differ by a factor of 2.303. Kaldi uses $\ln$, and Whisper uses $\log_{10}$.

**A per-frame floor.** Whisper floors against the loudest value in the whole 30-second chunk. A per-frame floor lifts silent frames into noise.

**Off-by-one frames.** A centered STFT of 480,000 samples has 3001 frames, and Whisper's encoder expects 3000.

**The wrong sample rate.** `librosa.load` resamples to 22,050 Hz unless you pass `sr=None` or `sr=16000`. Whisper expects 16 kHz, and features from another rate put every formant in the wrong band.

> [!WARNING]
> Kaldi feeds its front end integers on the 16-bit scale, from -32768 to 32767. Floats in $[-1, 1]$ shift every natural-log mel value by $\ln(32768^2) = 20.79$. Scale the input the way the model was trained.

## The exercise

You build `whisper_normalize`, `pad_or_trim`, and `whisper_log_mel`. Press **Run** to see Whisper's log-mel of a linear sweep, which bends on the mel axis, and to confirm that a clip 10 times quieter, 20 dB down, shifts every feature by $-0.5$.

The checks work the 2 by 3 compression example and show that the floor is global. They check that `pad_or_trim` pads and cuts to 480,000 samples, and match `whisper_log_mel` to Whisper's own `log_mel_spectrogram` within 1e-4, for 80 and 128 mels and for a padded chunk of shape `(80, 3000)`.
