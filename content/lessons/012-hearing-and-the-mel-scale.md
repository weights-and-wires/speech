---
title: Hearing and the mel scale
slug: 012-hearing-and-the-mel-scale
part: "Part 2 — Hearing and speech production"
summary: Space frequency bands the way the ear does, with the mel scale and a bank of triangular filters that turns 201 FFT bins into a few dozen bands.
minutes: 40
gpu: false
objectives:
  - Explain why a speech front end spends its bands on low frequencies, using how the cochlea analyzes sound.
  - Convert between hertz and mel on the HTK and Slaney scales, and say where the two disagree.
  - Build a triangular mel filterbank from mel-spaced edges, by hand for two filters and in numpy for eighty.
  - Say why Slaney normalization gives every triangle the same area, and what happens to filters narrower than one FFT bin.
prereqs:
  - 007-the-short-time-fourier-transform
exercise: 012-hearing-and-the-mel-scale
---

# Hearing and the mel scale

Play a tone at 100 Hz, then one at 200 Hz. The jump is large: you hear the pitch climb a full octave. Now play 4000 Hz, then 4100 Hz. The step is the same 100 Hz, and you can barely tell the two apart.

A spectrogram from the short-time Fourier transform ignores this. At 16 kHz with a 400-sample frame, it puts one frequency bin every 40 Hz, evenly, from 0 Hz to 8000 Hz. The step from 100 Hz to 200 Hz gets 2.5 bins and so does the step from 4000 Hz to 4100 Hz, although your ear treats them very differently. Most of the 201 bins sit where hearing is coarse.

This chapter answers one question: how do you space frequency bands so they match what the ear can tell apart? The answer is the *mel scale* and a *mel filterbank*, a few dozen overlapping triangles that are narrow at low frequencies and wide at high ones. Their outputs feed nearly every speech model in this course.

## What you already know

- A spectrogram has shape `(bins, frames)`, and row `k` is the frequency `k * sample_rate / n_fft`.
- A matrix product `W @ S` mixes the rows of `S`, so each output row is a weighted sum of input rows.

## How the ear spaces its frequency bands

Sound enters the inner ear at the *cochlea*, a coiled tube. Inside it runs the *basilar membrane*, stiff at the entrance and floppy at the far end. A high frequency makes the stiff end vibrate, and a low frequency travels further. Each place along the membrane responds to its own band of frequencies, so the cochlea behaves like a bank of band-pass filters laid out along a line.

Those filters are not evenly spaced. Below about 500 Hz, each is roughly 100 Hz wide. Above that, the width grows in proportion to the center frequency, at about 20% of it. Two tones in the same band, called a *critical band*, interact strongly, and a loud one can hide a quiet one. Around 24 critical bands cover the range of hearing.

This explains the opening experiment. At 100 Hz, a 100 Hz step crosses a whole band. At 4000 Hz it covers about an eighth of one, because the band there is near 800 Hz wide. The ear counts bands, not hertz, so a front end needs bands that widen as frequency grows.

## Why not just take the logarithm of frequency?

The widths grow in proportion to the center frequency, so a logarithmic frequency axis looks like the obvious choice. It fails at the bottom. The logarithm of 0 Hz is minus infinity, and below about 500 Hz the ear's bandwidth stops shrinking and stays near 100 Hz. The scale you want is straight at the bottom and logarithmic at the top.

The *mel scale* is that scale. It came from listening tests in the 1930s, where people tuned one tone until it sounded half as high as another, so equal steps in mel sound like equal steps in pitch. Two formulas for it dominate speech code, and they are not interchangeable.

### The HTK formula

HTK, Kaldi, and `torchaudio` use one logarithm:

$$
m_{\text{HTK}}(f) = 2595 \log_{10}\left(1 + \frac{f}{700}\right), \qquad f = 700\left(10^{m / 2595} - 1\right)
$$

Read it aloud: divide the frequency $f$ in hertz by 700, add 1, take the base-10 logarithm, and scale by 2595. The second formula undoes the first. Adding 1 keeps $f = 0$ at $m = 0$ and makes the curve nearly straight for small $f$. The constants put 1000 Hz at about 1000 mel.

```python
mel = 2595.0 * np.log10(1.0 + f / 700.0)
```

### The Slaney formula

The Slaney formula, from Malcolm Slaney's Auditory Toolbox, is the default in librosa and Whisper. It is exactly linear below 1000 Hz and exactly logarithmic above.

$$
m_{\text{Slaney}}(f) =
\begin{cases}
\dfrac{3f}{200}, & f < 1000 \\[2mm]
15 + \dfrac{27 \ln(f / 1000)}{\ln 6.4}, & f \ge 1000
\end{cases}
$$

Read it aloud: below 1 kHz, every 200/3 Hz, about 66.7 Hz, is one mel, so 1000 Hz is 15 mel. Above 1 kHz, every factor of 6.4 in frequency adds 27 mel. The inverse switches branches at 15 mel, not at 1000.

```python
log_branch = 15.0 + np.log(np.maximum(f, 1000.0) / 1000.0) / (np.log(6.4) / 27.0)
mel = np.where(f >= 1000.0, log_branch, f / (200.0 / 3.0))
```

`np.where` computes both branches for every `f`, so `np.maximum(f, 1000.0)` keeps a small frequency out of the logarithm.

### Check the scales on the opening experiment

Take $f = 440$ Hz. On the HTK scale, $2595 \log_{10}(1 + 440/700) = 549.64$ mel. On the Slaney scale, 440 Hz is below 1 kHz, so it is $3 \times 440 / 200 = 6.60$ mel. The following table lists more values, and your code must reproduce it.

| Frequency (Hz) | HTK (mel) | Slaney (mel) |
|---|---|---|
| 100 | 150.49 | 1.50 |
| 440 | 549.64 | 6.60 |
| 1000 | 999.99 | 15.00 |
| 2000 | 1521.36 | 25.08 |
| 4000 | 2146.06 | 35.16 |
| 8000 | 2840.02 | 45.25 |

The absolute numbers differ by a factor of about 60, which does not matter, because a filterbank only uses equal spacing on the scale. Return to the opening: 100 Hz to 200 Hz is 1.50 to 3.00 mel on the Slaney scale, a step of 1.50. The step from 4000 Hz to 4100 Hz is 35.16 to 35.52 mel, a step of 0.36. The scale agrees with your ear that the low step is about four times bigger.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="The Slaney mel scale against frequency in hertz: a steep rise below 1 kHz that flattens above it, with two 100 Hz steps marked">
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-grid"/>
<text x="64.0" y="274.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="220.0" x2="570.0" y2="220.0" class="stroke-grid"/>
<text x="64.0" y="224.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="170.0" x2="570.0" y2="170.0" class="stroke-grid"/>
<text x="64.0" y="174.0" class="fig-small" text-anchor="end">20</text>
<line x1="70.0" y1="120.0" x2="570.0" y2="120.0" class="stroke-grid"/>
<text x="64.0" y="124.0" class="fig-small" text-anchor="end">30</text>
<line x1="70.0" y1="70.0" x2="570.0" y2="70.0" class="stroke-grid"/>
<text x="64.0" y="74.0" class="fig-small" text-anchor="end">40</text>
<line x1="70.0" y1="270.0" x2="570.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="270.0" class="stroke-muted"/>
<line x1="70.0" y1="270.0" x2="70.0" y2="275.0" class="stroke-muted"/>
<text x="70.0" y="289.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="270.0" x2="195.0" y2="275.0" class="stroke-muted"/>
<text x="195.0" y="289.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="320.0" y1="270.0" x2="320.0" y2="275.0" class="stroke-muted"/>
<text x="320.0" y="289.0" class="fig-small" text-anchor="middle">4000</text>
<line x1="445.0" y1="270.0" x2="445.0" y2="275.0" class="stroke-muted"/>
<text x="445.0" y="289.0" class="fig-small" text-anchor="middle">6000</text>
<line x1="570.0" y1="270.0" x2="570.0" y2="275.0" class="stroke-muted"/>
<text x="570.0" y="289.0" class="fig-small" text-anchor="middle">8000</text>
<text x="320.0" y="308.0" text-anchor="middle">frequency (Hz)</text>
<text x="8.0" y="20.0" text-anchor="start">mel (Slaney)</text>
<polyline points="70.0,270.0 71.3,268.5 72.5,267.0 73.8,265.5 75.0,264.0 76.3,262.5 77.5,261.0 78.8,259.5 80.0,258.0 81.3,256.5 82.5,255.0 83.8,253.5 85.0,252.0 86.3,250.5 87.5,248.9 88.8,247.4 90.1,245.9 91.3,244.4 92.6,242.9 93.8,241.4 95.1,239.9 96.3,238.4 97.6,236.9 98.8,235.4 100.1,233.9 101.3,232.4 102.6,230.9 103.8,229.4 105.1,227.9 106.3,226.4 107.6,224.9 108.8,223.4 110.1,221.9 111.4,220.4 112.6,218.9 113.9,217.4 115.1,215.9 116.4,214.4 117.6,212.9 118.9,211.4 120.1,209.8 121.4,208.3 122.6,206.8 123.9,205.3 125.1,203.8 126.4,202.3 127.6,200.8 128.9,199.3 130.2,197.8 131.4,196.3 132.7,194.8 133.9,193.3 135.2,191.8 136.4,190.3 137.7,188.8 138.9,187.3 140.2,185.8 141.4,184.3 142.7,182.8 143.9,181.3 145.2,179.8 146.4,178.3 147.7,176.8 148.9,175.3 150.2,173.8 151.5,172.3 152.7,170.8 154.0,169.2 155.2,167.7 156.5,166.2 157.7,164.7 159.0,163.2 160.2,161.7 161.5,160.2 162.7,158.7 164.0,157.2 165.2,155.7 166.5,154.2 167.7,152.7 169.0,151.2 170.3,149.7 171.5,148.2 172.8,146.7 174.0,145.2 175.3,143.7 176.5,142.2 177.8,140.7 179.0,139.2 180.3,137.7 181.5,136.2 182.8,134.7 184.0,133.2 185.3,131.7 186.5,130.2 187.8,128.6 189.0,127.1 190.3,125.6 191.6,124.1 192.8,122.6 194.1,121.1 195.3,119.6 196.6,118.1 197.8,116.6 199.1,115.1 200.3,113.6 201.6,112.1 202.8,110.6 204.1,109.1 205.3,107.6 206.6,106.1 207.8,104.6 209.1,103.1 210.4,101.6 211.6,100.1 212.9,98.6 214.1,97.1 215.4,95.6 216.6,94.1 217.9,92.6 219.1,91.1 220.4,89.5 221.6,88.0 222.9,86.5 224.1,85.0 225.4,83.5 226.6,82.0 227.9,80.5 229.1,79.0 230.4,77.5 231.7,76.0 232.9,74.5 234.2,73.0 235.4,71.5 236.7,70.0 237.9,68.5 239.2,67.0 240.4,65.5 241.7,64.0 242.9,62.5 244.2,61.0 245.4,59.5 246.7,58.0 247.9,56.5 249.2,55.0 250.5,53.5 251.7,52.0 253.0,50.5 254.2,48.9 255.5,47.4 256.7,45.9 258.0,44.4 259.2,42.9 260.5,41.4 261.7,39.9 263.0,38.4 264.2,36.9 265.5,35.4 266.7,33.9 268.0,32.4 269.2,30.9 270.5,29.4 271.8,27.9 273.0,26.4 274.3,24.9 275.5,23.4 276.8,21.9 278.0,20.4 279.3,18.9 280.5,17.4 281.8,15.9 283.0,14.4 284.3,12.9 285.5,11.4 286.8,9.8 288.0,8.3 289.3,6.8 290.6,5.3 291.8,3.8 293.1,2.3 294.3,0.8 295.6,-0.7 296.8,-2.2 298.1,-3.7 299.3,-5.2 300.6,-6.7 301.8,-8.2 303.1,-9.7 304.3,-11.2 305.6,-12.7 306.8,-14.2 308.1,-15.7 309.3,-17.2 310.6,-18.7 311.9,-20.2 313.1,-21.7 314.4,-23.2 315.6,-24.7 316.9,-26.2 318.1,-27.7 319.4,-29.2 320.6,-30.8 321.9,-32.3 323.1,-33.8 324.4,-35.3 325.6,-36.8 326.9,-38.3 328.1,-39.8 329.4,-41.3 330.7,-42.8 331.9,-44.3 333.2,-45.8 334.4,-47.3 335.7,-48.8 336.9,-50.3 338.2,-51.8 339.4,-53.3 340.7,-54.8 341.9,-56.3 343.2,-57.8 344.4,-59.3 345.7,-60.8 346.9,-62.3 348.2,-63.8 349.4,-65.3 350.7,-66.8 352.0,-68.3 353.2,-69.8 354.5,-71.4 355.7,-72.9 357.0,-74.4 358.2,-75.9 359.5,-77.4 360.7,-78.9 362.0,-80.4 363.2,-81.9 364.5,-83.4 365.7,-84.9 367.0,-86.4 368.2,-87.9 369.5,-89.4 370.8,-90.9 372.0,-92.4 373.3,-93.9 374.5,-95.4 375.8,-96.9 377.0,-98.4 378.3,-99.9 379.5,-101.4 380.8,-102.9 382.0,-104.4 383.3,-105.9 384.5,-107.4 385.8,-108.9 387.0,-110.5 388.3,-112.0 389.5,-113.5 390.8,-115.0 392.1,-116.5 393.3,-118.0 394.6,-119.5 395.8,-121.0 397.1,-122.5 398.3,-124.0 399.6,-125.5 400.8,-127.0 402.1,-128.5 403.3,-130.0 404.6,-131.5 405.8,-133.0 407.1,-134.5 408.3,-136.0 409.6,-137.5 410.9,-139.0 412.1,-140.5 413.4,-142.0 414.6,-143.5 415.9,-145.0 417.1,-146.5 418.4,-148.0 419.6,-149.5 420.9,-151.1 422.1,-152.6 423.4,-154.1 424.6,-155.6 425.9,-157.1 427.1,-158.6 428.4,-160.1 429.6,-161.6 430.9,-163.1 432.2,-164.6 433.4,-166.1 434.7,-167.6 435.9,-169.1 437.2,-170.6 438.4,-172.1 439.7,-173.6 440.9,-175.1 442.2,-176.6 443.4,-178.1 444.7,-179.6 445.9,-181.1 447.2,-182.6 448.4,-184.1 449.7,-185.6 451.0,-187.1 452.2,-188.6 453.5,-190.2 454.7,-191.7 456.0,-193.2 457.2,-194.7 458.5,-196.2 459.7,-197.7 461.0,-199.2 462.2,-200.7 463.5,-202.2 464.7,-203.7 466.0,-205.2 467.2,-206.7 468.5,-208.2 469.7,-209.7 471.0,-211.2 472.3,-212.7 473.5,-214.2 474.8,-215.7 476.0,-217.2 477.3,-218.7 478.5,-220.2 479.8,-221.7 481.0,-223.2 482.3,-224.7 483.5,-226.2 484.8,-227.7 486.0,-229.2 487.3,-230.8 488.5,-232.3 489.8,-233.8 491.1,-235.3 492.3,-236.8 493.6,-238.3 494.8,-239.8 496.1,-241.3 497.3,-242.8 498.6,-244.3 499.8,-245.8 501.1,-247.3 502.3,-248.8 503.6,-250.3 504.8,-251.8 506.1,-253.3 507.3,-254.8 508.6,-256.3 509.8,-257.8 511.1,-259.3 512.4,-260.8 513.6,-262.3 514.9,-263.8 516.1,-265.3 517.4,-266.8 518.6,-268.3 519.9,-269.8 521.1,-271.4 522.4,-272.9 523.6,-274.4 524.9,-275.9 526.1,-277.4 527.4,-278.9 528.6,-280.4 529.9,-281.9 531.2,-283.4 532.4,-284.9 533.7,-286.4 534.9,-287.9 536.2,-289.4 537.4,-290.9 538.7,-292.4 539.9,-293.9 541.2,-295.4 542.4,-296.9 543.7,-298.4 544.9,-299.9 546.2,-301.4 547.4,-302.9 548.7,-304.4 549.9,-305.9 551.2,-307.4 552.5,-308.9 553.7,-310.5 555.0,-312.0 556.2,-313.5 557.5,-315.0 558.7,-316.5 560.0,-318.0 561.2,-319.5 562.5,-321.0 563.7,-322.5 565.0,-324.0 566.2,-325.5 567.5,-327.0 568.7,-328.5 570.0,-330.0" class="stroke-muted dashed" fill="none"/>
<polyline points="70.0,270.0 71.3,268.5 72.5,267.0 73.8,265.5 75.0,264.0 76.3,262.5 77.5,261.0 78.8,259.5 80.0,258.0 81.3,256.5 82.5,255.0 83.8,253.5 85.0,252.0 86.3,250.5 87.5,248.9 88.8,247.4 90.1,245.9 91.3,244.4 92.6,242.9 93.8,241.4 95.1,239.9 96.3,238.4 97.6,236.9 98.8,235.4 100.1,233.9 101.3,232.4 102.6,230.9 103.8,229.4 105.1,227.9 106.3,226.4 107.6,224.9 108.8,223.4 110.1,221.9 111.4,220.4 112.6,218.9 113.9,217.4 115.1,215.9 116.4,214.4 117.6,212.9 118.9,211.4 120.1,209.8 121.4,208.3 122.6,206.8 123.9,205.3 125.1,203.8 126.4,202.3 127.6,200.8 128.9,199.3 130.2,197.8 131.4,196.3 132.7,194.8 133.9,193.4 135.2,192.0 136.4,190.6 137.7,189.2 138.9,187.9 140.2,186.6 141.4,185.3 142.7,184.0 143.9,182.8 145.2,181.6 146.4,180.4 147.7,179.2 148.9,178.0 150.2,176.9 151.5,175.7 152.7,174.6 154.0,173.5 155.2,172.5 156.5,171.4 157.7,170.3 159.0,169.3 160.2,168.3 161.5,167.3 162.7,166.3 164.0,165.3 165.2,164.4 166.5,163.4 167.7,162.5 169.0,161.6 170.3,160.6 171.5,159.7 172.8,158.8 174.0,158.0 175.3,157.1 176.5,156.2 177.8,155.4 179.0,154.5 180.3,153.7 181.5,152.9 182.8,152.1 184.0,151.3 185.3,150.5 186.5,149.7 187.8,148.9 189.0,148.1 190.3,147.4 191.6,146.6 192.8,145.9 194.1,145.1 195.3,144.4 196.6,143.7 197.8,143.0 199.1,142.3 200.3,141.6 201.6,140.9 202.8,140.2 204.1,139.5 205.3,138.8 206.6,138.1 207.8,137.5 209.1,136.8 210.4,136.2 211.6,135.5 212.9,134.9 214.1,134.2 215.4,133.6 216.6,133.0 217.9,132.4 219.1,131.8 220.4,131.1 221.6,130.5 222.9,129.9 224.1,129.4 225.4,128.8 226.6,128.2 227.9,127.6 229.1,127.0 230.4,126.5 231.7,125.9 232.9,125.3 234.2,124.8 235.4,124.2 236.7,123.7 237.9,123.1 239.2,122.6 240.4,122.0 241.7,121.5 242.9,121.0 244.2,120.5 245.4,119.9 246.7,119.4 247.9,118.9 249.2,118.4 250.5,117.9 251.7,117.4 253.0,116.9 254.2,116.4 255.5,115.9 256.7,115.4 258.0,114.9 259.2,114.4 260.5,114.0 261.7,113.5 263.0,113.0 264.2,112.5 265.5,112.1 266.7,111.6 268.0,111.1 269.2,110.7 270.5,110.2 271.8,109.8 273.0,109.3 274.3,108.9 275.5,108.4 276.8,108.0 278.0,107.6 279.3,107.1 280.5,106.7 281.8,106.2 283.0,105.8 284.3,105.4 285.5,105.0 286.8,104.5 288.0,104.1 289.3,103.7 290.6,103.3 291.8,102.9 293.1,102.5 294.3,102.1 295.6,101.7 296.8,101.3 298.1,100.9 299.3,100.5 300.6,100.1 301.8,99.7 303.1,99.3 304.3,98.9 305.6,98.5 306.8,98.1 308.1,97.7 309.3,97.3 310.6,97.0 311.9,96.6 313.1,96.2 314.4,95.8 315.6,95.5 316.9,95.1 318.1,94.7 319.4,94.4 320.6,94.0 321.9,93.6 323.1,93.3 324.4,92.9 325.6,92.6 326.9,92.2 328.1,91.8 329.4,91.5 330.7,91.1 331.9,90.8 333.2,90.5 334.4,90.1 335.7,89.8 336.9,89.4 338.2,89.1 339.4,88.7 340.7,88.4 341.9,88.1 343.2,87.7 344.4,87.4 345.7,87.1 346.9,86.7 348.2,86.4 349.4,86.1 350.7,85.8 352.0,85.4 353.2,85.1 354.5,84.8 355.7,84.5 357.0,84.2 358.2,83.8 359.5,83.5 360.7,83.2 362.0,82.9 363.2,82.6 364.5,82.3 365.7,82.0 367.0,81.7 368.2,81.3 369.5,81.0 370.8,80.7 372.0,80.4 373.3,80.1 374.5,79.8 375.8,79.5 377.0,79.2 378.3,78.9 379.5,78.6 380.8,78.4 382.0,78.1 383.3,77.8 384.5,77.5 385.8,77.2 387.0,76.9 388.3,76.6 389.5,76.3 390.8,76.0 392.1,75.8 393.3,75.5 394.6,75.2 395.8,74.9 397.1,74.6 398.3,74.4 399.6,74.1 400.8,73.8 402.1,73.5 403.3,73.3 404.6,73.0 405.8,72.7 407.1,72.4 408.3,72.2 409.6,71.9 410.9,71.6 412.1,71.4 413.4,71.1 414.6,70.8 415.9,70.6 417.1,70.3 418.4,70.1 419.6,69.8 420.9,69.5 422.1,69.3 423.4,69.0 424.6,68.8 425.9,68.5 427.1,68.2 428.4,68.0 429.6,67.7 430.9,67.5 432.2,67.2 433.4,67.0 434.7,66.7 435.9,66.5 437.2,66.2 438.4,66.0 439.7,65.7 440.9,65.5 442.2,65.2 443.4,65.0 444.7,64.8 445.9,64.5 447.2,64.3 448.4,64.0 449.7,63.8 451.0,63.5 452.2,63.3 453.5,63.1 454.7,62.8 456.0,62.6 457.2,62.4 458.5,62.1 459.7,61.9 461.0,61.7 462.2,61.4 463.5,61.2 464.7,61.0 466.0,60.7 467.2,60.5 468.5,60.3 469.7,60.0 471.0,59.8 472.3,59.6 473.5,59.4 474.8,59.1 476.0,58.9 477.3,58.7 478.5,58.5 479.8,58.2 481.0,58.0 482.3,57.8 483.5,57.6 484.8,57.4 486.0,57.1 487.3,56.9 488.5,56.7 489.8,56.5 491.1,56.3 492.3,56.1 493.6,55.8 494.8,55.6 496.1,55.4 497.3,55.2 498.6,55.0 499.8,54.8 501.1,54.6 502.3,54.3 503.6,54.1 504.8,53.9 506.1,53.7 507.3,53.5 508.6,53.3 509.8,53.1 511.1,52.9 512.4,52.7 513.6,52.5 514.9,52.3 516.1,52.1 517.4,51.9 518.6,51.7 519.9,51.5 521.1,51.3 522.4,51.1 523.6,50.8 524.9,50.6 526.1,50.4 527.4,50.2 528.6,50.1 529.9,49.9 531.2,49.7 532.4,49.5 533.7,49.3 534.9,49.1 536.2,48.9 537.4,48.7 538.7,48.5 539.9,48.3 541.2,48.1 542.4,47.9 543.7,47.7 544.9,47.5 546.2,47.3 547.4,47.1 548.7,46.9 549.9,46.7 551.2,46.6 552.5,46.4 553.7,46.2 555.0,46.0 556.2,45.8 557.5,45.6 558.7,45.4 560.0,45.2 561.2,45.1 562.5,44.9 563.7,44.7 565.0,44.5 566.2,44.3 567.5,44.1 568.7,44.0 570.0,43.8" class="stroke-b" fill="none"/>
<rect x="76.2" y="255.0" width="6.2" height="7.5" class="fill-a"/>
<circle cx="76.2" cy="262.5" r="4" class="fill-a"/>
<circle cx="82.5" cy="255.0" r="4" class="fill-a"/>
<rect x="320.0" y="92.4" width="6.2" height="1.8" class="fill-c"/>
<circle cx="320.0" cy="94.2" r="4" class="fill-c"/>
<circle cx="326.2" cy="92.4" r="4" class="fill-c"/>
<text x="126.2" y="250.0" class="fig-small" text-anchor="start">100 to 200 Hz: 1.50 to 3.00 mel, a step of 1.50</text>
<text x="338.8" y="130.0" class="fig-small" text-anchor="start">4000 to 4100 Hz: 35.16 to 35.52 mel,</text>
<text x="338.8" y="142.0" class="fig-small" text-anchor="start">a step of only 0.36</text>
<text x="251.2" y="195.0" class="fig-small" text-anchor="start">dashed: 3f / 200,</text>
<text x="251.2" y="206.0" class="fig-small" text-anchor="start">the linear rule without the bend</text>
<text x="88.8" y="45.0" class="fig-bold" text-anchor="start">Slaney mel scale</text>
</svg>
Figure 1. The mel scale rises about 4 times faster per hertz at 100 Hz than at 4000 Hz. The same 100 Hz step is 1.50 mel low down and 0.36 mel high up, which is the opening experiment in numbers. The dashed line shows the straight rule f / 66.7 that the curve follows up to 1000 Hz and then leaves.
```

## From a scale to a filterbank

A *mel filterbank* is a matrix with one row per band. Each row is a triangle of weights laid over the FFT bins, so multiplying it by a power spectrogram sums the power under each triangle and turns 201 bins into, say, 80 bands.

To place the triangles, pick $n_{\text{mels}} + 2$ edge frequencies equally spaced in mel from $f_{\min}$ to $f_{\max}$, then convert them back to hertz: $f_0 < f_1 < \dots < f_{n+1}$. Equal steps in mel grow in hertz, which is the widening you wanted. Filter $i$ starts at $f_i$, peaks at $f_{i+1}$, and ends at $f_{i+2}$, so neighbors overlap by half. Its weight at a bin of frequency $f$ is the following:

$$
W_i(f) = \max\left(0,\ \min\left(\frac{f - f_i}{f_{i+1} - f_i},\ \frac{f_{i+2} - f}{f_{i+2} - f_{i+1}}\right)\right)
$$

Read it aloud: the first fraction is a ramp that rises from 0 at $f_i$ to 1 at $f_{i+1}$, and the second is a ramp that falls from 1 at $f_{i+1}$ to 0 at $f_{i+2}$. The smaller of the two is the triangle, and the $\max$ with 0 cuts off the negative parts. The ramps are straight in hertz, not in mel: the edges are mel-spaced, but each triangle is drawn in hertz.

With broadcasting, the whole bank is a few lines:

```python
lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
rising = (bin_hz[None, :] - lower) / (center - lower)
falling = (upper - bin_hz[None, :]) / (upper - center)
weights = np.maximum(0.0, np.minimum(rising, falling))
```

### Build a two-filter bank by hand

Take a sample rate of 8000 Hz, `n_fft=8`, and two Slaney mel filters from 0 Hz to 4000 Hz. The bins sit every 8000 / 8 = 1000 Hz, at 0, 1000, 2000, 3000, and 4000 Hz.

1. The top edge is $m(4000) = 35.1638$ mel. Four edges equally spaced in mel are 0, 11.7213, 23.4425, and 35.1638.
2. Back to hertz: 11.7213 mel is below 15, so it is $11.7213 \times 200/3 = 781.42$ Hz. 23.4425 mel is above 15, so it is $1000\,e^{(23.4425 - 15) \times 0.068752} = 1786.82$ Hz. The edges are 0, 781.42, 1786.82, and 4000 Hz.
3. Filter 0 runs from 0 Hz through 781.42 Hz to 1786.82 Hz. Only the bin at 1000 Hz is inside, on the falling side: $(1786.82 - 1000) / 1005.40 = 0.7826$.
4. Filter 1 runs from 781.42 Hz through 1786.82 Hz to 4000 Hz. At 1000 Hz it is rising: $(1000 - 781.42) / 1005.40 = 0.2174$. At 2000 Hz and 3000 Hz it is falling: $2000 / 2213.18 = 0.9037$ and $1000 / 2213.18 = 0.4518$.

The bank is `[[0, 0.7826, 0, 0, 0], [0, 0.2174, 0.9037, 0.4518, 0]]`. Between the first and last centers, peak-1 triangles add up to one, as the bin at 1000 Hz shows: 0.7826 + 0.2174 = 1.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Two triangular mel filters over five FFT bins at 8 kHz, with the weight each bin receives">
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-grid"/>
<text x="64.0" y="244.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="149.1" x2="570.0" y2="149.1" class="stroke-grid"/>
<text x="64.0" y="153.1" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="58.2" x2="570.0" y2="58.2" class="stroke-grid"/>
<text x="64.0" y="62.2" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="240.0" x2="570.0" y2="240.0" class="stroke-muted"/>
<line x1="70.0" y1="40.0" x2="70.0" y2="240.0" class="stroke-muted"/>
<line x1="70.0" y1="240.0" x2="70.0" y2="245.0" class="stroke-muted"/>
<text x="70.0" y="259.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="240.0" x2="195.0" y2="245.0" class="stroke-muted"/>
<text x="195.0" y="259.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="320.0" y1="240.0" x2="320.0" y2="245.0" class="stroke-muted"/>
<text x="320.0" y="259.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="445.0" y1="240.0" x2="445.0" y2="245.0" class="stroke-muted"/>
<text x="445.0" y="259.0" class="fig-small" text-anchor="middle">3000</text>
<line x1="570.0" y1="240.0" x2="570.0" y2="245.0" class="stroke-muted"/>
<text x="570.0" y="259.0" class="fig-small" text-anchor="middle">4000</text>
<text x="320.0" y="278.0" text-anchor="middle">frequency (Hz)  —  the FFT bins sit at 0, 1000, 2000, 3000, and 4000 Hz</text>
<text x="8.0" y="30.0" text-anchor="start">weight</text>
<polygon points="70.0,240.0 167.7,58.2 293.4,240.0" class="fill-soft-a"/>
<polyline points="70.0,240.0 167.7,58.2 293.4,240.0" class="stroke-a" fill="none"/>
<polygon points="167.7,240.0 293.4,58.2 570.0,240.0" class="fill-soft-b"/>
<polyline points="167.7,240.0 293.4,58.2 570.0,240.0" class="stroke-b" fill="none"/>
<line x1="195.0" y1="240.0" x2="195.0" y2="97.7" class="stroke-a"/>
<circle cx="195.0" cy="97.7" r="5" class="fill-a"/>
<line x1="195.0" y1="240.0" x2="195.0" y2="200.5" class="stroke-b"/>
<circle cx="195.0" cy="200.5" r="5" class="fill-b"/>
<line x1="320.0" y1="240.0" x2="320.0" y2="75.7" class="stroke-b"/>
<circle cx="320.0" cy="75.7" r="5" class="fill-b"/>
<line x1="445.0" y1="240.0" x2="445.0" y2="157.8" class="stroke-b"/>
<circle cx="445.0" cy="157.8" r="5" class="fill-b"/>
<text x="203.8" y="83.2" class="fig-small" text-anchor="start">0.7826</text>
<text x="203.8" y="209.6" class="fig-small" text-anchor="start">0.2174</text>
<text x="328.8" y="66.6" class="fig-small" text-anchor="start">0.9037</text>
<text x="453.8" y="148.8" class="fig-small" text-anchor="start">0.4518</text>
<text x="167.7" y="47.3" class="fig-small" text-anchor="middle">781.42 Hz</text>
<text x="300.9" y="47.3" class="fig-small" text-anchor="start">1786.82 Hz</text>
<text x="107.5" y="218.2" class="fig-bold" text-anchor="start">filter 0</text>
<text x="432.5" y="203.6" class="fig-bold" text-anchor="start">filter 1</text>
</svg>
Figure 2. Edges at 0, 781.42, 1786.82, and 4000 Hz give two triangles. The dots are the weights the five FFT bins receive. Notice that the bin at 1000 Hz belongs to both filters and that 0.7826 + 0.2174 = 1, so no power is dropped or counted twice between the first and last centers.
```

The following demo draws the filters over the FFT bins.

```demo
{"id": "mel-filterbank", "caption": "Raise the number of mel bands and switch the formula to HTK, and watch the lowest triangles narrow until the fewest bins in a band reads 1 and empty bands appear."}
```

### Normalize the triangles

Two normalizations are common. With `norm=None`, every peak is 1, as in the hand example, and `torchaudio` does this by default. With `norm="slaney"`, filter $i$ is multiplied by $2 / (f_{i+2} - f_i)$, one over half its width, so every triangle has an area of 1 in hertz. Without it, a wide high filter collects more power only because it covers more bins. For the hand example the factors are $2/1786.82 = 0.0011193$ for filter 0 and $2/3218.58 = 0.00062139$ for filter 1. librosa and Whisper use the Slaney normalization.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="Ten of Whisper's 80 mel filters drawn as triangles between 0 and 8000 hertz, narrow on the left and wide on the right">
<line x1="70.0" y1="260.0" x2="580.0" y2="260.0" class="stroke-grid"/>
<text x="64.0" y="264.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="160.0" x2="580.0" y2="160.0" class="stroke-grid"/>
<text x="64.0" y="164.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="60.0" x2="580.0" y2="60.0" class="stroke-grid"/>
<text x="64.0" y="64.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="260.0" x2="580.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="260.0" class="stroke-muted"/>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted"/>
<text x="70.0" y="279.0" class="fig-small" text-anchor="middle">0</text>
<line x1="197.5" y1="260.0" x2="197.5" y2="265.0" class="stroke-muted"/>
<text x="197.5" y="279.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="325.0" y1="260.0" x2="325.0" y2="265.0" class="stroke-muted"/>
<text x="325.0" y="279.0" class="fig-small" text-anchor="middle">4000</text>
<line x1="452.5" y1="260.0" x2="452.5" y2="265.0" class="stroke-muted"/>
<text x="452.5" y="279.0" class="fig-small" text-anchor="middle">6000</text>
<line x1="580.0" y1="260.0" x2="580.0" y2="265.0" class="stroke-muted"/>
<text x="580.0" y="279.0" class="fig-small" text-anchor="middle">8000</text>
<text x="325.0" y="298.0" text-anchor="middle">frequency (Hz)</text>
<text x="8.0" y="20.0" text-anchor="start">weight (peak 1)</text>
<polygon points="70.0,260.0 72.4,60.0 74.7,260.0" class="fill-soft-a"/>
<polyline points="70.0,260.0 72.4,60.0 74.7,260.0" class="stroke-a" fill="none"/>
<polygon points="89.0,260.0 91.4,60.0 93.7,260.0" class="fill-soft-b"/>
<polyline points="89.0,260.0 91.4,60.0 93.7,260.0" class="stroke-b" fill="none"/>
<polygon points="108.0,260.0 110.4,60.0 112.7,260.0" class="fill-soft-a"/>
<polyline points="108.0,260.0 110.4,60.0 112.7,260.0" class="stroke-a" fill="none"/>
<polygon points="127.0,260.0 129.3,60.0 131.7,260.0" class="fill-soft-b"/>
<polyline points="127.0,260.0 129.3,60.0 131.7,260.0" class="stroke-b" fill="none"/>
<polygon points="147.7,260.0 150.7,60.0 153.9,260.0" class="fill-soft-a"/>
<polyline points="147.7,260.0 150.7,60.0 153.9,260.0" class="stroke-a" fill="none"/>
<polygon points="175.6,260.0 179.8,60.0 184.1,260.0" class="fill-soft-b"/>
<polyline points="175.6,260.0 179.8,60.0 184.1,260.0" class="stroke-b" fill="none"/>
<polygon points="213.6,260.0 219.2,60.0 225.1,260.0" class="fill-soft-a"/>
<polyline points="213.6,260.0 219.2,60.0 225.1,260.0" class="stroke-a" fill="none"/>
<polygon points="265.3,260.0 272.9,60.0 280.8,260.0" class="fill-soft-b"/>
<polyline points="265.3,260.0 272.9,60.0 280.8,260.0" class="stroke-b" fill="none"/>
<polygon points="335.5,260.0 345.9,60.0 356.7,260.0" class="fill-soft-a"/>
<polyline points="335.5,260.0 345.9,60.0 356.7,260.0" class="stroke-a" fill="none"/>
<polygon points="431.0,260.0 445.1,60.0 459.8,260.0" class="fill-soft-b"/>
<polyline points="431.0,260.0 445.1,60.0 459.8,260.0" class="stroke-b" fill="none"/>
<text x="74.3" y="42.0" class="fig-small" text-anchor="start">filter 0: 74 Hz wide</text>
<text x="448.9" y="136.0" class="fig-small" text-anchor="start">filter 72 is</text>
<text x="448.9" y="160.0" class="fig-small" text-anchor="start">452 Hz wide</text>
</svg>
Figure 3. Every eighth filter of Whisper's 80-filter bank at 16 kHz. The edges are equally spaced in mel, so the triangles are 74 Hz wide at the bottom and 452 Hz wide near 6 kHz. The lowest filter is narrower than the 40 Hz spacing of the FFT bins.
```

At Whisper's settings, 16 kHz and `n_fft=400`, the bins are 40 Hz apart. The 82 edges for 80 mels start 37.2 Hz apart, so filter 0 is 74.5 Hz wide and touches only one bin, while the top filter spans 14 bins. The lowest filters sample the spectrum, and the highest average it.

> [!KEY]
> A mel filterbank places triangles whose edges are equally spaced on the mel scale, so the triangles are narrow where the ear is sharp and wide where it is coarse. Multiplying a power spectrogram by it keeps the bands the ear can tell apart.

> [!RECAP]
> - The ear's bands are about 100 Hz wide below 500 Hz and widen in proportion to frequency above it.
> - HTK mel is $2595 \log_{10}(1 + f/700)$. Slaney mel is $3f/200$ below 1 kHz and logarithmic above.
> - A filterbank has $n_{\text{mels}} + 2$ mel-spaced edges, and each triangle is straight in hertz.
> - Slaney normalization gives each triangle unit area.

## Build it step by step

The checks test the functions in this order.

1. **`hz_to_mel(f, htk=False)`** takes a scalar or array in hertz and returns mel as a float array. Branch on `htk`. For Slaney, compute both branches and choose with `np.where`. The usual mistake is writing `f / 200 * 3` for the whole range and forgetting the log branch above 1 kHz.
2. **`mel_to_hz(m, htk=False)`** inverts it, and `mel_to_hz(hz_to_mel(f))` must return `f` to 1e-9. The usual mistake is switching the Slaney branches at 1000 mel instead of 15.
3. **`mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, htk, norm)`** returns `(n_mels, n_fft // 2 + 1)`. Build the edges with `np.linspace` in mel, convert to hertz, and build every triangle by broadcasting. The usual mistake is drawing the ramps in mel instead of hertz, which gives curved triangles that miss librosa's peak-1 filters by up to 5e-3.

## Where this breaks

**Too many filters for the FFT.** At `n_fft=400`, 128 HTK filters leave filters 0, 3, 6, and 13 with no bins at all, so those bands are always zero. The Slaney scale avoids this by spacing the lowest filters further apart. Use a larger FFT when you need many narrow low bands.

> [!WARNING]
> A model trained with HTK mel and fed Slaney mel sees every band shifted. Check which scale and which normalization a model expects before you reuse its front end. Kaldi's $1127 \ln(1 + f/700)$ is the HTK curve in natural-log form, not a third scale.

## The exercise

You build `hz_to_mel`, `mel_to_hz`, and `mel_filterbank`. Press **Run** to hear the same 100 Hz step low and high, print its size in mel on both scales, and see Whisper's 80 filters as an image.

The checks verify the mel table and the exact inverse. They reproduce the two-filter hand example, compare your filterbank with numbers from `librosa.filters.mel` and with an independent construction for both scales and both normalizations, and check that peak-1 triangles add up to one.
