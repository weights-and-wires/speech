---
title: Pitch tracking
slug: 018-pitch-tracking
part: "Part 2 — Hearing and speech production"
summary: Find the period of a voice with YIN, track it frame by frame, decide which frames are voiced, and score the track by gross pitch error.
minutes: 45
gpu: false
objectives:
  - Explain why the largest autocorrelation peak often lands an octave too low.
  - Build YIN's difference function, cumulative mean normalization, absolute threshold, and parabolic interpolation.
  - Track F0 frame by frame, using YIN's aperiodicity as the voicing decision.
  - Score a pitch track against a known F0 by gross pitch error and voicing decision error.
prereqs:
  - 007-the-short-time-fourier-transform
  - 016-the-source-filter-model
  - 017-linear-prediction
exercise: 018-pitch-tracking
---

# Pitch tracking

The same words mean different things with a different melody. "You're leaving" falls at the end as a statement and rises as a question. In tonal languages such as Mandarin, the contour changes which word you said. A text-to-speech system must predict that contour, and voice conversion must move it.

The contour is the *fundamental frequency*, $F_0$, of the voice over time: the rate of the vocal folds, which you met as the source in the source-filter model. Estimating it sounds easy, because a voiced frame repeats itself once per period. It is not. A waveform that repeats every period also repeats every two periods, and the obvious method often picks the wrong one.

Figure 1 shows the problem on one frame. This chapter builds YIN, which fixes the obvious method in small steps, then tracks a voice frame by frame, decides which frames have a pitch at all, and scores the result.

```figure
<svg viewBox="0 0 600 380" role="img" aria-label="Autocorrelation of one voiced frame peaks highest at twice the period, while the normalized difference function of YIN dips first at the true period">
<line x1="60" y1="142.9" x2="570" y2="142.9" class="stroke-grid"/>
<text x="54" y="146.9" class="fig-small" text-anchor="end">-0.5</text>
<line x1="60" y1="107.6" x2="570" y2="107.6" class="stroke-grid"/>
<text x="54" y="111.6" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="72.4" x2="570" y2="72.4" class="stroke-grid"/>
<text x="54" y="76.4" class="fig-small" text-anchor="end">0.5</text>
<line x1="60" y1="37.1" x2="570" y2="37.1" class="stroke-grid"/>
<text x="54" y="41.1" class="fig-small" text-anchor="end">1</text>
<line x1="60" y1="150" x2="570" y2="150" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="150" class="stroke-muted"/>
<text x="60" y="22" class="fig-small" text-anchor="start">r(tau) / r(0)  (autocorrelation)</text>
<polyline points="121.1,154.8 123.0,153.4 124.9,151.8 126.9,149.9 128.8,147.8 130.7,145.3 132.6,142.4 134.5,139.1 136.4,135.4 138.3,131.2 140.2,126.6 142.1,121.7 144.0,116.4 146.0,110.6 147.9,104.7 149.8,98.4 151.7,92.0 153.6,85.4 155.5,78.9 157.4,72.5 159.3,66.4 161.2,60.8 163.1,55.8 165.1,51.3 167.0,47.6 168.9,44.7 170.8,42.8 172.7,41.9 174.6,42.1 176.5,43.5 178.4,45.9 180.3,49.3 182.2,53.5 184.2,58.4 186.1,63.9 188.0,70.0 189.9,76.4 191.8,83.2 193.7,90.1 195.6,97.1 197.5,103.8 199.4,110.3 201.3,116.5 203.3,122.2 205.2,127.6 207.1,132.6 209.0,137.0 210.9,140.9 212.8,144.5 214.7,147.4 216.6,149.7 218.5,151.6 220.4,153.2 222.4,154.4 224.3,155.2 226.2,155.8 228.1,156.1 230.0,156.1 231.9,155.7 233.8,155.2 235.7,154.4 237.6,153.3 239.6,151.9 241.5,150.2 243.4,148.1 245.3,145.7 247.2,142.7 249.1,139.3 251.0,135.4 252.9,131.0 254.8,126.1 256.7,120.8 258.7,115.2 260.6,109.1 262.5,102.7 264.4,96.0 266.3,89.2 268.2,82.3 270.1,75.6 272.0,69.1 273.9,63.0 275.8,57.4 277.8,52.4 279.7,48.0 281.6,44.7 283.5,42.1 285.4,40.6 287.3,40.3 289.2,41.2 291.1,43.1 293.0,46.1 294.9,50.0 296.9,54.8 298.8,60.1 300.7,66.1 302.6,72.7 304.5,79.5 306.4,86.6 308.3,93.8 310.2,100.8 312.1,107.6 314.0,114.2 316.0,120.4 317.9,126.0 319.8,131.3 321.7,136.1 323.6,140.3 325.5,144.0 327.4,147.2 329.3,149.8 331.2,151.9 333.1,153.5 335.1,154.8 337.0,155.6 338.9,156.2 340.8,156.4 342.7,156.4 344.6,156.2 346.5,155.5 348.4,154.7 350.3,153.6 352.2,152.1 354.2,150.5 356.1,148.5 358.0,146.1 359.9,143.3 361.8,140.1 363.7,136.4 365.6,132.2 367.5,127.6 369.4,122.5 371.3,117.1 373.3,111.3 375.2,105.2 377.1,98.8 379.0,92.1 380.9,85.3 382.8,78.5 384.7,72.0 386.6,65.8 388.5,60.0 390.4,54.9 392.4,50.3 394.3,46.5 396.2,43.6 398.1,41.6 400.0,40.8 401.9,41.1 403.8,42.5 405.7,45.1 407.6,48.6 409.6,52.9 411.5,57.9 413.4,63.6 415.3,69.8 417.2,76.5 419.1,83.4 421.0,90.5 422.9,97.6 424.8,104.5 426.7,111.0 428.7,117.3 430.6,123.3 432.5,128.7 434.4,133.7 436.3,138.2 438.2,142.1 440.1,145.5 442.0,148.4 443.9,150.7 445.8,152.6 447.8,154.0 449.7,155.1 451.6,155.8 453.5,156.3 455.4,156.4 457.3,156.2 459.2,155.9 461.1,155.2 463.0,154.2 464.9,153.0 466.9,151.4 468.8,149.6 470.7,147.4 472.6,144.8 474.5,141.8 476.4,138.3 478.3,134.3 480.2,129.9 482.1,125.0 484.0,119.7 486.0,114.0 487.9,107.9 489.8,101.6 491.7,95.0 493.6,88.2 495.5,81.4 497.4,74.7 499.3,68.4 501.2,62.4 503.1,56.9 505.1,52.1 507.0,47.9 508.9,44.6 510.8,42.3 512.7,40.9 514.6,40.7 516.5,41.8 518.4,43.8 520.3,46.9 522.2,50.9 524.2,55.7 526.1,61.1 528.0,67.1 529.9,73.6 531.8,80.4 533.7,87.6 535.6,94.7 537.5,101.7 539.4,108.5 541.3,114.9 543.3,120.9 545.2,126.5 547.1,131.8 549.0,136.4 550.9,140.6 552.8,144.2 554.7,147.4 556.6,149.8 558.5,151.9 560.4,153.5 562.4,154.7 564.3,155.6 566.2,156.2 568.1,156.4 570.0,156.3" class="stroke-a" fill="none"/>
<line x1="173.5" y1="30" x2="173.5" y2="150" class="stroke-grid dashed"/>
<line x1="287.1" y1="30" x2="287.1" y2="150" class="stroke-grid dashed"/>
<line x1="400.6" y1="30" x2="400.6" y2="150" class="stroke-grid dashed"/>
<line x1="514.2" y1="30" x2="514.2" y2="150" class="stroke-grid dashed"/>
<circle cx="287.3" cy="40.3" r="5" class="fill-bad"/>
<text x="298.8" y="38.9" class="fig-small" text-anchor="start">largest peak, tau = 119 (134 Hz): an octave too low</text>
<line x1="60" y1="325.0" x2="570" y2="325.0" class="stroke-grid"/>
<text x="54" y="329.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="278.8" x2="570" y2="278.8" class="stroke-grid"/>
<text x="54" y="282.8" class="fig-small" text-anchor="end">1</text>
<line x1="60" y1="232.7" x2="570" y2="232.7" class="stroke-grid"/>
<text x="54" y="236.7" class="fig-small" text-anchor="end">2</text>
<line x1="60" y1="325" x2="570" y2="325" class="stroke-muted"/>
<line x1="60" y1="205" x2="60" y2="325" class="stroke-muted"/>
<line x1="60.0" y1="325" x2="60.0" y2="329" class="stroke-muted"/>
<text x="60.0" y="342" class="fig-small" text-anchor="middle">0</text>
<line x1="155.5" y1="325" x2="155.5" y2="329" class="stroke-muted"/>
<text x="155.5" y="342" class="fig-small" text-anchor="middle">50</text>
<line x1="251.0" y1="325" x2="251.0" y2="329" class="stroke-muted"/>
<text x="251.0" y="342" class="fig-small" text-anchor="middle">100</text>
<line x1="346.5" y1="325" x2="346.5" y2="329" class="stroke-muted"/>
<text x="346.5" y="342" class="fig-small" text-anchor="middle">150</text>
<line x1="442.0" y1="325" x2="442.0" y2="329" class="stroke-muted"/>
<text x="442.0" y="342" class="fig-small" text-anchor="middle">200</text>
<line x1="537.5" y1="325" x2="537.5" y2="329" class="stroke-muted"/>
<text x="537.5" y="342" class="fig-small" text-anchor="middle">250</text>
<text x="315.0" y="359" text-anchor="middle">lag tau (samples; the true period is 59.4 samples)</text>
<text x="60" y="197" class="fig-small" text-anchor="start">d'(tau)  (YIN)</text>
<polyline points="61.9,278.8 63.8,259.9 65.7,245.2 67.6,235.0 69.6,228.2 71.5,223.7 73.4,220.7 75.3,218.6 77.2,217.2 79.1,216.7 81.0,216.5 82.9,217.1 84.8,218.0 86.7,219.2 88.7,220.7 90.6,222.3 92.5,224.1 94.4,225.9 96.3,227.8 98.2,229.8 100.1,231.9 102.0,234.1 103.9,236.4 105.8,238.6 107.8,240.7 109.7,242.8 111.6,245.0 113.5,247.1 115.4,249.2 117.3,251.1 119.2,253.1 121.1,255.1 123.0,257.0 124.9,259.0 126.9,261.0 128.8,263.0 130.7,265.0 132.6,267.1 134.5,269.3 136.4,271.6 138.3,274.1 140.2,276.6 142.1,279.3 144.0,282.1 146.0,285.1 147.9,288.2 149.8,291.5 151.7,294.9 153.6,298.4 155.5,301.9 157.4,305.3 159.3,308.7 161.2,311.8 163.1,314.8 165.1,317.4 167.0,319.7 168.9,321.5 170.8,322.8 172.7,323.4 174.6,323.3 176.5,322.3 178.4,320.6 180.3,318.2 182.2,315.1 184.2,311.5 186.1,307.3 188.0,302.7 189.9,297.8 191.8,292.6 193.7,287.4 195.6,282.2 197.5,277.2 199.4,272.4 201.3,268.0 203.3,263.9 205.2,260.3 207.1,257.1 209.0,254.3 210.9,252.1 212.8,250.2 214.7,248.8 216.6,247.8 218.5,247.1 220.4,246.7 222.4,246.6 224.3,246.7 226.2,247.0 228.1,247.4 230.0,248.1 231.9,248.9 233.8,249.9 235.7,250.9 237.6,252.1 239.6,253.5 241.5,255.0 243.4,256.7 245.3,258.6 247.2,260.8 249.1,263.1 251.0,265.7 252.9,268.5 254.8,271.6 256.7,274.8 258.7,278.2 260.6,281.9 262.5,285.8 264.4,289.7 266.3,293.8 268.2,298.0 270.1,302.0 272.0,306.0 273.9,309.7 275.8,313.1 277.8,316.2 279.7,319.0 281.6,321.2 283.5,322.8 285.4,323.8 287.3,324.1 289.2,323.5 291.1,322.1 293.0,320.1 294.9,317.4 296.9,314.0 298.8,310.2 300.7,305.8 302.6,301.1 304.5,296.2 306.4,291.1 308.3,285.9 310.2,280.9 312.1,276.0 314.0,271.4 316.0,267.1 317.9,263.3 319.8,259.7 321.7,256.6 323.6,253.9 325.5,251.6 327.4,249.8 329.3,248.3 331.2,247.3 333.1,246.6 335.1,246.2 337.0,246.0 338.9,246.1 340.8,246.4 342.7,246.8 344.6,247.4 346.5,248.3 348.4,249.2 350.3,250.3 352.2,251.6 354.2,253.0 356.1,254.7 358.0,256.5 359.9,258.5 361.8,260.7 363.7,263.2 365.6,266.0 367.5,269.0 369.4,272.2 371.3,275.6 373.3,279.2 375.2,283.0 377.1,287.0 379.0,291.2 380.9,295.4 382.8,299.6 384.7,303.8 386.6,307.7 388.5,311.3 390.4,314.6 392.4,317.6 394.3,320.1 396.2,322.0 398.1,323.3 400.0,323.9 401.9,323.7 403.8,322.8 405.7,321.0 407.6,318.6 409.6,315.6 411.5,312.1 413.4,308.1 415.3,303.7 417.2,298.9 419.1,294.1 421.0,289.0 422.9,284.0 424.8,279.2 426.7,274.6 428.7,270.3 430.6,266.2 432.5,262.5 434.4,259.2 436.3,256.2 438.2,253.7 440.1,251.6 442.0,249.9 443.9,248.5 445.8,247.6 447.8,246.8 449.7,246.4 451.6,246.3 453.5,246.3 455.4,246.5 457.3,246.9 459.2,247.5 461.1,248.2 463.0,249.2 464.9,250.3 466.9,251.6 468.8,253.1 470.7,254.8 472.6,256.6 474.5,258.7 476.4,261.1 478.3,263.8 480.2,266.7 482.1,269.9 484.0,273.3 486.0,276.9 487.9,280.7 489.8,284.7 491.7,288.9 493.6,293.2 495.5,297.5 497.4,301.8 499.3,305.9 501.2,309.7 503.1,313.3 505.1,316.4 507.0,319.1 508.9,321.3 510.8,322.9 512.7,323.8 514.6,324.0 516.5,323.3 518.4,321.9 520.3,319.8 522.2,317.0 524.2,313.7 526.1,309.9 528.0,305.7 529.9,301.1 531.8,296.4 533.7,291.4 535.6,286.4 537.5,281.5 539.4,276.8 541.3,272.3 543.3,268.2 545.2,264.4 547.1,260.9 549.0,257.7 550.9,255.0 552.8,252.7 554.7,250.7 556.6,249.3 558.5,248.1 560.4,247.2 562.4,246.6 564.3,246.3 566.2,246.1 568.1,246.3 570.0,246.6" class="stroke-b" fill="none"/>
<line x1="173.5" y1="205" x2="173.5" y2="325" class="stroke-grid dashed"/>
<line x1="287.1" y1="205" x2="287.1" y2="325" class="stroke-grid dashed"/>
<line x1="400.6" y1="205" x2="400.6" y2="325" class="stroke-grid dashed"/>
<line x1="514.2" y1="205" x2="514.2" y2="325" class="stroke-grid dashed"/>
<line x1="60" y1="320.4" x2="570" y2="320.4" class="stroke-muted dashed"/>
<text x="451.6" y="315.8" class="fig-small" text-anchor="start">threshold 0.1</text>
<circle cx="172.7" cy="323.4" r="5" class="fill-b"/>
<text x="188.0" y="310.4" class="fig-small" text-anchor="start">first dip under 0.1, tau = 59 (271 Hz)</text>
</svg>
Figure 1. One frame of a synthetic voice at 269 Hz, a period of 59.4 samples (dashed lines mark one to four periods). Notice that the autocorrelation is nearly as tall at every multiple of the period and its largest value lands at twice the period, while d'(tau) dips under 0.1 at the true period first.
```

## What you already know

- A voiced frame is a pulse train filtered by the vocal tract, so it repeats every $T_0 = 1/F_0$ seconds.
- The autocorrelation $r(\tau)$ multiplies a frame by a copy of itself shifted by $\tau$ samples, and adds.
- The cepstrum shows the pitch period as a peak at quefrency $f_s / F_0$.
- A centered STFT pads by half a frame and puts frame $t$ at sample $t \cdot H$ for hop $H$.
- Frames and lags are both counted in samples, so you convert with $f_s$ to get seconds or hertz.

## Two questions, not one

$F_0$ is a physical quantity, the repetition rate. *Pitch* is a perception, how high a sound seems. They usually agree, but harmonics at 400, 600, and 800 Hz with nothing at 200 Hz are heard as a pitch of 200 Hz. This *missing fundamental* is common: a telephone line cuts everything below 300 Hz, and a man's voice still sounds low. So a tracker should find the *period*, 1/200 s, whether or not the fundamental is present.

A tracker also decides whether a frame has a pitch at all. Vowels and "m" do. Silence and "s" do not. This *voicing decision* matters as much as the frequency, because a made-up pitch in silence ruins a contour.

## The obvious method and where it fails

Take the autocorrelation of a frame $x$ and pick the lag with the largest value. For a frame of $W + \tau_{\max}$ samples:

$$
r(\tau) = \sum_{j=0}^{W-1} x_j\, x_{j+\tau}
$$

Read it aloud: multiply the first $W$ samples by the $W$ samples that start $\tau$ later, and add. When $\tau$ is the period, the two stretches line up and the sum is large, so $F_0 = f_s / \tau$.

The trouble is that $2T_0$, $3T_0$, and so on line up equally well. In Figure 1 the autocorrelation at the true period, lag 59, is 0.931 of its value at lag 0, and at twice the period, lag 119, it is 0.954. Small changes in amplitude or noise decide which peak wins, and here the wrong one wins: 134 Hz for a 269 Hz voice. On 60 synthetic voices with random $F_0$ from 80 to 400 Hz, random formants, and 20 dB of noise, the largest autocorrelation peak is off by more than 20% on 52% of them.

> [!KEY]
> A signal that repeats every period also repeats every two periods. The largest autocorrelation peak often sits at a multiple of the period, and the estimate lands an octave or more too low.

## YIN: ask how different, not how similar

YIN, from de Cheveigné and Kawahara, changes the autocorrelation one step at a time. Their Table I measures the gross error rate on real speech: 10.0% for plain autocorrelation, 1.95% after step 1 below, 1.69% after step 2, 0.78% after step 3, and 0.77% after step 4. The sections below build those four steps.

### Step 1: the difference function

Instead of "how similar is the signal to its shifted self", ask "how different":

$$
d(\tau) = \sum_{j=0}^{W-1} \left(x_j - x_{j+\tau}\right)^2 = r_0(0) + r_\tau(0) - 2r(\tau)
$$

Read it aloud: subtract the shifted stretch from the first, square, and add. A periodic signal gives $d(T_0) = 0$. Expanding the square shows two energy terms, the energy of the first stretch and of the shifted one, minus twice the autocorrelation. When a voice grows louder across the frame, the shifted stretch has more energy and the autocorrelation favors a long lag. The difference function subtracts that energy out.

```python
shifted = sliding_window_view(frame, W)[: max_lag + 1]  # row tau is frame[tau : tau + W]
d = np.sum((shifted - frame[:W]) ** 2, axis=1)
```

### Step 2: normalize by the running mean

Even so, $d(0) = 0$ and $d$ is small at short lags for any smooth signal, so a global minimum is useless. YIN divides each value by the average of all the values before it:

$$
d'(\tau) = \frac{d(\tau)}{\frac{1}{\tau}\sum_{j=1}^{\tau} d(j)} \quad (\tau > 0), \qquad d'(0) = 1
$$

Read it aloud: the value at lag $\tau$ over the running mean of the differences from lag 1 up to $\tau$. A value near 1 means "no better than the lags before it", and a value near 0 means "this lag lines up much better". The result does not depend on how loud the signal is, so one threshold works for every frame.

```python
dprime = np.ones_like(d)
dprime[1:] = d[1:] * np.arange(1, len(d)) / np.cumsum(d[1:])
```

### Step 3: take the first dip under a threshold

The deepest dip is often at $2T_0$, which is the autocorrelation's mistake again. So scan upward from $\tau_{\min}$, stop at the first lag with $d'(\tau) < 0.1$, and slide forward while the next value is smaller, to reach the bottom of that dip. If no lag goes under 0.1, take the global minimum. The paper notes that the exact threshold matters little.

### Step 4: refine between lags

The true period is rarely a whole number of samples. At 16 kHz a 120 Hz voice has a period of 133.33 samples, and lag 133 gives 120.30 Hz. Fit a parabola through the dip and its two neighbors, and take its vertex:

$$
\tau^* = \tau + \frac{d'(\tau - 1) - d'(\tau + 1)}{2\left(d'(\tau - 1) - 2d'(\tau) + d'(\tau + 1)\right)}
$$

Read it aloud: the correction is the difference between the neighbors over twice the curvature. If the neighbors are equal, the dip is centered and the correction is 0. The value $d'(\tau)$ at the dip is the frame's *aperiodicity*: near 0 for a clean voice, near 1 for noise.

### Work it by hand

Take the signal $0, 1, 0, -1$ repeated, with period 4, $W = 4$, and lags 0 to 5. The frame is $[0, 1, 0, -1, 0, 1, 0, -1, 0]$.

1. Differences: $d(1) = (0-1)^2 + (1-0)^2 + (0+1)^2 + (-1-0)^2 = 4$, and the same way $d = [0, 4, 8, 4, 0, 4]$ for $\tau = 0, \dots, 5$.
2. Running sums from lag 1: 4, 12, 16, 16, 20. So $d'(1) = 4 \cdot 1/4 = 1$, $d'(2) = 8 \cdot 2/12 = 1.333$, $d'(3) = 4 \cdot 3/16 = 0.75$, $d'(4) = 0$, and $d'(5) = 4 \cdot 5/20 = 1$. With $d'(0) = 1$, that is $[1, 1, 1.333, 0.75, 0, 1]$, as in Figure 2.
3. The first value under 0.1 is at $\tau = 4$, so $F_0 = f_s / 4$.
4. Interpolation gives $\tau^* = 4 + (0.75 - 1) / (2(0.75 - 0 + 1)) = 3.929$. The period here is exactly 4, so the nudge is a small error. It pays off when the period falls between two lags, as at 133.33.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Bar chart of the normalized difference d prime for the hand example, which is below the threshold only at lag 4">
<line x1="60" y1="220.0" x2="560" y2="220.0" class="stroke-grid"/>
<text x="54" y="224.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="160.6" x2="560" y2="160.6" class="stroke-grid"/>
<text x="54" y="164.6" class="fig-small" text-anchor="end">0.5</text>
<line x1="60" y1="101.2" x2="560" y2="101.2" class="stroke-grid"/>
<text x="54" y="105.2" class="fig-small" text-anchor="end">1</text>
<line x1="60" y1="41.9" x2="560" y2="41.9" class="stroke-grid"/>
<text x="54" y="45.9" class="fig-small" text-anchor="end">1.5</text>
<line x1="60" y1="220" x2="560" y2="220" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="220" class="stroke-muted"/>
<line x1="101.7" y1="220" x2="101.7" y2="224" class="stroke-muted"/>
<text x="101.7" y="237" class="fig-small" text-anchor="middle">0</text>
<line x1="185.0" y1="220" x2="185.0" y2="224" class="stroke-muted"/>
<text x="185.0" y="237" class="fig-small" text-anchor="middle">1</text>
<line x1="268.3" y1="220" x2="268.3" y2="224" class="stroke-muted"/>
<text x="268.3" y="237" class="fig-small" text-anchor="middle">2</text>
<line x1="351.7" y1="220" x2="351.7" y2="224" class="stroke-muted"/>
<text x="351.7" y="237" class="fig-small" text-anchor="middle">3</text>
<line x1="435.0" y1="220" x2="435.0" y2="224" class="stroke-muted"/>
<text x="435.0" y="237" class="fig-small" text-anchor="middle">4</text>
<line x1="518.3" y1="220" x2="518.3" y2="224" class="stroke-muted"/>
<text x="518.3" y="237" class="fig-small" text-anchor="middle">5</text>
<text x="310.0" y="254" text-anchor="middle">lag tau (samples)</text>
<text x="60" y="22" class="fig-small" text-anchor="start">d'(tau)</text>
<rect x="83.7" y="101.2" width="36" height="118.8" class="fill-soft-a"/>
<text x="101.7" y="94.1" class="fig-small" text-anchor="middle">1</text>
<rect x="167.0" y="101.2" width="36" height="118.8" class="fill-soft-a"/>
<text x="185.0" y="94.1" class="fig-small" text-anchor="middle">1</text>
<rect x="250.3" y="61.7" width="36" height="158.3" class="fill-soft-a"/>
<text x="268.3" y="54.5" class="fig-small" text-anchor="middle">1.333</text>
<rect x="333.7" y="130.9" width="36" height="89.1" class="fill-soft-a"/>
<text x="351.7" y="123.8" class="fig-small" text-anchor="middle">0.750</text>
<rect x="417.0" y="220.0" width="36" height="0.0" class="fill-b"/>
<text x="435.0" y="196.2" class="fig-small" text-anchor="middle">0</text>
<rect x="500.3" y="101.2" width="36" height="118.8" class="fill-soft-a"/>
<text x="518.3" y="94.1" class="fig-small" text-anchor="middle">1</text>
<line x1="60" y1="208.1" x2="560" y2="208.1" class="stroke-ink dashed"/>
<text x="68.3" y="201.0" class="fig-small" text-anchor="start">threshold 0.1</text>
<text x="435.0" y="170.1" class="fig-small" text-anchor="middle">first value under 0.1</text>
<circle cx="435.0" cy="220.0" r="5" class="fill-b"/>
</svg>
Figure 2. The hand example, the signal 0, 1, 0, -1 repeated, which has a period of 4 samples. Notice that d'(tau) is 1 or more at lags 1, 2, and 5, dips to 0.75 at lag 3, and falls to 0 at lag 4, the only lag under the threshold.
```

> [!KEY]
> YIN looks for the first lag where the signal differs little from its shifted self, relative to the lags before it. Normalizing by the running mean and taking the first dip under a threshold removes most octave errors.

## Track frames and decide voicing

A pitch track runs YIN on every frame, using the STFT convention: pad the signal by half a frame at each end by reflection, and center frame $t$ on sample $t \cdot H$. The frame must hold $W + \tau_{\max} + 1$ samples, and $W$ should cover at least one period at $f_{\min}$. At 16 kHz with $f_{\min} = 60$ Hz, $\tau_{\max} = 267$, so a 1024-sample frame, 64 ms, leaves $W = 756$.

The voicing decision uses the aperiodicity: a frame with $d'(\tau) < 0.25$ is voiced and keeps YIN's $F_0$, and any other frame stores 0. Noise gives $d'$ close to 1 at every lag, and silence gives exactly 1 because the running sum is 0. Figure 3 shows the result on a synthetic voice that asks a question.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Pitch track of a synthetic question: the YIN estimate follows the true falling and rising contour and drops to zero in the two unvoiced gaps">
<line x1="60" y1="250.0" x2="570" y2="250.0" class="stroke-grid"/>
<text x="54" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="60" y1="207.7" x2="570" y2="207.7" class="stroke-grid"/>
<text x="54" y="211.7" class="fig-small" text-anchor="end">50</text>
<line x1="60" y1="165.4" x2="570" y2="165.4" class="stroke-grid"/>
<text x="54" y="169.4" class="fig-small" text-anchor="end">100</text>
<line x1="60" y1="123.1" x2="570" y2="123.1" class="stroke-grid"/>
<text x="54" y="127.1" class="fig-small" text-anchor="end">150</text>
<line x1="60" y1="80.8" x2="570" y2="80.8" class="stroke-grid"/>
<text x="54" y="84.8" class="fig-small" text-anchor="end">200</text>
<line x1="60" y1="38.5" x2="570" y2="38.5" class="stroke-grid"/>
<text x="54" y="42.5" class="fig-small" text-anchor="end">250</text>
<line x1="60" y1="250" x2="570" y2="250" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="250" class="stroke-muted"/>
<line x1="60.0" y1="250" x2="60.0" y2="254" class="stroke-muted"/>
<text x="60.0" y="267" class="fig-small" text-anchor="middle">0</text>
<line x1="162.0" y1="250" x2="162.0" y2="254" class="stroke-muted"/>
<text x="162.0" y="267" class="fig-small" text-anchor="middle">0.5</text>
<line x1="264.0" y1="250" x2="264.0" y2="254" class="stroke-muted"/>
<text x="264.0" y="267" class="fig-small" text-anchor="middle">1</text>
<line x1="366.0" y1="250" x2="366.0" y2="254" class="stroke-muted"/>
<text x="366.0" y="267" class="fig-small" text-anchor="middle">1.5</text>
<line x1="468.0" y1="250" x2="468.0" y2="254" class="stroke-muted"/>
<text x="468.0" y="267" class="fig-small" text-anchor="middle">2</text>
<line x1="570.0" y1="250" x2="570.0" y2="254" class="stroke-muted"/>
<text x="570.0" y="267" class="fig-small" text-anchor="middle">2.5</text>
<text x="315.0" y="284" text-anchor="middle">time (s)</text>
<text x="60" y="22" class="fig-small" text-anchor="start">F0 (Hz); 0 means unvoiced</text>
<polyline points="60.0,250.0 62.0,250.0 64.1,250.0 66.1,250.0 68.2,250.0 70.2,250.0 72.2,250.0 74.3,250.0 76.3,250.0 78.4,250.0 80.4,145.0 82.4,145.4 84.5,145.9 86.5,146.4 88.6,146.9 90.6,147.4 92.6,147.8 94.7,148.3 96.7,148.8 98.8,149.3 100.8,149.7 102.8,150.2 104.9,150.6 106.9,151.1 109.0,151.5 111.0,152.0 113.0,152.4 115.1,152.8 117.1,153.3 119.2,153.7 121.2,154.1 123.2,154.5 125.3,154.9 127.3,155.3 129.4,155.7 131.4,156.1 133.4,156.5 135.5,156.9 137.5,157.2 139.6,157.6 141.6,157.9 143.6,158.3 145.7,158.6 147.7,159.0 149.8,159.3 151.8,159.6 153.8,159.9 155.9,160.2 157.9,160.5 160.0,160.8 162.0,161.1 164.0,161.4 166.1,161.6 168.1,161.9 170.2,162.1 172.2,162.4 174.2,250.0 176.3,250.0 178.3,250.0 180.4,250.0 182.4,250.0 184.4,250.0 186.5,250.0 188.5,250.0 190.6,250.0 192.6,250.0 194.6,250.0 196.7,250.0 198.7,250.0 200.8,250.0 202.8,164.9 204.8,165.0 206.9,165.1 208.9,165.1 211.0,165.2 213.0,165.3 215.0,165.3 217.1,165.3 219.1,165.4 221.2,165.4 223.2,165.4 225.2,165.4 227.3,165.4 229.3,165.3 231.4,165.3 233.4,165.3 235.4,165.2 237.5,165.1 239.5,165.1 241.6,165.0 243.6,164.9 245.6,164.8 247.7,164.7 249.7,164.6 251.8,164.4 253.8,164.3 255.8,164.1 257.9,164.0 259.9,163.8 262.0,163.6 264.0,163.5 266.0,163.3 268.1,163.1 270.1,162.8 272.2,162.6 274.2,162.4 276.2,162.1 278.3,161.9 280.3,161.6 282.4,161.4 284.4,161.1 286.4,160.8 288.5,160.5 290.5,160.2 292.6,159.9 294.6,159.6 296.6,159.3 298.7,159.0 300.7,158.6 302.8,158.3 304.8,157.9 306.8,157.6 308.9,157.2 310.9,156.9 313.0,156.5 315.0,156.1 317.0,155.7 319.1,155.3 321.1,154.9 323.2,154.5 325.2,154.1 327.2,250.0 329.3,250.0 331.3,250.0 333.4,250.0 335.4,250.0 337.4,250.0 339.5,250.0 341.5,250.0 343.6,250.0 345.6,250.0 347.6,250.0 349.7,250.0 351.7,250.0 353.8,250.0 355.8,147.4 357.8,146.9 359.9,146.4 361.9,145.9 364.0,145.4 366.0,145.0 368.0,144.5 370.1,144.0 372.1,143.5 374.2,143.0 376.2,142.5 378.2,142.0 380.3,141.5 382.3,141.0 384.4,140.5 386.4,140.0 388.4,139.5 390.5,139.0 392.5,138.5 394.6,138.0 396.6,137.5 398.6,137.0 400.7,136.5 402.7,136.0 404.8,135.5 406.8,135.0 408.8,134.6 410.9,134.1 412.9,133.6 415.0,133.1 417.0,132.6 419.0,132.2 421.1,131.7 423.1,131.2 425.2,130.7 427.2,130.3 429.2,129.8 431.3,129.3 433.3,128.8 435.4,128.2 437.4,127.6 439.4,127.0 441.5,126.4 443.5,125.7 445.6,125.1 447.6,124.3 449.6,123.6 451.7,122.8 453.7,122.1 455.8,121.2 457.8,120.4 459.8,119.5 461.9,118.6 463.9,117.7 466.0,116.8 468.0,115.8 470.0,114.8 472.1,113.8 474.1,112.8 476.2,111.7 478.2,110.7 480.2,109.6 482.3,108.4 484.3,107.3 486.4,106.1 488.4,104.9 490.4,103.7 492.5,102.4 494.5,101.2 496.6,99.9 498.6,98.6 500.6,97.2 502.7,95.9 504.7,94.5 506.8,93.1 508.8,91.7 510.8,90.2 512.9,88.8 514.9,87.3 517.0,85.8 519.0,84.2 521.0,82.7 523.1,81.1 525.1,79.5 527.2,77.9 529.2,76.2 531.2,74.6 533.3,72.9 535.3,71.2 537.4,69.5 539.4,67.7 541.4,66.0 543.5,64.2 545.5,62.4 547.6,60.5 549.6,58.7 551.6,250.0 553.7,250.0 555.7,250.0 557.8,250.0 559.8,250.0 561.8,250.0 563.9,250.0 565.9,250.0 568.0,250.0 570.0,250.0" class="stroke-a" fill="none"/>
<polyline points="60.0,250.0 62.0,250.0 64.1,250.0 66.1,250.0 68.2,250.0 70.2,250.0 72.2,250.0 74.3,250.0 76.3,250.0 78.4,250.0 80.4,145.3 82.4,145.5 84.5,145.8 86.5,146.1 88.6,146.6 90.6,147.0 92.6,147.5 94.7,148.0 96.7,148.5 98.8,149.0 100.8,149.4 102.8,149.9 104.9,150.4 106.9,150.9 109.0,151.3 111.0,151.7 113.0,152.2 115.1,152.6 117.1,153.0 119.2,153.4 121.2,153.9 123.2,154.3 125.3,154.7 127.3,155.1 129.4,155.5 131.4,155.9 133.4,156.2 135.5,156.6 137.5,156.9 139.6,157.3 141.6,157.7 143.6,158.1 145.7,158.5 147.7,158.8 149.8,159.1 151.8,159.4 153.8,159.7 155.9,160.0 157.9,160.3 160.0,160.6 162.0,160.9 164.0,161.2 166.1,161.4 168.1,161.7 170.2,161.8 172.2,161.9 174.2,161.9 176.3,250.0 178.3,250.0 180.4,250.0 182.4,250.0 184.4,250.0 186.5,250.0 188.5,250.0 190.6,250.0 192.6,250.0 194.6,250.0 196.7,250.0 198.7,250.0 200.8,250.0 202.8,250.0 204.8,165.0 206.9,165.1 208.9,165.1 211.0,165.2 213.0,165.2 215.0,165.3 217.1,165.3 219.1,165.3 221.2,165.4 223.2,165.4 225.2,165.4 227.3,165.4 229.3,165.3 231.4,165.3 233.4,165.3 235.4,165.2 237.5,165.2 239.5,165.1 241.6,165.0 243.6,164.9 245.6,164.8 247.7,164.7 249.7,164.6 251.8,164.5 253.8,164.3 255.8,164.2 257.9,164.0 259.9,163.9 262.0,163.7 264.0,163.5 266.0,163.3 268.1,163.1 270.1,162.9 272.2,162.7 274.2,162.5 276.2,162.3 278.3,162.0 280.3,161.8 282.4,161.5 284.4,161.3 286.4,161.0 288.5,160.8 290.5,160.5 292.6,160.1 294.6,159.8 296.6,159.4 298.7,159.1 300.7,158.8 302.8,158.5 304.8,158.1 306.8,157.8 308.9,157.5 310.9,157.1 313.0,156.8 315.0,156.4 317.0,156.0 319.1,155.6 321.1,155.1 323.2,154.9 325.2,154.6 327.2,154.4 329.3,250.0 331.3,250.0 333.4,250.0 335.4,250.0 337.4,250.0 339.5,250.0 341.5,250.0 343.6,250.0 345.6,250.0 347.6,250.0 349.7,250.0 351.7,250.0 353.8,250.0 355.8,250.0 357.8,146.5 359.9,146.3 361.9,146.0 364.0,145.7 366.0,145.2 368.0,144.8 370.1,144.2 372.1,143.7 374.2,143.3 376.2,142.9 378.2,142.2 380.3,141.8 382.3,141.4 384.4,140.9 386.4,140.3 388.4,139.8 390.5,139.4 392.5,138.8 394.6,138.3 396.6,137.9 398.6,137.3 400.7,136.9 402.7,136.4 404.8,135.9 406.8,135.4 408.8,134.9 410.9,134.4 412.9,134.0 415.0,133.4 417.0,132.9 419.0,132.5 421.1,132.0 423.1,131.5 425.2,131.0 427.2,130.6 429.2,130.0 431.3,129.6 433.3,129.1 435.4,128.5 437.4,128.0 439.4,127.3 441.5,126.8 443.5,126.0 445.6,125.5 447.6,124.6 449.6,124.0 451.7,123.2 453.7,122.5 455.8,121.6 457.8,120.9 459.8,119.9 461.9,119.2 463.9,118.2 466.0,117.3 468.0,116.5 470.0,115.6 472.1,114.6 474.1,113.7 476.2,112.5 478.2,111.5 480.2,110.5 482.3,109.3 484.3,108.3 486.4,107.2 488.4,105.8 490.4,104.5 492.5,103.5 494.5,102.1 496.6,100.7 498.6,99.5 500.6,98.1 502.7,96.9 504.7,95.5 506.8,94.3 508.8,92.9 510.8,91.4 512.9,89.9 514.9,88.3 517.0,86.7 519.0,85.2 521.0,83.7 523.1,82.0 525.1,80.4 527.2,78.7 529.2,77.1 531.2,75.5 533.3,73.9 535.3,72.3 537.4,70.7 539.4,68.9 541.4,67.2 543.5,65.6 545.5,64.0 547.6,63.2 549.6,62.4 551.6,61.6 553.7,60.9 555.7,250.0 557.8,250.0 559.8,250.0 561.8,250.0 563.9,250.0 565.9,250.0 568.0,250.0 570.0,250.0" class="stroke-b dashed" fill="none"/>
<line x1="350" y1="45" x2="380" y2="45" class="stroke-a"/><text x="386" y="49" class="fig-small">true F0</text><line x1="350" y1="63" x2="380" y2="63" class="stroke-b dashed"/><text x="386" y="67" class="fig-small">YIN track</text>
</svg>
Figure 3. YIN on a synthetic voice that asks a question, with two hissy gaps at 0.55 s and 1.3 s. The track follows the fall and the final rise, and drops to 0 in both gaps. Against the truth it has a gross pitch error of 0.0% and a voicing error of 2.4%.
```

The following demo runs YIN on a short clip, so you can move the threshold and watch the errors appear.

```demo
{"id": "pitch-tracker", "caption": "Lower the threshold toward 0.02 to see octave-low errors in the creaky ending, then raise it past 0.2 to see breath frames get a pitch and, past 0.3, octave-high errors."}
```

## Score a pitch track

Two numbers score a track against a reference. The *gross pitch error* (GPE) is the share of frames voiced in both tracks whose estimate is off by more than 20%:

$$
\text{GPE} = \frac{\left|\{t : \hat f_t > 0,\ f_t > 0,\ |\hat f_t - f_t| > 0.2 f_t\}\right|}{\left|\{t : \hat f_t > 0,\ f_t > 0\}\right|}
$$

Read it aloud: among frames where the estimate $\hat f_t$ and the truth $f_t$ are both voiced, count those more than 20% off, and divide by how many there are. The 20% bound catches octave errors, which are 50% or 100% off. The *voicing decision error* is the share of all frames where the two tracks disagree about voicing.

For the reference $[0, 100, 100, 200, 200, 0, 150, 150, 0, 0]$ Hz and the estimate $[0, 101, 50, 205, 250, 120, 0, 179, 0, 0]$ Hz, five frames are voiced in both. Two are gross errors: 50 against 100, and 250 against 200, which is 25% off. So GPE is 2/5 = 40%. Voicing disagrees on frames 5 and 6, so the voicing error is 2/10 = 20%.

> [!KEY]
> You now know how to find the period of a voice without picking the wrong octave. YIN asks how different a frame is from its shifted self, takes the first normalized dip under a threshold, and its aperiodicity says whether the frame is voiced at all.

> [!RECAP]
> - Autocorrelation peaks at every multiple of the period, which causes octave errors.
> - YIN's $d(\tau)$ removes the energy bias, and $d'(\tau)$ divides by the running mean.
> - Take the first dip of $d'$ under 0.1, follow it to its bottom, and refine it with a parabola.
> - $d'$ at the dip is the aperiodicity, which decides voicing.
> - GPE counts voiced frames more than 20% off, and voicing error counts disagreements.

## Build it step by step

The checks test the functions in this order.

1. **`difference_function(frame, max_lag)`** returns `max_lag + 1` values. Use `sliding_window_view` to get every shifted stretch at once. The usual mistake is letting $W$ shrink with the lag, which biases $d$ downward at long lags.
2. **`cmndf(d)`** returns $d'$ with `np.cumsum`. The usual mistake is dividing by the running sum without multiplying by $\tau$, which is the sum, not the mean. Guard the zero sum of a silent frame.
3. **`absolute_threshold(dprime, tau_min, tau_max, threshold)`** scans, slides to the bottom, or falls back to `np.argmin`. The usual mistake is returning `np.argmin` every time, which brings the octave errors back.
4. **`parabolic_interpolation(y, tau)`** applies the vertex formula and returns `tau` at the edges or on a straight line.
5. **`yin(frame, sample_rate, fmin, fmax, threshold)`** sets $\tau_{\min} = \lfloor f_s / f_{\max} \rfloor$ and $\tau_{\max} = \lceil f_s / f_{\min} \rceil$, and chains the four functions. It returns $f_s / \tau^*$ and the aperiodicity.
6. **`gross_pitch_error(f0_est, f0_ref, tolerance)`** and **`voicing_error(f0_est, f0_ref)`** are boolean masks and means.
7. **`autocorr_pitch(frame, sample_rate, fmin, fmax)`** is the obvious method for comparison: a dot product instead of a squared difference, and `np.argmax`.
8. **`track_pitch(x, sample_rate, frame_length, hop, ...)`** pads by reflection, loops over frames, and applies the voicing threshold.

## Where this breaks

**The search range misses $F_0$.** A voice at 50 Hz with $f_{\min} = 60$ Hz gets a harmonic instead, so set the range for the speaker.

**Octave jumps between frames.** One frame can still pick $2T_0$. pYIN replaces the single threshold with a distribution over thresholds, keeps several candidates per frame, and chooses a smooth path through them with an HMM, which [Hidden Markov models](/lessons/022-hidden-markov-models) explains.

**Creaky, breathy, and boundary frames.** Irregular periods or strong noise push $d'$ up, and a voiced frame is marked unvoiced. A 64 ms frame at a vowel's edge holds half voice and half silence, so voicing flips a frame or two late.

**Polyphony.** Two voices put two periods in one frame, and YIN finds one or neither. Neural trackers such as CREPE, which classifies 1024-sample frames of 16 kHz audio into 360 pitch bins 20 cents apart, handle noise better but need training data.

> [!WARNING]
> A hop of 160 samples is 10 ms, and a lag of 160 samples is a pitch of 100 Hz. Convert with $f_s$ every time: $F_0 = f_s / \tau$, and time $= t \cdot H / f_s$.

## The exercise

You build `difference_function`, `cmndf`, `absolute_threshold`, `parabolic_interpolation`, `yin`, `gross_pitch_error`, `voicing_error`, `autocorr_pitch`, and `track_pitch`. Press **Run** to hear a synthetic voice ask a question, see its spectrogram and your pitch track against the truth, compare YIN with the autocorrelation peak on 60 random voices, and hear your track hummed back as a sine.

The checks work the hand examples for $d$, $d'$, and the threshold, compare $d$ with a brute-force sum, and verify the parabola's vertex. They require YIN within 0.5% on clean tones, within 0.1% between two lags, and right on a missing fundamental, with under 2% gross errors on 60 noisy voices. They also score a hand-made track and track a gliding voice with a noise burst and silence, requiring voicing right on 95% of frames.
