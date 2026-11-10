---
title: Voice activity detection
slug: 020-voice-activity-detection
part: "Part 2 — Hearing and speech production"
summary: Mark speech frames with an adaptive noise floor or a likelihood ratio, switch with two thresholds, score by frame F1, and find where an utterance ends.
minutes: 35
gpu: false
objectives:
  - Compute the energy of each frame in decibels, and explain why a threshold must sit relative to the recording's own noise floor.
  - Explain why a single threshold flickers, and run hysteresis with a hangover by hand.
  - Build an energy voice activity detector from frame energy, a percentile floor, and hysteresis.
  - Score a detector's decisions by frame F1, and say where an energy detector fails.
  - Track a noise floor that follows a changing background, and say why it freezes on speech frames.
  - Compute the log-likelihood ratio of two diagonal Gaussians over energy and spectral flatness.
  - Decide that a speaker has finished with a trailing-silence timeout, and name its trade-off.
prereqs:
  - 001-what-sound-is
  - 007-the-short-time-fourier-transform
exercise: 020-voice-activity-detection
---

# Voice activity detection

A recording of a meeting is mostly not speech. There are pauses, a fan, and a door. A recognizer that transcribes all of it wastes compute on silence and can invent words from the noise, and a phone call that sends all of it wastes bandwidth. Both want a *voice activity detector* (VAD): a program that marks which stretches of a recording hold speech.

The cue you reach for first is loudness. So how loud is loud enough? A fixed number fails the moment the microphone changes, and a single threshold chatters on and off whenever the energy hovers near it.

This chapter builds a detector from frame energy in three small steps, each fixing the failure of the one before. Figure 1 shows it on a noisy clip. Then it fixes two weak spots, a background that changes and a loudness cue that fails in noise, and it decides when the speaker has finished.

```figure
<svg viewBox="0 0 600 345" role="img" aria-label="Frame energy in decibels of a noisy recording with the noise floor and two thresholds, above two strips that show where speech really is and where the detector says it is">
<line x1="60" y1="201.4" x2="570" y2="201.4" class="stroke-grid"/>
<text x="54" y="205.4" class="fig-small" text-anchor="end">-40</text>
<line x1="60" y1="144.3" x2="570" y2="144.3" class="stroke-grid"/>
<text x="54" y="148.3" class="fig-small" text-anchor="end">-30</text>
<line x1="60" y1="87.1" x2="570" y2="87.1" class="stroke-grid"/>
<text x="54" y="91.1" class="fig-small" text-anchor="end">-20</text>
<line x1="60" y1="230" x2="570" y2="230" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="230" class="stroke-muted"/>
<line x1="60.0" y1="230" x2="60.0" y2="234" class="stroke-muted"/>
<text x="60.0" y="247" class="fig-small" text-anchor="middle">0</text>
<line x1="145.0" y1="230" x2="145.0" y2="234" class="stroke-muted"/>
<text x="145.0" y="247" class="fig-small" text-anchor="middle">0.5</text>
<line x1="230.0" y1="230" x2="230.0" y2="234" class="stroke-muted"/>
<text x="230.0" y="247" class="fig-small" text-anchor="middle">1</text>
<line x1="315.0" y1="230" x2="315.0" y2="234" class="stroke-muted"/>
<text x="315.0" y="247" class="fig-small" text-anchor="middle">1.5</text>
<line x1="400.0" y1="230" x2="400.0" y2="234" class="stroke-muted"/>
<text x="400.0" y="247" class="fig-small" text-anchor="middle">2</text>
<line x1="485.0" y1="230" x2="485.0" y2="234" class="stroke-muted"/>
<text x="485.0" y="247" class="fig-small" text-anchor="middle">2.5</text>
<line x1="570.0" y1="230" x2="570.0" y2="234" class="stroke-muted"/>
<text x="570.0" y="247" class="fig-small" text-anchor="middle">3</text>
<text x="60" y="22" class="fig-small" text-anchor="start">frame energy (dB)</text>
<polyline points="60.0,156.3 61.7,148.4 63.4,149.6 65.1,154.2 66.8,149.1 68.5,135.3 70.2,135.5 71.9,141.1 73.6,161.0 75.3,161.7 77.0,144.9 78.7,132.7 80.4,136.8 82.1,141.3 83.8,149.7 85.5,144.7 87.2,144.1 88.9,149.1 90.6,150.7 92.3,151.2 94.0,145.7 95.7,125.7 97.4,106.7 99.1,95.4 100.8,94.8 102.5,95.4 104.2,93.0 105.9,93.1 107.6,90.4 109.3,89.4 111.0,89.3 112.7,89.9 114.4,90.3 116.1,90.7 117.8,91.2 119.5,90.4 121.2,89.4 122.9,90.5 124.6,93.1 126.3,95.2 128.0,96.2 129.7,96.6 131.4,97.1 133.1,97.3 134.8,96.4 136.5,98.3 138.2,99.2 139.9,99.2 141.6,95.1 143.3,89.6 145.0,88.4 146.7,88.6 148.4,90.4 150.1,91.2 151.8,92.8 153.5,95.1 155.2,92.2 156.9,94.0 158.6,94.2 160.3,97.2 162.0,99.2 163.7,100.8 165.4,98.3 167.1,100.8 168.8,110.1 170.5,132.9 172.2,153.5 173.9,153.3 175.6,153.7 177.3,154.0 179.0,157.3 180.7,153.5 182.4,149.4 184.1,149.0 185.8,158.1 187.5,150.7 189.2,152.9 190.9,150.8 192.6,151.4 194.3,149.2 196.0,148.2 197.7,155.9 199.4,160.9 201.1,155.7 202.8,154.2 204.5,149.1 206.2,133.9 207.9,126.4 209.6,123.3 211.3,127.6 213.0,134.7 214.7,132.5 216.4,137.4 218.1,142.3 219.8,166.7 221.5,159.5 223.2,158.3 224.9,161.3 226.6,165.9 228.3,167.9 230.0,150.2 231.7,151.1 233.4,155.8 235.1,155.4 236.8,152.6 238.5,156.9 240.2,155.8 241.9,156.9 243.6,158.3 245.3,129.8 247.0,96.4 248.7,81.1 250.4,77.0 252.1,79.0 253.8,80.6 255.5,81.2 257.2,80.1 258.9,78.2 260.6,77.8 262.3,76.9 264.0,76.1 265.7,75.2 267.4,75.4 269.1,73.0 270.8,73.2 272.5,76.5 274.2,77.5 275.9,75.9 277.6,77.1 279.3,79.4 281.0,80.2 282.7,79.9 284.4,81.9 286.1,82.8 287.8,80.4 289.5,78.6 291.2,76.1 292.9,71.3 294.6,67.8 296.3,64.5 298.0,62.9 299.7,62.4 301.4,63.4 303.1,63.7 304.8,64.6 306.5,64.7 308.2,63.3 309.9,64.8 311.6,67.5 313.3,71.6 315.0,76.4 316.7,78.7 318.4,81.9 320.1,84.4 321.8,81.7 323.5,80.3 325.2,78.4 326.9,79.8 328.6,78.3 330.3,74.9 332.0,74.9 333.7,76.6 335.4,76.3 337.1,76.3 338.8,86.2 340.5,108.7 342.2,136.5 343.9,144.1 345.6,149.2 347.3,146.5 349.0,147.9 350.7,138.7 352.4,125.6 354.1,118.7 355.8,124.4 357.5,133.6 359.2,137.2 360.9,144.1 362.6,158.5 364.3,161.0 366.0,160.4 367.7,159.4 369.4,160.3 371.1,151.0 372.8,149.1 374.5,151.3 376.2,155.7 377.9,165.1 379.6,162.3 381.3,158.2 383.0,154.3 384.7,154.2 386.4,162.0 388.1,161.7 389.8,159.5 391.5,140.8 393.2,138.2 394.9,142.7 396.6,150.0 398.3,141.3 400.0,138.0 401.7,134.4 403.4,146.3 405.1,157.4 406.8,171.2 408.5,155.5 410.2,147.7 411.9,150.6 413.6,147.5 415.3,142.3 417.0,143.5 418.7,159.8 420.4,160.2 422.1,147.3 423.8,108.5 425.5,89.7 427.2,80.3 428.9,79.1 430.6,80.6 432.3,82.0 434.0,81.9 435.7,83.3 437.4,86.2 439.1,90.0 440.8,90.3 442.5,89.2 444.2,87.6 445.9,87.8 447.6,92.1 449.3,95.4 451.0,98.3 452.7,99.2 454.4,97.9 456.1,98.7 457.8,101.2 459.5,101.2 461.2,102.5 462.9,103.4 464.6,105.0 466.3,104.7 468.0,104.1 469.7,104.6 471.4,104.4 473.1,106.2 474.8,104.3 476.5,104.3 478.2,103.4 479.9,102.0 481.6,98.8 483.3,98.5 485.0,100.1 486.7,102.0 488.4,104.4 490.1,105.3 491.8,104.6 493.5,105.4 495.2,106.1 496.9,105.2 498.6,103.5 500.3,102.9 502.0,102.4 503.7,102.7 505.4,104.2 507.1,102.9 508.8,107.5 510.5,120.0 512.2,151.4 513.9,156.5 515.6,151.7 517.3,147.0 519.0,147.6 520.7,133.2 522.4,130.7 524.1,133.8 525.8,145.7 527.5,139.9 529.2,132.3 530.9,130.9 532.6,123.4 534.3,116.9 536.0,117.8 537.7,116.9 539.4,112.2 541.1,119.2 542.8,129.8 544.5,133.1 546.2,140.1 547.9,150.1 549.6,151.9 551.3,152.6 553.0,158.2 554.7,152.3 556.4,156.1 558.1,154.9 559.8,163.7 561.5,163.5 563.2,153.5 564.9,153.6 566.6,153.3 568.3,162.3 570.0,163.4" class="stroke-a" fill="none"/>
<line x1="60" y1="157.4" x2="570" y2="157.4" class="stroke-muted dashed"/>
<line x1="60" y1="128.8" x2="570" y2="128.8" class="stroke-muted dashed"/>
<line x1="60" y1="105.9" x2="570" y2="105.9" class="stroke-ink dashed"/>
<text x="68.5" y="185.4" class="fig-small" text-anchor="start">dark dashed: on threshold, floor + 9 dB</text>
<text x="68.5" y="202.6" class="fig-small" text-anchor="start">light dashed: off threshold, floor + 5 dB, and the floor (10th percentile)</text>
<text x="56" y="264" class="fig-small" text-anchor="end">truth</text>
<rect x="94.0" y="250" width="78.2" height="20" class="fill-soft-c"/>
<rect x="245.3" y="250" width="96.9" height="20" class="fill-soft-c"/>
<rect x="423.8" y="250" width="88.4" height="20" class="fill-soft-c"/>
<text x="56" y="290" class="fig-small" text-anchor="end">VAD</text>
<rect x="99.1" y="276" width="79.9" height="20" class="fill-b"/>
<rect x="247.0" y="276" width="103.7" height="20" class="fill-b"/>
<rect x="425.5" y="276" width="95.2" height="20" class="fill-b"/>
<line x1="60" y1="300" x2="570" y2="300" class="stroke-muted"/>
<line x1="60.0" y1="300" x2="60.0" y2="304" class="stroke-muted"/><text x="60.0" y="317" class="fig-small" text-anchor="middle">0</text>
<line x1="145.0" y1="300" x2="145.0" y2="304" class="stroke-muted"/><text x="145.0" y="317" class="fig-small" text-anchor="middle">0.5</text>
<line x1="230.0" y1="300" x2="230.0" y2="304" class="stroke-muted"/><text x="230.0" y="317" class="fig-small" text-anchor="middle">1</text>
<line x1="315.0" y1="300" x2="315.0" y2="304" class="stroke-muted"/><text x="315.0" y="317" class="fig-small" text-anchor="middle">1.5</text>
<line x1="400.0" y1="300" x2="400.0" y2="304" class="stroke-muted"/><text x="400.0" y="317" class="fig-small" text-anchor="middle">2</text>
<line x1="485.0" y1="300" x2="485.0" y2="304" class="stroke-muted"/><text x="485.0" y="317" class="fig-small" text-anchor="middle">2.5</text>
<line x1="570.0" y1="300" x2="570.0" y2="304" class="stroke-muted"/><text x="570.0" y="317" class="fig-small" text-anchor="middle">3</text>
<text x="315" y="336" text-anchor="middle">time (s)</text>
</svg>
Figure 1. A 3 s clip of three voiced bursts in noise at 10 dB SNR. The energy sits near the noise floor (-32.3 dB) in silence and about 10 dB above it during speech, so a threshold relative to the floor separates them. The VAD strip starts a few frames after the truth, while the energy climbs over the on threshold, and ends a few frames after it, because of the hangover. Its frame F1 is 0.94.
```

## What you already know

- A decibel is $10 \log_{10}$ of a power ratio, and the mean of $x^2$ over a frame is its power.
- A centered frame is padded by half its length at each end by reflection, so $L$ samples give $1 + \lfloor L / H \rfloor$ frames at hop $H$.
- A percentile, `np.percentile(v, 10)`, is the value that 10% of the numbers fall below.

## Measure the loudness of each frame

Slide a frame along the recording, as in the short-time Fourier transform, and ask one question per frame: how much power does it carry? The *frame energy* in decibels is:

$$
E_t = 10 \log_{10}\left(\frac{1}{N} \sum_{n=0}^{N-1} x_t[n]^2 + 10^{-10}\right)
$$

Read it aloud: square each of the $N$ samples in frame $t$, average them to get the power, add a tiny constant so that silence gives $-100$ dB instead of minus infinity, and convert to decibels. The unit is decibels relative to a full-scale sample of 1, so every value is negative inside $[-1, 1]$. A sine of amplitude 0.5 has a mean power of 0.125, which is $-9.03$ dB.

## Why a fixed threshold fails

The obvious detector says speech is any frame above a fixed level, such as $-30$ dB. On the clip in Figure 1, that works: it gives a frame F1 of 0.87, a score defined below. But the energies depend on the microphone gain. Record the same clip 20 dB quieter, which means scaling the samples by 0.1, and no frame reaches $-30$ dB, so the detector finds no speech at all. Lower the threshold to $-40$ dB to catch the quiet recording, and the original clip becomes speech in every frame, because its room noise sits at about $-32$ dB.

So the threshold must be a number *above the noise floor*, the energy of the recording when nobody speaks. You can estimate the floor as the 10th percentile of the frame energies, which works when at least a tenth of the recording is quiet. Speech is then any frame some margin above the floor. Scaling the recording moves the energies and the floor together, so the decisions do not change: on the same clip, the floor-relative detector gives the same decisions at gains of 1, 0.1, and 0.01.

> [!KEY]
> Loudness depends on the microphone, so compare each frame with the recording's own noise floor. A margin above the floor is a gain-free threshold.

## Why a single threshold flickers

Even a floor-relative threshold errs at the edges of words, because energy rises and falls inside a word and room noise bumps up and down between words. Wherever the energy hovers near the one threshold, the decision flips. On the clip in Figure 1, a single threshold at the floor plus 9 dB switches 10 times, which is five stretches, where the truth has three.

*Hysteresis* uses two thresholds. Switch on only above a high one, and switch off only below a lower one. In between, keep whatever state you were in. A *hangover* adds a delay: stay on until the energy has been below the off threshold for more than a few frames, so a quiet final consonant is not cut off. The exercise switches on at 9 dB above the floor and off at 5 dB above it, with a hangover of 5 frames, which is 50 ms at a 10 ms hop.

Run it by hand. Take the energies $[0, 12, 8, 3, 3, 3, 9, 2, 2]$ dB, an on threshold of 10, an off threshold of 5, and a hangover of 2 frames.

1. Frame 0 is 0 dB, not above 10, so the state is silence. Frame 1 is 12 dB, so speech starts.
2. Frame 2 is 8 dB, below 10 but not below 5, so speech continues.
3. Frames 3 and 4 are below 5, so the quiet count reaches 1, then 2, which is not more than the hangover. Speech continues.
4. Frame 5 makes the count 3, so speech ends.
5. Frame 6 is 9 dB, above the off threshold but not the on threshold, so silence continues, as it does at frames 7 and 8.

The decisions are $[0, 1, 1, 1, 1, 0, 0, 0, 0]$, as in Figure 2.

```figure
<svg viewBox="0 0 600 325" role="img" aria-label="Energies of nine frames, with the decisions of a single threshold at 10, a single threshold at 5, and hysteresis">
<line x1="150" y1="150.0" x2="570" y2="150.0" class="stroke-grid"/>
<text x="144" y="154.0" class="fig-small" text-anchor="end">0</text>
<line x1="150" y1="107.1" x2="570" y2="107.1" class="stroke-grid"/>
<text x="144" y="111.1" class="fig-small" text-anchor="end">5</text>
<line x1="150" y1="64.3" x2="570" y2="64.3" class="stroke-grid"/>
<text x="144" y="68.3" class="fig-small" text-anchor="end">10</text>
<line x1="150" y1="150" x2="570" y2="150" class="stroke-muted"/>
<line x1="150" y1="30" x2="150" y2="150" class="stroke-muted"/>
<text x="150" y="22" class="fig-small" text-anchor="start">energy (dB)</text>
<polyline points="173.3,150.0 220.0,47.1 266.7,81.4 313.3,124.3 360.0,124.3 406.7,124.3 453.3,72.9 500.0,132.9 546.7,132.9" class="stroke-a" fill="none"/>
<circle cx="173.3" cy="150.0" r="4" class="fill-a"/>
<circle cx="220.0" cy="47.1" r="4" class="fill-a"/>
<circle cx="266.7" cy="81.4" r="4" class="fill-a"/>
<circle cx="313.3" cy="124.3" r="4" class="fill-a"/>
<circle cx="360.0" cy="124.3" r="4" class="fill-a"/>
<circle cx="406.7" cy="124.3" r="4" class="fill-a"/>
<circle cx="453.3" cy="72.9" r="4" class="fill-a"/>
<circle cx="500.0" cy="132.9" r="4" class="fill-a"/>
<circle cx="546.7" cy="132.9" r="4" class="fill-a"/>
<line x1="150" y1="64.3" x2="570" y2="64.3" class="stroke-ink dashed"/>
<line x1="150" y1="107.1" x2="570" y2="107.1" class="stroke-muted dashed"/>
<text x="570.0" y="60.0" class="fig-small" text-anchor="end">on = 10</text>
<text x="570.0" y="102.9" class="fig-small" text-anchor="end">off = 5</text>
<text x="144" y="192" class="fig-small" text-anchor="end">one threshold, 10</text>
<rect x="156.3" y="175" width="34" height="24" class="fill-sunk"/>
<text x="173.3" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="203.0" y="175" width="34" height="24" class="fill-b"/>
<text x="220.0" y="192" class="fig-small" text-anchor="middle">1</text>
<rect x="249.7" y="175" width="34" height="24" class="fill-sunk"/>
<text x="266.7" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="296.3" y="175" width="34" height="24" class="fill-sunk"/>
<text x="313.3" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="343.0" y="175" width="34" height="24" class="fill-sunk"/>
<text x="360.0" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="389.7" y="175" width="34" height="24" class="fill-sunk"/>
<text x="406.7" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="436.3" y="175" width="34" height="24" class="fill-sunk"/>
<text x="453.3" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="483.0" y="175" width="34" height="24" class="fill-sunk"/>
<text x="500.0" y="192" class="fig-small" text-anchor="middle">0</text>
<rect x="529.7" y="175" width="34" height="24" class="fill-sunk"/>
<text x="546.7" y="192" class="fig-small" text-anchor="middle">0</text>
<text x="144" y="226" class="fig-small" text-anchor="end">one threshold, 5</text>
<rect x="156.3" y="209" width="34" height="24" class="fill-sunk"/>
<text x="173.3" y="226" class="fig-small" text-anchor="middle">0</text>
<rect x="203.0" y="209" width="34" height="24" class="fill-b"/>
<text x="220.0" y="226" class="fig-small" text-anchor="middle">1</text>
<rect x="249.7" y="209" width="34" height="24" class="fill-b"/>
<text x="266.7" y="226" class="fig-small" text-anchor="middle">1</text>
<rect x="296.3" y="209" width="34" height="24" class="fill-sunk"/>
<text x="313.3" y="226" class="fig-small" text-anchor="middle">0</text>
<rect x="343.0" y="209" width="34" height="24" class="fill-sunk"/>
<text x="360.0" y="226" class="fig-small" text-anchor="middle">0</text>
<rect x="389.7" y="209" width="34" height="24" class="fill-sunk"/>
<text x="406.7" y="226" class="fig-small" text-anchor="middle">0</text>
<rect x="436.3" y="209" width="34" height="24" class="fill-b"/>
<text x="453.3" y="226" class="fig-small" text-anchor="middle">1</text>
<rect x="483.0" y="209" width="34" height="24" class="fill-sunk"/>
<text x="500.0" y="226" class="fig-small" text-anchor="middle">0</text>
<rect x="529.7" y="209" width="34" height="24" class="fill-sunk"/>
<text x="546.7" y="226" class="fig-small" text-anchor="middle">0</text>
<text x="144" y="260" class="fig-small" text-anchor="end">hysteresis, hangover 2</text>
<rect x="156.3" y="243" width="34" height="24" class="fill-sunk"/>
<text x="173.3" y="260" class="fig-small" text-anchor="middle">0</text>
<rect x="203.0" y="243" width="34" height="24" class="fill-b"/>
<text x="220.0" y="260" class="fig-small" text-anchor="middle">1</text>
<rect x="249.7" y="243" width="34" height="24" class="fill-b"/>
<text x="266.7" y="260" class="fig-small" text-anchor="middle">1</text>
<rect x="296.3" y="243" width="34" height="24" class="fill-b"/>
<text x="313.3" y="260" class="fig-small" text-anchor="middle">1</text>
<rect x="343.0" y="243" width="34" height="24" class="fill-b"/>
<text x="360.0" y="260" class="fig-small" text-anchor="middle">1</text>
<rect x="389.7" y="243" width="34" height="24" class="fill-sunk"/>
<text x="406.7" y="260" class="fig-small" text-anchor="middle">0</text>
<rect x="436.3" y="243" width="34" height="24" class="fill-sunk"/>
<text x="453.3" y="260" class="fig-small" text-anchor="middle">0</text>
<rect x="483.0" y="243" width="34" height="24" class="fill-sunk"/>
<text x="500.0" y="260" class="fig-small" text-anchor="middle">0</text>
<rect x="529.7" y="243" width="34" height="24" class="fill-sunk"/>
<text x="546.7" y="260" class="fig-small" text-anchor="middle">0</text>
<line x1="150" y1="280" x2="570" y2="280" class="stroke-muted"/>
<text x="173.3" y="296" class="fig-small" text-anchor="middle">0</text>
<text x="220.0" y="296" class="fig-small" text-anchor="middle">1</text>
<text x="266.7" y="296" class="fig-small" text-anchor="middle">2</text>
<text x="313.3" y="296" class="fig-small" text-anchor="middle">3</text>
<text x="360.0" y="296" class="fig-small" text-anchor="middle">4</text>
<text x="406.7" y="296" class="fig-small" text-anchor="middle">5</text>
<text x="453.3" y="296" class="fig-small" text-anchor="middle">6</text>
<text x="500.0" y="296" class="fig-small" text-anchor="middle">7</text>
<text x="546.7" y="296" class="fig-small" text-anchor="middle">8</text>
<text x="360" y="314" text-anchor="middle">frame t</text>
</svg>
Figure 2. The hand example. A single threshold at 10 drops speech at frame 2, where the energy dips to 8. A single threshold at 5 keeps frame 2 but switches speech back on at frame 6, when the energy bumps to 9. Hysteresis keeps frames 1 to 4 as one stretch of speech and stays off at frame 6.
```

The whole state is two variables, the on or off state and the count of quiet frames in a row, updated in one loop over the frames.

> [!KEY]
> Two thresholds and a hangover turn a flickering signal into stretches. Hysteresis switches on high and off low, and the hangover protects the quiet end of a word.

## Score the decisions

Treat speech frames as the positive class. A *true positive* (TP) is a speech frame marked speech, a *false positive* (FP) is a non-speech frame marked speech, and a *false negative* (FN) is a speech frame that the detector missed. The frame F1 score combines precision and recall:

$$
F_1 = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
$$

Read it aloud: twice the hits, over twice the hits plus both kinds of mistake. It is 1 for a perfect detector. For decisions $[1, 1, 0, 0, 1, 0]$ against the truth $[1, 0, 0, 1, 1, 0]$, TP = 2, FP = 1, and FN = 1, so $F_1 = 4 / 6 = 0.667$.

On the clip of Figure 1, three voiced bursts in pink noise, the detector reaches a frame F1 of 0.92 at 20 dB signal-to-noise ratio (SNR) and 0.94 at 10 dB. At 5 dB it falls to 0.77, because quiet speech sinks into the noise floor.

## Track a floor that moves

The percentile floor assumes that the background stays put. In Figure 3 a fan switches on at 2 s and the noise rises by 10 dB. The floor stays at the old level, so the noise alone sits above the on threshold, and the fixed detector reports speech until the clip ends.

```figure
<svg viewBox="0 0 600 362" role="img" aria-label="Frame energy of a clip whose background noise rises 10 dB at 2 seconds, with a fixed on threshold and a tracked on threshold, above strips for the truth and the two detectors">
<line x1="60" y1="180.0" x2="570" y2="180.0" class="stroke-grid"/>
<text x="54" y="184.0" class="fig-small" text-anchor="end">-40</text>
<line x1="60" y1="130.0" x2="570" y2="130.0" class="stroke-grid"/>
<text x="54" y="134.0" class="fig-small" text-anchor="end">-30</text>
<line x1="60" y1="80.0" x2="570" y2="80.0" class="stroke-grid"/>
<text x="54" y="84.0" class="fig-small" text-anchor="end">-20</text>
<line x1="60" y1="230" x2="570" y2="230" class="stroke-muted"/>
<line x1="60" y1="30" x2="60" y2="230" class="stroke-muted"/>
<line x1="60.0" y1="230" x2="60.0" y2="234" class="stroke-muted"/>
<text x="60.0" y="247" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="230" x2="187.5" y2="234" class="stroke-muted"/>
<text x="187.5" y="247" class="fig-small" text-anchor="middle">1</text>
<line x1="315.0" y1="230" x2="315.0" y2="234" class="stroke-muted"/>
<text x="315.0" y="247" class="fig-small" text-anchor="middle">2</text>
<line x1="442.5" y1="230" x2="442.5" y2="234" class="stroke-muted"/>
<text x="442.5" y="247" class="fig-small" text-anchor="middle">3</text>
<line x1="570.0" y1="230" x2="570.0" y2="234" class="stroke-muted"/>
<text x="570.0" y="247" class="fig-small" text-anchor="middle">4</text>
<text x="60" y="22" class="fig-small" text-anchor="start">frame energy (dB)</text>
<polyline points="60.0,153.3 61.3,154.5 62.5,162.3 63.8,160.7 65.1,154.5 66.4,154.5 67.7,170.9 68.9,177.6 70.2,182.2 71.5,182.5 72.8,180.0 74.0,182.3 75.3,180.7 76.6,184.5 77.8,185.6 79.1,181.5 80.4,183.1 81.7,189.6 83.0,189.6 84.2,191.4 85.5,125.2 86.8,87.2 88.0,71.7 89.3,64.2 90.6,64.2 91.9,66.5 93.2,68.1 94.4,70.7 95.7,65.5 97.0,63.2 98.2,61.7 99.5,60.8 100.8,58.0 102.1,60.3 103.3,60.7 104.6,60.8 105.9,61.4 107.2,62.9 108.5,64.9 109.7,67.2 111.0,68.1 112.3,67.3 113.5,65.6 114.8,64.1 116.1,66.5 117.4,66.5 118.7,66.8 119.9,66.6 121.2,66.3 122.5,67.1 123.8,69.5 125.0,68.5 126.3,68.5 127.6,66.9 128.8,64.9 130.1,63.5 131.4,61.2 132.7,59.8 133.9,62.2 135.2,69.3 136.5,86.2 137.8,130.6 139.1,175.4 140.3,179.9 141.6,168.6 142.9,163.5 144.2,155.2 145.4,157.7 146.7,163.3 148.0,168.8 149.2,173.1 150.5,184.7 151.8,182.4 153.1,187.4 154.3,194.3 155.6,191.4 156.9,183.0 158.2,174.0 159.4,176.0 160.7,182.4 162.0,174.6 163.3,166.5 164.6,161.6 165.8,161.8 167.1,162.1 168.4,156.1 169.7,143.7 170.9,101.8 172.2,77.6 173.5,69.2 174.8,66.0 176.0,65.1 177.3,63.2 178.6,60.3 179.8,59.0 181.1,57.1 182.4,59.3 183.7,58.4 184.9,63.4 186.2,63.0 187.5,67.5 188.8,66.5 190.1,66.7 191.3,70.7 192.6,67.3 193.9,69.9 195.2,67.1 196.4,66.4 197.7,68.3 199.0,63.9 200.2,65.6 201.5,60.3 202.8,60.6 204.1,57.2 205.3,57.2 206.6,57.9 207.9,60.9 209.2,61.7 210.4,66.1 211.7,67.6 213.0,66.8 214.3,71.6 215.6,84.4 216.8,106.8 218.1,158.2 219.4,187.4 220.7,179.9 221.9,178.3 223.2,184.5 224.5,187.3 225.8,179.6 227.0,178.7 228.3,178.1 229.6,183.4 230.8,175.9 232.1,175.5 233.4,185.8 234.7,175.4 235.9,171.3 237.2,153.0 238.5,103.7 239.8,85.5 241.1,77.0 242.3,72.7 243.6,76.1 244.9,77.7 246.2,76.0 247.4,78.5 248.7,76.9 250.0,72.8 251.2,74.1 252.5,68.5 253.8,67.9 255.1,64.8 256.4,64.1 257.6,63.8 258.9,64.5 260.2,64.5 261.4,64.4 262.7,66.7 264.0,64.4 265.3,68.0 266.6,67.8 267.8,73.4 269.1,73.5 270.4,77.5 271.6,78.5 272.9,75.6 274.2,76.7 275.5,76.4 276.8,73.2 278.0,74.4 279.3,74.9 280.6,74.9 281.9,72.9 283.1,77.9 284.4,78.2 285.7,75.0 286.9,77.5 288.2,72.5 289.5,73.1 290.8,67.3 292.1,68.2 293.3,64.2 294.6,66.2 295.9,64.2 297.1,64.6 298.4,66.0 299.7,76.3 301.0,92.8 302.2,142.5 303.5,181.4 304.8,181.3 306.1,177.8 307.4,163.5 308.6,158.3 309.9,155.3 311.2,157.7 312.4,160.2 313.7,164.6 315.0,170.9 316.3,157.9 317.6,133.7 318.8,119.4 320.1,113.0 321.4,118.7 322.6,125.8 323.9,135.1 325.2,127.1 326.5,111.7 327.8,103.7 329.0,106.0 330.3,113.3 331.6,110.6 332.9,101.8 334.1,97.1 335.4,86.0 336.7,77.4 337.9,75.6 339.2,74.6 340.5,73.8 341.8,75.8 343.1,79.7 344.3,82.4 345.6,82.8 346.9,86.7 348.1,84.4 349.4,87.0 350.7,85.7 352.0,89.0 353.2,85.7 354.5,84.9 355.8,88.6 357.1,87.9 358.4,90.3 359.6,90.3 360.9,86.7 362.2,90.0 363.4,85.7 364.7,89.0 366.0,85.0 367.3,83.8 368.6,85.9 369.8,81.1 371.1,83.3 372.4,79.9 373.6,79.8 374.9,74.6 376.2,76.2 377.5,75.7 378.8,75.2 380.0,77.1 381.3,89.1 382.6,106.1 383.9,126.4 385.1,145.4 386.4,140.5 387.7,137.2 388.9,139.1 390.2,135.4 391.5,133.1 392.8,130.8 394.1,135.8 395.3,135.3 396.6,131.1 397.9,115.9 399.1,107.4 400.4,112.8 401.7,119.2 403.0,130.9 404.2,143.3 405.5,138.8 406.8,136.4 408.1,133.2 409.4,130.9 410.6,131.6 411.9,139.9 413.2,123.0 414.4,102.4 415.7,86.5 417.0,78.8 418.3,77.7 419.6,75.6 420.8,76.9 422.1,77.3 423.4,75.0 424.6,76.4 425.9,76.7 427.2,79.5 428.5,84.1 429.8,85.2 431.0,80.7 432.3,79.9 433.6,78.0 434.9,76.1 436.1,72.9 437.4,72.1 438.7,71.1 439.9,71.4 441.2,71.1 442.5,72.0 443.8,70.6 445.1,70.9 446.3,72.1 447.6,76.7 448.9,80.2 450.1,81.3 451.4,83.5 452.7,82.2 454.0,79.0 455.2,77.8 456.5,78.5 457.8,80.5 459.1,82.1 460.4,81.5 461.6,79.5 462.9,81.0 464.2,82.4 465.4,78.5 466.7,74.9 468.0,72.1 469.3,71.5 470.6,69.8 471.8,71.9 473.1,70.9 474.4,74.3 475.6,81.6 476.9,105.0 478.2,125.1 479.5,115.5 480.8,117.0 482.0,127.6 483.3,141.8 484.6,136.4 485.9,129.1 487.1,131.8 488.4,137.3 489.7,137.3 490.9,141.6 492.2,140.5 493.5,134.6 494.8,124.4 496.1,118.7 497.3,122.2 498.6,121.7 499.9,113.1 501.1,107.6 502.4,114.5 503.7,117.4 505.0,115.9 506.2,112.7 507.5,109.7 508.8,110.6 510.1,119.2 511.4,118.6 512.6,101.9 513.9,97.9 515.2,101.9 516.5,110.2 517.7,128.6 519.0,131.1 520.3,121.4 521.5,112.6 522.8,116.5 524.1,120.7 525.4,122.3 526.6,120.2 527.9,126.7 529.2,135.6 530.5,143.8 531.8,140.5 533.0,138.6 534.3,131.0 535.6,129.5 536.9,131.6 538.1,137.3 539.4,135.2 540.7,132.8 542.0,132.8 543.2,131.7 544.5,134.1 545.8,137.0 547.0,133.4 548.3,134.1 549.6,134.3 550.9,135.7 552.1,119.2 553.4,112.5 554.7,109.3 556.0,121.8 557.2,122.7 558.5,133.2 559.8,137.0 561.1,126.6 562.4,127.1 563.6,120.2 564.9,109.3 566.2,103.1 567.5,109.0 568.7,110.5 570.0,107.8" class="stroke-a" fill="none"/>
<line x1="60" y1="129.0" x2="570" y2="129.0" class="stroke-ink dashed"/>
<polyline points="60.0,108.3 61.3,108.7 62.5,111.3 63.8,112.6 65.1,111.1 66.4,109.6 67.7,114.5 68.9,119.9 70.2,125.1 71.5,128.8 72.8,130.7 74.0,132.7 75.3,133.6 76.6,135.4 77.8,136.9 79.1,136.5 80.4,137.0 81.7,139.3 83.0,140.9 84.2,142.5 85.5,142.5 86.8,142.5 88.0,142.5 89.3,142.5 90.6,142.5 91.9,142.5 93.2,142.5 94.4,142.5 95.7,142.5 97.0,142.5 98.2,142.5 99.5,142.5 100.8,142.5 102.1,142.5 103.3,142.5 104.6,142.5 105.9,142.5 107.2,142.5 108.5,142.5 109.7,142.5 111.0,142.5 112.3,142.5 113.5,142.5 114.8,142.5 116.1,142.5 117.4,142.5 118.7,142.5 119.9,142.5 121.2,142.5 122.5,142.5 123.8,142.5 125.0,142.5 126.3,142.5 127.6,142.5 128.8,142.5 130.1,142.5 131.4,142.5 132.7,142.5 133.9,142.5 135.2,142.5 136.5,142.5 137.8,142.5 139.1,141.0 140.3,139.5 141.6,138.0 142.9,136.5 144.2,135.0 145.4,133.5 146.7,132.0 148.0,130.5 149.2,129.0 150.5,132.2 151.8,133.8 153.1,136.4 154.3,140.2 155.6,142.1 156.9,140.6 158.2,139.1 159.4,137.6 160.7,137.4 162.0,135.9 163.3,134.4 164.6,132.9 165.8,131.4 167.1,129.9 168.4,128.4 169.7,128.4 170.9,128.4 172.2,128.4 173.5,128.4 174.8,128.4 176.0,128.4 177.3,128.4 178.6,128.4 179.8,128.4 181.1,128.4 182.4,128.4 183.7,128.4 184.9,128.4 186.2,128.4 187.5,128.4 188.8,128.4 190.1,128.4 191.3,128.4 192.6,128.4 193.9,128.4 195.2,128.4 196.4,128.4 197.7,128.4 199.0,128.4 200.2,128.4 201.5,128.4 202.8,128.4 204.1,128.4 205.3,128.4 206.6,128.4 207.9,128.4 209.2,128.4 210.4,128.4 211.7,128.4 213.0,128.4 214.3,128.4 215.6,128.4 216.8,128.4 218.1,126.9 219.4,131.5 220.7,132.5 221.9,132.7 223.2,134.8 224.5,137.0 225.8,135.5 227.0,134.0 228.3,133.1 229.6,134.7 230.8,133.2 232.1,131.7 233.4,134.4 234.7,132.9 235.9,131.4 237.2,129.9 238.5,129.9 239.8,129.9 241.1,129.9 242.3,129.9 243.6,129.9 244.9,129.9 246.2,129.9 247.4,129.9 248.7,129.9 250.0,129.9 251.2,129.9 252.5,129.9 253.8,129.9 255.1,129.9 256.4,129.9 257.6,129.9 258.9,129.9 260.2,129.9 261.4,129.9 262.7,129.9 264.0,129.9 265.3,129.9 266.6,129.9 267.8,129.9 269.1,129.9 270.4,129.9 271.6,129.9 272.9,129.9 274.2,129.9 275.5,129.9 276.8,129.9 278.0,129.9 279.3,129.9 280.6,129.9 281.9,129.9 283.1,129.9 284.4,129.9 285.7,129.9 286.9,129.9 288.2,129.9 289.5,129.9 290.8,129.9 292.1,129.9 293.3,129.9 294.6,129.9 295.9,129.9 297.1,129.9 298.4,129.9 299.7,129.9 301.0,128.4 302.2,126.9 303.5,129.8 304.8,131.7 306.1,132.1 307.4,130.6 308.6,129.1 309.9,127.6 311.2,126.1 312.4,124.6 313.7,123.1 315.0,123.9 316.3,122.4 317.6,122.4 318.8,122.4 320.1,122.4 321.4,122.4 322.6,122.4 323.9,122.4 325.2,122.4 326.5,122.4 327.8,122.4 329.0,122.4 330.3,122.4 331.6,122.4 332.9,122.4 334.1,122.4 335.4,122.4 336.7,122.4 337.9,122.4 339.2,122.4 340.5,122.4 341.8,122.4 343.1,122.4 344.3,122.4 345.6,122.4 346.9,122.4 348.1,122.4 349.4,122.4 350.7,122.4 352.0,122.4 353.2,122.4 354.5,122.4 355.8,122.4 357.1,122.4 358.4,122.4 359.6,122.4 360.9,122.4 362.2,122.4 363.4,122.4 364.7,122.4 366.0,122.4 367.3,122.4 368.6,122.4 369.8,122.4 371.1,122.4 372.4,122.4 373.6,122.4 374.9,122.4 376.2,122.4 377.5,122.4 378.8,122.4 380.0,122.4 381.3,122.4 382.6,120.9 383.9,119.4 385.1,117.9 386.4,116.4 387.7,114.9 388.9,113.4 390.2,111.9 391.5,110.4 392.8,108.9 394.1,107.4 395.3,105.9 396.6,104.4 397.9,102.9 399.1,101.4 400.4,99.9 401.7,98.4 403.0,96.9 404.2,97.3 405.5,95.8 406.8,94.3 408.1,92.8 409.4,91.3 410.6,89.8 411.9,91.4 413.2,89.9 414.4,88.4 415.7,88.4 417.0,88.4 418.3,88.4 419.6,88.4 420.8,88.4 422.1,88.4 423.4,88.4 424.6,88.4 425.9,88.4 427.2,88.4 428.5,88.4 429.8,88.4 431.0,88.4 432.3,88.4 433.6,88.4 434.9,88.4 436.1,88.4 437.4,88.4 438.7,88.4 439.9,88.4 441.2,88.4 442.5,88.4 443.8,88.4 445.1,88.4 446.3,88.4 447.6,88.4 448.9,88.4 450.1,88.4 451.4,88.4 452.7,88.4 454.0,88.4 455.2,88.4 456.5,88.4 457.8,88.4 459.1,88.4 460.4,88.4 461.6,88.4 462.9,88.4 464.2,88.4 465.4,88.4 466.7,88.4 468.0,88.4 469.3,88.4 470.6,88.4 471.8,88.4 473.1,88.4 474.4,88.4 475.6,86.9 476.9,85.4 478.2,83.9 479.5,82.4 480.8,80.9 482.0,81.4 483.3,86.0 484.6,87.6 485.9,86.1 487.1,86.3 488.4,88.1 489.7,89.4 490.9,91.5 492.2,92.7 493.5,91.2 494.8,89.7 496.1,88.2 497.3,86.7 498.6,85.2 499.9,83.7 501.1,82.2 502.4,80.7 503.7,79.2 505.0,77.7 506.2,76.2 507.5,74.7 508.8,73.2 510.1,73.5 511.4,73.5 512.6,72.0 513.9,70.5 515.2,69.0 516.5,67.5 517.7,72.4 519.0,76.5 520.3,76.4 521.5,74.9 522.8,73.4 524.1,74.1 525.4,75.0 526.6,75.1 527.9,77.0 529.2,81.1 530.5,86.4 531.8,89.1 533.0,90.5 534.3,89.0 535.6,87.5 536.9,86.6 538.1,88.3 539.4,88.9 540.7,87.8 542.0,87.8 543.2,86.7 544.5,87.4 545.8,88.8 547.0,88.4 548.3,88.6 549.6,88.8 550.9,89.4 552.1,87.9 553.4,86.4 554.7,84.9 556.0,83.4 557.2,81.9 558.5,83.8 559.8,86.3 561.1,84.8 562.4,83.3 563.6,81.8 564.9,80.3 566.2,78.8 567.5,77.3 568.7,75.8 570.0,74.3" class="stroke-b dashed" fill="none"/>
<line x1="315.0" y1="30" x2="315.0" y2="230" class="stroke-muted dashed"/>
<text x="311.0" y="44" class="fig-small" text-anchor="end">noise rises 10 dB</text>
<text x="68" y="124.0" class="fig-small" text-anchor="start">fixed on threshold</text>
<text x="378.8" y="117.4" class="fig-small" text-anchor="start">tracked floor + 9 dB</text>
<text x="56" y="264" class="fig-small" text-anchor="end">truth</text>
<rect x="85.5" y="250" width="52.3" height="20" class="fill-soft-c"/>
<rect x="170.9" y="250" width="47.2" height="20" class="fill-soft-c"/>
<rect x="238.5" y="250" width="63.8" height="20" class="fill-soft-c"/>
<rect x="335.4" y="250" width="48.5" height="20" class="fill-soft-c"/>
<rect x="414.4" y="250" width="63.8" height="20" class="fill-soft-c"/>
<text x="56" y="290" class="fig-small" text-anchor="end">fixed</text>
<rect x="85.5" y="276" width="59.9" height="20" class="fill-d"/>
<rect x="170.9" y="276" width="53.5" height="20" class="fill-d"/>
<rect x="238.5" y="276" width="71.4" height="20" class="fill-d"/>
<rect x="318.8" y="276" width="252.4" height="20" class="fill-d"/>
<text x="56" y="316" class="fig-small" text-anchor="end">tracked</text>
<rect x="85.5" y="302" width="59.9" height="20" class="fill-b"/>
<rect x="170.9" y="302" width="53.5" height="20" class="fill-b"/>
<rect x="238.5" y="302" width="71.4" height="20" class="fill-b"/>
<rect x="318.8" y="302" width="72.7" height="20" class="fill-b"/>
<rect x="415.7" y="302" width="68.8" height="20" class="fill-b"/>
<line x1="60" y1="326" x2="570" y2="326" class="stroke-muted"/>
<line x1="60.0" y1="326" x2="60.0" y2="330" class="stroke-muted"/>
<text x="60.0" y="342" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="326" x2="187.5" y2="330" class="stroke-muted"/>
<text x="187.5" y="342" class="fig-small" text-anchor="middle">1</text>
<line x1="315.0" y1="326" x2="315.0" y2="330" class="stroke-muted"/>
<text x="315.0" y="342" class="fig-small" text-anchor="middle">2</text>
<line x1="442.5" y1="326" x2="442.5" y2="330" class="stroke-muted"/>
<text x="442.5" y="342" class="fig-small" text-anchor="middle">3</text>
<line x1="570.0" y1="326" x2="570.0" y2="330" class="stroke-muted"/>
<text x="570.0" y="342" class="fig-small" text-anchor="middle">4</text>
<text x="315" y="359" text-anchor="middle">time (s)</text>
</svg>
Figure 3. A clip with the noise rising 10 dB at 2 s. The fixed on threshold (dark dashed line) sits below the new noise, so the fixed detector stays on after 2 s. The tracked threshold (orange dashed line) rises with the floor, from -38.4 dB to -27.9 dB, and the tracked detector follows the speech again. Frame F1 is 0.77 for the fixed floor and 0.91 for the tracked floor.
```

The cure is a floor that follows the background. Let $e_t$ be the energy of frame $t$ in decibels, $b_t$ the floor, $u$ the largest rise per frame in decibels (`up_db_per_frame`), and $\alpha$ the fraction of a drop to follow (`down_alpha`). Start with $b_0 = e_0$ and update:

$$
b_t =
\begin{cases}
b_{t-1} & \text{if frame } t \text{ is speech} \\
b_{t-1} + \alpha\,(e_t - b_{t-1}) & \text{if } e_t < b_{t-1} \\
b_{t-1} + \min(e_t - b_{t-1},\, u) & \text{otherwise}
\end{cases}
$$

Read it aloud: on speech, do nothing. Below the floor, move a fraction $\alpha$ of the way down, which is fast. Above it, move up by at most $u$ dB, which is slow. The freeze keeps speech out: at $u = 0.3$ dB per frame, a half-second word would lift an unfrozen floor by 15 dB.

Take $u = 1$, $\alpha = 0.5$, energies $[-40, -40, -20, -20, -40]$ dB, and the two loud frames marked as speech. The floor stays at $-40$. With no mask, it reads $[-40, -40, -39, -38, -39]$.

Which frames are speech? The detector's own output cannot say: a step bigger than the margin looks like speech, which freezes the floor, which keeps the step looking like speech. `adaptive_vad` breaks the loop with a second opinion that cannot lock, the minimum energy over the last 50 frames, which is 0.5 s. This borrows from Martin's (2001) minimum statistics, which tracks a window minimum because speech has pauses. A frame more than the margin above that minimum freezes the floor, and after a step the minimum catches up within 0.5 s. The detector then runs `hysteresis` on $e_t - b_t$, with margins of 9 dB and 5 dB.

## Compare likelihoods, not just loudness

A slammed door fools a loudness cue. *Spectral flatness* is a second cue: the geometric mean of the power spectrum over its arithmetic mean, in decibels. Noise is flat, near 0 dB. Voiced speech piles power into harmonics with deep valleys between them, so its flatness is far lower. In this chapter's 5 dB clips, noise averages $-4.3$ dB and speech $-11.6$ dB.

Sohn, Kim, and Sung (1999) turned cues like these into a test. Model each frame's features, here energy and flatness, as drawn from one of two Gaussians: speech, with mean $\mu_{s,d}$ and variance $\sigma^2_{s,d}$ for feature $d$, or noise, with $\mu_{n,d}$ and $\sigma^2_{n,d}$. The features are independent, so the log-likelihood ratio of frame $t$ is a sum:

$$
\Lambda_t = \sum_{d} \left[ -\frac{1}{2}\ln\frac{\sigma^2_{s,d}}{\sigma^2_{n,d}} - \frac{(x_{t,d} - \mu_{s,d})^2}{2\sigma^2_{s,d}} + \frac{(x_{t,d} - \mu_{n,d})^2}{2\sigma^2_{n,d}} \right]
$$

Read it aloud: per feature, measure the squared distance to each mean in units of that Gaussian's variance, and correct for one Gaussian being wider. A positive $\Lambda_t$, in nats, means speech is more likely. The log term is not optional, because a wide Gaussian spreads its probability thin. With $x = 0$, both means at 0, and variances of 4 and 1, the distance terms vanish and $\Lambda = -\tfrac12 \ln 4 = -0.693$: the narrow noise Gaussian wins. A version that forgets the log term gets 0, a tie.

To decide, average $\Lambda_t$ over 5 frames and run the same `hysteresis` with on $+1$, off $-1$, and a hangover of 5, after fitting the means and variances on labeled clips. At 5 dB SNR, energy alone scores a frame F1 of 0.84, and adding flatness raises it to 0.95, against 0.77 for the percentile energy detector.

WebRTC's classic VAD (`common_audio/vad/vad_core.c`) works this way, with a two-component Gaussian mixture per sub-band for speech and for noise, over six bands. It updates its noise means only on frames it judges to be noise. Modern systems use small neural models, such as Silero VAD, which outputs a speech probability for each 32 ms chunk, with thresholds of 0.5 to start and 0.35 to end.

## Decide that the speaker has stopped

A voice assistant must not answer while you pause to breathe or think, so it needs a decision beyond the VAD's: that the utterance is over. This is *endpointing*. The simplest endpointer is a trailing-silence timeout on the VAD output: after the first speech frame, count consecutive non-speech frames, reset the count at each speech frame, and declare the end when the count exceeds the timeout.

Take the VAD output $[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]$ with a timeout of 2 frames. The pause at frames 4 and 5 does not exceed 2, and speech resumes at frame 6, so the count resets. Frames 7, 8, and 9 make it 3, so `endpoint` returns 9. It is the hangover idea with a different job: the hangover cleans the edges of a stretch, and the timeout triggers an action.

A short timeout answers quickly but cuts off people who pause mid-thought. A long one is safe but makes every turn feel slow. A later lesson covers keyword spotting and endpointing in depth.

> [!KEY]
> You now know how to find the speech in a recording and when it ends: compare each frame with a floor that follows the background, weigh cues by likelihood, switch on high and off low, and end the utterance after a timeout of silence.

> [!RECAP]
> - Frame energy is $10 \log_{10}$ of the mean power, and thresholds sit above a noise floor, so the gain does not matter.
> - Hysteresis switches on high and off low, and the hangover keeps the end of a word.
> - A tracked floor falls fast, rises slowly, and freezes on speech.
> - A log-likelihood ratio of two Gaussians combines cues, including its log-variance term.
> - An endpoint is where non-speech frames after speech exceed a timeout.

## Build it step by step

The checks test the functions in this order.

1. **`frame_energy_db(x, frame_length, hop)`** pads by reflection, frames the signal by broadcasting, and averages the squares. The usual mistake is $20 \log_{10}$ of a power, which doubles every value.
2. **`hysteresis(energy_db, on_db, off_db, hangover)`** loops once with two state variables. The usual mistake is forgetting to reset the quiet count at or above the off threshold.
3. **`frame_f1(pred, ref)`** counts TP, FP, and FN with boolean masks. Return 1.0 when the denominator is 0.
4. **`energy_vad(x, sample_rate, ...)`** converts milliseconds to samples, finds the floor with `np.percentile`, and calls `hysteresis`.
5. **`track_noise_floor(energy_db, up_db_per_frame, down_alpha, speech_mask=None)`** loops once with the three-case rule. The usual mistake is updating on speech frames too, so the floor climbs into the speech.
6. **`adaptive_vad(energy_db, margin_db, ...)`** masks frames above the running minimum by the margin, tracks the floor, and runs `hysteresis` on `energy_db - floor`. It returns both.
7. **`gaussian_llr(features, speech_mean, speech_var, noise_mean, noise_var)`** subtracts summed log densities. The usual mistake is dropping the $-\tfrac12 \ln \sigma^2$ term. `frame_features` and `fit_gaussians` are provided.
8. **`endpoint(vad, hangover_frames)`** loops once with a started flag and a quiet count. The usual mistake is not resetting the count when speech resumes.

## Where this breaks

**Low SNR.** At 5 dB, quiet speech sits under the energy detector's on threshold, and the frame F1 falls to 0.77. The likelihood ratio recovers it only because the fit comes from similar clips.

**Loud noise that is not speech.** A door slam or music is loud, and energy cannot tell it from speech. You need spectral shape, periodicity, or a trained model.

**Quiet speech sounds.** Fricatives such as "f" and "th" carry little energy. A hangover protects word endings, and a zero-crossing rate catches noisy sounds that energy misses.

**A floor that lags.** Following a step takes about 0.5 s plus the climb. A recording with no pauses defeats the percentile and the running minimum.

## The exercise

You build `frame_energy_db`, `hysteresis`, `energy_vad`, `frame_f1`, `track_noise_floor`, `adaptive_vad`, `gaussian_llr`, and `endpoint`. Press **Run** to see the energy detector at three SNRs, the tracked floor on a noise step, and the likelihood-ratio detector.

The first five checks cover the energy detector and gain. The last five work the floor and `endpoint` by hand, require `adaptive_vad` to reach 0.85 on a noise step, compare `gaussian_llr` with `scipy.stats.norm.logpdf`, and require 0.9 from the likelihood-ratio detector at 5 dB.

## Further reading

- Sohn, Kim, and Sung, "A statistical model-based voice activity detection," IEEE Signal Processing Letters, 1999 (https://doi.org/10.1109/97.736233).
- Martin, "Noise power spectral density estimation based on optimal smoothing and minimum statistics," IEEE Transactions on Speech and Audio Processing, 2001 (https://doi.org/10.1109/89.928915).
- WebRTC, `common_audio/vad/vad_core.c`, the two-Gaussian-per-band VAD (https://webrtc.googlesource.com/src/+/refs/heads/main/common_audio/vad/vad_core.c).
