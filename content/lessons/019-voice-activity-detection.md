---
title: Voice activity detection
slug: 019-voice-activity-detection
part: "Part 2 — Hearing and speech production"
summary: Mark which frames of a recording hold speech by comparing loudness with the noise floor, switching with two thresholds, and scoring by frame F1.
minutes: 35
gpu: false
objectives:
  - Compute the energy of each frame in decibels, and explain why a threshold must sit relative to the recording's own noise floor.
  - Explain why a single threshold flickers, and run hysteresis with a hangover by hand.
  - Build an energy voice activity detector from frame energy, a percentile floor, and hysteresis.
  - Score a detector's decisions by frame F1, and say where an energy detector fails.
prereqs:
  - 001-what-sound-is
  - 007-the-short-time-fourier-transform
exercise: 019-voice-activity-detection
---

# Voice activity detection

A recording of a meeting is mostly not speech. There are pauses, a fan, and a door. A recognizer that transcribes all of it spends compute on silence and can invent words from the noise. A phone call that sends all of it wastes bandwidth. Both want a *voice activity detector* (VAD): a program that marks which stretches of a recording hold speech.

The cue you reach for first is loudness, because speech is louder than the room around it. So how loud is loud enough? A fixed number fails the moment the microphone changes, and a single threshold chatters on and off whenever the energy hovers near it.

This chapter builds a detector from frame energy in three small steps, each fixing the failure of the one before. Figure 1 shows the finished detector on a noisy clip.

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
- A centered frame is padded by half its length at each end by reflection, so a signal of $L$ samples has $1 + \lfloor L / H \rfloor$ frames at hop $H$.
- Speech is built from bursts of voiced sound, with silence or background noise between them.
- A percentile, `np.percentile(v, 10)`, is the value that 10% of the numbers fall below.

## Measure the loudness of each frame

Slide a frame along the recording, as in the short-time Fourier transform, and ask one question per frame: how much power does it carry? The *frame energy* in decibels is:

$$
E_t = 10 \log_{10}\left(\frac{1}{N} \sum_{n=0}^{N-1} x_t[n]^2 + 10^{-10}\right)
$$

Read it aloud: square each of the $N$ samples in frame $t$, average them to get the power, add a tiny constant so that silence gives $-100$ dB instead of minus infinity, and convert to decibels. The unit is decibels relative to a full-scale sample of 1, and every value is negative for a signal inside $[-1, 1]$. A sine of amplitude 0.5 has a mean power of 0.125, which is $-9.03$ dB.

```python
energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
```

## Why a fixed threshold fails

The obvious detector says speech is any frame above a fixed level, such as $-30$ dB. On the clip in Figure 1, that works: it gives a frame F1 of 0.87, a score defined below. But the energies depend on the microphone gain. Record the same clip 20 dB quieter, which means scaling the samples by 0.1, and no frame reaches $-30$ dB, so the detector finds no speech at all. Lower the threshold to $-40$ dB to catch the quiet recording, and the original clip becomes speech in every frame, because its room noise alone sits at about $-32$ dB. Even then the quiet recording scores a frame F1 of only 0.56.

So the threshold cannot be a number in decibels. It must be a number *above the noise floor*, the energy of the recording when nobody speaks. You do not know when nobody speaks, but you can estimate the floor as the 10th percentile of the frame energies. That works when at least a tenth of the recording is quiet. Speech is then any frame some margin above the floor, and scaling the recording moves the energies and the floor together, so the decisions do not change. On the same clip, the floor-relative detector gives the same decisions at gains of 1, 0.1, and 0.01.

> [!KEY]
> Loudness depends on the microphone, so compare each frame with the recording's own noise floor. A margin above the floor is a gain-free threshold.

## Why a single threshold flickers

Even a floor-relative threshold makes mistakes at the edges of words. Energy rises and falls inside a word, and room noise bumps up and down between words. Wherever the energy hovers near the one threshold, the decision flips on and off. On the clip in Figure 1, a single threshold at the floor plus 9 dB switches 10 times, which is five stretches, where the truth has three.

*Hysteresis* uses two thresholds. Switch on only above a high one, and switch off only below a lower one. In between, keep whatever state you were in. A *hangover* adds a delay: stay on until the energy has been below the off threshold for more than a few frames, so a quiet consonant at the end of a word is not cut off. The exercise switches on at 9 dB above the floor and off at 5 dB above it, with a hangover of 50 ms, which is 5 frames at a 10 ms hop. Real speech, with its weak final consonants, usually needs 100 ms to 300 ms.

Run it by hand. Take the energies $[0, 12, 8, 3, 3, 3, 9, 2, 2]$ dB, an on threshold of 10, an off threshold of 5, and a hangover of 2 frames.

1. Frame 0 is 0 dB, not above 10, so the state is silence.
2. Frame 1 is 12 dB, above 10, so speech starts.
3. Frame 2 is 8 dB. That is below 10 but not below 5, so speech continues.
4. Frames 3 and 4 are below 5, so the quiet count reaches 1, then 2. That is not more than the hangover, so speech continues.
5. Frame 5 makes the count 3, which is more than 2, so speech ends.
6. Frame 6 is 9 dB, above the off threshold but not the on threshold, so silence continues. Frames 7 and 8 stay silent.

The decisions are $[0, 1, 1, 1, 1, 0, 0, 0, 0]$, as in Figure 2. A single threshold at 10 cuts speech at frame 2, and one at 5 switches speech back on at frame 6.

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

The whole state is two variables, updated in one loop over the frames:

```python
speech, quiet = False, 0   # on or off, and the number of quiet frames in a row
```

> [!KEY]
> Two thresholds and a hangover turn a flickering signal into stretches. Hysteresis switches on high and off low, and the hangover protects the quiet end of a word.

## Score the decisions

Treat speech frames as the positive class. A *true positive* (TP) is a speech frame marked speech, a *false positive* (FP) is a non-speech frame marked speech, and a *false negative* (FN) is a speech frame that the detector missed. The frame F1 score combines precision and recall:

$$
F_1 = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
$$

Read it aloud: twice the hits, over twice the hits plus both kinds of mistake. It is 1 for a perfect detector and 0 for one that finds no speech. For decisions $[1, 1, 0, 0, 1, 0]$ against the truth $[1, 0, 0, 1, 1, 0]$, TP = 2, FP = 1, and FN = 1, so $F_1 = 4 / 6 = 0.667$.

Regenerate the clip of Figure 1, three voiced bursts in pink noise, at other signal-to-noise ratios, and the detector reaches a frame F1 of 0.92 at 20 dB signal-to-noise ratio and 0.94 at 10 dB. At 5 dB it falls to 0.77, because quiet speech sinks into the noise floor.

Two widely used detectors do better than energy alone. WebRTC's VAD, from the browser real-time audio stack, classifies frames of 10, 20, or 30 ms with Gaussian mixture models over the energies of six bands between 80 Hz and 4 kHz. Silero VAD is a small neural network that outputs a speech probability for each 512-sample chunk at 16 kHz, which is 32 ms. It also uses two thresholds: speech starts above 0.5 by default, and it ends below 0.35, which is the same hysteresis idea applied to a probability.

> [!KEY]
> You now know how to find the speech in a recording from loudness alone. Compare each frame's energy with the recording's noise floor, switch on high and off low with a hangover, and score the result by frame F1.

> [!RECAP]
> - Frame energy is $10 \log_{10}$ of the mean power, in decibels.
> - Set thresholds above a percentile noise floor, so the decisions do not depend on the microphone gain.
> - Hysteresis switches on high and off low, and the hangover keeps the end of a word.
> - Frame F1 is $2\text{TP} / (2\text{TP} + \text{FP} + \text{FN})$.

## Build it step by step

The checks test the functions in this order.

1. **`frame_energy_db(x, frame_length, hop)`** pads by half a frame with reflection, frames the signal by broadcasting, and averages the squares. The usual mistake is $20 \log_{10}$ of a power, which doubles every value.
2. **`hysteresis(energy_db, on_db, off_db, hangover)`** loops once over the frames with two state variables, the state and the count of quiet frames. The usual mistake is forgetting to reset the count when a frame is at or above the off threshold.
3. **`frame_f1(pred, ref)`** counts TP, FP, and FN with boolean masks. Return 1.0 when the denominator is 0, because neither track has speech.
4. **`energy_vad(x, sample_rate, ...)`** converts milliseconds to samples and frames, finds the floor with `np.percentile`, and calls `hysteresis` with thresholds above the floor.

## Where this breaks

**Low SNR.** At 5 dB, quiet speech sits under the on threshold, and the frame F1 falls to 0.77.

**Noise that changes.** A door slam or music is loud but not speech. Energy cannot tell them apart, so you need spectral shape, periodicity, or a trained model.

**Quiet speech sounds.** Fricatives such as "f" and "th" carry little energy. A hangover protects word endings, and a zero-crossing rate, the rate at which the signal changes sign, catches noisy sounds that energy misses.

**A recording with no silence.** The 10th percentile then sits inside the speech, and the thresholds sit too high. A live VAD tracks the floor with a slowly updated minimum instead.

## The exercise

You build `frame_energy_db`, `hysteresis`, `energy_vad`, and `frame_f1`. Press **Run** to hear and see the detector on noisy synthetic speech at three signal-to-noise ratios.

The checks verify frame energy against its formula, including a sine and silence. They work the hysteresis example with and without a hangover, compute F1 by hand, and require a frame F1 of at least 0.9 at 20 dB and 10 dB SNR on three noisy clips. They also scale a recording by 0.1 and 0.3 and require almost the same decisions.
