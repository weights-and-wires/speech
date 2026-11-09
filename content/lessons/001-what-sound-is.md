---
title: What sound is
slug: 001-what-sound-is
part: "Part 1 — Sound and signals"
summary: Describe a sound as an array of numbers, then give it one loudness number with RMS and decibels relative to full scale (dBFS).
minutes: 35
gpu: false
objectives:
  - Describe a tone by its frequency, amplitude, and phase, and store it as an array of samples.
  - Compute the root mean square (RMS) of a signal by hand, and explain why it tracks loudness better than the peak does.
  - "Convert an amplitude to dBFS with 20 log10, and explain why halving an amplitude always lowers the level by 6.02 dB."
  - Choose between 20 log10 for amplitudes and 10 log10 for powers.
exercise: 001-what-sound-is
---

# What sound is

You record the same sentence twice. In the second take you stand farther from the microphone, and the file sounds quieter. A program that has to decide whether anyone is talking, or that has to turn every clip in a dataset to the same loudness, needs a number that says how loud each clip is. Which number?

The answer starts with what a recording is. A microphone turns air pressure into a voltage, and a converter measures that voltage thousands of times a second. Your code sees a list of numbers, and every system in this course starts from such a list.

This chapter describes a pure tone as an array, tries two obvious loudness numbers, shows why one of them fails, and puts the winner on the scale that audio people use: decibels.

## What you already know

- A sine or cosine wave repeats itself, and the number of repeats per second is its frequency.
- A logarithm turns multiplication into addition: $\log_{10}(100) = 2$.
- numpy applies arithmetic to a whole array at once, so `np.cos(x)` works on every element.
- A mean adds the values and divides by how many there are.

## How the exercises work

Every chapter has one exercise that you write yourself. **Run** executes the `if __name__ == "__main__":` block at the bottom of your file, and a clip it plays appears as an audio player in the console. **Check** runs the tests against your functions, one line per check.

The first five chapters share one running example: a pure 440 Hz tone, the A above middle C, at an amplitude of 0.5. You hear it, measure it, sample it, quantize it, and take it apart into frequencies.

## Sound is pressure that changes over time

Sound is a wave of pressure that moves through air at about 343 meters per second. At one point in space, the pressure goes up and down, and a microphone follows it. The signal you record is a curve: how far the pressure is from rest, at each moment.

The simplest sound is a **pure tone**, in which that curve is a cosine. Figure 1 shows our running example. It has three properties:

- **Frequency** $f$, in hertz (Hz), is how many cycles it completes per second. People hear from about 20 Hz to 20 kHz. Speech carries most of its meaning below 8 kHz.
- **Amplitude** $A$ is how far the curve swings from zero. A digital signal has no physical unit, and full scale is 1.0.
- **Phase** $\varphi$, in radians, is where in its cycle the curve starts.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="A 440 hertz tone of amplitude 0.5 drawn as a smooth curve, with one orange dot for each sample taken at 16 kilohertz, with its peak and RMS levels marked">
<line x1="64" y1="218.5" x2="576" y2="218.5" class="stroke-grid"/>
<text x="58" y="222.5" class="fig-small" text-anchor="end">-1</text>
<line x1="64" y1="170.7" x2="576" y2="170.7" class="stroke-grid"/>
<text x="58" y="174.7" class="fig-small" text-anchor="end">-0.5</text>
<line x1="64" y1="123.0" x2="576" y2="123.0" class="stroke-grid"/>
<text x="58" y="127.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="75.3" x2="576" y2="75.3" class="stroke-grid"/>
<text x="58" y="79.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="64" y1="27.5" x2="576" y2="27.5" class="stroke-grid"/>
<text x="58" y="31.5" class="fig-small" text-anchor="end">1</text>
<line x1="64.0" y1="228" x2="64.0" y2="232" class="stroke-muted"/>
<text x="64.0" y="245" class="fig-small" text-anchor="middle">0</text>
<line x1="166.4" y1="228" x2="166.4" y2="232" class="stroke-muted"/>
<text x="166.4" y="245" class="fig-small" text-anchor="middle">1</text>
<line x1="268.8" y1="228" x2="268.8" y2="232" class="stroke-muted"/>
<text x="268.8" y="245" class="fig-small" text-anchor="middle">2</text>
<line x1="371.2" y1="228" x2="371.2" y2="232" class="stroke-muted"/>
<text x="371.2" y="245" class="fig-small" text-anchor="middle">3</text>
<line x1="473.6" y1="228" x2="473.6" y2="232" class="stroke-muted"/>
<text x="473.6" y="245" class="fig-small" text-anchor="middle">4</text>
<line x1="576.0" y1="228" x2="576.0" y2="232" class="stroke-muted"/>
<text x="576.0" y="245" class="fig-small" text-anchor="middle">5</text>
<line x1="64" y1="228" x2="576" y2="228" class="stroke-muted"/>
<line x1="64" y1="18" x2="64" y2="228" class="stroke-muted"/>
<text x="320.0" y="272" class="fig-small" text-anchor="middle">time (ms)</text>
<text x="14" y="123.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 123.0)">amplitude (full scale = 1)</text>
<polyline points="64.0,75.3 65.3,75.3 66.6,75.4 67.8,75.5 69.1,75.7 70.4,76.0 71.7,76.3 73.0,76.7 74.3,77.1 75.5,77.6 76.8,78.1 78.1,78.7 79.4,79.3 80.7,80.0 82.0,80.8 83.2,81.6 84.5,82.4 85.8,83.3 87.1,84.3 88.4,85.2 89.7,86.3 90.9,87.4 92.2,88.5 93.5,89.6 94.8,90.8 96.1,92.1 97.4,93.4 98.6,94.7 99.9,96.0 101.2,97.4 102.5,98.8 103.8,100.3 105.1,101.7 106.3,103.2 107.6,104.7 108.9,106.3 110.2,107.8 111.5,109.4 112.8,111.0 114.0,112.6 115.3,114.2 116.6,115.8 117.9,117.5 119.2,119.1 120.5,120.8 121.7,122.4 123.0,124.1 124.3,125.7 125.6,127.4 126.9,129.0 128.2,130.7 129.4,132.3 130.7,133.9 132.0,135.5 133.3,137.1 134.6,138.7 135.9,140.2 137.1,141.8 138.4,143.3 139.7,144.8 141.0,146.2 142.3,147.6 143.6,149.0 144.8,150.4 146.1,151.7 147.4,153.1 148.7,154.3 150.0,155.5 151.3,156.7 152.5,157.9 153.8,159.0 155.1,160.1 156.4,161.1 157.7,162.0 159.0,163.0 160.2,163.9 161.5,164.7 162.8,165.5 164.1,166.2 165.4,166.9 166.7,167.5 167.9,168.1 169.2,168.6 170.5,169.0 171.8,169.5 173.1,169.8 174.4,170.1 175.6,170.3 176.9,170.5 178.2,170.6 179.5,170.7 180.8,170.7 182.1,170.7 183.3,170.6 184.6,170.4 185.9,170.2 187.2,169.9 188.5,169.6 189.8,169.2 191.0,168.8 192.3,168.3 193.6,167.7 194.9,167.1 196.2,166.4 197.5,165.7 198.7,165.0 200.0,164.2 201.3,163.3 202.6,162.4 203.9,161.4 205.2,160.4 206.4,159.4 207.7,158.3 209.0,157.2 210.3,156.0 211.6,154.8 212.9,153.5 214.1,152.2 215.4,150.9 216.7,149.5 218.0,148.2 219.3,146.7 220.6,145.3 221.8,143.8 223.1,142.3 224.4,140.8 225.7,139.2 227.0,137.7 228.3,136.1 229.5,134.5 230.8,132.9 232.1,131.3 233.4,129.6 234.7,128.0 235.9,126.3 237.2,124.7 238.5,123.0 239.8,121.4 241.1,119.7 242.4,118.1 243.6,116.4 244.9,114.8 246.2,113.2 247.5,111.6 248.8,110.0 250.1,108.4 251.3,106.8 252.6,105.3 253.9,103.8 255.2,102.3 256.5,100.8 257.8,99.3 259.0,97.9 260.3,96.5 261.6,95.2 262.9,93.8 264.2,92.5 265.5,91.3 266.7,90.1 268.0,88.9 269.3,87.8 270.6,86.7 271.9,85.6 273.2,84.6 274.4,83.7 275.7,82.7 277.0,81.9 278.3,81.1 279.6,80.3 280.9,79.6 282.1,78.9 283.4,78.3 284.7,77.8 286.0,77.3 287.3,76.8 288.6,76.4 289.8,76.1 291.1,75.8 292.4,75.6 293.7,75.4 295.0,75.3 296.3,75.3 297.5,75.3 298.8,75.3 300.1,75.5 301.4,75.7 302.7,75.9 304.0,76.2 305.2,76.5 306.5,76.9 307.8,77.4 309.1,77.9 310.4,78.5 311.7,79.1 312.9,79.8 314.2,80.5 315.5,81.3 316.8,82.1 318.1,83.0 319.4,83.9 320.6,84.9 321.9,85.9 323.2,87.0 324.5,88.1 325.8,89.2 327.1,90.4 328.3,91.6 329.6,92.9 330.9,94.2 332.2,95.5 333.5,96.9 334.8,98.3 336.0,99.7 337.3,101.2 338.6,102.7 339.9,104.2 341.2,105.7 342.5,107.3 343.7,108.8 345.0,110.4 346.3,112.0 347.6,113.6 348.9,115.3 350.2,116.9 351.4,118.5 352.7,120.2 354.0,121.8 355.3,123.5 356.6,125.1 357.9,126.8 359.1,128.4 360.4,130.1 361.7,131.7 363.0,133.3 364.3,134.9 365.6,136.5 366.8,138.1 368.1,139.7 369.4,141.2 370.7,142.7 372.0,144.2 373.3,145.7 374.5,147.1 375.8,148.5 377.1,149.9 378.4,151.3 379.7,152.6 381.0,153.9 382.2,155.1 383.5,156.3 384.8,157.5 386.1,158.6 387.4,159.7 388.7,160.7 389.9,161.7 391.2,162.6 392.5,163.5 393.8,164.4 395.1,165.2 396.4,165.9 397.6,166.6 398.9,167.3 400.2,167.9 401.5,168.4 402.8,168.9 404.1,169.3 405.3,169.7 406.6,170.0 407.9,170.3 409.2,170.5 410.5,170.6 411.7,170.7 413.0,170.7 414.3,170.7 415.6,170.6 416.9,170.5 418.2,170.3 419.4,170.0 420.7,169.7 422.0,169.3 423.3,168.9 424.6,168.4 425.9,167.9 427.1,167.3 428.4,166.7 429.7,166.0 431.0,165.3 432.3,164.5 433.6,163.6 434.8,162.7 436.1,161.8 437.4,160.8 438.7,159.8 440.0,158.7 441.3,157.6 442.5,156.4 443.8,155.2 445.1,154.0 446.4,152.7 447.7,151.4 449.0,150.0 450.2,148.7 451.5,147.3 452.8,145.8 454.1,144.3 455.4,142.9 456.7,141.3 457.9,139.8 459.2,138.2 460.5,136.7 461.8,135.1 463.1,133.5 464.4,131.9 465.6,130.2 466.9,128.6 468.2,126.9 469.5,125.3 470.8,123.6 472.1,122.0 473.3,120.3 474.6,118.7 475.9,117.0 477.2,115.4 478.5,113.8 479.8,112.2 481.0,110.6 482.3,109.0 483.6,107.4 484.9,105.8 486.2,104.3 487.5,102.8 488.7,101.3 490.0,99.9 491.3,98.4 492.6,97.0 493.9,95.7 495.2,94.3 496.4,93.0 497.7,91.7 499.0,90.5 500.3,89.3 501.6,88.2 502.9,87.1 504.1,86.0 505.4,85.0 506.7,84.0 508.0,83.1 509.3,82.2 510.6,81.4 511.8,80.6 513.1,79.8 514.4,79.2 515.7,78.5 517.0,78.0 518.3,77.4 519.5,77.0 520.8,76.6 522.1,76.2 523.4,75.9 524.7,75.7 526.0,75.5 527.2,75.4 528.5,75.3 529.8,75.3 531.1,75.3 532.4,75.4 533.7,75.6 534.9,75.8 536.2,76.1 537.5,76.4 538.8,76.8 540.1,77.2 541.4,77.7 542.6,78.3 543.9,78.9 545.2,79.5 546.5,80.2 547.8,81.0 549.1,81.8 550.3,82.7 551.6,83.6 552.9,84.5 554.2,85.5 555.5,86.6 556.8,87.7 558.0,88.8 559.3,90.0 560.6,91.2 561.9,92.4 563.2,93.7 564.5,95.0 565.7,96.4 567.0,97.8 568.3,99.2 569.6,100.6 570.9,102.1 572.2,103.6 573.4,105.1 574.7,106.7 576.0,108.3" class="stroke-a" fill="none"/>
<circle cx="64.0" cy="75.3" r="2.5" class="fill-b"/>
<circle cx="70.4" cy="76.0" r="2.5" class="fill-b"/>
<circle cx="76.8" cy="78.1" r="2.5" class="fill-b"/>
<circle cx="83.2" cy="81.5" r="2.5" class="fill-b"/>
<circle cx="89.6" cy="86.2" r="2.5" class="fill-b"/>
<circle cx="96.0" cy="92.0" r="2.5" class="fill-b"/>
<circle cx="102.4" cy="98.7" r="2.5" class="fill-b"/>
<circle cx="108.8" cy="106.1" r="2.5" class="fill-b"/>
<circle cx="115.2" cy="114.1" r="2.5" class="fill-b"/>
<circle cx="121.6" cy="122.3" r="2.5" class="fill-b"/>
<circle cx="128.0" cy="130.5" r="2.5" class="fill-b"/>
<circle cx="134.4" cy="138.5" r="2.5" class="fill-b"/>
<circle cx="140.8" cy="146.0" r="2.5" class="fill-b"/>
<circle cx="147.2" cy="152.8" r="2.5" class="fill-b"/>
<circle cx="153.6" cy="158.8" r="2.5" class="fill-b"/>
<circle cx="160.0" cy="163.7" r="2.5" class="fill-b"/>
<circle cx="166.4" cy="167.4" r="2.5" class="fill-b"/>
<circle cx="172.8" cy="169.7" r="2.5" class="fill-b"/>
<circle cx="179.2" cy="170.7" r="2.5" class="fill-b"/>
<circle cx="185.6" cy="170.3" r="2.5" class="fill-b"/>
<circle cx="192.0" cy="168.4" r="2.5" class="fill-b"/>
<circle cx="198.4" cy="165.2" r="2.5" class="fill-b"/>
<circle cx="204.8" cy="160.7" r="2.5" class="fill-b"/>
<circle cx="211.2" cy="155.1" r="2.5" class="fill-b"/>
<circle cx="217.6" cy="148.6" r="2.5" class="fill-b"/>
<circle cx="224.0" cy="141.3" r="2.5" class="fill-b"/>
<circle cx="230.4" cy="133.4" r="2.5" class="fill-b"/>
<circle cx="236.8" cy="125.2" r="2.5" class="fill-b"/>
<circle cx="243.2" cy="117.0" r="2.5" class="fill-b"/>
<circle cx="249.6" cy="109.0" r="2.5" class="fill-b"/>
<circle cx="256.0" cy="101.3" r="2.5" class="fill-b"/>
<circle cx="262.4" cy="94.3" r="2.5" class="fill-b"/>
<circle cx="268.8" cy="88.2" r="2.5" class="fill-b"/>
<circle cx="275.2" cy="83.1" r="2.5" class="fill-b"/>
<circle cx="281.6" cy="79.2" r="2.5" class="fill-b"/>
<circle cx="288.0" cy="76.6" r="2.5" class="fill-b"/>
<circle cx="294.4" cy="75.4" r="2.5" class="fill-b"/>
<circle cx="300.8" cy="75.6" r="2.5" class="fill-b"/>
<circle cx="307.2" cy="77.2" r="2.5" class="fill-b"/>
<circle cx="313.6" cy="80.1" r="2.5" class="fill-b"/>
<circle cx="320.0" cy="84.4" r="2.5" class="fill-b"/>
<circle cx="326.4" cy="89.8" r="2.5" class="fill-b"/>
<circle cx="332.8" cy="96.2" r="2.5" class="fill-b"/>
<circle cx="339.2" cy="103.4" r="2.5" class="fill-b"/>
<circle cx="345.6" cy="111.1" r="2.5" class="fill-b"/>
<circle cx="352.0" cy="119.3" r="2.5" class="fill-b"/>
<circle cx="358.4" cy="127.5" r="2.5" class="fill-b"/>
<circle cx="364.8" cy="135.6" r="2.5" class="fill-b"/>
<circle cx="371.2" cy="143.3" r="2.5" class="fill-b"/>
<circle cx="377.6" cy="150.4" r="2.5" class="fill-b"/>
<circle cx="384.0" cy="156.7" r="2.5" class="fill-b"/>
<circle cx="390.4" cy="162.0" r="2.5" class="fill-b"/>
<circle cx="396.8" cy="166.2" r="2.5" class="fill-b"/>
<circle cx="403.2" cy="169.0" r="2.5" class="fill-b"/>
<circle cx="409.6" cy="170.5" r="2.5" class="fill-b"/>
<circle cx="416.0" cy="170.6" r="2.5" class="fill-b"/>
<circle cx="422.4" cy="169.2" r="2.5" class="fill-b"/>
<circle cx="428.8" cy="166.5" r="2.5" class="fill-b"/>
<circle cx="435.2" cy="162.5" r="2.5" class="fill-b"/>
<circle cx="441.6" cy="157.3" r="2.5" class="fill-b"/>
<circle cx="448.0" cy="151.1" r="2.5" class="fill-b"/>
<circle cx="454.4" cy="144.0" r="2.5" class="fill-b"/>
<circle cx="460.8" cy="136.3" r="2.5" class="fill-b"/>
<circle cx="467.2" cy="128.2" r="2.5" class="fill-b"/>
<circle cx="473.6" cy="120.0" r="2.5" class="fill-b"/>
<circle cx="480.0" cy="111.9" r="2.5" class="fill-b"/>
<circle cx="486.4" cy="104.0" r="2.5" class="fill-b"/>
<circle cx="492.8" cy="96.8" r="2.5" class="fill-b"/>
<circle cx="499.2" cy="90.3" r="2.5" class="fill-b"/>
<circle cx="505.6" cy="84.8" r="2.5" class="fill-b"/>
<circle cx="512.0" cy="80.5" r="2.5" class="fill-b"/>
<circle cx="518.4" cy="77.4" r="2.5" class="fill-b"/>
<circle cx="524.8" cy="75.6" r="2.5" class="fill-b"/>
<circle cx="531.2" cy="75.3" r="2.5" class="fill-b"/>
<circle cx="537.6" cy="76.4" r="2.5" class="fill-b"/>
<circle cx="544.0" cy="78.9" r="2.5" class="fill-b"/>
<circle cx="550.4" cy="82.7" r="2.5" class="fill-b"/>
<circle cx="556.8" cy="87.7" r="2.5" class="fill-b"/>
<circle cx="563.2" cy="93.7" r="2.5" class="fill-b"/>
<circle cx="569.6" cy="100.7" r="2.5" class="fill-b"/>
<circle cx="576.0" cy="108.3" r="2.5" class="fill-b"/>
<line x1="64" y1="75.3" x2="576" y2="75.3" class="stroke-muted dashed"/>
<line x1="64" y1="89.2" x2="576" y2="89.2" class="stroke-c dashed"/>
<text x="412.2" y="70.3" class="fig-small" text-anchor="middle">peak 0.5</text>
<text x="412.2" y="104.2" class="fig-small" text-anchor="middle">RMS 0.354</text>
<defs><marker id="c001-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="64.0" y1="197.5" x2="296.7" y2="197.5" class="stroke-ink" marker-end="url(#c001-arrow)"/>
<text x="302.7" y="201.5" class="fig-small" text-anchor="start">one period, 2.27 ms</text>
</svg>
Figure 1. The running example: a 440 Hz cosine with amplitude 0.5. The smooth curve is what the air does, and each orange dot is one sample taken at 16,000 per second. Count the dots in one period.
```

A computer cannot store a curve, which has a value at every instant. So it stores the curve's value at evenly spaced instants, one **sample** each, and the number of samples per second is the **sample rate** $f_s$, in hertz. Speech systems usually take 16 kHz, which is 16,000 numbers for each second of sound. Chapter [Sampling and aliasing](/lessons/002-sampling-and-aliasing) asks what this costs you. Sample number $n$ is the curve's value at time $n / f_s$, and the whole signal is an array.

A tone as a function of time $t$, in seconds, is:

$$
x(t) = A \cos(2 \pi f t + \varphi)
$$

Read it aloud: at time $t$, the wave has turned through $2\pi f t$ radians, plus its starting phase, and the amplitude times the cosine of that angle is the pressure. One cycle takes $1/f$ seconds, the **period**. For 440 Hz that is 2.27 milliseconds, or 36.4 samples at 16 kHz, which is why Figure 1 shows about 36 dots in each swing of the curve.

Put $t = n / f_s$ into the formula and you get the array. Square brackets, as in $x[n]$, mark a sampled signal. In numpy:

```python
n = np.arange(round(seconds * sample_rate))
x = amplitude * np.cos(2 * np.pi * frequency * n / sample_rate + phase)
```

The first five samples of the running example are 0.5, 0.4926, 0.4704, 0.4343, and 0.3853. They start at the peak because a cosine with a phase of 0 starts at its maximum, and they fall away as the wave turns.

## Why the peak is not loudness

The first number you might reach for is the **peak**: the largest absolute sample, `np.max(np.abs(x))`. It is easy to compute, and it matters, because a peak above 1.0 clips. But it does not say how loud a clip sounds.

Take two clips of four samples each, in Figure 2. The first, a buzz, is $0.5, -0.5, 0.5, -0.5$. The second, a click, is $0.5, 0, 0, 0$. Both have a peak of 0.5. The buzz keeps the air moving the whole time and sounds loud. The click is one brief tick and sounds faint. A peak gives them the same number, so it cannot be the loudness.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Two four-sample clips with the same peak of 0.5: a buzz with RMS 0.5 at minus 6.02 dBFS and a click with RMS 0.25 at minus 12.04 dBFS">
<line x1="40" y1="120" x2="248" y2="120" class="stroke-muted"/>
<line x1="66" y1="120" x2="66" y2="40.0" class="stroke-a"/>
<circle cx="66" cy="40.0" r="4" class="fill-b"/>
<line x1="118" y1="120" x2="118" y2="200.0" class="stroke-a"/>
<circle cx="118" cy="200.0" r="4" class="fill-b"/>
<line x1="170" y1="120" x2="170" y2="40.0" class="stroke-a"/>
<circle cx="170" cy="40.0" r="4" class="fill-b"/>
<line x1="222" y1="120" x2="222" y2="200.0" class="stroke-a"/>
<circle cx="222" cy="200.0" r="4" class="fill-b"/>
<line x1="330" y1="120" x2="538" y2="120" class="stroke-muted"/>
<line x1="356" y1="120" x2="356" y2="40.0" class="stroke-a"/>
<circle cx="356" cy="40.0" r="4" class="fill-b"/>
<line x1="408" y1="120" x2="408" y2="120" class="stroke-a"/>
<circle cx="408" cy="120" r="4" class="fill-b"/>
<line x1="460" y1="120" x2="460" y2="120" class="stroke-a"/>
<circle cx="460" cy="120" r="4" class="fill-b"/>
<line x1="512" y1="120" x2="512" y2="120" class="stroke-a"/>
<circle cx="512" cy="120" r="4" class="fill-b"/>
<text x="40" y="24" class="fig-bold" text-anchor="start">Buzz: 0.5, −0.5, 0.5, −0.5</text>
<text x="330" y="24" class="fig-bold" text-anchor="start">Click: 0.5, 0, 0, 0</text>
<line x1="40" y1="40.0" x2="248" y2="40.0" class="stroke-muted dashed"/>
<text x="40" y="218" class="fig-small" text-anchor="start">peak −6.02 dBFS</text>
<text x="40" y="236" class="fig-small fig-bold" text-anchor="start">RMS 0.5, −6.02 dBFS</text>
<line x1="330" y1="40.0" x2="538" y2="40.0" class="stroke-muted dashed"/>
<text x="330" y="218" class="fig-small" text-anchor="start">peak −6.02 dBFS</text>
<text x="330" y="236" class="fig-small fig-bold" text-anchor="start">RMS 0.25, −12.04 dBFS</text>
</svg>
Figure 2. Two clips with the same peak. The buzz has four times the energy of the click, so its RMS is twice as large, 6.02 dB higher.
```

Loudness follows how much energy the sound carries over time. The energy in a sample grows with the square of its size, because a wave twice as big pushes four times as much power. So average the squares, and take a square root to get back to the units of a sample. That is the **root mean square**, or **RMS**:

$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}
$$

Read it aloud: square every sample, take the mean of the $N$ squares, and take the square root. For the buzz, every square is 0.25, the mean is 0.25, and the RMS is 0.5. For the click, the squares are 0.25, 0, 0, and 0, so the mean is 0.0625 and the RMS is 0.25. The RMS separates the two clips, and the click comes out 6.02 dB lower. In numpy:

```python
rms = np.sqrt(np.mean(x ** 2))
```

For a sine of amplitude $A$ that fits whole cycles, the RMS is $A / \sqrt{2}$. The reason is that the squared cosine swings between 0 and 1 and averages $1/2$ over a cycle, so the mean square is $A^2/2$. The running example has an RMS of $0.5 / 1.4142 = 0.3536$, the green dashed line in Figure 1. It sits below the peak because the wave spends most of its time away from the top.

> [!KEY]
> The peak tells you how close a clip comes to clipping. The RMS tells you how much energy it carries, which is what loudness follows. Report both.

## Why a log scale

The RMS is a loudness number, but a raw RMS is awkward to read. Hearing covers a range of about a million to one in pressure, from a whisper to a jet engine. On a linear scale, a whisper is a speck next to a shout, and the difference between a speck and half a speck is invisible. Your ear does not work like that. Roughly, it judges loudness by ratios: a sound twice as big as another sounds like the same step up whether the starting sound is faint or loud.

A logarithm turns ratios into differences, so it matches the ear. The **decibel** (dB) is the logarithm scaled to be convenient. For an amplitude ratio:

$$
L = 20 \log_{10} \frac{A}{A_{\text{ref}}}
$$

Read it aloud: divide the amplitude by the reference, take the base-10 logarithm, and multiply by 20. Why 20 and not 10? Power grows with the square of amplitude, and the decibel was defined for powers as $10 \log_{10} (P / P_{\text{ref}})$. Because $\log_{10}(A^2) = 2 \log_{10} A$, the same rule for amplitudes needs a 20.

> [!KEY]
> Use `20 * np.log10` for amplitudes and magnitudes, and `10 * np.log10` for powers and energies. Mixing them up doubles or halves every level.

A digital signal has a natural reference: full scale, an amplitude of 1.0. A level measured against it is in **dBFS**, decibels relative to full scale. So 0 dBFS is the loudest value a float signal can hold, and everything else is negative. Figure 3 gives five amplitudes and their levels.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Five amplitudes drawn as bars with their dBFS levels: 1 is 0, 0.5 is minus 6.02, 0.25 is minus 12.04, 0.1 is minus 20, and 0.01 is minus 40">
<rect x="150" y="22" width="340.0" height="28" class="fill-soft-a"/>
<rect x="150" y="22" width="340" height="28" class="stroke-muted" fill="none"/>
<text x="140" y="41" class="fig-mono" text-anchor="end">1</text>
<text x="500" y="41" class="fig-bold" text-anchor="start">0 dBFS</text>
<rect x="150" y="64" width="170.0" height="28" class="fill-soft-a"/>
<rect x="150" y="64" width="340" height="28" class="stroke-muted" fill="none"/>
<text x="140" y="83" class="fig-mono" text-anchor="end">0.5</text>
<text x="500" y="83" class="fig-bold" text-anchor="start">−6.02 dBFS</text>
<rect x="150" y="106" width="85.0" height="28" class="fill-soft-a"/>
<rect x="150" y="106" width="340" height="28" class="stroke-muted" fill="none"/>
<text x="140" y="125" class="fig-mono" text-anchor="end">0.25</text>
<text x="500" y="125" class="fig-bold" text-anchor="start">−12.04 dBFS</text>
<rect x="150" y="148" width="34.0" height="28" class="fill-soft-a"/>
<rect x="150" y="148" width="340" height="28" class="stroke-muted" fill="none"/>
<text x="140" y="167" class="fig-mono" text-anchor="end">0.1</text>
<text x="500" y="167" class="fig-bold" text-anchor="start">−20.00 dBFS</text>
<rect x="150" y="190" width="3.4" height="28" class="fill-soft-a"/>
<rect x="150" y="190" width="340" height="28" class="stroke-muted" fill="none"/>
<text x="140" y="209" class="fig-mono" text-anchor="end">0.01</text>
<text x="500" y="209" class="fig-bold" text-anchor="start">−40.00 dBFS</text>
<text x="150" y="240" class="fig-small" text-anchor="start">amplitude as a fraction of full scale (bar length)</text>
<text x="10" y="14" class="fig-small" text-anchor="start">amplitude</text>
</svg>
Figure 3. Amplitude against level in dBFS. Each halving of the amplitude is the same 6.02 dB step down, and a hundredfold drop is 40 dB.
```

Work two of them by hand. An amplitude of 0.5 is $20 \log_{10} 0.5 = -6.02$ dBFS. An amplitude of 0.25 is half of that, so it is another 6.02 dB lower, at $-12.04$ dBFS. Doubling an amplitude therefore adds 6.02 dB, and doubling a power adds 3.01 dB.

You can measure two levels of a clip. The **peak level** applies the formula to the peak, and the **RMS level** applies it to the RMS. For the running example, the peak is 0.5, which is $-6.02$ dBFS, and the RMS is 0.3536, which is $-9.03$ dBFS. The gap of 3.01 dB is $20 \log_{10}\sqrt{2}$, and it is the same for every sine. A full-scale sine reads 0 dBFS peak and $-3.01$ dBFS RMS. Some tools follow the AES17 convention, which calls the RMS of a full-scale sine 0 dBFS. This course references the peak.

A silent clip has an RMS of 0, and `np.log10(0)` is minus infinity, which breaks every later mean. So floor the value at a tiny `eps` before the logarithm:

```python
level = 20 * np.log10(max(rms, eps))
```

> [!WARNING]
> dBFS describes a file, not a room. The same file plays at any volume, depending on the speaker. A level in dBSPL, decibels of sound pressure against 20 micropascals, needs a calibrated microphone, and you cannot get it from samples alone.

> [!KEY]
> Back to the two takes. The quieter take has a lower RMS, and the difference between them in dB is one number that does not depend on how loud you started. If your second take has half the amplitude of the first, its level is exactly 6.02 dB lower, whatever the first level was.

> [!RECAP]
> - A recording is an array: sample $n$ of a tone is `A * np.cos(2 * np.pi * f * n / fs + phase)`.
> - The peak is `np.max(np.abs(x))`, and it misses how long a sound lasts.
> - The RMS is `np.sqrt(np.mean(x ** 2))`, and a sine of amplitude $A$ has an RMS of $A/\sqrt{2}$.
> - A level in dBFS is `20 * np.log10(max(value, eps))` of an amplitude, and halving the amplitude lowers it by 6.02 dB.
> - Use 20 for amplitudes and 10 for powers.

## Build it step by step

The checks test the functions in this order.

1. **`rms(x)`** returns a float. Common mistake: `np.sqrt(np.sum(x ** 2))`, which grows with the length of the clip instead of staying at its typical size.
2. **`peak_dbfs(x, eps)`** and **`rms_dbfs(x, eps)`** floor `np.max(np.abs(x))` or your `rms` at `eps` and apply `20 * np.log10`. Common mistake: using 10, which reports a 0.5 sine at $-3.01$ dBFS instead of $-6.02$.
3. **`sample_tone(frequency, sample_rate, seconds, amplitude, phase)`** returns `round(seconds * sample_rate)` cosine samples at times `np.arange(N) / sample_rate`. Common mistake: `np.linspace(0, seconds, N)`, which includes the end point and stretches every step.

## Where this breaks

**A long clip with pauses.** The RMS over a whole recording averages in the silences, so a clip that is mostly quiet reads low. Measure level over short stretches when you care about the speech itself. The chapter [Voice activity detection](/lessons/020-voice-activity-detection) does exactly that.

**Clipping hides in the peak.** A float signal above 1.0 has no meaning in a 16-bit file, and the converter cuts it off. A peak level above 0 dBFS means the clip is damaged, or will be when saved.

## The exercise

You build `rms`, `peak_dbfs`, `rms_dbfs`, and `sample_tone`. Press **Run** to build the running example, print its peak and RMS levels, play it, and see the first 10 ms of it. It then plays the same tone at half the amplitude, so you can hear what 6.02 dB sounds like.

The checks confirm the RMS of three signals, including the buzz and the click, the levels of a full-scale sine and a half-scale sine, and that silence returns a finite floor instead of minus infinity. They compare your `sample_tone` with the cosine formula, and the last check measures the running example with your own functions and expects $-6.02$ dBFS peak and $-9.03$ dBFS RMS.
