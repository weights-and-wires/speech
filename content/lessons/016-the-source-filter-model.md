---
title: The source-filter model
slug: 016-the-source-filter-model
part: "Part 2 — Hearing and speech production"
summary: Build a vowel from a pulse train and a cascade of formant resonators, and see why pitch and vowel identity can change independently.
minutes: 40
gpu: false
objectives:
  - Explain the source-filter model of speech, and say what the source sets and what the filter sets.
  - Predict the formants of a neutral vowel from a uniform tube, and read the formants of /i/, /ɑ/, and /u/ from a table.
  - Build a formant resonator from its frequency and bandwidth, and say where its poles sit.
  - Synthesize a vowel from an impulse train and a cascade of resonators, and show that changing the pitch leaves the formants where they are.
prereqs:
  - 009-convolution-and-filters
  - 014-mfccs
exercise: 016-the-source-filter-model
---

# The source-filter model

Sing "ah" on a low note, then on a high note. The note changes and the vowel stays "ah". Now hold one note and slide from "ah" to "ee". The vowel changes and the note stays. Pitch and vowel are two separate controls, and your body moves them with two separate parts: your vocal folds set the note, and the shape of your throat, mouth, and lips sets the vowel.

That separation is the *source-filter model*. The source is the buzz of the vocal folds, which snap open and shut about 120 times a second and send a pulse of air up the throat at each snap. On its own, that buzz has a pitch and no vowel. The filter is the vocal tract, which shapes the buzz into "ah" or "ee". This is why a text-to-speech system can predict the pitch and the spectrum shape separately, and why the MFCCs of the last chapter, which keep the envelope and drop the harmonics, throw away the pitch and keep the vowel.

In this chapter you build the model in the forward direction: a pulse train goes in, resonators shape it, and a vowel comes out. You then check the claim that makes the model useful. You can change one without touching the other.

## What you already know

- An IIR filter computes each output from past outputs, and `scipy.signal.lfilter(b, a, x)` runs its difference equation.
- A filter's *poles* are the roots of its denominator polynomial, and a pole near the unit circle makes a sharp peak in the frequency response.
- Convolving two signals multiplies their spectra.
- An MFCC vector keeps the smooth envelope of a spectrum and drops the harmonics.
- A signal sampled at $f_s$ has frequencies from 0 to $f_s/2$, and a full turn of the unit circle is $2\pi$ radians.

## A source, a filter, and their product

In the language of the $z$-transform, where multiplying transforms is the same as convolving signals, speech is a chain:

$$
S(z) = E(z)\,G(z)\,V(z)\,R(z)
$$

Read it aloud: speech $S$ is the excitation $E$, shaped by the glottal pulse $G$, then by the vocal tract $V$, then by the radiation at the lips $R$. For a *voiced* sound such as a vowel, $E$ is a pulse train at the *fundamental frequency* $F_0$, the rate of the vocal folds. For an *unvoiced* sound such as "s", $E$ is noise from air rushing through a narrow gap. This chapter builds voiced vowels and folds $G$ and $R$ into a single vocal-tract filter.

A pulse train with period $T_0 = 1/F_0$ has a spectrum made only of lines at $F_0, 2F_0, 3F_0, \dots$, the *harmonics*. The filter does not move those lines. It raises and lowers them. The peaks of the filter's response are the *formants*, numbered $F_1, F_2, F_3$ from the bottom, and they decide the vowel. $F_0$ decides the pitch.

```figure
<svg viewBox="0 0 620 240" role="img" aria-label="Three panels: a pulse train's equal-height harmonic lines every 100 hertz, a three-resonator filter curve with peaks at 730, 1090, and 2440 hertz, and the harmonic lines of the output tracing that curve">
<line x1="40.0" y1="180.0" x2="190.0" y2="180.0" class="stroke-grid"/>
<text x="34.0" y="184.0" class="fig-small" text-anchor="end">-60</text>
<line x1="40.0" y1="151.1" x2="190.0" y2="151.1" class="stroke-grid"/>
<text x="34.0" y="155.1" class="fig-small" text-anchor="end">-40</text>
<line x1="40.0" y1="122.2" x2="190.0" y2="122.2" class="stroke-grid"/>
<text x="34.0" y="126.2" class="fig-small" text-anchor="end">-20</text>
<line x1="40.0" y1="93.3" x2="190.0" y2="93.3" class="stroke-grid"/>
<text x="34.0" y="97.3" class="fig-small" text-anchor="end">0</text>
<line x1="40.0" y1="64.4" x2="190.0" y2="64.4" class="stroke-grid"/>
<text x="34.0" y="68.4" class="fig-small" text-anchor="end">20</text>
<line x1="40.0" y1="180.0" x2="190.0" y2="180.0" class="stroke-muted"/>
<line x1="40.0" y1="50.0" x2="40.0" y2="180.0" class="stroke-muted"/>
<line x1="40.0" y1="180.0" x2="40.0" y2="185.0" class="stroke-muted"/>
<text x="40.0" y="199.0" class="fig-small" text-anchor="middle">0</text>
<line x1="115.0" y1="180.0" x2="115.0" y2="185.0" class="stroke-muted"/>
<text x="115.0" y="199.0" class="fig-small" text-anchor="middle">2k</text>
<line x1="190.0" y1="180.0" x2="190.0" y2="185.0" class="stroke-muted"/>
<text x="190.0" y="199.0" class="fig-small" text-anchor="middle">4k</text>
<line x1="43.8" y1="180.0" x2="43.8" y2="93.3" class="stroke-a"/>
<line x1="47.5" y1="180.0" x2="47.5" y2="93.3" class="stroke-a"/>
<line x1="51.2" y1="180.0" x2="51.2" y2="93.3" class="stroke-a"/>
<line x1="55.0" y1="180.0" x2="55.0" y2="93.3" class="stroke-a"/>
<line x1="58.8" y1="180.0" x2="58.8" y2="93.3" class="stroke-a"/>
<line x1="62.5" y1="180.0" x2="62.5" y2="93.3" class="stroke-a"/>
<line x1="66.2" y1="180.0" x2="66.2" y2="93.3" class="stroke-a"/>
<line x1="70.0" y1="180.0" x2="70.0" y2="93.3" class="stroke-a"/>
<line x1="73.8" y1="180.0" x2="73.8" y2="93.3" class="stroke-a"/>
<line x1="77.5" y1="180.0" x2="77.5" y2="93.3" class="stroke-a"/>
<line x1="81.2" y1="180.0" x2="81.2" y2="93.3" class="stroke-a"/>
<line x1="85.0" y1="180.0" x2="85.0" y2="93.3" class="stroke-a"/>
<line x1="88.8" y1="180.0" x2="88.8" y2="93.3" class="stroke-a"/>
<line x1="92.5" y1="180.0" x2="92.5" y2="93.3" class="stroke-a"/>
<line x1="96.2" y1="180.0" x2="96.2" y2="93.3" class="stroke-a"/>
<line x1="100.0" y1="180.0" x2="100.0" y2="93.3" class="stroke-a"/>
<line x1="103.8" y1="180.0" x2="103.8" y2="93.3" class="stroke-a"/>
<line x1="107.5" y1="180.0" x2="107.5" y2="93.3" class="stroke-a"/>
<line x1="111.2" y1="180.0" x2="111.2" y2="93.3" class="stroke-a"/>
<line x1="115.0" y1="180.0" x2="115.0" y2="93.3" class="stroke-a"/>
<line x1="118.8" y1="180.0" x2="118.8" y2="93.3" class="stroke-a"/>
<line x1="122.5" y1="180.0" x2="122.5" y2="93.3" class="stroke-a"/>
<line x1="126.2" y1="180.0" x2="126.2" y2="93.3" class="stroke-a"/>
<line x1="130.0" y1="180.0" x2="130.0" y2="93.3" class="stroke-a"/>
<line x1="133.8" y1="180.0" x2="133.8" y2="93.3" class="stroke-a"/>
<line x1="137.5" y1="180.0" x2="137.5" y2="93.3" class="stroke-a"/>
<line x1="141.2" y1="180.0" x2="141.2" y2="93.3" class="stroke-a"/>
<line x1="145.0" y1="180.0" x2="145.0" y2="93.3" class="stroke-a"/>
<line x1="148.8" y1="180.0" x2="148.8" y2="93.3" class="stroke-a"/>
<line x1="152.5" y1="180.0" x2="152.5" y2="93.3" class="stroke-a"/>
<line x1="156.2" y1="180.0" x2="156.2" y2="93.3" class="stroke-a"/>
<line x1="160.0" y1="180.0" x2="160.0" y2="93.3" class="stroke-a"/>
<line x1="163.8" y1="180.0" x2="163.8" y2="93.3" class="stroke-a"/>
<line x1="167.5" y1="180.0" x2="167.5" y2="93.3" class="stroke-a"/>
<line x1="171.2" y1="180.0" x2="171.2" y2="93.3" class="stroke-a"/>
<line x1="175.0" y1="180.0" x2="175.0" y2="93.3" class="stroke-a"/>
<line x1="178.8" y1="180.0" x2="178.8" y2="93.3" class="stroke-a"/>
<line x1="182.5" y1="180.0" x2="182.5" y2="93.3" class="stroke-a"/>
<line x1="186.2" y1="180.0" x2="186.2" y2="93.3" class="stroke-a"/>
<line x1="190.0" y1="180.0" x2="190.0" y2="93.3" class="stroke-a"/>
<text x="115.0" y="76.0" class="fig-small" text-anchor="middle">every line 0 dB</text>
<line x1="232.0" y1="180.0" x2="382.0" y2="180.0" class="stroke-grid"/>
<text x="226.0" y="184.0" class="fig-small" text-anchor="end">-60</text>
<line x1="232.0" y1="151.1" x2="382.0" y2="151.1" class="stroke-grid"/>
<text x="226.0" y="155.1" class="fig-small" text-anchor="end">-40</text>
<line x1="232.0" y1="122.2" x2="382.0" y2="122.2" class="stroke-grid"/>
<text x="226.0" y="126.2" class="fig-small" text-anchor="end">-20</text>
<line x1="232.0" y1="93.3" x2="382.0" y2="93.3" class="stroke-grid"/>
<text x="226.0" y="97.3" class="fig-small" text-anchor="end">0</text>
<line x1="232.0" y1="64.4" x2="382.0" y2="64.4" class="stroke-grid"/>
<text x="226.0" y="68.4" class="fig-small" text-anchor="end">20</text>
<line x1="232.0" y1="180.0" x2="382.0" y2="180.0" class="stroke-muted"/>
<line x1="232.0" y1="50.0" x2="232.0" y2="180.0" class="stroke-muted"/>
<line x1="232.0" y1="180.0" x2="232.0" y2="185.0" class="stroke-muted"/>
<text x="232.0" y="199.0" class="fig-small" text-anchor="middle">0</text>
<line x1="307.0" y1="180.0" x2="307.0" y2="185.0" class="stroke-muted"/>
<text x="307.0" y="199.0" class="fig-small" text-anchor="middle">2k</text>
<line x1="382.0" y1="180.0" x2="382.0" y2="185.0" class="stroke-muted"/>
<text x="382.0" y="199.0" class="fig-small" text-anchor="middle">4k</text>
<polyline points="232.0,93.3 232.4,93.3 232.8,93.3 233.1,93.3 233.5,93.3 233.9,93.2 234.3,93.2 234.6,93.2 235.0,93.1 235.4,93.0 235.8,93.0 236.1,92.9 236.5,92.8 236.9,92.7 237.3,92.6 237.6,92.5 238.0,92.4 238.4,92.3 238.8,92.1 239.1,92.0 239.5,91.8 239.9,91.7 240.3,91.5 240.6,91.3 241.0,91.1 241.4,90.9 241.8,90.7 242.2,90.5 242.5,90.3 242.9,90.1 243.3,89.8 243.7,89.5 244.0,89.3 244.4,89.0 244.8,88.7 245.2,88.4 245.5,88.1 245.9,87.7 246.3,87.4 246.7,87.0 247.0,86.7 247.4,86.3 247.8,85.9 248.2,85.4 248.5,85.0 248.9,84.5 249.3,84.0 249.7,83.5 250.0,83.0 250.4,82.4 250.8,81.9 251.2,81.3 251.5,80.6 251.9,80.0 252.3,79.3 252.7,78.5 253.1,77.7 253.4,76.9 253.8,76.1 254.2,75.1 254.6,74.2 254.9,73.1 255.3,72.0 255.7,70.8 256.1,69.5 256.4,68.1 256.8,66.7 257.2,65.1 257.6,63.4 257.9,61.6 258.3,59.9 258.7,58.3 259.1,57.2 259.4,56.7 259.8,57.0 260.2,57.9 260.6,59.1 260.9,60.3 261.3,61.6 261.7,62.8 262.1,63.9 262.5,64.8 262.8,65.7 263.2,66.4 263.6,67.1 264.0,67.6 264.3,68.1 264.7,68.5 265.1,68.8 265.5,69.1 265.8,69.3 266.2,69.4 266.6,69.4 267.0,69.4 267.3,69.4 267.7,69.2 268.1,69.0 268.5,68.7 268.8,68.4 269.2,67.9 269.6,67.4 270.0,66.8 270.3,66.1 270.7,65.4 271.1,64.6 271.5,63.8 271.8,63.1 272.2,62.6 272.6,62.4 273.0,62.7 273.4,63.5 273.7,64.7 274.1,66.1 274.5,67.7 274.9,69.3 275.2,70.9 275.6,72.4 276.0,73.9 276.4,75.3 276.7,76.6 277.1,77.9 277.5,79.1 277.9,80.3 278.2,81.4 278.6,82.4 279.0,83.4 279.4,84.4 279.7,85.3 280.1,86.2 280.5,87.0 280.9,87.8 281.2,88.6 281.6,89.4 282.0,90.1 282.4,90.8 282.8,91.5 283.1,92.2 283.5,92.8 283.9,93.5 284.3,94.1 284.6,94.7 285.0,95.2 285.4,95.8 285.8,96.3 286.1,96.9 286.5,97.4 286.9,97.9 287.3,98.4 287.6,98.9 288.0,99.3 288.4,99.8 288.8,100.2 289.1,100.7 289.5,101.1 289.9,101.5 290.3,101.9 290.6,102.3 291.0,102.7 291.4,103.1 291.8,103.4 292.2,103.8 292.5,104.1 292.9,104.5 293.3,104.8 293.7,105.1 294.0,105.4 294.4,105.7 294.8,106.0 295.2,106.3 295.5,106.6 295.9,106.9 296.3,107.2 296.7,107.4 297.0,107.7 297.4,107.9 297.8,108.2 298.2,108.4 298.5,108.6 298.9,108.8 299.3,109.1 299.7,109.3 300.0,109.5 300.4,109.7 300.8,109.8 301.2,110.0 301.5,110.2 301.9,110.3 302.3,110.5 302.7,110.6 303.1,110.8 303.4,110.9 303.8,111.0 304.2,111.2 304.6,111.3 304.9,111.4 305.3,111.5 305.7,111.6 306.1,111.6 306.4,111.7 306.8,111.8 307.2,111.8 307.6,111.9 307.9,111.9 308.3,111.9 308.7,112.0 309.1,112.0 309.4,112.0 309.8,111.9 310.2,111.9 310.6,111.9 310.9,111.8 311.3,111.8 311.7,111.7 312.1,111.6 312.5,111.5 312.8,111.4 313.2,111.3 313.6,111.2 314.0,111.0 314.3,110.8 314.7,110.6 315.1,110.4 315.5,110.2 315.8,109.9 316.2,109.6 316.6,109.3 317.0,109.0 317.3,108.7 317.7,108.3 318.1,107.9 318.5,107.5 318.8,107.0 319.2,106.5 319.6,106.0 320.0,105.4 320.3,104.9 320.7,104.3 321.1,103.7 321.5,103.1 321.8,102.6 322.2,102.2 322.6,101.9 323.0,101.7 323.4,101.8 323.7,102.0 324.1,102.5 324.5,103.2 324.9,104.0 325.2,105.0 325.6,106.0 326.0,107.0 326.4,108.1 326.7,109.2 327.1,110.2 327.5,111.3 327.9,112.3 328.2,113.2 328.6,114.2 329.0,115.1 329.4,116.0 329.7,116.9 330.1,117.7 330.5,118.5 330.9,119.3 331.2,120.1 331.6,120.8 332.0,121.5 332.4,122.2 332.8,122.9 333.1,123.6 333.5,124.2 333.9,124.9 334.3,125.5 334.6,126.1 335.0,126.7 335.4,127.3 335.8,127.9 336.1,128.4 336.5,129.0 336.9,129.5 337.3,130.0 337.6,130.6 338.0,131.1 338.4,131.6 338.8,132.1 339.1,132.6 339.5,133.0 339.9,133.5 340.3,134.0 340.6,134.4 341.0,134.9 341.4,135.3 341.8,135.8 342.2,136.2 342.5,136.6 342.9,137.1 343.3,137.5 343.7,137.9 344.0,138.3 344.4,138.7 344.8,139.1 345.2,139.5 345.5,139.9 345.9,140.3 346.3,140.6 346.7,141.0 347.0,141.4 347.4,141.8 347.8,142.1 348.2,142.5 348.5,142.8 348.9,143.2 349.3,143.5 349.7,143.9 350.0,144.2 350.4,144.6 350.8,144.9 351.2,145.2 351.5,145.6 351.9,145.9 352.3,146.2 352.7,146.5 353.1,146.9 353.4,147.2 353.8,147.5 354.2,147.8 354.6,148.1 354.9,148.4 355.3,148.7 355.7,149.0 356.1,149.3 356.4,149.6 356.8,149.9 357.2,150.2 357.6,150.5 357.9,150.8 358.3,151.1 358.7,151.3 359.1,151.6 359.4,151.9 359.8,152.2 360.2,152.5 360.6,152.7 360.9,153.0 361.3,153.3 361.7,153.5 362.1,153.8 362.5,154.1 362.8,154.3 363.2,154.6 363.6,154.8 364.0,155.1 364.3,155.3 364.7,155.6 365.1,155.8 365.5,156.1 365.8,156.3 366.2,156.6 366.6,156.8 367.0,157.1 367.3,157.3 367.7,157.6 368.1,157.8 368.5,158.0 368.8,158.3 369.2,158.5 369.6,158.7 370.0,159.0 370.3,159.2 370.7,159.4 371.1,159.6 371.5,159.9 371.8,160.1 372.2,160.3 372.6,160.5 373.0,160.8 373.4,161.0 373.7,161.2 374.1,161.4 374.5,161.6 374.9,161.8 375.2,162.1 375.6,162.3 376.0,162.5 376.4,162.7 376.7,162.9 377.1,163.1 377.5,163.3 377.9,163.5 378.2,163.7 378.6,163.9 379.0,164.1 379.4,164.3 379.7,164.5 380.1,164.7 380.5,164.9 380.9,165.1 381.2,165.3 381.6,165.5 382.0,165.7" class="stroke-d" fill="none"/>
<line x1="424.0" y1="180.0" x2="574.0" y2="180.0" class="stroke-grid"/>
<text x="418.0" y="184.0" class="fig-small" text-anchor="end">-60</text>
<line x1="424.0" y1="151.1" x2="574.0" y2="151.1" class="stroke-grid"/>
<text x="418.0" y="155.1" class="fig-small" text-anchor="end">-40</text>
<line x1="424.0" y1="122.2" x2="574.0" y2="122.2" class="stroke-grid"/>
<text x="418.0" y="126.2" class="fig-small" text-anchor="end">-20</text>
<line x1="424.0" y1="93.3" x2="574.0" y2="93.3" class="stroke-grid"/>
<text x="418.0" y="97.3" class="fig-small" text-anchor="end">0</text>
<line x1="424.0" y1="64.4" x2="574.0" y2="64.4" class="stroke-grid"/>
<text x="418.0" y="68.4" class="fig-small" text-anchor="end">20</text>
<line x1="424.0" y1="180.0" x2="574.0" y2="180.0" class="stroke-muted"/>
<line x1="424.0" y1="50.0" x2="424.0" y2="180.0" class="stroke-muted"/>
<line x1="424.0" y1="180.0" x2="424.0" y2="185.0" class="stroke-muted"/>
<text x="424.0" y="199.0" class="fig-small" text-anchor="middle">0</text>
<line x1="499.0" y1="180.0" x2="499.0" y2="185.0" class="stroke-muted"/>
<text x="499.0" y="199.0" class="fig-small" text-anchor="middle">2k</text>
<line x1="574.0" y1="180.0" x2="574.0" y2="185.0" class="stroke-muted"/>
<text x="574.0" y="199.0" class="fig-small" text-anchor="middle">4k</text>
<line x1="427.8" y1="180.0" x2="427.8" y2="93.0" class="stroke-b"/>
<line x1="431.5" y1="180.0" x2="431.5" y2="91.8" class="stroke-b"/>
<line x1="435.2" y1="180.0" x2="435.2" y2="89.8" class="stroke-b"/>
<line x1="439.0" y1="180.0" x2="439.0" y2="86.7" class="stroke-b"/>
<line x1="442.8" y1="180.0" x2="442.8" y2="81.9" class="stroke-b"/>
<line x1="446.5" y1="180.0" x2="446.5" y2="74.3" class="stroke-b"/>
<line x1="450.2" y1="180.0" x2="450.2" y2="60.2" class="stroke-b"/>
<line x1="454.0" y1="180.0" x2="454.0" y2="63.7" class="stroke-b"/>
<line x1="457.8" y1="180.0" x2="457.8" y2="69.2" class="stroke-b"/>
<line x1="461.5" y1="180.0" x2="461.5" y2="67.5" class="stroke-b"/>
<line x1="465.2" y1="180.0" x2="465.2" y2="63.2" class="stroke-b"/>
<line x1="469.0" y1="180.0" x2="469.0" y2="77.5" class="stroke-b"/>
<line x1="472.8" y1="180.0" x2="472.8" y2="87.6" class="stroke-b"/>
<line x1="476.5" y1="180.0" x2="476.5" y2="94.4" class="stroke-b"/>
<line x1="480.2" y1="180.0" x2="480.2" y2="99.6" class="stroke-b"/>
<line x1="484.0" y1="180.0" x2="484.0" y2="103.6" class="stroke-b"/>
<line x1="487.8" y1="180.0" x2="487.8" y2="106.8" class="stroke-b"/>
<line x1="491.5" y1="180.0" x2="491.5" y2="109.2" class="stroke-b"/>
<line x1="495.2" y1="180.0" x2="495.2" y2="110.9" class="stroke-b"/>
<line x1="499.0" y1="180.0" x2="499.0" y2="111.8" class="stroke-b"/>
<line x1="502.8" y1="180.0" x2="502.8" y2="111.9" class="stroke-b"/>
<line x1="506.5" y1="180.0" x2="506.5" y2="110.7" class="stroke-b"/>
<line x1="510.2" y1="180.0" x2="510.2" y2="107.7" class="stroke-b"/>
<line x1="514.0" y1="180.0" x2="514.0" y2="102.4" class="stroke-b"/>
<line x1="517.8" y1="180.0" x2="517.8" y2="106.4" class="stroke-b"/>
<line x1="521.5" y1="180.0" x2="521.5" y2="116.3" class="stroke-b"/>
<line x1="525.2" y1="180.0" x2="525.2" y2="123.8" class="stroke-b"/>
<line x1="529.0" y1="180.0" x2="529.0" y2="129.7" class="stroke-b"/>
<line x1="532.8" y1="180.0" x2="532.8" y2="134.6" class="stroke-b"/>
<line x1="536.5" y1="180.0" x2="536.5" y2="138.8" class="stroke-b"/>
<line x1="540.2" y1="180.0" x2="540.2" y2="142.6" class="stroke-b"/>
<line x1="544.0" y1="180.0" x2="544.0" y2="146.0" class="stroke-b"/>
<line x1="547.8" y1="180.0" x2="547.8" y2="149.1" class="stroke-b"/>
<line x1="551.5" y1="180.0" x2="551.5" y2="151.9" class="stroke-b"/>
<line x1="555.2" y1="180.0" x2="555.2" y2="154.6" class="stroke-b"/>
<line x1="559.0" y1="180.0" x2="559.0" y2="157.1" class="stroke-b"/>
<line x1="562.8" y1="180.0" x2="562.8" y2="159.4" class="stroke-b"/>
<line x1="566.5" y1="180.0" x2="566.5" y2="161.6" class="stroke-b"/>
<line x1="570.2" y1="180.0" x2="570.2" y2="163.7" class="stroke-b"/>
<line x1="574.0" y1="180.0" x2="574.0" y2="165.7" class="stroke-b"/>
<polyline points="424.0,93.3 424.4,93.3 424.8,93.3 425.1,93.3 425.5,93.3 425.9,93.2 426.3,93.2 426.6,93.2 427.0,93.1 427.4,93.0 427.8,93.0 428.1,92.9 428.5,92.8 428.9,92.7 429.3,92.6 429.6,92.5 430.0,92.4 430.4,92.3 430.8,92.1 431.1,92.0 431.5,91.8 431.9,91.7 432.3,91.5 432.6,91.3 433.0,91.1 433.4,90.9 433.8,90.7 434.2,90.5 434.5,90.3 434.9,90.1 435.3,89.8 435.7,89.5 436.0,89.3 436.4,89.0 436.8,88.7 437.2,88.4 437.5,88.1 437.9,87.7 438.3,87.4 438.7,87.0 439.0,86.7 439.4,86.3 439.8,85.9 440.2,85.4 440.5,85.0 440.9,84.5 441.3,84.0 441.7,83.5 442.0,83.0 442.4,82.4 442.8,81.9 443.2,81.3 443.5,80.6 443.9,80.0 444.3,79.3 444.7,78.5 445.1,77.7 445.4,76.9 445.8,76.1 446.2,75.1 446.6,74.2 446.9,73.1 447.3,72.0 447.7,70.8 448.1,69.5 448.4,68.1 448.8,66.7 449.2,65.1 449.6,63.4 449.9,61.6 450.3,59.9 450.7,58.3 451.1,57.2 451.4,56.7 451.8,57.0 452.2,57.9 452.6,59.1 452.9,60.3 453.3,61.6 453.7,62.8 454.1,63.9 454.5,64.8 454.8,65.7 455.2,66.4 455.6,67.1 456.0,67.6 456.3,68.1 456.7,68.5 457.1,68.8 457.5,69.1 457.8,69.3 458.2,69.4 458.6,69.4 459.0,69.4 459.3,69.4 459.7,69.2 460.1,69.0 460.5,68.7 460.8,68.4 461.2,67.9 461.6,67.4 462.0,66.8 462.3,66.1 462.7,65.4 463.1,64.6 463.5,63.8 463.8,63.1 464.2,62.6 464.6,62.4 465.0,62.7 465.4,63.5 465.7,64.7 466.1,66.1 466.5,67.7 466.9,69.3 467.2,70.9 467.6,72.4 468.0,73.9 468.4,75.3 468.7,76.6 469.1,77.9 469.5,79.1 469.9,80.3 470.2,81.4 470.6,82.4 471.0,83.4 471.4,84.4 471.7,85.3 472.1,86.2 472.5,87.0 472.9,87.8 473.2,88.6 473.6,89.4 474.0,90.1 474.4,90.8 474.8,91.5 475.1,92.2 475.5,92.8 475.9,93.5 476.3,94.1 476.6,94.7 477.0,95.2 477.4,95.8 477.8,96.3 478.1,96.9 478.5,97.4 478.9,97.9 479.3,98.4 479.6,98.9 480.0,99.3 480.4,99.8 480.8,100.2 481.1,100.7 481.5,101.1 481.9,101.5 482.3,101.9 482.6,102.3 483.0,102.7 483.4,103.1 483.8,103.4 484.2,103.8 484.5,104.1 484.9,104.5 485.3,104.8 485.7,105.1 486.0,105.4 486.4,105.7 486.8,106.0 487.2,106.3 487.5,106.6 487.9,106.9 488.3,107.2 488.7,107.4 489.0,107.7 489.4,107.9 489.8,108.2 490.2,108.4 490.5,108.6 490.9,108.8 491.3,109.1 491.7,109.3 492.0,109.5 492.4,109.7 492.8,109.8 493.2,110.0 493.5,110.2 493.9,110.3 494.3,110.5 494.7,110.6 495.1,110.8 495.4,110.9 495.8,111.0 496.2,111.2 496.6,111.3 496.9,111.4 497.3,111.5 497.7,111.6 498.1,111.6 498.4,111.7 498.8,111.8 499.2,111.8 499.6,111.9 499.9,111.9 500.3,111.9 500.7,112.0 501.1,112.0 501.4,112.0 501.8,111.9 502.2,111.9 502.6,111.9 502.9,111.8 503.3,111.8 503.7,111.7 504.1,111.6 504.5,111.5 504.8,111.4 505.2,111.3 505.6,111.2 506.0,111.0 506.3,110.8 506.7,110.6 507.1,110.4 507.5,110.2 507.8,109.9 508.2,109.6 508.6,109.3 509.0,109.0 509.3,108.7 509.7,108.3 510.1,107.9 510.5,107.5 510.8,107.0 511.2,106.5 511.6,106.0 512.0,105.4 512.3,104.9 512.7,104.3 513.1,103.7 513.5,103.1 513.8,102.6 514.2,102.2 514.6,101.9 515.0,101.7 515.4,101.8 515.7,102.0 516.1,102.5 516.5,103.2 516.9,104.0 517.2,105.0 517.6,106.0 518.0,107.0 518.4,108.1 518.7,109.2 519.1,110.2 519.5,111.3 519.9,112.3 520.2,113.2 520.6,114.2 521.0,115.1 521.4,116.0 521.7,116.9 522.1,117.7 522.5,118.5 522.9,119.3 523.2,120.1 523.6,120.8 524.0,121.5 524.4,122.2 524.8,122.9 525.1,123.6 525.5,124.2 525.9,124.9 526.3,125.5 526.6,126.1 527.0,126.7 527.4,127.3 527.8,127.9 528.1,128.4 528.5,129.0 528.9,129.5 529.3,130.0 529.6,130.6 530.0,131.1 530.4,131.6 530.8,132.1 531.1,132.6 531.5,133.0 531.9,133.5 532.3,134.0 532.6,134.4 533.0,134.9 533.4,135.3 533.8,135.8 534.2,136.2 534.5,136.6 534.9,137.1 535.3,137.5 535.7,137.9 536.0,138.3 536.4,138.7 536.8,139.1 537.2,139.5 537.5,139.9 537.9,140.3 538.3,140.6 538.7,141.0 539.0,141.4 539.4,141.8 539.8,142.1 540.2,142.5 540.5,142.8 540.9,143.2 541.3,143.5 541.7,143.9 542.0,144.2 542.4,144.6 542.8,144.9 543.2,145.2 543.5,145.6 543.9,145.9 544.3,146.2 544.7,146.5 545.1,146.9 545.4,147.2 545.8,147.5 546.2,147.8 546.6,148.1 546.9,148.4 547.3,148.7 547.7,149.0 548.1,149.3 548.4,149.6 548.8,149.9 549.2,150.2 549.6,150.5 549.9,150.8 550.3,151.1 550.7,151.3 551.1,151.6 551.4,151.9 551.8,152.2 552.2,152.5 552.6,152.7 552.9,153.0 553.3,153.3 553.7,153.5 554.1,153.8 554.5,154.1 554.8,154.3 555.2,154.6 555.6,154.8 556.0,155.1 556.3,155.3 556.7,155.6 557.1,155.8 557.5,156.1 557.8,156.3 558.2,156.6 558.6,156.8 559.0,157.1 559.3,157.3 559.7,157.6 560.1,157.8 560.5,158.0 560.8,158.3 561.2,158.5 561.6,158.7 562.0,159.0 562.3,159.2 562.7,159.4 563.1,159.6 563.5,159.9 563.8,160.1 564.2,160.3 564.6,160.5 565.0,160.8 565.4,161.0 565.7,161.2 566.1,161.4 566.5,161.6 566.9,161.8 567.2,162.1 567.6,162.3 568.0,162.5 568.4,162.7 568.7,162.9 569.1,163.1 569.5,163.3 569.9,163.5 570.2,163.7 570.6,163.9 571.0,164.1 571.4,164.3 571.7,164.5 572.1,164.7 572.5,164.9 572.9,165.1 573.2,165.3 573.6,165.5 574.0,165.7" class="stroke-d dashed" fill="none"/>
<text x="115.0" y="34.0" class="fig-small fig-bold" text-anchor="middle">source</text>
<text x="307.0" y="34.0" class="fig-small fig-bold" text-anchor="middle">filter</text>
<text x="499.0" y="34.0" class="fig-small fig-bold" text-anchor="middle">output</text>
<text x="200.0" y="120.0" class="fig-bold" text-anchor="middle">×</text>
<text x="392.0" y="120.0" class="fig-bold" text-anchor="middle">=</text>
<text x="40.0" y="226.0" class="fig-small" text-anchor="start">frequency (Hz); vertical axes in dB. Output line height = source line + filter curve at that harmonic.</text>
</svg>
Figure 1. The source-filter model in numbers, for the vowel /ɑ/ at F0 = 100 Hz. The pulse train puts equal power on every harmonic, the resonators add their gain at each harmonic's frequency, and the product keeps the pitch of the source and the shape of the filter. The tallest harmonic is the 7th at 700 Hz, next to the first formant at 730 Hz.
```

Why not build a vowel by adding up sinusoids at the harmonics? It works, but you must choose an amplitude for every harmonic, and a new pitch moves every harmonic, so you must choose them all again. The source-filter view writes the rule once. The filter is a curve over frequency, and each harmonic takes whatever height the curve has at its frequency, wherever the harmonics land.

## The source: a pulse train

The simplest voiced source is an *impulse train*: a 1 every period and 0 elsewhere. At 16 kHz with $F_0 = 100$ Hz, the period is $16000 / 100 = 160$ samples, and the impulses sit at samples 0, 160, 320, and so on. For $F_0 = 150$ Hz the period is 106.67 samples, which is not a whole number, so you round each position separately: 0, 107, 213, 320, 427. Rounding the period once, to 107, would put the fifth impulse at 428 and the tenth at 963 instead of 960, and the pitch would drift.

In code, the positions are `np.round(np.arange(0, n_samples, sample_rate / f0))`. Because the positions are exact multiples of the period, one second of the 100 Hz train has 100 impulses in step, so its spectrum has lines at 0, 100, 200, ... Hz and nothing in between. The check on this chapter's exercise tests exactly that.

## The filter: a tube with resonances

Model the vocal tract as a uniform tube, closed at the vocal folds and open at the lips. Sound inside bounces back and forth, and the tube rings at frequencies where the standing wave fits. Pressure peaks at the closed end and drops to zero at the open end, so an odd number of quarter wavelengths fits:

$$
F_n = \frac{(2n - 1)\,c}{4L}
$$

Read it aloud: the $n$-th resonance is an odd multiple, $2n - 1$, of the speed of sound $c$ divided by four times the tube's length $L$. With $L = 17.5$ cm, typical for an adult man, and $c = 350$ m/s in warm, moist air, the resonances are 500, 1500, and 2500 Hz, as in Figure 2. That is close to the neutral vowel in "sofa".

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="A uniform tube closed at the glottis and open at the lips, with the first three resonances drawn as standing pressure waves of 500, 1500, and 2500 hertz">
<rect x="70" y="48" width="480" height="44" class="fill-sunk stroke-muted"/>
<text x="66.0" y="40.0" class="fig-small" text-anchor="start">glottis (closed end)</text>
<text x="554.0" y="40.0" class="fig-small" text-anchor="end">lips (open end)</text>
<line x1="70" y1="152" x2="550" y2="152" class="stroke-grid"/>
<polyline points="70.0,130.0 74.0,130.0 78.1,130.0 82.1,130.0 86.1,130.0 90.2,130.0 94.2,130.1 98.2,130.1 102.3,130.1 106.3,130.2 110.3,130.2 114.4,130.2 118.4,130.3 122.4,130.3 126.5,130.4 130.5,130.4 134.5,130.5 138.6,130.6 142.6,130.6 146.6,130.7 150.7,130.8 154.7,130.8 158.7,130.9 162.8,131.0 166.8,131.1 170.8,131.2 174.9,131.3 178.9,131.4 182.9,131.5 187.0,131.6 191.0,131.7 195.0,131.8 199.1,131.9 203.1,132.1 207.1,132.2 211.2,132.3 215.2,132.4 219.2,132.6 223.3,132.7 227.3,132.9 231.3,133.0 235.4,133.1 239.4,133.3 243.4,133.4 247.5,133.6 251.5,133.8 255.5,133.9 259.6,134.1 263.6,134.3 267.6,134.4 271.7,134.6 275.7,134.8 279.7,135.0 283.8,135.2 287.8,135.4 291.8,135.5 295.9,135.7 299.9,135.9 303.9,136.1 308.0,136.3 312.0,136.5 316.1,136.8 320.1,137.0 324.1,137.2 328.2,137.4 332.2,137.6 336.2,137.8 340.3,138.1 344.3,138.3 348.3,138.5 352.4,138.7 356.4,139.0 360.4,139.2 364.5,139.4 368.5,139.7 372.5,139.9 376.6,140.2 380.6,140.4 384.6,140.7 388.7,140.9 392.7,141.2 396.7,141.4 400.8,141.7 404.8,141.9 408.8,142.2 412.9,142.5 416.9,142.7 420.9,143.0 425.0,143.2 429.0,143.5 433.0,143.8 437.1,144.1 441.1,144.3 445.1,144.6 449.2,144.9 453.2,145.1 457.2,145.4 461.3,145.7 465.3,146.0 469.3,146.3 473.4,146.5 477.4,146.8 481.4,147.1 485.5,147.4 489.5,147.7 493.5,148.0 497.6,148.2 501.6,148.5 505.6,148.8 509.7,149.1 513.7,149.4 517.7,149.7 521.8,150.0 525.8,150.3 529.8,150.5 533.9,150.8 537.9,151.1 541.9,151.4 546.0,151.7 550.0,152.0" class="stroke-a" fill="none"/>
<text x="64.0" y="156.0" class="fig-small" text-anchor="end">n=1</text>
<text x="80.0" y="120.0" class="fig-small" text-anchor="start">F1 = 500 Hz: a quarter wavelength fits</text>
<line x1="70" y1="214" x2="550" y2="214" class="stroke-grid"/>
<polyline points="70.0,192.0 74.0,192.0 78.1,192.1 82.1,192.2 86.1,192.3 90.2,192.4 94.2,192.6 98.2,192.8 102.3,193.1 106.3,193.4 110.3,193.7 114.4,194.1 118.4,194.4 122.4,194.9 126.5,195.3 130.5,195.8 134.5,196.3 138.6,196.8 142.6,197.4 146.6,197.9 150.7,198.5 154.7,199.2 158.7,199.8 162.8,200.5 166.8,201.2 170.8,201.9 174.9,202.7 178.9,203.4 182.9,204.2 187.0,205.0 191.0,205.8 195.0,206.6 199.1,207.4 203.1,208.3 207.1,209.1 211.2,210.0 215.2,210.8 219.2,211.7 223.3,212.5 227.3,213.4 231.3,214.3 235.4,215.2 239.4,216.0 243.4,216.9 247.5,217.8 251.5,218.6 255.5,219.5 259.6,220.3 263.6,221.1 267.6,221.9 271.7,222.8 275.7,223.5 279.7,224.3 283.8,225.1 287.8,225.8 291.8,226.6 295.9,227.3 299.9,227.9 303.9,228.6 308.0,229.2 312.0,229.9 316.1,230.5 320.1,231.0 324.1,231.6 328.2,232.1 332.2,232.6 336.2,233.0 340.3,233.4 344.3,233.8 348.3,234.2 352.4,234.5 356.4,234.8 360.4,235.1 364.5,235.3 368.5,235.5 372.5,235.7 376.6,235.8 380.6,235.9 384.6,236.0 388.7,236.0 392.7,236.0 396.7,236.0 400.8,235.9 404.8,235.8 408.8,235.6 412.9,235.4 416.9,235.2 420.9,235.0 425.0,234.7 429.0,234.4 433.0,234.1 437.1,233.7 441.1,233.3 445.1,232.9 449.2,232.4 453.2,231.9 457.2,231.4 461.3,230.8 465.3,230.3 469.3,229.7 473.4,229.0 477.4,228.4 481.4,227.7 485.5,227.0 489.5,226.3 493.5,225.6 497.6,224.8 501.6,224.1 505.6,223.3 509.7,222.5 513.7,221.7 517.7,220.9 521.8,220.0 525.8,219.2 529.8,218.3 533.9,217.5 537.9,216.6 541.9,215.7 546.0,214.9 550.0,214.0" class="stroke-b" fill="none"/>
<text x="64.0" y="218.0" class="fig-small" text-anchor="end">n=2</text>
<text x="80.0" y="182.0" class="fig-small" text-anchor="start">F2 = 1500 Hz: three quarters</text>
<line x1="70" y1="276" x2="550" y2="276" class="stroke-grid"/>
<polyline points="70.0,254.0 74.0,254.0 78.1,254.2 82.1,254.4 86.1,254.8 90.2,255.2 94.2,255.7 98.2,256.3 102.3,257.0 106.3,257.8 110.3,258.6 114.4,259.5 118.4,260.5 122.4,261.6 126.5,262.7 130.5,263.9 134.5,265.2 138.6,266.5 142.6,267.8 146.6,269.1 150.7,270.5 154.7,272.0 158.7,273.4 162.8,274.8 166.8,276.3 170.8,277.7 174.9,279.2 178.9,280.6 182.9,282.0 187.0,283.4 191.0,284.8 195.0,286.1 199.1,287.3 203.1,288.6 207.1,289.7 211.2,290.8 215.2,291.9 219.2,292.8 223.3,293.7 227.3,294.6 231.3,295.3 235.4,295.9 239.4,296.5 243.4,297.0 247.5,297.4 251.5,297.7 255.5,297.9 259.6,298.0 263.6,298.0 267.6,297.9 271.7,297.7 275.7,297.4 279.7,297.1 283.8,296.6 287.8,296.1 291.8,295.4 295.9,294.7 299.9,293.9 303.9,293.0 308.0,292.1 312.0,291.0 316.1,289.9 320.1,288.8 324.1,287.6 328.2,286.3 332.2,285.0 336.2,283.7 340.3,282.3 344.3,280.9 348.3,279.5 352.4,278.0 356.4,276.6 360.4,275.1 364.5,273.7 368.5,272.2 372.5,270.8 376.6,269.4 380.6,268.1 384.6,266.7 388.7,265.4 392.7,264.2 396.7,263.0 400.8,261.8 404.8,260.8 408.8,259.7 412.9,258.8 416.9,257.9 420.9,257.1 425.0,256.4 429.0,255.8 433.0,255.3 437.1,254.8 441.1,254.5 445.1,254.2 449.2,254.1 453.2,254.0 457.2,254.0 461.3,254.2 465.3,254.4 469.3,254.7 473.4,255.1 477.4,255.6 481.4,256.2 485.5,256.9 489.5,257.6 493.5,258.4 497.6,259.4 501.6,260.3 505.6,261.4 509.7,262.5 513.7,263.7 517.7,264.9 521.8,266.2 525.8,267.5 529.8,268.9 533.9,270.3 537.9,271.7 541.9,273.1 546.0,274.5 550.0,276.0" class="stroke-d" fill="none"/>
<text x="64.0" y="280.0" class="fig-small" text-anchor="end">n=3</text>
<text x="80.0" y="244.0" class="fig-small" text-anchor="start">F3 = 2500 Hz: five quarters</text>
<text x="310.0" y="74.0" class="fig-small" text-anchor="middle">uniform tube, L = 17.5 cm</text>
</svg>
Figure 2. Pressure standing waves in a tube closed at the left and open at the right. Pressure peaks at the closed end and drops to zero at the open end, so an odd number of quarter wavelengths fits: 1, 3, and 5. With c = 350 m/s and L = 17.5 cm that gives 500, 1500, and 2500 Hz.
```

Moving the tongue narrows the tube in one place and shifts each formant. The following table lists average formants for three vowels spoken by adult men, measured by Peterson and Barney in 1952.

| Vowel | Example | $F_1$ (Hz) | $F_2$ (Hz) | $F_3$ (Hz) |
|---|---|---|---|---|
| /i/ | heed | 270 | 2290 | 3010 |
| /ɑ/ | father | 730 | 1090 | 2440 |
| /u/ | who'd | 300 | 870 | 2240 |

$F_1$ falls as the tongue rises: /i/ and /u/ are high vowels, and /ɑ/ is low. $F_2$ rises as the tongue moves forward: /i/ is front, and /u/ is back.

## One formant is one resonator

A uniform tube gives only the neutral vowel. To make other vowels, you need one adjustable peak per formant. One formant is one pair of complex-conjugate poles. Put a pole at radius $r$ and angle $\theta$, and its mirror at $-\theta$. The angle sets the peak frequency, and the radius sets how sharp the peak is: the closer $r$ is to 1, the narrower the peak. A formant at $F$ Hz with a *bandwidth* of $B$ Hz, meaning the width of the peak 3 dB below its top, at sample rate $f_s$, uses the following:

$$
\begin{aligned}
r &= e^{-\pi B / f_s} \\
\theta &= \frac{2\pi F}{f_s} \\
y[n] &= b_0\,x[n] + 2r\cos\theta\; y[n-1] - r^2\, y[n-2]
\end{aligned}
$$

Read it aloud: the radius shrinks exponentially with bandwidth over sample rate, and the angle is the formant's share of a full turn of the unit circle. Each output sample is a scaled input plus two feedback terms. Dennis Klatt's synthesizer sets $b_0 = 1 - 2r\cos\theta + r^2$, which makes the gain at 0 Hz exactly 1, so a cascade of resonators does not change the overall level.

```python
a = np.array([1.0, -2 * r * np.cos(theta), r * r])
b = np.array([a.sum()])
y = scipy.signal.lfilter(b, a, x)
```

For $F = 500$ Hz and $B = 100$ Hz at 16 kHz: $r = e^{-\pi \cdot 100/16000} = 0.980557$ and $\theta = 0.19635$ rad, with $\cos\theta = 0.980785$. So $a = [1, -1.923431, 0.961491]$ and $b_0 = 0.038060$. The gain is 1 at 0 Hz and 5.06, or 14.1 dB, at 500 Hz. Figure 3 shows the poles and the gain curve.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="The pole pair of a 500 hertz formant with 100 hertz bandwidth inside the unit circle, and the gain curve it produces">
<circle cx="120" cy="125" r="85" fill="none" class="stroke-muted"/>
<line x1="25" y1="125" x2="215" y2="125" class="stroke-grid"/><line x1="120" y1="30" x2="120" y2="220" class="stroke-grid"/>
<line x1="120" y1="125" x2="201.7" y2="108.7" class="stroke-muted dashed"/>
<circle cx="201.7" cy="108.7" r="5" class="fill-b"/>
<line x1="120" y1="125" x2="201.7" y2="141.3" class="stroke-muted dashed"/>
<circle cx="201.7" cy="141.3" r="5" class="fill-b"/>
<text x="35.0" y="26.0" class="fig-small" text-anchor="start">unit circle</text>
<text x="195.7" y="96.7" class="fig-small" text-anchor="end">r = 0.9806</text>
<text x="35.0" y="234.0" class="fig-small" text-anchor="start">angle θ = 0.1963 rad = 2π·500/16000</text>
<line x1="330.0" y1="220.0" x2="580.0" y2="220.0" class="stroke-grid"/>
<text x="324.0" y="224.0" class="fig-small" text-anchor="end">-30</text>
<line x1="330.0" y1="184.0" x2="580.0" y2="184.0" class="stroke-grid"/>
<text x="324.0" y="188.0" class="fig-small" text-anchor="end">-20</text>
<line x1="330.0" y1="148.0" x2="580.0" y2="148.0" class="stroke-grid"/>
<text x="324.0" y="152.0" class="fig-small" text-anchor="end">-10</text>
<line x1="330.0" y1="112.0" x2="580.0" y2="112.0" class="stroke-grid"/>
<text x="324.0" y="116.0" class="fig-small" text-anchor="end">0</text>
<line x1="330.0" y1="76.0" x2="580.0" y2="76.0" class="stroke-grid"/>
<text x="324.0" y="80.0" class="fig-small" text-anchor="end">10</text>
<line x1="330.0" y1="40.0" x2="580.0" y2="40.0" class="stroke-grid"/>
<text x="324.0" y="44.0" class="fig-small" text-anchor="end">20</text>
<line x1="330.0" y1="220.0" x2="580.0" y2="220.0" class="stroke-muted"/>
<line x1="330.0" y1="40.0" x2="330.0" y2="220.0" class="stroke-muted"/>
<line x1="330.0" y1="220.0" x2="330.0" y2="225.0" class="stroke-muted"/>
<text x="330.0" y="239.0" class="fig-small" text-anchor="middle">0</text>
<line x1="413.3" y1="220.0" x2="413.3" y2="225.0" class="stroke-muted"/>
<text x="413.3" y="239.0" class="fig-small" text-anchor="middle">500</text>
<line x1="496.7" y1="220.0" x2="496.7" y2="225.0" class="stroke-muted"/>
<text x="496.7" y="239.0" class="fig-small" text-anchor="middle">1000</text>
<line x1="580.0" y1="220.0" x2="580.0" y2="225.0" class="stroke-muted"/>
<text x="580.0" y="239.0" class="fig-small" text-anchor="middle">1500</text>
<text x="455.0" y="258.0" text-anchor="middle">frequency (Hz)</text>
<text x="300.0" y="30.0" text-anchor="start">gain (dB)</text>
<polyline points="330.0,112.0 330.5,112.0 331.0,112.0 331.5,112.0 332.0,112.0 332.5,112.0 333.0,112.0 333.5,111.9 334.0,111.9 334.5,111.9 335.0,111.9 335.5,111.9 336.0,111.8 336.5,111.8 337.0,111.8 337.5,111.8 338.0,111.7 338.5,111.7 339.0,111.6 339.5,111.6 340.0,111.6 340.5,111.5 341.0,111.5 341.5,111.4 342.0,111.4 342.5,111.3 343.0,111.2 343.5,111.2 344.0,111.1 344.5,111.1 345.0,111.0 345.5,110.9 346.0,110.9 346.5,110.8 347.0,110.7 347.5,110.6 348.0,110.5 348.5,110.5 349.0,110.4 349.5,110.3 350.0,110.2 350.5,110.1 351.0,110.0 351.5,109.9 352.0,109.8 352.5,109.7 353.0,109.6 353.5,109.5 354.0,109.4 354.5,109.2 355.1,109.1 355.6,109.0 356.1,108.9 356.6,108.8 357.1,108.6 357.6,108.5 358.1,108.4 358.6,108.2 359.1,108.1 359.6,107.9 360.1,107.8 360.6,107.6 361.1,107.5 361.6,107.3 362.1,107.2 362.6,107.0 363.1,106.8 363.6,106.6 364.1,106.5 364.6,106.3 365.1,106.1 365.6,105.9 366.1,105.7 366.6,105.5 367.1,105.3 367.6,105.1 368.1,104.9 368.6,104.7 369.1,104.5 369.6,104.3 370.1,104.1 370.6,103.9 371.1,103.6 371.6,103.4 372.1,103.2 372.6,102.9 373.1,102.7 373.6,102.4 374.1,102.1 374.6,101.9 375.1,101.6 375.6,101.3 376.1,101.1 376.6,100.8 377.1,100.5 377.6,100.2 378.1,99.9 378.6,99.6 379.1,99.3 379.6,99.0 380.1,98.6 380.6,98.3 381.1,98.0 381.6,97.6 382.1,97.3 382.6,96.9 383.1,96.6 383.6,96.2 384.1,95.8 384.6,95.4 385.1,95.0 385.6,94.6 386.1,94.2 386.6,93.8 387.1,93.4 387.6,92.9 388.1,92.5 388.6,92.0 389.1,91.6 389.6,91.1 390.1,90.6 390.6,90.1 391.1,89.6 391.6,89.1 392.1,88.6 392.6,88.0 393.1,87.5 393.6,86.9 394.1,86.3 394.6,85.7 395.1,85.1 395.6,84.5 396.1,83.9 396.6,83.3 397.1,82.6 397.6,81.9 398.1,81.3 398.6,80.6 399.1,79.8 399.6,79.1 400.1,78.4 400.6,77.6 401.1,76.9 401.6,76.1 402.1,75.3 402.6,74.5 403.1,73.6 403.6,72.8 404.1,72.0 404.6,71.1 405.2,70.3 405.7,69.5 406.2,68.6 406.7,67.8 407.2,67.0 407.7,66.2 408.2,65.4 408.7,64.7 409.2,64.0 409.7,63.4 410.2,62.8 410.7,62.3 411.2,61.9 411.7,61.6 412.2,61.4 412.7,61.3 413.2,61.3 413.7,61.4 414.2,61.6 414.7,61.9 415.2,62.4 415.7,62.9 416.2,63.5 416.7,64.2 417.2,65.0 417.7,65.8 418.2,66.7 418.7,67.7 419.2,68.6 419.7,69.6 420.2,70.6 420.7,71.6 421.2,72.6 421.7,73.7 422.2,74.7 422.7,75.7 423.2,76.7 423.7,77.8 424.2,78.8 424.7,79.7 425.2,80.7 425.7,81.7 426.2,82.6 426.7,83.6 427.2,84.5 427.7,85.4 428.2,86.3 428.7,87.2 429.2,88.0 429.7,88.9 430.2,89.7 430.7,90.6 431.2,91.4 431.7,92.2 432.2,92.9 432.7,93.7 433.2,94.5 433.7,95.2 434.2,96.0 434.7,96.7 435.2,97.4 435.7,98.1 436.2,98.8 436.7,99.5 437.2,100.1 437.7,100.8 438.2,101.4 438.7,102.1 439.2,102.7 439.7,103.3 440.2,104.0 440.7,104.6 441.2,105.2 441.7,105.8 442.2,106.3 442.7,106.9 443.2,107.5 443.7,108.0 444.2,108.6 444.7,109.1 445.2,109.7 445.7,110.2 446.2,110.7 446.7,111.3 447.2,111.8 447.7,112.3 448.2,112.8 448.7,113.3 449.2,113.8 449.7,114.3 450.2,114.7 450.7,115.2 451.2,115.7 451.7,116.2 452.2,116.6 452.7,117.1 453.2,117.5 453.7,118.0 454.2,118.4 454.7,118.9 455.3,119.3 455.8,119.7 456.3,120.2 456.8,120.6 457.3,121.0 457.8,121.4 458.3,121.8 458.8,122.2 459.3,122.6 459.8,123.0 460.3,123.4 460.8,123.8 461.3,124.2 461.8,124.6 462.3,125.0 462.8,125.4 463.3,125.7 463.8,126.1 464.3,126.5 464.8,126.9 465.3,127.2 465.8,127.6 466.3,127.9 466.8,128.3 467.3,128.6 467.8,129.0 468.3,129.3 468.8,129.7 469.3,130.0 469.8,130.4 470.3,130.7 470.8,131.1 471.3,131.4 471.8,131.7 472.3,132.0 472.8,132.4 473.3,132.7 473.8,133.0 474.3,133.3 474.8,133.7 475.3,134.0 475.8,134.3 476.3,134.6 476.8,134.9 477.3,135.2 477.8,135.5 478.3,135.8 478.8,136.1 479.3,136.4 479.8,136.7 480.3,137.0 480.8,137.3 481.3,137.6 481.8,137.9 482.3,138.2 482.8,138.5 483.3,138.8 483.8,139.0 484.3,139.3 484.8,139.6 485.3,139.9 485.8,140.2 486.3,140.4 486.8,140.7 487.3,141.0 487.8,141.3 488.3,141.5 488.8,141.8 489.3,142.1 489.8,142.3 490.3,142.6 490.8,142.9 491.3,143.1 491.8,143.4 492.3,143.6 492.8,143.9 493.3,144.1 493.8,144.4 494.3,144.6 494.8,144.9 495.3,145.2 495.8,145.4 496.3,145.6 496.8,145.9 497.3,146.1 497.8,146.4 498.3,146.6 498.8,146.9 499.3,147.1 499.8,147.3 500.3,147.6 500.8,147.8 501.3,148.1 501.8,148.3 502.3,148.5 502.8,148.8 503.3,149.0 503.8,149.2 504.3,149.5 504.8,149.7 505.4,149.9 505.9,150.1 506.4,150.4 506.9,150.6 507.4,150.8 507.9,151.0 508.4,151.2 508.9,151.5 509.4,151.7 509.9,151.9 510.4,152.1 510.9,152.3 511.4,152.6 511.9,152.8 512.4,153.0 512.9,153.2 513.4,153.4 513.9,153.6 514.4,153.8 514.9,154.0 515.4,154.2 515.9,154.5 516.4,154.7 516.9,154.9 517.4,155.1 517.9,155.3 518.4,155.5 518.9,155.7 519.4,155.9 519.9,156.1 520.4,156.3 520.9,156.5 521.4,156.7 521.9,156.9 522.4,157.1 522.9,157.3 523.4,157.5 523.9,157.7 524.4,157.9 524.9,158.1 525.4,158.2 525.9,158.4 526.4,158.6 526.9,158.8 527.4,159.0 527.9,159.2 528.4,159.4 528.9,159.6 529.4,159.8 529.9,160.0 530.4,160.1 530.9,160.3 531.4,160.5 531.9,160.7 532.4,160.9 532.9,161.1 533.4,161.2 533.9,161.4 534.4,161.6 534.9,161.8 535.4,162.0 535.9,162.1 536.4,162.3 536.9,162.5 537.4,162.7 537.9,162.8 538.4,163.0 538.9,163.2 539.4,163.4 539.9,163.5 540.4,163.7 540.9,163.9 541.4,164.1 541.9,164.2 542.4,164.4 542.9,164.6 543.4,164.7 543.9,164.9 544.4,165.1 544.9,165.3 545.4,165.4 545.9,165.6 546.4,165.8 546.9,165.9 547.4,166.1 547.9,166.2 548.4,166.4 548.9,166.6 549.4,166.7 549.9,166.9 550.4,167.1 550.9,167.2 551.4,167.4 551.9,167.6 552.4,167.7 552.9,167.9 553.4,168.0 553.9,168.2 554.4,168.3 554.9,168.5 555.5,168.7 556.0,168.8 556.5,169.0 557.0,169.1 557.5,169.3 558.0,169.4 558.5,169.6 559.0,169.8 559.5,169.9 560.0,170.1 560.5,170.2 561.0,170.4 561.5,170.5 562.0,170.7 562.5,170.8 563.0,171.0 563.5,171.1 564.0,171.3 564.5,171.4 565.0,171.6 565.5,171.7 566.0,171.9 566.5,172.0 567.0,172.2 567.5,172.3 568.0,172.5 568.5,172.6 569.0,172.7 569.5,172.9 570.0,173.0 570.5,173.2 571.0,173.3 571.5,173.5 572.0,173.6 572.5,173.8 573.0,173.9 573.5,174.0 574.0,174.2 574.5,174.3 575.0,174.5 575.5,174.6 576.0,174.7 576.5,174.9 577.0,175.0 577.5,175.2 578.0,175.3 578.5,175.4 579.0,175.6 579.5,175.7 580.0,175.9" class="stroke-b" fill="none"/>
<circle cx="413.3" cy="61.3" r="5" class="fill-b"/>
<circle cx="330.0" cy="112.0" r="5" class="fill-a"/>
<text x="416.7" y="50.5" class="fig-small" text-anchor="start">14.1 dB at 500 Hz</text>
<text x="335.0" y="126.4" class="fig-small" text-anchor="start">0 dB at 0 Hz</text>
</svg>
Figure 3. One formant. Left: its two poles sit at radius 0.9806 and angle ±0.1963 rad, close to the unit circle because the bandwidth is narrow. Right: the gain is 0 dB at 0 Hz by design and 14.1 dB at 500 Hz, which is a factor of 5.06 in amplitude.
```

A vowel synthesizer is an impulse train through three of these in a row, one per formant. Each resonator multiplies the spectrum by its own peak, so the cascade multiplies the three peaks together. The following demo is that synthesizer.

```demo
{"id": "vowel-formants", "caption": "Drag on the chart to move the first two formants between vowels, then change the pitch of the source and hear the vowel stay the same as its harmonics spread apart."}
```

> [!KEY]
> A voiced vowel is an impulse train at $F_0$ passed through one two-pole resonator per formant. The pulse train sets the pitch, the resonators set the vowel, and changing one leaves the other alone.

> [!RECAP]
> - Speech is a source, pulses or noise, through a vocal-tract filter, and the formants are the filter's peaks.
> - A pulse train at $F_0$ has harmonics at $F_0, 2F_0, 3F_0, \dots$, and the filter sets their heights.
> - A uniform tube of length $L$ resonates at $(2n-1)c/4L$: 500, 1500, and 2500 Hz for 17.5 cm.
> - A formant at $F$ with bandwidth $B$ is a pole pair at radius $e^{-\pi B/f_s}$ and angle $2\pi F/f_s$.

## Build it step by step

The checks test the three functions in this order.

1. **`impulse_train(f0, sample_rate, n_samples)`** returns `n_samples` zeros with a one at the nearest sample to every multiple of the period `sample_rate / f0`. The usual mistake is truncating the period to an integer first, which drifts the pitch.
2. **`resonator(frequency, bandwidth, sample_rate)`** returns `(b, a)` from the formulas. The usual mistake is $r = e^{-2\pi B / f_s}$, which doubles the bandwidth.
3. **`synthesize_vowel(f0, formants, bandwidths, sample_rate, seconds)`** builds the impulse train, runs `lfilter` once per resonator, and scales the peak to 0.5. The usual mistake is forgetting the scale, so the peak depends on the pitch and the formants.

## Where this breaks

**Impulses are not glottal pulses.** A real glottal pulse is smooth, so its spectrum falls off at about 12 dB per octave, and an impulse train is flat. The synthetic vowel sounds buzzier and brighter than a voice. Pre-emphasis on the analysis side compensates in the other direction.

**Nasals and zeros.** In "m" and "n", the nasal cavity adds antiresonances, which are zeros. A cascade of all-pole resonators cannot make them.

**Fixed formants.** Real formants glide from one sound to the next. This chapter's vowels hold still, which is why they sound like a sustained note and not like speech.

**A crude tube.** The uniform tube gives 500, 1500, and 2500 Hz, close to the neutral vowel. Real vocal tracts are not uniform, and the table's averages come from many speakers.

> [!WARNING]
> A resonator's bandwidth is in hertz, and its pole radius depends on the sample rate too. The same 100 Hz bandwidth gives a different $r$ at 8 kHz than at 16 kHz, so pass the sample rate that the signal really has.

## The exercise

You build `impulse_train`, `resonator`, and `synthesize_vowel`. Press **Run** to hear /ɑ/, /i/, and /u/, see each vowel's harmonics sit under the three resonator peaks, and hear one vowel at two pitches.

The checks verify the impulse positions and the harmonic lines of the pulse train. They check the resonator's coefficients, 0 Hz gain, peak, and pole radius, and that a wider bandwidth lowers the peak. They compare your vowel with an independent cascade, and they confirm that your vowel keeps its formants when its pitch changes.
