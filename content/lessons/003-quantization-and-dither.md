---
title: Quantization and dither
slug: 003-quantization-and-dither
part: "Part 1 — Sound and signals"
summary: Round samples to N bits, predict the noise with the 6.02N + 1.76 dB rule, and use dither so the noise stops following the signal.
minutes: 40
gpu: false
objectives:
  - Quantize a signal to N bits by hand and in numpy, including the clipped top code.
  - Derive and measure the 6.02N + 1.76 dB signal-to-noise rule, and say what happens to it for a quiet signal.
  - Explain why rounding a quiet signal creates distortion rather than noise.
  - Add triangular dither, and state what it buys and what it costs.
prereqs:
  - 001-what-sound-is
  - 002-sampling-and-aliasing
exercise: 003-quantization-and-dither
---

# Quantization and dither

You save a plucked note at 6 bits instead of 16. Two things go wrong. The note sounds buzzy, as if a harmonic is hiding in the grit. And when the note fades, it does not fade out: it stops abruptly, well before the original reaches silence. Both come from one fact. A file cannot store a real number, so every sample gets rounded to the nearest of a few levels.

This chapter answers two questions about that rounding. How much noise does it add, and how does the amount depend on the number of bits and on how loud the signal is? And why does the noise sometimes stop sounding like noise at all? The second answer leads to **dither**, a small amount of deliberate noise that makes the first answer true again.

## What you already know

- A level in dBFS is $20 \log_{10}$ of an amplitude, and a power ratio in decibels is $10 \log_{10}$ of the ratio.
- The RMS of a sine of amplitude $A$ is $A / \sqrt{2}$, so its mean square, its power, is $A^2 / 2$.
- A sample is a number in $[-1, 1)$, with full scale at 1.0.
- `np.round` and `np.clip` round and cut off whole arrays at once.

## Round to the nearest level

A **quantizer** replaces each sample by the nearest value on a fixed grid. With $N$ bits there are $2^N$ levels. For signals in $[-1, 1)$, the levels are a **step** $\Delta$ apart:

$$
\Delta = \frac{2}{2^N} = 2^{1-N}
$$

The integer codes run from $-2^{N-1}$ to $2^{N-1} - 1$. So the lowest level is $-1$, and the highest is one step below $+1$, because a two's-complement integer has one more negative value than positive. The quantizer is then divide, round, clip, and multiply back:

$$
Q(x) = \Delta \cdot \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{x}{\Delta}\right),\, -2^{N-1},\, 2^{N-1} - 1\right)
$$

Read it aloud: express the sample in steps, round to a whole number of steps, cut the number to the codes that exist, and convert back to a level. In numpy:

```python
step = 2.0 ** (1 - bits)
codes = np.clip(np.round(x / step), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
y = codes * step
```

### Work a 3-bit example by hand

With 3 bits, $\Delta = 0.25$, and the eight levels are $-1, -0.75, \dots, 0.75$. Take the samples 0.3, $-0.62$, 0.9, 0.05, and 0.99.

- Divided by the step: 1.2, $-2.48$, 3.6, 0.2, and 3.96.
- Rounded: 1, $-2$, 4, 0, and 4.
- Clipped to $[-4, 3]$: 1, $-2$, 3, 0, and 3.
- Times the step: 0.25, $-0.5$, 0.75, 0, and 0.75.

The errors, input minus output, are 0.05, $-0.12$, 0.15, 0.05, and 0.24. Rounding never costs more than half a step, 0.125 here. The two larger errors, 0.15 and 0.24, come from the samples 0.9 and 0.99, which sit above the top level and were clipped. Figure 1 draws this staircase and the five samples on it.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A 3-bit quantizer: a staircase of eight levels 0.25 apart. The samples 0.3, minus 0.62, 0.9, 0.05, and 0.99 land on 0.25, minus 0.5, 0.75, 0, and 0.75">
<line x1="64" y1="237.5" x2="576" y2="237.5" class="stroke-grid"/>
<text x="58" y="241.5" class="fig-small" text-anchor="end">-1</text>
<line x1="64" y1="185.3" x2="576" y2="185.3" class="stroke-grid"/>
<text x="58" y="189.3" class="fig-small" text-anchor="end">-0.5</text>
<line x1="64" y1="133.0" x2="576" y2="133.0" class="stroke-grid"/>
<text x="58" y="137.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="80.7" x2="576" y2="80.7" class="stroke-grid"/>
<text x="58" y="84.7" class="fig-small" text-anchor="end">0.5</text>
<line x1="64.0" y1="248" x2="64.0" y2="252" class="stroke-muted"/>
<text x="64.0" y="265" class="fig-small" text-anchor="middle">-1</text>
<line x1="192.0" y1="248" x2="192.0" y2="252" class="stroke-muted"/>
<text x="192.0" y="265" class="fig-small" text-anchor="middle">-0.5</text>
<line x1="320.0" y1="248" x2="320.0" y2="252" class="stroke-muted"/>
<text x="320.0" y="265" class="fig-small" text-anchor="middle">0</text>
<line x1="448.0" y1="248" x2="448.0" y2="252" class="stroke-muted"/>
<text x="448.0" y="265" class="fig-small" text-anchor="middle">0.5</text>
<line x1="576.0" y1="248" x2="576.0" y2="252" class="stroke-muted"/>
<text x="576.0" y="265" class="fig-small" text-anchor="middle">1</text>
<line x1="64" y1="248" x2="576" y2="248" class="stroke-muted"/>
<line x1="64" y1="18" x2="64" y2="248" class="stroke-muted"/>
<text x="320.0" y="292" class="fig-small" text-anchor="middle">input sample</text>
<text x="14" y="133.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 133.0)">output of the 3-bit quantizer</text>
<polyline points="64.0,237.5 576.0,28.5" class="stroke-muted dashed" fill="none"/>
<polyline points="64.0,237.5 64.0,237.5 96.0,237.5 96.0,211.4 160.0,211.4 160.0,185.3 224.0,185.3 224.0,159.1 288.0,159.1 288.0,133.0 352.0,133.0 352.0,106.9 416.0,106.9 416.0,80.7 480.0,80.7 480.0,54.6 544.0,54.6 576.0,54.6" class="stroke-b" fill="none" stroke-width="2"/>
<circle cx="396.8" cy="106.9" r="4.5" class="fill-a"/>
<circle cx="161.3" cy="185.3" r="4.5" class="fill-a"/>
<circle cx="550.4" cy="54.6" r="4.5" class="fill-a"/>
<circle cx="332.8" cy="133.0" r="4.5" class="fill-a"/>
<circle cx="573.4" cy="54.6" r="4.5" class="fill-a"/>
<text x="404.8" y="122.9" class="fig-small" text-anchor="start">0.3 → 0.25</text>
<text x="155.3" y="177.3" class="fig-small" text-anchor="end">−0.62 → −0.5</text>
<text x="574" y="76.6" class="fig-small" text-anchor="end">0.9 and 0.99 → 0.75 (clipped)</text>
</svg>
Figure 1. A 3-bit quantizer. Every input rounds to the nearest of eight levels 0.25 apart. Inputs above 0.875, such as 0.9 and 0.99, land on the top level 0.75, which is why clipping errors are larger than rounding errors.
```

## How much noise does rounding add?

Start with the simplest claim you could make: each bit adds 6 dB of quality. Can you derive it? When the signal is loud and busy, the rounding error is equally likely to land anywhere in $[-\Delta/2, \Delta/2]$, so it behaves like noise. The mean square of a number uniform on an interval of width $\Delta$ is $\frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2} e^2 \, de = \Delta^2/12$. That is the **noise power**.

A sine of amplitude $A$ has a power of $A^2/2$. The **signal-to-noise ratio** (SNR) is the ratio of the two powers, in decibels:

$$
\text{SNR} = 10 \log_{10} \frac{A^2 / 2}{\Delta^2 / 12}
= 10 \log_{10}\!\left(1.5 \cdot 4^N\right) + 20 \log_{10} A
\approx 6.02 N + 1.76 + 20 \log_{10} A \ \text{dB}
$$

Read it aloud: the ratio of the sine's power to the noise power, which at full scale is ten times the log of 1.5 times four to the power $N$. The 4 hides the 6 dB. Each extra bit halves $\Delta$, which quarters the noise power, and $10 \log_{10} 4 = 6.02$ dB. A quieter sine loses exactly its level in dBFS, because its power falls and the noise does not.

```python
predicted = 10 * np.log10(1.5 * 4.0 ** bits) + 20 * np.log10(amplitude)
measured = 10 * np.log10(np.sum(x ** 2) / np.sum((quantize(x, bits) - x) ** 2))
```

At full scale, 8 bits predicts 49.93 dB, 16 bits predicts 98.09 dB, and 24 bits predicts 146.26 dB. Figure 2 measures a real sine and compares. The loud sine, at amplitude 0.99, sits just under the rule: at 8 bits it measures 49.73 dB against a prediction of 49.84 dB for that amplitude. The sine 20 dB quieter sits 20 dB lower, all the way along.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Measured signal-to-noise ratio against bits per sample for a loud sine and a sine 20 dB quieter: both rise 6 dB per bit, and the quiet one sits 20 dB lower">
<line x1="64" y1="248.0" x2="576" y2="248.0" class="stroke-grid"/>
<text x="58" y="252.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="204.2" x2="576" y2="204.2" class="stroke-grid"/>
<text x="58" y="208.2" class="fig-small" text-anchor="end">20</text>
<line x1="64" y1="160.4" x2="576" y2="160.4" class="stroke-grid"/>
<text x="58" y="164.4" class="fig-small" text-anchor="end">40</text>
<line x1="64" y1="116.6" x2="576" y2="116.6" class="stroke-grid"/>
<text x="58" y="120.6" class="fig-small" text-anchor="end">60</text>
<line x1="64" y1="72.8" x2="576" y2="72.8" class="stroke-grid"/>
<text x="58" y="76.8" class="fig-small" text-anchor="end">80</text>
<line x1="64" y1="29.0" x2="576" y2="29.0" class="stroke-grid"/>
<text x="58" y="33.0" class="fig-small" text-anchor="end">100</text>
<line x1="64.0" y1="248" x2="64.0" y2="252" class="stroke-muted"/>
<text x="64.0" y="265" class="fig-small" text-anchor="middle">4</text>
<line x1="149.3" y1="248" x2="149.3" y2="252" class="stroke-muted"/>
<text x="149.3" y="265" class="fig-small" text-anchor="middle">6</text>
<line x1="234.7" y1="248" x2="234.7" y2="252" class="stroke-muted"/>
<text x="234.7" y="265" class="fig-small" text-anchor="middle">8</text>
<line x1="320.0" y1="248" x2="320.0" y2="252" class="stroke-muted"/>
<text x="320.0" y="265" class="fig-small" text-anchor="middle">10</text>
<line x1="405.3" y1="248" x2="405.3" y2="252" class="stroke-muted"/>
<text x="405.3" y="265" class="fig-small" text-anchor="middle">12</text>
<line x1="490.7" y1="248" x2="490.7" y2="252" class="stroke-muted"/>
<text x="490.7" y="265" class="fig-small" text-anchor="middle">14</text>
<line x1="576.0" y1="248" x2="576.0" y2="252" class="stroke-muted"/>
<text x="576.0" y="265" class="fig-small" text-anchor="middle">16</text>
<line x1="64" y1="248" x2="576" y2="248" class="stroke-muted"/>
<line x1="64" y1="18" x2="64" y2="248" class="stroke-muted"/>
<text x="320.0" y="292" class="fig-small" text-anchor="middle">bits per sample N</text>
<text x="14" y="133.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 133.0)">signal-to-noise ratio (dB)</text>
<polyline points="64.0,191.4 106.7,178.2 149.3,165.0 192.0,151.8 234.7,138.6 277.3,125.5 320.0,112.3 362.7,99.1 405.3,85.9 448.0,72.7 490.7,59.5 533.3,46.3 576.0,33.1" class="stroke-muted dashed" fill="none"/>
<polyline points="64.0,196.4 106.7,181.4 149.3,166.6 192.0,151.9 234.7,139.1 277.3,125.7 320.0,112.4 362.7,99.3 405.3,86.2 448.0,72.9 490.7,59.7 533.3,46.5 576.0,33.3" class="stroke-a" fill="none" stroke-width="2"/>
<circle cx="64.0" cy="196.4" r="3" class="fill-a"/>
<circle cx="106.7" cy="181.4" r="3" class="fill-a"/>
<circle cx="149.3" cy="166.6" r="3" class="fill-a"/>
<circle cx="192.0" cy="151.9" r="3" class="fill-a"/>
<circle cx="234.7" cy="139.1" r="3" class="fill-a"/>
<circle cx="277.3" cy="125.7" r="3" class="fill-a"/>
<circle cx="320.0" cy="112.4" r="3" class="fill-a"/>
<circle cx="362.7" cy="99.3" r="3" class="fill-a"/>
<circle cx="405.3" cy="86.2" r="3" class="fill-a"/>
<circle cx="448.0" cy="72.9" r="3" class="fill-a"/>
<circle cx="490.7" cy="59.7" r="3" class="fill-a"/>
<circle cx="533.3" cy="46.5" r="3" class="fill-a"/>
<circle cx="576.0" cy="33.3" r="3" class="fill-a"/>
<polyline points="64.0,236.5 106.7,224.9 149.3,206.8 192.0,195.6 234.7,182.8 277.3,170.1 320.0,155.6 362.7,142.9 405.3,129.8 448.0,116.7 490.7,103.2 533.3,90.1 576.0,77.0" class="stroke-b" fill="none" stroke-width="2"/>
<circle cx="64.0" cy="236.5" r="3" class="fill-b"/>
<circle cx="106.7" cy="224.9" r="3" class="fill-b"/>
<circle cx="149.3" cy="206.8" r="3" class="fill-b"/>
<circle cx="192.0" cy="195.6" r="3" class="fill-b"/>
<circle cx="234.7" cy="182.8" r="3" class="fill-b"/>
<circle cx="277.3" cy="170.1" r="3" class="fill-b"/>
<circle cx="320.0" cy="155.6" r="3" class="fill-b"/>
<circle cx="362.7" cy="142.9" r="3" class="fill-b"/>
<circle cx="405.3" cy="129.8" r="3" class="fill-b"/>
<circle cx="448.0" cy="116.7" r="3" class="fill-b"/>
<circle cx="490.7" cy="103.2" r="3" class="fill-b"/>
<circle cx="533.3" cy="90.1" r="3" class="fill-b"/>
<circle cx="576.0" cy="77.0" r="3" class="fill-b"/>
<text x="72.5" y="33.0" class="fig-small" text-anchor="start">dashed: 6.02N + 1.76 at full scale</text>
<text x="332.8" y="128.4" class="fig-small" text-anchor="start">sine at 0.99</text>
<text x="413.9" y="147.8" class="fig-small" text-anchor="start">sine at 0.099 (−20 dB)</text>
</svg>
Figure 2. Measured SNR of a 1234.5 Hz sine quantized at 4 to 16 bits. Both lines climb 6 dB per bit. The dashed line is the full-scale rule, and the quieter sine (amplitude 0.099) stays 20 dB below the loud one.
```

> [!KEY]
> Every bit is worth about 6 dB of SNR, but only for a signal near full scale. A signal 20 dB below full scale gets 20 dB less.

Speech is never a full-scale sine. A recording that peaks at $-20$ dBFS has a typical RMS level near $-35$ dBFS, 32 dB below the $-3.01$ dBFS of a full-scale sine. So at 16 bits its SNR is about 66 dB instead of 98, and at 8 bits it is about 18 dB, which sounds harsh.

## When the error stops being noise

The derivation assumed the error is random and spread evenly. That fails for quiet signals, and your plucked note shows it. Quantize a constant that sits 0.3 steps above zero. It rounds to 0 on every sample, so the signal vanishes, and the error is exactly $-0.3\Delta$ every time. A tone below half a step does the same thing, and a tone just above it gets rounded into a staircase wave that follows the tone. The error is then not hiss. It repeats with the tone, so you hear it as extra harmonics, the buzz in your note. A fading tone even cuts off abruptly once it drops below half a step.

The obvious fix is more bits, but a fade-out always reaches quiet levels, and the problem returns at any bit depth. The better fix breaks the pattern. **Dither** is a little random noise added to each sample before you round it. It pushes the sample over or under the rounding threshold at random, and the closer the sample is to a threshold, the more often it crosses.

Check it on the constant 0.3 steps with a single uniform draw on $[-\Delta/2, \Delta/2]$. The sample plus the draw lands between $-0.2\Delta$ and $0.8\Delta$. It rounds up to one step exactly when it is above $0.5\Delta$, which happens when the draw exceeds $0.2\Delta$. That is a probability of 0.3. So the output is $\Delta$ with probability 0.3 and 0 otherwise, and the average output is $0.3\Delta$. The signal survives, hidden in how often the output flips.

**Triangular dither**, also called TPDF dither, uses the sum of two independent uniform draws on $[-\Delta/2, \Delta/2]$. A single draw fixes the mean but not the noise power, which then swings from 0 at a level of 0 to $\Delta^2/4$ at half a step, so the hiss pumps up and down with the signal. The triangular version fixes both: the error has a mean of zero and a power of $\Delta^2/4$ whatever the signal level.

```python
noise = rng.uniform(-step / 2, step / 2, x.shape) + rng.uniform(-step / 2, step / 2, x.shape)
y = quantize(x + noise, bits)
```

The price is noise. The two draws add $\Delta^2/6$ to the $\Delta^2/12$ of rounding, for a total of $\Delta^2/4$, three times as much, which is $10 \log_{10} 3 = 4.77$ dB less SNR. Figure 3 measures 200,000 samples of the constant 0.3 steps at 8 bits. Without dither every output is 0. With it, 66.1% are 0, 31.9% are one step, and 2.0% are minus one step. The mean is 0.299 steps.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Where a constant 0.3 step lands: without dither always on 0, with triangular dither on 0 about 66 percent of the time and on 1 about 32 percent, so the average is 0.299">
<text x="40" y="24" class="fig-bold" text-anchor="start">A constant input of 0.3 steps, 200,000 samples</text>
<text x="40" y="68" class="fig-bold" text-anchor="start">No dither</text>
<rect x="200" y="52" width="90" height="26" class="fill-sunk"/>
<rect x="200" y="52" width="0.0" height="26" class="fill-a"/>
<text x="245" y="96" class="fig-small" text-anchor="middle">output −1 step: 0.0%</text>
<rect x="320" y="52" width="90" height="26" class="fill-sunk"/>
<rect x="320" y="52" width="90.0" height="26" class="fill-a"/>
<text x="365" y="96" class="fig-small" text-anchor="middle">output 0 step: 100.0%</text>
<rect x="440" y="52" width="90" height="26" class="fill-sunk"/>
<rect x="440" y="52" width="0.0" height="26" class="fill-a"/>
<text x="485" y="96" class="fig-small" text-anchor="middle">output 1 step: 0.0%</text>
<text x="40" y="168" class="fig-bold" text-anchor="start">TPDF dither</text>
<rect x="200" y="152" width="90" height="26" class="fill-sunk"/>
<rect x="200" y="152" width="1.8" height="26" class="fill-b"/>
<text x="245" y="196" class="fig-small" text-anchor="middle">output −1 step: 2.0%</text>
<rect x="320" y="152" width="90" height="26" class="fill-sunk"/>
<rect x="320" y="152" width="59.5" height="26" class="fill-b"/>
<text x="365" y="196" class="fig-small" text-anchor="middle">output 0 step: 66.1%</text>
<rect x="440" y="152" width="90" height="26" class="fill-sunk"/>
<rect x="440" y="152" width="28.7" height="26" class="fill-b"/>
<text x="485" y="196" class="fig-small" text-anchor="middle">output 1 step: 31.9%</text>
<text x="40" y="262" class="fig-small fig-bold" text-anchor="start">Mean output: 0.000 steps without dither, 0.299 steps with it</text>
</svg>
Figure 3. The outputs of the 8-bit quantizer for a constant input of 0.3 steps. Without dither it returns 0 every time and the signal is lost. With triangular dither it returns 0, one step, or minus one step, and the average of those outputs is 0.299 steps.
```

The following demo lets you lower the bit depth of a quiet tone and switch dither on and off.

```demo
{"id": "quantization-and-dither", "caption": "Drop to a few bits, then turn on dither and listen for the distortion turning into a steady hiss."}
```

> [!KEY]
> Back to the plucked note. At 6 bits, the buzz is rounding error that follows the note, and the abrupt stop is the note dropping below half a step. Triangular dither turns both into a steady hiss, so the note fades out smoothly. It costs 4.77 dB of SNR and buys an error whose mean and power do not depend on the signal.

> [!RECAP]
> - `quantize` divides by $\Delta = 2^{1-N}$, rounds, clips to the $2^N$ codes, and multiplies back.
> - Rounding noise has power $\Delta^2/12$, so a sine of amplitude $A$ has an SNR of $10 \log_{10}(1.5 \cdot 4^N) + 20 \log_{10} A$ dB.
> - The rule assumes the error is noise, and it fails for quiet or repetitive signals.
> - Triangular dither is two uniform draws on $[-\Delta/2, \Delta/2]$, added before quantizing.
> - Dither costs 4.77 dB and keeps the error mean at zero and its power at $\Delta^2/4$.

## Build it step by step

The checks test the functions in this order.

1. **`quantize(x, bits)`** returns floats on the grid. Use `np.round` and `np.clip`. Common mistake: clipping to $2^{N-1}$ instead of $2^{N-1} - 1$, which invents a level at $+1.0$ that no integer format has.
2. **`predicted_snr_db(bits, amplitude)`** returns the formula. Common mistake: `6.02 * bits + 1.76`, which is off by 0.015 dB at 24 bits and ignores the amplitude.
3. **`tpdf_dither(x, bits, rng)`** adds two `rng.uniform` draws and calls your `quantize`. Common mistake: one draw of width $2\Delta$, which is uniform, not triangular, and leaves the noise power depending on the signal.

## Where this breaks

**Clipping hides in the top code.** An amplitude of 0.999 at 8 bits divides to 127.87, which rounds to 128, which clips to 127. The SNR of that sine comes out about 0.6 dB below the prediction. Leave headroom, or normalize to a peak below $1 - \Delta/2$.

**Tones that divide the sample rate do not give noise.** A 1 kHz tone at 16 kHz repeats every 16 samples, so its rounding error repeats too. It lands on harmonics of 1 kHz instead of spreading out, and the measured SNR can miss the prediction by more than 1 dB. Dither fixes this, and the checks use a 1234.5 Hz sine so that the rule holds without it.

**Libraries disagree about dither.** Kaldi's feature extractor adds dither by default, one integer step at 16 bits, so that digital silence never reaches $\log 0$. torchaudio's Kaldi-compatible functions turn it off by default, so the two disagree slightly on the same file.

## The exercise

You build `quantize`, `predicted_snr_db`, and `tpdf_dither`. Press **Run** to hear a decaying note at 6 bits, gritty without dither and cleanly hissing with it, and to see both errors as spectrograms. The error without dither is lines at the note's harmonics, and the error with dither is flat.

The checks confirm the 3-bit example, the 16 levels of 4 bits, and the clipped top code. They check the predicted SNR, a measured SNR within 0.25 dB of it at 8, 12, and 16 bits, and that a sine 20 dB quieter loses 20 dB. They confirm that dither recovers a constant of 0.3 steps, keeps the error power at $\Delta^2/4$ at three levels, and returns values on the grid.
