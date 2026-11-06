---
title: Convolution and filters
slug: 009-convolution-and-filters
part: "Part 1 — Sound and signals"
summary: Describe a filter by its impulse response, convolve through the FFT without wrap-around, and run IIR filters and pre-emphasis.
minutes: 45
gpu: false
objectives:
  - Describe a linear time-invariant system by its impulse response, and convolve with it directly and through the FFT.
  - Explain why FFT convolution needs zero-padding, and catch circular wrap-around.
  - Run an FIR or IIR filter from its difference equation and read its frequency response.
  - Apply pre-emphasis and state its gain at 0 Hz and near Nyquist.
prereqs:
  - 002-sampling-and-aliasing
  - 006-the-fft-and-windows
exercise: 009-convolution-and-filters
---

# Convolution and filters

Call someone on a telephone, and their voice sounds thin, because the line cuts the lows and the highs. Record the same person in a stairwell, and their words come back as a stream of echoes. A microphone, a hall, and an equalizer all change a sound without anyone changing the words. These are **filters**, and almost all of them share one property that makes them easy to model: each is completely described by how it responds to a single click.

That click response, the **impulse response**, turns filtering into one operation, **convolution**. In this chapter you convolve directly, then through the FFT, then run filters defined by feedback. Speech systems use all three: pre-emphasis in classical front ends, band-limiting before resampling, and convolution with room responses to teach recognizers that speech in a hall is still speech. The next chapter builds the hall.

## What you already know

- An FFT of $N$ samples costs $O(N \log N)$, and multiplying two spectra bin by bin costs $O(N)$.
- Bin $k$ of an $N$-point DFT is at frequency $k f_s / N$.
- The DFT treats a frame as one period of a signal that repeats forever.
- A windowed-sinc low-pass filter is a list of taps that you convolve with the signal.
- A power ratio in decibels is $10 \log_{10}$ of the ratio, and an amplitude ratio is $20 \log_{10}$.

## A filter is its response to a click

A **system** takes a signal in and gives a signal out. It is **linear** if scaling or adding inputs scales or adds the outputs, and **time-invariant** if delaying the input delays the output and changes nothing else. A room, a microphone, and an equalizer are all close to **LTI**, which stands for linear and time-invariant.

Here is why one click is enough. Every signal is a sum of scaled, delayed clicks: sample $k$ is a click of height $x[k]$ at time $k$. An LTI system answers each click with a scaled, delayed copy of its **impulse response** $h$, which is its answer to a single click of height 1. Because the system is linear, the full answer is the sum of all those copies. That sum is **convolution**:

$$
y[n] = (x * h)[n] = \sum_{k} h[k] \, x[n - k]
$$

Read it aloud: output sample $n$ is the sum, over every tap $k$ of the impulse response, of the tap times the input from $k$ samples ago. Each tap contributes a delayed, scaled copy of the whole input. For inputs of $L_x$ and $L_h$ samples, the output has $L_x + L_h - 1$ samples, because the last copy starts $L_h - 1$ samples late. In numpy, one tap at a time:

```python
y = np.zeros(len(x) + len(h) - 1)
for k, tap in enumerate(h):
    y[k : k + len(x)] += tap * x
```

Work it with $x = [1, 2, 3]$ and $h = [1, 1]$. Tap 0 adds $[1, 2, 3]$ at offset 0, and tap 1 adds $[1, 2, 3]$ at offset 1, as Figure 1 shows. The sum is $[1, 1 + 2, 2 + 3, 3] = [1, 3, 5, 3]$: four samples, as $3 + 2 - 1$ predicts.

```figure
<svg viewBox="0 0 580 222" role="img" aria-label="Convolution of [1, 2, 3] with [1, 1] as two shifted copies of the input that add up to [1, 3, 5, 3]">
<text x="285.0" y="28.0" class="fig-small" text-anchor="middle">n = 0</text>
<text x="355.0" y="28.0" class="fig-small" text-anchor="middle">n = 1</text>
<text x="425.0" y="28.0" class="fig-small" text-anchor="middle">n = 2</text>
<text x="495.0" y="28.0" class="fig-small" text-anchor="middle">n = 3</text>
<text x="20.0" y="70.0" text-anchor="start">tap h[0] = 1, times x</text>
<rect x="250.0" y="44.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="250.0" y="44.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="285.0" y="69.0" text-anchor="middle">1</text>
<rect x="320.0" y="44.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="320.0" y="44.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="355.0" y="69.0" text-anchor="middle">2</text>
<rect x="390.0" y="44.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="390.0" y="44.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="425.0" y="69.0" text-anchor="middle">3</text>
<rect x="460.0" y="44.0" width="70.0" height="38.0" class="fill-sunk"/>
<text x="20.0" y="126.0" text-anchor="start">tap h[1] = 1, times x, one sample later</text>
<rect x="250.0" y="100.0" width="70.0" height="38.0" class="fill-sunk"/>
<rect x="320.0" y="100.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="320.0" y="100.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="355.0" y="125.0" text-anchor="middle">1</text>
<rect x="390.0" y="100.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="390.0" y="100.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="425.0" y="125.0" text-anchor="middle">2</text>
<rect x="460.0" y="100.0" width="70.0" height="38.0" class="fill-soft-a"/>
<rect x="460.0" y="100.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="495.0" y="125.0" text-anchor="middle">3</text>
<text x="20.0" y="192.0" class="fig-bold" text-anchor="start">sum: y = x * h</text>
<rect x="250.0" y="166.0" width="70.0" height="38.0" class="fill-soft-b"/>
<rect x="250.0" y="166.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="285.0" y="191.0" class="fig-bold" text-anchor="middle">1</text>
<rect x="320.0" y="166.0" width="70.0" height="38.0" class="fill-soft-b"/>
<rect x="320.0" y="166.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="355.0" y="191.0" class="fig-bold" text-anchor="middle">3</text>
<rect x="390.0" y="166.0" width="70.0" height="38.0" class="fill-soft-b"/>
<rect x="390.0" y="166.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="425.0" y="191.0" class="fig-bold" text-anchor="middle">5</text>
<rect x="460.0" y="166.0" width="70.0" height="38.0" class="fill-soft-b"/>
<rect x="460.0" y="166.0" width="70.0" height="38.0" class="stroke-muted"/>
<text x="495.0" y="191.0" class="fig-bold" text-anchor="middle">3</text>
<line x1="20.0" y1="170.0" x2="530.0" y2="170.0" class="stroke-grid"/>
</svg>
Figure 1. Convolution as shifted copies. Notice that each tap of h adds one scaled copy of x, and that the output is one sample longer than x because h has two taps.
```

> [!KEY]
> An LTI system is its impulse response. Filtering with it is convolution, and the output is $L_x + L_h - 1$ samples long.

## Convolve through the FFT

Direct convolution costs $L_x L_h$ multiplications. A room response lasts a second, which is 16,000 taps at 16 kHz, so one second of reverberant speech would take 256 million multiplications. The FFT gives a shortcut. Multiplying two sinusoids' coefficients is cheap, and the **convolution theorem** says that convolution in time is multiplication in frequency:

$$
\operatorname{DFT}(x * h) = \operatorname{DFT}(x) \cdot \operatorname{DFT}(h),
\quad \text{when both are zero-padded to } N \ge L_x + L_h - 1
$$

Read it aloud: take the DFT of the input and of the impulse response, multiply them bin by bin, and invert, and you get the convolution, as long as both were padded to at least the output's length. That condition is the whole trick. The DFT treats each signal as one period of a loop, so the plain product gives **circular** convolution: whatever runs past the end of $N$ samples wraps around and lands on the start.

Try the example with $N = 3$, which is too short. The circular result is $[4, 3, 5]$, because the last output, 3, wrapped onto the first, 1. With $N = 4$ the result is $[1, 3, 5, 3]$, which is correct (Figure 2).

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Circular convolution with FFT length 3 gives [4, 3, 5], with the last output wrapped onto the first; length 4 gives the correct [1, 3, 5, 3]">
<text x="40.0" y="28.0" class="fig-bold" text-anchor="start">FFT length 3: too short</text>
<line x1="30.0" y1="190.0" x2="230.0" y2="190.0" class="stroke-muted"/>
<rect x="40.0" y="168.0" width="44.0" height="22.0" class="fill-a"/>
<rect x="40.0" y="102.0" width="44.0" height="66.0" class="fill-bad"/>
<text x="62.0" y="96.0" class="fig-bold" text-anchor="middle">4</text>
<text x="62.0" y="208.0" class="fig-small" text-anchor="middle">n = 0</text>
<rect x="100.0" y="124.0" width="44.0" height="66.0" class="fill-a"/>
<text x="122.0" y="118.0" class="fig-bold" text-anchor="middle">3</text>
<text x="122.0" y="208.0" class="fig-small" text-anchor="middle">n = 1</text>
<rect x="160.0" y="80.0" width="44.0" height="110.0" class="fill-a"/>
<text x="182.0" y="74.0" class="fig-bold" text-anchor="middle">5</text>
<text x="182.0" y="208.0" class="fig-small" text-anchor="middle">n = 2</text>
<text x="40.0" y="232.0" class="fig-small" text-anchor="start">The last output (3, red) wrapped onto n = 0</text>
<text x="330.0" y="28.0" class="fig-bold" text-anchor="start">FFT length 4: long enough</text>
<line x1="320.0" y1="190.0" x2="580.0" y2="190.0" class="stroke-muted"/>
<rect x="330.0" y="168.0" width="44.0" height="22.0" class="fill-a"/>
<text x="352.0" y="162.0" class="fig-bold" text-anchor="middle">1</text>
<text x="352.0" y="208.0" class="fig-small" text-anchor="middle">n = 0</text>
<rect x="390.0" y="124.0" width="44.0" height="66.0" class="fill-a"/>
<text x="412.0" y="118.0" class="fig-bold" text-anchor="middle">3</text>
<text x="412.0" y="208.0" class="fig-small" text-anchor="middle">n = 1</text>
<rect x="450.0" y="80.0" width="44.0" height="110.0" class="fill-a"/>
<text x="472.0" y="74.0" class="fig-bold" text-anchor="middle">5</text>
<text x="472.0" y="208.0" class="fig-small" text-anchor="middle">n = 2</text>
<rect x="510.0" y="124.0" width="44.0" height="66.0" class="fill-a"/>
<text x="532.0" y="118.0" class="fig-bold" text-anchor="middle">3</text>
<text x="532.0" y="208.0" class="fig-small" text-anchor="middle">n = 3</text>
<text x="330.0" y="232.0" class="fig-small" text-anchor="start">Correct: 3 + 2 - 1 = 4 outputs</text>
</svg>
Figure 2. The same convolution at two FFT lengths. Notice that the red segment, the fourth output, has nowhere to go at length 3 and lands on sample 0.
```

```python
n = len(x) + len(h) - 1
n_fft = 1 << (n - 1).bit_length()          # the next power of two
y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n]
```

> [!WARNING]
> Multiplying two FFTs of the signal's own length gives circular convolution. A reverberant tail from the end of a clip then appears at its start. Always pad to at least $L_x + L_h - 1$.

For very long inputs, such as an hour of audio, one giant FFT wastes memory. **Overlap-add** convolution cuts the input into blocks, convolves each block through the FFT, and adds the overlapping tails, which is what `scipy.signal.oaconvolve` does.

## Filters with feedback

A filter can also be defined by a rule that uses its own past outputs. The general **difference equation**, with $a_0 = 1$, is the following:

$$
y[n] = \sum_{k=0}^{M} b_k \, x[n - k] - \sum_{k=1}^{P} a_k \, y[n - k]
$$

Read it aloud: each output is a weighted sum of the last $M + 1$ inputs, minus a weighted sum of the last $P$ outputs. The $b_k$ are the **feedforward** coefficients, and the $a_k$ are the **feedback** coefficients. When $a_0$ isn't 1, divide every coefficient by it first.

With no feedback, $P = 0$, the filter is **FIR**, for finite impulse response: its impulse response is the $b_k$ themselves, and it ends. With feedback, the filter is **IIR**, for infinite impulse response. Take $y[n] = x[n] + 0.5\, y[n-1]$, which is $b = [1]$ and $a = [1, -0.5]$, on the input $[1, 0, 0, 0, 2]$. The outputs are $1$, then $0.5 \times 1 = 0.5$, then $0.25$, then $0.125$, then $2 + 0.5 \times 0.125 = 2.0625$ (Figure 3).

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Input [1, 0, 0, 0, 2] and output [1, 0.5, 0.25, 0.125, 2.0625] of the filter y[n] = x[n] + 0.5 y[n-1]">
<line x1="70.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-grid"/>
<text x="62.0" y="214.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="171.4" x2="560.0" y2="171.4" class="stroke-grid"/>
<text x="62.0" y="175.4" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="132.7" x2="560.0" y2="132.7" class="stroke-grid"/>
<text x="62.0" y="136.7" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="94.1" x2="560.0" y2="94.1" class="stroke-grid"/>
<text x="62.0" y="98.1" class="fig-small" text-anchor="end">1.5</text>
<line x1="70.0" y1="55.5" x2="560.0" y2="55.5" class="stroke-grid"/>
<text x="62.0" y="59.5" class="fig-small" text-anchor="end">2</text>
<line x1="119.0" y1="210.0" x2="119.0" y2="215.0" class="stroke-muted"/>
<text x="119.0" y="230.0" class="fig-small" text-anchor="middle">0</text>
<line x1="217.0" y1="210.0" x2="217.0" y2="215.0" class="stroke-muted"/>
<text x="217.0" y="230.0" class="fig-small" text-anchor="middle">1</text>
<line x1="315.0" y1="210.0" x2="315.0" y2="215.0" class="stroke-muted"/>
<text x="315.0" y="230.0" class="fig-small" text-anchor="middle">2</text>
<line x1="413.0" y1="210.0" x2="413.0" y2="215.0" class="stroke-muted"/>
<text x="413.0" y="230.0" class="fig-small" text-anchor="middle">3</text>
<line x1="511.0" y1="210.0" x2="511.0" y2="215.0" class="stroke-muted"/>
<text x="511.0" y="230.0" class="fig-small" text-anchor="middle">4</text>
<line x1="70.0" y1="210.0" x2="560.0" y2="210.0" class="stroke-muted"/>
<line x1="70.0" y1="210.0" x2="70.0" y2="40.0" class="stroke-muted"/>
<text x="315.0" y="250.0" text-anchor="middle">Sample n</text>
<text x="30.0" y="26.0" text-anchor="start">Value</text>
<line x1="111.0" y1="210.0" x2="111.0" y2="132.7" class="stroke-a"/>
<circle cx="111.0" cy="132.7" r="5" class="fill-a"/>
<line x1="127.0" y1="210.0" x2="127.0" y2="132.7" class="stroke-b"/>
<circle cx="127.0" cy="132.7" r="5" class="fill-b"/>
<text x="137.0" y="124.7" class="fig-small" text-anchor="start">1</text>
<line x1="209.0" y1="210.0" x2="209.0" y2="210.0" class="stroke-a"/>
<circle cx="209.0" cy="210.0" r="5" class="fill-a"/>
<line x1="225.0" y1="210.0" x2="225.0" y2="171.4" class="stroke-b"/>
<circle cx="225.0" cy="171.4" r="5" class="fill-b"/>
<text x="235.0" y="163.4" class="fig-small" text-anchor="start">0.5</text>
<line x1="307.0" y1="210.0" x2="307.0" y2="210.0" class="stroke-a"/>
<circle cx="307.0" cy="210.0" r="5" class="fill-a"/>
<line x1="323.0" y1="210.0" x2="323.0" y2="190.7" class="stroke-b"/>
<circle cx="323.0" cy="190.7" r="5" class="fill-b"/>
<text x="333.0" y="182.7" class="fig-small" text-anchor="start">0.25</text>
<line x1="405.0" y1="210.0" x2="405.0" y2="210.0" class="stroke-a"/>
<circle cx="405.0" cy="210.0" r="5" class="fill-a"/>
<line x1="421.0" y1="210.0" x2="421.0" y2="200.3" class="stroke-b"/>
<circle cx="421.0" cy="200.3" r="5" class="fill-b"/>
<text x="431.0" y="192.3" class="fig-small" text-anchor="start">0.125</text>
<line x1="503.0" y1="210.0" x2="503.0" y2="55.5" class="stroke-a"/>
<circle cx="503.0" cy="55.5" r="5" class="fill-a"/>
<line x1="519.0" y1="210.0" x2="519.0" y2="50.6" class="stroke-b"/>
<circle cx="519.0" cy="50.6" r="5" class="fill-b"/>
<text x="529.0" y="42.6" class="fig-small" text-anchor="start">2.0625</text>
<circle cx="300.0" cy="20.0" r="5" class="fill-a"/>
<text x="310.0" y="25.0" class="fig-small" text-anchor="start">input x</text>
<circle cx="400.0" cy="20.0" r="5" class="fill-b"/>
<text x="410.0" y="25.0" class="fig-small" text-anchor="start">output y</text>
</svg>
Figure 3. A feedback filter on the input [1, 0, 0, 0, 2]. Notice that the first click rings on, halving each step, and the second click lands on top of what is left.
```

One click rings on forever, halving each step. IIR filters make sharp responses from few coefficients, but the feedback can run away. The filter is stable only if every **pole**, every root of $1 + \sum_k a_k z^{-k}$, lies inside the unit circle. Here the one pole is at 0.5. With $a = [1, -1.1]$ the pole is at 1.1, and the output grows by 10% every sample.

A filter's **frequency response** says what it does to a sinusoid at each frequency $\omega$, in radians per sample from 0 to $\pi$:

$$
H(e^{j\omega}) = \frac{\sum_k b_k \, e^{-j\omega k}}{\sum_k a_k \, e^{-j\omega k}}
$$

Read it aloud: evaluate the feedforward coefficients as a polynomial in $e^{-j\omega}$, do the same with the feedback coefficients, and divide. Its magnitude is the gain at that frequency, and its angle is the phase shift. The frequency in hertz is $\omega f_s / (2\pi)$, so $\omega = \pi$ is the Nyquist frequency.

```python
z = np.exp(-1j * w)
H = np.polyval(b[::-1], z) / np.polyval(a[::-1], z)
```

> [!TIP]
> For IIR filters of order above about 4, use second-order sections: `scipy.signal.butter(..., output="sos")` with `sosfilt`. One long polynomial loses precision, and a stable design can turn unstable.

## Pre-emphasis

Voiced speech has more energy at low frequencies than high, so its spectrum tilts down by about 6 dB per octave. The high frequencies carry the consonants, and they are the quiet part. **Pre-emphasis** flattens the tilt with a two-tap FIR filter that subtracts most of the previous sample:

$$
y[n] = x[n] - \alpha\, x[n-1], \qquad |H(e^{j\omega})|^2 = 1 + \alpha^2 - 2\alpha \cos\omega
$$

Read it aloud: each output is the sample minus $\alpha$ times the one before. A slow change barely differs from the previous sample, so it nearly cancels, while a fast change survives or grows. The power gain is $1 + \alpha^2 - 2\alpha\cos\omega$.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Gain of the pre-emphasis filter y[n] = x[n] - 0.97 x[n-1] from 0 to 8000 hertz at a 16 kilohertz sample rate">
<line x1="70.0" y1="226.7" x2="560.0" y2="226.7" class="stroke-grid"/>
<text x="62.0" y="230.7" class="fig-small" text-anchor="end">-30</text>
<line x1="70.0" y1="180.0" x2="560.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">-20</text>
<line x1="70.0" y1="133.3" x2="560.0" y2="133.3" class="stroke-grid"/>
<text x="62.0" y="137.3" class="fig-small" text-anchor="end">-10</text>
<line x1="70.0" y1="86.7" x2="560.0" y2="86.7" class="stroke-grid"/>
<text x="62.0" y="90.7" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" class="fig-small" text-anchor="end">10</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="192.5" y1="250.0" x2="192.5" y2="255.0" class="stroke-muted"/>
<text x="192.5" y="270.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="315.0" y1="250.0" x2="315.0" y2="255.0" class="stroke-muted"/>
<text x="315.0" y="270.0" class="fig-small" text-anchor="middle">4000</text>
<line x1="437.5" y1="250.0" x2="437.5" y2="255.0" class="stroke-muted"/>
<text x="437.5" y="270.0" class="fig-small" text-anchor="middle">6000</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">8000</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="40.0" class="stroke-muted"/>
<text x="315.0" y="290.0" text-anchor="middle">Frequency (Hz)</text>
<text x="30.0" y="26.0" text-anchor="start">Gain (dB)</text>
<line x1="70.0" y1="86.7" x2="560.0" y2="86.7" class="stroke-muted dashed"/>
<polyline points="70.0,228.8 72.5,224.0 74.9,214.0 77.4,203.9 79.8,195.0 82.3,187.4 84.8,180.8 87.2,175.1 89.7,170.0 92.2,165.5 94.6,161.4 97.1,157.7 99.5,154.3 102.0,151.1 104.5,148.2 106.9,145.4 109.4,142.9 111.9,140.5 114.3,138.2 116.8,136.0 119.2,134.0 121.7,132.0 124.2,130.2 126.6,128.4 129.1,126.7 131.6,125.1 134.0,123.6 136.5,122.1 138.9,120.6 141.4,119.2 143.9,117.9 146.3,116.6 148.8,115.3 151.3,114.1 153.7,112.9 156.2,111.8 158.6,110.7 161.1,109.6 163.6,108.6 166.0,107.5 168.5,106.5 171.0,105.6 173.4,104.6 175.9,103.7 178.3,102.8 180.8,102.0 183.3,101.1 185.7,100.3 188.2,99.5 190.7,98.7 193.1,97.9 195.6,97.1 198.0,96.4 200.5,95.7 203.0,95.0 205.4,94.3 207.9,93.6 210.4,92.9 212.8,92.3 215.3,91.6 217.7,91.0 220.2,90.4 222.7,89.8 225.1,89.2 227.6,88.6 230.1,88.0 232.5,87.5 235.0,86.9 237.4,86.4 239.9,85.8 242.4,85.3 244.8,84.8 247.3,84.3 249.7,83.8 252.2,83.3 254.7,82.8 257.1,82.3 259.6,81.9 262.1,81.4 264.5,81.0 267.0,80.5 269.4,80.1 271.9,79.7 274.4,79.3 276.8,78.8 279.3,78.4 281.8,78.0 284.2,77.6 286.7,77.3 289.1,76.9 291.6,76.5 294.1,76.1 296.5,75.8 299.0,75.4 301.5,75.1 303.9,74.7 306.4,74.4 308.8,74.0 311.3,73.7 313.8,73.4 316.2,73.1 318.7,72.8 321.2,72.4 323.6,72.1 326.1,71.8 328.5,71.5 331.0,71.3 333.5,71.0 335.9,70.7 338.4,70.4 340.9,70.1 343.3,69.9 345.8,69.6 348.2,69.3 350.7,69.1 353.2,68.8 355.6,68.6 358.1,68.3 360.6,68.1 363.0,67.9 365.5,67.6 367.9,67.4 370.4,67.2 372.9,67.0 375.3,66.7 377.8,66.5 380.3,66.3 382.7,66.1 385.2,65.9 387.6,65.7 390.1,65.5 392.6,65.3 395.0,65.1 397.5,65.0 399.9,64.8 402.4,64.6 404.9,64.4 407.3,64.2 409.8,64.1 412.3,63.9 414.7,63.7 417.2,63.6 419.6,63.4 422.1,63.3 424.6,63.1 427.0,63.0 429.5,62.8 432.0,62.7 434.4,62.6 436.9,62.4 439.3,62.3 441.8,62.2 444.3,62.0 446.7,61.9 449.2,61.8 451.7,61.7 454.1,61.6 456.6,61.5 459.0,61.3 461.5,61.2 464.0,61.1 466.4,61.0 468.9,60.9 471.4,60.8 473.8,60.7 476.3,60.7 478.7,60.6 481.2,60.5 483.7,60.4 486.1,60.3 488.6,60.3 491.1,60.2 493.5,60.1 496.0,60.0 498.4,60.0 500.9,59.9 503.4,59.9 505.8,59.8 508.3,59.7 510.8,59.7 513.2,59.6 515.7,59.6 518.1,59.5 520.6,59.5 523.1,59.5 525.5,59.4 528.0,59.4 530.5,59.4 532.9,59.3 535.4,59.3 537.8,59.3 540.3,59.3 542.8,59.2 545.2,59.2 547.7,59.2 550.2,59.2 552.6,59.2 555.1,59.2 557.5,59.2 560.0,59.2" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="228.8" r="4.5" class="fill-b"/>
<text x="78.0" y="218.8" class="fig-small" text-anchor="start">-30.46 dB</text>
<circle cx="131.2" cy="125.3" r="4.5" class="fill-b"/>
<text x="143.2" y="133.3" class="fig-small" text-anchor="start">-8.28 dB</text>
<circle cx="315.0" cy="73.2" r="4.5" class="fill-b"/>
<text x="315.0" y="61.2" class="fig-small" text-anchor="middle">+2.88 dB</text>
<circle cx="560.0" cy="59.2" r="4.5" class="fill-b"/>
<text x="552.0" y="47.2" class="fig-small" text-anchor="end">+5.89 dB</text>
</svg>
Figure 4. The gain of pre-emphasis with alpha 0.97 at 16 kHz. Notice that it cuts 0 Hz by 30 dB and lifts the top of the band by about 6 dB.
```

With Kaldi's default $\alpha = 0.97$ at 16 kHz, the gain is $-30.46$ dB at 0 Hz, where $\cos\omega = 1$ and the power gain is $(1 - 0.97)^2 = 0.0009$. It is $-8.28$ dB at 1 kHz, $+2.88$ dB at 4 kHz, and $+5.89$ dB at 8 kHz, where the power gain is $(1 + 0.97)^2 = 3.88$. Pre-emphasis helped the linear prediction of classical front ends and MFCCs. Whisper and most neural front ends skip it and let the model learn the tilt.

> [!RECAP]
> - Direct convolution adds `h[k] * x` at offset `k` for every tap; the output has `len(x) + len(h) - 1` samples.
> - FFT convolution pads both signals to a power of two of at least that length, multiplies the `rfft`s, and trims the `irfft`.
> - A filter's difference equation subtracts the feedback terms, and its frequency response is $B(e^{-j\omega}) / A(e^{-j\omega})$.
> - A pole outside the unit circle makes an IIR filter blow up.
> - Pre-emphasis is `x[n] - 0.97 * x[n - 1]`.

## Build it step by step

The checks test the functions in this order.

1. **`convolve_direct(x, h)`** loops over the taps of `h` and adds shifted, scaled copies of `x`. A common mistake is allocating `len(x)` samples of output, which cuts off the tail.
2. **`convolve_fft(x, h)`** pads to the next power of two at or above `len(x) + len(h) - 1`, multiplies `np.fft.rfft`s, and trims the `np.fft.irfft`. A common mistake is padding to `max(len(x), len(h))`, which wraps the tail onto the start.
3. **`frequency_response(b, a, n_points)`** builds `w = np.pi * np.arange(n_points) / n_points` and divides two `np.polyval`s at `np.exp(-1j * w)`. A common mistake is forgetting to reverse the coefficients for `np.polyval`, which expects the highest power first.
4. **`lfilter(b, a, x)`** divides by `a[0]` and loops over samples. A common mistake is adding the feedback terms instead of subtracting them, which makes $a = [1, -0.5]$ decay with alternating signs.
5. **`preemphasis(x, alpha)`** keeps `x[0]` and subtracts `alpha * x[:-1]` from `x[1:]`. A common mistake is `np.roll`, which subtracts the last sample from the first.

## Where this breaks

**Circular wrap-around.** The most common bug in FFT convolution. The check for `convolve_fft` puts an impulse on the last sample so that any wrap shows up at the start.

**Direct convolution wins only for short filters.** A few dozen taps are cheaper to loop than to transform. Your `lfilter` loops in Python, so it is slow in wall-clock time even though its arithmetic is small. `scipy.signal.lfilter` runs the same loop in C.

**Phase distortion.** IIR filters delay different frequencies by different amounts. When timing matters, run the filter forward and then backward with `scipy.signal.filtfilt`, which cancels the phase at the cost of being non-causal.

**Unstable filters.** A pole on or outside the unit circle makes the output grow without bound. High-order filters designed as one polynomial can become unstable from rounding alone, so use second-order sections.

## The exercise

You build `convolve_direct`, `convolve_fft`, `frequency_response`, `lfilter`, and `preemphasis`. Press **Run** to hear three syllables and a clap through an echo with three taps, time direct against FFT convolution, hear a feedback filter fade the high harmonics, and hear pre-emphasis brighten the syllables.

The checks confirm the worked convolution and agreement with `np.convolve`, and that convolution is linear and time-invariant. They check FFT convolution with an impulse at the very end that must not wrap, and your frequency response against `scipy.signal.freqz`. They run the IIR example and compare your `lfilter` with `scipy.signal.lfilter`, including $a_0 \ne 1$, and confirm the impulse responses of an FIR and an IIR filter, including one that grows. The last check confirms pre-emphasis values and its gains of $-30.46$ and $+5.89$ dB.
