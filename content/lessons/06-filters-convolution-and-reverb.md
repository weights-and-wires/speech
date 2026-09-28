---
title: Filters, convolution, and reverb
slug: 06-filters-convolution-and-reverb
part: "Part 1 — Sound and signals"
summary: Convolve through the FFT without wrap-around, run an IIR filter from its difference equation, and build a room and measure its RT60.
minutes: 65
gpu: false
objectives:
  - Describe a linear time-invariant system by its impulse response, and convolve with it directly and through the FFT.
  - Explain why FFT convolution needs zero-padding, and catch circular wrap-around.
  - Run an FIR or IIR filter from its difference equation and read its frequency response.
  - Build a synthetic room impulse response and measure its RT60 by Schroeder integration.
prereqs:
  - 03-the-dft-and-the-fft
exercise: 06-filters-convolution-and-reverb
---

# Filters, convolution, and reverb

Record someone in a bathroom and in a closet, and the words are the same but
the recordings are not. The bathroom's hard walls send echoes back for a
second or more; the closet's clothes swallow them. A telephone line cuts the
lows and highs; a microphone adds its own coloring. All of these are
**filters**, and almost all of them share one property that makes them easy
to model: each is completely described by how it responds to a single click.

That click response, the **impulse response**, turns filtering into one
operation, **convolution**. In this lesson you convolve directly and through
the FFT, run filters defined by feedback, and build a synthetic room. Speech
systems use all three: pre-emphasis in classical front ends, band-limiting
before resampling, and convolution with room responses to teach recognizers
that speech in a hall is still speech.

## What you already know

- An FFT of $N$ samples costs $O(N \log N)$, and multiplying spectra bin by bin costs $O(N)$.
- Bin $k$ of an $N$-point DFT is at frequency $k f_s / N$.
- The DFT treats a frame as one period of a signal that repeats forever.
- A windowed-sinc low-pass filter is a list of taps that you convolve with the signal.
- A power ratio in decibels is $10 \log_{10}$ of the ratio, and an amplitude ratio $20 \log_{10}$.

## Systems, impulses, and convolution

A **system** takes a signal in and gives a signal out. It is **linear** if
scaling or adding inputs scales or adds the outputs, and **time-invariant**
if delaying the input delays the output and changes nothing else. A room,
a microphone, and an equalizer are all close to **LTI**, linear and
time-invariant.

Every signal is a sum of scaled, delayed impulses: sample $k$ is an impulse of
height $x[k]$ at time $k$. An LTI system answers each impulse with a scaled,
delayed copy of its impulse response $h$, and adds the answers. That sum is
**convolution**:

$$
y[n] = (x * h)[n] = \sum_{k} h[k] \, x[n - k]
$$

Read it aloud: output sample $n$ is the sum, over every tap $k$ of the
impulse response, of the tap times the input from $k$ samples ago. Each tap
adds a delayed, scaled copy of the whole input. For inputs of $L_x$ and
$L_h$ samples, the output has $L_x + L_h - 1$ samples. In numpy, one tap at a
time:

```python
y = np.zeros(len(x) + len(h) - 1)
for k, tap in enumerate(h):
    y[k : k + len(x)] += tap * x
```

Work it with $x = [1, 2, 3]$ and $h = [1, 1]$. Tap 0 adds $[1, 2, 3]$ at
offset 0, and tap 1 adds $[1, 2, 3]$ at offset 1. The sum is
$[1, 1 + 2, 2 + 3, 3] = [1, 3, 5, 3]$: four samples, as $3 + 2 - 1$ predicts.

> [!KEY]
> An LTI system is its impulse response. Filtering with it is convolution,
> and the output is $L_x + L_h - 1$ samples long.

## Convolve through the FFT

Direct convolution costs $L_x L_h$ multiplications. A room response lasts a
second, 16,000 taps at 16 kHz, so one second of reverberant speech would take
256 million. The **convolution theorem** gives a shortcut: convolution in
time is multiplication in frequency.

$$
\operatorname{DFT}(x * h) = \operatorname{DFT}(x) \cdot \operatorname{DFT}(h),
\quad \text{when both are zero-padded to } N \ge L_x + L_h - 1
$$

Read it aloud: take the DFT of the input and of the impulse response, multiply
them bin by bin, and invert, and you get the convolution, as long as both
were padded to at least the output's length. The condition is the whole
trick. The DFT treats each signal as one period of a loop, so its product
gives **circular** convolution: whatever runs past the end of $N$ samples
wraps around and lands on the start.

```python
n = len(x) + len(h) - 1
n_fft = 1 << (n - 1).bit_length()          # the next power of two
y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n]
```

Try it on the example with $N = 3$, too short. The circular result is
$[4, 3, 5]$: the last output, 3, wrapped around onto the first, $1$. With
$N = 4$ the result is $[1, 3, 5, 3]$, correct.

> [!WARNING]
> Multiplying two FFTs of the signal's own length gives circular convolution.
> A reverberant tail from the end of a clip then appears at its start. Always
> pad to at least $L_x + L_h - 1$.

For very long inputs, such as an hour of audio, one giant FFT wastes memory.
**Overlap-add** convolution cuts the input into blocks, convolves each block
through the FFT, and adds the overlapping tails, which is what
`scipy.signal.oaconvolve` does.

## FIR and IIR filters

A filter can also be defined by a rule that uses its own past outputs. The
general **difference equation**, with $a_0 = 1$, is the following:

$$
y[n] = \sum_{k=0}^{M} b_k \, x[n - k] - \sum_{k=1}^{P} a_k \, y[n - k]
$$

Read it aloud: each output is a weighted sum of the last $M + 1$ inputs,
minus a weighted sum of the last $P$ outputs. The $b_k$ are the
**feedforward** coefficients, and the $a_k$ are the **feedback** coefficients.
When $a_0$ isn't 1, divide every coefficient by it first.

With no feedback, $P = 0$, the filter is **FIR**, finite impulse response: its
impulse response is the $b_k$ themselves, and it ends. A symmetric FIR filter
has **linear phase**: it delays every frequency by the same $M/2$ samples, so
it changes the spectrum without smearing the timing.

With feedback, the filter is **IIR**, infinite impulse response. Take
$y[n] = x[n] + 0.5\, y[n-1]$, which is $b = [1]$ and $a = [1, -0.5]$, on the
input $[1, 0, 0, 0, 2]$. The outputs are $1$, then $0.5 \times 1 = 0.5$, then
$0.25$, then $0.125$, then $2 + 0.5 \times 0.125 = 2.0625$. One impulse rings
on forever, halving each step. IIR filters make sharp responses with few
coefficients, but the feedback can run away. The filter is stable only if
every **pole**, every root of $1 + \sum_k a_k z^{-k}$, lies inside the unit
circle; here the one pole is at 0.5.

A filter's **frequency response** says what it does to a sinusoid at each
frequency $\omega$, in radians per sample from 0 to $\pi$:

$$
H(e^{j\omega}) = \frac{\sum_k b_k \, e^{-j\omega k}}{\sum_k a_k \, e^{-j\omega k}}
$$

Read it aloud: evaluate the feedforward coefficients as a polynomial in
$e^{-j\omega}$, do the same with the feedback coefficients, and divide. Its
magnitude is the gain at that frequency, and its angle is the phase shift.
The frequency in hertz is $\omega f_s / (2\pi)$, so $\omega = \pi$ is the
Nyquist frequency.

```python
z = np.exp(-1j * w)
H = np.polyval(b[::-1], z) / np.polyval(a[::-1], z)
```

> [!TIP]
> For IIR filters of order above about 4, use second-order sections,
> `scipy.signal.butter(..., output="sos")` with `sosfilt`. One long
> polynomial loses precision, and a stable design can turn unstable.

## Pre-emphasis

Voiced speech has more energy at low frequencies than high: its spectrum
tilts down by about 6 dB per octave. **Pre-emphasis** flattens it with a
two-tap FIR filter that subtracts most of the previous sample:

$$
y[n] = x[n] - \alpha\, x[n-1], \qquad |H(e^{j\omega})|^2 = 1 + \alpha^2 - 2\alpha \cos\omega
$$

Read it aloud: each output is the sample minus $\alpha$ times the one before.
Slow changes nearly cancel, and fast changes pass or grow. The power gain is
$1 + \alpha^2 - 2\alpha\cos\omega$.

With Kaldi's default $\alpha = 0.97$, at 16 kHz, the gain is $-30.46$ dB at
0 Hz, where $\cos\omega = 1$ and the power gain is $(1 - 0.97)^2 = 0.0009$.
It is $-8.28$ dB at 1 kHz, $+2.88$ dB at 4 kHz, and $+5.89$ dB at 8 kHz, where
the power gain is $(1 + 0.97)^2 = 3.88$. Pre-emphasis helped the linear
prediction of classical front ends and MFCCs. Whisper and most neural front
ends skip it and let the model learn the tilt.

> [!RECAP]
> - Direct convolution adds `h[k] * x` at offset `k` for every tap; the output has `len(x) + len(h) - 1` samples.
> - FFT convolution pads both signals to a power of two of at least that length, multiplies the `rfft`s, and trims the `irfft`.
> - A filter's frequency response is $B(e^{-j\omega}) / A(e^{-j\omega})$, and its difference equation subtracts the feedback terms.
> - Pre-emphasis is `x[n] - 0.97 * x[n - 1]`.
> - A room is an impulse response; its RT60 is read from the slope of its Schroeder decay.

## Build a room and measure it

A room's impulse response starts with the **direct path**, the sound that
travels straight from the mouth to the microphone, followed by a dense tail
of reflections that decays roughly exponentially. The **reverberation time**
$\text{RT}_{60}$ is how long that tail takes to lose 60 dB of energy. A small
furnished office is around 0.3 s; a concert hall is around 2 s.

Sabine's formula estimates it from the room: $\text{RT}_{60} \approx 0.161\, V / A$,
where $V$ is the volume in cubic meters and $A$ is the total absorption, each
surface's area times its absorption coefficient, in square meters. A
$5 \times 4 \times 3$ m room has $V = 60$ and a surface of 94 m². With hard
walls that absorb 10% of the sound, $A = 9.4$ and $\text{RT}_{60} \approx 1.03$ s.
With soft furnishings at 30%, it falls to 0.34 s.

A synthetic tail is white noise under an exponential envelope. Losing 60 dB of
energy means the amplitude falls by a factor of 1000, which is
$e^{-\ln 1000} = e^{-6.908}$:

$$
h[n] = 0.1\, \nu[n] \, e^{-6.908\, n / (f_s \, \text{RT}_{60})}, \qquad h[0] = 1
$$

Read it aloud: each sample is Gaussian noise $\nu[n]$ scaled by 0.1 and by an
envelope that reaches one thousandth after $\text{RT}_{60}$ seconds, and the
first sample is replaced by a direct-path impulse of 1. With these numbers the
direct path is 5.3 dB quieter than the tail at an RT60 of 0.3 s, and 11.4 dB
quieter at 1.2 s, which is why a large room sounds distant.

To measure RT60 from an impulse response, use **Schroeder integration**:
for each moment, add up all the energy still to come, and plot it in dB. The
raw squared response is too noisy to read; the backward sum is smooth.

$$
E(n) = 10 \log_{10} \frac{\sum_{m \ge n} h[m]^2}{\sum_{m \ge 0} h[m]^2}
$$

Read it aloud: at sample $n$, sum the squares of the response from $n$ to the
end, divide by the total, and convert to decibels. It starts at 0 dB and only
falls. Fit a straight line between $-5$ dB and $-25$ dB, and extrapolate the
line to $-60$ dB. Measuring over 20 dB and multiplying by three is called
**T20**; the end of a real response is buried in noise, so nobody measures
all 60 dB.

```python
energy = np.cumsum((h ** 2)[::-1])[::-1]
decay = 10 * np.log10(energy / energy[0])
```

For a synthetic room with an RT60 of 0.5 s, the decay crosses $-5$ dB at
0.035 s and $-25$ dB at 0.203 s. Twenty decibels took 0.168 s, so 60 would
take about 0.505 s.

The following demo builds a room and plays speech through it.

```demo
{"id": "room-reverb", "caption": "Raise the reverberation time from a small office to a hall, and listen to each syllable start to blur into the next."}
```

> [!KEY]
> Reverb is convolution with a room's impulse response. RT60, the time for its
> energy to fall 60 dB, is the slope of the Schroeder decay extrapolated from
> 20 dB.

Convolving clean speech with measured or synthetic room responses is a
standard training augmentation for recognizers. Real responses can also be
simulated with the image-source method, which mirrors the source in every
wall.

## Build it step by step

The checks test the functions in this order.

1. **`convolve_direct(x, h)`** loops over the taps of `h` and adds shifted,
   scaled copies of `x`. Common mistake: allocating `len(x)` samples of
   output, which cuts off the tail.
2. **`convolve_fft(x, h)`** pads to the next power of two at or above
   `len(x) + len(h) - 1`, multiplies `np.fft.rfft`s, and trims the
   `np.fft.irfft`. Common mistake: padding to `max(len(x), len(h))`, which
   wraps the tail onto the start.
3. **`frequency_response(b, a, n_points)`** builds `w = np.pi * np.arange(n_points) / n_points`
   and divides two `np.polyval`s at `np.exp(-1j * w)`. Common mistake:
   forgetting to reverse the coefficients for `np.polyval`, which expects the
   highest power first.
4. **`lfilter(b, a, x)`** normalizes by `a[0]` and loops over samples. Common
   mistake: adding the feedback terms instead of subtracting them, which makes
   $a = [1, -0.5]$ decay with alternating signs.
5. **`preemphasis(x, alpha)`** keeps `x[0]` and subtracts `alpha * x[:-1]` from
   `x[1:]`. Common mistake: `np.roll`, which subtracts the last sample from
   the first.
6. **`schroeder_decay_db(h)`** reverses, cumulates, reverses back, normalizes,
   and takes `10 * np.log10`. Common mistake: `20 * np.log10`; this is energy.
7. **`estimate_rt60(h, sample_rate, start_db, stop_db)`** masks the decay
   between the two levels and fits `np.polyfit(t, decay, 1)`. Common mistake:
   fitting against sample index instead of seconds.
8. **`synthetic_rir(rt60, sample_rate, seconds, seed)`** builds the envelope
   with `np.exp(-np.log(1000.0) * t / rt60)`. Common mistake: `np.log(60)`,
   which confuses decibels with an amplitude ratio.

## Cost and scaling

Convolution is where filtering cost lives. The following table compares the
methods for an input of $L_x$ samples and an impulse response of $L_h$.

| Method | Time | 1 s of speech through a 0.45 s room |
|---|---|---|
| Direct | $O(L_x L_h)$ | about 48 ms in numpy, 115 million multiplies |
| One FFT | $O((L_x + L_h) \log (L_x + L_h))$ | about 0.4 ms |
| Overlap-add blocks of $B$ | $O(L_x \log B)$ when $B$ is near $L_h$ | similar, with memory $O(B)$ |
| IIR difference equation | $O(L_x (M + P))$ | cheap, but serial: each output needs the last |

Direct convolution wins only for short filters, a few dozen taps. Your
`lfilter` loops in Python, so it is slow in wall-clock time even though its
arithmetic is small; `scipy.signal.lfilter` runs the same loop in C.

## Where this breaks

**Circular wrap-around.** The most common bug in FFT convolution. The check
for `convolve_fft` puts an impulse on the last sample so any wrap shows up at
the start.

**Unstable IIR filters.** A pole on or outside the unit circle makes the
output grow without bound. High-order filters designed as one polynomial can
become unstable from rounding alone. Use second-order sections.

**Phase distortion.** IIR filters delay different frequencies by different
amounts. When timing matters, run the filter forward and then backward with
`scipy.signal.filtfilt`, which cancels the phase, at the cost of being
non-causal.

**Synthetic rooms are too simple.** Real rooms have distinct early
reflections, a decay that depends on frequency, and noise. A recognizer
trained only on exponential noise tails can still fail in a real hall.
Mix in measured responses where you can.

**Levels change.** Convolving with a room adds the tail's energy, and the
output can clip. Normalize after convolving, as the **Run** block does, and
measure SNR against a time-aligned target: the direct path delays nothing,
but a real one delays by the source distance.

## The exercise

You build `convolve_direct`, `convolve_fft`, `frequency_response`, `lfilter`,
`preemphasis`, `schroeder_decay_db`, `estimate_rt60`, and `synthetic_rir`.
Press **Run** to hear three syllables and a clap dry, in a room with an RT60
of 0.3 s, and in one of 1.2 s, and see how the large room smears the
spectrogram. Then see both rooms' Schroeder decay curves, time direct against
FFT convolution, and hear pre-emphasis brighten the syllables.

The checks confirm the worked convolution and agreement with `np.convolve`,
FFT convolution with an impulse at the very end that must not wrap, and your
frequency response against `scipy.signal.freqz`. They check the IIR example
and your `lfilter` against `scipy.signal.lfilter`, including $a_0 \ne 1$, and
pre-emphasis values and its gains of $-30.46$ and $+5.89$ dB. The last
checks compare your Schroeder decay with the closed form for an exponential,
your RT60 estimate within 5% on known decays, and your synthetic rooms within
10% of the RT60 you asked for.
