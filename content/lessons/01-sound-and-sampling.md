---
title: Sound and sampling
slug: 01-sound-and-sampling
part: "Part 1 — Sound and signals"
summary: Turn sound into numbers, measure its level in decibels, predict where a tone aliases, and resample with a windowed-sinc filter.
minutes: 60
gpu: false
objectives:
  - Describe a tone by its frequency, amplitude, and phase, and sample it at a given rate.
  - "Measure a level in dBFS, and pick the right rule: 20 log10 for an amplitude, 10 log10 for a power."
  - Predict the frequency any tone aliases to at any sample rate.
  - Build a windowed-sinc resampler that matches scipy.signal.resample_poly, and explain why every-third-sample decimation fails.
exercise: 01-sound-and-sampling
---

# Sound and sampling

Every system in this course starts from a list of numbers that a microphone
produced. A recognizer reads that list, a synthesizer writes one, and a codec
squeezes one into a few hundred bits per second. First you need to know what
the numbers mean, how loud they are, and what goes wrong when you take them
too slowly.

The course builds upward from that list. Part 1 turns a waveform into a
spectrogram and back. Later parts build recognizers, from templates and hidden
Markov models to CTC models and Whisper, then synthesizers and vocoders, neural
codecs that turn audio into tokens, diffusion and flow-matching generators,
and speech language models that listen and talk. Every lesson has one exercise
that you write yourself.

Each exercise has two buttons. **Run** executes the `if __name__ == "__main__":`
block at the bottom of your file. When it plays a clip, an audio player appears
in the console with a **Download WAV** link. When it draws a spectrogram, time
runs left to right in seconds, frequency runs bottom to top in hertz, and color
shows loudness in decibels. Hold the pointer over a cell to read its values.
**Check** runs the tests against your functions, one line per check, from the
smallest piece to the whole. Some checks draw a figure when they fail.

In this lesson you measure levels, sample a tone, find out why a 7 kHz tone can
turn into a 3 kHz one, and build a resampler that prevents it.

## What you already know

- A sine or cosine wave repeats itself; how often it repeats is its frequency.
- A logarithm turns multiplication into addition: $\log_{10}(100) = 2$.
- numpy applies arithmetic to a whole array at once, so `np.cos(x)` works on every element.
- Slicing takes part of an array: `x[::3]` keeps every third element.
- A weighted moving average replaces each value with a weighted sum of its neighbors, and `np.convolve(x, h)` computes one with weights `h`.

## Sound is pressure that changes over time

Sound is a wave of pressure moving through air at about 343 meters per
second. A microphone turns the pressure at one point into a voltage, and a
converter measures that voltage many times a second. The measurements are the
numbers your code sees.

The simplest sound is a **pure tone**: pressure that follows a cosine. It has
three properties:

- **Frequency** $f$, in hertz (Hz), is how many cycles it completes per second.
  People hear from about 20 Hz to 20 kHz. Speech carries most of its meaning
  below 8 kHz.
- **Amplitude** $A$ is how far it swings from zero. In a digital signal it has
  no unit; full scale is 1.0.
- **Phase** $\varphi$, in radians, is where in its cycle the wave starts.

A tone is written as a function of time $t$ in seconds:

$$
x(t) = A \cos(2 \pi f t + \varphi)
$$

Read it aloud: at time $t$, the wave has turned through $2\pi f t$ radians,
plus its starting phase, and the cosine of that angle, times the amplitude, is
the pressure. One cycle takes $1/f$ seconds, the **period**. A 440 Hz tone
repeats every 2.27 milliseconds.

Any sound that repeats, such as a sung vowel, is a sum of tones at whole
multiples of its lowest frequency, called **harmonics**.

## Measure a level in decibels

A **decibel** (dB) measures a ratio on a log scale. Hearing handles a range of
about a million to one in pressure, and a log scale turns that range into
numbers from 0 to about 120. The rule depends on what you measure:

$$
L_{\text{power}} = 10 \log_{10} \frac{P}{P_{\text{ref}}}, \qquad
L_{\text{amplitude}} = 20 \log_{10} \frac{A}{A_{\text{ref}}}
$$

Read it aloud: for a power, take ten times the log of the ratio. For an
amplitude, take twenty times the log. The two agree because power grows with
the square of amplitude, and $\log_{10}(A^2) = 2 \log_{10} A$. Doubling an
amplitude adds 6.02 dB. Doubling a power adds 3.01 dB.

> [!KEY]
> Use `20 * np.log10` for amplitudes and magnitudes, and `10 * np.log10` for
> powers and energies. Mixing them up doubles or halves every level.

A digital signal is measured in **dBFS**, decibels relative to full scale.
Full scale is an amplitude of 1.0, so 0 dBFS is the loudest sample a float
signal can hold. You can measure two things. The **peak** level uses the
largest absolute sample. The **RMS** level uses the root mean square: square
every sample, average, and take the square root. RMS tracks loudness better
than the peak does.

$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}
$$

Read it aloud: the RMS of $N$ samples is the square root of the mean of their
squares. In numpy:

```python
rms = np.sqrt(np.mean(x ** 2))
level = 20 * np.log10(max(rms, eps))
```

The `eps` floor matters. A silent clip has an RMS of 0, and `np.log10(0)` is
minus infinity, which breaks every later sum or mean.

### Work a level by hand

Take a sine of amplitude 0.5. Its peak is 0.5, so its peak level is
$20 \log_{10} 0.5 = -6.02$ dBFS. A sine of amplitude $A$ has an RMS of
$A/\sqrt{2}$, so this one has an RMS of $0.5 / 1.4142 = 0.3536$ and an RMS
level of $20 \log_{10} 0.3536 = -9.03$ dBFS. A full-scale sine reads 0 dBFS
peak and $-3.01$ dBFS RMS.

Some tools follow the AES17 convention instead, which calls the RMS of a
full-scale sine 0 dBFS. This course uses the peak-referenced one.

## Sample a signal

A computer stores a curve's value at evenly spaced instants. Each value is a
**sample**, and the number of samples per second is the **sample rate** $f_s$,
in hertz. Sample $n$ is the curve's value at time $n / f_s$:

$$
x[n] = x_c\!\left(\frac{n}{f_s}\right), \qquad n = 0, 1, 2, \dots
$$

Read it aloud: the $n$-th sample is the continuous signal $x_c$ at time
$n / f_s$. Square brackets mark a sampled signal. In numpy:

```python
n = np.arange(round(seconds * sample_rate))
x = amplitude * np.cos(2 * np.pi * frequency * n / sample_rate + phase)
```

Speech systems use a few standard rates. Telephone audio is 8 kHz. Most speech
models, including Whisper and wav2vec 2.0, take 16 kHz. CDs use 44.1 kHz and
video uses 48 kHz. A second of 16 kHz audio is 16,000 numbers.

The **Nyquist frequency** is half the sample rate, $f_s / 2$. The sampling
theorem says that if a signal holds no energy at or above the Nyquist
frequency, its samples determine it exactly. You can rebuild the curve between
the samples by placing a pulse called a **sinc**,
$\operatorname{sinc}(u) = \sin(\pi u) / (\pi u)$, on each sample and adding
them up. Each pulse is 1 at its own sample and 0 at every other one. The
resampler you build at the end of this lesson puts that idea into practice.

## Aliasing

Above the Nyquist frequency, the samples lie. A tone at frequency $f$ and a
tone at $f - k f_s$, for any whole number $k$, produce exactly the same
samples. So a tone above $f_s / 2$ shows up as a lower one. This is
**aliasing**, and the lower frequency is the **alias**.

### Work an alias by hand

Sample a cosine of amplitude 0.3 at 7 kHz with $f_s = 10$ kHz. Each sample
advances the angle by $2\pi \times 7000 / 10000 = 1.4\pi$ radians. The first
six samples are 0.3, $-0.0927$, $-0.2427$, 0.2427, 0.0927, and $-0.3$.

Now sample a 3 kHz cosine at the same rate. Each sample advances the angle by
$0.6\pi$. An advance of $1.4\pi$ is the same as $1.4\pi - 2\pi = -0.6\pi$, and
a cosine doesn't care about the sign of its angle. So the 3 kHz samples are
the same six numbers. From the samples alone, nothing tells the two tones
apart, and a player reconstructs the 3 kHz one.

The alias is the distance from $f$ to the nearest whole multiple of $f_s$:

$$
f_a = \left| f - f_s \cdot \operatorname{round}\!\left(\frac{f}{f_s}\right) \right|
$$

Read it aloud: divide the frequency by the sample rate and round to the
nearest whole number $k$. Subtract $k f_s$ from $f$ and take the absolute
value. The result always lies between 0 and $f_s/2$. In numpy it is one line:
`abs(f - fs * np.round(f / fs))`.

The following table works the formula for a few tones.

| Tone | Sample rate | $k$ | Alias |
|---|---|---|---|
| 7 kHz | 10 kHz | 1 | 3 kHz |
| 9 kHz | 16 kHz | 1 | 7 kHz |
| 10 kHz | 16 kHz | 1 | 6 kHz |
| 25 kHz | 16 kHz | 2 | 7 kHz |
| 440 Hz | 16 kHz | 0 | 440 Hz |

The following demo lets you move a sine past the Nyquist frequency and watch
the samples trace the alias.

```demo
{"id": "sampling-and-aliasing", "caption": "Raise the sine above half the sample rate, then play the samples and hear the alias the readout predicts."}
```

> [!KEY]
> Samples taken at $f_s$ can't tell a tone at $f$ from one at $f - k f_s$. Any
> energy above $f_s/2$ must be removed before sampling, or it lands at a false
> lower frequency and stays there.

So every converter has an **anti-aliasing filter**, a low-pass filter that
removes everything above the Nyquist frequency before sampling. A real filter
needs room to roll off: CDs sample at 44.1 kHz so the filter can pass 20 kHz
and still fall silent by 22.05 kHz.

## Resample with a windowed sinc

Datasets arrive at 44.1 or 48 kHz, and models want 16 kHz. **Resampling**
changes the rate of a digital signal, and it has the same problem. Changing
the rate by a ratio $L/M$ takes three steps:

1. **Upsample by $L$.** Insert $L - 1$ zeros after every sample. The signal is
   $L$ times longer, and its spectrum holds $L - 1$ unwanted copies, called
   **images**.
2. **Low-pass filter.** Remove everything above the lower of the two Nyquist
   frequencies, which also removes the images. In cycles per sample at the
   upsampled rate, the cutoff is $0.5 / \max(L, M)$. Scale the filter by $L$
   to restore the amplitude the zeros diluted.
3. **Downsample by $M$.** Keep every $M$-th sample.

Going from 48 kHz to 16 kHz is $L/M = 1/3$. Going from 16 kHz to 44.1 kHz is
$441/160$.

The ideal low-pass filter is a sinc that never ends. A **windowed-sinc**
filter cuts it to $K$ **taps** (filter coefficients) and tapers the cut with a
window $w[n]$, so it ends smoothly:

$$
h[n] = 2 f_c \operatorname{sinc}\!\big(2 f_c \, (n - \tfrac{K-1}{2})\big) \, w[n],
\qquad n = 0, \dots, K-1
$$

Read it aloud: tap $n$ is the ideal low-pass response for cutoff $f_c$, in
cycles per sample, centered on the middle tap, times the window. This lesson
uses the **Kaiser window**, `np.kaiser(K, beta)`, whose shape parameter
$\beta$ trades a sharper cutoff for more leakage. SciPy's `resample_poly` uses
$\beta = 5$ and $K = 2 \times 10 \max(L, M) + 1$ taps. After windowing, divide
the taps by their sum so that a constant signal passes through unchanged.

```python
m = np.arange(num_taps) - (num_taps - 1) / 2
h = 2 * cutoff * np.sinc(2 * cutoff * m) * np.kaiser(num_taps, beta)
h = h / h.sum()
```

### Work a small filter by hand

Take $K = 5$ taps and $f_c = 0.25$ cycles per sample. The offsets from the
middle are $m = -2, -1, 0, 1, 2$, and the ideal taps $0.5 \operatorname{sinc}(0.5 m)$
are 0, 0.3183, 0.5, 0.3183, and 0. The Kaiser window with $\beta = 5$ is
0.0367, 0.5529, 1, 0.5529, and 0.0367. Their products are 0, 0.1760, 0.5,
0.1760, and 0, which sum to 0.8520. Dividing by that sum gives the final taps:
0, 0.2066, 0.5869, 0.2066, and 0.

A symmetric filter of $K$ taps delays the signal by $(K-1)/2$ samples. For
48 kHz to 16 kHz, $K = 61$, so the filter delays by 30 samples at 48 kHz. The
resampler skips those 30 samples before it starts keeping every third one,
and the output lines up with the input.

> [!WARNING]
> `x[::3]` is not a resampler. It skips the low-pass filter, so a 10 kHz
> component at 48 kHz lands at 6 kHz after it, right in the middle of the
> speech band. Always filter before you drop samples.

> [!RECAP]
> - A level in dBFS is `20 * np.log10(max(value, eps))` of an amplitude; RMS is `np.sqrt(np.mean(x ** 2))`.
> - Sample $n$ of a tone is `A * np.cos(2 * np.pi * f * n / fs + phase)`.
> - A tone at $f$ shows up at $|f - f_s \operatorname{round}(f / f_s)|$.
> - A windowed-sinc low-pass filter is the sinc centered on the middle tap, times a Kaiser window, divided by its sum.
> - Resampling by $L/M$ is zero-stuffing by $L$, low-pass filtering with gain $L$, skipping the filter delay, and keeping every $M$-th sample.

## Build it step by step

The checks test the functions in this order.

1. **`rms(x)`** returns a float. Common mistake: `np.sqrt(np.sum(x ** 2))`,
   which grows with the length of the clip.
2. **`peak_dbfs(x, eps)`** and **`rms_dbfs(x, eps)`** floor `np.max(np.abs(x))`
   or your `rms` at `eps` and apply `20 * np.log10`. Common mistake: using 10,
   which reports a 0.5 sine at $-3.01$ dBFS instead of $-6.02$.
3. **`sample_tone(frequency, sample_rate, seconds, amplitude, phase)`** returns
   `round(seconds * sample_rate)` cosine samples at times `np.arange(N) / sample_rate`.
   Common mistake: `np.linspace(0, seconds, N)`, which stretches every step.
4. **`alias_frequency(frequency, sample_rate)`** returns a float in
   $[0, f_s/2]$. Use `np.round` and `abs`. Common mistake: `f % fs`, which
   returns 9 kHz for a 25 kHz tone at 16 kHz instead of 7 kHz.
5. **`lowpass_taps(num_taps, cutoff, beta)`** returns the windowed sinc from
   `np.sinc` and `np.kaiser`, divided by its sum. Common mistake:
   `np.sinc(np.pi * u)`; `np.sinc` already includes the $\pi$.
6. **`resample(x, up, down, half_width)`** reduces the ratio with `gcd`,
   builds `lowpass_taps(2 * half + 1, 0.5 / rate) * up`, zero-stuffs, calls
   `np.convolve`, and slices `[half : half + n_out * down : down]`. Common
   mistake: starting the slice at 0, which shifts the output by the filter
   delay and fails the comparison with `resample_poly`.

## Cost and scaling

Levels, sampling, and alias prediction are one pass over the data, $O(N)$ for
$N$ samples. The resampler is where the cost lives. The following table counts
multiplies per output sample for a filter of $K$ taps.

| Method | Multiplies per output sample | 48 to 16 kHz, $K = 61$ |
|---|---|---|
| Zero-stuff, convolve all, then keep every $M$-th | $K M$ | 183 |
| Polyphase: compute only the kept outputs and skip the zeros | about $K / L$ | about 61 |

Your `resample` is the first row: it computes every filtered sample, throws
most away, and spends most multiplies on inserted zeros. A **polyphase**
resampler, such as `scipy.signal.resample_poly`, computes only the outputs it
keeps, with the same result. For 16 kHz to 44.1 kHz, $K = 8821$ and the gap is
a factor of thousands. Memory is $O(N L)$ for the direct method and $O(N)$ for
polyphase.

## Where this breaks

**The filter has a transition band.** A 61-tap filter can't drop from full
gain to silence at exactly 8 kHz. Measured on the 48 kHz to 16 kHz filter, a
7 kHz tone loses 0.25 dB and a 7.5 kHz tone loses 1.8 dB. An 8.5 kHz tone
survives at $-14.3$ dB and aliases to 7.5 kHz. More taps narrow the band at a
higher cost.

**Libraries resample without telling you.** `librosa.load` resamples to
22,050 Hz and mixes to mono unless you pass `sr=None`. A 16 kHz model then
fails quietly. Print the rate of every clip you load.

**Resamplers disagree slightly.** SciPy, torchaudio, and SoX use different
windows and lengths, and their outputs differ at about the $10^{-3}$ level.
Compare resampled audio with a tolerance or a signal-to-noise ratio, never
sample by sample across libraries.

**The ends are only half filtered.** The first and last $(K-1)/2$ input
samples see only half the filter. The checks skip 200 output samples at each
end for this reason.

**Some resamplers assume the clip repeats.** `scipy.signal.resample` treats
the clip as one period of a loop, so the end leaks into the start. Use
`resample_poly` for audio.

## The exercise

You build `rms`, `peak_dbfs`, `rms_dbfs`, `sample_tone`, `alias_frequency`,
`lowpass_taps`, and `resample`. Press **Run** to hear a 7 kHz tone sampled at
10 kHz come out at 3 kHz. Then hear a sweep to 16 kHz brought down to a 16 kHz
rate two ways: by every third sample, where it turns around at 8 kHz, and by
your resampler, where it fades out.

The checks confirm your levels for full-scale sines, half-scale sines, and
silence, and that 7 kHz and 3 kHz give the same samples at 10 kHz. They test
`alias_frequency` against a table and against the spectrum of the sampled
tone, `lowpass_taps` against `scipy.signal.firwin`, and `resample` against
`scipy.signal.resample_poly`. The last two checks need a resampled 1 kHz tone
within 50 dB SNR of a synthesized one, and a 10 kHz tone at least 40 dB below
what naive decimation leaves.
