---
title: The spectrogram of a tone sweep
slug: 99-smoke
part: "Smoke test"
summary: Synthesise a tone sweep, hear it, and draw its spectrogram with numpy, end to end through every part of the app.
minutes: 20
gpu: false
objectives:
  - Synthesise a linear tone sweep from its phase, and hear it.
  - Cut a signal into overlapping windowed frames and take the magnitude of each frame's FFT.
  - "Read a spectrogram: time runs left to right, frequency runs bottom to top."
exercise: 99-smoke
---

# The spectrogram of a tone sweep

This lesson exists to test the app, not to teach the course. It exercises every
moving part at once: a lesson with mathematics and two interactive demos, an
exercise whose **Run** button plays a clip and draws a spectrogram and a plot,
and checks that draw a figure of their own. Someone deletes it once the real
lessons arrive.

The subject is still real. A *tone sweep*, also called a *chirp*, is a sine wave
whose frequency changes over time. Its spectrogram is a single bright line that
climbs from the bottom left to the top right, which makes it the easiest signal
to check a spectrogram against.

## What you already know

- A sine wave repeats at a fixed rate, its frequency, measured in hertz (Hz).
- A digital signal is a list of numbers, one sample every `1 / sample_rate` seconds.
- The FFT turns a list of samples into the strength of each frequency it holds.
- numpy indexing can gather many slices of an array in one step.

## Sampling a sweep

A sine wave at a fixed frequency $f$ is $\sin(2\pi f t)$. The quantity inside
the sine is its *phase*, the angle the wave has turned through by time $t$. For
a sweep, the frequency rises steadily from $f_0$ to $f_1$ over $T$ seconds, so
the phase has to grow faster and faster.

The phase of a linear sweep is the following:

$$
\phi(t) = 2\pi \left( f_0 t + \frac{f_1 - f_0}{2T} t^2 \right)
$$

Read it aloud: the phase at time $t$ is two pi times the starting frequency
times $t$, plus a term that grows with the square of $t$. The rate at which the
phase turns, divided by two pi, is the frequency at that moment. Differentiate
and you get $f_0 + (f_1 - f_0) t / T$, which starts at $f_0$ and reaches $f_1$
at $t = T$.

In numpy, the whole sweep is two lines:

```python
t = np.arange(n_samples) / sample_rate
x = amplitude * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * seconds)))
```

Sampling puts a ceiling on the frequencies a signal can hold. A signal sampled
at 16 kHz can represent frequencies up to 8 kHz, half the sample rate. Above
that, a sine turns into a lower one that fits the same samples. The following
demo lets you push a sine past that limit and hear what the samples encode.

```demo
{"id": "sampling-and-aliasing", "caption": "Raise the sine above half the sample rate and watch the samples trace a slower wave."}
```

> [!KEY]
> Samples taken at a rate of `fs` can't tell a sine at `f` from one at `f - fs`.

## Frames and windows

A spectrogram answers the question "which frequencies are present, and when?"
One FFT over the whole sweep would say that every frequency from 200 Hz to
4 kHz appears, and nothing about when. So you cut the signal into short
*frames*, each a few tens of milliseconds long, and take one FFT per frame.

Frames overlap. The *hop* is the number of samples between the start of one
frame and the start of the next. With a frame length of 512 samples and a hop
of 128, each sample lands in four frames.

Cutting a frame out of a signal leaves sharp edges, and sharp edges smear
energy across every frequency. A *window* fades each frame in and out before
the FFT. The periodic Hann window is the common choice:

$$
w[n] = \frac{1}{2} - \frac{1}{2} \cos\left( \frac{2 \pi n}{N} \right)
$$

Read it aloud: sample $n$ of an $N$-sample window is one half minus one half
of a cosine that completes one full turn across the window. It is zero at the
start and rises to one in the middle. In numpy:

```python
window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
```

### Work one frame by hand

Take a signal of 20 samples, the numbers 0 to 19, a frame length of 8, and a
hop of 4. The number of whole frames is $1 + (20 - 8) / 4 = 4$. Frame 0 holds
samples 0 to 7, frame 1 holds 4 to 11, frame 2 holds 8 to 15, and frame 3 holds
12 to 19. A fifth frame would start at sample 16 and run past the end, so it
is dropped.

## The spectrogram

Stack the magnitude of each frame's FFT as a column, and the result is a
*spectrogram*: a matrix with one row per frequency bin and one column per
frame. Row $k$ is the frequency $k \cdot f_s / N$, where $f_s$ is the sample
rate and $N$ is the FFT length.

The length of the frame decides what the spectrogram can show. A long frame
separates frequencies that are close together but blurs events that are close
in time. A short frame does the opposite. The following demo lets you try that
trade on a signal built to show both sides of it.

```demo
{"id": "stft-explorer", "caption": "Lengthen the window until the two tones separate, then notice what happens to the two clicks."}
```

> [!TIP]
> Draw every spectrogram in decibels. The quiet parts of a sound are often a
> thousand times weaker than the loud ones, and a linear scale hides them.

> [!RECAP]
> - A sweep's phase grows with the square of time.
> - A frame is a slice of the signal; the hop is the step between slices.
> - A window fades each frame in and out before its FFT.
> - A spectrogram has one row per frequency and one column per frame.

## Build it step by step

1. **`tone_sweep(f0, f1, sample_rate, seconds, amplitude)`** returns the sweep
   as a float64 array. Build the time axis with `np.arange`, then the phase
   formula. The mistake people make most often is to multiply the
   *instantaneous* frequency by `t`, which sweeps twice as far as it should.
2. **`frame_signal(x, frame_length, hop)`** returns an array of shape
   `(n_frames, frame_length)`. Build a two-dimensional index array with
   broadcasting and index `x` once. Keep only whole frames.
3. **`magnitude_spectrogram(x, n_fft, hop)`** frames the signal, multiplies by
   the Hann window, calls `np.fft.rfft` along each frame, and takes `np.abs`.
   Transpose the result so rows are frequencies.
4. **`peak_track(spectrogram, sample_rate, n_fft)`** takes `np.argmax` down
   each column and converts the bin number to hertz.

## Cost and scaling

The following table shows how the cost of each step grows with the signal
length $L$, the FFT length $N$, and the hop $H$.

| Step | Time | Memory |
|---|---|---|
| The sweep | $O(L)$ | $O(L)$ |
| The frames | $O(L N / H)$ | $O(L N / H)$ |
| The FFTs | $O((L / H) N \log N)$ | $O(L N / H)$ |

Framing copies each sample $N / H$ times. For a minute of audio at 16 kHz with
$N = 512$ and $H = 128$, that is about 3.8 million numbers, or 30 MB as
float64. Longer recordings are processed in blocks.

## Where this breaks

- **The frame is too short for low frequencies.** A 512-sample frame at 16 kHz
  lasts 32 ms, and its bins are 31 Hz apart. Two notes closer than that merge.
  Use a longer frame when frequency detail matters.
- **The sweep aliases.** Push `f1` above half the sample rate and the line in
  the spectrogram folds back down. Keep every frequency under the Nyquist
  limit, or raise the sample rate.
- **The edges get lost.** Without padding, the first frame is centred half a
  frame after the signal starts. Libraries such as librosa pad the signal by
  half a frame at each end for this reason.

> [!WARNING]
> `np.fft.rfft` returns complex numbers. Take `np.abs` before you plot or
> compare them, or you compare phases by accident.

## The exercise

You build `tone_sweep`, `frame_signal`, `magnitude_spectrogram`, and
`peak_track`. Press **Run** to hear a sweep from 200 Hz to 4 kHz, see its
spectrogram, and see the peak frequency climb. The checks verify the sweep's
length, amplitude, and frequency at each end, the framing of a short signal,
the shape of the spectrogram, the height of a pure tone's peak, and that the
peak track follows the sweep. The last check draws your peak track beside the
expected one.
