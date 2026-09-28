---
title: Pitch tracking
slug: 10-pitch-tracking
part: "Part 2 — Hearing and speech production"
summary: Build YIN step by step, track the pitch of a voice frame by frame, decide which frames are voiced, and score it by gross pitch error.
minutes: 60
gpu: false
objectives:
  - Explain the difference between F0 and pitch, and why autocorrelation makes octave errors.
  - Build YIN's difference function, cumulative mean normalized difference, absolute threshold, and parabolic interpolation.
  - Track F0 frame by frame with a voicing decision from YIN's aperiodicity.
  - Score a pitch track by gross pitch error and voicing decision error against a known F0.
prereqs:
  - 04-the-short-time-fourier-transform
  - 09-speech-production-and-lpc
exercise: 10-pitch-tracking
---

# Pitch tracking

The same words mean different things with a different melody. "You're
leaving" falls at the end as a statement and rises as a question. In tonal
languages such as Mandarin, the pitch contour changes which word you said.
Text-to-speech systems must predict that contour, voice conversion must move
it, and synthetic speech is often judged by how close its contour comes to
a real one.

The contour is the *fundamental frequency*, $F_0$, of the voice over time: the
rate of the vocal folds, which you met as the source in the source-filter
model. Estimating it sounds easy, because a voiced frame repeats itself once
per period. It is not. A waveform that repeats every period also repeats every
two periods, and the obvious method often picks the wrong one.

In this lesson you build YIN, a pitch estimator from 2002 that fixes the
obvious method in five small steps, each measured in the original paper. Then
you track a voice frame by frame, decide which frames have a pitch at all, and
score the result.

## What you already know

- A voiced frame is a pulse train filtered by the vocal tract, so it repeats every $T_0 = 1/F_0$ seconds.
- The autocorrelation $r(\tau)$ sums a frame times itself shifted by $\tau$ samples.
- A periodic signal's spectrum has harmonics at $F_0, 2F_0, 3F_0, \dots$
- The cepstrum shows the pitch period as a peak at quefrency $f_s / F_0$.
- A centered STFT pads by half a frame and puts frame $t$ at sample $t \cdot H$ for hop $H$.

## F0 and pitch

$F_0$ is a physical quantity: the repetition rate of a periodic signal. *Pitch*
is a perception: how high a sound seems. They usually agree, but not always.
Play harmonics at 400, 600, and 800 Hz with nothing at 200 Hz, and listeners
hear a 200 Hz pitch. This *missing fundamental* is common in practice: a
telephone line cuts everything below 300 Hz, and a man's voice still sounds
low. A pitch tracker should find the period, 1/200 s, whether or not the
fundamental is present.

A tracker also has to decide whether a frame has a pitch at all. Vowels and
voiced consonants such as "m" do. Silence, "s", and "f" do not. This is the
*voicing decision*, and it matters as much as the frequency.

## Autocorrelation and its octave errors

The classic method takes the autocorrelation of a frame and picks the lag
with the biggest value. With a frame $x$ of $W + \tau_{\max}$ samples:

$$
r(\tau) = \sum_{j=0}^{W-1} x_j\, x_{j+\tau}
$$

Read it aloud: multiply the first $W$ samples by the $W$ samples that start
$\tau$ later, and add. When $\tau$ is the period, the two stretches line up
and the sum is large. $F_0$ is then $f_s / \tau$.

The trouble is that $\tau = 2T_0$, $3T_0$, and so on line up equally well. Over
a search range of 60 Hz to 500 Hz, a voice at 300 Hz has five multiples of its
period in range, and small amplitude changes decide which peak wins. On 60
synthetic voices with random $F_0$ from 80 Hz to 400 Hz, random formants, and
20 dB of noise, picking the largest autocorrelation value gives the wrong
octave, or worse, on 52% of them. A strong first formant makes a second kind
of error: a peak at a short lag, one period of the formant.

> [!KEY]
> A signal that repeats every period also repeats every two periods. The
> largest autocorrelation peak often sits at a multiple of the period, and
> the estimate lands an octave or more too low.

## YIN in five steps

YIN, from de Cheveigné and Kawahara, starts from the autocorrelation and
changes one thing at a time. Their paper's Table I measures each step's gross
error rate on real speech, with a 25 ms window, a search range of 40 Hz to
800 Hz, and a threshold of 0.1. The following table reproduces it.

| Step | What it adds | Gross error |
|---|---|---|
| 1 | Autocorrelation | 10.0% |
| 2 | Difference function | 1.95% |
| 3 | Cumulative mean normalized difference | 1.69% |
| 4 | Absolute threshold | 0.78% |
| 5 | Parabolic interpolation | 0.77% |
| 6 | Best local estimate | 0.50% |

This lesson builds steps 2 to 5. Step 6 searches neighboring frames for a
better minimum and is left out.

### The difference function

Instead of asking "how similar is the signal to its shifted self", ask "how
different":

$$
d(\tau) = \sum_{j=0}^{W-1} \left(x_j - x_{j+\tau}\right)^2 = r_0(0) + r_\tau(0) - 2r(\tau)
$$

Read it aloud: subtract the shifted stretch from the first, square, and add.
A periodic signal gives $d(T_0) = 0$. Expanding the square gives two energy
terms, the energy of the first stretch $r_0(0)$ and of the shifted stretch
$r_\tau(0)$, minus twice the autocorrelation. When the signal grows louder over
the frame, the autocorrelation rises with lag and favors a long period. The
difference function subtracts that energy back out. That alone takes the error
rate from 10.0% to 1.95%.

```python
shifted = sliding_window_view(frame, W)[: max_lag + 1]  # row tau is frame[tau : tau + W]
d = np.sum((shifted - frame[:W]) ** 2, axis=1)
```

### The cumulative mean normalized difference

$d(0) = 0$ always, and $d$ is small at short lags for any smooth signal, so the
raw minimum is useless. YIN divides each value by the average of all the values
before it:

$$
d'(\tau) =
\begin{cases}
1, & \tau = 0 \\[1mm]
\dfrac{d(\tau)}{\frac{1}{\tau}\sum_{j=1}^{\tau} d(j)}, & \tau > 0
\end{cases}
$$

Read it aloud: at lag 0 the value is 1 by definition. At every other lag, it is
the difference at that lag over the running mean of the differences from lag 1
up to it. A value near 1 means "no better than average". A value near 0 means
"this lag lines up much better than the ones before it". The result does not
depend on the signal's loudness, so one threshold works for every frame.

```python
dprime = np.ones_like(d)
dprime[1:] = d[1:] * np.arange(1, len(d)) / np.cumsum(d[1:])
```

### The absolute threshold

A deeper dip often sits at $2T_0$. YIN takes the *first* dip below a fixed
threshold, 0.1 in the paper, rather than the deepest. It scans upward from
$\tau_{\min}$, stops at the first $\tau$ with $d'(\tau) < 0.1$, and slides
forward while the next value is smaller, to reach the bottom of that dip. When
no lag goes under the threshold, it takes the global minimum. The paper
notes that the exact threshold value matters little.

### Parabolic interpolation

The true period is rarely a whole number of samples. At 16 kHz, 120 Hz is a
period of 133.33 samples, and a lag of 133 gives 120.30 Hz. Fit a parabola
through the dip and its two neighbors, and take its vertex:

$$
\tau^* = \tau + \frac{d'(\tau - 1) - d'(\tau + 1)}{2\left(d'(\tau - 1) - 2d'(\tau) + d'(\tau + 1)\right)}
$$

Read it aloud: the correction is the difference between the left and right
neighbors, over twice the curvature. If the neighbors are equal, the dip is
centered and the correction is 0. The paper interpolates $d$; librosa
interpolates $d'$, and so does this lesson. The value $d'(\tau)$ at the dip is
the frame's *aperiodicity*: near 0 for a clean voice, near 1 for noise.

### Work YIN by hand

Take the signal $0, 1, 0, -1$ repeated, so the period is 4 samples, with
$W = 4$ and lags 0 to 5. The frame is $[0, 1, 0, -1, 0, 1, 0, -1, 0]$.

1. Differences: $d(1) = (0 - 1)^2 + (1 - 0)^2 + (0 + 1)^2 + (-1 - 0)^2 = 4$.
   In the same way, $d = [0, 4, 8, 4, 0, 4]$ for $\tau = 0, \dots, 5$.
2. Running sums from lag 1: 4, 12, 16, 16, 20. So
   $d'(1) = 4 \cdot 1 / 4 = 1$, $d'(2) = 8 \cdot 2 / 12 = 1.333$,
   $d'(3) = 4 \cdot 3 / 16 = 0.75$, $d'(4) = 0$, and $d'(5) = 4 \cdot 5 / 20 = 1$.
   With $d'(0) = 1$: $d' = [1, 1, 1.333, 0.75, 0, 1]$.
3. The first value under 0.1 is at $\tau = 4$, and $d'(5) = 1$ is larger, so the
   dip ends there. $F_0 = f_s / 4$.
4. Interpolation: $\tau^* = 4 + (0.75 - 1) / (2(0.75 - 0 + 1)) = 3.929$. The
   dip is not symmetric, so the vertex moves slightly toward the smaller
   neighbor.

Change the waveform to $0, 1, 0.5, -1$ and $d = [0, 4.5, 8.5, 4.5, 0, 4.5]$,
$d' = [1, 1, 1.308, 0.771, 0, 1.023]$, and YIN again picks $\tau = 4$.

> [!KEY]
> YIN looks for the first lag where the signal differs little from its shifted
> self, relative to the lags before it. Normalizing by the running mean and
> taking the first dip under a threshold removes most octave errors.

## Track frames and decide voicing

A pitch track runs YIN on every frame. Use the STFT convention: pad the
signal by half a frame at each end by reflection, and center frame $t$ on
sample $t \cdot H$. For YIN, the frame must hold $W + \tau_{\max} + 1$ samples,
and $W$ should cover at least one full period at $f_{\min}$. At 16 kHz with
$f_{\min} = 60$ Hz, $\tau_{\max} = 267$, so a 1024-sample frame, 64 ms, leaves
$W = 756$.

The voicing decision uses the aperiodicity. A frame with $d'(\tau) < 0.25$ is
voiced, and its $F_0$ is YIN's estimate. Otherwise the track stores 0. Noise
gives $d'$ close to 1 at every lag, and silence gives exactly 1, because the
running sum is 0.

The following demo runs YIN on a short clip.

```demo
{"id": "pitch-tracker", "caption": "Lower the threshold toward 0.02 to see octave-low errors in the creaky ending, then raise it past 0.2 to see breath frames get a pitch and, past 0.3, octave-high errors."}
```

## Score a pitch track

Two numbers score a track against a reference $F_0$. The *gross pitch error*
(GPE) is the share of frames voiced in both tracks whose estimate is off by
more than 20%:

$$
\text{GPE} = \frac{\left|\{t : \hat f_t > 0,\ f_t > 0,\ |\hat f_t - f_t| > 0.2 f_t\}\right|}{\left|\{t : \hat f_t > 0,\ f_t > 0\}\right|}
$$

Read it aloud: among frames where both the estimate $\hat f_t$ and the truth
$f_t$ are voiced, count those more than 20% off, and divide by how many there
are. The 20% bound catches octave errors, which are 50% or 100% off, and
ignores small inaccuracies. The YIN paper used it. The *voicing decision error*
is the share of all frames where the two tracks disagree about voicing.

For the reference $[0, 100, 100, 200, 200, 0, 150, 150, 0, 0]$ Hz and the
estimate $[0, 101, 50, 205, 250, 120, 0, 179, 0, 0]$ Hz, five frames are voiced
in both. Two are gross errors: 50 against 100 and 250 against 200, 25% off.
GPE is 2/5 = 40%. Voicing disagrees on frames 5 and 6, so the voicing error is
2/10 = 20%.

> [!RECAP]
> - Autocorrelation peaks at every multiple of the period, which causes octave errors.
> - YIN's difference function $d(\tau)$ removes the energy bias; $d'(\tau)$ divides by the running mean.
> - Take the first dip of $d'$ under 0.1, follow it to its bottom, and refine it with a parabola.
> - $d'$ at the dip is the aperiodicity, which decides voicing.
> - GPE counts voiced frames more than 20% off; voicing error counts disagreements.

## Build it step by step

The checks test the functions in this order.

1. **`difference_function(frame, max_lag)`** returns `max_lag + 1` values.
   Use `sliding_window_view` to get every shifted stretch at once. The usual
   mistake is letting $W$ shrink with the lag, which biases $d$ downward at
   long lags.
2. **`cmndf(d)`** returns $d'$ with `np.cumsum`. The usual mistake is dividing
   by the running sum without multiplying by $\tau$, which is the sum, not
   the mean. Guard the zero sum of a silent frame.
3. **`absolute_threshold(dprime, tau_min, tau_max, threshold)`** scans,
   slides to the bottom, or falls back to `np.argmin`. The usual mistake is
   returning `np.argmin` every time, which brings the octave errors back.
4. **`parabolic_interpolation(y, tau)`** applies the vertex formula, and
   returns `tau` at the edges or on a straight line.
5. **`yin(frame, sample_rate, fmin, fmax, threshold)`** sets
   $\tau_{\min} = \lfloor f_s / f_{\max} \rfloor$ and
   $\tau_{\max} = \lceil f_s / f_{\min} \rceil$, and chains the four
   functions. It returns $f_s / \tau^*$ and the aperiodicity.
6. **`gross_pitch_error(f0_est, f0_ref, tolerance)`** and
   **`voicing_error(f0_est, f0_ref)`** are boolean masks and means.
7. **`autocorr_pitch(frame, sample_rate, fmin, fmax)`** is step 1 of the
   table, for comparison: the same windows, a dot product instead of a
   squared difference, and `np.argmax`.
8. **`track_pitch(x, sample_rate, frame_length, hop, ...)`** pads by
   reflection, loops over frames, and applies the voicing threshold.

## Cost and scaling

For $T$ frames, a window $W$, and $\tau_{\max}$ lags, the steps cost the
following.

| Step | Time per frame | Memory per frame |
|---|---|---|
| Difference function, direct | $O(W \tau_{\max})$ | $O(W \tau_{\max})$ with `sliding_window_view` |
| Difference function by FFT | $O(N \log N)$, $N = W + \tau_{\max}$ | $O(N)$ |
| CMNDF, threshold, interpolation | $O(\tau_{\max})$ | $O(\tau_{\max})$ |

The direct form is about 200,000 multiply-adds per frame at the settings in
this lesson, or 20 million per second of audio at 100 frames per second. That
is fast enough on a laptop. Long recordings compute $r(\tau)$ with FFTs
instead, as librosa's `yin` does.

## Where this breaks

**The search range misses $F_0$.** A voice at 50 Hz with $f_{\min} = 60$ Hz
gets a harmonic instead. Set the range for the speaker.

**Octave jumps between frames.** A single frame can still pick $2T_0$. pYIN
replaces the single threshold with a distribution over thresholds, keeps
several candidates per frame, and chooses a smooth path through them with an
HMM. For more information about the Viterbi algorithm behind that path, see
*Hidden Markov models*.

**Creaky voice and breathy voice.** Irregular periods or strong noise push
$d'$ up, and the frame is marked unvoiced when it is voiced.

**Frames that straddle a boundary.** A 64 ms frame at a vowel's edge holds
half voice and half silence. Voicing flips a frame or two late.

**Polyphony.** Two voices or background music put two periods in one frame.
YIN finds one of them, or neither. Neural trackers such as CREPE, a
convolutional network over 1024-sample frames of 16 kHz audio with 360 pitch
bins 20 cents apart, handle noise better but need training data.

> [!WARNING]
> Frames and lags are both counted in samples, and they are easy to mix up. A
> hop of 160 samples is 10 ms, and a lag of 160 samples is a pitch of 100 Hz.
> Convert with $f_s$ every time: $F_0 = f_s / \tau$, time $= t \cdot H / f_s$.

## The exercise

You build `difference_function`, `cmndf`, `absolute_threshold`,
`parabolic_interpolation`, `yin`, `gross_pitch_error`, `voicing_error`,
`autocorr_pitch`, and `track_pitch`. Press **Run** to hear a synthetic voice
ask a question, see its spectrogram and your pitch track against the truth,
compare YIN with the autocorrelation peak on 60 random voices, and hear your
track hummed back as a sine.

The checks work the hand examples for $d$, $d'$, and the threshold, compare
$d$ with a brute-force sum, and verify the parabola's vertex. They require YIN
within 0.5% on clean harmonic tones and within 0.1% when the period falls
between two lags, 150 Hz on a missing fundamental, and a
gross error under 2% on 60 noisy voices. They score a hand-made track, check
`autocorr_pitch` against its definition, and track a gliding voice with a
noise burst and silence, requiring the voicing right on at least 95% of
frames and every clear voiced frame within 5%.
