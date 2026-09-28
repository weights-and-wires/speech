---
title: Phase and Griffin-Lim
slug: 05-phase-and-griffin-lim
part: "Part 1 — Sound and signals"
summary: See why a magnitude spectrogram isn't a signal, recover a phase for it with Griffin-Lim, and stretch time without changing pitch.
minutes: 70
gpu: false
objectives:
  - Explain what the phase of an STFT holds, and why most complex arrays are not the STFT of any signal.
  - Build Griffin-Lim, measure its spectral convergence, and confirm that it never gets worse.
  - Add the fast Griffin-Lim momentum step and compare the two at equal iterations.
  - Build a phase vocoder that stretches a signal in time and keeps its pitch.
prereqs:
  - 04-the-short-time-fourier-transform
exercise: 05-phase-and-griffin-lim
---

# Phase and Griffin-Lim

Many speech models predict a spectrogram, not a waveform. A text-to-speech
model draws a picture of the sound, and something else has to turn the
picture back into samples. The picture holds only magnitudes: how much of each
frequency, in each frame. The STFT also had a phase for every cell, and the
model threw it away. Without it, the inverse STFT from the previous lesson has
nothing to work with.

This lesson finds a phase that fits. **Griffin-Lim** guesses a phase, builds a
signal, measures that signal's own STFT, keeps its phase, and repeats. Each
round makes the magnitudes match a little better. It's the classic answer,
the baseline every neural vocoder is compared against, and the tool later
lessons use to hear the spectrograms their models generate. Then you use the
same phase bookkeeping for a different job: slowing speech down without
turning it into a lower voice.

## What you already know

- The STFT $X[k, m]$ is complex: its magnitude is how much of bin $k$ frame $m$ holds, and its angle is the phase.
- A spectrogram is (bins, frames), centered with reflect padding and a periodic Hann window.
- The least-squares inverse STFT overlap-adds windowed frames and divides by $\sum w^2$.
- With a hop of $N/4$, every sample sits in four frames.
- `np.angle(z)` returns an angle in $(-\pi, \pi]$, and `np.exp(1j * theta)` builds a unit complex number.

## Magnitude is not the whole signal

The phase of bin $k$ in frame $m$ says where, in its cycle, the sinusoid at
that bin sits at the frame's center. It depends on the exact position of the
frame, it wraps around every $2\pi$, and it looks like noise when you plot it.
But it carries the timing. A click and a burst of noise can have similar
magnitudes; their phases are what make one sharp and the other smeared.

Throw the phase away and invert anyway, and you hear what it did. Setting
every phase to zero lines every frame's content up at the same instants and
adds a mechanical buzz tied to the frame rate. Setting every phase at random
sounds rough and hoarse.

The deeper problem is **consistency**. With 512-sample frames and a hop of
128, each second of 16 kHz audio becomes 126 frames of 257 complex numbers:
about 32,000 complex values for 16,000 real samples. The frames overlap, so
neighboring frames must agree about the samples they share. Most complex
arrays of that shape don't agree, and so are not the STFT of any signal. Only
**consistent** arrays are.

The **spectral convergence** measures how far a signal's magnitudes are from
the target $S$:

$$
\text{SC}(X, S) = \frac{\big\lVert\, |X| - S \,\big\rVert_F}{\lVert S \rVert_F}
$$

Read it aloud: take the magnitude of $X$, subtract the target cell by cell,
and take the Frobenius norm, the square root of the sum of squares. Divide by
the norm of the target. It is 0 for a perfect match and 0.1 when the
magnitudes are off by 10% overall.

```python
sc = np.linalg.norm(np.abs(X) - S) / np.linalg.norm(S)
```

Measured on a 0.512 s chirp, a real STFT survives an inverse and a forward
transform with an SC below $10^{-12}$. The same magnitudes with random phases
come back from one round trip with an SC of 0.647. Nearly two thirds of the
magnitude has moved, because the overlapping frames disagreed and the inverse
averaged them.

## Griffin-Lim

Griffin-Lim alternates between two requirements. One set of arrays has the
right magnitude. The other set is consistent, the STFTs of real signals.
Griffin-Lim steps from one set to the other and back, and each step goes to
the nearest point of the set it moves to. That is an **alternating
projection**:

$$
x_{i+1} = \operatorname{ISTFT}\!\left(S \odot e^{j \angle X_i}\right), \qquad
X_{i+1} = \operatorname{STFT}(x_{i+1})
$$

Read it aloud: keep the phase of the current STFT $X_i$, written
$\angle X_i$, and put the target magnitude $S$ under it. The symbol $\odot$
means cell by cell. Invert that to a signal $x_{i+1}$ with the least-squares
inverse, then take the signal's STFT. The inverse finds the real signal
closest to the array you gave it, so its STFT is the nearest consistent array.
Start from $X_0 = S e^{j\phi_0}$ with random phases $\phi_0$.

```python
c = S * np.exp(2j * np.pi * rng.random(S.shape))
for _ in range(n_iter):
    c = stft(istft(S * np.exp(1j * np.angle(c)), hop), n_fft, hop)
x = istft(S * np.exp(1j * np.angle(c)), hop)
```

Griffin and Lim proved in 1984 that with the least-squares inverse, the
distance between the two sets never grows from one iteration to the next.
That distance is the numerator of the spectral convergence, so the SC never
rises. The following table shows it falling on the 0.512 s chirp.

| Iterations | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| Griffin-Lim | 0.647 | 0.368 | 0.314 | 0.272 | 0.237 | 0.205 | 0.165 | 0.128 |
| Fast Griffin-Lim | 0.647 | 0.368 | 0.301 | 0.244 | 0.205 | 0.152 | 0.105 | 0.089 |

The first iteration does the most. After that, the gains shrink: the
algorithm finds a phase that is locally good, not the original one. A result
at 0.1 to 0.2 sounds clearly like the source, with a slightly metallic,
phasey color.

> [!KEY]
> Griffin-Lim keeps the phase of the nearest real signal's STFT and puts the
> target magnitude back under it, again and again. With the least-squares
> inverse, the spectral convergence never gets worse.

Later in the course, a flow-matching model generates mel spectrograms of
spoken digits, and this lesson's `griffin_lim` turns them into sound. Its
signature is `griffin_lim(magnitude, n_fft, hop_length, n_iter=32, momentum=0.0, seed=0)`,
and it returns a 1-D signal of `hop_length * (frames - 1)` samples.

## Fast Griffin-Lim

Plain Griffin-Lim takes small steps once it is close. **Fast Griffin-Lim**,
from Perraudin, Balazs, and Søndergaard in 2013, adds momentum: it keeps
moving in the direction the last step went. Let $t_i$ be the consistent STFT
from iteration $i$. Instead of using $t_i$ directly, it extrapolates:

$$
c_i = t_i + \alpha \,(t_i - t_{i-1})
$$

Read it aloud: take the latest projection, and add $\alpha$ times the change
since the previous one. With $\alpha = 0$ this is plain Griffin-Lim. The next
iteration keeps the phase of $c_i$. On the first iteration there is no
$t_{i-1}$, so $c_1 = t_1$.

```python
t = stft(istft(S * np.exp(1j * np.angle(c)), hop), n_fft, hop)
c = t if previous is None else t + momentum * (t - previous)
previous = t
```

Work one cell by hand. Say the previous projection was $1$ and the latest is
$j$: the phase turned from 0° to 90°. With $\alpha = 0.99$,
$c = j + 0.99(j - 1) = -0.99 + 1.99j$, whose angle is 116.4°. The momentum
pushes the phase past where the projection stopped, in the direction it was
already moving.

librosa's `griffinlim` and torchaudio's `GriffinLim` both default to 32
iterations with a momentum of 0.99. librosa writes the update as
`t - (momentum / (1 + momentum)) * t_previous`, which is the same step divided
by $1 + \alpha$; only the phase is kept, so the scale doesn't matter. In the
table, momentum at 32 iterations is already below plain Griffin-Lim at 64.
The price is the guarantee: with momentum, the SC can rise from one
iteration to the next.

The following demo runs Griffin-Lim one iteration at a time.

```demo
{"id": "griffin-lim-iterations", "caption": "Step through the iterations with and without momentum, and listen to the roughness of the random start fade as the spectral convergence falls."}
```

## Stretch time with a phase vocoder

To slow speech down, you could play the samples at a lower rate, but that
lowers every frequency too. A **phase vocoder** changes the timing without
changing the frequencies. It reads the STFT at a different speed, for example
two output frames for every analysis frame to double the length, and
synthesizes with the original hop. The magnitudes are easy to interpolate.
The phases are the hard part, because each bin's phase must keep advancing at
the bin's own frequency across the output frames.

Each bin measures its frequency from how far its phase moves between frames.
A sinusoid at exactly bin $k$'s center turns by $\omega_k H$ radians in one
hop, where $\omega_k = 2\pi k / N$. Any extra turn means the true frequency
is a little off the center:

$$
\Delta\phi = \angle X[k, m+1] - \angle X[k, m] - \omega_k H, \qquad
\psi \leftarrow \psi + \omega_k H + \operatorname{wrap}(\Delta\phi)
$$

Read it aloud: the deviation $\Delta\phi$ is the measured phase change minus
the change the bin's center frequency would give. Wrap it to $[-\pi, \pi]$,
because phases only repeat every $2\pi$. The output phase $\psi$ then
advances by the expected turn plus the wrapped deviation for every output
frame, however many output frames one analysis step becomes.

### Work one bin by hand

Take a 440 Hz tone at 16 kHz, with $N = 1024$ and $H = 256$. The nearest bin
is $k = 28$, centered at $28 \times 16000 / 1024 = 437.5$ Hz. Its expected
turn per hop is $2\pi \times 28 \times 256 / 1024 = 14\pi = 43.982$ radians.
The tone really turns $2\pi \times 440 \times 256 / 16000 = 44.234$ radians.
The measured phase difference between two frames, wrapped, is 0.2513
radians more than the expected turn. So the bin's frequency is
$(43.982 + 0.2513) / (2\pi \times 256) \times 16000 = 440.00$ Hz, and the
output phase advances by 44.234 radians per output frame, whatever the
stretch. The pitch stays at 440 Hz.

> [!TIP]
> To shift pitch instead, stretch time by a factor $r$ with the phase vocoder,
> then resample by $1/r$ with the resampler from the lesson on sound and
> sampling. The length comes back and every frequency moves by $r$.

> [!RECAP]
> - The spectral convergence is `np.linalg.norm(np.abs(X) - S) / np.linalg.norm(S)`.
> - Griffin-Lim starts from `S * np.exp(2j * np.pi * rng.random(S.shape))` and repeats `stft(istft(S * np.exp(1j * np.angle(c))))`.
> - Fast Griffin-Lim replaces each projection $t$ with $t + \alpha(t - t_{\text{previous}})$.
> - The phase vocoder reads frames at steps of `1 / stretch`, interpolates magnitudes, and advances each bin's phase by $\omega_k H$ plus its wrapped deviation.

## Build it step by step

The functions `hann`, `stft`, and `istft` are provided. The checks test yours
in this order.

1. **`spectral_convergence(X, magnitude)`** returns a float. Use `np.abs` and
   `np.linalg.norm`. Common mistake: comparing power with magnitude, which
   reports a large error for a perfect match.
2. **`random_phase(magnitude, seed)`** draws `rng.random(magnitude.shape)` from
   `np.random.default_rng(seed)` and returns the complex array. Common
   mistake: drawing phases from `rng.normal`, which clusters them near 0.
3. **`griffin_lim(magnitude, n_fft, hop_length, n_iter, momentum, seed)`**
   starts from `random_phase`, loops `n_iter` times, and inverts at the end.
   Keep the previous projection in a variable that starts as `None`. Common
   mistake: applying momentum to the phase-only array `np.exp(1j * np.angle(c))`
   instead of to the projections themselves.
4. **`phase_vocoder(X, stretch, hop_length)`** appends a zero column, walks the
   positions `np.arange(0, frames, 1 / stretch)`, and keeps one running phase
   per bin. Common mistake: wrapping the running phase itself; wrap only the
   deviation.
5. **`time_stretch(x, stretch, n_fft, hop_length)`** chains `stft`,
   `phase_vocoder`, and `istft`. Common mistake: inverting with a hop of
   `stretch * hop_length`. The phase vocoder already made more frames at the
   original hop, so that stretches the signal twice.

## Cost and scaling

Each Griffin-Lim iteration is one inverse STFT and one STFT. For a signal of
$L$ samples, frame length $N$, and hop $H$, the following table gives the
cost.

| Step | Time | Memory |
|---|---|---|
| One Griffin-Lim iteration | $O((L / H) \, N \log N)$ | $O((L / H) \, N)$ for the STFT |
| $K$ iterations | $K$ times one iteration | the same, plus one previous STFT for momentum |
| Phase vocoder | $O((L / H) \, N \log N)$, plus $O(\text{stretch} \cdot (L / H) \cdot N)$ | $O(\text{stretch} \cdot (L / H) \cdot N)$ |

Thirty-two iterations on 0.512 s at $N = 512$ take about 25 ms on a laptop in
numpy. Griffin-Lim is serial: iteration $i + 1$ needs iteration $i$, so it
doesn't parallelize across iterations, only within each STFT. That, and its
metallic sound, are why neural vocoders replaced it for production speech.

## Where this breaks

**It needs a magnitude, not a power.** Feed Griffin-Lim $|X|^2$ and it
builds a signal whose magnitude is $|X|^2$: every loud component gets
louder and the result sounds harsh and wrong. Take the square root first. A
mel spectrogram must also be mapped back to linear bins before Griffin-Lim can
use it.

**The analysis settings must match.** The window, `n_fft`, and hop in the
loop must be the ones that made the magnitude. A different hop makes a
different set of consistent arrays, and the result doesn't converge.

**It recovers a signal, not the signal.** $x$ and $-x$ have the same
magnitudes, and so do many signals that differ by small shifts. Compare
Griffin-Lim results by spectral convergence, never sample by sample.

**Lengths lose the last partial hop.** The inverse returns
`hop_length * (frames - 1)` samples, so a signal that isn't a whole number of
hops comes back shorter, and its last frame can't match. Pad inputs to a
whole number of hops, as the Run block does with 1.536 s.

**The phase vocoder smears transients and freezes bin relationships.** Every
bin advances on its own, so the phases of neighboring bins drift apart, and
clicks and consonants turn soft and phasey. The relationships it starts from
are frame 0's. Stretch a sine that starts at sample 0 by 1.5 and its level
falls by 5.9 dB, because reflect padding makes frame 0's phases unlike every
later frame's. A cosine, which reflects cleanly, loses nothing. Phase-locked
vocoders, from Laroche and Dolson in 1999, tie each bin's phase to the
nearest spectral peak to fix both.

## The exercise

You build `spectral_convergence`, `random_phase`, `griffin_lim`,
`phase_vocoder`, and `time_stretch`. Press **Run** to hear a buzzy vowel-like
tone rebuilt from its magnitude four ways: zero phase, random phase,
Griffin-Lim, and fast Griffin-Lim. Then see the spectral convergence of both
algorithms against the iteration count, and hear the tone stretched to 1.5
times and 0.75 times its length at the same pitch.

The checks confirm the spectral convergence of an exact and a scaled STFT,
that random phases keep the magnitude and give an inconsistent array, and
that `griffin_lim` returns the right length, starts from `random_phase`, and
matches the reference update with and without momentum. They confirm that
plain Griffin-Lim never gets worse over 12 iterations, reaches an SC below
0.2 in 50, and that momentum 0.99 ends below 75% of plain Griffin-Lim at 32.
The last checks confirm that `phase_vocoder` returns its input at stretch 1,
and that a 440 Hz tone stretched by 1.5 and 0.75 keeps its length within a
hop and its pitch within a bin.
