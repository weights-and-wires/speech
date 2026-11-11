---
title: Phase and Griffin-Lim
slug: 008-phase-and-griffin-lim
part: "Part 1 — Sound and signals"
summary: See why a magnitude spectrogram isn't a signal, recover a phase for it with Griffin-Lim, and stretch time without changing pitch.
minutes: 45
gpu: false
objectives:
  - Explain what the phase of an STFT holds, and why most complex arrays are not the STFT of any signal.
  - Build Griffin-Lim, measure its spectral convergence, and confirm that it never gets worse.
  - Add the fast Griffin-Lim momentum step and compare the two at equal iterations.
  - Build a phase vocoder that stretches a signal in time and keeps its pitch.
prereqs:
  - 007-the-short-time-fourier-transform
exercise: 008-phase-and-griffin-lim
---

# Phase and Griffin-Lim

A text-to-speech model draws a spectrogram of the sentence it is about to say, and something else has to turn that picture into sound. The picture holds only magnitudes: how much of each frequency, in each frame. The STFT from the last chapter also kept a phase for every cell, and the model threw it away. Without it, the inverse STFT has nothing to work with.

You could fill the phase in with zeros or random numbers. Both sound wrong, and this chapter shows why. Then you build **Griffin-Lim**, the classic way to find a phase that fits: guess one, build a signal, look at that signal's own phase, and repeat. It is the baseline every neural vocoder is compared against. Finally you reuse the same phase bookkeeping to slow speech down without lowering the voice.

## What you already know

- The STFT $X[k, m]$ is complex: its magnitude says how much of bin $k$ frame $m$ holds, and its angle is the phase.
- A spectrogram is shaped (bins, frames), centered with reflect padding and a periodic Hann window.
- The least-squares inverse STFT overlap-adds windowed frames and divides by $\sum w^2$.
- With a hop of $N/4$, every sample sits in four frames.
- `np.angle(z)` returns an angle in $(-\pi, \pi]$, and `np.exp(1j * theta)` builds a unit complex number.

## Magnitude is not the whole signal

The phase of bin $k$ in frame $m$ says where, in its cycle, the sinusoid at that bin sits at the frame's center. It wraps around every $2\pi$ and looks like noise when you plot it, but it carries the timing. A click and a burst of noise can have similar magnitudes, and their phases are what make one sharp and the other smeared.

Throw the phase away and invert anyway, and you hear what it did. Zero phase lines every frame's content up at the same instants, which adds a mechanical buzz tied to the frame rate. Random phase sounds rough and hoarse.

The deeper problem is **consistency**. With 512-sample frames and a hop of 128, each second of 16 kHz audio becomes 126 frames of 257 complex numbers, about 32,000 complex values for 16,000 real samples. The frames overlap, so neighboring frames must agree about the samples they share. Most complex arrays of that shape don't agree, which means they are not the STFT of any signal. Only **consistent** arrays are.

To measure how far a signal's magnitudes are from a target magnitude $S$, use the **spectral convergence**:

$$
\text{SC}(X, S) = \frac{\big\lVert\, |X| - S \,\big\rVert_F}{\lVert S \rVert_F}
$$

Read it aloud: take the magnitude of $X$, subtract the target cell by cell, and take the Frobenius norm, which is the square root of the sum of squares. Divide by the norm of the target. It is 0 for a perfect match and 0.1 when the magnitudes are off by 10% overall.

```python
sc = np.linalg.norm(np.abs(X) - S) / np.linalg.norm(S)
```

Measured on a 0.512 s chirp, a real STFT survives an inverse and a forward transform with an SC below $10^{-12}$. The same magnitudes with random phases come back from one round trip with an SC of 0.647. Nearly two thirds of the magnitude moved, because the overlapping frames disagreed and the inverse averaged them.

> [!KEY]
> A magnitude spectrogram with made-up phases is almost never the STFT of any signal. Only a consistent array, whose overlapping frames agree about the samples they share, survives the inverse unchanged.

## Griffin-Lim

You have two requirements. The array must have the right magnitude $S$, and it must be consistent. Each requirement describes a set of arrays. Arrays with the right magnitude are easy to make, but they are not consistent. The STFTs of real signals are consistent, but their magnitudes are not $S$. You want an array in both sets.

Griffin-Lim steps from one set to the other and back, and each step lands on the nearest point of the set it moves to. This is an **alternating projection**, shown in Figure 1. One iteration is the following:

$$
\begin{aligned}
x_{i+1} &= \operatorname{ISTFT}\!\left(S \odot e^{j \angle X_i}\right) \\
X_{i+1} &= \operatorname{STFT}(x_{i+1})
\end{aligned}
$$

```figure
<svg viewBox="0 0 600 262" role="img" aria-label="The Griffin-Lim loop: combine the target magnitude with the current phase, invert to a signal, take its STFT, and keep that phase"><defs><marker id="c008-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="20.0" y="40.0" width="150.0" height="64.0" class="fill-soft-a"/>
<rect x="20.0" y="40.0" width="150.0" height="64.0" class="stroke-a"/>
<text x="95.0" y="62.0" class="fig-bold" text-anchor="middle">target S</text>
<text x="95.0" y="82.0" class="fig-small" text-anchor="middle">magnitude only</text>
<rect x="20.0" y="150.0" width="150.0" height="64.0" class="fill-soft-d"/>
<rect x="20.0" y="150.0" width="150.0" height="64.0" class="stroke-d"/>
<text x="95.0" y="172.0" class="fig-bold" text-anchor="middle">phase of X_i</text>
<text x="95.0" y="192.0" class="fig-small" text-anchor="middle">start: random</text>
<rect x="225.0" y="92.0" width="130.0" height="70.0" class="fill-soft-b"/>
<rect x="225.0" y="92.0" width="130.0" height="70.0" class="stroke-b"/>
<text x="290.0" y="114.0" class="fig-bold" text-anchor="middle">S · e^(j·phase)</text>
<text x="290.0" y="134.0" class="fig-small" text-anchor="middle">complex array</text>
<rect x="405.0" y="92.0" width="160.0" height="70.0" class="fill-soft-c"/>
<rect x="405.0" y="92.0" width="160.0" height="70.0" class="stroke-c"/>
<text x="485.0" y="114.0" class="fig-bold" text-anchor="middle">inverse STFT</text>
<text x="485.0" y="134.0" class="fig-small" text-anchor="middle">a real signal x</text>
<rect x="405.0" y="196.0" width="160.0" height="56.0" class="fill-soft-c"/>
<rect x="405.0" y="196.0" width="160.0" height="56.0" class="stroke-c"/>
<text x="485.0" y="218.0" class="fig-bold" text-anchor="middle">STFT of x</text>
<text x="485.0" y="238.0" class="fig-small" text-anchor="middle">X_(i+1)</text>
<line x1="170.0" y1="72.0" x2="223.0" y2="112.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="170.0" y1="182.0" x2="223.0" y2="142.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="355.0" y1="127.0" x2="403.0" y2="127.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<line x1="485.0" y1="162.0" x2="485.0" y2="194.0" class="stroke-ink" marker-end="url(#c008-arrow)"/>
<polyline points="405.0,224.0 95.0,224.0 95.0,216.0" class="stroke-ink" fill="none" marker-end="url(#c008-arrow)"/>
<text x="250.0" y="244.0" class="fig-small" text-anchor="middle">keep only its phase, then repeat</text>
<text x="20.0" y="24.0" class="fig-bold" text-anchor="start">One Griffin-Lim iteration</text>
</svg>
Figure 1. One Griffin-Lim iteration. Notice that the magnitude S never changes: only the phase is carried from one iteration to the next.
```

Read it aloud: keep the phase of the current STFT $X_i$, written $\angle X_i$, and put the target magnitude $S$ under it. The symbol $\odot$ means cell by cell. Invert that to a signal $x_{i+1}$ with the least-squares inverse, then take the signal's STFT. The inverse finds the real signal closest to the array you gave it, so its STFT is the nearest consistent array. Start from $X_0 = S e^{j\phi_0}$ with random phases $\phi_0$.

```python
c = S * np.exp(2j * np.pi * rng.random(S.shape))
for _ in range(n_iter):
    c = stft(istft(S * np.exp(1j * np.angle(c)), hop), n_fft, hop)
x = istft(S * np.exp(1j * np.angle(c)), hop)
```

Griffin and Lim proved in 1984 that with the least-squares inverse, the distance between the two sets never grows from one iteration to the next. That distance is the numerator of the spectral convergence, so the SC never rises. The following table and Figure 2 show it falling on the 0.512 s chirp.

| Iterations | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| Griffin-Lim | 0.647 | 0.368 | 0.314 | 0.272 | 0.237 | 0.205 | 0.165 | 0.128 |
| Fast Griffin-Lim | 0.647 | 0.368 | 0.301 | 0.244 | 0.205 | 0.152 | 0.105 | 0.089 |

The first iteration does the most. After that the gains shrink, because the algorithm finds a phase that is locally good, not the original one. A result at 0.1 to 0.2 sounds clearly like the source, with a metallic color.

> [!KEY]
> Griffin-Lim keeps the phase of the nearest real signal's STFT and puts the target magnitude back under it, again and again. With the least-squares inverse, the spectral convergence never gets worse.

In [Flow matching for speech](/lessons/067-flow-matching-for-speech), a model generates mel spectrograms of spoken digits, and this chapter's `griffin_lim` turns them into sound. Its signature is `griffin_lim(magnitude, n_fft, hop_length, n_iter=32, momentum=0.0, seed=0)`, and it returns a 1-D signal of `hop_length * (frames - 1)` samples.

## Fast Griffin-Lim

Plain Griffin-Lim takes small steps once it is close. **Fast Griffin-Lim**, from Perraudin, Balazs, and Søndergaard in 2013, adds momentum: it keeps moving in the direction the last step went. Let $t_i$ be the consistent STFT from iteration $i$. Instead of using $t_i$ directly, it extrapolates:

$$
c_i = t_i + \alpha \,(t_i - t_{i-1})
$$

Read it aloud: take the current projection, and add $\alpha$ times the change since the previous one. With $\alpha = 0$ this is plain Griffin-Lim. The next iteration keeps the phase of $c_i$. On the first iteration there is no $t_{i-1}$, so $c_1 = t_1$.

```python
t = stft(istft(S * np.exp(1j * np.angle(c)), hop), n_fft, hop)
c = t if previous is None else t + momentum * (t - previous)
previous = t
```

Work one cell by hand. Say the previous projection was $1$ and the current one is $j$, so the phase turned from 0° to 90°. With $\alpha = 0.99$, $c = j + 0.99(j - 1) = -0.99 + 1.99j$, whose angle is 116.4°. The momentum pushes the phase past where the projection stopped, in the direction it was already moving.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Spectral convergence against iterations for Griffin-Lim and fast Griffin-Lim on a 0.512 s chirp">
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0.0</text>
<line x1="70.0" y1="198.6" x2="560.0" y2="198.6" class="stroke-grid"/>
<text x="62.0" y="202.6" class="fig-small" text-anchor="end">0.2</text>
<line x1="70.0" y1="147.1" x2="560.0" y2="147.1" class="stroke-grid"/>
<text x="62.0" y="151.1" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="95.7" x2="560.0" y2="95.7" class="stroke-grid"/>
<text x="62.0" y="99.7" class="fig-small" text-anchor="end">0.6</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="140.0" y1="250.0" x2="140.0" y2="255.0" class="stroke-muted"/>
<text x="140.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="210.0" y1="250.0" x2="210.0" y2="255.0" class="stroke-muted"/>
<text x="210.0" y="270.0" class="fig-small" text-anchor="middle">2</text>
<line x1="280.0" y1="250.0" x2="280.0" y2="255.0" class="stroke-muted"/>
<text x="280.0" y="270.0" class="fig-small" text-anchor="middle">4</text>
<line x1="350.0" y1="250.0" x2="350.0" y2="255.0" class="stroke-muted"/>
<text x="350.0" y="270.0" class="fig-small" text-anchor="middle">8</text>
<line x1="420.0" y1="250.0" x2="420.0" y2="255.0" class="stroke-muted"/>
<text x="420.0" y="270.0" class="fig-small" text-anchor="middle">16</text>
<line x1="490.0" y1="250.0" x2="490.0" y2="255.0" class="stroke-muted"/>
<text x="490.0" y="270.0" class="fig-small" text-anchor="middle">32</text>
<line x1="560.0" y1="250.0" x2="560.0" y2="255.0" class="stroke-muted"/>
<text x="560.0" y="270.0" class="fig-small" text-anchor="middle">64</text>
<line x1="70.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="70.0" class="stroke-muted"/>
<text x="315.0" y="290.0" text-anchor="middle">Iterations</text>
<text x="30.0" y="56.0" text-anchor="start">Spectral convergence (lower is better)</text>
<polyline points="70.0,83.6 140.0,155.4 210.0,169.3 280.0,180.1 350.0,189.1 420.0,197.3 490.0,207.6 560.0,217.1" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="83.6" r="3.5" class="fill-b"/>
<circle cx="140.0" cy="155.4" r="3.5" class="fill-b"/>
<circle cx="210.0" cy="169.3" r="3.5" class="fill-b"/>
<circle cx="280.0" cy="180.1" r="3.5" class="fill-b"/>
<circle cx="350.0" cy="189.1" r="3.5" class="fill-b"/>
<circle cx="420.0" cy="197.3" r="3.5" class="fill-b"/>
<circle cx="490.0" cy="207.6" r="3.5" class="fill-b"/>
<circle cx="560.0" cy="217.1" r="3.5" class="fill-b"/>
<polyline points="70.0,83.6 140.0,155.4 210.0,172.6 280.0,187.3 350.0,197.3 420.0,210.9 490.0,223.0 560.0,227.1" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="83.6" r="3.5" class="fill-a"/>
<circle cx="140.0" cy="155.4" r="3.5" class="fill-a"/>
<circle cx="210.0" cy="172.6" r="3.5" class="fill-a"/>
<circle cx="280.0" cy="187.3" r="3.5" class="fill-a"/>
<circle cx="350.0" cy="197.3" r="3.5" class="fill-a"/>
<circle cx="420.0" cy="210.9" r="3.5" class="fill-a"/>
<circle cx="490.0" cy="223.0" r="3.5" class="fill-a"/>
<circle cx="560.0" cy="227.1" r="3.5" class="fill-a"/>
<text x="78.0" y="75.6" class="fig-small" text-anchor="start">0.647: random phase</text>
<text x="564.0" y="209.1" class="fig-small" text-anchor="end">0.128</text>
<text x="564.0" y="245.1" class="fig-small" text-anchor="end">0.089</text>
<line x1="300.0" y1="16.0" x2="325.0" y2="16.0" class="stroke-b"/>
<text x="331.0" y="20.0" class="fig-small" text-anchor="start">Griffin-Lim</text>
<line x1="300.0" y1="36.0" x2="325.0" y2="36.0" class="stroke-a"/>
<text x="331.0" y="40.0" class="fig-small" text-anchor="start">fast Griffin-Lim, momentum 0.99</text>
</svg>
Figure 2. Spectral convergence of plain and fast Griffin-Lim on a 0.512 s chirp. Notice that both curves fall fastest in the first iterations, and that momentum at 32 iterations (0.105) is already below plain Griffin-Lim at 64 (0.128).
```

librosa and torchaudio both default to 32 iterations with a momentum of 0.99. The price of momentum is the guarantee: the SC can rise from one iteration to the next. The following demo runs Griffin-Lim one iteration at a time.

```demo
{"id": "griffin-lim-iterations", "caption": "Run Griffin-Lim, then drag the iteration slider and play each step to hear the roughness of the random start fade as the spectral convergence falls, with plain Griffin-Lim dashed beside the momentum curve."}
```

## Stretch time with a phase vocoder

To slow speech down, you could play the samples at a lower rate, but that lowers every frequency too, and the voice drops in pitch. You want the timing to change and the frequencies to stay. A **phase vocoder** does that. It reads the STFT at a different speed, for example one analysis frame for every two output frames to double the length, and synthesizes with the original hop.

The magnitudes are easy: interpolate between the two neighboring analysis frames. The phases are the hard part. Copying a frame's phase twice would make the sinusoid jump back at the seam. Instead, each bin's phase must keep advancing at that bin's own frequency across the output frames, and the bin can measure its frequency from how far its phase moved between analysis frames.

A sinusoid at exactly bin $k$'s center turns by $\omega_k H$ radians in one hop, where $\omega_k = 2\pi k / N$. Any extra turn means the true frequency is a little off the center:

$$
\begin{aligned}
\Delta\phi &= \angle X[k, m+1] - \angle X[k, m] - \omega_k H \\
\psi &\leftarrow \psi + \omega_k H + \operatorname{wrap}(\Delta\phi)
\end{aligned}
$$

Read it aloud: the deviation $\Delta\phi$ is the measured phase change minus the change the bin's center frequency would give. Wrap it to $[-\pi, \pi]$, because phases only repeat every $2\pi$. The output phase $\psi$ advances by the expected turn plus the wrapped deviation for every output frame, however many output frames one analysis step becomes.

### Work one bin by hand

Take a 440 Hz tone at 16 kHz, with $N = 1024$ and $H = 256$. The nearest bin is $k = 28$, centered at $28 \times 16000 / 1024 = 437.5$ Hz. Its expected turn per hop is $2\pi \times 28 \times 256 / 1024 = 14\pi = 43.982$ radians. The tone really turns $2\pi \times 440 \times 256 / 16000 = 44.234$ radians, so the measured phase difference, wrapped, is 0.2513 radians more than expected. That extra turn gives the bin's frequency, $(43.982 + 0.2513) / (2\pi \times 256) \times 16000 = 440.00$ Hz. The output phase advances by 44.234 radians per output frame, whatever the stretch, so the pitch stays at 440 Hz.

```figure
<svg viewBox="0 0 700 330" role="img" aria-label="One phase vocoder bin: the phase turns by 14 pi radians in a hop, plus 0.2513 radians that reveal a frequency of 440 hertz"><defs><marker id="c008-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<circle cx="150" cy="150" r="100" class="stroke-grid" fill="none"/>
<line x1="40.0" y1="150.0" x2="260.0" y2="150.0" class="stroke-grid"/>
<line x1="150.0" y1="40.0" x2="150.0" y2="260.0" class="stroke-grid"/>
<line x1="150.0" y1="150.0" x2="250.0" y2="150.0" class="stroke-a dashed" marker-end="url(#c008-arrow2)"/>
<line x1="150.0" y1="150.0" x2="246.9" y2="125.1" class="stroke-b" marker-end="url(#c008-arrow2)"/>
<path d="M210.0 150.0 A60 60 0 0 0 208.1 135.1" class="stroke-d" fill="none"/>
<text x="216.0" y="144.0" class="fig-small" text-anchor="start">Δφ</text>
<text x="266.0" y="124.0" class="fig-small" text-anchor="start">frame m + 1: measured</text>
<text x="266.0" y="170.0" class="fig-small" text-anchor="start">frame m: expected</text>
<text x="150.0" y="278.0" class="fig-small" text-anchor="middle">Bin 28, frames m (dashed) and m + 1, as phasors</text>
<text x="420.0" y="70.0" text-anchor="start">Expected turn per hop: 14π = 43.982 rad</text>
<text x="420.0" y="98.0" text-anchor="start">Actual turn per hop: 44.234 rad</text>
<text x="420.0" y="126.0" text-anchor="start">Difference, already inside (-π, π]:</text>
<text x="420.0" y="150.0" class="fig-bold" text-anchor="start">Δφ = 0.2513 rad</text>
<text x="420.0" y="182.0" class="fig-small" text-anchor="start">Frequency = (43.982 + 0.2513)</text>
<text x="420.0" y="200.0" class="fig-small" text-anchor="start">÷ (2π × 256) × 16000 = 440.00 Hz</text>
</svg>
Figure 3. The phase of bin 28 in two frames. Notice that the expected turn of 14π lands back at angle 0, so the whole deviation shows as a small angle of 0.2513 rad, and that angle reveals the 440 Hz tone.
```

> [!RECAP]
> - The spectral convergence is `np.linalg.norm(np.abs(X) - S) / np.linalg.norm(S)`.
> - Griffin-Lim starts from `S * np.exp(2j * np.pi * rng.random(S.shape))` and repeats `stft(istft(S * np.exp(1j * np.angle(c))))`.
> - Fast Griffin-Lim replaces each projection $t$ with $t + \alpha(t - t_{\text{previous}})$.
> - The phase vocoder reads frames at steps of `1 / stretch`, interpolates magnitudes, and advances each bin's phase by $\omega_k H$ plus its wrapped deviation.

## Build it step by step

The functions `hann`, `stft`, and `istft` are provided. The checks test yours in this order.

1. **`spectral_convergence(X, magnitude)`** returns a float. Use `np.abs` and `np.linalg.norm`.
2. **`random_phase(magnitude, seed)`** draws `rng.random(magnitude.shape)` from `np.random.default_rng(seed)` and returns the complex array.
3. **`griffin_lim(magnitude, n_fft, hop_length, n_iter, momentum, seed)`** starts from `random_phase`, loops `n_iter` times, and inverts at the end. Keep the previous projection in a variable that starts as `None`. A common mistake is applying momentum to the phase-only array `np.exp(1j * np.angle(c))` instead of to the projections themselves.
4. **`phase_vocoder(X, stretch, hop_length)`** appends a zero column, walks the positions `np.arange(0, frames, 1 / stretch)`, and keeps one running phase per bin. A common mistake is wrapping the running phase itself; wrap only the deviation.
5. **`time_stretch(x, stretch, n_fft, hop_length)`** chains `stft`, `phase_vocoder`, and `istft`. A common mistake is inverting with a hop of `stretch * hop_length`. The phase vocoder already made more frames at the original hop, so that stretches the signal twice.

## Where this breaks

**It needs a magnitude, not a power.** Feed Griffin-Lim $|X|^2$ and it builds a signal whose magnitude is $|X|^2$, so every loud component gets louder and the result sounds harsh. Take the square root first. A mel spectrogram must also be mapped back to linear bins before Griffin-Lim can use it.

**It recovers a signal, not the signal.** A signal and its negative have the same magnitudes. Compare Griffin-Lim results by spectral convergence, never sample by sample, and keep the window, `n_fft`, and hop the same as the ones that made the magnitude.

**The phase vocoder smears transients.** Every bin advances on its own, so the phases of neighboring bins drift apart, and clicks and consonants turn soft and phasey. Phase-locked vocoders, from Laroche and Dolson in 1999, tie each bin's phase to the nearest spectral peak to fix this.

## The exercise

You build `spectral_convergence`, `random_phase`, `griffin_lim`, `phase_vocoder`, and `time_stretch`. Press **Run** to hear a buzzy vowel-like tone rebuilt from its magnitude four ways: zero phase, random phase, Griffin-Lim, and fast Griffin-Lim. Then see the spectral convergence of both algorithms against the iteration count, and hear the tone stretched to 1.5 times and 0.75 times its length at the same pitch.

The checks confirm the spectral convergence of an exact and a scaled STFT, that random phases keep the magnitude, and that `griffin_lim` returns the right length and matches the reference update with and without momentum. They confirm that plain Griffin-Lim never gets worse over 12 iterations, reaches an SC below 0.2 in 50, and that momentum 0.99 ends below 75% of plain at 32. The last checks confirm that `phase_vocoder` returns its input at stretch 1, and that a 440 Hz tone stretched by 1.5 and 0.75 keeps its length within a hop and its pitch within a bin.
