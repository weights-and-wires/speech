---
title: The short-time Fourier transform
slug: 04-the-short-time-fourier-transform
part: "Part 1 — Sound and signals"
summary: Cut a signal into centered, windowed frames, take the FFT of each, and invert the result exactly with a least-squares overlap-add.
minutes: 70
gpu: false
objectives:
  - Count and cut the frames of a centered, reflect-padded STFT, and state its layout of bins by frames.
  - Build the STFT and match torch.stft to within rounding.
  - "Explain the trade-off between time and frequency resolution: a longer window sees finer frequency and coarser time."
  - Test the COLA and NOLA conditions, and invert an STFT by least squares to 1e-10.
prereqs:
  - 03-the-dft-and-the-fft
exercise: 04-the-short-time-fourier-transform
---

# The short-time Fourier transform

One FFT of a whole utterance tells you which frequencies it contains, but not
when. Speech changes every few tens of milliseconds: a vowel gives way to a
hiss, a pitch rises, a click comes and goes. To follow that, you cut the
signal into short overlapping frames and take one FFT per frame. That is the
**short-time Fourier transform** (STFT), and its magnitude, laid out with
time across and frequency up, is the **spectrogram**.

A spectrogram is sheet music for sound. Every recognizer, vocoder, and codec
later in the course reads or writes one, so this lesson settles the
conventions they all use: how frames are counted and padded, which window,
which way round the matrix is, and when to square it. Then you run the
transform backward and recover the signal to ten decimal places.

## What you already know

- Bin $k$ of an $N$-point FFT is $k f_s / N$ hertz, and `np.fft.rfft` returns bins 0 to $N/2$.
- A window fades each frame in and out to reduce leakage, at the cost of a wider main lobe.
- A periodic Hann window is $0.5 - 0.5 \cos(2\pi n / N)$, and it suits overlapping frames.
- A tone that fits the frame lands in one bin; zero-padding refines the grid but not the resolution.
- numpy broadcasting builds a 2-D index array from a row and a column.

## Frames, hops, and counts

A **frame** is a slice of $N$ consecutive samples, where $N$ is the frame
length, called `n_fft` when it equals the FFT length. The **hop** $H$, called
`hop_length`, is the number of samples from the start of one frame to the
start of the next. Frames overlap when $H < N$. Whisper uses $N = 400$ and
$H = 160$ at 16 kHz: 25 ms frames every 10 ms.

Without padding, frame $m$ covers samples $mH$ to $mH + N - 1$, and only whole
frames fit. For a signal of $L$ samples, the count is the following:

$$
\begin{aligned}
M_{\text{uncentered}} &= 1 + \left\lfloor \frac{L - N}{H} \right\rfloor \\
M_{\text{centered}} &= 1 + \left\lfloor \frac{L}{H} \right\rfloor
\end{aligned}
$$

Read it aloud: without centering, one frame fits at the start, and one more
for every whole hop in the samples left over. With centering, the signal is
padded by $N/2$ samples at each end, so frame $m$ is **centered** on sample
$mH$, and there is one frame per whole hop plus one. $\lfloor \cdot \rfloor$
means round down, `//` in Python.

One second at 16 kHz with Whisper's settings gives $1 + \lfloor 15600/160 \rfloor = 98$
frames uncentered and $1 + \lfloor 16000/160 \rfloor = 101$ centered. Whisper's
30-second window holds 480,000 samples, which gives 3001 centered frames, and
Whisper drops the last one to get exactly 3000.

### Pad by reflection

Centering needs samples before the start and after the end. This course pads
by **reflection**, the default of `torch.stft` and `torchaudio`: the signal is
mirrored about its first and last sample, without repeating them. The
alternative, zeros, is librosa's default, and it makes the first and last
frames quieter.

Take the signal $0, 1, \dots, 9$, with $N = 8$ and $H = 4$. Centering pads 4
samples at each end: $4, 3, 2, 1$ before and $8, 7, 6, 5$ after. There are
$1 + \lfloor 10/4 \rfloor = 3$ frames:

- Frame 0 is $4, 3, 2, 1, 0, 1, 2, 3$, centered on sample 0.
- Frame 1 is $0, 1, 2, 3, 4, 5, 6, 7$, centered on sample 4.
- Frame 2 is $4, 5, 6, 7, 8, 9, 8, 7$, centered on sample 8.

Uncentered, only $1 + \lfloor 2/4 \rfloor = 1$ frame fits, samples 0 to 7, and
samples 8 and 9 are never analyzed.

```python
x = np.pad(x, n_fft // 2, mode="reflect")
index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
frames = x[index]        # (n_frames, n_fft)
```

## The STFT

The STFT windows every frame and takes its DFT:

$$
X[k, m] = \sum_{n=0}^{N-1} x[n + mH] \, w[n] \, e^{-2\pi j k n / N}
$$

Read it aloud: entry $(k, m)$ is bin $k$ of frame $m$: the samples of frame
$m$, multiplied by the window $w$, compared with a sinusoid at bin $k$. Here
$x$ is the padded signal. $X$ has $N/2 + 1$ rows, one per bin, and $M$ columns,
one per frame. With $N = 400$, that is 201 rows.

```python
X = np.fft.rfft(frames * window, axis=1).T      # (n_fft // 2 + 1, n_frames)
```

The **magnitude spectrogram** is $|X|$ and the **power spectrogram** is
$|X|^2$. Whisper and most mel front ends use power. In decibels the two agree
if you use the right rule: $20 \log_{10} |X|$ and $10 \log_{10} |X|^2$ are the
same number.

> [!KEY]
> The conventions of this course: frames are centered with reflect padding,
> the window is a periodic Hann, and a spectrogram is shaped (bins, frames)
> with row 0 at 0 Hz. Column $m$ is centered at $mH / f_s$ seconds.

That layout is what `torch.stft` returns and what `speechlab.show_spectrogram`
draws, with row 0 at the bottom. `np.fft.rfft(frames, axis=1)` gives frames
by bins, so transpose it. Pass `power=True` to `show_spectrogram` when you
hand it a power spectrogram, so it takes $10 \log_{10}$ instead of $20$.

## Time against frequency

The frame length sets both resolutions. The bins are $f_s / N$ hertz apart,
and each frame spans $N / f_s$ seconds. At 16 kHz, a 128-point frame spans
8 ms with bins 125 Hz apart. A 2048-point frame spans 128 ms with bins
7.8 Hz apart. You can't have both: a long frame separates close frequencies
but smears events that are close in time, and a short frame does the
opposite. This is a property of signals, not of the FFT. The spread of a
signal in time and in frequency can't both be small.

Speech analysis uses both kinds. A **wideband** spectrogram, about 5 ms per
frame, shows each glottal pulse as a vertical line and the broad resonances
of the vocal tract. A **narrowband** spectrogram, 30 to 50 ms, resolves the
individual harmonics of the voice as horizontal lines. Recognizers sit in
between, at 25 ms.

The hop sets how often you look, not how sharply. A hop of $N/4$ is common,
and Whisper's 160 of 400 is about $0.4N$. The following demo holds a sweep,
two tones 80 Hz apart, and two clicks 20 ms apart, so no single window length
shows everything sharply.

```demo
{"id": "stft-explorer", "caption": "Lengthen the window until the tones at 2000 and 2080 Hz separate, then notice that the two clicks have merged into one smear."}
```

> [!KEY]
> A frame of $N$ samples gives bins $f_s / N$ apart and blurs time over
> $N / f_s$ seconds. Pick $N$ for the detail you need, then pick the hop.

## Invert the STFT

Going back takes three steps: invert each frame's FFT, put each frame back
where it came from, and add the overlaps. That is **overlap-add**. The frames
were windowed, though, so the overlaps carry the window's shape. Dividing by
the right total removes it.

The **least-squares inverse**, due to Griffin and Lim, windows each inverted
frame a second time and divides by the sum of the squared windows:

$$
\hat{x}[n] = \frac{\sum_m w[n - mH] \, y_m[n - mH]}{\sum_m w^2[n - mH]}
$$

Read it aloud: $y_m$ is the inverse FFT of column $m$. At each sample $n$,
add up every frame that covers it, each weighted by the window value at that
position. Then divide by the sum of the squared window values at the same
positions. When $X$ really is the STFT of a signal, this returns the signal
exactly. When $X$ has been modified, as in the next lesson, it returns the
signal whose STFT is closest to $X$ in the least-squares sense.

```python
frames = np.fft.irfft(X.T, n=n_fft, axis=1) * window
for m in range(n_frames):
    out[m * hop : m * hop + n_fft] += frames[m]
    weight[m * hop : m * hop + n_fft] += window ** 2
x_hat = out[n_fft // 2 :][:length] / weight[n_fft // 2 :][:length]
```

The division needs the denominator to be positive everywhere. That is the
**NOLA** condition, for nonzero overlap-add. A periodic Hann window is 0 at its
first sample, so with a hop equal to $N$, every $N$-th sample has zero weight,
and nothing can recover it.

### COLA

A stronger property makes plain overlap-add work without any division. A
window is **COLA**, for constant overlap-add, at hop $H$ if its shifted copies
sum to the same value at every sample:

$$
\sum_m w[n - mH] = C \quad \text{for every } n
$$

Read it aloud: stack a copy of the window at every hop, add them up, and the
total must be flat. Sample $n$ is covered by the window values at positions
$n$, $n + H$, $n + 2H$, and so on, so it's enough to check that
`window[r::hop].sum()` is the same for every remainder $r$ from 0 to $H - 1$.

Work it with a periodic Hann of length 4, $[0, 0.5, 1, 0.5]$. At hop 2,
remainder 0 sums $0 + 1 = 1$ and remainder 1 sums $0.5 + 0.5 = 1$: COLA,
with $C = 1$. At hop 3, the sums are $0 + 0.5 = 0.5$, $0.5$, and $1$: not
COLA. The squared window $[0, 0.25, 1, 0.25]$ at hop 2 sums to 1 and 0.5, not
constant but never zero, so NOLA holds and the least-squares inverse still
works.

A periodic Hann of length 400 is COLA at hops 200 and 100, but not at
Whisper's 160, where the overlap-added windows ripple between 1.19 and 1.31.
The symmetric Hann is not exactly COLA at any of them. Whisper never
inverts its STFT, so it doesn't care, and your least-squares inverse doesn't
either, as long as NOLA holds.

> [!WARNING]
> Overlap-adding the windowed inverse frames without dividing by $\sum w^2$
> leaves a ripple at the hop rate: about 1.5 dB at Whisper's settings. You
> hear it as a faint buzz at $f_s / H$, 100 Hz for a hop of 160.

> [!RECAP]
> - Centered, the frame count is `1 + L // hop`; uncentered, it is `1 + (L - n_fft) // hop`.
> - Pad with `np.pad(x, n_fft // 2, mode="reflect")` and index every frame at once.
> - The STFT is `np.fft.rfft(frames * window, axis=1).T`, shaped (bins, frames); power is its absolute value squared.
> - COLA means `window[r::hop].sum()` is the same for every remainder `r`.
> - The least-squares inverse overlap-adds windowed inverse FFTs and divides by the overlap-added squared window.

## Build it step by step

The checks test the functions in this order.

1. **`frame_count(n_samples, frame_length, hop_length, center)`** returns an
   int from the two formulas. Common mistake: `L / H` instead of `L // H`,
   which returns a float and is off by a fraction.
2. **`frame_signal(x, frame_length, hop_length, center, pad_mode)`** pads when
   centering, builds the index array, and returns `(n_frames, frame_length)`.
   Common mistake: computing the frame count after padding with the centered
   formula, which counts the padding twice.
3. **`stft(x, n_fft, hop_length, window, center)`** frames, windows,
   transforms along `axis=1`, and transposes. Use `hann(n_fft)` when `window`
   is `None`. Common mistake: `np.hanning(n_fft)`, which is symmetric and
   fails the comparison with `torch.stft`.
4. **`spectrogram(x, n_fft, hop_length, power)`** returns
   `np.abs(stft(...)) ** power`. Common mistake: `np.real(X) ** 2`, which
   drops the imaginary part.
5. **`is_cola(window, hop_length)`** compares `window[r::hop_length].sum()`
   across remainders. Common mistake: checking only the middle of one
   overlap-added signal, where the edges fool the test.
6. **`istft(X, hop_length, window, center, length)`** recovers `n_fft` from
   `X.shape[0]`, overlap-adds the windowed `np.fft.irfft` frames and the
   squared window, trims `n_fft // 2` from the front, cuts to `length`, checks
   the weights, and divides. Common mistake: trimming `n_fft // 2` from the
   end too, which can leave fewer samples than `length` asks for.

## Cost and scaling

The following table gives the cost of each step for a signal of $L$
samples, frame length $N$, and hop $H$, so about $L / H$ frames.

| Step | Time | Memory |
|---|---|---|
| Framing | $O(L N / H)$ | $O(L N / H)$, each sample copied $N / H$ times |
| STFT | $O((L / H) \, N \log N)$ | $O((L / H) \, N)$ complex numbers |
| Least-squares inverse | $O((L / H) \, N \log N)$ | $O(L)$ for the output and the weights |

A minute of 16 kHz audio at $N = 400$ and $H = 160$ has 6001 frames of 201
bins: about 1.2 million complex numbers, 19 MB in `complex128`. The frame
array alone is 2.4 million floats. For long recordings, process blocks of
frames, or use `numpy.lib.stride_tricks.sliding_window_view`, which builds
the frames as a view without copying.

## Where this breaks

**Libraries pad differently.** `torch.stft` and torchaudio reflect; librosa
pads with zeros by default. The first and last frames differ, and every
comparison across libraries fails at the edges. Pass `pad_mode` explicitly.

**torch.stft defaults to no window.** Without `window=`, `torch.stft` uses a
rectangular window and leaks badly. Always pass
`torch.hann_window(n_fft)`, which is periodic.

**Uncentered frames lose the tail.** With `center=False`, the samples after
the last whole frame are never analyzed, and the inverse can't return them.
Pass `length=` to the inverse, and pad the input if you need the tail.

**Reflection needs enough signal.** A clip shorter than `n_fft // 2 + 1`
samples can't be reflect-padded by `torch.stft`, which raises an error. Pad
very short clips with zeros first.

**Scaling conventions differ.** `scipy.signal.stft` divides by the window's
sum by default, so its magnitudes are much smaller than those of
`np.fft.rfft`. Compare against `torch.stft` or your own code.

## The exercise

You build `frame_count`, `frame_signal`, `stft`, `spectrogram`, `is_cola`,
and `istft`. A periodic `hann` is provided. Press **Run** to see a sweep, two
close tones, and two clicks through an 8 ms window, where the clicks
separate, and a 128 ms window, where the tones do. Then hear the signal
rebuilt from its STFT, and see Hann windows overlap-added at a COLA hop and
at Whisper's.

The checks confirm the frame counts in both modes against `torch.stft`, the
uncentered and centered frames of the worked examples, and `stft` against
`torch.stft` at four settings. They check that an impulse lands in the frame
centered on it and a tone in its own row, that `spectrogram` squares only at
power 2, and that `is_cola` agrees with `scipy.signal.check_COLA`. The last
checks reconstruct a random signal to $10^{-10}$, match `torch.istft` on an
STFT with scrambled phase, which only the least-squares division gets right,
and expect a `ValueError` when NOLA fails.
