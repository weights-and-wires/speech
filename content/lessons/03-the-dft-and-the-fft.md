---
title: The DFT and the FFT
slug: 03-the-dft-and-the-fft
part: "Part 1 — Sound and signals"
summary: Build the DFT as a matrix and as a recursive FFT, read its bins in hertz, and measure how each window trades resolution for leakage.
minutes: 65
gpu: false
objectives:
  - Compute the DFT of a short signal by hand and as a matrix product.
  - Build a recursive radix-2 FFT and explain why it costs N log N instead of N squared.
  - Convert a bin index to hertz, and say which bins np.fft.rfft keeps and why.
  - "Explain spectral leakage, and measure each window's trade: a wider main lobe for lower side lobes."
prereqs:
  - 01-sound-and-sampling
exercise: 03-the-dft-and-the-fft
---

# The DFT and the FFT

A waveform tells you how the pressure moves, but not which pitches are
playing. To recognize a vowel, a speaker, or a word, you need the recipe: how
much of each frequency the sound holds. The **discrete Fourier transform**
(DFT) computes that recipe from a list of samples. Every spectrogram, mel
filterbank, and neural front end in this course starts with it.

Computed directly, the DFT of $N$ samples takes $N^2$ multiplications. The
**fast Fourier transform** (FFT) gets the same answer in about $N \log_2 N$,
which is the difference between 1 million and 10,000 operations for a
1024-sample frame. In this lesson you build both, learn to read the output in
hertz, and meet the problem every real spectrum has: energy that leaks from
one frequency into its neighbors, and the windows that control it.

## What you already know

- A signal sampled at $f_s$ can only represent frequencies up to $f_s / 2$.
- A tone is $A \cos(2\pi f t + \varphi)$, with amplitude, frequency, and phase.
- A complex number $a + bj$ has a magnitude $\sqrt{a^2 + b^2}$ and an angle, and $e^{j\theta} = \cos\theta + j \sin\theta$.
- A matrix-vector product takes the dot product of each row with the vector.
- Decibels of an amplitude ratio are $20 \log_{10}$ of the ratio.

## The DFT compares a signal with sinusoids

The DFT asks, for each of $N$ frequencies, how much the signal looks like a
sinusoid at that frequency. It measures that with a dot product: multiply the
signal by the sinusoid sample by sample, and add. A large result means a
strong match. Using a complex sinusoid, $e^{-j\theta} = \cos\theta - j\sin\theta$,
measures the cosine part and the sine part at once, so the answer carries
both the strength and the phase.

$$
X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-2\pi j k n / N}, \qquad k = 0, 1, \dots, N-1
$$

Read it aloud: output $k$ is the sum over all $N$ samples of the sample times
a complex sinusoid that completes $k$ turns across the frame. Each $X[k]$ is
one **bin**, and bin $k$ sits at frequency $k f_s / N$ hertz. Its magnitude
$|X[k]|$ says how much of that frequency is present, and its angle says the
phase.

Every output is a dot product with one row of a matrix, so the whole DFT is
one matrix-vector product with the **DFT matrix** $W[k, n] = e^{-2\pi j k n / N}$:

```python
k = np.arange(n)
W = np.exp(-2j * np.pi * np.outer(k, k) / n)
X = W @ x
```

The inverse puts the sinusoids back together: $x[n] = \frac{1}{N} \sum_k X[k] e^{2\pi j k n / N}$.
numpy follows this convention, with no $1/N$ on the forward transform and all
of it on the inverse.

### Work a four-point DFT by hand

Take $x = [1, 2, 0, -1]$. With $N = 4$, the sinusoid for bin $k$ turns by a
quarter circle, $e^{-j\pi/2} = -j$, per sample times $k$, so the twiddles are
1, $-j$, $-1$, and $j$.

- $X[0] = 1 + 2 + 0 + (-1) = 2$. Bin 0 is the sum, the **DC** component.
- $X[1] = 1 + 2(-j) + 0(-1) + (-1)(j) = 1 - 3j$.
- $X[2] = 1 - 2 + 0 - (-1) = 0$.
- $X[3] = 1 + 2(j) + 0(-1) + (-1)(-j) = 1 + 3j$.

Two properties show up already. $X[3]$ is the complex conjugate of $X[1]$.
And the energy is preserved: $\sum |x|^2 = 1 + 4 + 0 + 1 = 6$, and
$\frac{1}{4} \sum |X|^2 = \frac{1}{4}(4 + 10 + 0 + 10) = 6$. That second fact
is **Parseval's theorem**.

## Real signals need only half the bins

For a real signal, $X[N - k]$ is always the conjugate of $X[k]$. Bin $N - k$
is the negative frequency $-k f_s / N$, the mirror image of bin $k$, and it
carries no new information. So audio code keeps bins 0 through $N/2$ and
drops the rest. That is what `np.fft.rfft` returns: $N/2 + 1$ complex
numbers.

The frequencies of those bins are the following:

$$
f_k = \frac{k f_s}{N}, \qquad k = 0, 1, \dots, \frac{N}{2}
$$

Read it aloud: bin $k$ is $k$ times the sample rate divided by the frame
length. The **bin spacing** $f_s / N$ is the frequency step between bins, and
the last bin is the Nyquist frequency. `np.fft.rfftfreq(n, 1 / fs)` gives the
same numbers.

Whisper frames 16 kHz audio into 400-sample windows. Its spectrum has
$400/2 + 1 = 201$ bins, 40 Hz apart, from 0 Hz to 8000 Hz.

> [!KEY]
> Bin $k$ of an $N$-point DFT is $k f_s / N$ hertz. A real signal needs only
> bins 0 to $N/2$, which is what `np.fft.rfft` returns.

A cosine of amplitude $A$ that fits a whole number of cycles in the frame
lands in exactly one bin, with magnitude $A N / 2$. For $x = [1, 0, -1, 0]$,
a cosine at bin 1, the DFT is $[0, 2, 0, 2]$: half the energy in bin 1 and
half in its mirror, bin 3.

## The FFT splits the work in half, again and again

The **Cooley-Tukey** FFT rests on one observation. The DFT of $N$ samples
splits into the DFT of the $N/2$ even-indexed samples, $E$, and the DFT of
the $N/2$ odd-indexed samples, $O$:

$$
X[k] = E[k] + e^{-2\pi j k / N} O[k], \qquad
X[k + N/2] = E[k] - e^{-2\pi j k / N} O[k], \qquad k = 0, \dots, \tfrac{N}{2} - 1
$$

Read it aloud: bin $k$ is the even half's bin $k$ plus a **twiddle factor**
times the odd half's bin $k$. Bin $k + N/2$ uses the same two numbers with a
minus sign. So two half-size DFTs plus $N/2$ multiplications give the whole
thing. Apply the same split to each half, and to each quarter, until the
pieces have one sample, whose DFT is itself.

```python
even, odd = fft(x[0::2]), fft(x[1::2])
t = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
X = np.concatenate([even + t, even - t])
```

On $[1, 2, 0, -1]$, the even half $[1, 0]$ transforms to $E = [1, 1]$, and
the odd half $[2, -1]$ to $O = [1, 3]$. The twiddles for $N = 4$ are 1 and
$-j$, so $t = [1, -3j]$. The result is $[1 + 1, 1 - 3j, 1 - 1, 1 + 3j]$,
which is $[2, 1 - 3j, 0, 1 + 3j]$, as by hand.

There are $\log_2 N$ levels of splitting, and each level does about $N$ work,
so the FFT costs $O(N \log N)$. This split needs $N$ to be a power of two,
which is why frames of 256, 512, and 1024 samples are common. Libraries also
handle other lengths, such as Whisper's 400, with other factorizations.

## Leakage and windows

The DFT treats the frame as one period of a signal that repeats forever. If
the frame holds a whole number of cycles, the ends meet cleanly. If it
doesn't, the repeated signal has a jump where the end wraps to the start, and
that jump spreads energy into every bin. This is **spectral leakage**.

Take a 64-point frame. A cosine at exactly bin 10 gives 32 in bin 10 and 0
everywhere else. A cosine at bin 10.5 gives 20.1 in bin 10, 20.7 in bin 11,
and still $-26.2$ dB relative to the peak at bin 30, twenty bins away. A quiet
sound anywhere near a loud one disappears under that leakage.

A **window** fixes most of it. Multiply the frame by a curve that fades from
0 at the ends to 1 in the middle, so the ends meet cleanly whatever the
frequency. The cosine-sum windows share one formula:

$$
w[n] = a_0 - a_1 \cos\!\left(\frac{2\pi n}{P}\right) + a_2 \cos\!\left(\frac{4\pi n}{P}\right), \qquad n = 0, \dots, N-1
$$

Read it aloud: the window is a constant minus a cosine that turns once across
the window, plus a smaller cosine that turns twice. The **Hann** window uses
$a_0 = a_1 = 0.5$, the **Hamming** window uses 0.54 and 0.46, and the
**Blackman** window uses 0.42, 0.5, and 0.08. The **rectangular** window is
$a_0 = 1$, no window at all.

The period $P$ matters. A **symmetric** window uses $P = N - 1$, so its first
and last samples are equal. A **periodic** window uses $P = N$: it is the
symmetric window of length $N + 1$ with the last sample dropped. Periodic
windows tile cleanly when frames overlap, which is what the short-time
Fourier transform in the next lesson needs.

```python
phase = 2 * np.pi * np.arange(n) / n          # periodic; use n - 1 for symmetric
hann = 0.5 - 0.5 * np.cos(phase)
```

With a periodic Hann window, the bin-10.5 cosine above drops to $-92.7$ dB at
bin 30, instead of $-26.2$. The price shows at the peak: a cosine at exactly
bin 10 now spreads over three bins, as 8, 16, and 8.

> [!WARNING]
> `np.hanning(N)` is symmetric. `scipy.signal.get_window("hann", N)` and
> `torch.hann_window(N)` are periodic by default. The two differ by up to
> about $2.4/N$, enough to break an exact comparison with a library's STFT.

### Measure what each window trades

Every window's spectrum has a **main lobe**, the peak a pure tone spreads
into, and **side lobes**, the ripples on either side that carry leakage. To
see the shape, zero-pad the window by a factor of 64 and take its FFT. The
first null marks the edge of the main lobe, and the highest point beyond it
is the peak side lobe. The following table gives both for 64-point windows.

| Window | Main-lobe half-width | Peak side lobe |
|---|---|---|
| Rectangular | 1 bin | $-13.26$ dB |
| Hann | 2 bins | $-31.47$ dB |
| Hamming | 2 bins | $-42.45$ dB |
| Blackman | 3 bins | $-58.11$ dB |

A wider main lobe blurs two tones that are close together. Lower side lobes
let a quiet tone show next to a loud one.

> [!KEY]
> A window trades resolution for leakage. Going from rectangular to Blackman
> triples the main lobe and lowers the side lobes by 45 dB.

The following demo lets you slide a tone between bins and switch windows.

```demo
{"id": "window-leakage", "caption": "Move the tone halfway between two bins, then switch from rectangular to Hann and watch the far bins drop by tens of decibels."}
```

**Zero-padding** is not a cure for resolution. Padding a frame with zeros
before the FFT samples the same spectrum on a finer grid, but the main lobe
keeps its width. With a 25 ms Hann window, two tones 20 Hz apart stay one
peak however much you pad. Only a longer window separates them.

> [!RECAP]
> - The DFT matrix is `np.exp(-2j * np.pi * np.outer(k, k) / n)`, and the DFT is that matrix times the signal.
> - The FFT transforms the even and odd halves recursively and combines them with the twiddles $e^{-2\pi j k / N}$.
> - Bin $k$ is $k f_s / N$ hertz, and a real signal needs bins 0 to $N/2$.
> - Cosine-sum windows divide by $N$ when periodic and $N - 1$ when symmetric.
> - A window's leakage is its main-lobe half-width and peak side lobe, read from its zero-padded spectrum.

## Build it step by step

The checks test the functions in this order.

1. **`dft_matrix(n)`** returns the complex `(n, n)` matrix. Use `np.outer` and
   `np.exp`. Common mistake: a plus sign in the exponent, which gives the
   inverse transform's matrix.
2. **`dft(x)`** returns `dft_matrix(len(x)) @ x`.
3. **`fft_radix2(x)`** recurses on `x[0::2]` and `x[1::2]` and returns a
   complex array. Raise `ValueError` for a length that is not a power of two.
   Common mistake: stopping at length 2 with a special case that gets the
   sign of the twiddle wrong; stop at length 1 instead.
4. **`rfft_frequencies(n_fft, sample_rate)`** returns `n_fft // 2 + 1`
   frequencies. Common mistake: using `np.linspace(0, fs / 2, n_fft // 2)`,
   which is one bin short and slightly stretched.
5. **`get_window(name, n, periodic)`** looks up the coefficients in
   `COEFFICIENTS` and evaluates the cosine sum. Common mistake: dividing by
   `n - 1` for the periodic window.
6. **`leakage(window, oversample)`** zero-pads with `np.fft.rfft(window, len(window) * oversample)`,
   normalizes by the value at 0 Hz, walks to the first null, and returns the
   half-width and the peak side lobe in dB. Common mistake: taking the largest
   value after index 1, which is still on the main lobe.

## Cost and scaling

The FFT is the reason spectral analysis is cheap. The following table
compares the three ways to compute the DFT of $N$ samples.

| Method | Time | Memory | 1024 points on a laptop |
|---|---|---|---|
| Matrix DFT | $O(N^2)$ | $O(N^2)$ for the matrix | about 15 ms |
| Your recursive FFT | $O(N \log N)$ | $O(N \log N)$ over the recursion | about 3 ms |
| `np.fft.fft`, compiled | $O(N \log N)$ | $O(N)$ | about 0.02 ms |

Your FFT does the right amount of arithmetic, but it spends most of its time
on Python function calls and small arrays. Libraries work in place, loop in
compiled code, and handle any length, including primes. The matrix DFT at
$N = 16{,}384$ needs a 4 GB matrix; the FFT needs a few hundred kilobytes.

## Where this breaks

**Leakage hides quiet sounds.** In the **Run** block, a 1010 Hz tone sits
beside one 60 dB quieter at 2510 Hz. With no window, the leakage at 2 kHz is
$-39.5$ dB and buries the quiet tone. Use Hann or Blackman when levels differ
by more than about 30 dB.

**Hamming's side lobes fall slowly.** Hamming's nearest side lobes are lower
than Hann's, but measured on 64 points, its side lobes 20 bins out are still
at $-51$ dB, while Hann's have fallen to $-88$ dB. In the **Run** frame,
Hamming's leakage at 2 kHz is $-56.9$ dB and the quiet tone barely clears it.
Hann's is $-101$ dB.

**Magnitudes depend on the window.** A Hann window sums to $N/2$, so a tone of
amplitude $A$ reads about $AN/4$, not $AN/2$. Divide by `window.sum() / 2` to
read amplitudes, and use the same window everywhere you compare spectra.

**numpy doesn't scale the forward transform.** Some libraries divide by $N$ or
$\sqrt{N}$. `scipy.signal.stft` divides by the window's sum by default, so its
magnitudes differ from `np.fft.rfft` of the same frame.

**The full spectrum shows a mirror.** Plotting `np.abs(np.fft.fft(x))` over
all $N$ bins draws every tone twice. Use `rfft` for real signals.

## The exercise

You build `dft_matrix`, `dft`, `fft_radix2`, `rfft_frequencies`,
`get_window`, and `leakage`. Press **Run** to see the four-point example,
time the matrix DFT, your FFT, and numpy's on 1024 points, and see one 32 ms
frame of a loud tone and a quiet one through all four windows.

The checks confirm the twiddles of `dft_matrix(4)` and that $W W^H = N I$,
the worked example, and agreement with `np.fft.fft` for `dft` and for
`fft_radix2` at every power of two up to 1024. They test Parseval's theorem
and the conjugate symmetry of a real signal, and your bin frequencies against
`np.fft.rfftfreq`. They compare every window with `scipy.signal.get_window`,
periodic and symmetric, and your `leakage` with the table. The last check
shows that zero-padding a 25 ms window leaves 1000 Hz and 1020 Hz merged,
while a 200 ms window separates them.
