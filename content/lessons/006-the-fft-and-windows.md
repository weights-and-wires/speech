---
title: The FFT and window functions
slug: 006-the-fft-and-windows
part: "Part 1 — Sound and signals"
summary: Compute a spectrum in N log N steps by splitting the DFT in half, then fade each frame with a window so a tone doesn't leak.
minutes: 40
gpu: false
objectives:
  - Explain why the DFT of N samples splits into two DFTs of N/2 samples, and why that makes it cost N log N instead of N squared.
  - Build a recursive radix-2 FFT that matches np.fft.fft.
  - Explain spectral leakage, and why a tone between two bins smears into every bin.
  - "Build the Hann, Hamming, and Blackman windows, and measure each one's trade: a wider main lobe for lower side lobes."
prereqs:
  - 005-frequency-and-the-dft
exercise: 006-the-fft-and-windows
---

# The FFT and window functions

A speech recognizer looks at about 100 short frames of sound every second, and each frame needs a spectrum. The previous chapter gave you one way to compute it: multiply a matrix of sinusoids by the frame. That works, but it is slow. A 400-sample frame needs a 400 by 400 matrix, which is 160,000 complex multiplications. Whisper's 30-second window holds 3000 frames, so the matrix approach costs 480 million multiplications before the model sees anything.

The spectrum you get is also not quite honest. A tone that doesn't fit the frame a whole number of times smears into every bin, and a quiet sound next to a loud one disappears under the smear. This chapter fixes both problems. You will halve the work again and again until a 1024-point DFT costs about 10,000 operations instead of a million, and you will fade each frame in and out so that the smear drops by more than 60 dB.

## What you already know

- The DFT of $N$ samples is a matrix-vector product with the $N \times N$ matrix $W[k, n] = e^{-2\pi j k n / N}$, so it costs $N^2$ multiplications.
- Bin $k$ of an $N$-point DFT sits at $k f_s / N$ hertz, and `np.fft.rfft` keeps bins 0 to $N/2$ of a real signal.
- A tone that fits a whole number of cycles in the frame lands in exactly one bin.
- Multiplying by $e^{j\theta}$ turns a complex number by the angle $\theta$ and leaves its size alone.
- An amplitude ratio in decibels is $20 \log_{10}$ of the ratio.

## Why the matrix is too slow

The matrix DFT repeats work: every row of $W$ is made of powers of one number, so many products recur. Can you reuse them? Start with a small case. Take the four samples $x = [1, 2, 0, -1]$. Pull out the even-numbered samples $[1, 0]$ and the odd-numbered samples $[2, -1]$. Each half is a 2-point signal, and the DFT of two points $[a, b]$ is $[a + b, a - b]$. So the even half transforms to $E = [1, 1]$ and the odd half to $O = [1, 3]$. Figure 1 shows the plan: the 4-point answer should be buildable from $E$ and $O$ with a couple of extra multiplications.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The four-point DFT of [1, 2, 0, -1] computed by splitting into even and odd samples"><defs><marker id="c006-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="24.0" class="fig-bold" text-anchor="start">1. Split into even and odd samples</text>
<text x="250.0" y="24.0" class="fig-bold" text-anchor="start">2. Transform each half</text>
<text x="430.0" y="24.0" class="fig-bold" text-anchor="start">3. Combine</text>
<rect x="20.0" y="48.0" width="60.0" height="32.0" class="fill-soft-a"/>
<rect x="20.0" y="48.0" width="60.0" height="32.0" class="stroke-muted"/>
<text x="50.0" y="69.0" text-anchor="middle">x[0] = 1</text>
<rect x="20.0" y="90.0" width="60.0" height="32.0" class="fill-soft-d"/>
<rect x="20.0" y="90.0" width="60.0" height="32.0" class="stroke-muted"/>
<text x="50.0" y="111.0" text-anchor="middle">x[1] = 2</text>
<rect x="20.0" y="132.0" width="60.0" height="32.0" class="fill-soft-a"/>
<rect x="20.0" y="132.0" width="60.0" height="32.0" class="stroke-muted"/>
<text x="50.0" y="153.0" text-anchor="middle">x[2] = 0</text>
<rect x="20.0" y="174.0" width="60.0" height="32.0" class="fill-soft-d"/>
<rect x="20.0" y="174.0" width="60.0" height="32.0" class="stroke-muted"/>
<text x="50.0" y="195.0" text-anchor="middle">x[3] = -1</text>
<rect x="210.0" y="58.0" width="150.0" height="56.0" class="fill-soft-a"/>
<rect x="210.0" y="58.0" width="150.0" height="56.0" class="stroke-a"/>
<text x="285.0" y="80.0" text-anchor="middle">even: [1, 0]</text>
<text x="285.0" y="102.0" class="fig-bold" text-anchor="middle">E = [1, 1]</text>
<rect x="210.0" y="142.0" width="150.0" height="56.0" class="fill-soft-d"/>
<rect x="210.0" y="142.0" width="150.0" height="56.0" class="stroke-d"/>
<text x="285.0" y="164.0" text-anchor="middle">odd: [2, -1]</text>
<text x="285.0" y="186.0" class="fig-bold" text-anchor="middle">O = [1, 3]</text>
<line x1="80.0" y1="64.0" x2="208.0" y2="80.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<line x1="80.0" y1="148.0" x2="208.0" y2="88.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<line x1="80.0" y1="106.0" x2="208.0" y2="164.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<line x1="80.0" y1="190.0" x2="208.0" y2="176.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<rect x="420.0" y="58.0" width="160.0" height="140.0" class="fill-soft-b"/>
<rect x="420.0" y="58.0" width="160.0" height="140.0" class="stroke-b"/>
<text x="500.0" y="80.0" text-anchor="middle">t = twiddle × O</text>
<text x="500.0" y="100.0" text-anchor="middle">= [1, -3j]</text>
<text x="500.0" y="126.0" class="fig-bold" text-anchor="middle">E + t = [2, 1 - 3j]</text>
<text x="500.0" y="150.0" class="fig-bold" text-anchor="middle">E - t = [0, 1 + 3j]</text>
<text x="500.0" y="182.0" class="fig-small" text-anchor="middle">X = [2, 1-3j, 0, 1+3j]</text>
<line x1="360.0" y1="86.0" x2="418.0" y2="100.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<line x1="360.0" y1="170.0" x2="418.0" y2="150.0" class="stroke-ink" marker-end="url(#c006-arrow)"/>
<text x="20.0" y="236.0" class="fig-small" text-anchor="start">Two 2-point DFTs plus 2 multiplications replace one 4-point DFT.</text>
</svg>
Figure 1. The 4-point DFT of [1, 2, 0, -1] from two 2-point DFTs. Notice that bins 0 and 2 use the same two numbers, E[0] and t[0], once with a plus and once with a minus, and so do bins 1 and 3.
```

## Split the sum in two

To see why the plan works, write the DFT sum as two sums, one over the even-numbered samples $n = 2m$ and one over the odd-numbered samples $n = 2m + 1$:

$$
X[k] = \sum_{m=0}^{N/2-1} x[2m]\, e^{-2\pi j k m / (N/2)} \;+\; e^{-2\pi j k / N} \sum_{m=0}^{N/2-1} x[2m+1]\, e^{-2\pi j k m / (N/2)}
$$

Read it aloud: the first sum is the DFT of the even samples, which is an $N/2$-point DFT, and the second sum is the DFT of the odd samples. The odd samples sit one position later, so their sinusoid is shifted by one sample, and that shift pulls out the constant $e^{-2\pi j k / N}$ in front. Call that constant the **twiddle factor**, $\omega_k$. Write $E[k]$ and $O[k]$ for the two half-size DFTs. Then $X[k] = E[k] + \omega_k O[k]$.

The half-size DFTs repeat every $N/2$ bins, so $E[k + N/2] = E[k]$ and $O[k + N/2] = O[k]$. The twiddle does not repeat: moving $N/2$ bins turns it by half a circle, which flips its sign. That gives the second half of the output for free:

$$
\begin{aligned}
X[k] &= E[k] + \omega_k\, O[k] \\
X[k + N/2] &= E[k] - \omega_k\, O[k]
\end{aligned}
\qquad \omega_k = e^{-2\pi j k / N},\quad k = 0, \dots, \tfrac{N}{2} - 1
$$

Check it on the example. For $N = 4$ the twiddles are $\omega_0 = 1$ and $\omega_1 = e^{-j\pi/2} = -j$, so $\omega O = [1, -3j]$. Then $X[0] = 1 + 1 = 2$, $X[1] = 1 - 3j$, $X[2] = 1 - 1 = 0$, and $X[3] = 1 + 3j$, which is the four-point DFT from the previous chapter, reached with two multiplications instead of sixteen.

```python
even, odd = fft(x[0::2]), fft(x[1::2])
t = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
X = np.concatenate([even + t, even - t])
```

## Split again and again

The halves are DFTs too, so split each of them the same way, then each quarter, until the pieces hold one sample. The DFT of one sample is the sample itself, so the recursion stops there. This is the **Cooley-Tukey** algorithm, and with a split into two it is the **radix-2** FFT.

Count the work. There are $\log_2 N$ levels of splitting, and every level does about $N/2$ twiddle multiplications and $N$ additions. So the cost is proportional to $N \log_2 N$. For $N = 1024$ that is 10 levels, about 10,240 operations, against $1024^2 = 1{,}048{,}576$ for the matrix. The split needs an even length at every level, so $N$ must be a power of two, which is why frames of 256, 512, and 1024 are common.

> [!KEY]
> The FFT is not a different transform. It is the same DFT, computed by splitting the sum into even and odd samples again and again, which costs $N \log N$ instead of $N^2$.

## Why a frame leaks

Is the fast spectrum the one you wanted? Take a 64-sample frame of a cosine at exactly bin 10, which means it completes 10 cycles in the frame. The DFT puts 32 in bin 10 and 0 in every other bin. Move the cosine halfway to the next bin, 10.5 cycles per frame. The magnitudes in bins 9 to 12 become 6.5, 20.1, 20.7, and 7.1, and bin 30, twenty bins away, still sits at $-26.2$ dB below the peak (the orange curve in Figure 2).

The reason is in how the DFT reads its input. It treats the frame as one period of a signal that repeats forever. When the frame holds a whole number of cycles, the end meets the start without a seam. At 10.5 cycles the end of the frame is at the opposite phase from the start, so the repeated signal has a jump at every seam. A jump is a sudden change, and a sudden change contains every frequency. That energy is **spectral leakage**. Real speech never fits the frame, so leakage is the normal case, and a sound 40 dB quieter than its neighbor is lost under it.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Spectrum of a cosine halfway between bins 10 and 11, with no window and with a Hann window">
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">-100</text>
<line x1="70.0" y1="190.0" x2="570.0" y2="190.0" class="stroke-grid"/>
<text x="62.0" y="194.0" class="fig-small" text-anchor="end">-80</text>
<line x1="70.0" y1="150.0" x2="570.0" y2="150.0" class="stroke-grid"/>
<text x="62.0" y="154.0" class="fig-small" text-anchor="end">-60</text>
<line x1="70.0" y1="110.0" x2="570.0" y2="110.0" class="stroke-grid"/>
<text x="62.0" y="114.0" class="fig-small" text-anchor="end">-40</text>
<line x1="70.0" y1="70.0" x2="570.0" y2="70.0" class="stroke-grid"/>
<text x="62.0" y="74.0" class="fig-small" text-anchor="end">-20</text>
<line x1="70.0" y1="30.0" x2="570.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="148.1" y1="230.0" x2="148.1" y2="235.0" class="stroke-muted"/>
<text x="148.1" y="250.0" class="fig-small" text-anchor="middle">5</text>
<line x1="226.2" y1="230.0" x2="226.2" y2="235.0" class="stroke-muted"/>
<text x="226.2" y="250.0" class="fig-small" text-anchor="middle">10</text>
<line x1="304.4" y1="230.0" x2="304.4" y2="235.0" class="stroke-muted"/>
<text x="304.4" y="250.0" class="fig-small" text-anchor="middle">15</text>
<line x1="382.5" y1="230.0" x2="382.5" y2="235.0" class="stroke-muted"/>
<text x="382.5" y="250.0" class="fig-small" text-anchor="middle">20</text>
<line x1="460.6" y1="230.0" x2="460.6" y2="235.0" class="stroke-muted"/>
<text x="460.6" y="250.0" class="fig-small" text-anchor="middle">25</text>
<line x1="538.8" y1="230.0" x2="538.8" y2="235.0" class="stroke-muted"/>
<text x="538.8" y="250.0" class="fig-small" text-anchor="middle">30</text>
<line x1="70.0" y1="230.0" x2="570.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="30.0" class="stroke-muted"/>
<text x="320.0" y="270.0" text-anchor="middle">Bin k (64-point frame)</text>
<text x="30.0" y="16.0" text-anchor="start">dB below the peak</text>
<polyline points="70.0,82.6 85.6,82.3 101.2,81.2 116.9,79.5 132.5,77.2 148.1,74.2 163.8,70.4 179.4,65.7 195.0,59.4 210.6,50.1 226.2,30.5 241.9,30.0 257.5,48.6 273.1,56.9 288.8,62.2 304.4,66.1 320.0,69.0 335.6,71.4 351.2,73.3 366.9,75.0 382.5,76.3 398.1,77.5 413.8,78.5 429.4,79.4 445.0,80.1 460.6,80.7 476.2,81.3 491.9,81.7 507.5,82.0 523.1,82.3 538.8,82.5 554.4,82.6 570.0,82.6" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="82.6" r="2.5" class="fill-b"/>
<circle cx="85.6" cy="82.3" r="2.5" class="fill-b"/>
<circle cx="101.2" cy="81.2" r="2.5" class="fill-b"/>
<circle cx="116.9" cy="79.5" r="2.5" class="fill-b"/>
<circle cx="132.5" cy="77.2" r="2.5" class="fill-b"/>
<circle cx="148.1" cy="74.2" r="2.5" class="fill-b"/>
<circle cx="163.8" cy="70.4" r="2.5" class="fill-b"/>
<circle cx="179.4" cy="65.7" r="2.5" class="fill-b"/>
<circle cx="195.0" cy="59.4" r="2.5" class="fill-b"/>
<circle cx="210.6" cy="50.1" r="2.5" class="fill-b"/>
<circle cx="226.2" cy="30.5" r="2.5" class="fill-b"/>
<circle cx="241.9" cy="30.0" r="2.5" class="fill-b"/>
<circle cx="257.5" cy="48.6" r="2.5" class="fill-b"/>
<circle cx="273.1" cy="56.9" r="2.5" class="fill-b"/>
<circle cx="288.8" cy="62.2" r="2.5" class="fill-b"/>
<circle cx="304.4" cy="66.1" r="2.5" class="fill-b"/>
<circle cx="320.0" cy="69.0" r="2.5" class="fill-b"/>
<circle cx="335.6" cy="71.4" r="2.5" class="fill-b"/>
<circle cx="351.2" cy="73.3" r="2.5" class="fill-b"/>
<circle cx="366.9" cy="75.0" r="2.5" class="fill-b"/>
<circle cx="382.5" cy="76.3" r="2.5" class="fill-b"/>
<circle cx="398.1" cy="77.5" r="2.5" class="fill-b"/>
<circle cx="413.8" cy="78.5" r="2.5" class="fill-b"/>
<circle cx="429.4" cy="79.4" r="2.5" class="fill-b"/>
<circle cx="445.0" cy="80.1" r="2.5" class="fill-b"/>
<circle cx="460.6" cy="80.7" r="2.5" class="fill-b"/>
<circle cx="476.2" cy="81.3" r="2.5" class="fill-b"/>
<circle cx="491.9" cy="81.7" r="2.5" class="fill-b"/>
<circle cx="507.5" cy="82.0" r="2.5" class="fill-b"/>
<circle cx="523.1" cy="82.3" r="2.5" class="fill-b"/>
<circle cx="538.8" cy="82.5" r="2.5" class="fill-b"/>
<circle cx="554.4" cy="82.6" r="2.5" class="fill-b"/>
<circle cx="570.0" cy="82.6" r="2.5" class="fill-b"/>
<polyline points="70.0,230.0 85.6,178.5 101.2,164.8 116.9,154.9 132.5,145.8 148.1,136.1 163.8,124.9 179.4,111.0 195.0,91.8 210.6,58.0 226.2,30.0 241.9,30.0 257.5,58.0 273.1,91.7 288.8,110.8 304.4,124.5 320.0,135.2 335.6,144.1 351.2,151.6 366.9,158.2 382.5,164.1 398.1,169.5 413.8,174.4 429.4,179.0 445.0,183.5 460.6,187.9 476.2,192.2 491.9,196.8 507.5,201.8 523.1,207.7 538.8,215.3 554.4,227.7 570.0,230.0" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="230.0" r="2.5" class="fill-a"/>
<circle cx="85.6" cy="178.5" r="2.5" class="fill-a"/>
<circle cx="101.2" cy="164.8" r="2.5" class="fill-a"/>
<circle cx="116.9" cy="154.9" r="2.5" class="fill-a"/>
<circle cx="132.5" cy="145.8" r="2.5" class="fill-a"/>
<circle cx="148.1" cy="136.1" r="2.5" class="fill-a"/>
<circle cx="163.8" cy="124.9" r="2.5" class="fill-a"/>
<circle cx="179.4" cy="111.0" r="2.5" class="fill-a"/>
<circle cx="195.0" cy="91.8" r="2.5" class="fill-a"/>
<circle cx="210.6" cy="58.0" r="2.5" class="fill-a"/>
<circle cx="226.2" cy="30.0" r="2.5" class="fill-a"/>
<circle cx="241.9" cy="30.0" r="2.5" class="fill-a"/>
<circle cx="257.5" cy="58.0" r="2.5" class="fill-a"/>
<circle cx="273.1" cy="91.7" r="2.5" class="fill-a"/>
<circle cx="288.8" cy="110.8" r="2.5" class="fill-a"/>
<circle cx="304.4" cy="124.5" r="2.5" class="fill-a"/>
<circle cx="320.0" cy="135.2" r="2.5" class="fill-a"/>
<circle cx="335.6" cy="144.1" r="2.5" class="fill-a"/>
<circle cx="351.2" cy="151.6" r="2.5" class="fill-a"/>
<circle cx="366.9" cy="158.2" r="2.5" class="fill-a"/>
<circle cx="382.5" cy="164.1" r="2.5" class="fill-a"/>
<circle cx="398.1" cy="169.5" r="2.5" class="fill-a"/>
<circle cx="413.8" cy="174.4" r="2.5" class="fill-a"/>
<circle cx="429.4" cy="179.0" r="2.5" class="fill-a"/>
<circle cx="445.0" cy="183.5" r="2.5" class="fill-a"/>
<circle cx="460.6" cy="187.9" r="2.5" class="fill-a"/>
<circle cx="476.2" cy="192.2" r="2.5" class="fill-a"/>
<circle cx="491.9" cy="196.8" r="2.5" class="fill-a"/>
<circle cx="507.5" cy="201.8" r="2.5" class="fill-a"/>
<circle cx="523.1" cy="207.7" r="2.5" class="fill-a"/>
<circle cx="538.8" cy="215.3" r="2.5" class="fill-a"/>
<circle cx="554.4" cy="227.7" r="2.5" class="fill-a"/>
<circle cx="570.0" cy="230.0" r="2.5" class="fill-a"/>
<text x="538.8" y="70.5" class="fig-small" text-anchor="middle">bin 30: -26.2 dB</text>
<text x="538.8" y="170.0" class="fig-small" text-anchor="middle">bin 30: -92.7 dB</text>
<line x1="538.8" y1="176.0" x2="538.8" y2="211.3" class="stroke-muted"/>
<line x1="330.0" y1="46.0" x2="355.0" y2="46.0" class="stroke-b"/>
<text x="361.0" y="50.0" text-anchor="start">no window</text>
<line x1="450.0" y1="46.0" x2="475.0" y2="46.0" class="stroke-a"/>
<text x="481.0" y="50.0" text-anchor="start">Hann window</text>
</svg>
Figure 2. The 64-point spectrum of a cosine at 10.5 cycles per frame. Notice that the Hann window lowers bin 30 from -26.2 dB to -92.7 dB, and that its peak is wider.
```

## Fade the edges with a window

The leakage comes from the seam, so remove the seam. Multiply the frame by a curve that fades from 0 at the edges to 1 in the middle. Both ends then sit at zero, whatever the frequency, and they meet cleanly. The cosine-sum windows all use one formula:

$$
w[n] = a_0 - a_1 \cos\!\left(\frac{2\pi n}{P}\right) + a_2 \cos\!\left(\frac{4\pi n}{P}\right), \qquad n = 0, \dots, N-1
$$

Read it aloud: the window is a constant, minus a cosine that turns once across the window, plus a smaller cosine that turns twice. The **Hann** window uses $a_0 = a_1 = 0.5$, **Hamming** uses 0.54 and 0.46, and **Blackman** uses 0.42, 0.5, and 0.08. The **rectangular** window is $a_0 = 1$, which is no window at all.

The period $P$ has two choices. A **symmetric** window uses $P = N - 1$, so its first and last samples are equal. A **periodic** window uses $P = N$: it is the symmetric window of length $N + 1$ with its last sample dropped. Periodic windows tile cleanly when frames overlap, which the next chapter needs, so this course uses them.

```python
phase = 2 * np.pi * np.arange(n) / n          # periodic; use n - 1 for symmetric
hann = 0.5 - 0.5 * np.cos(phase)
```

With a periodic Hann window the 10.5-cycle cosine falls to $-92.7$ dB at bin 30, instead of $-26.2$ dB. Figure 2 shows the price: a cosine at exactly bin 10 then spreads over three bins, as 8, 16, and 8, and the peak at 10.5 is wider.

> [!WARNING]
> `np.hanning(N)` is symmetric. `scipy.signal.get_window("hann", N)` and `torch.hann_window(N)` are periodic by default. The two differ by up to about $2.4/N$, which is enough to break an exact comparison with a library's STFT.

### Measure what each window trades

Every window has a spectrum of its own. Its **main lobe** is the peak that a pure tone spreads into, and its **side lobes** are the ripples on either side, which carry the leakage. To see their shape, zero-pad the window by a factor of 64 and take its FFT. The first null marks the edge of the main lobe, and the highest point beyond it is the peak side lobe. Figure 3 draws the four spectra, and the following table gives the numbers for 64-point windows.

| Window | Main-lobe half-width | Peak side lobe |
|---|---|---|
| Rectangular | 1 bin | $-13.26$ dB |
| Hann | 2 bins | $-31.47$ dB |
| Hamming | 2 bins | $-42.45$ dB |
| Blackman | 3 bins | $-58.11$ dB |

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Spectra of four 64-point windows: the rectangular main lobe is narrowest and its side lobes highest">
<line x1="70.0" y1="230.0" x2="460.0" y2="230.0" class="stroke-grid"/>
<text x="62.0" y="234.0" class="fig-small" text-anchor="end">-80</text>
<line x1="70.0" y1="180.0" x2="460.0" y2="180.0" class="stroke-grid"/>
<text x="62.0" y="184.0" class="fig-small" text-anchor="end">-60</text>
<line x1="70.0" y1="130.0" x2="460.0" y2="130.0" class="stroke-grid"/>
<text x="62.0" y="134.0" class="fig-small" text-anchor="end">-40</text>
<line x1="70.0" y1="80.0" x2="460.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" class="fig-small" text-anchor="end">-20</text>
<line x1="70.0" y1="30.0" x2="460.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="230.0" x2="70.0" y2="235.0" class="stroke-muted"/>
<text x="70.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
<line x1="118.8" y1="230.0" x2="118.8" y2="235.0" class="stroke-muted"/>
<text x="118.8" y="250.0" class="fig-small" text-anchor="middle">1</text>
<line x1="167.5" y1="230.0" x2="167.5" y2="235.0" class="stroke-muted"/>
<text x="167.5" y="250.0" class="fig-small" text-anchor="middle">2</text>
<line x1="216.2" y1="230.0" x2="216.2" y2="235.0" class="stroke-muted"/>
<text x="216.2" y="250.0" class="fig-small" text-anchor="middle">3</text>
<line x1="265.0" y1="230.0" x2="265.0" y2="235.0" class="stroke-muted"/>
<text x="265.0" y="250.0" class="fig-small" text-anchor="middle">4</text>
<line x1="313.8" y1="230.0" x2="313.8" y2="235.0" class="stroke-muted"/>
<text x="313.8" y="250.0" class="fig-small" text-anchor="middle">5</text>
<line x1="362.5" y1="230.0" x2="362.5" y2="235.0" class="stroke-muted"/>
<text x="362.5" y="250.0" class="fig-small" text-anchor="middle">6</text>
<line x1="411.2" y1="230.0" x2="411.2" y2="235.0" class="stroke-muted"/>
<text x="411.2" y="250.0" class="fig-small" text-anchor="middle">7</text>
<line x1="460.0" y1="230.0" x2="460.0" y2="235.0" class="stroke-muted"/>
<text x="460.0" y="250.0" class="fig-small" text-anchor="middle">8</text>
<line x1="70.0" y1="230.0" x2="460.0" y2="230.0" class="stroke-muted"/>
<line x1="70.0" y1="230.0" x2="70.0" y2="30.0" class="stroke-muted"/>
<text x="265.0" y="270.0" text-anchor="middle">Distance from the peak (bins)</text>
<text x="30.0" y="16.0" text-anchor="start">dB below the peak</text>
<polyline points="70.0,30.0 71.5,30.0 73.0,30.1 74.6,30.3 76.1,30.6 77.6,30.9 79.1,31.3 80.7,31.7 82.2,32.3 83.7,32.9 85.2,33.6 86.8,34.4 88.3,35.3 89.8,36.3 91.3,37.3 92.9,38.5 94.4,39.8 95.9,41.2 97.4,42.8 98.9,44.5 100.5,46.4 102.0,48.4 103.5,50.7 105.0,53.3 106.6,56.1 108.1,59.4 109.6,63.1 111.1,67.5 112.7,72.8 114.2,79.6 115.7,88.9 117.2,104.6 118.8,230.0 120.3,106.0 121.8,91.7 123.3,83.7 124.8,78.3 126.4,74.3 127.9,71.3 129.4,69.0 130.9,67.2 132.5,65.8 134.0,64.8 135.5,64.0 137.0,63.5 138.6,63.2 140.1,63.1 141.6,63.3 143.1,63.6 144.6,64.2 146.2,64.9 147.7,65.9 149.2,67.1 150.7,68.5 152.3,70.2 153.8,72.2 155.3,74.5 156.8,77.2 158.4,80.5 159.9,84.4 161.4,89.3 162.9,95.7 164.5,104.7 166.0,120.0 167.5,230.0 169.0,120.6 170.5,106.0 172.1,97.7 173.6,92.0 175.1,87.8 176.6,84.6 178.2,82.0 179.7,79.9 181.2,78.3 182.7,77.0 184.3,76.0 185.8,75.3 187.3,74.8 188.8,74.6 190.4,74.5 191.9,74.7 193.4,75.1 194.9,75.7 196.4,76.5 198.0,77.5 199.5,78.7 201.0,80.3 202.5,82.1 204.1,84.3 205.6,86.9 207.1,90.0 208.6,93.8 210.2,98.6 211.7,104.8 213.2,113.7 214.7,128.8 216.2,230.0 217.8,129.3 219.3,114.6 220.8,106.2 222.3,100.4 223.9,96.1 225.4,92.7 226.9,90.0 228.4,87.9 230.0,86.2 231.5,84.8 233.0,83.7 234.5,82.9 236.1,82.3 237.6,82.0 239.1,81.9 240.6,82.0 242.1,82.2 243.7,82.8 245.2,83.5 246.7,84.4 248.2,85.6 249.8,87.1 251.3,88.8 252.8,91.0 254.3,93.5 255.9,96.6 257.4,100.3 258.9,105.0 260.4,111.2 262.0,120.0 263.5,135.1 265.0,230.0 266.5,135.4 268.0,120.6 269.6,112.2 271.1,106.3 272.6,102.0 274.1,98.6 275.7,95.8 277.2,93.6 278.7,91.9 280.2,90.4 281.8,89.3 283.3,88.5 284.8,87.8 286.3,87.5 287.9,87.3 289.4,87.3 290.9,87.6 292.4,88.1 293.9,88.7 295.5,89.6 297.0,90.8 298.5,92.2 300.0,93.9 301.6,96.0 303.1,98.5 304.6,101.5 306.1,105.2 307.7,109.9 309.2,116.0 310.7,124.8 312.2,139.9 313.8,230.0 315.3,140.2 316.8,125.3 318.3,116.8 319.8,111.0 321.4,106.6 322.9,103.1 324.4,100.4 325.9,98.2 327.5,96.3 329.0,94.9 330.5,93.7 332.0,92.8 333.6,92.2 335.1,91.8 336.6,91.6 338.1,91.6 339.6,91.8 341.2,92.3 342.7,92.9 344.2,93.8 345.7,94.9 347.3,96.3 348.8,98.0 350.3,100.1 351.8,102.5 353.4,105.5 354.9,109.2 356.4,113.9 357.9,120.0 359.5,128.7 361.0,143.8 362.5,230.0 364.0,144.0 365.5,129.2 367.1,120.6 368.6,114.7 370.1,110.3 371.6,106.9 373.2,104.1 374.7,101.8 376.2,100.0 377.7,98.5 379.3,97.4 380.8,96.4 382.3,95.8 383.8,95.4 385.4,95.1 386.9,95.1 388.4,95.3 389.9,95.8 391.4,96.4 393.0,97.3 394.5,98.4 396.0,99.7 397.5,101.4 399.1,103.4 400.6,105.9 402.1,108.9 403.6,112.5 405.2,117.2 406.7,123.3 408.2,132.0 409.7,147.0 411.2,230.0 412.8,147.2 414.3,132.4 415.8,123.8 417.3,117.9 418.9,113.5 420.4,110.0 421.9,107.2 423.4,104.9 425.0,103.1 426.5,101.6 428.0,100.4 429.5,99.5 431.1,98.8 432.6,98.4 434.1,98.1 435.6,98.1 437.1,98.3 438.7,98.7 440.2,99.3 441.7,100.2 443.2,101.3 444.8,102.6 446.3,104.3 447.8,106.3 449.3,108.8 450.9,111.7 452.4,115.4 453.9,120.0 455.4,126.1 457.0,134.8 458.5,149.8 460.0,230.0" class="stroke-b" fill="none"/>
<line x1="478.0" y1="44.0" x2="498.0" y2="44.0" class="stroke-b"/>
<text x="504.0" y="48.0" class="fig-small" text-anchor="start">rectangular</text>
<polyline points="70.0,30.0 71.5,30.0 73.0,30.1 74.6,30.1 76.1,30.2 77.6,30.3 79.1,30.5 80.7,30.7 82.2,30.9 83.7,31.1 85.2,31.4 86.8,31.7 88.3,32.0 89.8,32.3 91.3,32.7 92.9,33.1 94.4,33.6 95.9,34.0 97.4,34.5 98.9,35.1 100.5,35.6 102.0,36.2 103.5,36.8 105.0,37.5 106.6,38.2 108.1,38.9 109.6,39.7 111.1,40.5 112.7,41.3 114.2,42.2 115.7,43.1 117.2,44.1 118.8,45.1 120.3,46.1 121.8,47.2 123.3,48.3 124.8,49.5 126.4,50.7 127.9,52.0 129.4,53.3 130.9,54.7 132.5,56.2 134.0,57.7 135.5,59.3 137.0,61.0 138.6,62.7 140.1,64.6 141.6,66.5 143.1,68.5 144.6,70.6 146.2,72.9 147.7,75.3 149.2,77.9 150.7,80.6 152.3,83.6 153.8,86.8 155.3,90.3 156.8,94.1 158.4,98.5 159.9,103.5 161.4,109.4 162.9,116.8 164.5,126.7 166.0,142.9 167.5,230.0 169.0,145.4 170.5,131.7 172.1,124.2 173.6,119.4 175.1,116.0 176.6,113.5 178.2,111.7 179.7,110.4 181.2,109.5 182.7,109.0 184.3,108.7 185.8,108.7 187.3,108.9 188.8,109.3 190.4,109.9 191.9,110.8 193.4,111.8 194.9,113.0 196.4,114.4 198.0,116.0 199.5,117.9 201.0,120.0 202.5,122.4 204.1,125.2 205.6,128.4 207.1,132.1 208.6,136.4 210.2,141.7 211.7,148.5 213.2,157.9 214.7,173.6 216.2,230.0 217.8,175.0 219.3,160.8 220.8,152.9 222.3,147.6 223.9,143.8 225.4,140.9 226.9,138.7 228.4,137.0 230.0,135.7 231.5,134.8 233.0,134.2 234.5,133.8 236.1,133.7 237.6,133.8 239.1,134.1 240.6,134.6 242.1,135.3 243.7,136.3 245.2,137.4 246.7,138.8 248.2,140.4 249.8,142.2 251.3,144.4 252.8,146.9 254.3,149.8 255.9,153.3 257.4,157.4 258.9,162.5 260.4,169.0 262.0,178.2 263.5,193.7 265.0,230.0 266.5,194.7 268.0,180.3 269.6,172.2 271.1,166.7 272.6,162.7 274.1,159.6 275.7,157.3 277.2,155.4 278.7,154.0 280.2,152.9 281.8,152.1 283.3,151.6 284.8,151.3 286.3,151.2 287.9,151.4 289.4,151.7 290.9,152.3 292.4,153.1 293.9,154.1 295.5,155.3 297.0,156.8 298.5,158.5 300.0,160.5 301.6,162.9 303.1,165.7 304.6,169.0 306.1,173.0 307.7,178.0 309.2,184.4 310.7,193.5 312.2,208.8 313.8,230.0 315.3,209.7 316.8,195.1 318.3,186.9 319.8,181.3 321.4,177.2 322.9,174.0 324.4,171.6 325.9,169.6 327.5,168.1 329.0,166.9 330.5,166.0 332.0,165.4 333.6,165.0 335.1,164.8 336.6,164.9 338.1,165.2 339.6,165.7 341.2,166.4 342.7,167.3 344.2,168.4 345.7,169.8 347.3,171.4 348.8,173.4 350.3,175.7 351.8,178.4 353.4,181.6 354.9,185.6 356.4,190.4 357.9,196.8 359.5,205.8 361.0,221.1 362.5,230.0 364.0,221.8 365.5,207.2 367.1,198.9 368.6,193.2 370.1,189.0 371.6,185.8 373.2,183.2 374.7,181.2 376.2,179.6 377.7,178.4 379.3,177.4 380.8,176.7 382.3,176.3 383.8,176.1 385.4,176.1 386.9,176.3 388.4,176.7 389.9,177.3 391.4,178.2 393.0,179.3 394.5,180.6 396.0,182.2 397.5,184.1 399.1,186.3 400.6,189.0 402.1,192.2 403.6,196.0 405.2,200.9 406.7,207.2 408.2,216.1 409.7,230.0 411.2,230.0 412.8,230.0 414.3,217.3 415.8,208.9 417.3,203.2 418.9,199.0 420.4,195.7 421.9,193.1 423.4,191.0 425.0,189.4 426.5,188.1 428.0,187.1 429.5,186.4 431.1,185.9 432.6,185.6 434.1,185.6 435.6,185.8 437.1,186.1 438.7,186.7 440.2,187.5 441.7,188.6 443.2,189.8 444.8,191.4 446.3,193.2 447.8,195.4 449.3,198.1 450.9,201.2 452.4,205.0 453.9,209.8 455.4,216.1 457.0,225.0 458.5,230.0 460.0,230.0" class="stroke-a" fill="none"/>
<line x1="478.0" y1="64.0" x2="498.0" y2="64.0" class="stroke-a"/>
<text x="504.0" y="68.0" class="fig-small" text-anchor="start">Hann</text>
<polyline points="70.0,30.0 71.5,30.0 73.0,30.1 74.6,30.2 76.1,30.3 77.6,30.4 79.1,30.6 80.7,30.8 82.2,31.1 83.7,31.4 85.2,31.7 86.8,32.1 88.3,32.4 89.8,32.9 91.3,33.3 92.9,33.8 94.4,34.4 95.9,35.0 97.4,35.6 98.9,36.2 100.5,36.9 102.0,37.6 103.5,38.4 105.0,39.2 106.6,40.1 108.1,41.0 109.6,41.9 111.1,42.9 112.7,43.9 114.2,45.0 115.7,46.1 117.2,47.3 118.8,48.5 120.3,49.8 121.8,51.1 123.3,52.5 124.8,54.0 126.4,55.5 127.9,57.1 129.4,58.7 130.9,60.4 132.5,62.2 134.0,64.1 135.5,66.1 137.0,68.1 138.6,70.2 140.1,72.5 141.6,74.8 143.1,77.3 144.6,79.9 146.2,82.6 147.7,85.5 149.2,88.6 150.7,91.9 152.3,95.4 153.8,99.2 155.3,103.3 156.8,107.9 158.4,112.9 159.9,118.6 161.4,125.3 162.9,133.4 164.5,144.2 166.0,161.3 167.5,230.0 169.0,165.7 170.5,153.0 172.1,146.7 173.6,143.0 175.1,140.8 176.6,139.7 178.2,139.3 179.7,139.6 181.2,140.4 182.7,141.6 184.3,143.2 185.8,145.3 187.3,147.6 188.8,150.3 190.4,153.3 191.9,156.2 193.4,159.0 194.9,161.2 196.4,162.7 198.0,163.3 199.5,163.3 201.0,163.1 202.5,163.0 204.1,163.2 205.6,163.8 207.1,165.2 208.6,167.3 210.2,170.5 211.7,175.3 213.2,182.9 214.7,196.9 216.2,230.0 217.8,195.2 219.3,179.5 220.8,170.2 222.3,163.5 223.9,158.4 225.4,154.4 226.9,151.0 228.4,148.2 230.0,145.9 231.5,143.9 233.0,142.3 234.5,141.0 236.1,139.9 237.6,139.1 239.1,138.5 240.6,138.2 242.1,138.1 243.7,138.2 245.2,138.6 246.7,139.2 248.2,140.0 249.8,141.2 251.3,142.6 252.8,144.4 254.3,146.7 255.9,149.5 257.4,152.9 258.9,157.4 260.4,163.3 262.0,171.9 263.5,186.8 265.0,230.0 266.5,186.7 268.0,171.7 269.6,163.0 271.1,157.0 272.6,152.5 274.1,148.9 275.7,146.0 277.2,143.6 278.7,141.7 280.2,140.1 281.8,138.8 283.3,137.8 284.8,137.0 286.3,136.5 287.9,136.2 289.4,136.1 290.9,136.2 292.4,136.6 293.9,137.1 295.5,137.9 297.0,139.0 298.5,140.3 300.0,141.9 301.6,143.9 303.1,146.3 304.6,149.2 306.1,152.8 307.7,157.4 309.2,163.4 310.7,172.1 312.2,187.1 313.8,230.0 315.3,187.2 316.8,172.3 318.3,163.7 319.8,157.7 321.4,153.2 322.9,149.7 324.4,146.9 325.9,144.6 327.5,142.7 329.0,141.2 330.5,140.0 332.0,139.0 333.6,138.3 335.1,137.8 336.6,137.6 338.1,137.5 339.6,137.7 341.2,138.1 342.7,138.7 344.2,139.5 345.7,140.6 347.3,141.9 348.8,143.5 350.3,145.5 351.8,148.0 353.4,150.9 354.9,154.5 356.4,159.1 357.9,165.2 359.5,173.9 361.0,188.9 362.5,230.0 364.0,189.0 365.5,174.2 367.1,165.6 368.6,159.7 370.1,155.2 371.6,151.7 373.2,148.9 374.7,146.6 376.2,144.7 377.7,143.2 379.3,142.0 380.8,141.0 382.3,140.4 383.8,139.9 385.4,139.6 386.9,139.6 388.4,139.8 389.9,140.2 391.4,140.8 393.0,141.6 394.5,142.7 396.0,144.0 397.5,145.7 399.1,147.7 400.6,150.1 402.1,153.0 403.6,156.7 405.2,161.3 406.7,167.3 408.2,176.0 409.7,191.0 411.2,230.0 412.8,191.2 414.3,176.3 415.8,167.7 417.3,161.8 418.9,157.3 420.4,153.8 421.9,151.0 423.4,148.7 425.0,146.9 426.5,145.3 428.0,144.1 429.5,143.2 431.1,142.5 432.6,142.0 434.1,141.8 435.6,141.7 437.1,141.9 438.7,142.3 440.2,142.9 441.7,143.7 443.2,144.8 444.8,146.1 446.3,147.8 447.8,149.8 449.3,152.2 450.9,155.1 452.4,158.8 453.9,163.4 455.4,169.4 457.0,178.1 458.5,193.1 460.0,230.0" class="stroke-c" fill="none"/>
<line x1="478.0" y1="84.0" x2="498.0" y2="84.0" class="stroke-c"/>
<text x="504.0" y="88.0" class="fig-small" text-anchor="start">Hamming</text>
<polyline points="70.0,30.0 71.5,30.0 73.0,30.0 74.6,30.1 76.1,30.2 77.6,30.3 79.1,30.4 80.7,30.5 82.2,30.7 83.7,30.9 85.2,31.1 86.8,31.3 88.3,31.5 89.8,31.8 91.3,32.1 92.9,32.4 94.4,32.7 95.9,33.1 97.4,33.5 98.9,33.9 100.5,34.3 102.0,34.8 103.5,35.2 105.0,35.7 106.6,36.2 108.1,36.8 109.6,37.4 111.1,37.9 112.7,38.6 114.2,39.2 115.7,39.9 117.2,40.6 118.8,41.3 120.3,42.0 121.8,42.8 123.3,43.6 124.8,44.4 126.4,45.2 127.9,46.1 129.4,47.0 130.9,48.0 132.5,48.9 134.0,49.9 135.5,50.9 137.0,52.0 138.6,53.1 140.1,54.2 141.6,55.4 143.1,56.5 144.6,57.8 146.2,59.0 147.7,60.3 149.2,61.7 150.7,63.0 152.3,64.4 153.8,65.9 155.3,67.4 156.8,68.9 158.4,70.5 159.9,72.1 161.4,73.8 162.9,75.5 164.5,77.3 166.0,79.2 167.5,81.1 169.0,83.0 170.5,85.0 172.1,87.1 173.6,89.3 175.1,91.5 176.6,93.8 178.2,96.2 179.7,98.6 181.2,101.2 182.7,103.8 184.3,106.6 185.8,109.5 187.3,112.5 188.8,115.6 190.4,118.9 191.9,122.3 193.4,125.9 194.9,129.8 196.4,133.8 198.0,138.1 199.5,142.7 201.0,147.6 202.5,153.0 204.1,158.8 205.6,165.2 207.1,172.4 208.6,180.6 210.2,190.2 211.7,201.9 213.2,217.1 214.7,230.0 216.2,230.0 217.8,230.0 219.3,230.0 220.8,230.0 222.3,222.3 223.9,211.0 225.4,202.9 226.9,196.7 228.4,191.8 230.0,187.8 231.5,184.7 233.0,182.1 234.5,180.0 236.1,178.4 237.6,177.1 239.1,176.2 240.6,175.6 242.1,175.3 243.7,175.3 245.2,175.6 246.7,176.2 248.2,177.0 249.8,178.2 251.3,179.8 252.8,181.7 254.3,184.1 255.9,187.0 257.4,190.6 258.9,195.2 260.4,201.3 262.0,210.1 263.5,225.2 265.0,230.0 266.5,225.5 268.0,210.8 269.6,202.4 271.1,196.6 272.6,192.3 274.1,188.9 275.7,186.3 277.2,184.2 278.7,182.5 280.2,181.2 281.8,180.1 283.3,179.4 284.8,178.9 286.3,178.6 287.9,178.6 289.4,178.8 290.9,179.2 292.4,179.8 293.9,180.6 295.5,181.7 297.0,183.0 298.5,184.5 300.0,186.4 301.6,188.7 303.1,191.3 304.6,194.5 306.1,198.4 307.7,203.2 309.2,209.5 310.7,218.5 312.2,230.0 313.8,230.0 315.3,230.0 316.8,219.7 318.3,211.4 319.8,205.7 321.4,201.5 322.9,198.2 324.4,195.6 325.9,193.6 327.5,192.0 329.0,190.7 330.5,189.7 332.0,189.0 333.6,188.6 335.1,188.3 336.6,188.3 338.1,188.5 339.6,188.9 341.2,189.6 342.7,190.4 344.2,191.5 345.7,192.8 347.3,194.4 348.8,196.2 350.3,198.5 351.8,201.1 353.4,204.3 354.9,208.2 356.4,213.0 357.9,219.3 359.5,228.2 361.0,230.0 362.5,230.0 364.0,230.0 365.5,229.4 367.1,221.1 368.6,215.4 370.1,211.1 371.6,207.9 373.2,205.3 374.7,203.2 376.2,201.6 377.7,200.3 379.3,199.3 380.8,198.5 382.3,198.1 383.8,197.8 385.4,197.8 386.9,198.0 388.4,198.3 389.9,198.9 391.4,199.7 393.0,200.8 394.5,202.1 396.0,203.6 397.5,205.5 399.1,207.7 400.6,210.3 402.1,213.5 403.6,217.3 405.2,222.1 406.7,228.4 408.2,230.0 409.7,230.0 411.2,230.0 412.8,230.0 414.3,230.0 415.8,230.0 417.3,224.3 418.9,220.0 420.4,216.7 421.9,214.1 423.4,212.0 425.0,210.3 426.5,209.0 428.0,207.9 429.5,207.2 431.1,206.7 432.6,206.4 434.1,206.3 435.6,206.5 437.1,206.9 438.7,207.4 440.2,208.2 441.7,209.2 443.2,210.5 444.8,212.0 446.3,213.8 447.8,216.0 449.3,218.6 450.9,221.7 452.4,225.6 453.9,230.0 455.4,230.0 457.0,230.0 458.5,230.0 460.0,230.0" class="stroke-d" fill="none"/>
<line x1="478.0" y1="104.0" x2="498.0" y2="104.0" class="stroke-d"/>
<text x="504.0" y="108.0" class="fig-small" text-anchor="start">Blackman</text>
</svg>
Figure 3. Window spectra, zero-padded by 64. Notice that each step from rectangular to Blackman widens the main lobe and lowers the side lobes.
```

A wider main lobe blurs two tones that are close together. Lower side lobes let a quiet tone show next to a loud one. You can't have both.

> [!KEY]
> A window trades resolution for leakage. Going from rectangular to Blackman triples the main lobe and lowers the peak side lobe by about 45 dB.

The following demo lets you slide a tone between bins and switch windows.

```demo
{"id": "window-leakage", "caption": "With the tone halfway between two bins, switch the window between rectangular and Hann and watch the far bins drop by tens of decibels."}
```

**Zero-padding** does not buy resolution back. Padding a frame with zeros before the FFT samples the same spectrum on a finer grid, but the main lobe keeps its width. With a 25 ms Hann window, two tones 20 Hz apart stay one peak however much you pad. Only a longer window separates them.

> [!RECAP]
> - The FFT computes `X = concatenate([even + t, even - t])`, where `even` and `odd` are the FFTs of the even and odd samples and `t` is the twiddles times `odd`.
> - It recurses until a piece has one sample, so it costs $N \log N$ and needs a power-of-two length.
> - A frame that doesn't hold whole cycles has a seam when it repeats, and the seam leaks energy into every bin.
> - A cosine-sum window divides by $N$ when periodic and $N - 1$ when symmetric.
> - A window's trade is its main-lobe half-width against its peak side lobe, read from its zero-padded spectrum.

## Build it step by step

The checks test the functions in this order.

1. **`fft_radix2(x)`** returns the complex DFT of a 1-D signal. Raise `ValueError` for a length that is not a power of two, using `n & (n - 1)`. Recurse on `x[0::2]` and `x[1::2]`, and combine with `np.concatenate([even + t, even - t])`.
2. **`get_window(name, n, periodic)`** looks up the coefficients in `COEFFICIENTS` and evaluates the cosine sum. A common mistake is dividing by `n - 1` for the periodic window. A length-1 window is `[1.0]`.
3. **`leakage(window, oversample)`** zero-pads with `np.fft.rfft(window, len(window) * oversample)`, divides by the value at 0 Hz, walks to the first null, and returns the half-width in bins and the peak side lobe in dB.

## Where this breaks

**Leakage hides quiet sounds.** The **Run** block plays a 1010 Hz tone beside one 60 dB quieter at 2510 Hz. With no window, the leakage at 2 kHz is $-39.5$ dB and buries the quiet tone. Hann gets it to $-101$ dB. Use Hann or Blackman when levels differ by more than about 30 dB.

**Hamming's side lobes fall slowly.** Its nearest side lobes are lower than Hann's, but 20 bins out on 64 points they are still near $-51$ dB, while Hann's have fallen to about $-89$ dB. In the **Run** frame, Hamming's leakage at 2 kHz is $-56.9$ dB, and the quiet tone barely clears it.

**Magnitudes depend on the window.** A Hann window sums to $N/2$, so a tone of amplitude $A$ reads about $AN/4$ instead of $AN/2$. Use the same window whenever you compare spectra.

## The exercise

You build `fft_radix2`, `get_window`, and `leakage`. Press **Run** to see the four-point example, time a matrix DFT, your FFT, and numpy's on 1024 points, and push one 32 ms frame of a loud tone and a quiet one through all four windows.

The checks confirm the worked example, agreement with `np.fft.fft` at every power of two up to 1024, and the `ValueError` for other lengths. They compare every window with `scipy.signal.get_window` and your `leakage` with the table, confirm the $-92.7$ dB figure for Hann, and show that zero-padding a 25 ms window leaves 1000 Hz and 1020 Hz merged, while a 200 ms window separates them.
