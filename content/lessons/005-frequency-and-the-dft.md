---
title: Frequency and the DFT
slug: 005-frequency-and-the-dft
part: "Part 1 — Sound and signals"
summary: Find which frequencies a frame of samples holds by comparing it with sinusoids, and read the resulting bins in hertz.
minutes: 40
gpu: false
objectives:
  - Explain the discrete Fourier transform (DFT) as a bank of detectors, each comparing the signal with one complex sinusoid.
  - Compute a four-point DFT by hand and check its energy with Parseval's theorem.
  - Build the DFT as a matrix product, and explain why a real signal needs only half the bins.
  - Convert a bin index to hertz, and read the tones in a frame from the magnitudes.
prereqs:
  - 001-what-sound-is
  - 002-sampling-and-aliasing
exercise: 005-frequency-and-the-dft
---

# Frequency and the DFT

A tuner hears a guitar string and says "A, 440 Hz". All it has is a list of samples, a wiggling line that does not announce its pitch. To recognize a vowel, a speaker, or a word, a system needs the recipe: how much of each frequency the sound holds.

The **discrete Fourier transform** (DFT) computes that recipe from a frame of samples. Every spectrogram, mel filterbank, and neural front end in this course starts with it. This chapter builds it from one idea, comparing the frame with sinusoids, so that you can say why each number in the output means what it means. You work on the running example, the 440 Hz tone, and see it land in one bin.

## What you already know

- A tone is $A \cos(2\pi f t + \varphi)$, and a signal sampled at $f_s$ can only hold frequencies up to $f_s / 2$.
- A complex number $a + bj$ has a magnitude $\sqrt{a^2 + b^2}$ and an angle, and $e^{j\theta} = \cos\theta + j \sin\theta$.
- A dot product multiplies two lists sample by sample and adds. It is large when they move together.
- A matrix-vector product takes the dot product of each row with the vector.

## Ask each frequency whether it is there

The first idea is to count zero crossings: how often does the waveform pass through zero? A 440 Hz tone crosses zero 880 times a second, so a counter would say 440 Hz. But take a 25 ms frame of 440 Hz at amplitude 0.5 plus a 1000 Hz tone at amplitude 0.25. The counter reads 22 crossings, which still says 440 Hz, and the 1000 Hz tone has vanished. Counting works only for one tone, and a voice is dozens at once.

A better idea is to test each frequency separately. For a candidate frequency, build a cosine at that frequency, multiply it with the signal sample by sample, and add. This dot product is large when the signal contains that frequency, because the two move together and the products are mostly positive. It is near zero when they do not, because the products cancel.

Now a trap. Take a signal that is exactly a sine at the candidate frequency. A sine and a cosine at the same frequency are a quarter turn apart, and the dot product of one with the other is zero. A cosine-only detector reports "nothing there" about a tone at full strength. The detector must work whatever the phase.

The fix is to use two comparisons at once, one with a cosine and one with a sine. A complex sinusoid, $e^{-j\theta} = \cos\theta - j\sin\theta$, does both: the real part of the result measures the cosine match, and the imaginary part measures the sine match. Together they give the strength of the tone and its phase.

## The DFT is $N$ detectors

Take a frame of $N$ samples. The DFT tests $N$ candidate frequencies, one for each whole number $k$ of complete turns across the frame. Candidate $k$ is the complex sinusoid whose sample $n$ is $e^{-2\pi j k n / N}$. The result for candidate $k$ is one **bin**:

$$
X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-2\pi j k n / N}, \qquad k = 0, 1, \dots, N-1
$$

Read it aloud: bin $k$ is the sum, over all $N$ samples, of the sample times a complex sinusoid that turns $k$ times across the frame. Its magnitude $|X[k]|$ says how much of that frequency is present, and its angle says the phase. Figure 1 shows the sinusoid for $k = 1$ when $N = 4$. Each sample gets multiplied by one of four points a quarter turn apart on the unit circle: $1$, $-j$, $-1$, and $j$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The four twiddle factors of bin 1 for N equals 4 drawn on the unit circle: 1, minus j, minus 1, and j, each a quarter turn apart">
<defs><marker id="c005-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<circle cx="150" cy="150" r="80" class="stroke-muted" fill="none"/>
<line x1="55" y1="150" x2="245" y2="150" class="stroke-grid"/><line x1="150" y1="55" x2="150" y2="245" class="stroke-grid"/>
<line x1="150" y1="150" x2="230.0" y2="150.0" class="stroke-a" marker-end="url(#c005-arrow)"/>
<text x="242.0" y="142.0" class="fig-small" text-anchor="start">n=0: 1</text>
<line x1="150" y1="150" x2="150.0" y2="230.0" class="stroke-a" marker-end="url(#c005-arrow)"/>
<text x="158.0" y="246.0" class="fig-small" text-anchor="start">n=1: −j</text>
<line x1="150" y1="150" x2="70.0" y2="150.0" class="stroke-a" marker-end="url(#c005-arrow)"/>
<text x="62.0" y="140.0" class="fig-small" text-anchor="end">n=2: −1</text>
<line x1="150" y1="150" x2="150.0" y2="70.0" class="stroke-a" marker-end="url(#c005-arrow)"/>
<text x="160.0" y="62.0" class="fig-small" text-anchor="start">n=3: j</text>
<text x="150" y="290" class="fig-small" text-anchor="middle">e^(−2πj k n / N) for k = 1, N = 4</text>
<text x="330" y="60" class="fig-bold" text-anchor="start">Bin 1 multiplies sample n by one of these:</text>
<text x="330" y="86" class="fig-mono" text-anchor="start">x = 1, 2, 0, −1</text>
<text x="330" y="112" class="fig-mono fig-small" text-anchor="start">X[1] = 1·1 + 2·(−j) + 0·(−1) + (−1)·j</text>
<text x="330" y="136" class="fig-mono fig-small" text-anchor="start">     = 1 − 2j − j = 1 − 3j</text>
<text x="330" y="170" class="fig-mono fig-small" text-anchor="start">|X[1]| = √(1 + 9) = 3.162</text>
<text x="330" y="194" class="fig-mono fig-small" text-anchor="start">angle = −1.249 rad</text>
</svg>
Figure 1. The detector for bin 1 with a frame of four samples. Sample n is multiplied by the point at angle minus 2 pi n / 4: 1, then minus j, then minus 1, then j. The right side works X[1] for the frame 1, 2, 0, minus 1.
```

### Work a four-point DFT by hand

Take $x = [1, 2, 0, -1]$. The detector for bin $k$ uses the points $1$, $-j$, $-1$, $j$ raised to the power $k$, so for $k = 2$ the multipliers are $1, -1, 1, -1$ and for $k = 3$ they are $1, j, -1, -j$.

- $X[0] = 1 + 2 + 0 + (-1) = 2$. Bin 0 multiplies by 1 every time, so it is the sum, the **DC** component.
- $X[1] = 1 + 2(-j) + 0(-1) + (-1)(j) = 1 - 3j$.
- $X[2] = 1 - 2 + 0 - (-1) = 0$.
- $X[3] = 1 + 2(j) + 0(-1) + (-1)(-j) = 1 + 3j$.

The magnitudes are 2, 3.162, 0, and 3.162, drawn in Figure 2. Two properties show up already. $X[3]$ is the complex conjugate of $X[1]$. And the energy is preserved: $\sum |x|^2 = 1 + 4 + 0 + 1 = 6$, and $\frac{1}{4} \sum |X|^2 = \frac{1}{4}(4 + 10 + 0 + 10) = 6$. That is **Parseval's theorem**: the DFT moves energy between samples and bins without gaining or losing any.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Magnitude of the four-point DFT of 1, 2, 0, minus 1: bin 0 is 2, bins 1 and 3 are 3.162, and bin 2 is 0">
<line x1="70" y1="218.0" x2="570" y2="218.0" class="stroke-grid"/>
<text x="64" y="222.0" class="fig-small" text-anchor="end">0</text>
<line x1="70" y1="162.4" x2="570" y2="162.4" class="stroke-grid"/>
<text x="64" y="166.4" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="106.9" x2="570" y2="106.9" class="stroke-grid"/>
<text x="64" y="110.9" class="fig-small" text-anchor="end">2</text>
<line x1="70" y1="51.3" x2="570" y2="51.3" class="stroke-grid"/>
<text x="64" y="55.3" class="fig-small" text-anchor="end">3</text>
<line x1="149.5" y1="218" x2="149.5" y2="222" class="stroke-muted"/>
<text x="149.5" y="235" class="fig-small" text-anchor="middle">0</text>
<line x1="263.2" y1="218" x2="263.2" y2="222" class="stroke-muted"/>
<text x="263.2" y="235" class="fig-small" text-anchor="middle">1</text>
<line x1="376.8" y1="218" x2="376.8" y2="222" class="stroke-muted"/>
<text x="376.8" y="235" class="fig-small" text-anchor="middle">2</text>
<line x1="490.5" y1="218" x2="490.5" y2="222" class="stroke-muted"/>
<text x="490.5" y="235" class="fig-small" text-anchor="middle">3</text>
<line x1="70" y1="218" x2="570" y2="218" class="stroke-muted"/>
<line x1="70" y1="18" x2="70" y2="218" class="stroke-muted"/>
<text x="320.0" y="262" class="fig-small" text-anchor="middle">bin k</text>
<text x="14" y="118.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 118.0)">|X[k]|</text>
<rect x="129.5" y="106.9" width="40" height="111.1" class="fill-soft-b"/>
<text x="149.5" y="100.9" class="fig-small" text-anchor="middle">2.000</text>
<rect x="243.2" y="42.3" width="40" height="175.7" class="fill-soft-b"/>
<text x="263.2" y="36.3" class="fig-small" text-anchor="middle">3.162</text>
<rect x="356.8" y="218.0" width="40" height="0.0" class="fill-soft-b"/>
<text x="376.8" y="212.0" class="fig-small" text-anchor="middle">0.000</text>
<rect x="470.5" y="42.3" width="40" height="175.7" class="fill-soft-b"/>
<text x="490.5" y="36.3" class="fig-small" text-anchor="middle">3.162</text>
</svg>
Figure 2. The magnitude of the four-point DFT of 1, 2, 0, minus 1. Bins 1 and 3 hold equal magnitudes, because they are conjugates, and bin 2 is empty.
```

## One matrix, one product

Every bin is a dot product of the frame with one row of numbers. Stack those $N$ rows and the whole DFT becomes one matrix-vector product with the **DFT matrix** $W[k, n] = e^{-2\pi j k n / N}$:

$$
X = W x
$$

Read it aloud: the vector of bins is the DFT matrix times the vector of samples. In numpy, `np.outer(k, k)` gives every product $kn$ at once:

```python
k = np.arange(n)
W = np.exp(-2j * np.pi * np.outer(k, k) / n)
X = W @ x
```

numpy follows this convention: no $1/N$ on the forward transform. The inverse puts the sinusoids back together with a $1/N$, $x[n] = \frac{1}{N} \sum_k X[k] e^{2\pi j k n / N}$, so no information is lost. The matrix has $N^2$ entries, which suits small frames. The next chapter, [The FFT and window functions](/lessons/006-the-fft-and-windows), computes the same answer much faster.

## Read the bins in hertz

A bin number is a count of turns per frame, and you want hertz. A frame of $N$ samples at $f_s$ lasts $N / f_s$ seconds. A detector that turns $k$ times in that time turns $k f_s / N$ times per second, so bin $k$ sits at:

$$
f_k = \frac{k f_s}{N}
$$

Read it aloud: bin $k$ is $k$ times the sample rate divided by the frame length. The **bin spacing** is $f_s / N$.

For a real signal, there is a second saving. The sinusoid for bin $N - k$ turns the other way around the circle than bin $k$, so it is the mirror image of bin $k$, and for real samples $X[N - k]$ is the complex conjugate of $X[k]$. Bin $N - k$ is the negative frequency $-k f_s / N$ and carries no extra information. Audio code keeps bins 0 through $N/2$, which is what `np.fft.rfft` returns: $N/2 + 1$ complex numbers, the last one at the Nyquist frequency $f_s / 2$.

Whisper frames 16 kHz audio into 400-sample windows. Its spectrum has $400/2 + 1 = 201$ bins, 40 Hz apart, from 0 Hz to 8000 Hz.

> [!KEY]
> Bin $k$ of an $N$-point DFT is $k f_s / N$ hertz. A real signal needs only bins 0 to $N/2$, which is what `np.fft.rfft` returns.

## Find the tones in a frame

Take 400 samples of the running example, the 440 Hz tone with amplitude 0.5, at 16 kHz. The bin spacing is 40 Hz, and $440 / 40 = 11$, so the frame holds exactly 11 cycles. All the energy lands in bin 11. A cosine of amplitude $A$ that fits whole cycles lands in one bin with magnitude $A N / 2$, here $0.5 \times 400 / 2 = 100$, because the energy splits between bin $k$ and its mirror $N - k$. Every other bin is zero.

Now add a 1000 Hz tone with amplitude 0.25 and phase 0.7 radians. It falls on bin $1000 / 40 = 25$, with magnitude $0.25 \times 200 = 50$. Figure 3 shows the two spikes. The counter of zero crossings missed the second tone. The DFT shows it as a spike at half the height, 50 against 100, which is the ratio of the amplitudes, and its angle is 0.7 radians, the phase you put in.

```figure
<svg viewBox="0 0 600 280" role="img" aria-label="Magnitude of a 400-sample DFT of a mix of a 440 hertz tone of amplitude 0.5 and a 1000 hertz tone of amplitude 0.25: two spikes at bins 11 and 25, with magnitudes 100 and 50">
<line x1="64" y1="228.0" x2="576" y2="228.0" class="stroke-grid"/>
<text x="58" y="232.0" class="fig-small" text-anchor="end">0</text>
<line x1="64" y1="180.3" x2="576" y2="180.3" class="stroke-grid"/>
<text x="58" y="184.3" class="fig-small" text-anchor="end">25</text>
<line x1="64" y1="132.5" x2="576" y2="132.5" class="stroke-grid"/>
<text x="58" y="136.5" class="fig-small" text-anchor="end">50</text>
<line x1="64" y1="84.8" x2="576" y2="84.8" class="stroke-grid"/>
<text x="58" y="88.8" class="fig-small" text-anchor="end">75</text>
<line x1="64" y1="37.1" x2="576" y2="37.1" class="stroke-grid"/>
<text x="58" y="41.1" class="fig-small" text-anchor="end">100</text>
<line x1="64.0" y1="228" x2="64.0" y2="232" class="stroke-muted"/>
<text x="64.0" y="245" class="fig-small" text-anchor="middle">0</text>
<line x1="192.0" y1="228" x2="192.0" y2="232" class="stroke-muted"/>
<text x="192.0" y="245" class="fig-small" text-anchor="middle">500</text>
<line x1="320.0" y1="228" x2="320.0" y2="232" class="stroke-muted"/>
<text x="320.0" y="245" class="fig-small" text-anchor="middle">1000</text>
<line x1="448.0" y1="228" x2="448.0" y2="232" class="stroke-muted"/>
<text x="448.0" y="245" class="fig-small" text-anchor="middle">1500</text>
<line x1="576.0" y1="228" x2="576.0" y2="232" class="stroke-muted"/>
<text x="576.0" y="245" class="fig-small" text-anchor="middle">2000</text>
<line x1="64" y1="228" x2="576" y2="228" class="stroke-muted"/>
<line x1="64" y1="18" x2="64" y2="228" class="stroke-muted"/>
<text x="320.0" y="272" class="fig-small" text-anchor="middle">frequency (Hz)</text>
<text x="14" y="123.0" class="fig-small" text-anchor="middle" transform="rotate(-90 14 123.0)">|X[k]|</text>
<line x1="64.0" y1="228" x2="64.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="74.2" y1="228" x2="74.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="84.5" y1="228" x2="84.5" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="94.7" y1="228" x2="94.7" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="105.0" y1="228" x2="105.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="115.2" y1="228" x2="115.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="125.4" y1="228" x2="125.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="135.7" y1="228" x2="135.7" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="145.9" y1="228" x2="145.9" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="156.2" y1="228" x2="156.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="166.4" y1="228" x2="166.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="176.6" y1="228" x2="176.6" y2="37.1" class="stroke-b" stroke-width="2"/>
<line x1="186.9" y1="228" x2="186.9" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="197.1" y1="228" x2="197.1" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="207.4" y1="228" x2="207.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="217.6" y1="228" x2="217.6" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="227.8" y1="228" x2="227.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="238.1" y1="228" x2="238.1" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="248.3" y1="228" x2="248.3" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="258.6" y1="228" x2="258.6" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="268.8" y1="228" x2="268.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="279.0" y1="228" x2="279.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="289.3" y1="228" x2="289.3" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="299.5" y1="228" x2="299.5" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="309.8" y1="228" x2="309.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="320.0" y1="228" x2="320.0" y2="132.5" class="stroke-b" stroke-width="2"/>
<line x1="330.2" y1="228" x2="330.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="340.5" y1="228" x2="340.5" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="350.7" y1="228" x2="350.7" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="361.0" y1="228" x2="361.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="371.2" y1="228" x2="371.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="381.4" y1="228" x2="381.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="391.7" y1="228" x2="391.7" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="401.9" y1="228" x2="401.9" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="412.2" y1="228" x2="412.2" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="422.4" y1="228" x2="422.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="432.6" y1="228" x2="432.6" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="442.9" y1="228" x2="442.9" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="453.1" y1="228" x2="453.1" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="463.4" y1="228" x2="463.4" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="473.6" y1="228" x2="473.6" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="483.8" y1="228" x2="483.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="494.1" y1="228" x2="494.1" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="504.3" y1="228" x2="504.3" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="514.6" y1="228" x2="514.6" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="524.8" y1="228" x2="524.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="535.0" y1="228" x2="535.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="545.3" y1="228" x2="545.3" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="555.5" y1="228" x2="555.5" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="565.8" y1="228" x2="565.8" y2="228.0" class="stroke-b" stroke-width="2"/>
<line x1="576.0" y1="228" x2="576.0" y2="228.0" class="stroke-b" stroke-width="2"/>
<circle cx="176.6" cy="37.1" r="4" class="fill-b"/>
<circle cx="320.0" cy="132.5" r="4" class="fill-b"/>
<text x="186.6" y="41.1" class="fig-small" text-anchor="start">bin 11 = 440 Hz: 100</text>
<text x="330.0" y="136.5" class="fig-small" text-anchor="start">bin 25 = 1000 Hz: 50</text>
</svg>
Figure 3. The 400-point spectrum of a 440 Hz tone (amplitude 0.5) plus a 1000 Hz tone (amplitude 0.25). The spikes sit at bins 11 and 25, with magnitudes 100 and 50, and every other bin is empty.
```

> [!WARNING]
> This clean picture needs a tone that fits a whole number of cycles in the frame. A tone at 450 Hz would fall between bins 11 and 12 and smear into its neighbors. That is spectral leakage, and the window functions in the next chapter control it.

> [!KEY]
> Back to the tuner. The DFT compares a frame with $N$ complex sinusoids, and bin $k$ of the result tells you the strength and phase of the tone at $k f_s / N$ hertz. The running example sits in one bin at 440 Hz, and the second tone appears in its own.

> [!RECAP]
> - The DFT is $X[k] = \sum_n x[n] e^{-2\pi j k n / N}$: a dot product of the frame with one complex sinusoid per bin.
> - The DFT matrix is `np.exp(-2j * np.pi * np.outer(k, k) / n)`, and the DFT is that matrix times the signal.
> - Parseval: $\sum |x|^2 = \frac{1}{N}\sum |X|^2$.
> - Bin $k$ is $k f_s / N$ hertz, and a real signal needs bins 0 to $N/2$.
> - A cosine of amplitude $A$ with whole cycles in the frame has $|X[k]| = A N / 2$.

## Build it step by step

The checks test the functions in this order.

1. **`dft_matrix(n)`** returns the complex `(n, n)` matrix. Use `np.outer` and `np.exp`. Common mistake: a plus sign in the exponent, which gives the inverse transform's matrix.
2. **`dft(x)`** returns the complex DFT of a 1-D signal, using your `dft_matrix`. Do not scale by $1/N$.
3. **`rdft(x)`** returns the first `len(x) // 2 + 1` bins of your `dft`. Common mistake: `len(x) // 2`, which drops the Nyquist bin.
4. **`rfft_frequencies(n_fft, sample_rate)`** returns `n_fft // 2 + 1` frequencies. Common mistake: using `np.linspace(0, fs / 2, n_fft // 2)`, which is one bin short and slightly stretched.

## Where this breaks

**Leakage smears tones that do not fit.** Whole cycles in the frame is a special case. In general a tone spreads over several bins, and a quiet sound near a loud one can vanish under that spread. The next chapter measures it and fixes it with windows.

**Magnitudes depend on the frame length.** The peak of a tone is $A N / 2$, so a longer frame gives a taller peak for the same sound. Divide by $N/2$ to read amplitudes.

**Bins are coarse in a short frame.** A 25 ms frame at 16 kHz has 40 Hz bins, so two tones 20 Hz apart cannot be told apart, however you compute the transform. A longer frame has finer bins but blurs how the sound changes over time. The chapter after the next, [The short-time Fourier transform](/lessons/007-the-short-time-fourier-transform), is about that trade.

**The full spectrum shows a mirror.** Plotting `np.abs(np.fft.fft(x))` over all $N$ bins draws every tone twice. Use the real transform for real signals.

## The exercise

You build `dft_matrix`, `dft`, `rdft`, and `rfft_frequencies`. Press **Run** to see the four-point example, then find a 440 Hz tone and a quieter 1000 Hz tone in a 25 ms frame from their bins, and plot the spectrum.

The checks confirm the twiddles of `dft_matrix(4)` and that $W W^H = N I$, the worked example, and agreement with `np.fft.fft` on random signals. They show that a sine at bin 1 has the same magnitude as a cosine, test Parseval's theorem and the conjugate symmetry of a real signal, and compare `rdft` with `np.fft.rfft` at even and odd lengths. They test your bin frequencies against `np.fft.rfftfreq`, and the last check finds the running example in bin 11, 100 high, and the 1000 Hz tone in bin 25 with its phase.
