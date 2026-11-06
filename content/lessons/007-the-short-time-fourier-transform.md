---
title: The short-time Fourier transform
slug: 007-the-short-time-fourier-transform
part: "Part 1 — Sound and signals"
summary: Cut a signal into windowed frames, take the FFT of each to get a spectrogram, and invert it exactly by least-squares overlap-add.
minutes: 45
gpu: false
objectives:
  - Count and cut the frames of a centered, reflect-padded STFT, and state its layout of bins by frames.
  - Build the STFT and match torch.stft to within rounding.
  - "Explain the trade-off between time and frequency resolution: a longer window sees finer frequency and coarser time."
  - Test the COLA and NOLA conditions, and invert an STFT by least squares to 1e-10.
prereqs:
  - 006-the-fft-and-windows
exercise: 007-the-short-time-fourier-transform
---

# The short-time Fourier transform

Say the word "sip" and record it. The clip holds a hiss, then a vowel, then a short burst of air. One FFT of the whole clip tells you which frequencies were present, but it mixes all three sounds into one spectrum and says nothing about when each one happened. Speech changes every few tens of milliseconds, so a recognizer needs a spectrum that changes with it.

The fix is to cut the clip into short, overlapping frames and take one FFT per frame. Stack those spectra side by side, with time across and frequency up, and you have a **spectrogram**. It is sheet music for sound: each column says which notes are sounding at that moment. The whole stack is the **short-time Fourier transform** (STFT). This chapter fixes the conventions every later chapter uses, shows the one trade-off you can't avoid, and then runs the transform backward to recover the signal to ten decimal places.

## What you already know

- Bin $k$ of an $N$-point FFT sits at $k f_s / N$ hertz, and `np.fft.rfft` returns bins 0 to $N/2$.
- A window fades a frame in and out, so the frame has no seam when it repeats, at the cost of a wider main lobe.
- A periodic Hann window is $0.5 - 0.5 \cos(2\pi n / N)$, and it tiles cleanly when frames overlap.
- A tone that fits the frame lands in one bin, and zero-padding refines the grid but not the resolution.
- numpy broadcasting builds a 2-D index array from a row and a column.

## Cut the signal into frames

A **frame** is a slice of $N$ consecutive samples, where $N$ is the frame length, called `n_fft` when it equals the FFT length. The **hop** $H$, called `hop_length`, is the number of samples from the start of one frame to the start of the next. Frames overlap when $H < N$. Whisper uses $N = 400$ and $H = 160$ at 16 kHz, which means 25 ms frames every 10 ms.

Overlap matters because a window fades a frame's edges to nearly zero. With a hop smaller than the frame, every sound lands near the middle of some frame.

Without padding, frame $m$ covers samples $mH$ to $mH + N - 1$, and only whole frames fit. That drops the last few samples, and it makes frame 0 start at sample 0 rather than center on it. This course **centers** the frames instead: it pads $N/2$ samples at each end, so frame $m$ is centered on sample $mH$. For a signal of $L$ samples, the frame counts are the following:

$$
M_{\text{uncentered}} = 1 + \left\lfloor \frac{L - N}{H} \right\rfloor, \qquad
M_{\text{centered}} = 1 + \left\lfloor \frac{L}{H} \right\rfloor
$$

Read it aloud: without padding, one frame fits at the start, plus one for every whole hop in the samples left over. With centering, there is one frame for every whole hop in the signal, plus the one centered on sample 0. The symbol $\lfloor \cdot \rfloor$ means round down, `//` in Python. One second at 16 kHz with Whisper's settings gives $1 + \lfloor 15600/160 \rfloor = 98$ frames uncentered and $1 + \lfloor 16000/160 \rfloor = 101$ centered. Whisper's 30 seconds gives 3001 centered frames, and Whisper drops the last one to get 3000.

### Pad by reflection

What should the padding hold? Zeros would work, but they make the first and last frames quieter than the rest. This course pads by **reflection**, the default of `torch.stft`: the signal is mirrored about its first and last sample, without repeating them. librosa pads with zeros by default, so pass the mode explicitly when you compare libraries.

Take the signal $0, 1, \dots, 9$ with $N = 8$ and $H = 4$. Centering pads 4 samples at each end: $4, 3, 2, 1$ before and $8, 7, 6, 5$ after. There are $1 + \lfloor 10/4 \rfloor = 3$ frames, shown in Figure 1:

- Frame 0 is $4, 3, 2, 1, 0, 1, 2, 3$, centered on sample 0.
- Frame 1 is $0, 1, 2, 3, 4, 5, 6, 7$, centered on sample 4.
- Frame 2 is $4, 5, 6, 7, 8, 9, 8, 7$, centered on sample 8.

```figure
<svg viewBox="0 0 580 246" role="img" aria-label="The signal 0 to 9, reflect-padded by 4 samples at each end, cut into three frames of 8 samples with a hop of 4"><defs><marker id="c007-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="24.0" y="24.0" class="fig-bold" text-anchor="start">Padded signal: reflected samples (orange) around the original 0 to 9 (blue)</text>
<rect x="24.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="24.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="39.0" y="70.0" text-anchor="middle">4</text>
<rect x="54.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="54.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="69.0" y="70.0" text-anchor="middle">3</text>
<rect x="84.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="84.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="99.0" y="70.0" text-anchor="middle">2</text>
<rect x="114.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="114.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="129.0" y="70.0" text-anchor="middle">1</text>
<rect x="144.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="144.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="159.0" y="70.0" text-anchor="middle">0</text>
<rect x="174.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="174.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="189.0" y="70.0" text-anchor="middle">1</text>
<rect x="204.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="204.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="219.0" y="70.0" text-anchor="middle">2</text>
<rect x="234.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="234.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="249.0" y="70.0" text-anchor="middle">3</text>
<rect x="264.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="264.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="279.0" y="70.0" text-anchor="middle">4</text>
<rect x="294.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="294.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="309.0" y="70.0" text-anchor="middle">5</text>
<rect x="324.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="324.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="339.0" y="70.0" text-anchor="middle">6</text>
<rect x="354.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="354.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="369.0" y="70.0" text-anchor="middle">7</text>
<rect x="384.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="384.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="399.0" y="70.0" text-anchor="middle">8</text>
<rect x="414.0" y="50.0" width="30.0" height="30.0" class="fill-soft-a"/>
<rect x="414.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="429.0" y="70.0" text-anchor="middle">9</text>
<rect x="444.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="444.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="459.0" y="70.0" text-anchor="middle">8</text>
<rect x="474.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="474.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="489.0" y="70.0" text-anchor="middle">7</text>
<rect x="504.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="504.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="519.0" y="70.0" text-anchor="middle">6</text>
<rect x="534.0" y="50.0" width="30.0" height="30.0" class="fill-soft-b"/>
<rect x="534.0" y="50.0" width="30.0" height="30.0" class="stroke-muted"/>
<text x="549.0" y="70.0" text-anchor="middle">5</text>
<rect x="24.0" y="106.0" width="240.0" height="22.0" class="fill-soft-a"/>
<rect x="24.0" y="106.0" width="240.0" height="22.0" class="stroke-a"/>
<text x="144.0" y="122.0" class="fig-small" text-anchor="middle">frame 0: centered on sample 0</text>
<line x1="159.0" y1="80.0" x2="159.0" y2="106.0" class="stroke-muted dashed"/>
<rect x="144.0" y="140.0" width="240.0" height="22.0" class="fill-soft-c"/>
<rect x="144.0" y="140.0" width="240.0" height="22.0" class="stroke-c"/>
<text x="264.0" y="156.0" class="fig-small" text-anchor="middle">frame 1: centered on sample 4</text>
<line x1="279.0" y1="80.0" x2="279.0" y2="140.0" class="stroke-muted dashed"/>
<rect x="264.0" y="174.0" width="240.0" height="22.0" class="fill-soft-d"/>
<rect x="264.0" y="174.0" width="240.0" height="22.0" class="stroke-d"/>
<text x="384.0" y="190.0" class="fig-small" text-anchor="middle">frame 2: centered on sample 8</text>
<line x1="399.0" y1="80.0" x2="399.0" y2="174.0" class="stroke-muted dashed"/>
<text x="24.0" y="230.0" class="fig-small" text-anchor="start">Frame length 8, hop 4: each frame starts 4 samples after the last.</text>
</svg>
Figure 1. Three centered frames of the signal 0 to 9. Notice that the first and last frames reach into the reflected padding, so every original sample sits in the middle of some frame.
```

Uncentered, only $1 + \lfloor 2/4 \rfloor = 1$ frame fits, and samples 8 and 9 are never analyzed. Index every frame at once with broadcasting:

```python
x = np.pad(x, n_fft // 2, mode="reflect")
index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
frames = x[index]        # (n_frames, n_fft)
```

## Take an FFT of every frame

The STFT windows each frame and takes its DFT:

$$
X[k, m] = \sum_{n=0}^{N-1} x[n + mH] \, w[n] \, e^{-2\pi j k n / N}
$$

Read it aloud: entry $(k, m)$ is bin $k$ of frame $m$. Take the samples of frame $m$, multiply them by the window $w$, and compare the result with a sinusoid at bin $k$. Here $x$ is the padded signal. $X$ has $N/2 + 1$ rows, one per bin, and $M$ columns, one per frame. With $N = 400$ that is 201 rows.

```python
X = np.fft.rfft(frames * window, axis=1).T      # (n_fft // 2 + 1, n_frames)
```

The **magnitude spectrogram** is $|X|$ and the **power spectrogram** is $|X|^2$. Whisper and most mel front ends use power. In decibels the two agree if you use the right rule: $20 \log_{10} |X|$ and $10 \log_{10} |X|^2$ are the same number.

> [!KEY]
> The conventions of this course: frames are centered with reflect padding, the window is a periodic Hann, and a spectrogram is shaped (bins, frames) with row 0 at 0 Hz. Column $m$ is centered at $mH / f_s$ seconds.

`np.fft.rfft(frames, axis=1)` gives frames by bins, so transpose it. That layout is what `torch.stft` returns and what `speechlab.show_spectrogram` draws, with row 0 at the bottom. Pass `power=True` to `show_spectrogram` when you hand it a power spectrogram, so that it takes $10 \log_{10}$ instead of $20$.

## Time against frequency

You choose the frame length $N$, and it sets both resolutions at once. The bins are $f_s / N$ hertz apart, and each frame spans $N / f_s$ seconds. At 16 kHz, a 128-point frame spans 8 ms with bins 125 Hz apart. A 2048-point frame spans 128 ms with bins 7.8 Hz apart. A long frame separates close frequencies but smears events that are close in time, and a short frame does the opposite. Figure 2 shows two tones 80 Hz apart through both.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Spectrum of tones at 2000 and 2080 Hz through a 128-point and a 2048-point Hann window">
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="207.1" x2="570.0" y2="207.1" class="stroke-grid"/>
<text x="62.0" y="211.1" class="fig-small" text-anchor="end">0.25</text>
<line x1="70.0" y1="164.3" x2="570.0" y2="164.3" class="stroke-grid"/>
<text x="62.0" y="168.3" class="fig-small" text-anchor="end">0.5</text>
<line x1="70.0" y1="121.4" x2="570.0" y2="121.4" class="stroke-grid"/>
<text x="62.0" y="125.4" class="fig-small" text-anchor="end">0.75</text>
<line x1="70.0" y1="78.6" x2="570.0" y2="78.6" class="stroke-grid"/>
<text x="62.0" y="82.6" class="fig-small" text-anchor="end">1</text>
<line x1="113.1" y1="250.0" x2="113.1" y2="255.0" class="stroke-muted"/>
<text x="113.1" y="270.0" class="fig-small" text-anchor="middle">1800</text>
<line x1="199.3" y1="250.0" x2="199.3" y2="255.0" class="stroke-muted"/>
<text x="199.3" y="270.0" class="fig-small" text-anchor="middle">1900</text>
<line x1="285.5" y1="250.0" x2="285.5" y2="255.0" class="stroke-muted"/>
<text x="285.5" y="270.0" class="fig-small" text-anchor="middle">2000</text>
<line x1="371.7" y1="250.0" x2="371.7" y2="255.0" class="stroke-muted"/>
<text x="371.7" y="270.0" class="fig-small" text-anchor="middle">2100</text>
<line x1="457.9" y1="250.0" x2="457.9" y2="255.0" class="stroke-muted"/>
<text x="457.9" y="270.0" class="fig-small" text-anchor="middle">2200</text>
<line x1="544.1" y1="250.0" x2="544.1" y2="255.0" class="stroke-muted"/>
<text x="544.1" y="270.0" class="fig-small" text-anchor="middle">2300</text>
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="70.0" class="stroke-muted"/>
<text x="320.0" y="290.0" text-anchor="middle">Frequency (Hz)</text>
<text x="30.0" y="56.0" text-anchor="start">Relative magnitude</text>
<polyline points="70.0,246.9 71.7,246.5 73.4,246.1 75.1,245.7 76.7,245.2 78.4,244.6 80.1,244.1 81.8,243.5 83.5,242.8 85.2,242.2 86.8,241.5 88.5,240.8 90.2,240.0 91.9,239.2 93.6,238.4 95.3,237.6 96.9,236.7 98.6,235.9 100.3,235.0 102.0,234.0 103.7,233.1 105.4,232.1 107.0,231.0 108.7,230.0 110.4,228.9 112.1,227.9 113.8,226.7 115.5,225.6 117.1,224.4 118.8,223.3 120.5,222.0 122.2,220.8 123.9,219.6 125.6,218.3 127.2,217.0 128.9,215.7 130.6,214.3 132.3,213.0 134.0,211.6 135.7,210.2 137.3,208.8 139.0,207.3 140.7,205.9 142.4,204.4 144.1,202.9 145.8,201.4 147.5,199.9 149.1,198.3 150.8,196.8 152.5,195.2 154.2,193.6 155.9,192.0 157.6,190.4 159.2,188.8 160.9,187.2 162.6,185.5 164.3,183.9 166.0,182.2 167.7,180.6 169.3,178.9 171.0,177.2 172.7,175.5 174.4,173.8 176.1,172.1 177.8,170.4 179.4,168.7 181.1,167.0 182.8,165.3 184.5,163.6 186.2,161.9 187.9,160.2 189.5,158.5 191.2,156.8 192.9,155.1 194.6,153.4 196.3,151.7 198.0,150.0 199.6,148.4 201.3,146.7 203.0,145.0 204.7,143.4 206.4,141.7 208.1,140.1 209.7,138.5 211.4,136.9 213.1,135.3 214.8,133.7 216.5,132.1 218.2,130.5 219.9,129.0 221.5,127.4 223.2,125.9 224.9,124.4 226.6,122.9 228.3,121.4 230.0,120.0 231.6,118.6 233.3,117.1 235.0,115.7 236.7,114.4 238.4,113.0 240.1,111.7 241.7,110.4 243.4,109.1 245.1,107.8 246.8,106.6 248.5,105.3 250.2,104.1 251.8,103.0 253.5,101.8 255.2,100.7 256.9,99.6 258.6,98.5 260.3,97.5 261.9,96.4 263.6,95.4 265.3,94.5 267.0,93.5 268.7,92.6 270.4,91.7 272.0,90.8 273.7,90.0 275.4,89.2 277.1,88.4 278.8,87.7 280.5,87.0 282.1,86.3 283.8,85.6 285.5,85.0 287.2,84.4 288.9,83.8 290.6,83.2 292.3,82.7 293.9,82.2 295.6,81.8 297.3,81.4 299.0,81.0 300.7,80.6 302.4,80.3 304.0,79.9 305.7,79.7 307.4,79.4 309.1,79.2 310.8,79.0 312.5,78.9 314.1,78.8 315.8,78.7 317.5,78.6 319.2,78.6 320.9,78.6 322.6,78.6 324.2,78.7 325.9,78.8 327.6,78.9 329.3,79.0 331.0,79.2 332.7,79.4 334.3,79.7 336.0,80.0 337.7,80.3 339.4,80.6 341.1,81.0 342.8,81.4 344.4,81.8 346.1,82.3 347.8,82.7 349.5,83.3 351.2,83.8 352.9,84.4 354.6,85.0 356.2,85.6 357.9,86.3 359.6,87.0 361.3,87.7 363.0,88.5 364.7,89.2 366.3,90.0 368.0,90.9 369.7,91.7 371.4,92.6 373.1,93.5 374.8,94.5 376.4,95.5 378.1,96.5 379.8,97.5 381.5,98.5 383.2,99.6 384.9,100.7 386.5,101.8 388.2,103.0 389.9,104.2 391.6,105.4 393.3,106.6 395.0,107.9 396.6,109.1 398.3,110.4 400.0,111.7 401.7,113.1 403.4,114.4 405.1,115.8 406.7,117.2 408.4,118.6 410.1,120.0 411.8,121.5 413.5,123.0 415.2,124.5 416.8,126.0 418.5,127.5 420.2,129.0 421.9,130.6 423.6,132.1 425.3,133.7 427.0,135.3 428.6,136.9 430.3,138.5 432.0,140.2 433.7,141.8 435.4,143.4 437.1,145.1 438.7,146.8 440.4,148.4 442.1,150.1 443.8,151.8 445.5,153.5 447.2,155.2 448.8,156.9 450.5,158.6 452.2,160.3 453.9,162.0 455.6,163.7 457.3,165.4 458.9,167.1 460.6,168.8 462.3,170.5 464.0,172.2 465.7,173.9 467.4,175.6 469.0,177.3 470.7,179.0 472.4,180.6 474.1,182.3 475.8,184.0 477.5,185.6 479.1,187.2 480.8,188.9 482.5,190.5 484.2,192.1 485.9,193.7 487.6,195.3 489.2,196.8 490.9,198.4 492.6,199.9 494.3,201.5 496.0,203.0 497.7,204.5 499.4,205.9 501.0,207.4 502.7,208.8 504.4,210.2 506.1,211.6 507.8,213.0 509.5,214.4 511.1,215.7 512.8,217.0 514.5,218.3 516.2,219.6 517.9,220.9 519.6,222.1 521.2,223.3 522.9,224.5 524.6,225.7 526.3,226.8 528.0,227.9 529.7,229.0 531.3,230.1 533.0,231.1 534.7,232.1 536.4,233.1 538.1,234.1 539.8,235.0 541.4,235.9 543.1,236.8 544.8,237.6 546.5,238.5 548.2,239.3 549.9,240.1 551.5,240.8 553.2,241.5 554.9,242.2 556.6,242.9 558.3,243.5 560.0,244.1 561.6,244.7 563.3,245.2 565.0,245.7 566.7,246.1 568.4,246.5" class="stroke-b" fill="none"/>
<polyline points="70.0,250.0 71.7,250.0 73.4,250.0 75.1,250.0 76.7,250.0 78.4,250.0 80.1,250.0 81.8,250.0 83.5,250.0 85.2,250.0 86.8,250.0 88.5,250.0 90.2,250.0 91.9,250.0 93.6,250.0 95.3,250.0 96.9,250.0 98.6,250.0 100.3,250.0 102.0,250.0 103.7,250.0 105.4,250.0 107.0,250.0 108.7,250.0 110.4,250.0 112.1,250.0 113.8,250.0 115.5,250.0 117.1,250.0 118.8,250.0 120.5,250.0 122.2,250.0 123.9,250.0 125.6,250.0 127.2,250.0 128.9,250.0 130.6,250.0 132.3,250.0 134.0,250.0 135.7,250.0 137.3,250.0 139.0,250.0 140.7,250.0 142.4,250.0 144.1,250.0 145.8,250.0 147.5,250.0 149.1,250.0 150.8,250.0 152.5,250.0 154.2,250.0 155.9,250.0 157.6,250.0 159.2,250.0 160.9,250.0 162.6,250.0 164.3,250.0 166.0,250.0 167.7,250.0 169.3,250.0 171.0,250.0 172.7,250.0 174.4,250.0 176.1,250.0 177.8,250.0 179.4,250.0 181.1,250.0 182.8,250.0 184.5,250.0 186.2,250.0 187.9,250.0 189.5,250.0 191.2,250.0 192.9,250.0 194.6,250.0 196.3,250.0 198.0,250.0 199.6,250.0 201.3,250.0 203.0,250.0 204.7,250.0 206.4,250.0 208.1,250.0 209.7,250.0 211.4,250.0 213.1,250.0 214.8,249.9 216.5,250.0 218.2,250.0 219.9,250.0 221.5,249.9 223.2,249.9 224.9,250.0 226.6,249.9 228.3,249.9 230.0,249.9 231.6,250.0 233.3,249.9 235.0,249.9 236.7,249.9 238.4,250.0 240.1,249.9 241.7,249.8 243.4,249.8 245.1,250.0 246.8,249.8 248.5,249.7 250.2,249.7 251.8,250.0 253.5,249.6 255.2,249.4 256.9,249.5 258.6,250.0 260.3,249.2 261.9,248.6 263.6,248.7 265.3,250.0 267.0,247.9 268.7,245.8 270.4,245.8 272.0,250.0 273.7,239.3 275.4,220.9 277.1,195.1 278.8,164.3 280.5,132.4 282.1,104.5 283.8,85.4 285.5,78.6 287.2,85.3 288.9,104.4 290.6,132.3 292.3,164.2 293.9,195.1 295.6,220.9 297.3,239.4 299.0,249.9 300.7,245.8 302.4,245.9 304.0,248.0 305.7,249.9 307.4,248.8 309.1,248.7 310.8,249.3 312.5,249.8 314.1,249.5 315.8,249.5 317.5,249.7 319.2,249.7 320.9,249.7 322.6,249.7 324.2,249.5 325.9,249.5 327.6,249.9 329.3,249.3 331.0,248.7 332.7,248.8 334.3,249.9 336.0,247.9 337.7,245.8 339.4,245.8 341.1,249.8 342.8,238.8 344.4,220.0 346.1,194.0 347.8,163.0 349.5,131.1 351.2,103.5 352.9,84.8 354.6,78.6 356.2,85.9 357.9,105.5 359.6,133.6 361.3,165.5 363.0,196.2 364.7,221.8 366.3,239.9 368.0,249.7 369.7,245.7 371.4,245.9 373.1,248.0 374.8,249.9 376.4,248.7 378.1,248.6 379.8,249.2 381.5,250.0 383.2,249.4 384.9,249.4 386.5,249.6 388.2,250.0 389.9,249.7 391.6,249.7 393.3,249.8 395.0,250.0 396.6,249.8 398.3,249.8 400.0,249.9 401.7,250.0 403.4,249.9 405.1,249.9 406.7,249.9 408.4,250.0 410.1,249.9 411.8,249.9 413.5,249.9 415.2,250.0 416.8,249.9 418.5,249.9 420.2,250.0 421.9,250.0 423.6,250.0 425.3,249.9 427.0,250.0 428.6,250.0 430.3,250.0 432.0,250.0 433.7,250.0 435.4,250.0 437.1,250.0 438.7,250.0 440.4,250.0 442.1,250.0 443.8,250.0 445.5,250.0 447.2,250.0 448.8,250.0 450.5,250.0 452.2,250.0 453.9,250.0 455.6,250.0 457.3,250.0 458.9,250.0 460.6,250.0 462.3,250.0 464.0,250.0 465.7,250.0 467.4,250.0 469.0,250.0 470.7,250.0 472.4,250.0 474.1,250.0 475.8,250.0 477.5,250.0 479.1,250.0 480.8,250.0 482.5,250.0 484.2,250.0 485.9,250.0 487.6,250.0 489.2,250.0 490.9,250.0 492.6,250.0 494.3,250.0 496.0,250.0 497.7,250.0 499.4,250.0 501.0,250.0 502.7,250.0 504.4,250.0 506.1,250.0 507.8,250.0 509.5,250.0 511.1,250.0 512.8,250.0 514.5,250.0 516.2,250.0 517.9,250.0 519.6,250.0 521.2,250.0 522.9,250.0 524.6,250.0 526.3,250.0 528.0,250.0 529.7,250.0 531.3,250.0 533.0,250.0 534.7,250.0 536.4,250.0 538.1,250.0 539.8,250.0 541.4,250.0 543.1,250.0 544.8,250.0 546.5,250.0 548.2,250.0 549.9,250.0 551.5,250.0 553.2,250.0 554.9,250.0 556.6,250.0 558.3,250.0 560.0,250.0 561.6,250.0 563.3,250.0 565.0,250.0 566.7,250.0 568.4,250.0" class="stroke-a" fill="none"/>
<line x1="300.0" y1="16.0" x2="325.0" y2="16.0" class="stroke-b"/>
<text x="331.0" y="20.0" class="fig-small" text-anchor="start">128 points: 8 ms, bins 125 Hz apart</text>
<line x1="300.0" y1="36.0" x2="325.0" y2="36.0" class="stroke-a"/>
<text x="331.0" y="40.0" class="fig-small" text-anchor="start">2048 points: 128 ms, bins 7.8 Hz apart</text>
<line x1="285.5" y1="250.0" x2="285.5" y2="76.6" class="stroke-muted dashed"/>
<line x1="354.5" y1="250.0" x2="354.5" y2="76.6" class="stroke-muted dashed"/>
</svg>
Figure 2. Tones at 2000 and 2080 Hz through a short and a long Hann window. Notice that the 8 ms frame sees one hump, because its bins are wider than the gap, and the 128 ms frame sees two peaks.
```

This is a property of signals, not of the FFT. A sound that is short in time must be spread out in frequency, and the reverse. A **wideband** spectrogram, about 5 ms per frame, shows each glottal pulse as a vertical line, and a **narrowband** one, 30 to 50 ms, resolves the voice's harmonics as horizontal lines. Recognizers sit in between, at 25 ms. The hop sets how often you look, not how sharply.

```demo
{"id": "stft-explorer", "caption": "Lengthen the window until the tones at 2000 and 2080 Hz separate, then notice that the two clicks have merged into one smear."}
```

> [!KEY]
> A frame of $N$ samples gives bins $f_s / N$ apart and blurs time over $N / f_s$ seconds. Pick $N$ for the detail you need, then pick the hop.

## Put the frames back together

Can you go back from the spectrogram to the signal? Invert each frame's FFT, put each frame back where it came from, and add the overlaps. This is **overlap-add**. The obvious version fails in a quiet way. Each frame was multiplied by the window once on the way in, so the inverse FFT of frame $m$ gives $w[n] \, x[n + mH]$, and adding those back leaves the signal multiplied by the sum of the windows, which isn't flat.

The fix follows from the argument. Multiply each inverted frame by the window a second time, so every frame contributes $w^2 x$. Overlap-add the squared windows too, and divide by their sum. The window shape cancels. This is the **least-squares inverse**, due to Griffin and Lim:

$$
\hat{x}[n] = \frac{\sum_m w[n - mH] \, y_m[n - mH]}{\sum_m w^2[n - mH]}
$$

Read it aloud: $y_m$ is the inverse FFT of column $m$. At sample $n$, add up every frame that covers it, each weighted by the window at that position, then divide by the sum of the squared window values at the same positions. When $X$ really is the STFT of a signal, this returns the signal exactly. When $X$ has been modified, as in the next chapter, it returns the signal whose STFT is closest to $X$ in the least-squares sense.

```python
frames = np.fft.irfft(X.T, n=n_fft, axis=1) * window
for m in range(n_frames):
    out[m * hop : m * hop + n_fft] += frames[m]
    weight[m * hop : m * hop + n_fft] += window ** 2
x_hat = out[n_fft // 2 :][:length] / weight[n_fft // 2 :][:length]
```

The division needs a positive denominator everywhere. That is the **NOLA** condition, for nonzero overlap-add. A periodic Hann window is 0 at its first sample, so with a hop equal to $N$, every $N$-th sample has zero weight and nothing can recover it.

### COLA

A stronger property makes plain overlap-add work without any division. A window is **COLA**, for constant overlap-add, at hop $H$ if its shifted copies sum to the same value at every sample. Sample $n$ is covered by the window values at positions $n$, $n + H$, $n + 2H$, and so on, so it is enough to check that `window[r::hop].sum()` is the same for every remainder $r$ from 0 to $H - 1$.

Work it with a periodic Hann of length 4, $[0, 0.5, 1, 0.5]$. At hop 2, remainder 0 sums $0 + 1 = 1$ and remainder 1 sums $0.5 + 0.5 = 1$, so it is COLA with constant 1. At hop 3, the sums are $0 + 0.5 = 0.5$, $0.5$, and $1$, so it is not. The squared window $[0, 0.25, 1, 0.25]$ at hop 2 sums to 1 and 0.5. That isn't constant, but it is never zero, so NOLA holds and the least-squares inverse still works.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Sum of 400-sample periodic Hann windows shifted by a hop of 200 samples, which is flat, and by 160 samples, which ripples">
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-grid"/>
<text x="62.0" y="254.0" class="fig-small" text-anchor="end">0.9</text>
<line x1="70.0" y1="214.0" x2="570.0" y2="214.0" class="stroke-grid"/>
<text x="62.0" y="218.0" class="fig-small" text-anchor="end">1.0</text>
<line x1="70.0" y1="178.0" x2="570.0" y2="178.0" class="stroke-grid"/>
<text x="62.0" y="182.0" class="fig-small" text-anchor="end">1.1</text>
<line x1="70.0" y1="142.0" x2="570.0" y2="142.0" class="stroke-grid"/>
<text x="62.0" y="146.0" class="fig-small" text-anchor="end">1.2</text>
<line x1="70.0" y1="106.0" x2="570.0" y2="106.0" class="stroke-grid"/>
<text x="62.0" y="110.0" class="fig-small" text-anchor="end">1.3</text>
<line x1="70.0" y1="70.0" x2="570.0" y2="70.0" class="stroke-grid"/>
<text x="62.0" y="74.0" class="fig-small" text-anchor="end">1.4</text>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="195.0" y1="250.0" x2="195.0" y2="255.0" class="stroke-muted"/>
<text x="195.0" y="270.0" class="fig-small" text-anchor="middle">200</text>
<line x1="320.0" y1="250.0" x2="320.0" y2="255.0" class="stroke-muted"/>
<text x="320.0" y="270.0" class="fig-small" text-anchor="middle">400</text>
<line x1="445.0" y1="250.0" x2="445.0" y2="255.0" class="stroke-muted"/>
<text x="445.0" y="270.0" class="fig-small" text-anchor="middle">600</text>
<line x1="570.0" y1="250.0" x2="570.0" y2="255.0" class="stroke-muted"/>
<text x="570.0" y="270.0" class="fig-small" text-anchor="middle">800</text>
<line x1="70.0" y1="250.0" x2="570.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="70.0" class="stroke-muted"/>
<text x="320.0" y="290.0" text-anchor="middle">Sample</text>
<text x="30.0" y="56.0" text-anchor="start">Sum of shifted windows</text>
<polyline points="70.0,214.0 72.5,214.0 75.0,214.0 77.5,214.0 80.0,214.0 82.5,214.0 85.0,214.0 87.5,214.0 90.0,214.0 92.5,214.0 95.0,214.0 97.5,214.0 100.0,214.0 102.5,214.0 105.0,214.0 107.5,214.0 110.0,214.0 112.5,214.0 115.0,214.0 117.5,214.0 120.0,214.0 122.5,214.0 125.0,214.0 127.5,214.0 130.0,214.0 132.5,214.0 135.0,214.0 137.5,214.0 140.0,214.0 142.5,214.0 145.0,214.0 147.5,214.0 150.0,214.0 152.5,214.0 155.0,214.0 157.5,214.0 160.0,214.0 162.5,214.0 165.0,214.0 167.5,214.0 170.0,214.0 172.5,214.0 175.0,214.0 177.5,214.0 180.0,214.0 182.5,214.0 185.0,214.0 187.5,214.0 190.0,214.0 192.5,214.0 195.0,214.0 197.5,214.0 200.0,214.0 202.5,214.0 205.0,214.0 207.5,214.0 210.0,214.0 212.5,214.0 215.0,214.0 217.5,214.0 220.0,214.0 222.5,214.0 225.0,214.0 227.5,214.0 230.0,214.0 232.5,214.0 235.0,214.0 237.5,214.0 240.0,214.0 242.5,214.0 245.0,214.0 247.5,214.0 250.0,214.0 252.5,214.0 255.0,214.0 257.5,214.0 260.0,214.0 262.5,214.0 265.0,214.0 267.5,214.0 270.0,214.0 272.5,214.0 275.0,214.0 277.5,214.0 280.0,214.0 282.5,214.0 285.0,214.0 287.5,214.0 290.0,214.0 292.5,214.0 295.0,214.0 297.5,214.0 300.0,214.0 302.5,214.0 305.0,214.0 307.5,214.0 310.0,214.0 312.5,214.0 315.0,214.0 317.5,214.0 320.0,214.0 322.5,214.0 325.0,214.0 327.5,214.0 330.0,214.0 332.5,214.0 335.0,214.0 337.5,214.0 340.0,214.0 342.5,214.0 345.0,214.0 347.5,214.0 350.0,214.0 352.5,214.0 355.0,214.0 357.5,214.0 360.0,214.0 362.5,214.0 365.0,214.0 367.5,214.0 370.0,214.0 372.5,214.0 375.0,214.0 377.5,214.0 380.0,214.0 382.5,214.0 385.0,214.0 387.5,214.0 390.0,214.0 392.5,214.0 395.0,214.0 397.5,214.0 400.0,214.0 402.5,214.0 405.0,214.0 407.5,214.0 410.0,214.0 412.5,214.0 415.0,214.0 417.5,214.0 420.0,214.0 422.5,214.0 425.0,214.0 427.5,214.0 430.0,214.0 432.5,214.0 435.0,214.0 437.5,214.0 440.0,214.0 442.5,214.0 445.0,214.0 447.5,214.0 450.0,214.0 452.5,214.0 455.0,214.0 457.5,214.0 460.0,214.0 462.5,214.0 465.0,214.0 467.5,214.0 470.0,214.0 472.5,214.0 475.0,214.0 477.5,214.0 480.0,214.0 482.5,214.0 485.0,214.0 487.5,214.0 490.0,214.0 492.5,214.0 495.0,214.0 497.5,214.0 500.0,214.0 502.5,214.0 505.0,214.0 507.5,214.0 510.0,214.0 512.5,214.0 515.0,214.0 517.5,214.0 520.0,214.0 522.5,214.0 525.0,214.0 527.5,214.0 530.0,214.0 532.5,214.0 535.0,214.0 537.5,214.0 540.0,214.0 542.5,214.0 545.0,214.0 547.5,214.0 550.0,214.0 552.5,214.0 555.0,214.0 557.5,214.0 560.0,214.0 562.5,214.0 565.0,214.0 567.5,214.0" class="stroke-a" fill="none"/>
<polyline points="70.0,124.0 72.5,127.9 75.0,131.5 77.5,134.7 80.0,137.4 82.5,139.8 85.0,141.8 87.5,143.3 90.0,144.4 92.5,145.0 95.0,145.2 97.5,145.0 100.0,144.4 102.5,143.3 105.0,141.8 107.5,139.8 110.0,137.4 112.5,134.7 115.0,131.5 117.5,127.9 120.0,124.0 122.5,120.1 125.0,116.5 127.5,113.3 130.0,110.6 132.5,108.2 135.0,106.2 137.5,104.7 140.0,103.6 142.5,103.0 145.0,102.8 147.5,103.0 150.0,103.6 152.5,104.7 155.0,106.2 157.5,108.2 160.0,110.6 162.5,113.3 165.0,116.5 167.5,120.1 170.0,124.0 172.5,127.9 175.0,131.5 177.5,134.7 180.0,137.4 182.5,139.8 185.0,141.8 187.5,143.3 190.0,144.4 192.5,145.0 195.0,145.2 197.5,145.0 200.0,144.4 202.5,143.3 205.0,141.8 207.5,139.8 210.0,137.4 212.5,134.7 215.0,131.5 217.5,127.9 220.0,124.0 222.5,120.1 225.0,116.5 227.5,113.3 230.0,110.6 232.5,108.2 235.0,106.2 237.5,104.7 240.0,103.6 242.5,103.0 245.0,102.8 247.5,103.0 250.0,103.6 252.5,104.7 255.0,106.2 257.5,108.2 260.0,110.6 262.5,113.3 265.0,116.5 267.5,120.1 270.0,124.0 272.5,127.9 275.0,131.5 277.5,134.7 280.0,137.4 282.5,139.8 285.0,141.8 287.5,143.3 290.0,144.4 292.5,145.0 295.0,145.2 297.5,145.0 300.0,144.4 302.5,143.3 305.0,141.8 307.5,139.8 310.0,137.4 312.5,134.7 315.0,131.5 317.5,127.9 320.0,124.0 322.5,120.1 325.0,116.5 327.5,113.3 330.0,110.6 332.5,108.2 335.0,106.2 337.5,104.7 340.0,103.6 342.5,103.0 345.0,102.8 347.5,103.0 350.0,103.6 352.5,104.7 355.0,106.2 357.5,108.2 360.0,110.6 362.5,113.3 365.0,116.5 367.5,120.1 370.0,124.0 372.5,127.9 375.0,131.5 377.5,134.7 380.0,137.4 382.5,139.8 385.0,141.8 387.5,143.3 390.0,144.4 392.5,145.0 395.0,145.2 397.5,145.0 400.0,144.4 402.5,143.3 405.0,141.8 407.5,139.8 410.0,137.4 412.5,134.7 415.0,131.5 417.5,127.9 420.0,124.0 422.5,120.1 425.0,116.5 427.5,113.3 430.0,110.6 432.5,108.2 435.0,106.2 437.5,104.7 440.0,103.6 442.5,103.0 445.0,102.8 447.5,103.0 450.0,103.6 452.5,104.7 455.0,106.2 457.5,108.2 460.0,110.6 462.5,113.3 465.0,116.5 467.5,120.1 470.0,124.0 472.5,127.9 475.0,131.5 477.5,134.7 480.0,137.4 482.5,139.8 485.0,141.8 487.5,143.3 490.0,144.4 492.5,145.0 495.0,145.2 497.5,145.0 500.0,144.4 502.5,143.3 505.0,141.8 507.5,139.8 510.0,137.4 512.5,134.7 515.0,131.5 517.5,127.9 520.0,124.0 522.5,120.1 525.0,116.5 527.5,113.3 530.0,110.6 532.5,108.2 535.0,106.2 537.5,104.7 540.0,103.6 542.5,103.0 545.0,102.8 547.5,103.0 550.0,103.6 552.5,104.7 555.0,106.2 557.5,108.2 560.0,110.6 562.5,113.3 565.0,116.5 567.5,120.1" class="stroke-b" fill="none"/>
<line x1="300.0" y1="16.0" x2="325.0" y2="16.0" class="stroke-a"/>
<text x="331.0" y="20.0" class="fig-small" text-anchor="start">hop 200: constant 1.0</text>
<line x1="300.0" y1="36.0" x2="325.0" y2="36.0" class="stroke-b"/>
<text x="331.0" y="40.0" class="fig-small" text-anchor="start">hop 160: ripples between 1.19 and 1.31</text>
</svg>
Figure 3. Sums of 400-sample periodic Hann windows. Notice that hop 200 is flat, and Whisper's hop of 160 ripples. Hop 100 is also flat, at 2.0.
```

A periodic Hann of length 400 is COLA at hops 200 and 100, but not at Whisper's 160, where the sum ripples between 1.19 and 1.31. Whisper never inverts its STFT, so it doesn't care, and your least-squares inverse doesn't either, as long as NOLA holds.

> [!WARNING]
> Overlap-adding the windowed inverse frames without dividing by $\sum w^2$ leaves a ripple at the hop rate. At Whisper's settings the weights range from 0.857 to 1.018, about 1.5 dB, and you hear a faint buzz at $f_s / H$, which is 100 Hz for a hop of 160.

> [!RECAP]
> - Centered, the frame count is `1 + L // hop`; uncentered, it is `1 + (L - n_fft) // hop`.
> - Pad with `np.pad(x, n_fft // 2, mode="reflect")` and index every frame at once.
> - The STFT is `np.fft.rfft(frames * window, axis=1).T`, shaped (bins, frames); power is its absolute value squared.
> - A longer frame gives finer frequency and coarser time.
> - The least-squares inverse overlap-adds windowed inverse FFTs and divides by the overlap-added squared window. COLA means `window[r::hop].sum()` is the same for every remainder `r`.

## Build it step by step

The `hann` function is provided. The checks test the others in this order.

1. **`frame_count(n_samples, frame_length, hop_length, center)`** returns an int from the two formulas. A common mistake is `L / H`, which returns a float.
2. **`frame_signal(x, frame_length, hop_length, center, pad_mode)`** pads when centering, builds the index array, and returns `(n_frames, frame_length)`. A common mistake is counting frames after padding with the centered formula, which counts the padding twice.
3. **`stft(x, n_fft, hop_length, window, center)`** frames, windows, transforms along `axis=1`, and transposes. Use `hann(n_fft)` when `window` is `None`. A common mistake is `np.hanning(n_fft)`, which is symmetric and fails the comparison with `torch.stft`.
4. **`spectrogram(x, n_fft, hop_length, power)`** returns `np.abs(stft(...)) ** power`.
5. **`is_cola(window, hop_length)`** compares `window[r::hop_length].sum()` across remainders.
6. **`istft(X, hop_length, window, center, length)`** gets `n_fft` from `X.shape[0]`, overlap-adds the windowed `np.fft.irfft` frames and the squared window, trims `n_fft // 2` from the front, cuts to `length`, checks the weights, and divides. A common mistake is trimming `n_fft // 2` from the end too, which can leave fewer samples than `length` asks for.

## Where this breaks

**Libraries differ.** `torch.stft` reflects, and librosa pads with zeros by default, so the first and last frames differ. `torch.stft` also uses no window unless you pass `torch.hann_window(n_fft)`, and `scipy.signal.stft` divides by the window's sum. Compare against `torch.stft` or your own code.

**Uncentered frames lose the tail.** With `center=False`, the samples after the last whole frame are never analyzed, and the inverse can't return them. Pass `length=` to the inverse.

## The exercise

You build `frame_count`, `frame_signal`, `stft`, `spectrogram`, `is_cola`, and `istft`. Press **Run** to see a sweep, two close tones, and two clicks through an 8 ms window, where the clicks separate, and a 128 ms window, where the tones do. Then hear the signal rebuilt from its STFT, and see Hann windows overlap-added at a COLA hop and at Whisper's.

The checks confirm the frame counts against `torch.stft`, the worked frames, and `stft` against `torch.stft` at four settings. They check the layout, the power of `spectrogram`, and `is_cola` against `scipy.signal.check_COLA`. The last checks reconstruct a random signal to $10^{-10}$, match `torch.istft` on an STFT with scrambled phase, and expect a `ValueError` when NOLA fails.
