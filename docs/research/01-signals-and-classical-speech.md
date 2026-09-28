# Research notes 01: Signals, perception, and classical speech processing

These notes are source material for the first part of a from-scratch speech and audio ML
course. Each section gives the intuition, exact equations, a hand-checkable example,
canonical references, common mistakes, and a lab idea that can run and be auto-checked on
CPU in under 60 seconds.

- Research date: 2026-09-28.
- Every numeric example marked "(verified)" was recomputed with NumPy 2.4 and SciPy 1.17
  while preparing these notes.
- Library defaults quoted here were read from the current `main` branches of librosa,
  torchaudio, SciPy, Whisper, Silero VAD, and TorchCodec on 2026-09-28.
- Notation: $x[n]$ is a discrete signal, $f_s$ is the sample rate in Hz, $N$ is a frame or
  FFT length, $H$ is the hop length in samples, $j=\sqrt{-1}$, and $\log$ means natural log
  unless written $\log_{10}$.

---

## 0. Software environment in 2026

The course prefers NumPy and SciPy only. The following table summarizes the status of the
relevant packages as of 2026-09-28 (versions and dates from PyPI JSON):

| Package | Latest version (date) | Status and notes for the course |
|---|---|---|
| `numpy` | 2.5.3 (2026-09-06), Python >= 3.12 | Core. `np.fft` is the reference for all DFT checks. |
| `scipy` | 1.18.1 (2026-08-21), Python >= 3.12 | `scipy.signal` (windows, `resample_poly`, `lfilter`, `sosfilt`, `fftconvolve`, `check_COLA`, `check_NOLA`, `ShortTimeFFT`), `scipy.fft.dct`, `scipy.linalg.solve_toeplitz`, `scipy.io.wavfile`. `scipy.signal.stft`/`istft` are labeled "legacy"; `ShortTimeFFT` is the recommended class. |
| `librosa` | 1.0.0 (2026-08-11), Python >= 3.12 | First "stable" release; adopts Intended Effort Versioning. Expired deprecations in 1.0 include the `audioread` backend, `librosa.set_fftlib`, and `win_length` in `yin`/`pyin`. `griffinlim` now takes `rng=` instead of `random_state=`. Depends on numba, so it is not available in Pyodide. Use it as a grader-side reference, not a student dependency. |
| `torchaudio` | 2.11.0 (2026-03-23) | In maintenance mode since 2.9. Decoding and encoding APIs were deprecated in 2.8 and removed in 2.9; `torchaudio.load`/`save` are now aliases for `load_with_torchcodec`/`save_with_torchcodec`. 2.10 "marks the finalization" of the migration; `lfilter`, `RNNTLoss`, CUCTC, `forced_align`, and `overdrive` were kept. 2.11 release notes say it is compatible with torch 2.11 "and is compatible with future versions of torch," with no new features. `transforms`, `functional`, `compliance.kaldi`, `models`, and `pipelines` remain. `torchaudio.info` is gone. |
| `torchcodec` | 0.16.0 (2026-08-13) | Replacement for torchaudio I/O. Requires FFmpeg shared libraries (major versions 4-9). API: `from torchcodec.decoders import AudioDecoder`; `AudioDecoder(path, sample_rate=16000, num_channels=1).get_all_samples()` returns `AudioSamples` with `.data` (float tensor in [-1, 1], shape `(num_channels, num_samples)`), `.sample_rate`, `.pts_seconds`, `.duration_seconds`. Encoding uses `torchcodec.encoders.AudioEncoder(tensor, sample_rate=...).to_file(path)`. |
| `torch` | 2.14.0 (2026-09-02) | Needed only for GPU labs and for Whisper-style `torch.stft` parity checks. |
| `soundfile` | 0.14.0 (2026-06-06) | libsndfile wrapper; reads WAV, FLAC, OGG/Opus, MP3. Returns `float64` in [-1, 1) by default. Pyodide ships 0.12.1. |
| `silero-vad` | 6.2.3 (2026-09-23) | v6.0.0 released 2025-08-26. 6.2.3 made torchaudio optional (torchaudio or torchcodec backend). |
| `webrtcvad` | 2.0.10 (2017-01-07) | Unmaintained. The `webrtcvad-wheels` fork (2.0.14, 2024-09-05) ships prebuilt wheels. |
| `jiwer` | 4.0.0 (2025-06-19) | WER/CER reference implementation for graders. |
| `pesq` | 0.0.4 (2022-05-17) | Wraps ITU-T P.862 reference C code; stale but works. |
| `pystoi` | 0.4.1 (2023-12-29) | STOI reference. |
| `praat-parselmouth` | 0.4.7 (2025-11-27) | Praat pitch and formant references; optional. |
| `torchcrepe` | 0.0.24 (2025-05-16) | CREPE in PyTorch; optional demo. |

Browser-editor note: the Pyodide release `v314.0.7` (2026-09-14, Python 3.14.2) ships
`numpy` 2.4.6, `scipy` 1.18.0, `soundfile` 0.12.1, `soxr`, `scikit-learn`, and
`matplotlib`, but not `librosa`, `numba`, or `torch`. If student code runs in the browser,
keep student-facing labs to NumPy and SciPy, and precompute librosa, torchaudio, or Whisper
reference outputs as `.npy` fixtures on the grading side.

---

## 1. Physics of sound, decibels, and loudness

### 1.1 Pressure waves, frequency, amplitude, and phase

**Intuition.** Sound is a longitudinal pressure disturbance traveling through air. A
microphone converts the pressure deviation from ambient, $p(t)$, into a voltage; a digital
recording is a sampled version of that curve.

A pure tone is

$$x(t) = A\cos(2\pi f t + \varphi),$$

where $A$ is amplitude, $f$ is frequency in Hz (cycles per second), and $\varphi$ is phase in
radians. The period is $T = 1/f$ and the angular frequency is $\omega = 2\pi f$.

- Speed of sound in air: $c \approx 331.3 + 0.606\,\theta$ m/s, where $\theta$ is temperature
  in °C, so about 343 m/s at 20 °C.
- Wavelength: $\lambda = c/f$. At 1 kHz, $\lambda \approx 34.3$ cm; at 100 Hz, about 3.4 m.
- Human hearing spans roughly 20 Hz to 20 kHz. Classic telephony keeps about 300-3400 Hz;
  "wideband" speech is 50-7000 Hz at 16 kHz sampling.

By Fourier's theorem, any periodic signal with fundamental $f_0$ is a sum of harmonics at
$k f_0$. Speech voiced sounds are quasi-periodic, which is why harmonics appear as
horizontal striations in a narrowband spectrogram.

### 1.2 Decibels

**Intuition.** Decibels express ratios logarithmically because perceived loudness and the
dynamic range of hearing (about 120 dB) are roughly logarithmic.

- Power ratio: $L = 10\log_{10}(P/P_{\text{ref}})$ dB.
- Amplitude (field) ratio: $L = 20\log_{10}(A/A_{\text{ref}})$ dB, because $P \propto A^2$.
- Sound pressure level: $L_p = 20\log_{10}(p_{\text{rms}}/p_0)$ dB SPL with
  $p_0 = 20\ \mu\text{Pa}$. Example (verified): 1 Pa rms gives $20\log_{10}(1/2\times10^{-5}) = 93.98$ dB SPL.
- dBFS (decibels relative to full scale) is used for digital signals, where full scale is
  $|x| = 1$ for float audio or $2^{B-1}$ for $B$-bit integers. A peak of 0.5 is
  $20\log_{10}0.5 = -6.02$ dBFS (peak). AES17 defines 0 dBFS as the RMS of a full-scale
  sine, so under that RMS convention a full-scale square wave reads +3.01 dBFS. State which
  convention (peak or RMS) a lab uses.
- Doubling amplitude adds 6.02 dB; doubling power adds 3.01 dB.

**Gotchas.**

- Using $10\log_{10}$ on magnitudes or $20\log_{10}$ on power spectra doubles or halves every
  dB value. librosa encodes this in two functions: `power_to_db` ($10\log_{10}$, `amin=1e-10`)
  and `amplitude_to_db` ($20\log_{10}$, `amin=1e-5`); both default to `ref=1.0` and
  `top_db=80`.
- $\log_{10}(0)$ is $-\infty$. Always floor with a small `amin` or `eps`.

### 1.3 Loudness (LUFS), briefly

ITU-R BS.1770 (revision 5, November 2023) defines program loudness:

$$L_K = -0.691 + 10\log_{10}\sum_i G_i\, z_i \quad \text{LKFS},$$

where $z_i$ is the mean square of channel $i$ after K-weighting (a high-shelf "head" filter
followed by a high-pass RLB filter) and $G_i$ is a channel weight (1.0 for left, right, and
center; 1.41 for surround channels). Integrated loudness uses 400 ms gating blocks with 75%
overlap, an absolute gate at -70 LKFS, and a relative gate 10 LU below the absolute-gated
loudness. LUFS is the EBU R 128 synonym for LKFS; EBU R 128 targets -23 LUFS. One LU equals
one dB of broadband gain.

**Lab idea (L1-a).** Implement `db(x, kind)`, `rms_dbfs(x)`, and `peak_dbfs(x)`. The grader
checks full-scale sine → 0.00 dBFS peak and -3.01 dBFS RMS (peak-referenced RMS), a 0.5
amplitude sine → -6.02 dBFS, and rejects `-inf` on an all-zero input when `eps` is required.

---

## 2. Sampling, aliasing, and resampling

### 2.1 Nyquist-Shannon sampling theorem

**Intuition.** If a continuous signal contains no energy at or above $f_s/2$, its samples
determine it exactly, and ideal sinc interpolation reconstructs it.

Sampling: $x[n] = x_c(nT_s)$ with $T_s = 1/f_s$. The spectrum of the sampled signal is the
periodic repetition

$$X_s(f) = \frac{1}{T_s}\sum_{k=-\infty}^{\infty} X_c(f - k f_s).$$

If $X_c(f) = 0$ for $|f| \ge f_s/2$ (the Nyquist frequency), the copies do not overlap and

$$x_c(t) = \sum_{n=-\infty}^{\infty} x[n]\,\operatorname{sinc}\!\left(\frac{t - nT_s}{T_s}\right),\qquad \operatorname{sinc}(u) = \frac{\sin(\pi u)}{\pi u}.$$

### 2.2 Aliasing

A real tone at $f$ sampled at $f_s$ is indistinguishable from one at
$f_a = |f - k f_s|$ for the integer $k$ that places $f_a$ in $[0, f_s/2]$.

Example: a 7 kHz tone sampled at 10 kHz appears at $|7 - 10| = 3$ kHz. A 9 kHz tone sampled
at 16 kHz appears at 7 kHz.

**Anti-aliasing filter.** Before sampling or downsampling, low-pass filter to below the new
Nyquist frequency. A practical filter needs a transition band, so audio converters use
$f_s = 44.1$ kHz for a 20 kHz passband (a 2.05 kHz transition band).

**Gotcha.** Naive decimation `x[::3]` aliases. Students often "downsample" 48 kHz to 16 kHz
this way and then find a 10 kHz component folded to 6 kHz.

### 2.3 Resampling by a rational factor, windowed sinc, and polyphase filters

To change the rate by $L/M$:

1. Upsample by $L$ (insert $L-1$ zeros between samples); this creates spectral images.
2. Low-pass filter with cutoff $\omega_c = \min(\pi/L, \pi/M)$ and passband gain $L$.
3. Downsample by $M$ (keep every $M$-th sample).

Examples: 48 kHz to 16 kHz is $L/M = 1/3$; 16 kHz to 44.1 kHz is $441/160$.

**Windowed-sinc FIR design.** The ideal low-pass impulse response
$h_d[n] = \frac{\omega_c}{\pi}\operatorname{sinc}\!\left(\frac{\omega_c n}{\pi}\right)$
is infinite, so truncate it with a window: $h[n] = h_d[n - \tfrac{K-1}{2}]\,w[n]$ for
$n = 0,\dots,K-1$.

**Polyphase implementation.** Most products in the upsample-filter-downsample chain involve
inserted zeros or discarded outputs. Split $h$ into $L$ phases $h_p[k] = h[kL + p]$ and
compute only the outputs you keep. The cost per output sample drops from $K$ to about
$K/L$ multiplies. `scipy.signal.resample_poly` does this.

**Continuous-time (bandlimited) interpolation.** For arbitrary ratios, evaluate a windowed
sinc at fractional positions (Julius O. Smith's resampling method, used by `libsamplerate`
and `resampy`).

**Library defaults (read from source, 2026-09-28).**

| Function | Method and default |
|---|---|
| `scipy.signal.resample_poly(x, up, down)` | Polyphase FIR, `window=('kaiser', 5.0)`, half length `10 * max(up, down)`, cutoff `1/max(up, down)` of Nyquist. |
| `scipy.signal.resample` | FFT-based (assumes periodic signal, causes edge ringing on non-periodic audio). |
| `torchaudio.functional.resample` | Windowed sinc, `lowpass_filter_width=6`, `rolloff=0.99`, `resampling_method="sinc_interp_hann"` (Kaiser option). |
| `librosa.resample` | `res_type="soxr_hq"` (SoX resampler). |
| `librosa.load` | Resamples to `sr=22050` and downmixes to mono by default. Pass `sr=None` to keep the native rate. This is a very common silent bug. |

**Gotchas.**

- Different resamplers disagree at around 1e-3 relative error, so graders need a tolerance
  or a spectral check, not exact equality.
- `scipy.signal.resample` wraps the end of the signal into the start.
- Filter delay: a linear-phase FIR of length $K$ delays by $(K-1)/2$ samples; `resample_poly`
  compensates, a hand-rolled `np.convolve` does not.

**Lab idea (L1-b).** Implement `alias_frequency(f, fs)`, `naive_decimate`, and
`resample_poly_numpy(x, up, down, taps)`. Checks: (1) `alias_frequency(7000, 10000) == 3000`;
(2) a 10 kHz tone at 48 kHz decimated with the student's filter has at least 60 dB less
energy at 6 kHz than naive decimation; (3) output matches `scipy.signal.resample_poly` within
1e-6 when the student uses SciPy's own filter taps (`scipy.signal.firwin` with Kaiser 5.0).

---

## 3. Quantization, bit depth, dithering, and companding

### 3.1 Uniform quantization and the 6.02N + 1.76 dB rule

**Intuition.** Rounding to $2^N$ levels adds an error that behaves like uniform white noise
when the signal is busy. Each extra bit halves the step and gains about 6 dB.

For a quantizer with range $[-1, 1)$ and $N$ bits, the step is $\Delta = 2/2^N = 2^{1-N}$.
If the error $e$ is uniform on $[-\Delta/2, \Delta/2]$, then

$$\sigma_e^2 = \frac{\Delta^2}{12} = \frac{2^{-2N}}{3}.$$

A full-scale sine $A\sin(\cdot)$ with $A=1$ has power $1/2$, so

$$\text{SNR} = 10\log_{10}\frac{1/2}{2^{-2N}/3} = 10\log_{10}\!\left(1.5\cdot 2^{2N}\right) = 6.02N + 1.76\ \text{dB}.$$

Examples (verified): 8 bits → 49.92 dB, 16 bits → 98.08 dB, 24 bits → 146.24 dB. An
empirical 0.999-amplitude sine quantized with `np.round(s * 2**(N-1)) / 2**(N-1)` gives
49.93 dB (8 bits) and 98.09 dB (16 bits).

**Gotcha.** The rule assumes a full-scale sine. A speech signal peaking at -20 dBFS with a
crest factor near 15 dB sits about 35 dB below that SNR.

**Integer conventions.** 16-bit PCM stores signed integers in $[-32768, 32767]$. The standard
float conversion is `x_float = x_int16 / 32768.0` (Whisper's `load_audio` uses exactly this).
Unsigned 8-bit WAV is offset by 128.

### 3.2 Dither

**Intuition.** For low-level or simple signals, quantization error is correlated with the
signal and sounds like distortion. Adding a small random signal before quantizing
decorrelates the error at the cost of a little more noise.

- RPDF dither (rectangular, 1 LSB peak to peak) makes the first moment of the error
  independent of the signal.
- TPDF dither (triangular, sum of two independent RPDF values, 2 LSB peak to peak) makes the
  first and second moments independent of the signal, which removes noise modulation. The
  dither variance is $\Delta^2/6$, so total noise is $\Delta^2/12 + \Delta^2/6 = \Delta^2/4$,
  a 4.77 dB SNR penalty.
- Noise shaping moves the requantization noise toward frequencies where hearing is less
  sensitive.
- Kaldi's `compute-fbank-feats` defaults to `dither=1.0` (in 16-bit integer units) to avoid
  $\log 0$ on digital silence; `torchaudio.compliance.kaldi.fbank` defaults to `dither=0.0`.

Reference: Lipshitz, Wannamaker, and Vanderkooy, "Quantization and dither: A theoretical
survey," *J. Audio Eng. Soc.* 40(5), 1992.

### 3.3 Companding: mu-law and A-law

**Intuition.** Speech has most samples near zero. A logarithmic compressor before a uniform
quantizer spends more levels on quiet samples, which gives roughly constant relative
(percentage) error.

mu-law (North America and Japan, ITU-T G.711), for $x \in [-1, 1]$ and $\mu = 255$:

$$F(x) = \operatorname{sgn}(x)\frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)},\qquad F^{-1}(y) = \operatorname{sgn}(y)\frac{(1+\mu)^{|y|} - 1}{\mu}.$$

A-law (Europe, G.711), with $A = 87.6$:

$$F(x) = \operatorname{sgn}(x)\begin{cases}\dfrac{A|x|}{1 + \ln A}, & |x| < \dfrac{1}{A}\\[2mm] \dfrac{1 + \ln(A|x|)}{1 + \ln A}, & \dfrac{1}{A} \le |x| \le 1.\end{cases}$$

G.711 itself uses a segmented, piecewise-linear 8-bit approximation of these curves at
8 kHz (64 kb/s). The smooth formula is what ML code implements.

**Why WaveNet used 8-bit mu-law.** WaveNet (van den Oord et al., 2016, arXiv:1609.03499,
Section 2.2) models $p(x_t \mid x_{<t})$ with a softmax. With 16-bit audio this would need
65,536 outputs per step, so the authors "apply a µ-law companding transformation ... and then
quantize it to 256 possible values," with $\mu = 255$. They report that this "non-linear
quantization produces a significantly better reconstruction than a simple linear
quantization scheme."

Worked example (verified), mapping $F(x)$ to integers with
$q = \lfloor (F(x) + 1)/2 \cdot 255 + 0.5 \rfloor$:

| $x$ | $F(x)$ | code $q$ |
|---|---|---|
| 0.001 | 0.0410 | 133 |
| 0.01 | 0.2285 | 157 |
| 0.1 | 0.5910 | 203 |
| 0.5 | 0.8757 | 239 |
| 1.0 | 1.0 | 255 |

The range $0 \le x \le 0.01$ (1% of amplitude range) uses codes 128-157, about 12% of all
codes.

**Gotchas.** Clip to $[-1, 1]$ before companding. Several libraries map codes differently
(`floor` vs `round`, 0-255 vs -128-127); `torchaudio.functional.mu_law_encoding` uses
`((x_mu + 1) / 2 * mu + 0.5).to(int64)`.

**Lab idea (L1-c).** Implement `quantize(x, bits)`, `tpdf_dither(x, bits, rng)`,
`mu_law_encode(x, mu=255)`, and `mu_law_decode(q, mu=255)`. Checks: measured SNR of a
full-scale sine within 0.1 dB of $6.02N + 1.76$ for $N \in \{8, 12, 16\}$; encode-decode
round trip error below $\Delta_\mu$ bounds; on a seeded synthetic speech-like signal
(Laplacian samples), 8-bit mu-law SNR exceeds 8-bit linear SNR by at least 10 dB.

---

## 4. Audio file formats and channels

- **WAV (RIFF/WAVE).** A container with a `fmt ` chunk (format tag 1 = integer PCM,
  3 = IEEE float, `0xFFFE` = WAVE_FORMAT_EXTENSIBLE; channels; sample rate; byte rate;
  block align; bits per sample) and a `data` chunk. All fields are little-endian. Samples are
  interleaved by channel: `L0 R0 L1 R1 ...`. The canonical PCM header is 44 bytes, but files
  often contain extra chunks (`LIST`, `fact`), so parse chunks rather than seeking to byte 44.
- **FLAC.** Lossless. Each block is predicted with fixed or LPC predictors of order up to 32,
  and the residual is Rice coded. Supports 1-8 channels, 4-32 bits, and sample rates up to
  1,048,575 Hz. Standardized as IETF RFC 9639 (December 2024).
- **MP3 (MPEG-1/2 Audio Layer III, ISO/IEC 11172-3).** Lossy perceptual coder with a hybrid
  polyphase-plus-MDCT filterbank; 1152 samples per frame; a psychoacoustic model sets
  quantization so noise stays below the masking threshold. Encoders add a delay (padding) at
  the start, so decoded length and alignment differ from the source.
- **Opus (IETF RFC 6716, 2012).** Combines SILK (linear prediction, speech) and CELT (MDCT,
  music) layers. It "scales from 6 kbit/s narrowband mono speech to 510 kbit/s fullband
  stereo music." Internal bandwidths run from narrowband (8 kHz effective) to fullband
  (48 kHz effective); Opus never codes above 20 kHz.
- **Channels.** Downmix to mono with the channel mean (`x.mean(axis=0)`), not the sum, to
  avoid clipping. Be explicit about layout: `soundfile` returns `(frames, channels)`;
  TorchCodec and torchaudio return `(channels, frames)`.

**Gotchas.**

- `scipy.io.wavfile.read` returns raw integers (`int16`, `int32`, or `uint8`) without
  scaling; `soundfile.read` returns `float64` scaled to [-1, 1).
- 24-bit WAV is often read as `int32` left-justified, so the scale factor is $2^{31}$, not
  $2^{23}$.
- Lossy decoders pad and delay. Do not compute sample-aligned metrics between an MP3 decode
  and the original without alignment.

**Lab idea (L1-d).** Implement `write_wav_pcm16(path, x, sr)` and `read_wav(path)` with
`struct` and NumPy only, including chunk skipping. The grader round-trips a seeded
stereo signal and compares with `scipy.io.wavfile.read` and `soundfile.read`, and feeds a
file with an extra `LIST` chunk.

---

## 5. The DFT and FFT, spectral leakage, windows, and zero-padding

### 5.1 DFT definition

$$X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-j2\pi kn/N},\qquad x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{j2\pi kn/N},$$

for $k = 0,\dots,N-1$. Bin $k$ corresponds to frequency $f_k = k f_s/N$. For real $x$,
$X[N-k] = X[k]^*$, so only $N/2 + 1$ bins are unique (`np.fft.rfft`). Parseval's theorem:
$\sum_n |x[n]|^2 = \frac{1}{N}\sum_k |X[k]|^2$.

**Worked 4-point DFT (verified).** For $x = [1, 2, 0, -1]$, the twiddles are
$e^{-j\pi kn/2} \in \{1, -j, -1, j\}$:

- $X[0] = 1 + 2 + 0 - 1 = 2$
- $X[1] = 1 + 2(-j) + 0(-1) + (-1)(j) = 1 - 3j$
- $X[2] = 1 - 2 + 0 + 1 = 0$
- $X[3] = X[1]^* = 1 + 3j$

Parseval check: $\sum|x|^2 = 6$, and $\frac{1}{4}(4 + 10 + 0 + 10) = 6$.

A second example: $x = [1, 0, -1, 0]$ (a cosine at bin 1) gives $X = [0, 2, 0, 2]$. The
amplitude-1 cosine appears as $N/2 = 2$ in each of bins $\pm 1$.

**FFT.** Radix-2 Cooley-Tukey splits even and odd indices:
$X[k] = E[k] + e^{-j2\pi k/N} O[k]$ and $X[k + N/2] = E[k] - e^{-j2\pi k/N} O[k]$, which
gives $O(N\log N)$ cost.

### 5.2 Spectral leakage and windows

**Intuition.** The DFT treats the frame as one period of a periodic signal. If the frame
does not hold an integer number of cycles, the wrap-around discontinuity spreads energy into
all bins (leakage). Tapering the frame with a window trades a wider main lobe for lower side
lobes.

Symmetric windows of length $N$, $n = 0,\dots,N-1$ (periodic, "DFT-even" versions replace
$N-1$ with $N$):

- Rectangular: $w[n] = 1$.
- Hann: $w[n] = 0.5 - 0.5\cos\!\left(\frac{2\pi n}{N-1}\right)$.
- Hamming: $w[n] = 0.54 - 0.46\cos\!\left(\frac{2\pi n}{N-1}\right)$.
- Blackman: $w[n] = 0.42 - 0.5\cos\!\left(\frac{2\pi n}{N-1}\right) + 0.08\cos\!\left(\frac{4\pi n}{N-1}\right)$.
- Kaiser: $w[n] = \dfrac{I_0\!\left(\beta\sqrt{1 - \left(\frac{2n}{N-1} - 1\right)^2}\right)}{I_0(\beta)}$, where $I_0$ is the zeroth-order modified Bessel function.

Kaiser design formulas (Oppenheim and Schafer), for stopband attenuation $A$ dB and transition
width $\Delta\omega$ rad/sample:

$$\beta = \begin{cases}0.1102(A - 8.7), & A > 50\\ 0.5842(A - 21)^{0.4} + 0.07886(A - 21), & 21 \le A \le 50\\ 0, & A < 21\end{cases}\qquad N - 1 \approx \frac{A - 8}{2.285\,\Delta\omega}.$$

`scipy.signal.kaiserord` implements these.

Measured properties (verified with $N = 64$ symmetric windows and 64x zero-padding):

| Window | Main-lobe half-width (first null, bins) | Peak side lobe | Side-lobe roll-off |
|---|---|---|---|
| Rectangular | 1 | -13.3 dB | -6 dB/octave |
| Hann | 2 | -31.5 dB | -18 dB/octave |
| Hamming | about 2 | -42.4 dB (about -43 dB for large $N$) | -6 dB/octave |
| Blackman | 3 | -58.1 dB | -18 dB/octave |

Roll-off values are from Harris (1978).

**Amplitude estimation.** A sinusoid of amplitude $A$ centered on bin $k$ has
$|X[k]| = A\sum_n w[n]/2$. For a rectangular window that is $AN/2$; for Hann it is about
$AN/4$. Normalize by `w.sum()` (coherent gain) to read amplitudes.

### 5.3 Zero-padding

Zero-padding a length-$N$ frame to $M > N$ samples interpolates the same underlying spectrum
(the DTFT of the windowed frame) on a finer grid ($f_s/M$ spacing). It does not improve
resolution: two tones closer than the main-lobe width stay merged. Resolution is set by the
window length $N$, not the FFT length.

### 5.4 Time-frequency tradeoff

With frame length $N$, bin spacing is $f_s/N$ and time support is $N/f_s$. At 16 kHz,
$N = 400$ gives 25 ms frames and 40 Hz bins. The Gabor limit
$\sigma_t\sigma_f \ge \frac{1}{4\pi}$ (standard deviations of $|x|^2$ and $|X|^2$) is met with
equality by Gaussians. A wideband spectrogram (about 5 ms window) resolves glottal pulses
and formants; a narrowband spectrogram (about 30-50 ms) resolves harmonics.

**Gotchas.**

- Mixing up `np.hanning(N)` (symmetric) with `scipy.signal.get_window('hann', N)` (periodic
  by default, `fftbins=True`) and `torch.hann_window(N)` (periodic by default). Periodic
  windows are the right choice for STFT and COLA.
- Plotting `abs(X)` over all $N$ bins shows the mirror image; use `rfft`.
- Forgetting that `np.fft` is unnormalized: the forward transform has no $1/N$.

**Lab idea (L2-a).** Implement `dft_matrix(N)`, `dft(x)` via the matrix, a recursive
`fft_radix2(x)`, and `hann(N, periodic)`, `hamming`, `blackman`, and `kaiser(N, beta)` from
their formulas (with `np.i0`). Checks: `allclose` against `np.fft.fft` on seeded random
inputs of size 1-1024; windows against `scipy.signal.get_window`; peak side lobe of the
student's Hann within 0.5 dB of -31.5 dB; a two-tone resolution test showing that
zero-padding does not separate tones 20 Hz apart with a 25 ms window.

---

## 6. STFT, COLA and NOLA, ISTFT, and frame counting

### 6.1 Definition

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH]\,w[n]\,e^{-j2\pi kn/N_{\text{fft}}},$$

where $m$ is the frame index, $H$ is the hop, $w$ is the analysis window of length $N$
(`win_length`), and $N_{\text{fft}} \ge N$. The number of frequency bins is
$N_{\text{fft}}/2 + 1$ (201 for `n_fft=400`). The spectrogram is $|X|$ (magnitude) or
$|X|^2$ (power).

### 6.2 Frame counts and centering

- `center=False` (Kaldi `snip_edges=True`): $M = 1 + \lfloor (L - N)/H \rfloor$ frames for a
  signal of $L$ samples.
- `center=True` (librosa, `torch.stft` default): the signal is padded by $N_{\text{fft}}/2$ on
  both sides, so frame $m$ is centered on sample $mH$ and $M = 1 + \lfloor L/H \rfloor$.
- Example (verified): 1 s at 16 kHz with $N = 400$, $H = 160$ gives 101 frames centered and
  98 frames uncentered.
- Pad modes: librosa uses `pad_mode="constant"` (zeros); `torch.stft` and
  `torchaudio.transforms.Spectrogram` use `"reflect"`. The first and last frames differ.
- Whisper: 30 s × 16 kHz = 480,000 samples, centered STFT gives 3001 frames, and Whisper drops
  the last one (`stft[..., :-1]`) to get exactly 3000.

### 6.3 Reconstruction: overlap-add, COLA, and NOLA

**Weighted overlap-add ISTFT (least squares; Griffin and Lim, 1984).**

$$\hat{x}[n] = \frac{\sum_m w[n - mH]\; y_m[n - mH]}{\sum_m w^2[n - mH]},$$

where $y_m = \text{IDFT}(X[m, \cdot])$ truncated to $N$ samples. This is the least-squares
signal whose STFT is closest to $X$, and it is exact when $X$ is a valid STFT.

- **NOLA** (nonzero overlap-add): $\sum_m w^2[n - mH] > 0$ for all $n$. Required for the
  division; `scipy.signal.check_NOLA`.
- **COLA** (constant overlap-add): $\sum_m w[n - mH] = C$ for all $n$. With COLA you can
  reconstruct with a plain overlap-add of unwindowed synthesis frames (divide by $C$).
  `scipy.signal.check_COLA`.
- Verified: periodic Hann of length 400 is COLA at hops 200 and 100 but not at 150;
  periodic Hamming is COLA at hop 200; the symmetric Hann of length 400 is not exactly COLA
  at hop 200.

**Gotchas.**

- Reconstructing without dividing by $\sum w^2$ causes amplitude ripple at the hop rate.
- Losing the last partial frame when `center=False`; pass `length=` to the ISTFT to trim.
- `torch.stft` requires `return_complex=True` and defaults to a rectangular window if
  `window=None`.
- `scipy.signal.stft` defaults to `scaling='spectrum'`, which divides by `win.sum()`, so its
  magnitudes differ from `np.fft.rfft` of the same frame.

**Lab idea (L2-b).** Implement `frame(x, N, H, center, pad_mode)`, `stft(...)`, and
`istft(...)` with the least-squares normalization. Checks: frame count for 20 seeded lengths
in both modes; STFT equal to `librosa.stft` fixtures (grader-side) within 1e-5; perfect
reconstruction error below 1e-10 for Hann at $H = N/4$; a test that a non-NOLA configuration
raises `ValueError`.

---

## 7. Phase, Griffin-Lim, and the phase vocoder

### 7.1 Why phase matters

Magnitude spectrograms discard phase. Phase is hard to model directly because it wraps and
depends on the exact frame position, yet a random phase produces a rough, phasey sound.
Neural vocoders (later in the course) exist largely to solve this.

**Spectral consistency.** Not every complex array is the STFT of some signal. Overlapping
frames impose redundancy; the set of consistent STFTs is the image of the STFT operator.

### 7.2 Griffin-Lim algorithm (GLA)

Given a target magnitude $S = |Y|$, alternate projections onto two sets: $\mathcal{C}_1$
(consistent spectrograms) and $\mathcal{C}_2$ (spectrograms with magnitude $S$).

$$P_{\mathcal{C}_2}(X) = S \odot \frac{X}{|X|},\qquad P_{\mathcal{C}_1}(X) = \text{STFT}(\text{ISTFT}(X)).$$

Update, starting from $X_0 = S\,e^{j\phi_0}$ (random or zero phase):

$$x_{i+1} = \text{ISTFT}\!\left(S \odot e^{j\angle X_i}\right),\qquad X_{i+1} = \text{STFT}(x_{i+1}).$$

With the least-squares ISTFT of Section 6.3, Griffin and Lim proved the squared distance
$\| |X_i| - S \|$-type objective is non-increasing. Track the spectral convergence

$$\text{SC}_i = \frac{\big\| |X_i| - S \big\|_F}{\|S\|_F}.$$

**Fast Griffin-Lim** (Perraudin, Balazs, and Søndergaard, WASPAA 2013) adds momentum:

$$t_n = P_{\mathcal{C}_1}(P_{\mathcal{C}_2}(c_{n-1})),\qquad c_n = t_n + \alpha(t_n - t_{n-1}),$$

with $\alpha$ near 0.99. librosa's `griffinlim` (defaults `n_iter=32`, `momentum=0.99`,
`init="random"`) implements the equivalent normalized form
`angles = rebuilt - (momentum / (1 + momentum)) * tprev`, then `angles /= abs(angles) + eps`.
torchaudio's `GriffinLim` defaults to `n_iter=32`, `momentum=0.99`, `rand_init=True`.

**Gotchas.** Feeding a power spectrogram as a magnitude; forgetting to use the same window
and hop in STFT and ISTFT; comparing waveforms sample-by-sample (GLA recovers the signal only
up to sign and small time shifts, so compare spectral convergence instead).

### 7.3 Phase vocoder and time stretching

**Intuition.** To slow down speech without changing pitch, analyze with hop $H_a$ and
resynthesize with hop $H_s = rH_a$, advancing each bin's phase according to its measured
instantaneous frequency.

For bin $k$ with center $\omega_k = 2\pi k/N_{\text{fft}}$ and analysis phases $\phi_m[k]$:

$$\Delta\phi = \phi_m[k] - \phi_{m-1}[k] - \omega_k H_a,\qquad \hat\omega_k = \omega_k + \frac{\operatorname{princarg}(\Delta\phi)}{H_a},$$

$$\psi_m[k] = \psi_{m-1}[k] + \hat\omega_k H_s,\qquad Y[m, k] = |X[m, k]|\,e^{j\psi_m[k]},$$

where $\operatorname{princarg}$ wraps to $(-\pi, \pi]$. The result has duration scaled by
$H_s/H_a$. Pitch shifting is time stretching followed by resampling. Artifacts ("phasiness")
come from losing vertical phase coherence across bins; Laroche and Dolson (1999) introduced
peak-based phase locking. librosa 1.0 adds variable-rate phase vocoding, and its
`phase_vocoder` now ignores `hop_length` and `n_fft` (to be removed in 1.1).

**Lab idea (L2-c).** Implement `griffin_lim(S, n_iter, momentum, rng)` and
`time_stretch(x, rate)`. Checks: on a seeded chirp, spectral convergence after 50 iterations
is below 0.1 and monotone non-increasing for `momentum=0`; FGLA reaches a lower SC than GLA at
equal iterations; time-stretching a 440 Hz tone by 1.5 keeps the spectral peak within one bin
of 440 Hz and scales the duration within one hop.

---

## 8. Filters, convolution, impulse responses, reverb, and pre-emphasis

### 8.1 LTI systems and convolution

A linear time-invariant system is fully described by its impulse response $h[n]$:

$$y[n] = (x * h)[n] = \sum_k h[k]\,x[n-k].$$

Linear convolution of lengths $L_x$ and $L_h$ has length $L_x + L_h - 1$. By the convolution
theorem, $\text{DFT}(x * h) = X \cdot H$ when both are zero-padded to at least
$L_x + L_h - 1$; otherwise the product gives circular convolution (a classic bug). For long
signals, use overlap-add or overlap-save block convolution (`scipy.signal.oaconvolve`,
`fftconvolve`).

### 8.2 FIR and IIR filters

Difference equation, with $a_0 = 1$:

$$y[n] = \sum_{k=0}^{M} b_k\,x[n-k] - \sum_{k=1}^{P} a_k\,y[n-k],\qquad H(z) = \frac{\sum_k b_k z^{-k}}{1 + \sum_k a_k z^{-k}}.$$

- **FIR** ($P = 0$): always stable; exactly linear phase when $b$ is symmetric, with group
  delay $(M)/2$ samples for $M + 1$ taps. Design with windowed sinc (`firwin`) or
  Parks-McClellan (`remez`).
- **IIR**: fewer coefficients for sharp responses, stable only if all poles lie inside the
  unit circle, nonlinear phase. Butterworth, Chebyshev, elliptic designs (`butter`,
  `cheby1`, `ellip`). Use second-order sections (`output='sos'`, `sosfilt`) for high orders
  to avoid coefficient-precision instability.
- `filtfilt` and `sosfiltfilt` run forward and backward for zero phase (non-causal).

### 8.3 Room impulse responses and convolution reverb

Reverberant speech is modeled as $y = x * h_{\text{room}} + \text{noise}$. The reverberation
time $\text{RT}_{60}$ is the time for energy to decay by 60 dB. Sabine's formula (metric):
$\text{RT}_{60} \approx 0.161\,V/A$, where $V$ is volume in m³ and $A = \sum_i S_i\alpha_i$ is
absorption area in m² (Sabine).

A synthetic RIR for labs: $h[n] = \nu[n]\,e^{-6.908\,n/(f_s \text{RT}_{60})}$ with white noise
$\nu$, because a 60 dB energy decay is an amplitude factor of $10^{-3} = e^{-6.908}$. Prepend
a direct-path impulse. Real RIRs can be simulated with the image-source method (Allen and
Berkley, 1979; `pyroomacoustics`). RIR convolution is a standard data augmentation.

### 8.4 Pre-emphasis

$$y[n] = x[n] - \alpha\,x[n-1],\qquad |H(e^{j\omega})|^2 = 1 + \alpha^2 - 2\alpha\cos\omega.$$

Voiced speech has a spectral tilt of about -6 dB/octave (glottal source about -12 dB/octave,
lip radiation about +6 dB/octave). Pre-emphasis with $\alpha = 0.97$ (Kaldi default
`preemphasis_coefficient=0.97`) flattens it, which improves LPC conditioning. Verified gains:
+5.89 dB at Nyquist and -30.46 dB at DC. Modern log-mel front ends (Whisper, most
torchaudio pipelines) skip pre-emphasis.

**Lab idea (L3-a).** Implement `convolve_direct`, `convolve_fft` (with correct zero-padding),
`lfilter_numpy(b, a, x)`, `preemphasis`, and `synthetic_rir(rt60, fs, rng)`. Checks:
agreement with `np.convolve` and `scipy.signal.lfilter` to 1e-10; a test catching circular
wrap-around; the measured RT60 of the student's RIR by Schroeder backward integration within
10% of the target.

---

## 9. Human auditory perception

**Cochlea as a filterbank.** The basilar membrane is tonotopic: high frequencies peak near the
base (oval window) and low frequencies near the apex. Each place acts as a band-pass filter
whose bandwidth grows with center frequency, roughly constant below about 500 Hz and roughly
proportional above. Inner hair cells half-wave rectify and compress.

**Critical bands.** Tones within one critical band interact strongly (masking, loudness
summation). About 24 critical bands span the audible range (the Bark scale). Bandwidth is
about 100 Hz below 500 Hz and about 20% of center frequency above.

**Masking.**

- Simultaneous masking: a loud component raises the detection threshold of nearby
  frequencies, asymmetrically (upward spread of masking toward higher frequencies).
- Temporal masking: pre-masking extends about 5-20 ms before a masker onset, and
  post-masking about 100-200 ms after offset.

**Absolute threshold of hearing** (Terhardt 1979 approximation used in codec literature,
for example Painter and Spanias 2000), with $f$ in Hz:

$$T_q(f) = 3.64\left(\tfrac{f}{1000}\right)^{-0.8} - 6.5\,e^{-0.6\left(\frac{f}{1000} - 3.3\right)^2} + 10^{-3}\left(\tfrac{f}{1000}\right)^4\ \text{dB SPL}.$$

**Equal-loudness contours.** The phon level of a tone is the SPL of an equally loud 1 kHz
tone. Hearing is most sensitive near 3-4 kHz and much less sensitive at low frequencies.
Fletcher and Munson (1933) measured the first contours; the current standard is ISO 226:2023,
whose differences from ISO 226:2003 are at most 0.6 dB. A-weighting approximates the inverse
of the 40-phon contour. Loudness in sones roughly doubles per 10 phon above 40 phon
(1 sone = 40 phon).

**Why this matters.**

- Features: mel and Bark spacing approximate critical-band spacing; log compression
  approximates loudness compression; features discard phase because the ear is relatively
  insensitive to the phase of stationary components.
- Codecs: MP3, AAC, and Opus allocate bits so that quantization noise stays under the masking
  threshold computed from a psychoacoustic model.
- Metrics: PESQ and STOI embed auditory models (Section 22).

**Lab idea (L3-b).** Implement `threshold_in_quiet(f)` and a simple two-tone masking
demonstration: generate a 1 kHz masker at 60 dB and probe tones, and compute which probes fall
below a triangular spreading function in the Bark domain. Auto-check the threshold function
against tabulated values and the spreading function against a reference array. Perception
itself cannot be auto-graded, so keep listening optional.

---

## 10. Perceptual frequency scales and auditory filterbanks

### 10.1 Mel scale: HTK versus Slaney

The mel scale (Stevens, Volkmann, and Newman, 1937) is a pitch-perception scale. Two
formulas dominate ML code, and they are not interchangeable.

**HTK (O'Shaughnessy 1987) formula**, used by HTK, Kaldi, `torchaudio` (default
`mel_scale="htk"`), and librosa with `htk=True`:

$$m_{\text{HTK}}(f) = 2595\log_{10}\!\left(1 + \frac{f}{700}\right) = 1127\ln\!\left(1 + \frac{f}{700}\right),\qquad f = 700\left(10^{m/2595} - 1\right).$$

(The constants agree to rounding: $1127\ln 10 \approx 2595.0$. Kaldi and SLP3 write
$1127\ln(1 + f/700)$.)

**Slaney (Auditory Toolbox, 1998) formula**, the librosa default (`htk=False`) and therefore
the Whisper default. It is linear below 1 kHz and logarithmic above:

$$m_{\text{Slaney}}(f) = \begin{cases}\dfrac{f}{200/3}, & f < 1000\\[2mm] 15 + \dfrac{\ln(f/1000)}{\ln(6.4)/27}, & f \ge 1000,\end{cases}$$

with inverse $f = \frac{200}{3}m$ for $m < 15$ and $f = 1000\,e^{(m - 15)\ln(6.4)/27}$
otherwise. The log step is $\ln(6.4)/27 = 0.068752$. (This is the exact librosa
`hz_to_mel` source: `f_sp = 200.0 / 3`, `min_log_hz = 1000.0`, `logstep = np.log(6.4) / 27.0`.)

Worked values (verified):

| $f$ (Hz) | HTK mel | Slaney mel |
|---|---|---|
| 100 | 150.49 | 1.50 |
| 440 | 549.64 | 6.60 |
| 1000 | 999.99 | 15.00 |
| 2000 | 1521.36 | 25.08 |
| 4000 | 2146.06 | 35.16 |
| 8000 | 2840.02 | 45.25 |

HTK maps 1000 Hz to about 1000 mel by construction; Slaney maps it to exactly 15. Only
relative spacing matters for filterbanks, but the two scales place centers differently
below 1 kHz.

### 10.2 Triangular mel filterbank

Choose $n_{\text{mels}} + 2$ edge points equally spaced in mel between $f_{\min}$ and $f_{\max}$
and convert them to Hz: $f_0 < f_1 < \dots < f_{n+1}$. Filter $i$ (for
$i = 0,\dots,n_{\text{mels}}-1$) evaluated at FFT bin frequency $f$:

$$W_i(f) = \max\!\left(0, \min\!\left(\frac{f - f_i}{f_{i+1} - f_i}, \frac{f_{i+2} - f}{f_{i+2} - f_{i+1}}\right)\right).$$

librosa builds the triangles in Hz (linear ramps between mel-spaced Hz edges), not in the
mel domain. Normalization options:

- `norm="slaney"` (librosa default): multiply filter $i$ by $2/(f_{i+2} - f_i)$ so each
  filter has roughly constant area (energy).
- `norm=None` (torchaudio default): peak height 1.

Whisper-specific fact (verified): at $f_s = 16$ kHz and `n_fft=400` (40 Hz bins), the lowest
Slaney mel spacing is 37.2 Hz for 80 mels and 23.4 Hz for 128 mels. With 128 mels, the first
several filters each touch a single FFT bin. No filter is empty, so librosa does not warn,
but low filters are effectively single-bin samples.

**Gotchas.** Mixing HTK centers with Slaney normalization (or the reverse) when trying to
match a reference; forgetting `fmax=sr/2`; applying the filterbank to magnitude when the
reference used power (`power=2.0`).

### 10.3 Bark, ERB, and gammatone

- **Bark** (Zwicker 1961). Zwicker and Terhardt (1980):
  $z = 13\arctan(0.00076f) + 3.5\arctan\!\left((f/7500)^2\right)$. Traunmüller (1990):
  $z = \frac{26.81 f}{1960 + f} - 0.53$. At 1 kHz (verified): 8.51 and 8.53 Bark.
- **ERB** (equivalent rectangular bandwidth; Glasberg and Moore 1990):
  $\text{ERB}(f) = 24.7\left(\frac{4.37f}{1000} + 1\right)$ Hz and ERB-rate
  $E(f) = 21.4\log_{10}\!\left(\frac{4.37f}{1000} + 1\right)$. At 1 kHz (verified):
  ERB = 132.6 Hz and ERB-rate = 15.62.
- **Gammatone filter** (Patterson et al.; Slaney 1993 efficient implementation):
  $g(t) = a\,t^{n-1}e^{-2\pi b\,\text{ERB}(f_c)\,t}\cos(2\pi f_c t + \varphi)$, with order
  $n = 4$ and $b \approx 1.019$. Center frequencies are spaced uniformly on the ERB-rate scale.
  Gammatone filterbanks underlie auditory models and some features (gammatone cepstral
  coefficients). `scipy.signal.gammatone` designs FIR or IIR gammatone filters.
- **PLP** (Hermansky 1990) combines Bark filtering, equal-loudness pre-emphasis, cube-root
  intensity-loudness compression, and an all-pole model.

**Lab idea (L4-a).** Implement `hz_to_mel(f, htk)`, `mel_to_hz(m, htk)`, and
`mel_filterbank(sr, n_fft, n_mels, fmin, fmax, htk, norm)`. Checks: the table of worked
values to 1e-2; exact inversion; filterbank equal to a stored
`librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)` fixture within 1e-6 for both norm
options and both scales; also equal to `torchaudio.functional.melscale_fbanks` fixtures
(transposed; torchaudio returns `(n_freqs, n_mels)`).

---

## 11. Log-mel spectrograms: exact conventions

The following table lists the defaults (read from source on 2026-09-28):

| Setting | librosa `feature.melspectrogram` | torchaudio `transforms.MelSpectrogram` | Kaldi `fbank` (`torchaudio.compliance.kaldi.fbank`) | Whisper `log_mel_spectrogram` |
|---|---|---|---|---|
| Sample rate | `sr=22050` | `sample_rate=16000` | `sample_frequency=16000` | 16000 (fixed) |
| FFT size | `n_fft=2048` | `n_fft=400` | 25 ms frame (400 samples) rounded to 512 (`round_to_power_of_two=True`) | `N_FFT=400` |
| Hop | `hop_length=512` | `win_length // 2` (200) | 10 ms (160) | `HOP_LENGTH=160` |
| Window | Hann, periodic | `torch.hann_window` (periodic) | `povey` = symmetric Hann$^{0.85}$ | `torch.hann_window(400)` (periodic) |
| Centering | `center=True`, zero pad | `center=True`, reflect pad | none (`snip_edges=True`) | `torch.stft` default: `center=True`, reflect pad; last frame dropped |
| Spectrum | power (`power=2.0`) | power (`power=2.0`) | power (`use_power=True`) | power (`abs() ** 2`) |
| Mel scale | Slaney (`htk=False`) | HTK (`mel_scale="htk"`) | HTK (1127 ln) | Slaney (librosa filters) |
| Filter norm | `"slaney"` | `None` | none | `"slaney"` |
| Mels | 128 | 128 | 23 (`num_mel_bins`) | 80, or 128 for large-v3 and large-v3-turbo |
| Freq range | 0 to sr/2 | 0 to sr/2 | 20 Hz to Nyquist | 0 to 8000 Hz |
| Other | none | none | DC removal, pre-emphasis 0.97, `dither=0.0` in torchaudio | none |
| Log | none (use `power_to_db`: $10\log_{10}$, `amin=1e-10`, `top_db=80`) | none (use `AmplitudeToDB`) | natural log with float epsilon floor | $\log_{10}$ with clamp `1e-10` |

**Whisper exact pipeline** (from `whisper/audio.py`):

```python
window = torch.hann_window(N_FFT).to(audio.device)
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2
filters = mel_filters(audio.device, n_mels)   # librosa.filters.mel(sr=16000, n_fft=400, n_mels=80|128)
mel_spec = filters @ magnitudes
log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
log_spec = (log_spec + 4.0) / 4.0
```

Interpretation:

- $L = \log_{10}\max(M, 10^{-10})$, where $M$ is the mel power.
- Clip to at most 8 decades (80 dB) below the global maximum of the whole input (not per
  frame). In `transcribe`, the spectrogram is computed once over the full padded file, so the
  maximum is file-global.
- Rescale with $(L + 4)/4$, so values land roughly in $[-1, 1.5]$.
- Constants: `CHUNK_LENGTH=30`, `N_SAMPLES=480000`, `N_FRAMES=3000`, 100 frames per second
  (10 ms), and the encoder's stride-2 convolution gives 50 tokens per second (20 ms).
- Audio loading shells out to the `ffmpeg` CLI, requests `s16le` mono at 16 kHz, and divides
  by 32768.0.
- Large-v3 (released November 2023) changed only two things relative to large-v2: 128 mel
  bins instead of 80, and a new Cantonese language token.

**Kaldi-scale gotcha.** Kaldi recipes feed waveforms on the 16-bit integer scale. If you pass
floats in [-1, 1] to `kaldi.fbank`, every natural-log mel value shifts by
$\ln(32768^2) = 20.79$.

**Common student mistakes.**

- Magnitude versus power (factor of 2 in dB, different mel weighting).
- $\ln$ versus $\log_{10}$ (factor 2.303).
- Per-frame instead of global max in the Whisper clamp.
- Off-by-one frames (3001 versus 3000) from not dropping the last frame.
- Reflect versus zero padding at the edges.
- `librosa.load` silently resampling to 22,050 Hz.

**Lab idea (L4-b).** Implement `whisper_log_mel(audio_np, n_mels)` in NumPy only (framing,
reflect padding, periodic Hann, `rfft`, power, provided Slaney filterbank, log10 clamp, max-8,
and rescale). The grader compares against stored `whisper.audio.log_mel_spectrogram` outputs
for three seeded signals (tolerance 1e-4, float32), checks shape `(80, 3000)` after
`pad_or_trim`, and has a separate test for `n_mels=128`.

---

## 12. MFCCs, deltas, liftering, and CMVN

### 12.1 MFCC via DCT-II

**Intuition.** Log-mel energies of neighboring bands are highly correlated. A DCT decorrelates
them (approximating PCA for this kind of data) and separates the smooth spectral envelope
(low-order coefficients) from fine structure. This made diagonal-covariance GMMs viable
(Davis and Mermelstein 1980).

Given log-mel energies $L_m$, $m = 0,\dots,M-1$, the orthonormal DCT-II is

$$c_n = \sqrt{\frac{2 - \delta_{n0}}{M}}\sum_{m=0}^{M-1} L_m\cos\!\left(\frac{\pi n (2m + 1)}{2M}\right),\qquad n = 0,\dots,N_c - 1.$$

This matches `scipy.fft.dct(L, type=2, norm="ortho")`. Keep the first 13 (classic HTK, Kaldi
`num_ceps=13`) or 20 (librosa default `n_mfcc=20`); torchaudio defaults to `n_mfcc=40`.
$c_0$ is proportional to log energy and is often replaced by the frame log energy.

Library details:

- librosa `feature.mfcc`: `power_to_db(melspectrogram(...))` (so dB, $10\log_{10}$,
  `top_db=80`), then `scipy.fft.dct(type=2, norm="ortho")` along the mel axis.
- torchaudio `transforms.MFCC`: `AmplitudeToDB("power", top_db=80)` by default
  (`log_mels=False`); with `log_mels=True` it uses $\ln(M + 10^{-6})$ instead. DCT type 2,
  `norm="ortho"`.
- HTK and Kaldi use natural log energies, so MFCC values differ from librosa's by a factor
  of $10/\ln 10 = 4.343$ (plus normalization differences).

### 12.2 Liftering

HTK sinusoidal lifter with parameter $L$ (typically 22):

$$\tilde c_n = \left(1 + \frac{L}{2}\sin\frac{\pi n}{L}\right)c_n.$$

librosa applies `1 + (lifter/2) * sin(pi * k / lifter)` with `k = 1..n_mfcc` to rows
`0..n_mfcc-1`, so it scales $c_0$ by the $n = 1$ factor. This is an indexing difference from
HTK, where $n$ runs over the cepstral index. Check which convention a fixture uses.

### 12.3 Deltas

HTK regression formula with window $\Theta$ (typically 2):

$$d_t = \frac{\sum_{\theta=1}^{\Theta}\theta\,(c_{t+\theta} - c_{t-\theta})}{2\sum_{\theta=1}^{\Theta}\theta^2}.$$

Edges replicate the first and last frames. Delta-deltas apply the same formula to $d_t$.
librosa's `feature.delta` instead uses a Savitzky-Golay filter (`width=9`, `mode="interp"`),
so its values differ from the HTK formula.

### 12.4 Cepstral mean and variance normalization (CMVN)

$$\hat c_t = \frac{c_t - \mu}{\sigma},\qquad \mu = \frac{1}{T}\sum_t c_t,\quad \sigma^2 = \frac{1}{T}\sum_t (c_t - \mu)^2.$$

Because convolution with a fixed channel $h$ becomes addition in the log-spectral (and
cepstral) domain, mean subtraction removes stationary channel effects. Variants:
per-utterance, per-speaker (Kaldi `apply-cmvn --utt2spk`), and sliding-window (online,
typically 300 frames, 3 s). Kaldi applies CMN by default and variance normalization optionally.

**Lab idea (L4-c).** Implement `dct2_ortho(x)` from the formula, `mfcc(y, sr, ...)`, `lifter`,
`deltas(c, width)`, and `cmvn(c)`. Checks: DCT equals `scipy.fft.dct(..., norm="ortho")`;
MFCC equals stored librosa fixtures to 1e-3; deltas of a linear ramp equal its slope in the
interior; CMVN output has zero mean and unit variance per dimension; a test that CMN removes
the effect of convolving the input with a short FIR channel (the MFCC difference drops by at
least 90%).

---

## 13. Constant-Q transform and chroma

**Constant-Q transform (Brown 1991).** Bins are geometrically spaced,
$f_k = f_{\min}2^{k/b}$ with $b$ bins per octave. Each bin has the same quality factor

$$Q = \frac{f_k}{\Delta f_k} = \frac{1}{2^{1/b} - 1},$$

so window length varies per bin: $N_k = \left\lceil Q f_s / f_k \right\rceil$. For $b = 12$,
$Q = 16.82$; for $b = 36$, $Q = 51.44$ (verified). Efficient implementations use spectral
kernels (Brown and Puckette 1992) or recursive octave downsampling (Schörkhuber and Klapuri
2010; librosa `cqt`). CQT suits music (log-frequency pitch grid) and some pitch features.

**Chroma.** Fold spectral energy into 12 pitch classes regardless of octave:
$\text{pc}(f) = \operatorname{round}\!\left(12\log_2(f/f_{\text{ref}})\right) \bmod 12$
(with $f_{\text{ref}}$ = C, for example 261.63 Hz). librosa's `filters.chroma` uses Gaussian
bumps in the log-frequency domain (`ctroct=5.0`, `octwidth=2`, `norm=2`) rather than hard
binning.

**Lab idea (L4-d).** Implement a naive CQT (direct inner products with per-bin windowed
complex exponentials) and `chroma_from_stft`. Checks: a 440 Hz tone peaks at the A bin of
chroma; a C-major triad activates C, E, and G; the naive CQT magnitude of pure tones peaks at
the expected bin across three octaves.

---

## 14. Pitch estimation

**F0 versus pitch.** F0 is a physical property (the fundamental frequency of a periodic
signal). Pitch is a percept; it can be heard at a missing fundamental (harmonics at 400, 600,
800 Hz evoke 200 Hz). Typical speaking F0 is roughly 85-180 Hz for adult men and 165-255 Hz
for adult women (commonly cited ranges). Pitch trackers must also make a voicing decision.

### 14.1 Autocorrelation

$$r_t(\tau) = \sum_{j=t+1}^{t+W} x_j x_{j+\tau}.$$

The first major peak at $\tau > 0$ estimates the period $T_0$, and $F_0 = f_s/T_0$. Failure
modes: octave errors (a peak at $2T_0$ is nearly as high) and formant peaks (a strong F1 makes
a short-lag peak). Center clipping and normalization mitigate these.

### 14.2 YIN (de Cheveigné and Kawahara, JASA 111(4), 2002)

YIN adds a sequence of steps to autocorrelation. The paper's Table I gives gross error rates
for the cumulative steps (integration window 25 ms, search range 40-800 Hz, threshold 0.1):

| Step | Description | Gross error |
|---|---|---|
| 1 | Autocorrelation | 10.0% |
| 2 | Difference function | 1.95% |
| 3 | Cumulative mean normalized difference | 1.69% |
| 4 | Absolute threshold | 0.78% |
| 5 | Parabolic interpolation | 0.77% |
| 6 | Best local estimate | 0.50% |

Step 2, difference function:

$$d_t(\tau) = \sum_{j=1}^{W}(x_j - x_{j+\tau})^2 = r_t(0) + r_{t+\tau}(0) - 2r_t(\tau).$$

Step 3, cumulative mean normalized difference function (CMNDF):

$$d'_t(\tau) = \begin{cases}1, & \tau = 0\\[1mm] \dfrac{d_t(\tau)}{\frac{1}{\tau}\sum_{j=1}^{\tau} d_t(j)}, & \tau > 0.\end{cases}$$

This removes the dip at $\tau = 0$ and normalizes for amplitude.

Step 4: choose the smallest $\tau$ that gives a local minimum of $d'$ below an absolute
threshold (0.1 in the paper). If none exists, choose the global minimum. The paper notes that
the exact threshold "does not critically affect error rates."

Step 5: refine $\tau$ with parabolic interpolation through $d'(\tau-1), d'(\tau), d'(\tau+1)$:

$$\tau^* = \tau + \frac{d'(\tau - 1) - d'(\tau + 1)}{2\left(d'(\tau - 1) - 2d'(\tau) + d'(\tau + 1)\right)}.$$

(The paper interpolates on $d$ rather than $d'$; librosa interpolates on the CMNDF.)

Step 6: best local estimate. Search nearby frames for a better minimum, like a local
smoothing step.

The value $d'(\tau^*)$ serves as an aperiodicity or voicing measure.

**Hand example (verified).** For $x = [0, 1, 0, -1]$ repeated (period 4) and $W = 4$:
$d = [0, 4, 8, 4, 0, 4]$ for $\tau = 0..5$, and
$d' = [1, 1, 1.333, 0.75, 0, 1]$. The first value below 0.1 is at $\tau = 4$, so
$F_0 = f_s/4$. Changing the waveform to $[0, 1, 0.5, -1]$ gives
$d' = [1, 1, 1.308, 0.771, 0, 1.023]$, again selecting $\tau = 4$.

librosa `yin` defaults: `frame_length=2048`, `trough_threshold=0.1`, and required `fmin` and
`fmax`. librosa 1.0 removed the `win_length` parameter from `yin` and `pyin`.

### 14.3 pYIN (Mauch and Dixon, ICASSP 2014)

pYIN replaces the single threshold with a distribution over thresholds (a Beta prior; librosa
`beta_parameters=(2, 18)` over `n_thresholds=100`), producing multiple F0 candidates with
probabilities per frame. An HMM with voiced and unvoiced pitch states, decoded by Viterbi,
then picks a smooth track. librosa defaults: `resolution=0.1` semitones,
`max_transition_rate=35.92` octaves per second, `switch_prob=0.01`, `no_trough_prob=0.01`,
`boltzmann_parameter=2`, and unvoiced frames filled with `NaN`.

### 14.4 CREPE (Kim, Salamon, Li, and Bello, ICASSP 2018, arXiv:1802.06182)

A convolutional network over 1024-sample frames of 16 kHz audio with a 10 ms hop outputs 360
pitch bins spaced 20 cents apart. The official code's bin centers are
`np.linspace(0, 7180, 360) + 1997.3794` cents relative to 10 Hz, which covers about 31.7 Hz to
2005 Hz (verified). The estimate is a weighted average of cents in a neighborhood of the
argmax; optional Viterbi smoothing is available. Pretrained models are in `crepe` (TensorFlow)
and `torchcrepe`.

**Gotchas.** Frequency search range must contain F0; octave errors dominate error metrics;
evaluate on voiced frames (gross pitch error: more than 20% or 50 cents off, depending on the
paper); YIN frame length must exceed $2\tau_{\max}$.

**Lab idea (L5-a).** Implement `autocorr_f0`, `yin_difference`, `cmndf`, and `yin_f0(frame,
sr, fmin, fmax, threshold)` with parabolic interpolation. Checks: the hand example; on seeded
harmonic signals with random F0 in 80-400 Hz, mixed phases, and 20 dB SNR noise, gross error
below 2% (autocorrelation should fail the octave test where YIN passes); compare with stored
`librosa.yin` fixtures within 1 Hz on clean tones.

---

## 15. Source-filter model, LPC, and the cepstrum

### 15.1 Source-filter model

**Intuition.** Speech is an excitation (glottal pulse train for voiced sounds, turbulent noise
for fricatives) filtered by the vocal tract, whose resonances are formants, then radiated at
the lips:

$$S(z) = E(z)\,G(z)\,V(z)\,R(z),$$

with excitation $E$, glottal shape $G$, vocal tract $V$, and radiation $R \approx 1 - z^{-1}$.
Source and filter are largely independent: you can change pitch without changing formants
(vowel identity), which underlies LPC vocoders and much of TTS.

**Vocal tract as a uniform tube**, closed at the glottis and open at the lips (a quarter-wave
resonator):

$$F_n = \frac{(2n - 1)c}{4L}.$$

With $L = 17.5$ cm and $c = 350$ m/s: $F_1 = 500$, $F_2 = 1500$, $F_3 = 2500$ Hz, which is
close to a neutral schwa. Constrictions shift formants; a concatenation of tube sections is
equivalent to a lattice all-pole filter whose reflection coefficients relate to area ratios
(Kelly-Lochbaum). Peterson and Barney (1952) average male formants: /i/ about 270 and
2290 Hz; /ɑ/ about 730 and 1090 Hz; /u/ about 300 and 870 Hz. F1 tracks tongue height
(inversely); F2 tracks frontness.

### 15.2 Linear predictive coding (LPC), autocorrelation method

**Intuition.** If the vocal tract is all-pole, each sample is predictable from the previous
$p$ samples; the prediction error is the (whitened) excitation.

Model and error, for a windowed frame $x[n]$, $n = 0,\dots,N-1$:

$$\hat x[n] = \sum_{k=1}^{p}\alpha_k x[n-k],\qquad e[n] = x[n] - \hat x[n],\qquad A(z) = 1 - \sum_{k=1}^{p}\alpha_k z^{-k}.$$

Minimizing $\sum_n e[n]^2$ over all $n$ (with the frame zero outside $[0, N-1]$) gives the
Yule-Walker normal equations with a symmetric Toeplitz matrix:

$$\sum_{k=1}^{p}\alpha_k\,r[|i - k|] = r[i],\quad i = 1,\dots,p,\qquad r[k] = \sum_{n=k}^{N-1}x[n]\,x[n-k].$$

**Levinson-Durbin recursion** (solves the system in $O(p^2)$ and guarantees $|k_i| < 1$, so
$1/A(z)$ is stable):

1. Initialize $E_0 = r[0]$.
2. For $i = 1,\dots,p$:
   1. $k_i = \dfrac{r[i] - \sum_{j=1}^{i-1}\alpha_j^{(i-1)}r[i-j]}{E_{i-1}}$.
   2. $\alpha_i^{(i)} = k_i$.
   3. $\alpha_j^{(i)} = \alpha_j^{(i-1)} - k_i\,\alpha_{i-j}^{(i-1)}$ for $j = 1,\dots,i-1$.
   4. $E_i = (1 - k_i^2)E_{i-1}$.
3. The solution is $\alpha_j = \alpha_j^{(p)}$, and $E_p$ is the minimum prediction-error
   energy.

The $k_i$ are reflection (PARCOR) coefficients. Sign conventions differ: some texts write
$A(z) = 1 + \sum a_k z^{-k}$ with $a_k = -\alpha_k$ and $k_i$ negated.

**Worked example (verified).** $r = [1, 0.5, 0.1]$, $p = 2$:

- $E_0 = 1$, $k_1 = 0.5$, $\alpha_1^{(1)} = 0.5$, $E_1 = 0.75$.
- $k_2 = (0.1 - 0.5 \cdot 0.5)/0.75 = -0.2$, $\alpha_2 = -0.2$,
  $\alpha_1 = 0.5 - (-0.2)(0.5) = 0.6$, $E_2 = 0.75(1 - 0.04) = 0.72$.
- Check: `scipy.linalg.solve_toeplitz([1, 0.5], [0.5, 0.1])` returns `[0.6, -0.2]`.
  In $A(z)$ coefficient form this is `[1, -0.6, 0.2]`.

**LPC spectral envelope.** With gain $G^2 = E_p$:

$$|H(e^{j\omega})|^2 = \frac{G^2}{\left|1 - \sum_{k=1}^{p}\alpha_k e^{-j\omega k}\right|^2}.$$

Evaluate with `scipy.signal.freqz(np.sqrt(E), a_poly, worN=...)`. Formants come from the roots
$z_i$ of $A(z)$: frequency $F = \frac{\angle z_i}{2\pi}f_s$ and bandwidth
$B = -\frac{f_s}{\pi}\ln|z_i|$. Rule of thumb for order: $p = f_s/1000 + 2$ (18 at 16 kHz), one
pole pair per kHz plus two for glottal and radiation shape. Pre-emphasize before LPC.

**LPC to cepstrum recursion** (for $H = G/A$ with $A(z) = 1 - \sum\alpha_k z^{-k}$):
$c_0 = \ln G$, and for $1 \le n \le p$,
$c_n = \alpha_n + \sum_{k=1}^{n-1}\frac{k}{n}c_k\,\alpha_{n-k}$.

### 15.3 Cepstrum and homomorphic deconvolution

**Intuition.** Convolution becomes multiplication in frequency and addition after the log.
The log spectrum is a slowly varying envelope (vocal tract) plus a rapidly varying ripple
(harmonics). A second Fourier transform separates them by "quefrency."

Real cepstrum:

$$c[n] = \text{IDFT}\!\left(\ln|\text{DFT}(x)|\right).$$

For $x = e * h$, $\ln|X| = \ln|E| + \ln|H|$, so $c = c_e + c_h$. Low-quefrency coefficients
(below about 2-3 ms) describe the envelope $h$; a peak at quefrency $n_0 = f_s/F_0$ samples
reveals the pitch period (Noll 1967). Liftering (windowing in quefrency) and transforming back
yields a smoothed envelope. Terms: cepstrum, quefrency, liftering, and rahmonics (Bogert,
Healy, and Tukey 1963). MFCCs are a "mel cepstrum" computed with a DCT on log-mel energies.

**Gotchas.** $\ln 0$ for silent frames (add epsilon); forgetting the frame window; confusing
the complex cepstrum (needs phase unwrapping) with the real cepstrum.

**Lab idea (L5-b).** Implement `autocorrelation(frame, p)`, `levinson_durbin(r, p)` returning
$(\alpha, E, k)$, `lpc_envelope`, `formants_from_lpc`, and `real_cepstrum`. Checks: the worked
example exactly; `levinson_durbin` equal to `solve_toeplitz` on seeded AR processes and all
$|k_i| < 1$; formants of a synthetic vowel (impulse train at 120 Hz through a known all-pole
filter with poles at 500, 1500, and 2500 Hz) within 5% of truth; cepstral pitch within 2% of
120 Hz.

---

## 16. Phonetics, G2P, and text normalization

**Phones and phonemes.** A phoneme is a contrastive sound category in a language (/p/ versus
/b/ distinguishes "pat" and "bat"). Phones are the actual sounds. Allophones are phones that
realize one phoneme in different contexts: English /p/ is aspirated [pʰ] in "pin" and
unaspirated [p] in "spin"; /t/ becomes the flap [ɾ] in American "butter." Coarticulation
means articulators anticipate and carry over neighboring sounds, so phone acoustics depend on
context (which motivates triphones in Section 20).

**Transcription systems.**

- IPA: International Phonetic Alphabet, language-independent.
- ARPAbet: ASCII phone set for American English. CMUdict uses 39 phonemes (the
  `cmudict.phones` file has 39 lines), with vowels carrying stress digits 0 (none),
  1 (primary), and 2 (secondary), for example `SPEECH  S P IY1 CH`. The `cmudict.dict` file on
  GitHub has 135,166 lines (entries including pronunciation variants); SLP3 describes it as
  about 134,000 words.
- TIMIT uses 61 phone labels, conventionally folded to 39 for evaluation (Lee and Hon 1989).

**Articulatory dimensions.** Consonants by place (bilabial, labiodental, dental, alveolar,
postalveolar, palatal, velar, glottal), manner (stop, fricative, affricate, nasal,
approximant), and voicing. Vowels by height, backness, and rounding, plus diphthongs.

**Prosody.** Suprasegmental features: F0 contour (intonation, tone), duration (rhythm, phrase
lengthening), and energy (stress, emphasis). TTS systems predict all three; FastSpeech-style
variance adaptors (later in the course) predict them explicitly.

**Grapheme-to-phoneme (G2P) conversion.**

1. Dictionary lookup (CMUdict) for known words.
2. Rule-based letter-to-sound rules (for example, Elovitz et al. 1976, NRL rules; eSpeak NG
   rules, wrapped by the `phonemizer` package and used by many TTS systems).
3. Joint-sequence (graphone) n-gram models: Sequitur G2P (Bisani and Ney 2008); WFST-based
   Phonetisaurus (Novak et al.).
4. Neural sequence-to-sequence: LSTM encoder-decoders (Yao and Zweig 2015,
   arXiv:1506.00196), Transformers, and multilingual byte-level ByT5 G2P (CharsiuG2P, Zhu
   et al. 2022, arXiv:2204.03067). `g2p_en` (last release 2019) combines CMUdict lookup,
   part-of-speech disambiguation of homographs, and a neural fallback.

Homographs ("read," "live," "record") need context. Metrics: phoneme error rate (PER) and word
error rate on pronunciations.

**Text normalization.** Converting non-standard words to spoken form: "Dr." to "doctor" or
"drive," "1995" to "nineteen ninety five," "$3.50" to "three dollars fifty cents" (Sproat et
al. 2001, "Normalization of non-standard words"). Production systems use WFST grammars (for
example, NVIDIA NeMo text processing, Zhang et al. 2021, arXiv:2104.05055) or neural models
with covering grammars (Sproat and Jaitly 2016, arXiv:1611.00068). ASR evaluation needs the
inverse direction and a canonical normalizer before WER; Whisper ships
`EnglishTextNormalizer` for this purpose.

**Lab idea (L6-a).** Provide a 2,000-word CMUdict subset. Students implement
`lookup_g2p`, a tiny rule-based fallback, `strip_stress`, `arpabet_to_ipa`, and a number
normalizer `normalize_number("1995", style="year")`. Checks: exact matches on held-out
dictionary words; PER of the fallback below a threshold on held-out words; 30 normalization
unit tests.

---

## 17. Voice activity detection

**Energy-based VAD.** Frame log energy $E_t = 10\log_{10}\!\left(\frac{1}{N}\sum_n x_t[n]^2 + \epsilon\right)$.
Estimate a noise floor (for example, the 10th percentile of $E_t$, or a slowly updated
minimum), label frames speech when $E_t$ exceeds the floor by a margin (for example, 6-10 dB),
and apply hysteresis (two thresholds) and hangover (keep speech for about 100-300 ms after it
ends). Zero-crossing rate helps catch low-energy unvoiced fricatives. Simple and fast, but
brittle in non-stationary noise.

**WebRTC VAD** (Google, from the WebRTC project; Python binding `webrtcvad`):

- Input: 16-bit mono PCM at 8, 16, 32, or 48 kHz; frames of exactly 10, 20, or 30 ms;
  aggressiveness mode 0-3 (3 is most aggressive at filtering non-speech).
- Internals: a filterbank splits 0-4 kHz into six sub-bands (80-250, 250-500, 500-1000,
  1000-2000, 2000-3000, and 3000-4000 Hz); log energies form a six-dimensional feature. Per
  band, two-component Gaussian mixture models for speech and noise give log-likelihood ratios,
  tested against mode-dependent local and global thresholds. Models adapt online, and a
  hangover extends speech segments.

**Silero VAD** (neural; MIT license; `silero-vad` 6.2.3):

- Supports 8 kHz and 16 kHz. The streaming model takes chunks of exactly 512 samples at 16 kHz
  (32 ms) or 256 samples at 8 kHz, and internally prepends a context of 64 or 32 samples from
  the previous chunk. It outputs a speech probability per chunk.
- The JIT model is about 2 MB; one chunk processes in under 1 ms on one CPU thread (project
  README).
- `get_speech_timestamps` defaults: `threshold=0.5`, `neg_threshold=threshold - 0.15`,
  `min_speech_duration_ms=250`, `min_silence_duration_ms=100`, `speech_pad_ms=30`,
  `window_size_samples=512`, and `max_speech_duration_s=inf`.
- v6.0.0 (2025-08-26) reported 16% fewer errors on noisy real-world data. 6.2.3 lets I/O use
  either torchaudio or TorchCodec.

**Metrics.** Frame-level precision, recall, and F1; ROC and AUC over thresholds; detection
cost; segment-level miss and false-alarm rates.

**Lab idea (L6-b).** Implement `frame_energy_db`, `energy_vad(x, sr, margin_db, hangover_ms)`,
and a hysteresis smoother. The grader builds seeded mixtures of synthetic voiced bursts
(harmonic signals with formant filtering) and pink noise at 5, 10, and 20 dB SNR with known
labels, and checks frame F1 at or above 0.9 at 20 dB SNR. An optional GPU or CPU extension
compares against Silero VAD probabilities.

---

## 18. Dynamic time warping and template matching

**Intuition.** Two utterances of the same word differ in timing. DTW finds the monotonic
alignment of two feature sequences that minimizes total frame distance.

For sequences $a_1..a_n$ and $b_1..b_m$ with local cost $c(i, j) = \|a_i - b_j\|$:

$$D(0, 0) = 0,\quad D(i, 0) = D(0, j) = \infty\ (i, j > 0),$$

$$D(i, j) = c(i, j) + \min\{D(i-1, j),\ D(i, j-1),\ D(i-1, j-1)\}.$$

The DTW distance is $D(n, m)$; backtracking from $(n, m)$ gives the warping path. Constraints
from Sakoe and Chiba (1978) and Itakura (1975):

- Boundary: the path starts at $(1, 1)$ and ends at $(n, m)$.
- Monotonicity and continuity from the step pattern.
- Global constraints: the Sakoe-Chiba band $|i - j| \le w$, or the Itakura parallelogram.
- Slope constraints and weighted step patterns (for example, Sakoe-Chiba symmetric weights
  of 2 on the diagonal step with normalization by $n + m$).

**Worked 3x3 example (verified).** $a = [0, 1, 2]$, $b = [0, 2, 2]$, absolute-difference cost:

Cost matrix $c$ (rows $a_i$, columns $b_j$):

|  | 0 | 2 | 2 |
|---|---|---|---|
| **0** | 0 | 2 | 2 |
| **1** | 1 | 1 | 1 |
| **2** | 2 | 0 | 0 |

Accumulated cost $D$:

|  | $j=1$ | $j=2$ | $j=3$ |
|---|---|---|---|
| $i=1$ | 0 | 2 | 4 |
| $i=2$ | 1 | 1 | 2 |
| $i=3$ | 3 | 1 | 1 |

DTW distance is 1, with path $(1,1) \to (2,1) \to (3,2) \to (3,3)$. A second check:
$a = [1, 2, 3]$ and $b = [1, 1, 2, 3]$ give distance 0, because warping absorbs the repeated
frame.

**Template-based recognition.** Store one or more MFCC templates per word; classify a test
utterance by the minimum (length-normalized) DTW distance. This was the dominant isolated-word
approach in the 1970s before HMMs, and it is still a good first "recognizer" lab.

**Gotchas.** Forgetting infinite boundary initialization (paths can then start anywhere);
comparing unnormalized DTW distances between templates of different lengths; $O(nm)$ memory
for long sequences. `librosa.sequence.dtw` is a grader-side reference.

**Lab idea (L7-a).** Implement `dtw(a, b, metric, band)` returning distance and path, and
`classify_by_templates`. Checks: both hand examples; agreement with `librosa.sequence.dtw`
fixtures; band-constrained results equal unconstrained results when the band is wide; a
10-word synthetic "digits" task (formant-synthesized vowel sequences with random time warps)
above 95% accuracy.

---

## 19. Hidden Markov models and Gaussian mixtures

### 19.1 HMM definition

An HMM $\lambda = (A, B, \pi)$ has $N$ hidden states, transition probabilities
$a_{ij} = P(q_{t+1} = j \mid q_t = i)$, emission densities $b_j(o_t) = p(o_t \mid q_t = j)$, and
initial probabilities $\pi_i$. Rabiner's three problems: evaluation ($P(O \mid \lambda)$),
decoding (best state sequence), and learning (estimate $\lambda$).

### 19.2 Forward, backward, and Viterbi

Forward, for $t = 1,\dots,T-1$:

$$\alpha_1(i) = \pi_i\,b_i(o_1),\qquad \alpha_{t+1}(j) = \left[\sum_{i=1}^{N}\alpha_t(i)\,a_{ij}\right]b_j(o_{t+1}),\qquad P(O \mid \lambda) = \sum_i \alpha_T(i).$$

Backward:

$$\beta_T(i) = 1,\qquad \beta_t(i) = \sum_{j=1}^{N}a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j).$$

Viterbi:

$$\delta_1(i) = \pi_i\,b_i(o_1),\qquad \delta_{t+1}(j) = \max_i\left[\delta_t(i)\,a_{ij}\right]b_j(o_{t+1}),\qquad \psi_{t+1}(j) = \arg\max_i\left[\delta_t(i)\,a_{ij}\right].$$

Backtrack from $q_T^* = \arg\max_i\delta_T(i)$ with $q_t^* = \psi_{t+1}(q_{t+1}^*)$.

**Log space.** Products of hundreds of probabilities underflow float64. Use
$\tilde\alpha = \log\alpha$ and log-sum-exp:

$$\tilde\alpha_{t+1}(j) = \operatorname{LSE}_i\!\left(\tilde\alpha_t(i) + \log a_{ij}\right) + \log b_j(o_{t+1}),\qquad \operatorname{LSE}(v) = v_{\max} + \log\sum_i e^{v_i - v_{\max}}.$$

Viterbi in log space replaces products with sums and needs no LSE. Rabiner's alternative is
per-frame scaling (normalize $\alpha_t$ to sum to 1 and accumulate $\log$ of the scale
factors). Use `scipy.special.logsumexp`, and represent $\log 0$ as `-np.inf`.

**Worked toy example (verified).** Two states, three symbols:
$\pi = [0.6, 0.4]$, $A = \begin{bmatrix}0.7 & 0.3\\0.4 & 0.6\end{bmatrix}$,
$B = \begin{bmatrix}0.5 & 0.4 & 0.1\\0.1 & 0.3 & 0.6\end{bmatrix}$, and $O = (0, 1, 2)$.

- Forward: $\alpha_1 = [0.3, 0.04]$; $\alpha_2 = [(0.21 + 0.016)\cdot0.4,\ (0.09 + 0.024)\cdot0.3] = [0.0904, 0.0342]$;
  $\alpha_3 = [0.007696, 0.028584]$; $P(O) = 0.03628$.
- Backward: $\beta_1 = [0.106, 0.112]$, and
  $\sum_i \pi_i b_i(o_1)\beta_1(i) = 0.3\cdot0.106 + 0.04\cdot0.112 = 0.03628$ (consistency
  check).
- Viterbi: $\delta_1 = [0.3, 0.04]$, $\delta_2 = [0.084, 0.027]$,
  $\delta_3 = [0.00588, 0.01512]$; best path $(0, 0, 1)$ with probability 0.01512.

### 19.3 Baum-Welch (EM for HMMs)

State and transition posteriors:

$$\gamma_t(i) = \frac{\alpha_t(i)\,\beta_t(i)}{P(O \mid \lambda)},\qquad \xi_t(i, j) = \frac{\alpha_t(i)\,a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)}{P(O \mid \lambda)}.$$

Re-estimation (sum over all training sequences in practice):

$$\hat\pi_i = \gamma_1(i),\qquad \hat a_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i, j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\qquad \hat b_j(v_k) = \frac{\sum_{t:\,o_t = v_k}\gamma_t(j)}{\sum_{t=1}^{T}\gamma_t(j)}.$$

Each iteration does not decrease $P(O \mid \lambda)$. In log space, compute
$\log\gamma = \tilde\alpha + \tilde\beta - \log P(O)$.

### 19.4 GMM emissions and EM

Continuous-density HMMs use a Gaussian mixture per state, usually with diagonal covariance:

$$b_j(o) = \sum_{m=1}^{M}c_{jm}\,\mathcal N(o;\mu_{jm},\Sigma_{jm}),\qquad \log\mathcal N(o;\mu,\operatorname{diag}\sigma^2) = -\frac{1}{2}\left[D\log 2\pi + \sum_d\log\sigma_d^2 + \sum_d\frac{(o_d - \mu_d)^2}{\sigma_d^2}\right].$$

Component posterior within state $j$:
$\gamma_t(j, m) = \gamma_t(j)\,\dfrac{c_{jm}\mathcal N(o_t;\mu_{jm},\Sigma_{jm})}{\sum_{k}c_{jk}\mathcal N(o_t;\mu_{jk},\Sigma_{jk})}$.

M-step:
$\hat c_{jm} = \dfrac{\sum_t\gamma_t(j, m)}{\sum_t\gamma_t(j)}$,
$\hat\mu_{jm} = \dfrac{\sum_t\gamma_t(j, m)\,o_t}{\sum_t\gamma_t(j, m)}$,
$\hat\sigma^2_{jm} = \dfrac{\sum_t\gamma_t(j, m)(o_t - \hat\mu_{jm})^2}{\sum_t\gamma_t(j, m)}$.

A standalone GMM is the same with $\gamma_t(j) = 1$. Practical tricks: variance flooring
(for example, 0.01 of the global variance), mixture splitting ("mixing up" by perturbing
means by $\pm 0.2\sigma$), and k-means initialization. Viterbi training replaces soft
$\gamma$ with a hard alignment.

**Gotchas.** Underflow (use log space throughout); collapsing variances; forgetting that
`xi` sums over $t = 1..T-1$; confusing row-stochastic $A$ orientation (rows sum to 1).

**Lab idea (L8-a).** Implement `log_forward`, `log_backward`, `viterbi`, and one
`baum_welch_step` for discrete HMMs, plus `gmm_em_step` for diagonal GMMs. Checks: the toy
example exactly; $\log P(O)$ from forward equals the backward computation to 1e-10 for
seeded random HMMs with $T = 2000$ (a naive linear-space implementation underflows here);
log-likelihood non-decreasing over 10 EM iterations; recovery of a seeded 3-component 2-D GMM
within tolerance. Compare with `hmmlearn` fixtures if desired.

---

## 20. HMM-GMM speech recognition

### 20.1 Noisy-channel formulation

$$\hat W = \arg\max_W P(W \mid O) = \arg\max_W \underbrace{P(O \mid W)}_{\text{acoustic model}}\ \underbrace{P(W)}_{\text{language model}}.$$

The acoustic likelihood expands through the lexicon (pronunciation model) into phone HMMs:
$P(O \mid W) \approx \max_{Q}P(O \mid Q)P(Q \mid W)$. In practice:

$$\hat W = \arg\max_W\ \log P(O \mid W) + \lambda_{\text{LM}}\log P(W) + |W|\log\text{WIP},$$

with an LM scale $\lambda_{\text{LM}}$ (often around 10 for GMM systems; tuned on dev data)
compensating for overconfident frame-independent acoustic scores, and a word insertion
penalty.

### 20.2 Acoustic units

- Each phone is a left-to-right HMM with three emitting states (begin, middle, end) and
  self-loops. A self-loop probability $a$ gives geometric durations with mean $1/(1-a)$ frames.
- Context-dependent triphones (written `k-ae+t`, meaning `ae` with left context `k` and right
  context `t`) capture coarticulation. With about 40 phones there are up to $40^3 = 64{,}000$
  triphones, times three states. Most are rare or unseen.
- **Decision-tree state tying** (Young, Odell, and Woodland 1994). For each phone and state
  position, pool all triphone states and greedily split them with binary phonetic questions
  ("Is the left context a nasal?", "Is the right context a front vowel?"). Choose the question
  that maximizes the log-likelihood gain under single-Gaussian statistics. Stop when the gain
  falls below a threshold or occupancy is too low. Leaves are tied states ("senones"), usually
  a few thousand. Unseen triphones get models by walking the tree.

### 20.3 Training pipeline (Kaldi-style)

1. Features: 13 MFCCs plus deltas and delta-deltas (39 dimensions) with CMVN.
2. Monophone training with flat start (global mean and variance), then Baum-Welch or Viterbi
   training.
3. Alignment, then triphone tree building and training.
4. Feature transforms (LDA+MLLT) and speaker adaptation (fMLLR/SAT).
5. Discriminative training (MMI, MPE, and boosted MMI).
6. Hybrid DNN-HMM systems later replaced GMM emission scores with scaled DNN posteriors,
   $p(o \mid s) \propto P(s \mid o)/P(s)$.

### 20.4 WFST decoding, $H \circ C \circ L \circ G$

Weighted finite-state transducers (Mohri, Pereira, and Riley 2002) represent each knowledge
source as a transducer over the tropical semiring (weights are negative log probabilities,
$\oplus = \min$, $\otimes = +$):

- $G$: the language model as an acceptor over words (n-gram with backoff arcs).
- $L$: the lexicon, mapping phone sequences to words.
- $C$: context dependency, mapping context-dependent phones to phones.
- $H$: HMM structure, mapping transition-ids (or HMM states) to context-dependent phones.

Composition ($\circ$), determinization (det), and minimization (min) produce one static graph.
Kaldi's recipe, quoted from its documentation, is

$$\text{HCLG} = \text{asl}(\min(\text{rds}(\det(H' \circ \min(\det(C \circ \min(\det(L \circ G)))))))),$$

where asl adds self-loops, rds removes disambiguation symbols, and $H'$ is $H$ without
self-loops. Disambiguation symbols (`#0`, `#1`, ...) mark homophones, lexicon prefixes, and LM
backoff arcs so that determinization is possible. HCLG input symbols are transition-ids and
output symbols are words. Decoding is a Viterbi beam search over HCLG, with acoustic scores
supplied per frame, producing lattices for rescoring.

**Kaldi** (Povey et al., ASRU 2011) is the canonical open-source HMM-GMM and hybrid toolkit,
with OpenFst underneath. Its successors ("next-gen Kaldi": k2, icefall, Lhotse, and
sherpa-onnx) target neural transducer and CTC models.

**Lab idea (L8-b).** A toy isolated-digit recognizer: synthetic formant-synthesized
"digits" (or the free Free Spoken Digit Dataset, pre-extracted as MFCC `.npy` so labs stay
under 60 s), one left-to-right 5-state GMM-HMM per word trained with Viterbi training, and
recognition by maximum log-likelihood. Checks: training log-likelihood increases; test
accuracy above a threshold; a unit test of a hand-built WFST composition for a two-word
lexicon (students compose a tiny $L$ and $G$ represented as arc lists and verify the best path
weight).

---

## 21. N-gram language models, Kneser-Ney smoothing, and perplexity

**Maximum likelihood bigram.**
$P_{\text{MLE}}(w_i \mid w_{i-1}) = \dfrac{C(w_{i-1}w_i)}{C(w_{i-1})}$. Unseen n-grams get
probability 0, which makes perplexity infinite, so smoothing is required.

**Absolute discounting** subtracts a fixed $d$ (about 0.75) from every nonzero count and
redistributes the mass to a lower-order distribution (Church and Gale's held-out observation,
SLP3 Appendix C).

**Interpolated Kneser-Ney, bigram form** (SLP3 Appendix C, eq. C.7):

$$P_{\text{KN}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}w_i) - d,\ 0)}{C(w_{i-1})} + \lambda(w_{i-1})\,P_{\text{CONT}}(w_i),$$

$$\lambda(w_{i-1}) = \frac{d}{C(w_{i-1})}\,\big|\{w : C(w_{i-1}w) > 0\}\big|,\qquad P_{\text{CONT}}(w) = \frac{\big|\{v : C(vw) > 0\}\big|}{\big|\{(u', w') : C(u'w') > 0\}\big|}.$$

The continuation probability asks how many distinct contexts a word completes, so a word like
"Francisco," frequent only after "San," gets a low unigram continuation probability. The
recursive form uses continuation counts at every lower order. Modified Kneser-Ney (Chen and
Goodman 1998) uses three discounts $D_1, D_2, D_{3+}$ estimated from count-of-counts. KenLM
(Heafield 2011) builds modified-KN models and writes the ARPA format.

**Worked KN example (verified by hand and in code).** Corpus with sentence markers:
`<s> a b </s>`, `<s> a c </s>`, `<s> b c </s>`. Bigram types: (`<s>` a), (a b), (b `</s>`),
(a c), (c `</s>`), (`<s>` b), (b c), so 7 types. With $d = 0.75$ and history "a"
($C(a) = 2$, two distinct followers):

- $\lambda(a) = 0.75/2 \times 2 = 0.75$.
- $P_{\text{CONT}}$: a = 1/7, b = 2/7, c = 2/7, `</s>` = 2/7.
- $P_{\text{KN}}(b \mid a) = 0.25/2 + 0.75 \cdot 2/7 = 0.3393$.
- $P_{\text{KN}}(c \mid a) = 0.3393$.
- $P_{\text{KN}}(\texttt{</s>} \mid a) = 0.75 \cdot 2/7 = 0.2143$.
- $P_{\text{KN}}(a \mid a) = 0.75 \cdot 1/7 = 0.1071$.
- The four values sum to 1.0.

**Perplexity.** For a test sequence of $N$ tokens:

$$\text{PP}(W) = P(w_1 \dots w_N)^{-1/N} = 2^{H},\qquad H = -\frac{1}{N}\sum_{i=1}^{N}\log_2 P(w_i \mid w_{<i}).$$

A uniform model over 10 digits has perplexity 10. Conventions: include `</s>` in $N$ but not
`<s>`; perplexities are comparable only with the same vocabulary and tokenization.

**Gotchas.** Mixing natural log and $\log_2$; counting `<s>` as a predicted token; OOV
handling (`<unk>`); forgetting to renormalize after interpolation.

**Lab idea (L7-b).** Implement `count_ngrams`, `kn_bigram_prob`, `interpolated_kn(order=3)`,
and `perplexity`. Checks: the worked example to 1e-4; each conditional distribution sums to 1
over the vocabulary for 50 random histories; KN perplexity on a held-out split of a small
bundled corpus is lower than add-one perplexity.

---

## 22. Evaluation fundamentals

### 22.1 WER and CER via Levenshtein alignment

$$\text{WER} = \frac{S + D + I}{N},$$

where $S$, $D$, and $I$ are substitutions, deletions, and insertions in a minimum-edit
alignment, and $N$ is the number of reference words. Dynamic programming:

$$D(i, 0) = i,\quad D(0, j) = j,\quad D(i, j) = \min\{D(i-1, j) + 1,\ D(i, j-1) + 1,\ D(i-1, j-1) + [r_i \ne h_j]\}.$$

Backtracking yields the operation counts.

**Examples (verified).**

- Reference `i like green tea`, hypothesis `i like the green tee`: one insertion ("the") and
  one substitution ("tea" to "tee"), so WER = 2/4 = 50%.
- Reference `the cat sat on the mat`, hypothesis `the cat sit on mat today`: edit distance 3,
  WER = 50%. The alignment is not unique: one alignment is S+D+I = 1+1+1, and another is
  three substitutions. WER is unique; the S/D/I breakdown depends on tie-breaking.
- CER for `kitten` versus `sitting`: distance 3 (two substitutions and one insertion), CER =
  3/6 = 50%.

**Gotchas.** WER can exceed 100% (many insertions); normalize text (case, punctuation,
numbers, contractions) identically on both sides; corpus WER is total errors over total
reference words, not the mean of per-utterance WERs; empty references need special handling.
`jiwer` 4.0 is a convenient reference.

### 22.2 SNR and SI-SDR

$$\text{SNR} = 10\log_{10}\frac{\|s\|^2}{\|s - \hat s\|^2}.$$

**SI-SDR** (Le Roux, Wisdom, Erdogan, and Hershey, "SDR - half-baked or well done?", ICASSP
2019, arXiv:1811.02508). Project the estimate onto the reference:

$$\alpha = \frac{\hat s^{\top}s}{\|s\|^2},\qquad e_{\text{target}} = \alpha s,\qquad e_{\text{res}} = \hat s - \alpha s,\qquad \text{SI-SDR} = 10\log_{10}\frac{\|\alpha s\|^2}{\|\hat s - \alpha s\|^2}.$$

Subtract the means of $s$ and $\hat s$ first (zero-mean convention in the paper). Scaling the
estimate by any nonzero constant does not change SI-SDR, unlike SNR. SI-SDR improvement
(SI-SDRi) is SI-SDR of the output minus SI-SDR of the input mixture. BSS Eval SDR (Vincent,
Gribonval, and Févotte 2006) allows a 512-tap distortion filter and is not scale-invariant in
the same way.

**Worked example (verified).** $s = [1, 2, 3, 4]$, $\hat s = 0.5s + [0.1, -0.1, 0.2, 0]$
$= [0.6, 0.9, 1.7, 2.0]$ (no mean removal, for hand calculation): $\hat s^{\top}s = 15.5$,
$\|s\|^2 = 30$, $\alpha = 0.5167$, and SI-SDR = 21.90 dB, while plain SNR is only 6.28 dB
because SNR penalizes the 0.5 gain.

### 22.3 PESQ and STOI, briefly

- **PESQ** (ITU-T P.862, 2001): an intrusive perceptual quality model (level alignment, time
  alignment, auditory transform, disturbance aggregation). Raw scores range from -0.5 to 4.5;
  P.862.1 maps to MOS-LQO, and P.862.2 is the wideband (16 kHz, 50-7000 Hz) mode that outputs
  only MOS-LQO. POLQA (P.863) is the successor. The `pesq` package supports only 8 kHz
  (narrowband) and 16 kHz (wideband).
- **STOI** (Taal, Hendriks, Heusdens, and Jensen, IEEE TASLP 2011): resample to 10 kHz, 256-
  sample frames (50% overlap, 512-point FFT), 15 one-third-octave bands starting at 150 Hz,
  correlation of short-time temporal envelopes over 30-frame (384 ms) segments, with clipping
  at a -15 dB lower SDR bound. Output is about 0 to 1; silent frames within 40 dB of the
  maximum-energy frame are removed first (constants from `pystoi`).

**Lab idea (L9-a).** Implement `levenshtein_align(ref, hyp)` returning counts and alignment,
`wer`, `cer`, `snr`, and `si_sdr`. Checks: the worked examples; agreement with `jiwer` on
200 seeded random edit pairs; `si_sdr(s, c * s_hat) == si_sdr(s, s_hat)` for random nonzero
`c`; `snr` changes under scaling. PESQ and STOI are demonstrations (call the packages) rather
than implement-from-scratch tasks.

---

## 23. Proposed lesson ordering for this part of the course

Each lesson lists a one-line lab. All labs run on CPU in under 60 seconds with seeded
synthetic signals and grader-side fixtures.

1. **Sound, sampling, and quantization** (Sections 1-4). Lab: implement dB and dBFS helpers,
   alias-frequency prediction, a polyphase resampler, quantization with the 6.02N + 1.76 check,
   mu-law encode and decode, and a NumPy-only WAV reader and writer.
2. **The DFT, windows, and the STFT** (Sections 5-6). Lab: DFT matrix and radix-2 FFT checked
   against `np.fft`, window formulas checked against SciPy, and an STFT and least-squares ISTFT
   with exact frame counts and perfect reconstruction.
3. **Phase: Griffin-Lim and the phase vocoder** (Section 7). Lab: GLA and fast GLA with
   monotone spectral convergence, and a phase-vocoder time stretch that preserves a tone's
   frequency.
4. **Filters, convolution, and reverb** (Section 8). Lab: FFT convolution without circular
   wrap-around, `lfilter` from the difference equation, pre-emphasis, and a synthetic RIR with
   measured RT60.
5. **Hearing and perceptual features** (Sections 9-13). Lab: mel conversion (HTK and Slaney),
   a mel filterbank matching librosa, a NumPy Whisper log-mel matching fixtures to 1e-4, and
   MFCC with DCT-II, deltas, and CMVN.
6. **Pitch and the source-filter model** (Sections 14-15). Lab: YIN (difference function,
   CMNDF, threshold, parabolic interpolation) plus Levinson-Durbin LPC, formant estimation, and
   cepstral pitch on synthetic vowels.
7. **Phonetics, text, and VAD** (Sections 16-17). Lab: CMUdict lookup with a rule fallback,
   number normalization, and an energy VAD with hysteresis scored by frame F1 on synthetic
   noisy mixtures.
8. **Alignment and language models: DTW and n-grams** (Sections 18 and 21). Lab: DTW with
   band constraints and template digit classification, and interpolated Kneser-Ney with
   perplexity.
9. **HMM-GMM recognition and evaluation** (Sections 19, 20, and 22). Lab: log-space
   forward, backward, Viterbi, and Baum-Welch, a GMM-HMM word recognizer on precomputed MFCCs,
   and WER, CER, and SI-SDR matched against `jiwer`.

If the course needs only eight lessons, merge lessons 3 and 4.

---

## 24. Bibliography

### Books and course texts

- Jurafsky, D., and Martin, J. H. *Speech and Language Processing*, 3rd ed. draft, release of
  August 19, 2026. https://web.stanford.edu/~jurafsky/slp3/
  - Ch. 2, Words and Tokens (includes edit distance): https://web.stanford.edu/~jurafsky/slp3/2.pdf
  - Ch. 3, N-gram Language Models: https://web.stanford.edu/~jurafsky/slp3/3.pdf
  - Ch. 15, Phonetics and Speech Feature Extraction: https://web.stanford.edu/~jurafsky/slp3/15.pdf
  - Ch. 16, Automatic Speech Recognition (CTC, encoder-decoder, HuBERT, WER; no HMM-GMM): https://web.stanford.edu/~jurafsky/slp3/16.pdf
  - Ch. 17, Text-to-Speech: https://web.stanford.edu/~jurafsky/slp3/17.pdf
  - Appendix A, Hidden Markov Models: https://web.stanford.edu/~jurafsky/slp3/A.pdf
  - Appendix C, Kneser-Ney Smoothing: https://web.stanford.edu/~jurafsky/slp3/C.pdf
  - Appendix D, Spelling Correction and the Noisy Channel: https://web.stanford.edu/~jurafsky/slp3/D.pdf
- Smith, J. O. *Mathematics of the Discrete Fourier Transform (DFT)*. https://ccrma.stanford.edu/~jos/mdft/
- Smith, J. O. *Introduction to Digital Filters*. https://ccrma.stanford.edu/~jos/filters/
- Smith, J. O. *Spectral Audio Signal Processing* (windows, STFT, COLA, phase vocoder). https://ccrma.stanford.edu/~jos/sasp/
- Smith, J. O. *Physical Audio Signal Processing* (tubes, reverberation). https://ccrma.stanford.edu/~jos/pasp/
- Smith, J. O. Digital Audio Resampling Home Page. https://ccrma.stanford.edu/~jos/resample/
- Oppenheim, A. V., and Schafer, R. W. *Discrete-Time Signal Processing*, 3rd ed. Pearson, 2010.
- Rabiner, L. R., and Schafer, R. W. *Theory and Applications of Digital Speech Processing*. Pearson, 2010.
- Rabiner, L. R., and Schafer, R. W. "Introduction to Digital Speech Processing." *Foundations and Trends in Signal Processing* 1(1-2), 2007. https://doi.org/10.1561/2000000001
- Müller, M. *Fundamentals of Music Processing*, 2nd ed. Springer, 2021; FMP notebooks: https://www.audiolabs-erlangen.de/FMP
- Gales, M., and Young, S. "The Application of Hidden Markov Models in Speech Recognition." *Foundations and Trends in Signal Processing* 1(3), 2008. https://doi.org/10.1561/2000000004

### Papers and standards

- Allen, J. B., and Rabiner, L. R. "A unified approach to short-time Fourier analysis and synthesis." *Proc. IEEE* 65(11), 1977.
- Allen, J. B., and Berkley, D. A. "Image method for efficiently simulating small-room acoustics." *JASA* 65(4), 1979.
- Bisani, M., and Ney, H. "Joint-sequence models for grapheme-to-phoneme conversion." *Speech Communication* 50(5), 2008.
- Bogert, B. P., Healy, M. J. R., and Tukey, J. W. "The quefrency alanysis of time series for echoes." 1963.
- Brown, J. C. "Calculation of a constant Q spectral transform." *JASA* 89(1), 1991.
- Chen, S. F., and Goodman, J. "An empirical study of smoothing techniques for language modeling." Harvard TR-10-98, 1998.
- Davis, S. B., and Mermelstein, P. "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." *IEEE TASSP* 28(4), 1980.
- de Cheveigné, A., and Kawahara, H. "YIN, a fundamental frequency estimator for speech and music." *JASA* 111(4), 1917-1930, 2002. https://pubs.aip.org/asa/jasa/article/111/4/1917/547221 (PDF mirror: http://recherche.ircam.fr/equipes/pcm/cheveign/pss/2002_JASA_YIN.pdf)
- Fletcher, H., and Munson, W. A. "Loudness, its definition, measurement and calculation." *JASA* 5, 1933.
- Flanagan, J. L., and Golden, R. M. "Phase vocoder." *Bell System Technical Journal* 45, 1966.
- Glasberg, B. R., and Moore, B. C. J. "Derivation of auditory filter shapes from notched-noise data." *Hearing Research* 47, 1990.
- Griffin, D., and Lim, J. "Signal estimation from modified short-time Fourier transform." *IEEE TASSP* 32(2), 1984.
- Harris, F. J. "On the use of windows for harmonic analysis with the discrete Fourier transform." *Proc. IEEE* 66(1), 1978.
- Heafield, K. "KenLM: Faster and smaller language model queries." WMT 2011. https://kheafield.com/code/kenlm/
- Hermansky, H. "Perceptual linear predictive (PLP) analysis of speech." *JASA* 87(4), 1990.
- ISO 226:2023, Acoustics: Normal equal-loudness-level contours. https://www.iso.org/standard/83117.html
- ITU-R BS.1770-5 (11/2023), Algorithms to measure audio programme loudness and true-peak audio level. https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
- ITU-T G.711, Pulse code modulation (PCM) of voice frequencies. https://www.itu.int/rec/T-REC-G.711
- ITU-T P.862, Perceptual evaluation of speech quality (PESQ). https://www.itu.int/rec/T-REC-P.862
- Kim, J. W., Salamon, J., Li, P., and Bello, J. P. "CREPE: A convolutional representation for pitch estimation." ICASSP 2018. https://arxiv.org/abs/1802.06182
- Kneser, R., and Ney, H. "Improved backing-off for m-gram language modeling." ICASSP 1995.
- Laroche, J., and Dolson, M. "Improved phase vocoder time-scale modification of audio." *IEEE Trans. Speech and Audio Processing* 7(3), 1999.
- Le Roux, J., Wisdom, S., Erdogan, H., and Hershey, J. R. "SDR - half-baked or well done?" ICASSP 2019. https://arxiv.org/abs/1811.02508
- Lee, K.-F., and Hon, H.-W. "Speaker-independent phone recognition using hidden Markov models." *IEEE TASSP* 37(11), 1989.
- Lipshitz, S. P., Wannamaker, R. A., and Vanderkooy, J. "Quantization and dither: A theoretical survey." *JAES* 40(5), 1992.
- Mauch, M., and Dixon, S. "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." ICASSP 2014.
- Mohri, M., Pereira, F., and Riley, M. "Weighted finite-state transducers in speech recognition." *Computer Speech and Language* 16(1), 2002.
- Noll, A. M. "Cepstrum pitch determination." *JASA* 41(2), 1967.
- Painter, T., and Spanias, A. "Perceptual coding of digital audio." *Proc. IEEE* 88(4), 2000.
- Perraudin, N., Balazs, P., and Søndergaard, P. L. "A fast Griffin-Lim algorithm." WASPAA 2013.
- Peterson, G. E., and Barney, H. L. "Control methods used in a study of the vowels." *JASA* 24(2), 1952.
- Povey, D., et al. "The Kaldi speech recognition toolkit." ASRU 2011. https://kaldi-asr.org/ ; graph recipe: https://kaldi-asr.org/doc/graph.html
- Rabiner, L. R. "A tutorial on hidden Markov models and selected applications in speech recognition." *Proc. IEEE* 77(2), 1989.
- RFC 6716, Definition of the Opus Audio Codec, 2012. https://www.rfc-editor.org/rfc/rfc6716
- RFC 9639, Free Lossless Audio Codec (FLAC), 2024. https://www.rfc-editor.org/rfc/rfc9639.html
- Sakoe, H., and Chiba, S. "Dynamic programming algorithm optimization for spoken word recognition." *IEEE TASSP* 26(1), 1978.
- Schörkhuber, C., and Klapuri, A. "Constant-Q transform toolbox for music processing." SMC 2010.
- Slaney, M. "Auditory Toolbox, Version 2." Interval Research TR 1998-010, 1998; "An efficient implementation of the Patterson-Holdsworth auditory filter bank," Apple TR 35, 1993.
- Sproat, R., et al. "Normalization of non-standard words." *Computer Speech and Language* 15(3), 2001.
- Sproat, R., and Jaitly, N. "RNN approaches to text normalization: A challenge." 2016. https://arxiv.org/abs/1611.00068
- Stevens, S. S., Volkmann, J., and Newman, E. B. "A scale for the measurement of the psychological magnitude pitch." *JASA* 8(3), 1937.
- Taal, C. H., Hendriks, R. C., Heusdens, R., and Jensen, J. "An algorithm for intelligibility prediction of time-frequency weighted noisy speech." *IEEE TASLP* 19(7), 2011. https://sps.ewi.tudelft.nl/pubs/Taal2011_1.pdf
- Traunmüller, H. "Analytical expressions for the tonotopic sensory scale." *JASA* 88(1), 1990.
- van den Oord, A., et al. "WaveNet: A generative model for raw audio." 2016. https://arxiv.org/abs/1609.03499
- Vincent, E., Gribonval, R., and Févotte, C. "Performance measurement in blind audio source separation." *IEEE TASLP* 14(4), 2006.
- Yao, K., and Zweig, G. "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion." 2015. https://arxiv.org/abs/1506.00196
- Young, S. J., Odell, J. J., and Woodland, P. C. "Tree-based state tying for high accuracy acoustic modelling." ARPA HLT Workshop, 1994.
- Zhang, Y., et al. "NeMo inverse text normalization: From development to production." 2021. https://arxiv.org/abs/2104.05055
- Zhu, J., Zhang, C., and Jurgens, D. "ByT5 model for massively multilingual grapheme-to-phoneme conversion." 2022. https://arxiv.org/abs/2204.03067
- Zwicker, E. "Subdivision of the audible frequency range into critical bands." *JASA* 33(2), 1961.

### Software sources consulted (2026-09-28)

- Whisper audio front end: https://github.com/openai/whisper/blob/main/whisper/audio.py
- Whisper large-v3 release discussion: https://github.com/openai/whisper/discussions/1762
- librosa source and 1.0 changelog: https://github.com/librosa/librosa and https://github.com/librosa/librosa/blob/main/docs/changelog.rst
- torchaudio future plan: https://github.com/pytorch/audio/issues/3902 ; releases: https://github.com/pytorch/audio/releases ; docs: https://docs.pytorch.org/audio/stable/torchaudio.html
- TorchCodec: https://github.com/meta-pytorch/torchcodec
- SciPy signal: https://docs.scipy.org/doc/scipy/reference/signal.html
- Silero VAD: https://github.com/snakers4/silero-vad
- py-webrtcvad: https://github.com/wiseman/py-webrtcvad
- CMUdict: https://github.com/cmusphinx/cmudict
- CREPE: https://github.com/marl/crepe
- pystoi: https://github.com/mpariente/pystoi
- jiwer: https://github.com/jitsi/jiwer
- Pyodide releases: https://github.com/pyodide/pyodide/releases

---

## 25. Items not independently verified

The following statements come from memory or secondary sources and were not checked against
a primary source during this research pass:

- Exact PESQ MOS-LQO mapping range (P.862.1). The -0.5 to 4.5 raw range is widely cited;
  the mapped range was not checked.
- Typical adult F0 ranges (85-180 Hz and 165-255 Hz) and Peterson-Barney formant values
  (quoted from memory of the 1952 table).
- The Terhardt threshold-in-quiet formula and temporal masking durations (standard
  textbook values).
- WebRTC VAD internals (six sub-bands, two-Gaussian GMMs) come from secondary write-ups, not
  the WebRTC C source.
- The typical LM scale "around 10" for GMM systems is a rule of thumb.
- Roll-off rates in the window table (-6 and -18 dB/octave) come from Harris (1978) as
  remembered; the main-lobe widths and peak side lobes were measured.
- Whether torchaudio 2.11 works with torch 2.14 in practice was not tested; the 2.11 release
  notes claim forward compatibility.
