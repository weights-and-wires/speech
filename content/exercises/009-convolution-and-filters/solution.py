"""Convolution and filters: the worked solution."""

import time

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The full linear convolution of x and h, len(x) + len(h) - 1 samples, one tap at a time."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    y = np.zeros(len(x) + len(h) - 1)
    for k, tap in enumerate(h):
        y[k : k + len(x)] += tap * x
    return y


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The same convolution through the FFT, zero-padded so nothing wraps around."""
    n_out = len(x) + len(h) - 1
    n_fft = 1 << (n_out - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)
    return y[:n_out]


def frequency_response(b: np.ndarray, a: np.ndarray, n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """H(e^{jw}) = B / A at n_points frequencies w = pi k / n_points, k = 0 .. n_points - 1."""
    w = np.pi * np.arange(n_points) / n_points
    z = np.exp(-1j * w)
    numerator = np.polyval(np.asarray(b, dtype=float)[::-1], z)
    denominator = np.polyval(np.asarray(a, dtype=float)[::-1], z)
    return w, numerator / denominator


def lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Run the difference equation a0 y[n] = sum b_k x[n-k] - sum_{k>=1} a_k y[n-k]."""
    b = np.asarray(b, dtype=float) / a[0]
    a = np.asarray(a, dtype=float) / a[0]
    x = np.asarray(x, dtype=float)
    y = np.zeros(len(x))
    for n in range(len(x)):
        acc = 0.0
        for k in range(min(len(b), n + 1)):
            acc += b[k] * x[n - k]
        for k in range(1, min(len(a), n + 1)):
            acc -= a[k] * y[n - k]
        y[n] = acc
    return y


def preemphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - alpha x[n - 1], with y[0] = x[0]."""
    x = np.asarray(x, dtype=float)
    return np.concatenate([x[:1], x[1:] - alpha * x[:-1]])


def _dry_signal(rate: int) -> np.ndarray:
    """Three vowel-like syllables and a clap."""
    t = np.arange(int(2.0 * rate)) / rate
    x = np.zeros_like(t)
    for start, f0 in [(0.1, 150), (0.45, 180), (0.8, 130)]:
        gate = (t >= start) & (t < start + 0.25)
        env = np.sin(np.pi * np.clip((t - start) / 0.25, 0, 1)) ** 2
        voice = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
        x += np.where(gate, env * voice, 0)
    clap = (t >= 1.3) & (t < 1.32)
    x[clap] += np.random.default_rng(1).standard_normal(clap.sum()) * np.exp(-np.arange(clap.sum()) / 60)
    return 0.3 * x / np.max(np.abs(x))


def _spectrogram(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block, as in the lesson on the STFT."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    starts = np.arange(0, len(x) - n_fft + 1, hop)
    return np.abs(np.fft.rfft(x[starts[:, None] + np.arange(n_fft)[None, :]] * window, axis=1)).T


if __name__ == "__main__":
    rate = 16000
    dry = _dry_signal(rate)
    play(dry, rate, label="Dry: three syllables and a clap")

    # A room reduced to its simplest form: the direct sound, then two echoes.
    echo = np.zeros(rate // 2)
    echo[[0, 3200, 6400]] = [1.0, 0.5, 0.25]
    wet = convolve_fft(dry, echo)
    wet *= 0.3 / np.max(np.abs(wet))
    play(wet, rate, label="Through an echo with taps at 0, 0.2, and 0.4 s")
    show_spectrogram(_spectrogram(wet), rate, 128, label="Each syllable now has two fainter copies behind it")

    x, h = dry[:rate], echo
    for name, func in [("direct", convolve_direct), ("FFT", convolve_fft)]:
        start = time.perf_counter()
        result = func(x, h)
        print(f"{name} convolution of {len(x)} by {len(h)} samples: {1000 * (time.perf_counter() - start):.1f} ms, "
              f"max difference from np.convolve {np.max(np.abs(result - np.convolve(x, h))):.1e}")

    # Feedback: y[n] = x[n] + 0.9 y[n - 1] rings, and smooths.
    ring = lfilter([1.0], [1.0, -0.9], np.array([1.0, 0, 0, 0, 2.0] + [0.0] * 3))
    print("y[n] = x[n] + 0.9 y[n-1] on [1, 0, 0, 0, 2, 0, 0, 0]:", np.round(ring, 4))
    w, H = frequency_response([1.0], [1.0, -0.9])
    show_plot((w * rate / (2 * np.pi), 20 * np.log10(np.abs(H))), label="Gain of y[n] = x[n] + 0.9 y[n-1]", xlabel="Frequency (Hz)", ylabel="Gain (dB)")
    smooth = lfilter([0.1], [1.0, -0.9], dry)
    play(0.3 * smooth / np.max(np.abs(smooth)), rate, label="Through the feedback filter: the high harmonics fade")

    emphasized = preemphasis(dry)
    w, H = frequency_response([1.0, -0.97], [1.0])
    print(f"pre-emphasis gain: {20 * np.log10(abs(H[0])):.2f} dB at 0 Hz, {20 * np.log10(abs(H[-1])):.2f} dB just below 8 kHz")
    freqs = np.fft.rfftfreq(len(dry), 1 / rate)
    spectra = []
    for y in (dry, emphasized):
        spectrum = np.abs(np.fft.rfft(y))
        spectra.append((freqs, 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-6))))
    show_plot(*spectra, labels=["before", "after pre-emphasis"], label="Spectrum of the dry signal", xlabel="Frequency (Hz)", ylabel="dB below the peak", ylim=(-100, 5))
    play(0.3 * emphasized / np.max(np.abs(emphasized)), rate, label="Pre-emphasized: thinner, brighter")
