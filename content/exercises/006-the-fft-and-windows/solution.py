"""The FFT and window functions: the worked solution."""

import time

import numpy as np

from speechlab import play, show_plot

COEFFICIENTS = {
    "rectangular": (1.0, 0.0, 0.0),
    "hann": (0.5, 0.5, 0.0),
    "hamming": (0.54, 0.46, 0.0),
    "blackman": (0.42, 0.5, 0.08),
}


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """The DFT of `x` by recursive radix-2 Cooley-Tukey. len(x) must be a power of two."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 0 or n & (n - 1):
        raise ValueError(f"fft_radix2 needs a power-of-two length, got {n}")
    if n == 1:
        return x.copy()
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    twiddled = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddled, even - twiddled])


def get_window(name: str, n: int, periodic: bool = True) -> np.ndarray:
    """A rectangular, Hann, Hamming, or Blackman window of length n."""
    if name not in COEFFICIENTS:
        raise ValueError(f"unknown window {name!r}; use one of {', '.join(COEFFICIENTS)}")
    if n == 1:
        return np.ones(1)
    a0, a1, a2 = COEFFICIENTS[name]
    period = n if periodic else n - 1
    phase = 2 * np.pi * np.arange(n) / period
    return a0 - a1 * np.cos(phase) + a2 * np.cos(2 * phase)


def leakage(window: np.ndarray, oversample: int = 64) -> tuple[float, float]:
    """(main-lobe half-width in bins, peak side lobe in dB) of a window's spectrum."""
    window = np.asarray(window, dtype=float)
    spectrum = np.abs(np.fft.rfft(window, len(window) * oversample))
    spectrum = spectrum / spectrum[0]
    k = 1
    while k < len(spectrum) - 1 and spectrum[k + 1] < spectrum[k]:
        k += 1
    return k / oversample, float(20 * np.log10(np.max(spectrum[k:])))


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 0.0, -1.0])
    print("radix-2 FFT of [1, 2, 0, -1]:", np.round(fft_radix2(x), 6))

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(1024)
    k = np.arange(1024)
    matrix = np.exp(-2j * np.pi * np.outer(k, k) / 1024)
    for name, func in [("matrix DFT", lambda s: matrix @ s), ("your radix-2 FFT", fft_radix2), ("np.fft.fft", np.fft.fft)]:
        start = time.perf_counter()
        result = func(signal)
        print(f"{name:>17}: {1000 * (time.perf_counter() - start):7.2f} ms for 1024 points, max error {np.max(np.abs(result - np.fft.fft(signal))):.1e}")

    # A loud tone between two bins and a quiet one 60 dB down, 1500 Hz higher.
    rate, n = 16000, 512
    t = np.arange(rate) / rate
    clip = 0.5 * np.sin(2 * np.pi * 1010 * t) + 0.0005 * np.sin(2 * np.pi * 2510 * t)
    play(clip, rate, label="A 1010 Hz tone, with a 2510 Hz one 60 dB quieter")

    freqs = np.fft.rfftfreq(n, 1 / rate)
    curves, names = [], []
    for name in COEFFICIENTS:
        w = get_window(name, n)
        spectrum = np.abs(np.fft.rfft(clip[:n] * w))
        db = 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-12))
        curves.append((freqs, db))
        half_width, side_lobe = leakage(w)
        names.append(f"{name} ({side_lobe:.1f} dB side lobes)")
        floor = db[np.argmin(np.abs(freqs - 2000))]
        quiet = db[(freqs > 2440) & (freqs < 2580)].max()
        print(f"{name:>11}: main lobe +-{half_width:.2f} bins, side lobes {side_lobe:.2f} dB; leakage at 2 kHz {floor:.1f} dB, quiet tone {quiet:.1f} dB")
    show_plot(*curves, labels=names, label="One 32 ms frame through four windows", xlabel="Frequency (Hz)", ylabel="dB below the peak", ylim=(-140, 5))
    show_plot(*[get_window(name, n) for name in COEFFICIENTS], labels=list(COEFFICIENTS), label="The windows", xlabel="Sample", ylabel="Weight")
