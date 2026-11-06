"""The FFT and window functions.

Fill in the functions below, then press Run to time your FFT against the
matrix DFT and to see one frame of audio through four windows. Press Check
to test your code.
"""

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
    """The DFT of `x` by the recursive radix-2 Cooley-Tukey algorithm, complex, shape (len(x),).

    It costs O(n log n) and equals np.fft.fft(x) to rounding. Raise
    ValueError when len(x) is not a power of two.
    """
    raise NotImplementedError("fft_radix2")


def get_window(name: str, n: int, periodic: bool = True) -> np.ndarray:
    """A window of length n, shape (n,): "rectangular", "hann", "hamming", or "blackman".

    Each is a0 - a1 cos(2 pi i / P) + a2 cos(4 pi i / P) with the coefficients
    in COEFFICIENTS, where P is n for a periodic window and n - 1 for a
    symmetric one. A window of length 1 is [1.0]. Raise ValueError for any
    other name.
    """
    raise NotImplementedError("get_window")


def leakage(window: np.ndarray, oversample: int = 64) -> tuple[float, float]:
    """Measure a window's main lobe and side lobes from its zero-padded spectrum.

    The spectrum is the magnitude of the window's DFT zero-padded to
    `len(window) * oversample` points, scaled to 1 at 0 Hz. The first null,
    the edge of the main lobe, is the first local minimum after 0 Hz.

    Returns (half_width, side_lobe_db): the index of the first null divided by
    `oversample`, which is the main lobe's half-width in bins, and 20 log10 of
    the largest value from the null onward.
    """
    raise NotImplementedError("leakage")


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
