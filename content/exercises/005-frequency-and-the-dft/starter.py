"""Frequency and the DFT.

Fill in the four functions below, then press Run to find the 440 Hz and 1000 Hz
tones in a 25 ms frame from their bins. Press Check to test your code.

`dft_matrix` and `dft` return complex arrays with no scaling, the same
convention as numpy. Sample rates are in hertz, and `rfft_frequencies`
returns hertz.
"""

import numpy as np

from speechlab import play, show_plot


def dft_matrix(n: int) -> np.ndarray:
    """The n-by-n DFT matrix, complex, with W[k, m] = exp(-2 pi j k m / n).

    Row k holds the complex sinusoid that bin k of the DFT is compared against:
    it turns k times around the circle across the n samples. `dft_matrix(4)`
    holds the entries 1, -j, -1, and j.
    """
    raise NotImplementedError("dft_matrix")


def dft(x: np.ndarray) -> np.ndarray:
    """The DFT of a 1-D signal `x`, complex, shape (len(x),), computed as one product with `dft_matrix`.

    X[k] is the sum over n of x[n] * exp(-2 pi j k n / N). It equals
    `np.fft.fft(x)` to rounding, with no scaling on the forward transform.
    """
    raise NotImplementedError("dft")


def rdft(x: np.ndarray) -> np.ndarray:
    """The bins of `dft(x)` that a real signal needs, complex, shape (len(x) // 2 + 1,).

    For a real `x`, X[N - k] is the conjugate of X[k], so bins 0 to N // 2 hold
    everything. It equals `np.fft.rfft(x)` to rounding. Use your own `dft`.
    """
    raise NotImplementedError("rdft")


def rfft_frequencies(n_fft: int, sample_rate: float) -> np.ndarray:
    """The frequency in Hz of each of the n_fft // 2 + 1 bins `rdft` returns, shape (n_fft // 2 + 1,).

    Bin k sits at k * sample_rate / n_fft. For n_fft = 400 at 16000 Hz the
    bins are 0, 40, 80, ... 8000 Hz.
    """
    raise NotImplementedError("rfft_frequencies")


if __name__ == "__main__":
    x = np.array([1.0, 2.0, 0.0, -1.0])
    print("DFT of [1, 2, 0, -1]:", np.round(dft(x), 6))
    print("magnitudes:", np.round(np.abs(dft(x)), 4))

    # The running example, a 440 Hz tone with amplitude 0.5, plus a 1000 Hz tone
    # with amplitude 0.25, in one 400-sample frame (25 ms) at 16 kHz.
    rate, n = 16000, 400
    t = np.arange(n) / rate
    frame = 0.5 * np.cos(2 * np.pi * 440 * t) + 0.25 * np.cos(2 * np.pi * 1000 * t + 0.7)
    spectrum = rdft(frame)
    freqs = rfft_frequencies(n, rate)
    magnitude = np.abs(spectrum)
    print(f"bins: {len(freqs)}, spacing {freqs[1]:.0f} Hz, last bin {freqs[-1]:.0f} Hz")
    for k in np.argsort(magnitude)[::-1][:2]:
        print(f"bin {k} = {freqs[k]:.0f} Hz: magnitude {magnitude[k]:.1f}, phase {np.angle(spectrum[k]):.2f} rad")
    print(f"largest difference from np.fft.rfft: {np.max(np.abs(spectrum - np.fft.rfft(frame))):.1e}")
    play(np.tile(frame, 40), rate, label="The two tones: 440 Hz and, quieter, 1000 Hz")
    show_plot((freqs, magnitude), label="The spectrum of the 25 ms frame", xlabel="Frequency (Hz)", ylabel="|X[k]|", xlim=(0, 2000))
