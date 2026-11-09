"""Hearing and the mel scale.

Build the two mel formulas and a triangular mel filterbank that matches
librosa's. Fill in the functions below, then press Run to hear why the mel
scale exists and see Whisper's 80 filters, and Check to test your code.
"""

import numpy as np

from speechlab import play, show_image

SAMPLE_RATE = 16000
N_FFT = 400


def hz_to_mel(f, htk: bool = False) -> np.ndarray:
    """Convert frequencies in Hz to mel.

    `f` is a scalar or an array of any shape; return a float array of the same
    shape. With `htk=True`, use 2595 * log10(1 + f / 700). With `htk=False`,
    use the Slaney scale: f / (200 / 3) below 1000 Hz, and
    15 + ln(f / 1000) / (ln(6.4) / 27) at and above 1000 Hz.
    """
    raise NotImplementedError


def mel_to_hz(m, htk: bool = False) -> np.ndarray:
    """Convert mel to Hz: the exact inverse of `hz_to_mel` on the same scale.

    `m` is a scalar or an array of any shape; return a float array of the same
    shape. On the Slaney scale, the break point is 15 mel (1000 Hz).
    """
    raise NotImplementedError


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | None = "slaney",
) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft // 2 + 1).

    The n_mels + 2 edges of the triangles are equally spaced in mel from
    `fmin` to `fmax` (default `sample_rate / 2`), and filter i spans edges i
    to i + 2 with its peak at edge i + 1. Each triangle is straight in Hz and
    is evaluated at the bin frequencies k * sample_rate / n_fft. With
    norm="slaney" every triangle has unit area in Hz; with norm=None every
    peak is 1. The result equals `librosa.filters.mel` with the same arguments.
    """
    raise NotImplementedError


if __name__ == "__main__":
    rate = SAMPLE_RATE
    t = np.arange(int(0.6 * rate)) / rate

    def step(f_a, f_b):
        half = len(t) // 2
        tone = np.where(np.arange(len(t)) < half, np.sin(2 * np.pi * f_a * t), np.sin(2 * np.pi * f_b * t))
        return 0.3 * tone * np.hanning(len(t)) ** 0.2

    play(step(100.0, 200.0), rate, label="A 100 Hz step low down: 100 Hz to 200 Hz")
    play(step(4000.0, 4100.0), rate, label="The same 100 Hz step high up: 4000 Hz to 4100 Hz")
    for f in (100.0, 200.0, 4000.0, 4100.0):
        print(f"{f:6.0f} Hz = {hz_to_mel(f):6.2f} Slaney mel = {hz_to_mel(f, htk=True):7.2f} HTK mel")

    filters = mel_filterbank(rate, N_FFT, 80)
    show_image(filters, label="Whisper's 80 Slaney mel filters", xlabel="FFT bin (40 Hz each)", ylabel="Mel filter", origin="lower")
    edges = mel_to_hz(np.linspace(0.0, hz_to_mel(rate / 2.0), 82))
    print(f"lowest filter edges: {np.round(edges[:4], 1)} Hz, FFT bins are {rate / N_FFT:.0f} Hz apart")
    print(f"FFT bins touched by filters 0, 1, 40, and 79: {[int((filters[i] > 0).sum()) for i in (0, 1, 40, 79)]}")
