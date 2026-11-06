"""Quantization and dither: the worked solution."""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Round `x` to the nearest of 2**bits levels, returned as floats on the grid.

    For signals in [-1, 1), the levels are a step of 2 ** (1 - bits) apart,
    and the integer codes run from -2 ** (bits - 1) to 2 ** (bits - 1) - 1, so
    the highest level is one step below +1. Samples above the top level are
    clipped to it. At 3 bits the step is 0.25, and 0.9 comes out as 0.75.
    """
    step = 2.0 ** (1 - bits)
    codes = np.clip(np.round(np.asarray(x, dtype=float) / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return codes * step


def predicted_snr_db(bits: int, amplitude: float = 1.0) -> float:
    """The signal-to-noise ratio, in dB, of a sine of this amplitude quantized to `bits` bits.

    A sine of amplitude A has power A**2 / 2, and rounding noise has power
    step**2 / 12, so the ratio is 10 * log10(1.5 * 4**bits) + 20 * log10(A).
    At full scale this is 6.02 N + 1.76 dB.
    """
    return float(10 * np.log10(1.5 * 4.0**bits) + 20 * np.log10(amplitude))


def tpdf_dither(x: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    """Add triangular dither, then quantize to `bits` bits, and return the quantized signal.

    The dither is the sum of two independent uniform draws on
    [-step / 2, step / 2], where step = 2 ** (1 - bits), drawn from `rng`. Its
    error has a mean of zero and a power of step**2 / 4 at every signal level.
    Call your own `quantize` on the dithered signal.
    """
    step = 2.0 ** (1 - bits)
    x = np.asarray(x, dtype=float)
    noise = rng.uniform(-step / 2, step / 2, x.shape) + rng.uniform(-step / 2, step / 2, x.shape)
    return quantize(x + noise, bits)


def _spectrogram(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    starts = np.arange(0, len(x) - n_fft + 1, hop)
    frames = x[starts[:, None] + np.arange(n_fft)[None, :]] * window
    return np.abs(np.fft.rfft(frames, axis=1)).T


if __name__ == "__main__":
    rate = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(2.5 * rate)) / rate

    # A plucked note: five harmonics of 440 Hz that die away, peaking at -12 dBFS.
    note = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 6)) * np.exp(-2.0 * t)
    note *= 0.25 / np.max(np.abs(note))
    play(note, rate, label="The original note")

    bits = 6
    plain = quantize(note, bits)
    dithered = tpdf_dither(note, bits, rng)
    for name, y in [("6 bits, no dither", plain), ("6 bits, TPDF dither", dithered)]:
        snr = 10 * np.log10(np.sum(note**2) / np.sum((y - note) ** 2))
        print(f"{name}: SNR {snr:.1f} dB")
    print(f"prediction for a full-scale sine at {bits} bits: {predicted_snr_db(bits):.2f} dB")
    play(plain, rate, label="6 bits, no dither: gritty, then it cuts out")
    play(dithered, rate, label="6 bits, TPDF dither: a steady hiss, and the note fades smoothly")
    show_spectrogram(_spectrogram(plain - note), rate, 256, label="Error without dither: harmonics of the note")
    show_spectrogram(_spectrogram(dithered - note), rate, 256, label="Error with dither: flat noise")

    # The SNR of a sine, measured at each bit depth, against the rule.
    n = np.arange(rate)
    sine = 0.99 * np.sin(2 * np.pi * 1234.5 * n / rate + 0.1)
    depths = np.arange(4, 17)
    measured = [10 * np.log10(np.sum(sine**2) / np.sum((quantize(sine, b) - sine) ** 2)) for b in depths]
    show_plot((depths, measured), (depths, [predicted_snr_db(int(b), 0.99) for b in depths]), labels=["measured", "predicted"], label="SNR of a sine against bits", xlabel="Bits per sample", ylabel="SNR (dB)")
