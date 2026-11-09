"""MFCCs: the worked solution."""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_image, show_plot

# Provided: the mel power spectrogram, from the earlier chapters ----------

def mel_power(x: np.ndarray, sample_rate: int = 16000, n_fft: int = 400, hop: int = 160, n_mels: int = 40) -> np.ndarray:
    """Slaney mel power spectrogram, shape (n_mels, 1 + len(x) // hop). Provided."""
    x = np.asarray(x, dtype=float)
    padded = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    power = (np.abs(np.fft.rfft(padded[index] * window, axis=1)) ** 2).T

    def to_mel(f):
        f = np.asarray(f, dtype=float)
        return np.where(f >= 1000.0, 15.0 + 27.0 * np.log(np.maximum(f, 1000.0) / 1000.0) / np.log(6.4), 3.0 * f / 200.0)

    def to_hz(m):
        m = np.asarray(m, dtype=float)
        return np.where(m >= 15.0, 1000.0 * np.exp((np.maximum(m, 15.0) - 15.0) * np.log(6.4) / 27.0), 200.0 * m / 3.0)

    edges = to_hz(np.linspace(0.0, to_mel(sample_rate / 2.0), n_mels + 2))
    bins = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.maximum(0.0, np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid))) * (2.0 / (hi - lo))
    return bank @ power



def vowel(f0, formants, seconds, rate=16000):
    """An impulse train through formant resonators, for the Run block."""
    n = int(seconds * rate)
    pulses = np.zeros(n)
    pulses[:: int(round(rate / f0))] = 1.0
    y = pulses
    for f, bw in formants:
        r = np.exp(-np.pi * bw / rate)
        y = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
    return y


# The exercise -----------------------------------------------------------------

def dct2_ortho(x: np.ndarray) -> np.ndarray:
    """Orthonormal DCT-II along axis 0. `x` is (M,) or (M, T); the result has the same shape."""
    x = np.asarray(x, dtype=float)
    M = x.shape[0]
    n = np.arange(M)[:, None]
    m = np.arange(M)[None, :]
    basis = np.cos(np.pi * n * (2 * m + 1) / (2 * M))
    scale = np.full((M, 1), np.sqrt(2.0 / M))
    scale[0] = np.sqrt(1.0 / M)
    return (scale * basis) @ x


def power_to_db(S: np.ndarray, amin: float = 1e-10, top_db: float | None = 80.0) -> np.ndarray:
    """10 log10 of a power matrix, floored at the global maximum minus `top_db`."""
    db = 10.0 * np.log10(np.maximum(np.asarray(S, dtype=float), amin))
    if top_db is not None:
        db = np.maximum(db, db.max() - top_db)
    return db


def mfcc(power_mel: np.ndarray, n_mfcc: int = 13, top_db: float | None = 80.0) -> np.ndarray:
    """MFCCs from a mel power matrix (n_mels, T): shape (n_mfcc, T)."""
    return dct2_ortho(power_to_db(power_mel, top_db=top_db))[:n_mfcc]


def lifter(c: np.ndarray, L: int = 22) -> np.ndarray:
    """HTK sinusoidal liftering: row n is scaled by 1 + (L / 2) sin(pi n / L)."""
    c = np.asarray(c, dtype=float)
    n = np.arange(c.shape[0])
    weights = 1.0 + (L / 2.0) * np.sin(np.pi * n / L)
    return c * weights[:, None]


if __name__ == "__main__":
    rate = 16000
    parts = [vowel(120, [(730, 90), (1090, 110), (2440, 170)], 0.45),
             vowel(120, [(270, 60), (2290, 100), (3010, 170)], 0.45),
             vowel(120, [(300, 60), (870, 90), (2240, 170)], 0.45)]
    speech = np.concatenate(parts)
    speech = 0.5 * speech / np.max(np.abs(speech)) + 0.002 * np.random.default_rng(0).standard_normal(len(speech))
    play(speech, rate, label="Three vowels, /ɑ/ /i/ /u/, at 120 Hz")

    mel = mel_power(speech)
    db = power_to_db(mel)
    coeffs = mfcc(mel, n_mfcc=13)
    show_image(coeffs, label="The first 13 MFCCs of the three vowels", xlabel="Frame (10 ms)", ylabel="Coefficient", origin="lower", colormap="viridis")

    # Rebuild one /ɑ/ frame from its first 13 coefficients and compare with the log-mel.
    frame = 20
    basis = dct2_ortho(np.eye(db.shape[0]))  # (M, M): row n is the n-th DCT basis function
    full = dct2_ortho(db)[:, frame]
    kept = np.zeros_like(full)
    kept[:13] = full[:13]
    show_plot(db[:, frame], basis.T @ kept, labels=["40 log-mel bands (dB)", "rebuilt from 13 coefficients"],
              label="One /ɑ/ frame and its 13-coefficient envelope", xlabel="Mel band", ylabel="Power (dB)")
    print(f"mfcc shape: {coeffs.shape}; the energy kept by 13 of 40 coefficients: {np.sum(kept**2) / np.sum(full**2):.4f}")
    print(f"the lifter weights run from {lifter(np.ones((13, 1)))[0, 0]:.2f} to {lifter(np.ones((13, 1))).max():.2f}")
