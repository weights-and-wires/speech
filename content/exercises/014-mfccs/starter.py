"""MFCCs.

Turn a mel power spectrogram into MFCCs: decibels, an orthonormal DCT-II down
the mel axis, and the first 13 coefficients, plus HTK liftering. `mel_power` is
provided; it is the front end from the earlier chapters. Fill in the four
functions under "The exercise", then press Run to see the first 13 MFCCs of
three vowels and one frame rebuilt from them, and Check to test your code.
"""

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
    """The orthonormal DCT-II along axis 0.

    `x` has shape (M,) or (M, T). Return an array of the same shape whose row n
    is sqrt((2 - [n == 0]) / M) * sum over m of x[m] * cos(pi * n * (2m + 1) / (2M)).
    Build it from the formula; this equals scipy.fft.dct(x, type=2, norm="ortho", axis=0).
    """
    raise NotImplementedError


def power_to_db(S: np.ndarray, amin: float = 1e-10, top_db: float | None = 80.0) -> np.ndarray:
    """Power in decibels, same shape as `S`.

    Return 10 * log10(max(S, amin)). When `top_db` is not None, raise every value
    to at least the maximum of the whole result minus `top_db`. This is
    librosa.power_to_db with ref=1.0.
    """
    raise NotImplementedError


def mfcc(power_mel: np.ndarray, n_mfcc: int = 13, top_db: float | None = 80.0) -> np.ndarray:
    """MFCCs from a mel power matrix of shape (n_mels, T): shape (n_mfcc, T).

    Row n is the n-th orthonormal DCT-II coefficient of each frame's mel
    energies in dB, with `top_db` passed to `power_to_db`.
    """
    raise NotImplementedError


def lifter(c: np.ndarray, L: int = 22) -> np.ndarray:
    """HTK sinusoidal liftering of cepstra `c`, shape (n_ceps, T), same shape out.

    Multiply row n (n = 0, 1, ...) by 1 + (L / 2) * sin(pi * n / L), so row 0
    is unchanged.
    """
    raise NotImplementedError


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
