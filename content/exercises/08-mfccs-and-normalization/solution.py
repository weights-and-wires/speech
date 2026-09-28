"""MFCCs and normalization: the worked solution."""

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from speechlab import play, show_image


# Provided: the mel power spectrogram from Hearing and the mel scale ----------

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


def deltas(c: np.ndarray, width: int = 2) -> np.ndarray:
    """HTK regression deltas along axis 1 (time), replicating the edge frames."""
    c = np.asarray(c, dtype=float)
    T = c.shape[1]
    padded = np.pad(c, ((0, 0), (width, width)), mode="edge")
    numerator = np.zeros_like(c)
    for theta in range(1, width + 1):
        numerator += theta * (padded[:, width + theta: width + theta + T] - padded[:, width - theta: width - theta + T])
    return numerator / (2.0 * sum(theta**2 for theta in range(1, width + 1)))


def cmvn(c: np.ndarray, variance: bool = True, eps: float = 1e-10) -> np.ndarray:
    """Per-utterance cepstral mean (and variance) normalization along axis 1."""
    c = np.asarray(c, dtype=float)
    out = c - c.mean(axis=1, keepdims=True)
    if variance:
        out = out / np.maximum(c.std(axis=1, keepdims=True), eps)
    return out


def mfcc_features(x: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """The classic 39-dimensional front end: liftered MFCCs, deltas, delta-deltas, then CMVN."""
    c = lifter(mfcc(mel_power(x, sample_rate), n_mfcc))
    d = deltas(c)
    dd = deltas(d)
    return cmvn(np.vstack([c, d, dd]))


if __name__ == "__main__":
    rate = 16000

    def vowel(f0, formants, seconds):
        n = int(seconds * rate)
        pulses = np.zeros(n)
        pulses[:: int(round(rate / f0))] = 1.0
        y = pulses
        for f, bw in formants:
            r = np.exp(-np.pi * bw / rate)
            y = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
        return y

    parts = [vowel(120, [(730, 90), (1090, 110), (2440, 170)], 0.45),
             vowel(120, [(270, 60), (2290, 100), (3010, 170)], 0.45),
             vowel(120, [(300, 60), (870, 90), (2240, 170)], 0.45)]
    speech = np.concatenate(parts)
    speech = 0.5 * speech / np.max(np.abs(speech)) + 0.002 * np.random.default_rng(0).standard_normal(len(speech))
    phone = sosfilt(butter(4, [300, 3400], btype="band", fs=rate, output="sos"), speech)
    play(speech, rate, label="Three vowels, /ɑ/ /i/ /u/, at 120 Hz")
    play(phone, rate, label="The same vowels through a 300 Hz to 3400 Hz telephone band")

    raw_clean = lifter(mfcc(mel_power(speech)))
    raw_phone = lifter(mfcc(mel_power(phone)))
    show_image(raw_clean, label="Liftered MFCCs of the clean vowels", xlabel="Frame (10 ms)", ylabel="Coefficient", origin="lower", colormap="viridis")
    feats = mfcc_features(speech)
    show_image(feats, label="39-dimensional features after CMVN", xlabel="Frame (10 ms)", ylabel="Feature", origin="lower")

    before = np.mean(np.abs(raw_clean - raw_phone))
    after = np.mean(np.abs(cmvn(raw_clean, variance=False) - cmvn(raw_phone, variance=False)))
    print(f"features: {feats.shape[0]} dimensions x {feats.shape[1]} frames")
    print(f"mean |clean - telephone| per MFCC: {before:.2f} before mean normalization, {after:.2f} after")
