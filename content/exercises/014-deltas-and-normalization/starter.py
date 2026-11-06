"""Deltas and normalization.

Add motion and channel robustness to MFCCs: regression deltas, per-utterance
cepstral mean and variance normalization, and the classic 39-dimensional
feature stack. The front end from the earlier chapters (`mel_power`, `dct2_ortho`,
`power_to_db`, `mfcc`, and `lifter`) is provided in full. Fill in the three
functions under "The exercise", then press Run to hear clean and
telephone-band vowels and see how mean normalization cancels the channel, and
Check to test your code.
"""

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from speechlab import play, show_image

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


# The exercise -----------------------------------------------------------------

def deltas(c: np.ndarray, width: int = 2) -> np.ndarray:
    """HTK regression deltas along axis 1 (time), same shape as `c` (rows, T).

    d_t = sum over theta = 1..width of theta * (c_{t + theta} - c_{t - theta}),
    divided by 2 * sum of theta^2. Frames before the start and after the end
    repeat the first and last frame.
    """
    raise NotImplementedError


def cmvn(c: np.ndarray, variance: bool = True, eps: float = 1e-10) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalization along axis 1 (time).

    Subtract each row's mean over time. When `variance` is True, also divide by
    each row's standard deviation over time (np.std, which divides by T),
    using max(std, eps) so that a constant row comes out as zeros.
    """
    raise NotImplementedError


def mfcc_features(x: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """The classic 39-dimensional features of a 16 kHz signal, shape (3 * n_mfcc, 1 + len(x) // 160).

    The mel power comes from the provided `mel_power`. Rows 0 to n_mfcc - 1
    are the MFCCs liftered with L = 22, the next n_mfcc rows are their deltas,
    and the last n_mfcc rows are the deltas of those deltas, all with width 2.
    Every row is then normalized with CMVN, variance included.
    """
    raise NotImplementedError


if __name__ == "__main__":
    rate = 16000

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
