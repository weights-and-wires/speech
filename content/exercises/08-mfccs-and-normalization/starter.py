"""MFCCs and normalization.

Turn a mel power spectrogram into the classic 39-dimensional speech features:
MFCCs by an orthonormal DCT-II, sinusoidal liftering, regression deltas, and
per-utterance cepstral mean and variance normalization. `mel_power` is
provided; it is the front end from Hearing and the mel scale. Fill in the other
functions, then press Run to hear clean and telephone-band vowels and see how
mean normalization cancels the channel, and Check to test your code.
"""

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
