"""Log-mel spectrograms.

Build Whisper's exact log-mel front end. The mel filterbank and the power
spectrogram come from earlier chapters and are provided in full below. Fill in
the three functions under "The exercise", then press Run to see Whisper's view of a
sweep and what a quieter recording does to it, and Check to test your code.
"""

import numpy as np

from speechlab import play, show_spectrogram

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 30 * SAMPLE_RATE  # 480,000 samples, one 30-second Whisper chunk

# Provided: the mel filterbank and the power spectrogram, from earlier chapters ---

# The Slaney scale's constants, as librosa writes them.
F_SP = 200.0 / 3.0  # Hz per mel below 1 kHz
MIN_LOG_HZ = 1000.0
MIN_LOG_MEL = MIN_LOG_HZ / F_SP  # 15 mel
LOGSTEP = np.log(6.4) / 27.0  # natural-log step per mel above 1 kHz


def hz_to_mel(f, htk: bool = False) -> np.ndarray:
    """Convert frequencies in Hz to mel, on the HTK or the Slaney scale."""
    f = np.asarray(f, dtype=float)
    if htk:
        return 2595.0 * np.log10(1.0 + f / 700.0)
    linear = f / F_SP
    log = MIN_LOG_MEL + np.log(np.maximum(f, MIN_LOG_HZ) / MIN_LOG_HZ) / LOGSTEP
    return np.where(f >= MIN_LOG_HZ, log, linear)


def mel_to_hz(m, htk: bool = False) -> np.ndarray:
    """Convert mel back to Hz: the exact inverse of `hz_to_mel`."""
    m = np.asarray(m, dtype=float)
    if htk:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    linear = F_SP * m
    log = MIN_LOG_HZ * np.exp(LOGSTEP * (np.maximum(m, MIN_LOG_MEL) - MIN_LOG_MEL))
    return np.where(m >= MIN_LOG_MEL, log, linear)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | None = "slaney",
) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = sample_rate / 2.0
    bin_hz = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mels = np.linspace(hz_to_mel(fmin, htk), hz_to_mel(fmax, htk), n_mels + 2)
    edges = mel_to_hz(mels, htk)  # n_mels + 2 edge frequencies in Hz

    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bin_hz[None, :] - lower) / (center - lower)
    falling = (upper - bin_hz[None, :]) / (upper - center)
    weights = np.maximum(0.0, np.minimum(rising, falling))

    if norm == "slaney":
        weights *= (2.0 / (upper - lower))
    return weights


def power_spectrogram(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """|STFT|^2 with center=True reflect padding and a periodic Hann window."""
    x = np.asarray(x, dtype=float)
    padded = np.pad(x, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    spectrum = np.fft.rfft(padded[index] * window, axis=1)
    return (np.abs(spectrum) ** 2).T


# The exercise -----------------------------------------------------------------

def whisper_normalize(mel_power: np.ndarray) -> np.ndarray:
    """Whisper's log compression of a mel power matrix, same shape as the input.

    The result is log10 of the powers, floored at log10(1e-10) and at 8 below
    the largest log value anywhere in the matrix, then rescaled by
    (x + 4) / 4.
    """
    raise NotImplementedError


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Zero-pad `audio` at the end, or cut it, to exactly `length` samples."""
    raise NotImplementedError


def whisper_log_mel(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Whisper's log-mel front end for 16 kHz audio, shape (n_mels, len(audio) // 160).

    It matches `whisper.audio.log_mel_spectrogram(audio, n_mels)`: the power
    spectrogram with 400-point frames every 160 samples, without its last
    frame, the Slaney-scale, Slaney-normalized filterbank from 0 Hz to 8000 Hz
    (with `n_mels` filters), and `whisper_normalize`.
    """
    raise NotImplementedError


if __name__ == "__main__":
    rate = SAMPLE_RATE
    seconds = 2.0
    ts = np.arange(int(seconds * rate)) / rate
    sweep = 0.3 * np.sin(2 * np.pi * (50.0 * ts + (7900.0 - 50.0) * ts**2 / (2 * seconds)))
    play(sweep, rate, label="A linear sweep from 50 Hz to 7.9 kHz")

    features = whisper_log_mel(sweep)
    show_spectrogram(features, rate, HOP_LENGTH, label="Whisper log-mel of the sweep", db=False, y_axis="mel", fmin=0.0, fmax=8000.0, value_label="(log10 + 4) / 4")
    print(f"{len(sweep)} samples -> log-mel of shape {features.shape}, values from {features.min():.3f} to {features.max():.3f}")

    quiet = whisper_log_mel(0.1 * sweep)
    print(f"a clip 10 times quieter (20 dB down) shifts every feature by {np.mean(quiet - features):+.3f}")
    padded = whisper_log_mel(pad_or_trim(sweep))
    print(f"padded to 30 s -> shape {padded.shape}")
