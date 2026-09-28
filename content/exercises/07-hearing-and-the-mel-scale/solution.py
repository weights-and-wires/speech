"""Hearing and the mel scale: the worked solution."""

import numpy as np

from speechlab import play, show_image, show_spectrogram

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 30 * SAMPLE_RATE  # 480,000 samples, one 30-second Whisper chunk

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


def whisper_normalize(mel_power: np.ndarray) -> np.ndarray:
    """Whisper's log compression: log10, an 8-decade floor, then (x + 4) / 4."""
    log_spec = np.log10(np.maximum(mel_power, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Zero-pad `audio` at the end, or cut it, to exactly `length` samples."""
    audio = np.asarray(audio)
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def whisper_log_mel(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Whisper's log-mel front end, shape (n_mels, len(audio) // 160)."""
    power = power_spectrogram(audio, N_FFT, HOP_LENGTH)[:, :-1]  # drop the last frame
    filters = mel_filterbank(SAMPLE_RATE, N_FFT, n_mels)
    return whisper_normalize(filters @ power)


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

    seconds = 2.0
    ts = np.arange(int(seconds * rate)) / rate
    sweep = 0.3 * np.sin(2 * np.pi * (50.0 * ts + (7900.0 - 50.0) * ts**2 / (2 * seconds)))
    play(sweep, rate, label="A linear sweep from 50 Hz to 7.9 kHz")

    filters = mel_filterbank(rate, N_FFT, 80)
    show_image(filters, label="Whisper's 80 Slaney mel filters", xlabel="FFT bin (40 Hz each)", ylabel="Mel filter", origin="lower")

    features = whisper_log_mel(sweep)
    show_spectrogram(features, rate, HOP_LENGTH, label="Whisper log-mel of the sweep", db=False, y_axis="mel", fmin=0.0, fmax=8000.0, value_label="(log10 + 4) / 4")
    print(f"{len(sweep)} samples -> log-mel of shape {features.shape}, values from {features.min():.3f} to {features.max():.3f}")
    padded = whisper_log_mel(pad_or_trim(sweep))
    print(f"padded to 30 s -> shape {padded.shape}")
