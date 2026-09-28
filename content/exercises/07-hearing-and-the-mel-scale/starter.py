"""Hearing and the mel scale.

Build the two mel formulas, a triangular mel filterbank that matches librosa's,
and Whisper's exact log-mel front end. Fill in the functions below, then press
Run to hear why the mel scale exists and see Whisper's view of a sweep, and
Check to test your code.
"""

import numpy as np

from speechlab import play, show_image, show_spectrogram

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_SAMPLES = 30 * SAMPLE_RATE  # 480,000 samples, one 30-second Whisper chunk


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

    Place n_mels + 2 edge frequencies equally spaced in mel from `fmin` to
    `fmax` (default `sample_rate / 2`), and convert them back to Hz. Filter i
    rises linearly in Hz from edge i to edge i + 1, where it reaches 1, and
    falls back to 0 at edge i + 2. Evaluate it at each FFT bin frequency
    k * sample_rate / n_fft, for k = 0 ... n_fft // 2. With norm="slaney",
    multiply filter i by 2 / (edge[i + 2] - edge[i]); with norm=None, leave
    its peak at 1. This is `librosa.filters.mel`.
    """
    raise NotImplementedError


def power_spectrogram(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> np.ndarray:
    """The power spectrogram |STFT|^2, shape (n_fft // 2 + 1, 1 + len(x) // hop).

    Use the convention of the STFT lesson: pad n_fft // 2 samples at each end
    by reflection, cut frames every `hop` samples, multiply each by a periodic
    Hann window of length `n_fft`, and take the squared magnitude of its real
    FFT. Row k is frequency bin k; column t is the frame centered on sample
    t * hop of the unpadded signal.
    """
    raise NotImplementedError


def whisper_normalize(mel_power: np.ndarray) -> np.ndarray:
    """Whisper's log compression of a mel power matrix, same shape as the input.

    Take log10 after clamping every value to at least 1e-10. Raise every value
    to at least the maximum of the whole matrix minus 8. Map x to (x + 4) / 4.
    """
    raise NotImplementedError


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Zero-pad `audio` at the end, or cut it, to exactly `length` samples."""
    raise NotImplementedError


def whisper_log_mel(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Whisper's log-mel front end for 16 kHz audio, shape (n_mels, len(audio) // 160).

    Compute the power spectrogram with n_fft=400 and hop=160, drop its last
    frame, apply the Slaney-normalized Slaney-scale filterbank for 16 kHz and
    n_fft=400 with `n_mels` filters from 0 Hz to 8000 Hz, and pass the result
    through `whisper_normalize`.
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
