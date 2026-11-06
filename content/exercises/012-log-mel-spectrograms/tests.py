"""Checks for the log-mel spectrogram exercise.

The CPU image has no Whisper package, so the references here are numbers
computed offline with openai-whisper's `whisper.audio.log_mel_spectrogram`,
hard-coded below, plus an independent pipeline built from torch.stft.
"""

import numpy as np
import torch

from speechcheck import check, close, fail, has, shape

RATE = 16000


def ref_slaney_mel(f):
    f = np.asarray(f, dtype=float)
    out = np.empty_like(f)
    low = f < 1000.0
    out[low] = 3.0 * f[low] / 200.0
    out[~low] = 15.0 + 27.0 * np.log(f[~low] / 1000.0) / np.log(6.4)
    return out


def ref_slaney_hz(m):
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m)
    low = m < 15.0
    out[low] = 200.0 * m[low] / 3.0
    out[~low] = 1000.0 * np.exp((m[~low] - 15.0) * np.log(6.4) / 27.0)
    return out


def ref_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, htk=False, norm="slaney"):
    """One filter at a time, each triangle drawn with np.interp."""
    fmax = sr / 2.0 if fmax is None else fmax
    if htk:
        to_mel = lambda f: 1127.0 * np.log1p(np.asarray(f, dtype=float) / 700.0)
        to_hz = lambda m: 700.0 * np.expm1(np.asarray(m, dtype=float) / 1127.0)
    else:
        to_mel, to_hz = ref_slaney_mel, ref_slaney_hz
    edges = to_hz(np.linspace(to_mel(np.array([fmin]))[0], to_mel(np.array([fmax]))[0], n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.interp(freqs, [lo, mid, hi], [0.0, 1.0, 0.0], left=0.0, right=0.0)
        if norm == "slaney":
            bank[i] *= 2.0 / (hi - lo)
    return bank


def ref_whisper(audio, n_mels):
    """Whisper's own pipeline in torch, with the independent filterbank above."""
    audio = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    window = torch.hann_window(400)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    mags = stft[..., :-1].abs() ** 2
    filters = torch.as_tensor(ref_filterbank(RATE, 400, n_mels).astype(np.float32))
    log_spec = torch.clamp(filters @ mags, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).numpy().astype(float)


def tone_plus_sweep():
    t = np.arange(8000) / RATE
    x = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * (200 * t + 3000 * t**2))
    return x.astype(np.float32)


@check("whisper_normalize takes log10, clamps at 1e-10, and maps x to (x + 4) / 4")
def test_normalize(sub):
    whisper_normalize = has(sub, "whisper_normalize")
    mel = np.array([[1.0, 1e-3, 0.0], [1e-5, 10.0, 1e-12]])
    got = np.asarray(whisper_normalize(mel), dtype=float)
    shape(got, (2, 3), label="the normalized matrix")
    close(got, [[1.0, 0.25, -0.75], [-0.25, 1.25, -0.75]], atol=1e-12, label="the worked 2 by 3 example")
    close(whisper_normalize(np.full((2, 2), 1e-4)), np.zeros((2, 2)), atol=1e-12, label="a flat matrix of 1e-4 (log10 = -4, so (x + 4) / 4 = 0)")


@check("whisper_normalize floors at the global maximum minus 8, not per frame")
def test_global_floor(sub):
    whisper_normalize = has(sub, "whisper_normalize")
    # Frame 0 is 60 dB quieter than frame 1. A per-frame floor would leave its
    # quietest band at 1e-12; the global floor lifts it to 10 ** (2 - 8).
    mel = np.array([[1e-4, 100.0], [1e-12, 1e-12]])
    got = whisper_normalize(mel)
    close(got[:, 0], [0.0, -0.5], atol=1e-12, label="a quiet frame, floored against the loudest value in the whole input")
    close(got[:, 1], [1.5, -0.5], atol=1e-12, label="the loud frame")
    power = np.array([[1.0, 4.0], [9.0, 16.0]])
    close(whisper_normalize(0.01 * power), whisper_normalize(power) - 0.5, atol=1e-12, label="a mel matrix 100 times weaker (power x 0.01), which lowers every feature by 0.5")


@check("pad_or_trim returns exactly the requested length, with zeros added at the end")
def test_pad_or_trim(sub):
    pad_or_trim = has(sub, "pad_or_trim")
    x = tone_plus_sweep()
    padded = np.asarray(pad_or_trim(x))
    shape(padded, (480000,), label="pad_or_trim of a half-second clip")
    close(padded[:8000], x, atol=0.0, label="the original samples, left in place")
    close(padded[8000:].max(initial=0.0), 0.0, label="the padding, which must be zeros at the end")
    short = np.asarray(pad_or_trim(np.arange(10.0), 6))
    close(short, np.arange(6.0), label="a clip cut to 6 samples")
    shape(np.asarray(pad_or_trim(np.ones(500000))), (480000,), label="pad_or_trim of a 31.25-second clip")


@check("whisper_log_mel matches Whisper's own log_mel_spectrogram on a tone plus a sweep")
def test_whisper(sub):
    whisper_log_mel = has(sub, "whisper_log_mel")
    x = tone_plus_sweep()
    got = np.asarray(whisper_log_mel(x), dtype=float)
    shape(got, (80, 50), label="the log-mel of 8000 samples (the last STFT frame is dropped)")
    # whisper.audio.log_mel_spectrogram(x) from openai-whisper, computed offline.
    close(got[[0, 10, 40, 79], [0, 10, 25, 49]], [1.1004275, 1.3002839, 1.1829932, -0.4505475], atol=1e-4,
          label="entries (0, 0), (10, 10), (40, 25), and (79, 49) against Whisper")
    close([got.min(), got.max(), got.mean()], [-0.59225166, 1.4077483, -0.32348186], atol=1e-4,
          label="the minimum, maximum, and mean against Whisper")
    close(got, ref_whisper(x, 80), atol=1e-4, label="every entry against Whisper's pipeline in torch")


@check("A 30-second chunk gives 3000 frames, and 128 mels match Whisper large-v3's front end")
def test_chunk_and_128(sub):
    pad_or_trim, whisper_log_mel = has(sub, "pad_or_trim"), has(sub, "whisper_log_mel")
    x = tone_plus_sweep()
    features = np.asarray(whisper_log_mel(pad_or_trim(x)), dtype=float)
    shape(features, (80, 3000), label="the log-mel of one 30-second chunk")
    close(features.mean(), -0.58729243, atol=1e-4, label="the mean of the padded chunk's log-mel, against Whisper")
    got = np.asarray(whisper_log_mel(x, n_mels=128), dtype=float)
    shape(got, (128, 50), label="the 128-mel log-mel")
    close(got[[0, 64, 127], [3, 20, 40]], [-0.3504529, -0.331746, -0.5465279], atol=1e-4,
          label="entries (0, 3), (64, 20), and (127, 40) against Whisper with n_mels=128")
