"""The short-time Fourier transform.

Fill in the functions below, then press Run to see one signal through a short
and a long window, rebuild it from its STFT, and hear the result. Press Check
to test your code.

Conventions, used by every later chapter: frames are centered with reflect
padding, the window is a periodic Hann, and a spectrogram is shaped
(bins, frames) with row 0 the lowest frequency.
"""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def hann(n: int) -> np.ndarray:
    """The periodic Hann window of length n. Provided: you built it in The FFT and window functions."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def frame_count(n_samples: int, frame_length: int, hop_length: int, center: bool = True) -> int:
    """How many frames the STFT of `n_samples` samples produces.

    With `center=True` the signal is padded by frame_length // 2 at each end,
    so there are 1 + n_samples // hop_length frames. With `center=False`
    there are 1 + (n_samples - frame_length) // hop_length.
    """
    raise NotImplementedError("frame_count")


def frame_signal(
    x: np.ndarray, frame_length: int, hop_length: int, center: bool = True, pad_mode: str = "reflect"
) -> np.ndarray:
    """Cut `x` into overlapping frames, shape (frame_count(...), frame_length).

    With `center=True`, `x` is padded by frame_length // 2 samples at each end
    in `pad_mode` (a mode name np.pad accepts), so frame m is centered on
    sample m * hop_length of the original. With `center=False`, frame m starts
    at sample m * hop_length, and only whole frames are kept.
    """
    raise NotImplementedError("frame_signal")


def stft(x: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray | None = None, center: bool = True) -> np.ndarray:
    """The complex short-time Fourier transform, shape (n_fft // 2 + 1, n_frames).

    Frames are cut as `frame_signal` cuts them, and `window` defaults to a
    periodic Hann of length n_fft. The result is bins by frames: row k is the
    frequency k * sample_rate / n_fft, and column m is frame m. It equals
    torch.stft(..., center=True, pad_mode="reflect", return_complex=True) with
    the same window.
    """
    raise NotImplementedError("stft")


def spectrogram(x: np.ndarray, n_fft: int, hop_length: int, power: float = 2.0) -> np.ndarray:
    """The magnitude of the STFT raised to `power`, shape (n_fft // 2 + 1, n_frames).

    power=1.0 gives the magnitude spectrogram, power=2.0 the power spectrogram.
    """
    raise NotImplementedError("spectrogram")


def is_cola(window: np.ndarray, hop_length: int) -> bool:
    """Whether `window` satisfies the constant overlap-add condition at this hop.

    Copies of the window shifted by every multiple of hop_length must sum to
    the same value at every sample, to a relative tolerance of 1e-10.
    """
    raise NotImplementedError("is_cola")


def istft(
    X: np.ndarray,
    hop_length: int,
    window: np.ndarray | None = None,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """The least-squares inverse STFT of X, shape (n_fft // 2 + 1, n_frames), as a 1-D signal.

    n_fft is 2 * (X.shape[0] - 1), and `window` defaults to a periodic Hann.
    The result is the signal whose STFT is closest to X in the least-squares
    sense (Griffin and Lim, 1984), so it inverts `stft` exactly.

    With `center=True`, the padding is removed. Return `length`
    samples, or hop_length * (n_frames - 1) when `length` is None. Raise
    ValueError when any returned sample has a summed squared window below
    1e-11, or when fewer than `length` samples exist.
    """
    raise NotImplementedError("istft")


def _test_signal(rate: int) -> np.ndarray:
    """A sweep, two tones 80 Hz apart, and two clicks 20 ms apart."""
    t = np.arange(rate) / rate
    x = 0.25 * np.sin(2 * np.pi * (300 * t + (6000 - 300) * t**2 / 2))
    x += 0.12 * np.sin(2 * np.pi * 2000 * t) + 0.12 * np.sin(2 * np.pi * 2080 * t)
    for at in (0.5, 0.52):
        x[int(at * rate) : int(at * rate) + 4] += 0.5 * np.array([0.5, -1, 1, -0.5])
    fade = np.minimum(1, np.minimum(t, t[::-1]) / 0.03)
    return x * fade


if __name__ == "__main__":
    rate = 16000
    x = _test_signal(rate)
    play(x, rate, label="A sweep, two tones at 2000 and 2080 Hz, and two clicks 20 ms apart")

    for n_fft in (128, 2048):
        hop = n_fft // 4
        S = spectrogram(x, n_fft, hop)
        print(f"n_fft={n_fft} ({1000 * n_fft / rate:.0f} ms), hop={hop}: {S.shape[0]} bins x {S.shape[1]} frames, "
              f"bins {rate / n_fft:.1f} Hz apart; formula says {frame_count(len(x), n_fft, hop)} frames")
        show_spectrogram(S, rate, hop, power=True, label=f"{n_fft}-point window: {'the clicks separate' if n_fft == 128 else 'the tones separate'}")

    X = stft(x, 512, 128)
    y = istft(X, 128, length=len(x))
    print(f"istft(stft(x)) differs from x by at most {np.max(np.abs(y - x)):.2e}")
    play(y, rate, label="Rebuilt from its STFT by the least-squares inverse")

    print(f"Whisper: 30 s at 16 kHz with n_fft=400, hop=160 gives {frame_count(480000, 400, 160)} frames; it keeps 3000")
    for n_fft, hop in [(400, 200), (400, 100), (400, 160)]:
        print(f"periodic Hann {n_fft}, hop {hop}: COLA {is_cola(hann(n_fft), hop)}")

    w = hann(400)
    curves = []
    for hop in (100, 160):
        total = np.zeros(400 + 160 * 12)
        for m in range(0, len(total) - 400 + 1, hop):
            total[m : m + 400] += w
        curves.append(total[400:-400])
    show_plot(*curves, labels=["hop 100: COLA", "hop 160: not COLA"], label="Overlap-added Hann windows, 400 samples", xlabel="Sample", ylabel="Sum of windows")
