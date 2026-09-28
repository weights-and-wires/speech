"""The short-time Fourier transform: the worked solution.

Conventions, used by every later lesson: frames are centered with reflect
padding, the window is a periodic Hann, and a spectrogram is shaped
(bins, frames) with row 0 the lowest frequency.
"""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def hann(n: int) -> np.ndarray:
    """The periodic Hann window of length n."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def frame_count(n_samples: int, frame_length: int, hop_length: int, center: bool = True) -> int:
    """How many frames the STFT of n_samples produces."""
    if center:
        return 1 + n_samples // hop_length
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(
    x: np.ndarray, frame_length: int, hop_length: int, center: bool = True, pad_mode: str = "reflect"
) -> np.ndarray:
    """Cut `x` into overlapping frames, shape (n_frames, frame_length)."""
    x = np.asarray(x, dtype=float)
    n_frames = frame_count(len(x), frame_length, hop_length, center)
    if center:
        x = np.pad(x, frame_length // 2, mode=pad_mode)
    index = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return x[index]


def stft(x: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray | None = None, center: bool = True) -> np.ndarray:
    """The complex STFT, shape (n_fft // 2 + 1, n_frames): bins by frames."""
    w = hann(n_fft) if window is None else np.asarray(window, dtype=float)
    frames = frame_signal(x, n_fft, hop_length, center)
    return np.fft.rfft(frames * w, axis=1).T


def spectrogram(x: np.ndarray, n_fft: int, hop_length: int, power: float = 2.0) -> np.ndarray:
    """|STFT| ** power: power=1 is the magnitude spectrogram, power=2 the power spectrogram."""
    return np.abs(stft(x, n_fft, hop_length)) ** power


def is_cola(window: np.ndarray, hop_length: int) -> bool:
    """Whether copies of `window` shifted by every multiple of `hop_length` sum to a constant."""
    window = np.asarray(window, dtype=float)
    sums = np.array([window[r::hop_length].sum() for r in range(hop_length)])
    return bool(np.allclose(sums, sums[0], rtol=1e-10, atol=1e-12))


def istft(
    X: np.ndarray,
    hop_length: int,
    window: np.ndarray | None = None,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """The least-squares inverse STFT: windowed overlap-add divided by the summed squared window."""
    n_fft = 2 * (X.shape[0] - 1)
    w = hann(n_fft) if window is None else np.asarray(window, dtype=float)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1) * w
    n_frames = frames.shape[0]
    total = n_fft + hop_length * (n_frames - 1)

    signal = np.zeros(total)
    weight = np.zeros(total)
    for m in range(n_frames):
        signal[m * hop_length : m * hop_length + n_fft] += frames[m]
        weight[m * hop_length : m * hop_length + n_fft] += w**2

    start = n_fft // 2 if center else 0
    if length is None:
        length = total - 2 * start
    signal = signal[start : start + length]
    weight = weight[start : start + length]
    if len(signal) < length or np.any(weight < 1e-11):
        raise ValueError(
            "the window and hop fail the NOLA condition: some samples are covered by no window weight"
        )
    return signal / weight


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
