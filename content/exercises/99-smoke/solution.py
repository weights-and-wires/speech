"""Draw the spectrogram of a tone sweep: the worked solution."""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def tone_sweep(f0: float, f1: float, sample_rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """A sine whose frequency rises linearly from `f0` to `f1` Hz."""
    t = np.arange(int(round(seconds * sample_rate))) / sample_rate
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * seconds))
    return amplitude * np.sin(phase)


def frame_signal(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Cut `x` into overlapping frames, shape (n_frames, frame_length)."""
    n_frames = 1 + (len(x) - frame_length) // hop
    index = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[index]


def magnitude_spectrogram(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """The magnitude of the short-time Fourier transform, shape (n_fft // 2 + 1, n_frames)."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = frame_signal(x, n_fft, hop) * window
    return np.abs(np.fft.rfft(frames, axis=1)).T


def peak_track(spectrogram: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """The frequency in Hz of the strongest bin in each frame, shape (n_frames,)."""
    return np.argmax(spectrogram, axis=0) * sample_rate / n_fft


if __name__ == "__main__":
    rate = 16000
    sweep = tone_sweep(200.0, 4000.0, rate, 1.0)
    play(sweep, rate, label="A sweep from 200 Hz to 4 kHz")

    n_fft, hop = 512, 128
    spec = magnitude_spectrogram(sweep, n_fft, hop)
    show_spectrogram(spec, rate, hop, label="Its spectrogram")

    track = peak_track(spec, rate, n_fft)
    times = (np.arange(track.size) * hop + n_fft / 2) / rate
    show_plot((times, track), label="The strongest frequency in each frame", xlabel="Time (s)", ylabel="Frequency (Hz)")
    print(f"{spec.shape[1]} frames; the peak rises from {track[0]:.0f} Hz to {track[-1]:.0f} Hz")
