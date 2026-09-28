"""Draw the spectrogram of a tone sweep.

Fill in the four functions below, then press Run to hear the sweep and see
its spectrogram, and Check to test your code.
"""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def tone_sweep(f0: float, f1: float, sample_rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    """A sine whose frequency rises linearly from `f0` to `f1` Hz.

    Returns a float64 array of `round(seconds * sample_rate)` samples, with
    sample `n` taken at time `n / sample_rate` and peak value `amplitude`.
    """
    raise NotImplementedError


def frame_signal(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """Cut `x` into overlapping frames, shape (n_frames, frame_length).

    Frame `t` holds `x[t * hop : t * hop + frame_length]`. Keep only the
    frames that fit entirely inside `x`: `1 + (len(x) - frame_length) // hop`
    of them.
    """
    raise NotImplementedError


def magnitude_spectrogram(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """The magnitude of the short-time Fourier transform, shape (n_fft // 2 + 1, n_frames).

    Frame the signal with `frame_signal(x, n_fft, hop)`, multiply each frame by
    a periodic Hann window of length `n_fft`, and take the magnitude of its
    real FFT. Row `k` is the frequency `k * sample_rate / n_fft`.
    """
    raise NotImplementedError


def peak_track(spectrogram: np.ndarray, sample_rate: int, n_fft: int) -> np.ndarray:
    """The frequency in Hz of the strongest bin in each frame, shape (n_frames,)."""
    raise NotImplementedError


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
