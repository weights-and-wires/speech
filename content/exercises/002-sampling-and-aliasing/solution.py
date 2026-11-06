"""Sampling and aliasing: the worked solution."""

from math import gcd

import numpy as np

from speechlab import play, show_plot, show_spectrogram


# Provided: you wrote this in the chapter "What sound is".
def sample_tone(
    frequency: float,
    sample_rate: float,
    seconds: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Samples of `amplitude * cos(2 pi f t + phase)` at `t = n / sample_rate`."""
    n = np.arange(int(round(seconds * sample_rate)))
    return amplitude * np.cos(2 * np.pi * frequency * n / sample_rate + phase)


def alias_frequency(frequency: float, sample_rate: float) -> float:
    """The frequency in [0, sample_rate / 2], in Hz, that the samples of a tone at `frequency` show.

    It is the distance from `frequency` to the nearest whole multiple of
    `sample_rate`: for a 7000 Hz tone at 10000 Hz it is 3000, and for a
    25000 Hz tone at 16000 Hz it is 7000.
    """
    return float(abs(frequency - sample_rate * np.round(frequency / sample_rate)))


def lowpass_taps(num_taps: int, cutoff: float, beta: float = 5.0) -> np.ndarray:
    """A windowed-sinc low-pass filter, shape (num_taps,), that sums to 1.

    `cutoff` is in cycles per sample, between 0 and 0.5. Tap `n` is
    `2 * cutoff * sinc(2 * cutoff * (n - (num_taps - 1) / 2))` times
    `np.kaiser(num_taps, beta)`, divided by the sum of all the taps so that a
    constant signal passes through unchanged. It equals
    `scipy.signal.firwin(num_taps, 2 * cutoff, window=("kaiser", beta))`.
    """
    m = np.arange(num_taps) - (num_taps - 1) / 2
    ideal = 2 * cutoff * np.sinc(2 * cutoff * m)
    h = ideal * np.kaiser(num_taps, beta)
    return h / h.sum()


def resample(x: np.ndarray, up: int, down: int, half_width: int = 10) -> np.ndarray:
    """Change the sample rate of `x` by the ratio `up / down`.

    Returns `ceil(len(x) * up / down)` samples, equal to
    `scipy.signal.resample_poly(x, up, down)`. After `up` and `down` are
    reduced by their greatest common divisor, the steps are: insert `up - 1`
    zeros after each sample, filter with `lowpass_taps` (cutoff
    `0.5 / max(up, down)` cycles per sample, `half_width * max(up, down)` taps
    on each side of its middle, gain `up`), then keep every `down`-th sample,
    starting after the filter's delay of `half` samples, so the output lines
    up with the input.
    """
    x = np.asarray(x, dtype=float)
    g = gcd(up, down)
    up, down = up // g, down // g
    if up == down == 1:
        return x.copy()
    rate = max(up, down)
    half = half_width * rate
    h = lowpass_taps(2 * half + 1, 0.5 / rate) * up

    stuffed = np.zeros(len(x) * up)
    stuffed[::up] = x
    filtered = np.convolve(stuffed, h)

    n_out = -(-len(x) * up // down)
    return filtered[half : half + n_out * down : down]


def _spectrogram(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block. The chapter on the short-time Fourier transform builds a proper one."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    starts = np.arange(0, len(x) - n_fft + 1, hop)
    frames = x[starts[:, None] + np.arange(n_fft)[None, :]] * window
    return np.abs(np.fft.rfft(frames, axis=1)).T


if __name__ == "__main__":
    # One tone that aliases: a 7 kHz cosine taken at 10 kHz has the samples of a 3 kHz one.
    high = sample_tone(7000, 10000, 1.0, amplitude=0.3)
    low = sample_tone(3000, 10000, 1.0, amplitude=0.3)
    print(f"7 kHz at 10 kHz is heard at {alias_frequency(7000, 10000):.0f} Hz")
    print(f"largest difference between the two sample lists: {np.max(np.abs(high - low)):.2e}")
    play(high, 10000, label="A 7 kHz cosine sampled at 10 kHz: you hear 3 kHz")

    # A sweep from 200 Hz to 16 kHz at 48 kHz, brought down to 16 kHz two ways.
    rate_in, seconds = 48000, 2.0
    t = np.arange(int(rate_in * seconds)) / rate_in
    sweep = 0.5 * np.sin(2 * np.pi * (200 * t + (16000 - 200) * t**2 / (2 * seconds)))

    naive = sweep[::3]
    proper = resample(sweep, 1, 3)
    play(naive, 16000, label="Every third sample: the sweep folds back down at 8 kHz")
    play(proper, 16000, label="Low-passed first: the sweep fades out at 8 kHz")
    show_spectrogram(_spectrogram(naive), 16000, 128, label="Every third sample")
    show_spectrogram(_spectrogram(proper), 16000, 128, label="Resampled with a windowed-sinc filter")

    # Where each input frequency lands after naive decimation to 16 kHz.
    inputs = np.linspace(0, 24000, 481)
    heard = np.array([alias_frequency(f, 16000) for f in inputs])
    show_plot((inputs, heard), label="The frequency the 16 kHz samples show", xlabel="Input frequency (Hz)", ylabel="Heard frequency (Hz)")
