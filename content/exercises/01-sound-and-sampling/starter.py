"""Sound and sampling.

Fill in the functions below, then press Run to hear an alias and a sweep that
folds back on itself, and Check to test your code.

Every signal is a 1-D float array of samples. A frequency is in hertz, a
sample rate in samples per second, and a level in decibels.
"""

from math import gcd

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def rms(x: np.ndarray) -> float:
    """The root mean square of `x`, as a float.

    A sine of amplitude A, measured over whole periods, has an RMS of
    A / sqrt(2).
    """
    raise NotImplementedError("rms")


def peak_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The peak level of `x` in dBFS: 20 log10 of its largest absolute sample.

    Full scale is 1.0, so a peak of 1.0 is 0 dBFS and a peak of 0.5 is about
    -6.02 dBFS. Floor the peak at `eps` before the log, so silence returns
    20 log10(eps) instead of minus infinity.
    """
    raise NotImplementedError("peak_dbfs")


def rms_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The RMS level of `x` in dBFS: 20 log10 of `rms(x)`, floored at `eps`.

    This is the peak-referenced convention: a full-scale sine reads -3.01 dBFS.
    """
    raise NotImplementedError("rms_dbfs")


def sample_tone(
    frequency: float,
    sample_rate: float,
    seconds: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Sample the continuous tone `amplitude * cos(2 pi frequency t + phase)`.

    Returns a float array of `round(seconds * sample_rate)` samples, where
    sample `n` is the tone's value at time `t = n / sample_rate` seconds.
    """
    raise NotImplementedError("sample_tone")


def alias_frequency(frequency: float, sample_rate: float) -> float:
    """The frequency, in Hz, that the samples of a tone at `frequency` show.

    The result lies in [0, sample_rate / 2]. A tone already below half the
    sample rate is unchanged, and a 7000 Hz tone at 10000 Hz shows as 3000 Hz.
    """
    raise NotImplementedError("alias_frequency")


def lowpass_taps(num_taps: int, cutoff: float, beta: float = 5.0) -> np.ndarray:
    """A windowed-sinc low-pass FIR filter, shape (num_taps,).

    `cutoff` is in cycles per sample, between 0 and 0.5 (0.5 is the Nyquist
    frequency). The filter is symmetric about its middle tap, uses a Kaiser
    window with shape `beta`, and has a gain of exactly 1 at 0 Hz. It matches
    `scipy.signal.firwin(num_taps, 2 * cutoff, window=("kaiser", beta))`.
    """
    raise NotImplementedError("lowpass_taps")


def resample(x: np.ndarray, up: int, down: int, half_width: int = 10) -> np.ndarray:
    """Change the sample rate of `x` by the ratio `up / down`.

    Returns `ceil(len(x) * up / down)` samples, equal to
    `scipy.signal.resample_poly(x, up, down)`. After `up` and `down` are
    reduced by their greatest common divisor, the anti-aliasing filter is a
    `lowpass_taps` filter with `half_width * max(up, down)` taps on each side
    of its middle, cut off at the lower of the two Nyquist frequencies. The
    output lines up with the input: the filter's delay is removed.
    """
    raise NotImplementedError("resample")


def _spectrogram(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block. Lesson 04 builds a proper one."""
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
    print(f"the sweep peaks at {peak_dbfs(sweep):.2f} dBFS, with an RMS level of {rms_dbfs(sweep):.2f} dBFS")

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
