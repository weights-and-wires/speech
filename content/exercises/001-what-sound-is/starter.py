"""What sound is.

Fill in the four functions below, then press Run to measure the running
example, a 440 Hz tone with amplitude 0.5. Press Check to test your code.

Every function takes a 1-D numpy array of samples with values in [-1, 1].
Levels are in dBFS, decibels relative to a full-scale amplitude of 1.0.
"""

import numpy as np

from speechlab import play, show_waveform


def rms(x: np.ndarray) -> float:
    """The root mean square of `x`, as a float: the square root of the mean of its squares.

    A full-scale square wave of +1 and -1 has an RMS of 1.0, and a sine of
    amplitude A over whole cycles has an RMS of A / sqrt(2).
    """
    raise NotImplementedError("rms")


def peak_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The level of the largest absolute sample, in dBFS, as a float.

    The peak is floored at `eps` before the logarithm, so silence returns
    20 * log10(eps) instead of minus infinity. Use 20, because a sample is an
    amplitude, not a power.
    """
    raise NotImplementedError("peak_dbfs")


def rms_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The level of the RMS of `x`, in dBFS, as a float, floored at `eps` like `peak_dbfs`.

    A full-scale sine reads -3.01 dBFS, and a sine of amplitude 0.5 reads -9.03 dBFS.
    """
    raise NotImplementedError("rms_dbfs")


def sample_tone(
    frequency: float,
    sample_rate: float,
    seconds: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """The samples of `amplitude * cos(2 pi f n / sample_rate + phase)`, shape (round(seconds * sample_rate),).

    Sample n is taken at time n / sample_rate, so the first sample is
    `amplitude * cos(phase)`.
    """
    raise NotImplementedError("sample_tone")


if __name__ == "__main__":
    rate = 16000

    # The running example: a 440 Hz tone with amplitude 0.5, for one second.
    a440 = sample_tone(440, rate, 1.0, amplitude=0.5)
    print(f"{len(a440)} samples, first five: {np.round(a440[:5], 4)}")
    print(f"peak {peak_dbfs(a440):.2f} dBFS, RMS {rms(a440):.4f}, RMS level {rms_dbfs(a440):.2f} dBFS")
    play(a440, rate, label="A 440 Hz tone, amplitude 0.5")
    show_waveform(a440[:160], rate, label="The first 10 ms of the tone")

    # The same tone at half the amplitude sits exactly 6.02 dB lower.
    quieter = 0.5 * a440
    print(f"half the amplitude: RMS level {rms_dbfs(quieter):.2f} dBFS, {rms_dbfs(a440) - rms_dbfs(quieter):.2f} dB lower")
    play(quieter, rate, label="The same tone at half the amplitude: 6.02 dB lower")

    # The buzz and the click from the chapter: same peak, different RMS.
    buzz = np.array([0.5, -0.5, 0.5, -0.5])
    click = np.array([0.5, 0.0, 0.0, 0.0])
    for name, clip in [("buzz", buzz), ("click", click)]:
        print(f"{name}: peak {peak_dbfs(clip):.2f} dBFS, RMS {rms(clip):.2f}, RMS level {rms_dbfs(clip):.2f} dBFS")
