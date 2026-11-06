"""What sound is: the worked solution."""

import numpy as np

from speechlab import play, show_waveform


def rms(x: np.ndarray) -> float:
    """The root mean square of `x`: the square root of the mean of its squares."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def peak_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The largest absolute sample, in decibels relative to full scale (1.0)."""
    peak = float(np.max(np.abs(np.asarray(x, dtype=float))))
    return 20.0 * float(np.log10(max(peak, eps)))


def rms_dbfs(x: np.ndarray, eps: float = 1e-10) -> float:
    """The RMS level in decibels relative to a full-scale peak of 1.0."""
    return 20.0 * float(np.log10(max(rms(x), eps)))


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
