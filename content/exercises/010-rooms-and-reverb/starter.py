"""Rooms and reverb.

Fill in the functions below, then press Run to hear three syllables and a
clap dry, in a small room, and in a large one, and to see each room's decay
curve. Press Check to test your code.
"""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The full linear convolution of x and h through the FFT, len(x) + len(h) - 1 samples.

    Provided: you built it in Convolution and filters.
    """
    n_out = len(x) + len(h) - 1
    n_fft = 1 << (n_out - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:n_out]


def synthetic_rir(rt60: float, sample_rate: int, seconds: float | None = None, seed: int = 0) -> np.ndarray:
    """A synthetic room impulse response, shape (round(seconds * sample_rate),).

    `seconds` defaults to 1.5 * rt60. Sample n, at time t = n / sample_rate,
    is 0.1 * noise[n] * exp(-ln(1000) * t / rt60), where noise comes from
    np.random.default_rng(seed).standard_normal. The amplitude envelope falls
    by a factor of 1000, which is 60 dB of energy, in rt60 seconds. Sample 0
    is set to 1.0, the direct path from the source to the listener.
    """
    raise NotImplementedError("synthetic_rir")


def schroeder_decay_db(h: np.ndarray) -> np.ndarray:
    """The Schroeder decay curve of an impulse response, in dB, shape (len(h),).

    Entry n is 10 log10 of the energy from sample n to the end, sum of h**2,
    divided by the total energy. It starts at 0 dB and only falls.
    """
    raise NotImplementedError("schroeder_decay_db")


def estimate_rt60(h: np.ndarray, sample_rate: int, start_db: float = -5.0, stop_db: float = -25.0) -> float:
    """The reverberation time of h in seconds, measured from its Schroeder decay.

    The estimate comes from a least-squares straight line through the decay,
    in dB against time in seconds, over the samples where it lies between
    start_db and stop_db, extrapolated to a fall of 60 dB.
    """
    raise NotImplementedError("estimate_rt60")


def _dry_signal(rate: int) -> np.ndarray:
    """Three vowel-like syllables and a clap."""
    t = np.arange(int(2.0 * rate)) / rate
    x = np.zeros_like(t)
    for start, f0 in [(0.1, 150), (0.45, 180), (0.8, 130)]:
        gate = (t >= start) & (t < start + 0.25)
        env = np.sin(np.pi * np.clip((t - start) / 0.25, 0, 1)) ** 2
        voice = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
        x += np.where(gate, env * voice, 0)
    clap = (t >= 1.3) & (t < 1.32)
    x[clap] += np.random.default_rng(1).standard_normal(clap.sum()) * np.exp(-np.arange(clap.sum()) / 60)
    return 0.3 * x / np.max(np.abs(x))


def _spectrogram(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block, as in the lesson on the STFT."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    starts = np.arange(0, len(x) - n_fft + 1, hop)
    return np.abs(np.fft.rfft(x[starts[:, None] + np.arange(n_fft)[None, :]] * window, axis=1)).T


if __name__ == "__main__":
    rate = 16000
    dry = _dry_signal(rate)
    play(dry, rate, label="Dry: three syllables and a clap")
    show_spectrogram(_spectrogram(dry), rate, 128, label="The dry syllables")

    for rt60 in (0.3, 1.2):
        h = synthetic_rir(rt60, rate)
        wet = convolve_fft(dry, h)[: len(dry) + rate // 2]
        wet *= 0.3 / np.max(np.abs(wet))
        print(f"RT60 {rt60} s: {len(h)} taps, measured by Schroeder integration as {estimate_rt60(h, rate):.3f} s")
        play(wet, rate, label=f"Through a room with an RT60 of {rt60} s")
    show_spectrogram(_spectrogram(wet), rate, 128, label="The same syllables in the large room: each one smears into the next")

    curves = []
    for rt60 in (0.3, 1.2):
        h = synthetic_rir(rt60, rate, seconds=1.5)
        curves.append((np.arange(len(h)) / rate, schroeder_decay_db(h)))
    show_plot(*curves, labels=["RT60 0.3 s", "RT60 1.2 s"], label="Schroeder decay curves", xlabel="Time (s)", ylabel="Energy still to come (dB)", ylim=(-80, 2))
