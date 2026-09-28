"""Filters, convolution, and reverb.

Fill in the functions below, then press Run to hear three syllables in a
small room and a large one, see their decay curves, time two ways of
convolving, and hear pre-emphasis. Press Check to test your code.
"""

import time

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def convolve_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The full linear convolution y[n] = sum_k h[k] x[n - k], shape (len(x) + len(h) - 1,).

    Computed directly in the time domain, without the FFT.
    """
    raise NotImplementedError("convolve_direct")


def convolve_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The same full linear convolution, computed through the FFT, shape (len(x) + len(h) - 1,).

    Use an FFT length that is a power of two, with enough zero padding that
    no circular wrap-around can occur. The result equals `convolve_direct` to
    rounding error.
    """
    raise NotImplementedError("convolve_fft")


def frequency_response(b: np.ndarray, a: np.ndarray, n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """The response H(e^{jw}) of the filter with coefficients b and a, as (w, H).

    w is n_points frequencies in radians per sample, pi * k / n_points for
    k = 0 .. n_points - 1. H is complex: sum_k b[k] e^{-jwk} divided by
    sum_k a[k] e^{-jwk}. This matches scipy.signal.freqz(b, a, worN=n_points).
    """
    raise NotImplementedError("frequency_response")


def lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter x with the difference equation, shape (len(x),).

    a[0] y[n] = sum_{k=0}^{len(b)-1} b[k] x[n - k] - sum_{k=1}^{len(a)-1} a[k] y[n - k],
    with x and y taken as 0 before the start. Divide b and a by a[0] first.
    This matches scipy.signal.lfilter(b, a, x).
    """
    raise NotImplementedError("lfilter")


def preemphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - alpha * x[n - 1] for n >= 1, and y[0] = x[0]. Shape (len(x),)."""
    raise NotImplementedError("preemphasis")


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

    x, h = dry[:rate], synthetic_rir(0.3, rate)
    for name, func in [("direct", convolve_direct), ("FFT", convolve_fft)]:
        start = time.perf_counter()
        func(x, h)
        print(f"{name} convolution of {len(x)} by {len(h)} samples: {1000 * (time.perf_counter() - start):.1f} ms")

    emphasized = preemphasis(dry)
    w, H = frequency_response([1.0, -0.97], [1.0])
    print(f"pre-emphasis gain: {20 * np.log10(abs(H[0])):.2f} dB at 0 Hz, {20 * np.log10(abs(H[-1])):.2f} dB just below 8 kHz")
    freqs = np.fft.rfftfreq(len(dry), 1 / rate)
    spectra = []
    for y in (dry, emphasized):
        spectrum = np.abs(np.fft.rfft(y))
        spectra.append((freqs, 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-6))))
    show_plot(*spectra, labels=["before", "after pre-emphasis"], label="Spectrum of the dry signal", xlabel="Frequency (Hz)", ylabel="dB below the peak", ylim=(-100, 5))
    play(0.3 * emphasized / np.max(np.abs(emphasized)), rate, label="Pre-emphasized: thinner, brighter")
