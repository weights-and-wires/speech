"""The source-filter model: the worked solution."""

import numpy as np
from scipy.signal import freqz, lfilter

from speechlab import play, show_plot

RATE = 16000


def impulse_train(f0: float, sample_rate: int, n_samples: int) -> np.ndarray:
    """Unit impulses at samples round(k * sample_rate / f0), k = 0, 1, 2, ..."""
    x = np.zeros(n_samples)
    period = sample_rate / f0
    positions = np.round(np.arange(0.0, n_samples, period)).astype(int)
    x[positions[positions < n_samples]] = 1.0
    return x


def resonator(frequency: float, bandwidth: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Klatt's two-pole resonator with unit gain at 0 Hz, as (b, a) for lfilter."""
    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2.0 * np.pi * frequency / sample_rate
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    b = np.array([a.sum()])
    return b, a


def synthesize_vowel(f0: float, formants, bandwidths, sample_rate: int, seconds: float) -> np.ndarray:
    """An impulse train at f0 through a cascade of formant resonators, peak 0.5."""
    y = impulse_train(f0, sample_rate, int(round(seconds * sample_rate)))
    for frequency, bandwidth in zip(formants, bandwidths):
        b, a = resonator(frequency, bandwidth, sample_rate)
        y = lfilter(b, a, y)
    return 0.5 * y / np.max(np.abs(y))


if __name__ == "__main__":
    # Average formants of three vowels for adult men, after Peterson and Barney (1952).
    vowels = {
        "/ɑ/ as in 'father'": [730, 1090, 2440],
        "/i/ as in 'heed'": [270, 2290, 3010],
        "/u/ as in 'who'd'": [300, 870, 2240],
    }
    bandwidths = [80, 100, 150]
    n_fft = 4096
    freqs = np.arange(n_fft // 2 + 1) * RATE / n_fft
    for name, targets in vowels.items():
        y = synthesize_vowel(120.0, targets, bandwidths, RATE, 0.8)
        play(y, RATE, label=f"{name}, F0 = 120 Hz")
        frame = y[4000:4000 + 2048]
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(2048), n_fft)) ** 2
        response = np.ones_like(freqs)
        for f, bw in zip(targets, bandwidths):
            b, a = resonator(f, bw, RATE)
            response = response * np.abs(freqz(b, a, worN=freqs, fs=RATE)[1])
        show_plot((freqs, 10 * np.log10(spectrum + 1e-12)), (freqs, 20 * np.log10(response) + 10 * np.log10(spectrum.max())),
                  labels=["harmonics of the vowel", "the three resonators (shifted to line up)"],
                  label=f"{name}: harmonics under the formant peaks", xlabel="Frequency (Hz)", ylabel="Power (dB)", xlim=(0, 4000))
        print(f"{name}: formants set at {targets} Hz, the source has a harmonic every 120 Hz")

    low = synthesize_vowel(100.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    high = synthesize_vowel(200.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    play(np.concatenate([low, high]), RATE, label="/ɑ/ at 100 Hz, then at 200 Hz: new pitch, same vowel")
