"""The source-filter model.

Build a vowel the way the vocal tract does: a pulse train for the vocal folds,
and one two-pole resonator for each formant. Fill in the three functions below,
then press Run to hear three vowels, see where their harmonics sit under the
formant peaks, and hear one vowel at two pitches. Press Check to test your
code.
"""

import numpy as np
from scipy.signal import freqz, lfilter

from speechlab import play, show_plot

RATE = 16000


def impulse_train(f0: float, sample_rate: int, n_samples: int) -> np.ndarray:
    """A glottal source: unit impulses one period apart, shape (n_samples,).

    Put a 1.0 at every sample round(k * sample_rate / f0) for k = 0, 1, 2, ...
    that is below n_samples, and 0.0 everywhere else.
    """
    raise NotImplementedError


def resonator(frequency: float, bandwidth: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """One formant as Klatt's two-pole resonator, returned as (b, a) for lfilter.

    With r = exp(-pi * bandwidth / sample_rate) and
    theta = 2 * pi * frequency / sample_rate, the denominator is
    a = [1, -2 r cos(theta), r^2]. The numerator b is the one-element array
    [sum(a)], which makes the gain at 0 Hz exactly 1.
    """
    raise NotImplementedError


def synthesize_vowel(f0: float, formants, bandwidths, sample_rate: int, seconds: float) -> np.ndarray:
    """A vowel from the source-filter model, shape (round(seconds * sample_rate),).

    The source is an impulse train at `f0`, and the filter is a cascade of
    one `resonator` per formant. The result is scaled so its largest absolute
    value is 0.5.
    """
    raise NotImplementedError


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
