"""Speech production and linear prediction.

Build a source-filter vowel synthesizer, then take it apart again: estimate the
vocal tract with autocorrelation LPC and the Levinson-Durbin recursion, read the
formants from the poles, and find the pitch in the real cepstrum. Fill in the
functions below, then press Run to hear three vowels and see their LPC
envelopes, and Check to test your code.
"""

import numpy as np
from scipy.signal import lfilter

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

    Pass `impulse_train(f0, ...)` through one `resonator` per formant in turn,
    with scipy.signal.lfilter, and scale the result so its largest absolute
    value is 0.5.
    """
    raise NotImplementedError


def autocorrelation(frame: np.ndarray, p: int) -> np.ndarray:
    """The autocorrelation of a frame at lags 0 to p, shape (p + 1,).

    r[k] = sum over n of frame[n] * frame[n - k], with the frame taken as zero
    outside its own samples.
    """
    raise NotImplementedError


def levinson_durbin(r: np.ndarray, p: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the order-p LPC normal equations from autocorrelations r[0..p].

    Returns (alpha, E, k): the predictor coefficients alpha, shape (p,), such
    that x[n] is predicted by sum_k alpha[k - 1] * x[n - k]; the final
    prediction-error energy E, a float; and the reflection coefficients k,
    shape (p,). Use the O(p^2) recursion, not a matrix solve.
    """
    raise NotImplementedError


def lpc(frame: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Autocorrelation-method LPC of one frame: (alpha, E).

    Multiply the frame by np.hamming(len(frame)), take its autocorrelation up
    to lag `order`, and run `levinson_durbin`.
    """
    raise NotImplementedError


def lpc_envelope(alpha: np.ndarray, E: float, n_fft: int) -> np.ndarray:
    """The LPC power spectrum E / |A(e^jw)|^2, shape (n_fft // 2 + 1,).

    A(z) = 1 - sum_k alpha[k - 1] z^-k. Evaluate it at the rfft bin
    frequencies k * sample_rate / n_fft, k = 0 ... n_fft // 2.
    """
    raise NotImplementedError


def formants(alpha: np.ndarray, sample_rate: int, max_bandwidth: float = 400.0, min_frequency: float = 90.0):
    """Formants from the poles of 1 / A(z): (frequencies, bandwidths) in Hz.

    Find the roots of A(z) = 1 - sum_k alpha[k - 1] z^-k and keep those with a
    positive imaginary part. A root z has frequency angle(z) * sample_rate / (2 pi)
    and bandwidth -ln|z| * sample_rate / pi. Keep roots whose frequency is
    above `min_frequency` and whose bandwidth is below `max_bandwidth`, and
    return both arrays sorted by frequency.
    """
    raise NotImplementedError


def real_cepstrum(frame: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """The real cepstrum of a frame, shape (len(frame),).

    The inverse DFT of ln(|DFT(frame)| + eps). The result is real.
    """
    raise NotImplementedError


def cepstral_pitch(frame: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """F0 in Hz from the real cepstrum of a Hamming-windowed frame.

    Find the quefrency q with the largest cepstral value between
    floor(sample_rate / fmax) and ceil(sample_rate / fmin), inclusive, and
    return sample_rate / q.
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
    n_fft = 1024
    freqs = np.arange(n_fft // 2 + 1) * RATE / n_fft
    for name, targets in vowels.items():
        y = synthesize_vowel(120.0, targets, bandwidths, RATE, 0.8)
        play(y, RATE, label=f"{name}, F0 = 120 Hz")
        frame = y[4000:4000 + 512]
        alpha, E = lpc(frame, 18)
        found, _ = formants(alpha, RATE)
        spectrum = np.abs(np.fft.rfft(frame * np.hamming(512), n_fft)) ** 2
        envelope = lpc_envelope(alpha, E, n_fft)
        show_plot((freqs, 10 * np.log10(spectrum + 1e-12)), (freqs, 10 * np.log10(envelope)),
                  labels=["frame spectrum", "LPC envelope, order 18"], label=f"{name}: spectrum and LPC envelope",
                  xlabel="Frequency (Hz)", ylabel="Power (dB)")
        print(f"{name}: set {targets} Hz, LPC finds {np.round(found[:3]).astype(int).tolist()} Hz, "
              f"cepstral pitch {cepstral_pitch(frame, RATE):.1f} Hz")

    low = synthesize_vowel(100.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    high = synthesize_vowel(200.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    play(np.concatenate([low, high]), RATE, label="/ɑ/ at 100 Hz, then at 200 Hz: new pitch, same vowel")
