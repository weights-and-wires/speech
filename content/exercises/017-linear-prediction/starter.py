"""Linear prediction.

Predict each sample of a vowel from the samples before it, and read the vocal
tract out of the predictor. Build the prediction error, the autocorrelation,
the Levinson-Durbin recursion, the LPC envelope, formants from the poles, and
the real cepstrum. The vowel synthesizer from the source-filter chapter is
provided in full. Fill in the functions below, then press Run to hear three
vowels and see each frame's LPC envelope, and Check to test your code.
"""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot

RATE = 16000


def impulse_train(f0: float, sample_rate: int, n_samples: int) -> np.ndarray:
    """Provided in full, from the source-filter chapter. Unit impulses at samples round(k * sample_rate / f0), k = 0, 1, 2, ..."""
    x = np.zeros(n_samples)
    period = sample_rate / f0
    positions = np.round(np.arange(0.0, n_samples, period)).astype(int)
    x[positions[positions < n_samples]] = 1.0
    return x


def resonator(frequency: float, bandwidth: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Provided in full, from the source-filter chapter. Klatt's two-pole resonator with unit gain at 0 Hz, as (b, a) for lfilter."""
    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2.0 * np.pi * frequency / sample_rate
    a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
    b = np.array([a.sum()])
    return b, a


def synthesize_vowel(f0: float, formants, bandwidths, sample_rate: int, seconds: float) -> np.ndarray:
    """Provided in full, from the source-filter chapter. An impulse train at f0 through a cascade of formant resonators, peak 0.5."""
    y = impulse_train(f0, sample_rate, int(round(seconds * sample_rate)))
    for frequency, bandwidth in zip(formants, bandwidths):
        b, a = resonator(frequency, bandwidth, sample_rate)
        y = lfilter(b, a, y)
    return 0.5 * y / np.max(np.abs(y))




def prediction_error(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """How wrong the predictor is at every sample, shape (len(x),).

    The prediction of x[n] is sum over k = 1 ... p of alpha[k - 1] * x[n - k],
    where x is taken as zero before sample 0. Return e[n] = x[n] minus that
    prediction. Filtering x with the polynomial [1, -alpha[0], ..., -alpha[p - 1]]
    does it in one call.
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

    The frame is windowed with np.hamming(len(frame)) before its
    autocorrelation, up to lag `order`, is computed.
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
        residual = prediction_error(frame * np.hamming(512), alpha)
        gain = 10 * np.log10(np.sum((frame * np.hamming(512)) ** 2) / np.sum(residual ** 2))
        ceps = real_cepstrum(frame * np.hamming(512))
        peak = 100 + int(np.argmax(ceps[100:200]))
        print(f"{name}: set {targets} Hz, LPC finds {np.round(found[:3]).astype(int).tolist()} Hz, "
              f"the error is {gain:.1f} dB below the signal, the cepstrum peaks at quefrency {peak} = {RATE / peak:.1f} Hz")

    low = synthesize_vowel(100.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    high = synthesize_vowel(200.0, vowels["/ɑ/ as in 'father'"], bandwidths, RATE, 0.8)
    play(np.concatenate([low, high]), RATE, label="/ɑ/ at 100 Hz, then at 200 Hz: new pitch, same vowel")
