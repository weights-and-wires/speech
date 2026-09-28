"""Speech production and linear prediction: the worked solution."""

import numpy as np
from scipy.signal import lfilter

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


def autocorrelation(frame: np.ndarray, p: int) -> np.ndarray:
    """r[k] = sum_n frame[n] frame[n - k] for k = 0 ... p, shape (p + 1,)."""
    frame = np.asarray(frame, dtype=float)
    N = len(frame)
    return np.array([np.dot(frame[k:], frame[: N - k]) for k in range(p + 1)])


def levinson_durbin(r: np.ndarray, p: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the order-p normal equations: returns (alpha, E, k)."""
    r = np.asarray(r, dtype=float)
    alpha = np.zeros(p)
    k = np.zeros(p)
    E = r[0]
    for i in range(1, p + 1):
        acc = r[i] - np.dot(alpha[: i - 1], r[i - 1 : 0 : -1])
        k_i = acc / E
        previous = alpha[: i - 1].copy()
        alpha[: i - 1] = previous - k_i * previous[::-1]
        alpha[i - 1] = k_i
        k[i - 1] = k_i
        E = (1.0 - k_i * k_i) * E
    return alpha, E, k


def lpc(frame: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Autocorrelation-method LPC of a Hamming-windowed frame: (alpha, E)."""
    windowed = np.asarray(frame, dtype=float) * np.hamming(len(frame))
    alpha, E, _ = levinson_durbin(autocorrelation(windowed, order), order)
    return alpha, E


def lpc_envelope(alpha: np.ndarray, E: float, n_fft: int) -> np.ndarray:
    """The all-pole power spectrum E / |A(e^jw)|^2 at the n_fft // 2 + 1 rfft bins."""
    A = np.fft.rfft(np.concatenate([[1.0], -np.asarray(alpha, dtype=float)]), n_fft)
    return E / np.abs(A) ** 2


def formants(alpha: np.ndarray, sample_rate: int, max_bandwidth: float = 400.0, min_frequency: float = 90.0):
    """Formant frequencies and bandwidths in Hz from the roots of A(z), sorted by frequency."""
    roots = np.roots(np.concatenate([[1.0], -np.asarray(alpha, dtype=float)]))
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * sample_rate / (2.0 * np.pi)
    bands = -np.log(np.abs(roots)) * sample_rate / np.pi
    keep = (freqs > min_frequency) & (bands < max_bandwidth)
    order = np.argsort(freqs[keep])
    return freqs[keep][order], bands[keep][order]


def real_cepstrum(frame: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """The real cepstrum: the inverse FFT of the log magnitude spectrum, shape (len(frame),)."""
    frame = np.asarray(frame, dtype=float)
    return np.fft.irfft(np.log(np.abs(np.fft.rfft(frame)) + eps), len(frame))


def cepstral_pitch(frame: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    """F0 in Hz from the largest cepstral peak between quefrencies sample_rate / fmax and sample_rate / fmin."""
    ceps = real_cepstrum(np.asarray(frame, dtype=float) * np.hamming(len(frame)))
    lo, hi = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    q = lo + int(np.argmax(ceps[lo : hi + 1]))
    return sample_rate / q


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
