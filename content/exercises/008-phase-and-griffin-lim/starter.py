"""Phase and Griffin-Lim.

Fill in the functions below, then press Run to hear a magnitude spectrogram
turned back into sound four ways, watch Griffin-Lim converge, and hear a
time stretch that keeps the pitch. Press Check to test your code.

Every STFT here is shaped (bins, frames), centered with reflect padding, with
a periodic Hann window, as in The short-time Fourier transform.
"""

import numpy as np

from speechlab import play, show_plot, show_spectrogram


# Provided: the STFT and its least-squares inverse, from The short-time Fourier transform.

def hann(n: int) -> np.ndarray:
    """The periodic Hann window of length n."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """The centered, reflect-padded, Hann-windowed STFT, shape (n_fft // 2 + 1, frames)."""
    x = np.pad(np.asarray(x, dtype=float), n_fft // 2, mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop_length
    index = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[index] * hann(n_fft), axis=1).T


def istft(X: np.ndarray, hop_length: int, length: int | None = None) -> np.ndarray:
    """The least-squares inverse of `stft`: hop_length * (frames - 1) samples unless `length` is given."""
    n_fft = 2 * (X.shape[0] - 1)
    w = hann(n_fft)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1) * w
    total = n_fft + hop_length * (frames.shape[0] - 1)
    signal, weight = np.zeros(total), np.zeros(total)
    for m, frame in enumerate(frames):
        signal[m * hop_length : m * hop_length + n_fft] += frame
        weight[m * hop_length : m * hop_length + n_fft] += w**2
    length = total - n_fft if length is None else length
    start = n_fft // 2
    return signal[start : start + length] / np.maximum(weight[start : start + length], 1e-11)


# The exercise.

def spectral_convergence(X: np.ndarray, magnitude: np.ndarray) -> float:
    """How far |X| is from a target magnitude S, relative to S, as a float.

    || |X| - S ||_F / || S ||_F, with the Frobenius norm np.linalg.norm.
    `X` can be a complex STFT or a magnitude; both are (bins, frames).
    """
    raise NotImplementedError("spectral_convergence")


def random_phase(magnitude: np.ndarray, seed: int = 0) -> np.ndarray:
    """The target magnitude with a random phase in every cell, complex, same shape.

    The phases, in radians, are 2 pi * rng.random(magnitude.shape) with
    rng = np.random.default_rng(seed), so the same seed gives the same array.
    """
    raise NotImplementedError("random_phase")


def griffin_lim(
    magnitude: np.ndarray,
    n_fft: int,
    hop_length: int,
    n_iter: int = 32,
    momentum: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """A 1-D signal whose STFT magnitude approximates `magnitude`, shape (bins, frames).

    Fast Griffin-Lim (Perraudin et al., 2013) with the provided `stft` and
    `istft`, started from random_phase(magnitude, seed) and run for n_iter
    iterations. `momentum` extrapolates each consistent projection from the
    previous one, as the chapter describes; the first iteration has no previous
    projection, and momentum=0 gives plain Griffin-Lim.

    Returns the inverse STFT of the target magnitude under the final phase,
    hop_length * (frames - 1) samples. With n_iter=0 that is the inverse STFT
    of the random start.
    """
    raise NotImplementedError("griffin_lim")


def phase_vocoder(X: np.ndarray, stretch: float, hop_length: int) -> np.ndarray:
    """Stretch a complex STFT X in time by `stretch`, keeping each bin's frequency.

    Output frame i reads X at the fractional position i / stretch, for the
    positions np.arange(0, frames, 1 / stretch); past the last frame, X counts
    as zeros. The output magnitude interpolates |X| linearly between the two
    neighboring frames. The output phase starts at np.angle(X[:, 0]) and, from
    one output frame to the next, advances each bin by its measured frequency
    over one hop. n_fft is 2 * (bins - 1).

    Returns a complex array of shape (bins, len(positions)). With stretch=1
    it returns X itself.
    """
    raise NotImplementedError("phase_vocoder")


def time_stretch(x: np.ndarray, stretch: float, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Make `x` about `stretch` times longer, within one hop, without changing its pitch.

    Uses the provided `stft` and `istft` at these settings, and your
    `phase_vocoder`.
    """
    raise NotImplementedError("time_stretch")


def _voice(rate: int, seconds: float) -> np.ndarray:
    """A buzzy vowel-like tone: harmonics of a gliding pitch, shaped by two resonances."""
    t = np.arange(int(rate * seconds)) / rate
    f0 = 140 + 40 * t / seconds
    phase = 2 * np.pi * np.cumsum(f0) / rate
    x = np.zeros_like(t)
    for k in range(1, 40):
        f = k * f0
        gain = 1 / (1 + ((f - 700) / 150) ** 2) + 0.5 / (1 + ((f - 1200) / 200) ** 2) + 0.02
        x += np.where(f < rate / 2, gain * np.sin(k * phase), 0)
    x *= np.minimum(1, np.minimum(t, t[::-1]) / 0.05)
    return 0.4 * x / np.max(np.abs(x))


if __name__ == "__main__":
    rate, n_fft, hop = 16000, 512, 128
    x = _voice(rate, 1.536)  # 192 hops exactly, so the inverse returns every sample
    S = np.abs(stft(x, n_fft, hop))
    play(x, rate, label="The original")

    zero = istft(S.astype(complex), hop)
    noisy = griffin_lim(S, n_fft, hop, n_iter=0)
    gla = griffin_lim(S, n_fft, hop, n_iter=32)
    fgla = griffin_lim(S, n_fft, hop, n_iter=32, momentum=0.99)
    for name, y in [("zero phase", zero), ("random phase", noisy), ("Griffin-Lim, 32 iterations", gla), ("fast Griffin-Lim, 32 iterations", fgla)]:
        print(f"{name:>32}: spectral convergence {spectral_convergence(stft(y, n_fft, hop), S):.3f}")
    peak = max(np.max(np.abs(y)) for y in (zero, noisy, gla, fgla))
    play(0.4 * zero / peak, rate, label="Zero phase: a mechanical buzz tied to the frame rate")
    play(0.4 * noisy / peak, rate, label="Random phase: rough and hoarse")
    play(0.4 * gla / peak, rate, label="Griffin-Lim, 32 iterations")
    play(0.4 * fgla / peak, rate, label="Fast Griffin-Lim, 32 iterations, momentum 0.99")
    show_spectrogram(stft(fgla, n_fft, hop), rate, hop, label="The fast Griffin-Lim result")

    counts = [0, 1, 2, 4, 8, 16, 32, 64]
    curves = []
    for momentum in (0.0, 0.99):
        sc = [spectral_convergence(stft(griffin_lim(S, n_fft, hop, n, momentum), n_fft, hop), S) for n in counts]
        curves.append((np.array(counts), np.array(sc)))
    show_plot(*curves, labels=["Griffin-Lim", "fast Griffin-Lim, momentum 0.99"], label="Spectral convergence against iterations", xlabel="Iterations", ylabel="Spectral convergence", y_scale="log")

    for stretch in (1.5, 0.75):
        y = time_stretch(x, stretch)
        print(f"stretch {stretch}: {len(x) / rate:.2f} s became {len(y) / rate:.2f} s")
        play(y, rate, label=f"Time-stretched by {stretch}, same pitch")
