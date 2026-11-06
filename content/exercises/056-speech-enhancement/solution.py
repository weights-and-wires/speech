"""Speech enhancement: the worked solution."""

import numpy as np
import torch
from scipy.signal import istft as scipy_istft
from scipy.signal import lfilter
from scipy.signal import stft as scipy_stft

from speechlab import play, show_spectrogram


# Scoring and mixing -------------------------------------------------------------


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant signal-to-distortion ratio in dB, over the last axis.

    est, ref: tensors of the same shape (..., T). Removes the mean of each,
    projects est onto ref to get the target part, and returns
    10 log10(|target|^2 / |est - target|^2) with shape (...,).
    Differentiable, so -si_sdr works as a loss.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10(((target**2).sum(-1) + eps) / ((noise**2).sum(-1) + eps))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale noise so that clean + noise has the requested SNR in dB.

    Power is the mean square over the whole clip. Returns (mixture, scaled noise).
    """
    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2)
    gain = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    scaled = gain * noise
    return clean + scaled, scaled


# Classical enhancement ------------------------------------------------------------


def spectral_subtraction(Y: np.ndarray, noise_power: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
    """Power spectral subtraction with over-subtraction and a spectral floor.

    Y: complex STFT of the noisy signal, (F, T). noise_power: (F,), the
    estimated noise power in each bin. The estimated clean power is
    max(|Y|^2 - alpha * noise_power, beta * |Y|^2); the phase is the noisy
    phase. Returns the complex estimate, (F, T).
    """
    power = np.abs(Y) ** 2
    clean_power = np.maximum(power - alpha * noise_power[:, None], beta * power)
    return np.sqrt(clean_power) * np.exp(1j * np.angle(Y))


def wiener_gain(xi: np.ndarray) -> np.ndarray:
    """The Wiener gain xi / (1 + xi) for an a priori SNR xi (a power ratio, not dB)."""
    return xi / (1.0 + xi)


def decision_directed_gains(Y: np.ndarray, noise_power: np.ndarray, a: float = 0.98, xi_min: float = 10 ** (-25 / 10)) -> np.ndarray:
    """Wiener gains with the decision-directed a priori SNR estimate.

    Y: complex noisy STFT, (F, T). noise_power: (F,). Frame by frame:
        gamma = |Y[:, t]|^2 / noise_power
        xi = a * |S_prev|^2 / noise_power + (1 - a) * max(gamma - 1, 0)
        xi = max(xi, xi_min)
        G[:, t] = wiener_gain(xi), and S_prev = G[:, t] * Y[:, t]
    with S_prev = 0 before the first frame. Returns the gains, (F, T).
    """
    n_bins, n_frames = Y.shape
    gains = np.empty((n_bins, n_frames))
    prev = np.zeros(n_bins)
    for t in range(n_frames):
        power = np.abs(Y[:, t]) ** 2
        gamma = power / noise_power
        xi = a * prev / noise_power + (1 - a) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)
        gains[:, t] = wiener_gain(xi)
        prev = gains[:, t] ** 2 * power
    return gains


def ideal_masks(S: np.ndarray, N: np.ndarray, lc_db: float = 0.0, beta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The ideal binary mask and the ideal ratio mask, both (F, T).

    S, N: complex STFTs of the clean speech and the noise. The IBM is 1 where
    10 log10(|S|^2 / |N|^2) > lc_db and 0 elsewhere. The IRM is
    (|S|^2 / (|S|^2 + |N|^2)) ** beta. Both use powers, never dB values.
    """
    ps, pn = np.abs(S) ** 2, np.abs(N) ** 2
    tiny = 1e-20
    ibm = (10 * np.log10((ps + tiny) / (pn + tiny)) > lc_db).astype(float)
    irm = (ps / (ps + pn + tiny)) ** beta
    return ibm, irm


# Run helpers -----------------------------------------------------------------------

VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def vowel_train(rng, rate, seconds, f0, gap=0.0):
    """Source-filter vowels with a gliding pitch, 150-250 ms each, with optional pauses."""
    out = []
    while sum(len(o) for o in out) < seconds * rate:
        n = int(rng.uniform(0.15, 0.25) * rate)
        pitch = f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / rate))
        phase = np.cumsum(pitch / rate)
        src = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
        y = src
        for f, bw in zip(VOWELS[rng.integers(len(VOWELS))], (80, 100, 140)):
            r = np.exp(-np.pi * bw / rate)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
        out.append(y * np.hanning(n) ** 0.3)
        if gap:
            out.append(np.zeros(int(gap * rate)))
    x = np.concatenate(out)[: int(seconds * rate)]
    return 0.5 * x / np.max(np.abs(x))


def stft(x, n_fft=512, hop=128):
    return scipy_stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann")[2]


def istft(X, length, n_fft=512, hop=128):
    return scipy_istft(X, nperseg=n_fft, noverlap=n_fft - hop, window="hann")[1][:length]


def score(est, ref):
    return float(si_sdr(torch.as_tensor(est, dtype=torch.float64), torch.as_tensor(ref, dtype=torch.float64)))


def enhancement_demo(rng):
    rate = 16000
    clean = np.concatenate([np.zeros(rate // 4), vowel_train(rng, rate, 2.25, 120.0, gap=0.08)])
    noisy, noise = mix_at_snr(clean, rng.standard_normal(clean.size), 5.0)
    Y, S, N = stft(noisy), stft(clean), stft(noise)
    noise_power = np.mean(np.abs(Y[:, : (rate // 4) // 128]) ** 2, axis=1)  # the first 0.25 s is noise only
    subtracted = istft(spectral_subtraction(Y, noise_power), clean.size)
    wiener = istft(decision_directed_gains(Y, noise_power) * Y, clean.size)
    ibm, irm = ideal_masks(S, N)
    oracle = istft(irm * Y, clean.size)
    print("Enhancement at 5 dB SNR, SI-SDR against the clean signal:")
    for name, x in [("noisy input", noisy), ("spectral subtraction", subtracted),
                    ("decision-directed Wiener", wiener), ("oracle IBM", istft(ibm * Y, clean.size)),
                    ("oracle IRM", oracle)]:
        print(f"  {name:26} {score(x, clean):6.2f} dB")
    peak = np.max(np.abs(noisy))
    play(noisy / peak, rate, label="Noisy vowels at 5 dB SNR")
    play(wiener / peak, rate, label="After the decision-directed Wiener filter")
    play(subtracted / peak, rate, label="After spectral subtraction: listen for musical noise")
    show_spectrogram(np.abs(stft(wiener)), rate, 128, label="The Wiener output's spectrogram")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    enhancement_demo(rng)
