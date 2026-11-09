"""Pitch tracking: the worked solution."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from speechlab import play, show_plot, show_spectrogram

RATE = 16000


def autocorr_pitch(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float) -> float:
    """F0 in Hz from the largest autocorrelation peak between the lags for fmax and fmin."""
    frame = np.asarray(frame, dtype=float)
    tau_min, tau_max = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    W = len(frame) - tau_max
    r = sliding_window_view(frame, W)[: tau_max + 1] @ frame[:W]
    tau = tau_min + int(np.argmax(r[tau_min : tau_max + 1]))
    return sample_rate / tau


def difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """YIN's d(tau) = sum_{j < W} (x[j] - x[j + tau])^2 for tau = 0 ... max_lag, with W = len(frame) - max_lag."""
    frame = np.asarray(frame, dtype=float)
    W = len(frame) - max_lag
    shifted = sliding_window_view(frame, W)[: max_lag + 1]  # row tau is frame[tau : tau + W]
    return np.sum((shifted - frame[:W]) ** 2, axis=1)


def cmndf(d: np.ndarray) -> np.ndarray:
    """The cumulative mean normalized difference: 1 at tau = 0, then d(tau) over its running mean."""
    d = np.asarray(d, dtype=float)
    out = np.ones_like(d)
    running = np.cumsum(d[1:])
    tau = np.arange(1, len(d))
    safe = running > 0
    out[1:][safe] = d[1:][safe] * tau[safe] / running[safe]
    return out


def absolute_threshold(dprime: np.ndarray, tau_min: int, tau_max: int, threshold: float = 0.1) -> int:
    """The first dip below `threshold` in [tau_min, tau_max], followed down to its minimum, or the global minimum."""
    for tau in range(tau_min, tau_max + 1):
        if dprime[tau] < threshold:
            while tau + 1 <= tau_max and dprime[tau + 1] < dprime[tau]:
                tau += 1
            return tau
    return tau_min + int(np.argmin(dprime[tau_min : tau_max + 1]))


def parabolic_interpolation(y: np.ndarray, tau: int) -> float:
    """The abscissa of the vertex of the parabola through y[tau - 1], y[tau], y[tau + 1]."""
    if tau <= 0 or tau >= len(y) - 1:
        return float(tau)
    left, mid, right = y[tau - 1], y[tau], y[tau + 1]
    curvature = left - 2.0 * mid + right
    if curvature == 0:
        return float(tau)
    return tau + 0.5 * (left - right) / curvature


def yin(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float, threshold: float = 0.1) -> tuple[float, float]:
    """YIN on one frame: (F0 in Hz, aperiodicity d'(tau) at the chosen integer lag)."""
    tau_min, tau_max = int(np.floor(sample_rate / fmax)), int(np.ceil(sample_rate / fmin))
    dprime = cmndf(difference_function(frame, tau_max + 1))
    tau = absolute_threshold(dprime, tau_min, tau_max, threshold)
    return sample_rate / parabolic_interpolation(dprime, tau), float(dprime[tau])


def track_pitch(x, sample_rate, frame_length=1024, hop=160, fmin=60.0, fmax=500.0, threshold=0.1, voicing_threshold=0.25):
    """YIN every `hop` samples: (f0, voiced), each of shape (1 + len(x) // hop,), with f0 = 0 when unvoiced."""
    x = np.asarray(x, dtype=float)
    padded = np.pad(x, frame_length // 2, mode="reflect")
    n_frames = 1 + len(x) // hop
    f0 = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for t in range(n_frames):
        frame = padded[t * hop : t * hop + frame_length]
        estimate, aperiodicity = yin(frame, sample_rate, fmin, fmax, threshold)
        if aperiodicity < voicing_threshold:
            f0[t], voiced[t] = estimate, True
    return f0, voiced


def gross_pitch_error(f0_est: np.ndarray, f0_ref: np.ndarray, tolerance: float = 0.2) -> float:
    """The share of frames voiced in both tracks whose estimate is off by more than `tolerance` of the reference."""
    f0_est, f0_ref = np.asarray(f0_est, dtype=float), np.asarray(f0_ref, dtype=float)
    both = (f0_est > 0) & (f0_ref > 0)
    if not np.any(both):
        return 0.0
    return float(np.mean(np.abs(f0_est[both] - f0_ref[both]) > tolerance * f0_ref[both]))


def voicing_error(f0_est: np.ndarray, f0_ref: np.ndarray) -> float:
    """The share of frames where the two tracks disagree about whether the frame is voiced."""
    return float(np.mean((np.asarray(f0_est) > 0) != (np.asarray(f0_ref) > 0)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    seconds, hop = 2.5, 160
    n = int(seconds * RATE)
    t = np.arange(n) / RATE

    # A question's intonation: the pitch falls, then rises at the end.
    true_f0 = 130.0 - 30.0 * np.sin(np.pi * t / 1.6) + 90.0 * np.clip((t - 1.8) / 0.7, 0.0, 1.0) ** 2
    voiced_mask = ~(((t > 0.55) & (t < 0.7)) | ((t > 1.3) & (t < 1.45)) | (t < 0.1) | (t > 2.4))
    phase = np.cumsum(2.0 * np.pi * true_f0 / RATE)
    pulses = np.maximum(np.cos(phase), 0.0) ** 40  # a narrow pulse once per period
    source = np.where(voiced_mask, pulses, 0.0)
    noise_bursts = (((t > 0.55) & (t < 0.7)) | ((t > 1.3) & (t < 1.45))) * 0.02 * rng.standard_normal(n)
    speech = source
    for f, bw in [(650, 80), (1100, 100), (2450, 150)]:
        r = np.exp(-np.pi * bw / RATE)
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r]
        speech = lfilter([sum(a)], a, speech)
    speech = 0.5 * speech / np.max(np.abs(speech)) + noise_bursts
    speech = speech + 0.003 * rng.standard_normal(n)
    play(speech, RATE, label="A synthetic voice asking a question, with two hissy consonants")
    frames = np.pad(speech, 512, mode="reflect")[np.arange(1024)[None, :] + hop * np.arange(1 + n // hop)[:, None]]
    show_spectrogram(np.abs(np.fft.rfft(frames * np.hanning(1024), axis=1)).T, RATE, hop,
                     label="Its spectrogram: the harmonics trace the pitch contour")

    f0, voiced = track_pitch(speech, RATE, hop=hop)
    frame_times = np.arange(len(f0)) * hop / RATE
    reference = np.where(voiced_mask[np.minimum(np.arange(len(f0)) * hop, n - 1)], true_f0[np.minimum(np.arange(len(f0)) * hop, n - 1)], 0.0)
    show_plot((frame_times, reference), (frame_times, f0), labels=["true F0", "YIN"], label="The pitch track",
              xlabel="Time (s)", ylabel="F0 (Hz, 0 = unvoiced)")

    print(f"{len(f0)} frames, {voiced.sum()} voiced by YIN, {int((reference > 0).sum())} voiced in truth")
    print(f"YIN: gross pitch error {100 * gross_pitch_error(f0, reference):.1f}%, voicing error {100 * voicing_error(f0, reference):.1f}%")

    # Single frames of 60 voices with random F0 from 80 Hz to 400 Hz, random formants, 20 dB SNR.
    truth, by_yin, by_acf = [], [], []
    for _ in range(60):
        f0_true = rng.uniform(80.0, 400.0)
        tt = np.arange(1024) / RATE
        frame = sum(np.sin(2 * np.pi * k * f0_true * tt + rng.uniform(0, 2 * np.pi)) / k for k in range(1, int(4000 // f0_true) + 1))
        for f, bw in [(rng.uniform(300, 900), 80), (rng.uniform(900, 2200), 120)]:
            r = np.exp(-np.pi * bw / RATE)
            a = [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r]
            frame = lfilter([sum(a)], a, frame)
        frame = frame / np.std(frame) + 0.1 * rng.standard_normal(1024)
        truth.append(f0_true)
        by_yin.append(yin(frame, RATE, 60.0, 500.0)[0])
        by_acf.append(autocorr_pitch(frame, RATE, 60.0, 500.0))
    print(f"60 random voices: gross pitch error {100 * gross_pitch_error(by_yin, truth):.1f}% for YIN, "
          f"{100 * gross_pitch_error(by_acf, truth):.1f}% for the autocorrelation peak")

    resynth_phase = np.cumsum(2.0 * np.pi * np.interp(t, frame_times, f0) / RATE)
    hum = 0.3 * np.sin(resynth_phase) * np.interp(t, frame_times, voiced.astype(float))
    play(hum, RATE, label="Your pitch track, hummed back as a sine")
