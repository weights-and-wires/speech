"""Pitch tracking.

Build YIN step by step: the difference function, its cumulative mean
normalization, the absolute threshold, and parabolic interpolation. Then track
the pitch of a synthetic voice frame by frame, decide which frames are voiced,
and score the track. Fill in the functions below, then press Run to hear a
voice, see its pitch track, and hear the track hummed back, and Check to test
your code.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from speechlab import play, show_plot, show_spectrogram

RATE = 16000


def autocorr_pitch(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float) -> float:
    """F0 in Hz by picking the largest autocorrelation value, step 1 of YIN.

    With tau_min = floor(sample_rate / fmax), tau_max = ceil(sample_rate / fmin),
    and W = len(frame) - tau_max, compute r(tau) = sum over j < W of
    frame[j] * frame[j + tau]. Return sample_rate / tau for the tau in
    [tau_min, tau_max] with the largest r(tau).
    """
    raise NotImplementedError


def difference_function(frame: np.ndarray, max_lag: int) -> np.ndarray:
    """YIN's difference function, shape (max_lag + 1,).

    With W = len(frame) - max_lag, d(tau) = sum over j = 0 ... W - 1 of
    (frame[j] - frame[j + tau])^2, for tau = 0 ... max_lag.
    """
    raise NotImplementedError


def cmndf(d: np.ndarray) -> np.ndarray:
    """The cumulative mean normalized difference function, same shape as `d`.

    d'(0) = 1, and d'(tau) = d(tau) * tau / (d(1) + ... + d(tau)) for tau >= 1.
    Where that running sum is 0 (a silent frame), set d'(tau) = 1.
    """
    raise NotImplementedError


def absolute_threshold(dprime: np.ndarray, tau_min: int, tau_max: int, threshold: float = 0.1) -> int:
    """YIN's choice of lag, an int in [tau_min, tau_max].

    The result is the bottom of the first dip under `threshold` in
    [tau_min, tau_max]: of the lags from the first one under the threshold
    onward, the last one before dprime stops falling, without leaving the
    range. If no lag in the range is under the threshold, the result is the
    lag of the smallest dprime in [tau_min, tau_max].
    """
    raise NotImplementedError


def parabolic_interpolation(y: np.ndarray, tau: int) -> float:
    """The lag of the vertex of the parabola through (tau - 1, y[tau - 1]), (tau, y[tau]), (tau + 1, y[tau + 1]).

    Return float(tau) unchanged when tau is the first or last index of `y`, or
    when the three points lie on a straight line.
    """
    raise NotImplementedError


def yin(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float, threshold: float = 0.1) -> tuple[float, float]:
    """YIN on one frame: (F0 in Hz, aperiodicity).

    The search range is tau_min = floor(sample_rate / fmax) to
    tau_max = ceil(sample_rate / fmin), and d' is computed up to lag
    tau_max + 1 so the parabola has a right neighbor. F0 is sample_rate over
    the interpolated lag, and the aperiodicity is d' at the integer lag that
    the absolute threshold picked.
    """
    raise NotImplementedError


def track_pitch(x, sample_rate, frame_length=1024, hop=160, fmin=60.0, fmax=500.0, threshold=0.1, voicing_threshold=0.25):
    """A YIN pitch track: (f0, voiced), both of shape (1 + len(x) // hop,).

    Pad x by frame_length // 2 samples at each end by reflection, as in the
    STFT lesson, so frame t is padded[t * hop : t * hop + frame_length] and is
    centered on sample t * hop. A frame is voiced when
    its aperiodicity is below `voicing_threshold`; then f0[t] is YIN's estimate.
    Otherwise f0[t] = 0 and voiced[t] = False.
    """
    raise NotImplementedError


def gross_pitch_error(f0_est: np.ndarray, f0_ref: np.ndarray, tolerance: float = 0.2) -> float:
    """The gross pitch error, a float in [0, 1].

    Among frames where both f0_est and f0_ref are above 0, the share where
    |f0_est - f0_ref| > tolerance * f0_ref. Return 0.0 if there are none.
    """
    raise NotImplementedError


def voicing_error(f0_est: np.ndarray, f0_ref: np.ndarray) -> float:
    """The voicing decision error: the share of frames where (f0_est > 0) differs from (f0_ref > 0)."""
    raise NotImplementedError


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
