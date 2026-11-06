"""Checks for the pitch tracking exercise."""

import numpy as np
from scipy.signal import lfilter

from speechcheck import check, close, equal, fail, finite, has, shape

RATE = 16000


def harmonic_voice(rng, f0, n=1024, snr_db=20.0, first=1):
    """Harmonics `first`.. of f0 up to 4 kHz, random phases, two random formants, and noise."""
    t = np.arange(n) / RATE
    x = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) / k for k in range(first, int(4000 // f0) + 1))
    for f, bw in [(rng.uniform(300, 900), 80.0), (rng.uniform(900, 2200), 120.0)]:
        r = np.exp(-np.pi * bw / RATE)
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r]
        x = lfilter([sum(a)], a, x)
    x = x / np.std(x)
    return x + 10 ** (-snr_db / 20) * rng.standard_normal(n)


@check("difference_function reproduces the hand example and a brute-force sum")
def test_difference(sub):
    difference_function = has(sub, "difference_function")
    x = np.array([0, 1, 0, -1, 0, 1, 0, -1, 0], dtype=float)
    got = np.asarray(difference_function(x, 5), dtype=float)
    shape(got, (6,), label="d for max_lag 5")
    close(got, [0, 4, 8, 4, 0, 4], atol=1e-12, label="d of [0, 1, 0, -1] repeated, W = 4")
    x = np.random.default_rng(0).standard_normal(300)
    W = 300 - 120
    want = np.array([np.sum((x[:W] - x[tau : tau + W]) ** 2) for tau in range(121)])
    close(difference_function(x, 120), want, rtol=1e-10, atol=1e-9, label="d of random noise, max_lag 120")


@check("cmndf starts at 1, divides by the running mean, and survives silence")
def test_cmndf(sub):
    cmndf = has(sub, "cmndf")
    close(cmndf(np.array([0.0, 4, 8, 4, 0, 4])), [1, 1, 4 / 3, 0.75, 0, 1], atol=1e-12, label="d' for the hand example")
    close(cmndf(np.array([0.0, 4.5, 8.5, 4.5, 0.0, 4.5])), [1, 1, 1.3077, 0.7714, 0, 1.0227], atol=1e-4,
          label="d' for [0, 1, 0.5, -1] repeated")
    silent = np.asarray(cmndf(np.zeros(50)), dtype=float)
    finite(silent, label="d' of a silent frame")
    close(silent, np.ones(50), label="d' of a silent frame, where every lag is equally (a)periodic")


@check("absolute_threshold takes the first dip under the threshold, follows it to its bottom, or falls back to the global minimum")
def test_threshold(sub):
    absolute_threshold = has(sub, "absolute_threshold")
    equal(int(absolute_threshold(np.array([1, 1, 4 / 3, 0.75, 0, 1]), 1, 5, 0.1)), 4, label="the hand example")
    # Dips at 10 (0.08, the bottom at 11) and at 20 (0.01, deeper): YIN wants the first.
    d = np.ones(30)
    d[9:13] = [0.3, 0.08, 0.05, 0.2]
    d[19:22] = [0.2, 0.01, 0.3]
    equal(int(absolute_threshold(d, 2, 28, 0.1)), 11, label="the first dip, followed to its bottom")
    equal(int(absolute_threshold(d, 2, 28, 0.02)), 20, label="the first dip under a stricter threshold")
    equal(int(absolute_threshold(d, 2, 28, 0.001)), 20, label="the global minimum when no dip is under the threshold")
    equal(int(absolute_threshold(d, 2, 15, 0.001)), 11, label="the global minimum inside [tau_min, tau_max]")


@check("parabolic_interpolation finds the vertex of a sampled parabola")
def test_parabola(sub):
    parabolic_interpolation = has(sub, "parabolic_interpolation")
    lags = np.arange(12, dtype=float)
    for vertex in (5.0, 5.3, 4.62):
        y = 2.0 * (lags - vertex) ** 2 + 0.1
        close(parabolic_interpolation(y, int(round(vertex))), vertex, atol=1e-10, label=f"the vertex at {vertex}")
    close(parabolic_interpolation(lags, 0), 0.0, label="a minimum at the first lag, which has no left neighbor")


@check("yin finds clean tones within 0.5%, half-sample periods within 0.1%, and a missing fundamental")
def test_yin_clean(sub):
    yin = has(sub, "yin")
    t = np.arange(1024) / RATE
    for f0 in (97.0, 155.5, 220.0, 333.3):
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
        got, aperiodicity = yin(x, RATE, 60.0, 500.0, 0.1)
        if abs(got - f0) > 0.005 * f0:
            fail(f"yin returned {got:.2f} Hz for a harmonic tone at {f0} Hz; expected within 0.5%")
        if not 0.0 <= aperiodicity < 0.05:
            fail(f"the aperiodicity of a clean tone at {f0} Hz is {aperiodicity:.3f}; expected under 0.05")
    # A period of 45.5 samples falls between two lags; only interpolation gets it right.
    for period in (40.5, 45.5, 60.5):
        f0 = RATE / period
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
        got, _ = yin(x, RATE, 60.0, 500.0, 0.1)
        if abs(got - f0) > 0.001 * f0:
            fail(f"yin returned {got:.2f} Hz for a tone with a period of {period} samples, {f0:.2f} Hz; "
                 f"expected within 0.1%, which needs parabolic interpolation between lags")

    t = np.arange(1024) / RATE
    x = sum(np.sin(2 * np.pi * k * 150.0 * t + 0.7 * k) for k in range(2, 21))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(1024)))
    if spectrum[int(round(150 * 1024 / RATE))] > 0.05 * spectrum.max():
        fail("internal: the test signal should have no energy at 150 Hz")
    got, _ = yin(x, RATE, 60.0, 500.0, 0.1)
    if abs(got - 150.0) > 1.5:
        fail(f"yin returned {got:.1f} Hz; the period of harmonics 2 to 20 of 150 Hz is 1/150 s, so expected 150 Hz")


@check("gross_pitch_error and voicing_error score a track against the truth")
def test_metrics(sub):
    gross_pitch_error, voicing_error = has(sub, "gross_pitch_error"), has(sub, "voicing_error")
    ref = np.array([0.0, 100, 100, 200, 200, 0, 150, 150, 0, 0])
    est = np.array([0.0, 101, 50, 205, 250, 120, 0, 179, 0, 0])
    # Both voiced: frames 1, 2, 3, 4, 7. Off by more than 20%: frame 2 (50 vs 100) and frame 4 (250 vs 200 is 25%).
    close(gross_pitch_error(est, ref), 2 / 5, atol=1e-12, label="gross pitch error")
    close(gross_pitch_error(est, ref, tolerance=0.3), 1 / 5, atol=1e-12, label="gross pitch error with a 30% tolerance")
    # Voicing disagrees on frame 5 (a false alarm) and frame 6 (a miss).
    close(voicing_error(est, ref), 2 / 10, atol=1e-12, label="voicing decision error")


@check("autocorr_pitch returns the largest autocorrelation peak over the same lag range")
def test_autocorr(sub):
    autocorr_pitch = has(sub, "autocorr_pitch")
    rng = np.random.default_rng(1)
    for _ in range(5):
        f0 = rng.uniform(80, 400)
        x = harmonic_voice(rng, f0)
        lo, hi = int(np.floor(RATE / 500.0)), int(np.ceil(RATE / 60.0))
        W = 1024 - hi
        r = np.array([x[:W] @ x[tau : tau + W] for tau in range(hi + 1)])
        want = RATE / (lo + np.argmax(r[lo : hi + 1]))
        close(autocorr_pitch(x, RATE, 60.0, 500.0), want, rtol=1e-9, label=f"autocorr_pitch of a voice at {f0:.1f} Hz")


@check("yin makes under 2% gross errors on 60 noisy voices from 80 Hz to 400 Hz")
def test_yin_gpe(sub):
    yin, gross_pitch_error = has(sub, "yin"), has(sub, "gross_pitch_error")
    rng = np.random.default_rng(2)
    truth, got = [], []
    for _ in range(60):
        f0 = rng.uniform(80, 400)
        truth.append(f0)
        got.append(yin(harmonic_voice(rng, f0), RATE, 60.0, 500.0, 0.1)[0])
    gpe = gross_pitch_error(np.array(got), np.array(truth))
    if gpe >= 0.02:
        wrong = [(round(a), round(b)) for a, b in zip(got, truth) if abs(a - b) > 0.2 * b][:5]
        fail(f"yin's gross pitch error is {100 * gpe:.1f}%, expected under 2%. Some (estimate, truth) pairs: {wrong}")


@check("track_pitch follows a gliding voice, and marks noise and silence unvoiced")
def test_track(sub):
    track_pitch = has(sub, "track_pitch")
    rng = np.random.default_rng(3)
    n = RATE  # one second
    t = np.arange(n) / RATE
    f0_true = 110.0 + 120.0 * t  # a glide from 110 Hz to 230 Hz
    voiced = (t > 0.1) & (t < 0.45) | (t > 0.6) & (t < 0.9)
    phase = np.cumsum(2 * np.pi * f0_true / RATE)
    x = np.where(voiced, sum(np.sin(k * phase) / k for k in range(1, 12)), 0.0)
    x = x + np.where((t > 0.45) & (t < 0.6), 0.3 * rng.standard_normal(n), 0.0) + 0.001 * rng.standard_normal(n)
    f0, flags = track_pitch(x, RATE, 1024, 160, 60.0, 500.0, 0.1, 0.25)
    f0, flags = np.asarray(f0, dtype=float), np.asarray(flags, dtype=bool)
    shape(f0, (101,), label="f0 for one second at hop 160")
    shape(flags, (101,), label="the voicing flags")
    if np.any(f0[~flags] != 0):
        fail("f0 must be 0 in every frame marked unvoiced")
    centers = np.arange(101) * 160
    ref = np.where(voiced[np.minimum(centers, n - 1)], f0_true[np.minimum(centers, n - 1)], 0.0)
    # Frames within 32 ms of a voicing boundary mix both sides; score the rest.
    edges = np.array([0.1, 0.45, 0.6, 0.9]) * RATE
    clear = np.min(np.abs(centers[:, None] - edges[None, :]), axis=1) > 512
    disagree = float(np.mean((f0[clear] > 0) != (ref[clear] > 0)))
    if disagree > 0.05:
        fail(f"the voicing decision is wrong on {100 * disagree:.0f}% of frames away from the boundaries; expected at most 5%")
    both = clear & (f0 > 0) & (ref > 0)
    worst = float(np.max(np.abs(f0[both] - ref[both]) / ref[both]))
    if worst > 0.05:
        fail(f"a voiced frame's F0 is {100 * worst:.1f}% off the glide; expected every clear frame within 5%")
