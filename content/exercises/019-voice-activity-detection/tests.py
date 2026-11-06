"""Checks for the voice activity detection exercise."""

import numpy as np
from scipy.signal import lfilter

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000


def noisy_speech(seed, snr_db, seconds=3.0):
    """Voiced bursts through formant filters, in pink noise at snr_db; truth is one flag per 10 ms frame."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    truth, speech, start = np.zeros(n, dtype=bool), np.zeros(n), 0.2
    while True:
        length = rng.uniform(0.2, 0.6)
        if start + length > seconds - 0.2:
            break
        span = (t >= start) & (t < start + length)
        truth |= span
        envelope = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0.0, 1.0) * span * rng.uniform(0.5, 1.0)
        f0 = rng.uniform(90.0, 250.0)
        phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))) / RATE
        speech += envelope * sum(np.sin(k * phase) / k for k in range(1, int(4000 // f0)))
        start += length + rng.uniform(0.15, 0.5)
    for f, bw in [(rng.uniform(400, 800), 80), (rng.uniform(1000, 2000), 100), (2500, 150)]:
        r = np.exp(-np.pi * bw / RATE)
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r]
        speech = lfilter([sum(a)], a, speech)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(np.arange(len(spectrum)), 1)), n)
    noise = pink * np.sqrt(np.mean(speech[truth] ** 2) / np.mean(pink**2) / 10 ** (snr_db / 10))
    mix = speech + noise
    return 0.5 * mix / np.max(np.abs(mix)), truth[np.minimum(np.arange(1 + n // 160) * 160, n - 1)]


@check("frame_energy_db is the log mean power of centered frames")
def test_energy(sub):
    frame_energy_db = has(sub, "frame_energy_db")
    x = np.random.default_rng(0).standard_normal(4000) * 0.1
    got = np.asarray(frame_energy_db(x, 400, 160), dtype=float)
    shape(got, (26,), label="energies of 4000 samples at hop 160")
    padded = np.pad(x, 200, mode="reflect")
    want = [10 * np.log10(np.mean(padded[t * 160 : t * 160 + 400] ** 2) + 1e-10) for t in range(26)]
    close(got, want, atol=1e-9, label="10 log10(mean power + 1e-10)")
    tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(8000) / RATE)
    close(np.asarray(frame_energy_db(tone))[5:-5], np.full(41, 10 * np.log10(0.125)), atol=0.01,
          label="a sine of amplitude 0.5, whose mean power is 0.125, or -9.03 dB")
    close(np.asarray(frame_energy_db(np.zeros(1600))), np.full(11, -100.0), atol=1e-9, label="silence, floored at -100 dB")


@check("hysteresis switches on above on_db and off only after the hangover")
def test_hysteresis(sub):
    hysteresis = has(sub, "hysteresis")
    e = np.array([0, 12, 8, 3, 3, 3, 9, 2, 2], dtype=float)
    equal(np.asarray(hysteresis(e, 10.0, 5.0, 2), dtype=bool),
          np.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool), label="the worked example, hangover 2")
    equal(np.asarray(hysteresis(e, 10.0, 5.0, 0), dtype=bool),
          np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool), label="the same energies with no hangover")
    e2 = np.array([11, 3, 6, 3, 3, 3, 11], dtype=float)
    equal(np.asarray(hysteresis(e2, 10.0, 5.0, 2), dtype=bool),
          np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool), label="a frame back above off_db resets the hangover count")


@check("frame_f1 scores decisions by 2 TP / (2 TP + FP + FN)")
def test_f1(sub):
    frame_f1 = has(sub, "frame_f1")
    pred = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    ref = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
    close(frame_f1(pred, ref), 2 * 2 / (2 * 2 + 1 + 1), atol=1e-12, label="F1 with TP 2, FP 1, FN 1")
    close(frame_f1(ref, ref), 1.0, label="F1 of a perfect detector")
    close(frame_f1(~ref, ref), 0.0, atol=1e-12, label="F1 of a detector that is wrong on every frame")
    close(frame_f1(np.zeros(5, bool), np.zeros(5, bool)), 1.0, label="F1 when neither track has speech")


@check("energy_vad reaches frame F1 0.9 at 20 dB and 10 dB SNR")
def test_vad(sub):
    energy_vad, frame_f1 = has(sub, "energy_vad"), has(sub, "frame_f1")
    for snr in (20.0, 10.0):
        for seed in range(3):
            mix, truth = noisy_speech(seed, snr)
            decisions = np.asarray(energy_vad(mix, RATE), dtype=bool)
            shape(decisions, truth.shape, label="the VAD decisions, one per 10 ms frame")
            score = frame_f1(decisions, truth)
            if score < 0.9:
                fail(f"energy_vad's frame F1 is {score:.3f} on noisy speech at {snr:.0f} dB SNR (seed {seed}); expected at least 0.9")


@check("energy_vad ignores the recording's gain, because its thresholds sit above its own noise floor")
def test_gain(sub):
    energy_vad = has(sub, "energy_vad")
    mix, _ = noisy_speech(1, 15.0)
    base = np.asarray(energy_vad(mix, RATE), dtype=bool)
    for gain in (0.1, 0.3):
        quieter = np.asarray(energy_vad(gain * mix, RATE), dtype=bool)
        different = int(np.sum(quieter != base))
        if different > 2:
            fail(f"scaling the recording by {gain} changed {different} decisions; thresholds relative to the noise floor should change at most 2")
