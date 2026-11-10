"""Checks for the voice activity detection exercise."""

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000


def noisy_speech(seed, snr_db, seconds=3.0, step_db=0.0, step_at=None):
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
    if step_at is not None:  # the noise rises by step_db dB over 50 ms, like a fan switching on
        noise = noise * 10 ** (step_db / 20 * np.clip((t - step_at) / 0.05, 0.0, 1.0))
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


def features(x):
    """Per-frame [log energy in dB, spectral flatness in dB], computed here so a check does not depend on the reader's code."""
    padded = np.pad(x, 200, mode="reflect")
    frames = padded[np.arange(400)[None, :] + 160 * np.arange(1 + len(x) // 160)[:, None]]
    energy = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    band = (np.abs(np.fft.rfft(frames * np.hanning(400), axis=1)) ** 2 + 1e-12)[:, 3:101]
    return np.stack([energy, 10 * np.log10(np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1))], axis=1)


@check("track_noise_floor falls fast, rises slowly, and freezes on speech frames")
def test_floor(sub):
    track = has(sub, "track_noise_floor")
    # The wrong version updates the floor on speech frames too, so the floor climbs into the speech.
    e = np.array([-40, -40, -20, -20, -40], dtype=float)
    mask = np.array([0, 0, 1, 1, 0], dtype=bool)
    close(track(e, 1.0, 0.5, mask), [-40, -40, -40, -40, -40], atol=1e-9,
          label="floor with the two loud frames marked as speech (frozen, not climbing 1 dB per frame)")
    close(track(e, 1.0, 0.5), [-40, -40, -39, -38, -39], atol=1e-9,
          label="floor with no mask (up by at most 1 dB per frame, then down by half the gap)")
    e2 = np.array([-30, -40, -40, -30, -30, -30], dtype=float)
    close(track(e2, 2.0, 0.5), [-30, -35, -37.5, -35.5, -33.5, -31.5], atol=1e-9,
          label="floor on a drop and a rise: down by down_alpha * gap, up by at most 2 dB")
    mix, truth = noisy_speech(0, 20.0)
    energy = np.asarray(has(sub, "frame_energy_db")(mix), dtype=float)
    floor = np.asarray(track(energy, 0.3, 0.3, truth), dtype=float)
    quiet_level = float(np.median(floor[~truth]))
    if floor[truth].max() - quiet_level > 3.0:
        fail(f"with the true speech frames frozen, the floor reached {floor[truth].max():.1f} dB inside speech "
             f"while the quiet floor is {quiet_level:.1f} dB; expected a rise of at most 3 dB")


@check("adaptive_vad follows a noise step that defeats a fixed floor")
def test_adaptive(sub):
    adaptive_vad, energy_vad, frame_f1 = has(sub, "adaptive_vad"), has(sub, "energy_vad"), has(sub, "frame_f1")
    frame_energy_db = has(sub, "frame_energy_db")
    for seed in range(3):
        mix, truth = noisy_speech(seed, 20.0, seconds=4.0, step_db=10.0, step_at=2.0)
        energy = np.asarray(frame_energy_db(mix), dtype=float)
        decisions, floor = adaptive_vad(energy)
        shape(np.asarray(decisions), truth.shape, label="the adaptive decisions, one per 10 ms frame")
        adaptive_score, fixed_score = frame_f1(np.asarray(decisions, dtype=bool), truth), frame_f1(energy_vad(mix), truth)
        if adaptive_score < 0.85 or adaptive_score < fixed_score + 0.05:
            fail(f"seed {seed}, noise up 10 dB at 2 s: adaptive frame F1 {adaptive_score:.3f}, fixed-floor frame F1 "
                 f"{fixed_score:.3f}; expected adaptive at least 0.85 and 0.05 above the fixed floor")
        late = np.flatnonzero(~truth)[-40:]
        if abs(float(np.median(np.asarray(floor)[late])) - float(np.median(energy[late]))) > 4.0:
            fail(f"seed {seed}: the floor at the end is {np.median(np.asarray(floor)[late]):.1f} dB, expected within 4 dB "
                 f"of the new noise level {np.median(energy[late]):.1f} dB")
    mix, truth = noisy_speech(0, 20.0)
    score = frame_f1(np.asarray(adaptive_vad(np.asarray(frame_energy_db(mix), dtype=float))[0], dtype=bool), truth)
    if score < 0.9:
        fail(f"adaptive_vad's frame F1 is {score:.3f} on steady noise at 20 dB SNR; expected at least 0.9")


@check("gaussian_llr is the log density ratio of two diagonal Gaussians")
def test_llr(sub):
    llr = has(sub, "gaussian_llr")
    # The wrong version forgets the -0.5 log(variance) term, which cancels only when the variances are equal.
    got = np.asarray(llr(np.array([[0.0]]), [0.0], [4.0], [0.0], [1.0]), dtype=float)
    close(got, [-0.5 * np.log(4.0)], atol=1e-9, label="LLR at x = 0 when both means are 0 and the variances are 4 and 1")
    x = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]])
    sm, sv, nm, nv = np.array([1.0, 0.0]), np.array([1.0, 4.0]), np.array([0.0, 2.0]), np.array([1.0, 0.25])
    want = (norm.logpdf(x, sm, np.sqrt(sv)) - norm.logpdf(x, nm, np.sqrt(nv))).sum(axis=1)
    got = np.asarray(llr(x, sm, sv, nm, nv), dtype=float)
    shape(got, (3,), label="one LLR per frame")
    close(got, want, atol=1e-9, label="LLR of three two-feature frames against scipy.stats.norm.logpdf")
    close(np.asarray(llr(x, sm, sv, sm, sv), dtype=float), np.zeros(3), atol=1e-12, label="LLR when both Gaussians are the same")


@check("a smoothed energy-and-flatness LLR detector reaches frame F1 0.9 at 5 dB SNR")
def test_llr_vad(sub):
    llr, hysteresis, frame_f1 = has(sub, "gaussian_llr"), has(sub, "hysteresis"), has(sub, "frame_f1")
    train = [noisy_speech(100 + s, 5.0) for s in range(3)]
    f, labels = np.concatenate([features(m) for m, _ in train]), np.concatenate([r for _, r in train])
    sm, sv, nm, nv = f[labels].mean(0), f[labels].var(0), f[~labels].mean(0), f[~labels].var(0)
    for seed in range(4):
        mix, truth = noisy_speech(seed, 5.0)
        smooth = np.convolve(np.asarray(llr(features(mix), sm, sv, nm, nv), dtype=float), np.ones(5) / 5, mode="same")
        score = frame_f1(np.asarray(hysteresis(smooth, 1.0, -1.0, 5), dtype=bool), truth)
        if score < 0.9:
            fail(f"the LLR detector's frame F1 is {score:.3f} on noisy speech at 5 dB SNR (seed {seed}); expected at least 0.9")


@check("endpoint returns the frame where the quiet count first exceeds the hangover")
def test_endpoint(sub):
    endpoint = has(sub, "endpoint")
    # The wrong version never resets the count after a short pause, so it ends the utterance at frame 7.
    v = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    equal(int(endpoint(v, 2)), 9, label="end frame with hangover 2 (the pause at frames 4 and 5 is not long enough)")
    equal(int(endpoint(v, 3)), 10, label="end frame with hangover 3")
    equal(int(endpoint(np.zeros(10, dtype=bool), 2)), -1, label="end frame when nobody speaks")
    equal(int(endpoint(np.array([1, 1, 0, 0], dtype=bool), 2)), -1, label="end frame when the quiet run is only 2 frames")
    equal(int(endpoint(np.array([1, 1, 0, 0], dtype=bool), 1)), 3, label="end frame with hangover 1")
