"""Checks for the dynamic time warping exercise."""

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from scipy.spatial.distance import cdist

from speechcheck import check, close, fail, has, shape, timed


def all_paths(n, m, allowed=None):
    """Every monotone path from (0, 0) to (n - 1, m - 1), by brute force."""
    out = []

    def walk(i, j, path):
        if allowed is not None and not allowed[i, j]:
            return
        path = path + [(i, j)]
        if (i, j) == (n - 1, m - 1):
            out.append(path)
            return
        for di, dj in ((1, 1), (1, 0), (0, 1)):
            if i + di < n and j + dj < m:
                walk(i + di, j + dj, path)

    walk(0, 0, [])
    return out


def path_cost(cost, path):
    return float(sum(cost[i, j] for i, j in path))


def band_mask(n, m, band):
    slope = (m - 1) / (n - 1)
    return np.abs(np.arange(m)[None, :] - np.arange(n)[:, None] * slope) <= band


def check_path(path, n, m, label):
    path = [tuple(int(v) for v in step) for step in path]
    if not path or path[0] != (0, 0):
        fail(f"{label} must start at (0, 0), got {path[:1]}")
    if path[-1] != (n - 1, m - 1):
        fail(f"{label} must end at ({n - 1}, {m - 1}), got {path[-1]}")
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        if (i1 - i0, j1 - j0) not in ((1, 0), (0, 1), (1, 1)):
            fail(f"{label} steps from {(i0, j0)} to {(i1, j1)}; each step must be (1, 0), (0, 1), or (1, 1)")
    return path


# ---------------------------------------------------------------------------
# Provided: synthetic spoken "words" and their features. You don't need to
# change anything in this block.
#
# A word is a sequence of phone-like segments. A vowel is a harmonic tone at a
# random pitch, shaped by three formant peaks. A fricative is a burst of
# band-pass noise. Every segment gets a random duration, and the whole word
# gets silence at both ends and white noise at a chosen SNR.
# ---------------------------------------------------------------------------

RATE = 16000
PHONES = ["sil", "a", "e", "i", "o", "u", "s", "sh", "f"]
FORMANTS = {"a": (800, 1200, 2500), "e": (500, 1800, 2500), "i": (300, 2300, 3000),
            "o": (500, 900, 2400), "u": (320, 800, 2300)}
NOISE_BANDS = {"s": (4000, 7000, 0.5), "sh": (2000, 4000, 0.6), "f": (1000, 7500, 0.25)}
WORDS = {"sa": ["s", "a"], "shi": ["sh", "i"], "fu": ["f", "u"], "ai": ["a", "i"],
         "ia": ["i", "a"], "osa": ["o", "s", "a"], "ue": ["u", "e"], "eshu": ["e", "sh", "u"]}


def _vowel(name, n, f0, rng):
    t = np.arange(n) / RATE
    out = np.zeros(n)
    for k in range(1, int(5000 // f0) + 1):
        gain = sum(1.0 / (1.0 + ((k * f0 - f) / 90.0) ** 2) for f in FORMANTS[name])
        out += gain * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(np.mean(out**2))


def _fricative(name, n, rng):
    low, high, level = NOISE_BANDS[name]
    sos = butter(4, [low, high], btype="band", fs=RATE, output="sos")
    out = sosfilt(sos, rng.standard_normal(n))
    return level * out / np.sqrt(np.mean(out**2))


def synth_word(phones, rng, snr_db=20.0):
    """Synthesize one utterance of a word, given as a list of phone names.

    Returns (samples, labels): float64 samples at RATE with peak 0.25, and the
    index into PHONES of the segment each sample belongs to (0 is silence).
    """
    f0 = rng.uniform(100, 160)
    ramp_n = int(0.01 * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp_n) / ramp_n)
    lead = int(rng.uniform(0.04, 0.08) * RATE)
    pieces, labels = [np.zeros(lead)], [np.zeros(lead, dtype=int)]
    for p in phones:
        if p in FORMANTS:
            n = int(rng.uniform(0.06, 0.30) * RATE)
            seg = _vowel(p, n, f0 * rng.uniform(0.95, 1.05), rng)
        else:
            n = int(rng.uniform(0.05, 0.18) * RATE)
            seg = _fricative(p, n, rng)
        seg[:ramp_n] *= ramp
        seg[-ramp_n:] *= ramp[::-1]
        pieces.append(seg)
        labels.append(np.full(n, PHONES.index(p)))
    tail = int(rng.uniform(0.04, 0.08) * RATE)
    pieces.append(np.zeros(tail))
    labels.append(np.zeros(tail, dtype=int))
    x, lab = np.concatenate(pieces), np.concatenate(labels)
    speech_rms = np.sqrt(np.mean(x[lab > 0] ** 2))
    x = x + rng.standard_normal(x.size) * speech_rms * 10 ** (-snr_db / 20)
    return 0.25 * x / np.max(np.abs(x)), lab


def log_mel(x, n_mels=24, n_fft=512, win=400, hop=160):
    """Log mel energies, shape (n_frames, n_mels): 25 ms frames every 10 ms."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(0, mel(RATE / 2), n_mels + 2) / 2595) - 1)
    freqs = np.arange(n_fft // 2 + 1) * RATE / n_fft
    low, mid, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.maximum(0, np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)))
    n_frames = 1 + (len(x) - win) // hop
    index = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    power = np.abs(np.fft.rfft(x[index] * window, n=n_fft, axis=1)) ** 2
    return np.log(power @ bank.T + 1e-8)


def mfcc(x, n_ceps=13):
    """Thirteen MFCCs per frame with the utterance mean removed, shape (n_frames, 13)."""
    c = dct(log_mel(x), type=2, norm="ortho", axis=1)[:, :n_ceps]
    return c - c.mean(axis=0)


@check("local_cost gives the Euclidean distance between every pair of frames")
def test_local_cost(sub):
    local_cost = has(sub, "local_cost")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    got = local_cost(a, b)
    shape(got, (5, 7), label="local_cost(a, b)")
    close(got, cdist(a, b), atol=1e-10, label="local_cost against scipy's cdist")


@check("accumulated_cost reproduces the 3 by 3 table worked in the lesson")
def test_hand_table(sub):
    accumulated_cost = has(sub, "accumulated_cost")
    cost = np.abs(np.array([0.0, 1.0, 2.0])[:, None] - np.array([0.0, 2.0, 2.0])[None, :])
    want = np.array([[0.0, 2.0, 4.0], [1.0, 1.0, 2.0], [3.0, 1.0, 1.0]])
    close(accumulated_cost(cost), want, atol=1e-12, label="D for a = [0, 1, 2], b = [0, 2, 2]")


@check("The last cell of accumulated_cost equals the cheapest of every warping path, found by brute force")
def test_brute_force(sub):
    accumulated_cost = has(sub, "accumulated_cost")
    rng = np.random.default_rng(1)
    for n, m in ((4, 5), (5, 3), (6, 6), (2, 7)):
        cost = rng.uniform(0, 1, size=(n, m))
        want = min(path_cost(cost, p) for p in all_paths(n, m))
        got = float(accumulated_cost(cost)[-1, -1])
        if abs(got - want) > 1e-10:
            fail(f"for a {n} by {m} cost matrix, D[-1, -1] is {got:.6f}, but the cheapest path costs {want:.6f}")


@check("A Sakoe-Chiba band blocks the cells outside it and finds the cheapest path inside it")
def test_band(sub):
    accumulated_cost = has(sub, "accumulated_cost")
    rng = np.random.default_rng(2)
    for n, m, band in ((6, 6, 1), (5, 7, 1.5), (7, 5, 2)):
        cost = rng.uniform(0, 1, size=(n, m))
        allowed = band_mask(n, m, band)
        D = np.asarray(accumulated_cost(cost, band), dtype=float)
        if np.any(np.isfinite(D[~allowed])):
            fail(f"with n={n}, m={m}, band={band}, cells outside the band must hold np.inf, "
                 f"but {int(np.isfinite(D[~allowed]).sum())} of them are finite")
        want = min(path_cost(cost, p) for p in all_paths(n, m, allowed))
        if abs(D[-1, -1] - want) > 1e-10:
            fail(f"with n={n}, m={m}, band={band}, D[-1, -1] is {D[-1, -1]:.6f}, "
                 f"but the cheapest path inside the band costs {want:.6f}")
    cost = rng.uniform(0, 1, size=(6, 8))
    close(accumulated_cost(cost, 100), accumulated_cost(cost), atol=1e-12,
          label="a band wider than the matrix, against no band")


@check("backtrack returns a valid path whose cost adds up to D[-1, -1]")
def test_backtrack(sub):
    accumulated_cost, backtrack = has(sub, "accumulated_cost"), has(sub, "backtrack")
    cost = np.abs(np.array([0.0, 1.0, 2.0])[:, None] - np.array([0.0, 2.0, 2.0])[None, :])
    path = check_path(backtrack(accumulated_cost(cost)), 3, 3, "the hand example's path")
    if path_cost(cost, path) != 1.0:
        fail(f"the hand example's path {path} costs {path_cost(cost, path)}, expected 1")
    rng = np.random.default_rng(3)
    for n, m in ((8, 5), (6, 9), (10, 10)):
        cost = rng.uniform(0, 1, size=(n, m))
        D = accumulated_cost(cost)
        path = check_path(backtrack(D), n, m, f"the path through a {n} by {m} table")
        if abs(path_cost(cost, path) - D[-1, -1]) > 1e-10:
            fail(f"the path's cells add up to {path_cost(cost, path):.6f}, but D[-1, -1] is {D[-1, -1]:.6f}")


@check("dtw absorbs a repeated frame, and gives the same distance in both directions")
def test_dtw(sub):
    dtw = has(sub, "dtw")
    a = np.array([[1.0], [2.0], [3.0]])
    b = np.array([[1.0], [1.0], [2.0], [3.0]])
    distance, path = dtw(a, b)
    if abs(distance) > 1e-12:
        fail(f"dtw([1, 2, 3], [1, 1, 2, 3]) should be 0, because warping absorbs the repeat; got {distance}")
    check_path(path, 3, 4, "the path for [1, 2, 3] against [1, 1, 2, 3]")
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(12, 4))
    forward, _ = dtw(x, y)
    reverse, _ = dtw(y, x)
    if abs(forward - reverse) > 1e-9:
        fail(f"dtw(x, y) is {forward:.6f} but dtw(y, x) is {reverse:.6f}; the unit-weight recurrence is symmetric")


@check("With a band of 0 on equal lengths, dtw is the plain frame-by-frame distance")
def test_band_zero(sub):
    dtw = has(sub, "dtw")
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
    distance, path = dtw(a, b, 0)
    want = float(np.sum(np.linalg.norm(a - b, axis=1)))
    if abs(distance - want) > 1e-9:
        fail(f"with band 0 the only path is the diagonal, costing {want:.6f}; dtw returned {distance:.6f}")
    if [tuple(int(v) for v in step) for step in path] != [(k, k) for k in range(10)]:
        fail(f"with band 0 the path must be the diagonal (0, 0), (1, 1), ..., (9, 9); got {path}")


@check("classify names 38 or more of 40 noisy synthetic words from two templates each")
def test_classify(sub):
    classify = has(sub, "classify")
    rng = np.random.default_rng(11)
    names = list(WORDS)
    templates, labels = [], []
    for name in names:
        for _ in range(2):
            templates.append(mfcc(synth_word(WORDS[name], rng)[0]))
            labels.append(name)
    correct, wrong = 0, []
    with timed("40 words against 16 templates"):
        for _ in range(5):
            for name in names:
                query = mfcc(synth_word(WORDS[name], rng, snr_db=10.0)[0])
                guess = classify(query, templates, labels, 15)
                if guess == name:
                    correct += 1
                else:
                    wrong.append(f"{name} as {guess}")
    if correct < 38:
        fail(f"{correct} of 40 words recognized, expected at least 38. Mistakes: {', '.join(wrong[:6])}. "
             "Check that you divide each distance by its path length before comparing.")
