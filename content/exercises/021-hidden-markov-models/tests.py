"""Checks for the hidden Markov model exercise."""

import itertools

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from speechcheck import check, close, equal, fail, has, shape, timed


# The two-state, three-symbol toy HMM worked in the lesson.
TOY_PI = np.array([0.6, 0.4])
TOY_A = np.array([[0.7, 0.3], [0.4, 0.6]])
TOY_B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
TOY_O = [0, 1, 2]


def toy_logs():
    return np.log(TOY_PI), np.log(TOY_A), np.log(TOY_B[:, TOY_O].T)


def random_hmm(rng, N, T, K=4):
    """A random discrete HMM and observation sequence, as log arrays."""
    pi = rng.dirichlet(np.ones(N))
    A = rng.dirichlet(np.ones(N), size=N)
    B = rng.dirichlet(np.ones(K), size=N)
    O = rng.integers(0, K, size=T)
    return np.log(pi), np.log(A), np.log(B[:, O].T), (pi, A, B, O)


def scaled_forward_log_likelihood(pi, A, B, O):
    """Rabiner's scaled forward pass in linear space: an independent reference."""
    alpha = pi * B[:, O[0]]
    total = np.log(alpha.sum())
    alpha /= alpha.sum()
    for o in O[1:]:
        alpha = (alpha @ A) * B[:, o]
        total += np.log(alpha.sum())
        alpha /= alpha.sum()
    return total


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


@check("log_gaussian_diag gives -0.9189 for a standard normal at its mean")
def test_log_gaussian_hand(sub):
    log_gaussian_diag = has(sub, "log_gaussian_diag")
    got = log_gaussian_diag(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
    close(got, [[-0.5 * np.log(2 * np.pi)]], atol=1e-12, label="log N(0; 0, 1)")
    # One frame at (2, 2), one state with mean 0 and variance 4 in two dimensions.
    got = log_gaussian_diag(np.array([[2.0, 2.0]]), np.zeros((1, 2)), np.full((1, 2), 4.0))
    want = -0.5 * (2 * np.log(2 * np.pi) + 2 * np.log(4.0) + 2 * (4.0 / 4.0))
    close(got, [[want]], atol=1e-12, label="log density with variance 4")


@check("log_gaussian_diag matches scipy's multivariate normal log density")
def test_log_gaussian(sub):
    log_gaussian_diag = has(sub, "log_gaussian_diag")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    means = rng.normal(size=(4, 3))
    variances = rng.uniform(0.2, 2.0, size=(4, 3))
    got = log_gaussian_diag(X, means, variances)
    shape(got, (6, 4), label="log_gaussian_diag(X, means, variances)")
    want = np.stack([multivariate_normal(means[j], np.diag(variances[j])).logpdf(X) for j in range(4)], axis=1)
    close(got, want, atol=1e-10, label="log densities")


@check("forward reproduces the toy HMM's alphas and P(O) = 0.03628")
def test_forward_toy(sub):
    forward = has(sub, "forward")
    log_alpha, log_p = forward(*toy_logs())
    shape(log_alpha, (3, 2), label="log_alpha")
    want = np.array([[0.3, 0.04], [0.0904, 0.0342], [0.007696, 0.028584]])
    close(np.exp(log_alpha), want, atol=1e-12, label="alpha")
    close(np.exp(log_p), 0.03628, atol=1e-12, label="P(O)")


@check("forward stays finite over 2,000 frames and matches a scaled linear-space pass")
def test_forward_long(sub):
    forward = has(sub, "forward")
    rng = np.random.default_rng(2)
    log_pi, log_A, log_B, (pi, A, B, O) = random_hmm(rng, 4, 2000)
    with timed("forward over 2000 frames"):
        log_alpha, log_p = forward(log_pi, log_A, log_B)
    if not np.isfinite(log_p):
        fail(f"log P(O) over 2000 frames came out as {log_p}; work in log space with logsumexp")
    want = scaled_forward_log_likelihood(pi, A, B, O)
    if abs(log_p - want) > 1e-8 * abs(want):
        fail(f"log P(O) is {log_p:.10f}, but a scaled forward pass gives {want:.10f}")


@check("forward survives a forbidden transition, where log_A is -inf")
def test_forward_zero(sub):
    forward = has(sub, "forward")
    log_pi = np.array([0.0, -np.inf])
    with np.errstate(divide="ignore"):
        log_A = np.log([[0.5, 0.5], [0.0, 1.0]])
    log_B = np.log([[0.5, 0.2], [0.4, 0.6], [0.3, 0.9]])
    log_alpha, log_p = forward(log_pi, log_A, log_B)
    # Start in state 0 with probability 1. Paths: 000, 001, 011. State 1 never leads back to 0.
    want = 0.5 * (0.5 * 0.4 * 0.5 * 0.3 + 0.5 * 0.4 * 0.5 * 0.9 + 0.5 * 0.6 * 1.0 * 0.9)
    close(np.exp(log_p), want, atol=1e-12, label="P(O) with a forbidden transition")
    if not np.all(np.isfinite(log_alpha[:, 0])):
        fail("log_alpha for the state that can still be reached must stay finite")



@check("viterbi finds the best state path, checked on the toy and against brute force")
def test_viterbi(sub):
    viterbi = has(sub, "viterbi")
    path, log_p = viterbi(*toy_logs())
    equal(np.asarray(path), np.array([0, 0, 1]), label="the toy's best path")
    close(np.exp(log_p), 0.01512, atol=1e-12, label="the toy's best path probability")
    rng = np.random.default_rng(4)
    for trial in range(3):
        log_pi, log_A, log_B, _ = random_hmm(rng, 3, 6)
        best, best_path = -np.inf, None
        for q in itertools.product(range(3), repeat=6):
            score = log_pi[q[0]] + log_B[0, q[0]] + sum(
                log_A[q[t - 1], q[t]] + log_B[t, q[t]] for t in range(1, 6))
            if score > best:
                best, best_path = score, q
        path, log_p = viterbi(log_pi, log_A, log_B)
        if abs(log_p - best) > 1e-9:
            fail(f"trial {trial}: viterbi's log probability is {log_p:.6f}, but brute force over "
                 f"all 729 paths finds {best:.6f} for the path {best_path}")
        equal(np.asarray(path), np.array(best_path), label=f"trial {trial}'s best path")


@check("The best path is never more probable than the sum over every path")
def test_viterbi_vs_forward(sub):
    forward, viterbi = has(sub, "forward"), has(sub, "viterbi")
    rng = np.random.default_rng(7)
    for trial in range(5):
        log_pi, log_A, log_B, _ = random_hmm(rng, 4, 40)
        _, log_p = forward(log_pi, log_A, log_B)
        _, best = viterbi(log_pi, log_A, log_B)
        if best > log_p + 1e-9:
            fail(f"trial {trial}: the best path has log probability {best:.4f}, above log P(O) = {log_p:.4f}, "
                 "which sums over every path including that one")
    # With one state there is one path, so the two agree.
    log_pi, log_A, log_B, _ = random_hmm(rng, 1, 10)
    close(viterbi(log_pi, log_A, log_B)[1], forward(log_pi, log_A, log_B)[1], atol=1e-10,
          label="a one-state model has one path")
