"""Checks for the Baum-Welch training exercise."""

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


def three_state_data(rng, n_sequences=4):
    """Sequences from a three-state left-to-right source with well-separated means."""
    true_means = np.array([[0.0, 0.0], [3.0, 1.0], [0.0, 4.0]])
    sequences = []
    for _ in range(n_sequences):
        states = np.repeat([0, 1, 2], rng.integers(8, 20, size=3))
        sequences.append(true_means[states] + rng.normal(size=(len(states), 2)))
    return sequences


def left_to_right_start(rng):
    A = np.array([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]])
    with np.errstate(divide="ignore"):
        return A, (np.log([1.0, 0.0, 0.0]), np.log(A), rng.normal(size=(3, 2)), np.ones((3, 2)))


@check("backward gives beta_1 = [0.106, 0.112] on the toy, and alpha times beta gives P(O) at every frame")
def test_backward(sub):
    forward, backward = has(sub, "forward"), has(sub, "backward")
    log_pi, log_A, log_B = toy_logs()
    log_beta = backward(log_A, log_B)
    shape(log_beta, (3, 2), label="log_beta")
    close(np.exp(log_beta[0]), [0.106, 0.112], atol=1e-12, label="beta at the first frame")
    close(log_beta[-1], [0.0, 0.0], atol=1e-12, label="log beta at the last frame")
    rng = np.random.default_rng(3)
    log_pi, log_A, log_B, _ = random_hmm(rng, 5, 300)
    _, log_p = forward(log_pi, log_A, log_B)
    log_alpha, _ = forward(log_pi, log_A, log_B)
    per_frame = logsumexp(log_alpha + backward(log_A, log_B), axis=1)
    close(per_frame, np.full(300, log_p), atol=1e-8, label="log sum_i alpha_t(i) beta_t(i) at every t")


@check("baum_welch_step on a one-state model gives the sample mean and variance")
def test_baum_welch_one_state(sub):
    step = has(sub, "baum_welch_step")
    rng = np.random.default_rng(5)
    sequences = [rng.normal(2.0, 1.5, size=(40, 3)), rng.normal(2.0, 1.5, size=(25, 3))]
    everything = np.concatenate(sequences)
    out = step(sequences, np.zeros(1), np.zeros((1, 1)), np.zeros((1, 3)), np.ones((1, 3)))
    if len(out) != 5:
        fail(f"baum_welch_step must return (log_pi, log_A, means, variances, log_likelihood); got {len(out)} values")
    _, _, means, variances, total = out
    close(means, everything.mean(axis=0, keepdims=True), atol=1e-10, label="re-estimated mean")
    close(variances, everything.var(axis=0, keepdims=True), atol=1e-10, label="re-estimated variance")
    want = float(np.sum(multivariate_normal(np.zeros(3), np.eye(3)).logpdf(everything)))
    close(total, want, atol=1e-8, label="log-likelihood under the parameters passed in")


@check("baum_welch_step never lowers the likelihood over eight iterations")
def test_baum_welch_monotone(sub):
    step = has(sub, "baum_welch_step")
    rng = np.random.default_rng(6)
    sequences = three_state_data(rng)
    A, params = left_to_right_start(rng)
    history = []
    for _ in range(8):
        *params, total = step(sequences, *params)
        history.append(total)
    drops = [(k, history[k], history[k + 1]) for k in range(7) if history[k + 1] < history[k] - 1e-6]
    if drops:
        k, before, after = drops[0]
        fail(f"the log-likelihood fell from {before:.4f} to {after:.4f} after iteration {k + 1}; "
             "EM must never lower it")
    if history[-1] <= history[0]:
        fail(f"eight steps should raise the log-likelihood from {history[0]:.2f}; it ended at {history[-1]:.2f}")


@check("baum_welch_step keeps every row of A summing to 1, and keeps a zero transition at zero")
def test_baum_welch_structure(sub):
    step = has(sub, "baum_welch_step")
    rng = np.random.default_rng(6)
    sequences = three_state_data(rng)
    A, params = left_to_right_start(rng)
    for _ in range(3):
        *params, _ = step(sequences, *params)
    log_A = np.asarray(params[1])
    close(np.exp(log_A).sum(axis=1), np.ones(3), atol=1e-9, label="each row of the re-estimated A")
    if np.any(np.exp(log_A)[A == 0] > 0):
        fail("a transition that started at probability 0 became nonzero; Baum-Welch keeps zeros at zero")



@check("train_word_hmm records one rising log-likelihood per iteration and returns a valid model")
def test_train_word_hmm(sub):
    train_word_hmm = has(sub, "train_word_hmm")
    rng = np.random.default_rng(8)
    takes = three_state_data(rng)
    params, history = train_word_hmm(takes, 3, 6)
    if len(history) != 6:
        fail(f"history must hold one log-likelihood per Baum-Welch step, 6 of them; got {len(history)}")
    drops = [k for k in range(5) if history[k + 1] < history[k] - 1e-6]
    if drops:
        fail(f"the log-likelihood fell after iteration {drops[0] + 1}: {history}")
    log_pi, log_A, means, variances = params
    close(np.exp(log_A).sum(axis=1), np.ones(3), atol=1e-9, label="each row of the trained A")
    shape(means, (3, 2), label="trained means")


@check("An HMM per word, trained with train_word_hmm, recognizes 36 or more of 40 noisy words")
def test_recognizer(sub):
    train_word_hmm, recognize = has(sub, "train_word_hmm"), has(sub, "recognize")
    rng = np.random.default_rng(12)
    names = list(WORDS)
    models = {}
    with timed("training 8 word models"):
        for name in names:
            takes = [mfcc(synth_word(WORDS[name], rng)[0]) for _ in range(4)]
            params, history = train_word_hmm(takes, 2 * len(WORDS[name]) + 2, 5)
            models[name] = params
    correct, wrong = 0, []
    for _ in range(5):
        for name in names:
            guess = recognize(mfcc(synth_word(WORDS[name], rng, snr_db=10.0)[0]), models)
            if guess == name:
                correct += 1
            else:
                wrong.append(f"{name} as {guess}")
    if correct < 36:
        fail(f"{correct} of 40 words recognized, expected at least 36. Mistakes: {', '.join(wrong[:6])}")
