"""Score and align a sequence with a hidden Markov model, in log space.

Fill in the three functions below, then press Run to build one left-to-right
HMM per synthetic word, align a new take of "osa" with Viterbi, and score 40
noisy words against all eight models. Press Check to test your code.

Everything is in log space: log_pi has shape (N,), log_A has shape (N, N)
with log_A[i, j] = log P(next state j | state i), and log_B has shape (T, N)
with log_B[t, j] = log p(o_t | state j). log(0) is -np.inf.
"""

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from scipy.special import logsumexp

from speechlab import play, show_image


def log_gaussian_diag(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Diagonal Gaussian log densities for every frame and every state.

    Args:
        X: Observations, shape (T, D).
        means: One mean per state, shape (N, D).
        variances: One variance vector per state, shape (N, D), all positive.

    Returns:
        Shape (T, N). Entry [t, j] is log N(X[t]; means[j], diag(variances[j])).
    """
    raise NotImplementedError("log_gaussian_diag")


def forward(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, float]:
    """The forward algorithm in log space.

    Args:
        log_pi: Log initial probabilities, shape (N,).
        log_A: Log transition probabilities, shape (N, N); rows are "from".
        log_B: Log emission likelihoods, shape (T, N).

    Returns:
        (log_alpha, log_likelihood): log_alpha has shape (T, N), with
        log_alpha[t, j] = log P(o_1 .. o_t, state j at t). log_likelihood is
        log P(O), a float.
    """
    raise NotImplementedError("forward")


def viterbi(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, float]:
    """The single most probable state sequence.

    Args:
        log_pi, log_A, log_B: As in forward.

    Returns:
        (path, log_probability): path is an int array, shape (T,), of state
        indices; log_probability is the joint log P(O, path), a float.
    """
    raise NotImplementedError("viterbi")


def flat_start(sequences, n_states, self_loop=0.6):
    """A left-to-right HMM initialized from an even split of every sequence.

    Provided. State s starts from the mean and variance of the s-th of
    n_states equal chunks of each sequence. The model starts in state 0 and
    can only stay put or move one state to the right.
    """
    chunks = [[] for _ in range(n_states)]
    for X in sequences:
        for s, part in enumerate(np.array_split(X, n_states)):
            chunks[s].append(part)
    means = np.array([np.concatenate(c).mean(axis=0) for c in chunks])
    variances = np.array([np.maximum(np.concatenate(c).var(axis=0), 1e-3) for c in chunks])
    A = np.eye(n_states) * self_loop + np.eye(n_states, k=1) * (1 - self_loop)
    A[-1, -1] = 1.0
    with np.errstate(divide="ignore"):
        return np.log(np.eye(n_states)[0]), np.log(A), means, variances


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


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    names = list(WORDS)

    # One left-to-right HMM per word, started from an even split of four takes.
    # Two states per phone and one for each silence. No training yet.
    models = {}
    for name in names:
        takes = [mfcc(synth_word(WORDS[name], rng)[0]) for _ in range(4)]
        models[name] = flat_start(takes, 2 * len(WORDS[name]) + 2)

    # Align a new take of "osa" with Viterbi, and compare with the forward score.
    x, _ = synth_word(WORDS["osa"], rng)
    play(x, RATE, label='A new take of "osa"')
    X = mfcc(x)
    log_pi, log_A, means, variances = models["osa"]
    log_B = log_gaussian_diag(X, means, variances)
    path, best = viterbi(log_pi, log_A, log_B)
    _, total = forward(log_pi, log_A, log_B)
    occupancy = np.zeros((len(means), len(X)))
    occupancy[path, np.arange(len(X))] = 1.0
    show_image(occupancy, label="The Viterbi state path for the new take", xlabel="Frame",
               ylabel="State", origin="lower", colormap="gray")
    print(f"frames per state: {np.bincount(path, minlength=len(means))}")
    print(f"log P(best path) = {best:.1f}, log P(O) summed over every path = {total:.1f}")

    # Score 40 noisy words against every model and answer with the highest forward score.
    correct = 0
    for _ in range(5):
        for name in names:
            X = mfcc(synth_word(WORDS[name], rng, snr_db=10.0)[0])
            scores = {}
            for other, (log_pi, log_A, means, variances) in models.items():
                scores[other] = forward(log_pi, log_A, log_gaussian_diag(X, means, variances))[1]
            correct += max(scores, key=scores.get) == name
    print(f"Untrained HMMs, forward-score recognizer at 10 dB SNR: {correct} of 40 words")
