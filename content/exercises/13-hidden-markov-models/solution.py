"""Hidden Markov models in log space, and an isolated-word recognizer: the worked solution."""

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from scipy.special import logsumexp

from speechlab import play, show_image, show_plot


def log_gaussian_diag(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """log N(x_t; mu_j, diag(var_j)) for every frame t and state j, shape (T, N)."""
    D = X.shape[1]
    diff2 = (X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
    return -0.5 * (D * np.log(2 * np.pi) + np.sum(np.log(variances), axis=1)[None, :] + diff2.sum(axis=2))


def log_gmm(X: np.ndarray, weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """log sum_m c_jm N(x_t; mu_jm, diag(var_jm)) for every frame and state, shape (T, N)."""
    N, M, D = means.shape
    per_component = log_gaussian_diag(X, means.reshape(N * M, D), variances.reshape(N * M, D))
    per_component = per_component.reshape(len(X), N, M)
    with np.errstate(divide="ignore"):
        return logsumexp(per_component + np.log(weights)[None, :, :], axis=2)


def forward(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, float]:
    """The forward table log alpha, shape (T, N), and log P(O | model)."""
    T, N = log_B.shape
    log_alpha = np.empty((T, N))
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]
    return log_alpha, float(logsumexp(log_alpha[-1]))


def backward(log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """The backward table log beta, shape (T, N), with log beta[T - 1] = 0."""
    T, N = log_B.shape
    log_beta = np.zeros((T, N))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)
    return log_beta


def viterbi(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, float]:
    """The most probable state sequence, shape (T,), and its log probability."""
    T, N = log_B.shape
    delta = log_pi + log_B[0]
    psi = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        scores = delta[:, None] + log_A  # scores[i, j]: best path ending in i, then i -> j
        psi[t] = np.argmax(scores, axis=0)
        delta = scores[psi[t], np.arange(N)] + log_B[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(T - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path, float(delta[path[-1]])


def baum_welch_step(sequences, log_pi, log_A, means, variances, var_floor=1e-3):
    """One EM update of an HMM with one diagonal Gaussian per state.

    Returns (log_pi, log_A, means, variances, total_log_likelihood), where the
    log-likelihood is that of the sequences under the parameters passed in.
    """
    N, D = means.shape
    occupancy = np.zeros(N)
    first = np.zeros(N)
    leaving = np.zeros(N)  # expected visits at t = 0 .. T - 2
    transitions = np.zeros((N, N))
    sum_x = np.zeros((N, D))
    sum_x2 = np.zeros((N, D))
    total = 0.0
    for X in sequences:
        log_B = log_gaussian_diag(X, means, variances)
        log_alpha, log_p = forward(log_pi, log_A, log_B)
        log_beta = backward(log_A, log_B)
        total += log_p
        gamma = np.exp(log_alpha + log_beta - log_p)
        log_xi = (log_alpha[:-1, :, None] + log_A[None, :, :]
                  + (log_B[1:] + log_beta[1:])[:, None, :] - log_p)
        transitions += np.exp(log_xi).sum(axis=0)
        first += gamma[0]
        leaving += gamma[:-1].sum(axis=0)
        occupancy += gamma.sum(axis=0)
        sum_x += gamma.T @ X
        sum_x2 += gamma.T @ X**2
    with np.errstate(divide="ignore", invalid="ignore"):
        new_log_pi = np.log(first / first.sum())
        new_log_A = np.where(leaving[:, None] > 0, np.log(transitions / leaving[:, None]), log_A)
    used = occupancy > 1e-10
    new_means = np.where(used[:, None], sum_x / np.maximum(occupancy, 1e-10)[:, None], means)
    raw_var = sum_x2 / np.maximum(occupancy, 1e-10)[:, None] - new_means**2
    new_variances = np.where(used[:, None], np.maximum(raw_var, var_floor), variances)
    return new_log_pi, new_log_A, new_means, new_variances, float(total)


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


def train_word_hmm(sequences, n_states, n_iter=5):
    """Flat-start a left-to-right HMM, then run n_iter Baum-Welch steps.

    Returns ((log_pi, log_A, means, variances), log-likelihoods before each step).
    """
    params = flat_start(sequences, n_states)
    history = []
    for _ in range(n_iter):
        *params, total = baum_welch_step(sequences, *params)
        history.append(total)
    return tuple(params), history


def recognize(X, models):
    """The name of the model with the highest forward log-likelihood for X."""
    scores = {}
    for name, (log_pi, log_A, means, variances) in models.items():
        scores[name] = forward(log_pi, log_A, log_gaussian_diag(X, means, variances))[1]
    return max(scores, key=scores.get)


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

    # One left-to-right HMM per word: two states per phone and one for each silence.
    models, curves = {}, []
    for name in names:
        takes = [mfcc(synth_word(WORDS[name], rng)[0]) for _ in range(4)]
        params, history = train_word_hmm(takes, 2 * len(WORDS[name]) + 2, n_iter=6)
        models[name] = params
        curves.append(np.array(history) / sum(len(x) for x in takes))
    show_plot(*curves[:6], labels=names[:6], label="Baum-Welch: log-likelihood per frame never falls",
              xlabel="Iteration", ylabel="Log-likelihood per frame (nats)")
    print("Log-likelihood per frame for 'osa', one entry per iteration:",
          np.round(curves[names.index("osa")], 2))

    # Align one new take of "osa" with Viterbi and look at the states it uses.
    x, labels = synth_word(WORDS["osa"], rng)
    play(x, RATE, label='A new take of "osa"')
    X = mfcc(x)
    log_pi, log_A, means, variances = models["osa"]
    path, best = viterbi(log_pi, log_A, log_gaussian_diag(X, means, variances))
    occupancy = np.zeros((len(means), len(X)))
    occupancy[path, np.arange(len(X))] = 1.0
    show_image(occupancy, label="The Viterbi state path for the new take", xlabel="Frame",
               ylabel="State", origin="lower", colormap="gray")
    print(f"Viterbi log probability {best:.1f}; frames per state: {np.bincount(path, minlength=len(means))}")

    # Recognize 40 noisy words.
    correct = 0
    for _ in range(5):
        for name in names:
            correct += recognize(mfcc(synth_word(WORDS[name], rng, snr_db=10.0)[0]), models) == name
    print(f"HMM recognizer accuracy at 10 dB SNR: {correct} of 40 words")
