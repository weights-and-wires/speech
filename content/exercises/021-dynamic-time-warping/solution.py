"""Dynamic time warping and a template recognizer: the worked solution."""

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, sosfilt

from speechlab import play, show_image


def local_cost(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every frame of `a` and every frame of `b`."""
    return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))


def in_band(n: int, m: int, band: float | None) -> np.ndarray:
    """Boolean mask, shape (n, m), of the cells a Sakoe-Chiba band allows."""
    if band is None:
        return np.ones((n, m), dtype=bool)
    slope = (m - 1) / (n - 1) if n > 1 else 0.0
    i = np.arange(n)[:, None]
    j = np.arange(m)[None, :]
    return np.abs(j - i * slope) <= band


def accumulated_cost(cost: np.ndarray, band: float | None = None) -> np.ndarray:
    """The DTW table D, shape (n, m), with np.inf in every cell outside the band."""
    n, m = cost.shape
    allowed = in_band(n, m, band)
    D = np.full((n, m), np.inf)
    for i in range(n):
        for j in range(m):
            if not allowed[i, j]:
                continue
            if i == 0 and j == 0:
                best = 0.0
            else:
                up = D[i - 1, j] if i > 0 else np.inf
                left = D[i, j - 1] if j > 0 else np.inf
                diag = D[i - 1, j - 1] if i > 0 and j > 0 else np.inf
                best = min(diag, up, left)
            D[i, j] = cost[i, j] + best
    return D


def backtrack(D: np.ndarray) -> list[tuple[int, int]]:
    """The warping path from (0, 0) to (n - 1, m - 1), read back from D."""
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i, j)]
    while (i, j) != (0, 0):
        steps = []
        if i > 0 and j > 0:
            steps.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            steps.append((D[i - 1, j], i - 1, j))
        if j > 0:
            steps.append((D[i, j - 1], i, j - 1))
        _, i, j = min(steps, key=lambda step: step[0])  # min keeps the first on a tie
        path.append((i, j))
    return path[::-1]


def dtw(a: np.ndarray, b: np.ndarray, band: float | None = None) -> tuple[float, list[tuple[int, int]]]:
    """The DTW distance D[n - 1, m - 1] between two sequences, and its path."""
    D = accumulated_cost(local_cost(a, b), band)
    if not np.isfinite(D[-1, -1]):
        return float("inf"), []
    return float(D[-1, -1]), backtrack(D)


def classify(
    query: np.ndarray, templates: list[np.ndarray], labels: list[str], band: float | None = None
) -> str:
    """The label of the template nearest to `query` in average cost per path step."""
    best_label, best_score = None, np.inf
    for template, label in zip(templates, labels):
        total, path = dtw(query, template, band)
        score = total / len(path) if path else np.inf
        if score < best_score:
            best_label, best_score = label, score
    return best_label


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
    rng = np.random.default_rng(7)

    # Two takes of "osa" at different speeds, and one take of "ai".
    x1, _ = synth_word(WORDS["osa"], rng)
    x2, _ = synth_word(WORDS["osa"], rng)
    x3, _ = synth_word(WORDS["ai"], rng)
    play(x1, RATE, label='"osa", first take')
    play(x2, RATE, label='"osa", second take')
    a, b, c = mfcc(x1), mfcc(x2), mfcc(x3)
    print(f'The "osa" takes are {len(a)} and {len(b)} frames long; "ai" is {len(c)}.')

    cost = local_cost(a, b)
    path = backtrack(accumulated_cost(cost, band=15))
    picture = cost.copy()
    for i, j in path:
        picture[i, j] = 0.0  # draw the path as a dark line on the cost matrix
    show_image(picture, label="Frame distances between the two takes, with the warping path",
               xlabel="Second take (frame)", ylabel="First take (frame)", colormap="magma")
    same, same_path = dtw(a, b, band=15)
    other, other_path = dtw(a, c, band=15)
    print(f"Average cost per path step: osa to osa {same / len(same_path):.2f}, "
          f"osa to ai {other / len(other_path):.2f}")

    # A template recognizer: two templates per word, 40 noisy test words.
    names = list(WORDS)
    templates, labels = [], []
    for name in names:
        for _ in range(2):
            templates.append(mfcc(synth_word(WORDS[name], rng)[0]))
            labels.append(name)
    confusion = np.zeros((len(names), len(names)))
    for _ in range(5):
        for k, name in enumerate(names):
            query = mfcc(synth_word(WORDS[name], rng, snr_db=10.0)[0])
            confusion[k, names.index(classify(query, templates, labels, band=15))] += 1
    accuracy = np.trace(confusion) / confusion.sum()
    show_image(confusion, label="Recognizer confusion matrix at 10 dB SNR", xlabel="Predicted",
               ylabel="Spoken", x_labels=names, y_labels=names)
    print(f"Template recognizer accuracy at 10 dB SNR: {accuracy:.0%} of {int(confusion.sum())} words")
