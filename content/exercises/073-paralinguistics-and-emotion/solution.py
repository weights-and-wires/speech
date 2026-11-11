"""Paralinguistics and emotion: the worked solution."""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot

RATE = 16000


# Voice-quality and prosody descriptors ---------------------------------------------


def jitter(periods: np.ndarray) -> float:
    """Local jitter: the mean absolute difference of consecutive periods over the mean period."""
    t = np.asarray(periods, dtype=float)
    return float(np.mean(np.abs(np.diff(t))) / np.mean(t))


def shimmer(amplitudes: np.ndarray) -> float:
    """Local shimmer: the mean absolute difference of consecutive peak amplitudes over the mean amplitude."""
    a = np.asarray(amplitudes, dtype=float)
    return float(np.mean(np.abs(np.diff(a))) / np.mean(a))


def f0_stats(f0: np.ndarray) -> tuple[float, float]:
    """The mean F0 in hertz and the 20th-to-80th percentile range in semitones, over voiced frames."""
    voiced = np.asarray(f0, dtype=float)
    voiced = voiced[voiced > 0]
    p20, p80 = np.percentile(voiced, [20, 80])
    return float(np.mean(voiced)), float(12 * np.log2(p80 / p20))


# Agreement -------------------------------------------------------------------------


def ccc(pred: np.ndarray, gold: np.ndarray) -> float:
    """Lin's concordance correlation coefficient, with population (ddof=0) moments."""
    x = np.asarray(pred, dtype=float)
    y = np.asarray(gold, dtype=float)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def cohen_kappa(table: np.ndarray) -> float:
    """Cohen's kappa from a (K, K) table of counts: rows annotator A, columns annotator B."""
    t = np.asarray(table, dtype=float)
    n = t.sum()
    observed = np.trace(t) / n
    chance = np.sum(t.sum(axis=1) * t.sum(axis=0)) / n**2
    return float((observed - chance) / (1 - chance))


# Scoring a classifier --------------------------------------------------------------


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """A (K, K) matrix of counts: row = true class, column = predicted class."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(m, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return m


def accuracy(confusion: np.ndarray) -> float:
    """The share of all clips on the diagonal."""
    c = np.asarray(confusion, dtype=float)
    return float(np.trace(c) / c.sum())


def uar(confusion: np.ndarray) -> float:
    """Unweighted average recall: the mean over classes of the diagonal over its row sum."""
    c = np.asarray(confusion, dtype=float)
    return float(np.mean(np.diag(c) / c.sum(axis=1)))


# A linear classifier ---------------------------------------------------------------


def fit_linear(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1.0, balanced: bool = True) -> np.ndarray:
    """Weighted ridge regression onto one-hot targets. Returns W of shape (d + 1, K), bias in the last row."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(n_classes)[y]
    if balanced:
        w = n / (n_classes * np.bincount(y, minlength=n_classes))[y]
    else:
        w = np.ones(n)
    penalty = l2 * np.eye(Xb.shape[1])
    penalty[-1, -1] = 0.0
    A = Xb.T @ (w[:, None] * Xb) + penalty
    return np.linalg.solve(A, Xb.T @ (w[:, None] * Y))


def predict_linear(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The class with the largest score, for each row of X."""
    X = np.asarray(X, dtype=float)
    return np.argmax(np.hstack([X, np.ones((len(X), 1))]) @ W, axis=1)


# Demonstration ---------------------------------------------------------------------


def voice(periods: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """A buzzy vowel: a pulse of height `amplitudes[i]` at the start of cycle i, through a resonator at 700 Hz."""
    starts = np.concatenate([[0], np.cumsum(periods)]).astype(int)[:-1]
    pulses = np.zeros(int(np.sum(periods)) + 200)
    pulses[starts] = amplitudes
    r = 0.97
    a = [1.0, -2 * r * np.cos(2 * np.pi * 700 / RATE), r * r]
    out = lfilter([1.0], a, pulses)
    return 0.8 * out / np.max(np.abs(out))


CLASSES = ["neutral", "angry", "sad"]
# Per-class means of (F0 mean Hz, F0 range st, jitter, shimmer, energy dB, rate syll/s).
MEANS = np.array(
    [
        [120.0, 5.0, 0.010, 0.050, -24.0, 4.0],
        [150.0, 9.0, 0.015, 0.070, -18.0, 5.0],
        [105.0, 3.0, 0.014, 0.080, -30.0, 3.0],
    ]
)
SPREAD = np.array([14.0, 2.5, 0.004, 0.02, 4.0, 0.7])


def sample_utterances(counts: list[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = np.repeat(np.arange(len(counts)), counts)
    X = MEANS[y] + SPREAD * rng.standard_normal((len(y), MEANS.shape[1]))
    return X, y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_cycles = 90
    base = RATE / 120.0
    steady_t = base * (1 + 0.002 * rng.standard_normal(n_cycles))
    steady_a = 1 + 0.01 * rng.standard_normal(n_cycles)
    rough_t = base * (1 + 0.025 * rng.standard_normal(n_cycles))
    rough_a = 1 + 0.15 * rng.standard_normal(n_cycles)
    for name, t, a in [("steady", steady_t, steady_a), ("rough", rough_t, rough_a)]:
        print(f"{name} voice: jitter {jitter(t):.4f}, shimmer {shimmer(a):.4f}")
        play(voice(t, a), RATE, label=f"{name} voice")

    train_X, train_y = sample_utterances([300, 60, 20], rng)
    test_X, test_y = sample_utterances([300, 60, 20], rng)
    mu, sd = train_X.mean(axis=0), train_X.std(axis=0)
    train_X, test_X = (train_X - mu) / sd, (test_X - mu) / sd
    for balanced in (False, True):
        W = fit_linear(train_X, train_y, 3, balanced=balanced)
        cm = confusion_matrix(test_y, predict_linear(W, test_X), 3)
        print(f"\nbalanced={balanced}: accuracy {accuracy(cm):.3f}, UAR {uar(cm):.3f}")
        print(f"rows are true {CLASSES}, columns are predicted:")
        print(cm)

    gold = rng.uniform(-1, 1, 200)
    biased = 0.5 * gold + 0.4 + 0.05 * rng.standard_normal(200)
    r = float(np.corrcoef(biased, gold)[0, 1])
    print(f"\nA predictor that follows arousal but is compressed and shifted: Pearson {r:.3f}, CCC {ccc(biased, gold):.3f}")
    show_plot((gold[np.argsort(gold)], biased[np.argsort(gold)]), label="Predicted against gold arousal", xlabel="gold", ylabel="predicted")
