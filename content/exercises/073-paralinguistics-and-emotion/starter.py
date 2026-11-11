"""Paralinguistics and emotion.

Fill in the descriptors (jitter, shimmer, f0_stats), the agreement measures
(ccc, cohen_kappa), the scores (confusion_matrix, accuracy, uar), and a linear
classifier (fit_linear, predict_linear). Press Check to test them, and Run to
hear a steady and a rough voice, train an emotion classifier on an imbalanced
set, and compare CCC with Pearson correlation.
"""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot

RATE = 16000


# Voice-quality and prosody descriptors ---------------------------------------------


def jitter(periods: np.ndarray) -> float:
    """Local jitter of a sequence of glottal periods (in any unit, one value per cycle).

    The mean absolute difference of consecutive periods, divided by the mean
    of all the periods. Returns a fraction, so 0.01 means 1%.
    """
    raise NotImplementedError("jitter")


def shimmer(amplitudes: np.ndarray) -> float:
    """Local shimmer of a sequence of per-cycle peak amplitudes.

    The mean absolute difference of consecutive amplitudes, divided by the
    mean of all the amplitudes. Returns a fraction.
    """
    raise NotImplementedError("shimmer")


def f0_stats(f0: np.ndarray) -> tuple[float, float]:
    """Summarize an F0 track in hertz, where 0 marks an unvoiced frame.

    Returns (mean F0 in hertz, range in semitones), both over voiced frames
    only. The range is 12 * log2(p80 / p20), with p20 and p80 the 20th and 80th
    percentiles (np.percentile, linear interpolation) of the voiced values.
    """
    raise NotImplementedError("f0_stats")


# Agreement -------------------------------------------------------------------------


def ccc(pred: np.ndarray, gold: np.ndarray) -> float:
    """Lin's concordance correlation coefficient between two 1-D arrays.

    2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2), where every
    moment uses the population convention (divide by n, not n - 1).
    """
    raise NotImplementedError("ccc")


def cohen_kappa(table: np.ndarray) -> float:
    """Cohen's kappa from a (K, K) table of counts: rows annotator A, columns annotator B.

    (observed agreement - chance agreement) / (1 - chance agreement), where
    chance agreement sums, over classes, the product of the two annotators'
    class shares.
    """
    raise NotImplementedError("cohen_kappa")


# Scoring a classifier --------------------------------------------------------------


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """A (K, K) integer matrix of counts: row = true class, column = predicted class."""
    raise NotImplementedError("confusion_matrix")


def accuracy(confusion: np.ndarray) -> float:
    """The share of all clips that sit on the diagonal of a confusion matrix."""
    raise NotImplementedError("accuracy")


def uar(confusion: np.ndarray) -> float:
    """Unweighted average recall: the mean over classes of (diagonal / row sum)."""
    raise NotImplementedError("uar")


# A linear classifier ---------------------------------------------------------------


def fit_linear(X: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 1.0, balanced: bool = True) -> np.ndarray:
    """Weighted ridge regression of one-hot labels on [X, 1]. Returns W of shape (d + 1, K).

    X is (n, d), y holds integer classes. Append a column of ones for the bias,
    which is the last row of W and is not penalized. Each clip i has weight w_i:
    n / (K * count of its class) when `balanced`, else 1. Solve
    (Xb^T diag(w) Xb + l2 * P) W = Xb^T diag(w) Y, where P is the identity
    with a 0 for the bias.
    """
    raise NotImplementedError("fit_linear")


def predict_linear(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The class with the largest score [X, 1] @ W, for each row of X, as an integer array."""
    raise NotImplementedError("predict_linear")


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
