"""Pronunciation assessment: goodness of pronunciation and its scoring.

Fill in every function, then press Run.

`posteriors` is a `(T, P)` array of frame posteriors over P phones, and each row
sums to 1. `alignment` is an `(S, 3)` integer array with one row per phone
segment: `[phone_id, start_frame, end_frame]`, where `end_frame` is one past the
last frame. Clip posteriors at 1e-10 before taking logs.
"""

import numpy as np


def gop_log_posterior(posteriors: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """Return `(S,)`: the mean over a segment's frames of log P(canonical phone)."""
    raise NotImplementedError("gop_log_posterior")


def gop(posteriors: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """Return `(S,)`: the mean over a segment's frames of
    log P(canonical phone) - log max over phones of P. Every value is at most 0."""
    raise NotImplementedError("gop")


def detect_mispronunciations(gop_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return a boolean `(S,)` array that is True where GOP is below `threshold`."""
    raise NotImplementedError("detect_mispronunciations")


def precision_recall_f1(predicted: np.ndarray, gold: np.ndarray) -> tuple:
    """Return `(precision, recall, f1)` for boolean arrays, 0.0 when a ratio is
    undefined. A flagged phone is positive."""
    raise NotImplementedError("precision_recall_f1")


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Return float ranks starting at 1. Tied values share the mean of their ranks."""
    raise NotImplementedError("average_ranks")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation of two equal-length 1-D arrays."""
    raise NotImplementedError("pearson")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Spearman correlation, computed on tie-averaged ranks."""
    raise NotImplementedError("spearman")


def phone_confusion(canonical: np.ndarray, recognized: np.ndarray, n_phones: int) -> np.ndarray:
    """Return an `(n_phones, n_phones)` integer count matrix. Row = canonical
    phone, column = recognized phone, for aligned pairs `canonical[i], recognized[i]`."""
    raise NotImplementedError("phone_confusion")


if __name__ == "__main__":
    from speechlab import show_image, show_plot

    rng = np.random.default_rng(0)
    n_phones, n_frames = 5, 40
    canonical = np.repeat(np.arange(n_phones), 8)
    posteriors = rng.dirichlet(np.ones(n_phones) * 0.3, size=n_frames)
    posteriors[np.arange(n_frames), canonical] += 3.0
    posteriors[32:40] = np.roll(posteriors[32:40], 1, axis=1)  # phone 4 is replaced
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    alignment = np.array([[p, 8 * p, 8 * p + 8] for p in range(n_phones)])

    scores = gop(posteriors, alignment)
    print("GOP per phone", np.round(scores, 2))
    flagged = detect_mispronunciations(scores, -1.0)
    print("flagged", flagged, "precision, recall, F1",
          precision_recall_f1(flagged, np.array([0, 0, 0, 0, 1], bool)))
    human = np.array([3, 5, 5, 8, 9.0])
    model = np.array([-1.2, -0.9, -0.6, -0.5, -0.1])
    print("Pearson", round(pearson(model, human), 4),
          "Spearman", round(spearman(model, human), 4))
    recognized = np.array([3, 2, 0, 1, 3, 3])
    matrix = phone_confusion(np.array([2, 2, 0, 1, 2, 3]), recognized, 4)
    show_plot(scores, label="GOP per phone", xlabel="Phone index", ylabel="GOP (nats per frame)")
    show_image(matrix, label="Phone confusion", xlabel="Recognized", ylabel="Canonical")
