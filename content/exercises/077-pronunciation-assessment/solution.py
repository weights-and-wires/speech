"""Reference solution for 077-pronunciation-assessment."""

import numpy as np

FLOOR = 1e-10


def _log_post(posteriors):
    return np.log(np.maximum(np.asarray(posteriors, dtype=float), FLOOR))


def gop_log_posterior(posteriors, alignment):
    """Mean log posterior of the canonical phone over each segment's frames."""
    logp = _log_post(posteriors)
    out = np.zeros(len(alignment))
    for i, (phone, start, end) in enumerate(alignment):
        out[i] = logp[start:end, phone].mean()
    return out


def gop(posteriors, alignment):
    """Mean over frames of log P(canonical) minus log of the largest posterior."""
    logp = _log_post(posteriors)
    out = np.zeros(len(alignment))
    for i, (phone, start, end) in enumerate(alignment):
        block = logp[start:end]
        out[i] = (block[:, phone] - block.max(axis=1)).mean()
    return out


def detect_mispronunciations(gop_scores, threshold):
    """Flag a phone as mispronounced when its GOP is below the threshold."""
    return np.asarray(gop_scores, dtype=float) < threshold


def precision_recall_f1(predicted, gold):
    """Precision, recall, and F1 of the flagged phones, 0.0 when undefined."""
    predicted = np.asarray(predicted, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    tp = int(np.sum(predicted & gold))
    fp = int(np.sum(predicted & ~gold))
    fn = int(np.sum(~predicted & gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def average_ranks(x):
    """Ranks from 1, where tied values share the mean of the ranks they span."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tied = x == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def pearson(x, y):
    """Pearson correlation of two equal-length 1-D arrays."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def spearman(x, y):
    """Spearman correlation: Pearson on tie-averaged ranks."""
    return pearson(average_ranks(x), average_ranks(y))


def phone_confusion(canonical, recognized, n_phones):
    """Count matrix with canonical phones on rows and recognized phones on columns."""
    matrix = np.zeros((n_phones, n_phones), dtype=int)
    np.add.at(matrix, (np.asarray(canonical), np.asarray(recognized)), 1)
    return matrix


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
