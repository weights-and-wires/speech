"""Long-form and trustworthy ASR: stitching, word times, calibration, loops, and subgroup WER.

Transcripts are lists of words. A cross-attention matrix has one row per token
and one column per audio frame.
"""

import numpy as np

from speechlab import show_image


def stitch_chunks(chunk_outputs: list[list[str]], overlap: int, min_run: int = 2) -> list[str]:
    """Join the word lists of overlapping windows into one transcript.

    Args:
        chunk_outputs: One word list per window, in time order.
        overlap: How many words at each seam to search. The tail of the text so
            far and the head of the next window are each cut to this length.
        min_run: The shortest common run that counts as a match.

    Returns:
        The stitched word list. At each seam, find the longest run of words
        that appears in both the tail and the head (the earliest in the head
        on a tie). Keep the text so far up to the end of that run, then
        continue with the next window after the run. When no run is at least
        `min_run` words long, append the whole next window.
    """
    merged = list(chunk_outputs[0])
    for chunk in chunk_outputs[1:]:
        chunk = list(chunk)
        offset = max(0, len(merged) - overlap)
        tail, head = merged[offset:], chunk[:overlap]
        best_k, best_i, best_j = 0, 0, 0
        for j in range(len(head)):
            for i in range(len(tail)):
                k = 0
                while i + k < len(tail) and j + k < len(head) and tail[i + k] == head[j + k]:
                    k += 1
                if k > best_k:
                    best_k, best_i, best_j = k, i, j
        if best_k >= min_run:
            merged = merged[: offset + best_i + best_k] + chunk[best_j + best_k:]
        else:
            merged = merged + chunk
    return merged


def dtw_path(cost: np.ndarray) -> list[tuple[int, int]]:
    """The cheapest monotone path from (0, 0) to (n - 1, m - 1), as (row, column) pairs.

    Steps are down, right, or diagonal. This is the routine from the dynamic
    time warping chapter, given here so you can focus on what to do with it.
    """
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    i, j = n, m
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        steps = [(acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
        path.append((i - 1, j - 1))
    return path[::-1]


def word_timestamps(attn: np.ndarray, word_token_spans: list[tuple[int, int]], hop: float) -> np.ndarray:
    """Start and end of every word in seconds, shape (n_words, 2), from cross-attention.

    Args:
        attn: Cross-attention weights, shape (n_tokens, n_frames). A large
            value means the token listens to that frame.
        word_token_spans: One (first_token, end_token) pair per word, with
            end_token one past the word's last token, covering the tokens in order.
        hop: Seconds per frame.

    Returns:
        First normalize every frame (column) to zero mean and unit standard
        deviation across the tokens, as `(a - mean) / (std + 1e-6)`, because
        raw weights are all positive and DTW would then prefer a longer path.
        Warp the tokens onto the frames with `dtw_path` on the cost `-normalized`.
        A token's jump time is the first frame at which the path is in its row.
        A word starts at the jump time of its first token and ends where the
        next word starts. The last word ends at `n_frames * hop`.
    """
    attn = np.asarray(attn, dtype=float)
    z = (attn - attn.mean(axis=0, keepdims=True)) / (attn.std(axis=0, keepdims=True) + 1e-6)
    path = dtw_path(-z)
    jump = np.zeros(attn.shape[0], dtype=int)
    seen = set()
    for row, col in path:
        if row not in seen:
            seen.add(row)
            jump[row] = col
    starts = np.array([jump[first] for first, _ in word_token_spans], dtype=float)
    ends = np.append(starts[1:], attn.shape[1])
    return np.stack([starts, ends], axis=1) * hop


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by samples.

    Args:
        conf: Confidences in [0, 1], shape (n,).
        correct: 1 where the word was right and 0 where it was wrong, shape (n,).
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        The sum over bins of (bin count / n) * |accuracy - mean confidence|.
        Bins are right-closed, (0, 1/M], (1/M, 2/M], and so on, and a
        confidence of exactly 0 falls in the first bin.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    which = np.searchsorted(edges, conf, side="left")
    total = 0.0
    for b in range(n_bins):
        mask = which == b
        if mask.any():
            total += mask.sum() / len(conf) * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def repetition_score(words: list[str], min_repeats: int = 3, max_block: int = 6) -> float:
    """The fraction of words inside a loop, between 0 and 1.

    A loop is a block of 1 to `max_block` words that repeats back to back at
    least `min_repeats` times. Every word of every repeat counts once, even
    where loops overlap. Empty text scores 0.
    """
    words = list(words)
    marked = np.zeros(len(words), dtype=bool)
    for n in range(1, max_block + 1):
        for i in range(len(words) - n + 1):
            r = 1
            while words[i + r * n: i + (r + 1) * n] == words[i: i + n]:
                r += 1
            if r >= min_repeats:
                marked[i: i + r * n] = True
    return float(marked.mean()) if len(words) else 0.0


def subgroup_wer_with_ci(errors, ref_lens, groups, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Corpus WER and a bootstrap interval for every subgroup.

    Args:
        errors: Word errors per utterance, shape (n,).
        ref_lens: Reference words per utterance, shape (n,).
        groups: A group label per utterance, such as an accent.
        n_boot: Bootstrap resamples per group.
        seed: Seed for one `np.random.default_rng` shared by all groups.
        alpha: The interval covers the central 1 - alpha of the resamples.

    Returns:
        A dict from group label to (wer, low, high). `wer` is summed errors
        over summed reference words. Visit groups in sorted order. For each,
        draw `rng.integers(0, n_g, size=(n_boot, n_g))` utterance indices and
        take the percentiles `100 * alpha / 2` and `100 * (1 - alpha / 2)` of
        the resampled WERs.
    """
    errors = np.asarray(errors, dtype=float)
    ref_lens = np.asarray(ref_lens, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    out = {}
    for g in sorted(set(groups.tolist())):
        e, n = errors[groups == g], ref_lens[groups == g]
        picks = rng.integers(0, len(e), size=(n_boot, len(e)))
        wers = e[picks].sum(axis=1) / n[picks].sum(axis=1)
        low, high = np.percentile(wers, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[g] = (float(e.sum() / n.sum()), float(low), float(high))
    return out


if __name__ == "__main__":
    # 1. Two 8-word windows that share the words "to be self evident".
    a = "we hold these truths to be self evident".split()
    b = "to be self evident that all men".split()
    print("Stitched:", " ".join(stitch_chunks([a, b], overlap=6)))

    # 2. Word times from a planted cross-attention matrix: 3 tokens over 8 frames of 20 ms.
    attn = np.full((3, 8), 0.05)
    attn[0, 0:2] = attn[1, 2:6] = attn[2, 6:8] = 0.9
    show_image(attn, label="Cross-attention, tokens down and frames across", xlabel="frame (20 ms)", ylabel="token")
    print("Word times (s):", word_timestamps(attn, [(0, 1), (1, 3)], 0.02).round(2).tolist())

    # 3. Calibration: a recognizer that says 0.9 and is right 60% of the time.
    rng = np.random.default_rng(0)
    conf = np.full(1000, 0.9)
    correct = (rng.random(1000) < 0.6).astype(float)
    print(f"ECE of an overconfident recognizer: {ece(conf, correct):.3f}")

    # 4. A loop and clean text.
    loop = "thanks for watching thanks for watching thanks for watching".split()
    print("Loop score:", repetition_score(loop), "clean score:", repetition_score("she said that that was fine".split()))

    # 5. WER per accent with a 95% interval.
    errors = [0, 1, 0, 2, 1, 3, 2, 4]
    lens = [10, 9, 11, 10, 10, 12, 9, 11]
    accent = ["a", "a", "a", "a", "b", "b", "b", "b"]
    for g, (w, lo, hi) in subgroup_wer_with_ci(errors, lens, accent).items():
        print(f"Accent {g}: WER {w:.1%}, 95% interval [{lo:.1%}, {hi:.1%}]")
