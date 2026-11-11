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
    raise NotImplementedError("stitch_chunks")


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
    raise NotImplementedError("word_timestamps")


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
    raise NotImplementedError("ece")


def repetition_score(words: list[str], min_repeats: int = 3, max_block: int = 6) -> float:
    """The fraction of words inside a loop, between 0 and 1.

    A loop is a block of 1 to `max_block` words that repeats back to back at
    least `min_repeats` times. Every word of every repeat counts once, even
    where loops overlap. Empty text scores 0.
    """
    raise NotImplementedError("repetition_score")


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
    raise NotImplementedError("subgroup_wer_with_ci")


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
