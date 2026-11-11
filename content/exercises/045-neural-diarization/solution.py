"""Neural diarization: permutation-invariant loss, overlap scoring, attractors, and a target-speaker mask.

Activity matrices have one row per frame and one column per speaker, so a
frame with two active speakers has two ones in its row. The scoring functions
take the boolean (speakers, frames) layout the previous chapter used.
"""

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from speechlab import show_image


def bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy over every frame and speaker, from logits.

    Args:
        logits: Raw scores, shape (T, C), for T frames and C speaker slots.
        labels: Targets in {0, 1}, shape (T, C).

    Use the stable form max(z, 0) - z * y + log(1 + exp(-|z|)) so that a large
    logit of either sign does not overflow.
    """
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def pit_bce_loss(logits: np.ndarray, labels: np.ndarray) -> tuple[float, tuple[int, ...]]:
    """Permutation-invariant BCE: the smallest mean BCE over all slot orderings.

    Args:
        logits: Raw scores, shape (T, C).
        labels: Targets in {0, 1}, shape (T, C).

    Returns:
        (loss, perm). perm[c] is the label column that output slot c is scored
        against, so the loss is bce_with_logits(logits, labels[:, list(perm)]).
        One ordering covers the whole clip. On a tie, return the first
        ordering in itertools.permutations order.
    """
    best_loss, best_perm = None, None
    for perm in itertools.permutations(range(logits.shape[1])):
        loss = bce_with_logits(logits, labels[:, list(perm)])
        if best_loss is None or loss < best_loss:
            best_loss, best_perm = loss, perm
    return best_loss, best_perm


def pit_bce_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The gradient of pit_bce_loss with respect to the logits, shape (T, C).

    The best ordering is a constant while you differentiate, so the gradient is
    that ordering's BCE gradient: (sigmoid(z) - y_perm) / (T * C).
    """
    _, perm = pit_bce_loss(logits, labels)
    y = np.asarray(labels, dtype=float)[:, list(perm)]
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
    return (p - y) / y.size


def overlap_frames(activity: np.ndarray) -> np.ndarray:
    """Which frames have two or more active speakers.

    Args:
        activity: Boolean speaker activity, shape (speakers, frames).

    Returns:
        A boolean array of shape (frames,).
    """
    return np.asarray(activity, dtype=bool).sum(axis=0) >= 2


def der_components(reference: np.ndarray, hypothesis: np.ndarray) -> dict:
    """Missed speech, false alarm, and confusion in speaker-frames, with the denominator.

    Args:
        reference: Boolean activity, shape (speakers, frames).
        hypothesis: Boolean activity, shape (slots, frames). Slots are
            unordered, so map them to reference speakers one to one first,
            choosing the mapping that matches the most speaker-frames.

    Returns:
        A dict with float values for "miss", "false_alarm", "confusion", "total"
        (reference speaker-frames, counting an overlapped frame once per
        speaker), and "der", their sum over the total. Per frame, with n_ref
        and n_hyp active speakers and n_ok matched ones: miss is
        max(n_ref - n_hyp, 0), false alarm is max(n_hyp - n_ref, 0), and
        confusion is min(n_ref, n_hyp) - n_ok.
    """
    ref = np.asarray(reference, dtype=bool)
    hyp = np.asarray(hypothesis, dtype=bool)
    table = ref.astype(int) @ hyp.T.astype(int)
    rows, cols = linear_sum_assignment(-table)
    ok = np.zeros(ref.shape[1], dtype=int)
    for r, c in zip(rows, cols):
        ok += ref[r] & hyp[c]
    n_ref, n_hyp = ref.sum(axis=0), hyp.sum(axis=0)
    miss = float(np.maximum(n_ref - n_hyp, 0).sum())
    false_alarm = float(np.maximum(n_hyp - n_ref, 0).sum())
    confusion = float((np.minimum(n_ref, n_hyp) - ok).sum())
    total = float(n_ref.sum())
    return {
        "miss": miss,
        "false_alarm": false_alarm,
        "confusion": confusion,
        "total": total,
        "der": (miss + false_alarm + confusion) / total,
    }


def attractor_count(attractor_probs: np.ndarray, threshold: float = 0.5) -> int:
    """How many speakers an encoder-decoder attractor model finds.

    Args:
        attractor_probs: The existence probability of each attractor, in the
            order the decoder produced them, shape (S,).
        threshold: Stop at the first probability below this value.

    Returns:
        The number of attractors before the first one below the threshold. A
        later probability above the threshold does not count, because the
        decoder has already stopped.
    """
    probs = np.asarray(attractor_probs, dtype=float)
    below = np.flatnonzero(probs < threshold)
    return int(below[0]) if below.size else int(probs.size)


def attractor_activity(embeddings: np.ndarray, attractors: np.ndarray) -> np.ndarray:
    """Speaker activity from frame embeddings and attractors.

    Args:
        embeddings: One embedding per frame, shape (T, D).
        attractors: One attractor per speaker, shape (S, D).

    Returns:
        sigmoid(embeddings @ attractors.T), shape (T, S): the probability that
        each speaker is active in each frame.
    """
    z = np.asarray(embeddings, dtype=float) @ np.asarray(attractors, dtype=float).T
    return 1.0 / (1.0 + np.exp(-z))


def target_speaker_mask(frame_embeddings: np.ndarray, enrollment: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Frames that sound like the enrolled speaker, by cosine similarity.

    Args:
        frame_embeddings: One embedding per frame, shape (T, D).
        enrollment: The target speaker's embedding, shape (D,).
        threshold: A frame is the target when its cosine is at least this.

    Returns:
        A boolean array of shape (T,). Normalize both sides first, so loudness
        does not change the answer.
    """
    f = np.asarray(frame_embeddings, dtype=float)
    e = np.asarray(enrollment, dtype=float)
    cos = (f @ e) / (np.linalg.norm(f, axis=1) * np.linalg.norm(e) + 1e-12)
    return cos >= threshold


if __name__ == "__main__":
    ref = np.zeros((2, 14), dtype=bool)
    ref[0, 0:8] = True
    ref[1, 5:12] = True
    hyp = np.zeros((2, 14), dtype=bool)
    hyp[1, 0:10] = True
    hyp[0, 10:14] = True
    print("overlap frames:", np.flatnonzero(overlap_frames(ref)).tolist())
    print("DER components:", der_components(ref, hyp))
    labels = ref.T.astype(float)
    logits = np.where(hyp.T, 4.0, -4.0)
    loss, perm = pit_bce_loss(logits, labels)
    print(f"fixed-order BCE {bce_with_logits(logits, labels):.3f}, PIT BCE {loss:.3f} with slot order {perm}")
    show_image(np.vstack([ref, hyp]).astype(float), label="reference rows 0-1, system slots 2-3", xlabel="frame")
    print("attractors found:", attractor_count(np.array([0.93, 0.81, 0.20, 0.74]), 0.5))
