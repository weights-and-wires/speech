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

    Compute it in a form that does not overflow for a large logit of either
    sign.
    """
    raise NotImplementedError("bce_with_logits")

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
    raise NotImplementedError("pit_bce_loss")

def pit_bce_grad(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The gradient of pit_bce_loss with respect to the logits, shape (T, C).

    The best ordering is a constant while you differentiate, so the gradient is
    the gradient of that one ordering's mean BCE.
    """
    raise NotImplementedError("pit_bce_grad")

def overlap_frames(activity: np.ndarray) -> np.ndarray:
    """Which frames have two or more active speakers.

    Args:
        activity: Boolean speaker activity, shape (speakers, frames).

    Returns:
        A boolean array of shape (frames,).
    """
    raise NotImplementedError("overlap_frames")

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
    raise NotImplementedError("der_components")

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
    raise NotImplementedError("attractor_count")

def attractor_activity(embeddings: np.ndarray, attractors: np.ndarray) -> np.ndarray:
    """Speaker activity from frame embeddings and attractors.

    Args:
        embeddings: One embedding per frame, shape (T, D).
        attractors: One attractor per speaker, shape (S, D).

    Returns:
        sigmoid(embeddings @ attractors.T), shape (T, S): the probability that
        each speaker is active in each frame.
    """
    raise NotImplementedError("attractor_activity")

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
    raise NotImplementedError("target_speaker_mask")

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
