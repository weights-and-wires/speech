"""CTC by brute force: collapse, enumerate, and sum.

Fill in the five functions below, then press Run to list every path that
spells a label in the worked example, compare your loss with torch, and see
how fast the number of paths grows. Press Check to test your code.

Conventions, the same as torch.nn.functional.ctc_loss: blank is index 0, and
log_probs are natural-log probabilities, shape (T, C): one row per frame, one
column per symbol, each row already normalized. A target is a list of label
indices, none of them blank. A path is a tuple of T symbol indices.
"""

import itertools

import numpy as np
import torch

from speechlab import show_image, show_plot


def collapse(path, blank: int = 0) -> list[int]:
    """The CTC collapse function.

    Args:
        path: One symbol per frame, any sequence of ints.
        blank: The blank index.

    Returns:
        A list of ints: first merge each run of identical consecutive symbols
        into one, then remove every blank.
    """
    raise NotImplementedError("collapse")


def path_log_prob(log_probs: np.ndarray, path) -> float:
    """The log probability of one path.

    Args:
        log_probs: Shape (T, C).
        path: T symbol indices, one per frame.

    Returns:
        The sum over frames t of log_probs[t, path[t]], as a float.
    """
    raise NotImplementedError("path_log_prob")


def paths_for(target, n_frames: int, n_symbols: int, blank: int = 0) -> list[tuple[int, ...]]:
    """Every path that collapses to the target.

    Args:
        target: L label indices.
        n_frames: The number of frames T.
        n_symbols: The number of symbols C, blank included.
        blank: The blank index.

    Returns:
        A list of tuples of length n_frames, in itertools.product order, whose
        collapse equals the target. It is empty when no path fits.
    """
    raise NotImplementedError("paths_for")


def ctc_loss_brute_force(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """The CTC loss, by summing over every path.

    Args:
        log_probs: Shape (T, C). Keep T and C tiny: there are C**T paths.
        target: L label indices.
        blank: The blank index.

    Returns:
        -log of the summed probability of every path that collapses to the
        target, as a float. float("inf") when no path fits in T frames.
    """
    raise NotImplementedError("ctc_loss_brute_force")


def min_frames(target) -> int:
    """The fewest frames that can spell the target.

    Args:
        target: L label indices.

    Returns:
        L plus the number of adjacent equal pairs in the target.
    """
    raise NotImplementedError("min_frames")


if __name__ == "__main__":
    # The worked example: blank = 0, a = 1, P(a) = 0.6, 0.3, 0.8 in three frames.
    p_a = np.array([0.6, 0.3, 0.8])
    log_probs = np.log(np.stack([1 - p_a, p_a], axis=1))
    show_image(np.exp(log_probs).T, label="Frame posteriors: row 0 is blank, row 1 is a",
               xlabel="Frame", ylabel="Symbol", y_labels=["blank", "a"], vmin=0.0, vmax=1.0)
    paths = paths_for([1], 3, 2)
    total = 0.0
    for path in paths:
        p = np.exp(path_log_prob(log_probs, path))
        total += p
        print("".join("_a"[s] for s in path), f"{p:.3f}")
    print(f"{len(paths)} paths collapse to 'a', and their probabilities sum to {total:.3f}")
    loss = ctc_loss_brute_force(log_probs, [1])
    torch_loss = torch.nn.functional.ctc_loss(
        torch.tensor(log_probs)[:, None], torch.tensor([[1]]), [3], [1], reduction="sum").item()
    print(f"Brute-force loss {loss:.5f}; torch says {torch_loss:.5f}")

    # How fast the number of paths grows with the number of frames.
    frames = np.arange(2, 10)
    counts = np.array([len(paths_for([1, 2], int(t), 3)) for t in frames], dtype=float)
    print("Paths that spell 'ab' with 3 symbols:", dict(zip(frames.tolist(), counts.astype(int).tolist())))
    show_plot((frames, counts), (frames, 3.0 ** frames), label="Paths that spell 'ab' against all 3^T paths",
              xlabel="Frames T", ylabel="Number of paths", y_scale="log", labels=["spell 'ab'", "all paths"])
