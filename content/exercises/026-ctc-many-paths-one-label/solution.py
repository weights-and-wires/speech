"""CTC by brute force: collapse, enumerate, and sum. The worked solution."""

import itertools

import numpy as np
import torch

from speechlab import show_image, show_plot


def collapse(path, blank: int = 0) -> list[int]:
    """The CTC collapse: merge runs of the same symbol, then drop blanks."""
    out, previous = [], None
    for symbol in (int(s) for s in path):
        if symbol != previous and symbol != blank:
            out.append(symbol)
        previous = symbol
    return out


def path_log_prob(log_probs: np.ndarray, path) -> float:
    """The log probability of one path: the sum of its per-frame log probabilities."""
    path = np.asarray(path, dtype=int)
    return float(log_probs[np.arange(len(path)), path].sum())


def paths_for(target, n_frames: int, n_symbols: int, blank: int = 0) -> list[tuple[int, ...]]:
    """Every path of n_frames symbols that collapses to the target."""
    target = [int(s) for s in target]
    return [
        path
        for path in itertools.product(range(n_symbols), repeat=n_frames)
        if collapse(path, blank) == target
    ]


def ctc_loss_brute_force(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """-log p(target | x), summed over every path. inf when no path fits."""
    n_frames, n_symbols = log_probs.shape
    paths = paths_for(target, n_frames, n_symbols, blank)
    if not paths:
        return float("inf")
    total = np.logaddexp.reduce([path_log_prob(log_probs, p) for p in paths])
    return float(-total)


def min_frames(target) -> int:
    """The fewest frames that can spell the target: one per label, plus one per adjacent repeat."""
    target = [int(s) for s in target]
    repeats = sum(1 for a, b in zip(target[:-1], target[1:]) if a == b)
    return len(target) + repeats


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
