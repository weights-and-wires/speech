"""Fit a diagonal mixture, tie triphone states, and scale posteriors.

The worked solution. Press Run to print the six-frame likelihood curve.
"""

import numpy as np
from scipy.special import logsumexp

from speechlab import show_plot


def _log_gauss(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Diagonal Gaussian log density, shape (N, K)."""
    n_dims = X.shape[1]
    diff2 = (X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
    return -0.5 * (
        n_dims * np.log(2 * np.pi)
        + np.log(variances).sum(axis=1)[None, :]
        + diff2.sum(axis=2)
    )


def gmm_em(
    X: np.ndarray,
    n_components: int,
    n_steps: int,
    variance_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a diagonal Gaussian mixture with EM.

    `X` has shape `(N, D)`. Returns weights `(K,)`, means `(K, D)`,
    variances `(K, D)`, and log-likelihoods `(n_steps + 1,)`. Entry 0 is
    the likelihood of the initial parameters, in nats.
    """
    X = np.asarray(X, dtype=float)
    n_rows = X.shape[0]
    order = np.argsort(X[:, 0], kind="mergesort")
    if n_components == 1:
        ranks = np.array([n_rows // 2])
    else:
        ranks = np.round(np.linspace(0, n_rows - 1, n_components)).astype(int)
    means = X[order[ranks]].copy()
    shared = np.maximum(X.var(axis=0), variance_floor)
    variances = np.repeat(shared[None, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)
    log_likelihoods = []
    for _ in range(n_steps):
        log_joint = np.log(weights)[None, :] + _log_gauss(X, means, variances)
        log_norm = logsumexp(log_joint, axis=1)
        log_likelihoods.append(float(log_norm.sum()))
        responsibilities = np.exp(log_joint - log_norm[:, None])
        counts = responsibilities.sum(axis=0)
        weights = counts / n_rows
        means = (responsibilities.T @ X) / counts[:, None]
        diff = X[:, None, :] - means[None, :, :]
        variances = (responsibilities[:, :, None] * diff**2).sum(axis=0) / counts[:, None]
        variances = np.maximum(variances, variance_floor)
    log_joint = np.log(weights)[None, :] + _log_gauss(X, means, variances)
    log_likelihoods.append(float(logsumexp(log_joint, axis=1).sum()))
    return weights, means, variances, np.asarray(log_likelihoods, dtype=float)


def triphones(phones: list[str], boundary: str = "sil") -> list[str]:
    """One `left-phone+right` label per phone.

    The phone before the first, and the phone after the last, is `boundary`.
    """
    sequence = [boundary, *list(phones), boundary]
    return [
        f"{sequence[index]}-{sequence[index + 1]}+{sequence[index + 2]}"
        for index in range(len(phones))
    ]


def count_unseen(train: list[str], test: list[str]) -> int:
    """How many labels in `test` are absent from `train`, counting repeats."""
    seen = set(train)
    return sum(1 for label in test if label not in seen)


def _cluster_loglik(X: np.ndarray, variance_floor: float) -> float:
    """Sum of diagonal-Gaussian log densities at the maximum-likelihood fit."""
    if X.shape[0] == 0:
        return 0.0
    n_rows, n_dims = X.shape
    variance = np.maximum(X.var(axis=0), variance_floor)
    squared = ((X - X.mean(axis=0)) ** 2).sum(axis=0)
    return float(
        -0.5
        * (
            n_rows * (n_dims * np.log(2 * np.pi) + np.log(variance).sum())
            + (squared / variance).sum()
        )
    )


def best_question_split(
    X: np.ndarray,
    masks: list[np.ndarray],
    variance_floor: float = 1e-6,
) -> tuple[int, float]:
    """Index and likelihood gain of the best yes/no question.

    `X` has shape `(N, D)`. Each mask has shape `(N,)` and is true on the
    yes side. Ties go to the smaller index. An empty side scores 0.
    """
    X = np.asarray(X, dtype=float)
    parent = _cluster_loglik(X, variance_floor)
    best_index, best_gain = 0, -np.inf
    for index, mask in enumerate(masks):
        yes = np.asarray(mask, dtype=bool)
        gain = (
            _cluster_loglik(X[yes], variance_floor)
            + _cluster_loglik(X[~yes], variance_floor)
            - parent
        )
        if gain > best_gain:
            best_index, best_gain = index, gain
    return best_index, float(best_gain)


def scaled_likelihoods(posteriors: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Posterior divided by the prior.

    `posteriors` has shape `(T, N)` and `priors` has shape `(N,)`.
    Returns shape `(T, N)`.
    """
    return np.asarray(posteriors, dtype=float) / np.asarray(priors, dtype=float)


def viterbi(
    log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
) -> tuple[np.ndarray, float]:
    """Best state path and its log probability.

    Same layout as the HMM chapter: `log_pi` is `(N,)`, `log_A[i, j]` is
    the log probability of moving from `i` to `j`, and `log_B` is `(T, N)`.
    """
    log_pi = np.asarray(log_pi, dtype=float)
    log_A = np.asarray(log_A, dtype=float)
    log_B = np.asarray(log_B, dtype=float)
    n_frames, n_states = log_B.shape
    delta = log_pi + log_B[0]
    psi = np.zeros((n_frames, n_states), dtype=int)
    for time in range(1, n_frames):
        scores = delta[:, None] + log_A
        psi[time] = np.argmax(scores, axis=0)
        delta = scores[psi[time], np.arange(n_states)] + log_B[time]
    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(delta))
    for time in range(n_frames - 1, 0, -1):
        path[time - 1] = psi[time, path[time]]
    return path, float(delta[path[-1]])


if __name__ == "__main__":
    frames = np.array([[-1.0], [0.0], [1.0], [5.0], [6.0], [7.0]])
    weights, means, variances, log_likelihoods = gmm_em(frames, 2, 8)
    print("log-likelihood (nats):", np.array2string(log_likelihoods, precision=3))
    print("means:", means.ravel())
    print("triphones:", triphones(["k", "ae", "t"]))
    show_plot(
        log_likelihoods,
        label="EM log-likelihood of the six frames",
        xlabel="EM step",
        ylabel="log-likelihood (nats)",
    )
