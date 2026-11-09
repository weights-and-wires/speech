"""Fit a diagonal mixture, tie triphone states, and scale posteriors.

Fill in the six functions, then press Run. The log-likelihood curve of the
six-frame example is what Run prints once the functions work.
"""

import numpy as np

from speechlab import show_plot


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
    raise NotImplementedError("gmm_em")


def triphones(phones: list[str], boundary: str = "sil") -> list[str]:
    """One `left-phone+right` label per phone.

    The phone before the first, and the phone after the last, is `boundary`.
    """
    raise NotImplementedError("triphones")


def count_unseen(train: list[str], test: list[str]) -> int:
    """How many labels in `test` are absent from `train`, counting repeats."""
    raise NotImplementedError("count_unseen")


def best_question_split(
    X: np.ndarray,
    masks: list[np.ndarray],
    variance_floor: float = 1e-6,
) -> tuple[int, float]:
    """Index and likelihood gain of the best yes/no question.

    `X` has shape `(N, D)`. Each mask has shape `(N,)` and is true on the
    yes side. Ties go to the smaller index. An empty side scores 0.
    """
    raise NotImplementedError("best_question_split")


def scaled_likelihoods(posteriors: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Posterior divided by the prior.

    `posteriors` has shape `(T, N)` and `priors` has shape `(N,)`.
    Returns shape `(T, N)`.
    """
    raise NotImplementedError("scaled_likelihoods")


def viterbi(
    log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray
) -> tuple[np.ndarray, float]:
    """Best state path and its log probability.

    Same layout as the HMM chapter: `log_pi` is `(N,)`, `log_A[i, j]` is
    the log probability of moving from `i` to `j`, and `log_B` is `(T, N)`.
    """
    raise NotImplementedError("viterbi")


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
