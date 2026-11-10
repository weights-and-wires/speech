"""Sequence-level training: MMI, MWER on an n-best list, and word posteriors.

Fill in the nine functions, from `path_logscore` to `consensus_decode`, then press
Run to see the lesson's three worked examples. Press Check to test your code.

Conventions
-----------
- `loglik` has shape (T, S): `loglik[t, s]` is log p(x_t | state s), the
  acoustic log-likelihood of frame t under state s, in natural log.
- `log_start` has shape (S,) and `log_trans` has shape (S, S): together they
  are the toy language model. `log_start[s]` is log P(first state is s) and
  `log_trans[p, s]` is log P(next state is s | previous state p).
- A path is a sequence of T state indices.
"""

import numpy as np

from speechlab import show_plot


def logsumexp(x, axis=None):
    """Stable log of a sum of exponentials, given in full."""
    x = np.asarray(x, dtype=float)
    top = np.max(x, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    out = np.log(np.sum(np.exp(x - top), axis=axis, keepdims=True)) + top
    return out.squeeze() if axis is None else np.squeeze(out, axis=axis)


# Part 1: maximum mutual information --------------------------------------


def path_logscore(loglik, log_start, log_trans, path) -> float:
    """log of P(x | path) P(path): the emissions plus the language model score."""
    raise NotImplementedError("path_logscore")


def mmi_denominator(loglik, log_start, log_trans) -> float:
    """log of the sum over every state path of P(x | path) P(path), by the forward algorithm."""
    raise NotImplementedError("mmi_denominator")


def mmi_loss(loglik, log_start, log_trans, path) -> float:
    """The MMI loss: minus the log of numerator over denominator."""
    raise NotImplementedError("mmi_loss")


def state_posteriors(loglik, log_start, log_trans) -> np.ndarray:
    """gamma[t, s]: the probability that frame t is in state s, over all paths, shape (T, S)."""
    raise NotImplementedError("state_posteriors")


def mmi_grad(loglik, log_start, log_trans, path) -> np.ndarray:
    """The gradient of the MMI loss with respect to `loglik`: occupancy minus the reference, shape (T, S)."""
    raise NotImplementedError("mmi_grad")


# Part 2: minimum word error rate -----------------------------------------


def mwer_loss(nbest_logps, nbest_errors) -> float:
    """Expected word errors over the n-best list, with probabilities renormalized over the list."""
    raise NotImplementedError("mwer_loss")


def mwer_grad(nbest_logps, nbest_errors) -> np.ndarray:
    """The gradient of `mwer_loss` with respect to each hypothesis's log probability, shape (N,)."""
    raise NotImplementedError("mwer_grad")


# Part 3: word posteriors and consensus -----------------------------------

EPS = "<eps>"


def word_posteriors(nbest, logps) -> list[dict]:
    """For each slot of the aligned n-best list, a dict from word to posterior probability.

    `nbest` is a list of N hypotheses, each a list of the same length K whose
    entries are words, with "<eps>" for a deletion. `logps` holds each
    hypothesis's log score.
    """
    raise NotImplementedError("word_posteriors")


def consensus_decode(posteriors) -> tuple[list[str], list[float]]:
    """Pick the likeliest word in every slot, drop "<eps>", and return the words with their posteriors."""
    raise NotImplementedError("consensus_decode")


if __name__ == "__main__":
    # Part 1: two states, two frames, with the numbers worked in the lesson.
    loglik = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
    log_start = np.log(np.array([0.5, 0.5]))
    log_trans = np.log(np.array([[0.2, 0.8], [0.6, 0.4]]))
    ref = [0, 1]
    print(f"MMI loss for path {ref}: {mmi_loss(loglik, log_start, log_trans, ref):.4f}")
    print("gradient with respect to loglik:")
    print(np.round(mmi_grad(loglik, log_start, log_trans, ref), 4))

    # Part 2: three hypotheses whose raw probabilities cover 60% of the mass.
    logps = np.log([0.30, 0.18, 0.12])
    errors = [2, 0, 3]
    print(f"expected word errors: {mwer_loss(logps, errors):.3f}")
    print("gradient on the log probabilities:", np.round(mwer_grad(logps, errors), 3))

    # Part 3: three aligned hypotheses and their consensus.
    nbest = [["call", "mom", "back"], ["call", "tom", "home"], ["all", "mom", "home"]]
    slots = word_posteriors(nbest, np.log([0.4, 0.3, 0.3]))
    words, confidence = consensus_decode(slots)
    print("consensus:", " ".join(words), np.round(confidence, 2))
    boosts = np.linspace(-2.0, 4.0, 25)
    curve = [mwer_loss(np.log([0.30, 0.18, 0.12]) + np.array([0.0, b, 0.0]), errors) for b in boosts]
    show_plot(
        (boosts, np.array(curve)),
        label="Raising the log probability of the error-free hypothesis lowers the expected errors",
        xlabel="boost to hypothesis 2's log probability",
        ylabel="expected word errors",
    )
