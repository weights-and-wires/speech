"""Checks for the hybrid acoustic model exercise.

Two plausible wrong implementations fail checks here:

- `gmm_em_hard_assign` replaces the soft responsibilities with a one-hot
  argmax before the M-step. On `overlap_frames()` its log-likelihood falls
  from about -59.402 to about -59.467. `test_log_likelihood_is_monotone`
  rejects that drop.
- `scaled_likelihoods_times_prior` multiplies the posterior by the prior.
  For posteriors [0.6, 0.4] and priors [0.8, 0.2] it returns [0.48, 0.08].
  `test_scaled_likelihoods_divide_by_the_prior` expects [0.75, 2.0].
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from speechcheck import almost, check, close, equal, fail, has


SIX = np.array([[-1.0], [0.0], [1.0], [5.0], [6.0], [7.0]])


def overlap_frames():
    """The 48th draw, index 47, from this seeded generator.

    `gmm_em_hard_assign` lowers the log-likelihood on this array.
    """
    rng = np.random.default_rng(0)
    frames = None
    for _ in range(48):
        frames = np.concatenate(
            [
                rng.normal(0.0, 1.0, size=(10, 2)),
                rng.normal(1.2, 1.0, size=(10, 2)),
            ]
        )
    return frames


def reference_one_step(X, floor=1e-3):
    """One EM step from the documented start, using scipy's normal log density."""
    n_rows = len(X)
    order = np.argsort(X[:, 0], kind="mergesort")
    ranks = np.round(np.linspace(0, n_rows - 1, 2)).astype(int)
    means = X[order[ranks]].copy()
    variance = np.maximum(X.var(axis=0), floor)
    variances = np.repeat(variance[None, :], 2, axis=0)
    weights = np.full(2, 0.5)

    def log_joint(means_now, variances_now, weights_now):
        columns = [
            norm.logpdf(X[:, 0], loc=means_now[k, 0], scale=np.sqrt(variances_now[k, 0]))
            for k in range(2)
        ]
        return np.log(weights_now) + np.column_stack(columns)

    joint = log_joint(means, variances, weights)
    log_norm = logsumexp(joint, axis=1)
    responsibilities = np.exp(joint - log_norm[:, None])
    counts = responsibilities.sum(axis=0)
    weights = counts / n_rows
    means = (responsibilities.T @ X) / counts[:, None]
    diff = X[:, None, :] - means[None, :, :]
    variances = np.maximum(
        (responsibilities[:, :, None] * diff**2).sum(axis=0) / counts[:, None],
        floor,
    )
    updated = log_joint(means, variances, weights)
    return float(log_norm.sum()), float(logsumexp(updated, axis=1).sum()), means, variances, weights


def monotone(log_likelihoods, label):
    drops = np.where(np.diff(log_likelihoods) < -1e-6)[0]
    if len(drops):
        index = int(drops[0])
        fail(
            f"{label}: log-likelihood fell at step {index}, "
            f"got {log_likelihoods[index + 1]:.6f} after {log_likelihoods[index]:.6f}, "
            f"expected a value at least {log_likelihoods[index]:.6f}"
        )


# The Viterbi from lesson 021. log_A[i, j] = log P(next = j | current = i).
def hmm_oracle(log_pi, log_A, log_B):
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


def noisy_posteriors():
    """Seeded frame posteriors. The rare state is likely only after scaling."""
    rng = np.random.default_rng(0)
    true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    posteriors = np.zeros((len(true), 2))
    for time, state in enumerate(true):
        rare = 0.08 if state == 0 else 0.42
        rare = float(np.clip(rare + rng.normal(0, 0.01), 0.05, 0.49))
        posteriors[time] = [1.0 - rare, rare]
    return posteriors


@check("gmm_em matches one EM step on the six frames")
def test_one_step(sub):
    gmm_em = has(sub, "gmm_em")
    got = gmm_em(SIX, 2, 1)
    if not isinstance(got, tuple) or len(got) != 4:
        fail(f"gmm_em returned {type(got).__name__}, expected a tuple of 4 arrays")
    weights, means, variances, log_likelihoods = got
    want_ll0, want_ll1, want_means, want_variances, want_weights = reference_one_step(SIX)
    close(np.asarray(log_likelihoods), [want_ll0, want_ll1], atol=1e-6, label="log-likelihoods")
    close(np.asarray(means), want_means, atol=1e-6, label="means after one step")
    close(np.asarray(variances), want_variances, atol=1e-6, label="variances after one step")
    close(np.asarray(weights), want_weights, atol=1e-6, label="weights after one step")


@check("gmm_em log-likelihood does not fall")
def test_log_likelihood_is_monotone(sub):
    gmm_em = has(sub, "gmm_em")
    _, _, _, six = gmm_em(SIX, 2, 8)
    equal(np.asarray(six).shape, (9,), label="six-frame curve length")
    monotone(np.asarray(six, dtype=float), "six frames")
    if six[-1] < six[0] + 4.0:
        fail(
            f"six frames: log-likelihood ended at {six[-1]:.4f}, "
            f"expected a rise of at least 4 nats from {six[0]:.4f}"
        )
    _, _, _, overlap = gmm_em(overlap_frames(), 2, 6)
    monotone(np.asarray(overlap, dtype=float), "overlap frames")


@check("gmm_em recovers a known two-component mixture")
def test_recovers_mixture(sub):
    gmm_em = has(sub, "gmm_em")
    rng = np.random.default_rng(1)
    frames = np.concatenate(
        [
            rng.normal(-3.0, 0.4, size=(120, 1)),
            rng.normal(3.0, 0.5, size=(120, 1)),
        ]
    )
    weights, means, variances, log_likelihoods = gmm_em(frames, 2, 20)
    monotone(np.asarray(log_likelihoods, dtype=float), "recovery")
    order = np.argsort(np.asarray(means)[:, 0])
    almost(means[order[0], 0], -3.0, 0.15, label="low component mean")
    almost(means[order[1], 0], 3.0, 0.15, label="high component mean")
    almost(variances[order[0], 0], 0.16, 0.08, label="low component variance")
    almost(variances[order[1], 0], 0.25, 0.08, label="high component variance")
    almost(weights[order[0]], 0.5, 0.1, label="low component weight")
    almost(weights[order[1]], 0.5, 0.1, label="high component weight")


@check("triphones labels cat with silence at both ends")
def test_triphones(sub):
    triphones = has(sub, "triphones")
    equal(
        triphones(["k", "ae", "t"]),
        ["sil-k+ae", "k-ae+t", "ae-t+sil"],
        label="cat",
    )
    equal(triphones(["a"]), ["sil-a+sil"], label="one phone")
    equal(
        triphones(["k", "ae"], boundary="sp"),
        ["sp-k+ae", "k-ae+sp"],
        label="boundary sp",
    )


@check("count_unseen counts each missing occurrence")
def test_count_unseen(sub):
    count_unseen = has(sub, "count_unseen")
    train = ["sil-k+ae", "k-ae+t"]
    test = ["sil-k+ae", "ae-t+sil", "ae-t+sil", "k-ae+t"]
    equal(count_unseen(train, test), 2, label="two copies of ae-t+sil")
    equal(count_unseen(train, train), 0, label="nothing unseen")


@check("best_question_split picks the vowel question")
def test_best_question(sub):
    best_question_split = has(sub, "best_question_split")
    frames = np.array([0.0, 0.2, -0.2, 0.1, 3.8, 4.0, 4.2, 3.9])[:, None]
    masks = [
        np.array([True, True, True, True, False, False, False, False]),
        np.array([True, True, False, False, True, True, False, False]),
        np.ones(8, dtype=bool),
    ]
    index, gain = best_question_split(frames, masks)
    equal(int(index), 0, label="question index")
    if not np.isfinite(gain):
        fail(f"gain is {gain}, expected 20.756560")
    close(gain, 20.756560209454037, atol=1e-6, label="vowel-question gain")


@check("scaled_likelihoods divides by the prior")
def test_scaled_likelihoods_divide_by_the_prior(sub):
    scaled_likelihoods = has(sub, "scaled_likelihoods")
    posteriors = np.array([[0.6, 0.4], [0.9, 0.1]])
    priors = np.array([0.8, 0.2])
    close(
        scaled_likelihoods(posteriors, priors),
        np.array([[0.75, 2.0], [1.125, 0.5]]),
        atol=1e-12,
        label="scaled likelihoods",
    )


def _decode(sub, posteriors, priors, log_pi, log_A, label):
    scaled_likelihoods = has(sub, "scaled_likelihoods")
    viterbi = has(sub, "viterbi")
    scaled = scaled_likelihoods(posteriors, priors)
    path, score = viterbi(log_pi, log_A, np.log(scaled))
    want_path, want_score = hmm_oracle(log_pi, log_A, np.log(posteriors / priors))
    equal(np.asarray(path), want_path, label=f"{label} scaled path")
    close(score, want_score, atol=1e-6, label=f"{label} scaled log probability")
    posterior_path, _ = hmm_oracle(log_pi, log_A, np.log(posteriors))
    if np.array_equal(np.asarray(path), posterior_path):
        fail(
            f"{label}: posterior-only path {posterior_path.tolist()} equals the "
            f"scaled path {np.asarray(path).tolist()}, expected the scaled path "
            f"{want_path.tolist()} to differ"
        )


@check("viterbi on scaled likelihoods matches the HMM oracle on four frames")
def test_viterbi_four_frames(sub):
    posteriors = np.array(
        [[0.90, 0.10], [0.62, 0.38], [0.62, 0.38], [0.90, 0.10]]
    )
    priors = np.array([0.8, 0.2])
    transitions = np.array([[0.7, 0.3], [0.3, 0.7]])
    _decode(
        sub,
        posteriors,
        priors,
        np.log(np.array([0.6, 0.4])),
        np.log(transitions),
        "four frames",
    )
    # The hand path is part of the lesson, so a transposed log_A fails here
    # even if it is consistent with itself.
    viterbi = has(sub, "viterbi")
    scaled = has(sub, "scaled_likelihoods")(posteriors, priors)
    path, score = viterbi(np.log([0.6, 0.4]), np.log(transitions), np.log(scaled))
    equal(np.asarray(path), np.array([0, 1, 1, 1]), label="hand path")
    close(score, -1.7198046885281641, atol=1e-6, label="hand path log probability")


@check("viterbi on a noisy draw matches the HMM oracle, not the posteriors")
def test_viterbi_noisy(sub):
    priors = np.array([0.85, 0.15])
    transitions = np.array([[0.8, 0.2], [0.4, 0.6]])
    _decode(
        sub,
        noisy_posteriors(),
        priors,
        np.log(np.array([0.9, 0.1])),
        np.log(transitions),
        "noisy draw",
    )
