"""Checks for the sequence-level training exercise."""

import itertools
import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, finite_difference, has, shape

LOGLIK = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
START = np.log(np.array([0.5, 0.5]))
TRANS = np.log(np.array([[0.2, 0.8], [0.6, 0.4]]))


def toy(seed, T=3, S=3):
    """A seeded 3-state, 3-frame problem: 27 paths, small enough to enumerate."""
    rng = np.random.default_rng(seed)
    loglik = np.log(rng.dirichlet(np.ones(S), size=T))
    start = np.log(rng.dirichlet(np.ones(S)))
    trans = np.log(rng.dirichlet(np.ones(S), size=S))
    return loglik, start, trans


def enumerate_scores(loglik, start, trans):
    """The oracle: every path's log score, in a plain loop that shares no code with the solution."""
    T, S = loglik.shape
    scores = {}
    for path in itertools.product(range(S), repeat=T):
        lp = start[path[0]] + loglik[0, path[0]]
        for t in range(1, T):
            lp += trans[path[t - 1], path[t]] + loglik[t, path[t]]
        scores[path] = float(lp)
    return scores


def lse(values):
    values = np.asarray(list(values), dtype=float)
    top = values.max()
    return float(top + np.log(np.exp(values - top).sum()))


@check("path_logscore adds the language model and the emissions: log 0.168 for the path (a, b)")
def test_path_logscore(sub):
    score = has(sub, "path_logscore")
    # 0.6 * 0.5 for the first frame, then 0.8 * 0.7 for the second.
    almost(score(LOGLIK, START, TRANS, [0, 1]), math.log(0.6 * 0.5 * 0.8 * 0.7), 1e-9, label="path (0, 1)")
    almost(score(LOGLIK, START, TRANS, [1, 0]), math.log(0.4 * 0.5 * 0.6 * 0.3), 1e-9, label="path (1, 0)")
    loglik, start, trans = toy(1)
    oracle = enumerate_scores(loglik, start, trans)
    for path in [(0, 1, 2), (2, 2, 0)]:
        almost(score(loglik, start, trans, list(path)), oracle[path], 1e-9, label=f"path {path}")


@check("mmi_denominator equals the log of the sum over all 27 paths, enumerated by brute force")
def test_denominator(sub):
    den = has(sub, "mmi_denominator")
    almost(den(LOGLIK, START, TRANS), math.log(0.278), 1e-9, label="the lesson's two-frame example")
    for seed in (2, 3, 4):
        loglik, start, trans = toy(seed)
        want = lse(enumerate_scores(loglik, start, trans).values())
        almost(den(loglik, start, trans), want, 1e-9, label=f"log denominator, seed {seed}")


@check("mmi_loss is the log denominator minus the log numerator: 0.5037 on the lesson's example")
def test_mmi_loss(sub):
    loss = has(sub, "mmi_loss")
    almost(loss(LOGLIK, START, TRANS, [0, 1]), -math.log(0.168 / 0.278), 1e-9, label="loss for the path (0, 1)")
    loglik, start, trans = toy(5)
    scores = enumerate_scores(loglik, start, trans)
    den = lse(scores.values())
    for path in [(0, 0, 0), (1, 2, 1)]:
        almost(loss(loglik, start, trans, list(path)), den - scores[path], 1e-9, label=f"loss for {path}")
    if any(loss(loglik, start, trans, list(p)) < -1e-12 for p in scores):
        fail("the loss is a negative log probability of one path among all paths, so it must not be negative")


@check("The denominator includes the language model: a flat LM gives a loss of 0.8675 here, the real one 0.5037")
def test_lm_matters(sub):
    loss = has(sub, "mmi_loss")
    # Plausible wrong implementation: a denominator that sums the acoustics over every state sequence
    # and leaves out the language model. Its loss for (0, 1) is -log(0.42) = 0.8675, not 0.5037.
    got = loss(LOGLIK, START, TRANS, [0, 1])
    flat = np.log(np.full((2, 2), 0.5))
    flat_start = np.log(np.full(2, 0.5))
    flat_loss = loss(LOGLIK, flat_start, flat, [0, 1])
    almost(flat_loss, -math.log(0.42), 1e-9, label="loss with a flat language model, which is -log(0.6 * 0.7)")
    if abs(got - flat_loss) < 0.1:
        fail(f"the LM must change the loss: got {got:.4f} with the LM and {flat_loss:.4f} with a flat one")
    almost(got, -math.log(0.168 / 0.278), 1e-9, label="loss with the LM")


@check("state_posteriors match the occupancy of every enumerated path, and each frame sums to 1")
def test_posteriors(sub):
    post = has(sub, "state_posteriors")
    loglik, start, trans = toy(6)
    scores = enumerate_scores(loglik, start, trans)
    den = lse(scores.values())
    want = np.zeros((3, 3))
    for path, lp in scores.items():
        for t, s in enumerate(path):
            want[t, s] += math.exp(lp - den)
    got = np.asarray(post(loglik, start, trans), dtype=float)
    shape(got, (3, 3), label="state posteriors")
    close(got.sum(axis=1), np.ones(3), atol=1e-9, label="each frame's total probability")
    close(got, want, atol=1e-9, label="state posteriors against enumeration")


@check("mmi_grad equals occupancy minus the reference, and matches finite differences")
def test_mmi_grad(sub):
    loss, grad = has(sub, "mmi_loss"), has(sub, "mmi_grad")
    for seed, path in [(7, [0, 1, 2]), (8, [2, 2, 1])]:
        loglik, start, trans = toy(seed)
        got = np.asarray(grad(loglik, start, trans, path), dtype=float)
        want = finite_difference(lambda x: loss(x, start, trans, path), loglik)
        shape(got, (3, 3), label="gradient")
        close(got, want, atol=1e-6, label=f"mmi_grad for path {path} against finite differences")
    got = np.asarray(grad(LOGLIK, START, TRANS, [0, 1]), dtype=float)
    # gamma = [[0.6691, 0.3309], [0.1942, 0.8058]]; subtract the reference one-hot rows.
    close(got, np.array([[-0.3309, 0.3309], [0.1942, -0.1942]]), atol=1e-4, label="the lesson's example")


@check("mwer_loss is the expected word errors with probabilities renormalized over the list: 1.6")
def test_mwer_loss(sub):
    loss = has(sub, "mwer_loss")
    logps = np.log([0.30, 0.18, 0.12])  # raw probabilities sum to 0.6, not 1
    almost(loss(logps, [2, 0, 3]), 0.5 * 2 + 0.3 * 0 + 0.2 * 3, 1e-9, label="expected word errors")
    # Plausible wrong implementation: using exp(logps) without renormalizing gives 0.3*2 + 0.12*3 = 0.96.
    almost(loss(logps + 5.0, [2, 0, 3]), 1.6, 1e-9, label="the same list with every log probability shifted by 5")
    almost(loss(np.log([0.25, 0.25]), [4, 4]), 4.0, 1e-9, label="equal errors give that error, whatever the probabilities")


@check("mwer_grad is p_i (e_i - mean error): [0.2, -0.48, 0.28] here, sums to zero, and matches finite differences")
def test_mwer_grad(sub):
    loss, grad = has(sub, "mwer_loss"), has(sub, "mwer_grad")
    logps = np.log([0.30, 0.18, 0.12])
    errors = np.array([2.0, 0.0, 3.0])
    got = np.asarray(grad(logps, errors), dtype=float)
    shape(got, (3,), label="gradient")
    # Plausible wrong implementation: p_i * e_i, without subtracting the mean error. It gives [1.0, 0.0, 0.6].
    close(got, np.array([0.2, -0.48, 0.28]), atol=1e-9, label="closed-form gradient")
    almost(float(got.sum()), 0.0, 1e-9, label="the gradient's sum, which is zero when probabilities are renormalized")
    close(got, finite_difference(lambda x: loss(x, errors), logps), atol=1e-6, label="gradient against finite differences")
    rng = np.random.default_rng(9)
    logps, errors = rng.normal(size=6), rng.integers(0, 6, size=6).astype(float)
    close(grad(logps, errors), finite_difference(lambda x: loss(x, errors), logps), atol=1e-6, label="random list of 6")


@check("word_posteriors weighs each hypothesis by its renormalized probability, against an enumerated oracle")
def test_word_posteriors(sub):
    post = has(sub, "word_posteriors")
    nbest = [["call", "mom", "back"], ["call", "tom", "home"], ["all", "mom", "home"]]
    slots = post(nbest, np.log([0.4, 0.3, 0.3]))
    if len(slots) != 3:
        fail(f"expected one dict per slot, 3 in all; got {len(slots)}")
    want = [{"call": 0.7, "all": 0.3}, {"mom": 0.7, "tom": 0.3}, {"back": 0.4, "home": 0.6}]
    for k, (got, expect) in enumerate(zip(slots, want)):
        equal(set(got), set(expect), label=f"words in slot {k}")
        for word, value in expect.items():
            almost(got[word], value, 1e-9, label=f"posterior of '{word}' in slot {k}")
    # Unnormalized scores, a deletion, and unequal weights. Counting hypotheses instead of weighing them fails here.
    nbest = [["a", "b"], ["a", "<eps>"], ["c", "<eps>"], ["a", "b"]]
    logps = np.log([0.2, 0.1, 0.05, 0.05]) + 3.0
    slots = post(nbest, logps)
    total = 0.4
    almost(slots[0]["a"], 0.35 / total, 1e-9, label="posterior of 'a' in slot 0")
    almost(slots[0]["c"], 0.05 / total, 1e-9, label="posterior of 'c' in slot 0")
    almost(slots[1]["<eps>"], 0.15 / total, 1e-9, label="posterior of a deletion in slot 1")
    for k, slot in enumerate(slots):
        almost(sum(slot.values()), 1.0, 1e-9, label=f"total probability of slot {k}")


@check("consensus_decode beats the 1-best on expected word errors, and drops a slot whose best entry is a deletion")
def test_consensus(sub):
    post, decode = has(sub, "word_posteriors"), has(sub, "consensus_decode")
    nbest = [["call", "mom", "back"], ["call", "tom", "home"], ["all", "mom", "home"]]
    probs = np.array([0.4, 0.3, 0.3])
    slots = post(nbest, np.log(probs))
    words, conf = decode(slots)
    equal(list(words), ["call", "mom", "home"], label="consensus words (not one of the three hypotheses)")
    close(np.asarray(conf, dtype=float), np.array([0.7, 0.7, 0.6]), atol=1e-9, label="confidence per word")
    # Expected word errors, position by position, since the slots are aligned.
    expected = lambda hyp: sum(p * sum(a != b for a, b in zip(hyp, h)) for p, h in zip(probs, nbest))
    almost(expected(words), 1.0, 1e-9, label="expected errors of the consensus")
    almost(expected(nbest[0]), 1.2, 1e-9, label="expected errors of the 1-best")
    words, conf = decode(post([["go", "back"], ["go", "<eps>"], ["go", "<eps>"]], np.log([0.2, 0.4, 0.4])))
    equal(list(words), ["go"], label="the deletion wins slot 1, so the output has one word")
    equal(len(conf), 1, label="number of confidences")
