"""Checks for the decoding-CTC exercise: greedy search and prefix beam search."""

import itertools
import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, has


def log_softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def collapse(path, blank=0):
    out, previous = [], None
    for s in path:
        if s != previous and s != blank:
            out.append(s)
        previous = s
    return tuple(out)


def brute_force_marginals(log_probs, blank=0):
    """log p(labeling) for every labeling with nonzero probability, by summing every path."""
    T, C = log_probs.shape
    totals = {}
    for path in itertools.product(range(C), repeat=T):
        lp = float(sum(log_probs[t, s] for t, s in enumerate(path)))
        key = collapse(path, blank)
        totals[key] = np.logaddexp(totals.get(key, -math.inf), lp)
    return totals


def random_emissions(seed, T, C, scale=1.5):
    rng = np.random.default_rng(seed)
    return log_softmax(rng.normal(0.0, scale, size=(T, C)))


def as_dict(results, label):
    if not isinstance(results, list) or not results:
        fail(f"{label} should be a non-empty list of (prefix, score) pairs, got {results!r}")
    out = {}
    for item in results:
        prefix, score = item
        out[tuple(int(s) for s in prefix)] = float(score)
    return out


@check("greedy_decode takes the best symbol per frame, merges repeats, and drops blanks")
def test_greedy(sub):
    greedy = has(sub, "greedy_decode")
    # Frame winners: a a _ a b b _ b, which collapses to a a b b.
    winners = [1, 1, 0, 1, 2, 2, 0, 2]
    log_probs = np.log(np.full((8, 3), 0.1))
    log_probs[np.arange(8), winners] = np.log(0.8)
    got = [int(s) for s in greedy(log_probs)]
    equal(got, [1, 1, 2, 2], label="greedy output for the path a a _ a b b _ b")
    log_probs = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
    got = [int(s) for s in greedy(log_probs)]
    equal(got, [], label="greedy output when blank wins every frame")


@check("prefix_beam_search finds 'a' where greedy finds nothing: p = 0.64 against 0.36")
def test_beam_counterexample(sub):
    beam = has(sub, "prefix_beam_search")
    log_probs = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
    results = as_dict(beam(log_probs, 4), "prefix_beam_search's result")
    best = beam(log_probs, 4)[0][0]
    equal(tuple(int(s) for s in best), (1,), label="the best prefix")
    if results.get((1, 1), -math.inf) > -math.inf:
        fail("'aa' needs a blank between its two a's, so it cannot fit in two frames; drop prefixes with probability 0")
    for prefix, want in [((1,), 0.64), ((), 0.36)]:
        if prefix not in results:
            fail(f"the prefix {prefix} is missing from the beam; got {sorted(results)}")
        almost(math.exp(results[prefix]), want, 1e-9, label=f"p({prefix})")


@check("With a wide beam, every prefix's score is its exact CTC probability (brute force, T = 5)")
def test_beam_exact(sub):
    beam = has(sub, "prefix_beam_search")
    for seed in range(3):
        log_probs = random_emissions(seed, 5, 3)
        want = brute_force_marginals(log_probs)
        got = as_dict(beam(log_probs, 1000), "prefix_beam_search's result")
        if set(got) != set(want):
            missing = sorted(set(want) - set(got))[:4]
            extra = sorted(set(got) - set(want))[:4]
            fail(f"seed {seed}: the beam's prefixes differ from brute force; missing {missing}, extra {extra}")
        for prefix, value in want.items():
            almost(got[prefix], value, 1e-8, label=f"seed {seed}, log p{prefix}")


@check("A narrow beam returns at most beam_width prefixes, best first, each a lower bound")
def test_beam_narrow(sub):
    beam = has(sub, "prefix_beam_search")
    log_probs = random_emissions(7, 6, 4, scale=1.0)
    want = brute_force_marginals(log_probs)
    results = beam(log_probs, 3)
    if len(results) > 3:
        fail(f"a beam of width 3 returned {len(results)} prefixes")
    scores = [float(s) for _, s in results]
    if any(a < b - 1e-12 for a, b in zip(scores, scores[1:])):
        fail(f"the results should be sorted best first, got scores {scores}")
    for prefix, score in results:
        exact = want[tuple(int(s) for s in prefix)]
        if score > exact + 1e-9:
            fail(
                f"the score {score:.6f} for {tuple(prefix)} exceeds its exact log probability "
                f"{exact:.6f}; pruning can only lose paths, never add them"
            )


@check("An extra prefix score is added to every prefix's score, and can change the winner")
def test_prefix_score(sub):
    beam = has(sub, "prefix_beam_search")
    log_probs = random_emissions(5, 5, 3)
    marginals = brute_force_marginals(log_probs)
    extra = lambda p: 0.5 * len(p) - 0.1 * sum(p)
    got = as_dict(beam(log_probs, 1000, prefix_score=extra), "the beam with a prefix score")
    for prefix, value in marginals.items():
        if prefix not in got:
            fail(f"the prefix {prefix} is missing from the beam")
        almost(got[prefix], value + extra(prefix), 1e-8, label=f"score of {prefix}: log p_CTC plus the extra term")
    bonus = lambda p: 5.0 if p == (2, 1) else 0.0
    best = beam(log_probs, 1000, prefix_score=bonus)[0][0]
    equal(tuple(int(s) for s in best), (2, 1), label="the best prefix when (2, 1) earns a bonus of 5")


@check("On peaky emissions the beam and greedy decoding agree")
def test_peaky(sub):
    beam, greedy = has(sub, "prefix_beam_search"), has(sub, "greedy_decode")
    winners = [0, 2, 2, 0, 1, 0, 1, 1, 0, 3, 0]
    probs = np.full((len(winners), 4), 0.01)
    probs[np.arange(len(winners)), winners] = 0.97
    log_probs = np.log(probs / probs.sum(axis=1, keepdims=True))
    want = [int(s) for s in greedy(log_probs)]
    equal(want, [2, 1, 1, 3], label="greedy output on near one-hot frames")
    got = tuple(int(s) for s in beam(log_probs, 5)[0][0])
    equal(got, tuple(want), label="the best prefix of the beam")
