"""Checks for the decoding-CTC exercise."""

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


@check("train_char_bigram counts, smooths, and normalizes: P(b | a) = 0.75 for 'ab' and 'abb'")
def test_bigram(sub):
    train = has(sub, "train_char_bigram")
    lm = np.asarray(train(["ab", "abb"], "_ab", smoothing=1.0), dtype=float)
    equal(lm.shape, (3, 3), label="the LM's shape")
    want = np.log(np.array([[0.75, 0.25], [0.25, 0.75], [1 / 3, 2 / 3]]))
    close(lm[:, 1:], want, atol=1e-9, label="log P(next | previous), rows start, a, b")
    if not np.all(np.isneginf(lm[:, 0])):
        fail(f"the LM must never predict the blank: column 0 should be -inf, got {lm[:, 0]}")
    lm = np.asarray(train(["the cat", "a hat"], "_ aceht", smoothing=0.5))
    close(np.exp(lm[:, 1:]).sum(axis=1), np.ones(7), atol=1e-9, label="each row's total probability")


@check("lm_score and hotword_score match the lesson's hand-worked values")
def test_scores(sub):
    score = has(sub, "lm_score")
    hot = has(sub, "hotword_score")
    lm = np.full((3, 3), -np.inf)
    lm[:, 1:] = np.log(np.array([[0.75, 0.25], [0.25, 0.75], [1 / 3, 2 / 3]]))
    # log P(ab) = log 0.75 + log 0.75.
    almost(score((1, 2), lm, 1.0, 0.0), 2 * math.log(0.75), 1e-9, label="lm_score('ab', alpha=1, beta=0)")
    almost(score((1, 2), lm, 0.5, 2.0), 0.5 * 2 * math.log(0.75) + 4.0, 1e-9, label="lm_score('ab', alpha=0.5, beta=2)")
    almost(score((), lm, 0.5, 2.0), 0.0, 1e-12, label="lm_score of the empty prefix")
    zoe = [(3, 4, 5)]
    for prefix, want in [((9, 3), 2.0), ((9, 3, 4), 4.0), ((9, 3, 4, 5), 6.0), ((9, 3, 4, 7), 0.0), ((3, 4, 5, 9, 3), 8.0)]:
        almost(hot(prefix, zoe, 2.0), want, 1e-12, label=f"hotword_score({prefix}) for the hotword (3, 4, 5)")


@check("Shallow fusion: with a wide beam, every score is log p_CTC + alpha log p_LM + beta |prefix|")
def test_fusion_exact(sub):
    beam = has(sub, "prefix_beam_search")
    score = has(sub, "lm_score")
    rng = np.random.default_rng(11)
    lm = log_softmax(rng.normal(size=(3, 3)))
    lm[:, 0] = -np.inf
    lm[:, 1:] = log_softmax(rng.normal(size=(3, 2)))
    log_probs = random_emissions(12, 5, 3)
    marginals = brute_force_marginals(log_probs)
    alpha, beta = 0.7, 0.4
    got = as_dict(beam(log_probs, 1000, prefix_score=lambda p: score(p, lm, alpha, beta)), "the fused beam")
    for prefix, value in marginals.items():
        want = value + alpha * sum(lm[a, b] for a, b in zip((0,) + prefix, prefix)) + beta * len(prefix)
        if prefix not in got:
            fail(f"the prefix {prefix} is missing from the fused beam")
        almost(got[prefix], want, 1e-8, label=f"the fused score of {prefix}")
    best_want = max(marginals, key=lambda p: marginals[p] + alpha * sum(lm[a, b] for a, b in zip((0,) + p, p)) + beta * len(p))
    best_got = beam(log_probs, 1000, prefix_score=lambda p: score(p, lm, alpha, beta))[0][0]
    equal(tuple(int(s) for s in best_got), best_want, label="the best fused prefix")


@check("On a sentence, the LM turns 'kat' into 'cat' and the hotword turns 'soe' into 'zoe'")
def test_whole(sub):
    beam = has(sub, "prefix_beam_search")
    train = has(sub, "train_char_bigram")
    score = has(sub, "lm_score")
    hot = has(sub, "hotword_score")
    vocab = "_ acehkostz"
    enc = lambda s: tuple(vocab.index(ch) for ch in s)
    dec = lambda p: "".join(vocab[int(i)] for i in p)
    rng = np.random.default_rng(3)
    rows = []

    def frame(probs):
        p = np.full(len(vocab), 0.01)
        for ch, v in probs.items():
            p[vocab.index(ch)] = v
        p *= np.exp(rng.normal(0, 0.03, len(vocab)))
        return np.log(p / p.sum())

    for ch in "a cat to zoe":
        rows.append(frame({"_": 0.9}))
        if ch == "c":
            rows.append(frame({"k": 0.5, "c": 0.4, "_": 0.05}))
        elif ch == "z":
            rows.append(frame({"s": 0.55, "z": 0.38, "_": 0.05}))
        else:
            rows.append(frame({ch: 0.9}))
    rows.append(frame({"_": 0.9}))
    log_probs = np.stack(rows)

    plain = dec(beam(log_probs, 6)[0][0])
    equal(plain, "a kat to soe", label="the acoustic-only beam result")
    lm = train(["a cat", "the cat", "a cat to eat", "tea to a cat", "the hat"], vocab, smoothing=0.5)
    fused = dec(beam(log_probs, 6, prefix_score=lambda p: score(p, lm, 0.8, 1.0))[0][0])
    equal(fused, "a cat to soe", label="the result with the bigram LM (alpha 0.8, beta 1.0)")
    boosted = dec(
        beam(log_probs, 6, prefix_score=lambda p: score(p, lm, 0.8, 1.0) + hot(p, [enc("zoe")], 2.0))[0][0]
    )
    equal(boosted, "a cat to zoe", label="the result with the LM and the hotword 'zoe' (bonus 2.0)")
