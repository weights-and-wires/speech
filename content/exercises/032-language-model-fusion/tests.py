"""Checks for the language model fusion and hotwords exercise."""

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


@check("lm_score is alpha times the bigram log probability, starting from the start context, plus beta per symbol")
def test_lm_score(sub):
    score = has(sub, "lm_score")
    lm = np.full((3, 3), -np.inf)
    lm[:, 1:] = np.log(np.array([[0.75, 0.25], [0.25, 0.75], [1 / 3, 2 / 3]]))
    # log P(ab) = log 0.75 + log 0.75.
    almost(score((1, 2), lm, 1.0, 0.0), 2 * math.log(0.75), 1e-9, label="lm_score('ab', alpha=1, beta=0)")
    almost(score((1, 2), lm, 0.5, 2.0), 0.5 * 2 * math.log(0.75) + 4.0, 1e-9, label="lm_score('ab', alpha=0.5, beta=2)")
    almost(score((2, 1), lm, 1.0, 0.0), math.log(0.25) + math.log(1 / 3), 1e-9, label="lm_score('ba', alpha=1, beta=0)")
    almost(score((), lm, 0.5, 2.0), 0.0, 1e-12, label="lm_score of the empty prefix")


@check("hotword_score rewards finished hotwords and partial matches at the end, and drops a broken match")
def test_hotword(sub):
    hot = has(sub, "hotword_score")
    zoe = [(3, 4, 5)]
    for prefix, want in [((9, 3), 2.0), ((9, 3, 4), 4.0), ((9, 3, 4, 5), 6.0), ((9, 3, 4, 7), 0.0), ((3, 4, 5, 9, 3), 8.0)]:
        almost(hot(prefix, zoe, 2.0), want, 1e-12, label=f"hotword_score({prefix}) for the hotword (3, 4, 5)")


@check("With alpha, beta, and the hotword bonus all zero, the fused beam equals the plain beam")
def test_zero_weights(sub):
    beam = has(sub, "prefix_beam_search")
    score, hot = has(sub, "lm_score"), has(sub, "hotword_score")
    rng = np.random.default_rng(21)
    lm = np.full((3, 3), -np.inf)
    lm[:, 1:] = log_softmax(rng.normal(size=(3, 2)))
    log_probs = random_emissions(22, 6, 3)
    plain = as_dict(beam(log_probs, 4), "the plain beam")
    fused = as_dict(beam(log_probs, 4, prefix_score=lambda p: score(p, lm, 0.0, 0.0) + hot(p, [(1, 2)], 0.0)),
                    "the fused beam with every weight at zero")
    equal(set(fused), set(plain), label="the prefixes of the two beams")
    for prefix, value in plain.items():
        almost(fused[prefix], value, 1e-9, label=f"score of {prefix}")


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


# Wrong implementation these checks catch: adding the internal LM instead of
# subtracting it ("log p_e2e + mu log p_ilm + lam log p_ext") gives -4.0 for joe
# and -5.1 for zoe where the right scores are -3.0 and -2.1, so joe wins and
# the target-domain name is lost. A second wrong version computes the internal
# LM with the blank still in the softmax, and a third keeps the encoder term.
@check("ILME subtracts the internal LM: score = log p_e2e - mu log p_ilm + lam log p_ext, and it flips joe to zoe")
def test_ilme_subtract(sub):
    ilme = has(sub, "ilme_subtract")
    e2e, ilm, ext = np.array([-2.0, -2.6]), np.array([-1.0, -3.0]), np.array([-3.0, -2.0])
    shallow = np.asarray(ilme(e2e, ilm, ext, 0.0, 0.5), dtype=float)
    close(shallow, np.array([-3.5, -3.6]), atol=1e-9, label="shallow fusion scores for [joe, zoe] (mu 0, lam 0.5)")
    got = np.asarray(ilme(e2e, ilm, ext, 0.5, 0.5), dtype=float)
    close(got, np.array([-3.0, -2.1]), atol=1e-9, label="ILME scores for [joe, zoe] (mu 0.5, lam 0.5)")
    equal(int(np.argmax(shallow)), 0, label="the shallow-fusion winner (0 is joe)")
    equal(int(np.argmax(got)), 1, label="the ILME winner (1 is zoe)")
    almost(float(ilme(-1.0, -2.0, -3.0, 1.0, 2.0)), -1.0 + 2.0 - 6.0, 1e-12, label="ilme_subtract on scalars")


@check("internal_lm_logprobs zeroes the encoder term, drops the blank, and renormalizes over the other tokens")
def test_internal_lm(sub):
    ilm_fn = has(sub, "internal_lm_logprobs")
    pred = np.array([[0.3, 1.0, 2.0, 0.0], [1.0, -1.0, 0.0, 1.0]])
    got = np.asarray(ilm_fn(pred), dtype=float)
    equal(got.shape, (2, 3), label="the internal LM's shape (U, C - 1)")
    want = np.stack([log_softmax(pred[:, 1:])[0], log_softmax(pred[:, 1:])[1]])
    close(got, want, atol=1e-9, label="log P_ILM(token | history) for tokens 1 to 3")
    close(np.exp(got).sum(axis=1), np.ones(2), atol=1e-9, label="each row's total probability over non-blank tokens")
    # The blank logit must not matter: it is removed before the softmax.
    loud = pred.copy()
    loud[:, 0] = 40.0
    close(np.asarray(ilm_fn(loud), dtype=float), want, atol=1e-9, label="the internal LM with the blank logit set to 40")
    # With the encoder term present, the joint's non-blank scores are different.
    enc = np.array([[0.0, 0.0, 1.5, -1.0]] * 2)
    full = log_softmax(enc + pred)[:, 1:]
    if np.allclose(got, full, atol=1e-3):
        fail("the internal LM equals the joint network's output with the encoder present; zero the encoder term")


@check("rescore_nbest reranks by am + lam * lm + length_bonus * len, best first")
def test_rescore(sub):
    rescore = has(sub, "rescore_nbest")
    hyps = [(1, 2), (1, 2, 3), (1, 3), (1, 2, 3, 3)]
    am = [-5.0, -4.0, -6.0, -3.5]
    lm = [-1.0, -1.0, -2.0, -8.0]
    out = rescore(hyps, am, lm, 0.5, 0.5)
    if not isinstance(out, list) or len(out) != 4:
        fail(f"rescore_nbest should return a list of 4 (hyp, score) pairs, got {out!r}")
    order = [tuple(int(s) for s in h) for h, _ in out]
    want_order = [(1, 2, 3), (1, 2), (1, 2, 3, 3), (1, 3)]
    if order != want_order:
        fail(f"the reranked order is wrong: expected {want_order}, got {order}")
    close(np.array([s for _, s in out]), np.array([-3.0, -4.5, -5.5, -6.0]), atol=1e-9, label="the reranked scores")
    acoustic = rescore(hyps, am, lm, 0.0, 0.0)
    equal(tuple(h for h, _ in acoustic)[0], (1, 2, 3, 3), label="the best hypothesis with lam 0 and no bonus (acoustic only)")
    no_bonus = rescore(hyps, am, lm, 0.5, 0.0)
    equal(tuple(h for h, _ in no_bonus)[-1], (1, 2, 3, 3), label="the worst hypothesis with lam 0.5 and no length bonus")
