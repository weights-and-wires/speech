"""Checks for the language model and word error rate exercise."""

import math
from collections import Counter
from functools import lru_cache

import numpy as np

from speechcheck import almost, check, equal, fail, has

WORKED = [["a", "b"], ["a", "c"], ["b", "c"]]


def random_corpus(rng, n, vocab=("x", "y", "z", "w", "v"), max_len=6):
    return [[str(rng.choice(vocab)) for _ in range(int(rng.integers(1, max_len + 1)))] for _ in range(n)]


def edit_distance(a, b):
    """Brute-force edit distance by memoized recursion, independent of any table."""
    a, b = tuple(a), tuple(b)

    @lru_cache(maxsize=None)
    def go(i, j):
        if i == len(a):
            return len(b) - j
        if j == len(b):
            return len(a) - i
        return min(go(i + 1, j) + 1, go(i, j + 1) + 1, go(i + 1, j + 1) + (a[i] != b[j]))

    return go(0, 0)


@check("bigram_counts pads every sentence with <s> and </s> and counts each pair")
def test_counts(sub):
    bigram_counts = has(sub, "bigram_counts")
    counts = bigram_counts(WORKED)
    want = {("<s>", "a"): 2, ("a", "b"): 1, ("b", "</s>"): 1, ("a", "c"): 1, ("c", "</s>"): 2,
            ("<s>", "b"): 1, ("b", "c"): 1}
    got = {k: v for k, v in dict(counts).items() if v}
    if got != want:
        fail(f"for the corpus 'a b', 'a c', 'b c', expected the counts {want}, got {got}")


@check("continuation_probs gives a = 1/7 and b = c = </s> = 2/7 on the worked corpus")
def test_continuation(sub):
    bigram_counts, continuation_probs = has(sub, "bigram_counts"), has(sub, "continuation_probs")
    p = continuation_probs(bigram_counts(WORKED))
    for word, want in (("a", 1 / 7), ("b", 2 / 7), ("c", 2 / 7), ("</s>", 2 / 7)):
        almost(p.get(word, 0.0), want, 1e-12, label=f"P_cont({word})")
    if p.get("<s>", 0.0) != 0.0:
        fail(f"<s> is never predicted, so P_cont(<s>) must be 0; got {p.get('<s>')}")


@check("kn_prob reproduces the worked Kneser-Ney probabilities after the history 'a'")
def test_kn_worked(sub):
    bigram_counts, kn_prob = has(sub, "bigram_counts"), has(sub, "kn_prob")
    counts = bigram_counts(WORKED)
    for word, want in (("b", 0.25 / 2 + 0.75 * 2 / 7), ("c", 0.25 / 2 + 0.75 * 2 / 7),
                       ("</s>", 0.75 * 2 / 7), ("a", 0.75 * 1 / 7)):
        almost(kn_prob(word, "a", counts, 0.75), want, 1e-12, label=f"P_KN({word} | a)")


@check("Every kn_prob distribution sums to 1, and d = 0 gives the maximum-likelihood bigram")
def test_kn_normalized(sub):
    bigram_counts, kn_prob = has(sub, "bigram_counts"), has(sub, "kn_prob")
    rng = np.random.default_rng(0)
    counts = bigram_counts(random_corpus(rng, 30))
    vocab = sorted({w for (_, w) in counts})
    histories = sorted({h for (h, _) in counts})
    for d in (0.5, 0.75, 0.9):
        for h in histories + ["never-seen"]:
            total = sum(kn_prob(w, h, counts, d) for w in vocab)
            if abs(total - 1.0) > 1e-9:
                fail(f"with d = {d}, the probabilities after '{h}' sum to {total:.10f} over the "
                     f"{len(vocab)} predicted tokens, not 1")
    c_x = sum(c for (h, _), c in counts.items() if h == "x")
    for w in vocab:
        almost(kn_prob(w, "x", counts, 0.0), counts.get(("x", w), 0) / c_x, 1e-12, label=f"P({w} | x) with d = 0")


@check("perplexity counts </s> but not <s>, and a uniform model's perplexity is its vocabulary size")
def test_perplexity(sub):
    perplexity = has(sub, "perplexity")
    uniform = lambda word, history: 1 / 10
    almost(perplexity([["a", "b", "c"], ["d"]], uniform), 10.0, 1e-9, label="uniform over 10 tokens")
    # P(a|<s>) = 0.5, P(b|a) = 0.25, P(</s>|b) = 1: three predicted tokens.
    table = {("<s>", "a"): 0.5, ("a", "b"): 0.25, ("b", "</s>"): 1.0}
    want = (0.5 * 0.25 * 1.0) ** (-1 / 3)
    almost(perplexity([["a", "b"]], lambda w, h: table[(h, w)]), want, 1e-9, label="perplexity of 'a b'")


@check("Kneser-Ney beats add-one smoothing on held-out sentences")
def test_kn_beats_add_one(sub):
    bigram_counts, kn_prob, perplexity = has(sub, "bigram_counts"), has(sub, "kn_prob"), has(sub, "perplexity")
    rng = np.random.default_rng(1)
    def sentence():
        subject = [str(rng.choice(["cat", "dog", "bird"]))]
        if rng.random() < 0.3:
            subject = ["big"] + subject
        return ["the"] + subject + [str(rng.choice(["sat", "ran", "slept"])), "on", "the",
                                    str(rng.choice(["mat", "rug", "sofa"]))]

    train = [sentence() for _ in range(40)]
    test = [sentence() for _ in range(40)]
    counts = bigram_counts(train)
    vocab = sorted({w for (_, w) in counts})
    history_counts = Counter()
    for (h, _), c in counts.items():
        history_counts[h] += c
    add_one = perplexity(test, lambda w, h: (counts.get((h, w), 0) + 1) / (history_counts[h] + len(vocab)))
    kn = perplexity(test, lambda w, h: kn_prob(w, h, counts, 0.75))
    if not kn < add_one:
        fail(f"held-out perplexity is {kn:.3f} with Kneser-Ney and {add_one:.3f} with add-one; "
             "Kneser-Ney should be lower")


@check("levenshtein_align counts substitutions, deletions, and insertions on the worked pairs")
def test_align_worked(sub):
    align = has(sub, "levenshtein_align")
    cases = [("i like green tea", "i like the green tee", (1, 0, 1)),
             ("the cat sat on the mat", "a cat sat on mat", (1, 1, 0)),
             ("one two three", "", (0, 3, 0)),
             ("", "uh um", (0, 0, 2))]
    for ref, hyp, want in cases:
        s, d, i, alignment = align(ref.split(), hyp.split())
        if (s, d, i) != want:
            fail(f"'{ref}' against '{hyp}': expected (S, D, I) = {want}, got {(s, d, i)}")
        rebuilt_ref = [r for op, r, h in alignment if op != "I"]
        rebuilt_hyp = [h for op, r, h in alignment if op != "D"]
        if rebuilt_ref != ref.split() or rebuilt_hyp != hyp.split():
            fail(f"reading the alignment for '{ref}' against '{hyp}' back gives '{' '.join(map(str, rebuilt_ref))}'"
                 f" and '{' '.join(map(str, rebuilt_hyp))}'; it must spell out both sequences in order")


@check("levenshtein_align matches brute-force edit distance on 200 random pairs")
def test_align_random(sub):
    align = has(sub, "levenshtein_align")
    rng = np.random.default_rng(2)
    for trial in range(200):
        ref = list(rng.choice(list("abcd"), size=int(rng.integers(0, 7))))
        hyp = list(rng.choice(list("abcd"), size=int(rng.integers(0, 7))))
        s, d, i, alignment = align(ref, hyp)
        want = edit_distance(ref, hyp)
        if s + d + i != want:
            fail(f"{''.join(ref)!r} against {''.join(hyp)!r}: S + D + I = {s + d + i}, "
                 f"but the edit distance is {want}")
        correct = sum(1 for op, _, _ in alignment if op == "C")
        if correct + s + d != len(ref) or correct + s + i != len(hyp):
            fail(f"{''.join(ref)!r} against {''.join(hyp)!r}: hits + S + D must equal the reference "
                 f"length and hits + S + I the hypothesis length")


@check("wer is corpus-level and can pass 100%, and cer('kitten', 'sitting') is 0.5")
def test_wer_cer(sub):
    wer, cer = has(sub, "wer"), has(sub, "cer")
    refs = ["the cat sat on the mat", "hello"]
    hyps = ["a cat sat on mat", "hello there my friend"]
    # 2 errors over 6 words, then 3 insertions over 1 word: corpus WER 5 / 7.
    almost(wer(refs, hyps), 5 / 7, 1e-12, label="corpus WER, total errors over total reference words")
    almost(wer(["yes"], ["no no no"]), 3.0, 1e-12, label="WER of 'no no no' against 'yes'")
    almost(cer("kitten", "sitting"), 0.5, 1e-12, label="cer('kitten', 'sitting')")
    almost(cer("a b", "ab"), 1 / 3, 1e-12, label="cer('a b', 'ab'), where the space counts")


@check("rescore trades acoustic score against the language model and the insertion penalty")
def test_rescore(sub):
    rescore = has(sub, "rescore")
    lm_table = {("<s>", "write"): 0.5, ("write", "to"): 0.9, ("write", "two"): 0.01,
                ("to", "mom"): 0.9, ("two", "mom"): 0.01, ("mom", "</s>"): 1.0, ("<s>", "right"): 0.01,
                ("right", "to"): 0.1, ("right", "two"): 0.1}
    lm = lambda w, h: lm_table.get((h, w), 1e-4)
    nbest = [("right two mom", 0.2), ("write to mom", 0.0), ("write two mom", 0.1)]
    equal(rescore(nbest, lm, 0.0, 0.0), "right two mom", label="with lm_weight 0, the acoustic best")
    equal(rescore(nbest, lm, 1.0, 0.0), "write to mom", label="with lm_weight 1")
    short_long = [("call mom", 0.0), ("call mom now", -0.5)]
    flat = lambda w, h: 0.5
    equal(rescore(short_long, flat, 1.0, 0.0), "call mom", label="no insertion bonus")
    equal(rescore(short_long, flat, 1.0, 2.0), "call mom now", label="an insertion bonus of 2 per word")
