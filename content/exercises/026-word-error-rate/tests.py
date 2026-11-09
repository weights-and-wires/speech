"""Checks for the word error rate exercise."""

from functools import lru_cache

import numpy as np

from speechcheck import almost, check, fail, has


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


@check("levenshtein_align counts substitutions, deletions, and insertions on the worked pairs")
def test_align_counts(sub):
    align = has(sub, "levenshtein_align")
    cases = [("i like green tea", "i like the green tee", (1, 0, 1)),
             ("the cat sat on the mat", "a cat sat on mat", (1, 1, 0)),
             ("one two three", "", (0, 3, 0)),
             ("", "uh um", (0, 0, 2))]
    for ref, hyp, want in cases:
        s, d, i, _ = align(ref.split(), hyp.split())
        if (s, d, i) != want:
            fail(f"'{ref}' against '{hyp}': expected (S, D, I) = {want}, got {(s, d, i)}")


@check("The alignment spells out both the reference and the hypothesis, in order")
def test_align_spells(sub):
    align = has(sub, "levenshtein_align")
    ref, hyp = "i like green tea".split(), "i like the green tee".split()
    _, _, _, alignment = align(ref, hyp)
    rebuilt_ref = [r for op, r, h in alignment if op != "I"]
    rebuilt_hyp = [h for op, r, h in alignment if op != "D"]
    if rebuilt_ref != ref or rebuilt_hyp != hyp:
        fail(f"reading the alignment back gives '{' '.join(map(str, rebuilt_ref))}' and "
             f"'{' '.join(map(str, rebuilt_hyp))}'; it must spell out both sequences in order")
    ops = [op for op, _, _ in alignment]
    if ops != ["C", "C", "I", "C", "S"]:
        fail(f"expected the operations C C I C S, got {' '.join(ops)}; each entry is (op, ref_token, hyp_token)")


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


@check("wer is corpus-level: total errors over total reference words")
def test_wer_corpus(sub):
    wer = has(sub, "wer")
    refs = ["the cat sat on the mat", "hello"]
    hyps = ["a cat sat on mat", "hello there my friend"]
    # 2 errors over 6 words, then 3 insertions over 1 word: corpus WER 5 / 7, not the mean of 1/3 and 3.
    almost(wer(refs, hyps), 5 / 7, 1e-12, label="corpus WER, total errors over total reference words")
    almost(wer(["a b c d"], ["a b c d"]), 0.0, 1e-12, label="WER of a perfect transcript")


@check("wer can pass 100% when the recognizer inserts words")
def test_wer_over_100(sub):
    wer = has(sub, "wer")
    almost(wer(["yes"], ["no no no"]), 3.0, 1e-12, label="WER of 'no no no' against 'yes'")


@check("cer counts characters, spaces included: cer('kitten', 'sitting') is 0.5")
def test_cer(sub):
    cer = has(sub, "cer")
    almost(cer("kitten", "sitting"), 0.5, 1e-12, label="cer('kitten', 'sitting')")
    almost(cer("a b", "ab"), 1 / 3, 1e-12, label="cer('a b', 'ab'), where the space counts")
