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


@check("paired_bootstrap on degenerate cases: one utterance, identical systems, B better everywhere")
def test_bootstrap_degenerate(sub):
    boot = has(sub, "paired_bootstrap")
    # One utterance has one possible resample: A makes 3 errors, B makes 1, over 4 words, so A - B = 0.5.
    got = boot([3], [1], [4], 50, 0)
    for value, want, label in zip(got, (1.0, 0.5, 0.5), ("p_b_better", "ci_low", "ci_high")):
        almost(value, want, 1e-12, label=f"one utterance, {label}")
    # Identical systems never differ, and a tie is not "B is better".
    got = boot([2, 0, 5], [2, 0, 5], [6, 3, 9], 200, 0)
    for value, label in zip(got, ("p_b_better", "ci_low", "ci_high")):
        almost(value, 0.0, 1e-12, label=f"identical systems, {label}")
    # B makes one error fewer than A on every utterance, so every resample favors B.
    lengths = [4, 9, 2, 7, 12, 5]
    errors_a = [2, 3, 1, 2, 4, 2]
    p, low, high = boot(errors_a, [e - 1 for e in errors_a], lengths, 300, 0)
    almost(p, 1.0, 1e-12, label="p_b_better when B wins on every utterance")
    if low <= 0:
        fail(f"B wins on every utterance, so the whole interval must be above 0; got ci_low = {low:.4f}")


@check("paired_bootstrap reports WER(A) - WER(B): swapping the systems flips the interval")
def test_bootstrap_sign(sub):
    boot = has(sub, "paired_bootstrap")
    errors_a, errors_b, lengths = [2, 1, 3, 1], [1, 2, 2, 0], [6, 2, 8, 4]
    p, low, high = boot(errors_a, errors_b, lengths, 2000, 3)
    p_swap, low_swap, high_swap = boot(errors_b, errors_a, lengths, 2000, 3)
    almost(low_swap, -high, 1e-9, label="swapped ci_low, which should be -ci_high")
    almost(high_swap, -low, 1e-9, label="swapped ci_high, which should be -ci_low")
    if p + p_swap > 1 + 1e-12:
        fail(f"p_b_better is {p:.4f} and {p_swap:.4f} after swapping; the two cannot add to more than 1")
    again = boot(errors_a, errors_b, lengths, 2000, 3)
    if again != (p, low, high):
        fail(f"the same seed gave {(p, low, high)} and then {again}; a seeded bootstrap must repeat")


def reference_bootstrap(errors_a, errors_b, lengths, n, seed):
    """The definition, drawn with a different generator: utterance indices, pooled edits."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(lengths), size=(n, len(lengths)))
    diff = np.array([(errors_a[row].sum() - errors_b[row].sum()) / lengths[row].sum() for row in picks])
    return float(np.mean(diff > 0)), *np.percentile(diff, [2.5, 97.5])


@check("paired_bootstrap matches a reference bootstrap over utterances")
def test_bootstrap_reference(sub):
    boot = has(sub, "paired_bootstrap")
    # Wrong implementations this check catches, on this data (40 utterances, three of them short):
    #   - Resampling words instead of utterances: P(B better) 0.911 and ci_high 0.046, against 0.764 and 0.071.
    #   - Averaging per-utterance WERs instead of summing edits over summed lengths: P(B better) 0.970
    #     and ci_high 0.201, because the short utterances count as much as the long ones.
    rng = np.random.default_rng(6)
    lengths = rng.integers(1, 31, size=40)
    lengths[:3] = [1, 2, 1]
    errors_a = rng.binomial(lengths, 0.24)
    errors_b = rng.binomial(lengths, 0.20)
    errors_a[:3], errors_b[:3] = [1, 2, 1], [0, 0, 0]
    want_p, want_low, want_high = reference_bootstrap(errors_a, errors_b, lengths, 40000, 99)
    p, low, high = boot(errors_a, errors_b, lengths, 4000, 1)
    almost(p, want_p, 0.03, label="p_b_better against the reference bootstrap")
    almost(low, want_low, 0.01, label="ci_low against the reference bootstrap")
    almost(high, want_high, 0.01, label="ci_high against the reference bootstrap")


@check("normalize lowercases, strips punctuation, and writes number words as digits")
def test_normalize(sub):
    normalize = has(sub, "normalize")
    cases = [("Set a timer for 10 minutes, please.", "set a timer for 10 minutes please"),
             ("Call Mom at six o'clock!", "call mom at 6 oclock"),
             ("  Thanks,   that's ALL. ", "thanks thats all"),
             ("Ten, two; ZERO", "10 2 0"),
             ("", "")]
    for text, want in cases:
        got = normalize(text)
        if got != want:
            fail(f"normalize({text!r}): expected {want!r}, got {got!r}")


@check("Normalizing both sides moves corpus WER from 66.7% to 13.3% to 0%")
def test_normalize_effect(sub):
    wer, normalize = has(sub, "wer"), has(sub, "normalize")
    refs = ["Set a timer for 10 minutes, please.", "Call Mom at 6 o'clock.", "Thanks, that's all."]
    hyps = ["set a timer for ten minutes please", "call mom at six o'clock", "thanks that's all"]
    almost(wer(refs, hyps), 10 / 15, 1e-12, label="raw corpus WER, 10 edits over 15 words")
    lower = lambda text: " ".join(text.lower().replace(",", "").replace(".", "").split())
    almost(wer([lower(r) for r in refs], [lower(h) for h in hyps]), 2 / 15, 1e-12,
           label="WER after lowercasing and stripping punctuation, 2 edits over 15 words")
    almost(wer([normalize(r) for r in refs], [normalize(h) for h in hyps]), 0.0, 1e-12,
           label="WER after normalize, 0 edits over 15 words")


@check("cer scores text with no spaces: one wrong Chinese character is 1/7, while wer sees one word")
def test_cer_unsegmented(sub):
    wer, cer = has(sub, "wer"), has(sub, "cer")
    ref, hyp = "我想订一张机票", "我想定一张机票"
    almost(cer(ref, hyp), 1 / 7, 1e-12, label="cer of one substituted character in seven")
    almost(wer([ref], [hyp]), 1.0, 1e-12, label="wer when the whole string is one word")
    almost(cer(ref, ref + "了"), 1 / 7, 1e-12, label="cer with one inserted character")
