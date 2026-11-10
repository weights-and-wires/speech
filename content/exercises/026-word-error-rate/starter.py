"""Word error rate from a Levenshtein alignment.

Fill in the five functions below, then press Run to align a hypothesis against a
reference, see the edit-distance table, compare word and character error
rates, normalize two transcripts, and test whether one system beats another.
Press Check to test your code.
"""

from collections import Counter

import numpy as np

from speechlab import show_image


def levenshtein_align(ref: list, hyp: list):
    """Minimum-edit alignment of a hypothesis against a reference.

    Args:
        ref: The reference tokens, a list.
        hyp: The hypothesis tokens, a list.

    Returns:
        (S, D, I, alignment): the numbers of substitutions, deletions, and
        insertions in one minimum-edit alignment, and the alignment itself as
        a list of (op, ref_token, hyp_token) in order, with op one of "C"
        (correct), "S", "D", or "I", and None for the missing side of a
        deletion or an insertion.
    """
    raise NotImplementedError("levenshtein_align")


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus word error rate.

    Args:
        refs: Reference transcripts, a list of strings.
        hyps: Hypothesis transcripts, the same length.

    Returns:
        The total S + D + I over all pairs divided by the total number of
        reference words, a float. Split each string on whitespace.
    """
    raise NotImplementedError("wer")


def cer(ref: str, hyp: str) -> float:
    """Character error rate of one hypothesis.

    Args:
        ref: The reference string.
        hyp: The hypothesis string.

    Returns:
        The character-level edit distance divided by len(ref). Spaces count
        as characters.
    """
    raise NotImplementedError("cer")


def normalize(text: str) -> str:
    """Normalize a transcript before scoring it.

    Args:
        text: One transcript, a string.

    Returns:
        The text lowercased, with apostrophes deleted, every other punctuation
        mark turned into a space, spaces collapsed, and the number words "zero"
        to "ten" written as the digits "0" to "10".
    """
    raise NotImplementedError("normalize")


def paired_bootstrap(errors_a, errors_b, ref_lengths, n_resamples=1000, seed=0):
    """Paired bootstrap over utterances: does system B beat system A?

    Args:
        errors_a: Edits (S + D + I) of system A on each utterance, length n.
        errors_b: Edits of system B on the same utterances, length n.
        ref_lengths: Reference words in each utterance, length n.
        n_resamples: How many resamples to draw.
        seed: Seed for numpy's default_rng.

    Returns:
        (p_b_better, ci_low, ci_high). Each resample draws n utterance indices
        with replacement, the same indices for both systems, and takes
        WER(A) - WER(B) as total edits over total reference words. p_b_better
        is the fraction of resamples where that difference is above 0. ci_low
        and ci_high are its 2.5th and 97.5th percentiles.
    """
    raise NotImplementedError("paired_bootstrap")


if __name__ == "__main__":
    ref = "set a timer for two minutes".split()
    hyp = "set the timer four to minutes please".split()
    S, D, I, alignment = levenshtein_align(ref, hyp)
    # Entry (i, j) is the edit distance between the first i and the first j words.
    table = np.array([[sum(levenshtein_align(ref[:i], hyp[:j])[:3]) for j in range(len(hyp) + 1)]
                      for i in range(len(ref) + 1)], dtype=float)
    show_image(table, label="Edit distance table, reference down and hypothesis across",
               x_labels=["-"] + hyp, y_labels=["-"] + ref, value_label="Edits")
    print(" ".join(f"{op}:{r or '-'}/{h or '-'}" for op, r, h in alignment))
    print(f"S={S} D={D} I={I}, WER = {(S + D + I) / len(ref):.1%}")

    # A homophone costs a whole word but often only a few characters.
    refs = ["write to mom", "call mom in two minutes"]
    hyps = ["right two mom", "call mom in to minutes"]
    for r, h in zip(refs, hyps):
        print(f"'{r}' -> '{h}': WER {wer([r], [h]):.1%}, CER {cer(r, h):.1%}")

    # Corpus WER weights every reference word equally; a mean of per-sentence WERs does not.
    refs = ["the cat sat on the mat", "hello"]
    hyps = ["a cat sat on mat", "hello there my friend"]
    per_sentence = [wer([r], [h]) for r, h in zip(refs, hyps)]
    print(f"Per-sentence WER {[f'{w:.1%}' for w in per_sentence]}, their mean {np.mean(per_sentence):.1%}, "
          f"corpus WER {wer(refs, hyps):.1%}")

    # Normalization decides the score: the same three pairs, scored three ways.
    refs = ["Set a timer for 10 minutes, please.", "Call Mom at 6 o'clock.", "Thanks, that's all."]
    hyps = ["set a timer for ten minutes please", "call mom at six o'clock", "thanks that's all"]
    lower_only = lambda text: " ".join(text.lower().split())
    for name, fn in [("raw", str), ("lowercase only", lower_only), ("lowercase, punctuation, numbers", normalize)]:
        print(f"{name:>32}: WER {wer([fn(r) for r in refs], [fn(h) for h in hyps]):.1%}")

    # Is system B really better than system A? Four utterances cannot say; 400 can.
    errors_a, errors_b, lengths = [2, 1, 3, 1], [1, 2, 2, 0], [6, 2, 8, 4]
    for copies in (1, 100):
        p, low, high = paired_bootstrap(errors_a * copies, errors_b * copies, lengths * copies, 10000, 0)
        print(f"{4 * copies} utterances: WER A {sum(errors_a) / sum(lengths):.1%}, WER B {sum(errors_b) / sum(lengths):.1%}, "
              f"P(B better) {p:.3f}, 95% interval of A - B [{low:+.1%}, {high:+.1%}]")

    # Without spaces, WER sees one word. CER sees the one wrong character.
    print(f"Mandarin: WER {wer(['我想订一张机票'], ['我想定一张机票']):.1%}, CER {cer('我想订一张机票', '我想定一张机票'):.1%}")
