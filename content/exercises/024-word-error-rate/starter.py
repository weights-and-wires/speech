"""Word error rate from a Levenshtein alignment.

Fill in the three functions below, then press Run to align a hypothesis against a
reference, see the edit-distance table, and compare word and character error
rates. Press Check to test your code.
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
