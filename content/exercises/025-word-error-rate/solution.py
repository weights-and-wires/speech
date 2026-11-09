"""Word error rate from a Levenshtein alignment.

The worked solution fills in the three functions below, then press Run to align a hypothesis against a
reference, see the edit-distance table, and compare word and character error
rates. Press Check to test your code.
"""

from collections import Counter

import numpy as np

from speechlab import show_image


def levenshtein_align(ref: list, hyp: list):
    """Minimum-edit alignment. Returns (S, D, I, alignment).

    alignment is a list of (op, ref_token, hyp_token) with op one of "C", "S",
    "D", or "I", and None for the missing side of a deletion or insertion.
    """
    n, m = len(ref), len(hyp)
    E = np.zeros((n + 1, m + 1), dtype=int)
    E[:, 0] = np.arange(n + 1)
    E[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i, j] = min(E[i - 1, j] + 1, E[i, j - 1] + 1, E[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, alignment = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and E[i, j] == E[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "C" if ref[i - 1] == hyp[j - 1] else "S"
            alignment.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and E[i, j] == E[i - 1, j] + 1:
            alignment.append(("D", ref[i - 1], None))
            i -= 1
        else:
            alignment.append(("I", None, hyp[j - 1]))
            j -= 1
    alignment.reverse()
    ops = Counter(op for op, _, _ in alignment)
    return ops["S"], ops["D"], ops["I"], alignment


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus word error rate: total S + D + I over total reference words."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps):
        s, d, i, _ = levenshtein_align(ref.split(), hyp.split())
        errors += s + d + i
        words += len(ref.split())
    return errors / words


def cer(ref: str, hyp: str) -> float:
    """Character error rate: the edit distance over characters, spaces included."""
    s, d, i, _ = levenshtein_align(list(ref), list(hyp))
    return (s + d + i) / len(ref)


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
