"""Word error rate from a Levenshtein alignment.

The worked solution fills in the five functions below, then press Run to align a hypothesis against a
reference, see the edit-distance table, compare word and character error
rates, normalize two transcripts, and test whether one system beats another.
Press Check to test your code.
"""

import re
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


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])}


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces, and write 0 to 10 as digits."""
    text = re.sub(r"[^\w\s]", " ", text.lower().replace("'", "").replace("\u2019", ""))
    return " ".join(NUMBER_WORDS.get(word, word) for word in text.split())


def paired_bootstrap(errors_a, errors_b, ref_lengths, n_resamples=1000, seed=0):
    """Paired bootstrap over utterances: does system B beat system A?

    Each resample draws len(ref_lengths) utterance indices with replacement, uses the same
    indices for both systems, and recomputes corpus WER as summed edits over summed reference
    words. The difference is WER(A) - WER(B), so a positive value means B is better.
    Returns (p_b_better, ci_low, ci_high): the fraction of resamples with a strictly positive
    difference, and the 2.5th and 97.5th percentiles of the difference.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    ref_lengths = np.asarray(ref_lengths, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ref_lengths), size=(n_resamples, len(ref_lengths)))
    diff = (errors_a[picks].sum(axis=1) - errors_b[picks].sum(axis=1)) / ref_lengths[picks].sum(axis=1)
    low, high = np.percentile(diff, [2.5, 97.5])
    return float(np.mean(diff > 0)), float(low), float(high)


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
