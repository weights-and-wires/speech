"""Language models and word error rate.

Fill in the functions below, then press Run to train a Kneser-Ney bigram model
on a small voice-assistant corpus, compare its perplexity with add-one
smoothing, align a hypothesis against a reference, and watch language-model
rescoring fix homophones. Press Check to test your code.
"""

import math
from collections import Counter

import numpy as np

from speechlab import show_image, show_plot

BOS, EOS = "<s>", "</s>"


def bigram_counts(sentences: list[list[str]]) -> Counter:
    """Count every bigram in a corpus.

    Args:
        sentences: A list of sentences, each a list of word strings.

    Returns:
        A Counter mapping (history, word) to its count, after adding BOS before
        and EOS after every sentence. The sentence ["a", "b"] contributes
        ("<s>", "a"), ("a", "b"), and ("b", "</s>").
    """
    raise NotImplementedError("bigram_counts")


def continuation_probs(counts: Counter) -> dict[str, float]:
    """The Kneser-Ney continuation probability of every predicted token.

    Args:
        counts: Bigram counts from bigram_counts.

    Returns:
        A dict mapping each word w that appears second in some bigram to
        (number of distinct histories v with counts[(v, w)] > 0) divided by
        (number of distinct bigram types).
    """
    raise NotImplementedError("continuation_probs")


def kn_prob(word: str, history: str, counts: Counter, d: float = 0.75) -> float:
    """Interpolated Kneser-Ney bigram probability P(word | history).

    P = max(C(history word) - d, 0) / C(history) + lambda(history) P_cont(word),
    with lambda(history) = d * (number of distinct words after history) / C(history),
    where C(history) is the total count of bigrams that start with history.
    For a history that never occurs, return P_cont(word).

    Args:
        word: The predicted token.
        history: The previous token.
        counts: Bigram counts from bigram_counts.
        d: The absolute discount, between 0 and 1.

    Returns:
        The probability, a float.
    """
    raise NotImplementedError("kn_prob")


def perplexity(sentences: list[list[str]], prob) -> float:
    """The perplexity of a bigram model on a list of sentences.

    Args:
        sentences: A list of sentences, each a list of word strings.
        prob: A function prob(word, history) returning P(word | history).

    Returns:
        2 ** (-(1 / N) * sum of log2 P(w_i | w_{i-1})), where the sum runs over
        every token of every sentence plus its closing EOS, and N counts those
        same tokens. BOS is a history only, never predicted. Returns math.inf
        when any probability is 0.
    """
    raise NotImplementedError("perplexity")


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


def rescore(nbest: list[tuple[str, float]], prob, lm_weight: float, insertion_penalty: float) -> str:
    """Pick the best hypothesis from an n-best list by the noisy-channel score.

    Each hypothesis scores acoustic + lm_weight * log P_LM(words) +
    insertion_penalty * (number of words), where log P_LM is the natural-log
    bigram probability of the words with BOS before and EOS after.

    Args:
        nbest: A list of (text, acoustic_log_score) pairs.
        prob: A function prob(word, history) returning P(word | history).
        lm_weight: The language model scale.
        insertion_penalty: A per-word bonus; negative values penalize words.

    Returns:
        The text of the best-scoring hypothesis.
    """
    raise NotImplementedError("rescore")


def make_corpus(rng, n):
    """Provided. Sentences from a small voice-assistant grammar."""
    numbers = ["one", "two", "three", "four", "five", "ten"]
    templates = [
        lambda: ["set", "a", "timer", "for", rng.choice(numbers), "minutes"],
        lambda: ["call", "mom", "in", rng.choice(numbers), "minutes"],
        lambda: ["turn", rng.choice(["left", "right"]), "in", rng.choice(numbers), "blocks"],
        lambda: ["write", "to", rng.choice(["mom", "dad", "ann"])],
        lambda: ["play", "track", rng.choice(numbers)],
        lambda: ["turn", rng.choice(["on", "off"]), "the", rng.choice(["light", "fan"])],
    ]
    return [templates[rng.integers(len(templates))]() for _ in range(n)]


# Words a recognizer can't tell apart by sound.
HOMOPHONES = {"two": ["to", "too"], "to": ["two", "too"], "for": ["four"], "four": ["for"],
              "write": ["right"], "right": ["write"], "one": ["won"]}


def fake_nbest(words, rng):
    """Provided. A simulated recognizer's 4-best list for one sentence.

    A homophone swap sounds the same as the truth, so its acoustic log score
    differs from the truth's only by noise. Other confusions score lower.
    """
    truth = " ".join(words)
    out = {truth: 0.0}
    for _ in range(50):
        if len(out) == 4:
            break
        w = list(words)
        k = int(rng.integers(len(w)))
        if w[k] in HOMOPHONES and rng.random() < 0.8:
            w[k] = str(rng.choice(HOMOPHONES[w[k]]))
            out.setdefault(" ".join(w), float(rng.normal(0.0, 0.5)))
        else:
            w[k] = str(rng.choice(["the", "a", "on", "and", "fun", "time"]))
            out.setdefault(" ".join(w), -float(rng.uniform(1.0, 4.0)))
    items = list(out.items())
    order = rng.permutation(len(items))
    return [items[k] for k in order]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    train, test = make_corpus(rng, 400), make_corpus(rng, 100)
    counts = bigram_counts(train)
    vocab = sorted({w for (_, w) in counts})
    print(f"{len(train)} training sentences, {len(counts)} bigram types, {len(vocab)} predicted tokens")

    ds = np.linspace(0.05, 0.95, 19)
    pps = [perplexity(test, lambda w, h, d=d: kn_prob(w, h, counts, d)) for d in ds]
    history_counts = Counter()
    for (h, _), c in counts.items():
        history_counts[h] += c
    add_one = perplexity(test, lambda w, h: (counts.get((h, w), 0) + 1) / (history_counts[h] + len(vocab)))
    show_plot((ds, pps), (ds, np.full_like(ds, add_one)), labels=["Kneser-Ney", "add-one"],
              label="Held-out perplexity against the discount d", xlabel="Discount d", ylabel="Perplexity")
    print(f"Held-out perplexity: Kneser-Ney with d = 0.75 gives {pps[14]:.2f}; add-one gives {add_one:.2f}")

    ref = "set a timer for two minutes".split()
    hyp = "set the timer four to minutes please".split()
    S, D, I, alignment = levenshtein_align(ref, hyp)
    table = np.zeros((len(ref) + 1, len(hyp) + 1))
    table[:, 0], table[0, :] = np.arange(len(ref) + 1), np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1,
                              table[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    show_image(table, label="Edit distance table, reference down and hypothesis across",
               x_labels=["-"] + hyp, y_labels=["-"] + ref, value_label="Edits")
    print(" ".join(f"{op}:{r or '-'}/{h or '-'}" for op, r, h in alignment))
    print(f"S={S} D={D} I={I}, WER = {(S + D + I) / len(ref):.1%}")

    refs, first, rescored = [], [], []
    lm = lambda w, h: kn_prob(w, h, counts)
    for words in test[:60]:
        nbest = fake_nbest(words, rng)
        refs.append(" ".join(words))
        first.append(max(nbest, key=lambda item: item[1])[0])  # acoustics alone
        rescored.append(rescore(nbest, lm, lm_weight=1.0, insertion_penalty=0.0))
    print(f"WER from acoustics alone: {wer(refs, first):.1%}; after LM rescoring: {wer(refs, rescored):.1%}")
