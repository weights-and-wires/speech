"""An interpolated Kneser-Ney bigram language model, and noisy-channel rescoring.

Fill in the five functions below, then press Run to train a bigram model on a small
voice-assistant corpus, compare its perplexity with add-one smoothing, and
watch language-model rescoring fix homophones in a simulated recognizer's
n-best lists. Press Check to test your code.
"""

import math
from collections import Counter

import numpy as np

from speechlab import show_plot

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


# Provided: a small voice-assistant grammar, and a simulated recognizer. You don't
# need to change anything in this block.


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

    # Rescore simulated 4-best lists. The true sentence has acoustic score 0.0.
    lm = lambda w, h: kn_prob(w, h, counts)
    first_right = rescored_right = 0
    shown = False
    for words in test[:60]:
        nbest = fake_nbest(words, rng)
        truth = " ".join(words)
        best_acoustic = max(nbest, key=lambda item: item[1])[0]
        best_rescored = rescore(nbest, lm, lm_weight=1.0, insertion_penalty=0.0)
        first_right += best_acoustic == truth
        rescored_right += best_rescored == truth
        if not shown and best_acoustic != truth and best_rescored == truth:
            shown = True
            print(f"Acoustics alone pick '{best_acoustic}'; with the language model, '{best_rescored}'.")
    print(f"Sentences picked correctly out of 60: acoustics alone {first_right}, after rescoring {rescored_right}")
