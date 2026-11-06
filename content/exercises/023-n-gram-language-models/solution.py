"""An interpolated Kneser-Ney bigram language model, and noisy-channel rescoring.

The worked solution fills in the five functions below, then press Run to train a bigram model on a small
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
    """Counts of every bigram (history, word), with <s> and </s> added to each sentence."""
    counts = Counter()
    for words in sentences:
        padded = [BOS] + list(words) + [EOS]
        counts.update(zip(padded[:-1], padded[1:]))
    return counts


def continuation_probs(counts: Counter) -> dict[str, float]:
    """P_cont(w): the share of distinct bigram types that end in w."""
    predecessors = Counter(w for (_, w) in counts)
    types = len(counts)
    return {w: n / types for w, n in predecessors.items()}


def kn_prob(word: str, history: str, counts: Counter, d: float = 0.75) -> float:
    """Interpolated Kneser-Ney bigram probability P(word | history)."""
    p_cont = continuation_probs(counts).get(word, 0.0)
    c_history = sum(c for (h, _), c in counts.items() if h == history)
    if c_history == 0:
        return p_cont
    followers = sum(1 for (h, _) in counts if h == history)
    discounted = max(counts.get((history, word), 0) - d, 0.0) / c_history
    backoff_weight = d * followers / c_history
    return discounted + backoff_weight * p_cont


def perplexity(sentences: list[list[str]], prob) -> float:
    """Perplexity of `prob(word, history)` on the sentences, counting </s> but not <s>."""
    log2_total, n = 0.0, 0
    for words in sentences:
        padded = [BOS] + list(words) + [EOS]
        for history, word in zip(padded[:-1], padded[1:]):
            p = prob(word, history)
            if p <= 0:
                return math.inf
            log2_total += math.log2(p)
            n += 1
    return 2 ** (-log2_total / n)


def rescore(nbest: list[tuple[str, float]], prob, lm_weight: float, insertion_penalty: float) -> str:
    """The hypothesis with the best acoustic + weighted LM + insertion score."""

    def score(item):
        text, acoustic = item
        padded = [BOS] + text.split() + [EOS]
        lm = sum(math.log(max(prob(w, h), 1e-300)) for h, w in zip(padded[:-1], padded[1:]))
        return acoustic + lm_weight * lm + insertion_penalty * len(text.split())

    return max(nbest, key=score)[0]


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
