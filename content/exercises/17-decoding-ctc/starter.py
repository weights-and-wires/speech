"""Decoding CTC.

Every function takes the emissions of one utterance as a (T, C) array of
natural-log probabilities: row t is frame t, column k is symbol k, and each
row is already a log-softmax. Column 0 is the blank. A prefix is a tuple of
label indices, never containing the blank.

Fill in the five functions below, then press Run to decode a synthetic
sentence four ways, and Check to test your code.
"""

import math
from collections import defaultdict

import numpy as np

from speechlab import play, show_image, show_plot

BLANK = 0
NEG_INF = -math.inf


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK) -> list[int]:
    """Best-path decoding.

    Takes the most likely symbol in each frame of `log_probs`, shape (T, C),
    then collapses the path: merge runs of the same symbol, then drop blanks.
    Returns the labels as a list of ints, possibly empty.
    """
    raise NotImplementedError("greedy_decode")


def prefix_beam_search(
    log_probs: np.ndarray,
    beam_width: int,
    blank: int = BLANK,
    prefix_score=None,
) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search, with an optional extra score per prefix.

    Keeps, for every prefix in the beam, two log probabilities: p_b, the total
    probability of the paths so far that collapse to the prefix and end in a
    blank, and p_nb, those that end in the prefix's last label. Starts from
    {(): (0.0, -inf)}. After each frame, keeps the `beam_width` prefixes with
    the highest score, where

        score(prefix) = logaddexp(p_b, p_nb) + prefix_score(prefix)

    and `prefix_score` is None (meaning 0) or a function from a prefix tuple
    to a float, such as an LM score or a hotword bonus.

    Returns a list of (prefix, score) pairs, best first, at most `beam_width`
    long. Each prefix is a tuple of ints. Drop any prefix whose probability is
    zero (a score of -inf), such as "aa" after only two frames.
    """
    raise NotImplementedError("prefix_beam_search")


def train_char_bigram(texts: list[str], vocab: str, smoothing: float = 1.0) -> np.ndarray:
    """Add-k smoothed character bigram log probabilities.

    `vocab[i]` is the character of symbol i, and `vocab[0]` stands for the
    blank, which never appears in text. Returns `lm`, shape (C, C) with
    C = len(vocab), where lm[p, c] = log P(c | previous symbol p). Row 0 is the
    start-of-text context. Column 0 is -inf in every row, because the LM never
    predicts the blank. With counts n(p, c) over all texts:

        P(c | p) = (n(p, c) + smoothing) / (sum over c' >= 1 of n(p, c') + smoothing * (C - 1))
    """
    raise NotImplementedError("train_char_bigram")


def lm_score(prefix: tuple[int, ...], lm: np.ndarray, alpha: float, beta: float) -> float:
    """The shallow-fusion score of a prefix.

    Returns alpha * log P_LM(prefix) + beta * len(prefix), where
    log P_LM(prefix) sums lm[previous, current] over the prefix, with the
    previous symbol of the first label being 0, the start context. The empty
    prefix scores 0.
    """
    raise NotImplementedError("lm_score")


def hotword_score(
    prefix: tuple[int, ...], hotwords: list[tuple[int, ...]], bonus: float
) -> float:
    """A bonus for hotword symbols in a prefix.

    Returns bonus * (complete + partial), where `complete` is the total length
    of every occurrence of a hotword inside the prefix, and `partial` is the
    longest k, with 1 <= k < len(word) for some hotword, such that the prefix
    ends with that hotword's first k symbols. A partial match that fails to
    continue stops counting, so its bonus disappears.
    """
    raise NotImplementedError("hotword_score")


# The rest of the file sets up the Run experiment -------------------------------

VOCAB = "_ abcdefghijklmnopqrstuvwxyz"  # index 0 is the blank

CORPUS = [
    "the cat sat on the mat",
    "a cat is here",
    "the cat is here and the dog is there",
    "tell the cat to sit",
    "is the cat here",
    "the dog sat on the cat mat",
    "tell them the cat ran",
    "a kind man sat here",
    "the man and the cat",
    "tell me where the cat is",
    "the hat is on the cat",
    "here is the man",
]


def encode(text: str) -> tuple[int, ...]:
    return tuple(VOCAB.index(ch) for ch in text)


def decode(labels) -> str:
    return "".join(VOCAB[i] for i in labels)


def synthetic_emissions(spec, rng) -> np.ndarray:
    """Build (T, C) log probabilities from a list of (char, prob, rival, rival_prob, frames)."""
    rows = []
    size = len(VOCAB)

    def frame(probs):
        p = np.full(size, 0.0)
        for idx, value in probs.items():
            p[idx] += value
        rest = 1.0 - p.sum()
        p += rest / size
        p *= np.exp(rng.normal(0.0, 0.05, size))
        return np.log(p / p.sum())

    for ch, prob, rival, rival_prob, frames in spec:
        for _ in range(int(rng.integers(1, 3))):
            rows.append(frame({BLANK: 0.96}))
        for _ in range(frames):
            probs = {VOCAB.index(ch): prob}
            if rival:
                probs[VOCAB.index(rival)] = rival_prob
            probs[BLANK] = probs.get(BLANK, 0.0) + max(0.0, 0.97 - sum(probs.values()))
            rows.append(frame(probs))
    rows.append(frame({BLANK: 0.96}))
    return np.stack(rows)


def sonify(path: np.ndarray, rate: int = 16000, frame_seconds: float = 0.04) -> np.ndarray:
    """One short beep per non-blank frame, pitched by symbol; silence for blank."""
    n = int(rate * frame_seconds)
    t = np.arange(n) / rate
    fade = np.sin(np.pi * np.arange(n) / n)
    out = []
    for symbol in path:
        if symbol == BLANK:
            out.append(np.zeros(n))
        else:
            freq = 220.0 * 2 ** (symbol / 12)
            out.append(0.3 * fade * np.sin(2 * np.pi * freq * t))
    return np.concatenate(out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    truth = "tell zoe the cat is here"
    # Three hard spots: the z of "zoe" sounds like an s, the c of "cat" like a
    # k, and the h of "here" is faint and smeared over two frames.
    hard = {
        truth.index("zoe"): ("z", 0.40, "s", 0.55, 1),
        truth.index("cat"): ("c", 0.42, "k", 0.50, 1),
        truth.index("here"): ("h", 0.45, None, 0.0, 2),
    }
    spec = [hard.get(i, (ch, 0.9, None, 0.0, 1)) for i, ch in enumerate(truth)]
    log_probs = synthetic_emissions(spec, rng)
    print(f"Emissions: {log_probs.shape[0]} frames by {log_probs.shape[1]} symbols (blank is column 0)")

    show_image(
        np.exp(log_probs).T,
        label="The posteriorgram: one column per frame, one row per symbol",
        xlabel="Frame",
        ylabel="Symbol",
        y_labels=list(VOCAB),
        colormap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    show_plot(
        np.exp(log_probs[:, BLANK]),
        label="The blank's probability in each frame: CTC is peaky",
        xlabel="Frame",
        ylabel="P(blank)",
        ylim=(0.0, 1.0),
    )

    greedy = greedy_decode(log_probs)
    print(f"greedy          : {decode(greedy)!r}")
    beam = prefix_beam_search(log_probs, beam_width=8)
    print(f"beam            : {decode(beam[0][0])!r}  (log p = {beam[0][1]:.2f})")

    lm = train_char_bigram(CORPUS, VOCAB, smoothing=0.5)
    alpha, beta = 0.8, 1.0
    fused = prefix_beam_search(
        log_probs, beam_width=8, prefix_score=lambda p: lm_score(p, lm, alpha, beta)
    )
    print(f"beam + bigram LM: {decode(fused[0][0])!r}")

    hotwords = [encode("zoe")]
    boosted = prefix_beam_search(
        log_probs,
        beam_width=8,
        prefix_score=lambda p: lm_score(p, lm, alpha, beta) + hotword_score(p, hotwords, 2.0),
    )
    print(f"+ hotword 'zoe' : {decode(boosted[0][0])!r}")
    print(f"truth           : {truth!r}")

    path = np.argmax(log_probs, axis=1)
    play(sonify(path), 16000, label="The greedy path: one beep per non-blank frame, silence for blank")
