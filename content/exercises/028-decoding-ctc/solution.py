"""Decoding CTC: greedy search and prefix beam search. The worked solution.

Every function takes the emissions of one utterance as a (T, C) array of
natural-log probabilities: row t is frame t, column k is symbol k, and each
row is already a log-softmax. Column 0 is the blank. A prefix is a tuple of
label indices, never containing the blank.
"""

import math
from collections import defaultdict

import numpy as np

from speechlab import play, show_image, show_plot

BLANK = 0
NEG_INF = -math.inf


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK) -> list[int]:
    """Best-path decoding: the most likely symbol per frame, then collapse."""
    path = np.argmax(log_probs, axis=1)
    out = []
    previous = None
    for symbol in path.tolist():
        if symbol != previous and symbol != blank:
            out.append(symbol)
        previous = symbol
    return out


def prefix_beam_search(
    log_probs: np.ndarray,
    beam_width: int,
    blank: int = BLANK,
    prefix_score=None,
) -> list[tuple[tuple[int, ...], float]]:
    """CTC prefix beam search, with an optional extra score per prefix."""
    extra_cache: dict[tuple[int, ...], float] = {}

    def extra(prefix):
        if prefix_score is None:
            return 0.0
        if prefix not in extra_cache:
            extra_cache[prefix] = float(prefix_score(prefix))
        return extra_cache[prefix]

    def total(item):
        prefix, (p_b, p_nb) = item
        return np.logaddexp(p_b, p_nb) + extra(prefix)

    beam = {(): (0.0, NEG_INF)}
    n_symbols = log_probs.shape[1]
    for frame in log_probs:
        nxt = defaultdict(lambda: (NEG_INF, NEG_INF))
        for prefix, (p_b, p_nb) in beam.items():
            p_all = np.logaddexp(p_b, p_nb)
            # The frame is a blank: the prefix stays, and now ends in a blank.
            b, nb = nxt[prefix]
            nxt[prefix] = (np.logaddexp(b, p_all + frame[blank]), nb)
            for c in range(n_symbols):
                if c == blank:
                    continue
                p_c = frame[c]
                extended = prefix + (c,)
                if prefix and c == prefix[-1]:
                    # A repeat only starts a new copy after a blank.
                    b, nb = nxt[extended]
                    nxt[extended] = (b, np.logaddexp(nb, p_b + p_c))
                    # Without a blank, it merges into the same prefix.
                    b, nb = nxt[prefix]
                    nxt[prefix] = (b, np.logaddexp(nb, p_nb + p_c))
                else:
                    b, nb = nxt[extended]
                    nxt[extended] = (b, np.logaddexp(nb, p_all + p_c))
        ranked = sorted(nxt.items(), key=total, reverse=True)
        # A prefix with probability zero, such as "aa" after two frames, is dropped.
        beam = {p: s for p, s in ranked[:beam_width] if np.logaddexp(*s) > NEG_INF}
    ranked = sorted(beam.items(), key=total, reverse=True)
    return [(prefix, float(total((prefix, scores)))) for prefix, scores in ranked]


# The rest of the file sets up the Run experiment -------------------------------

VOCAB = "_ abcdefghijklmnopqrstuvwxyz"  # index 0 is the blank

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

    print(f"truth           : {truth!r}")

    path = np.argmax(log_probs, axis=1)
    play(sonify(path), 16000, label="The greedy path: one beep per non-blank frame, silence for blank")
