#!/usr/bin/env python3
"""Numbers and bar geometry for the spoken-language-understanding lesson.

The worked example in content/lessons/063-spoken-language-understanding.md
uses these counts. Run it from the repo root:

    python3 scripts/figures/063-spoken-language-understanding.py

It prints slot F1, the n-best intent masses, sentence BLEU, the character
F-score, and the SVG bar rectangles those values produce.
"""

from __future__ import annotations

import math
from collections import Counter


HYP_TOKENS = ["set", "an", "alarm"]
REF_TOKENS = ["set", "the", "kitchen", "alarm"]
HYP_TEXT = "set an alarm"
REF_TEXT = "set the kitchen alarm"

# One-best SLU hypothesis drops the room. Gold keeps both pairs.
ONE_BEST_SLOTS = {"time": "07:00"}
GOLD_SLOTS = {"time": "07:00", "room": "kitchen"}
FULL_SLOTS = {"time": "07:00", "room": "kitchen"}

# N-best, best first. Natural logs of 0.40, 0.35, and 0.25.
NBEST_INTENTS = ["PlayMusic", "SetAlarm", "SetAlarm"]
NBEST_PROBS = [0.40, 0.35, 0.25]


def slot_counts(pred: dict, gold: dict) -> tuple[int, int, int]:
    """True positives, false positives, and false negatives for one utterance."""
    pred_pairs = set(pred.items())
    gold_pairs = set(gold.items())
    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int) -> float:
    """Micro F1. Both sides empty is a perfect 1."""
    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def clipped(hyp: list[str], ref: list[str], n: int) -> tuple[int, int]:
    """Clipped matches and the hypothesis n-gram count, before smoothing."""
    hyp_counts = Counter(ngrams(hyp, n))
    ref_counts = Counter(ngrams(ref, n))
    matches = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    return matches, sum(hyp_counts.values())


def sentence_bleu(hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Sentence BLEU, N=2, uniform weights, Chen and Cherry smoothing 2.

    Smoothing 2 adds 1 to the match count and the hypothesis count for
    every order n >= 2. Unigrams stay raw. The brevity penalty is 1 when
    the hypothesis is longer than the reference, and exp(1 - r/c) otherwise.
    """
    precisions = []
    for n in (1, 2):
        matches, total = clipped(hyp, ref, n)
        if n >= 2:
            matches += 1
            total += 1
        precisions.append(matches / total)
    c, r = len(hyp), len(ref)
    brevity = 1.0 if c > r else math.exp(1.0 - r / c)
    geometric = math.exp(0.5 * math.log(precisions[0]) + 0.5 * math.log(precisions[1]))
    return {
        "p1": precisions[0],
        "p2": precisions[1],
        "brevity": brevity,
        "geometric": geometric,
        "bleu": brevity * geometric,
        "bleu_without_brevity": geometric,
    }


def character_f1(hyp: str, ref: str) -> dict[str, float]:
    """Character unigram F1. Spaces are not characters. Beta is 1."""
    hyp_counts = Counter(ch for ch in hyp if ch != " ")
    ref_counts = Counter(ch for ch in ref if ch != " ")
    overlap = sum(min(count, ref_counts[ch]) for ch, count in hyp_counts.items())
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    score = 2.0 * precision * recall / (precision + recall)
    return {
        "overlap": overlap,
        "hyp": sum(hyp_counts.values()),
        "ref": sum(ref_counts.values()),
        "precision": precision,
        "recall": recall,
        "chrf": score,
        "hyp_counts": dict(sorted(hyp_counts.items())),
        "ref_counts": dict(sorted(ref_counts.items())),
    }


def bars(values: list[float], vmax: float, x0: float, width: float, gap: float,
         baseline: float, span: float) -> list[tuple[float, float, float, float]]:
    """Rectangles (x, y, width, height) with y growing downward.

    Coordinates are rounded to three decimals, which is what the lesson SVG uses.
    """
    rects = []
    for index, value in enumerate(values):
        height = value / vmax * span
        x = x0 + index * (width + gap)
        y = baseline - height
        rects.append((round(x, 3), round(y, 3), round(width, 3), round(height, 3)))
    return rects


def main() -> None:
    tp, fp, fn = slot_counts(ONE_BEST_SLOTS, GOLD_SLOTS)
    one = f1(tp, fp, fn)
    tp2, fp2, fn2 = slot_counts(FULL_SLOTS, GOLD_SLOTS)
    micro = f1(tp + tp2, fp + fp2, fn + fn2)
    print(f"one-best slot counts tp={tp} fp={fp} fn={fn} f1={one:.12f}")
    print(f"micro counts tp={tp + tp2} fp={fp + fp2} fn={fn + fn2} f1={micro:.12f}")
    print(f"macro mean of the two F1 scores={(one + f1(tp2, fp2, fn2)) / 2:.12f}")

    masses: dict[str, float] = {}
    for intent, prob in zip(NBEST_INTENTS, NBEST_PROBS):
        masses[intent] = masses.get(intent, 0.0) + prob
    print("n-best masses", {name: round(value, 2) for name, value in masses.items()})
    print("log probs", [round(math.log(prob), 6) for prob in NBEST_PROBS])

    bleu = sentence_bleu(HYP_TOKENS, REF_TOKENS)
    for name, value in bleu.items():
        print(f"bleu {name}={value:.12f}")

    chrf = character_f1(HYP_TEXT, REF_TEXT)
    print(
        f"chrf overlap={chrf['overlap']} hyp={chrf['hyp']} ref={chrf['ref']} "
        f"p={chrf['precision']:.12f} r={chrf['recall']:.12f} f={chrf['chrf']:.12f}"
    )
    print("hyp character counts", chrf["hyp_counts"])
    print("ref character counts", chrf["ref_counts"])

    # Figure 1. Four scores on a 0-to-1 axis. Baseline y=190, span=140.
    scores = [1.0, 1.0, 0.5, one]
    print("figure 1 rects", bars(scores, 1.0, 108, 78, 36, 190, 140))

    # Figure 2. Three hypothesis probabilities, then two intent masses.
    # A wider gap separates the groups. Baseline y=188, span=140, vmax=0.70.
    hyp_rects = bars(NBEST_PROBS, 0.70, 92, 52, 18, 188, 140)
    mass_rects = bars([masses["PlayMusic"], masses["SetAlarm"]], 0.70, 360, 64, 28, 188, 140)
    print("figure 2 hypothesis rects", hyp_rects)
    print("figure 2 mass rects", mass_rects)


if __name__ == "__main__":
    main()
