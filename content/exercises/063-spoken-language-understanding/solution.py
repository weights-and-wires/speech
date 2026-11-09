"""Spoken language understanding: the worked functions."""

import math
from collections import Counter

from speechlab import show_plot


def intent_accuracy(pred: list[str], gold: list[str]) -> float:
    """Fraction of utterances whose predicted intent equals the gold intent.

    `pred` and `gold` are equal-length lists of intent strings, and `gold`
    is non-empty. Returns a float in [0, 1].
    """
    if len(pred) != len(gold) or not gold:
        raise ValueError("pred and gold must be non-empty and the same length")
    hits = sum(one == other for one, other in zip(pred, gold))
    return hits / len(gold)


def slot_f1(pred: list[dict], gold: list[dict]) -> float:
    """Micro F1 over slot-value pairs.

    `pred` and `gold` are equal-length lists of slot dicts, one dict per
    utterance. A pair is `(name, value)`. Pool true positives, false
    positives, and false negatives across utterances, then return one F1.
    When every count is zero, return 1.0.
    """
    true_pos = false_pos = false_neg = 0
    for one_pred, one_gold in zip(pred, gold):
        pred_pairs = set(one_pred.items())
        gold_pairs = set(one_gold.items())
        true_pos += len(pred_pairs & gold_pairs)
        false_pos += len(pred_pairs - gold_pairs)
        false_neg += len(gold_pairs - pred_pairs)
    if true_pos + false_pos + false_neg == 0:
        return 1.0
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def nbest_intent(hypotheses: list[dict], log_probs: list[float]) -> str:
    """Intent with the highest total probability.

    `hypotheses[i]["intent"]` is a string. `log_probs[i]` is the natural log
    of that hypothesis's probability. Sum the probabilities inside each
    intent and return the intent with the largest sum.
    """
    if len(hypotheses) != len(log_probs) or not hypotheses:
        raise ValueError("hypotheses and log_probs must be non-empty and equal")
    # Shift by the largest log so a large negative score does not underflow.
    # The shift is a shared factor and does not change the winning intent.
    peak = max(log_probs)
    mass: dict[str, float] = {}
    for hyp, log_p in zip(hypotheses, log_probs):
        intent = hyp["intent"]
        mass[intent] = mass.get(intent, 0.0) + math.exp(log_p - peak)
    return max(mass, key=mass.get)


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _precision(hyp: list[str], ref: list[str], n: int) -> float:
    """Modified precision. Smoothing 2 adds one count for n >= 2."""
    hyp_counts = Counter(_ngrams(hyp, n))
    ref_counts = Counter(_ngrams(ref, n))
    matches = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    total = sum(hyp_counts.values())
    if n >= 2:
        matches += 1
        total += 1
    return matches / total


def bleu(hypothesis: str, reference: str) -> float:
    """Sentence BLEU with orders 1 and 2 and smoothing 2.

    Split both strings on whitespace. Clip n-gram counts by the reference.
    For order 2, add 1 to the clipped hits and to the hypothesis count.
    Weights are 1/2. The brevity penalty is exp(1 - r/c) when the hypothesis
    has c words and the reference has r words with c <= r, and 1 otherwise.
    """
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not hyp or not ref:
        raise ValueError("hypothesis and reference must each contain a word")
    p1 = _precision(hyp, ref, 1)
    p2 = _precision(hyp, ref, 2)
    c, r = len(hyp), len(ref)
    penalty = 1.0 if c > r else math.exp(1.0 - r / c)
    mean = math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))
    return penalty * mean


def chrf(hypothesis: str, reference: str) -> float:
    """Character unigram F1. Spaces are not characters. Beta is 1.

    Clip each hypothesis character count by the reference count, then
    return 2PR/(P+R).
    """
    hyp_counts = Counter(ch for ch in hypothesis.lower() if ch != " ")
    ref_counts = Counter(ch for ch in reference.lower() if ch != " ")
    if not hyp_counts or not ref_counts:
        raise ValueError("hypothesis and reference must each contain a character")
    overlap = sum(min(count, ref_counts[ch]) for ch, count in hyp_counts.items())
    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def update_dialogue_state(state: dict, user_act: dict, system_act: dict) -> dict:
    """Merge one user act and one system act into a fresh state.

    `state` and `user_act` have `"intent"` and `"slots"`. `system_act` is
    stored under `"system_act"`. Copy the slots, then update that copy with
    the user act. A user intent that is not None replaces the stored intent.
    Do not edit `state`.
    """
    slots = dict(state["slots"])
    slots.update(user_act["slots"])
    intent = user_act["intent"] if user_act.get("intent") is not None else state["intent"]
    return {
        "intent": intent,
        "slots": slots,
        "system_act": dict(system_act),
    }


if __name__ == "__main__":
    print("intent accuracy", intent_accuracy(["SetAlarm"], ["SetAlarm"]))
    print(
        "slot F1",
        slot_f1([{"time": "07:00"}], [{"time": "07:00", "room": "kitchen"}]),
    )
    hypotheses = [
        {"intent": "PlayMusic", "slots": {}},
        {"intent": "SetAlarm", "slots": {"time": "07:00"}},
        {"intent": "SetAlarm", "slots": {"time": "07:00", "room": "kitchen"}},
    ]
    log_probs = [math.log(0.40), math.log(0.35), math.log(0.25)]
    print("n-best intent", nbest_intent(hypotheses, log_probs))
    print("BLEU", bleu("set an alarm", "set the kitchen alarm"))
    print("chrF", chrf("set an alarm", "set the kitchen alarm"))
    state = {"intent": None, "slots": {}, "system_act": None}
    state = update_dialogue_state(
        state,
        {"intent": "SetAlarm", "slots": {"time": "07:00"}},
        {"act": "request", "slot": "room"},
    )
    state = update_dialogue_state(
        state,
        {"intent": "SetAlarm", "slots": {"room": "kitchen"}},
        {"act": "confirm", "slot": "room"},
    )
    print(state)
    show_plot(
        [0.40, 0.35, 0.25],
        label="N-best probabilities, best first",
        xlabel="Hypothesis rank",
        ylabel="Probability",
    )
