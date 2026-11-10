"""Spoken language understanding.

Fill in intent accuracy, micro slot F1, an n-best intent, sentence BLEU,
character F1, and a dialogue-state update. Press Check to test them, and
Run to score the alarm request from the chapter.
"""

import math

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
    raise NotImplementedError("slot_f1")


def nbest_intent(hypotheses: list[dict], log_probs: list[float]) -> str:
    """Intent with the highest total probability.

    `hypotheses[i]["intent"]` is a string. `log_probs[i]` is the natural log
    of that hypothesis's probability. Sum the probabilities inside each
    intent and return the intent with the largest sum.
    """
    raise NotImplementedError("nbest_intent")


def bleu(hypothesis: str, reference: str) -> float:
    """Sentence BLEU with orders 1 and 2 and smoothing 2.

    Split both strings on whitespace. Clip n-gram counts by the reference.
    For order 2, add 1 to the clipped hits and to the hypothesis count.
    Weights are 1/2. The brevity penalty is exp(1 - r/c) when the hypothesis
    has c words and the reference has r words with c <= r, and 1 otherwise.
    """
    raise NotImplementedError("bleu")


def chrf(hypothesis: str, reference: str) -> float:
    """Character unigram F1. Spaces are not characters. Beta is 1.

    Clip each hypothesis character count by the reference count, then
    return 2PR/(P+R).
    """
    raise NotImplementedError("chrf")


def update_dialogue_state(state: dict, user_act: dict, system_act: dict) -> dict:
    """Merge one user act and one system act into a fresh state.

    `state` and `user_act` have `"intent"` and `"slots"`. `system_act` is
    stored under `"system_act"`. Copy the slots, then update that copy with
    the user act. A user intent that is not None replaces the stored intent.
    Do not edit `state`.
    """
    raise NotImplementedError("update_dialogue_state")


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
