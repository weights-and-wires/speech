"""Checks for the spoken language understanding exercise.

The numbers are the alarm request from the chapter. No check draws random
numbers. Wrong implementations named in the comments are not imported. The
checks fail them by expecting the hand-computed value.
"""

import math

from speechcheck import almost, check, fail, has


def _call(fn, *args, expected: str):
    """Call `fn`. A NotImplementedError fails with the expected value."""
    try:
        return fn(*args)
    except NotImplementedError:
        fail(f"expected {expected}, actual NotImplementedError")


@check("intent_accuracy is 1 on the one-best SetAlarm pair")
def test_intent_accuracy(sub):
    fn = has(sub, "intent_accuracy")
    one = _call(fn, ["SetAlarm"], ["SetAlarm"], expected="intent accuracy 1.0")
    half = _call(
        fn,
        ["SetAlarm", "PlayMusic"],
        ["SetAlarm", "SetAlarm"],
        expected="intent accuracy 0.5",
    )
    almost(one, 1.0, 1e-12, label="one-best SetAlarm")
    almost(half, 0.5, 1e-12, label="one match out of two")


@check("slot_f1 falls to 2/3 when the room is missing")
def test_slot_f1(sub):
    # intent_as_slot_f1 reports intent accuracy as the slot score and fails
    # the missing-room example: accuracy is 1, slot F1 is 2/3.
    fn = has(sub, "slot_f1")
    missing = [{"time": "07:00"}]
    gold = [{"time": "07:00", "room": "kitchen"}]
    perfect = [{"time": "07:00", "room": "kitchen"}]
    one = _call(fn, missing, gold, expected="slot F1 2/3")
    pooled = _call(
        fn,
        missing + perfect,
        gold + gold,
        expected="pooled slot F1 6/7",
    )
    almost(one, 2.0 / 3.0, 1e-12, label="missing room")
    almost(pooled, 6.0 / 7.0, 1e-12, label="pooled with a perfect utterance")


@check("nbest_intent sums probability mass and returns SetAlarm")
def test_nbest_intent(sub):
    # one_best_intent returns the intent of the first hypothesis and fails
    # this list: the first intent is PlayMusic, the mass is SetAlarm.
    fn = has(sub, "nbest_intent")
    hypotheses = [
        {"intent": "PlayMusic", "slots": {}},
        {"intent": "SetAlarm", "slots": {"time": "07:00"}},
        {"intent": "SetAlarm", "slots": {"time": "07:00", "room": "kitchen"}},
    ]
    log_probs = [math.log(0.40), math.log(0.35), math.log(0.25)]
    got = _call(fn, hypotheses, log_probs, expected="intent 'SetAlarm'")
    if got != "SetAlarm":
        fail(f"expected intent 'SetAlarm', actual {got!r}")


def _hand_bleu() -> float:
    """Sentence BLEU for 'set an alarm' against 'set the kitchen alarm'."""
    p1 = 2.0 / 3.0
    p2 = 1.0 / 3.0
    penalty = math.exp(1.0 - 4.0 / 3.0)
    mean = math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))
    return penalty * mean


@check("bleu matches the hand count, including the brevity penalty")
def test_bleu(sub):
    # bleu_without_brevity forgets the penalty and fails because the
    # hypothesis is shorter than the reference: about 0.4714, not 0.3378.
    fn = has(sub, "bleu")
    want = _hand_bleu()
    got = _call(
        fn,
        "set an alarm",
        "set the kitchen alarm",
        expected=f"sentence BLEU {want:.6f}",
    )
    almost(got, want, 1e-9, label="sentence BLEU")


@check("chrf matches the hand-counted character F-score 9/14")
def test_chrf(sub):
    fn = has(sub, "chrf")
    got = _call(
        fn,
        "set an alarm",
        "set the kitchen alarm",
        expected="character F1 9/14",
    )
    almost(got, 9.0 / 14.0, 1e-12, label="character F1")


@check("update_dialogue_state keeps the time on the second turn")
def test_dialogue_state(sub):
    # replace_state keeps only this turn and fails on the second turn:
    # the slots become {"room": "kitchen"} and drop time 07:00.
    fn = has(sub, "update_dialogue_state")
    state = {"intent": None, "slots": {}, "system_act": None}
    first_user = {"intent": "SetAlarm", "slots": {"time": "07:00"}}
    first_system = {"act": "request", "slot": "room"}
    second_user = {"intent": "SetAlarm", "slots": {"room": "kitchen"}}
    second_system = {"act": "confirm", "slot": "room"}
    want_slots = {"time": "07:00", "room": "kitchen"}
    mid = _call(
        fn,
        state,
        first_user,
        first_system,
        expected="slots {'time': '07:00'} after turn 1",
    )
    got = _call(
        fn,
        mid,
        second_user,
        second_system,
        expected=f"slots {want_slots} after turn 2",
    )
    if not isinstance(got, dict):
        fail(f"expected a state dict, actual {got!r}")
    if got.get("slots") != want_slots:
        fail(f"expected slots {want_slots}, actual {got.get('slots')}")
    if got.get("intent") != "SetAlarm":
        fail(f"expected intent 'SetAlarm', actual {got.get('intent')!r}")
    if got.get("system_act") != second_system:
        fail(
            f"expected system_act {second_system}, actual {got.get('system_act')}"
        )
