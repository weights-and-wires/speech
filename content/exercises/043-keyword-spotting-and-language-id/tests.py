"""Checks for the keyword spotting, endpointing, and language identification exercise.

Plausible wrong implementations these checks catch:

- `detect` without a hold-off, so a bump that stays above the threshold for many
  frames fires on every one of them. `test_planted_stream` and `test_holdoff`
  catch it: the false accepts per hour jump from 60 to several hundred.
- `evaluate_stream` that divides by the keyword-only duration instead of the
  whole stream. `test_evaluate_stream` and `test_planted_stream` check the exact
  rate.
"""

import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape

HOP = 0.01


def make_stream(seed=0):
    """36 slots of 1000 frames: 20 keywords, 4 weak ones, 6 false bumps, 3 spikes, 3 empty."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 0.1, 36000)
    kinds = ["hit"] * 20 + ["weak"] * 4 + ["bump"] * 6 + ["spike"] * 3 + ["empty"] * 3
    spans = []
    for slot, k in enumerate(rng.permutation(len(kinds))):
        kind, start = kinds[k], slot * 1000 + 300
        if kind in ("hit", "weak"):
            scores[start : start + 60] += 0.9 if kind == "hit" else 0.3
            spans.append((start, start + 60))
        elif kind == "bump":
            scores[start : start + 40] += 0.8
        elif kind == "spike":
            scores[start] += 0.9
    return scores, spans


def script(parts):
    """Build (vad, eoq) from (n_frames, is_speech, eoq values) parts."""
    vad, eoq = [], []
    for n, speech, probs in parts:
        vad += [int(speech)] * n
        eoq += list(probs) if probs is not None else [0.0] * n
    return np.array(vad), np.array(eoq, dtype=float)


UTTERANCES = {
    # speech 0-5, a 3-frame pause, speech 9-11, then silence. Last speech frame 11.
    "A": (script([(6, 1, None), (3, 0, [0.1, 0.1, 0.2]), (3, 1, None), (8, 0, [0.3, 0.7] + [0.9] * 6)]), 11),
    # speech 0-7, then silence. Last speech frame 7.
    "B": (script([(8, 1, None), (8, 0, [0.2, 0.2, 0.3, 0.8] + [0.9] * 4)]), 7),
    # speech 0-3, a 6-frame hesitation, speech 10-12, then silence. Last speech frame 12.
    "C": (script([(4, 1, None), (6, 0, [0.1] * 6), (3, 1, None), (8, 0, [0.4, 0.8] + [0.9] * 6)]), 12),
}


@check("smooth_posteriors is a causal moving average that shortens the window at the start")
def test_smooth(sub):
    got = has(sub, "smooth_posteriors")
    raw = [0.1, 0.2, 0.9, 0.1, 0.8, 0.9, 0.7, 0.2, 0.1, 0.1, 0.1]
    out = got(np.array(raw), 3)
    shape(out, (11,), label="smoothed posteriors")
    want = [0.1, 0.15, 0.4, 0.4, 0.6, 0.6, 0.8, 0.6, 1 / 3, 2 / 15, 0.1]
    close(out, want, atol=1e-12, label="window 3 on the worked example")
    close(got(np.array(raw), 1), raw, atol=1e-12, label="a window of 1 changes nothing")
    # A causal filter must not read the future: changing the last frame leaves the rest alone.
    changed = np.array(raw)
    changed[-1] = 1.0
    close(got(changed, 3)[:-1], out[:-1], atol=1e-12, label="earlier frames after the last one changes")


@check("detect fires at the first frame over the threshold, then waits out the hold-off")
def test_holdoff(sub):
    got = has(sub, "detect")
    scores = np.array([0.1, 0.6, 0.7, 0.8, 0.2, 0.9, 0.9, 0.1, 0.9, 0.9])
    equal(list(got(scores, 0.5, 3)), [1, 5, 9], label="detections with hold-off 3")
    equal(list(got(scores, 0.5, 0)), [1, 2, 3, 5, 6, 8, 9], label="detections with no hold-off")
    equal(list(got(scores, 0.5, 100)), [1], label="detections with a long hold-off")
    equal(list(got(scores, 0.95, 3)), [], label="detections when nothing reaches the threshold")
    equal(list(got(np.array([0.5, 0.5]), 0.5, 0)), [0, 1], label="a score equal to the threshold fires")


@check("evaluate_stream counts false accepts per stream hour and the rate of missed keywords")
def test_evaluate_stream(sub):
    got = has(sub, "evaluate_stream")
    spans = [(10, 20), (50, 60), (90, 100)]
    fa, frr = got(np.array([10, 12, 20, 40, 95, 130]), spans, 1200, HOP)
    # 12 s is 1/300 hour. Three detections fall outside every span, and the span at 50 is missed.
    # The end of a span is excluded, so frame 20 is a false accept. Dividing by the 30 frames
    # of keyword audio instead would give 36000 per hour.
    almost(fa, 900.0, 1e-9, label="false accepts per hour")
    almost(frr, 1 / 3, 1e-12, label="false-reject rate")
    fa, frr = got(np.array([], dtype=int), spans, 3600, 1.0)
    almost(fa, 0.0, 1e-12, label="false accepts per hour with no detections")
    almost(frr, 1.0, 1e-12, label="false-reject rate with no detections")


@check("det_points lists p_miss and p_fa at every distinct score, plus infinity")
def test_det(sub):
    got = has(sub, "det_points")
    target = [0.9, 0.7, 0.5, 0.3]
    other = [0.6, 0.4, 0.2, 0.1]
    thr, p_miss, p_fa = got(target, other)
    shape(thr, (9,), label="thresholds")
    close(thr[:-1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9], atol=1e-12, label="finite thresholds")
    if not np.isposinf(thr[-1]):
        fail(f"the last threshold is {thr[-1]}, expected +inf, where nothing is accepted")
    close(p_miss, [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 1.0], atol=1e-12, label="p_miss")
    close(p_fa, [1.0, 0.75, 0.5, 0.5, 0.25, 0.25, 0, 0, 0], atol=1e-12, label="p_fa")
    if np.any(np.diff(p_miss) < 0) or np.any(np.diff(p_fa) > 0):
        fail("p_miss must never fall and p_fa must never rise as the threshold rises")


@check("The planted stream: smoothing and hold-off give 60 false accepts per hour and an FRR of 1/6")
def test_planted_stream(sub):
    smooth, det, ev = has(sub, "smooth_posteriors"), has(sub, "detect"), has(sub, "evaluate_stream")
    scores, spans = make_stream(0)
    equal(len(spans), 24, label="keyword spans in the stream")
    found = det(smooth(scores, 5), 0.5, 100)
    fa, frr = ev(found, spans, len(scores), HOP)
    # Six false bumps in 0.1 hour is 60 per hour. Four of the 24 keywords are weak, so 1/6 are missed.
    almost(fa, 60.0, 1e-9, label="false accepts per hour with window 5 and hold-off 100")
    almost(frr, 1 / 6, 1e-12, label="false-reject rate")
    raw_found = det(scores, 0.5, 100)
    raw_fa, _ = ev(raw_found, spans, len(scores), HOP)
    almost(raw_fa, 90.0, 1e-9, label="false accepts per hour without smoothing, where three spikes also fire")
    bursty_fa, _ = ev(det(smooth(scores, 5), 0.5, 0), spans, len(scores), HOP)
    if bursty_fa <= 60.0:
        fail(f"with no hold-off each false bump fires many times, but got only {bursty_fa} per hour")


@check("endpoint ends the query at a short timeout when the end-of-query probability is high")
def test_endpoint(sub):
    got = has(sub, "endpoint")
    (vad, eoq), _ = UTTERANCES["A"]
    equal(got(vad, eoq, 5, 5), 17, label="fixed 5-frame timeout")
    equal(got(vad, eoq, 2, 2), 8, label="fixed 2-frame timeout, which cuts the pause")
    equal(got(vad, eoq, 1, 5), 13, label="short 1 when end-of-query is high, long 5 otherwise")
    equal(got(vad, eoq, 1, 5, 0.95), 17, label="the same with a threshold of 0.95")
    equal(got(np.zeros(6), np.ones(6), 1, 5), -1, label="no speech, so no endpoint")
    equal(got(np.array([1, 1, 0]), np.ones(3), 1, 5), -1, label="a pause shorter than the limit never ends")


@check("endpoint_metrics scores cut-off rate and mean latency on scripted utterances")
def test_endpoint_metrics(sub):
    ep, metrics = has(sub, "endpoint"), has(sub, "endpoint_metrics")
    last = [UTTERANCES[k][1] for k in "ABC"]
    for name, args, cut_want, lat_want in [
        ("fixed 5 frames", (5, 5), 1 / 3, 600.0),
        ("fixed 2 frames", (2, 2), 2 / 3, 300.0),
        ("short 1, long 8, with end-of-query", (1, 8), 0.0, 800.0 / 3),
    ]:
        decisions = [ep(UTTERANCES[k][0][0], UTTERANCES[k][0][1], *args) for k in "ABC"]
        cut, latency = metrics(decisions, last, 100.0)
        almost(cut, cut_want, 1e-12, label=f"cut-off rate, {name}")
        almost(latency, lat_want, 1e-9, label=f"mean latency in ms, {name}")
    cut, latency = metrics([3], [5], 10.0)
    almost(cut, 1.0, 1e-12, label="cut-off rate when the only decision is early")
    if not math.isnan(latency):
        fail(f"mean latency is {latency}, expected nan when every utterance is cut off")


@check("language_llr compares each language with the mean likelihood of the others")
def test_llr(sub):
    got = has(sub, "language_llr")
    ll = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = got(ll)
    shape(out, (2, 3), label="LLR matrix")
    m = math.log((math.e**2 + 1) / 2)
    close(out[0], [2.0, -m, -m], atol=1e-12, label="LLRs of the first trial")
    m1 = math.log((math.e + 1) / 2)
    close(out[1], [-m1, 1.0, -m1], atol=1e-12, label="LLRs of the second trial")
    close(got(ll + 7.0), out, atol=1e-12, label="LLRs after adding a constant to every score")


@check("cavg matches the worked example at beta = 1 and beta = 9, and is 0 for a perfect system")
def test_cavg(sub):
    got = has(sub, "cavg")
    ll = np.array([[2, 0, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0], [0, 0, 3], [0, 2, 0]], dtype=float)
    labels = np.array([0, 0, 1, 1, 2, 2])
    # beta = 1: P_miss = 1/2, 0, 1/2 and the P_fa terms sum to 1.5, so Cavg = (1 + 1.5 / 2) / 3.
    almost(got(ll, labels), 1.75 / 3, 1e-12, label="Cavg at beta = 1")
    # beta = 9 raises the threshold to log 9 = 2.197. Only the trial [0, 0, 3] clears it, so
    # P_miss = 1, 1, 1/2 and there are no false alarms: Cavg = 2.5 / 3.
    almost(got(ll, labels, 9.0), 2.5 / 3, 1e-12, label="Cavg at beta = 9")
    perfect = np.array([[5, 0, 0], [5, 0, 0], [0, 5, 0], [0, 5, 0], [0, 0, 5], [0, 0, 5]], dtype=float)
    almost(got(perfect, labels), 0.0, 1e-12, label="Cavg of a perfect system")
    almost(got(ll + 3.0, labels), got(ll, labels), 1e-12, label="Cavg after adding a constant to every score")
