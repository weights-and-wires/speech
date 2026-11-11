"""Checks for the streaming latency and on-device exercise."""

import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape

EVENTS = [
    (560, "partial", "turn"),
    (800, "partial", "turn on"),
    (1000, "partial", "turn on a"),
    (1480, "partial", "turn on the light"),
    (1900, "final", "turn on the lights"),
]
WORD_ENDS = [480, 640, 760, 1240]


@check("rtf is compute seconds per audio second")
def test_rtf(sub):
    rtf = has(sub, "rtf")
    almost(rtf(10.0, 3.0), 0.3, 1e-9, label="3 s of compute for 10 s of audio")
    almost(rtf(2.0, 3.0), 1.5, 1e-9, label="3 s of compute for 2 s of audio")


@check("algorithmic_latency gives each position its wait plus look-ahead")
def test_algorithmic(sub):
    fn = has(sub, "algorithmic_latency")
    got = np.asarray(fn(4, 2, 40))
    shape(got, (4,), label="delays")
    close(got, [240, 200, 160, 120], label="C=4, L=2, 40 ms")
    # Dropping the frame's own length gives 200, 160, 120, 80.
    close(np.asarray(fn(4, 0, 40)), [160, 120, 80, 40], label="C=4, L=0, 40 ms")
    almost(float(np.mean(got)), 180.0, 1e-9, label="mean delay")


@check("streaming_latency adds chunk compute, and is infinite when RTF is 1 or more")
def test_streaming(sub):
    fn = has(sub, "streaming_latency")
    # 0.5 * (4 * 40 ms) = 80 ms of compute for the chunk.
    close(np.asarray(fn(4, 2, 40, 0.5)), [320, 280, 240, 200], label="RTF 0.5")
    close(np.asarray(fn(4, 2, 40, 0.0)), [240, 200, 160, 120], label="RTF 0")
    got = np.asarray(fn(4, 2, 40, 1.0))
    if not np.all(np.isinf(got)):
        fail(f"at RTF 1.0 the backlog grows without bound, expected inf, got {got.tolist()}")


@check("partial_latencies measures from the word end, not from the chunk or audio start")
def test_first_final(sub):
    fn = has(sub, "partial_latencies")
    got = fn(EVENTS, WORD_ENDS, 1640)
    # Wrong implementation: report the event time itself (560 and 1900), which
    # measures from the start of the audio instead of from when the word ended.
    equal(got["first_partial_ms"], 80, label="first partial, 560 - 480")
    equal(got["finalization_ms"], 660, label="finalization, 1900 - 1240")


@check("partial_latencies measures endpoint-to-final from the endpoint decision")
def test_endpoint(sub):
    fn = has(sub, "partial_latencies")
    got = fn(EVENTS, WORD_ENDS, 1640)
    equal(got["endpoint_to_final_ms"], 260, label="1900 - 1640")
    # Finalization is the endpoint wait (1640 - 1240) plus the endpoint-to-final time.
    equal(got["finalization_ms"], (1640 - 1240) + got["endpoint_to_final_ms"], label="finalization split")
    events = [(300, "partial", ""), (450, "partial", "yes"), (900, "final", "yes")]
    got = fn(events, [400], 700)
    equal(got["first_partial_ms"], 50, label="an empty partial is not a first partial")


@check("unstable_prefix_rate counts shown words after the first word that differs from the final")
def test_unstable(sub):
    fn = has(sub, "unstable_prefix_rate")
    partials = [e[2] for e in EVENTS[:-1]]
    almost(fn(partials, "turn on the lights"), 0.2, 1e-9, label="2 of 10 shown words")
    almost(fn(["turn", "turn on"], "turn on"), 0.0, 1e-9, label="every partial is a prefix")
    almost(fn([], "turn on"), 0.0, 1e-9, label="nothing shown")


@check("unstable_prefix_rate does not call a partial stable because its length matches")
def test_unstable_length(sub):
    fn = has(sub, "unstable_prefix_rate")
    # Wrong implementation: compare len(partial) with len(final), which scores
    # this partial as stable although "of" never becomes "on".
    almost(fn(["turn of"], "turn on"), 0.5, 1e-9, label="same length, different word")
    # A wrong word breaks the prefix, so a later correct word is unstable too.
    almost(fn(["tune on the"], "turn on the"), 1.0, 1e-9, label="first word wrong")


@check("model_bytes and chunk_activation_bytes follow the arithmetic")
def test_bytes(sub):
    mb, act = has(sub, "model_bytes"), has(sub, "chunk_activation_bytes")
    equal(mb(30_000_000, 8), 30_000_000, label="30M at 8 bits")
    equal(mb(30_000_000, 4), 15_000_000, label="30M at 4 bits")
    equal(mb(7, 4), 4, label="7 weights at 4 bits round up")
    # (2*16*256 + 2*80*256 + 4*16*80) values * 2 bytes.
    equal(int(act(16, 64, 256, 4, 2)), 108_544, label="chunk of 16, 64 left, d=256, 4 heads, fp16")


@check("fits_on_device compares weights plus activations to the budget, and RTF decides real time")
def test_device(sub):
    fits, mb, rtf_fn = has(sub, "fits_on_device"), has(sub, "model_bytes"), has(sub, "device_rtf")
    act = 108_544
    exact = mb(30_000_000, 8) + act
    equal(bool(fits(30_000_000, 8, act, exact)), True, label="budget exactly equal to the need")
    equal(bool(fits(30_000_000, 8, act, exact - 1)), False, label="one byte short (activations count)")
    equal(bool(fits(30_000_000, 16, act, 40_000_000)), False, label="60 MB of weights in 40 MB")
    equal(bool(fits(30_000_000, 4, act, 40_000_000)), True, label="15 MB of weights in 40 MB")
    almost(rtf_fn(750e6, 2e9), 0.375, 1e-9, label="750M MAC/s on a 2G MAC/s device")
    if not math.isfinite(rtf_fn(750e6, 2e9)):
        fail("device_rtf must be finite")
