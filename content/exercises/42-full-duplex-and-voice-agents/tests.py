"""Checks for the full-duplex models and voice agents exercise."""

import itertools

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


@check("inner_monologue places each word at its start frame, with PAD between and EPAD just before each word")
def test_monologue(sub):
    mono = has(sub, "inner_monologue")
    got = np.asarray(mono([2, 7, 13], [[101, 102], [103], [104, 105, 106]], 18, pad=0, epad=1))
    equal(got.tolist(), [0, 1, 101, 102, 0, 0, 1, 103, 0, 0, 0, 0, 1, 104, 105, 106, 0, 0], label="text stream")
    # A word that starts at frame 0 has no frame before it, and back-to-back words leave no room for EPAD.
    got = np.asarray(mono([0, 2], [[7, 8], [9]], 5, pad=0, epad=1))
    equal(got.tolist(), [7, 8, 9, 0, 0], label="text stream with back-to-back words")


@check("moshi_delays lays out 17 streams, delay_streams shifts each by its delay, and undelay_streams inverts it")
def test_delay(sub):
    moshi, delay, undelay = has(sub, "moshi_delays"), has(sub, "delay_streams"), has(sub, "undelay_streams")
    equal(list(moshi(8, 2)), [0, 0] + [2] * 7 + [0] + [2] * 7, label="moshi_delays(8, 2)")
    equal(list(moshi(2, 1)), [0, 0, 1, 0, 1], label="moshi_delays(2, 1)")
    codes = np.arange(1, 13).reshape(3, 4)
    got = np.asarray(delay(codes, [0, 1, 2], pad=0))
    equal(got.tolist(), [[1, 2, 3, 4, 0, 0], [0, 5, 6, 7, 8, 0], [0, 0, 9, 10, 11, 12]], label="delayed grid")
    rng = np.random.default_rng(0)
    codes = rng.integers(2, 50, size=(17, 30))
    ds = list(moshi(8, 1))
    delayed = np.asarray(delay(codes, ds, pad=-1))
    shape(delayed, (17, 31), label="delayed Moshi grid")
    for k, d in enumerate(ds):
        equal(delayed[k, d : d + 30], codes[k], label=f"stream {k}")
    equal(np.asarray(undelay(delayed, ds, 30)), codes, label="undelay_streams(delay_streams(codes))")


@check("interleave_fixed_ratio alternates 13 text and 26 speech tokens, then finishes the longer stream")
def test_interleave(sub):
    inter = has(sub, "interleave_fixed_ratio")
    text, speech = np.arange(20), np.arange(60)
    got = np.asarray(inter(text, speech, 13, 26, speech_offset=1000))
    want = list(range(13)) + list(range(1000, 1026)) + list(range(13, 20)) + list(range(1026, 1052)) + list(range(1052, 1060))
    equal(got.tolist(), want, label="interleaved tokens")
    got = np.asarray(inter(np.arange(3), np.arange(4), 2, 1, speech_offset=10))
    equal(got.tolist(), [0, 1, 10, 2, 11, 12, 13], label="interleave with 2:1 and short streams")


# A tiny random RQ-Transformer: a temporal context from past frames, a depth model over codebooks.
V, K, D = 3, 2, 4
_rng = np.random.default_rng(1)
EMB = _rng.normal(size=(K, V, D))
A = _rng.normal(size=(D, D))
W = _rng.normal(size=(K, V, D))
E = _rng.normal(size=(K, V, V))


def temporal_fn(prev):
    if len(prev) == 0:
        return np.zeros(D)
    h = sum(EMB[k, prev[:, k]].sum(axis=0) for k in range(K))
    return np.tanh(A @ h / len(prev))


def depth_fn(z, prefix):
    k = len(prefix)
    logits = W[k] @ z
    for j, c in enumerate(prefix):
        logits = logits + E[j, c] * (k - j)
    return logits


@check("frame_log_prob defines a distribution: summed over all V^K frames it gives probability 1")
def test_frame(sub):
    frame_lp = has(sub, "frame_log_prob")
    z = np.array([0.3, -1.0, 0.5, 2.0])
    total = sum(np.exp(frame_lp(z, np.array(f), depth_fn)) for f in itertools.product(range(V), repeat=K))
    almost(total, 1.0, 1e-9, label="sum over every frame of exp(frame_log_prob)")
    logits0 = depth_fn(z, np.array([], dtype=int))
    logits1 = depth_fn(z, np.array([2]))
    want = (logits0[2] - np.log(np.exp(logits0).sum())) + (logits1[1] - np.log(np.exp(logits1).sum()))
    almost(frame_lp(z, np.array([2, 1]), depth_fn), want, 1e-9, label="frame_log_prob(z, [2, 1])")


@check("sequence_log_prob sums the frames with a causal context, and the joint over all sequences sums to 1")
def test_sequence(sub):
    seq_lp, frame_lp = has(sub, "sequence_log_prob"), has(sub, "frame_log_prob")
    frames = list(itertools.product(range(V), repeat=K))
    total = 0.0
    for a, b in itertools.product(frames, repeat=2):
        total += np.exp(seq_lp(np.array([a, b]), temporal_fn, depth_fn))
    almost(total, 1.0, 1e-9, label="sum over all 81 two-frame sequences")
    codes = np.array([[0, 2], [1, 1], [2, 0]])
    want = sum(frame_lp(temporal_fn(codes[:s]), codes[s], depth_fn) for s in range(3))
    almost(seq_lp(codes, temporal_fn, depth_fn), want, 1e-9, label="sequence_log_prob of 3 frames")
    changed = codes.copy()
    changed[2] = [1, 2]
    almost(seq_lp(changed[:2], temporal_fn, depth_fn), seq_lp(codes[:2], temporal_fn, depth_fn), 1e-12,
           label="the first two frames' score after changing frame 3")


@check("sample_frame draws frames with the probabilities the depth model assigns")
def test_sample(sub):
    sample, frame_lp = has(sub, "sample_frame"), has(sub, "frame_log_prob")
    z = np.array([1.0, 0.2, -0.4, 0.0])
    rng = np.random.default_rng(2)
    counts = {}
    n = 6000
    for _ in range(n):
        f = tuple(int(c) for c in sample(z, depth_fn, K, rng))
        counts[f] = counts.get(f, 0) + 1
    tv = 0.5 * sum(abs(counts.get(f, 0) / n - np.exp(frame_lp(z, np.array(f), depth_fn)))
                   for f in itertools.product(range(V), repeat=K))
    if tv > 0.03:
        fail(f"the sampled frames are {tv:.3f} away in total variation from the model's distribution, expected under 0.03")


@check("causal_conv1d matches torch's conv1d with left padding, and ignores the future")
def test_causal(sub):
    conv = has(sub, "causal_conv1d")
    rng = np.random.default_rng(3)
    x, w, b = rng.normal(size=(2, 50)), rng.normal(size=(3, 2, 4)), rng.normal(size=3)
    got = np.asarray(conv(x, w, b, dilation=2))
    shape(got, (3, 50), label="output")
    xp = torch.nn.functional.pad(torch.tensor(x)[None], (6, 0))
    want = torch.nn.functional.conv1d(xp, torch.tensor(w), torch.tensor(b), dilation=2)[0].numpy()
    close(got, want, rtol=1e-10, atol=1e-12, label="causal_conv1d against torch")
    x2 = x.copy()
    x2[:, 30:] += 5.0
    close(np.asarray(conv(x2, w, b, dilation=2))[:, :30], got[:, :30], rtol=0, atol=1e-12,
          label="outputs before frame 30 after changing the input from frame 30 on")


@check("StreamingConv1d fed in uneven chunks gives exactly the offline output")
def test_streaming(sub):
    conv, Streaming = has(sub, "causal_conv1d"), has(sub, "StreamingConv1d")
    rng = np.random.default_rng(4)
    x, w, b = rng.normal(size=(2, 200)), rng.normal(size=(3, 2, 5)), rng.normal(size=3)
    offline = np.asarray(conv(x, w, b, dilation=3))
    for sizes in ([1] * 200, [7, 0, 13, 1, 50, 129], [200]):
        stream = Streaming(w, b, dilation=3)
        out, i = [], 0
        for n in sizes:
            out.append(np.asarray(stream.step(x[:, i : i + n])))
            i += n
        close(np.concatenate(out, axis=1), offline, rtol=0, atol=1e-10, label=f"chunk sizes {sizes[:6]}")


@check("The latency budget adds up, and Moshi's algorithmic latency is 160 ms")
def test_latency(sub):
    resp, duplex = has(sub, "response_latency_ms"), has(sub, "full_duplex_latency_ms")
    almost(resp(500, 50, 250, 120, 40, 40), 1000.0, 1e-9, label="response_latency_ms")
    almost(duplex(12.5, 1), 160.0, 1e-9, label="full_duplex_latency_ms(12.5, 1)")
    almost(duplex(12.5, 2), 240.0, 1e-9, label="full_duplex_latency_ms(12.5, 2)")


@check("turn_taking fires the right events, ignores a backchannel, and yields to a barge-in")
def test_turn_taking(sub):
    tt = has(sub, "turn_taking")
    speech = np.zeros(300, dtype=bool)
    speech[10:60] = True    # the user's question
    speech[62:64] = True    # a two-frame blip inside the endpoint window
    speech[150:155] = True  # "mm-hm" while the agent talks: 5 frames, too short
    speech[200:260] = True  # the user interrupts
    states, events = tt(speech, 10, min_speech_ms=100, endpoint_ms=300, response_ms=200, reply_ms=1500)
    equal(len(states), 300, label="number of states")
    # Detected at frame 19 (10 frames of speech); the silence run restarts after the blip at 62-63,
    # so the endpoint fires at frame 93; the agent starts 20 frames later and is cut off at frame 209,
    # and the interruption's own endpoint comes 30 silent frames after it ends, at frame 289.
    equal([tuple(e) for e in events],
          [(19, "user_start"), (93, "endpoint"), (113, "agent_start"), (209, "barge_in"), (289, "endpoint")],
          label="events")
    equal([states[t] for t in (5, 30, 61, 70, 100, 152, 209, 280)],
          ["LISTENING", "USER_SPEAKING", "ENDPOINT_PENDING", "ENDPOINT_PENDING", "RESPONDING",
           "AGENT_SPEAKING", "USER_SPEAKING", "ENDPOINT_PENDING"], label="states at frames 5, 30, 61, 70, 100, 152, 209, 280")
