"""Checks for the streaming encoders exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


def as_bool(m):
    return np.asarray(m, dtype=bool)


@check("chunked_attention_mask lets each frame see its own chunk and every chunk before it")
def test_mask(sub):
    mask_fn = has(sub, "chunked_attention_mask")
    m = mask_fn(6, 2)
    shape(m, (6, 6), label="mask")
    want = np.array([[1, 1, 0, 0, 0, 0]] * 2 + [[1, 1, 1, 1, 0, 0]] * 2 + [[1] * 6] * 2, dtype=bool)
    equal(as_bool(m).tolist(), want.tolist(), label="the mask for T=6, chunk 2, all left context")


@check("chunked_attention_mask keeps only left_chunks earlier chunks when asked")
def test_mask_left(sub):
    mask_fn = has(sub, "chunked_attention_mask")
    m = as_bool(mask_fn(7, 2, left_chunks=1))
    ref = np.array([[(j // 2 <= i // 2) and (j // 2 >= i // 2 - 1) for j in range(7)] for i in range(7)])
    equal(m.tolist(), ref.tolist(), label="the mask for T=7, chunk 2, one left chunk")


@check("context_frames counts the frames each query may see")
def test_context(sub):
    mask_fn, context = has(sub, "chunked_attention_mask"), has(sub, "context_frames")
    equal([int(v) for v in context(mask_fn(6, 2))], [2, 2, 4, 4, 6, 6], label="context with all left chunks")
    equal([int(v) for v in context(mask_fn(6, 2, left_chunks=1))], [2, 2, 4, 4, 4, 4], label="context with one left chunk")


@check("lookahead_frames and delay_ms count the wait inside a chunk")
def test_delay(sub):
    look, delay = has(sub, "lookahead_frames"), has(sub, "delay_ms")
    equal([int(v) for v in look(4)], [3, 2, 1, 0], label="look-ahead for a chunk of 4")
    close(np.asarray(delay(4, 40.0), dtype=float), np.array([160.0, 120.0, 80.0, 40.0]), label="delay for a chunk of 4 frames of 40 ms")
    almost(float(np.max(delay(14, 80.0))), 1120.0, 1e-9, label="the worst delay of a chunk of 14 frames of 80 ms")
    almost(float(np.min(delay(7, 25.0))), 25.0, 1e-9, label="the last frame of a chunk waits only for itself")


@check("masked_self_attention matches a loop, and masked keys get weight exactly 0")
def test_attention(sub):
    mask_fn, attn = has(sub, "chunked_attention_mask"), has(sub, "masked_self_attention")
    torch.manual_seed(0)
    T, d = 6, 3
    x = torch.randn(T, d, dtype=torch.float64)
    mask = mask_fn(T, 2, left_chunks=1)
    out, weights = attn(x, mask)
    shape(out, (T, d), label="output")
    shape(weights, (T, T), label="weights")
    xn, mn = x.numpy(), as_bool(mask)
    want_w = np.zeros((T, T))
    for i in range(T):
        scores = np.array([xn[i] @ xn[j] / math.sqrt(d) if mn[i, j] else -np.inf for j in range(T)])
        e = np.exp(scores - scores.max())
        want_w[i] = e / e.sum()
    close(weights.detach().numpy(), want_w, atol=1e-12, label="attention weights")
    close(out.detach().numpy(), want_w @ xn, atol=1e-12, label="attention output")
    if np.any(weights.detach().numpy()[~mn] != 0.0):
        fail("a frame outside the mask got a nonzero weight; set its score to -inf before the softmax")


@check("With the chunked mask, changing future chunks leaves earlier frames' outputs unchanged")
def test_streaming(sub):
    mask_fn, attn = has(sub, "chunked_attention_mask"), has(sub, "masked_self_attention")
    torch.manual_seed(1)
    x = torch.randn(12, 8, dtype=torch.float64)
    mask = mask_fn(12, 4)
    y1, _ = attn(x, mask)
    x2 = x.clone()
    x2[8:] += torch.randn(4, 8, dtype=torch.float64)
    y2, _ = attn(x2, mask)
    close(y2[:8].detach().numpy(), y1[:8].detach().numpy(), atol=1e-12, label="outputs of frames 0 to 7 after changing frames 8 to 11")
    if torch.allclose(y2[8:], y1[8:]):
        fail("the last chunk's outputs didn't change when its own frames did; the mask blocks too much")
    full, _ = attn(x, torch.ones(12, 12, dtype=torch.bool))
    x3 = x.clone()
    x3[8:] += 1.0
    full3, _ = attn(x3, torch.ones(12, 12, dtype=torch.bool))
    if torch.allclose(full[:8], full3[:8]):
        fail("with no mask, early frames should depend on future frames; masked_self_attention ignores its keys")


@check("A bigger chunk buys more context and costs more delay; left_chunks caps the context")
def test_tradeoff(sub):
    mask_fn, context, delay = has(sub, "chunked_attention_mask"), has(sub, "context_frames"), has(sub, "delay_ms")
    means, delays = [], []
    for c in (1, 2, 4, 8):
        means.append(float(context(mask_fn(64, c, left_chunks=2)).float().mean()))
        delays.append(float(np.mean(delay(c, 40.0))))
    if not (np.all(np.diff(means) > 0) and np.all(np.diff(delays) > 0)):
        fail(f"mean context {means} and mean delay {delays} should both grow with the chunk size")
    longest = int(context(mask_fn(400, 4, left_chunks=2)).max())
    equal(longest, 12, label="the most frames any query sees with chunk 4 and 2 left chunks, however long the audio")
