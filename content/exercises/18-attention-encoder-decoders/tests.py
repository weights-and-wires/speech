"""Checks for the attention encoder-decoder exercise."""

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from speechcheck import almost, check, close, equal, fail, has, shape


def t(x):
    return torch.tensor(np.asarray(x), dtype=torch.float64)


def numpy_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@check("dot_product_attention gives softmax(q k / sqrt(d)) weights and their weighted sum of values")
def test_dot(sub):
    attend = has(sub, "dot_product_attention")
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(2, 4)), rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 3))
    context, weights = attend(t(q), t(k), t(v))
    shape(weights, (2, 5), label="weights")
    shape(context, (2, 3), label="context")
    want_w = numpy_softmax(np.einsum("bd,bud->bu", q, k) / 2.0)
    close(weights.detach().numpy(), want_w, atol=1e-10, label="attention weights")
    close(context.detach().numpy(), np.einsum("bu,bud->bd", want_w, v), atol=1e-10, label="context vector")


@check("dot_product_attention gives masked frames zero weight, as if they were never there")
def test_dot_mask(sub):
    attend = has(sub, "dot_product_attention")
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(1, 4)), rng.normal(size=(1, 6, 4)), rng.normal(size=(1, 6, 2))
    mask = torch.tensor([[True, True, True, True, False, False]])
    context, weights = attend(t(q), t(k), t(v), mask)
    close(weights[0, 4:].detach().numpy(), np.zeros(2), atol=0.0, label="weights on the two padded frames")
    short_c, short_w = attend(t(q), t(k[:, :4]), t(v[:, :4]))
    close(weights[0, :4].detach().numpy(), short_w[0].detach().numpy(), atol=1e-12, label="weights on the real frames")
    close(context.detach().numpy(), short_c.detach().numpy(), atol=1e-12, label="context with padding")


def location_reference(s, h, a_prev, W, V, b, U, Fil, w):
    B, T, _ = h.shape
    n_f, _, k = Fil.shape
    half = k // 2
    padded = np.pad(a_prev, ((0, 0), (half, half)))
    f = np.zeros((B, T, n_f))
    for bi in range(B):
        for u in range(T):
            f[bi, u] = Fil[:, 0, :] @ padded[bi, u : u + k]
    e = np.zeros((B, T))
    for bi in range(B):
        for u in range(T):
            e[bi, u] = w @ np.tanh(W @ s[bi] + V @ h[bi, u] + U @ f[bi, u] + b)
    weights = numpy_softmax(e)
    return np.einsum("bt,btd->bd", weights, h), weights


@check("location_aware_attention matches a frame-by-frame loop over w . tanh(W s + V h + U f + b)")
def test_location(sub):
    attend = has(sub, "location_aware_attention")
    rng = np.random.default_rng(2)
    B, T, d_s, d_h, d_a, n_f, k = 2, 7, 3, 4, 5, 2, 3
    s, h = rng.normal(size=(B, d_s)), rng.normal(size=(B, T, d_h))
    a_prev = numpy_softmax(rng.normal(size=(B, T)))
    W, V, b = rng.normal(size=(d_a, d_s)), rng.normal(size=(d_a, d_h)), rng.normal(size=d_a)
    U, Fil, w = rng.normal(size=(d_a, n_f)), rng.normal(size=(n_f, 1, k)), rng.normal(size=d_a)
    context, weights = attend(t(s), t(h), t(a_prev), t(W), t(V), t(b), t(U), t(Fil), t(w))
    want_c, want_w = location_reference(s, h, a_prev, W, V, b, U, Fil, w)
    shape(weights, (B, T), label="weights")
    close(weights.detach().numpy(), want_w, atol=1e-10, label="attention weights")
    close(context.detach().numpy(), want_c, atol=1e-10, label="context vector")


@check("With a shift filter and no content term, location-aware attention steps forward one frame")
def test_location_moves(sub):
    attend = has(sub, "location_aware_attention")
    T = 8
    h = torch.zeros(1, T, 2, dtype=torch.float64)
    h[0, :, 0] = torch.arange(T, dtype=torch.float64)
    a_prev = torch.zeros(1, T, dtype=torch.float64)
    a_prev[0, 3] = 1.0
    zeros = torch.zeros(1, 1, dtype=torch.float64)
    shift = torch.tensor([[[5.0, 0.0, 0.0]]], dtype=torch.float64)  # f_u = 5 * alpha_prev[u - 1]
    context, weights = attend(
        torch.zeros(1, 1, dtype=torch.float64), h, a_prev,
        zeros, torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64), shift, torch.ones(1, dtype=torch.float64),
    )
    peak = int(torch.argmax(weights[0]))
    if peak != 4:
        fail(f"the previous attention peaked at frame 3, so the shift filter should move the peak to frame 4; it is at frame {peak}")


@check("label_smoothing_loss matches F.cross_entropy(label_smoothing=eps, reduction='sum') / B, skipping padding")
def test_smoothing(sub):
    loss_fn = has(sub, "label_smoothing_loss")
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 7, dtype=torch.float64)
    targets = torch.randint(0, 7, (3, 5))
    targets[1, 3:] = -1
    targets[2, 1:] = -1
    for eps in (0.0, 0.1, 0.3):
        got = loss_fn(logits, targets, eps)
        want = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), ignore_index=-1, label_smoothing=eps, reduction="sum") / 3
        almost(float(got), float(want), 1e-9, label=f"the loss with epsilon {eps}")
    uniform = loss_fn(torch.zeros(1, 2, 4, dtype=torch.float64), torch.tensor([[1, 2]]), 0.1)
    almost(float(uniform), 2 * math.log(4), 1e-9, label="the loss of two tokens under uniform logits, 2 ln 4")


@check("joint_ctc_attention_loss mixes lam * CTC + (1 - lam) * attention, with <eos> appended to each target")
def test_joint(sub):
    joint = has(sub, "joint_ctc_attention_loss")
    torch.manual_seed(1)
    T, B, C, U = 9, 2, 5, 3
    ctc_lp = torch.randn(T, B, C, dtype=torch.float64).log_softmax(-1)
    ctc_len = torch.tensor([9, 7])
    targets = torch.tensor([[1, 2, 3], [4, 2, -1]])
    t_len = torch.tensor([3, 2])
    logits = torch.randn(B, U + 1, C, dtype=torch.float64)
    att_targets = torch.tensor([[1, 2, 3, 0], [4, 2, 0, -1]])
    att = F.cross_entropy(logits.reshape(-1, C), att_targets.reshape(-1), ignore_index=-1, label_smoothing=0.1, reduction="sum") / B
    ctc = F.ctc_loss(ctc_lp, targets.clamp(min=0), ctc_len, t_len, reduction="sum") / B
    for lam in (0.0, 0.3, 1.0):
        got = joint(ctc_lp, ctc_len, logits, targets, t_len, lam, 0.1)
        almost(float(got), float(lam * ctc + (1 - lam) * att), 1e-9, label=f"the joint loss with lam {lam}")


def toy_decoder(prefix):
    """A fixed next-token distribution over {eos=0, 1, 2} that depends on the whole prefix."""
    rng = np.random.default_rng(hash(tuple(prefix)) % (2**32))
    logits = rng.normal(0.0, 1.5, size=3)
    logits[0] += 0.3 * len(prefix) - 0.5
    return logits - np.log(np.exp(logits).sum())


def brute_force(step_fn, max_len, penalty):
    best = []
    for n in range(max_len + 1):
        for seq in itertools.product([1, 2], repeat=n):
            score = sum(step_fn(seq[:i])[seq[i]] for i in range(n))
            if n < max_len:
                score += step_fn(seq)[0]
            lp = 1.0 if penalty is None else ((5 + n) / 6) ** penalty
            best.append((seq, score / lp))
    best.sort(key=lambda item: item[1], reverse=True)
    return best


@check("beam_search with a wide beam finds the brute-force best hypothesis, with and without a length penalty")
def test_beam_exact(sub):
    beam = has(sub, "beam_search")
    for penalty in (None, 1.0):
        want = brute_force(toy_decoder, 4, penalty)
        got = beam(toy_decoder, 0, 100, 4, length_penalty=penalty)
        best, score = got[0]
        equal(tuple(int(x) for x in best), want[0][0], label=f"the best hypothesis, length_penalty={penalty}")
        almost(float(score), want[0][1], 1e-9, label=f"its score, length_penalty={penalty}")
        if len(got) != len(want):
            fail(f"a beam of 100 should keep all {len(want)} hypotheses of up to 4 tokens, got {len(got)}")


@check("The GNMT length penalty flips the ranking from the short hypothesis to the long one")
def test_length_penalty(sub):
    beam = has(sub, "beam_search")
    table = {(): [0.5, 0.5, 0.0], (1,): [0.05, 0.05, 0.9], (1, 2): [0.9, 0.05, 0.05]}

    def step_fn(prefix):
        p = np.array(table.get(tuple(prefix), [0.9, 0.05, 0.05])) + 1e-12
        return np.log(p / p.sum())

    raw = beam(step_fn, 0, 3, 5, length_penalty=None)[0]
    equal(tuple(int(x) for x in raw[0]), (), label="the raw-score winner, log 0.5 = -0.693 against log 0.405 = -0.904")
    normed = beam(step_fn, 0, 3, 5, length_penalty=1.0)[0]
    equal(tuple(int(x) for x in normed[0]), (1, 2), label="the winner with length_penalty=1, -0.775 against -0.832")
    almost(float(normed[1]), math.log(0.5 * 0.9 * 0.9) / (7 / 6), 1e-6, label="the normalized score of (1, 2)")


@check("A tiny LAS trained with your attention and joint loss learns: the loss falls from about 23 to under 9")
def test_training(sub):
    train = has(sub, "train_las")
    torch.manual_seed(0)
    _, losses = train(60, 16, seed=0)
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-10:]))
    if not np.isfinite(last):
        fail(f"the loss became {last}; look for -inf energies on unmasked frames")
    if last > 9.0:
        fail(f"after 60 steps the joint loss averages {last:.2f} over the last 10 steps (it started at {first:.2f}); expected under 9")
