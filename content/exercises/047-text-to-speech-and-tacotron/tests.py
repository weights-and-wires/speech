"""Checks for the Tacotron attention and alignment-loss exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, fail, has, shape


def _params(rng, q_dim=5, m_dim=4, att=6, filters=3, kernel=5):
    def t(*size, scale=0.5):
        return torch.tensor(rng.normal(size=size) * scale)

    return {
        "W_query": t(att, q_dim),
        "W_memory": t(att, m_dim),
        "location_conv": t(filters, 1, kernel),
        "W_location": t(att, filters),
        "v": t(att),
        "bias": t(att, scale=0.1),
    }


def _reference_attention(query, memory, cumulative, mask, p):
    """A loop-by-loop numpy version of location-sensitive attention."""
    q, m, c = query.numpy(), memory.numpy(), cumulative.numpy()
    kernel = p["location_conv"].numpy()
    filters, _, k = kernel.shape
    half = (k - 1) // 2
    batch, n_text, _ = m.shape
    weights = np.zeros((batch, n_text))
    contexts = np.zeros((batch, m.shape[2]))
    for b in range(batch):
        padded = np.concatenate([np.zeros(half), c[b], np.zeros(half)])
        energies = np.full(n_text, -np.inf)
        for j in range(n_text):
            if not mask[b, j]:
                continue
            f = np.array([np.dot(kernel[i, 0], padded[j : j + k]) for i in range(filters)])
            hidden = (p["W_query"].numpy() @ q[b] + p["W_memory"].numpy() @ m[b, j]
                      + p["W_location"].numpy() @ f + p["bias"].numpy())
            energies[j] = p["v"].numpy() @ np.tanh(hidden)
        e = np.exp(energies - energies.max())
        weights[b] = e / e.sum()
        contexts[b] = weights[b] @ m[b]
    return weights, contexts


@check("guided_attention_weights is zero on the diagonal and near one in the far corner")
def test_guide_weights(sub):
    fn = has(sub, "guided_attention_weights")
    w = fn(8, 4, 0.2)
    shape(w, (8, 4), label="W for 8 frames and 4 symbols")
    almost(w[4, 2], 0.0, 1e-12, label="W at frame 4, symbol 2, where t/T = n/N")
    almost(w[7, 0], 1 - math.exp(-(0.875**2) / 0.08), 1e-9, label="W at frame 7, symbol 0")
    t = np.arange(8)[:, None] / 8
    n = np.arange(4)[None, :] / 4
    close(w, 1 - np.exp(-((n - t) ** 2) / 0.08), atol=1e-9, label="the whole weight grid")


@check("guided_attention_loss scores a diagonal alignment far below an anti-diagonal one")
def test_guide_loss_order(sub):
    fn = has(sub, "guided_attention_loss")
    frames, texts = 12, 4
    diag = torch.zeros(1, frames, texts, dtype=torch.float64)
    anti = torch.zeros(1, frames, texts, dtype=torch.float64)
    for t in range(frames):
        diag[0, t, t * texts // frames] = 1.0
        anti[0, t, texts - 1 - t * texts // frames] = 1.0
    lengths_t, lengths_n = torch.tensor([frames]), torch.tensor([texts])
    good = float(fn(diag, lengths_t, lengths_n))
    bad = float(fn(anti, lengths_t, lengths_n))
    if not good < 0.2 * bad:
        fail(f"a diagonal alignment scored {good:.4f} and an anti-diagonal one {bad:.4f}; "
             "the diagonal should cost less than a fifth as much")


@check("guided_attention_loss uses each utterance's own lengths and ignores padding")
def test_guide_loss_batched(sub):
    fn = has(sub, "guided_attention_loss")
    rng = np.random.default_rng(1)
    frame_lengths = torch.tensor([9, 6])
    text_lengths = torch.tensor([3, 5])
    att = torch.tensor(rng.random((2, 9, 5)))
    total, count = 0.0, 0
    for b in range(2):
        t_b, n_b = int(frame_lengths[b]), int(text_lengths[b])
        t = np.arange(t_b)[:, None] / t_b
        n = np.arange(n_b)[None, :] / n_b
        w = 1 - np.exp(-((n - t) ** 2) / (2 * 0.2**2))
        total += float((att[b, :t_b, :n_b].numpy() * w).sum())
        count += t_b * n_b
    got = fn(att, frame_lengths, text_lengths)
    almost(got, total / count, 1e-9, label="the loss over the valid cells of both utterances")
    att2 = att.clone()
    att2[0, :, 3:] = 100.0
    att2[1, 6:, :] = 100.0
    almost(fn(att2, frame_lengths, text_lengths), total / count, 1e-9,
           label="the loss after changing only padded cells")


@check("stop_token_loss is binary cross-entropy with the single positive at the last real frame")
def test_stop_loss(sub):
    fn = has(sub, "stop_token_loss")
    rng = np.random.default_rng(2)
    lengths = torch.tensor([5, 3])
    logits = torch.tensor(rng.normal(size=(2, 6)))
    pw = 4.0
    total, count = 0.0, 0
    for b in range(2):
        for n in range(int(lengths[b])):
            y = 1.0 if n == int(lengths[b]) - 1 else 0.0
            p = 1 / (1 + math.exp(-float(logits[b, n])))
            total += -(pw * y * math.log(p) + (1 - y) * math.log(1 - p))
            count += 1
    almost(fn(logits, lengths, pos_weight=pw), total / count, 1e-9, label="the stop-token loss")
    confident = torch.full((2, 6), -30.0, dtype=torch.float64)
    confident[0, 4] = 30.0
    confident[1, 2] = 30.0
    confident[1, 3:] = 30.0  # padding: must not count
    almost(fn(confident, lengths), 0.0, 1e-9,
           label="the loss when every real frame is predicted confidently and correctly")


@check("location_sensitive_attention returns weights that sum to one and skip padding")
def test_attention_shapes(sub):
    fn = has(sub, "location_sensitive_attention")
    rng = np.random.default_rng(3)
    p = _params(rng)
    query = torch.tensor(rng.normal(size=(2, 5)))
    memory = torch.tensor(rng.normal(size=(2, 7, 4)))
    cumulative = torch.tensor(rng.random((2, 7)))
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    weights, context = fn(query, memory, cumulative, mask, p)
    shape(weights, (2, 7), label="weights")
    shape(context, (2, 4), label="context")
    close(weights.sum(dim=1), [1.0, 1.0], atol=1e-9, label="the sum of each row of weights")
    close(weights[1, 4:], [0.0, 0.0, 0.0], atol=0.0, label="the weights on padded symbols")


@check("location_sensitive_attention matches a loop-by-loop reference")
def test_attention_reference(sub):
    fn = has(sub, "location_sensitive_attention")
    rng = np.random.default_rng(4)
    p = _params(rng)
    query = torch.tensor(rng.normal(size=(3, 5)))
    memory = torch.tensor(rng.normal(size=(3, 6, 4)))
    cumulative = torch.tensor(rng.random((3, 6)))
    mask = torch.tensor([[True] * 6, [True] * 5 + [False], [True] * 3 + [False] * 3])
    weights, context = fn(query, memory, cumulative, mask, p)
    want_w, want_c = _reference_attention(query, memory, cumulative, mask, p)
    close(weights, want_w, atol=1e-9, label="attention weights")
    close(context, want_c, atol=1e-9, label="context vectors")


@check("The location term alone moves attention to the first uncovered symbol")
def test_location_moves_forward(sub):
    fn = has(sub, "location_sensitive_attention")
    p = {
        "W_query": torch.zeros(1, 2, dtype=torch.float64),
        "W_memory": torch.zeros(1, 3, dtype=torch.float64),
        # f_j = cumulative[j - 1] - cumulative[j]: positive only just past the covered symbols.
        "location_conv": torch.tensor([[[1.0, -1.0, 0.0]]], dtype=torch.float64),
        "W_location": torch.tensor([[8.0]], dtype=torch.float64),
        "v": torch.tensor([4.0], dtype=torch.float64),
        "bias": torch.zeros(1, dtype=torch.float64),
    }
    memory = torch.tensor(np.random.default_rng(5).normal(size=(1, 6, 3)))
    cumulative = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    mask = torch.ones(1, 6, dtype=torch.bool)
    weights, _ = fn(torch.zeros(1, 2, dtype=torch.float64), memory, cumulative, mask, p)
    e = 4 * np.tanh(8 * np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    want = np.exp(e) / np.exp(e).sum()
    close(weights[0], want, atol=1e-9,
          label="weights when only the location term acts (is the convolution padded by "
                "(kernel - 1) // 2 on both sides?)")
    if int(torch.argmax(weights[0])) != 3:
        fail(f"symbols 0 to 2 are covered, so attention should peak at symbol 3; it peaked at "
             f"{int(torch.argmax(weights[0]))}")


@check("diagonality is 1 for a diagonal alignment and low for an anti-diagonal one")
def test_diagonality(sub):
    fn = has(sub, "diagonality")
    frames, texts = 24, 6
    diag = torch.zeros(frames, texts, dtype=torch.float64)
    anti = torch.zeros(frames, texts, dtype=torch.float64)
    for t in range(frames):
        diag[t, t * texts // frames] = 1.0
        anti[t, texts - 1 - t * texts // frames] = 1.0
    almost(fn(diag, 0.15), 1.0, 1e-9, label="the diagonality of a diagonal alignment")
    almost(fn(anti, 0.15), 0.125, 1e-9, label="the diagonality of an anti-diagonal alignment")
    uniform = torch.full((frames, texts), 1.0 / texts, dtype=torch.float64)
    t = np.arange(frames)[:, None] / frames
    n = np.arange(texts)[None, :] / texts
    almost(fn(uniform, 0.15), float((np.abs(n - t) <= 0.15).mean()), 1e-9,
           label="the diagonality of a uniform alignment")
