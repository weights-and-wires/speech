"""Checks for the transducers-and-streaming exercise."""

import itertools
import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


def hand_lattice():
    """The lesson's T = 2, target 'a' example, as log probabilities (1, 2, 2, 2)."""
    p = torch.tensor([[[0.4, 0.6], [0.7, 0.3]], [[0.5, 0.5], [0.9, 0.1]]], dtype=torch.float64)
    return p[None].log()


def random_lattice(seed, B, T, U, V):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, U + 1, V, generator=g, dtype=torch.float64).log_softmax(-1)


def brute_force_rnnt(lp, y):
    """log P(y | x) by enumerating every order of T blanks and U labels that ends in a blank."""
    T, U = lp.shape[0], len(y)
    total = -math.inf
    for label_slots in itertools.combinations(range(T + U - 1), U):
        t = u = 0
        score = 0.0
        for move in range(T + U):
            if move in label_slots:
                score += lp[t, u, y[u]]
                u += 1
            else:
                score += lp[t, u, 0]
                t += 1
        total = np.logaddexp(total, score)
    return total


@check("rnnt_log_alpha reproduces the hand-worked lattice: alpha = 1, 0.6, 0.4, 0.62")
def test_alpha_hand(sub):
    alpha_fn = has(sub, "rnnt_log_alpha")
    alpha = alpha_fn(hand_lattice(), torch.tensor([[1]]))
    shape(alpha, (1, 2, 2), label="alpha")
    close(alpha.exp().detach().numpy()[0], np.array([[1.0, 0.6], [0.4, 0.62]]), atol=1e-12, label="alpha(t, u), rows t = 0, 1")


@check("rnnt_loss gives -log 0.558 on the hand example and matches brute force on random lattices")
def test_loss_brute(sub):
    loss_fn = has(sub, "rnnt_loss")
    loss = loss_fn(hand_lattice(), torch.tensor([[1]]), torch.tensor([2]), torch.tensor([1]))
    almost(float(loss[0]), -math.log(0.558), 1e-12, label="the loss of the hand example")
    for seed, (T, U) in enumerate([(3, 2), (4, 2), (4, 1), (2, 3)]):
        lp = random_lattice(seed, 1, T, U, 3)
        y = [int(v) for v in torch.randint(1, 3, (U,), generator=torch.Generator().manual_seed(seed))]
        got = float(loss_fn(lp, torch.tensor([y]), torch.tensor([T]), torch.tensor([U]))[0])
        want = -brute_force_rnnt(lp[0].numpy(), y)
        almost(got, want, 1e-9, label=f"the loss for T={T}, U={U}, target {y}")


@check("rnnt_loss on a padded batch equals the loss of each utterance on its own")
def test_loss_batch(sub):
    loss_fn = has(sub, "rnnt_loss")
    lp = random_lattice(10, 3, 6, 3, 4)
    targets = torch.tensor([[1, 2, 3], [2, 2, 0], [3, 0, 0]])
    T_len, U_len = torch.tensor([6, 4, 5]), torch.tensor([3, 2, 1])
    batch = loss_fn(lp, targets, T_len, U_len)
    shape(batch, (3,), label="the batch of losses")
    for b in range(3):
        T, U = int(T_len[b]), int(U_len[b])
        alone = loss_fn(lp[b : b + 1, :T, : U + 1], targets[b : b + 1, :U], torch.tensor([T]), torch.tensor([U]))
        almost(float(batch[b]), float(alone[0]), 1e-10, label=f"utterance {b} in the batch against alone")


@check("Every path takes T blanks and U labels: the gradient of log P sums to T on blanks and U on labels")
def test_occupancy(sub):
    loss_fn = has(sub, "rnnt_loss")
    T, U = 5, 3
    lp = random_lattice(20, 1, T, U, 4).requires_grad_(True)
    target = torch.tensor([[2, 1, 3]])
    loss = loss_fn(lp, target, torch.tensor([T]), torch.tensor([U]))
    if not loss.requires_grad:
        fail("rnnt_loss returned a tensor with no gradient; build it from torch operations on log_probs")
    (grad,) = torch.autograd.grad(-loss.sum(), lp)
    blank_total = float(grad[0, :, :, 0].sum())
    label_total = float(grad[0].sum() - grad[0, :, :, 0].sum())
    almost(blank_total, float(T), 1e-8, label="the expected number of blank moves")
    almost(label_total, float(U), 1e-8, label="the expected number of label moves")


def scripted_model(script, V=4):
    """A joiner that emits the next token of script[t] until that frame's list is used up."""
    def predictor(token, state):
        return None, (0 if state is None else state + 1)

    def make_joiner():
        position = {"t": -1, "count": 0}

        def joiner(f_t, g):
            t = int(f_t[0])
            if t != position["t"]:
                position.update(t=t, count=0)
            wanted = script[t]
            out = torch.full((V,), -5.0)
            if position["count"] < len(wanted):
                out[wanted[position["count"]]] = 0.0
                position["count"] += 1
            else:
                out[0] = 0.0
            return out.log_softmax(-1)
        return joiner

    return predictor, make_joiner()


@check("greedy_transducer_decode emits zero, one, or several labels per frame, then moves on at a blank")
def test_greedy(sub):
    decode = has(sub, "greedy_transducer_decode")
    script = [[], [1], [2, 3], [], [1, 1]]
    predictor, joiner = scripted_model(script)
    enc = torch.arange(5, dtype=torch.float64)[:, None]
    got = decode(enc, predictor, joiner, blank=0, max_symbols=10)
    equal([int(k) for k in got], [1, 2, 3, 1, 1], label="the decoded tokens")


@check("greedy_transducer_decode stops at max_symbols labels per frame even if blank never wins")
def test_greedy_cap(sub):
    decode = has(sub, "greedy_transducer_decode")

    def predictor(token, state):
        return None, None

    def joiner(f_t, g):
        return torch.tensor([-3.0, -0.1, -4.0])  # label 1 always wins

    got = decode(torch.zeros(4, 1), predictor, joiner, blank=0, max_symbols=3)
    equal(len(got), 12, label="the number of tokens from 4 frames with max_symbols=3")


def brute_force_tdt(tok, dur, y, durations):
    T, U = tok.shape[0], len(y)
    total = -math.inf

    def walk(t, u, score):
        nonlocal total
        if t == T:
            if u == U:
                total = np.logaddexp(total, score)
            return
        for j, d in enumerate(durations):
            if t + d > T:
                continue
            if d >= 1:
                walk(t + d, u, score + tok[t, u, 0] + dur[t, u, j])
            if u < U:
                walk(t + d, u + 1, score + tok[t, u, y[u]] + dur[t, u, j])

    walk(0, 0, 0.0)
    return total


@check("tdt_log_likelihood gives 0.4181 on the hand example and matches brute-force path enumeration")
def test_tdt(sub):
    tdt = has(sub, "tdt_log_likelihood")
    tok = np.zeros((2, 2, 2))
    tok[0, 0], tok[0, 1], tok[1, 0], tok[1, 1] = [0.4, 0.6], [0.7, 0.3], [0.5, 0.5], [0.9, 0.1]
    dur = np.tile([0.2, 0.5, 0.3], (2, 2, 1))
    almost(math.exp(tdt(np.log(tok), np.log(dur), [1], [0, 1, 2])), 0.4181, 1e-10, label="P(a) for the hand example")
    rng = np.random.default_rng(5)
    for T, U, durations in [(4, 2, [0, 1, 2]), (5, 2, [0, 1, 2, 3]), (3, 3, [0, 1])]:
        tok = rng.normal(size=(T, U + 1, 3))
        tok -= np.log(np.exp(tok).sum(-1, keepdims=True))
        dur = rng.normal(size=(T, U + 1, len(durations)))
        dur -= np.log(np.exp(dur).sum(-1, keepdims=True))
        y = list(rng.integers(1, 3, size=U))
        almost(tdt(tok, dur, y, durations), brute_force_tdt(tok, dur, y, durations), 1e-9,
               label=f"log P for T={T}, U={U}, durations {durations}")


@check("tdt_greedy_decode jumps by the predicted duration and calls the joiner fewer times than there are frames")
def test_tdt_greedy(sub):
    decode = has(sub, "tdt_greedy_decode")
    durations = [0, 1, 2, 4]
    # At each frame: the token that wins and the duration index that wins.
    plan = {0: (2, 2), 2: (0, 3), 6: (3, 0), 7: (0, 0)}
    visited = []

    def predictor(token, state):
        return None, None

    def joiner(f_t, g):
        t = int(f_t[0])
        visited.append(t)
        k, j = plan.get(t, (0, 1))
        if t == 6 and visited.count(6) > 1:
            k, j = 1, 1  # the second look at frame 6 emits 1 and moves on
        tok = torch.full((4,), -4.0)
        tok[k] = 0.0
        dur = torch.full((4,), -4.0)
        dur[j] = 0.0
        return tok.log_softmax(-1), dur.log_softmax(-1)

    tokens, calls = decode(torch.arange(9, dtype=torch.float64)[:, None], predictor, joiner, durations, blank=0)
    equal([int(k) for k in tokens], [2, 3, 1], label="the decoded tokens")
    equal(visited, [0, 2, 6, 6, 7, 8], label="the frames the joiner saw (a blank with duration 0 must still move 1)")
    equal(int(calls), 6, label="the number of joiner calls for 9 frames")


@check("chunked_attention_mask lets each frame see its own chunk and the left chunks only")
def test_mask(sub):
    mask_fn = has(sub, "chunked_attention_mask")
    m = mask_fn(6, 2)
    want = np.array([[1, 1, 0, 0, 0, 0]] * 2 + [[1, 1, 1, 1, 0, 0]] * 2 + [[1] * 6] * 2, dtype=bool)
    equal(np.asarray(m, dtype=bool).tolist(), want.tolist(), label="the mask for T=6, chunk 2, all left context")
    m = np.asarray(mask_fn(7, 2, left_chunks=1), dtype=bool)
    ref = np.array([[(j // 2 <= i // 2) and (j // 2 >= i // 2 - 1) for j in range(7)] for i in range(7)])
    equal(m.tolist(), ref.tolist(), label="the mask for T=7, chunk 2, one left chunk")


@check("With the chunked mask, changing future chunks leaves earlier frames' attention output unchanged")
def test_streaming(sub):
    mask_fn = has(sub, "chunked_attention_mask")
    torch.manual_seed(0)
    attn = torch.nn.MultiheadAttention(8, 2, batch_first=True).double().eval()
    x = torch.randn(1, 12, 8, dtype=torch.float64)
    allowed = torch.as_tensor(mask_fn(12, 4)).bool()
    y1, _ = attn(x, x, x, attn_mask=~allowed)
    x2 = x.clone()
    x2[:, 8:] += torch.randn(1, 4, 8, dtype=torch.float64)
    y2, _ = attn(x2, x2, x2, attn_mask=~allowed)
    close(y2[0, :8].detach().numpy(), y1[0, :8].detach().numpy(), atol=1e-12, label="outputs of frames 0 to 7 after changing frames 8 to 11")
    if torch.allclose(y2[0, 8:], y1[0, 8:]):
        fail("the last chunk's outputs didn't change when its own frames did; the mask blocks too much")
