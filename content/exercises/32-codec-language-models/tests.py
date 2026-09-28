"""Checks for the codec language models exercise."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from speechcheck import almost, check, close, equal, fail, shape

P = -1  # the pad value the delay checks use


@check("apply_delay turns the lesson's 3-by-4 grid into the 3-by-6 delayed grid")
def test_delay_hand(sub):
    codes = np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
    got = sub.apply_delay(codes, P)
    want = np.array([
        [11, 12, 13, 14, P, P],
        [P, 21, 22, 23, 24, P],
        [P, P, 31, 32, 33, 34],
    ])
    shape(got, (3, 6), label="delayed grid, (K, T + K - 1)")
    equal(np.asarray(got).tolist(), want.tolist(), label="delayed grid")


@check("undo_delay inverts apply_delay for any grid size")
def test_delay_roundtrip(sub):
    rng = np.random.default_rng(0)
    for K, T in [(1, 5), (2, 7), (4, 3), (8, 20)]:
        codes = rng.integers(0, 1024, size=(K, T))
        delayed = sub.apply_delay(codes, 1024)
        shape(delayed, (K, T + K - 1), label=f"delayed grid for K={K}, T={T}")
        equal(np.asarray(sub.undo_delay(delayed)).tolist(), codes.tolist(), label=f"round trip for K={K}, T={T}")


@check("In the delayed grid, codebook k of frame t comes after codebooks below k of the same frame")
def test_delay_dependency(sub):
    K, T = 4, 6
    codes = 100 * np.arange(K)[:, None] + np.arange(T)[None, :]
    delayed = np.asarray(sub.apply_delay(codes, P))
    step = {int(v): s for k in range(K) for s, v in enumerate(delayed[k]) if v != P}
    for k in range(1, K):
        for t in range(T):
            if not step[100 * (k - 1) + t] < step[100 * k + t]:
                fail(f"codebook {k} of frame {t} sits at step {step[100 * k + t]}, not after codebook {k - 1} "
                     f"of the same frame at step {step[100 * (k - 1) + t]}")
            if t and not step[100 * k + t - 1] < step[100 * k + t]:
                fail(f"codebook {k} of frame {t} comes before frame {t - 1} of the same codebook")


@check("valle_ar_example predicts only the target codes and the end token")
def test_valle_ar(sub):
    phonemes = np.array([5, 6, 7])
    prompt = np.array([100, 101])
    target = np.array([200, 201, 202, 203])
    inputs, targets, mask = sub.valle_ar_example(phonemes, prompt, target, 999)
    seq = [5, 6, 7, 100, 101, 200, 201, 202, 203, 999]
    equal(np.asarray(inputs).tolist(), seq[:-1], label="inputs")
    equal(np.asarray(targets).tolist(), seq[1:], label="targets")
    equal(np.asarray(targets)[np.asarray(mask)].tolist(), [200, 201, 202, 203, 999], label="the targets the loss counts")


@check("nar_input sums the embeddings of every codebook below the stage")
def test_nar_input(sub):
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(8, 10))
    emb = rng.normal(size=(8, 16, 5))
    for stage in (1, 3, 7):
        want = sum(emb[k][codes[k]] for k in range(stage))
        close(sub.nar_input(codes, emb, stage), want, label=f"input for stage {stage}")
    close(sub.nar_input(codes, emb, 1), emb[0][codes[0]], label="stage 1 sees codebook 0 only")


@check("maskgit_schedule follows the cosine: 16 tokens over 4 iterations leave 14, 11, 6, then 0 masked")
def test_schedule(sub):
    equal(list(sub.maskgit_schedule(16, 4)), [14, 11, 6, 0], label="masked after each iteration")
    for n, it in [(100, 8), (1000, 12), (50, 50)]:
        s = list(sub.maskgit_schedule(n, it))
        equal(len(s), it, label=f"one entry per iteration for N={n}, I={it}")
        if s[-1] != 0 or any(b >= a for a, b in zip([n] + s[:-1], s)):
            fail(f"for N={n}, I={it} the schedule {s[:6]}... must fall every iteration and end at 0")


@check("maskgit_decode reveals the most confident guesses first and never changes a revealed token")
def test_maskgit_decode(sub):
    n, V, mask_id = 12, 5, 5
    rng = np.random.default_rng(2)
    answer = rng.integers(0, V, size=n)
    sharpness = np.linspace(1.0, 6.0, n)[rng.permutation(n)]   # a different confidence per position
    seen = []

    def oracle(tokens):
        seen.append(tokens.copy())
        logits = np.zeros((n, V))
        logits[np.arange(n), answer] = sharpness
        return logits

    tokens, history = sub.maskgit_decode(oracle, n, 4, mask_id)
    equal(list(history), list(sub.maskgit_schedule(n, 4)), label="tokens still masked after each iteration")
    equal(np.asarray(tokens).tolist(), answer.tolist(), label="the decoded tokens")
    first = np.flatnonzero(seen[1] != mask_id)
    want = np.argsort(-sharpness)[: n - sub.maskgit_schedule(n, 4)[0]]
    equal(sorted(first.tolist()), sorted(want.tolist()), label="positions revealed in the first iteration (the most confident)")
    for a, b in zip(seen, seen[1:]):
        done = a != mask_id
        if np.any(a[done] != b[done]):
            fail("a revealed token changed in a later iteration")


@check("markov_code_entropy matches a brute-force conditional entropy of the source")
def test_entropy(sub):
    rng = np.random.default_rng(3)
    V = 5
    A = rng.dirichlet(np.full(V, 0.5), size=V)
    B = rng.dirichlet(np.full(V, 0.5), size=V)
    w, v = np.linalg.eig(A.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    # Joint of (c0 at t-1, c0 at t, c1 at t).
    joint = pi[:, None, None] * A[:, :, None] * B[None, :, :]
    h_c0 = -np.sum(joint.sum(2) * np.log(A))
    h_c1_given_c0 = -np.sum(joint * np.log(B[None]))
    p_prev_c1 = joint.sum(1)
    h_c1_given_prev = -np.sum(p_prev_c1 * np.log(p_prev_c1 / pi[:, None]))
    delay, parallel = sub.markov_code_entropy(A, B)
    almost(delay, h_c0 + h_c1_given_c0, 1e-8, label="delay floor, nats per frame")
    almost(parallel, h_c0 + h_c1_given_prev, 1e-8, label="parallel floor, nats per frame")


@check("lm_inputs_targets shifts by one step, and masked_cross_entropy ignores the pads")
def test_inputs_and_loss(sub):
    grid = torch.tensor([[[3, 4, 5, 9], [9, 6, 7, 8]]])   # pad = 9
    inputs, targets, mask = sub.lm_inputs_targets(grid, 9, 10)
    equal(inputs.tolist(), [[[10, 3, 4, 5], [10, 9, 6, 7]]], label="inputs, with a BOS column of 10")
    equal(targets.tolist(), grid.tolist(), label="targets")
    equal(mask.tolist(), [[[True, True, True, False], [False, True, True, True]]], label="mask")
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 4, 9)
    got = sub.masked_cross_entropy(logits, torch.clamp(targets, max=8), mask)
    flat = logits[mask]
    want = F.cross_entropy(flat, torch.clamp(targets, max=8)[mask])
    almost(float(got), float(want), 1e-6, label="mean cross-entropy over the six real tokens")


@check("TinyCodecLM trained on delayed codes reaches the delay floor; on parallel codes it can't")
def test_training(sub):
    V, T = 8, 32
    A = np.zeros((V, V))
    for i in range(V):
        A[i, (i + 1) % V] += 0.5
        A[i, (i - 1) % V] += 0.3
        A[i, i] += 0.2
    B = np.full((V, V), 0.2 / (V - 1))
    for j in range(V):
        B[j, (2 * j) % V] = 0.8
    rng = np.random.default_rng(4)
    codes = np.zeros((256, 2, T), dtype=np.int64)
    codes[:, 0, 0] = rng.integers(V, size=256)
    for t in range(T):
        if t:
            for n in range(256):
                codes[n, 0, t] = rng.choice(V, p=A[codes[n, 0, t - 1]])
        for n in range(256):
            codes[n, 1, t] = rng.choice(V, p=B[codes[n, 0, t]])
    floor_delay, floor_parallel = sub.markov_code_entropy(A, B)

    def train(grid):
        torch.manual_seed(0)
        model = sub.TinyCodecLM(2, V)
        logits = model(torch.full((1, 2, 3), V + 1))
        shape(logits.detach(), (1, 2, 3, V), label="TinyCodecLM output, (N, K, S, V)")
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(0)
        losses = []
        for _ in range(400):
            batch = grid[torch.randint(0, len(grid), (32,), generator=gen)]
            inputs, targets, mask = sub.lm_inputs_targets(batch, V, V + 1)
            loss = sub.masked_cross_entropy(model(inputs), targets, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(2 * loss.item())
        return float(np.mean(losses[-50:]))

    delayed = torch.from_numpy(np.stack([np.asarray(sub.apply_delay(c, V)) for c in codes]))
    got_delay = train(delayed)
    got_parallel = train(torch.from_numpy(codes))
    if abs(got_delay - floor_delay) > 0.08:
        fail(f"with the delay pattern the loss settled at {got_delay:.3f} nats per frame, expected {floor_delay:.3f} +- 0.08")
    if got_parallel < floor_parallel - 0.08:
        fail(f"with the parallel pattern the loss reached {got_parallel:.3f}, below its floor {floor_parallel:.3f}; "
             "the model must see only the previous step")
    if got_parallel - got_delay < 0.4:
        fail(f"the parallel pattern ({got_parallel:.3f}) should cost about {floor_parallel - floor_delay:.2f} nats "
             f"per frame more than the delay pattern ({got_delay:.3f})")
