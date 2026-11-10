"""Checks for the spoken language models exercise."""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechcheck import almost, check, close, equal, fail, has, shape


def blobs(rng, n_per=60, k=4, d=3, spread=0.3):
    centers = rng.normal(size=(k, d)) * 4.0
    x = np.concatenate([c + spread * rng.normal(size=(n_per, d)) for c in centers])
    return x, centers


@check("assign_units picks the nearest centroid for every frame")
def test_assign(sub):
    assign = has(sub, "assign_units")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 5))
    c = rng.normal(size=(7, 5))
    got = np.asarray(assign(x, c))
    shape(got, (50,), label="units")
    want = np.array([np.argmin([np.sum((f - m) ** 2) for m in c]) for f in x])
    equal(got, want, label="units")


@check("kmeans never raises the inertia, and matches Lloyd's algorithm from the same start")
def test_kmeans(sub):
    kmeans = has(sub, "kmeans")
    rng = np.random.default_rng(1)
    x, _ = blobs(rng)
    init = x[rng.choice(len(x), 4, replace=False)]
    init_copy = init.copy()
    centroids, inertias = kmeans(x, init, 10)
    inertias = np.asarray(inertias, dtype=float)
    shape(inertias, (10,), label="inertias")
    if np.any(np.diff(inertias) > 1e-9):
        fail(f"the inertia rose between iterations: {np.round(inertias, 3)}")
    equal(init, init_copy, label="the init array (kmeans must not modify it)")
    ref = init.astype(float).copy()
    ref_inertia = []
    for _ in range(10):
        d = ((x[:, None, :] - ref[None]) ** 2).sum(-1)
        u = d.argmin(1)
        ref_inertia.append(d[np.arange(len(x)), u].sum())
        for k in range(len(ref)):
            if np.any(u == k):
                ref[k] = x[u == k].mean(0)
    close(inertias, np.array(ref_inertia), rtol=1e-8, label="inertia per iteration")
    close(centroids, ref, rtol=1e-8, atol=1e-10, label="final centroids")


@check("deduplicate turns 5 5 5 12 12 7 into 5 12 7 with durations 3 2 1")
def test_dedup(sub):
    dedup = has(sub, "deduplicate")
    units, dur = dedup(np.array([5, 5, 5, 12, 12, 7]))
    equal(np.asarray(units), np.array([5, 12, 7]), label="deduplicated units")
    equal(np.asarray(dur), np.array([3, 2, 1]), label="durations")
    rng = np.random.default_rng(2)
    seq = rng.integers(0, 4, size=200)
    units, dur = dedup(seq)
    equal(np.repeat(np.asarray(units), np.asarray(dur)), seq, label="np.repeat(units, durations)")
    if np.any(np.asarray(units)[1:] == np.asarray(units)[:-1]):
        fail("two neighbors in the deduplicated sequence are still equal")


@check("unit_bigram_lm gives add-alpha probabilities whose rows sum to one")
def test_bigram(sub):
    lm = has(sub, "unit_bigram_lm")
    logp = np.asarray(lm([np.array([0, 1, 1, 2]), np.array([2, 0])], 3, alpha=1.0))
    shape(logp, (3, 3), label="log_probs")
    close(np.exp(logp).sum(axis=1), np.ones(3), rtol=1e-10, label="row sums of exp(log_probs)")
    # Pairs: (0,1) (1,1) (1,2) (2,0). Row 1 counts are [0, 1, 1] + 1 = [1, 2, 2] over 5.
    close(np.exp(logp[1]), np.array([1, 2, 2]) / 5, rtol=1e-10, label="P(next | previous = 1)")
    close(np.exp(logp[0]), np.array([1, 2, 1]) / 4, rtol=1e-10, label="P(next | previous = 0)")
    seq_lp = has(sub, "sequence_log_prob")
    almost(seq_lp(np.array([0, 1, 2]), logp), np.log(2 / 4) + np.log(2 / 5), 1e-10,
           label="sequence_log_prob([0, 1, 2])")


@check("Units, deduplication, and a unit LM prefer real 'words' to shuffled ones (a toy sWUGGY)")
def test_toy_swuggy(sub):
    assign, kmeans = has(sub, "assign_units"), has(sub, "kmeans")
    dedup, lm, seq_lp = has(sub, "deduplicate"), has(sub, "unit_bigram_lm"), has(sub, "sequence_log_prob")
    rng = np.random.default_rng(3)
    n_phones = 12
    phones = rng.normal(size=(n_phones, 4)) * 3.0
    lexicon = []
    while len(lexicon) < 5:
        word = rng.choice(n_phones, size=6, replace=False)
        lexicon.append(word)

    def speak(word):
        frames = np.repeat(word, rng.integers(2, 6, size=len(word)))
        return phones[frames] + 0.4 * rng.normal(size=(len(frames), 4)), frames

    spoken = [speak(lexicon[i % len(lexicon)]) for i in range(60)]
    feats = np.concatenate([f for f, _ in spoken])
    labels = np.concatenate([p for _, p in spoken])
    # Seed k-means with one frame of each phone, so the check tests your code, not the luck of the init.
    init = feats[[int(np.flatnonzero(labels == p)[0]) for p in range(n_phones)]]
    centroids, _ = kmeans(feats, init, 15)
    to_units = lambda f: np.asarray(dedup(assign(f, centroids))[0])
    logp = lm([to_units(f) for f, _ in spoken], n_phones, alpha=0.1)
    wins = 0
    trials = 40
    for i in range(trials):
        word = lexicon[i % len(lexicon)]
        real = to_units(speak(word)[0])
        fake = to_units(speak(rng.permutation(word))[0])
        wins += seq_lp(real, logp) / max(len(real) - 1, 1) > seq_lp(fake, logp) / max(len(fake) - 1, 1)
    if wins < 0.75 * trials:
        fail(f"the unit LM preferred the real word in {wins} of {trials} pairs, expected at least {int(0.75 * trials)}")


@check("stack_frames concatenates k neighbors and drops the remainder")
def test_stack(sub):
    stack = has(sub, "stack_frames")
    h = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    got = stack(h, 4)
    shape(got, (2, 2, 12), label="stack_frames(h, 4) for T = 10, d = 3")
    close(got[1, 1].numpy(), torch.cat([h[1, 4], h[1, 5], h[1, 6], h[1, 7]]).numpy(), label="batch 1, stacked frame 1")
    shape(stack(h, 5), (2, 2, 15), label="stack_frames(h, 5)")


@check("Projector maps each stacked frame through Linear, GELU, Linear, one frame at a time")
def test_projector(sub):
    Projector = has(sub, "Projector")
    torch.manual_seed(0)
    proj = Projector(12, 7, 16)
    n = sum(p.numel() for p in proj.parameters())
    want_n = 12 * 16 + 16 + 16 * 7 + 7
    if n != want_n:
        fail(f"the projector has {n} parameters, expected {want_n} for Linear(12, 16) and Linear(16, 7)")
    x = torch.randn(3, 5, 12)
    y = proj(x).detach()
    shape(y, (3, 5, 7), label="projector output")
    x2 = x.clone()
    x2[:, 2] += 1.0
    y2 = proj(x2).detach()
    changed = (y2 - y).abs().amax(dim=(0, 2)) > 1e-6
    if changed.tolist() != [False, False, True, False, False]:
        fail(f"changing stacked frame 2 changed outputs {torch.where(changed)[0].tolist()}, expected only [2]")
    linears = [m for m in proj.modules() if isinstance(m, nn.Linear)]
    if len(linears) == 2:
        a, b = linears
        want = b(F.gelu(a(x)))
        close(y.detach().numpy(), want.detach().numpy(), rtol=1e-5, atol=1e-6, label="W2 gelu(W1 x + b1) + b2")


@check("build_adapter_inputs lays out [audio | prompt | target] and masks audio and prompt labels")
def test_build(sub):
    build = has(sub, "build_adapter_inputs")
    torch.manual_seed(1)
    embed = nn.Embedding(20, 6)
    audio = torch.randn(2, 4, 6)
    prompt = torch.tensor([3, 4, 5])
    target = torch.tensor([[7, 0], [9, 0]])
    inputs, labels = build(audio, prompt, target, embed)
    inputs = inputs.detach()
    shape(inputs, (2, 9, 6), label="inputs_embeds")
    shape(labels, (2, 9), label="labels")
    close(inputs[:, :4].detach().numpy(), audio.numpy(), label="audio positions")
    close(inputs[1, 4:7].detach().numpy(), embed.weight[prompt].detach().numpy(), label="prompt positions")
    close(inputs[0, 7:].detach().numpy(), embed.weight[target[0]].detach().numpy(), label="target positions")
    equal(labels.tolist(), [[-100] * 7 + [7, 0], [-100] * 7 + [9, 0]], label="labels")


@check("next_token_loss averages -log p(label) over labeled positions, and masked logits get no gradient")
def test_loss(sub):
    loss_fn = has(sub, "next_token_loss")
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 3, 1, 0], [-100, -100, -100, -100, 2, 4]])
    loss = loss_fn(logits, labels)
    logp = torch.log_softmax(logits.detach(), -1)
    terms = [-logp[b, t - 1, labels[b, t]] for b in range(2) for t in range(6) if labels[b, t] != -100]
    almost(loss.item(), float(sum(terms) / len(terms)), 1e-6, label="loss")
    loss.backward()
    g = logits.grad.abs().sum(-1)
    predicts_label = torch.zeros(2, 6, dtype=torch.bool)
    predicts_label[:, :-1] = labels[:, 1:] != -100
    if torch.any(g[~predicts_label] > 1e-12):
        fail(f"positions whose next label is masked received gradient: {g[~predicts_label].tolist()}")
    if torch.any(g[predicts_label] < 1e-9):
        fail("a position that predicts a target token received no gradient")


class ToyLM(nn.Module):
    """A frozen one-layer causal attention LM with a tied output head."""

    def __init__(self, vocab=12, d=32):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.5)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        n = x.shape[1]
        q, k, v = self.qkv(x).chunk(3, -1)
        a = (q @ k.transpose(1, 2)) / x.shape[-1] ** 0.5
        a = a.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        return self.norm(x + self.out(a @ v)) @ self.embed.weight.T


@check("The whole adapter: a projector trained through a frozen toy LM learns to 'transcribe'")
def test_adapter_end_to_end(sub):
    stack, Projector = has(sub, "stack_frames"), has(sub, "Projector")
    build, loss_fn = has(sub, "build_adapter_inputs"), has(sub, "next_token_loss")
    lm = ToyLM()
    means = torch.randn(10, 6, generator=torch.Generator().manual_seed(123))
    prompt, eos = torch.tensor([10]), 11

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(10, (n,), generator=g)
        return means[y][:, None, :].expand(n, 12, 6) + 0.7 * torch.randn(n, 12, 6, generator=g), y

    torch.manual_seed(1)
    proj = Projector(12, 32, 64)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    for step in range(800):
        x, y = data(32, 1000 + step)
        target = torch.stack([y, torch.full_like(y, eos)], 1)
        inputs, labels = build(proj(stack(x, 2)), prompt, target, lm.embed)
        loss = loss_fn(lm(inputs), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = data(500, 7)
    with torch.no_grad():
        z = proj(stack(x, 2))
        pred = lm(torch.cat([z, lm.embed(prompt)[None].expand(500, -1, -1)], 1))[:, -1].argmax(-1)
    acc = (pred == y).float().mean().item()
    if acc < 0.7:
        fail(f"after 800 steps the toy LM wrote the right class for {acc:.0%} of held-out inputs, expected at least 70%")
