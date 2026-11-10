"""Checks for the self-supervised speech exercise."""

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape

KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@check("conv_output_length gives 49 frames for one second, layer by layer")
def test_conv_output_length(sub):
    got = has(sub, "conv_output_length")
    want = [16000, 3199, 1599, 799, 399, 199, 99, 49]
    for i, w in enumerate(want):
        n = got(16000, KERNELS[:i], STRIDES[:i])
        if n != w:
            fail(f"after {i} conv layers, 16000 samples should give {w} frames, got {n}")
    for n_samples in (400, 12345, 48000):
        x = torch.zeros(1, 1, n_samples)
        for k, s in zip(KERNELS, STRIDES):
            x = torch.nn.functional.conv1d(x, torch.zeros(1, 1, k), stride=s)
        equal(got(n_samples, KERNELS, STRIDES), x.shape[-1], label=f"frames for {n_samples} samples")
    equal(got(399, KERNELS, STRIDES), 0, label="frames for 399 samples, too short for one frame")


@check("receptive_field is 400 samples with a hop of 320 for the wav2vec 2.0 encoder")
def test_receptive_field(sub):
    got = has(sub, "receptive_field")
    field, hop = got(KERNELS, STRIDES)
    equal(int(field), 400, label="receptive field in samples")
    equal(int(hop), 320, label="hop in samples")
    # Brute force: which input samples move output frame 0 of a stack of all-ones kernels?
    x = torch.zeros(1, 1, 2000, requires_grad=True)
    y = x
    for k, s in zip([4, 3, 5], [2, 3, 1]):
        y = torch.nn.functional.conv1d(y, torch.ones(1, 1, k), stride=s)
    y[0, 0, 0].backward()
    seen = int((x.grad[0, 0] != 0).sum())
    field, hop = got([4, 3, 5], [2, 3, 1])
    equal(int(field), seen, label="receptive field of kernels (4, 3, 5) with strides (2, 3, 1)")
    equal(int(hop), 6, label="hop of strides (2, 3, 1)")


@check("span_mask expands starts into overlapping spans and masks about 49% at p = 0.065, M = 10")
def test_span_mask(sub):
    got = has(sub, "span_mask")
    starts = np.zeros(10, dtype=bool)
    starts[[0, 2, 8]] = True
    mask = got(starts, 3)
    shape(mask, (10,), label="mask")
    equal(np.asarray(mask, dtype=bool), np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 1], dtype=bool),
          label="mask for starts at 0, 2, and 8 with span 3")
    rng = np.random.default_rng(0)
    starts = rng.random(200000) < 0.065
    fraction = float(np.mean(got(starts, 10)))
    want = 1 - (1 - 0.065) ** 10
    almost(fraction, want, 0.01, label="masked fraction, expected 1 - (1 - p)^M")


@check("sample_distractors draws K other masked frames for every masked frame")
def test_sample_distractors(sub):
    got = has(sub, "sample_distractors")
    masked = np.array([2, 3, 4, 9, 10, 11, 12, 30])
    d = np.asarray(got(masked, 100, np.random.default_rng(0)))
    shape(d, (8, 100), label="distractors")
    for i, t in enumerate(masked):
        if np.any(d[i] == t):
            fail(f"row {i} (frame {t}) contains its own frame as a distractor")
        if not np.all(np.isin(d[i], masked)):
            bad = sorted(set(d[i].tolist()) - set(masked.tolist()))
            fail(f"row {i} contains frames {bad}, which are not masked")
    big = np.asarray(got(masked, 7000, np.random.default_rng(1)))[0]
    freq = np.array([np.mean(big == t) for t in masked[1:]])
    if np.any(np.abs(freq - 1 / 7) > 0.02):
        fail(f"frame 2's distractors should be spread evenly over the 7 other masked frames "
             f"(1/7 each), got frequencies {np.round(freq, 3)}")


def reference_contrastive(c, q, masked, distractors, kappa):
    c = torch.tensor(c[masked])
    cand = torch.cat([torch.tensor(q[masked])[:, None], torch.tensor(q[distractors])], dim=1)
    logits = torch.nn.functional.cosine_similarity(c[:, None], cand, dim=-1) / kappa
    return float(torch.nn.functional.cross_entropy(logits, torch.zeros(len(masked), dtype=torch.long)))


@check("contrastive_loss matches a torch cross-entropy over cosine logits")
def test_contrastive_loss(sub):
    got = has(sub, "contrastive_loss")
    # By hand: c points at its own target, and both distractors are orthogonal.
    c = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    loss = got(c, c.copy(), np.array([0]), np.array([[1, 2]]), 0.1)
    almost(float(loss), float(np.log(1 + 2 * np.exp(-10.0))), 1e-9,
           label="loss with logits (10, 0, 0), expected log(1 + 2 e^-10)")
    rng = np.random.default_rng(0)
    c, q = rng.normal(size=(40, 16)), rng.normal(size=(40, 16))
    masked = np.sort(rng.choice(40, 15, replace=False))
    distractors = masked[rng.integers(0, 15, size=(15, 10))]
    for kappa in (0.1, 1.0):
        want = reference_contrastive(c, q, masked, distractors, kappa)
        almost(float(got(c, q, masked, distractors, kappa)), want, 1e-6, label=f"loss at kappa = {kappa}")
    # Plausible wrong implementation: a dot product of the raw context and targets, without scaling
    # both to unit length. Scaling the context by 5 changes its loss, so the check below rejects it.
    scaled = float(got(5.0 * c, q, masked, distractors, 0.1))
    almost(scaled, reference_contrastive(c, q, masked, distractors, 0.1), 1e-6,
           label="loss after scaling the context by 5, which cosine similarity ignores")


@check("diversity_loss is -log(V)/V when codebooks are used evenly and 0 when they collapse")
def test_diversity_loss(sub):
    got = has(sub, "diversity_loss")
    n, g, v = 50, 2, 320
    almost(float(got(np.zeros((n, g, v)))), -np.log(v) / v, 1e-9, label="loss for uniform use")
    collapsed = np.full((n, g, v), -1e4)
    collapsed[:, :, 7] = 0.0
    almost(float(got(collapsed)), 0.0, 1e-9, label="loss when every frame picks entry 7")
    logits = np.random.default_rng(0).normal(size=(30, 2, 12)) * 2
    p = torch.softmax(torch.tensor(logits), dim=2).mean(0)
    want = float((p * torch.log(p)).sum() / (2 * 12))
    almost(float(got(logits)), want, 1e-9, label="loss on random logits")


@check("masked_prediction_loss matches torch and ignores unmasked frames")
def test_masked_prediction(sub):
    got = has(sub, "masked_prediction_loss")
    rng = np.random.default_rng(0)
    t, d, e, c = 30, 12, 8, 20
    o, a, emb = rng.normal(size=(t, d)), rng.normal(size=(e, d)), rng.normal(size=(c, e))
    targets = rng.integers(0, c, t)
    mask = rng.random(t) < 0.5
    proj = torch.tensor(o @ a.T)
    logits = torch.nn.functional.cosine_similarity(proj[:, None], torch.tensor(emb)[None], dim=-1) / 0.1
    want = float(torch.nn.functional.cross_entropy(logits[torch.tensor(mask)], torch.tensor(targets[mask])))
    loss = float(got(o, a, emb, targets, mask, 0.1))
    almost(loss, want, 1e-6, label="masked prediction loss")
    changed = targets.copy()
    changed[~mask] = (changed[~mask] + 1) % c
    almost(float(got(o, a, emb, changed, mask, 0.1)), loss, 1e-12,
           label="loss after changing only the unmasked frames' targets")


@check("best_rq_targets stacks four frames and picks the codeword with the largest cosine")
def test_best_rq(sub):
    got = has(sub, "best_rq_targets")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(42, 10))
    a = rng.normal(size=(16, 40))
    codebook = rng.normal(size=(64, 16))
    labels = np.asarray(got(feats, a, codebook, 4))
    shape(labels, (10,), label="labels for 42 frames stacked by 4")
    x = feats[:40].reshape(10, 40)
    want = np.argmax(unit(x @ a.T) @ unit(codebook).T, axis=1)
    equal(labels.astype(int), want, label="labels")
    again = np.asarray(got(feats, a, codebook * 3.0, 4))
    equal(again.astype(int), want, label="labels after scaling the codebook by 3")


@check("kmeans finds four separated blobs and stops at a fixed point")
def test_kmeans(sub):
    got = has(sub, "kmeans")
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    truth = np.repeat(np.arange(4), 50)
    x = centers[truth] + rng.normal(size=(200, 3))
    cents, labels = got(x, 4, 25, 0)
    shape(cents, (4, 3), label="centroids")
    labels = np.asarray(labels)
    table = np.zeros((4, 4), dtype=int)
    np.add.at(table, (truth, labels), 1)
    if not np.all(np.sort(table.max(axis=1)) == 50) or len(set(table.argmax(axis=1))) != 4:
        fail(f"each blob should land in its own cluster; blob-by-cluster counts are\n{table}")
    nearest = np.argmin(((x[:, None] - cents[None]) ** 2).sum(-1), axis=1)
    equal(labels, nearest, label="each label is the nearest centroid")
    for j in range(4):
        close(cents[j], x[labels == j].mean(axis=0), atol=1e-8, label=f"centroid {j} is its members' mean")


@check("fit_linear_probe solves ridge regression in closed form and separates separable data")
def test_probe(sub):
    fit = has(sub, "fit_linear_probe")
    acc = has(sub, "probe_accuracy")
    rng = np.random.default_rng(0)
    n, d, c, lam = 60, 5, 3, 2.5
    x = rng.normal(size=(n, d))
    y = rng.integers(0, c, n)
    w = np.asarray(fit(x, y, c, lam))
    shape(w, (d + 1, c), label="probe weights")
    # The same minimum by ordinary least squares on an augmented system.
    xa = np.vstack([np.hstack([x, np.ones((n, 1))]), np.hstack([np.sqrt(lam) * np.eye(d), np.zeros((d, 1))])])
    ya = np.vstack([np.eye(c)[y], np.zeros((d, c))])
    want = np.linalg.lstsq(xa, ya, rcond=None)[0]
    close(w, want, atol=1e-8, label="ridge weights, with the bias in the last row")
    means = np.array([[4, 0], [0, 4], [-4, -4]], dtype=float)
    labels = np.repeat(np.arange(3), 40)
    pts = means[labels] + rng.normal(size=(120, 2)) * 0.5
    w = fit(pts, labels, 3, 0.1)
    almost(float(acc(w, pts, labels)), 1.0, 1e-12, label="accuracy on three separated blobs")
