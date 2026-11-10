"""Checks for the self-supervised speech exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, finite, has, shape

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


@check("mix_for_denoising changes only the span and sets the energy ratio to gain_db")
def test_mix_for_denoising(sub):
    got = has(sub, "mix_for_denoising")
    # By hand: the primary span has energy 2 and the secondary span 18. A gain of 10 log10(4) dB
    # scales the secondary energy to 2 / 4, so the scale is sqrt(2 / (4 * 18)) = 1/6 and the
    # added span is (0.5, -0.5).
    primary = np.ones(6)
    secondary = np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0])
    mixed = np.asarray(got(primary, secondary, 2, 2, 10 * math.log10(4)), dtype=float)
    # Plausible wrong implementation: scaling by the energy ratio without the square root, which
    # adds (0.083, -0.083) here instead of (0.5, -0.5). The test below rejects it.
    close(mixed, np.array([1.0, 1.0, 1.5, 0.5, 1.0, 1.0]), atol=1e-9,
          label="mix of 3 and -3 into ones at gain 6.02 dB, expected (1, 1, 1.5, 0.5, 1, 1)")
    rng = np.random.default_rng(0)
    primary = rng.normal(size=400)
    secondary = 7.0 * rng.normal(size=400)
    before = (primary.copy(), secondary.copy())
    for gain_db in (-5.0, 0.0, 5.0, 20.0):
        mixed = np.asarray(got(primary, secondary, 120, 150, gain_db), dtype=float)
        shape(mixed, (400,), label="mixed signal")
        outside = np.ones(400, dtype=bool)
        outside[120:270] = False
        close(mixed[outside], primary[outside], atol=0.0, label="samples outside the span")
        added = mixed[120:270] - primary[120:270]
        ratio = 10 * math.log10(np.sum(primary[120:270] ** 2) / np.sum(added ** 2))
        almost(ratio, gain_db, 1e-6, label=f"energy ratio in the span for gain {gain_db} dB")
    close(primary, before[0], atol=0.0, label="primary after the calls")
    close(secondary, before[1], atol=0.0, label="secondary after the calls")


def reference_bucket(rel, n_buckets, max_distance):
    half = n_buckets // 2
    exact = half // 2
    offset = half if rel > 0 else 0
    d = abs(rel)
    if d < exact:
        return offset + d
    large = exact + int(math.log(d / exact) / math.log(max_distance / exact) * (half - exact))
    return offset + min(large, half - 1)


@check("bucket_relative_position keeps the sign of the offset and spaces far offsets in log steps")
def test_bucket_relative_position(sub):
    got = has(sub, "bucket_relative_position")
    # By hand with 8 buckets and a maximum distance of 16: 4 buckets per side, 2 exact.
    small = np.asarray(got(np.array([-20, -1, 0, 1, 4, 20]), 8, 16)).astype(int)
    equal(small, np.array([3, 1, 0, 5, 6, 7]), label="buckets for offsets -20, -1, 0, 1, 4, 20 with n = 8, max = 16")
    # Plausible wrong implementation: bucketing |rel| and dropping the sign, so that offsets +4
    # and -4 share bucket 2 instead of landing in buckets 6 and 2.
    rel = np.arange(-1000, 1001)
    buckets = np.asarray(got(rel, 320, 800)).astype(int)
    want = np.array([reference_bucket(int(r), 320, 800) for r in rel])
    equal(buckets, want, label="buckets for offsets -1000 to 1000 with n = 320, max = 800")
    equal(int(np.asarray(got(np.array([3]), 320, 800))[0]), 163, label="bucket of offset +3, expected 160 + 3")
    equal(int(np.asarray(got(np.array([-3]), 320, 800))[0]), 3, label="bucket of offset -3")
    equal(int(buckets.min()), 0, label="smallest bucket")
    equal(int(buckets.max()), 319, label="largest bucket, shared by every offset past +800")


@check("gated_relative_bias scales each row by a gate computed from its query")
def test_gated_relative_bias(sub):
    got = has(sub, "gated_relative_bias")
    # Query 0 has q.u = 0 and q.v = 0, so both gates are 1/2 and the scale is 1 + 0.5 + 0.5 * 1 * 0.5 = 1.75.
    # Query 1 has q.u = 2 and q.v = -2, so the gates are 0.8808 and 0.1192 and the scale is 1.8950.
    queries = np.array([[0.0, 0.0], [2.0, -2.0]])
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    out = np.asarray(got(queries, u, v, 1.0, np.full((2, 2), 2.0)), dtype=float)
    shape(out, (2, 2), label="gated bias")
    sig = lambda x: 1 / (1 + math.exp(-x))
    scale1 = 1 + sig(2) + (1 - sig(2)) * 1.0 * sig(-2)
    close(out, np.array([[3.5, 3.5], [2 * scale1, 2 * scale1]]), atol=1e-9,
          label="gated bias for a constant bias of 2, expected (3.5, 3.5) and (3.790013, 3.790013)")
    rng = np.random.default_rng(0)
    t, d = 6, 8
    q, u, v = rng.normal(size=(t, d)), rng.normal(size=d), rng.normal(size=d)
    bias = rng.normal(size=(t, t))
    out = np.asarray(got(q, u, v, 0.7, bias), dtype=float)
    want = np.zeros((t, t))
    for i in range(t):
        gu, gr = sig(float(q[i] @ u)), sig(float(q[i] @ v))
        want[i] = bias[i] * (1 + gu + (1 - gu) * 0.7 * gr)
    close(out, want, atol=1e-9, label="gated bias on random queries, one scale per row")
    saturated = np.asarray(got(np.array([[50.0, 0.0]]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 3.0,
                               np.array([[1.0]])), dtype=float)
    almost(float(saturated[0, 0]), 2.0, 1e-6, label="scale when the update gate is 1, expected 2")


@check("superb_weighted_sum takes a softmax over layer weights before mixing")
def test_superb_weighted_sum(sub):
    got = has(sub, "superb_weighted_sum")
    # By hand: softmax of (0, 0, ln 2) is (1/4, 1/4, 1/2), so layers (4, 8, 12) mix to 1 + 2 + 6 = 9.
    # Plausible wrong implementation: using the raw weights without a softmax gives 0.6931 * 12 = 8.3178.
    layers = np.array([4.0, 8.0, 12.0]).reshape(3, 1, 1)
    out = np.asarray(got(layers, np.array([0.0, 0.0, math.log(2)])), dtype=float)
    shape(out, (1, 1), label="mixed layers")
    almost(float(out[0, 0]), 9.0, 1e-9, label="mix of layers (4, 8, 12), expected 9 (raw weights would give 8.3178)")
    rng = np.random.default_rng(0)
    layers = rng.normal(size=(5, 7, 4))
    weights = rng.normal(size=5)
    want = (torch.softmax(torch.tensor(weights), 0)[:, None, None] * torch.tensor(layers)).sum(0).numpy()
    close(got(layers, weights), want, atol=1e-9, label="mix of five random layers")
    close(got(layers, weights + 100.0), want, atol=1e-9, label="mix after adding 100 to every weight, which softmax ignores")
    close(got(layers, np.zeros(5)), layers.mean(axis=0), atol=1e-9, label="mix with equal weights, expected the layer mean")
    out = np.asarray(got(layers, np.array([1000.0, 0, 0, 0, 0])), dtype=float)
    finite(out, label="mix with a weight of 1000")
    close(out, layers[0], atol=1e-9, label="mix when one weight dominates")
