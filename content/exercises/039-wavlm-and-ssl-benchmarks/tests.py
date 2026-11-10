"""Checks for the WavLM and SUPERB exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, finite, has, shape


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
