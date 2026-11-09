"""Checks for the deltas and normalization exercise."""

import numpy as np

from speechcheck import check, close, fail, finite, has, shape

RATE = 16000


def ref_deltas(c, width):
    """A plain double loop, with edge frames repeated."""
    rows, T = c.shape
    out = np.zeros_like(c)
    denom = 2.0 * sum(k * k for k in range(1, width + 1))
    for t in range(T):
        for k in range(1, width + 1):
            ahead = c[:, min(t + k, T - 1)]
            behind = c[:, max(t - k, 0)]
            out[:, t] += k * (ahead - behind)
    return out / denom


def voiced_signal(seconds=1.0):
    rng = np.random.default_rng(0)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 110 + 40 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    x = sum(np.sin(k * phase) / k for k in range(1, 40)) * (1 + 0.8 * np.sin(2 * np.pi * 3 * t))
    return x + 0.05 * rng.standard_normal(n)


@check("deltas reproduces the worked example [1, 2, 4, 7, 11] with repeated edge frames")
def test_deltas_hand(sub):
    deltas = has(sub, "deltas")
    got = np.asarray(deltas(np.array([[1.0, 2.0, 4.0, 7.0, 11.0]]), 2), dtype=float)
    shape(got, (1, 5), label="the deltas of one coefficient over five frames")
    close(got, [[0.7, 1.5, 2.5, 2.5, 1.8]], atol=1e-12, label="the worked example")
    close(deltas(np.array([[3.0, 3.0, 3.0, 3.0]]), 2), np.zeros((1, 4)), atol=1e-12, label="a constant row, whose slope is 0 everywhere")


@check("deltas gives a ramp its slope and matches a brute-force loop at widths 1, 2, and 3")
def test_deltas_general(sub):
    deltas = has(sub, "deltas")
    ramp = 0.25 * np.arange(12, dtype=float)[None, :].repeat(3, axis=0)
    close(deltas(ramp, 2)[:, 2:-2], np.full((3, 8), 0.25), atol=1e-12, label="the interior deltas of a ramp of slope 0.25")
    c = np.random.default_rng(3).standard_normal((13, 20))
    for width in (1, 2, 3):
        close(deltas(c, width), ref_deltas(c, width), atol=1e-12, label=f"deltas with width {width}")


@check("cmvn reproduces the worked example [2, 4, 6, 8], and mean-only normalization keeps the spread")
def test_cmvn_hand(sub):
    cmvn = has(sub, "cmvn")
    close(cmvn(np.array([[2.0, 4.0, 6.0, 8.0]])), [[-1.3416408, -0.4472136, 0.4472136, 1.3416408]], atol=1e-6, label="the worked example")
    close(cmvn(np.array([[2.0, 4.0, 6.0, 8.0]]), variance=False), [[-3.0, -1.0, 1.0, 3.0]], atol=1e-12, label="the worked example with variance=False")


@check("cmvn gives every row zero mean and unit variance, and leaves a constant row finite")
def test_cmvn(sub):
    cmvn = has(sub, "cmvn")
    c = 5.0 + 3.0 * np.random.default_rng(4).standard_normal((13, 200))
    c[7] = 2.5  # a coefficient that never changes
    got = np.asarray(cmvn(c), dtype=float)
    finite(got, label="CMVN output with a constant row")
    close(got.mean(axis=1), np.zeros(13), atol=1e-10, label="the mean of every row")
    close(np.delete(got.std(axis=1), 7), np.ones(12), atol=1e-10, label="the standard deviation of every non-constant row")
    close(cmvn(c, variance=False).std(axis=1), c.std(axis=1), atol=1e-10, label="standard deviations with variance=False")


@check("Mean normalization removes a fixed channel: the MFCC difference drops by at least 90%")
def test_channel(sub):
    mfcc, cmvn, mel_power = has(sub, "mfcc"), has(sub, "cmvn"), has(sub, "mel_power")
    x = voiced_signal()
    y = np.convolve(x, [1.0, 0.6, 0.3])[: len(x)]
    a, b = mfcc(mel_power(x)), mfcc(mel_power(y))
    before = float(np.mean(np.abs(a - b)))
    after = float(np.mean(np.abs(cmvn(a, variance=False) - cmvn(b, variance=False))))
    if after > 0.1 * before:
        fail(f"the mean |MFCC difference| went from {before:.3f} to {after:.3f}, a drop of "
             f"{100 * (1 - after / before):.1f}%; subtracting each row's mean over time should remove at least 90%")


@check("mfcc_features stacks 13 liftered MFCCs, deltas, and delta-deltas into 39 normalized rows")
def test_features(sub):
    mfcc_features = has(sub, "mfcc_features")
    x = voiced_signal(0.5)
    got = np.asarray(mfcc_features(x), dtype=float)
    shape(got, (39, 1 + len(x) // 160), label="the feature matrix")
    finite(got, label="the features")
    close(got.mean(axis=1), np.zeros(39), atol=1e-8, label="the mean of every feature row")
    close(got.std(axis=1), np.ones(39), atol=1e-6, label="the standard deviation of every feature row")
    c = sub.lifter(sub.mfcc(sub.mel_power(x), 13))
    d = sub.deltas(c)
    want = sub.cmvn(np.vstack([c, d, sub.deltas(d)]))
    close(got, want, atol=1e-8, label="cmvn(vstack([lifter(mfcc), deltas, deltas of deltas]))")
