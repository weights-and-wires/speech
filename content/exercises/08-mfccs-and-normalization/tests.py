"""Checks for the MFCCs and normalization exercise.

The MFCC reference numbers were computed offline with librosa 0.11.0:
`librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)`.
"""

import numpy as np
import scipy.fft

from speechcheck import check, close, fail, finite, has, shape

RATE = 16000


def fixture_power():
    b = np.arange(40)[:, None]
    t = np.arange(30)[None, :]
    return 10.0 ** (3.0 * np.sin(0.3 * b + 0.7 * t) - 0.05 * b)


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


@check("dct2_ortho matches scipy.fft.dct(type=2, norm='ortho') on a vector and along the rows of a matrix")
def test_dct(sub):
    dct2_ortho = has(sub, "dct2_ortho")
    close(dct2_ortho(np.array([4.0, 2.0, 2.0, 0.0])), [4.0, 2.6131259, 0.0, 1.0823922], atol=1e-6, label="the worked example [4, 2, 2, 0]")
    x = np.random.default_rng(1).standard_normal((24, 7))
    got = dct2_ortho(x)
    shape(got, (24, 7), label="the DCT of a (24, 7) matrix")
    close(got, scipy.fft.dct(x, type=2, norm="ortho", axis=0), atol=1e-10, label="DCT-II down each column")


@check("dct2_ortho is orthonormal: it keeps energy, and its transpose undoes it")
def test_orthonormal(sub):
    dct2_ortho = has(sub, "dct2_ortho")
    C = dct2_ortho(np.eye(16))
    close(C.T @ C, np.eye(16), atol=1e-10, label="C^T C for the 16-point DCT matrix")
    x = np.random.default_rng(2).standard_normal(16)
    close(np.sum(dct2_ortho(x) ** 2), np.sum(x**2), rtol=1e-10, label="the energy of a vector and of its DCT")


@check("power_to_db takes 10 log10 with a floor of amin and a top_db cut below the global maximum")
def test_power_to_db(sub):
    power_to_db = has(sub, "power_to_db")
    S = np.array([[1.0, 100.0], [1e-3, 0.0]])
    close(power_to_db(S, top_db=None), [[0.0, 20.0], [-30.0, -100.0]], atol=1e-9, label="power_to_db with no top_db")
    close(power_to_db(S, top_db=80.0), [[0.0, 20.0], [-30.0, -60.0]], atol=1e-9, label="power_to_db with top_db=80")
    close(power_to_db(S, top_db=40.0), [[0.0, 20.0], [-20.0, -20.0]], atol=1e-9, label="power_to_db with top_db=40")


@check("mfcc matches librosa.feature.mfcc on a 40-band, 30-frame mel power matrix")
def test_mfcc(sub):
    mfcc = has(sub, "mfcc")
    got = np.asarray(mfcc(fixture_power(), n_mfcc=13), dtype=float)
    shape(got, (13, 30), label="13 MFCCs of 30 frames")
    close(got[[0, 1, 5, 12], [0, 7, 15, 29]], [-57.9414787, 24.9937075, 49.5100268, 1.1552404], atol=1e-3,
          label="entries (0, 0), (1, 7), (5, 15), and (12, 29) against librosa")
    close(got.mean(), -0.786521573535041, atol=1e-3, label="the mean coefficient against librosa")


@check("lifter scales row n by 1 + (L / 2) sin(pi n / L), the HTK convention, and leaves c0 alone")
def test_lifter(sub):
    lifter = has(sub, "lifter")
    c = np.ones((13, 3))
    got = np.asarray(lifter(c, 22), dtype=float)
    shape(got, (13, 3), label="liftered coefficients")
    want = 1.0 + 11.0 * np.sin(np.pi * np.arange(13) / 22.0)
    close(got[:, 0], want, atol=1e-10, label="the lifter weights for L = 22")
    close(got[[0, 1, 11, 12], 1], [1.0, 2.5654632, 12.0, 11.8880359], atol=1e-6, label="the weights on rows 0, 1, 11, and 12")


@check("deltas matches the HTK regression formula with repeated edge frames")
def test_deltas(sub):
    deltas = has(sub, "deltas")
    close(deltas(np.array([[1.0, 2.0, 4.0, 7.0, 11.0]]), 2), [[0.7, 1.5, 2.5, 2.5, 1.8]], atol=1e-12, label="the worked example")
    ramp = 0.25 * np.arange(12, dtype=float)[None, :].repeat(3, axis=0)
    close(deltas(ramp, 2)[:, 2:-2], np.full((3, 8), 0.25), atol=1e-12, label="the interior deltas of a ramp of slope 0.25")
    c = np.random.default_rng(3).standard_normal((13, 20))
    for width in (1, 2, 3):
        close(deltas(c, width), ref_deltas(c, width), atol=1e-12, label=f"deltas with width {width}")


@check("cmvn gives every row zero mean and unit variance, and leaves a constant row finite")
def test_cmvn(sub):
    cmvn = has(sub, "cmvn")
    close(cmvn(np.array([[2.0, 4.0, 6.0, 8.0]])), [[-1.3416408, -0.4472136, 0.4472136, 1.3416408]], atol=1e-6, label="the worked example")
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
