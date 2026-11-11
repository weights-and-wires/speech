"""Checks for the neural diarization exercise."""

import itertools

import numpy as np

from speechcheck import almost, check, close, equal, fail, finite_difference, has, shape


def reference_bce(z, y):
    p = 1 / (1 + np.exp(-z))
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def planted():
    """Three frames, two speakers; the network puts speaker A in slot 1 and B in slot 0."""
    labels = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    probs = np.array([[0.1, 0.9], [0.8, 0.9], [0.9, 0.1]])
    return np.log(probs / (1 - probs)), labels


def timeline():
    """Two speakers over 14 frames; A talks in 0-7 and B in 5-11, so frames 5-7 overlap."""
    ref = np.zeros((2, 14), dtype=bool)
    ref[0, 0:8] = True
    ref[1, 5:12] = True
    hyp = np.zeros((2, 14), dtype=bool)
    hyp[1, 0:10] = True
    hyp[0, 10:14] = True
    return ref, hyp


@check("bce_with_logits matches the textbook formula and survives large logits")
def test_bce(sub):
    got = has(sub, "bce_with_logits")
    z, y = planted()
    almost(float(got(z, y[:, ::-1])), 0.125, 5e-4, label="BCE of the planted example, correct slot order")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(7, 3))
    y = (rng.random((7, 3)) > 0.5).astype(float)
    almost(float(got(z, y)), reference_bce(z, y), 1e-10, label="BCE of random logits")
    big = float(got(np.array([[800.0, -800.0]]), np.array([[1.0, 0.0]])))
    if not np.isfinite(big) or abs(big) > 1e-9:
        fail(f"logits of +800 and -800 with matching labels should give a loss of 0, got {big}")


# Wrong implementation caught here: BCE with a fixed slot order and no permutation search.
# It scores this example at 1.590 instead of 0.125 and reports the identity order.
@check("pit_bce_loss finds the swapped order on the planted example")
def test_pit_swapped(sub):
    got = has(sub, "pit_bce_loss")
    z, y = planted()
    loss, perm = got(z, y)
    almost(float(loss), 0.125, 5e-4, label="PIT loss (a fixed-order BCE gives 1.590)")
    equal(tuple(perm), (1, 0), label="best slot order")
    loss2, perm2 = got(z, y[:, ::-1])
    almost(float(loss2), 0.125, 5e-4, label="PIT loss with the labels swapped back")
    equal(tuple(perm2), (0, 1), label="best slot order for swapped labels")


@check("pit_bce_loss matches a brute-force search over three speakers")
def test_pit_brute(sub):
    got = has(sub, "pit_bce_loss")
    rng = np.random.default_rng(1)
    for trial in range(8):
        z = rng.normal(scale=2.0, size=(20, 3))
        y = (rng.random((20, 3)) > 0.6).astype(float)
        want = min(reference_bce(z, y[:, list(p)]) for p in itertools.permutations(range(3)))
        loss, perm = got(z, y)
        almost(float(loss), want, 1e-10, label=f"PIT loss, trial {trial}")
        if sorted(perm) != [0, 1, 2]:
            fail(f"perm must be an ordering of 0, 1, 2, got {tuple(perm)}")


@check("pit_bce_grad matches finite differences, including through a swap")
def test_pit_grad(sub):
    loss_fn = has(sub, "pit_bce_loss")
    grad = has(sub, "pit_bce_grad")
    z, y = planted()
    rng = np.random.default_rng(2)
    z = z + 0.1 * rng.normal(size=z.shape)
    g = np.asarray(grad(z, y))
    shape(g, z.shape, label="gradient")
    want = finite_difference(lambda w: loss_fn(w.reshape(z.shape), y)[0], z.ravel()).reshape(z.shape)
    close(g, want, rtol=1e-5, atol=1e-7, label="gradient of the PIT loss")


@check("overlap_frames marks frames with two or more speakers")
def test_overlap(sub):
    got = has(sub, "overlap_frames")
    ref, _ = timeline()
    out = np.asarray(got(ref))
    shape(out, (14,), label="overlap mask")
    equal(np.flatnonzero(out).tolist(), [5, 6, 7], label="overlap frames")
    three = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
    equal(np.flatnonzero(got(three)).tolist(), [0, 1], label="overlap frames with three speakers")


# Wrong implementation caught here: an overlapped frame counted once in the
# denominator (12 speech frames instead of 15 speaker-frames), which reports
# 7/12 = 58.3% instead of 7/15 = 46.7%.
@check("der_components splits the planted timeline into miss, false alarm, and confusion")
def test_der(sub):
    got = has(sub, "der_components")
    ref, hyp = timeline()
    out = got(ref, hyp)
    for key, want in (("miss", 3.0), ("false_alarm", 2.0), ("confusion", 2.0), ("total", 15.0)):
        almost(float(out[key]), want, 1e-12, label=f"{key} in speaker-frames")
    almost(float(out["der"]), 7 / 15, 1e-12, label="DER (an overlapped frame counts once per speaker)")


@check("der_components ignores slot order and equals zero for a perfect system")
def test_der_mapping(sub):
    got = has(sub, "der_components")
    ref, _ = timeline()
    out = got(ref, ref[::-1])
    almost(float(out["der"]), 0.0, 1e-12, label="DER of the reference with its rows swapped")
    rng = np.random.default_rng(3)
    for trial in range(10):
        r = rng.random((3, 30)) > 0.6
        h = rng.random((3, 30)) > 0.6
        perm_best = None
        n_ref, n_hyp = r.sum(0), h.sum(0)
        for p in itertools.permutations(range(3)):
            ok = sum(int((r[i] & h[j]).sum()) for i, j in enumerate(p))
            perm_best = ok if perm_best is None else max(perm_best, ok)
        want = (np.maximum(n_ref, n_hyp).sum() - perm_best) / n_ref.sum()
        almost(float(got(r, h)["der"]), float(want), 1e-12, label=f"DER of random timeline {trial}")


@check("attractor_count stops at the first low probability and attractor_activity gives overlap")
def test_attractors(sub):
    count = has(sub, "attractor_count")
    equal(int(count(np.array([0.93, 0.81, 0.20]), 0.5)), 2, label="speakers for [0.93, 0.81, 0.20]")
    equal(int(count(np.array([0.93, 0.81, 0.20, 0.74]), 0.5)), 2, label="speakers when a later attractor looks real")
    equal(int(count(np.array([0.3, 0.9]), 0.5)), 0, label="speakers when the first attractor is absent")
    equal(int(count(np.array([0.9, 0.8]), 0.5)), 2, label="speakers when no attractor falls below the threshold")
    activity = has(sub, "attractor_activity")
    e = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    a = np.array([[3.0, -1.0], [-1.0, 3.0]])
    got = np.asarray(activity(e, a))
    want = np.array([[0.953, 0.269], [0.881, 0.881], [0.269, 0.953]])
    shape(got, (3, 2), label="activity")
    close(got, want, atol=5e-4, label="activity (rows are frames, columns are attractors)")


@check("target_speaker_mask uses the cosine, so loudness does not matter")
def test_target(sub):
    got = has(sub, "target_speaker_mask")
    enroll = np.array([1.0, 0.0])
    frames = np.array([[3.0, 0.0], [2.0, 2.0], [0.0, 5.0]])
    equal(np.asarray(got(frames, enroll, 0.5)).tolist(), [True, True, False], label="mask at threshold 0.5")
    equal(np.asarray(got(frames, enroll, 0.9)).tolist(), [True, False, False], label="mask at threshold 0.9")
    # A quiet target frame has a small dot product but a cosine of 1; a loud
    # frame of another voice has a large one that a cosine does not give it.
    enroll = np.array([3.0, 4.0])
    frames = np.array([[0.03, 0.04], [40.0, -30.0], [6.0, 8.0], [-3.0, -4.0]])
    equal(np.asarray(got(frames, enroll, 0.5)).tolist(), [True, False, True, False], label="mask for quiet and loud frames")
