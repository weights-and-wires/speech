"""Checks for the diarization exercise."""

import itertools
import math

import numpy as np

from speechcheck import almost, check, equal, fail, has, shape


def blobs(rng, sizes, dim=16, spread=0.15):
    centers = np.linalg.qr(rng.normal(size=(dim, dim)))[0][: len(sizes)]
    labels = np.repeat(np.arange(len(sizes)), sizes)
    return centers[labels] + spread * rng.normal(size=(labels.size, dim)), labels


@check("affinity_matrix prunes each row to its largest cosines and comes out symmetric")
def test_affinity(sub):
    got = has(sub, "affinity_matrix")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 4))
    a = np.asarray(got(x, 0.3))
    shape(a, (10, 10), label="affinity")
    if not np.allclose(a, a.T):
        fail("the affinity matrix must be symmetric")
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    cos = np.maximum(unit @ unit.T, 0)
    keep = np.zeros_like(cos, dtype=bool)
    order = np.argsort(-cos, axis=1, kind="stable")
    for i in range(10):
        keep[i, order[i, :3]] = True
    want = np.maximum(np.where(keep, cos, 0), np.where(keep, cos, 0).T)
    if not np.allclose(a, want, atol=1e-12):
        fail(f"with keep = 0.3 and 10 rows, each row keeps its 3 largest cosines before symmetrizing; "
             f"the largest difference from that is {np.max(np.abs(a - want)):.3g}")


@check("speaker_count reads the number of speakers off the largest eigengap")
def test_speaker_count(sub):
    got = has(sub, "speaker_count")
    for k in (2, 3, 5):
        blocks = np.zeros((6 * k, 6 * k))
        for j in range(k):
            blocks[6 * j : 6 * j + 6, 6 * j : 6 * j + 6] = 1.0
        equal(int(got(blocks, 8)), k, label=f"speakers in a {k}-block affinity matrix")
    rng = np.random.default_rng(3)
    affinity = has(sub, "affinity_matrix")
    x, _ = blobs(rng, [8, 12, 10, 6])
    equal(int(got(affinity(x, 0.3), 8)), 4, label="speakers in four noisy clusters of embeddings")


@check("spectral_cluster recovers four synthetic speakers exactly")
def test_spectral(sub):
    got = has(sub, "spectral_cluster")
    rng = np.random.default_rng(4)
    x, truth = blobs(rng, [9, 14, 7, 11], spread=0.2)
    labels = np.asarray(got(x, 8, 0.3))
    shape(labels, truth.shape, label="labels")
    table = np.zeros((4, len(set(labels.tolist()))), dtype=int)
    for t, h in zip(truth, labels):
        table[t, list(sorted(set(labels.tolist()))).index(h)] += 1
    if table.shape[1] != 4 or np.any((table > 0).sum(axis=1) != 1) or np.any((table > 0).sum(axis=0) != 1):
        fail(f"each true speaker should map to exactly one cluster; true-by-found counts are\n{table}")


def brute_der(ref, hyp):
    speech = ref >= 0
    errors = np.sum(speech & (hyp < 0)) + np.sum(~speech & (hyp >= 0))
    both = speech & (hyp >= 0)
    hyp_ids = sorted(set(hyp[both].tolist()))
    ref_ids = sorted(set(ref[both].tolist()))
    best = 0
    slots = ref_ids + [None] * len(hyp_ids)
    for perm in itertools.permutations(slots, len(hyp_ids)):
        mapping = dict(zip(hyp_ids, perm))
        best = max(best, sum(int(mapping[h] == r) for r, h in zip(ref[both], hyp[both])))
    return (errors + np.sum(both) - best) / np.sum(speech)


@check("diarization_error_rate is 30% on the worked example")
def test_der(sub):
    got = has(sub, "diarization_error_rate")
    ref = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    hyp = np.array([5, 5, 5, -1, 5, 7, 7, 7, 7, 7, 7, -1])
    # One missed frame, one false alarm, and frame 4 confused: 3 of 10 speech frames.
    almost(float(got(ref, hyp)), 0.3, 1e-12, label="DER of the worked example")


@check("diarization_error_rate matches a brute-force search over speaker mappings on random labelings")
def test_der_random(sub):
    got = has(sub, "diarization_error_rate")
    rng = np.random.default_rng(5)
    for trial in range(30):
        ref = rng.integers(-1, 3, 25)
        hyp = rng.integers(-1, 4, 25)
        if np.sum(ref >= 0) == 0:
            continue
        almost(float(got(ref, hyp)), float(brute_der(ref, hyp)), 1e-12, label=f"DER of random labeling {trial}")


def activity(labels, n):
    """One boolean row per speaker from one label per frame (-1 is silence)."""
    return np.array([labels == k for k in range(n)])


def brute_multi_der(ref, hyp, collar, score_overlap):
    """A slow reference: loops over frames and tries every speaker mapping."""
    n_frames = ref.shape[1]
    keep = np.ones(n_frames, dtype=bool)
    edges = set()
    for row in ref:
        wide = [False, *row.tolist(), False]
        edges |= {t for t in range(n_frames + 1) if wide[t] != wide[t + 1]}
    for t in edges:
        for f in range(t - collar, t + collar):
            if 0 <= f < n_frames and collar > 0:
                keep[f] = False
    if not score_overlap:
        keep &= ref.sum(axis=0) < 2
    frames = np.flatnonzero(keep)
    slots = list(range(ref.shape[0])) + [None] * hyp.shape[0]
    best = 0
    for perm in itertools.permutations(slots, hyp.shape[0]):
        best = max(best, sum(int(ref[r, f] and hyp[h, f]) for h, r in enumerate(perm) if r is not None for f in frames))
    total = sum(int(ref[:, f].sum()) for f in frames)
    errors = sum(max(int(ref[:, f].sum()), int(hyp[:, f].sum())) for f in frames) - best
    return errors / max(total, 1)


def overlap_pair():
    """Speaker A talks in frames 0-7, speaker B in frames 5-11; the system hears B only from frame 8."""
    ref = np.zeros((2, 12), dtype=bool)
    ref[0, :8], ref[1, 5:] = True, True
    hyp = np.zeros((2, 12), dtype=bool)
    hyp[0, :8], hyp[1, 8:] = True, True
    return ref, hyp


@check("der_collar_overlap with no overlap and no collar matches diarization_error_rate")
def test_der_multi_single(sub):
    got = has(sub, "der_collar_overlap")
    rng = np.random.default_rng(8)
    for trial in range(20):
        ref, hyp = rng.integers(-1, 3, 30), rng.integers(-1, 4, 30)
        if np.sum(ref >= 0) == 0:
            continue
        almost(float(got(activity(ref, 3), activity(hyp, 4))), float(brute_der(ref, hyp)), 1e-12,
               label=f"DER of single-label trial {trial}")


@check("der_collar_overlap counts the second speaker of an overlap as missed speech")
def test_der_overlap(sub):
    # Wrong implementation caught here: scoring one label per frame, so overlapped frames count once.
    # It reports 0% for this pair, because the system gets every frame's first speaker right.
    got = has(sub, "der_collar_overlap")
    ref, hyp = overlap_pair()
    # 15 reference speaker-frames; frames 5, 6, and 7 each lose speaker B.
    almost(float(got(ref, hyp)), 3 / 15, 1e-12, label="DER with overlap scored, no collar")
    # Dropping the 3 overlapped frames leaves 9 reference speaker-frames and no errors.
    almost(float(got(ref, hyp, score_overlap=False)), 0.0, 1e-12, label="DER with overlapped frames left out")


@check("der_collar_overlap forgives frames around reference boundaries, not hypothesis boundaries")
def test_der_collar(sub):
    # Wrong implementation caught here: taking the boundaries from the hypothesis. It forgives frames 0, 7, 8,
    # and 11 instead of 0, 4, 5, 7, 8, and 11, and scores 2 of 10 (20%) here, not 1 of 7.
    got = has(sub, "der_collar_overlap")
    ref, hyp = overlap_pair()
    # A collar of 1 frame forgives frames 0, 4, 5, 7, 8, and 11, which holds 8 reference speaker-frames.
    almost(float(got(ref, hyp, collar=1)), 1 / 7, 1e-12, label="DER with a collar of 1 frame")
    # Shift the system's start of speaker A late by 2 frames: a collar of 2 forgives it, a collar of 1 does not.
    late = hyp.copy()
    late[0, :2] = False
    almost(float(got(ref, late, collar=2)), brute_multi_der(ref, late, 2, True), 1e-12, label="DER of a late start, collar 2")
    almost(float(got(ref, late, collar=1)), brute_multi_der(ref, late, 1, True), 1e-12, label="DER of a late start, collar 1")


@check("der_collar_overlap matches a slow reference on random overlapping speakers, collar, and overlap settings")
def test_der_multi_random(sub):
    got = has(sub, "der_collar_overlap")
    rng = np.random.default_rng(9)
    for trial in range(25):
        ref = rng.random((2, 30)) < 0.45
        hyp = rng.random((3, 30)) < 0.4
        for collar, overlap in ((0, True), (2, True), (3, False)):
            almost(float(got(ref, hyp, collar, overlap)), float(brute_multi_der(ref, hyp, collar, overlap)), 1e-12,
                   label=f"DER of trial {trial} with collar {collar}, overlap scored {overlap}")
