"""Checks for the speaker recognition and diarization exercise."""

import itertools
import math

import numpy as np
import torch
from scipy.stats import norm

from speechcheck import almost, check, equal, fail, has, shape


def two_class_case(target_cos, other_cos):
    """Unit vectors whose cosines with the two class centers are the given values."""
    weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    rest = math.sqrt(1 - target_cos**2 - other_cos**2)
    x = torch.tensor([[target_cos, other_cos, rest]], dtype=torch.float64)
    return x, weights, torch.tensor([0])


@check("aam_softmax_loss matches the worked example: 0.0486 without a margin, 2.54 with m = 0.2")
def test_aam_worked(sub):
    got = has(sub, "aam_softmax_loss")
    x, w, y = two_class_case(0.5, 0.4)
    plain = float(got(x, w, y, 30.0, 0.0))
    almost(plain, math.log(1 + math.exp(-3.0)), 1e-9, label="loss with m = 0, logits 15 and 12")
    shifted = 0.5 * math.cos(0.2) - math.sqrt(0.75) * math.sin(0.2)
    want = math.log(1 + math.exp(30 * 0.4 - 30 * shifted))
    almost(float(got(x, w, y, 30.0, 0.2)), want, 1e-9, label="loss with m = 0.2")
    almost(float(got(3.0 * x, 2.0 * w, y, 30.0, 0.2)), want, 1e-9,
           label="loss after scaling the embedding and centers, which cosines ignore")


@check("aam_softmax_loss matches cross-entropy on a batch and keeps the target logit falling past pi - m")
def test_aam_batch(sub):
    got = has(sub, "aam_softmax_loss")
    torch.manual_seed(0)
    x = torch.randn(16, 8, dtype=torch.float64)
    w = torch.randn(5, 8, dtype=torch.float64)
    y = torch.randint(0, 5, (16,))
    cos = torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(w, dim=1).T
    want = torch.nn.functional.cross_entropy(20.0 * cos, y)
    almost(float(got(x, w, y, 20.0, 0.0)), float(want), 1e-9, label="batch loss with m = 0")
    # Sweep the target angle from 0 to pi: the loss must rise the whole way.
    losses = []
    for theta in np.linspace(0.01, math.pi - 0.01, 200):
        xt = torch.tensor([[math.cos(theta), 0.0, math.sin(theta)]], dtype=torch.float64)
        wt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
        losses.append(float(got(xt, wt, torch.tensor([0]), 30.0, 0.5)))
    if np.any(np.diff(losses) <= 0):
        where = float(np.linspace(0.01, math.pi - 0.01, 200)[np.argmax(np.diff(losses) <= 0)])
        fail(f"with m = 0.5 the loss should grow as the target angle grows, but it stops growing near "
             f"{where:.2f} rad; past pi - m = {math.pi - 0.5:.2f} use cos(theta) - m sin(m)")
    xt = torch.tensor([[math.cos(3.0), 0.0, math.sin(3.0)]], dtype=torch.float64)
    wt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    fallback = math.cos(3.0) - 0.5 * math.sin(0.5)
    almost(float(got(xt, wt, torch.tensor([0]), 30.0, 0.5)), math.log(1 + math.exp(-30 * fallback)), 1e-9,
           label="loss at theta = 3.0 rad with m = 0.5, which uses the fallback")


@check("equal_error_rate is 25% on the worked example and 0 when the scores separate")
def test_eer_worked(sub):
    got = has(sub, "equal_error_rate")
    almost(float(got(np.array([0.9, 0.8, 0.7, 0.4]), np.array([0.6, 0.3, 0.2, 0.1]))), 0.25, 1e-12,
           label="EER of the worked example")
    almost(float(got(np.array([0.9, 0.8]), np.array([0.2, 0.1]))), 0.0, 1e-12, label="EER of separated scores")
    almost(float(got(np.array([0.1, 0.2]), np.array([0.8, 0.9]))), 1.0, 1e-12, label="EER of reversed scores")


@check("equal_error_rate matches the Gaussian closed form Phi(-d/2) on 4,000 trials")
def test_eer_gaussian(sub):
    got = has(sub, "equal_error_rate")
    rng = np.random.default_rng(0)
    for d in (1.0, 2.0, 3.0):
        target = rng.normal(d, 1.0, 2000)
        nontarget = rng.normal(0.0, 1.0, 2000)
        almost(float(got(target, nontarget)), float(norm.cdf(-d / 2)), 0.02,
               label=f"EER for unit-variance Gaussians {d} apart")


@check("min_dcf matches the worked example and a brute-force threshold sweep")
def test_min_dcf(sub):
    got = has(sub, "min_dcf")
    almost(float(got(np.array([0.9, 0.8, 0.7, 0.4]), np.array([0.6, 0.3, 0.2, 0.1]), 0.05)), 0.25, 1e-12,
           label="minDCF of the worked example at P_target = 0.05")
    rng = np.random.default_rng(1)
    target, nontarget = rng.normal(1.5, 1, 300), rng.normal(0, 1, 700)
    for p_target in (0.05, 0.01):
        best = np.inf
        for t in np.concatenate([np.sort(np.concatenate([target, nontarget])), [np.inf]]):
            cost = np.mean(target < t) * p_target + np.mean(nontarget >= t) * (1 - p_target)
            best = min(best, cost / min(p_target, 1 - p_target))
        almost(float(got(target, nontarget, p_target)), float(best), 1e-12, label=f"minDCF at P_target = {p_target}")


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


@check("diarization_error_rate matches the worked example and a brute-force best mapping")
def test_der(sub):
    got = has(sub, "diarization_error_rate")
    ref = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    hyp = np.array([5, 5, 5, -1, 5, 7, 7, 7, 7, 7, 7, -1])
    # One missed frame, one false alarm, and frame 4 confused: 3 of 10 speech frames.
    almost(float(got(ref, hyp)), 0.3, 1e-12, label="DER of the worked example")
    rng = np.random.default_rng(5)
    for trial in range(30):
        ref = rng.integers(-1, 3, 25)
        hyp = rng.integers(-1, 4, 25)
        if np.sum(ref >= 0) == 0:
            continue
        almost(float(got(ref, hyp)), float(brute_der(ref, hyp)), 1e-12, label=f"DER of random labeling {trial}")


def brute_align(log_probs, tokens):
    t_frames, v = log_probs.shape
    best, best_path = -np.inf, None
    for path in itertools.product(range(v), repeat=t_frames):
        collapsed = [p for i, p in enumerate(path) if p != 0 and (i == 0 or path[i - 1] != p)]
        if collapsed == list(tokens):
            score = sum(log_probs[t, p] for t, p in enumerate(path))
            if score > best:
                best, best_path = score, path
    return best, best_path


@check("forced_align finds the best CTC path by Viterbi, matching brute force, repeats included")
def test_forced_align(sub):
    got = has(sub, "forced_align")
    lp = np.log(np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]]))
    states, score = got(lp, [1])
    equal(np.asarray(states).tolist(), [0, 1, 2], label="states for 'a' over three frames")
    almost(float(score), math.log(0.336), 1e-9, label="best path log-probability, log(0.6 * 0.7 * 0.8)")
    rng = np.random.default_rng(6)
    for tokens, t_frames in (([1, 2], 5), ([1, 1], 5), ([2, 1, 2], 6), ([1, 1, 1], 5)):
        logits = rng.normal(size=(t_frames, 3)) * 2
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        states, score = got(lp, tokens)
        want, path = brute_align(lp, tokens)
        almost(float(score), float(want), 1e-9, label=f"best score for tokens {tokens} over {t_frames} frames")
        ext = [0]
        for tok in tokens:
            ext += [tok, 0]
        states = np.asarray(states)
        steps = np.diff(states)
        legal = states[0] <= 1 and states[-1] >= len(ext) - 2 and np.all((steps >= 0) & (steps <= 2))
        legal = legal and all(ext[s] != 0 and ext[s] != ext[s - 2] for s, d in zip(states[1:], steps) if d == 2)
        if not legal:
            fail(f"states {states.tolist()} are not a legal CTC path through {ext}")
        labels = [ext[s] for s in states]
        along = sum(lp[t, s] for t, s in enumerate(labels))
        almost(float(along), float(want), 1e-9, label=f"log-probability along your returned path for {tokens}")
    try:
        got(np.log(np.full((2, 3), 1 / 3)), [1, 1])
    except ValueError:
        pass
    else:
        fail("two frames cannot hold 'a a', which needs a blank between the letters; expected ValueError")
