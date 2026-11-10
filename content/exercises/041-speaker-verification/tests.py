"""Checks for the speaker verification exercise."""

import math

import numpy as np
import torch
from scipy.stats import norm

from speechcheck import almost, check, close, equal, fail, has, shape


def two_class_case(target_cos, other_cos):
    """Unit vectors whose cosines with the two class centers are the given values."""
    weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    rest = math.sqrt(1 - target_cos**2 - other_cos**2)
    x = torch.tensor([[target_cos, other_cos, rest]], dtype=torch.float64)
    return x, weights, torch.tensor([0])


@check("cosine_scores gives the cosine of each pair of rows and ignores length")
def test_cosine(sub):
    got = has(sub, "cosine_scores")
    a = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    b = np.array([[0.0, 3.0], [5.0, 5.0], [-1.0, 0.0]])
    shape(got(a, b), (3,), label="one score per pair")
    close(got(a, b), [0.0, 1.0, -1.0], atol=1e-12, label="orthogonal, parallel, and opposite pairs")
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
    close(got(3.0 * x, 0.5 * y), got(x, y), atol=1e-12, label="scores after rescaling the embeddings")
    in_range = got(x, y)
    if np.any(np.abs(in_range) > 1 + 1e-12):
        fail("a cosine must lie in [-1, 1]")


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


@check("stats_pooling concatenates the per-channel mean and standard deviation over frames")
def test_stats_pooling(sub):
    got = has(sub, "stats_pooling")
    frames = np.array([[1.0, 5.0], [2.0, 5.0], [6.0, 5.0]])
    out = got(frames)
    shape(out, (4,), label="2 channels give a mean and a standard deviation each")
    close(out, [3.0, 5.0, math.sqrt(41 / 3 - 9), 0.0], atol=1e-9,
          label="means 3 and 5 then population standard deviations 2.1602 and 0 for the worked frames")
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, size=(50, 6))
    close(got(x), np.concatenate([x.mean(0), x.std(0)]), atol=1e-9, label="pooling 50 random frames of 6 channels")
    close(got(x[::-1]), got(x), atol=1e-9, label="pooling after reversing the frame order")


@check("attentive_stats_pooling matches the worked example: weighted mean 3.75 and weighted std 2.2776")
def test_attentive_pooling(sub):
    got = has(sub, "attentive_stats_pooling")
    pool = has(sub, "stats_pooling")
    frames = np.array([[1.0], [2.0], [6.0]])
    # Attention weights are 0.25, 0.25, 0.5. A wrong implementation subtracts the
    # unweighted mean 3 instead of the weighted mean 3.75, and gets std 3.2016.
    close(got(frames, np.array([0.0, 0.0, math.log(2)])), [3.75, math.sqrt(19.25 - 3.75**2)], atol=1e-9,
          label="weighted mean and weighted std with weights 0.25, 0.25, 0.5")
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 2.0, size=(40, 5))
    close(got(x, np.zeros(40)), pool(x), atol=1e-9, label="equal attention scores reduce to plain statistics pooling")
    logits = rng.normal(size=(40, 5))
    close(got(x, logits + 7.0), got(x, logits), atol=1e-9, label="adding a constant to every score before the softmax")
    alpha = np.exp(logits) / np.exp(logits).sum(0)
    mu = (alpha * x).sum(0)
    var = (alpha * (x - mu) ** 2).sum(0)
    close(got(x, logits), np.concatenate([mu, np.sqrt(var)]), atol=1e-9, label="random frames with per-channel scores")
    shape(got(x, logits[:, 0]), (10,), label="one score per frame shared by all channels")
    flat = got(np.full((6, 2), 3.0), rng.normal(size=6))
    if not np.all(np.isfinite(flat)):
        fail(f"constant frames must give a finite std through the variance floor, got {flat}")
    almost(float(flat[2]), math.sqrt(1e-9), 1e-12, label="std of constant frames, which is sqrt(floor)")


@check("enroll normalizes each embedding, averages them, then normalizes the average")
def test_enroll(sub):
    got = has(sub, "enroll")
    # A wrong implementation averages the raw rows first, so the long row dominates:
    # it returns (0.0995, 0.9950) here instead of (0.7071, 0.7071).
    close(got(np.array([[1.0, 0.0], [0.0, 10.0]])), [math.sqrt(0.5), math.sqrt(0.5)], atol=1e-9,
          label="enrolling a length-1 and a length-10 embedding at right angles")
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 8))
    out = got(x)
    shape(out, (8,), label="one enrollment vector")
    almost(float(np.linalg.norm(out)), 1.0, 1e-9, label="length of the enrollment vector")
    close(got(x[::-1]), out, atol=1e-9, label="enrolling the same clips in reverse order")
    close(got(x * np.array([[1.0], [3.0], [0.2], [9.0]])), out, atol=1e-9, label="rescaling each clip's embedding")
    test = rng.normal(size=8)
    test /= np.linalg.norm(test)
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    almost(float(np.mean(unit @ test)), float(np.linalg.norm(unit.mean(0)) * (out @ test)), 1e-9,
           label="average of the four cosines, which equals the length of the mean times the enrollment cosine")


@check("as_norm keeps only the top-K cohort scores on each side: 5.2052 on the worked example, not 3.2252")
def test_as_norm(sub):
    got = has(sub, "as_norm")
    enroll_cohort = np.array([0.30, 0.20, 0.10, 0.05, 0.0, -0.10])
    test_cohort = np.array([0.20, 0.15, 0.10, 0.0, -0.05, -0.10])
    top_e_std, top_t_std = math.sqrt(0.02 / 3), math.sqrt(0.0050 / 3)
    want = 0.5 * ((0.45 - 0.20) / top_e_std + (0.45 - 0.15) / top_t_std)
    # A wrong implementation uses the whole cohort for mu and sigma, and returns 3.2252.
    almost(float(got(0.45, enroll_cohort, test_cohort, 3)), want, 1e-9, label="AS-norm of 0.45 with K = 3")
    whole_e, whole_t = enroll_cohort, test_cohort
    all_k = 0.5 * ((0.45 - whole_e.mean()) / whole_e.std() + (0.45 - whole_t.mean()) / whole_t.std())
    almost(float(got(0.45, enroll_cohort, test_cohort, 6)), float(all_k), 1e-9, label="AS-norm with K equal to the cohort size")
    almost(float(got(0.45, enroll_cohort[::-1], test_cohort[::-1], 3)), want, 1e-9, label="AS-norm with the cohort order reversed")
    almost(float(got(0.45, test_cohort, enroll_cohort, 3)), 0.5 * ((0.45 - 0.15) / top_t_std + (0.45 - 0.20) / top_e_std),
           1e-9, label="AS-norm with the two sides swapped, which is symmetric")
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 0.1, 200), rng.normal(0.05, 0.2, 200)
    got_k = got(0.3, a, b, 20)
    ta, tb = np.sort(a)[-20:], np.sort(b)[-20:]
    almost(float(got_k), float(0.5 * ((0.3 - ta.mean()) / ta.std() + (0.3 - tb.mean()) / tb.std())), 1e-9,
           label="AS-norm of 0.3 against two random 200-score cohorts with K = 20")
