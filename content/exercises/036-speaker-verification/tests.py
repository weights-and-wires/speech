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
