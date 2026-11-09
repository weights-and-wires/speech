#!/usr/bin/env python3
"""Compute the hybrid-model worked example and draw its likelihood curve.

The numbers in content/lessons/023-hybrid-acoustic-models.md come from this
script. Run it from the repo root:

    python3 scripts/figures/hybrid-acoustic-models.py

It prints the EM table, the tied-state gains, the scaled-likelihood path,
and the SVG for Figure 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


FLOOR = 1e-3
FRAMES = np.array([[-1.0], [0.0], [1.0], [5.0], [6.0], [7.0]])


def start(X: np.ndarray, n_components: int, floor: float = FLOOR):
    """Means at evenly spaced ranks of the first feature, shared variance."""
    n_rows = X.shape[0]
    order = np.argsort(X[:, 0], kind="mergesort")
    if n_components == 1:
        ranks = np.array([n_rows // 2])
    else:
        ranks = np.round(np.linspace(0, n_rows - 1, n_components)).astype(int)
    means = X[order[ranks]].copy()
    variance = np.maximum(X.var(axis=0), floor)
    variances = np.repeat(variance[None, :], n_components, axis=0)
    weights = np.full(n_components, 1.0 / n_components)
    return weights, means, variances


def log_likelihood_and_responsibilities(X, weights, means, variances):
    """Observed log-likelihood and the responsibility matrix, shape (N, K)."""
    n_dims = X.shape[1]
    diff2 = (X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
    log_gauss = -0.5 * (
        n_dims * np.log(2 * np.pi)
        + np.log(variances).sum(axis=1)[None, :]
        + diff2.sum(axis=2)
    )
    log_joint = np.log(weights)[None, :] + log_gauss
    log_norm = logsumexp(log_joint, axis=1)
    responsibilities = np.exp(log_joint - log_norm[:, None])
    return float(log_norm.sum()), responsibilities, log_gauss


def m_step(X, responsibilities, floor: float = FLOOR):
    """Maximum-likelihood weights, means, and diagonal variances."""
    n_rows = X.shape[0]
    counts = responsibilities.sum(axis=0)
    weights = counts / n_rows
    means = (responsibilities.T @ X) / counts[:, None]
    diff = X[:, None, :] - means[None, :, :]
    variances = (responsibilities[:, :, None] * diff**2).sum(axis=0) / counts[:, None]
    return weights, means, np.maximum(variances, floor), counts


def em_curve(X, n_steps: int):
    """Log-likelihood before the updates and after each of `n_steps` updates."""
    weights, means, variances = start(X, 2)
    curve = []
    snapshot = None
    for step in range(n_steps):
        log_likelihood, responsibilities, log_gauss = log_likelihood_and_responsibilities(
            X, weights, means, variances
        )
        curve.append(log_likelihood)
        if step == 0:
            snapshot = (log_gauss, responsibilities, means, variances)
        weights, means, variances, counts = m_step(X, responsibilities)
        if step == 0:
            snapshot = snapshot + (weights, means, variances, counts)
    log_likelihood, _, _ = log_likelihood_and_responsibilities(
        X, weights, means, variances
    )
    curve.append(log_likelihood)
    return np.array(curve), weights, means, variances, snapshot


def cluster_loglik(X, floor: float = 1e-6) -> float:
    """Sum of diagonal-Gaussian log densities at the maximum-likelihood fit."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 0:
        return 0.0
    n_rows, n_dims = X.shape
    variance = np.maximum(X.var(axis=0), floor)
    squared = ((X - X.mean(axis=0)) ** 2).sum(axis=0)
    return float(
        -0.5
        * (
            n_rows * (n_dims * np.log(2 * np.pi) + np.log(variance).sum())
            + (squared / variance).sum()
        )
    )


def figure_svg(curve: np.ndarray) -> str:
    """SVG of the EM log-likelihood. Coordinates are the computed curve."""
    # y maps -17 nats to 220 and -11 nats to 40: 30 units per nat.
    def y_of(log_likelihood: float) -> float:
        return 220.0 - (log_likelihood + 17.0) * 30.0

    def x_of(step: int) -> float:
        return 80.0 + step * 60.0

    points = " ".join(
        f"{x_of(step):.1f},{y_of(value):.1f}" for step, value in enumerate(curve)
    )
    circles = "\n".join(
        f'  <circle cx="{x_of(step):.1f}" cy="{y_of(value):.1f}" r="4" class="fill-b stroke-b"/>'
        for step, value in enumerate(curve)
    )
    grids = []
    for nats in (-16, -14, -12):
        y_pos = y_of(float(nats))
        grids.append(
            f'  <line x1="80.0" y1="{y_pos:.1f}" x2="560.0" y2="{y_pos:.1f}" class="stroke-grid"/>'
        )
        grids.append(
            f'  <text x="72.0" y="{y_pos + 4:.1f}" class="fig-small" text-anchor="end">{nats}</text>'
        )
    ticks = "\n".join(
        f'  <text x="{x_of(step):.1f}" y="246.0" class="fig-small" text-anchor="middle">{step}</text>'
        for step in range(0, 9, 2)
    )
    grid = "\n".join(grids)
    return f"""<svg viewBox="0 0 620 280" role="img" aria-label="Log-likelihood of six frames rising over eight EM steps and then flattening">
  <line x1="80.0" y1="40.0" x2="80.0" y2="220.0" class="stroke-muted"/>
  <line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-muted"/>
{grid}
  <polyline points="{points}" class="stroke-b" fill="none"/>
{circles}
{ticks}
  <text x="72.0" y="28.0" class="fig-small" text-anchor="start">log-likelihood (nats)</text>
  <text x="320.0" y="268.0" class="fig-small" text-anchor="middle">EM step</text>
</svg>"""


def main() -> None:
    curve, weights, means, variances, snapshot = em_curve(FRAMES, 8)
    log_gauss, responsibilities, means0, variances0, weights1, means1, variances1, counts = snapshot
    print("population variance", float(FRAMES.var()))
    print("start means", means0.ravel())
    print("start variances", variances0.ravel())
    print("log gauss frame 0", log_gauss[0])
    print("responsibilities\n", responsibilities)
    print("counts", counts)
    print("means after 1", means1.ravel())
    print("variances after 1", variances1.ravel())
    print("weights after 1", weights1)
    print("curve")
    for step, value in enumerate(curve):
        print(f"  {step}: {value:.10f}")
    print("final means", means.ravel(), "variances", variances.ravel(), "weights", weights)

    tied = np.array([[0.0], [0.2], [-0.2], [0.1], [3.8], [4.0], [4.2], [3.9]])
    vowel = np.array([True, True, True, True, False, False, False, False])
    nasal = np.array([True, True, False, False, True, True, False, False])
    parent = cluster_loglik(tied)
    print("L(parent)", parent)
    print("L(vowel)", cluster_loglik(tied[vowel]), cluster_loglik(tied[~vowel]))
    print("gain vowel", cluster_loglik(tied[vowel]) + cluster_loglik(tied[~vowel]) - parent)
    print("gain nasal", cluster_loglik(tied[nasal]) + cluster_loglik(tied[~nasal]) - parent)
    one = -0.5 * len(FRAMES) * (math.log(2 * math.pi) + math.log(float(FRAMES.var())) + 1)
    print("one gaussian", one)
    print(figure_svg(curve))


if __name__ == "__main__":
    main()
