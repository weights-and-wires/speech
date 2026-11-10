#!/usr/bin/env python3
"""Compute the worked examples of content/lessons/065-voice-conversion-and-spoofing.md.

Run it from the repo root:

    python3 scripts/figures/voice-conversion-and-spoofing.py

It prints the kNN-VC hand example (Figure 1), the effect of k on content and
speaker (Figure 2), and the error-rate table and interpolation (Figure 3),
followed by the SVG for Figures 2 and 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend/app/runner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "content/exercises/065-voice-conversion-and-spoofing"))
import solution as sol  # noqa: E402

# Figure 1: one source frame against four matching frames ------------------------------
x = np.array([[1.0, 0.0]])
r = np.array([[0.9, 0.1], [0.2, 1.0], [1.0, -0.2], [-1.0, 0.0]])
cos = (x / np.linalg.norm(x, axis=1, keepdims=True)) @ (r / np.linalg.norm(r, axis=1, keepdims=True)).T
print("cosines", np.round(cos[0], 3), "k=2 output", sol.knn_vc(x, r, 2)[0], "k=1 output", sol.knn_vc(x, r, 1)[0])

# Figure 2: content and speaker against k ------------------------------------------------
rng = np.random.default_rng(7)
phones = rng.normal(size=(8, 32)) * 2.0
speakers = rng.normal(size=(2, 32))


def utter(spk, n):
    labels = rng.integers(0, 8, size=n)
    return phones[labels] + speakers[spk] + 0.5 * rng.normal(size=(n, 32)), labels


src, src_labels = utter(0, 120)
match, _ = utter(1, 400)
KS = [1, 4, 16, 64, 128, 400]
content, spread = [], []
for k in KS:
    out = sol.knn_vc(src, match, k)
    phone_of = np.argmin(((out[:, None, :] - speakers[1][None, None] - phones[None]) ** 2).sum(-1), axis=1)
    content.append(float(np.mean(phone_of == src_labels)))
    spread.append(float(out.std(axis=0).mean()))
print("k", KS, "content kept", np.round(content, 3), "feature spread", np.round(spread, 2))

# Figure 3: error rates of 3 bona fide and 4 spoof scores ------------------------------------
bona = np.array([0.9, 0.5, 0.4])
spoof = np.array([0.6, 0.3, 0.1, 0.0])
th = np.unique(np.concatenate([bona, spoof]))
th = np.append(th, th[-1] + 1.0)
frr, far = sol.error_rates(bona, spoof, th)
for t, f, a in zip(th, frr, far):
    print(f"threshold {t:.2f}: FRR {f:.3f}, FAR {a:.3f}")
eer, eer_th = sol.equal_error_rate(bona, spoof)
print("EER", eer, "at", eer_th)


def k_figure() -> str:
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Fraction of converted frames that keep the source phone for k from 1 to 400">']
    out.append('  <line x1="70" y1="180" x2="560" y2="180" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="180" class="stroke-muted"/>')
    out.append('  <text x="16" y="105" class="fig-small" transform="rotate(-90 16 105)" text-anchor="middle">frames keeping the phone</text>')
    for i, (k, c) in enumerate(zip(KS, content)):
        x0 = 90 + 78 * i
        h = 150 * c
        out.append(f'  <rect x="{x0}" y="{180 - h:.1f}" width="50" height="{max(h, 1):.1f}" class="fill-soft-b stroke-b"/>')
        out.append(f'  <text x="{x0 + 25}" y="{180 - h - 6:.1f}" text-anchor="middle" class="fig-small fig-bold">{c:.0%}</text>')
        out.append(f'  <text x="{x0 + 25}" y="198" text-anchor="middle" class="fig-small">{k}</text>')
    out.append('  <text x="315" y="222" text-anchor="middle" class="fig-small">k, the number of neighbors averaged (matching set: 400 frames)</text>')
    out.append("</svg>")
    return "\n".join(out)


def eer_figure() -> str:
    # x: threshold 0 to 1; y: rate 0 to 1. Staircase curves plus the crossing.
    X = lambda v: 70 + 480 * v
    Y = lambda v: 190 - 150 * v
    # The rates at every distinct score, joined by straight lines: the curve equal_error_rate interpolates along.
    fine = np.unique(np.concatenate([[0.0, 1.0], bona, spoof]))
    f_r, f_a = sol.error_rates(bona, spoof, fine)

    def path(vals):
        return "M" + " L".join(f"{X(t):.1f} {Y(v):.1f}" for t, v in zip(fine, vals))

    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="False rejection and false acceptance rates against threshold for three bona fide and four spoof scores, crossing at 0.25 near threshold 0.475">']
    out.append(f'  <line x1="70" y1="190" x2="550" y2="190" class="stroke-muted"/><line x1="70" y1="40" x2="70" y2="190" class="stroke-muted"/>')
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'  <line x1="70" y1="{Y(v):.1f}" x2="550" y2="{Y(v):.1f}" class="stroke-grid"/><text x="62" y="{Y(v) + 4:.1f}" text-anchor="end" class="fig-small">{v:.2f}</text>')
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'  <text x="{X(v):.1f}" y="208" text-anchor="middle" class="fig-small">{v:.1f}</text>')
    out.append(f'  <path d="{path(f_r)}" fill="none" class="stroke-b" stroke-width="2"/>')
    out.append(f'  <path d="{path(f_a)}" fill="none" class="stroke-a" stroke-width="2"/>')
    for t, v in zip(fine, f_r):
        out.append(f'  <circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3" class="fill-b"/>')
    for t, v in zip(fine, f_a):
        out.append(f'  <circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="3" class="fill-a"/>')
    out.append(f'  <circle cx="{X(eer_th):.1f}" cy="{Y(eer):.1f}" r="5" class="fill-c"/>')
    out.append(f'  <line x1="{X(eer_th) + 4:.1f}" y1="{Y(eer) + 4:.1f}" x2="{X(0.52):.1f}" y2="{Y(0.1) - 8:.1f}" class="stroke-muted"/>')
    out.append(f'  <text x="{X(0.52):.1f}" y="{Y(0.1):.1f}" class="fig-small fig-bold">EER {eer:.2f} at threshold {eer_th:.3f}</text>')
    out.append(f'  <text x="{X(0.62):.1f}" y="{Y(0.9):.1f}" class="fig-small">false rejection (bona fide below)</text>')
    out.append(f'  <text x="{X(0.05):.1f}" y="{Y(0.8):.1f}" class="fig-small">false acceptance (spoof at or above)</text>')
    out.append('  <text x="310" y="232" text-anchor="middle" class="fig-small">threshold (countermeasure score)</text>')
    out.append('  <text x="16" y="115" class="fig-small" transform="rotate(-90 16 115)" text-anchor="middle">error rate</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    print(k_figure())
    print(eer_figure())
