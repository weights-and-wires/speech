#!/usr/bin/env python3
"""Compute the numbers in content/lessons/045-neural-diarization.md.

Run it from the repo root:

    python3 scripts/figures/045-neural-diarization.py

It prints the PIT worked example, the EDA activity example, the overlap DER
breakdown, and the target-speaker cosines, then the SVG for Figure 1 (the
timeline) and Figure 2 (the PIT loss table).
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(p, y):
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# PIT: 3 frames, 2 speakers. The network puts speaker A in slot 1 and B in slot 0.
Y = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
P = np.array([[0.1, 0.9], [0.8, 0.9], [0.9, 0.1]])
fixed = bce(P, Y)
swapped = bce(P, Y[:, ::-1])
print("PIT: fixed-order loss %.4f, swapped loss %.4f" % (fixed, swapped))
for perm in itertools.permutations(range(2)):
    print("  perm", perm, "loss %.4f" % bce(P, Y[:, list(perm)]))

# EDA: activity = sigmoid(E a^T)
E = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
A = np.array([[3.0, -1.0], [-1.0, 3.0]])
act = sigmoid(E @ A.T)
print("EDA activity per frame (rows) and attractor (columns):")
print(np.round(act, 3))
exist = np.array([0.93, 0.81, 0.20])
print("attractor existence", exist, "-> count", int(np.argmax(exist < 0.5)))

# Overlap DER components, 14 frames.
T = 14
ref = np.zeros((2, T), dtype=bool)
ref[0, 0:8] = True
ref[1, 5:12] = True
hyp = np.zeros((2, T), dtype=bool)
hyp[1, 0:10] = True  # slot 1 is A, and it wrongly keeps talking in frames 8 and 9
hyp[0, 10:14] = True  # slot 0 is B
table = ref.astype(int) @ hyp.T.astype(int)
r, h = linear_sum_assignment(-table)
mapping = dict(zip(r, h))
ok = sum((ref[i] & hyp[j]) for i, j in mapping.items())
n_ref, n_hyp = ref.sum(0), hyp.sum(0)
miss = np.maximum(n_ref - n_hyp, 0).sum()
fa = np.maximum(n_hyp - n_ref, 0).sum()
conf = (np.minimum(n_ref, n_hyp) - ok).sum()
print("mapping ref->hyp", mapping, "miss", miss, "fa", fa, "conf", conf,
      "total", n_ref.sum(), "DER %.4f" % ((miss + fa + conf) / n_ref.sum()),
      "wrong denominator (speech frames)", int((n_ref > 0).sum()),
      "-> %.4f" % ((miss + fa + conf) / (n_ref > 0).sum()))
print("overlap frames", np.flatnonzero(n_ref >= 2).tolist())

# Target speaker
enroll = np.array([1.0, 0.0])
frames = np.array([[3.0, 0.0], [2.0, 2.0], [0.0, 5.0]])
cos = frames @ enroll / (np.linalg.norm(frames, axis=1) * np.linalg.norm(enroll))
print("cosines", np.round(cos, 3), "mask at 0.5", (cos >= 0.5).tolist())


def rect(x, y, w, cls):
    return f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="3" class="{cls}"/>'


def figure_timeline():
    x0, w = 90, 34
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Reference activity for two speakers over 14 frames and a system output with a fixed order, with the overlap frames 5 to 7 marked">']
    rows = [("A", ref[0], "fill-a", 50), ("B", ref[1], "fill-c", 80)]
    out.append('<text x="20" y="38" class="fig-small fig-bold">reference</text>')
    for name, row, cls, y in rows:
        out.append(f'<text x="70" y="{y + 16}" text-anchor="end" class="fig-small">{name}</text>')
        for t in range(T):
            out.append(rect(x0 + t * 32, y, 30, cls if row[t] else "fill-sunk"))
    out.append('<text x="20" y="148" class="fig-small fig-bold">system slots</text>')
    for k, (name, row, cls, y) in enumerate([("0", hyp[0], "fill-c", 160), ("1", hyp[1], "fill-a", 190)]):
        out.append(f'<text x="70" y="{y + 16}" text-anchor="end" class="fig-small">slot {name}</text>')
        for t in range(T):
            out.append(rect(x0 + t * 32, y, 30, cls if row[t] else "fill-sunk"))
    out.append(f'<rect x="{x0 + 5 * 32 - 2}" y="46" width="{3 * 32}" height="60" rx="4" class="stroke-bad" fill="none"/>')
    out.append(f'<text x="{x0 + 5 * 32 + 46}" y="124" text-anchor="middle" class="fig-small fig-bold">overlap</text>')
    for t in range(0, T, 2):
        out.append(f'<text x="{x0 + t * 32 + 15}" y="232" text-anchor="middle" class="fig-small">{t}</text>')
    out.append('<text x="300" y="248" text-anchor="middle" class="fig-small">frame index</text>')
    out.append("</svg>")
    return "\n".join(out)


def figure_pit():
    scale = 60.0  # units per nat
    out = ['<svg viewBox="0 0 600 190" role="img" aria-label="Two bars: the fixed-order loss is 1.590 and the loss under the swapped assignment is 0.125">']
    out.append('<line x1="150" y1="30" x2="150" y2="140" class="stroke-muted"/>')
    for i, (name, val, cls) in enumerate([("fixed order (0, 1)", fixed, "fill-bad"), ("swapped (1, 0)", swapped, "fill-b")]):
        y = 40 + i * 50
        out.append(f'<text x="142" y="{y + 20}" text-anchor="end" class="fig-small">{name}</text>')
        out.append(f'<rect x="150" y="{y}" width="{val * scale:.1f}" height="30" rx="3" class="{cls}"/>')
        out.append(f'<text x="{150 + val * scale + 8:.1f}" y="{y + 20}" class="fig-small fig-bold">{val:.3f}</text>')
    out.append('<text x="150" y="170" class="fig-small">mean BCE per frame and speaker, in nats; PIT keeps the smaller bar</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    print(figure_timeline())
    print(figure_pit())
