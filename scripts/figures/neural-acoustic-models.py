#!/usr/bin/env python3
"""Compute the TDNN worked examples and draw the subsampling figure.

The numbers in content/lessons/028-neural-acoustic-models.md come from this
script. Run it from the repo root:

    python3 scripts/figures/neural-acoustic-models.py

It prints the receptive fields, the frames each layer must compute with and
without subsampling, the hybrid scaling example, and the SVG for Figure 4.
"""

from __future__ import annotations

import numpy as np

# Layer contexts, bottom layer first (Peddinti, Povey, Khudanpur, Interspeech 2015, Table 1).
SMALL = [[-2, -1, 0, 1, 2], [-1, 2], [-3, 3], [0]]
PAPER_SUB = [[-2, -1, 0, 1, 2], [-1, 2], [-3, 3], [-7, 2], [0]]
PAPER_DENSE = [[-2, -1, 0, 1, 2], [-1, 0, 1, 2], list(range(-3, 4)), list(range(-7, 3)), [0]]


def receptive_field(contexts):
    return (sum(max(0, -min(o)) for o in contexts), sum(max(0, max(o)) for o in contexts))


def needed(contexts):
    """Frames each layer's output must exist at, for the single output frame t = 0.

    Returns a list from the top: index 0 is the final layer's output, the last
    entry is the input features.
    """
    frames = {0}
    out = [sorted(frames)]
    for offsets in reversed(contexts):
        frames = {t + o for t in frames for o in offsets}
        out.append(sorted(frames))
    return out


def svg(sub, dense):
    names = ["layer 4 out", "layer 3 out", "layer 2 out", "layer 1 out", "input"]
    lines = ['<svg viewBox="0 0 600 270" role="img" aria-label="Frames a TDNN computes at each layer for one output frame: 1, 2, 4, and 7 with subsampling, against 1, 10, 16, and 19 without">']
    x = lambda o: 80 + (o + 13) * 20
    for row, name in enumerate(names):
        y = 40 + row * 42
        s_set, d_set = set(sub[row + 1]), set(dense[row + 1])
        lines.append(f'  <text x="8" y="{y + 4}" class="fig-small">{name}</text>')
        for o in sorted(d_set):
            cls = "fill-b stroke-b" if o in s_set else "fill-sunk stroke-muted"
            lines.append(f'  <circle cx="{x(o)}" cy="{y}" r="6" class="{cls}"/>')
    lines.append(f'  <line x1="{x(0)}" y1="22" x2="{x(0)}" y2="226" class="stroke-muted dashed"/>')
    for o in (-13, 0, 9):
        lines.append(f'  <text x="{x(o)}" y="246" text-anchor="middle" class="fig-small">{o:+d}</text>')
    lines.append('  <text x="310" y="264" text-anchor="middle" class="fig-small">frame offset from the output frame</text>')
    lines.append("</svg>")
    return "\n".join(lines)


if __name__ == "__main__":
    print("small stack", SMALL, "->", receptive_field(SMALL))
    print("max-over-layers (wrong)", (max(max(0, -min(o)) for o in SMALL), max(max(0, max(o)) for o in SMALL)))
    print("paper stack", PAPER_SUB, "->", receptive_field(PAPER_SUB))
    sub, dense = needed(PAPER_SUB), needed(PAPER_DENSE)
    for name, s, d in zip(["L4 out", "L3 out", "L2 out", "L1 out", "input"], sub[1:], dense[1:]):
        print(f"{name}: subsampled {len(s)} {s}   contiguous {len(d)} ({d[0]}..{d[-1]})")
    print("total activations (layers 1-4):", sum(len(s) for s in sub[1:-1]), "vs", sum(len(d) for d in dense[1:-1]))
    print("layers for the small stack, by hand:", needed(SMALL)[-1][0], needed(SMALL)[-1][-1])
    post, prior = np.array([0.5, 0.3, 0.2]), np.array([0.8, 0.15, 0.05])
    print("scaled likelihoods", post / prior)
    print(svg(sub, dense))
