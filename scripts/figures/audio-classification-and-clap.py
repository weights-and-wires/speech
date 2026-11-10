#!/usr/bin/env python3
"""Compute the worked examples of content/lessons/064-audio-classification-and-clap.md.

Run it from the repo root:

    python3 scripts/figures/audio-classification-and-clap.py

It prints the CLAP loss terms (Figure 2), the tagging-loss example, the
average-precision table (Figure 3), and the zero-shot cosines (Figure 4),
followed by the SVG for Figures 3 and 4.
"""

from __future__ import annotations

import numpy as np

# Figure 2: the CLAP loss on a 2 x 2 batch -------------------------------------------
a = np.array([[1.0, 0.0], [0.0, 1.0]])
t = np.array([[0.8, 0.6], [0.0, 1.0]])
S = a @ t.T / 0.1
rows = [np.log(np.exp(S[i]).sum()) - S[i, i] for i in range(2)]
cols = [np.log(np.exp(S[:, i]).sum()) - S[i, i] for i in range(2)]
print("CLAP S =", S.tolist(), "rows", np.round(rows, 5), "cols", np.round(cols, 5), "loss", round((sum(rows) + sum(cols)) / 4, 4))

# Multi-label tagging: sigmoid + BCE against softmax -------------------------------------
x = np.array([2.0, 1.0, -3.0])
y = np.array([1.0, 1.0, 0.0])
s = 1 / (1 + np.exp(-x))
bce = -(y * np.log(s) + (1 - y) * np.log(1 - s))
soft = np.exp(x) / np.exp(x).sum()
print("sigmoid", np.round(s, 4), "bce per class", np.round(bce, 4), "mean", round(bce.mean(), 4))
print("softmax", np.round(soft, 4))

# Figure 3: average precision ----------------------------------------------------------
labels = np.array([1, 0, 1, 1, 0, 0])  # in rank order
prec = np.cumsum(labels) / np.arange(1, 7)
ap = prec[labels == 1].mean()
print("precision@rank", np.round(prec, 4), "AP", round(ap, 4), "all-ranks mean", round(prec.mean(), 4))


def ap_figure() -> str:
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="Precision at each of six ranks; bars at the three positive ranks are highlighted and average to 0.806">']
    out.append('  <line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>')
    out.append('  <line x1="60" y1="40" x2="60" y2="190" class="stroke-muted"/>')
    out.append('  <text x="14" y="120" class="fig-small" transform="rotate(-90 14 120)" text-anchor="middle">precision at rank</text>')
    for i, (lab, p) in enumerate(zip(labels, prec)):
        x0 = 85 + 80 * i
        h = 140 * p
        cls = "fill-soft-b stroke-b" if lab else "fill-sunk stroke-grid"
        out.append(f'  <rect x="{x0}" y="{190 - h:.1f}" width="50" height="{h:.1f}" class="{cls}"/>')
        out.append(f'  <text x="{x0 + 25}" y="{190 - h - 6:.1f}" text-anchor="middle" class="fig-small{" fig-bold" if lab else ""}">{p:.2f}</text>')
        out.append(f'  <text x="{x0 + 25}" y="208" text-anchor="middle" class="fig-small">{i + 1}</text>')
        out.append(f'  <text x="{x0 + 25}" y="224" text-anchor="middle" class="fig-small">{"dog" if lab else "other"}</text>')
    out.append('  <text x="310" y="244" text-anchor="middle" class="fig-small">rank, by descending score</text>')
    out.append(f'  <text x="310" y="262" text-anchor="middle" class="fig-small fig-bold">AP = ({prec[0]:.2f} + {prec[2]:.2f} + {prec[3]:.2f}) / 3 = {ap:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


# Figure 4: zero-shot ----------------------------------------------------------------
audio = np.array([1.0, 1.2])
prompts = {"the sound of a dog": np.array([4.0, 0.0]), "the sound of rain": np.array([0.0, 1.0]), "the sound of a siren": np.array([-1.0, -1.0])}
unit = lambda v: v / np.linalg.norm(v)
for name, p in prompts.items():
    print(f"{name}: dot {audio @ p:.2f}, cosine {unit(audio) @ unit(p):.3f}")


def zero_shot_figure() -> str:
    cx, cy, r = 160, 125, 90
    pt = lambda v: (cx + r * v[0], cy - r * v[1])
    out = ['<svg viewBox="0 0 600 250" role="img" aria-label="Unit vectors for a clip and three prompts; the clip points closest to the rain prompt">']
    out.append('  <defs><marker id="c064-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-a"/></marker>')
    out.append('  <marker id="c064-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-b"/></marker></defs>')
    out.append(f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" class="stroke-grid"/>')
    out.append(f'  <line x1="{cx - r - 15}" y1="{cy}" x2="{cx + r + 15}" y2="{cy}" class="stroke-grid"/><line x1="{cx}" y1="{cy - r - 15}" x2="{cx}" y2="{cy + r + 15}" class="stroke-grid"/>')
    for (name, p), short in zip(prompts.items(), ("dog", "rain", "siren")):
        x1, y1 = pt(unit(p))
        out.append(f'  <line x1="{cx}" y1="{cy}" x2="{x1:.1f}" y2="{y1:.1f}" class="stroke-b" marker-end="url(#c064-arrow-b)"/>')
        lx = x1 + (8 if x1 >= cx else -8)
        out.append(f'  <text x="{lx:.0f}" y="{y1 + (14 if y1 > cy else -6):.0f}" class="fig-small" text-anchor="{"start" if x1 >= cx else "end"}">{short}</text>')
    x1, y1 = pt(unit(audio))
    out.append(f'  <line x1="{cx}" y1="{cy}" x2="{x1:.1f}" y2="{y1:.1f}" class="stroke-a" stroke-width="2.5" marker-end="url(#c064-arrow)"/>')
    out.append(f'  <text x="{x1 + 8:.0f}" y="{y1 - 4:.0f}" class="fig-small fig-bold">clip</text>')
    out.append('  <text x="330" y="60" class="fig-small fig-bold">prompt</text><text x="470" y="60" class="fig-small fig-bold">dot</text><text x="530" y="60" class="fig-small fig-bold">cosine</text>')
    for n, (name, p) in enumerate(prompts.items()):
        y = 85 + 30 * n
        best = unit(audio) @ unit(p) == max(unit(audio) @ unit(q) for q in prompts.values())
        out.append(f'  <text x="330" y="{y}" class="fig-small{" fig-bold" if best else ""}">{name}</text>')
        out.append(f'  <text x="470" y="{y}" class="fig-small">{audio @ p:.2f}</text><text x="530" y="{y}" class="fig-small{" fig-bold" if best else ""}">{unit(audio) @ unit(p):.3f}</text>')
    out.append('  <text x="330" y="190" class="fig-small">Raw dot products pick the dog prompt,</text><text x="330" y="208" class="fig-small">because its vector is long. Cosines pick rain.</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    print(ap_figure())
    print(zero_shot_figure())
