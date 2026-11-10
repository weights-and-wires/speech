#!/usr/bin/env python3
"""Compute the WavLM and SUPERB worked examples and draw the figures.

The numbers in content/lessons/039-wavlm-and-ssl-benchmarks.md for the
denoising mix, the relative position buckets, the gated bias, and the SUPERB
weighted sum come from this script. Run it from the repo root:

    python3 scripts/figures/039-wavlm-and-ssl-benchmarks.py

It prints the worked examples, then the SVG for Figure 1 (the denoising mix),
Figure 2 (offset to bucket), and Figure 3 (softmax weights against raw weights).
"""

from __future__ import annotations

import math

import numpy as np

N_BUCKETS, MAX_DISTANCE = 320, 800


def bucket(rel: np.ndarray, n: int, m: int) -> np.ndarray:
    half, exact = n // 2, n // 4
    d = np.abs(rel)
    large = exact + (np.log(np.maximum(d, 1) / exact) / np.log(m / exact) * (half - exact)).astype(int)
    return np.where(d < exact, d, np.minimum(large, half - 1)) + np.where(rel > 0, half, 0)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def worked_examples() -> None:
    primary, secondary = np.ones(6), np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0])
    gain = 10 * math.log10(4)
    scale = math.sqrt(np.sum(primary[2:4] ** 2) / (10 ** (gain / 10) * np.sum(secondary[2:4] ** 2)))
    mixed = primary.copy()
    mixed[2:4] += scale * secondary[2:4]
    print(f"mix: gain {gain:.2f} dB, scale {scale:.4f}, mixed {mixed.tolist()}")
    print("buckets, n=8, max=16, offsets -20 -1 0 1 4 20:", bucket(np.array([-20, -1, 0, 1, 4, 20]), 8, 16).tolist())
    print("buckets, n=320, max=800, offsets 0 1 79 80 160 800 -3 3:",
          bucket(np.array([0, 1, 79, 80, 160, 800, -3, 3]), 320, 800).tolist())
    for name, qu, qv in (("query 0", 0.0, 0.0), ("query 1", 2.0, -2.0)):
        gu, gr = sigmoid(qu), sigmoid(qv)
        s = 1 + gu + (1 - gu) * 1.0 * gr
        print(f"{name}: g_u {gu:.4f}, g_r {gr:.4f}, scale {s:.4f}, bias 2.0 -> {2 * s:.4f}")
    w = np.array([0.0, 0.0, math.log(2)])
    p = np.exp(w) / np.exp(w).sum()
    print(f"weighted sum: softmax {p.tolist()}, mix {p @ np.array([4.0, 8.0, 12.0]):.4f}, raw {w @ np.array([4.0, 8.0, 12.0]):.4f}")


def bucket_figure() -> str:
    x0, x1, y0, y1, lo, hi = 70.0, 520.0, 220.0, 30.0, -900, 900

    def sx(r): return x0 + (r - lo) / (hi - lo) * (x1 - x0)
    def sy(b): return y0 - b / (N_BUCKETS - 1) * (y0 - y1)

    out = ['<svg viewBox="0 0 560 280" role="img" aria-label="Relative position bucket against the signed offset i minus j for 320 buckets and a maximum distance of 800">']
    for r in (-800, -400, 0, 400, 800):
        out.append(f'<line x1="{sx(r):.1f}" y1="30" x2="{sx(r):.1f}" y2="220" class="stroke-grid"/>')
        out.append(f'<text x="{sx(r):.1f}" y="236" text-anchor="middle" class="fig-small">{r}</text>')
    for b in (0, 80, 160, 240, 319):
        out.append(f'<line x1="70" y1="{sy(b):.1f}" x2="520" y2="{sy(b):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="62" y="{sy(b) + 4:.1f}" text-anchor="end" class="fig-small">{b}</text>')
    for side in (np.arange(lo, 1), np.arange(1, hi + 1)):
        pts = " ".join(f"{sx(r):.1f},{sy(b):.1f}" for r, b in zip(side[::6], bucket(side, N_BUCKETS, MAX_DISTANCE)[::6]))
        out.append(f'<polyline points="{pts}" class="stroke-b" fill="none"/>')
    for r, dx, dy, anchor in ((-3, -10, -10, "end"), (3, 10, 4, "start"), (800, -8, 16, "end")):
        b = int(bucket(np.array([r]), N_BUCKETS, MAX_DISTANCE)[0])
        out.append(f'<circle cx="{sx(r):.1f}" cy="{sy(b):.1f}" r="5" class="fill-a"/>')
        out.append(f'<text x="{sx(r) + dx:.1f}" y="{sy(b) + dy:.1f}" text-anchor="{anchor}" class="fig-small">offset {r:+d}: bucket {b}</text>')
    out.append('<text x="295" y="258" text-anchor="middle">offset i - j (frames)</text>')
    out.append('<text x="18" y="125" transform="rotate(-90 18 125)" text-anchor="middle">bucket index</text>')
    out.append("</svg>")
    return "\n".join(out)


LAYER_VALUES = np.array([4.0, 8.0, 12.0])
LAYER_SCORES = np.array([0.0, 0.0, math.log(2)])


def mix_figure() -> str:
    primary, secondary = np.ones(6), np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0])
    gain = 10 * math.log10(4)
    scale = math.sqrt(np.sum(primary[2:4] ** 2) / (10 ** (gain / 10) * np.sum(secondary[2:4] ** 2)))
    added = np.zeros(6)
    added[2:4] = scale * secondary[2:4]
    mixed = primary + added
    rows = (
        ("primary", primary, "fill-a"),
        (f"secondary x {scale:.4f}", added, "fill-b"),
        ("mixed", mixed, "fill-c"),
    )
    out = ['<svg viewBox="0 0 600 330" role="img" aria-label="Six samples of a primary signal, the scaled secondary signal in a two-sample span, and their sum">']
    out.append('<rect x="293" y="30" width="129" height="285" class="fill-soft-d"/>')
    out.append('<text x="357" y="24" text-anchor="middle" class="fig-small">span</text>')
    for r, (name, values, fill) in enumerate(rows):
        base = 80 + 95 * r
        out.append(f'<text x="20" y="{base + 4}" class="fig-small">{name}</text>')
        out.append(f'<line x1="170" y1="{base}" x2="580" y2="{base}" class="stroke-muted"/>')
        for i, v in enumerate(values):
            cx = 195 + 65 * i
            h = abs(v) * 36
            top = base - h if v >= 0 else base
            if h > 0:
                out.append(f'<rect x="{cx - 15}" y="{top:.1f}" width="30" height="{h:.1f}" class="{fill}"/>')
            ly = base - h - 6 if v >= 0 else base + h + 14
            if h == 0:
                ly = base - 6
            out.append(f'<text x="{cx}" y="{ly:.1f}" text-anchor="middle" class="fig-small fig-mono">{v:g}</text>')
    out.append('<text x="380" y="326" text-anchor="middle" class="fig-small">sample index 0 to 5 (left to right)</text>')
    out.append("</svg>")
    return "\n".join(out)


def superb_figure() -> str:
    soft = np.exp(LAYER_SCORES) / np.exp(LAYER_SCORES).sum()
    panels = (
        ("softmax weights", soft, float(soft @ LAYER_VALUES), 40, "fill-b"),
        ("raw scores as weights", LAYER_SCORES, float(LAYER_SCORES @ LAYER_VALUES), 340, "fill-bad"),
    )
    out = ['<svg viewBox="0 0 600 290" role="img" aria-label="Weights on three layers with values 4, 8, and 12, as a softmax and as raw scores, and the mix each gives">']
    for name, w, mix, x0, fill in panels:
        out.append(f'<text x="{x0}" y="22" class="fig-bold">{name}: mix {mix:.4f}</text>')
        out.append(f'<line x1="{x0}" y1="220" x2="{x0 + 230}" y2="220" class="stroke-muted"/>')
        for i, (weight, value) in enumerate(zip(w, LAYER_VALUES)):
            cx = x0 + 30 + 80 * i
            h = float(weight) * 160
            out.append(f'<rect x="{cx - 22}" y="{220 - h:.1f}" width="44" height="{h:.1f}" class="{fill}"/>')
            out.append(f'<text x="{cx}" y="{220 - h - 6:.1f}" text-anchor="middle" class="fig-small fig-mono">{weight:.4f}</text>')
            out.append(f'<text x="{cx}" y="238" text-anchor="middle" class="fig-small">layer value {value:g}</text>')
    out.append('<text x="300" y="276" text-anchor="middle" class="fig-small">bar height is the weight on the layer; the weights of the softmax sum to 1</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    worked_examples()
    for name, svg in (("Figure 1", mix_figure()), ("Figure 2", bucket_figure()), ("Figure 3", superb_figure())):
        print(f"--- {name}")
        print(svg)
