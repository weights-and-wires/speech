#!/usr/bin/env python3
"""Compute the numbers and draw the figures for 037-efficient-speech-encoders.

Run it from the repo root:

    python3 scripts/figures/037-efficient-speech-encoders.py [--svg fig1|fig2]

Without `--svg` it prints the worked-example values the lesson quotes. With
`--svg` it prints that figure's SVG, drawn from the same numbers.
"""

from __future__ import annotations

import math
import sys

import numpy as np


def encoder_flops_parts(frames, dim, layers, subsample):
    """Return (linear, quadratic) FLOPs. One multiply-accumulate counts as 2 FLOPs."""
    t = frames // subsample
    linear = layers * 24 * t * dim * dim
    quadratic = layers * 4 * t * t * dim
    return linear, quadratic


def stage_params(channels, stages, separable):
    total = 1 * channels * 9 + channels
    for _ in range(stages - 1):
        if separable:
            total += channels * 9 + channels + channels * channels + channels
        else:
            total += channels * channels * 9 + channels
    return total


def downsample(x, logits, factor):
    t = x.shape[0]
    pad = (-t) % factor
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return np.einsum("tfd,f->td", x.reshape(-1, factor, x.shape[1]), w)


def ssm(a, b, c, x):
    h = np.zeros_like(a)
    ys = []
    for v in x:
        h = a * h + b * v
        ys.append(float(c @ h))
    return ys


def report():
    d = 256
    print("worked example: one layer, 1000 frames (10 s at 100 Hz), d = 256")
    for s in (4, 8):
        lin, quad = encoder_flops_parts(1000, d, 1, s)
        print(f"  subsample {s}: T={1000 // s} linear={lin:,} quadratic={quad:,} total={lin + quad:,}")
    t4 = sum(encoder_flops_parts(1000, d, 1, 4))
    t8 = sum(encoder_flops_parts(1000, d, 1, 8))
    print(f"  ratio 4x/8x = {t4 / t8:.3f}")
    for secs in (10, 60):
        r = sum(encoder_flops_parts(secs * 100, d, 1, 4)) / sum(encoder_flops_parts(secs * 100, d, 1, 8))
        print(f"  ratio for {secs} s: {r:.3f}")
    print("subsampler parameters, 256 channels, 3 stages")
    print(f"  dense {stage_params(256, 3, False):,}  separable {stage_params(256, 3, True):,}"
          f"  ratio {stage_params(256, 3, False) / stage_params(256, 3, True):.2f}")
    print("frames at stage rates for 10 s:", [math.ceil(10 * r) for r in (50, 25, 12.5, 6.25, 12.5, 25)])
    x = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])
    print("downsample 5 frames, factor 2, logits [0, ln 3]:",
          downsample(x, np.array([0.0, math.log(3)]), 2).ravel())
    a, b, c = np.array([0.5, 0.25]), np.array([1.0, 2.0]), np.array([1.0, 1.0])
    print("ssm y for x=[1,0,0,2]:", ssm(a, b, c, [1.0, 0.0, 0.0, 2.0]))
    print("kernel K[0..3]:", [float((c * a**k * b).sum()) for k in range(4)])


def fig1():
    d, secs = 256, 10
    rows = []
    for s in (2, 4, 8, 16):
        lin, quad = encoder_flops_parts(secs * 100, d, 16, s)
        rows.append((s, lin / 1e9, quad / 1e9))
    top = max(l + q for _, l, q in rows)
    scale = 150 / top
    out = ['<svg viewBox="0 0 560 250" role="img" aria-label="Stacked bars of encoder compute in GFLOPs for 2x, 4x, 8x, and 16x subsampling of 10 seconds of audio">',
           '<line x1="70" y1="200" x2="530" y2="200" class="stroke-ink"/>']
    for i, (s, l, q) in enumerate(rows):
        x = 100 + i * 110
        hl, hq = l * scale, q * scale
        out.append(f'<rect x="{x}" y="{200 - hl:.1f}" width="60" height="{hl:.1f}" class="fill-a"/>')
        out.append(f'<rect x="{x}" y="{200 - hl - hq:.1f}" width="60" height="{hq:.1f}" class="fill-b"/>')
        out.append(f'<text x="{x + 30}" y="{200 - hl - hq - 8:.1f}" text-anchor="middle">{l + q:.1f}</text>')
        out.append(f'<text x="{x + 30}" y="220" text-anchor="middle">{s}x ({secs * 100 // s} frames)</text>')
        print(f"fig1 {s}x: linear {l:.2f} quadratic {q:.2f} GFLOPs", file=sys.stderr)
    out.append('<rect x="70" y="14" width="12" height="12" class="fill-a"/><text x="88" y="25" class="fig-small">projections and feed-forward (linear in T)</text>')
    out.append('<rect x="340" y="14" width="12" height="12" class="fill-b"/><text x="358" y="25" class="fig-small">attention scores (quadratic)</text>')
    out.append('<text x="300" y="242" text-anchor="middle" class="fig-small">subsampling factor, with encoder frames for 10 s; bar labels are GFLOPs, 16 layers, d = 256</text>')
    out.append('</svg>')
    return "\n".join(out)


def fig2():
    rates = [50, 25, 12.5, 6.25, 12.5, 25]
    names = ["1", "2", "3", "4", "5", "6"]
    out = ['<svg viewBox="0 0 560 250" role="img" aria-label="Frame rate inside the six Zipformer stacks: 50, 25, 12.5, 6.25, 12.5, and 25 hertz, a U shape">',
           '<line x1="50" y1="190" x2="540" y2="190" class="stroke-ink"/>']
    scale = 140 / 50
    for i, (r, n) in enumerate(zip(rates, names)):
        x = 70 + i * 78
        h = r * scale
        out.append(f'<rect x="{x}" y="{190 - h:.1f}" width="56" height="{h:.1f}" class="fill-soft-a stroke-a"/>')
        out.append(f'<text x="{x + 28}" y="{190 - h - 8:.1f}" text-anchor="middle">{r:g} Hz</text>')
        out.append(f'<text x="{x + 28}" y="210" text-anchor="middle">stack {n}</text>')
    out.append('<text x="300" y="18" text-anchor="middle" class="fig-small">each stack downsamples on entry and upsamples back to 50 Hz on exit</text>')
    out.append('<text x="300" y="240" text-anchor="middle" class="fig-small">frames per second inside each stack, for a 50 Hz embedding</text>')
    out.append('</svg>')
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--svg":
        print({"fig1": fig1, "fig2": fig2}[sys.argv[2]]())
    else:
        report()
