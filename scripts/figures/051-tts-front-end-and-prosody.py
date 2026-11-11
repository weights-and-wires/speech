#!/usr/bin/env python3
"""Compute the numbers behind the figures in 051-tts-front-end-and-prosody.

Run it from the repo root:

    python3 scripts/figures/051-tts-front-end-and-prosody.py

It prints the bin shift that a 1.2 pitch scale causes under log-spaced and
linear-spaced bins (Figure 1), and the style-token attention worked example
(Figure 2). With --svg it also prints the two figures' SVG.
"""

from __future__ import annotations

import sys

import numpy as np

V_MIN, V_MAX, K = 71.0, 795.0, 256


def bin_edges(log_scale: bool) -> np.ndarray:
    if log_scale:
        return np.exp(np.linspace(np.log(V_MIN), np.log(V_MAX), K - 1))
    return np.linspace(V_MIN, V_MAX, K - 1)


def bins(values: np.ndarray, log_scale: bool) -> np.ndarray:
    return np.searchsorted(bin_edges(log_scale), values, side="left")


def shifts(pitches: np.ndarray, scale: float, log_scale: bool) -> np.ndarray:
    return bins(pitches * scale, log_scale) - bins(pitches, log_scale)


def style_attention(ref: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = np.tanh(tokens) @ ref / np.sqrt(tokens.shape[1])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return w @ tokens, w


PITCHES = np.arange(100.0, 501.0, 50.0)
TOKENS = 3.0 * np.eye(4)
REF = np.array([0.0, 0.0, 4.0, 0.0])


def fig1_svg() -> str:
    x0, x1, y0, y1 = 70.0, 540.0, 200.0, 40.0
    ymax = 40.0

    def px(p: float) -> float:
        return x0 + (p - 100.0) / 400.0 * (x1 - x0)

    def py(s: float) -> float:
        return y0 - s / ymax * (y0 - y1)

    out = ['<svg viewBox="0 0 600 260" role="img" aria-label="Bins moved by a 1.2 pitch scale: about 19 for log-spaced bins at every pitch, but growing from 7 to 35 for linear bins">']
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="stroke-muted"/>')
    for s in (0, 10, 20, 30, 40):
        out.append(f'<line x1="{x0}" y1="{py(s):.1f}" x2="{x1}" y2="{py(s):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{py(s) + 4:.1f}" text-anchor="end" class="fig-small">{s}</text>')
    for p in (100, 200, 300, 400, 500):
        out.append(f'<text x="{px(p):.1f}" y="{y0 + 18}" text-anchor="middle" class="fig-small">{p}</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{y0 + 40}" text-anchor="middle" class="fig-small">pitch before scaling (Hz)</text>')
    out.append(f'<text x="14" y="{(y0 + y1) / 2}" text-anchor="middle" class="fig-small" transform="rotate(-90 14 {(y0 + y1) / 2})">bins moved</text>')
    for log_scale, cls in ((True, "b"), (False, "a")):
        s = shifts(PITCHES, 1.2, log_scale)
        pts = " ".join(f"{px(p):.1f},{py(v):.1f}" for p, v in zip(PITCHES, s))
        out.append(f'<polyline points="{pts}" fill="none" class="stroke-{cls}" stroke-width="2.5"/>')
        for p, v in zip(PITCHES, s):
            out.append(f'<circle cx="{px(p):.1f}" cy="{py(v):.1f}" r="3.5" class="fill-{cls}"/>')
    out.append(f'<text x="{px(300)}" y="{py(20) - 10:.1f}" text-anchor="middle" class="fig-small fig-bold">log bins: 19 or 20 everywhere</text>')
    out.append(f'<text x="{px(500) - 6}" y="{py(35) - 8:.1f}" text-anchor="end" class="fig-small fig-bold">linear bins</text>')
    out.append("</svg>")
    return "\n".join(out)


def fig2_svg() -> str:
    _, w = style_attention(REF, TOKENS)
    out = ['<svg viewBox="0 0 600 220" role="img" aria-label="Attention weights over four style tokens: 0.709 on token 2 and 0.097 on each of the others">']
    base, top = 170.0, 40.0
    out.append(f'<line x1="90" y1="{base}" x2="510" y2="{base}" class="stroke-muted"/>')
    for i, v in enumerate(w):
        x = 120 + i * 100
        h = v * (base - top) / 0.8
        cls = "fill-b" if i == int(np.argmax(w)) else "fill-soft-a stroke-a"
        out.append(f'<rect x="{x}" y="{base - h:.1f}" width="60" height="{h:.1f}" rx="3" class="{cls}"/>')
        out.append(f'<text x="{x + 30}" y="{base - h - 6:.1f}" text-anchor="middle" class="fig-small fig-bold">{v:.3f}</text>')
        out.append(f'<text x="{x + 30}" y="{base + 18}" text-anchor="middle" class="fig-small">token {i}</text>')
    out.append(f'<text x="300" y="206" text-anchor="middle" class="fig-small">attention weight (sums to 1)</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    print("pitch (Hz):         ", PITCHES.astype(int).tolist())
    print("bins, log:          ", bins(PITCHES, True).tolist())
    print("bins, linear:       ", bins(PITCHES, False).tolist())
    print("shift at 1.2, log:  ", shifts(PITCHES, 1.2, True).tolist())
    print("shift at 1.2, lin:  ", shifts(PITCHES, 1.2, False).tolist())
    step = np.log(V_MAX / V_MIN) / (K - 2)
    print("log bin width:", round(float(step), 5), "ln units; bins per semitone:", round(float(np.log(2) / 12 / step), 2))
    style, w = style_attention(REF, TOKENS)
    print("logit of token 2:", round(float(np.tanh(3.0) * 4 / 2), 3))
    print("weights:", np.round(w, 4).tolist(), "sum", round(float(w.sum()), 6))
    print("style vector:", np.round(style, 3).tolist())
    if "--svg" in sys.argv:
        print(fig1_svg())
        print(fig2_svg())


if __name__ == "__main__":
    main()
