#!/usr/bin/env python3
"""Compute the numbers and the SVG for 046-long-form-and-trustworthy-asr.

Run it from the repo root:

    python3 scripts/figures/046-long-form-and-trustworthy-asr.py [--svg]

It prints the window layout of Figure 1 and the reliability bins, ECE, and
temperature-free accuracy of Figure 2. With --svg it also prints both SVGs.
"""

from __future__ import annotations

import sys

import numpy as np

# Figure 1: a 70 s recording cut into 30 s windows that start every 25 s.
DURATION, WINDOW, STRIDE, WORDS_PER_SECOND = 70.0, 30.0, 25.0, 2.5

# Figure 2: a recognizer that is overconfident. Confidence is Beta(6, 1.6), and a word
# is right with probability conf ** 2.
N_WORDS, N_BINS, SEED = 2000, 10, 0


def windows():
    starts = np.arange(0.0, DURATION - WINDOW + STRIDE, STRIDE)
    return [(float(s), float(min(s + WINDOW, DURATION))) for s in starts if s < DURATION]


def reliability():
    rng = np.random.default_rng(SEED)
    conf = rng.beta(6.0, 1.6, N_WORDS)
    correct = (rng.random(N_WORDS) < conf**2).astype(float)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    which = np.searchsorted(edges, conf, side="left")
    rows = []
    ece = 0.0
    for b in range(N_BINS):
        m = which == b
        if m.any():
            rows.append((b, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
            ece += m.sum() / N_WORDS * abs(correct[m].mean() - conf[m].mean())
        else:
            rows.append((b, 0, float("nan"), float("nan")))
    return rows, float(ece), float(correct.mean()), float(conf.mean())


def svg_windows(wins):
    sx = lambda t: 70 + t * (480 / DURATION)
    out = ['<svg viewBox="0 0 600 190" role="img" aria-label="Three 30-second windows over a 70-second recording, with the 5-second overlaps shaded">',
           '<line x1="70" y1="150" x2="550" y2="150" class="stroke-muted"/>']
    for t in range(0, 71, 10):
        out.append(f'<line x1="{sx(t):.0f}" y1="146" x2="{sx(t):.0f}" y2="154" class="stroke-muted"/>')
        out.append(f'<text x="{sx(t):.0f}" y="171" text-anchor="middle" class="fig-small">{t}</text>')
    for k, (a, b) in enumerate(wins):
        y = 24 + 38 * k
        out.append(f'<rect x="{sx(a):.0f}" y="{y}" width="{sx(b) - sx(a):.0f}" height="26" rx="4" class="fill-soft-a stroke-a"/>')
        out.append(f'<text x="{sx(a) + 8:.0f}" y="{y + 18}" class="fig-small">window {k + 1}: {a:.0f} to {b:.0f} s</text>')
    for (a0, b0), (a1, b1) in zip(wins, wins[1:]):
        out.append(f'<rect x="{sx(a1):.0f}" y="16" width="{sx(b0) - sx(a1):.0f}" height="122" class="fill-soft-b" opacity="0.55"/>')
        out.append(f'<text x="{(sx(a1) + sx(b0)) / 2:.0f}" y="12" text-anchor="middle" class="fig-small">{b0 - a1:.0f} s overlap</text>')
    out.append(f'<text x="310" y="188" text-anchor="middle" class="fig-small">time in the recording (seconds)</text>')
    out.append('</svg>')
    return "\n".join(out)


def svg_reliability(rows, ece):
    x0, y0, w, h = 70, 20, 260, 260
    out = ['<svg viewBox="0 0 600 320" role="img" aria-label="A reliability diagram: accuracy in each confidence bin sits below the diagonal">']
    for q in (0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{y0 + h - q * h:.0f}" x2="{x0 + w}" y2="{y0 + h - q * h:.0f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y0 + h - q * h + 4:.0f}" text-anchor="end" class="fig-small">{q:.1f}</text>')
        out.append(f'<text x="{x0 + q * w:.0f}" y="{y0 + h + 18}" text-anchor="middle" class="fig-small">{q:.1f}</text>')
    bw = w / N_BINS
    for b, n, c, a in rows:
        if n == 0:
            continue
        out.append(f'<rect x="{x0 + b * bw + 2:.1f}" y="{y0 + h - a * h:.1f}" width="{bw - 4:.1f}" height="{a * h:.1f}" class="fill-soft-b stroke-b"/>')
    out.append(f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0}" class="stroke-ink dashed"/>')
    out.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 40}" text-anchor="middle" class="fig-small">confidence (bin of width 0.1)</text>')
    out.append(f'<text x="14" y="{y0 + h / 2}" class="fig-small" transform="rotate(-90 14 {y0 + h / 2})" text-anchor="middle">accuracy</text>')
    out.append('<text x="370" y="50" class="fig-bold">What the bars show</text>')
    big = max(rows, key=lambda r: r[1])
    out.append(f'<text x="370" y="72" class="fig-small">ECE = {ece:.3f}</text>')
    out.append(f'<text x="370" y="94" class="fig-small">fullest bin: {big[1]} of {N_WORDS} words</text>')
    out.append(f'<text x="370" y="116" class="fig-small">at confidence {big[2]:.2f}, accuracy {big[3]:.2f}</text>')
    out.append('<text x="370" y="150" class="fig-small">dashed line: perfect calibration</text>')
    out.append('</svg>')
    return "\n".join(out)


if __name__ == "__main__":
    wins = windows()
    print("Windows (start, end in s):", wins)
    for (a0, b0), (a1, b1) in zip(wins, wins[1:]):
        print(f"Overlap {a1:.0f} to {b0:.0f} s = {b0 - a1:.0f} s = {(b0 - a1) * WORDS_PER_SECOND:.0f} words at {WORDS_PER_SECOND} words/s")
    rows, e, acc, mc = reliability()
    print(f"Mean confidence {mc:.3f}, accuracy {acc:.3f}, ECE {e:.4f}")
    for b, n, c, a in rows:
        print(f"bin {b}: n={n} mean conf {c:.3f} acc {a:.3f}")
    if "--svg" in sys.argv:
        print(svg_windows(wins))
        print(svg_reliability(rows, e))
