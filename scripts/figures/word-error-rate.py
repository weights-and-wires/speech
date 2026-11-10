#!/usr/bin/env python3
"""Compute the worked examples in content/lessons/026-word-error-rate.md and draw its new figures.

The numbers in the normalization and bootstrap sections come from the exercise
solution, not from this script's own arithmetic. Run it from the repo root:

    python3 scripts/figures/word-error-rate.py

It prints the numbers the lesson quotes, then the SVG for Figure 3 (WER under
each normalization) and Figure 4 (bootstrap intervals for the same 10-point gap
on 4 and on 400 utterances).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "content/exercises/026-word-error-rate/solution.py"

REFS = ["Set a timer for 10 minutes, please.", "Call Mom at 6 o'clock.", "Thanks, that's all."]
HYPS = ["set a timer for ten minutes please", "call mom at six o'clock", "thanks that's all"]
ERRORS_A, ERRORS_B, LENGTHS = [2, 1, 3, 1], [1, 2, 2, 0], [6, 2, 8, 4]


def load_solution():
    """Import the solution with a stand-in for speechlab, which only draws."""
    stub = types.ModuleType("speechlab")
    stub.show_image = lambda *args, **kwargs: None
    sys.modules.setdefault("speechlab", stub)
    spec = importlib.util.spec_from_file_location("wer_solution", SOLUTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(sol):
    """Corpus WER of the three pairs under four treatments."""
    strip = lambda text: " ".join(text.lower().replace(",", "").replace(".", "").split())
    lower = lambda text: " ".join(text.lower().split())
    rows = [("raw", str), ("lowercase", lower), ("+ punctuation", strip), ("+ numbers (normalize)", sol.normalize)]
    return [(name, sol.wer([fn(r) for r in REFS], [fn(h) for h in HYPS])) for name, fn in rows]


def bars_svg(rows) -> str:
    """Figure 3: one bar per treatment, width proportional to WER."""
    lines = ['<svg viewBox="0 0 620 230" role="img" aria-label="Bars of corpus word error rate for the same three transcript pairs: '
             + ", ".join(f"{name} {value:.1%}" for name, value in rows) + '">',
             '<line x1="210.0" y1="20.0" x2="210.0" y2="180.0" class="stroke-muted"/>']
    for k, (name, value) in enumerate(rows):
        y = 30 + 40 * k
        width = 340.0 * value / rows[0][1]
        cls = "fill-d" if k == 0 else "fill-b"
        lines.append(f'<text x="200.0" y="{y + 18}" class="fig-small" text-anchor="end">{name}</text>')
        if width > 0:
            lines.append(f'<rect x="210.0" y="{y}" width="{width:.1f}" height="26.0" class="{cls}" />')
        lines.append(f'<text x="{218 + width:.1f}" y="{y + 18}" class="fig-small" text-anchor="start">{value:.1%}</text>')
    lines.append('<text x="310.0" y="215.0" class="fig-small" text-anchor="middle">Corpus WER of the same 15 reference words under each treatment.</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def intervals_svg(results) -> str:
    """Figure 4: the 95% interval of WER(A) - WER(B), in percentage points, for each corpus size."""
    lo_axis, hi_axis = -20.0, 25.0
    x = lambda v: 150 + (v - lo_axis) / (hi_axis - lo_axis) * 350
    lines = ['<svg viewBox="0 0 620 220" role="img" aria-label="95 percent intervals of the WER difference, in percentage points, '
             'for 4 utterances and for 400 utterances">',
             f'<line x1="{x(0):.1f}" y1="30.0" x2="{x(0):.1f}" y2="150.0" class="stroke-ink dashed"/>',
             f'<text x="{x(0):.1f}" y="22.0" class="fig-small" text-anchor="middle">no difference</text>']
    for k, (name, point, low, high) in enumerate(results):
        y = 70 + 50 * k
        lines.append(f'<text x="140.0" y="{y + 5}" class="fig-small" text-anchor="end">{name}</text>')
        lines.append(f'<rect x="{x(low):.1f}" y="{y - 5}" width="{x(high) - x(low):.1f}" height="10.0" class="fill-b" />')
        lines.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="6" class="fill-ink"/>')
        lines.append(f'<text x="{x(high) + 8:.1f}" y="{y + 5}" class="fig-small" text-anchor="start">[{low:+.1f}, {high:+.1f}]</text>')
    for v in (-20, -10, 0, 10, 20):
        lines.append(f'<text x="{x(v):.1f}" y="172.0" class="fig-small" text-anchor="middle">{v:+d}</text>'.replace("+0", "0"))
    lines.append('<text x="365.0" y="200.0" class="fig-small" text-anchor="middle">WER(A) minus WER(B), in percentage points; the dot is the observed gap.</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    sol = load_solution()
    rows = scores(sol)
    print("normalization:", [(n, round(v, 4)) for n, v in rows], "words:", sum(len(r.split()) for r in REFS))
    gap = (sum(ERRORS_A) - sum(ERRORS_B)) / sum(LENGTHS)
    print(f"WER A {sum(ERRORS_A) / sum(LENGTHS):.4f}, WER B {sum(ERRORS_B) / sum(LENGTHS):.4f}, gap {gap:.4f}")
    # Two resamples drawn by hand in the lesson, 1-indexed.
    for draw in [(4, 2, 4, 1), (2, 2, 1, 2)]:
        idx = np.array(draw) - 1
        n = sum(LENGTHS[i] for i in idx)
        a, b = sum(ERRORS_A[i] for i in idx), sum(ERRORS_B[i] for i in idx)
        print(f"draw {draw}: words {n}, A {a} ({a / n:.4f}), B {b} ({b / n:.4f}), A - B {(a - b) / n:+.4f}")
    results = []
    for copies, name in ((1, "4 utterances"), (100, "400 utterances")):
        p, low, high = sol.paired_bootstrap(ERRORS_A * copies, ERRORS_B * copies, LENGTHS * copies, 10000, 0)
        print(f"{name}: p_b_better {p:.4f}, interval [{low:+.4f}, {high:+.4f}]")
        results.append((name, 100 * gap, 100 * low, 100 * high))
    print(bars_svg(rows))
    print(intervals_svg(results))


if __name__ == "__main__":
    main()
