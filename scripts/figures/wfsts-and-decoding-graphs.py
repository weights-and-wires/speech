#!/usr/bin/env python3
"""Compute the lesson's composed transducer and draw it.

The path costs in content/lessons/025-wfsts-and-decoding-graphs.md come from
this script. Run it from the repo root:

    python3 scripts/figures/wfsts-and-decoding-graphs.py

It prints the path costs, the log-semiring total, the determinization
arithmetic, and the SVG for Figure 1.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = (
    ROOT
    / "content"
    / "exercises"
    / "025-wfsts-and-decoding-graphs"
    / "solution.py"
)


def load_solution():
    spec = importlib.util.spec_from_file_location("wfst_solution", SOLUTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shorten(x1, y1, x2, y2, r1, r2):
    """Move the endpoints out to the circle edges so the arrow meets the rim."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    return (
        x1 + ux * r1,
        y1 + uy * r1,
        x2 - ux * r2,
        y2 - uy * r2,
    )


def fmt(value):
    return f"{value:.1f}"


def figure(path_a, path_cat, arc_ah, arc_k, arc_cat, final_a, final_cat):
    """SVG of the composed graph. Weights are the computed ones."""
    radius = 24
    start, mid, word_a, word_cat = (88, 168), (300, 250), (520, 78), (520, 250)
    ah = shorten(*start, *word_a, radius, radius)
    ka = shorten(*start, *mid, radius, radius)
    tc = shorten(*mid, *word_cat, radius, radius)
    return f"""<svg viewBox="0 0 640 320" role="img" aria-label="Composed lexicon and bigram. The path ah to a costs {fmt(path_a)}. The path k t to cat costs {fmt(path_cat)}.">
  <defs>
    <marker id="c025-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" class="fill-ink"/>
    </marker>
  </defs>
  <text x="320" y="22" text-anchor="middle" class="fig-bold">L composed with G, phones in and words out</text>
  <line x1="{ah[0]:.1f}" y1="{ah[1]:.1f}" x2="{ah[2]:.1f}" y2="{ah[3]:.1f}" class="stroke-b" marker-end="url(#c025-arrow)"/>
  <text x="270" y="86" text-anchor="middle" class="fig-mono">ah:a  {fmt(arc_ah)}</text>
  <text x="270" y="104" text-anchor="middle" class="fig-small">path cost {fmt(path_a)}</text>
  <line x1="{ka[0]:.1f}" y1="{ka[1]:.1f}" x2="{ka[2]:.1f}" y2="{ka[3]:.1f}" class="stroke-a" marker-end="url(#c025-arrow)"/>
  <text x="176" y="236" text-anchor="middle" class="fig-mono fig-small">k:eps  {fmt(arc_k)}</text>
  <line x1="{tc[0]:.1f}" y1="{tc[1]:.1f}" x2="{tc[2]:.1f}" y2="{tc[3]:.1f}" class="stroke-b" marker-end="url(#c025-arrow)"/>
  <text x="408" y="236" text-anchor="middle" class="fig-mono">t:cat  {fmt(arc_cat)}</text>
  <text x="408" y="300" text-anchor="middle" class="fig-small">path cost {fmt(path_cat)}</text>
  <circle cx="{start[0]}" cy="{start[1]}" r="{radius}" class="fill-soft-a stroke-a"/>
  <text x="{start[0]}" y="{start[1] + 5}" text-anchor="middle" class="fig-bold">0</text>
  <text x="{start[0]}" y="{start[1] + 42}" text-anchor="middle" class="fig-small">start</text>
  <circle cx="{mid[0]}" cy="{mid[1]}" r="{radius}" class="fill-soft-b stroke-b"/>
  <text x="{mid[0]}" y="{mid[1] + 5}" text-anchor="middle" class="fig-bold">1</text>
  <text x="{mid[0]}" y="{mid[1] + 42}" text-anchor="middle" class="fig-small">after k</text>
  <circle cx="{word_a[0]}" cy="{word_a[1]}" r="{radius}" class="fill-soft-c stroke-c"/>
  <circle cx="{word_a[0]}" cy="{word_a[1]}" r="18" class="stroke-c" fill="none"/>
  <text x="{word_a[0]}" y="{word_a[1] + 5}" text-anchor="middle" class="fig-bold">a</text>
  <text x="{word_a[0]}" y="{word_a[1] - 34}" text-anchor="middle" class="fig-small">final +{fmt(final_a)}</text>
  <circle cx="{word_cat[0]}" cy="{word_cat[1]}" r="{radius}" class="fill-soft-c stroke-c"/>
  <circle cx="{word_cat[0]}" cy="{word_cat[1]}" r="18" class="stroke-c" fill="none"/>
  <text x="{word_cat[0]}" y="{word_cat[1] + 5}" text-anchor="middle" class="fig-bold">cat</text>
  <text x="{word_cat[0]}" y="{word_cat[1] - 34}" text-anchor="middle" class="fig-small">final +{fmt(final_cat)}</text>
</svg>"""


def determinize():
    """Hand-sized determinization. Residuals use the tropical min."""
    light, heavy = 0.4, 1.0
    follow_x, follow_y = 1.0, 0.2
    kept = min(light, heavy)
    residual_heavy = heavy - kept
    y_after = follow_y + residual_heavy
    before = {"ax": light + follow_x, "ay": heavy + follow_y}
    after = {"ax": kept + follow_x, "ay": kept + y_after}
    return kept, residual_heavy, y_after, before, after


def main():
    solution = load_solution()
    graph = solution.compose(solution._lexicon(), solution._grammar())
    paths = solution.n_best(graph, 5)
    by_word = {outputs: (cost, inputs) for cost, inputs, outputs in paths}
    cost_a, phones_a = by_word[("a",)]
    cost_cat, phones_cat = by_word[("cat",)]
    total = solution.total_weight(graph)
    # Arc costs on the composed graph, read back from the arcs compose built.
    arc_cost = {}
    for _src, _dst, ilabel, olabel, weight in graph["arcs"]:
        arc_cost[(ilabel, olabel)] = weight
    arc_ah = arc_cost[("ah", "a")]
    arc_k = arc_cost[("k", solution.EPS)]
    arc_cat = arc_cost[("t", "cat")]
    final_a, final_cat = 0.4, 0.2
    assert phones_a == ("ah",)
    assert phones_cat == ("k", "t")
    assert abs(cost_a - (arc_ah + final_a)) < 1e-9
    assert abs(cost_cat - (arc_k + arc_cat + final_cat)) < 1e-9
    assert abs(total - float(-np.logaddexp(-cost_a, -cost_cat))) < 1e-12
    kept, residual, y_after, before, after = determinize()
    assert abs(before["ax"] - after["ax"]) < 1e-12
    assert abs(before["ay"] - after["ay"]) < 1e-12
    print(f"path a: phones {phones_a} cost {cost_a:.4f}")
    print(f"path cat: phones {phones_cat} cost {cost_cat:.4f}")
    print(f"arcs ah:a {arc_ah:.4f}  k:eps {arc_k:.4f}  t:cat {arc_cat:.4f}")
    print(f"log total {total:.4f}")
    print(
        f"determinize kept {kept:.1f}, residual {residual:.1f}, "
        f"y after {y_after:.1f}, paths {before} -> {after}"
    )
    print(figure(cost_a, cost_cat, arc_ah, arc_k, arc_cat, final_a, final_cat))


if __name__ == "__main__":
    main()
