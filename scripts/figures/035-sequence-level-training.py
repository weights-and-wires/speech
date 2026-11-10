#!/usr/bin/env python3
"""Compute the worked examples in content/lessons/035-sequence-level-training.md and draw its figures.

Every number comes from the exercise solution. Run it from the repo root:

    python3 scripts/figures/035-sequence-level-training.py [OUT_DIR]

It prints the values the lesson quotes. With OUT_DIR, it also writes the three
figure blocks to fig1.md, fig2.md, and fig3.md there.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOLUTION = ROOT / "content/exercises/035-sequence-level-training/solution.py"


def load_solution():
    """Import the solution with a stand-in for speechlab, which only draws."""
    stub = types.ModuleType("speechlab")
    stub.show_plot = lambda *args, **kwargs: None
    sys.modules.setdefault("speechlab", stub)
    spec = importlib.util.spec_from_file_location("sequence_solution", SOLUTION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bar_rows(rows, x0, bar_max, scale_max, cls_for, label_x, row_h=34, top=40):
    """Horizontal bars: rows of (label, value, class); returns SVG lines."""
    out = []
    for k, (label, value, cls) in enumerate(rows):
        y = top + k * row_h
        width = bar_max * value / scale_max
        out.append(f'<text x="{label_x}" y="{y + 17}" text-anchor="end" class="fig-mono">{label}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="24" class="{cls}"/>')
        out.append(f'<text x="{x0 + width + 8:.1f}" y="{y + 17}" class="fig-small">{value:.3f}</text>')
    return out


def figure_mmi(sol):
    loglik = np.log(np.array([[0.6, 0.4], [0.3, 0.7]]))
    start = np.log(np.array([0.5, 0.5]))
    trans = np.log(np.array([[0.2, 0.8], [0.6, 0.4]]))
    names = "ab"
    rows, total = [], 0.0
    for path in itertools.product(range(2), repeat=2):
        p = float(np.exp(sol.path_logscore(loglik, start, trans, list(path))))
        total += p
        rows.append(("".join(names[s] for s in path), p, "fill-b" if path == (0, 1) else "fill-soft-a"))
    den = float(np.exp(sol.mmi_denominator(loglik, start, trans)))
    loss = sol.mmi_loss(loglik, start, trans, [0, 1])
    grad = sol.mmi_grad(loglik, start, trans, [0, 1])
    print(f"path scores {[(n, round(v, 4)) for n, v, _ in rows]}; denominator {den:.4f} (sum {total:.4f})")
    print(f"numerator share {rows[1][1] / den:.4f}; loss {loss:.4f}")
    print(f"gradient with respect to loglik:\n{np.round(grad, 4)}")
    flat = sol.mmi_loss(loglik, np.log([0.5, 0.5]), np.log(np.full((2, 2), 0.5)), [0, 1])
    print(f"loss with a flat LM {flat:.4f}")
    svg = ['<svg viewBox="0 0 600 220" role="img" aria-label="Scores of the four state paths aa, ab, ba, and bb; the reference path ab holds 0.168 of the 0.278 total">']
    svg += bar_rows(rows, 90, 340, 0.2, None, 70)
    svg.append('<line x1="90" y1="34" x2="90" y2="176" class="stroke-muted"/>')
    svg.append(f'<text x="90" y="205" class="fig-small">score = P(x | path) times P(path); the four scores sum to the denominator, {den:.3f}</text>')
    svg.append('<text x="470" y="92" class="fig-small fig-bold">numerator</text>')
    svg.append(f'<text x="470" y="110" class="fig-small">{rows[1][1]:.3f} / {den:.3f} = {rows[1][1] / den:.3f}</text>')
    svg.append("</svg>")
    cap = (f"Figure 1. Training pushes up the share of the reference path ab among all four paths, {rows[1][1] / den:.3f} here. "
           "The loss is minus the log of that share, so it falls as ab gains share and the others lose it.")
    return "```figure\n" + "\n".join(svg) + "\n" + cap + "\n```"


def figure_mwer(sol):
    logps = np.log([0.30, 0.18, 0.12])
    errors = np.array([2.0, 0.0, 3.0])
    p = np.exp(logps - np.log(np.exp(logps).sum()))
    loss = sol.mwer_loss(logps, errors)
    grad = sol.mwer_grad(logps, errors)
    print(f"renormalized {np.round(p, 3)}; expected errors {loss:.3f}; gradient {np.round(grad, 3)}")
    cx, half = 460, 90
    svg = ['<svg viewBox="0 0 600 230" role="img" aria-label="Three hypotheses with renormalized probabilities 0.5, 0.3, and 0.2, word errors 2, 0, and 3, and gradients 0.2, minus 0.48, and 0.28">']
    svg.append('<text x="14" y="26" class="fig-small fig-bold">hypothesis</text><text x="120" y="26" class="fig-small fig-bold">P over the list</text>')
    svg.append('<text x="235" y="26" class="fig-small fig-bold">word errors</text>')
    svg.append(f'<text x="{cx}" y="26" text-anchor="middle" class="fig-small fig-bold">gradient on log p</text>')
    svg.append(f'<line x1="{cx}" y1="36" x2="{cx}" y2="176" class="stroke-muted"/>')
    for k in range(3):
        y = 44 + 46 * k
        svg.append(f'<text x="14" y="{y + 17}" class="fig-mono">h{k + 1}</text>')
        svg.append(f'<rect x="120" y="{y}" width="{p[k] * 120:.1f}" height="24" class="fill-a"/>')
        svg.append(f'<text x="{128 + p[k] * 120:.1f}" y="{y + 17}" class="fig-small">{p[k]:.1f}</text>')
        svg.append(f'<text x="262" y="{y + 17}" class="fig-small">{int(errors[k])}</text>')
        w = grad[k] / 0.5 * half
        x = cx if w >= 0 else cx + w
        cls = "fill-bad" if grad[k] > 0 else "fill-c"
        svg.append(f'<rect x="{x:.1f}" y="{y}" width="{abs(w):.1f}" height="24" class="{cls}"/>')
        tx = cx + w + (8 if w >= 0 else -8)
        svg.append(f'<text x="{tx:.1f}" y="{y + 17}" text-anchor="{"start" if w >= 0 else "end"}" class="fig-small">{grad[k]:+.2f}</text>')
    svg.append(f'<text x="14" y="206" class="fig-small">expected word errors = {loss:.1f}. Positive gradient: lowering that log probability lowers the loss.</text>')
    svg.append("</svg>")
    cap = ("Figure 2. The gradient is p times the gap between a hypothesis's errors and the list's mean, 1.6. "
           "Hypotheses worse than the mean get pushed down, h2 gets pushed up, and the three values sum to zero.")
    return "```figure\n" + "\n".join(svg) + "\n" + cap + "\n```"


def figure_confusion(sol):
    nbest = [["call", "mom", "back"], ["call", "tom", "home"], ["all", "mom", "home"]]
    probs = [0.4, 0.3, 0.3]
    slots = sol.word_posteriors(nbest, np.log(probs))
    words, conf = sol.consensus_decode(slots)
    print("slots", [{w: round(v, 2) for w, v in s.items()} for s in slots], "consensus", words, np.round(conf, 2))
    errs = lambda hyp: sum(p * sum(a != b for a, b in zip(hyp, h)) for p, h in zip(probs, nbest))
    print(f"expected word errors: 1-best {errs(nbest[0]):.1f}, consensus {errs(words):.1f}")
    svg = ['<svg viewBox="0 0 600 250" role="img" aria-label="A confusion network with three slots. Slot 1: call 0.7, all 0.3. Slot 2: mom 0.7, tom 0.3. Slot 3: home 0.6, back 0.4. The consensus is call mom home.">']
    for k, slot in enumerate(slots):
        x = 40 + k * 190
        svg.append(f'<text x="{x}" y="24" class="fig-small fig-bold">slot {k + 1}</text>')
        top = max(slot, key=slot.get)
        for j, (word, value) in enumerate(sorted(slot.items(), key=lambda kv: -kv[1])):
            y = 40 + j * 40
            cls = "fill-b" if word == top else "fill-soft-a"
            svg.append(f'<rect x="{x}" y="{y}" width="{value * 150:.1f}" height="28" class="{cls}"/>')
            svg.append(f'<text x="{x + 6}" y="{y + 19}" class="fig-mono">{word}</text>')
            svg.append(f'<text x="{x + value * 150 + 6:.1f}" y="{y + 19}" class="fig-small">{value:.1f}</text>')
    svg.append('<text x="40" y="150" class="fig-small">The 1-best hypothesis is call mom back, with 1.2 expected word errors.</text>')
    svg.append('<text x="40" y="172" class="fig-small fig-bold">The consensus, call mom home, appears in no hypothesis and has 1.0.</text>')
    svg.append('<text x="40" y="194" class="fig-small">Bar length is the slot posterior. The orange bar is the winner of each slot.</text>')
    svg.append("</svg>")
    cap = "Figure 3. Each slot's bars sum to 1. Choosing the tallest bar in every slot gives a transcript that none of the three hypotheses wrote, and its expected errors are lower than the 1-best's."
    return "```figure\n" + "\n".join(svg) + "\n" + cap + "\n```"


def main() -> None:
    sol = load_solution()
    figs = [figure_mmi(sol), figure_mwer(sol), figure_confusion(sol)]
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        out.mkdir(parents=True, exist_ok=True)
        for k, fig in enumerate(figs, 1):
            (out / f"fig{k}.md").write_text(fig + "\n")


if __name__ == "__main__":
    main()
