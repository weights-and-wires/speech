#!/usr/bin/env python3
"""Compute the ASR-recipe worked examples and draw the learning-rate curve.

The numbers in content/lessons/040-asr-recipe.md come from this script.
Run it from the repo root:

    python3 scripts/figures/asr-recipe.py

It prints the byte-pair counts, the speed-perturbation lengths, the
padding waste, the Noam schedule at the steps the lesson quotes, and
the SVG for Figure 1.
"""

from __future__ import annotations

from collections import Counter
from math import ceil

import numpy as np


D_MODEL = 256
WARMUP = 4000
STEPS = (1, 100, 1000, 4000, 8000, 16000)


def noam_lr(step: float, d_model: int = D_MODEL, warmup: int = WARMUP) -> float:
    """Vaswani et al. learning rate. `step` starts at 1."""
    step = float(step)
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def merge_pair(seq: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace non-overlapping occurrences of `pair`, scanning from the left."""
    left, right = pair
    merged: list[str] = []
    index = 0
    while index < len(seq):
        if (
            index + 1 < len(seq)
            and seq[index] == left
            and seq[index + 1] == right
        ):
            merged.append(left + right)
            index += 2
        else:
            merged.append(seq[index])
            index += 1
    return merged


def train_bpe(corpus: list[str], n_merges: int) -> list[tuple[str, str]]:
    """Most frequent adjacent symbol pair, ties broken lexicographically."""
    sequences = [list(text) for text in corpus]
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for seq in sequences:
            for left, right in zip(seq, seq[1:]):
                counts[(left, right)] += 1
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        print("count", dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))))
        merges.append(pair)
        sequences = [merge_pair(seq, pair) for seq in sequences]
    print("merges", merges)
    print("encoded", sequences)
    return merges


def figure_svg() -> str:
    """SVG of lr(step) for d_model 256 and warmup 4,000. Points are computed."""
    # Plot box: x maps step 0..16000 to 88..612, y maps lr 0..0.0012 to 232..28.
    x0, x1 = 88.0, 612.0
    y0, y1 = 232.0, 28.0
    step_max = 16000.0
    lr_max = 0.0012

    def x_of(step: float) -> float:
        return x0 + step / step_max * (x1 - x0)

    def y_of(lr: float) -> float:
        return y0 - lr / lr_max * (y0 - y1)

    sample_steps = np.linspace(1.0, step_max, 81)
    sample_lrs = [noam_lr(step) for step in sample_steps]
    points = " ".join(
        f"{x_of(step):.1f},{y_of(lr):.1f}"
        for step, lr in zip(sample_steps, sample_lrs)
    )
    peak_step = float(WARMUP)
    peak_lr = noam_lr(peak_step)
    half_step = 16000.0
    half_lr = noam_lr(half_step)
    warmup_x = x_of(peak_step)

    y_ticks = []
    for lr, label in ((0.0, "0"), (0.0005, "0.0005"), (0.001, "0.001")):
        y_pos = y_of(lr)
        y_ticks.append(
            f'  <line x1="{x0:.1f}" y1="{y_pos:.1f}" x2="{x1:.1f}" y2="{y_pos:.1f}" class="stroke-grid"/>'
        )
        y_ticks.append(
            f'  <text x="{x0 - 8:.1f}" y="{y_pos + 4:.1f}" class="fig-small" text-anchor="end">{label}</text>'
        )
    x_ticks = []
    for step in (0, 4000, 8000, 12000, 16000):
        x_pos = x_of(float(step))
        x_ticks.append(
            f'  <text x="{x_pos:.1f}" y="{y0 + 18:.1f}" class="fig-small" text-anchor="middle">{step}</text>'
        )

    return f"""<svg viewBox="0 0 640 280" role="img" aria-label="Learning rate versus step. The curve rises through a warmup to a peak at step 4000, then falls to half that peak at step 16000.">
  <rect x="{x0:.1f}" y="{y1:.1f}" width="{warmup_x - x0:.1f}" height="{y0 - y1:.1f}" class="fill-soft-a"/>
{chr(10).join(y_ticks)}
  <line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x0:.1f}" y2="{y0:.1f}" class="stroke-muted"/>
  <line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" class="stroke-muted"/>
  <line x1="{warmup_x:.1f}" y1="{y1:.1f}" x2="{warmup_x:.1f}" y2="{y0:.1f}" class="stroke-muted dashed"/>
  <polyline points="{points}" class="stroke-b" fill="none"/>
  <circle cx="{x_of(peak_step):.1f}" cy="{y_of(peak_lr):.1f}" r="4.5" class="fill-b"/>
  <circle cx="{x_of(half_step):.1f}" cy="{y_of(half_lr):.1f}" r="4.5" class="fill-a"/>
{chr(10).join(x_ticks)}
  <text x="{x_of(2000):.1f}" y="{y1 + 16:.1f}" class="fig-small">warmup</text>
  <text x="{x_of(9000):.1f}" y="{y_of(peak_lr) - 10:.1f}" class="fig-small">inverse square root</text>
  <text x="16" y="16" class="fig-small">learning rate</text>
  <text x="{(x0 + x1) / 2:.1f}" y="268" class="fig-small" text-anchor="middle">step</text>
</svg>"""


def main() -> None:
    corpus = ["cat", "cat", "car", "car", "card"]
    train_bpe(corpus, 3)
    for factor, numer, denom in ((0.9, 9, 10), (1.1, 11, 10)):
        # length = ceil(N * denom / numer) for factor = numer/denom
        print(factor, ceil(16000 * denom / numer))
    lengths = [40, 10, 42, 11, 100, 12]
    print("real frames", sum(lengths), "padded", 220, "waste", 5 / 220)
    for step in STEPS:
        print(f"step {step}: {noam_lr(step):.12e}")
    print("peak", noam_lr(WARMUP), "half", noam_lr(16000))
    print(figure_svg())


if __name__ == "__main__":
    main()
