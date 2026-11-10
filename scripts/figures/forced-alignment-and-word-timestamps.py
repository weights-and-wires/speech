#!/usr/bin/env python3
"""Compute the forced-alignment worked example and draw its trellis.

The numbers in content/lessons/041-forced-alignment-and-word-timestamps.md
come from this script. Run it from the repo root:

    python3 scripts/figures/forced-alignment-and-word-timestamps.py

It prints the Viterbi table for the transcript `a b`, the same emissions
for the transcript `a a`, the word timestamps, and the SVG for Figure 1.
"""

from __future__ import annotations

import numpy as np

BLANK = 0
# Rows are frames, columns are the probabilities of (blank, a, b).
PROBS = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])
NAMES = ["_", "a", "b"]


def extend(tokens):
    ext = [BLANK]
    for token in tokens:
        ext += [token, BLANK]
    return ext


def trellis(probs, tokens, allow_repeat_skip=False):
    """Viterbi table (T, S) of best path probabilities, and the backpointers."""
    ext = extend(tokens)
    t_frames, s_states = len(probs), len(ext)
    delta = np.zeros((t_frames, s_states))
    back = np.full((t_frames, s_states), -1)
    delta[0, 0] = probs[0, ext[0]]
    delta[0, 1] = probs[0, ext[1]]
    for t in range(1, t_frames):
        for s in range(s_states):
            options = [(delta[t - 1, s], s)]
            if s >= 1:
                options.append((delta[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != BLANK and (allow_repeat_skip or ext[s] != ext[s - 2]):
                options.append((delta[t - 1, s - 2], s - 2))
            best, source = max(options, key=lambda o: o[0])
            delta[t, s] = best * probs[t, ext[s]]
            back[t, s] = source
    return delta, back


def best_path(delta, back):
    s_states = delta.shape[1]
    last = max([s_states - 1, s_states - 2], key=lambda s: delta[-1, s])
    path = [last]
    for t in range(len(delta) - 1, 0, -1):
        path.append(back[t, path[-1]])
    return path[::-1]


def figure(delta, back, path, ext):
    t_frames, s_states = delta.shape
    x0, y0, w, h, gx, gy = 150, 60, 80, 40, 100, 52
    out = ['<svg viewBox="0 0 600 340" role="img" aria-label="A Viterbi trellis for the transcript a b over four frames, with the best path blank, a, b, blank highlighted">',
           '<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>']
    for s in range(s_states):
        out.append(f'<text x="20" y="{y0 + s * gy + 25}" class="fig-small">state {s}: {"blank" if ext[s] == BLANK else NAMES[ext[s]]}</text>')
    for t in range(t_frames):
        out.append(f'<text x="{x0 + t * gx + w // 2}" y="{y0 - 14}" text-anchor="middle" class="fig-small fig-bold">frame {t}</text>')
    for t in range(t_frames):
        for s in range(s_states):
            on = path[t] == s
            reach = delta[t, s] > 0
            cls = "fill-soft-b stroke-b" if on else "fill-sunk stroke-muted"
            label = f"{delta[t, s]:.4f}".rstrip("0").rstrip(".") if reach else "0"
            out.append(f'<rect x="{x0 + t * gx}" y="{y0 + s * gy}" width="{w}" height="{h}" rx="6" class="{cls}"/>')
            out.append(f'<text x="{x0 + t * gx + w // 2}" y="{y0 + s * gy + 25}" text-anchor="middle" class="{"fig-bold" if on else "fig-small"}">{label}</text>')
    for t in range(1, t_frames):
        a, b = path[t - 1], path[t]
        out.append(f'<line x1="{x0 + (t - 1) * gx + w}" y1="{y0 + a * gy + h // 2}" x2="{x0 + t * gx - 2}" y2="{y0 + b * gy + h // 2}" class="stroke-b" marker-end="url(#c041-arrow)"/>')
    out.append(f'<text x="300" y="{y0 + s_states * gy + 18}" text-anchor="middle" class="fig-small">frame (1 frame = 20 ms); each cell is the best probability of any path that ends there</text>')
    out.append("</svg>")
    return "\n".join(out)


def timestamps(path, word_lengths, hop):
    spans = []
    for k in range(sum(word_lengths)):
        frames = [t for t, s in enumerate(path) if s == 2 * k + 1]
        spans.append((frames[0], frames[-1] + 1))
    out, k = [], 0
    for n in word_lengths:
        out.append((spans[k][0] * hop, spans[k + n - 1][1] * hop))
        k += n
    return spans, out


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    delta, back = trellis(PROBS, [1, 2])
    path = best_path(delta, back)
    print("transcript a b: delta\n", delta, "\nbackpointers\n", back, "\npath", path, "prob", delta[-1, path[-1]])
    d2, b2 = trellis(PROBS, [1, 1])
    print("transcript a a: delta\n", d2, "\nbackpointers\n", b2, "\npath", best_path(d2, b2))
    d3, b3 = trellis(PROBS, [1, 1], allow_repeat_skip=True)
    print("a a with the illegal skip: delta\n", d3, "\npath", best_path(d3, b3), "prob", d3[-1, best_path(d3, b3)[-1]])
    print(timestamps(path, [2], 0.02))
    print(figure(delta, back, path, extend([1, 2])))
