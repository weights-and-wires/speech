#!/usr/bin/env python3
"""Compute the worked examples and draw the figures for 043-keyword-spotting-and-language-id.

Run it from the repo root:

    python3 scripts/figures/043-keyword-spotting-and-language-id.py

It prints every number the lesson quotes, then the SVG for Figure 1 and Figure 2.
It uses only numpy and the standard library.
"""

from __future__ import annotations

import math

import numpy as np


def smooth(p, w):
    p = np.asarray(p, dtype=float)
    c = np.concatenate([[0.0], np.cumsum(p)])
    t = np.arange(len(p))
    lo = np.maximum(0, t - w + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def detect(scores, threshold, holdoff):
    out, blocked = [], -1
    for t in np.flatnonzero(np.asarray(scores) >= threshold):
        if t > blocked:
            out.append(int(t))
            blocked = t + holdoff
    return np.array(out, dtype=int)


def endpoint(vad, eoq, short, long_, thr=0.5):
    started, quiet = False, 0
    for t in range(len(vad)):
        if vad[t]:
            started, quiet = True, 0
        elif started:
            quiet += 1
            if quiet > (short if eoq[t] >= thr else long_):
                return t
    return -1


def excerpt():
    """A 3.2 s stream: a keyword at frames 30-89, a spike at 140, and a false bump at 250-289."""
    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 0.1, 320)
    s[30:90] += 0.9
    s[140] += 0.9
    s[250:290] += 0.8
    return s


def worked_numbers():
    raw = [0.1, 0.2, 0.9, 0.1, 0.8, 0.9, 0.7, 0.2, 0.1, 0.1, 0.1]
    sm = smooth(raw, 3)
    print("smoothing worked example, window 3:", np.round(sm, 3).tolist())
    print("  frames at or above 0.5:", np.flatnonzero(sm >= 0.5).tolist(), "raw:", np.flatnonzero(np.array(raw) >= 0.5).tolist())
    s = excerpt()
    sm5 = smooth(s, 5)
    for label, x in (("raw", s), ("smoothed w=5", sm5)):
        for h in (0, 100):
            d = detect(x, 0.5, h)
            print(f"{label:13s} hold-off {h:3d}: {len(d)} detections, first {d[:4].tolist()}")
    print("peak smoothed value of the spike:", round(float(sm5[140:145].max()), 3))
    utter = {
        "A": ([1] * 6 + [0] * 3 + [1] * 3 + [0] * 8, [0.0] * 6 + [0.1, 0.1, 0.2] + [0.0] * 3 + [0.3, 0.7] + [0.9] * 6, 11),
        "B": ([1] * 8 + [0] * 8, [0.0] * 8 + [0.2, 0.2, 0.3, 0.8] + [0.9] * 4, 7),
        "C": ([1] * 4 + [0] * 6 + [1] * 3 + [0] * 8, [0.0] * 4 + [0.1] * 6 + [0.0] * 3 + [0.4, 0.8] + [0.9] * 6, 12),
    }
    for name, (short, long_) in (("fixed 5", (5, 5)), ("fixed 2", (2, 2)), ("short 1, long 8", (1, 8))):
        ends = [endpoint(np.array(v), np.array(e), short, long_) for v, e, _ in utter.values()]
        last = [u[2] for u in utter.values()]
        cut = [d < l for d, l in zip(ends, last)]
        lat = [(d - l) * 100 for d, l, c in zip(ends, last, cut) if not c]
        print(f"{name:16s} decisions {ends}, cut-offs {sum(cut)}/3, mean latency {np.mean(lat):.1f} ms")
    # Figure 2 uses utterance A.
    return s, utter["A"]


def figure_1(s):
    sm = smooth(s, 5)
    x0, x1, y0, y1 = 60, 570, 40, 210

    def X(t):
        return x0 + (x1 - x0) * t / 320

    def Y(v):
        return y1 - (y1 - y0) * v

    out = ['<svg viewBox="0 0 600 316" role="img" aria-label="Keyword score over 3.2 seconds: a keyword, a one-frame spike, and a false bump, with the smoothed score, a threshold of 0.5, and where detections fire with and without a hold-off">']
    for v in (0.0, 0.5, 1.0):
        out.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" class="fig-small" text-anchor="end">{v:g}</text>')
    out.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="stroke-muted"/>')
    for sec in (0, 1, 2, 3):
        t = sec * 100
        out.append(f'<line x1="{X(t):.1f}" y1="{y1}" x2="{X(t):.1f}" y2="{y1 + 4}" class="stroke-muted"/>')
        out.append(f'<text x="{X(t):.1f}" y="{y1 + 18}" class="fig-small" text-anchor="middle">{sec:g}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{y1 + 36}" class="fig-small" text-anchor="middle">time (s)</text>')
    out.append(f'<text x="{x0}" y="26" class="fig-small">keyword score</text>')
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in enumerate(s))
    out.append(f'<polyline points="{pts}" class="stroke-muted" fill="none"/>')
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in enumerate(sm))
    out.append(f'<polyline points="{pts}" class="stroke-b" fill="none"/>')
    out.append(f'<line x1="{x0}" y1="{Y(0.5):.1f}" x2="{x1}" y2="{Y(0.5):.1f}" class="stroke-ink dashed"/>')
    out.append(f'<text x="{x1}" y="{Y(0.5) - 5:.1f}" class="fig-small" text-anchor="end">threshold 0.5</text>')
    out.append(f'<text x="{X(60):.1f}" y="{y0 - 6}" class="fig-small" text-anchor="middle">keyword</text>')
    out.append(f'<text x="{X(140):.1f}" y="{y0 - 6}" class="fig-small" text-anchor="middle">spike</text>')
    out.append(f'<text x="{X(270):.1f}" y="{y0 - 6}" class="fig-small" text-anchor="middle">false bump</text>')
    rows = [("no hold-off", detect(sm, 0.5, 0), 270, "fill-bad"), ("hold-off 100", detect(sm, 0.5, 100), 292, "fill-b")]
    for label, d, y, cls in rows:
        for t in d:
            out.append(f'<circle cx="{X(t):.1f}" cy="{y}" r="2.5" class="{cls}"/>')
        out.append(f'<text x="300" y="{y + 4}" class="fig-small">{label}: {len(d)} detections</text>')
    out.append("</svg>")
    return "\n".join(out)


def figure_2(u):
    vad, eoq, last = u
    x0, w = 90, 26
    out = ['<svg viewBox="0 0 600 270" role="img" aria-label="One utterance in 18 frames of 100 ms: speech, a pause, more speech, then silence, with the end-of-query probability and the frame where each of three endpointers fires">']
    out.append('<text x="12" y="44" class="fig-small">frame</text>')
    for t in range(18):
        cx = x0 + t * (w + 2) + w / 2
        if t % 2 == 0:
            out.append(f'<text x="{cx:.0f}" y="44" class="fig-small" text-anchor="middle">{t}</text>')
        cls = "fill-soft-a stroke-a" if vad[t] else "fill-sunk stroke-muted"
        out.append(f'<rect x="{x0 + t * (w + 2)}" y="54" width="{w}" height="26" rx="3" class="{cls}"/>')
        h = 60 * eoq[t]
        out.append(f'<rect x="{x0 + t * (w + 2)}" y="{148 - h:.1f}" width="{w}" height="{h:.1f}" class="fill-soft-b stroke-b"/>')
    out.append('<text x="12" y="72" class="fig-small">speech</text>')
    out.append('<text x="12" y="120" class="fig-small">end of</text><text x="12" y="134" class="fig-small">query</text>')
    thr_y = 148 - 60 * 0.5
    out.append(f'<line x1="{x0}" y1="{thr_y}" x2="{x0 + 18 * (w + 2)}" y2="{thr_y}" class="stroke-ink dashed"/>')
    out.append(f'<text x="{x0 - 6}" y="{thr_y + 4}" class="fig-small" text-anchor="end">0.5</text>')
    marks = [("fixed 2 frames", 8, "fill-bad", 182), ("end-of-query, 1 and 5", 13, "fill-b", 208), ("fixed 5 frames", 17, "fill-c", 234)]
    for label, t, cls, y in marks:
        cx = x0 + t * (w + 2) + w / 2
        out.append(f'<circle cx="{cx:.0f}" cy="{y}" r="6" class="{cls}"/>')
        out.append(f'<text x="{cx - 12:.0f}" y="{y + 4}" class="fig-small" text-anchor="end">{label}: frame {t}</text>')
    out.append(f'<line x1="{x0 + last * (w + 2) + w:.0f}" y1="50" x2="{x0 + last * (w + 2) + w:.0f}" y2="244" class="stroke-muted dashed"/>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    s, u = worked_numbers()
    print("\n--- Figure 1 ---")
    print(figure_1(s))
    print("\n--- Figure 2 ---")
    print(figure_2(u))
