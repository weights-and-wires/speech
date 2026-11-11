#!/usr/bin/env python3
"""Compute the numbers and draw the figures for 049-streaming-latency-and-on-device.

Run it from the repo root:

    python3 scripts/figures/049-streaming-latency-and-on-device.py [--svg fig1|fig2|fig3]

Without `--svg` it prints the values the lesson quotes. With `--svg` it prints
that figure's SVG, drawn from the same numbers.
"""

from __future__ import annotations

import math
import sys

import numpy as np

EVENTS = [
    (560, "partial", "turn"),
    (800, "partial", "turn on"),
    (1000, "partial", "turn on a"),
    (1480, "partial", "turn on the light"),
    (1900, "final", "turn on the lights"),
]
WORDS = ["turn", "on", "the", "lights"]
WORD_ENDS = [480, 640, 760, 1240]
ENDPOINT = 1640
HOP = 40
LOOKAHEAD = 2
RTF = 0.3
CHUNKS = [1, 2, 4, 8, 16]
PARAMS = 30_000_000
BUDGET = 40_000_000


def algorithmic(c, l, hop):
    return (c - np.arange(c) + l) * hop


def activation_bytes(c, left, d, heads, b):
    ctx = left + c
    return (2 * c * d + 2 * ctx * d + heads * c * ctx) * b


def unstable_rate(partials, final):
    target, shown, bad = final.split(), 0, 0
    for text in partials:
        w = text.split()
        k = 0
        while k < len(w) and k < len(target) and w[k] == target[k]:
            k += 1
        shown += len(w)
        bad += len(w) - k
    return bad / shown, shown, bad


def values():
    first = next(t for t, k, x in EVENTS if k == "partial")
    final = next(t for t, k, x in EVENTS if k == "final")
    rows = []
    for c in CHUNKS:
        algo = float(np.mean(algorithmic(c, LOOKAHEAD, HOP)))
        comp = RTF * c * HOP
        rows.append((c, algo, comp))
    act = activation_bytes(16, 64, 256, 4, 2)
    return {
        "first_partial_ms": first - WORD_ENDS[0],
        "finalization_ms": final - WORD_ENDS[-1],
        "endpoint_wait_ms": ENDPOINT - WORD_ENDS[-1],
        "endpoint_to_final_ms": final - ENDPOINT,
        "unstable": unstable_rate([x for _, k, x in EVENTS if k == "partial"], EVENTS[-1][2]),
        "rows": rows,
        "act": act,
        "weights": {b: math.ceil(PARAMS * b / 8) for b in (16, 8, 4)},
    }


def fig1(v):
    x0, x1, scale = 50, 530, 480 / 2000
    px = lambda t: x0 + t * scale
    out = ['<svg viewBox="0 0 580 270" role="img" aria-label="Timeline of an utterance: word ends, partial results, the endpoint, and the final result, with the latencies marked">']
    out.append(f'<line x1="{x0}" y1="150" x2="{x1}" y2="150" class="stroke-muted"/>')
    for t in range(0, 2001, 500):
        out.append(f'<line x1="{px(t):.1f}" y1="146" x2="{px(t):.1f}" y2="154" class="stroke-muted"/>')
        out.append(f'<text x="{px(t):.1f}" y="168" text-anchor="middle" class="fig-small">{t}</text>')
    out.append('<text x="290" y="188" text-anchor="middle">time since the audio started (ms)</text>')
    out.append('<text x="50" y="16" class="fig-bold fig-small">word ends (what was said)</text>')
    for i, (w, e) in enumerate(zip(WORDS, WORD_ENDS)):
        ly = 34 + 18 * (i % 2)
        out.append(f'<circle cx="{px(e):.1f}" cy="150" r="5" class="fill-a"/>')
        out.append(f'<line x1="{px(e):.1f}" y1="150" x2="{px(e):.1f}" y2="{ly + 4}" class="stroke-a dashed"/>')
        out.append(f'<text x="{px(e):.1f}" y="{ly}" text-anchor="middle" class="fig-small">{w} {e}</text>')
    for t, k, text in EVENTS:
        cls = "fill-d" if k == "final" else "fill-b"
        out.append(f'<rect x="{px(t) - 5:.1f}" y="205" width="10" height="10" class="{cls}"/>')
    out.append(f'<text x="{px(560):.1f}" y="232" text-anchor="middle" class="fig-small">560</text>')
    out.append(f'<text x="{px(1000):.1f}" y="232" text-anchor="middle" class="fig-small">1000</text>')
    out.append(f'<text x="{px(1480):.1f}" y="232" text-anchor="middle" class="fig-small">1480</text>')
    out.append(f'<text x="{px(1900):.1f}" y="232" text-anchor="middle" class="fig-small">final 1900</text>')
    out.append(f'<line x1="{px(ENDPOINT):.1f}" y1="100" x2="{px(ENDPOINT):.1f}" y2="150" class="stroke-c"/>')
    out.append(f'<text x="{px(ENDPOINT):.1f}" y="94" text-anchor="middle" class="fig-small">endpoint {ENDPOINT}</text>')
    out.append(f'<text x="50" y="256" class="fig-small">orange squares are partials, purple is the final. First partial latency {v["first_partial_ms"]} ms, finalization {v["finalization_ms"]} ms.</text>')
    out.append('</svg>')
    return "\n".join(out)


def fig2(v):
    rows = v["rows"]
    top = max(a + c for _, a, c in rows)
    sx = 380 / top
    out = ['<svg viewBox="0 0 580 270" role="img" aria-label="Mean latency in milliseconds for chunk sizes 1 to 16 frames, split into algorithmic delay and compute">']
    for i, (c, a, comp) in enumerate(rows):
        y = 30 + i * 40
        out.append(f'<text x="100" y="{y + 17}" text-anchor="end" class="fig-small">chunk {c}</text>')
        out.append(f'<rect x="110" y="{y}" width="{a * sx:.1f}" height="24" class="fill-a"/>')
        out.append(f'<rect x="{110 + a * sx:.1f}" y="{y}" width="{comp * sx:.1f}" height="24" class="fill-b"/>')
        out.append(f'<text x="{116 + (a + comp) * sx:.1f}" y="{y + 17}" class="fig-small">{a + comp:.0f} ms</text>')
    out.append('<text x="110" y="250" class="fig-small">blue is algorithmic delay, orange is compute at RTF 0.3 (hop 40 ms, 2 look-ahead frames)</text>')
    out.append('</svg>')
    return "\n".join(out)


def fig3(v):
    w = v["weights"]
    scale = 380 / 60_000_000
    out = ['<svg viewBox="0 0 580 230" role="img" aria-label="Weight bytes of a 30 million parameter model at 16, 8, and 4 bits against a 40 MB budget">']
    for i, b in enumerate((16, 8, 4)):
        y = 30 + i * 46
        total = w[b] + v["act"]
        cls = "fill-b" if total <= BUDGET else "fill-bad"
        out.append(f'<text x="100" y="{y + 17}" text-anchor="end" class="fig-small">{b} bits</text>')
        out.append(f'<rect x="110" y="{y}" width="{w[b] * scale:.1f}" height="24" class="{cls}"/>')
        out.append(f'<rect x="{110 + w[b] * scale:.1f}" y="{y}" width="{max(v["act"] * scale, 1.5):.1f}" height="24" class="fill-a"/>')
        out.append(f'<text x="{118 + total * scale:.1f}" y="{y + 17}" class="fig-small">{total / 1e6:.1f} MB</text>')
    bx = 110 + BUDGET * scale
    out.append(f'<line x1="{bx:.1f}" y1="20" x2="{bx:.1f}" y2="170" class="stroke-ink dashed"/>')
    out.append(f'<text x="{bx:.1f}" y="14" text-anchor="middle" class="fig-small">budget 40 MB</text>')
    out.append('<text x="110" y="205" class="fig-small">orange fits, red does not. The thin blue slice is one chunk of activations, 0.1 MB.</text>')
    out.append('</svg>')
    return "\n".join(out)


if __name__ == "__main__":
    v = values()
    if len(sys.argv) == 3 and sys.argv[1] == "--svg":
        print({"fig1": fig1, "fig2": fig2, "fig3": fig3}[sys.argv[2]](v))
    else:
        for key in ("first_partial_ms", "finalization_ms", "endpoint_wait_ms", "endpoint_to_final_ms", "unstable", "act", "weights"):
            print(key, v[key])
        for c, a, comp in v["rows"]:
            print(f"chunk {c:2d}: algorithmic mean {a:6.1f} ms, compute {comp:5.1f} ms, total {a + comp:6.1f} ms")
