#!/usr/bin/env python3
"""Compute the overlap and collar worked example and draw its figure.

The numbers in content/lessons/040-diarization.md come from this script.
Run it from the repo root:

    python3 scripts/figures/diarization.py

It prints the DER for each scoring convention and the SVG for Figure 3.
The eigengap and 12-frame figures use the exercise's Run and the single-label
worked example in the lesson.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

REF = np.zeros((2, 12), dtype=bool)
REF[0, :8], REF[1, 5:] = True, True
HYP = np.zeros((2, 12), dtype=bool)
HYP[0, :8], HYP[1, 8:] = True, True


def forgiven(ref: np.ndarray, collar: int) -> np.ndarray:
    scored = np.ones(ref.shape[1], dtype=bool)
    padded = np.pad(ref, ((0, 0), (1, 1)))
    for t in np.flatnonzero((padded[:, 1:] != padded[:, :-1]).any(axis=0)):
        scored[max(0, t - collar) : t + collar] = False
    return ~scored


def der(ref, hyp, collar=0, score_overlap=True):
    scored = ~forgiven(ref, collar) if collar else np.ones(ref.shape[1], dtype=bool)
    if not score_overlap:
        scored &= ref.sum(axis=0) < 2
    r, h = ref[:, scored], hyp[:, scored]
    n_ref, n_hyp = r.sum(axis=0), h.sum(axis=0)
    overlap = r.astype(int) @ h.T.astype(int)
    matched = overlap[linear_sum_assignment(-overlap)].sum()
    errors = np.maximum(n_ref, n_hyp).sum() - matched
    return errors, n_ref.sum()


def figure() -> str:
    x0, w, gap = 110, 36, 4
    rows = [("reference A", REF[0], 60), ("reference B", REF[1], 104), ("system X", HYP[0], 164), ("system Y", HYP[1], 208)]
    skip = forgiven(REF, 1)
    errors = REF.sum(axis=0) > HYP.sum(axis=0)  # frames 5 to 7 lose speaker B
    out = ['<svg viewBox="0 0 600 300" role="img" aria-label="Two reference speakers that overlap in frames 5 to 7 and a system that hears the second speaker from frame 8: frames 5, 6, and 7 are missed, and frames 5 and 7 fall inside the collar">']
    for t in range(12):
        out.append(f'<text x="{x0 + t * (w + gap) + w // 2}" y="46" text-anchor="middle" class="fig-small">{t}</text>')
    out.append('<text x="20" y="46" class="fig-small">frame</text>')
    for name, row, y in rows:
        out.append(f'<text x="20" y="{y + 24}" class="fig-small fig-bold">{name}</text>')
        for t in range(12):
            cls = ("fill-soft-a stroke-a" if y < 150 else "fill-soft-c stroke-c") if row[t] else "fill-sunk stroke-muted"
            out.append(f'<rect x="{x0 + t * (w + gap)}" y="{y}" width="{w}" height="32" rx="4" class="{cls}"/>')
    for t in range(12):
        if skip[t]:
            out.append(f'<rect x="{x0 + t * (w + gap) - 2}" y="54" width="{w + 4}" height="192" rx="5" class="stroke-muted dashed" fill="none"/>')
        if errors[t]:
            out.append(f'<rect x="{x0 + t * (w + gap) - 2}" y="98" width="{w + 4}" height="44" rx="5" class="stroke-bad" fill="none"/>')
    e0, d0 = der(REF, HYP)
    e1, d1 = der(REF, HYP, collar=1)
    e2, d2 = der(REF, HYP, score_overlap=False)
    out.append(f'<text x="300" y="274" text-anchor="middle" class="fig-small">no collar: {e0} errors / {d0} speaker-frames = {e0 / d0:.0%}   |   collar of 1 frame: {e1} / {d1} = {e1 / d1:.1%}</text>')
    out.append(f'<text x="300" y="292" text-anchor="middle" class="fig-small">overlap not scored: {e2} / {d2} = {e2 / d2:.0%}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    for name, kwargs in [("overlap scored, no collar", {}), ("collar of 1 frame", {"collar": 1}), ("overlap not scored", {"score_overlap": False})]:
        errors, total = der(REF, HYP, **kwargs)
        print(f"{name}: {errors} errors over {total} reference speaker-frames, DER {errors / total:.4f}")
    print("forgiven frames with a collar of 1:", np.flatnonzero(forgiven(REF, 1)).tolist())
    print(figure())
