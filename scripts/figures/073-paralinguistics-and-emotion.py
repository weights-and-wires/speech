#!/usr/bin/env python3
"""Compute the worked examples and figures of content/lessons/073-paralinguistics-and-emotion.md.

Run it from the repo root:

    python3 scripts/figures/073-paralinguistics-and-emotion.py

It prints the jitter, shimmer, CCC, kappa, and UAR examples, and then the SVG
for Figures 1 to 3. Standard library and numpy only.
"""

from __future__ import annotations

import numpy as np

RATE = 16000


def jitter(t):
    t = np.asarray(t, float)
    return float(np.mean(np.abs(np.diff(t))) / np.mean(t))


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


# Hand examples ------------------------------------------------------------------------
T = np.array([100.0, 102.0, 98.0, 101.0])
print("periods", T.tolist(), "diffs", np.abs(np.diff(T)).tolist(), "mean diff", np.abs(np.diff(T)).mean(), "mean T", T.mean(), "jitter", round(jitter(T), 6))
A = np.array([0.50, 0.55, 0.45, 0.52])
print("amplitudes", A.tolist(), "diffs", np.round(np.abs(np.diff(A)), 4).tolist(), "mean diff", round(float(np.abs(np.diff(A)).mean()), 5), "mean A", A.mean(), "shimmer", round(jitter(A), 5))
gold = np.array([-1.0, 0.0, 1.0])
print("CCC shift +1", round(ccc(gold + 1, gold), 4), "scale 2", round(ccc(2 * gold, gold), 4), "pearson", np.corrcoef(2 * gold, gold)[0, 1])
tab = np.array([[20, 5], [10, 15]])
n = tab.sum()
po, pe = np.trace(tab) / n, (tab.sum(1) * tab.sum(0)).sum() / n**2
print("kappa table", tab.tolist(), "po", po, "pe", pe, "kappa", (po - pe) / (1 - pe))
tab2 = np.array([[81, 9], [9, 1]])
n = tab2.sum()
po, pe = np.trace(tab2) / n, (tab2.sum(1) * tab2.sum(0)).sum() / n**2
print("kappa table", tab2.tolist(), "po", po, "pe", pe, "kappa", round((po - pe) / (1 - pe), 6))
cm = np.array([[70, 8, 2], [5, 8, 2], [1, 1, 3]])
rec = np.diag(cm) / cm.sum(1)
print("confusion", cm.tolist(), "accuracy", np.trace(cm) / cm.sum(), "recalls", np.round(rec, 4).tolist(), "UAR", round(rec.mean(), 4))
print("always neutral: accuracy 0.8, UAR", round(1 / 3, 4))
st = np.array([100, 100, 150, 150, 200, 200.0])
print("f0 mean", st.mean(), "p20 p80", np.percentile(st, [20, 80]).tolist(), "range st", 12 * np.log2(200 / 100))
print("shift by 2 semitones multiplies F0 by", round(2 ** (2 / 12), 4))

# Figure 1: periods and peak amplitudes of a steady and a rough voice -----------------------
rng = np.random.default_rng(0)
n_cyc = 20
base = RATE / 120.0
steady_t = base * (1 + 0.002 * rng.standard_normal(n_cyc))
steady_a = 1 + 0.01 * rng.standard_normal(n_cyc)
rough_t = base * (1 + 0.025 * rng.standard_normal(n_cyc))
rough_a = 1 + 0.15 * rng.standard_normal(n_cyc)
j_s, j_r = jitter(steady_t), jitter(rough_t)
s_s, s_r = jitter(steady_a), jitter(rough_a)  # same formula, applied to amplitudes
print(f"fig1 over {n_cyc} cycles: steady jitter {j_s:.4f} shimmer {s_s:.4f}; rough jitter {j_r:.4f} shimmer {s_r:.4f}")


def panel(x0, series, lo, hi, title, ylabel, fmt):
    w, h, top = 230, 150, 50
    out = [f'  <line x1="{x0}" y1="{top + h}" x2="{x0 + w}" y2="{top + h}" class="stroke-muted"/>', f'  <line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + h}" class="stroke-muted"/>']
    out.append(f'  <text x="{x0}" y="{top - 22}" class="fig-bold">{title}</text>')
    for v in (lo, (lo + hi) / 2, hi):
        y = top + h - (v - lo) / (hi - lo) * h
        out.append(f'  <line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" class="stroke-grid"/>')
        out.append(f'  <text x="{x0 - 6}" y="{y + 4:.1f}" class="fig-small" text-anchor="end">{fmt(v)}</text>')
    for vals, cls in series:
        pts = " ".join(f"{x0 + (i + 0.5) / n_cyc * w:.1f},{top + h - (v - lo) / (hi - lo) * h:.1f}" for i, v in enumerate(vals))
        out.append(f'  <polyline points="{pts}" fill="none" class="{cls}"/>')
    out.append(f'  <text x="{x0 + w / 2}" y="{top + h + 20}" class="fig-small" text-anchor="middle">cycle number</text>')
    out.append(f'  <text x="{x0 - 44}" y="{top + h / 2}" class="fig-small" text-anchor="middle" transform="rotate(-90 {x0 - 44} {top + h / 2})">{ylabel}</text>')
    return out


fig1 = ['<svg viewBox="0 0 600 270" role="img" aria-label="Period and peak amplitude of 20 glottal cycles for a steady voice, nearly flat, and a rough voice, scattered">']
fig1 += panel(70, [(steady_t, "stroke-a"), (rough_t, "stroke-b")], 120, 148, "Period of each cycle", "period (samples)", lambda v: f"{v:.0f}")
fig1 += panel(350, [(steady_a, "stroke-a"), (rough_a, "stroke-b")], 0.4, 1.6, "Peak amplitude of each cycle", "amplitude (relative)", lambda v: f"{v:.1f}")
fig1.append('  <text x="70" y="256" class="fig-small fig-bold">' + f"steady: jitter {100 * j_s:.1f}%, shimmer {100 * s_s:.1f}%" + "</text>")
fig1.append('  <text x="350" y="256" class="fig-small fig-bold">' + f"rough: jitter {100 * j_r:.1f}%, shimmer {100 * s_r:.1f}%" + "</text>")
fig1.append("</svg>")
print("\n".join(fig1))

# Figure 3: CCC against Pearson for three predictions of one gold sequence ----------------
cx0, cy0, sc = 300, 140, 50.0  # origin and scale: 1 unit = 50 px
fig2 = ['<svg viewBox="0 0 600 290" role="img" aria-label="Three predictions plotted against the gold values -1, 0, 1: a perfect one, one shifted by 1, and one doubled, with CCC 1, 0.571, and 0.8, all with Pearson 1">']
fig2.append(f'  <line x1="{cx0 - 130}" y1="{cy0}" x2="{cx0 + 130}" y2="{cy0}" class="stroke-grid"/><line x1="{cx0}" y1="{cy0 - 125}" x2="{cx0}" y2="{cy0 + 125}" class="stroke-grid"/>')
fig2.append(f'  <text x="{cx0 + 130}" y="{cy0 + 18}" class="fig-small" text-anchor="end">gold</text><text x="{cx0 + 8}" y="{cy0 - 112}" class="fig-small">predicted</text>')
specs = [("perfect, y = x", gold, "stroke-ink", "fill-ink"), ("shifted, y = x + 1", gold + 1, "stroke-b", "fill-b"), ("doubled, y = 2x", 2 * gold, "stroke-a", "fill-a")]
for k, (name, pred, ln, fl) in enumerate(specs):
    pts = " ".join(f"{cx0 + g * sc:.1f},{cy0 - p * sc:.1f}" for g, p in zip(gold, pred))
    fig2.append(f'  <polyline points="{pts}" fill="none" class="{ln}"/>')
    for g, p in zip(gold, pred):
        fig2.append(f'  <circle cx="{cx0 + g * sc:.1f}" cy="{cy0 - p * sc:.1f}" r="4" class="{fl}"/>')
    r = np.corrcoef(pred, gold)[0, 1]
    ty = 40 + 22 * k
    fig2.append(f'  <line x1="20" y1="{ty - 4}" x2="44" y2="{ty - 4}" class="{ln}"/>')
    fig2.append(f'  <text x="52" y="{ty}" class="fig-small">{name}: Pearson {r:.2f}, CCC {ccc(pred, gold):.3f}</text>')
fig2.append("</svg>")
print("\n".join(fig2))

# Figure 2: a confusion matrix with accuracy and UAR --------------------------------------
names = ["neutral", "angry", "sad"]
fig3 = ['<svg viewBox="0 0 600 270" role="img" aria-label="Confusion matrix of 100 clips with recalls 0.875, 0.533, and 0.600; accuracy 0.81 and unweighted average recall 0.669">']
x0, y0, cell = 150, 50, 62
fig3.append(f'  <text x="{x0 + 1.5 * cell}" y="{y0 - 28}" class="fig-small" text-anchor="middle">predicted</text>')
fig3.append(f'  <text x="40" y="{y0 + 1.5 * cell}" class="fig-small" text-anchor="middle" transform="rotate(-90 40 {y0 + 1.5 * cell})">true</text>')
for j, nm in enumerate(names):
    fig3.append(f'  <text x="{x0 + (j + 0.5) * cell}" y="{y0 - 8}" class="fig-small" text-anchor="middle">{nm}</text>')
for i, nm in enumerate(names):
    fig3.append(f'  <text x="{x0 - 8}" y="{y0 + (i + 0.5) * cell + 4}" class="fig-small" text-anchor="end">{nm}</text>')
    for j in range(3):
        share = cm[i, j] / cm[i].sum()
        cls = "fill-soft-b" if i == j else "fill-sunk"
        fig3.append(f'  <rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell}" height="{cell}" class="{cls} stroke-grid"/>')
        fig3.append(f'  <text x="{x0 + (j + 0.5) * cell}" y="{y0 + (i + 0.5) * cell + 5}" text-anchor="middle" class="{"fig-bold" if i == j else ""}">{cm[i, j]}</text>')
    fig3.append(f'  <text x="{x0 + 3 * cell + 16}" y="{y0 + (i + 0.5) * cell + 4}" class="fig-small">recall {rec[i]:.3f} ({cm[i].sum()} clips)</text>')
fig3.append(f'  <text x="{x0}" y="{y0 + 3 * cell + 30}" class="fig-small fig-bold">accuracy = 81 / 100 = {np.trace(cm) / cm.sum():.2f}</text>')
fig3.append(f'  <text x="{x0}" y="{y0 + 3 * cell + 50}" class="fig-small fig-bold">UAR = ({rec[0]:.3f} + {rec[1]:.3f} + {rec[2]:.3f}) / 3 = {rec.mean():.3f}</text>')
fig3.append("</svg>")
print("\n".join(fig3))
