#!/usr/bin/env python3
"""Compute the numbers and draw Figure 3 of content/lessons/020-voice-activity-detection.md.

Run it from the repo root:

    python3 scripts/figures/020-voice-activity-detection.py

It prints every number the lesson quotes about the adaptive noise floor, the
likelihood-ratio detector, and endpointing, then the SVG for Figure 3. The clip
is the exercise's synthetic speech with the noise rising 10 dB at 2 s.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXERCISE = ROOT / "content" / "exercises" / "020-voice-activity-detection"
sys.path.insert(0, str(ROOT / "backend" / "app" / "runner"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sol = load("vad_solution", EXERCISE / "solution.py")
tests = load("vad_tests", EXERCISE / "tests.py")
RATE = 16000


def numbers() -> None:
    for snr in (20.0, 10.0, 5.0):
        mix, ref = tests.noisy_speech(0, snr)
        print(f"energy_vad at {snr:.0f} dB SNR: frame F1 {sol.frame_f1(sol.energy_vad(mix, RATE), ref):.3f}")
    scores = []
    for seed in range(3):
        mix, ref = tests.noisy_speech(seed, 20.0, seconds=4.0, step_db=10.0, step_at=2.0)
        energy = sol.frame_energy_db(mix)
        adaptive, _ = sol.adaptive_vad(energy)
        scores.append((sol.frame_f1(sol.energy_vad(mix, RATE), ref), sol.frame_f1(adaptive, ref)))
    print("noise step, seeds 0 to 2 (fixed, adaptive):", [(round(a, 3), round(b, 3)) for a, b in scores])
    train = [tests.noisy_speech(100 + s, 5.0) for s in range(3)]
    params = sol.fit_gaussians(np.concatenate([sol.frame_features(m) for m, _ in train]), np.concatenate([r for _, r in train]))
    print("speech mean, var:", np.round(params[0], 1), np.round(params[1], 1))
    print("noise mean, var:", np.round(params[2], 1), np.round(params[3], 1))
    for snr in (10.0, 5.0):
        mix, ref = tests.noisy_speech(0, snr)
        feats = sol.frame_features(mix)
        for name, cols in (("energy only", [0]), ("energy and flatness", [0, 1])):
            llr = np.convolve(sol.gaussian_llr(feats[:, cols], params[0][cols], params[1][cols], params[2][cols], params[3][cols]), np.ones(5) / 5, mode="same")
            print(f"LLR VAD, {name}, {snr:.0f} dB SNR: frame F1 {sol.frame_f1(sol.hysteresis(llr, 1.0, -1.0, 5), ref):.3f}")
    mix, ref = tests.noisy_speech(0, 5.0)
    llr = np.convolve(sol.gaussian_llr(sol.frame_features(mix), *params), np.ones(5) / 5, mode="same")
    decisions = sol.hysteresis(llr, 1.0, -1.0, 5)
    last = np.flatnonzero(decisions[: np.flatnonzero(~decisions[np.argmax(decisions):])[0] + np.argmax(decisions)])[-1]
    print(f"first utterance: last speech frame {last}, endpoint(300 ms) at frame {sol.endpoint(decisions, 30)}")


def figure() -> str:
    mix, truth = tests.noisy_speech(1, 20.0, seconds=4.0, step_db=10.0, step_at=2.0)
    energy = sol.frame_energy_db(mix)
    fixed = sol.energy_vad(mix, RATE)
    adaptive, floor = sol.adaptive_vad(energy)
    fixed_on = np.percentile(energy, 10) + 9.0
    print(f"Figure 3: fixed F1 {sol.frame_f1(fixed, truth):.3f}, adaptive F1 {sol.frame_f1(adaptive, truth):.3f}, "
          f"fixed on threshold {fixed_on:.1f} dB, floor before {floor[190]:.1f} dB, floor after {floor[-1]:.1f} dB")
    x0, x1, top, bottom = 60.0, 570.0, 30.0, 230.0
    lo, hi = np.floor(energy.min() / 10) * 10, np.ceil(max(energy.max(), (floor + 9).max()) / 10) * 10

    def px(t):
        return x0 + (x1 - x0) * t / (len(energy) - 1)

    def py(v):
        return bottom - (bottom - top) * (v - lo) / (hi - lo)

    out = ['<svg viewBox="0 0 600 362" role="img" aria-label="Frame energy of a clip whose background noise rises 10 dB at 2 seconds, with a fixed on threshold and a tracked on threshold, above strips for the truth and the two detectors">']
    for v in np.arange(lo + 10, hi, 10):
        out.append(f'<line x1="60" y1="{py(v):.1f}" x2="570" y2="{py(v):.1f}" class="stroke-grid"/>')
        out.append(f'<text x="54" y="{py(v) + 4:.1f}" class="fig-small" text-anchor="end">{v:.0f}</text>')
    out.append('<line x1="60" y1="230" x2="570" y2="230" class="stroke-muted"/>')
    out.append('<line x1="60" y1="30" x2="60" y2="230" class="stroke-muted"/>')
    for s in range(5):
        xs = x0 + (x1 - x0) * s / 4
        out.append(f'<line x1="{xs:.1f}" y1="230" x2="{xs:.1f}" y2="234" class="stroke-muted"/>')
        out.append(f'<text x="{xs:.1f}" y="247" class="fig-small" text-anchor="middle">{s}</text>')
    out.append('<text x="60" y="22" class="fig-small" text-anchor="start">frame energy (dB)</text>')
    pts = " ".join(f"{px(t):.1f},{py(v):.1f}" for t, v in enumerate(energy))
    out.append(f'<polyline points="{pts}" class="stroke-a" fill="none"/>')
    out.append(f'<line x1="60" y1="{py(fixed_on):.1f}" x2="570" y2="{py(fixed_on):.1f}" class="stroke-ink dashed"/>')
    pts = " ".join(f"{px(t):.1f},{py(v + 9):.1f}" for t, v in enumerate(floor))
    out.append(f'<polyline points="{pts}" class="stroke-b dashed" fill="none"/>')
    out.append(f'<line x1="{px(200):.1f}" y1="30" x2="{px(200):.1f}" y2="230" class="stroke-muted dashed"/>')
    out.append(f'<text x="{px(200) - 4:.1f}" y="44" class="fig-small" text-anchor="end">noise rises 10 dB</text>')
    out.append(f'<text x="68" y="{py(fixed_on) - 5:.1f}" class="fig-small" text-anchor="start">fixed on threshold</text>')
    out.append(f'<text x="{px(250):.1f}" y="{py(floor[250] + 9) - 5:.1f}" class="fig-small" text-anchor="start">tracked floor + 9 dB</text>')
    rows = [("truth", truth, 250, "fill-soft-c"), ("fixed", fixed, 276, "fill-d"), ("tracked", adaptive, 302, "fill-b")]
    for name, mask, y, cls in rows:
        out.append(f'<text x="56" y="{y + 14}" class="fig-small" text-anchor="end">{name}</text>')
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            out.append(f'<rect x="{px(a):.1f}" y="{y}" width="{px(b) - px(a):.1f}" height="20" class="{cls}"/>')
    out.append('<line x1="60" y1="326" x2="570" y2="326" class="stroke-muted"/>')
    for s in range(5):
        xs = x0 + (x1 - x0) * s / 4
        out.append(f'<line x1="{xs:.1f}" y1="326" x2="{xs:.1f}" y2="330" class="stroke-muted"/>')
        out.append(f'<text x="{xs:.1f}" y="342" class="fig-small" text-anchor="middle">{s}</text>')
    out.append('<text x="315" y="359" text-anchor="middle">time (s)</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    numbers()
    print(figure())
