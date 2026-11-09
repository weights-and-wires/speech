#!/usr/bin/env python3
"""Compute the microphone-array figures and the worked-example numbers.

The delays, the GCC-PHAT curve, and both SVGs in
content/lessons/011-microphone-arrays.md come from this script. Run it
from the repo root:

    python3 scripts/figures/microphone-arrays.py

It prints the sample delays, the parabolic-interpolation bias of a sinc,
and the SVG for Figure 1 and Figure 2.
"""

from __future__ import annotations

import numpy as np

SPEED_M_S = 343.0  # dry air at 20 C; see the lesson's further reading
SAMPLE_RATE = 16000
SPACING_M = 0.05
ANGLE_DEG = 30.0
N_MICS = 4


def sample_delays() -> np.ndarray:
    """Arrival delay of a plane wave, in samples, at each microphone."""
    sine = np.sin(np.deg2rad(ANGLE_DEG))
    seconds = np.arange(N_MICS) * SPACING_M * sine / SPEED_M_S
    return seconds * SAMPLE_RATE


def sinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    nonzero = np.abs(x) > 1e-12
    out[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
    return out


def parabola_bias() -> float:
    """Largest |estimate - tau| of a three-point parabola on sinc(k - tau)."""

    def estimate(tau: float) -> float:
        lags = np.arange(-8, 9)
        values = sinc(lags - tau)
        k = int(lags[int(np.argmax(values))])
        left, mid, right = (sinc(np.array([k - 1 - tau, k - tau, k + 1 - tau])))
        denom = left - 2 * mid + right
        delta = 0.0 if denom == 0 else 0.5 * (left - right) / denom
        return k + delta

    taus = np.linspace(-0.5, 0.5, 2001)
    return float(np.max(np.abs([estimate(float(tau)) - tau for tau in taus])))


def gcc_curve(delay: float = 2.5, n: int = 256, seed: int = 0):
    """GCC-PHAT of a circular delay. Lags are samples by which x2 lags x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    freq = np.fft.rfftfreq(n)
    x2 = np.fft.irfft(np.fft.rfft(x1) * np.exp(-2j * np.pi * freq * delay), n)
    cross = np.fft.rfft(x1) * np.conj(np.fft.rfft(x2))
    magnitude = np.abs(cross)
    phat = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-12)
    corr = np.fft.irfft(phat, n)
    lags = np.arange(n)
    lags = np.where(lags <= n // 2, lags, lags - n)
    # X1 * conj(X2) peaks at the opposite sign from "x2 lags x1".
    order = np.argsort(-lags)
    return -lags[order], corr[order]


def figure_delays(delays: np.ndarray) -> str:
    """Bar chart of the four computed sample delays."""
    baseline = 228
    left = 108
    step = 120
    width = 56
    pixels_per_sample = 46
    bars = []
    for index, delay in enumerate(delays):
        x = left + index * step
        height = delay * pixels_per_sample
        y = baseline - height
        label = f"{delay:.3f}"
        bars.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height:.1f}" class="fill-a"/>'
        )
        bars.append(
            f'  <text x="{x + width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="fig-mono fig-small">{label}</text>'
        )
        bars.append(
            f'  <text x="{x + width / 2:.1f}" y="{baseline + 22:.1f}" text-anchor="middle" class="fig-small">mic {index}</text>'
        )
    grid = []
    for samples, label in ((0, "0"), (1, "1"), (2, "2"), (3, "3")):
        y = baseline - samples * pixels_per_sample
        grid.append(
            f'  <line x1="78" y1="{y:.1f}" x2="580" y2="{y:.1f}" class="stroke-grid"/>'
        )
        grid.append(
            f'  <text x="70" y="{y + 4:.1f}" text-anchor="end" class="fig-small">{label}</text>'
        )
    body = "\n".join(grid + bars)
    return f'''<svg viewBox="0 0 640 280" role="img" aria-label="Four microphones in a line. A 30 degree plane wave arrives 0, 1.166, 2.332, and 3.499 samples later.">
  <text x="320" y="28" text-anchor="middle" class="fig-bold">Arrival delay along a 5 cm array</text>
  <text x="24" y="120" class="fig-small">samples</text>
{body}
  <line x1="78" y1="{baseline}" x2="580" y2="{baseline}" class="stroke-muted"/>
  <text x="320" y="268" text-anchor="middle" class="fig-small">30 degrees from broadside, 343 m/s, 16 kHz</text>
</svg>'''


def figure_gcc(lags: np.ndarray, corr: np.ndarray) -> str:
    """The GCC-PHAT peak around a planted delay of 2.5 samples."""
    keep = (lags >= -1) & (lags <= 7)
    lags = lags[keep]
    corr = corr[keep]
    x0, x1 = 78, 600
    y0, y1 = 214, 48  # y0 is the bottom (lag axis), y1 is the top

    def x_of(lag: float) -> float:
        return x0 + (lag - (-1)) / 8 * (x1 - x0)

    def y_of(value: float) -> float:
        return y0 - (value - (-0.30)) / 1.05 * (y0 - y1)

    points = " ".join(f"{x_of(lag):.1f},{y_of(value):.1f}" for lag, value in zip(lags, corr))
    ticks = []
    for lag in range(0, 8):
        x = x_of(lag)
        ticks.append(f'  <line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 6}" class="stroke-muted"/>')
        ticks.append(f'  <text x="{x:.1f}" y="{y0 + 22}" text-anchor="middle" class="fig-small">{lag}</text>')
    for value, label in ((0.0, "0"), (0.5, "0.5")):
        y = y_of(value)
        ticks.append(f'  <line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="stroke-grid"/>')
        ticks.append(f'  <text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="fig-small">{label}</text>')
    peak_x = x_of(2.5)
    # The two bins beside 2.5 are equal, so the line between them is the peak height.
    peak_y = y_of(float(corr[lags == 2][0]))
    tick_body = "\n".join(ticks)
    return f'''<svg viewBox="0 0 640 280" role="img" aria-label="GCC-PHAT against lag in samples. The curve peaks halfway between lag 2 and lag 3, at the planted delay of 2.5 samples.">
  <text x="320" y="28" text-anchor="middle" class="fig-bold">GCC-PHAT of a 2.5-sample delay</text>
  <text x="22" y="130" class="fig-small">correlation</text>
{tick_body}
  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="stroke-muted"/>
  <line x1="{x0}" y1="{y1 - 8}" x2="{x0}" y2="{y0}" class="stroke-muted"/>
  <polyline points="{points}" class="stroke-a" fill="none"/>
  <line x1="{peak_x:.1f}" y1="{y1}" x2="{peak_x:.1f}" y2="{y0}" class="stroke-b dashed"/>
  <circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="5" class="fill-b"/>
  <text x="{peak_x + 8:.1f}" y="{peak_y - 10:.1f}" class="fig-small">2.5 samples</text>
  <text x="340" y="268" text-anchor="middle" class="fig-small">lag of microphone 2 behind microphone 1 (samples)</text>
</svg>'''


def main() -> None:
    delays = sample_delays()
    print("samples per spacing", f"{delays[1]:.6f}")
    print("delays", " ".join(f"{delay:.3f}" for delay in delays))
    print("alias frequency Hz", SPEED_M_S / (2 * SPACING_M))
    print("sinc parabola bias samples", f"{parabola_bias():.3f}")
    print("10 log10(4)", 10 * np.log10(4))
    lags, corr = gcc_curve()
    for lag in range(-1, 8):
        value = float(corr[np.where(np.isclose(lags, lag))[0][0]])
        print(f"lag {lag:2d} {value:.6f}")
    print("FIGURE 1")
    print(figure_delays(delays))
    print("FIGURE 2")
    print(figure_gcc(lags, corr))


if __name__ == "__main__":
    main()
