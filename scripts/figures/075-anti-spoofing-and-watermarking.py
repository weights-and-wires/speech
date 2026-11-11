#!/usr/bin/env python3
"""Compute the worked examples of content/lessons/075-anti-spoofing-and-watermarking.md.

Run it from the repo root:

    python3 scripts/figures/075-anti-spoofing-and-watermarking.py

It prints the filterbank edges (Figure 1), the toy spoof detector's error rates,
the t-DCF table, the watermark error rates under noise and low-pass attacks
(Figure 2), and the attacker EERs for the anonymization example.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend/app/runner"))
sys.path.insert(0, str(ROOT / "content/exercises/075-anti-spoofing-and-watermarking"))
import solution as sol  # noqa: E402

RATE = 16000
N_FFT = 512

# Figure 1: where 20 filters put their centers ------------------------------------------
mel = lambda f: 2595 * np.log10(1 + f / 700)
inv = lambda m: 700 * (10 ** (m / 2595) - 1)
lin_edges = np.linspace(0, RATE / 2, 22)
mel_edges = inv(np.linspace(0, mel(RATE / 2), 22))
print("linear centers (Hz)", np.round(lin_edges[1:-1]).astype(int).tolist())
print("mel centers (Hz)   ", np.round(mel_edges[1:-1]).astype(int).tolist())
print("centers above 4 kHz: linear", int((lin_edges[1:-1] > 4000).sum()), "mel", int((mel_edges[1:-1] > 4000).sum()))
print("centers within 5.5-6.5 kHz: linear", int(((lin_edges[1:-1] > 5500) & (lin_edges[1:-1] < 6500)).sum()),
      "mel", int(((mel_edges[1:-1] > 5500) & (mel_edges[1:-1] < 6500)).sum()))


def mel_bank(n_filters):
    bins = np.fft.rfftfreq(N_FFT, 1 / RATE)
    e = inv(np.linspace(0, mel(RATE / 2), n_filters + 2))
    lo, mid, hi = e[:-2, None], e[1:-1, None], e[2:, None]
    return np.maximum(0.0, np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)))


# Toy detector: a voiced signal against the same signal with a 5.5-6.5 kHz dip of 0.35 dB -----------
def voice(rng):
    f0 = rng.uniform(110, 220)
    t = np.arange(RATE) / RATE
    x = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h**0.7 for h in range(1, int(7500 // f0)))
    x = lfilter([1.0, -rng.uniform(-0.5, 0.8)], [1.0], x)  # a random microphone and room tilt
    noise = 10 ** rng.uniform(-2.5, -1.5)
    return 0.2 * x / np.abs(x).max() + noise * rng.standard_normal(len(t))


band = butter(4, [5500, 6500], btype="bandpass", fs=RATE, output="sos")


def spoof_of(x):
    return x - 0.04 * sosfiltfilt(band, x)


def utterance_features(x, fb):
    return sol.lfcc(sol.power_spectrogram(x), fb, 20)[1:].mean(axis=1)


def logistic_fit(X, y, steps=2000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + 1e-3 * w)
        b -= lr * np.mean(p - y)
    return lambda X: ((X - mu) / sd) @ w + b


rng = np.random.default_rng(0)
voices = [voice(rng) for _ in range(160)]
for name, fb in (("LFCC (20 linear)", sol.lfcc_filterbank(20, N_FFT, RATE)), ("MFCC-style (20 mel)", mel_bank(20))):
    bona = np.array([utterance_features(v, fb) for v in voices])
    spf = np.array([utterance_features(spoof_of(v), fb) for v in voices])
    tr, te = slice(0, 80), slice(80, 160)
    X = np.vstack([bona[tr], spf[tr]])
    y = np.r_[np.ones(80), np.zeros(80)]
    score = logistic_fit(X, y)
    eer, th = sol.equal_error_rate(score(bona[te]), score(spf[te]))
    print(f"{name}: held-out EER {eer:.3f}")

# t-DCF worked table ----------------------------------------------------------------------
c1, c2 = sol.tdcf_costs(0.05, 0.01, 0.3)
print(f"\nC1 {c1:.6f} C2 {c2:.6f} ratio {c1 / c2:.4f}")
bona, spoof = np.array([0.9, 0.5, 0.4]), np.array([0.6, 0.3, 0.1, 0.0])
th = np.append(np.unique(np.r_[bona, spoof]), 0.91)
frr, far = sol.error_rates(bona, spoof, th)
for t, a, b in zip(th, frr, far):
    print(f"threshold {t:.2f} FRR {a:.3f} FAR {b:.3f} cost {(c1 * a + c2 * b) / min(c1, c2):.3f}")
print("min t-DCF", sol.min_tdcf(bona, spoof, c1, c2), "EER", sol.equal_error_rate(bona, spoof))

# Figure 2: watermark errors against attacks --------------------------------------------
N_BITS, TRIALS = 32, 20
host = sol.host_signal()


def ber(attack, strength=0.01):
    errors = 0
    for trial in range(TRIALS):
        r = np.random.default_rng(100 + trial)
        bits = r.integers(0, 2, N_BITS)
        marked = sol.embed_watermark(host, bits, key=trial, strength=strength)
        got = sol.detect_watermark(attack(marked, r), N_BITS, key=trial)
        errors += int(np.sum(got != bits))
    return errors / (N_BITS * TRIALS)


print("\nwatermark strength 0.01, host-to-mark", round(10 * np.log10(np.sum(host**2) / (0.01**2 * 32 * (len(host) // 32))), 1), "dB")
SNRS, CUTOFFS = (20, 10, 0, -5, -10, -15, -20), (7000, 4000, 2000, 1000, 500)
noise_ber, lowpass_ber = [], []
print("bit error rate against noise at SNR (dB):")
for snr in SNRS:
    noise_ber.append(ber(lambda y, r: sol.mix_at_snr(y, r.standard_normal(len(y)), snr)))
    print(f"  {snr:>4}: {noise_ber[-1]:.3f}")
print("bit error rate against a low-pass at (Hz):")
for cutoff in CUTOFFS:
    lowpass_ber.append(ber(lambda y, r: sol.lowpass(y, cutoff)))
    print(f"  {cutoff:>5}: {lowpass_ber[-1]:.3f}")


def resample_attack(y, r):
    """Down to 8 kHz and back up by linear interpolation, a stand-in for a lossy resampling path."""
    low = y[::2]
    return np.interp(np.arange(len(y)), np.arange(len(low)) * 2, low)


print("resample 16 -> 8 -> 16 kHz:", f"{ber(resample_attack):.3f}")
print("no attack:", f"{ber(lambda y, r: y):.3f}")

# Anonymization ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
n, d = 60, 16
spk = rng.standard_normal((n, d))
enrolled = spk + 0.5 * rng.standard_normal((n, d))
trials = spk + 0.5 * rng.standard_normal((n, d))
pool = rng.standard_normal((300, d))
print("\nattacker EER, no anonymization:", round(sol.attacker_eer(enrolled, trials), 3))
print("attacker EER, per-utterance swap, enrolled originals:", round(sol.attacker_eer(enrolled, sol.anonymize_embeddings(trials, pool, 20, rng)), 3))
fixed = sol.anonymize_embeddings(spk, pool, 20, rng)
print("attacker EER, one pseudo-speaker per person, enrolled anonymized:",
      round(sol.attacker_eer(fixed + 0.1 * rng.standard_normal((n, d)), fixed + 0.1 * rng.standard_normal((n, d))), 3))
a = sol.anonymize_embeddings(spk, pool, 20, rng)
b = sol.anonymize_embeddings(spk, pool, 20, rng)
print("attacker EER, a fresh pseudo-speaker each utterance, enrolled anonymized:", round(sol.attacker_eer(a, b), 3))


# SVG for Figures 1 and 2 -------------------------------------------------------------------
def fx(f):
    return 70 + f / 8000 * 480


svg1 = ['<svg viewBox="0 0 600 190" role="img" aria-label="Centers of 20 linear and 20 mel filters from 0 to 8 kilohertz; the linear filters cover 5.5 to 6.5 kilohertz with three centers, the mel filters with one">',
        f'  <rect x="{fx(5500):.0f}" y="30" width="{fx(6500) - fx(5500):.0f}" height="105" class="fill-soft-b"/>',
        '  <line x1="70" y1="135" x2="550" y2="135" class="stroke-muted"/>']
for f in (0, 2000, 4000, 6000, 8000):
    svg1.append(f'  <text x="{fx(f):.0f}" y="153" text-anchor="middle" class="fig-small">{f // 1000}</text>')
for y, edges, cls, name in ((65, lin_edges, "fill-a", "linear (LFCC)"), (110, mel_edges, "fill-c", "mel (MFCC)")):
    svg1.append(f'  <text x="70" y="{y - 14}" class="fig-small fig-bold">{name}</text>')
    for f in edges[1:-1]:
        svg1.append(f'  <circle cx="{fx(f):.1f}" cy="{y}" r="4" class="{cls}"/>')
svg1 += ['  <text x="310" y="176" text-anchor="middle" class="fig-small">filter center frequency (kHz)</text>', '</svg>']
print("\n".join(svg1))


def px(i, n, x0, x1):
    return x0 + i * (x1 - x0) / (n - 1)


def py(v):
    return 200 - v / 0.5 * 150


svg2 = ['<svg viewBox="0 0 620 270" role="img" aria-label="Bit error rate of a spread-spectrum watermark against added noise and against a low-pass filter">']
for panel, (x0, x1), vals, ticks, title in (
    (0, (70, 280), noise_ber, [f"{v}" for v in SNRS], "added noise, SNR (dB)"),
    (1, (370, 580), lowpass_ber, [f"{c / 1000:g}" for c in CUTOFFS], "low-pass cutoff (kHz)"),
):
    svg2.append(f'  <line x1="{x0 - 10}" y1="200" x2="{x1 + 10}" y2="200" class="stroke-muted"/><line x1="{x0 - 10}" y1="50" x2="{x0 - 10}" y2="200" class="stroke-muted"/>')
    for g in (0, 0.25, 0.5):
        svg2.append(f'  <line x1="{x0 - 10}" y1="{py(g):.1f}" x2="{x1 + 10}" y2="{py(g):.1f}" class="stroke-grid"/><text x="{x0 - 16}" y="{py(g) + 4:.1f}" text-anchor="end" class="fig-small">{g:.2f}</text>')
    pts = [(px(i, len(vals), x0, x1), py(v)) for i, v in enumerate(vals)]
    svg2.append('  <path d="M' + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) + '" fill="none" class="stroke-b" stroke-width="2"/>')
    for (a, b), v, t in zip(pts, vals, ticks):
        svg2.append(f'  <circle cx="{a:.1f}" cy="{b:.1f}" r="3.5" class="fill-b"/><text x="{a:.1f}" y="218" text-anchor="middle" class="fig-small">{t}</text>')
    svg2.append(f'  <text x="{(x0 + x1) / 2:.0f}" y="244" text-anchor="middle" class="fig-small">{title}</text>')
svg2.append('  <text x="16" y="125" class="fig-small" transform="rotate(-90 16 125)" text-anchor="middle">bit error rate</text>')
svg2.append("</svg>")
print("\n".join(svg2))
