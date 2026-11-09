"""Checks for the source-filter model exercise."""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz, lfilter

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000


def reference_vowel(f0, formants, bandwidths, seconds=0.5):
    """An independent source-filter vowel: impulses through scipy's two-pole filters, peak 1."""
    n = int(seconds * RATE)
    x = np.zeros(n)
    x[np.round(np.arange(0.0, n, RATE / f0)).astype(int).clip(max=n - 1)] = 1.0
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / RATE)
        x = lfilter([1.0], [1.0, -2.0 * r * np.cos(2.0 * np.pi * f / RATE), r * r], x)
    return x / np.max(np.abs(x))


def formants_by_lpc(frame, order=18):
    """Formant frequencies in Hz from an independent autocorrelation LPC, with narrow poles only."""
    x = frame * np.hamming(len(frame))
    r = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) + order]
    alpha = solve_toeplitz(r[:order], r[1 : order + 1])
    roots = np.roots(np.concatenate([[1.0], -alpha]))
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * RATE / (2.0 * np.pi)
    bands = -np.log(np.abs(roots)) * RATE / np.pi
    keep = (freqs > 90.0) & (bands < 400.0)
    return np.sort(freqs[keep])


@check("impulse_train puts unit impulses one period apart")
def test_impulses(sub):
    impulse_train = has(sub, "impulse_train")
    x = np.asarray(impulse_train(100.0, RATE, 1000), dtype=float)
    shape(x, (1000,), label="impulse_train(100, 16000, 1000)")
    equal(np.nonzero(x)[0], np.array([0, 160, 320, 480, 640, 800, 960]), label="the impulse positions at 100 Hz")
    close(x.sum(), 7.0, label="the number of impulses")
    y = np.asarray(impulse_train(150.0, RATE, 500), dtype=float)
    equal(np.nonzero(y)[0], np.array([0, 107, 213, 320, 427]), label="the impulse positions at 150 Hz, rounded")


@check("An impulse train has a spectrum of lines at F0, 2 F0, 3 F0, and nothing in between")
def test_harmonics(sub):
    impulse_train = has(sub, "impulse_train")
    x = np.asarray(impulse_train(100.0, RATE, RATE), dtype=float)  # one second: one FFT bin per hertz
    magnitude = np.abs(np.fft.rfft(x))
    on_lines = magnitude[::100]
    close(on_lines, np.full(len(on_lines), 100.0), rtol=1e-9, label="the magnitude at 0, 100, 200, ... Hz (100 impulses add up in step)")
    between = np.delete(magnitude, np.arange(0, len(magnitude), 100))
    close(between, np.zeros(len(between)), atol=1e-8, label="the magnitude between the harmonics")


@check("resonator gives Klatt's coefficients, unit gain at 0 Hz, and a peak at its formant")
def test_resonator(sub):
    resonator = has(sub, "resonator")
    b, a = resonator(500.0, 100.0, RATE)
    close(np.asarray(a, dtype=float), [1.0, -1.92343087, 0.96149116], atol=1e-7, label="a for 500 Hz, 100 Hz bandwidth")
    close(np.asarray(b, dtype=float).ravel(), [0.03806029], atol=1e-7, label="b for 500 Hz, 100 Hz bandwidth")
    _, h = freqz(b, a, worN=[0.0, 500.0], fs=RATE)
    close(np.abs(h), [1.0, 5.05996592], atol=1e-6, label="the gain at 0 Hz and at 500 Hz (14.1 dB)")
    for f, bw in [(700.0, 80.0), (2300.0, 150.0)]:
        b, a = resonator(f, bw, RATE)
        grid = np.arange(0.0, 8000.0, 1.0)
        _, h = freqz(b, a, worN=grid, fs=RATE)
        close(abs(h[0]), 1.0, atol=1e-9, label=f"the gain at 0 Hz of the {f:.0f} Hz resonator")
        peak = grid[np.argmax(np.abs(h))]
        if abs(peak - f) > 0.02 * f:
            fail(f"the {f:.0f} Hz resonator peaks at {peak:.0f} Hz, expected within 2% of {f:.0f} Hz")
        radius = np.abs(np.roots(a))
        close(radius, np.full(2, np.exp(-np.pi * bw / RATE)), atol=1e-9, label="the pole radius exp(-pi B / fs)")


@check("A wider bandwidth moves the poles inward and lowers the peak")
def test_bandwidth(sub):
    resonator = has(sub, "resonator")
    gains, radii = [], []
    for bw in (50.0, 100.0, 400.0):
        b, a = resonator(1000.0, bw, RATE)
        _, h = freqz(b, a, worN=[1000.0], fs=RATE)
        gains.append(float(np.abs(h[0])))
        radii.append(float(np.abs(np.roots(a))[0]))
    if not (gains[0] > gains[1] > gains[2]):
        fail(f"the gains at 1000 Hz for bandwidths 50, 100, and 400 Hz are {np.round(gains, 2)}; expected them to fall as the bandwidth grows")
    if not (radii[0] > radii[1] > radii[2]):
        fail(f"the pole radii for bandwidths 50, 100, and 400 Hz are {np.round(radii, 4)}; expected them to shrink as the bandwidth grows")


@check("synthesize_vowel has the right length and peak, and matches an independent cascade")
def test_synthesize(sub):
    synthesize_vowel = has(sub, "synthesize_vowel")
    cases = [(120.0, [730.0, 1090.0, 2440.0], [80.0, 100.0, 150.0]), (210.0, [300.0, 870.0, 2240.0], [60.0, 90.0, 170.0])]
    for f0, formants, bandwidths in cases:
        y = np.asarray(synthesize_vowel(f0, formants, bandwidths, RATE, 0.5), dtype=float)
        shape(y, (8000,), label=f"a half-second vowel at {f0:.0f} Hz")
        close(np.max(np.abs(y)), 0.5, atol=1e-9, label="the peak amplitude")
        close(y, 0.5 * reference_vowel(f0, formants, bandwidths), atol=1e-9, label=f"the vowel at {f0:.0f} Hz against scipy's cascade")


@check("Source and filter are independent: your vowels at 100 Hz and 160 Hz keep the same formants")
def test_source_filter(sub):
    synthesize_vowel = has(sub, "synthesize_vowel")
    targets = np.array([730.0, 1090.0, 2440.0])
    for f0 in (100.0, 160.0):
        y = np.asarray(synthesize_vowel(f0, targets, [80.0, 100.0, 150.0], RATE, 0.5), dtype=float)
        period = int(RATE / f0)
        drift = float(np.max(np.abs(y[2000 + period : 6000 + period] - y[2000:6000])))
        if drift > 1e-6:
            fail(f"after the first 2000 samples, the vowel at {f0:.0f} Hz should repeat exactly every {period} samples, "
                 f"but y[n + {period}] and y[n] differ by up to {drift:.3g}")
        freqs = formants_by_lpc(y[3000:3512])
        if len(freqs) < 3 or np.any(np.abs(freqs[:3] - targets) > 0.05 * targets):
            fail(f"at F0 = {f0:.0f} Hz your vowel's formants are {np.round(freqs[:3])} Hz; expected {targets} within 5%")
