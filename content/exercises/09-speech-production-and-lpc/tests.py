"""Checks for the speech production and linear prediction exercise."""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz, lfilter

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000


def reference_vowel(f0, formants, bandwidths, seconds=0.5):
    """An independent source-filter vowel: impulses through scipy's two-pole filters."""
    n = int(seconds * RATE)
    x = np.zeros(n)
    x[np.round(np.arange(0.0, n, RATE / f0)).astype(int).clip(max=n - 1)] = 1.0
    for f, bw in zip(formants, bandwidths):
        r = np.exp(-np.pi * bw / RATE)
        x = lfilter([1.0], [1.0, -2.0 * r * np.cos(2.0 * np.pi * f / RATE), r * r], x)
    return x / np.max(np.abs(x))


def lpc_of(sub, frame, order):
    r = sub.autocorrelation(frame * np.hamming(len(frame)), order)
    alpha, E, _ = sub.levinson_durbin(r, order)
    return np.asarray(alpha, dtype=float), float(E)


@check("impulse_train puts unit impulses one period apart")
def test_impulses(sub):
    impulse_train = has(sub, "impulse_train")
    x = np.asarray(impulse_train(100.0, RATE, 1000), dtype=float)
    shape(x, (1000,), label="impulse_train(100, 16000, 1000)")
    equal(np.nonzero(x)[0], np.array([0, 160, 320, 480, 640, 800, 960]), label="the impulse positions at 100 Hz")
    close(x.sum(), 7.0, label="the number of impulses")
    y = np.asarray(impulse_train(150.0, RATE, 500), dtype=float)
    equal(np.nonzero(y)[0], np.array([0, 107, 213, 320, 427]), label="the impulse positions at 150 Hz, rounded")


@check("resonator gives Klatt's coefficients, unit gain at 0 Hz, and a peak at its formant")
def test_resonator(sub):
    resonator = has(sub, "resonator")
    b, a = resonator(500.0, 100.0, RATE)
    close(np.asarray(a, dtype=float), [1.0, -1.92343087, 0.96149116], atol=1e-7, label="a for 500 Hz, 100 Hz bandwidth")
    close(np.asarray(b, dtype=float).ravel(), [0.03806029], atol=1e-7, label="b for 500 Hz, 100 Hz bandwidth")
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


@check("autocorrelation matches np.correlate for lags 0 to p")
def test_autocorrelation(sub):
    autocorrelation = has(sub, "autocorrelation")
    x = np.random.default_rng(0).standard_normal(300)
    got = np.asarray(autocorrelation(x, 12), dtype=float)
    shape(got, (13,), label="autocorrelation(x, 12)")
    close(got, np.correlate(x, x, mode="full")[299:312], atol=1e-10, label="r[0] to r[12]")


@check("levinson_durbin reproduces the worked example r = [1, 0.5, 0.1], p = 2")
def test_levinson_hand(sub):
    levinson_durbin = has(sub, "levinson_durbin")
    alpha, E, k = levinson_durbin(np.array([1.0, 0.5, 0.1]), 2)
    close(alpha, [0.6, -0.2], atol=1e-12, label="alpha")
    close(E, 0.72, atol=1e-12, label="the prediction error E")
    close(k, [0.5, -0.2], atol=1e-12, label="the reflection coefficients k")


@check("levinson_durbin solves the Toeplitz normal equations, keeps |k| < 1, and E is the residual energy")
def test_levinson_general(sub):
    levinson_durbin = has(sub, "levinson_durbin")
    rng = np.random.default_rng(1)
    for trial, p in enumerate((4, 10, 18)):
        # A stable AR process: poles inside the unit circle.
        poles = 0.9 * np.exp(1j * rng.uniform(0.2, 2.9, 3))
        a_true = np.real(np.poly(np.concatenate([poles, poles.conj()])))
        x = lfilter([1.0], a_true, rng.standard_normal(2000)) * np.hamming(2000)
        r = np.correlate(x, x, mode="full")[1999:2000 + p]
        alpha, E, k = levinson_durbin(r, p)
        close(alpha, solve_toeplitz(r[:p], r[1 : p + 1]), rtol=1e-7, atol=1e-9, label=f"alpha at order {p}")
        if np.any(np.abs(k) >= 1.0):
            fail(f"a reflection coefficient has magnitude {np.max(np.abs(k)):.4f}; the autocorrelation method keeps every |k_i| < 1")
        residual = lfilter(np.concatenate([[1.0], -np.asarray(alpha)]), [1.0], np.concatenate([x, np.zeros(p)]))
        close(E, np.sum(residual**2), rtol=1e-7, label=f"E against the energy of the prediction error at order {p}")


@check("lpc_envelope equals E / |A|^2 from scipy.signal.freqz")
def test_envelope(sub):
    lpc_envelope = has(sub, "lpc_envelope")
    alpha = np.array([1.3, -0.9, 0.3])
    got = np.asarray(lpc_envelope(alpha, 0.5, 512), dtype=float)
    shape(got, (257,), label="lpc_envelope with n_fft=512")
    _, h = freqz([np.sqrt(0.5)], np.concatenate([[1.0], -alpha]), worN=257, include_nyquist=True)
    close(got, np.abs(h) ** 2, rtol=1e-8, label="the all-pole power spectrum")


@check("formants finds 500, 1500, and 2500 Hz in a synthetic neutral vowel, within 5%")
def test_formants(sub):
    formants = has(sub, "formants")
    for f0 in (100.0, 130.0):
        y = reference_vowel(f0, [500.0, 1500.0, 2500.0], [60.0, 90.0, 120.0])
        alpha, _ = lpc_of(sub, y[3000:3512], 18)
        freqs, bands = formants(alpha, RATE)
        freqs = np.asarray(freqs, dtype=float)
        if len(freqs) < 3:
            fail(f"found only {len(freqs)} formants, {np.round(freqs)} Hz, at F0 = {f0:.0f} Hz; expected three")
        error = np.abs(freqs[:3] - [500.0, 1500.0, 2500.0]) / np.array([500.0, 1500.0, 2500.0])
        if np.any(error > 0.05):
            fail(f"at F0 = {f0:.0f} Hz the first three formants are {np.round(freqs[:3])} Hz; expected 500, 1500, and 2500 within 5%")
        if np.any(np.asarray(bands, dtype=float) >= 400.0):
            fail("formants should drop any pole whose bandwidth is 400 Hz or more")


@check("real_cepstrum is the inverse FFT of the log magnitude, and an echo shows up at its delay")
def test_cepstrum(sub):
    real_cepstrum = has(sub, "real_cepstrum")
    x = np.random.default_rng(2).standard_normal(256)
    got = np.asarray(real_cepstrum(x), dtype=float)
    shape(got, (256,), label="the cepstrum of 256 samples")
    want = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(x)) + 1e-10)))
    close(got, want, atol=1e-9, label="ifft(log|fft(x)|)")
    echo = np.zeros(32)
    echo[0], echo[4] = 1.0, 0.5
    c = np.asarray(real_cepstrum(echo), dtype=float)
    close(c[[4, 8, 12]], [0.2507, -0.0639, 0.0240], atol=1e-3, label="the cepstrum of a 0.5 echo at 4 samples, at quefrencies 4, 8, and 12")


@check("cepstral_pitch finds F0 within 2% on synthetic vowels at 120 Hz and 210 Hz")
def test_cepstral_pitch(sub):
    cepstral_pitch = has(sub, "cepstral_pitch")
    for f0, formants_hz in [(120.0, [730.0, 1090.0, 2440.0]), (210.0, [300.0, 870.0, 2240.0])]:
        y = reference_vowel(f0, formants_hz, [80.0, 100.0, 150.0])
        got = float(cepstral_pitch(y[2000:2800], RATE, 60.0, 400.0))
        if abs(got - f0) > 0.02 * f0:
            fail(f"cepstral_pitch returned {got:.1f} Hz for a vowel at {f0:.0f} Hz; expected within 2%")


@check("Source and filter are independent: your vowels at 100 Hz and 160 Hz keep the same formants")
def test_source_filter(sub):
    synthesize_vowel, formants = has(sub, "synthesize_vowel"), has(sub, "formants")
    targets = np.array([730.0, 1090.0, 2440.0])
    found = []
    for f0 in (100.0, 160.0):
        y = np.asarray(synthesize_vowel(f0, targets, [80.0, 100.0, 150.0], RATE, 0.5), dtype=float)
        shape(y, (8000,), label=f"a half-second vowel at {f0:.0f} Hz")
        close(np.max(np.abs(y)), 0.5, atol=1e-9, label="the peak amplitude")
        period = int(RATE / f0)
        drift = float(np.max(np.abs(y[2000 + period : 6000 + period] - y[2000:6000])))
        if drift > 1e-6:
            fail(f"after the first 2000 samples, the vowel at {f0:.0f} Hz should repeat exactly every {period} samples, "
                 f"but y[n + {period}] and y[n] differ by up to {drift:.3g}")
        alpha, _ = lpc_of(sub, y[3000:3512], 18)
        freqs = np.asarray(formants(alpha, RATE)[0], dtype=float)
        found.append(freqs[:3])
        if len(freqs) < 3 or np.any(np.abs(freqs[:3] - targets) > 0.05 * targets):
            fail(f"at F0 = {f0:.0f} Hz your vowel's formants are {np.round(freqs[:3])} Hz; expected {targets} within 5%")
