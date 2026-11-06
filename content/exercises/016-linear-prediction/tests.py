"""Checks for the linear prediction exercise."""

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import freqz, lfilter

from speechcheck import check, close, fail, finite, has, shape

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


@check("prediction_error subtracts the weighted past, and undoes an all-pole filter")
def test_prediction_error(sub):
    prediction_error = has(sub, "prediction_error")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    got = np.asarray(prediction_error(x, np.array([2.0, -1.0])), dtype=float)
    shape(got, (4,), label="prediction_error of four samples")
    close(got, [1.0, 0.0, 0.0, 0.0], atol=1e-12, label="the error of the ramp 1, 2, 3, 4 under x[n] = 2 x[n-1] - x[n-2]")
    rng = np.random.default_rng(0)
    driver = rng.standard_normal(500)
    x = lfilter([1.0], [1.0, -1.3, 0.9], driver)
    close(prediction_error(x, np.array([1.3, -0.9])), driver, atol=1e-9,
          label="the error of an all-pole signal, which is the noise that drove it")


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


@check("lpc windows the frame, matches the Toeplitz solution, and leaves an error far below the vowel")
def test_lpc(sub):
    lpc, prediction_error = has(sub, "lpc"), has(sub, "prediction_error")
    y = reference_vowel(120.0, [730.0, 1090.0, 2440.0], [80.0, 100.0, 150.0])
    frame = y[3000:3512]
    alpha, E = lpc(frame, 18)
    alpha = np.asarray(alpha, dtype=float)
    shape(alpha, (18,), label="lpc(frame, 18)")
    w = frame * np.hamming(512)
    r = np.correlate(w, w, mode="full")[511:530]
    close(alpha, solve_toeplitz(r[:18], r[1:19]), rtol=1e-6, atol=1e-8, label="alpha against the Toeplitz solve of the windowed frame")
    close(E, np.sum(prediction_error(np.concatenate([w, np.zeros(18)]), alpha) ** 2), rtol=1e-6, label="E against the energy of the prediction error")
    gain = 10 * np.log10(np.sum(w**2) / E)
    if gain < 20.0:
        fail(f"the prediction error is only {gain:.1f} dB below the windowed vowel; expected at least 20 dB at order 18")


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
