"""Checks for the filters, convolution, and reverb exercise."""

import numpy as np
import scipy.signal

from speechcheck import almost, check, close, equal, fail, has, shape
from speechlab import show_plot

RATE = 16000


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


def reference_rt60(h, rate, lo=-5.0, hi=-25.0):
    energy = np.cumsum((h**2)[::-1])[::-1]
    decay = 10 * np.log10(energy / energy[0])
    t = np.arange(len(h)) / rate
    keep = (decay <= lo) & (decay >= hi)
    return -60.0 / np.polyfit(t[keep], decay[keep], 1)[0]


@check("convolve_direct gives [1, 3, 5, 3] for [1, 2, 3] and [1, 1], and matches np.convolve")
def test_direct(sub):
    convolve_direct = has(sub, "convolve_direct")
    close(convolve_direct(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0])), [1, 3, 5, 3], label="the worked example")
    rng = np.random.default_rng(0)
    for nx, nh in [(50, 7), (7, 50), (1, 1), (300, 120)]:
        x, h = rng.standard_normal(nx), rng.standard_normal(nh)
        y = np.asarray(convolve_direct(x, h))
        shape(y, (nx + nh - 1,), label=f"convolving {nx} by {nh} samples")
        close(y, np.convolve(x, h), atol=1e-10, label=f"{nx} by {nh}")


@check("convolve_fft matches np.convolve, with no circular wrap-around")
def test_fft(sub):
    convolve_fft = has(sub, "convolve_fft")
    rng = np.random.default_rng(1)
    for nx, nh in [(1000, 257), (4096, 4096), (33, 1000), (5, 3)]:
        x, h = rng.standard_normal(nx), rng.standard_normal(nh)
        close(convolve_fft(x, h), np.convolve(x, h), atol=1e-9, label=f"{nx} by {nh}")
    x = np.zeros(1024)
    x[-1] = 1.0  # an impulse at the very end, and a long response after it
    h = np.exp(-np.arange(512) / 100.0)
    y = np.asarray(convolve_fft(x, h))
    if y.shape != (1535,):
        fail(f"convolving 1024 by 512 samples should give 1535, got {y.shape}")
    head = float(np.max(np.abs(y[:1023])))
    if head > 1e-9:
        fail(f"the response to an impulse at the last sample leaked into the start of the output (up to {head:.3g}); pad the FFT to at least len(x) + len(h) - 1")


@check("frequency_response matches scipy.signal.freqz")
def test_frequency_response(sub):
    frequency_response = has(sub, "frequency_response")
    filters = [
        ([1.0, -0.97], [1.0]),
        ([0.2, 0.3, 0.2], [1.0]),
        ([1.0], [1.0, -0.9]),
        ([0.05, 0.1, 0.05], [1.0, -1.3, 0.5]),
        ([2.0, 1.0], [2.0, -1.0]),
    ]
    for b, a in filters:
        w, H = frequency_response(b, a, 256)
        want_w, want_H = scipy.signal.freqz(b, a, worN=256)
        close(w, want_w, atol=1e-12, label=f"frequencies for b={b}, a={a}")
        cclose(H, want_H, atol=1e-9, label=f"response for b={b}, a={a}")


@check("lfilter runs the difference equation and matches scipy.signal.lfilter")
def test_lfilter(sub):
    lfilter = has(sub, "lfilter")
    close(lfilter([1.0], [1.0, -0.5], np.array([1.0, 0, 0, 0, 2.0])), [1, 0.5, 0.25, 0.125, 2.0625], atol=1e-12, label="y[n] = x[n] + 0.5 y[n-1]")
    x = np.random.default_rng(2).standard_normal(2000)
    filters = [
        ([0.2, 0.3, 0.2], [1.0]),
        ([1.0], [1.0, -0.9]),
        ([0.05, 0.1, 0.05], [1.0, -1.3, 0.5]),
        ([2.0, 1.0, 0.5], [2.0, -1.0, 0.3, 0.1]),
    ]
    for b, a in filters:
        close(lfilter(b, a, x), scipy.signal.lfilter(b, a, x), atol=1e-10, label=f"b={b}, a={a}")


@check("preemphasis subtracts 0.97 of the previous sample: -30.46 dB at 0 Hz, +5.89 dB at Nyquist")
def test_preemphasis(sub):
    preemphasis = has(sub, "preemphasis")
    frequency_response = has(sub, "frequency_response")
    close(preemphasis(np.array([1.0, 2.0, 3.0, 4.0])), [1.0, 1.03, 1.06, 1.09], atol=1e-12, label="preemphasis of [1, 2, 3, 4]")
    x = np.random.default_rng(3).standard_normal(1000)
    close(preemphasis(x, 0.9), scipy.signal.lfilter([1.0, -0.9], [1.0], x), atol=1e-12, label="alpha 0.9, against lfilter")
    _, H = frequency_response([1.0, -0.97], [1.0], 4096)
    almost(20 * np.log10(abs(H[0])), -30.4576, 1e-3, label="gain at 0 Hz in dB")
    almost(20 * np.log10(abs(H[-1])), 5.8893, 1e-3, label="gain just below Nyquist in dB")


@check("schroeder_decay_db integrates the remaining energy backward from the end")
def test_schroeder(sub):
    schroeder = has(sub, "schroeder_decay_db")
    n = np.arange(2000)
    a = 0.003
    h = np.exp(-a * n)
    got = np.asarray(schroeder(h))
    shape(got, (2000,), label="the decay curve")
    q = np.exp(-2 * a)
    want = 10 * np.log10((q**n - q**2000) / (1 - q**2000))
    close(got[:1900], want[:1900], atol=1e-8, label="the decay of an exponential")
    almost(got[0], 0.0, 1e-12, label="the first value")
    if np.any(np.diff(got) > 1e-12):
        fail("the Schroeder decay must never rise: the energy still to come can only shrink")


@check("estimate_rt60 reads the reverberation time of a known decay within 5%")
def test_estimate(sub):
    estimate = has(sub, "estimate_rt60")
    rng = np.random.default_rng(4)
    for rt60 in (0.3, 0.8):
        n = int(1.5 * rt60 * RATE)
        h = rng.standard_normal(n) * 10 ** (-3 * np.arange(n) / (rt60 * RATE))
        got = float(estimate(h, RATE))
        if abs(got - rt60) > 0.05 * rt60:
            fail(f"a decay built to fall 60 dB in {rt60} s measured {got:.3f} s, expected within 5%")


@check("synthetic_rir starts with the direct path and decays with the requested RT60")
def test_synthetic_rir(sub):
    synthetic_rir = has(sub, "synthetic_rir")
    h = np.asarray(synthetic_rir(0.5, RATE, seconds=0.75, seed=3))
    shape(h, (12000,), label="0.75 s at 16 kHz")
    almost(h[0], 1.0, 1e-12, label="the direct-path sample")
    equal(np.asarray(synthetic_rir(0.5, RATE, seconds=0.75, seed=3)), h, label="a second call with the same seed")
    worst = []
    for rt60 in (0.25, 0.5, 1.0):
        h = np.asarray(synthetic_rir(rt60, RATE, seed=0))
        measured = reference_rt60(h, RATE)
        worst.append(measured)
        if abs(measured - rt60) > 0.1 * rt60:
            energy = np.cumsum((h**2)[::-1])[::-1]
            show_plot((np.arange(len(h)) / RATE, 10 * np.log10(energy / energy[0])), label=f"Schroeder decay of your RIR, RT60 {rt60} s", xlabel="Time (s)", ylabel="dB")
            fail(f"an RIR asked for RT60 {rt60} s measures {measured:.3f} s by Schroeder integration, expected within 10%")
