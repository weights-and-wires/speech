"""Checks for the convolution and filters exercise."""

import numpy as np
import scipy.signal

from speechcheck import almost, check, close, fail, has, shape


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


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


@check("Convolution is linear and time-invariant: it adds, scales, and carries a delay through")
def test_lti(sub):
    convolve_direct = has(sub, "convolve_direct")
    rng = np.random.default_rng(5)
    x1, x2, h = rng.standard_normal(40), rng.standard_normal(40), rng.standard_normal(9)
    close(convolve_direct(2.0 * x1 - 3.0 * x2, h), 2.0 * convolve_direct(x1, h) - 3.0 * convolve_direct(x2, h), atol=1e-10, label="linearity: (2 x1 - 3 x2) * h")
    delayed = np.concatenate([np.zeros(5), x1])
    close(convolve_direct(delayed, h)[5:], convolve_direct(x1, h), atol=1e-10, label="a 5-sample delay of the input delays the output")
    impulse = np.zeros(6)
    impulse[0] = 1.0
    close(convolve_direct(impulse, h)[: len(h)], h, atol=1e-12, label="an impulse in gives the impulse response out")


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


@check("An FIR filter's impulse response is its coefficients; an IIR filter's rings on")
def test_impulse_responses(sub):
    lfilter = has(sub, "lfilter")
    impulse = np.zeros(12)
    impulse[0] = 1.0
    close(lfilter([0.2, 0.5, 0.3], [1.0], impulse), [0.2, 0.5, 0.3] + [0.0] * 9, atol=1e-12, label="FIR: the impulse response is b, then zeros")
    close(lfilter([1.0], [1.0, -0.5], impulse), 0.5 ** np.arange(12), atol=1e-12, label="IIR with a pole at 0.5: 1, 0.5, 0.25, ...")
    unstable = np.asarray(lfilter([1.0], [1.0, -1.1], impulse))
    almost(unstable[-1], 1.1**11, 1e-9, label="a pole at 1.1, outside the unit circle, grows by 1.1 each step")


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
