"""Checks for the sampling and aliasing exercise."""

import numpy as np
from scipy.signal import firwin, resample_poly

from speechcheck import check, close, fail, has, min_snr, shape


@check("alias_frequency folds a tone into [0, fs/2]")
def test_alias_table(sub):
    alias = has(sub, "alias_frequency")
    table = [
        (7000, 10000, 3000), (9000, 16000, 7000), (10000, 16000, 6000),
        (25000, 16000, 7000), (8000, 16000, 8000), (440, 16000, 440),
        (15000, 16000, 1000), (48000, 16000, 0),
    ]
    for f, fs, want in table:
        got = alias(f, fs)
        if abs(float(got) - want) > 1e-6:
            fail(f"alias_frequency({f}, {fs}) returned {got}, expected {want}")


@check("A tone and its alias give exactly the same samples")
def test_alias_same_samples(sub):
    alias = has(sub, "alias_frequency")
    sample_tone = has(sub, "sample_tone")
    for f, fs in [(7000, 10000), (9000, 16000), (25000, 16000), (30000, 8000)]:
        a = float(alias(f, fs))
        close(sample_tone(f, fs, 0.01), sample_tone(a, fs, 0.01), atol=1e-8, label=f"a {f} Hz tone and its alias {a:g} Hz, both sampled at {fs} Hz")


@check("alias_frequency agrees with the spectrum of the sampled tone")
def test_alias_brute_force(sub):
    alias = has(sub, "alias_frequency")
    rng = np.random.default_rng(0)
    fs = 8000
    n = np.arange(fs)
    for _ in range(12):
        # Whole-hertz frequencies land exactly on the 1 Hz bins of a one-second FFT.
        f = int(rng.integers(50, 40000))
        spectrum = np.abs(np.fft.rfft(np.cos(2 * np.pi * f * n / fs)))
        measured = float(np.argmax(spectrum))
        got = float(alias(f, fs))
        if abs(got - measured) > 0.5:
            fail(f"a {f} Hz tone sampled at {fs} Hz peaks at {measured:.0f} Hz, but alias_frequency says {got:g}")


@check("lowpass_taps matches scipy.signal.firwin with a Kaiser window")
def test_lowpass(sub):
    lowpass = has(sub, "lowpass_taps")
    h5 = np.asarray(lowpass(5, 0.25, 5.0))
    close(h5, [0.0, 0.2066, 0.5869, 0.2066, 0.0], atol=1e-4, label="the five-tap example worked by hand in the chapter")
    for taps, cutoff, beta in [(61, 1 / 6, 5.0), (21, 0.25, 5.0), (101, 0.1, 8.0)]:
        h = lowpass(taps, cutoff, beta)
        shape(h, (taps,), label=f"lowpass_taps({taps}, {cutoff:.3g})")
        want = firwin(taps, 2 * cutoff, window=("kaiser", beta))
        close(h, want, atol=1e-12, label=f"taps for {taps} taps, cutoff {cutoff:.3g} cycles/sample")


@check("resample matches scipy.signal.resample_poly sample for sample")
def test_resample_matches_scipy(sub):
    resample = has(sub, "resample")
    rng = np.random.default_rng(1)
    for up, down, n in [(1, 3, 1000), (2, 3, 999), (3, 1, 200), (160, 441, 500), (4, 6, 300)]:
        x = rng.standard_normal(n)
        got = np.asarray(resample(x, up, down))
        want = resample_poly(x, up, down)
        if got.shape != want.shape:
            fail(f"resample(x, {up}, {down}) on {n} samples gave {got.shape[0]} samples, expected {want.shape[0]} = ceil({n} * {up} / {down})")
        close(got, want, atol=1e-9, label=f"resample by {up}/{down}")


@check("A 1 kHz tone resampled from 48 kHz to 16 kHz matches one synthesized at 16 kHz")
def test_resample_snr(sub):
    resample = has(sub, "resample")
    n_in = np.arange(48000)
    x = 0.5 * np.cos(2 * np.pi * 1000 * n_in / 48000)
    y = np.asarray(resample(x, 1, 3))
    shape(y, (16000,), label="one second at 16 kHz")
    want = 0.5 * np.cos(2 * np.pi * 1000 * np.arange(16000) / 16000)
    inner = slice(200, -200)  # the filter has only half its support at each end
    min_snr(y[inner], want[inner], 50.0, label="resampled against synthesized")


@check("Resampling removes the alias that every-third-sample decimation lets in")
def test_resample_antialias(sub):
    resample = has(sub, "resample")
    n_in = np.arange(48000)
    x = 0.5 * np.cos(2 * np.pi * 10000 * n_in / 48000)
    naive = x[::3]
    y = np.asarray(resample(x, 1, 3))[200:-200]
    level = 20 * np.log10(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(naive**2)))
    if level > -40:
        fail(f"a 10 kHz tone should vanish at 16 kHz, but it survives at {level:.1f} dB relative to naive decimation; expected below -40 dB")
