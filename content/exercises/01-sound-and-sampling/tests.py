"""Checks for the sound and sampling exercise."""

import numpy as np
from scipy.signal import firwin, resample_poly

from speechcheck import almost, check, close, fail, has, min_snr, shape


@check("rms is the square root of the mean square")
def test_rms(sub):
    rms = has(sub, "rms")
    almost(rms(np.array([0.5, -0.5, 0.5, -0.5])), 0.5, 1e-12, label="rms of [0.5, -0.5, 0.5, -0.5]")
    almost(rms(np.array([3.0, 4.0])), np.sqrt(12.5), 1e-12, label="rms of [3, 4]")
    n = np.arange(16000)
    sine = 0.8 * np.sin(2 * np.pi * 440 * n / 16000)
    almost(rms(sine), 0.8 / np.sqrt(2), 1e-6, label="rms of a 0.8-amplitude sine over whole periods")


@check("peak_dbfs and rms_dbfs give 0 and -3.01 dBFS for a full-scale sine")
def test_dbfs(sub):
    peak_dbfs = has(sub, "peak_dbfs")
    rms_dbfs = has(sub, "rms_dbfs")
    n = np.arange(16000)
    full = np.sin(2 * np.pi * 1000 * n / 16000 + np.pi / 2)
    half = 0.5 * full
    almost(peak_dbfs(full), 0.0, 1e-9, label="peak of a full-scale sine")
    almost(rms_dbfs(full), -3.0103, 1e-3, label="RMS of a full-scale sine")
    almost(peak_dbfs(half), -6.0206, 1e-3, label="peak of a 0.5 sine")
    almost(rms_dbfs(half), -9.0309, 1e-3, label="RMS of a 0.5 sine")
    silent = peak_dbfs(np.zeros(100))
    if not np.isfinite(silent):
        fail(f"peak_dbfs of silence returned {silent}; floor the level at eps before the log")
    almost(silent, -200.0, 1e-6, label="peak of silence with eps=1e-10")


@check("sample_tone takes A cos(2 pi f n / fs + phase) at every sample")
def test_sample_tone(sub):
    sample_tone = has(sub, "sample_tone")
    x = sample_tone(440.0, 16000, 0.5, amplitude=0.25, phase=0.3)
    shape(x, (8000,), label="half a second at 16 kHz")
    n = np.arange(8000)
    close(x, 0.25 * np.cos(2 * np.pi * 440.0 * n / 16000 + 0.3), atol=1e-12, label="the samples")
    high = sample_tone(7000.0, 10000, 0.01)
    low = sample_tone(3000.0, 10000, 0.01)
    close(high, low, atol=1e-9, label="7 kHz and 3 kHz cosines sampled at 10 kHz")


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
