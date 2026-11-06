"""Checks for the FFT and window functions exercise."""

import numpy as np
import scipy.signal

from speechcheck import almost, check, close, fail, has, shape, timed


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


SCIPY_NAMES = {"rectangular": "boxcar", "hann": "hann", "hamming": "hamming", "blackman": "blackman"}


@check("fft_radix2 gives [2, 1 - 3j, 0, 1 + 3j] for [1, 2, 0, -1]")
def test_worked_example(sub):
    fft = has(sub, "fft_radix2")
    got = np.asarray(fft(np.array([1.0, 2.0, 0.0, -1.0])))
    shape(got, (4,), label="fft_radix2 of four samples")
    cclose(got, [2, 1 - 3j, 0, 1 + 3j], atol=1e-12, label="the worked example")
    cclose(fft(np.array([1.0])), [1.0], atol=1e-12, label="a single sample is its own DFT")


@check("fft_radix2 matches np.fft.fft for every power of two up to 1024")
def test_fft(sub):
    fft = has(sub, "fft_radix2")
    rng = np.random.default_rng(1)
    for p in range(11):
        n = 2**p
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        with timed(f"fft_radix2 of {n} points"):
            got = fft(x)
        cclose(got, np.fft.fft(x), atol=1e-9 * n, label=f"length {n}")


@check("fft_radix2 raises ValueError for a length that is not a power of two")
def test_fft_length(sub):
    fft = has(sub, "fft_radix2")
    for n in (12, 3, 1000):
        try:
            fft(np.ones(n))
        except ValueError:
            continue
        fail(f"fft_radix2 of {n} samples should raise ValueError, because {n} is not a power of two")


@check("get_window matches scipy.signal.get_window, periodic and symmetric")
def test_windows(sub):
    get_window = has(sub, "get_window")
    for name, scipy_name in SCIPY_NAMES.items():
        for periodic in (True, False):
            for n in (1, 2, 7, 64, 400):
                got = get_window(name, n, periodic)
                want = scipy.signal.get_window(scipy_name, n, fftbins=periodic)
                close(got, want, atol=1e-12, label=f"{name}, n={n}, periodic={periodic}")


@check("A symmetric Hann window is 0 at both ends; a periodic one is 0 only at the start")
def test_periodic_vs_symmetric(sub):
    get_window = has(sub, "get_window")
    symmetric = np.asarray(get_window("hann", 8, False))
    periodic = np.asarray(get_window("hann", 8, True))
    almost(symmetric[0], 0.0, 1e-12, label="the first sample of the symmetric window")
    almost(symmetric[-1], 0.0, 1e-12, label="the last sample of the symmetric window")
    almost(periodic[0], 0.0, 1e-12, label="the first sample of the periodic window")
    if abs(periodic[-1]) < 1e-3:
        fail(f"the last sample of a periodic Hann of length 8 should be {0.5 - 0.5 * np.cos(2 * np.pi * 7 / 8):.4f}, not 0: it is the symmetric window of length 9 without its last sample")
    almost(periodic[4], 1.0, 1e-12, label="the middle sample of the periodic window")
    try:
        get_window("kaiser", 8)
    except ValueError:
        return
    fail("get_window('kaiser', 8) should raise ValueError for a name it does not know")


def _reference_leakage(w, oversample=64):
    s = np.abs(np.fft.rfft(w, len(w) * oversample))
    s = s / s[0]
    k = 1
    while k < len(s) - 1 and s[k + 1] < s[k]:
        k += 1
    return k / oversample, 20 * np.log10(s[k:].max())


@check("leakage finds each window's main-lobe width and peak side lobe")
def test_leakage(sub):
    leakage = has(sub, "leakage")
    known = {"rectangular": (1.0, -13.26), "hann": (2.0, -31.47), "hamming": (2.0, -42.45), "blackman": (3.0, -58.11)}
    for name, (width, level) in known.items():
        w = scipy.signal.get_window(SCIPY_NAMES[name], 64)
        got_width, got_level = leakage(w)
        want_width, want_level = _reference_leakage(w)
        almost(got_width, want_width, 1e-9, label=f"{name}: main-lobe half-width in bins")
        almost(got_level, want_level, 0.05, label=f"{name}: peak side lobe in dB")
        almost(got_level, level, 0.1, label=f"{name}: peak side lobe against the textbook value")
        almost(got_width, width, 0.01, label=f"{name}: half-width against the textbook value")


@check("A Hann window pushes the far leakage of an off-bin cosine from -26 dB to below -90 dB")
def test_far_leakage(sub):
    get_window = has(sub, "get_window")
    n = 64
    x = np.cos(2 * np.pi * 10.5 * np.arange(n) / n)  # halfway between bins 10 and 11

    def far_db(window):
        spectrum = np.abs(np.fft.rfft(x * window))
        return 20 * np.log10(spectrum[30] / spectrum.max())

    rectangular = far_db(np.asarray(get_window("rectangular", n)))
    hann = far_db(np.asarray(get_window("hann", n)))
    almost(rectangular, -26.2, 0.1, label="bin 30 with no window, in dB below the peak")
    almost(hann, -92.7, 0.5, label="bin 30 with a periodic Hann window, in dB below the peak")


@check("Zero-padding refines the grid but can't separate tones closer than the window allows")
def test_zero_padding(sub):
    get_window = has(sub, "get_window")
    rate = 16000
    t = np.arange(rate) / rate
    x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1020 * t)

    def peaks(length, n_fft):
        spec = np.abs(np.fft.rfft(x[:length] * get_window("hann", length), n_fft))
        f = np.fft.rfftfreq(n_fft, 1 / rate)
        band = (f > 900) & (f < 1120)
        s = spec[band] / spec[band].max()
        return int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > 0.5)))

    short = peaks(400, 16384)   # 25 ms: the main lobe is 80 Hz wide either side
    long = peaks(3200, 16384)   # 200 ms: 10 Hz either side
    if short != 1:
        fail(f"a 25 ms Hann window zero-padded to 16384 points should show one merged peak near 1 kHz, found {short}")
    if long != 2:
        fail(f"a 200 ms Hann window should separate 1000 Hz and 1020 Hz into 2 peaks, found {long}")
