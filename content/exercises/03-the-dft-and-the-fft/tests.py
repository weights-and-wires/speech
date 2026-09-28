"""Checks for the DFT and FFT exercise."""

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


@check("dft_matrix(4) holds the twiddles 1, -j, -1, and j")
def test_dft_matrix(sub):
    dft_matrix = has(sub, "dft_matrix")
    W = np.asarray(dft_matrix(4))
    shape(W, (4, 4), label="dft_matrix(4)")
    want = np.array([[1, 1, 1, 1], [1, -1j, -1, 1j], [1, -1, 1, -1], [1, 1j, -1, -1j]])
    cclose(W, want, atol=1e-12, label="dft_matrix(4)")
    W = np.asarray(dft_matrix(16))
    cclose(W @ W.conj().T, 16 * np.eye(16), atol=1e-9, label="W times its conjugate transpose, which should be 16 I")


@check("dft gives [2, 1 - 3j, 0, 1 + 3j] for [1, 2, 0, -1] and matches np.fft.fft")
def test_dft(sub):
    dft = has(sub, "dft")
    cclose(dft(np.array([1.0, 2.0, 0.0, -1.0])), [2, 1 - 3j, 0, 1 + 3j], atol=1e-12, label="the worked example")
    rng = np.random.default_rng(0)
    for n in (1, 5, 17, 64):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        cclose(dft(x), np.fft.fft(x), atol=1e-9, label=f"a random length-{n} signal")


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
    try:
        fft(np.ones(12))
    except ValueError:
        return
    fail("fft_radix2 of 12 samples should raise ValueError, because 12 is not a power of two")


@check("The spectrum keeps the energy: Parseval's theorem holds for fft_radix2")
def test_parseval(sub):
    fft = has(sub, "fft_radix2")
    x = np.random.default_rng(2).standard_normal(256)
    X = np.asarray(fft(x))
    almost(np.sum(np.abs(X) ** 2) / 256, np.sum(x**2), 1e-8, label="sum |X|^2 / N against sum |x|^2")
    cclose(X[1:128], np.conj(X[:128:-1]), atol=1e-9, label="X[N - k] against conj(X[k]) for a real signal")


@check("rfft_frequencies gives 201 bins from 0 to 8000 Hz for n_fft=400 at 16 kHz")
def test_bins(sub):
    freqs = has(sub, "rfft_frequencies")
    f = np.asarray(freqs(400, 16000))
    shape(f, (201,), label="rfft_frequencies(400, 16000)")
    close(f[:3], [0, 40, 80], label="the first three bins")
    for n, fs in [(400, 16000), (512, 16000), (1023, 22050)]:
        close(freqs(n, fs), np.fft.rfftfreq(n, 1 / fs), label=f"n_fft={n}, fs={fs}")


@check("get_window matches scipy.signal.get_window, periodic and symmetric")
def test_windows(sub):
    get_window = has(sub, "get_window")
    for name, scipy_name in SCIPY_NAMES.items():
        for periodic in (True, False):
            for n in (1, 2, 7, 64, 400):
                got = get_window(name, n, periodic)
                want = scipy.signal.get_window(scipy_name, n, fftbins=periodic)
                close(got, want, atol=1e-12, label=f"{name}, n={n}, periodic={periodic}")


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


@check("Zero-padding refines the grid but can't separate tones closer than the window allows")
def test_zero_padding(sub):
    get_window = has(sub, "get_window")
    freqs = has(sub, "rfft_frequencies")
    rate = 16000
    t = np.arange(rate) / rate
    x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1020 * t)

    def peaks(length, n_fft):
        spec = np.abs(np.fft.rfft(x[:length] * get_window("hann", length), n_fft))
        f = np.asarray(freqs(n_fft, rate))
        band = (f > 900) & (f < 1120)
        s = spec[band] / spec[band].max()
        return int(np.sum((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]) & (s[1:-1] > 0.5)))

    short = peaks(400, 16384)   # 25 ms: the main lobe is 80 Hz wide either side
    long = peaks(3200, 16384)   # 200 ms: 10 Hz either side
    if short != 1:
        fail(f"a 25 ms Hann window zero-padded to 16384 points should show one merged peak near 1 kHz, found {short}")
    if long != 2:
        fail(f"a 200 ms Hann window should separate 1000 Hz and 1020 Hz into 2 peaks, found {long}")
