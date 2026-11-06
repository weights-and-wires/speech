"""Checks for the frequency and DFT exercise."""

import numpy as np

from speechcheck import almost, check, close, fail, has, shape

RATE = 16000


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


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


@check("A detector that uses only a cosine misses a sine, but dft sees both")
def test_phase(sub):
    dft = has(sub, "dft")
    n = np.arange(8)
    sine = np.sin(2 * np.pi * n / 8)
    cosine = np.cos(2 * np.pi * n / 8)
    for name, x in [("a sine", sine), ("a cosine", cosine)]:
        X = np.asarray(dft(x))
        almost(abs(X[1]), 4.0, 1e-9, label=f"|X[1]| of {name} at bin 1 of 8, which is A N / 2 = 4")
    X = np.asarray(dft(sine))
    almost(float(np.sum(sine * cosine)), 0.0, 1e-9, label="the cosine-only detector on the sine")
    almost(float(np.angle(X[1])), -np.pi / 2, 1e-9, label="the phase of a sine, minus a quarter turn")


@check("The spectrum keeps the energy and mirrors a real signal: Parseval and conjugate symmetry")
def test_parseval(sub):
    dft = has(sub, "dft")
    x = np.random.default_rng(2).standard_normal(64)
    X = np.asarray(dft(x))
    almost(np.sum(np.abs(X) ** 2) / 64, np.sum(x**2), 1e-8, label="sum |X|^2 / N against sum |x|^2")
    cclose(X[1:32], np.conj(X[:32:-1]), atol=1e-9, label="X[N - k] against conj(X[k]) for a real signal")


@check("rdft keeps bins 0 to N/2 and matches np.fft.rfft for even and odd lengths")
def test_rdft(sub):
    rdft = has(sub, "rdft")
    rng = np.random.default_rng(3)
    for n in (8, 9, 64, 400):
        x = rng.standard_normal(n)
        got = np.asarray(rdft(x))
        shape(got, (n // 2 + 1,), label=f"rdft of {n} samples")
        cclose(got, np.fft.rfft(x), atol=1e-8, label=f"length {n}")


@check("rfft_frequencies gives 201 bins from 0 to 8000 Hz for n_fft=400 at 16 kHz")
def test_bins(sub):
    freqs = has(sub, "rfft_frequencies")
    f = np.asarray(freqs(400, 16000))
    shape(f, (201,), label="rfft_frequencies(400, 16000)")
    close(f[:3], [0, 40, 80], label="the first three bins")
    for n, fs in [(400, 16000), (512, 16000), (1023, 22050)]:
        close(freqs(n, fs), np.fft.rfftfreq(n, 1 / fs), label=f"n_fft={n}, fs={fs}")


@check("The running example: 440 Hz at amplitude 0.5 peaks in bin 11 with magnitude A N / 2 = 100")
def test_running_example(sub):
    rdft = has(sub, "rdft")
    freqs = has(sub, "rfft_frequencies")
    n = 400
    t = np.arange(n) / RATE
    frame = 0.5 * np.cos(2 * np.pi * 440 * t)
    mag = np.abs(np.asarray(rdft(frame)))
    k = int(np.argmax(mag))
    f = np.asarray(freqs(n, RATE))
    almost(float(f[k]), 440.0, 1e-9, label="the frequency of the strongest bin")
    almost(float(mag[k]), 100.0, 1e-6, label="its magnitude")
    if np.sum(mag > 1e-6) != 1:
        fail(f"a tone that fits a whole number of cycles should fill one bin, but {int(np.sum(mag > 1e-6))} bins are above 1e-6")
    mix = frame + 0.25 * np.cos(2 * np.pi * 1000 * t + 0.7)
    spec = np.asarray(rdft(mix))
    top = np.argsort(np.abs(spec))[::-1][:2]
    got = sorted(float(f[i]) for i in top)
    if got != [440.0, 1000.0]:
        fail(f"the two strongest bins of the 440 Hz plus 1000 Hz mix are at {got} Hz, expected [440.0, 1000.0]")
    almost(float(np.abs(spec[25])), 50.0, 1e-6, label="the magnitude of the 1000 Hz tone, A N / 2 = 50")
    almost(float(np.angle(spec[25])), 0.7, 1e-6, label="the phase of the 1000 Hz tone in radians")
