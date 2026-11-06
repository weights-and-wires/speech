"""Checks for the MFCC exercise.

The MFCC reference numbers were computed offline with librosa 0.11.0:
`librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)`.
"""

import numpy as np
import scipy.fft

from speechcheck import check, close, fail, finite, has, shape

RATE = 16000


def fixture_power():
    b = np.arange(40)[:, None]
    t = np.arange(30)[None, :]
    return 10.0 ** (3.0 * np.sin(0.3 * b + 0.7 * t) - 0.05 * b)


@check("dct2_ortho matches scipy.fft.dct(type=2, norm='ortho') on a vector and along the rows of a matrix")
def test_dct(sub):
    dct2_ortho = has(sub, "dct2_ortho")
    close(dct2_ortho(np.array([4.0, 2.0, 2.0, 0.0])), [4.0, 2.6131259, 0.0, 1.0823922], atol=1e-6, label="the worked example [4, 2, 2, 0]")
    x = np.random.default_rng(1).standard_normal((24, 7))
    got = dct2_ortho(x)
    shape(got, (24, 7), label="the DCT of a (24, 7) matrix")
    close(got, scipy.fft.dct(x, type=2, norm="ortho", axis=0), atol=1e-10, label="DCT-II down each column")


@check("dct2_ortho is orthonormal: it keeps energy, and its transpose undoes it")
def test_orthonormal(sub):
    dct2_ortho = has(sub, "dct2_ortho")
    C = dct2_ortho(np.eye(16))
    close(C.T @ C, np.eye(16), atol=1e-10, label="C^T C for the 16-point DCT matrix")
    x = np.random.default_rng(2).standard_normal(16)
    close(np.sum(dct2_ortho(x) ** 2), np.sum(x**2), rtol=1e-10, label="the energy of a vector and of its DCT")


@check("Keeping the first two DCT coefficients rebuilds a smooth slope: [4, 2, 2, 0] becomes [3.707, 2.707, 1.293, 0.293]")
def test_truncation(sub):
    dct2_ortho = has(sub, "dct2_ortho")
    C = np.asarray(dct2_ortho(np.eye(4)), dtype=float)
    coeffs = np.asarray(dct2_ortho(np.array([4.0, 2.0, 2.0, 0.0])), dtype=float)
    coeffs[2:] = 0.0
    close(C.T @ coeffs, [3.7071068, 2.7071068, 1.2928932, 0.2928932], atol=1e-6, label="the inverse DCT of c0 and c1")


@check("power_to_db takes 10 log10 with a floor of amin and a top_db cut below the global maximum")
def test_power_to_db(sub):
    power_to_db = has(sub, "power_to_db")
    S = np.array([[1.0, 100.0], [1e-3, 0.0]])
    close(power_to_db(S, top_db=None), [[0.0, 20.0], [-30.0, -100.0]], atol=1e-9, label="power_to_db with no top_db")
    close(power_to_db(S, top_db=80.0), [[0.0, 20.0], [-30.0, -60.0]], atol=1e-9, label="power_to_db with top_db=80")
    close(power_to_db(S, top_db=40.0), [[0.0, 20.0], [-20.0, -20.0]], atol=1e-9, label="power_to_db with top_db=40")


@check("mfcc matches librosa.feature.mfcc on a 40-band, 30-frame mel power matrix")
def test_mfcc(sub):
    mfcc = has(sub, "mfcc")
    got = np.asarray(mfcc(fixture_power(), n_mfcc=13), dtype=float)
    shape(got, (13, 30), label="13 MFCCs of 30 frames")
    close(got[[0, 1, 5, 12], [0, 7, 15, 29]], [-57.9414787, 24.9937075, 49.5100268, 1.1552404], atol=1e-3,
          label="entries (0, 0), (1, 7), (5, 15), and (12, 29) against librosa")
    close(got.mean(), -0.786521573535041, atol=1e-3, label="the mean coefficient against librosa")


@check("lifter scales row n by 1 + (L / 2) sin(pi n / L), the HTK convention, and leaves c0 alone")
def test_lifter(sub):
    lifter = has(sub, "lifter")
    c = np.ones((13, 3))
    got = np.asarray(lifter(c, 22), dtype=float)
    shape(got, (13, 3), label="liftered coefficients")
    want = 1.0 + 11.0 * np.sin(np.pi * np.arange(13) / 22.0)
    close(got[:, 0], want, atol=1e-10, label="the lifter weights for L = 22")
    close(got[[0, 1, 11, 12], 1], [1.0, 2.5654632, 12.0, 11.8880359], atol=1e-6, label="the weights on rows 0, 1, 11, and 12")
