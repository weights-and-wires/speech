"""Checks for the hearing and mel scale exercise.

The CPU image has no librosa, so the references here are built two ways: an
independent construction from the formulas (with np.interp for the triangles),
and numbers computed offline with librosa 0.11.0 `librosa.filters.mel`,
hard-coded below.
"""

import numpy as np

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000


def ref_slaney_mel(f):
    f = np.asarray(f, dtype=float)
    out = np.empty_like(f)
    low = f < 1000.0
    out[low] = 3.0 * f[low] / 200.0
    out[~low] = 15.0 + 27.0 * np.log(f[~low] / 1000.0) / np.log(6.4)
    return out


def ref_slaney_hz(m):
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m)
    low = m < 15.0
    out[low] = 200.0 * m[low] / 3.0
    out[~low] = 1000.0 * np.exp((m[~low] - 15.0) * np.log(6.4) / 27.0)
    return out


def ref_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None, htk=False, norm="slaney"):
    """One filter at a time, each triangle drawn with np.interp."""
    fmax = sr / 2.0 if fmax is None else fmax
    if htk:
        to_mel = lambda f: 1127.0 * np.log1p(np.asarray(f, dtype=float) / 700.0)
        to_hz = lambda m: 700.0 * np.expm1(np.asarray(m, dtype=float) / 1127.0)
    else:
        to_mel, to_hz = ref_slaney_mel, ref_slaney_hz
    edges = to_hz(np.linspace(to_mel(np.array([fmin]))[0], to_mel(np.array([fmax]))[0], n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.interp(freqs, [lo, mid, hi], [0.0, 1.0, 0.0], left=0.0, right=0.0)
        if norm == "slaney":
            bank[i] *= 2.0 / (hi - lo)
    return bank


@check("hz_to_mel matches the worked table on both scales")
def test_hz_to_mel(sub):
    hz_to_mel = has(sub, "hz_to_mel")
    f = np.array([100.0, 440.0, 1000.0, 2000.0, 4000.0, 8000.0])
    close(hz_to_mel(f), [1.50, 6.60, 15.00, 25.08, 35.16, 45.25], atol=1e-2, label="Slaney mel of 100 Hz to 8 kHz")
    close(hz_to_mel(f, htk=True), [150.49, 549.64, 999.99, 1521.36, 2146.06, 2840.02], atol=1e-2, label="HTK mel of 100 Hz to 8 kHz")
    close(hz_to_mel(np.array([1000.0])), [15.0], atol=1e-12, label="Slaney mel of exactly 1 kHz")
    grid = np.linspace(0.0, 11025.0, 301)
    close(hz_to_mel(grid), ref_slaney_mel(grid), rtol=1e-10, atol=1e-10, label="Slaney mel on a grid from 0 Hz to 11,025 Hz")


@check("mel_to_hz inverts hz_to_mel on both scales")
def test_inverse(sub):
    hz_to_mel, mel_to_hz = has(sub, "hz_to_mel"), has(sub, "mel_to_hz")
    grid = np.concatenate([np.linspace(0.0, 11025.0, 400), [999.999, 1000.0, 1000.001]])
    for htk in (False, True):
        back = mel_to_hz(hz_to_mel(grid, htk=htk), htk=htk)
        close(back, grid, rtol=1e-9, atol=1e-9, label=f"mel_to_hz(hz_to_mel(f)) with htk={htk}")
    close(mel_to_hz(np.array([15.0, 45.0])), [1000.0, 1000.0 * 6.4 ** (30.0 / 27.0)], rtol=1e-10, label="Slaney Hz of 15 and 45 mel")


@check("mel_filterbank reproduces the two-filter hand example at 8 kHz with n_fft=8")
def test_hand_bank(sub):
    mel_filterbank = has(sub, "mel_filterbank")
    peak_one = np.asarray(mel_filterbank(8000, 8, 2, norm=None), dtype=float)
    shape(peak_one, (2, 5), label="a two-filter bank over five FFT bins")
    close(peak_one, [[0.0, 0.78259152, 0.0, 0.0, 0.0], [0.0, 0.21740848, 0.90367695, 0.45183848, 0.0]],
          atol=1e-7, label="the peak-1 weights")
    slaney = np.asarray(mel_filterbank(8000, 8, 2), dtype=float)
    close(slaney[0, 1], 0.78259152 * 2.0 / 1786.819732, rtol=1e-6, label="filter 0 at 1000 Hz, with Slaney normalization")
    close(slaney[1, 2], 0.90367695 * 2.0 / 3218.580268, rtol=1e-6, label="filter 1 at 2000 Hz, with Slaney normalization")


@check("mel_filterbank reproduces librosa's 80-filter Whisper bank at 16 kHz and n_fft=400")
def test_librosa_values(sub):
    fb = np.asarray(has(sub, "mel_filterbank")(RATE, 400, 80), dtype=float)
    shape(fb, (80, 201), label="the filterbank")
    # librosa.filters.mel(sr=16000, n_fft=400, n_mels=80), computed offline.
    want = {
        0: ([1], [0.024862595]),
        1: ([1, 2], [0.001990822, 0.022871772]),
        10: ([10, 11], [0.019908218, 0.004954375]),
        40: ([42, 43, 44], [0.005411105, 0.014735566, 0.006518190]),
        79: (list(range(186, 200)), [0.000366742, 0.000833070, 0.001299398, 0.001765727, 0.002232055,
                                     0.002698383, 0.003164712, 0.003141313, 0.002692554, 0.002243795,
                                     0.001795036, 0.001346277, 0.000897518, 0.000448759]),
    }
    for row, (bins, values) in want.items():
        nonzero = np.nonzero(fb[row] > 1e-12)[0]
        equal(nonzero, np.array(bins), label=f"the FFT bins filter {row} touches")
        close(fb[row, bins], values, rtol=0, atol=1e-8, label=f"filter {row}'s weights")
    close(fb.sum(), 1.999024092387117, rtol=1e-6, label="the sum of every weight")


@check("mel_filterbank matches an independent triangle construction for both scales, both norms, and a band limit")
def test_filterbank_general(sub):
    mel_filterbank = has(sub, "mel_filterbank")
    cases = [
        dict(sample_rate=16000, n_fft=400, n_mels=128),
        dict(sample_rate=16000, n_fft=400, n_mels=40, htk=True, norm=None),
        dict(sample_rate=16000, n_fft=512, n_mels=64, fmin=20.0, fmax=7600.0),
        dict(sample_rate=22050, n_fft=2048, n_mels=128, htk=True, norm="slaney"),
    ]
    for case in cases:
        got = np.asarray(mel_filterbank(**case), dtype=float)
        want = ref_filterbank(case["sample_rate"], case["n_fft"], case["n_mels"], case.get("fmin", 0.0),
                              case.get("fmax"), case.get("htk", False), case.get("norm", "slaney"))
        close(got, want, rtol=1e-6, atol=1e-10, label=f"mel_filterbank({case})")
    # librosa.filters.mel(sr=16000, n_fft=400, n_mels=40, htk=True, norm=None), row 20, computed offline.
    got = np.asarray(mel_filterbank(16000, 400, 40, htk=True, norm=None), dtype=float)[20, 43:51]
    close(got, [0.177276939, 0.440950543, 0.704624116, 0.968297720, 0.781857073, 0.533901751, 0.285946399, 0.037991077],
          atol=1e-7, label="HTK, peak-1 filter 20 against librosa")


@check("Peak-1 triangles overlap so that they add up to one between the first and last centers")
def test_partition(sub):
    mel_filterbank = has(sub, "mel_filterbank")
    fb = np.asarray(mel_filterbank(16000, 1024, 40, htk=True, norm=None), dtype=float)
    if abs(fb.max() - 1.0) > 0.05:
        fail(f"with norm=None the tallest weight is {fb.max():.4f}; each triangle peaks at 1, so expected close to 1")
    edges = 700.0 * np.expm1(np.linspace(0.0, np.log1p(8000.0 / 700.0), 42))
    freqs = np.arange(513) * 16000 / 1024
    inside = (freqs >= edges[1]) & (freqs <= edges[-2])
    close(fb[:, inside].sum(axis=0), np.ones(inside.sum()), atol=1e-9, label="the column sums between the first and last centers")
