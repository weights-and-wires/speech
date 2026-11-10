"""Checks for the speech enhancement exercise."""

import numpy as np
import torch
from scipy.signal import istft, lfilter, stft

from speechcheck import almost, check, close, equal, fail, has, shape

RATE = 16000


def ref_si_sdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    target = (est @ ref) / (ref @ ref) * ref
    return 10 * np.log10(np.sum(target**2) / np.sum((est - target) ** 2))


def vowels(rng, seconds=1.5):
    """Speech-like test signal: formant-filtered pulse trains, with pauses, after 0.25 s of silence."""
    out = [np.zeros(RATE // 4)]
    formants = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240)]
    while sum(len(o) for o in out) < seconds * RATE:
        n = int(0.2 * RATE)
        src = np.zeros(n)
        src[:: RATE // int(rng.uniform(100, 140))] = 1.0
        y = src
        for f, bw in zip(formants[rng.integers(3)], (80, 100, 140)):
            r = np.exp(-np.pi * bw / RATE)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
        out += [y * np.hanning(n) ** 0.3, np.zeros(RATE // 16)]
    x = np.concatenate(out)[: int(seconds * RATE)]
    return 0.5 * x / np.abs(x).max()


def spec(x):
    return stft(x, nperseg=512, noverlap=384, window="hann")[2]


def unspec(X, n):
    return istft(X, nperseg=512, noverlap=384, window="hann")[1][:n]


@check("si_sdr matches the projection formula and ignores the estimate's scale")
def test_si_sdr(sub):
    si_sdr = has(sub, "si_sdr")
    rng = np.random.default_rng(0)
    ref = rng.normal(size=800)
    est = ref + 0.3 * rng.normal(size=800) + 0.5
    got = float(si_sdr(torch.tensor(est), torch.tensor(ref)))
    almost(got, ref_si_sdr(est, ref), 1e-6, label="si_sdr(est, ref)")
    almost(float(si_sdr(torch.tensor(3.7 * est), torch.tensor(ref))), got, 1e-6, label="si_sdr(3.7 * est, ref)")
    # An error orthogonal to the reference with a tenth of its energy gives exactly 10 dB.
    ref0 = ref - ref.mean()
    e = rng.normal(size=800)
    e -= e.mean()
    e -= (e @ ref0) / (ref0 @ ref0) * ref0
    e *= np.sqrt(0.1 * (ref0 @ ref0) / (e @ e))
    almost(float(si_sdr(torch.tensor(ref0 + e), torch.tensor(ref0))), 10.0, 1e-6, label="10 dB orthogonal error")
    batch = si_sdr(torch.tensor(np.stack([est, ref])), torch.tensor(np.stack([ref, ref])))
    shape(batch, (2,), label="si_sdr over a batch of two")


@check("mix_at_snr scales the noise to the requested SNR")
def test_mix(sub):
    mix = has(sub, "mix_at_snr")
    rng = np.random.default_rng(1)
    clean, noise = rng.normal(size=4000), 3.0 * rng.normal(size=4000)
    for snr in (-5.0, 0.0, 12.0):
        y, n = mix(clean, noise, snr)
        almost(10 * np.log10(np.mean(clean**2) / np.mean(n**2)), snr, 1e-9, label=f"measured SNR at {snr} dB")
        close(y, clean + n, label="mixture")


@check("spectral_subtraction applies the floor and keeps the noisy phase")
def test_subtraction(sub):
    subtract = has(sub, "spectral_subtraction")
    Y = np.array([[3 + 4j, 1 + 0j], [0 + 2j, -1 - 1j]])  # powers 25, 1, 4, 2
    noise = np.array([4.0, 1.5])
    got = np.asarray(subtract(Y, noise, alpha=2.0, beta=0.1))
    # Bin 0: max(25 - 8, 2.5) = 17 and max(1 - 8, 0.1) = 0.1. Bin 1: max(4 - 3, 0.4) = 1 and max(2 - 3, 0.2) = 0.2.
    close(np.abs(got) ** 2, np.array([[17.0, 0.1], [1.0, 0.2]]), rtol=1e-9, label="estimated clean power")
    close(np.angle(got), np.angle(Y), atol=1e-9, label="phase")


@check("wiener_gain and decision_directed_gains follow the Ephraim-Malah recursion")
def test_wiener(sub):
    gain = has(sub, "wiener_gain")
    close(np.asarray(gain(np.array([0.0, 1.0, 9.0]))), np.array([0.0, 0.5, 0.9]), label="wiener_gain([0, 1, 9])")
    dd = has(sub, "decision_directed_gains")
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(5, 12)) + 1j * rng.normal(size=(5, 12))
    Y[:, 6:] *= 4.0
    pn = np.full(5, 2.0)
    got = np.asarray(dd(Y, pn, a=0.9, xi_min=0.01))
    shape(got, (5, 12), label="gains")
    want, prev = np.empty((5, 12)), np.zeros(5)
    for t in range(12):
        p = np.abs(Y[:, t]) ** 2
        xi = np.maximum(0.9 * prev / pn + 0.1 * np.maximum(p / pn - 1, 0), 0.01)
        want[:, t] = xi / (1 + xi)
        prev = want[:, t] ** 2 * p
    close(got, want, rtol=1e-9, label="gains")


@check("ideal_masks gives the IBM and IRM from powers, with the IRM at beta = 1 equal to the Wiener gain")
def test_masks(sub):
    masks = has(sub, "ideal_masks")
    S = np.array([[2.0 + 0j, 1j], [0.1, 3.0]])
    N = np.array([[1.0 + 0j, 2.0], [1.0, 1j]])
    ibm, irm = masks(S, N, lc_db=0.0, beta=0.5)
    equal(np.asarray(ibm).astype(int).tolist(), [[1, 0], [0, 1]], label="IBM")
    close(np.asarray(irm), np.sqrt(np.array([[4 / 5, 1 / 5], [0.01 / 1.01, 9 / 10]])), rtol=1e-9, label="IRM, beta = 0.5")
    _, irm1 = masks(S, N, lc_db=0.0, beta=1.0)
    xi = np.abs(S) ** 2 / np.abs(N) ** 2
    close(np.asarray(irm1), xi / (1 + xi), rtol=1e-9, label="IRM, beta = 1")


@check("On noisy vowels, the oracle IRM gains over 8 dB of SI-SDR and the Wiener filter over 5 dB")
def test_enhancement(sub):
    rng = np.random.default_rng(3)
    clean = vowels(rng)
    noise = rng.normal(size=clean.size)
    noisy, noise = sub.mix_at_snr(clean, noise, 0.0)
    Y, S, N = spec(noisy), spec(clean), spec(noise)
    pn = np.mean(np.abs(Y[:, :30]) ** 2, axis=1)
    base = ref_si_sdr(noisy, clean)
    _, irm = sub.ideal_masks(S, N)
    oracle = ref_si_sdr(unspec(np.asarray(irm) * Y, clean.size), clean) - base
    wiener = ref_si_sdr(unspec(np.asarray(sub.decision_directed_gains(Y, pn)) * Y, clean.size), clean) - base
    if oracle < 8.0:
        fail(f"the oracle IRM improved SI-SDR by {oracle:.1f} dB, expected more than 8 dB")
    if wiener < 5.0:
        fail(f"the decision-directed Wiener filter improved SI-SDR by {wiener:.1f} dB, expected more than 5 dB")
