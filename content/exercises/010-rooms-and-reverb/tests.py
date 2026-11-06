"""Checks for the rooms and reverb exercise."""

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape
from speechlab import show_plot

RATE = 16000


def reference_rt60(h, rate, lo=-5.0, hi=-25.0):
    energy = np.cumsum((h**2)[::-1])[::-1]
    decay = 10 * np.log10(energy / energy[0])
    t = np.arange(len(h)) / rate
    keep = (decay <= lo) & (decay >= hi)
    return -60.0 / np.polyfit(t[keep], decay[keep], 1)[0]


@check("schroeder_decay_db gives 0, -6.07, -12.30, -19.29 dB for [1, 0.5, 0.25, 0.125]")
def test_schroeder_by_hand(sub):
    schroeder = has(sub, "schroeder_decay_db")
    got = np.asarray(schroeder(np.array([1.0, 0.5, 0.25, 0.125])))
    shape(got, (4,), label="the decay curve")
    close(got, [0.0, -6.0721, -12.3045, -19.2941], atol=1e-3, label="the worked example")


@check("schroeder_decay_db integrates the remaining energy backward and never rises")
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


@check("estimate_rt60 gives the same answer from a 20 dB span and a 30 dB span of a clean decay")
def test_estimate_spans(sub):
    estimate = has(sub, "estimate_rt60")
    n = int(1.5 * 0.6 * RATE)
    h = np.random.default_rng(6).standard_normal(n) * 10 ** (-3 * np.arange(n) / (0.6 * RATE))
    t20 = float(estimate(h, RATE, -5.0, -25.0))
    t30 = float(estimate(h, RATE, -5.0, -35.0))
    almost(t20, t30, 0.06, label="the T20 and T30 estimates of a 0.6 s room, in seconds")
    almost(t20, reference_rt60(h, RATE), 1e-9, label="the T20 estimate against the reference")


@check("synthetic_rir starts with the direct path and is the same for the same seed")
def test_synthetic_rir(sub):
    synthetic_rir = has(sub, "synthetic_rir")
    h = np.asarray(synthetic_rir(0.5, RATE, seconds=0.75, seed=3))
    shape(h, (12000,), label="0.75 s at 16 kHz")
    almost(h[0], 1.0, 1e-12, label="the direct-path sample")
    equal(np.asarray(synthetic_rir(0.5, RATE, seconds=0.75, seed=3)), h, label="a second call with the same seed")
    if np.array_equal(np.asarray(synthetic_rir(0.5, RATE, seconds=0.75, seed=4)), h):
        fail("seed=4 gave the same response as seed=3; the seed should set the noise")
    shape(np.asarray(synthetic_rir(0.4, RATE)), (9600,), label="the default length, 1.5 * rt60 seconds")


@check("synthetic_rir decays with the requested RT60")
def test_synthetic_rir_decay(sub):
    synthetic_rir = has(sub, "synthetic_rir")
    for rt60 in (0.25, 0.5, 1.0):
        h = np.asarray(synthetic_rir(rt60, RATE, seed=0))
        measured = reference_rt60(h, RATE)
        if abs(measured - rt60) > 0.1 * rt60:
            energy = np.cumsum((h**2)[::-1])[::-1]
            show_plot((np.arange(len(h)) / RATE, 10 * np.log10(energy / energy[0])), label=f"Schroeder decay of your RIR, RT60 {rt60} s", xlabel="Time (s)", ylabel="dB")
            fail(f"an RIR asked for RT60 {rt60} s measures {measured:.3f} s by Schroeder integration, expected within 10%")


@check("A longer RT60 leaves more energy in the tail after a click")
def test_tail_energy(sub):
    synthetic_rir = has(sub, "synthetic_rir")
    convolve_fft = has(sub, "convolve_fft")
    click = np.zeros(200)
    click[0] = 1.0
    tails = []
    for rt60 in (0.3, 1.2):
        wet = np.asarray(convolve_fft(click, np.asarray(synthetic_rir(rt60, RATE))))
        tails.append(float(np.sum(wet[1:] ** 2)))
    if not tails[1] > 3 * tails[0]:
        fail(f"the tail energy after a click should grow with RT60 (about 4 times from 0.3 s to 1.2 s), got {tails[0]:.3f} and {tails[1]:.3f}")
