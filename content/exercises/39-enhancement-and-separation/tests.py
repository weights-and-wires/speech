"""Checks for the enhancement and separation exercise."""

import itertools

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


@check("pit_loss finds the permutation that undoes a shuffle, and matches brute force for three sources")
def test_pit(sub):
    pit, si_sdr = has(sub, "pit_loss"), has(sub, "si_sdr")
    torch.manual_seed(4)
    ref = torch.randn(4, 3, 500, dtype=torch.float64)
    shuffles = [(2, 0, 1), (0, 1, 2), (1, 2, 0), (2, 1, 0)]
    est = torch.stack([ref[b, list(p)] for b, p in enumerate(shuffles)]) + 0.2 * torch.randn(4, 3, 500, dtype=torch.float64)
    loss, perm = pit(est, ref)
    equal(torch.as_tensor(perm).tolist(), [list(p) for p in shuffles], label="perm")
    want = []
    for b in range(4):
        want.append(min(float(-si_sdr(est[b], ref[b, list(p)]).mean()) for p in itertools.permutations(range(3))))
    almost(float(loss), float(np.mean(want)), 1e-6, label="pit loss")
    est.requires_grad_(True)
    loss, _ = pit(est, ref)
    loss.backward()
    if not torch.all(torch.isfinite(est.grad)) or est.grad.abs().sum() == 0:
        fail("pit_loss gives no finite, nonzero gradient with respect to the estimates")


@check("A stack of TCNBlocks keeps the shape and sees exactly tcn_receptive_field frames")
def test_tcn(sub):
    rf = has(sub, "tcn_receptive_field")
    equal(int(rf(3, 8, 3)), 1531, label="tcn_receptive_field(3, 8, 3), the Conv-TasNet paper's setting")
    TCNBlock = has(sub, "TCNBlock")
    torch.manual_seed(5)
    stack = torch.nn.Sequential(*[TCNBlock(8, 16, 3, 2**i) for _ in range(2) for i in range(3)]).double().eval()
    x = torch.randn(1, 8, 101, dtype=torch.float64)
    with torch.no_grad():
        y = stack(x)
        shape(y, (1, 8, 101), label="output of 6 blocks")
        x2 = x.clone()
        x2[0, :, 50] += 1.0
        touched = ((stack(x2) - y).abs().amax(dim=1)[0] > 1e-12).nonzero().flatten()
    width = int(touched.max() - touched.min() + 1) if len(touched) else 0
    if width != int(rf(2, 3, 3)):
        fail(f"a change at frame 50 reached {width} output frames, expected the receptive field {int(rf(2, 3, 3))}")


@check("TinyConvTasNet returns one waveform per source, the input's length, and PIT training improves it")
def test_convtasnet(sub):
    Model, pit = has(sub, "TinyConvTasNet"), has(sub, "pit_loss")
    torch.manual_seed(6)
    model = Model(n_src=2, n_filters=16, length=16, bottleneck=16, hidden=32, kernel=3, blocks=3, repeats=1)
    for n in (800, 803, 1000):
        shape(model(torch.randn(2, n)).detach(), (2, 2, n), label=f"output for a {n}-sample input")
    t = torch.arange(1600) / 8000.0
    low = torch.sin(2 * torch.pi * 300 * t)
    high = torch.sign(torch.sin(2 * torch.pi * 1900 * t)) * 0.5
    refs = torch.stack([low, high])[None].repeat(4, 1, 1) * torch.linspace(0.5, 1.5, 4)[:, None, None]
    mix = refs.sum(1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start, _ = pit(model(mix), refs)
    for _ in range(60):
        loss, _ = pit(model(mix), refs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    gain = start.item() - loss.item()
    if gain < 10.0:
        fail(f"60 PIT steps raised SI-SDR by {gain:.1f} dB on a fixed batch, expected at least 10 dB")
