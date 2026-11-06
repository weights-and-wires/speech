"""Checks for the source separation exercise."""

import itertools

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape

RATE = 8000


@check("pit_loss keeps the order when the outputs are in order and swaps it when they are not")
def test_pit_two(sub):
    pit, si_sdr = has(sub, "pit_loss"), has(sub, "si_sdr")
    torch.manual_seed(7)
    ref = torch.randn(3, 2, 600, dtype=torch.float64)
    est = ref + 0.3 * torch.randn(3, 2, 600, dtype=torch.float64)
    loss, perm = pit(est, ref)
    equal(torch.as_tensor(perm).tolist(), [[0, 1]] * 3, label="perm for outputs in order")
    in_order = float(-si_sdr(est, ref).mean())
    almost(float(loss), in_order, 1e-6, label="loss for outputs in order")
    loss_swapped, perm_swapped = pit(est.flip(1), ref)
    equal(torch.as_tensor(perm_swapped).tolist(), [[1, 0]] * 3, label="perm for swapped outputs")
    almost(float(loss_swapped), in_order, 1e-6, label="loss for swapped outputs")
    fixed_order = float(-si_sdr(est.flip(1), ref).mean())
    if fixed_order < in_order + 5.0:
        fail(f"a fixed-order loss on swapped outputs is {fixed_order:.1f}, expected far above {in_order:.1f}")


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


@check("tcn_receptive_field matches the formula 1 + R (P - 1) (2^X - 1)")
def test_rf(sub):
    rf = has(sub, "tcn_receptive_field")
    equal(int(rf(3, 8, 3)), 1531, label="tcn_receptive_field(3, 8, 3), the Conv-TasNet paper's setting")
    equal(int(rf(2, 5, 3)), 125, label="tcn_receptive_field(2, 5, 3), the exercise model")
    equal(int(rf(1, 3, 5)), 29, label="tcn_receptive_field(1, 3, 5)")


@check("A stack of TCNBlocks keeps the shape and sees exactly tcn_receptive_field frames")
def test_stack(sub):
    rf = has(sub, "tcn_receptive_field")
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


@check("TinyConvTasNet returns one waveform per source, at the input's length")
def test_shapes(sub):
    Model = has(sub, "TinyConvTasNet")
    torch.manual_seed(6)
    model = Model(n_src=2, n_filters=16, length=16, bottleneck=16, hidden=32, kernel=3, blocks=3, repeats=1)
    for n in (800, 803, 1000):
        shape(model(torch.randn(2, n)).detach(), (2, 2, n), label=f"output for a {n}-sample input")


@check("Sixty PIT steps raise a Conv-TasNet's SI-SDR by at least 10 dB")
def test_training(sub):
    Model, pit = has(sub, "TinyConvTasNet"), has(sub, "pit_loss")
    torch.manual_seed(6)
    model = Model(n_src=2, n_filters=16, length=16, bottleneck=16, hidden=32, kernel=3, blocks=3, repeats=1)
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
