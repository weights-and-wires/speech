"""Checks for the efficient speech encoders exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, finite, has, shape

A = np.array([0.5, 0.25])
B = np.array([1.0, 2.0])
C = np.array([1.0, 1.0])


def random_layer(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 0.95, n), rng.normal(size=n), rng.normal(size=n)


@check("encoder_flops counts the linear and the quadratic term at T = frames // subsample")
def test_flops(sub):
    flops = has(sub, "encoder_flops")
    equal(int(flops(1000, 256, 1, 4)), 457_216_000, label="1 layer, 1000 frames, 4x")
    equal(int(flops(1000, 256, 1, 8)), 212_608_000, label="1 layer, 1000 frames, 8x")
    equal(int(flops(1000, 256, 16, 8)), 16 * 212_608_000, label="16 layers")
    equal(int(flops(1003, 256, 1, 4)), 457_216_000, label="1003 frames floors to 250")
    ratio = flops(1000, 256, 1, 4) / flops(1000, 256, 1, 8)
    almost(ratio, 2.151, 0.001, label="the 4x to 8x saving, between 2 (all linear) and 4 (all quadratic)")


@check("subsample_len and subsample_params match the hand counts")
def test_formulas(sub):
    length, params = has(sub, "subsample_len"), has(sub, "subsample_params")
    equal([int(length(n, 3)) for n in (80, 83, 1)], [10, 11, 1], label="length after 3 stages for n=80, 83, 1")
    equal(int(length(83, 2)), 21, label="length after 2 stages")
    equal(int(params(256, 3, False)), 1_182_720, label="dense parameters, C=256, 3 stages")
    equal(int(params(256, 3, True)), 139_264, label="separable parameters, C=256, 3 stages")
    equal(int(params(16, 1, True)), 160, label="one stage has no separable part")


@check("build_subsampler gives the right shape and exactly the counted parameters")
def test_subsampler(sub):
    build, params = has(sub, "build_subsampler"), has(sub, "subsample_params")
    for separable, want in ((False, 4800), (True, 1024)):
        net = build(16, 3, separable)
        got = sum(p.numel() for p in net.parameters())
        equal(got, want, label=f"parameters of the module, separable={separable}")
        equal(int(params(16, 3, separable)), want, label=f"subsample_params, separable={separable}")
        out = net(torch.zeros(1, 1, 83, 80))
        equal(tuple(out.shape), (1, 16, 11, 10), label=f"output shape, separable={separable}")


@check("downsample averages adjacent frames with softmax weights and pads with the last frame")
def test_downsample(sub):
    down = has(sub, "downsample")
    x = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])
    close(down(x, np.array([0.0, math.log(3.0)]), 2), np.array([[2.5], [6.5], [9.0]]), label="weights 1/4 and 3/4, 5 frames")
    y = down(np.arange(21, dtype=float).reshape(7, 3), np.zeros(2), 2)
    shape(y, (4, 3), label="7 frames, factor 2")
    close(y[0], np.array([1.5, 2.5, 3.5]), label="the plain mean of frames 0 and 1")
    shape(down(np.ones((9, 2)), np.zeros(4), 4), (3, 2), label="9 frames, factor 4")


@check("upsample repeats frames, and a down then up round trip restores a smooth signal")
def test_upsample(sub):
    down, up = has(sub, "downsample"), has(sub, "upsample")
    z = np.array([[1.0], [2.0], [3.0]])
    close(up(z, 2, 5), np.array([[1.0], [1.0], [2.0], [2.0], [3.0]]), label="3 frames, factor 2, length 5")
    const = np.full((7, 3), 4.0)
    close(up(down(const, np.array([0.3, -0.1]), 2), 2, 7), const, label="constant signal round trip")
    paired = np.repeat(np.arange(4.0)[:, None], 2, axis=0)
    close(up(down(paired, np.array([0.7, 0.2]), 2), 2, 8), paired, label="frames that already come in equal pairs")


@check("bypass mixes (1 - c) * x + c * y per channel")
def test_bypass(sub):
    byp, down, up = has(sub, "bypass"), has(sub, "downsample"), has(sub, "upsample")
    x, y = np.array([[4.0, 8.0]]), np.array([[0.0, 0.0]])
    # A wrong implementation swaps the weights, c * x + (1 - c) * y.
    close(byp(x, y, np.array([0.25, 0.75])), np.array([[3.0, 2.0]]), label="c = [0.25, 0.75]")
    close(byp(x, y, np.zeros(2)), x, label="c = 0 keeps the input")
    close(byp(x, y, np.ones(2)), y, label="c = 1 keeps the module output")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(9, 3))
    out = up(down(z, np.zeros(2), 2), 2, 9)
    close(byp(z, out, np.zeros(3)), z, label="c = 0 around a down and up stack is the identity")


@check("ssm_recurrent matches the hand values and continues from a given state")
def test_recurrent(sub):
    rec = has(sub, "ssm_recurrent")
    y, h = rec(A, B, C, np.array([1.0, 0.0, 0.0, 2.0]))
    close(y, np.array([3.0, 1.0, 0.375, 6.15625]), rtol=1e-12, atol=1e-12, label="outputs")
    close(h, np.array([2.125, 4.03125]), rtol=1e-12, atol=1e-12, label="final state")
    y2, h2 = rec(A, B, C, np.array([0.0]), h0=np.array([1.0, 1.0]))
    close(y2, np.array([0.75]), rtol=1e-12, atol=1e-12, label="one step from state [1, 1]")
    close(h2, np.array([0.5, 0.25]), rtol=1e-12, atol=1e-12, label="state after that step")


@check("ssm_kernel and ssm_convolve reproduce the recurrence to 1e-10")
def test_convolutional(sub):
    kern, conv, rec = has(sub, "ssm_kernel"), has(sub, "ssm_convolve"), has(sub, "ssm_recurrent")
    # A wrong implementation weights the input with a**(k - 1), which shifts the kernel by one step.
    close(kern(A, B, C, 4), np.array([3.0, 1.0, 0.375, 0.15625]), rtol=1e-12, atol=1e-12, label="K[0..3]")
    close(conv(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0])), np.array([1.0, 3.0, 6.0, 6.0]), label="causal convolution")
    a, b, c = random_layer()
    x = np.random.default_rng(1).normal(size=64)
    want, _ = rec(a, b, c, x)
    got = conv(kern(a, b, c, len(x)), x)
    shape(got, (64,), label="convolutional output")
    finite(got, label="convolutional output")
    close(got, want, rtol=1e-10, atol=1e-10, label="convolutional form vs recurrent form")


@check("ssm_chunk with the state carried between chunks equals the full pass")
def test_chunks(sub):
    chunk, rec = has(sub, "ssm_chunk"), has(sub, "ssm_recurrent")
    a, b, c = random_layer()
    x = np.random.default_rng(2).normal(size=64)
    want, want_h = rec(a, b, c, x)
    # A wrong implementation starts every chunk from h = 0, which loses the past at each boundary.
    for cuts in ([0, 7, 20, 64], [0, 32, 64], list(range(65))):
        h, pieces = np.zeros(4), []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            y, h = chunk(a, b, c, x[lo:hi], h)
            pieces.append(np.asarray(y))
        close(np.concatenate(pieces), want, rtol=1e-10, atol=1e-10, label=f"{len(cuts) - 1} chunks vs the full pass")
        close(h, want_h, rtol=1e-10, atol=1e-10, label=f"state after {len(cuts) - 1} chunks")
