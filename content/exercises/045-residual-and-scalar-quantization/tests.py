"""Checks for the residual and scalar quantization exercise."""

import numpy as np
import torch

from speechcheck import check, close, equal, fail, finite_difference, shape


def _codebooks_hand():
    c1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    c2 = np.array([[0.1, 0.1], [-0.1, -0.2], [0.2, -0.2], [-0.2, 0.2]])
    return c1, c2


@check("rvq_encode reproduces the chapter's two-stage example exactly")
def test_rvq_hand(sub):
    c1, c2 = _codebooks_hand()
    codes, recon, norms = sub.rvq_encode(np.array([[0.9, -0.3]]), [c1, c2])
    equal(np.asarray(codes).tolist(), [[0, 1]], label="codes")
    close(recon, [[0.9, -0.2]], label="reconstruction")
    close(norms, [[np.sqrt(0.9), np.sqrt(0.1), 0.1]], label="residual norms [0.949, 0.316, 0.100]")


@check("Each stage quantizes the residual left by the stages before it, not the input")
def test_rvq_residual(sub):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 3))
    c1 = rng.normal(size=(8, 3))
    c2 = rng.normal(size=(8, 3)) * 0.4
    codes, _, _ = sub.rvq_encode(x, [c1, c2])
    first = np.argmin(((x[:, None] - c1[None]) ** 2).sum(-1), axis=1)
    equal(np.asarray(codes)[:, 0], first, label="stage 1 codes")
    residual = x - c1[first]
    second = np.argmin(((residual[:, None] - c2[None]) ** 2).sum(-1), axis=1)
    equal(np.asarray(codes)[:, 1], second, label="stage 2 codes, which must come from x - (stage 1 codeword)")


@check("RVQ residual norms never grow, and column 0 is the input's norm")
def test_rvq_norms(sub):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 4))
    codebooks = []
    for stage in range(4):
        cb = rng.normal(size=(15, 4)) * 0.6**stage
        codebooks.append(np.vstack([np.zeros((1, 4)), cb]))  # a zero code guarantees no growth
    codes, recon, norms = sub.rvq_encode(x, codebooks)
    shape(codes, (200, 4), label="codes")
    shape(recon, (200, 4), label="reconstruction")
    shape(norms, (200, 5), label="residual norms")
    close(norms[:, 0], np.linalg.norm(x, axis=1), label="column 0 of the norms")
    grew = np.diff(norms, axis=1) > 1e-12
    if grew.any():
        row, col = np.argwhere(grew)[0]
        fail(f"vector {row}'s residual norm grew at stage {col + 1}: {norms[row, col]:.4f} -> {norms[row, col + 1]:.4f}")
    close(norms[:, -1], np.linalg.norm(x - recon, axis=1), label="last norm against |x - reconstruction|")


@check("rvq_decode inverts rvq_encode, and decoding a prefix of the stages gives a coarser vector")
def test_rvq_decode(sub):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 4))
    codebooks = [rng.normal(size=(16, 4)) * 0.6**s for s in range(3)]
    codes, recon, _ = sub.rvq_encode(x, codebooks)
    close(sub.rvq_decode(codes, codebooks), recon, label="decode(encode(x)) against the reconstruction")
    partial = codebooks[0][codes[:, 0]] + codebooks[1][codes[:, 1]]
    close(sub.rvq_decode(codes[:, :2], codebooks), partial, label="decoding with the first two stages only")
    one = np.linalg.norm(x - sub.rvq_decode(codes[:, :1], codebooks), axis=1).mean()
    three = np.linalg.norm(x - sub.rvq_decode(codes, codebooks), axis=1).mean()
    if not three < one:
        fail(f"three stages left a mean error of {three:.3f}, no smaller than one stage's {one:.3f}")


@check("fsq_quantize and fsq_index reproduce the chapter's example, index 103")
def test_fsq_hand(sub):
    z = torch.tensor([[0.3, -2.0, 1.2]], dtype=torch.float64)
    z_hat = sub.fsq_quantize(z, [5, 5, 5])
    close(z_hat.detach().numpy(), [[1.0, -2.0, 2.0]], label="quantized values")
    equal(np.asarray(sub.fsq_index(z_hat.detach().numpy(), [5, 5, 5])).tolist(), [103], label="index")


@check("fsq_quantize passes the tanh gradient straight through the rounding")
def test_fsq_grad(sub):
    z = torch.tensor([[0.3, -2.0, 1.2]], dtype=torch.float64, requires_grad=True)
    z_hat = sub.fsq_quantize(z, [5, 5, 5])
    z_hat.sum().backward()
    if z.grad is None:
        fail("no gradient reached z; the rounding must be passed straight through")
    want = finite_difference(lambda v: float((2.0 * np.tanh(v)).sum()), np.array([0.3, -2.0, 1.2]))
    close(z.grad.numpy().ravel(), want, atol=1e-6, label="gradient, which is the derivative of 2 tanh(z)")


@check("fsq_index and fsq_unindex are a bijection over the whole grid, and uniform inputs use every code")
def test_fsq_bijection(sub):
    levels = [7, 5, 3]
    size = 7 * 5 * 3
    grid = sub.fsq_unindex(np.arange(size), levels)
    shape(grid, (size, 3), label="the unindexed grid")
    equal(np.asarray(sub.fsq_index(grid, levels)).tolist(), list(range(size)), label="index(unindex(i))")
    if len({tuple(row) for row in np.asarray(grid).tolist()}) != size:
        fail("fsq_unindex maps two indices to the same grid point")
    rng = np.random.default_rng(5)
    z = torch.tensor(np.arctanh(rng.uniform(-0.999, 0.999, size=(20000, 3))))
    idx = np.asarray(sub.fsq_index(sub.fsq_quantize(z, levels).detach().numpy(), levels))
    used = np.unique(idx).size
    if used != size:
        fail(f"uniform inputs used {used} of the {size} codes; FSQ reaches every grid point")
