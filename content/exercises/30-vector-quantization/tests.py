"""Checks for the vector quantization exercise."""

import numpy as np
import torch

from speechcheck import check, close, equal, fail, finite_difference, shape


def _codebooks_hand():
    c1 = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    c2 = np.array([[0.1, 0.1], [-0.1, -0.2], [0.2, -0.2], [-0.2, 0.2]])
    return c1, c2


@check("nearest_codeword picks the closest codeword, matching a brute-force search")
def test_nearest(sub):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 6))
    codebook = rng.normal(size=(40, 6))
    idx, q = sub.nearest_codeword(z, codebook)
    shape(idx, (300,), label="indices")
    shape(q, (300, 6), label="quantized vectors")
    brute = np.argmin(((z[:, None, :] - codebook[None, :, :]) ** 2).sum(-1), axis=1)
    equal(np.asarray(idx), brute, label="indices against brute force")
    close(q, codebook[brute], label="quantized vectors")
    c1, _ = _codebooks_hand()
    idx, q = sub.nearest_codeword(np.array([[0.9, -0.3]]), c1)
    equal(int(idx[0]), 0, label="the lesson's example, squared distances [0.10, 2.50, 3.70, 1.30]")


@check("straight_through has the quantized value and passes the gradient to z_e unchanged")
def test_straight_through(sub):
    torch.manual_seed(0)
    z_e = torch.randn(5, 3, requires_grad=True)
    z_q = torch.randn(5, 3)
    out = sub.straight_through(z_e, z_q)
    close(out.detach().numpy(), z_q.numpy(), label="forward value")
    weights = torch.randn(5, 3)
    (out * weights).sum().backward()
    if z_e.grad is None:
        fail("no gradient reached z_e; the output must depend on z_e outside a detach")
    close(z_e.grad.numpy(), weights.numpy(), label="gradient with respect to z_e (should be the identity)")


@check("vq_losses sends the codebook loss only to the codebook and the commitment loss only to the encoder")
def test_vq_losses(sub):
    torch.manual_seed(1)
    z_np = np.random.default_rng(1).normal(size=(12, 4))
    cb_np = np.random.default_rng(2).normal(size=(6, 4))

    z_e = torch.tensor(z_np, requires_grad=True)
    codebook = torch.tensor(cb_np, requires_grad=True)
    z_q, idx, cb_loss, commit = sub.vq_losses(z_e, codebook)
    brute = np.argmin(((z_np[:, None] - cb_np[None]) ** 2).sum(-1), axis=1)
    equal(idx.numpy(), brute, label="indices")
    want = ((z_np - cb_np[brute]) ** 2).sum(1).mean()
    close(float(cb_loss), want, label="codebook loss value (mean over vectors of the squared distance)")
    close(float(commit), want, label="commitment loss value")

    cb_loss.backward(retain_graph=True)
    if z_e.grad is not None and float(z_e.grad.abs().max()) > 0:
        fail("the codebook loss sent a gradient to z_e; stop the gradient on z_e inside it")
    if codebook.grad is None:
        fail("the codebook loss sent no gradient to the codebook")
    codebook.grad = None
    commit.backward()
    if codebook.grad is not None and float(codebook.grad.abs().max()) > 0:
        fail("the commitment loss sent a gradient to the codebook; stop the gradient on the codeword")

    def commitment_value(z_flat):
        z = z_flat.reshape(12, 4)
        return float(((z - cb_np[brute]) ** 2).sum(1).mean())

    fd = finite_difference(commitment_value, z_np.reshape(-1)).reshape(12, 4)
    close(z_e.grad.numpy(), fd, atol=1e-5, label="commitment gradient against finite differences")
    close(z_q.detach().numpy(), cb_np[brute], label="z_q's value")


@check("ema_update reproduces the lesson's hand-worked step")
def test_ema(sub):
    codebook = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([[0.2, 0.0], [0.0, 0.2], [0.9, 0.1], [1.3, -0.1]])
    idx = np.array([0, 0, 1, 1])
    counts, sums, new = sub.ema_update(np.ones(3), codebook.copy(), z, idx, decay=0.9, eps=0.0)
    close(counts, [1.1, 1.1, 0.9], label="EMA counts")
    close(sums, [[0.02, 0.02], [1.12, 0.0], [0.0, 0.9]], label="EMA sums")
    close(new, [[0.02 / 1.1, 0.02 / 1.1], [1.12 / 1.1, 0.0], [0.0, 1.0]], label="new codebook")
    _, _, smoothed = sub.ema_update(np.array([1.0, 0.0]), np.array([[1.0], [0.0]]), np.array([[2.0]]), np.array([0]), decay=0.5, eps=1e-5)
    if not np.all(np.isfinite(smoothed)):
        fail(f"a code with zero count gave {smoothed.ravel()}; the Laplace smoothing keeps the division finite")


@check("restart_dead_codes replaces only the dead codes, with vectors from the batch")
def test_restart(sub):
    rng = np.random.default_rng(3)
    codebook = rng.normal(size=(8, 3))
    counts = np.array([5.0, 0.1, 3.0, 0.0, 2.5, 1.9, 4.0, 2.0])
    sums = codebook * counts[:, None]
    z = rng.normal(size=(20, 3)) + 10.0
    new_cb, new_counts, new_sums, n = sub.restart_dead_codes(codebook, counts, sums, z, 2.0, np.random.default_rng(0))
    equal(int(n), 3, label="number of codes restarted (counts 0.1, 0.0, and 1.9 are below 2)")
    live = counts >= 2.0
    close(new_cb[live], codebook[live], label="live codes")
    for k in np.flatnonzero(~live):
        if not np.any(np.all(np.isclose(z, new_cb[k]), axis=1)):
            fail(f"restarted code {k} is {new_cb[k]}, which is not a row of the batch")
    close(new_sums / new_counts[:, None], new_cb, label="sums / counts after the restart (must equal the codebook)")


@check("codebook_perplexity is K for uniform use and 1 for a collapsed codebook")
def test_perplexity(sub):
    close(sub.codebook_perplexity(np.arange(64) % 16, 16), 16.0, label="uniform use of 16 codes")
    close(sub.codebook_perplexity(np.zeros(50, dtype=int), 16), 1.0, label="one code only")
    close(sub.codebook_perplexity(np.array([0, 0, 1, 1]), 8), 2.0, label="two codes, equally often")


@check("rvq_encode reproduces the lesson's two-stage example exactly")
def test_rvq_hand(sub):
    c1, c2 = _codebooks_hand()
    codes, recon, norms = sub.rvq_encode(np.array([[0.9, -0.3]]), [c1, c2])
    equal(np.asarray(codes).tolist(), [[0, 1]], label="codes")
    close(recon, [[0.9, -0.2]], label="reconstruction")
    close(norms, [[np.sqrt(0.9), np.sqrt(0.1), 0.1]], label="residual norms [0.949, 0.316, 0.100]")
    commitment = (np.asarray(norms)[:, 1:] ** 2).sum(1).mean()
    close(commitment, 0.11, label="commitment loss from the residual norms")


@check("RVQ residual norms never grow, and rvq_decode inverts rvq_encode for any prefix of stages")
def test_rvq_general(sub):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 4))
    codebooks = []
    for stage in range(4):
        cb = rng.normal(size=(15, 4)) * 0.6**stage
        codebooks.append(np.vstack([np.zeros((1, 4)), cb]))  # a zero code guarantees no growth
    codes, recon, norms = sub.rvq_encode(x, codebooks)
    shape(codes, (200, 4), label="codes")
    shape(norms, (200, 5), label="residual norms")
    close(norms[:, 0], np.linalg.norm(x, axis=1), label="column 0 of the norms is the input's norm")
    grew = np.diff(norms, axis=1) > 1e-12
    if grew.any():
        row, col = np.argwhere(grew)[0]
        fail(f"vector {row}'s residual norm grew at stage {col + 1}: {norms[row, col]:.4f} -> {norms[row, col + 1]:.4f}")
    close(sub.rvq_decode(codes, codebooks), recon, label="decode(encode(x)) against the returned reconstruction")
    partial = codebooks[0][codes[:, 0]] + codebooks[1][codes[:, 1]]
    close(sub.rvq_decode(codes[:, :2], codebooks), partial, label="decoding with the first two stages only")


@check("fsq_quantize matches the lesson's example and passes the tanh gradient straight through")
def test_fsq_quantize(sub):
    z = torch.tensor([[0.3, -2.0, 1.2]], dtype=torch.float64, requires_grad=True)
    z_hat = sub.fsq_quantize(z, [5, 5, 5])
    close(z_hat.detach().numpy(), [[1.0, -2.0, 2.0]], label="quantized values")
    equal(np.asarray(sub.fsq_index(z_hat.detach().numpy(), [5, 5, 5])).tolist(), [103], label="index")
    z_hat.sum().backward()
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
