"""Checks for the flow-matching-for-speech exercise.

Convention: t = 0 is noise and t = 1 is data. Mels are (batch, bands, frames),
and a mask is (batch, frames) with 1 marking a frame the model must generate.
"""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


@check("sway matches F5-TTS's worked grid, is the identity at s = 0, and stays monotone for s in [-1, 1.75]")
def test_sway(sub):
    sway = has(sub, "sway")
    u = np.arange(9) / 8
    want = [0, 0.0192, 0.0761, 0.1685, 0.2929, 0.4444, 0.6173, 0.8049, 1.0]
    close(sway(u, -1.0), want, atol=1e-4, label="sway(i / 8, s = -1)")
    close(sway(u, -1.0), 1 - np.cos(np.pi * u / 2), atol=1e-12, label="sway(u, -1) against 1 - cos(pi u / 2)")
    close(sway(u, 0.0), u, atol=1e-15, label="sway(u, 0)")
    close(sway(torch.tensor(u), -1.0).numpy(), sway(u, -1.0), atol=1e-12, label="sway on a torch tensor")
    dense = np.linspace(0, 1, 2001)
    for s in (-1.0, -0.5, 0.5, 1.75):
        steps = np.diff(sway(dense, s))
        if steps.min() < -1e-12:
            fail(f"sway(u, {s}) decreases somewhere; the smallest step is {steps.min():.3g}")
        close([sway(0.0, s), sway(1.0, s)], [0.0, 1.0], atol=1e-12, label=f"the end points at s = {s}")


def _batch(seed=0, b=3, f=4, n=6):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(b, f, n, generator=g, dtype=torch.float64)
    x1 = torch.randn(b, f, n, generator=g, dtype=torch.float64)
    t = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64)[:b]
    mask = torch.zeros(b, n, dtype=torch.float64)
    mask[:, 2:5] = 1
    mask[0] = 1
    return x0, x1, t, mask


@check("infilling_batch builds x_t, the unmasked context, and the target x1 - x0")
def test_infilling_batch(sub):
    infilling_batch = has(sub, "infilling_batch")
    x0, x1, t, mask = _batch()
    x_t, context, target = infilling_batch(x0, x1, t, mask)
    for name, value in (("x_t", x_t), ("context", context), ("target", target)):
        shape(value, x0.shape, label=name)
    close(x_t[0].numpy(), x0[0].numpy(), atol=1e-12, label="x_t at t = 0")
    close(x_t[1].numpy(), (0.6 * x0[1] + 0.4 * x1[1]).numpy(), atol=1e-12, label="x_t at t = 0.4")
    close(x_t[2].numpy(), x1[2].numpy(), atol=1e-12, label="x_t at t = 1")
    close(target.numpy(), (x1 - x0).numpy(), atol=1e-12, label="the target")
    close(context[0].numpy(), np.zeros((4, 6)), atol=0, label="the context of a fully masked row")
    close(context[1, :, :2].numpy(), x1[1, :, :2].numpy(), atol=0, label="the context on unmasked frames")
    close(context[1, :, 2:5].numpy(), np.zeros((4, 3)), atol=0, label="the context on masked frames")


@check("masked_flow_loss averages over masked frames only, and unmasked frames get no gradient")
def test_masked_loss(sub):
    loss_fn = has(sub, "masked_flow_loss")
    x0, x1, t, mask = _batch(1)
    target = x1 - x0
    v = torch.randn(x0.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64, requires_grad=True)
    loss = loss_fn(v, target, mask)
    m = mask[:, None, :]
    want = (((v - target) ** 2) * m).sum() / (m.sum() * 4)
    almost(loss.item(), want.item(), 1e-10, label="the loss")
    loss.backward()
    unmasked = v.grad * (1 - m)
    if float(unmasked.abs().max()) != 0.0:
        fail(f"the gradient on unmasked frames should be exactly 0, got up to {float(unmasked.abs().max()):.3g}")
    moved = v.detach() + 5.0 * (1 - m)
    almost(float(loss_fn(moved, target, mask)), float(want), 1e-10,
           label="the loss after changing only unmasked predictions")


@check("cfg_velocity follows the w-convention: alpha = 0 is conditional, and alpha equals s - 1")
def test_cfg(sub):
    cfg = has(sub, "cfg_velocity")
    rng = np.random.default_rng(3)
    vc, vu = rng.standard_normal(5), rng.standard_normal(5)
    close(cfg(vc, vu, 0.0), vc, atol=1e-15, label="alpha = 0")
    close(cfg(vc, vu, 2.0), 3 * vc - 2 * vu, atol=1e-12, label="alpha = 2, F5-TTS's default")
    s = 1.7
    close(cfg(vc, vu, s - 1), vu + s * (vc - vu), atol=1e-12, label="alpha = 0.7 against the s-convention with s = 1.7")


class _Linear:
    """A mock velocity field that records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, t, cond):
        self.calls.append((x.shape[0], t.clone()))
        return -0.5 * x + cond[:, None, None].to(x.dtype) + t[:, None, None]


@check("sample runs Euler on the sway grid and batches the guided and unguided passes into one call")
def test_sample_mechanics(sub):
    sample = has(sub, "sample")
    sway = has(sub, "sway")
    x0 = torch.randn(3, 2, 5, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    cond = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    null = torch.zeros(3, dtype=torch.float64)

    field = _Linear()
    got = sample(field, x0, cond, null, 6, -1.0, 2.0)
    equal(len(field.calls), 6, label="the number of calls to the field with guidance, NFE 6")
    equal([c[0] for c in field.calls], [6] * 6, label="the batch size of each guided call (2B)")
    grid = sway(np.linspace(0, 1, 7), -1.0)
    close(np.array([float(c[1][0]) for c in field.calls]), grid[:6], atol=1e-7, label="the times passed to the field")

    x = x0.clone()
    for k in range(6):
        t = torch.full((3,), float(grid[k]), dtype=torch.float64)
        vc = -0.5 * x + cond[:, None, None] + t[:, None, None]
        vu = -0.5 * x + null[:, None, None] + t[:, None, None]
        x = x + (grid[k + 1] - grid[k]) * (vc + 2.0 * (vc - vu))
    close(got.numpy(), x.numpy(), rtol=1e-6, atol=1e-6, label="the sample against a reference Euler loop")

    field = _Linear()
    sample(field, x0, cond, null, 4, 0.0, 0.0)
    equal([c[0] for c in field.calls], [3] * 4, label="the batch size of each call without guidance (B)")


def _two_class_field(mus, s):
    """The exact velocity of a two-class Gaussian toy: class c is N(mus[c], s^2) per element."""

    def cond_v(x, t, mu):
        var = (1 - t) ** 2 + t**2 * s**2
        return mu + (t * s**2 - (1 - t)) / var * (x - t * mu)

    def field(x, t, cond):
        t = t.reshape(-1, 1, 1)
        var = (1 - t) ** 2 + t**2 * s**2
        v0, v1 = cond_v(x, t, mus[0]), cond_v(x, t, mus[1])
        # The unconditional field averages the two, weighted by p(class | x_t).
        log0 = -((x - t * mus[0]) ** 2).sum(dim=(1, 2)) / (2 * var.flatten())
        log1 = -((x - t * mus[1]) ** 2).sum(dim=(1, 2)) / (2 * var.flatten())
        w1 = torch.sigmoid(log1 - log0).reshape(-1, 1, 1)
        mixture = (1 - w1) * v0 + w1 * v1
        c = cond.reshape(-1, 1, 1)
        return torch.where(c == 0, v0, torch.where(c == 1, v1, mixture))

    return field


@check("On an exact two-class Gaussian field, sample reaches the class and guidance pushes past it")
def test_sample_gaussian(sub):
    sample = has(sub, "sample")
    mus, s = (-1.0, 1.0), 0.5
    field = _two_class_field(mus, s)
    x0 = torch.randn(300, 1, 4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    cond = torch.ones(300, dtype=torch.long)
    null = torch.full((300,), 2, dtype=torch.long)
    plain = sample(field, x0, cond, null, 32, 0.0, 0.0)
    almost(float(plain.mean()), 1.0, 0.05, label="the mean of class-1 samples without guidance")
    almost(float(plain.std()), 0.5, 0.05, label="their standard deviation (the data has 0.5)")
    guided = sample(field, x0, cond, null, 32, -1.0, 2.0)
    if not float(guided.mean()) > float(plain.mean()) + 0.05:
        fail(f"guidance with alpha = 2 should push samples past the class mean away from the other class; "
             f"the mean went from {float(plain.mean()):.3f} to {float(guided.mean()):.3f}")


@check("chunk_causal_mask matches a brute-force rule, and its extremes are causal and full attention")
def test_chunk_mask(sub):
    chunk_mask = has(sub, "chunk_causal_mask")
    for n, c in ((7, 3), (10, 4), (5, 1), (6, 6), (6, 10)):
        got = np.asarray(chunk_mask(n, c)).astype(bool)
        shape(got, (n, n), label=f"the mask for {n} frames")
        want = np.array([[j // c <= i // c for j in range(n)] for i in range(n)])
        equal(got, want, label=f"the mask for {n} frames in chunks of {c}")
    equal(np.asarray(chunk_mask(5, 1)).astype(bool), np.tril(np.ones((5, 5), bool)), label="chunks of 1 (fully causal)")
    equal(np.asarray(chunk_mask(5, 5)).astype(bool), np.ones((5, 5), bool), label="one chunk (no mask)")


def _filterbank(n_mels=12, n_freq=65):
    centers = np.linspace(0, n_freq - 1, n_mels + 2)
    f = np.arange(n_freq)
    fb = np.maximum(0, 1 - np.abs(f[None, :] - centers[1:-1, None]) / (centers[1] - centers[0]))
    return fb


@check("mel_to_magnitude inverts the filterbank by least squares and never returns a negative power")
def test_mel_inverse(sub):
    invert = has(sub, "mel_to_magnitude")
    fb = _filterbank()
    rng = np.random.default_rng(6)
    power = np.exp(rng.standard_normal((65, 9)))
    mel = fb @ power
    mag = np.asarray(invert(mel, fb))
    shape(mag, (65, 9), label="the magnitude")
    if mag.min() < 0:
        fail("a magnitude is never negative")
    want = np.sqrt(np.maximum(np.linalg.pinv(fb) @ mel, 0))
    close(mag, want, rtol=1e-6, atol=1e-9, label="the magnitude against sqrt(max(pinv(fb) @ mel, 0))")
    smooth = np.outer(np.linspace(2, 1, 65), np.ones(9))
    back = fb @ np.asarray(invert(fb @ smooth, fb)) ** 2
    close(back, fb @ smooth, rtol=0.02, label="the mel of the reconstruction of a smooth spectrum")


@check("A tiny conditional flow trained with your pieces generates the class you ask for")
def test_tiny_training(sub):
    infilling_batch = has(sub, "infilling_batch")
    loss_fn = has(sub, "masked_flow_loss")
    sample = has(sub, "sample")
    torch.manual_seed(0)
    torch.set_num_threads(2)
    f, n = 6, 8
    templates = torch.stack([torch.linspace(-1, 1, f)[:, None].expand(f, n),
                             torch.linspace(1, -1, f)[:, None].expand(f, n)])
    net = torch.nn.Sequential(torch.nn.Linear(2 * f * n + 1 + 3, 128), torch.nn.SiLU(),
                              torch.nn.Linear(128, 128), torch.nn.SiLU(), torch.nn.Linear(128, f * n))

    def model(x, t, cond, context=None):
        context = torch.zeros_like(x) if context is None else context
        onehot = torch.nn.functional.one_hot(cond, 3).float()
        h = torch.cat([x.reshape(len(x), -1), context.reshape(len(x), -1), t[:, None], onehot], 1)
        return net(h).reshape(x.shape)

    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(600):
        label = torch.randint(0, 2, (128,), generator=g)
        x1 = templates[label] + 0.1 * torch.randn(128, f, n, generator=g)
        x0 = torch.randn(128, f, n, generator=g)
        t = torch.rand(128, generator=g)
        mask = torch.ones(128, n)
        x_t, context, target = infilling_batch(x0, x1, t, mask)
        cond = torch.where(torch.rand(128, generator=g) < 0.2, torch.full_like(label, 2), label)
        loss = loss_fn(model(x_t, t, cond, context), target, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        want = torch.tensor([0] * 20 + [1] * 20)
        x = sample(model, torch.randn(40, f, n, generator=torch.Generator().manual_seed(1)), want,
                   torch.full((40,), 2), 8, -1.0, 1.0)
        dist = ((x[:, None] - templates[None]) ** 2).mean(dim=(2, 3))
        right = float((dist.argmin(1) == want).float().mean())
        err = float(((x - templates[want]) ** 2).mean())
    if right < 0.95 or err > 0.1:
        fail(f"expected at least 95% of samples nearest their own class template with a mean squared "
             f"error under 0.1; got {right:.0%} and {err:.3f}")
