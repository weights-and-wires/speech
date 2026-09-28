"""Checks for the normalizing-flow and VITS exercise."""

import math

import numpy as np
import torch
from torch.autograd.functional import jacobian

from speechcheck import almost, check, close, fail, has

F64 = torch.float64


class _Net(torch.nn.Module):
    """A small random network returning (s, t), for coupling tests."""

    def __init__(self, d_in, d_out, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = torch.randn(d_in, 8, generator=g, dtype=F64)
        self.w2 = torch.randn(8, 2 * d_out, generator=g, dtype=F64) * 0.5

    def forward(self, x):
        s, t = (torch.tanh(x @ self.w1) @ self.w2).chunk(2, dim=1)
        return s, t


def _two_moons(n, rng):
    theta = rng.uniform(0, np.pi, n)
    upper = rng.random(n) < 0.5
    x = np.where(upper, np.cos(theta), 1 - np.cos(theta))
    y = np.where(upper, np.sin(theta), 0.5 - np.sin(theta))
    pts = np.stack([x, y], axis=1) + 0.08 * rng.standard_normal((n, 2))
    return (pts - np.array([0.5, 0.25])) / np.array([0.87, 0.5])


@check("coupling_forward reproduces the worked 2-D example, log p = -11.071883")
def test_coupling_example(sub):
    fn = has(sub, "coupling_forward")
    x = torch.tensor([[1.0, 2.0]], dtype=F64)
    y, logdet = fn(x, lambda a: (0.5 * a, a))
    close(y, [[1.0, 2 * math.exp(0.5) + 1]], atol=1e-9, label="y for x = (1, 2)")
    close(logdet, [0.5], atol=1e-12, label="log |det J|")
    log_p = -math.log(2 * math.pi) - 0.5 * float((y**2).sum()) + float(logdet[0])
    almost(log_p, -11.071883, 1e-6, label="log p_X(x) built from your y and log-determinant")


@check("coupling_inverse undoes coupling_forward, with either half conditioning")
def test_coupling_inverse(sub):
    fwd = has(sub, "coupling_forward")
    inv = has(sub, "coupling_inverse")
    x = torch.tensor(np.random.default_rng(0).normal(size=(5, 6)))
    net = _Net(3, 3, seed=1)
    for flip in (False, True):
        y, _ = fwd(x, net, flip=flip)
        keep = slice(3, 6) if flip else slice(0, 3)
        close(y[:, keep], x[:, keep], atol=0, label=f"the conditioning half with flip={flip}, which must pass through")
        close(inv(y, net, flip=flip), x, atol=1e-10, label=f"inverse(forward(x)) with flip={flip}")


@check("The coupling log-determinant matches slogdet of the autograd Jacobian")
def test_coupling_jacobian(sub):
    fwd = has(sub, "coupling_forward")
    net = _Net(2, 2, seed=2)
    x0 = torch.tensor(np.random.default_rng(3).normal(size=4))
    for flip in (False, True):
        J = jacobian(lambda v: fwd(v[None], net, flip=flip)[0][0], x0)
        _, logdet = fwd(x0[None], net, flip=flip)
        almost(logdet[0], torch.linalg.slogdet(J).logabsdet, 1e-9, label=f"log |det J| with flip={flip}")


@check("actnorm_init whitens each channel, and actnorm_forward's log-determinant is T times the sum of log-scales")
def test_actnorm(sub):
    init = has(sub, "actnorm_init")
    fwd = has(sub, "actnorm_forward")
    rng = np.random.default_rng(4)
    x = torch.tensor(rng.normal(size=(6, 3, 10)) * np.array([2.0, 0.5, 5.0])[None, :, None] + 3.0)
    log_scale, bias = init(x)
    y, _ = fwd(x, log_scale, bias)
    close(y.mean(dim=(0, 2)), np.zeros(3), atol=1e-10, label="per-channel mean after initialization")
    close(y.std(dim=(0, 2), unbiased=False), np.ones(3), atol=1e-10, label="per-channel standard deviation after initialization")
    small = torch.tensor(rng.normal(size=(1, 3, 4)))
    ls, b = torch.tensor([0.3, -0.2, 0.5], dtype=F64), torch.tensor([1.0, 0.0, -1.0], dtype=F64)
    J = jacobian(lambda v: fwd(v.reshape(1, 3, 4), ls, b)[0].reshape(-1), small.reshape(-1))
    _, logdet = fwd(small, ls, b)
    almost(logdet[0], torch.linalg.slogdet(J).logabsdet, 1e-9, label="ActNorm log |det J| on a (1, 3, 4) input")
    almost(logdet[0], 4 * 0.6, 1e-12, label="ActNorm log |det J| against T * sum(log_scale) = 4 * 0.6")


@check("invconv_forward's log-determinant is T log |det W|, and invconv_inverse undoes it")
def test_invconv(sub):
    fwd = has(sub, "invconv_forward")
    inv = has(sub, "invconv_inverse")
    rng = np.random.default_rng(5)
    W = torch.tensor(np.linalg.qr(rng.normal(size=(3, 3)))[0] @ np.diag([1.5, 0.7, 2.0]))
    x = torch.tensor(rng.normal(size=(2, 3, 4)))
    J = jacobian(lambda v: fwd(v.reshape(1, 3, 4), W)[0].reshape(-1), x[:1].reshape(-1))
    y, logdet = fwd(x, W)
    almost(logdet[0], torch.linalg.slogdet(J).logabsdet, 1e-9, label="1x1-convolution log |det J| on a (1, 3, 4) input")
    almost(logdet[1], 4 * math.log(1.5 * 0.7 * 2.0), 1e-9, label="the same against T log |det W|")
    close(inv(y, W), x, atol=1e-10, label="inverse(forward(x))")


@check("RealNVP.log_prob is a density: it integrates to 1 over the plane")
def test_log_prob_normalized(sub):
    cls = has(sub, "RealNVP")
    torch.manual_seed(6)
    model = cls(dim=2, layers=4, hidden=16).double()
    xs = torch.linspace(-9, 9, 361, dtype=F64)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="xy"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        total = float(torch.exp(model.log_prob(grid)).sum()) * float(xs[1] - xs[0]) ** 2
    almost(total, 1.0, 0.01, label="the integral of exp(log_prob) over [-9, 9] squared")


@check("RealNVP.sample inverts RealNVP.forward: forward(sample(n, seed)) gives back the Gaussian draws")
def test_sample_inverts(sub):
    cls = has(sub, "RealNVP")
    torch.manual_seed(7)
    model = cls(dim=2, layers=4, hidden=16)
    with torch.no_grad():
        x = model.sample(50, seed=11)
        z, _ = model(x)
    want = torch.randn(50, 2, generator=torch.Generator().manual_seed(11))
    close(z, want, atol=1e-4, label="forward(sample(50, seed=11)) against torch.randn(50, 2) with seed 11")


@check("train_flow fits two moons far better than the best single Gaussian")
def test_train_flow(sub):
    fn = has(sub, "train_flow")
    rng = np.random.default_rng(8)
    data = torch.tensor(_two_moons(3000, rng), dtype=torch.float32)
    test = torch.tensor(_two_moons(2000, rng), dtype=torch.float32)
    model = fn(data, steps=600, seed=0)
    with torch.no_grad():
        nll = -model.log_prob(test).mean().item()
    cov = torch.tensor(np.cov(data.numpy().T), dtype=torch.float32)
    gauss = torch.distributions.MultivariateNormal(data.mean(dim=0), cov)
    base = -gauss.log_prob(test).mean().item()
    if not nll < base - 0.4:
        fail(f"after 600 steps the flow's held-out NLL is {nll:.3f} nats, and the best Gaussian's is "
             f"{base:.3f}; the flow should beat it by at least 0.4 nats")


@check("vits_kl matches the single-sample formula and ignores masked frames")
def test_vits_kl_formula(sub):
    fn = has(sub, "vits_kl")
    rng = np.random.default_rng(9)
    shape = (2, 3, 5)
    z_p, logs_q, m_p, logs_p = (torch.tensor(rng.normal(size=shape) * s) for s in (1.0, 0.3, 1.0, 0.3))
    mask = torch.ones(2, 1, 5, dtype=F64)
    mask[1, :, 3:] = 0
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
    want = float((kl * mask).sum() / (mask.expand(shape)).sum())
    got = fn(z_p, logs_q, m_p, logs_p, mask.expand(shape))
    almost(got, want, 1e-10, label="the masked mean of the KL term")
    z_p2 = z_p.clone()
    z_p2[1, :, 3:] = 100.0
    almost(fn(z_p2, logs_q, m_p, logs_p, mask.expand(shape)), want, 1e-10, label="the KL after changing masked frames")


@check("Averaged over posterior samples, vits_kl equals the closed-form Gaussian KL")
def test_vits_kl_expectation(sub):
    fn = has(sub, "vits_kl")
    m_q, logs_q = torch.tensor([0.3, -1.0], dtype=F64), torch.tensor([-0.5, 0.2], dtype=F64)
    m_p, logs_p = torch.tensor([0.0, 0.5], dtype=F64), torch.tensor([0.1, -0.3], dtype=F64)
    eps = torch.tensor(np.random.default_rng(10).standard_normal((200000, 2)))
    z = m_q + torch.exp(logs_q) * eps  # identity flow: z_p = z
    ones = torch.ones_like(z)
    got = float(fn(z, logs_q.expand_as(z), m_p.expand_as(z), logs_p.expand_as(z), ones))
    s_q, s_p = torch.exp(logs_q), torch.exp(logs_p)
    closed = (logs_p - logs_q + (s_q**2 + (m_q - m_p) ** 2) / (2 * s_p**2) - 0.5).mean()
    almost(got, float(closed), 0.01, label="the Monte Carlo mean against KL(q || p) per dimension")
