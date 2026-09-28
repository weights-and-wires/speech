"""Checks for the few-step generation and audio metrics exercise.

Convention: t = 0 is noise and t = 1 is data, as in lessons 36 and 37.
"""

import math

import numpy as np
import torch
from scipy import linalg

from speechcheck import almost, check, close, fail, has, shape


class _Affine:
    """A mock step-conditioned field v = a x + b t + c d, with a call log."""

    def __init__(self, a=-0.5, b=1.0, c=2.0):
        self.a, self.b, self.c = a, b, c

    def __call__(self, x, t, d):
        return self.a * x + self.b * t[:, None] + self.c * d[:, None]


@check("shortcut_target averages two half steps and carries no gradient")
def test_shortcut_target(sub):
    target_fn = has(sub, "shortcut_target")
    f = _Affine()
    x = torch.tensor([[1.0, -2.0], [0.5, 0.0]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.25, 0.5], dtype=torch.float64)
    d = torch.tensor([0.125, 0.25], dtype=torch.float64)
    got = target_fn(f, x, t, d)
    first = f(x, t, d)
    mid = x + d[:, None] * first
    want = 0.5 * (first + f(mid, t + d, d))
    close(got.detach().numpy(), want.detach().numpy(), atol=1e-12, label="the two-step average")
    if got.requires_grad:
        fail("the target must be computed without gradient (under torch.no_grad or detached)")


@check("shortcut_loss is plain flow matching on rows with d = 0")
def test_shortcut_loss_flow_rows(sub):
    loss_fn = has(sub, "shortcut_loss")
    g = torch.Generator().manual_seed(0)
    x0, x1 = torch.randn(8, 2, generator=g, dtype=torch.float64), torch.randn(8, 2, generator=g, dtype=torch.float64)
    t = torch.rand(8, generator=g, dtype=torch.float64)
    d = torch.zeros(8, dtype=torch.float64)
    f = _Affine()
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    want = ((f(x_t, t, d) - (x1 - x0)) ** 2).mean()
    almost(float(loss_fn(f, x0, x1, t, d)), float(want), 1e-10, label="the loss with every d = 0")


@check("shortcut_loss asks the step-2d output to match the stop-gradient two-step target")
def test_shortcut_loss_consistency_rows(sub):
    loss_fn = has(sub, "shortcut_loss")
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()

    def model(x, t, d):
        return net(torch.cat([x, t[:, None], d[:, None]], 1))

    g = torch.Generator().manual_seed(2)
    x0, x1 = torch.randn(6, 2, generator=g, dtype=torch.float64), torch.randn(6, 2, generator=g, dtype=torch.float64)
    t = torch.tensor([0.1, 0.3, 0.5, 0.0, 0.25, 0.5], dtype=torch.float64)
    d = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.125, 0.25], dtype=torch.float64)
    loss = loss_fn(model, x0, x1, t, d)
    net.zero_grad()
    loss.backward()
    got = [p.grad.clone() for p in net.parameters()]

    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    with torch.no_grad():
        first = model(x_t, t, d)
        second = model(x_t + d[:, None] * first, t + d, d)
    target = torch.where((d > 0)[:, None], 0.5 * (first + second), x1 - x0)
    step = torch.where(d > 0, 2 * d, d)
    want_loss = ((model(x_t, t, step) - target) ** 2).mean()
    almost(loss.item(), want_loss.item(), 1e-10, label="the mixed loss")
    net.zero_grad()
    want_loss.backward()
    for i, (a, p) in enumerate(zip(got, net.parameters())):
        close(a.numpy(), p.grad.numpy(), rtol=1e-6, atol=1e-10,
              label=f"gradient {i}; a mismatch usually means gradient flows through the target")


def _gaussian_average_velocity(mu, sd):
    """The exact average velocity of the 1-D Gaussian flow between times s < e, in torch."""

    def sigma(t):
        return torch.sqrt((1 - t) ** 2 + t**2 * sd**2)

    def u(x, s, e):
        s, e = s[:, None], e[:, None]
        start = (x - s * mu) / sigma(s)  # the noise sample this trajectory started from
        x_e = e * mu + sigma(e) * start
        return (x_e - x) / (e - s)

    def v(x, t):
        t = t[:, None]
        return mu + (t * sd**2 - (1 - t)) / sigma(t) ** 2 * (x - t * mu)

    return u, v


@check("meanflow_target is a fixed point for the exact average velocity of a Gaussian flow")
def test_meanflow_fixed_point(sub):
    target_fn = has(sub, "meanflow_target")
    u, v = _gaussian_average_velocity(2.0, 0.5)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(10, 1, generator=g, dtype=torch.float64)
    s = 0.6 * torch.rand(10, generator=g, dtype=torch.float64)
    e = s + 0.05 + 0.3 * torch.rand(10, generator=g, dtype=torch.float64)
    got = target_fn(u, x, s, e, v(x, s))
    shape(got, (10, 1), label="the target")
    close(got.detach().numpy(), u(x, s, e).detach().numpy(), rtol=1e-6, atol=1e-8,
          label="u + (e - s) du/ds against u itself; a sign error in the direction of time breaks this")


@check("meanflow_target reduces to the instantaneous velocity when e = s")
def test_meanflow_reduces(sub):
    target_fn = has(sub, "meanflow_target")
    torch.manual_seed(4)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).double()

    def model(x, s, e):
        return net(torch.cat([x, s[:, None], e[:, None]], 1))

    x = torch.randn(5, 2, dtype=torch.float64)
    s = torch.rand(5, dtype=torch.float64)
    velocity = torch.randn(5, 2, dtype=torch.float64)
    got = target_fn(model, x, s, s.clone(), velocity)
    close(got.detach().numpy(), velocity.numpy(), atol=1e-12, label="the target at e = s")
    if got.requires_grad:
        fail("the target must be detached, so no gradient flows through it")


@check("frechet_distance matches the 1-D, diagonal, and commuting closed forms")
def test_frechet(sub):
    fd = has(sub, "frechet_distance")
    almost(fd(np.array([0.0]), np.array([[1.0]]), np.array([1.0]), np.array([[4.0]])), 2.0, 1e-9,
           label="N(0, 1) against N(1, 2^2), where the answer is 1 + (1 - 2)^2 = 2")
    rng = np.random.default_rng(5)
    mu1, mu2 = rng.standard_normal(4), rng.standard_normal(4)
    s1, s2 = rng.uniform(0.5, 2, 4), rng.uniform(0.5, 2, 4)
    want = np.sum((mu1 - mu2) ** 2) + np.sum((s1 - s2) ** 2)
    almost(fd(mu1, np.diag(s1**2), mu2, np.diag(s2**2)), want, 1e-8, label="diagonal covariances")
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    l1, l2 = rng.uniform(0.2, 3, 4), rng.uniform(0.2, 3, 4)
    c1, c2 = q @ np.diag(l1) @ q.T, q @ np.diag(l2) @ q.T
    want = np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(l1) - np.sqrt(l2)) ** 2)
    almost(fd(mu1, c1, mu2, c2), want, 1e-7, label="covariances that share eigenvectors")
    almost(fd(mu1, c1, mu1, c1), 0.0, 1e-7, label="a Gaussian against itself")


@check("fad fits a mean and an unbiased covariance to each set, and grows with added noise")
def test_fad(sub):
    fad = has(sub, "fad")
    fd = has(sub, "frechet_distance")
    rng = np.random.default_rng(6)
    a = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))
    b = a + 0.3 * rng.standard_normal((500, 6))
    want = fd(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
    almost(fad(a, b), want, 1e-8, label="fad(a, b) against frechet_distance of the fitted Gaussians")
    almost(fad(a, a), 0.0, 1e-6, label="fad of a set against itself")
    c = a + 1.0 * rng.standard_normal((500, 6))
    if not fad(a, c) > fad(a, b) > 0:
        fail(f"more noise must mean a larger FAD; got {fad(a, b):.4f} for noise 0.3 and {fad(a, c):.4f} for noise 1.0")


@check("FAD is biased upward for small sets, and fad_infinity removes most of the bias")
def test_fad_infinity(sub):
    fad = has(sub, "fad")
    fad_inf = has(sub, "fad_infinity")
    rng = np.random.default_rng(7)
    dim = 32
    ref = rng.standard_normal((20000, dim))
    gen = rng.standard_normal((2000, dim)) * 1.1 + 0.1
    true = dim * 0.1**2 + dim * (1.1 - 1.0) ** 2  # the population FAD, 0.64
    small = fad(ref, gen[:250])
    full = fad(ref, gen)
    if not small > full > true:
        fail(f"expected FAD at 250 samples > FAD at 2000 > the true value {true:.3f}; got {small:.3f} and {full:.3f}")
    extrapolated = fad_inf(ref, gen, [250, 500, 1000, 2000], seed=0)
    if not abs(extrapolated - true) < abs(full - true):
        fail(f"FAD-infinity is {extrapolated:.3f}; it should be closer to the true {true:.3f} than FAD on "
             f"all 2000 samples ({full:.3f})")


@check("clap_score is the mean cosine similarity of matching rows")
def test_clap(sub):
    clap = has(sub, "clap_score")
    rng = np.random.default_rng(8)
    text = rng.standard_normal((6, 5))
    audio = rng.standard_normal((6, 5))
    want = np.mean([t @ a / (np.linalg.norm(t) * np.linalg.norm(a)) for t, a in zip(text, audio)])
    almost(clap(text, audio), want, 1e-12, label="the score of random pairs")
    almost(clap(text, 3.0 * text), 1.0, 1e-12, label="audio embeddings that point where the text does")
    almost(clap(text, audio * np.arange(1, 7)[:, None]), want, 1e-12, label="the score after rescaling rows")


def _ring(n, g):
    k = torch.randint(0, 4, (n,), generator=g)
    a = 2 * math.pi * k / 4
    return 2 * torch.stack([torch.cos(a), torch.sin(a)], 1) + 0.1 * torch.randn(n, 2, generator=g)


@check("A shortcut model trained with your loss lands on the data in one step; a plain flow can't")
def test_shortcut_training(sub):
    loss_fn = has(sub, "shortcut_loss")
    torch.set_num_threads(2)

    def make():
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64),
                                  torch.nn.SiLU(), torch.nn.Linear(64, 2))
        return net, lambda x, t, d: net(torch.cat([x, t[:, None], d[:, None]], 1))

    results = {}
    for name in ("flow", "shortcut"):
        net, model = make()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(1)
        for _ in range(1200):
            x0, x1 = torch.randn(256, 2, generator=g), _ring(256, g)
            d = torch.zeros(256)
            if name == "shortcut":
                d[:64] = 2.0 ** (-torch.randint(1, 5, (64,), generator=g).float())
            u = torch.rand(256, generator=g)
            t = torch.where(d > 0, torch.floor(u / (2 * d).clamp(min=1e-9).clamp(max=1)) * 2 * d, u)
            loss = loss_fn(model, x0, x1, t, d)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = torch.randn(1000, 2, generator=torch.Generator().manual_seed(2))
            out = z + model(z, torch.zeros(1000), torch.ones(1000) if name == "shortcut" else torch.zeros(1000))
        results[name] = out.norm(dim=1).mean().item()
    if not (results["flow"] < 1.0 and abs(results["shortcut"] - 2.0) < 0.35):
        fail(f"one step should land near the mean (radius near 0) for a plain flow and on the ring "
             f"(radius near 2) for a shortcut model; got {results['flow']:.2f} and {results['shortcut']:.2f}")
