"""Checks for the diffusion-from-scratch exercise.

Convention: data at t = 0. Schedules have T + 1 entries, betas[0] = 0, alpha_bar[0] = 1.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from speechcheck import almost, check, close, equal, fail, shape

F64 = torch.float64


def _hand():
    betas = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=F64)
    return betas, torch.cumprod(1 - betas, 0)


@check("linear_betas and alpha_bars follow the DDPM convention: index 0 is the data")
def test_linear(sub):
    betas = sub.linear_betas(1000)
    shape(betas, (1001,), label="betas for T = 1000 (one entry per step, plus index 0)")
    almost(float(betas[0]), 0.0, 0.0, label="betas[0]")
    close(betas[[1, 1000]].numpy(), [1e-4, 0.02], label="betas[1] and betas[T]")
    ab = sub.alpha_bars(betas)
    almost(float(ab[0]), 1.0, 0.0, label="alpha_bar[0]")
    almost(float(ab[-1]), 4.0358e-5, 1e-8, label="alpha_bar_T for the DDPM linear schedule")
    close(sub.alpha_bars(_hand()[0]).numpy(), [1.0, 0.9, 0.72, 0.504, 0.3024], label="alpha_bar for betas (0.1, 0.2, 0.3, 0.4)")


@check("cosine_betas matches the lesson's four-step values, including the clip at 0.999")
def test_cosine(sub):
    betas = sub.cosine_betas(4)
    shape(betas, (5,), label="cosine betas for T = 4")
    close(betas.numpy(), [0.0, 0.152988, 0.416958, 0.707859, 0.999], atol=1e-6, label="cosine betas")
    close(sub.alpha_bars(betas)[:4].numpy(), [1.0, 0.847012, 0.493844, 0.144272], atol=1e-6, label="cosine alpha_bar")
    T, s = 1000, 0.008
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    want = f / f[0]
    got = sub.alpha_bars(sub.cosine_betas(T)).numpy()
    close(got[:990], want[:990], rtol=1e-6, atol=1e-10, label="cosine alpha_bar for T = 1000, away from the clipped end")


@check("q_sample reproduces x_2 = 1.113103 and broadcasts one step per example")
def test_q_sample(sub):
    betas, ab = _hand()
    x2 = sub.q_sample(torch.tensor([[1.0]], dtype=F64), torch.tensor([2]), torch.tensor([[0.5]], dtype=F64), ab)
    almost(float(x2.reshape(-1)[0]), 1.113103, 1e-6, label="x_2 for x0 = 1, eps = 0.5")
    # Batch 3 of (3, 2, 3) signals: B == L, so a wrong broadcast silently mixes steps.
    x0 = torch.arange(18, dtype=F64).reshape(3, 2, 3) / 10
    eps = torch.ones(3, 2, 3, dtype=F64)
    t = torch.tensor([1, 3, 4])
    got = sub.q_sample(x0, t, eps, ab)
    want = torch.stack([ab[k].sqrt() * x0[i] + (1 - ab[k]).sqrt() * eps[i] for i, k in enumerate(t.tolist())])
    close(got.numpy(), want.numpy(), label="a batch with a different step for each example")


@check("The conversions between noise, data, velocity, and score match the worked example")
def test_conversions(sub):
    ab = torch.tensor(0.72, dtype=F64)
    x_t = torch.tensor(1.113103, dtype=F64)
    almost(float(sub.predict_x0(x_t, torch.tensor(0.5, dtype=F64), ab)), 1.0, 1e-6, label="x0 from x_t and eps")
    v = sub.velocity(torch.tensor(1.0, dtype=F64), torch.tensor(0.5, dtype=F64), ab)
    almost(float(v), -0.104886, 1e-6, label="v")
    x0, eps = sub.from_velocity(x_t, v, ab)
    almost(float(x0), 1.0, 1e-6, label="x0 recovered from v")
    almost(float(eps), 0.5, 1e-6, label="eps recovered from v")
    almost(float(sub.score_from_eps(torch.tensor(0.5, dtype=F64), ab)), -0.944911, 1e-6, label="score")
    # The score of the Gaussian kernel, checked by finite differences of its log density.
    x0, x = 0.3, 0.9
    logq = lambda y: -0.5 * (y - math.sqrt(0.72) * x0) ** 2 / 0.28
    fd = (logq(x + 1e-6) - logq(x - 1e-6)) / 2e-6
    eps = (x - math.sqrt(0.72) * x0) / math.sqrt(0.28)
    almost(float(sub.score_from_eps(torch.tensor(eps, dtype=F64), ab)), fd, 1e-6, label="score against d/dx log q(x_t | x0)")


@check("posterior gives mean 1.033199 and variance 0.071429, and matches Bayes' rule")
def test_posterior(sub):
    betas, ab = _hand()
    mean, var = sub.posterior(torch.tensor(1.0, dtype=F64), torch.tensor(1.113103, dtype=F64), 2, betas, ab)
    almost(float(mean), 1.033199, 1e-6, label="posterior mean")
    almost(float(var), 0.071429, 1e-6, label="posterior variance")
    rng = np.random.default_rng(0)
    for t in (2, 3, 4):
        x0, xt = rng.normal(size=2)
        # prior q(x_{t-1} | x0) times likelihood q(x_t | x_{t-1}), as a product of Gaussians
        m1, v1 = math.sqrt(ab[t - 1]) * x0, 1 - float(ab[t - 1])
        a = math.sqrt(1 - float(betas[t]))
        prec = 1 / v1 + a * a / float(betas[t])
        want_var = 1 / prec
        want_mean = want_var * (m1 / v1 + a * xt / float(betas[t]))
        mean, var = sub.posterior(torch.tensor(x0, dtype=F64), torch.tensor(xt, dtype=F64), t, betas, ab)
        almost(float(mean), want_mean, 1e-9, label=f"posterior mean at t = {t}")
        almost(float(var), want_var, 1e-9, label=f"posterior variance at t = {t}")


@check("ddpm_loss noises the data with q_sample at a random step and regresses the noise")
def test_loss(sub):
    betas = sub.linear_betas(50, 1e-3, 0.3)
    ab = torch.cumprod(1 - betas, 0)
    x0 = torch.randn(4000, 3, generator=torch.Generator().manual_seed(1), dtype=F64) * 0.5 + 1.0
    seen = {}

    def model(x_t, t):
        seen["x_t"], seen["t"] = x_t.detach().clone(), t.clone()
        return torch.full_like(x_t, 0.1)

    loss = sub.ddpm_loss(model, x0, ab, torch.Generator().manual_seed(0))
    t = seen["t"]
    if t.min() < 1 or t.max() > 50:
        fail(f"steps range over [{int(t.min())}, {int(t.max())}], expected [1, 50]")
    a = ab[t].reshape(-1, 1)
    eps = (seen["x_t"] - a.sqrt() * x0) / (1 - a).sqrt()
    almost(float(eps.mean()), 0.0, 0.03, label="mean of the noise implied by x_t")
    almost(float(eps.std()), 1.0, 0.03, label="std of the noise implied by x_t")
    almost(float(loss), float(((eps - 0.1) ** 2).mean()), 1e-9, label="the loss against mean((eps - prediction)^2)")


@check("ddpm_sample with the exact denoiser for Gaussian data returns that Gaussian")
def test_sample_oracle(sub):
    T = 1000
    betas = sub.linear_betas(T)
    ab = torch.cumprod(1 - betas, 0)
    mu, s = 1.5, 0.5

    def oracle(x, t):
        a = ab[t].reshape(-1, 1)
        return (1 - a).sqrt() * (x - a.sqrt() * mu) / (a * s * s + 1 - a)

    out = sub.ddpm_sample(oracle, (20000, 1), betas, ab, torch.Generator().manual_seed(0))
    x = out[0] if isinstance(out, tuple) else out
    almost(float(x.mean()), mu, 0.03, label="sample mean (data mean 1.5)")
    almost(float(x.std()), s, 0.03, label="sample std (data std 0.5)")


@check("A small DDPM trained with your loss and sampler puts its samples on the ring of eight modes")
def test_ring(sub):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    T = 100
    betas = sub.cosine_betas(T)
    ab = sub.alpha_bars(betas)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("freqs", torch.exp(torch.arange(8) * math.log(1000.0) / 7))
            self.net = nn.Sequential(nn.Linear(18, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

        def forward(self, x, t):
            angle = (t.float() / T)[:, None] * self.freqs
            return self.net(torch.cat([x.float(), torch.sin(angle), torch.cos(angle)], 1)).to(x.dtype)

    def ring(n):
        k = torch.randint(0, 8, (n,), generator=gen) * (2 * math.pi / 8)
        return (2.0 * torch.stack([torch.cos(k), torch.sin(k)], 1) + 0.1 * torch.randn(n, 2, generator=gen)).double()

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    for _ in range(3000):
        loss = sub.ddpm_loss(model, ring(512), ab, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        decay.step()
    out = sub.ddpm_sample(model, (2000, 2), betas, ab, gen)
    s = (out[0] if isinstance(out, tuple) else out).detach().numpy()
    radius = np.linalg.norm(s, axis=1).mean()
    centers = 2.0 * np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
    near = (np.min(np.linalg.norm(s[:, None] - centers[None], axis=2), axis=1) < 0.4).mean()
    almost(radius, 2.0, 0.1, label="mean radius of the samples")
    if near < 0.85:
        fail(f"{near:.0%} of samples lie within 0.4 of a mode, expected at least 85%")
