"""Checks for the straighter-paths-and-better-solvers exercise.

Convention: t = 0 is noise and t = 1 is data, as in Lipman et al. and F5-TTS.
"""

import itertools
import math

import numpy as np
import torch

from speechcheck import almost, check, close, fail, has, shape


@check("integrate reproduces the worked Euler and midpoint numbers on the Gaussian flow")
def test_integrate_values(sub):
    integrate = has(sub, "integrate")
    velocity = has(sub, "gaussian_velocity")
    field = lambda x, t: velocity(x, t, 2.0, 0.5)
    for n, want in ((1, 2.0), (2, 2.26), (4, 2.442703), (8, 2.539220), (32, 2.620573)):
        almost(float(integrate(field, 1.3, n, "euler")), want, 1e-5, label=f"Euler with {n} steps")
    for n, want in ((1, 2.52), (2, 2.642703), (4, 2.647779)):
        almost(float(integrate(field, 1.3, n, "midpoint")), want, 1e-5, label=f"midpoint with {n} steps")
    almost(float(integrate(field, 1.3, 16, "rk4")), 2.65, 1e-6, label="RK4 with 16 steps (exact answer 2.65)")

    seen = []

    def spy(x, t):
        seen.append(t)
        return 0.0 * x

    integrate(spy, np.zeros(1), 7, "euler")
    close(np.array(seen), np.arange(7) / 7, atol=1e-15, label="the times Euler asks the field for with 7 steps")


@check("Euler, midpoint, and RK4 converge with orders near 1, 2, and 4 on dx/dt = t x")
def test_orders(sub):
    integrate = has(sub, "integrate")
    exact = 1.0 * math.exp(0.5)
    steps = [8, 16, 32, 64]
    for method, order in (("euler", 1), ("midpoint", 2), ("rk4", 4)):
        errors = [abs(float(integrate(lambda x, t: t * x, 1.0, n, method)) - exact) for n in steps]
        slope = -np.polyfit(np.log(steps), np.log(errors), 1)[0]
        if abs(slope - order) > 0.3:
            fail(f"{method} converges with order {slope:.2f}, expected about {order}. Errors: {errors}")


@check("minibatch_ot_pairing finds the cheapest pairing, checked against every permutation")
def test_ot_pairing(sub):
    pairing = has(sub, "minibatch_ot_pairing")
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((7, 2))
    x1 = rng.standard_normal((7, 2)) + 1.0
    perm = np.asarray(pairing(x0, x1))
    if sorted(perm.tolist()) != list(range(7)):
        fail(f"expected a permutation of 0..6, got {perm.tolist()}")

    def cost(p):
        return float(((x0 - x1[list(p)]) ** 2).sum())

    best = min(cost(p) for p in itertools.permutations(range(7)))
    almost(cost(perm), best, 1e-9, label="the total squared distance of your pairing")


@check("straightness is zero for straight flows and positive for the curved Gaussian flow")
def test_straightness(sub):
    straightness = has(sub, "straightness")
    velocity = has(sub, "gaussian_velocity")
    x0 = np.random.default_rng(3).standard_normal((200, 1))
    almost(straightness(lambda x, t: 0 * x + 1.5, x0, 16), 0.0, 1e-12, label="a constant field")
    mu, s = 2.0, 0.5

    def reflowed(x, t):
        # The rectified flow of the pairs (x0, mu + s x0): straight lines.
        start = (x - t * mu) / (1 - t + t * s)
        return mu + (s - 1) * start

    almost(straightness(reflowed, x0, 16), 0.0, 1e-10, label="the reflowed Gaussian field")
    curved = straightness(lambda x, t: velocity(x, t, mu, s), x0, 16)
    if not curved > 0.05:
        fail(f"the Gaussian flow's paths bend, so its straightness must be clearly positive; got {curved:.4g}")


def _ring(n, gen):
    k = torch.randint(0, 4, (n,), generator=gen)
    angle = 2 * math.pi * k / 4
    return 2 * torch.stack([torch.cos(angle), torch.sin(angle)], 1) + 0.1 * torch.randn(n, 2, generator=gen)


def _mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(3, 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 2)
    )


def _train(loss_fn, sample_pair, steps, seed):
    torch.manual_seed(seed)
    net = _mlp()

    def model(x, t):
        tt = torch.as_tensor(t, dtype=x.dtype).expand(x.shape[0]).reshape(-1, 1)
        return net(torch.cat([x, tt], 1))

    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x0, x1 = sample_pair(gen)
        loss = loss_fn(model, x0, x1, torch.rand(len(x0), generator=gen))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@check("One round of reflow straightens a trained 2-D flow and fixes its one-step samples")
def test_reflow(sub):
    loss_fn = has(sub, "flow_matching_loss")
    integrate = has(sub, "integrate")
    straightness = has(sub, "straightness")
    torch.set_num_threads(2)
    base = _train(loss_fn, lambda g: (torch.randn(256, 2, generator=g), _ring(256, g)), 1500, seed=0)
    with torch.no_grad():
        z0 = torch.randn(2048, 2, generator=torch.Generator().manual_seed(1))
        z1 = integrate(base, z0, 16, "midpoint")

    def pairs(g):
        idx = torch.randint(0, len(z0), (256,), generator=g)
        return z0[idx], z1[idx]

    reflowed = _train(loss_fn, pairs, 1500, seed=2)
    with torch.no_grad():
        probe = torch.randn(512, 2, generator=torch.Generator().manual_seed(4))
        s_base = straightness(base, probe, 16)
        s_reflow = straightness(reflowed, probe, 16)
        r_base = integrate(base, probe, 1, "euler").norm(dim=1).mean().item()
        r_reflow = integrate(reflowed, probe, 1, "euler").norm(dim=1).mean().item()
    if not s_reflow < 0.3 * s_base:
        fail(f"straightness went from {s_base:.3f} to {s_reflow:.3f}; expected reflow to cut it by more than 70%")
    if not (r_base < 1.0 and abs(r_reflow - 2.0) < 0.3):
        fail(
            f"one Euler step should land near the mean (radius near 0) before reflow and on the ring "
            f"(radius near 2) after; got mean radius {r_base:.2f} before and {r_reflow:.2f} after"
        )
