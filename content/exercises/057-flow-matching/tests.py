"""Checks for the flow-matching exercise.

Convention: t = 0 is noise and t = 1 is data, as in Lipman et al. and F5-TTS.
"""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, fail, has, shape


@check("ot_path gives the worked values, with and without sigma_min")
def test_ot_path_values(sub):
    ot_path = has(sub, "ot_path")
    x_t, u = ot_path(np.array([-1.0]), np.array([2.0]), np.array([0.25]))
    close(x_t, [-0.25], atol=1e-12, label="x_t for x0 = -1, x1 = 2, t = 0.25")
    close(u, [3.0], atol=1e-12, label="the target velocity")
    x_t, u = ot_path(np.array([-1.0]), np.array([2.0]), np.array([0.25]), sigma_min=1e-4)
    close(x_t, [-0.250025], atol=1e-12, label="x_t with sigma_min = 1e-4")
    close(u, [2.9999], atol=1e-12, label="the target with sigma_min = 1e-4")


@check("ot_path broadcasts one time per example over a (B, C, T) batch and hits both ends")
def test_ot_path_broadcast(sub):
    ot_path = has(sub, "ot_path")
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((4, 3, 5))
    x1 = rng.standard_normal((4, 3, 5))
    t = np.array([0.0, 0.3, 0.7, 1.0])
    x_t, u = ot_path(x0, x1, t)
    shape(x_t, (4, 3, 5), label="x_t")
    close(x_t[0], x0[0], atol=1e-12, label="x_t at t = 0 (pure noise)")
    close(x_t[3], x1[3], atol=1e-12, label="x_t at t = 1 (pure data)")
    close(x_t[1], 0.7 * x0[1] + 0.3 * x1[1], atol=1e-12, label="x_t at t = 0.3")
    close(u, x1 - x0, atol=1e-12, label="the target x1 - x0")
    xt_torch, _ = ot_path(torch.tensor(x0), torch.tensor(x1), torch.tensor(t))
    close(xt_torch.numpy(), x_t, atol=1e-12, label="ot_path on torch tensors")


@check("flow_matching_loss is the mean squared error to the target, and zero for a perfect model")
def test_loss(sub):
    loss_fn = has(sub, "flow_matching_loss")
    torch.manual_seed(0)
    x0 = torch.randn(6, 2, dtype=torch.float64)
    x1 = torch.randn(6, 2, dtype=torch.float64)
    t = torch.rand(6, dtype=torch.float64)
    w = torch.tensor([[0.5, -0.2], [0.1, 0.9]], dtype=torch.float64)

    def model(x, tt):
        return x @ w + tt[:, None]

    got = loss_fn(model, x0, x1, t)
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    want = ((x_t @ w + t[:, None] - (x1 - x0)) ** 2).mean()
    almost(float(got), float(want), 1e-10, label="the loss for a linear model")

    def perfect(x, tt):
        return x1 - x0

    almost(float(loss_fn(perfect, x0, x1, t)), 0.0, 1e-12, label="the loss of a model that outputs x1 - x0")


@check("gaussian_velocity is the time derivative of the exact Gaussian flow map")
def test_gaussian_velocity(sub):
    velocity = has(sub, "gaussian_velocity")
    mu, s = 2.0, 0.5
    x0 = np.array([-1.5, 0.2, 1.3])

    def flow_map(t):
        return t * mu + np.sqrt((1 - t) ** 2 + t**2 * s**2) * x0

    for t in (0.0, 0.25, 0.5, 0.9):
        eps = 1e-6
        lo, hi = max(t - eps, 0.0), t + eps
        want = (flow_map(hi) - flow_map(lo)) / (hi - lo)
        close(velocity(flow_map(t), t, mu, s), want, rtol=1e-5, atol=1e-6,
              label=f"the velocity along the flow at t = {t}")


@check("euler reproduces the worked numbers on the Gaussian flow and one step lands on the data mean")
def test_euler_values(sub):
    euler = has(sub, "euler")
    velocity = has(sub, "gaussian_velocity")
    field = lambda x, t: velocity(x, t, 2.0, 0.5)
    for n, want in ((1, 2.0), (2, 2.26), (4, 2.442703), (8, 2.539220), (32, 2.620573)):
        almost(float(euler(field, 1.3, n)), want, 1e-5, label=f"Euler with {n} steps")
    for x0 in (-2.0, 0.0, 1.3):
        almost(float(euler(field, x0, 1)), 2.0, 1e-12, label=f"one Euler step from noise {x0} (the data mean)")

    seen = []

    def spy(x, t):
        seen.append(t)
        return 0.0 * x

    euler(spy, np.zeros(1), 7)
    close(np.array(seen), np.arange(7) / 7, atol=1e-15, label="the times Euler asks the field for with 7 steps")
    out = euler(lambda x, t: 0 * x + 1.5, np.zeros((3, 2)), 5)
    close(out, np.full((3, 2), 1.5), atol=1e-12, label="Euler on a constant field of 1.5 over unit time")


@check("to_diffusion rewrites a flow state as a DDPM state with the same signal-to-noise ratio")
def test_to_diffusion(sub):
    to_diffusion = has(sub, "to_diffusion")
    rng = np.random.default_rng(1)
    data = rng.standard_normal(5)
    noise = rng.standard_normal(5)
    for t in (0.2, 0.5, 0.8):
        x_t = t * data + (1 - t) * noise
        u = data - noise
        alpha_bar, x_vp, v = to_diffusion(x_t, t, u)
        want_ab = t**2 / (t**2 + (1 - t) ** 2)
        almost(float(alpha_bar), want_ab, 1e-12, label=f"alpha_bar at t = {t}")
        a, sg = math.sqrt(want_ab), math.sqrt(1 - want_ab)
        close(x_vp, a * data + sg * noise, atol=1e-12, label=f"the DDPM point at t = {t}")
        close(v, a * noise - sg * data, atol=1e-12, label=f"the v-prediction target at t = {t}")
    _, _, v = to_diffusion(0.5 * data + 0.5 * noise, 0.5, data - noise)
    close(v, -(data - noise) / math.sqrt(2), atol=1e-12, label="v at t = 0.5, which is -u / sqrt(2)")
