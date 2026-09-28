"""Checks for the samplers, scores, and guidance exercise.

DDIM: data at t = 0, alpha_bar[0] = 1. VP ODE: t in [0, 1], data at 0. EDM: data at sigma = 0.
"""

import math

import numpy as np

from speechcheck import almost, check, close, fail, shape

MU, S = 2.0, 0.5


def _linear_alpha_bar(T=1000):
    betas = np.concatenate([[0.0], np.linspace(1e-4, 0.02, T)])
    return np.cumprod(1 - betas)


def _eps_oracle(ab):
    return lambda x, t: math.sqrt(1 - ab[t]) * (x - math.sqrt(ab[t]) * MU) / (ab[t] * S * S + 1 - ab[t])


@check("ddim_step reproduces x_1 = 1.106797, and ddim_sigma with eta = 1 is DDPM's posterior noise")
def test_ddim_step(sub):
    almost(float(sub.ddim_step(1.113103, 0.5, 0.72, 0.9, 0.0, 0.0)), 1.106797, 1e-6, label="x_1 from x_2 = 1.113103, eps = 0.5")
    almost(sub.ddim_sigma(0.72, 0.9, 0.0), 0.0, 0.0, label="sigma for eta = 0")
    almost(sub.ddim_sigma(0.72, 0.9, 1.0) ** 2, 0.1 / 0.28 * 0.2, 1e-12, label="sigma^2 for eta = 1 (the posterior variance 0.071429)")
    ab = _linear_alpha_bar()
    for t in (10, 500, 1000):
        beta_t = 1 - ab[t] / ab[t - 1]
        want = (1 - ab[t - 1]) / (1 - ab[t]) * beta_t
        almost(sub.ddim_sigma(ab[t], ab[t - 1], 1.0) ** 2, want, 1e-12, label=f"eta = 1 variance at t = {t}")


@check("DDIM with the exact denoiser converges toward the exact answer 2.65 as steps grow")
def test_ddim_converges(sub):
    ab = _linear_alpha_bar()
    x_T = math.sqrt(ab[1000]) * MU + math.sqrt(ab[1000] * S * S + 1 - ab[1000]) * 1.3
    got = {}
    for n in (1, 10, 50, 1000):
        steps = np.round(np.linspace(1000, 0, n + 1)).astype(int)[:-1]
        got[n] = float(sub.ddim_sample(_eps_oracle(ab), x_T, steps, ab, 0.0, np.random.default_rng(0)))
    want = {1: 2.0021, 10: 2.4809, 50: 2.6128, 1000: 2.6480}
    for n in want:
        almost(got[n], want[n], 2e-3, label=f"DDIM output with {n} step(s)")


@check("DDIM with eta = 1 over all 1000 steps samples the right Gaussian")
def test_ddim_eta1(sub):
    ab = _linear_alpha_bar()
    rng = np.random.default_rng(1)
    x_T = rng.standard_normal(20000)
    steps = np.arange(1000, 0, -1)
    x = sub.ddim_sample(_eps_oracle(ab), x_T, steps, ab, 1.0, np.random.default_rng(2))
    shape(x, (20000,), label="samples")
    almost(float(np.mean(x)), MU, 0.03, label="sample mean (data 2.0)")
    almost(float(np.std(x)), S, 0.02, label="sample std (data 0.5)")


@check("vp_to_edm rescales a VP sample so one DDIM step is one Euler step in sigma")
def test_vp_to_edm(sub):
    x, sigma = sub.vp_to_edm(1.113103, 0.72)
    almost(float(sigma), math.sqrt(0.28 / 0.72), 1e-12, label="sigma for alpha_bar = 0.72")
    almost(float(x), 1.113103 / math.sqrt(0.72), 1e-12, label="the rescaled sample")
    ab = _linear_alpha_bar()
    rng = np.random.default_rng(3)
    for t, tp in [(900, 700), (300, 100)]:
        x_t = rng.standard_normal()
        eps = _eps_oracle(ab)(x_t, t)
        x_prev = sub.ddim_step(x_t, eps, ab[t], ab[tp], 0.0, 0.0)
        y, s1 = sub.vp_to_edm(x_t, ab[t])
        y_prev, s0 = sub.vp_to_edm(x_prev, ab[tp])
        almost(float(y_prev), float(y + (s0 - s1) * eps), 1e-9, label=f"DDIM step {t} -> {tp} as an Euler step in sigma")


@check("pf_ode_sample integrates the VP probability-flow ODE to the data quantile, error falling with steps")
def test_pf_ode(sub):
    def marg(t):
        integral = 0.1 * t + 0.5 * 19.9 * t * t
        a = math.exp(-0.5 * integral)
        return a, math.sqrt(1 - a * a)

    def score(x, t):
        a, sig = marg(t)
        return -(x - a * MU) / (a * a * S * S + sig * sig)

    a1, s1 = marg(1.0)
    x1 = a1 * MU + math.sqrt(a1 * a1 * S * S + s1 * s1) * 1.3
    ae, se = marg(1e-3)
    exact = ae * MU + math.sqrt(ae * ae * S * S + se * se) * 1.3
    errs = [abs(float(sub.pf_ode_sample(score, x1, n)) - exact) for n in (20, 40, 80)]
    if not errs[2] < 0.01:
        fail(f"with 80 Euler steps the endpoint misses the exact {exact:.4f} by {errs[2]:.4f}, expected under 0.01")
    ratios = [errs[0] / errs[1], errs[1] / errs[2]]
    if not all(1.6 < r < 2.5 for r in ratios):
        fail(f"doubling the steps shrank the error by {ratios[0]:.2f} and {ratios[1]:.2f}; Euler is first order, so about 2")


@check("edm_sigmas gives (80, 17.528, 2.515, 0.1698, 0.002) and a final 0 for N = 5")
def test_edm_sigmas(sub):
    sig = sub.edm_sigmas(5)
    shape(sig, (6,), label="sigmas for N = 5, plus the final 0")
    close(sig, [80.0, 17.527832, 2.515219, 0.169753, 0.002, 0.0], rtol=1e-5, label="sigmas")


@check("edm_sample with Heun is second order and Euler first order on the exact Gaussian ODE")
def test_edm_heun(sub):
    D = lambda x, sigma: (S * S * x + sigma * sigma * MU) / (S * S + sigma * sigma)
    x0 = MU + 1.3 * math.sqrt(80.0**2 + S * S)
    exact = MU + 1.3 * S
    heun = [abs(float(sub.edm_sample(D, x0, sub.edm_sigmas(N), heun=True)) - exact) for N in (16, 32, 64)]
    euler = [abs(float(sub.edm_sample(D, x0, sub.edm_sigmas(N), heun=False)) - exact) for N in (16, 32, 64)]
    if not all(3.2 < heun[i] / heun[i + 1] < 5.5 for i in range(2)):
        fail(f"Heun errors {['%.2e' % e for e in heun]} for N = 16, 32, 64 should shrink about 4x per doubling")
    if not all(1.6 < euler[i] / euler[i + 1] < 2.5 for i in range(2)):
        fail(f"Euler errors {['%.2e' % e for e in euler]} for N = 16, 32, 64 should shrink about 2x per doubling")
    rng = np.random.default_rng(4)
    x = sub.edm_sample(D, MU + rng.standard_normal(20000) * math.sqrt(80.0**2 + S * S), sub.edm_sigmas(32))
    almost(float(np.mean(x)), MU, 0.02 * MU, label="Heun sample mean")
    almost(float(np.std(x)), S, 0.02 * S, label="Heun sample std")


@check("cfg_combine uses the w convention, and guiding exact scores gives the tilted distribution's score")
def test_cfg(sub):
    almost(float(sub.cfg_combine(2.0, 1.0, 0.0)), 2.0, 0.0, label="w = 0 returns the conditional prediction")
    almost(float(sub.cfg_combine(2.0, 1.0, 3.0)), 5.0, 1e-12, label="(1 + 3) * 2 - 3 * 1")
    # p(x | c) = N(3, 1), p(x) = N(0, 4): the guided score is that of p(x | c) p(c | x)^w.
    x = np.linspace(-2, 5, 8)
    s_c, s_u = -(x - 3.0), -x / 4.0
    w = 2.0
    logp = lambda y: -0.5 * (y - 3) ** 2 + w * (-0.5 * (y - 3) ** 2 + y * y / 8)  # log p(x|c) + w log(p(x|c)/p(x))
    fd = (logp(x + 1e-6) - logp(x - 1e-6)) / 2e-6
    close(sub.cfg_combine(s_c, s_u, w), fd, atol=1e-6, label="guided score against d/dx log[p(x|c) p(c|x)^w]")


@check("Guidance on a two-class mixture pushes samples away from the other class and narrows them")
def test_cfg_sampling(sub):
    means, s = np.array([-6.0, 6.0]), 2.0

    def D_cond(x, sigma):
        return (s * s * x + sigma * sigma * means[1]) / (s * s + sigma * sigma)

    def D_uncond(x, sigma):
        var = s * s + sigma * sigma
        logw = -0.5 * (x[..., None] - means) ** 2 / var
        wts = np.exp(logw - logw.max(-1, keepdims=True))
        wts /= wts.sum(-1, keepdims=True)
        return (wts * (s * s * x[..., None] + sigma * sigma * means) / var).sum(-1)

    z = np.random.default_rng(5).standard_normal(4000) * math.sqrt(80.0**2 + s * s) + 6.0
    stats = []
    for w in (0.0, 1.0, 4.0):
        x = sub.edm_sample(lambda y, sg, w=w: sub.cfg_combine(D_cond(y, sg), D_uncond(y, sg), w), z, sub.edm_sigmas(32))
        stats.append((float(np.mean(x)), float(np.std(x))))
    almost(stats[0][0], 6.0, 0.15, label="mean with w = 0 (the class mean)")
    almost(stats[0][1], 2.0, 0.1, label="std with w = 0 (the class std)")
    if not (stats[0][0] < stats[1][0] < stats[2][0] and stats[0][1] > stats[1][1] > stats[2][1]):
        fail(f"(mean, std) for w = 0, 1, 4 were {[(round(m, 2), round(d, 2)) for m, d in stats]}; "
             "guidance must move the mean away from the other class and shrink the spread")
