"""Checks for the classifier-free guidance exercise.

Guidance weight w: w = 0 is no guidance. The s convention is s = 1 + w. EDM puts data at sigma = 0.
"""

import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, shape


@check("drop_condition swaps whole rows for the null at about rate p, and leaves the rest alone")
def test_drop(sub):
    rng = np.random.default_rng(0)
    cond = rng.standard_normal((5000, 4))
    null = np.zeros(4)
    before = cond.copy()
    out = sub.drop_condition(cond, null, 0.2, np.random.default_rng(1))
    shape(out, (5000, 4), label="dropped batch")
    close(cond, before, label="the input must stay unchanged")
    dropped = np.all(out == null, axis=1)
    kept = ~dropped
    almost(float(dropped.mean()), 0.2, 0.02, label="fraction of rows replaced by the null")
    close(out[kept], cond[kept], label="rows that were kept")
    equal(np.asarray(sub.drop_condition(cond, null, 0.0, np.random.default_rng(2))), cond, label="p = 0 keeps every row")


@check("cfg_combine is (1 + w) cond - w uncond: w = 0 is conditional, w = -1 is unconditional, w = 3 gives 5")
def test_cfg_values(sub):
    almost(float(sub.cfg_combine(2.0, 1.0, 0.0)), 2.0, 0.0, label="w = 0 returns the conditional prediction")
    almost(float(sub.cfg_combine(2.0, 1.0, 1.0)), 3.0, 1e-12, label="w = 1")
    almost(float(sub.cfg_combine(2.0, 1.0, 3.0)), 5.0, 1e-12, label="(1 + 3) * 2 - 3 * 1")
    almost(float(sub.cfg_combine(2.0, 1.0, -1.0)), 1.0, 1e-12, label="w = -1 returns the unconditional prediction")
    a, b = np.arange(6.0).reshape(2, 3), np.ones((2, 3))
    shape(sub.cfg_combine(a, b, 2.0), (2, 3), label="array input")


@check("cfg_combine_s is the same guidance with s = 1 + w, so s = 3 is F5-TTS's default w = 2")
def test_cfg_s(sub):
    almost(float(sub.cfg_combine_s(2.0, 1.0, 1.0)), 2.0, 1e-12, label="s = 1 is no guidance")
    rng = np.random.default_rng(0)
    c, u = rng.standard_normal(8), rng.standard_normal(8)
    for w in (0.0, 0.5, 2.0, 4.0):
        close(sub.cfg_combine_s(c, u, 1 + w), sub.cfg_combine(c, u, w), label=f"s = {1 + w} against w = {w}")


@check("Guiding exact scores gives the score of the tilted distribution p(x|c) p(c|x)^w")
def test_cfg_tilt(sub):
    # p(x | c) = N(3, 1), p(x) = N(0, 4): the guided score is that of p(x | c) p(c | x)^w.
    x = np.linspace(-2, 5, 8)
    s_c, s_u = -(x - 3.0), -x / 4.0
    w = 2.0
    logp = lambda y: -0.5 * (y - 3) ** 2 + w * (-0.5 * (y - 3) ** 2 + y * y / 8)  # log p(x|c) + w log(p(x|c)/p(x))
    fd = (logp(x + 1e-6) - logp(x - 1e-6)) / 2e-6
    close(sub.cfg_combine(s_c, s_u, w), fd, atol=1e-6, label="guided score against d/dx log[p(x|c) p(c|x)^w]")
    # The tilted density is N(3.6, 0.4): its score at x = 2 is -(2 - 3.6) / 0.4 = 4.
    almost(float(sub.cfg_combine(1.0, -0.5, w)), 4.0, 1e-12, label="guided score at x = 2")


@check("guided_denoiser calls both denoisers at the same point and combines them with w")
def test_guided_denoiser(sub):
    D_c = lambda x, sigma: x + sigma
    D_u = lambda x, sigma: 0.5 * x
    x = np.array([1.0, 2.0, 3.0])
    close(sub.guided_denoiser(D_c, D_u, 0.0)(x, 2.0), D_c(x, 2.0), label="w = 0 is the conditional denoiser")
    close(sub.guided_denoiser(D_c, D_u, 2.0)(x, 2.0), 3 * D_c(x, 2.0) - 2 * D_u(x, 2.0), label="w = 2")


def _edm_sigmas(N, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    i = np.arange(N)
    sig = (sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(sig, 0.0)


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
        x = sub.edm_sample(sub.guided_denoiser(D_cond, D_uncond, w), z, _edm_sigmas(32))
        stats.append((float(np.mean(x)), float(np.std(x))))
    almost(stats[0][0], 6.0, 0.15, label="mean with w = 0 (the class mean)")
    almost(stats[0][1], 2.0, 0.1, label="std with w = 0 (the class std)")
    if not (stats[0][0] < stats[1][0] < stats[2][0] and stats[0][1] > stats[1][1] > stats[2][1]):
        fail(f"(mean, std) for w = 0, 1, 4 were {[(round(m, 2), round(d, 2)) for m, d in stats]}; "
             "guidance must move the mean away from the other class and shrink the spread")
