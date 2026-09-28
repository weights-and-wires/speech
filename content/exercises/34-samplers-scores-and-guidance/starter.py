"""Samplers, scores, and guidance.

Conventions: DDPM and DDIM put data at t = 0 and noise at t = T, with
alpha_bar[0] = 1. The VP probability-flow ODE runs over t in [0, 1], data at
t = 0. EDM indexes noise by sigma, data at sigma = 0. Flow matching, in a
later lesson, puts noise at t = 0 instead.

Fill in the functions below, then press Run to watch DDIM converge, compare
Euler and Heun on EDM's schedule, and hear classifier-free guidance narrow
a distribution of musical notes.
"""

import math

import numpy as np

from speechlab import play, show_plot


# DDIM -------------------------------------------------------------------------


def ddim_sigma(ab_t: float, ab_prev: float, eta: float) -> float:
    """The noise scale of a DDIM step from alpha_bar ab_t to the less noisy ab_prev.

    sigma = eta * sqrt((1 - ab_prev) / (1 - ab_t)) * sqrt(1 - ab_t / ab_prev).
    """
    raise NotImplementedError("ddim_sigma")


def ddim_step(x_t, eps_hat, ab_t: float, ab_prev: float, eta: float, z):
    """One DDIM update from alpha_bar ab_t to ab_prev.

    x0_hat = (x_t - sqrt(1 - ab_t) eps_hat) / sqrt(ab_t), then
    x_prev = sqrt(ab_prev) x0_hat + sqrt(1 - ab_prev - sigma^2) eps_hat + sigma z,
    with sigma = ddim_sigma(ab_t, ab_prev, eta). Works on floats or arrays.
    """
    raise NotImplementedError("ddim_step")


def ddim_sample(eps_fn, x_T, timesteps, alpha_bar, eta: float, rng):
    """Run DDIM from x_T through a descending list of integer steps, then one step to t = 0.

    timesteps is, for example, [1000, 900, ..., 100]. Step from each entry to
    the next, and from the last entry to 0, where alpha_bar[0] = 1. eps_fn(x, t)
    returns the noise estimate. Draw z with rng.standard_normal(x.shape) when
    eta > 0, and use zeros when eta = 0. Returns the final array.
    """
    raise NotImplementedError("ddim_sample")


def vp_to_edm(x_t, ab):
    """The EDM view of a VP sample: returns (x_t / sqrt(ab), sqrt((1 - ab) / ab))."""
    raise NotImplementedError("vp_to_edm")


# The probability-flow ODE ---------------------------------------------------------


def vp_marginal(t, beta_min=0.1, beta_max=20.0):
    """Provided: (a(t), sigma(t)) of the continuous VP process, x_t = a x_0 + sigma eps."""
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    a = math.exp(-0.5 * integral)
    return a, math.sqrt(1 - a * a)


def pf_ode_sample(score_fn, x1, n_steps: int, beta_min=0.1, beta_max=20.0, t_end=1e-3):
    """Euler integration of the VP probability-flow ODE from t = 1 down to t_end.

    dx/dt = -1/2 beta(t) (x + score_fn(x, t)), with beta(t) = beta_min +
    t (beta_max - beta_min). Use the n_steps + 1 times np.linspace(1, t_end,
    n_steps + 1); each step is x += (t_next - t) * drift(x, t).
    """
    raise NotImplementedError("pf_ode_sample")


# EDM --------------------------------------------------------------------------


def edm_sigmas(N: int, sigma_min=0.002, sigma_max=80.0, rho=7.0) -> np.ndarray:
    """Karras et al.'s noise levels, shape (N + 1,).

    sigma_i = (sigma_max^(1/rho) + i / (N - 1) (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
    for i = 0..N-1, followed by a final 0.
    """
    raise NotImplementedError("edm_sigmas")


def edm_sample(denoiser, x, sigmas, heun: bool = True):
    """Integrate dx/dsigma = (x - D(x; sigma)) / sigma from sigmas[0] down to sigmas[-1] = 0.

    denoiser(x, sigma) returns D. Each step takes an Euler step with slope
    d = (x - D(x; s)) / s. With heun=True, and when the next sigma is not 0,
    recompute the slope at the Euler point and step with the average of the
    two slopes. The last step, to sigma = 0, is always plain Euler.
    """
    raise NotImplementedError("edm_sample")


# Classifier-free guidance ---------------------------------------------------------


def cfg_combine(cond, uncond, w: float):
    """Classifier-free guidance in the w convention: (1 + w) cond - w uncond.

    w = 0 is no guidance. It applies to noise, denoiser, score, or velocity
    predictions alike.
    """
    raise NotImplementedError("cfg_combine")


# Run ------------------------------------------------------------------------


def _gaussian_denoiser(mu, s):
    return lambda x, sigma: (s * s * x + sigma * sigma * mu) / (s * s + sigma * sigma)


def _mixture_denoiser(means, s):
    means = np.asarray(means, dtype=float)

    def D(x, sigma):
        var = s * s + sigma * sigma
        logw = -0.5 * (x[..., None] - means) ** 2 / var
        w = np.exp(logw - logw.max(axis=-1, keepdims=True))
        w /= w.sum(axis=-1, keepdims=True)
        per = (s * s * x[..., None] + sigma * sigma * means) / var
        return (w * per).sum(axis=-1)

    return D


def _notes(semitones, rate=16000, note_s=0.3):
    t = np.arange(int(rate * note_s)) / rate
    env = np.minimum(1, np.minimum(t, note_s - t) / 0.02)
    out = []
    for m in semitones:
        f = 440.0 * 2 ** ((m - 69) / 12)
        out.append(env * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)))
    return 0.4 * np.concatenate(out) / 1.3


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. DDIM with the exact denoiser for x0 ~ N(2, 0.5^2), from one fixed starting point.
    T = 1000
    betas = np.concatenate([[0.0], np.linspace(1e-4, 0.02, T)])
    ab = np.cumprod(1 - betas)
    mu, s = 2.0, 0.5
    eps_fn = lambda x, t: math.sqrt(1 - ab[t]) * (x - math.sqrt(ab[t]) * mu) / (ab[t] * s * s + 1 - ab[t])
    x_T = math.sqrt(ab[T]) * mu + math.sqrt(ab[T] * s * s + 1 - ab[T]) * 1.3
    counts = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    finals = []
    for n in counts:
        steps = np.round(np.linspace(T, 0, n + 1)).astype(int)[:-1]
        finals.append(float(ddim_sample(eps_fn, x_T, steps, ab, 0.0, rng)))
    print("DDIM (eta = 0) from the latent 1.3; the exact answer is 2 + 0.5 * 1.3 = 2.65")
    for n, v in zip(counts, finals):
        print(f"  {n:5d} steps -> {v:.4f}")
    show_plot((np.array(counts), np.array(finals)), (np.array(counts), np.full(len(counts), 2.65)),
              labels=["DDIM", "exact"], label="DDIM's output against its number of steps", xlabel="Steps", ylabel="x_0", x_scale="log")

    # 2. EDM: Euler against Heun on the exact Gaussian ODE.
    exact = mu + 1.3 * s   # the ODE maps the latent 1.3 to the data quantile 1.3
    D = _gaussian_denoiser(mu, s)
    nfe_e, err_e, nfe_h, err_h = [], [], [], []
    for N in (4, 8, 16, 32, 64):
        sig = edm_sigmas(N, sigma_max=80.0)
        x0 = mu + 1.3 * math.sqrt(80.0**2 + s * s)
        err_e.append(abs(edm_sample(D, x0, sig, heun=False) - exact))
        err_h.append(abs(edm_sample(D, x0, sig, heun=True) - exact))
        nfe_e.append(N)
        nfe_h.append(2 * N - 1)
    show_plot((np.array(nfe_e), np.array(err_e)), (np.array(nfe_h), np.array(err_h)), labels=["Euler", "Heun"],
              label="EDM sampler error against network evaluations", xlabel="Network evaluations", ylabel="|error|",
              x_scale="log", y_scale="log")
    print(f"EDM with 8 levels: Euler error {err_e[1]:.2e}, Heun error {err_h[1]:.2e}")

    # 3. Classifier-free guidance on pitch: two classes of notes, low and high.
    centre, means, spread = 63.0, np.array([-6.0, 6.0]), 2.0
    D_cond = _gaussian_denoiser(means[1], spread)
    D_uncond = _mixture_denoiser(means, spread)
    sig = edm_sigmas(32, sigma_max=80.0)
    z = rng.standard_normal(4000) * 80.0
    widths = []
    for w in (0.0, 1.0, 4.0):
        guided = lambda x, sigma, w=w: cfg_combine(D_cond(x, sigma), D_uncond(x, sigma), w)
        x = edm_sample(guided, z, sig)
        widths.append((w, x.mean() + centre, x.std()))
        print(f"guidance w = {w:.0f}: mean {x.mean() + centre:.2f} semitones, std {x.std():.2f} (class: 69 +- 2)")
        if w in (0.0, 4.0):
            play(_notes(np.round(x[:12] + centre)), 16000, label=f"Twelve high-class notes sampled with w = {w:.0f}")
    grid = np.linspace(50, 80, 200)
    curves = []
    for w, m, sd in widths:
        curves.append((grid, np.exp(-0.5 * ((grid - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))))
    show_plot(*curves, labels=[f"w = {w:.0f}" for w, _, _ in widths], label="Sampled pitch density as guidance grows",
              xlabel="Pitch (MIDI semitones)", ylabel="Density")
