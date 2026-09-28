"""Samplers, scores, and guidance: the worked solution.

Conventions: DDPM and DDIM put data at t = 0 and noise at t = T, with
alpha_bar[0] = 1. The VP probability-flow ODE runs over t in [0, 1], data at
t = 0. EDM indexes noise by sigma, data at sigma = 0. Flow matching, in a
later lesson, puts noise at t = 0 instead.
"""

import math

import numpy as np

from speechlab import play, show_plot


# DDIM -------------------------------------------------------------------------


def ddim_sigma(ab_t: float, ab_prev: float, eta: float) -> float:
    """The noise scale of a DDIM step from alpha_bar ab_t to the less noisy ab_prev."""
    return eta * math.sqrt((1 - ab_prev) / (1 - ab_t)) * math.sqrt(1 - ab_t / ab_prev)


def ddim_step(x_t, eps_hat, ab_t: float, ab_prev: float, eta: float, z):
    """One DDIM update. eta = 0 is deterministic; eta = 1 matches DDPM's posterior noise."""
    sigma = ddim_sigma(ab_t, ab_prev, eta)
    x0_hat = (x_t - math.sqrt(1 - ab_t) * eps_hat) / math.sqrt(ab_t)
    return math.sqrt(ab_prev) * x0_hat + math.sqrt(1 - ab_prev - sigma**2) * eps_hat + sigma * z


def ddim_sample(eps_fn, x_T, timesteps, alpha_bar, eta: float, rng):
    """Run DDIM through a descending list of steps, then one last step to t = 0."""
    x = np.array(x_T, dtype=float)
    steps = list(timesteps) + [0]
    for t, t_prev in zip(steps[:-1], steps[1:]):
        z = rng.standard_normal(x.shape) if eta > 0 else np.zeros_like(x)
        x = ddim_step(x, eps_fn(x, t), alpha_bar[t], alpha_bar[t_prev], eta, z)
    return x


def vp_to_edm(x_t, ab):
    """The EDM view of a VP sample: (x_t / sqrt(ab), sigma = sqrt((1 - ab) / ab))."""
    return x_t / np.sqrt(ab), np.sqrt((1 - ab) / ab)


# The probability-flow ODE ---------------------------------------------------------


def vp_marginal(t, beta_min=0.1, beta_max=20.0):
    """(a(t), sigma(t)) of the continuous VP process: x_t = a x_0 + sigma eps."""
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
    a = math.exp(-0.5 * integral)
    return a, math.sqrt(1 - a * a)


def pf_ode_sample(score_fn, x1, n_steps: int, beta_min=0.1, beta_max=20.0, t_end=1e-3):
    """Euler integration of dx/dt = -1/2 beta(t) (x + score(x, t)) from t = 1 down to t_end."""
    x = np.array(x1, dtype=float)
    ts = np.linspace(1.0, t_end, n_steps + 1)
    for t, t_next in zip(ts[:-1], ts[1:]):
        beta = beta_min + t * (beta_max - beta_min)
        drift = -0.5 * beta * (x + score_fn(x, t))
        x = x + (t_next - t) * drift
    return x


# EDM --------------------------------------------------------------------------


def edm_sigmas(N: int, sigma_min=0.002, sigma_max=80.0, rho=7.0) -> np.ndarray:
    """Karras et al.'s noise levels, N of them from sigma_max to sigma_min, then a final 0."""
    i = np.arange(N)
    sig = (sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(sig, 0.0)


def edm_sample(denoiser, x, sigmas, heun: bool = True):
    """Integrate dx/dsigma = (x - D(x; sigma)) / sigma down the sigmas, with Heun's correction
    on every step except the last one to sigma = 0."""
    x = np.array(x, dtype=float)
    for s, s_next in zip(sigmas[:-1], sigmas[1:]):
        d = (x - denoiser(x, s)) / s
        x_next = x + (s_next - s) * d
        if heun and s_next > 0:
            d_next = (x_next - denoiser(x_next, s_next)) / s_next
            x_next = x + (s_next - s) * 0.5 * (d + d_next)
        x = x_next
    return x


# Classifier-free guidance ---------------------------------------------------------


def cfg_combine(cond, uncond, w: float):
    """Guided prediction (1 + w) cond - w uncond; w = 0 is no guidance."""
    return (1 + w) * cond - w * uncond


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
