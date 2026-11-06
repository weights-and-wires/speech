"""Classifier-free guidance.

Conventions: the guidance weight w follows Ho and Salimans, so w = 0 is no
guidance. The s convention of diffusers is s = 1 + w. EDM puts data at
sigma = 0, so the samplers below integrate from large sigma down to 0.

Fill in the four guidance functions, then press Run to watch guidance move and
narrow a distribution of musical notes, and hear the result.
"""

import math

import numpy as np

from speechlab import play, show_plot


# Provided: the EDM sampler from the previous chapter ----------------------------


def edm_sigmas(N: int, sigma_min=0.002, sigma_max=80.0, rho=7.0) -> np.ndarray:
    """Provided from the previous chapter: Karras et al.'s noise levels, N of them from sigma_max to sigma_min, then a final 0."""
    i = np.arange(N)
    sig = (sigma_max ** (1 / rho) + i / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return np.append(sig, 0.0)


def edm_sample(denoiser, x, sigmas, heun: bool = True):
    """Provided from the previous chapter: integrate dx/dsigma = (x - D(x; sigma)) / sigma down the sigmas, with Heun's correction
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


def drop_condition(cond, null, p: float, rng):
    """Replace each row of cond by null with probability p, for classifier-free training.

    cond has shape (batch, dim) and null has shape (dim,). Draw one number per
    row with rng.random(len(cond)) and drop the row when it is below p. Returns
    a new array of the same shape, and leaves cond unchanged.
    """
    raise NotImplementedError("drop_condition")


def cfg_combine(cond, uncond, w: float):
    """Guidance in the w convention: (1 + w) cond - w uncond; w = 0 is no guidance.

    It applies to noise, denoiser, score, or velocity predictions alike, and to
    floats or arrays.
    """
    raise NotImplementedError("cfg_combine")


def cfg_combine_s(cond, uncond, s: float):
    """Guidance in the s convention of diffusers: uncond + s (cond - uncond).

    s = 1 is no guidance, so s = 1 + w.
    """
    raise NotImplementedError("cfg_combine_s")


def guided_denoiser(D_cond, D_uncond, w: float):
    """Return a denoiser D(x, sigma) that guides D_cond against D_uncond with weight w.

    The returned function calls both denoisers at the same (x, sigma) and
    combines their outputs with cfg_combine.
    """
    raise NotImplementedError("guided_denoiser")


# Provided: exact denoisers and a note synthesizer for the Run block -----------


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

    # 1. The hand example: conditional 2, unconditional 1.
    for w in (0.0, 1.0, 3.0):
        print(f"w = {w:.0f}: guided prediction {cfg_combine(2.0, 1.0, w):.1f}  (s = {1 + w:.0f}: {cfg_combine_s(2.0, 1.0, 1 + w):.1f})")

    # 2. Classifier-free guidance on pitch: two classes of notes, low and high.
    centre, means, spread = 63.0, np.array([-6.0, 6.0]), 2.0
    D_cond = _gaussian_denoiser(means[1], spread)
    D_uncond = _mixture_denoiser(means, spread)
    sig = edm_sigmas(32, sigma_max=80.0)
    z = rng.standard_normal(4000) * 80.0
    widths = []
    for w in (0.0, 1.0, 4.0):
        x = edm_sample(guided_denoiser(D_cond, D_uncond, w), z, sig)
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
