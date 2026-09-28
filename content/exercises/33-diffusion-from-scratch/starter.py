"""Diffusion from scratch.

Convention: data at t = 0, noise at t = T. Every schedule array has T + 1
entries so that index t means step t: betas[0] = 0 and alpha_bar[0] = 1.
Steps t run from 1 to T.

Fill in the functions below, then press Run to hear a voice-like signal come
out of white noise under the exact denoiser, and to train a small DDPM on a
ring of eight Gaussians.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from speechlab import play, show_image, show_plot, show_spectrogram


# Schedules --------------------------------------------------------------------


def linear_betas(T: int, beta_1: float = 1e-4, beta_T: float = 0.02) -> torch.Tensor:
    """The linear noise schedule, as a float64 tensor of length T + 1.

    betas[0] = 0, and betas[1] .. betas[T] rise linearly from beta_1 to beta_T
    (torch.linspace).
    """
    raise NotImplementedError("linear_betas")


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol and Dhariwal's cosine schedule, float64, length T + 1, betas[0] = 0.

    alpha_bar_t = f(t) / f(0) with f(t) = cos^2((t / T + s) / (1 + s) * pi / 2),
    and beta_t = min(1 - alpha_bar_t / alpha_bar_{t-1}, 0.999) for t = 1..T.
    """
    raise NotImplementedError("cosine_betas")


def alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """alpha_bar_t = prod_{s <= t} (1 - beta_s), same length as betas, so alpha_bar[0] = 1."""
    raise NotImplementedError("alpha_bars")


# The forward process ----------------------------------------------------------


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """Jump straight to step t: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps.

    x0 and eps have shape (B, ...); t is a LongTensor of shape (B,) with one
    step per example. Index alpha_bar[t] and reshape it to (B, 1, ..., 1) so
    it broadcasts over every axis after the first.
    """
    raise NotImplementedError("q_sample")


# Conversions between parameterizations -------------------------------------------
# In these four, ab is alpha_bar_t already broadcast (or a scalar tensor).


def predict_x0(x_t, eps, ab):
    """The data implied by x_t and a noise estimate: (x_t - sqrt(1 - ab) eps) / sqrt(ab)."""
    raise NotImplementedError("predict_x0")


def velocity(x0, eps, ab):
    """The velocity target v = sqrt(ab) eps - sqrt(1 - ab) x0."""
    raise NotImplementedError("velocity")


def from_velocity(x_t, v, ab):
    """(x0, eps) implied by x_t and a velocity estimate.

    x0 = sqrt(ab) x_t - sqrt(1 - ab) v and eps = sqrt(1 - ab) x_t + sqrt(ab) v.
    """
    raise NotImplementedError("from_velocity")


def score_from_eps(eps, ab):
    """The score grad_x log p(x_t) implied by a noise estimate: -eps / sqrt(1 - ab)."""
    raise NotImplementedError("score_from_eps")


# The posterior ----------------------------------------------------------------


def posterior(x0, x_t, t: int, betas: torch.Tensor, alpha_bar: torch.Tensor):
    """Mean and variance of q(x_{t-1} | x_t, x0) for one integer step t >= 1.

    mean = sqrt(ab_{t-1}) beta_t / (1 - ab_t) * x0
           + sqrt(alpha_t) (1 - ab_{t-1}) / (1 - ab_t) * x_t
    var  = (1 - ab_{t-1}) / (1 - ab_t) * beta_t
    where alpha_t = 1 - beta_t and ab is alpha_bar. Returns (mean, var).
    """
    raise NotImplementedError("posterior")


# Training and sampling ----------------------------------------------------------


def ddpm_loss(model, x0: torch.Tensor, alpha_bar: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """The simplified DDPM loss for one batch.

    Draw one step per example with torch.randint(1, T + 1, (B,), generator=...),
    where T = len(alpha_bar) - 1, draw eps with torch.randn(x0.shape,
    generator=..., dtype=x0.dtype), build x_t with q_sample, and return the
    mean over every element of (model(x_t, t) - eps) ** 2.
    """
    raise NotImplementedError("ddpm_loss")


@torch.no_grad()
def ddpm_sample(model, shape, betas: torch.Tensor, alpha_bar: torch.Tensor, generator: torch.Generator, keep=()):
    """Ancestral sampling from pure noise at t = T down to t = 0.

    Start from x_T = torch.randn(shape, generator=..., dtype=torch.float64).
    For t = T, ..., 1, with eps = model(x, t) and t a LongTensor of shape
    (shape[0],) filled with t:
        mean = (x - beta_t / sqrt(1 - ab_t) * eps) / sqrt(1 - beta_t)
        x_{t-1} = mean + sqrt(var_t) z, z ~ N(0, I), for t > 1
        x_0 = mean                                   (no noise on the last step)
    where var_t is the posterior variance (1 - ab_{t-1}) / (1 - ab_t) beta_t.

    Returns (x_0, saved), where saved maps each step in `keep` to a copy of
    x at that step (x_T included when T is in keep).
    """
    raise NotImplementedError("ddpm_sample")


# Run ------------------------------------------------------------------------


class _MLP(nn.Module):
    """A small denoiser: the point and a sinusoidal embedding of t/T in, the noise estimate out."""

    def __init__(self, T, dim=2, hidden=128):
        super().__init__()
        self.T = T
        self.register_buffer("freqs", torch.exp(torch.arange(8) * math.log(1000.0) / 7))
        self.net = nn.Sequential(
            nn.Linear(dim + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        angle = (t.float() / self.T)[:, None] * self.freqs
        return self.net(torch.cat([x.float(), torch.sin(angle), torch.cos(angle)], dim=1))


def _ring(n, gen):
    angle = torch.randint(0, 8, (n,), generator=gen) * (2 * math.pi / 8)
    centers = 2.0 * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    return centers + 0.1 * torch.randn(n, 2, generator=gen)


def _histogram(points, lim=3.0, bins=64):
    h, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=bins, range=[[-lim, lim], [-lim, lim]])
    return h


def _voice_psd(n, rate, f0=140.0, variance=0.01):
    """A power spectrum with harmonics of f0 shaped by three formants, scaled to a per-sample variance."""
    f = np.fft.rfftfreq(n, 1 / rate)
    envelope = sum(1 / (1 + ((f - fc) / bw) ** 2) for fc, bw in [(700, 90), (1200, 110), (2600, 160)])
    comb = sum(np.exp(-0.5 * ((f - h * f0) / 6.0) ** 2) for h in range(1, int(4000 / f0)))
    psd = envelope * comb + 1e-6
    return psd * variance / psd.mean()


class _GaussianOracle:
    """The exact noise prediction when the data is a zero-mean Gaussian process with power spectrum `psd`."""

    def __init__(self, psd, alpha_bar):
        self.psd, self.alpha_bar = torch.from_numpy(psd), alpha_bar

    def __call__(self, x_t, t):
        ab = self.alpha_bar[t[0]]
        gain = (1 - ab).sqrt() / (ab * self.psd + 1 - ab)
        return torch.fft.irfft(gain * torch.fft.rfft(x_t, dim=1), n=x_t.shape[1], dim=1)


if __name__ == "__main__":
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    # 1. Hear the forward process, and walk it back with the exact denoiser.
    rate, n, T = 16000, 16000, 200
    betas = linear_betas(T, 1e-4, 0.1)
    ab = alpha_bars(betas)
    psd = _voice_psd(n, rate)
    print(f"a voice-like Gaussian signal with per-sample std {math.sqrt(psd.mean()):.2f}; T = {T}, alpha_bar_T = {ab[-1]:.2e}")
    snr_db = 10 * torch.log10(ab[1:] * psd.mean() / (1 - ab[1:]))
    show_plot((np.arange(1, T + 1), snr_db.numpy()), label="SNR of x_t for this signal", xlabel="Step t", ylabel="SNR (dB)")
    keep = (T, 25, 8, 0)
    x0, saved = ddpm_sample(_GaussianOracle(psd, ab), (1, n), betas, ab, gen, keep=keep)
    for t in keep:
        clip = np.clip(saved[t][0].numpy() * (0.3 / saved[t][0].std().item()), -1, 1)
        if t:
            label = f"x_{t} on the way back, SNR {snr_db[t - 1]:.0f} dB (loudness normalized)"
        else:
            label = "x_0, the generated voice"
        play(clip, rate, label=label)
    spec = np.abs(np.fft.rfft(x0[0].numpy().reshape(-1, 400) * np.hanning(400), axis=1)).T
    show_spectrogram(spec, rate, 400, label="The generated signal: harmonics under three formants")

    # 2. Train a small DDPM on a ring of eight Gaussians.
    T2 = 100
    betas2 = cosine_betas(T2)
    ab2 = alpha_bars(betas2)
    model = _MLP(T2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000)
    losses = []
    for step in range(3000):
        loss = ddpm_loss(model, _ring(512, gen).double(), ab2, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        decay.step()
        losses.append(loss.item())
    print(f"ring: loss {np.mean(losses[:50]):.3f} -> {np.mean(losses[-200:]):.3f}")
    show_plot(np.array(losses), label="DDPM training loss on the ring", xlabel="Step", ylabel="MSE on the noise", y_scale="log")
    samples, _ = ddpm_sample(lambda x, t: model(x, t).double(), (4000, 2), betas2, ab2, gen)
    s = samples.numpy()
    radius = np.linalg.norm(s, axis=1)
    centers = 2.0 * np.stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)], 1)
    near = np.min(np.linalg.norm(s[:, None] - centers[None], axis=2), axis=1) < 0.4
    print(f"samples: mean radius {radius.mean():.3f} (data 2.0), {near.mean():.1%} within 0.4 of a mode")
    show_image(_histogram(s), label="4000 samples from your DDPM", origin="lower", xlabel="x", ylabel="y")
