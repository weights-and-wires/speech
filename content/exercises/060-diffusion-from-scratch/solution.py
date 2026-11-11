"""Diffusion from scratch: the worked solution.

Convention: data at t = 0, noise at t = T. Every schedule array has T + 1
entries so that index t means step t: betas[0] = 0 and alpha_bar[0] = 1.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from speechlab import play, show_image, show_plot, show_spectrogram


# Schedules --------------------------------------------------------------------


def linear_betas(T: int, beta_1: float = 1e-4, beta_T: float = 0.02) -> torch.Tensor:
    """beta_t rising linearly from beta_1 to beta_T, as float64 of length T + 1 with betas[0] = 0."""
    return torch.cat([torch.zeros(1, dtype=torch.float64), torch.linspace(beta_1, beta_T, T, dtype=torch.float64)])


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol and Dhariwal's cosine schedule, length T + 1 with betas[0] = 0, clipped at 0.999."""
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = torch.clamp(1 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
    return torch.cat([torch.zeros(1, dtype=torch.float64), betas])


def alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """The cumulative product of 1 - beta, so alpha_bar[0] = 1."""
    return torch.cumprod(1 - betas, dim=0)


def _at(values: torch.Tensor, t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """values[t], reshaped to (B, 1, ..., 1) so it broadcasts against `like`."""
    return values[t].reshape(-1, *([1] * (like.dim() - 1))).to(like.dtype)


# The forward process ----------------------------------------------------------


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    """x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, for a batch of steps t."""
    ab = _at(alpha_bar, t, x0)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


# Conversions between parameterizations -------------------------------------------


def predict_x0(x_t, eps, ab):
    """The data implied by x_t and a noise estimate."""
    return (x_t - (1 - ab) ** 0.5 * eps) / ab**0.5


def velocity(x0, eps, ab):
    """v = sqrt(alpha_bar) eps - sqrt(1 - alpha_bar) x0."""
    return ab**0.5 * eps - (1 - ab) ** 0.5 * x0


def from_velocity(x_t, v, ab):
    """(x0, eps) implied by x_t and a velocity estimate."""
    a, s = ab**0.5, (1 - ab) ** 0.5
    return a * x_t - s * v, s * x_t + a * v


def score_from_eps(eps, ab):
    """The score, grad log p(x_t), implied by a noise estimate."""
    return -eps / (1 - ab) ** 0.5


# The posterior ----------------------------------------------------------------


def posterior(x0, x_t, t: int, betas: torch.Tensor, alpha_bar: torch.Tensor):
    """Mean and variance of q(x_{t-1} | x_t, x0)."""
    beta, ab, ab_prev = betas[t], alpha_bar[t], alpha_bar[t - 1]
    alpha = 1 - beta
    mean = (ab_prev.sqrt() * beta / (1 - ab)) * x0 + (alpha.sqrt() * (1 - ab_prev) / (1 - ab)) * x_t
    var = (1 - ab_prev) / (1 - ab) * beta
    return mean, var


# Training and sampling ----------------------------------------------------------


def ddpm_loss(model, x0: torch.Tensor, alpha_bar: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """The simplified DDPM loss: mean squared error between the noise and its prediction."""
    T = alpha_bar.shape[0] - 1
    t = torch.randint(1, T + 1, (x0.shape[0],), generator=generator)
    eps = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    x_t = q_sample(x0, t, eps, alpha_bar)
    return ((model(x_t, t) - eps) ** 2).mean()


@torch.no_grad()
def ddpm_sample(model, shape, betas: torch.Tensor, alpha_bar: torch.Tensor, generator: torch.Generator, keep=()):
    """Ancestral sampling from t = T down to 0. Returns (x0, {t: x_t for t in keep})."""
    T = betas.shape[0] - 1
    x = torch.randn(shape, generator=generator, dtype=torch.float64)
    saved = {T: x.clone()} if T in keep else {}
    for t in range(T, 0, -1):
        tt = torch.full((shape[0],), t, dtype=torch.long)
        eps = model(x, tt).to(torch.float64)
        beta, ab = betas[t], alpha_bar[t]
        mean = (x - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()
        if t > 1:
            var = (1 - alpha_bar[t - 1]) / (1 - ab) * beta
            x = mean + var.sqrt() * torch.randn(shape, generator=generator, dtype=torch.float64)
        else:
            x = mean
        if t - 1 in keep:
            saved[t - 1] = x.clone()
    return x, saved


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
