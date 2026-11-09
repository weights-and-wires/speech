"""Normalizing flows from coupling layers, ActNorm, and 1x1 convolutions, plus the VITS KL term: the worked solution."""

import math
import time

import numpy as np
import torch
from torch import nn

from speechlab import play, show_image, show_plot


def coupling_forward(x: torch.Tensor, net, flip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    half = x.shape[1] // 2
    x_a, x_b = (x[:, half:], x[:, :half]) if flip else (x[:, :half], x[:, half:])
    s, t = net(x_a)
    y_b = x_b * torch.exp(s) + t
    y = torch.cat([y_b, x_a], dim=1) if flip else torch.cat([x_a, y_b], dim=1)
    return y, s.sum(dim=1)


def coupling_inverse(y: torch.Tensor, net, flip: bool = False) -> torch.Tensor:
    half = y.shape[1] // 2
    y_a, y_b = (y[:, half:], y[:, :half]) if flip else (y[:, :half], y[:, half:])
    s, t = net(y_a)
    x_b = (y_b - t) * torch.exp(-s)
    return torch.cat([x_b, y_a], dim=1) if flip else torch.cat([y_a, x_b], dim=1)


def actnorm_init(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(0, 2))
    std = x.std(dim=(0, 2), unbiased=False)
    return -torch.log(std), -mean / std


def actnorm_forward(x: torch.Tensor, log_scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = x * torch.exp(log_scale)[None, :, None] + bias[None, :, None]
    logdet = x.shape[2] * log_scale.sum() * torch.ones(x.shape[0], dtype=x.dtype)
    return y, logdet


def invconv_forward(x: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.einsum("ij,bjt->bit", W, x)
    logdet = x.shape[2] * torch.linalg.slogdet(W).logabsdet * torch.ones(x.shape[0], dtype=x.dtype)
    return y, logdet


def invconv_inverse(y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjt->bit", torch.linalg.inv(W), y)


class CouplingNet(nn.Module):
    """Maps the conditioning half to (s, t), with s bounded by a tanh for stability."""

    def __init__(self, dim_in: int, dim_out: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 2 * dim_out))
        self.scale = nn.Parameter(torch.ones(dim_out))

    def forward(self, x_a):
        s, t = self.body(x_a).chunk(2, dim=1)
        return self.scale * torch.tanh(s), t


class RealNVP(nn.Module):
    """A stack of affine couplings over 2-D points, alternating which half is transformed."""

    def __init__(self, dim: int = 2, layers: int = 8, hidden: int = 64):
        super().__init__()
        self.dim = dim
        self.nets = nn.ModuleList(CouplingNet(dim // 2, dim - dim // 2, hidden) for _ in range(layers))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], dtype=x.dtype)
        for i, net in enumerate(self.nets):
            x, ld = coupling_forward(x, net, flip=i % 2 == 1)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x)
        log_pz = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + logdet

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        z = torch.randn(n, self.dim, generator=torch.Generator().manual_seed(seed)).to(self.nets[0].scale.dtype)
        for i in reversed(range(len(self.nets))):
            z = coupling_inverse(z, self.nets[i], flip=i % 2 == 1)
        return z


def train_flow(data: torch.Tensor, steps: int = 1000, lr: float = 3e-3, batch: int = 256, seed: int = 0) -> RealNVP:
    torch.manual_seed(seed)
    model = RealNVP(dim=data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, data.shape[0], (batch,), generator=generator)
        loss = -model.log_prob(data[idx]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def vits_kl(z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2.0 * logs_p)
    return (kl * mask).sum() / mask.sum()


# The Run experiment ------------------------------------------------------------

RATE = 16000
PENTATONIC = np.array([0, 2, 4, 7, 9, 12])  # semitones above A3 (220 Hz)


def _melody_data(n: int, rng) -> np.ndarray:
    """2-D points: (pitch in semitones above A3, log of the note's length in seconds)."""
    pitch = rng.choice(PENTATONIC, n, p=[0.25, 0.1, 0.2, 0.2, 0.15, 0.1]) + 0.15 * rng.standard_normal(n)
    short = rng.random(n) < 0.65
    log_len = np.where(short, np.log(0.16), np.log(0.42)) + 0.06 * rng.standard_normal(n)
    return np.stack([pitch, log_len], axis=1)


def _render(notes: np.ndarray) -> np.ndarray:
    pieces = []
    for semitones, log_len in notes:
        n = int(np.clip(np.exp(log_len), 0.08, 0.8) * RATE)
        t = np.arange(n) / RATE
        f = 220.0 * 2 ** (semitones / 12)
        env = np.minimum(1.0, t / 0.01) * np.exp(-3.0 * t)
        pieces.append(env * sum(np.sin(2 * np.pi * k * f * t) / k**2 for k in (1, 2, 3)))
    return 0.4 * np.concatenate(pieces)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    raw = _melody_data(4000, rng)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    data = torch.tensor((raw - mean) / std, dtype=torch.float32)
    test = torch.tensor((_melody_data(2000, rng) - mean) / std, dtype=torch.float32)

    start = time.perf_counter()
    model = train_flow(data, steps=2500, lr=2e-3)
    print(f"Trained an 8-layer RealNVP on 4,000 notes in {time.perf_counter() - start:.1f} s.")
    with torch.no_grad():
        nll = -model.log_prob(test).mean().item()
    cov = np.cov(data.numpy().T)
    gauss = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor(cov, dtype=torch.float32))
    print(f"Held-out negative log-likelihood: flow {nll:.3f} nats per note, "
          f"best single Gaussian {-gauss.log_prob(test).mean().item():.3f}.")

    xs = torch.linspace(-2.2, 2.6, 120)
    ys = torch.linspace(-1.2, 2.2, 90)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        density = torch.exp(model.log_prob(grid)).reshape(len(ys), len(xs))
    cell = float((xs[1] - xs[0]) * (ys[1] - ys[0]))
    print(f"The density sums to {density.sum().item() * cell:.3f} over the plotted window.")
    show_image(density, label="The flow's density: pitch across, note length up", xlabel="Pitch (standardized)",
               ylabel="Log length (standardized)", origin="lower", colormap="magma", value_label="p(x)")
    hist, _, _ = np.histogram2d(raw[:, 1], raw[:, 0], bins=(45, 60))
    show_image(hist, label="The training notes, binned", xlabel="Pitch", ylabel="Log length", origin="lower",
               colormap="magma", value_label="count")

    with torch.no_grad():
        many = model.sample(400, seed=4).numpy() * std + mean
        notes = model.sample(16, seed=3).numpy() * std + mean
    nearest = np.round(many[:, 0])
    in_tune = np.mean((np.abs(many[:, 0] - nearest) < 0.5) & np.isin(nearest, PENTATONIC))
    print(f"{100 * in_tune:.0f}% of 400 sampled notes lie within half a semitone of a note of the scale; "
          "the rest fall in the gaps a flow can't fully empty.")
    print("Sixteen sampled notes (semitones above A3, seconds):")
    print("  " + ", ".join(f"({p:+.1f}, {np.exp(d):.2f})" for p, d in notes))
    show_plot((np.cumsum(np.exp(notes[:, 1])), notes[:, 0]), label="The sampled melody",
              xlabel="Time (s)", ylabel="Semitones above A3")
    play(_render(notes), RATE, label="A melody sampled from your flow")
