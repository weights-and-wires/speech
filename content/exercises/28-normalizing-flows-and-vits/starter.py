"""Normalizing flows from coupling layers, ActNorm, and 1x1 convolutions, plus the VITS KL term.

Fill in the functions and the three RealNVP methods, then press Run. The run
trains an 8-layer flow on 4,000 notes of a pentatonic tune, each a point
(pitch, log length), draws the density it learned, and plays a melody it
samples.

Every log-determinant is of the data-to-latent direction, one value per
example, in nats.
"""

import math
import time

import numpy as np
import torch
from torch import nn

from speechlab import play, show_image, show_plot


def coupling_forward(x: torch.Tensor, net, flip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """An affine coupling layer, data to latent.

    Args:
        x: Shape (B, D), D even. The first half is x_a and the second x_b; with
            flip=True the roles swap, so the second half conditions the first.
        net: A callable mapping x_a, shape (B, D / 2), to (s, t), each (B, D / 2).
        flip: Which half is transformed.

    Returns:
        (y, logdet). y keeps x_a unchanged in its place and replaces x_b with
        x_b * exp(s) + t. logdet has shape (B,): the sum of s over dimensions.
    """
    raise NotImplementedError("coupling_forward")


def coupling_inverse(y: torch.Tensor, net, flip: bool = False) -> torch.Tensor:
    """Invert coupling_forward: x_b = (y_b - t) * exp(-s), with (s, t) = net(y_a).
    """
    raise NotImplementedError("coupling_inverse")


def actnorm_init(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ActNorm's data-dependent initialization.

    x has shape (B, C, T). Return (log_scale, bias), each shape (C,), such that
    x * exp(log_scale) + bias has zero mean and unit variance per channel, over
    the batch and time axes. Use the biased variance (divide by B T).
    """
    raise NotImplementedError("actnorm_init")


def actnorm_forward(x: torch.Tensor, log_scale: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ActNorm: y = x * exp(log_scale) + bias per channel, for x of shape (B, C, T).

    Return (y, logdet) with logdet of shape (B,): T * sum(log_scale) for every
    example.
    """
    raise NotImplementedError("actnorm_forward")


def invconv_forward(x: torch.Tensor, W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An invertible 1x1 convolution: y[:, :, t] = W x[:, :, t] at every time step.

    x has shape (B, C, T) and W shape (C, C). Return (y, logdet) with logdet of
    shape (B,): T * log |det W| for every example. Use torch.linalg.slogdet.
    """
    raise NotImplementedError("invconv_forward")


def invconv_inverse(y: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Invert invconv_forward: apply the inverse of W at every time step.
    """
    raise NotImplementedError("invconv_inverse")


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
        """Data to latent: run every coupling in order and add up their log-determinants.

        Layer i uses self.nets[i] with flip = (i is odd). x has shape (B, dim).
        Return (z, logdet), shapes (B, dim) and (B,).
        """
        raise NotImplementedError("forward")

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """The exact log-density of each point, shape (B,), in nats.

        log p(x) = log N(z; 0, I) + logdet, where (z, logdet) = self.forward(x).
        """
        raise NotImplementedError("log_prob")

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        """Draw n points: z = torch.randn(n, dim) from a torch.Generator seeded with
        `seed` (cast to the parameters' dtype), then invert the couplings in
        reverse order. Return shape (n, dim).
        """
        raise NotImplementedError("sample")


def train_flow(data: torch.Tensor, steps: int = 1000, lr: float = 3e-3, batch: int = 256, seed: int = 0) -> RealNVP:
    """Fit a RealNVP to data of shape (N, dim) by maximum likelihood, and return it.

    Seed torch with `seed`, build RealNVP(dim=data.shape[1]), and take `steps`
    Adam steps at learning rate lr. Each step draws `batch` random rows and
    minimizes the mean negative log_prob.
    """
    raise NotImplementedError("train_flow")


def vits_kl(z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor,
            mask: torch.Tensor) -> torch.Tensor:
    """VITS's KL term: a single-sample estimate of log q(z | x) - log p(z | text).

    Args:
        z_p: The posterior sample after the prior flow, shape (B, C, T).
        logs_q: The posterior's log standard deviations, shape (B, C, T).
        m_p, logs_p: The text prior's mean and log standard deviation, expanded
            to frames by the alignment, shape (B, C, T).
        mask: 1 on real frames and 0 on padding, broadcastable to (B, C, T).

    Returns:
        A scalar: the sum over real entries of
        logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * exp(-2 logs_p),
        divided by the number of real entries (the sum of the expanded mask).
    """
    raise NotImplementedError("vits_kl")


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
