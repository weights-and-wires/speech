"""Flow matching: the worked solution.

Convention (Lipman et al. 2022, Matcha-TTS, Voicebox, F5-TTS): t = 0 is noise,
t = 1 is data. x0 is a noise sample and x1 is a data sample.
"""

import math

import numpy as np
import torch

from speechlab import play, show_image, show_plot


def _expand(t, x):
    """Give a per-example time of shape (B,) trailing axes so it broadcasts against x."""
    if isinstance(t, (int, float)) or t.ndim == 0:
        return t
    return t.reshape(tuple(t.shape) + (1,) * (x.ndim - t.ndim))


def ot_path(x0, x1, t, sigma_min=0.0):
    """The point on the straight conditional path at time t, and its target velocity."""
    tt = _expand(t, x0)
    x_t = (1 - (1 - sigma_min) * tt) * x0 + tt * x1
    target = x1 - (1 - sigma_min) * x0
    return x_t, target


def flow_matching_loss(model, x0, x1, t, sigma_min=0.0):
    """The conditional flow-matching loss: mean squared error to the path's velocity."""
    x_t, target = ot_path(x0, x1, t, sigma_min)
    return ((model(x_t, t) - target) ** 2).mean()


def gaussian_velocity(x, t, mu, s):
    """The exact marginal velocity when the data is N(mu, s^2) and the noise is N(0, 1)."""
    var = (1 - t) ** 2 + t**2 * s**2
    return mu + (t * s**2 - (1 - t)) / var * (x - t * mu)


def euler(velocity, x0, n_steps):
    """Integrate dx/dt = velocity(x, t) from t = 0 to t = 1 with n_steps Euler steps."""
    h = 1.0 / n_steps
    x = x0
    for k in range(n_steps):
        x = x + h * velocity(x, k / n_steps)
    return x


def to_diffusion(x_t, t, u):
    """Rewrite a flow-matching state in the DDPM convention (data at 0).

    Returns (alpha_bar, x_vp, v): the DDPM cumulative product with the same
    signal-to-noise ratio, the matching variance-preserving point
    sqrt(alpha_bar) * data + sqrt(1 - alpha_bar) * noise, and the diffusion
    v-prediction target alpha * noise - sigma * data for that point.
    """
    r = math.sqrt(t**2 + (1 - t) ** 2)
    alpha, sigma = t / r, (1 - t) / r
    data = x_t + (1 - t) * u
    noise = x_t - t * u
    return alpha**2, x_t / r, alpha * noise - sigma * data


# The Run block: a 2-D flow whose samples are notes -------------------------

RATE = 16000
SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])  # C major, in semitones above C4


def ring(n, gen):
    """Eight Gaussian blobs of standard deviation 0.1 on a circle of radius 2."""
    k = torch.randint(0, 8, (n,), generator=gen)
    angle = 2 * math.pi * k / 8
    centers = 2 * torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    return centers + 0.1 * torch.randn(n, 2, generator=gen)


class Velocity(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 2),
        )

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype).expand(x.shape[0]).reshape(-1, 1)
        return self.net(torch.cat([x, t], dim=1))


def train(sample_pair, steps, seed):
    torch.manual_seed(seed)
    model = Velocity()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x0, x1 = sample_pair(gen)
        t = torch.rand(len(x0), generator=gen)
        loss = flow_matching_loss(model, x0, x1, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def notes(points, seconds=0.22):
    """One note per 2-D point: the angle picks the pitch, the radius the loudness."""
    pts = np.asarray(points, dtype=float)
    pts = pts[np.argsort(np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi))]
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    envelope = np.minimum(1, t / 0.01) * np.exp(-4 * t)
    out = []
    for x, y in pts:
        position = np.mod(np.arctan2(y, x), 2 * np.pi) / (2 * np.pi / 8)
        low = int(np.floor(position)) % 8
        frac = position - np.floor(position)
        high = SCALE[low + 1] if low < 7 else SCALE[0] + 12
        semitone = SCALE[low] + frac * (high - SCALE[low])
        freq = 261.63 * 2 ** (semitone / 12)
        loud = 0.05 + 0.3 * min(np.hypot(x, y) / 2, 1.0)
        out.append(loud * envelope * np.sin(2 * np.pi * freq * t))
    return np.concatenate(out)


def histogram(points, lim=3.0, bins=64):
    pts = np.asarray(points, dtype=float)
    h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=bins, range=[[-lim, lim], [-lim, lim]])
    return h


if __name__ == "__main__":
    torch.set_num_threads(2)

    # 1. Euler on the Gaussian flow, where the exact endpoint is known.
    mu, s, x0 = 2.0, 0.5, 1.3
    field = lambda x, t: gaussian_velocity(x, t, mu, s)
    print(f"Gaussian flow, data N({mu}, {s}^2), x0 = {x0}: the exact endpoint is {mu + s * x0}")
    for n in (1, 2, 4, 8, 32):
        print(f"Euler with {n:2d} steps ends at {euler(field, x0, n):.4f}")

    # 2. Train a flow on the ring with random pairs.
    def independent(gen):
        return torch.randn(512, 2, generator=gen), ring(512, gen)

    base = train(independent, 3000, seed=0)

    # 3. What one and eight Euler steps make.
    with torch.no_grad():
        noise = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3))
        for nfe in (1, 8):
            out = euler(base, noise, nfe).numpy()
            radius = np.hypot(out[:, 0], out[:, 1])
            print(f"NFE {nfe}: mean radius {radius.mean():.2f} (data: 2.00)")
            show_image(histogram(out), label=f"flow, {nfe} Euler step(s)",
                       origin="lower", xlabel="x", ylabel="y", colormap="magma")
        few = noise[:12]
        play(notes(euler(base, few, 1).numpy()), RATE,
             label="Flow, 1 Euler step: every note lands near the middle, quiet and out of tune")
        play(notes(euler(base, few, 8).numpy()), RATE,
             label="Flow, 8 Euler steps: the notes land on the C major scale")

        # 4. Trajectories of the first coordinate against time.
        grid = np.linspace(0, 1, 33)
        z, path = noise[:3], [noise[:3, 0].numpy()]
        for k in range(32):
            z = z + base(z, k / 32) / 32
            path.append(z[:, 0].numpy())
        path = np.array(path)
        show_plot(*[(grid, path[:, i]) for i in range(3)],
                  label="x-coordinate of three samples against flow time",
                  xlabel="t (0 is noise, 1 is data)", ylabel="x")
