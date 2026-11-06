"""Flow matching.

Convention: t = 0 is noise and t = 1 is data (Lipman et al. 2022, Matcha-TTS,
Voicebox, F5-TTS). x0 is a noise sample, x1 is a data sample. The diffusion
chapters use the opposite direction; `to_diffusion` converts between the two.

Fill in the five functions below, then press Run to train a 2-D flow and hear
what one Euler step and eight Euler steps make of it. Press Check to test your
code. Every function must work on numpy arrays and on torch tensors unless its
docstring says otherwise, so use arithmetic and `.reshape`, not numpy-only calls.
"""

import math

import numpy as np
import torch

from speechlab import play, show_image, show_plot


def ot_path(x0, x1, t, sigma_min=0.0):
    """The point on the straight conditional path at time t, and its target velocity.

    x0 (noise) and x1 (data) have the same shape (B, ...). t is a float or an
    array of shape (B,), one time per example; reshape it to (B, 1, ..., 1) so
    it broadcasts. Returns (x_t, target):

        x_t    = (1 - (1 - sigma_min) t) x0 + t x1
        target = x1 - (1 - sigma_min) x0
    """
    raise NotImplementedError


def flow_matching_loss(model, x0, x1, t, sigma_min=0.0):
    """The conditional flow-matching loss, a scalar torch tensor.

    Build x_t and the target with `ot_path`, call `model(x_t, t)`, and return
    the mean over every element of the squared difference to the target.
    """
    raise NotImplementedError


def gaussian_velocity(x, t, mu, s):
    """The exact marginal velocity field when the data is N(mu, s^2) in 1-D.

    With independent noise x0 ~ N(0, 1) and sigma_min = 0, x_t is Gaussian
    with mean t mu and variance (1 - t)^2 + t^2 s^2. Returns the velocity at
    the point x and the float time t, same shape as x.
    """
    raise NotImplementedError


def euler(velocity, x0, n_steps):
    """Integrate dx/dt = velocity(x, t) from t = 0 to t = 1 with n_steps Euler steps.

    `velocity(x, t)` takes a state and a float time. Step k starts at
    t = k / n_steps exactly, and each step is x + h * velocity(x, t) with
    h = 1 / n_steps. Returns the state at t = 1, same shape as x0.
    """
    raise NotImplementedError


def to_diffusion(x_t, t, u):
    """Rewrite a flow-matching state in the DDPM convention (data at time 0).

    x_t is a point on the path t * data + (1 - t) * noise, t is a float in
    (0, 1), and u is the flow velocity data - noise at that point. Returns
    (alpha_bar, x_vp, v):

        alpha_bar  the DDPM cumulative product with the same signal-to-noise
                   ratio as time t, a float
        x_vp       sqrt(alpha_bar) * data + sqrt(1 - alpha_bar) * noise
        v          the v-prediction target alpha * noise - sigma * data, with
                   alpha = sqrt(alpha_bar) and sigma = sqrt(1 - alpha_bar)
    """
    raise NotImplementedError


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
