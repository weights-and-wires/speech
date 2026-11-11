"""Straighter paths and better solvers.

Convention: t = 0 is noise and t = 1 is data. x0 is a noise sample, x1 is a
data sample. `ot_path`, `flow_matching_loss`, and `gaussian_velocity` come from
the flow-matching chapter and are given in full below.

Fill in the three functions marked below, then press Run to race the solvers,
train and reflow a 2-D flow, and hear its samples as notes. Press Check to test
your code. Every function must work on numpy arrays and on torch tensors unless
its docstring says otherwise, so use arithmetic and `.reshape`, not numpy-only
calls.
"""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from speechlab import play, show_image, show_plot



# Given from the flow-matching chapter: do not change these three. ----------

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


# Your three functions. -------------------------------------------------

def integrate(velocity, x0, n_steps, method="euler"):
    """Integrate dx/dt = velocity(x, t) from t = 0 to t = 1 in n_steps equal steps.

    `velocity(x, t)` takes a state and a float time. `method` is "euler",
    "midpoint", or "rk4". Step k starts at t = k / n_steps exactly. Returns the
    state at t = 1, same shape as x0.
    """
    raise NotImplementedError


def minibatch_ot_pairing(x0, x1):
    """The pairing of a noise batch with a data batch that moves the least.

    x0 and x1 are numpy arrays of shape (n, ...). Returns an integer array
    perm of shape (n,), a permutation of 0..n-1, that minimizes
    sum_i ||x0[i] - x1[perm[i]]||^2.
    """
    raise NotImplementedError


def straightness(velocity, x0, n_steps):
    """How far a flow's trajectories are from straight lines, a float.

    Run Euler from x0 (shape (N, D)) for n_steps, keeping each state z_k and
    each velocity v_k = velocity(z_k, k / n_steps). With z1 the final state,
    return the mean over steps k and samples i of
    ||(z1[i] - x0[i]) - v_k[i]||^2, the squared norm taken over D.
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

    # 1. Solvers on the Gaussian flow, where the exact endpoint is known, and
    #    their orders on the generic field dx/dt = t x, where x(1) = x0 e^(1/2).
    mu, s, x0 = 2.0, 0.5, 1.3
    exact = mu + s * x0
    field = lambda x, t: gaussian_velocity(x, t, mu, s)
    print(f"Gaussian flow, data N({mu}, {s}^2), x0 = {x0}: the exact endpoint is {exact}")
    curves = []
    steps = [1, 2, 4, 8, 16, 32]
    for method, cost in (("euler", 1), ("midpoint", 2), ("rk4", 4)):
        errors = [abs(integrate(field, x0, n, method) - exact) for n in steps]
        tx = [abs(integrate(lambda x, t: t * x, 1.0, n, method) - math.exp(0.5)) for n in (8, 16, 32, 64)]
        order = -np.polyfit(np.log([8, 16, 32, 64]), np.log(tx), 1)[0]
        print(f"{method:8s} 1 step: {integrate(field, x0, 1, method):.4f}   "
              f"4 steps: {integrate(field, x0, 4, method):.4f}   "
              f"32 steps: {integrate(field, x0, 32, method):.6f}   order on dx/dt = t x: {order:.2f}")
        curves.append((np.array(steps) * cost, np.array(errors)))
    show_plot(*curves, labels=["Euler", "midpoint", "RK4"], x_scale="log", y_scale="log",
              label="Error against network evaluations on the Gaussian flow",
              xlabel="Network evaluations (NFE)", ylabel="|x(1) - exact|")

    # 2. Train a flow on the ring, then reflow it once.
    def independent(gen):
        return torch.randn(512, 2, generator=gen), ring(512, gen)

    base = train(independent, 3000, seed=0)
    with torch.no_grad():
        z0 = torch.randn(4096, 2, generator=torch.Generator().manual_seed(1))
        z1 = integrate(base, z0, 32, "midpoint")
        straight_base = straightness(base, z0[:1024], 32)

    def coupled(gen):
        idx = torch.randint(0, len(z0), (512,), generator=gen)
        return z0[idx], z1[idx]

    reflowed = train(coupled, 3000, seed=2)
    with torch.no_grad():
        straight_reflow = straightness(reflowed, z0[:1024], 32)
    print(f"Straightness before reflow: {straight_base:.3f}, after one reflow: {straight_reflow:.3f}")

    # 3. What each model makes in 1 and 8 Euler steps.
    with torch.no_grad():
        noise = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3))
        for name, model in (("flow", base), ("reflowed flow", reflowed)):
            for nfe in (1, 8):
                out = integrate(model, noise, nfe, "euler").numpy()
                radius = np.hypot(out[:, 0], out[:, 1])
                print(f"{name:13s} NFE {nfe}: mean radius {radius.mean():.2f} (data: 2.00)")
                if nfe == 1 or name == "flow":
                    show_image(histogram(out), label=f"{name}, {nfe} Euler step(s)",
                               origin="lower", xlabel="x", ylabel="y", colormap="magma")
        few = noise[:12]
        play(notes(integrate(base, few, 1, "euler").numpy()), RATE,
             label="Flow, 1 Euler step: every note lands near the middle, quiet and out of tune")
        play(notes(integrate(base, few, 8, "euler").numpy()), RATE,
             label="Flow, 8 Euler steps: the notes land on the C major scale")
        play(notes(integrate(reflowed, few, 1, "euler").numpy()), RATE,
             label="Reflowed flow, 1 Euler step: close to the scale already")

        # 4. Trajectories of the first coordinate against time.
        grid = np.linspace(0, 1, 33)
        for name, model in (("flow", base), ("reflowed flow", reflowed)):
            z, path = noise[:3], [noise[:3, 0].numpy()]
            for k in range(32):
                z = z + model(z, k / 32) / 32
                path.append(z[:, 0].numpy())
            path = np.array(path)
            show_plot(*[(grid, path[:, i]) for i in range(3)],
                      label=f"x-coordinate of three samples against flow time: {name}",
                      xlabel="t (0 is noise, 1 is data)", ylabel="x")
