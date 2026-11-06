"""Few-step generation and audio metrics.

Convention (as in the two flow-matching chapters): t = 0 is noise and t = 1 is data, x0 is
noise, x1 is data, and a flow's velocity points from noise toward data.
MeanFlow's paper runs time the other way; write `meanflow_target` in this
course's direction.

Fill in the seven functions below, then press Run to train a plain flow, a
shortcut model, and a MeanFlow model, hear what each makes in one step, and
score them with FAD. Press Check to test your code.
"""

import math

import numpy as np
import torch
from scipy import linalg

from speechlab import play, show_image, show_plot


def shortcut_target(model, x_t, t, d):
    """The self-consistency target of a shortcut model, without gradient.

    `model(x, t, d)` is a velocity conditioned on the step size d. x_t is
    (B, D), and t and d are (B,). Take one step of size d from x_t, then a
    second step of size d from where it lands, and return the average of the
    two velocities, shape (B, D). Compute it under torch.no_grad().
    """
    raise NotImplementedError


def shortcut_loss(model, x0, x1, t, d):
    """The shortcut-model loss (Frans et al.), a scalar tensor.

    x_t = (1 - t) x0 + t x1. Rows with d == 0 regress model(x_t, t, 0) onto
    the flow target x1 - x0. Rows with d > 0 regress model(x_t, t, 2d) onto
    shortcut_target(model, x_t, t, d). Return the mean squared error over
    every row and element.
    """
    raise NotImplementedError


def meanflow_target(u_fn, x, s, e, v):
    """The MeanFlow regression target, detached, in the noise-at-0 convention.

    `u_fn(x, s, e)` predicts the average velocity from time s to a later time
    e, so that a jump x + (e - s) u_fn(x, s, e) lands where the flow is at e.
    x and v are (B, D), and s and e are (B,). Return, shape (B, D),

        v + (e - s) * du/ds

    where du/ds = v . grad_x u + d_s u is the derivative of u_fn along the
    flow with e held fixed. One torch.func.jvp with tangents (v, ones, zeros)
    computes it. Detach the result so no gradient flows through the target.
    """
    raise NotImplementedError


def frechet_distance(mu1, cov1, mu2, cov2):
    """The Fréchet distance between N(mu1, cov1) and N(mu2, cov2), a float.

    ||mu1 - mu2||^2 + tr(cov1 + cov2 - 2 (cov1 cov2)^(1/2)). Use
    scipy.linalg.sqrtm, and keep only the real part of its result.
    """
    raise NotImplementedError


def fad(emb_a, emb_b):
    """The Fréchet audio distance between two embedding sets of shape (n, dim).

    Fit each set's mean and its unbiased covariance, np.cov(emb, rowvar=False),
    and return their Fréchet distance.
    """
    raise NotImplementedError


def fad_infinity(emb_ref, emb_gen, sizes, seed=0):
    """FAD extrapolated to an infinite generated set (fadtk's FAD-infinity).

    For each N in `sizes`, draw N rows of emb_gen without replacement with
    np.random.default_rng(seed) and compute fad(emb_ref, subset). Fit a
    straight line of FAD against 1 / N and return its value at 1 / N = 0.
    """
    raise NotImplementedError


def clap_score(text_emb, audio_emb):
    """The mean cosine similarity between row i of text_emb and row i of audio_emb."""
    raise NotImplementedError


# The Run block: three generators, one step each, heard and measured ----------

RATE = 16000
SCALE = np.array([0, 2, 4, 5, 7, 9, 11, 12])  # C major, semitones above C4


def ring(n, gen):
    """Eight Gaussian blobs of standard deviation 0.1 on a circle of radius 2."""
    k = torch.randint(0, 8, (n,), generator=gen)
    angle = 2 * math.pi * k / 8
    return 2 * torch.stack([torch.cos(angle), torch.sin(angle)], 1) + 0.1 * torch.randn(n, 2, generator=gen)


class Net(torch.nn.Module):
    """An MLP on (x, t, extra): extra is the step size d, or the end time e."""

    def __init__(self, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, width), torch.nn.SiLU(), torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(), torch.nn.Linear(width, 2))

    def forward(self, x, t, extra):
        return self.net(torch.cat([x, t[:, None], extra[:, None]], 1))


def train_flow(steps, gen):
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = torch.randn(512, 2, generator=gen), ring(512, gen)
        t = torch.rand(512, generator=gen)
        loss = shortcut_loss(model, x0, x1, t, torch.zeros(512))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_shortcut(steps, gen, levels=5):
    """Step sizes d = 1/32 ... 1/2 (and 2d up to 1); a quarter of each batch trains self-consistency."""
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = torch.randn(512, 2, generator=gen), ring(512, gen)
        d = torch.zeros(512)
        k = torch.randint(1, levels + 1, (128,), generator=gen)
        d[:128] = 2.0 ** (-k.float())
        # Start each self-consistency row on its own grid, so t + 2d <= 1.
        grid = torch.floor(torch.rand(512, generator=gen) / (2 * d.clamp(min=1e-9)).clamp(max=1)) * 2 * d
        t = torch.where(d > 0, grid, torch.rand(512, generator=gen))
        loss = shortcut_loss(model, x0, x1, t, d)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_meanflow(steps, gen):
    """A quarter of each batch uses an interval s < e; the rest have s = e, plain flow matching."""
    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = torch.randn(512, 2, generator=gen), ring(512, gen)
        a, b = torch.rand(512, generator=gen), torch.rand(512, generator=gen)
        s, e = torch.minimum(a, b), torch.maximum(a, b)
        e = torch.where(torch.arange(512) < 128, e, s)
        x_s = (1 - s[:, None]) * x0 + s[:, None] * x1
        target = meanflow_target(model, x_s, s, e, x1 - x0)
        loss = ((model(x_s, s, e) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def notes(points, seconds=0.2):
    """One note per 2-D point: the angle picks the pitch, the radius the loudness."""
    pts = np.asarray(points, dtype=float)
    angle = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)
    position = angle / (2 * np.pi / 8)
    low = np.floor(position).astype(int) % 8
    high = np.where(low < 7, SCALE[np.minimum(low + 1, 7)], 12)
    semitone = SCALE[low] + (position - np.floor(position)) * (high - SCALE[low])
    freq = 261.63 * 2 ** (semitone / 12)
    loud = 0.05 + 0.3 * np.minimum(np.hypot(pts[:, 0], pts[:, 1]) / 2, 1.0)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = np.minimum(1, t / 0.01) * np.exp(-4 * t)
    return loud[:, None] * envelope[None, :] * np.sin(2 * np.pi * freq[:, None] * t[None, :])


def embed(clips):
    """A stand-in audio embedding: 12 log band energies of each note's first 32 ms."""
    spectrum = np.abs(np.fft.rfft(clips[:, :512] * np.hanning(512), axis=1)) ** 2
    edges = np.geomspace(200, 1100, 13) / RATE * 512
    bands = np.stack([spectrum[:, int(lo):int(hi) + 1].sum(axis=1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    return np.log(bands + 0.01 * bands.sum(axis=1, keepdims=True))


if __name__ == "__main__":
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    models = {"flow": train_flow(2000, gen), "shortcut": train_shortcut(2000, gen),
              "MeanFlow": train_meanflow(2000, gen)}

    def generate(name, noise, nfe):
        model, x, h = models[name], noise, 1.0 / nfe
        with torch.no_grad():
            for k in range(nfe):
                t = torch.full((len(x),), k * h)
                if name == "flow":
                    x = x + h * model(x, t, torch.zeros_like(t))
                elif name == "shortcut":
                    x = x + h * model(x, t, torch.full_like(t, h))
                else:
                    x = x + h * model(x, t, t + h)
        return x.numpy()

    real = ring(2000, torch.Generator().manual_seed(9)).numpy()
    real_emb = embed(notes(real))
    noise = torch.randn(2000, 2, generator=torch.Generator().manual_seed(3))
    nfes = [1, 2, 4, 8]
    curves = {}
    for name in models:
        curves[name] = [fad(real_emb, embed(notes(generate(name, noise, n)))) for n in nfes]
        print(f"{name:9s} FAD on note embeddings at NFE {nfes}: " + ", ".join(f"{v:.2f}" for v in curves[name]))
    show_plot(*[(np.array(nfes), np.array(curves[n])) for n in models], labels=list(models),
              x_scale="log", y_scale="log", label="FAD against the number of steps (lower is better)",
              xlabel="Network evaluations (NFE)", ylabel="FAD on note embeddings")

    for name in models:
        out = generate(name, noise, 1)
        h, _, _ = np.histogram2d(out[:, 1], out[:, 0], bins=64, range=[[-3, 3], [-3, 3]])
        show_image(h, label=f"{name}, one step", origin="lower", colormap="magma", xlabel="x", ylabel="y")

    order = lambda pts: pts[np.argsort(np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi))]
    for name in models:
        clip = notes(order(generate(name, noise[:12], 1))).reshape(-1)
        play(clip, RATE, label=f"{name}, one step: twelve samples as notes")

    # FAD's biases: sample size, and the embedding you pick.
    other_real = embed(notes(ring(2000, torch.Generator().manual_seed(10)).numpy()))
    print(f"FAD between two independent sets of 2000 real notes: {fad(real_emb, other_real):.3f}")
    pool = torch.randn(6000, 2, generator=torch.Generator().manual_seed(4))
    fake_emb = embed(notes(generate("flow", pool, 8)))
    sizes = [100, 200, 400, 800, 1600]
    rng = np.random.default_rng(0)
    finite = [np.mean([fad(real_emb, fake_emb[rng.choice(len(fake_emb), n, replace=False)]) for _ in range(8)])
              for n in sizes]
    print("Flow at NFE 8, FAD by generated-set size: " + ", ".join(f"{n}: {v:.3f}" for n, v in zip(sizes, finite))
          + f"; FAD-infinity {fad_infinity(real_emb, fake_emb, sizes):.3f}")
    show_plot((1 / np.array(sizes), np.array(finite)), label="FAD grows as the generated set shrinks",
              xlabel="1 / N", ylabel="FAD (mean of 8 draws)")
    point_fad = [fad(real, generate(n, noise, 1)) for n in models]
    print("One-step FAD on the raw 2-D points instead of note embeddings: " + ", ".join(
        f"{n} {v:.2f}" for n, v in zip(models, point_fad)))
