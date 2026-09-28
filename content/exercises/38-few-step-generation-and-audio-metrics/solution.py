"""Few-step generation and audio metrics: the worked solution.

Convention (as in lessons 36 and 37): t = 0 is noise and t = 1 is data, x0 is
noise, x1 is data, and a flow's velocity points from noise toward data.
MeanFlow's paper uses the opposite direction; `meanflow_target` is written in
this course's direction.
"""

import math

import numpy as np
import torch
from scipy import linalg

from speechlab import play, show_image, show_plot


def shortcut_target(model, x_t, t, d):
    """The self-consistency target for one step of size 2d: the average of two steps of size d."""
    with torch.no_grad():
        first = model(x_t, t, d)
        x_mid = x_t + d[:, None] * first
        second = model(x_mid, t + d, d)
    return 0.5 * (first + second)


def shortcut_loss(model, x0, x1, t, d):
    """Flow matching on rows with d == 0, self-consistency at step 2d on rows with d > 0."""
    x_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    flow = d == 0
    target = torch.empty_like(x_t)
    target[flow] = (x1 - x0)[flow]
    if (~flow).any():
        target[~flow] = shortcut_target(model, x_t[~flow], t[~flow], d[~flow])
    step = torch.where(flow, d, 2 * d)
    return ((model(x_t, t, step) - target) ** 2).mean()


def meanflow_target(u_fn, x, s, e, v):
    """The MeanFlow regression target u = v + (e - s) du/ds, in the noise-at-0 convention.

    du/ds is the total derivative along the flow, v . grad_x u + d_s u, which
    one Jacobian-vector product with tangents (v, 1, 0) computes.
    """
    _, du_ds = torch.func.jvp(u_fn, (x, s, e), (v, torch.ones_like(s), torch.zeros_like(e)))
    return (v + (e - s)[:, None] * du_ds).detach()


def frechet_distance(mu1, cov1, mu2, cov2):
    """The Fréchet distance between two Gaussians."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    cov1, cov2 = np.atleast_2d(cov1), np.atleast_2d(cov2)
    root = linalg.sqrtm(cov1 @ cov2)
    if np.iscomplexobj(root):
        root = root.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - 2 * root))


def fad(emb_a, emb_b):
    """The Fréchet audio distance between two sets of embeddings, each (n, dim)."""
    return frechet_distance(emb_a.mean(axis=0), np.cov(emb_a, rowvar=False),
                            emb_b.mean(axis=0), np.cov(emb_b, rowvar=False))


def fad_infinity(emb_ref, emb_gen, sizes, seed=0):
    """Extrapolate FAD to an infinite generated set: fit FAD against 1/N and read off 1/N = 0."""
    rng = np.random.default_rng(seed)
    scores = [fad(emb_ref, emb_gen[rng.choice(len(emb_gen), n, replace=False)]) for n in sizes]
    slope, intercept = np.polyfit(1.0 / np.asarray(sizes, dtype=float), scores, 1)
    return float(intercept)


def clap_score(text_emb, audio_emb):
    """The mean cosine similarity between each prompt's text embedding and its audio's embedding."""
    a = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    b = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)
    return float(np.mean(np.sum(a * b, axis=1)))


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
