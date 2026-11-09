"""Checks for the diffusion-for-audio exercise.

Convention: data at t = 0, steps t = 1..T, alpha_bar[0] = 1. The DiffWave
repository counts steps from 0; the reference alignment here adds 1 to match.
"""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, shape

TRAIN = np.linspace(1e-4, 0.05, 50)
BASE = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])


def _reference_alignment(train, infer):
    """The loop from the DiffWave repository's inference code, shifted to count steps from 1."""
    talpha_cum = np.cumprod(1 - train)
    alpha_cum = np.cumprod(1 - infer)
    out = []
    for s in range(len(infer)):
        for t in range(len(train) - 1):
            if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                out.append(t + twiddle + 1)
                break
    return np.array(out)


@check("diffusion_embedding is DiffWave's 128-dimensional sinusoid, for whole and fractional steps")
def test_embedding(sub):
    t = torch.tensor([0.0, 3.0, 3.5, 4.0])
    e = sub.diffusion_embedding(t)
    shape(e.detach(), (4, 128), label="embedding of 4 steps")
    j = np.arange(64)
    want = np.concatenate([np.sin(3.0 * 10 ** (4 * j / 63)), np.cos(3.0 * 10 ** (4 * j / 63))])
    close(e[1].detach().numpy(), want, atol=5e-3, label="embedding of t = 3 (float32 angles reach 3e4 radians)")
    close(e[0].detach().numpy(), np.concatenate([np.zeros(64), np.ones(64)]), atol=1e-7, label="embedding of t = 0")
    half = np.concatenate([np.sin(3.5 * 10 ** (4 * j / 63)), np.cos(3.5 * 10 ** (4 * j / 63))])
    close(e[2].detach().numpy(), half, atol=5e-3, label="embedding of the fractional step t = 3.5 (don't round t)")


def _block(sub, C=4, d=3, n_mels=5, seed=0):
    torch.manual_seed(seed)
    return sub.ResidualBlock(C, d, n_mels, emb_dim=512)


@check("ResidualBlock returns a residual and a skip, each (B, C, L), with the gated activation")
def test_block_math(sub):
    blk = _block(sub)
    torch.manual_seed(1)
    x, cond, emb = torch.randn(2, 4, 30), torch.randn(2, 5, 30), torch.randn(2, 512)
    with torch.no_grad():
        res, skip = blk(x, cond, emb)
        shape(res, (2, 4, 30), label="residual output")
        shape(skip, (2, 4, 30), label="skip output")
        y = x + blk.diffusion_projection(emb)[:, :, None]
        y = blk.dilated_conv(y) + blk.conditioner_projection(cond)
        y = torch.sigmoid(y[:, :4]) * torch.tanh(y[:, 4:])
        out = blk.output_projection(y)
        close(res.numpy(), ((x + out[:, :4]) / math.sqrt(2)).numpy(), atol=1e-6, label="residual, (x + first half) / sqrt(2)")
        close(skip.numpy(), out[:, 4:].numpy(), atol=1e-6, label="skip, the second half")


@check("ResidualBlock's dilated convolution sees exactly dilation samples on each side")
def test_block_field(sub):
    for d in (1, 4, 16):
        blk = _block(sub, d=d, seed=2)
        torch.manual_seed(3)
        x, cond, emb = torch.randn(1, 4, 101), torch.randn(1, 5, 101), torch.randn(1, 512)
        with torch.no_grad():
            base = blk(x, cond, emb)[1]
            x2 = x.clone()
            x2[0, :, 50] += 1.0
            moved = (blk(x2, cond, emb)[1] - base).abs().sum(1)[0].numpy()
        where = np.flatnonzero(moved > 1e-7)
        if where.size == 0 or where.min() < 50 - d or where.max() > 50 + d:
            span = f"{where.min()}..{where.max()}" if where.size else "nowhere"
            fail(f"with dilation {d}, a change at sample 50 moved the skip output at samples {span}, expected {50 - d}..{50 + d}")


@check("diffwave_alignment matches the DiffWave reference for the BASE fast schedule")
def test_alignment(sub):
    got = np.asarray(sub.diffwave_alignment(TRAIN, BASE), dtype=float)
    want = _reference_alignment(TRAIN, BASE)
    shape(got, (6,), label="aligned steps")
    close(got, want, atol=1e-9, label="aligned steps, counted from 1")
    long = np.linspace(1e-4, 0.02, 200)
    large = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.7])
    close(np.asarray(sub.diffwave_alignment(long, large), dtype=float), _reference_alignment(long, large), atol=1e-9,
          label="the LARGE schedule aligned to the 200-step training schedule")


@check("fast_sampling_coefficients gives c1, c2, and sigma, with the BASE schedule ending at gamma_bar = 0.37579")
def test_coefficients(sub):
    c1, c2, sigma = sub.fast_sampling_coefficients(BASE)
    gbar = np.cumprod(1 - BASE)
    almost(float(gbar[-1]), 0.37579, 1e-5, label="gamma_bar at the last fast step")
    close(c1, 1 / np.sqrt(1 - BASE), label="c1 = 1 / sqrt(gamma_s)")
    close(c2, BASE / np.sqrt(1 - gbar), label="c2 = eta_s / sqrt(1 - gamma_bar_s)")
    close(sigma[1:], np.sqrt((1 - gbar[:-1]) / (1 - gbar[1:]) * BASE[1:]), label="sigma for s >= 2")
    almost(float(sigma[0]), 0.0, 0.0, label="sigma at the last step taken (s = 1), which adds no noise")


@check("grad_tts_stats matches the Grad-TTS numbers and a simulation of its forward SDE")
def test_grad_tts(sub):
    for t, keep, lam in [(0.1, 0.949, 0.0995), (0.5, 0.284, 0.9194), (1.0, 0.0067, 0.99996)]:
        mean, var = sub.grad_tts_stats(1.0, 0.0, t)
        almost(float(mean), keep, 5e-4, label=f"coefficient on X_0 at t = {t}")
        almost(float(var), lam, 5e-5, label=f"variance lambda_t at t = {t}")
    rng = np.random.default_rng(0)
    x0, mu, n, steps = 2.0, -1.0, 20000, 500
    x = np.full(n, x0)
    dt = 0.5 / steps
    for k in range(steps):
        beta = 0.05 + (20 - 0.05) * (k + 0.5) * dt
        x = x + 0.5 * (mu - x) * beta * dt + math.sqrt(beta * dt) * rng.standard_normal(n)
    mean, var = sub.grad_tts_stats(x0, mu, 0.5)
    almost(float(np.mean(x)), float(mean), 0.03, label="simulated mean at t = 0.5 against grad_tts_stats")
    almost(float(np.var(x)), float(var), 0.03, label="simulated variance at t = 0.5 against grad_tts_stats")


@check("diffwave_loss noises the audio at a random step and regresses the noise, given the mel")
def test_loss(sub):
    ab = torch.cumprod(1 - torch.cat([torch.zeros(1), torch.tensor(TRAIN)]), 0).float()
    audio = torch.randn(3000, 16, generator=torch.Generator().manual_seed(1)) * 0.2
    mel = torch.rand(3000, 5, 2)
    seen = {}

    def model(x_t, m, t):
        seen["x_t"], seen["mel"], seen["t"] = x_t.clone(), m, t.clone()
        return torch.full_like(x_t, 0.05)

    loss = sub.diffwave_loss(model, audio, mel, ab, torch.Generator().manual_seed(0))
    if seen["mel"] is not mel:
        fail("the model must receive the mel spectrogram unchanged as its second argument")
    t = seen["t"].long()
    if t.min() < 1 or t.max() > 50:
        fail(f"steps range over [{int(t.min())}, {int(t.max())}], expected [1, 50]")
    a = ab[t][:, None]
    eps = (seen["x_t"] - a.sqrt() * audio) / (1 - a).sqrt()
    almost(float(eps.std()), 1.0, 0.03, label="std of the noise implied by x_t")
    almost(float(loss), float(((eps - 0.05) ** 2).mean()), 1e-5, label="loss against mean((eps - prediction)^2)")


@check("fast_sample runs the fast schedule from the last step to the first, at the aligned steps")
def test_fast_sample(sub):
    t_align = _reference_alignment(TRAIN, BASE)
    calls = []

    def model(x, mel, t):
        calls.append(float(t[0]))
        return 0.3 * x + 0.1 * mel[:, 0, :1]

    mel = torch.rand(2, 3, 4)
    got = sub.fast_sample(model, mel, 64, BASE, t_align, torch.Generator().manual_seed(7))
    close(np.array(calls), t_align[::-1], atol=1e-6, label="the steps passed to the model, in order")
    # The same loop, written out.
    gbar = np.cumprod(1 - BASE)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 64, generator=g)
    for s in range(5, -1, -1):
        x = (x - BASE[s] / math.sqrt(1 - gbar[s]) * model(x, mel, torch.full((2,), t_align[s]))) / math.sqrt(1 - BASE[s])
        if s > 0:
            x = x + math.sqrt((1 - gbar[s - 1]) / (1 - gbar[s]) * BASE[s]) * torch.randn(2, 64, generator=g)
        x = torch.clamp(x, -1, 1)
    close(got.numpy(), x.numpy(), atol=1e-5, label="samples against the reference loop with the same noise")
