"""Checks for the neural audio codecs exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, shape, tone, white_noise


@check("frame_rate and bitrate reproduce the codec table")
def test_bitrate(sub):
    cases = [
        ("EnCodec 24 kHz at 6 kbps", 24000, (2, 4, 5, 8), 8, 1024, 75.0, 6000.0),
        ("EnCodec 24 kHz at 1.5 kbps", 24000, (2, 4, 5, 8), 2, 1024, 75.0, 1500.0),
        ("Mimi with 8 codebooks", 24000, (4, 5, 6, 8, 2), 8, 2048, 12.5, 1100.0),
        ("DAC 44.1 kHz", 44100, (2, 4, 8, 8), 9, 1024, 44100 / 512, 44100 / 512 * 90),
    ]
    for name, rate, strides, n_cb, size, fps, bps in cases:
        almost(sub.frame_rate(rate, strides), fps, 1e-9, label=f"{name}: frames per second")
        almost(sub.bitrate(fps, n_cb, size), bps, 1e-6, label=f"{name}: bits per second")


@check("sample_quantizer_counts keeps every stage with probability 1 - p and draws uniformly otherwise")
def test_dropout(sub):
    gen = torch.Generator().manual_seed(0)
    counts = sub.sample_quantizer_counts(40000, 8, 0.5, gen)
    shape(counts, (40000,), label="counts")
    c = np.asarray(counts)
    if c.min() < 1 or c.max() > 8:
        fail(f"counts range over [{c.min()}, {c.max()}], expected [1, 8]")
    freq = np.bincount(c, minlength=9)[1:] / len(c)
    want = np.full(8, 0.5 / 8)
    want[-1] += 0.5
    close(freq, want, atol=0.01, label="frequency of each count 1..8 with p = 0.5")
    none = np.asarray(sub.sample_quantizer_counts(1000, 8, 0.0, torch.Generator().manual_seed(1)))
    equal(bool(np.all(none == 8)), True, label="p = 0 keeps all 8 stages for every example")


def _hand_codebooks():
    c1 = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    c2 = [[0.1, 0.1], [-0.1, -0.2], [0.2, -0.2], [-0.2, 0.2]]
    return torch.tensor([c1, c2], dtype=torch.float64)


@check("rvq_forward reproduces the two-stage example from the residual quantization chapter")
def test_rvq_hand(sub):
    z = torch.tensor([0.9, -0.3], dtype=torch.float64).reshape(1, 2, 1)
    z_q, codes, cb_loss, commit = sub.rvq_forward(z, _hand_codebooks(), 2)
    shape(z_q, (1, 2, 1), label="z_q")
    shape(codes, (1, 2, 1), label="codes, (batch, stages, frames)")
    equal(codes.reshape(-1).tolist(), [0, 1], label="codes")
    close(z_q.detach().reshape(-1).numpy(), [0.9, -0.2], label="quantized latent")
    almost(float(cb_loss), 0.11, 1e-9, label="codebook loss summed over stages")
    almost(float(commit), 0.11, 1e-9, label="commitment loss summed over stages")


@check("rvq_forward's gradient passes straight through to z, and each loss reaches only its side")
def test_rvq_grad(sub):
    torch.manual_seed(0)
    z = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    codebooks = (torch.randn(3, 16, 4, dtype=torch.float64) * 0.7).requires_grad_(True)
    z_q, codes, cb_loss, commit = sub.rvq_forward(z, codebooks, 3)
    w = torch.randn(3, 4, 5, dtype=torch.float64)
    (z_q * w).sum().backward(retain_graph=True)
    close(z.grad.numpy(), w.numpy(), label="d(sum(w * z_q)) / dz, which the straight-through estimator makes w")
    if codebooks.grad is not None and float(codebooks.grad.abs().max()) > 0:
        fail("the quantized output sent a gradient to the codebooks; only the codebook loss may")
    z.grad = None
    cb_loss.backward(retain_graph=True)
    if z.grad is not None and float(z.grad.abs().max()) > 0:
        fail("the codebook loss sent a gradient to z")
    if codebooks.grad is None or float(codebooks.grad.abs().max()) == 0:
        fail("the codebook loss sent no gradient to the codebooks")
    codebooks.grad = None
    commit.backward()
    if codebooks.grad is not None and float(codebooks.grad.abs().max()) > 0:
        fail("the commitment loss sent a gradient to the codebooks")
    if z.grad is None or float(z.grad.abs().max()) == 0:
        fail("the commitment loss sent no gradient to z")


@check("rvq_forward uses only the first n_q[b] stages for example b, and marks the rest -1")
def test_rvq_counts(sub):
    torch.manual_seed(1)
    z = torch.randn(3, 4, 6, dtype=torch.float64)
    codebooks = torch.randn(4, 8, 4, dtype=torch.float64)
    n_q = torch.tensor([1, 4, 2])
    z_q, codes, _, _ = sub.rvq_forward(z, codebooks, n_q)
    full_q, full_codes, _, _ = sub.rvq_forward(z, codebooks, 4)
    for b, k in enumerate(n_q.tolist()):
        equal(codes[b, :k].tolist(), full_codes[b, :k].tolist(), label=f"example {b}'s first {k} stages")
        if k < 4 and not bool(torch.all(codes[b, k:] == -1)):
            fail(f"example {b} uses {k} stages, so codes[{b}, {k}:] must be -1, got {codes[b, k:].tolist()}")
        want = sum(codebooks[i][full_codes[b, i]] for i in range(k)).T
        close(z_q[b].detach().numpy(), want.numpy(), atol=1e-10, label=f"example {b}'s quantized latent")


@check("ToyCodec turns L samples into L / hop frames of codes and back into L samples")
def test_toy_shapes(sub):
    torch.manual_seed(0)
    model = sub.ToyCodec(strides=(2, 4, 4), channels=8, dim=6, n_q=3, codebook_size=16)
    equal(model.hop, 32, label="hop, the product of the strides")
    shape(model.codebooks.detach(), (3, 16, 6), label="codebooks")
    x = torch.randn(2, 1, 32 * 20)
    with torch.no_grad():
        z = model.encode(x)
        x_hat, codes, cb_loss, commit = model(x)
    shape(z, (2, 6, 20), label="encode(x)")
    shape(x_hat, (2, 1, 640), label="the reconstruction")
    shape(codes, (2, 3, 20), label="codes")
    if not (0 <= int(codes.min()) and int(codes.max()) < 16):
        fail(f"codes range over [{int(codes.min())}, {int(codes.max())}], expected [0, 15]")


@check("ToyCodec decodes the same audio from its codes, and trains its encoder through the quantizer")
def test_toy_consistency(sub):
    torch.manual_seed(2)
    model = sub.ToyCodec(strides=(2, 4), channels=8, dim=4, n_q=3, codebook_size=8)
    x = torch.randn(2, 1, 8 * 12)
    x_hat, codes, cb_loss, commit = model(x, 2)
    if not bool(torch.all(codes[:, 2] == -1)):
        fail("with n_q = 2, the third stage's codes must all be -1")
    again = model.codes_to_audio(codes[:, :2])
    close(again.detach().numpy(), x_hat.detach().numpy(), atol=1e-5, label="codes_to_audio(codes) against forward(x, 2)")
    loss = (x_hat - x).pow(2).mean() + cb_loss + 0.25 * commit
    loss.backward()
    first = next(model.encoder.parameters()) if hasattr(model, "encoder") else next(model.parameters())
    if first.grad is None or float(first.grad.abs().max()) == 0:
        fail("the encoder's first layer got no gradient; the straight-through estimator must connect it to the loss")
    if model.codebooks.grad is None or float(model.codebooks.grad.abs().max()) == 0:
        fail("the codebooks got no gradient from the codebook loss")


@check("si_sdr matches the lesson's example and ignores the estimate's scale")
def test_si_sdr(sub):
    s = np.array([1.0, -1.0, 1.0, -1.0])
    est = 2 * s + np.array([0.1, 0.1, -0.1, -0.1])
    almost(sub.si_sdr(est, s), 10 * np.log10(16 / 0.04), 1e-9, label="SI-SDR of the lesson's example (26.02 dB)")
    ref = tone(440.0, 16000, 0.5)
    rng = np.random.default_rng(0)
    noisy = ref + 0.05 * rng.standard_normal(ref.size)
    base = sub.si_sdr(noisy, ref)
    almost(sub.si_sdr(0.3 * noisy, ref), base, 1e-9, label="SI-SDR after scaling the estimate by 0.3")
    almost(sub.si_sdr(noisy + 0.2, ref), base, 1e-6, label="SI-SDR after adding a constant offset")
    alpha = noisy @ ref / (ref @ ref)
    want = 10 * np.log10(np.sum((alpha * ref) ** 2) / np.sum((noisy - alpha * ref) ** 2))
    almost(base, want, 1e-2, label="SI-SDR of a tone in white noise")


def _reference_log_mel(x, rate, n_fft, n_mels):
    hop = n_fft // 4
    n_frames = 1 + (len(x) - n_fft) // hop
    frames = np.stack([x[i * hop : i * hop + n_fft] for i in range(n_frames)])
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    mag = np.abs(np.fft.rfft(frames * window, axis=1)).T
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = imel(np.linspace(0, mel(rate / 2), n_mels + 2))
    freqs = np.arange(n_fft // 2 + 1) * rate / n_fft
    fb = np.zeros((n_mels, freqs.size))
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return np.log10(np.maximum(fb @ mag, 1e-5))


@check("mel_distance is zero for identical signals and matches a reference multi-scale log-mel L1")
def test_mel_distance(sub):
    rate = 24000
    x = tone(300.0, rate, 0.5) + white_noise(rate, 0.5, 0.01, seed=0)
    y = tone(310.0, rate, 0.5, amplitude=0.4) + white_noise(rate, 0.5, 0.02, seed=1)
    almost(sub.mel_distance(x, x, rate), 0.0, 1e-12, label="distance from a signal to itself")
    got = sub.mel_distance(x, y, rate)
    almost(sub.mel_distance(y, x, rate), got, 1e-12, label="the distance is symmetric")
    scales = ((256, 20), (512, 40), (1024, 80), (2048, 160))
    want = np.mean([np.mean(np.abs(_reference_log_mel(x, rate, n, m) - _reference_log_mel(y, rate, n, m))) for n, m in scales])
    almost(got, want, 1e-6, label="mel distance of two noisy tones")
    near = sub.mel_distance(x, x + white_noise(rate, 0.5, 0.001, seed=2), rate)
    if not near < got:
        fail(f"a slightly noisier copy scored {near:.3f}, no closer than a different tone ({got:.3f})")
