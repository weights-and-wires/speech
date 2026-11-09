"""Checks for the GAN vocoder exercise. They run on the CPU, on synthetic signals."""

import numpy as np
import torch

from speechcheck import almost, check, close, fail, finite_difference, has, shape, tone, white_noise

RATE = 16000
RESOLUTIONS = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))


def _numpy_stft_magnitude(x, n_fft, hop, win):
    """A centered, reflect-padded STFT magnitude with a periodic Hann window of length win, in numpy."""
    x = np.pad(x, n_fft // 2, mode="reflect")
    window = np.zeros(n_fft)
    start = (n_fft - win) // 2
    window[start : start + win] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.maximum(np.abs(np.fft.rfft(x[idx] * window, axis=1)).T, 1e-7)


@check("snake gives the hand-worked values and adds a per-channel periodic bump")
def test_snake_values(sub):
    fn = has(sub, "snake")
    x = torch.full((1, 2, 1), 0.5, dtype=torch.float64)
    got = fn(x, torch.tensor([1.0, 5.0], dtype=torch.float64))
    close(got[0, :, 0], [0.5 + np.sin(0.5) ** 2, 0.5 + 0.2 * np.sin(2.5) ** 2], atol=1e-12,
          label="snake(0.5) with alpha = 1 and alpha = 5")
    close(got[0, :, 0], [0.7298, 0.5716], atol=1e-4, label="the same values rounded as in the lesson")
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(3, 4, 7)))
    alpha = torch.tensor(rng.uniform(0.5, 3, 4))
    want = x + torch.sin(alpha[None, :, None] * x) ** 2 / alpha[None, :, None]
    close(fn(x, alpha), want, atol=1e-12, label="snake on a (3, 4, 7) batch with one alpha per channel")


@check("snake's gradient is 1 + sin(2 alpha x) in x, and matches finite differences in alpha")
def test_snake_grad(sub):
    fn = has(sub, "snake")
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(2, 3, 5)), requires_grad=True)
    alpha = torch.tensor(rng.uniform(0.5, 2, 3), requires_grad=True)
    fn(x, alpha).sum().backward()
    close(x.grad, 1 + torch.sin(2 * alpha.detach()[None, :, None] * x.detach()), atol=1e-10,
          label="d snake / dx")
    xn = x.detach().numpy()
    numeric = finite_difference(
        lambda a: float(fn(torch.tensor(xn), torch.tensor(a)).sum()), alpha.detach().numpy().copy(), eps=1e-6
    )
    close(alpha.grad, numeric, rtol=1e-5, atol=1e-6, label="d sum(snake) / d alpha")


@check("period_reshape folds 8,192 samples into columns for periods 2, 3, 5, 7, and 11")
def test_period_shapes(sub):
    fn = has(sub, "period_reshape")
    x = torch.arange(2 * 8192, dtype=torch.float64).reshape(2, 1, 8192)
    for p in (2, 3, 5, 7, 11):
        out = fn(x, p)
        rows = -(-8192 // p)
        shape(out, (2, 1, rows, p), label=f"period {p}")
        close(out[1, 0, 100], x[1, 0, 100 * p : 101 * p], atol=0, label=f"row 100 at period {p}")
        close(out[0, 0, :50, p - 1], x[0, 0, p - 1 : 50 * p : p], atol=0,
              label=f"column {p - 1} at period {p}, which holds every {p}th sample")


@check("period_reshape pads the tail by reflection")
def test_period_pad(sub):
    fn = has(sub, "period_reshape")
    x = torch.arange(10, dtype=torch.float64).reshape(1, 1, 10)
    out = fn(x, 4)
    shape(out, (1, 1, 3, 4), label="10 samples at period 4")
    close(out[0, 0, 2], [8.0, 9.0, 8.0, 7.0], atol=0, label="the last row, padded by reflecting samples 8 and 7")


@check("stft_magnitude matches a numpy STFT with a centered Hann window shorter than the FFT")
def test_stft(sub):
    fn = has(sub, "stft_magnitude")
    x = white_noise(RATE, 0.2, amplitude=0.1, seed=2)
    for n_fft, hop, win in RESOLUTIONS:
        got = fn(torch.tensor(x)[None], n_fft, hop, win)
        want = _numpy_stft_magnitude(x, n_fft, hop, win)
        shape(got, (1,) + want.shape, label=f"the ({n_fft}, {hop}, {win}) magnitude")
        close(got[0], want, rtol=1e-6, atol=1e-9, label=f"the ({n_fft}, {hop}, {win}) magnitude")


@check("spectral_convergence and log_magnitude_loss give the hand-worked 0.2 and 0.1")
def test_spectral_pieces(sub):
    sc = has(sub, "spectral_convergence")
    lm = has(sub, "log_magnitude_loss")
    s = torch.tensor([[6.0, 8.0], [0.0, 0.0]], dtype=torch.float64) + 1e-3  # Frobenius norm about 10
    s = s * (10.0 / torch.linalg.norm(s))
    err = torch.tensor([[1.2, 1.6], [0.0, 0.0]], dtype=torch.float64)  # Frobenius norm 2
    almost(sc(s, s + err), 0.2, 1e-12, label="spectral convergence with |S| = 10 and |error| = 2")
    rng = np.random.default_rng(3)
    mag = torch.tensor(rng.uniform(0.01, 5, size=(2, 9, 6)))
    almost(lm(mag, mag * np.exp(0.1)), 0.1, 1e-12, label="the log-magnitude loss when every bin is off by e^0.1")
    almost(lm(mag, mag / np.exp(0.1)), 0.1, 1e-12, label="the same, off by e^-0.1")


@check("multi_resolution_stft_loss is zero for a perfect copy, scale-free, and rises with noise")
def test_mr_stft(sub):
    fn = has(sub, "multi_resolution_stft_loss")
    x = tone(220, RATE, 0.3, 0.3) + white_noise(RATE, 0.3, 0.05, seed=4)
    xt = torch.tensor(x)[None]
    almost(fn(xt, xt.clone(), RESOLUTIONS), 0.0, 1e-12, label="the loss of a perfect copy")
    values = []
    for level in (0.01, 0.03, 0.1):
        noisy = torch.tensor(x + white_noise(RATE, 0.3, level, seed=5))[None]
        values.append(float(fn(xt, noisy, RESOLUTIONS)))
    if not values[0] < values[1] < values[2]:
        fail(f"the loss should rise as the noise grows from 0.01 to 0.03 to 0.1; got {values}")
    noisy = torch.tensor(x + white_noise(RATE, 0.3, 0.03, seed=5))[None]
    almost(fn(3 * xt, 3 * noisy, RESOLUTIONS), values[1], 1e-9, label="the loss after scaling both signals by 3")
    want = 0.0
    for n_fft, hop, win in RESOLUTIONS:
        s, s_hat = _numpy_stft_magnitude(x, n_fft, hop, win), _numpy_stft_magnitude(noisy[0].numpy(), n_fft, hop, win)
        want += np.linalg.norm(s - s_hat) / np.linalg.norm(s) + np.mean(np.abs(np.log(s) - np.log(s_hat)))
    almost(values[1], want / 3, 1e-9, label="the mean over three resolutions of spectral convergence plus log magnitude")


@check("The LSGAN losses match their formulas, and the generator gradient is 2 (D - 1) / N")
def test_lsgan(sub):
    d_loss = has(sub, "lsgan_discriminator_loss")
    g_loss = has(sub, "lsgan_generator_loss")
    f64 = torch.float64
    real = [torch.tensor([1.0, 0.5], dtype=f64), torch.tensor([[0.0, 2.0, 1.0]], dtype=f64)]
    fake = [torch.tensor([0.0, 0.4], dtype=f64), torch.tensor([[1.0, -1.0, 0.5]], dtype=f64)]
    want_d = (0.0 + 0.25) / 2 + 0.16 / 2 + (1.0 + 1.0 + 0.0) / 3 + (1.0 + 1.0 + 0.25) / 3
    almost(d_loss(real, fake), want_d, 1e-9, label="the discriminator loss")
    want_g = (1.0 + 0.36) / 2 + (0.0 + 4.0 + 0.25) / 3
    almost(g_loss(fake), want_g, 1e-9, label="the generator loss")
    rng = np.random.default_rng(6)
    out = rng.normal(size=5)
    t = torch.tensor(out, requires_grad=True)
    g_loss([t]).backward()
    numeric = finite_difference(lambda v: float(g_loss([torch.tensor(v)])), out.copy())
    close(t.grad, numeric, rtol=1e-5, atol=1e-7, label="d generator loss / d D(G(s)) by finite differences")
    close(t.grad, 2 * (out - 1) / 5, atol=1e-12, label="d generator loss / d D(G(s)) against 2 (D - 1) / N")


@check("feature_matching_loss sums the mean absolute difference over every layer of every discriminator")
def test_feature_matching(sub):
    fn = has(sub, "feature_matching_loss")
    rng = np.random.default_rng(7)
    real = [[torch.tensor(rng.normal(size=(2, 3, 4))), torch.tensor(rng.normal(size=(2, 5)))],
            [torch.tensor(rng.normal(size=(2, 1, 6, 2)))]]
    fake = [[r + torch.tensor(rng.normal(size=r.shape)) for r in layers] for layers in real]
    want = sum(float(torch.mean(torch.abs(r - f))) for rl, fl in zip(real, fake) for r, f in zip(rl, fl))
    almost(fn(real, fake), want, 1e-9, label="feature matching over two discriminators")
    almost(fn(real, real), 0.0, 1e-12, label="feature matching of identical features")


@check("A period-5 sub-discriminator built on your reshape keeps its columns independent")
def test_mpd_columns(sub):
    disc_cls = has(sub, "PeriodDiscriminator")
    torch.manual_seed(8)
    disc = disc_cls(5, channels=(4, 4, 4, 4)).double().eval()
    x = torch.tensor(white_noise(RATE, 0.1, 0.3, seed=9))[None, None]
    bumped = x.clone()
    bumped[0, 0, 803] += 1.0  # sample 803 lives in column 803 % 5 = 3
    with torch.no_grad():
        _, feats = disc(x)
        _, feats_b = disc(bumped)
    for i, (a, b) in enumerate(zip(feats, feats_b)):
        changed = torch.nonzero((a - b).abs().amax(dim=(0, 1, 2)) > 1e-12).flatten().tolist()
        if changed != [3]:
            fail(f"bumping sample 803 should change only column 3 of layer {i}'s features; "
                 f"columns {changed} changed. Does period_reshape put sample t in column t % p?")
