"""The losses, discriminator reshape, and activation of a HiFi-GAN-style vocoder: the worked solution."""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from speechlab import asset_path, has_asset, play, read_audio, show_spectrogram


def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.reshape(1, -1, *([1] * (x.dim() - 2)))
    return x + torch.sin(a * x) ** 2 / a


def period_reshape(x: torch.Tensor, period: int) -> torch.Tensor:
    batch, channels, length = x.shape
    extra = (-length) % period
    if extra:
        x = F.pad(x, (0, extra), mode="reflect")
    return x.reshape(batch, channels, (length + extra) // period, period)


def stft_magnitude(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, periodic=True, dtype=x.dtype, device=x.device)
    spec = torch.stft(x, n_fft, hop, win, window=window, center=True, pad_mode="reflect", return_complex=True)
    return spec.abs().clamp(min=1e-7)


def spectral_convergence(mag_true: torch.Tensor, mag_pred: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(mag_true - mag_pred) / torch.linalg.norm(mag_true)


def log_magnitude_loss(mag_true: torch.Tensor, mag_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(torch.log(mag_true) - torch.log(mag_pred)))


def multi_resolution_stft_loss(
    x: torch.Tensor, x_hat: torch.Tensor,
    resolutions: tuple = ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240)),
) -> torch.Tensor:
    total = 0.0
    for n_fft, hop, win in resolutions:
        s, s_hat = stft_magnitude(x, n_fft, hop, win), stft_magnitude(x_hat, n_fft, hop, win)
        total = total + spectral_convergence(s, s_hat) + log_magnitude_loss(s, s_hat)
    return total / len(resolutions)


def lsgan_discriminator_loss(real_outputs: list, fake_outputs: list) -> torch.Tensor:
    return sum(torch.mean((r - 1) ** 2) + torch.mean(f**2) for r, f in zip(real_outputs, fake_outputs))


def lsgan_generator_loss(fake_outputs: list) -> torch.Tensor:
    return sum(torch.mean((f - 1) ** 2) for f in fake_outputs)


def feature_matching_loss(real_features: list, fake_features: list) -> torch.Tensor:
    return sum(
        torch.mean(torch.abs(r - f))
        for real_layers, fake_layers in zip(real_features, fake_features)
        for r, f in zip(real_layers, fake_layers)
    )


# Provided: the networks and the mel front end ----------------------------------

RATE = 16000
N_FFT, HOP, N_MELS = 1024, 256, 80


def mel_filterbank(sample_rate: int = RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = 0.0, fmax: float = 8000.0) -> np.ndarray:
    """Slaney-style triangular mel filters, shape (n_mels, n_fft // 2 + 1), as in the mel scale chapter."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = f / (200.0 / 3)
        log_region = f >= 1000.0
        return np.where(log_region, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / (np.log(6.4) / 27.0), mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = m * (200.0 / 3)
        return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)

    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    lower = (freqs[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    upper = (edges[2:, None] - freqs[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    return weights * (2.0 / (edges[2:] - edges[:-2]))[:, None]


class LogMel(nn.Module):
    """HiFi-GAN's front end: frames of 1024 every 256 samples, 80 bands, natural log floored at 1e-5.

    The signal is reflect-padded by (n_fft - hop) / 2 on each side without centering, so a clip of
    T samples, T a multiple of 256, gives exactly T / 256 frames.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("fb", torch.tensor(mel_filterbank(), dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(N_FFT, periodic=True))

    def forward(self, x):  # (B, T) -> (B, 80, T / 256)
        pad = (N_FFT - HOP) // 2
        x = F.pad(x[:, None], (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(x, N_FFT, HOP, N_FFT, window=self.window, center=False, return_complex=True).abs()
        return torch.log(torch.clamp(self.fb @ spec, min=1e-5))


class ResBlock(nn.Module):
    """HiFi-GAN's residual block, with Snake in place of leaky ReLU: pairs of (dilation d, dilation 1) convs."""

    def __init__(self, channels, kernel, dilations):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.convs1 = nn.ModuleList(wn(nn.Conv1d(channels, channels, kernel, dilation=d, padding=d * (kernel - 1) // 2))
                                    for d in dilations)
        self.convs2 = nn.ModuleList(wn(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2))
                                    for _ in dilations)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(channels)) for _ in dilations)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(channels)) for _ in dilations)

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):
            x = x + c2(snake(c1(snake(x, a1)), a2))
        return x


class Generator(nn.Module):
    """A HiFi-GAN generator: 80 mel bands in, 256 samples out per frame (upsampling 8 x 8 x 2 x 2)."""

    def __init__(self, channels=128, kernels=(3, 7, 11), dilations=(1, 3, 5)):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.pre = wn(nn.Conv1d(N_MELS, channels, 7, padding=3))
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        self.alphas = nn.ParameterList()
        ch = channels
        for rate in (8, 8, 2, 2):
            self.alphas.append(nn.Parameter(torch.ones(ch)))
            self.ups.append(wn(nn.ConvTranspose1d(ch, ch // 2, 2 * rate, rate, padding=rate // 2)))
            ch //= 2
            self.mrfs.append(nn.ModuleList(ResBlock(ch, k, dilations) for k in kernels))
        self.alpha_post = nn.Parameter(torch.ones(ch))
        self.post = wn(nn.Conv1d(ch, 1, 7, padding=3))

    def forward(self, mel):  # (B, 80, N) -> (B, 1, 256 N)
        x = self.pre(mel)
        for alpha, up, mrf in zip(self.alphas, self.ups, self.mrfs):
            x = up(snake(x, alpha))
            x = sum(block(x) for block in mrf) / len(mrf)  # multi-receptive-field fusion
        return torch.tanh(self.post(snake(x, self.alpha_post)))


class PeriodDiscriminator(nn.Module):
    """One MPD sub-discriminator: reshape by the period, then 2-D convolutions of width 1 across columns."""

    def __init__(self, period, channels=(32, 128, 512, 1024)):
        super().__init__()
        wn = nn.utils.parametrizations.weight_norm
        self.period = period
        sizes = (1,) + tuple(channels)
        self.convs = nn.ModuleList(
            wn(nn.Conv2d(sizes[i], sizes[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(len(channels))
        )
        self.last = wn(nn.Conv2d(sizes[-1], sizes[-1], (5, 1), 1, padding=(2, 0)))
        self.out = wn(nn.Conv2d(sizes[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):  # (B, 1, T) -> (score map, list of feature maps)
        h = period_reshape(x, self.period)
        features = []
        for conv in list(self.convs) + [self.last]:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        h = self.out(h)
        features.append(h)
        return h.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, channels=(32, 128, 512, 1024), periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.subs = nn.ModuleList(PeriodDiscriminator(p, channels) for p in periods)

    def forward(self, x):
        outs = [d(x) for d in self.subs]
        return [o[0] for o in outs], [o[1] for o in outs]


def griffin_lim_from_mel(log_mel: np.ndarray, iterations: int = 60) -> np.ndarray:
    """The Griffin-Lim algorithm, after a least-squares inverse of the mel filterbank."""
    fb = mel_filterbank()
    magnitude = np.maximum(np.linalg.pinv(fb) @ np.exp(log_mel), 0.0)
    mag = torch.tensor(magnitude)
    window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
    length = (mag.shape[1] - 1) * HOP
    phase = torch.exp(2j * math.pi * torch.rand(mag.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
    for _ in range(iterations):
        wave = torch.istft(mag * phase, N_FFT, HOP, window=window, length=length)
        spec = torch.stft(wave, N_FFT, HOP, window=window, return_complex=True)
        phase = spec / spec.abs().clamp(min=1e-8)
    wave = torch.istft(mag * phase, N_FFT, HOP, window=window, length=length).numpy()
    # Centered frame i sits at sample 256 i; HiFi-GAN framing puts it at 256 i + 128.
    return np.concatenate([np.zeros(HOP // 2), wave, np.zeros(HOP // 2)])


def _load_speaker(seconds: float, holdout_seconds: float):
    """A few minutes of LibriSpeech speaker 1272 for training, and one utterance from another chapter to test."""
    root = Path(asset_path("librispeech-dev-clean")) / "LibriSpeech" / "dev-clean" / "1272"
    train, total = [], 0.0
    for path in sorted(root.glob("12810*/*.flac")) + sorted(root.glob("135031/*.flac")):
        audio, rate = read_audio(path)
        train.append(audio.astype(np.float32))
        total += len(audio) / rate
        if total >= seconds:
            break
    held, _ = read_audio(sorted(root.glob("141231/*.flac"))[1])
    return np.concatenate(train), held[: int(holdout_seconds * RATE)].astype(np.float32)


def _synthetic(seconds: float, rng) -> np.ndarray:
    """A stand-in when the recordings are missing: vowel-like buzzes that glide in pitch."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 140 + 40 * np.sin(2 * np.pi * 0.7 * t) + 20 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    formant = 600 + 400 * np.sin(2 * np.pi * 0.37 * t)
    x = sum(np.exp(-((k * f0 - formant) / 300) ** 2) * np.sin(k * phase) / k for k in range(1, 30))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.9 * t) ** 2
    return (0.4 * envelope * x / np.abs(x).max()).astype(np.float32)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    if device == "cuda":
        cfg = dict(seconds=240, channels=128, kernels=(3, 7, 11), mpd=(32, 64, 128, 256), batch=16,
                   segment=8192, steps=5000, warmup=1000, limit=420)
    else:
        print("No GPU here, so this is the small version: a tiny generator, about a minute of audio, and 35 s "
              "of training. Its output will still be noisy; the GPU run trains a model 20 times larger for minutes.")
        cfg = dict(seconds=40, channels=32, kernels=(3, 7), mpd=(8, 16, 32, 32), batch=4,
                   segment=4096, steps=700, warmup=350, limit=35)
    if has_asset("librispeech-dev-clean"):
        train_audio, held_out = _load_speaker(cfg["seconds"], 3.0)
        print(f"Training on {len(train_audio) / RATE:.0f} s of LibriSpeech speaker 1272; "
              f"testing on {len(held_out) / RATE:.1f} s from a chapter it never hears.")
    else:
        print("The librispeech-dev-clean asset is missing, so this run trains on synthetic buzzes instead.")
        train_audio, held_out = _synthetic(cfg["seconds"], rng), _synthetic(3.0, rng)[::-1].copy()
    held_out = held_out[: len(held_out) // HOP * HOP]

    to_mel = LogMel().to(device)
    gen = Generator(cfg["channels"], cfg["kernels"], (1, 3, 5)).to(device)
    mpd = MultiPeriodDiscriminator(cfg["mpd"]).to(device)
    print(f"Generator: {sum(p.numel() for p in gen.parameters()) / 1e6:.2f}M parameters; "
          f"discriminator: {sum(p.numel() for p in mpd.parameters()) / 1e6:.2f}M.")
    # HiFi-GAN uses 2e-4 over 2.5M steps; a run of minutes needs a larger step.
    opt_g = torch.optim.AdamW(gen.parameters(), 1e-3, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(mpd.parameters(), 1e-3, betas=(0.8, 0.99))
    audio = torch.from_numpy(train_audio).to(device)

    start = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        offsets = rng.integers(0, len(train_audio) - cfg["segment"], cfg["batch"]) // HOP * HOP
        y = torch.stack([audio[o : o + cfg["segment"]] for o in offsets])  # (B, T)
        mel = to_mel(y)
        y_hat = gen(mel)[:, 0]
        adversarial = step > cfg["warmup"]

        if adversarial:  # discriminator step: G(s) is detached
            real, _ = mpd(y[:, None])
            fake, _ = mpd(y_hat.detach()[:, None])
            loss_d = lsgan_discriminator_loss(real, fake)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        mel_l1 = F.l1_loss(to_mel(y_hat), mel)
        stft = multi_resolution_stft_loss(y, y_hat)
        loss_g = 45 * mel_l1 + 2.5 * stft
        if adversarial:  # generator step: gradients flow through D into G
            _, real_feats = mpd(y[:, None])
            fake, fake_feats = mpd(y_hat[:, None])
            loss_g = loss_g + lsgan_generator_loss(fake) + 2 * feature_matching_loss(real_feats, fake_feats)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if step % max(1, cfg["steps"] // 10) == 0:
            extra = f", D {loss_d.item():.3f}" if adversarial else " (reconstruction only)"
            print(f"step {step:5d}: mel L1 {mel_l1.item():.3f}, MR-STFT {stft.item():.3f}{extra}")
        if time.perf_counter() - start > cfg["limit"]:
            print(f"Stopping at step {step}: the time budget of {cfg['limit']} s is spent.")
            break
    print(f"Trained for {time.perf_counter() - start:.0f} s.")

    gen.eval()
    with torch.no_grad():
        clip = torch.from_numpy(held_out).to(device)[None]
        mel = to_mel(clip)
        vocoded = gen(mel)[0, 0]
        mel_gan = F.l1_loss(to_mel(vocoded[None]), mel).item()
        gl = torch.from_numpy(griffin_lim_from_mel(mel[0].double().cpu().numpy())).float().to(device)
        mel_gl = F.l1_loss(to_mel(gl[None]), mel).item()
        stft_gan = multi_resolution_stft_loss(clip, vocoded[None]).item()
        stft_gl = multi_resolution_stft_loss(clip, gl[None]).item()
    print(f"Held-out mel L1 (natural-log mel, lower is better): your GAN vocoder {mel_gan:.3f}, "
          f"Griffin-Lim {mel_gl:.3f}.")
    print(f"Held-out MR-STFT loss: your GAN vocoder {stft_gan:.3f}, Griffin-Lim {stft_gl:.3f}. "
          "Listen too: neither number hears phase.")
    play(held_out, RATE, label="The original held-out clip")
    play(vocoded.cpu().numpy(), RATE, label="Your GAN vocoder, from the clip's 80-band log-mel")
    play(gl.cpu().numpy(), RATE, label="Griffin-Lim, from the same log-mel")
    show_spectrogram(mel[0].cpu(), RATE, HOP, label="The held-out clip's log-mel (the vocoder's input)",
                     db=False, y_axis="mel", fmax=8000, value_label="log magnitude")
    show_spectrogram(to_mel(vocoded[None])[0].cpu(), RATE, HOP, label="The log-mel of your vocoder's output",
                     db=False, y_axis="mel", fmax=8000, value_label="log magnitude")
