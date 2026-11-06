"""Diffusion for audio: the worked solution.

Convention, as in the diffusion chapters: data at t = 0, noise at t = T,
schedules of length T + 1 with alpha_bar[0] = 1, steps t = 1..T. The DiffWave
repository counts steps from 0, so its fractional steps are one less than
the ones here.
"""

import glob
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechlab import asset_path, has_asset, play, read_audio, show_plot, show_spectrogram


# The diffusion-step embedding -----------------------------------------------------


def diffusion_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """DiffWave's sinusoidal step embedding: [sin(t 10^(4j/(h-1))), cos(...)], h = dim / 2."""
    half = dim // 2
    freqs = 10.0 ** (4.0 * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1))
    angle = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


# The residual block ---------------------------------------------------------------


class ResidualBlock(nn.Module):
    """One DiffWave layer: a dilated, gated convolution conditioned on the step and the mel."""

    def __init__(self, channels: int, dilation: int, n_mels: int, emb_dim: int = 512):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(emb_dim, channels)
        self.conditioner_projection = nn.Conv1d(n_mels, 2 * channels, 1)
        self.output_projection = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x, cond, emb):
        y = x + self.diffusion_projection(emb)[:, :, None]
        y = self.dilated_conv(y) + self.conditioner_projection(cond)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = torch.chunk(self.output_projection(y), 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


# Fast sampling ---------------------------------------------------------------------


def diffwave_alignment(train_betas: np.ndarray, infer_betas: np.ndarray) -> np.ndarray:
    """The fractional training step whose sqrt(alpha_bar) matches each fast-sampling step's."""
    train_ab = np.cumprod(1 - np.asarray(train_betas))      # train_ab[n] is alpha_bar at step n + 1
    infer_ab = np.cumprod(1 - np.asarray(infer_betas))
    steps = []
    for g in infer_ab:
        for n in range(len(train_ab) - 1):
            if train_ab[n + 1] <= g <= train_ab[n]:
                frac = (math.sqrt(train_ab[n]) - math.sqrt(g)) / (math.sqrt(train_ab[n]) - math.sqrt(train_ab[n + 1]))
                steps.append(n + 1 + frac)
                break
        else:
            raise ValueError(f"alpha_bar {g:.4f} lies outside the training schedule")
    return np.array(steps)


def fast_sampling_coefficients(infer_betas: np.ndarray):
    """(c1, c2, sigma) for each fast step s: x <- c1 (x - c2 eps) + sigma z."""
    eta = np.asarray(infer_betas, dtype=float)
    gamma = 1 - eta
    gamma_bar = np.cumprod(gamma)
    c1 = 1 / np.sqrt(gamma)
    c2 = eta / np.sqrt(1 - gamma_bar)
    sigma = np.zeros_like(eta)
    sigma[1:] = np.sqrt((1 - gamma_bar[:-1]) / (1 - gamma_bar[1:]) * eta[1:])
    return c1, c2, sigma


def diffwave_loss(model, audio: torch.Tensor, mel: torch.Tensor, alpha_bar: torch.Tensor, generator) -> torch.Tensor:
    """The noise-prediction loss for a mel-conditioned waveform model. audio (B, L), mel (B, n_mels, frames)."""
    T = alpha_bar.shape[0] - 1
    t = torch.randint(1, T + 1, (audio.shape[0],), generator=generator).to(audio.device)
    eps = torch.randn(audio.shape, generator=generator).to(audio.device)
    ab = alpha_bar[t].float()[:, None]
    x_t = ab.sqrt() * audio + (1 - ab).sqrt() * eps
    return ((model(x_t, mel, t.float()) - eps) ** 2).mean()


@torch.no_grad()
def fast_sample(model, mel: torch.Tensor, length: int, infer_betas, t_align, generator) -> torch.Tensor:
    """DiffWave's fast sampler: len(infer_betas) steps from noise, each at its aligned step."""
    c1, c2, sigma = fast_sampling_coefficients(infer_betas)
    x = torch.randn(mel.shape[0], length, generator=generator).to(mel.device)
    for s in range(len(infer_betas) - 1, -1, -1):
        t = torch.full((mel.shape[0],), float(t_align[s]), device=mel.device)
        x = c1[s] * (x - c2[s] * model(x, mel, t))
        if s > 0:
            x = x + sigma[s] * torch.randn(x.shape, generator=generator).to(x.device)
        x = torch.clamp(x, -1.0, 1.0)
    return x


# Grad-TTS -----------------------------------------------------------------------


def grad_tts_stats(x0, mu, t: float, beta0: float = 0.05, beta1: float = 20.0):
    """Mean and variance of X_t given X_0 under Grad-TTS's forward SDE with Sigma = I."""
    integral = beta0 * t + 0.5 * (beta1 - beta0) * t * t
    keep = math.exp(-0.5 * integral)
    return keep * x0 + (1 - keep) * mu, 1 - math.exp(-integral)


# Provided: the rest of a small DiffWave, and a log-mel front end -------------------


class TinyDiffWave(nn.Module):
    """Input projection, a stack of your residual blocks, a mel upsampler, and skip outputs."""

    def __init__(self, n_mels: int, hop: int, channels: int = 64, layers: int = 20, cycle: int = 10):
        super().__init__()
        self.input_projection = nn.Conv1d(1, channels, 1)
        self.embed = nn.Sequential(nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU())
        self.upsample = nn.ConvTranspose1d(n_mels, n_mels, 2 * hop, stride=hop, padding=hop // 2)
        self.blocks = nn.ModuleList(ResidualBlock(channels, 2 ** (i % cycle), n_mels) for i in range(layers))
        self.skip_projection = nn.Conv1d(channels, channels, 1)
        self.output_projection = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, x, mel, t):
        h = F.relu(self.input_projection(x[:, None]))
        emb = self.embed(diffusion_embedding(t))
        cond = F.leaky_relu(self.upsample(mel), 0.4)[:, :, : x.shape[1]]
        skips = 0
        for block in self.blocks:
            h, skip = block(h, cond, emb)
            skips = skips + skip
        h = F.relu(self.skip_projection(skips / math.sqrt(len(self.blocks))))
        return self.output_projection(h)[:, 0]


def log_mel(audio: np.ndarray, rate: int, n_fft: int, hop: int, n_mels: int) -> np.ndarray:
    """DiffWave-style normalized log-mel, in [0, 1], shape (n_mels, len(audio) // hop)."""
    pad = np.pad(audio, (n_fft // 2, n_fft // 2), mode="reflect")
    frames = len(audio) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    mag = np.abs(np.fft.rfft(pad[index] * window, axis=1)).T
    mel_of = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(0, mel_of(rate / 2), n_mels + 2) / 2595) - 1)
    bins = np.arange(n_fft // 2 + 1) * rate / n_fft
    fb = np.maximum(0, np.minimum((bins - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None]),
                                  (edges[2:, None] - bins) / (edges[2:, None] - edges[1:-1, None])))
    db = 20 * np.log10(np.maximum(fb @ mag, 1e-5)) - 20
    return np.clip((db + 100) / 100, 0, 1).astype(np.float32)


# Run ------------------------------------------------------------------------


BASE_FAST = np.array([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])


def _fsdd_clips(length, speaker="george"):
    root = asset_path("fsdd") / "free-spoken-digit-dataset-1.0.10" / "recordings"
    clips = []
    for path in sorted(glob.glob(str(root / f"*_{speaker}_*.wav"))):
        x, rate = read_audio(path)
        x = x[:length]
        x = np.pad(x, (0, length - len(x)))
        clips.append(0.95 * x / (np.max(np.abs(x)) + 1e-9))
    return np.stack(clips).astype(np.float32), rate


def _synthetic_clips(n, length, rate, rng):
    t = np.arange(length) / rate
    out = []
    for _ in range(n):
        f0 = rng.uniform(100, 300) * (1 + 0.2 * t / t[-1])
        phase = 2 * np.pi * np.cumsum(f0) / rate
        x = sum(np.sin(h * phase) / h for h in range(1, 6)) * np.hanning(length)
        out.append(0.5 * x / np.max(np.abs(x)))
    return np.stack(out).astype(np.float32)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    small = device == "cpu"
    if small:
        torch.backends.nnpack.set_flags(False)  # NNPACK is slow for 1-D convolutions on some CPUs
        print("No GPU: running the small version (8 layers of 16 channels, 40 seconds of training) "
              "so it finishes in under a minute. Expect noise shaped like the digit, not a clean digit.")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)

    rate, n_fft, hop, n_mels, length = 8000, 256, 64, 40, 4096
    if has_asset("fsdd"):
        clips, rate = _fsdd_clips(length)
        print(f"{len(clips)} fsdd recordings by one speaker, {length / rate:.3f} s each at {rate} Hz")
    else:
        clips = _synthetic_clips(256, length, rate, rng)
        print("The fsdd asset is not here, so training on synthetic gliding tones instead.")
    mels = np.stack([log_mel(c, rate, n_fft, hop, n_mels) for c in clips])
    held_out, train = 0, np.arange(1, len(clips))
    audio_t, mel_t = torch.from_numpy(clips).to(device), torch.from_numpy(mels).to(device)

    T = 50
    train_betas = np.linspace(1e-4, 0.05, T)
    alpha_bar = torch.cumprod(1 - torch.cat([torch.zeros(1), torch.tensor(train_betas)]), 0).to(device)
    t_align = diffwave_alignment(train_betas, BASE_FAST)
    print("fast schedule aligned to training steps:", np.round(t_align, 3))

    channels, layers, budget, batch = (16, 8, 40.0, 8) if small else (64, 20, 420.0, 32)
    model = TinyDiffWave(n_mels, hop, channels=channels, layers=layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
          f"{layers} layers of {channels} channels")
    losses, start, step = [], time.time(), 0
    while time.time() - start < budget:
        idx = torch.from_numpy(rng.choice(train, batch))
        loss = diffwave_loss(model, audio_t[idx], mel_t[idx], alpha_bar, torch.Generator().manual_seed(step))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        step += 1
    print(f"trained {step} steps in {time.time() - start:.0f} s: loss {np.mean(losses[:20]):.3f} -> {np.mean(losses[-50:]):.3f}")
    show_plot(np.array(losses), label="DiffWave training loss", xlabel="Step", ylabel="MSE on the noise", y_scale="log")

    model.eval()
    mel_in = mel_t[held_out : held_out + 1]
    gen_fast = fast_sample(model, mel_in, length, BASE_FAST, t_align, gen)[0].cpu().numpy()
    gen_full = fast_sample(model, mel_in, length, train_betas, np.arange(1, T + 1, dtype=float), gen)[0].cpu().numpy()
    def envelope(x):
        return np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))

    target = envelope(clips[held_out])
    corr = np.corrcoef(envelope(gen_fast), target)[0, 1]
    quiet = target < 0.1 * target.max()
    print(f"6-step sample: loudness envelope correlates {corr:.2f} with the original's; "
          f"RMS {envelope(gen_fast)[quiet].mean():.3f} where the original is silent (hiss)")
    play(clips[held_out], rate, label="A held-out recording")
    play(gen_fast, rate, label="Your DiffWave from its mel, 6 fast steps")
    play(gen_full, rate, label=f"Your DiffWave from its mel, all {T} steps")
    show_spectrogram(np.abs(np.fft.rfft(gen_fast[: length // 128 * 128].reshape(-1, 128) * np.hanning(128), axis=1)).T,
                     rate, 128, label="Spectrogram of the 6-step sample")
