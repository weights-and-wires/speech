"""Diffusion for audio.

Fill in the functions and the ResidualBlock class below, then press Run to
train a small DiffWave-style vocoder on one speaker's spoken digits from
fsdd and hear it turn a mel spectrogram back into sound in 6 steps.

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
    """DiffWave's sinusoidal step embedding, shape (B, dim).

    t is a float tensor of shape (B,) that may hold fractional steps. With
    h = dim // 2 and frequencies 10 ** (4 j / (h - 1)) for j = 0..h-1, return
    [sin(t * freq_j) for all j] followed by [cos(t * freq_j) for all j].
    Build the frequencies on t's device.
    """
    raise NotImplementedError("diffusion_embedding")


# The residual block ---------------------------------------------------------------


class ResidualBlock(nn.Module):
    """One DiffWave layer: a dilated, gated convolution conditioned on the step and the mel.

    Build four layers in __init__, with these exact names:
        dilated_conv: nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        diffusion_projection: nn.Linear(emb_dim, channels)
        conditioner_projection: nn.Conv1d(n_mels, 2 * channels, 1)
        output_projection: nn.Conv1d(channels, 2 * channels, 1)

    forward(x, cond, emb), with x (B, C, L), cond (B, n_mels, L), and emb
    (B, emb_dim), returns (out, skip), each (B, C, L). It computes the
    chapter's layer: the step embedding's projection joins x at every time step
    before the dilated convolution, and the mel projection joins after it. Of
    those 2C channels, the first C are the sigmoid gate and the last C the tanh
    filter. Of output_projection's 2C channels, the first C are the residual
    and the last C the skip, and out is (x + residual) / sqrt(2).
    """

    def __init__(self, channels: int, dilation: int, n_mels: int, emb_dim: int = 512):
        super().__init__()
        raise NotImplementedError("ResidualBlock.__init__")

    def forward(self, x, cond, emb):
        raise NotImplementedError("ResidualBlock.forward")


# Fast sampling ---------------------------------------------------------------------


def diffwave_alignment(train_betas: np.ndarray, infer_betas: np.ndarray) -> np.ndarray:
    """The fractional training step aligned to each fast-sampling step, counted from 1.

    train_betas has the T training betas (no leading 0), infer_betas the S
    fast ones. With alpha_bar_t = prod_{i <= t} (1 - beta_i) over training
    steps and gamma_bar_s the same product over the fast schedule, find the
    t with alpha_bar_{t+1} <= gamma_bar_s <= alpha_bar_t and return
        t + (sqrt(alpha_bar_t) - sqrt(gamma_bar_s)) / (sqrt(alpha_bar_t) - sqrt(alpha_bar_{t+1})).
    Returns an array of S floats between 1 and T.
    """
    raise NotImplementedError("diffwave_alignment")


def fast_sampling_coefficients(infer_betas: np.ndarray):
    """(c1, c2, sigma), each of shape (S,), for the fast sampler's update x <- c1 (x - c2 eps) + sigma z.

    With eta_s = infer_betas[s], gamma_s = 1 - eta_s, and gamma_bar_s their
    cumulative product: c1 = 1 / sqrt(gamma_s), c2 = eta_s / sqrt(1 - gamma_bar_s),
    sigma_s = sqrt((1 - gamma_bar_{s-1}) / (1 - gamma_bar_s) * eta_s), and
    sigma = 0 for the first fast step (index 0), which is the last one taken.
    """
    raise NotImplementedError("fast_sampling_coefficients")


def diffwave_loss(model, audio: torch.Tensor, mel: torch.Tensor, alpha_bar: torch.Tensor, generator) -> torch.Tensor:
    """The noise-prediction loss for a mel-conditioned waveform model.

    audio (B, L), mel (B, n_mels, frames), alpha_bar of length T + 1 with
    alpha_bar[0] = 1. Draw t with torch.randint(1, T + 1, (B,), generator=...)
    and eps with torch.randn(audio.shape, generator=...), both on the CPU,
    then move them to audio's device. Build x_t, and return the mean of
    (model(x_t, mel, t.float()) - eps) ** 2.
    """
    raise NotImplementedError("diffwave_loss")


@torch.no_grad()
def fast_sample(model, mel: torch.Tensor, length: int, infer_betas, t_align, generator) -> torch.Tensor:
    """DiffWave's fast sampler. Returns audio of shape (B, length).

    Start from x = torch.randn(B, length, generator=...) moved to mel's device.
    For s = S - 1 down to 0: call model(x, mel, t) with t filled with
    t_align[s], set x = c1[s] (x - c2[s] eps), add sigma[s] times fresh
    torch.randn noise when s > 0, and clamp x to [-1, 1].
    """
    raise NotImplementedError("fast_sample")


# Grad-TTS -----------------------------------------------------------------------


def grad_tts_stats(x0, mu, t: float, beta0: float = 0.05, beta1: float = 20.0):
    """Mean and variance of X_t given X_0 = x0 under Grad-TTS's forward SDE, Sigma = I.

    With B(t) = integral_0^t beta = beta0 t + (beta1 - beta0) t^2 / 2:
    mean = exp(-B/2) x0 + (1 - exp(-B/2)) mu, and variance = 1 - exp(-B).
    Returns (mean, variance).
    """
    raise NotImplementedError("grad_tts_stats")


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
