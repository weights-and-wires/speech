"""Source separation: the worked solution."""

import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter

from speechlab import play, show_plot


# Scoring (given) ----------------------------------------------------------------------


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant signal-to-distortion ratio in dB, over the last axis.

    est, ref: tensors of the same shape (..., T). Removes the mean of each,
    projects est onto ref to get the target part, and returns
    10 log10(|target|^2 / |est - target|^2) with shape (...,).
    Differentiable, so -si_sdr works as a loss. This function is given in
    full, because the chapter before this one taught it.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10(((target**2).sum(-1) + eps) / ((noise**2).sum(-1) + eps))


# Separation ----------------------------------------------------------------------


def pit_loss(est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Utterance-level permutation-invariant negative SI-SDR.

    est, ref: (B, C, T). For each example, tries every assignment of the C
    outputs to the C references and keeps the one with the lowest mean
    -SI-SDR. Returns (the mean of those minima over the batch, a scalar,
    and perm of shape (B, C), where output c is matched to reference perm[b, c]).
    """
    batch, n_src, _ = est.shape
    pair = -si_sdr(est[:, :, None, :], ref[:, None, :, :])  # (B, C, C): output i against reference j
    perms = list(itertools.permutations(range(n_src)))
    index = torch.tensor(perms, device=est.device)  # (P, C)
    cost = pair[:, torch.arange(n_src), index].mean(-1)  # (B, P)
    best = cost.argmin(dim=1)
    return cost[torch.arange(batch), best].mean(), index[best]


def tcn_receptive_field(repeats: int, blocks: int, kernel: int) -> int:
    """Receptive field, in encoder frames, of `repeats` stacks of `blocks` dilated
    convolutions with dilations 1, 2, 4, ..., 2**(blocks - 1) and the given kernel size."""
    return 1 + repeats * (kernel - 1) * (2**blocks - 1)


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each frame, for (B, C, T) tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """One Conv-TasNet block with a residual connection, for (B, channels, T).

    1x1 conv to `hidden`, PReLU, ChannelNorm, a depthwise conv with the given
    kernel and dilation and "same" padding, PReLU, ChannelNorm, and a 1x1 conv
    back to `channels`, added to the input. Output shape equals input shape.
    """

    def __init__(self, channels: int, hidden: int, kernel: int, dilation: int):
        """Build the layers as self.net. The depthwise conv uses groups=hidden and
        padding dilation * (kernel - 1) // 2, which keeps the length for an odd kernel."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            ChannelNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2, groups=hidden),
            nn.PReLU(),
            ChannelNorm(hidden),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, channels, T) to (B, channels, T): the input plus the block's output."""
        return x + self.net(x)


class TinyConvTasNet(nn.Module):
    """A small Conv-TasNet: learned encoder, TCN mask estimator, learned decoder.

    Input (B, T) waveforms; output (B, n_src, T) separated waveforms, the same
    length as the input for any T.
    """

    def __init__(self, n_src: int = 2, n_filters: int = 64, length: int = 16, bottleneck: int = 32,
                 hidden: int = 64, kernel: int = 3, blocks: int = 5, repeats: int = 2):
        super().__init__()
        self.n_src, self.n_filters, self.length = n_src, n_filters, length
        self.stride = length // 2
        self.encoder = nn.Conv1d(1, n_filters, length, stride=self.stride, bias=False)
        self.separator = nn.Sequential(
            ChannelNorm(n_filters),
            nn.Conv1d(n_filters, bottleneck, 1),
            *[TCNBlock(bottleneck, hidden, kernel, 2**i) for _ in range(repeats) for i in range(blocks)],
            nn.PReLU(),
            nn.Conv1d(bottleneck, n_src * n_filters, 1),
        )
        self.decoder = nn.ConvTranspose1d(n_filters, 1, length, stride=self.stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) mixtures to (B, n_src, T) estimates.

        Right-pad x with zeros so the encoder's frames cover every sample,
        encode and apply ReLU to get w of shape (B, N, frames), run the
        separator and a sigmoid to get n_src masks of shape (N, frames) each,
        decode w * mask for every source, and trim the result to T samples.
        """
        batch, n = x.shape
        # Pad so the frames cover the whole input: n + pad = length + stride * (frames - 1).
        frames = max(1, -(-(n - self.length) // self.stride) + 1)
        pad = self.length + self.stride * (frames - 1) - n
        w = F.relu(self.encoder(F.pad(x, (0, pad))[:, None]))  # (B, N, frames)
        masks = torch.sigmoid(self.separator(w)).view(batch, self.n_src, self.n_filters, -1)
        masked = (w[:, None] * masks).reshape(batch * self.n_src, self.n_filters, -1)
        out = self.decoder(masked).view(batch, self.n_src, -1)
        return out[..., :n]


# Run helpers -----------------------------------------------------------------------

VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def vowel_train(rng, rate, seconds, f0, gap=0.0):
    """Source-filter vowels with a gliding pitch, 150-250 ms each, with optional pauses."""
    out = []
    while sum(len(o) for o in out) < seconds * rate:
        n = int(rng.uniform(0.15, 0.25) * rate)
        pitch = f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / rate))
        phase = np.cumsum(pitch / rate)
        src = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
        y = src
        for f, bw in zip(VOWELS[rng.integers(len(VOWELS))], (80, 100, 140)):
            r = np.exp(-np.pi * bw / rate)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
        out.append(y * np.hanning(n) ** 0.3)
        if gap:
            out.append(np.zeros(int(gap * rate)))
    x = np.concatenate(out)[: int(seconds * rate)]
    return 0.5 * x / np.max(np.abs(x))


def chirp_source(rng, rate, n):
    t = np.arange(n) / rate
    f0, f1 = rng.uniform(300, 1500), rng.uniform(300, 3000)
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1])) + rng.uniform(0, 2 * np.pi))
    return 0.5 * x * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))


def two_source_batch(rng, batch, rate=8000, seconds=0.5):
    n = int(rate * seconds)
    refs = np.stack([np.stack([vowel_train(rng, rate, seconds, rng.uniform(90, 200)), chirp_source(rng, rate, n)])
                     for _ in range(batch)])
    refs *= 10 ** (rng.uniform(-2.5, 2.5, size=(batch, 2, 1)) / 20)
    refs = torch.tensor(refs, dtype=torch.float32)
    return refs.sum(1), refs


def separation_demo(rng, steps=300):
    torch.set_num_threads(1)  # Convolutions this small run faster on one thread than on many.
    torch.manual_seed(0)
    model = TinyConvTasNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    test_mix, test_ref = two_source_batch(np.random.default_rng(99), 16)
    base = si_sdr(test_mix[:, None].expand_as(test_ref), test_ref).mean().item() + 0.0
    losses, started = [], time.time()
    for _ in range(steps):
        mix, ref = two_source_batch(rng, 8)
        loss, _ = pit_loss(model(mix), ref)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        est = model(test_mix)
        loss, perm = pit_loss(est, test_ref)
    print(f"\nSeparation: trained a {sum(p.numel() for p in model.parameters()):,}-parameter Conv-TasNet for "
          f"{len(losses)} steps ({time.time() - started:.0f} s).")
    print(f"  held-out SI-SDR: mixture {abs(base) if abs(base) < 0.05 else base:.1f} dB, separated {-loss.item():.1f} dB, "
          f"improvement {-loss.item() - base:.1f} dB")
    show_plot(-np.array(losses), label="Training SI-SDR (PIT)", xlabel="Step", ylabel="SI-SDR (dB)")
    play(test_mix[0].numpy() / test_mix[0].abs().max().item(), 8000, label="A held-out mixture: vowels plus a chirp")
    for c in range(2):
        e = est[0, c].numpy()
        play(0.9 * e / np.abs(e).max(), 8000, label=f"Output {c}, matched to reference {int(perm[0, c])}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    separation_demo(rng)
