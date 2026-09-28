"""A dilated causal WaveNet, mu-law targets, and cached sample-by-sample generation: the worked solution."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from speechlab import play, show_plot, show_spectrogram


def receptive_field(dilations: list[int], kernel_size: int = 2) -> int:
    return 1 + (kernel_size - 1) * int(sum(dilations))


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    f = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((f + 1) / 2 * mu + 0.5).astype(np.int64)


def mu_law_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    y = 2 * np.asarray(q, dtype=np.float64) / mu - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, dilation: int = 1) -> torch.Tensor:
    k = weight.shape[-1]
    return F.conv1d(F.pad(x, ((k - 1) * dilation, 0)), weight, bias, dilation=dilation)


def gated_activation(x: torch.Tensor) -> torch.Tensor:
    filt, gate = x.chunk(2, dim=1)
    return torch.tanh(filt) * torch.sigmoid(gate)


class WaveNet(nn.Module):
    """A small unconditional WaveNet over 256 mu-law classes."""

    def __init__(self, channels: int = 32, skip: int = 64, dilations: tuple[int, ...] = (1, 2, 4, 8), classes: int = 256):
        super().__init__()
        self.dilations = list(dilations)
        self.embed = nn.Embedding(classes, channels)
        self.dilated = nn.ModuleList(nn.Conv1d(channels, 2 * channels, 2, dilation=d) for d in dilations)
        self.residual = nn.ModuleList(nn.Conv1d(channels, channels, 1) for _ in dilations)
        self.skip = nn.ModuleList(nn.Conv1d(channels, skip, 1) for _ in dilations)
        self.head1 = nn.Conv1d(skip, skip, 1)
        self.head2 = nn.Conv1d(skip, classes, 1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        x = self.embed(codes).transpose(1, 2)  # (B, C, T)
        skips = 0
        for d, conv, res, skip in zip(self.dilations, self.dilated, self.residual, self.skip):
            z = gated_activation(causal_conv1d(x, conv.weight, conv.bias, d))
            x = x + res(z)
            skips = skips + skip(z)
        return self.head2(F.relu(self.head1(F.relu(skips))))

    def init_cache(self, batch: int) -> list[torch.Tensor]:
        """One zero buffer per layer, shape (batch, C, d): the inputs before the clip starts."""
        channels = self.embed.embedding_dim
        return [torch.zeros(batch, channels, d, dtype=self.embed.weight.dtype) for d in self.dilations]

    def step(self, code: torch.Tensor, cache: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(code)  # (B, C)
        skips = 0
        new_cache = []
        for d, conv, res, skip, past in zip(self.dilations, self.dilated, self.residual, self.skip, cache):
            w = conv.weight  # (2C, C, 2): tap 0 sees d samples back, tap 1 sees now
            h = past[:, :, 0] @ w[:, :, 0].T + x @ w[:, :, 1].T + conv.bias
            new_cache.append(torch.cat([past[:, :, 1:], x[:, :, None]], dim=2))
            z = gated_activation(h[:, :, None])[:, :, 0]
            x = x + z @ res.weight[:, :, 0].T + res.bias
            skips = skips + z @ skip.weight[:, :, 0].T + skip.bias
        hidden = F.relu(F.relu(skips) @ self.head1.weight[:, :, 0].T + self.head1.bias)
        return hidden @ self.head2.weight[:, :, 0].T + self.head2.bias, new_cache


@torch.no_grad()
def generate(model: WaveNet, n_samples: int, temperature: float = 1.0, seed: int = 0, first: int = 128) -> np.ndarray:
    generator = torch.Generator().manual_seed(seed)
    cache = model.init_cache(1)
    code = torch.tensor([first])
    out = np.empty(n_samples, dtype=np.int64)
    for t in range(n_samples):
        logits, cache = model.step(code, cache)
        if temperature == 0:
            code = torch.argmax(logits, dim=1)
        else:
            probs = torch.softmax(logits / temperature, dim=1)
            code = torch.multinomial(probs, 1, generator=generator)[:, 0]
        out[t] = int(code)
    return out


# The Run experiment ------------------------------------------------------------

RATE = 8000


def _vowel(seconds: float, rng) -> np.ndarray:
    """A sung vowel: a 160 Hz buzz with a slow vibrato, shaped by two formants."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f0 = 160 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    x = np.zeros(n)
    for k in range(1, 25):
        freq = k * 160
        gain = np.exp(-((freq - 700) / 200) ** 2) + 0.5 * np.exp(-((freq - 1200) / 250) ** 2) + 0.02
        x += gain / k**0.5 * np.sin(k * phase)
    return 0.6 * x / np.abs(x).max() + 0.003 * rng.standard_normal(n)


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    wave = _vowel(3.0, rng)
    codes = torch.from_numpy(mu_law_encode(wave))
    model = WaveNet(channels=32, skip=64, dilations=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32))
    rf = receptive_field(model.dilations)
    print(f"Receptive field: {rf} samples, {1000 * rf / RATE:.1f} ms at {RATE} Hz; "
          f"one pitch period is {RATE / 160:.0f} samples.")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = time.perf_counter()
    steps, crop = 300, 1024
    for step in range(1, steps + 1):
        starts = rng.integers(0, len(codes) - crop - 1, 8)
        batch = torch.stack([codes[s : s + crop + 1] for s in starts])
        logits = model(batch[:, :-1])  # position t predicts sample t + 1
        loss = F.cross_entropy(logits, batch[:, 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 60 == 0:
            print(f"step {step}: cross-entropy {loss.item():.3f} nats (a uniform guess is {np.log(256):.3f})")
        if time.perf_counter() - start > 30:
            print(f"Stopping at step {step}: this machine used up the 30-second training budget.")
            break
    print(f"Trained in {time.perf_counter() - start:.1f} s.")

    model.eval()
    n = RATE // 2
    start = time.perf_counter()
    generated = generate(model, n, temperature=1.0, seed=0, first=int(codes[0]))
    elapsed = time.perf_counter() - start
    print(f"Generated {n} samples one at a time in {elapsed:.1f} s: {n / elapsed:.0f} samples per second, "
          f"{n / elapsed / RATE:.2f} times real time.")
    audio = mu_law_decode(generated)
    play(wave[: RATE // 2], RATE, label="Training audio: a sung vowel at 8 kHz")
    play(audio, RATE, label="WaveNet's own half second, sampled one step at a time")
    window = torch.hann_window(512, dtype=torch.float64)
    spec = torch.stft(torch.from_numpy(audio), 512, 64, window=window, return_complex=True).abs()
    show_spectrogram(spec, RATE, 64, label="The spectrogram of the generated audio")
    show_plot(audio[:400], wave[:400], labels=["generated", "training audio"],
              label="The first 50 ms of each", xlabel="Sample", ylabel="Amplitude")
