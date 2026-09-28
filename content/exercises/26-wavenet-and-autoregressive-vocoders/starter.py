"""A dilated causal WaveNet, mu-law targets, and cached sample-by-sample generation.

Fill in the functions and the two WaveNet methods, then press Run. The run
trains a small WaveNet on three seconds of a sung vowel at 8 kHz for about 15
seconds, then generates half a second of new audio one sample at a time with
your cache, and plays it.

Audio is in [-1, 1]. Codes are integers 0 to 255. Tensors are laid out
(batch, channels, time).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from speechlab import play, show_plot, show_spectrogram


def receptive_field(dilations: list[int], kernel_size: int = 2) -> int:
    """How many input samples one output of a stack of dilated causal convolutions can see.

    dilations lists one dilation per layer; every layer has the same kernel_size.
    Return 1 + (kernel_size - 1) * sum(dilations), as an int.
    """
    raise NotImplementedError("receptive_field")


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Compand x in [-1, 1] with the mu-law and quantize it to integer codes 0 .. mu.

    f = sign(x) * log(1 + mu |x|) / log(1 + mu), then code = floor((f + 1) / 2 * mu + 0.5).
    Clip x to [-1, 1] first. Return an int64 array with the shape of x.
    """
    raise NotImplementedError("mu_law_encode")


def mu_law_decode(q: np.ndarray, mu: int = 255) -> np.ndarray:
    """Invert mu_law_encode: codes 0 .. mu back to amplitudes in [-1, 1].

    y = 2 * code / mu - 1, then x = sign(y) * ((1 + mu) ** |y| - 1) / mu.
    Return a float64 array with the shape of q.
    """
    raise NotImplementedError("mu_law_decode")


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, dilation: int = 1) -> torch.Tensor:
    """A dilated convolution whose output at time t sees only inputs at t and before.

    Args:
        x: Input, shape (B, C_in, T).
        weight: Kernel, shape (C_out, C_in, K), in F.conv1d's layout: tap K - 1
            multiplies the current sample, tap 0 the one (K - 1) * dilation back.
        bias: Shape (C_out,), or None.
        dilation: The gap between taps, in samples.

    Returns:
        Shape (B, C_out, T), the same length as the input. Samples
        before the start count as zeros.
    """
    raise NotImplementedError("causal_conv1d")


def gated_activation(x: torch.Tensor) -> torch.Tensor:
    """WaveNet's gate: tanh of the first half of the channels times sigmoid of the second half.

    x has shape (B, 2C, ...). Return shape (B, C, ...).
    """
    raise NotImplementedError("gated_activation")


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
        """Logits for every next sample, computed in parallel.

        Args:
            codes: Integer codes, shape (B, T).

        Returns:
            Logits, shape (B, 256, T). Column t is the model's distribution over
            code t + 1 given codes 0 .. t.

        The architecture is the one in the lesson: embedded codes enter the
        stack as (B, C, T). Layer i is the gated activation of the causal
        convolution with dilated[i]'s weights and dilation. The layer's input
        plus residual[i] of that output feeds the next layer, and skip[i] of it
        adds to a skip sum that starts at zero. The head applies ReLU, head1,
        ReLU, and head2 to the skip sum.
        """
        raise NotImplementedError("forward")

    def init_cache(self, batch: int) -> list[torch.Tensor]:
        """One zero buffer per layer, shape (batch, C, d): the inputs before the clip starts."""
        channels = self.embed.embedding_dim
        return [torch.zeros(batch, channels, d, dtype=self.embed.weight.dtype) for d in self.dilations]

    def step(self, code: torch.Tensor, cache: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """One step of generation with a cache: the same logits forward gives, for one new sample.

        Args:
            code: The newest code, shape (B,), integers.
            cache: One tensor per layer, from init_cache or the previous step.
                cache[i] has shape (B, C, d_i) and holds that layer's inputs at the
                last d_i steps, oldest first.

        Returns:
            (logits, new_cache). logits has shape (B, 256) and equals column t of
            forward(codes 0 .. t). new_cache has the same layout, with this step's
            layer inputs appended and the oldest dropped. Don't call forward, and
            don't modify the old cache in place.
        """
        raise NotImplementedError("step")


@torch.no_grad()
def generate(model: WaveNet, n_samples: int, temperature: float = 1.0, seed: int = 0, first: int = 128) -> np.ndarray:
    """Generate n_samples codes one at a time, starting from the code `first`.

    Use model.init_cache(1) and model.step. At each step, pick the next code
    from the logits: the argmax when temperature is 0, and otherwise a draw from
    softmax(logits / temperature) with torch.multinomial and a torch.Generator
    seeded with `seed`. Feed each new code back in. Return the n_samples new
    codes as an int64 numpy array, not including `first`.
    """
    raise NotImplementedError("generate")


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
