"""Location-sensitive attention and the losses that teach it to align: the worked solution."""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from speechlab import play, show_image, show_spectrogram


def guided_attention_weights(n_frames: int, n_text: int, g: float = 0.2) -> torch.Tensor:
    t = torch.arange(n_frames, dtype=torch.float64)[:, None] / n_frames
    n = torch.arange(n_text, dtype=torch.float64)[None, :] / n_text
    return 1.0 - torch.exp(-((n - t) ** 2) / (2 * g**2))


def guided_attention_loss(
    attention: torch.Tensor, frame_lengths: torch.Tensor, text_lengths: torch.Tensor, g: float = 0.2
) -> torch.Tensor:
    batch, t_max, n_max = attention.shape
    weights = torch.zeros(batch, t_max, n_max, dtype=attention.dtype)
    for b in range(batch):
        t_b, n_b = int(frame_lengths[b]), int(text_lengths[b])
        weights[b, :t_b, :n_b] = guided_attention_weights(t_b, n_b, g).to(attention.dtype)
    frames = torch.arange(t_max)[None, :, None] < frame_lengths[:, None, None]
    texts = torch.arange(n_max)[None, None, :] < text_lengths[:, None, None]
    valid = frames & texts
    return (attention * weights * valid).sum() / valid.sum()


def stop_token_loss(logits: torch.Tensor, lengths: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    t_max = logits.shape[1]
    frame = torch.arange(t_max)[None, :]
    target = (frame == (lengths[:, None] - 1)).to(logits.dtype)
    valid = (frame < lengths[:, None]).to(logits.dtype)
    per_frame = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, dtype=logits.dtype), reduction="none"
    )
    return (per_frame * valid).sum() / valid.sum()


def location_sensitive_attention(
    query: torch.Tensor,
    memory: torch.Tensor,
    cumulative: torch.Tensor,
    mask: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = params["location_conv"]
    f = F.conv1d(cumulative[:, None, :], kernel, padding=(kernel.shape[-1] - 1) // 2)  # (B, F, N)
    f = f.transpose(1, 2)  # (B, N, F)
    hidden = (
        (query @ params["W_query"].T)[:, None, :]
        + memory @ params["W_memory"].T
        + f @ params["W_location"].T
        + params["bias"]
    )
    energies = torch.tanh(hidden) @ params["v"]  # (B, N)
    energies = energies.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(energies, dim=1)
    context = torch.bmm(weights[:, None, :], memory)[:, 0, :]
    return weights, context


def diagonality(attention: torch.Tensor, width: float = 0.1) -> float:
    t_frames, n_text = attention.shape
    t = torch.arange(t_frames, dtype=torch.float64)[:, None] / t_frames
    n = torch.arange(n_text, dtype=torch.float64)[None, :] / n_text
    band = (torch.abs(n - t) <= width).to(attention.dtype)
    return float((attention * band).sum() / attention.sum())


# The Run experiment: a toy text-to-speech system -----------------------------

RATE = 8000
N_FFT = 256
HOP = 64
N_SYMBOLS = 8
R = 2  # frames per decoder step
# Each symbol is a stationary "phone": two partials (Hz) and a length in frames.
PHONES = [
    (300, 900, 4), (400, 1600, 7), (500, 1200, 5), (250, 2200, 8),
    (600, 1000, 4), (350, 2600, 6), (450, 1900, 5), (700, 1300, 7),
]


def _window() -> torch.Tensor:
    return torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)


def _utterance(symbols: list[int]) -> np.ndarray:
    pieces = []
    for s in symbols:
        f1, f2, frames = PHONES[s]
        t = np.arange(frames * HOP) / RATE
        pieces.append(0.3 * np.sin(2 * np.pi * f1 * t) + 0.15 * np.sin(2 * np.pi * f2 * t))
    return np.concatenate(pieces)


def _log_magnitude(wave: np.ndarray) -> torch.Tensor:
    spec = torch.stft(torch.from_numpy(wave), N_FFT, HOP, window=_window(), center=True,
                      pad_mode="reflect", return_complex=True)
    return torch.log(spec.abs().clamp(min=1e-4)).T.float()  # (frames, bins)


def _griffin_lim(magnitude: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Griffin-Lim, as in the phase chapter: a waveform whose STFT magnitude matches `magnitude` (bins, frames)."""
    mag = torch.from_numpy(magnitude.astype(np.float64))
    length = (mag.shape[1] - 1) * HOP
    rng = np.random.default_rng(0)
    phase = torch.from_numpy(np.exp(2j * np.pi * rng.random(mag.shape)))
    for _ in range(iterations):
        wave = torch.istft(mag * phase, N_FFT, HOP, window=_window(), center=True, length=length)
        spec = torch.stft(wave, N_FFT, HOP, window=_window(), center=True, pad_mode="reflect",
                          return_complex=True)
        phase = spec / spec.abs().clamp(min=1e-8)
    return torch.istft(mag * phase, N_FFT, HOP, window=_window(), center=True, length=length).numpy()


class TinyTacotron(nn.Module):
    """A small Tacotron that predicts R frames per decoder step (Tacotron's reduction factor)."""

    def __init__(self, bins: int, r: int = 2, dim: int = 48, att: int = 32, filters: int = 8, kernel: int = 7):
        super().__init__()
        self.embed = nn.Embedding(N_SYMBOLS, dim)
        self.encoder = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.prenet = nn.Sequential(nn.Linear(bins, 64), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5))
        self.cell = nn.GRUCell(32 + dim, 96)
        self.att = nn.ParameterDict({
            "W_query": nn.Parameter(torch.randn(att, 96) * 0.1),
            "W_memory": nn.Parameter(torch.randn(att, dim) * 0.1),
            "location_conv": nn.Parameter(torch.randn(filters, 1, kernel) * 0.3),
            "W_location": nn.Parameter(torch.randn(att, filters) * 0.3),
            "v": nn.Parameter(torch.randn(att) * 0.3),
            "bias": nn.Parameter(torch.zeros(att)),
        })
        self.out = nn.Linear(96 + dim, r * bins)
        self.stop = nn.Linear(96 + dim, 1)
        self.bins, self.r = bins, r

    def encode(self, symbols):
        return self.encoder(self.embed(symbols))[0]

    def step(self, frame, state, context, memory, cumulative, mask):
        state = self.cell(torch.cat([self.prenet(frame), context], dim=1), state)
        weights, context = location_sensitive_attention(state, memory, cumulative, mask, dict(self.att))
        joint = torch.cat([state, context], dim=1)
        frames = self.out(joint).reshape(-1, self.r, self.bins)
        return frames, self.stop(joint)[:, 0], state, context, weights

    def forward(self, symbols, text_lengths, targets):
        """Teacher-forced decoding. targets has shape (B, R * steps, bins)."""
        memory = self.encode(symbols)
        batch, n_text, _ = memory.shape
        mask = torch.arange(n_text)[None, :] < text_lengths[:, None]
        state = memory.new_zeros(batch, 96)
        context = memory.new_zeros(batch, memory.shape[2])
        cumulative = memory.new_zeros(batch, n_text)
        frame = memory.new_full((batch, self.bins), math.log(1e-4))  # the "go" frame is silence
        outs, stops, aligns = [], [], []
        for k in range(targets.shape[1] // self.r):
            out, stop, state, context, weights = self.step(frame, state, context, memory, cumulative, mask)
            cumulative = cumulative + weights
            outs.append(out)
            stops.append(stop)
            aligns.append(weights)
            frame = targets[:, (k + 1) * self.r - 1]  # teacher forcing: the true last frame of this step
        return torch.cat(outs, 1), torch.stack(stops, 1), torch.stack(aligns, 1)

    @torch.no_grad()
    def infer(self, symbols, max_steps=60):
        memory = self.encode(symbols[None])
        mask = torch.ones(1, memory.shape[1], dtype=torch.bool)
        state = memory.new_zeros(1, 96)
        context = memory.new_zeros(1, memory.shape[2])
        cumulative = memory.new_zeros(1, memory.shape[1])
        frame = memory.new_full((1, self.bins), math.log(1e-4))
        outs, aligns = [], []
        for _ in range(max_steps):
            out, stop, state, context, weights = self.step(frame, state, context, memory, cumulative, mask)
            cumulative = cumulative + weights
            outs.append(out[0])
            aligns.append(weights[0])
            frame = out[:, -1]
            if torch.sigmoid(stop)[0] > 0.5:
                break
        return torch.cat(outs), torch.stack(aligns)


def _batch(rng, size):
    seqs = [list(rng.integers(0, N_SYMBOLS, rng.integers(4, 8))) for _ in range(size)]
    specs = [_log_magnitude(_utterance(s)) for s in seqs]
    text_lengths = torch.tensor([len(s) for s in seqs])
    frame_lengths = torch.tensor([len(x) for x in specs])
    symbols = torch.zeros(size, int(text_lengths.max()), dtype=torch.long)
    t_max = -(-int(frame_lengths.max()) // R) * R  # a whole number of decoder steps
    targets = torch.full((size, t_max, specs[0].shape[1]), math.log(1e-4))
    for i, (s, x) in enumerate(zip(seqs, specs)):
        symbols[i, : len(s)] = torch.tensor(s)
        targets[i, : len(x)] = x
    return symbols, text_lengths, targets, frame_lengths


if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    bins = N_FFT // 2 + 1
    model = TinyTacotron(bins, r=R)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    steps = 300
    print(f"Training a toy Tacotron for up to {steps} steps on random strings of {N_SYMBOLS} symbols, "
          f"{R} frames per decoder step.")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        symbols, text_lengths, targets, frame_lengths = _batch(rng, 16)
        pred, stop_logits, align = model(symbols, text_lengths, targets)
        frame_mask = (torch.arange(targets.shape[1])[None, :] < frame_lengths[:, None])[..., None]
        spec_loss = ((pred - targets).abs() * frame_mask).sum() / (frame_mask.sum() * bins)
        step_lengths = -(-frame_lengths // R)  # decoder steps per utterance
        stop = stop_token_loss(stop_logits, step_lengths, pos_weight=5.0)
        guide = guided_attention_loss(align, step_lengths, text_lengths)
        loss = spec_loss + stop + 10.0 * guide
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d}: spectrogram L1 {spec_loss.item():.3f}, stop {stop.item():.3f}, "
                  f"guided attention {guide.item():.4f}")
        if time.perf_counter() - start > 30:
            print(f"Stopping at step {step}: this machine used up the 30-second training budget.")
            break
    print(f"Trained in {time.perf_counter() - start:.1f} s.")

    model.eval()
    model.prenet.train()  # Tacotron 2 keeps pre-net dropout on at inference
    text = [1, 4, 6, 3, 0, 5]
    torch.manual_seed(1)
    frames, align = model.infer(torch.tensor(text))
    target = _log_magnitude(_utterance(text))
    print(f"Free-running decoding stopped after {len(align)} steps, {len(frames)} frames; "
          f"the true length is {len(target)} frames.")
    print(f"Diagonality of the alignment (mass within 0.1 of the diagonal): {diagonality(align.double()):.2f}")
    show_image(align.T, label="Attention: text position against decoder step", xlabel="Decoder step",
               ylabel="Symbol", y_labels=[str(s) for s in text], origin="lower", value_label="weight")
    show_spectrogram(torch.exp(frames).T, RATE, HOP, label="The predicted spectrogram")
    play(_utterance(text), RATE, label="The target utterance")
    audio = _griffin_lim(torch.exp(frames).T.numpy())
    play(0.9 * audio / max(1e-6, np.abs(audio).max()), RATE, label="The toy Tacotron's output through Griffin-Lim")
