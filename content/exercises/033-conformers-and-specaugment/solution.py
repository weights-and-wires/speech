"""Conformers and SpecAugment: the worked solution.

Encoder activations are batch-first, shape (B, T, d). A padding mask has shape
(B, T) and is True on real frames. Features for SpecAugment are one utterance
at a time, shape (n_mels, frames), already normalized to zero mean.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechlab import asset_path, has_asset, play, show_plot, show_spectrogram


def spec_augment(
    features,
    freq_mask_param,
    n_freq_masks,
    time_mask_param,
    time_ratio,
    n_time_masks,
    generator,
    length=None,
    mask_value=0.0,
):
    """SpecAugment frequency and time masking, on a copy of `features`."""
    out = features.clone()
    n_mels, frames = out.shape
    tau = frames if length is None else int(length)

    def randint(low, high):  # inclusive of both ends
        return int(torch.randint(low, high + 1, (1,), generator=generator))

    for _ in range(n_freq_masks):
        f = randint(0, freq_mask_param)
        f0 = randint(0, n_mels - f)
        out[f0 : f0 + f, :] = mask_value
    widest = min(time_mask_param, int(time_ratio * tau))
    for _ in range(n_time_masks):
        t = randint(0, widest)
        t0 = randint(0, tau - t)
        out[:, t0 : t0 + t] = mask_value
    return out


def relative_position_encoding(length, d_model):
    """Sinusoids for the relative distances L - 1, L - 2, ..., -(L - 1), shape (2L - 1, d_model)."""
    r = torch.arange(length - 1, -length, -1, dtype=torch.float32)[:, None]
    rate = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    pe = torch.zeros(2 * length - 1, d_model)
    pe[:, 0::2] = torch.sin(r * rate)
    pe[:, 1::2] = torch.cos(r * rate)
    return pe


def rel_shift(x):
    """Turn scores indexed by relative position into scores indexed by key frame."""
    *lead, L, P = x.shape  # P = 2L - 1
    padded = F.pad(x, (1, 0))  # (..., L, 2L)
    padded = padded.reshape(*lead, P + 1, L)
    return padded[..., 1:, :].reshape(*lead, L, P)[..., :L]


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Transformer-XL relative positions."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mask=None):
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.h, self.d_k)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(-1, self.h, self.d_k).transpose(0, 1)  # (H, 2T - 1, d_k)
        content = (q + self.pos_bias_u).transpose(1, 2) @ k.transpose(-2, -1)  # (B, H, T, T)
        position = rel_shift((q + self.pos_bias_v).transpose(1, 2) @ p.transpose(-2, -1))
        scores = (content + position) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(B, T, self.h * self.d_k)
        return self.linear_out(out)


class FeedForwardModule(nn.Module):
    """LayerNorm, Linear d -> 4d, Swish, Dropout, Linear 4d -> d, Dropout."""

    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, expansion * d_model)
        self.linear2 = nn.Linear(expansion * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.dropout(F.silu(self.linear1(self.norm(x))))))


class ConvolutionModule(nn.Module):
    """LayerNorm, pointwise conv d -> 2d, GLU, depthwise conv, BatchNorm, Swish, pointwise conv, Dropout."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y = self.norm(x).transpose(1, 2)  # (B, d, T)
        y = F.glu(self.pointwise1(y), dim=1)
        if mask is not None:
            y = y.masked_fill(~mask[:, None, :], 0.0)  # padding must not leak into real frames
        y = F.silu(self.batch_norm(self.depthwise(y)))
        return self.dropout(self.pointwise2(y)).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Half-step FFN, relative-position MHSA, convolution module, half-step FFN, LayerNorm."""

    def __init__(self, d_model, n_heads, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, dropout=dropout)
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.mhsa = RelPositionMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, dropout=dropout)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, mask=None):
        x = x + 0.5 * self.ff1(x)
        x = x + self.dropout(self.mhsa(self.norm_mhsa(x), pos_emb, mask))
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.norm_out(x)


# The rest of the file is given: the front end, the model, and the experiment --

RATE = 16000
N_FFT, HOP, N_MELS = 400, 160, 80  # 25 ms windows, 10 ms hop
HELD_OUT = "theo"
FSDD = "fsdd/free-spoken-digit-dataset-1.0.10/recordings"


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, rate=RATE):
    """Triangular HTK mel filters, shape (n_mels, n_fft // 2 + 1)."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    edges = hz(np.linspace(mel(0.0), mel(rate / 2), n_mels + 2))
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    lower = (freqs[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    upper = (edges[2:, None] - freqs[None, :]) / (edges[2:, None] - edges[1:-1, None])
    return torch.tensor(np.maximum(0.0, np.minimum(lower, upper)), dtype=torch.float32)


MEL = mel_filterbank()


def log_mel(waves, n_samples):
    """Normalized log-mel features for a padded batch of clips, 100 frames per second.

    `waves` has shape (B, samples) and `n_samples` holds each clip's true
    length. Returns features of shape (B, frames, 80) and frame counts (B,).
    Each utterance is normalized to zero mean and unit variance over its own
    frames, and padded frames are zero.
    """
    window = torch.hann_window(N_FFT, device=waves.device)
    spec = torch.stft(waves, N_FFT, HOP, window=window, return_complex=True)  # (B, 201, frames)
    feats = torch.log(MEL.to(waves.device) @ spec.abs() ** 2 + 1e-6).transpose(1, 2)
    lengths = 1 + n_samples // HOP
    valid = (torch.arange(feats.shape[1], device=waves.device)[None, :] < lengths[:, None])[..., None]
    count = valid.sum((1, 2), keepdim=True) * N_MELS
    mean = (feats * valid).sum((1, 2), keepdim=True) / count
    std = torch.sqrt((((feats - mean) * valid) ** 2).sum((1, 2), keepdim=True) / count)
    return ((feats - mean) / (std + 1e-5)) * valid, lengths


class Subsampler(nn.Module):
    """Two 3x3 Conv2d layers with stride 2: 100 frames per second become 25."""

    def __init__(self, n_mels, d_model, channels=None):
        super().__init__()
        channels = channels or d_model
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, 2), nn.ReLU(), nn.Conv2d(channels, channels, 3, 2), nn.ReLU()
        )
        self.proj = nn.Linear(channels * (((n_mels - 1) // 2 - 1) // 2), d_model)

    @staticmethod
    def out_length(lengths):
        return ((lengths - 1) // 2 - 1) // 2

    def forward(self, x, lengths):  # x: (B, frames, n_mels)
        y = self.conv(x.unsqueeze(1))  # (B, d, frames', mels')
        B, C, T, M = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, C * M)), self.out_length(lengths)


class ConformerCTC(nn.Module):
    def __init__(self, d_model=144, n_heads=4, n_blocks=4, kernel_size=15, n_symbols=11, dropout=0.1, sub_channels=None):
        super().__init__()
        self.subsample = Subsampler(N_MELS, d_model, sub_channels)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(ConformerBlock(d_model, n_heads, kernel_size, dropout) for _ in range(n_blocks))
        self.head = nn.Linear(d_model, n_symbols)
        self.d_model = d_model

    def forward(self, feats, lengths):  # feats: (B, frames, n_mels)
        x, out_lengths = self.subsample(feats, lengths)
        x = self.dropout(x)
        T = x.shape[1]
        mask = torch.arange(T, device=x.device)[None, :] < out_lengths[:, None]
        pos_emb = self.dropout(relative_position_encoding(T, self.d_model).to(x.device))
        for block in self.blocks:
            x = block(x, pos_emb, mask)
        return self.head(x).log_softmax(-1).transpose(0, 1), out_lengths  # (T, B, C)


def load_fsdd():
    """Every recording at 16 kHz, grouped as {speaker: [(digit, wave), ...]}."""
    from scipy.signal import resample_poly
    from speechlab import read_audio

    root = asset_path(FSDD)
    speakers = {}
    for path in sorted(root.glob("*.wav")):
        digit, speaker, _ = path.stem.split("_")
        wave, rate = read_audio(path)
        wave = resample_poly(wave, RATE, rate).astype(np.float32)
        wave = wave / (np.abs(wave).max() + 1e-6) * 0.5
        speakers.setdefault(speaker, []).append((int(digit), wave))
    return speakers


def make_utterances(pool, n, rng, device):
    """n utterances of 2 to 5 random digits in a row, with 30 to 120 ms gaps."""
    waves, labels = [], []
    for _ in range(n):
        picks = rng.integers(0, len(pool), size=int(rng.integers(2, 6)))
        pieces, digits = [], []
        for i in picks:
            digit, wave = pool[i]
            pieces += [wave, np.zeros(int(rng.integers(480, 1920)), dtype=np.float32)]
            digits.append(digit + 1)  # symbol 0 is the blank, so digit d is symbol d + 1
        waves.append(np.concatenate(pieces))
        labels.append(digits)
    n_samples = torch.tensor([len(w) for w in waves])
    padded = np.zeros((n, int(n_samples.max())), dtype=np.float32)
    for row, w in zip(padded, waves):
        row[: len(w)] = w
    feats, lengths = log_mel(torch.from_numpy(padded).to(device), n_samples.to(device))
    targets = nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels], batch_first=True).to(device)
    target_lengths = torch.tensor([len(l) for l in labels], device=device)
    return feats, lengths, targets, target_lengths, waves, labels


def greedy_digits(log_probs, lengths):
    best = log_probs.argmax(-1).T.tolist()  # (B, T)
    out = []
    for path, n in zip(best, lengths.tolist()):
        digits, prev = [], 0
        for s in path[:n]:
            if s != prev and s != 0:
                digits.append(s - 1)
            prev = s
        out.append(digits)
    return out


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return row[-1]


def digit_error_rate(model, test, device):
    model.eval()
    feats, lengths, targets, target_lengths, _, labels = test
    with torch.no_grad():
        log_probs, out_lengths = model(feats, lengths)
    hyps = greedy_digits(log_probs, out_lengths)
    model.train()
    return sum(edit_distance(h, [d - 1 for d in r]) for h, r in zip(hyps, labels)) / sum(len(r) for r in labels), hyps


def run_experiment(config, device, seed=0, log=print):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    speakers = load_fsdd()
    train_pool = [item for name, items in speakers.items() if name != HELD_OUT for item in items]
    test_pool = speakers[HELD_OUT]
    test = make_utterances(test_pool, config["n_test"], np.random.default_rng(1234), device)
    log(f"{len(train_pool)} training recordings from 5 speakers; {config['n_test']} test utterances from '{HELD_OUT}', never seen in training")

    model = ConformerCTC(
        config["d_model"], config["n_heads"], config["n_blocks"], config["kernel"],
        dropout=config["dropout"], sub_channels=config["sub_channels"],
    ).to(device)
    blocks = f"{config['n_blocks']} block" + ("s" if config["n_blocks"] > 1 else "")
    log(f"Conformer-CTC: {blocks} of width {config['d_model']}, {sum(p.numel() for p in model.parameters()):,} parameters")
    before, _ = digit_error_rate(model, test, device)
    log(f"Digit error rate before training: {before:.1%}")

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), weight_decay=1e-3)
    steps, warmup = config["steps"], config["warmup"]

    def lr_factor(step):  # linear warmup, then cosine decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))

    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    gen = torch.Generator().manual_seed(seed)
    losses, start = [], time.time()
    for step in range(steps):
        feats, lengths, targets, target_lengths, _, _ = make_utterances(train_pool, config["batch"], rng, device)
        if config["spec_augment"]:
            augmented = [
                spec_augment(f[:n].T, 15, 2, 20, 0.2, 2, gen).T for f, n in zip(feats, lengths.tolist())
            ]
            feats = nn.utils.rnn.pad_sequence(augmented, batch_first=True)
        log_probs, out_lengths = model(feats, lengths)
        loss = F.ctc_loss(log_probs, targets, out_lengths, target_lengths, zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        schedule.step()
        losses.append(loss.item())
        if step % config["log_every"] == 0 or step == steps - 1:
            log(f"step {step:5d}  CTC loss {loss.item():.3f}  ({time.time() - start:.0f} s)")
    after, hyps = digit_error_rate(model, test, device)
    log(f"Digit error rate after {steps} steps: {after:.1%} (from {before:.1%})")
    return model, losses, test, hyps, before, after


GPU_CONFIG = dict(d_model=144, n_heads=4, n_blocks=6, kernel=15, dropout=0.1, spec_augment=True, sub_channels=144, batch=64, steps=3000, warmup=300, lr=2e-3, n_test=300, log_every=250)
CPU_CONFIG = dict(d_model=96, n_heads=4, n_blocks=1, kernel=15, dropout=0.0, spec_augment=False, sub_channels=32, batch=16, steps=150, warmup=20, lr=3e-3, n_test=60, log_every=20)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    masked = spec_augment(torch.ones(8, 10), 3, 1, 4, 0.2, 1, torch.Generator().manual_seed(0))
    print(f"SpecAugment hid {int((masked == 0).sum())} of the 80 cells of an 8 by 10 spectrogram.")
    if not has_asset(FSDD):
        print("This run needs the 'fsdd' asset (the Free Spoken Digit Dataset), which this runner doesn't have. Stopping.")
    else:
        config = GPU_CONFIG if device == "cuda" else CPU_CONFIG
        if device == "cpu":
            print(f"No GPU here, so this is the small version: 1 block of width 96, no SpecAugment, {config['steps']} steps, "
                  "to finish in under a minute. On a GPU it trains 6 blocks of width 144 for 3,000 steps.")
        model, losses, test, hyps, before, after = run_experiment(config, device)
        show_plot(np.array(losses), label="CTC loss while training", xlabel="Step", ylabel="Loss per target symbol")
        feats, lengths, _, _, waves, labels = test
        i = int(np.argmax([len(l) for l in labels[:20]]))
        truth = [d - 1 for d in labels[i]]
        print(f"Test utterance {i}: said {truth}, recognized {hyps[i]}")
        play(waves[i], RATE, label=f"A held-out speaker says {' '.join(map(str, truth))}")
        show_spectrogram(
            feats[i, : lengths[i]].T.cpu().numpy(), RATE, HOP, db=False, y_axis="mel",
            label=f"Its normalized log-mel features; recognized: {' '.join(map(str, hyps[i]))}",
            value_label="normalized log power",
        )
