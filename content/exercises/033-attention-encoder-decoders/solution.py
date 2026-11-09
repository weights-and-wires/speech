"""Attention encoder-decoders: the worked solution.

Batches are batch-first, (B, U, ...) for encoder frames, except CTC log
probabilities, which keep the (T, B, C) layout of torch.nn.functional.ctc_loss.
Token 0 plays two roles: it is the blank of the CTC head and the start and end
token (<eos>) of the attention decoder. Tokens 1 to 6 are the six tones.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechlab import play, show_image, show_plot

EOS = 0
PAD = -1


def dot_product_attention(query, keys, values, mask=None):
    """Scaled dot-product attention for one decoder step."""
    d = query.shape[-1]
    scores = torch.einsum("bd,bud->bu", query, keys) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = torch.einsum("bu,bud->bd", weights, values)
    return context, weights


def location_aware_attention(s_prev, h, alpha_prev, W, V, b, U, filters, w, mask=None):
    """Chorowski et al.'s location-aware attention for one decoder step."""
    k = filters.shape[-1]
    f = F.conv1d(alpha_prev.unsqueeze(1), filters, padding=k // 2).transpose(1, 2)  # (B, T, n_f)
    energies = torch.tanh((s_prev @ W.T).unsqueeze(1) + h @ V.T + b + f @ U.T) @ w  # (B, T)
    if mask is not None:
        energies = energies.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(energies, dim=-1)
    context = torch.einsum("bt,btd->bd", weights, h)
    return context, weights


def label_smoothing_loss(logits, targets, epsilon, pad=PAD):
    """Cross-entropy against (1 - eps) one-hot + eps / K, summed over tokens, per utterance."""
    n_classes = logits.shape[-1]
    log_p = torch.log_softmax(logits, dim=-1)
    keep = targets != pad
    safe = targets.clamp(min=0)
    nll = -log_p.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    uniform = -log_p.mean(dim=-1)
    per_token = (1.0 - epsilon) * nll + epsilon * uniform
    return (per_token * keep).sum() / logits.shape[0]


def joint_ctc_attention_loss(
    ctc_log_probs, ctc_lengths, att_logits, targets, target_lengths, lam=0.3, epsilon=0.1
):
    """lam * CTC + (1 - lam) * label-smoothed attention loss, both per utterance."""
    B = targets.shape[0]
    U = att_logits.shape[1]
    positions = torch.arange(U)[None, :]
    padded = torch.full((B, U), PAD, dtype=torch.long)
    width = min(targets.shape[1], U)
    padded[:, :width] = targets[:, :width]
    att_targets = torch.where(positions < target_lengths[:, None], padded, PAD)
    att_targets = torch.where(positions == target_lengths[:, None], EOS, att_targets)
    att = label_smoothing_loss(att_logits, att_targets, epsilon)
    ctc = F.ctc_loss(
        ctc_log_probs, targets.clamp(min=0), ctc_lengths, target_lengths,
        blank=0, reduction="sum", zero_infinity=True,
    ) / B
    return lam * ctc + (1.0 - lam) * att


def beam_search(step_fn, eos, beam_width, max_len, length_penalty=None):
    """Beam search over an autoregressive decoder, ranked with the GNMT length penalty."""
    alive = [((), 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for prefix, score in alive:
            log_p = np.asarray(step_fn(prefix), dtype=float)
            for k, lp in enumerate(log_p):
                if k == eos:
                    finished.append((prefix, score + lp))
                else:
                    candidates.append((prefix + (k,), score + lp))
        candidates.sort(key=lambda item: item[1], reverse=True)
        alive = candidates[:beam_width]
    finished.extend(alive)

    def penalty(n):
        return 1.0 if length_penalty is None else ((5.0 + n) / 6.0) ** length_penalty

    ranked = [(prefix, score / penalty(len(prefix))) for prefix, score in finished]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


# The rest of the file is the experiment: a tiny listen-attend-spell model ---

RATE = 16000
FREQS = [400.0, 600.0, 850.0, 1100.0, 1450.0, 1800.0]
N_TOKENS = len(FREQS) + 1  # <eos>/blank plus six tones
N_FFT, HOP, BANDS = 400, 160, 40


def tone_sequence(rng, min_len=2, max_len=6):
    """A clip of 2 to 6 tones with random lengths and gaps, and its token labels."""
    n = int(rng.integers(min_len, max_len + 1))
    tokens = rng.integers(1, N_TOKENS, size=n)
    pieces = [np.zeros(int(rng.integers(800, 2400)))]
    for tok in tokens:
        dur = int(rng.integers(960, 2400))
        t = np.arange(dur) / RATE
        ramp = np.minimum(1.0, np.minimum(np.arange(dur), dur - np.arange(dur)) / 160.0)
        pitch = FREQS[tok - 1] * (1 + rng.normal(0, 0.02))
        pieces.append(0.4 * ramp * np.sin(2 * np.pi * pitch * t))
        pieces.append(np.zeros(int(rng.integers(160, 1600))))
    x = np.concatenate(pieces) + rng.normal(0, 0.02, size=sum(len(p) for p in pieces))
    return x.astype(np.float32), tokens.astype(np.int64)


def features(x):
    """Log band energies: a 400-point STFT with hop 160, pooled into 40 bands, shape (T, 40)."""
    spec = torch.stft(torch.from_numpy(x), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    bands = spec.abs()[:200].reshape(BANDS, 5, -1).mean(1)
    return torch.log(bands + 1e-3).T


def make_batch(n, rng):
    """Features (B, T, 40), frame counts (B,), padded targets (B, U), target lengths (B,), clips."""
    clips, feats, toks = [], [], []
    for _ in range(n):
        x, t = tone_sequence(rng)
        clips.append(x)
        feats.append((features(x) + 1.5) / 0.65)  # roughly zero mean, unit variance
        toks.append(torch.from_numpy(t))
    lengths = torch.tensor([f.shape[0] for f in feats])
    X = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    targets = nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=PAD)
    return X, lengths, targets, torch.tensor([len(t) for t in toks]), clips


class TinyLAS(nn.Module):
    """A pyramidal BiGRU listener, a GRU speller with location-aware attention, and a CTC head."""

    def __init__(self, d=64, d_att=64, n_filters=8, kernel=9):
        super().__init__()
        self.d = d
        self.inp = nn.Linear(BANDS, d)
        self.rnn1 = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(2 * d, d // 2, batch_first=True, bidirectional=True)
        self.ctc_head = nn.Linear(d, N_TOKENS)
        self.embed = nn.Embedding(N_TOKENS, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.W = nn.Parameter(torch.randn(d_att, d) * d**-0.5)
        self.V = nn.Parameter(torch.randn(d_att, d) * d**-0.5)
        self.b = nn.Parameter(torch.zeros(d_att))
        self.U = nn.Parameter(torch.randn(d_att, n_filters) * n_filters**-0.5)
        self.filters = nn.Parameter(torch.randn(n_filters, 1, kernel) * 0.3)
        self.w = nn.Parameter(torch.randn(d_att) * d_att**-0.5)
        self.out = nn.Linear(2 * d, N_TOKENS)

    def encode(self, x, lengths):
        h, _ = self.rnn1(torch.relu(self.inp(x)))
        T = h.shape[1] // 2 * 2
        h = h[:, :T].reshape(h.shape[0], T // 2, -1)  # the pyramid: join frame pairs
        h, _ = self.rnn2(h)
        return h, lengths // 2

    def start(self, h, h_lengths):
        B, T, _ = h.shape
        mask = torch.arange(T)[None, :] < h_lengths[:, None]
        alpha = mask.float() / h_lengths[:, None].float()
        return torch.zeros(B, self.d), torch.zeros(B, self.d), alpha, mask

    def step(self, prev_token, s, c, alpha, h, mask):
        s = self.cell(torch.cat([self.embed(prev_token), c], -1), s)
        c, alpha = location_aware_attention(s, h, alpha, self.W, self.V, self.b, self.U, self.filters, self.w, mask)
        return self.out(torch.cat([s, c], -1)), s, c, alpha

    def forward(self, x, lengths, targets, target_lengths):
        h, h_lengths = self.encode(x, lengths)
        s, c, alpha, mask = self.start(h, h_lengths)
        B = targets.shape[0]
        inputs = torch.cat([torch.full((B, 1), EOS), targets.clamp(min=0)], 1)  # teacher forcing
        logits, alphas = [], []
        for i in range(inputs.shape[1]):
            out, s, c, alpha = self.step(inputs[:, i], s, c, alpha, h, mask)
            logits.append(out)
            alphas.append(alpha)
        ctc_log_probs = self.ctc_head(h).log_softmax(-1).transpose(0, 1)  # (T, B, C)
        return torch.stack(logits, 1), torch.stack(alphas, 1), ctc_log_probs, h_lengths


def train_las(steps, batch, seed=0, lam=0.3, epsilon=0.1, lr=5e-3, log_every=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = TinyLAS()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        X, L, Y, YL, _ = make_batch(batch, rng)
        logits, _, ctc_lp, h_lengths = model(X, L, Y, YL)
        loss = joint_ctc_attention_loss(ctc_lp, h_lengths, logits, Y, YL, lam, epsilon)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  joint loss {loss.item():7.3f}")
    return model, losses


def decoder_step_fn(model, x, length):
    """Wrap one utterance so step_fn(prefix) returns the decoder's next-token log probs."""
    with torch.no_grad():
        h, h_lengths = model.encode(x[None], length[None])

    def step_fn(prefix):
        with torch.no_grad():
            s, c, alpha, mask = model.start(h, h_lengths)
            for token in (EOS,) + tuple(prefix):
                out, s, c, alpha = model.step(torch.tensor([token]), s, c, alpha, h, mask)
            return torch.log_softmax(out[0], -1).numpy()

    return step_fn


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return row[-1]


if __name__ == "__main__":
    start = time.time()
    model, losses = train_las(steps=250, batch=32, log_every=50)
    print(f"Trained a {sum(p.numel() for p in model.parameters()):,}-parameter LAS in {time.time() - start:.1f} s")
    show_plot(np.array(losses), label="Joint CTC-attention loss while training", xlabel="Step", ylabel="Loss")

    X, L, Y, YL, clips = make_batch(40, np.random.default_rng(123))
    errors = 0
    for i in range(len(clips)):
        hyp, _ = beam_search(decoder_step_fn(model, X[i], L[i]), EOS, beam_width=4, max_len=10, length_penalty=1.0)[0]
        errors += edit_distance(list(hyp), Y[i, : YL[i]].tolist())
    print(f"Token error rate with beam 4 on 40 new clips: {errors / YL.sum().item():.1%}")

    i = int(torch.argmax(YL))  # the longest clip, for the clearest picture
    truth = Y[i, : YL[i]].tolist()
    hyp, score = beam_search(decoder_step_fn(model, X[i], L[i]), EOS, beam_width=4, max_len=10, length_penalty=1.0)[0]
    print(f"Clip {i}: truth {truth}, beam search {list(hyp)} (score {score:.3f})")
    play(clips[i], RATE, label=f"A test clip: tones {truth}")
    with torch.no_grad():
        _, alphas, _, h_lengths = model(X[i : i + 1], L[i : i + 1], Y[i : i + 1, : YL[i]], YL[i : i + 1])
    show_image(
        alphas[0, :, : h_lengths[0]].numpy(),
        label="Attention weights: each output token (row) looks at a few encoder frames (columns)",
        xlabel="Encoder frame (20 ms each)",
        ylabel="Output token",
        y_labels=[str(t) for t in truth] + ["<eos>"],
        colormap="viridis",
    )
