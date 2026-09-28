"""Transducers and streaming: the worked solution.

Transducer log probabilities are batch-first, shape (B, T, U + 1, V): entry
[b, t, u, k] is log P(k | frame t, first u labels emitted) for utterance b.
Frames t and label positions u are 0-based here. The blank is index 0, the
course's convention; torchaudio's rnnt_loss puts it last by default.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from speechlab import play, show_image, show_plot

BLANK = 0


def rnnt_log_alpha(log_probs, targets, blank=BLANK):
    """The RNN-T forward variables in log space, shape (B, T, U + 1)."""
    B, T, U1, _ = log_probs.shape
    U = U1 - 1
    blank_lp = log_probs[..., blank]  # (B, T, U + 1)
    labels = targets[:, :U].clamp(min=0)
    emit_lp = log_probs[:, :, :U, :].gather(-1, labels[:, None, :, None].expand(B, T, U, 1))[..., 0]  # (B, T, U)
    alpha = [[None] * U1 for _ in range(T)]
    for t in range(T):
        for u in range(U1):
            if t == 0 and u == 0:
                alpha[t][u] = log_probs.new_zeros(B)
            elif t == 0:
                alpha[t][u] = alpha[t][u - 1] + emit_lp[:, t, u - 1]
            elif u == 0:
                alpha[t][u] = alpha[t - 1][u] + blank_lp[:, t - 1, u]
            else:
                alpha[t][u] = torch.logaddexp(
                    alpha[t - 1][u] + blank_lp[:, t - 1, u],
                    alpha[t][u - 1] + emit_lp[:, t, u - 1],
                )
    return torch.stack([torch.stack(row, dim=-1) for row in alpha], dim=1)


def rnnt_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK):
    """Per-utterance RNN-T negative log-likelihood, shape (B,)."""
    alpha = rnnt_log_alpha(log_probs, targets, blank)
    b = torch.arange(log_probs.shape[0])
    t_last = input_lengths - 1
    return -(alpha[b, t_last, target_lengths] + log_probs[b, t_last, target_lengths, blank])


def greedy_transducer_decode(encoder_out, predictor, joiner, blank=BLANK, max_symbols=10):
    """Greedy transducer decoding with at most `max_symbols` labels per frame."""
    tokens = []
    g, state = predictor(blank, None)
    for t in range(encoder_out.shape[0]):
        emitted = 0
        while emitted < max_symbols:
            k = int(torch.as_tensor(joiner(encoder_out[t], g)).argmax())
            if k == blank:
                break
            tokens.append(k)
            g, state = predictor(k, state)
            emitted += 1
    return tokens


def tdt_log_likelihood(token_log_probs, duration_log_probs, targets, durations, blank=BLANK):
    """log P(y | x) for a token-and-duration transducer, by its forward recursion."""
    token_log_probs = np.asarray(token_log_probs, dtype=float)
    duration_log_probs = np.asarray(duration_log_probs, dtype=float)
    T, U1, _ = token_log_probs.shape
    U = U1 - 1
    alpha = np.full((T + 1, U1), -np.inf)
    alpha[0, 0] = 0.0
    for t in range(T + 1):
        for u in range(U1):
            terms = [alpha[t, u]] if (t, u) == (0, 0) else []
            for j, d in enumerate(durations):
                src = t - d
                if src < 0 or src >= T:
                    continue
                if d >= 1:  # a blank must move forward
                    terms.append(alpha[src, u] + token_log_probs[src, u, blank] + duration_log_probs[src, u, j])
                if u >= 1:
                    y = targets[u - 1]
                    terms.append(alpha[src, u - 1] + token_log_probs[src, u - 1, y] + duration_log_probs[src, u - 1, j])
            if terms:
                alpha[t, u] = np.logaddexp.reduce(terms)
    return float(alpha[T, U])


def tdt_greedy_decode(encoder_out, predictor, joiner, durations, blank=BLANK, max_symbols=10):
    """Greedy TDT decoding that jumps ahead by the predicted duration."""
    tokens = []
    calls = 0
    g, state = predictor(blank, None)
    t, emitted_here = 0, 0
    T = encoder_out.shape[0]
    while t < T:
        token_lp, duration_lp = joiner(encoder_out[t], g)
        calls += 1
        k = int(torch.as_tensor(token_lp).argmax())
        d = durations[int(torch.as_tensor(duration_lp).argmax())]
        if k != blank:
            tokens.append(k)
            g, state = predictor(k, state)
            emitted_here += 1
        if k == blank and d == 0:
            d = 1  # a blank that stays put would loop forever
        if d == 0 and emitted_here >= max_symbols:
            d = 1
        if d > 0:
            emitted_here = 0
        t += d
    return tokens, calls


def chunked_attention_mask(T, chunk_size, left_chunks=None):
    """Bool (T, T): True where query frame i may attend to key frame j."""
    chunk = torch.arange(T) // chunk_size
    allowed = chunk[None, :] <= chunk[:, None]
    if left_chunks is not None:
        allowed &= chunk[None, :] >= chunk[:, None] - left_chunks
    return allowed


# The rest of the file is the experiment: a tiny streaming transducer ----------

RATE = 16000
FREQS = [400.0, 600.0, 850.0, 1100.0, 1450.0, 1800.0]
V = len(FREQS) + 1  # blank plus six tones
N_FFT, HOP, BANDS = 400, 160, 40
STACK = 4  # feature frames per encoder frame


def tone_sequence(rng, min_len=2, max_len=6):
    """A clip of 2 to 6 tones with random lengths and gaps, and its token labels."""
    n = int(rng.integers(min_len, max_len + 1))
    tokens = rng.integers(1, V, size=n)
    pieces = [np.zeros(int(rng.integers(800, 2400)))]
    for tok in tokens:
        dur = int(rng.integers(960, 2400))
        t = np.arange(dur) / RATE
        ramp = np.minimum(1.0, np.minimum(np.arange(dur), dur - np.arange(dur)) / 160.0)
        pitch = FREQS[tok - 1] * (1 + rng.normal(0, 0.02))
        pieces.append(0.4 * ramp * np.sin(2 * np.pi * pitch * t))
        pieces.append(np.zeros(int(rng.integers(480, 1600))))
    x = np.concatenate(pieces) + rng.normal(0, 0.02, size=sum(len(p) for p in pieces))
    return x.astype(np.float32), tokens.astype(np.int64)


def features(x):
    """Log band energies, 400-point STFT with hop 160 pooled into 40 bands, normalized, shape (T, 40)."""
    spec = torch.stft(torch.from_numpy(x), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    bands = spec.abs()[:200].reshape(BANDS, 5, -1).mean(1)
    return (torch.log(bands + 1e-3).T + 1.5) / 0.65


def make_batch(n, rng):
    clips, feats, toks = [], [], []
    for _ in range(n):
        x, t = tone_sequence(rng)
        clips.append(x)
        feats.append(features(x))
        toks.append(torch.from_numpy(t))
    lengths = torch.tensor([f.shape[0] for f in feats])
    X = nn.utils.rnn.pad_sequence(feats, batch_first=True)
    Y = nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)
    return X, lengths, Y, torch.tensor([len(t) for t in toks]), clips


class TinyTransducer(nn.Module):
    """A causal GRU encoder at 40 ms frames, a stateless predictor, and an additive joiner."""

    def __init__(self, d=64):
        super().__init__()
        self.inp = nn.Linear(STACK * BANDS, d)  # stack frames: 10 ms -> 40 ms
        self.rnn = nn.GRU(d, d, num_layers=2, batch_first=True)  # one direction only: streamable
        self.embed = nn.Embedding(V, d)  # the predictor sees only the last label
        self.enc_proj = nn.Linear(d, d)
        self.pred_proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, V)

    def encode(self, x, lengths):
        T = x.shape[1] // STACK * STACK
        x = x[:, :T].reshape(x.shape[0], T // STACK, STACK * BANDS)
        h, _ = self.rnn(torch.relu(self.inp(x)))
        return self.enc_proj(h), lengths // STACK

    def joint(self, f, g):
        return torch.log_softmax(self.out(torch.tanh(f + g)), dim=-1)

    def forward(self, x, lengths, targets):
        f, f_lengths = self.encode(x, lengths)  # (B, T, d)
        previous = torch.cat([torch.zeros_like(targets[:, :1]), targets], dim=1)  # blank starts
        g = self.pred_proj(self.embed(previous))  # (B, U + 1, d)
        return self.joint(f[:, :, None, :], g[:, None, :, :]), f_lengths  # (B, T, U + 1, V)

    def predictor(self, token, state):
        return self.pred_proj(self.embed(torch.tensor(token))), None

    def joiner(self, f_t, g):
        return self.joint(f_t, g)


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return row[-1]


def train_transducer(steps, batch, seed=0, lr=5e-3, log_every=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = TinyTransducer()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        X, L, Y, YL, _ = make_batch(batch, rng)
        log_probs, f_lengths = model(X, L, Y)
        loss = rnnt_loss(log_probs, Y, f_lengths, YL).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  RNN-T loss {loss.item():7.3f}")
    return model, losses


if __name__ == "__main__":
    start = time.time()
    model, losses = train_transducer(steps=400, batch=32, log_every=50)
    print(f"Trained a {sum(p.numel() for p in model.parameters()):,}-parameter transducer in {time.time() - start:.1f} s")
    show_plot(np.array(losses), label="RNN-T loss while training", xlabel="Step", ylabel="Loss per utterance")

    X, L, Y, YL, clips = make_batch(40, np.random.default_rng(123))
    errors = 0
    with torch.no_grad():
        f, f_lengths = model.encode(X, L)
        for i in range(len(clips)):
            hyp = greedy_transducer_decode(f[i, : f_lengths[i]], model.predictor, model.joiner, max_symbols=3)
            errors += edit_distance(hyp, Y[i, : YL[i]].tolist())
    print(f"Token error rate, greedy with max_symbols=3, on 40 new clips: {errors / YL.sum().item():.1%}")

    i = int(torch.argmax(YL))
    truth = Y[i, : YL[i]].tolist()
    with torch.no_grad():
        hyp = greedy_transducer_decode(f[i, : f_lengths[i]], model.predictor, model.joiner, max_symbols=3)
    print(f"Clip {i}: truth {truth}, greedy {hyp}")
    play(clips[i], RATE, label=f"A test clip: tones {truth}")

    # Where the probability flows in the lattice: the gradient of log P with
    # respect to each edge's log probability is that edge's posterior.
    log_probs, f_lengths = model(X[i : i + 1], L[i : i + 1], Y[i : i + 1, : YL[i]])
    log_probs = log_probs.detach().requires_grad_(True)
    nll = rnnt_loss(log_probs, Y[i : i + 1, : YL[i]], f_lengths, YL[i : i + 1])
    (grad,) = torch.autograd.grad(-nll.sum(), log_probs)
    occupancy = grad[0, : f_lengths[0]].sum(-1).T.numpy()  # (U + 1, T)
    show_image(
        occupancy,
        label="The RNN-T lattice: how likely the path passes each node (frame, labels emitted)",
        xlabel="Encoder frame (40 ms each)",
        ylabel="Labels emitted so far",
        y_labels=[str(u) for u in range(occupancy.shape[0])],
        origin="lower",
        colormap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    show_image(
        chunked_attention_mask(24, 4, left_chunks=2).float().numpy(),
        label="A chunked attention mask: chunks of 4 frames, 2 chunks of left context",
        xlabel="Key frame",
        ylabel="Query frame",
        colormap="gray",
    )
