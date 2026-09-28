"""Transducers and streaming.

Transducer log probabilities are batch-first, shape (B, T, U + 1, V): entry
[b, t, u, k] is log P(k | frame t, first u labels emitted) for utterance b.
Frames t and label positions u are 0-based here. The blank is index 0, the
course's convention; torchaudio's rnnt_loss puts it last by default.

Fill in the six functions below, then press Run to train a tiny streaming
transducer on tone sequences, and Check to test your code.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from speechlab import play, show_image, show_plot

BLANK = 0


def rnnt_log_alpha(log_probs, targets, blank=BLANK):
    """The RNN-T forward variables in log space.

    Args:
        log_probs: shape (B, T, U + 1, V), a log-softmax over the last axis.
        targets: int64, shape (B, U). Values past an utterance's length can be
            anything in [0, V).
        blank: The blank's index.

    Returns:
        alpha, shape (B, T, U + 1), where alpha[b, t, u] is the log of the
        total probability of reaching node (t, u): frame t, with the first u
        labels emitted. alpha[:, 0, 0] = 0, and

            alpha[t, u] = logaddexp(alpha[t - 1, u] + log_probs[t - 1, u, blank],
                                    alpha[t, u - 1] + log_probs[t, u - 1, targets[u - 1]])

        with the first term missing when t = 0 and the second when u = 0.
        Build it from torch operations, so gradients flow to log_probs.
    """
    raise NotImplementedError("rnnt_log_alpha")


def rnnt_loss(log_probs, targets, input_lengths, target_lengths, blank=BLANK):
    """Per-utterance RNN-T negative log-likelihood.

    Args:
        log_probs: shape (B, T, U + 1, V), padded to the longest utterance.
        targets: int64, shape (B, U), padded.
        input_lengths: frames per utterance T_b, shape (B,).
        target_lengths: labels per utterance U_b, shape (B,).

    Returns:
        shape (B,): -(alpha[b, T_b - 1, U_b] + log_probs[b, T_b - 1, U_b, blank]).
        The final blank leaves the last frame.
    """
    raise NotImplementedError("rnnt_loss")


def greedy_transducer_decode(encoder_out, predictor, joiner, blank=BLANK, max_symbols=10):
    """Greedy transducer decoding for one utterance.

    Args:
        encoder_out: shape (T, d), one row per encoder frame.
        predictor: predictor(token, state) -> (g, new_state). Start it with
            predictor(blank, None), and call it again after every label.
        joiner: joiner(encoder_out[t], g) -> log probabilities, shape (V,).
        max_symbols: The most labels to emit on one frame.

    At each frame, take the joiner's argmax. A blank moves to the next frame.
    A label is appended to the output, updates the predictor, and stays on
    the same frame, until the frame has emitted `max_symbols` labels.

    Returns:
        The labels, a list of ints.
    """
    raise NotImplementedError("greedy_transducer_decode")


def tdt_log_likelihood(token_log_probs, duration_log_probs, targets, durations, blank=BLANK):
    """log P(y | x) for a token-and-duration transducer (Xu et al., 2023).

    Args:
        token_log_probs: numpy, shape (T, U + 1, V): log P_T(v | t, u).
        duration_log_probs: numpy, shape (T, U + 1, D): column j is
            log P_D(durations[j] | t, u).
        targets: the U labels, a list of ints.
        durations: the D allowed durations, in frames, such as [0, 1, 2, 3, 4].

    From node (t, u), with t in 0..T-1, the model emits a token v and a
    duration d, with probability P_T(v | t, u) P_D(d | t, u). A blank moves to
    (t + d, u) and needs d >= 1. The next label targets[u] moves to
    (t + d, u + 1), and d can be 0. The path starts at (0, 0) and must land
    exactly on (T, U).

    Returns:
        The log of the total probability of every such path, a float.
    """
    raise NotImplementedError("tdt_log_likelihood")


def tdt_greedy_decode(encoder_out, predictor, joiner, durations, blank=BLANK, max_symbols=10):
    """Greedy TDT decoding that jumps ahead by the predicted duration.

    Args:
        encoder_out: shape (T, d).
        predictor: as in greedy_transducer_decode.
        joiner: joiner(encoder_out[t], g) -> (token_log_probs of shape (V,),
            duration_log_probs of shape (D,)).
        durations: the allowed durations, in frames.
        max_symbols: The most labels to emit on one frame without moving.

    Start at t = 0. At each step, call the joiner once, take the argmax token
    and the argmax duration d = durations[j]. Append a label and update the
    predictor; don't for a blank. Then move t forward by d. A blank with
    d = 0 moves 1 frame instead, and so does a label with d = 0 once the
    frame has emitted `max_symbols` labels. Stop when t >= T.

    Returns:
        (tokens, calls): the labels as a list of ints, and how many times the
        joiner was called.
    """
    raise NotImplementedError("tdt_greedy_decode")


def chunked_attention_mask(T, chunk_size, left_chunks=None):
    """A chunked self-attention mask for a streaming encoder.

    Frame i belongs to chunk i // chunk_size. Query frame i may attend to key
    frame j when j's chunk is i's chunk or earlier, and, if `left_chunks` is
    given, no more than `left_chunks` chunks earlier.

    Returns:
        A bool tensor, shape (T, T), True where attention is allowed.
    """
    raise NotImplementedError("chunked_attention_mask")


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
