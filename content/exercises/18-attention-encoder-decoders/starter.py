"""Attention encoder-decoders.

Batches are batch-first, (B, U, ...) for encoder frames, except CTC log
probabilities, which keep the (T, B, C) layout of torch.nn.functional.ctc_loss.
Token 0 plays two roles: it is the blank of the CTC head and the start and end
token (<eos>) of the attention decoder. Tokens 1 to 6 are the six tones.

Fill in the five functions below, then press Run to train a tiny
listen-attend-spell model on tone sequences, and Check to test your code.
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
    """Scaled dot-product attention for one decoder step.

    Args:
        query: The decoder's query, shape (B, d).
        keys: One key per encoder frame, shape (B, T, d).
        values: One value per encoder frame, shape (B, T, d_v).
        mask: Optional bool tensor, shape (B, T), True for real frames and
            False for padding. Padded frames must get a weight of exactly 0.

    Returns:
        (context, weights): context has shape (B, d_v) and is the weighted sum
        of the values; weights has shape (B, T) and is
        softmax over frames of (query . key) / sqrt(d).
    """
    raise NotImplementedError("dot_product_attention")


def location_aware_attention(s_prev, h, alpha_prev, W, V, b, U, filters, w, mask=None):
    """Location-aware attention (Chorowski et al., 2015) for one decoder step.

    Computes f = conv1d(alpha_prev, filters) with padding k // 2, then the
    energy of frame u, e_u = w . tanh(W s_prev + V h_u + U f_u + b), then
    weights = softmax over frames of e, and context = sum_u weights_u h_u.

    Args:
        s_prev: Decoder state, shape (B, d_s).
        h: Encoder frames, used as both keys and values, shape (B, T, d_h).
        alpha_prev: The previous step's attention weights, shape (B, T).
        W: shape (d_a, d_s). V: shape (d_a, d_h). b: shape (d_a,).
        U: shape (d_a, n_f). filters: conv1d weights, shape (n_f, 1, k), k odd.
        w: shape (d_a,).
        mask: Optional bool tensor, shape (B, T), True for real frames.

    Returns:
        (context, weights) with shapes (B, d_h) and (B, T).
    """
    raise NotImplementedError("location_aware_attention")


def label_smoothing_loss(logits, targets, epsilon, pad=PAD):
    """Label-smoothed cross-entropy, summed over tokens and averaged over utterances.

    The target distribution for a token with label y over K classes is
    q(k) = (1 - epsilon) * [k == y] + epsilon / K. The loss of one token is
    -sum_k q(k) log softmax(logits)_k.

    Args:
        logits: shape (B, U, K).
        targets: int64, shape (B, U). Positions equal to `pad` are skipped.
        epsilon: The smoothing weight, between 0 and 1.

    Returns:
        A scalar tensor: the total loss over every non-pad token, divided by B.
    """
    raise NotImplementedError("label_smoothing_loss")


def joint_ctc_attention_loss(
    ctc_log_probs, ctc_lengths, att_logits, targets, target_lengths, lam=0.3, epsilon=0.1
):
    """The joint CTC-attention loss, lam * L_CTC + (1 - lam) * L_att.

    Args:
        ctc_log_probs: CTC head log-softmax output, shape (T, B, C), blank 0.
        ctc_lengths: Encoder frames per utterance, shape (B,).
        att_logits: Decoder logits, shape (B, U + 1, C): one step per target
            token plus one for <eos>.
        targets: int64, shape (B, U'), padded with anything past each length.
        target_lengths: Tokens per utterance, shape (B,).
        lam: The CTC weight.
        epsilon: The label-smoothing weight of the attention loss.

    L_CTC is torch.nn.functional.ctc_loss with reduction="sum" and
    zero_infinity=True, divided by B. L_att is label_smoothing_loss against
    the attention targets: each utterance's tokens, then EOS at position
    target_lengths[b], then PAD.

    Returns:
        A scalar tensor.
    """
    raise NotImplementedError("joint_ctc_attention_loss")


def beam_search(step_fn, eos, beam_width, max_len, length_penalty=None):
    """Beam search over an autoregressive decoder.

    Args:
        step_fn: A function from a prefix (a tuple of token ids, without the
            start token) to a 1-D array of next-token log probabilities.
        eos: The end token's id.
        beam_width: How many unfinished hypotheses to keep after each step.
        max_len: The most tokens a hypothesis can hold, not counting <eos>.
        length_penalty: None to rank by the raw log probability, or alpha to
            divide it by the GNMT penalty ((5 + n) / 6) ** alpha, where n is
            the number of tokens without <eos>.

    Start from one empty hypothesis with log probability 0. At each of
    `max_len` steps, extend every unfinished hypothesis by every token. An
    extension by `eos` becomes a finished hypothesis; keep the `beam_width`
    best of the rest, by raw log probability, as the unfinished ones. After
    the last step, the surviving unfinished hypotheses count as finished too.

    Returns:
        Every finished hypothesis as (tokens, score) pairs, best first by the
        ranking score. `tokens` is a tuple without <eos>.
    """
    raise NotImplementedError("beam_search")


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
