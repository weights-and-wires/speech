"""Codec language models: the worked solution."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechlab import play, show_image, show_plot


# Provided: a synthetic two-codebook source with a known entropy.


def sample_markov_codes(A: np.ndarray, B: np.ndarray, n_seq: int, T: int, rng) -> np.ndarray:
    """Codes of shape (n_seq, 2, T): codebook 0 is a Markov chain with transitions A,
    and codebook 1 at frame t is drawn from B[codebook 0 at frame t]."""
    V = A.shape[0]
    codes = np.zeros((n_seq, 2, T), dtype=np.int64)
    codes[:, 0, 0] = rng.integers(V, size=n_seq)
    cumA, cumB = np.cumsum(A, axis=1), np.cumsum(B, axis=1)
    for t in range(T):
        if t > 0:
            u = rng.random(n_seq)[:, None]
            codes[:, 0, t] = np.minimum((u > cumA[codes[:, 0, t - 1]]).sum(1), V - 1)
        u = rng.random(n_seq)[:, None]
        codes[:, 1, t] = np.minimum((u > cumB[codes[:, 0, t]]).sum(1), V - 1)
    return codes


def melody_source(V: int = 8):
    """The source the Run and the checks use: a wandering melody and a peaky timbre."""
    A = np.zeros((V, V))
    for i in range(V):
        A[i, (i + 1) % V] += 0.5
        A[i, (i - 1) % V] += 0.3
        A[i, i] += 0.2
    B = np.full((V, V), 0.2 / (V - 1))
    for j in range(V):
        B[j, (2 * j) % V] = 0.8
    return A, B


# The delay pattern ----------------------------------------------------------


def apply_delay(codes: np.ndarray, pad: int) -> np.ndarray:
    """Shift codebook k right by k steps: (K, T) -> (K, T + K - 1), gaps filled with `pad`."""
    K, T = codes.shape
    out = np.full((K, T + K - 1), pad, dtype=codes.dtype)
    for k in range(K):
        out[k, k : k + T] = codes[k]
    return out


def undo_delay(delayed: np.ndarray) -> np.ndarray:
    """The inverse of apply_delay: (K, S) -> (K, S - K + 1)."""
    K, S = delayed.shape
    T = S - K + 1
    return np.stack([delayed[k, k : k + T] for k in range(K)])


# VALL-E inputs ----------------------------------------------------------------


def valle_ar_example(phonemes, prompt_c1, target_c1, eos: int):
    """Next-token inputs, targets, and loss mask for VALL-E's autoregressive stage."""
    seq = np.concatenate([phonemes, prompt_c1, target_c1, [eos]]).astype(np.int64)
    inputs, targets = seq[:-1], seq[1:]
    start = len(phonemes) + len(prompt_c1)          # index in seq of the first target token
    mask = np.arange(1, len(seq)) >= start
    return inputs, targets, mask


def nar_input(codes: np.ndarray, embeddings: np.ndarray, stage: int) -> np.ndarray:
    """VALL-E's non-autoregressive input for codebook `stage`: the sum of the embeddings of
    codebooks 0 .. stage - 1. codes (K, T), embeddings (K, V, d) -> (T, d)."""
    out = np.zeros((codes.shape[1], embeddings.shape[2]))
    for k in range(stage):
        out += embeddings[k][codes[k]]
    return out


# MaskGIT decoding -------------------------------------------------------------


def maskgit_schedule(n_tokens: int, n_iters: int) -> list[int]:
    """Tokens still masked after each iteration: floor(N cos(pi/2 * i / I)), ending at 0."""
    remaining, prev = [], n_tokens
    for i in range(1, n_iters + 1):
        n = int(math.floor(n_tokens * math.cos(math.pi / 2 * i / n_iters)))
        n = max(0, min(n, prev - 1))
        remaining.append(n)
        prev = n
    remaining[-1] = 0
    return remaining


def maskgit_decode(logits_fn, n_tokens: int, n_iters: int, mask_id: int):
    """Confidence-based parallel decoding. Returns (tokens, masked_after_each_iteration)."""
    tokens = np.full(n_tokens, mask_id, dtype=np.int64)
    history = []
    for n_keep in maskgit_schedule(n_tokens, n_iters):
        logits = logits_fn(tokens.copy())
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        guess = probs.argmax(axis=1)
        confidence = probs[np.arange(n_tokens), guess]
        masked = tokens == mask_id
        confidence[~masked] = np.inf
        order = np.argsort(confidence, kind="stable")      # least confident first
        still_masked = order[:n_keep]
        reveal = masked.copy()
        reveal[still_masked] = False
        tokens[reveal] = guess[reveal]
        history.append(int((tokens == mask_id).sum()))
    return tokens, history


# A tiny codec language model -------------------------------------------------


def markov_code_entropy(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """The lowest loss per frame, in nats, of a model that sees the previous step,
    for the delay pattern and for the parallel pattern."""
    w, v = np.linalg.eig(A.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = pi / pi.sum()

    def H(rows):
        rows = np.clip(rows, 1e-300, 1.0)
        return -(rows * np.log(rows)).sum(axis=1)

    delay = float(pi @ H(A) + pi @ H(B))
    parallel = float(pi @ H(A) + pi @ H(A @ B))
    return delay, parallel


def lm_inputs_targets(grid: torch.Tensor, pad: int, bos: int):
    """Inputs are the grid shifted right one step with a BOS column; targets are the grid.
    grid (N, K, S) -> inputs (N, K, S), targets (N, K, S), mask (N, K, S) true where not pad."""
    bos_col = torch.full_like(grid[:, :, :1], bos)
    inputs = torch.cat([bos_col, grid[:, :, :-1]], dim=2)
    return inputs, grid, grid != pad


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy in nats over the positions where mask is true.
    logits (N, K, S, V), targets and mask (N, K, S)."""
    V = logits.shape[-1]
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    nll = F.cross_entropy(logits.reshape(-1, V), safe.reshape(-1), reduction="none").reshape(targets.shape)
    return (nll * mask).sum() / mask.sum()


class TinyCodecLM(nn.Module):
    """Predicts every codebook of step s from the tokens of step s - 1."""

    def __init__(self, n_codebooks: int, vocab: int, dim: int = 64):
        super().__init__()
        self.n_codebooks, self.vocab = n_codebooks, vocab
        self.embed = nn.ModuleList([nn.Embedding(vocab + 2, dim) for _ in range(n_codebooks)])
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.head = nn.Linear(dim, n_codebooks * vocab)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        N, K, S = inputs.shape
        h = sum(self.embed[k](inputs[:, k]) for k in range(K))       # (N, S, dim)
        out = self.head(self.mlp(h))                                 # (N, S, K * V)
        return out.reshape(N, S, K, self.vocab).permute(0, 2, 1, 3)  # (N, K, S, V)


# Run ------------------------------------------------------------------------


def _grids(codes, pattern, pad):
    if pattern == "delay":
        return torch.from_numpy(np.stack([apply_delay(c, pad) for c in codes]))
    return torch.from_numpy(codes)


def _train(codes, pattern, V, steps, seed=0):
    torch.manual_seed(seed)
    pad, bos = V, V + 1
    grid = _grids(codes, pattern, pad)
    model = TinyCodecLM(2, V)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for _ in range(steps):
        batch = grid[torch.randint(0, len(grid), (32,), generator=gen)]
        inputs, targets, mask = lm_inputs_targets(batch, pad, bos)
        loss = masked_cross_entropy(model(inputs), targets, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(2 * loss.item())   # nats per frame: two codebooks per frame
    return model, np.array(curve)


def _generate_delay(model, V, T, rng):
    """Sample a delayed grid column by column, then undo the delay."""
    pad, bos = V, V + 1
    K = 2
    grid = np.full((K, T + K - 1), pad, dtype=np.int64)
    prev = np.full(K, bos)
    with torch.no_grad():
        for s in range(T + K - 1):
            logits = model(torch.from_numpy(prev).reshape(1, K, 1))[0, :, 0]
            probs = torch.softmax(logits, dim=-1).numpy()
            for k in range(K):
                if k <= s < k + T:          # a real token in this row at this step
                    grid[k, s] = rng.choice(V, p=probs[k])
            prev = grid[:, s]
    return undo_delay(grid)


def _render(codes, rate=16000, frame_s=0.125):
    """Codebook 0 picks a note of a major scale; codebook 1 picks how bright it sounds."""
    scale = 220.0 * 2 ** (np.array([0, 2, 4, 5, 7, 9, 11, 12]) / 12)
    n = int(rate * frame_s)
    t = np.arange(n) / rate
    env = np.minimum(1, np.minimum(t, frame_s - t) / 0.01)
    out = []
    for note, bright in codes.T:
        partials = 1 + int(bright)
        wave = sum(np.sin(2 * np.pi * scale[note] * h * t) / h for h in range(1, partials + 1))
        out.append(env * wave)
    audio = np.concatenate(out)
    return 0.4 * audio / np.max(np.abs(audio))


if __name__ == "__main__":
    V, T, steps = 8, 64, 600
    A, B = melody_source(V)
    rng = np.random.default_rng(0)
    codes = sample_markov_codes(A, B, 512, T, rng)
    floor_delay, floor_parallel = markov_code_entropy(A, B)
    print(f"entropy floor per frame: delay {floor_delay:.3f} nats, parallel {floor_parallel:.3f} nats")

    show_image(apply_delay(codes[0, :, :12], V), label="The first 12 frames with the delay applied (8 is the pad)",
               xlabel="Step", ylabel="Codebook", y_labels=["0", "1"], colormap="viridis")

    curves = {}
    for pattern in ("delay", "parallel"):
        model, curve = _train(codes, pattern, V, steps)
        curves[pattern] = (model, curve)
        print(f"{pattern:8s}: final loss {curve[-50:].mean():.3f} nats per frame")

    x = np.arange(steps)
    show_plot(curves["delay"][1], curves["parallel"][1], (x, np.full(steps, floor_delay)), (x, np.full(steps, floor_parallel)),
              labels=["delay pattern", "parallel pattern", "delay floor", "parallel floor"],
              label="Training loss against the entropy floors", xlabel="Step", ylabel="Nats per frame", ylim=(0, 5))

    sample = _generate_delay(curves["delay"][0], V, 32, np.random.default_rng(1))
    play(_render(codes[0, :, :32]), 16000, label="Four seconds of the source")
    play(_render(sample), 16000, label="Four seconds sampled from your delay-pattern model")
    ok = np.mean([A[a, b] > 0 for a, b in zip(sample[0, :-1], sample[0, 1:])])
    print(f"{ok:.0%} of the sampled note steps are moves the source allows")
