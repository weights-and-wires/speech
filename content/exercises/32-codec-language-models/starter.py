"""Codec language models.

Fill in the functions and the TinyCodecLM class below, then press Run to train
your model on a synthetic two-codebook source with the delay pattern and the
parallel pattern, see both losses against their entropy floors, and hear a
melody sampled from your model.

`sample_markov_codes` and `melody_source` are provided: they make the data.
"""

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
    """Shift codebook k right by k steps.

    codes has shape (K, T). Return an array of shape (K, T + K - 1) with the
    same dtype, where row k holds codes[k] in columns k .. k + T - 1 and
    `pad` everywhere else.
    """
    raise NotImplementedError("apply_delay")


def undo_delay(delayed: np.ndarray) -> np.ndarray:
    """The inverse of apply_delay: (K, S) -> (K, S - K + 1)."""
    raise NotImplementedError("undo_delay")


# VALL-E inputs ----------------------------------------------------------------


def valle_ar_example(phonemes, prompt_c1, target_c1, eos: int):
    """Next-token inputs, targets, and loss mask for VALL-E's autoregressive stage.

    The sequence is phonemes, then the prompt's codebook-1 codes, then the
    target's codebook-1 codes, then `eos`, all in one shared vocabulary.
    Return (inputs, targets, mask), each of length len(sequence) - 1:
    inputs = sequence[:-1], targets = sequence[1:], and mask is true exactly
    where the target is a target code or the end token.
    """
    raise NotImplementedError("valle_ar_example")


def nar_input(codes: np.ndarray, embeddings: np.ndarray, stage: int) -> np.ndarray:
    """VALL-E's non-autoregressive input for codebook `stage` (0-indexed, >= 1).

    codes has shape (K, T) and embeddings has shape (K, V, d), one table per
    codebook. Return the (T, d) sum over k < stage of embeddings[k][codes[k]].
    """
    raise NotImplementedError("nar_input")


# MaskGIT decoding -------------------------------------------------------------


def maskgit_schedule(n_tokens: int, n_iters: int) -> list[int]:
    """How many tokens are still masked after each of the n_iters iterations.

    After iteration i (1-indexed), floor(n_tokens * cos(pi / 2 * i / n_iters))
    tokens stay masked. Clamp each entry to at most one less than the
    previous one (starting from n_tokens), and make the last entry 0.
    """
    raise NotImplementedError("maskgit_schedule")


def maskgit_decode(logits_fn, n_tokens: int, n_iters: int, mask_id: int):
    """Decode n_tokens positions in n_iters parallel steps.

    Start with every token equal to mask_id. At each iteration, call
    logits_fn(tokens) to get (n_tokens, V) logits, take the argmax at every
    position and its softmax probability as its confidence, and give
    already-revealed positions infinite confidence. Keep the
    maskgit_schedule(...)[i] least confident positions masked and reveal the
    rest of the masked ones with their argmax.

    Returns (tokens, history): the final int array, and the number of tokens
    still masked after each iteration.
    """
    raise NotImplementedError("maskgit_decode")


# A tiny codec language model -------------------------------------------------


def markov_code_entropy(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """The entropy floors of sample_markov_codes, in nats per frame.

    Codebook 0 is a Markov chain with transition matrix A (rows sum to 1) and
    stationary distribution pi, and codebook 1 at frame t is drawn from
    B[codebook 0 at frame t]. A model that sees only the previous step of the
    grid can reach:
        delay:    H(c0_t | c0_{t-1}) + H(c1_t | c0_t)
        parallel: H(c0_t | c0_{t-1}) + H(c1_t | c0_{t-1}),
    where p(c1_t | c0_{t-1} = i) is row i of A @ B. Each conditional entropy
    is sum_i pi_i H(row i). Find pi as the eigenvector of A.T for eigenvalue 1.
    Returns (delay, parallel).
    """
    raise NotImplementedError("markov_code_entropy")


def lm_inputs_targets(grid: torch.Tensor, pad: int, bos: int):
    """Next-step inputs, targets, and loss mask for a grid of codes.

    grid has shape (N, K, S). inputs is the grid shifted right by one step,
    with a first column of `bos`; targets is the grid itself; mask is true
    where targets != pad. All three have shape (N, K, S).
    """
    raise NotImplementedError("lm_inputs_targets")


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy in nats over the positions where mask is true.

    logits has shape (N, K, S, V); targets and mask have shape (N, K, S).
    Targets at masked-out positions may hold the pad id, which is not a valid
    class, so replace them before calling F.cross_entropy with
    reduction="none".
    """
    raise NotImplementedError("masked_cross_entropy")


class TinyCodecLM(nn.Module):
    """Predicts every codebook of step s from the tokens of step s - 1.

    Build, in __init__:
        embed: nn.ModuleList of n_codebooks nn.Embedding(vocab + 2, dim)
            tables, one per codebook; ids vocab and vocab + 1 are pad and BOS.
        mlp: Linear(dim, dim), GELU, Linear(dim, dim), GELU.
        head: Linear(dim, n_codebooks * vocab).
    forward takes inputs (N, K, S), sums the K embeddings at each step,
    applies mlp and head, and returns logits of shape (N, K, S, vocab).
    """

    def __init__(self, n_codebooks: int, vocab: int, dim: int = 64):
        super().__init__()
        raise NotImplementedError("TinyCodecLM.__init__")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TinyCodecLM.forward")


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
