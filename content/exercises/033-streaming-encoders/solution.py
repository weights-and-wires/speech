"""Streaming encoders: a chunked attention mask and what it costs in latency.

Frames are encoder frames. A mask has shape (T, T): entry [i, j] is True when
query frame i may attend to key frame j. Delays are in milliseconds, and one
encoder frame lasts `frame_ms`.

Fill in the five functions below, then press Run to compare chunk sizes, and
Check to test your code.
"""

import math

import numpy as np
import torch

from speechlab import show_image, show_plot


def chunked_attention_mask(T, chunk_size, left_chunks=None):
    """A chunked self-attention mask for a streaming encoder.

    Frame i belongs to chunk i // chunk_size. Query frame i may attend to key
    frame j when j's chunk is i's chunk or earlier, and, if `left_chunks` is
    given, no more than `left_chunks` chunks earlier.

    Returns:
        A bool tensor, shape (T, T), True where attention is allowed.
    """
    chunk = torch.arange(T) // chunk_size
    allowed = chunk[None, :] <= chunk[:, None]
    if left_chunks is not None:
        allowed &= chunk[None, :] >= chunk[:, None] - left_chunks
    return allowed


def context_frames(mask):
    """How many frames each query frame may attend to, shape (T,) of int.

    This is the number of True entries in each row of the mask, counting the
    frame itself.
    """
    return mask.sum(dim=1)


def lookahead_frames(chunk_size):
    """The future frames each position inside a chunk waits for, shape (chunk_size,).

    Position p (0-based) in a chunk of `chunk_size` frames sees the frames up
    to the chunk's end, so it waits for chunk_size - 1 - p frames that have not
    arrived yet. Returns an int array, first position first.
    """
    return np.arange(chunk_size - 1, -1, -1)


def delay_ms(chunk_size, frame_ms):
    """The delay from the start of each frame to its output, shape (chunk_size,).

    A frame lasts `frame_ms`. Its output is ready when the last frame of its
    chunk has arrived, so the delay is its own frame plus its look-ahead,
    (lookahead + 1) * frame_ms, for each position in the chunk.
    """
    return (lookahead_frames(chunk_size) + 1) * frame_ms


def masked_self_attention(x, mask):
    """Single-head self-attention with x as query, key, and value.

    Args:
        x: shape (T, d), float tensor.
        mask: bool tensor, shape (T, T), True where attention is allowed.

    The scores are x x^T / sqrt(d). Disallowed scores become -inf before the
    softmax over keys.

    Returns:
        (output, weights): shapes (T, d) and (T, T). Row i of weights sums to
        1 and is exactly 0 where the mask is False.
    """
    d = x.shape[-1]
    scores = (x @ x.T) / math.sqrt(d)
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return weights @ x, weights


if __name__ == "__main__":
    T, frame_ms = 100, 40
    print(f"{T} frames of {frame_ms} ms each, {T * frame_ms / 1000:.0f} s of audio.")
    delays, contexts = [], []
    for chunk_size in (1, 2, 4, 8, 16, 32):
        mask = chunked_attention_mask(T, chunk_size, left_chunks=2)
        context = context_frames(mask).float().mean().item()
        delay = float(np.mean(delay_ms(chunk_size, frame_ms)))
        contexts.append(context)
        delays.append(delay)
        print(f"chunk of {chunk_size:2d} frames: mean delay {delay:6.0f} ms, mean context {context:5.1f} frames")
    show_plot(
        (np.array(delays), np.array(contexts)),
        label="More look-ahead buys more context (2 left chunks)",
        xlabel="Mean delay (ms)",
        ylabel="Mean frames each frame may attend to",
    )
    show_image(
        chunked_attention_mask(24, 4, left_chunks=2).float().numpy(),
        label="A chunked attention mask: chunks of 4 frames, 2 chunks of left context",
        xlabel="Key frame",
        ylabel="Query frame",
        colormap="gray",
    )
    x = torch.randn(12, 8, dtype=torch.float64)
    mask = chunked_attention_mask(12, 4)
    out, _ = masked_self_attention(x, mask)
    x2 = x.clone()
    x2[8:] += 1.0
    out2, _ = masked_self_attention(x2, mask)
    same = torch.allclose(out[:8], out2[:8])
    print(f"Changing the last chunk leaves the first two chunks' outputs unchanged: {same}")
