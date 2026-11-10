"""Efficient speech encoders: compute counts, subsampling, a U-Net step, and a state-space layer.

Frames are encoder frames. Arrays are float64. A state-space layer here is a
single input channel with N diagonal states: h[t] = a * h[t-1] + b * x[t] and
y[t] = c . h[t], where a, b, and c have shape (N,).

Fill in the functions below, then press Run to see the numbers, and Check to
test your code.
"""

import numpy as np
import torch
from torch import nn

from speechlab import show_plot


def encoder_flops(frames, dim, layers, subsample):
    """FLOPs of a self-attention encoder after subsampling, counting a multiply-accumulate as 2.

    The encoder sees T = frames // subsample frames. One layer has four d x d
    projections, one feed-forward block with width 4d (two matrices), and
    attention over T frames (scores, then the weighted sum of values).
    Returns an int.
    """
    raise NotImplementedError("encoder_flops")


def subsample_len(n, stages):
    """Length after `stages` Conv2d layers with kernel 3, stride 2, padding 1.

    Takes an int `n` and returns an int.
    """
    raise NotImplementedError("subsample_len")


def subsample_params(channels, stages, separable):
    """Parameters (weights and biases) of the subsampling stack, one input channel.

    The first stage is always a dense Conv2d(1, channels, 3). Each later stage
    is a dense Conv2d(channels, channels, 3) when `separable` is False, and a
    depthwise Conv2d(channels, channels, 3, groups=channels) followed by a 1x1
    pointwise Conv2d(channels, channels, 1) when it is True. Every conv has a
    bias. Returns an int.
    """
    raise NotImplementedError("subsample_params")


def build_subsampler(channels, stages, separable):
    """A torch module: (B, 1, T, F) to (B, channels, T', F').

    Every conv that halves the size uses kernel 3, stride 2, padding 1. Put a
    ReLU after each stage. Its parameter count must equal `subsample_params`.
    """
    raise NotImplementedError("build_subsampler")


def downsample(x, logits, factor):
    """Zipformer-style downsample: softmax-weighted average of `factor` adjacent frames.

    Args:
        x: shape (T, d).
        logits: shape (factor,), the learnable weights before the softmax.
        factor: int.

    If T is not a multiple of `factor`, repeat the last frame to pad the end.
    Returns shape (ceil(T / factor), d).
    """
    raise NotImplementedError("downsample")


def upsample(x, factor, length):
    """Repeat each frame `factor` times, then keep the first `length` frames.

    Takes x of shape (T', d) and returns shape (length, d).
    """
    raise NotImplementedError("upsample")


def bypass(x, y, c):
    """Mix a module's input and output: (1 - c) * x + c * y, with one scale per channel.

    x and y have shape (T, d), and c has shape (d,).
    """
    raise NotImplementedError("bypass")


def ssm_recurrent(a, b, c, x, h0=None):
    """Run h[t] = a * h[t-1] + b * x[t], y[t] = c . h[t], one step at a time.

    Args:
        a, b, c: shape (N,). x: shape (T,). h0: shape (N,) or None for zeros.

    Returns:
        (y, h): y has shape (T,), and h is the final state, shape (N,).
    """
    raise NotImplementedError("ssm_recurrent")


def ssm_kernel(a, b, c, length):
    """The convolution kernel of the layer, shape (length,).

    K[k] = sum over states n of c[n] * a[n]**k * b[n], for k = 0 .. length - 1.
    Starting from h = 0, y[t] is the sum over k of K[k] * x[t - k].
    """
    raise NotImplementedError("ssm_kernel")


def ssm_convolve(kernel, x):
    """Causal convolution: y[t] = sum over k <= t of kernel[k] * x[t - k]. Shape (T,)."""
    raise NotImplementedError("ssm_convolve")


def ssm_chunk(a, b, c, x, h):
    """Process one chunk in the parallel form, starting from state `h`.

    Args:
        x: this chunk's inputs, shape (L,). h: the state before the chunk, shape (N,).

    Returns:
        (y, new_h): the chunk's outputs, shape (L,), and the state after it,
        shape (N,). Concatenating the y of consecutive chunks, each started from
        the previous new_h, must equal one pass over the whole input.
    """
    raise NotImplementedError("ssm_chunk")


if __name__ == "__main__":
    for s in (2, 4, 8, 16):
        print(f"{s:2d}x subsampling: {encoder_flops(1000, 256, 16, s) / 1e9:5.1f} GFLOPs for 10 s of audio")
    for sep in (False, True):
        print(f"{'separable' if sep else 'dense    '} 8x subsampler, 256 channels: {subsample_params(256, 3, sep):,} parameters")
    net = build_subsampler(16, 3, True)
    print("a (1, 1, 83, 80) input gives", tuple(net(torch.zeros(1, 1, 83, 80)).shape))

    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(0.5, 0.95, 4), rng.normal(size=4), rng.normal(size=4)
    x = rng.normal(size=48)
    full, _ = ssm_recurrent(a, b, c, x)
    conv = ssm_convolve(ssm_kernel(a, b, c, len(x)), x)
    print(f"recurrent vs convolutional form: max difference {np.max(np.abs(full - conv)):.2e}")
    h, pieces = np.zeros(4), []
    for lo, hi in ((0, 7), (7, 20), (20, 48)):
        y, h = ssm_chunk(a, b, c, x[lo:hi], h)
        pieces.append(y)
    print(f"chunks of 7, 13, and 28 frames vs the full pass: max difference {np.max(np.abs(np.concatenate(pieces) - full)):.2e}")
    show_plot(full, np.concatenate(pieces), label="Full pass and chunked pass overlap", xlabel="Frame", labels=["full", "chunked"])
