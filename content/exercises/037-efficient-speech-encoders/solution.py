"""Efficient speech encoders: compute counts, subsampling, a U-Net step, and a state-space layer.

Frames are encoder frames. Arrays are float64. A state-space layer here is a
single input channel with N diagonal states: h[t] = a * h[t-1] + b * x[t] and
y[t] = c . h[t], where a, b, and c have shape (N,).
"""

import numpy as np
import torch
from torch import nn

from speechlab import show_plot


def encoder_flops(frames, dim, layers, subsample):
    """FLOPs of a self-attention encoder after subsampling, counting a multiply-accumulate as 2."""
    t = frames // subsample
    return layers * (24 * t * dim * dim + 4 * t * t * dim)


def subsample_len(n, stages):
    """Length after `stages` Conv2d layers with kernel 3, stride 2, padding 1."""
    for _ in range(stages):
        n = (n - 1) // 2 + 1
    return n


def subsample_params(channels, stages, separable):
    """Parameters (weights and biases) of the subsampling stack, one input channel."""
    total = 9 * channels + channels
    for _ in range(stages - 1):
        if separable:
            total += (9 * channels + channels) + (channels * channels + channels)
        else:
            total += 9 * channels * channels + channels
    return total


def build_subsampler(channels, stages, separable):
    """A torch module: (B, 1, T, F) to (B, channels, T', F')."""
    layers = [nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.ReLU()]
    for _ in range(stages - 1):
        if separable:
            layers.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels))
            layers.append(nn.Conv2d(channels, channels, 1))
        else:
            layers.append(nn.Conv2d(channels, channels, 3, stride=2, padding=1))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def downsample(x, logits, factor):
    """Zipformer-style downsample: softmax-weighted average of `factor` adjacent frames."""
    t, d = x.shape
    pad = (-t) % factor
    if pad:
        x = np.concatenate([x, np.repeat(x[-1:], pad, axis=0)])
    w = np.exp(logits - np.max(logits))
    w = w / w.sum()
    return np.einsum("tfd,f->td", x.reshape(-1, factor, d), w)


def upsample(x, factor, length):
    """Repeat each frame `factor` times, then keep the first `length` frames."""
    return np.repeat(x, factor, axis=0)[:length]


def bypass(x, y, c):
    """(1 - c) * x + c * y with one scale per channel."""
    return (1.0 - c) * x + c * y


def ssm_recurrent(a, b, c, x, h0=None):
    """Run the recurrence one step at a time. Returns (y, final_state)."""
    h = np.zeros_like(a) if h0 is None else h0.copy()
    y = np.empty(len(x))
    for t, v in enumerate(x):
        h = a * h + b * v
        y[t] = c @ h
    return y, h


def ssm_kernel(a, b, c, length):
    """K[k] = sum_n c[n] * a[n]**k * b[n], for k = 0 .. length - 1."""
    k = np.arange(length)[:, None]
    return ((a[None, :] ** k) * (b * c)[None, :]).sum(axis=1)


def ssm_convolve(kernel, x):
    """Causal convolution: y[t] = sum_{k <= t} kernel[k] * x[t - k]."""
    return np.convolve(x, kernel)[: len(x)]


def ssm_chunk(a, b, c, x, h):
    """One chunk in the parallel form, starting from state `h`. Returns (y, new_state)."""
    n = len(x)
    y = ssm_convolve(ssm_kernel(a, b, c, n), x)
    y = y + (a[None, :] ** np.arange(1, n + 1)[:, None]) @ (c * h)
    new_h = a**n * h + (a[None, :] ** (n - 1 - np.arange(n))[:, None] * b[None, :]).T @ x
    return y, new_h


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
