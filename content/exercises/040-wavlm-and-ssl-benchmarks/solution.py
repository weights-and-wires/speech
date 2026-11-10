"""WavLM and SUPERB: the denoising mix, the relative position bias, and the weighted layer sum.

Signals are 1-D float arrays in samples. Energies are sums of squares, and gains
are in decibels. Attention arrays are indexed (query frame, key frame).
"""

import math

import numpy as np


def mix_for_denoising(
    primary: np.ndarray, secondary: np.ndarray, start: int, length: int, gain_db: float
) -> np.ndarray:
    """WavLM's noisy or overlapped input: add `secondary` into one span of `primary`.

    Only samples start .. start + length - 1 change. The secondary signal's
    span is scaled so that the energy (the sum of squares) of the primary span
    is `gain_db` decibels above the energy of the scaled secondary span. The
    pretraining targets stay those of the clean primary signal. Returns a new
    array, and leaves both inputs alone.
    """
    mixed = np.array(primary, dtype=float)
    span = slice(start, start + length)
    energy_primary = np.sum(mixed[span] ** 2)
    energy_secondary = np.sum(np.asarray(secondary, dtype=float)[span] ** 2)
    scale = np.sqrt(energy_primary / (10.0 ** (gain_db / 10.0) * energy_secondary))
    mixed[span] += scale * np.asarray(secondary, dtype=float)[span]
    return mixed


def bucket_relative_position(rel: np.ndarray, n_buckets: int = 320, max_distance: int = 800) -> np.ndarray:
    """The T5 and WavLM bucket of each signed offset rel = i - j, as ints.

    The first half of the buckets holds offsets with rel <= 0, and the second
    half holds rel > 0. Inside a half, |rel| below n_buckets / 4 gets its own
    bucket, larger distances share buckets spaced evenly in log(|rel|) up to
    max_distance, and everything at or beyond max_distance shares the last
    bucket of the half.
    """
    rel = np.asarray(rel)
    half = n_buckets // 2
    exact = half // 2
    distance = np.abs(rel)
    log_part = np.log(np.maximum(distance, 1) / exact) / np.log(max_distance / exact)
    large = np.minimum(exact + (log_part * (half - exact)).astype(int), half - 1)
    return np.where(distance < exact, distance, large) + np.where(rel > 0, half, 0)


def gated_relative_bias(
    queries: np.ndarray, u: np.ndarray, v: np.ndarray, w: float, bias: np.ndarray
) -> np.ndarray:
    """WavLM's gated relative position bias, shape (T, T).

    For query i, the update gate is g_u = sigmoid(q_i . u) and the reset gate is
    g_r = sigmoid(q_i . v). Row i of the result is bias[i] * (1 + g_u + (1 - g_u) * w * g_r),
    so the gates depend on the query alone and every key in the row shares one scale.
    """
    gate_update = 1.0 / (1.0 + np.exp(-(queries @ u)))
    gate_reset = 1.0 / (1.0 + np.exp(-(queries @ v)))
    scale = 1.0 + gate_update + (1.0 - gate_update) * w * gate_reset
    return bias * scale[:, None]


def superb_weighted_sum(layers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The SUPERB input to a downstream model, shape (T, D).

    `layers` has shape (L, T, D), the hidden states of L layers. `weights` has
    shape (L,), one learnable score per layer. Returns the sum over layers of
    softmax(weights)[l] * layers[l].
    """
    shifted = np.exp(weights - np.max(weights))
    mix = shifted / shifted.sum()
    return np.tensordot(mix, layers, axes=1)


if __name__ == "__main__":
    mixed = mix_for_denoising(np.ones(6), np.array([0.0, 0.0, 3.0, -3.0, 0.0, 0.0]), 2, 2, 10 * math.log10(4))
    print(f"WavLM mixes a span into a clip of ones at 6.02 dB: {np.round(mixed, 3).tolist()}.")
    buckets = bucket_relative_position(np.array([-20, -1, 0, 1, 4, 20]), 8, 16)
    print(f"With 8 buckets and a maximum distance of 16, offsets -20, -1, 0, 1, 4, 20 land in buckets {buckets.tolist()}.")
    queries = np.array([[0.0, 0.0], [2.0, -2.0]])
    gated = gated_relative_bias(queries, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, np.full((2, 2), 2.0))
    print(f"A constant bias of 2 becomes {np.round(gated, 4).tolist()} after the gates.")
    layers = np.array([4.0, 8.0, 12.0]).reshape(3, 1, 1)
    mix = superb_weighted_sum(layers, np.array([0.0, 0.0, math.log(2)]))
    print(f"SUPERB mixes layers (4, 8, 12) with scores (0, 0, ln 2) into {mix[0, 0]:.4f}.")
