"""Residual and scalar quantization.

Fill in the functions below, then press Run to hear a synthetic voice coded
with one to four residual quantizer stages, and Check to test your code.

`nearest_codeword` is provided in full: you built it in the chapter on vector
quantization. numpy handles the residual bookkeeping and the FSQ index. torch
handles FSQ's rounding, which needs a gradient.
"""

import numpy as np
import scipy.signal
import torch

from speechlab import play, show_plot


# Provided: nearest-codeword lookup (built in the vector quantization chapter) ---


def nearest_codeword(z: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The index and value of the closest codeword to each row of `z`.

    Args:
        z: Vectors to quantize, shape (N, D).
        codebook: Codewords, shape (K, D).

    Returns:
        (indices, quantized): integer indices of shape (N,), and the chosen
        codewords, shape (N, D).
    """
    distances = (
        (z**2).sum(axis=1, keepdims=True)
        - 2.0 * z @ codebook.T
        + (codebook**2).sum(axis=1)[None, :]
    )
    indices = np.argmin(distances, axis=1)
    return indices, codebook[indices]


# Residual vector quantization ---------------------------------------------


def rvq_encode(x: np.ndarray, codebooks: list[np.ndarray]):
    """Quantize `x` stage by stage, each stage coding what the earlier ones missed.

    Args:
        x: Vectors, shape (N, D).
        codebooks: A list of N_q codebooks, each of shape (K, D).

    Returns:
        (codes, reconstruction, residual_norms):
            codes: integers, shape (N, N_q), column i from codebook i.
            reconstruction: the sum of the chosen codewords, shape (N, D).
            residual_norms: shape (N, N_q + 1). Column 0 is |x|, and column
                i is the norm of the residual left after stage i.
    """
    raise NotImplementedError("rvq_encode")


def rvq_decode(codes: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Sum the chosen codeword of each stage, shape (N, D).

    `codes` has shape (N, n) with n <= len(codebooks); decode with the first
    n codebooks only, which is how a codec lowers its bitrate.
    """
    raise NotImplementedError("rvq_decode")


# Finite scalar quantization -----------------------------------------------


def fsq_quantize(z: torch.Tensor, levels: list[int]) -> torch.Tensor:
    """Finite scalar quantization with odd numbers of levels.

    Args:
        z: shape (N, d), with d = len(levels).
        levels: An odd number of levels L_i per dimension, such as [5, 5, 5].

    Returns:
        z_hat, shape (N, d): round(floor(L_i / 2) * tanh(z_i)) in each
        dimension, an integer in [-floor(L_i / 2), floor(L_i / 2)], with the
        rounding passed straight through so the gradient is that of
        floor(L_i / 2) * tanh(z_i).
    """
    raise NotImplementedError("fsq_quantize")


def fsq_index(z_hat: np.ndarray, levels: list[int]) -> np.ndarray:
    """The integer index of each FSQ grid point, shape (N,).

    Shift each dimension to a digit in {0, ..., L_i - 1} by adding
    floor(L_i / 2), then read the digits as a mixed-radix number with the
    first dimension varying fastest: idx = sum_i digit_i * prod_{j<i} L_j.
    """
    raise NotImplementedError("fsq_index")


def fsq_unindex(indices: np.ndarray, levels: list[int]) -> np.ndarray:
    """The inverse of fsq_index: grid points of shape (N, len(levels)) as floats."""
    raise NotImplementedError("fsq_unindex")


# Run -----------------------------------------------------------------------


def _vowel_sequence(rate: int, seconds: float) -> np.ndarray:
    """A buzzy voice that glides in pitch and moves through three vowels."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    f0 = 110.0 + 50.0 * t / seconds
    phase = np.cumsum(f0 / rate)
    source = np.diff(np.floor(phase), prepend=0.0)
    formants = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240)]
    out = np.zeros(n)
    edges = np.linspace(0, n, len(formants) + 1).astype(int)
    for (a, b), freqs in zip(zip(edges[:-1], edges[1:]), formants):
        seg = source[a:b]
        for f in freqs:
            r = np.exp(-np.pi * 80.0 / rate)
            seg = scipy.signal.lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * f / rate), r * r], seg)
        out[a:b] = seg
    return 0.5 * out / np.max(np.abs(out))


def _kmeans(data, K, rng, iters=12):
    """Fit one codebook with Lloyd's k-means, using nearest_codeword."""
    codebook = data[rng.choice(len(data), K, replace=False)].copy()
    for _ in range(iters):
        idx, _ = nearest_codeword(data, codebook)
        for k in range(K):
            members = data[idx == k]
            if len(members):
                codebook[k] = members.mean(axis=0)
            else:
                codebook[k] = data[rng.integers(len(data))]
    return codebook


def _snr(got, want):
    return 10 * np.log10(np.sum(want**2) / np.sum((got - want) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rate, dim, K, n_stages = 16000, 8, 64, 4

    speech = _vowel_sequence(rate, 2.0)
    vectors = speech[: len(speech) // dim * dim].reshape(-1, dim)
    print(f"{len(vectors)} vectors of {dim} samples; {rate // dim} vectors per second")

    # Residual VQ: each stage learns the residual the stages before it left.
    codebooks, residual = [], vectors.copy()
    for stage in range(n_stages):
        cb = _kmeans(residual, K, rng)
        codebooks.append(cb)
        _, q = nearest_codeword(residual, cb)
        residual = residual - q

    codes, recon, norms = rvq_encode(vectors, codebooks)
    snrs = []
    for n in range(1, n_stages + 1):
        decoded = rvq_decode(codes[:, :n], codebooks).reshape(-1)
        snrs.append(_snr(decoded, speech[: decoded.size]))
        kbps = rate / dim * n * np.log2(K) / 1000
        print(f"{n} stage(s): {kbps:5.1f} kbps, SNR {snrs[-1]:5.1f} dB, "
              f"mean residual norm {norms[:, n].mean():.4f}")

    # FSQ: the same vectors on a fixed grid, with nothing to train.
    levels = [5, 5, 5]
    z = torch.tensor(vectors[:, :3] * 4.0, dtype=torch.float32)
    idx = fsq_index(fsq_quantize(z, levels).numpy(), levels)
    print(f"FSQ with levels {levels}: {len(np.unique(idx))} of {int(np.prod(levels))} grid points used")

    play(speech, rate, label="The original voice")
    play(rvq_decode(codes[:, :1], codebooks).reshape(-1), rate, label="One RVQ stage (12 kbps)")
    play(recon.reshape(-1), rate, label=f"All {n_stages} stages (48 kbps)")
    show_plot(
        (np.arange(n_stages + 1), norms.mean(axis=0)),
        label="Mean residual norm after each stage",
        xlabel="Stages used",
        ylabel="Residual norm",
    )
    show_plot(
        (np.arange(1, n_stages + 1), np.array(snrs)),
        label="Reconstruction SNR against stages",
        xlabel="Stages used",
        ylabel="SNR (dB)",
    )
