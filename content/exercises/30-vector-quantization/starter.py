"""Vector quantization.

Fill in the functions below, then press Run to hear a synthetic voice coded
with one to four residual quantizer stages, and Check to test your code.

numpy handles the codebook bookkeeping (nearest codeword, EMA updates, RVQ,
FSQ indices). torch handles the two pieces that need gradients: the
straight-through estimator and the VQ losses, plus FSQ's rounding.
"""

import numpy as np
import scipy.signal
import torch

from speechlab import play, show_plot


# Nearest codeword ---------------------------------------------------------


def nearest_codeword(z: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The index and value of the closest codeword to each row of `z`.

    Args:
        z: Vectors to quantize, shape (N, D).
        codebook: Codewords, shape (K, D).

    Returns:
        (indices, quantized): integer indices of shape (N,), and the chosen
        codewords, shape (N, D). Compute the squared distances with the
        expansion |z|^2 - 2 z C^T + |C|^2, never with an (N, K, D) array.
    """
    raise NotImplementedError("nearest_codeword")


# Gradients through the argmin ---------------------------------------------


def straight_through(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """A tensor whose value is `z_q` and whose gradient flows to `z_e` unchanged.

    Both tensors have the same shape. The backward pass treats the output as
    if it were `z_e`, so d(out)/d(z_e) is the identity.
    """
    raise NotImplementedError("straight_through")


def vq_losses(z_e: torch.Tensor, codebook: torch.Tensor):
    """Quantize `z_e` against `codebook` and return the two VQ-VAE losses.

    Args:
        z_e: Encoder outputs, shape (N, D), which may require grad.
        codebook: Codewords, shape (K, D), which may require grad.

    Returns:
        (z_q_st, indices, codebook_loss, commitment_loss):
            z_q_st: the chosen codewords with the straight-through gradient,
                shape (N, D).
            indices: a LongTensor of shape (N,).
            codebook_loss: mean over the N vectors of |sg[z_e] - e|^2, whose
                gradient reaches only the codebook.
            commitment_loss: mean over the N vectors of |z_e - sg[e]|^2, whose
                gradient reaches only z_e. It is unweighted; the caller
                multiplies it by beta.
        Here e is the chosen codeword and sg is the stop-gradient (.detach()).
    """
    raise NotImplementedError("vq_losses")


# EMA codebook and dead-code restart ---------------------------------------


def ema_update(counts, sums, z, indices, decay=0.99, eps=1e-5):
    """One exponential-moving-average codebook step.

    Args:
        counts: EMA assignment counts N_k, shape (K,).
        sums: EMA sums of assigned vectors m_k, shape (K, D).
        z: This batch's vectors, shape (B, D).
        indices: The code each vector chose, shape (B,).
        decay: gamma, the weight on the old statistics.
        eps: Laplace smoothing added to every count before dividing.

    Returns:
        (counts, sums, codebook): the updated N_k and m_k, and the codebook
        m_k / N~_k, where N~_k = (N_k + eps) / (sum_j N_j + K eps) * sum_j N_j.
        Don't modify the inputs in place.
    """
    raise NotImplementedError("ema_update")


def restart_dead_codes(codebook, counts, sums, z, threshold, rng):
    """Replace every code whose EMA count is below `threshold` with a random row of `z`.

    Args:
        codebook: shape (K, D). counts: shape (K,). sums: shape (K, D).
        z: This batch's vectors, shape (B, D).
        threshold: A code with counts[k] < threshold is dead.
        rng: A numpy Generator; draw the replacement rows with rng.choice.

    Returns:
        (codebook, counts, sums, n_restarted): new arrays (copies), where each
        dead code k gets codebook[k] = a row of z, counts[k] = threshold, and
        sums[k] = codebook[k] * threshold, so sums / counts stays equal to the
        codebook. Live codes are unchanged.
    """
    raise NotImplementedError("restart_dead_codes")


def codebook_perplexity(indices: np.ndarray, K: int) -> float:
    """exp(-sum_k p_k log p_k), where p_k is the fraction of `indices` equal to k.

    It is K when all K codes are used equally and 1 when only one code is used.
    Skip codes with p_k = 0, whose term is 0.
    """
    raise NotImplementedError("codebook_perplexity")


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


def _train_stage(data, K, rng, iters=40, restart=True, init=None):
    """Fit one codebook with EMA updates and optional dead-code restart."""
    codebook = data[rng.choice(len(data), K, replace=False)] if init is None else init.copy()
    counts = np.ones(K)
    sums = codebook.copy()
    for _ in range(iters):
        batch = data[rng.choice(len(data), 1024, replace=False)]
        idx, _ = nearest_codeword(batch, codebook)
        counts, sums, codebook = ema_update(counts, sums, batch, idx, decay=0.8)
        if restart:
            codebook, counts, sums, _ = restart_dead_codes(codebook, counts, sums, batch, 1.0, rng)
    return codebook


def _snr(got, want):
    return 10 * np.log10(np.sum(want**2) / np.sum((got - want) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rate, dim, K, n_stages = 16000, 8, 64, 4

    speech = _vowel_sequence(rate, 2.0)
    vectors = speech[: len(speech) // dim * dim].reshape(-1, dim)
    print(f"{len(vectors)} vectors of {dim} samples; {rate // dim} vectors per second")

    # Collapse and its fix: start every code far from the data.
    far = rng.normal(size=(K, dim)) * 0.01 + 3.0
    for restart in (False, True):
        cb = _train_stage(vectors, K, rng, restart=restart, init=far)
        idx, _ = nearest_codeword(vectors, cb)
        print(f"one codebook, far init, restart={restart}: perplexity {codebook_perplexity(idx, K):.1f} of {K}")

    # Residual VQ: each stage learns the residual the stages before it left.
    codebooks, residual = [], vectors.copy()
    for stage in range(n_stages):
        cb = _train_stage(residual, K, rng)
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
