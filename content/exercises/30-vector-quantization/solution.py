"""Vector quantization: the worked solution."""

import numpy as np
import scipy.signal
import torch

from speechlab import play, show_plot


# Nearest codeword ---------------------------------------------------------


def nearest_codeword(z: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The index and value of the closest codeword to each row of `z`."""
    distances = (
        (z**2).sum(axis=1, keepdims=True)
        - 2.0 * z @ codebook.T
        + (codebook**2).sum(axis=1)[None, :]
    )
    indices = np.argmin(distances, axis=1)
    return indices, codebook[indices]


# Gradients through the argmin ---------------------------------------------


def straight_through(z_e: torch.Tensor, z_q: torch.Tensor) -> torch.Tensor:
    """A tensor whose value is `z_q` and whose gradient flows to `z_e` unchanged."""
    return z_e + (z_q - z_e).detach()


def vq_losses(z_e: torch.Tensor, codebook: torch.Tensor):
    """Quantize `z_e` and return (z_q_st, indices, codebook_loss, commitment_loss)."""
    distances = (
        (z_e.detach() ** 2).sum(dim=1, keepdim=True)
        - 2.0 * z_e.detach() @ codebook.detach().T
        + (codebook.detach() ** 2).sum(dim=1)[None, :]
    )
    indices = torch.argmin(distances, dim=1)
    e = codebook[indices]
    codebook_loss = ((z_e.detach() - e) ** 2).sum(dim=1).mean()
    commitment_loss = ((z_e - e.detach()) ** 2).sum(dim=1).mean()
    return straight_through(z_e, e), indices, codebook_loss, commitment_loss


# EMA codebook and dead-code restart ---------------------------------------


def ema_update(counts, sums, z, indices, decay=0.99, eps=1e-5):
    """One exponential-moving-average step. Returns (counts, sums, codebook)."""
    K = counts.shape[0]
    batch_counts = np.bincount(indices, minlength=K).astype(float)
    batch_sums = np.zeros_like(sums)
    np.add.at(batch_sums, indices, z)
    counts = decay * counts + (1.0 - decay) * batch_counts
    sums = decay * sums + (1.0 - decay) * batch_sums
    total = counts.sum()
    smoothed = (counts + eps) / (total + K * eps) * total
    return counts, sums, sums / smoothed[:, None]


def restart_dead_codes(codebook, counts, sums, z, threshold, rng):
    """Replace every code whose EMA count is below `threshold` with a random row of `z`."""
    codebook, counts, sums = codebook.copy(), counts.copy(), sums.copy()
    dead = np.flatnonzero(counts < threshold)
    if dead.size:
        picks = rng.choice(len(z), size=dead.size, replace=dead.size > len(z))
        codebook[dead] = z[picks]
        counts[dead] = threshold
        sums[dead] = z[picks] * threshold
    return codebook, counts, sums, int(dead.size)


def codebook_perplexity(indices: np.ndarray, K: int) -> float:
    """exp of the entropy of the code-usage histogram: K when uniform, 1 when collapsed."""
    p = np.bincount(indices, minlength=K) / len(indices)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# Residual vector quantization ---------------------------------------------


def rvq_encode(x: np.ndarray, codebooks: list[np.ndarray]):
    """Quantize `x` stage by stage. Returns (codes, reconstruction, residual_norms)."""
    residual = x.astype(float).copy()
    reconstruction = np.zeros_like(residual)
    codes = np.zeros((len(x), len(codebooks)), dtype=np.int64)
    norms = [np.linalg.norm(residual, axis=1)]
    for stage, codebook in enumerate(codebooks):
        idx, q = nearest_codeword(residual, codebook)
        codes[:, stage] = idx
        reconstruction += q
        residual -= q
        norms.append(np.linalg.norm(residual, axis=1))
    return codes, reconstruction, np.stack(norms, axis=1)


def rvq_decode(codes: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Sum the chosen codeword of each stage. `codes` may hold fewer stages than `codebooks`."""
    out = np.zeros((codes.shape[0], codebooks[0].shape[1]))
    for stage in range(codes.shape[1]):
        out += codebooks[stage][codes[:, stage]]
    return out


# Finite scalar quantization -----------------------------------------------


def fsq_quantize(z: torch.Tensor, levels: list[int]) -> torch.Tensor:
    """Bound each dimension with tanh, round it, and pass the gradient straight through."""
    half = torch.tensor([(L - 1) // 2 for L in levels], dtype=z.dtype)
    bounded = half * torch.tanh(z)
    return bounded + (torch.round(bounded) - bounded).detach()


def fsq_index(z_hat: np.ndarray, levels: list[int]) -> np.ndarray:
    """The integer index of each FSQ grid point, with the first dimension varying fastest."""
    levels = np.asarray(levels)
    digits = np.rint(np.asarray(z_hat)).astype(np.int64) + levels // 2
    basis = np.concatenate([[1], np.cumprod(levels[:-1])])
    return digits @ basis


def fsq_unindex(indices: np.ndarray, levels: list[int]) -> np.ndarray:
    """The inverse of fsq_index: grid points of shape (N, len(levels))."""
    levels = np.asarray(levels)
    basis = np.concatenate([[1], np.cumprod(levels[:-1])])
    digits = (np.asarray(indices)[:, None] // basis[None, :]) % levels[None, :]
    return (digits - levels // 2).astype(float)


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
