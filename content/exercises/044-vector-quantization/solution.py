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
    rate, dim, K = 16000, 8, 64

    speech = _vowel_sequence(rate, 2.0)
    vectors = speech[: len(speech) // dim * dim].reshape(-1, dim)
    bits = np.log2(K)
    print(f"{len(vectors)} vectors of {dim} samples; {rate // dim} vectors per second, "
          f"{bits:.0f} bits each, so {rate / dim * bits / 1000:.0f} kbps")

    # Collapse and its fix: start every code far from the data.
    far = rng.normal(size=(K, dim)) * 0.01 + 3.0
    books = {}
    for restart in (False, True):
        cb = _train_stage(vectors, K, rng, restart=restart, init=far)
        books[restart] = cb
        idx, _ = nearest_codeword(vectors, cb)
        print(f"far init, dead-code restart={restart}: perplexity {codebook_perplexity(idx, K):.1f} of {K}")

    for restart, name in ((False, "without restart"), (True, "with restart")):
        _, q = nearest_codeword(vectors, books[restart])
        decoded = q.reshape(-1)
        print(f"Codebook {name}: SNR {_snr(decoded, speech[: decoded.size]):.1f} dB")
    _, q = nearest_codeword(vectors, books[True])
    play(speech, rate, label="The original voice")
    play(q.reshape(-1), rate, label="One 64-code codebook with restart (12 kbps)")
    idx_bad, _ = nearest_codeword(vectors, books[False])
    idx_good, _ = nearest_codeword(vectors, books[True])
    show_plot(
        (np.arange(K), np.bincount(idx_bad, minlength=K) / len(idx_bad)),
        (np.arange(K), np.bincount(idx_good, minlength=K) / len(idx_good)),
        labels=["no restart", "with restart"],
        label="How often each code is used",
        xlabel="Code index",
        ylabel="Fraction of vectors",
    )
