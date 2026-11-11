"""Anti-spoofing and watermarking: the worked solution."""

import numpy as np
from scipy.fft import dct
from scipy.signal import butter, filtfilt

from speechlab import play, show_plot

RATE = 16000


# Provided: scoring and mixing helpers, from the earlier chapters -------------------


def error_rates(bonafide, spoof, thresholds):
    """FRR and FAR at each threshold. A trial is accepted when its score is at or above it."""
    bonafide, spoof, thresholds = (np.asarray(v, float) for v in (bonafide, spoof, thresholds))
    frr = np.searchsorted(np.sort(bonafide), thresholds, side="left") / bonafide.size
    far = 1.0 - np.searchsorted(np.sort(spoof), thresholds, side="left") / spoof.size
    return frr, far


def equal_error_rate(bonafide, spoof):
    """Interpolated EER of a detector whose higher scores mean "bona fide". Returns (eer, threshold)."""
    bonafide, spoof = np.asarray(bonafide, float), np.asarray(spoof, float)
    th = np.unique(np.concatenate([bonafide, spoof]))
    th = np.append(th, th[-1] + 1.0)
    frr, far = error_rates(bonafide, spoof, th)
    i = int(np.argmax(frr >= far))
    if i == 0 or frr[i] == far[i]:
        return float((frr[i] + far[i]) / 2), float(th[i])
    d0, d1 = far[i - 1] - frr[i - 1], far[i] - frr[i]
    w = d0 / (d0 - d1)
    return float(frr[i - 1] + w * (frr[i] - frr[i - 1])), float(th[i - 1] + w * (th[i] - th[i - 1]))


def mix_at_snr(clean, noise, snr_db):
    """Scale noise so that clean + noise has the requested SNR in dB. Returns the mixture."""
    gain = np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10 ** (snr_db / 10)))
    return clean + gain * noise


def power_spectrogram(x, n_fft=512, hop=160):
    """Hann-windowed power spectrogram, shape (1 + n_fft // 2, frames). Provided."""
    n = 1 + (len(x) - n_fft) // hop
    frames = x[np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]] * np.hanning(n_fft)
    return np.abs(np.fft.rfft(frames, axis=1)).T ** 2


# LFCC ------------------------------------------------------------------------------


def lfcc_filterbank(n_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Triangular filters with centers equally spaced in hertz.

    The n_filters + 2 edges run from 0 Hz to sample_rate / 2 in equal steps.
    Filter m rises from edge m to edge m + 1 and falls to edge m + 2, with a
    peak of 1. Returns (n_filters, 1 + n_fft // 2), one column per rfft bin.
    """
    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = np.linspace(0.0, sample_rate / 2, n_filters + 2)
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0.0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))


def lfcc(power: np.ndarray, filterbank: np.ndarray, n_ceps: int = 20) -> np.ndarray:
    """Linear-frequency cepstral coefficients, shape (n_ceps, frames).

    power: (bins, frames). Apply the filterbank, take the natural log of
    (energy + 1e-10), then an orthonormal DCT-II down the filter axis and keep
    the first n_ceps rows.
    """
    energies = filterbank @ power
    return dct(np.log(energies + 1e-10), type=2, axis=0, norm="ortho")[:n_ceps]


# Tandem detection cost ----------------------------------------------------------------


def tdcf_costs(
    asv_pmiss: float,
    asv_pfa: float,
    asv_pmiss_spoof: float,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
    p_spoof: float = 0.05,
    c_miss_asv: float = 1.0,
    c_fa_asv: float = 10.0,
    c_miss_cm: float = 1.0,
    c_fa_cm: float = 10.0,
) -> tuple[float, float]:
    """The weights (C1, C2) of the ASVspoof 2019 tandem detection cost.

    t-DCF(threshold) = C0 + C1 * Pmiss_cm + C2 * Pfa_cm, where
      C1 = p_tar * (c_miss_cm - c_miss_asv * asv_pmiss) - p_non * c_fa_asv * asv_pfa
      C2 = c_fa_cm * p_spoof * (1 - asv_pmiss_spoof)
    and C0 does not depend on the countermeasure, so it is dropped.
    """
    c1 = p_tar * (c_miss_cm - c_miss_asv * asv_pmiss) - p_non * c_fa_asv * asv_pfa
    c2 = c_fa_cm * p_spoof * (1.0 - asv_pmiss_spoof)
    return float(c1), float(c2)


def min_tdcf(bonafide: np.ndarray, spoof: np.ndarray, c1: float, c2: float) -> tuple[float, float]:
    """The minimum normalized t-DCF over all thresholds, and the threshold that gives it.

    Sweep every distinct score plus one value above the largest. At each
    threshold the cost is (c1 * FRR + c2 * FAR) / min(c1, c2): normalizing by
    min(c1, c2) is the cost of the better trivial countermeasure, one that
    accepts everything or rejects everything. Returns (cost, threshold),
    taking the lowest threshold on a tie.
    """
    bonafide, spoof = np.asarray(bonafide, float), np.asarray(spoof, float)
    th = np.unique(np.concatenate([bonafide, spoof]))
    th = np.append(th, th[-1] + 1.0)
    frr, far = error_rates(bonafide, spoof, th)
    cost = (c1 * frr + c2 * far) / min(c1, c2)
    i = int(np.argmin(cost))
    return float(cost[i]), float(th[i])


# Spread-spectrum watermark -----------------------------------------------------------


def chip_sequences(key: int, n_bits: int, length: int) -> np.ndarray:
    """One seeded +1/-1 chip sequence per bit, shape (n_bits, length). Provided."""
    return np.random.default_rng(key).choice([-1.0, 1.0], size=(n_bits, length))


def embed_watermark(x: np.ndarray, bits: np.ndarray, key: int = 0, strength: float = 0.01) -> np.ndarray:
    """Add one chip sequence per bit, with sign +1 for a 1 and -1 for a 0.

    Split x into len(bits) equal segments of length len(x) // len(bits). In
    segment i add strength * (2 * bits[i] - 1) * chips[i], with chips from
    chip_sequences(key, ...). Samples past the last whole segment are left
    alone. Returns a new float array.
    """
    bits = np.asarray(bits)
    seg = len(x) // len(bits)
    chips = chip_sequences(key, len(bits), seg)
    mark = np.zeros(len(x))
    mark[: seg * len(bits)] = ((2.0 * bits - 1.0)[:, None] * strength * chips).ravel()
    return np.asarray(x, float) + mark


def detect_watermark(y: np.ndarray, n_bits: int, key: int = 0) -> np.ndarray:
    """Recover n_bits bits by correlating each segment with its chip sequence.

    Subtract each segment's mean first, so a constant offset in the host does
    not leak into the correlation. A bit is 1 when the correlation is positive.
    Returns an int array of length n_bits.
    """
    seg = len(y) // n_bits
    parts = np.asarray(y, float)[: seg * n_bits].reshape(n_bits, seg)
    parts = parts - parts.mean(axis=1, keepdims=True)
    corr = np.sum(parts * chip_sequences(key, n_bits, seg), axis=1)
    return (corr > 0).astype(int)


# Anonymization -------------------------------------------------------------------------


def anonymize_embeddings(vectors: np.ndarray, pool: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Replace every speaker vector with the mean of k pool vectors.

    vectors: (N, d). pool: (P, d) of pseudo-speaker vectors. For each row, draw
    k distinct pool rows with rng.choice(P, size=k, replace=False), in row
    order, and average them. The result does not depend on `vectors` except for
    its shape. Returns (N, d).
    """
    out = np.empty_like(np.asarray(vectors, float))
    for i in range(len(out)):
        out[i] = pool[rng.choice(len(pool), size=k, replace=False)].mean(axis=0)
    return out


def attacker_eer(enrolled: np.ndarray, trials: np.ndarray) -> float:
    """The EER of an attacker who links trial vectors to enrolled speakers.

    enrolled and trials: (N, d), where row i of each is speaker i. The score of
    a pair is the cosine similarity of enrolled[i] and trials[j]. Pairs with
    i == j are target trials. Pairs with i != j are non-target trials, and the
    diagonal is not one of them. Returns the interpolated EER. 0.5 means the
    attacker does no better than a coin.
    """
    a = enrolled / np.linalg.norm(enrolled, axis=1, keepdims=True)
    b = trials / np.linalg.norm(trials, axis=1, keepdims=True)
    scores = a @ b.T
    off_diagonal = ~np.eye(len(scores), dtype=bool)
    return equal_error_rate(np.diag(scores), scores[off_diagonal])[0]


# Run helpers -------------------------------------------------------------------------


def host_signal(seconds=1.0, f0=140.0, seed=0):
    """A voiced-like host: harmonics with falling amplitude and a small noise floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h for h in range(1, 30))
    return 0.2 * x / np.abs(x).max() + 0.002 * rng.standard_normal(len(t))


def lowpass(y, cutoff_hz):
    b, a = butter(6, cutoff_hz / (RATE / 2))
    return filtfilt(b, a, y)


def watermark_demo():
    rng = np.random.default_rng(1)
    host = host_signal()
    bits = rng.integers(0, 2, 32)
    marked = embed_watermark(host, bits, key=7)
    inaudible = 10 * np.log10(np.sum(host**2) / np.sum((marked - host) ** 2))
    print(f"Watermark: 32 bits, key 7. Host-to-watermark ratio {inaudible:.1f} dB.")
    attacks = {
        "no attack": marked,
        "noise at 20 dB": mix_at_snr(marked, rng.standard_normal(len(marked)), 20.0),
        "noise at 0 dB": mix_at_snr(marked, rng.standard_normal(len(marked)), 0.0),
        "low-pass at 4 kHz": lowpass(marked, 4000),
        "low-pass at 1 kHz": lowpass(marked, 1000),
    }
    for name, y in attacks.items():
        errors = int(np.sum(detect_watermark(y, 32, key=7) != bits))
        print(f"  {name:<20} bit errors: {errors} of 32")
    print(f"  wrong key           bit errors: {int(np.sum(detect_watermark(marked, 32, key=8) != bits))} of 32")
    play(marked, RATE, label="Watermarked host")


def anonymization_demo():
    rng = np.random.default_rng(0)
    n, d = 60, 16
    speakers = rng.standard_normal((n, d))
    enrolled = speakers + 0.5 * rng.standard_normal((n, d))
    trials = speakers + 0.5 * rng.standard_normal((n, d))
    pool = rng.standard_normal((300, d))
    print(f"\nAttacker EER on {n} speakers:")
    print(f"  no anonymization          {attacker_eer(enrolled, trials):.1%}")
    swapped = anonymize_embeddings(trials, pool, 20, rng)
    print(f"  swap, enrolled originals  {attacker_eer(enrolled, swapped):.1%}")
    fixed = anonymize_embeddings(speakers, pool, 20, rng)
    linked = attacker_eer(fixed + 0.1 * rng.standard_normal((n, d)), fixed + 0.1 * rng.standard_normal((n, d)))
    print(f"  one fixed pseudo-speaker per person, attacker enrolls anonymized speech: {linked:.1%}")


if __name__ == "__main__":
    watermark_demo()
    anonymization_demo()
    fb = lfcc_filterbank(20, 512, RATE)
    ceps = lfcc(power_spectrogram(host_signal()), fb)
    print(f"\nLFCC of the host: {ceps.shape[0]} coefficients x {ceps.shape[1]} frames, c0 mean {ceps[0].mean():.1f}")
    show_plot(fb[0], fb[5], fb[10], fb[15], label="Four of the 20 linear filters", xlabel="FFT bin", ylabel="Weight")
