"""An ASR training recipe: the worked solution."""

from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from speechlab import play, show_plot


def _merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Join non-overlapping occurrences of `pair`, scanning from the left."""
    left, right = pair
    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if (
            index + 1 < len(symbols)
            and symbols[index] == left
            and symbols[index + 1] == right
        ):
            merged.append(left + right)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


def train_bpe(corpus: list[str], n_merges: int) -> list[tuple[str, str]]:
    """Learn `n_merges` byte-pair merges from `corpus`.

    Each string is one occurrence. Symbols start as characters. At each
    step, merge the adjacent symbol pair that occurs most often. Ties go
    to the lexicographically smaller pair. Returns the merges in order.
    """
    sequences = [list(text) for text in corpus]
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for seq in sequences:
            for left, right in zip(seq, seq[1:]):
                counts[(left, right)] += 1
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        merges.append(pair)
        sequences = [_merge_pair(seq, pair) for seq in sequences]
    return merges


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply `merges` in order and return the symbol list.

    Scan from the left. Join a pair only when both symbols are still
    separate, so one symbol joins at most once in a pass.
    """
    symbols = list(text)
    for pair in merges:
        symbols = _merge_pair(symbols, pair)
    return symbols


def decode(symbols: list[str]) -> str:
    """Join symbols back into the string they were split from."""
    return "".join(symbols)


def speed_perturb(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample `x` so it plays `factor` times as fast at the same rate.

    `factor` = p/q in lowest terms. The result has
    ceil(len(x) * q / p) samples, and every frequency is multiplied by
    `factor`. A factor above 1 shortens the clip.
    """
    samples = np.asarray(x, dtype=float)
    if factor <= 0:
        raise ValueError("factor must be positive")
    ratio = Fraction(float(factor)).limit_denominator(1000)
    up, down = ratio.denominator, ratio.numerator
    if up == down:
        return samples.copy()
    return np.asarray(resample_poly(samples, up, down), dtype=float)


def mix_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale `noise` so `clean + noise` has SNR `snr_db` decibels.

    Power is the mean square. `clean` and `noise` are 1-D and the same
    length. Returns `(mixture, scaled_noise)`.
    """
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noise, dtype=float)
    power_clean = np.mean(clean**2)
    power_noise = np.mean(noise**2)
    gain = np.sqrt(power_clean / (power_noise * 10 ** (snr_db / 10)))
    scaled = gain * noise
    return clean + scaled, scaled


def bucket_batches(lengths: list[int], max_frames: int) -> list[list[int]]:
    """Pack indices into batches whose padded size stays within budget.

    Sort by length. A batch of n utterances whose longest has L frames
    uses n * L frames. Start a new batch when the next utterance would
    pass `max_frames`. One utterance longer than the budget is its own
    batch. Returns lists of indices into `lengths`.
    """
    lengths = [int(length) for length in lengths]
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        trial = current + [index]
        longest = max(lengths[item] for item in trial)
        if current and len(trial) * longest > max_frames:
            batches.append(current)
            current = [index]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    """Warmup then inverse-square-root learning rate at `step` >= 1.

    d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5).
    """
    step_f = float(step)
    scale = float(d_model) ** -0.5
    decay = step_f ** -0.5
    ramp = step_f * float(warmup) ** -1.5
    return scale * min(decay, ramp)


def average_checkpoints(state_dicts: list[dict]) -> dict:
    """Elementwise mean of each array, one key at a time.

    Every dict has the same keys. Values are arrays of the same shape.
    """
    if not state_dicts:
        return {}
    count = len(state_dicts)
    averaged = {}
    for key in state_dicts[0]:
        total = np.zeros_like(np.asarray(state_dicts[0][key], dtype=float))
        for state in state_dicts:
            total = total + np.asarray(state[key], dtype=float)
        averaged[key] = total / count
    return averaged


if __name__ == "__main__":
    corpus = ["cat", "cat", "car", "car", "card"]
    merges = train_bpe(corpus, 3)
    print("merges:", merges)
    print("card ->", decode(encode("card", merges)))

    rate = 16000
    x = 0.2 * np.sin(2 * np.pi * 400 * np.arange(rate) / rate)
    fast = speed_perturb(x, 1.1)
    print(f"speed 1.1 length {len(fast)}")
    play(x, rate, label="400 Hz")
    play(fast, rate, label="Speed 1.1, near 440 Hz")

    rng = np.random.default_rng(0)
    mixed, _scaled = mix_at_snr(x, rng.standard_normal(x.shape), 10.0)
    play(mixed, rate, label="400 Hz mixed at 10 dB")

    steps = np.arange(1, 16001, 80)
    rates = np.array([noam_lr(int(step), 256, 4000) for step in steps])
    show_plot(
        (steps, rates),
        label="Noam learning rate, d_model 256, warmup 4000",
        xlabel="step",
        ylabel="learning rate",
    )
