"""An ASR training recipe: tokens, augmentation, batches, and the schedule.

Fill in the eight functions, then press Run. Run plays a 400 Hz sine, the
same sine at speed 1.1, and the sine mixed with noise at 10 dB.
"""

from collections import Counter
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from speechlab import play, show_plot


def train_bpe(corpus: list[str], n_merges: int) -> list[tuple[str, str]]:
    """Learn `n_merges` byte-pair merges from `corpus`.

    Each string is one occurrence. Symbols start as characters. At each
    step, merge the adjacent symbol pair that occurs most often. Ties go
    to the lexicographically smaller pair. Returns the merges in order.
    """
    raise NotImplementedError("train_bpe")


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply `merges` in order and return the symbol list.

    Scan from the left. Join a pair only when both symbols are still
    separate, so one symbol joins at most once in a pass.
    """
    raise NotImplementedError("encode")


def decode(symbols: list[str]) -> str:
    """Join symbols back into the string they were split from."""
    raise NotImplementedError("decode")


def speed_perturb(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample `x` so it plays `factor` times as fast at the same rate.

    `factor` = p/q in lowest terms. The result has
    ceil(len(x) * q / p) samples, and every frequency is multiplied by
    `factor`. A factor above 1 shortens the clip.
    """
    raise NotImplementedError("speed_perturb")


def mix_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale `noise` so `clean + noise` has SNR `snr_db` decibels.

    Power is the mean square. `clean` and `noise` are 1-D and the same
    length. Returns `(mixture, scaled_noise)`.
    """
    raise NotImplementedError("mix_at_snr")


def bucket_batches(lengths: list[int], max_frames: int) -> list[list[int]]:
    """Pack indices into batches whose padded size stays within budget.

    Sort by length. A batch of n utterances whose longest has L frames
    uses n * L frames. Start a new batch when the next utterance would
    pass `max_frames`. One utterance longer than the budget is its own
    batch. Returns lists of indices into `lengths`.
    """
    raise NotImplementedError("bucket_batches")


def noam_lr(step: int, d_model: int, warmup: int) -> float:
    """Warmup then inverse-square-root learning rate at `step` >= 1.

    d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5).
    """
    raise NotImplementedError("noam_lr")


def average_checkpoints(state_dicts: list[dict]) -> dict:
    """Elementwise mean of each array, one key at a time.

    Every dict has the same keys. Values are arrays of the same shape.
    """
    raise NotImplementedError("average_checkpoints")


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
