"""Numbers behind the figures in 079-speech-llm-training-and-evaluation.

Figure 1 is the loss mask of one ASR sample. Figure 2 is the bootstrap
distribution of a win rate with ties counted as half. Prints every value drawn.
"""

import numpy as np

# Figure 1: layout of build_audio_instruction("asr", 3, [7, 8], [20, 21]).
ids = [10, 1, 1, 1, 7, 8, 20, 21, 2]
n_head = 1 + 3 + 2
mask = [int(i >= n_head) for i in range(len(ids))]
print("ids ", ids)
print("mask", mask, "trained", sum(mask), "of", len(ids))

# Figure 2: 9 wins, 6 ties, 5 losses over 20 judgments.
scores = np.array([1.0] * 9 + [0.5] * 6 + [0.0] * 5)
rng = np.random.default_rng(0)
picks = rng.integers(0, len(scores), size=(2000, len(scores)))
means = scores[picks].mean(axis=1)
low, high = np.percentile(means, [2.5, 97.5])
print("win rate", scores.mean(), "ties dropped", 9 / 14)
print("95% CI", round(float(low), 3), round(float(high), 3))
edges = np.linspace(0.3, 0.9, 13)
counts, _ = np.histogram(means, bins=edges)
for lo, hi, c in zip(edges[:-1], edges[1:], counts):
    print(f"bin {lo:.2f}-{hi:.2f}: {c}")
