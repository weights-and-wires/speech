"""Checks for the speech LLM training and evaluation exercise.

The plausible wrong implementations these checks catch:
- A loss mask that also trains on the audio placeholders or the prompt
  (caught by "The audio placeholders, task tag, and prompt never take loss").
- A win rate that drops ties from the denominator, or counts them as losses
  (caught by "Ties count as half a win and stay in the denominator").
- A modality gap bootstrap that resamples the two modalities independently
  (caught by "The gap bootstrap resamples items in pairs").
"""

import numpy as np

from speechcheck import almost, check, close, equal, fail, has

AUDIO_ID, EOS_ID = 1, 2
TAGS = {"asr": 10, "translate": 11, "qa": 12, "caption": 13}


@check("build_audio_instruction lays out tag, audio, prompt, response, and EOS, at any audio length")
def test_layout(sub):
    build = has(sub, "build_audio_instruction")
    ids, mask = build("asr", 3, [7, 8], [20, 21])
    equal(np.asarray(ids).tolist(), [10, 1, 1, 1, 7, 8, 20, 21, 2], label="ids")
    equal(len(mask), 9, label="mask length")
    ids, _ = build("caption", 2, [5], [30])
    equal(np.asarray(ids).tolist(), [13, 1, 1, 5, 30, 2], label="caption ids")
    for task in TAGS:
        for audio_len in (0, 1, 12):
            ids, mask = build(task, audio_len, [7, 8], [20, 21, 22])
            equal(len(ids), 1 + audio_len + 2 + 3 + 1, label=f"{task} length at audio_len={audio_len}")
            equal(int(np.sum(mask)), 4, label=f"{task} trained tokens at audio_len={audio_len}")
            equal(int(ids[0]), TAGS[task], label="task tag")


@check("The audio placeholders, task tag, and prompt never take loss")
def test_mask_excludes_inputs(sub):
    build = has(sub, "build_audio_instruction")
    ids, mask = build("qa", 4, [7, 8, 9], [20, 21])
    ids, mask = np.asarray(ids), np.asarray(mask, dtype=bool)
    n_input = 1 + 4 + 3
    if mask[:n_input].any():
        fail(f"positions 0 to {n_input - 1} (tag, audio, prompt) must be masked out, "
             f"got mask {mask.astype(int).tolist()}")
    if mask[ids == AUDIO_ID].any():
        fail("an audio placeholder position has a loss mask of 1")


@check("The response tokens and the final EOS take loss")
def test_mask_keeps_response(sub):
    build = has(sub, "build_audio_instruction")
    ids, mask = build("qa", 4, [7, 8, 9], [20, 21])
    want = [0] * 8 + [1, 1, 1]
    equal(np.asarray(mask, dtype=bool).astype(int).tolist(), want, label="mask")
    if np.asarray(ids)[-1] != EOS_ID:
        fail(f"the last id must be EOS {EOS_ID}, got {np.asarray(ids)[-1]}")


@check("interleave switches only at word boundaries and marks every switch")
def test_interleave(sub):
    interleave = has(sub, "interleave")
    words = [("the", 2), ("cat", 3), ("sat", 2), ("down", 3)]
    got = interleave(words, ["text", "text", "speech", "speech"])
    want = ["[TEXT]", "the", "cat", "[SPEECH]", "u:sat:0", "u:sat:1", "u:down:0", "u:down:1", "u:down:2"]
    equal(list(got), want, label="stream")
    got = interleave(words, ["speech", "text", "speech", "text"])
    want = ["[SPEECH]", "u:the:0", "u:the:1", "[TEXT]", "cat", "[SPEECH]", "u:sat:0", "u:sat:1", "[TEXT]", "down"]
    equal(list(got), want, label="alternating stream")


@check("make_schedule is deterministic and its switch rate follows p_switch")
def test_schedule(sub):
    make = has(sub, "make_schedule")
    equal(list(make(50, 0.3, 4)), list(make(50, 0.3, 4)), label="same seed")
    if list(make(50, 0.3, 4)) == list(make(50, 0.3, 5)):
        fail("seeds 4 and 5 gave the same 50-word schedule; the seed is not being used")
    if len(set(make(30, 0.0, 1))) != 1:
        fail("with p_switch 0 every word must have one modality")
    alt = make(10, 1.0, 1)
    if any(a == b for a, b in zip(alt, alt[1:])):
        fail(f"with p_switch 1 the modalities must alternate, got {alt}")
    s = make(4000, 0.25, 0)
    rate = np.mean([a != b for a, b in zip(s, s[1:])])
    almost(rate, 0.25, 0.03, label="observed switch rate")


@check("speech_fraction is the share of speech words, near one half over many seeds")
def test_fraction(sub):
    make, frac = has(sub, "make_schedule"), has(sub, "speech_fraction")
    almost(frac(["speech", "text", "speech", "speech"]), 0.75, 1e-12, label="hand fraction")
    fractions = [frac(make(200, 0.2, seed)) for seed in range(100)]
    almost(float(np.mean(fractions)), 0.5, 0.05, label="mean speech fraction over 100 seeds")


@check("modality_gap gives the per-item and mean gap on the worked scores")
def test_gap_values(sub):
    gap = has(sub, "modality_gap")
    gaps, mean, low, high = gap([0.82, 0.74, 0.90, 0.66], [0.70, 0.71, 0.78, 0.50])
    close(gaps, [0.12, 0.03, 0.12, 0.16], label="per-item gaps")
    almost(mean, 0.1075, 1e-12, label="mean gap")
    if not (low <= mean <= high):
        fail(f"the CI [{low:.4f}, {high:.4f}] must contain the mean {mean:.4f}")
    if not (0.12 - 1e-9 >= low >= 0.03 and 0.03 <= high <= 0.16 + 1e-9):
        fail(f"the CI [{low:.4f}, {high:.4f}] must lie inside the range of the gaps, 0.03 to 0.16")


@check("The gap bootstrap resamples items in pairs")
def test_gap_paired(sub):
    gap = has(sub, "modality_gap")
    rng = np.random.default_rng(0)
    speech = rng.uniform(0.2, 0.9, size=40)
    text = speech + 0.15  # every item has exactly the same gap, though the scores vary a lot
    _, mean, low, high = gap(text, speech)
    almost(mean, 0.15, 1e-9, label="mean gap")
    if high - low > 1e-9:
        fail(f"every item has a gap of 0.15, so a paired bootstrap has zero width, "
             f"got [{low:.4f}, {high:.4f}]; resample item indices once and apply them to the gaps")
    a = gap(text[:10], speech[:10], seed=1)[1:]
    b = gap(text[:10], speech[:10], seed=1)[1:]
    equal(a, b, label="same seed, same result")


@check("Ties count as half a win and stay in the denominator")
def test_win_rate_ties(sub):
    win = has(sub, "win_rate")
    rate, low, high = win(["win"] * 9 + ["tie"] * 6 + ["loss"] * 5)
    almost(rate, 0.6, 1e-12, label="win rate (9 wins, 6 ties, 5 losses; dropping ties gives 0.643)")
    rate, _, _ = win(["tie"] * 4)
    almost(rate, 0.5, 1e-12, label="all ties")
    rate, _, _ = win(["win", "loss", "tie", "tie"])
    almost(rate, 0.5, 1e-12, label="one win, one loss, two ties")


@check("The win-rate CI contains the rate, has the expected width, and flips under swapping A and B")
def test_win_rate_ci(sub):
    win = has(sub, "win_rate")
    judgments = ["win"] * 40 + ["tie"] * 20 + ["loss"] * 40
    rate, low, high = win(judgments)
    almost(rate, 0.5, 1e-12, label="rate")
    if not (low < rate < high):
        fail(f"the CI [{low:.4f}, {high:.4f}] must contain the rate {rate}")
    # Scores 1, 0.5, 0 with shares 40%, 20%, 40% have variance 0.2, so the standard
    # error of the mean over 100 judgments is sqrt(0.2 / 100) and a 95% CI is 3.92 of those wide.
    almost(high - low, 3.92 * np.sqrt(0.2 / 100), 0.03, label="CI width")
    swap = {"win": "loss", "loss": "win", "tie": "tie"}
    r2, l2, h2 = win([swap[j] for j in judgments])
    close([r2, l2, h2], [1 - rate, 1 - high, 1 - low], atol=1e-9, label="swapped (rate, low, high)")
    equal(win(["win"] * 10)[1:], (1.0, 1.0), label="all wins CI")
