"""Speech LLM training and evaluation: build audio instruction data, interleave
text and speech, and measure a modality gap and a win rate with bootstrap intervals.

Fill in each function, then press Run.
"""

import numpy as np

AUDIO_ID = 1
EOS_ID = 2
TASK_TAGS = {"asr": 10, "translate": 11, "qa": 12, "caption": 13}


def build_audio_instruction(task, audio_len, prompt_ids, response_ids):
    """Build one training sequence and its loss mask.

    Layout: [task tag] + [AUDIO_ID] * audio_len + prompt_ids + response_ids + [EOS_ID].
    Returns (ids, mask): ids is an int64 array, mask is a bool array of the same length
    that is True only for the response tokens and the final EOS_ID.
    """
    raise NotImplementedError("build_audio_instruction")


def make_schedule(n_words, p_switch, seed):
    """A modality for each word: "text" or "speech".

    The first word's modality is a fair coin flip. Before each later word, the
    modality switches with probability p_switch. Uses np.random.default_rng(seed).
    """
    raise NotImplementedError("make_schedule")


def interleave(words, schedule):
    """Render (word, n_units) pairs as one mixed token stream.

    A text word becomes its string. A speech word becomes n_units tokens
    "u:WORD:0", "u:WORD:1", and so on. The marker "[TEXT]" or "[SPEECH]" comes
    before the first word and at every change of modality, never inside a word.
    """
    raise NotImplementedError("interleave")


def speech_fraction(schedule):
    """The fraction of words whose modality is "speech"."""
    raise NotImplementedError("speech_fraction")


def modality_gap(text_scores, speech_scores, n_boot=2000, seed=0):
    """Per-item gap (text minus speech), its mean, and a paired bootstrap 95% CI.

    Returns (gaps, mean, low, high). Resample item indices once per draw and use
    them for both modalities. low and high are the 2.5th and 97.5th percentiles
    of the resampled means. Uses np.random.default_rng(seed).
    """
    raise NotImplementedError("modality_gap")


def win_rate(judgments, n_boot=2000, seed=0):
    """Win rate of system A over system B, with a bootstrap 95% CI.

    judgments holds "win", "tie", or "loss" for A. A tie counts as half a win and
    stays in the denominator. Returns (rate, low, high). Uses np.random.default_rng(seed).
    """
    raise NotImplementedError("win_rate")


if __name__ == "__main__":
    ids, mask = build_audio_instruction("asr", 3, [7, 8], [20, 21])
    print("ids ", ids.tolist())
    print("mask", mask.astype(int).tolist())
    words = [("the", 2), ("cat", 3), ("sat", 2), ("down", 3)]
    schedule = make_schedule(len(words), 0.5, 3)
    print(schedule)
    print(interleave(words, schedule))
    gaps, mean, low, high = modality_gap([0.82, 0.74, 0.90, 0.66], [0.70, 0.71, 0.78, 0.50])
    print("gaps", np.round(gaps, 2), "mean", round(mean, 4), "CI", round(low, 3), round(high, 3))
    print("win rate", win_rate(["win"] * 9 + ["tie"] * 6 + ["loss"] * 5))
