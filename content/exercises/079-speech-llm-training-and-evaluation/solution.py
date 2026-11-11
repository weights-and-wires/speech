"""Reference solution for the speech LLM training and evaluation exercise."""

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
    head = [TASK_TAGS[task]] + [AUDIO_ID] * audio_len + list(prompt_ids)
    ids = np.array(head + list(response_ids) + [EOS_ID], dtype=np.int64)
    mask = np.zeros(len(ids), dtype=bool)
    mask[len(head):] = True
    return ids, mask


def make_schedule(n_words, p_switch, seed):
    """A modality for each word: "text" or "speech".

    The first word's modality is a fair coin flip. Before each later word, the
    modality switches with probability p_switch. Uses np.random.default_rng(seed).
    """
    rng = np.random.default_rng(seed)
    current = "speech" if rng.random() < 0.5 else "text"
    schedule = []
    for i in range(n_words):
        if i > 0 and rng.random() < p_switch:
            current = "text" if current == "speech" else "speech"
        schedule.append(current)
    return schedule


def interleave(words, schedule):
    """Render (word, n_units) pairs as one mixed token stream.

    A text word becomes its string. A speech word becomes n_units tokens
    "u:WORD:0", "u:WORD:1", and so on. The marker "[TEXT]" or "[SPEECH]" comes
    before the first word and at every change of modality, never inside a word.
    """
    tokens, previous = [], None
    for (word, n_units), modality in zip(words, schedule):
        if modality != previous:
            tokens.append("[TEXT]" if modality == "text" else "[SPEECH]")
            previous = modality
        if modality == "text":
            tokens.append(word)
        else:
            tokens.extend(f"u:{word}:{j}" for j in range(n_units))
    return tokens


def speech_fraction(schedule):
    """The fraction of words whose modality is "speech"."""
    return float(np.mean([m == "speech" for m in schedule]))


def modality_gap(text_scores, speech_scores, n_boot=2000, seed=0):
    """Per-item gap (text minus speech), its mean, and a paired bootstrap 95% CI.

    Returns (gaps, mean, low, high). Resample item indices once per draw and use
    them for both modalities. low and high are the 2.5th and 97.5th percentiles
    of the resampled means. Uses np.random.default_rng(seed).
    """
    gaps = np.asarray(text_scores, dtype=float) - np.asarray(speech_scores, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(gaps), size=(n_boot, len(gaps)))
    means = gaps[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return gaps, float(gaps.mean()), float(low), float(high)


def win_rate(judgments, n_boot=2000, seed=0):
    """Win rate of system A over system B, with a bootstrap 95% CI.

    judgments holds "win", "tie", or "loss" for A. A tie counts as half a win and
    stays in the denominator. Returns (rate, low, high). Uses np.random.default_rng(seed).
    """
    value = {"win": 1.0, "tie": 0.5, "loss": 0.0}
    scores = np.array([value[j] for j in judgments])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(scores), size=(n_boot, len(scores)))
    means = scores[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(scores.mean()), float(low), float(high)


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
