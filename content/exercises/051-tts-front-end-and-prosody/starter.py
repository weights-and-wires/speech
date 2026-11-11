"""The TTS front end and prosody controls.

Fill in the functions, then press Run. The run turns two sentences into phones
with a homograph rule layer, then plays a contour of six phone pitches twice:
once as given, and once four semitones higher and 1.25 times faster.

Frames are 10 ms long. Pitch is in hertz, durations are whole frames, and a
rate of 2 means twice as fast.
"""

import numpy as np
import torch

from speechlab import play, show_plot

FRAME_SECONDS = 0.01


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Provided: the length regulator from the previous chapter.

    Repeats h[b, i] durations[b, i] times. h has shape (B, L, D), durations
    has shape (B, L), and the result is (frames (B, N_max, D), mask (B, N_max)).
    """
    ends = torch.cumsum(durations, dim=1)
    starts = ends - durations
    frames = torch.arange(int(ends[:, -1].max()))[None, None, :]
    align = (starts[..., None] <= frames) & (frames < ends[..., None])
    out = torch.einsum("bln,bld->bnd", align.to(h.dtype), h)
    return out, frames[:, 0, :] < ends[:, -1:]


def spell_out(word: str) -> str:
    """Provided: the last-resort pronunciation, the word's letters, spaced and upper case."""
    return " ".join(c.upper() for c in word if c.isalpha())


def semitones_to_scale(semitones: float) -> float:
    """Return the pitch scale factor for a shift in semitones.

    12 semitones is one octave, so the factor is 2 ** (semitones / 12).
    """
    raise NotImplementedError("semitones_to_scale")


def scale_durations(durations: np.ndarray, rate: float) -> np.ndarray:
    """Change speaking rate by scaling whole-frame durations.

    Args:
        durations: Frames per phone, integers, shape (L,).
        rate: Speed factor. 2.0 is twice as fast.

    Returns:
        Integer array of shape (L,): round(durations / rate), but never below 1,
        so no phone disappears.
    """
    raise NotImplementedError("scale_durations")


def choose_pronunciation(word: str, prev: str | None, nxt: str | None, entry: dict) -> str:
    """Pick a homograph's pronunciation from its neighbors.

    Args:
        word: The homograph.
        prev: The word before it, or None at the start of the sentence.
        nxt: The word after it, or None at the end.
        entry: {"default": pron, "rules": [rule, ...]}. A rule is a dict with
            "pron" and any of "prev" and "next", each a set of lower-case words.
            A rule matches when every cue it names matches. Compare lower-case.
            A cue never matches when the neighbor is None.

    Returns:
        The pron of the first matching rule, or entry["default"].
    """
    raise NotImplementedError("choose_pronunciation")


def g2p_with_homographs(words, lexicon, homographs, fallback=spell_out):
    """Return one pronunciation string per word.

    Args:
        words: Normalized words of one sentence, original case.
        lexicon: Lower-case word -> pronunciation.
        homographs: Lower-case word -> entry for choose_pronunciation.
        fallback: Called with the word when it is in neither table.

    Homographs are checked first, then the lexicon, then fallback. A word's
    neighbors are the adjacent items of `words`.
    """
    raise NotImplementedError("g2p_with_homographs")


def style_attention(ref: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix a bank of style tokens by attending from a reference embedding.

    Args:
        ref: Reference-encoder output, shape (D,).
        tokens: The style token bank, shape (T, D).

    Returns:
        (style, weights). The keys are tanh(tokens), the logits are
        keys @ ref / sqrt(D), weights = softmax(logits) has shape (T,), and
        style = weights @ tokens has shape (D,). Must not overflow for large logits.
    """
    raise NotImplementedError("style_attention")


def variance_adaptor(
    h, durations, pitch, energy, pitch_table, energy_table,
    pitch_range=(71.0, 795.0), energy_range=(0.0, 8.0), pitch_scale=1.0, rate=1.0,
):
    """Add pitch and energy embeddings to phone states, then length-regulate.

    Args:
        h: Phone hidden states, (B, L, D), float.
        durations: Frames per phone, (B, L), long. Padded phones are 0.
        pitch: Per-phone pitch in hertz, (B, L). 0 means unvoiced.
        energy: Per-phone energy, (B, L).
        pitch_table, energy_table: Embedding tables, (K, D), K bins each.
        pitch_range: (v_min, v_max) in hertz. K - 1 edges, spaced evenly on a
            LOG scale.
        energy_range: (v_min, v_max). K - 1 edges, spaced evenly (linear).
        pitch_scale: Multiply the pitch by this before binning.
        rate: Speed factor applied to the durations with scale_durations;
            padded phones stay 0.

    Returns:
        (frames, mask, pitch_bins): frames (B, N, D), mask (B, N), and the
        pitch bin of each phone, (B, L), long. A value equal to an edge falls
        in the lower bin, as torch.bucketize does by default.
    """
    raise NotImplementedError("variance_adaptor")


if __name__ == "__main__":
    lexicon = {"i": "AY1", "have": "HH AE1 V", "will": "W IH1 L", "the": "DH AH0", "to": "T UW1"}
    homographs = {
        "read": {"default": "R IY1 D", "rules": [{"prev": {"have", "had"}, "pron": "R EH1 D"}]},
        "record": {"default": "R EH1 K ER0 D", "rules": [{"prev": {"to", "will"}, "pron": "R IH0 K AO1 R D"}]},
    }
    for sentence in ("I have read the record", "I will read to record"):
        words = sentence.split()
        print(sentence, "->", " | ".join(g2p_with_homographs(words, lexicon, homographs)))

    L, D, K = 6, 8, 256
    pitch = torch.tensor([180.0, 220.0, 200.0, 260.0, 240.0, 190.0])
    d = torch.tensor([[6, 9, 7, 10, 8, 12]])
    h = torch.zeros(1, L, D)
    pt, et = torch.zeros(K, D), torch.zeros(K, D)
    clips = {}
    for name, semis, rate in (("original", 0.0, 1.0), ("up 4 semitones, 1.25 times faster", 4.0, 1.25)):
        _, mask, bins = variance_adaptor(h, d, pitch[None], torch.ones(1, L), pt, et,
                                         pitch_scale=semitones_to_scale(semis), rate=rate)
        dur = scale_durations(d[0].numpy(), rate)
        f0 = np.repeat(pitch.numpy() * semitones_to_scale(semis), dur)
        phase = 2 * np.pi * np.cumsum(np.repeat(f0, 160)) / 16000
        clips[name] = 0.3 * np.sin(phase)
        print(f"{name}: {int(mask.sum())} frames, pitch bins {bins[0].tolist()}")
        play(clips[name], 16000, label=name)
    show_plot(*[(np.arange(len(c)) / 16000, c) for c in clips.values()], label="Both clips", xlabel="seconds")
