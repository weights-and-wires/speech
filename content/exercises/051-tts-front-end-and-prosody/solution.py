"""The TTS front end and prosody controls: the worked solution."""

import numpy as np
import torch

from speechlab import play, show_plot

FRAME_SECONDS = 0.01


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ends = torch.cumsum(durations, dim=1)
    starts = ends - durations
    frames = torch.arange(int(ends[:, -1].max()))[None, None, :]
    align = (starts[..., None] <= frames) & (frames < ends[..., None])
    out = torch.einsum("bln,bld->bnd", align.to(h.dtype), h)
    return out, frames[:, 0, :] < ends[:, -1:]


def spell_out(word: str) -> str:
    return " ".join(c.upper() for c in word if c.isalpha())


def semitones_to_scale(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def scale_durations(durations: np.ndarray, rate: float) -> np.ndarray:
    return np.maximum(1, np.round(np.asarray(durations) / rate)).astype(int)


def choose_pronunciation(word: str, prev: str | None, nxt: str | None, entry: dict) -> str:
    for rule in entry["rules"]:
        if "prev" in rule and (prev is None or prev.lower() not in rule["prev"]):
            continue
        if "next" in rule and (nxt is None or nxt.lower() not in rule["next"]):
            continue
        return rule["pron"]
    return entry["default"]


def g2p_with_homographs(words, lexicon, homographs, fallback=spell_out):
    out = []
    for i, word in enumerate(words):
        key = word.lower()
        prev = words[i - 1] if i > 0 else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        if key in homographs:
            out.append(choose_pronunciation(word, prev, nxt, homographs[key]))
        elif key in lexicon:
            out.append(lexicon[key])
        else:
            out.append(fallback(word))
    return out


def style_attention(ref: torch.Tensor, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tanh(tokens) @ ref / tokens.shape[1] ** 0.5
    weights = torch.softmax(logits, dim=0)
    return weights @ tokens, weights


def variance_adaptor(
    h, durations, pitch, energy, pitch_table, energy_table,
    pitch_range=(71.0, 795.0), energy_range=(0.0, 8.0), pitch_scale=1.0, rate=1.0,
):
    n_bins = pitch_table.shape[0]
    p_edges = torch.exp(torch.linspace(np.log(pitch_range[0]), np.log(pitch_range[1]), n_bins - 1, dtype=pitch.dtype))
    e_edges = torch.linspace(energy_range[0], energy_range[1], n_bins - 1, dtype=energy.dtype)
    pitch_bins = torch.bucketize(pitch * pitch_scale, p_edges)
    energy_bins = torch.bucketize(energy, e_edges)
    h = h + pitch_table[pitch_bins] + energy_table[energy_bins]
    d = torch.as_tensor(scale_durations(durations.numpy(), rate)) * (durations > 0).long()
    frames, mask = length_regulate(h, d)
    return frames, mask, pitch_bins


if __name__ == "__main__":
    lexicon = {"i": "AY1", "have": "HH AE1 V", "will": "W IH1 L", "the": "DH AH0", "to": "T UW1"}
    homographs = {
        "read": {"default": "R IY1 D", "rules": [{"prev": {"have", "had"}, "pron": "R EH1 D"}]},
        "record": {"default": "R EH1 K ER0 D", "rules": [{"prev": {"to", "will"}, "pron": "R IH0 K AO1 R D"}]},
    }
    for sentence in ("I have read the record", "I will read to record"):
        words = sentence.split()
        print(sentence, "->", " | ".join(g2p_with_homographs(words, lexicon, homographs)))

    rng = np.random.default_rng(0)
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
