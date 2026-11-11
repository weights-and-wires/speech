"""Checks for the TTS front end and prosody exercise."""

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, finite_difference, shape

LEXICON = {"i": "AY1", "have": "HH AE1 V", "will": "W IH1 L", "the": "DH AH0", "a": "AH0", "to": "T UW1", "pipe": "P AY1 P"}
HOMOGRAPHS = {
    "read": {"default": "R IY1 D", "rules": [
        {"prev": {"have", "had", "was"}, "pron": "R EH1 D"},
        {"next": {"yesterday", "earlier"}, "pron": "R EH1 D"},
    ]},
    "lead": {"default": "L IY1 D", "rules": [{"next": {"pipe", "paint"}, "pron": "L EH1 D"}]},
    "record": {"default": "R EH1 K ER0 D", "rules": [{"prev": {"to", "will", "can"}, "pron": "R IH0 K AO1 R D"}]},
}


@check("scale_durations divides by the rate, keeps every phone at one frame, and semitones_to_scale doubles per octave")
def test_rate_and_semitones(sub):
    got = sub.scale_durations(np.array([4, 1, 6]), 2.0)
    equal(got.tolist(), [2, 1, 3], label="durations [4, 1, 6] at rate 2 (0.5 rounds to 0, then clamps to 1)")
    equal(sub.scale_durations(np.array([4, 1, 6]), 1.0).tolist(), [4, 1, 6], label="rate 1")
    almost(sub.semitones_to_scale(12.0), 2.0, 1e-9, label="+12 semitones")
    almost(sub.semitones_to_scale(-12.0), 0.5, 1e-9, label="-12 semitones")
    almost(sub.semitones_to_scale(7.0), 1.4983070768766815, 1e-9, label="+7 semitones")


@check("Total frames never grow as the rate rises, and slower speech is longer")
def test_rate_monotone(sub):
    # Plausible wrong implementation: multiplying by the rate, which makes fast speech long.
    d = np.random.default_rng(0).integers(1, 20, size=40)
    totals = [int(sub.scale_durations(d, r).sum()) for r in (0.5, 0.75, 1.0, 1.5, 2.0, 4.0)]
    if any(b > a for a, b in zip(totals, totals[1:])):
        fail(f"total frames at rates 0.5, 0.75, 1, 1.5, 2, 4 are {totals}, expected a non-increasing sequence")
    if not totals[0] > totals[2] > totals[-1]:
        fail(f"expected slow > normal > fast, got totals {totals}")
    if sub.scale_durations(d, 50.0).min() < 1:
        fail("a phone dropped below one frame at rate 50")


@check("style_attention weights sum to one, pick the planted token, and give the hand-worked numbers")
def test_style_hand(sub):
    tokens = 3.0 * torch.eye(4, dtype=torch.float64)
    ref = torch.tensor([0.0, 0.0, 4.0, 0.0], dtype=torch.float64)
    style, w = sub.style_attention(ref, tokens)
    shape(w, (4,), label="weights")
    shape(style, (4,), label="style")
    almost(float(w.sum()), 1.0, 1e-9, label="weights sum")
    equal(int(torch.argmax(w)), 2, label="strongest token")
    close(w, [0.0969, 0.0969, 0.7092, 0.0969], atol=5e-4, label="weights (logit 4 * tanh(3) / 2 = 1.990 on token 2)")
    close(style, [0.291, 0.291, 2.128, 0.291], atol=2e-3, label="style vector")


@check("style_attention survives huge logits and follows a rotated reference")
def test_style_stable(sub):
    # Plausible wrong implementation: a softmax without subtracting the maximum, which overflows.
    tokens = 3.0 * torch.eye(4)
    _, w = sub.style_attention(torch.tensor([0.0, 1000.0, 0.0, 0.0]), tokens)
    if not torch.isfinite(w).all():
        fail(f"weights are {w.tolist()} for a large reference; expected finite values")
    equal(int(torch.argmax(w)), 1, label="strongest token for a reference on token 1")
    _, w = sub.style_attention(torch.tensor([0.0, 0.0, 0.0, -5.0]), tokens)
    equal(int(torch.argmin(w)), 3, label="weakest token for a reference opposite token 3")


@check("choose_pronunciation uses the previous word, the next word, and only fires when every cue matches")
def test_choose(sub):
    entry = {"default": "A", "rules": [{"prev": {"a"}, "next": {"b"}, "pron": "B"}]}
    got = [sub.choose_pronunciation("x", p, n, entry) for p, n in (("a", "b"), ("a", "c"), ("z", "b"), (None, "b"), ("A", "B"))]
    equal(got, ["B", "A", "A", "A", "B"], label="rule needing prev=a and next=b, for (a,b) (a,c) (z,b) (None,b) (A,B)")
    entry = {"default": "A", "rules": [{"prev": {"a"}, "pron": "B"}, {"prev": {"a"}, "pron": "C"}]}
    equal(sub.choose_pronunciation("x", "a", None, entry), "B", label="first matching rule wins")


@check("g2p_with_homographs separates read, lead, and record by context")
def test_homograph_table(sub):
    cases = [
        (["I", "have", "read", "the", "pipe"], 2, "R EH1 D"),
        (["I", "will", "read", "the", "pipe"], 2, "R IY1 D"),
        (["I", "read", "yesterday"], 1, "R EH1 D"),
        (["the", "lead", "pipe"], 1, "L EH1 D"),
        (["I", "will", "lead", "the", "pipe"], 2, "L IY1 D"),
        (["I", "will", "record", "the", "pipe"], 2, "R IH0 K AO1 R D"),
        (["a", "record"], 1, "R EH1 K ER0 D"),
        (["Record"], 0, "R EH1 K ER0 D"),
    ]
    for words, i, want in cases:
        got = sub.g2p_with_homographs(words, LEXICON, HOMOGRAPHS)[i]
        if got != want:
            fail(f"in {' '.join(words)!r}, word {words[i]!r} got {got!r}, expected {want!r}")


@check("g2p_with_homographs looks up the lexicon, ignores case, and calls fallback for unknown words")
def test_g2p_sentence(sub):
    got = sub.g2p_with_homographs(["I", "Have", "the", "xq"], LEXICON, HOMOGRAPHS)
    equal(got, ["AY1", "HH AE1 V", "DH AH0", "X Q"], label="default fallback spells the letters")
    got = sub.g2p_with_homographs(["zed"], LEXICON, HOMOGRAPHS, fallback=lambda w: f"<{w}>")
    equal(got, ["<zed>"], label="custom fallback")


def _hand_inputs(dtype=torch.float32):
    h = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=dtype)
    durations = torch.tensor([[2, 1, 3]])
    pitch = torch.tensor([[210.0, 300.0, 450.0]], dtype=dtype)
    energy = torch.tensor([[1.0, 3.0, 7.0]], dtype=dtype)
    k = torch.arange(8, dtype=dtype)
    pitch_table = torch.stack([k, torch.zeros_like(k)], dim=1)
    energy_table = torch.stack([torch.zeros_like(k), k], dim=1)
    return h, durations, pitch, energy, pitch_table, energy_table


@check("variance_adaptor adds log-bin pitch and linear-bin energy embeddings per phone, then expands to frames")
def test_adaptor_hand(sub):
    # Plausible wrong implementations: adding the embeddings after length regulation with
    # phone-level indices (a shape mismatch, 3 against 6), or using linear pitch bins.
    h, d, p, e, pt, et = _hand_inputs()
    frames, mask, bins = sub.variance_adaptor(h, d, p, e, pt, et, pitch_range=(100.0, 800.0), energy_range=(0.0, 8.0))
    shape(frames, (1, 6, 2), label="frames")
    shape(mask, (1, 6), label="mask")
    equal(bins.tolist(), [[3, 4, 5]], label="pitch bins of 210, 300, 450 Hz with 8 log bins from 100 to 800 Hz")
    want = [[4, 1], [4, 1], [4, 4], [6, 7], [6, 7], [6, 7]]
    close(frames[0], want, label="frames: h + pitch row + energy row (energy bins 1, 3, 6)")
    equal(mask.tolist(), [[True] * 6], label="mask")


@check("A 1.2 pitch scale moves every log bin by 19 or 20, and rate shortens the output")
def test_adaptor_control(sub):
    rng = np.random.default_rng(0)
    n, k = 64, 256
    pitch = torch.tensor(rng.uniform(100.0, 500.0, size=(1, n)))
    h = torch.zeros(1, n, 4, dtype=torch.float64)
    table = torch.zeros(k, 4, dtype=torch.float64)
    d = torch.full((1, n), 4)
    energy = torch.ones(1, n, dtype=torch.float64)
    base = sub.variance_adaptor(h, d, pitch, energy, table, table)[2]
    up = sub.variance_adaptor(h, d, pitch, energy, table, table, pitch_scale=1.2)[2]
    shift = (up - base).numpy()
    if not np.all((shift == 19) | (shift == 20)):
        fail(f"bin shifts for a 1.2 pitch scale span {np.unique(shift).tolist()}, expected only 19 and 20 with 255 log-spaced bins over 71 to 795 Hz")
    frames_fast = sub.variance_adaptor(h, d, pitch, energy, table, table, rate=2.0)[0]
    shape(frames_fast, (1, 2 * n, 4), label="frames at rate 2 (each 4-frame phone becomes 2)")
    unvoiced = torch.zeros(1, n, dtype=torch.float64)
    equal(int(sub.variance_adaptor(h, d, unvoiced, energy, table, table, pitch_scale=1.2)[2].max()), 0, label="unvoiced bins after scaling")


@check("variance_adaptor sends gradient to h, d_i per frame, and to the used table rows")
def test_adaptor_grad(sub):
    h, d, p, e, pt, et = _hand_inputs(torch.float64)
    weights = torch.arange(1, 13, dtype=torch.float64).reshape(1, 6, 2)

    def loss(hh, ptab, etab):
        frames, _, _ = sub.variance_adaptor(hh, d, p, e, ptab, etab, pitch_range=(100.0, 800.0))
        return (frames * weights).sum()

    hh = h.clone().requires_grad_(True)
    ptab = pt.clone().requires_grad_(True)
    etab = et.clone().requires_grad_(True)
    loss(hh, ptab, etab).backward()
    numeric = finite_difference(lambda x: float(loss(torch.tensor(x), pt, et)), h.numpy())
    close(hh.grad, numeric, atol=1e-6, label="gradient of h against finite differences")
    # Phones 1, 2, 3 use pitch rows 3, 4, 5 and own 2, 1, 3 frames; unused rows get zero.
    want = np.zeros((8, 2))
    want[3] = weights[0, 0:2, :].sum(0).numpy()
    want[4] = weights[0, 2, :].numpy()
    want[5] = weights[0, 3:6, :].sum(0).numpy()
    close(ptab.grad, want, atol=1e-9, label="pitch table gradient, nonzero only in rows 3, 4, 5")
