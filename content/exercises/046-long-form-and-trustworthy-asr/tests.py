"""Checks for the long-form and trustworthy ASR exercise."""

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape


def text(words):
    return " ".join(words)


@check("stitch_chunks joins two windows at the words they share, with no duplicates")
def test_stitch_two(sub):
    stitch = has(sub, "stitch_chunks")
    a = "we hold these truths to be self evident".split()
    b = "to be self evident that all men".split()
    got = stitch([a, b], 6)
    want = "we hold these truths to be self evident that all men".split()
    if got != want:
        fail(f"expected '{text(want)}', got '{text(got)}'")


@check("stitch_chunks recovers a planted transcript through garbled window edges and repeated words")
def test_stitch_planted(sub):
    # Wrong implementation caught here: plain concatenation, which repeats the six overlapping
    # words at every seam.
    stitch = has(sub, "stitch_chunks")
    truth = ("the cat sat on the mat and the dog sat on the log while the bird sang "
             "to the cat and the dog and the cat sat on the mat again").split()
    size, step = 16, 10
    chunks = []
    for start in range(0, len(truth) - size + step, step):
        chunk = list(truth[start:start + size])
        if start > 0:
            chunk[0] = "xq" + chunk[0]
        if start + size < len(truth):
            chunk[-1] = chunk[-1][:2] + "#"
        chunks.append(chunk)
    got = stitch(chunks, 6)
    if got != truth:
        fail(f"expected {len(truth)} words '{text(truth)}', got {len(got)} words '{text(got)}'")


@check("stitch_chunks appends the next window when the seam shares fewer than min_run words")
def test_stitch_no_match(sub):
    stitch = has(sub, "stitch_chunks")
    got = stitch([["one", "two", "the"], ["the", "three", "four"]], 3)
    want = ["one", "two", "the", "the", "three", "four"]
    if got != want:
        fail(f"a single shared word is not a seam; expected {want}, got {got}")
    got = stitch([["a", "b"], ["c", "d"]], 2)
    if got != ["a", "b", "c", "d"]:
        fail(f"expected ['a', 'b', 'c', 'd'], got {got}")


@check("word_timestamps reads word times from a hand-built cross-attention matrix")
def test_timestamps_hand(sub):
    f = has(sub, "word_timestamps")
    attn = np.full((3, 8), 0.05)
    attn[0, 0:2] = attn[1, 2:6] = attn[2, 6:8] = 0.9
    got = np.asarray(f(attn, [(0, 1), (1, 3)], 0.02), dtype=float)
    shape(got, (2, 2), label="word times")
    close(got, np.array([[0.0, 0.04], [0.04, 0.16]]), atol=1e-9, label="word start and end in seconds")


@check("word_timestamps recovers planted token boundaries from noisy attention")
def test_timestamps_planted(sub):
    # Wrong implementation caught here: running DTW on the raw -attn. Every weight is positive, so
    # the path prefers a detour through extra cells, and the boundaries slide. Normalizing each
    # frame across the tokens makes the weak cells cost something.
    f = has(sub, "word_timestamps")
    rng = np.random.default_rng(7)
    lengths = [5, 3, 8, 4, 6, 2, 7]
    attn = np.zeros((len(lengths), sum(lengths)))
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    for i in range(len(lengths)):
        attn[i, bounds[i]:bounds[i + 1]] = 0.6 + 0.3 * rng.random(lengths[i])
    attn += 0.08 * rng.random(attn.shape)
    words = [(0, 2), (2, 3), (3, 6), (6, 7)]
    got = np.asarray(f(attn, words, 0.02), dtype=float)
    first = [bounds[a] for a, _ in words]
    want = np.stack([first, first[1:] + [bounds[-1]]], axis=1) * 0.02
    close(got, want, atol=1e-9, label="word start and end in seconds")


@check("ece matches the lesson's worked example, 0.15, and weights bins by their size")
def test_ece_worked(sub):
    # Wrong implementation caught here: averaging the per-bin gaps without weights gives 1/6 = 0.1667.
    ece = has(sub, "ece")
    conf = np.array([0.9, 0.8, 0.7, 0.2])
    correct = np.array([1, 0, 1, 0])
    almost(float(ece(conf, correct, 2)), 0.15, 1e-9, label="ECE with 2 bins")


@check("ece is 0 when every bin is calibrated and equals the gap for a constant confidence")
def test_ece_closed_form(sub):
    ece = has(sub, "ece")
    conf = np.concatenate([np.full(10, 0.25), np.full(10, 0.75)])
    # Accuracy 0.3 at confidence 0.25 and 0.7 at confidence 0.75: each bin is off by 0.05.
    correct = np.concatenate([[1] * 3 + [0] * 7, [1] * 7 + [0] * 3]).astype(float)
    almost(float(ece(conf, correct, 10)), 0.05, 1e-9, label="ECE with every bin off by 0.05")
    # Accuracy 0.1 at confidence 0.25 and 0.7 at confidence 0.75: the gaps are 0.15 and 0.05.
    uneven = np.concatenate([[1] + [0] * 9, [1] * 7 + [0] * 3]).astype(float)
    almost(float(ece(conf, uneven, 10)), (0.15 + 0.05) / 2, 1e-9, label="ECE with gaps 0.15 and 0.05 in equal bins")
    constant = np.full(100, 1.0)
    right = np.r_[np.ones(60), np.zeros(40)]
    almost(float(ece(constant, right, 10)), 0.4, 1e-9, label="confidence 1.0 and accuracy 0.6, so ECE is 0.4")
    rng = np.random.default_rng(3)
    c = rng.random(500)
    value = float(ece(c, (rng.random(500) < c).astype(float), 10))
    if not 0.0 <= value < 0.1:
        fail(f"draws that are correct with probability equal to their confidence should give ECE near 0, got {value:.4f}")


@check("repetition_score flags a planted loop and spares clean text")
def test_repetition(sub):
    f = has(sub, "repetition_score")
    loop = "thanks for watching thanks for watching thanks for watching thanks for watching".split()
    almost(float(f(loop)), 1.0, 1e-9, label="a pure 3-word loop repeated 4 times")
    mixed = "the meeting starts at noon".split() + "see you see you see you".split()
    almost(float(f(mixed)), 6 / 11, 1e-9, label="6 of 11 words sit in a see-you loop")
    clean = "she said that that was what she had had in mind".split()
    almost(float(f(clean)), 0.0, 1e-9, label="two adjacent repeats ('that that', 'had had') are below min_repeats=3")
    almost(float(f([])), 0.0, 1e-9, label="empty text")


@check("subgroup_wer_with_ci gives pooled WER per group and matches a reference bootstrap")
def test_subgroup(sub):
    # Wrong implementation caught here too: the mean of per-utterance WERs. Group "b" has a short
    # utterance with 2 errors in 2 words, so macro-averaging gives (1.0 + 0 + 0.1 + 0.1) / 4 = 0.30
    # where pooled WER is 6 / 62 = 0.097.
    f = has(sub, "subgroup_wer_with_ci")
    errors = [1, 2, 1, 1, 2, 0, 3, 1]
    lens = [10, 20, 10, 10, 2, 20, 30, 10]
    groups = ["a", "a", "a", "a", "b", "b", "b", "b"]
    got = f(errors, lens, groups, 500, 0, 0.1)
    equal(sorted(got), ["a", "b"], label="group labels")
    almost(got["a"][0], 5 / 50, 1e-9, label="pooled WER of group a")
    almost(got["b"][0], 6 / 62, 1e-9, label="pooled WER of group b (not the mean of per-utterance WERs)")
    rng = np.random.default_rng(0)
    e, n = np.array(errors, float), np.array(lens, float)
    g = np.array(groups)
    for key in ["a", "b"]:
        ek, nk = e[g == key], n[g == key]
        picks = rng.integers(0, len(ek), size=(500, len(ek)))
        low, high = np.percentile(ek[picks].sum(1) / nk[picks].sum(1), [5, 95])
        close(np.array(got[key][1:], dtype=float), np.array([low, high]), atol=1e-12, label=f"90% interval of group {key}")
    flat = f([2, 2, 2], [10, 10, 10], ["x"] * 3, 200, 1)
    close(np.array(flat["x"]), np.array([0.2, 0.2, 0.2]), atol=1e-12, label="identical utterances give a zero-width interval")


@check("A group with fewer utterances gets a wider interval, and a seed fixes the answer")
def test_subgroup_width(sub):
    f = has(sub, "subgroup_wer_with_ci")
    rng = np.random.default_rng(5)
    lens = np.full(210, 10)
    errors = rng.binomial(10, 0.2, size=210)
    groups = np.array(["big"] * 200 + ["small"] * 10)
    got = f(errors, lens, groups, 400, 3)
    width = {k: v[2] - v[1] for k, v in got.items()}
    if not width["small"] > 2 * width["big"]:
        fail(f"10 utterances should give an interval over twice as wide as 200: small {width['small']:.4f}, big {width['big']:.4f}")
    for k, (w, lo, hi) in got.items():
        if not lo <= w <= hi:
            fail(f"group {k}: the WER {w:.4f} should lie inside its interval [{lo:.4f}, {hi:.4f}]")
    again = f(errors, lens, groups, 400, 3)
    equal(again, got, label="the same seed gives the same result")
