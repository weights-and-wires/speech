"""Checks for the pronunciation assessment exercise.

Plausible wrong implementations these checks catch:

- GOP summed over a segment's frames instead of averaged. A long phone then looks
  worse than a short one with the same per-frame posteriors. "GOP is a mean per
  frame" catches it.
- Spearman computed on ordinal ranks (argsort of argsort) instead of tie-averaged
  ranks. "Tied values share an average rank" and "Spearman uses tie-averaged
  ranks" catch it.
"""

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape

POST = np.array([
    [0.80, 0.10, 0.05, 0.05],
    [0.70, 0.20, 0.05, 0.05],
    [0.10, 0.80, 0.05, 0.05],
    [0.10, 0.70, 0.10, 0.10],
    [0.05, 0.90, 0.025, 0.025],
    [0.05, 0.05, 0.20, 0.70],
    [0.10, 0.10, 0.30, 0.50],
])
ALIGN = np.array([[0, 0, 2], [1, 2, 5], [2, 5, 7]])  # k, ae, t


@check("gop_log_posterior averages log P(canonical) over each segment")
def test_lpp(sub):
    fn = has(sub, "gop_log_posterior")
    got = np.asarray(fn(POST, ALIGN), dtype=float)
    shape(got, (3,), label="one score per segment")
    want = np.array([
        np.mean(np.log([0.8, 0.7])),
        np.mean(np.log([0.8, 0.7, 0.9])),
        np.mean(np.log([0.2, 0.3])),
    ])
    close(got, want, rtol=1e-6, label="mean log posterior (-0.2899, -0.2284, -1.4067 expected)")


@check("gop subtracts the best phone: 0 when the canonical phone wins every frame")
def test_gop_zero(sub):
    fn = has(sub, "gop")
    got = np.asarray(fn(POST, ALIGN), dtype=float)
    shape(got, (3,), label="one score per segment")
    close(got[:2], np.zeros(2), atol=1e-9, label="GOP of k and ae")


@check("gop finds the planted mispronounced phone")
def test_gop_planted(sub):
    fn = has(sub, "gop")
    got = np.asarray(fn(POST, ALIGN), dtype=float)
    want = np.mean([np.log(0.2 / 0.7), np.log(0.3 / 0.5)])
    almost(got[2], want, 1e-6, label="GOP of the t that sounds like s (expected -0.8818)")
    if not np.all(got <= 1e-12):
        fail(f"GOP must be at most 0, got {np.round(got, 4)}")


@check("gop is a mean per frame: a stretched phone scores the same")
def test_gop_length(sub):
    fn = has(sub, "gop")
    frames = np.tile(np.array([[0.2, 0.7, 0.1], [0.3, 0.5, 0.2]]), (3, 1))  # 6 frames
    short = np.asarray(fn(frames[:2], np.array([[0, 0, 2]])), dtype=float)
    long = np.asarray(fn(frames, np.array([[0, 0, 6]])), dtype=float)
    almost(long[0], short[0], 1e-9, label="6-frame GOP, equal to the 2-frame GOP -0.8818")


@check("detect_mispronunciations flags GOP strictly below the threshold")
def test_detect(sub):
    fn = has(sub, "detect_mispronunciations")
    scores = np.array([-0.1, -1.0, -2.0, -0.99])
    got = np.asarray(fn(scores, -1.0))
    equal(got.astype(bool), np.array([False, False, True, False]), label="flags at threshold -1.0")


@check("precision_recall_f1 matches the hand-worked detection example")
def test_prf(sub):
    fn = has(sub, "precision_recall_f1")
    gop_scores = np.array([-0.1, -0.4, -2.0, -0.3, -0.8, -1.5, -0.2, -2.5])
    gold = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
    p, r, f = fn(gop_scores < -1.0, gold)
    almost(p, 2 / 3, 1e-9, label="precision at threshold -1.0")
    almost(r, 1 / 2, 1e-9, label="recall at threshold -1.0")
    almost(f, 4 / 7, 1e-9, label="F1 at threshold -1.0")
    p, r, f = fn(gop_scores < -0.35, gold)
    almost(p, 0.8, 1e-9, label="precision at threshold -0.35")
    almost(r, 1.0, 1e-9, label="recall at threshold -0.35")
    p, r, f = fn(np.zeros(3, bool), np.array([1, 0, 1], bool))
    equal((p, r, f), (0.0, 0.0, 0.0), label="nothing flagged returns zeros")


@check("Tied values share an average rank")
def test_ranks(sub):
    fn = has(sub, "average_ranks")
    close(fn(np.array([3, 5, 5, 8, 9.0])), np.array([1, 2.5, 2.5, 4, 5]), label="ranks of [3, 5, 5, 8, 9]")
    close(fn(np.array([2.0, 2.0, 2.0, 1.0])), np.array([3, 3, 3, 1]), label="three-way tie")


@check("pearson matches the closed form and ignores scale and shift")
def test_pearson(sub):
    fn = has(sub, "pearson")
    x = np.array([-1.2, -0.9, -0.6, -0.5, -0.1])
    h = np.array([3, 5, 5, 8, 9.0])
    almost(fn(x, h), 0.9324, 1e-4, label="Pearson of the worked example (expected 0.9324)")
    almost(fn(10 * x + 3, h), fn(x, h), 1e-9, label="Pearson after scaling x by 10 and adding 3")
    almost(fn(x, -h), -fn(x, h), 1e-9, label="Pearson flips sign with y")


@check("Spearman uses tie-averaged ranks")
def test_spearman(sub):
    fn = has(sub, "spearman")
    x = np.array([-1.2, -0.9, -0.6, -0.5, -0.1])
    h = np.array([3, 5, 5, 8, 9.0])
    almost(fn(x, h), 0.9747, 1e-4, label="Spearman with a tie (expected 0.9747, not 1.0)")
    rng = np.random.default_rng(0)
    u = rng.normal(size=30)
    almost(fn(u, np.exp(u)), 1.0, 1e-9, label="Spearman of a monotone transform")


@check("phone_confusion counts aligned pairs, canonical on rows")
def test_confusion(sub):
    fn = has(sub, "phone_confusion")
    can = np.array([2, 2, 0, 1, 2, 3])
    rec = np.array([3, 2, 0, 1, 3, 3])
    got = np.asarray(fn(can, rec, 4))
    want = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])
    shape(got, (4, 4), label="confusion matrix")
    equal(got, want, label="counts (row 2 is t: once t, twice s)")
    equal(got.sum(axis=1), np.bincount(can, minlength=4), label="row sums equal canonical counts")
