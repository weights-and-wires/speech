"""Checks for the paralinguistics and emotion exercise."""

import numpy as np

from speechcheck import almost, check, close, equal, fail, finite, has, shape


def alternating(base, delta, n):
    """`n` values base * (1 + delta), base * (1 - delta), ...: a planted perturbation of known size."""
    return base * (1 + delta * (-1.0) ** np.arange(n))


def blobs(counts, rng, gap=1.2):
    """Four-feature Gaussian classes with the given counts, so the rare classes are the hard ones."""
    means = gap * np.array([[0.0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]])
    y = np.repeat(np.arange(len(counts)), counts)
    return means[y] + rng.standard_normal((len(y), 4)), y


@check("jitter divides the mean absolute period difference by the mean of all the periods")
def test_jitter(sub):
    jitter = has(sub, "jitter")
    # Hand example: differences 2, 4, 3 average 3; the four periods average 100.25.
    almost(jitter(np.array([100.0, 102.0, 98.0, 101.0])), 3 / 100.25, 1e-12, label="[100, 102, 98, 101]")
    # Planted voice: every period is 1% off in alternating directions, so each
    # difference is 2% of the base period and, for an even count, the mean is the base.
    almost(jitter(alternating(133.33, 0.01, 8)), 0.02, 1e-12, label="8 periods, planted 1%")
    # Five periods: the mean is 1.01 of the base, not the base. Dividing by the
    # mean of the first four periods (a plausible wrong implementation) gives 0.1.
    almost(jitter(alternating(133.33, 0.05, 5)), 0.1 / 1.01, 1e-12, label="5 periods, planted 5%")
    almost(jitter(np.full(6, 80.0)), 0.0, 1e-12, label="a perfectly regular voice")
    almost(jitter(alternating(133.33, 0.02, 8) * 3.0), jitter(alternating(133.33, 0.02, 8)), 1e-12,
           label="jitter of the same voice an octave-and-a-half lower")


@check("shimmer is jitter's twin for peak amplitudes and ignores loudness")
def test_shimmer(sub):
    shimmer = has(sub, "shimmer")
    # Differences 0.05, 0.10, 0.07 average 0.07333; the amplitudes average 0.505.
    almost(shimmer(np.array([0.50, 0.55, 0.45, 0.52])), (0.22 / 3) / 0.505, 1e-12, label="hand example")
    amps = alternating(0.6, 0.1, 8)
    almost(shimmer(amps), 0.2, 1e-12, label="planted 10%")
    almost(shimmer(amps * 0.01), 0.2, 1e-12, label="the same voice 40 dB quieter")
    almost(shimmer(np.full(5, 0.3)), 0.0, 1e-12, label="constant amplitudes")


@check("f0_stats summarizes voiced frames only, with the range in semitones")
def test_f0_stats(sub):
    f0_stats = has(sub, "f0_stats")
    # Voiced values sort to 100 100 150 150 200 200: p20 = 100, p80 = 200, an octave.
    mean, spread = f0_stats(np.array([0, 100, 100, 200, 200, 0, 150, 150, 0, 0], dtype=float))
    almost(mean, 150.0, 1e-9, label="mean F0 (zeros are unvoiced, not 0 Hz)")
    almost(spread, 12.0, 1e-9, label="range in semitones (an octave is 12)")
    mean, spread = f0_stats(np.full(10, 220.0))
    almost(mean, 220.0, 1e-9, label="mean of a monotone voice")
    almost(spread, 0.0, 1e-9, label="range of a monotone voice")


@check("ccc matches hand values, including a pure shift and a pure rescaling")
def test_ccc_hand(sub):
    ccc = has(sub, "ccc")
    gold = np.array([-1.0, 0.0, 1.0])
    almost(ccc(gold, gold), 1.0, 1e-12, label="identical sequences")
    almost(ccc(-gold, gold), -1.0, 1e-12, label="reversed sequence")
    # Means differ by 1: 2 (2/3) / (2/3 + 2/3 + 1) = 4/7, though Pearson is 1.
    almost(ccc(gold + 1.0, gold), 4 / 7, 1e-12, label="shifted by 1")
    # Variances 8/3 and 2/3: 2 (4/3) / (8/3 + 2/3) = 0.8, though Pearson is 1.
    almost(ccc(2.0 * gold, gold), 0.8, 1e-12, label="scaled by 2")
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(50), rng.standard_normal(50) + 0.3
    rho = np.corrcoef(x, y)[0, 1]
    want = 2 * rho * x.std() * y.std() / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    almost(ccc(x, y), want, 1e-12, label="Lin's formula with rho, sigma, mu on random data")


@check("ccc is below Pearson under a bias, and equals Pearson only when mean and variance match")
def test_ccc_vs_pearson(sub):
    ccc = has(sub, "ccc")
    rng = np.random.default_rng(1)
    gold = rng.uniform(-1, 1, 100)
    for name, pred in [("shifted by 0.5", gold + 0.5), ("compressed to 0.3", 0.3 * gold)]:
        # A plausible wrong implementation computes np.corrcoef here and returns 1.0.
        got = ccc(pred, gold)
        if not got < 0.8:
            fail(f"ccc of a prediction {name} is {got:.4f}, expected below 0.8 (Pearson is 1.0, which ignores bias)")
    almost(ccc(gold, gold + 0.5), ccc(gold + 0.5, gold), 1e-12, label="symmetry")
    shuffled = rng.permutation(gold)  # same mean and variance, so CCC is exactly Pearson
    almost(ccc(shuffled, gold), np.corrcoef(shuffled, gold)[0, 1], 1e-12, label="equal mean and variance")


@check("cohen_kappa subtracts chance agreement: 0.4, 0.635, and 0 at 82% raw agreement")
def test_kappa(sub):
    cohen_kappa = has(sub, "cohen_kappa")
    # Observed 35/50 = 0.7, chance (25 * 30 + 25 * 20) / 50^2 = 0.5.
    almost(cohen_kappa(np.array([[20, 5], [10, 15]])), 0.4, 1e-12, label="two classes")
    # Observed 23/30, chance (12 * 13 + 12 * 11 + 6 * 6) / 30^2 = 0.36.
    almost(cohen_kappa(np.array([[10, 2, 0], [3, 8, 1], [0, 1, 5]])), (23 / 30 - 0.36) / 0.64, 1e-12, label="three classes")
    # Both annotators say "neutral" 90% of the time: agreement 0.82 is exactly what chance gives.
    almost(cohen_kappa(np.array([[81, 9], [9, 1]])), 0.0, 1e-12, label="82% agreement, all of it chance")
    almost(cohen_kappa(np.diag([7, 5, 3])), 1.0, 1e-12, label="perfect agreement")


@check("confusion_matrix counts rows as true classes and keeps classes that never occur")
def test_confusion(sub):
    confusion_matrix = has(sub, "confusion_matrix")
    cm = np.asarray(confusion_matrix(np.array([0, 0, 1, 2, 2, 2]), np.array([0, 1, 1, 2, 0, 2]), 3))
    shape(cm, (3, 3), label="confusion matrix")
    equal(cm, np.array([[1, 1, 0], [0, 1, 0], [1, 0, 2]]), label="rows true, columns predicted")
    cm = np.asarray(confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), 4))
    shape(cm, (4, 4), label="four classes, two never seen")
    equal(int(cm.sum()), 3, label="every clip counted once")


@check("uar averages per-class recall, so a majority-class guesser scores 1/3 where accuracy says 0.8")
def test_uar(sub):
    accuracy, uar = has(sub, "accuracy"), has(sub, "uar")
    cm = np.array([[70, 8, 2], [5, 8, 2], [1, 1, 3]])
    almost(accuracy(cm), 0.81, 1e-12, label="accuracy of the hand example")
    # Recalls 70/80, 8/15, 3/5. Normalizing by column sums instead gives a different number.
    almost(uar(cm), (70 / 80 + 8 / 15 + 3 / 5) / 3, 1e-12, label="UAR of the hand example")
    always_neutral = np.array([[80, 0, 0], [15, 0, 0], [5, 0, 0]])
    almost(accuracy(always_neutral), 0.8, 1e-12, label="accuracy of always saying neutral")
    almost(uar(always_neutral), 1 / 3, 1e-12, label="UAR of always saying neutral")
    almost(uar(np.array([[8, 2], [1, 9]])), (0.8 + 0.9) / 2, 1e-12, label="two balanced classes")


@check("fit_linear solves weighted ridge regression, with an unpenalized bias and balanced class weights")
def test_fit_linear(sub):
    fit_linear = has(sub, "fit_linear")
    rng = np.random.default_rng(2)
    X, y = blobs([40, 15, 5], rng)
    n = len(y)
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(3)[y]
    for balanced in (False, True):
        w = n / (3 * np.bincount(y))[y] if balanced else np.ones(n)
        # With no penalty this is least squares on rows scaled by sqrt(w).
        want = np.linalg.lstsq(np.sqrt(w)[:, None] * Xb, np.sqrt(w)[:, None] * Y, rcond=None)[0]
        got = np.asarray(fit_linear(X, y, 3, l2=0.0, balanced=balanced))
        shape(got, (5, 3), label="W")
        close(got, want, rtol=1e-7, atol=1e-8, label=f"l2 = 0, balanced = {balanced}")
    # A huge penalty drives the weights to 0 but leaves the bias free: it becomes the
    # weighted mean of the one-hot targets, the class shares, or 1/3 each when balanced.
    big = np.asarray(fit_linear(X, y, 3, l2=1e9, balanced=False))
    close(big[:-1], np.zeros((4, 3)), atol=1e-4, label="weights under a huge penalty")
    close(big[-1], np.bincount(y) / n, atol=1e-4, label="bias, unbalanced: the class shares")
    big = np.asarray(fit_linear(X, y, 3, l2=1e9, balanced=True))
    close(big[-1], np.full(3, 1 / 3), atol=1e-4, label="bias, balanced: 1/3 each")


@check("balanced weights raise UAR on an imbalanced set, because the rare classes stop being ignored")
def test_balanced_uar(sub):
    fit_linear, predict_linear = has(sub, "fit_linear"), has(sub, "predict_linear")
    confusion_matrix, uar = has(sub, "confusion_matrix"), has(sub, "uar")
    rng = np.random.default_rng(3)
    X, y = blobs([300, 60, 20], rng)
    Xt, yt = blobs([300, 60, 20], rng)
    scores = {}
    for balanced in (False, True):
        pred = np.asarray(predict_linear(fit_linear(X, y, 3, l2=1.0, balanced=balanced), Xt))
        shape(pred, (len(yt),), label="predictions")
        finite(pred, label="predictions")
        scores[balanced] = uar(confusion_matrix(yt, pred, 3))
    if not scores[True] > scores[False] + 0.05:
        fail(f"UAR with balanced weights is {scores[True]:.3f} and without is {scores[False]:.3f}; expected balanced to win by at least 0.05")
    if not scores[True] > 0.6:
        fail(f"UAR with balanced weights is {scores[True]:.3f}, expected above 0.6 on this set")
