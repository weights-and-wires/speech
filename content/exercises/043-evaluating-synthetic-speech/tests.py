"""Checks for the synthetic-speech evaluation exercise."""

import itertools

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, in_range


def _brute_force_dtw(cost):
    """The cheapest monotone path cost, by enumerating every sequence of moves."""
    n, m = cost.shape
    best = np.inf
    # A path is a sequence of moves: diagonal (1, 1), down (1, 0), right (0, 1).
    for n_diag in range(min(n, m)):
        downs, rights = n - 1 - n_diag, m - 1 - n_diag
        for moves in set(itertools.permutations("d" * n_diag + "v" * downs + "h" * rights)):
            i = j = 0
            total = cost[0, 0]
            for mv in moves:
                i += mv in "dv"
                j += mv in "dh"
                total += cost[i, j]
            best = min(best, total)
    return best


def _check_path(path, n, m):
    path = [tuple(int(v) for v in p) for p in path]
    if path[0] != (0, 0) or path[-1] != (n - 1, m - 1):
        fail(f"the path must run from (0, 0) to ({n - 1}, {m - 1}); it runs from {path[0]} to {path[-1]}")
    for (a, b), (c, d) in zip(path, path[1:]):
        if (c - a, d - b) not in ((1, 1), (1, 0), (0, 1)):
            fail(f"the step from {(a, b)} to {(c, d)} is not a diagonal, down, or right move")
    return path


@check("mel_cepstral_distortion gives the hand-worked 1.504 dB and ignores c0")
def test_mcd_hand(sub):
    fn = has(sub, "mel_cepstral_distortion")
    c = np.array([7.0, 1.0, 0.5, -0.2])
    c_hat = np.array([-3.0, 0.9, 0.3, -0.1])
    almost(fn(c, c_hat), 10 / np.log(10) * np.sqrt(0.12), 1e-9, label="MCD with c0 = 7 and -3, which must not count")
    almost(fn(c, c_hat), 1.504, 1e-3, label="the same, rounded")


@check("mel_cepstral_distortion works frame by frame, and is zero and symmetric where it should be")
def test_mcd_frames(sub):
    fn = has(sub, "mel_cepstral_distortion")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(6, 14)), rng.normal(size=(6, 14))
    got = np.asarray(fn(a, b))
    want = 10 / np.log(10) * np.sqrt(2 * np.sum((a[:, 1:] - b[:, 1:]) ** 2, axis=1))
    close(got, want, atol=1e-10, label="MCD for six frame pairs")
    close(fn(a, a), np.zeros(6), atol=1e-12, label="MCD of frames against themselves")
    close(fn(b, a), got, atol=1e-12, label="MCD with the arguments swapped")


@check("dtw_path returns a valid path with the brute-force minimum cost")
def test_dtw_brute(sub):
    fn = has(sub, "dtw_path")
    rng = np.random.default_rng(1)
    for n, m in [(3, 3), (4, 5), (5, 4), (2, 6), (1, 4), (5, 5)]:
        cost = rng.random((n, m))
        path = _check_path(fn(cost), n, m)
        got = sum(cost[i, j] for i, j in path)
        almost(got, _brute_force_dtw(cost), 1e-9, label=f"the path cost on a {n} by {m} cost matrix")


@check("mcd_dtw is zero for a time-stretched copy and symmetric in its arguments")
def test_mcd_dtw(sub):
    fn = has(sub, "mcd_dtw")
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(12, 10))
    stretched = np.repeat(ref, [1, 2, 1, 3, 1, 1, 2, 1, 1, 2, 1, 1], axis=0)
    almost(fn(ref, stretched), 0.0, 1e-12, label="MCD-DTW against the same frames with some repeated")
    other = ref + 0.3 * rng.normal(size=ref.shape)
    almost(fn(ref, other), fn(other, ref), 1e-9, label="MCD-DTW with the arguments swapped")
    per_frame = 10 / np.log(10) * np.sqrt(2 * np.sum((ref[:, 1:] - other[:, 1:]) ** 2, axis=1))
    if fn(ref, other) > per_frame.mean() + 1e-9:
        fail(f"DTW may only lower the cost below the frame-by-frame pairing ({per_frame.mean():.3f} dB); "
             f"got {fn(ref, other):.3f} dB")


@check("f0_rmse gives 7.91 Hz and 50% V/UV error on the hand-worked contours")
def test_f0(sub):
    fn = has(sub, "f0_rmse")
    rmse, vuv = fn(np.array([200.0, 210, 0, 220]), np.array([190.0, 215, 180, 0]))
    almost(rmse, np.sqrt(62.5), 1e-9, label="F0 RMSE over the two frames voiced in both")
    almost(vuv, 0.5, 1e-12, label="the fraction of frames whose voicing disagrees")
    rmse, vuv = fn(np.array([100.0, 0, 0, 150]), np.array([110.0, 0, 0, 140]))
    almost(rmse, 10.0, 1e-9, label="F0 RMSE when both agree on voicing")
    almost(vuv, 0.0, 1e-12, label="V/UV error when both agree")


@check("speaker_similarity is the cosine of each pair of rows, whatever their lengths")
def test_similarity(sub):
    fn = has(sub, "speaker_similarity")
    rng = np.random.default_rng(3)
    e1, e2 = rng.normal(size=(5, 192)), rng.normal(size=(5, 192))
    want = np.sum(e1 * e2, axis=1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))
    close(fn(e1, e2), want, atol=1e-12, label="cosine similarity of five pairs")
    close(fn(3 * e1, 0.1 * e2), want, atol=1e-12, label="the same after rescaling the embeddings")
    close(fn(e1, e1), np.ones(5), atol=1e-12, label="each embedding against itself")
    close(fn(e1[0], -e1[0]), [-1.0], atol=1e-12, label="an embedding against its negative")


@check("bootstrap_mos returns the mean inside a seeded interval that widens with rater effects")
def test_bootstrap(sub):
    fn = has(sub, "bootstrap_mos")
    rng = np.random.default_rng(4)
    raters, items = 20, 30
    offsets = rng.normal(0, 0.5, (raters, 1)) + rng.normal(0, 0.5, (1, items))
    ratings = 3.5 + offsets + rng.normal(0, 0.5, (raters, items))
    mean, low, high = fn(ratings, n_boot=1000, level=0.95, seed=0)
    almost(mean, ratings.mean(), 1e-12, label="the returned mean")
    if not low < mean < high:
        fail(f"the interval [{low:.3f}, {high:.3f}] should contain the mean {mean:.3f}")
    again = fn(ratings, n_boot=1000, level=0.95, seed=0)
    close(np.array(again), np.array([mean, low, high]), atol=0, label="a second call with the same seed")
    se = np.sqrt(ratings.mean(axis=1).var(ddof=1) / raters + ratings.mean(axis=0).var(ddof=1) / items)
    half = (high - low) / 2
    if not 0.88 * 1.96 * se < half < 1.15 * 1.96 * se:
        fail(f"with rater and item effects the 95% half-width should be near 1.96 * {se:.3f} = "
             f"{1.96 * se:.3f}; got {half:.3f}. Resample raters and items, not single ratings.")
    naive = 1.96 * ratings.std(ddof=1) / np.sqrt(ratings.size)
    if half < 2 * naive:
        fail(f"the half-width {half:.3f} should be far wider than the {naive:.3f} that treats every "
             "rating as independent")


@check("word_error_rate normalizes case and punctuation, then counts word edits over reference words")
def test_wer(sub):
    fn = has(sub, "word_error_rate")
    almost(fn(["Hello, World!"], ["hello world"]), 0.0, 1e-12, label="WER after normalization")
    almost(fn(["the cat sat on the mat"], ["the cat sat on a mat"]), 1 / 6, 1e-12, label="one substitution in six words")
    almost(fn(["a b c", "d e f g"], ["a c", "d e x f g"]), 2 / 7, 1e-12,
           label="a deletion and an insertion over 7 reference words")
    in_range(fn(["one two"], ["three four five six"]), 1.0, 2.0, label="a WER can pass 100%")
