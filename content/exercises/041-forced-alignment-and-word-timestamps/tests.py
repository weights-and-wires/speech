"""Checks for the forced alignment and word timestamps exercise."""

import itertools
import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape

# Emission probabilities of (blank, a, b) over four frames: the worked example in the lesson.
PROBS = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])


def table_check(got, want, label):
    """Compare a score table that holds probabilities, where 0 means unreachable (-inf in logs)."""
    got = np.asarray(got, dtype=float)
    shape(got, want.shape, label=label)
    reachable = want > 0
    if not np.array_equal(np.isfinite(got), reachable):
        fail(f"{label}: cells that are reachable should be finite and the rest -inf; "
             f"expected finite cells\n{reachable.astype(int)}\nbut got\n{np.isfinite(got).astype(int)}")
    close(np.exp(got[reachable]), want[reachable], rtol=1e-9, label=label + " (probabilities of the reachable cells)")


@check("viterbi_trellis fills the worked table for the transcript a b, with the a-to-b skip")
def test_trellis(sub):
    got = has(sub, "viterbi_trellis")
    with np.errstate(divide="ignore"):
        delta, back = got(np.log(PROBS), [1, 2])
    want = np.array([[0.6, 0.3, 0, 0, 0],
                     [0.06, 0.42, 0.03, 0.06, 0],
                     [0.006, 0.084, 0.042, 0.294, 0.006],
                     [0.003, 0.0084, 0.042, 0.1176, 0.147]])
    table_check(delta, want, "Viterbi scores for 'a b'")
    back = np.asarray(back)
    shape(back, (4, 5), label="backpointers")
    equal(int(back[2, 3]), 1, label="the state before (frame 2, state 3), a skip over the blank from state 1")
    equal(int(back[3, 4]), 3, label="the state before (frame 3, state 4)")


@check("viterbi_trellis refuses the skip between two equal tokens in the transcript a a")
def test_trellis_repeat(sub):
    # Wrong implementation caught here: a trellis that only tests "not blank" before a skip, so a a may
    # jump from state 1 to state 3 and collapse to a single a. It scores state 3 at frame 1 as 0.21, not 0.
    got = has(sub, "viterbi_trellis")
    with np.errstate(divide="ignore"):
        delta, _ = got(np.log(PROBS), [1, 1])
    want = np.array([[0.6, 0.3, 0, 0, 0],
                     [0.06, 0.42, 0.03, 0, 0],
                     [0.006, 0.084, 0.042, 0.006, 0],
                     [0.003, 0.0084, 0.042, 0.0042, 0.003]])
    table_check(delta, want, "Viterbi scores for 'a a'")


def brute_align(log_probs, tokens):
    t_frames, v = log_probs.shape
    best, best_path = -np.inf, None
    for path in itertools.product(range(v), repeat=t_frames):
        collapsed = [p for i, p in enumerate(path) if p != 0 and (i == 0 or path[i - 1] != p)]
        if collapsed == list(tokens):
            score = sum(log_probs[t, p] for t, p in enumerate(path))
            if score > best:
                best, best_path = score, path
    return best, best_path


@check("forced_align solves the three-frame example and refuses a transcript that cannot fit")
def test_forced_align(sub):
    got = has(sub, "forced_align")
    lp = np.log(np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]]))
    states, score = got(lp, [1])
    equal(np.asarray(states).tolist(), [0, 1, 2], label="states for 'a' over three frames")
    almost(float(score), math.log(0.336), 1e-9, label="best path log-probability, log(0.6 * 0.7 * 0.8)")
    try:
        got(np.log(np.full((2, 3), 1 / 3)), [1, 1])
    except ValueError:
        pass
    else:
        fail("two frames cannot hold 'a a', which needs a blank between the letters; expected ValueError")


@check("forced_align matches brute force over every CTC path, repeated letters included")
def test_forced_align_brute(sub):
    got = has(sub, "forced_align")
    rng = np.random.default_rng(6)
    for tokens, t_frames in (([1, 2], 5), ([1, 1], 5), ([2, 1, 2], 6), ([1, 1, 1], 5)):
        logits = rng.normal(size=(t_frames, 3)) * 2
        lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        states, score = got(lp, tokens)
        want, path = brute_align(lp, tokens)
        almost(float(score), float(want), 1e-9, label=f"best score for tokens {tokens} over {t_frames} frames")
        ext = [0]
        for tok in tokens:
            ext += [tok, 0]
        states = np.asarray(states)
        steps = np.diff(states)
        legal = states[0] <= 1 and states[-1] >= len(ext) - 2 and np.all((steps >= 0) & (steps <= 2))
        legal = legal and all(ext[s] != 0 and ext[s] != ext[s - 2] for s, d in zip(states[1:], steps) if d == 2)
        if not legal:
            fail(f"states {states.tolist()} are not a legal CTC path through {ext}")
        labels = [ext[s] for s in states]
        along = sum(lp[t, s] for t, s in enumerate(labels))
        almost(float(along), float(want), 1e-9, label=f"log-probability along your returned path for {tokens}")


@check("token_spans reads each token's frames off a path, with and without a skip")
def test_token_spans(sub):
    got = has(sub, "token_spans")
    equal(np.asarray(got(np.array([0, 1, 1, 2, 3, 4, 4]), 2)).tolist(), [[1, 3], [4, 5]],
          label="spans of a path with a blank between the tokens")
    equal(np.asarray(got(np.array([0, 1, 3, 3, 4]), 2)).tolist(), [[1, 2], [2, 4]],
          label="spans of a path that skips the blank between the tokens")


@check("word_timestamps turns a frame path into start and end seconds, without the blanks around a word")
def test_word_timestamps(sub):
    # Wrong implementations caught here: an end time of the last frame's start (0.10 s, not 0.12 s for the
    # first word), and an end time that runs on to the next token's start (0.14 s).
    got = has(sub, "word_timestamps")
    states = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 6])
    close(got(states, [2, 1], 0.02), np.array([[0.04, 0.12], [0.14, 0.18]]), label="words at 20 ms per frame")
    close(got(states, [2, 1], 0.04), np.array([[0.08, 0.24], [0.28, 0.36]]), label="the same path at 40 ms per frame")
    close(got(states, [3], 0.02), np.array([[0.04, 0.18]]), label="one word of three tokens")


@check("word_timestamps recovers the planted word times from aligned emissions")
def test_word_timestamps_planted(sub):
    align, stamps = has(sub, "forced_align"), has(sub, "word_timestamps")
    rng = np.random.default_rng(7)
    # Two words of 3 and 2 tokens over a vocabulary of blank plus 4 symbols, planted frame by frame.
    plan = [(0, 5), (1, 3), (0, 3), (2, 2), (0, 4), (1, 3), (0, 9), (3, 3), (0, 3), (4, 2), (0, 5)]
    truth = np.concatenate([[symbol] * count for symbol, count in plan])
    logits = 6.0 * np.eye(5)[truth] + 0.5 * rng.normal(size=(len(truth), 5))
    lp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    states, _ = align(lp, [1, 2, 1, 3, 4])
    got = np.asarray(stamps(states, [3, 2], 0.02))
    want = np.array([[5 * 0.02, 20 * 0.02], [29 * 0.02, 37 * 0.02]])
    # Words start at the first frame of their first token and end after the last frame of their last token.
    if not np.allclose(got, want, atol=1e-9):
        fail(f"expected word times {want.tolist()} s from the planted path, got {got.tolist()} s")
