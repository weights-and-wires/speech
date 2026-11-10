"""Forced alignment and word timestamps: when each token and word was said.

Emission log-probabilities have shape (T frames, V symbols), with the blank at
index 0. A path is one state per frame, where a state is an index into the
blank-extended sequence [blank, tokens[0], blank, ..., tokens[-1], blank].
"""

import numpy as np

from speechlab import play, show_image


def viterbi_trellis(log_probs: np.ndarray, tokens: list[int], blank: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The Viterbi score table and backpointers for one transcript.

    Args:
        log_probs: Per-frame log-probabilities, shape (T, V).
        tokens: The transcript as symbol ids, none of them the blank.
        blank: The blank's id.

    Returns:
        (delta, back), both of shape (T, S) with S = 2 * len(tokens) + 1.
        delta[t, s] is the best summed log-probability of any path that is in
        state s at frame t, or -inf when no legal path gets there. back[t, s]
        is the state at frame t - 1 on that best path (0 for t = 0).

    A path starts in state 0 or 1. From frame to frame a state may stay, or
    step forward one state, or skip forward two states. The skip is legal only
    when it lands on a token (not a blank) that differs from the token it
    jumps over, because two equal tokens in a row need a blank between them.
    """
    raise NotImplementedError("viterbi_trellis")


def forced_align(log_probs: np.ndarray, tokens: list[int], blank: int = 0) -> tuple[np.ndarray, float]:
    """The single best CTC path that spells `tokens`, by Viterbi.

    Args:
        log_probs: Per-frame log-probabilities, shape (T, V).
        tokens: The transcript as symbol ids, none of them the blank.
        blank: The blank's id.

    Returns:
        (states, score). `states` has shape (T,): the index into the
        blank-extended sequence [blank, tokens[0], blank, ..., tokens[-1], blank]
        that each frame occupies, so odd states belong to token (state - 1) // 2.
        `score` is the path's summed log-probability. Raises ValueError when
        T frames cannot fit the transcript.
    """
    raise NotImplementedError("forced_align")


def token_spans(states: np.ndarray, n_tokens: int) -> np.ndarray:
    """The frames each token occupies, shape (n_tokens, 2) of int.

    Row k is (start, end) for token k: `start` is the first frame whose state
    is 2k + 1, and `end` is one past the last such frame, so the token spans
    frames start..end-1. Raises ValueError when a token has no frame.
    """
    raise NotImplementedError("token_spans")


def word_timestamps(states: np.ndarray, word_lengths: list[int], hop_seconds: float) -> np.ndarray:
    """Start and end of every word in seconds, shape (n_words, 2).

    `word_lengths[w]` is the number of tokens in word w, in transcript order.
    A word starts at the start of its first token and ends at the end of its
    last token, where frame t covers [t * hop_seconds, (t + 1) * hop_seconds).
    The blanks around a word belong to no word.
    """
    raise NotImplementedError("word_timestamps")


# The Run experiment -----------------------------------------------------------

HOP = 0.02  # seconds per frame

if __name__ == "__main__":
    # A four-frame alignment you can check by hand: "a b" (ids 1 and 2).
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])
    states, score = forced_align(np.log(probs), [1, 2])
    print(f"Best path for 'a b' over 4 frames: states {states.tolist()}, probability {np.exp(score):.3f}")
    print("Token spans in frames:", token_spans(states, 2).tolist())

    # Forced alignment of "hello world" on emissions from a make-believe peaky acoustic model.
    rng = np.random.default_rng(0)
    letters = "_ehlowrd"
    text = "hello world"
    tokens = [letters.index(c) for c in text if c != " "]
    truth = [0] * 6
    for i, c in enumerate(c for c in text if c != " "):
        truth += [letters.index(c)] * int(rng.integers(2, 4)) + [0] * int(rng.integers(3, 6))
        if i == 4:
            truth += [0] * 8  # the pause between the words
    logits = 4.0 * np.eye(len(letters))[truth] + rng.normal(size=(len(truth), len(letters)))
    log_probs = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    states, score = forced_align(log_probs, tokens)
    times = word_timestamps(states, [5, 5], HOP)
    for word, (start, end) in zip(text.split(), times):
        print(f"{word}: {start:.2f} s to {end:.2f} s")
    spans = token_spans(states, len(tokens))
    print("First token spans in frames:", spans[:3].tolist(), f"at {HOP * 1000:.0f} ms per frame")
    show_image(log_probs.T, label="Emission log-probabilities for 'hello world'", xlabel="Frame", ylabel="Symbol",
               y_labels=list("_ehlowrd"), value_label="log p")
