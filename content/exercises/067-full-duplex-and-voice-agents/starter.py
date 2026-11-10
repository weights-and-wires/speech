"""Full-duplex models and voice agents.

Fill in the stream builders (inner monologue, delays, interleaving), the
RQ-Transformer's factored likelihood and sampler, the causal and streaming
convolutions, the latency budget, and the turn-taking state machine. Press
Check to test them, and Run to hear a simulated conversation with a
barge-in.
"""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_image, show_plot

PAD, EPAD = 0, 1  # The two special tokens of the text stream in this exercise.


# Streams --------------------------------------------------------------------------


def inner_monologue(word_starts: list, word_tokens: list, n_frames: int, pad: int = PAD, epad: int = EPAD) -> np.ndarray:
    """Moshi's time-aligned text stream, one token per audio frame.

    word_starts[i] is the frame where word i starts in the audio, and
    word_tokens[i] its text tokens. Word i's tokens fill frames
    word_starts[i], word_starts[i] + 1, and so on. Every other frame holds
    pad, except that the frame just before each word's start holds epad
    ("end of padding") when that frame exists and is not already a token.
    Returns an int array of shape (n_frames,).
    """
    raise NotImplementedError


def moshi_delays(n_q: int, tau: int) -> list:
    """The delay of each of Moshi's 2 * n_q + 1 streams, in frames.

    Order: text, then Moshi's semantic codebook and its n_q - 1 acoustic
    codebooks, then the user's semantic and acoustic codebooks. Text and
    semantic streams have delay 0; every acoustic codebook has delay tau.
    """
    raise NotImplementedError


def delay_streams(codes: np.ndarray, delays: list, pad: int) -> np.ndarray:
    """Shift stream k of a (K, T) grid right by delays[k] frames.

    Returns (K, T + max(delays)) with out[k, t + delays[k]] = codes[k, t]
    and pad everywhere else.
    """
    raise NotImplementedError


def undelay_streams(delayed: np.ndarray, delays: list, n_frames: int) -> np.ndarray:
    """Invert delay_streams: returns the (K, n_frames) grid of aligned codes."""
    raise NotImplementedError


def interleave_fixed_ratio(text: np.ndarray, speech: np.ndarray, n_text: int = 13, n_speech: int = 26,
                           speech_offset: int = 1000) -> np.ndarray:
    """GLM-4-Voice-style interleaving: n_text text tokens, then n_speech speech
    tokens, repeated. When one stream runs out, the rest of the other follows.
    Speech ids are shifted by speech_offset so the two vocabularies don't
    collide. Returns a 1-D int array of length len(text) + len(speech).
    """
    raise NotImplementedError


# The RQ-Transformer's depth step ----------------------------------------------------


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """A numerically stable log softmax of a 1-D array of logits (given, not graded)."""
    shifted = logits - np.max(logits)
    return shifted - np.log(np.sum(np.exp(shifted)))


def frame_log_prob(z: np.ndarray, frame_codes: np.ndarray, depth_fn) -> float:
    """log p(V_s | z_s) = sum over k of log p(V_{s,k} | z_s, V_{s,<k}).

    depth_fn(z, prefix) returns the logits (V,) for the next codebook, given
    the frame's context z and the codes already chosen in this frame.
    """
    raise NotImplementedError


def sequence_log_prob(codes: np.ndarray, temporal_fn, depth_fn) -> float:
    """log p(V_1, ..., V_T) for a (T, K) grid of codes.

    temporal_fn(previous_frames) takes the (s, K) codes of frames before s
    and returns z_s. Sum frame_log_prob over the frames.
    """
    raise NotImplementedError


def sample_frame(z: np.ndarray, depth_fn, n_codebooks: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one frame's K codes, codebook by codebook, from the depth model."""
    raise NotImplementedError


# Streaming --------------------------------------------------------------------------


def causal_conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int = 1) -> np.ndarray:
    """A causal dilated convolution: pad (K - 1) * dilation zeros on the left only.

    x: (C_in, T). weight: (C_out, C_in, K). bias: (C_out,). Returns (C_out, T),
    where out[:, t] = bias + sum over j of weight[:, :, j] @ x[:, t - (K - 1 - j) * dilation].
    """
    raise NotImplementedError


class StreamingConv1d:
    """The same causal convolution, fed one chunk at a time.

    It keeps the last (K - 1) * dilation input samples of each channel as its
    state, starting from zeros, so the concatenated outputs of any chunking
    equal causal_conv1d on the whole signal.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray, dilation: int = 1):
        """Store the parameters and a (C_in, (K - 1) * dilation) buffer of zeros."""
        raise NotImplementedError

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """(C_in, n) in, (C_out, n) out, for any n including 0."""
        raise NotImplementedError


# Turn-taking ------------------------------------------------------------------------


def response_latency_ms(endpoint_ms: float, asr_ms: float, llm_ttft_ms: float, tts_first_chunk_ms: float,
                        network_ms: float, buffer_ms: float) -> float:
    """A cascaded agent's delay from the end of the user's speech to its first audio."""
    raise NotImplementedError


def full_duplex_latency_ms(frame_rate_hz: float, acoustic_delay: int) -> float:
    """The algorithmic latency of a Moshi-style model: one frame plus the acoustic delay."""
    raise NotImplementedError


def turn_taking(user_speech: np.ndarray, frame_ms: float, min_speech_ms: float, endpoint_ms: float,
                response_ms: float, reply_ms: float) -> tuple[list, list]:
    """A turn-taking state machine for a cascaded voice agent.

    user_speech: one bool per frame from a VAD. Durations convert to frames
    with round(ms / frame_ms). Keep two run lengths: consecutive speech
    frames and consecutive silent frames, each reset by the other. "Detected
    speech" means the speech run has reached min_speech frames, so shorter
    sounds (backchannels such as "mm-hm") are ignored. At each frame t:

    - LISTENING: detected speech -> USER_SPEAKING, event "user_start".
    - USER_SPEAKING or ENDPOINT_PENDING: a speech frame -> USER_SPEAKING.
      A silent frame -> RESPONDING with event "endpoint" if the silent run
      has reached endpoint frames, and ENDPOINT_PENDING otherwise.
    - RESPONDING: detected speech -> USER_SPEAKING, event "user_start".
      Otherwise, response frames after the "endpoint" frame ->
      AGENT_SPEAKING, event "agent_start".
    - AGENT_SPEAKING: detected speech -> USER_SPEAKING, event "barge_in".
      Otherwise, reply frames after the "agent_start" frame -> LISTENING,
      event "agent_end".

    Returns (the state after each frame, a list of (frame, event) pairs).
    """
    raise NotImplementedError


# Run helpers -----------------------------------------------------------------------

RATE = 16000
FRAME_MS = 10


def voice(seconds, f0, formants, rng):
    """A buzzy synthetic voice: a jittered pulse train through three formant resonators."""
    n = int(seconds * RATE)
    pitch = f0 * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 4) * np.arange(n) / RATE))
    y = (np.diff(np.floor(np.cumsum(pitch / RATE)), prepend=0) > 0).astype(float)
    for f, bw in zip(formants, (80, 100, 140)):
        r = np.exp(-np.pi * bw / RATE)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
    syllables = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * np.arange(n) / RATE)
    return 0.5 * y / np.abs(y).max() * syllables * np.hanning(n) ** 0.1


def conversation_demo(rng):
    seconds = 6.0
    user = np.zeros(int(seconds * RATE))
    for start, length in [(0.3, 1.4), (3.0, 0.12), (4.3, 1.2)]:  # a question, an "mm-hm", a barge-in
        a = int(start * RATE)
        user[a : a + int(length * RATE)] += voice(length, 120, (700, 1200, 2500), rng)
    user += 0.003 * rng.standard_normal(user.size)
    frame = RATE * FRAME_MS // 1000
    energy_db = 10 * np.log10(np.mean(user[: user.size // frame * frame].reshape(-1, frame) ** 2, axis=1) + 1e-12)
    speech = energy_db > -40.0  # a fixed-threshold energy VAD, the simplest one from the VAD lesson
    budget = dict(endpoint_ms=500, asr_ms=50, llm_ttft_ms=250, tts_first_chunk_ms=120, network_ms=40, buffer_ms=40)
    states, events = turn_taking(speech, FRAME_MS, min_speech_ms=200, endpoint_ms=budget["endpoint_ms"],
                                 response_ms=response_latency_ms(**budget) - budget["endpoint_ms"], reply_ms=3000)
    agent = np.zeros_like(user)
    reply = voice(3.0, 200, (500, 1700, 2600), rng)
    start = next(t for t, e in events if e == "agent_start") * frame
    stop = next((t for t, e in events if e == "barge_in"), len(states)) * frame
    agent[start:stop] = reply[: stop - start]
    print("Turn-taking events (frame = 10 ms):")
    for t, e in events:
        print(f"  {t * FRAME_MS / 1000:5.2f} s  {e}")
    silence_end = next(t for t, e in events if e == "endpoint") - budget["endpoint_ms"] // FRAME_MS
    print(f"The user stopped at about {silence_end * FRAME_MS / 1000:.2f} s; the agent spoke "
          f"{start / RATE - silence_end * FRAME_MS / 1000:.2f} s later (budget {response_latency_ms(**budget):.0f} ms).")
    print("The 0.12 s 'mm-hm' at 3.0 s was shorter than 200 ms, so it did not interrupt the agent.")
    play(np.stack([user, agent]), RATE, label="Left: the user. Right: the agent, cut off by a barge-in")
    names = ["LISTENING", "USER_SPEAKING", "ENDPOINT_PENDING", "RESPONDING", "AGENT_SPEAKING"]
    t_axis = np.arange(len(states)) * FRAME_MS / 1000
    show_plot((t_axis, np.array([names.index(s) for s in states], dtype=float)), (t_axis, speech.astype(float) * 4.2),
              labels=["state (0 listening ... 4 agent speaking)", "VAD"], label="The state machine over time",
              xlabel="Time (s)", ylabel="State")


def streams_demo(rng):
    n_q, tau, n_frames = 8, 1, 20
    text = inner_monologue([2, 7, 13], [[101, 102], [103], [104, 105, 106]], n_frames)
    print("\nInner monologue for 3 words over 20 frames (0 = PAD, 1 = EPAD):")
    print("  " + " ".join(map(str, text)))
    moshi = rng.integers(2, 2048, size=(n_q, n_frames))
    user = rng.integers(2, 2048, size=(n_q, n_frames))
    grid = np.concatenate([text[None], moshi, user])
    delayed = delay_streams(grid, moshi_delays(n_q, tau), pad=0)
    labels = ["text"] + [f"moshi {k}" for k in range(n_q)] + [f"user {k}" for k in range(n_q)]
    show_image(delayed > 1, label=f"Moshi's 17 streams, acoustic delay {tau} (dark = PAD or EPAD)",
               xlabel="Frame (80 ms each)", y_labels=labels, colormap="gray")
    print(f"Moshi's algorithmic latency: {full_duplex_latency_ms(12.5, tau):.0f} ms (one 80 ms frame plus {tau} frame of delay)")

    weight, bias = rng.normal(size=(4, 1, 5)) * 0.3, np.zeros(4)
    x = rng.normal(size=(1, RATE))
    offline = causal_conv1d(x, weight, bias, dilation=3)
    stream = StreamingConv1d(weight, bias, dilation=3)
    online = np.concatenate([stream.step(x[:, i : i + 320]) for i in range(0, RATE, 320)], axis=1)
    print(f"Streaming a causal conv in 20 ms chunks: largest difference from offline = {np.abs(online - offline).max():.1e}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    conversation_demo(rng)
    streams_demo(rng)
