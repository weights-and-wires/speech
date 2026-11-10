"""Full-duplex models and voice agents: the worked solution."""

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
    stream = np.full(n_frames, pad, dtype=np.int64)
    taken = np.zeros(n_frames, dtype=bool)
    for start, tokens in zip(word_starts, word_tokens):
        stream[start : start + len(tokens)] = tokens
        taken[start : start + len(tokens)] = True
    for start in word_starts:
        if start > 0 and not taken[start - 1]:
            stream[start - 1] = epad
    return stream


def moshi_delays(n_q: int, tau: int) -> list:
    """The delay of each of Moshi's 2 * n_q + 1 streams, in frames.

    Order: text, then Moshi's semantic codebook and its n_q - 1 acoustic
    codebooks, then the user's semantic and acoustic codebooks. Text and
    semantic streams have delay 0; every acoustic codebook has delay tau.
    """
    audio = [0] + [tau] * (n_q - 1)
    return [0] + audio + audio


def delay_streams(codes: np.ndarray, delays: list, pad: int) -> np.ndarray:
    """Shift stream k of a (K, T) grid right by delays[k] frames.

    Returns (K, T + max(delays)) with out[k, t + delays[k]] = codes[k, t]
    and pad everywhere else.
    """
    n_streams, n_frames = codes.shape
    out = np.full((n_streams, n_frames + max(delays)), pad, dtype=codes.dtype)
    for k, d in enumerate(delays):
        out[k, d : d + n_frames] = codes[k]
    return out


def undelay_streams(delayed: np.ndarray, delays: list, n_frames: int) -> np.ndarray:
    """Invert delay_streams: returns the (K, n_frames) grid of aligned codes."""
    return np.stack([delayed[k, d : d + n_frames] for k, d in enumerate(delays)])


def interleave_fixed_ratio(text: np.ndarray, speech: np.ndarray, n_text: int = 13, n_speech: int = 26,
                           speech_offset: int = 1000) -> np.ndarray:
    """GLM-4-Voice-style interleaving: n_text text tokens, then n_speech speech
    tokens, repeated. When one stream runs out, the rest of the other follows.
    Speech ids are shifted by speech_offset so the two vocabularies don't
    collide. Returns a 1-D int array of length len(text) + len(speech).
    """
    out, i, j = [], 0, 0
    speech = np.asarray(speech) + speech_offset
    while i < len(text) or j < len(speech):
        out.extend(text[i : i + n_text])
        i += n_text
        out.extend(speech[j : j + n_speech])
        j += n_speech
    return np.asarray(out, dtype=np.int64)


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
    return float(sum(log_softmax(depth_fn(z, frame_codes[:k]))[frame_codes[k]] for k in range(len(frame_codes))))


def sequence_log_prob(codes: np.ndarray, temporal_fn, depth_fn) -> float:
    """log p(V_1, ..., V_T) for a (T, K) grid of codes.

    temporal_fn(previous_frames) takes the (s, K) codes of frames before s
    and returns z_s. Sum frame_log_prob over the frames.
    """
    return float(sum(frame_log_prob(temporal_fn(codes[:s]), codes[s], depth_fn) for s in range(len(codes))))


def sample_frame(z: np.ndarray, depth_fn, n_codebooks: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one frame's K codes, codebook by codebook, from the depth model."""
    prefix = []
    for _ in range(n_codebooks):
        p = np.exp(log_softmax(depth_fn(z, np.array(prefix, dtype=np.int64))))
        prefix.append(int(rng.choice(len(p), p=p)))
    return np.array(prefix, dtype=np.int64)


# Streaming --------------------------------------------------------------------------


def causal_conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int = 1) -> np.ndarray:
    """A causal dilated convolution: pad (K - 1) * dilation zeros on the left only.

    x: (C_in, T). weight: (C_out, C_in, K). bias: (C_out,). Returns (C_out, T),
    where out[:, t] = bias + sum over j of weight[:, :, j] @ x[:, t - (K - 1 - j) * dilation].
    """
    c_out, _, k = weight.shape
    pad = (k - 1) * dilation
    xp = np.pad(x, ((0, 0), (pad, 0)))
    n = x.shape[1]
    out = np.tile(bias[:, None], (1, n)).astype(float)
    for j in range(k):
        out += weight[:, :, j] @ xp[:, j * dilation : j * dilation + n]
    return out


class StreamingConv1d:
    """The same causal convolution, fed one chunk at a time.

    It keeps the last (K - 1) * dilation input samples of each channel as its
    state, starting from zeros, so the concatenated outputs of any chunking
    equal causal_conv1d on the whole signal.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray, dilation: int = 1):
        """Store the parameters and a (C_in, (K - 1) * dilation) buffer of zeros."""
        self.weight, self.bias, self.dilation = weight, bias, dilation
        self.state = np.zeros((weight.shape[1], (weight.shape[2] - 1) * dilation))

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """(C_in, n) in, (C_out, n) out, for any n including 0."""
        c_out, _, k = self.weight.shape
        n = chunk.shape[1]
        xp = np.concatenate([self.state, chunk], axis=1)
        out = np.tile(self.bias[:, None], (1, n)).astype(float)
        for j in range(k):
            out += self.weight[:, :, j] @ xp[:, j * self.dilation : j * self.dilation + n]
        if self.state.shape[1]:
            self.state = xp[:, xp.shape[1] - self.state.shape[1] :]
        return out


# Turn-taking ------------------------------------------------------------------------


def response_latency_ms(endpoint_ms: float, asr_ms: float, llm_ttft_ms: float, tts_first_chunk_ms: float,
                        network_ms: float, buffer_ms: float) -> float:
    """A cascaded agent's delay from the end of the user's speech to its first audio."""
    return endpoint_ms + asr_ms + llm_ttft_ms + tts_first_chunk_ms + network_ms + buffer_ms


def full_duplex_latency_ms(frame_rate_hz: float, acoustic_delay: int) -> float:
    """The algorithmic latency of a Moshi-style model: one frame plus the acoustic delay."""
    return (1 + acoustic_delay) * 1000.0 / frame_rate_hz


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
    frames = lambda ms: int(round(ms / frame_ms))
    min_speech, endpoint, response, reply = map(frames, (min_speech_ms, endpoint_ms, response_ms, reply_ms))
    state, since, speech_run, silence_run = "LISTENING", 0, 0, 0
    states, events = [], []
    for t, speech in enumerate(np.asarray(user_speech, dtype=bool)):
        speech_run, silence_run = (speech_run + 1, 0) if speech else (0, silence_run + 1)
        detected = speech_run >= min_speech
        if state == "LISTENING":
            if detected:
                state = "USER_SPEAKING"
                events.append((t, "user_start"))
        elif state in ("USER_SPEAKING", "ENDPOINT_PENDING"):
            if speech:
                state = "USER_SPEAKING"
            elif silence_run >= endpoint:
                state, since = "RESPONDING", t
                events.append((t, "endpoint"))
            else:
                state = "ENDPOINT_PENDING"
        elif state == "RESPONDING":
            if detected:
                state = "USER_SPEAKING"
                events.append((t, "user_start"))
            elif t - since >= response:
                state, since = "AGENT_SPEAKING", t
                events.append((t, "agent_start"))
        elif state == "AGENT_SPEAKING":
            if detected:
                state = "USER_SPEAKING"
                events.append((t, "barge_in"))
            elif t - since >= reply:
                state = "LISTENING"
                events.append((t, "agent_end"))
        states.append(state)
    return states, events


def classify_overlap(duration_s: float, has_words: bool, level_db: float, max_backchannel_s: float = 1.0,
                     loud_db: float = -35.0) -> str:
    """Decide whether the user's speech over the agent is a backchannel or a barge-in.

    duration_s is how long the overlapping speech lasts, has_words says
    whether a recognizer found words in it, and level_db is its level in dBFS.
    Speech that lasts max_backchannel_s or longer is a "barge_in". Shorter
    speech is a "barge_in" only when it has words and its level is at least
    loud_db; every other short sound ("mm-hm", a quiet "yeah") is a
    "backchannel". Returns "backchannel" or "barge_in".
    """
    if duration_s >= max_backchannel_s:
        return "barge_in"
    if has_words and level_db >= loud_db:
        return "barge_in"
    return "backchannel"


def _is_backchannel(segment: tuple, other: list, max_backchannel_s: float) -> bool:
    start, end = segment
    return end - start < max_backchannel_s and any(s <= start < e for s, e in other)


def turn_taking_stats(user_segments: list, agent_segments: list, max_backchannel_s: float = 1.0) -> dict:
    """Gaps, overlaps, and the turn-change rate of a two-channel timeline.

    Each segment is a (start_s, end_s) pair of speech on one channel, in
    seconds. A segment shorter than max_backchannel_s that starts while the
    other speaker is talking is a backchannel, and it is dropped. The
    remaining segments sort by start time, and consecutive segments of one
    speaker form one turn (start of the first, end of the last). For each pair
    of consecutive turns by different speakers, next.start - previous.end is a
    gap when positive and an overlap of the negated value when negative; zero
    counts as neither.

    Returns {"gaps": [...], "overlaps": [...], "turn_changes": int,
    "turn_rate_per_min": float}, with the lists in time order, in seconds.
    The rate is turn_changes per minute of the span from the first start to
    the last end, or 0.0 when the span is 0.
    """
    user = [tuple(s) for s in user_segments]
    agent = [tuple(s) for s in agent_segments]
    kept = [(s, e, 0) for s, e in user if not _is_backchannel((s, e), agent, max_backchannel_s)]
    kept += [(s, e, 1) for s, e in agent if not _is_backchannel((s, e), user, max_backchannel_s)]
    kept.sort()
    turns = []
    for s, e, who in kept:
        if turns and turns[-1][0] == who:
            turns[-1][2] = max(turns[-1][2], e)
        else:
            turns.append([who, s, e])
    gaps, overlaps = [], []
    for prev, nxt in zip(turns, turns[1:]):
        d = nxt[1] - prev[2]
        if d > 0:
            gaps.append(d)
        elif d < 0:
            overlaps.append(-d)
    span = max((e for _, e, _ in kept), default=0.0) - min((s for s, _, _ in kept), default=0.0)
    changes = max(len(turns) - 1, 0)
    return {"gaps": gaps, "overlaps": overlaps, "turn_changes": changes,
            "turn_rate_per_min": changes / span * 60.0 if span > 0 else 0.0}


def takeover_rate(samples: list, max_backchannel_s: float = 1.0, max_backchannel_words: int = 1) -> float:
    """Full-Duplex-Bench's takeover rate: the share of samples where the model takes the turn.

    Each sample is a list of (duration_s, n_words) pairs, one per stretch of
    speech the model produced. A stretch is a backchannel when it lasts less
    than max_backchannel_s and holds at most max_backchannel_words words. A
    sample's takeover is 1 when it holds any stretch that is not a
    backchannel, and 0 when it is silent or holds only backchannels. Returns
    the mean over samples, or 0.0 for no samples.
    """
    if not samples:
        return 0.0
    def taken(sample):
        return any(not (d < max_backchannel_s and w <= max_backchannel_words) for d, w in sample)
    return sum(taken(s) for s in samples) / len(samples)


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


def speech_segments(flags, frame_ms):
    """(start_s, end_s) pairs for each run of True in a per-frame bool array."""
    d = np.diff(np.concatenate([[0], np.asarray(flags, dtype=int), [0]]))
    return [(a * frame_ms / 1000, b * frame_ms / 1000) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


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
    stats = turn_taking_stats(speech_segments(speech, FRAME_MS), [(start / RATE, stop / RATE)])
    print(f"Turn changes: {stats['turn_changes']}; gaps {[round(float(g), 2) for g in stats['gaps']]} s; "
          f"overlaps {[round(float(o), 2) for o in stats['overlaps']]} s; {stats['turn_rate_per_min']:.1f} turn changes per minute.")
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
