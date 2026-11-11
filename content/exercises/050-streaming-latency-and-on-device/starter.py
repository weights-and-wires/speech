"""Streaming latency and on-device budgets.

Audio durations are in seconds, latencies are in milliseconds, memory is in
bytes, and compute is in multiply-accumulates (MACs).

Fill in the functions below, then press Run to see a latency table and a
memory budget, and Check to test your code.
"""

import math

import numpy as np


def rtf(audio_seconds, compute_seconds):
    """Real-time factor: seconds of compute per second of audio."""
    raise NotImplementedError


def algorithmic_latency(chunk_frames, lookahead_frames, hop_ms):
    """Delay in ms for each position in a chunk, shape (chunk_frames,).

    Position p waits for the rest of its chunk, then for `lookahead_frames`
    more frames, and its own frame lasts one hop.
    """
    raise NotImplementedError


def streaming_latency(chunk_frames, lookahead_frames, hop_ms, rtf_value):
    """Algorithmic delay plus the compute for one chunk, shape (chunk_frames,).

    Computing a chunk of `chunk_frames * hop_ms` ms of audio takes
    `rtf_value` times that long. When `rtf_value >= 1` the backlog grows
    without bound, so every entry is infinite.
    """
    raise NotImplementedError


def partial_latencies(events, word_ends, endpoint_ms):
    """Latencies in ms from a timeline of (time_ms, kind, text) events.

    kind is "partial" or "final". `word_ends` holds the end time of each
    reference word. Returns a dict with the keys first_partial_ms,
    finalization_ms, and endpoint_to_final_ms.
    """
    raise NotImplementedError


def unstable_prefix_rate(partials, final):
    """Fraction of shown words that sit after the first word that differs from `final`.

    `partials` is a list of strings and `final` is a string. Returns 0.0
    when no words were shown.
    """
    raise NotImplementedError


def model_bytes(params, bits):
    """Bytes to store `params` weights at `bits` bits each, rounded up."""
    raise NotImplementedError


def chunk_activation_bytes(chunk_frames, left_frames, d_model, n_heads, bytes_per_value):
    """Peak bytes for one layer on one chunk.

    Counts the layer input and output for the chunk, the keys and values
    for the chunk plus its left context, and the attention scores.
    """
    raise NotImplementedError


def fits_on_device(params, bits, chunk_activation_bytes, budget_bytes):
    """True when the weights plus one chunk's activations fit the budget."""
    raise NotImplementedError


def device_rtf(macs_per_audio_second, device_macs_per_second):
    """RTF of a model that needs `macs_per_audio_second` on a device."""
    raise NotImplementedError


if __name__ == "__main__":
    for chunk in (1, 2, 4, 8, 16):
        mean = float(np.mean(streaming_latency(chunk, 2, 40, 0.3)))
        print(f"chunk of {chunk:2d} frames: mean latency {mean:6.1f} ms at RTF 0.3")
    events = [(560, "partial", "turn"), (800, "partial", "turn on"), (1000, "partial", "turn on a"),
              (1480, "partial", "turn on the light"), (1900, "final", "turn on the lights")]
    print(partial_latencies(events, [480, 640, 760, 1240], 1640))
    print("unstable-prefix rate:", unstable_prefix_rate([e[2] for e in events[:-1]], events[-1][2]))
    act = chunk_activation_bytes(16, 64, 256, 4, 2)
    for bits in (16, 8, 4):
        print(f"{bits:2d} bits: {model_bytes(30_000_000, bits):,} bytes, fits 40 MB: "
              f"{fits_on_device(30_000_000, bits, act, 40_000_000)}")
    print("device RTF:", device_rtf(750e6, 2e9))
