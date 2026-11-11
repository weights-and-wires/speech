"""Streaming latency and on-device budgets.

Audio durations are in seconds, latencies are in milliseconds, memory is in
bytes, and compute is in multiply-accumulates (MACs).
"""

import math

import numpy as np


def rtf(audio_seconds, compute_seconds):
    """Real-time factor: seconds of compute per second of audio."""
    return compute_seconds / audio_seconds


def algorithmic_latency(chunk_frames, lookahead_frames, hop_ms):
    """Delay in ms for each position in a chunk, shape (chunk_frames,).

    Position p waits for the rest of its chunk, then for `lookahead_frames`
    more frames, and its own frame lasts one hop.
    """
    p = np.arange(chunk_frames)
    return (chunk_frames - p + lookahead_frames) * float(hop_ms)


def streaming_latency(chunk_frames, lookahead_frames, hop_ms, rtf_value):
    """Algorithmic delay plus the compute for one chunk, shape (chunk_frames,).

    Computing a chunk of `chunk_frames * hop_ms` ms of audio takes
    `rtf_value` times that long. When `rtf_value >= 1` the backlog grows
    without bound, so every entry is infinite.
    """
    algo = algorithmic_latency(chunk_frames, lookahead_frames, hop_ms)
    if rtf_value >= 1.0:
        return np.full_like(algo, np.inf)
    return algo + rtf_value * chunk_frames * hop_ms


def partial_latencies(events, word_ends, endpoint_ms):
    """Latencies in ms from a timeline of (time_ms, kind, text) events.

    kind is "partial" or "final". `word_ends` holds the end time of each
    reference word. Returns a dict with first_partial_ms,
    finalization_ms, and endpoint_to_final_ms.
    """
    first = next(t for t, kind, text in events if kind == "partial" and text.split())
    final = next(t for t, kind, text in events if kind == "final")
    return {
        "first_partial_ms": first - word_ends[0],
        "finalization_ms": final - word_ends[-1],
        "endpoint_to_final_ms": final - endpoint_ms,
    }


def unstable_prefix_rate(partials, final):
    """Fraction of shown words that sit after the first word that differs from `final`."""
    target = final.split()
    shown = 0
    unstable = 0
    for text in partials:
        words = text.split()
        stable = 0
        while stable < len(words) and stable < len(target) and words[stable] == target[stable]:
            stable += 1
        shown += len(words)
        unstable += len(words) - stable
    return unstable / shown if shown else 0.0


def model_bytes(params, bits):
    """Bytes to store `params` weights at `bits` bits each, rounded up."""
    return math.ceil(params * bits / 8)


def chunk_activation_bytes(chunk_frames, left_frames, d_model, n_heads, bytes_per_value):
    """Peak bytes for one layer on one chunk.

    Counts the layer input and output for the chunk, the keys and values
    for the chunk plus its left context, and the attention scores.
    """
    context = left_frames + chunk_frames
    values = 2 * chunk_frames * d_model + 2 * context * d_model + n_heads * chunk_frames * context
    return values * bytes_per_value


def fits_on_device(params, bits, chunk_activation_bytes, budget_bytes):
    """True when the weights plus one chunk's activations fit the budget."""
    return model_bytes(params, bits) + chunk_activation_bytes <= budget_bytes


def device_rtf(macs_per_audio_second, device_macs_per_second):
    """RTF of a model that needs `macs_per_audio_second` on a device."""
    return macs_per_audio_second / device_macs_per_second


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
