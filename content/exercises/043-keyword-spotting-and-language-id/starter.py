"""Keyword spotting, endpointing, and language identification.

Build the pieces of an always-on detector and the measures that grade it:
smooth keyword posteriors, fire detections with a hold-off, count false accepts
per hour and missed keywords on a stream, list the points of a DET curve, decide
when an utterance has ended, and score multi-language trials with the NIST
average cost. Fill in the functions below, then press Run to see the detector
on a planted six-minute stream, and Check to test your code.
"""

import numpy as np
from scipy.special import logsumexp

from speechlab import show_plot

HOP_SECONDS = 0.01


def smooth_posteriors(posteriors, window):
    """Causal moving average: output t is the mean of posteriors[max(0, t - window + 1) : t + 1]."""
    raise NotImplementedError


def detect(scores, threshold, holdoff):
    """Frame indices where a detection fires, as an int array.

    A frame fires when its score is at or above `threshold` and it is more than
    `holdoff` frames after the previous detection. Frames t + 1 to t + holdoff
    after a detection at t are suppressed.
    """
    raise NotImplementedError


def evaluate_stream(detections, spans, n_frames, hop_seconds):
    """Return (false accepts per hour, false-reject rate) for a scored stream.

    `spans` is a list of (start, end) keyword frames, start included and end
    excluded. A detection inside a span accepts that keyword. A detection
    outside every span is a false accept. Repeat detections inside a span that
    is already accepted are ignored. The hour count is the whole stream.
    """
    raise NotImplementedError


def det_points(target_scores, nontarget_scores):
    """Return (thresholds, p_miss, p_fa), one entry per threshold, from low to high.

    The thresholds are every distinct score in ascending order, then +inf. A
    trial is accepted when its score is at or above the threshold, so
    p_miss = mean(target < threshold) and p_fa = mean(nontarget >= threshold).
    """
    raise NotImplementedError


def endpoint(vad, eoq, short_timeout, long_timeout, threshold=0.5):
    """The frame where the utterance is declared finished, or -1 if it never is.

    After the first speech frame, count consecutive non-speech frames and reset
    the count at each speech frame. At a non-speech frame whose end-of-query
    probability `eoq[t]` is at or above `threshold`, the limit is
    `short_timeout`; otherwise it is `long_timeout`. Return the first frame
    where the count exceeds its limit.
    """
    raise NotImplementedError


def endpoint_metrics(decisions, last_speech, hop_ms):
    """Return (cut-off rate, mean latency in ms over the utterances not cut off).

    `decisions[i]` is the frame where utterance i was declared finished and
    `last_speech[i]` its last frame of speech. A decision before the last speech
    frame is a cut-off. Latency is (decision - last speech frame) * hop_ms.
    The mean latency is nan when every utterance was cut off.
    """
    raise NotImplementedError


def language_llr(loglik):
    """Log-likelihood ratio of each language against the rest, shape (n_trials, n_languages).

    For trial scores l_i, LLR_i = l_i - log(mean over j != i of exp(l_j)).
    """
    raise NotImplementedError


def cavg(loglik, labels, beta=1.0):
    """The NIST LRE average detection cost.

    Cavg = (1/N) * [sum_T P_miss(T) + (beta / (N - 1)) * sum_T sum_{L != T} P_fa(T, L)],
    where N is the number of languages. Language T is accepted for a trial when its
    LLR is at or above log(beta).
    """
    raise NotImplementedError


def make_stream(seed=0):
    """A 6-minute stream of frame scores with planted events. Returns (scores, spans).

    There are 36 slots of 1000 frames. Twenty hold a keyword the model scores at
    about 0.95 for 60 frames, four hold a weak keyword near 0.35, six hold a false
    bump near 0.85 for 40 frames, three hold a one-frame spike, and three are empty.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 0.1, 36000)
    kinds = ["hit"] * 20 + ["weak"] * 4 + ["bump"] * 6 + ["spike"] * 3 + ["empty"] * 3
    spans = []
    for slot, k in enumerate(rng.permutation(len(kinds))):
        kind, start = kinds[k], slot * 1000 + 300
        if kind in ("hit", "weak"):
            scores[start : start + 60] += 0.9 if kind == "hit" else 0.3
            spans.append((start, start + 60))
        elif kind == "bump":
            scores[start : start + 40] += 0.8
        elif kind == "spike":
            scores[start] += 0.9
    return scores, spans


if __name__ == "__main__":
    scores, spans = make_stream()
    print(f"Stream: {len(scores) * HOP_SECONDS / 60:.0f} minutes, {len(spans)} keywords")
    for window in (1, 5):
        for holdoff in (0, 100):
            found = detect(smooth_posteriors(scores, window), 0.5, holdoff)
            fa, frr = evaluate_stream(found, spans, len(scores), HOP_SECONDS)
            print(f"window {window}, hold-off {holdoff:3d}: {len(found):4d} detections, "
                  f"{fa:7.1f} false accepts per hour, FRR {frr:.3f}")

    rng = np.random.default_rng(1)
    target, other = rng.normal(2.0, 1.0, 300), rng.normal(0.0, 1.0, 3000)
    thresholds, p_miss, p_fa = det_points(target, other)
    print(f"DET: {len(thresholds)} points")
    show_plot((p_fa, p_miss), label="DET points of a simulated detector", xlabel="False accept rate", ylabel="Miss rate")

    vad = np.array([1] * 6 + [0] * 3 + [1] * 3 + [0] * 8)
    eoq = np.array([0.0] * 6 + [0.1, 0.1, 0.2] + [0.0] * 3 + [0.3, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    print(f"Fixed 5-frame timeout ends at frame {endpoint(vad, eoq, 5, 5)}; "
          f"with end-of-query probability at frame {endpoint(vad, eoq, 1, 5)}")

    logp = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.4, 0.4, 0.2]]))
    print(f"Cavg on four toy trials: {cavg(logp, np.array([0, 1, 2, 0])):.3f}")
