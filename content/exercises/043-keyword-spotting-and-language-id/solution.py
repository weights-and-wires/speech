"""Keyword spotting, endpointing, and language identification.

Reference solution. It smooths keyword posteriors, fires detections with a
hold-off, scores a stream in false accepts per hour and false-reject rate,
lists the points of a DET curve, decides when an utterance has ended, and
scores multi-language trials with the NIST average cost.
"""

import numpy as np
from scipy.special import logsumexp

from speechlab import show_plot

HOP_SECONDS = 0.01


def smooth_posteriors(posteriors, window):
    """Causal moving average: output t is the mean of posteriors[max(0, t - window + 1) : t + 1]."""
    p = np.asarray(posteriors, dtype=float)
    total = np.concatenate([[0.0], np.cumsum(p)])
    t = np.arange(len(p))
    start = np.maximum(0, t - window + 1)
    return (total[t + 1] - total[start]) / (t + 1 - start)


def detect(scores, threshold, holdoff):
    """Frame indices where a detection fires, as an int array.

    A frame fires when its score is at or above `threshold` and it is more than
    `holdoff` frames after the previous detection. Frames t + 1 to t + holdoff
    after a detection at t are suppressed.
    """
    scores = np.asarray(scores, dtype=float)
    fired = []
    blocked_until = -1
    for t in np.flatnonzero(scores >= threshold):
        if t > blocked_until:
            fired.append(int(t))
            blocked_until = t + holdoff
    return np.array(fired, dtype=int)


def evaluate_stream(detections, spans, n_frames, hop_seconds):
    """Return (false accepts per hour, false-reject rate) for a scored stream.

    `spans` is a list of (start, end) keyword frames, start included and end
    excluded. A detection inside a span accepts that keyword. A detection
    outside every span is a false accept. Repeat detections inside a span that
    is already accepted are ignored. The hour count is the whole stream.
    """
    hit = np.zeros(len(spans), dtype=bool)
    false_accepts = 0
    for d in np.asarray(detections, dtype=int):
        inside = [i for i, (start, end) in enumerate(spans) if start <= d < end]
        if inside:
            hit[inside[0]] = True
        else:
            false_accepts += 1
    hours = n_frames * hop_seconds / 3600.0
    return false_accepts / hours, float(1.0 - hit.mean())


def det_points(target_scores, nontarget_scores):
    """Return (thresholds, p_miss, p_fa), one entry per threshold, from low to high.

    The thresholds are every distinct score in ascending order, then +inf. A
    trial is accepted when its score is at or above the threshold, so
    p_miss = mean(target < threshold) and p_fa = mean(nontarget >= threshold).
    """
    t = np.asarray(target_scores, dtype=float)
    n = np.asarray(nontarget_scores, dtype=float)
    thresholds = np.append(np.unique(np.concatenate([t, n])), np.inf)
    p_miss = np.array([(t < x).mean() for x in thresholds])
    p_fa = np.array([(n >= x).mean() for x in thresholds])
    return thresholds, p_miss, p_fa


def endpoint(vad, eoq, short_timeout, long_timeout, threshold=0.5):
    """The frame where the utterance is declared finished, or -1 if it never is.

    After the first speech frame, count consecutive non-speech frames and reset
    the count at each speech frame. At a non-speech frame whose end-of-query
    probability `eoq[t]` is at or above `threshold`, the limit is
    `short_timeout`; otherwise it is `long_timeout`. Return the first frame
    where the count exceeds its limit.
    """
    started = False
    quiet = 0
    for t in range(len(vad)):
        if vad[t]:
            started = True
            quiet = 0
        elif started:
            quiet += 1
            limit = short_timeout if eoq[t] >= threshold else long_timeout
            if quiet > limit:
                return t
    return -1


def endpoint_metrics(decisions, last_speech, hop_ms):
    """Return (cut-off rate, mean latency in ms over the utterances not cut off).

    `decisions[i]` is the frame where utterance i was declared finished and
    `last_speech[i]` its last frame of speech. A decision before the last speech
    frame is a cut-off. Latency is (decision - last speech frame) * hop_ms.
    The mean latency is nan when every utterance was cut off.
    """
    d = np.asarray(decisions)
    last = np.asarray(last_speech)
    cut = d < last
    latency = (d - last)[~cut] * hop_ms
    return float(cut.mean()), float(latency.mean()) if latency.size else float("nan")


def language_llr(loglik):
    """Log-likelihood ratio of each language against the rest, shape (n_trials, n_languages).

    For trial scores l_i, LLR_i = l_i - log(mean over j != i of exp(l_j)).
    """
    loglik = np.asarray(loglik, dtype=float)
    n = loglik.shape[1]
    out = np.empty_like(loglik)
    for i in range(n):
        rest = np.delete(loglik, i, axis=1)
        out[:, i] = loglik[:, i] - (logsumexp(rest, axis=1) - np.log(n - 1))
    return out


def cavg(loglik, labels, beta=1.0):
    """The NIST LRE average detection cost.

    Cavg = (1/N) * [sum_T P_miss(T) + (beta / (N - 1)) * sum_T sum_{L != T} P_fa(T, L)],
    where N is the number of languages. Language T is accepted for a trial when its
    LLR is at or above log(beta).
    """
    loglik = np.asarray(loglik, dtype=float)
    labels = np.asarray(labels)
    n = loglik.shape[1]
    accept = language_llr(loglik) >= np.log(beta)
    miss = sum(1.0 - accept[labels == i, i].mean() for i in range(n))
    fa = sum(accept[labels == j, i].mean() for i in range(n) for j in range(n) if j != i)
    return float((miss + beta / (n - 1) * fa) / n)


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
