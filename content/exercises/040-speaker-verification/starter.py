"""Speaker embeddings and verification: the angular margin loss, cosine scores, EER, and minDCF.

Scores are cosine similarities, larger for "same speaker". A trial is a pair of
clips; a target trial is the same speaker, a non-target trial is two speakers.
"""

import math

import numpy as np
import torch
from scipy.signal import lfilter

from speechlab import play, show_plot


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cosine similarity of each row of `a` with the same row of `b`.

    Args:
        a: One embedding per row, shape (N, D).
        b: One embedding per row, shape (N, D).

    Returns:
        Shape (N,): row i holds a[i] . b[i] / (|a[i]| |b[i]|).
    """
    raise NotImplementedError("cosine_scores")


def aam_softmax_loss(
    embeddings: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor, scale: float = 30.0, margin: float = 0.2
) -> torch.Tensor:
    """Additive angular margin (ArcFace) softmax loss, averaged over the batch.

    Args:
        embeddings: One embedding per row, shape (N, D).
        weights: One class center per row, shape (C, D).
        labels: The true class of each row, shape (N,) of int64.
        scale: The logit scale s.
        margin: The angular margin m, in radians.

    The logit of class j is s * cos(theta_j), where cos(theta_j) is the cosine
    between the embedding and center j. The true class's logit is instead
    s * cos(theta_y + m), computed as cos(theta) cos(m) - sin(theta) sin(m),
    except where cos(theta_y) <= cos(pi - m): there theta_y + m would pass pi,
    so use s * (cos(theta_y) - m sin(m)) to keep the logit falling as the
    angle grows.
    """
    raise NotImplementedError("aam_softmax_loss")


def equal_error_rate(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> float:
    """The equal error rate of a verification system.

    Accept a trial when its score is at least the threshold. Try every
    distinct score as the threshold, plus +inf. At each one, P_miss is the
    fraction of target scores below it and P_fa is the fraction of non-target
    scores at or above it. Return (P_miss + P_fa) / 2 at the threshold where
    |P_miss - P_fa| is smallest.
    """
    raise NotImplementedError("equal_error_rate")


def min_dcf(
    target_scores: np.ndarray, nontarget_scores: np.ndarray, p_target: float = 0.05, c_miss: float = 1.0, c_fa: float = 1.0
) -> float:
    """The normalized minimum detection cost over the same thresholds as equal_error_rate.

    C(t) = c_miss * P_miss(t) * p_target + c_fa * P_fa(t) * (1 - p_target),
    divided by min(c_miss * p_target, c_fa * (1 - p_target)).
    """
    raise NotImplementedError("min_dcf")


def stats_pooling(frames: np.ndarray) -> np.ndarray:
    """Statistics pooling: the mean and standard deviation of every channel over time.

    Args:
        frames: Frame-level activations, shape (T, C): T frames, C channels.

    Returns:
        Shape (2 * C,): the C channel means followed by the C channel standard
        deviations. Use the population standard deviation (divide by T).
    """
    raise NotImplementedError("stats_pooling")


def attentive_stats_pooling(frames: np.ndarray, attn_logits: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Attentive statistics pooling, as in ECAPA-TDNN.

    Args:
        frames: Frame-level activations h, shape (T, C).
        attn_logits: Attention scores before the softmax, shape (T, C), or (T,)
            to share one score per frame across channels.
        floor: A small lower bound on the weighted variance, so the square
            root stays finite and its gradient is defined.

    Softmax each channel's scores over the T frames to get weights alpha, then
    mean = sum_t alpha_t h_t and std = sqrt(max(sum_t alpha_t h_t^2 - mean^2, floor)).
    Returns shape (2 * C,): the weighted means followed by the weighted standard deviations.
    """
    raise NotImplementedError("attentive_stats_pooling")


def enroll(embeddings: np.ndarray) -> np.ndarray:
    """Combine several enrollment embeddings of one speaker into one.

    L2-normalize each row, average the rows, then L2-normalize the average.

    Args:
        embeddings: One enrollment utterance per row, shape (N, D).

    Returns:
        Shape (D,), unit length.
    """
    raise NotImplementedError("enroll")


def as_norm(score: float, enroll_cohort_scores: np.ndarray, test_cohort_scores: np.ndarray, top_k: int) -> float:
    """Adaptive symmetric score normalization (AS-norm) of one trial score.

    Args:
        score: The raw cosine score s of the trial (enrollment e, test t).
        enroll_cohort_scores: Cosine scores of e against every imposter in the cohort.
        test_cohort_scores: Cosine scores of t against the same cohort.
        top_k: How many of the highest cohort scores to keep for each side.

    Keep the top_k largest scores on each side, take their mean mu and
    population standard deviation sigma, and return
    0.5 * ((s - mu_e) / sigma_e + (s - mu_t) / sigma_t).
    """
    raise NotImplementedError("as_norm")


# The Run experiment -----------------------------------------------------------

RATE = 16000
VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def make_speaker(rng):
    """A synthetic voice: pitch, vocal-tract scale, and spectral tilt."""
    return {"f0": rng.uniform(85, 240), "scale": rng.uniform(0.82, 1.22), "tilt": rng.uniform(0.85, 0.97)}


def speak(voice, seconds, rng):
    """A buzz through formant resonators, one random vowel every 120 to 300 ms."""
    out = []
    while sum(map(len, out)) < seconds * RATE:
        n = int(rng.uniform(0.12, 0.3) * RATE)
        f0 = voice["f0"] * (1 + 0.03 * np.sin(2 * np.pi * 5 * np.arange(n) / RATE))
        phase = np.cumsum(f0 / RATE)
        source = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
        source = lfilter([1.0], [1.0, -voice["tilt"]], source)
        sound = source
        for f, bw in zip(np.array(VOWELS[rng.integers(len(VOWELS))]) * voice["scale"], (80, 100, 140)):
            r = np.exp(-np.pi * bw / RATE)
            sound = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], sound)
        out.append(sound * np.hanning(n) ** 0.3)
    audio = np.concatenate(out)[: int(seconds * RATE)]
    return audio / (np.max(np.abs(audio)) + 1e-9)


def voice_features(audio):
    """The mean log power spectrum up to 2.5 kHz, in 15.6 Hz bins over 64 ms frames, shape (160,)."""
    frames = np.lib.stride_tricks.sliding_window_view(audio, 1024)[::160] * np.hanning(1024)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(power[:, 1:161] + 1e-8).mean(axis=0)


def trials(embeddings, speakers):
    """Cosine scores of every pair, split into same-speaker and different-speaker trials."""
    i, j = np.triu_indices(len(speakers), 1)
    scores = cosine_scores(embeddings[i], embeddings[j])
    same = speakers[i] == speakers[j]
    return scores[same], scores[~same]


if __name__ == "__main__":
    # The worked example from the chapter: cosines 0.5 and 0.4, scale 30.
    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    x = torch.tensor([[0.5, 0.4, math.sqrt(1 - 0.25 - 0.16)]], dtype=torch.float64)
    for m in (0.0, 0.2):
        print(f"Margin {m}: loss {float(aam_softmax_loss(x, w, torch.tensor([0]), 30.0, m)):.4f} nats")
    print(f"Worked trials: EER {equal_error_rate(np.array([0.9, 0.8, 0.7, 0.4]), np.array([0.6, 0.3, 0.2, 0.1])):.2f}")

    rng = np.random.default_rng(0)
    voices = [make_speaker(rng) for _ in range(70)]  # 60 to train on, 10 never seen in training
    demo_rng = np.random.default_rng(1)  # a separate stream, so the clips below do not change the experiment
    for k in (0, 1):
        play(0.5 * speak(voices[k], 1.0, demo_rng).astype(np.float32), RATE, label=f"Synthetic voice {k}")
    feats, who = [], []
    for s, voice in enumerate(voices):
        for _ in range(6 if s < 60 else 8):
            feats.append(voice_features(speak(voice, 1.0, rng)))
            who.append(s)
    feats, who = np.array(feats), np.array(who)
    train, test = who < 60, who >= 60
    mean, std = feats[train].mean(0), feats[train].std(0) + 1e-6
    feats = (feats - mean) / std

    tgt, non = trials(feats[test], who[test])
    print(f"Raw features, 10 unseen voices: EER {equal_error_rate(tgt, non):.1%}, minDCF {min_dcf(tgt, non):.3f}")

    x, y = torch.tensor(feats[train], dtype=torch.float32), torch.tensor(who[train])
    for final_margin in (0.0, 0.2):
        torch.manual_seed(0)
        project = torch.nn.Linear(160, 32)
        centers = torch.nn.Parameter(torch.randn(60, 32))
        optimizer = torch.optim.Adam(list(project.parameters()) + [centers], lr=0.01)
        history = []
        for step in range(300):
            margin = final_margin * min(1.0, step / 100)  # warm the margin up over 100 steps
            loss = aam_softmax_loss(project(x), centers, y, 30.0, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        with torch.no_grad():
            embed = project(torch.tensor(feats, dtype=torch.float32)).numpy()
        tgt, non = trials(embed[test], who[test])
        print(f"Trained with margin {final_margin}: EER {equal_error_rate(tgt, non):.1%}, "
              f"minDCF {min_dcf(tgt, non):.3f}")
    # The same margin-0.2 embedding, now with the chapter's back-end steps. embed holds the last model.
    embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    held, held_who = embed[test], who[test]
    first, second = np.triu_indices(len(held_who), 1)
    raw = np.sum(held[first] * held[second], axis=1)
    cohort_scores = held @ embed[train].T  # every unseen clip against the 360 training clips' embeddings
    normed = np.array([as_norm(raw[n], cohort_scores[first[n]], cohort_scores[second[n]], 30) for n in range(len(raw))])
    same = held_who[first] == held_who[second]
    print(f"AS-norm, top 30 of 360 cohort clips: EER {equal_error_rate(normed[same], normed[~same]):.1%}, "
          f"minDCF {min_dcf(normed[same], normed[~same]):.3f}")
    speakers = np.unique(held_who)
    for n_enroll in (1, 3):  # enroll from the first n clips, test on the last 5 clips of every voice
        tar, imp = [], []
        for a in speakers:
            enrolled = enroll(held[held_who == a][:n_enroll])
            for b in speakers:
                (tar if a == b else imp).extend(held[held_who == b][3:] @ enrolled)
        print(f"Enrolled with {n_enroll} clip{'s' if n_enroll > 1 else ''}: "
              f"EER {equal_error_rate(np.array(tar), np.array(imp)):.1%}, minDCF {min_dcf(np.array(tar), np.array(imp)):.3f}")
    show_plot(np.array(history), label="AAM-softmax training loss, margin 0.2", xlabel="Step", ylabel="Loss (nats)")
