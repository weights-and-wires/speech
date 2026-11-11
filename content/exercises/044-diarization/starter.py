"""Diarization: who spoke when, and how to score it.

Diarization labels are one integer per frame, with -1 for frames where nobody
speaks. Overlapped speech uses one boolean row per speaker instead.
"""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import lfilter

from speechlab import play, show_image, show_plot


def aam_softmax_loss(
    embeddings: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor, scale: float = 30.0, margin: float = 0.2
) -> torch.Tensor:
    """Provided in full: you built this in the speaker verification chapter.

    Additive angular margin (ArcFace) softmax loss, averaged over the batch.

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
    cos = torch.nn.functional.normalize(embeddings, dim=1) @ torch.nn.functional.normalize(weights, dim=1).T
    rows = torch.arange(cos.shape[0])
    cos_t = cos[rows, labels]
    sin_t = torch.sqrt(torch.clamp(1.0 - cos_t**2, min=0.0))
    shifted = cos_t * math.cos(margin) - sin_t * math.sin(margin)
    shifted = torch.where(cos_t > math.cos(math.pi - margin), shifted, cos_t - margin * math.sin(margin))
    logits = cos.clone()
    logits[rows, labels] = shifted
    return torch.nn.functional.cross_entropy(scale * logits, labels)


def affinity_matrix(embeddings: np.ndarray, keep: float = 0.3) -> np.ndarray:
    """A pruned, symmetric cosine affinity between segments, shape (N, N).

    Take the cosine similarity of every pair of rows and set negative values
    to 0. In each row, keep the ceil(keep * N) largest entries and set the rest
    to 0. Symmetrize with the elementwise maximum of the matrix and its
    transpose.
    """
    raise NotImplementedError("affinity_matrix")


def speaker_count(affinity: np.ndarray, max_speakers: int) -> int:
    """The number of speakers from the largest eigengap of the normalized Laplacian.

    L = I - D^(-1/2) A D^(-1/2), with D the diagonal of row sums. With the
    eigenvalues sorted ascending as lambda_1 <= lambda_2 <= ..., return the k in
    1..max_speakers that maximizes lambda_(k+1) - lambda_k.
    """
    raise NotImplementedError("speaker_count")


def kmeans_rows(points: np.ndarray, k: int, n_iter: int = 50) -> np.ndarray:
    """Provided in full: a small deterministic k-means: farthest-point seeding, then Lloyd's updates. Returns labels."""
    centers = [points[0]]
    for _ in range(1, k):
        dist = np.min([np.sum((points - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(points[np.argmax(dist)])
    centers = np.array(centers)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return labels


def spectral_cluster(embeddings: np.ndarray, max_speakers: int = 8, keep: float = 0.3) -> np.ndarray:
    """Cluster segment embeddings into speakers, shape (N,) of int labels.

    Build affinity_matrix, pick k with speaker_count, take the eigenvectors of
    the normalized Laplacian for its k smallest eigenvalues as the columns of
    an (N, k) matrix, scale each row to unit length, and cluster the rows with
    kmeans_rows.
    """
    raise NotImplementedError("spectral_cluster")


def diarization_error_rate(reference: np.ndarray, hypothesis: np.ndarray) -> float:
    """Frame-level DER with the best one-to-one speaker mapping, no collar.

    Both arrays hold one label per frame, -1 for nobody. Missed speech is a
    reference speech frame the hypothesis leaves silent, a false alarm is the
    reverse, and confusion is a frame where both have speech but the labels
    differ after mapping hypothesis speakers onto reference speakers to match
    the most frames. DER = (missed + false alarm + confusion) / reference
    speech frames.
    """
    raise NotImplementedError("diarization_error_rate")


def der_collar_overlap(reference: np.ndarray, hypothesis: np.ndarray, collar: int = 0, score_overlap: bool = True) -> float:
    """DER for overlapping speakers, with a forgiveness collar around reference boundaries.

    Args:
        reference: Boolean speaker activity, shape (R, T): row r, column t is
            True when reference speaker r talks in frame t. Several rows can be
            True in one frame, which is overlapped speech.
        hypothesis: The same layout for the system's speakers, shape (H, T).
        collar: Frames to forgive on each side of every reference boundary.
        score_overlap: When False, drop every frame where two or more
            reference speakers talk.

    A boundary is a frame edge where one reference speaker starts or stops,
    counting the start and end of the recording as silence. A boundary between
    frames t - 1 and t forgives frames t - collar through t + collar - 1, and
    the forgiven frames leave both the errors and the denominator. Only the
    reference boundaries count, never the hypothesis ones.

    On each scored frame, with n_ref and n_hyp active speakers and n_ok of
    them matched, the error is max(n_ref, n_hyp) - n_ok. The speaker mapping
    is the one-to-one assignment that maximizes the matched speaker-frames
    over the scored frames. DER = total error / total n_ref.
    """
    raise NotImplementedError("der_collar_overlap")


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


if __name__ == "__main__":
    # Overlapped speech you can check by hand: A talks in frames 0-7, B in frames 5-11, and the
    # system misses B until frame 8.
    ref = np.zeros((2, 12), dtype=bool)
    ref[0, :8], ref[1, 5:] = True, True
    hyp = np.zeros((2, 12), dtype=bool)
    hyp[0, :8], hyp[1, 8:] = True, True
    print(f"DER with overlap and no collar: {der_collar_overlap(ref, hyp):.1%}; "
          f"with a collar of 1 frame: {der_collar_overlap(ref, hyp, collar=1):.1%}; "
          f"without scoring overlap: {der_collar_overlap(ref, hyp, score_overlap=False):.1%}")

    rng = np.random.default_rng(0)
    voices = [make_speaker(rng) for _ in range(70)]  # 60 to train embeddings on, 10 never seen
    feats, who = [], []
    for s, voice in enumerate(voices):
        for _ in range(6 if s < 60 else 8):
            feats.append(voice_features(speak(voice, 1.0, rng)))
            who.append(s)
    feats, who = np.array(feats), np.array(who)
    train = who < 60
    mean, std = feats[train].mean(0), feats[train].std(0) + 1e-6
    feats = (feats - mean) / std

    # Embeddings trained with the angular margin from the previous chapter.
    x, y = torch.tensor(feats[train], dtype=torch.float32), torch.tensor(who[train])
    torch.manual_seed(0)
    project = torch.nn.Linear(160, 32)
    centers = torch.nn.Parameter(torch.randn(60, 32))
    optimizer = torch.optim.Adam(list(project.parameters()) + [centers], lr=0.01)
    for step in range(300):
        margin = 0.2 * min(1.0, step / 100)  # warm the margin up over 100 steps
        loss = aam_softmax_loss(project(x), centers, y, 30.0, margin)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Diarization: three unseen voices take turns, with pauses.
    parts, reference = [], []
    for turn in range(9):
        speaker = [60, 61, 62][turn % 3] if turn < 6 else [62, 60, 61][turn % 3]
        seconds = rng.uniform(0.8, 1.6)
        parts.append(0.6 * speak(voices[speaker], seconds, rng))
        reference += [speaker - 60] * (len(parts[-1]) // 160)
        gap = int(rng.uniform(0.1, 0.3) * RATE)
        parts.append(np.zeros(gap))
        reference += [-1] * (gap // 160)
    audio = np.concatenate(parts)
    reference = np.array(reference[: len(audio) // 160])
    play(audio.astype(np.float32), RATE, label="Three synthetic voices taking turns")

    window, hop = 50, 25  # 0.5 s windows every 0.25 s, in 10 ms frames
    starts = [s for s in range(0, len(reference) - window + 1, hop) if np.mean(reference[s : s + window] >= 0) > 0.5]
    segs = [voice_features(audio[s * 160 : (s + window) * 160]) for s in starts]
    with torch.no_grad():
        seg_embed = project(torch.tensor((np.array(segs) - mean) / std, dtype=torch.float32)).numpy()
    labels = spectral_cluster(seg_embed, max_speakers=6)
    show_image(affinity_matrix(seg_embed), label="Segment affinity", xlabel="Segment", ylabel="Segment")
    centers_t = np.array(starts) + window // 2
    nearest = np.argmin(np.abs(np.arange(len(reference))[:, None] - centers_t[None, :]), axis=1)
    hypothesis = np.where(reference >= 0, labels[nearest], -1)  # oracle speech regions
    seconds = np.arange(len(reference)) / 100
    show_plot((seconds, reference), (seconds, hypothesis), labels=["reference", "hypothesis"],
              label="Who spoke when (-1 is silence)", xlabel="Time (s)", ylabel="Speaker")
    print(f"Spectral clustering found {len(set(labels.tolist()))} speakers in {len(starts)} segments; "
          f"DER {diarization_error_rate(reference, hypothesis):.1%}")
