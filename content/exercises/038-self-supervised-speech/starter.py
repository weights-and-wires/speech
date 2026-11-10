"""Self-supervised speech: the pieces of wav2vec 2.0, HuBERT, and BEST-RQ.

Every time axis is in encoder frames unless a name says samples. One frame of
wav2vec 2.0 or HuBERT is 320 samples, 20 ms at 16 kHz. Losses are means over
the masked frames, in nats.
"""

import numpy as np

from speechlab import has_asset, play, show_plot, show_spectrogram


def conv_output_length(n_samples: int, kernels: list[int], strides: list[int]) -> int:
    """The number of frames a stack of unpadded 1-D convolutions produces.

    Each layer maps a length L to floor((L - kernel) / stride) + 1. Returns 0
    when the input is shorter than the stack can read.
    """
    raise NotImplementedError("conv_output_length")


def receptive_field(kernels: list[int], strides: list[int]) -> tuple[int, int]:
    """The receptive field and the hop of one output frame, both in samples.

    The receptive field is 1 + sum_i (k_i - 1) * prod_{j < i} s_j, and the hop
    is the product of all the strides.
    """
    raise NotImplementedError("receptive_field")


def span_mask(starts: np.ndarray, span: int) -> np.ndarray:
    """Expand span starts into a mask, shape (T,) of bool.

    Frame t is masked when some start s satisfies s <= t < s + span. Spans can
    overlap, and a span that runs past the end is cut at frame T - 1.
    """
    raise NotImplementedError("span_mask")


def sample_distractors(masked: np.ndarray, n_distractors: int, rng: np.random.Generator) -> np.ndarray:
    """Draw distractor frames for each masked frame, shape (M, K) of int.

    `masked` holds the M masked frame indices. Row i holds K frame indices
    drawn uniformly, with replacement, from the other M - 1 masked frames, so
    row i never contains masked[i]. Needs M >= 2.
    """
    raise NotImplementedError("sample_distractors")


def contrastive_loss(
    context: np.ndarray,
    targets: np.ndarray,
    masked: np.ndarray,
    distractors: np.ndarray,
    kappa: float = 0.1,
) -> float:
    """The wav2vec 2.0 contrastive loss, averaged over masked frames.

    Args:
        context: Transformer outputs c_t, shape (T, D).
        targets: Quantized targets q_t, shape (T, D).
        masked: The M masked frame indices, shape (M,).
        distractors: Distractor frame indices, shape (M, K).
        kappa: The temperature.

    For masked frame t, the candidates are q_t followed by the K distractors'
    q vectors. The loss is -log softmax(cos(c_t, candidate) / kappa)[0].
    """
    raise NotImplementedError("contrastive_loss")


def diversity_loss(logits: np.ndarray) -> float:
    """The wav2vec 2.0 diversity loss, (1 / GV) * sum_g sum_v pbar log pbar.

    `logits` has shape (N, G, V): N frames, G codebooks, V entries each. pbar
    is the softmax over V averaged over the N frames, without Gumbel noise.
    The loss is -log(V) / V when every entry is used equally, and 0 when each
    codebook collapses onto one entry.
    """
    raise NotImplementedError("diversity_loss")


def masked_prediction_loss(
    outputs: np.ndarray,
    projection: np.ndarray,
    code_embeddings: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    tau: float = 0.1,
) -> float:
    """HuBERT's masked prediction loss, averaged over masked frames.

    Args:
        outputs: Transformer outputs o_t, shape (T, D).
        projection: The matrix A, shape (E, D).
        code_embeddings: One embedding e_c per cluster, shape (C, E).
        targets: The k-means cluster of every frame, shape (T,) of int.
        mask: Which frames were masked, shape (T,) of bool.
        tau: The temperature.

    The logit of cluster c at frame t is cos(A o_t, e_c) / tau. Only masked
    frames count.
    """
    raise NotImplementedError("masked_prediction_loss")


def best_rq_targets(
    features: np.ndarray, projection: np.ndarray, codebook: np.ndarray, stack: int = 4
) -> np.ndarray:
    """BEST-RQ labels from a frozen random-projection quantizer, shape (T // stack,).

    Args:
        features: Normalized log-mel frames, shape (T, F).
        projection: The frozen random matrix A, shape (h, stack * F).
        codebook: The frozen random codebook, shape (n, h).
        stack: How many consecutive frames make one input vector.

    Stack each group of `stack` frames into one vector x (dropping any leftover
    frames at the end), project it to A x, and return the index of the
    codeword closest to it after both are scaled to unit length.
    """
    raise NotImplementedError("best_rq_targets")


def kmeans(features: np.ndarray, k: int, n_iter: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with k-means++ seeding.

    Args:
        features: The points, shape (N, D).
        k: The number of clusters.
        n_iter: The maximum number of assign-then-update rounds.
        seed: Seeds np.random.default_rng for the k-means++ draws.

    Returns:
        (centroids, labels): shape (k, D) and (N,) of int. Every label is the
        index of the nearest centroid, and every centroid with members is the
        mean of its members.
    """
    raise NotImplementedError("kmeans")


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, n_classes: int, l2: float) -> np.ndarray:
    """A ridge-regression probe onto one-hot labels, shape (D + 1, n_classes).

    Append a column of ones to the (N, D) features, so the last row of the
    result is the bias. Minimize ||X W - Y||^2 + l2 * ||W[:D]||^2, which leaves
    the bias unpenalized, in closed form.
    """
    raise NotImplementedError("fit_linear_probe")


def probe_accuracy(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    """The fraction of rows whose largest probe output is at their label."""
    raise NotImplementedError("probe_accuracy")


# The Run experiment -----------------------------------------------------------

KERNELS = [10, 3, 3, 3, 3, 2, 2]
STRIDES = [5, 2, 2, 2, 2, 2, 2]
SPEAKERS = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]


def load_digits(per_pair: int):
    """fsdd clips resampled from 8 kHz to 16 kHz, with digit and speaker labels."""
    from scipy.signal import resample_poly

    from speechlab import asset_path, read_audio

    folder = asset_path("fsdd") / "free-spoken-digit-dataset-1.0.10" / "recordings"
    clips, digits, speakers = [], [], []
    for s, name in enumerate(SPEAKERS):
        for digit in range(10):
            for i in range(per_pair):
                audio, rate = read_audio(folder / f"{digit}_{name}_{i}.wav")
                clips.append(resample_poly(audio, 16000 // rate, 1).astype(np.float32))
                digits.append(digit)
                speakers.append(s)
    return clips, np.array(digits), np.array(speakers)


def layer_features(model_class, name: str, clips, device: str):
    """Mean-pooled hidden states of every layer, shape (clips, layers, 768), and one layer's frames."""
    import torch

    from speechlab import asset_path

    model = model_class.from_pretrained(str(asset_path(name))).to(device).eval()
    pooled, frames = [], []
    with torch.inference_mode():
        for audio in clips:
            # Both preprocessor configs set do_normalize, so standardize each clip.
            x = (audio - audio.mean()) / (audio.std() + 1e-7)
            hidden = model(torch.from_numpy(x)[None].to(device), output_hidden_states=True).hidden_states
            pooled.append(torch.stack([h[0].mean(0) for h in hidden]).cpu().numpy())
            frames.append(hidden[6][0].cpu().numpy())
    del model
    return np.stack(pooled), frames


def layer_curve(pooled: np.ndarray, labels: np.ndarray, train: np.ndarray, n_classes: int) -> np.ndarray:
    """Held-out probe accuracy for each layer."""
    accuracy = []
    for layer in range(pooled.shape[1]):
        x = pooled[:, layer]
        mean, std = x[train].mean(0), x[train].std(0) + 1e-6
        x = (x - mean) / std
        w = fit_linear_probe(x[train], labels[train], n_classes, l2=100.0)
        accuracy.append(probe_accuracy(w, x[~train], labels[~train]))
    return np.array(accuracy)


def dedupe(units: np.ndarray) -> list[int]:
    keep = np.concatenate([[True], units[1:] != units[:-1]])
    return units[keep].tolist()


def synthetic_fallback():
    """What the Run does without the models: the same pieces on three synthetic vowels."""
    rate = 16000
    t = np.arange(rate // 3) / rate
    # Three steady "vowels", each a 120 Hz buzz shaped by two formant-like peaks.
    vowels = []
    for f1, f2 in ((300, 2300), (700, 1200), (400, 800)):
        harmonics = np.arange(1, 60) * 120.0
        gain = np.exp(-(((harmonics - f1) / 120) ** 2)) + 0.5 * np.exp(-(((harmonics - f2) / 150) ** 2))
        vowels.append(np.sum(gain[:, None] * np.sin(2 * np.pi * harmonics[:, None] * t), axis=0))
    audio = np.concatenate(vowels)
    audio = (0.5 * audio / np.max(np.abs(audio))).astype(np.float32)
    play(audio, rate, label="Three synthetic vowels")
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160] * np.hanning(400)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    show_spectrogram(spec.T, rate, 160, label="Their spectrogram")
    feats = np.log(spec + 1e-3)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    rng = np.random.default_rng(0)
    projection = rng.normal(size=(16, 4 * feats.shape[1])) * np.sqrt(2 / (16 + 4 * feats.shape[1]))
    labels = best_rq_targets(feats, projection, rng.normal(size=(64, 16)))
    print(f"BEST-RQ labels for {labels.size} stacked frames, deduplicated: {dedupe(labels)}")
    _, units = kmeans(feats, 3, seed=0)
    print(f"k-means units with k = 3, deduplicated: {dedupe(units)}")


if __name__ == "__main__":
    n = conv_output_length(16000, KERNELS, STRIDES)
    field, hop = receptive_field(KERNELS, STRIDES)
    print(f"One second at 16 kHz gives {n} frames; each frame sees {field} samples and moves {hop}.")
    starts = np.random.default_rng(0).random(5000) < 0.065
    print(f"Span masking with p = 0.065 and M = 10 masks {span_mask(starts, 10).mean():.1%} of frames.")

    try:
        import torch
        from transformers import HubertModel, Wav2Vec2Model
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()
        have_models = all(has_asset(a) for a in ("wav2vec2-base", "hubert-base", "fsdd"))
    except ImportError:
        have_models = False

    if not have_models:
        print("This run needs the wav2vec2-base, hubert-base, and fsdd assets and transformers;")
        print("without them it runs the same pieces on a synthetic sweep instead.")
        synthetic_fallback()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        per_pair = 50 if device == "cuda" else 4
        if device == "cpu":
            print(f"No GPU, so this is the small version: {per_pair} clips per speaker and digit instead of 50.")
        clips, digits, speakers = load_digits(per_pair)
        print(f"{len(clips)} clips, {sum(map(len, clips)) / 16000:.0f} s of audio, 6 speakers.")
        play(clips[3 * per_pair], 16000, label=f"The digit 3, spoken by {SPEAKERS[0]}")

        curves, hubert_frames = {}, None
        for name, cls in (("wav2vec2-base", Wav2Vec2Model), ("hubert-base", HubertModel)):
            pooled, frames = layer_features(cls, name, clips, device)
            # Digits: train on five speakers, test on the sixth, averaged over all six.
            digit_acc = np.mean([layer_curve(pooled, digits, speakers != s, 10) for s in range(6)], axis=0)
            # Speakers: train on the digits 0 to 4, test on the digits 5 to 9.
            speaker_acc = layer_curve(pooled, speakers, digits < 5, 6)
            curves[name] = (digit_acc, speaker_acc)
            print(f"{name}: digit accuracy peaks at layer {digit_acc.argmax()} ({digit_acc.max():.0%}), "
                  f"speaker accuracy at layer {speaker_acc.argmax()} ({speaker_acc.max():.0%})")
            if name == "hubert-base":
                hubert_frames = frames

        layers = np.arange(13)
        show_plot((layers, curves["wav2vec2-base"][0]), (layers, curves["hubert-base"][0]),
                  labels=["wav2vec2-base", "hubert-base"], label="Digit probe, held-out speaker",
                  xlabel="Layer (0 is the CNN output)", ylabel="Accuracy", ylim=(0, 1))
        show_plot((layers, curves["wav2vec2-base"][1]), (layers, curves["hubert-base"][1]),
                  labels=["wav2vec2-base", "hubert-base"], label="Speaker probe, held-out digits",
                  xlabel="Layer (0 is the CNN output)", ylabel="Accuracy", ylim=(0, 1))

        k = 100 if device == "cuda" else 50
        stacked = np.concatenate(hubert_frames)
        centroids, _ = kmeans(stacked[:: max(1, stacked.shape[0] // 20000)], k, seed=0)
        units = [np.argmin(((f[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1) for f in hubert_frames]
        for s in (0, 1):
            clip = s * 10 * per_pair + 3 * per_pair
            print(f"'3' by {SPEAKERS[s]:8s} -> {len(dedupe(units[clip]))} units: {dedupe(units[clip])}")
        bags = np.stack([np.bincount(u, minlength=k) / len(u) for u in units])
        digit_bag = np.mean([layer_curve(bags[:, None], digits, speakers != s, 10)[0] for s in range(6)])
        speaker_bag = layer_curve(bags[:, None], speakers, digits < 5, 6)[0]
        print(f"A bag of {k} HuBERT layer-6 units predicts the digit {digit_bag:.0%} of the time "
              f"and the speaker {speaker_bag:.0%} of the time (chance: 10% and 17%).")
