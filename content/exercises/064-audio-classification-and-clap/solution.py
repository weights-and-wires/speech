"""Audio classification and CLAP: the worked solution."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import lfilter
from scipy.signal import stft as scipy_stft

from speechlab import show_image


# Classification ------------------------------------------------------------------


def ast_patch_grid(n_mels: int, n_frames: int, patch: int = 16, stride: int = 10) -> tuple[int, int]:
    """How many patches AST cuts along frequency and along time.

    Patches are patch x patch squares taken every `stride` bins and frames,
    keeping only patches that fit. Returns (n_freq, n_time), each
    floor((size - patch) / stride) + 1.
    """
    return (n_mels - patch) // stride + 1, (n_frames - patch) // stride + 1


def patchify(spec: np.ndarray, patch: int = 16, stride: int = 10) -> np.ndarray:
    """Cut a (n_mels, n_frames) spectrogram into overlapping square patches.

    Returns (n_freq * n_time, patch * patch). Row i_f * n_time + i_t holds
    spec[i_f*stride : i_f*stride + patch, i_t*stride : i_t*stride + patch]
    flattened in row-major order, the order a stride-10 Conv2d produces.
    """
    n_f, n_t = ast_patch_grid(*spec.shape, patch, stride)
    rows = stride * np.arange(n_f)[:, None, None, None] + np.arange(patch)[None, None, :, None]
    cols = stride * np.arange(n_t)[None, :, None, None] + np.arange(patch)[None, None, None, :]
    return spec[rows, cols].reshape(n_f * n_t, patch * patch)


def multilabel_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy of independent sigmoids, averaged over clips and classes.

    logits, targets: (N, C). targets holds 0.0 or 1.0, and a clip can have
    several 1s. Computes mean of -[y log s(x) + (1 - y) log(1 - s(x))] with
    s the sigmoid, in the form max(x, 0) - x*y + log(1 + exp(-|x|)) so a
    logit of +-1000 neither overflows nor returns infinity.
    """
    return (logits.clamp(min=0) - logits * targets + torch.log1p(torch.exp(-logits.abs()))).mean()


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision of one class: the mean of precision@rank over the ranks of the positives.

    scores: (n,) real numbers, higher means more likely positive. labels: (n,)
    of 0 and 1. Rank clips by descending score (assume no ties). Returns NaN
    when there are no positives.
    """
    labels = np.asarray(labels)[np.argsort(-np.asarray(scores, float), kind="stable")]
    if labels.sum() == 0:
        return float("nan")
    precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
    return float(precision[labels == 1].mean())


def mean_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """mAP: the mean of average_precision over the classes that have a positive clip.

    scores, labels: (N, C). A class whose column of labels is all 0 is skipped.
    """
    aps = [average_precision(scores[:, c], labels[:, c]) for c in range(scores.shape[1])]
    return float(np.nanmean(aps))


def clap_loss(audio_emb: torch.Tensor, text_emb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """The symmetric contrastive (InfoNCE) loss of CLAP.

    audio_emb, text_emb: (N, d), where row i of each is a matching pair.
    tau: a positive scalar temperature. L2-normalize both, form
    S = a @ t.T / tau, and return the mean of the audio-to-text and
    text-to-audio cross-entropies with the diagonal as the correct class.
    """
    a = F.normalize(audio_emb, dim=-1)
    t = F.normalize(text_emb, dim=-1)
    logits = a @ t.T / tau
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def zero_shot_classify(audio_emb: torch.Tensor, class_text_emb: torch.Tensor) -> torch.Tensor:
    """The index of the class prompt with the highest cosine similarity, shape (N,)."""
    return (F.normalize(audio_emb, dim=-1) @ F.normalize(class_text_emb, dim=-1).T).argmax(dim=-1)


# Run helpers -------------------------------------------------------------------------

RATE = 16000
N_FFT, HOP = 512, 160  # 32 ms window, 10 ms hop
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
          "e": (530, 1840, 2480), "o": (570, 840, 2410)}


def speak(letters, f0, formant_scale, rng, each=0.25):
    """A synthetic voice: a jittered pulse train through three formant resonators per vowel."""
    out, labels = [], []
    for letter in letters:
        n = int(each * RATE)
        pitch = f0 * (1 + 0.03 * np.sin(2 * np.pi * 3 * np.arange(n) / RATE + rng.uniform(0, 6)))
        src = (np.diff(np.floor(np.cumsum(pitch / RATE)), prepend=0) > 0).astype(float)
        y = src
        for f, bw in zip(VOWELS[letter], (80, 100, 140)):
            f = f * formant_scale
            r = np.exp(-np.pi * bw / RATE)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
        out.append(y / np.abs(y).max() * np.hanning(n) ** 0.2)
        labels += [letter] * (n // HOP)
    x = np.concatenate(out)
    return 0.5 * x / np.abs(x).max(), labels


def stft(x):
    return scipy_stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")[2]


def mel_matrix(n_mels=40):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = 700 * (10 ** (np.linspace(0, mel(RATE / 2), n_mels + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(N_FFT, 1 / RATE)
    fb = np.zeros((n_mels, bins.size))
    for m in range(n_mels):
        lo, mid, hi = hz[m : m + 3]
        fb[m] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    return fb


def tagging_demo(rng):
    n, c = 600, 4
    labels = (rng.random((n, c)) < [0.5, 0.2, 0.05, 0.02]).astype(float)
    scores = 1.5 * labels + rng.normal(size=(n, c))  # a noisy tagger: positives sit 1.5 above negatives
    print("\nA toy tagger on 600 clips and 4 classes (prevalence 50%, 20%, 5%, 2%):")
    for k in range(c):
        print(f"  class {k}: AP {average_precision(scores[:, k], labels[:, k]):.3f}, chance level {labels[:, k].mean():.3f}")
    print(f"  mAP {mean_average_precision(scores, labels):.3f}")
    prompts = torch.tensor(rng.normal(size=(5, 16)), dtype=torch.float32)
    truth = torch.tensor(rng.integers(0, 5, size=200))
    clips = prompts[truth] * 0.5 + torch.tensor(rng.normal(size=(200, 16)), dtype=torch.float32)
    accuracy = (zero_shot_classify(clips, prompts) == truth).float().mean().item()
    print(f"Zero-shot over 5 synthetic prompts: {accuracy:.0%} of 200 clips land on their own prompt (chance 20%)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_f, n_t = ast_patch_grid(128, 1024)
    print(f"AST on 128 mel bins x 1024 frames: {n_f} x {n_t} = {n_f * n_t} patches of 16 x 16")
    spec = np.log(mel_matrix(128) @ np.abs(stft(speak("aiu", 120.0, 1.0, rng)[0])) ** 2 + 1e-8)
    patches = patchify(spec)
    print(f"a {spec.shape[1]}-frame log-mel clip gives {patches.shape[0]} patches, each a row of {patches.shape[1]} numbers")
    show_image(patches[:40].T, label="The first 40 patches, one per column", xlabel="Patch", ylabel="Pixel in the patch")
    tagging_demo(rng)
