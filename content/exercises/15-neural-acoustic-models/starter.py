"""Neural acoustic models: context windows, subsampling, and a frame classifier.

Fill in the functions and the two modules below, then press Run to train a
frame classifier on synthetic phones at -5 dB SNR with six context widths,
see its frame posteriors for one word, and pass the word through a 4x
convolutional front end. Press Check to test your code.

Feature matrices have one row per frame: shape (T, n_mels).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from torch import nn

from speechlab import play, show_image, show_plot, show_spectrogram


def stack_frames(features: np.ndarray, left: int, right: int) -> np.ndarray:
    """Splice each frame with its neighbors into one long vector.

    Args:
        features: Shape (T, D).
        left: Frames of context before each frame.
        right: Frames of context after each frame.

    Returns:
        Shape (T, (left + 1 + right) * D). Row t is frames t - left .. t + right
        of `features`, concatenated in time order. A neighbor before frame 0
        repeats frame 0, and one after frame T - 1 repeats frame T - 1.
    """
    raise NotImplementedError("stack_frames")


def conv_output_length(length, kernel: int, stride: int, padding: int = 0, dilation: int = 1):
    """The number of output frames of a convolution, as torch computes it.

    Args:
        length: Input length, an int or an integer tensor or array.
        kernel, stride, padding, dilation: The convolution's settings.

    Returns:
        floor((length + 2 padding - dilation (kernel - 1) - 1) / stride) + 1,
        with the same type as `length`.
    """
    raise NotImplementedError("conv_output_length")


def subsampled_length(length):
    """Output frames of two unpadded convolutions with kernel 3 and stride 2.

    Args:
        length: Input length, an int or an integer tensor or array.

    Returns:
        The output length after both convolutions, same type as `length`.
    """
    raise NotImplementedError("subsampled_length")


class Conv2dSubsampling(nn.Module):
    """A convolutional front end that keeps one frame in four.

    Holds three submodules with these exact names:
    - conv1: nn.Conv2d(1, channels, kernel_size=3, stride=2), no padding.
    - conv2: nn.Conv2d(channels, channels, kernel_size=3, stride=2), no padding.
    - out: nn.Linear(channels * subsampled_length(n_mels), d_model).

    forward(x, lengths) takes x of shape (N, T, n_mels), read as a
    one-channel image with time as height, and lengths of shape (N,). Each
    convolution is followed by a ReLU. The features that reach `out` for one
    output frame are ordered channel-major: channel c, frequency f sits at
    index c * subsampled_length(n_mels) + f. It returns (y, out_lengths): y
    has shape (N, subsampled_length(T), d_model) and
    out_lengths = subsampled_length(lengths).
    """

    def __init__(self, n_mels: int, d_model: int, channels: int = 32):
        super().__init__()
        raise NotImplementedError("Conv2dSubsampling.__init__")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Conv2dSubsampling.forward")


class FrameClassifier(nn.Module):
    """An MLP from one stacked frame to class logits.

    Linear(input_dim, hidden), ReLU, Linear(hidden, hidden), ReLU,
    Linear(hidden, n_classes). forward takes (T, input_dim) and returns
    unnormalized logits of shape (T, n_classes).
    """

    def __init__(self, input_dim: int, hidden: int, n_classes: int):
        super().__init__()
        raise NotImplementedError("FrameClassifier.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FrameClassifier.forward")


def frame_accuracy(logits, labels, ignore_index: int = -100) -> float:
    """The share of frames whose most likely class is the label.

    Args:
        logits: Shape (T, C), a tensor or array.
        labels: Shape (T,), integer class indices; frames labeled
            `ignore_index` are left out.

    Returns:
        A float between 0 and 1.
    """
    raise NotImplementedError("frame_accuracy")


def log_priors(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Class log priors from training labels.

    Args:
        labels: Integer labels of every training frame, shape (n,).
        n_classes: The number of classes.

    Returns:
        Shape (n_classes,): log of each class's share of the frames, -inf for
        a class that never occurs.
    """
    raise NotImplementedError("log_priors")


def scaled_log_likelihoods(log_posteriors: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Hybrid DNN-HMM emission scores from a classifier's posteriors.

    Args:
        log_posteriors: log P(s | x_t), shape (T, S).
        priors: log P(s), shape (S,).

    Returns:
        Shape (T, S): log P(s | x_t) - log P(s), which equals log p(x_t | s)
        minus log p(x_t), a constant for each frame.
    """
    raise NotImplementedError("scaled_log_likelihoods")


def train_frame_classifier(inputs, labels, n_classes, hidden=128, steps=500, batch_size=256, lr=1e-3, seed=0):
    """Train a FrameClassifier on frames with cross-entropy.

    Seed torch with `seed`, build FrameClassifier(input_dim, hidden, n_classes),
    and take `steps` Adam steps at learning rate `lr`, each on `batch_size`
    frames drawn at random with torch.randint.

    Args:
        inputs: Stacked, normalized frames, shape (n, input_dim), an array.
        labels: Integer labels, shape (n,).
        n_classes: The number of classes.

    Returns:
        The trained FrameClassifier.
    """
    raise NotImplementedError("train_frame_classifier")


# ---------------------------------------------------------------------------
# Provided: synthetic spoken "words" and their features. You don't need to
# change anything in this block.
#
# A word is a sequence of phone-like segments. A vowel is a harmonic tone at a
# random pitch, shaped by three formant peaks. A fricative is a burst of
# band-pass noise. Every segment gets a random duration, and the whole word
# gets silence at both ends and white noise at a chosen SNR.
# ---------------------------------------------------------------------------

RATE = 16000
PHONES = ["sil", "a", "e", "i", "o", "u", "s", "sh", "f"]
FORMANTS = {"a": (800, 1200, 2500), "e": (500, 1800, 2500), "i": (300, 2300, 3000),
            "o": (500, 900, 2400), "u": (320, 800, 2300)}
NOISE_BANDS = {"s": (4000, 7000, 0.5), "sh": (2000, 4000, 0.6), "f": (1000, 7500, 0.25)}
WORDS = {"sa": ["s", "a"], "shi": ["sh", "i"], "fu": ["f", "u"], "ai": ["a", "i"],
         "ia": ["i", "a"], "osa": ["o", "s", "a"], "ue": ["u", "e"], "eshu": ["e", "sh", "u"]}


def _vowel(name, n, f0, rng):
    t = np.arange(n) / RATE
    out = np.zeros(n)
    for k in range(1, int(5000 // f0) + 1):
        gain = sum(1.0 / (1.0 + ((k * f0 - f) / 90.0) ** 2) for f in FORMANTS[name])
        out += gain * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(np.mean(out**2))


def _fricative(name, n, rng):
    low, high, level = NOISE_BANDS[name]
    sos = butter(4, [low, high], btype="band", fs=RATE, output="sos")
    out = sosfilt(sos, rng.standard_normal(n))
    return level * out / np.sqrt(np.mean(out**2))


def synth_word(phones, rng, snr_db=20.0):
    """Synthesize one utterance of a word, given as a list of phone names.

    Returns (samples, labels): float64 samples at RATE with peak 0.25, and the
    index into PHONES of the segment each sample belongs to (0 is silence).
    """
    f0 = rng.uniform(100, 160)
    ramp_n = int(0.01 * RATE)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp_n) / ramp_n)
    lead = int(rng.uniform(0.04, 0.08) * RATE)
    pieces, labels = [np.zeros(lead)], [np.zeros(lead, dtype=int)]
    for p in phones:
        if p in FORMANTS:
            n = int(rng.uniform(0.06, 0.30) * RATE)
            seg = _vowel(p, n, f0 * rng.uniform(0.95, 1.05), rng)
        else:
            n = int(rng.uniform(0.05, 0.18) * RATE)
            seg = _fricative(p, n, rng)
        seg[:ramp_n] *= ramp
        seg[-ramp_n:] *= ramp[::-1]
        pieces.append(seg)
        labels.append(np.full(n, PHONES.index(p)))
    tail = int(rng.uniform(0.04, 0.08) * RATE)
    pieces.append(np.zeros(tail))
    labels.append(np.zeros(tail, dtype=int))
    x, lab = np.concatenate(pieces), np.concatenate(labels)
    speech_rms = np.sqrt(np.mean(x[lab > 0] ** 2))
    x = x + rng.standard_normal(x.size) * speech_rms * 10 ** (-snr_db / 20)
    return 0.25 * x / np.max(np.abs(x)), lab


def log_mel(x, n_mels=24, n_fft=512, win=400, hop=160):
    """Log mel energies, shape (n_frames, n_mels): 25 ms frames every 10 ms."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(0, mel(RATE / 2), n_mels + 2) / 2595) - 1)
    freqs = np.arange(n_fft // 2 + 1) * RATE / n_fft
    low, mid, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.maximum(0, np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)))
    n_frames = 1 + (len(x) - win) // hop
    index = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    power = np.abs(np.fft.rfft(x[index] * window, n=n_fft, axis=1)) ** 2
    return np.log(power @ bank.T + 1e-8)


def mfcc(x, n_ceps=13):
    """Thirteen MFCCs per frame with the utterance mean removed, shape (n_frames, 13)."""
    c = dct(log_mel(x), type=2, norm="ortho", axis=1)[:, :n_ceps]
    return c - c.mean(axis=0)


def frame_labels(labels, win=400, hop=160):
    """Provided. The phone at the center sample of every log_mel frame."""
    n_frames = 1 + (len(labels) - win) // hop
    return labels[hop * np.arange(n_frames) + win // 2]


def make_dataset(rng, n, snr_db):
    """Provided. n synthetic words cycling through WORDS: (log-mel list, frame-label list)."""
    names = list(WORDS)
    feats, labels = [], []
    for k in range(n):
        x, lab = synth_word(WORDS[names[k % len(names)]], rng, snr_db=snr_db)
        feats.append(log_mel(x))
        labels.append(frame_labels(lab))
    return feats, labels


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    train_feats, train_labels = make_dataset(rng, 64, snr_db=-5.0)
    test_feats, test_labels = make_dataset(rng, 24, snr_db=-5.0)
    everything = np.concatenate(train_feats)
    mean, std = everything.mean(axis=0), everything.std(axis=0)
    norm = lambda f: (f - mean) / std
    y_train, y_test = np.concatenate(train_labels), np.concatenate(test_labels)
    print(f"{len(y_train)} training frames and {len(y_test)} test frames at -5 dB SNR, {len(PHONES)} classes")

    contexts, accuracies, models = [0, 1, 2, 4, 6, 8], [], {}
    for c in contexts:
        x_train = np.concatenate([stack_frames(norm(f), c, c) for f in train_feats])
        x_test = np.concatenate([stack_frames(norm(f), c, c) for f in test_feats])
        model = train_frame_classifier(x_train, y_train, len(PHONES))
        with torch.no_grad():
            accuracies.append(frame_accuracy(model(torch.as_tensor(x_test, dtype=torch.float32)), y_test))
        models[c] = model
        print(f"context +/-{c} frames ({2 * c + 1} in all): test frame accuracy {accuracies[-1]:.1%}")
    show_plot((np.array(contexts), np.array(accuracies)), label="Frame accuracy against context width",
              xlabel="Frames of context on each side", ylabel="Test frame accuracy")

    x, lab = synth_word(WORDS["eshu"], rng, snr_db=-5.0)
    play(x, RATE, label='"eshu" at -5 dB SNR')
    feats = log_mel(x)
    show_spectrogram(feats.T, RATE, 160, label="Its log-mel features", db=False, y_axis="mel",
                     value_label="log energy")
    with torch.no_grad():
        logits = models[4](torch.as_tensor(stack_frames(norm(feats), 4, 4), dtype=torch.float32))
    show_image(torch.softmax(logits, dim=-1).T.numpy(), label="Frame posteriors with +/-4 frames of context",
               xlabel="Frame", ylabel="Phone", y_labels=PHONES, origin="upper")
    truth = frame_labels(lab)
    print(f'"eshu": frame accuracy {frame_accuracy(logits, truth):.1%} over {len(truth)} frames')

    front_end = Conv2dSubsampling(n_mels=24, d_model=144)
    out, out_len = front_end(torch.as_tensor(feats, dtype=torch.float32)[None], torch.tensor([len(feats)]))
    print(f"Conv2dSubsampling: {len(feats)} frames at 100 per second become {int(out_len[0])} "
          f"at 25 per second, output shape {tuple(out.shape)}")
