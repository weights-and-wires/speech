"""Checks for the neural acoustic model exercise."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dct
from scipy.signal import butter, sosfilt
from torch import nn

from speechcheck import almost, check, close, equal, fail, has, shape, timed


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
    n_frames = 1 + (len(labels) - win) // hop
    return labels[hop * np.arange(n_frames) + win // 2]


def make_dataset(rng, n, snr_db):
    names = list(WORDS)
    feats, labels = [], []
    for k in range(n):
        x, lab = synth_word(WORDS[names[k % len(names)]], rng, snr_db=snr_db)
        feats.append(log_mel(x))
        labels.append(frame_labels(lab))
    return feats, labels


@check("stack_frames puts each frame beside its neighbors, repeating the edge frames")
def test_stack(sub):
    stack_frames = has(sub, "stack_frames")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(7, 3))
    got = stack_frames(feats, 2, 1)
    shape(got, (7, 12), label="stack_frames(features of shape (7, 3), 2, 1)")
    want = np.zeros((7, 12))
    for t in range(7):
        want[t] = np.concatenate([feats[min(max(t + k, 0), 6)] for k in (-2, -1, 0, 1)])
    close(got, want, atol=0, label="stacked frames")
    close(stack_frames(feats, 0, 0), feats, atol=0, label="no context")


@check("conv_output_length matches torch's Conv1d for kernels, strides, padding, and dilation")
def test_conv_length(sub):
    conv_output_length = has(sub, "conv_output_length")
    rng = np.random.default_rng(1)
    for _ in range(60):
        kernel, stride = int(rng.integers(1, 8)), int(rng.integers(1, 5))
        padding, dilation = int(rng.integers(0, 4)), int(rng.integers(1, 4))
        length = int(rng.integers(dilation * (kernel - 1) + 1, 200))
        conv = nn.Conv1d(1, 1, kernel, stride=stride, padding=padding, dilation=dilation)
        want = conv(torch.zeros(1, 1, length)).shape[-1]
        got = conv_output_length(length, kernel, stride, padding, dilation)
        if got != want:
            fail(f"length {length}, kernel {kernel}, stride {stride}, padding {padding}, dilation "
                 f"{dilation}: expected {want} output frames, got {got}")


@check("subsampled_length matches two stacked stride-2 Conv2d layers, for ints and tensors")
def test_subsampled_length(sub):
    subsampled_length = has(sub, "subsampled_length")
    stack = nn.Sequential(nn.Conv2d(1, 1, 3, stride=2), nn.Conv2d(1, 1, 3, stride=2))
    for length in (7, 8, 9, 50, 98, 100, 101, 1000):
        want = stack(torch.zeros(1, 1, length, 80)).shape[2]
        got = subsampled_length(length)
        if int(got) != want:
            fail(f"{length} input frames give {want} output frames, got {got}")
    lengths = torch.tensor([100, 73, 9])
    got = subsampled_length(lengths)
    if not torch.is_tensor(got) or got.tolist() != [24, 17, 1]:
        fail(f"subsampled_length(torch.tensor([100, 73, 9])) should be tensor([24, 17, 1]), got {got!r}")


@check("Conv2dSubsampling returns (N, T', d_model) and the subsampled lengths")
def test_subsampling_shapes(sub):
    Conv2dSubsampling = has(sub, "Conv2dSubsampling")
    torch.manual_seed(0)
    module = Conv2dSubsampling(24, 64, channels=8)
    y, lengths = module(torch.randn(3, 100, 24), torch.tensor([100, 73, 50]))
    shape(y.detach(), (3, 24, 64), label="output for input (3, 100, 24)")
    equal(torch.as_tensor(lengths).tolist(), [24, 17, 11], label="output lengths")


@check("Conv2dSubsampling applies conv1, ReLU, conv2, ReLU, then flattens channels and frequency")
def test_subsampling_values(sub):
    Conv2dSubsampling = has(sub, "Conv2dSubsampling")
    torch.manual_seed(1)
    module = Conv2dSubsampling(20, 16, channels=4)
    for name in ("conv1", "conv2", "out"):
        if not hasattr(module, name):
            fail(f"Conv2dSubsampling needs a submodule named `{name}`; see the docstring")
    x = torch.randn(2, 30, 20)
    with torch.no_grad():
        got, _ = module(x, torch.tensor([30, 30]))
        h = F.relu(F.conv2d(x[:, None], module.conv1.weight, module.conv1.bias, stride=2))
        h = F.relu(F.conv2d(h, module.conv2.weight, module.conv2.bias, stride=2))
        n, c, t, f = h.shape
        want = F.linear(h.permute(0, 2, 1, 3).reshape(n, t, c * f), module.out.weight, module.out.bias)
    close(got.numpy(), want.numpy(), atol=1e-5, label="output against the same weights applied by hand")


@check("FrameClassifier is a two-hidden-layer MLP that maps (T, input_dim) to (T, n_classes)")
def test_classifier_shape(sub):
    FrameClassifier = has(sub, "FrameClassifier")
    torch.manual_seed(2)
    model = FrameClassifier(117, 64, 9)
    shape(model(torch.randn(50, 117)).detach(), (50, 9), label="logits")
    count = sum(p.numel() for p in model.parameters())
    want = 117 * 64 + 64 + 64 * 64 + 64 + 64 * 9 + 9
    if count != want:
        fail(f"FrameClassifier(117, 64, 9) has {count} parameters; Linear(117, 64), Linear(64, 64), "
             f"and Linear(64, 9) have {want}")


@check("frame_accuracy counts correct frames and skips labels equal to ignore_index")
def test_accuracy(sub):
    frame_accuracy = has(sub, "frame_accuracy")
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0], [0.0, 5.0]])
    almost(frame_accuracy(logits, torch.tensor([0, 1, 1, 1])), 0.75, 1e-12, label="3 of 4 correct")
    almost(frame_accuracy(logits, torch.tensor([0, 1, -100, -100])), 1.0, 1e-12, label="padding ignored")


@check("scaled_log_likelihoods turn posteriors into likelihoods up to one constant per frame")
def test_hybrid(sub):
    log_priors, scaled = has(sub, "log_priors"), has(sub, "scaled_log_likelihoods")
    close(np.exp(log_priors(np.array([0, 0, 1, 2, 2, 2]), 4)), [2 / 6, 1 / 6, 3 / 6, 0.0], atol=1e-12,
          label="priors from label counts")
    rng = np.random.default_rng(3)
    joint = rng.dirichlet(np.ones(15)).reshape(5, 3)  # P(x, s) for 5 observations and 3 states
    prior = joint.sum(axis=0)
    posterior = joint / joint.sum(axis=1, keepdims=True)
    likelihood = joint / prior[None, :]
    got = scaled(np.log(posterior), np.log(prior))
    shape(got, (5, 3), label="scaled log-likelihoods")
    offset = got - np.log(likelihood)
    close(offset - offset[:, :1], np.zeros((5, 3)), atol=1e-10,
          label="log P(s|x) - log P(s) - log p(x|s), which must not depend on s")


@check("train_frame_classifier reaches 83% frame accuracy at -5 dB, and context beats a single frame")
def test_training(sub):
    stack_frames, train, accuracy = has(sub, "stack_frames"), has(sub, "train_frame_classifier"), has(sub, "frame_accuracy")
    rng = np.random.default_rng(1)
    train_feats, train_labels = make_dataset(rng, 64, -5.0)
    test_feats, test_labels = make_dataset(rng, 24, -5.0)
    everything = np.concatenate(train_feats)
    mean, std = everything.mean(axis=0), everything.std(axis=0)
    y_train, y_test = np.concatenate(train_labels), np.concatenate(test_labels)
    results = {}
    for context in (0, 4):
        x_train = np.concatenate([stack_frames((f - mean) / std, context, context) for f in train_feats])
        x_test = np.concatenate([stack_frames((f - mean) / std, context, context) for f in test_feats])
        with timed(f"training with context {context}"):
            model = train(x_train, y_train, len(PHONES))
        with torch.no_grad():
            results[context] = accuracy(model(torch.as_tensor(x_test, dtype=torch.float32)), y_test)
    if results[4] < 0.83:
        fail(f"with 4 frames of context on each side, test frame accuracy is {results[4]:.1%}; expected 83% or more")
    if results[4] < results[0] + 0.03:
        fail(f"context should help at -5 dB: accuracy is {results[0]:.1%} with one frame and "
             f"{results[4]:.1%} with nine, but nine should win by 3 points or more")
