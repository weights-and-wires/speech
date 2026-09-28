"""Connectionist temporal classification in numpy.

Fill in the seven functions below, then press Run to check the worked T = 3
example against torch and train a small model on synthetic tone sequences
with your loss and gradient as its only training signal. Press Check to test
your code.

Conventions, the same as torch.nn.functional.ctc_loss: blank is index 0, and
log_probs are log-softmax outputs, shape (T, C) for one utterance or
(T, N, C) for a batch. A target is a sequence of label indices, none of them
blank. State s of the extended sequence is 0-based.
"""

import numpy as np
import torch
from scipy.special import log_softmax, logsumexp

from speechlab import play, show_image, show_plot, show_spectrogram


def extend_with_blanks(target, blank: int = 0) -> np.ndarray:
    """The blank-extended label sequence.

    Args:
        target: L label indices.
        blank: The blank index.

    Returns:
        An int array of length 2L + 1: blank, target[0], blank, target[1], ...,
        target[L - 1], blank.
    """
    raise NotImplementedError("extend_with_blanks")


def collapse(path, blank: int = 0) -> list[int]:
    """The CTC collapse function B.

    Args:
        path: One symbol per frame, any sequence of ints.
        blank: The blank index.

    Returns:
        A list of ints: first merge each run of identical consecutive symbols
        into one, then remove every blank.
    """
    raise NotImplementedError("collapse")


def ctc_forward(log_probs: np.ndarray, target, blank: int = 0) -> np.ndarray:
    """The CTC forward variables in log space.

    Args:
        log_probs: Log-softmax outputs for one utterance, shape (T, C).
        target: L label indices.
        blank: The blank index.

    Returns:
        log_alpha, shape (T, 2L + 1). log_alpha[t, s] is the log of the total
        probability of every path prefix over frames 0 .. t that ends in
        state s of the extended sequence, including frame t's emission.
        Unreachable entries are -inf.
    """
    raise NotImplementedError("ctc_forward")


def ctc_backward(log_probs: np.ndarray, target, blank: int = 0) -> np.ndarray:
    """The CTC backward variables in log space.

    Args:
        log_probs: Shape (T, C).
        target: L label indices.
        blank: The blank index.

    Returns:
        log_beta, shape (T, 2L + 1). log_beta[t, s] is the log probability of
        emitting frames t + 1 .. T - 1 and finishing validly, given state s at
        frame t. It excludes frame t's own emission, so the last row is 0 on
        the last two states and -inf elsewhere.
    """
    raise NotImplementedError("ctc_backward")


def ctc_loss(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """The CTC loss of one utterance.

    Args:
        log_probs: Shape (T, C).
        target: L label indices.
        blank: The blank index.

    Returns:
        -log p(target | x) as a float: minus the log-sum-exp of the last row
        of log_alpha over the final label and the final blank. float("inf")
        when no path fits in T frames.
    """
    raise NotImplementedError("ctc_loss")


def ctc_grad(logits: np.ndarray, target, blank: int = 0) -> tuple[float, np.ndarray]:
    """The CTC loss of raw logits and its gradient with respect to them.

    Args:
        logits: Unnormalized scores for one utterance, shape (T, C). The
            model's probabilities are softmax(logits) along axis 1.
        target: L label indices.
        blank: The blank index.

    Returns:
        (loss, grad): loss as ctc_loss returns it, and grad of shape (T, C),
        d loss / d logits. For an impossible target return
        (float("inf"), zeros of shape (T, C)).
    """
    raise NotImplementedError("ctc_grad")


def ctc_loss_batch(log_probs, targets, input_lengths, target_lengths, blank: int = 0, reduction: str = "mean"):
    """CTC over a padded batch, in torch's layout and with torch's reductions.

    Args:
        log_probs: Shape (T, N, C).
        targets: Padded labels, shape (N, S_max); row n's labels are
            targets[n, :target_lengths[n]].
        input_lengths: Shape (N,): utterance n uses frames 0 .. input_lengths[n] - 1.
        target_lengths: Shape (N,).
        blank: The blank index.
        reduction: "none" returns the N losses as an array, "sum" their sum,
            and "mean" the mean of each loss divided by its target length
            (a length of 0 counts as 1), all as torch does.

    Returns:
        An array of shape (N,) for "none", otherwise a float.
    """
    raise NotImplementedError("ctc_loss_batch")


# ---------------------------------------------------------------------------
# Provided: synthetic "tone words" and a tiny model trained with your CTC. You
# don't need to change anything below.
# ---------------------------------------------------------------------------

RATE = 16000
TONES = {1: 400.0, 2: 700.0, 3: 1100.0, 4: 1700.0}  # label -> frequency in Hz


def tone_word(labels, rng, snr_db=10.0):
    """A sequence of tones, one per label, with random durations and short gaps."""
    pieces = [np.zeros(int(rng.uniform(0.03, 0.08) * RATE))]
    for label in labels:
        n = int(rng.uniform(0.06, 0.16) * RATE)
        t = np.arange(n) / RATE
        f = TONES[label] * rng.uniform(0.97, 1.03)
        seg = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
        seg *= np.hanning(n) ** 0.3
        pieces += [seg, np.zeros(int(rng.uniform(0.02, 0.05) * RATE))]
    pieces.append(np.zeros(int(rng.uniform(0.03, 0.08) * RATE)))
    x = np.concatenate(pieces)
    x = x + rng.standard_normal(x.size) * np.sqrt(np.mean(x**2)) * 10 ** (-snr_db / 20)
    return 0.3 * x / np.max(np.abs(x))


def log_spectrum(x, n_fft=512, hop=160, n_bins=64):
    """Log power of the lowest n_bins FFT bins of 25 ms frames every 10 ms, shape (T, n_bins)."""
    n_frames = 1 + (len(x) - 400) // hop
    index = np.arange(400)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[index] * np.hanning(400)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)[:, :n_bins]) ** 2
    feats = np.log(power + 1e-6)
    return (feats - feats.mean()) / feats.std()


class _NumpyCTC(torch.autograd.Function):
    """Wraps ctc_grad so torch can train through it: logits (T, N, C) in, summed loss out."""

    @staticmethod
    def forward(ctx, logits, targets):
        array = logits.detach().double().numpy()
        losses, grads = zip(*(ctc_grad(array[:, n], t) for n, t in enumerate(targets)))
        ctx.save_for_backward(torch.as_tensor(np.stack(grads, axis=1), dtype=logits.dtype))
        return logits.new_tensor(sum(losses))

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output, None


class TinyCTCModel(torch.nn.Module):
    """Three 1-D convolutions over the log spectrum, one output per frame."""

    def __init__(self, n_in, n_classes, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_in, hidden, 5, padding=2), torch.nn.ReLU(),
            torch.nn.Conv1d(hidden, hidden, 5, padding=2), torch.nn.ReLU(),
            torch.nn.Conv1d(hidden, n_classes, 1),
        )

    def forward(self, x):  # x: (T, N, n_in) -> logits (T, N, n_classes)
        return self.net(x.permute(1, 2, 0)).permute(2, 0, 1)


def random_labels(rng):
    """Two to four labels; about one sequence in three repeats a label."""
    labels = list(rng.integers(1, 5, size=int(rng.integers(2, 5))))
    if rng.random() < 0.35:
        k = int(rng.integers(len(labels)))
        labels.insert(k, labels[k])
    return [int(v) for v in labels]


if __name__ == "__main__":
    # The worked example: T = 3, the label "a" = 1, with P(a) = 0.6, 0.3, 0.8.
    p_a = np.array([0.6, 0.3, 0.8])
    log_probs = np.log(np.stack([1 - p_a, p_a], axis=1))
    print("Alpha for the worked example:\n", np.round(np.exp(ctc_forward(log_probs, [1])), 4))
    print(f"Loss {ctc_loss(log_probs, [1]):.5f}; torch says "
          f"{torch.nn.functional.ctc_loss(torch.tensor(log_probs)[:, None], torch.tensor([[1]]), [3], [1], reduction='sum').item():.5f}")

    # Train a tiny model on tone words with your loss and gradient.
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = TinyCTCModel(64, 5)
    print(f"Training a {sum(p.numel() for p in model.parameters())}-parameter convolutional model "
          "for 500 steps, with your ctc_grad as its only loss")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for step in range(500):
        batch = [random_labels(rng) for _ in range(8)]
        feats = [log_spectrum(tone_word(labels, rng)) for labels in batch]
        T = max(len(f) for f in feats)
        x = torch.zeros(T, len(feats), 64)
        for n, f in enumerate(feats):
            x[: len(f), n] = torch.as_tensor(f, dtype=torch.float32)
            x[len(f):, n] = x[len(f) - 1, n]  # repeat the last frame as padding
        loss = _NumpyCTC.apply(model(x), batch) / len(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    show_plot(np.array(history), label="Your CTC loss while training", xlabel="Step", ylabel="Loss per utterance (nats)")
    print(f"Loss per utterance: {np.mean(history[:10]):.2f} over the first 10 steps, "
          f"{np.mean(history[-10:]):.2f} over the last 10")

    # Look at what it learned: peaky posteriors and a greedy decode.
    correct = 0
    for k in range(20):
        labels = random_labels(rng)
        x_audio = tone_word(labels, rng)
        feats = log_spectrum(x_audio)
        with torch.no_grad():
            logits = model(torch.as_tensor(feats, dtype=torch.float32)[:, None])[:, 0]
        guess = collapse(logits.argmax(dim=-1).tolist())
        correct += guess == labels
        if k == 0:
            play(x_audio, RATE, label=f"Tone word {labels}")
            show_spectrogram(feats.T, RATE, 160, db=False, label="Its log spectrum (lowest 64 bins)")
            show_image(torch.softmax(logits, dim=-1).T.numpy(), label="Per-frame posteriors: blank is row 0",
                       xlabel="Frame", ylabel="Symbol", y_labels=["blank", "1", "2", "3", "4"])
            print(f"Truth {labels}, greedy decode {guess}")
    print(f"Greedy decoding gets {correct} of 20 held-out tone words exactly right. "
          "Decoding CTC, the next lesson, searches for better answers than the per-frame argmax.")
