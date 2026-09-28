"""Connectionist temporal classification in numpy: the worked solution."""

import numpy as np
import torch
from scipy.special import log_softmax, logsumexp

from speechlab import play, show_image, show_plot, show_spectrogram


def extend_with_blanks(target, blank: int = 0) -> np.ndarray:
    """The target with a blank before, between, and after its labels: length 2L + 1."""
    target = np.asarray(target, dtype=int)
    ext = np.full(2 * len(target) + 1, blank, dtype=int)
    ext[1::2] = target
    return ext


def collapse(path, blank: int = 0) -> list[int]:
    """The CTC collapse: merge runs of the same symbol, then drop blanks."""
    out, previous = [], None
    for symbol in (int(s) for s in path):
        if symbol != previous and symbol != blank:
            out.append(symbol)
        previous = symbol
    return out


def _allow_skip(ext: np.ndarray, blank: int) -> np.ndarray:
    """allow[s]: may a path jump from state s - 2 straight to state s?"""
    allow = np.zeros(len(ext), dtype=bool)
    allow[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    return allow


def ctc_forward(log_probs: np.ndarray, target, blank: int = 0) -> np.ndarray:
    """log alpha, shape (T, 2L + 1): log probability of every path prefix ending in state s at frame t."""
    ext = extend_with_blanks(target, blank)
    T, S = len(log_probs), len(ext)
    skip = _allow_skip(ext, blank)
    emit = log_probs[:, ext]  # (T, S): log y^t of the symbol at each state
    log_alpha = np.full((T, S), -np.inf)
    log_alpha[0, 0] = emit[0, 0]
    if S > 1:
        log_alpha[0, 1] = emit[0, 1]
    for t in range(1, T):
        prev = log_alpha[t - 1]
        total = prev.copy()
        total[1:] = np.logaddexp(total[1:], prev[:-1])
        total[2:] = np.where(skip[2:], np.logaddexp(total[2:], prev[:-2]), total[2:])
        log_alpha[t] = total + emit[t]
    return log_alpha


def ctc_backward(log_probs: np.ndarray, target, blank: int = 0) -> np.ndarray:
    """log beta, shape (T, 2L + 1): log probability of frames t + 1 .. T - 1 given state s at frame t."""
    ext = extend_with_blanks(target, blank)
    T, S = len(log_probs), len(ext)
    skip = _allow_skip(ext, blank)
    emit = log_probs[:, ext]
    log_beta = np.full((T, S), -np.inf)
    log_beta[-1, -1] = 0.0
    if S > 1:
        log_beta[-1, -2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = log_beta[t + 1] + emit[t + 1]
        total = nxt.copy()
        total[:-1] = np.logaddexp(total[:-1], nxt[1:])
        total[:-2] = np.where(skip[2:], np.logaddexp(total[:-2], nxt[2:]), total[:-2])
        log_beta[t] = total
    return log_beta


def ctc_loss(log_probs: np.ndarray, target, blank: int = 0) -> float:
    """-log p(target | x), or inf when no path fits in T frames."""
    log_alpha = ctc_forward(log_probs, target, blank)
    return float(-logsumexp(log_alpha[-1, -2:]))


def ctc_grad(logits: np.ndarray, target, blank: int = 0) -> tuple[float, np.ndarray]:
    """The CTC loss of raw logits (T, C), and its gradient with respect to them."""
    log_probs = log_softmax(logits, axis=1)
    ext = extend_with_blanks(target, blank)
    log_alpha = ctc_forward(log_probs, target, blank)
    log_beta = ctc_backward(log_probs, target, blank)
    log_p = logsumexp(log_alpha[-1, -2:])
    if not np.isfinite(log_p):
        return float("inf"), np.zeros_like(logits)
    gamma = np.exp(log_alpha + log_beta - log_p)  # (T, S) state occupancy
    occupancy = np.zeros_like(logits)
    np.add.at(occupancy, (slice(None), ext), gamma)
    return float(-log_p), np.exp(log_probs) - occupancy


def ctc_loss_batch(log_probs, targets, input_lengths, target_lengths, blank: int = 0, reduction: str = "mean"):
    """CTC over a padded batch in torch's layout, with torch's three reductions."""
    losses = np.array([
        ctc_loss(log_probs[: int(input_lengths[n]), n], targets[n, : int(target_lengths[n])], blank)
        for n in range(log_probs.shape[1])
    ])
    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(losses.sum())
    return float(np.mean(losses / np.maximum(np.asarray(target_lengths), 1)))


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
