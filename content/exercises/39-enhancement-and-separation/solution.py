"""Enhancement and separation: the worked solution."""

import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import istft as scipy_istft
from scipy.signal import lfilter
from scipy.signal import stft as scipy_stft

from speechlab import play, show_plot, show_spectrogram


# Scoring and mixing -------------------------------------------------------------


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant signal-to-distortion ratio in dB, over the last axis.

    est, ref: tensors of the same shape (..., T). Removes the mean of each,
    projects est onto ref to get the target part, and returns
    10 log10(|target|^2 / |est - target|^2) with shape (...,).
    Differentiable, so -si_sdr works as a loss.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps)
    target = scale * ref
    noise = est - target
    return 10 * torch.log10(((target**2).sum(-1) + eps) / ((noise**2).sum(-1) + eps))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale noise so that clean + noise has the requested SNR in dB.

    Power is the mean square over the whole clip. Returns (mixture, scaled noise).
    """
    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2)
    gain = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    scaled = gain * noise
    return clean + scaled, scaled


# Classical enhancement ------------------------------------------------------------


def spectral_subtraction(Y: np.ndarray, noise_power: np.ndarray, alpha: float = 2.0, beta: float = 0.01) -> np.ndarray:
    """Power spectral subtraction with over-subtraction and a spectral floor.

    Y: complex STFT of the noisy signal, (F, T). noise_power: (F,), the
    estimated noise power in each bin. The estimated clean power is
    max(|Y|^2 - alpha * noise_power, beta * |Y|^2); the phase is the noisy
    phase. Returns the complex estimate, (F, T).
    """
    power = np.abs(Y) ** 2
    clean_power = np.maximum(power - alpha * noise_power[:, None], beta * power)
    return np.sqrt(clean_power) * np.exp(1j * np.angle(Y))


def wiener_gain(xi: np.ndarray) -> np.ndarray:
    """The Wiener gain xi / (1 + xi) for an a priori SNR xi (a power ratio, not dB)."""
    return xi / (1.0 + xi)


def decision_directed_gains(Y: np.ndarray, noise_power: np.ndarray, a: float = 0.98, xi_min: float = 10 ** (-25 / 10)) -> np.ndarray:
    """Wiener gains with the decision-directed a priori SNR estimate.

    Y: complex noisy STFT, (F, T). noise_power: (F,). Frame by frame:
        gamma = |Y[:, t]|^2 / noise_power
        xi = a * |S_prev|^2 / noise_power + (1 - a) * max(gamma - 1, 0)
        xi = max(xi, xi_min)
        G[:, t] = wiener_gain(xi), and S_prev = G[:, t] * Y[:, t]
    with S_prev = 0 before the first frame. Returns the gains, (F, T).
    """
    n_bins, n_frames = Y.shape
    gains = np.empty((n_bins, n_frames))
    prev = np.zeros(n_bins)
    for t in range(n_frames):
        power = np.abs(Y[:, t]) ** 2
        gamma = power / noise_power
        xi = a * prev / noise_power + (1 - a) * np.maximum(gamma - 1.0, 0.0)
        xi = np.maximum(xi, xi_min)
        gains[:, t] = wiener_gain(xi)
        prev = gains[:, t] ** 2 * power
    return gains


def ideal_masks(S: np.ndarray, N: np.ndarray, lc_db: float = 0.0, beta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The ideal binary mask and the ideal ratio mask, both (F, T).

    S, N: complex STFTs of the clean speech and the noise. The IBM is 1 where
    10 log10(|S|^2 / |N|^2) > lc_db and 0 elsewhere. The IRM is
    (|S|^2 / (|S|^2 + |N|^2)) ** beta. Both use powers, never dB values.
    """
    ps, pn = np.abs(S) ** 2, np.abs(N) ** 2
    tiny = 1e-20
    ibm = (10 * np.log10((ps + tiny) / (pn + tiny)) > lc_db).astype(float)
    irm = (ps / (ps + pn + tiny)) ** beta
    return ibm, irm


# Separation ----------------------------------------------------------------------


def pit_loss(est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Utterance-level permutation-invariant negative SI-SDR.

    est, ref: (B, C, T). For each example, tries every assignment of the C
    outputs to the C references and keeps the one with the lowest mean
    -SI-SDR. Returns (the mean of those minima over the batch, a scalar,
    and perm of shape (B, C), where output c is matched to reference perm[b, c]).
    """
    batch, n_src, _ = est.shape
    pair = -si_sdr(est[:, :, None, :], ref[:, None, :, :])  # (B, C, C): output i against reference j
    perms = list(itertools.permutations(range(n_src)))
    index = torch.tensor(perms, device=est.device)  # (P, C)
    cost = pair[:, torch.arange(n_src), index].mean(-1)  # (B, P)
    best = cost.argmin(dim=1)
    return cost[torch.arange(batch), best].mean(), index[best]


def tcn_receptive_field(repeats: int, blocks: int, kernel: int) -> int:
    """Receptive field, in encoder frames, of `repeats` stacks of `blocks` dilated
    convolutions with dilations 1, 2, 4, ..., 2**(blocks - 1) and the given kernel size."""
    return 1 + repeats * (kernel - 1) * (2**blocks - 1)


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each frame, for (B, C, T) tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TCNBlock(nn.Module):
    """One Conv-TasNet block with a residual connection, for (B, channels, T).

    1x1 conv to `hidden`, PReLU, ChannelNorm, a depthwise conv with the given
    kernel and dilation and "same" padding, PReLU, ChannelNorm, and a 1x1 conv
    back to `channels`, added to the input. Output shape equals input shape.
    """

    def __init__(self, channels: int, hidden: int, kernel: int, dilation: int):
        """Build the layers as self.net. The depthwise conv uses groups=hidden and
        padding dilation * (kernel - 1) // 2, which keeps the length for an odd kernel."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            ChannelNorm(hidden),
            nn.Conv1d(hidden, hidden, kernel, dilation=dilation, padding=dilation * (kernel - 1) // 2, groups=hidden),
            nn.PReLU(),
            ChannelNorm(hidden),
            nn.Conv1d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, channels, T) to (B, channels, T): the input plus the block's output."""
        return x + self.net(x)


class TinyConvTasNet(nn.Module):
    """A small Conv-TasNet: learned encoder, TCN mask estimator, learned decoder.

    Input (B, T) waveforms; output (B, n_src, T) separated waveforms, the same
    length as the input for any T.
    """

    def __init__(self, n_src: int = 2, n_filters: int = 64, length: int = 16, bottleneck: int = 32,
                 hidden: int = 64, kernel: int = 3, blocks: int = 5, repeats: int = 2):
        super().__init__()
        self.n_src, self.n_filters, self.length = n_src, n_filters, length
        self.stride = length // 2
        self.encoder = nn.Conv1d(1, n_filters, length, stride=self.stride, bias=False)
        self.separator = nn.Sequential(
            ChannelNorm(n_filters),
            nn.Conv1d(n_filters, bottleneck, 1),
            *[TCNBlock(bottleneck, hidden, kernel, 2**i) for _ in range(repeats) for i in range(blocks)],
            nn.PReLU(),
            nn.Conv1d(bottleneck, n_src * n_filters, 1),
        )
        self.decoder = nn.ConvTranspose1d(n_filters, 1, length, stride=self.stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) mixtures to (B, n_src, T) estimates.

        Right-pad x with zeros so the encoder's frames cover every sample,
        encode and apply ReLU to get w of shape (B, N, frames), run the
        separator and a sigmoid to get n_src masks of shape (N, frames) each,
        decode w * mask for every source, and trim the result to T samples.
        """
        batch, n = x.shape
        # Pad so the frames cover the whole input: n + pad = length + stride * (frames - 1).
        frames = max(1, -(-(n - self.length) // self.stride) + 1)
        pad = self.length + self.stride * (frames - 1) - n
        w = F.relu(self.encoder(F.pad(x, (0, pad))[:, None]))  # (B, N, frames)
        masks = torch.sigmoid(self.separator(w)).view(batch, self.n_src, self.n_filters, -1)
        masked = (w[:, None] * masks).reshape(batch * self.n_src, self.n_filters, -1)
        out = self.decoder(masked).view(batch, self.n_src, -1)
        return out[..., :n]


# Run helpers -----------------------------------------------------------------------

VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def vowel_train(rng, rate, seconds, f0, gap=0.0):
    """Source-filter vowels with a gliding pitch, 150-250 ms each, with optional pauses."""
    out = []
    while sum(len(o) for o in out) < seconds * rate:
        n = int(rng.uniform(0.15, 0.25) * rate)
        pitch = f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 3) * np.arange(n) / rate))
        phase = np.cumsum(pitch / rate)
        src = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
        y = src
        for f, bw in zip(VOWELS[rng.integers(len(VOWELS))], (80, 100, 140)):
            r = np.exp(-np.pi * bw / rate)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
        out.append(y * np.hanning(n) ** 0.3)
        if gap:
            out.append(np.zeros(int(gap * rate)))
    x = np.concatenate(out)[: int(seconds * rate)]
    return 0.5 * x / np.max(np.abs(x))


def stft(x, n_fft=512, hop=128):
    return scipy_stft(x, nperseg=n_fft, noverlap=n_fft - hop, window="hann")[2]


def istft(X, length, n_fft=512, hop=128):
    return scipy_istft(X, nperseg=n_fft, noverlap=n_fft - hop, window="hann")[1][:length]


def score(est, ref):
    return float(si_sdr(torch.as_tensor(est, dtype=torch.float64), torch.as_tensor(ref, dtype=torch.float64)))


def enhancement_demo(rng):
    rate = 16000
    clean = np.concatenate([np.zeros(rate // 4), vowel_train(rng, rate, 2.25, 120.0, gap=0.08)])
    noisy, noise = mix_at_snr(clean, rng.standard_normal(clean.size), 5.0)
    Y, S, N = stft(noisy), stft(clean), stft(noise)
    noise_power = np.mean(np.abs(Y[:, : (rate // 4) // 128]) ** 2, axis=1)  # the first 0.25 s is noise only
    subtracted = istft(spectral_subtraction(Y, noise_power), clean.size)
    wiener = istft(decision_directed_gains(Y, noise_power) * Y, clean.size)
    ibm, irm = ideal_masks(S, N)
    oracle = istft(irm * Y, clean.size)
    print("Enhancement at 5 dB SNR, SI-SDR against the clean signal:")
    for name, x in [("noisy input", noisy), ("spectral subtraction", subtracted),
                    ("decision-directed Wiener", wiener), ("oracle IBM", istft(ibm * Y, clean.size)),
                    ("oracle IRM", oracle)]:
        print(f"  {name:26} {score(x, clean):6.2f} dB")
    peak = np.max(np.abs(noisy))
    play(noisy / peak, rate, label="Noisy vowels at 5 dB SNR")
    play(wiener / peak, rate, label="After the decision-directed Wiener filter")
    play(subtracted / peak, rate, label="After spectral subtraction: listen for musical noise")
    show_spectrogram(np.abs(stft(wiener)), rate, 128, label="The Wiener output's spectrogram")


def chirp_source(rng, rate, n):
    t = np.arange(n) / rate
    f0, f1 = rng.uniform(300, 1500), rng.uniform(300, 3000)
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t[-1])) + rng.uniform(0, 2 * np.pi))
    return 0.5 * x * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))


def two_source_batch(rng, batch, rate=8000, seconds=0.5):
    n = int(rate * seconds)
    refs = np.stack([np.stack([vowel_train(rng, rate, seconds, rng.uniform(90, 200)), chirp_source(rng, rate, n)])
                     for _ in range(batch)])
    refs *= 10 ** (rng.uniform(-2.5, 2.5, size=(batch, 2, 1)) / 20)
    refs = torch.tensor(refs, dtype=torch.float32)
    return refs.sum(1), refs


def separation_demo(rng, steps=300):
    torch.set_num_threads(1)  # Convolutions this small run faster on one thread than on many.
    torch.manual_seed(0)
    model = TinyConvTasNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    test_mix, test_ref = two_source_batch(np.random.default_rng(99), 16)
    base = si_sdr(test_mix[:, None].expand_as(test_ref), test_ref).mean().item() + 0.0
    losses, started = [], time.time()
    for _ in range(steps):
        mix, ref = two_source_batch(rng, 8)
        loss, _ = pit_loss(model(mix), ref)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        est = model(test_mix)
        loss, perm = pit_loss(est, test_ref)
    print(f"\nSeparation: trained a {sum(p.numel() for p in model.parameters()):,}-parameter Conv-TasNet for "
          f"{len(losses)} steps ({time.time() - started:.0f} s).")
    print(f"  held-out SI-SDR: mixture {abs(base) if abs(base) < 0.05 else base:.1f} dB, separated {-loss.item():.1f} dB, "
          f"improvement {-loss.item() - base:.1f} dB")
    show_plot(-np.array(losses), label="Training SI-SDR (PIT)", xlabel="Step", ylabel="SI-SDR (dB)")
    play(test_mix[0].numpy() / test_mix[0].abs().max().item(), 8000, label="A held-out mixture: vowels plus a chirp")
    for c in range(2):
        e = est[0, c].numpy()
        play(0.9 * e / np.abs(e).max(), 8000, label=f"Output {c}, matched to reference {int(perm[0, c])}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    enhancement_demo(rng)
    separation_demo(rng)
