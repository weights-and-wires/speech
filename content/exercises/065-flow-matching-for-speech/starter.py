"""Flow matching for speech.

Convention (Lipman et al., Voicebox, F5-TTS): t = 0 is noise and t = 1 is
data. x0 is noise, x1 is data, and sigma_min = 0, so the target is x1 - x0.
Mel spectrograms are laid out as (batch, bands, frames), and a mask is
(batch, frames) with 1 marking a frame the model must generate.

Fill in the seven functions below, then press Run to train a small flow that
says the digits zero to nine, and Check to test your code. Everything after
"Finished helpers" is given: the log-mel front end from the mel-scale chapter, the
Griffin-Lim vocoder from its chapter, the data, and the network.
"""

import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from speechlab import asset_path, has_asset, play, read_audio, show_image, show_plot, show_spectrogram


def sway(u, s=-1.0):
    """F5-TTS sway sampling: f(u) = u + s (cos(pi u / 2) - 1 + u).

    u is a float, a numpy array, or a torch tensor of times in [0, 1]; return
    the same type. With s < 0 the warped times crowd toward t = 0, the noise
    end. s = 0 leaves u unchanged.
    """
    raise NotImplementedError


def infilling_batch(x0, x1, t, mask):
    """The inputs and target of the masked infilling loss (Voicebox, F5-TTS).

    x0 (noise) and x1 (data) are (B, F, N) tensors, t is (B,), and mask is
    (B, N) with 1 on the frames to generate. Returns (x_t, context, target):

        x_t      (1 - t) x0 + t x1
        context  x1 on the frames the mask leaves visible, 0 on masked frames
        target   x1 - x0
    """
    raise NotImplementedError


def masked_flow_loss(v_pred, target, mask):
    """The mean squared error over masked frames only, a scalar tensor.

    v_pred and target are (B, F, N) and mask is (B, N). Sum the squared error
    on masked frames and divide by (number of masked frames) * F, so the loss
    doesn't depend on how many frames were masked.
    """
    raise NotImplementedError


def cfg_velocity(v_cond, v_uncond, alpha):
    """Classifier-free guidance in the w-convention: v_c + alpha (v_c - v_u).

    alpha = 0 means no guidance. F5-TTS's default is alpha = 2.
    """
    raise NotImplementedError


def sample(velocity, x0, cond, null_cond, nfe, sway_s=-1.0, cfg_alpha=2.0):
    """Generate by Euler from t = 0 to t = 1 on a sway-warped time grid.

    `velocity(x, t, cond)` takes a (B', F, N) state, a (B',) tensor of times on x0's device,
    and a (B',) condition. The grid is sway(linspace(0, 1, nfe + 1), sway_s),
    and step k moves from grid[k] to grid[k + 1]. With cfg_alpha != 0, make
    one call per step on a batch of 2B rows, the states twice with `cond`
    then `null_cond`, and combine the halves with `cfg_velocity`. With
    cfg_alpha == 0, make one call on the B rows with `cond`. Returns the
    state at t = 1.
    """
    raise NotImplementedError


def chunk_causal_mask(n_frames, chunk_size):
    """CosyVoice 2's chunk-aware attention mask, a boolean (n_frames, n_frames) array.

    Entry [i, j] is True when query frame i may attend to key frame j: when
    frame j's chunk, j // chunk_size, is not later than frame i's chunk.
    """
    raise NotImplementedError


def mel_to_magnitude(mel_power, filterbank):
    """Invert a mel power spectrogram to a linear magnitude spectrogram.

    mel_power is (n_mels, frames) and filterbank is (n_mels, n_freq). Solve
    for the power with the filterbank's pseudo-inverse, clip negative values
    to 0, and return the square root, shape (n_freq, frames).
    """
    raise NotImplementedError


# Finished helpers: the front end, the vocoder, the data, and the model -----

RATE, N_FFT, HOP, N_MELS, SECONDS = 16000, 1024, 256, 80, 0.6
LENGTH = int(RATE * SECONDS)
NULL = 10  # the label that means "no condition"


def hz_to_mel(f):
    """The Slaney mel scale, librosa's default: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=float)
    log_part = 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27)
    return np.where(f >= 1000, log_part, f / (200 / 3))


def mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    return np.where(m >= 15, 1000 * np.exp((m - 15) * np.log(6.4) / 27), m * 200 / 3)


def mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=None):
    """Triangular filters with Slaney area normalization, shape (n_mels, n_fft // 2 + 1)."""
    fmax = sample_rate / 2 if fmax is None else fmax
    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs[None, :] - lo) / (mid - lo)
    falling = (hi - freqs[None, :]) / (hi - mid)
    return np.maximum(0, np.minimum(rising, falling)) * (2 / (hi - lo))


def hann(n):
    """The periodic Hann window of length n."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def stft(x, n_fft, hop_length):
    """The centered, reflect-padded, Hann-windowed STFT of the Griffin-Lim chapter, shape (n_fft // 2 + 1, frames)."""
    x = np.pad(np.asarray(x, dtype=float), n_fft // 2, mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop_length
    index = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[index] * hann(n_fft), axis=1).T


def istft(X, hop_length, length=None):
    """The least-squares inverse of `stft`: hop_length * (frames - 1) samples unless `length` is given."""
    n_fft = 2 * (X.shape[0] - 1)
    w = hann(n_fft)
    frames = np.fft.irfft(X.T, n=n_fft, axis=1) * w
    total = n_fft + hop_length * (frames.shape[0] - 1)
    signal, weight = np.zeros(total), np.zeros(total)
    for m, frame in enumerate(frames):
        signal[m * hop_length : m * hop_length + n_fft] += frame
        weight[m * hop_length : m * hop_length + n_fft] += w**2
    length = total - n_fft if length is None else length
    start = n_fft // 2
    return signal[start : start + length] / np.maximum(weight[start : start + length], 1e-11)


def griffin_lim(magnitude, n_fft, hop_length, n_iter=32, momentum=0.0, seed=0):
    """Fast Griffin-Lim, as built in the Griffin-Lim chapter: a signal whose STFT magnitude approximates `magnitude`."""
    rng = np.random.default_rng(seed)
    c = magnitude * np.exp(2j * np.pi * rng.random(magnitude.shape))
    previous = None
    for _ in range(n_iter):
        t = stft(istft(magnitude * np.exp(1j * np.angle(c)), hop_length), n_fft, hop_length)
        c = t if previous is None else t + momentum * (t - previous)
        previous = t
    return istft(magnitude * np.exp(1j * np.angle(c)), hop_length)


FILTERBANK = mel_filterbank(RATE, N_FFT, N_MELS)


def log_mel(x):
    """Natural-log mel power spectrogram with a 1e-5 floor, shape (80, frames)."""
    power = np.abs(stft(x, N_FFT, HOP)) ** 2
    return np.log(np.maximum(FILTERBANK @ power, 1e-5))


def fit_clip(x):
    """Cut the loudest 0.6 s of a clip, or pad a shorter one, and peak-normalize it."""
    if len(x) < LENGTH:
        extra = LENGTH - len(x)
        x = np.pad(x, (extra // 2, extra - extra // 2))
    else:
        energy = np.concatenate([[0.0], np.cumsum(x.astype(float) ** 2)])
        start = int(np.argmax(energy[LENGTH:] - energy[:-LENGTH]))
        x = x[start : start + LENGTH]
    return 0.5 * x / (np.abs(x).max() + 1e-8)


def load_digits(per_digit=None):
    """FSDD at 16 kHz as (clips, labels, is_test); indices 0 to 4 of each speaker are the test set."""
    folder = asset_path("fsdd") / "free-spoken-digit-dataset-1.0.10" / "recordings"
    clips, labels, test = [], [], []
    counts = {}
    for path in sorted(glob.glob(str(folder / "*.wav"))):
        digit, _, index = os.path.basename(path)[:-4].split("_")
        if per_digit and counts.get(digit, 0) >= per_digit:
            continue
        counts[digit] = counts.get(digit, 0) + 1
        audio, rate = read_audio(path)
        clips.append(fit_clip(resample_poly(audio, RATE // rate, 1)))
        labels.append(int(digit))
        test.append(int(index) < 5)
    return np.stack(clips).astype(np.float32), np.array(labels), np.array(test)


def synthetic_digits(n_per_digit=60, seed=0):
    """A stand-in when fsdd is missing: ten classes of gliding buzzy vowels."""
    rng = np.random.default_rng(seed)
    t = np.arange(LENGTH) / RATE
    clips, labels = [], []
    for digit in range(10):
        for _ in range(n_per_digit):
            f0 = (110 + 12 * digit) * (1 + 0.3 * (digit % 3 - 1) * t / SECONDS) * rng.uniform(0.95, 1.05)
            phase = 2 * np.pi * np.cumsum(f0) / RATE
            voice = sum(np.sin(k * phase) / k for k in range(1, 25))
            formant = 400 + 180 * digit
            shaped = np.real(np.fft.irfft(np.fft.rfft(voice) * np.exp(
                -((np.fft.rfftfreq(LENGTH, 1 / RATE) - formant) / 500) ** 2), n=LENGTH))
            envelope = np.sin(np.pi * np.clip((t - 0.1) / 0.4, 0, 1)) ** 2
            clips.append(fit_clip(shaped * envelope + 0.01 * rng.standard_normal(LENGTH)))
            labels.append(digit)
    labels = np.array(labels)
    test = np.tile(np.arange(n_per_digit) < n_per_digit // 10, 10)
    return np.stack(clips).astype(np.float32), labels, test


class FrameConv(torch.nn.Module):
    """A kernel-3 dilated convolution over frames, written as one matrix product."""

    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.linear = torch.nn.Linear(3 * cin, cout)

    def forward(self, x):  # (B, frames, channels)
        d, n = self.dilation, x.shape[1]
        p = F.pad(x, (0, 0, d, d))
        return self.linear(torch.cat([p[:, :n], p[:, d : d + n], p[:, 2 * d : 2 * d + n]], dim=-1))


class Block(torch.nn.Module):
    def __init__(self, width, dilation, emb):
        super().__init__()
        self.norm1, self.conv1 = torch.nn.LayerNorm(width), FrameConv(width, width, dilation)
        self.norm2, self.conv2 = torch.nn.LayerNorm(width), FrameConv(width, width)
        self.film = torch.nn.Linear(emb, 2 * width)

    def forward(self, x, emb):
        scale, shift = self.film(emb)[:, None, :].chunk(2, dim=-1)
        y = self.conv1(F.silu(self.norm1(x)))
        y = self.norm2(y) * (1 + scale) + shift
        return x + self.conv2(F.silu(y))


class MelFlow(torch.nn.Module):
    """A small dilated-convolution velocity network: v(x_t, t, digit, context)."""

    def __init__(self, width=96, n_blocks=8, emb=128):
        super().__init__()
        self.emb = emb
        self.inp = FrameConv(2 * N_MELS, width)
        self.label = torch.nn.Embedding(11, emb)
        self.time = torch.nn.Sequential(torch.nn.Linear(emb, emb), torch.nn.SiLU(), torch.nn.Linear(emb, emb))
        self.blocks = torch.nn.ModuleList(Block(width, 2 ** (i % 4), emb) for i in range(n_blocks))
        self.norm = torch.nn.LayerNorm(width)
        self.out = torch.nn.Linear(width, N_MELS)
        # A per-band gain on x_t, set by the time: the linear part of the field.
        self.skip = torch.nn.Linear(emb, N_MELS)
        for layer in (self.out, self.skip):
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x, t, label, context):
        half = self.emb // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
        angle = 1000 * t[:, None] * freqs[None, :]
        temb = self.time(torch.cat([angle.sin(), angle.cos()], dim=1))
        emb = temb + self.label(label)
        h = self.inp(torch.cat([x, context], dim=1).transpose(1, 2))
        for block in self.blocks:
            h = block(h, emb)
        return self.out(F.silu(self.norm(h))).transpose(1, 2) + self.skip(temb)[:, :, None] * x


def random_mask(batch, frames, gen):
    """Half the rows mask the whole clip; the rest mask one span of 70 to 100 percent."""
    span = (frames * (0.7 + 0.3 * torch.rand(batch, generator=gen))).long()
    start = (torch.rand(batch, generator=gen) * (frames - span + 1)).long()
    position = torch.arange(frames)[None, :]
    mask = ((position >= start[:, None]) & (position < (start + span)[:, None])).float()
    mask[torch.rand(batch, generator=gen) < 0.5] = 1.0
    return mask


def linear_probe(features, labels, steps=300):
    """A softmax classifier on flattened mels, to score generated digits."""
    torch.manual_seed(0)
    w = torch.zeros(features.shape[1], 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(features @ w + b, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda f: (f @ w + b).argmax(dim=1)


def to_db(log_power):
    return 10 * np.log10(np.e) * log_power


def experiment(device, width, n_blocks, batch, steps, lr):
    """Train the mel flow on fsdd, score it, show its mels, and play them."""
    started = time.perf_counter()
    if has_asset("fsdd"):
        clips, labels, is_test = load_digits()
        source = "fsdd"
    else:
        print("The fsdd asset is not on this runner, so the run uses synthetic buzzy vowels instead.")
        clips, labels, is_test = synthetic_digits()
        source = "synthetic"
    mels = np.stack([log_mel(c) for c in clips]).astype(np.float32)
    frames = mels.shape[2]
    print(f"{len(clips)} {source} clips of {SECONDS} s at {RATE} Hz -> log-mels of {N_MELS} bands x "
          f"{frames} frames ({RATE / HOP} frames per second)")

    train_mask = ~is_test
    mean = mels[train_mask].mean(axis=(0, 2), keepdims=True)
    std = mels[train_mask].std(axis=(0, 2), keepdims=True)
    data = torch.tensor((mels - mean) / std)
    digits = torch.tensor(labels)
    train_x, train_y = data[train_mask].to(device), digits[train_mask].to(device)

    torch.manual_seed(0)
    model = MelFlow(width, n_blocks).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(0)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(train_x), (batch,), generator=gen).to(device)
        x1, y = train_x[idx], train_y[idx].clone()
        x0 = torch.randn(x1.shape, generator=gen).to(device)
        t = torch.rand(len(x1), generator=gen).to(device)
        mask = random_mask(len(x1), frames, gen).to(device)
        x_t, context, target = infilling_batch(x0, x1, t, mask)
        drop = (torch.rand(len(x1), generator=gen) < 0.2).to(device)  # p_uncond = 0.2
        y[drop] = NULL
        context[drop] = 0
        loss = masked_flow_loss(model(x_t, t, y, context), target, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {np.mean(losses[-50:]):.3f}  ({time.perf_counter() - started:.0f} s)")
    show_plot(np.convolve(losses, np.ones(25) / 25, mode="valid"), label="Masked flow-matching loss",
              xlabel="Step", ylabel="Loss (25-step average)")

    # Score generated mels with a linear classifier trained on real ones.
    model.eval()
    flat = data.reshape(len(data), -1)
    probe = linear_probe(flat[train_mask], digits[train_mask])
    real_acc = (probe(flat[is_test]) == digits[is_test]).float().mean().item()
    print(f"A linear classifier trained on real mels scores {real_acc:.0%} on held-out real digits.")

    def velocity(x, t, label):
        return model(x, t, label, torch.zeros_like(x))

    def generate(labels_, nfe, alpha, seed):
        noise = torch.randn(len(labels_), N_MELS, frames, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            out = sample(velocity, noise.to(device), labels_.to(device),
                         torch.full_like(labels_, NULL).to(device), nfe, -1.0, alpha)
        return out.cpu()

    wanted = torch.arange(10).repeat(10)
    for nfe, alpha in ((16, 0.0), (16, 2.0), (4, 2.0)):
        fake = generate(wanted, nfe, alpha, seed=1)
        acc = (probe(fake.reshape(len(fake), -1)) == wanted).float().mean().item()
        print(f"Generated digits, NFE {nfe:2d}, CFG alpha {alpha}: the classifier reads {acc:.0%} of them as the right digit")

    ten = generate(torch.arange(10), 16, 2.0, seed=2).numpy() * std + mean
    real_ten = np.stack([mels[np.flatnonzero(is_test & (labels == d))[0]] for d in range(10)])
    show_spectrogram(to_db(np.concatenate(list(real_ten), axis=1)), RATE, HOP, db=False, y_axis="mel",
                     fmax=RATE / 2, label="Real test digits 0 to 9, log-mel", value_label="dB")
    show_spectrogram(to_db(np.concatenate(list(ten), axis=1)), RATE, HOP, db=False, y_axis="mel",
                     fmax=RATE / 2, label="Generated digits 0 to 9 (NFE 16, sway -1, CFG alpha 2)", value_label="dB")

    gap = np.zeros(int(0.15 * RATE))

    def vocode(log_mels):
        pieces = []
        for m in log_mels:
            magnitude = mel_to_magnitude(np.exp(m), FILTERBANK)
            wave = griffin_lim(magnitude, N_FFT, HOP, n_iter=64, momentum=0.99, seed=0)
            pieces += [0.8 * wave / (np.abs(wave).max() + 1e-8), gap]
        return np.concatenate(pieces)

    play(vocode(ten), RATE, label="Generated: zero to nine, through the pseudo-inverse and Griffin-Lim")
    play(vocode(real_ten), RATE, label="Real mels through the same vocoder, the best this pipeline can sound")
    show_image(chunk_causal_mask(frames, 8).astype(float), label="CosyVoice-style chunk mask, 8-frame chunks",
               xlabel="Key frame", ylabel="Query frame", colormap="gray")
    print(f"Done in {time.perf_counter() - started:.0f} s")
    return ten, real_ten


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        experiment(device, width=256, n_blocks=12, batch=128, steps=10000, lr=1e-3)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        print("No GPU here, so this is the small version: a 0.74M-parameter model trained for "
              "700 steps, about 30 s. On a GPU the same block trains a 5.7M-parameter model "
              "for 10,000 steps.")
        experiment(device, width=96, n_blocks=8, batch=64, steps=700, lr=2e-3)
