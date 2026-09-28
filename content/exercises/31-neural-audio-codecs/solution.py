"""Neural audio codecs: the worked solution."""

import glob
import json
import math
import time

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from speechlab import asset_path, has_asset, play, read_audio, show_plot


# Provided: a log-mel front end (built in the lesson on hearing and the mel scale).


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """HTK-style triangular mel filters from 0 Hz to Nyquist, shape (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    bins = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = mel_to_hz(np.linspace(0.0, hz_to_mel(sample_rate / 2), n_mels + 2))
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rise = (bins[None, :] - lower) / (centre - lower)
    fall = (upper - bins[None, :]) / (upper - centre)
    return np.maximum(0.0, np.minimum(rise, fall))


def log_mel(x: np.ndarray, sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """log10 of a magnitude mel spectrogram, floored at 1e-5, shape (n_mels, n_frames).

    Frames of `n_fft` samples with hop `n_fft // 4`, a periodic Hann window, no padding.
    """
    hop = n_fft // 4
    n_frames = 1 + (len(x) - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    magnitude = np.abs(np.fft.rfft(x[index] * window, axis=1)).T
    return np.log10(np.maximum(mel_filterbank(sample_rate, n_fft, n_mels) @ magnitude, 1e-5))


# Bitrate --------------------------------------------------------------------


def frame_rate(sample_rate: float, strides) -> float:
    """Frames per second after an encoder whose strides multiply to its hop."""
    return sample_rate / math.prod(strides)


def bitrate(frames_per_second: float, n_codebooks: int, codebook_size: int) -> float:
    """Bits per second when every frame sends one index from each codebook."""
    return frames_per_second * n_codebooks * math.log2(codebook_size)


# Quantizer dropout ------------------------------------------------------------


def sample_quantizer_counts(batch: int, n_q: int, p: float, generator: torch.Generator) -> torch.Tensor:
    """How many RVQ stages each example uses: uniform in 1..n_q with probability p, else n_q."""
    counts = torch.randint(1, n_q + 1, (batch,), generator=generator)
    drop = torch.rand(batch, generator=generator) < p
    return torch.where(drop, counts, torch.full_like(counts, n_q))


# Residual VQ in torch ---------------------------------------------------------


def rvq_forward(z: torch.Tensor, codebooks: torch.Tensor, n_q):
    """Residual VQ over a (B, D, T) latent with a per-example number of stages.

    Returns (z_q, codes, codebook_loss, commitment_loss).
    """
    B, D, T = z.shape
    N_q = codebooks.shape[0]
    if not torch.is_tensor(n_q):
        n_q = torch.full((B,), int(n_q))
    x = z.permute(0, 2, 1)                                  # (B, T, D)
    residual = x
    hard = torch.zeros_like(x.detach())
    codes = torch.full((B, N_q, T), -1, dtype=torch.long)
    codebook_loss = z.new_zeros(())
    commitment_loss = z.new_zeros(())
    for i in range(N_q):
        active = (n_q > i).to(z.device)                     # (B,)
        if not bool(active.any()):
            break
        C = codebooks[i]
        r = residual.detach()
        dist = (r**2).sum(-1, keepdim=True) - 2 * r @ C.detach().T + (C.detach() ** 2).sum(-1)
        idx = dist.argmin(-1)                                # (B, T)
        q = C[idx]                                           # (B, T, D), carries grad to C
        weight = active[:, None].float().expand(B, T)
        count = weight.sum()
        codebook_loss = codebook_loss + (((r - q) ** 2).sum(-1) * weight).sum() / count
        commitment_loss = commitment_loss + (((residual - q.detach()) ** 2).sum(-1) * weight).sum() / count
        q_hard = q.detach() * active[:, None, None]
        hard = hard + q_hard
        residual = residual - q_hard
        codes[:, i] = torch.where(active[:, None], idx, torch.full_like(idx, -1))
    z_q = x + (hard - x).detach()
    return z_q.permute(0, 2, 1), codes, codebook_loss, commitment_loss


# A toy codec ----------------------------------------------------------------


class ToyCodec(nn.Module):
    """A convolutional encoder, residual VQ, and a mirrored decoder."""

    def __init__(self, strides=(2, 4, 4), channels=32, dim=16, n_q=4, codebook_size=64):
        super().__init__()
        self.strides = tuple(strides)
        self.hop = math.prod(strides)
        self.n_q = n_q
        enc = [nn.Conv1d(1, channels, 7, padding=3)]
        for s in strides:
            enc += [nn.ELU(), nn.Conv1d(channels, channels, s, stride=s), nn.ELU(), nn.Conv1d(channels, channels, 7, padding=3)]
        enc += [nn.ELU(), nn.Conv1d(channels, dim, 1)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv1d(dim, channels, 7, padding=3)]
        for s in reversed(strides):
            dec += [nn.ELU(), nn.ConvTranspose1d(channels, channels, s, stride=s), nn.ELU(), nn.Conv1d(channels, channels, 7, padding=3)]
        dec += [nn.ELU(), nn.Conv1d(channels, 1, 7, padding=3)]
        self.decoder = nn.Sequential(*dec)
        self.codebooks = nn.Parameter(torch.randn(n_q, codebook_size, dim) * 0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor, n_q=None):
        z = self.encode(x)
        z_q, codes, cb_loss, commit = rvq_forward(z, self.codebooks, self.n_q if n_q is None else n_q)
        return self.decode(z_q), codes, cb_loss, commit

    def codes_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(codes.shape[0], codes.shape[2], self.codebooks.shape[2])
        for i in range(codes.shape[1]):
            z = z + self.codebooks[i][codes[:, i]]
        return self.decode(z.permute(0, 2, 1))


# Quality metrics ------------------------------------------------------------


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    alpha = (estimate @ reference) / (reference @ reference)
    target = alpha * reference
    noise = estimate - target
    return float(10 * np.log10((target @ target) / (noise @ noise)))


def mel_distance(x: np.ndarray, y: np.ndarray, sample_rate: int, scales=((256, 20), (512, 40), (1024, 80), (2048, 160))) -> float:
    """Mean absolute difference of log10 mel spectrograms, averaged over (n_fft, n_mels) scales."""
    n = min(len(x), len(y))
    total = 0.0
    for n_fft, n_mels in scales:
        total += np.mean(np.abs(log_mel(x[:n], sample_rate, n_fft, n_mels) - log_mel(y[:n], sample_rate, n_fft, n_mels)))
    return float(total / len(scales))


# Run ------------------------------------------------------------------------


def _librispeech_clips(n_clips: int, seconds: float):
    """The first `seconds` of utterances from different speakers, resampled to 24 kHz."""
    root = asset_path("librispeech-dev-clean")
    speakers = sorted(glob.glob(str(root / "LibriSpeech" / "dev-clean" / "*")))
    clips = []
    for speaker in speakers:
        files = sorted(glob.glob(f"{speaker}/*/*.flac"))
        x, rate = read_audio(files[0])
        if len(x) < seconds * rate:
            continue
        x = x[: int(seconds * rate)]
        clips.append(scipy.signal.resample_poly(x, 3, 2).astype(np.float32))
        if len(clips) == n_clips:
            break
    return clips


def _to16k(x):
    return np.clip(scipy.signal.resample_poly(x, 2, 3), -1, 1)


def _sweep_pretrained(device, small):
    from transformers import EncodecModel, MimiModel

    n_clips, seconds = (2, 3.0) if small else (16, 6.0)
    clips = _librispeech_clips(n_clips, seconds)
    print(f"{len(clips)} LibriSpeech dev-clean clips of {seconds:.0f} s, resampled from 16 kHz to 24 kHz")

    enc_cfg = json.loads((asset_path("encodec-24khz") / "config.json").read_text())
    mimi_cfg = json.loads((asset_path("mimi") / "config.json").read_text())
    enc_fps = frame_rate(enc_cfg["sampling_rate"], enc_cfg["upsampling_ratios"])
    # Mimi's convolutions stop at 25 Hz; one more stride-2 layer halves it to 12.5 Hz.
    mimi_fps = frame_rate(mimi_cfg["sampling_rate"], list(mimi_cfg["upsampling_ratios"]) + [2])
    print(f"EnCodec: {enc_cfg['sampling_rate']} Hz / {math.prod(enc_cfg['upsampling_ratios'])} = {enc_fps} frames/s, "
          f"codebooks of {enc_cfg['codebook_size']}")
    print(f"Mimi:    {mimi_cfg['sampling_rate']} Hz / {math.prod(mimi_cfg['upsampling_ratios']) * 2} = {mimi_fps} frames/s "
          f"(config says {mimi_cfg['frame_rate']}), codebooks of {mimi_cfg['codebook_size']}")

    encodec = EncodecModel.from_pretrained(str(asset_path("encodec-24khz"))).to(device).eval()
    mimi = MimiModel.from_pretrained(str(asset_path("mimi"))).to(device).eval()
    print(f"Mimi in transformers has {mimi.config.num_quantizers} codebooks; "
          f"the first {mimi.config.num_semantic_quantizers} is the semantic one")

    rate = enc_cfg["sampling_rate"]
    results = {"EnCodec": [], "Mimi": []}
    heard = {}
    with torch.no_grad():
        for bw in enc_cfg["target_bandwidths"]:
            n_cb = None
            sis, mels = [], []
            for k, clip in enumerate(clips):
                x = torch.from_numpy(clip)[None, None].to(device)
                out = encodec.encode(x, bandwidth=bw)
                n_cb = out.audio_codes.shape[-2]
                y = encodec.decode(out.audio_codes, out.audio_scales)[0][0, 0, : len(clip)].cpu().numpy()
                sis.append(si_sdr(y, clip))
                mels.append(mel_distance(y, clip, rate))
                if k == 0:
                    heard[("EnCodec", bw)] = y
            kbps = bitrate(enc_fps, n_cb, enc_cfg["codebook_size"]) / 1000
            results["EnCodec"].append((kbps, np.mean(sis), np.mean(mels)))
            print(f"EnCodec {n_cb:2d} codebooks  {kbps:6.2f} kbps  SI-SDR {np.mean(sis):6.2f} dB  mel distance {np.mean(mels):.3f}")
        for n_cb in (1, 2, 4, 8, 16, 32):
            sis, mels = [], []
            for k, clip in enumerate(clips):
                x = torch.from_numpy(clip)[None, None].to(device)
                codes = mimi.encode(x, num_quantizers=n_cb).audio_codes
                y = mimi.decode(codes)[0][0, 0, : len(clip)].cpu().numpy()
                sis.append(si_sdr(y, clip))
                mels.append(mel_distance(y, clip, rate))
                if k == 0:
                    heard[("Mimi", n_cb)] = y
            kbps = bitrate(mimi_fps, n_cb, mimi_cfg["codebook_size"]) / 1000
            results["Mimi"].append((kbps, np.mean(sis), np.mean(mels)))
            print(f"Mimi    {n_cb:2d} codebooks  {kbps:6.2f} kbps  SI-SDR {np.mean(sis):6.2f} dB  mel distance {np.mean(mels):.3f}")

    play(_to16k(clips[0]), 16000, label="The original clip")
    play(_to16k(heard[("EnCodec", 1.5)]), 16000, label="EnCodec at 1.5 kbps (2 codebooks)")
    play(_to16k(heard[("EnCodec", 24.0)]), 16000, label="EnCodec at 24 kbps (32 codebooks)")
    play(_to16k(heard[("Mimi", 1)]), 16000, label="Mimi at 0.14 kbps (the semantic codebook only)")
    play(_to16k(heard[("Mimi", 32)]), 16000, label="Mimi at 4.4 kbps (32 codebooks)")
    return results


def _synthetic_voices(n, length, rate, rng):
    """Buzzy vowel-like clips: an impulse train through three random resonances."""
    out = np.zeros((n, length), dtype=np.float32)
    for i in range(n):
        f0 = rng.uniform(90, 220)
        source = np.diff(np.floor(np.arange(length) * f0 / rate), prepend=0.0)
        y = source
        for f in (rng.uniform(300, 900), rng.uniform(900, 2400), rng.uniform(2400, 3500)):
            r = np.exp(-np.pi * 100.0 / rate)
            y = scipy.signal.lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / rate), r * r], y)
        out[i] = 0.5 * y / (np.max(np.abs(y)) + 1e-9)
    return out


def _train_toy(device, seconds_budget):
    rate, length = 16000, 4096
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ToyCodec().to(device)
    data = torch.from_numpy(_synthetic_voices(256, length, rate, rng))[:, None].to(device)
    with torch.no_grad():  # start every codebook on real latents, stage by stage
        z = model.encode(data[:64]).permute(0, 2, 1).reshape(-1, model.codebooks.shape[2])
        for i in range(model.n_q):
            model.codebooks[i] = z[torch.randperm(len(z))[: model.codebooks.shape[1]]] * (0.5**i)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    start, step, losses = time.time(), 0, []
    while time.time() - start < seconds_budget:
        batch = data[torch.randint(0, len(data), (16,), generator=gen)]
        n_q = sample_quantizer_counts(len(batch), model.n_q, 0.5, gen)
        x_hat, _, cb_loss, commit = model(batch, n_q)
        # A normalized squared error, weighted so it outweighs the VQ losses;
        # with equal weights the encoder shrinks its latents to a point.
        reconstruction = (x_hat - batch).pow(2).mean() / batch.pow(2).mean()
        loss = 100.0 * reconstruction + cb_loss + 0.25 * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
    print(f"trained the toy codec for {step} steps in {time.time() - start:.0f} s; loss {losses[0]:.1f} -> {np.mean(losses[-20:]):.1f}")

    test = torch.from_numpy(_synthetic_voices(4, rate, rate, np.random.default_rng(1)))[:, None].to(device)
    fps = frame_rate(rate, model.strides)
    results, heard = [], {}
    with torch.no_grad():
        for n_q in range(1, model.n_q + 1):
            y = model(test, n_q)[0][:, 0].cpu().numpy()
            ref = test[:, 0].cpu().numpy()
            kbps = bitrate(fps, n_q, model.codebooks.shape[1]) / 1000
            sis = np.mean([si_sdr(y[i], ref[i]) for i in range(len(ref))])
            mels = np.mean([mel_distance(y[i], ref[i], rate) for i in range(len(ref))])
            results.append((kbps, sis, mels))
            heard[n_q] = y[0]
            print(f"toy codec {n_q} codebook(s)  {kbps:5.1f} kbps  SI-SDR {sis:6.2f} dB  mel distance {mels:.3f}")
    play(test[0, 0].cpu().numpy(), rate, label="A synthetic voice")
    play(np.clip(heard[1], -1, 1), rate, label="Toy codec, 1 codebook")
    play(np.clip(heard[model.n_q], -1, 1), rate, label=f"Toy codec, {model.n_q} codebooks")
    return {"toy codec": results}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    small = device == "cpu"
    if small:
        # Some CPU builds of torch route 1-D convolutions through NNPACK, which is
        # many times slower for audio-shaped inputs than the default kernels.
        torch.backends.nnpack.set_flags(False)
    try:
        import transformers  # noqa: F401
        have_models = True
    except ImportError:
        have_models = False
    have_assets = all(has_asset(a) for a in ("encodec-24khz", "mimi", "librispeech-dev-clean"))

    if have_models and have_assets:
        if small:
            print("No GPU: running the small version, 2 clips of 3 s, so the sweep finishes in under a minute.")
        results = _sweep_pretrained(device, small)
    else:
        missing = [a for a in ("encodec-24khz", "mimi", "librispeech-dev-clean") if not has_asset(a)]
        why = "transformers is not installed" if not have_models else f"missing assets: {', '.join(missing)}"
        print(f"The sweep needs encodec-24khz, mimi, and librispeech-dev-clean with transformers ({why}).")
        print("Training your ToyCodec on synthetic voices for 20 seconds instead.")
        results = _train_toy(device, 20.0)

    names = list(results)
    show_plot(*[(np.array([r[0] for r in results[n]]), np.array([r[1] for r in results[n]])) for n in names],
              labels=names, label="SI-SDR against bitrate", xlabel="Bitrate (kbps)", ylabel="SI-SDR (dB)", x_scale="log")
    show_plot(*[(np.array([r[0] for r in results[n]]), np.array([r[2] for r in results[n]])) for n in names],
              labels=names, label="Mel distance against bitrate", xlabel="Bitrate (kbps)", ylabel="Mel distance", x_scale="log")
