"""Spoken language models: discrete units and the adapter pattern, worked solution."""

import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter, resample_poly

from speechlab import asset_path, has_asset, play, show_plot, show_spectrogram

IGNORE = -100  # The label that tells cross-entropy to skip a position.


# Part 1: discrete units -------------------------------------------------------


def assign_units(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """The index of the nearest centroid for every frame.

    features: (T, d) frame vectors. centroids: (K, d).
    Returns an int64 array of shape (T,), the argmin of the squared Euclidean
    distance from each frame to every centroid.
    """
    sq = (
        np.sum(features**2, axis=1)[:, None]
        - 2.0 * features @ centroids.T
        + np.sum(centroids**2, axis=1)[None, :]
    )
    return np.argmin(sq, axis=1).astype(np.int64)


def kmeans(features: np.ndarray, init: np.ndarray, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means from the given initial centroids.

    Each iteration assigns every frame to its nearest centroid, records the
    inertia (the sum of squared distances to the assigned centroids), and then
    moves each centroid to the mean of its frames. A centroid that receives no
    frames stays where it is.
    Returns (centroids of shape (K, d), inertias of shape (n_iter,)).
    """
    centroids = np.array(init, dtype=float, copy=True)
    inertias = np.empty(n_iter)
    for it in range(n_iter):
        units = assign_units(features, centroids)
        inertias[it] = np.sum((features - centroids[units]) ** 2)
        for k in range(len(centroids)):
            members = features[units == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return centroids, inertias


def deduplicate(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse runs of repeated units.

    Returns (the units with each run kept once, the length of each run), so
    that np.repeat(deduplicated, durations) gives back the input.
    """
    units = np.asarray(units)
    if units.size == 0:
        return units.copy(), np.zeros(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, units[1:] != units[:-1]])
    durations = np.diff(np.r_[starts, units.size])
    return units[starts], durations.astype(np.int64)


def unit_bigram_lm(sequences: list, n_units: int, alpha: float = 1.0) -> np.ndarray:
    """An add-alpha bigram model over units.

    Counts every pair (previous, next) of neighbors inside each sequence, adds
    alpha to every cell, and normalizes each row.
    Returns log P(next | previous) as an (n_units, n_units) array whose rows
    each sum to one after np.exp.
    """
    counts = np.full((n_units, n_units), float(alpha))
    for seq in sequences:
        seq = np.asarray(seq)
        np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def sequence_log_prob(sequence: np.ndarray, log_probs: np.ndarray) -> float:
    """The log probability of a unit sequence under a bigram model, in nats.

    Sums log P(u_t | u_{t-1}) over every neighbor pair. The first unit is
    given, so a sequence of length one scores 0.
    """
    sequence = np.asarray(sequence)
    return float(np.sum(log_probs[sequence[:-1], sequence[1:]]))


# Part 2: the adapter ------------------------------------------------------------


def stack_frames(h: torch.Tensor, k: int) -> torch.Tensor:
    """Concatenate every k consecutive frames into one wider frame.

    h: (B, T, d). Returns (B, T // k, k * d), where output frame j holds
    h[:, j*k], h[:, j*k + 1], ..., h[:, j*k + k - 1] side by side. The last
    T % k frames are dropped.
    """
    batch, frames, dim = h.shape
    kept = (frames // k) * k
    return h[:, :kept].reshape(batch, frames // k, k * dim)


class Projector(nn.Module):
    """A two-layer MLP applied to every stacked frame: Linear, GELU, Linear.

    Maps (B, J, d_in) to (B, J, d_model), with a hidden width of d_hidden.
    """

    def __init__(self, d_in: int, d_model: int, d_hidden: int):
        """Create two nn.Linear layers: d_in to d_hidden, then d_hidden to d_model."""
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, J, d_in) to (B, J, d_model): the second layer applied to GELU of the first."""
        return self.fc2(F.gelu(self.fc1(x)))


def build_adapter_inputs(
    audio_embeds: torch.Tensor,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    embed: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splice audio soft tokens, a text prompt, and the target text.

    audio_embeds: (B, J, d), the projector's output.
    prompt_ids: (P,) token ids shared by every example.
    target_ids: (B, U) token ids of each transcript, ending with end-of-text.
    embed: the language model's input embedding table, (V, d).
    Returns inputs_embeds of shape (B, J + P + U, d) laid out as
    [audio | prompt | target], and labels of shape (B, J + P + U) that hold
    IGNORE (-100) at every audio and prompt position and the target ids at
    the target positions.
    """
    batch, n_audio, _ = audio_embeds.shape
    prompt = embed(prompt_ids.to(audio_embeds.device))[None].expand(batch, -1, -1)
    target = embed(target_ids)
    inputs = torch.cat([audio_embeds, prompt.to(audio_embeds.dtype), target.to(audio_embeds.dtype)], dim=1)
    labels = torch.full(inputs.shape[:2], IGNORE, dtype=torch.long, device=inputs.device)
    labels[:, n_audio + len(prompt_ids) :] = target_ids
    return inputs, labels


def next_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over the positions that carry a label.

    logits: (B, L, V). labels: (B, L), IGNORE where there is no target.
    The logits at position t predict the label at position t + 1, so shift
    by one, then average -log softmax over every labeled position.
    """
    vocab = logits.shape[-1]
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
    )


# Run helpers ----------------------------------------------------------------------

RATE = 16000
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
          "e": (530, 1840, 2480), "o": (570, 840, 2410)}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def vowel_string(letters: str, f0: float = 120.0, seconds_each: float = 0.2) -> np.ndarray:
    """A source-filter vowel sequence: an impulse train through three formant resonators."""
    out = []
    for letter in letters:
        n = int(seconds_each * RATE)
        src = np.zeros(n)
        src[:: int(RATE / f0)] = 1.0
        y = src
        for f, bw in zip(VOWELS[letter], (80, 100, 140)):
            r = np.exp(-np.pi * bw / RATE)
            y = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
        out.append(y * np.hanning(n) ** 0.25)
    x = np.concatenate(out)
    return 0.5 * x / np.max(np.abs(x))


def log_spectra(x: np.ndarray, n_fft: int = 512, hop: int = 320) -> np.ndarray:
    """Log-magnitude frames at 50 Hz (a 20 ms hop), shape (frames, n_fft // 2 + 1)."""
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.log(np.abs(np.fft.rfft(x[idx] * np.hanning(n_fft), axis=1)) + 1e-3)


def textless_demo(rng: np.random.Generator) -> None:
    letters = "".join(rng.choice(list("aiueo"), 15))
    x = vowel_string(letters)
    play(x, RATE, label=f"A synthetic vowel string: {letters}")
    feats = log_spectra(x)
    show_spectrogram(np.exp(feats).T, RATE, 320, label="Its spectrogram, 50 frames a second")
    init = feats[rng.choice(len(feats), 8, replace=False)]
    centroids, inertia = kmeans(feats, init, 20)
    units = assign_units(feats, centroids)
    dedup, dur = deduplicate(units)
    print(f"k-means inertia fell from {inertia[0]:.0f} to {inertia[-1]:.0f} over 20 iterations")
    print("first 40 units:", " ".join(map(str, units[:40])))
    print(f"deduplicated:   {' '.join(map(str, dedup[:20]))} ...  ({len(units)} frames -> {len(dedup)} units)")


class ToyLM(nn.Module):
    """A frozen, randomly initialized one-layer causal attention LM with a tied head."""

    def __init__(self, vocab: int = 12, d: int = 32):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(vocab, d)
        nn.init.normal_(self.embed.weight, std=0.5)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        q, k, v = self.qkv(x).chunk(3, -1)
        a = (q @ k.transpose(1, 2)) / x.shape[-1] ** 0.5
        a = a.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float("-inf")).softmax(-1)
        return self.norm(x + self.out(a @ v)) @ self.embed.weight.T


def toy_adapter() -> None:
    """The same adapter on synthetic 'speech': 10 classes of noisy frames, a frozen toy LM."""
    lm = ToyLM()
    means = torch.randn(10, 6, generator=torch.Generator().manual_seed(123))
    prompt, eos = torch.tensor([10]), 11

    def data(n, seed):
        g = torch.Generator().manual_seed(seed)
        y = torch.randint(10, (n,), generator=g)
        return means[y][:, None, :].expand(n, 12, 6) + 0.7 * torch.randn(n, 12, 6, generator=g), y

    torch.manual_seed(1)
    proj = Projector(12, 32, 64)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    x_test, y_test = data(500, 7)

    def accuracy():
        with torch.no_grad():
            z = proj(stack_frames(x_test, 2))
            logits = lm(torch.cat([z, lm.embed(prompt)[None].expand(500, -1, -1)], 1))
            return (logits[:, -1].argmax(-1) == y_test).float().mean().item()

    print(f"toy adapter, held-out accuracy before training: {accuracy():.0%}")
    losses = []
    for step in range(800):
        x, y = data(32, 1000 + step)
        target = torch.stack([y, torch.full_like(y, eos)], 1)
        inputs, labels = build_adapter_inputs(proj(stack_frames(x, 2)), prompt, target, lm.embed)
        loss = next_token_loss(lm(inputs), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    show_plot(np.array(losses), label="Toy adapter loss", xlabel="Step", ylabel="Cross-entropy (nats)", y_scale="log")
    print(f"toy adapter, held-out accuracy after 800 steps: {accuracy():.0%}")


def fsdd_files() -> list:
    root = asset_path("fsdd")
    return sorted(glob.glob(os.path.join(str(root), "*", "recordings", "*.wav")))


def load_clip(path: str, seconds: float = 1.0) -> np.ndarray:
    from speechlab import read_audio

    x, rate = read_audio(path)
    x = resample_poly(x, RATE, rate).astype(np.float32)  # 8 kHz to 16 kHz
    n = int(seconds * RATE)
    return np.pad(x[:n], (0, max(0, n - len(x[:n]))))


def adapter_experiment(device: str, full: bool) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    torch.manual_seed(0)
    started = time.time()
    files = fsdd_files()
    if not full:
        files = [f for f in files if int(os.path.basename(f).split("_")[2][:-4]) < 2]
    speakers = [os.path.basename(f).split("_")[1] for f in files]
    digits = torch.tensor([int(os.path.basename(f).split("_")[0]) for f in files])
    test = torch.tensor([s == "theo" for s in speakers])

    extractor = WhisperFeatureExtractor.from_pretrained(asset_path("whisper-tiny"))
    encoder = WhisperModel.from_pretrained(asset_path("whisper-tiny")).encoder.to(device).eval()
    tok = AutoTokenizer.from_pretrained(asset_path("smollm2-135m"))
    lm = AutoModelForCausalLM.from_pretrained(asset_path("smollm2-135m"), dtype=torch.float32).to(device).eval()
    for p in list(lm.parameters()) + list(encoder.parameters()):
        p.requires_grad_(False)
    embed = lm.get_input_embeddings()

    # Frozen encoder: 30 s padded input, 1500 frames at 50 Hz; keep the first second.
    clips = [load_clip(f) for f in files]
    feats = []
    with torch.no_grad():
        for i in range(0, len(clips), 64):
            mel = extractor(clips[i : i + 64], sampling_rate=RATE, return_tensors="pt").input_features
            feats.append(encoder(mel.to(device)).last_hidden_state[:, :50].float())
    feats = torch.cat(feats)
    print(f"encoded {len(files)} clips into {tuple(feats.shape)} Whisper frames in {time.time() - started:.1f} s")

    k = 4
    word_ids = torch.tensor([tok(" " + w).input_ids[0] for w in WORDS])
    eos = tok.eos_token_id
    prompt_ids = torch.tensor(tok(" Transcript:").input_ids)
    projector = Projector(k * feats.shape[-1], embed.weight.shape[1], 1024).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    print(f"trainable projector: {n_params:,} parameters; frozen LM: {sum(p.numel() for p in lm.parameters()):,}")

    def targets(idx):
        return torch.stack([word_ids[digits[idx]], torch.full((len(idx),), eos)], 1).to(device)

    @torch.no_grad()
    def transcribe(idx):
        z = projector(stack_frames(feats[idx], k))
        x = torch.cat([z, embed(prompt_ids.to(device))[None].expand(len(idx), -1, -1)], 1)
        out = []
        for _ in range(3):
            nxt = lm(inputs_embeds=x).logits[:, -1].argmax(-1)
            out.append(nxt)
            x = torch.cat([x, embed(nxt)[:, None]], 1)
        out = torch.stack(out, 1).cpu()
        return [tok.decode(o).split(tok.eos_token)[0].strip() for o in out]

    def accuracy(idx):
        texts = []
        for i in range(0, len(idx), 100):
            texts += transcribe(idx[i : i + 100])
        return float(np.mean([t == WORDS[digits[j]] for t, j in zip(texts, idx.tolist())])), texts

    train_idx, test_idx = torch.where(~test)[0], torch.where(test)[0]
    shown = test_idx[torch.randperm(len(test_idx), generator=torch.Generator().manual_seed(0))[:6]]
    acc_before, _ = accuracy(test_idx)
    print(f"before training, held-out speaker accuracy {acc_before:.0%}:")
    for j, t in zip(shown.tolist(), transcribe(shown)):
        print(f"  said {WORDS[digits[j]]!r:9} -> {t!r}")

    steps, batch = (1500, 32) if full else (80, 16)
    opt = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    losses = []
    t0 = time.time()
    for step in range(steps):
        idx = train_idx[torch.randint(len(train_idx), (batch,), generator=g)]
        inputs, labels = build_adapter_inputs(projector(stack_frames(feats[idx], k)), prompt_ids, targets(idx), embed)
        loss = next_token_loss(lm(inputs_embeds=inputs).logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"trained {steps} steps of {batch} in {time.time() - t0:.1f} s; loss {losses[0]:.2f} -> {np.mean(losses[-10:]):.3f}")
    show_plot(np.array(losses), label="Loss on the transcript tokens", xlabel="Step", ylabel="Cross-entropy (nats)", y_scale="log")

    acc_after, texts = accuracy(test_idx)
    print(f"after training, held-out speaker accuracy {acc_after:.0%} on {len(test_idx)} clips:")
    for j, t in zip(shown.tolist(), transcribe(shown)):
        print(f"  said {WORDS[digits[j]]!r:9} -> {t!r}")
    j = int(shown[0])
    play(clips[j], RATE, label=f"Held-out speaker says {WORDS[digits[j]]!r}; the model wrote {transcribe(shown[:1])[0]!r}")
    print(f"total {time.time() - started:.1f} s on {device}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Part 1: units from a synthetic vowel string")
    textless_demo(rng)

    print("\nPart 2: frozen whisper-tiny encoder -> your projector -> frozen smollm2-135m")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    try:
        import transformers  # noqa: F401

        ready = all(has_asset(a) for a in ("whisper-tiny", "smollm2-135m", "fsdd"))
    except ImportError:
        ready = False
    if not ready:
        print("The real experiment needs transformers and the assets whisper-tiny, smollm2-135m, "
              "and fsdd, which only the GPU worker has. Running the same adapter on synthetic "
              "frames and a frozen toy LM instead.")
        toy_adapter()
    else:
        full = device == "cuda"
        if not full:
            print("No GPU, so this is the small version: 120 clips, 80 steps, 20 held-out clips.")
        adapter_experiment(device, full)
