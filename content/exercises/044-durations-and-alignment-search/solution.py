"""Durations, the length regulator, variance targets, and monotonic alignment search: the worked solution."""

import numpy as np
import torch

from speechlab import play, show_image, show_plot


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ends = torch.cumsum(durations, dim=1)  # (B, L)
    starts = ends - durations
    n_max = int(ends[:, -1].max())
    frames = torch.arange(n_max)[None, None, :]  # (1, 1, N)
    align = (starts[..., None] <= frames) & (frames < ends[..., None])  # (B, L, N)
    out = torch.einsum("bln,bld->bnd", align.to(h.dtype), h)
    mask = frames[:, 0, :] < ends[:, -1:]
    return out, mask


def durations_from_log(log_durations: torch.Tensor, mask: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    d = torch.round(alpha * (torch.exp(log_durations) - 1.0)).long()
    return torch.clamp(d, min=1) * mask.long()


def frame_energy(magnitude: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(magnitude) ** 2, axis=0))


def phone_average(values: np.ndarray, durations: np.ndarray, voiced: np.ndarray | None = None) -> np.ndarray:
    phone_of_frame = np.repeat(np.arange(len(durations)), durations)
    use = np.ones(len(values), bool) if voiced is None else np.asarray(voiced, bool)
    totals = np.bincount(phone_of_frame[use], weights=values[use], minlength=len(durations))
    counts = np.bincount(phone_of_frame[use], minlength=len(durations))
    return np.divide(totals, counts, out=np.zeros(len(durations)), where=counts > 0)


def quantize(values: np.ndarray, v_min: float, v_max: float, n_bins: int = 256, log_scale: bool = False) -> np.ndarray:
    if log_scale:
        edges = np.exp(np.linspace(np.log(v_min), np.log(v_max), n_bins - 1))
    else:
        edges = np.linspace(v_min, v_max, n_bins - 1)
    return np.searchsorted(edges, values, side="left")


def gaussian_log_likelihood(z: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    inv_var = np.exp(-2.0 * log_sigma)  # (L, D)
    const = -0.5 * np.log(2 * np.pi) * z.shape[1] - log_sigma.sum(axis=1)  # (L,)
    # -(z - mu)^2 / (2 sigma^2), expanded into matrix products over the D dimensions.
    quad = (z**2) @ inv_var.T - 2 * z @ (mu * inv_var).T + np.sum(mu**2 * inv_var, axis=1)  # (T, L)
    return const[:, None] - 0.5 * quad.T


def monotonic_alignment_search(P: np.ndarray) -> np.ndarray:
    n_text, n_frames = P.shape
    Q = np.full((n_text, n_frames), -np.inf)
    Q[0, 0] = P[0, 0]
    for j in range(1, n_frames):
        stay = Q[:, j - 1]
        advance = np.concatenate([[-np.inf], Q[:-1, j - 1]])
        Q[:, j] = np.maximum(stay, advance) + P[:, j]
    path = np.zeros(n_frames, dtype=np.int64)
    i = n_text - 1
    path[-1] = i
    for j in range(n_frames - 1, 0, -1):
        if i > 0 and Q[i - 1, j - 1] > Q[i, j - 1]:
            i -= 1
        path[j - 1] = i
    return path


def path_to_durations(path: np.ndarray, n_text: int) -> np.ndarray:
    return np.bincount(path, minlength=n_text)


# The Run experiment ------------------------------------------------------------

RATE = 16000
HOP = 160  # 10 ms frames


def _sing(f0: np.ndarray, voiced: np.ndarray, rng) -> np.ndarray:
    """A buzzy source for voiced frames and noise for unvoiced ones, from a frame-rate F0 contour."""
    f = np.repeat(np.where(voiced, f0, 0.0), HOP)
    v = np.repeat(voiced.astype(float), HOP)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    buzz = sum(np.sin(k * phase) / k for k in range(1, 8))
    noise = rng.standard_normal(len(f))
    fade = np.convolve(v, np.ones(160) / 160, mode="same")
    return 0.15 * (fade * buzz + 0.3 * (1 - fade) * noise)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_text, dim = 8, 4
    true_d = rng.integers(8, 26, n_text)
    mu = rng.normal(size=(n_text, dim)) * 2.0
    log_sigma = np.full((n_text, dim), np.log(0.8))
    path_true = np.repeat(np.arange(n_text), true_d)
    z = mu[path_true] + 0.8 * rng.standard_normal((len(path_true), dim))

    P = gaussian_log_likelihood(z, mu, log_sigma)
    path = monotonic_alignment_search(P)
    found = path_to_durations(path, n_text)
    print(f"{n_text} phones, {len(path)} frames.")
    print(f"True durations:  {true_d.tolist()}")
    print(f"MAS durations:   {found.tolist()}")
    align = np.zeros((n_text, len(path)))
    align[path, np.arange(len(path))] = 1.0
    show_image(P - P.max(), label="Log-likelihood of each frame under each phone",
               xlabel="Frame (10 ms)", ylabel="Phone", origin="lower", colormap="viridis", value_label="log p")
    show_image(align, label="The monotonic alignment MAS chose", xlabel="Frame (10 ms)", ylabel="Phone",
               origin="lower", colormap="magma", value_label="aligned")

    # Pitch and energy targets: a melody of phones, two of them unvoiced.
    notes = np.array([220.0, 247.0, 0.0, 262.0, 294.0, 262.0, 0.0, 196.0])
    voiced_phone = notes > 0
    frame_f0 = np.repeat(notes, found) * (1 + 0.01 * np.sin(2 * np.pi * 5 * np.arange(len(path)) / 100))
    frame_voiced = np.repeat(voiced_phone, found)
    audio = _sing(frame_f0, frame_voiced, rng)
    spec = torch.stft(torch.from_numpy(audio), 512, HOP, window=torch.hann_window(512, dtype=torch.float64),
                      center=True, return_complex=True).abs().numpy()[:, : len(path)]
    energy = frame_energy(spec)
    phone_f0 = phone_average(frame_f0, found, frame_voiced)
    phone_energy = phone_average(energy, found)
    print(f"Per-phone F0 (Hz):      {np.round(phone_f0, 1).tolist()}")
    print(f"Per-phone pitch bins:   {quantize(phone_f0[voiced_phone], 71.0, 795.0, 256, log_scale=True).tolist()}")
    print(f"Per-phone energy:       {np.round(phone_energy, 1).tolist()}")
    show_plot(np.where(frame_voiced, frame_f0, np.nan), np.repeat(np.where(voiced_phone, phone_f0, np.nan), found),
              labels=["frame F0", "per-phone target"], label="Frame-level F0 and the per-phone pitch target",
              xlabel="Frame (10 ms)", ylabel="Hz")

    # Speed control: the length regulator with durations scaled by alpha.
    log_d = torch.log(torch.tensor(found, dtype=torch.float64) + 1.0)[None]
    mask = torch.ones(1, n_text, dtype=torch.bool)
    h = torch.tensor(np.stack([phone_f0, voiced_phone.astype(float)], axis=1))[None]
    for alpha in (1.0, 1.6):
        d = durations_from_log(log_d, mask, alpha)
        frames, frame_mask = length_regulate(h, d)
        f0_track, voicing = frames[0, :, 0].numpy(), frames[0, :, 1].numpy() > 0.5
        clip = _sing(f0_track * (1 + 0.01 * np.sin(2 * np.pi * 5 * np.arange(len(f0_track)) / 100)), voicing, rng)
        print(f"alpha = {alpha}: {int(d.sum())} frames, {len(clip) / RATE:.2f} s")
        play(clip, RATE, label=f"The phones expanded by the length regulator, alpha = {alpha}")
