"""Durations, the length regulator, variance targets, and monotonic alignment search.

Fill in the functions, then press Run. The run hides eight phones' durations
in a sequence of noisy frames, finds them again with your alignment search,
computes per-phone pitch and energy targets, and plays the phones expanded by
your length regulator at two speeds.

Frames are 10 ms long. A duration is a whole number of frames.
"""

import numpy as np
import torch

from speechlab import play, show_image, show_plot


def length_regulate(h: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each phone's hidden vector for its duration, for a padded batch.

    Args:
        h: Phone hidden states, shape (B, L, D), float.
        durations: Frames per phone, shape (B, L), integers. Padded phones
            have duration 0.

    Returns:
        (frames, mask). frames has shape (B, N_max, D), where N_max is the
        largest total duration in the batch: frames[b] holds h[b, i] repeated
        durations[b, i] times, in order, then zeros. mask has shape
        (B, N_max), bool, True on real frames. The result must stay
        differentiable with respect to h.
    """
    raise NotImplementedError("length_regulate")


def durations_from_log(log_durations: torch.Tensor, mask: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Turn predicted log durations into whole frame counts.

    Args:
        log_durations: The duration predictor's output, which targets
            log(d + 1), shape (B, L).
        mask: True on real phones, shape (B, L), bool.
        alpha: The speed factor. alpha > 1 makes speech slower.

    Returns:
        A long tensor of shape (B, L): round(alpha * (exp(l) - 1)), at least 1
        on real phones, and 0 on padded phones.
    """
    raise NotImplementedError("durations_from_log")


def frame_energy(magnitude: np.ndarray) -> np.ndarray:
    """The energy of each frame: the L2 norm of its STFT magnitude column.

    magnitude has shape (bins, frames). Return shape (frames,).
    """
    raise NotImplementedError("frame_energy")


def phone_average(values: np.ndarray, durations: np.ndarray, voiced: np.ndarray | None = None) -> np.ndarray:
    """Average a frame-level contour over each phone's frames.

    Args:
        values: One value per frame, shape (N,).
        durations: Frames per phone, shape (L,), summing to N.
        voiced: Optional bool mask, shape (N,). When given, only voiced frames
            count toward a phone's average.

    Returns:
        Shape (L,). Phone i's average is over frames sum(durations[:i]) to
        sum(durations[:i + 1]) - 1. A phone with no counted frames gets 0.
    """
    raise NotImplementedError("phone_average")


def quantize(values: np.ndarray, v_min: float, v_max: float, n_bins: int = 256, log_scale: bool = False) -> np.ndarray:
    """Map values to bin indices 0 .. n_bins - 1, the way FastSpeech 2 embeds pitch and energy.

    The bins are separated by n_bins - 1 edges from v_min to v_max, spaced evenly
    in log(value) when log_scale is True and evenly in value otherwise. A value's
    index is the number of edges strictly below it, so anything at or below v_min
    lands in bin 0 and anything above v_max in bin n_bins - 1.
    """
    raise NotImplementedError("quantize")


def gaussian_log_likelihood(z: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray) -> np.ndarray:
    """Log-likelihood of every frame under every phone's diagonal Gaussian.

    Args:
        z: Frames, shape (T, D).
        mu: Per-phone means, shape (L, D).
        log_sigma: Per-phone log standard deviations, shape (L, D).

    Returns:
        P, shape (L, T): P[i, j] = sum over d of log N(z[j, d]; mu[i, d],
        exp(log_sigma[i, d]) ** 2). Compute it without a Python loop.
    """
    raise NotImplementedError("gaussian_log_likelihood")


def monotonic_alignment_search(P: np.ndarray) -> np.ndarray:
    """The most likely monotonic alignment of T frames to L phones.

    P has shape (L, T) with L <= T. Every frame goes to one phone, frame 0 to
    phone 0 and frame T - 1 to phone L - 1, and from one frame to the next the
    phone either stays or moves forward by one. Every phone gets at least one
    frame. Return the path: an int array of shape (T,) holding the phone of
    each frame, chosen to maximize the sum of P[path[j], j] over frames.
    """
    raise NotImplementedError("monotonic_alignment_search")


def path_to_durations(path: np.ndarray, n_text: int) -> np.ndarray:
    """The number of frames the path gives each of the n_text phones, shape (n_text,).
    """
    raise NotImplementedError("path_to_durations")


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
