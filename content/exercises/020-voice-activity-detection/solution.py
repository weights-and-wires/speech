"""Voice activity detection: the worked solution."""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot


def frame_energy_db(x: np.ndarray, frame_length: int = 400, hop: int = 160) -> np.ndarray:
    """10 log10(mean power + 1e-10) of centered, reflect-padded frames, shape (1 + len(x) // hop,)."""
    x = np.asarray(x, dtype=float)
    padded = np.pad(x, frame_length // 2, mode="reflect")
    n_frames = 1 + len(x) // hop
    index = np.arange(frame_length)[None, :] + hop * np.arange(n_frames)[:, None]
    return 10.0 * np.log10(np.mean(padded[index] ** 2, axis=1) + 1e-10)


def hysteresis(energy_db: np.ndarray, on_db: float, off_db: float, hangover: int) -> np.ndarray:
    """Speech starts above on_db and ends after more than `hangover` frames below off_db."""
    speech, quiet = False, 0
    out = np.zeros(len(energy_db), dtype=bool)
    for t, e in enumerate(energy_db):
        if not speech:
            if e > on_db:
                speech, quiet = True, 0
        elif e < off_db:
            quiet += 1
            if quiet > hangover:
                speech = False
        else:
            quiet = 0
        out[t] = speech
    return out


def energy_vad(x, sample_rate=16000, frame_ms=25.0, hop_ms=10.0, on_margin_db=9.0, off_margin_db=5.0,
               hangover_ms=50.0, floor_percentile=10.0) -> np.ndarray:
    """Energy VAD: thresholds relative to a percentile noise floor, with hysteresis and hangover."""
    frame_length = int(round(frame_ms * sample_rate / 1000))
    hop = int(round(hop_ms * sample_rate / 1000))
    energy = frame_energy_db(x, frame_length, hop)
    floor = np.percentile(energy, floor_percentile)
    return hysteresis(energy, floor + on_margin_db, floor + off_margin_db, int(round(hangover_ms / hop_ms)))


def frame_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    """F1 of the speech frames: 2 TP / (2 TP + FP + FN), or 1.0 when neither has any speech."""
    pred, ref = np.asarray(pred, dtype=bool), np.asarray(ref, dtype=bool)
    tp = np.sum(pred & ref)
    denominator = 2 * tp + np.sum(pred & ~ref) + np.sum(~pred & ref)
    return 1.0 if denominator == 0 else float(2 * tp / denominator)


def track_noise_floor(energy_db: np.ndarray, up_db_per_frame: float, down_alpha: float,
                      speech_mask: np.ndarray | None = None) -> np.ndarray:
    """Track the noise floor in dB, fast down and slow up, frozen on speech frames.

    floor[0] = energy_db[0]. For t >= 1, with prev = floor[t - 1] and e = energy_db[t]:
      * speech_mask[t] is True:  floor[t] = prev (frozen).
      * e < prev:                floor[t] = prev + down_alpha * (e - prev).
      * e >= prev:               floor[t] = prev + min(e - prev, up_db_per_frame).
    A mask of None means no frame is speech.
    """
    energy = np.asarray(energy_db, dtype=float)
    floor = np.empty(len(energy))
    if len(energy) == 0:
        return floor
    mask = np.zeros(len(energy), dtype=bool) if speech_mask is None else np.asarray(speech_mask, dtype=bool)
    floor[0] = energy[0]
    for t in range(1, len(energy)):
        prev = floor[t - 1]
        if mask[t]:
            floor[t] = prev  # the wrong version updates here too, so the floor climbs into the speech
        elif energy[t] < prev:
            floor[t] = prev + down_alpha * (energy[t] - prev)
        else:
            floor[t] = prev + min(energy[t] - prev, up_db_per_frame)
    return floor


def adaptive_vad(energy_db: np.ndarray, margin_db: float = 9.0, gap_db: float = 4.0, hangover: int = 5,
                 up_db_per_frame: float = 0.3, down_alpha: float = 0.3, window_frames: int = 50):
    """An energy VAD whose noise floor follows the background; returns (decisions, floor).

    1. running_min[t] is the minimum of energy_db over the last `window_frames` frames, shortened at the start.
    2. A frame is clearly speech when energy_db[t] > running_min[t] + margin_db.
    3. floor = track_noise_floor(energy_db, up_db_per_frame, down_alpha, clearly_speech).
    4. decisions = hysteresis(energy_db - floor, margin_db, margin_db - gap_db, hangover).
    """
    energy = np.asarray(energy_db, dtype=float)
    running_min = np.array([energy[max(0, t - window_frames + 1) : t + 1].min() for t in range(len(energy))])
    floor = track_noise_floor(energy, up_db_per_frame, down_alpha, energy > running_min + margin_db)
    return hysteresis(energy - floor, margin_db, margin_db - gap_db, hangover), floor


def frame_features(x: np.ndarray, frame_length: int = 400, hop: int = 160) -> np.ndarray:
    """Provided: per-frame [log energy in dB, spectral flatness in dB], shape (1 + len(x) // hop, 2).

    Flatness is 10 log10 of the geometric over the arithmetic mean of the Hann-windowed power
    spectrum from bin 3 to 4 kHz: near 0 dB for noise, far below it for voiced speech.
    """
    x = np.asarray(x, dtype=float)
    padded = np.pad(x, frame_length // 2, mode="reflect")
    index = np.arange(frame_length)[None, :] + hop * np.arange(1 + len(x) // hop)[:, None]
    frames = padded[index]
    energy = 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    power = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1)) ** 2 + 1e-12
    band = power[:, 3 : int(4000 * frame_length / 16000) + 1]
    flatness = 10.0 * np.log10(np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1))
    return np.stack([energy, flatness], axis=1)


def fit_gaussians(features: np.ndarray, is_speech: np.ndarray):
    """Provided: (speech_mean, speech_var, noise_mean, noise_var) from labeled frames, each of shape (D,)."""
    features, is_speech = np.asarray(features, dtype=float), np.asarray(is_speech, dtype=bool)
    return (features[is_speech].mean(axis=0), features[is_speech].var(axis=0) + 1e-6,
            features[~is_speech].mean(axis=0), features[~is_speech].var(axis=0) + 1e-6)


def gaussian_llr(features: np.ndarray, speech_mean: np.ndarray, speech_var: np.ndarray,
                 noise_mean: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
    """log N(x; speech) - log N(x; noise) per frame, for diagonal Gaussians; features is (T, D), returns (T,)."""
    x = np.asarray(features, dtype=float)

    def log_density(mean, var):
        mean, var = np.asarray(mean, dtype=float), np.asarray(var, dtype=float)
        # The -0.5 log(var) term matters: a wrong version drops it and favors the wider Gaussian.
        return np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mean) ** 2 / var, axis=1)

    return log_density(speech_mean, speech_var) - log_density(noise_mean, noise_var)


def endpoint(vad: np.ndarray, hangover_frames: int) -> int:
    """The frame at which the utterance is judged to end, or -1 if it has not ended.

    After the first speech frame, count consecutive non-speech frames; the utterance ends at the frame
    where that count first exceeds hangover_frames, which is the last speech frame + hangover_frames + 1.
    """
    quiet, started = 0, False
    for t, is_speech in enumerate(np.asarray(vad, dtype=bool)):
        if is_speech:
            started, quiet = True, 0
        elif started:
            quiet += 1
            if quiet > hangover_frames:
                return t
    return -1


if __name__ == "__main__":
    rate = 16000

    def noisy_speech(seed, snr_db, seconds=3.0, step_db=0.0, step_at=None):
        """Voiced bursts through formant filters, in pink noise at snr_db; truth is one flag per 10 ms frame."""
        rng = np.random.default_rng(seed)
        n = int(seconds * rate)
        t = np.arange(n) / rate
        truth, speech, start = np.zeros(n, dtype=bool), np.zeros(n), 0.2
        while True:
            length = rng.uniform(0.2, 0.6)
            if start + length > seconds - 0.2:
                break
            span = (t >= start) & (t < start + length)
            truth |= span
            envelope = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0.0, 1.0) * span * rng.uniform(0.5, 1.0)
            f0 = rng.uniform(90.0, 250.0)
            phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))) / rate
            speech += envelope * sum(np.sin(k * phase) / k for k in range(1, int(4000 // f0)))
            start += length + rng.uniform(0.15, 0.5)
        for f, bw in [(rng.uniform(400, 800), 80), (rng.uniform(1000, 2000), 100), (2500, 150)]:
            r = np.exp(-np.pi * bw / rate)
            a = [1.0, -2 * r * np.cos(2 * np.pi * f / rate), r * r]
            speech = lfilter([sum(a)], a, speech)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(np.arange(len(spectrum)), 1)), n)
        noise = pink * np.sqrt(np.mean(speech[truth] ** 2) / np.mean(pink**2) / 10 ** (snr_db / 10))
        if step_at is not None:  # the noise rises by step_db dB over 50 ms, like a fan switching on
            noise = noise * 10 ** (step_db / 20 * np.clip((t - step_at) / 0.05, 0.0, 1.0))
        mix = speech + noise
        return 0.5 * mix / np.max(np.abs(mix)), truth[np.minimum(np.arange(1 + n // 160) * 160, n - 1)]

    for snr in (20.0, 10.0, 5.0):
        mix, ref = noisy_speech(0, snr)
        decisions = energy_vad(mix, rate)
        print(f"SNR {snr:4.0f} dB: frame F1 {frame_f1(decisions, ref):.3f}")
        if snr == 10.0:
            play(mix, rate, label="Voiced bursts in pink noise at 10 dB SNR")
            energy = frame_energy_db(mix)
            times = np.arange(len(energy)) * 160 / rate
            floor = np.percentile(energy, 10)
            show_plot((times, energy), (times, floor + 20 * decisions), (times, floor + 20 * ref),
                      labels=["frame energy (dB)", "VAD decision", "truth"], label="Energy VAD at 10 dB SNR",
                      xlabel="Time (s)", ylabel="dB")

    mix, ref = noisy_speech(1, 20.0, seconds=4.0, step_db=10.0, step_at=2.0)
    energy = frame_energy_db(mix)
    adaptive, floor = adaptive_vad(energy)
    print(f"Fan switches on at 2 s: fixed-floor frame F1 {frame_f1(energy_vad(mix, rate), ref):.3f}, "
          f"tracked-floor frame F1 {frame_f1(adaptive, ref):.3f}")
    times = np.arange(len(energy)) * 160 / rate
    show_plot((times, energy), (times, floor), (times, np.percentile(energy, 10) + 9.0 + 0 * times),
              labels=["frame energy (dB)", "tracked noise floor", "fixed on threshold"],
              label="A noise step: tracked floor against a fixed threshold", xlabel="Time (s)", ylabel="dB")

    train = [noisy_speech(100 + s, 5.0) for s in range(3)]
    params = fit_gaussians(np.concatenate([frame_features(m) for m, _ in train]), np.concatenate([r for _, r in train]))
    mix, ref = noisy_speech(0, 5.0)
    llr = np.convolve(gaussian_llr(frame_features(mix), *params), np.ones(5) / 5, mode="same")
    decisions = hysteresis(llr, 1.0, -1.0, 5)
    print(f"SNR 5 dB, two-feature LLR VAD: frame F1 {frame_f1(decisions, ref):.3f}, "
          f"energy VAD {frame_f1(energy_vad(mix, rate), ref):.3f}")
    print(f"The utterance ends at frame {endpoint(decisions, 30)} with a 300 ms timeout")
