"""Voice activity detection.

Find the speech in a noisy recording. Build frame log energy, a two-threshold
detector with a hangover, and the frame F1 score that grades it. Then track a
moving noise floor, compare Gaussian likelihoods, and find where an utterance
ends. Fill in the functions below, then press Run to hear and see the detectors
on noisy synthetic speech, and Check to test your code.
"""

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot


def frame_energy_db(x: np.ndarray, frame_length: int = 400, hop: int = 160) -> np.ndarray:
    """Frame log energy in dB, shape (1 + len(x) // hop,).

    Pad x by frame_length // 2 samples at each end by reflection, take frame t
    as padded[t * hop : t * hop + frame_length], and return
    10 * log10(mean(frame ** 2) + 1e-10) for each frame.
    """
    raise NotImplementedError


def hysteresis(energy_db: np.ndarray, on_db: float, off_db: float, hangover: int) -> np.ndarray:
    """Speech decisions with two thresholds and a hangover, a bool array like `energy_db`.

    Start in silence. In silence, switch to speech at a frame whose energy is
    above on_db. In speech, count consecutive frames below off_db; a frame at
    or above off_db resets the count to 0; when the count exceeds `hangover`,
    switch to silence at that frame. Each output is the state after its frame.
    """
    raise NotImplementedError


def energy_vad(x, sample_rate=16000, frame_ms=25.0, hop_ms=10.0, on_margin_db=9.0, off_margin_db=5.0,
               hangover_ms=50.0, floor_percentile=10.0) -> np.ndarray:
    """An energy voice activity detector, one bool per hop, shape (1 + len(x) // hop,).

    Frame lengths are round(ms * sample_rate / 1000) samples. The noise floor
    is the `floor_percentile` percentile of the frame energies in dB. Speech
    switches on above floor + on_margin_db and off below floor + off_margin_db,
    with a hangover of round(hangover_ms / hop_ms) frames.
    """
    raise NotImplementedError


def frame_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    """The F1 score of speech frames: 2 TP / (2 TP + FP + FN), or 1.0 when that denominator is 0."""
    raise NotImplementedError


def track_noise_floor(energy_db: np.ndarray, up_db_per_frame: float, down_alpha: float,
                      speech_mask: np.ndarray | None = None) -> np.ndarray:
    """Track the noise floor in dB, fast down and slow up, frozen on speech frames. Shape like `energy_db`.

    floor[0] = energy_db[0]. For t >= 1, with prev = floor[t - 1] and e = energy_db[t]:
      * speech_mask[t] is True:  floor[t] = prev (frozen).
      * e < prev:                floor[t] = prev + down_alpha * (e - prev).
      * e >= prev:               floor[t] = prev + min(e - prev, up_db_per_frame).
    A mask of None means no frame is speech.
    """
    raise NotImplementedError


def adaptive_vad(energy_db: np.ndarray, margin_db: float = 9.0, gap_db: float = 4.0, hangover: int = 5,
                 up_db_per_frame: float = 0.3, down_alpha: float = 0.3, window_frames: int = 50):
    """An energy VAD whose noise floor follows the background; returns (decisions, floor).

    1. running_min[t] is the minimum of energy_db over the last `window_frames` frames, shortened at the start.
    2. A frame is clearly speech when energy_db[t] > running_min[t] + margin_db.
    3. floor = track_noise_floor(energy_db, up_db_per_frame, down_alpha, clearly_speech).
    4. decisions = hysteresis(energy_db - floor, margin_db, margin_db - gap_db, hangover).
    """
    raise NotImplementedError


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
    raise NotImplementedError


def endpoint(vad: np.ndarray, hangover_frames: int) -> int:
    """The frame at which the utterance is judged to end, or -1 if it has not ended.

    After the first speech frame, count consecutive non-speech frames; the utterance ends at the frame
    where that count first exceeds hangover_frames, which is the last speech frame + hangover_frames + 1.
    """
    raise NotImplementedError


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
