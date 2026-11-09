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


if __name__ == "__main__":
    rate = 16000

    def noisy_speech(seed, snr_db, seconds=3.0):
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
