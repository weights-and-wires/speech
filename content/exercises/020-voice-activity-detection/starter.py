"""Voice activity detection.

Find the speech in a noisy recording. Build frame log energy, a two-threshold
detector with a hangover, and the frame F1 score that grades it. Fill in the
functions below, then press Run to hear and see the detector on noisy synthetic
speech at three signal-to-noise ratios, and Check to test your code.
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
