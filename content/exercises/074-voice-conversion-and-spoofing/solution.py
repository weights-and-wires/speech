"""Voice conversion and spoofing: the worked solution."""

import numpy as np
from scipy.fft import dct
from scipy.signal import istft as scipy_istft
from scipy.signal import lfilter
from scipy.signal import stft as scipy_stft

from speechlab import play, show_plot, show_spectrogram


# Voice conversion ------------------------------------------------------------------


def cosine_knn(query: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k keys closest to each query under cosine distance.

    query: (T, d). keys: (M, d). Returns (T, k) int indices, nearest first.
    Cosine distance is 1 - x.r / (|x| |r|).
    """
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    r = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    sim = q @ r.T
    top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def knn_vc(source: np.ndarray, matching: np.ndarray, k: int = 4) -> np.ndarray:
    """kNN voice conversion: replace each source frame with the mean of its
    k nearest matching-set frames. source: (T, d). matching: (M, d). Returns (T, d)."""
    return matching[cosine_knn(source, matching, k)].mean(axis=1)


# Spoofing countermeasures ------------------------------------------------------------


def error_rates(bonafide: np.ndarray, spoof: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False rejection and false acceptance rates of a countermeasure at each threshold.

    Higher scores mean "bona fide", and a trial is accepted when its score is at
    or above the threshold. Returns (frr, far), each shaped like `thresholds`:
    frr is the fraction of bona fide scores below the threshold, and far is the
    fraction of spoof scores at or above it.
    """
    bonafide, spoof, thresholds = (np.asarray(v, float) for v in (bonafide, spoof, thresholds))
    frr = np.searchsorted(np.sort(bonafide), thresholds, side="left") / bonafide.size
    far = 1.0 - np.searchsorted(np.sort(spoof), thresholds, side="left") / spoof.size
    return frr, far


def equal_error_rate(bonafide: np.ndarray, spoof: np.ndarray) -> tuple[float, float]:
    """The EER of a countermeasure whose higher scores mean "bona fide".

    Sweep the threshold over every score (and one value above the largest),
    find the first threshold where FRR >= FAR, and interpolate linearly
    between it and the previous threshold to where the two rates cross.
    Returns (eer, threshold).
    """
    bonafide, spoof = np.asarray(bonafide, float), np.asarray(spoof, float)
    th = np.unique(np.concatenate([bonafide, spoof]))
    th = np.append(th, th[-1] + 1.0)
    frr, far = error_rates(bonafide, spoof, th)
    i = int(np.argmax(frr >= far))
    if i == 0 or frr[i] == far[i]:
        return float((frr[i] + far[i]) / 2), float(th[i])
    d0, d1 = far[i - 1] - frr[i - 1], far[i] - frr[i]
    w = d0 / (d0 - d1)
    eer = frr[i - 1] + w * (frr[i] - frr[i - 1])
    return float(eer), float(th[i - 1] + w * (th[i] - th[i - 1]))


# Run helpers -------------------------------------------------------------------------

RATE = 16000
N_FFT, HOP = 512, 160  # 32 ms window, 10 ms hop
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240),
          "e": (530, 1840, 2480), "o": (570, 840, 2410)}


def speak(letters, f0, formant_scale, rng, each=0.25):
    """A synthetic voice: a jittered pulse train through three formant resonators per vowel."""
    out, labels = [], []
    for letter in letters:
        n = int(each * RATE)
        pitch = f0 * (1 + 0.03 * np.sin(2 * np.pi * 3 * np.arange(n) / RATE + rng.uniform(0, 6)))
        src = (np.diff(np.floor(np.cumsum(pitch / RATE)), prepend=0) > 0).astype(float)
        y = src
        for f, bw in zip(VOWELS[letter], (80, 100, 140)):
            f = f * formant_scale
            r = np.exp(-np.pi * bw / RATE)
            y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
        out.append(y / np.abs(y).max() * np.hanning(n) ** 0.2)
        labels += [letter] * (n // HOP)
    x = np.concatenate(out)
    return 0.5 * x / np.abs(x).max(), labels


def stft(x):
    return scipy_stft(x, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")[2]


def istft(X, n):
    return scipy_istft(X, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann")[1][:n]


def mel_matrix(n_mels=40):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = 700 * (10 ** (np.linspace(0, mel(RATE / 2), n_mels + 2) / 2595) - 1)
    bins = np.fft.rfftfreq(N_FFT, 1 / RATE)
    fb = np.zeros((n_mels, bins.size))
    for m in range(n_mels):
        lo, mid, hi = hz[m : m + 3]
        fb[m] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0, None)
    return fb


def content_features(mag):
    """Stand-in for WavLM layer 6: MFCCs 1-19, mean-normalized per speaker."""
    mfcc = dct(np.log(mel_matrix() @ mag**2 + 1e-8), axis=0, norm="ortho")[1:20].T
    return mfcc - mfcc.mean(axis=0)


def griffin_lim(mag, n, n_iter=48, seed=0):
    """Griffin-Lim: alternate between the target magnitudes and a consistent STFT, for n samples."""
    phase = np.exp(2j * np.pi * np.random.default_rng(seed).random(mag.shape))
    for _ in range(n_iter):
        phase = np.exp(1j * np.angle(stft(istft(mag * phase, n))))
    return istft(mag * phase, n)


def pitch(x):
    """F0 in Hz of a voiced clip, from the autocorrelation peak between 70 and 400 Hz."""
    ac = np.correlate(x[: RATE // 2], x[: RATE // 2], "full")[RATE // 2 - 1 :]
    lo, hi = RATE // 400, RATE // 70
    return RATE / (lo + np.argmax(ac[lo:hi]))


def vc_demo(rng):
    source_text = "aeiouoiea"
    src, src_labels = speak(source_text, 110.0, 1.0, rng)
    ref, ref_labels = speak("".join(rng.choice(list("aeiou"), 40)), 210.0, 1.18, rng)  # 10 s matching set
    S, R = stft(src), stft(ref)
    fs, fr = content_features(np.abs(S)), content_features(np.abs(R))
    src_lab = np.array(src_labels + [src_labels[-1]] * S.shape[1])[: S.shape[1]]
    ref_lab = np.array(ref_labels + [ref_labels[-1]] * R.shape[1])[: R.shape[1]]
    idx = cosine_knn(fs, fr, 4)
    agree = np.mean(ref_lab[idx] == src_lab[:, None])
    converted_mag = np.abs(R)[:, idx].mean(axis=2)  # average the k matched frames' magnitude spectra
    out = griffin_lim(converted_mag, len(src))
    out = 0.5 * out / np.abs(out).max()
    print(f"kNN-VC with k = 4 over a {len(ref) / RATE:.0f} s matching set of the target voice:")
    print(f"  {agree:.0%} of the matched frames are the same vowel as the source frame")
    print(f"  pitch: source {pitch(src):.0f} Hz, target {pitch(ref):.0f} Hz, converted {pitch(out):.0f} Hz")
    play(src, RATE, label=f"Source voice (110 Hz) says {source_text}")
    play(ref[: 2 * RATE], RATE, label="Two seconds of the target voice (210 Hz, formants x1.18)")
    play(out, RATE, label="Converted: the source's vowels in the target's voice")
    show_spectrogram(converted_mag, RATE, HOP, label="The converted magnitude spectrogram")


def eer_demo(rng):
    bona = rng.normal(1.5, 1.0, 400)
    spoof = rng.normal(-1.0, 1.0, 400)
    eer, th = equal_error_rate(bona, spoof)
    grid = np.linspace(-4, 5, 200)
    frr = [(bona < t).mean() for t in grid]
    far = [(spoof >= t).mean() for t in grid]
    show_plot((grid, np.array(frr)), (grid, np.array(far)), labels=["false rejection", "false acceptance"],
              label="A toy countermeasure's error rates", xlabel="Threshold", ylabel="Rate")
    print(f"\nToy spoofing countermeasure: EER {eer:.1%} at threshold {th:.2f} (theory for d' = 2.5: 10.6%)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vc_demo(rng)
    eer_demo(rng)
