"""Objective and subjective scores for synthetic speech.

Fill in the functions, then press Run. The run synthesizes a reference voice,
two "systems" that imitate it, and a different speaker, scores each against
the reference with MCD-DTW, F0 RMSE, and a speaker similarity, simulates a
listening test with bootstrap intervals, and plays all four voices.

Cepstra are (frames, coefficients), with coefficient 0 the frame's energy.
F0 contours are one value per frame in Hz, with 0 meaning unvoiced.
"""

import re

import numpy as np

from speechlab import play, show_image

MCD_CONSTANT = 10.0 / np.log(10.0)  # about 4.3429 dB per unit of cepstral distance


def mel_cepstral_distortion(c_ref: np.ndarray, c_syn: np.ndarray) -> np.ndarray:
    """Mel cepstral distortion in dB between matching frames.

    c_ref and c_syn have the same shape (..., D + 1). Coefficient 0 is energy and
    is left out. Return shape (...): (10 / ln 10) * sqrt(2 * sum over d >= 1 of
    (c_ref[..., d] - c_syn[..., d]) ** 2). Broadcasting must work, so that
    c_ref[:, None, :] against c_syn[None, :, :] gives every pair of frames.
    """
    raise NotImplementedError("mel_cepstral_distortion")


def dtw_path(cost: np.ndarray) -> list[tuple[int, int]]:
    """The cheapest monotone path through a cost matrix, as in the dynamic time warping chapter.

    cost has shape (N, M). The path starts at (0, 0), ends at (N - 1, M - 1), and
    each step moves diagonally (1, 1), down (1, 0), or right (0, 1). Minimize the
    sum of cost over the cells visited. Return a list of (i, j) tuples in order.
    """
    raise NotImplementedError("dtw_path")


def mcd_dtw(c_ref: np.ndarray, c_syn: np.ndarray) -> float:
    """MCD after dynamic time warping: the mean frame MCD along the best DTW path.

    c_ref has shape (N, D + 1) and c_syn shape (M, D + 1). Build the (N, M)
    matrix of frame MCDs, find dtw_path on it, and average the cost over the
    path's cells. Return a Python float in dB.
    """
    raise NotImplementedError("mcd_dtw")


def f0_rmse(f0_ref: np.ndarray, f0_syn: np.ndarray) -> tuple[float, float]:
    """F0 RMSE over frames voiced in both contours, and the voicing error rate.

    f0_ref and f0_syn have the same shape (N,), in Hz, with 0 meaning unvoiced.
    Return (rmse_hz, vuv_error): the root mean squared difference over frames
    where both are voiced (NaN if there are none), and the fraction of all
    frames where exactly one of the two is voiced.
    """
    raise NotImplementedError("f0_rmse")


def speaker_similarity(e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """Cosine similarity between speaker embeddings, one value per pair of rows.

    e1 and e2 have shape (N, E), or (E,) for a single pair. Return shape (N,).
    """
    raise NotImplementedError("speaker_similarity")


def bootstrap_mos(ratings: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """The mean opinion score with a bootstrap interval over raters and items.

    ratings has shape (raters, items). Seed np.random.default_rng(seed). For each
    of n_boot replicates, draw `raters` rater indices and `items` item indices
    with replacement, and take the mean of ratings[np.ix_(rater_idx, item_idx)].
    Return (mean, low, high): the plain mean of all ratings, and the
    (1 - level) / 2 and (1 + level) / 2 percentiles of the replicate means.
    """
    raise NotImplementedError("bootstrap_mos")


def normalize_text(text: str) -> list[str]:
    """Lowercase, turn hyphens into spaces, drop every character except a-z, 0-9,
    apostrophes, and spaces, and split into words. Return a list of words.
    """
    raise NotImplementedError("normalize_text")


def word_error_rate(references: list[str], transcripts: list[str]) -> float:
    """Corpus word error rate of ASR transcripts against the text a TTS system read.

    references and transcripts are lists of strings, one pair per utterance.
    Normalize both with normalize_text, count the word-level Levenshtein edits
    (substitutions, deletions, insertions) of each pair, and return the total
    edits divided by the total number of reference words.
    """
    raise NotImplementedError("word_error_rate")


# Provided: a toy voice, its cepstra, and simulated listeners ---------------------

RATE = 16000
HOP = 160
N_FFT = 512


def _voice(f0: np.ndarray, formants: list[tuple[float, float]], seconds_per_vowel: float, rng,
           breath: float = 0.0) -> np.ndarray:
    """A source-filter voice: an impulse-train source through two-pole formant resonators, one vowel per step.

    f0 holds one pitch per vowel (0 for an unvoiced, noisy vowel); formants holds (F1, F2) per vowel.
    """
    from scipy.signal import lfilter

    out = []
    for pitch, (f1, f2) in zip(f0, formants):
        n = int(seconds_per_vowel * RATE)
        if pitch > 0:
            phase = np.cumsum(np.full(n, pitch / RATE))
            source = np.diff(np.floor(phase), prepend=0.0) + breath * rng.standard_normal(n)
        else:
            source = 0.3 * rng.standard_normal(n)
        y = source
        for f, bw in ((f1, 80.0), (f2, 120.0), (2500.0, 200.0)):
            r = np.exp(-np.pi * bw / RATE)
            y = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r], y)
        out.append(y * np.hanning(n) ** 0.25)
    x = np.concatenate(out)
    return 0.5 * x / np.abs(x).max()


def cepstra(x: np.ndarray, n_mels: int = 40, n_ceps: int = 25) -> np.ndarray:
    """MFCC-style cepstra, shape (frames, n_ceps): log-mel amplitudes and an orthonormal DCT (the mel and MFCC chapters).

    The log is of amplitude, not power, as in the mel-cepstra MCD was defined on, and it is floored
    60 dB below the loudest band so silent frames don't dominate.
    """
    from scipy.fft import dct

    pad = np.pad(x, N_FFT // 2, mode="reflect")
    frames = 1 + (len(pad) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    power = np.abs(np.fft.rfft(pad[idx] * window, axis=1)) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = 700 * (10 ** (np.linspace(mel(0), mel(RATE / 2), n_mels + 2) / 2595) - 1)
    freqs = np.arange(N_FFT // 2 + 1) * RATE / N_FFT
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    fb = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    mel_power = power @ fb.T
    log_amp = 0.5 * np.log(np.maximum(mel_power, mel_power.max() * 1e-6))
    return dct(log_amp, type=2, norm="ortho", axis=1)[:, :n_ceps]


def _frame_f0(f0: np.ndarray, seconds_per_vowel: float, n_frames: int) -> np.ndarray:
    per = int(seconds_per_vowel * RATE / HOP)
    return np.repeat(f0, per)[:n_frames] if len(f0) * per >= n_frames else np.pad(np.repeat(f0, per), (0, n_frames - len(f0) * per))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    f0 = np.array([120, 125, 132, 0, 128, 118, 110, 0, 115, 122.0])
    vowels = [(730, 1090), (270, 2290), (530, 1840), (500, 1500), (660, 1720),
              (300, 870), (570, 840), (500, 1500), (440, 1020), (270, 2290)]
    ref = _voice(f0, vowels, 0.18, rng)
    # System A: close to the reference, but 15% slower and 4% higher.
    sys_a = _voice(f0 * 1.04, vowels, 0.18 * 1.15, rng)
    # System B: flattened pitch, formants pulled toward a neutral vowel, and a breathy source.
    flat = np.where(f0 > 0, 121.0, 0.0)
    blurred = [(0.75 * f1 + 0.25 * 500, 0.75 * f2 + 0.25 * 1500) for f1, f2 in vowels]
    sys_b = _voice(flat, blurred, 0.18, rng, breath=0.1)
    # Another speaker saying the same vowels: higher pitch and a shorter vocal tract.
    other = _voice(f0 * 1.8, [(1.18 * f1, 1.18 * f2) for f1, f2 in vowels], 0.18, rng)

    c_ref = cepstra(ref)
    voices = {"A": (sys_a, f0 * 1.04, 0.18 * 1.15), "B": (sys_b, flat, 0.18), "other speaker": (other, f0 * 1.8, 0.18)}
    ceps = {name: cepstra(x) for name, (x, _, _) in voices.items()}
    # A stand-in speaker embedding: each voice's mean cepstrum without c0, centered on the mean over all
    # four voices. Real systems use a trained speaker verifier instead (the speaker verification chapter).
    means = {name: c[:, 1:].mean(axis=0) for name, c in ceps.items()} | {"reference": c_ref[:, 1:].mean(axis=0)}
    center = np.mean(list(means.values()), axis=0)
    for name, (x, pitch, per) in voices.items():
        c = ceps[name]
        cost = mel_cepstral_distortion(c_ref[:, None, :], c[None, :, :])
        path = dtw_path(cost)
        mcd = float(np.mean([cost[i, j] for i, j in path]))
        pairs = np.array(path)
        rmse, vuv = f0_rmse(_frame_f0(f0, 0.18, len(c_ref))[pairs[:, 0]], _frame_f0(pitch, per, len(c))[pairs[:, 1]])
        sim = speaker_similarity(means["reference"] - center, means[name] - center)[0]
        print(f"{name:13s}: MCD-DTW {mcd:5.2f} dB over {len(path)} path steps, F0 RMSE {rmse:5.1f} Hz, "
              f"V/UV error {100 * vuv:4.1f}%, speaker similarity {sim:+.2f}")
        if name == "B":
            show_image(cost.T, label="MCD between every reference frame and every system-B frame, in dB",
                       xlabel="Reference frame", ylabel="System B frame", origin="lower", colormap="viridis",
                       value_label="dB")
    print(f"MCD-DTW of the reference against itself: {mcd_dtw(c_ref, c_ref):.2f} dB.")

    # A simulated listening test: 24 raters by 30 items per system. Each rater has a personal offset.
    true_mos = {"reference": 4.4, "A": 3.9, "B": 3.1}
    raters = rng.normal(0, 0.4, (24, 1))
    items = rng.normal(0, 0.3, (1, 30))
    print("Mean opinion scores with 95% bootstrap intervals over raters and items:")
    for name, mos in true_mos.items():
        ratings = np.clip(np.round(mos + raters + items + rng.normal(0, 0.7, (24, 30))), 1, 5)
        mean, low, high = bootstrap_mos(ratings, seed=1)
        naive = 1.96 * ratings.std(ddof=1) / np.sqrt(ratings.size)
        print(f"  {name:9s} {mean:.2f} [{low:.2f}, {high:.2f}], against +/- {naive:.2f} if all 720 ratings "
              "were independent")

    wer = word_error_rate(["The birch canoe slid on the smooth planks."], ["the birch canoe slid on smooth planks"])
    print(f"WER of a transcript that drops one word of eight: {100 * wer:.1f}%")

    play(ref, RATE, label="Reference voice")
    play(sys_a, RATE, label="System A: 15% slower and 4% higher")
    play(sys_b, RATE, label="System B: flat pitch, blurred vowels, breathy")
    play(other, RATE, label="Another speaker, same vowels")
