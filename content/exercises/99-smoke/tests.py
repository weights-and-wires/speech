"""Checks for the smoke-test exercise."""

import numpy as np

from speechcheck import check, close, equal, fail, in_range, near_frequency, shape, tone
from speechlab import show_plot

RATE = 16000


@check("tone_sweep returns one second of samples within the amplitude")
def test_sweep_length(sub):
    x = sub.tone_sweep(200.0, 4000.0, RATE, 1.0, amplitude=0.5)
    shape(x, (RATE,), label="sweep")
    in_range(x, -0.5 - 1e-9, 0.5 + 1e-9, label="sweep")
    close(x[0], 0.0, atol=1e-12, label="the first sample")


@check("The sweep starts near f0 and ends near f1")
def test_sweep_frequency(sub):
    x = sub.tone_sweep(500.0, 3000.0, RATE, 1.0)
    head = x[: RATE // 20]
    tail = x[-RATE // 20 :]
    near_frequency(head, RATE, 500.0 + 2500.0 * 0.025, tol_hz=60.0, label="the first 50 ms")
    near_frequency(tail, RATE, 3000.0 - 2500.0 * 0.025, tol_hz=60.0, label="the last 50 ms")


@check("frame_signal keeps only whole frames, each a slice of the signal")
def test_frames(sub):
    x = np.arange(20, dtype=float)
    frames = sub.frame_signal(x, 8, 4)
    shape(frames, (4, 8), label="frames")
    equal(frames[1], x[4:12], label="frame 1")
    equal(frames[-1], x[12:20], label="the last frame")


@check("magnitude_spectrogram has one row per frequency bin and one column per frame")
def test_spectrogram_shape(sub):
    x = np.random.default_rng(0).standard_normal(4096)
    spec = sub.magnitude_spectrogram(x, 256, 64)
    shape(spec, (129, 1 + (4096 - 256) // 64), label="spectrogram")
    if np.any(spec < 0):
        fail("a magnitude spectrogram holds no negative values")


@check("A pure tone peaks in its own bin, with the Hann window's height")
def test_spectrogram_tone(sub):
    n_fft = 512
    bin_ = 40
    x = tone(bin_ * RATE / n_fft, RATE, 0.25, amplitude=1.0)
    spec = sub.magnitude_spectrogram(x, n_fft, 128)
    peaks = np.argmax(spec, axis=0)
    equal(peaks, np.full(spec.shape[1], bin_), label="the strongest bin in each frame")
    # A periodic Hann window sums to n_fft / 2, so a unit sine gives n_fft / 4.
    close(spec[bin_], np.full(spec.shape[1], n_fft / 4), rtol=1e-6, label="the peak's height")


@check("peak_track follows the sweep upwards")
def test_peak_track(sub):
    x = sub.tone_sweep(200.0, 4000.0, RATE, 1.0)
    n_fft, hop = 512, 128
    track = sub.peak_track(sub.magnitude_spectrogram(x, n_fft, hop), RATE, n_fft)
    centres = (np.arange(track.size) * hop + n_fft / 2) / RATE
    want = 200.0 + 3800.0 * centres
    show_plot((centres, track), (centres, want), labels=["yours", "expected"], label="The peak track", xlabel="Time (s)", ylabel="Hz")
    worst = float(np.max(np.abs(track - want)))
    if worst > 2 * RATE / n_fft:
        fail(f"the track strays {worst:.0f} Hz from the sweep, more than two bins")
