"""Checks for the what-sound-is exercise."""

import numpy as np

from speechcheck import almost, check, close, fail, has, shape

RATE = 16000


@check("rms is the square root of the mean square")
def test_rms(sub):
    rms = has(sub, "rms")
    almost(rms(np.array([0.5, -0.5, 0.5, -0.5])), 0.5, 1e-12, label="rms of the buzz [0.5, -0.5, 0.5, -0.5]")
    almost(rms(np.array([0.5, 0.0, 0.0, 0.0])), 0.25, 1e-12, label="rms of the click [0.5, 0, 0, 0]")
    almost(rms(np.array([3.0, 4.0])), np.sqrt(12.5), 1e-12, label="rms of [3, 4]")


@check("rms of a sine over whole cycles is A divided by the square root of 2, whatever the length")
def test_rms_sine(sub):
    rms = has(sub, "rms")
    for seconds in (0.5, 1.0, 2.0):
        n = np.arange(int(seconds * RATE))
        sine = 0.8 * np.sin(2 * np.pi * 440 * n / RATE)
        got = rms(sine)
        if abs(got - 0.8 / np.sqrt(2)) > 1e-3:
            fail(f"rms of a 0.8-amplitude sine lasting {seconds} s is {got:.4f}, expected {0.8 / np.sqrt(2):.4f}; the RMS must not grow with the length of the clip")


@check("peak_dbfs reads 0 dBFS at full scale and minus 6.02 dBFS at half")
def test_peak_dbfs(sub):
    peak_dbfs = has(sub, "peak_dbfs")
    n = np.arange(RATE)
    full = np.sin(2 * np.pi * 1000 * n / RATE + np.pi / 2)
    almost(peak_dbfs(full), 0.0, 1e-9, label="peak of a full-scale sine")
    almost(peak_dbfs(0.5 * full), -6.0206, 1e-3, label="peak of a 0.5 sine")
    almost(peak_dbfs(0.25 * full), -12.0412, 1e-3, label="peak of a 0.25 sine")
    almost(peak_dbfs(np.array([-0.1, 0.05])), -20.0, 1e-9, label="peak of [-0.1, 0.05], which has the sign dropped")


@check("rms_dbfs reads minus 3.01 dBFS for a full-scale sine and minus 9.03 dBFS for a half-scale one")
def test_rms_dbfs(sub):
    rms_dbfs = has(sub, "rms_dbfs")
    n = np.arange(RATE)
    full = np.sin(2 * np.pi * 1000 * n / RATE + np.pi / 2)
    almost(rms_dbfs(full), -3.0103, 1e-3, label="RMS level of a full-scale sine")
    almost(rms_dbfs(0.5 * full), -9.0309, 1e-3, label="RMS level of a 0.5 sine")
    almost(rms_dbfs(np.array([0.5, 0.0, 0.0, 0.0])), -12.0412, 1e-3, label="RMS level of the click")


@check("Halving the amplitude lowers the level by 6.02 dB, and silence stays finite")
def test_halving_and_silence(sub):
    peak_dbfs = has(sub, "peak_dbfs")
    rms_dbfs = has(sub, "rms_dbfs")
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(4000)
    for name, level in [("peak_dbfs", peak_dbfs), ("rms_dbfs", rms_dbfs)]:
        drop = level(x) - level(0.5 * x)
        almost(drop, 6.0206, 1e-3, label=f"{name} of a clip minus {name} of the same clip at half the amplitude")
    for name, level in [("peak_dbfs", peak_dbfs), ("rms_dbfs", rms_dbfs)]:
        silent = level(np.zeros(100))
        if not np.isfinite(silent):
            fail(f"{name} of silence returned {silent}; floor the value at eps before the log")
        almost(silent, -200.0, 1e-6, label=f"{name} of silence with eps=1e-10")


@check("sample_tone takes A cos(2 pi f n / fs + phase) at every sample")
def test_sample_tone(sub):
    sample_tone = has(sub, "sample_tone")
    x = sample_tone(440.0, RATE, 0.5, amplitude=0.25, phase=0.3)
    shape(x, (8000,), label="half a second at 16 kHz")
    n = np.arange(8000)
    close(x, 0.25 * np.cos(2 * np.pi * 440.0 * n / RATE + 0.3), atol=1e-12, label="the samples")
    y = sample_tone(440.0, RATE, 1.0, amplitude=0.5)
    close(y[:5], [0.5, 0.4926, 0.4704, 0.4343, 0.3853], atol=5e-5, label="the first five samples of the running example")
    almost(float(sample_tone(100.0, 8000, 0.01)[0]), 1.0, 1e-12, label="the first sample of a cosine with phase 0")


@check("The running example, a 440 Hz tone at amplitude 0.5, measures minus 6.02 dBFS peak and minus 9.03 dBFS RMS")
def test_running_example(sub):
    sample_tone = has(sub, "sample_tone")
    peak_dbfs = has(sub, "peak_dbfs")
    rms_dbfs = has(sub, "rms_dbfs")
    a440 = sample_tone(440.0, RATE, 1.0, amplitude=0.5)
    almost(peak_dbfs(a440), -6.0206, 0.01, label="peak level of the running example")
    almost(rms_dbfs(a440), -9.0309, 0.01, label="RMS level of the running example")
    almost(peak_dbfs(a440) - rms_dbfs(a440), 3.0103, 0.01, label="the gap between peak and RMS level of a sine")
