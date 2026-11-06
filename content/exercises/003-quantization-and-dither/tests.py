"""Checks for the quantization and dither exercise."""

import numpy as np

from speechcheck import almost, check, close, fail, has, snr_db

RATE = 16000


def _busy_sine(amplitude=0.99, seconds=1.0):
    # 1234.5 Hz shares no short period with 16 kHz, so the rounding error looks like noise.
    n = np.arange(int(seconds * RATE))
    return amplitude * np.sin(2 * np.pi * 1234.5 * n / RATE + 0.1)


@check("quantize rounds to the nearest of 2**bits levels")
def test_quantize_small(sub):
    quantize = has(sub, "quantize")
    x = np.array([0.3, -0.62, 0.9, 0.05, 0.99, -1.0])
    close(quantize(x, 3), [0.25, -0.5, 0.75, 0.0, 0.75, -1.0], atol=1e-12, label="3-bit quantization")
    y = quantize(np.linspace(-1, 1, 5001), 4)
    levels = np.unique(np.round(y, 12))
    if levels.size != 16:
        fail(f"4 bits should give 16 distinct levels, got {levels.size}: {levels}")
    close(levels, np.arange(-8, 8) / 8, atol=1e-12, label="the 4-bit levels")


@check("quantize clips to the top code, one step below +1, and never goes past -1")
def test_quantize_clips(sub):
    quantize = has(sub, "quantize")
    for bits in (3, 8, 16):
        step = 2.0 ** (1 - bits)
        y = np.asarray(quantize(np.array([1.0, 1.5, 100.0, -1.0, -3.0]), bits))
        close(y, [1 - step, 1 - step, 1 - step, -1.0, -1.0], atol=1e-12, label=f"clipped values at {bits} bits")


@check("predicted_snr_db is 6.02 N + 1.76 dB for a full-scale sine")
def test_predicted(sub):
    predicted = has(sub, "predicted_snr_db")
    for bits, want in [(8, 49.9257), (16, 98.0905), (24, 146.2553)]:
        almost(predicted(bits), want, 1e-3, label=f"{bits} bits")
    almost(predicted(16, 0.1), 78.0905, 1e-3, label="16 bits, a sine 20 dB below full scale")


@check("The measured SNR of a quantized sine matches the prediction at 8, 12, and 16 bits")
def test_measured_snr(sub):
    quantize = has(sub, "quantize")
    predicted = has(sub, "predicted_snr_db")
    x = _busy_sine()
    for bits in (8, 12, 16):
        measured = snr_db(quantize(x, bits), x)
        want = predicted(bits, 0.99)
        if abs(measured - want) > 0.25:
            fail(f"at {bits} bits the measured SNR is {measured:.2f} dB, the prediction {want:.2f} dB; expected them within 0.25 dB")


@check("A quiet sine loses exactly its level in dB of SNR")
def test_quiet(sub):
    quantize = has(sub, "quantize")
    predicted = has(sub, "predicted_snr_db")
    loud = snr_db(quantize(_busy_sine(0.99), 12), _busy_sine(0.99))
    quiet = snr_db(quantize(_busy_sine(0.099), 12), _busy_sine(0.099))
    if abs((loud - quiet) - 20.0) > 0.5:
        fail(f"a sine 20 dB quieter should lose 20 dB of SNR at 12 bits, but it lost {loud - quiet:.2f} dB ({loud:.2f} dB down to {quiet:.2f} dB)")
    almost(predicted(12, 0.099), quiet, 0.5, label="predicted_snr_db for the quiet sine")


@check("TPDF dither makes the average output follow a signal smaller than one step")
def test_dither_mean(sub):
    quantize = has(sub, "quantize")
    dither = has(sub, "tpdf_dither")
    bits = 8
    step = 2.0 ** (1 - bits)
    x = np.full(200_000, 0.3 * step)
    plain = float(np.mean(quantize(x, bits)))
    if abs(plain) > 1e-12:
        fail(f"without dither, 0.3 of a step should round to 0, got a mean of {plain / step:.3f} steps")
    mean = float(np.mean(dither(x, bits, np.random.default_rng(0))))
    if abs(mean / step - 0.3) > 0.01:
        fail(f"with TPDF dither the mean output should be 0.300 steps, got {mean / step:.3f}")


@check("TPDF dither gives the same noise power, one step squared over 4, at every input level")
def test_dither_power(sub):
    dither = has(sub, "tpdf_dither")
    bits = 8
    step = 2.0 ** (1 - bits)
    rng = np.random.default_rng(1)
    for level in (0.0, 0.25, 0.5):
        x = np.full(200_000, level * step)
        err = dither(x, bits, rng) - x
        ratio = float(np.mean(err**2) / (step**2 / 4))
        if abs(ratio - 1) > 0.05:
            fail(f"at {level} of a step the error power is {ratio:.3f} times step**2 / 4, expected 1 within 0.05")


@check("tpdf_dither returns values on the quantizer grid, and the same shape it was given")
def test_dither_grid(sub):
    dither = has(sub, "tpdf_dither")
    bits = 5
    step = 2.0 ** (1 - bits)
    x = _busy_sine(0.5, 0.1)
    y = np.asarray(dither(x, bits, np.random.default_rng(2)))
    if y.shape != x.shape:
        fail(f"tpdf_dither returned shape {y.shape}, expected {x.shape}")
    codes = y / step
    if np.max(np.abs(codes - np.round(codes))) > 1e-9:
        fail("the dithered output is not a whole number of steps; call quantize after you add the noise")
    if y.max() > 1 - step + 1e-12 or y.min() < -1 - 1e-12:
        fail(f"the dithered output spans [{y.min():.3f}, {y.max():.3f}], outside the 5-bit range [-1, {1 - step:.4f}]")
