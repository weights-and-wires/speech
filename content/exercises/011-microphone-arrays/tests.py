"""Checks for the microphone-array exercise.

A delay-and-sum that averages the rows in place, without advancing each
microphone by its delay, fails the click check and the SNR check. On the
click it returns a smear of height 1/3. On the noisy scene it gains about
0.9 dB instead of 10 log10(4) = 6.02 dB.

An MVDR weight that returns R^{-1} a without dividing by a^H R^{-1} a fails
the distortionless constraint: the two-microphone example would return
[1, 0.25] instead of [0.8, 0.2].
"""

import importlib.util
from pathlib import Path

import numpy as np

from speechcheck import almost, check, close, fail, finite, has, shape, snr_db

# Three-point parabolic interpolation of sinc(k - tau) is biased by at most
# 0.119 samples (the maximum over tau in [-0.5, 0.5]). The tolerance sits
# above that bias and below half a sample, so an integer-bin peak fails a
# planted delay of 2.5 samples, which is 0.5 samples from either integer.
GCC_TOL = 0.2

# A filter that never updates scores 0 dB, because the error is the echo.
# The reference canceler on this seed exceeds 50 dB on the second half.
# 20 dB is a factor of 100 in power, under the reference and far from 0 dB.
ERLE_DB = 20.0


def _plant_delay(x: np.ndarray, delay: float) -> np.ndarray:
    """Circular delay of `delay` samples, as a linear phase on the FFT."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * delay), len(x))


def _rooms():
    """`synthetic_rir` from the rooms lesson. The harness can see that file."""
    import speechcheck

    root = Path(speechcheck.__file__).resolve().parents[3]
    path = root / "content" / "exercises" / "010-rooms-and-reverb" / "solution.py"
    if not path.is_file():
        fail(f"expected the rooms lesson at {path}, and it is not there")
    spec = importlib.util.spec_from_file_location("rooms_lesson", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic_rir


@check("gcc_phat recovers a planted fractional delay")
def test_gcc_phat(sub):
    gcc_phat = has(sub, "gcc_phat")
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal(4096)
    for delay in (2.5, -3.5):
        x2 = _plant_delay(x1, delay)
        estimate = float(gcc_phat(x1, x2))
        close(estimate, delay, rtol=0, atol=GCC_TOL, label=f"a delay of {delay} samples")


@check("delay_and_sum lines up the worked click")
def test_delay_and_sum_click(sub):
    delay_and_sum = has(sub, "delay_and_sum")
    signals = np.zeros((3, 7))
    signals[0, 2] = 1.0
    signals[1, 3] = 1.0
    signals[2, 4] = 1.0
    got = np.asarray(delay_and_sum(signals, np.array([0.0, 1.0, 2.0])), dtype=float)
    shape(got, (7,), label="the aligned click")
    # Without the shifts the average is [0, 0, 1/3, 1/3, 1/3, 0, 0].
    close(got, [0, 0, 1, 0, 0, 0, 0], atol=1e-8, label="the click after shifts of 0, 1, and 2 samples")


@check("delay_and_sum raises the SNR by about 10 log10(M) on uncorrelated noise")
def test_delay_and_sum_snr(sub):
    delay_and_sum = has(sub, "delay_and_sum")
    rate = 16000
    n = 8000
    microphones = 4
    delays = np.array([0.0, 3.0, 6.0, 9.0])
    time = np.arange(n) / rate
    voice = sum(np.sin(2 * np.pi * freq * time) for freq in (200, 400, 800, 1600))
    voice = voice / np.sqrt(np.mean(voice**2))
    rng = np.random.default_rng(0)
    signals = np.vstack(
        [
            np.interp(np.arange(n) - delay, np.arange(n), voice, left=0.0, right=0.0)
            + rng.standard_normal(n)
            for delay in delays
        ]
    )
    valid = slice(0, n - int(delays.max()))
    reference = voice[valid]
    got = np.asarray(delay_and_sum(signals, delays), dtype=float)
    shape(got, (n,), label="the beamformed signal")
    finite(got, label="the beamformed signal")
    gain = snr_db(got[valid], reference) - snr_db(signals[0, valid], reference)
    # Sample variances of N i.i.d. Gaussian samples have relative standard
    # deviation sqrt(2/N). The gain is a ratio of two of them, so the relative
    # standard deviation is sqrt(4/N), which is (10/ln(10)) * sqrt(4/N) decibels.
    # The two estimates share one channel, so this slightly overstates the scatter.
    # Five of those standard deviations is the tolerance. The target is
    # 10 log10(M), not a measured array gain.
    n_valid = valid.stop - valid.start
    sigma_db = (10 / np.log(10)) * np.sqrt(4 / n_valid)
    tolerance = 5 * sigma_db
    want = 10 * np.log10(microphones)
    almost(gain, want, tolerance, label="SNR gain in dB")
    unshifted = snr_db(signals.mean(axis=0)[valid], reference) - snr_db(signals[0, valid], reference)
    if abs(unshifted - want) <= tolerance:
        fail(
            f"the no-shift average gained {unshifted:.3f} dB, expected well away from "
            f"{want:.3f} dB, so this scene no longer catches a missing shift"
        )


@check("mvdr_weights meets the distortionless constraint on the worked example")
def test_mvdr_hand(sub):
    mvdr_weights = has(sub, "mvdr_weights")
    covariance = np.diag([1.0, 4.0])
    steering = np.array([1.0, 1.0])
    weights = np.asarray(mvdr_weights(covariance, steering))
    shape(weights, (2,), label="the two weights")
    finite(weights, label="the two weights")
    close(weights, [0.8, 0.2], atol=1e-8, label="weights for diag(1, 4)")
    close(np.vdot(weights, steering), 1.0 + 0.0j, atol=1e-8, label="w conjugate-dot a")


@check("mvdr_weights beats delay-and-sum on a directional interferer")
def test_mvdr_interferer(sub):
    mvdr_weights = has(sub, "mvdr_weights")
    microphones = 4
    speed = 343.0
    spacing = 0.05
    frequency = 2000.0
    tau = np.arange(microphones) * spacing * np.sin(np.deg2rad(40.0)) / speed
    interferer = np.exp(-2j * np.pi * frequency * tau)
    steering = np.ones(microphones, dtype=complex)
    covariance = 0.1 * np.eye(microphones) + 10.0 * np.outer(interferer, interferer.conj())
    weights = np.asarray(mvdr_weights(covariance, steering), dtype=complex)
    shape(weights, (microphones,), label="the MVDR weights")
    close(np.vdot(weights, steering), 1.0 + 0.0j, atol=1e-6, label="distortionless constraint")
    power = float(np.real(np.vdot(weights, covariance @ weights)))
    delay_and_sum = steering / np.vdot(steering, steering)
    das_power = float(np.real(np.vdot(delay_and_sum, covariance @ delay_and_sum)))
    closed = 1.0 / float(np.real(np.vdot(steering, np.linalg.solve(covariance, steering))))
    if not power < 0.5 * das_power:
        fail(
            f"output noise power is {power:.4f}, expected under half of "
            f"delay-and-sum's {das_power:.4f}"
        )
    close(power, closed, rtol=1e-5, atol=1e-6, label="MVDR output noise power")


@check("nlms_echo_cancel matches the one-tap update")
def test_nlms_hand(sub):
    nlms_echo_cancel = has(sub, "nlms_echo_cancel")
    error = np.asarray(
        nlms_echo_cancel(np.array([2.0, 2.0]), np.array([1.0, 1.0]), 1, 1.0, 0.0),
        dtype=float,
    )
    shape(error, (2,), label="the one-tap error")
    # Without the division by the frame energy the second error is -3, not 0.
    close(error, [1.0, 0.0], atol=1e-8, label="error for far [2, 2] and mic [1, 1]")


@check("nlms_echo_cancel reaches the ERLE threshold on a synthetic room")
def test_nlms_erle(sub):
    nlms_echo_cancel = has(sub, "nlms_echo_cancel")
    synthetic_rir = _rooms()
    rate = 16000
    response = synthetic_rir(0.2, rate, seconds=0.02, seed=0)
    rng = np.random.default_rng(1)
    far = rng.standard_normal(8000)
    echo = np.convolve(far, response)[: far.shape[0]]
    error = np.asarray(
        nlms_echo_cancel(far, echo, len(response), 0.5, 1e-8),
        dtype=float,
    )
    shape(error, far.shape, label="the residual")
    finite(error, label="the residual")
    half = slice(far.shape[0] // 2, None)
    echo_power = float(np.sum(echo[half] ** 2))
    residual_power = float(np.sum(error[half] ** 2))
    if residual_power == 0.0:
        return
    erle = 10 * np.log10(echo_power / residual_power)
    if erle < ERLE_DB:
        fail(
            f"ERLE on the second half is {erle:.2f} dB, expected at least {ERLE_DB:.0f} dB "
            f"(echo power {echo_power:.4g}, residual power {residual_power:.4g})"
        )
