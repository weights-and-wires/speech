"""Microphone arrays: delay-and-sum, GCC-PHAT, MVDR, and an NLMS echo canceler.

Fill in the four functions, then press Run.
"""

import importlib.util
from pathlib import Path

import numpy as np

from speechlab import play, show_waveform


def gcc_phat(x1: np.ndarray, x2: np.ndarray) -> float:
    """Delay of `x2` behind `x1`, in samples.

    Both arrays are 1-D and the same length. A positive return means `x2`
    is a later copy of `x1`.
    """
    raise NotImplementedError("gcc_phat")


def delay_and_sum(signals: np.ndarray, delays: np.ndarray) -> np.ndarray:
    """Average `signals` after advancing each row by its delay.

    `signals` has shape `(M, N)` and `delays` has shape `(M,)`, in samples.
    A positive delay means that row arrived later. Samples that fall outside
    a row are zeros. Returns shape `(N,)`.
    """
    raise NotImplementedError("delay_and_sum")


def mvdr_weights(noise_cov: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """MVDR weights for one frequency bin.

    `noise_cov` has shape `(M, M)` and `steering` has shape `(M,)`.
    Returns `w` of shape `(M,)` with `conj(w)` dot `steering` equal to 1,
    and with the smallest noise power `conj(w)` dot `noise_cov` dot `w`
    that this constraint allows.
    """
    raise NotImplementedError("mvdr_weights")


def nlms_echo_cancel(
    far: np.ndarray,
    mic: np.ndarray,
    n_taps: int,
    mu: float,
    delta: float,
) -> np.ndarray:
    """Subtract an NLMS estimate of the echo in `mic` from the loudspeaker `far`.

    Both signals are 1-D with the same length `N`. The tap vector at time `n`
    is `far[n], far[n-1], ...`, and `far[k] = 0` for `k < 0`. Returns the
    error signal, shape `(N,)`.
    """
    raise NotImplementedError("nlms_echo_cancel")


def _half(samples: np.ndarray) -> np.ndarray:
    """Scale a clip to a peak of 0.5 so playback does not clip."""
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return samples
    return 0.5 * samples / peak


def _room():
    """The synthetic room from the rooms lesson."""
    import speechlab

    root = Path(speechlab.__file__).resolve().parents[3]
    path = root / "content" / "exercises" / "010-rooms-and-reverb" / "solution.py"
    spec = importlib.util.spec_from_file_location("rooms_lesson", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.synthetic_rir


if __name__ == "__main__":
    rate = 16000
    n = rate // 2
    rng = np.random.default_rng(0)
    voice = np.zeros(n)
    # Broadband bursts, so GCC-PHAT has energy in every bin. A pure tone does not.
    for start in range(0, n, 2000):
        burst = rng.standard_normal(200) * np.hanning(200)
        voice[start : start + 200] = burst
    delays = np.array([0.0, 3.0, 6.0, 9.0])
    mics = np.vstack(
        [
            np.interp(np.arange(n) - delay, np.arange(n), voice, left=0.0, right=0.0)
            + 0.05 * rng.standard_normal(n)
            for delay in delays
        ]
    )
    print(f"GCC-PHAT on microphones 0 and 1: {gcc_phat(mics[0], mics[1]):.3f} samples, planted 3")
    beam = delay_and_sum(mics, delays)
    play(_half(mics[0]), rate, label="One microphone, voice plus uncorrelated noise")
    show_waveform(_half(mics[0]), rate, label="One microphone")
    play(_half(beam), rate, label="Delay-and-sum of four microphones")
    show_waveform(_half(beam), rate, label="Delay-and-sum")

    room = _room()(0.2, rate, seconds=0.02, seed=0)
    far = 0.2 * rng.standard_normal(n)
    echo = np.convolve(far, room)[:n]
    cancelled = nlms_echo_cancel(far, echo, len(room), 0.5, 1e-8)
    half = slice(n // 2, None)
    erle = 10 * np.log10(np.sum(echo[half] ** 2) / np.sum(cancelled[half] ** 2))
    print(f"ERLE on the second half of a {len(room)}-tap room: {erle:.1f} dB")
    play(_half(echo), rate, label="Microphone: the loudspeaker through the room")
    play(_half(cancelled), rate, label="After NLMS")
