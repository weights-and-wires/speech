"""The assertion vocabulary an exercise's `tests.py` writes against.

A check is a function registered with `@check`, taking the reader's module as its
only argument. It passes by returning and fails by raising `CheckFailure`, which
the helpers here raise on your behalf with a message the reader can act on.

The harness imports this module under the name `speechcheck` and reads `REGISTRY`.

Beside the general helpers it carries a few for audio: deterministic test
signals (`tone`, `chirp`, `white_noise`), a signal-to-noise comparison
(`snr_db`, `min_snr`), and a check on where a signal's energy sits in
frequency (`peak_frequency`, `near_frequency`). A check can also call the
`speechlab` output helpers, and whatever it draws appears in the console.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence


class CheckFailure(AssertionError):
    """Raised when a check's expectation is not met."""


@dataclass
class Check:
    description: str
    func: Callable[[Any], None]
    timings: dict[str, float] = field(default_factory=dict)


REGISTRY: list[Check] = []

# Populated by `timed`, drained by the harness after each check.
_TIMINGS: dict[str, float] = {}


def check(description: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Register a check. The description is the line the reader sees."""

    def decorate(func: Callable[[Any], None]) -> Callable[[Any], None]:
        REGISTRY.append(Check(description=description, func=func))
        return func

    return decorate


def fail(message: str) -> None:
    """Fail the current check with `message`."""
    raise CheckFailure(message)


def _numpy():
    import numpy as np

    return np


def _describe(value: Any) -> str:
    np = _numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return f"{array.item():.6g}"
    if array.size <= 8:
        return np.array2string(array, precision=6, suppress_small=True)
    flat = array.ravel()
    head = np.array2string(flat[:4], precision=6, suppress_small=True)
    tail = np.array2string(flat[-2:], precision=6, suppress_small=True)
    return f"shape {array.shape}, {head[:-1]} ... {tail[1:]}"


def _label(label: str | None) -> str:
    return f"{label}: " if label else ""


def close(
    got: Any,
    want: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    label: str | None = None,
) -> None:
    """Fail unless `got` and `want` agree elementwise within tolerance."""
    np = _numpy()
    got_a = np.asarray(got, dtype=float)
    want_a = np.asarray(want, dtype=float)
    if got_a.shape != want_a.shape:
        fail(
            f"{_label(label)}shape {got_a.shape} does not match the expected "
            f"{want_a.shape}"
        )
    if not np.all(np.isfinite(got_a)):
        fail(f"{_label(label)}result holds NaN or infinity: {_describe(got_a)}")
    if np.allclose(got_a, want_a, rtol=rtol, atol=atol):
        return
    diff = np.abs(got_a - want_a)
    worst = float(diff.max()) if diff.size else 0.0
    fail(
        f"{_label(label)}largest elementwise difference is {worst:.3g}, above the "
        f"tolerance rtol={rtol:g} atol={atol:g}.\n"
        f"  got:  {_describe(got_a)}\n"
        f"  want: {_describe(want_a)}"
    )


def almost(got: Any, want: Any, tol: float, label: str | None = None) -> None:
    """Fail unless two scalars agree within `tol`."""
    got_f = float(got)
    want_f = float(want)
    if abs(got_f - want_f) > tol:
        fail(
            f"{_label(label)}got {got_f:.6g}, expected {want_f:.6g} within {tol:g}"
        )


def equal(got: Any, want: Any, label: str | None = None) -> None:
    """Fail unless two values are exactly equal."""
    np = _numpy()
    got_a = np.asarray(got)
    want_a = np.asarray(want)
    if got_a.shape != want_a.shape or not np.array_equal(got_a, want_a):
        fail(
            f"{_label(label)}values differ.\n"
            f"  got:  {_describe(got_a)}\n"
            f"  want: {_describe(want_a)}"
        )


def shape(value: Any, want: Sequence[int], label: str | None = None) -> None:
    """Fail unless `value` has the given shape."""
    np = _numpy()
    got = tuple(np.asarray(value).shape)
    if got != tuple(want):
        fail(f"{_label(label)}shape is {got}, expected {tuple(want)}")


def finite(value: Any, label: str | None = None) -> None:
    """Fail when any element is NaN or infinite."""
    np = _numpy()
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        fail(f"{_label(label)}result holds NaN or infinity: {_describe(array)}")


def has(module: Any, name: str) -> Any:
    """Return `module.name`, failing with a readable message when it is missing."""
    if not hasattr(module, name):
        fail(f"your module defines no `{name}`")
    return getattr(module, name)


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Record how long a block takes, and report it beside the check."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _TIMINGS[label] = time.perf_counter() - start


def drain_timings() -> dict[str, float]:
    """Take and clear the timings recorded since the last call."""
    taken = dict(_TIMINGS)
    _TIMINGS.clear()
    return taken


def finite_difference(
    func: Callable[[Any], float], x: Any, eps: float = 1e-6
) -> Any:
    """Central-difference gradient of a scalar function, for checking gradients."""
    np = _numpy()
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        index = it.multi_index
        original = x[index]
        x[index] = original + eps
        high = float(func(x))
        x[index] = original - eps
        low = float(func(x))
        x[index] = original
        grad[index] = (high - low) / (2 * eps)
        it.iternext()
    return grad


# Audio ------------------------------------------------------------------


def tone(
    frequency: float,
    sample_rate: int,
    seconds: float,
    amplitude: float = 0.5,
    phase: float = 0.0,
) -> Any:
    """A sine wave, `amplitude * sin(2 pi f t + phase)`, as float64."""
    np = _numpy()
    t = np.arange(int(round(seconds * sample_rate))) / float(sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def chirp(
    f0: float,
    f1: float,
    sample_rate: int,
    seconds: float,
    amplitude: float = 0.5,
    method: str = "linear",
) -> Any:
    """A sweep from `f0` to `f1` Hz, linear or exponential (`method="log"`) in frequency."""
    np = _numpy()
    n = int(round(seconds * sample_rate))
    t = np.arange(n) / float(sample_rate)
    if method == "linear":
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * seconds))
    elif method == "log":
        if f0 <= 0 or f1 <= 0:
            raise ValueError("an exponential chirp needs positive frequencies")
        k = (f1 / f0) ** (1 / seconds)
        phase = 2 * np.pi * f0 * (k**t - 1) / np.log(k) if k != 1 else 2 * np.pi * f0 * t
    else:
        raise ValueError(f"method must be 'linear' or 'log', got {method!r}")
    return amplitude * np.sin(phase)


def white_noise(sample_rate: int, seconds: float, amplitude: float = 0.1, seed: int = 0) -> Any:
    """Gaussian noise with standard deviation `amplitude`, from a seeded generator."""
    np = _numpy()
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(int(round(seconds * sample_rate)))


def snr_db(got: Any, want: Any) -> float:
    """The signal-to-noise ratio of `got` against the reference `want`, in dB.

    The noise is `got - want`. Identical signals give infinity.
    """
    np = _numpy()
    got_a = np.asarray(got, dtype=float)
    want_a = np.asarray(want, dtype=float)
    if got_a.shape != want_a.shape:
        fail(f"shape {got_a.shape} does not match the reference {want_a.shape}")
    noise = float(np.sum((got_a - want_a) ** 2))
    signal = float(np.sum(want_a**2))
    if noise == 0.0:
        return float("inf")
    if signal == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(signal / noise))


def min_snr(got: Any, want: Any, db: float, label: str | None = None) -> None:
    """Fail unless `got` matches `want` with a signal-to-noise ratio of at least `db`."""
    finite(got, label)
    value = snr_db(got, want)
    if value < db:
        fail(
            f"{_label(label)}signal-to-noise ratio is {value:.1f} dB against the "
            f"reference, below the {db:g} dB this check needs.\n"
            f"  got:  {_describe(got)}\n"
            f"  want: {_describe(want)}"
        )


def peak_frequency(signal: Any, sample_rate: float) -> float:
    """The frequency in Hz of the largest bin in the signal's Hann-windowed spectrum."""
    np = _numpy()
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size < 2:
        fail(f"expected a 1-D signal, got shape {x.shape}")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.argmax(spectrum) * sample_rate / x.size)


def near_frequency(
    signal: Any, sample_rate: float, hz: float, tol_hz: float | None = None, label: str | None = None
) -> None:
    """Fail unless the signal's strongest frequency is within `tol_hz` of `hz`.

    The tolerance defaults to two FFT bins, `2 * sample_rate / len(signal)`.
    """
    np = _numpy()
    x = np.asarray(signal, dtype=float)
    finite(x, label)
    tol = tol_hz if tol_hz is not None else 2.0 * sample_rate / max(x.size, 1)
    found = peak_frequency(x, sample_rate)
    if abs(found - hz) > tol:
        fail(f"{_label(label)}the strongest frequency is {found:.1f} Hz, expected {hz:g} Hz within {tol:.1f} Hz")


def in_range(value: Any, low: float, high: float, label: str | None = None) -> None:
    """Fail unless every element lies in [low, high]."""
    np = _numpy()
    array = np.asarray(value, dtype=float)
    finite(array, label)
    if array.size and (array.min() < low or array.max() > high):
        fail(
            f"{_label(label)}values run from {array.min():.6g} to {array.max():.6g}, "
            f"outside [{low:g}, {high:g}]"
        )
