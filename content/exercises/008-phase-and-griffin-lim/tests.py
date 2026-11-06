"""Checks for the phase and Griffin-Lim exercise."""

import numpy as np

from speechcheck import almost, chirp, check, close, equal, fail, finite, has, near_frequency, shape, tone
from speechlab import show_plot

RATE = 16000
N_FFT, HOP = 512, 128


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


def target(sub, seconds=0.512):
    x = chirp(300, 3000, RATE, seconds, amplitude=0.5)
    return x, np.abs(sub.stft(x, N_FFT, HOP))


def reference_griffin_lim(sub, S, n_iter, momentum, seed):
    rng = np.random.default_rng(seed)
    c = S * np.exp(2j * np.pi * rng.random(S.shape))
    previous = None
    for _ in range(n_iter):
        t = sub.stft(sub.istft(S * np.exp(1j * np.angle(c)), HOP), N_FFT, HOP)
        c = t if previous is None else t + momentum * (t - previous)
        previous = t
    return sub.istft(S * np.exp(1j * np.angle(c)), HOP)


@check("spectral_convergence is 0 for a match and 0.1 for a magnitude 10% too large")
def test_spectral_convergence(sub):
    sc = has(sub, "spectral_convergence")
    x, S = target(sub)
    X = sub.stft(x, N_FFT, HOP)
    almost(sc(X, S), 0.0, 1e-12, label="the signal's own STFT")
    almost(sc(1.1 * X, S), 0.1, 1e-9, label="1.1 times the STFT")
    almost(sc(np.abs(X) * 1.1, S), 0.1, 1e-9, label="a magnitude instead of a complex STFT")


@check("random_phase keeps the magnitude, and its result is not the STFT of any signal")
def test_random_phase(sub):
    random_phase = has(sub, "random_phase")
    sc = has(sub, "spectral_convergence")
    x, S = target(sub)
    Z = np.asarray(random_phase(S, seed=3))
    shape(Z, S.shape, label="random_phase")
    close(np.abs(Z), S, rtol=1e-12, atol=1e-12, label="|random_phase(S)|")
    rng = np.random.default_rng(3)
    cclose(Z, S * np.exp(2j * np.pi * rng.random(S.shape)), atol=1e-12, label="the phases drawn from default_rng(seed).random(S.shape)")
    true = sc(sub.stft(sub.istft(sub.stft(x, N_FFT, HOP), HOP), N_FFT, HOP), S)
    rough = sc(sub.stft(sub.istft(Z, HOP), N_FFT, HOP), S)
    if true > 1e-8 or rough < 0.3:
        fail(f"a real STFT should survive istft then stft (convergence {true:.2e}, expected below 1e-8); "
             f"a random-phase one should not (convergence {rough:.3f}, expected above 0.3)")


@check("griffin_lim returns hop * (frames - 1) samples, seeded, and n_iter=0 inverts random_phase")
def test_griffin_lim_shape(sub):
    griffin_lim = has(sub, "griffin_lim")
    random_phase = has(sub, "random_phase")
    _, S = target(sub)
    y = np.asarray(griffin_lim(S, N_FFT, HOP, n_iter=4))
    shape(y, (HOP * (S.shape[1] - 1),), label="griffin_lim output")
    finite(y, label="griffin_lim output")
    close(griffin_lim(S, N_FFT, HOP, n_iter=4), y, atol=0, label="a second run with the same seed")
    other = np.asarray(griffin_lim(S, N_FFT, HOP, n_iter=4, seed=1))
    if np.allclose(other, y):
        fail("seed=1 gave the same signal as seed=0; the seed should set the initial random phase")
    close(griffin_lim(S, N_FFT, HOP, n_iter=0, seed=5), sub.istft(random_phase(S, 5), HOP), atol=1e-12, label="zero iterations")


@check("griffin_lim matches the reference update, with and without momentum")
def test_griffin_lim_reference(sub):
    griffin_lim = has(sub, "griffin_lim")
    _, S = target(sub)
    for momentum in (0.0, 0.99):
        got = griffin_lim(S, N_FFT, HOP, n_iter=6, momentum=momentum, seed=2)
        want = reference_griffin_lim(sub, S, 6, momentum, 2)
        close(got, want, rtol=0, atol=1e-8, label=f"6 iterations at momentum {momentum}")


@check("Plain Griffin-Lim never gets worse from one iteration to the next")
def test_monotone(sub):
    griffin_lim = has(sub, "griffin_lim")
    sc = has(sub, "spectral_convergence")
    _, S = target(sub, seconds=0.384)
    history = [sc(sub.stft(griffin_lim(S, N_FFT, HOP, n_iter=n), N_FFT, HOP), S) for n in range(13)]
    rises = [n for n in range(12) if history[n + 1] > history[n] + 1e-9]
    if rises:
        show_plot(np.array(history), label="Spectral convergence by iteration", xlabel="Iterations", ylabel="Spectral convergence")
        fail(f"spectral convergence rose after iteration {rises[0]}: {history[rises[0]]:.4f} to {history[rises[0] + 1]:.4f}")


@check("Griffin-Lim converges, and momentum 0.99 gets further in the same 32 iterations")
def test_convergence(sub):
    griffin_lim = has(sub, "griffin_lim")
    sc = has(sub, "spectral_convergence")
    _, S = target(sub)
    plain50 = sc(sub.stft(griffin_lim(S, N_FFT, HOP, n_iter=50), N_FFT, HOP), S)
    if plain50 > 0.2:
        fail(f"after 50 iterations the spectral convergence is {plain50:.3f}, expected below 0.2")
    plain = sc(sub.stft(griffin_lim(S, N_FFT, HOP, n_iter=32), N_FFT, HOP), S)
    fast = sc(sub.stft(griffin_lim(S, N_FFT, HOP, n_iter=32, momentum=0.99), N_FFT, HOP), S)
    if fast > 0.75 * plain:
        fail(f"at 32 iterations, momentum 0.99 reaches {fast:.3f} and plain Griffin-Lim {plain:.3f}; expected momentum below 75% of plain")


@check("phase_vocoder with stretch 1 returns the STFT it was given")
def test_vocoder_identity(sub):
    phase_vocoder = has(sub, "phase_vocoder")
    x = chirp(200, 2000, RATE, 0.4)
    X = sub.stft(x, 1024, 256)
    Y = np.asarray(phase_vocoder(X, 1.0, 256))
    shape(Y, X.shape, label="phase_vocoder(X, 1.0)")
    cclose(Y, X, atol=1e-8, label="phase_vocoder(X, 1.0)")


@check("time_stretch changes a tone's length and keeps its pitch")
def test_time_stretch(sub):
    time_stretch = has(sub, "time_stretch")
    x = tone(440, RATE, 1.0, phase=np.pi / 2)
    for stretch in (1.5, 0.75):
        y = np.asarray(time_stretch(x, stretch, 1024, 256))
        want = stretch * len(x)
        if abs(len(y) - want) > 256:
            fail(f"stretch {stretch}: {len(y)} samples, expected {want:.0f} within one hop of 256")
        near_frequency(y[1024:-1024], RATE, 440, tol_hz=RATE / 1024, label=f"stretch {stretch}")
