"""Checks for the short-time Fourier transform exercise."""

import numpy as np
import scipy.signal
import torch

from speechcheck import check, close, equal, fail, has, shape, tone


def cclose(got, want, atol=1e-9, label=None):
    """`close` for complex arrays: it compares real parts only, so compare both parts."""
    got = np.asarray(got, dtype=complex)
    want = np.asarray(want, dtype=complex)
    close(got.real, want.real, rtol=0, atol=atol, label=f"{label} (real part)")
    close(got.imag, want.imag, rtol=0, atol=atol, label=f"{label} (imaginary part)")


def torch_stft(x, n_fft, hop, center=True):
    window = torch.hann_window(n_fft, dtype=torch.float64)
    return torch.stft(torch.from_numpy(np.asarray(x, dtype=float)), n_fft, hop, window=window,
                      center=center, pad_mode="reflect", return_complex=True).numpy()


@check("frame_count gives 101 frames centered and 98 uncentered for 1 s at 400/160")
def test_frame_count(sub):
    frame_count = has(sub, "frame_count")
    equal(frame_count(16000, 400, 160, True), 101, label="1 s at 16 kHz, centered")
    equal(frame_count(16000, 400, 160, False), 98, label="1 s at 16 kHz, uncentered")
    equal(frame_count(480000, 400, 160, True), 3001, label="Whisper's 30 s window, centered")
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = int(rng.integers(600, 5000))
        for center in (True, False):
            want = torch_stft(np.zeros(n), 512, 128, center).shape[1]
            equal(frame_count(n, 512, 128, center), want, label=f"{n} samples, n_fft 512, hop 128, center={center}, against torch.stft")


@check("frame_signal cuts whole frames uncentered, and reflects half a frame at each end centered")
def test_frames(sub):
    frame_signal = has(sub, "frame_signal")
    x = np.arange(20, dtype=float)
    frames = np.asarray(frame_signal(x, 8, 4, center=False))
    shape(frames, (4, 8), label="20 samples in frames of 8, hop 4, uncentered")
    equal(frames[1], x[4:12], label="frame 1")
    equal(frames[3], x[12:20], label="the last frame")
    frames = np.asarray(frame_signal(np.arange(10, dtype=float), 8, 4))
    want = np.array([[4, 3, 2, 1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 8, 7]], dtype=float)
    equal(frames, want, label="the centered frames of 0..9 with frame 8, hop 4")


@check("stft matches torch.stft with a periodic Hann window, centered, reflect-padded")
def test_stft_torch(sub):
    stft = has(sub, "stft")
    x = np.random.default_rng(1).standard_normal(3001)
    for n_fft, hop in [(400, 160), (512, 128), (256, 100), (8, 4)]:
        X = np.asarray(stft(x, n_fft, hop))
        want = torch_stft(x, n_fft, hop)
        shape(X, want.shape, label=f"stft with n_fft={n_fft}, hop={hop}: (bins, frames)")
        cclose(X, want, atol=1e-9, label=f"stft with n_fft={n_fft}, hop={hop}")


@check("Frame m is centered on sample m * hop, and row k is k * fs / n_fft Hz")
def test_stft_layout(sub):
    stft = has(sub, "stft")
    x = np.zeros(4000)
    x[800] = 1.0
    energy = np.sum(np.abs(np.asarray(stft(x, 400, 160))) ** 2, axis=0)
    if int(np.argmax(energy)) != 5:
        fail(f"an impulse at sample 800 with hop 160 should be strongest in frame 5, found frame {int(np.argmax(energy))}")
    rate = 16000
    X = np.abs(np.asarray(stft(tone(40 * rate / 512, rate, 0.5, amplitude=1.0), 512, 128)))
    peaks = np.unique(np.argmax(X[:, 2:-2], axis=0))
    equal(peaks, [40], label="the strongest row for a tone at 40 * fs / 512 Hz")


@check("spectrogram gives magnitude at power 1 and power at power 2")
def test_spectrogram(sub):
    spectrogram = has(sub, "spectrogram")
    x = np.random.default_rng(2).standard_normal(2000)
    magnitude = np.abs(torch_stft(x, 256, 64))
    close(spectrogram(x, 256, 64, power=1.0), magnitude, rtol=1e-9, atol=1e-9, label="power=1")
    close(spectrogram(x, 256, 64), magnitude**2, rtol=1e-9, atol=1e-9, label="the default, power=2")


@check("is_cola agrees with scipy.signal.check_COLA")
def test_cola(sub):
    is_cola = has(sub, "is_cola")
    hann = scipy.signal.get_window("hann", 400)
    cases = [
        ("periodic Hann 400", hann, 200), ("periodic Hann 400", hann, 100),
        ("periodic Hann 400", hann, 150), ("periodic Hann 400", hann, 160),
        ("symmetric Hann 400", np.hanning(400), 200),
        ("periodic Hamming 400", scipy.signal.get_window("hamming", 400), 200),
        ("rectangular 256", np.ones(256), 256), ("periodic Hann 512", scipy.signal.get_window("hann", 512), 384),
    ]
    for name, w, hop in cases:
        want = bool(scipy.signal.check_COLA(w, len(w), len(w) - hop))
        got = bool(is_cola(w, hop))
        if got != want:
            fail(f"{name} at hop {hop}: is_cola says {got}, scipy.signal.check_COLA says {want}")


@check("istft reconstructs a signal from its STFT to 1e-10")
def test_istft_roundtrip(sub):
    stft = has(sub, "stft")
    istft = has(sub, "istft")
    x = np.random.default_rng(3).standard_normal(3001)
    for n_fft, hop in [(512, 128), (400, 100), (400, 160)]:
        y = np.asarray(istft(np.asarray(stft(x, n_fft, hop)), hop, length=len(x)))
        shape(y, x.shape, label=f"istft with length={len(x)}")
        close(y, x, rtol=0, atol=1e-10, label=f"reconstruction with n_fft={n_fft}, hop={hop}")
    X = np.asarray(stft(x, 512, 128))
    y = np.asarray(istft(X, 128))
    equal(len(y), 128 * (X.shape[1] - 1), label="the default length, hop * (frames - 1)")


@check("istft is the least-squares inverse: it matches torch.istft on an STFT with scrambled phase")
def test_istft_least_squares(sub):
    istft = has(sub, "istft")
    rng = np.random.default_rng(4)
    X = torch_stft(rng.standard_normal(3001), 512, 128)
    Z = np.abs(X) * np.exp(2j * np.pi * rng.random(X.shape))  # not the STFT of any signal
    got = np.asarray(istft(Z, 128, length=3001))
    want = torch.istft(torch.from_numpy(Z), 512, 128, window=torch.hann_window(512, dtype=torch.float64),
                       center=True, length=3001).numpy()
    close(got, want, rtol=0, atol=1e-9, label="istft of an inconsistent STFT")


@check("istft raises ValueError when the window and hop leave samples uncovered")
def test_istft_nola(sub):
    stft = has(sub, "stft")
    istft = has(sub, "istft")
    x = np.random.default_rng(5).standard_normal(4096)
    X = np.asarray(stft(x, 512, 512))  # a periodic Hann is 0 at its first sample, so hop 512 leaves gaps
    try:
        istft(X, 512)
    except ValueError:
        return
    fail("istft with a 512-point periodic Hann window and hop 512 should raise ValueError: every 512th sample has zero window weight")
