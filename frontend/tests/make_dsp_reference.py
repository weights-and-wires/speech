"""Writes the reference values that tests/dsp.test.ts checks src/demos/dsp.ts against.

It needs numpy, scipy, and librosa, which the app itself does not:

    python -m venv /tmp/ref && /tmp/ref/bin/pip install numpy scipy librosa
    /tmp/ref/bin/python frontend/tests/make_dsp_reference.py

Run it again only when the reference itself should change.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import scipy
import scipy.fft
import scipy.signal


def rounded(array):
    return [float(f"{v:.15g}") for v in np.asarray(array, dtype=float).ravel()]


rng = np.random.default_rng(0)
signal = rng.standard_normal(256)
fft_input = rng.standard_normal(16)
spectrum = np.fft.fft(fft_input)
dct_input = rng.standard_normal(13)
hz = [0.0, 60.0, 440.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0]

reference = {
    "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "librosa": librosa.__version__},
    "fft": {"input": rounded(fft_input), "re": rounded(spectrum.real), "im": rounded(spectrum.imag)},
    "hann16": rounded(scipy.signal.get_window("hann", 16)),
    "hamming16": rounded(scipy.signal.get_window("hamming", 16)),
    "hann16_symmetric": rounded(scipy.signal.windows.hann(16, sym=True)),
    "mel": {
        "hz": hz,
        "slaney": rounded(librosa.hz_to_mel(hz, htk=False)),
        "htk": rounded(librosa.hz_to_mel(hz, htk=True)),
    },
    "filterbank": {
        "sample_rate": 8000,
        "n_fft": 64,
        "n_mels": 8,
        "fmin": 50.0,
        "fmax": 3800.0,
        "slaney": rounded(librosa.filters.mel(sr=8000, n_fft=64, n_mels=8, fmin=50.0, fmax=3800.0, htk=False, norm="slaney")),
        "htk_unnormalised": rounded(librosa.filters.mel(sr=8000, n_fft=64, n_mels=8, fmin=50.0, fmax=3800.0, htk=True, norm=None)),
    },
    "stft": {
        "signal": rounded(signal),
        "n_fft": 64,
        "hop": 16,
        "magnitude": rounded(np.abs(librosa.stft(signal, n_fft=64, hop_length=16, window="hann", center=True, pad_mode="reflect"))),
        "shape": list(librosa.stft(signal, n_fft=64, hop_length=16, center=True, pad_mode="reflect").shape),
    },
    "dct": {
        "input": rounded(dct_input),
        "ortho": rounded(scipy.fft.dct(dct_input, type=2, norm="ortho")),
        "backward": rounded(scipy.fft.dct(dct_input, type=2)),
    },
}

out = Path(__file__).with_name("dsp-reference.json")
out.write_text(json.dumps(reference) + "\n")
print(f"Wrote {out}")
