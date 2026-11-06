"""Companding and audio files.

Fill in the six functions below, then press Run to compare 8-bit linear and
mu-law on a quiet note, and to write, read, and play a stereo WAV file. Press
Check to test your code.

`quantize` is provided, because you wrote it in the chapter "Quantization and
dither". Samples are floats in [-1, 1]. A mu-law code is an integer from 0 to
255, and a 16-bit PCM sample is an int16.
"""

import struct

import numpy as np

from speechlab import play, show_plot, show_spectrogram


# Provided: you wrote this in the chapter "Quantization and dither".
def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Round `x` to the nearest of 2**bits levels, a step of 2**(1 - bits) apart."""
    step = 2.0 ** (1 - bits)
    codes = np.clip(np.round(np.asarray(x, dtype=float) / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return codes * step


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Compand `x` with the mu-law curve and map it to integer codes 0 to `mu`, as int64.

    First clip `x` to [-1, 1]. Then F(x) = sign(x) * log(1 + mu |x|) / log(1 + mu),
    which lies in [-1, 1], and the code is floor((F + 1) / 2 * mu + 0.5). So
    0.01 gives code 157, 1.0 gives 255, -1.0 gives 0, and 0.0 gives 128.
    """
    raise NotImplementedError("mu_law_encode")


def mu_law_decode(codes: np.ndarray, mu: int = 255) -> np.ndarray:
    """Map integer codes 0 to `mu` back to floats in [-1, 1] and undo the mu-law curve.

    The code maps to y = 2 * code / mu - 1, and the inverse curve is
    sign(y) * ((1 + mu) ** |y| - 1) / mu. Code 128 decodes to about 0.000086,
    not to 0, because 256 codes have no middle one.
    """
    raise NotImplementedError("mu_law_decode")


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    """Floats in [-1, 1] to 16-bit integers, dtype int16: times 32768, rounded, clipped to [-32768, 32767].

    Clip before the cast. A value of 1.0 would be 32768, which does not fit
    in an int16 and would wrap to -32768, a loud click.
    """
    raise NotImplementedError("float_to_pcm16")


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """16-bit integers to float64 samples in [-1, 1): divided by 32768."""
    raise NotImplementedError("pcm16_to_float")


def wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """A 16-bit PCM WAV file, as bytes, holding `pcm`: shape (frames,) or (frames, channels).

    The file is the 12-byte RIFF header, a 16-byte `fmt ` chunk (format 1,
    channels, rate, bytes per second, bytes per frame, 16 bits), and a `data`
    chunk of interleaved little-endian samples. Everything is little-endian.
    The result matches what Python's `wave` module writes, byte for byte.
    """
    raise NotImplementedError("wav_bytes")


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a 16-bit PCM WAV file into `(pcm, sample_rate)`.

    `pcm` is int16 with shape (frames,) for one channel and (frames, channels)
    for more. Walk the chunks from byte 12: each is a four-letter id, a
    four-byte little-endian size, and that many bytes, followed by one padding
    byte when the size is odd. Skip every chunk except `fmt ` and `data`.
    Raise ValueError for a file that is not RIFF WAVE, has no data chunk, or
    is not 16-bit integer PCM.
    """
    raise NotImplementedError("read_wav_bytes")


if __name__ == "__main__":
    rate = 16000
    t = np.arange(int(2.5 * rate)) / rate

    # A plucked note: five harmonics of 440 Hz that die away, peaking at -12 dBFS.
    note = sum(np.sin(2 * np.pi * 440 * k * t) / k for k in range(1, 6)) * np.exp(-2.0 * t)
    note *= 0.25 / np.max(np.abs(note))
    play(note, rate, label="The original note")

    # Eight bits two ways, on a quiet version of the note.
    quiet = note * 0.1
    linear = quantize(quiet, 8)
    companded = mu_law_decode(mu_law_encode(quiet))
    for name, y in [("8-bit linear", linear), ("8-bit mu-law", companded)]:
        print(f"{name} on the quiet note: SNR {10 * np.log10(np.sum(quiet**2) / np.sum((y - quiet) ** 2)):.1f} dB")
    play(linear * 10, rate, label="8-bit linear, quiet note, turned back up: grainy")
    play(companded * 10, rate, label="8-bit mu-law, quiet note, turned back up: cleaner")

    x = np.linspace(-1, 1, 801)
    show_plot((x, mu_law_decode(mu_law_encode(x))), (x, quantize(x, 8)), labels=["mu-law", "linear"], label="Two 8-bit quantizers", xlabel="Input", ylabel="Output")

    # A stereo WAV file, built by hand and read back.
    stereo = float_to_pcm16(np.stack([note, 0.5 * note[::-1]], axis=1))
    blob = wav_bytes(stereo, rate)
    back, back_rate = read_wav_bytes(blob)
    tag, channels, header_rate, byte_rate, align, width = struct.unpack("<HHIIHH", blob[20:36])
    print(f"WAV: {len(blob)} bytes, format {tag}, {channels} channels, {header_rate} Hz, {byte_rate} bytes/s, {width} bits")
    print(f"read back {back.shape} at {back_rate} Hz; identical: {np.array_equal(back, stereo)}; rewritten bytes identical: {wav_bytes(back, back_rate) == blob}")
    play(pcm16_to_float(back).T, back_rate, label="The stereo file, read back")
