"""Quantization and audio files.

Fill in the functions below, then press Run to hear a note at 6 bits with and
without dither, compare 8-bit linear and mu-law quantizers, and write and read
a WAV file by hand. Press Check to test your code.

A float signal lies in [-1, 1]. A PCM signal is int16. A multichannel array
is shaped (frames, channels), the order the samples sit in a WAV file.
"""

import struct

import numpy as np

from speechlab import play, show_plot, show_spectrogram


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """Round `x` to a uniform grid of 2**bits levels, returned as floats.

    The step is `2 ** (1 - bits)`. Divide by the step, round to the nearest
    integer, clip the integer to [-2**(bits - 1), 2**(bits - 1) - 1], and
    multiply by the step again. The top level is one step below 1.0.
    """
    raise NotImplementedError("quantize")


def predicted_snr_db(bits: int, amplitude: float = 1.0) -> float:
    """The predicted SNR in dB of a sine of this peak amplitude quantized to `bits` bits.

    At full scale this is 10 log10(1.5 * 4**bits), about 6.02 * bits + 1.76.
    A sine `amplitude` times smaller loses 20 log10(amplitude) dB of it.
    """
    raise NotImplementedError("predicted_snr_db")


def tpdf_dither(x: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    """Add triangular (TPDF) dither to `x`, then call `quantize(..., bits)`.

    The dither for each sample is the sum of two independent draws from
    `rng.uniform(-step / 2, step / 2, x.shape)`, where step = 2 ** (1 - bits).
    """
    raise NotImplementedError("tpdf_dither")


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """Integer mu-law codes from 0 to `mu`, as an int64 array shaped like `x`.

    Clip `x` to [-1, 1], compand it with F(x) = sign(x) ln(1 + mu |x|) / ln(1 + mu),
    and map F from [-1, 1] to codes with floor((F + 1) / 2 * mu + 0.5).
    """
    raise NotImplementedError("mu_law_encode")


def mu_law_decode(codes: np.ndarray, mu: int = 255) -> np.ndarray:
    """Floats in [-1, 1] from mu-law codes: the inverse of `mu_law_encode`, up to rounding.

    Map each code back to y = 2 * code / mu - 1, then expand it with
    sign(y) ((1 + mu) ** |y| - 1) / mu.
    """
    raise NotImplementedError("mu_law_decode")


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    """Floats to int16: multiply by 32768, round, and clip to [-32768, 32767]."""
    raise NotImplementedError("float_to_pcm16")


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """int16 to float64: divide by 32768, so the result lies in [-1, 1)."""
    raise NotImplementedError("pcm16_to_float")


def wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """A complete 16-bit PCM WAV file, as bytes.

    `pcm` is int16, shaped (frames,) for mono or (frames, channels). Write the
    44-byte canonical header, every field little-endian, then the samples in
    interleaved order:

    - "RIFF", the file size minus 8 (uint32), "WAVE".
    - "fmt ", the chunk size 16 (uint32), then format tag 1 (uint16), channels
      (uint16), sample rate (uint32), bytes per second (uint32), bytes per
      frame (uint16), and bits per sample 16 (uint16).
    - "data", the number of sample bytes (uint32), and the samples.
    """
    raise NotImplementedError("wav_bytes")


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a 16-bit PCM WAV file into (pcm, sample_rate).

    Walk the chunks after the 12-byte RIFF header. Each chunk is a 4-byte id,
    a uint32 size, and that many bytes, plus one padding byte when the size is
    odd. Read "fmt ", skip any chunk you do not know, and return the samples
    of "data" as int16, shaped (frames,) for mono or (frames, channels).
    Raise ValueError for anything other than 16-bit integer PCM.
    """
    raise NotImplementedError("read_wav_bytes")


def _spectrogram(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """A quick magnitude spectrogram for the Run block. Lesson 04 builds a proper one."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    starts = np.arange(0, len(x) - n_fft + 1, hop)
    frames = x[starts[:, None] + np.arange(n_fft)[None, :]] * window
    return np.abs(np.fft.rfft(frames, axis=1)).T


if __name__ == "__main__":
    rate = 16000
    rng = np.random.default_rng(0)
    t = np.arange(int(2.5 * rate)) / rate

    # A plucked note: five harmonics of 220 Hz that die away, peaking at -12 dBFS.
    note = sum(np.sin(2 * np.pi * 220 * k * t) / k for k in range(1, 6)) * np.exp(-2.0 * t)
    note *= 0.25 / np.max(np.abs(note))
    play(note, rate, label="The original note")

    bits = 6
    plain = quantize(note, bits)
    dithered = tpdf_dither(note, bits, rng)
    for name, y in [("6 bits, no dither", plain), ("6 bits, TPDF dither", dithered)]:
        snr = 10 * np.log10(np.sum(note**2) / np.sum((y - note) ** 2))
        print(f"{name}: SNR {snr:.1f} dB")
    print(f"prediction for a full-scale sine at {bits} bits: {predicted_snr_db(bits):.2f} dB")
    play(plain, rate, label="6 bits, no dither: gritty, then it cuts out")
    play(dithered, rate, label="6 bits, TPDF dither: a steady hiss, and the note fades smoothly")
    show_spectrogram(_spectrogram(plain - note), rate, 256, label="Error without dither: harmonics of the note")
    show_spectrogram(_spectrogram(dithered - note), rate, 256, label="Error with dither: flat noise")

    # Eight bits two ways, on a quiet version of the note.
    quiet = note * 0.1
    linear = quantize(quiet, 8)
    companded = mu_law_decode(mu_law_encode(quiet))
    for name, y in [("8-bit linear", linear), ("8-bit mu-law", companded)]:
        print(f"{name} on the quiet note: SNR {10 * np.log10(np.sum(quiet**2) / np.sum((y - quiet) ** 2)):.1f} dB")
    play(linear * 10, rate, label="8-bit linear, quiet note, turned back up")
    play(companded * 10, rate, label="8-bit mu-law, quiet note, turned back up")

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
