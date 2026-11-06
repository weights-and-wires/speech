"""Companding and audio files: the worked solution."""

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
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((y + 1) / 2 * mu + 0.5).astype(np.int64)


def mu_law_decode(codes: np.ndarray, mu: int = 255) -> np.ndarray:
    """Map integer codes 0 to `mu` back to floats in [-1, 1] and undo the mu-law curve.

    The code maps to y = 2 * code / mu - 1, and the inverse curve is
    sign(y) * ((1 + mu) ** |y| - 1) / mu. Code 128 decodes to about 0.000086,
    not to 0, because 256 codes have no middle one.
    """
    y = 2 * np.asarray(codes, dtype=float) / mu - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    """Floats in [-1, 1] to 16-bit integers, dtype int16: times 32768, rounded, clipped to [-32768, 32767].

    Clip before the cast. A value of 1.0 would be 32768, which does not fit
    in an int16 and would wrap to -32768, a loud click.
    """
    return np.clip(np.round(np.asarray(x, dtype=float) * 32768), -32768, 32767).astype(np.int16)


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    """16-bit integers to float64 samples in [-1, 1): divided by 32768."""
    return np.asarray(pcm).astype(np.float64) / 32768.0


def wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """A 16-bit PCM WAV file, as bytes, holding `pcm`: shape (frames,) or (frames, channels).

    The file is the 12-byte RIFF header, a 16-byte `fmt ` chunk (format 1,
    channels, rate, bytes per second, bytes per frame, 16 bits), and a `data`
    chunk of interleaved little-endian samples. Everything is little-endian.
    The result matches what Python's `wave` module writes, byte for byte.
    """
    pcm = np.asarray(pcm, dtype=np.int16)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    data = pcm.astype("<i2").tobytes()  # row-major (frames, channels) is interleaved order
    block_align = channels * 2
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, sample_rate * block_align, block_align, 16)
    return (
        b"RIFF" + struct.pack("<I", 4 + (8 + len(fmt)) + (8 + len(data))) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(data)) + data
    )


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a 16-bit PCM WAV file into `(pcm, sample_rate)`.

    `pcm` is int16 with shape (frames,) for one channel and (frames, channels)
    for more. Walk the chunks from byte 12: each is a four-letter id, a
    four-byte little-endian size, and that many bytes, followed by one padding
    byte when the size is odd. Skip every chunk except `fmt ` and `data`.
    Raise ValueError for a file that is not RIFF WAVE, has no data chunk, or
    is not 16-bit integer PCM.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF WAVE file")
    fmt = None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif chunk_id == b"data":
            if fmt is None:
                raise ValueError("the data chunk comes before the fmt chunk")
            tag, channels, rate, _, _, bits = fmt
            if tag != 1 or bits != 16:
                raise ValueError(f"only 16-bit integer PCM is supported, got format {tag} with {bits} bits")
            pcm = np.frombuffer(body, dtype="<i2").astype(np.int16)
            if channels > 1:
                pcm = pcm.reshape(-1, channels)
            return pcm, rate
        pos += 8 + size + (size % 2)  # chunks are padded to an even length
    raise ValueError("no data chunk")


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
