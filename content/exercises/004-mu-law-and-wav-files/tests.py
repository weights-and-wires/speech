"""Checks for the mu-law and WAV files exercise."""

import io
import struct
import wave

import numpy as np
import soundfile

from speechcheck import almost, check, close, equal, fail, has, snr_db


@check("mu_law_encode gives the 8-bit codes WaveNet used for known inputs")
def test_mu_law_codes(sub):
    encode = has(sub, "mu_law_encode")
    x = np.array([0.001, 0.01, 0.1, 0.5, 1.0, -0.5, -1.0, 0.0, 1.5])
    got = np.asarray(encode(x))
    equal(got.astype(np.int64), [133, 157, 203, 239, 255, 16, 0, 128, 255], label="codes")


@check("mu_law_decode maps code 0 to -1, code 255 to 1, and code 128 to a tiny positive number")
def test_mu_law_decode_ends(sub):
    decode = has(sub, "mu_law_decode")
    got = np.asarray(decode(np.array([0, 255, 127, 128, 157], dtype=np.int64)), dtype=float)
    close(got[:2], [-1.0, 1.0], atol=1e-12, label="the end codes")
    almost(got[3], 8.6212e-05, 1e-8, label="code 128, which is not exactly zero")
    almost(got[2], -got[3], 1e-12, label="code 127, the mirror image of code 128")
    almost(got[4], 0.01, 5e-4, label="code 157, which the chapter's table gives for an input of 0.01")


@check("mu-law decodes within 4% and beats 8-bit linear by 10 dB on quiet speech-like samples")
def test_mu_law_quality(sub):
    encode = has(sub, "mu_law_encode")
    decode = has(sub, "mu_law_decode")
    quantize = has(sub, "quantize")
    x = np.concatenate([np.geomspace(0.01, 1, 400), -np.geomspace(0.01, 1, 400)])
    back = np.asarray(decode(encode(x)), dtype=float)
    worst = float(np.max(np.abs(back - x) / np.abs(x)))
    if worst > 0.04:
        fail(f"the relative round-trip error reaches {worst:.3f} for inputs of 0.01 or more, expected at most 0.04")
    rng = np.random.default_rng(2)
    speech = np.clip(rng.laplace(scale=0.02, size=20_000), -1, 1)
    mu = snr_db(decode(encode(speech)), speech)
    lin = snr_db(quantize(speech, 8), speech)
    if mu < lin + 10:
        fail(f"on Laplacian samples at about -31 dBFS, mu-law gives {mu:.1f} dB and linear 8-bit {lin:.1f} dB; expected mu-law ahead by 10 dB")


@check("float_to_pcm16 and pcm16_to_float scale by 32768 and round-trip every integer")
def test_pcm(sub):
    to_pcm = has(sub, "float_to_pcm16")
    to_float = has(sub, "pcm16_to_float")
    got = np.asarray(to_pcm(np.array([0.0, 0.5, -0.5, -1.0, 1.0, 0.99999, -1.2, 1e-5])))
    if got.dtype != np.int16:
        fail(f"float_to_pcm16 should return int16, got {got.dtype}")
    equal(got, np.array([0, 16384, -16384, -32768, 32767, 32767, -32768, 0], dtype=np.int16), label="int16 values")
    every = np.arange(-32768, 32768).astype(np.int16)
    f = np.asarray(to_float(every))
    almost(f.min(), -1.0, 0, label="the smallest float")
    equal(np.asarray(to_pcm(f)), every, label="int16 -> float -> int16")


@check("wav_bytes writes the header from the chapter: 56 bytes for three stereo frames at 16 kHz")
def test_wav_header(sub):
    wav_bytes = has(sub, "wav_bytes")
    pcm = np.array([[1, -1], [2, -2], [3, -3]], dtype=np.int16)
    blob = bytes(wav_bytes(pcm, 16000))
    equal(len(blob), 56, label="the file length")
    equal(blob[0:4], b"RIFF", label="bytes 0 to 3")
    equal(blob[4:8], bytes([0x30, 0, 0, 0]), label="bytes 4 to 7, the size of everything after them: 36 plus 12 sample bytes is 48")
    equal(blob[8:16], b"WAVEfmt ", label="bytes 8 to 15")
    equal(blob[24:28], bytes([0x80, 0x3E, 0, 0]), label="bytes 24 to 27, the sample rate 16000")
    equal(blob[36:40], b"data", label="bytes 36 to 39")
    equal(blob[44:], bytes([1, 0, 0xFF, 0xFF, 2, 0, 0xFE, 0xFF, 3, 0, 0xFD, 0xFF]), label="the sample bytes, left then right for each frame")


@check("wav_bytes writes the same bytes as Python's wave module")
def test_wav_bytes(sub):
    wav_bytes = has(sub, "wav_bytes")
    rng = np.random.default_rng(3)
    for channels in (1, 2):
        shape = (1001,) if channels == 1 else (1001, 2)
        pcm = rng.integers(-32768, 32768, size=shape).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(channels)
            out.setsampwidth(2)
            out.setframerate(22050)
            out.writeframes(pcm.astype("<i2").tobytes())
        want = buffer.getvalue()
        got = bytes(wav_bytes(pcm, 22050))
        if got != want:
            where = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), min(len(got), len(want)))
            fail(f"{channels}-channel file: {len(got)} bytes against {len(want)}; first difference at byte {where} (header is bytes 0 to 43)")
        read, rate = soundfile.read(io.BytesIO(got), dtype="int16")
        equal(rate, 22050, label="the rate soundfile reads")
        equal(read, pcm, label=f"the {channels}-channel samples soundfile reads")


@check("read_wav_bytes skips extra chunks and round-trips a file byte for byte")
def test_read_wav(sub):
    read_wav = has(sub, "read_wav_bytes")
    wav_bytes = has(sub, "wav_bytes")
    rng = np.random.default_rng(4)
    stereo = rng.integers(-32768, 32768, size=(500, 2)).astype(np.int16)
    buffer = io.BytesIO()
    soundfile.write(buffer, stereo, 44100, format="WAV", subtype="PCM_16")
    pcm, rate = read_wav(buffer.getvalue())
    equal(rate, 44100, label="the sample rate")
    equal(np.asarray(pcm), stereo, label="stereo samples written by soundfile, shape (frames, channels)")

    mono = rng.integers(-32768, 32768, size=300).astype(np.int16)
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    info = b"INFOISFT\x05\x00\x00\x00test\x00"  # an odd-length chunk body, padded to even below
    body = (
        b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
        + b"LIST" + struct.pack("<I", len(info)) + info + b"\x00"
        + b"data" + struct.pack("<I", mono.size * 2) + mono.astype("<i2").tobytes()
    )
    blob = b"RIFF" + struct.pack("<I", len(body)) + body
    pcm, rate = read_wav(blob)
    equal(rate, 8000, label="the rate of a file with a LIST chunk")
    equal(np.asarray(pcm), mono, label="samples after a LIST chunk")

    original = wav_bytes(mono, 8000)
    again = wav_bytes(*read_wav(original))
    if bytes(again) != bytes(original):
        fail("writing, reading, and writing again changed the bytes")
