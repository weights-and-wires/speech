"""Checks for the quantization and audio files exercise."""

import io
import struct
import wave

import numpy as np
import soundfile

from speechcheck import almost, check, close, equal, fail, has, snr_db

RATE = 16000


def _busy_sine(amplitude=0.99, seconds=1.0):
    # 1234.5 Hz shares no short period with 16 kHz, so the rounding error looks like noise.
    n = np.arange(int(seconds * RATE))
    return amplitude * np.sin(2 * np.pi * 1234.5 * n / RATE + 0.1)


@check("quantize rounds to the nearest of 2**bits levels")
def test_quantize_small(sub):
    quantize = has(sub, "quantize")
    x = np.array([0.3, -0.62, 0.9, 0.05, 0.99, -1.0])
    close(quantize(x, 3), [0.25, -0.5, 0.75, 0.0, 0.75, -1.0], atol=1e-12, label="3-bit quantization")
    y = quantize(np.linspace(-1, 1, 5001), 4)
    levels = np.unique(np.round(y, 12))
    if levels.size != 16:
        fail(f"4 bits should give 16 distinct levels, got {levels.size}: {levels}")
    close(levels, np.arange(-8, 8) / 8, atol=1e-12, label="the 4-bit levels")


@check("predicted_snr_db is 6.02 N + 1.76 dB for a full-scale sine")
def test_predicted(sub):
    predicted = has(sub, "predicted_snr_db")
    for bits, want in [(8, 49.9257), (16, 98.0905), (24, 146.2553)]:
        almost(predicted(bits), want, 1e-3, label=f"{bits} bits")
    almost(predicted(16, 0.1), 78.0905, 1e-3, label="16 bits, a sine 20 dB below full scale")


@check("The measured SNR of a quantized sine matches the prediction at 8, 12, and 16 bits")
def test_measured_snr(sub):
    quantize = has(sub, "quantize")
    predicted = has(sub, "predicted_snr_db")
    x = _busy_sine()
    for bits in (8, 12, 16):
        measured = snr_db(quantize(x, bits), x)
        want = predicted(bits, 0.99)
        if abs(measured - want) > 0.25:
            fail(f"at {bits} bits the measured SNR is {measured:.2f} dB, the prediction {want:.2f} dB; expected them within 0.25 dB")


@check("TPDF dither makes the average output follow a signal smaller than one step")
def test_dither_mean(sub):
    quantize = has(sub, "quantize")
    dither = has(sub, "tpdf_dither")
    bits = 8
    step = 2.0 ** (1 - bits)
    x = np.full(200_000, 0.3 * step)
    plain = float(np.mean(quantize(x, bits)))
    if abs(plain) > 1e-12:
        fail(f"without dither, 0.3 of a step should round to 0, got a mean of {plain / step:.3f} steps")
    mean = float(np.mean(dither(x, bits, np.random.default_rng(0))))
    if abs(mean / step - 0.3) > 0.01:
        fail(f"with TPDF dither the mean output should be 0.300 steps, got {mean / step:.3f}")


@check("TPDF dither gives the same noise power, one step squared over 4, at every input level")
def test_dither_power(sub):
    dither = has(sub, "tpdf_dither")
    bits = 8
    step = 2.0 ** (1 - bits)
    rng = np.random.default_rng(1)
    for level in (0.0, 0.25, 0.5):
        x = np.full(200_000, level * step)
        err = dither(x, bits, rng) - x
        ratio = float(np.mean(err**2) / (step**2 / 4))
        if abs(ratio - 1) > 0.05:
            fail(f"at {level} of a step the error power is {ratio:.3f} times step**2 / 4, expected 1 within 0.05")


@check("mu_law_encode gives the 8-bit codes WaveNet used for known inputs")
def test_mu_law_codes(sub):
    encode = has(sub, "mu_law_encode")
    x = np.array([0.001, 0.01, 0.1, 0.5, 1.0, -0.5, -1.0, 0.0, 1.5])
    got = np.asarray(encode(x))
    equal(got.astype(np.int64), [133, 157, 203, 239, 255, 16, 0, 128, 255], label="codes")


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
