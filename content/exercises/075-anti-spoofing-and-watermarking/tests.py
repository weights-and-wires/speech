"""Checks for the anti-spoofing and watermarking exercise."""

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape, snr_db, tone

RATE = 16000


def dct_matrix(n):
    k, m = np.arange(n)[:, None], np.arange(n)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * k * (m + 0.5) / n)
    mat[0] /= np.sqrt(2.0)
    return mat


def host(seconds=1.0, offset=0.0):
    return offset + tone(220.0, RATE, seconds, amplitude=0.2) + tone(660.0, RATE, seconds, amplitude=0.1)


def mix_at_snr(clean, noise, snr):
    gain = np.sqrt(np.mean(clean**2) / (np.mean(noise**2) * 10 ** (snr / 10)))
    return clean + gain * noise


@check("lfcc_filterbank matches the closed-form triangles and spaces peaks equally in hertz")
def test_filterbank(sub):
    # Wrong implementation: mel-spaced edges, which bunch the filters at low frequencies.
    fb_fn = has(sub, "lfcc_filterbank")
    fb = np.asarray(fb_fn(3, 16, 16000))
    shape(fb, (3, 9), label="filterbank")
    want = np.zeros((3, 9))
    want[0, :5] = [0, 0.5, 1, 0.5, 0]
    want[1, 2:7] = [0, 0.5, 1, 0.5, 0]
    want[2, 4:9] = [0, 0.5, 1, 0.5, 0]
    close(fb, want, atol=1e-9, label="3 filters, 16-point FFT, 16 kHz")
    big = np.asarray(fb_fn(20, 512, RATE))
    peaks = np.argmax(big, axis=1) * RATE / 512
    steps = np.diff(peaks)
    if steps.max() - steps.min() > 2 * RATE / 512:
        fail(f"filter peaks are {steps.min():.0f} to {steps.max():.0f} Hz apart, expected equal steps")


@check("lfcc is exact on a flat spectrum and matches an explicit DCT matrix")
def test_lfcc(sub):
    lfcc, fb_fn = has(sub, "lfcc"), has(sub, "lfcc_filterbank")
    fb = fb_fn(3, 16, 16000)
    flat = np.asarray(lfcc(np.ones((9, 4)), fb, n_ceps=3))
    shape(flat, (3, 4), label="lfcc")
    want = np.zeros((3, 4))
    want[0] = np.sqrt(3.0) * np.log(2.0)  # every filter sums to 2, so c0 = sqrt(3) ln 2 and the rest vanish
    close(flat, want, atol=1e-6, label="LFCC of a flat spectrum of ones")
    rng = np.random.default_rng(0)
    power = rng.random((257, 6)) + 0.1
    big = fb_fn(20, 512, RATE)
    got = np.asarray(lfcc(power, big, n_ceps=13))
    close(got, (dct_matrix(20) @ np.log(big @ power + 1e-10))[:13], atol=1e-6, label="13 coefficients of 20 filters")


@check("tdcf_costs returns the ASVspoof 2019 weights C1 and C2")
def test_costs(sub):
    costs = has(sub, "tdcf_costs")
    c1, c2 = costs(0.05, 0.01, 0.3)
    almost(c1, 0.892525, 1e-9, label="C1 for ASV miss 0.05, false alarm 0.01")
    almost(c2, 0.35, 1e-9, label="C2 for ASV spoof miss 0.3 (spoof accepted 0.7)")
    c1, c2 = costs(0.1, 0.2, 0.5, p_tar=0.5, p_non=0.1, p_spoof=0.4, c_miss_asv=1.0, c_fa_asv=2.0, c_miss_cm=2.0, c_fa_cm=3.0)
    almost(c1, 0.91, 1e-9, label="C1 with custom costs")
    almost(c2, 0.6, 1e-9, label="C2 with custom costs")


@check("min_tdcf normalizes by min(C1, C2) and matches a hand-worked table")
def test_min_tdcf(sub):
    # Wrong implementation: normalizing by C1 (or by C1 + C2) instead of min(C1, C2). It reports 0.098
    # for the table below where 0.25 is right, and it makes an uninformative detector look excellent.
    fn = has(sub, "min_tdcf")
    bona, spoof = np.array([0.9, 0.5, 0.4]), np.array([0.6, 0.3, 0.1, 0.0])
    cost, th = fn(bona, spoof, 0.892525, 0.35)
    almost(cost, 0.25, 1e-9, label="min t-DCF, C1 = 0.892525, C2 = 0.35")
    almost(th, 0.4, 1e-9, label="the threshold that gives it")
    cost, th = fn(bona, spoof, 0.6, 0.91)
    almost(cost, 0.91 * 0.25 / 0.6, 1e-9, label="min t-DCF when C1 < C2 (C1 = 0.6, C2 = 0.91), normalized by C1")
    almost(th, 0.4, 1e-9, label="its threshold")
    rng = np.random.default_rng(0)
    same = fn(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000), 0.892525, 0.35)[0]
    if not 0.9 <= same <= 1.0 + 1e-9:
        fail(f"an uninformative detector scored {same:.3f}, expected just under 1 (the cost of the better trivial detector)")
    perfect = fn(np.array([1.0, 2.0]), np.array([-1.0, 0.0]), 0.892525, 0.35)[0]
    almost(perfect, 0.0, 1e-9, label="min t-DCF of a perfectly separating detector")


@check("The watermark decodes with no attack, ignores a DC offset, and needs the right key")
def test_clean(sub):
    # Wrong implementation: correlating the raw segments without removing each segment's mean. A host with a
    # constant offset adds offset * sum(chips) to every correlation, which swamps the watermark and flips bits.
    embed, detect = has(sub, "embed_watermark"), has(sub, "detect_watermark")
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 32)
    x = host(1.0, offset=0.5)
    marked = np.asarray(embed(x, bits, key=3, strength=0.03))
    shape(marked, x.shape, label="watermarked signal")
    close(np.abs(marked - x)[: 500 * 32], np.full(500 * 32, 0.03), atol=1e-9, label="watermark amplitude")
    close(marked[500 * 32 :], x[500 * 32 :], atol=1e-12, label="samples past the last whole segment")
    got = np.asarray(detect(marked, 32, key=3))
    equal(got, bits, label="decoded bits with no attack")
    wrong = np.asarray(detect(marked, 32, key=4))
    if np.mean(wrong != bits) < 0.2:
        fail(f"the wrong key decoded {np.mean(wrong != bits):.0%} of bits wrongly, expected about half")
    other = np.asarray(embed(x, bits, key=3, strength=0.03))
    close(other, marked, label="the same key and bits embed the same way")


@check("The watermark survives 20 dB noise and an MP3-like low-pass")
def test_attacks(sub):
    from scipy.signal import butter, filtfilt

    embed, detect = has(sub, "embed_watermark"), has(sub, "detect_watermark")
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 32)
    x = host(1.0, offset=0.1)
    marked = np.asarray(embed(x, bits, key=5, strength=0.03))
    noisy = mix_at_snr(marked, rng.standard_normal(len(marked)), 20.0)
    almost(snr_db(noisy, marked), 20.0, 0.01, label="SNR of the attack")
    ber = np.mean(np.asarray(detect(noisy, 32, key=5)) != bits)
    if ber > 1 / 32:
        fail(f"bit error rate {ber:.3f} after 20 dB noise, expected at most {1 / 32:.3f} (one bit in 32)")
    b, a = butter(6, 4000 / (RATE / 2))
    low = filtfilt(b, a, marked)
    ber = np.mean(np.asarray(detect(low, 32, key=5)) != bits)
    if ber > 0.1:
        fail(f"bit error rate {ber:.3f} after a 4 kHz low-pass, expected at most 0.1")


@check("anonymize_embeddings averages k pool vectors and ignores the source")
def test_anonymize(sub):
    fn = has(sub, "anonymize_embeddings")
    pool = np.eye(10)
    out = np.asarray(fn(np.ones((6, 10)), pool, 2, np.random.default_rng(0)))
    shape(out, (6, 10), label="anonymized vectors")
    equal((out > 0).sum(axis=1), np.full(6, 2), label="nonzero entries per row, k = 2 of a one-hot pool")
    close(out.sum(axis=1), np.ones(6), label="row sums")
    if np.allclose(out, out[0]):
        fail("every output row is identical, expected a fresh draw for each row")
    rng = np.random.default_rng(1)
    vectors, big_pool = rng.standard_normal((300, 16)), rng.standard_normal((200, 16))
    swapped = np.asarray(fn(vectors, big_pool, 20, np.random.default_rng(2)))
    cos = np.sum(vectors * swapped, axis=1) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(swapped, axis=1))
    if abs(cos.mean()) > 0.05:
        fail(f"mean cosine between sources and outputs is {cos.mean():.3f}, expected about 0 for a swap")


@check("attacker_eer ignores the diagonal in non-target trials and reflects privacy")
def test_attacker(sub):
    # Wrong implementation: scoring every off-diagonal and diagonal pair as non-target, so the matching pairs
    # also sit among the non-target scores and the EER of a perfect attacker is far above 0.
    fn = has(sub, "attacker_eer")
    almost(fn(np.eye(4), np.eye(4)), 0.0, 1e-9, label="EER of a perfect attacker (identity vectors)")
    rng = np.random.default_rng(0)
    n, d = 60, 16
    speakers = rng.standard_normal((n, d))
    enrolled = speakers + 0.5 * rng.standard_normal((n, d))
    trials = speakers + 0.5 * rng.standard_normal((n, d))
    before = fn(enrolled, trials)
    if before > 0.1:
        fail(f"attacker EER {before:.3f} before anonymization, expected at most 0.1")
    pool = rng.standard_normal((300, d))
    swap = has(sub, "anonymize_embeddings")
    after = fn(enrolled, np.asarray(swap(trials, pool, 20, np.random.default_rng(3))))
    if not 0.4 <= after <= 0.6:
        fail(f"attacker EER {after:.3f} after the swap, expected between 0.4 and 0.6")
