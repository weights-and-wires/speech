"""Checks for the voice conversion and spoofing exercise."""

import math

import numpy as np

from speechcheck import almost, check, close, equal, fail, has, shape


@check("cosine_knn finds the k nearest keys by angle, nearest first")
def test_knn(sub):
    knn = has(sub, "cosine_knn")
    rng = np.random.default_rng(4)
    q, keys = rng.normal(size=(20, 6)), rng.normal(size=(50, 6))
    got = np.asarray(knn(q, keys, 3))
    shape(got, (20, 3), label="indices")
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    kn = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    want = np.argsort(-(qn @ kn.T), axis=1)[:, :3]
    equal(got, want, label="indices")
    equal(np.asarray(knn(7.0 * q, 0.1 * keys, 3)), want, label="indices after rescaling (cosine ignores length)")


@check("knn_vc averages the k nearest matching frames, and copies frames exactly when k = 1")
def test_knn_vc(sub):
    vc = has(sub, "knn_vc")
    rng = np.random.default_rng(5)
    src, match = rng.normal(size=(15, 4)), rng.normal(size=(40, 4))
    one = np.asarray(vc(src, match, 1))
    shape(one, (15, 4), label="knn_vc(k=1)")
    for row in one:
        if not np.any(np.all(np.isclose(match, row, atol=1e-12), axis=1)):
            fail("with k = 1, every output frame must be an exact copy of a matching-set frame")
    sim = (src / np.linalg.norm(src, axis=1, keepdims=True)) @ (match / np.linalg.norm(match, axis=1, keepdims=True)).T
    want = match[np.argsort(-sim, axis=1)[:, :4]].mean(axis=1)
    close(np.asarray(vc(src, match, 4)), want, rtol=1e-10, label="knn_vc(k=4)")


def utterances(seed, n_src=200, n_match=600):
    """Synthetic features: frame = phone vector + speaker offset + noise, for two speakers."""
    rng = np.random.default_rng(seed)
    phones = rng.normal(size=(8, 32)) * 2.0
    speakers = rng.normal(size=(2, 32))

    def utter(spk, n):
        labels = rng.integers(0, 8, size=n)
        return phones[labels] + speakers[spk] + 0.5 * rng.normal(size=(n, 32)), labels

    src, src_labels = utter(0, n_src)
    match, _ = utter(1, n_match)
    return phones, speakers, src, src_labels, match


def content_and_speaker(out, phones, speakers, src_labels):
    """Fraction of frames that keep the source phone, and fraction closer to the target speaker."""
    phone_of = np.argmin(((out[:, None, :] - speakers[1][None, None] - phones[None]) ** 2).sum(-1), axis=1)
    residual = out - phones[src_labels]
    to_target = np.linalg.norm(residual - speakers[1], axis=1) < np.linalg.norm(residual - speakers[0], axis=1)
    return float(np.mean(phone_of == src_labels)), float(np.mean(to_target))


@check("kNN-VC changes the speaker and keeps the content on synthetic features")
def test_conversion(sub):
    vc = has(sub, "knn_vc")
    phones, speakers, src, src_labels, match = utterances(6)
    content, to_target = content_and_speaker(np.asarray(vc(src, match, 4)), phones, speakers, src_labels)
    if content < 0.95:
        fail(f"only {content:.0%} of converted frames keep the source phone, expected at least 95%")
    if to_target < 0.95:
        fail(f"only {to_target:.0%} of converted frames sound like the target speaker, expected at least 95%")


@check("knn_vc keeps content at small k and washes it out when k covers the whole matching set")
def test_k_sweep(sub):
    # Wrong implementation: matching on the wrong axis, as cosine_knn(matching, source), which
    # finds source frames for each target frame, so the output has the wrong length and no target frames.
    vc = has(sub, "knn_vc")
    phones, speakers, src, src_labels, match = utterances(7, n_src=120, n_match=400)
    out = np.asarray(vc(src, match, 4))
    shape(out, src.shape, label="knn_vc(k=4)")
    small, _ = content_and_speaker(out, phones, speakers, src_labels)
    everything = np.asarray(vc(src, match, len(match)))
    close(everything, np.broadcast_to(match.mean(axis=0), src.shape), atol=1e-9, label="knn_vc with k = len(matching): every frame is the matching-set mean")
    large, _ = content_and_speaker(everything, phones, speakers, src_labels)
    if small < 0.95:
        fail(f"at k = 4, only {small:.0%} of converted frames keep the source phone, expected at least 95%")
    if large > 0.4:
        fail(f"at k = len(matching), {large:.0%} of frames keep the phone, expected a collapse to at most 40%")


@check("error_rates counts a score equal to the threshold as accepted")
def test_error_rates(sub):
    # Wrong implementation: strict inequalities (bona fide <= th is rejected, spoof > th is accepted),
    # which moves one trial from each side at a tied threshold.
    rates = has(sub, "error_rates")
    bona, spoof = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])
    frr, far = rates(bona, spoof, np.array([1.0, 2.0, 5.0]))
    close(np.asarray(frr), [0.0, 1 / 3, 1.0], label="false rejection rates at thresholds 1, 2, 5")
    close(np.asarray(far), [2 / 3, 1 / 3, 0.0], label="false acceptance rates at thresholds 1, 2, 5")
    shape(np.asarray(frr), (3,), label="frr")


@check("equal_error_rate is exact on crafted scores and near theory for two Gaussians")
def test_eer(sub):
    eer_fn = has(sub, "equal_error_rate")
    eer, _ = eer_fn(np.array([0.9, 0.8, 0.7, 0.3]), np.array([0.6, 0.4, 0.2, 0.1]))
    almost(eer, 0.25, 1e-9, label="EER, one error on each side")
    eer, _ = eer_fn(np.array([2.0, 3.0]), np.array([0.0, 1.0]))
    almost(eer, 0.0, 1e-9, label="EER, perfectly separated")
    rng = np.random.default_rng(3)
    eer, th = eer_fn(rng.normal(2.0, 1.0, 4000), rng.normal(0.0, 1.0, 4000))
    theory = 0.5 * math.erfc(1.0 / math.sqrt(2))  # Phi(-1) for d' = 2
    almost(eer, theory, 0.02, label="EER for d' = 2")
    almost(th, 1.0, 0.1, label="the EER threshold, midway between the means")


@check("equal_error_rate interpolates the crossing between two thresholds")
def test_eer_interpolation(sub):
    # Wrong implementation: return the first threshold where FRR >= FAR without interpolating,
    # which reports an EER of 1/3 at threshold 0.5 where 1/4 at 0.475 is right.
    eer_fn = has(sub, "equal_error_rate")
    eer, th = eer_fn(np.array([0.9, 0.5, 0.4]), np.array([0.6, 0.3, 0.1, 0.0]))
    almost(eer, 0.25, 1e-9, label="EER of 3 bona fide and 4 spoof scores")
    almost(th, 0.475, 1e-9, label="threshold where the rates cross")
    eer_swapped, _ = eer_fn(np.array([0.6, 0.3, 0.1, 0.0]), np.array([0.9, 0.5, 0.4]))
    if eer_swapped < 0.5:
        fail(f"with the classes swapped the EER is {eer_swapped:.3f}, expected a value above 0.5 because the detector now points the wrong way")
