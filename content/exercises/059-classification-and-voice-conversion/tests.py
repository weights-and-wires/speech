"""Checks for the audio classification and voice conversion exercise."""

import math

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


@check("ast_patch_grid gives 12 x 101 = 1212 patches for 128 mel bins and 1024 frames")
def test_grid(sub):
    grid = has(sub, "ast_patch_grid")
    equal(tuple(int(v) for v in grid(128, 1024)), (12, 101), label="ast_patch_grid(128, 1024)")
    equal(tuple(int(v) for v in grid(128, 100)), (12, 9), label="ast_patch_grid(128, 100)")
    equal(tuple(int(v) for v in grid(40, 26, patch=16, stride=16)), (2, 1), label="ast_patch_grid(40, 26, 16, 16)")


@check("patchify cuts overlapping 16 x 16 patches in frequency-major order")
def test_patchify(sub):
    patchify = has(sub, "patchify")
    rng = np.random.default_rng(0)
    spec = rng.normal(size=(36, 47))
    got = np.asarray(patchify(spec, 16, 10))
    shape(got, (3 * 4, 256), label="patches of a 36 x 47 spectrogram")
    want = np.stack([spec[f : f + 16, t : t + 16].ravel() for f in (0, 10, 20) for t in (0, 10, 20, 30)])
    close(got, want, label="patches")
    conv = torch.nn.functional.conv2d(torch.tensor(spec)[None, None], torch.eye(256, dtype=torch.float64).view(256, 1, 16, 16), stride=10)
    close(got, conv[0].flatten(1).T.numpy(), label="patches against a stride-10 Conv2d")


def ref_clap(a, t, tau):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    s = a @ t.T / tau
    n = len(a)
    row = [-(s[i, i] - np.log(np.exp(s[i]).sum())) for i in range(n)]
    col = [-(s[i, i] - np.log(np.exp(s[:, i]).sum())) for i in range(n)]
    return (sum(row) + sum(col)) / (2 * n)


@check("clap_loss matches the symmetric InfoNCE formula")
def test_clap_value(sub):
    loss = has(sub, "clap_loss")
    rng = np.random.default_rng(1)
    a, t = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    got = float(loss(torch.tensor(a), torch.tensor(t), torch.tensor(0.07, dtype=torch.float64)))
    almost(got, ref_clap(a, t, 0.07), 1e-8, label="clap_loss")
    almost(float(loss(torch.tensor(5 * a), torch.tensor(0.2 * t), torch.tensor(0.07, dtype=torch.float64))), got, 1e-8,
           label="clap_loss after rescaling the embeddings")


@check("clap_loss equals log N when every embedding is the same, and is symmetric in audio and text")
def test_clap_props(sub):
    loss = has(sub, "clap_loss")
    same = torch.ones(5, 4, dtype=torch.float64)
    almost(float(loss(same, same, torch.tensor(0.1, dtype=torch.float64))), math.log(5), 1e-9, label="identical embeddings")
    rng = np.random.default_rng(2)
    a, t = torch.tensor(rng.normal(size=(7, 3))), torch.tensor(rng.normal(size=(7, 3)))
    almost(float(loss(a, t, torch.tensor(0.5, dtype=torch.float64))), float(loss(t, a, torch.tensor(0.5, dtype=torch.float64))), 1e-9, label="swapping audio and text")
    tau = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    loss(a, t, tau).backward()
    h = 1e-6
    numeric = (ref_clap(a.numpy(), t.numpy(), 0.3 + h) - ref_clap(a.numpy(), t.numpy(), 0.3 - h)) / (2 * h)
    almost(float(tau.grad), numeric, 1e-5, label="d loss / d tau against a finite difference")


@check("zero_shot_classify picks the prompt with the highest cosine similarity")
def test_zero_shot(sub):
    zs = has(sub, "zero_shot_classify")
    prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    audio = torch.tensor([[10.0, 1.0], [0.2, 3.0], [-0.1, -0.2], [0.9, 1.0]])
    equal(torch.as_tensor(zs(audio, prompts)).tolist(), [0, 1, 2, 1], label="predicted classes")


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


@check("kNN-VC changes the speaker and keeps the content on synthetic features")
def test_conversion(sub):
    vc = has(sub, "knn_vc")
    rng = np.random.default_rng(6)
    phones = rng.normal(size=(8, 32)) * 2.0
    speakers = rng.normal(size=(2, 32))

    def utter(spk, n):
        labels = rng.integers(0, 8, size=n)
        return phones[labels] + speakers[spk] + 0.5 * rng.normal(size=(n, 32)), labels

    src, src_labels = utter(0, 200)
    match, _ = utter(1, 600)
    out = np.asarray(vc(src, match, 4))
    # A speaker probe: which speaker's offset explains the frame better, once its phone is removed.
    phone_of = np.argmin(((out[:, None, :] - speakers[1][None, None] - phones[None]) ** 2).sum(-1), axis=1)
    content = np.mean(phone_of == src_labels)
    residual = out - phones[src_labels]
    to_target = np.mean(np.linalg.norm(residual - speakers[1], axis=1) < np.linalg.norm(residual - speakers[0], axis=1))
    if content < 0.95:
        fail(f"only {content:.0%} of converted frames keep the source phone, expected at least 95%")
    if to_target < 0.95:
        fail(f"only {to_target:.0%} of converted frames sound like the target speaker, expected at least 95%")
