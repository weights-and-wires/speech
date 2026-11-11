"""Checks for the audio classification and CLAP exercise."""

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


def ref_bce(x, y):
    s = 1.0 / (1.0 + np.exp(-x))
    return float(np.mean(-(y * np.log(s) + (1 - y) * np.log(1 - s))))


@check("multilabel_bce matches the per-class sigmoid cross-entropy, and its gradient is (sigmoid - target) / (N C)")
def test_bce_value(sub):
    bce = has(sub, "multilabel_bce")
    rng = np.random.default_rng(10)
    x = rng.normal(scale=2.0, size=(5, 4))
    y = (rng.random((5, 4)) < 0.4).astype(float)
    y[0] = [1.0, 1.0, 0.0, 0.0]  # a clip with two labels at once
    logits = torch.tensor(x, requires_grad=True)
    loss = bce(logits, torch.tensor(y))
    almost(float(loss.detach()), ref_bce(x, y), 1e-10, label="multilabel_bce")
    loss.backward()
    numeric = np.zeros_like(x)
    h = 1e-6
    for idx in np.ndindex(*x.shape):
        up, down = x.copy(), x.copy()
        up[idx] += h
        down[idx] -= h
        numeric[idx] = (ref_bce(up, y) - ref_bce(down, y)) / (2 * h)
    close(logits.grad.numpy(), numeric, rtol=1e-5, atol=1e-8, label="gradient against a finite difference")
    close(logits.grad.numpy(), (1 / (1 + np.exp(-x)) - y) / x.size, rtol=1e-8, label="gradient against (sigmoid - target) / (N C)")


@check("multilabel_bce stays finite and exact for logits of plus or minus 1000")
def test_bce_stable(sub):
    # Wrong implementation: -(y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))), which rounds
    # sigmoid(40) to 1.0 in float64 and returns inf for a confident wrong logit.
    bce = has(sub, "multilabel_bce")
    x = torch.tensor([[40.0, -40.0], [-1000.0, 1000.0]], dtype=torch.float64)
    y = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    got = float(bce(x, y))
    if not math.isfinite(got):
        fail(f"multilabel_bce returned {got}, expected 520.0 for four confident wrong logits of 40, 40, 1000, and 1000")
    almost(got, 520.0, 1e-9, label="multilabel_bce on confident wrong logits")
    almost(float(bce(-x, y)), 0.0, 1e-9, label="multilabel_bce on confident right logits")


@check("average_precision averages precision over the positive ranks only")
def test_ap(sub):
    # Wrong implementation: averaging precision@rank over every rank instead of only the positive
    # ranks, which gives 0.669 where 0.806 is right for the labels 1 0 1 1 0 0 in rank order.
    ap = has(sub, "average_precision")
    scores = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.2])  # shuffled; sorted by score the clips run 0.9 0.7 0.5 0.3 0.2 0.1
    labels = np.array([1, 1, 0, 0, 1, 0])  # which gives the labels 1 0 1 1 0 0 in rank order
    almost(ap(scores, labels), (1 + 2 / 3 + 3 / 4) / 3, 1e-12, label="AP for labels 1 0 1 1 0 0 in rank order")
    almost(ap(np.array([3.0, 2.0, 1.0]), np.array([1, 1, 0])), 1.0, 1e-12, label="AP of a perfect ranking")
    almost(ap(np.array([3.0, 2.0, 1.0]), np.array([0, 0, 1])), 1 / 3, 1e-12, label="AP when the one positive is ranked last of 3")
    if not math.isnan(ap(np.array([1.0, 2.0]), np.array([0, 0]))):
        fail("average_precision with no positive clips must return NaN")


@check("mean_average_precision averages over the classes that have positives, skipping the rest")
def test_map(sub):
    mean_ap = has(sub, "mean_average_precision")
    scores = np.array([[6, 1, 5], [5, 2, 4], [4, 3, 3], [3, 4, 2], [2, 5, 1], [1, 6, 0]], dtype=float)
    labels = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=float)
    got = mean_ap(scores, labels)
    almost(got, ((1 + 2 / 3 + 3 / 4) / 3 + 1.0) / 2, 1e-12,
           label="mAP of one imperfect class, one perfect class, and one class with no positives")


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


@check("zero_shot_classify normalizes both sides, so a long prompt embedding does not win by length")
def test_zero_shot_norm(sub):
    # Wrong implementation: argmax of the raw dot product audio @ prompts.T, with no L2 normalization;
    # the dog prompt (4, 0) wins the clip (1, 1.2) with a dot product of 4 against 1.2, where the rain
    # prompt (0, 1) is closer by angle (cosine 0.768 against 0.640).
    zs = has(sub, "zero_shot_classify")
    prompts = torch.tensor([[4.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    audio = torch.tensor([[1.0, 1.2], [3.0, 0.5], [-0.2, -0.1]])
    got = torch.as_tensor(zs(audio, prompts))
    shape(got.numpy(), (3,), label="predicted classes")
    equal(got.tolist(), [1, 0, 2], label="predicted classes")
    equal(torch.as_tensor(zs(100.0 * audio, prompts * torch.tensor([[0.1], [30.0], [2.0]]))).tolist(), [1, 0, 2],
          label="predicted classes after rescaling every clip and prompt")
