"""Checks for the durations and alignment-search exercise."""

import itertools

import numpy as np
import torch

from speechcheck import almost, check, close, equal, fail, has, shape


def _brute_force(P):
    """The best monotonic alignment score, by trying every segmentation."""
    n_text, n_frames = P.shape
    best = -np.inf
    for cuts in itertools.combinations(range(1, n_frames), n_text - 1):
        bounds = (0,) + cuts + (n_frames,)
        score = sum(P[i, bounds[i]:bounds[i + 1]].sum() for i in range(n_text))
        best = max(best, score)
    return best


def _check_path(path, n_text, n_frames, label):
    path = np.asarray(path)
    shape(path, (n_frames,), label=label)
    if path[0] != 0 or path[-1] != n_text - 1:
        fail(f"{label} must start at phone 0 and end at phone {n_text - 1}; got {path.tolist()}")
    steps = np.diff(path)
    if np.any((steps != 0) & (steps != 1)):
        fail(f"{label} may only stay or move forward by one phone per frame; got {path.tolist()}")


@check("length_regulate expands [h1, h2, h3] with durations [2, 1, 3] into six frames")
def test_lr_hand(sub):
    fn = has(sub, "length_regulate")
    h = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    out, mask = fn(h, torch.tensor([[2, 1, 3]]))
    shape(out, (1, 6, 2), label="the expanded sequence")
    close(out[0, :, 0], [1, 1, 2, 3, 3, 3], label="the first feature of each frame")
    equal(np.asarray(mask), np.ones((1, 6), bool), label="the frame mask")


@check("length_regulate pads a batch, matches repeat_interleave, and sends each h_i gradient d_i")
def test_lr_batch(sub):
    fn = has(sub, "length_regulate")
    rng = np.random.default_rng(0)
    h = torch.tensor(rng.normal(size=(3, 4, 5)), requires_grad=True)
    d = torch.tensor([[2, 3, 1, 4], [1, 1, 2, 0], [3, 0, 0, 0]])
    out, mask = fn(h, d)
    shape(out.detach(), (3, 10, 5), label="the expanded batch")
    for b in range(3):
        n_b = int(d[b].sum())
        want = torch.repeat_interleave(h[b], d[b], dim=0)
        close(out[b, :n_b].detach(), want.detach(), atol=1e-12, label=f"utterance {b}")
        close(out[b, n_b:].detach(), np.zeros((10 - n_b, 5)), atol=0.0, label=f"the padding of utterance {b}")
        equal(np.asarray(mask[b]), np.arange(10) < n_b, label=f"the mask of utterance {b}")
    out.sum().backward()
    close(h.grad[:, :, 0], d.double(), atol=1e-12,
          label="d(sum of outputs)/dh, which must equal how many frames copy each phone")


@check("durations_from_log inverts log(d + 1), scales by alpha, keeps at least one frame, and zeros padding")
def test_durations(sub):
    fn = has(sub, "durations_from_log")
    true = torch.tensor([[4.0, 1.0, 6.0, 2.0]])
    mask = torch.tensor([[True, True, True, False]])
    equal(np.asarray(fn(torch.log(true + 1), mask, 1.0)), [[4, 1, 6, 0]], label="alpha = 1")
    equal(np.asarray(fn(torch.log(true + 1), mask, 1.3)), [[5, 1, 8, 0]], label="alpha = 1.3")
    equal(np.asarray(fn(torch.tensor([[-3.0, 0.1, 2.0]]), torch.ones(1, 3, dtype=torch.bool), 1.0)),
          [[1, 1, 6]], label="predictions that round to 0 or less, and exp(2) - 1 = 6.39")


@check("frame_energy is the L2 norm of each STFT column, and phone_average averages over voiced frames")
def test_energy_and_average(sub):
    energy = has(sub, "frame_energy")
    average = has(sub, "phone_average")
    rng = np.random.default_rng(1)
    mag = np.abs(rng.normal(size=(9, 7)))
    close(energy(mag), np.linalg.norm(mag, axis=0), atol=1e-12, label="frame energy")
    values = np.array([100.0, 110, 0, 0, 200, 210, 220, 0, 300])
    durations = np.array([3, 2, 3, 1])
    voiced = values > 0
    close(average(values, durations), [70.0, 100.0, 430.0 / 3, 300.0], atol=1e-12,
          label="phone averages over every frame")
    close(average(values, durations, voiced), [105.0, 200.0, 215.0, 300.0], atol=1e-12,
          label="phone averages over voiced frames only")
    close(average(values, np.array([2, 2, 5]), voiced), [105.0, 0.0, 232.5], atol=1e-12,
          label="a phone with no voiced frames gets 0")


@check("quantize puts 200 Hz in log bin 109 of 256 between 71 and 795 Hz, and clips at both ends")
def test_quantize(sub):
    fn = has(sub, "quantize")
    got = np.asarray(fn(np.array([200.0, 71.0, 50.0, 795.0, 900.0]), 71.0, 795.0, 256, log_scale=True))
    equal(got, [109, 0, 0, 254, 255], label="log-scale bins for 200, 71, 50, 795, and 900 Hz")
    got = np.asarray(fn(np.array([0.0, 0.5, 1.0, 1.01, -2.0]), 0.0, 1.0, 5, log_scale=False))
    equal(got, [0, 2, 3, 4, 0], label="uniform bins for 0, 0.5, 1, 1.01, and -2 with 5 bins on [0, 1]")


@check("gaussian_log_likelihood matches the diagonal Gaussian log-density, token by frame")
def test_loglik(sub):
    fn = has(sub, "gaussian_log_likelihood")
    rng = np.random.default_rng(2)
    z = rng.normal(size=(7, 3))
    mu = rng.normal(size=(4, 3))
    log_sigma = rng.normal(size=(4, 3)) * 0.3
    got = fn(z, mu, log_sigma)
    shape(got, (4, 7), label="P")
    sigma = np.exp(log_sigma)
    want = np.zeros((4, 7))
    for i in range(4):
        for j in range(7):
            want[i, j] = np.sum(-0.5 * np.log(2 * np.pi) - log_sigma[i]
                                - (z[j] - mu[i]) ** 2 / (2 * sigma[i] ** 2))
    close(got, want, atol=1e-10, label="P[i, j] = log N(z_j; mu_i, sigma_i)")


@check("monotonic_alignment_search solves the hand-worked 3 by 4 example")
def test_mas_hand(sub):
    fn = has(sub, "monotonic_alignment_search")
    P = np.array([[-1.0, -2, -6, -8], [-5, -1, -1, -7], [-9, -6, -3, -1]])
    path = fn(P.copy())
    _check_path(path, 3, 4, "the path")
    equal(np.asarray(path), [0, 1, 1, 2], label="the path")
    durations = has(sub, "path_to_durations")(np.asarray(path), 3)
    equal(np.asarray(durations), [1, 2, 1], label="the durations")


@check("monotonic_alignment_search finds the best score that brute force finds")
def test_mas_brute(sub):
    fn = has(sub, "monotonic_alignment_search")
    rng = np.random.default_rng(3)
    for n_text, n_frames in [(3, 5), (4, 7), (5, 9), (2, 8), (4, 4), (1, 6)]:
        P = rng.normal(size=(n_text, n_frames)) * 3
        path = np.asarray(fn(P.copy()))
        _check_path(path, n_text, n_frames, f"the path for a {n_text} by {n_frames} P")
        got = float(P[path, np.arange(n_frames)].sum())
        want = _brute_force(P)
        almost(got, want, 1e-9, label=f"the path's total log-likelihood for a {n_text} by {n_frames} P")


@check("MAS recovers the durations of frames drawn from each phone's Gaussian")
def test_mas_recovers(sub):
    loglik = has(sub, "gaussian_log_likelihood")
    mas = has(sub, "monotonic_alignment_search")
    to_d = has(sub, "path_to_durations")
    rng = np.random.default_rng(4)
    n_text, dim = 12, 6
    true = rng.integers(2, 15, n_text)
    mu = rng.normal(size=(n_text, dim)) * 3
    log_sigma = np.zeros((n_text, dim))
    z = mu[np.repeat(np.arange(n_text), true)] + rng.standard_normal((true.sum(), dim))
    found = np.asarray(to_d(np.asarray(mas(loglik(z, mu, log_sigma))), n_text))
    equal(found, true, label="durations recovered from 12 phones")
