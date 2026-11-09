"""Checks for the CTC many-paths exercise."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax

from speechcheck import almost, check, close, equal, fail, has


def worked_log_probs():
    """The chapter's T = 3 example: blank = 0, a = 1, P(a) = 0.6, 0.3, 0.8."""
    p_a = np.array([0.6, 0.3, 0.8])
    return np.log(np.stack([1 - p_a, p_a], axis=1))


def random_log_probs(rng, T, C, scale=1.5):
    return log_softmax(rng.normal(size=(T, C)) * scale, axis=1)


def torch_loss(log_probs, target):
    lp = torch.tensor(log_probs)[:, None, :]
    targets = torch.tensor([list(target) or [1]], dtype=torch.long)
    return F.ctc_loss(lp, targets, [len(log_probs)], [len(target)], blank=0, reduction="sum").item()


@check("collapse merges repeated symbols first, then drops blanks")
def test_collapse(sub):
    collapse = has(sub, "collapse")
    a, b, _ = 1, 2, 0
    for path, want in (([a, a, _, a, b, _, _], [a, a, b]), ([a, _, a], [a, a]), ([a, a], [a]),
                       ([_, _, _], []), ([b, b, a, a, _, _, a], [b, a, a])):
        got = [int(v) for v in collapse(path)]
        if got != want:
            fail(f"collapse({path}) should be {want}, got {got}")


@check("path_log_prob multiplies one probability per frame, which is a sum of logs")
def test_path_prob(sub):
    path_log_prob = has(sub, "path_log_prob")
    log_probs = worked_log_probs()
    almost(float(np.exp(path_log_prob(log_probs, [0, 0, 1]))), 0.4 * 0.7 * 0.8, 1e-12, label="path __a")
    almost(float(np.exp(path_log_prob(log_probs, [1, 1, 1]))), 0.6 * 0.3 * 0.8, 1e-12, label="path aaa")


@check("paths_for lists the six paths that spell 'a' in 3 frames, and the one path that spells 'aa'")
def test_paths_for(sub):
    paths_for = has(sub, "paths_for")
    got = {tuple(int(s) for s in p) for p in paths_for([1], 3, 2)}
    want = {(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1)}
    equal(got, want, label="the paths that spell 'a' in 3 frames")
    got = {tuple(int(s) for s in p) for p in paths_for([1, 1], 3, 2)}
    equal(got, {(1, 0, 1)}, label="the paths that spell 'aa' in 3 frames")
    equal(len(paths_for([1, 1], 2, 2)), 0, label="the number of paths that spell 'aa' in 2 frames")


@check("ctc_loss_brute_force reproduces the worked example: p = 0.608")
def test_worked(sub):
    loss = has(sub, "ctc_loss_brute_force")
    almost(loss(worked_log_probs(), [1]), -np.log(0.608), 1e-10, label="-ln 0.608")


@check("ctc_loss_brute_force matches torch's ctc_loss, including targets with repeated labels")
def test_against_torch(sub):
    loss = has(sub, "ctc_loss_brute_force")
    rng = np.random.default_rng(1)
    for target in ([1], [1, 2], [1, 1], [2, 2, 1], [2, 1, 2]):
        log_probs = random_log_probs(rng, 6, 3)
        want = torch_loss(log_probs, target)
        got = loss(log_probs, target)
        if abs(got - want) > 1e-8 * max(1.0, abs(want)):
            fail(f"target {target}, T = 6, C = 3: torch gives {want:.10f}, your loss gives {got:.10f}")


@check("A transcript can be likelier than the best path: p('a') = 0.64 against p(empty) = 0.36")
def test_more_than_best_path(sub):
    loss = has(sub, "ctc_loss_brute_force")
    log_probs = np.log(np.array([[0.6, 0.4], [0.6, 0.4]]))
    almost(float(np.exp(-loss(log_probs, [1]))), 0.64, 1e-10, label="p('a'), from the paths a_, _a, and aa")
    almost(float(np.exp(-loss(log_probs, []))), 0.36, 1e-10, label="p(empty), from the one path __")


@check("min_frames counts one frame per label plus one per repeat, and fewer frames give an infinite loss")
def test_min_frames(sub):
    min_frames, loss = has(sub, "min_frames"), has(sub, "ctc_loss_brute_force")
    equal(min_frames([1]), 1, label="min_frames of 'a'")
    equal(min_frames([1, 1]), 3, label="min_frames of 'aa'")
    equal(min_frames([1, 2, 2, 3, 3]), 7, label="min_frames of 'abbcc'")
    rng = np.random.default_rng(2)
    for target in ([1, 1, 2], [2, 2, 2]):
        need = min_frames(target)
        enough = loss(random_log_probs(rng, need, 3), target)
        if not np.isfinite(enough):
            fail(f"target {target} fits in {need} frames, but the loss is {enough}")
        short = loss(random_log_probs(rng, need - 1, 3), target)
        if short != float("inf"):
            fail(f"target {target} cannot fit in {need - 1} frames, so the loss must be inf; got {short}")
