"""Checks for the connectionist temporal classification exercise."""

import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import log_softmax, logsumexp

from speechcheck import almost, check, close, equal, fail, finite_difference, has, shape, timed


def worked_log_probs():
    """The lesson's T = 3 example: blank = 0, a = 1, P(a) = 0.6, 0.3, 0.8."""
    p_a = np.array([0.6, 0.3, 0.8])
    return np.log(np.stack([1 - p_a, p_a], axis=1))


def random_log_probs(rng, T, C, scale=1.5):
    return log_softmax(rng.normal(size=(T, C)) * scale, axis=1)


def torch_loss(log_probs, target):
    """torch's CTC loss for one utterance, with blank 0 and the (T, N, C) layout."""
    lp = torch.tensor(log_probs)[:, None, :]
    targets = torch.tensor([list(target) or [1]], dtype=torch.long)
    return F.ctc_loss(lp, targets, [len(log_probs)], [len(target)], blank=0, reduction="sum").item()


def brute_force(log_probs, target):
    """-log of the summed probability of every path that collapses to target."""
    T, C = log_probs.shape
    total = -np.inf
    for path in itertools.product(range(C), repeat=T):
        collapsed = [s for k, s in enumerate(path) if s != 0 and (k == 0 or s != path[k - 1])]
        if collapsed == list(target):
            total = np.logaddexp(total, sum(log_probs[t, s] for t, s in enumerate(path)))
    return -total


@check("extend_with_blanks interleaves blanks, and collapse merges repeats before dropping blanks")
def test_extend_collapse(sub):
    extend, collapse = has(sub, "extend_with_blanks"), has(sub, "collapse")
    equal(np.asarray(extend([3, 1, 1])), np.array([0, 3, 0, 1, 0, 1, 0]), label="extend_with_blanks([3, 1, 1])")
    equal(np.asarray(extend([])), np.array([0]), label="extend_with_blanks([])")
    a, b, _ = 1, 2, 0
    for path, want in (([a, a, _, a, b, _, _], [a, a, b]), ([a, _, a], [a, a]), ([a, a], [a]),
                       ([_, _, _], []), ([b, b, a, a, _, _, a], [b, a, a])):
        got = [int(v) for v in collapse(path)]
        if got != want:
            fail(f"collapse({path}) should be {want}, got {got}")


@check("ctc_forward reproduces the T = 3 table worked in the lesson")
def test_forward_table(sub):
    ctc_forward = has(sub, "ctc_forward")
    log_alpha = ctc_forward(worked_log_probs(), [1])
    shape(log_alpha, (3, 3), label="log_alpha for T = 3 and the label [1]")
    want = np.array([[0.4, 0.6, 0.0], [0.28, 0.30, 0.42], [0.056, 0.464, 0.144]])
    close(np.exp(log_alpha), want, atol=1e-12, label="alpha")


@check("ctc_loss equals the brute-force sum over every path, including repeated labels")
def test_brute_force(sub):
    ctc_loss = has(sub, "ctc_loss")
    almost(ctc_loss(worked_log_probs(), [1]), -np.log(0.608), 1e-10, label="the worked example, -ln 0.608")
    rng = np.random.default_rng(0)
    for target in ([1], [1, 2], [1, 1], [2, 2, 1], [], [2, 1, 2]):
        log_probs = random_log_probs(rng, 5, 3)
        want = brute_force(log_probs, target)
        got = ctc_loss(log_probs, target)
        if abs(got - want) > 1e-9:
            fail(f"target {target}, T = 5, C = 3: the sum over all 243 paths gives {want:.8f}, "
                 f"your ctc_loss gives {got:.8f}")


@check("ctc_loss matches torch.nn.functional.ctc_loss with reduction='sum'")
def test_against_torch(sub):
    ctc_loss = has(sub, "ctc_loss")
    rng = np.random.default_rng(1)
    for target in ([1, 1, 2], [3, 3, 3], [2, 1, 2, 1], [4], [1, 2, 3, 4, 4, 1]):
        log_probs = random_log_probs(rng, 14, 5)
        want = torch_loss(log_probs, target)
        got = ctc_loss(log_probs, target)
        if abs(got - want) > 1e-8 * max(1.0, abs(want)):
            fail(f"target {target}, T = 14: torch gives {want:.10f}, your ctc_loss gives {got:.10f}")


@check("A repeated label needs a blank between its copies: 'aa' fits in 3 frames but not in 2")
def test_feasibility(sub):
    ctc_loss = has(sub, "ctc_loss")
    almost(ctc_loss(worked_log_probs(), [1, 1]), -np.log(0.6 * 0.7 * 0.8), 1e-10,
           label="'aa' in 3 frames, whose only path is a_a")
    got = ctc_loss(worked_log_probs()[:2], [1, 1])
    if got != np.inf:
        fail(f"'aa' in 2 frames has no valid path, so the loss must be inf (torch agrees); got {got}")
    if torch_loss(worked_log_probs()[:2], [1, 1]) != np.inf:
        fail("torch disagrees with this check's claim; tell the course authors")


@check("ctc_backward ends with log 1 on the last two states, and alpha times beta gives p at every frame")
def test_backward(sub):
    ctc_forward, ctc_backward = has(sub, "ctc_forward"), has(sub, "ctc_backward")
    log_beta = ctc_backward(worked_log_probs(), [1])
    shape(log_beta, (3, 3), label="log_beta for the worked example")
    close(np.exp(log_beta), [[0.86, 0.44, 0.14], [0.8, 1.0, 0.2], [0.0, 1.0, 1.0]], atol=1e-12,
          label="beta for the worked example")
    rng = np.random.default_rng(2)
    for target in ([1, 2, 2, 3], [3], [1, 1, 1]):
        log_probs = random_log_probs(rng, 20, 4)
        log_p = logsumexp(ctc_forward(log_probs, target)[-1, -2:])
        per_frame = logsumexp(ctc_forward(log_probs, target) + ctc_backward(log_probs, target), axis=1)
        close(per_frame, np.full(20, log_p), atol=1e-9,
              label=f"log sum_s alpha_t(s) beta_t(s) at every frame for target {target}")


@check("ctc_grad matches torch's autograd through log_softmax and ctc_loss")
def test_grad_torch(sub):
    ctc_grad = has(sub, "ctc_grad")
    rng = np.random.default_rng(3)
    for target in ([1, 2, 2], [3, 1, 3, 3], [2]):
        logits = rng.normal(size=(12, 4)) * 2
        loss, grad = ctc_grad(logits, target)
        shape(grad, (12, 4), label="the gradient")
        x = torch.tensor(logits, requires_grad=True)
        want = F.ctc_loss(torch.log_softmax(x, dim=1)[:, None], torch.tensor([target]), [12], [len(target)],
                          blank=0, reduction="sum")
        want.backward()
        almost(loss, want.item(), 1e-9, label=f"the loss for target {target}")
        close(grad, x.grad.numpy(), atol=1e-9, label=f"d loss / d logits for target {target}")


@check("ctc_grad matches finite differences, and each row of it sums to zero")
def test_grad_fd(sub):
    ctc_grad, ctc_loss = has(sub, "ctc_grad"), has(sub, "ctc_loss")
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(5, 3))
    target = [1, 1]
    _, grad = ctc_grad(logits, target)
    numeric = finite_difference(lambda u: ctc_loss(log_softmax(u, axis=1), target), logits.copy(), eps=1e-6)
    close(grad, numeric, atol=1e-6, label="analytic against central-difference gradient")
    close(np.asarray(grad).sum(axis=1), np.zeros(5), atol=1e-12,
          label="row sums, softmax minus an occupancy that sums to 1")


@check("ctc_loss_batch matches torch on a padded (T, N, C) batch for 'none', 'sum', and 'mean'")
def test_batch(sub):
    ctc_loss_batch = has(sub, "ctc_loss_batch")
    rng = np.random.default_rng(5)
    T, N, C = 16, 4, 6
    log_probs = log_softmax(rng.normal(size=(T, N, C)) * 1.5, axis=2)
    input_lengths = np.array([16, 12, 9, 14])
    target_lengths = np.array([5, 3, 1, 4])
    targets = np.zeros((N, 5), dtype=int)
    for n, length in enumerate(target_lengths):
        targets[n, :length] = rng.integers(1, C, size=length)
    targets[0, :5] = [2, 2, 3, 3, 1]  # repeats need blanks
    for reduction in ("none", "sum", "mean"):
        want = F.ctc_loss(torch.tensor(log_probs), torch.tensor(targets), torch.tensor(input_lengths),
                          torch.tensor(target_lengths), blank=0, reduction=reduction).numpy()
        got = np.asarray(ctc_loss_batch(log_probs, targets, input_lengths, target_lengths, 0, reduction), dtype=float)
        close(got, want, rtol=1e-9, atol=1e-9, label=f"reduction='{reduction}'")


@check("Gradient descent with your ctc_grad teaches free logits to spell a target with a repeat")
def test_training(sub):
    ctc_grad, collapse = has(sub, "ctc_grad"), has(sub, "collapse")
    logits = np.zeros((12, 4))
    target = [1, 2, 2, 3]
    history = []
    with timed("300 gradient steps"):
        for _ in range(300):
            loss, grad = ctc_grad(logits, target)
            logits -= 1.0 * np.asarray(grad)
            history.append(loss)
    if not history[-1] < 0.05:
        fail(f"after 300 steps the loss is {history[-1]:.4f} (it started at {history[0]:.4f}); "
             "expected below 0.05")
    decoded = [int(v) for v in collapse(np.argmax(logits, axis=1))]
    if decoded != target:
        fail(f"the trained logits decode greedily to {decoded}, expected {target}")
