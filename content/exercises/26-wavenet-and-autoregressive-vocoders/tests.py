"""Checks for the WaveNet exercise."""

import numpy as np
import torch
import torch.nn.functional as F

from speechcheck import check, close, equal, fail, has, in_range, snr_db, tone


def _model(sub, dilations=(1, 2, 4, 8), channels=6, skip=5, seed=0):
    torch.manual_seed(seed)
    model = has(sub, "WaveNet")(channels=channels, skip=skip, dilations=dilations).double()
    # Larger random weights make every dependency visible in the impulse test.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)
    return model.eval()


def _reference_forward(model, codes):
    """The WaveNet forward pass written out from the model's weights."""
    x = model.embed.weight[codes].transpose(1, 2)
    skips = 0
    for d, conv, res, skip in zip(model.dilations, model.dilated, model.residual, model.skip):
        a = F.conv1d(F.pad(x, (d, 0)), conv.weight, conv.bias, dilation=d)
        c = a.shape[1] // 2
        z = torch.tanh(a[:, :c]) * torch.sigmoid(a[:, c:])
        x = x + F.conv1d(z, res.weight, res.bias)
        skips = skips + F.conv1d(z, skip.weight, skip.bias)
    h = F.relu(F.conv1d(F.relu(skips), model.head1.weight, model.head1.bias))
    return F.conv1d(h, model.head2.weight, model.head2.bias)


@check("receptive_field gives 3,070 samples for three stacks of dilations 1 to 512")
def test_receptive_field(sub):
    fn = has(sub, "receptive_field")
    stack = [2**m for m in range(10)]
    equal(fn(stack * 3, 2), 3070, label="three stacks of 1..512 with kernel 2")
    equal(fn(stack, 2), 1024, label="one stack of 1..512 with kernel 2")
    equal(fn([1, 2, 4], 3), 15, label="dilations 1, 2, 4 with kernel 3")


@check("mu_law_encode gives the hand-worked codes and spans 0 to 255")
def test_mu_law_encode(sub):
    fn = has(sub, "mu_law_encode")
    got = np.asarray(fn(np.array([0.5, 0.001, -0.01, 0.0, -1.0, 1.0])))
    equal(got, [239, 133, 98, 128, 0, 255], label="codes for 0.5, 0.001, -0.01, 0, -1, and 1")
    if not np.issubdtype(got.dtype, np.integer):
        fail(f"codes must be integers, got dtype {got.dtype}")


@check("mu_law_decode inverts the encoder, and mu-law beats 8-bit linear on a quiet tone")
def test_mu_law_round_trip(sub):
    enc = has(sub, "mu_law_encode")
    dec = has(sub, "mu_law_decode")
    close(dec(np.array([0, 128, 255])), [-1.0, 8.62116e-05, 1.0], atol=1e-9, label="decoded codes 0, 128, 255")
    x = np.linspace(-0.99, 0.99, 1001)
    back = dec(enc(x))
    worst = np.max(np.abs(back - x) / (np.abs(x) + 0.004))
    if worst > 0.04:
        fail(f"decode(encode(x)) should be within about 3% of x (plus a small floor near 0), "
             f"but the worst relative error is {worst:.3f}")
    quiet = tone(440, 8000, 0.25, amplitude=0.01)
    linear = np.round((quiet + 1) / 2 * 255) / 255 * 2 - 1
    mu_snr, lin_snr = snr_db(dec(enc(quiet)), quiet), snr_db(linear, quiet)
    if mu_snr < lin_snr + 15:
        fail(f"on a tone of amplitude 0.01, mu-law should beat 8-bit linear quantization by "
             f"at least 15 dB; got {mu_snr:.1f} dB against {lin_snr:.1f} dB")


@check("causal_conv1d matches a loop over past samples and keeps the length")
def test_causal_conv(sub):
    fn = has(sub, "causal_conv1d")
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(2, 3, 20)))
    w = torch.tensor(rng.normal(size=(4, 3, 3)))
    b = torch.tensor(rng.normal(size=4))
    d = 2
    y = fn(x, w, b, d)
    want = np.zeros((2, 4, 20))
    xn, wn = x.numpy(), w.numpy()
    for t in range(20):
        for k in range(3):
            src = t - (2 - k) * d  # tap k looks (K - 1 - k) * d samples back
            if src >= 0:
                want[:, :, t] += np.einsum("oc,bc->bo", wn[:, :, k], xn[:, :, src])
    want += b.numpy()[None, :, None]
    close(y, want, atol=1e-10, label="causal convolution with kernel 3 and dilation 2")


@check("gated_activation is tanh of the first half of the channels times sigmoid of the second")
def test_gate(sub):
    fn = has(sub, "gated_activation")
    x = torch.tensor(np.random.default_rng(2).normal(size=(2, 8, 5)))
    close(fn(x), torch.tanh(x[:, :4]) * torch.sigmoid(x[:, 4:]), atol=1e-12, label="the gated activation")


@check("WaveNet.forward matches the architecture written out from its weights")
def test_forward_reference(sub):
    model = _model(sub)
    codes = torch.tensor(np.random.default_rng(3).integers(0, 256, (2, 40)))
    with torch.no_grad():
        got = model(codes)
        want = _reference_forward(model, codes)
    close(got, want, atol=1e-9, label="logits, shape (batch, 256, time)")


@check("An impulse reaches exactly the receptive field: nothing before it, nothing past t + R - 1")
def test_impulse(sub):
    model = _model(sub, dilations=(1, 2, 4, 8, 1, 2))
    rf = has(sub, "receptive_field")(model.dilations, 2)
    rng = np.random.default_rng(4)
    codes = torch.tensor(rng.integers(0, 256, (1, 60)))
    t0 = 20
    bumped = codes.clone()
    bumped[0, t0] = (bumped[0, t0] + 97) % 256
    with torch.no_grad():
        diff = (model(bumped) - model(codes)).abs().amax(dim=1)[0].numpy()
    changed = np.nonzero(diff > 1e-9)[0]
    want = np.arange(t0, t0 + rf)
    if changed.size == 0 or changed.min() < t0:
        fail(f"changing the input at t = {t0} changed outputs at {changed.tolist()}; "
             f"no output before t = {t0} may change, or the model sees the future")
    if changed.tolist() != want.tolist():
        fail(f"changing the input at t = {t0} should change exactly the outputs at t = {t0} to "
             f"{t0 + rf - 1} (a receptive field of {rf}); it changed {changed.tolist()}")


@check("WaveNet.step with its cache reproduces the full forward pass sample by sample")
def test_cache(sub):
    model = _model(sub, dilations=(1, 2, 4, 8, 1, 2))
    codes = torch.tensor(np.random.default_rng(5).integers(0, 256, (2, 30)))
    with torch.no_grad():
        full = model(codes)
        cache = [c.double() for c in model.init_cache(2)]
        steps = []
        for t in range(codes.shape[1]):
            logits, cache = model.step(codes[:, t], cache)
            steps.append(logits)
    close(torch.stack(steps, dim=2), full, atol=1e-9, label="logits from 30 cached steps")


@check("generate with temperature 0 follows the argmax of the full model, step by step")
def test_generate(sub):
    fn = has(sub, "generate")
    model = _model(sub, dilations=(1, 2, 4), seed=6)
    n = 25
    got = np.asarray(fn(model, n, temperature=0.0, seed=0, first=128))
    equal(got.shape, (n,), label="the number of generated codes")
    seq = [128]
    with torch.no_grad():
        for _ in range(n):
            logits = model(torch.tensor([seq]))
            seq.append(int(torch.argmax(logits[0, :, -1])))
    equal(got, seq[1:], label="greedy codes against re-running the full model on each prefix")
    a = np.asarray(fn(model, 50, temperature=1.0, seed=3))
    b = np.asarray(fn(model, 50, temperature=1.0, seed=3))
    equal(a, b, label="two sampled runs with the same seed")
    in_range(a, 0, 255, label="sampled codes")
