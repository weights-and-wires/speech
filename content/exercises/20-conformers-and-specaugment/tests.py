"""Checks for the Conformers-and-SpecAugment exercise. They run on the CPU with no assets."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from speechcheck import almost, check, close, equal, fail, has, shape


def masked_rows_cols(out):
    zero = out == 0
    rows = zero.all(dim=1)
    cols = zero.all(dim=0)
    return zero, rows, cols


@check("spec_augment zeroes whole mel bands and whole frames, within the limits, on a copy")
def test_specaug_structure(sub):
    aug = has(sub, "spec_augment")
    for seed in range(20):
        x = torch.ones(80, 200)
        out = aug(x, 27, 2, 40, 1.0, 2, torch.Generator().manual_seed(seed))
        if not torch.equal(x, torch.ones(80, 200)):
            fail("spec_augment changed its input; mask a clone")
        shape(out, (80, 200), label="the masked features")
        zero, rows, cols = masked_rows_cols(out)
        union = rows[:, None] | cols[None, :]
        if not torch.equal(zero, union):
            fail(f"seed {seed}: some zeroed cells are not part of a whole masked band or a whole masked frame")
        if int(rows.sum()) > 2 * 27:
            fail(f"seed {seed}: {int(rows.sum())} bands are masked; two masks of at most 27 bands cover at most 54")
        if int(cols.sum()) > 2 * 40:
            fail(f"seed {seed}: {int(cols.sum())} frames are masked; two masks of at most 40 frames cover at most 80")
    same = aug(torch.ones(80, 200), 27, 2, 40, 1.0, 2, torch.Generator().manual_seed(3))
    again = aug(torch.ones(80, 200), 27, 2, 40, 1.0, 2, torch.Generator().manual_seed(3))
    equal(bool(torch.equal(same, again)), True, label="two runs with the same seed agree")


@check("spec_augment keeps time masks inside the true length and under time_ratio times it")
def test_specaug_length(sub):
    aug = has(sub, "spec_augment")
    widest = 0
    for seed in range(40):
        x = torch.ones(40, 150)
        out = aug(x, 0, 0, 40, 0.05, 1, torch.Generator().manual_seed(seed), length=100)
        _, _, cols = masked_rows_cols(out)
        if bool(cols[100:].any()):
            fail(f"seed {seed}: a time mask reached past the true length of 100 frames into the padding")
        widest = max(widest, int(cols.sum()))
    if widest > 5:
        fail(f"a time mask covered {widest} frames; with time_ratio 0.05 and length 100 the widest allowed is 5")
    if widest == 0:
        fail("40 draws never masked a single frame; the width should be uniform on 0 to 5")


@check("spec_augment draws mask widths uniformly from 0 to F: the mean width of one mask is near F / 2")
def test_specaug_stats(sub):
    aug = has(sub, "spec_augment")
    widths = []
    top_touched = False
    for seed in range(800):
        out = aug(torch.ones(80, 10), 27, 1, 0, 1.0, 0, torch.Generator().manual_seed(seed))
        _, rows, _ = masked_rows_cols(out)
        widths.append(int(rows.sum()))
        top_touched |= bool(rows[-1])
    mean = float(np.mean(widths))
    if abs(mean - 13.5) > 1.5:
        fail(f"the mean width of one frequency mask with F = 27 is {mean:.2f}; a uniform draw on 0 to 27 gives 13.5")
    if max(widths) > 27:
        fail(f"a single mask covered {max(widths)} bands, more than F = 27")
    if not top_touched:
        fail("in 800 draws no mask reached the top band; the start f0 should be uniform on 0 to n_mels - f")


@check("relative_position_encoding gives sin and cos of r / 10000^(2i/d) for r = L - 1 down to -(L - 1)")
def test_pos_encoding(sub):
    enc = has(sub, "relative_position_encoding")
    L, d = 5, 8
    pe = torch.as_tensor(enc(L, d)).double()
    shape(pe, (2 * L - 1, d), label="the encoding")
    r = np.arange(L - 1, -L, -1)[:, None]
    rate = 10000.0 ** (-np.arange(0, d, 2) / d)
    want = np.zeros((2 * L - 1, d))
    want[:, 0::2] = np.sin(r * rate)
    want[:, 1::2] = np.cos(r * rate)
    close(pe.numpy(), want, atol=1e-6, label="the encoding table")
    close(pe[L - 1].numpy(), np.tile([0.0, 1.0], d // 2), atol=1e-7, label="the row for distance 0")


@check("rel_shift moves the score for distance i - j to column j: out[i, j] = x[i, L - 1 - i + j]")
def test_rel_shift(sub):
    shift = has(sub, "rel_shift")
    torch.manual_seed(0)
    L = 5
    x = torch.randn(2, 3, L, 2 * L - 1)
    out = shift(x)
    shape(out, (2, 3, L, L), label="the shifted scores")
    idx = torch.tensor([[L - 1 - i + j for j in range(L)] for i in range(L)])
    want = x.gather(-1, idx.expand(2, 3, L, L))
    close(out.numpy(), want.numpy(), atol=0.0, label="the shifted scores")


def mhsa_reference(m, x, pe, mask):
    """Brute force: score(i, j) = ((q_i + u) . k_j + (q_i + v) . p_{i - j}) / sqrt(d_k), per head."""
    B, T, d = x.shape
    H, dk = m.pos_bias_u.shape
    q = (x @ m.linear_q.weight.T + m.linear_q.bias).view(B, T, H, dk)
    k = (x @ m.linear_k.weight.T + m.linear_k.bias).view(B, T, H, dk)
    v = (x @ m.linear_v.weight.T + m.linear_v.bias).view(B, T, H, dk)
    p = (pe @ m.linear_pos.weight.T).view(2 * T - 1, H, dk)
    out = torch.zeros(B, T, H, dk, dtype=x.dtype)
    for b in range(B):
        for h in range(H):
            scores = torch.full((T, T), float("-inf"), dtype=x.dtype)
            for i in range(T):
                for j in range(T):
                    if mask[b, j]:
                        rel = p[T - 1 - (i - j), h]
                        scores[i, j] = ((q[b, i, h] + m.pos_bias_u[h]) @ k[b, j, h] + (q[b, i, h] + m.pos_bias_v[h]) @ rel) / math.sqrt(dk)
            out[b, :, h] = torch.softmax(scores, -1) @ v[b, :, h]
    return out.reshape(B, T, H * dk) @ m.linear_out.weight.T + m.linear_out.bias


def sinusoids(L, d):
    r = torch.arange(L - 1, -L, -1, dtype=torch.float64)[:, None]
    rate = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    pe = torch.zeros(2 * L - 1, d, dtype=torch.float64)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(r * rate), torch.cos(r * rate)
    return pe


@check("RelPositionMultiHeadAttention matches a brute-force loop over heads, queries, and keys, with a mask")
def test_mhsa(sub):
    cls = has(sub, "RelPositionMultiHeadAttention")
    torch.manual_seed(1)
    m = cls(8, 2, dropout=0.0).double().eval()
    with torch.no_grad():
        m.pos_bias_u.normal_()
        m.pos_bias_v.normal_()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    pe = sinusoids(5, 8)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    with torch.no_grad():
        got = m(x, pe, mask)
        want = mhsa_reference(m, x, pe, mask)
    shape(got, (2, 5, 8), label="the attention output")
    close(got.numpy(), want.numpy(), atol=1e-10, label="the attention output")


@check("FeedForwardModule and ConvolutionModule match their layer-by-layer definitions")
def test_modules(sub):
    ff_cls = has(sub, "FeedForwardModule")
    conv_cls = has(sub, "ConvolutionModule")
    torch.manual_seed(2)
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    ff = ff_cls(8, dropout=0.0).double().eval()
    with torch.no_grad():
        want = ff.linear2(F.silu(ff.linear1(F.layer_norm(x, (8,), ff.norm.weight, ff.norm.bias))))
        close(ff(x).numpy(), want.numpy(), atol=1e-10, label="the feed-forward module")
    conv = conv_cls(8, kernel_size=5, dropout=0.0).double().eval()
    with torch.no_grad():
        conv.batch_norm.running_mean.normal_()
        conv.batch_norm.running_var.uniform_(0.5, 2.0)
        y = F.layer_norm(x, (8,), conv.norm.weight, conv.norm.bias).transpose(1, 2)
        y = F.glu(F.conv1d(y, conv.pointwise1.weight, conv.pointwise1.bias), dim=1)
        y = F.conv1d(y, conv.depthwise.weight, conv.depthwise.bias, padding=2, groups=8)
        bn = conv.batch_norm
        y = (y - bn.running_mean[:, None]) / torch.sqrt(bn.running_var[:, None] + bn.eps) * bn.weight[:, None] + bn.bias[:, None]
        y = F.conv1d(F.silu(y), conv.pointwise2.weight, conv.pointwise2.bias).transpose(1, 2)
        close(conv(x).numpy(), y.numpy(), atol=1e-10, label="the convolution module")


@check("ConvolutionModule with a padding mask gives real frames the same output as the unpadded utterance")
def test_conv_padding(sub):
    conv_cls = has(sub, "ConvolutionModule")
    torch.manual_seed(3)
    conv = conv_cls(8, kernel_size=5, dropout=0.0).double().eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64)
    padded = torch.cat([x, 100 * torch.randn(1, 4, 8, dtype=torch.float64)], dim=1)
    mask = torch.tensor([[True] * 6 + [False] * 4])
    with torch.no_grad():
        alone = conv(x, torch.ones(1, 6, dtype=torch.bool))
        batched = conv(padded, mask)
    close(batched[:, :6].numpy(), alone.numpy(), atol=1e-10, label="real frames next to large padding values")


@check("ConformerBlock wires FFN/2, attention, convolution, FFN/2, and LayerNorm, and ignores padding")
def test_block(sub):
    block_cls = has(sub, "ConformerBlock")
    torch.manual_seed(4)
    block = block_cls(8, 2, kernel_size=5, dropout=0.0).double().eval()
    x = torch.randn(1, 6, 8, dtype=torch.float64)
    pe = sinusoids(6, 8)
    full = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        h = x + 0.5 * block.ff1(x)
        h = h + block.mhsa(block.norm_mhsa(h), pe, full)
        h = h + block.conv(h, full)
        h = h + 0.5 * block.ff2(h)
        want = block.norm_out(h)
        got = block(x, pe, full)
    close(got.numpy(), want.numpy(), atol=1e-10, label="the block output against its definition")
    padded = torch.cat([x, torch.randn(1, 3, 8, dtype=torch.float64)], dim=1)
    mask = torch.tensor([[True] * 6 + [False] * 3])
    with torch.no_grad():
        batched = block(padded, sinusoids(9, 8), mask)
    close(batched[:, :6].numpy(), got.numpy(), atol=1e-9, label="real frames of a padded batch against the utterance alone")
