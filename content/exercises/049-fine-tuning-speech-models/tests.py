"""Checks for fine-tuning speech models.

Wrong implementation `merge_low_rank_product_only`: merge returns
(alpha / r) * (B @ A) and drops the frozen weight. On the worked
example that product is
[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], and the forward value
of the product alone is [2, 1, 0]. The merged matrix must be the
frozen weight plus that product, whose forward value is [3, 3, 3].

Wrong implementation `train_feature_encoder_too`: freeze_mask marks
every name trainable, including
wav2vec2.feature_extractor.conv_layers.0.conv.weight, which
freeze_feature_encoder must set to False.
"""

import numpy as np

from speechcheck import almost, check, close, fail


WEIGHT = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
)
MATRIX_A = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
MATRIX_B = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
FEATURES = np.array([1.0, 2.0, 3.0, 4.0])

NAMES = [
    "wav2vec2.feature_extractor.conv_layers.0.conv.weight",
    "wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias",
    "wav2vec2.feature_encoder.pos_conv.0.weight",
    "wav2vec2.encoder.layers.0.attention.k_proj.weight",
    "wav2vec2.encoder.layers.11.feed_forward.output_dense.weight",
    "ctc_head.weight",
    "ctc_head.bias",
    "lm_head.weight",
]

# Seed 0 of the procedure in the lesson, from scripts/figures/fine-tuning-speech-models.py.
FINETUNE_LOSS = 0.103521
SCRATCH_LOSS = 1.566033
SOURCE_BEFORE = 0.146689
SOURCE_AFTER = 10.359814


def _scale(alpha: float) -> float:
    return alpha / MATRIX_A.shape[0]


def _dense(features: np.ndarray, alpha: float) -> np.ndarray:
    scale = _scale(alpha)
    return features @ WEIGHT.T + scale * (features @ MATRIX_A.T) @ MATRIX_B.T


def _merged(alpha: float) -> np.ndarray:
    return WEIGHT + _scale(alpha) * (MATRIX_B @ MATRIX_A)


def _forward(sub, features, alpha, expected):
    try:
        layer = sub.LoRALinear(WEIGHT, MATRIX_A, MATRIX_B, alpha)
        return layer.forward(features)
    except NotImplementedError:
        fail(
            "LoRALinear.forward is not implemented. "
            f"expected {np.array2string(expected, precision=6)}, got no result"
        )


def _merge(sub, alpha, expected):
    try:
        layer = sub.LoRALinear(WEIGHT, MATRIX_A, MATRIX_B, alpha)
        return layer.merge()
    except NotImplementedError:
        fail(
            "LoRALinear.merge is not implemented. "
            f"expected {np.array2string(expected, precision=6)}, got no result"
        )


def _count(sub, expected):
    try:
        layer = sub.LoRALinear(WEIGHT, MATRIX_A, MATRIX_B, 2.0)
        return layer.trainable_count()
    except NotImplementedError:
        fail(
            "LoRALinear.trainable_count is not implemented. "
            f"expected {expected}, got no result"
        )


def _mask(sub, policy, expected):
    try:
        return sub.freeze_mask(list(NAMES), policy)
    except NotImplementedError:
        fail(
            f"freeze_mask is not implemented. expected {expected}, got no result"
        )


def _losses(sub):
    expected = (FINETUNE_LOSS, SCRATCH_LOSS, SOURCE_BEFORE, SOURCE_AFTER)
    try:
        return sub.adaptation_losses(0)
    except NotImplementedError:
        fail(
            "adaptation_losses is not implemented. "
            f"expected {expected}, got no result"
        )


def _score(sub, before, after, expected):
    try:
        return sub.forgetting_score(before, after)
    except NotImplementedError:
        fail(
            "forgetting_score is not implemented. "
            f"expected {expected}, got no result"
        )


@check("LoRALinear.forward matches the frozen map plus the scaled product")
def test_forward(sub):
    if not hasattr(sub, "LoRALinear"):
        fail(
            "your module defines no LoRALinear. "
            "expected forward [3, 3, 3], got no class"
        )
    got = _forward(sub, FEATURES, 2.0, np.array([3.0, 3.0, 3.0]))
    close(got, _dense(FEATURES, 2.0), label="alpha 2")
    scaled = _forward(sub, FEATURES, 4.0, np.array([5.0, 4.0, 3.0]))
    close(scaled, _dense(FEATURES, 4.0), label="alpha 4")
    batch = np.stack([FEATURES, 2.0 * FEATURES])
    close(_forward(sub, batch, 2.0, _dense(batch, 2.0)), _dense(batch, 2.0), label="batch")


@check("LoRALinear.merge adds the scaled product onto the frozen weight")
def test_merge(sub):
    # `merge_low_rank_product_only` fails here: it returns the product
    # and drops WEIGHT.
    if not hasattr(sub, "LoRALinear"):
        fail(
            "your module defines no LoRALinear. "
            "expected merge of weight plus B @ A, got no class"
        )
    got = np.asarray(_merge(sub, 2.0, _merged(2.0)), dtype=float)
    wrong = _scale(2.0) * (MATRIX_B @ MATRIX_A)
    if np.allclose(got, wrong):
        fail(
            "merge_low_rank_product_only dropped the frozen weight. "
            f"got {np.array2string(got, precision=6)}, "
            f"expected {np.array2string(_merged(2.0), precision=6)}"
        )
    close(got, _merged(2.0), label="alpha 2")
    close(_merge(sub, 4.0, _merged(4.0)), _merged(4.0), label="alpha 4")
    features = FEATURES
    close(features @ got.T, _dense(features, 2.0), label="merged times x")


@check("LoRALinear.trainable_count is the entries in A and B")
def test_count(sub):
    if not hasattr(sub, "LoRALinear"):
        fail(
            "your module defines no LoRALinear. expected trainable count 14, got no class"
        )
    got = _count(sub, 14)
    if got == int(WEIGHT.size + MATRIX_A.size + MATRIX_B.size):
        fail(
            "trainable_count includes the frozen weight. "
            f"got {got}, expected {int(MATRIX_A.size + MATRIX_B.size)}"
        )
    if int(got) != int(MATRIX_A.size + MATRIX_B.size):
        fail(
            "trainable_count "
            f"got {got}, expected {int(MATRIX_A.size + MATRIX_B.size)}"
        )
    wide_a = np.zeros((8, 768))
    wide_b = np.zeros((768, 8))
    try:
        wide = sub.LoRALinear(np.zeros((768, 768)), wide_a, wide_b, 8.0)
        wide_count = wide.trainable_count()
    except NotImplementedError:
        fail(
            "LoRALinear.trainable_count is not implemented. "
            "expected 12288, got no result"
        )
    if int(wide_count) != 12288:
        fail(f"rank-8 768 projection got {wide_count}, expected 12288")


@check("freeze_mask freezes the feature encoder and trains the rest")
def test_freeze_encoder(sub):
    # `train_feature_encoder_too` fails here: the conv weight stays trainable.
    if not hasattr(sub, "freeze_mask"):
        fail(
            "your module defines no freeze_mask. "
            "expected False for the feature extractor conv weight, got no function"
        )
    expected = {
        name: not any(
            part in name for part in ("feature_encoder", "feature_extractor")
        )
        for name in NAMES
    }
    got = _mask(sub, "freeze_feature_encoder", expected)
    conv = "wav2vec2.feature_extractor.conv_layers.0.conv.weight"
    if isinstance(got, dict) and got.get(conv) is True:
        fail(
            "train_feature_encoder_too left the feature extractor trainable. "
            f"got True for {conv}, expected False"
        )
    if got != expected:
        fail(
            "freeze_feature_encoder differs.\n"
            f"  got:  {got}\n"
            f"  want: {expected}"
        )


@check("freeze_mask ctc_head_only trains the CTC head")
def test_head_only(sub):
    if not hasattr(sub, "freeze_mask"):
        fail(
            "your module defines no freeze_mask. "
            "expected True for ctc_head.weight, got no function"
        )
    expected = {
        name: any(part in name for part in ("ctc_head", "lm_head"))
        for name in NAMES
    }
    got = _mask(sub, "ctc_head_only", expected)
    if got != expected:
        fail(
            "ctc_head_only differs.\n"
            f"  got:  {got}\n"
            f"  want: {expected}"
        )


@check("few-shot fine-tuning beats training from scratch on the tones")
def test_adaptation(sub):
    if not hasattr(sub, "adaptation_losses"):
        fail(
            "your module defines no adaptation_losses. "
            f"expected finetune {FINETUNE_LOSS:.6f} and scratch {SCRATCH_LOSS:.6f}, "
            "got no function"
        )
    got = _losses(sub)
    if not isinstance(got, tuple) or len(got) != 4:
        fail(
            "adaptation_losses returned "
            f"{got!r}, expected four floats "
            f"({FINETUNE_LOSS:.6f}, {SCRATCH_LOSS:.6f}, "
            f"{SOURCE_BEFORE:.6f}, {SOURCE_AFTER:.6f})"
        )
    finetune, scratch, before, after = (float(value) for value in got)
    almost(finetune, FINETUNE_LOSS, 1e-4, label="finetune loss")
    almost(scratch, SCRATCH_LOSS, 1e-4, label="scratch loss")
    if not finetune + 1.0 < scratch:
        fail(
            f"few-shot loss is {finetune:.6f}, "
            f"expected it below scratch {scratch:.6f} by more than 1"
        )
    almost(before, SOURCE_BEFORE, 1e-4, label="source loss before")
    almost(after, SOURCE_AFTER, 1e-4, label="source loss after")
    if not after > before + 1.0:
        fail(
            f"source loss after is {after:.6f}, "
            f"expected it to rise above {before:.6f}"
        )


@check("forgetting_score is the rise on the planted pair")
def test_forgetting(sub):
    if not hasattr(sub, "forgetting_score"):
        fail(
            "your module defines no forgetting_score. "
            "expected 10.213125 on the planted pair, got no function"
        )
    planted = _score(sub, SOURCE_BEFORE, SOURCE_AFTER, SOURCE_AFTER - SOURCE_BEFORE)
    almost(planted, SOURCE_AFTER - SOURCE_BEFORE, 1e-6, label="planted rise")
    improved = _score(sub, 0.5, 0.2, -0.3)
    almost(improved, -0.3, 1e-9, label="a fall is negative")
    same = _score(sub, 1.25, 1.25, 0.0)
    almost(same, 0.0, 1e-9, label="no change")
