"""Fine-tune a tiny encoder: a low-rank update, a freeze mask, and a seeded CTC gap."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from speechlab import show_plot


FEATURE_PARTS = ("feature_encoder", "feature_extractor")
HEAD_PARTS = ("ctc_head", "lm_head")

FEAT = 4
HIDDEN = 8
CLASSES = 4
FRAMES = 8
PROTO = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


class LoRALinear:
    """A frozen dense map plus a rank-r update, scaled by alpha / r.

    weight is (out, in) and stays frozen. A is (r, in) and B is (out, r).
    Both A and B train. forward(x) takes (..., in) and returns (..., out).
    merge() returns the (out, in) matrix stored at inference.
    trainable_count() is the number of entries in A and B.
    """

    def __init__(self, weight, A, B, alpha: float):
        """Store weight, A, B, and alpha. Rank r is A.shape[0]."""
        self.weight = np.asarray(weight, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.alpha = float(alpha)

    def forward(self, x):
        """Return x @ W.T + (alpha / r) * (x @ A.T) @ B.T."""
        features = np.asarray(x, dtype=float)
        scale = self.alpha / self.A.shape[0]
        dense = features @ self.weight.T
        update = (features @ self.A.T) @ self.B.T
        return dense + scale * update

    def merge(self):
        """Return W + (alpha / r) * (B @ A), shape (out, in)."""
        scale = self.alpha / self.A.shape[0]
        return self.weight + scale * (self.B @ self.A)

    def trainable_count(self) -> int:
        """Return the number of trainable entries, A.size + B.size."""
        return int(self.A.size + self.B.size)


def freeze_mask(names: list[str], policy: str) -> dict[str, bool]:
    """Return name -> True when that parameter trains.

    policy "freeze_feature_encoder": False when the name contains
    feature_encoder or feature_extractor, True otherwise.
    policy "ctc_head_only": True only when the name contains ctc_head
    or lm_head.
    """
    if policy == "freeze_feature_encoder":
        return {
            name: not any(part in name for part in FEATURE_PARTS) for name in names
        }
    if policy == "ctc_head_only":
        return {name: any(part in name for part in HEAD_PARTS) for name in names}
    raise ValueError(f"unknown policy {policy}")


def forgetting_score(loss_before: float, loss_after: float) -> float:
    """How far the original-domain loss moved. Positive means it rose."""
    return float(loss_after) - float(loss_before)


class _TinyCTC(nn.Module):
    """Linear encoder, tanh, linear CTC head. Biases included."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(FEAT, HIDDEN)
        self.ctc_head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ctc_head(torch.tanh(self.encoder(features)))


def _batch(n: int, seed: int, noise: float = 0.1):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, CLASSES, (n, 2), generator=generator)
    features = torch.zeros(n, FRAMES, FEAT)
    features[:, 0:3] = PROTO[labels[:, 0]].unsqueeze(1)
    features[:, 4:7] = PROTO[labels[:, 1]].unsqueeze(1)
    features = features + noise * torch.randn(n, FRAMES, FEAT, generator=generator)
    return features, labels


def _ctc(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(-1).transpose(0, 1)
    batch, time, _ = logits.shape
    input_lengths = torch.full((batch,), time, dtype=torch.long)
    target_lengths = torch.full((batch,), targets.shape[1], dtype=torch.long)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )


def _train(model, features, labels, steps, lr, eval_features=None, eval_labels=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    curve = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _ctc(model(features), labels)
        loss.backward()
        optimizer.step()
        if eval_features is not None:
            with torch.no_grad():
                curve.append(float(_ctc(model(eval_features), eval_labels)))
    return curve


def _run(seed: int):
    pre_x, pre_y = _batch(64, seed + 1)
    few_x, few_y = _batch(8, seed + 2)
    held_x, held_y = _batch(32, seed + 3)

    torch.manual_seed(seed)
    pretrained = _TinyCTC()
    _train(pretrained, pre_x, pre_y, 60, 0.5)
    with torch.no_grad():
        source_before = float(_ctc(pretrained(pre_x), pre_y))
    saved = {name: value.detach().clone() for name, value in pretrained.state_dict().items()}

    finetuned = _TinyCTC()
    finetuned.load_state_dict(saved)
    finetune_curve = _train(finetuned, few_x, few_y, 10, 0.5, held_x, held_y)

    torch.manual_seed(seed)
    scratch = _TinyCTC()
    scratch_curve = _train(scratch, few_x, few_y, 10, 0.5, held_x, held_y)

    conflict = _TinyCTC()
    conflict.load_state_dict(saved)
    cycled = pre_y % 3 + 1
    _train(conflict, pre_x, cycled, 60, 0.5)
    with torch.no_grad():
        source_after = float(_ctc(conflict(pre_x), pre_y))
    return {
        "finetune": finetune_curve[-1],
        "scratch": scratch_curve[-1],
        "before": source_before,
        "after": source_after,
        "finetune_curve": finetune_curve,
        "scratch_curve": scratch_curve,
    }


def adaptation_losses(seed: int) -> tuple[float, float, float, float]:
    """Few-shot loss, from-scratch loss, and the source loss before and after a conflicting update.

    Returns (finetune_loss, scratch_loss, source_before, source_after).
    Clips use Generator seeds seed+1, seed+2, and seed+3. Construct each
    fresh model immediately after torch.manual_seed(seed).
    """
    result = _run(seed)
    return (result["finetune"], result["scratch"], result["before"], result["after"])


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    print(f"cuda: {cuda}")
    print(
        "A wav2vec2-base fine-tune freezes the feature extractor and trains a CTC head "
        "on the GPU worker. This run stays on CPU and does not download weights."
    )
    weight = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    matrix_a = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    matrix_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    features = np.array([1.0, 2.0, 3.0, 4.0])
    layer = LoRALinear(weight, matrix_a, matrix_b, alpha=2.0)
    print("forward", layer.forward(features))
    print("trainable", layer.trainable_count())
    result = _run(0)
    print(
        f"finetune {result['finetune']:.6f}  scratch {result['scratch']:.6f}"
    )
    print(
        f"source before {result['before']:.6f}  after {result['after']:.6f}  "
        f"forgetting {forgetting_score(result['before'], result['after']):.6f}"
    )
    steps = np.arange(1, 11)
    show_plot(
        (steps, np.array(result["finetune_curve"])),
        (steps, np.array(result["scratch_curve"])),
        label="Held-out CTC loss",
        xlabel="adaptation step",
        ylabel="loss",
        labels=["fine-tune", "from scratch"],
    )
