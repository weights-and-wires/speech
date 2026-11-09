#!/usr/bin/env python3
"""Compute the fine-tuning lesson's numbers and draw the loss-gap figure.

The numbers in content/lessons/041-fine-tuning-speech-models.md come from
this script. Run it from the repo root:

    python3 scripts/figures/fine-tuning-speech-models.py

It prints the worked LoRA product, the trainable count on one 768 by 768
projection, the seeded CTC losses, and the SVG for Figure 1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TinyCTC(nn.Module):
    """Linear encoder, tanh, linear CTC head. Biases included. 76 parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(FEAT, HIDDEN)
        self.ctc_head = nn.Linear(HIDDEN, CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ctc_head(torch.tanh(self.encoder(features)))


def tone_batch(n: int, seed: int, noise: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """n clips of 8 frames. Labels are two classes in 1, 2, 3. Blank frames are zeros."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(1, CLASSES, (n, 2), generator=generator)
    features = torch.zeros(n, FRAMES, FEAT)
    features[:, 0:3] = PROTO[labels[:, 0]].unsqueeze(1)
    features[:, 4:7] = PROTO[labels[:, 1]].unsqueeze(1)
    features = features + noise * torch.randn(
        n, FRAMES, FEAT, generator=generator
    )
    return features, labels


def ctc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean CTC loss. logits are (batch, time, classes), blank is class 0."""
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


def train_steps(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    lr: float,
    eval_features: torch.Tensor | None = None,
    eval_labels: torch.Tensor | None = None,
) -> list[float]:
    """Full-batch SGD. When an eval batch is set, record its loss after each step."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    curve: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ctc_loss(model(features), labels)
        loss.backward()
        optimizer.step()
        if eval_features is not None and eval_labels is not None:
            with torch.no_grad():
                curve.append(float(ctc_loss(model(eval_features), eval_labels)))
    return curve


def adaptation_curves(seed: int = 0) -> dict[str, float | list[float]]:
    """The seeded run the lesson quotes. Data seeds are seed+1, seed+2, seed+3."""
    pre_x, pre_y = tone_batch(64, seed + 1)
    few_x, few_y = tone_batch(8, seed + 2)
    held_x, held_y = tone_batch(32, seed + 3)

    torch.manual_seed(seed)
    pretrained = TinyCTC()
    train_steps(pretrained, pre_x, pre_y, 60, 0.5)
    with torch.no_grad():
        source_before = float(ctc_loss(pretrained(pre_x), pre_y))
    saved = {name: value.detach().clone() for name, value in pretrained.state_dict().items()}

    finetuned = TinyCTC()
    finetuned.load_state_dict(saved)
    finetune_curve = train_steps(
        finetuned, few_x, few_y, 10, 0.5, held_x, held_y
    )

    torch.manual_seed(seed)
    scratch = TinyCTC()
    scratch_curve = train_steps(scratch, few_x, few_y, 10, 0.5, held_x, held_y)

    conflict = TinyCTC()
    conflict.load_state_dict(saved)
    cycled = pre_y % 3 + 1
    train_steps(conflict, pre_x, cycled, 60, 0.5)
    with torch.no_grad():
        source_after = float(ctc_loss(conflict(pre_x), pre_y))

    return {
        "finetune_curve": finetune_curve,
        "scratch_curve": scratch_curve,
        "finetune_loss": finetune_curve[-1],
        "scratch_loss": scratch_curve[-1],
        "source_before": source_before,
        "source_after": source_after,
    }


def lora_example() -> None:
    """The 3 by 4 worked product. Scale is alpha / r = 1."""
    weight = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    matrix_a = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    matrix_b = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    features = np.array([1.0, 2.0, 3.0, 4.0])
    scale = 2.0 / matrix_a.shape[0]
    product = matrix_b @ matrix_a
    merged = weight + scale * product
    output = features @ weight.T + scale * (features @ matrix_a.T) @ matrix_b.T
    print("scale", scale)
    print("BA", product.tolist())
    print("y", output.tolist())
    print("merged", merged.tolist())
    print("trainable", int(matrix_a.size + matrix_b.size))
    print("frozen", int(weight.size))
    rank, width = 8, 768
    trainable = rank * (width + width)
    frozen = width * width
    print("base projection trainable", trainable, "frozen", frozen, "ratio", frozen // trainable)


def figure_svg(finetune: list[float], scratch: list[float]) -> str:
    """Held-out CTC loss over the 10 adaptation steps. Points are the curves."""
    x0, x1 = 88.0, 600.0
    y_top, y_bot = 46.0, 228.0
    loss_max = 1.8

    def x_of(step: float) -> float:
        return x0 + (step - 1.0) / 9.0 * (x1 - x0)

    def y_of(loss: float) -> float:
        return y_bot - loss / loss_max * (y_bot - y_top)

    def polyline(curve: list[float]) -> str:
        return " ".join(
            f"{x_of(step):.1f},{y_of(loss):.1f}"
            for step, loss in enumerate(curve, start=1)
        )

    y_ticks = []
    for loss, label in ((0.0, "0"), (0.6, "0.6"), (1.2, "1.2"), (1.8, "1.8")):
        y_pos = y_of(loss)
        y_ticks.append(
            f'  <line x1="{x0:.1f}" y1="{y_pos:.1f}" x2="{x1:.1f}" y2="{y_pos:.1f}" class="stroke-grid"/>'
        )
        y_ticks.append(
            f'  <text x="{x0 - 8:.1f}" y="{y_pos + 4:.1f}" class="fig-small" text-anchor="end">{label}</text>'
        )
    x_ticks = []
    for step in (1, 5, 10):
        x_pos = x_of(float(step))
        x_ticks.append(
            f'  <text x="{x_pos:.1f}" y="{y_bot + 18:.1f}" class="fig-small" text-anchor="middle">{step}</text>'
        )

    return f"""<svg viewBox="0 0 640 280" role="img" aria-label="Held-out CTC loss over 10 steps. Fine-tuning stays near 0.10 while training from scratch stays near 1.57.">
  <text x="16" y="22" class="fig-small">CTC loss</text>
  <line x1="330" y1="16" x2="358" y2="16" class="stroke-b"/>
  <text x="364" y="20" class="fig-small">fine-tune</text>
  <line x1="450" y1="16" x2="478" y2="16" class="stroke-a"/>
  <text x="484" y="20" class="fig-small">from scratch</text>
{chr(10).join(y_ticks)}
  <line x1="{x0:.1f}" y1="{y_top:.1f}" x2="{x0:.1f}" y2="{y_bot:.1f}" class="stroke-muted"/>
  <line x1="{x0:.1f}" y1="{y_bot:.1f}" x2="{x1:.1f}" y2="{y_bot:.1f}" class="stroke-muted"/>
  <polyline points="{polyline(scratch)}" class="stroke-a" fill="none"/>
  <polyline points="{polyline(finetune)}" class="stroke-b" fill="none"/>
  <circle cx="{x_of(10):.1f}" cy="{y_of(scratch[-1]):.1f}" r="4.5" class="fill-a"/>
  <circle cx="{x_of(10):.1f}" cy="{y_of(finetune[-1]):.1f}" r="4.5" class="fill-b"/>
{chr(10).join(x_ticks)}
  <text x="{(x0 + x1) / 2:.1f}" y="268" class="fig-small" text-anchor="middle">adaptation step</text>
</svg>"""


def main() -> None:
    lora_example()
    result = adaptation_curves(0)
    finetune = result["finetune_curve"]
    scratch = result["scratch_curve"]
    assert isinstance(finetune, list) and isinstance(scratch, list)
    print("finetune curve", [f"{value:.6f}" for value in finetune])
    print("scratch curve", [f"{value:.6f}" for value in scratch])
    before = float(result["source_before"])
    after = float(result["source_after"])
    print(f"finetune {result['finetune_loss']:.6f}")
    print(f"scratch {result['scratch_loss']:.6f}")
    print(f"gap {float(result['scratch_loss']) - float(result['finetune_loss']):.6f}")
    print(f"source before {before:.6f} after {after:.6f}")
    print(f"forgetting {after - before:.6f}")
    print(figure_svg(finetune, scratch))


if __name__ == "__main__":
    main()
