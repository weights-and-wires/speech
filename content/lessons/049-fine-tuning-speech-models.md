---
title: Fine-tuning speech models
slug: 049-fine-tuning-speech-models
part: "Part 4 — Neural speech recognition"
summary: Attach a CTC head or a low-rank update to a frozen encoder, and compare few-shot loss with training from scratch.
minutes: 40
gpu: false
objectives:
  - Attach a CTC head and state which pretrained layers stay frozen.
  - Apply a low-rank update and count its trainable parameters.
  - Compare few-shot fine-tuning with training from scratch on seeded tones.
  - Compute a forgetting score when the original-domain loss rises.
prereqs:
  - 030-ctc-forward-backward
  - 039-self-supervised-speech
  - 041-whisper
  - 048-asr-recipe
exercise: 049-fine-tuning-speech-models
---

# Fine-tuning speech models

A recognizer trained on audiobooks hears a short support call and returns an empty transcript. You have 8 minutes of transcribed calls, not 960 hours. Another encoder trained from scratch spends those minutes learning what speech sounds like. Fine-tuning keeps the encoder you already have and spends the labels on the difference: a CTC head, a frozen feature encoder, and a low-rank change where a full matrix has too many entries.

[An ASR training recipe](/lessons/048-asr-recipe) starts from random weights. This chapter starts from a pretrained encoder. On synthetic tones, 10 steps of that fine-tune beat the same 10 steps from a fresh initialization, and a conflicting update makes the original tones worse.

## What you already know

- CTC trains an encoder against a transcript without an alignment, from [Computing CTC efficiently](/lessons/030-ctc-forward-backward).
- wav2vec 2.0's feature encoder turns the waveform into frames, and a later CTC fine-tune turns the encoder into a recognizer, from [Self-supervised speech](/lessons/039-self-supervised-speech).
- Whisper's prompt starts with a language token and a task token, from [Whisper and large-scale recognition](/lessons/041-whisper).
- A from-scratch recipe builds the vocabulary, the batches, and the learning rate, from [An ASR training recipe](/lessons/048-asr-recipe).
- A linear layer multiplies a weight matrix by the vector it reads.

## Freeze the feature encoder and train a head

Picture the stack as three blocks. The convolutional front end has already learned edges in the waveform. The Transformer has started to mean "this sound, in this context." The last linear layer is random, because pretraining never asked for characters.

[Baevski et al.](https://arxiv.org/abs/2006.11477) add that linear layer on the Transformer and train it with CTC, over 29 character tokens plus a word-boundary token. For the first 10,000 updates only that layer trains. After that the Transformer trains too. The feature encoder stays frozen for the whole fine-tune, because a few minutes of target audio can drag raw-waveform filters off the hours used to fit them, and every frame above them moves with them.

Table 1 of that paper is the labeled-data ladder. Base has 95 million parameters and model dimension 768. Large has 317 million. Ten minutes of labels already reach a single-digit test-clean error when the unlabeled pretraining is large, and one hour cuts that error further:

| Labeled audio | Unlabeled audio | Decode | test-clean | test-other |
|---|---|---|---|---|
| 10 minutes | LibriSpeech 960 h, Base | 4-gram | 9.1% | 15.6% |
| 10 minutes | LibriVox 53,200 h, Large | Transformer LM | 4.8% | 8.2% |
| 1 hour | LibriVox 53,200 h, Large | Transformer LM | 2.9% | 5.8% |

[SUPERB](https://arxiv.org/abs/2105.01051) freezes the shared encoder for every task and trains a light head on a weighted sum of the hidden layers, because the last layer is not always the one the task reads. You freeze the whole encoder when you want to compare encoders. You unfreeze the Transformer, and keep the feature encoder frozen, when you want the recognizer on one domain.

`freeze_mask(names, policy)` returns true when a parameter trains. Under `freeze_feature_encoder`, a name trains unless it contains `feature_encoder` or `feature_extractor`. The paper says feature encoder, and the library calls that stack `feature_extractor`. Both strings mark it, so `wav2vec2.feature_extractor.conv_layers.0.conv.weight` is false, while `wav2vec2.encoder.layers.0.attention.k_proj.weight` and `ctc_head.weight` are true. Under `ctc_head_only`, a name trains only when it contains `ctc_head` or `lm_head`, which is the first 10,000 updates. A mask that also trains the feature extractor leaves the conv weight true, and the check rejects it.

The paper also masks encoder frames during the fine-tune, which is the idea in [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment). This chapter does not rebuild that mask.

> [!KEY]
> The feature encoder stays frozen. The CTC head trains first, and the Transformer follows.

## Change a wide matrix by a thin product

One attention projection in wav2vec 2.0 Base is 768 by 768, so it holds 589,824 entries. Eight minutes of labels cannot pin every entry down, and you still want the projection to move a little.

[LoRA](https://arxiv.org/abs/2106.09685) freezes the pretrained matrix $W_0$ and learns the change as a product of two thin matrices. $W_0$ has shape $(d, k)$, with $d$ outputs and $k$ inputs. $B$ has shape $(d, r)$ and $A$ has shape $(r, k)$, with rank $r$ far below $d$ and $k$. Hu et al. initialize $A$ at random and $B$ at zero, so the change starts at zero, and the update is scaled by $\alpha / r$.

Take $d = 3$, $k = 4$, $r = 2$, and $\alpha = 2$, so the scale is 1. The frozen weight, the two factors, and one input are the following:

```python
weight = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0]]
A = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
B = [[0, 1],
     [1, 0],
     [0, 0]]
x = [1, 2, 3, 4]
```

$W_0 x$ is $(1, 2, 3)$. $A x$ is $(1, 2)$. $B(Ax)$ is $(2, 1, 0)$. Adding that product gives $(3, 3, 3)$. The product $BA$ itself is the matrix with rows $(0, 1, 0, 0)$, $(1, 0, 0, 0)$, and $(0, 0, 0, 0)$. Adding it to $W_0$ gives rows $(1, 1, 0, 0)$, $(1, 1, 0, 0)$, and $(0, 0, 1, 0)$, and that matrix times $x$ is $(3, 3, 3)$ again. $A$ holds 8 entries and $B$ holds 6, so you train 14 numbers. $W_0$ holds 12, and those stay frozen.

The paper multiplies $W_0$ on the left of a column. In numpy the input is a row, so the weight is transposed:

$$
h = W_0 x + \frac{\alpha}{r} B A x
$$

$W_0 x$ is the frozen map, $BAx$ is the low-rank change, and $\alpha / r$ scales that change. $r$ is the rank, `A.shape[0]` in the code. At deployment you add the change onto $W_0$ and drop $A$ and $B$:

$$
W = W_0 + \frac{\alpha}{r} B A
$$

The trainable count is $r(d + k)$, not $dk$. For one Base projection at rank 8, that is $8 \times (768 + 768) = 12{,}288$ numbers against 589,824 frozen ones, which is 1/48 of the matrix. The following code is the forward pass and the merge:

```python
scale = alpha / A.shape[0]
y = x @ weight.T + scale * (x @ A.T) @ B.T
merged = weight + scale * (B @ A)
```

> [!WARNING]
> Adding the low-rank product and forgetting $W_0$ drops the pretrained map. On this example the product alone sends $x$ to $(2, 1, 0)$, not $(3, 3, 3)$.

> [!KEY]
> The merged matrix is the frozen weight plus the scaled low-rank product. The product alone is a different layer.

## Compare a few clips with a fresh start

Picture eight clips, each two tones long. A pretrained encoder already maps each tone onto a direction a CTC head can read, so 10 further steps only adjust that direction. A fresh encoder has to discover the directions in those same 10 steps.

The tones are synthetic, so the gap comes from the initialization, not from a downloaded checkpoint. Each clip has 8 frames and 4 features, and a label of length 2 drawn from classes 1, 2, and 3. Class 0, the blank, has prototype $(0, 0, 0, 0)$. Classes 1, 2, and 3 sit on the first three axes, $(1, 0, 0, 0)$, $(0, 1, 0, 0)$, and $(0, 0, 1, 0)$. Frames 0 to 2 copy the first label, frames 4 to 6 copy the second, and frames 3 and 7 stay on the blank. Add Gaussian noise of standard deviation 0.1 from `torch.Generator().manual_seed(data_seed)`.

The model is `nn.Linear(4, 8)`, a tanh, and `nn.Linear(8, 4)`, biases included. That is $4 \times 8 + 8 + 8 \times 4 + 4 = 76$ parameters. Training is full-batch SGD at learning rate 0.5. The loss is mean CTC with blank 0, time moved to the front, input length 8, target length 2, and `zero_infinity=True`.

For model seed $s$, build the three batches before you touch the global generator. Use 64 pretraining clips at data seed $s + 1$, 8 few-shot clips at $s + 2$, and 32 held-out clips at $s + 3$. Then follow this order:

1. Call `torch.manual_seed(s)`, construct the model, and train 60 steps on the 64 clips. The mean CTC loss on those clips is the source loss before adaptation.
2. Copy that state into a second module. Train 10 steps on the 8 clips. The mean CTC loss on the 32 held-out clips is the fine-tune loss.
3. Call `torch.manual_seed(s)` again, construct a fresh model, and train 10 steps on the same 8 clips. The held-out loss is the from-scratch loss.

At $s = 0$ the fine-tune loss is 0.103521 and the from-scratch loss is 1.566033, a gap of 1.462512. Figure 1 is the held-out loss after each of the 10 steps.

```figure
<svg viewBox="0 0 640 280" role="img" aria-label="Held-out CTC loss over 10 steps. Fine-tuning stays near 0.10 while training from scratch stays near 1.57.">
  <text x="16" y="22" class="fig-small">CTC loss</text>
  <line x1="330" y1="16" x2="358" y2="16" class="stroke-b"/>
  <text x="364" y="20" class="fig-small">fine-tune</text>
  <line x1="450" y1="16" x2="478" y2="16" class="stroke-a"/>
  <text x="484" y="20" class="fig-small">from scratch</text>
  <line x1="88.0" y1="228.0" x2="600.0" y2="228.0" class="stroke-grid"/>
  <text x="80.0" y="232.0" class="fig-small" text-anchor="end">0</text>
  <line x1="88.0" y1="167.3" x2="600.0" y2="167.3" class="stroke-grid"/>
  <text x="80.0" y="171.3" class="fig-small" text-anchor="end">0.6</text>
  <line x1="88.0" y1="106.7" x2="600.0" y2="106.7" class="stroke-grid"/>
  <text x="80.0" y="110.7" class="fig-small" text-anchor="end">1.2</text>
  <line x1="88.0" y1="46.0" x2="600.0" y2="46.0" class="stroke-grid"/>
  <text x="80.0" y="50.0" class="fig-small" text-anchor="end">1.8</text>
  <line x1="88.0" y1="46.0" x2="88.0" y2="228.0" class="stroke-muted"/>
  <line x1="88.0" y1="228.0" x2="600.0" y2="228.0" class="stroke-muted"/>
  <polyline points="88.0,57.5 144.9,61.1 201.8,60.1 258.7,61.0 315.6,61.4 372.4,62.5 429.3,63.8 486.2,65.4 543.1,67.4 600.0,69.7" class="stroke-a" fill="none"/>
  <polyline points="88.0,214.6 144.9,215.0 201.8,215.4 258.7,215.8 315.6,216.1 372.4,216.4 429.3,216.7 486.2,217.0 543.1,217.3 600.0,217.5" class="stroke-b" fill="none"/>
  <circle cx="600.0" cy="69.7" r="4.5" class="fill-a"/>
  <circle cx="600.0" cy="217.5" r="4.5" class="fill-b"/>
  <text x="88.0" y="246.0" class="fig-small" text-anchor="middle">1</text>
  <text x="315.6" y="246.0" class="fig-small" text-anchor="middle">5</text>
  <text x="600.0" y="246.0" class="fig-small" text-anchor="middle">10</text>
  <text x="344.0" y="268" class="fig-small" text-anchor="middle">adaptation step</text>
</svg>
Figure 1. Ten steps on eight clips leave the pretrained encoder near a held-out CTC loss of 0.10. The same ten steps from a fresh initialization stay near 1.57.
```

> [!TIP]
> Call `torch.manual_seed(s)` immediately before each fresh module. A copy of the pretrained state does not need another seed. A from-scratch module does, or its initialization is whatever the generator has left.

> [!KEY]
> Ten steps on eight clips move a pretrained encoder a little and leave a random encoder near where it started. The gap is the pretraining.

## Score the original domain

After the 60 pretraining steps, the loss on those 64 clips is 0.146689. Keep that state. Replace each label $y$ with $y \bmod 3 + 1$, which sends 1 to 2, 2 to 3, and 3 to 1, and train 60 more steps on the same frames. The loss on the original labels is then 10.359814.

The forgetting score is how far that loss moved:

$$
s = L_{\mathrm{after}} - L_{\mathrm{before}}
$$

$L_{\mathrm{before}}$ is the original-domain loss before the conflicting steps, and $L_{\mathrm{after}}$ is the same loss afterward. Here $s = 10.359814 - 0.146689 = 10.213125$. A positive score means the original domain got worse. A negative score means the adaptation helped there too. This pair is the planted case the check uses.

> [!WARNING]
> Subtracting in the other order reports an improvement as forgetting. On the planted pair the score is positive, because 10.359814 is larger than 0.146689.

## Share one vocabulary across languages

[XLS-R](https://arxiv.org/abs/2111.09296) is one wav2vec 2.0 model pretrained on 436,000 hours in 128 languages, then fine-tuned per task. During pretraining, language $l$ is drawn with probability proportional to $(n_l / N)^{0.5}$, where $n_l$ is that language's hours and $N$ is the total, so a low-resource language is not drowned by the ones with thousands of hours. You still fine-tune a head on the shared encoder. You do not train a separate encoder per language.

Whisper already has the vocabulary, so a fine-tune does not add a CTC head. The [Hugging Face fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper) sets the tokenizer's language and task, and the label sequence starts with those tokens. For Hindi transcription the prefix is `<|startoftranscript|><|hi|><|transcribe|><|notimestamps|>`. The token layout is the one in [Whisper and large-scale recognition](/lessons/041-whisper). The guide fine-tunes `whisper-small` at learning rate `1e-5` for 5,000 steps. This chapter does not add an exercise for that prefix.

## Fine-tune on a GPU

A full fine-tune loads `wav2vec2-base` from the worker image, freezes every parameter whose name contains `feature_extractor`, and trains a CTC head with the schedule above. That run needs the asset and a GPU. The checks never load a checkpoint, and this run does not download one. The Run block prints that plan, then trains the 76-parameter encoder on CPU so the losses match this page even when a GPU is present.

> [!KEY]
> What you now know: you keep a pretrained encoder, freeze its feature encoder, and either train a CTC head or a low-rank change on top. On the seeded tones, those few steps beat training from scratch, and a conflicting update raises the original-domain loss.

> [!RECAP]
> - `freeze_feature_encoder` trains every name except `feature_encoder` and `feature_extractor`. `ctc_head_only` trains `ctc_head` and `lm_head`.
> - The forward pass is $W_0 x + (\alpha / r) B A x$, and the merge adds that same product onto $W_0$.
> - Trainable parameters are $r(d + k)$. The worked rank-2 layer trains 14 numbers, and a 768 by 768 projection at rank 8 trains 12,288.
> - At seed 0, 10 few-shot steps reach held-out CTC loss 0.103521. The same steps from scratch reach 1.566033.
> - The forgetting score is the original-domain loss after the conflicting update, minus the loss before it. On the planted pair that is 10.213125.

## Build it step by step

The checks call the pieces in this order.

1. **`LoRALinear.forward(x)`.** `weight` is `(out, in)`, `A` is `(r, in)`, and `B` is `(out, r)`. Return `x @ weight.T + (alpha / r) * (x @ A.T) @ B.T`, which must accept a batch of rows. The usual mistake is to forget the scale, so an $\alpha$ other than $r$ still behaves as scale 1.
2. **`LoRALinear.merge()`.** Return `weight + (alpha / r) * (B @ A)`. The usual mistake is to return the product and drop `weight`.
3. **`LoRALinear.trainable_count()`.** Return `A.size + B.size`. The usual mistake is to count `weight` as well. A rank-8 factor pair on a 768 by 768 matrix returns 12,288.
4. **`freeze_mask(names, policy)`.** Return a dict of name to bool. `freeze_feature_encoder` is false for `feature_encoder` and `feature_extractor`. `ctc_head_only` is true only for `ctc_head` and `lm_head`. The usual mistake is to leave the feature extractor trainable.
5. **`adaptation_losses(seed)`.** Return `(finetune_loss, scratch_loss, source_before, source_after)` from the procedure above. The usual mistakes are to score the 8 training clips instead of the 32 held-out clips, to forget the second `torch.manual_seed(seed)` before the fresh model, and to keep training the fine-tuned weights when you measure forgetting. The conflicting labels are `labels % 3 + 1`, and you score the original labels.
6. **`forgetting_score(loss_before, loss_after)`.** Return `loss_after - loss_before`. The usual mistake is to subtract in the other order, which makes the planted rise negative.

## Cost and scaling

A full update of one $d$ by $k$ matrix trains $dk$ numbers and stores a separate copy per task. A rank-$r$ update trains $r(d + k)$ numbers, and one frozen $W_0$ plus a pair $(A, B)$ is enough per task. After you merge, inference is one dense multiply of the original shape.

## Where this breaks

The planted update fits the cycled labels, and the source loss rises from 0.146689 to 10.359814. If you still need the original domain, mix some of its clips into the fine-tune, or keep both copies. A rank of 8 moves a 768 by 768 projection a little. A domain that needs a different feature geometry needs a larger rank or an unfrozen Transformer, and $r(d + k)$ is the cost. Leave the conv stack frozen unless you have enough audio to refit filters that see raw samples.

## The exercise

You write `LoRALinear.forward`, `merge`, and `trainable_count`, then `freeze_mask`, `adaptation_losses`, and `forgetting_score`. The checks match the forward pass and the merge to the dense product, the trainable count, both freeze policies, the seeded loss gap, and the planted forgetting score.
