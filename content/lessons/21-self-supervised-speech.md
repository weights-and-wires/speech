---
title: Self-supervised speech
slug: 21-self-supervised-speech
part: "Part 4 — Neural speech recognition"
summary: Build the losses and targets behind wav2vec 2.0, HuBERT, and BEST-RQ, then probe what each layer of two pretrained models knows.
minutes: 80
gpu: true
objectives:
  - Compute the frame count, hop, and receptive field of the wav2vec 2.0 feature encoder, and the fraction of frames that span masking hides.
  - Implement the contrastive loss with distractors from masked frames, and the diversity loss that keeps a codebook in use.
  - Implement HuBERT's masked prediction of k-means targets and BEST-RQ's frozen random-projection targets.
  - Fit a closed-form linear probe to every layer of wav2vec2-base and hubert-base, and read where speaker and content information lives.
  - Turn a layer's features into discrete units with k-means, and explain what those units keep and drop.
prereqs:
  - 15-neural-acoustic-models
  - 16-connectionist-temporal-classification
  - 20-conformers-and-specaugment
exercise: 21-self-supervised-speech
---

# Self-supervised speech

A transcribed hour of speech costs hours of careful listening and typing. An
untranscribed hour costs almost nothing, and audiobooks, podcasts, and
broadcasts supply millions of them. The recognizers from the previous lessons
need the transcribed kind.

Self-supervised learning uses the cheap kind. It trains an encoder on audio
alone, with a task made from the audio itself: hide some frames, and ask the
model to recognize what was hidden from the context around it. To do that
well, the model has to learn what speech sounds are and how they follow each
other. Afterward, fine-tuning with CTC on a little labeled speech makes a
recognizer: wav2vec 2.0 BASE reaches 6.1% WER on LibriSpeech test-clean from
100 labeled hours, with no language model.

You build the pieces of three designs: wav2vec 2.0, which picks the hidden
frame out of a line-up of impostors; HuBERT, which predicts labels made by
k-means; and BEST-RQ, which predicts labels from a random quantizer that never
trains. Then you open two pretrained models and measure, layer by layer, where
they keep who is speaking and where they keep what is said.

## What you already know

- A strided convolution shortens a sequence, as in Neural acoustic models.
- CTC trains an encoder against a transcript without an alignment, as in
  Connectionist temporal classification.
- SpecAugment hides spans of the input during training, as in Conformers and
  SpecAugment.
- Cross-entropy is minus the log of the probability a softmax puts on the
  right answer.
- Cosine similarity, $a^\top b / (\lVert a \rVert \lVert b \rVert)$, measures
  the angle between two vectors and ignores their lengths.

## From raw samples to frames

wav2vec 2.0 reads the waveform itself, at 16,000 samples per second (16 kHz).
A **feature encoder** of seven unpadded 1-D convolutions turns the samples
into 512-dimensional vectors $z_t$, one per **frame**. The kernel widths are
(10, 3, 3, 3, 3, 2, 2) samples and the strides are (5, 2, 2, 2, 2, 2, 2).
Each layer maps a length $L$ to

$$
L' = \left\lfloor \frac{L - k}{s} \right\rfloor + 1
$$

Read it aloud: slide a window of $k$ samples in steps of $s$, and count how
many windows fit. In numpy it's one integer division per layer:
`length = (length - k) // s + 1`.

Work one second by hand. From 16,000 samples, the first layer gives
$\lfloor 15990 / 5 \rfloor + 1 = 3199$, and the next six give 1599, 799, 399,
199, 99, and 49. One second becomes 49 frames, not 50, because every layer
loses a little at the edge.

The **hop** between frames is the product of the strides, $5 \times 2^6 = 320$
samples, or 20 ms. The **receptive field** is how many input samples one frame
depends on:

$$
R = 1 + \sum_{i=1}^{7} (k_i - 1) \prod_{j<i} s_j
$$

Read it aloud: start at one sample, and let each layer widen the view by its
extra kernel taps, counted in the step size of its own input. Here
$R = 1 + 9 + 10 + 20 + 40 + 80 + 80 + 160 = 400$ samples, or 25 ms. Each frame
sees 25 ms every 20 ms, the shape of a classical spectrogram frame.

A Transformer, the **context network**, turns the frames into outputs $c_t$
that can look at the whole utterance. The BASE model has 12 blocks of width 768
and about 95 million parameters. HuBERT BASE has the same feature encoder and
Transformer size.

## Masking spans

Before the Transformer, wav2vec 2.0 picks each frame as the start of a
**span** with probability $p = 0.065$, and replaces that frame and the $M - 1
= 9$ after it with one learned vector. Spans can overlap. A frame stays
visible only if none of the $M$ frames at or before it started a span, so the
masked fraction is

$$
P(\text{masked}) = 1 - (1 - p)^M
$$

Read it aloud: $(1-p)^M$ is the chance that all $M$ possible starts missed.
With $p = 0.065$ and $M = 10$, $1 - 0.935^{10} = 0.489$: about 49% of frames
are hidden, the figure the wav2vec 2.0 paper reports. In numpy, loop over
`np.flatnonzero(starts)` and set `mask[s : s + span] = True`. HuBERT uses the
same scheme with $p = 0.08$.

## The contrastive loss

The wav2vec 2.0 task is multiple choice. For a masked frame $t$, the output
$c_t$ must pick the true target $q_t$ out of a line-up that also holds $K$
**distractors**, targets from other masked frames of the same utterance. The
targets come from quantizing the *unmasked* features, so the answer isn't in
the model's input:

$$
\mathcal{L}_m = -\log \frac{\exp\left(\operatorname{sim}(c_t, q_t)/\kappa\right)}{\sum_{\tilde q \in Q_t} \exp\left(\operatorname{sim}(c_t, \tilde q)/\kappa\right)}
$$

Read it aloud: $\operatorname{sim}$ is cosine similarity, $\kappa$ is a
**temperature**, and $Q_t$ holds $q_t$ and the $K$ distractors. The fraction is
a softmax over the line-up, and the loss is minus the log of the probability
on the true target. wav2vec 2.0 uses $\kappa = 0.1$ and $K = 100$, and the
exercise averages over masked frames. In numpy, with the positive in column 0:

```python
cands = np.concatenate([q[masked][:, None], q[distractors]], axis=1)
logits = np.einsum("md,mkd->mk", unit(c[masked]), unit(cands)) / kappa
loss = -np.mean(log_softmax(logits, axis=1)[:, 0])
```

Draw the distractors uniformly, with replacement, from the other $M - 1$
masked frames.

### Work the loss by hand

Take $c_t = (1, 0)$, the target $(0.8, 0.6)$, and distractors $(0, 1)$ and
$(-0.6, 0.8)$. All are unit length, so the cosines are 0.8, 0, and $-0.6$.

- With $\kappa = 0.1$, the logits are 8, 0, and $-6$, and the loss is
  $\log(1 + e^{-8} + e^{-14}) = 0.000336$ nats.
- With $\kappa = 1$, the logits are 0.8, 0, and $-0.6$, and the loss is
  $\log(1 + e^{-0.8} + e^{-1.4}) = 0.528$ nats.

The same angles cost 1,500 times less at the low temperature, so training
spends its effort on the hard line-ups.

> [!KEY]
> The contrastive loss is cross-entropy over a line-up: the true quantized
> frame against distractors from other masked frames, scored by cosine
> similarity over a temperature.

## The quantizer and the diversity loss

wav2vec 2.0 makes $q_t$ with a **product quantizer**: $G = 2$ codebooks of
$V = 320$ entries. A linear layer scores each codebook's entries, a Gumbel
softmax picks one per codebook (a hard choice forward, soft gradients
backward), and the two picks are concatenated and projected. That allows
$320^2 = 102{,}400$ targets.

Left alone, a quantizer collapses onto a few entries. The **diversity loss**
pushes each codebook's average use toward uniform:

$$
\mathcal{L}_d = \frac{1}{GV} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar p_{g,v} \log \bar p_{g,v}
$$

Read it aloud: $\bar p_{g,v}$ is the softmax probability of entry $v$ in
codebook $g$, without Gumbel noise, averaged over the batch. The sum is minus
each codebook's entropy, so minimizing it spreads use out. The minimum, with
every entry used equally, is $-\log V / V = -0.0180$ for $V = 320$; a
codebook that collapses onto one entry scores 0. The objective is
$\mathcal{L}_m + 0.1\,\mathcal{L}_d$. In numpy,
`p_bar = softmax(logits, axis=2).mean(axis=0)` and then
`np.sum(p_bar * np.log(p_bar)) / (G * V)`.

Work it with one codebook of two entries and two frames. If the frames put
(0.9, 0.1) and (0.1, 0.9) on the entries, $\bar p = (0.5, 0.5)$ and
$\mathcal{L}_d = -0.347$. If both put (0.9, 0.1), $\mathcal{L}_d = (0.9 \log
0.9 + 0.1 \log 0.1)/2 = -0.163$, which is worse. Each frame is equally sure in
both cases; only the spread across frames differs.

> [!WARNING]
> Average the probabilities over frames first, then take $p \log p$. Averaging
> each frame's entropy instead rewards every frame for being unsure.

## HuBERT: predict cluster labels

HuBERT replaces the learned quantizer with labels made before training. It
runs k-means on features of the whole training set and gives every frame the
index of its nearest cluster. The first iteration clusters 39-dimensional MFCCs
into 100 groups; the second clusters layer 6 of the first model into 500
groups and trains again. The model predicts the label at masked frames:

$$
p(c \mid t) = \frac{\exp\left(\operatorname{sim}(A o_t, e_c)/\tau\right)}{\sum_{c'} \exp\left(\operatorname{sim}(A o_t, e_{c'})/\tau\right)}
$$

Read it aloud: project the Transformer output $o_t$ with $A$, take its cosine
with each cluster's learned embedding $e_c$, divide by $\tau = 0.1$, and
normalize. The loss is cross-entropy against the k-means label over masked
frames only. HuBERT's authors found masked frames alone work best: at an
unmasked frame, the answer is in the input.

**k-means** alternates two steps until nothing changes: label each point with
its nearest center, then move each center to the mean of its points. The
exercise seeds it with k-means++, which places each new center at a point drawn
with probability proportional to its squared distance from the nearest center
so far.

## BEST-RQ: targets from a random quantizer

BEST-RQ asks whether targets need training at all. It stacks 4 frames of
normalized log-mel features into $x$, projects with a frozen random matrix
$A$, and picks the nearest entry of a frozen random codebook of 8,192 vectors
in 16 dimensions:

$$
y = \operatorname*{arg\,min}_i \left\lVert \frac{c_i}{\lVert c_i \rVert} - \frac{A x}{\lVert A x \rVert} \right\rVert
$$

Read it aloud: scale the projection and every codeword to unit length, and
take the closest codeword. Between unit vectors the squared distance is
$2 - 2\cos$, so it's also the codeword with the largest cosine. The encoder
learns to predict $y$ at masked frames, and because the quantizer never
depends on the encoder, any encoder can use it, streaming ones included.

Work one. Let $Ax = (3, 4)$, which scales to $(0.6, 0.8)$, and let the
codebook hold $(1, 0)$, $(0, 2)$, and $(-1, 1)$. The cosines are 0.6, 0.8, and
0.141, and the squared distances are 0.8, 0.4, and 1.717, so the label is 1.
The length of $(0, 2)$ doesn't count.

## What the layers learn

A BASE encoder holds 13 representations: the feature encoder's output, called
layer 0 here, and one after each Transformer block. A **linear probe** asks how
much of a property a layer holds in a form a straight line can read. Freeze the model, average each clip's frames per layer, and fit a
linear classifier. The exercise fits it by ridge regression onto one-hot
labels $Y$, with a column of ones appended to the features $X$:

$$
W = \left(X^\top X + \lambda P\right)^{-1} X^\top Y
$$

Read it aloud: $\lambda$ sets how hard to shrink the weights, and $P$ is the
identity with its last diagonal entry set to 0, so the bias isn't shrunk. The
predicted class is the column of $XW$ with the largest value. In numpy,
`W = np.linalg.solve(X.T @ X + l2 * P, X.T @ Y)`.

The Run probes the Free Spoken Digit Dataset: six speakers saying 0 to 9,
resampled from 8 kHz to 16 kHz. Each layer gets two probes:

- **Content.** Train a digit probe on five speakers and test on the sixth.
- **Speaker.** Train a speaker probe on the digits 0 to 4 and test on 5 to 9.

With all 3,000 clips, the digit probe on `wav2vec2-base` peaks at layer 4
(95%) and falls to 78% at layer 12. On `hubert-base` it stays between 91% and
93% from layer 2 to the top. The speaker probe peaks early on both, at layer 1
for `wav2vec2-base` (96%) and layer 2 for `hubert-base` (97%), and falls to
73% and 61% at the top. The WavLM authors found the same pattern in the layer
weights their models learned for the SUPERB benchmark.

> [!KEY]
> Self-supervised encoders keep who is speaking in their lower layers and what
> is said in their middle and upper layers. Pick the layer for the task.

Run k-means on one layer's frames and each 20 ms frame becomes a cluster
index, a **unit**. Collapse runs of the same unit and a clip becomes a short
sequence in an alphabet the model invented. Generative spoken language
modeling uses HuBERT layer 6 with 50 to 200 units. In the Run, a histogram of
100 layer-6 units names the digit of an unseen speaker 87% of the time, but
the speaker of an unseen digit only 40% of the time. Units keep phones and drop
most of the voice.

> [!RECAP]
> - One second at 16 kHz gives 49 frames, with a 400-sample receptive field and
>   a 320-sample hop.
> - The contrastive loss is cross-entropy over cosine logits divided by
>   $\kappa$, with the true target in column 0 and masked-frame distractors.
> - The diversity loss averages probabilities over frames before $p \log p$.
> - HuBERT and BEST-RQ predict discrete labels at masked frames; BEST-RQ's come
>   from a frozen random projection.
> - A ridge probe is $(X^\top X + \lambda P)^{-1} X^\top Y$ with an unpenalized
>   bias.

## Build it step by step

The checks test the functions in this order.

1. **`conv_output_length(n_samples, kernels, strides)`.** Returns an int.
   Common mistake: letting a short input go negative instead of returning 0.
2. **`receptive_field(kernels, strides)`.** Returns `(field, hop)` in samples.
   Add `(k - 1) * product` before multiplying the running stride product by
   this layer's stride. Common mistake: updating the product first.
3. **`span_mask(starts, span)`.** Boolean `(T,)` in and out. Common mistake:
   masking `span + 1` frames.
4. **`sample_distractors(masked, n_distractors, rng)`.** Returns `(M, K)`
   frame indices. Draw `rng.integers(0, M - 1, size=(M, K))` and add 1 where
   the draw is at or past the row's own position. Common mistake: drawing from
   all frames.
5. **`contrastive_loss(context, targets, masked, distractors, kappa)`.**
   Gather `(M, K + 1, D)` candidates, scale to unit length, and take a
   log-softmax over axis 1. Common mistake: not normalizing the context.
6. **`diversity_loss(logits)`.** `(N, G, V)` logits in, a float out. Common
   mistake: the order in the warning.
7. **`masked_prediction_loss(outputs, projection, code_embeddings, targets, mask, tau)`.**
   Use `outputs[mask] @ projection.T`, cosines with every embedding, and
   mean cross-entropy. Common mistake: counting unmasked frames.
8. **`best_rq_targets(features, projection, codebook, stack)`.** Reshape
   `features[: t * stack]` to `(t, stack * F)`. Common mistake: comparing
   vectors before scaling them to unit length.
9. **`kmeans(features, k, n_iter, seed)`.** Returns `(centroids, labels)`.
   Common mistake: returning labels from before the last centroid update.
10. **`fit_linear_probe(features, labels, n_classes, l2)` and
    `probe_accuracy(weights, features, labels)`.** `(D + 1, C)` weights and a
    float. Use `np.linalg.solve`. Common mistake: penalizing the bias.

## Cost and scaling

The following table lists costs for $T$ frames, $M$ masked frames, $K$
distractors, $D$ dimensions, and $k$ clusters.

| Piece | Time | Memory |
|---|---|---|
| Transformer layer | $T^2 D$ for attention | $T^2$ weights per head |
| Contrastive loss | $M K D$ | an $(M, K+1, D)$ candidate array |
| One k-means round | $N k D$ for $N$ frames | an $(N, k)$ distance matrix |
| Ridge probe | $N D^2 + D^3$ | a $(D+1, D+1)$ matrix |

Pretraining is the expensive part, hundreds of thousands of steps on many
GPUs. Probing is cheap: the full Run took about two minutes on a laptop CPU.

## Where this breaks

**The wrong input scaling.** Both `preprocessor_config.json` files in the
assets set `do_normalize`, so the Run scales each clip to zero mean and unit
variance. Read the config; a mismatch costs accuracy silently.

**Codebook collapse.** Without the diversity loss, the quantizer uses a few
entries and the task becomes easy and useless. Watch codebook perplexity.

**Probes that learn the task.** Keep the probe linear and regularized, and
test on unseen speakers or words.

**The top of wav2vec 2.0.** Its last layers specialize in the contrastive task
and hold less content, as the digit probe shows. SUPERB-style systems use a
learned weighted sum of all layers.

**Units lose prosody.** Units drop pitch and most of the voice, so
resynthesis from units needs a separate pitch or speaker input.

## The exercise

You build `conv_output_length`, `receptive_field`, `span_mask`, and
`sample_distractors`, then the objectives `contrastive_loss`,
`diversity_loss`, and `masked_prediction_loss`. You add `best_rq_targets`,
`kmeans`, `fit_linear_probe`, and `probe_accuracy`.

The checks compare your frame counts with real `torch` convolutions, the
receptive field with a gradient, the masked fraction with $1 - (1-p)^M$, the
losses with `torch` cross-entropy and closed forms, and the ridge weights with
least squares on an augmented system. The **Run** probes all 13 layers of `wav2vec2-base` and `hubert-base` on spoken
digits, plots accuracy against layer, and prints unit sequences for one digit
from two speakers.
