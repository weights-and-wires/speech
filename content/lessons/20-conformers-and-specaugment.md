---
title: Conformers and SpecAugment
slug: 20-conformers-and-specaugment
part: "Part 4 — Neural speech recognition"
summary: Build SpecAugment, relative positional attention, and a Conformer block piece by piece, then train a Conformer-CTC on spoken digits.
minutes: 90
gpu: true
objectives:
  - Mask random bands and spans of a log-mel spectrogram with SpecAugment, within the policy's limits.
  - Compute Transformer-XL relative positional attention, including the rel_shift trick, and check it against a brute-force loop.
  - Build a Conformer block from its half-step feed-forward, attention, and convolution modules, with padding masked correctly.
  - Train a Conformer-CTC on sequences of spoken digits and measure its digit error rate on a speaker it never heard.
prereqs:
  - 15-neural-acoustic-models
  - 16-connectionist-temporal-classification
  - 18-attention-encoder-decoders
exercise: 20-conformers-and-specaugment
---

# Conformers and SpecAugment

The loss functions of the last few lessons, CTC, attention, and the
transducer, all sit on top of an encoder. The encoder decides how good the
recognizer can be. For several years the **Conformer** was the encoder of
choice for speech, and its descendants, such as the FastConformer inside
NVIDIA's Parakeet models, power open recognizers.

The Conformer's idea is simple to say. Self-attention sees the whole
utterance at once, which is good for context, such as which word a sound
belongs to. Convolution sees a few neighboring frames, which is good for local
detail, such as the quick formant change that separates "b" from "d". A
Conformer block does both, one after the other, wrapped between two
feed-forward layers.

A large encoder also overfits a small dataset quickly. **SpecAugment** is the
cheap fix that made these models train well: blank out random stripes of the
spectrogram so the model can't lean on any one cue. In this lesson you build
SpecAugment, attention with relative positions, and a Conformer block, and
then train a small Conformer-CTC to recognize strings of spoken digits from a
speaker it has never heard.

## What you already know

- A log-mel spectrogram has 80 mel bands and 100 frames per second, from *Hearing and the mel scale*.
- A convolutional front end with two stride-2 layers cuts 100 frames per second to about 25, from *Neural acoustic models*.
- CTC trains an encoder with no frame labels, from *Connectionist temporal classification*.
- Self-attention scores every pair of frames with a dot product, from *Attention encoder-decoders*.
- A residual block adds its output to its input, and LayerNorm rescales each frame to zero mean and unit variance.

## SpecAugment

A model trained on clean features learns to rely on whatever is easiest: a
particular band, or one frame of a burst. Hide that cue at random and the
model must learn several ways to hear the same thing. **SpecAugment** (Park et
al., 2019) does this on the features, not the audio, so it costs almost
nothing.

It applies two kinds of mask to a spectrogram with $\nu$ mel bands and
$\tau$ frames. A **frequency mask** hides a band of $f$ channels starting at
$f_0$, in every frame. A **time mask** hides a span of $t$ frames starting at
$t_0$, in every band:

$$
\begin{aligned}
f &\sim \mathcal{U}\{0, \dots, F\}, \quad f_0 \sim \mathcal{U}\{0, \dots, \nu - f\} \\
t &\sim \mathcal{U}\{0, \dots, \min(T, \lfloor p\,\tau \rfloor)\}, \quad t_0 \sim \mathcal{U}\{0, \dots, \tau - t\}
\end{aligned}
$$

Read it aloud: draw the mask's width uniformly from 0 to its limit, then draw
where it starts so that it fits inside the spectrogram. Frequency masks are at
most $F$ bands wide. Time masks are at most $T$ frames wide, and never wider
than a fraction $p$ of the utterance. You repeat each kind $m_F$ and $m_T$
times. The original paper also warps the time axis. Many systems skip that step,
because the paper found it the most expensive step and the least helpful. In torch:

```python
f = int(torch.randint(0, F + 1, (1,), generator=gen))
f0 = int(torch.randint(0, n_mels - f + 1, (1,), generator=gen))
out[f0 : f0 + f, :] = mask_value
```

### Work a mask by hand

Take 8 bands and 10 frames, with $F = 3$, $T = 4$, $p = 0.2$, and one mask of
each kind. Suppose the draws are $f = 2$ and $f_0 = 5$: bands 5 and 6 go dark
across all 10 frames. The time mask's limit is
$\min(4, \lfloor 0.2 \times 10 \rfloor) = 2$ frames. Suppose it draws $t = 2$
and $t_0 = 3$: frames 3 and 4 go dark across all 8 bands. The two masks cover
$2 \times 10 + 8 \times 2 - 2 \times 2 = 32$ of the 80 cells, because the four
cells where they cross are counted twice.

The paper's policies for LibriSpeech and Switchboard are the following:

| Policy | $F$ | $m_F$ | $T$ | $p$ | $m_T$ |
|---|---|---|---|---|---|
| LibriSpeech basic (LB) | 27 | 1 | 100 | 1.0 | 1 |
| LibriSpeech double (LD) | 27 | 2 | 100 | 1.0 | 2 |
| Switchboard mild (SM) | 15 | 2 | 70 | 0.2 | 2 |
| Switchboard strong (SS) | 27 | 2 | 70 | 0.2 | 2 |

These numbers assume utterances of several seconds. A single spoken digit
lasts about 0.4 seconds, 40 frames, so a 100-frame time mask would erase whole
digits. This lesson's run uses $F = 15$, $T = 20$, $p = 0.2$, and two masks of
each kind.

> [!WARNING]
> Mask with the mean, which is 0 after you normalize each utterance. A 0 in
> an unnormalized log-mel spectrogram means a loud signal, not silence. Mask
> only real frames, never padding, and only during training.

## Relative positional attention

Self-attention has no sense of order: shuffle the frames and every output
shuffles the same way. Transformers fix this with **positional encodings**.
The original Transformer adds a sinusoid for each absolute position to the
input. But in speech, what matters is how far apart two frames are, not
where they sit in the file. A phone sounds the same at second 1 as at
second 9.

**Relative positional encoding** (Dai et al., 2019, from Transformer-XL)
scores a query $i$ against a key $j$ with a term that depends only on the
distance $r = i - j$. With $q_i$, $k_j$, and a projected encoding $p_r$ of the
distance, all of length $d_k$ per head:

$$
s_{ij} = \frac{(q_i + u)^{\top} k_j + (q_i + v)^{\top} p_{i-j}}{\sqrt{d_k}}
$$

Read it aloud: the first term is ordinary content attention, plus a learned
vector $u$ that says which keys are worth attending to anywhere. The second
term matches the query against the distance, plus a learned vector $v$ that
says which distances matter regardless of content. Both $u$ and $v$ are
shared across all positions, with one pair per head. The distance encodings are
sinusoids, with rows for $r = L - 1$ down to $-(L - 1)$:

$$
\begin{aligned}
e_r[2i] &= \sin\left(r \cdot 10000^{-2i/d}\right) \\
e_r[2i+1] &= \cos\left(r \cdot 10000^{-2i/d}\right)
\end{aligned}
$$

Read it aloud: each pair of columns is one sine wave and one cosine wave in
the distance, each pair slower than the last. A learned linear layer turns
$e_r$ into $p_r$. In torch:

```python
r = torch.arange(L - 1, -L, -1)[:, None]                       # (2L - 1, 1)
pe[:, 0::2], pe[:, 1::2] = torch.sin(r * rate), torch.cos(r * rate)
```

For $d = 2$ there's one pair, and $r = 1$ gives
$(\sin 1, \cos 1) = (0.8415, 0.5403)$. The row for $r = 0$ is always
$(0, 1, 0, 1, \dots)$.

### The rel_shift trick

Computing $(q_i + v)^\top p_r$ for every query and every distance is one
matrix product, with shape $(L, 2L - 1)$. Row $i$ holds the scores against
distances $L - 1, \dots, -(L - 1)$. But attention wants an $(L, L)$ matrix,
indexed by key $j$. The distance for key $j$ is $i - j$, which sits in column
$k = L - 1 - i + j$. So each row needs a different slice, shifted by one
column per row:

$$
\operatorname{out}[i, j] = x[i,\; L - 1 - i + j]
$$

For $L = 3$, row 0 takes columns 2, 3, 4, row 1 takes columns 1, 2, 3, and row
2 takes columns 0, 1, 2. On the matrix $x$ that holds 0 to 14 in order, the
result is `[[2, 3, 4], [6, 7, 8], [10, 11, 12]]`. You can write this with
`gather`, but implementations use a faster trick: pad one zero column on the
left, reshape so the rows slide by one, drop a row, and reshape back.

```python
padded = F.pad(x, (1, 0)).reshape(*lead, 2 * L, L)          # rows slide by one
out = padded[..., 1:, :].reshape(*lead, L, 2 * L - 1)[..., :L]
```

> [!KEY]
> Relative attention adds a score for the distance $i - j$ to the content
> score, so the model learns "attend two frames back" instead of "attend to
> frame 7". The rel_shift reshape turns one $(L, 2L - 1)$ product into the
> $(L, L)$ scores you need.

## The Conformer block

A Conformer block (Gulati et al., 2020) sandwiches attention and convolution
between two feed-forward modules, each of which adds only half its output.
This half-step design comes from Macaron-Net. For block input $x$:

$$
\tilde{x} = x + \tfrac12 \operatorname{FFN}(x), \quad
x' = \tilde{x} + \operatorname{MHSA}(\tilde{x}), \quad
x'' = x' + \operatorname{Conv}(x'), \quad
y = \operatorname{LayerNorm}\big(x'' + \tfrac12 \operatorname{FFN}(x'')\big)
$$

Read it aloud: add half a feed-forward step, then attention, then
convolution, then the other half of the feed-forward, each as a residual, and
normalize at the end. In torch:

```python
x = x + 0.5 * self.ff1(x)                      # half a feed-forward step
# The attention and convolution residuals go here.
return self.norm_out(x + 0.5 * self.ff2(x))    # the other half, then LayerNorm
```

Each module starts with its own LayerNorm, so every residual branch sees a
normalized input. That's called **pre-norm**. The modules are the following:

- **Feed-forward:** LayerNorm, Linear from $d$ to $4d$, Swish, Dropout, Linear back to $d$, Dropout. Swish is $x \cdot \sigma(x)$, `F.silu` in torch.
- **Attention:** LayerNorm, multi-head self-attention with relative positions, Dropout.
- **Convolution:** LayerNorm, a pointwise convolution from $d$ to $2d$ channels, a gated linear unit (GLU) back to $d$, a depthwise convolution of kernel $k$, BatchNorm, Swish, a pointwise convolution, Dropout.

A **pointwise** convolution has kernel size 1: it mixes channels within each
frame, like a linear layer. A **depthwise** convolution has one filter per
channel, so it slides along time without mixing channels. The GLU splits its
$2d$ inputs into halves $a$ and $b$ and returns $a \odot \sigma(b)$. The
paper uses a kernel of 32; open implementations use 31, because an odd kernel
pads symmetrically. In torch the convolution module's middle is:

```python
y = F.glu(self.pointwise1(y), dim=1)                 # (B, 2d, T) -> (B, d, T)
y = F.silu(self.batch_norm(self.depthwise(y)))
```

The paper's three sizes, trained as transducers on LibriSpeech with
SpecAugment, are the following:

| Model | Parameters | Blocks | Width | Heads | Kernel |
|---|---|---|---|---|---|
| Conformer S | 10.3M | 16 | 144 | 4 | 32 |
| Conformer M | 30.7M | 16 | 256 | 4 | 32 |
| Conformer L | 118.8M | 17 | 512 | 8 | 32 |

Conformer L reached a word error rate of 2.1% on test-clean and 4.3% on
test-other without a language model, and 1.9% and 3.9% with one. Its
SpecAugment used $F = 27$ with ten time masks, each at most 5% of the
utterance.

> [!KEY]
> A Conformer block is FFN/2, attention, convolution, FFN/2, then LayerNorm,
> each module a pre-norm residual. Attention supplies global context and the
> depthwise convolution supplies local detail.

### Mask the padding

A batch pads short utterances to the length of the longest one. Attention
ignores the padding if you mask the keys. The depthwise convolution doesn't:
a kernel of 15 reaches 7 frames past the end of an utterance and mixes
whatever the padding holds into the last real frames. So set padded frames to
0 right before the depthwise convolution.

BatchNorm is the other trap. In training mode it averages over every frame in
the batch, padding included, and in evaluation mode it uses running averages.
The checks run in evaluation mode, where a padded utterance's real frames
must come out exactly as if it were alone.

> [!TIP]
> Test padding by appending large random values to an utterance. If the real
> frames' output changes, something leaks.

## Train a Conformer-CTC on digits

The **Run** block is the experiment. The model is the given front end, your
Conformer blocks, and a linear head over 11 symbols: the blank and the digits
0 to 9. The front end is two $3 \times 3$ convolutions with stride 2, which
turn $n$ feature frames into $\lfloor(\lfloor(n - 1)/2\rfloor - 1)/2\rfloor$
encoder frames. One second of audio gives $n = 101$ frames, since the STFT
centers its first window on sample 0. The front end turns that into
$\lfloor (50 - 1)/2 \rfloor = 24$ encoder frames, 40 ms apart. The CTC input
lengths must use this formula, or the loss reads padding as speech.

The data comes from the Free Spoken Digit Dataset, `fsdd`: 3,000 recordings
of the digits 0 to 9 by six speakers, at 8 kHz. The run resamples them to
16 kHz and builds utterances by joining 2 to 5 random recordings with short
gaps, so a training example is a string such as "4 0 7". It trains on five
speakers and tests on the sixth, `theo`, whose voice it never hears. The
score is the **digit error rate**: the edit distance between the recognized
and the spoken digits, divided by the number of spoken digits. It's the word
error rate of *Language models and word error rate*, with digits as words.

On a GPU, the run trains 6 blocks of width 144, 3.6 million parameters, with
SpecAugment, for 3,000 steps of 64 utterances. Without a GPU it shrinks to one
block of width 96, a thinner front end, and no SpecAugment, for 150 steps of
16 utterances, and says so.

The following table lists measured runs, tested on 60 or 300 utterances from
the held-out speaker:

| Run | Model | Steps of 16 | SpecAugment | Time | Digit error rate |
|---|---|---|---|---|---|
| The CPU fallback | 1 block, width 96, 295,000 parameters | 150 | No | 25 s | 89.2% to 46.9% |
| A longer CPU run | the same | 1,000 | Yes | 7 min | 92.3% to 9.7% |
| The GPU model, a third of its schedule, on an Apple GPU | 6 blocks, width 144, 3.6M | 1,000 of 64 | Yes | 8 min | 103.4% to 10.5% |

The loss falls from about 24 to below 0.5 in the short run. Trained longer
with SpecAugment, even the small model gets fewer than one digit in ten wrong
for a voice it never heard.

> [!RECAP]
> - SpecAugment draws a width uniformly up to its limit, then a start that fits,
>   and masks whole bands or whole frames on a copy.
> - Relative attention scores $(q_i + u)^\top k_j + (q_i + v)^\top p_{i-j}$,
>   divided by $\sqrt{d_k}$, with rows of $p$ running from $r = L - 1$ to
>   $-(L - 1)$.
> - `rel_shift` sets `out[i, j] = x[i, L - 1 - i + j]`.
> - The block is `x + ff1/2`, `+ mhsa(norm(x))`, `+ conv(x)`, `+ ff2/2`, then
>   `norm_out`, with padding zeroed before the depthwise convolution.

## Build it step by step

The checks test the pieces in this order. Each module's `__init__` is given,
so you write the `forward` methods.

1. **`spec_augment(features, ...)`.** Takes `(n_mels, frames)` and returns a
   masked copy. Clone first, draw with `torch.randint(low, high + 1, (1,),
   generator=generator)`, and assign slices. Common mistake: drawing the time
   limit from the padded length instead of `length`, which masks padding and
   can hide a whole short utterance.
2. **`relative_position_encoding(length, d_model)`.** Returns `(2L - 1, d)`.
   Build `r = torch.arange(L - 1, -L, -1)` and the rates, then fill even
   columns with `sin` and odd ones with `cos`. Common mistake: rows in
   ascending order, which flips every distance.
3. **`rel_shift(x)`.** Returns `(..., L, L)`. Use the pad-and-reshape trick or
   `gather`. Common mistake: keeping the last $L$ columns instead of the first.
4. **`RelPositionMultiHeadAttention.forward(x, pos_emb, mask)`.** Returns
   `(B, T, d)`. Add `pos_bias_u` and `pos_bias_v` to `q` while it's
   `(B, T, H, d_k)`, then transpose to heads. Common mistake: projecting
   `pos_emb` with `linear_k`, which ties content and position together.
5. **`FeedForwardModule.forward` and `ConvolutionModule.forward`.** Follow the
   layer lists. Transpose to `(B, d, T)` for the convolutions and back.
   Common mistake: applying `F.glu` over the time axis instead of channels.
6. **`ConformerBlock.forward(x, pos_emb, mask)`.** Four residuals and a final
   LayerNorm. Common mistake: putting `norm_out` inside the last residual.

## Cost and scaling

The table uses $T$ encoder frames, width $d$, kernel $k$, and one block. The
parameter counts are for $d = 144$ and $k = 31$.

| Module | Work per utterance | Parameters |
|---|---|---|
| Each feed-forward module | $O(T \cdot 8d^2)$ | 166,896 |
| Attention | $O(T \cdot 6d^2 + T^2 d)$ | 104,832 |
| Convolution module | $O(T \cdot 3d^2 + T d k)$ | 67,824 |
| Final LayerNorm | $O(Td)$ | 288 |
| Whole block | | 506,736 |

The two feed-forward modules hold two thirds of a block's parameters. The
attention's $T^2$ term is what grows fastest: at 25 frames per second, 30
seconds of audio gives 750 frames and 562,500 scores per head and layer, and
an hour gives 90,000 frames. This is why FastConformer (Rekesh et al., 2023)
subsamples by 8 instead of 4, to 12.5 frames per second, and uses a kernel of
9, and why long-form models limit attention to a local window.

## Where this breaks

**BatchNorm and streaming don't mix.** As *Transducers and streaming*
showed, statistics over one chunk differ from those over a whole utterance,
and small batches make them noisy too. Streaming Conformers often swap
BatchNorm for LayerNorm in the convolution module.

**Training is unstable at first.** Deep attention stacks need a learning-rate
warmup, and CTC sits on a plateau where it predicts only blanks before it
finds the alignments. The GPU run warms up for its first 300 steps, then
decays with a cosine.

**SpecAugment can erase the label.** A time mask wider than a word removes
the only evidence for it, and the model learns to guess. Scale $T$ to your
utterances, as this lesson's run does for digits.

**Relative positions are easy to get subtly wrong.** A flipped row order or
an off-by-one shift still trains, but worse.

**A held-out speaker is harder than held-out recordings.** Six speakers are
too few to cover how voices vary, so expect a gap between training and test
error.

## The exercise

You build `spec_augment`, `relative_position_encoding`, `rel_shift`, and the
`forward` methods of `RelPositionMultiHeadAttention`, `FeedForwardModule`,
`ConvolutionModule`, and `ConformerBlock`. Press **Run** to train the
Conformer-CTC on spoken digits. The run prints the digit error rate on the
held-out speaker before and after training, plots the loss, and plays one test
utterance with its transcript and features.

The checks run on the CPU with no data. They confirm that SpecAugment masks
whole bands and frames within the policy's limits, respects the true length,
and draws widths with a mean near $F/2$. They compare the encoding with its
formula, `rel_shift` with `gather`, and your attention with a brute-force loop
over heads, queries, and keys. They check both modules against their layer
lists, and they confirm that padding changes nothing, in the convolution
module and in the whole block.
