---
title: Neural acoustic models
slug: 15-neural-acoustic-models
part: "Part 4 — Neural speech recognition"
summary: Classify every frame with a neural network, splice in context, and shrink the frame rate with a convolutional front end of exact length.
minutes: 60
gpu: false
objectives:
  - Stack each frame with its neighbors into a context window, and measure what the context buys.
  - Compute the exact output length of any convolution, and of the 4x Conv2d subsampling front end.
  - Build that front end in torch, and train a frame classifier on synthetic phones.
  - Turn a classifier's posteriors into the scaled likelihoods a hybrid DNN-HMM system uses.
prereqs:
  - 07-hearing-and-the-mel-scale
  - 13-hidden-markov-models
exercise: 15-neural-acoustic-models
---

# Neural acoustic models

The Gaussians inside an HMM answer one question for every 10 ms frame: how
well does this frame fit this sound? In 2012, neural networks started
answering it much better. The rest of the HMM machinery stayed the same. Only
the scorer changed, and word error rates on hard benchmarks fell sharply.

A neural acoustic model is a classifier that looks at a frame, plus a little
of what surrounds it, and names the sound. That's the first half of this
lesson. The second half is the piece every later model in this course starts
with: a convolutional front end that turns 100 frames a second into 25, so the
expensive layers after it have less to do.

Both halves hinge on counting frames exactly. A frame count that is off by
one breaks an alignment, a mask, or a loss, and it breaks it silently.

## What you already know

- A log-mel spectrogram describes each 25 ms frame, taken every 10 ms, with
  one log energy per mel band. For more information, see *Hearing and the mel
  scale*.
- An HMM needs $\log p(o_t \mid s)$, a score for every frame and state, as
  *Hidden Markov models* showed.
- A classifier trained with cross-entropy outputs a probability for each class,
  and its argmax is its guess.
- A 1-D convolution slides a kernel along a sequence. Its stride is how far the
  kernel moves between outputs.
- In torch, `nn.Conv2d` takes a batch shaped `(N, channels, height, width)`.

## Classify frames

Suppose every frame has a label: which sound, or which HMM state, it belongs
to. Then training a neural acoustic model is ordinary classification. The loss
is the cross-entropy summed over frames:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(s_t \mid x_t)
$$

Read it aloud: for every frame $t$, take the log of the probability the
network gave to the true label $s_t$, add them up, and flip the sign. In torch
it's one call on a `(T, C)` matrix of logits and `T` labels:
`F.cross_entropy(logits, labels, reduction="sum")`.

The catch is the labels. Real corpora such as LibriSpeech come with one
transcript per utterance, not a label per frame. Hybrid systems got frame
labels from an earlier GMM-HMM by forced alignment, the Viterbi path through
the HMM of the known transcript. The synthetic words in this exercise come
with frame labels for free, because the generator knows where each segment
starts. The exercise's `log_mel` frames without centering, so frame $t$
starts at sample $160t$ and is 400 samples long. Label it with the phone at
its center sample, $160t + 200$.

From here on, features are time-first: a `(T, D)` array with one row per
frame, the transpose of the `(bins, frames)` spectrogram layout. Torch layers
treat the last axis as features, so neural models take this layout.

The score of the classifier is **frame accuracy**, the share of frames whose
argmax matches the label:

```python
keep = labels != ignore_index            # padding frames carry label -100
accuracy = (logits.argmax(-1)[keep] == labels[keep]).float().mean()
```

## Add context with stacked frames

One 25 ms frame is a thin slice. A frame on the edge of an "s" looks like
noise, and a frame of a vowel at a low signal-to-noise ratio (SNR) looks like
several vowels. Its neighbors disambiguate it. The simplest way to show them
to a network is to **stack** them: concatenate each frame with $\ell$ frames
before it and $r$ after it into one long vector:

$$
\tilde x_t = \big[x_{t-\ell},\ \dots,\ x_t,\ \dots,\ x_{t+r}\big], \qquad \tilde x_t \in \mathbb{R}^{(\ell + 1 + r) D}
$$

Read it aloud: the input for frame $t$ is the $D$ features of every frame from
$\ell$ before to $r$ after, laid end to end. At the edges, repeat the first or
last frame. In numpy, one clipped index array does all of it:

```python
index = np.clip(np.arange(T)[:, None] + np.arange(-left, right + 1)[None, :], 0, T - 1)
stacked = features[index].reshape(T, -1)
```

For example, take four one-number frames, $[10, 20, 30, 40]$, with one frame
of context on each side. Frame 0 has no left neighbor, so it repeats itself.
The rows are $[10, 10, 20]$, $[10, 20, 30]$, $[20, 30, 40]$, and
$[30, 40, 40]$.

The exercise measures what context buys. At $-5$ dB SNR, where the noise is
louder than the speech, a two-hidden-layer MLP trained on 3,023 frames scores
the following on 1,116 held-out frames:

| Context on each side | Frames seen | Test frame accuracy |
|---|---|---|
| 0 | 1 | 80.6% |
| 1 | 3 | 83.2% |
| 2 | 5 | 85.3% |
| 4 | 9 | 86.5% |
| 8 | 17 | 88.1% |

The gains shrink as the window grows. Each extra frame adds $D$ inputs to the
first layer, and far neighbors belong to other sounds.

> [!KEY]
> A single frame is ambiguous; its neighbors aren't. Stacking frames gives a
> feed-forward network context for the price of a wider first layer.

## Use a classifier inside an HMM

A frame classifier outputs posteriors, $P(s \mid x_t)$: how likely each state
is, given the frame. An HMM needs likelihoods, $p(x_t \mid s)$: how likely the
frame is, given each state. Bayes' rule connects them:

$$
p(x_t \mid s) = \frac{P(s \mid x_t)\, p(x_t)}{P(s)} \quad\Longrightarrow\quad \log p(x_t \mid s) = \log P(s \mid x_t) - \log P(s) + \text{const}_t
$$

Read it aloud: divide the posterior by the state's **prior** $P(s)$, the share
of training frames that carry that label. The remaining factor $p(x_t)$ is the
same for every state at frame $t$, so the Viterbi and forward algorithms
ignore it. These are the **scaled likelihoods** of a hybrid DNN-HMM system. In
numpy:

```python
priors = np.log(np.bincount(labels, minlength=n_classes) / len(labels))
scaled = log_posteriors - priors[None, :]
```

The division matters most for rare states. Suppose silence covers 80% of the
training frames and two phone states cover 15% and 5%. A frame gets posteriors
$[0.5, 0.3, 0.2]$, so the classifier alone would call it silence. Dividing by
the priors gives $[0.625, 2.0, 4.0]$, and the rarest state scores highest. The
classifier had learned to favor silence because silence is common. The HMM
already models how common each state is through its transitions, so counting
the prior twice would bias it.

## Subsample with convolutions

A speech frame every 10 ms is more than a recognizer needs: a single phone
usually lasts several frames. The self-attention layers of later lessons
cost time proportional to the square of the sequence length. So modern
encoders start with a **subsampling front end**, a few strided convolutions
that cut the frame rate before anything expensive runs.

Every length downstream depends on this front end. CTC, in the next lesson,
needs the number of encoder frames to set up its loss, and attention masks
need it to hide padding. So you need the exact output length of a
convolution:

$$
T_{\text{out}} = \left\lfloor \frac{T_{\text{in}} + 2p - d\,(k - 1) - 1}{s} \right\rfloor + 1
$$

Read it aloud: add the padding $p$ to both ends, subtract the span of the
kernel, which is $d(k-1) + 1$ samples for kernel size $k$ and dilation $d$,
count how many strides of $s$ fit, and add one for the first position. This
is the formula torch uses:

```python
out = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
```

Floor division with `//` works on Python integers, numpy arrays, and integer
tensors, so one function serves a single length and a batch of them.

### The 4x Conv2d front end

The front end used by many Transformer and Conformer recognizers treats the
log-mel spectrogram as a one-channel image, with time as height and mel bands
as width. Two `Conv2d` layers with kernel 3, stride 2, and no padding each
halve both axes, with a ReLU after each. The result has shape
$(N, C, T', F')$. Move time next to the batch, flatten the $C$ channels and
$F'$ bands into one vector per frame, and project it to the model width
$d_{\text{model}}$ with a linear layer.

### Work the lengths by hand

Take one second of features, $T = 100$ frames, with 80 mel bands:

- The first convolution: $\lfloor (100 - 2 - 1)/2 \rfloor + 1 = 48 + 1 = 49$
  frames.
- The second: $\lfloor (49 - 2 - 1)/2 \rfloor + 1 = 23 + 1 = 24$ frames.
- The frequency axis goes $80 \to 39 \to 19$ the same way, so the linear layer
  reads $19C$ numbers per frame.

So 100 frames become 24, not 25. The two convolutions together give
$\lfloor(\lfloor(T-1)/2\rfloor - 1)/2\rfloor$. The frame rate is about 25 per
second, a 40 ms stride.

> [!KEY]
> Every length downstream comes from the convolution formula, applied layer
> by layer. The 4x front end turns 100 frames into 24, not 25.

Other systems use other front ends, all computed by the same formula:

- **Whisper** runs two `Conv1d` layers with kernel 3 and padding 1, the second
  with stride 2, so 3,000 log-mel frames become 1,500 encoder positions, one
  per 20 ms.
- **wav2vec 2.0** runs seven 1-D convolutions straight on the 16 kHz
  waveform, with strides 5, 2, 2, 2, 2, 2, and 2, a total of 320 samples. One
  second of 16,000 samples becomes 49 frames.
- **FastConformer** subsamples by 8, to 12.5 frames per second, with three
  stride-2 stages.

> [!WARNING]
> Pass the lengths after subsampling to anything downstream. A padded batch
> of utterances has one length per utterance, and each one shrinks by the
> formula, not by dividing by 4. Getting this wrong makes CTC score padding as
> speech.

> [!RECAP]
> - `stack_frames` gathers frames `t - left` to `t + right` with a clipped
>   index array and reshapes to `(T, (left + 1 + right) * D)`.
> - A convolution's output length is `(T + 2p - d(k - 1) - 1) // s + 1`.
> - The 4x Conv2d front end is conv, ReLU, conv, ReLU, flatten channels and
>   frequency, then a linear layer.
> - Frame accuracy compares `argmax` with the labels, skipping `-100`.
> - Hybrid systems use $\log P(s \mid x) - \log P(s)$ as the emission score.

## Build it step by step

The checks test the functions in this order. The file provides the synthetic
words, `log_mel`, `frame_labels`, and `make_dataset`.

1. **`stack_frames(features, left, right)`.** Takes `(T, D)` and returns
   `(T, (left + 1 + right) * D)`. Build the index with broadcasting and
   `np.clip`, then index once. Common mistake: padding with zeros instead of
   repeating the edge frames.
2. **`conv_output_length(length, kernel, stride, padding, dilation)`.** One
   line of integer arithmetic. Common mistake: using `/` and `math.floor`,
   which breaks on tensors.
3. **`subsampled_length(length)`.** Apply `conv_output_length` twice with
   kernel 3 and stride 2. Common mistake: `length // 4`.
4. **`Conv2dSubsampling`.** Create `conv1`, `conv2`, and `out` with those
   names. In `forward`, `x.unsqueeze(1)` adds the channel axis. After the
   convolutions, `transpose(1, 2)` puts time second, and `reshape(n, t, c * f)`
   flattens. Common mistake: flattening before the transpose, which mixes
   frames together.
5. **`FrameClassifier`.** An `nn.Sequential` of three `Linear` layers with
   ReLUs between them. Common mistake: a softmax at the end, which
   `F.cross_entropy` already applies.
6. **`frame_accuracy(logits, labels, ignore_index)`.** Mask the ignored
   frames, compare `argmax(-1)` with the labels, and average. Common mistake:
   counting padding frames as errors.
7. **`log_priors` and `scaled_log_likelihoods`.** `np.bincount` gives the
   counts; subtract the log priors from every row. Common mistake: adding the
   prior instead of subtracting it.
8. **`train_frame_classifier(inputs, labels, n_classes, ...)`.** Seed torch,
   build the model, and loop: draw a batch with `torch.randint`, compute the
   cross-entropy, and step Adam. Common mistake: forgetting
   `optimizer.zero_grad()`, so the gradients pile up across steps.

## Cost and scaling

The following table uses $T$ input frames, $D$ features per frame, a context
of $w = \ell + 1 + r$ frames, hidden width $H$, and $C$ convolution channels.

| Piece | Parameters | Time per utterance |
|---|---|---|
| Stacked input | none | $O(TwD)$ memory, a copy of every frame $w$ times |
| MLP first layer | $wDH$ | $O(TwDH)$ |
| Conv2d front end | $9C + 9C^2 + C F' d_{\text{model}}$, plus biases | $O(T F C^2)$ |
| Everything after the front end | depends on the model | on $T/4$ frames instead of $T$ |

The front end pays for itself. A Transformer layer after it costs time
proportional to the square of the length, so subsampling by 4 cuts that cost
by 16. Stacking is the opposite trade: it multiplies the input size by $w$, so
windows wider than a few dozen frames are rare in feed-forward models.

## Where this breaks

**Frame labels are expensive.** A hybrid system needs a GMM-HMM to align the
data before a network can be trained on it, and alignment errors become label
noise. CTC, the next lesson, trains from transcripts alone.

**Frames are classified independently.** A frame classifier can flicker
between labels from one frame to the next. The HMM smooths that in a hybrid
system. End-to-end models learn it.

**Mismatched conditions.** Tested at 10 dB, the exercise's single-frame
classifier scores 98.6% when it's trained at 10 dB, and 81.9% when it's
trained at 15 dB. The noise floor in the log-mel features moves, and the
network never saw it move. Train on the conditions you test on, or add noise
to the training data.

**Subsampling can go too far.** At 80 ms per frame, a fast speaker can say
more characters than there are frames. CTC can't emit more labels than it has
frames, which is one reason 8x models predict subword units instead of single
characters.

**Normalization leaks.** Normalize the test set with the mean and standard
deviation of the *training* frames. Computing them over the test set leaks
information into the evaluation.

## The exercise

You build `stack_frames`, `conv_output_length`, `subsampled_length`, the
`Conv2dSubsampling` front end, the `FrameClassifier` MLP, `frame_accuracy`,
the hybrid pieces `log_priors` and `scaled_log_likelihoods`, and
`train_frame_classifier`. Press **Run** to train on synthetic phones at
$-5$ dB with six context widths, plot accuracy against context, hear one noisy
word, see its log-mel features and frame posteriors, and pass it through the
front end.

The checks compare `stack_frames` with an explicit loop, and the length
functions with the output shapes of real `Conv1d` and `Conv2d` layers. They
confirm the front end's shapes and lengths on a padded batch, and recompute
its output from its own weights. They check the classifier's parameter count,
frame accuracy with ignored frames, and that scaled likelihoods match Bayes'
rule on a small joint distribution. The last check trains two classifiers:
with four frames of context on each side, the classifier must reach 83% frame
accuracy at $-5$ dB and beat a single frame by 3 points.
