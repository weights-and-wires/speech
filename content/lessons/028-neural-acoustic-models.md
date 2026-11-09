---
title: Neural acoustic models
slug: 028-neural-acoustic-models
part: "Part 4 — Neural speech recognition"
summary: A network names each frame from the frames around it, then a strided front end cuts the frame rate, and you count the frames exactly.
minutes: 40
gpu: false
objectives:
  - Explain why one 25 ms frame is ambiguous in noise and why stacking neighbor frames helps.
  - Stack each frame with its neighbors into one context vector, repeating the edge frames.
  - Compute the exact output length of any convolution, and of the 4x Conv2d front end.
  - Build that front end and a frame classifier in torch, and train the classifier on noisy synthetic phones.
prereqs:
  - 012-hearing-and-the-mel-scale
  - 022-hidden-markov-models
exercise: 028-neural-acoustic-models
---

# Neural acoustic models

A voice assistant in a noisy kitchen hears the start of "sa". For 25 ms the microphone picks up the hiss of a pan and a little of the speaker's "s", and the two look almost the same in that one slice. Whatever decides which sound this is has too little to go on.

A neural acoustic model fixes this in two moves. It shows the network the frames around the one it is naming, so the network can use what came before and after. Then it throws away most of the frames, because 100 decisions a second is far more than a phone needs. Both moves depend on counting frames exactly, and a count that is off by one breaks every alignment and loss later in the course without raising an error.

This chapter builds both moves and the classifier on top of them. Later chapters keep the frame-rate cutter and replace the classifier.

## What you already know

- A log-mel spectrogram has one column per 25 ms frame, taken every 10 ms, from [Hearing and the mel scale](/lessons/012-hearing-and-the-mel-scale).
- An HMM needs a score for every frame and state, as [Hidden Markov models](/lessons/022-hidden-markov-models) showed.
- A classifier trained with cross-entropy gives a probability to each class, and its argmax is its guess.
- A 1-D convolution slides a kernel along a sequence, and its stride is how far the kernel moves between outputs.
- In torch, `nn.Conv2d` takes a batch shaped `(N, channels, height, width)`.

## Name every frame

Suppose every frame has a label: which sound it belongs to. Then training a neural acoustic model is ordinary classification. The loss is the cross-entropy summed over frames:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(s_t \mid x_t)
$$

Read it aloud: for every frame $t$, take the log of the probability the network gave to the true label $s_t$, add the logs up, and flip the sign. Here $x_t$ is the features of frame $t$ and $T$ is the number of frames. In torch it is one call on a `(T, C)` matrix of logits and `T` labels: `F.cross_entropy(logits, labels, reduction="sum")`.

The catch is the labels. A corpus such as LibriSpeech gives one transcript per recording, not one label per frame. Hybrid systems got frame labels from an earlier GMM-HMM by forced alignment, the Viterbi path through the HMM of the known transcript. The synthetic words in this chapter's exercise come with labels for free, because the generator knows where each segment starts: frame $t$ starts at sample $160t$, is 400 samples long, and takes the phone at its center sample, $160t + 200$.

From here on, features are time-first: a `(T, D)` array with one row per frame, the transpose of a spectrogram's `(bins, frames)` layout, because torch layers treat the last axis as features. The classifier's score is **frame accuracy**, the share of frames whose argmax equals the label.

## Why one frame is not enough

The obvious classifier looks at frame $t$ and nothing else. In the exercise's noisy words it names the right phone for 80.6% of held-out frames at $-5$ dB signal-to-noise ratio (SNR), where the noise is louder than the speech. That leaves almost one frame in five wrong.

The errors cluster where a frame is ambiguous: the edge of an "s" looks like noise, and a vowel under noise looks like several vowels. The neighbors settle the question, because they belong to the same sound or to its clear neighbors. So show them. The simplest way is to **stack** them: concatenate each frame with $\ell$ frames before it and $r$ frames after it into one long vector.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Four one-number frames, 10, 20, 30, and 40, each stacked with one neighbor on each side; the first and last rows repeat an edge frame">
  <text x="30" y="28" class="fig-bold">frame t</text>
  <text x="200" y="28" class="fig-bold">stacked input for frame t (left 1, right 1)</text>
  <rect x="30" y="44" width="70" height="34" rx="5" class="fill-soft-a stroke-a"/>
  <text x="65" y="66" text-anchor="middle">t=0: 10</text>
  <rect x="30" y="88" width="70" height="34" rx="5" class="fill-soft-a stroke-a"/>
  <text x="65" y="110" text-anchor="middle">t=1: 20</text>
  <rect x="30" y="132" width="70" height="34" rx="5" class="fill-soft-a stroke-a"/>
  <text x="65" y="154" text-anchor="middle">t=2: 30</text>
  <rect x="30" y="176" width="70" height="34" rx="5" class="fill-soft-a stroke-a"/>
  <text x="65" y="198" text-anchor="middle">t=3: 40</text>
  <rect x="200" y="44" width="70" height="34" rx="5" class="fill-soft-b stroke-b dashed"/>
  <text x="235" y="66" text-anchor="middle" class="fig-mono">10</text>
  <rect x="280" y="44" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="315" y="66" text-anchor="middle" class="fig-mono fig-bold">10</text>
  <rect x="360" y="44" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="395" y="66" text-anchor="middle" class="fig-mono">20</text>
  <rect x="200" y="88" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="235" y="110" text-anchor="middle" class="fig-mono">10</text>
  <rect x="280" y="88" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="315" y="110" text-anchor="middle" class="fig-mono fig-bold">20</text>
  <rect x="360" y="88" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="395" y="110" text-anchor="middle" class="fig-mono">30</text>
  <rect x="200" y="132" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="235" y="154" text-anchor="middle" class="fig-mono">20</text>
  <rect x="280" y="132" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="315" y="154" text-anchor="middle" class="fig-mono fig-bold">30</text>
  <rect x="360" y="132" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="395" y="154" text-anchor="middle" class="fig-mono">40</text>
  <rect x="200" y="176" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="235" y="198" text-anchor="middle" class="fig-mono">30</text>
  <rect x="280" y="176" width="70" height="34" rx="5" class="fill-sunk stroke-b"/>
  <text x="315" y="198" text-anchor="middle" class="fig-mono fig-bold">40</text>
  <rect x="360" y="176" width="70" height="34" rx="5" class="fill-soft-b stroke-b dashed"/>
  <text x="395" y="198" text-anchor="middle" class="fig-mono">40</text>
  <text x="450" y="66" class="fig-small">dashed: a repeat of the edge frame</text>
  <text x="450" y="154" class="fig-small">bold: the frame itself</text>
  <text x="30" y="236" class="fig-small">Each row is one input vector, three numbers wide instead of one.</text>
</svg>
Figure 1. Stacking one frame of context on each side turns four one-number frames into four three-number inputs. The first row has no left neighbor, so it repeats frame 0, and the last row repeats frame 3.
```

Work the example by hand. Take four one-number frames, $[10, 20, 30, 40]$, with one frame of context on each side, as in Figure 1. Frame 0 has no left neighbor, so it repeats itself and its row is $[10, 10, 20]$. The middle rows are $[10, 20, 30]$ and $[20, 30, 40]$, and the last row is $[30, 40, 40]$. Repeating the edge frame beats padding with zeros, because a row of zeros looks like a sound the network has never heard.

In symbols, the input for frame $t$ is

$$
\tilde x_t = \big[x_{t-\ell},\ \dots,\ x_t,\ \dots,\ x_{t+r}\big], \qquad \tilde x_t \in \mathbb{R}^{(\ell + 1 + r) D}
$$

Read it aloud: lay the $D$ features of every frame from $\ell$ before to $r$ after end to end. The vector is $(\ell + 1 + r)$ times as wide, so only the first layer of the network grows. In numpy, one clipped index array does all of it, and the clip is what repeats the edges:

```python
index = np.clip(np.arange(T)[:, None] + np.arange(-left, right + 1)[None, :], 0, T - 1)
stacked = features[index].reshape(T, -1)
```

The exercise measures what context buys. A two-hidden-layer network trained on 3,023 frames at $-5$ dB scores the following on 1,116 held-out frames.

| Context on each side | Frames seen | Test frame accuracy |
|---|---|---|
| 0 | 1 | 80.6% |
| 1 | 3 | 83.2% |
| 2 | 5 | 85.3% |
| 4 | 9 | 86.5% |
| 8 | 17 | 88.1% |

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Test frame accuracy rises from 80.6 percent with no context to 88.1 percent with eight frames of context on each side, with smaller gains at each step">
  <line x1="80" y1="200" x2="540" y2="200" class="stroke-muted"/>
  <line x1="80" y1="200" x2="80" y2="40" class="stroke-muted"/>
  <line x1="75" y1="200" x2="80" y2="200" class="stroke-muted"/>
  <text x="70" y="204" text-anchor="end" class="fig-small">78</text>
  <line x1="75" y1="120" x2="80" y2="120" class="stroke-muted"/>
  <text x="70" y="124" text-anchor="end" class="fig-small">84</text>
  <line x1="75" y1="40" x2="80" y2="40" class="stroke-muted"/>
  <text x="70" y="44" text-anchor="end" class="fig-small">90</text>
  <line x1="80" y1="120" x2="540" y2="120" class="stroke-grid dashed"/>
  <text x="30" y="30" class="fig-small">test frame accuracy (%)</text>
  <polyline points="80,165.3 135,130.7 190,102.7 300,86.7 520,65.3" class="stroke-b" fill="none"/>
  <circle cx="80" cy="165.3" r="5" class="fill-b"/>
  <circle cx="135" cy="130.7" r="5" class="fill-b"/>
  <circle cx="190" cy="102.7" r="5" class="fill-b"/>
  <circle cx="300" cy="86.7" r="5" class="fill-b"/>
  <circle cx="520" cy="65.3" r="5" class="fill-b"/>
  <text x="80" y="188" text-anchor="middle" class="fig-small">80.6</text>
  <text x="135" y="153" text-anchor="middle" class="fig-small">83.2</text>
  <text x="190" y="124" text-anchor="middle" class="fig-small">85.3</text>
  <text x="300" y="108" text-anchor="middle" class="fig-small">86.5</text>
  <text x="520" y="87" text-anchor="middle" class="fig-small">88.1</text>
  <line x1="80" y1="200" x2="80" y2="205" class="stroke-muted"/>
  <text x="80" y="222" text-anchor="middle" class="fig-small">0</text>
  <line x1="135" y1="200" x2="135" y2="205" class="stroke-muted"/>
  <text x="135" y="222" text-anchor="middle" class="fig-small">1</text>
  <line x1="190" y1="200" x2="190" y2="205" class="stroke-muted"/>
  <text x="190" y="222" text-anchor="middle" class="fig-small">2</text>
  <line x1="300" y1="200" x2="300" y2="205" class="stroke-muted"/>
  <text x="300" y="222" text-anchor="middle" class="fig-small">4</text>
  <line x1="520" y1="200" x2="520" y2="205" class="stroke-muted"/>
  <text x="520" y="222" text-anchor="middle" class="fig-small">8</text>
  <text x="310" y="250" text-anchor="middle">frames of context on each side</text>
</svg>
Figure 2. Accuracy at −5 dB SNR climbs with context, but each extra frame buys less. Going from 0 to 1 frame gains 2.6 points, and going from 4 to 8 gains 1.6 points for four times as many added frames.
```

The gains shrink because far neighbors belong to other sounds, while each extra frame still adds $D$ inputs to the first layer.

> [!KEY]
> A single frame is ambiguous, and its neighbors are not. Stacking frames gives a feed-forward network context for the price of a wider first layer.

## Why 100 decisions a second is too many

Now the second question. A phone lasts several frames, so a network that decides every 10 ms repeats itself many times per sound, and that repetition costs real money later. The self-attention layers of the encoders in [Conformers and SpecAugment](/lessons/036-conformers-and-specaugment) cost time proportional to the square of the number of frames, so halving the frames cuts that cost by 4.

The obvious fix is to keep every fourth frame. That throws information away, because a short burst such as a "t" can fall entirely between the kept frames. A **strided convolution** keeps it. Each output is a weighted sum over several neighboring inputs, and the stride says how far the kernel jumps between outputs, so every input frame still contributes and the number of outputs shrinks. A stack of such layers at the front of the encoder is a **subsampling front end**.

Every later length depends on this front end. The CTC loss in [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label) needs the number of encoder frames, and attention masks need it to hide padding. So you need the exact output length of a convolution:

$$
T_{\text{out}} = \left\lfloor \frac{T_{\text{in}} + 2p - d\,(k - 1) - 1}{s} \right\rfloor + 1
$$

Read it aloud: pad both ends by $p$, subtract the span the kernel covers, which is $d(k-1) + 1$ samples for kernel size $k$ and dilation $d$, count how many strides of $s$ fit in what is left, and add one for the first position. In integer arithmetic this is one line, and floor division `//` works on Python integers, numpy arrays, and integer tensors alike:

```python
out = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
```

### The 4x front end

The front end in many Transformer and Conformer recognizers treats the log-mel spectrogram as a one-channel image, with time as height and mel bands as width. Two `Conv2d` layers with kernel 3, stride 2, and no padding each halve both axes, with a ReLU after each. The result has shape $(N, C, T', F')$, where $C$ is the channel count. You move time next to the batch, flatten the $C$ channels and $F'$ bands into one vector per frame, and project it to the model width $d_{\text{model}}$ with a linear layer.

Work the lengths by hand for one second of features, $T = 100$ frames, with 80 mel bands:

- The first convolution gives $\lfloor (100 - 2 - 1)/2 \rfloor + 1 = 48 + 1 = 49$ frames.
- The second gives $\lfloor (49 - 2 - 1)/2 \rfloor + 1 = 23 + 1 = 24$ frames.
- The frequency axis goes $80 \to 39 \to 19$ the same way, so the linear layer reads $19C$ numbers per frame.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Three strips of frames, 100 at 10 milliseconds, 49 at 20 milliseconds, and 24 at 40 milliseconds, shrinking after each stride-2 convolution">
  <text x="20" y="38" class="fig-small">input</text>
  <text x="20" y="54" class="fig-small">T = 100</text>
  <rect x="120" y="30" width="400" height="28" rx="3" class="fill-soft-a stroke-a"/>
  <text x="530" y="49" class="fig-small">10 ms each</text>
  <text x="20" y="98" class="fig-small">after conv 1</text>
  <text x="20" y="114" class="fig-small">T = 49</text>
  <rect x="120" y="90" width="196" height="28" rx="3" class="fill-soft-b stroke-b"/>
  <text x="326" y="109" class="fig-small">20 ms each</text>
  <text x="20" y="158" class="fig-small">after conv 2</text>
  <text x="20" y="174" class="fig-small">T = 24</text>
  <rect x="120" y="150" width="96" height="28" rx="3" class="fill-soft-b stroke-b"/>
  <text x="226" y="169" class="fig-small">40 ms each</text>
  <line x1="320" y1="62" x2="320" y2="86" class="stroke-muted dashed"/>
  <text x="332" y="78" class="fig-small">floor((100 − 3) / 2) + 1</text>
  <line x1="220" y1="122" x2="220" y2="146" class="stroke-muted dashed"/>
  <text x="232" y="138" class="fig-small">floor((49 − 3) / 2) + 1</text>
  <text x="120" y="212" class="fig-small">One bar is one second. Widths are four units per frame: 400, 196, and 96.</text>
</svg>
Figure 3. Two stride-2 convolutions turn 100 frames into 24, not 25. Each unpadded convolution loses a frame at the edge, which is why you compute lengths with the formula and never divide by 4.
```

So 100 frames become 24, not 25. The frame rate falls to about 25 per second, one frame per 40 ms. For a batch, you apply the formula to every utterance's own length, and you never divide the padded length by 4.

> [!KEY]
> Every length downstream comes from the convolution formula, applied layer by layer. The 4x front end turns 100 frames into 24.

Other systems use the same formula with other settings. The numbers below are the ones the research notes for this course record:

- **Whisper** runs two `Conv1d` layers with kernel 3 and padding 1, the second with stride 2, so 3,000 log-mel frames become 1,500 encoder positions, one per 20 ms.
- **wav2vec 2.0** runs seven 1-D convolutions straight on the 16 kHz waveform, with strides 5, 2, 2, 2, 2, 2, and 2, a total of 320 samples. One second of 16,000 samples becomes 49 frames.
- **FastConformer** subsamples by 8, to 12.5 frames per second, with three stride-2 stages.

> [!WARNING]
> Pass the lengths after subsampling to anything downstream. A padded batch has one length per utterance, and each shrinks by the formula. If you pass the old lengths, CTC scores padding as speech.

> [!RECAP]
> - `stack_frames` gathers frames `t - left` to `t + right` with a clipped index array.
> - A convolution's output length is `(T + 2p - d(k - 1) - 1) // s + 1`.
> - The 4x front end is conv, ReLU, conv, ReLU, flatten channels and frequency, then a linear layer.
> - Frame accuracy compares `argmax` with the labels and skips padding frames.

## Build it step by step

The checks test the functions in this order. The file provides the synthetic words, `log_mel`, `frame_labels`, and `make_dataset`.

1. **`stack_frames(features, left, right)`.** Takes `(T, D)` and returns `(T, (left + 1 + right) * D)`. Build the index with broadcasting and `np.clip`, then index once. The common mistake is padding with zeros.
2. **`conv_output_length(length, kernel, stride, padding, dilation)`.** One line of integer arithmetic. The common mistake is using `/` and `math.floor`, which breaks on tensors.
3. **`subsampled_length(length)`.** Apply `conv_output_length` twice with kernel 3 and stride 2. The common mistake is `length // 4`.
4. **`Conv2dSubsampling`.** Create `conv1`, `conv2`, and `out` with those names. In `forward`, `x.unsqueeze(1)` adds the channel axis. After the convolutions, `transpose(1, 2)` puts time second, and `reshape(n, t, c * f)` flattens. The common mistake is flattening before the transpose, which mixes frames together.
5. **`FrameClassifier`.** An `nn.Sequential` of three `Linear` layers with ReLUs between them. The common mistake is a softmax at the end, which `F.cross_entropy` already applies.
6. **`frame_accuracy(logits, labels, ignore_index)`.** Mask the ignored frames, compare `argmax(-1)` with the labels, and average. The common mistake is counting padding frames as errors.
7. **`train_frame_classifier(inputs, labels, n_classes, ...)`.** Seed torch, build the model, and loop: draw a batch with `torch.randint`, compute the cross-entropy, and step Adam. The common mistake is forgetting `optimizer.zero_grad()`, so the gradients pile up across steps.

## Where this breaks

**Frame labels are expensive.** A hybrid system needs a GMM-HMM to align the data before a network can be trained on it, and every alignment error becomes label noise. The next chapters train from transcripts alone.

**Hybrid systems divide by the prior.** An HMM needs the likelihood $p(x_t \mid s)$, and a classifier gives the posterior $P(s \mid x_t)$. Bayes' rule links them: divide the posterior by the state's prior $P(s)$, the share of training frames with that label. If silence covers 80% of the frames and two phone states cover 15% and 5%, posteriors $[0.5, 0.3, 0.2]$ divide to $[0.625, 2.0, 4.0]$, so the rarest state wins. Without the division the HMM counts how common silence is twice.

**Mismatched conditions.** Tested at 10 dB, a single-frame classifier scores 98.6% when trained at 10 dB and 81.9% when trained at 15 dB. The noise floor moved, and the network never saw it move. Train on the conditions you test on, or add noise to the training data.

**Subsampling can go too far.** At 80 ms per frame, a fast speaker can say more characters than there are frames, and CTC cannot emit more labels than it has frames. This is one reason 8x models predict subword units instead of single characters.

## The exercise

You build `stack_frames`, `conv_output_length`, `subsampled_length`, the `Conv2dSubsampling` front end, the `FrameClassifier` network, `frame_accuracy`, and `train_frame_classifier`. Press **Run** to train at six context widths, plot accuracy against context, hear one noisy word, and pass it through the front end.

The checks compare `stack_frames` with an explicit loop, and the length functions with the output shapes of real `Conv1d` and `Conv2d` layers. They confirm the front end's shapes and lengths on a padded batch, and recompute its output from its own weights. The last check trains two classifiers: with four frames of context on each side, the classifier must reach 83% frame accuracy at $-5$ dB and beat a single frame by 3 points.

> [!KEY]
> What you now know: one 25 ms frame is too ambiguous in noise, so a neural acoustic model stacks neighbors to see context, then cuts the frame rate with strided convolutions whose output lengths you can compute exactly. That answers the opening question: the hiss of the pan and the hiss of the "s" differ in what surrounds them.
