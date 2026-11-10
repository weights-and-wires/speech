---
title: Neural acoustic models
slug: 028-neural-acoustic-models
part: "Part 4 — Neural speech recognition"
summary: A network names each frame from its neighbors, TDNNs and LSTMs supply the context, and you count frames exactly.
minutes: 40
gpu: false
objectives:
  - Explain why one 25 ms frame is ambiguous in noise and why stacking neighbor frames helps.
  - Stack each frame with its neighbors into one context vector, repeating the edge frames.
  - Work out the receptive field of a stack of time-delay layers, and run one layer as a dilated convolution.
  - Say how a frame classifier's posteriors become the HMM scores of a hybrid system.
  - Compute the exact output length of any convolution, and of the 4x Conv2d front end.
  - Build that front end and a frame classifier in torch, and train the classifier on noisy synthetic phones.
prereqs:
  - 012-hearing-and-the-mel-scale
  - 022-hidden-markov-models
  - 024-hybrid-acoustic-models
exercise: 028-neural-acoustic-models
---

# Neural acoustic models

A voice assistant in a noisy kitchen hears the start of "sa". For 25 ms the microphone picks up the hiss of a pan and a little of the speaker's "s", and the two look almost the same. Whatever names this sound has too little to go on.

A neural acoustic model fixes this in two moves. It shows the network the frames around the one it is naming. Then it throws away most of the frames, because 100 decisions a second is far more than a phone needs. A frame count that is off by one breaks every alignment and loss later in the course without raising an error.

This chapter builds both moves, then the TDNN and LSTM that put context inside the network. Later chapters replace the classifier.

## What you already know

- A log-mel spectrogram has one column per 25 ms frame, taken every 10 ms, from [Hearing and the mel scale](/lessons/012-hearing-and-the-mel-scale).
- An HMM needs a score for every frame and state, as [Hidden Markov models](/lessons/022-hidden-markov-models) showed.
- A cross-entropy classifier gives a probability to each class.
- A 1-D convolution slides a kernel along a sequence, with a stride between outputs.
- In torch, `nn.Conv2d` takes `(N, channels, height, width)`.

## Name every frame

Suppose every frame has a label: which sound it belongs to. Then training a neural acoustic model is ordinary classification. The loss is the cross-entropy summed over frames:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(s_t \mid x_t)
$$

Read it aloud: for each of the $T$ frames, take the log of the probability the network gave to the true label $s_t$ given the features $x_t$, add the logs up, and flip the sign.

The catch is the labels. A corpus gives one transcript per recording, not one label per frame, so hybrid systems got frame labels from an earlier GMM-HMM by forced alignment. The exercise's synthetic words come with labels: frame $t$ takes the phone at its center sample, $160t + 200$.

Features are time-first from here on: a `(T, D)` array, one row per frame. The score is **frame accuracy**, the share of frames whose argmax equals the label.

## Why one frame is not enough

A classifier that looks at frame $t$ alone names the right phone for 80.6% of held-out frames in the exercise's noisy words at $-5$ dB signal-to-noise ratio (SNR), where the noise is louder than the speech. Errors cluster where a frame is ambiguous, such as the edge of an "s". The neighbors settle the question, and the simplest way to show them is to **stack** them: concatenate each frame with $\ell$ frames before it and $r$ frames after it into one long vector.

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

Take four one-number frames, $[10, 20, 30, 40]$, with one frame of context on each side, as in Figure 1. Frame 0 has no left neighbor, so it repeats itself and its row is $[10, 10, 20]$. The middle rows are $[10, 20, 30]$ and $[20, 30, 40]$, and the last row is $[30, 40, 40]$. Repeating the edge frame beats padding with zeros, because a row of zeros looks like a sound the network has never heard.

In symbols, the input for frame $t$ is

$$
\tilde x_t = \big[x_{t-\ell},\ \dots,\ x_t,\ \dots,\ x_{t+r}\big], \qquad \tilde x_t \in \mathbb{R}^{(\ell + 1 + r) D}
$$

The $D$ features of every frame from $\ell$ before to $r$ after lie end to end, so only the first layer of the network grows. In numpy, one clipped index array does all of it, and the clip is what repeats the edges:

```python
index = np.clip(np.arange(T)[:, None] + np.arange(-left, right + 1)[None, :], 0, T - 1)
stacked = features[index].reshape(T, -1)
```

In the exercise, a two-hidden-layer network trained at $-5$ dB scores 80.6%, 83.2%, 85.3%, 86.5%, and 88.1% on held-out frames with 0, 1, 2, 4, and 8 frames of context on each side.

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

The gains shrink because far neighbors belong to other sounds.

> [!KEY]
> A single frame is ambiguous, and its neighbors are not. Stacking frames gives a feed-forward network context for the price of a wider first layer.

## Put the context inside the network

Stacking splices all the context at the input, so the first layer carries it. A **time-delay neural network (TDNN)**, introduced by Waibel and colleagues in 1989 for phoneme recognition, splices at every layer instead. A time-delay layer computes frame $t$ from the layer below at a few offsets, with the same weights at every $t$:

$$
h_t = W \,[\, x_{t+o_1};\ x_{t+o_2};\ \dots;\ x_{t+o_K} \,]
$$

Read it aloud: join the input frames $x$ at $K$ offsets $o_k$ from frame $t$ and multiply by the weights $W$ to get the output $h_t$. That is a 1-D convolution over frames, with a gap between offsets as its dilation: $\{-3, +3\}$ is a kernel of size 2 with dilation 6.

Work out how much input one output frame depends on, for four layers with offsets $\{-2, \dots, +2\}$, $\{-1, +2\}$, $\{-3, +3\}$, and $\{0\}$. Start at the top. The last layer reads layer 3 at $t$. Layer 3 reads layer 2 at $t \pm 3$. Layer 2 reads layer 1 one frame left and two right of those, at $t - 4$, $t - 1$, $t + 2$, and $t + 5$. Layer 1 reads the input two frames either side, from $t - 6$ to $t + 7$. Each layer widens the window by its own extent, so the extents add: left $2 + 1 + 3 = 6$ and right $2 + 2 + 3 = 7$. Taking the largest extent instead of the sum gives $(3, 3)$, less than half the real window.

Peddinti, Povey, and Khudanpur (2015) add a layer $\{-7, +2\}$ before the last: $2 + 1 + 3 + 7 = 13$ frames left and $2 + 2 + 3 + 2 = 9$ right, 23 frames in all. They favor the left because right context adds latency in online decoding.

Their second idea is to compute fewer offsets. Neighboring frames share most of their inputs, so computing every frame repeats work. With gaps between the offsets, one output frame needs only some of each layer's frames, as Figure 3 shows. The gaps 3, 6, and 9 are multiples of 3 so the needed frames land on one grid. Layers 1 to 3 need 7, 4, and 2 frames instead of 19, 16, and 10: 14 activations instead of 46, with the same 23-frame window. The authors report about five times faster training.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Frames a TDNN computes at each layer for one output frame: 1, 2, 4, and 7 with subsampling, against 1, 10, 16, and 19 without">
  <text x="8" y="44" class="fig-small">layer 4 out</text>
  <circle cx="340" cy="40" r="6" class="fill-b stroke-b"/>
  <text x="8" y="86" class="fig-small">layer 3 out</text>
  <circle cx="200" cy="82" r="6" class="fill-b stroke-b"/>
  <circle cx="220" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="240" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="260" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="280" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="300" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="320" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="340" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="360" cy="82" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="380" cy="82" r="6" class="fill-b stroke-b"/>
  <text x="8" y="128" class="fig-small">layer 2 out</text>
  <circle cx="140" cy="124" r="6" class="fill-b stroke-b"/>
  <circle cx="160" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="180" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="200" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="220" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="240" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="260" cy="124" r="6" class="fill-b stroke-b"/>
  <circle cx="280" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="300" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="320" cy="124" r="6" class="fill-b stroke-b"/>
  <circle cx="340" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="360" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="380" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="400" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="420" cy="124" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="440" cy="124" r="6" class="fill-b stroke-b"/>
  <text x="8" y="170" class="fig-small">layer 1 out</text>
  <circle cx="120" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="140" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="160" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="180" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="200" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="220" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="240" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="260" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="280" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="300" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="320" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="340" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="360" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="380" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="400" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="420" cy="166" r="6" class="fill-b stroke-b"/>
  <circle cx="440" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="460" cy="166" r="6" class="fill-sunk stroke-muted"/>
  <circle cx="480" cy="166" r="6" class="fill-b stroke-b"/>
  <text x="8" y="212" class="fig-small">input</text>
  <circle cx="80" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="100" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="120" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="140" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="160" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="180" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="200" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="220" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="240" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="260" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="280" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="300" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="320" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="340" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="360" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="380" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="400" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="420" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="440" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="460" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="480" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="500" cy="208" r="6" class="fill-b stroke-b"/>
  <circle cx="520" cy="208" r="6" class="fill-b stroke-b"/>
  <line x1="340" y1="22" x2="340" y2="226" class="stroke-muted dashed"/>
  <text x="80" y="246" text-anchor="middle" class="fig-small">-13</text>
  <text x="340" y="246" text-anchor="middle" class="fig-small">+0</text>
  <text x="520" y="246" text-anchor="middle" class="fig-small">+9</text>
  <text x="310" y="264" text-anchor="middle" class="fig-small">frame offset from the output frame</text>
</svg>
Figure 3. For one output frame at offset 0, the filled dots are the frames the subsampled TDNN computes at each layer: 1, 2, 4, and 7. The hollow dots are the extra frames a contiguous splice would also compute, 10, 16, and 19 at layers 3, 2, and 1.
```

A long short-term memory network (LSTM; Hochreiter and Schmidhuber, 1997) supplies context another way. It reads one frame at a time and carries a gated state forward, so frame $t$ can depend on every earlier frame. There is no receptive field to count. Sak, Senior, and Beaufays (2014) beat a feed-forward network with about ten times as many parameters using a two-layer LSTM.

Two costs follow. The state at $t$ needs the state at $t - 1$, so the frames of one utterance cannot run in parallel as a TDNN's can. And right context needs a second LSTM that runs backward from the end, a **bidirectional LSTM (BLSTM)**. Each output then sees the whole utterance, but none can appear until the last frame arrives. Later encoders keep a convolutional front end and replace the recurrence with self-attention, where any two frames interact in one parallel step. [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment) builds that.

## Why 100 decisions a second is too many

A phone lasts several frames, so a network that decides every 10 ms repeats itself, and the repetition costs real money later. Self-attention cost grows with the square of the frame count, so halving the frames cuts it by 4.

Keeping every fourth frame throws information away, because a short burst such as a "t" can fall between the kept frames. A **strided convolution** keeps it: each output is a weighted sum over several neighboring inputs, so every input frame still contributes while the number of outputs shrinks. A stack of such layers at the front of the encoder is a **subsampling front end**.

The CTC loss in [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label) and attention masks both need the exact number of encoder frames, so you need the exact output length of a convolution:

$$
T_{\text{out}} = \left\lfloor \frac{T_{\text{in}} + 2p - d\,(k - 1) - 1}{s} \right\rfloor + 1
$$

Read it aloud: pad both ends by $p$, subtract the $d(k-1) + 1$ samples the kernel covers (kernel size $k$, dilation $d$), count how many strides of $s$ fit in what is left, and add one. Floor division `//` works on integers, arrays, and integer tensors alike:

```python
out = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
```

### The 4x front end

Many Transformer and Conformer recognizers treat the log-mel spectrogram as a one-channel image, with time as height and mel bands as width. Two `Conv2d` layers with kernel 3, stride 2, and no padding each halve both axes, with a ReLU after each. The result has shape $(N, C, T', F')$. You flatten the $C$ channels and $F'$ bands into one vector per frame and project it to the model width $d_{\text{model}}$ with a linear layer.

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
Figure 4. Two stride-2 convolutions turn 100 frames into 24, not 25. Each unpadded convolution loses a frame at the edge, which is why you compute lengths with the formula and never divide by 4.
```

So 100 frames become 24, not 25. For a batch, apply the formula to each utterance's own length, and never divide the padded length by 4.

> [!KEY]
> Every length downstream comes from the convolution formula, applied layer by layer. The 4x front end turns 100 frames into 24.

> [!WARNING]
> Pass the subsampled lengths downstream, or CTC scores padding as speech.

## Hand the posteriors to an HMM

Each network above ends in a softmax, so frame $t$ gets a distribution $P(s \mid x_t)$ over states $s$. An HMM needs the opposite, $p(x_t \mid s)$. Divide the posterior by the prior $P(s)$, the share of training frames with label $s$, and the HMM scores each frame with the **scaled likelihood** $P(s \mid x_t) / P(s)$. This pairing is the DNN-HMM hybrid. [Hybrid acoustic models](/lessons/024-hybrid-acoustic-models) derives the division and decodes the result with Viterbi, so this chapter only supplies the network. A real hybrid classifies tied triphone states, not nine phones. If silence covers 80% of the frames and two phone states 15% and 5%, posteriors $[0.5, 0.3, 0.2]$ divide to $[0.625, 2.0, 4.0]$, and the rarest state wins.

> [!RECAP]
> - A time-delay layer is a dilated 1-D convolution, and a stack's receptive field is the sum of the per-layer extents.
> - An LSTM has unbounded left context but runs frame by frame, and a BLSTM waits for the whole utterance.
> - A convolution's output length is `(T + 2p - d(k - 1) - 1) // s + 1`.
> - A hybrid system divides the classifier's posteriors by the state priors.

## Build it step by step

The checks test the functions in this order. The file provides `log_mel`, `frame_labels`, and `make_dataset`.

1. **`stack_frames(features, left, right)`.** Takes `(T, D)` and returns `(T, (left + 1 + right) * D)`. Index with broadcasting and `np.clip`. The common mistake is padding with zeros.
2. **`conv_output_length(length, kernel, stride, padding, dilation)`.** One line of integer arithmetic. The common mistake is `/`, which breaks on tensors.
3. **`subsampled_length(length)`.** Apply `conv_output_length` twice with kernel 3 and stride 2. The common mistake is `length // 4`.
4. **`Conv2dSubsampling`.** Create `conv1`, `conv2`, and `out`. In `forward`, `unsqueeze(1)` adds channels, `transpose(1, 2)` puts time second, and `reshape(n, t, c * f)` flattens. The common mistake is flattening before the transpose, which mixes frames together.
5. **`FrameClassifier`.** Three `Linear` layers with ReLUs between them. The common mistake is a final softmax, which `F.cross_entropy` already applies.
6. **`frame_accuracy(logits, labels, ignore_index)`.** Mask the ignored frames, compare `argmax(-1)` with the labels, and average. The common mistake is scoring padding.
7. **`train_frame_classifier(inputs, labels, n_classes, ...)`.** Seed torch, then loop: draw a batch with `torch.randint`, compute the cross-entropy, and step Adam. The common mistake is skipping `optimizer.zero_grad()`.
8. **`tdnn_layer(x, weights, offsets)`.** Keep the frames $t$ where every `t + o` exists, index with broadcasting but no clip, and multiply the rows by `weights.T`. The common mistake is dropping one frame too many.
9. **`tdnn_receptive_field(contexts)`.** Per layer, take `max(0, -min(offsets))` and `max(0, max(offsets))`, then add over layers. The common mistake is the max over layers instead of the sum.

## Where this breaks

**Frame labels are expensive.** Every alignment error from the GMM-HMM becomes label noise. The next chapters train from transcripts alone.

**Mismatched conditions.** Tested at 10 dB, a single-frame classifier scores 98.6% when trained at 10 dB and 81.9% when trained at 15 dB, so train on the conditions you test on.

**Subsampling can go too far.** At 80 ms per frame, a fast speaker can say more characters than there are frames, and CTC cannot emit more labels than frames.

## The exercise

Press **Run** to train at six context widths, plot accuracy against context, and pass one noisy word through the front end.

The checks compare the length functions with real `Conv1d` and `Conv2d` shapes and recompute the front end from its own weights. One check trains two classifiers: with four frames of context on each side, the classifier must reach 83% frame accuracy at $-5$ dB and beat a single frame by 3 points. `tdnn_layer` must match `torch.nn.functional.conv1d` with the matching dilation, and `tdnn_receptive_field` must return $(6, 7)$ and $(13, 9)$ for the two stacks above.

> [!KEY]
> What you now know: one 25 ms frame is too ambiguous in noise, so a neural acoustic model sees context, from stacked frames, TDNN layers, or an LSTM, then cuts the frame rate with strided convolutions whose lengths you can compute exactly. The hiss of the pan and the hiss of the "s" differ in what surrounds them.

## Further reading

The numbers and the claims in this chapter come from these sources:

- Alexander Waibel, Toshiyuki Hanazawa, Geoffrey Hinton, Kiyohiro Shikano, and Kevin Lang, "Phoneme recognition using time-delay neural networks," IEEE Transactions on Acoustics, Speech, and Signal Processing, 37(3), 328 to 339, 1989.
- [A time delay neural network architecture for efficient modeling of long temporal contexts](https://www.isca-archive.org/interspeech_2015/peddinti15b_interspeech.html), Vijayaditya Peddinti, Daniel Povey, and Sanjeev Khudanpur, Interspeech, 2015.
- Sepp Hochreiter and Jürgen Schmidhuber, "Long short-term memory," Neural Computation, 9(8), 1735 to 1780, 1997.
- [Long short-term memory recurrent neural network architectures for large scale acoustic modeling](https://www.isca-archive.org/interspeech_2014/sak14_interspeech.html), Haşim Sak, Andrew Senior, and Françoise Beaufays, Interspeech, 2014.
- [Speech Recognition and Understanding (11-751)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, 2022. The 24 October lecture is "Deep learning for speech recognition," and the 31 October lecture is "Advanced neural network architectures for acoustic model."
