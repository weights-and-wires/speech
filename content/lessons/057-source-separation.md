---
title: Separating voices
slug: 057-source-separation
part: "Part 8 — Beyond recognition and synthesis"
summary: Train a network to split a mixture into voices with permutation-invariant training, using a small Conv-TasNet on two synthetic sources.
minutes: 45
gpu: false
objectives:
  - Explain why a separator trained with a fixed output order fails, and how permutation-invariant training removes the problem.
  - Compute the permutation-invariant loss from a matrix of pairwise SI-SDR losses.
  - Describe Conv-TasNet's encoder, dilated TCN blocks, masks, and decoder, and compute the TCN's receptive field.
prereqs:
  - 056-speech-enhancement
exercise: 057-source-separation
---

# Separating voices

Two people talk at once on one microphone, and you want each voice on its own track. The [previous chapter](/lessons/056-speech-enhancement) removed noise by measuring it while nobody talked and turning down the cells it filled. That trick fails here, for two reasons. The unwanted sound is a voice, so it has no quiet stretch to measure. And you want both sounds back, not one.

The goal is clear: a model that takes one mixture and returns two signals. A new puzzle comes with it. The model's two outputs have no natural order, so how do you tell it which output should be which voice? This chapter shows why the obvious answer fails, fixes it with one change to the loss, and builds a small network that learns the masks.

## What you already know

- A mask multiplies each time-frequency cell by a number between 0 and 1, which keeps or removes sound.
- SI-SDR, from [Speech enhancement](/lessons/056-speech-enhancement), scores an estimate against a reference in decibels, ignoring the estimate's gain.
- Because the exercise gives you `si_sdr` in full, you can use $-\text{SI-SDR}$ as a loss.
- A 1-D convolution with dilation $d$ skips $d - 1$ samples between its taps.

## Why a fixed order fails

The obvious approach: build a network with two outputs, and train output 0 toward voice A and output 1 toward voice B, as you would for any regression. This works when the two sources differ in a way the network can name, such as "output 0 is always the low voice". It fails when they don't, which is the usual case. In a training set with many speakers, a pair of voices shows up in both roles, and nothing in the mixture says which voice is "source 1".

Put numbers on it. Let the two voices $a$ and $b$ be uncorrelated with equal power, and suppose the training set shows the same mixture $a + b$ half the time with $(a, b)$ as the targets, and half the time with $(b, a)$. A network that can't tell the cases apart does best under a squared-error loss by sending the same thing to both outputs: the average of the two targets, $\hat s = 0.5(a + b)$. Score it against $a$. The projection coefficient is 0.5, so the matching part is $0.5\,a$ and what is left is $0.5\,b$. Both have the same energy, so SI-SDR is $10 \log_{10}(1) = 0$ dB. The mixture itself scores 0 dB against $a$, so the trained network has learned nothing.

The failure has nothing to do with how well the network *could* separate. It splits the voices fine and still scores badly because it put them in the other order. Figure 1 shows the scores for one pair of outputs.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="A two by two loss matrix with entries minus 3, minus 12, minus 10, minus 2, and the two permutation costs minus 2.5 for keeping the order and minus 11 for swapping">
  <text x="130" y="28" class="fig-small">reference 0</text><text x="230" y="28" class="fig-small">reference 1</text>
  <text x="30" y="75" class="fig-small">output 0</text><text x="30" y="135" class="fig-small">output 1</text>
  <rect x="120" y="40" width="100" height="80" class="fill-soft-a stroke-grid"/><rect x="220" y="40" width="100" height="80" class="fill-soft-b stroke-b"/>
  <rect x="120" y="120" width="100" height="60" class="fill-soft-b stroke-b"/><rect x="220" y="120" width="100" height="60" class="fill-soft-a stroke-grid"/>
  <text x="160" y="85" class="fig-bold">-3</text><text x="260" y="85" class="fig-bold">-12</text>
  <text x="155" y="155" class="fig-bold">-10</text><text x="260" y="155" class="fig-bold">-2</text>
  <text x="360" y="70">keep order: (-3 + -2) / 2 = -2.5</text>
  <text x="360" y="150" class="fig-bold">swap: (-12 + -10) / 2 = -11</text>
  <text x="360" y="172" class="fig-small">PIT keeps the lower cost, -11.</text>
</svg>
Figure 1. Each cell is the loss of one output against one reference, in negative dB. The orange cells are the swapped assignment, and their mean is far lower, so permutation-invariant training picks it.
```

## Permutation-invariant training

The fix: stop asking which output is which. *Permutation-invariant training* (PIT) scores every assignment of outputs to references and keeps the best one, so the network is judged on how well it separates and not on which track it used:

$$
\mathcal{L}_{\text{uPIT}} = \min_{\pi \in \mathcal{S}_C} \frac{1}{C} \sum_{c=1}^{C} -\text{SI-SDR}\left(\hat s_c,\ s_{\pi(c)}\right)
$$

Read it aloud: $C$ is the number of sources, $\hat s_c$ is output $c$, and $s_j$ is reference $j$. For each permutation $\pi$ of the references, average the negative SI-SDR of output $c$ against reference $\pi(c)$. The loss is the smallest of those averages. The "u" is for *utterance-level*: one permutation covers the whole clip, so a voice doesn't hop between tracks halfway through.

Work Figure 1's numbers. Let $L_{ij}$ be the loss of output $i$ against reference $j$, and take $L = \begin{bmatrix} -3 & -12 \\ -10 & -2 \end{bmatrix}$. Keeping the order uses the diagonal, $(-3 + -2)/2 = -2.5$. Swapping uses the other diagonal, $(-12 + -10)/2 = -11$. PIT keeps $-11$, because output 0 matches reference 1 at 12 dB and output 1 matches reference 0 at 10 dB. A fixed-order loss would have reported $-2.5$ and pushed the network the wrong way.

Compute the $C \times C$ matrix once, then score each permutation by indexing it. In torch, broadcasting builds the matrix and `itertools.permutations` lists the orders:

```python
pair = -si_sdr(est[:, :, None, :], ref[:, None, :, :])  # (B, C, C): output i against reference j
cost = pair[:, torch.arange(C), perms].mean(-1)          # (B, number of permutations)
loss = cost.min(dim=1).values.mean()
```

There are $C!$ permutations, which is cheap for 2 or 3 sources and grows fast after that. The demo separates a chirp from a vowel with an oracle mask and scores both orders.

```demo
{"id": "separation-pit", "caption": "Press Shuffle the outputs and watch the fixed-order loss jump while the PIT loss stays put, then lower the chirp level to see where the oracle masks start to fail."}
```

> [!KEY]
> A separator's outputs have no natural order. PIT takes the minimum loss over all orders, so the network learns to separate and doesn't waste capacity guessing which track is which.

## Conv-TasNet: learn the mask

Now you need a network that outputs the voices. Luo and Mesgarani's Conv-TasNet (2019) drops the STFT and learns its own filterbank, so the masks act on cells it designed itself. Figure 2 shows the three parts.

```figure
<svg viewBox="0 0 620 200" role="img" aria-label="Conv-TasNet: a waveform goes into an encoder giving w, a TCN separator gives two masks, each mask multiplies w, and a decoder turns each product into one output waveform">
  <defs>
    <marker id="c057-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker>
  </defs>
  <rect x="10" y="80" width="70" height="40" class="fill-soft-a stroke-a"/><text x="22" y="105" class="fig-small">mixture x</text>
  <rect x="115" y="80" width="90" height="40" class="fill-soft-c stroke-grid"/><text x="127" y="98" class="fig-small">encoder</text><text x="127" y="113" class="fig-small">conv + ReLU</text>
  <rect x="245" y="20" width="100" height="40" class="fill-soft-d stroke-grid"/><text x="255" y="38" class="fig-small">TCN separator</text><text x="255" y="53" class="fig-small">sigmoid masks</text>
  <rect x="385" y="80" width="60" height="40" class="fill-paper stroke-grid"/><text x="396" y="105" class="fig-small">w * m</text>
  <rect x="480" y="80" width="80" height="40" class="fill-soft-c stroke-grid"/><text x="490" y="98" class="fig-small">decoder</text><text x="490" y="113" class="fig-small">transp. conv</text>
  <rect x="580" y="50" width="34" height="30" class="fill-soft-b stroke-b"/><text x="588" y="70" class="fig-small">s1</text>
  <rect x="580" y="130" width="34" height="30" class="fill-soft-b stroke-b"/><text x="588" y="150" class="fig-small">s2</text>
  <line x1="80" y1="100" x2="112" y2="100" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="205" y1="100" x2="222" y2="100" class="stroke-muted"/><line x1="222" y1="100" x2="222" y2="40" class="stroke-muted"/><line x1="222" y1="40" x2="242" y2="40" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="222" y1="100" x2="382" y2="100" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="345" y1="40" x2="415" y2="40" class="stroke-muted"/><line x1="415" y1="40" x2="415" y2="77" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="445" y1="100" x2="477" y2="100" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="560" y1="100" x2="572" y2="100" class="stroke-muted"/><line x1="572" y1="100" x2="572" y2="65" class="stroke-muted"/><line x1="572" y1="65" x2="577" y2="65" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <line x1="572" y1="100" x2="572" y2="145" class="stroke-muted"/><line x1="572" y1="145" x2="577" y2="145" class="stroke-muted" marker-end="url(#c057-arrow)"/>
  <text x="115" y="150" class="fig-small">N = 64 filters of L = 16 samples,</text><text x="115" y="166" class="fig-small">stride 8</text>
  <text x="245" y="86" class="fig-small">w</text>
</svg>
Figure 2. The encoder turns the waveform into a filterbank representation w. The separator reads w and outputs one mask per source. Each mask multiplies w, and the decoder turns the product back into a waveform.
```

- **Encoder.** A 1-D convolution with $N$ filters of length $L$ and stride $L / 2$, then ReLU, turns the waveform into $w$, with $N$ channels and $T'$ frames.
- **Separator.** A temporal convolutional network (TCN) reads $w$ and outputs one sigmoid mask per source. A sigmoid keeps each mask between 0 and 1, the same range as the gains in the previous chapter.
- **Decoder.** A transposed convolution overlap-adds a learned basis: $\hat s_c = \text{Decoder}(w \odot m_c)$.

Each separator block is a 1x1 convolution, a *depthwise* convolution (it filters each channel on its own) with kernel $P$ and dilation $d$, and a 1x1 convolution, with PReLU and normalization between them, added to its input. The dilations double, $1, 2, 4, \dots, 2^{X - 1}$, and the stack repeats $R$ times. Doubling is the point: a voice is a pattern over many frames, and each block must see further than the last without adding cost per frame.

### How far the network sees

Each block widens the view by $(P - 1) d$ frames. The *receptive field*, the number of input frames that can affect one output frame, is therefore

$$
\text{RF} = 1 + R\,(P - 1) \sum_{i=0}^{X - 1} 2^i = 1 + R\,(P - 1)(2^X - 1)
$$

Read it aloud: one frame, plus $P - 1$ times the sum of the dilations, $2^X - 1$, for each of the $R$ repeats. Figure 3 draws the smallest case, $R = 1$, $X = 3$, and $P = 3$: blocks with dilations 1, 2, and 4 widen the view to 3, 7, and 15 frames, and the formula gives $1 + 2 \cdot 7 = 15$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Fifteen encoder frames, and the span of frames that reaches the center output frame after three blocks with dilations 1, 2, and 4: 3, 7, then 15 frames">
  <text x="70" y="28" class="fig-small">input frames</text>
  <rect x="70" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="100" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="130" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="160" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="190" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="220" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="250" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="280" y="34" width="30" height="22" class="fill-b"/>
  <rect x="310" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="340" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="370" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="400" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="430" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="460" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="490" y="34" width="30" height="22" class="fill-paper stroke-grid"/>
  <text x="70" y="84" class="fig-small">after block 1 (d = 1)</text>
  <rect x="70" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="100" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="130" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="160" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="190" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="220" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="250" y="90" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="280" y="90" width="30" height="22" class="fill-b"/>
  <rect x="310" y="90" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="340" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="370" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="400" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="430" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="460" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="490" y="90" width="30" height="22" class="fill-paper stroke-grid"/>
  <text x="526" y="106" class="fig-small">3</text>
  <text x="70" y="140" class="fig-small">after block 2 (d = 2)</text>
  <rect x="70" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="100" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="130" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="160" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="190" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="220" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="250" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="280" y="146" width="30" height="22" class="fill-b"/>
  <rect x="310" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="340" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="370" y="146" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="400" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="430" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="460" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <rect x="490" y="146" width="30" height="22" class="fill-paper stroke-grid"/>
  <text x="526" y="162" class="fig-small">7</text>
  <text x="70" y="196" class="fig-small">after block 3 (d = 4)</text>
  <rect x="70" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="100" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="130" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="160" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="190" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="220" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="250" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="280" y="202" width="30" height="22" class="fill-b"/>
  <rect x="310" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="340" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="370" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="400" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="430" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="460" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <rect x="490" y="202" width="30" height="22" class="fill-soft-b stroke-b"/>
  <text x="526" y="218" class="fig-small">15</text>
  <text x="70" y="244" class="fig-small">frame index 0 to 14; the dark cell is the center frame, whose output the blocks compute</text>
</svg>
Figure 3. Each doubling of the dilation adds twice as many frames as the block before it, so three blocks with kernel 3 see 15 frames. The view grows exponentially in the number of blocks, not linearly.
```

The paper's setting is $N = 512$ filters of $L = 16$ samples (2 ms at 8 kHz), $R = 3$, $X = 8$, and $P = 3$, which gives $1 + 3 \cdot 2 \cdot 255 = 1531$ frames, about 1.5 seconds of context at a hop of 8 samples.

The exercise's model is smaller: $N = 64$, $L = 16$, $R = 2$, $X = 5$, and $P = 3$, with 55,541 parameters. Its receptive field is $1 + 2 \cdot 2 \cdot 31 = 125$ frames, or $124 \times 8 + 16 = 1008$ samples, 126 ms at 8 kHz. The exercise normalizes each frame over its channels, so a change at one frame reaches only the receptive field, which a check measures. The encoder covers only whole frames, so pad the input on the right with `F.pad` and trim the decoder's output back to the input length.

The Run trains this model with PIT for 300 steps, about 20 seconds on a laptop CPU, and held-out mixtures go from 0 dB SI-SDR to about 14 dB.

> [!RECAP]
> - A fixed output order fails because two voices have no natural order, and the best fixed-order guess is the average of both, which scores 0 dB.
> - PIT takes the minimum of $-\text{SI-SDR}$ over every assignment of outputs to references.
> - Conv-TasNet encodes with a learned 1-D convolution, predicts one sigmoid mask per source, and decodes $w \odot m_c$.
> - The TCN's dilations double, so its receptive field is $1 + R(P - 1)(2^X - 1)$ frames.

## Build it step by step

The exercise gives you `si_sdr`. Build the rest in the order the checks test them.

1. **`pit_loss(est, ref)`** takes `(B, C, T)` tensors and returns the loss and a `(B, C)` permutation. Build the pairwise matrix by broadcasting, index it with every row of `itertools.permutations`, and take `argmin`. The common mistake is to take one minimum over the whole batch; take the minimum per example, then average.
2. **`tcn_receptive_field(repeats, blocks, kernel)`** returns the formula's integer. The common mistake is to forget the leading 1.
3. **`TCNBlock`** uses `groups=hidden` and `padding=dilation * (kernel - 1) // 2`, which keeps the length for an odd kernel. The common mistake is to drop the residual connection, so the block no longer returns input plus output.
4. **`TinyConvTasNet.forward`** pads, encodes, masks with a sigmoid, decodes all sources in one batched call, and trims to `T`. The common mistake is to forget the trim, so the output is a few samples longer than the input.

## Where this breaks

- **Source counts.** PIT checks $C!$ orders, and a model has a fixed number of outputs, so it can't separate an unknown number of speakers without extra machinery.
- **Unseen sources.** The model learns the kinds of sources it trained on. A separator trained on chirps and vowels has learned nothing about two talkers.
- **Long clips.** A receptive field of 1.5 seconds sees one stretch at a time. Streaming a long recording means cutting it in chunks, and the same voice can land on different tracks in different chunks.

## The exercise

You build `pit_loss`, `tcn_receptive_field`, `TCNBlock`, and `TinyConvTasNet.forward`. Press **Run** to train a separator that pulls a chirp out of a vowel train, and to hear each output.

The checks confirm that PIT keeps or swaps the order correctly for two sources, matches a brute-force minimum for three sources, and gives a finite gradient. They also check that your formula gives 1531 for the paper's setting and that a stack of your blocks sees exactly that many frames, that the model returns the input's length for any input, and that sixty PIT steps raise its SI-SDR by at least 10 dB.

> [!KEY]
> A separator has to decide per cell which voice owns it, and its outputs have no natural order. PIT scores every order and keeps the best, so a network can learn to separate without being told which track is which. Conv-TasNet learns the filterbank and the masks together, and its dilated blocks give it the context to tell two voices apart. That answers the opening question: you don't tell the model which output is which.
