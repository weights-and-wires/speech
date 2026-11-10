---
title: Conformers and SpecAugment
slug: 037-conformers-and-specaugment
part: "Part 4 — Neural speech recognition"
summary: Hide random stripes of the spectrogram, score frames by distance, and add a convolution beside attention, then train a Conformer-CTC on digits.
minutes: 80
gpu: true
objectives:
  - Mask random bands and spans of a log-mel spectrogram with SpecAugment, within the policy's limits.
  - Compute Transformer-XL relative positional attention, including the rel_shift trick, and check it against a brute-force loop.
  - Build a Conformer block from its half-step feed-forward, attention, and convolution modules, with padding masked correctly.
  - Train a Conformer-CTC on sequences of spoken digits and measure its digit error rate on a speaker it never heard.
prereqs:
  - 028-neural-acoustic-models
  - 030-ctc-forward-backward
  - 033-attention-encoder-decoders
exercise: 037-conformers-and-specaugment
---

# Conformers and SpecAugment

The losses of the last few chapters all sit on top of an encoder, and the encoder decides how good the recognizer can be. The **Conformer** was the encoder of choice for several years.

Self-attention sees the whole utterance at once, which is good for context. Convolution sees a few neighboring frames, which is good for local detail, such as the quick formant change that separates "b" from "d". A Conformer block does both, between two feed-forward layers. A big encoder also overfits a small dataset quickly, so you also build **SpecAugment**, which blanks random stripes of the spectrogram so the model cannot lean on any one cue.

## What you already know

- A log-mel spectrogram has 80 mel bands and 100 frames per second, and a strided convolutional front end cuts the frame rate to about 25 per second, from [Neural acoustic models](/lessons/028-neural-acoustic-models).
- CTC trains an encoder with no frame labels, from [Computing CTC efficiently](/lessons/030-ctc-forward-backward).
- Self-attention scores every pair of frames with a dot product, from [Attention encoder-decoders](/lessons/033-attention-encoder-decoders).
- A residual block adds its output to its input, and LayerNorm rescales each frame to zero mean and unit variance.

## Hide random stripes

A model trained on clean features leans on whatever is easiest, such as one band. Hide that cue at random and the model must learn several ways to hear the same thing. **SpecAugment** (Park et al., 2019) does this on the features, not the audio, so it costs almost nothing. A **frequency mask** hides a band of $f$ channels starting at $f_0$, in every frame. A **time mask** hides a span of $t$ frames starting at $t_0$, in every band. You draw each width uniformly from 0 to its limit, then draw a start that fits inside the spectrogram, with $\nu$ mel bands and $\tau$ frames:

$$
f \sim \mathcal{U}\{0, \dots, F\}, \; f_0 \sim \mathcal{U}\{0, \dots, \nu - f\}, \qquad
t \sim \mathcal{U}\{0, \dots, \min(T, \lfloor p\,\tau \rfloor)\}, \; t_0 \sim \mathcal{U}\{0, \dots, \tau - t\}
$$

Frequency masks are at most $F$ bands wide. Time masks are at most $T$ frames wide, and never wider than a fraction $p$ of the utterance. In torch:

```python
f = int(torch.randint(0, F + 1, (1,), generator=gen))
f0 = int(torch.randint(0, n_mels - f + 1, (1,), generator=gen))
out[f0 : f0 + f, :] = mask_value
```

Take 8 bands and 10 frames, with $F = 3$, $T = 4$, $p = 0.2$, and one mask of each kind. Suppose the draws are $f = 2$ and $f_0 = 5$: bands 5 and 6 go dark across all 10 frames. The time mask's limit is $\min(4, \lfloor 0.2 \times 10 \rfloor) = 2$ frames. Suppose it draws $t = 2$ and $t_0 = 3$: frames 3 and 4 go dark across all 8 bands. The masks cover $2 \times 10 + 8 \times 2 - 2 \times 2 = 32$ of the 80 cells, because the four cells where they cross are counted twice.

```figure
<svg viewBox="0 0 600 360" role="img" aria-label="An 8 band by 10 frame spectrogram with a frequency mask over bands 5 and 6 and a time mask over frames 3 and 4; 32 of 80 cells are hidden">
<rect x="70" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="274" width="32" height="32" class="fill-ink"/>
<rect x="206" y="274" width="32" height="32" class="fill-ink"/>
<rect x="240" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="274" width="32" height="32" class="fill-soft-a"/>
<rect x="70" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="240" width="32" height="32" class="fill-ink"/>
<rect x="206" y="240" width="32" height="32" class="fill-ink"/>
<rect x="240" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="240" width="32" height="32" class="fill-soft-a"/>
<rect x="70" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="206" width="32" height="32" class="fill-ink"/>
<rect x="206" y="206" width="32" height="32" class="fill-ink"/>
<rect x="240" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="206" width="32" height="32" class="fill-soft-a"/>
<rect x="70" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="172" width="32" height="32" class="fill-ink"/>
<rect x="206" y="172" width="32" height="32" class="fill-ink"/>
<rect x="240" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="172" width="32" height="32" class="fill-soft-a"/>
<rect x="70" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="138" width="32" height="32" class="fill-ink"/>
<rect x="206" y="138" width="32" height="32" class="fill-ink"/>
<rect x="240" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="138" width="32" height="32" class="fill-soft-a"/>
<rect x="70" y="104" width="32" height="32" class="fill-ink"/>
<rect x="104" y="104" width="32" height="32" class="fill-ink"/>
<rect x="138" y="104" width="32" height="32" class="fill-ink"/>
<rect x="172" y="104" width="32" height="32" class="fill-bad"/>
<rect x="206" y="104" width="32" height="32" class="fill-bad"/>
<rect x="240" y="104" width="32" height="32" class="fill-ink"/>
<rect x="274" y="104" width="32" height="32" class="fill-ink"/>
<rect x="308" y="104" width="32" height="32" class="fill-ink"/>
<rect x="342" y="104" width="32" height="32" class="fill-ink"/>
<rect x="376" y="104" width="32" height="32" class="fill-ink"/>
<rect x="70" y="70" width="32" height="32" class="fill-ink"/>
<rect x="104" y="70" width="32" height="32" class="fill-ink"/>
<rect x="138" y="70" width="32" height="32" class="fill-ink"/>
<rect x="172" y="70" width="32" height="32" class="fill-bad"/>
<rect x="206" y="70" width="32" height="32" class="fill-bad"/>
<rect x="240" y="70" width="32" height="32" class="fill-ink"/>
<rect x="274" y="70" width="32" height="32" class="fill-ink"/>
<rect x="308" y="70" width="32" height="32" class="fill-ink"/>
<rect x="342" y="70" width="32" height="32" class="fill-ink"/>
<rect x="376" y="70" width="32" height="32" class="fill-ink"/>
<rect x="70" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="104" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="138" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="172" y="36" width="32" height="32" class="fill-ink"/>
<rect x="206" y="36" width="32" height="32" class="fill-ink"/>
<rect x="240" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="274" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="308" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="342" y="36" width="32" height="32" class="fill-soft-a"/>
<rect x="376" y="36" width="32" height="32" class="fill-soft-a"/>
<text x="86.0" y="28" text-anchor="middle" class="fig-small">0</text>
<text x="120.0" y="28" text-anchor="middle" class="fig-small">1</text>
<text x="154.0" y="28" text-anchor="middle" class="fig-small">2</text>
<text x="188.0" y="28" text-anchor="middle" class="fig-small">3</text>
<text x="222.0" y="28" text-anchor="middle" class="fig-small">4</text>
<text x="256.0" y="28" text-anchor="middle" class="fig-small">5</text>
<text x="290.0" y="28" text-anchor="middle" class="fig-small">6</text>
<text x="324.0" y="28" text-anchor="middle" class="fig-small">7</text>
<text x="358.0" y="28" text-anchor="middle" class="fig-small">8</text>
<text x="392.0" y="28" text-anchor="middle" class="fig-small">9</text>
<text x="62" y="294.0" text-anchor="end" class="fig-small">0</text>
<text x="62" y="260.0" text-anchor="end" class="fig-small">1</text>
<text x="62" y="226.0" text-anchor="end" class="fig-small">2</text>
<text x="62" y="192.0" text-anchor="end" class="fig-small">3</text>
<text x="62" y="158.0" text-anchor="end" class="fig-small">4</text>
<text x="62" y="124.0" text-anchor="end" class="fig-small">5</text>
<text x="62" y="90.0" text-anchor="end" class="fig-small">6</text>
<text x="62" y="56.0" text-anchor="end" class="fig-small">7</text>
<text x="240.0" y="330" text-anchor="middle">frame t</text>
<text x="20" y="172.0" transform="rotate(-90 20 172.0)" text-anchor="middle">mel band</text>
<rect x="430" y="60" width="14" height="14" class="fill-ink"/><text x="452" y="72" class="fig-small">masked once</text>
<rect x="430" y="84" width="14" height="14" class="fill-bad"/><text x="452" y="96" class="fig-small">in both masks</text>
<rect x="430" y="108" width="14" height="14" class="fill-soft-a"/><text x="452" y="120" class="fig-small">visible</text>
<text x="430" y="160" class="fig-small">bands 5, 6: 2 x 10 = 20</text>
<text x="430" y="176" class="fig-small">frames 3, 4: 8 x 2 = 16</text>
<text x="430" y="192" class="fig-small">overlap: 4 cells</text>
<text x="430" y="212" class="fig-bold fig-small">20 + 16 - 4 = 32</text>
<text x="430" y="228" class="fig-small">of 80 cells hidden</text>
</svg>
Figure 1. One frequency mask (bands 5 and 6) and one time mask (frames 3 and 4) on an 8 by 10 spectrogram. Notice that the four cells where the masks cross are counted once, so 32 cells are hidden, not 36.
```

The paper's LibriSpeech basic policy uses $F = 27$, $T = 100$, $p = 1.0$, and one mask of each kind. A spoken digit lasts about 40 frames, so a 100-frame time mask would erase whole digits, and this chapter's run uses $F = 15$, $T = 20$, $p = 0.2$, and two masks of each kind.

> [!WARNING]
> Mask with the mean, which is 0 after you normalize each utterance. A 0 in an unnormalized log-mel spectrogram means a loud signal, not silence. Mask only real frames, never padding, and only during training.

## Score frames by distance

Self-attention has no sense of order: shuffle the frames and every output shuffles the same way. In speech what matters is how far apart two frames are, not where they sit in the file, because a phone sounds the same at second 1 as at second 9.

**Relative positional encoding** (Dai et al., 2019, from Transformer-XL) scores a query $i$ against a key $j$ with a term that depends only on the distance $r = i - j$. With $q_i$, $k_j$, and a projected encoding $p_r$ of the distance, all of length $d_k$ per head:

$$
s_{ij} = \frac{(q_i + u)^{\top} k_j + (q_i + v)^{\top} p_{i-j}}{\sqrt{d_k}}
$$

Read it aloud: the first term is ordinary content attention, plus a learned vector $u$ that says which keys are worth attending to anywhere. The second term matches the query against the distance, plus a learned vector $v$ that says which distances matter regardless of content. The distance encodings are sinusoids, with rows for $r = L - 1$ down to $-(L - 1)$, where $e_r[2i] = \sin(r \cdot 10000^{-2i/d})$ and $e_r[2i+1] = \cos(r \cdot 10000^{-2i/d})$, and a learned linear layer turns $e_r$ into $p_r$:

```python
r = torch.arange(L - 1, -L, -1)[:, None]                       # (2L - 1, 1)
pe[:, 0::2], pe[:, 1::2] = torch.sin(r * rate), torch.cos(r * rate)
```

For $d = 2$, $r = 1$ gives $(\sin 1, \cos 1) = (0.8415, 0.5403)$, and the row for $r = 0$ is always $(0, 1, 0, 1, \dots)$.

Computing $(q_i + v)^\top p_r$ for every query and distance is one matrix product of shape $(L, 2L - 1)$, with row $i$ holding the scores against distances $L - 1, \dots, -(L - 1)$. But attention wants an $(L, L)$ matrix indexed by key $j$. The distance for key $j$ is $i - j$, which sits in column $L - 1 - i + j$, so each row needs a slice shifted one column further. That is `rel_shift`: $\operatorname{out}[i, j] = x[i,\; L - 1 - i + j]$. For $L = 3$ on the matrix that holds 0 to 14 in order, the result is `[[2, 3, 4], [6, 7, 8], [10, 11, 12]]`.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A 3 by 5 matrix holding 0 to 14; rel_shift keeps columns 2 to 4 of row 0, 1 to 3 of row 1, and 0 to 2 of row 2">
<rect x="90" y="50" width="49" height="49" class="fill-sunk"/>
<text x="115.0" y="81.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="142" y="50" width="49" height="49" class="fill-sunk"/>
<text x="167.0" y="81.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="194" y="50" width="49" height="49" class="fill-b"/>
<text x="219.0" y="81.0" text-anchor="middle" class="fig-bold">2</text>
<rect x="246" y="50" width="49" height="49" class="fill-b"/>
<text x="271.0" y="81.0" text-anchor="middle" class="fig-bold">3</text>
<rect x="298" y="50" width="49" height="49" class="fill-b"/>
<text x="323.0" y="81.0" text-anchor="middle" class="fig-bold">4</text>
<rect x="90" y="102" width="49" height="49" class="fill-sunk"/>
<text x="115.0" y="133.0" text-anchor="middle" class="fig-bold">5</text>
<rect x="142" y="102" width="49" height="49" class="fill-b"/>
<text x="167.0" y="133.0" text-anchor="middle" class="fig-bold">6</text>
<rect x="194" y="102" width="49" height="49" class="fill-b"/>
<text x="219.0" y="133.0" text-anchor="middle" class="fig-bold">7</text>
<rect x="246" y="102" width="49" height="49" class="fill-b"/>
<text x="271.0" y="133.0" text-anchor="middle" class="fig-bold">8</text>
<rect x="298" y="102" width="49" height="49" class="fill-sunk"/>
<text x="323.0" y="133.0" text-anchor="middle" class="fig-bold">9</text>
<rect x="90" y="154" width="49" height="49" class="fill-b"/>
<text x="115.0" y="185.0" text-anchor="middle" class="fig-bold">10</text>
<rect x="142" y="154" width="49" height="49" class="fill-b"/>
<text x="167.0" y="185.0" text-anchor="middle" class="fig-bold">11</text>
<rect x="194" y="154" width="49" height="49" class="fill-b"/>
<text x="219.0" y="185.0" text-anchor="middle" class="fig-bold">12</text>
<rect x="246" y="154" width="49" height="49" class="fill-sunk"/>
<text x="271.0" y="185.0" text-anchor="middle" class="fig-bold">13</text>
<rect x="298" y="154" width="49" height="49" class="fill-sunk"/>
<text x="323.0" y="185.0" text-anchor="middle" class="fig-bold">14</text>
<text x="115.0" y="40" text-anchor="middle" class="fig-small">2</text>
<text x="167.0" y="40" text-anchor="middle" class="fig-small">1</text>
<text x="219.0" y="40" text-anchor="middle" class="fig-small">0</text>
<text x="271.0" y="40" text-anchor="middle" class="fig-small">-1</text>
<text x="323.0" y="40" text-anchor="middle" class="fig-small">-2</text>
<text x="220.0" y="20" text-anchor="middle" class="fig-small">distance r = i - j, from L - 1 = 2 down to -(L - 1) = -2</text>
<text x="80" y="81.0" text-anchor="end" class="fig-small">i = 0</text>
<text x="80" y="133.0" text-anchor="end" class="fig-small">i = 1</text>
<text x="80" y="185.0" text-anchor="end" class="fig-small">i = 2</text>
<text x="372" y="62" class="fig-small">out[i, j] = x[i, L-1-i+j]</text>
<text x="372" y="84" class="fig-small">row 0: 2 3 4</text>
<text x="372" y="102" class="fig-small">row 1: 6 7 8</text>
<text x="372" y="120" class="fig-small">row 2: 10 11 12</text>
</svg>
Figure 2. A 3 by 5 score matrix with distances across the columns. Orange cells are the three entries each query row keeps: column L - 1 - i + j for key j, so the kept window slides one column left per row. Notice that in row 0 the keys j = 0, 1, 2 sit at distances 0, -1, -2, which is why the window starts at column 2.
```

You can write this with `gather`, but implementations use a faster trick: pad one zero column on the left, reshape so the rows slide by one, drop a row, and reshape back.

```python
padded = F.pad(x, (1, 0)).reshape(*lead, 2 * L, L)          # rows slide by one
out = padded[..., 1:, :].reshape(*lead, L, 2 * L - 1)[..., :L]
```

> [!KEY]
> Relative attention adds a score for the distance $i - j$ to the content score, so the model learns "attend two frames back" instead of "attend to frame 7". The `rel_shift` reshape turns one $(L, 2L - 1)$ product into the $(L, L)$ scores you need.

## The Conformer block

A Conformer block (Gulati et al., 2020) sandwiches attention and convolution between two feed-forward modules, each of which adds only half its output, a half-step design from Macaron-Net. For block input $x$:

$$
\tilde{x} = x + \tfrac12 \operatorname{FFN}(x), \quad
x' = \tilde{x} + \operatorname{MHSA}(\tilde{x}), \quad
x'' = x' + \operatorname{Conv}(x'), \quad
y = \operatorname{LayerNorm}\big(x'' + \tfrac12 \operatorname{FFN}(x'')\big)
$$

Read it aloud: add half a feed-forward step, then attention, then convolution, then the other half of the feed-forward, each as a residual, and normalize at the end. Each module starts with its own LayerNorm, called **pre-norm**, so every residual branch sees a normalized input.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The Conformer block: half a feed-forward step, self-attention, convolution, half a feed-forward step, then a final LayerNorm, each added back to its input">
<defs><marker id="c033-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="24" y="110" width="100" height="64" class="fill-soft-c stroke-c"/>
<text x="74.0" y="132" text-anchor="middle" class="fig-bold fig-small">FFN</text>
<text x="74.0" y="148" text-anchor="middle" class="fig-small">x 1/2</text>
<text x="74.0" y="164" text-anchor="middle" class="fig-small">166,896 params</text>
<line x1="124" y1="142" x2="148" y2="142" class="stroke-ink" marker-end="url(#c033-arrow)"/>
<rect x="150" y="110" width="100" height="64" class="fill-soft-a stroke-a"/>
<text x="200.0" y="132" text-anchor="middle" class="fig-bold fig-small">Self-attention</text>
<text x="200.0" y="148" text-anchor="middle" class="fig-small">relative positions</text>
<text x="200.0" y="164" text-anchor="middle" class="fig-small">104,832 params</text>
<line x1="250" y1="142" x2="274" y2="142" class="stroke-ink" marker-end="url(#c033-arrow)"/>
<rect x="276" y="110" width="100" height="64" class="fill-soft-b stroke-b"/>
<text x="326.0" y="132" text-anchor="middle" class="fig-bold fig-small">Convolution</text>
<text x="326.0" y="148" text-anchor="middle" class="fig-small">depthwise conv</text>
<text x="326.0" y="164" text-anchor="middle" class="fig-small">67,824 params</text>
<line x1="376" y1="142" x2="400" y2="142" class="stroke-ink" marker-end="url(#c033-arrow)"/>
<rect x="402" y="110" width="100" height="64" class="fill-soft-c stroke-c"/>
<text x="452.0" y="132" text-anchor="middle" class="fig-bold fig-small">FFN</text>
<text x="452.0" y="148" text-anchor="middle" class="fig-small">x 1/2</text>
<text x="452.0" y="164" text-anchor="middle" class="fig-small">166,896 params</text>
<path d="M12 142 L12 86 L136 86 L136 142" fill="none" class="stroke-muted dashed"/>
<path d="M138 142 L138 86 L262 86 L262 142" fill="none" class="stroke-muted dashed"/>
<path d="M264 142 L264 86 L388 86 L388 142" fill="none" class="stroke-muted dashed"/>
<path d="M390 142 L390 86 L514 86 L514 142" fill="none" class="stroke-muted dashed"/>
<text x="24" y="76" class="fig-small">each module adds its output back (residual)</text>
<rect x="528" y="118" width="60" height="48" class="fill-soft-d stroke-d"/>
<text x="558" y="140" text-anchor="middle" class="fig-bold fig-small">Layer</text>
<text x="558" y="154" text-anchor="middle" class="fig-bold fig-small">Norm</text>
<line x1="502" y1="142" x2="526" y2="142" class="stroke-ink" marker-end="url(#c033-arrow)"/>
<line x1="8" y1="142" x2="22" y2="142" class="stroke-ink" marker-end="url(#c033-arrow)"/>
<text x="8" y="192" class="fig-small">x</text>
<text x="564" y="192" class="fig-small">y</text>
<text x="24" y="250" class="fig-small">Parameters per block at width 144: 2 x 166,896 + 104,832 + 67,824 + 288 = 506,736.</text>
<text x="24" y="270" class="fig-small">Global context comes from attention, local detail from the depthwise convolution.</text>
</svg>
Figure 3. One Conformer block at width 144. Every module is a pre-norm residual, and the two feed-forward modules each add half their output. Notice that the two feed-forward modules hold two thirds of the parameters.
```

The modules are the following:

- **Feed-forward:** LayerNorm, Linear from $d$ to $4d$, Swish, Dropout, Linear back to $d$, Dropout. Swish is $x \cdot \sigma(x)$, `F.silu` in torch.
- **Attention:** LayerNorm, multi-head self-attention with relative positions, Dropout.
- **Convolution:** LayerNorm, a pointwise convolution from $d$ to $2d$ channels, a gated linear unit (GLU) back to $d$, a depthwise convolution of kernel $k$, BatchNorm, Swish, a pointwise convolution, Dropout.

A **pointwise** convolution has kernel size 1: it mixes channels within each frame, like a linear layer. A **depthwise** convolution has one filter per channel, so it slides along time without mixing channels. The GLU splits its $2d$ inputs into halves $a$ and $b$ and returns $a \odot \sigma(b)$. Open implementations use an odd kernel, such as 31, because it pads symmetrically.

> [!KEY]
> A Conformer block is FFN/2, attention, convolution, FFN/2, then LayerNorm, each module a pre-norm residual. Attention supplies global context and the depthwise convolution supplies local detail.

### Mask the padding

A batch pads short utterances to the length of the longest one. Attention ignores the padding if you mask the keys. The depthwise convolution does not: a kernel of 15 reaches 7 frames past the end of an utterance and mixes whatever the padding holds into the last real frames. So set padded frames to 0 right before the depthwise convolution. BatchNorm is the other trap: in training mode it averages over padding too, so the checks run in evaluation mode, where a padded utterance's real frames must come out exactly as if it were alone.

## Train a Conformer-CTC on digits

The **Run** block is the experiment. The model is the given front end, your Conformer blocks, and a linear head over 11 symbols: the blank and the digits 0 to 9. The front end is two $3 \times 3$ convolutions with stride 2, which turn $n$ feature frames into $\lfloor(\lfloor(n - 1)/2\rfloor - 1)/2\rfloor$ encoder frames. One second of audio gives $n = 101$ frames, since the STFT centers its first window on sample 0, and the front end turns that into 24 encoder frames, 40 ms apart. The CTC input lengths must use this formula, or the loss reads padding as speech.

The data comes from the Free Spoken Digit Dataset, `fsdd`: 3,000 recordings of the digits 0 to 9 by six speakers. The run joins 2 to 5 random recordings with short gaps, so a training example is a string such as "4 0 7". It trains on five speakers and tests on the sixth, `theo`, whose voice it never hears. The score is the **digit error rate**: the edit distance between the recognized and the spoken digits, divided by the number of spoken digits.

On a GPU, the run trains 6 blocks of width 144, 3.6 million parameters, with SpecAugment, for 3,000 steps of 64 utterances. Without a GPU it shrinks to one block of width 96, a thinner front end, and no SpecAugment, for 150 steps of 16 utterances, and says so. The following table lists measured runs, tested on 60 or 300 utterances from the held-out speaker:

| Run | Model | Steps of 16 | SpecAugment | Time | Digit error rate |
|---|---|---|---|---|---|
| The CPU fallback | 1 block, width 96, 295,000 parameters | 150 | No | 25 s | 89.2% to 46.9% |
| A longer CPU run | the same | 1,000 | Yes | 7 min | 92.3% to 9.7% |
| The GPU model, a third of its schedule, on an Apple GPU | 6 blocks, width 144, 3.6M | 1,000 of 64 | Yes | 8 min | 103.4% to 10.5% |

Each error rate runs from before training to after it. Trained longer with SpecAugment, even the small model gets fewer than one digit in ten wrong for a voice it never heard.

> [!KEY]
> What you now know: an encoder that must beat a held-out speaker needs both global and local views, so a Conformer pairs attention, with relative distances, with a depthwise convolution. SpecAugment keeps it from memorizing the training voices.

> [!RECAP]
> - SpecAugment draws a width uniformly up to its limit, then a start that fits, and masks whole bands or whole frames on a copy.
> - Relative attention scores $(q_i + u)^\top k_j + (q_i + v)^\top p_{i-j}$, divided by $\sqrt{d_k}$, with rows of $p$ running from $r = L - 1$ to $-(L - 1)$.
> - `rel_shift` sets `out[i, j] = x[i, L - 1 - i + j]`.
> - The block is `x + ff1/2`, `+ mhsa(norm(x))`, `+ conv(x)`, `+ ff2/2`, then `norm_out`, with padding zeroed before the depthwise convolution.

## Build it step by step

The checks test the pieces in this order. Each module's `__init__` is given, so you write the `forward` methods.

1. **`spec_augment(features, ...)`.** Takes `(n_mels, frames)` and returns a masked copy. Clone first, draw with `torch.randint(low, high + 1, (1,), generator=generator)`, and assign slices. Common mistake: drawing the time limit from the padded length instead of `length`.
2. **`relative_position_encoding(length, d_model)`.** Returns `(2L - 1, d)`, with `sin` in even columns and `cos` in odd ones. Common mistake: rows in ascending order, which flips every distance.
3. **`rel_shift(x)`.** Returns `(..., L, L)`. Use the pad-and-reshape trick or `gather`. Common mistake: keeping the last $L$ columns instead of the first.
4. **`RelPositionMultiHeadAttention.forward(x, pos_emb, mask)`.** Returns `(B, T, d)`. Add `pos_bias_u` and `pos_bias_v` to `q` while it is `(B, T, H, d_k)`. Common mistake: projecting `pos_emb` with `linear_k`, which ties content and position together.
5. **`FeedForwardModule.forward` and `ConvolutionModule.forward`.** Follow the layer lists, and transpose to `(B, d, T)` for the convolutions. Common mistake: applying `F.glu` over time instead of channels.
6. **`ConformerBlock.forward(x, pos_emb, mask)`.** Four residuals and a final LayerNorm. Common mistake: putting `norm_out` inside the last residual.

## Where this breaks

**BatchNorm and streaming do not mix.** As [Streaming encoders](/lessons/036-streaming-encoders) showed, statistics over one chunk differ from those over a whole utterance, so streaming Conformers often use LayerNorm in the convolution module.

**SpecAugment can erase the label.** A time mask wider than a word removes the only evidence for it. Scale $T$ to your utterances.

## The exercise

You build `spec_augment`, `relative_position_encoding`, `rel_shift`, and the `forward` methods of `RelPositionMultiHeadAttention`, `FeedForwardModule`, `ConvolutionModule`, and `ConformerBlock`. Press **Run** to train the Conformer-CTC on spoken digits. It prints the digit error rate on the held-out speaker before and after training, plots the loss, and plays one test utterance with its transcript and features.

The checks run on the CPU with no data. They confirm that SpecAugment masks whole bands and frames within the policy's limits, respects the true length, and draws widths with a mean near $F/2$. They compare the encoding with its formula, `rel_shift` with `gather`, and your attention with a brute-force loop. They check both modules against their layer lists, and confirm that padding changes nothing, in the convolution module and in the whole block.
