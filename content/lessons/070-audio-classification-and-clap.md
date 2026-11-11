---
title: Audio classification and CLAP
slug: 070-audio-classification-and-clap
part: "Part 8 — Beyond recognition and synthesis"
summary: Cut spectrograms into patches for AST, tag clips with sigmoid losses and score them by mAP, and train CLAP so any text prompt becomes a class.
minutes: 55
gpu: false
objectives:
  - Count and cut the overlapping spectrogram patches of the Audio Spectrogram Transformer.
  - Train a multi-label tagger with one sigmoid per class and a numerically stable binary cross-entropy, and say when softmax is the right loss instead.
  - Compute average precision and mean average precision, and explain why averaging over only the positive ranks is correct.
  - Write CLAP's symmetric contrastive loss, and classify sounds zero-shot from text prompts by cosine similarity.
prereqs:
  - 012-hearing-and-the-mel-scale
  - 013-log-mel-spectrograms
exercise: 070-audio-classification-and-clap
---

# Audio classification and CLAP

Not every question about audio is "what was said". A baby monitor asks whether that was crying. A city sensor asks whether a siren, a dog, or a drill is making the noise. These are *classification* problems: a label for a whole clip, not one per frame, so you first turn the clip into vectors that a classifier can read.

Real sound is messier than one label per clip. A street recording has speech, traffic, and a barking dog at once, so the classifier must say "yes" to several classes, and you need a score that works for that. A fixed list of labels has another limit: nobody trained it to say "squeaky door". If you could describe the class in words, you would not need to retrain.

The idea in this chapter is to turn audio into vectors and compare them. You cut a spectrogram into patches for a Transformer, train it to tag many classes at once, and score it by how well it ranks each class. Then you train audio and text vectors into one space, so a sentence becomes a class.

## What you already know

- A log-mel spectrogram has one row per mel band and one column per 10 ms frame.
- The sigmoid $\sigma(x) = 1 / (1 + e^{-x})$ turns a logit into a probability between 0 and 1.
- Cross-entropy over a softmax is the loss for picking one class out of many.
- Cosine similarity compares directions, not lengths: $x \cdot r / (\lVert x \rVert \lVert r \rVert)$.

## AST: a spectrogram as an image

A Transformer can't take a 1,024-frame, 128-band spectrogram as one token per cell, because that is 131,072 tokens. The Audio Spectrogram Transformer (AST; Gong et al., 2021) treats the log-mel spectrogram as a picture and cuts it into square *patches*, the way a Vision Transformer does. Each patch is flattened and projected to one 768-dimensional token, and a learned `[CLS]` token in front feeds the classifier.

AST cuts 16 by 16 patches every 10 bins and frames, so neighbors overlap by 6. Overlap costs tokens and buys detail, because a sound on a patch edge still sits whole inside the next patch. Figure 1 shows one axis of a 46-frame clip.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Four overlapping 16-frame patches at starts 0, 10, 20, and 30 covering a 46-frame axis">
  <line x1="40" y1="150" x2="500" y2="150" class="stroke-muted"/>
  <rect x="40" y="20" width="160" height="22" class="fill-soft-a stroke-a"/>
  <rect x="140" y="50" width="160" height="22" class="fill-soft-b stroke-b"/>
  <rect x="240" y="80" width="160" height="22" class="fill-soft-a stroke-a"/>
  <rect x="340" y="110" width="160" height="22" class="fill-soft-b stroke-b"/>
  <text x="46" y="36" class="fig-small">patch 0: frames 0-15</text>
  <text x="146" y="66" class="fig-small">patch 1: frames 10-25</text>
  <text x="246" y="96" class="fig-small">patch 2: frames 20-35</text>
  <text x="346" y="126" class="fig-small">patch 3: frames 30-45</text>
  <text x="34" y="168" class="fig-small">0</text><text x="134" y="168" class="fig-small">10</text><text x="234" y="168" class="fig-small">20</text><text x="334" y="168" class="fig-small">30</text><text x="434" y="168" class="fig-small">40</text>
  <text x="200" y="186">time (frames, 10 ms each)</text>
</svg>
Figure 1. At stride 10, 46 frames give four patches. Patches that did not overlap would give only two, and would drop the last 14 frames.
```

The patch count along each axis is the output size of a strided convolution:

$$
N_f = \left\lfloor \frac{F - 16}{10} \right\rfloor + 1, \qquad
N_t = \left\lfloor \frac{T - 16}{10} \right\rfloor + 1
$$

Read it aloud: slide a 16-wide window in steps of 10 and count the positions that fit. $F$ is the number of mel bands and $T$ the number of frames. For AudioSet's 1,024-frame input, $N_f = \lfloor 112 / 10 \rfloor + 1 = 12$ and $N_t = \lfloor 1008 / 10 \rfloor + 1 = 101$, so the Transformer reads $12 \times 101 = 1{,}212$ tokens. Patch $(i_f, i_t)$ covers bands $10 i_f$ to $10 i_f + 15$ and frames $10 i_t$ to $10 i_t + 15$, so build all the indices with broadcasting and index once:

```python
rows = 10 * np.arange(n_f)[:, None, None, None] + np.arange(16)[None, None, :, None]
cols = 10 * np.arange(n_t)[None, :, None, None] + np.arange(16)[None, None, None, :]
patches = spec[rows, cols].reshape(n_f * n_t, 256)
```

A `Conv2d` with a 16 by 16 kernel and stride 10 computes this patchify and the projection in one step. AST starts from an ImageNet Vision Transformer and reports 0.485 mAP on AudioSet with an ensemble, 95.6% on ESC-50, and 98.1% on Speech Commands v2.

> [!KEY]
> AST is a Vision Transformer that reads overlapping 16 by 16 spectrogram patches. The overlap costs tokens, 1,212 for 10 seconds, and buys detail that non-overlapping patches would cut in half.

## Tagging: many labels at once

AudioSet, the dataset AST reports on, holds about 2 million ten-second YouTube clips and 527 sound classes. A clip carries every label that applies, so a street recording can be `Speech`, `Dog`, and `Vehicle` together. That makes the task *multi-label*, which changes the loss.

A softmax forces the class probabilities to sum to 1, so the classes compete: raising `Dog` lowers `Speech`. That is right when exactly one label is true, as in ESC-50. For tagging, give each class its own sigmoid, and score each one with *binary cross-entropy* (BCE) as if it were a separate yes-or-no question. With logit $x_c$ and target $y_c$ (1 if the label applies, else 0) for class $c$:

$$
\mathcal{L} = \frac{1}{N C} \sum_{i=1}^{N} \sum_{c=1}^{C} -\left[ y_{ic} \log \sigma(x_{ic}) + (1 - y_{ic}) \log \left(1 - \sigma(x_{ic})\right) \right]
$$

Read it aloud: for each clip and class, penalize the log of the probability given to the true answer, then average. Take one clip with logits $(2, 1, -3)$ and targets $(1, 1, 0)$. The sigmoids are $(0.881, 0.731, 0.047)$, so the per-class losses are $(0.127, 0.313, 0.049)$ and the mean is 0.163. A softmax over the same logits gives $(0.728, 0.268, 0.005)$. It can't put high probability on both true classes, because the three numbers share one unit.

Computing $\log \sigma(x)$ directly fails for large logits. In float64, $\sigma(40)$ rounds to exactly 1, so $\log(1 - \sigma(40))$ is $\log 0$, which is $-\infty$, even though the true loss is a plain 40. The fix is to rewrite the loss without ever forming $\sigma$ of a large number:

```python
loss = (x.clamp(min=0) - x * y + torch.log1p(torch.exp(-x.abs()))).mean()
```

This is the same value, because $\log(1 + e^{x}) = \max(x, 0) + \log(1 + e^{-|x|})$. The exponent is never positive, so nothing overflows. The gradient is the prediction minus the target: $(\sigma(x_{ic}) - y_{ic}) / (N C)$.

### Average precision and mAP

Accuracy is a poor score for tagging, because a model that always says "no" to a rare class is nearly always right and finds nothing. AudioSet uses *mean average precision* (mAP) instead, which asks how well the model *ranks* the clips of each class.

For one class, sort the clips by descending score. At each rank $r$, the *precision at rank* $P(r)$ is the fraction of the top $r$ clips that truly have the class. *Average precision* (AP) averages $P(r)$ over only the ranks where a true positive sits:

$$
\mathrm{AP} = \frac{1}{n_+} \sum_{r \,:\, y_{(r)} = 1} P(r)
$$

Here $n_+$ counts the positive clips and $y_{(r)}$ is the label at rank $r$. Read it aloud: each time you reach a dog clip going down the list, note how clean the list was so far, and average those notes. Take six clips whose labels, in rank order, are 1, 0, 1, 1, 0, 0. The precisions at the three positive ranks are $1/1$, $2/3$, and $3/4$, so AP is $(1 + 0.667 + 0.75)/3 = 0.806$ (Figure 3).

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Precision at each of six ranks; bars at the three positive ranks are highlighted and average to 0.806">
  <line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>
  <line x1="60" y1="40" x2="60" y2="190" class="stroke-muted"/>
  <text x="14" y="120" class="fig-small" transform="rotate(-90 14 120)" text-anchor="middle">precision at rank</text>
  <rect x="85" y="50.0" width="50" height="140.0" class="fill-soft-b stroke-b"/>
  <text x="110" y="44.0" text-anchor="middle" class="fig-small fig-bold">1.00</text>
  <text x="110" y="208" text-anchor="middle" class="fig-small">1</text>
  <text x="110" y="224" text-anchor="middle" class="fig-small">dog</text>
  <rect x="165" y="120.0" width="50" height="70.0" class="fill-sunk stroke-grid"/>
  <text x="190" y="114.0" text-anchor="middle" class="fig-small">0.50</text>
  <text x="190" y="208" text-anchor="middle" class="fig-small">2</text>
  <text x="190" y="224" text-anchor="middle" class="fig-small">other</text>
  <rect x="245" y="96.7" width="50" height="93.3" class="fill-soft-b stroke-b"/>
  <text x="270" y="90.7" text-anchor="middle" class="fig-small fig-bold">0.67</text>
  <text x="270" y="208" text-anchor="middle" class="fig-small">3</text>
  <text x="270" y="224" text-anchor="middle" class="fig-small">dog</text>
  <rect x="325" y="85.0" width="50" height="105.0" class="fill-soft-b stroke-b"/>
  <text x="350" y="79.0" text-anchor="middle" class="fig-small fig-bold">0.75</text>
  <text x="350" y="208" text-anchor="middle" class="fig-small">4</text>
  <text x="350" y="224" text-anchor="middle" class="fig-small">dog</text>
  <rect x="405" y="106.0" width="50" height="84.0" class="fill-sunk stroke-grid"/>
  <text x="430" y="100.0" text-anchor="middle" class="fig-small">0.60</text>
  <text x="430" y="208" text-anchor="middle" class="fig-small">5</text>
  <text x="430" y="224" text-anchor="middle" class="fig-small">other</text>
  <rect x="485" y="120.0" width="50" height="70.0" class="fill-sunk stroke-grid"/>
  <text x="510" y="114.0" text-anchor="middle" class="fig-small">0.50</text>
  <text x="510" y="208" text-anchor="middle" class="fig-small">6</text>
  <text x="510" y="224" text-anchor="middle" class="fig-small">other</text>
  <text x="310" y="244" text-anchor="middle" class="fig-small">rank, by descending score</text>
  <text x="310" y="262" text-anchor="middle" class="fig-small fig-bold">AP = (1.00 + 0.67 + 0.75) / 3 = 0.806</text>
</svg>
Figure 3. Precision at rank 1, 3, and 4 (orange) is the precision right after each dog clip, and AP averages only those three. Averaging all six ranks would give 0.669 and reward the 0.50 and 0.60 that follow clips with no dog.
```

The common mistake is to average $P(r)$ over all six ranks, which gives 0.669. The ranks with no dog would drag even a perfect ranking below 1, but AP of a ranking with every positive on top is exactly 1.

```python
order = np.argsort(-scores, kind="stable")
y = labels[order]
precision = np.cumsum(y) / np.arange(1, len(y) + 1)
ap = precision[y == 1].mean()
```

*Mean* average precision averages AP over classes. A class with no positive clip in the evaluation set has an undefined AP, so skip it rather than score it as 0. Chance is not 0 either: random scores give an AP equal to the class's prevalence, the fraction of positives. In the Run, a noisy toy tagger over four classes reaches AP 0.863, 0.703, 0.434, and 0.274 against chance levels 0.532, 0.225, 0.052, and 0.022, for an mAP of 0.569.

> [!KEY]
> Tagging uses one sigmoid and one binary cross-entropy per class, so labels don't compete. It is scored by mAP, which ranks the clips of each class and averages precision at the positive ranks only.

## CLAP: audio and text in one space

A classifier with 527 fixed labels can't answer "is this a squeaky door?". CLAP (contrastive language-audio pretraining) trains an audio encoder and a text encoder together, so a clip and its caption land near each other. Then any sentence is a class.

Take a batch of $N$ clip-caption pairs, embed them, and normalize each embedding to unit length, giving $a_i$ and $t_i$. Score every clip against every caption:

$$
S_{ij} = \frac{a_i^\top t_j}{\tau}, \qquad
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left[ -\log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} - \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}} \right]
$$

Read it aloud: $S_{ij}$ is the cosine of clip $i$ and caption $j$ divided by a temperature $\tau$. The first term asks each clip to pick its own caption out of the batch, and the second asks each caption to pick its own clip. Averaging the two cross-entropies makes the loss *symmetric*. A small $\tau$ sharpens the softmax.

```python
a, t = F.normalize(audio_emb, dim=-1), F.normalize(text_emb, dim=-1)
logits = a @ t.T / tau
loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
```

### Work the loss by hand

Take $N = 2$, clips $a_1 = (1, 0)$ and $a_2 = (0, 1)$, captions $t_1 = (0.8, 0.6)$ and $t_2 = (0, 1)$, and $\tau = 0.1$. The cosines are 0.8, 0, 0.6, and 1, so $S = \begin{bmatrix} 8 & 0 \\ 6 & 10 \end{bmatrix}$. Figure 2 shows the four terms.

```figure
<svg viewBox="0 0 600 210" role="img" aria-label="The score matrix 8, 0, 6, 10 with row terms 0.00034 and 0.01815 and column terms 0.12693 and 0.00005">
  <text x="130" y="28" class="fig-small">caption 1</text><text x="230" y="28" class="fig-small">caption 2</text>
  <text x="50" y="75" class="fig-small">clip 1</text><text x="50" y="135" class="fig-small">clip 2</text>
  <rect x="120" y="40" width="100" height="80" class="fill-soft-a stroke-a"/><rect x="220" y="40" width="100" height="80" class="fill-paper stroke-grid"/>
  <rect x="120" y="120" width="100" height="60" class="fill-paper stroke-grid"/><rect x="220" y="120" width="100" height="60" class="fill-soft-a stroke-a"/>
  <text x="160" y="85" class="fig-bold">8</text><text x="266" y="85">0</text><text x="166" y="155">6</text><text x="258" y="155" class="fig-bold">10</text>
  <text x="340" y="70" class="fig-small">row 1: log(1 + e^-8) = 0.00034</text>
  <text x="340" y="90" class="fig-small">row 2: log(1 + e^-4) = 0.01815</text>
  <text x="340" y="130" class="fig-small fig-bold">col 1: log(1 + e^-2) = 0.12693</text>
  <text x="340" y="150" class="fig-small">col 2: log(1 + e^-10) = 0.00005</text>
  <text x="120" y="200" class="fig-small">loss = (0.00034 + 0.01815 + 0.12693 + 0.00005) / 4 = 0.0364</text>
</svg>
Figure 2. Column 1 dominates the loss: clip 2 scores 6 against caption 1, close to the 8 that clip 1 scores, so caption 1 can't tell its own clip apart.
```

When every embedding is the same, every entry of $S$ is equal and each term is $\log N$, which makes a quick test. The loss uses a softmax even though tagging does not, because here exactly one caption is right for each clip, and the batch supplies the $N - 1$ wrong ones.

## Zero-shot classification

Once audio and text share a space, a class is a sentence. To classify *zero-shot*, with no labeled examples of the target classes, embed one prompt per class, such as "the sound of a dog", normalize it, and compare the clip's embedding to every prompt:

$$
\hat{c} = \operatorname*{arg\,max}_c \; \frac{a^\top t_c}{\lVert a \rVert \, \lVert t_c \rVert}
$$

Here $a$ is the clip's embedding and $t_c$ is the embedding of the prompt for class $c$. Read it aloud: the predicted class is the prompt whose direction is closest to the clip's. The temperature is gone because it rescales every score equally and can't change the argmax.

The normalization matters, and forgetting it is the common bug. Take a clip $a = (1, 1.2)$ and three prompts: dog at $(4, 0)$, rain at $(0, 1)$, and siren at $(-1, -1)$. The raw dot products are 4.00, 1.20, and $-2.20$, so a missing normalization picks dog. The cosines are 0.640, 0.768, and $-0.996$, so the angle picks rain (Figure 4).

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Unit vectors for a clip and three prompts; the clip points closest to the rain prompt">
  <defs><marker id="c064-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-a"/></marker>
  <marker id="c064-arrow-b" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-b"/></marker></defs>
  <circle cx="160" cy="125" r="90" fill="none" class="stroke-grid"/>
  <line x1="55" y1="125" x2="265" y2="125" class="stroke-grid"/><line x1="160" y1="20" x2="160" y2="230" class="stroke-grid"/>
  <line x1="160" y1="125" x2="250.0" y2="125.0" class="stroke-b" marker-end="url(#c064-arrow-b)"/>
  <text x="258" y="119" class="fig-small" text-anchor="start">dog</text>
  <line x1="160" y1="125" x2="160.0" y2="35.0" class="stroke-b" marker-end="url(#c064-arrow-b)"/>
  <text x="168" y="29" class="fig-small" text-anchor="start">rain</text>
  <line x1="160" y1="125" x2="96.4" y2="188.6" class="stroke-b" marker-end="url(#c064-arrow-b)"/>
  <text x="88" y="203" class="fig-small" text-anchor="end">siren</text>
  <line x1="160" y1="125" x2="217.6" y2="55.9" class="stroke-a" stroke-width="2.5" marker-end="url(#c064-arrow)"/>
  <text x="226" y="52" class="fig-small fig-bold">clip</text>
  <text x="330" y="60" class="fig-small fig-bold">prompt</text><text x="470" y="60" class="fig-small fig-bold">dot</text><text x="530" y="60" class="fig-small fig-bold">cosine</text>
  <text x="330" y="85" class="fig-small">the sound of a dog</text>
  <text x="470" y="85" class="fig-small">4.00</text><text x="530" y="85" class="fig-small">0.640</text>
  <text x="330" y="115" class="fig-small fig-bold">the sound of rain</text>
  <text x="470" y="115" class="fig-small">1.20</text><text x="530" y="115" class="fig-small fig-bold">0.768</text>
  <text x="330" y="145" class="fig-small">the sound of a siren</text>
  <text x="470" y="145" class="fig-small">-2.20</text><text x="530" y="145" class="fig-small">-0.996</text>
  <text x="330" y="190" class="fig-small">Raw dot products pick the dog prompt,</text><text x="330" y="208" class="fig-small">because its vector is long. Cosines pick rain.</text>
</svg>
Figure 4. After normalization the clip points between rain and dog, closer to rain by angle. The long dog vector wins a raw dot product only because of its length.
```

```python
a = F.normalize(audio_emb, dim=-1)
t = F.normalize(class_text_emb, dim=-1)
predicted = (a @ t.T).argmax(dim=-1)
```

Prompt wording changes the embedding, so people average several templates per class. The CLAP score in [Few-step generation and audio metrics](/lessons/067-few-step-generation-and-audio-metrics) uses the same embeddings to score generated audio against its prompt.

> [!KEY]
> CLAP trains audio and text into one unit-length space with a symmetric softmax over the batch, so a sentence is a class. Zero-shot classification is an argmax over cosines, and it only works if both sides are normalized.

> [!RECAP]
> - AST's patch grid is $\lfloor (F - 16)/10 \rfloor + 1$ by $\lfloor (T - 16)/10 \rfloor + 1$: 12 by 101 for 1,024 frames.
> - Multi-label tagging uses a sigmoid and binary cross-entropy per class, in the stable form $\max(x, 0) - xy + \log(1 + e^{-|x|})$.
> - AP averages precision at the positive ranks only, and mAP averages AP over the classes that have positives.
> - CLAP averages the audio-to-text and text-to-audio cross-entropies of $a t^\top / \tau$ over normalized embeddings.
> - Zero-shot classification is the argmax of the cosine between a clip and one prompt per class.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ast_patch_grid(n_mels, n_frames, patch, stride)`** returns `(n_freq, n_time)` from the formula with integer division. The common mistake is rounding up, which counts a patch that runs off the edge.
2. **`patchify(spec, patch, stride)`** returns `(n_freq * n_time, patch * patch)`. Build 4-D row and column grids by broadcasting, index once, and reshape. Frequency is the outer loop.
3. **`multilabel_bce(logits, targets)`** takes `(N, C)` tensors and returns one scalar. Use the `clamp`, `log1p` form so a logit of $\pm 1000$ stays finite. The common mistake is `log(sigmoid(x))`, which returns infinity for a confident wrong logit.
4. **`average_precision(scores, labels)`** sorts the labels by descending score, divides the cumulative count of positives by the rank, and averages only where the label is 1. Return NaN when there are no positives. The common mistake is averaging over every rank.
5. **`mean_average_precision(scores, labels)`** calls `average_precision` on each column of the `(N, C)` arrays and averages with `np.nanmean`, so a class with no positives is skipped rather than counted as 0.
6. **`clap_loss(audio_emb, text_emb, tau)`** normalizes with `F.normalize`, forms `a @ t.T / tau`, and averages `F.cross_entropy` over rows and over the transpose. The common mistake is using one direction only.
7. **`zero_shot_classify(audio_emb, class_text_emb)`** normalizes both arguments, then returns the `argmax` of the cosine matrix along the class axis. The common mistake is normalizing only the clips.

## Where this breaks

- **Leaky evaluation splits.** ESC-50 clips cut from one recording share a fold, so a random split inflates accuracy.
- **Tied scores.** `average_precision` assumes distinct scores. When many clips tie, such as saturated sigmoids at 1.0, the order inside the tie is arbitrary, and so is the AP.
- **A rare class.** With 5 positives, one misranked clip moves AP by several points, so report the positive count next to each AP.
- **Zero-shot is only as clean as the pretraining data.** If an evaluation clip was in the training pairs, the score measures memory, not transfer.

## The exercise

You build `ast_patch_grid` and `patchify` for AST, `multilabel_bce`, `average_precision`, and `mean_average_precision` for tagging, and `clap_loss` and `zero_shot_classify` for CLAP. Press **Run** to print AST's patch counts, score a toy tagger, and classify synthetic clips zero-shot against five prompts.

The checks verify your patches against a stride-10 `Conv2d`, `multilabel_bce` against the formula with a finite-difference gradient and logits of 40 and 1000, AP on a ranking worked by hand, and mAP with a class that has no positives. They check your CLAP loss against the formula, its $\log N$ case, its symmetry, and its derivative in $\tau$. Last, they check that zero-shot predictions follow the cosine and not the dot product.

## Further reading

- Gong, Chung, and Glass, "AST: Audio Spectrogram Transformer," Interspeech 2021 (https://arxiv.org/abs/2104.01778).
- Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," ICASSP 2017. The dataset behind the 527-class tagging benchmark that AST reports mAP on (https://research.google/pubs/pub45857/).
- Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation," ICASSP 2023 (https://arxiv.org/abs/2211.06687).
- Elizalde et al., "CLAP: Learning Audio Concepts From Natural Language Supervision," ICASSP 2023 (https://arxiv.org/abs/2206.04769).
- Bochen Li, "Machine Learning for Audio Signals," ECE 272/472 Audio Signal Processing, University of Rochester. The lecture covers classifiers, evaluation measures, and sound classification (https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/lectures/Lecture_MachineLearningForAudio.pdf).
