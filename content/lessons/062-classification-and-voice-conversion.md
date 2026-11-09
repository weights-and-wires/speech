---
title: Audio classification and voice conversion
slug: 062-classification-and-voice-conversion
part: "Part 8 — Beyond recognition and synthesis"
summary: Cut spectrograms into patches, train audio and text into one space with CLAP's loss, score spoof detectors by EER, and convert voices by kNN matching.
minutes: 70
gpu: false
objectives:
  - Count and cut the overlapping spectrogram patches of the Audio Spectrogram Transformer.
  - Write CLAP's symmetric contrastive loss, and classify sounds zero-shot from text prompts.
  - Score a spoofing countermeasure by its equal error rate.
  - Convert a voice with kNN-VC, and explain why matching self-supervised features changes the speaker but keeps the words.
  - State the consent rules that apply before you convert anyone's voice.
prereqs:
  - 011-hearing-and-the-mel-scale
  - 036-self-supervised-speech
  - 038-speaker-verification
exercise: 062-classification-and-voice-conversion
---

# Audio classification and voice conversion

Not every question about audio is "what was said". A baby monitor asks whether that was crying. A bank asks whether the voice on the line is a person or a machine. These are *classification* problems: one label for a whole clip. A recognizer's per-frame output is the wrong shape for them, so you first turn the clip into vectors that a classifier can read.

The same vectors let you do something more unsettling. If a model separates *what* is said from *who* says it, you keep the first and swap the second. That is *voice conversion*, and it works well enough that detecting it has become a field of its own.

The idea in this chapter is one move made three ways: turn audio into vectors, then compare vectors by angle. You cut a spectrogram into patch vectors, train audio and text vectors into one space, and convert a voice by swapping each frame for its nearest neighbors in another speaker's frames.

## What you already know

- A log-mel spectrogram has one row per mel band and one column per 10 ms frame.
- Self-supervised models such as WavLM give one vector per 20 ms, and their middle layers carry phonetic content.
- The equal error rate (EER) is where false acceptances equal false rejections, from [Speaker embeddings and verification](/lessons/038-speaker-verification).
- Cosine similarity compares directions, not lengths: $x \cdot r / (\lVert x \rVert \lVert r \rVert)$.
- Cross-entropy over a softmax is the loss for picking one class out of many.

## AST: a spectrogram as an image

The obvious classifier feeds the whole spectrogram to a network. A Transformer can't take 1,024 frames by 128 bands as one sequence of cells, because that is 131,072 tokens. The Audio Spectrogram Transformer (AST; Gong et al., 2021) treats the log-mel spectrogram as a picture and cuts it into square *patches*, the way a Vision Transformer does. Each patch is flattened and projected to one 768-dimensional token, and a learned `[CLS]` token in front feeds the classifier.

AST cuts 16 by 16 patches every 10 bins and frames, so neighbors overlap by 6. Overlap costs tokens and buys detail, because a sound that falls on a patch edge still sits whole inside the next patch. Figure 1 shows one axis of a 46-frame clip.

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

When every embedding is the same, every entry of $S$ is equal and each term is $\log N$, which makes a quick test. For *zero-shot* classification, embed one prompt per class, such as "the sound of a dog barking", and pick the prompt with the highest cosine to the clip. The CLAP score in [Few-step generation and audio metrics](/lessons/059-few-step-generation-and-audio-metrics) uses the same embeddings to score generated audio.

## Detect the fakes: the equal error rate

The ASVspoof challenges score *countermeasures*, classifiers that label a recording bona fide or spoofed. A countermeasure outputs a score, and the EER from [Speaker embeddings and verification](/lessons/038-speaker-verification) measures it, with bona fide speech as the target trials. A false rejection is a bona fide score below the threshold, a false acceptance is a spoofed score at or above it, and the EER is where the two rates cross. The Run draws both curves for a toy detector whose scores are two unit Gaussians 2.5 apart. Theory puts that EER at $\Phi(-1.25) = 10.6\%$, and the 400 seeded samples per class measure 11.0%.

Detectors overfit to the attacks they trained on, and an unseen vocoder can break them. The complementary approach is *provenance*: mark generated audio when it is made, as AudioSeal (2024) does with an imperceptible watermark.

## Voice conversion by nearest neighbors

Voice conversion keeps what is said and changes who says it. Every method splits speech into *content*, the phones and their timing, and *speaker identity*, the timbre of one vocal tract. kNN-VC (Baas et al., 2023) needs no training for the conversion itself. It relies on what [Self-supervised speech](/lessons/036-self-supervised-speech) measured: the middle layers of WavLM encode phones nearly the same way for every speaker, while each frame still carries the speaker's timbre. So a target-speaker frame of the same phone is a near neighbor, and it already sounds like the target.

The paper extracts layer 6 of WavLM-Large, one 1,024-dimensional vector per 20 ms. It embeds all the target's reference audio, the *matching set*, and replaces each source frame $x_t$ with the average of its $k$ nearest matching frames:

$$
y_t = \frac{1}{k} \sum_{r \in \text{kNN}_k(x_t)} r, \qquad d(x, r) = 1 - \frac{x \cdot r}{\lVert x \rVert\, \lVert r \rVert}
$$

Read it aloud: find the $k$ reference frames with the smallest cosine distance to the source frame, and average them. The paper uses $k = 4$. Averaging smooths over the quirks of any single reference frame, and a HiFi-GAN vocoder turns the result into audio.

```python
q = source / np.linalg.norm(source, axis=1, keepdims=True)
r = matching / np.linalg.norm(matching, axis=1, keepdims=True)
nearest = np.argpartition(-(q @ r.T), k - 1, axis=1)[:, :k]
converted = matching[nearest].mean(axis=1)
```

### Work kNN-VC by hand

Take one source frame $x = (1, 0)$ and four matching frames $r_1 = (0.9, 0.1)$, $r_2 = (0.2, 1)$, $r_3 = (1, -0.2)$, and $r_4 = (-1, 0)$. The cosines are 0.994, 0.196, 0.981, and $-1$. With $k = 2$, the nearest are $r_1$ and $r_3$, and the converted frame is their mean, $(0.95, -0.05)$. With $k = 1$, it would be an exact copy of $r_1$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A source frame at (1, 0), four matching frames, and the converted frame at (0.95, -0.05), the mean of the two nearest by angle">
  <defs><marker id="c055-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-a"/></marker></defs>
  <line x1="210" y1="125" x2="500" y2="125" class="stroke-grid"/><line x1="300" y1="20" x2="300" y2="235" class="stroke-grid"/>
  <line x1="300" y1="125" x2="388" y2="125" class="stroke-a" marker-end="url(#c055-arrow)"/>
  <circle cx="381" cy="116" r="5" class="fill-b"/><circle cx="318" cy="35" r="5" class="fill-muted"/>
  <circle cx="390" cy="143" r="5" class="fill-b"/><circle cx="210" cy="125" r="5" class="fill-muted"/>
  <circle cx="385" cy="130" r="5" class="fill-c"/>
  <text x="400" y="104" class="fig-small">r1 (0.9, 0.1)</text><text x="330" y="40" class="fig-small">r2 (0.2, 1)</text>
  <text x="400" y="160" class="fig-small">r3 (1, -0.2)</text><text x="216" y="115" class="fig-small">r4 (-1, 0)</text>
  <text x="405" y="135" class="fig-small fig-bold">converted (0.95, -0.05)</text>
  <text x="330" y="150" class="fig-small">x (1, 0)</text>
</svg>
Figure 3. The source frame x points along the horizontal axis. The two nearest frames by angle, r1 and r3 (orange), average to the converted frame (green), which lies in the target's region of feature space. The far frames r2 and r4 play no part.
```

The paper reports that below about 30 seconds of reference audio, kNN-VC falls behind other methods, though even 5 seconds keeps moderate quality. The Run can't load WavLM or a vocoder on the CPU, so it keeps the method and swaps the parts: two synthetic voices, a 110 Hz source and a 210 Hz target with formants raised 18%, mel cepstra with each speaker's mean removed as the features, and Griffin-Lim from [Phase and Griffin-Lim](/lessons/008-phase-and-griffin-lim) as the vocoder. On its seeded voices, 94% of matched frames are the same vowel as the source frame, and the output's pitch is the target's 211 Hz.

> [!KEY]
> kNN-VC changes the speaker by replacing every source frame with the average of its nearest neighbors in the target's own speech. It works because content features match across speakers while every matched frame already carries the target's timbre.

## Consent and misuse

A voice identifies a person. The EU's GDPR treats biometric data used to identify someone as a special category, and Illinois's BIPA regulates voiceprints. A converted voice can impersonate someone to their bank, their family, or their employer. The rules for this course follow: convert only your own voice, with your own opt-in, or voices from public corpora whose licenses allow it, such as LibriSpeech and VCTK. Label converted audio as synthetic when you share it, and watermark it when you can.

> [!WARNING]
> "It's only a demo" is not consent. The pipeline that converts a synthetic voice in this exercise converts a real one, so the rules apply the moment the input is a recording of a person.

> [!RECAP]
> - AST's patch grid is $\lfloor (F - 16)/10 \rfloor + 1$ by $\lfloor (T - 16)/10 \rfloor + 1$: 12 by 101 for 1,024 frames.
> - CLAP averages the audio-to-text and text-to-audio cross-entropies of $a t^\top / \tau$ over normalized embeddings.
> - A countermeasure's EER is where false rejections of bona fide speech equal false acceptances of spoofs.
> - kNN-VC replaces each source frame with the mean of its $k = 4$ nearest target frames by cosine distance.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ast_patch_grid(n_mels, n_frames, patch, stride)`** returns `(n_freq, n_time)` from the formula with integer division. The common mistake is rounding up, which counts a patch that runs off the edge.
2. **`patchify(spec, patch, stride)`** returns `(n_freq * n_time, patch * patch)`. Build 4-D row and column grids by broadcasting, index once, and reshape. Frequency is the outer loop.
3. **`clap_loss(audio_emb, text_emb, tau)`** normalizes with `F.normalize`, forms `a @ t.T / tau`, and averages `F.cross_entropy` over rows and over the transpose. The common mistake is using one direction only.
4. **`zero_shot_classify`** returns the `argmax` of the cosine matrix along the class axis.
5. **`equal_error_rate(bonafide, spoof)`** returns `(eer, threshold)`. Use `np.searchsorted` on sorted scores to count errors at every threshold, find where the rejection rate first reaches the acceptance rate, and interpolate. The common mistake is swapping the classes, which reports 1 minus the EER.
6. **`cosine_knn(query, keys, k)`** returns `(T, k)` indices, nearest first. Normalize, multiply, take `np.argpartition` on the negated similarities, then sort those $k$.
7. **`knn_vc`** indexes the matching set with those indices and averages over axis 1.

## Where this breaks

- **Leaky evaluation splits.** ESC-50 clips cut from the same recording share a fold. A random split puts near-duplicates in train and test and inflates accuracy.
- **A matching set that is too small.** With a few seconds of target speech, some phones have no good neighbor, and kNN-VC copies the wrong one.
- **Countermeasures that don't generalize.** A detector with a low EER on known attacks can fail on a vocoder it never saw. Report results on unseen attacks.

## The exercise

You build `ast_patch_grid` and `patchify` for AST, `clap_loss` and `zero_shot_classify` for CLAP, `equal_error_rate` for a spoofing countermeasure, and `cosine_knn` and `knn_vc` for voice conversion. Press **Run** to print AST's patch counts, hear a synthetic source voice, the target voice, and the source converted into the target, and see a toy countermeasure's error curves with its EER.

The checks verify the patch grid, your patches against a stride-10 `Conv2d`, your CLAP loss against the formula, its $\log N$ case, its symmetry, and a finite-difference derivative in $\tau$. They check zero-shot predictions, the EER on crafted scores and on two Gaussians, and your neighbors against a full sort. Finally, kNN-VC on synthetic features must keep the source phone and move to the target speaker for at least 95% of frames.
