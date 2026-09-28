---
title: Audio classification and voice conversion
slug: 40-classification-and-voice-conversion
part: "Part 8 — Beyond recognition and synthesis"
summary: Classify sounds with AST patches and CLAP's contrastive loss, score spoof detectors by EER, and convert voices by kNN matching.
minutes: 70
gpu: false
objectives:
  - Count and cut the overlapping spectrogram patches of the Audio Spectrogram Transformer.
  - Write CLAP's symmetric contrastive loss, and classify sounds zero-shot from text prompts.
  - Score a spoofing countermeasure by its equal error rate.
  - Convert a voice with kNN-VC, and explain why matching self-supervised features changes the speaker but keeps the words.
  - State the consent rules that apply before you convert anyone's voice.
prereqs:
  - 07-hearing-and-the-mel-scale
  - 21-self-supervised-speech
  - 23-speaker-recognition-and-diarization
exercise: 40-classification-and-voice-conversion
---

# Audio classification and voice conversion

Not every question about audio is "what was said". A baby monitor asks
whether that was crying. A phone asks whether you said its wake word. A bank
asks whether the voice on the line is a person or a machine. These are
*classification* problems: one label, or a few, for a whole clip.

The same features that answer them also let you do something more unsettling.
If a model can separate *what* is said from *who* says it, you can keep the
first and swap the second. That is *voice conversion*, and it works well
enough that the detection problem in the previous paragraph has become a field
of its own.

This lesson covers both sides. You cut spectrograms into patches the way the
Audio Spectrogram Transformer (AST) does, train audio and text into one
embedding space with CLAP's loss, score a spoofing detector, and convert one
synthetic voice into another with kNN-VC.

## What you already know

- A log-mel spectrogram has one row per mel band and one column per 10 ms frame.
- Self-supervised models such as WavLM give one vector per 20 ms, and their middle layers carry phonetic content.
- The equal error rate is where false acceptances equal false rejections, from lesson 23, *Speaker recognition and diarization*.
- Cosine similarity compares directions, not lengths: $x \cdot r / (\lVert x \rVert \lVert r \rVert)$.
- Cross-entropy over a softmax is the loss for picking one class out of many.

## Classify a clip

The datasets define the task. ESC-50 has 2,000 environmental clips of 5
seconds in 50 classes, split into 5 folds, and results are 5-fold
cross-validation accuracy. Speech Commands v2 has 105,829 one-second
utterances of 35 words, with a standard 12-class keyword-spotting task.
AudioSet has about 2 million 10-second clips with 527 labels, several per
clip, so it reports *mean average precision* (mAP), the area under each
class's precision-recall curve averaged over classes.

Every modern classifier starts from a log-mel spectrogram. The question is
what reads it.

## AST: a spectrogram as an image

The Audio Spectrogram Transformer (Gong et al., 2021) treats the log-mel
spectrogram as a picture and feeds it to a Vision Transformer. Its input is
128 mel bands with a 25 ms window every 10 ms. It cuts 16 by 16 patches with
an overlap of 6 in both directions, so a stride of 10, flattens each patch,
and projects it to a 768-dimensional token. A learned `[CLS]` token goes in
front, and its output feeds the classifier.

The patch count along each axis is the output size of a strided convolution:

$$
N_f = \left\lfloor \frac{F - 16}{10} \right\rfloor + 1, \qquad
N_t = \left\lfloor \frac{T - 16}{10} \right\rfloor + 1
$$

Read it aloud: slide a 16-wide window in steps of 10 and count the positions
that fit. $F$ is the number of mel bands and $T$ the number of frames. For
AudioSet's 1,024-frame input, $N_f = \lfloor 112 / 10 \rfloor + 1 = 12$ and
$N_t = \lfloor 1008 / 10 \rfloor + 1 = 101$, so the Transformer reads
$12 \times 101 = 1{,}212$ patch tokens. An ESC-50 clip of 5 seconds, about
500 frames, gives $12 \times 49 = 588$.

Patch $(i_f, i_t)$ covers bands $10 i_f$ to $10 i_f + 15$ and frames
$10 i_t$ to $10 i_t + 15$. Build all the indices with broadcasting and index
once:

```python
rows = 10 * np.arange(n_f)[:, None, None, None] + np.arange(16)[None, None, :, None]
cols = 10 * np.arange(n_t)[None, :, None, None] + np.arange(16)[None, None, None, :]
patches = spec[rows, cols].reshape(n_f * n_t, 256)
```

A `Conv2d` with a 16 by 16 kernel and stride 10 computes exactly this
patchify followed by the linear projection, which is how implementations do
it. AST starts from an ImageNet Vision Transformer (DeiT), averages its
input weights over the three color channels, and interpolates its positional
embeddings to the 12 by 101 grid. It reports 0.485 mAP on AudioSet with an
ensemble, 95.6% on ESC-50, and 98.1% on Speech Commands v2.

> [!KEY]
> AST is a Vision Transformer that reads overlapping 16 by 16 spectrogram patches. The overlap costs tokens, 1,212 for 10 seconds, and buys detail that non-overlapping patches would cut in half.

## CLAP: audio and text in one space

A classifier with 527 fixed labels can't answer "is this a squeaky door?".
CLAP (contrastive language-audio pretraining) trains an audio encoder and a
text encoder together, so that a clip and its caption land near each other.
Any sentence then becomes a class.

Take a batch of $N$ clip-caption pairs, embed them, and normalize each
embedding to unit length, giving $a_i$ and $t_i$. Every clip is scored
against every caption:

$$
S_{ij} = \frac{a_i^\top t_j}{\tau}, \qquad
\mathcal{L} = \frac{1}{2N} \sum_{i=1}^{N} \left[ -\log \frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}} - \log \frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}} \right]
$$

Read it aloud: $S_{ij}$ is the cosine similarity of clip $i$ and caption $j$
divided by a temperature $\tau$. The first term asks each clip to pick its
own caption out of the batch, and the second asks each caption to pick its
own clip. The loss averages the two cross-entropies, which is why it is
called *symmetric*. A small $\tau$ sharpens the softmax. Implementations
learn $\tau$ and clamp it, because an unbounded temperature destabilizes
training.

```python
a, t = F.normalize(audio_emb, dim=-1), F.normalize(text_emb, dim=-1)
logits = a @ t.T / tau
loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
```

### Work the loss by hand

Take $N = 2$, clips $a_1 = (1, 0)$ and $a_2 = (0, 1)$, captions
$t_1 = (0.8, 0.6)$ and $t_2 = (0, 1)$, all unit length, and $\tau = 0.1$.
The cosines are 0.8, 0, 0.6, and 1, so

$$
S = \begin{bmatrix} 8 & 0 \\ 6 & 10 \end{bmatrix}
$$

- Row 1: $-\log \frac{e^8}{e^8 + e^0} = \log(1 + e^{-8}) = 0.00034$.
- Row 2: $\log(1 + e^{6 - 10}) = 0.01815$.
- Column 1: caption 1 compares 8 with 6, so $\log(1 + e^{-2}) = 0.12693$.
- Column 2: $\log(1 + e^{-10}) = 0.00005$.
- The loss is $(0.00034 + 0.01815 + 0.12693 + 0.00005) / 4 = 0.0364$.

Column 1 contributes most: clip 2 also looks a lot like caption 1. When every
embedding is the same, every entry of $S$ is equal, and each term is
$\log N$, which makes a quick test.

For *zero-shot* classification, embed one prompt per class, such as "the
sound of a dog barking", and pick the prompt with the highest cosine to the
clip. The CLAP score of lesson 38, *Few-step generation and audio metrics*,
uses the same embeddings to score generated audio.

## Voice conversion

Voice conversion keeps what is said and changes who says it. Every method
splits speech into *content*, the phones and their timing, and *speaker
identity*, the timbre of one vocal tract. AutoVC (2019) does it with a
bottleneck. Its content encoder is squeezed so narrow that speaker
information can't fit through, and a decoder takes the speaker from a
separate embedding. Too wide a bottleneck leaks the source speaker, and too
narrow loses words.

kNN-VC (Baas et al., 2023) needs no training for the conversion itself. It
relies on what lesson 21, *Self-supervised speech*, measured: the middle
layers of self-supervised models such as WavLM encode phones in a way that is nearly the same
for every speaker, while each frame still carries the speaker's timbre. So a
frame of the target speaker saying the same phone is a near neighbor, and it
already sounds like the target.

The paper extracts layer 6 of WavLM-Large, one 1,024-dimensional vector per
20 ms. It embeds all of the target speaker's reference audio, the *matching
set*, and replaces each source frame $x_t$ with the average of its $k$
nearest matching frames:

$$
y_t = \frac{1}{k} \sum_{r \in \text{kNN}_k(x_t)} r, \qquad d(x, r) = 1 - \frac{x \cdot r}{\lVert x \rVert\, \lVert r \rVert}
$$

Read it aloud: find the $k$ reference frames with the smallest cosine
distance to the source frame, and average them. The paper uses $k = 4$.
Averaging smooths over the quirks of any single reference frame. A HiFi-GAN
vocoder, trained to turn WavLM layer-6 features into audio, makes the result
audible. The *prematched* version of that vocoder is trained on features
that were themselves replaced by kNN matching, so it has already heard the
small artifacts matching creates.

```python
q = source / np.linalg.norm(source, axis=1, keepdims=True)
r = matching / np.linalg.norm(matching, axis=1, keepdims=True)
nearest = np.argpartition(-(q @ r.T), k - 1, axis=1)[:, :k]
converted = matching[nearest].mean(axis=1)
```

### Work kNN-VC by hand

Take one source frame $x = (1, 0)$ and four matching frames
$r_1 = (0.9, 0.1)$, $r_2 = (0.2, 1)$, $r_3 = (1, -0.2)$, and
$r_4 = (-1, 0)$. The cosine similarities are $0.9 / 0.906 = 0.994$,
$0.2 / 1.020 = 0.196$, $1 / 1.020 = 0.981$, and $-1$. With $k = 2$, the
nearest are $r_1$ and $r_3$, and the converted frame is their mean,
$(0.95, -0.05)$. With $k = 1$, it would be an exact copy of $r_1$.

The paper reports that below about 30 seconds of reference audio, kNN-VC
falls behind other methods, though even 5 seconds keeps moderate
intelligibility and speaker similarity. At 50 frames per second, 8 minutes
of reference audio is 24,000 frames, about 49 MB in 16-bit floats.

The Run can't load WavLM or a vocoder on the CPU, so it keeps the method and
swaps the parts. It synthesizes two voices, a 110 Hz source and a 210 Hz
target with formants raised 18%. Mel cepstra with each speaker's mean
removed stand in for WavLM layer 6. After matching, it averages the
*magnitude spectra* of the four target frames and turns them into sound with
Griffin-Lim from lesson 5, *Phase and Griffin-Lim*. On its seeded voices, 94% of matched
frames are the same vowel as the source frame, and the output's pitch is the
target's 211 Hz.

> [!KEY]
> kNN-VC changes the speaker by replacing every source frame with the average of its nearest neighbors in the target's own speech. It works because content features match across speakers while every matched frame already carries the target's timbre.

## Detect the fakes

The ASVspoof challenges score *countermeasures*, classifiers that label a
recording bona fide or spoofed. Their 2019 edition split attacks into
logical access, speech made by TTS or voice conversion, and physical access,
recorded replays. AASIST (2022), a graph attention network on raw waveforms,
has a light version, AASIST-L, with only 85k parameters.

A countermeasure outputs a score, and the equal error rate of lesson 23,
*Speaker recognition and diarization*, measures it, with bona fide speech in
the role of the target trials. A false rejection is a bona fide score below
the threshold, a false acceptance is a spoofed score at or above it, and the
EER is where the two rates cross. The Run draws both curves for a toy detector whose scores are two unit
Gaussians 2.5 apart. Theory puts that EER at $\Phi(-1.25) = 10.6\%$, and the
400 seeded samples per class measure 11.0%.

Detectors overfit to the attacks they trained on, and an unseen vocoder or a
phone codec can break them. The complementary approach is *provenance*: mark
generated audio when it's made. AudioSeal (2024) adds an imperceptible
watermark and detects it sample by sample, so it can find a watermarked
segment inside an edited clip.

## Consent and misuse

A voice identifies a person. The EU's GDPR treats biometric data used to
identify someone as a special category, and Illinois's BIPA regulates
voiceprints. A converted voice can impersonate someone to their bank, their
family, or their employer.

The rules for this course follow from that. Convert only your own voice,
with your own opt-in, or voices from public corpora whose licenses allow it,
such as LibriSpeech and VCTK. Never convert a third party's recordings
without their explicit consent. Label converted audio as synthetic when you
share it, and watermark it when you can.

> [!WARNING]
> "It's only a demo" is not consent. The same pipeline that converts a synthetic voice in this exercise converts a real one, so the rules apply the moment the input is a recording of a person.

> [!RECAP]
> - AST's patch grid is $\lfloor (F - 16)/10 \rfloor + 1$ by $\lfloor (T - 16)/10 \rfloor + 1$: 12 by 101 for 1,024 frames.
> - CLAP averages the audio-to-text and text-to-audio cross-entropies of $a t^\top / \tau$ over normalized embeddings.
> - A countermeasure's EER is where false rejections of bona fide speech equal false acceptances of spoofs.
> - kNN-VC replaces each source frame with the mean of its $k = 4$ nearest target frames by cosine distance.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ast_patch_grid(n_mels, n_frames, patch, stride)`** returns `(n_freq, n_time)` from the formula with integer division. The common mistake is rounding up, which counts a patch that runs off the edge.
2. **`patchify(spec, patch, stride)`** returns `(n_freq * n_time, patch * patch)`. Build 4-D row and column index grids by broadcasting, index once, and reshape. Frequency is the outer loop: row `i_f * n_time + i_t`.
3. **`clap_loss(audio_emb, text_emb, tau)`** normalizes with `F.normalize`, forms `a @ t.T / tau`, and averages `F.cross_entropy` over rows and over the transpose. The common mistake is using one direction only.
4. **`zero_shot_classify(audio_emb, class_text_emb)`** returns the `argmax` of the cosine matrix along the class axis.
5. **`equal_error_rate(bonafide, spoof)`** returns `(eer, threshold)`. Sort the scores, use `np.searchsorted` to count scores below each threshold, find the first threshold where the rejection rate reaches the acceptance rate, and interpolate linearly from the one before. The common mistake is swapping the classes, which reports 1 minus the EER.
6. **`cosine_knn(query, keys, k)`** returns `(T, k)` indices, nearest first. Normalize, multiply, take `np.argpartition` on the negated similarities, then sort those $k$.
7. **`knn_vc(source, matching, k)`** indexes the matching set with those indices and averages over axis 1.

## Cost and scaling

The following table uses $T$ source frames, $M$ matching frames, feature
width $d$, and batch size $N$.

| Operation | Time | Memory |
|---|---|---|
| AST self-attention on 1,212 tokens | $O(1212^2 \cdot 768)$ per layer | 1.5 million attention weights per head |
| CLAP loss | $O(N^2 d)$ | $N^2$ logits |
| EER by sorting | $O(n \log n)$ | $O(n)$ |
| kNN-VC matching | $O(T M d)$ | $T \times M$ similarities |

kNN-VC's cost is one matrix product. Converting 10 seconds, 500 frames,
against 8 minutes of reference, 24,000 frames, with $d = 1{,}024$, is about
12 billion multiply-adds and a 12-million-entry similarity matrix. Long
matching sets are processed in blocks, or with an approximate nearest-neighbor
index. CLAP's cost grows with the square of the batch, and a larger batch gives
each positive more negatives, so contrastive models train with large batches.

## Where this breaks

- **Leaky evaluation splits.** ESC-50 clips cut from the same recording share a fold, and UrbanSound8K ships 10 fixed folds. A random split puts near-duplicates in train and test and inflates accuracy.
- **A matching set that is too small.** With a few seconds of target speech, some phones have no good neighbor, and kNN-VC copies the wrong one. Collect more reference audio, or accept lower intelligibility.
- **Features that carry the speaker.** If the content features still encode who is speaking, the source's frames match the target's poorly, and conversion fails. This is why kNN-VC uses layer 6, and why the Run removes each speaker's mean.
- **Countermeasures that don't generalize.** A detector with a low EER on known attacks can fail on a vocoder it never saw. Report results on unseen attacks, and pair detection with watermarking.
- **Prompts that don't match captions.** CLAP's zero-shot accuracy depends on wording. "Dog" and "the sound of a dog barking" can give different answers, so try a few templates on held-out data.

## The exercise

You build `ast_patch_grid` and `patchify` for AST, `clap_loss` and
`zero_shot_classify` for CLAP, `equal_error_rate` for a spoofing
countermeasure, and `cosine_knn` and `knn_vc` for voice conversion. Press
**Run** to print AST's patch counts, hear a synthetic source voice, the
target voice, and the source converted into the target, and see a toy
countermeasure's error curves with its EER.

The checks verify the patch grid for AudioSet's input and two others, your
patches against a stride-10 `Conv2d`, and your CLAP loss against the formula,
its invariance to embedding length, the $\log N$ case, symmetry, and a
finite-difference derivative in $\tau$. They check zero-shot predictions, the
EER on crafted scores and on two Gaussians against theory, and your
neighbors against a full sort. Finally, kNN-VC on synthetic features must
keep the source phone and move to the target speaker for at least 95% of
frames.
