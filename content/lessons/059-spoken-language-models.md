---
title: Spoken language models
slug: 059-spoken-language-models
part: "Part 8 — Beyond recognition and synthesis"
summary: Turn speech into units a language model can read, then connect a frozen speech encoder to a frozen text LLM with a trained projector.
minutes: 80
gpu: true
objectives:
  - Turn frame features into discrete units with k-means, collapse repeated units, and score unit sequences with a bigram unit language model.
  - Explain the adapter pattern, and compute the token rate after frame stacking for Whisper's encoder.
  - Build a projector, splice its output into a language model's input, and mask the loss so only the transcript is trained.
  - Train a projector between frozen whisper-tiny and frozen SmolLM2 until the pair transcribes spoken digits from an unseen speaker.
prereqs:
  - 023-n-gram-language-models
  - 034-self-supervised-speech
  - 035-whisper
exercise: 059-spoken-language-models
---

# Spoken language models

A text language model has read more than any person could. It knows that "seven" is a number, that questions get answers, and a great deal about the world. None of that helps with speech until the model can read it. A language model reads a sequence of embeddings, and sound is a waveform, so something has to translate. This chapter is about the two ways to do that.

The first way turns speech into text-like symbols. Cluster the frames of a self-supervised model into a few hundred *units*, and a language model can train on unit sequences exactly as it trains on words. The second way keeps the text model as it is and teaches it to listen: a frozen speech encoder turns audio into vectors, and a small trained *projector* maps them into the language model's input space, where they sit in front of a text prompt like a few extra words.

You build both, then connect Whisper's encoder to a 135-million-parameter language model that has never heard sound, and teach it to write down the digits people say.

## What you already know

- A bigram model estimates $P(w_t \mid w_{t-1})$ from counts, with smoothing for unseen pairs.
- HuBERT's training targets are k-means clusters of frame features, and middle layers of self-supervised models carry phonetic content.
- Whisper's encoder turns 30 seconds of audio into 1,500 vectors, one every 20 ms.
- A causal language model predicts token $t + 1$ from tokens $1$ to $t$, and trains with cross-entropy on that next token.
- Setting `requires_grad` to `False` freezes a parameter: autograd still flows through it, but the optimizer never changes it.

## Units from k-means

[Self-supervised speech](/lessons/034-self-supervised-speech) turned a layer's 20 ms frames into *units*. Frames of the same phone, spoken by different people, gather in the same region of feature space. k-means finds $K$ centers for those regions, and each frame's unit is the index of its nearest center:

$$
u_t = \operatorname*{arg\,min}_{k} \lVert x_t - \mu_k \rVert_2^2
$$

Read it aloud: frame $x_t$ gets the label $k$ of the center $\mu_k$ at the smallest squared distance. Expand the square as $\lVert x \rVert^2 - 2\, x \cdot \mu + \lVert \mu \rVert^2$ and one matrix product computes every distance at once:

```python
sq = (x**2).sum(1)[:, None] - 2 * x @ centroids.T + (centroids**2).sum(1)[None, :]
units = sq.argmin(axis=1)
```

Lloyd's algorithm alternates two steps: assign every frame to its nearest center, then move every center to the mean of its frames. The *inertia*, the total squared distance from frames to their centers, never increases.

### Work k-means by hand

Take four 2-D frames, $x_1 = (0, 0)$, $x_2 = (1, 0)$, $x_3 = (4, 4)$, and $x_4 = (5, 4)$, and start the centers at $\mu_1 = x_1$ and $\mu_2 = x_2$.

- Iteration 1: $x_3$ is 32 from $\mu_1$ and 25 from $\mu_2$, and $x_4$ is 41 and 32, so both go to $\mu_2$. The inertia is $0 + 0 + 25 + 32 = 57$. The centers move to $(0, 0)$ and $(3.33, 2.67)$, the mean of $x_2, x_3, x_4$.
- Iteration 2: $x_2$ is 1 from $\mu_1$ and 12.56 from $\mu_2$, so it switches. The inertia is $0 + 1 + 2.22 + 4.56 = 7.78$. The centers move to $(0.5, 0)$ and $(4.5, 4)$.
- Iteration 3: every frame is 0.25 from its center, so the inertia is 1.0, and nothing moves again.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Four points at (0,0), (1,0), (4,4), (5,4). The second center starts at (1,0), moves to (3.33,2.67) after iteration 1, and settles at (4.5,4). The first center settles at (0.5,0).">
  <defs><marker id="c056-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
  <line x1="80" y1="200" x2="340" y2="200" class="stroke-muted"/><line x1="80" y1="30" x2="80" y2="200" class="stroke-muted"/>
  <line x1="105" y1="105" x2="250" y2="125" class="stroke-grid dashed"/>
  <circle cx="80" cy="200" r="5" class="fill-a"/><circle cx="115" cy="200" r="5" class="fill-a"/>
  <circle cx="220" cy="60" r="5" class="fill-b"/><circle cx="255" cy="60" r="5" class="fill-b"/>
  <circle cx="115" cy="200" r="9" class="stroke-muted" fill="none"/>
  <circle cx="197" cy="107" r="9" class="stroke-muted" fill="none"/>
  <path d="M115 193 L195 112" class="stroke-muted" fill="none" marker-end="url(#c056-arrow)"/>
  <path d="M203 100 L232 68" class="stroke-muted" fill="none" marker-end="url(#c056-arrow)"/>
  <rect x="93" y="195" width="10" height="10" class="fill-d"/><rect x="233" y="55" width="10" height="10" class="fill-d"/>
  <text x="125" y="196" class="fig-small">x2</text><text x="86" y="216" class="fig-small">x1</text><text x="262" y="56" class="fig-small">x4</text><text x="198" y="52" class="fig-small">x3</text>
  <text x="206" y="125" class="fig-small">mu2 after iteration 1</text>
  <text x="360" y="70">inertia 57 after the first assignment</text>
  <text x="360" y="100">inertia 7.78 after iteration 2</text>
  <text x="360" y="130">inertia 1.0 at (0.5, 0) and (4.5, 4)</text>
  <text x="360" y="170" class="fig-small">Purple squares: final centers.</text>
  <text x="130" y="236" class="fig-small">feature 1</text>
</svg>
Figure 1. The second center starts on x2, is pulled to the mean of three points, and then loses x2 to the first center. The inertia only falls.
```

### Collapse the runs

A vowel lasts many frames, so unit sequences stutter: `5 5 5 12 12 7`. Unit language models read *deduplicated* units, `5 12 7`, with the run lengths `3 2 1` kept aside, because durations matter again when a vocoder turns units back into sound.

```python
starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
units, durations = u[starts], np.diff(np.r_[starts, len(u)])
```

GSLM (Lakhotia et al., 2021) built the first full pipeline this way: HuBERT features at 50 Hz, k-means with $K$ of 50, 100, or 200, a Transformer on the deduplicated units, and a unit-to-speech synthesizer.

## A unit language model

Once speech is a unit sequence, any language model applies. The simplest is the bigram model from [n-gram language models](/lessons/023-n-gram-language-models), with add-$\alpha$ smoothing:

$$
P(u_t = j \mid u_{t-1} = i) = \frac{c(i, j) + \alpha}{\sum_{j'} c(i, j') + \alpha K}
$$

Read it aloud: count how often unit $j$ follows unit $i$, add $\alpha$ to every count so no pair has probability zero, and divide by the row's total. With $K = 3$, $\alpha = 1$, and the sequences `0 1 1 2` and `2 0`, the pairs are (0, 1), (1, 1), (1, 2), and (2, 0). Row 1 has counts $(0, 1, 1)$, which become $(1, 2, 2)$ over 5, so $P(\cdot \mid 1) = (0.2, 0.4, 0.4)$.

```python
counts = np.full((K, K), alpha)
np.add.at(counts, (seq[:-1], seq[1:]), 1)
log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
```

Perplexity can't compare unit models with different $K$, because the units differ. Textless NLP instead scores preferences. *sWUGGY* plays a real word and a similar nonword and asks which one the model finds more likely. The exercise's toy version checks that your units and bigram model prefer real "words" to shuffled ones.

> [!KEY]
> Discrete units make speech look like text: k-means labels each frame, deduplication removes the stutter, and any language model can train on what is left.

## The adapter pattern

Units throw information away, such as pitch and speaker detail. The adapter pattern keeps a continuous signal instead. Its pieces are three models, and only the smallest one trains:

1. A frozen **speech encoder** turns audio into frames. Whisper's encoder turns 30 seconds into 1,500 frames at 50 Hz.
2. A trained **projector** shortens the sequence and maps each frame to the width of the language model's embeddings.
3. A frozen **language model** reads the projected frames as if they were token embeddings, followed by a text prompt, and writes text.

```figure
<svg viewBox="0 0 640 200" role="img" aria-label="Pipeline from audio through a frozen Whisper encoder giving 1,500 frames at 50 Hz, a trainable projector giving 375 soft tokens at 12.5 Hz, and a frozen language model that writes text">
  <defs><marker id="c056-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
  <rect x="10" y="60" width="100" height="60" class="fill-soft-a stroke-a"/><text x="30" y="85">30 s of</text><text x="30" y="103">audio</text>
  <rect x="150" y="60" width="130" height="60" class="fill-sunk stroke-muted"/><text x="160" y="85" class="fig-bold">Encoder</text><text x="160" y="103" class="fig-small">frozen, 1500 x d</text>
  <rect x="320" y="60" width="130" height="60" class="fill-soft-b stroke-b"/><text x="330" y="85" class="fig-bold">Projector</text><text x="330" y="103" class="fig-small">trained, 375 x 576</text>
  <rect x="490" y="60" width="140" height="60" class="fill-sunk stroke-muted"/><text x="500" y="85" class="fig-bold">Language model</text><text x="500" y="103" class="fig-small">frozen, writes text</text>
  <line x1="110" y1="90" x2="146" y2="90" class="stroke-muted" marker-end="url(#c056-arrow2)"/>
  <line x1="280" y1="90" x2="316" y2="90" class="stroke-muted" marker-end="url(#c056-arrow2)"/>
  <line x1="450" y1="90" x2="486" y2="90" class="stroke-muted" marker-end="url(#c056-arrow2)"/>
  <text x="150" y="150" class="fig-small">50 Hz</text><text x="320" y="150" class="fig-small">stack k = 4: 12.5 Hz</text>
  <text x="440" y="40" class="fig-small">+ text prompt embeddings</text>
</svg>
Figure 2. Only the orange projector trains. Stacking four frames cuts 1,500 frames to 375 soft tokens, and the language model sees them like words.
```

The simplest projector *stacks* $k$ neighboring frames into one wider frame and passes it through a two-layer MLP:

$$
z_j = W_2\, \operatorname{GELU}\!\left( W_1 \left[ h_{jk};\ h_{jk+1};\ \dots;\ h_{jk+k-1} \right] + b_1 \right) + b_2
$$

Read it aloud: concatenate frames $jk$ to $jk + k - 1$ of the encoder output $h$, apply a linear layer, a GELU, and a second linear layer, and get one vector $z_j$ with the language model's width. Stacking by $k$ divides the frame rate by $k$: $k = 4$ turns 50 Hz into 12.5 Hz, so 30 seconds become 375 soft tokens.

```python
stacked = h[:, : (T // k) * k].reshape(B, T // k, k * d)
z = fc2(F.gelu(fc1(stacked)))
```

Real systems pick their rates this way: Voxtral stacks by 4 to reach 12.5 Hz, and LLaMA-Omni stacks by 5 to reach 10 Hz. For the Run, whisper-tiny's encoder has width 384, and each digit clip is padded to one second, keeping the first 50 frames. Stacking by 4 gives 12 soft tokens of width $4 \times 384 = 1{,}536$ and drops the last 2 frames. The projector maps 1,536 to a hidden width of 1,024 and then to SmolLM2's 576: $1536 \cdot 1024 + 1024 + 1024 \cdot 576 + 576 = 2{,}164{,}288$ trainable parameters, against 134,515,008 frozen.

## Splice and mask

The language model reads one sequence of embeddings, laid out as `[audio | prompt | target]`: the $J$ projected audio vectors, the embeddings of a fixed text prompt of $P$ tokens, and the embeddings of the transcript, $U$ tokens ending with end-of-text. The model predicts every next token, but only the transcript is the task. Training on the prompt would teach the model a constant string, and the audio positions have no token to predict. So the loss counts only target positions:

$$
\mathcal{L} = -\frac{1}{U} \sum_{i=1}^{U} \log p_\theta\!\left( y_i \mid z_{1:J},\ x^{\text{prompt}},\ y_{<i} \right)
$$

Read it aloud: average, over the $U$ transcript tokens, the negative log probability of each one given all the audio, the prompt, and the transcript so far. Only the projector's weights $\theta$ change, but the gradient flows back through every frozen layer of the language model to reach them. In code, give every masked position the label $-100$, which `F.cross_entropy` skips with `ignore_index`, and shift by one, because the logits at position $t$ predict the token at position $t + 1$:

```python
labels = torch.full((B, J + P + U), -100)
labels[:, J + P :] = target_ids
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
```

### Work the layout by hand

In the Run, the prompt ` Transcript:` is 3 SmolLM2 tokens. The word ` seven` is 1 token, id 4962, and end-of-text is id 0. With $J = 12$, the sequence has $12 + 3 + 2 = 17$ positions:

| Positions | Content | Label |
|---|---|---|
| 0 to 11 | projected audio | $-100$ |
| 12 to 14 | ` Transcript:` | $-100$ |
| 15 | ` seven` | 4962 |
| 16 | end-of-text | 0 |

After the shift, the logits at position 14, the colon, must predict 4962, and the logits at position 15 must predict end-of-text. Those are the only 2 of the 16 shifted positions in the loss.

> [!WARNING]
> Forgetting the shift asks each position to predict its own input token, which a model with tied embeddings does almost for free. With the exercise's toy LM, the unshifted loss starts at 0.0005 and held-out accuracy stays at 0%. If your loss is near zero from the first step, check the shift first.

## Train only the projector

This stage, with the encoder and the language model both frozen, is how a typical recipe starts: align the projector on transcription and captioning pairs. The Run does it on the Free Spoken Digit Dataset: 3,000 recordings of the digits 0 to 9 by 6 speakers, resampled from 8 kHz to 16 kHz. One speaker, `theo`, is held out for testing. Before training, the language model answers every clip the same way: it writes `The`, whatever the audio says. After training, it writes the digit word.

On a laptop CPU, the small version encodes 120 clips, trains 80 steps of 16 in under 20 seconds, and writes the correct word for 19 of 20 held-out clips. The full version, the one the GPU runs, encodes all 3,000 clips and trains 1,500 steps of 32. Run on a laptop CPU, it took about 10 minutes and wrote the correct word for all 500 clips of the held-out speaker.

> [!KEY]
> An adapter trains a few million parameters between two frozen models. The encoder already hears and the language model already writes; the projector only has to translate one's frames into the other's embeddings.

> [!RECAP]
> - Units are the argmin of squared distance to k-means centers; deduplication keeps each run once and stores its length.
> - An add-$\alpha$ bigram model scores unit sequences, and preferences such as sWUGGY evaluate it.
> - Stacking $k$ frames divides the frame rate by $k$; a two-layer MLP maps each stack to the language model's width.
> - The input is `[audio | prompt | target]`, and the labels are $-100$ everywhere except the target.
> - The loss shifts by one and averages only over labeled positions.

## Build it step by step

Build the functions in the order the checks test them.

1. **`assign_units(features, centroids)`** returns `(T,)` integers. Expand the squared distance so one `@` computes all of them, then `argmin(axis=1)`.
2. **`kmeans(features, init, n_iter)`** returns the centers and one inertia per iteration. Record the inertia after assigning and before moving the centers, leave an empty cluster's center where it is, and copy `init` first.
3. **`deduplicate(units)`** returns the collapsed units and their durations.
4. **`unit_bigram_lm(sequences, n_units, alpha)`** returns a `(K, K)` log-probability matrix. Use `np.add.at`, because `counts[a, b] += 1` counts a repeated pair only once. **`sequence_log_prob`** sums `log_probs[seq[:-1], seq[1:]]`.
5. **`stack_frames(h, k)`** slices off the remainder and reshapes to `(B, T // k, k * d)`.
6. **`Projector(d_in, d_model, d_hidden)`** holds two `nn.Linear` layers with GELU between them.
7. **`build_adapter_inputs`** concatenates the three parts along the time axis and builds the labels. Expand the prompt embeddings with `expand(B, -1, -1)`.
8. **`next_token_loss(logits, labels)`** shifts, flattens, and calls `F.cross_entropy` with `ignore_index=-100`.

## Where this breaks

- **A projector that only memorizes.** With few speakers, a projector can learn voices instead of words. The Run holds out a whole speaker for exactly this reason.
- **Units that lose prosody.** Deduplicated units discard duration and pitch, so a unit model can't tell a question from a statement. The adapter keeps continuous features.

## The exercise

In Part 1, you build `assign_units`, `kmeans`, `deduplicate`, `unit_bigram_lm`, and `sequence_log_prob`. In Part 2, you build `stack_frames`, the `Projector`, `build_adapter_inputs`, and `next_token_loss`. Press **Run** to cluster a synthetic vowel string into units and hear it, then to train the adapter. On a GPU, the Run connects `whisper-tiny` to `smollm2-135m` on every `fsdd` clip. On a CPU with the assets, it runs the small version.

The checks run on the CPU with no assets. They test your units against brute force, your k-means against Lloyd's algorithm from the same start, and your deduplication and bigram model against the hand examples. A toy sWUGGY must prefer real unit words to shuffled ones. They check the shapes and order of stacked frames, the exact layout and labels of your spliced input, and that your loss matches the formula and sends no gradient to masked positions. The last check trains your projector through a frozen toy language model and requires at least 70% held-out accuracy.
