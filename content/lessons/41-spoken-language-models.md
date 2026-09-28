---
title: Spoken language models
slug: 41-spoken-language-models
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
  - 14-language-models-and-word-error-rate
  - 21-self-supervised-speech
  - 22-whisper-and-large-scale-recognition
exercise: 41-spoken-language-models
---

# Spoken language models

A text language model has read more than any person could. It knows that
"seven" is a number, that questions end with answers, and a great deal about
the world. None of that helps with speech until the model can read it. This
lesson is about the two ways to give it that ability.

The first way turns speech into text-like symbols. Cluster the frames of a
self-supervised model into a few hundred *units*, and a language model can
train on unit sequences exactly as it trains on words. That line of work,
called *textless NLP*, showed that speech alone carries enough structure to
learn words and some grammar.

The second way keeps the text model as it is and teaches it to listen. A
frozen speech encoder turns audio into vectors, and a small trained
*projector* maps them into the language model's input space, where they sit
in front of a text prompt like a few extra words. Many speech LLMs work this
way. You build both, and then you connect Whisper's encoder to a
135-million-parameter language model that has never heard sound, and teach it
to write down the digits people say.

## What you already know

- A bigram model estimates $P(w_t \mid w_{t-1})$ from counts, with smoothing for unseen pairs.
- HuBERT's training targets are k-means clusters of frame features, and middle layers of self-supervised models carry phonetic content.
- Whisper's encoder turns 30 seconds of audio into 1,500 vectors, one every 20 ms.
- A causal language model predicts token $t + 1$ from tokens $1$ to $t$, and trains with cross-entropy on that next token.
- Setting `requires_grad` to `False` freezes a parameter: autograd still flows through it, but the optimizer never changes it.

## Units from k-means

Lesson 21, *Self-supervised speech*, turned a layer's 20 ms frames into
*units*. Frames of the same phone, spoken by different people, gather in the
same region of feature space. k-means finds $K$ centers for those regions,
and each frame's unit is the index of its nearest center:

$$
u_t = \operatorname*{arg\,min}_{k} \lVert x_t - \mu_k \rVert_2^2
$$

Read it aloud: frame $x_t$ gets the label $k$ of the center $\mu_k$ at the
smallest squared distance. Expand the square as
$\lVert x \rVert^2 - 2\, x \cdot \mu + \lVert \mu \rVert^2$ and one matrix
product computes every distance at once:

```python
sq = (x**2).sum(1)[:, None] - 2 * x @ centroids.T + (centroids**2).sum(1)[None, :]
units = sq.argmin(axis=1)
```

Lloyd's algorithm, from the same lesson, alternates two steps: assign every frame to its nearest
center, then move every center to the mean of its frames. The *inertia*, the
total squared distance from frames to their centers, never increases.

### Work k-means by hand

Take four 2-D frames, $x_1 = (0, 0)$, $x_2 = (1, 0)$, $x_3 = (4, 4)$, and
$x_4 = (5, 4)$, and start the centers at $\mu_1 = x_1$ and $\mu_2 = x_2$.

- Iteration 1: $x_3$ is 32 from $\mu_1$ and 25 from $\mu_2$, and $x_4$ is 41 and 32, so both go to $\mu_2$. The inertia is $0 + 0 + 25 + 32 = 57$. The centers move to $(0, 0)$ and the mean of $x_2, x_3, x_4$, which is $(3.33, 2.67)$.
- Iteration 2: $x_2$ is 1 from $\mu_1$ and 12.56 from $\mu_2$, so it switches. The inertia is $0 + 1 + 2.22 + 4.56 = 7.78$. The centers move to $(0.5, 0)$ and $(4.5, 4)$.
- Iteration 3: every frame is 0.25 from its center, so the inertia is 1.0, and nothing moves again.

### Collapse the runs

A vowel lasts many frames, so unit sequences stutter: `5 5 5 12 12 7`. Unit
language models read *deduplicated* units, `5 12 7`, with the run lengths
`3 2 1` kept aside. The durations matter again when a vocoder turns units back
into sound, and the pGSLM model predicts them explicitly.

```python
starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
units, durations = u[starts], np.diff(np.r_[starts, len(u)])
```

GSLM (Lakhotia et al., 2021) built the first full pipeline this way: HuBERT
features at 50 Hz, k-means with $K$ of 50, 100, or 200, a Transformer on the
deduplicated units, and a unit-to-speech synthesizer. HuBERT units with 100
or 200 clusters worked best.

## A unit language model

Once speech is a unit sequence, any language model applies. The simplest is
the bigram model from lesson 14, *Language models and word error rate*, with
add-$\alpha$ smoothing:

$$
P(u_t = j \mid u_{t-1} = i) = \frac{c(i, j) + \alpha}{\sum_{j'} c(i, j') + \alpha K}
$$

Read it aloud: count how often unit $j$ follows unit $i$, add $\alpha$ to
every count so no pair has probability zero, and divide by the row's total.
A sequence's log probability is the sum of $\log P$ over its neighbor pairs.

With $K = 3$ units, $\alpha = 1$, and the two sequences `0 1 1 2` and `2 0`,
the pairs are (0, 1), (1, 1), (1, 2), and (2, 0). Row 1 has counts
$(0, 1, 1)$, which become $(1, 2, 2)$ over 5, so
$P(\cdot \mid 1) = (0.2, 0.4, 0.4)$. Row 0 becomes $(1, 2, 1)$ over 4.

```python
counts = np.full((K, K), alpha)
np.add.at(counts, (seq[:-1], seq[1:]), 1)
log_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
```

Perplexity can't compare unit models with different $K$, because the units
differ. Textless NLP instead scores preferences. *sWUGGY* plays a real word
and a similar nonword and asks which one the model finds more likely.
*sBLIMP* does the same for a grammatical and an ungrammatical sentence. The
exercise's last Part 1 check is a toy sWUGGY: your units and bigram model
must prefer real "words" to shuffled ones.

Later systems scaled this up. TWIST (2023) starts the unit model from a
pretrained text LM instead of from scratch, which helps consistently.
SpiritLM (2024) continues Llama 2 on text, units, and sequences that switch
between the two at word boundaries. SpeechGPT (2023) adds 1,000 units to
LLaMA's vocabulary so one model reads and writes both.

> [!KEY]
> Discrete units make speech look like text: k-means labels each frame, deduplication removes the stutter, and any language model can train on what is left.

## The adapter pattern

Units throw information away, such as pitch and speaker detail. The adapter
pattern keeps a continuous signal instead. Its pieces are three models, and
only the smallest one trains:

1. A frozen **speech encoder** turns audio into frames. Whisper's encoder is the usual choice: 30 seconds become 1,500 frames at 50 Hz.
2. A trained **projector** shortens the sequence and maps each frame to the width of the language model's embeddings.
3. A frozen **language model** reads the projected frames as if they were token embeddings, followed by a text prompt, and writes text.

The simplest projector *stacks* $k$ neighboring frames into one wider frame
and passes it through a two-layer MLP:

$$
z_j = W_2\, \operatorname{GELU}\!\left( W_1 \left[ h_{jk};\ h_{jk+1};\ \dots;\ h_{jk+k-1} \right] + b_1 \right) + b_2
$$

Read it aloud: concatenate frames $jk$ to $jk + k - 1$ of the encoder output
$h$, apply a linear layer, a GELU, and a second linear layer, and get one
vector $z_j$ with the language model's width. Stacking by $k$ divides the
frame rate by $k$: $k = 4$ turns 50 Hz into 12.5 Hz, so 30 seconds become 375
soft tokens instead of 1,500.

```python
stacked = h[:, : (T // k) * k].reshape(B, T // k, k * d)
z = fc2(F.gelu(fc1(stacked)))
```

Real systems pick their rates this way. Voxtral stacks by 4 to reach
12.5 Hz, so its 32k-token context holds about 40 minutes of audio.
LLaMA-Omni stacks by 5 to reach 10 Hz. Qwen2-Audio pools with stride 2,
about 25 Hz. SALMONN uses a Q-Former instead, a few learned queries that
cross-attend to windows of frames and emit one vector each.

For the Run, the numbers are the following. Whisper-tiny's encoder has width
384. Each digit clip is padded to one second, and the first 50 encoder frames
are kept. Stacking by 4 gives 12 soft tokens of width $4 \times 384 = 1{,}536$
and drops the last 2 frames. The projector maps 1,536 to a hidden width of
1,024 and then to SmolLM2's 576, which is
$1536 \cdot 1024 + 1024 + 1024 \cdot 576 + 576 = 2{,}164{,}288$ trainable
parameters. The language model's 134,515,008 stay frozen.

> [!TIP]
> Check the norm of your projector's output against the norm of the language model's token embeddings. If the soft tokens are ten times louder than any word, the model attends to nothing else. A LayerNorm or RMSNorm on the projector's output is a common fix.

## Splice and mask

The language model reads one sequence of embeddings, laid out as
`[audio | prompt | target]`:

- The $J$ projected audio vectors.
- The embeddings of a fixed text prompt, $P$ tokens.
- The embeddings of the transcript, $U$ tokens, ending with the end-of-text token.

The model predicts every next token, but only the transcript is the task.
Training on the prompt would teach the model to predict a constant string,
and the audio positions have no token to predict at all. So the loss counts
only target positions:

$$
\mathcal{L} = -\frac{1}{U} \sum_{i=1}^{U} \log p_\theta\!\left( y_i \mid z_{1:J},\ x^{\text{prompt}},\ y_{<i} \right)
$$

Read it aloud: average, over the $U$ transcript tokens, the negative log
probability of each one given all the audio, the prompt, and the transcript
so far. Only the projector's weights $\theta$ change, but the gradient flows
back through every frozen layer of the language model to reach them.

In code, give every masked position the label $-100$, which
`F.cross_entropy` skips with `ignore_index`. Then shift by one, because the
logits at position $t$ predict the token at position $t + 1$:

```python
labels = torch.full((B, J + P + U), -100)
labels[:, J + P :] = target_ids
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
```

### Work the layout by hand

In the Run, the prompt ` Transcript:` is 3 SmolLM2 tokens, `ĠTrans`,
`cript`, and `:`. The word ` seven` is 1 token, id 4962, and end-of-text is
id 0. With $J = 12$, the sequence has $12 + 3 + 2 = 17$ positions:

| Positions | Content | Label |
|---|---|---|
| 0 to 11 | projected audio | $-100$ |
| 12 to 14 | ` Transcript:` | $-100$ |
| 15 | ` seven` | 4962 |
| 16 | end-of-text | 0 |

After the shift, the logits at position 14, the colon, must predict 4962,
and the logits at position 15 must predict end-of-text. Those are the only 2
of the 16 shifted positions in the loss. Position 16's logits predict nothing
and are dropped.

> [!WARNING]
> Forgetting the shift asks each position to predict its own input token, which a model with tied embeddings does almost for free. With the exercise's toy LM, the unshifted loss starts at 0.0005 and held-out accuracy stays at 0%. If your loss is near zero from the first step, check the shift first.

## Train only the projector

This stage, with the encoder and the language model both frozen, is how a
typical recipe starts: align the projector on transcription and captioning
pairs. Later stages fine-tune the language model lightly on varied
instructions about audio, often with LoRA, as SALMONN does.

The Run does the first stage on the Free Spoken Digit Dataset: 3,000
recordings of the digits 0 to 9 by 6 speakers, resampled from 8 kHz to
16 kHz. One speaker, `theo`, is held out for testing. Before training, the
language model answers every clip the same way: it writes `The`, whatever
the audio says. After training, it writes the digit word.

On a laptop CPU, the small version encodes 120 clips, trains 80 steps of 16
in under 20 seconds, and writes the correct word for 19 of 20 held-out clips.
The full version, the one the GPU runs, encodes all 3,000 clips and trains
1,500 steps of 32. Run on a laptop CPU, it took about 10 minutes and wrote the
correct word for all 500 clips of the held-out speaker.

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

1. **`assign_units(features, centroids)`** returns `(T,)` integers. Expand the squared distance so one `@` computes all of them, then `argmin(axis=1)`. The common mistake is a Python loop over frames, which is correct but a hundred times slower.
2. **`kmeans(features, init, n_iter)`** returns the centers and one inertia per iteration. Record the inertia after assigning and before moving the centers. Leave an empty cluster's center where it is. Copy `init` first, or you change the caller's array.
3. **`deduplicate(units)`** returns the collapsed units and their durations. Find run starts with `np.r_[True, u[1:] != u[:-1]]`.
4. **`unit_bigram_lm(sequences, n_units, alpha)`** returns a `(K, K)` log-probability matrix. Use `np.add.at`, because `counts[a, b] += 1` counts a repeated pair only once.
5. **`sequence_log_prob(sequence, log_probs)`** sums `log_probs[seq[:-1], seq[1:]]`.
6. **`stack_frames(h, k)`** slices off the remainder and reshapes to `(B, T // k, k * d)`.
7. **`Projector(d_in, d_model, d_hidden)`** holds two `nn.Linear` layers and applies GELU between them.
8. **`build_adapter_inputs(audio_embeds, prompt_ids, target_ids, embed)`** concatenates the three parts along the time axis and builds the labels. Expand the prompt embeddings to the batch with `expand(B, -1, -1)`.
9. **`next_token_loss(logits, labels)`** shifts, flattens, and calls `F.cross_entropy` with `ignore_index=-100`. The common mistake is forgetting the shift.

## Cost and scaling

The following table compares the two routes, with $K$ units, $T$ encoder
frames per clip, stacking factor $k$, and a language model with $P$
parameters.

| Quantity | Units | Adapter |
|---|---|---|
| Tokens per second of speech | about 50 before deduplication, fewer after | $50 / k$ |
| Added parameters | $K$ embeddings, or a whole model | the projector, a few million |
| Training cost per step | a normal LM step | a forward and backward pass through the frozen $P$ parameters |
| Can the LM speak? | yes, by writing units | no, it writes text only |

Freezing the language model saves optimizer memory, but not compute: the
gradient still passes through every layer on the way to the projector. The
Run's 17-token sequences make each step cheap. Real adapters feed 375 soft
tokens per 30 seconds, and a 10-minute recording at 12.5 Hz is 7,500 tokens,
so attention's cost, quadratic in length, dominates long audio.

## Where this breaks

- **Whisper expects 30 seconds.** Its encoder was trained on padded 30-second windows, and the Hugging Face encoder rejects any other input length. Short clips are padded, as the Run does, and long ones are chunked.
- **Loss on the wrong positions.** Training on audio or prompt positions wastes capacity and can destabilize training. Print the labels of one example before you train.
- **A projector that only memorizes.** With few speakers, a projector can learn voices instead of words. The Run holds out a whole speaker for exactly this reason, and a held-out speaker is the test that counts.
- **Units that lose prosody.** Deduplicated units discard duration and pitch, so a unit model can't tell a question from a statement. pGSLM adds duration and pitch streams, and the adapter keeps continuous features.
- **Instruction collapse.** After heavy transcription training, the model transcribes no matter what the prompt asks. Mix in varied tasks, or add a short stage like SALMONN's activation tuning.

## The exercise

In Part 1, you build `assign_units`, `kmeans`, `deduplicate`,
`unit_bigram_lm`, and `sequence_log_prob`. In Part 2, you build
`stack_frames`, the `Projector`, `build_adapter_inputs`, and
`next_token_loss`. Press **Run** to cluster a synthetic vowel string into
units and hear it, then to train the adapter. On a GPU, the Run connects
`whisper-tiny` to `smollm2-135m` on every `fsdd` clip. On a CPU with the
assets, it runs the small version. Without them, it trains the same adapter
on synthetic frames through a frozen toy language model.

The checks run on the CPU with no assets. They test your units against brute
force, your k-means against Lloyd's algorithm from the same start, and your
deduplication and bigram model against the hand examples. A toy sWUGGY must
prefer real unit words to shuffled ones. They check the shapes and order of
stacked frames, that your projector treats each frame on its own, the exact
layout and labels of your spliced input, and that your loss matches the
formula and sends no gradient to masked positions. The last check trains your
projector through a frozen toy language model and requires at least 70%
held-out accuracy.
