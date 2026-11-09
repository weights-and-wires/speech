---
title: Codec language models
slug: 048-codec-language-models
part: "Part 6 — Codecs and token models"
summary: Lay codec tokens out for a language model with the delay pattern, VALL-E's two stages, and MaskGIT, and train one to its entropy floor.
minutes: 40
gpu: false
objectives:
  - Explain semantic, acoustic, and hybrid tokens, and what each gives a speech generator.
  - Build the MusicGen delay pattern and its inverse, and count the steps each multi-codebook layout needs.
  - Build VALL-E's autoregressive and non-autoregressive inputs and the loss mask that goes with them.
  - Decode tokens in parallel with MaskGIT's cosine schedule and confidence-based selection.
  - Compute the entropy floor of a synthetic codec source and train a small model that reaches it.
prereqs:
  - 047-neural-audio-codecs
  - 041-wavenet
exercise: 048-codec-language-models
---

# Codec language models

Once a codec turns speech into a grid of integers, speech generation becomes the problem that large language models already solve: predict the next token. Give the model a transcript and three seconds of someone's voice as codec tokens, and let it continue. The codec's decoder turns whatever it writes back into sound, in that person's voice. That recipe, from VALL-E in 2023, is how much of speech synthesis has worked since.

The catch is that a codec frame is 8 or 32 integers, one per codebook. A language model writes one token at a time, so how do you order a grid? The order decides how many steps the model takes, and whether it has seen what it needs when it predicts each token. You build the layouts real systems use. Then you train a small model on a synthetic source whose entropy you can compute, and watch the delay pattern reach that floor while the naive layout cannot.

## What you already know

- A codec sends $N_q$ codebook indices per frame, at a frame rate of 12.5 to 75 Hz (see [Neural audio codecs](/lessons/047-neural-audio-codecs)).
- The first RVQ stage carries the coarse content, and later stages refine it.
- An autoregressive model factorizes $p(x_1, \ldots, x_n)$ as $\prod_i p(x_i \mid x_{<i})$ and trains with cross-entropy on the next token.
- The entropy of a distribution is the lowest average cross-entropy any model can reach on samples from it.
- WaveNet generated audio one sample at a time, which was slow for exactly the reason this chapter addresses (see [WaveNet](/lessons/041-wavenet)).

## Which tokens

Speech models predict three kinds of tokens:

- **Semantic tokens** are cluster indices of self-supervised speech features, such as k-means over HuBERT layers. They keep what was said and some prosody, and drop most of the speaker and the room. They are easy to predict from text, but you cannot decode them to audio on their own.
- **Acoustic tokens** are RVQ codes from a codec such as EnCodec or DAC. They decode to audio, but they carry every detail of the recording, so they are harder to predict.
- **Hybrid tokens** come from a codec whose first codebook is trained to be semantic. Mimi distills WavLM into its first codebook. In the previous chapter, that codebook alone scored an SI-SDR of $-44$ dB and still carried recognizable speech.

AudioLM made the hierarchy explicit: predict semantic tokens first for long-range structure, then coarse acoustic tokens, then fine ones (Borsos et al., 2023).

## Lay out the codebooks

Write a clip's codes as a grid with $K$ rows, one per codebook, and $T$ columns, one per frame. A language model needs a sequence of steps, and at each step it predicts one or more grid cells. The layouts below follow the language of MusicGen's *patterns* (Copet et al., 2023):

- **Flatten.** One cell per step, frame by frame. Exact, but $K T$ steps: 24,000 steps for 10 seconds of EnCodec at 24 kbps.
- **Parallel.** A whole column per step, $T$ steps. But codebook $k$ of a frame is predicted without codebooks below $k$ of the same frame, which it depends on.
- **Delay.** Shift row $k$ right by $k$ steps, then predict a whole column per step. $T + K - 1$ steps.
- **Two stages.** VALL-E predicts row 1 autoregressively, then fills the other rows in parallel passes.
- **Hierarchical.** A large Transformer takes one step per frame, and a small *depth* Transformer predicts that frame's $K$ codes one by one. Moshi and Sesame CSM work this way.

Flattening sees everything but is slow, and the parallel layout is fast but blind to the dependency between codebooks. Is there a layout that is both? That is what the delay pattern gives you.

### The delay pattern

The delay pattern shifts codebook $k$ (counting from 0) right by $k$ steps, and fills the gaps with a pad token:

$$
D[k, s] =
\begin{cases}
C[k, s - k] & \text{if } 0 \le s - k < T \\
\text{pad} & \text{otherwise}
\end{cases}
$$

Read it aloud: at step $s$, row $k$ holds frame $s - k$ of codebook $k$, or the pad token when that frame does not exist. When the model predicts codebook $k$ of frame $t$ at step $t + k$, it has already seen codebooks $0$ to $k - 1$ of that frame. The cost is only $K - 1$ extra steps.

```python
out = np.full((K, T + K - 1), pad)
for k in range(K):
    out[k, k:k + T] = codes[k]
```

Take $K = 3$ codebooks and $T = 4$ frames, with codes $a_t$, $b_t$, and $c_t$. The delayed grid has $3 + 4 - 1 = 6$ steps, against 12 for flattening and 4 for parallel.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="The delay pattern: three codebooks of four frames become a 3 by 6 grid, with row k shifted right by k steps and padded with P">
<text x="131.0" y="38" class="fig-small" text-anchor="middle">step 1</text>
<text x="193.0" y="38" class="fig-small" text-anchor="middle">step 2</text>
<text x="255.0" y="38" class="fig-small" text-anchor="middle">step 3</text>
<text x="317.0" y="38" class="fig-small" text-anchor="middle">step 4</text>
<text x="379.0" y="38" class="fig-small" text-anchor="middle">step 5</text>
<text x="441.0" y="38" class="fig-small" text-anchor="middle">step 6</text>
<text x="90" y="76.0" class="fig-small" text-anchor="end">codebook 1</text>
<rect x="100" y="50" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="129.0" y="74.0" class="fig-small" text-anchor="middle">a1</text>
<rect x="162" y="50" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="191.0" y="74.0" class="fig-small" text-anchor="middle">a2</text>
<rect x="224" y="50" width="58" height="40" rx="4" class="fill-soft-b stroke-b"/>
<text x="253.0" y="74.0" class="fig-small" text-anchor="middle">a3</text>
<rect x="286" y="50" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="315.0" y="74.0" class="fig-small" text-anchor="middle">a4</text>
<rect x="348" y="50" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="377.0" y="74.0" class="fig-small" text-anchor="middle">P</text>
<rect x="410" y="50" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="439.0" y="74.0" class="fig-small" text-anchor="middle">P</text>
<text x="90" y="120.0" class="fig-small" text-anchor="end">codebook 2</text>
<rect x="100" y="94" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="129.0" y="118.0" class="fig-small" text-anchor="middle">P</text>
<rect x="162" y="94" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="191.0" y="118.0" class="fig-small" text-anchor="middle">b1</text>
<rect x="224" y="94" width="58" height="40" rx="4" class="fill-soft-b stroke-b"/>
<text x="253.0" y="118.0" class="fig-small" text-anchor="middle">b2</text>
<rect x="286" y="94" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="315.0" y="118.0" class="fig-small" text-anchor="middle">b3</text>
<rect x="348" y="94" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="377.0" y="118.0" class="fig-small" text-anchor="middle">b4</text>
<rect x="410" y="94" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="439.0" y="118.0" class="fig-small" text-anchor="middle">P</text>
<text x="90" y="164.0" class="fig-small" text-anchor="end">codebook 3</text>
<rect x="100" y="138" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="129.0" y="162.0" class="fig-small" text-anchor="middle">P</text>
<rect x="162" y="138" width="58" height="40" rx="4" class="fill-sunk stroke-grid"/>
<text x="191.0" y="162.0" class="fig-small" text-anchor="middle">P</text>
<rect x="224" y="138" width="58" height="40" rx="4" class="fill-soft-b stroke-b"/>
<text x="253.0" y="162.0" class="fig-small" text-anchor="middle">c1</text>
<rect x="286" y="138" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="315.0" y="162.0" class="fig-small" text-anchor="middle">c2</text>
<rect x="348" y="138" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="377.0" y="162.0" class="fig-small" text-anchor="middle">c3</text>
<rect x="410" y="138" width="58" height="40" rx="4" class="fill-soft-a stroke-a"/>
<text x="439.0" y="162.0" class="fig-small" text-anchor="middle">c4</text>
<text x="100" y="208" class="fig-small">Step 3 (orange) predicts a3, b2, and c1 together.</text>
<text x="100" y="228" class="fig-small">c1 depends on a1 and b1, which came earlier. Cost: 6 steps, against 12 flattened.</text>
</svg>
Figure 1. Row k is shifted right by k steps, and the gaps hold the pad token P. Notice that the orange column predicts a3, b2, and c1 together, and c1 already has its inputs a1 and b1 from earlier columns.
```

MusicGen and Parler-TTS use this pattern. To decode, undo the delay: row $k$ of the original is columns $k$ to $k + T - 1$ of row $k$.

The following demo builds the grid and shows what each step can see.

```demo
{"id": "delay-pattern", "caption": "Pick a cell in the delayed grid and see which cells the model has already produced when it predicts it; then switch to the parallel layout and see what goes missing."}
```

> [!WARNING]
> Forgetting to undo the delay before decoding gives audio that is garbled but recognizably speech-like, which makes the bug hard to spot. Mask the pad positions out of the loss too, or the model learns to predict pads.

## VALL-E: one hard stage, then easy ones

VALL-E predicts EnCodec tokens at 6 kbps: 75 frames per second, 8 codebooks of 1024 (Wang et al., 2023). It splits the grid by codebook:

$$
p(C \mid x, \tilde C) = \prod_{t} p(c_{t,1} \mid c_{<t,1}, \tilde c_{:,1}, x) \prod_{j=2}^{8} p(c_{:,j} \mid c_{:,<j}, x, \tilde C)
$$

Read it aloud: $x$ is the phoneme sequence and $\tilde C$ the codes of a three-second prompt of the target speaker. The first product is an autoregressive model over codebook 1, one frame at a time, conditioned on the phonemes and the prompt. The second is a non-autoregressive model that predicts all frames of codebook $j$ in one pass, given every codebook below $j$.

For the autoregressive stage, the training sequence is the phonemes, then the prompt's codebook-1 codes, then the target's, then an end token. The loss counts only the target codes and the end token, because the phonemes and prompt are context. The non-autoregressive model for stage $j$ reads the sum of the embeddings of the codebooks already known, with one embedding table $E_k$ per codebook:

$$
h_t^{(j)} = \sum_{k < j} E_k[c_{t,k}]
$$

Read it aloud: at frame $t$, look up each lower codebook's code in its own table and add the vectors. For a 10-second utterance, that is 750 autoregressive steps and 7 parallel passes, instead of 6,000 steps.

```python
h = sum(E[k][codes[k]] for k in range(stage))   # (T, d)
```

## MaskGIT: guess everything, keep the best

SoundStorm replaced the acoustic stages of AudioLM with a masked model that decodes many tokens at once (Borsos et al., 2023). The procedure is MaskGIT's. Start with every token masked. At each iteration, predict every masked token, keep the ones the model is most confident about, and mask the rest again. A *cosine schedule* decides how many stay masked:

$$
n_i = \left\lfloor N \cos\left(\frac{\pi}{2} \cdot \frac{i}{I}\right) \right\rfloor
$$

Read it aloud: after iteration $i$ of $I$, $n_i$ of the $N$ tokens are still masked. The cosine falls slowly at first and fast at the end, so the model commits to few tokens while it knows little, and to many once the context is rich.

Take $N = 16$ tokens and $I = 4$ iterations. At $i = 1$, $16 \cos(\pi/8) = 14.78$, so 14 stay masked and 2 are revealed. At $i = 2$, $16 \cos(\pi/4) = 11.31$, so 11 stay masked. At $i = 3$, $16 \cos(3\pi/8) = 6.12$, so 6 stay masked. At $i = 4$, $\cos(\pi/2) = 0$, so the last 6 are revealed.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="MaskGIT cosine schedule for 16 tokens and 4 iterations: masked counts 16, 14, 11, 6, 0">
<line x1="60" y1="170" x2="560" y2="170" class="stroke-muted"/><line x1="60" y1="170" x2="60" y2="20" class="stroke-muted"/>
<rect x="90" y="42" width="60" height="128" class="fill-a"/>
<text x="120" y="36" class="fig-small" text-anchor="middle">16</text>
<text x="120" y="190" class="fig-small" text-anchor="middle">start</text>
<rect x="190" y="58" width="60" height="112" class="fill-a"/>
<text x="220" y="52" class="fig-small" text-anchor="middle">14</text>
<text x="220" y="190" class="fig-small" text-anchor="middle">after 1</text>
<rect x="290" y="82" width="60" height="88" class="fill-a"/>
<text x="320" y="76" class="fig-small" text-anchor="middle">11</text>
<text x="320" y="190" class="fig-small" text-anchor="middle">after 2</text>
<rect x="390" y="122" width="60" height="48" class="fill-a"/>
<text x="420" y="116" class="fig-small" text-anchor="middle">6</text>
<text x="420" y="190" class="fig-small" text-anchor="middle">after 3</text>

<text x="520" y="164" class="fig-small" text-anchor="middle">0</text>
<text x="520" y="190" class="fig-small" text-anchor="middle">after 4</text>
<text x="310" y="212" class="fig-small" text-anchor="middle">iteration</text>
<text x="66" y="16" class="fig-small">tokens still masked</text>
</svg>
Figure 2. The masked count for N = 16 and I = 4. Notice that the schedule reveals 2, 3, 5, and 6 tokens, which is slow at first and fast at the end.
```

The confidence of a guess is its softmax probability. Tokens already revealed get infinite confidence so they never change, and the $n_i$ least confident guesses go back under the mask.

```python
conf[~masked] = np.inf
keep_masked = np.argsort(conf)[:n_i]
```

SoundStorm decodes 12 levels of SoundStream codes at 50 Hz, level by level, with 16 iterations for the first level and 1 for each of the other 11. Thirty seconds of audio, 18,000 tokens, takes 27 forward passes.

> [!KEY]
> Every layout trades steps for dependencies. Flattening sees everything and takes $KT$ steps. The delay pattern takes $T + K - 1$ steps and still lets each codebook see the ones below it. Masked decoding takes a fixed number of passes and relies on the schedule to reveal tokens in a good order.

## The floor a model can reach

The exercise's synthetic source makes the trade measurable. Codebook 0 is a melody: a Markov chain over 8 notes that steps up with probability 0.5, down with 0.3, and stays with 0.2, wrapping around. Codebook 1 is a timbre that depends on the same frame's note: note $j$ picks timbre $2j \bmod 8$ with probability 0.8. So codebook 1 of frame $t$ depends on codebook 0 of frame $t$, as a codec's fine codebook depends on its coarse one.

A model can do no better than the conditional entropy of what it predicts given what it sees. Write $\pi$ for the stationary distribution of the chain $A$, and $H(\cdot)$ for the entropy of one row:

$$
H_{\text{delay}} = \sum_i \pi_i H(A_i) + \sum_i \pi_i H(B_i), \qquad
H_{\text{parallel}} = \sum_i \pi_i H(A_i) + \sum_i \pi_i H\big((AB)_i\big)
$$

Read it aloud: both layouts pay the entropy of the next note. The delay layout then predicts the timbre after seeing its own frame's note, so it pays the entropy of $B$. The parallel layout predicts the timbre from the previous note only, so it pays the entropy of the mixture $(AB)_i$, the timbre distribution averaged over where the melody might go next.

The melody chain is symmetric, so $\pi$ is uniform, and every row has the same entropy. The next note costs $-(0.5 \ln 0.5 + 0.3 \ln 0.3 + 0.2 \ln 0.2) = 1.030$ nats. The timbre given its note costs $-(0.8 \ln 0.8 + 7 \times \tfrac{0.2}{7} \ln \tfrac{0.2}{7}) = 0.890$ nats. The timbre given the previous note comes from row 0 of $AB$, which is $(0.183, 0.029, 0.414, 0.029, 0.029, 0.029, 0.260, 0.029)$, with entropy 1.534 nats.

So $H_{\text{delay}} = 1.030 + 0.890 = 1.919$ nats per frame, and $H_{\text{parallel}} = 1.030 + 1.534 = 2.564$.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="Entropy floors in nats per frame: the delay layout 1.919, the parallel layout 2.564, with trained models at 1.924 and 2.576">
<line x1="170.0" y1="30" x2="170.0" y2="150" class="stroke-grid"/><text x="170.0" y="168" class="fig-small" text-anchor="middle">0</text>
<line x1="290.0" y1="30" x2="290.0" y2="150" class="stroke-grid"/><text x="290.0" y="168" class="fig-small" text-anchor="middle">1</text>
<line x1="410.0" y1="30" x2="410.0" y2="150" class="stroke-grid"/><text x="410.0" y="168" class="fig-small" text-anchor="middle">2</text>
<line x1="530.0" y1="30" x2="530.0" y2="150" class="stroke-grid"/><text x="530.0" y="168" class="fig-small" text-anchor="middle">3</text>
<text x="160" y="66" class="fig-small" text-anchor="end">delay layout</text>
<rect x="170" y="44" width="123.6" height="32" class="fill-a"/>
<rect x="293.6" y="44" width="106.8" height="32" class="fill-b"/>
<line x1="400.9" y1="38" x2="400.9" y2="82" class="stroke-ink" stroke-width="2.5"/>
<text x="410.3" y="64" class="fig-small">1.919</text>
<text x="160" y="118" class="fig-small" text-anchor="end">parallel layout</text>
<rect x="170" y="96" width="123.6" height="32" class="fill-a"/>
<rect x="293.6" y="96" width="184.1" height="32" class="fill-b"/>
<line x1="479.1" y1="90" x2="479.1" y2="134" class="stroke-ink" stroke-width="2.5"/>
<text x="487.6" y="116" class="fig-small">2.564</text>
<text x="170" y="196" class="fig-small">entropy in nats per frame. Blue: next note, 1.030. Orange: timbre given what the model sees.</text>
<text x="170" y="212" class="fig-small">The black tick is where the trained tiny model lands.</text>
</svg>
Figure 3. The two floors in nats per frame, with the loss the tiny model reaches as a black tick. Notice that the model lands almost on its floor in both layouts, and that the gap of 0.64 nats between the floors comes from the layout, not from the model.
```

In the **Run**, the tiny model reaches 1.924 with the delay pattern and 2.576 with the parallel one.

> [!KEY]
> A model's loss cannot fall below the conditional entropy of what it predicts given what it sees. The layout decides what it sees, so the layout sets the floor, whatever the model's size.

Modern systems mostly sidestep the problem. Single-codebook tokenizers at low frame rates make TTS ordinary language modeling again: 10 seconds is 250 CosyVoice 2 tokens, against 6,000 for EnCodec at 6 kbps. Systems such as VibeVoice predict continuous latents with a diffusion head instead, which the [diffusion chapters](/lessons/049-diffusion-from-scratch) build.

> [!RECAP]
> - Semantic tokens are easy to predict and cannot be decoded alone. Acoustic tokens decode and are hard to predict. Hybrid codecs put semantics in codebook 0.
> - The delay pattern shifts codebook $k$ right by $k$ steps: $T + K - 1$ steps, and each codebook sees the ones below it in its frame.
> - VALL-E's autoregressive stage puts the loss on the target codes only, and its parallel stages read summed embeddings of the lower codebooks.
> - MaskGIT keeps $\lfloor N \cos(\frac{\pi}{2} \frac{i}{I}) \rfloor$ tokens masked after iteration $i$, choosing the least confident.
> - A model's loss is bounded below by the conditional entropy of what it predicts given what it sees.

## Build it step by step

Build the functions in the order the checks test them.

1. **`apply_delay(codes, pad)` and `undo_delay(delayed)`.** Fill an array with `pad`, write row $k$ at offset $k$, and slice it back out. Common mistake: delaying by $k + 1$, which wastes a step and breaks the round trip.
2. **`valle_ar_example(phonemes, prompt_c1, target_c1, eos)`.** Concatenate, shift by one, and mask the targets from the first target code on. Common mistake: counting the prompt in the loss, which trains the model to copy the prompt.
3. **`nar_input(codes, embeddings, stage)`.** Sum `embeddings[k][codes[k]]` for `k < stage`.
4. **`maskgit_schedule(n_tokens, n_iters)`.** Apply the cosine with `math.floor`, clamp so each entry falls by at least one, and end at 0.
5. **`maskgit_decode(logits_fn, n_tokens, n_iters, mask_id)`.** Softmax, `argmax`, confidences, infinite confidence for revealed tokens, and `np.argsort` to choose which to keep masked. Common mistake: letting a revealed token be re-predicted and changed.
6. **`markov_code_entropy(A, B)`.** Find $\pi$ as the eigenvector of `A.T` for eigenvalue 1, normalized to sum to 1, and weight the row entropies of `A`, `B`, and `A @ B`.
7. **`lm_inputs_targets(grid, pad, bos)` and `masked_cross_entropy(logits, targets, mask)`.** Shift right with a BOS column, and average the per-token cross-entropy over the mask. Common mistake: passing pad ids to `F.cross_entropy`, which fails because they are not valid classes.
8. **`TinyCodecLM`.** One embedding table per codebook, summed, then the MLP and a head reshaped to `(N, K, S, V)`. Common mistake: reshaping the head's output to `(N, K, S, V)` directly, when its layout is `(N, S, K * V)`. Reshape to `(N, S, K, V)` and permute.

## Where this breaks

**Loops and babble.** Autoregressive codec models repeat a token forever or stop too early. VALL-E samples instead of using beam search, and VALL-E 2 resamples whenever a token repeats too often within a recent window.

**Error accumulation and hallucination.** A wrong codebook-1 token cannot be fixed by later stages, and a model can skip or repeat words. Sample several outputs and rerank them with an ASR model.

**Cloning without consent.** Three seconds of audio is enough to imitate a voice. VALL-E's weights were not released, and some open systems add an inaudible watermark to everything they generate.

## The exercise

You build `apply_delay`, `undo_delay`, `valle_ar_example`, `nar_input`, `maskgit_schedule`, `maskgit_decode`, `markov_code_entropy`, `lm_inputs_targets`, `masked_cross_entropy`, and `TinyCodecLM`. The checks test the delayed grid, its round trip and dependency order, the VALL-E inputs, the cosine schedule $(14, 11, 6, 0)$, and confidence-based decoding. They compare your entropy floors with brute force, and train your model with both layouts: the delay pattern must land within 0.08 nats of its floor, and the parallel pattern must stay about 0.6 nats above it.

Press **Run** to see the delayed grid, the two training curves against their floors, and hear four seconds of the source melody and four seconds sampled from your delay-pattern model.
