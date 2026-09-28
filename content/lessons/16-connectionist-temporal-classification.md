---
title: Connectionist temporal classification
slug: 16-connectionist-temporal-classification
part: "Part 4 — Neural speech recognition"
summary: Train a recognizer from transcripts alone by summing over every alignment with a blank symbol, and derive the CTC loss and its gradient.
minutes: 80
gpu: false
objectives:
  - Explain the blank symbol and the collapse function, and why a repeated label needs a blank between its copies.
  - Compute the CTC forward and backward recursions over the blank-extended label sequence in log space.
  - Derive the gradient of the CTC loss with respect to the logits as model posterior minus alignment posterior.
  - Match torch.nn.functional.ctc_loss exactly, including its (T, N, C) layout and its reductions.
prereqs:
  - 13-hidden-markov-models
  - 15-neural-acoustic-models
exercise: 16-connectionist-temporal-classification
---

# Connectionist temporal classification

A frame classifier needs a label for every frame, and real speech doesn't
come with them. LibriSpeech gives you an audio file and the sentence that was
read, nothing more. Somewhere in those 1,500 frames is the "h" of "hello",
but nobody wrote down where.

**Connectionist temporal classification** (CTC), from Graves and colleagues in
2006, removes the need to know. The network emits one symbol per frame, from
the alphabet plus an extra "nothing here" symbol called **blank**. Many
different frame-by-frame sequences spell the same transcript. CTC adds up the
probability of all of them and trains the network to make that total large.
It's like grading a student on whether the essay says the right thing,
without caring which line each sentence starts on.

CTC is how many recognizers are trained. wav2vec 2.0 is fine-tuned with it,
and Conformer-CTC and FastConformer-CTC models use it as their only loss. In
this lesson you build the loss, its gradient, and a batched version that
matches torch to the last digit.

## What you already know

- The forward algorithm sums over every path through a trellis in $O(TN^2)$
  time, with log-sum-exp to avoid underflow, as *Hidden Markov models*
  showed.
- The backward algorithm runs the same sum from the end, and
  $\sum_i \alpha_t(i)\beta_t(i)$ is the total probability at every frame.
- A subsampling front end sets the number of encoder frames $T$, as *Neural
  acoustic models* showed.
- For softmax outputs $y = \operatorname{softmax}(u)$ and a cross-entropy loss
  with target distribution $q$, the gradient with respect to the logits $u$
  is $y - q$.

## Blank and the collapse function

At each of $T$ frames, the network outputs a distribution $y^t$ over $C$
symbols: blank, written $\varnothing$ or `_`, at index 0, and the labels at
indices 1 to $C - 1$. A **path** $\pi$ picks one symbol per frame. The
**collapse function** $\mathcal{B}$ turns a path into a transcript in two
steps: merge each run of repeated symbols into one, then delete the blanks.
The following are examples:

- $\mathcal{B}(\texttt{aa\_ab\_\_}) = \texttt{aab}$. The run `aa` merges, and
  the blank between the next two `a`s keeps them apart.
- $\mathcal{B}(\texttt{a\_a}) = \texttt{aa}$, but $\mathcal{B}(\texttt{aa}) =
  \texttt{a}$.
- $\mathcal{B}(\texttt{\_\_\_})$ is the empty transcript.

The order matters. Merging first is what lets a label last several frames,
and the blank is what lets a label appear twice in a row. The "ll" in "hello"
can only come out of a path with a blank between the two runs of `l`.

The network's outputs at different frames are treated as independent given
the audio, so a path's probability is a product, and a transcript
$\mathbf{l}$'s probability is a sum over the paths that collapse to it:

$$
p(\mathbf{l} \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y^t_{\pi_t}, \qquad \mathcal{L}_{\text{CTC}} = -\log p(\mathbf{l} \mid x)
$$

Read it aloud: for every path that collapses to the transcript, multiply the
probabilities of its symbols frame by frame, then add up all those products.
The loss is minus the log of the total. Here $\mathcal{B}^{-1}(\mathbf{l})$ is
the set of paths that collapse to $\mathbf{l}$.

## The extended sequence and its lattice

The number of paths grows exponentially with $T$, so you sum them with
dynamic programming, as in the forward algorithm. The trick is a small HMM
built from the transcript. Put a blank before, between, and after the labels.
For $\mathbf{l} = (l_1, \dots, l_L)$, the **extended sequence** is

$$
\mathbf{l}' = (\varnothing,\ l_1,\ \varnothing,\ l_2,\ \dots,\ l_L,\ \varnothing), \qquad S = |\mathbf{l}'| = 2L + 1
$$

Read it aloud: the labels sit at the odd positions and blanks at the even
ones. In numpy, `ext = np.full(2 * L + 1, blank)` and then
`ext[1::2] = target`.

Every valid path walks through the states of $\mathbf{l}'$ from left to right,
one state per frame. From state $s$ it can **stay** at $s$, **advance** to
$s + 1$, or **skip** to $s + 2$. A skip jumps over a blank, and it's allowed
only when $l'_{s+2}$ is a label different from $l'_s$. A skip into a blank
would jump a label. A skip between two copies of the same label would merge
them under $\mathcal{B}$, so the path must pass through the blank between
them.

The following demo draws this lattice, states down and frames across, and
every path through it.

```demo
{"id": "ctc-lattice", "caption": "Type the label eee and lower T until the loss becomes infinite: it needs five frames, one per letter plus a blank between each pair of copies."}
```

## The forward recursion

Define $\alpha_t(s)$ as the total probability of every path prefix over
frames 1 to $t$ that ends in state $s$, including the emission at frame $t$.
A path starts in the first blank or the first label:

$$
\alpha_1(1) = y^1_{\varnothing}, \qquad \alpha_1(2) = y^1_{l_1}, \qquad \alpha_1(s) = 0 \text{ for } s > 2
$$

Each later frame adds up the allowed predecessors and multiplies by the
emission:

$$
\alpha_t(s) = \Big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \big[\text{skip allowed}\big]\,\alpha_{t-1}(s-2)\Big)\, y^t_{l'_s}
$$

A valid path ends on the last label or the final blank, so
$p(\mathbf{l} \mid x) = \alpha_T(S) + \alpha_T(S-1)$.

Read it aloud: the probability of being in state $s$ at frame $t$ is the
probability of having stayed, advanced from $s - 1$, or skipped from $s - 2$
where that's allowed, times the chance of emitting this state's symbol now.
Equations count frames and states from 1; the code counts from 0.

In log space, shift the previous row by one and by two and combine with
`np.logaddexp`. A boolean mask `skip[s]` marks where the $s - 2$ term is
allowed:

```python
total = prev.copy()
total[1:] = np.logaddexp(total[1:], prev[:-1])
total[2:] = np.where(skip[2:], np.logaddexp(total[2:], prev[:-2]), total[2:])
log_alpha[t] = total + log_probs[t, ext]
```

> [!KEY]
> CTC is the forward algorithm on a left-to-right HMM built from the
> transcript: blanks between labels, and a skip that's forbidden between two
> copies of the same label.

### Work T = 3 by hand

Take a vocabulary of blank (0) and `a` (1). The network gives $P(a)$ of 0.6,
0.3, and 0.8 at the three frames, so the blank probabilities are 0.4, 0.7, and
0.2. The target is `a`, so $\mathbf{l}' = (\varnothing, a, \varnothing)$.

Exactly six paths collapse to `a`. Their probabilities are the following:

| Path | Probability |
|---|---|
| `a__` | $0.6 \times 0.7 \times 0.2 = 0.084$ |
| `_a_` | $0.4 \times 0.3 \times 0.2 = 0.024$ |
| `__a` | $0.4 \times 0.7 \times 0.8 = 0.224$ |
| `aa_` | $0.6 \times 0.3 \times 0.2 = 0.036$ |
| `_aa` | $0.4 \times 0.3 \times 0.8 = 0.096$ |
| `aaa` | $0.6 \times 0.3 \times 0.8 = 0.144$ |

They sum to 0.608. The forward recursion finds the same number in nine cells:

| $t$ | $\alpha_t(\varnothing_1)$ | $\alpha_t(a)$ | $\alpha_t(\varnothing_2)$ |
|---|---|---|---|
| 1 | 0.4 | 0.6 | 0 |
| 2 | $0.4 \times 0.7 = 0.28$ | $(0.6 + 0.4) \times 0.3 = 0.30$ | $(0 + 0.6) \times 0.7 = 0.42$ |
| 3 | $0.28 \times 0.2 = 0.056$ | $(0.30 + 0.28) \times 0.8 = 0.464$ | $(0.42 + 0.30) \times 0.2 = 0.144$ |

So $p(a \mid x) = 0.464 + 0.144 = 0.608$, and the loss is
$-\ln 0.608 = 0.49758$, which is exactly what
`F.ctc_loss(..., reduction="sum")` returns. With every probability at 0.5,
the six paths each have probability $1/8$, so $p = 0.75$.

## Feasibility: how many frames a transcript needs

A path spends at least one frame on each label, and at least one blank frame
between each pair of identical neighbors. So a transcript of $L$ labels with
$r$ adjacent repeats needs

$$
T \ge L + r
$$

Read it aloud: one frame per label, plus one per repeat. It doesn't need
$2L + 1$ frames; that's the number of states, not frames. With fewer frames,
$p = 0$ and the loss is $+\infty$. In the worked example, the target `aa` in
three frames has one path, `a_a`, with probability
$0.6 \times 0.7 \times 0.8 = 0.336$. In two frames it has none.

> [!WARNING]
> $T$ is the number of frames after the subsampling front end. A 4x front end
> turns one second into about 24 frames, so a fast speaker's character
> transcript can become infeasible. torch then returns `inf`, unless you pass
> `zero_infinity=True`.

## The backward recursion

The backward variable $\beta_t(s)$ is the probability of the remaining frames
$t + 1$ to $T$, given state $s$ at frame $t$. In this convention it excludes
frame $t$'s own emission:

$$
\beta_T(S) = \beta_T(S - 1) = 1, \qquad \beta_t(s) = \sum_{s'} \beta_{t+1}(s')\, y^{t+1}_{l'_{s'}}
$$

Read it aloud: at the last frame, the two final states finish for free. At
each earlier frame, add up over the next state $s'$, which is $s$, $s + 1$,
or $s + 2$ when that skip is allowed, the probability of emitting its symbol
at frame $t + 1$ and finishing from there. In the worked example,
$\beta_2 = [0.8, 1.0, 0.2]$ and $\beta_1 = [0.86, 0.44, 0.14]$.

Because $\alpha$ includes frame $t$ and $\beta$ doesn't, their product counts
every emission exactly once, so $\sum_s \alpha_t(s)\beta_t(s) = p(\mathbf{l}
\mid x)$ at every frame. At frame 1: $0.4 \times 0.86 + 0.6 \times 0.44 +
0 \times 0.14 = 0.608$. Graves uses the other convention, where $\beta$
includes the emission, and divides by $y^t_{l'_s}$ instead. Pick one and state
it.

## The gradient

The **state occupancy** $\gamma_t(s) = \alpha_t(s)\beta_t(s) / p(\mathbf{l}
\mid x)$ is the probability that the alignment is in state $s$ at frame $t$.
Add it up over the states that carry symbol $k$, and you get how much frame $t$
belongs to $k$. With $y^t = \operatorname{softmax}(u^t)$ over logits $u^t$:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial u^t_k} = y^t_k - \sum_{s:\ l'_s = k} \gamma_t(s)
$$

Read it aloud: the gradient for symbol $k$ at frame $t$ is what the model
predicts minus how much the alignments use $k$ there. It's the cross-entropy
gradient $y - q$ with a soft target $q$ that the forward-backward pass works
out. Each row sums to zero, because both terms are distributions. In numpy,
`np.add.at` adds each state's occupancy into its symbol's column, even when
several states share a symbol:

```python
gamma = np.exp(log_alpha + log_beta - log_p)          # (T, S)
np.add.at(occupancy, (slice(None), ext), gamma)       # (T, C)
grad = np.exp(log_probs) - occupancy
```

In the worked example, the occupancy of `a` is 0.4342, 0.4934, and 0.7632 at
the three frames. At frame 1, $P(a) = 0.6$, so the gradient for `a` is
$0.6 - 0.4342 = 0.1658$ and the gradient for blank is $-0.1658$. The full
gradient is $[[-0.1658, 0.1658], [0.1934, -0.1934], [-0.0368, 0.0368]]$, the
same as torch's autograd.

> [!KEY]
> The CTC gradient is model posterior minus alignment posterior. The
> forward-backward pass supplies soft frame labels, so CTC trains like frame
> classification without anyone labeling frames.

## Match torch's CTC

torch's loss is `F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
blank=0, reduction="mean")`. It has three traps:

- `log_probs` must have shape `(T, N, C)`, time first, and must already be
  log-softmax outputs. Raw logits, or batch first, run without an error and
  train nothing useful.
- `input_lengths` gives each utterance's frame count after subsampling, and
  `target_lengths` each transcript's length. Padding past them is ignored.
- `reduction="mean"` divides each utterance's loss by its target length, with
  0 counted as 1, and then averages over the batch. To compare with a
  hand-written loss, use `reduction="sum"`.

## Peaky outputs

A trained CTC model predicts blank on almost every frame, and each label as a
spike one or two frames wide. Zeyer and colleagues (2021) showed that the loss
prefers these peaky alignments. The Run block shows it: after 500 training
steps on synthetic tone sequences, the posteriors are near one-hot, and the
argmax path collapses to the right transcript for all 20 held-out sequences.
That argmax, **greedy decoding**, is the simplest decoder. It isn't always the
most probable transcript, which is where the next lesson, *Decoding CTC*,
begins.

> [!RECAP]
> - `ext = [blank, l1, blank, ..., lL, blank]`, and the skip into `ext[s]`
>   needs `ext[s] != blank` and `ext[s] != ext[s - 2]`.
> - Forward: stay, advance, or allowed skip, times the emission; the loss is
>   minus the log-sum-exp of the last two states at the last frame.
> - Backward excludes frame $t$'s emission, so $\sum_s \alpha_t\beta_t = p$
>   at every $t$.
> - Gradient: `softmax(logits)` minus the occupancy, summed per symbol with
>   `np.add.at`.
> - torch wants `(T, N, C)` log-softmax inputs; `mean` divides by target
>   lengths.

## Build it step by step

The checks test the functions in this order. The file provides the tone
generator, the features, the model, and a `torch.autograd.Function` that
trains through your `ctc_grad`.

1. **`extend_with_blanks(target, blank)`.** Returns an int array of length
   `2L + 1`. Fill with blanks and assign `ext[1::2]`. Common mistake: leaving
   off the final blank.
2. **`collapse(path, blank)`.** Walk the path, keep a symbol when it differs
   from the previous one and isn't blank. Common mistake: removing blanks
   first, which merges `a_a` into `a`.
3. **`ctc_forward(log_probs, target, blank)`.** Returns `(T, 2L + 1)`. Start
   from `-np.inf`, set the first two states at frame 0, then loop over frames
   with shifted `np.logaddexp`. Common mistake: allowing the skip into blank
   states.
4. **`ctc_loss(log_probs, target, blank)`.** Minus `logsumexp` of the last two
   entries of the last row. Common mistake: using only the final blank.
5. **`ctc_backward(log_probs, target, blank)`.** Mirror the forward pass from
   the last frame, shifting the other way. Add `log_probs[t + 1, ext]` before
   combining. Common mistake: including frame $t$'s emission, which breaks
   the $\alpha\beta$ identity.
6. **`ctc_grad(logits, target, blank)`.** Take `log_softmax`, run both passes,
   form `gamma`, scatter it with `np.add.at`, and subtract from the softmax.
   Common mistake: `occupancy[:, ext] += gamma`, which drops all but one of
   the states that share a symbol.
7. **`ctc_loss_batch(...)`.** Slice each utterance to its lengths, call
   `ctc_loss`, then reduce. Common mistake: dividing by the input length for
   `"mean"`.

## Cost and scaling

The following table uses $T$ frames, $L$ labels, $C$ symbols, and $N$
utterances.

| Operation | Time | Memory |
|---|---|---|
| Forward or backward | $O(TL)$ | $O(TL)$ |
| Loss only | $O(TL)$ | $O(L)$, one row at a time |
| Gradient | $O(TL + TC)$ | $O(TL + TC)$ |
| A batch | $N$ times one utterance | the largest utterance's table |

The cost depends on the transcript length, not the vocabulary: the lattice has
$2L + 1$ states whatever $C$ is. The network's output layer, $T \times C$,
usually costs more than the loss. The exercise loops over frames in Python;
torch's CUDA kernel runs the states of each frame in parallel.

## Where this breaks

**Conditional independence.** CTC scores each frame alone, so it can't learn
that "their" and "there" depend on the next word. External language models,
fused during decoding, fix much of this.

**Infeasible targets.** Too much subsampling, or a long character transcript,
makes $T < L + r$. The loss is infinite and its gradient is undefined. Filter
such utterances, use subword units, or set `zero_infinity=True`.

**Timing.** Peaky spikes mark where the network became confident, not where
a word begins or ends. For word boundaries, use forced alignment with a
trained model.

**Blank conventions.** torch and Hugging Face's `Wav2Vec2ForCTC` put blank at
index 0; torchaudio's `rnnt_loss` and many NeMo models put it last. Mixing
them trains a model to emit a letter as silence.

**Numerics.** A hand-written log-sum-exp returns `nan` when every input is
$-\infty$. `np.logaddexp` returns $-\infty$, which is the right answer for an
unreachable state.

## The exercise

You build `extend_with_blanks`, `collapse`, `ctc_forward`, `ctc_backward`,
`ctc_loss`, the gradient `ctc_grad`, and `ctc_loss_batch`. Press **Run** to
compare the worked example with torch, train a small convolutional model on
synthetic tone sequences with your `ctc_grad` as its only loss, hear one
sequence, and see its peaky posteriors and greedy decode.

The checks compare your recursions with the T = 3 tables, your loss with a
brute-force sum over every path and with `torch.nn.functional.ctc_loss` on
targets with repeated labels, and confirm the feasibility rule. They check
the $\alpha\beta$ identity at every frame, your gradient against torch's
autograd and against finite differences, and `ctc_loss_batch` against torch's
three reductions on a padded batch. The last check trains free logits with
your gradient until they spell a target with a repeat. Decoding those
outputs well is the subject of *Decoding CTC*.
