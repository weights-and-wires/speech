---
title: Transducers and streaming
slug: 19-transducers-and-streaming
part: "Part 4 — Neural speech recognition"
summary: Build the RNN-T lattice and its loss, greedy transducer decoding, token-and-duration transducers, and the chunked masks that let an encoder stream.
minutes: 75
gpu: false
objectives:
  - Explain how a transducer's encoder, predictor, and joiner combine, and why it can emit several labels on one frame.
  - Compute the RNN-T forward recursion in log space in torch, and check it against brute-force path enumeration.
  - Decode a transducer greedily with a per-frame symbol cap, and a token-and-duration transducer that skips frames.
  - Build a chunked attention mask, and show that a masked encoder's early outputs never depend on future audio.
prereqs:
  - 16-connectionist-temporal-classification
  - 18-attention-encoder-decoders
exercise: 19-transducers-and-streaming
---

# Transducers and streaming

A voice assistant can't wait for you to finish a sentence before it starts
recognizing it. It must write words while the audio still arrives, with a
delay of a fraction of a second. An attention encoder-decoder can't do this,
because its decoder looks at the whole utterance. CTC can stream, but it
treats every frame's output as independent of what it already wrote.

The **transducer**, or **RNN-T** (Graves, 2012), gets both. Like CTC, it walks
through the audio frame by frame and emits a blank when it has nothing to say.
Like an attention decoder, it conditions each label on the labels before it.
On-device recognizers use it, and NVIDIA's Parakeet-TDT models use a faster
variant of it that skips frames.

In this lesson you build the transducer's loss as a dynamic program over a
two-dimensional lattice, decode it greedily, and extend it to the
token-and-duration transducer. Then you build the attention mask that makes an
encoder streamable, and you train a tiny streaming transducer on tones.

## What you already know

- CTC adds a blank and sums over every alignment with a forward recursion, from *Connectionist temporal classification*.
- A CTC model emits at most one label per frame, so it needs at least as many frames as labels.
- Attention masks set some scores to $-\infty$ so their weights become 0, from *Attention encoder-decoders*.
- `torch.logaddexp(a, b)` computes $\log(e^a + e^b)$ and carries gradients.
- The gradient of a log-likelihood with respect to a log probability is how often that probability is used, on average.

## The transducer model

A transducer has three parts. The **encoder** turns audio into frames
$f_1, \dots, f_T$, as in CTC. The **predictor** is a small language model over
the labels emitted so far: after $u$ labels it gives a vector $g_u$. The
**joiner** combines one frame with one predictor state and returns a
distribution over the vocabulary, blank included:

$$
\begin{aligned}
z_{t,u} &= W_{\text{out}} \tanh\big(W_{\text{enc}} f_t + W_{\text{pred}} g_u + b\big) \\
P(k \mid t, u) &= \operatorname{softmax}(z_{t,u})_k
\end{aligned}
$$

Read it aloud: project the frame and the predictor state to the same size,
add them, squash with $\tanh$, and project to one score per symbol. The
softmax of those scores is the probability of symbol $k$ at frame $t$ after
$u$ labels. The model in this lesson's **Run** block does this in one
broadcast:

```python
logits = out(torch.tanh(f[:, :, None, :] + g[:, None, :, :]))   # (B, T, U + 1, V)
log_probs = torch.log_softmax(logits, dim=-1)
```

The predictor doesn't need to be large. Ghodsi et al. (2020) showed that a
**stateless** predictor, which looks only at the last label, works about as
well as an LSTM with subword units. The model in the **Run** block uses one: an
embedding of the previous label.

## The lattice and its forward recursion

Picture a grid with frames on the horizontal axis and labels emitted on the
vertical axis. Node $(t, u)$ means "at frame $t$, with the first $u$ labels
written." From each node the model does one of two things. It emits a blank
and moves right to the next frame. Or it emits the next label $y_{u+1}$ and
moves up, staying on the same frame. A frame can emit several labels in a row,
so a transducer has no rule that $T \ge U$.

Every path starts at $(1, 0)$ and ends with a blank that leaves the last node
$(T, U)$. The **forward variable** $\alpha(t, u)$ is the total probability of
every path prefix that reaches node $(t, u)$. Write $\varnothing(t, u)$ for
$P(\varnothing \mid t, u)$ and $y(t, u)$ for $P(y_{u+1} \mid t, u)$:

$$
\begin{aligned}
\alpha(1, 0) &= 1 \\
\alpha(t, u) &= \alpha(t - 1, u)\,\varnothing(t - 1, u) + \alpha(t, u - 1)\, y(t, u - 1)
\end{aligned}
$$

$$
\begin{aligned}
P(\mathbf{y} \mid x) &= \alpha(T, U)\, \varnothing(T, U) \\
\mathcal{L}_{\text{RNNT}} &= -\log P(\mathbf{y} \mid x)
\end{aligned}
$$

Read it aloud: you reach $(t, u)$ either from the left, by a blank at the
previous frame, or from below, by emitting label $u$ on this frame. Add the
two. The probability of the transcript is the probability of reaching the
last node, times the final blank. Terms outside the grid are zero. In log
space every product is a sum and every sum a `logaddexp`:

```python
alpha[t][u] = torch.logaddexp(alpha[t - 1][u] + blank_lp[:, t - 1, u],
                              alpha[t][u - 1] + emit_lp[:, t, u - 1])
```

The code in this lesson counts $t$ and $u$ from 0, so the start is
`alpha[0][0] = 0`, the log of 1. Unlike CTC, every node of the grid is
reachable, so no entry is ever $-\infty$.

### Work a lattice by hand

Take $T = 2$ frames, the target `a` ($U = 1$), and the vocabulary {blank, `a`}.
The joiner gives these probabilities:

| Node $(t, u)$ | $\varnothing(t, u)$ | $y(t, u)$ |
|---|---|---|
| $(1, 0)$ | 0.4 | 0.6 |
| $(1, 1)$ | 0.7 | none |
| $(2, 0)$ | 0.5 | 0.5 |
| $(2, 1)$ | 0.9 | none |

The recursion fills the grid:

- $\alpha(1, 0) = 1$.
- $\alpha(1, 1) = \alpha(1, 0)\, y(1, 0) = 0.6$.
- $\alpha(2, 0) = \alpha(1, 0)\, \varnothing(1, 0) = 0.4$.
- $\alpha(2, 1) = \alpha(1, 1)\, \varnothing(1, 1) + \alpha(2, 0)\, y(2, 0) = 0.42 + 0.20 = 0.62$.

So $P(a \mid x) = 0.62 \times 0.9 = 0.558$, and the loss is
$-\ln 0.558 = 0.5834$. Check it against the two paths. Emitting `a` at frame
1, then two blanks, gives $0.6 \times 0.7 \times 0.9 = 0.378$. A blank, then
`a` at frame 2, then a blank, gives $0.4 \times 0.5 \times 0.9 = 0.18$. The
sum is $0.558$.

> [!KEY]
> The transducer lattice has frames on one axis and labels on the other. A
> blank moves right, a label moves up, and $\alpha$ adds the two ways into each
> node. The loss is $-\log$ of $\alpha(T, U)$ times the final blank.

### Let autograd do the backward pass

Build $\alpha$ from torch operations and `loss.backward()` gives every
gradient. It also gives something useful to look at. The gradient of
$\log P$ with respect to one edge's log probability is the **posterior** of
that edge: the probability that the true path uses it. Every path takes
exactly $T$ blanks and $U$ labels. So the gradients on all blank entries add
up to $T$, and those on all label entries add up to $U$. One check in this
exercise tests exactly that.

## Greedy decoding with a symbol cap

Greedy decoding follows the lattice. Start the predictor with the blank as a
start symbol. At each frame, ask the joiner for its best symbol. A blank moves
to the next frame. A label is written down, updates the predictor, and the
same frame is asked again:

```python
for t in range(T):
    for _ in range(max_symbols):
        k = int(joiner(enc[t], g).argmax())
        if k == blank:
            break
        out.append(k)
        g, state = predictor(k, state)
```

The cap `max_symbols` matters. An undertrained model can prefer a label on
every step, and without the cap it never leaves the frame. NeMo's greedy
decoder defaults to a cap of 10 per frame.

The transducer has the same blind spot for repeats as an attention decoder. A
stateless predictor that has emitted `5` can't tell whether a second `5` is
still the first tone or a second one. Only the audio can say. In a test on 200
clips, the **Run** block's model got 72 wrong, and every one of them held a
repeated tone.

## Token-and-duration transducers

Most frames emit nothing but a blank, and a standard transducer still runs
the joiner on every one of them. The **token-and-duration transducer** (TDT;
Xu et al., 2023) predicts two things at each node: a symbol, and how many
frames to skip. The two outputs are separate softmaxes over disjoint slices of
the joiner's output, so $P(v, d \mid t, u) = P_T(v \mid t, u)\, P_D(d \mid t, u)$
for durations $d$ in a set $\mathcal{D}$ such as $\{0, 1, 2, 3, 4\}$. The
forward recursion sums over durations:

$$
\alpha(t, u) = \sum_{d \in \mathcal{D},\, d \ge 1} \alpha(t - d, u)\, P(\varnothing, d \mid t - d, u)
+ \sum_{d \in \mathcal{D}} \alpha(t - d, u - 1)\, P(y_u, d \mid t - d, u - 1)
$$

Read it aloud: you reach $(t, u)$ by a blank that jumped $d$ frames from
$(t - d, u)$, or by label $u$ that jumped $d$ frames from $(t - d, u - 1)$. A
blank must move at least one frame; a label can stay put with $d = 0$. The
transcript's probability is $\alpha(T + 1, U)$: the path must land exactly one
past the last frame. A final label can jump there directly, so a TDT path
doesn't need a closing blank.

### Work a TDT lattice by hand

Reuse the two-frame lattice, with the same token probabilities, and add
durations $\{0, 1, 2\}$ with $P_D = (0.2, 0.5, 0.3)$ at every node. The
following six paths reach the end:

| Path | Probability |
|---|---|
| `a` $d{=}0$, blank $d{=}1$, blank $d{=}1$ | $0.6 \cdot 0.2 \times 0.7 \cdot 0.5 \times 0.9 \cdot 0.5 = 0.0189$ |
| `a` $d{=}0$, blank $d{=}2$ | $0.6 \cdot 0.2 \times 0.7 \cdot 0.3 = 0.0252$ |
| `a` $d{=}1$, blank $d{=}1$ | $0.6 \cdot 0.5 \times 0.9 \cdot 0.5 = 0.1350$ |
| `a` $d{=}2$ | $0.6 \cdot 0.3 = 0.1800$ |
| blank $d{=}1$, `a` $d{=}0$, blank $d{=}1$ | $0.4 \cdot 0.5 \times 0.5 \cdot 0.2 \times 0.9 \cdot 0.5 = 0.0090$ |
| blank $d{=}1$, `a` $d{=}1$ | $0.4 \cdot 0.5 \times 0.5 \cdot 0.5 = 0.0500$ |

They sum to $P(a \mid x) = 0.4181$. Some probability goes nowhere: a blank
with $d = 2$ from the start lands at the end with no label written, and a blank
with $d = 0$ is forbidden. The paper accepts this small loss of normalization
because it makes the gradients simpler.

At decoding time, TDT jumps. After each joiner call it moves forward by the
predicted duration, so a model that predicts $d = 4$ on silence calls the
joiner once per four frames. The paper reports inference up to 2.82 times
faster than a standard transducer, with better accuracy. A blank predicted
with $d = 0$ would loop forever, so decoders move such a blank one frame.

> [!TIP]
> Test a TDT implementation with $\mathcal{D} = \{1\}$ first. Every edge then
> moves one frame, so the lattice is small enough to enumerate by hand.

## Streaming: chunked attention masks

A transducer streams only if its encoder does. A unidirectional GRU streams by
construction, because each output depends only on the past; the **Run**
block's model uses one. A self-attention encoder, such as the Conformer in the
next lesson, sees every frame in both directions. To stream it, you limit what
each frame may attend to.

A **chunked attention mask** splits the frames into chunks of $C$ frames.
Frame $i$ may attend to frame $j$ when

$$
\left\lfloor j / C \right\rfloor \le \left\lfloor i / C \right\rfloor
\quad\text{and}\quad
\left\lfloor j / C \right\rfloor \ge \left\lfloor i / C \right\rfloor - N_L
$$

Read it aloud: a frame sees every frame in its own chunk and in earlier
chunks, but not in later ones, and at most $N_L$ chunks back. Inside a chunk,
frames see a little of the future, up to the chunk's end. That's the
**look-ahead**, and it's what the system waits for. In torch:

```python
chunk = torch.arange(T) // chunk_size
allowed = chunk[None, :] <= chunk[:, None]          # (T, T), True where i may see j
```

For $T = 6$ and $C = 2$, frames 0 and 1 see frames 0 and 1. Frames 2 and 3
see frames 0 to 3. Frames 4 and 5 see all six. Pass `~allowed` as the
`attn_mask` of `nn.MultiheadAttention`, where `True` means blocked, and the
outputs for chunk 0 stay the same whatever comes after it.

The chunk size sets the delay. NVIDIA's cache-aware streaming FastConformer
counts context in 80 ms frames; a right context of 13 frames gives a latency
of $(13 + 1) \times 80 = 1{,}120$ ms. Training with random chunk sizes, as
WeNet's dynamic chunk training does, lets one model serve several latencies.

> [!KEY]
> An encoder streams when no output depends on future chunks. The chunk size
> is the look-ahead you pay in latency, and the left-chunk limit bounds the
> memory each step needs.

> [!WARNING]
> PyTorch's boolean `attn_mask` marks the positions to block, the opposite of
> the mask you build here. Invert it, or you block exactly the frames you
> meant to allow.

> [!RECAP]
> - The joiner gives $P(k \mid t, u)$ for every node; a blank moves to the next
>   frame and a label moves up to $u + 1$ on the same frame.
> - $\alpha(t, u)$ is `logaddexp` of the blank from the left and the label from
>   below; the loss is $-(\alpha(T, U) + \log\varnothing(T, U))$.
> - Greedy decoding asks the same frame again after each label, up to
>   `max_symbols` times.
> - TDT sums over durations, forbids a blank with $d = 0$, and ends at
>   $\alpha(T + 1, U)$.
> - A chunked mask allows $j$ for $i$ when $\lfloor j/C \rfloor \le \lfloor i/C \rfloor$.

## Build it step by step

The checks test the functions in this order.

1. **`rnnt_log_alpha(log_probs, targets, blank)`.** Takes `(B, T, U + 1, V)`
   and `(B, U)`, and returns `(B, T, U + 1)`. Gather the label log
   probabilities once with `gather`, then fill a list of lists of `(B,)`
   tensors in order of `t`, then `u`, and `torch.stack` at the end. Common
   mistake: writing into a preallocated tensor in place, which breaks autograd.
2. **`rnnt_loss(log_probs, targets, input_lengths, target_lengths, blank)`.**
   Returns a `(B,)` tensor. Index `alpha` at each utterance's own last node
   and add the final blank. Common mistake: using the padded `T - 1` and `U`,
   which scores padding as if it were audio.
3. **`greedy_transducer_decode(encoder_out, predictor, joiner, blank, max_symbols)`.**
   Returns a list of ints. Loop over frames, with an inner loop that stops at
   a blank or at the cap. Common mistake: moving to the next frame after every
   label, which turns the transducer into CTC.
4. **`tdt_log_likelihood(token_log_probs, duration_log_probs, targets, durations, blank)`.**
   Returns a float. Fill a `(T + 1, U + 1)` array with `-inf`, set the start
   to 0, and collect incoming terms for each node with `np.logaddexp.reduce`.
   Common mistake: letting an edge start at $t = T$, which isn't a real frame.
5. **`tdt_greedy_decode(encoder_out, predictor, joiner, durations, blank, max_symbols)`.**
   Returns `(tokens, calls)`. Use a `while t < T` loop and add the duration to
   `t`. Common mistake: forgetting that a blank with $d = 0$ must still move.
6. **`chunked_attention_mask(T, chunk_size, left_chunks)`.** Returns a `(T, T)`
   bool tensor. Compare chunk indices by broadcasting. Common mistake: using
   `j <= i` instead of chunk indices, which gives a causal mask with no
   look-ahead at all.

## Cost and scaling

The table uses a batch of $B$, $T$ frames, $U$ labels, and $V$ symbols.

| Quantity | How it grows |
|---|---|
| Joiner output for the loss | $B \times T \times (U + 1) \times V$ floats |
| Forward recursion | $O(TU)$ steps per utterance; a Python loop runs $TU$ small operations |
| Greedy decoding | $O(T + U)$ joiner calls |
| TDT greedy decoding | about $T / \bar d + U$ calls, for an average jump of $\bar d$ frames |
| Chunked attention | $O(T \cdot C (N_L + 1))$ scores instead of $O(T^2)$ |

The joiner tensor is the problem. With $B = 16$, $T = 250$ (10 seconds at
40 ms), $U = 80$, and $V = 1{,}024$, it holds $3.3 \times 10^8$ floats: 1.33 GB
in fp32, and as much again for its gradient. Fused kernels compute the log
softmax inside the loss so no second copy exists. Pruned RNN-T (Kuang et al.,
2022) first finds, with a cheap linear joiner, which $u$ matter at each frame,
then runs the full joiner only on a band of 5 values of $u$. That cuts the
tensor to $16 \times 250 \times 5 \times 1{,}024$ floats, or 82 MB.

## Where this breaks

**Training sits on a plateau.** A fresh transducer learns first that blanks
are common and emits nothing for a while. The **Run** block's loss falls to
about 8 within 50 steps, creeps down until about step 100, and only then
drops. Coarser encoder frames
shorten the wait: the model stacks four feature frames into one 40 ms frame,
and in testing it left the plateau about twice as fast as with 20 ms frames.
Starting from a CTC-trained encoder also helps.

**Emissions come late.** A transducer can wait for more audio before emitting,
because waiting costs nothing in the loss. Streaming systems add FastEmit
(Yu et al., 2021), a regularizer that rewards emitting early.

**Blank index conventions disagree.** As with CTC, this course puts the blank
at 0, while `torchaudio.functional.rnnt_loss` defaults to the last index, as
NeMo does.
A mismatch trains without errors and learns nothing useful.

**BatchNorm breaks chunked inference.** Statistics computed over a full
utterance differ from those over one chunk. Many streaming models use
LayerNorm in their convolution modules for this reason.

**Pseudo-streaming isn't streaming.** Rerunning a full-context model on
overlapping windows gives text while audio arrives, but each window waits for
its whole length and repeats work. Measure latency from a word's end to its
emission, not from the chunk size alone.

## The exercise

You build `rnnt_log_alpha` and `rnnt_loss` in torch, then
`greedy_transducer_decode`, `tdt_log_likelihood`, `tdt_greedy_decode`, and
`chunked_attention_mask`. Press **Run** to train a 69,000-parameter streaming
transducer, a causal GRU encoder with a stateless predictor, on clips of two
to six tones with your loss. It then decodes 40 unseen clips with your greedy
decoder, plays one, draws the posterior of each lattice node, and draws a
chunked mask.

The checks confirm the hand-worked lattice and its loss of $-\ln 0.558$, and
match `rnnt_loss` against brute-force enumeration of every path on small
lattices. They confirm a padded batch matches each utterance alone, and that
the gradient adds up to $T$ blanks and $U$ labels. They drive both greedy
decoders with scripted joiners, including one that never emits a blank. They
match `tdt_log_likelihood` against the hand-worked 0.4181 and against brute
force. The last checks compare your mask with the rule, and confirm that
changing future chunks leaves an attention layer's earlier outputs unchanged.
