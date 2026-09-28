---
title: Attention encoder-decoders
slug: 18-attention-encoder-decoders
part: "Part 4 — Neural speech recognition"
summary: Build content and location-aware attention, label smoothing, the joint CTC-attention loss, and length-normalized beam search, then train a tiny LAS.
minutes: 75
gpu: false
objectives:
  - Explain how an encoder-decoder writes one token at a time and uses attention to choose which frames to listen to.
  - Build scaled dot-product attention and location-aware attention with padding masks, in torch.
  - Train with label smoothing and the joint CTC-attention loss, and say what each one prevents.
  - Build beam search with the GNMT length penalty, and show when it flips the winner.
prereqs:
  - 16-connectionist-temporal-classification
  - 17-decoding-ctc
exercise: 18-attention-encoder-decoders
---

# Attention encoder-decoders

CTC makes one strong assumption: given the audio, the symbol in each frame is
independent of the symbols in the other frames. That makes training fast and
decoding simple. It also means the model can't learn that "q" is usually
followed by "u" unless the sound says so. It has no memory of what it already
wrote.

An **attention encoder-decoder** (AED) drops that assumption. An **encoder**
listens to the whole utterance and turns it into a shorter sequence of
vectors. A **decoder** then writes the transcript one token at a time, like a
person taking dictation. Before each token, it looks back over the encoder's
vectors and decides which ones matter right now. That choice is
**attention**. No alignment is given; the model learns where to look.

This is the design of Listen, Attend and Spell (LAS), and of Whisper, which
you meet later in this part. In this lesson you build the attention step in
two forms, the training losses that keep an AED stable, and the beam search
that decodes it. Then you train a tiny LAS on sequences of tones and watch its
attention sweep across the audio.

## What you already know

- CTC sums over monotonic alignments with a blank, from the lesson "Connectionist temporal classification."
- Prefix beam search keeps the best few partial transcripts, from the lesson "Decoding CTC."
- A GRU turns a sequence into a sequence of hidden states, one per step.
- Transformer attention takes a softmax of dot products between a query and many keys.
- Cross-entropy is $-\log p_y$, the negative log probability of the correct class $y$.

## Listen, attend, and spell

The encoder, called the **listener** in LAS, reads the feature frames and
returns a sequence $h_1, \dots, h_T$. Speech has far more frames than
characters, so the listener shortens the sequence. LAS uses a **pyramidal**
BiLSTM: each layer concatenates pairs of adjacent frames before it reads them,
halving the frame rate. Three pyramid layers cut 100 frames per second to
12.5. The model in this lesson has one pyramid step, so its encoder frames are
20 ms apart.

The decoder, called the **speller**, keeps a state $s_i$. At step $i$ it takes
the previous token and the previous **context vector** $c_{i-1}$, updates its
state, attends, and predicts:

$$
s_i = \operatorname{RNN}\big(s_{i-1}, [y_{i-1}; c_{i-1}]\big), \qquad
c_i = \sum_{u} \alpha_{i,u}\, h_u, \qquad
P(y_i \mid x, y_{<i}) = \operatorname{softmax}\big(W_o [s_i; c_i]\big)
$$

Read it aloud: the new state mixes the old state with the last token and the
last context. The context is a weighted average of the encoder frames, with
**attention weights** $\alpha_{i,u}$ that are positive and sum to 1 over $u$.
The next-token distribution comes from the state and the context together.
The brackets $[a; b]$ mean concatenation. The model's decoder step is three
lines:

```python
s = cell(torch.cat([embed(prev_token), c], -1), s)       # (B, d)
c, alpha = attend(s, h, alpha)                           # (B, d), (B, T)
logits = out(torch.cat([s, c], -1))                      # (B, K)
```

Training uses **teacher forcing**: at step $i$ the decoder receives the true
previous token $y^*_{i-1}$, not its own guess. The first input is a start token,
and the last target is an end token, `<eos>`, so the model learns when to
stop. The loss sums the cross-entropy over all $U + 1$ steps:

$$
\mathcal{L}_{\text{att}} = -\sum_{i=1}^{U+1} \log P\big(y^*_i \mid x, y^*_{<i}\big)
$$

In this lesson's model, token 0 does three jobs: the start token, `<eos>`, and
the blank of a CTC head that you add later. They never collide, because the
decoder never emits a blank and CTC never sees `<eos>`.

## Content attention

The simplest way to choose frames is to compare. The decoder's state becomes a
**query** $q$. Each encoder frame offers a **key** $k_u$. Frames whose keys
point the same way as the query get high scores:

$$
e_u = \frac{q \cdot k_u}{\sqrt{d}}, \qquad
\alpha_u = \frac{\exp(e_u)}{\sum_{u'} \exp(e_{u'})}, \qquad
c = \sum_u \alpha_u v_u
$$

Read it aloud: the **energy** $e_u$ of frame $u$ is the dot product of the
query and that frame's key, divided by the square root of their length $d$.
The weights are a softmax of the energies over frames. The context is the
weighted sum of the **values** $v_u$. Dividing by $\sqrt{d}$ keeps the
energies from growing with $d$, which would push the softmax toward one-hot. In
torch, for a batch:

```python
scores = torch.einsum("bd,bud->bu", query, keys) / math.sqrt(d)   # (B, T)
weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
context = torch.einsum("bu,bud->bd", weights, values)             # (B, d_v)
```

The mask matters. A batch pads short utterances with zeros to the length of
the longest one. An energy of $-\infty$ becomes a weight of exactly 0, so
padded frames drop out as if they were never there.

### Work one step by hand

Take $d = 2$, the query $q = (1, 0)$, and three keys $(1, 0)$, $(0, 1)$, and
$(1, 1)$. The dot products are 1, 0, and 1. Divided by $\sqrt 2$ they become
$0.7071$, $0$, and $0.7071$. Their exponentials are $2.0281$, $1$, and
$2.0281$, which sum to $5.0562$. So the weights are $0.4011$, $0.1978$, and
$0.4011$. With scalar values 1, 2, and 3, the context is
$0.4011 + 2 \times 0.1978 + 3 \times 0.4011 = 2.0000$.

## Location-aware attention

Content attention has a weakness in speech. A query that matches the first
"s" in "sits" also matches the second, and content alone doesn't say which
comes next. The decoder needs to know where it looked last.

**Location-aware attention** (Chorowski et al., 2015) feeds the previous step's
weights back in. It convolves $\alpha_{i-1}$ with a few learned filters $F$, so
each frame learns how much attention it and its neighbors received last time:

$$
f_i = F * \alpha_{i-1}, \qquad
e_{i,u} = w^{\top} \tanh\big(W s_{i} + V h_u + U f_{i,u} + b\big), \qquad
\alpha_{i,u} = \operatorname{softmax}_u\, e_{i,u}
$$

Read it aloud: $f_i$ holds one feature vector per frame, the previous weights
seen through $n_f$ short filters. The energy of frame $u$ adds three
projections: the decoder state through $W$, the frame through $V$, and the
location features through $U$, plus a bias $b$. A $\tanh$ and a dot product
with the vector $w$ turn that sum into one number. In torch:

```python
f = F.conv1d(alpha_prev.unsqueeze(1), filters, padding=k // 2).transpose(1, 2)  # (B, T, n_f)
energies = torch.tanh((s @ W.T).unsqueeze(1) + h @ V.T + b + f @ U.T) @ w      # (B, T)
```

The equation writes the state as $s_i$, the state after this step's update,
which is what the model in this lesson passes in. Chorowski's paper uses
$s_{i-1}$; both orders work, as long as training and decoding agree.

### Work a shift by hand

Turn off the content terms by setting $W$ and $V$ to zero. Use one filter,
$F = (5, 0, 0)$, and $U = w = 1$ with $b = 0$. PyTorch's `conv1d` is a
cross-correlation, so with padding 1 this filter computes
$f_u = 5\,\alpha_{i-1}(u - 1)$: each frame sees five times the weight its
left neighbor had.

Suppose the last step attended only to frame 3 of 8. Then $f_4 = 5$ and every
other $f_u = 0$. The energies are $\tanh 5 = 0.9999$ at frame 4 and 0
elsewhere. After the softmax, frame 4 gets
$e^{0.9999} / (e^{0.9999} + 7) = 0.2797$ and each other frame gets $0.1029$.
The peak moved forward one frame, with no content at all. A trained model
learns filters like this, which is what lets attention march steadily through
an utterance. The following demo shows attention weights over a whole
utterance.

```demo
{"id": "attention-alignment", "caption": "Step through the output tokens and watch the attention peak march from left to right across the encoder frames."}
```

> [!KEY]
> Attention turns the decoder's state into a weighted average of encoder
> frames. Location-aware attention adds the previous weights to the energies,
> so the decoder knows where it looked last and moves forward from there.

## Label smoothing

A decoder trained on one-hot targets learns to be certain. Certainty hurts
beam search: when the model gives 0.999 to its first choice, the other
hypotheses in the beam are all near zero, and an LM can't rescue them.
**Label smoothing** moves a little of the target's probability to every class:

$$
q'(k) = (1 - \varepsilon)\,\mathbb{1}[k = y] + \frac{\varepsilon}{K}, \qquad
\ell = -\sum_{k=1}^{K} q'(k) \log p_k
$$

Read it aloud: the smoothed target gives the true class $y$ a weight of
$1 - \varepsilon$, and spreads $\varepsilon$ evenly over all $K$ classes,
including $y$. The loss is cross-entropy against that softer target. It
splits into two terms, $(1 - \varepsilon)(-\log p_y)$ plus
$\varepsilon$ times the mean of $-\log p_k$ over all classes. That's how you
compute it:

```python
per_token = (1 - eps) * (-log_p.gather(-1, y[..., None])[..., 0]) + eps * (-log_p.mean(-1))
```

For example, take $K = 4$, $\varepsilon = 0.1$, $y = 0$, and logits
$(2, 0, 0, 0)$. The softmax gives $p = (0.7112, 0.0963, 0.0963, 0.0963)$, so
$-\log p = (0.3408, 2.3408, 2.3408, 2.3408)$. Plain cross-entropy is
$0.3408$. The smoothed loss is
$0.9 \times 0.3408 + 0.1 \times 1.8408 = 0.4908$, where $1.8408$ is the mean of
the four values. This loss keeps rewarding the model for being right, but it
also charges for putting almost nothing on the other classes. ESPnet's
LibriSpeech recipes use $\varepsilon = 0.1$.

Teacher forcing has a cost of its own, called **exposure bias**. In training
the decoder always sees the true history. At test time it sees its own
guesses, and one early mistake leads it into states it never saw.

## Joint CTC-attention training

Attention is flexible, and that's also its danger. Nothing forces the weights
to move forward. An undertrained or confused model can loop back and repeat a
word, skip ahead, or emit `<eos>` halfway through. CTC can't do any of that,
because its alignments are monotonic by construction.

**Joint CTC-attention training** (Kim, Hori, and Watanabe, 2017) puts a CTC
head on the same encoder and trains both:

$$
\mathcal{L} = \lambda\, \mathcal{L}_{\text{CTC}} + (1 - \lambda)\, \mathcal{L}_{\text{att}}
$$

Read it aloud: the loss is a weighted mix of the CTC loss on the encoder's
output and the attention loss on the decoder's output, with the CTC weight
$\lambda$ between 0 and 1. The CTC term teaches the encoder to keep things in
order, and the attention decoder builds on an encoder that already aligns
well. Both terms here are summed over each utterance and averaged over the
batch, so they're on the same scale. ESPnet's LibriSpeech Conformer recipes
use $\lambda = 0.3$ with label smoothing 0.1.

The attention targets need `<eos>` appended. For the batch of targets
`[[1, 2, 3], [4, 2, pad]]` with lengths 3 and 2, the decoder's targets are
`[[1, 2, 3, eos], [4, 2, eos, pad]]`. The CTC targets stay as they are.

At decoding time, the same CTC head can score the decoder's hypotheses, and
ESPnet adds that CTC prefix score to the beam's ranking. A hypothesis that
loops gets a very low CTC score, so the beam drops it.

> [!KEY]
> The joint loss gives the encoder CTC's monotonic discipline and gives the
> output attention's full dependence on history. A CTC weight of 0.3 is a
> common choice.

## Beam search with a length penalty

Greedy decoding takes the most likely token at each step. **Beam search**
keeps the $B$ most likely unfinished hypotheses instead. At each step it
extends each of them by every token and keeps the best $B$ extensions. An
extension by `<eos>` is finished and moves to a separate list.

Every token multiplies the probability by a number below 1, so raw scores
prefer short hypotheses. That's the same bias the insertion bonus fixed in the
lesson "Decoding CTC." Here the fix is to divide by a function of length. The
GNMT length penalty (Wu et al., 2016) is

$$
\operatorname{lp}(n) = \left(\frac{5 + n}{6}\right)^{\alpha}, \qquad
\operatorname{score}(y) = \frac{\log P(y \mid x)}{\operatorname{lp}(|y|)}
$$

Read it aloud: a hypothesis with $n$ tokens has its log probability divided
by $((5 + n)/6)^\alpha$. With $\alpha = 0$ nothing changes. With $\alpha = 1$
a long hypothesis is divided by a larger number, which makes its negative
score less negative. Whisper uses this penalty when you give it one, and
otherwise divides by the length itself. In Python:

```python
penalty = 1.0 if alpha is None else ((5 + len(tokens)) / 6) ** alpha
ranked = sorted(finished, key=lambda h: h.log_prob / penalty(h), reverse=True)
```

### Work the flip by hand

Suppose the decoder gives `<eos>` a probability of 0.5 at the first step. The
alternative is the token `1` (0.5), then `2` (0.9), then `<eos>` (0.9). The
two finished hypotheses are:

- The empty transcript, with $\log 0.5 = -0.6931$.
- The transcript `1 2`, with $\log(0.5 \times 0.9 \times 0.9) = \log 0.405 = -0.9039$.

Raw scores pick the empty transcript. With $\alpha = 1$, the empty one is
divided by $5/6$ and scores $-0.8318$, while `1 2` is divided by $7/6$ and
scores $-0.7747$. The two-token transcript wins.

> [!WARNING]
> Apply the length penalty only when you rank finished hypotheses. Prune the
> unfinished ones by their raw log probability, or hypotheses of different
> lengths compete on unequal terms mid-search.

> [!RECAP]
> - Content attention is a softmax of $q \cdot k / \sqrt d$ over frames, with
>   $-\infty$ on padded frames.
> - Location-aware attention adds $U f_u$, where $f$ is `conv1d` of the previous
>   weights, inside a $\tanh$, then takes $w^\top$ of it.
> - Label smoothing is $(1 - \varepsilon)(-\log p_y) + \varepsilon \cdot \operatorname{mean}_k(-\log p_k)$.
> - The joint loss is $\lambda \mathcal{L}_{\text{CTC}} + (1 - \lambda)\mathcal{L}_{\text{att}}$,
>   with `<eos>` appended to the attention targets.
> - Beam search prunes by raw log probability and ranks finished hypotheses by
>   $\log P / ((5 + n)/6)^\alpha$.

## Build it step by step

The checks test the functions in this order.

1. **`dot_product_attention(query, keys, values, mask)`.** Takes `(B, d)`,
   `(B, T, d)`, `(B, T, d_v)`, and an optional `(B, T)` bool mask. Returns
   `(context, weights)`. Use `torch.einsum`, `masked_fill`, and
   `torch.softmax(dim=-1)`. Common mistake: masking after the softmax, which
   leaves weights that no longer sum to 1.
2. **`location_aware_attention(s_prev, h, alpha_prev, W, V, b, U, filters, w, mask)`.**
   Returns `(context, weights)`. Use `F.conv1d` with `padding=k // 2`, then
   transpose to `(B, T, n_f)`. Common mistake: forgetting `.unsqueeze(1)` on
   the state term, so it doesn't broadcast across frames.
3. **`label_smoothing_loss(logits, targets, epsilon, pad)`.** Returns a scalar:
   the total over non-pad tokens, divided by the batch size. Use
   `torch.log_softmax`, `gather`, and a mask. Common mistake: spreading
   $\varepsilon$ over $K - 1$ classes, which isn't what PyTorch's
   `label_smoothing` does.
4. **`joint_ctc_attention_loss(ctc_log_probs, ctc_lengths, att_logits, targets, target_lengths, lam, epsilon)`.**
   Build the attention targets with `torch.where` on a position grid, then
   combine `label_smoothing_loss` and `F.ctc_loss(..., reduction="sum",
   zero_infinity=True) / B`. Common mistake: forgetting `<eos>` in the
   attention targets, so the model never learns to stop.
5. **`beam_search(step_fn, eos, beam_width, max_len, length_penalty)`.**
   Returns every finished hypothesis as `(tokens, score)`, best first. Keep an
   `alive` list and a `finished` list, and move the survivors to `finished`
   after the last step. Common mistake: letting finished hypotheses take up
   slots in the beam.

## Cost and scaling

The table uses $T$ encoder frames, $U$ output tokens, model width $d$, and a
beam of $B$.

| Part | Work | Memory |
|---|---|---|
| One attention step | $O(Td)$ | $O(T)$ weights |
| All decoder steps | $O(UTd)$, and the $U$ steps must run in order | $O(UT)$ weights for training |
| Location features | $O(T n_f k)$ per step | $O(T n_f)$ |
| Beam search | $B$ times greedy, plus sorting $BK$ candidates per step | $B$ decoder states |

Whisper's encoder turns 30 seconds of audio into 1,500 frames, 20 ms apart. A
decoder that writes 200 tokens then computes 300,000 attention weights per
head and layer. The encoder runs once, but the decoder runs once per token,
so for long outputs the decoder dominates the time.

## Where this breaks

**It can't stream.** The decoder attends over the whole utterance, so it can't
start until the audio ends. The lesson "Transducers and streaming" covers
models that emit while audio arrives.

**It loops, skips, or stops early.** Attention has no built-in order.
Location-aware energies, the joint CTC loss, and CTC scores in the beam all
push it back toward monotonic alignments.

**Long utterances drift.** An AED trained on 15-second clips often fails on a
minute of audio, because attention never learned to travel that far. Systems
cut long audio into chunks; Whisper works in 30-second windows.

**Exposure bias compounds.** One wrong token early changes every later
decoder state. Scheduled sampling (Bengio et al., 2015) sometimes feeds the
model its own predictions during training, with a probability that grows over
time.

**It can make up text.** A decoder is also a language model, so over silence
or noise it can write fluent words that nobody said. Large AEDs such as
Whisper do this, and the fixes are voice activity detection before decoding
and checks on the decoder's confidence.

## The exercise

You build `dot_product_attention` and `location_aware_attention`, then
`label_smoothing_loss`, `joint_ctc_attention_loss`, and `beam_search`. The
file gives you a `TinyLAS` model, a pyramidal BiGRU listener and a GRU speller
with a CTC head, that calls your attention and your loss. Press **Run** to
train it for 250 steps on clips of two to six tones, decode 40 new clips with
your beam search, hear one of them, and see its attention map. The reference
solution reaches a token error rate of 2.5% on 200 new clips. Every clip it
got wrong held a repeated tone, dropped or doubled: with no symbol between two
copies, the decoder must count them from the audio alone.

The checks compare both attention functions against direct computations,
including a frame-by-frame loop and the shift filter from the lesson. They
match your smoothed loss against `F.cross_entropy` with `label_smoothing`,
your joint loss against its two parts, and your beam search against brute
force over every hypothesis of up to four tokens. One check confirms the length
penalty flips the hand-worked example. The last one trains `TinyLAS` for 60
steps with your code and requires the loss to fall below 9.
