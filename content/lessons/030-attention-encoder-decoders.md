---
title: Attention encoder-decoders
slug: 030-attention-encoder-decoders
part: "Part 4 — Neural speech recognition"
summary: Let a decoder write one token at a time and choose which frames to listen to, then keep it honest with a joint CTC loss and beam search.
minutes: 60
gpu: false
objectives:
  - Explain how an encoder-decoder writes one token at a time and uses attention to choose which frames to listen to.
  - Build scaled dot-product attention and location-aware attention with padding masks, in torch.
  - Train with label smoothing and the joint CTC-attention loss, and say what each one prevents.
  - Build beam search with the GNMT length penalty, and show when it flips the winner.
prereqs:
  - 027-ctc-forward-backward
  - 028-decoding-ctc
exercise: 030-attention-encoder-decoders
---

# Attention encoder-decoders

A CTC recognizer hears the word "queen" and has to write it. At every frame it picks a letter, and it picks each one on its own, without looking at the letters it already wrote. So it has no way to know that a "q" is nearly always followed by a "u" unless the sound says so. It has no memory of its own output.

An **attention encoder-decoder** (AED) fixes this by writing the transcript the way a person takes dictation. An **encoder** listens to the whole utterance and turns it into a shorter sequence of vectors, one per frame. A **decoder** then writes one token at a time, and before each token it looks back over the encoder's vectors and decides which ones matter right now. That decision is **attention**. Nobody tells the model where each letter sits in the audio, so it has to learn where to look.

This is the design of Listen, Attend and Spell (LAS), and of Whisper, which you meet later in this part. This chapter builds the attention step, the two tricks that keep a decoder stable, and the beam search that reads out its answer.

## What you already know

- CTC sums over alignments with a blank and treats frames as independent, from [Computing CTC efficiently](/lessons/027-ctc-forward-backward).
- Prefix beam search keeps the best few partial transcripts, from [Decoding CTC](/lessons/028-decoding-ctc).
- A GRU turns a sequence into a sequence of hidden states, one per step.
- Transformer attention takes a softmax of dot products between a query and many keys.
- Cross-entropy is $-\log p_y$, the negative log probability of the correct class $y$.

## Why not copy one frame per token?

The obvious plan is to hand the decoder the one frame that lines up with each token. It fails because you do not know which frame that is: a word that takes 0.4 s in one recording takes 0.9 s in another. So the decoder takes a *weighted average* of all the frames, with weights that add up to 1. Heavy weight on a few frames approximates a choice, and because an average is smooth, gradient descent can nudge the weights. That average is the **context vector** $c_i$.

## Attention is a weighted average

The decoder's state becomes a **query** $q$. Each encoder frame $u$ offers a **key** $k_u$ to match against and a **value** $v_u$ to hand over. A frame whose key points the same way as the query scores high. Take $d = 2$, the query $q = (1, 0)$, three keys $(1, 0)$, $(0, 1)$, and $(1, 1)$, and scalar values 1, 2, and 3.

1. The dot products are 1, 0, and 1. Divide by $\sqrt 2$ to get 0.7071, 0, and 0.7071.
2. Their exponentials are 2.0281, 1, and 2.0281, which sum to 5.0562.
3. So the weights are 0.4011, 0.1978, and 0.4011, and the context is $0.4011 + 2 \times 0.1978 + 3 \times 0.4011 = 2.0000$.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Three encoder frames with keys and values, the attention weight each one gets from a query, and the context vector they blend into">
<text x="20" y="24" class="fig-bold">decoder query q = (1, 0)</text>
<rect x="40" y="44" width="130" height="52" class="fill-soft-a stroke-a"/>
<text x="105" y="64" text-anchor="middle" class="fig-bold">frame 1</text>
<text x="105" y="84" text-anchor="middle" class="fig-small">key (1, 0), value 1</text>
<rect x="75" y="133.8" width="60" height="80.2" class="fill-b"/>
<text x="105" y="127.8" text-anchor="middle" class="fig-small fig-bold">0.4011</text>
<text x="105" y="234" text-anchor="middle" class="fig-small">score 0.7071</text>
<rect x="200" y="44" width="130" height="52" class="fill-soft-a stroke-a"/>
<text x="265" y="64" text-anchor="middle" class="fig-bold">frame 2</text>
<text x="265" y="84" text-anchor="middle" class="fig-small">key (0, 1), value 2</text>
<rect x="235" y="174.4" width="60" height="39.6" class="fill-b"/>
<text x="265" y="168.4" text-anchor="middle" class="fig-small fig-bold">0.1978</text>
<text x="265" y="234" text-anchor="middle" class="fig-small">score 0.0000</text>
<rect x="360" y="44" width="130" height="52" class="fill-soft-a stroke-a"/>
<text x="425" y="64" text-anchor="middle" class="fig-bold">frame 3</text>
<text x="425" y="84" text-anchor="middle" class="fig-small">key (1, 1), value 3</text>
<rect x="395" y="133.8" width="60" height="80.2" class="fill-b"/>
<text x="425" y="127.8" text-anchor="middle" class="fig-small fig-bold">0.4011</text>
<text x="425" y="234" text-anchor="middle" class="fig-small">score 0.7071</text>
<line x1="30" y1="214" x2="500" y2="214" class="stroke-muted"/>
<text x="520" y="150" class="fig-bold">context</text>
<text x="520" y="170" class="fig-small">0.4011 x 1</text>
<text x="520" y="186" class="fig-small">+ 0.1978 x 2</text>
<text x="520" y="202" class="fig-small">+ 0.4011 x 3</text>
<text x="520" y="222" class="fig-bold">= 2.0000</text>
<text x="20" y="258" class="fig-small">orange bars: attention weights (softmax of the scores)</text>
</svg>
Figure 1. Frames 1 and 3 point the same way as the query, so each gets 0.4011 of the weight and frame 2 gets 0.1978. Notice that the context, 2.0000, is a blend of the three values, not a pick of one.
```

Written as formulas, with $d$ the length of the key:

$$
\begin{aligned}
e_u &= \frac{q \cdot k_u}{\sqrt{d}} \\
\alpha_u &= \frac{\exp(e_u)}{\sum_{u'} \exp(e_{u'})} \\
c &= \sum_u \alpha_u v_u
\end{aligned}
$$

Read it aloud: the **energy** $e_u$ is the dot product of the query and key, divided by $\sqrt d$. The **attention weights** $\alpha_u$ are a softmax over frames. The context is the weighted sum of values. Dividing by $\sqrt d$ stops the energies from growing with $d$, which would push the softmax toward one-hot. For a batch:

```python
scores = torch.einsum("bd,bud->bu", query, keys) / math.sqrt(d)   # (B, T)
weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
context = torch.einsum("bu,bud->bd", weights, values)             # (B, d_v)
```

The mask matters. A batch pads short utterances with zeros to the length of the longest one, and an energy of $-\infty$ becomes a weight of exactly 0, so padded frames drop out as if they were never there.

At each step the decoder updates its state from the previous token and context, attends, and predicts from the state and the new context together: $P(y_i \mid x, y_{<i}) = \operatorname{softmax}\big(W_o [s_i; c_i]\big)$, where $[a; b]$ is concatenation. Training uses **teacher forcing**: the decoder receives the true previous token, not its own guess. The first input is a start token and the last target is `<eos>`, so the model learns when to stop. Here token 0 is the start token, `<eos>`, and the CTC blank, which never collide, because the decoder never emits a blank and CTC never sees `<eos>`.

## Content alone gets lost

Content attention has a weakness in speech. A query that matches the first "s" in "sits" also matches the second, and the content does not say which one comes next. The decoder needs to know where it looked last.

**Location-aware attention** (Chorowski et al., 2015) feeds the previous weights back in. It convolves $\alpha_{i-1}$ with $n_f$ short learned filters $F$, so each frame learns how much attention it and its neighbors received last time:

$$
f_i = F * \alpha_{i-1}, \qquad
e_{i,u} = w^{\top} \tanh\big(W s_i + V h_u + U f_{i,u} + b\big)
$$

Read it aloud: the energy of frame $u$ adds three projections, the decoder state through $W$, the frame through $V$, and the location features through $U$, plus a bias $b$. A $\tanh$ and a dot product with $w$ turn the sum into one number, and a softmax over $u$ gives the weights as before. In torch:

```python
f = F.conv1d(alpha_prev.unsqueeze(1), filters, padding=k // 2).transpose(1, 2)  # (B, T, n_f)
energies = torch.tanh((s @ W.T).unsqueeze(1) + h @ V.T + b + f @ U.T) @ w      # (B, T)
```

To see location do something with no content at all, set $W$ and $V$ to zero, use one filter $F = (5, 0, 0)$, and set $U = w = 1$ and $b = 0$. PyTorch's `conv1d` is a cross-correlation, so with padding 1 this filter computes $f_u = 5\,\alpha_{i-1}(u - 1)$: each frame sees five times the weight its left neighbor had. If the last step attended only to frame 3 of 8, then $f_4 = 5$ and every other $f_u = 0$. The energies are $\tanh 5 = 0.9999$ at frame 4 and 0 elsewhere, so the softmax gives frame 4 the weight $e^{0.9999} / (e^{0.9999} + 7) = 0.2797$ and every other frame 0.1029.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Previous attention peaked on frame 3 of 8; with a shift filter and no content, the new attention peaks on frame 4">
<text x="20" y="16" class="fig-bold">last step: attention weights</text>
<text x="20" y="150" class="fig-bold">this step: shift filter, no content</text>
<rect x="60" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="78" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="78" y="256" text-anchor="middle" class="fig-small">1</text>
<rect x="120" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="138" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="138" y="256" text-anchor="middle" class="fig-small">2</text>
<rect x="180" y="40.0" width="36" height="60.0" class="fill-a"/>
<text x="198" y="34.0" text-anchor="middle" class="fig-small">1.0000</text>
<rect x="180" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="198" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="198" y="256" text-anchor="middle" class="fig-small">3</text>
<rect x="240" y="168.5" width="36" height="61.5" class="fill-b"/>
<text x="258" y="162.5" text-anchor="middle" class="fig-small">0.2797</text>
<text x="258" y="256" text-anchor="middle" class="fig-small">4</text>
<rect x="300" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="318" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="318" y="256" text-anchor="middle" class="fig-small">5</text>
<rect x="360" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="378" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="378" y="256" text-anchor="middle" class="fig-small">6</text>
<rect x="420" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="438" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="438" y="256" text-anchor="middle" class="fig-small">7</text>
<rect x="480" y="207.4" width="36" height="22.6" class="fill-b"/>
<text x="498" y="201.4" text-anchor="middle" class="fig-small">0.1029</text>
<text x="498" y="256" text-anchor="middle" class="fig-small">8</text>
<line x1="50" y1="100" x2="540" y2="100" class="stroke-muted"/>
<line x1="50" y1="230" x2="540" y2="230" class="stroke-muted"/>
<text x="550" y="256" class="fig-small">frame</text>
</svg>
Figure 2. The previous step put all its weight on frame 3. A filter that adds five times the left neighbor's weight lifts frame 4 to 0.2797 while every other frame stays at 0.1029. Notice that the peak moved forward with no help from the audio.
```

A trained model learns filters like this, which is what lets attention march steadily through an utterance. The following demo shows the weights over a whole utterance.

```demo
{"id": "attention-alignment", "caption": "Pick the monotonic alignment and drag the output character to watch the attention peak march left to right across the encoder frames, then pick the repeating one to see it jump back.", "props": {"task": "asr"}}
```

> [!KEY]
> Attention turns the decoder's state into a weighted average of encoder frames, so the decoder can listen somewhere different for every token. Location-aware attention adds the previous weights to the energies, so it knows where it looked last and moves forward from there.

## Label smoothing

A decoder trained on one-hot targets learns to be certain. Certainty hurts beam search, because every other hypothesis in the beam is then near zero. **Label smoothing** gives the true class $y$ the weight $1 - \varepsilon$ and spreads $\varepsilon$ evenly over all $K$ classes, including $y$. The loss splits into two terms:

$$
\ell = (1 - \varepsilon)\,(-\log p_y) + \varepsilon \cdot \frac{1}{K}\sum_{k=1}^{K} (-\log p_k)
$$

Take $K = 4$, $\varepsilon = 0.1$, $y = 0$, and logits $(2, 0, 0, 0)$. The softmax gives $p = (0.7112, 0.0963, 0.0963, 0.0963)$, so $-\log p = (0.3408, 2.3408, 2.3408, 2.3408)$. Plain cross-entropy is 0.3408. The smoothed loss is $0.9 \times 0.3408 + 0.1 \times 1.8408 = 0.4908$, where 1.8408 is the mean of the four values. ESPnet's LibriSpeech recipes use $\varepsilon = 0.1$.

## Joint CTC-attention training

Nothing forces attention to move forward, so a confused model can loop back and repeat a word, skip ahead, or emit `<eos>` halfway through. CTC cannot, because its alignments are monotonic by construction.

**Joint CTC-attention training** (Kim, Hori, and Watanabe, 2017) puts a CTC head on the same encoder and trains both, with a CTC weight $\lambda$ between 0 and 1:

$$
\mathcal{L} = \lambda\, \mathcal{L}_{\text{CTC}} + (1 - \lambda)\, \mathcal{L}_{\text{att}}
$$

The CTC term teaches the encoder to keep frames in order. Both terms are summed over each utterance and averaged over the batch, so they sit on the same scale. ESPnet's LibriSpeech Conformer recipes use $\lambda = 0.3$. The attention targets need `<eos>` appended: `[[1, 2, 3], [4, 2, pad]]` with lengths 3 and 2 becomes `[[1, 2, 3, eos], [4, 2, eos, pad]]`, while the CTC targets stay as they are. At decoding time ESPnet also adds the CTC prefix score to the beam, so a looping hypothesis drops out.

## Beam search with a length penalty

Greedy decoding takes the most likely token at each step. **Beam search** keeps the $B$ most likely unfinished hypotheses: each step extends every one by every token and keeps the best $B$. An extension by `<eos>` is finished and moves to a separate list. Every token multiplies the probability by a number below 1, so raw scores prefer short hypotheses, the bias the insertion bonus fixed in [Decoding CTC](/lessons/028-decoding-ctc). The GNMT length penalty (Wu et al., 2016) divides the log probability by a function of the length $n$:

$$
\operatorname{lp}(n) = \left(\frac{5 + n}{6}\right)^{\alpha}, \qquad
\operatorname{score}(y) = \frac{\log P(y \mid x)}{\operatorname{lp}(|y|)}
$$

With $\alpha = 0$ nothing changes. With $\alpha = 1$ a longer hypothesis is divided by a larger number, which makes its negative score less negative. Whisper divides by the length itself unless you give it this penalty.

Work a flip by hand. Suppose the decoder gives `<eos>` probability 0.5 at the first step. The alternative is the token `1` (0.5), then `2` (0.9), then `<eos>` (0.9). The empty transcript scores $\log 0.5 = -0.6931$, and `1 2` scores $\log(0.5 \times 0.9 \times 0.9) = \log 0.405 = -0.9039$, so raw scores pick the empty one. With $\alpha = 1$ the empty one is divided by $5/6$ and scores $-0.8318$, while `1 2` is divided by $7/6$ and scores $-0.7747$. The two-token transcript wins.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Scores of the empty transcript and of the transcript 1 2, before and after the length penalty; the winner flips">
<text x="20" y="22" class="fig-bold">log probability of each finished hypothesis (higher is better)</text>
<line x1="50" y1="60" x2="290" y2="60" class="stroke-muted"/>
<text x="60" y="252" class="fig-bold">raw score</text>
<rect x="70" y="60" width="80" height="117.8" class="fill-b"/>
<text x="110" y="193.8" text-anchor="middle" class="fig-small fig-bold">-0.6931</text>
<text x="110" y="52" text-anchor="middle" class="fig-small">empty (n = 0)</text>
<rect x="180" y="60" width="80" height="153.7" class="fill-soft-a"/>
<text x="220" y="229.7" text-anchor="middle" class="fig-small fig-bold">-0.9039</text>
<text x="220" y="52" text-anchor="middle" class="fig-small">1 2 (n = 2)</text>
<line x1="330" y1="60" x2="570" y2="60" class="stroke-muted"/>
<text x="340" y="252" class="fig-bold">divided by lp(n), alpha = 1</text>
<rect x="350" y="60" width="80" height="141.4" class="fill-soft-a"/>
<text x="390" y="217.4" text-anchor="middle" class="fig-small fig-bold">-0.8318</text>
<text x="390" y="52" text-anchor="middle" class="fig-small">empty (n = 0)</text>
<rect x="460" y="60" width="80" height="131.7" class="fill-b"/>
<text x="500" y="207.7" text-anchor="middle" class="fig-small fig-bold">-0.7747</text>
<text x="500" y="52" text-anchor="middle" class="fig-small">1 2 (n = 2)</text>
</svg>
Figure 3. Before the penalty, the empty transcript wins with -0.6931 against -0.9039. Dividing by (5 + n) / 6 gives -0.8318 and -0.7747, so the two-token transcript wins. The orange bar marks the winner in each group.
```

> [!WARNING]
> Apply the length penalty only when you rank finished hypotheses. Prune the unfinished ones by their raw log probability, or hypotheses of different lengths compete on unequal terms in the middle of the search.

> [!KEY]
> What you now know: a CTC recognizer cannot learn that "q" is followed by "u", but an attention decoder can, because it conditions each token on the tokens before it and chooses which frames to hear by a weighted average. Location-aware attention, label smoothing, the CTC term, and a length-penalized beam keep that freedom from going wrong.

> [!RECAP]
> - Content attention is a softmax of $q \cdot k / \sqrt d$ over frames, with $-\infty$ on padded frames.
> - Location-aware attention adds $U f_u$, where $f$ is `conv1d` of the previous weights, inside a $\tanh$, then takes $w^\top$ of it.
> - Label smoothing is $(1 - \varepsilon)(-\log p_y) + \varepsilon \cdot \operatorname{mean}_k(-\log p_k)$.
> - The joint loss is $\lambda \mathcal{L}_{\text{CTC}} + (1 - \lambda)\mathcal{L}_{\text{att}}$, with `<eos>` appended to the attention targets.
> - Beam search prunes by raw log probability and ranks finished hypotheses by $\log P / ((5 + n)/6)^\alpha$.

## Build it step by step

The checks test the functions in this order.

1. **`dot_product_attention(query, keys, values, mask)`.** Takes `(B, d)`, `(B, T, d)`, `(B, T, d_v)`, and an optional `(B, T)` bool mask, and returns `(context, weights)`. Use `einsum`, `masked_fill`, and `softmax`. Common mistake: masking after the softmax, which leaves weights that no longer sum to 1.
2. **`location_aware_attention(s_prev, h, alpha_prev, W, V, b, U, filters, w, mask)`.** Returns `(context, weights)`. Use `F.conv1d` with `padding=k // 2`, then transpose to `(B, T, n_f)`. Common mistake: forgetting `.unsqueeze(1)` on the state term, so it does not broadcast across frames.
3. **`label_smoothing_loss(logits, targets, epsilon, pad)`.** Returns the total over non-pad tokens, divided by the batch size. Common mistake: spreading $\varepsilon$ over $K - 1$ classes.
4. **`joint_ctc_attention_loss(ctc_log_probs, ctc_lengths, att_logits, targets, target_lengths, lam, epsilon)`.** Write `<eos>` at each utterance's own length, then mix your smoothing loss with torch's CTC loss. Common mistake: forgetting `<eos>`, so the model never learns to stop.
5. **`beam_search(step_fn, eos, beam_width, max_len, length_penalty)`.** Returns every finished hypothesis as `(tokens, score)`, best first. Keep an `alive` list and a `finished` list. Common mistake: letting finished hypotheses take up slots in the beam.

## Where this breaks

**It cannot stream.** The decoder attends over the whole utterance, so it cannot start until the audio ends. [Transducers](/lessons/031-transducers) emit while audio arrives.

**It loops, skips, or stops early.** Location-aware energies, the joint CTC loss, and CTC scores in the beam all push it back toward monotonic alignments.

**Long utterances drift.** An AED trained on 15-second clips often fails on a minute of audio, which is why Whisper works in 30-second windows of 1,500 encoder frames, 20 ms apart.

**Exposure bias compounds.** In training the decoder sees the true history, at test time its own guesses, so one early mistake leads it into states it never saw.

**It can make up text.** A decoder is also a language model, so over silence it can write fluent words that nobody said. Voice activity detection before decoding is the usual fix.

## The exercise

You build `dot_product_attention`, `location_aware_attention`, `label_smoothing_loss`, `joint_ctc_attention_loss`, and `beam_search`. The file gives you a `TinyLAS` model, a pyramidal BiGRU listener and a GRU speller with a CTC head, that calls your attention and your loss. Press **Run** to train it for 250 steps on clips of two to six tones, decode 40 unseen clips with your beam search, hear one, and see its attention map. The reference solution reaches a token error rate of 2.5% on 200 unseen clips, and every clip it got wrong held a repeated tone: with no symbol between two copies, the decoder must count them from the audio alone.

The checks compare both attention functions against a frame-by-frame loop and the shift filter, your smoothed loss against `F.cross_entropy`, your joint loss against its two parts, and your beam search against brute force over every hypothesis of up to four tokens. One check confirms the length penalty flips the hand-worked example. The last trains `TinyLAS` for 60 steps with your code and requires the loss to fall below 9.
