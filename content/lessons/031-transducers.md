---
title: Transducers
slug: 031-transducers
part: "Part 4 — Neural speech recognition"
summary: Build the RNN-T lattice and its loss, decode it greedily, and extend it to the token-and-duration transducer that skips frames.
minutes: 55
gpu: false
objectives:
  - Explain how a transducer's encoder, predictor, and joiner combine, and why it can emit several labels on one frame.
  - Compute the RNN-T forward recursion in log space in torch, and check it against brute-force path enumeration.
  - Decode a transducer greedily with a per-frame symbol cap.
  - Sum a token-and-duration transducer's lattice over frame jumps, and decode it by skipping frames.
prereqs:
  - 027-ctc-forward-backward
  - 030-attention-encoder-decoders
exercise: 031-transducers
---

# Transducers

A voice assistant cannot wait for you to finish a sentence before it starts to recognize it. It must write words while the audio still arrives, a fraction of a second behind. An attention encoder-decoder cannot do this, because its decoder looks at the whole utterance. CTC can stream, but it treats each frame's output as independent of what it already wrote.

The **transducer**, or **RNN-T** (Graves, 2012), gets both. Like CTC, it walks through the audio frame by frame and emits a blank when it has nothing to say. Like an attention decoder, it conditions each label on the labels before it. On-device recognizers use it, and NVIDIA's Parakeet-TDT models use a faster variant that skips frames. This chapter builds the transducer's loss as a dynamic program on a grid, decodes it greedily, and then teaches it to skip.

## What you already know

- CTC adds a blank and sums over every alignment with a forward recursion, from [Computing CTC efficiently](/lessons/027-ctc-forward-backward).
- A CTC model emits at most one label per frame, so it needs at least as many frames as labels.
- An attention decoder conditions each token on the tokens before it, from [Attention encoder-decoders](/lessons/030-attention-encoder-decoders).
- `torch.logaddexp(a, b)` computes $\log(e^a + e^b)$ and carries gradients.
- The gradient of a log-likelihood with respect to a log probability is how often that probability is used, on average.

## Three parts: encoder, predictor, joiner

The **encoder** turns audio into frames $f_1, \dots, f_T$, as in CTC. The **predictor** is a small language model over the labels written so far: after $u$ labels it gives a vector $g_u$. The **joiner** combines one frame with one predictor state and returns a distribution over the vocabulary, blank included:

$$
z_{t,u} = W_{\text{out}} \tanh\big(W_{\text{enc}} f_t + W_{\text{pred}} g_u + b\big), \qquad
P(k \mid t, u) = \operatorname{softmax}(z_{t,u})_k
$$

Read it aloud: project the frame and the predictor state to the same size, add them, squash with $\tanh$, and project to one score per symbol. The softmax is the probability of symbol $k$ at frame $t$ after $u$ labels. One broadcast scores every pair at once:

```python
logits = out(torch.tanh(f[:, :, None, :] + g[:, None, :, :]))   # (B, T, U + 1, V)
log_probs = torch.log_softmax(logits, dim=-1)
```

The predictor does not need to be large. Ghodsi et al. (2020) found that a **stateless** predictor, which sees only the last label, works about as well as an LSTM with subword units. The model in the **Run** block uses one: an embedding of the previous label.

## A grid of choices

Picture a grid with frames on the horizontal axis and labels written on the vertical axis. Node $(t, u)$ means "at frame $t$, with the first $u$ labels written." From each node the model makes one of two moves. It emits a blank and goes right to the next frame, or it emits the next label $y_{u+1}$ and goes up, staying on the same frame. A frame can emit several labels in a row, so a transducer has no rule that $T \ge U$.

Every path starts at $(1, 0)$ and ends with a blank that leaves the last node $(T, U)$. Many paths write the same transcript, so the probability of the transcript is the sum over all of them. Listing the paths does not scale, because their number grows exponentially, so you add them up as you go. The **forward variable** $\alpha(t, u)$ is the total probability of every path prefix that reaches node $(t, u)$. Write $\varnothing(t, u)$ for $P(\varnothing \mid t, u)$ and $y(t, u)$ for $P(y_{u+1} \mid t, u)$.

Work a tiny grid: $T = 2$ frames, the target `a` ($U = 1$), and the vocabulary {blank, `a`}. The joiner gives $\varnothing(1,0) = 0.4$ and $y(1,0) = 0.6$, then $\varnothing(1,1) = 0.7$, then $\varnothing(2,0) = 0.5$ and $y(2,0) = 0.5$, then $\varnothing(2,1) = 0.9$.

- $\alpha(1, 0) = 1$, the start.
- $\alpha(1, 1) = \alpha(1, 0)\, y(1, 0) = 0.6$, reached by emitting `a` on frame 1.
- $\alpha(2, 0) = \alpha(1, 0)\, \varnothing(1, 0) = 0.4$, reached by a blank.
- $\alpha(2, 1) = \alpha(1, 1)\, \varnothing(1, 1) + \alpha(2, 0)\, y(2, 0) = 0.42 + 0.20 = 0.62$, the two ways in added.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The two-frame, one-label transducer lattice: nodes hold forward values 1, 0.6, 0.4, and 0.62; blanks move right and the label a moves up">
<defs><marker id="c031-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="198" y1="190" x2="360" y2="190" class="stroke-a" marker-end="url(#c031-arrow)"/>
<line x1="198" y1="90" x2="360" y2="90" class="stroke-a" marker-end="url(#c031-arrow)"/>
<line x1="418" y1="90" x2="540" y2="90" class="stroke-a" marker-end="url(#c031-arrow)"/>
<line x1="170" y1="162" x2="170" y2="120" class="stroke-b" marker-end="url(#c031-arrow)"/>
<line x1="390" y1="162" x2="390" y2="120" class="stroke-b" marker-end="url(#c031-arrow)"/>
<line x1="60" y1="190" x2="140" y2="190" class="stroke-muted" marker-end="url(#c031-arrow)"/>
<text x="280" y="182" text-anchor="middle" class="fig-small">blank 0.4</text>
<text x="280" y="82" text-anchor="middle" class="fig-small">blank 0.7</text>
<text x="470" y="82" text-anchor="middle" class="fig-small">blank 0.9</text>
<text x="160" y="144" text-anchor="end" class="fig-small">a 0.6</text>
<text x="400" y="144" class="fig-small">a 0.5</text>
<text x="60" y="182" class="fig-small">start</text>
<circle cx="170" cy="190" r="27" class="fill-soft-d stroke-d"/>
<text x="170" y="195" text-anchor="middle" class="fig-bold">1</text>
<circle cx="170" cy="90" r="27" class="fill-soft-d stroke-d"/>
<text x="170" y="95" text-anchor="middle" class="fig-bold">0.6</text>
<circle cx="390" cy="190" r="27" class="fill-soft-d stroke-d"/>
<text x="390" y="195" text-anchor="middle" class="fig-bold">0.4</text>
<circle cx="390" cy="90" r="27" class="fill-soft-d stroke-d"/>
<text x="390" y="95" text-anchor="middle" class="fig-bold">0.62</text>
<text x="170" y="250" text-anchor="middle" class="fig-small">frame 1</text>
<text x="390" y="250" text-anchor="middle" class="fig-small">frame 2</text>
<text x="20" y="40" class="fig-small">labels emitted u = 1</text>
<text x="20" y="228" class="fig-small">u = 0</text>
</svg>
Figure 1. Each node holds alpha, the probability of every path that reaches it. Orange arrows emit the label a and move up, blue arrows emit a blank and move right. Notice that alpha(2, 1) = 0.6 x 0.7 + 0.4 x 0.5 = 0.62 adds the two ways in, and the final blank 0.9 gives P(a | x) = 0.558.
```

The last step is the final blank, so $P(a \mid x) = 0.62 \times 0.9 = 0.558$ and the loss is $-\ln 0.558 = 0.5834$. Check it against the two paths. Emit `a` on frame 1, then two blanks: $0.6 \times 0.7 \times 0.9 = 0.378$. A blank, then `a` on frame 2, then a blank: $0.4 \times 0.5 \times 0.9 = 0.18$. The sum is 0.558.

As a rule, you reach $(t, u)$ either from the left by a blank, or from below by a label, and you add the two:

$$
\alpha(t, u) = \alpha(t - 1, u)\,\varnothing(t - 1, u) + \alpha(t, u - 1)\, y(t, u - 1), \qquad
P(\mathbf{y} \mid x) = \alpha(T, U)\, \varnothing(T, U)
$$

Terms outside the grid are zero. In log space every product becomes a sum and every sum a `logaddexp`:

```python
alpha[t][u] = torch.logaddexp(alpha[t - 1][u] + blank_lp[:, t - 1, u],
                              alpha[t][u - 1] + emit_lp[:, t, u - 1])
```

The code counts $t$ and $u$ from 0, so the start is `alpha[0][0] = 0`, the log of 1. Unlike CTC, every node of the grid is reachable, so no entry is ever $-\infty$. The loss is $-(\alpha(T, U) + \log\varnothing(T, U))$.

> [!KEY]
> The transducer lattice has frames on one axis and labels on the other. A blank moves right, a label moves up, and $\alpha$ adds the two ways into each node.

### Let autograd do the backward pass

Build $\alpha$ from torch operations and `loss.backward()` gives every gradient. The gradient of $\log P$ with respect to one edge's log probability is the **posterior** of that edge: the probability that the true path uses it. Every path takes exactly $T$ blanks and $U$ labels, so the gradients on all blank entries add up to $T$, and those on all label entries add up to $U$. One check tests exactly that.

## Greedy decoding with a symbol cap

Greedy decoding follows the grid. Start the predictor with the blank as a start symbol. At each frame, ask the joiner for its best symbol. A blank moves to the next frame. A label is written, updates the predictor, and the same frame is asked again:

```python
for t in range(T):
    for _ in range(max_symbols):
        k = int(joiner(enc[t], g).argmax())
        if k == blank:
            break
        out.append(k)
        g, state = predictor(k, state)
```

The cap matters. An undertrained model can prefer a label on every step, and without the cap it never leaves the frame. NeMo's greedy decoder defaults to a cap of 10 per frame. The transducer also shares an attention decoder's blind spot for repeats: a stateless predictor that has written `5` cannot tell whether another `5` is still the first tone or a second one, so only the audio can say. In a test on 200 clips, the **Run** block's model got 72 wrong, and every one held a repeated tone.

## Token-and-duration transducers

Most frames emit only a blank, and a standard transducer still runs the joiner on every one. The **token-and-duration transducer** (TDT; Xu et al., 2023) predicts two things at each node: a symbol, and how many frames to skip. The two outputs are separate softmaxes over disjoint slices of the joiner's output, so $P(v, d \mid t, u) = P_T(v \mid t, u)\, P_D(d \mid t, u)$ for durations $d$ in a set $\mathcal{D}$ such as $\{0, 1, 2, 3, 4\}$. You reach $(t, u)$ by a blank that jumped $d \ge 1$ frames from $(t - d, u)$, or by label $u$ that jumped $d \ge 0$ frames from $(t - d, u - 1)$. A blank must move at least one frame, because a blank with $d = 0$ would loop forever. The transcript's probability is $\alpha(T + 1, U)$, since the path must land exactly one past the last frame, and a final label can jump there directly with no closing blank.

Reuse the two-frame grid with durations $\{0, 1, 2\}$ and $P_D = (0.2, 0.5, 0.3)$ at every node. Six paths write `a`, and Figure 2 draws them. For example, `a` with $d = 0$, then blank with $d = 1$, then blank with $d = 1$ has probability $0.6 \cdot 0.2 \times 0.7 \cdot 0.5 \times 0.9 \cdot 0.5 = 0.0189$, and `a` with $d = 2$ is $0.6 \cdot 0.3 = 0.18$.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Probability of each of the six token-and-duration paths for the transcript a; they sum to 0.4181">
<text x="20" y="22" class="fig-bold">path probability</text>
<text x="20" y="56" class="fig-small">a (d=0), blank (d=1), blank (d=1)</text>
<rect x="270" y="40" width="24.2" height="20" class="fill-b"/>
<text x="300.1" y="55" class="fig-small fig-mono">0.0189</text>
<text x="20" y="86" class="fig-small">a (d=0), blank (d=2)</text>
<rect x="270" y="70" width="32.2" height="20" class="fill-b"/>
<text x="308.2" y="85" class="fig-small fig-mono">0.0252</text>
<text x="20" y="116" class="fig-small">a (d=1), blank (d=1)</text>
<rect x="270" y="100" width="172.5" height="20" class="fill-b"/>
<text x="448.5" y="115" class="fig-small fig-mono">0.1350</text>
<text x="20" y="146" class="fig-small">a (d=2)</text>
<rect x="270" y="130" width="230.0" height="20" class="fill-b"/>
<text x="506.0" y="145" class="fig-small fig-mono">0.1800</text>
<text x="20" y="176" class="fig-small">blank (d=1), a (d=0), blank (d=1)</text>
<rect x="270" y="160" width="11.5" height="20" class="fill-b"/>
<text x="287.5" y="175" class="fig-small fig-mono">0.0090</text>
<text x="20" y="206" class="fig-small">blank (d=1), a (d=1)</text>
<rect x="270" y="190" width="63.9" height="20" class="fill-b"/>
<text x="339.9" y="205" class="fig-small fig-mono">0.0500</text>
<line x1="270" y1="36" x2="270" y2="220" class="stroke-muted"/>
<text x="20" y="244" class="fig-bold">total P(a | x) = 0.4181</text>
</svg>
Figure 2. The six paths that write a and land exactly one frame past the last frame. Notice that the largest path jumps straight to the end, so a joiner that predicts long durations on quiet frames spends very few calls.
```

They sum to $P(a \mid x) = 0.4181$. Some probability goes nowhere: a blank with $d = 2$ from the start lands at the end with no label written, and a blank with $d = 0$ is forbidden. The paper accepts this small loss of normalization because it makes the gradients simpler. At decoding time, TDT jumps: after each joiner call it moves forward by the predicted duration, so a model that predicts $d = 4$ on silence calls the joiner once per four frames. The paper reports inference up to 2.82 times faster than a standard transducer, with better accuracy. A blank predicted with $d = 0$ still moves one frame.

> [!TIP]
> Test a TDT implementation with $\mathcal{D} = \{1\}$ first. Every edge then moves one frame, so the grid is small enough to enumerate by hand.

> [!KEY]
> What you now know: a transducer streams and still conditions on what it wrote, because it walks the frames in order and a predictor remembers the labels. Its loss adds every grid path with a forward recursion, and TDT adds frame jumps so the decoder spends no calls on silence.

> [!RECAP]
> - The joiner gives $P(k \mid t, u)$ for every node; a blank moves to the next frame and a label moves up to $u + 1$ on the same frame.
> - $\alpha(t, u)$ is `logaddexp` of the blank from the left and the label from below; the loss is $-(\alpha(T, U) + \log\varnothing(T, U))$.
> - Greedy decoding asks the same frame again after each label, up to `max_symbols` times.
> - TDT sums over durations, forbids a blank with $d = 0$, and ends at $\alpha(T + 1, U)$.

## Build it step by step

The checks test the functions in this order.

1. **`rnnt_log_alpha(log_probs, targets, blank)`.** Takes `(B, T, U + 1, V)` and `(B, U)`, and returns `(B, T, U + 1)`. Gather the label log probabilities once with `gather`, then fill a list of lists of `(B,)` tensors in order of `t`, then `u`, and `torch.stack` at the end. Common mistake: writing into a preallocated tensor in place, which breaks autograd.
2. **`rnnt_loss(log_probs, targets, input_lengths, target_lengths, blank)`.** Returns a `(B,)` tensor. Index `alpha` at each utterance's own last node and add the final blank. Common mistake: using the padded `T - 1` and `U`, which scores padding as if it were audio.
3. **`greedy_transducer_decode(encoder_out, predictor, joiner, blank, max_symbols)`.** Returns a list of ints. Loop over frames, with an inner loop that stops at a blank or at the cap. Common mistake: moving to the next frame after every label, which turns the transducer into CTC.
4. **`tdt_log_likelihood(token_log_probs, duration_log_probs, targets, durations, blank)`.** Returns a float. Fill a `(T + 1, U + 1)` array with `-inf`, set the start to 0, and collect incoming terms for each node with `np.logaddexp.reduce`. Common mistake: letting an edge start at $t = T$, which is not a real frame.
5. **`tdt_greedy_decode(encoder_out, predictor, joiner, durations, blank, max_symbols)`.** Returns `(tokens, calls)`. Use a `while t < T` loop and add the duration to `t`. Common mistake: forgetting that a blank with $d = 0$ must still move.

## Cost and scaling

The joiner output is a tensor of $B \times T \times (U + 1) \times V$ floats, and it is the memory problem. With $B = 16$, $T = 250$ (10 seconds at 40 ms), $U = 80$, and $V = 1{,}024$, it holds $3.3 \times 10^8$ floats: 1.33 GB in fp32, and as much again for its gradient. Pruned RNN-T (Kuang et al., 2022) first finds, with a cheap linear joiner, which $u$ matter at each frame, then runs the full joiner on a band of 5 values of $u$, cutting the tensor to 82 MB. The forward recursion itself is $O(TU)$ steps per utterance.

## Where this breaks

**Training sits on a plateau.** A fresh transducer learns first that blanks are common and emits nothing for a while. The **Run** block's loss falls to about 8 within 50 steps, creeps down until about step 100, and only then drops. Stacking four feature frames into one 40 ms frame shortens the wait: in testing the model left the plateau about twice as fast as with 20 ms frames.

**Emissions come late.** A transducer can wait for more audio before emitting, because waiting costs nothing in the loss. Streaming systems add FastEmit (Yu et al., 2021), a regularizer that rewards emitting early.

**Blank index conventions disagree.** This course puts the blank at 0, while `torchaudio.functional.rnnt_loss` defaults to the last index, as NeMo does. A mismatch trains without errors and learns nothing useful.

A transducer streams only if its encoder does, which is the next chapter, [Streaming encoders](/lessons/032-streaming-encoders).

## The exercise

You build `rnnt_log_alpha` and `rnnt_loss` in torch, then `greedy_transducer_decode`, `tdt_log_likelihood`, and `tdt_greedy_decode`. Press **Run** to train a 69,000-parameter streaming transducer, a causal GRU encoder with a stateless predictor, on clips of two to six tones with your loss. It then decodes 40 unseen clips with your greedy decoder, plays one, and draws the posterior of each grid node.

The checks confirm the hand-worked grid and its loss of $-\ln 0.558$, and match `rnnt_loss` against brute-force enumeration of every path on small grids. They confirm a padded batch matches each utterance alone, and that the gradient adds up to $T$ blanks and $U$ labels. They drive both greedy decoders with scripted joiners, including one that never emits a blank, and match `tdt_log_likelihood` against the hand-worked 0.4181 and against brute force.
