---
title: Computing CTC efficiently
slug: 027-ctc-forward-backward
part: "Part 4 — Neural speech recognition"
summary: Add up every CTC path in time proportional to frames times labels, with forward and backward passes that also give the gradient.
minutes: 45
gpu: false
objectives:
  - Build the extended label sequence and the stay, advance, and skip moves of the CTC lattice.
  - Compute the forward recursion in log space, and read the loss off its last frame.
  - Compute the backward recursion, and check that alpha times beta sums to the same total at every frame.
  - Derive the gradient with respect to the logits as model posterior minus alignment posterior, and match torch's ctc_loss.
prereqs:
  - 021-hidden-markov-models
  - 022-training-hmms
  - 026-ctc-many-paths-one-label
exercise: 027-ctc-forward-backward
---

# Computing CTC efficiently

The previous chapter ended with a loss you could write down and could not run. A 20-letter transcript in 150 frames has about $10^{39}$ paths that spell it, and the loss adds up every one. A training run needs that sum for thousands of utterances, and it needs the gradient too.

The way out is to notice that the paths are not independent. Two paths that agree on their first 100 frames share all the work for those frames. If you keep one running total for "all the ways to be at this point in the transcript at this frame", you never enumerate a path at all. This is the forward algorithm of a hidden Markov model, and CTC is one: a small HMM built from the transcript.

This chapter builds that HMM, runs the sum in $O(TL)$ time, and then reuses the same machinery to get the gradient. At the end, your code matches torch's `ctc_loss` to the last digit.

## What you already know

- The forward algorithm sums over every path through a trellis by keeping a running total per state, with log-sum-exp to avoid underflow, as [Hidden Markov models](/lessons/021-hidden-markov-models) showed.
- The backward algorithm runs the same sum from the end of the utterance, as [Training HMMs with Baum-Welch](/lessons/022-training-hmms) showed.
- CTC's probability for a transcript is a sum over paths of per-frame products, and a path needs at least $L + r$ frames, from [CTC: many paths, one label](/lessons/026-ctc-many-paths-one-label).
- For softmax outputs $y = \operatorname{softmax}(u)$ and a cross-entropy loss with target distribution $q$, the gradient with respect to the logits $u$ is $y - q$.

## A small HMM built from the transcript

Every path that spells the transcript visits its labels in order, with blanks around them. So build one state for each label and one for each blank slot. For $\mathbf{l} = (l_1, \dots, l_L)$, the **extended sequence** puts a blank before, between, and after the labels:

$$
\mathbf{l}' = (\varnothing,\ l_1,\ \varnothing,\ l_2,\ \dots,\ l_L,\ \varnothing), \qquad S = |\mathbf{l}'| = 2L + 1
$$

Read it aloud: the labels sit at the odd positions and the blanks at the even ones. In numpy, `ext = np.full(2 * L + 1, blank)` and then `ext[1::2] = target`. The number of states is $S$, which depends on the transcript and never on the alphabet size.

A path walks through these states from left to right, one state per frame. From state $s$ it can **stay** at $s$, which is the same symbol repeated, **advance** to $s + 1$, or **skip** to $s + 2$. A skip jumps over a blank, so it is allowed only when the state it lands on is a label that differs from the one it left. A skip onto a blank would jump over a label. A skip between two copies of the same label would let the path merge them under $\mathcal{B}$, so the path must pass through the blank between them.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A lattice of five states, blank a blank b blank, over four frames, with stay, advance, and one skip edge from state a to state b">
  <line x1="150" y1="50" x2="260" y2="50" class="stroke-muted"/>
  <line x1="150" y1="50" x2="260" y2="100" class="stroke-muted"/>
  <line x1="150" y1="100" x2="260" y2="100" class="stroke-muted"/>
  <line x1="150" y1="100" x2="260" y2="150" class="stroke-muted"/>
  <line x1="150" y1="100" x2="260" y2="200" class="stroke-b"/>
  <line x1="260" y1="50" x2="370" y2="100" class="stroke-muted"/>
  <line x1="260" y1="100" x2="370" y2="100" class="stroke-muted"/>
  <line x1="260" y1="100" x2="370" y2="150" class="stroke-muted"/>
  <line x1="260" y1="100" x2="370" y2="200" class="stroke-b"/>
  <line x1="260" y1="150" x2="370" y2="150" class="stroke-muted"/>
  <line x1="260" y1="150" x2="370" y2="200" class="stroke-muted"/>
  <line x1="260" y1="200" x2="370" y2="200" class="stroke-muted"/>
  <line x1="260" y1="200" x2="370" y2="250" class="stroke-muted"/>
  <line x1="370" y1="100" x2="480" y2="200" class="stroke-b"/>
  <line x1="370" y1="150" x2="480" y2="200" class="stroke-muted"/>
  <line x1="370" y1="200" x2="480" y2="200" class="stroke-muted"/>
  <line x1="370" y1="200" x2="480" y2="250" class="stroke-muted"/>
  <line x1="370" y1="250" x2="480" y2="250" class="stroke-muted"/>
  <circle cx="150" cy="50" r="13" class="fill-a"/>
  <circle cx="150" cy="100" r="13" class="fill-a"/>
  <circle cx="260" cy="50" r="13" class="fill-paper stroke-ink"/>
  <circle cx="260" cy="100" r="13" class="fill-paper stroke-ink"/>
  <circle cx="260" cy="150" r="13" class="fill-paper stroke-ink"/>
  <circle cx="260" cy="200" r="13" class="fill-paper stroke-ink"/>
  <circle cx="370" cy="100" r="13" class="fill-paper stroke-ink"/>
  <circle cx="370" cy="150" r="13" class="fill-paper stroke-ink"/>
  <circle cx="370" cy="200" r="13" class="fill-paper stroke-ink"/>
  <circle cx="370" cy="250" r="13" class="fill-paper stroke-ink"/>
  <circle cx="480" cy="200" r="13" class="fill-c"/>
  <circle cx="480" cy="250" r="13" class="fill-c"/>
  <text x="30" y="55" class="fig-mono">s=0  _</text>
  <text x="30" y="105" class="fig-mono">s=1  a</text>
  <text x="30" y="155" class="fig-mono">s=2  _</text>
  <text x="30" y="205" class="fig-mono">s=3  b</text>
  <text x="30" y="255" class="fig-mono">s=4  _</text>
  <text x="150" y="22" text-anchor="middle" class="fig-small">frame 1</text>
  <text x="260" y="22" text-anchor="middle" class="fig-small">frame 2</text>
  <text x="370" y="22" text-anchor="middle" class="fig-small">frame 3</text>
  <text x="480" y="22" text-anchor="middle" class="fig-small">frame 4</text>
  <text x="14" y="288" class="fig-small">blue: a path may start here. green: a path may end here. orange line: the skip from a to b.</text>
</svg>
Figure 1. The lattice for the transcript ab over four frames, with the states blank, a, blank, b, blank. A path starts in one of the first two states, ends in one of the last two, and may skip from a to b without a blank because the two labels differ.
```

Every path from a blue node to a green node is one path in the sum. There are at most $T \times S$ nodes, a few thousand for a sentence, so a running total per node is cheap.

## The forward recursion

Define $\alpha_t(s)$ as the total probability of every path prefix over frames 1 to $t$ that ends in state $s$, including the emission at frame $t$. Here $y^t_k$ is the network's probability of symbol $k$ at frame $t$. A path starts in the first blank or the first label, so

$$
\alpha_1(1) = y^1_{\varnothing}, \qquad \alpha_1(2) = y^1_{l_1}, \qquad \alpha_1(s) = 0 \text{ for } s > 2
$$

Each later frame adds up the allowed predecessors and multiplies by the emission:

$$
\alpha_t(s) = \Big(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \big[\text{skip allowed}\big]\,\alpha_{t-1}(s-2)\Big)\, y^t_{l'_s}
$$

Read it aloud: the probability of being in state $s$ at frame $t$ is the probability of having been in $s$, in $s - 1$, or in $s - 2$ where the skip is allowed, times the chance of emitting this state's symbol now. A valid path ends on the last label or the final blank, so $p(\mathbf{l} \mid x) = \alpha_T(S) + \alpha_T(S-1)$. The equations count from 1, and the code counts from 0.

In log space, shift the previous row by one and by two, and combine with `np.logaddexp`. A boolean mask `skip[s]` marks where the $s - 2$ term is allowed:

```python
total = prev.copy()
total[1:] = np.logaddexp(total[1:], prev[:-1])
total[2:] = np.where(skip[2:], np.logaddexp(total[2:], prev[:-2]), total[2:])
log_alpha[t] = total + log_probs[t, ext]
```

> [!KEY]
> CTC is the forward algorithm on a left-to-right HMM built from the transcript: blanks between labels, and a skip that is forbidden between two copies of the same label.

### Work T = 3 by hand

Use the example from the previous chapter. The vocabulary is blank (0) and `a` (1), $P(a)$ is 0.6, 0.3, and 0.8, and the target is `a`, so $\mathbf{l}' = (\varnothing, a, \varnothing)$. Frame 1 starts the first two states at $0.4$ and $0.6$. At frame 2:

- State 0 (blank) can only stay: $0.4 \times 0.7 = 0.28$.
- State 1 (`a`) can stay or advance from state 0: $(0.6 + 0.4) \times 0.3 = 0.30$.
- State 2 (blank) can stay or advance from state 1: $(0 + 0.6) \times 0.7 = 0.42$.

At frame 3 the same rules give $0.056$, $(0.30 + 0.28) \times 0.8 = 0.464$, and $(0.42 + 0.30) \times 0.2 = 0.144$. The left grid in Figure 2 shows all nine cells. So $p(a \mid x) = 0.464 + 0.144 = 0.608$, the same number the six-path table gave, and the loss is $-\ln 0.608 = 0.49758$. Nine cells replaced six paths here, and for 150 frames they replace $10^{39}$.

## The backward recursion

The gradient needs to know, for every frame and state, how much of the total probability flows through that cell. The forward pass counts what happens before it. The **backward** variable $\beta_t(s)$ counts what happens after: the probability of frames $t + 1$ to $T$, given state $s$ at frame $t$. In this chapter's convention it leaves out frame $t$'s own emission:

$$
\beta_T(S) = \beta_T(S - 1) = 1, \qquad \beta_t(s) = \sum_{s'} \beta_{t+1}(s')\, y^{t+1}_{l'_{s'}}
$$

Read it aloud: at the last frame, the final two states finish for free. At each earlier frame, add up over the next state $s'$, which is $s$, $s + 1$, or $s + 2$ where that skip is allowed, the probability of emitting its symbol at frame $t + 1$ and finishing from there. In the example, $\beta_2 = [0.8, 1.0, 0.2]$ and $\beta_1 = [0.86, 0.44, 0.14]$.

Because $\alpha$ includes frame $t$'s emission and $\beta$ leaves it out, their product counts every emission once. That gives a check you can run at every frame:

$$
\sum_{s} \alpha_t(s)\,\beta_t(s) = p(\mathbf{l} \mid x)
$$

At frame 1 of the example it is $0.4 \times 0.86 + 0.6 \times 0.44 + 0 \times 0.14 = 0.608$, as Figure 2 shows. Graves's paper uses the other convention, where $\beta$ includes the emission and you divide by it.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The forward table alpha and the backward table beta for the three-frame example, with alpha times beta summing to 0.608 at frame 1">
  <text x="30" y="30" class="fig-bold">alpha: includes frame t</text>
  <text x="98" y="52" text-anchor="middle" class="fig-small">t=1</text>
  <text x="156" y="52" text-anchor="middle" class="fig-small">t=2</text>
  <text x="214" y="52" text-anchor="middle" class="fig-small">t=3</text>
  <text x="66" y="86" text-anchor="end" class="fig-mono fig-small">_  s=0</text>
  <rect x="72" y="60" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="98" y="84" text-anchor="middle" class="fig-mono fig-small">0.4</text>
  <rect x="130" y="60" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="156" y="84" text-anchor="middle" class="fig-mono fig-small">0.28</text>
  <rect x="188" y="60" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="214" y="84" text-anchor="middle" class="fig-mono fig-small">0.056</text>
  <text x="66" y="130" text-anchor="end" class="fig-mono fig-small">a  s=1</text>
  <rect x="72" y="104" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="98" y="128" text-anchor="middle" class="fig-mono fig-small">0.6</text>
  <rect x="130" y="104" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="156" y="128" text-anchor="middle" class="fig-mono fig-small">0.3</text>
  <rect x="188" y="104" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="214" y="128" text-anchor="middle" class="fig-mono fig-small">0.464</text>
  <text x="66" y="174" text-anchor="end" class="fig-mono fig-small">_  s=2</text>
  <rect x="72" y="148" width="52" height="38" rx="4" class="fill-sunk stroke-muted"/>
  <text x="98" y="172" text-anchor="middle" class="fig-mono fig-small">0</text>
  <rect x="130" y="148" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="156" y="172" text-anchor="middle" class="fig-mono fig-small">0.42</text>
  <rect x="188" y="148" width="52" height="38" rx="4" class="fill-soft-a stroke-a"/>
  <text x="214" y="172" text-anchor="middle" class="fig-mono fig-small">0.144</text>
  <text x="320" y="30" class="fig-bold">beta: leaves frame t out</text>
  <text x="388" y="52" text-anchor="middle" class="fig-small">t=1</text>
  <text x="446" y="52" text-anchor="middle" class="fig-small">t=2</text>
  <text x="504" y="52" text-anchor="middle" class="fig-small">t=3</text>
  <text x="356" y="86" text-anchor="end" class="fig-mono fig-small">_  s=0</text>
  <rect x="362" y="60" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="388" y="84" text-anchor="middle" class="fig-mono fig-small">0.86</text>
  <rect x="420" y="60" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="446" y="84" text-anchor="middle" class="fig-mono fig-small">0.8</text>
  <rect x="478" y="60" width="52" height="38" rx="4" class="fill-sunk stroke-muted"/>
  <text x="504" y="84" text-anchor="middle" class="fig-mono fig-small">0</text>
  <text x="356" y="130" text-anchor="end" class="fig-mono fig-small">a  s=1</text>
  <rect x="362" y="104" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="388" y="128" text-anchor="middle" class="fig-mono fig-small">0.44</text>
  <rect x="420" y="104" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="446" y="128" text-anchor="middle" class="fig-mono fig-small">1</text>
  <rect x="478" y="104" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="504" y="128" text-anchor="middle" class="fig-mono fig-small">1</text>
  <text x="356" y="174" text-anchor="end" class="fig-mono fig-small">_  s=2</text>
  <rect x="362" y="148" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="388" y="172" text-anchor="middle" class="fig-mono fig-small">0.14</text>
  <rect x="420" y="148" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="446" y="172" text-anchor="middle" class="fig-mono fig-small">0.2</text>
  <rect x="478" y="148" width="52" height="38" rx="4" class="fill-soft-d stroke-d"/>
  <text x="504" y="172" text-anchor="middle" class="fig-mono fig-small">1</text>
  <text x="30" y="226" class="fig-mono">t=1: 0.4 × 0.86 + 0.6 × 0.44 + 0 × 0.14 = 0.608</text>
  <text x="30" y="250" class="fig-small">Every frame gives the same total, p(a | x) = 0.608, so alpha and beta together count each emission once.</text>
</svg>
Figure 2. The forward table and the backward table for the worked example. Gray cells are unreachable. Multiplying the two tables cell by cell and adding down a column gives 0.608 at every frame.
```

## The gradient

Training needs the derivative of the loss with respect to the network's logits. The **state occupancy** $\gamma_t(s) = \alpha_t(s)\,\beta_t(s) / p(\mathbf{l} \mid x)$ is the probability, over all alignments that spell the transcript, of being in state $s$ at frame $t$. Add up $\gamma_t(s)$ over the states that carry symbol $k$, and you get how much frame $t$ belongs to $k$ in the alignments. With $y^t = \operatorname{softmax}(u^t)$ over logits $u^t$:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial u^t_k} = y^t_k - \sum_{s:\ l'_s = k} \gamma_t(s)
$$

Read it aloud: the gradient for symbol $k$ at frame $t$ is what the model predicts minus how much the alignments use $k$ there. It is the cross-entropy gradient $y - q$ from the prerequisites, with a soft target $q$ that the forward and backward passes work out. Each row sums to zero, because both terms are distributions. Several states can share a symbol, because every blank slot is the same symbol, so `np.add.at` adds each state's occupancy into its symbol's column:

```python
gamma = np.exp(log_alpha + log_beta - log_p)          # (T, S)
np.add.at(occupancy, (slice(None), ext), gamma)       # (T, C)
grad = np.exp(log_probs) - occupancy
```

In the example, the occupancy of `a` is $0.4342$, $0.4934$, and $0.7632$ at the three frames, and the model's probabilities are $0.6$, $0.3$, and $0.8$. At frame 1 the model is more confident in `a` than the alignments warrant, so the gradient for `a` is $0.6 - 0.4342 = +0.1658$ and gradient descent lowers that logit. At frame 2 the model is too unsure, so the gradient is $0.3 - 0.4934 = -0.1934$ and descent raises it. Figure 3 draws the three frames. The full gradient matches torch's autograd.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="For each of three frames, a bar for the model probability of a and a bar for the alignment occupancy of a, with the gradient as their difference">
  <line x1="60" y1="200" x2="560" y2="200" class="stroke-muted"/>
  <rect x="100" y="110.0" width="50" height="90.0" class="fill-a"/>
  <rect x="160" y="134.9" width="50" height="65.1" class="fill-c"/>
  <text x="125" y="104.0" text-anchor="middle" class="fig-small fig-mono">0.60</text>
  <text x="185" y="128.9" text-anchor="middle" class="fig-small fig-mono">0.4342</text>
  <text x="155" y="220" text-anchor="middle">frame 1</text>
  <text x="155" y="240" text-anchor="middle" class="fig-mono">+0.1658</text>
  <rect x="270" y="155.0" width="50" height="45.0" class="fill-a"/>
  <rect x="330" y="126.0" width="50" height="74.0" class="fill-c"/>
  <text x="295" y="149.0" text-anchor="middle" class="fig-small fig-mono">0.30</text>
  <text x="355" y="120.0" text-anchor="middle" class="fig-small fig-mono">0.4934</text>
  <text x="325" y="220" text-anchor="middle">frame 2</text>
  <text x="325" y="240" text-anchor="middle" class="fig-mono">-0.1934</text>
  <rect x="440" y="80.0" width="50" height="120.0" class="fill-a"/>
  <rect x="500" y="85.5" width="50" height="114.5" class="fill-c"/>
  <text x="465" y="74.0" text-anchor="middle" class="fig-small fig-mono">0.80</text>
  <text x="525" y="79.5" text-anchor="middle" class="fig-small fig-mono">0.7632</text>
  <text x="495" y="220" text-anchor="middle">frame 3</text>
  <text x="495" y="240" text-anchor="middle" class="fig-mono">+0.0368</text>
  <rect x="60" y="30" width="14" height="14" class="fill-a"/>
  <text x="80" y="42" class="fig-small">model probability of a</text>
  <rect x="260" y="30" width="14" height="14" class="fill-c"/>
  <text x="280" y="42" class="fig-small">occupancy of a (alignment posterior)</text>
  <text x="14" y="262" class="fig-small">Below each frame: the gradient of the loss for the logit of a, the model minus the occupancy.</text>
</svg>
Figure 3. The model probability of a and its alignment occupancy at each frame, with the gradient below. Where the model is below the occupancy the gradient is negative, which pushes the logit up.
```

> [!KEY]
> The CTC gradient is model posterior minus alignment posterior. The forward and backward passes supply soft frame labels, so CTC trains like frame classification without anyone labeling frames.

## Match torch's CTC

torch's loss is `F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean")`. It has three traps:

- `log_probs` must have shape `(T, N, C)`, time first, and must already be log-softmax outputs. Raw logits, or batch first, run without an error and train nothing useful.
- `input_lengths` gives each utterance's frame count after subsampling, and `target_lengths` each transcript's length. Padding past them is ignored.
- `reduction="mean"` divides each utterance's loss by its target length, with 0 counted as 1, and then averages over the batch. To compare with a hand-written loss, use `reduction="sum"`.

> [!RECAP]
> - A skip into `ext[s]` needs `ext[s] != blank` and `ext[s] != ext[s - 2]`.
> - Forward: stay, advance, or allowed skip, times the emission. The loss is minus the log-sum-exp of the last two states at the last frame.
> - Backward leaves out frame $t$'s emission, so $\sum_s \alpha_t\beta_t = p$ at every $t$.
> - Gradient: `softmax(logits)` minus the occupancy, summed per symbol with `np.add.at`.

## Build it step by step

The checks test the functions in this order. The file provides `collapse` from the previous chapter, the tone generator, the features, the model, and a `torch.autograd.Function` that trains through your `ctc_grad`.

1. **`extend_with_blanks(target, blank)`.** Returns an int array of length `2L + 1`. Fill with blanks and assign `ext[1::2]`. The common mistake is leaving off the final blank.
2. **`ctc_forward(log_probs, target, blank)`.** Returns `(T, 2L + 1)`. Start from `-np.inf`, set the first two states at frame 0, then loop over frames with shifted `np.logaddexp`. The common mistake is allowing the skip into blank states.
3. **`ctc_loss(log_probs, target, blank)`.** Minus `logsumexp` of the last two entries of the last row. The common mistake is using only the final blank.
4. **`ctc_backward(log_probs, target, blank)`.** Mirror the forward pass from the last frame, shifting the other way. Add `log_probs[t + 1, ext]` before combining. The common mistake is including frame $t$'s emission, which breaks the $\alpha\beta$ identity.
5. **`ctc_grad(logits, target, blank)`.** Take `log_softmax`, run both passes, form `gamma`, scatter it with `np.add.at`, and subtract it from the softmax. The common mistake is `occupancy[:, ext] += gamma`, which drops all but one of the states that share a symbol.
6. **`ctc_loss_batch(...)`.** Slice each utterance to its lengths, call `ctc_loss`, then reduce. The common mistake is dividing by the input length for `"mean"`.

## Cost and scaling

The forward and backward passes each take $O(TL)$ time and memory for $T$ frames and $L$ labels, and the gradient adds $O(TC)$ for $C$ symbols. The cost depends on the transcript length, not on the alphabet size, because the lattice has $2L + 1$ states whatever $C$ is. A batch of $N$ utterances costs $N$ times one. The exercise loops over frames in Python, and torch's CUDA kernel runs the states of each frame in parallel.

## Where this breaks

**Numerics.** A hand-written log-sum-exp returns `nan` when every input is $-\infty$. `np.logaddexp` returns $-\infty$, the right answer for an unreachable state. **Infeasible targets.** When $T < L + r$ the loss is $+\infty$ and the gradient is undefined, so `ctc_grad` returns an infinite loss and a zero gradient.

## The exercise

You build `extend_with_blanks`, `ctc_forward`, `ctc_backward`, `ctc_loss`, the gradient `ctc_grad`, and `ctc_loss_batch`. Press **Run** to compare the worked example with torch, train a small convolutional model on synthetic tone sequences with your `ctc_grad` as its only loss, hear one sequence, and see its posteriors and greedy decode.

The checks compare your recursions with the T = 3 tables, your loss with a brute-force sum over every path and with `torch.nn.functional.ctc_loss`, and the $\alpha\beta$ identity at every frame. They check your gradient against torch's autograd and finite differences, and `ctc_loss_batch` against torch's three reductions. The last check trains free logits with your gradient until they spell a target with a repeat.

> [!KEY]
> What you now know: the sum over $10^{39}$ paths is a table of $T \times (2L + 1)$ cells, filled by stay, advance, and skip, and the same table run backward gives the gradient. That answers the opening question: paths share their prefixes, so you add them up once.
