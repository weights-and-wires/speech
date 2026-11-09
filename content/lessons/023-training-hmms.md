---
title: Training HMMs with Baum-Welch
slug: 023-training-hmms
part: "Part 3 — Classical recognition"
summary: Train an HMM when no one labeled the states, with the backward pass and Baum-Welch, and build a recognizer with one model per word.
minutes: 40
gpu: false
objectives:
  - Explain why training an HMM is a chicken-and-egg problem, and why hard labels throw information away.
  - Run the backward algorithm in log space and use alpha times beta to get the state occupancy of every frame.
  - Re-estimate means, variances, and transitions as weighted averages, and confirm that the likelihood never falls.
  - Build an isolated-word recognizer that trains one left-to-right HMM per word.
prereqs:
  - 022-hidden-markov-models
exercise: 023-training-hmms
---

# Training HMMs with Baum-Welch

You have an HMM for "osa" with eight states, and four recordings of the word. To set a state's Gaussian you need the frames that belong to that state. But no one labeled the frames, and you cannot align the frames to the states without a model whose Gaussians are already set. Each step needs the other.

The way out is to start with a rough guess, use it to decide how much each frame belongs to each state, refit the model from those shares, and repeat. This chapter builds that loop, called **Baum-Welch**, and then trains one model per word and uses the models to recognize words.

## What you already know

- The forward algorithm gives $\log \alpha_t(j)$, the log probability of the first $t$ frames and being in state $j$ at frame $t$. See [Hidden Markov models](/lessons/022-hidden-markov-models).
- Viterbi finds the single best state path.
- A weighted average of numbers $x_t$ with weights $w_t$ is $\sum_t w_t x_t / \sum_t w_t$.
- `scipy.special.logsumexp` sums probabilities stored as logs.

## Why hard labels fall short

The obvious plan is to label every frame with one state, then average the frames of each state. Start by splitting each recording into equal chunks, one per state. This **flat start** is a crude guess, because real states last different lengths. Then run Viterbi to relabel every frame under the model you just fit, refit, and repeat.

This discards information at every boundary. Take frame 2 of the toy from the last chapter, where the model is unsure whether the frame belongs to state 0 or state 1. A hard label hands the frame entirely to one state, so that state's mean is pulled toward the frame by a full weight, and the other state ignores it, although the model said the split was close to 60 to 40. A small error in the labels then feeds the next refit, which feeds the next labeling.

The softer plan keeps the doubt. Give each frame a **share** in every state, and use the shares as weights in the averages. Baum-Welch is the expectation-maximization (EM) algorithm for HMMs, and it does exactly this. Each lap never lowers $P(O \mid \lambda)$.

```figure
<svg viewBox="0 0 600 235" role="img" aria-label="A loop with two boxes: soft alignment from forward and backward, then refit the model from the soft labels">
<defs><marker id="c022-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="40.0" y="60.0" width="220.0" height="90.0" class="fill-soft-a stroke-a" />
<text x="150.0" y="90.0" class="fig-bold" text-anchor="middle">1. Soft alignment</text>
<text x="150.0" y="112.0" class="fig-small" text-anchor="middle">forward and backward give</text>
<text x="150.0" y="130.0" class="fig-small" text-anchor="middle">gamma and xi for every frame</text>
<rect x="340.0" y="60.0" width="220.0" height="90.0" class="fill-soft-b stroke-b" />
<text x="450.0" y="90.0" class="fig-bold" text-anchor="middle">2. Refit</text>
<text x="450.0" y="112.0" class="fig-small" text-anchor="middle">weighted means, variances,</text>
<text x="450.0" y="130.0" class="fig-small" text-anchor="middle">and transition counts</text>
<path d="M260 90 C 290 60, 310 60, 338 90" class="stroke-ink" fill="none" marker-end="url(#c022-arrow)"/>
<path d="M340 125 C 310 160, 290 160, 262 125" class="stroke-ink" fill="none" marker-end="url(#c022-arrow)"/>
<text x="300.0" y="48.0" class="fig-small" text-anchor="middle">new model</text>
<text x="300.0" y="178.0" class="fig-small" text-anchor="middle">soft labels</text>
<text x="300.0" y="215.0" class="fig-bold fig-small" text-anchor="middle">Each lap raises log P(O) or leaves it where it was.</text>
</svg>
Figure 1. One lap of Baum-Welch. The soft alignment uses the current model to decide how much each frame belongs to each state, and the refit builds a new model from those shares. Notice that nothing in the loop needs a human label.
```

## Look backward as well as forward

The share of frame $t$ in state $i$ depends on the whole utterance, including the frames after $t$. The forward variable covers only the frames up to $t$, so you need a partner that covers the rest. The **backward variable** $\beta_t(i)$ is the probability of the observations after frame $t$, given state $i$ at frame $t$. It runs from the end:

$$
\begin{aligned}
\beta_T(i) &= 1 \\
\beta_t(i) &= \sum_{j=1}^{N} a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)
\end{aligned}
$$

Read it aloud: at the last frame nothing is left to explain. Earlier, add up over every next state $j$ the chance of moving there, emitting the next frame, and explaining the rest. In log space this is one `logsumexp` per frame, over the destination states:

```python
log_beta[t] = logsumexp(log_A + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)
```

On the toy HMM from the last chapter ($O = (0, 1, 2)$), $\beta_3 = [1, 1]$, $\beta_2 = [0.25, 0.40]$, and $\beta_1 = [0.106, 0.112]$.

The product $\alpha_t(i)\beta_t(i)$ is the probability of all the observations *and* state $i$ at frame $t$, because $\alpha$ explains the past and $\beta$ the future. Summing over $i$ must therefore give $P(O)$ at every frame. At frame 1 of the toy, $0.3 \times 0.106 + 0.04 \times 0.112 = 0.03628$, the number the forward pass produced. That is a strong test of both passes.

## Turn the products into shares

Divide $\alpha_t(i)\beta_t(i)$ by $P(O \mid \lambda)$ and you get the share. Two kinds of share are needed. The **state occupancy** $\gamma_t(i)$ is the probability of state $i$ at frame $t$, given all the observations. The **transition posterior** $\xi_t(i, j)$ is the probability of moving from $i$ at frame $t$ to $j$ at frame $t+1$:

$$
\begin{aligned}
\gamma_t(i) &= \frac{\alpha_t(i)\,\beta_t(i)}{P(O \mid \lambda)} \\
\xi_t(i, j) &= \frac{\alpha_t(i)\, a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)}{P(O \mid \lambda)}
\end{aligned}
$$

Read it aloud: $\gamma$ is the forward and backward probability through state $i$ at frame $t$, as a share of the total. $\xi$ is the same for one transition: arrive in $i$, move to $j$, emit the next frame, and explain the rest. For the toy, $\gamma_1 = [0.877, 0.123]$, $\gamma_2 = [0.623, 0.377]$, and $\gamma_3 = [0.212, 0.788]$. Figure 2 draws them.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Three stacked bars, one per frame, splitting each frame between state 0 and state 1 according to gamma">
<rect x="120.0" y="40.0" width="80.0" height="157.8" class="fill-soft-a stroke-a" />
<rect x="120.0" y="197.8" width="80.0" height="22.2" class="fill-soft-b stroke-b" />
<text x="160.0" y="123.9" class="fig-small" text-anchor="middle">0.877</text>
<text x="160.0" y="213.9" class="fig-small" text-anchor="middle">0.123</text>
<text x="160.0" y="240.0" class="fig-small" text-anchor="middle">frame 1</text>
<rect x="270.0" y="40.0" width="80.0" height="112.1" class="fill-soft-a stroke-a" />
<rect x="270.0" y="152.1" width="80.0" height="67.9" class="fill-soft-b stroke-b" />
<text x="310.0" y="101.1" class="fig-small" text-anchor="middle">0.623</text>
<text x="310.0" y="191.1" class="fig-small" text-anchor="middle">0.377</text>
<text x="310.0" y="240.0" class="fig-small" text-anchor="middle">frame 2</text>
<rect x="420.0" y="40.0" width="80.0" height="38.2" class="fill-soft-a stroke-a" />
<rect x="420.0" y="78.2" width="80.0" height="141.8" class="fill-soft-b stroke-b" />
<text x="460.0" y="64.1" class="fig-small" text-anchor="middle">0.212</text>
<text x="460.0" y="154.1" class="fig-small" text-anchor="middle">0.788</text>
<text x="460.0" y="240.0" class="fig-small" text-anchor="middle">frame 3</text>
<rect x="525.0" y="70.0" width="14.0" height="14.0" class="fill-soft-a stroke-a" />
<text x="547.0" y="82.0" class="fig-small" text-anchor="start">state 0</text>
<rect x="525.0" y="100.0" width="14.0" height="14.0" class="fill-soft-b stroke-b" />
<text x="547.0" y="112.0" class="fig-small" text-anchor="start">state 1</text>
<text x="310.0" y="262.0" class="fig-small" text-anchor="middle">Each bar is one frame, split between the two states; the shares add up to 1</text>
</svg>
Figure 2. The occupancy gamma for the three frames of the toy. Notice that frame 1 belongs mostly to state 0, frame 3 mostly to state 1, and frame 2 is split, which is the doubt that a hard label would erase.
```

## Refit with weighted counts

With the shares in hand, the new parameters are weighted counts and weighted averages:

$$
\begin{aligned}
\hat a_{ij} &= \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)} \\
\hat\mu_j &= \frac{\sum_{t} \gamma_t(j)\, o_t}{\sum_{t} \gamma_t(j)} \\
\hat\sigma^2_j &= \frac{\sum_t \gamma_t(j)\, o_t^2}{\sum_t \gamma_t(j)} - \hat\mu_j^2
\end{aligned}
$$

Read it aloud: the new transition probability is the expected number of $i$-to-$j$ moves divided by the expected number of moves out of $i$. The new mean is the average frame, with each frame weighted by how much it belongs to state $j$. The new variance is the weighted average of the squares minus the square of the mean. With several training sequences, add up the numerators and denominators over all of them before dividing. In numpy, with `gamma` of shape `(T, N)`:

```python
sum_x += gamma.T @ X            # (N, D)
occupancy += gamma.sum(axis=0)  # (N,)
means = sum_x / occupancy[:, None]
```

On the toy, the transition update works out to $\hat a_{00} = 0.7532 / 1.4994 = 0.502$ and $\hat a_{01} = 0.498$. To see the mean update, suppose the three frames carried the single feature values 0, 1, and 2. The shares of state 0 add up to $1.712$, so $\hat\mu_0 = (0.877 \times 0 + 0.623 \times 1 + 0.212 \times 2)/1.712 = 0.612$. State 0 has a mean near the frames it mostly owns, and state 1, with $\hat\mu_1 = 1.516$, sits near the later ones.

A transition that starts at probability 0 has $\xi = 0$ forever, so Baum-Welch keeps the shape of the model. That is how a left-to-right model stays left-to-right.

> [!WARNING]
> The transition sums run over $t = 1$ to $T-1$, and so does the denominator of $\hat a_{ij}$. Dividing by the occupancy over all $T$ frames makes the rows of $\hat A$ sum to less than 1.

> [!KEY]
> Baum-Welch alternates soft alignment ($\gamma$ and $\xi$ from forward and backward) with refitting (weighted averages). The likelihood never falls, so a drop means a bug.

## Watch the likelihood climb

Figure 3 plots the log-likelihood per frame of four word models over six laps. Each value is the likelihood of the training takes under the parameters *before* that lap, and the first is the flat start. The "osa" model climbs from $-16.04$ to $-10.88$ nats per frame, with most of the gain in the first lap, and then the curve flattens. EM stops improving at a **local optimum**, a model that no small change improves, which depends on where it started.

```figure
<svg viewBox="0 0 600 292" role="img" aria-label="Four curves of log-likelihood per frame rising over six Baum-Welch iterations and flattening">
<line x1="90.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-muted"/>
<line x1="90.0" y1="30.0" x2="90.0" y2="230.0" class="stroke-muted"/>
<line x1="90.0" y1="230.0" x2="540.0" y2="230.0" class="stroke-grid"/>
<text x="84.0" y="234.0" class="fig-small" text-anchor="end">-20</text>
<line x1="90.0" y1="163.3" x2="540.0" y2="163.3" class="stroke-grid"/>
<text x="84.0" y="167.3" class="fig-small" text-anchor="end">-15</text>
<line x1="90.0" y1="96.7" x2="540.0" y2="96.7" class="stroke-grid"/>
<text x="84.0" y="100.7" class="fig-small" text-anchor="end">-10</text>
<line x1="90.0" y1="30.0" x2="540.0" y2="30.0" class="stroke-grid"/>
<text x="84.0" y="34.0" class="fig-small" text-anchor="end">-5</text>
<text x="90.0" y="248.0" class="fig-small" text-anchor="middle">0</text>
<text x="180.0" y="248.0" class="fig-small" text-anchor="middle">1</text>
<text x="270.0" y="248.0" class="fig-small" text-anchor="middle">2</text>
<text x="360.0" y="248.0" class="fig-small" text-anchor="middle">3</text>
<text x="450.0" y="248.0" class="fig-small" text-anchor="middle">4</text>
<text x="540.0" y="248.0" class="fig-small" text-anchor="middle">5</text>
<polyline points="90.0,175.8 180.0,142.8 270.0,141.2 360.0,140.7 450.0,140.6 540.0,140.6" class="stroke-a" fill="none"/>
<line x1="110.0" y1="274.0" x2="140.0" y2="274.0" class="stroke-a"/>
<text x="146.0" y="278.0" class="fig-small" text-anchor="start">sa</text>
<polyline points="90.0,144.5 180.0,111.2 270.0,105.9 360.0,105.8 450.0,105.8 540.0,105.8" class="stroke-b" fill="none"/>
<line x1="210.0" y1="274.0" x2="240.0" y2="274.0" class="stroke-b"/>
<text x="246.0" y="278.0" class="fig-small" text-anchor="start">shi</text>
<polyline points="90.0,177.1 180.0,111.8 270.0,108.6 360.0,108.4 450.0,108.3 540.0,108.3" class="stroke-c" fill="none"/>
<line x1="310.0" y1="274.0" x2="340.0" y2="274.0" class="stroke-c"/>
<text x="346.0" y="278.0" class="fig-small" text-anchor="start">osa</text>
<polyline points="90.0,137.0 180.0,105.3 270.0,94.5 360.0,88.7 450.0,88.6 540.0,88.6" class="stroke-d" fill="none"/>
<line x1="410.0" y1="274.0" x2="440.0" y2="274.0" class="stroke-d"/>
<text x="446.0" y="278.0" class="fig-small" text-anchor="start">eshu</text>
<text x="84.0" y="20.0" class="fig-small" text-anchor="start">log P(O) per frame (nats)</text>
<text x="315.0" y="266.0" class="fig-small" text-anchor="middle">iteration</text>
</svg>
Figure 3. Log-likelihood per frame for four word models over six Baum-Welch iterations. Notice that every curve rises or stays level, never falls, and that most of the gain comes in the first one or two laps.
```

## Build a word recognizer

For isolated words, give each word its own left-to-right HMM, trained from four takes of that word. The flat start splits each take into equal chunks to set the first means and variances, and it sets every self-loop to 0.6, a mean stay of 2.5 frames or 25 ms, which Baum-Welch then adjusts to fit the data. To recognize an utterance, run the forward algorithm under every word's model and answer with the word whose $\log P(O \mid \lambda_w)$ is highest. With four takes per word and six Baum-Welch steps, the Run block's recognizer names all 40 noisy test words.

> [!RECAP]
> - `backward` runs from `log_beta[T-1] = 0`, and $\sum_i \alpha_t(i)\beta_t(i) = P(O)$ at every frame.
> - $\gamma_t(i) = \alpha_t(i)\beta_t(i)/P(O)$ and $\xi_t(i,j)$ are the soft labels.
> - The new means, variances, and transitions are weighted averages, summed over every training sequence before dividing.
> - The transition sums stop at $T - 1$, and variances need a floor.
> - Recognition scores each word's model with the forward algorithm.

## Build it step by step

The checks test the functions in this order. The file gives you `log_gaussian_diag` and `forward` from the last chapter in full, plus `flat_start`.

1. **`backward(log_A, log_B)`.** Returns `log_beta` of shape `(T, N)`, with the last row 0. Fill from the end with `logsumexp` over `axis=1`. The usual mistake is `log_B[t]` where the formula needs `log_B[t + 1]`.
2. **`baum_welch_step(sequences, log_pi, log_A, means, variances, var_floor)`.** For each sequence, run forward and backward, build `gamma` and `xi`, and add to running sums. Divide once at the end, and floor the variances with `np.maximum`. Return the log-likelihood under the *old* parameters. The usual mistake is normalizing per sequence and averaging, which weights a short take as much as a long one.
3. **`train_word_hmm(sequences, n_states, n_iter)`.** Call `flat_start`, then `baum_welch_step` in a loop, and record the log-likelihood each step returns.
4. **`recognize(X, models)`.** Score with `forward` under each model and return the best name. The usual mistake is comparing Viterbi scores from one model with forward scores from another.

## Where this breaks

**Variances collapse.** A state that owns only a few nearly identical frames drives its variance toward zero and its likelihood toward infinity. Floor the variances, as `var_floor` does.

**A bad start gives a bad model.** EM climbs from wherever it starts. The flat start is a reasonable guess for a left-to-right word, and not for every model.

**Whole-word models do not scale.** Each word needs several takes of its own. Large-vocabulary systems share phone models between words, which [Word error rate](/lessons/026-word-error-rate) describes.

## The exercise

You build `backward`, `baum_welch_step` for Gaussian HMMs, `train_word_hmm`, and `recognize`. Press **Run** to train one HMM per synthetic word, see the log-likelihood per frame rise with every iteration, and recognize 40 noisy words.

The checks compare `backward` with the toy and confirm that $\alpha_t \beta_t$ sums to $P(O)$ at every frame. They confirm that one `baum_welch_step` on a one-state model gives the sample mean and variance, that eight steps never lower the likelihood, and that rows of $A$ sum to 1 with zeros staying zero. The last two checks run `train_word_hmm` and require 36 of 40 noisy words to be recognized.
