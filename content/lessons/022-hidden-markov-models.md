---
title: Hidden Markov models
slug: 022-hidden-markov-models
part: "Part 3 — Classical recognition"
summary: Model a word as hidden states that emit frames, then score a sequence with the forward algorithm and align it with Viterbi, in log space.
minutes: 40
gpu: false
objectives:
  - Compute diagonal Gaussian log-likelihoods for every frame and state.
  - Run the forward algorithm in log space, and check it on a toy HMM worked by hand.
  - Explain why a linear-space forward pass returns 0 on a real utterance.
  - Run Viterbi to find the single best state path, and explain how it differs from the forward score.
prereqs:
  - 021-dynamic-time-warping
exercise: 022-hidden-markov-models
---

# Hidden Markov models

A template recognizer from [Dynamic time warping](/lessons/021-dynamic-time-warping) keeps whole recordings and compares against each one. It learns nothing about which parts of a word vary and which parts stay fixed. A second speaker, a different microphone, or a slightly different vowel all look equally far from the template.

A **hidden Markov model** (HMM) replaces the recording with a small model that can answer one question: how likely is it that this stretch of frames came from this word? Picture a word as a row of rooms that you walk through from left to right. In each room you linger for a while, and every 10 ms you make a sound drawn from that room's own distribution. You hear only the sounds, and the rooms stay hidden.

From the 1980s until deep learning, almost every speech recognizer was built from HMMs. In this chapter you build the two algorithms that use one: the forward algorithm, which scores a sequence, and Viterbi, which says which room produced each frame.

## What you already know

- An MFCC vector describes one 10 ms frame with 13 numbers. See [MFCCs](/lessons/014-mfccs).
- A Gaussian density $\mathcal{N}(x; \mu, \sigma^2)$ is the bell curve with mean $\mu$ and variance $\sigma^2$.
- DTW fills a table where each cell extends the best path to one of its neighbors.
- `scipy.special.logsumexp(v)` computes $\log \sum_i e^{v_i}$ without overflow.

## A word as a row of rooms

An HMM with $N$ states has three sets of parameters, written together as $\lambda = (\pi, A, B)$:

- The **initial probabilities** $\pi_i$: the chance of starting in state $i$.
- The **transition probabilities** $a_{ij} = P(q_{t+1} = j \mid q_t = i)$: the chance of moving from state $i$ to state $j$ between frames, where $q_t$ is the state at frame $t$. Each row of $A$ sums to 1.
- The **emission densities** $b_j(o_t) = p(o_t \mid q_t = j)$: how likely state $j$ is to produce the observation $o_t$, the feature vector of frame $t$.

The model is *Markov* because the next state depends only on the current one, and *hidden* because you observe $o_1, \dots, o_T$ but never $q_1, \dots, q_T$. Figure 1 shows a toy with two states and three discrete symbols, which the rest of the chapter reuses so that every number can be checked by hand.

```figure
<svg viewBox="0 0 580 378" role="img" aria-label="Two states with arrows labeled by transition probability and a bar chart of the symbols each state emits">
<defs><marker id="c021-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<circle cx="150.0" cy="110.0" r="34" class="fill-soft-a stroke-a"/>
<text x="150.0" y="115.0" class="fig-bold" text-anchor="middle">state 0</text>
<circle cx="420.0" cy="110.0" r="34" class="fill-soft-a stroke-a"/>
<text x="420.0" y="115.0" class="fig-bold" text-anchor="middle">state 1</text>
<path d="M130 79 C 100 25, 200 25, 170 79" class="stroke-ink" fill="none" marker-end="url(#c021-arrow)"/>
<text x="150.0" y="32.0" class="fig-small" text-anchor="middle">0.7</text>
<path d="M400 79 C 370 25, 470 25, 440 79" class="stroke-ink" fill="none" marker-end="url(#c021-arrow)"/>
<text x="420.0" y="32.0" class="fig-small" text-anchor="middle">0.6</text>
<path d="M184 100 C 240 80, 330 80, 384 100" class="stroke-ink" fill="none" marker-end="url(#c021-arrow)"/>
<text x="285.0" y="80.0" class="fig-small" text-anchor="middle">0.3</text>
<path d="M386 124 C 330 155, 240 155, 186 124" class="stroke-ink" fill="none" marker-end="url(#c021-arrow)"/>
<text x="285.0" y="162.0" class="fig-small" text-anchor="middle">0.4</text>
<text x="80.0" y="114.0" class="fig-small" text-anchor="middle">start 0.6</text>
<text x="498.0" y="114.0" class="fig-small" text-anchor="middle">start 0.4</text>
<line x1="100.0" y1="285.0" x2="200.0" y2="285.0" class="stroke-muted"/>
<rect x="108.0" y="240.0" width="24.0" height="45.0" class="fill-b" />
<text x="120.0" y="301.0" class="fig-small" text-anchor="middle">0</text>
<text x="120.0" y="235.0" class="fig-small" text-anchor="middle">0.5</text>
<rect x="140.0" y="249.0" width="24.0" height="36.0" class="fill-b" />
<text x="152.0" y="301.0" class="fig-small" text-anchor="middle">1</text>
<text x="152.0" y="244.0" class="fig-small" text-anchor="middle">0.4</text>
<rect x="172.0" y="276.0" width="24.0" height="9.0" class="fill-b" />
<text x="184.0" y="301.0" class="fig-small" text-anchor="middle">2</text>
<text x="184.0" y="271.0" class="fig-small" text-anchor="middle">0.1</text>
<text x="150.0" y="323.0" class="fig-small" text-anchor="middle">state 0 emits symbol 0, 1, or 2</text>
<line x1="150.0" y1="146.0" x2="150.0" y2="180.0" class="stroke-muted dashed"/>
<line x1="370.0" y1="285.0" x2="470.0" y2="285.0" class="stroke-muted"/>
<rect x="378.0" y="276.0" width="24.0" height="9.0" class="fill-b" />
<text x="390.0" y="301.0" class="fig-small" text-anchor="middle">0</text>
<text x="390.0" y="271.0" class="fig-small" text-anchor="middle">0.1</text>
<rect x="410.0" y="258.0" width="24.0" height="27.0" class="fill-b" />
<text x="422.0" y="301.0" class="fig-small" text-anchor="middle">1</text>
<text x="422.0" y="253.0" class="fig-small" text-anchor="middle">0.3</text>
<rect x="442.0" y="231.0" width="24.0" height="54.0" class="fill-b" />
<text x="454.0" y="301.0" class="fig-small" text-anchor="middle">2</text>
<text x="454.0" y="226.0" class="fig-small" text-anchor="middle">0.6</text>
<text x="420.0" y="323.0" class="fig-small" text-anchor="middle">state 1 emits symbol 0, 1, or 2</text>
<line x1="420.0" y1="146.0" x2="420.0" y2="180.0" class="stroke-muted dashed"/>
<text x="300.0" y="362.0" class="fig-small" text-anchor="middle">Transition probabilities (top) and emission probabilities (bottom) of the toy HMM</text>
</svg>
Figure 1. A two-state toy HMM. The arrows carry the transition probabilities (each state's outgoing arrows add up to 1), and the bars show how likely each state is to emit symbol 0, 1, or 2. Notice that state 0 mostly emits symbols 0 and 1 and state 1 mostly emits symbol 2.
```

For speech the emission is a density over 13-dimensional MFCC vectors. The simplest choice is one Gaussian per state with a **diagonal covariance**, which means one variance per feature and no correlations between features. State $j$ then scores a frame $o$ with

$$
\log b_j(o) = -\frac{1}{2}\left[D \log 2\pi + \sum_{d=1}^{D} \log \sigma_{jd}^2 + \sum_{d=1}^{D} \frac{(o_d - \mu_{jd})^2}{\sigma_{jd}^2}\right]
$$

Read it aloud: a constant for the $D$ dimensions, plus a penalty for wide variances, plus the squared distance from the mean in units of each feature's variance, all times minus one half. For one dimension at its own mean with variance 1, this is $-\tfrac12 \log 2\pi = -0.919$. The exercise computes every frame against every state at once, an array of shape `(T, N)`:

```python
diff2 = (X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
log_B = -0.5 * (D * np.log(2 * np.pi) + np.log(variances).sum(1)[None, :] + diff2.sum(2))
```

Classic recognizers gave each state a mixture of several Gaussians so that one state could cover sounds that come in several shapes. This chapter keeps one Gaussian per state, because the algorithms below do not care how $\log b_j(o_t)$ was computed.

## Score a sequence with the forward algorithm

The likelihood of the observations, $P(O \mid \lambda)$, is a sum over every state sequence that could have produced them, because you do not know which one did. There are $N^T$ of those, which is hopeless for 100 frames. The trick is the one DTW used. Define the **forward variable** $\alpha_t(j)$, the probability of the first $t$ observations *and* being in state $j$ at frame $t$. Each $\alpha_{t+1}(j)$ needs only the previous column:

$$
\begin{aligned}
\alpha_1(j) &= \pi_j\, b_j(o_1) \\
\alpha_{t+1}(j) &= \Big[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\Big]\, b_j(o_{t+1}) \\
P(O \mid \lambda) &= \sum_{j=1}^{N} \alpha_T(j)
\end{aligned}
$$

Read it aloud: start with the chance of starting in $j$ times the chance that $j$ emits the first frame. At each later frame, add up every way of arriving in $j$ from the previous frame, then multiply by the chance that $j$ emits this frame. At the end, add up over the final states. The equations count frames from 1, and the code counts from 0. This is DTW with the minimum replaced by a sum and the costs replaced by probabilities, and it takes $O(TN^2)$ time instead of $O(N^T)$.

### Work the forward pass by hand

Use the toy from Figure 1: $\pi = [0.6, 0.4]$, transitions $A = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix}$, and emissions $B = \begin{bmatrix} 0.5 & 0.4 & 0.1 \\ 0.1 & 0.3 & 0.6 \end{bmatrix}$, where row $j$ is state $j$'s distribution over the symbols 0, 1, and 2. Observe $O = (0, 1, 2)$:

- Frame 1: $\alpha_1 = [0.6 \times 0.5,\ 0.4 \times 0.1] = [0.3,\ 0.04]$.
- Frame 2, state 0: $(0.3 \times 0.7 + 0.04 \times 0.4) \times 0.4 = 0.226 \times 0.4 = 0.0904$.
- Frame 2, state 1: $(0.3 \times 0.3 + 0.04 \times 0.6) \times 0.3 = 0.114 \times 0.3 = 0.0342$.
- Frame 3, state 0: $(0.0904 \times 0.7 + 0.0342 \times 0.4) \times 0.1 = 0.007696$.
- Frame 3, state 1: $(0.0904 \times 0.3 + 0.0342 \times 0.6) \times 0.6 = 0.028584$.

So $P(O \mid \lambda) = 0.007696 + 0.028584 = 0.03628$. Figure 2 draws the same numbers as a trellis, the grid of states against frames.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="A trellis of two states across three frames with the forward value alpha written in each node">
<text x="150.0" y="40.0" class="fig-small" text-anchor="middle">frame 1, symbol 0</text>
<text x="310.0" y="40.0" class="fig-small" text-anchor="middle">frame 2, symbol 1</text>
<text x="470.0" y="40.0" class="fig-small" text-anchor="middle">frame 3, symbol 2</text>
<text x="30.0" y="105.0" class="fig-small" text-anchor="start">state 0</text>
<text x="30.0" y="205.0" class="fig-small" text-anchor="start">state 1</text>
<line x1="180.0" y1="100.0" x2="280.0" y2="100.0" class="stroke-grid"/>
<line x1="180.0" y1="106.0" x2="280.0" y2="194.0" class="stroke-grid"/>
<line x1="180.0" y1="194.0" x2="280.0" y2="106.0" class="stroke-grid"/>
<line x1="180.0" y1="200.0" x2="280.0" y2="200.0" class="stroke-grid"/>
<line x1="340.0" y1="100.0" x2="440.0" y2="100.0" class="stroke-grid"/>
<line x1="340.0" y1="106.0" x2="440.0" y2="194.0" class="stroke-grid"/>
<line x1="340.0" y1="194.0" x2="440.0" y2="106.0" class="stroke-grid"/>
<line x1="340.0" y1="200.0" x2="440.0" y2="200.0" class="stroke-grid"/>
<circle cx="150.0" cy="100.0" r="30" class="fill-sunk stroke-muted"/>
<text x="150.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.3</text>
<circle cx="150.0" cy="200.0" r="30" class="fill-sunk stroke-muted"/>
<text x="150.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.04</text>
<circle cx="310.0" cy="100.0" r="30" class="fill-sunk stroke-muted"/>
<text x="310.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.0904</text>
<circle cx="310.0" cy="200.0" r="30" class="fill-sunk stroke-muted"/>
<text x="310.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.0342</text>
<circle cx="470.0" cy="100.0" r="30" class="fill-sunk stroke-muted"/>
<text x="470.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.007696</text>
<circle cx="470.0" cy="200.0" r="30" class="fill-sunk stroke-muted"/>
<text x="470.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.028584</text>
<text x="300.0" y="270.0" class="fig-bold" text-anchor="middle">P(O) = 0.007696 + 0.028584 = 0.03628</text>
</svg>
Figure 2. The forward trellis for O = (0, 1, 2). Each node holds alpha for that state and frame, and every node is built from both nodes in the column to its left. Notice that the last column adds up to P(O).
```

### Stay in log space

Each frame multiplies in a probability below 1, so $\alpha$ shrinks exponentially. On the exercise's random four-state HMM with 2,000 frames, $\log P(O) = -2799.5$ nats, which is about $10^{-1216}$. The smallest positive float64 is about $5 \times 10^{-324}$, so Figure 3 shows the forward probability reaching the floor within about 530 frames, and a linear-space pass returns exactly 0 long before the end.

```figure
<svg viewBox="0 0 640 300" role="img" aria-label="The log10 of the probability of the observations so far falls steadily with each frame and crosses the smallest float64 value after about 530 frames">
<line x1="90.0" y1="250.0" x2="560.0" y2="250.0" class="stroke-muted"/>
<line x1="90.0" y1="40.0" x2="90.0" y2="250.0" class="stroke-muted"/>
<line x1="90.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="84.0" y="44.0" class="fig-small" text-anchor="end">0</text>
<line x1="90.0" y1="90.4" x2="560.0" y2="90.4" class="stroke-grid"/>
<text x="84.0" y="94.4" class="fig-small" text-anchor="end">-300</text>
<line x1="90.0" y1="140.8" x2="560.0" y2="140.8" class="stroke-grid"/>
<text x="84.0" y="144.8" class="fig-small" text-anchor="end">-600</text>
<line x1="90.0" y1="191.2" x2="560.0" y2="191.2" class="stroke-grid"/>
<text x="84.0" y="195.2" class="fig-small" text-anchor="end">-900</text>
<line x1="90.0" y1="241.6" x2="560.0" y2="241.6" class="stroke-grid"/>
<text x="84.0" y="245.6" class="fig-small" text-anchor="end">-1200</text>
<text x="90.0" y="268.0" class="fig-small" text-anchor="middle">0</text>
<text x="207.5" y="268.0" class="fig-small" text-anchor="middle">500</text>
<text x="325.0" y="268.0" class="fig-small" text-anchor="middle">1000</text>
<text x="442.5" y="268.0" class="fig-small" text-anchor="middle">1500</text>
<text x="560.0" y="268.0" class="fig-small" text-anchor="middle">2000</text>
<polyline points="90.0,40.1 94.7,42.1 99.4,44.1 104.1,46.3 108.8,48.2 113.5,50.3 118.2,52.4 122.9,54.4 127.6,56.5 132.3,58.5 137.0,60.5 141.7,62.6 146.4,64.6 151.1,66.6 155.8,68.7 160.5,70.8 165.2,72.7 169.9,74.8 174.6,76.9 179.3,78.9 184.0,81.1 188.7,83.1 193.4,85.1 198.1,87.2 202.8,89.2 207.5,91.4 212.2,93.5 216.9,95.5 221.6,97.5 226.3,99.6 231.0,101.5 235.7,103.6 240.4,105.6 245.1,107.6 249.8,109.7 254.5,111.8 259.2,113.8 263.9,115.8 268.6,117.8 273.3,119.9 278.0,121.9 282.7,123.9 287.4,126.0 292.1,128.0 296.8,130.1 301.5,132.1 306.2,134.1 310.9,136.1 315.6,138.3 320.3,140.4 325.0,142.3 329.7,144.4 334.4,146.4 339.1,148.5 343.8,150.5 348.5,152.5 353.2,154.6 357.9,156.6 362.6,158.6 367.3,160.6 372.0,162.6 376.7,164.7 381.4,166.7 386.1,168.8 390.8,170.8 395.5,172.9 400.2,174.8 404.9,176.9 409.6,179.0 414.3,181.1 419.0,183.1 423.7,185.0 428.4,187.1 433.1,189.2 437.8,191.2 442.5,193.3 447.2,195.3 451.9,197.3 456.6,199.4 461.3,201.4 466.0,203.5 470.7,205.5 475.4,207.5 480.1,209.6 484.8,211.7 489.5,213.8 494.2,215.8 498.9,217.8 503.6,219.9 508.3,221.9 513.0,224.0 517.7,226.1 522.4,228.1 527.1,230.2 531.8,232.2 536.5,234.3 541.2,236.4 545.9,238.4 550.6,240.4 555.3,242.3 559.8,244.3" class="stroke-b" fill="none"/>
<line x1="90.0" y1="94.3" x2="560.0" y2="94.3" class="stroke-bad dashed"/>
<text x="560.0" y="88.3" class="fig-small" text-anchor="end">smallest float64, about 10^-323</text>
<text x="566.0" y="248.3" class="fig-small" text-anchor="start">10^-1216</text>
<text x="325.0" y="288.0" class="fig-small" text-anchor="middle">frame t (one frame is 10 ms)</text>
<text x="84.0" y="28.0" class="fig-small" text-anchor="end">log10 P(o1 .. ot)</text>
</svg>
Figure 3. The base-10 logarithm of P(o1 .. ot) for that 2,000-frame example. Notice that the curve is nearly a straight line, so the probability falls by a constant factor per frame, and that it crosses the dashed float64 floor after about 530 frames (5.3 seconds).
```

The fix is to store $\log \alpha$ and turn every sum of products into a log-sum-exp:

$$
\begin{aligned}
\log \alpha_{t+1}(j) &= \operatorname{LSE}_i\big(\log \alpha_t(i) + \log a_{ij}\big) + \log b_j(o_{t+1}) \\
\operatorname{LSE}(v) &= v_{\max} + \log \sum_i e^{v_i - v_{\max}}
\end{aligned}
$$

Read it aloud: add logs where the linear version multiplies, and combine the incoming paths with log-sum-exp, which subtracts the largest term before exponentiating so nothing underflows. A forbidden transition has $\log a_{ij} = -\infty$, which `logsumexp` handles. In numpy:

```python
log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]
```

> [!KEY]
> Work in log space from the start. Probabilities of real utterances are far below the smallest float64, so multiply by adding logs and sum with `logsumexp`.

## Align with Viterbi

The forward score sums over every path, and it does not say which frames belong to which room. Alignment asks for the single best state sequence, which is the job of the **Viterbi algorithm**. Replace the sum in the forward pass with a maximum, and remember which predecessor won:

$$
\begin{aligned}
\delta_1(j) &= \pi_j\, b_j(o_1) \\
\delta_{t+1}(j) &= \max_i \big[\delta_t(i)\, a_{ij}\big]\, b_j(o_{t+1}) \\
\psi_{t+1}(j) &= \operatorname*{arg\,max}_i \big[\delta_t(i)\, a_{ij}\big]
\end{aligned}
$$

Read it aloud: $\delta_t(j)$ is the probability of the best path that ends in state $j$ at frame $t$, and $\psi_t(j)$ records the state it came from. At the end, take the best final state and follow $\psi$ backward. In log space the products become sums, and no log-sum-exp is needed.

In the toy, $\delta_1 = [0.3, 0.04]$, $\delta_2 = [0.084, 0.027]$, and $\delta_3 = [0.00588, 0.01512]$. State 1 wins at frame 3, and its best predecessor was state 0, since $0.084 \times 0.3 = 0.0252$ beats $0.027 \times 0.6 = 0.0162$. The best path is $(0, 0, 1)$ with probability 0.01512.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="The same trellis with the Viterbi value delta in each node, one kept arrow per node, and the best path shaded">
<defs><marker id="c021-arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="150.0" y="40.0" class="fig-small" text-anchor="middle">frame 1, symbol 0</text>
<text x="310.0" y="40.0" class="fig-small" text-anchor="middle">frame 2, symbol 1</text>
<text x="470.0" y="40.0" class="fig-small" text-anchor="middle">frame 3, symbol 2</text>
<text x="30.0" y="105.0" class="fig-small" text-anchor="start">state 0</text>
<text x="30.0" y="205.0" class="fig-small" text-anchor="start">state 1</text>
<line x1="180.0" y1="100.0" x2="280.0" y2="100.0" class="stroke-ink" marker-end="url(#c021-arrow2)"/>
<line x1="180.0" y1="100.0" x2="280.0" y2="200.0" class="stroke-ink" marker-end="url(#c021-arrow2)"/>
<line x1="340.0" y1="100.0" x2="440.0" y2="100.0" class="stroke-ink" marker-end="url(#c021-arrow2)"/>
<line x1="340.0" y1="100.0" x2="440.0" y2="200.0" class="stroke-ink" marker-end="url(#c021-arrow2)"/>
<circle cx="150.0" cy="100.0" r="30" class="fill-soft-b stroke-b"/>
<text x="150.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.3</text>
<circle cx="150.0" cy="200.0" r="30" class="fill-sunk stroke-muted"/>
<text x="150.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.04</text>
<circle cx="310.0" cy="100.0" r="30" class="fill-soft-b stroke-b"/>
<text x="310.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.084</text>
<circle cx="310.0" cy="200.0" r="30" class="fill-sunk stroke-muted"/>
<text x="310.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.027</text>
<circle cx="470.0" cy="100.0" r="30" class="fill-sunk stroke-muted"/>
<text x="470.0" y="105.0" class="fig-bold fig-small" text-anchor="middle">0.00588</text>
<circle cx="470.0" cy="200.0" r="30" class="fill-soft-b stroke-b"/>
<text x="470.0" y="205.0" class="fig-bold fig-small" text-anchor="middle">0.01512</text>
<text x="300.0" y="270.0" class="fig-bold" text-anchor="middle">Best path: state 0, state 0, state 1, with probability 0.01512</text>
</svg>
Figure 4. The same trellis, with the Viterbi value delta in each node. Each node keeps only the arrow from its best predecessor, and the shaded nodes are the best path. Notice that 0.01512 is smaller than P(O) = 0.03628, because P(O) also adds the other paths.
```

That last remark is a general fact: the best path is one term of the sum that defines $P(O \mid \lambda)$, so $\log P(\text{best path}) \le \log P(O \mid \lambda)$, with equality only when one path carries all the probability. The following demo draws the trellis for a model you can change.

```demo
{"id": "viterbi-trellis", "caption": "Drag S1's self-loop a11 up from 0.05 and watch the best path stay in S1 for more frames, then compare log P(best path) with log P(O), which sums over every path."}
```

## Use it to recognize a word

To recognize an utterance, give each word its own **left-to-right** HMM: the model starts in state 0, and each state can only stay put or move one state to the right. Run the forward algorithm under every word's model and answer with the word whose $\log P(O \mid \lambda_w)$ is highest. The exercise uses two states per phone plus one for each silence, so "osa" gets eight states. Even with models that nobody has trained, built by splitting four takes of each word into equal chunks, the Run block recognizes all 40 noisy test words this way. The next chapter, [Training HMMs with Baum-Welch](/lessons/023-training-hmms), shows how to improve a model's fit to its data.

> [!KEY]
> The forward algorithm answers "how likely is this sound under this model?" by summing over every alignment. Viterbi answers "which alignment is best?" by keeping only the maximum. Both fill a states-by-frames table in log space.

> [!RECAP]
> - Emissions are `(T, N)` log-likelihoods, here from diagonal Gaussians.
> - Forward: `log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]`, and $\log P(O)$ is the `logsumexp` of the last row.
> - A linear-space pass underflows within a few hundred frames, so every product becomes a sum of logs.
> - Viterbi uses `max` and `argmax` where forward uses `logsumexp`, then follows the stored predecessors back.

## Build it step by step

The checks test the functions in this order. The file provides `flat_start`, the synthetic words, and `mfcc`.

1. **`log_gaussian_diag(X, means, variances)`.** Takes `(T, D)`, `(N, D)`, and `(N, D)`, and returns `(T, N)`. Broadcast to `(T, N, D)` and sum over `D`. The usual mistake is using standard deviations where the formula needs variances.
2. **`forward(log_pi, log_A, log_B)`.** Returns `(log_alpha, log_p)`. Use one `logsumexp` over `axis=0` per frame. The usual mistake is `axis=1`, which sums over destinations instead of sources.
3. **`viterbi(log_pi, log_A, log_B)`.** Keep `delta` of shape `(N,)` and `psi` of shape `(T, N)`, then walk back from the best final state. The usual mistake is backtracking with `psi[t - 1]` instead of `psi[t]`, which shifts the path by a frame.

## Where this breaks

**Frames are not independent.** Neighboring frames of speech are strongly correlated, but the model scores each one alone. Systems appended delta and delta-delta features, 39 dimensions in all.

**Durations are geometric.** A self-loop with probability $a$ keeps the model in a state for a geometrically distributed number of frames, with mean $1/(1-a)$, and one frame is the most likely stay in every state. Chaining several states per phone enforces a minimum duration.

**Whole-word models do not scale.** Each word gets its own states. Large-vocabulary systems share phone models between words, as [Word error rate](/lessons/026-word-error-rate) shows.

## The exercise

You build `log_gaussian_diag`, `forward`, and `viterbi`. Press **Run** to build an HMM for each synthetic word, align a new take of "osa" and see which state owns each frame, compare $\log P(\text{best path})$ with $\log P(O)$, and score 40 noisy words against all eight models.

The checks compare your densities with `scipy.stats.multivariate_normal`, forward with the toy worked above and with a scaled linear-space reference over 2,000 frames, and Viterbi with the toy and a brute-force search over 729 paths. They also confirm that forward survives a forbidden transition and that the best path never beats the sum over all paths.
