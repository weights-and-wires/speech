---
title: Hidden Markov models
slug: 13-hidden-markov-models
part: "Part 3 — Classical recognition"
summary: Model a word as hidden states that emit frames, then score, align, and train it with forward, Viterbi, and Baum-Welch in log space.
minutes: 80
gpu: false
objectives:
  - Compute diagonal Gaussian and Gaussian mixture log-likelihoods for every frame and state.
  - Run the forward, backward, and Viterbi algorithms in log space, and check them on a toy HMM worked by hand.
  - Re-estimate an HMM with Baum-Welch and confirm that the likelihood never falls.
  - Build an isolated-word recognizer with one left-to-right HMM per word.
prereqs:
  - 12-dynamic-time-warping
exercise: 13-hidden-markov-models
---

# Hidden Markov models

A template recognizer from *Dynamic time warping* keeps whole recordings and
compares against each one. It learns nothing about which parts of a word vary
and which parts don't. A second speaker, a different microphone, or a slightly
different vowel all look equally far from the template.

A **hidden Markov model** (HMM) replaces the recording with a small model.
Picture a word as a row of rooms that you walk through from left to right. In each room you linger for a while, and every 10 ms you make a
sound drawn from that room's own distribution. You hear only the sounds; the
rooms are hidden. An HMM gives the probability of any sequence of sounds, and
it can say which room produced each one.

From the 1980s until deep learning, almost every speech recognizer was built
from HMMs. In this lesson you build the three algorithms that make them work:
the forward algorithm to score a word, Viterbi to align it, and Baum-Welch to
train it. You finish with a recognizer that trains one HMM per word.

## What you already know

- MFCCs describe one 10 ms frame with 13 numbers. For more information, see
  *MFCCs and normalization*.
- A Gaussian density $\mathcal{N}(x; \mu, \sigma^2)$ is the bell curve with
  mean $\mu$ and variance $\sigma^2$.
- Dynamic time warping fills a table where each cell extends the best path to
  one of its neighbors.
- `scipy.special.logsumexp(v)` computes $\log \sum_i e^{v_i}$ without
  overflow.

## States, transitions, and emissions

An HMM with $N$ states has three sets of parameters, written together as
$\lambda = (\pi, A, B)$:

- The **initial probabilities** $\pi_i$: the chance of starting in state $i$.
- The **transition probabilities** $a_{ij} = P(q_{t+1} = j \mid q_t = i)$:
  the chance of moving from state $i$ to state $j$ between frames. Here $q_t$
  is the state at frame $t$. Each row of $A$ sums to 1.
- The **emission densities** $b_j(o_t) = p(o_t \mid q_t = j)$: how likely
  state $j$ is to produce the observation $o_t$, the feature vector of
  frame $t$.

It's *Markov* because the next state depends only on the current one, and
*hidden* because you observe $o_1, \dots, o_T$ but never $q_1, \dots, q_T$.

For speech, the emission is a density over 13-dimensional MFCC vectors. The
simplest choice is one Gaussian per state with a **diagonal covariance**, one
variance per feature and no correlations between features:

$$
\log b_j(o) = -\frac{1}{2}\left[D \log 2\pi + \sum_{d=1}^{D} \log \sigma_{jd}^2 + \sum_{d=1}^{D} \frac{(o_d - \mu_{jd})^2}{\sigma_{jd}^2}\right]
$$

Read it aloud: a constant for the $D$ dimensions, plus a penalty for wide
variances, plus the squared distance from the mean in units of each feature's
variance, all times minus one half. The exercise computes every frame against
every state at once, an array of shape `(T, N)`:

```python
diff2 = (X[:, None, :] - means[None, :, :]) ** 2 / variances[None, :, :]
log_B = -0.5 * (D * np.log(2 * np.pi) + np.log(variances).sum(1)[None, :] + diff2.sum(2))
```

### Gaussian mixture emissions

One Gaussian can't describe a sound that comes in several shapes, such as a
vowel said by men and women. Classic recognizers gave each state a
**Gaussian mixture model** (GMM), a weighted sum of $M$ Gaussians:

$$
b_j(o) = \sum_{m=1}^{M} c_{jm}\,\mathcal{N}(o;\ \mu_{jm},\ \operatorname{diag}\sigma^2_{jm})
$$

Read it aloud: the density of state $j$ is a blend of $M$ bell curves, where
component $m$ gets weight $c_{jm}$ and the weights sum to 1. Compute it in
log space, adding each component's log weight to its log density and
combining with `logsumexp` over components. Exponentiating first fails: a
frame far from every mean has densities like $e^{-1000}$, which round to 0.

A GMM trains with the same idea as Baum-Welch, later in this lesson: each
frame is shared among components in proportion to
$c_{jm}\mathcal{N}(o_t; \mu_{jm}, \sigma^2_{jm})$, and each component is refit
to its share. Systems started with one Gaussian per state, then split each
one in two by nudging its mean by about $\pm 0.2\sigma$, a step called
*mixing up*.

## Score a sequence with the forward algorithm

The likelihood of the observations, $P(O \mid \lambda)$, sums over every
possible state sequence. There are $N^T$ of those, but the same trick as DTW
applies. Define the **forward variable** $\alpha_t(j)$, the probability of the
first $t$ observations and ending in state $j$ at frame $t$:

$$
\begin{aligned}
\alpha_1(j) &= \pi_j\, b_j(o_1) \\
\alpha_{t+1}(j) &= \Big[\sum_{i=1}^{N} \alpha_t(i)\, a_{ij}\Big]\, b_j(o_{t+1}) \\
P(O \mid \lambda) &= \sum_{j=1}^{N} \alpha_T(j)
\end{aligned}
$$

Read it aloud: start with the chance of starting in $j$ times the chance that
$j$ emits the first frame. At each later frame, add up every way of arriving
in $j$ from the previous frame, then multiply by the chance that $j$ emits
this frame. At the end, add up over the final states. Equations count frames
from 1; the code counts from 0.

This is DTW with the minimum replaced by a sum and the costs replaced by
probabilities. It takes $O(TN^2)$ time instead of $O(N^T)$.

### Work the forward pass by hand

Take a two-state HMM whose emissions are three discrete symbols, so
$b_j(o)$ is a table lookup:

$$
\begin{aligned}
\pi &= \begin{bmatrix} 0.6 & 0.4 \end{bmatrix}, \qquad
A = \begin{bmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{bmatrix} \\
B &= \begin{bmatrix} 0.5 & 0.4 & 0.1 \\ 0.1 & 0.3 & 0.6 \end{bmatrix}
\end{aligned}
$$

Row $j$ of $B$ is state $j$'s distribution over the symbols 0, 1, and 2.
Observe $O = (0, 1, 2)$, and work the three frames:

- Frame 1: $\alpha_1 = [0.6 \times 0.5,\ 0.4 \times 0.1] = [0.3,\ 0.04]$.
- Frame 2, state 0: $(0.3 \times 0.7 + 0.04 \times 0.4) \times 0.4 = 0.226 \times 0.4 = 0.0904$.
- Frame 2, state 1: $(0.3 \times 0.3 + 0.04 \times 0.6) \times 0.3 = 0.114 \times 0.3 = 0.0342$.
- Frame 3, state 0: $(0.0904 \times 0.7 + 0.0342 \times 0.4) \times 0.1 = 0.007696$.
- Frame 3, state 1: $(0.0904 \times 0.3 + 0.0342 \times 0.6) \times 0.6 = 0.028584$.

So $P(O \mid \lambda) = 0.007696 + 0.028584 = 0.03628$.

### Stay in log space

Each frame multiplies in probabilities below 1, so $\alpha$ shrinks
exponentially. On the exercise's random four-state HMM with 2,000 frames,
$\log P(O) = -2799.5$ nats, which is about $10^{-1216}$. Even the smallest
subnormal float64 is about $5 \times 10^{-324}$, so a linear-space forward
pass returns exactly 0. Store $\log \alpha$ instead, and turn every sum of
products into a log-sum-exp:

$$
\begin{aligned}
\log \alpha_{t+1}(j) &= \operatorname{LSE}_i\big(\log \alpha_t(i) + \log a_{ij}\big) + \log b_j(o_{t+1}) \\
\operatorname{LSE}(v) &= v_{\max} + \log \sum_i e^{v_i - v_{\max}}
\end{aligned}
$$

Read it aloud: add logs where the linear version multiplies, and combine the
incoming paths with log-sum-exp, which subtracts the largest term before
exponentiating so nothing underflows. A forbidden transition has
$\log a_{ij} = -\infty$, which `logsumexp` handles. In numpy:

```python
log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_B[t]
```

> [!KEY]
> Work in log space from the start. Probabilities of real utterances are far
> below the smallest float64, so multiply by adding logs and sum with
> `logsumexp`.

## The backward algorithm

The **backward variable** $\beta_t(i)$ is the probability of the observations
after frame $t$, given state $i$ at frame $t$. It runs from the end:

$$
\begin{aligned}
\beta_T(i) &= 1 \\
\beta_t(i) &= \sum_{j=1}^{N} a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)
\end{aligned}
$$

Read it aloud: at the last frame nothing is left to explain. Earlier, add up
over every next state $j$ the chance of moving there, emitting the next frame,
and explaining the rest. In the toy,
$\beta_2 = [0.25,\ 0.40]$ and $\beta_1 = [0.106,\ 0.112]$.

The product $\alpha_t(i)\beta_t(i)$ is the probability of all the observations
*and* state $i$ at frame $t$. Summing over $i$ gives $P(O)$ at every frame,
which makes a strong test. At frame 1 of the toy,
$0.3 \times 0.106 + 0.04 \times 0.112 = 0.03628$, the same number as before.

## Align with Viterbi

Alignment asks for the single best state sequence rather than the sum over
all of them: which frames belong to which sound. That's the job of the
**Viterbi algorithm**. Replace the sum in the forward pass with a maximum, and remember which predecessor won:

$$
\begin{aligned}
\delta_1(j) &= \pi_j\, b_j(o_1) \\
\delta_{t+1}(j) &= \max_i \big[\delta_t(i)\, a_{ij}\big]\, b_j(o_{t+1}) \\
\psi_{t+1}(j) &= \operatorname*{arg\,max}_i \big[\delta_t(i)\, a_{ij}\big]
\end{aligned}
$$

Read it aloud: $\delta_t(j)$ is the probability of the best path that ends in
state $j$ at frame $t$, and $\psi_t(j)$ records the state it came from. At the
end, take the best final state and follow $\psi$ backward. In log space the
products become sums, and no log-sum-exp is needed.

In the toy, $\delta_1 = [0.3, 0.04]$, $\delta_2 = [0.084, 0.027]$, and
$\delta_3 = [0.00588, 0.01512]$. State 1 wins at frame 3. Its best predecessor
was state 0, since $0.084 \times 0.3 = 0.0252$ beats $0.027 \times 0.6 =
0.0162$. The best path is $(0, 0, 1)$ with probability 0.01512.

The following demo draws the trellis, the grid of states against frames,
with its best path.

```demo
{"id": "viterbi-trellis", "caption": "Drag S1's self-loop a11 up from 0.05 and watch the best path stay in S1 for more frames, then compare log P(best path) with log P(O), which sums over every path."}
```

## Train with Baum-Welch

Nobody labels frames with states, so you can't fit the Gaussians directly.
**Baum-Welch** is the expectation-maximization (EM) algorithm for HMMs. It guesses soft labels from the current model, refits the
model to those soft labels, and repeats. Each round never lowers
$P(O \mid \lambda)$.

The soft labels are two posteriors. The **state occupancy** $\gamma_t(i)$ is
the probability of state $i$ at frame $t$ given all the observations. The
**transition posterior** $\xi_t(i, j)$ is the probability of moving from $i$
at frame $t$ to $j$ at frame $t+1$:

$$
\begin{aligned}
\gamma_t(i) &= \frac{\alpha_t(i)\,\beta_t(i)}{P(O \mid \lambda)} \\
\xi_t(i, j) &= \frac{\alpha_t(i)\, a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j)}{P(O \mid \lambda)}
\end{aligned}
$$

Read it aloud: $\gamma$ is the forward and backward probability through
state $i$ at frame $t$, as a share of the total. $\xi$ is the same for one
transition: arrive in $i$, move to $j$, emit the next frame, and explain the
rest. In the toy, $\gamma_1 = [0.877, 0.123]$ and $\gamma_3 = [0.212, 0.788]$.

The updated parameters are weighted counts:

$$
\begin{aligned}
\hat a_{ij} &= \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)} \\
\hat\mu_j &= \frac{\sum_{t} \gamma_t(j)\, o_t}{\sum_{t} \gamma_t(j)} \\
\hat\sigma^2_j &= \frac{\sum_t \gamma_t(j)\, o_t^2}{\sum_t \gamma_t(j)} - \hat\mu_j^2
\end{aligned}
$$

Read it aloud: the updated transition probability is the expected number of
$i$-to-$j$ moves divided by the expected number of moves out of $i$. The updated
mean is the average frame, with each frame weighted by how much it belongs to
state $j$. The updated variance is the weighted average square minus the square
of the mean. With several training sequences, add up the numerators and
denominators over all of them before dividing. In numpy, with `gamma` of shape
`(T, N)`:

```python
sum_x += gamma.T @ X            # (N, D)
occupancy += gamma.sum(axis=0)  # (N,)
means = sum_x / occupancy[:, None]
```

Re-estimating the toy from its one sequence gives $\hat a_{00} = 0.502$ and
$\hat a_{01} = 0.498$. A transition that starts at 0 has $\xi = 0$ forever, so
Baum-Welch keeps the shape of the model. That's how a left-to-right model
stays left-to-right.

> [!WARNING]
> The transition sums run over $t = 1$ to $T-1$, and so does the denominator
> of $\hat a_{ij}$. Dividing by the occupancy over all $T$ frames makes the
> rows of $\hat A$ sum to less than 1.

> [!KEY]
> Baum-Welch alternates soft alignment ($\gamma$ and $\xi$ from forward and
> backward) with refitting (weighted averages). The likelihood never falls, so
> a drop means a bug.

## Build a word recognizer

For isolated words, give each word its own **left-to-right** HMM: the model
starts in state 0, and each state can only stay put or move one state to the
right. The exercise uses two states per phone, plus one for the leading and
one for the trailing silence, so "osa" gets eight states.

A self-loop with probability $a$ keeps the model in a state for a
geometrically distributed number of frames, with mean $1/(1-a)$. The flat
start sets $a = 0.6$, a mean of 2.5 frames, or 25 ms, and Baum-Welch adjusts
it to fit the data. The flat start also splits each training take into equal
chunks, one per state, to set the initial means and variances.

To recognize an utterance, run the forward algorithm under every word's model
and answer with the word whose $\log P(O \mid \lambda_w)$ is highest. With
four training takes per word and five Baum-Welch steps, this recognizes all
40 of the exercise's noisy test words.

> [!RECAP]
> - Emissions are `(T, N)` log-likelihoods: diagonal Gaussians, or mixtures
>   combined with `logsumexp`.
> - Forward: `log_alpha[t] = logsumexp(log_alpha[t-1][:, None] + log_A, axis=0) + log_B[t]`.
> - Backward runs from `log_beta[T-1] = 0`, and $\sum_i \alpha_t(i)\beta_t(i)
>   = P(O)$ at every frame.
> - Viterbi uses `max` and `argmax` where forward uses `logsumexp`, then
>   follows the stored predecessors back.
> - Baum-Welch turns $\gamma$ and $\xi$ into weighted counts, summed over
>   every training sequence.

## Build it step by step

The checks test the functions in this order. The file provides `flat_start`,
the synthetic words, and `mfcc`.

1. **`log_gaussian_diag(X, means, variances)`.** Takes `(T, D)`, `(N, D)`,
   and `(N, D)`, and returns `(T, N)`. Broadcast to `(T, N, D)`, sum over
   `D`. Common mistake: using standard deviations where the formula needs
   variances.
2. **`log_gmm(X, weights, means, variances)`.** Reshape the means to
   `(N * M, D)`, call your Gaussian, reshape back to `(T, N, M)`, add
   `np.log(weights)`, and `logsumexp` over the last axis. Common mistake:
   summing densities in linear space.
3. **`forward(log_pi, log_A, log_B)`.** Returns `(log_alpha, log_p)`. One
   `logsumexp` over `axis=0` per frame. Common mistake: `axis=1`, which sums
   over destinations instead of sources.
4. **`backward(log_A, log_B)`.** Fill from the last row, which is 0, down to
   the first, with `logsumexp` over `axis=1`. Common mistake: using
   `log_B[t]` where the formula needs `log_B[t + 1]`.
5. **`viterbi(log_pi, log_A, log_B)`.** Keep `delta` of shape `(N,)` and
   `psi` of shape `(T, N)`. Common mistake: backtracking with `psi[t - 1]`
   instead of `psi[t]`, which shifts the path by a frame.
6. **`baum_welch_step(sequences, log_pi, log_A, means, variances, var_floor)`.**
   For each sequence, run forward and backward, build `gamma` and `xi`, and
   add to running sums. Divide once at the end, and floor the variances with
   `np.maximum`. Return the log-likelihood under the *old* parameters. Common
   mistake: normalizing per sequence and averaging, which weights a short
   take as much as a long one.
7. **`train_word_hmm` and `recognize`.** Call `flat_start`, then
   `baum_welch_step` in a loop. For `recognize`, score with `forward` and
   return the best name. Common mistake: comparing Viterbi scores from one
   model with forward scores from another.

## Cost and scaling

The following table uses $T$ frames, $N$ states, $M$ mixture components, $D$
feature dimensions, and $W$ words.

| Operation | Time | Memory |
|---|---|---|
| Emission log-likelihoods | $O(TNMD)$ | $O(TN)$ |
| Forward, backward, or Viterbi, dense $A$ | $O(TN^2)$ | $O(TN)$ |
| The same, left-to-right $A$ | $O(TN)$ | $O(TN)$ |
| One Baum-Welch step | forward plus backward per sequence | $O(TN^2)$ for $\xi$ |
| Recognizing one word | $W$ forward passes | one model at a time |

A left-to-right model has two nonzero entries per row, so real decoders skip
the dense product. In large systems the emissions dominate: thousands of tied
states, each a mixture of Gaussians, evaluated every 10 ms.

## Where this breaks

**Frames aren't independent.** Neighboring frames of speech are strongly
correlated, but the model scores each one alone. Systems appended delta and
delta-delta features, 39 dimensions in all.

**Durations are geometric.** A self-loop makes one frame the most likely stay
in every state, which is rarely true. Chaining several states per phone
enforces a minimum duration and shapes the distribution.

**EM finds a local optimum.** Baum-Welch climbs from wherever it starts, so a
bad initialization gives a bad model.

**Variances collapse.** A state that owns only a few nearly identical frames
drives its variance toward zero and its likelihood toward infinity. Floor the
variances, as `var_floor` does.

**Whole-word models don't scale.** Large-vocabulary systems share phone
models between words. *Language models and word error rate* shows how they're
composed with a lexicon and a language model.

## The exercise

You build `log_gaussian_diag` and `log_gmm` for the emissions, `forward`,
`backward`, and `viterbi` in log space, `baum_welch_step` for Gaussian HMMs,
and the recognizer pieces `train_word_hmm` and `recognize`. Press **Run** to
train one HMM per synthetic word, see the log-likelihood per frame rise with
every iteration, see the Viterbi alignment of an unseen take of "osa", and
recognize 40 noisy words.

The checks compare your densities with `scipy.stats.multivariate_normal`,
and forward, backward, and Viterbi with the toy worked earlier. They run
forward over 2,000 frames against a scaled linear-space reference, and
Viterbi against a brute-force search over 729 paths. They confirm that one
Baum-Welch step on a one-state model gives the sample mean and variance, and
that eight steps never lower the likelihood or break a left-to-right model.
The last check requires 36 of 40 noisy words to be recognized.
