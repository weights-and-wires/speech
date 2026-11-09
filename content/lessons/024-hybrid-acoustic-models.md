---
title: Hybrid acoustic models
slug: 024-hybrid-acoustic-models
part: "Part 3 — Classical recognition"
summary: Fit a diagonal Gaussian mixture, tie triphone states, and turn posteriors into HMM likelihoods.
minutes: 40
gpu: false
objectives:
  - Fit a diagonal Gaussian mixture with EM and read a log-likelihood that does not fall.
  - Build triphone labels and pick the context question with the largest likelihood gain.
  - Divide frame posteriors by the state priors and decode the scaled likelihoods with Viterbi.
prereqs:
  - 022-hidden-markov-models
  - 023-training-hmms
exercise: 024-hybrid-acoustic-models
---

# Hybrid acoustic models

A recognizer with one Gaussian per state hears the vowel in "cat" and the vowel in "cap" as the same bell. The /ae/ next to /t/ does not sit where the /ae/ next to /p/ sits, and a single mean lands in the gap. Both words then get nearly the same acoustic score.

This chapter gives that state three things one bell does not have. A mixture of bells, so one state can cover two shapes. A triphone, so the vowel keeps its neighbors. And, once a network predicts the state, a division by how common that state is, so a rare phone is not scored like silence. The decoder is still Viterbi. What changes is the emission you hand it.

## What you already know

- One diagonal Gaussian per state scores a frame with the formula in [Hidden Markov models](/lessons/022-hidden-markov-models). Viterbi there finds the best state path. `log_A[i, j]` is the log probability of moving from state $i$ to state $j$.
- Baum-Welch, in [Training HMMs with Baum-Welch](/lessons/023-training-hmms), finds the state shares. This chapter uses those shares and does not derive them again.
- A weighted average of values $x_n$ with weights $r_n$ is $\sum_n r_n x_n / \sum_n r_n$.
- `scipy.special.logsumexp` adds probabilities stored as logs.

## One bell curve for two shapes

Picture the frames of one vowel as dots on a line. Some sit near 0 and some near 6, because the following consonant pulled them apart. One Gaussian puts its mean in the gap and stretches until the bell covers both piles. The center of that bell is a sound that never occurred.

A *Gaussian mixture model* picks a component $k$ with probability $\pi_k$, then draws the frame from that component's own Gaussian. You never see $k$. That is the same expectation-maximization idea as Baum-Welch, with a simpler hidden variable: the component, not the state. Peter Bell's ASR lecture 6 at Edinburgh (29 January 2026) fits the mixture this way. CMU 11-751 taught it on 28 September 2022, after the single Gaussian.

The log density, with one variance per feature and no correlation between features, is a weighted sum of those bells:

$$
\log p(x) = \log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x; \mu_k, \mathrm{diag}(\sigma_k^2))
$$

Read it aloud: weight each bell by $\pi_k$, add them, and take the log. In code the sum is `logsumexp`.

## Fit the mixture

The E-step is the share of frame $n$ in component $k$, the *responsibility* $r_{nk}$:

$$
r_{nk} = \frac{\pi_k \, \mathcal{N}(x_n; \mu_k, \mathrm{diag}(\sigma_k^2))}{\sum_j \pi_j \, \mathcal{N}(x_n; \mu_j, \mathrm{diag}(\sigma_j^2))}
$$

The M-step refits from those shares. $N$ is the number of frames and $N_k = \sum_n r_{nk}$ is the soft count:

$$
\begin{aligned}
\pi_k &= N_k / N \\
\mu_k &= \frac{1}{N_k} \sum_n r_{nk} x_n \\
\sigma_{kd}^2 &= \frac{1}{N_k} \sum_n r_{nk} (x_{nd} - \mu_{kd})^2
\end{aligned}
$$

$\pi_k$ is the soft count divided by $N$, $\mu_k$ is the weighted mean, and the variance is the weighted mean of squared distances. Divide by $N_k$, not by $N_k - 1$. The log-likelihood $\sum_n \log p(x_n)$ is in nats, and an exact M-step does not lower it. A one-hot argmax in place of $r_{nk}$ is the usual way to make it fall.

### Work six frames by hand

The frames are $-1, 0, 1, 5, 6, 7$. The exercise starts a component on the lowest frame and one on the highest, so the means begin at $-1$ and $7$. Both variances begin at the population variance of all six frames, $29/3 \approx 9.667$, and both weights at $1/2$.

One Gaussian on these frames scores $-15.320$ nats. The mixture starts worse, at $-16.414$, because the means sit on the outer frames while the variances are still the wide pooled variance.

For $x = -1$ the log densities are $-2.053$ under the mean at $-1$ and $-5.364$ under the mean at $7$. Add $\log 1/2 = -0.693$ and the components contribute $-2.746$ and $-6.057$. The responsibilities are $0.965$ and $0.035$. The six frames are symmetric around $3$, so each soft count is $3$, the weights stay $1/2$, the means move to $0.462$ and $5.538$, and both variances fall to $3.225$. The log-likelihood rises to $-13.898$.

Figure 1 is the rest of that run. By step 3 the log-likelihood is $-11.456$ nats, and further steps leave it there. The means sit at $0$ and $6$, and each variance is $2/3$, the variance of $\{-1, 0, 1\}$. One bell's best score on these frames remains $-15.320$.

```figure
<svg viewBox="0 0 620 280" role="img" aria-label="Log-likelihood of six frames rising over eight EM steps and then flattening">
  <line x1="80.0" y1="40.0" x2="80.0" y2="220.0" class="stroke-muted"/>
  <line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-muted"/>
  <line x1="80.0" y1="190.0" x2="560.0" y2="190.0" class="stroke-grid"/>
  <text x="72.0" y="194.0" class="fig-small" text-anchor="end">-16</text>
  <line x1="80.0" y1="130.0" x2="560.0" y2="130.0" class="stroke-grid"/>
  <text x="72.0" y="134.0" class="fig-small" text-anchor="end">-14</text>
  <line x1="80.0" y1="70.0" x2="560.0" y2="70.0" class="stroke-grid"/>
  <text x="72.0" y="74.0" class="fig-small" text-anchor="end">-12</text>
  <polyline points="80.0,202.4 140.0,127.0 200.0,64.6 260.0,53.7 320.0,53.7 380.0,53.7 440.0,53.7 500.0,53.7 560.0,53.7" class="stroke-b" fill="none"/>
  <circle cx="80.0" cy="202.4" r="4" class="fill-b stroke-b"/>
  <circle cx="140.0" cy="127.0" r="4" class="fill-b stroke-b"/>
  <circle cx="200.0" cy="64.6" r="4" class="fill-b stroke-b"/>
  <circle cx="260.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <circle cx="320.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <circle cx="380.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <circle cx="440.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <circle cx="500.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <circle cx="560.0" cy="53.7" r="4" class="fill-b stroke-b"/>
  <text x="80.0" y="246.0" class="fig-small" text-anchor="middle">0</text>
  <text x="200.0" y="246.0" class="fig-small" text-anchor="middle">2</text>
  <text x="320.0" y="246.0" class="fig-small" text-anchor="middle">4</text>
  <text x="440.0" y="246.0" class="fig-small" text-anchor="middle">6</text>
  <text x="560.0" y="246.0" class="fig-small" text-anchor="middle">8</text>
  <text x="72.0" y="28.0" class="fig-small" text-anchor="start">log-likelihood (nats)</text>
  <text x="320.0" y="268.0" class="fig-small" text-anchor="middle">EM step</text>
</svg>
Figure 1. Log-likelihood, in nats, of the six frames. It rises from -16.41 to -11.46 and then stays level. The last five steps land on the same point.
```

Entry 0 of the curve the exercise returns is this start, before any update.

```python
log_joint = np.log(weights) + log_gauss  # (N, K)
responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])
```

> [!KEY]
> A Gaussian mixture is a weighted sum of bells. EM refits each bell from the responsibilities, and the log-likelihood does not fall.

> [!TIP]
> Floor each variance, at $10^{-3}$ in this exercise. A component that collapses onto one frame otherwise sends the next log-likelihood to NaN.

## Name the neighbors

A mixture gives one state several shapes. It does not say which shape belongs to which neighbor. A *triphone* label is `left-phone+right`. For `k ae t`, with silence where the string has no neighbor, the labels are `sil-k+ae`, `k-ae+t`, and `ae-t+sil`. The vowel is `ae` as said between `k` and `t`, not a bare `ae`.

Bell's ASR lecture 7 (3 February 2025) gives the count. Forty phones give $40^3 = 64{,}000$ triphones, and a training set misses most of them. A triphone with no frames has no mean, so a new sentence that uses it has nothing to score. Tying is how those unseen labels borrow a Gaussian. The same label function covers a neighbor across a word boundary once that phone is in the string you pass.

## Split on likelihood gain

Tying keeps one Gaussian for every triphone that answers the same yes-or-no question: "Is the left phone a vowel?", "Is the right phone a nasal?" Lecture 7, and the 10 October 2022 lecture of CMU 11-751, grow the tree by picking the question that raises the likelihood the most.

Fit one diagonal Gaussian to the pooled frames and call its total log-likelihood $L(S)$. A question cuts the pool in two. The gain is how much the two Gaussians beat the shared one:

$$
\Delta = L(S_{\mathrm{yes}}) + L(S_{\mathrm{no}}) - L(S)
$$

When the variance floor does not bind, a cluster of $N$ frames in $D$ features has a closed form. $\sigma_d^2$ is the population variance of feature $d$:

$$
L(S) = -\frac{N}{2}\left(D \log 2\pi + \sum_{d=1}^{D} \log \sigma_d^2 + D\right)
$$

The $+D$ is the squared-distance term: at the maximum-likelihood variance it equals $D$. An empty side scores 0, so a question that puts every frame on one side gains 0.

Eight frames show the gap. The values $0.0, 0.2, -0.2, 0.1$ had a vowel on the left, and $3.8, 4.0, 4.2, 3.9$ had a stop. The pool has variance $3.922$ and $L(S) = -16.818$. Each half has variance $0.021875$ and log-likelihood $1.969$. The vowel question gains $1.969 + 1.969 - (-16.818) = 20.757$ nats. A question that cuts each cluster in half gains $0.012$. The vowel question wins.

```python
gain = loglik(frames[yes]) + loglik(frames[~yes]) - loglik(frames)
```

## Divide by the prior

A frame classifier outputs $P(q_t = j \mid o_t)$. Viterbi wants $p(o_t \mid q_t = j)$. Bayes' rule connects them:

$$
p(o \mid q) = \frac{P(q \mid o)\, p(o)}{P(q)}
$$

The frame density $p(o)$ is the same for every state, so it cannot change the path. Drop it. The *scaled likelihood* is the posterior divided by the prior:

$$
\tilde{b}_j(o_t) = \frac{P(q_t = j \mid o_t)}{P(q_t = j)}
$$

CS224S lecture 10 (2025) writes $P(x \mid s) = P(s \mid x)\, P(x) / P(s)$ and drops $P(x)$. Hinton and colleagues, in the November 2012 IEEE Signal Processing Magazine, divide network posteriors by the state frequencies in the alignment. The division matters when labels are unbalanced. Silence is most frames, so a moderate silence posterior is weak evidence.

Priors $0.8$ and $0.2$, and posteriors $0.6$ and $0.4$, prefer state 0 before the division and state 1 after it: $0.6/0.8 = 0.75$ and $0.4/0.2 = 2.0$. Multiplying by the prior pushes the other way.

Four frames with those priors make it a path. The posteriors are $(0.90, 0.10)$, $(0.62, 0.38)$, $(0.62, 0.38)$, $(0.90, 0.10)$, so the scaled likelihoods are $(1.125, 0.5)$, $(0.775, 1.9)$, $(0.775, 1.9)$, $(1.125, 0.5)$. With $\pi = (0.6, 0.4)$ and

$$
A = \begin{bmatrix} 0.7 & 0.3 \\ 0.3 & 0.7 \end{bmatrix}
$$

Viterbi on the scaled likelihoods returns $0, 1, 1, 1$, log probability $-1.720$. Viterbi on the posteriors returns $0, 0, 0, 0$. The last frame's scaled likelihood prefers state 0, and the path stays in state 1 because the self-loop $0.7$ beats the switch. Framewise argmax is not Viterbi.

Pass $\log \tilde{b}_j(o_t)$ to the `viterbi` from [Hidden Markov models](/lessons/022-hidden-markov-models). Do not transpose `log_A`.

> [!WARNING]
> Divide the posterior by the prior. Multiplying by it rewards the common state, and on these four frames the path becomes $0, 0, 0, 0$ instead of $0, 1, 1, 1$.

> [!KEY]
> A network posterior is not an emission likelihood. Divide by the state prior, then decode with the same Viterbi as the single-Gaussian model.

## When the hybrid wins, and when CTC does

The hybrid fits better when a pronunciation dictionary exists and words share phones. A tied state pools every context on the same branch, so a rare triphone borrows frames. Hinton and colleagues report that decoding scaled likelihoods from a network beat the Gaussian system on the tasks they measured, sometimes by a large margin. Their Google Voice Input model scores 7,969 tied states.

The hybrid loses when the dictionary misses a name, because that name has no path. It also loses when the alignment behind the network is wrong: the network learns those labels and cannot grow a state the tree never split off. [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label) trains from the transcript alone, with no dictionary in the sum. CS224S lecture 10 names the dictionary and the alignment recipe as the reason to leave the hybrid. [Neural acoustic models](/lessons/028-neural-acoustic-models) builds the network. This chapter says what to do with its probabilities.

> [!KEY]
> A mixture covers two shapes, a tied triphone covers a context you have barely seen, and dividing by the prior keeps a rare state audible. CTC drops the dictionary and the alignment when those are the part that fails.

> [!RECAP]
> - Start mixture means at evenly spaced ranks of the first feature, and divide variances by the soft count $N_k$.
> - A triphone label is `left-phone+right`. A missing neighbor is the boundary phone.
> - Pick the question whose two Gaussians beat the shared one by the most nats. An empty side scores 0.
> - Scaled likelihoods are posteriors divided by priors. `viterbi` keeps the HMM chapter's `log_A`.

## Build it step by step

1. **`gmm_em(X, n_components, n_steps, variance_floor=1e-3)`** takes `X` of shape `(N, D)` and returns weights `(K,)`, means `(K, D)`, variances `(K, D)`, and log-likelihoods `(n_steps + 1,)`. Entry 0 is the start, before any update. Place component $k$ on the row at rank `round(linspace(0, N - 1, K))` after `np.argsort(X[:, 0], kind="mergesort")`. Copy the floored population variance `X.var(axis=0)` onto every component, and start the weights equal. The usual mistakes are the divisor $N - 1$, and a hard argmax in place of the responsibilities.

2. **`triphones(phones, boundary="sil")`** returns one `f"{left}-{phone}+{right}"` per phone. The missing neighbor on either end is `boundary`. The usual mistake is to use the phone as its own left context.

3. **`count_unseen(train, test)`** counts labels in `test` that are absent from `train`, once per occurrence. Two copies of a missing triphone count as 2.

4. **`best_question_split(X, masks, variance_floor=1e-6)`** takes masks of shape `(N,)` and returns the index of the largest gain, and the gain. Ties go to the smaller index. Use the floored population variance and the sum of log densities, so a floor does not pretend the $+D$ term still holds. An empty side scores 0. Forgetting to subtract $L(S)$ keeps the winning index and gets the gain wrong.

5. **`scaled_likelihoods(posteriors, priors)`** returns `posteriors / priors`, shapes `(T, N)` and `(N,)`. The usual mistake is to multiply.

6. **`viterbi(log_pi, log_A, log_B)`** matches the HMM chapter. Shapes are `(N,)`, `(N, N)`, and `(T, N)`, with `log_A[i, j] = log P(next = j | current = i)`. Return the path `(T,)` and its log probability. Pass `np.log` of the scaled likelihoods as `log_B`.

## Cost and scaling

One EM update grows as frames times components times features. Before tying, 40 phones want on the order of 64,000 triphones, most of them empty. Decoding costs the same Viterbi as the single-Gaussian model. Only the emission table changed.

## Where this breaks

EM can raise the likelihood while both components sit on one cluster. The start here separates them along the first feature, which fails when the clusters differ only in another feature. A rising curve is not proof that every mode was found. The tree can only ask questions you wrote down. A prior of 0 is not a divisor: a state missing from the alignment cannot be scored this way.

## The exercise

You write `gmm_em`, `triphones`, `count_unseen`, `best_question_split`, `scaled_likelihoods`, and `viterbi`. The checks match one EM step on the six frames, require the log-likelihood not to fall, and recover two known bells. They check the "cat" triphones, the unseen count, and the vowel question. Viterbi on the scaled likelihoods must match the HMM chapter's Viterbi, and the posterior-only path must differ.

## Further reading

The numbers and the claims in this chapter come from these sources:

- [Gaussian Mixture Models](https://opencourse.inf.ed.ac.uk/sites/default/files/https/opencourse.inf.ed.ac.uk/asr/2026/asr06-gmms.pdf), Peter Bell, ASR lecture 6, University of Edinburgh, 29 January 2026.
- [Context-dependent phone models](https://www.inf.ed.ac.uk/teaching/courses/asr/2024-25/asr07-cdhmm.pdf), Peter Bell, ASR lecture 7, University of Edinburgh, 3 February 2025.
- [ASR with neural nets](https://web.stanford.edu/class/cs224s/semesters/2025-spring/lecture-slides/224s.25.lec10.pdf), Andrew Maas, CS224S lecture 10, Stanford University, 2025.
- [Speech Recognition and Understanding (11-751)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, 2022. The 28 September lecture covers the single Gaussian, the GMM, and EM. The 10 October lecture covers phonetic decision trees. The 24 October lecture covers deep networks for acoustic modeling.
- [Deep Neural Networks for Acoustic Modeling in Speech Recognition](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/HintonDengYuEtAl-SPM2012.pdf), Geoffrey Hinton and colleagues, IEEE Signal Processing Magazine, November 2012.
