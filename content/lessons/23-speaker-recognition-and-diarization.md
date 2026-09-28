---
title: Speaker recognition and diarization
slug: 23-speaker-recognition-and-diarization
part: "Part 4 — Neural speech recognition"
summary: Train speaker embeddings with an angular margin, score them with EER and minDCF, find who spoke when, and align a transcript to frames.
minutes: 75
gpu: false
objectives:
  - Implement the additive angular margin softmax, including its fallback past pi minus the margin, and explain why the margin tightens clusters.
  - Compute the equal error rate and the normalized minimum detection cost from trial scores.
  - Cluster segment embeddings spectrally, with the number of speakers read off the eigengap of the normalized Laplacian.
  - Compute the diarization error rate with the best one-to-one speaker mapping.
  - Align a known transcript to CTC emissions by Viterbi over the blank-extended sequence, and read token times off the path.
prereqs:
  - 13-hidden-markov-models
  - 16-connectionist-temporal-classification
  - 21-self-supervised-speech
exercise: 23-speaker-recognition-and-diarization
---

# Speaker recognition and diarization

A recognizer answers "what was said?" A bank's phone line asks instead "is
this the account holder?", a meeting recorder asks "who said which part?", and
a subtitle tool that already has the words asks "when was each one said?"

The first two questions are about voices, not words. The standard answer maps
any stretch of speech to a fixed-size vector, a **speaker embedding**, so that
two clips of the same person land close together and clips of different people
land far apart. Verification compares two embeddings. Diarization embeds short
segments of a recording and clusters them.

The third question is **forced alignment**: find the single best path through
the CTC lattice that spells the known transcript, and read each token's frames
off it. In this lesson you build all three, and the metrics that score them.

## What you already know

- Cross-entropy over a softmax trains a classifier, and cosine similarity
  compares directions, as in Self-supervised speech.
- k-means groups points by nearest center, from the same lesson.
- CTC scores a transcript by summing over paths through the blank-extended
  sequence, as in Connectionist temporal classification.
- Viterbi finds the best path through a trellis by replacing a sum with a max
  and keeping backpointers, as in Hidden Markov models.
- An eigenvector $v$ of a symmetric matrix $L$ satisfies $L v = \lambda v$.

## From a voice to a vector

A speaker embedding network reads frames of log-mel features and pools them
over time into one vector. The x-vector design runs dilated 1-D convolutions
over frames, then a **statistics pooling** layer concatenates the mean and the
standard deviation of every channel over time, then a few dense layers give
the embedding. ECAPA-TDNN adds attention to the pooling, so the network can
weight the frames that say most about the voice.
`speechbrain/spkrec-ecapa-voxceleb` reports a 0.80% equal error rate on the
cleaned VoxCeleb1 test set.

Training treats every training speaker as a class. At test time only the
embedding is kept, because the test speakers were never seen. So the loss must
do more than separate the training speakers: it must pack each one tightly,
leaving room for unseen speakers in between.

## The additive angular margin

Scale each embedding $x$ and each class center $W_j$ to unit length, so the
logit of class $j$ is $\cos\theta_j$, the cosine of the angle between them.
The **additive angular margin** softmax, also called ArcFace, adds a margin $m$
to the true class's angle before the softmax:

$$
\mathcal{L} = -\log \frac{e^{s\cos(\theta_y + m)}}{e^{s\cos(\theta_y + m)} + \sum_{j \ne y} e^{s\cos\theta_j}}
$$

Read it aloud: $\theta_y$ is the angle to the true speaker's center, $m$ is an
extra angle in radians, and $s$ is a scale that stretches cosines, which lie in
$[-1, 1]$, into logits large enough for a sharp softmax. The loss is ordinary
cross-entropy, except the true class is scored as if the embedding were $m$
radians farther away than it is. To win, an embedding has to beat every other
center by at least that margin.

Compute the shifted cosine from the identity
$\cos(\theta + m) = \cos\theta\cos m - \sin\theta\sin m$, with
$\sin\theta = \sqrt{1 - \cos^2\theta}$, so you never call `arccos`. In torch:

```python
cos = normalize(x, dim=1) @ normalize(W, dim=1).T
cos_t = cos[rows, y]
shifted = cos_t * math.cos(m) - torch.sqrt(torch.clamp(1 - cos_t**2, min=0)) * math.sin(m)
```

Once $\theta_y + m$ passes $\pi$, $\cos(\theta_y + m)$ starts rising again, so
the loss would reward moving away from the true center. Where
$\cos\theta_y \le \cos(\pi - m)$, use $\cos\theta_y - m\sin m$ instead, which
keeps falling as the angle grows.

### Work the margin by hand

Take two speakers, $s = 30$, and an embedding with cosine 0.5 to its own
center ($\theta = 1.0472$ rad) and 0.4 to the other.

- Without a margin, the logits are 15 and 12. The softmax gives the true class
  0.9526, and the loss is 0.0486 nats.
- With $m = 0.2$, the true cosine becomes
  $0.5 \times 0.9801 - 0.8660 \times 0.1987 = 0.3180$, and its logit is 9.54.
  The softmax gives it 0.0787, and the loss is 2.54 nats.

An embedding that plain softmax calls done still pays 2.54 nats under the
margin. Its loss falls toward zero only when its angle to its own center is at
least 0.2 rad smaller than its angle to any other.

> [!KEY]
> The angular margin makes the true class win by an angle $m$, not merely win.
> That packs each training speaker's embeddings tightly, which is what lets
> unseen speakers separate at test time.

The Run makes 70 synthetic voices, each a buzz at its own pitch through
formant filters scaled to its own vocal tract, and trains a linear embedding on
60 of them. On the 10 unseen voices, the raw spectra give an 11.4% equal error
rate, training with $m = 0$ gives 2.2%, and training with $m = 0.2$ gives
0.7%.

## Scoring trials

A **trial** is a pair of clips and a question: same speaker or not? The score
is the cosine between their embeddings. A **target** trial is a same-speaker
pair, and a **non-target** trial is a different-speaker pair. Accept a trial
when its score is at least a threshold $\tau$. Two errors trade against each
other as $\tau$ moves:

- A **miss**: a target trial scored below $\tau$. $P_{\text{miss}}(\tau)$ is
  the fraction of target scores below it.
- A **false alarm**: a non-target trial scored at or above $\tau$.
  $P_{\text{fa}}(\tau)$ is the fraction of non-target scores at or above it.

The **equal error rate** (EER) is the error where the two rates meet. The
exercise tries every distinct score as $\tau$, and returns
$(P_{\text{miss}} + P_{\text{fa}})/2$ where they are closest.

The **detection cost** weights the errors by how costly and how common they
are:

$$
C(\tau) = \frac{C_{\text{miss}}\,P_{\text{miss}}(\tau)\,P_{\text{tar}} + C_{\text{fa}}\,P_{\text{fa}}(\tau)\,(1 - P_{\text{tar}})}{\min\left(C_{\text{miss}} P_{\text{tar}},\ C_{\text{fa}} (1 - P_{\text{tar}})\right)}
$$

Read it aloud: $P_{\text{tar}}$ is the prior probability that a trial is a
target, and $C_{\text{miss}}$ and $C_{\text{fa}}$ are the costs of each
error. The denominator is the cost of the better of two trivial systems,
accept everything or reject everything, so 1.0 means no better than guessing.
**minDCF** is the minimum over thresholds. VoxSRC uses $P_{\text{tar}} = 0.05$
with both costs 1, so the formula becomes $P_{\text{miss}} + 19\,P_{\text{fa}}$.
In numpy, over a vector of thresholds:

```python
p_miss = np.mean(target[None, :] < thresholds[:, None], axis=1)
p_fa = np.mean(nontarget[None, :] >= thresholds[:, None], axis=1)
```

### Work the metrics by hand

Take target scores 0.9, 0.8, 0.7, and 0.4, and non-target scores 0.6, 0.3,
0.2, and 0.1. At $\tau = 0.6$, one target (0.4) is missed and one non-target
(0.6) is accepted, so both rates are 25% and the EER is 25%. For minDCF, at
$\tau = 0.7$ the miss rate is 25% and the false-alarm rate is 0, so the cost
is $0.25 + 19 \times 0 = 0.25$. No threshold does better, so minDCF is 0.25.

> [!TIP]
> Report minDCF with its $P_{\text{tar}}$. A system tuned for 0.05 can look
> poor at the 0.01 some NIST evaluations use, because false alarms then cost
> 99 times as much as misses.

## Who spoke when

Diarization labels a recording with speaker turns, without knowing the
speakers in advance. The classic pipeline cuts speech into short overlapping
windows, embeds each, and clusters the embeddings.

**Spectral clustering** turns the clustering into a graph problem. Build an
**affinity matrix** $A$ with one row and column per segment: the cosine
between their embeddings, with negative values set to 0. Keep only each row's
largest entries (the exercise keeps the top 30%), which cuts weak links
between speakers, then symmetrize with $\max(A, A^\top)$. Then form the
**normalized Laplacian**:

$$
L = I - D^{-1/2} A D^{-1/2}
$$

Read it aloud: $D$ is the diagonal matrix of the row sums of $A$, so each
entry of $A$ is divided by the square root of both its row's and its column's
total. $L$'s eigenvalues lie between 0 and 2. A graph that falls into $k$
disconnected groups has exactly $k$ eigenvalues equal to 0. With noise, it has
$k$ small eigenvalues and then a jump, the **eigengap**. So pick the $k$ that
maximizes $\lambda_{k+1} - \lambda_k$. Then take the eigenvectors of the $k$
smallest eigenvalues as the columns of an $N \times k$ matrix, scale each row
to unit length, and run k-means on the rows. In numpy:

```python
d = 1 / np.sqrt(A.sum(axis=1))
L = np.eye(len(A)) - d[:, None] * A * d[None, :]
eigenvalues, vectors = np.linalg.eigh(L)
```

Work a tiny case. Four segments in two pairs, with affinity 1 inside each pair
and 0 across, give row sums of 2, so $D^{-1/2} A D^{-1/2} = A/2$. Its
eigenvalues are 1, 1, 0, and 0, so $L$'s are 0, 0, 1, and 1. The gaps are 0,
1, and 0, and the largest is at $k = 2$: two speakers.

### Diarization error rate

The **diarization error rate** (DER) counts three kinds of error over time:

$$
\text{DER} = \frac{\text{missed speech} + \text{false alarm} + \text{speaker confusion}}{\text{total reference speech}}
$$

Read it aloud: missed speech is reference speech the system called silence, a
false alarm is the reverse, and confusion is speech given to the wrong speaker.
Cluster labels are arbitrary, so before counting confusion, map each system
speaker to at most one reference speaker, choosing the mapping that matches the
most time. `scipy.optimize.linear_sum_assignment` finds it from the table of
overlaps. The exercise scores 10 ms frames with no forgiveness collar at
boundaries, one speaker per frame.

Work 12 frames by hand. The reference is `0 0 0 0 1 1 1 1 1 1 - -` and the
system says `5 5 5 - 5 7 7 7 7 7 7 -`, where `-` is silence. Frame 3 is
missed, and frame 10 is a false alarm. Mapping 5 to 0 and 7 to 1 matches 8 of
the 9 frames where both have speech, so frame 4 is confused. The DER is
$(1 + 1 + 1)/10 = 30\%$.

The Run plays three unseen synthetic voices taking nine turns, embeds 0.5 s
windows every 0.25 s, and clusters them. The eigengap finds 3 speakers in 43
windows, and the DER is 2.1%. Every error lies within 0.12 s of a turn
boundary, where a window straddles two voices.

## Forced alignment with CTC

Given CTC emissions, per-frame log-probabilities $\log y^t_k$ from an acoustic
model, and a known transcript $l$, forced alignment asks for the single most
likely path that spells $l$. Build the blank-extended sequence
$l' = (\varnothing, l_1, \varnothing, l_2, \ldots, l_L, \varnothing)$ of
length $S = 2L + 1$, as for the CTC loss. Run the forward recursion with a
maximum in place of the sum:

$$
\delta_t(s) = \log y^t_{l'_s} + \max\left(\delta_{t-1}(s),\ \delta_{t-1}(s-1),\ \delta_{t-1}(s-2)\right)
$$

Read it aloud: the best path into state $s$ at frame $t$ scores this frame's
log-probability of that state's symbol, plus the best of three ways to arrive:
stay, step forward one state, or skip a blank. The skip
from $s - 2$ is allowed only when $l'_s$ is not blank and differs from
$l'_{s-2}$, the same rule as the CTC loss: two identical letters need a blank
between them. Start in state 0 or 1, end in state $S-1$ or $S-2$, keep a
backpointer to the choice at every cell, and walk back from the better end.

Odd states belong to tokens: state $2k + 1$ is token $k$. The frames spent in
that state are the token's span, and at 20 ms per frame they give times.

### Work three frames by hand

Align `a` over three frames whose probabilities of (blank, `a`) are
(0.6, 0.4), (0.3, 0.7), and (0.8, 0.2). Six paths spell `a`: `a__` (0.096),
`_a_` (0.336), `__a` (0.036), `aa_` (0.224), `_aa` (0.084), and `aaa` (0.056).
The CTC loss sums them, 0.832. Forced alignment keeps the best, `_a_`, with
probability $0.6 \times 0.7 \times 0.8 = 0.336$ and states $(0, 1, 2)$: `a`
occupies frame 1 alone.

> [!KEY]
> Forced alignment is Viterbi on the CTC lattice of one known transcript. The
> path's odd states give each token's frames.

> [!RECAP]
> - AAM softmax replaces the true cosine with
>   $\cos\theta\cos m - \sin\theta\sin m$, or $\cos\theta - m\sin m$ past
>   $\pi - m$, then applies cross-entropy to $s$ times the cosines.
> - EER is where miss and false-alarm rates meet; minDCF is the minimum of the
>   weighted cost, divided by the better trivial system's cost.
> - Spectral clustering prunes and symmetrizes cosine affinities, reads $k$
>   off the Laplacian's eigengap, and clusters eigenvector rows.
> - DER counts misses, false alarms, and confusion after the best one-to-one
>   speaker mapping.
> - Forced alignment is the CTC forward recursion with max, backpointers, and
>   the same skip rule.

## Build it step by step

The checks test the functions in this order.

1. **`aam_softmax_loss(embeddings, weights, labels, scale, margin)`.** Takes
   torch tensors and returns a scalar tensor. Clone the cosine matrix and write
   the shifted value into `logits[rows, labels]`. Common mistake: applying the
   margin to every class, not only the true one.
2. **`equal_error_rate(target_scores, nontarget_scores)`.** Returns a float.
   Use the distinct scores plus `np.inf` as thresholds. Common mistake:
   counting a score equal to the threshold as a miss.
3. **`min_dcf(target_scores, nontarget_scores, p_target, c_miss, c_fa)`.**
   Returns a float. Common mistake: forgetting the normalization.
4. **`affinity_matrix(embeddings, keep)`.** Returns `(N, N)`. Use
   `np.argsort(-sim, axis=1)` to find each row's top `ceil(keep * N)` entries.
   Common mistake: symmetrizing before pruning.
5. **`speaker_count(affinity, max_speakers)`.** Returns an int from
   `np.linalg.eigvalsh`. Common mistake: an off-by-one, since the gap after the
   first eigenvalue means one speaker.
6. **`spectral_cluster(embeddings, max_speakers, keep)`.** Returns `(N,)`
   labels. Use `np.linalg.eigh` and the provided `kmeans_rows`. Common
   mistake: clustering the eigenvector columns instead of the rows.
7. **`diarization_error_rate(reference, hypothesis)`.** Returns a float.
   Common mistake: mapping two system speakers to one reference speaker.
8. **`forced_align(log_probs, tokens, blank)`.** Returns `(states, score)`.
   Fill a `(T, S)` table of `-np.inf`, and raise `ValueError` when both end
   states stay at `-np.inf`. Common mistake: allowing the skip between two
   identical letters.

## Cost and scaling

The following table lists costs for $N$ embeddings or segments, $C$ training
speakers, $n$ trials, $T$ frames, and $L$ tokens.

| Piece | Time | Memory |
|---|---|---|
| AAM loss for a batch | $N C D$ | an $(N, C)$ logit matrix |
| EER and minDCF by sorting | $n \log n$ | the scores |
| Spectral clustering | $N^3$ for the eigendecomposition | an $(N, N)$ affinity |
| DER mapping | $K^3$ for $K$ speakers, plus one pass over frames | a $(K, K)$ table |
| Forced alignment | $T L$ | a $(T, 2L + 1)$ backpointer table |

The $N^3$ eigendecomposition dominates long recordings: an hour in 0.25 s
steps is 14,400 segments, so long-form systems cluster in chunks. Forced
alignment of a whole audiobook needs a table in the gigabytes, so tools align
chapter by chapter.

## Where this breaks

**Overlapping speech.** A clustering pipeline gives each window one speaker,
so two people talking at once count as missed speech. End-to-end diarization
models, such as EEND and pyannote's segmentation model, predict several
speakers per frame.

**Short segments.** An embedding from 0.5 s is noisy, and one from 3 s blurs
turn boundaries. Systems resegment boundaries after clustering.

**The wrong number of speakers.** When a speaker talks only briefly, the
eigengap can miss them. Cap `max_speakers` sensibly, or pass the count when
you know it.

**Peaky CTC timing.** CTC models emit each token as a spike a frame or two
from where a phonetician would put it. Decide whether a word's end extends
into the following blanks, and report the frame rate. The transcript must
also be normalized to the model's vocabulary before alignment.

## The exercise

You build `aam_softmax_loss`, `equal_error_rate`, and `min_dcf` for
verification, then `affinity_matrix`, `speaker_count`, `spectral_cluster`, and
`diarization_error_rate` for diarization, and `forced_align` for alignment.

The checks confirm the worked margin example and the fallback past
$\pi - m$, the EER against the Gaussian closed form $\Phi(-d/2)$, minDCF
against a brute-force sweep, the eigengap on block matrices, exact recovery of
four synthetic speakers, DER against a brute-force search over mappings, and
`forced_align` against enumeration of every path, repeated letters included.
The **Run** trains embeddings on synthetic voices, prints EER and minDCF with
and without the margin, diarizes a conversation you can hear, and aligns
`hello` to emissions from a make-believe acoustic model.
