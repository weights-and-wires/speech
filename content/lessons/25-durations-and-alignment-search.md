---
title: Durations and alignment search
slug: 25-durations-and-alignment-search
part: "Part 5 — Speech synthesis"
summary: Replace attention with durations, expand phones to frames, add pitch and energy targets, and find alignments with a dynamic program.
minutes: 65
gpu: false
objectives:
  - Expand a padded batch of phone vectors to frames with the FastSpeech length regulator, and change speaking rate with a speed factor.
  - Compute per-phone pitch and energy targets, and quantize them into the 256 bins FastSpeech 2 embeds.
  - Derive monotonic alignment search as a dynamic program, and verify it against brute-force enumeration.
  - Explain where durations come from in FastSpeech, FastSpeech 2, and Glow-TTS.
prereqs:
  - 24-text-to-speech-and-tacotron
  - 12-dynamic-time-warping
  - 10-pitch-tracking
exercise: 25-durations-and-alignment-search
---

# Durations and alignment search

Tacotron learns when to read each symbol with attention, and attention is
fragile. It skips words, repeats them, and stops early, most of all on long
sentences. It also forces decoding into a loop of one frame at a time.

FastSpeech removes attention by making timing explicit. A small network
predicts how many frames each phone lasts, its *duration*. Then each phone's
vector is copied that many times, and every frame is predicted at once, in
parallel. The reported speedup over an autoregressive teacher was 270 times
for the mel spectrogram.

That moves the hard problem to training: where do the true durations come
from? In this lesson you build the length regulator that uses durations, the
pitch and energy targets that FastSpeech 2 adds beside them, and monotonic
alignment search, the dynamic program with which Glow-TTS and VITS find
durations on their own.

## What you already know

- Tacotron's attention decides which symbol each output frame reads, and it can skip or repeat (lesson 24, *Text to speech and Tacotron*).
- Dynamic time warping fills a table of best partial costs and backtracks the best path (lesson 12, *Dynamic time warping*).
- YIN gives an F0 estimate and a voicing decision for each frame (lesson 10, *Pitch tracking*).
- A diagonal Gaussian's log-density is a sum over dimensions of $-\tfrac12 \log 2\pi - \log\sigma - (z - \mu)^2 / 2\sigma^2$.
- An STFT magnitude column describes one frame's spectrum (lesson 4, *The short-time Fourier transform*).

## The length regulator

The length regulator is a photocopier. Given phone vectors $h_1, \ldots, h_L$
and whole-number durations $d_1, \ldots, d_L$, it outputs $h_1$ repeated $d_1$
times, then $h_2$ repeated $d_2$ times, and so on. The output has
$N = \sum_i d_i$ frames.

It helps to write the copy as a matrix. The *alignment matrix*
$A \in \{0, 1\}^{L \times N}$ has a 1 where frame $n$ belongs to phone $i$:

$$
A_{i n} = \mathbb{1}\left[\,e_i - d_i \le n < e_i\,\right], \qquad
e_i = \sum_{k=1}^{i} d_k, \qquad
\tilde H = A^\top H.
$$

Read it aloud: $e_i$ is the cumulative sum of durations, the frame where phone
$i$ ends. Phone $i$ owns the frames from its start, $e_i - d_i$, up to but
not including its end. Multiplying by $A^\top$ copies each phone's row of $H$
into the frames it owns. In torch, for a batch:

```python
ends = torch.cumsum(durations, dim=1)
A = ((ends - durations)[..., None] <= n) & (n < ends[..., None])
frames = torch.einsum("bln,bld->bnd", A.float(), h)
```

### Work one example by hand

Take three phones with durations $d = [2, 1, 3]$. The cumulative sums are
$e = [2, 3, 6]$, so the starts are $[0, 2, 3]$ and $N = 6$:

$$
A = \begin{bmatrix} 1 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 1 & 1 \end{bmatrix},
\qquad \tilde H = [h_1, h_1, h_2, h_3, h_3, h_3].
$$

The matrix form has two advantages over a loop of `repeat_interleave` calls.
It works on a padded batch in one call: a padded phone has duration 0, so it
owns no frames. And gradients flow back to $H$: phone $i$ receives the sum of
the gradients of the $d_i$ frames that copied it.

> [!KEY]
> A duration model turns alignment into bookkeeping: cumulative sums of the
> durations say which frames each phone owns, and every frame can then be
> predicted in parallel.

### Durations at inference, and speed

A *duration predictor*, two 1-D convolutions and a linear layer, predicts one
number per phone. It is trained with mean squared error on $\log(d + 1)$,
because durations are skewed: most phones last a few frames, and a few last
dozens. At inference you invert the log, scale by a speed factor $\alpha$,
and round:

$$
\hat d_i = \max\left(1,\ \operatorname{round}\left(\alpha\,(e^{\hat \ell_i} - 1)\right)\right).
$$

Read it aloud: exponentiate the predicted log duration $\hat\ell_i$, subtract
the 1 that the target added, and multiply by $\alpha$. Round to whole frames,
and give every phone at least one frame so no phone vanishes. A factor
$\alpha > 1$ slows speech down. For example, $d = [4, 1, 6]$ with
$\alpha = 1.3$ gives $5.2, 1.3, 7.8$, which rounds to $[5, 1, 8]$.

## Where durations come from

The length regulator needs true durations during training. Three systems got
them in three ways:

- **FastSpeech** trained an autoregressive Transformer TTS teacher first. It picked the attention head with the highest *focus rate*, the average over decoder steps of the largest attention weight, and counted how many decoder steps chose each phone as their argmax.
- **FastSpeech 2** took durations from the Montreal Forced Aligner, an HMM-based aligner in the style of lesson 23's forced alignment. The paper found these more accurate than teacher attention, and dropping the teacher made training about three times faster.
- **Glow-TTS** and later **VITS** search for the alignment inside the model, during training, with monotonic alignment search. That section comes after the variance targets.

## Pitch and energy targets

Durations remove one source of the one-to-many problem, but not all of them.
The same phones with the same durations can still be said on a high or a low
pitch, loud or soft. A model trained with a mean squared error on such
targets predicts their average, which sounds flat. FastSpeech 2's *variance
adaptor* gives the decoder the answer instead: a pitch value and an energy
value per position, true ones in training and predicted ones at inference.

The targets come from the recording:

- **Pitch.** Estimate F0 per frame with a pitch tracker. Frames without voicing have no pitch, so either skip them or fill them by linear interpolation. Many open implementations then average the voiced frames of each phone, which gives one target per phone.
- **Energy.** The energy of frame $t$ is the L2 norm of its STFT magnitude column, $E_t = \sqrt{\sum_k |X_{k t}|^2}$. It is averaged per phone the same way, over all frames.

Before the decoder sees a pitch or energy value, FastSpeech 2 turns it into
one of 256 bins and looks up a learned embedding. Pitch bins are spaced
evenly on a log scale, because pitch is heard as ratios. Energy bins are
spaced evenly. With $K$ bins there are $K - 1$ edges $b_0 < \cdots < b_{K-2}$
from $v_{\min}$ to $v_{\max}$, and a value's bin is the number of edges below
it:

$$
b_k = \exp\left(\log v_{\min} + \frac{k}{K-2}\left(\log v_{\max} - \log v_{\min}\right)\right), \qquad
\operatorname{bin}(v) = \left|\{k : b_k < v\}\right|.
$$

Read it aloud: the edges run from $v_{\min}$ to $v_{\max}$ in equal steps of
the log. A value at or below the first edge lands in bin 0, and one above
the last edge in bin $K - 1$. In numpy:

```python
edges = np.exp(np.linspace(np.log(v_min), np.log(v_max), n_bins - 1))
bins = np.searchsorted(edges, values, side="left")
```

For pitch from 71 Hz to 795 Hz with 256 bins, the ratio between neighboring
edges is $(795/71)^{1/254} = 1.00956$, about 16.5 cents. An F0 of 200 Hz sits
$\ln(200/71) / \ln(1.00956) = 1.03564 / 0.009510 = 108.9$ steps above the
first edge. So edges 0 to 108 lie below it, 109 of them, and 200 Hz falls in
bin 109.

> [!TIP]
> Compute pitch and energy statistics on the training set only, and save the
> bin edges with the model. A model trained with one set of edges and run with
> another hears every pitch as a different note.

## Monotonic alignment search

Glow-TTS learns durations with no aligner and no teacher. Its decoder is a
normalizing flow (lesson 28) that maps each mel frame to a latent vector
$z_j$. The text encoder predicts a Gaussian, a mean $\mu_i$ and a scale
$\sigma_i$, for each phone $i$. If you knew which phone each frame belongs
to, the log-likelihood of the frames would be a sum of Gaussian
log-densities. So the training loop alternates two steps, like hard EM:

1. With the model fixed, find the alignment that makes the frames most likely.
2. With that alignment fixed, take a gradient step on the model.

Step 1 is a search over *monotonic* alignments. The first frame goes to the
first phone, the last frame to the last phone, and from one frame to the next
the phone either stays or moves forward by one. Every phone gets at least one
frame, and none is skipped. Let $P_{ij} = \log \mathcal{N}(z_j; \mu_i,
\sigma_i)$ be the log-likelihood of frame $j$ under phone $i$. Let $Q_{ij}$ be
the best total log-likelihood of any alignment of frames $1..j$ that ends on
phone $i$:

$$
Q_{i j} = \max\left(Q_{i, j-1},\ Q_{i-1, j-1}\right) + P_{i j}, \qquad
Q_{1 1} = P_{1 1}, \qquad Q_{i 1} = -\infty \text{ for } i > 1.
$$

Read it aloud: to end frame $j$ on phone $i$, frame $j-1$ was either on the
same phone or on the phone before it. Take the better of the two, and add
this frame's own log-likelihood. The first frame can only be on phone 1. The
best total is $Q_{L T}$. To recover the path, start at phone $L$ on the last
frame, and walk back: step down a phone whenever $Q_{i-1, j-1} > Q_{i, j-1}$.
The durations are the number of frames on each phone. This is lesson 12's
DTW table with a different set of allowed moves. In numpy, one column at a
time:

```python
advance = np.concatenate([[-np.inf], Q[:-1, j - 1]])
Q[:, j] = np.maximum(Q[:, j - 1], advance) + P[:, j]
```

### Work one search by hand

Take three phones and four frames with these log-likelihoods, and fill $Q$
column by column:

$$
P = \begin{bmatrix} -1 & -2 & -6 & -8 \\ -5 & -1 & -1 & -7 \\ -9 & -6 & -3 & -1 \end{bmatrix}
\quad\Rightarrow\quad
Q = \begin{bmatrix} -1 & -3 & -9 & -17 \\ -\infty & -2 & -3 & -10 \\ -\infty & -\infty & -5 & -4 \end{bmatrix}.
$$

For example, $Q_{2,3} = \max(Q_{2,2}, Q_{1,2}) + P_{2,3} = \max(-2, -3) - 1 = -3$,
and $Q_{3,4} = \max(Q_{3,3}, Q_{2,3}) + P_{3,4} = \max(-5, -3) - 1 = -4$.
Backtracking from $(3, 4)$: $Q_{2,3} = -3$ beats $Q_{3,3} = -5$, so frame 3
was on phone 2. $Q_{2,2} = -2$ beats $Q_{1,2} = -3$, so frame 2 was on phone
2. Frame 1 is on phone 1. The path is $[1, 2, 2, 3]$, the durations are
$[1, 2, 1]$, and the total is $-1 - 1 - 1 - 1 = -4$.

Brute force agrees. Four frames split into three nonempty runs in
$\binom{3}{2} = 3$ ways, with totals $-6$, $-4$, and $-5$. In general there
are $\binom{T-1}{L-1}$ alignments, which is about $10^{20}$ for 20 phones and
100 frames. The dynamic program takes $LT = 2{,}000$ steps.

> [!KEY]
> Monotonic alignment search is Viterbi over "stay or advance by one": the
> best alignment of the first $j$ frames ending on phone $i$ extends the better
> of the two ways to reach the previous frame.

MAS runs without gradients, under `torch.no_grad()`. Gradients flow only
through $P$, evaluated on the chosen path. The Glow-TTS paper reports that
MAS took under 20 ms per iteration on a CPU, less than 2% of training time.
At inference there is no mel spectrogram to align, so a duration predictor,
trained on the durations MAS found, supplies them.

> [!RECAP]
> - The length regulator builds $A_{in} = \mathbb{1}[e_i - d_i \le n < e_i]$ from cumulative sums and computes $A^\top H$; padded phones have duration 0.
> - Predicted durations come from $\max(1, \operatorname{round}(\alpha(e^{\hat\ell} - 1)))$.
> - Energy is the L2 norm of each STFT column; pitch and energy are averaged per phone and quantized into 256 bins.
> - $P_{ij}$ is a sum of Gaussian log-densities over dimensions, computed with matrix products.
> - MAS fills $Q_{ij} = \max(Q_{i,j-1}, Q_{i-1,j-1}) + P_{ij}$ and backtracks from the last phone at the last frame.

## Build it step by step

Build the functions in the order the checks test them.

1. **`length_regulate(h, durations)`** takes `(B, L, D)` vectors and
   `(B, L)` durations and returns `(frames, mask)`. Build `ends` with
   `torch.cumsum`, compare against `torch.arange(N_max)`, and apply the
   alignment with `torch.einsum`. The common mistake is using `ends` as the
   start, which shifts every phone one duration late.
2. **`durations_from_log(log_durations, mask, alpha)`** exponentiates,
   subtracts 1, scales, rounds with `torch.round`, clamps at 1, and
   multiplies by the mask. The common mistake is clamping after masking,
   which gives padded phones a frame.
3. **`frame_energy(magnitude)`** and **`phone_average(values, durations, voiced)`**:
   the first is one `np.sqrt(np.sum(...))` down the bins. The second maps
   frames to phones with `np.repeat` and sums with `np.bincount`. The common
   mistake is dividing by the duration instead of by the number of voiced
   frames.
4. **`quantize(values, v_min, v_max, n_bins, log_scale)`** builds
   `n_bins - 1` edges and calls `np.searchsorted(..., side="left")`. The
   common mistake is building `n_bins` edges, which makes 257 bins.
5. **`gaussian_log_likelihood(z, mu, log_sigma)`** returns `P` of shape
   `(L, T)`. Expand the square into three matrix products. The common mistake
   is forgetting the $-\log\sigma$ term, which lets a wide Gaussian win
   everything.
6. **`monotonic_alignment_search(P)`** and **`path_to_durations(path, n_text)`**:
   fill `Q` one column at a time, backtrack from `(L - 1, T - 1)`, and count
   with `np.bincount`. The common mistake is letting a phone start before
   its frame index allows. Starting every column except the first from
   $-\infty$ in the rows it can't reach prevents that.

## Cost and scaling

The following table shows how each piece grows with $L$ phones, $N$ frames,
$D$ dimensions, and batch size $B$.

| Piece | Time | Memory | Parallel over frames? |
|---|---|---|---|
| Length regulator (matrix form) | $O(BLND)$ | $O(BLN)$ for $A$ | Yes |
| Length regulator (`repeat_interleave`) | $O(BND)$ | $O(BND)$ | Yes |
| Gaussian log-likelihoods $P$ | $O(LND)$ | $O(LN)$ | Yes |
| Monotonic alignment search | $O(LN)$ | $O(LN)$ | No: one column after another |
| Brute force | $O\left(\binom{N-1}{L-1} N\right)$ | $O(N)$ | Not feasible |

For a 10-second clip at 86 frames per second and 100 phones, $P$ and $Q$ hold
86,100 numbers each, which is small. The column loop is the cost: in pure
Python it takes a few milliseconds per utterance, so real systems compile it
with Cython or numba, or run it on the GPU.

## Where this breaks

**Rounding drift.** Rounding every phone's duration separately can add or
lose several frames over a long sentence. When a total length matters, as in
dubbing, round the cumulative sums instead and take differences.

**Aligner errors become model errors.** A forced aligner that puts a boundary
40 ms late teaches the duration predictor that phone is long. Check the
aligner on a sample of utterances by listening to the cut points.

**Averaged prosody.** A duration predictor trained with squared error
predicts the mean duration, which gives flat, metronome-like rhythm. VITS
replaces it with a stochastic duration predictor that samples durations from
a flow.

**Unvoiced pitch.** Averaging raw F0 over a phone that is half voiced mixes
real pitch with zeros. Average voiced frames only, or interpolate first.

> [!WARNING]
> MAS needs at least as many frames as phones. A clip shorter than its phone
> string has no monotonic alignment, and the table's last cell stays
> $-\infty$. Filter such clips out of the training set.

## The exercise

You implement `length_regulate`, `durations_from_log`, `frame_energy`,
`phone_average`, `quantize`, `gaussian_log_likelihood`,
`monotonic_alignment_search`, and `path_to_durations`. The checks run the
hand-worked expansion, compare a padded batch with `torch.repeat_interleave`,
and confirm that each phone's gradient equals its duration. They check that
200 Hz lands in pitch bin 109, compare $P$ with a loop over Gaussian
log-densities, and compare the score of your alignment with brute-force
enumeration on six random matrices. The last check hides 12 durations in
noisy frames and requires MAS to find all of them.

Press **Run** to hide eight phones' durations in 109 noisy frames and find
them again with your search. The run prints per-phone pitch, pitch bins, and
energy, draws the likelihood table and the chosen path, and plays the phones
expanded by your length regulator at $\alpha = 1$ and $\alpha = 1.6$.
