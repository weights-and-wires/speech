---
title: Vector quantization
slug: 30-vector-quantization
part: "Part 6 — Codecs and token models"
summary: Turn continuous vectors into a few integers with learned codebooks, residual stages, and rounded grids, and keep gradients flowing.
minutes: 70
gpu: false
objectives:
  - Quantize vectors to their nearest codeword with the distance expansion, and count the bits each index costs.
  - Train through the argmin with the straight-through estimator, the codebook loss, and the commitment loss.
  - Update a codebook by exponential moving averages, and detect and repair codebook collapse with perplexity and dead-code restart.
  - Build residual vector quantization and explain why eight codebooks of 1024 replace one codebook of 2^80.
  - Build finite scalar quantization and its mixed-radix index.
prereqs:
  - 02-quantization-and-audio-files
exercise: 30-vector-quantization
---

# Vector quantization

Part 5 ended with systems that turn text into mel spectrograms and mel
spectrograms into audio. Part 6 turns audio into *tokens*, integers from a
fixed list that a language model can predict. The tool that makes them is
quantization. In lesson 2, *Quantization and audio files*, you rounded each
sample to one of $2^{16}$ levels. That is *scalar* quantization: every number is rounded on
its own. A neural codec does something bolder. It rounds a whole vector of
128 or 512 numbers at once, to one entry in a list of about a thousand
allowed vectors. The list is called a *codebook*. The vector is then sent as
one small integer, its position in the list.

Think of a codebook as a very short vocabulary for sound. A speaker with a
vocabulary of 1024 words can't say everything, but each word costs only 10
bits. Pick the vocabulary well and most of what matters survives.

This lesson builds the machinery that every codec in the rest of the course
relies on: finding the nearest codeword, training through that choice,
keeping every codeword in use, and stacking small codebooks.

## What you already know

- Uniform quantization rounds a number to the nearest of $2^N$ levels, and each extra bit buys about 6 dB of SNR.
- A neural network is trained by gradient descent, and backpropagation needs every operation to have a gradient.
- In torch, `tensor.detach()` returns the same values with no gradient attached.
- k-means clustering moves each center to the mean of the points assigned to it.
- The entropy of a distribution $p$ is $-\sum_k p_k \log p_k$, in nats with the natural log.

## Quantize to the nearest codeword

A *codebook* is a matrix $C$ of shape $(K, D)$: $K$ *codewords*, each a
vector of $D$ numbers. To quantize a vector $z$, find the codeword closest to
it and replace $z$ with that codeword:

$$
k^* = \operatorname*{arg\,min}_k \lVert z - e_k \rVert_2^2, \qquad z_q = e_{k^*}
$$

Read it aloud: $k^*$ is the index of the codeword $e_k$ nearest to $z$, and
$z_q$ is that codeword. You send only $k^*$, which costs $\log_2 K$ bits:
10 bits for 1024 entries, whatever $D$ is.

For $N$ vectors, the obvious code builds an $(N, K, D)$ array and runs out of
memory. Expand the square instead:

$$
\lVert z - e_k \rVert^2 = \lVert z \rVert^2 - 2\, z^\top e_k + \lVert e_k \rVert^2
$$

Read it aloud: the squared distance is the squared length of $z$, minus twice
the dot product, plus the squared length of the codeword. The middle term for
every pair is one matrix product, so the whole table costs one `(N, D) @ (D, K)`.

```python
d = (z**2).sum(1, keepdims=True) - 2 * z @ C.T + (C**2).sum(1)[None, :]
idx = d.argmin(1)
```

### Work one lookup by hand

Take $z = (0.9, -0.3)$ and a codebook of the four unit directions
$(1, 0)$, $(0, 1)$, $(-1, 0)$, and $(0, -1)$. The squared distances are
$0.1^2 + 0.3^2 = 0.10$, $0.9^2 + 1.3^2 = 2.50$, $1.9^2 + 0.3^2 = 3.70$, and
$0.9^2 + 0.7^2 = 1.30$. The smallest is index 0, so $z_q = (1, 0)$, and the
index costs $\log_2 4 = 2$ bits.

## Train through the argmin

In a codec, an encoder network produces $z_e$, the quantizer replaces it with
$z_q$, and a decoder rebuilds the audio from $z_q$. The trouble is the
argmin. Its output is an integer that jumps from one value to another, so its
gradient is zero almost everywhere. No gradient reaches the encoder, and the
encoder never learns.

The *straight-through estimator* is a deliberate lie that works. In the
forward pass, use $z_q$. In the backward pass, pretend the quantizer was the
identity and copy the decoder's gradient to $z_e$ unchanged. VQ-VAE
(van den Oord et al., 2017) made this the standard way to train through a
codebook:

$$
z_q^{\text{ST}} = z_e + \operatorname{sg}[z_q - z_e]
$$

Read it aloud: add to $z_e$ the gap between the codeword and $z_e$, with a
stop-gradient $\operatorname{sg}$ around the gap. The value is $z_q$, because
$z_e$ cancels. The gradient with respect to $z_e$ is 1, because the gap is a
constant as far as backpropagation knows.

```python
z_q_st = z_e + (z_q - z_e).detach()
```

Nothing yet moves the codewords, and nothing stops the encoder drifting away
from them. VQ-VAE adds two terms to the reconstruction loss:

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lVert \operatorname{sg}[z_e] - e \rVert_2^2 + \beta\, \lVert z_e - \operatorname{sg}[e] \rVert_2^2, \qquad \beta = 0.25
$$

Read it aloud: the second term is the *codebook loss*, the squared distance
between the chosen codeword $e$ and a frozen copy of the encoder output. Its
gradient reaches only the codeword and pulls it toward the vectors that chose
it. The third term is the *commitment loss*, the same distance with the
codeword frozen. Its gradient reaches only the encoder and keeps it close to
the codeword it picked. The weight $\beta = 0.25$ is the paper's choice, and
it reported results were robust for $\beta$ from 0.1 to 2.0.

The two terms have the same value. Only the stop-gradient differs.

```python
codebook_loss = ((z_e.detach() - e) ** 2).sum(-1).mean()
commitment_loss = ((z_e - e.detach()) ** 2).sum(-1).mean()
```

For the vector in the lookup example, $z_e = (0.9, -0.3)$ and $e = (1, 0)$,
both terms equal $0.10$. The codebook loss has gradient
$2(e - z_e) = (0.2, 0.6)$ with respect to $e$, so a gradient step moves the
codeword toward $z_e$. The commitment loss, weighted by $\beta$, has gradient
$2\beta(z_e - e) = (-0.05, -0.15)$ with respect to $z_e$, so a step moves the
encoder output toward the codeword.

> [!KEY]
> The straight-through estimator gives the encoder a gradient, the codebook loss moves the codewords toward the encoder, and the commitment loss keeps the encoder near its codewords. The stop-gradient decides which is which.

## Update codebooks by averaging, and keep them alive

The codebook loss is one way to move codewords. The steadier way, used by
VQ-VAE-2, SoundStream, and EnCodec, runs k-means online. Each code keeps two
running statistics: $N_k$, how many vectors chose it, and $m_k$, the sum of
those vectors. After each batch, both decay toward
the batch's values, and the codeword is their ratio:

$$
N_k \leftarrow \gamma N_k + (1 - \gamma)\, n_k, \qquad
m_k \leftarrow \gamma m_k + (1 - \gamma) \sum_{i:\, k_i^* = k} z_i, \qquad
e_k = \frac{m_k}{\tilde N_k}
$$

Read it aloud: $n_k$ is how many vectors in this batch chose code $k$. The
updated count is $\gamma$ times the old one plus $1 - \gamma$ times the batch
count, and the same for the sum. The codeword is the running sum divided by
the running count, which is a running mean of the vectors that chose it. The
decay $\gamma$ is 0.99 in SoundStream and EnCodec. $\tilde N_k$ is $N_k$
after Laplace smoothing,
$\tilde N_k = \frac{N_k + \epsilon}{\sum_j N_j + K\epsilon} \sum_j N_j$, which
keeps an unused code from dividing by zero.

```python
n_k = np.bincount(idx, minlength=K)
s_k = np.zeros_like(sums); np.add.at(s_k, idx, z)
counts = decay * counts + (1 - decay) * n_k
sums = decay * sums + (1 - decay) * s_k
```

### Work one EMA step by hand

Take three codes at $(0, 0)$, $(1, 0)$, and $(0, 1)$, each with count 1 and
sum equal to itself, and decay $\gamma = 0.9$. A batch of four vectors,
$(0.2, 0)$, $(0, 0.2)$, $(0.9, 0.1)$, and $(1.3, -0.1)$, chooses codes
0, 0, 1, and 1. No vector chooses code 2.

| Code | $n_k$ | Batch sum | $N_k$ | $m_k$ | $e_k = m_k / N_k$ |
|---|---|---|---|---|---|
| 0 | 2 | $(0.2, 0.2)$ | $0.9 + 0.2 = 1.1$ | $(0.02, 0.02)$ | $(0.0182, 0.0182)$ |
| 1 | 2 | $(2.2, 0.0)$ | $1.1$ | $(1.12, 0.00)$ | $(1.0182, 0.0000)$ |
| 2 | 0 | $(0, 0)$ | $0.9$ | $(0.00, 0.90)$ | $(0.0000, 1.0000)$ |

Code 0 moves toward its two vectors, and code 1 moves toward their mean,
$(1.1, 0)$. Code 2 doesn't move, but its count shrinks by 10 percent. After
ten more batches that ignore it, its count is $0.9^{11} \approx 0.31$.

### Codebook collapse

That shrinking count warns of *codebook collapse*: a few codes win every
vector, and a codebook of 1024 entries behaves like one of 20 while still
costing 10 bits.

Measure it with the *perplexity* of the code-usage histogram:

$$
\text{perplexity} = \exp\left(-\sum_k \bar p_k \log \bar p_k\right)
$$

Read it aloud: $\bar p_k$ is the fraction of vectors that chose code $k$.
Exponentiate the entropy of those fractions to get the number of codes in
effective use: $K$ for uniform use, and 1 when every vector picks one code.

The fixes are the following:

- **k-means initialization.** SoundStream sets the codebook from k-means on the first batch, so every code starts near data.
- **Dead-code restart.** Replace any code whose EMA count falls below a threshold with a random encoder output from the batch. SoundStream uses decay 0.99 and threshold 2.
- **Low-dimensional, normalized lookup.** DAC projects to 8 dimensions and L2-normalizes both sides before the lookup, which turns distance into cosine similarity.
- **No learned codebook at all.** Finite scalar quantization, later in this lesson.

In this lesson's **Run**, a codebook of 64 codes initialized far from the
data collapses to a perplexity of 1.0. With dead-code restart, the same
training reaches 59.1.

> [!WARNING]
> Don't read the codebook loss or the reconstruction loss to spot collapse. Both can look healthy while three codes do all the work. Log the perplexity of every codebook during training.

## Stack quantizers on the residual

A codec at 24 kHz with 75 frames per second and a budget of 6 kbps can spend
$6000 / 75 = 80$ bits on each frame. One codebook with 80-bit indices would
need $2^{80}$ entries, about $10^{24}$. You can't store that table, let alone
search it.

*Residual vector quantization* (RVQ) spends the 80 bits in eight stages of
10 bits each. The first codebook quantizes the vector. The second quantizes
what the first one got wrong, the *residual*. The third quantizes what is
left after that, and so on. SoundStream (Zeghidour et al., 2021) brought it
to neural audio codecs:

$$
r_0 = z, \qquad q_i = Q_i(r_{i-1}), \qquad r_i = r_{i-1} - q_i, \qquad \hat z = \sum_{i=1}^{N_q} q_i
$$

Read it aloud: start with the residual equal to the input. At stage $i$,
quantize the residual with codebook $i$, and subtract the result from the
residual. The reconstruction is the sum of all the stages' codewords. Each
stage stores one index, so a frame costs $N_q \log_2 K$ bits.

```python
residual = x.copy()
for i, C in enumerate(codebooks):
    idx, q = nearest_codeword(residual, C)
    recon += q
    residual -= q
```

The first stage captures the coarse shape and later stages refine it. Drop
the last stages and you get a coarser version of the same vector, which is
how one codec offers several bitrates. EnCodec at 24 kHz has up to 32
codebooks of 1024: 2 give 1.5 kbps, and 32 give 24 kbps.

### Work two stages by hand

Take $x = (0.9, -0.3)$ again. Stage 1 uses the four unit directions and picks
$(1, 0)$, as before, leaving $r_1 = (-0.1, -0.3)$. Stage 2 has the codebook
$(0.1, 0.1)$, $(-0.1, -0.2)$, $(0.2, -0.2)$, and $(-0.2, 0.2)$. The squared
distances from $r_1$ are $[0.20, 0.01, 0.10, 0.26]$, so stage 2 picks index 1,
$q_2 = (-0.1, -0.2)$, leaving $r_2 = (0, -0.1)$.

The following values come out of the two stages:

- The codes are $(0, 1)$, which cost $2 \times \log_2 4 = 4$ bits.
- The reconstruction is $(1, 0) + (-0.1, -0.2) = (0.9, -0.2)$.
- The residual norms fall from $\lVert x \rVert = 0.949$ to $0.316$ to $0.100$.
- The commitment loss summed over stages, as EnCodec writes it, is $\lVert r_1 \rVert^2 + \lVert r_2 \rVert^2 = 0.10 + 0.01 = 0.11$.

The following demo runs RVQ on 2-D points, stage by stage.

```demo
{"id": "rvq-2d", "caption": "Raise the stages used one at a time and watch each codebook tile a smaller region around the leftover error, while the error readout falls with every stage."}
```

> [!KEY]
> Residual quantization turns an exponential table into a sum: $N_q$ codebooks of $K$ entries give $K^{N_q}$ possible reconstructions for the storage and search cost of $N_q K$ codewords.

## Round on a fixed grid instead

*Finite scalar quantization* (FSQ) drops the learned codebook
(Mentzer et al., 2024). Project to a few dimensions, squash each into a
bounded range, and round it. The codebook is the grid of integer
combinations, and it has nothing to collapse:

$$
\hat z_i = \operatorname{round}\left(\left\lfloor \tfrac{L_i}{2} \right\rfloor \tanh(z_i)\right)
$$

Read it aloud: in dimension $i$, squash $z_i$ into $(-1, 1)$ with tanh,
stretch it to $\pm\lfloor L_i / 2 \rfloor$, and round. With $L_i = 5$ levels
the result is one of $-2, -1, 0, 1, 2$. The implicit codebook has
$\prod_i L_i$ entries. Rounding has no gradient either, so FSQ uses the
straight-through trick on the rounding step.

To send $\hat z$, turn it into one integer. Shift each dimension to a digit
from 0 to $L_i - 1$ and read the digits as a mixed-radix number, the way
digits of different bases make up a clock time:

$$
\text{idx} = \sum_i \left(\hat z_i + \left\lfloor \tfrac{L_i}{2} \right\rfloor\right) \prod_{j < i} L_j
$$

```python
bounded = half * torch.tanh(z)
z_hat = bounded + (torch.round(bounded) - bounded).detach()
```

### Work one FSQ code by hand

Take $L = [5, 5, 5]$, a grid of 125 codes, and $z = (0.3, -2.0, 1.2)$:

- Bounded: $2\tanh(z) = (0.583, -1.928, 1.667)$.
- Rounded: $\hat z = (1, -2, 2)$.
- Shifted to digits: $(3, 0, 4)$.
- Index: $3 + 0 \times 5 + 4 \times 25 = 103$.

The FSQ paper recommends $[8, 5, 5, 5]$ for a 1000-code grid, and at least
5 levels per dimension. CosyVoice 2's speech tokenizer uses 8 dimensions of 3
levels each, $3^8 = 6561$ codes, and reports 100 percent of them in use.
XCodec2 uses a single FSQ codebook of $4^8 = 65{,}536$ codes.

> [!TIP]
> An even number of levels has no grid point at 0, so the reference FSQ code shifts the tanh by half a step for even $L_i$. The exercise uses odd levels, which need no offset.

> [!RECAP]
> - The nearest codeword comes from the expansion $\lVert z \rVert^2 - 2 z^\top e + \lVert e \rVert^2$, and its index costs $\log_2 K$ bits.
> - $z_e + \operatorname{sg}[z_q - z_e]$ has the value of $z_q$ and the gradient of $z_e$.
> - The codebook loss detaches $z_e$ and the commitment loss detaches $e$. EMA updates replace the codebook loss with running counts and sums.
> - RVQ quantizes the residual stage by stage, and perplexity with dead-code restart keeps every codebook in use.
> - FSQ rounds $\lfloor L/2 \rfloor \tanh(z)$ and indexes the grid as a mixed-radix number.

## Build it step by step

Build the functions in the order the checks test them.

1. **`nearest_codeword(z, codebook)`.** Returns `(indices, quantized)` of shapes `(N,)` and `(N, D)`. Use the expansion and `argmin(axis=1)`. Common mistake: broadcasting to `(N, K, D)`.
2. **`straight_through(z_e, z_q)`.** One line with `.detach()`. Common mistake: returning `z_q.detach()`, which has the right value and no gradient.
3. **`vq_losses(z_e, codebook)`.** Find the indices from detached tensors, gather `e = codebook[indices]`, and return the straight-through output, the indices, and the two losses. Common mistake: swapping the two `.detach()` calls.
4. **`ema_update(counts, sums, z, indices, decay, eps)`.** Use `np.bincount` and `np.add.at`, decay both statistics, smooth the counts, and divide. Common mistake: dividing by unsmoothed counts, which gives `nan` for an unused code.
5. **`restart_dead_codes(codebook, counts, sums, z, threshold, rng)`.** Draw a row of `z` with `rng.choice` for each code with `counts < threshold`, and reset its count to `threshold` and its sum to the row times `threshold`. Common mistake: resetting the codeword but not the sum, so the next EMA step pulls it back.
6. **`codebook_perplexity(indices, K)`.** A normalized `np.bincount`, then the exponential of the entropy over the nonzero entries.
7. **`rvq_encode(x, codebooks)`.** Call `nearest_codeword` on the running residual at each stage, and return the codes, the reconstruction, and an `(N, N_q + 1)` array of residual norms. Common mistake: quantizing `x` at every stage instead of the residual.
8. **`rvq_decode(codes, codebooks)`.** Sum `codebooks[i][codes[:, i]]` over the columns `codes` has.
9. **`fsq_quantize(z, levels)`.** Multiply $\lfloor L_i / 2 \rfloor$ by `torch.tanh(z)` and round with a straight-through step.
10. **`fsq_index` and `fsq_unindex`.** Digits times the cumulative products `[1, L_0, L_0 L_1, ...]`, and back with `//` and `%`. Common mistake: forgetting the shift by $\lfloor L_i / 2 \rfloor$.

## Cost and scaling

The following table shows the cost for $N$ vectors of dimension $D$, with
$N_q$ stages of $K$ codewords and $d$ FSQ dimensions.

| Quantizer | Lookup time | Parameters | Bits per vector |
|---|---|---|---|
| One VQ codebook | $O(N K D)$ | $K D$ | $\log_2 K$ |
| RVQ | $O(N N_q K D)$ | $N_q K D$ | $N_q \log_2 K$ |
| FSQ | $O(N d)$ | none, plus the projections | $\sum_i \log_2 L_i$ |

Doubling $K$ adds one bit and doubles the lookup. Adding a stage adds
$\log_2 K$ bits for a linear cost, so codecs grow bitrate by adding stages.
The distance table is $N \times K$ floats: about 80 MB in float32 for 10,000
frames and $K = 2048$.

## Where this breaks

**Codebook collapse.** A random initialization far from the encoder outputs
leaves most codes unused. Initialize with k-means, restart dead codes, or
switch to FSQ, and watch perplexity.

**The stop-gradient on the wrong side.** Detach the codeword in the codebook
loss and the codebook never learns. Detach nothing and the encoder and
codebook chase each other.

**Late RVQ stages that learn nothing.** If the early stages leave residuals
far smaller than the later codewords, the later stages choose the same few
codes. Initialize each stage from its own residuals, as the **Run** does.

**Straight-through bias.** The estimator pretends quantization is the
identity. When codewords are far apart, that gradient is a poor guide. FSQ's
fine grid makes the error smaller.

## The exercise

You build `nearest_codeword`, `straight_through`, `vq_losses`, `ema_update`,
`restart_dead_codes`, `codebook_perplexity`, `rvq_encode`, `rvq_decode`,
`fsq_quantize`, `fsq_index`, and `fsq_unindex`. Press **Run** to train a
64-code, four-stage RVQ on 8-sample frames of a synthetic voice, and hear it
at one stage (12 kbps, about 16 dB SNR) and at four (48 kbps, about 37 dB).

The checks compare `nearest_codeword` with brute force, test the
straight-through value and gradient, and confirm that each VQ loss sends its
gradient only where it belongs, against finite differences. They reproduce the
hand-worked EMA step and RVQ example, and test that restart touches only dead
codes, that residual norms never grow, that decoding inverts encoding, and
that FSQ's index is a bijection.
