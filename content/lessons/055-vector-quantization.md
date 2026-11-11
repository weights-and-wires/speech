---
title: Vector quantization
slug: 055-vector-quantization
part: "Part 6 — Codecs and token models"
summary: Replace a vector with its nearest codeword, train the encoder through that choice, and keep every codeword in use.
minutes: 40
gpu: false
objectives:
  - Quantize vectors to their nearest codeword with the distance expansion, and count the bits each index costs.
  - Train through the argmin with the straight-through estimator, the codebook loss, and the commitment loss.
  - Update a codebook by exponential moving averages, and say what each running statistic holds.
  - Detect codebook collapse with perplexity, and repair it with dead-code restart.
prereqs:
  - 003-quantization-and-dither
exercise: 055-vector-quantization
---

# Vector quantization

A neural codec compresses audio into a short list of integers. Every 13 milliseconds its encoder produces a vector of 128 or 512 numbers, and a language model can only predict an integer from a fixed list. So the vector must become one small number, and the encoder that made the vector must still be trainable.

The obvious fix is to round each of the 128 numbers on its own, as you rounded samples in [Quantization and dither](/lessons/003-quantization-and-dither). That fails twice. Rounding 128 numbers to 4 levels each costs 256 bits per frame, and it gives 128 integers where you wanted one. This chapter builds the alternative: a list of about a thousand allowed vectors, a *codebook*, and an index into it.

Think of a codebook as a very short vocabulary for sound. A speaker with 1024 words can't say everything, but each word costs only 10 bits, and a well-chosen vocabulary keeps most of what matters.

## What you already know

- Uniform quantization rounds a number to the nearest of $2^N$ levels, and each extra bit buys about 6 dB of SNR.
- A neural network trains by gradient descent, and backpropagation needs every operation to have a gradient.
- In torch, `tensor.detach()` returns the same values with no gradient attached.
- k-means moves each center to the mean of the points assigned to it.
- The entropy of a distribution $p$ is $-\sum_k p_k \ln p_k$, in nats.

## Replace a vector by its nearest codeword

A codebook is a matrix $C$ of shape $(K, D)$: $K$ *codewords* $e_k$, each a vector of $D$ numbers. To quantize a vector $z$, find the codeword closest to it and send that codeword's index. Figure 1 shows the idea in two dimensions, with the four unit directions as the codebook.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="A plane with four codewords at the unit directions and the vector (0.9, -0.3), which lies nearest the codeword (1, 0)">
<line x1="40.0" y1="170.0" x2="340.0" y2="170.0" class="stroke-grid"/>
<line x1="190.0" y1="20.0" x2="190.0" y2="320.0" class="stroke-grid"/>
<line x1="41.5" y1="318.5" x2="338.5" y2="21.5" class="stroke-muted" stroke-dasharray="5 4"/>
<line x1="41.5" y1="21.5" x2="338.5" y2="318.5" class="stroke-muted" stroke-dasharray="5 4"/>
<line x1="289.0" y1="203.0" x2="300.0" y2="170.0" class="stroke-b"/>
<line x1="289.0" y1="203.0" x2="190.0" y2="60.0" class="stroke-muted" stroke-dasharray="4 3"/>
<line x1="289.0" y1="203.0" x2="80.0" y2="170.0" class="stroke-muted" stroke-dasharray="4 3"/>
<line x1="289.0" y1="203.0" x2="190.0" y2="280.0" class="stroke-muted" stroke-dasharray="4 3"/>
<circle cx="300.0" cy="170.0" r="8" class="fill-b"/>
<text x="314.0" y="160.0" class="fig-small" text-anchor="start">e0</text>
<circle cx="190.0" cy="60.0" r="8" class="fill-b"/>
<text x="202.0" y="48.0" class="fig-small" text-anchor="start">e1</text>
<circle cx="80.0" cy="170.0" r="8" class="fill-b"/>
<text x="66.0" y="160.0" class="fig-small" text-anchor="end">e2</text>
<circle cx="190.0" cy="280.0" r="8" class="fill-b"/>
<text x="202.0" y="300.0" class="fig-small" text-anchor="start">e3</text>
<circle cx="289.0" cy="203.0" r="6" class="fill-a"/>
<text x="299.0" y="225.0" class="fig-small" text-anchor="start">z = (0.9, -0.3)</text>
<text x="42.0" y="188.0" class="fig-small" text-anchor="start">dim 1</text>
<text x="198.0" y="32.0" class="fig-small" text-anchor="start">dim 2</text>
<text x="400.0" y="80.0" class="fig-bold" text-anchor="start">Squared distance from z</text>
<rect x="400.0" y="96.0" width="4.1" height="22.0" class="fill-b"/>
<text x="558.0" y="112.0" class="fig-small" text-anchor="start">e0: 0.10</text>
<rect x="400.0" y="130.0" width="101.4" height="22.0" class="fill-soft-a"/>
<text x="558.0" y="146.0" class="fig-small" text-anchor="start">e1: 2.50</text>
<rect x="400.0" y="164.0" width="150.0" height="22.0" class="fill-soft-a"/>
<text x="558.0" y="180.0" class="fig-small" text-anchor="start">e2: 3.70</text>
<rect x="400.0" y="198.0" width="52.7" height="22.0" class="fill-soft-a"/>
<text x="558.0" y="214.0" class="fig-small" text-anchor="start">e3: 1.30</text>
<text x="400.0" y="258.0" class="fig-small" text-anchor="start">Smallest is e0, so you send index 0.</text>
<text x="400.0" y="278.0" class="fig-small" text-anchor="start">Four codewords cost log2(4) = 2 bits.</text>
</svg>
Figure 1. The dashed diagonals cut the plane into four cells, one per codeword. Notice that z falls in e0's cell, and that the nearest-codeword rule is just "which cell am I in".
```

The rule is a one-line formula:

$$
k^* = \operatorname*{arg\,min}_k \lVert z - e_k \rVert_2^2, \qquad z_q = e_{k^*}.
$$

Read it aloud: $k^*$ is the index of the codeword nearest to $z$, and $z_q$ is that codeword. You send only $k^*$, which costs $\log_2 K$ bits, 10 bits for 1024 codewords, whatever $D$ is.

Work the figure's example. Take $z = (0.9, -0.3)$. The squared distances to $(1, 0)$, $(0, 1)$, $(-1, 0)$, and $(0, -1)$ are $0.1^2 + 0.3^2 = 0.10$, $0.9^2 + 1.3^2 = 2.50$, $1.9^2 + 0.3^2 = 3.70$, and $0.9^2 + 0.7^2 = 1.30$. The smallest is index 0, so $z_q = (1, 0)$, and the index costs $\log_2 4 = 2$ bits.

For $N$ vectors, the obvious code builds an $(N, K, D)$ array of differences, and that array runs out of memory. Expand the square instead:

$$
\lVert z - e_k \rVert^2 = \lVert z \rVert^2 - 2\, z^\top e_k + \lVert e_k \rVert^2.
$$

Read it aloud: the squared distance is the squared length of $z$, minus twice the dot product, plus the squared length of the codeword. The dot products for every pair come from one matrix product, so the whole table costs one `(N, D) @ (D, K)` multiplication:

```python
d = (z**2).sum(1, keepdims=True) - 2 * z @ C.T + (C**2).sum(1)[None, :]
idx = d.argmin(1)
```

## Train through the argmin

In a codec, an encoder produces $z_e$, the quantizer replaces it with $z_q$, and a decoder rebuilds the audio from $z_q$. The trouble is the argmin. Its output is an integer that jumps from one value to another, so its gradient is zero almost everywhere. No gradient reaches the encoder, and the encoder never learns.

The *straight-through estimator* is a deliberate lie that works. In the forward pass, use $z_q$. In the backward pass, pretend the quantizer was the identity, and copy the decoder's gradient to $z_e$ unchanged (Figure 2). VQ-VAE, from 2017, made this the standard way to train through a codebook.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Forward pass runs through the quantizer; the backward gradient skips it and goes straight from the quantized vector to the encoder output">
<defs><marker id="c044-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="20.0" y="80.0" width="100.0" height="50.0" class="fill-soft-a" rx="6"/>
<rect x="20.0" y="80.0" width="100.0" height="50.0" class="stroke-a" fill="none" rx="6"/>
<text x="70.0" y="110.0" text-anchor="middle">encoder</text>
<rect x="250.0" y="80.0" width="110.0" height="50.0" class="fill-soft-b" rx="6"/>
<rect x="250.0" y="80.0" width="110.0" height="50.0" class="stroke-b" fill="none" rx="6"/>
<text x="305.0" y="105.0" text-anchor="middle">quantizer</text>
<text x="305.0" y="122.0" class="fig-small" text-anchor="middle">argmin</text>
<rect x="480.0" y="80.0" width="100.0" height="50.0" class="fill-soft-c" rx="6"/>
<rect x="480.0" y="80.0" width="100.0" height="50.0" class="stroke-c" fill="none" rx="6"/>
<text x="530.0" y="110.0" text-anchor="middle">decoder</text>
<line x1="120.0" y1="105.0" x2="250.0" y2="105.0" class="stroke-a" marker-end="url(#c044-arrow)"/>
<line x1="360.0" y1="105.0" x2="480.0" y2="105.0" class="stroke-a" marker-end="url(#c044-arrow)"/>
<text x="185.0" y="66.0" class="fig-small" text-anchor="middle">z_e = (0.9, -0.3)</text>
<text x="420.0" y="66.0" class="fig-small" text-anchor="middle">z_q = (1, 0)</text>
<text x="185.0" y="96.0" class="fig-small" text-anchor="middle">forward</text>
<path d="M480 135 C 440 200, 160 200, 120 135" class="stroke-b dashed" fill="none" marker-end="url(#c044-arrow)"/>
<text x="300.0" y="206.0" class="fig-small" text-anchor="middle">backward: copy the gradient unchanged</text>
<text x="300.0" y="232.0" class="fig-small" text-anchor="middle">The argmin has zero gradient almost everywhere, so the copy replaces it.</text>
</svg>
Figure 2. The forward pass sends the codeword to the decoder, and the backward pass copies the decoder's gradient to the encoder output. Notice that the gradient never meets the argmin.
```

You write the trick as one expression:

$$
z_q^{\text{ST}} = z_e + \operatorname{sg}[z_q - z_e].
$$

Read it aloud: add to $z_e$ the gap between the codeword and $z_e$, with a stop-gradient $\operatorname{sg}$ around the gap. The value is $z_q$, because $z_e$ cancels. The gradient with respect to $z_e$ is 1, because backpropagation treats the gap as a constant.

```python
z_q_st = z_e + (z_q - z_e).detach()
```

> [!KEY]
> The straight-through estimator has the value of the codeword and the gradient of the encoder output, so the decoder's error teaches the encoder even though the argmin can't.

## Keep the encoder and codebook together

Nothing yet moves the codewords, and nothing stops the encoder from drifting away from them. VQ-VAE adds two terms to the reconstruction loss:

$$
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lVert \operatorname{sg}[z_e] - e \rVert_2^2 + \beta\, \lVert z_e - \operatorname{sg}[e] \rVert_2^2, \qquad \beta = 0.25.
$$

Read it aloud: the second term is the *codebook loss*, the squared distance between the chosen codeword $e$ and a frozen copy of the encoder output. Its gradient reaches only the codeword and pulls it toward the vectors that chose it. The third term is the *commitment loss*, the same distance with the codeword frozen. Its gradient reaches only the encoder and keeps it near the codeword it picked. The VQ-VAE paper chose $\beta = 0.25$ and reported that results were robust for $\beta$ from 0.1 to 2.0.

The two terms have the same value, and only the stop-gradient differs. For the example above, $z_e = (0.9, -0.3)$ and $e = (1, 0)$, so both terms equal $0.10$. The codebook loss has gradient $2(e - z_e) = (0.2, 0.6)$ with respect to $e$, so a gradient step moves the codeword toward $z_e$. The commitment loss has gradient $2\beta(z_e - e) = (-0.05, -0.15)$ with respect to $z_e$, so a step moves the encoder output toward the codeword.

```python
codebook_loss = ((z_e.detach() - e) ** 2).sum(-1).mean()
commitment_loss = ((z_e - e.detach()) ** 2).sum(-1).mean()
```

> [!WARNING]
> Detach the codeword in the codebook loss and the codebook never learns. Detach nothing and the encoder and codebook chase each other.

## Average instead of descending

The codebook loss is one way to move codewords. A steadier way, used by SoundStream and EnCodec, runs k-means online. Each code keeps two running statistics: $N_k$, how many vectors chose it, and $m_k$, the sum of those vectors. After each batch, both decay toward the batch's values, and the codeword is their ratio:

$$
N_k \leftarrow \gamma N_k + (1 - \gamma)\, n_k, \qquad
m_k \leftarrow \gamma m_k + (1 - \gamma) \sum_{i:\, k_i^* = k} z_i, \qquad
e_k = \frac{m_k}{\tilde N_k}.
$$

Read it aloud: $n_k$ is how many vectors in this batch chose code $k$. The new count is $\gamma$ times the old count plus $1 - \gamma$ times the batch count, and the sum updates the same way. The codeword is the running sum over the running count, which is a running mean of the vectors that chose it. SoundStream and EnCodec use $\gamma = 0.99$. The symbol $\tilde N_k$ is $N_k$ after Laplace smoothing, $\tilde N_k = \frac{N_k + \epsilon}{\sum_j N_j + K\epsilon} \sum_j N_j$, which keeps an unused code from dividing by zero. The hand example below sets $\epsilon = 0$.

```python
n_k = np.bincount(idx, minlength=K)
s_k = np.zeros_like(sums); np.add.at(s_k, idx, z)
counts = decay * counts + (1 - decay) * n_k
sums = decay * sums + (1 - decay) * s_k
```

Take three codes at $(0, 0)$, $(1, 0)$, and $(0, 1)$, each with count 1 and sum equal to itself, and $\gamma = 0.9$. A batch of four vectors, $(0.2, 0)$, $(0, 0.2)$, $(0.9, 0.1)$, and $(1.3, -0.1)$, chooses codes 0, 0, 1, and 1. No vector chooses code 2.

| Code | $n_k$ | Batch sum | $N_k$ | $m_k$ | $e_k = m_k / N_k$ |
|---|---|---|---|---|---|
| 0 | 2 | $(0.2, 0.2)$ | $0.9 + 0.2 = 1.1$ | $(0.02, 0.02)$ | $(0.0182, 0.0182)$ |
| 1 | 2 | $(2.2, 0.0)$ | $1.1$ | $(1.12, 0.00)$ | $(1.0182, 0.0000)$ |
| 2 | 0 | $(0, 0)$ | $0.9$ | $(0.00, 0.90)$ | $(0.0000, 1.0000)$ |

Code 0 moves toward its two vectors, and code 1 moves toward their mean, $(1.1, 0)$. Code 2 doesn't move, but its count shrinks by 10 percent. After ten more batches that ignore it, its count is $0.9^{11} \approx 0.31$.

## Codebook collapse and dead codes

That shrinking count is a warning. A code that no vector chooses gets no update, so it can stay unused forever while a few popular codes take every vector. This is *codebook collapse*: a codebook of 1024 entries behaves like one of 20 while still costing 10 bits.

You measure it with the *perplexity* of the usage histogram, the number of codes in effective use (Figure 3):

$$
\text{perplexity} = \exp\left(-\sum_k \bar p_k \ln \bar p_k\right).
$$

Read it aloud: $\bar p_k$ is the fraction of vectors that chose code $k$. Exponentiating the entropy of those fractions gives $K$ for uniform use and 1 when every vector picks one code. For fractions $(0.7, 0.1, 0.1, 0.1)$ the entropy is $0.9405$ nats and the perplexity is $e^{0.9405} = 2.56$ codes in use, out of four.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Three histograms of code use over four codes, with perplexities 4, 2.56, and 1">
<line x1="24.0" y1="200.0" x2="166.0" y2="200.0" class="stroke-muted"/>
<rect x="30.0" y="175.0" width="28.0" height="25.0" class="fill-a"/>
<text x="44.0" y="216.0" class="fig-small" text-anchor="middle">0</text>
<rect x="64.0" y="175.0" width="28.0" height="25.0" class="fill-a"/>
<text x="78.0" y="216.0" class="fig-small" text-anchor="middle">1</text>
<rect x="98.0" y="175.0" width="28.0" height="25.0" class="fill-a"/>
<text x="112.0" y="216.0" class="fig-small" text-anchor="middle">2</text>
<rect x="132.0" y="175.0" width="28.0" height="25.0" class="fill-a"/>
<text x="146.0" y="216.0" class="fig-small" text-anchor="middle">3</text>
<text x="30.0" y="50.0" class="fig-bold" text-anchor="start">Uniform use</text>
<text x="30.0" y="70.0" class="fig-small" text-anchor="start">perplexity 4.00</text>
<line x1="219.0" y1="200.0" x2="361.0" y2="200.0" class="stroke-muted"/>
<rect x="225.0" y="130.0" width="28.0" height="70.0" class="fill-a"/>
<text x="239.0" y="216.0" class="fig-small" text-anchor="middle">0</text>
<rect x="259.0" y="190.0" width="28.0" height="10.0" class="fill-a"/>
<text x="273.0" y="216.0" class="fig-small" text-anchor="middle">1</text>
<rect x="293.0" y="190.0" width="28.0" height="10.0" class="fill-a"/>
<text x="307.0" y="216.0" class="fig-small" text-anchor="middle">2</text>
<rect x="327.0" y="190.0" width="28.0" height="10.0" class="fill-a"/>
<text x="341.0" y="216.0" class="fig-small" text-anchor="middle">3</text>
<text x="225.0" y="50.0" class="fig-bold" text-anchor="start">A few codes dominate</text>
<text x="225.0" y="70.0" class="fig-small" text-anchor="start">perplexity 2.56</text>
<line x1="414.0" y1="200.0" x2="556.0" y2="200.0" class="stroke-muted"/>
<rect x="420.0" y="100.0" width="28.0" height="100.0" class="fill-a"/>
<text x="434.0" y="216.0" class="fig-small" text-anchor="middle">0</text>
<text x="468.0" y="216.0" class="fig-small" text-anchor="middle">1</text>
<text x="502.0" y="216.0" class="fig-small" text-anchor="middle">2</text>
<text x="536.0" y="216.0" class="fig-small" text-anchor="middle">3</text>
<text x="420.0" y="50.0" class="fig-bold" text-anchor="start">Collapsed</text>
<text x="420.0" y="70.0" class="fig-small" text-anchor="start">perplexity 1.00</text>
<text x="300.0" y="245.0" class="fig-small" text-anchor="middle">code index (K = 4)</text>
</svg>
Figure 3. Perplexity counts the codes in effective use: 4.00 for uniform use of four codes, 2.56 when one code takes 70 percent, and 1.00 when one code takes everything.
```

The standard repair is *dead-code restart*. At each step, find every code whose EMA count has fallen below a threshold, and replace it with a random encoder output from the batch. SoundStream uses a decay of 0.99 and a threshold of 2. Restarting also resets the code's count to the threshold and its sum to the new vector times the threshold, so the next EMA step doesn't pull the code back to where it was. DAC avoids most dead codes another way: it projects to 8 dimensions and L2-normalizes both sides before the lookup, which turns distance into cosine similarity.

In this chapter's **Run**, a codebook of 64 codes initialized far from the data collapses to a perplexity of 1.0. With dead-code restart, the same training reaches 59.1.

> [!WARNING]
> The reconstruction loss and the codebook loss can look healthy while three codes do all the work. Log the perplexity of every codebook during training.

> [!KEY]
> A codebook is a short vocabulary: a vector becomes the index of its nearest codeword. The straight-through estimator lets the encoder learn through that choice, the two losses keep encoder and codebook together, and dead-code restart keeps every word in use.

> [!RECAP]
> - The nearest codeword comes from $\lVert z \rVert^2 - 2 z^\top e + \lVert e \rVert^2$, and its index costs $\log_2 K$ bits.
> - $z_e + \operatorname{sg}[z_q - z_e]$ has the value of $z_q$ and the gradient of $z_e$.
> - The codebook loss detaches $z_e$, and the commitment loss detaches $e$.
> - EMA updates keep a decayed count and sum per code, and the codeword is their ratio.
> - Perplexity counts the codes in use, and dead-code restart revives the unused ones.

## Build it step by step

Build the functions in the order the checks test them.

1. **`nearest_codeword(z, codebook)`** returns `(indices, quantized)` of shapes `(N,)` and `(N, D)`. Use the expansion and `argmin(axis=1)`. The common mistake is broadcasting to `(N, K, D)`.
2. **`straight_through(z_e, z_q)`** is one line with `.detach()`. The common mistake is returning `z_q.detach()`, which has the right value and no gradient.
3. **`vq_losses(z_e, codebook)`** finds the indices from detached tensors, gathers `e = codebook[indices]`, and returns the straight-through output, the indices, and the two losses. The common mistake is swapping the two `.detach()` calls.
4. **`ema_update(counts, sums, z, indices, decay, eps)`** uses `np.bincount` and `np.add.at`, decays both statistics, smooths the counts, and divides. The common mistake is dividing by unsmoothed counts, which gives `nan` for an unused code.
5. **`restart_dead_codes(codebook, counts, sums, z, threshold, rng)`** draws a row of `z` with `rng.choice` for each code with `counts < threshold`, and resets its count to `threshold` and its sum to the row times `threshold`. The common mistake is resetting the codeword but not the sum.
6. **`codebook_perplexity(indices, K)`** is a normalized `np.bincount`, then the exponential of the entropy over the nonzero entries.

## Cost and scaling

Doubling $K$ adds one bit and doubles the lookup, which costs $O(NKD)$ for $N$ vectors. The distance table holds $N \times K$ floats, about 80 MB in float32 for 10,000 frames and $K = 2048$. To add bits cheaply, stack several small codebooks, as the next chapter, [Residual and scalar quantization](/lessons/056-residual-and-scalar-quantization), shows.

## Where this breaks

**Collapse from a bad start.** A random initialization far from the encoder outputs leaves most codes unused. Initialize from k-means on the first batch, as SoundStream does, restart dead codes, or switch to a quantizer with no learned codebook.

**Straight-through bias.** The estimator pretends quantization is the identity. When codewords are far apart, that gradient is a poor guide, because the true effect of moving $z_e$ is a jump to another codeword.

## The exercise

You build `nearest_codeword`, `straight_through`, `vq_losses`, `ema_update`, `restart_dead_codes`, and `codebook_perplexity`. The checks compare `nearest_codeword` with brute force, test the straight-through value and gradient, and confirm that each VQ loss sends its gradient only where it belongs, against finite differences. They reproduce the hand-worked EMA step, test that restart touches only dead codes, and check perplexity at $K$, 1, and 2.

Press **Run** to train a 64-code codebook on 8-sample frames of a synthetic voice, which is 12 kbps at 16 kHz. It trains twice from the same far-away start, without and with dead-code restart, prints the perplexity and SNR of each, and plays the voice coded with the repaired codebook.
