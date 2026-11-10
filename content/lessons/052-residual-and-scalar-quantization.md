---
title: Residual and scalar quantization
slug: 052-residual-and-scalar-quantization
part: "Part 6 — Codecs and token models"
summary: Spend a bit budget in small stages with residual quantization, or round on a fixed grid with FSQ, and see why neither needs a giant codebook.
minutes: 30
gpu: false
objectives:
  - Explain why a 80-bit index per frame cannot come from one codebook, and why eight 10-bit stages can.
  - Quantize a vector in stages, each stage coding the residual the earlier stages left, and read the residual norms.
  - Decode only the first few stages to get a coarser version of the same vector.
  - Round a vector onto a fixed grid with finite scalar quantization, and turn the grid point into one integer.
prereqs:
  - 003-quantization-and-dither
  - 051-vector-quantization
exercise: 052-residual-and-scalar-quantization
---

# Residual and scalar quantization

A speech codec at 24 kHz produces 75 frames per second. If its budget is 6 kbps, each frame can spend $6000 / 75 = 80$ bits. An index with 80 bits points into a table of $2^{80}$ entries, about $10^{24}$ codewords. No machine can store that table, and no machine can search it for the nearest entry.

The previous chapter gave you a single codebook of about a thousand entries, which spends 10 bits. This chapter answers the question that follows: how do you spend the other 70 bits? You will see two answers. One stacks small codebooks so that each fixes the mistakes of the one before. The other drops the learned codebook and rounds each number to a fixed grid.

## What you already know

- A codebook is a $(K, D)$ matrix, and quantizing a vector means replacing it with its nearest codeword, which costs $\log_2 K$ bits (see [Vector quantization](/lessons/051-vector-quantization)).
- `nearest_codeword(z, codebook)` returns each vector's index and its codeword.
- The straight-through estimator passes a gradient through a rounding step as if the step were the identity.
- Uniform quantization rounds a number to the nearest of $2^N$ levels (see [Quantization and dither](/lessons/003-quantization-and-dither)).
- A vector's length is its norm, $\lVert v \rVert = \sqrt{\sum_i v_i^2}$, in numpy `np.linalg.norm(v)`.

## Why one codebook cannot do it

The obvious plan is to enlarge the codebook until the quantization error is small enough. Each doubling of $K$ adds one bit, so 80 bits means $K = 2^{80}$. That fails twice. The table does not fit in memory, and even a table of $2^{20}$ entries makes every lookup compare against a million codewords. Training fails too, because most codewords would never be the nearest to any training vector.

Think of giving a street address. You do not keep a single list of every house on Earth. You say the country, then the city, then the street, then the house number. Each step is a short list, and each step narrows what the previous one left open. *Residual vector quantization* (RVQ) does the same with vectors. The first codebook says roughly where the vector is. The second says how far off the first guess was. The third corrects the second, and so on. Eight lists of 1024 entries cost 80 bits, and you store and search only $8 \times 1024 = 8192$ codewords.

## Quantize what the last stage missed

Call what a stage failed to capture the *residual*: the input minus the codeword that stage chose. Stage 1 quantizes the input, stage 2 quantizes stage 1's residual, and stage 3 quantizes stage 2's residual. SoundStream (Zeghidour et al., 2021) brought this to neural audio codecs.

### Work two stages by hand

Take the input $x = (0.9, -0.3)$. Stage 1 has four codewords, the unit directions $(1, 0)$, $(0, 1)$, $(-1, 0)$, and $(0, -1)$. The squared distances from $x$ are $0.10$, $2.50$, $3.70$, and $1.30$, so stage 1 picks index 0, the codeword $(1, 0)$. The residual is $r_1 = x - (1, 0) = (-0.1, -0.3)$, with length $0.316$.

Stage 2 has four small codewords: $(0.1, 0.1)$, $(-0.1, -0.2)$, $(0.2, -0.2)$, and $(-0.2, 0.2)$. It quantizes $r_1$, not $x$. The squared distances from $r_1$ are $0.20$, $0.01$, $0.10$, and $0.26$, so it picks index 1, the codeword $(-0.1, -0.2)$. The next residual is $r_2 = (-0.1, -0.3) - (-0.1, -0.2) = (0, -0.1)$, with length $0.100$.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Two residual quantizer stages in two dimensions: stage 1 moves from the origin to (1, 0), stage 2 adds (-0.1, -0.2), and a gap of 0.1 to the input (0.9, -0.3) remains">
<defs><marker id="c045-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="50" y1="50" x2="490.00000000000006" y2="50" class="stroke-grid"/>
<line x1="50" y1="10.0" x2="50" y2="210.0" class="stroke-grid"/>
<text x="42" y="54" class="fig-small" text-anchor="end">0</text>
<text x="510.00000000000006" y="54" class="fig-small" text-anchor="end">dim 1</text>
<text x="56" y="22.0" class="fig-small">dimension 2</text>
<line x1="50" y1="50" x2="441" y2="50" class="stroke-b" stroke-width="2.5" marker-end="url(#c045-arrow)"/>
<text x="250.0" y="40" class="fig-small" text-anchor="middle">stage 1 codeword (1, 0)</text>
<line x1="450" y1="50" x2="412.0" y2="122.0" class="stroke-c" stroke-width="2.5" marker-end="url(#c045-arrow)"/>
<text x="430.0" y="78.0" class="fig-small" text-anchor="end">stage 2 codeword (-0.1, -0.2)</text>
<circle cx="410.0" cy="170.0" r="6" class="fill-a"/>
<text x="424.0" y="175.0" class="fig-small">input x = (0.9, -0.3)</text>
<circle cx="410.0" cy="130.0" r="4" class="fill-ink"/>
<line x1="410.0" y1="134.0" x2="410.0" y2="164.0" class="stroke-bad" stroke-width="2"/>
<text x="230.0" y="150.0" class="fig-small" text-anchor="middle">what is left after two stages: (0, -0.1), length 0.100</text>
<text x="230.0" y="166.0" class="fig-small" text-anchor="middle">after stage 1 it was (-0.1, -0.3), length 0.316</text>
</svg>
Figure 1. The input is reached in two jumps. Notice that stage 2's codewords are small, because they only have to cover what stage 1 left, so a few entries can place the sum accurately.
```

The code for this input is $(0, 1)$, which costs $2 \times \log_2 4 = 4$ bits. The reconstruction is the sum of the chosen codewords, $(1, 0) + (-0.1, -0.2) = (0.9, -0.2)$, which differs from $x$ by exactly $r_2$.

Each stage works on a smaller vector than the one before. The residual lengths fall from $\lVert x \rVert = 0.949$ to $0.316$ to $0.100$.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Residual length shrinks from 0.949 to 0.316 to 0.100 as two 2-bit stages are sent">
<text x="20" y="22" class="fig-bold">Length of what is still unexplained</text>
<text x="20" y="62" class="fig-small">input x</text>
<rect x="150" y="42" width="360.5" height="28" class="fill-a"/>
<text x="518.5" y="61" class="fig-small">0.949  (0 bits sent)</text>
<text x="20" y="108" class="fig-small">after stage 1</text>
<rect x="150" y="88" width="120.2" height="28" class="fill-a"/>
<text x="278.2" y="107" class="fig-small">0.316  (2 bits sent)</text>
<text x="20" y="154" class="fig-small">after stage 2</text>
<rect x="150" y="134" width="38.0" height="28" class="fill-a"/>
<text x="196.0" y="153" class="fig-small">0.100  (4 bits sent)</text>
<line x1="150" y1="36" x2="150" y2="176" class="stroke-muted"/>
</svg>
Figure 2. Each 2-bit stage shrinks the unexplained part of the input. A codec gets higher quality by adding stages, not by growing one codebook.
```

### The formula

Here is the summary. Start with the residual equal to the input, quantize it at each stage with that stage's codebook $Q_i$, and subtract:

$$
r_0 = z, \qquad q_i = Q_i(r_{i-1}), \qquad r_i = r_{i-1} - q_i, \qquad \hat z = \sum_{i=1}^{N_q} q_i
$$

Read it aloud: the residual before stage 1 is the input $z$. At stage $i$, the chosen codeword $q_i$ is the nearest entry of codebook $i$ to the previous residual $r_{i-1}$, and the next residual is what that codeword missed. The reconstruction $\hat z$ is the sum of every stage's codeword. A frame costs $N_q \log_2 K$ bits, one index per stage.

```python
residual = x.copy()
for i, C in enumerate(codebooks):
    idx, q = nearest_codeword(residual, C)
    recon += q
    residual -= q
```

The codec's training loss uses the same residuals. EnCodec sums the commitment loss over stages, which for the example is $\lVert r_1 \rVert^2 + \lVert r_2 \rVert^2 = 0.10 + 0.01 = 0.11$.

## Fewer stages, coarser sound

The first stage captures the coarse shape and each later stage refines it. So you can stop early: decode only the first $n$ stages, and you get a coarser version of the same vector at fewer bits. This is how a single codec offers several bitrates. EnCodec at 24 kHz has up to 32 codebooks of 1024 entries. Using 2 of them gives 1.5 kbps and using all 32 gives 24 kbps, from one trained model.

For the example, decoding with one stage gives $(1, 0)$, an error of length $0.316$. Decoding with two gives $(0.9, -0.2)$, an error of $0.100$.

The following demo runs RVQ on 2-D points, one stage at a time.

```demo
{"id": "rvq-2d", "caption": "Raise the stages used one at a time and watch each codebook tile a smaller region around the leftover error, while the error readout falls with every stage."}
```

> [!KEY]
> Residual quantization turns an exponential table into a sum. $N_q$ codebooks of $K$ entries give $K^{N_q}$ possible reconstructions, for the storage and search cost of only $N_q K$ codewords.

## Round on a fixed grid instead

RVQ still has learned codebooks, and learned codebooks can collapse: a few codewords win every vector while the rest sit unused. *Finite scalar quantization* (FSQ) removes that failure by removing the codebook (Mentzer et al., 2024). You already know how to round one number. FSQ rounds each number in a short vector, and calls the set of all rounded combinations the codebook.

The idea has three moves. First, project the encoder output to a few dimensions, such as 3. Second, squash each dimension into a fixed range with $\tanh$, so no value runs off the grid. Third, stretch to the grid's size and round to the nearest integer. With 5 levels, each dimension ends up as one of $-2, -1, 0, 1, 2$, and three dimensions give $5^3 = 125$ grid points. There are no learned codewords that can fall out of use, because the grid is fixed and every grid point is reachable.

$$
\hat z_i = \operatorname{round}\left(\left\lfloor \tfrac{L_i}{2} \right\rfloor \tanh(z_i)\right)
$$

Read it aloud: in dimension $i$, which has $L_i$ levels, squash $z_i$ into $(-1, 1)$, multiply by $\lfloor L_i / 2 \rfloor$ so the range is $\pm 2$ for $L_i = 5$, and round. Rounding has no gradient, so FSQ uses the straight-through trick on the rounding step. The gradient is then the derivative of the $\tanh$ part.

```python
bounded = half * torch.tanh(z)
z_hat = bounded + (torch.round(bounded) - bounded).detach()
```

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="FSQ with five levels: the curve 2 tanh(z) and its rounded staircase, with the three inputs 0.3, -2.0, and 1.2 marked">
<line x1="90" y1="250" x2="510" y2="250" class="stroke-grid"/>
<text x="82" y="254" class="fig-small" text-anchor="end">-2</text>
<line x1="90" y1="200" x2="510" y2="200" class="stroke-grid"/>
<text x="82" y="204" class="fig-small" text-anchor="end">-1</text>
<line x1="90" y1="150" x2="510" y2="150" class="stroke-grid"/>
<text x="82" y="154" class="fig-small" text-anchor="end">0</text>
<line x1="90" y1="100" x2="510" y2="100" class="stroke-grid"/>
<text x="82" y="104" class="fig-small" text-anchor="end">1</text>
<line x1="90" y1="50" x2="510" y2="50" class="stroke-grid"/>
<text x="82" y="54" class="fig-small" text-anchor="end">2</text>
<text x="90" y="270" class="fig-small" text-anchor="middle">-3</text>
<text x="160" y="270" class="fig-small" text-anchor="middle">-2</text>
<text x="230" y="270" class="fig-small" text-anchor="middle">-1</text>
<text x="300" y="270" class="fig-small" text-anchor="middle">0</text>
<text x="370" y="270" class="fig-small" text-anchor="middle">1</text>
<text x="440" y="270" class="fig-small" text-anchor="middle">2</text>
<text x="510" y="270" class="fig-small" text-anchor="middle">3</text>
<line x1="90" y1="150" x2="510" y2="150" class="stroke-muted"/>
<polyline points="90.0,249.5 93.5,249.5 97.0,249.4 100.5,249.3 104.0,249.3 107.5,249.2 111.0,249.1 114.5,249.0 118.0,248.9 121.5,248.8 125.0,248.7 128.5,248.5 132.0,248.4 135.5,248.2 139.0,248.0 142.5,247.8 146.0,247.6 149.5,247.3 153.0,247.0 156.5,246.7 160.0,246.4 163.5,246.0 167.0,245.6 170.5,245.2 174.0,244.7 177.5,244.1 181.0,243.5 184.5,242.9 188.0,242.2 191.5,241.4 195.0,240.5 198.5,239.6 202.0,238.5 205.5,237.4 209.0,236.2 212.5,234.8 216.0,233.4 219.5,231.8 223.0,230.0 226.5,228.2 230.0,226.2 233.5,224.0 237.0,221.6 240.5,219.1 244.0,216.4 247.5,213.5 251.0,210.4 254.5,207.2 258.0,203.7 261.5,200.1 265.0,196.2 268.5,192.2 272.0,188.0 275.5,183.6 279.0,179.1 282.5,174.5 286.0,169.7 289.5,164.9 293.0,160.0 296.5,155.0 300.0,150.0 303.5,145.0 307.0,140.0 310.5,135.1 314.0,130.3 317.5,125.5 321.0,120.9 324.5,116.4 328.0,112.0 331.5,107.8 335.0,103.8 338.5,99.9 342.0,96.3 345.5,92.8 349.0,89.6 352.5,86.5 356.0,83.6 359.5,80.9 363.0,78.4 366.5,76.0 370.0,73.8 373.5,71.8 377.0,70.0 380.5,68.2 384.0,66.6 387.5,65.2 391.0,63.8 394.5,62.6 398.0,61.5 401.5,60.4 405.0,59.5 408.5,58.6 412.0,57.8 415.5,57.1 419.0,56.5 422.5,55.9 426.0,55.3 429.5,54.8 433.0,54.4 436.5,54.0 440.0,53.6 443.5,53.3 447.0,53.0 450.5,52.7 454.0,52.4 457.5,52.2 461.0,52.0 464.5,51.8 468.0,51.6 471.5,51.5 475.0,51.3 478.5,51.2 482.0,51.1 485.5,51.0 489.0,50.9 492.5,50.8 496.0,50.7 499.5,50.7 503.0,50.6 506.5,50.5 510.0,50.5" class="stroke-a" fill="none" stroke-width="2"/>
<polyline points="90.0,250.0 232.1,250.0 232.1,200.0 282.1,200.0 282.1,150.0 318.2,150.0 318.2,100.0 368.2,100.0 368.2,50.0 510.0,50.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="321.0" cy="100" r="5" class="fill-b"/>
<circle cx="321.0" cy="120.8687387548409" r="4" class="fill-a"/>
<circle cx="160.0" cy="250" r="5" class="fill-b"/>
<circle cx="160.0" cy="246.4027580075817" r="4" class="fill-a"/>
<circle cx="384.0" cy="50" r="5" class="fill-b"/>
<circle cx="384.0" cy="66.63453929878447" r="4" class="fill-a"/>
<text x="331.0" y="116" class="fig-small">0.3 to 1</text>
<text x="172" y="228" class="fig-small">-2.0 to -2</text>
<text x="322.0" y="40" class="fig-small">1.2 to 2</text>
<text x="300" y="290" class="fig-small" text-anchor="middle">encoder output z (one dimension)</text>
<text x="90" y="18" class="fig-small">quantized value</text>
<line x1="398.0" y1="227.5" x2="426.0" y2="227.5" class="stroke-a" stroke-width="2"/><text x="429.5" y="231.5" class="fig-small">2 tanh(z)</text>
<line x1="398.0" y1="247.5" x2="426.0" y2="247.5" class="stroke-b" stroke-width="2.5"/><text x="429.5" y="251.5" class="fig-small">rounded</text>
</svg>
Figure 3. The smooth blue curve is what the encoder sees in the backward pass, and the orange staircase is what the decoder receives. Notice that large inputs such as -2.0 saturate near the end of the grid instead of running off it.
```

### Work one FSQ code by hand

Take $L = [5, 5, 5]$ and $z = (0.3, -2.0, 1.2)$:

- Squash and stretch: $2\tanh(z) = (0.583, -1.928, 1.667)$.
- Round: $\hat z = (1, -2, 2)$.
- Shift each value by $\lfloor L_i / 2 \rfloor = 2$ to get a digit from 0 to 4: $(3, 0, 4)$.
- Read the digits as a number. The first digit counts ones, the second counts fives, and the third counts twenty-fives: $3 + 0 \times 5 + 4 \times 25 = 103$.

That last step is a *mixed-radix* number, the way hours, minutes, and seconds are one number written in bases 24, 60, and 60. In general:

$$
\text{idx} = \sum_i \left(\hat z_i + \left\lfloor \tfrac{L_i}{2} \right\rfloor\right) \prod_{j < i} L_j
$$

Read it aloud: shift each value to a digit, weight digit $i$ by the product of the sizes of all earlier dimensions, and add. The inverse divides by those weights with `//` and takes the remainder with `%`.

Real systems use FSQ because of how well it uses its grid. The FSQ paper recommends levels such as $[8, 5, 5, 5]$ for about 1000 codes. CosyVoice 2's speech tokenizer uses 8 dimensions of 3 levels each, $3^8 = 6561$ codes, and reports all of them in use. XCodec2 uses one FSQ codebook of $4^8 = 65{,}536$ codes.

> [!TIP]
> An even number of levels has no grid point at 0, so the reference FSQ code shifts the $\tanh$ by half a step for even $L_i$. The exercise uses odd levels, which need no offset.

> [!RECAP]
> - One codebook with an 80-bit index would need $2^{80}$ entries, so a codec spends its bits in stages.
> - RVQ quantizes the residual stage by stage, and the reconstruction is the sum of the chosen codewords.
> - Decoding the first $n$ stages gives a coarser vector at $n \log_2 K$ bits, which is how one codec serves many bitrates.
> - FSQ rounds $\lfloor L/2 \rfloor \tanh(z)$ to a fixed grid, and indexes the grid as a mixed-radix number.

## Build it step by step

`nearest_codeword` is provided in the starter in full, because you built it in the previous chapter. Build the functions in the order the checks test them.

1. **`rvq_encode(x, codebooks)`.** Keep a running residual. At each stage, call `nearest_codeword` on the residual, add the codeword to the reconstruction, subtract it from the residual, and record `np.linalg.norm(residual, axis=1)`. It returns the codes `(N, N_q)`, the reconstruction `(N, D)`, and the norms `(N, N_q + 1)`. The most common mistake is quantizing `x` at every stage instead of the residual, which makes every stage choose the same codeword.
2. **`rvq_decode(codes, codebooks)`.** Add `codebooks[i][codes[:, i]]` for each column that `codes` has. It takes no search, and it accepts fewer columns than there are codebooks.
3. **`fsq_quantize(z, levels)`.** Multiply `torch.tanh(z)` by $\lfloor L_i / 2 \rfloor$, then round with the straight-through step. The common mistake is `torch.round(bounded)` alone, which has zero gradient.
4. **`fsq_index` and `fsq_unindex`.** Shift to digits, multiply by the basis `[1, L_0, L_0 L_1, ...]`, and invert with `//` and `%`. The common mistake is forgetting the shift by $\lfloor L_i / 2 \rfloor$, which gives negative digits.

## Where this breaks

**Late stages that learn nothing.** If you train every stage on the original vectors instead of on residuals, later codebooks sit far from the data they will see. Train each stage on the residuals the earlier stages leave, as the **Run** does.

**Bits that do not buy quality.** Each added stage costs the same $\log_2 K$ bits but removes less error, because the residual is already small. Codecs add stages until the next one stops paying for itself.

**Too few FSQ levels.** With 3 levels per dimension, every dimension carries about 1.6 bits and the grid is coarse. Use at least 5 levels per dimension, or more dimensions, as the FSQ paper advises.

## The exercise

You build `rvq_encode`, `rvq_decode`, `fsq_quantize`, `fsq_index`, and `fsq_unindex`. The checks reproduce the hand-worked two-stage example, test that each stage quantizes the residual, and test that residual norms never grow and that decoding a prefix of the stages works. They also reproduce the FSQ example with index 103, check the straight-through gradient against a finite difference, and confirm that the index is a bijection over the whole grid.

Press **Run** to train a 64-code, four-stage RVQ with k-means on 8-sample frames of a synthetic voice. You hear it at one stage (12 kbps) and at all four (48 kbps), and see the residual norm and SNR at each stage.
