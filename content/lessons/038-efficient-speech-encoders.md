---
title: Efficient speech encoders
slug: 038-efficient-speech-encoders
part: "Part 4 — Neural speech recognition"
summary: Cut an encoder's cost with 8x subsampling, a Zipformer U-Net, and a state-space layer, and count what each saves.
minutes: 45
gpu: false
objectives:
  - Count an encoder's FLOPs as a linear term plus a quadratic attention term, and predict the saving from 4x to 8x subsampling.
  - Build a depthwise-separable subsampler and count its parameters against a dense one.
  - Write a Zipformer-style downsample, upsample, and bypass.
  - Show that a diagonal state-space layer gives the same output as a recurrence and as a convolution, and that a chunked pass with a carried state matches the full pass.
prereqs:
  - 028-neural-acoustic-models
  - 036-streaming-encoders
  - 037-conformers-and-specaugment
exercise: 038-efficient-speech-encoders
---

# Efficient speech encoders

A Conformer reads 10 seconds of audio as 250 frames, and every layer lets each frame look at every other frame. That is fine on a server GPU. On a phone, or on a server that must handle thousands of streams at once, the encoder is where almost all of the compute goes, because it runs a deep stack of layers over every frame of every second of audio.

Two questions follow. Can the encoder run on fewer frames without losing words? And can a layer do its work without scoring every pair of frames? Open recipes answer yes to both: FastConformer, Zipformer, Branchformer, and encoders built on state-space layers. You will count what each saves, and write the pieces that make them work.

## What you already know

- A 4x Conv2d front end turns 100 feature frames per second into 25, from [Neural acoustic models](/lessons/028-neural-acoustic-models).
- A chunked mask lets an encoder stream, at the price of look-ahead delay, from [Streaming encoders](/lessons/036-streaming-encoders).
- A Conformer block is two feed-forward halves around self-attention and a convolution module, from [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment).
- Self-attention scores every pair of frames, so its cost grows with the square of the frame count.
- A recurrence carries a state from one step to the next.

## Where the compute goes

Take an encoder with $L$ layers, width $d$, and $T$ frames after subsampling. Count one multiply-accumulate as 2 FLOPs, and count only the large matrix products. One layer has three parts:

- Four $d \times d$ projections (query, key, value, output): $4Td^2$ multiply-accumulates.
- A feed-forward block of width $4d$, which is two matrices: $8Td^2$.
- Attention scores and the weighted sum of values, each $T \times T \times d$: $2T^2d$.

So the encoder costs

$$
F = L\left(24\,T\,d^2 + 4\,T^2 d\right), \qquad T = \lfloor \text{frames} / s \rfloor
$$

Read it aloud: a term that grows in step with the frame count, and a term that grows with its square. The factor $s$ is the subsampling factor. In numpy, `layers * (24 * t * d * d + 4 * t * t * d)`.

Work one layer, $d = 256$, and 10 seconds of audio at 100 frames per second, so 1000 frames:

| Subsampling | $T$ | Linear term | Quadratic term | Total |
|---|---|---|---|---|
| 4x | 250 | 393,216,000 | 64,000,000 | 457,216,000 |
| 8x | 125 | 196,608,000 | 16,000,000 | 212,608,000 |

Halving $T$ halves the linear term and quarters the quadratic term, so the saving, $457{,}216{,}000 / 212{,}608{,}000 \approx 2.15$, falls between 2 and 4. It sits close to 2 because the linear term dominates at $d = 256$. On 60 seconds of audio the saving grows to about 2.66, because the quadratic term has grown.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Stacked bars of encoder compute in GFLOPs for 2x, 4x, 8x, and 16x subsampling of 10 seconds of audio">
<line x1="70" y1="200" x2="530" y2="200" class="stroke-ink"/>
<rect x="100" y="86.8" width="60" height="113.2" class="fill-a"/>
<rect x="100" y="50.0" width="60" height="36.8" class="fill-b"/>
<text x="130" y="42.0" text-anchor="middle">16.7</text>
<text x="130" y="220" text-anchor="middle">2x (500 frames)</text>
<rect x="210" y="143.4" width="60" height="56.6" class="fill-a"/>
<rect x="210" y="134.2" width="60" height="9.2" class="fill-b"/>
<text x="240" y="126.2" text-anchor="middle">7.3</text>
<text x="240" y="220" text-anchor="middle">4x (250 frames)</text>
<rect x="320" y="171.7" width="60" height="28.3" class="fill-a"/>
<rect x="320" y="169.4" width="60" height="2.3" class="fill-b"/>
<text x="350" y="161.4" text-anchor="middle">3.4</text>
<text x="350" y="220" text-anchor="middle">8x (125 frames)</text>
<rect x="430" y="186.0" width="60" height="14.0" class="fill-a"/>
<rect x="430" y="185.4" width="60" height="0.6" class="fill-b"/>
<text x="460" y="177.4" text-anchor="middle">1.6</text>
<text x="460" y="220" text-anchor="middle">16x (62 frames)</text>
<rect x="70" y="14" width="12" height="12" class="fill-a"/><text x="88" y="25" class="fig-small">projections and feed-forward (linear in T)</text>
<rect x="340" y="14" width="12" height="12" class="fill-b"/><text x="358" y="25" class="fig-small">attention scores (quadratic)</text>
<text x="300" y="242" text-anchor="middle" class="fig-small">subsampling factor, with encoder frames for 10 s; bar labels are GFLOPs, 16 layers, d = 256</text>
</svg>
Figure 1. Compute for 16 layers at d = 256 and 10 seconds of audio. Each doubling of the factor roughly halves the cost, and the attention part (orange) shrinks faster than the rest.
```

> [!KEY]
> Encoder cost has a part that is linear in the frame count and a part that is quadratic. Lowering the frame rate cuts both, and the quadratic part fastest.

## FastConformer: subsample by 8, cheaply

The obvious route is a third stride-2 convolution. It works, but a dense Conv2d from $C$ to $C$ channels with a 3 by 3 kernel has $9C^2$ weights, so the front end grows costly.

FastConformer (Rekesh et al., 2023) keeps the Conformer block and redesigns the front end around a downsampling schema. NeMo's documentation describes 8x depthwise convolutional subsampling with 256 channels. This chapter builds that idea in separable form: a dense first stage on one input channel, then stages that each split into two cheaper convolutions.

A **depthwise** convolution filters each channel on its own (`groups=channels`), so it has $9C$ weights instead of $9C^2$. A **pointwise** convolution has a 1 by 1 kernel and mixes channels, with $C^2$ weights. Together they do the work of one dense convolution for much less. With bias terms and $C = 256$ channels over 3 stages:

- Dense: $(9 \cdot 256 + 256) + 2\,(9 \cdot 256^2 + 256) = 1{,}182{,}720$ parameters.
- Separable: $2560 + 2\,[(9 \cdot 256 + 256) + (256^2 + 256)] = 139{,}264$ parameters, which is 8.5 times fewer.

Each stage uses kernel 3, stride 2, and padding 1, so it maps a length $n$ to $\lfloor (n-1)/2 \rfloor + 1$. Three stages turn 83 frames into 42, then 21, then 11.

**The trade.** At 8x, a frame lasts 80 ms, so the encoder runs on 12.5 frames per second, and the decoder must emit every subword token from them. A CTC or transducer loss needs at least as many frames as labels. The NeMo page reports an encoder about 2.4 times faster than a regular Conformer with no significant quality loss, and 2.7 times with 128 channels, where quality starts to degrade. The paper reports 2.8 times.

> [!WARNING]
> Check the frame rate against the token rate before you subsample harder. The margin between frames and labels shrinks with every factor of 2.

## Zipformer: a U-Net over time

FastConformer lowers the frame rate once, for every layer. Zipformer (Yao et al., ICLR 2024) lowers it further, but only in the middle of the network, like a U-Net in image segmentation. The middle layers build slow, abstract context and need less time resolution than the first and last layers.

The input is 100 frames per second. A convolutional embedding halves that to 50. Six stacks of blocks then run at 50, 25, 12.5, 6.25, 12.5, and 25 Hz, and a final downsample turns the output into 25 Hz.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Frame rate inside the six Zipformer stacks: 50, 25, 12.5, 6.25, 12.5, and 25 hertz, a U shape">
<line x1="50" y1="190" x2="540" y2="190" class="stroke-ink"/>
<rect x="70" y="50.0" width="56" height="140.0" class="fill-soft-a stroke-a"/>
<text x="98" y="42.0" text-anchor="middle">50 Hz</text>
<text x="98" y="210" text-anchor="middle">stack 1</text>
<rect x="148" y="120.0" width="56" height="70.0" class="fill-soft-a stroke-a"/>
<text x="176" y="112.0" text-anchor="middle">25 Hz</text>
<text x="176" y="210" text-anchor="middle">stack 2</text>
<rect x="226" y="155.0" width="56" height="35.0" class="fill-soft-a stroke-a"/>
<text x="254" y="147.0" text-anchor="middle">12.5 Hz</text>
<text x="254" y="210" text-anchor="middle">stack 3</text>
<rect x="304" y="172.5" width="56" height="17.5" class="fill-soft-a stroke-a"/>
<text x="332" y="164.5" text-anchor="middle">6.25 Hz</text>
<text x="332" y="210" text-anchor="middle">stack 4</text>
<rect x="382" y="155.0" width="56" height="35.0" class="fill-soft-a stroke-a"/>
<text x="410" y="147.0" text-anchor="middle">12.5 Hz</text>
<text x="410" y="210" text-anchor="middle">stack 5</text>
<rect x="460" y="120.0" width="56" height="70.0" class="fill-soft-a stroke-a"/>
<text x="488" y="112.0" text-anchor="middle">25 Hz</text>
<text x="488" y="210" text-anchor="middle">stack 6</text>
<text x="300" y="18" text-anchor="middle" class="fig-small">each stack downsamples on entry and upsamples back to 50 Hz on exit</text>
<text x="300" y="240" text-anchor="middle" class="fig-small">frames per second inside each stack, for a 50 Hz embedding</text>
</svg>
Figure 2. The six stacks run at 50, 25, 12.5, 6.25, 12.5, and 25 Hz. For 10 seconds of audio that is 500, 250, 125, 63, 125, and 250 frames, so the middle stack handles an eighth of the first one's frames.
```

Three small modules make this work:

- **Downsample** by a factor of 2 averages each pair of frames with two learnable scalar weights, normalized by a softmax. Frames $[1, 3]$ with weights $[1/4, 3/4]$ become $0.25 \cdot 1 + 0.75 \cdot 3 = 2.5$. This lesson pads an odd end by repeating the last frame.
- **Upsample** by a factor of 2 repeats each frame twice, then trims to the original length.
- **Bypass** mixes a module's input $x$ and output $y$ with a learned scale per channel:

$$
(1 - c) \odot x + c \odot y
$$

Read it aloud: when $c = 0$ the module is skipped, and when $c = 1$ it is fully used. The paper restricts $c$ to $[0.9, 1.0]$ at first and lowers the floor to 0.2 after 20,000 steps, so a deep stack starts close to straight-through. In numpy, `(1 - c) * x + c * y`.

## Branchformer in one paragraph

Branchformer (Peng et al., 2022) replaces the Conformer's serial attention-then-convolution with two parallel branches in every layer. One branch uses self-attention for long-range context, and the other uses a convolutional gating MLP for local context. A merge step combines them. The authors report that it matches or beats Conformer on speech recognition benchmarks, and that the separate branches allow cutting computation, even choosing the cost at inference time.

## State-space encoders: no attention matrix at all

Every encoder so far scores pairs of frames. A **state-space layer** does not. The [machine learning course](https://ml.ifkash.dev/) owns the concepts behind state-space models and linear attention. This section covers only what an encoder needs: one layer, its two equivalent forms, and its streaming property.

Take one input channel and $N$ states. A diagonal layer keeps a state vector $h$ and updates it at every frame:

$$
h_t = a \odot h_{t-1} + b\,x_t, \qquad y_t = c^\top h_t
$$

Read it aloud: each state decays by its own factor $a_n$ (between 0 and 1, so old input fades), takes in the incoming frame through $b_n$, and the output reads the states through $c_n$. In numpy, `h = a * h + b * x[t]` then `y[t] = c @ h`. A real encoder layer runs $d$ of these channels and mixes them with linear layers.

Work $a = [0.5, 0.25]$, $b = [1, 2]$, $c = [1, 1]$ on $x = [1, 0, 0, 2]$, starting from $h = 0$:

| $t$ | $x_t$ | $h_t$ | $y_t$ |
|---|---|---|---|
| 0 | 1 | $[1, 2]$ | 3 |
| 1 | 0 | $[0.5, 0.5]$ | 1 |
| 2 | 0 | $[0.25, 0.125]$ | 0.375 |
| 3 | 2 | $[2.125, 4.03125]$ | 6.15625 |

**The convolutional form.** Unroll the recurrence. Frame $x_j$ reaches the output at step $t$ after $t - j$ decays, so $y_t = \sum_{j \le t} c^\top(a^{t-j} \odot b)\,x_j$. This is a causal convolution with kernel

$$
K_k = \sum_n c_n\, a_n^{k}\, b_n, \qquad y_t = \sum_{k=0}^{t} K_k\, x_{t-k}
$$

The exponent is $k$, not $k - 1$, because the input enters the state before the output reads it. Here $K = [3, 1, 0.375, 0.15625, \dots]$, and $y_3 = K_3 \cdot 1 + K_0 \cdot 2 = 0.15625 + 6 = 6.15625$, the same as the table. Training uses this form, because a convolution runs in parallel over all frames. Inference uses the recurrence.

**Streaming.** The recurrence streams by construction. Its only memory is the state $h$, $N$ numbers per channel, which does not grow with the audio. There is no mask, no look-ahead, and no cache of past keys. You can also process a chunk in the parallel form, if you carry the state across the boundary. For a chunk of $L$ frames that starts from state $h$:

$$
y_t = \sum_{k=0}^{t} K_k\, x_{t-k} + c^\top\!\left(a^{t+1} \odot h\right), \qquad h_{\text{new}} = a^{L} \odot h + \sum_{j=0}^{L-1} a^{L-1-j} \odot b\, x_j
$$

Read it aloud: the first term is the chunk on its own, the second is what the past still contributes after $t + 1$ decays, and the updated state is the old state decayed by the whole chunk plus the chunk's inputs. Drop the carried state and every chunk starts as if no audio came before it.

> [!WARNING]
> Resetting the state at each chunk boundary looks harmless, and the first chunk is exactly right. Every later chunk is wrong near its start, by an amount that depends on how slowly the states decay.

Mamba (Gu and Dao, 2023) makes $a$, $b$, and $c$ depend on the input. The kernel then differs at every position, so the single-kernel convolution no longer applies, and training uses a parallel scan. The recurrence and its state carry stay the same. The machine learning course covers that mechanism.

> [!KEY]
> A state-space layer is one function with two evaluations. Run it as a convolution to train in parallel, and as a recurrence with a carried state to stream.

## Where this breaks

- **Too few frames.** Aggressive subsampling can leave fewer frames than labels. Check the token rate first.
- **Mixed frame rates.** Zipformer's stacks run at different rates, so a chunk size must say which stack it counts.
- **Small state.** An $N$-dimensional state is a compressed summary of the past, and attention can recall any single frame exactly. Many recipes keep a few attention layers beside state-space ones.
- **One direction.** A causal layer sees only the past. An offline encoder can run one layer in each direction, which gives up streaming.

> [!RECAP]
> - Encoder FLOPs are $L(24Td^2 + 4T^2d)$, so a higher subsampling factor cuts the cost by a factor between $s$ and $s^2$.
> - FastConformer subsamples by 8 with a dense first stage and separable later stages, which has 8.5 times fewer parameters than a dense stack at 256 channels.
> - Zipformer runs its middle stacks at lower frame rates, using Downsample, Upsample, and a bypass mix.
> - A diagonal state-space layer computes the same output by recurrence and by convolution, to floating-point precision.
> - Carrying the state between chunks makes a chunked pass equal the full pass.

## Build it step by step

1. **`encoder_flops(frames, dim, layers, subsample)`** returns an int. Use `T = frames // subsample` and the formula above. The usual mistake is to drop the $T^2$ term, which makes the saving exactly 2.
2. **`subsample_len(n, stages)`** and **`subsample_params(channels, stages, separable)`** return ints. The usual mistake is to forget that the first stage is always dense.
3. **`build_subsampler(channels, stages, separable)`** returns an `nn.Sequential`. Use `groups=channels` for the depthwise convolution. Its parameter count must equal `subsample_params`.
4. **`downsample(x, logits, factor)`** and **`upsample(x, factor, length)`** return `(ceil(T / factor), d)` and `(length, d)`. Use `np.repeat` to upsample. The usual mistake is to skip the padding when `T` is not a multiple of `factor`.
5. **`bypass(x, y, c)`** returns `(1 - c) * x + c * y`. The usual mistake is to swap the weights.
6. **`ssm_recurrent(a, b, c, x, h0=None)`** returns `(y, h)`. Update the state before you read the output.
7. **`ssm_kernel(a, b, c, length)`** and **`ssm_convolve(kernel, x)`** return `(length,)` and `(T,)`. Build the powers with `a[None, :] ** np.arange(length)[:, None]`. The usual mistake is an off-by-one exponent, which shifts the kernel by a step.
8. **`ssm_chunk(a, b, c, x, h)`** returns `(y, new_h)`. Add the carried-state term to the chunk's own convolution. The usual mistake is to start every chunk from zeros.

## The exercise

You write the eleven functions above. The checks compare FLOP and parameter counts with the hand values in this chapter, verify shapes and round trips for the Zipformer pieces, and test the state-space layer. The recurrent and convolutional forms must agree to $10^{-10}$, and chunked passes of uneven sizes, down to one frame at a time, must match the full pass and end in the same state.

## Further reading

- Yao et al., "Zipformer: A faster and better encoder for automatic speech recognition" (ICLR 2024), for the stacks, the modules, and the BiasNorm and ScaledAdam changes this chapter skips.
- Rekesh et al., "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition" (2023), and the NeMo ASR models page for its configurations.
- Peng et al., "Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding" (ICML 2022).
- Gu, Goel, and Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), and Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", for the theory. Start with the [machine learning course](https://ml.ifkash.dev/).
