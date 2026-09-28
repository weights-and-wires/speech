---
title: WaveNet and autoregressive vocoders
slug: 26-wavenet-and-autoregressive-vocoders
part: "Part 5 — Speech synthesis"
summary: Build a WaveNet from dilated causal convolutions and mu-law classes, and generate audio one sample at a time with a cache.
minutes: 70
gpu: false
objectives:
  - Build dilated causal convolutions, and compute and verify a stack's receptive field with an impulse test.
  - Encode audio with mu-law companding into 256 classes, and explain why it beats linear quantization for quiet sounds.
  - Implement the gated residual block with skip connections, and train it as a next-sample classifier.
  - Generate audio sample by sample with a per-layer cache that matches the parallel forward pass exactly.
prereqs:
  - 02-quantization-and-audio-files
  - 06-filters-convolution-and-reverb
  - 24-text-to-speech-and-tacotron
exercise: 26-wavenet-and-autoregressive-vocoders
---

# WaveNet and autoregressive vocoders

A vocoder turns a mel spectrogram into a waveform. The mel spectrogram keeps
the loudness of each band in each frame and throws away the rest: the phase,
the fine frequency detail, and every one of the 256 samples inside a frame.
Griffin-Lim fills them in by iteration, and it sounds metallic. A neural
vocoder learns how real speech fills them in.

WaveNet, published in 2016, was the first neural model to generate raw audio
that listeners could barely tell from recordings. Its idea is blunt: predict
each sample from all the samples before it, one at a time, 16,000 or 24,000
times per second of audio. Tacotron 2 used a WaveNet as its vocoder.

In this lesson you build a small WaveNet: the dilated causal convolutions
that let it see far back cheaply, the mu-law coding that turns each sample
into one of 256 classes, the gated residual stack, and a generator with a
cache that makes one-sample steps fast. Then you train it on a sung vowel and
listen to what it makes up.

## What you already know

- Quantization to $N$ bits maps each sample to one of $2^N$ levels, and mu-law companding spends more levels on quiet sounds (lesson 2, *Quantization and audio files*).
- A convolution slides a small kernel along a signal, and a causal filter uses only past inputs (lesson 6, *Filters, convolution, and reverb*).
- A classifier outputs logits, and cross-entropy scores the probability it gave the right class.
- An autoregressive model factors a sequence's probability into one prediction per step, each given the steps before (lesson 24, *Text to speech and Tacotron*).

## One sample at a time

WaveNet models a waveform $x_1, \ldots, x_T$ as a product of next-sample
predictions:

$$
p(x) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}).
$$

Read it aloud: the probability of the whole clip is the probability of the
first sample, times the probability of the second given the first, and so on.
A network outputs each factor. During training every factor is computed at
once, because the whole clip is known. That is *teacher forcing*, as in
Tacotron. During generation each sample must exist before the next can be
predicted, so the loop runs $T$ times.

A vocoder conditions every factor on the mel spectrogram as well:
$p(x_t \mid x_{<t}, h)$. This lesson's exercise leaves the conditioning out,
so you can hear what the model learns about the waveform alone. The last
section explains how the mel spectrogram enters.

## Mu-law classes

WaveNet treats each sample as a class, not a number. A softmax over classes
can express any distribution, including one with two peaks, where a squared
error on a number would pick the average of the peaks. But 16-bit audio has
65,536 levels, too many for one softmax. WaveNet uses 256 classes and spends
them where the ear needs them. Its mu-law companding stretches small
amplitudes before quantizing:

$$
f(x) = \operatorname{sign}(x)\,\frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \qquad
q = \left\lfloor \frac{f(x) + 1}{2}\,\mu + \frac12 \right\rfloor, \qquad \mu = 255.
$$

Read it aloud: $f$ keeps the sign and takes a logarithm of the magnitude,
scaled so that $\pm 1$ maps to $\pm 1$. Then the range $[-1, 1]$ is split
into 256 equal steps, and $q$ is the nearest step, an integer from 0 to 255.
Decoding inverts both steps, $y = 2q/\mu - 1$ and
$x = \operatorname{sign}(y)\,((1 + \mu)^{|y|} - 1)/\mu$. In numpy:

```python
f = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
q = np.floor((f + 1) / 2 * mu + 0.5).astype(np.int64)
```

Work three samples by hand, with $\ln 256 = 5.5452$:

- $x = 0.5$: $f = \ln(128.5)/5.5452 = 4.8559/5.5452 = 0.8757$, and $(0.8757 + 1)/2 \times 255 + 0.5 = 239.65$, so $q = 239$.
- $x = 0.001$: $f = \ln(1.255)/5.5452 = 0.0410$, and $q = \lfloor 133.22 \rfloor = 133$.
- $x = -0.01$: $f = -\ln(3.55)/5.5452 = -0.2285$, and $q = \lfloor 98.87 \rfloor = 98$.

Between $x = 0$ and $x = 0.01$ lie codes 128 to 157, 30 of them. A linear
8-bit quantizer, with steps of $2/255 = 0.0078$, gives that range little more
than one. On a tone of
amplitude 0.01, the exercise measures a signal-to-noise ratio of 33.7 dB with
mu-law against 9.3 dB with 8-bit linear steps.

> [!KEY]
> WaveNet is a classifier: at every sample it predicts a softmax over 256
> mu-law codes, trained with cross-entropy on the next sample.

## Dilated causal convolutions

To predict sample $t$ well, the network must see several pitch periods back,
hundreds of samples. A plain causal convolution with kernel size 2 sees one
more sample per layer, so it would need hundreds of layers.

A *dilated* convolution skips inputs. With dilation $d$ and kernel size $k$,
output $t$ combines inputs $t, t - d, \ldots, t - (k-1)d$. Doubling the
dilation at every layer, 1, 2, 4, 8, and so on, doubles how far back the
stack sees with each layer. The number of input samples one output depends
on is the *receptive field*:

$$
R = 1 + (k - 1) \sum_{\ell} d_\ell.
$$

Read it aloud: each layer with dilation $d_\ell$ reaches $(k-1)\,d_\ell$
samples further back, and the reaches add up. The 1 is the current sample
itself. WaveNet used $k = 2$ and dilations $1, 2, 4, \ldots, 512$, repeated
three times: $R = 1 + 3 \times 1023 = 3070$ samples, about 192 ms at 16 kHz.
One block of $1..512$ alone gives $R = 1024$ from only 10 layers.

> [!KEY]
> Doubling the dilation at every layer doubles the receptive field with each
> layer, so 10 layers see 1,024 samples where 10 undilated layers see 11.

*Causal* means the output at $t$ never sees $t + 1$. The simplest way to get
that is to pad $(k-1)d$ zeros on the left and none on the right:

```python
y = F.conv1d(F.pad(x, ((k - 1) * d, 0)), weight, bias, dilation=d)
```

For example, take $x = [1, 2, 3, 4]$, a kernel with weight 1 on the current
sample and 0.5 on the one two back, and $d = 2$. Then
$y_t = x_t + 0.5\,x_{t-2}$, with zeros before the start:
$y = [1,\ 2,\ 3 + 0.5,\ 4 + 1] = [1, 2, 3.5, 5]$.

The receptive field is also something you can measure. Change one input
sample at time $t_0$ and watch which outputs change. Exactly the outputs at
$t_0, t_0 + 1, \ldots, t_0 + R - 1$ must change: none earlier, because the
stack is causal, and none later, because they can't see that far back. The
following demo draws that cone of dependence for a stack you choose, with
its kernel size, layers per stack, stacks, and sample rate.

```demo
{"id": "wavenet-receptive-field", "caption": "Add layers per stack and then stacks, and compare the receptive field with the same layers without dilation: each layer whose dilation doubles also doubles how far back the output sees."}
```

> [!WARNING]
> `padding="same"` or symmetric padding lets output $t$ see the future. The
> training loss then drops fast, because the model copies the answer, and
> generation produces noise. Run the impulse test on every stack.

## The gated residual stack

Each WaveNet layer applies a dilated causal convolution with $2C$ output
channels and splits the result into a *filter* half and a *gate* half:

$$
z = \tanh(W_f * x) \odot \sigma(W_g * x).
$$

Read it aloud: the tanh half proposes a value between $-1$ and 1, and the
sigmoid half decides, per channel and time, how much of it passes. The
$\odot$ is elementwise multiplication. This is the LSTM's gate applied to a
convolution. Two $1 \times 1$ convolutions then read $z$. One adds a
*residual* back to the layer's input, so the next layer sees $x + R z$. The
other adds to a running *skip* sum. After the last layer, the skip sum passes
through ReLU, a $1 \times 1$ convolution, ReLU, and a final
$1 \times 1$ convolution to 256 logits per time step.

```python
filt, gate = conv_out.chunk(2, dim=1)
z = torch.tanh(filt) * torch.sigmoid(gate)
x, skips = x + residual(z), skips + skip(z)
```

The residual path keeps gradients flowing through dozens of layers. The skip
sum lets the output read every layer directly, short-range and long-range
alike.

## Generation with a cache

To generate, feed the model its own samples. Pick $x_1$, compute
$p(x_2 \mid x_1)$, sample $x_2$, compute $p(x_3 \mid x_1, x_2)$, and so on.
The obvious loop runs the whole network over the last $R$ samples at every
step. Almost all of that work repeats the step before.

A cache removes the repetition. With kernel size 2, layer $\ell$ at time $t$
needs exactly two inputs: its current input, and its input $d_\ell$ steps ago.
Keep, for each layer, a queue of its last $d_\ell$ inputs. One step then
costs one pass through the layers at a single time position:

1. Embed the newest code.
2. For each layer, combine the current input with the oldest entry in that layer's queue, apply the gate, and update the residual and skip.
3. Push the current input onto the queue and drop the oldest.
4. Run the output head, and sample the next code from the softmax.

The queue starts full of zeros, which is exactly what left padding gave the
parallel forward pass. So the cached logits must equal the forward pass's
logits to rounding error, and the exercise checks that they do. This trick
is known as *Fast WaveNet*.

> [!TIP]
> Test a cache against the parallel forward pass before you trust a single
> generated sample. A cache that is off by one step still produces audio, and
> it sounds like a model that trained badly.

## Conditioning and the vocoder family

A vocoder adds the mel spectrogram to every gate. The mel runs at a frame
rate, 86 frames per second at 22,050 Hz with a hop of 256, so it is first
upsampled to the sample rate by transposed convolutions or repetition, giving
$y$. Each layer then adds a $1 \times 1$ projection of $y$ inside both halves:
$z = \tanh(W_f * x + V_f * y) \odot \sigma(W_g * x + V_g * y)$. A speaker
embedding enters the same way, as one vector added at every time step.

The cost of one network pass per sample drove a family of faster
autoregressive vocoders:

- **WaveRNN** replaced the convolution stack with a single GRU of 896 units, and split each 16-bit sample into 8 coarse and 8 fine bits, each predicted by its own 256-way softmax. Pruned to over 96% sparsity, it ran on mobile CPUs.
- **LPCNet** put signal processing back in. A 16th-order linear predictor from lesson 9 handles the spectral envelope, and a small WaveRNN models only the residual, in 8-bit mu-law. It runs in under 3 GFLOPS.
- **Parallel WaveNet** and **ClariNet** distilled a WaveNet teacher into a flow that generates all samples at once. Lesson 28 covers flows, and lesson 27 covers the GAN vocoders that replaced them all.

> [!RECAP]
> - The receptive field of a dilated stack is $1 + (k-1)\sum_\ell d_\ell$; an impulse test must change exactly that many outputs.
> - Mu-law maps $x$ to $\lfloor (f(x) + 1)/2 \cdot 255 + 0.5 \rfloor$, giving quiet samples many more of the 256 codes.
> - A causal convolution pads $(k-1)d$ zeros on the left only.
> - Each layer gates with $\tanh \odot \sigma$ and feeds a residual and a skip sum.
> - Cached generation keeps each layer's last $d$ inputs and must match the forward pass exactly.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`receptive_field(dilations, kernel_size)`** returns an `int`. The
   common mistake is forgetting the 1 for the current sample.
2. **`mu_law_encode(x, mu)`** clips to $[-1, 1]$, compands with `np.log1p`,
   and quantizes with `np.floor(... + 0.5)`. Return `int64`. The common
   mistake is `np.round`, which rounds halves to even and moves some codes.
3. **`mu_law_decode(q, mu)`** maps codes to $[-1, 1]$ and inverts the
   logarithm. The common mistake is inverting with `np.exp` instead of
   raising $1 + \mu$ to $|y|$.
4. **`causal_conv1d(x, weight, bias, dilation)`** pads on the left with
   `F.pad` and calls `F.conv1d`. The common mistake is padding
   `kernel_size - 1` without multiplying by the dilation.
5. **`gated_activation(x)`** splits the channels with `chunk(2, dim=1)`.
6. **`WaveNet.forward(codes)`** embeds, transposes to `(B, C, T)`, runs each
   layer, and applies the head to the skip sum. Its column $t$ predicts
   sample $t + 1$. The common mistake is feeding the skip sum or $x$ from the
   wrong layer to the head.
7. **`WaveNet.step(code, cache)`** computes one time position. Tap 1 of each
   dilated kernel multiplies the current input and tap 0 the oldest cached
   one. A $1 \times 1$ convolution at one position is a matrix product with
   `weight[:, :, 0]`. The common mistake is appending to the cache before
   reading its oldest entry.
8. **`generate(model, n_samples, temperature, seed, first)`** loops over
   `model.step`, samples with `torch.multinomial` and one seeded
   `torch.Generator`, and uses the argmax when the temperature is 0.

## Cost and scaling

The following table compares the three ways to run a WaveNet with $L$ layers
of $C$ channels, receptive field $R$, and $T$ samples.

| Mode | Work | Sequential steps | Memory |
|---|---|---|---|
| Training (teacher forcing) | $O(T L C^2)$ | 1, parallel over $T$ | $O(T L C)$ activations |
| Naive generation | $O(T R L C^2)$ | $T$ | $O(R L C)$ |
| Cached generation | $O(T L C^2)$ | $T$ | $O(C \sum_\ell d_\ell)$ for the queues |

The cache makes generation as cheap in arithmetic as training, but it can't
remove the $T$ sequential steps. One second of 24 kHz audio needs 24,000 of
them, each a pass through every layer. In the HiFi-GAN paper's comparison, a
WaveNet vocoder with a mixture-of-logistics output ran at 0.07 kHz on a V100
GPU, about 0.003 times real time. The GPU spends most of each step waiting,
because one time position gives it too little work. That sequential loop is
what the rest of this part removes.

## Where this breaks

**The future leaks in.** Symmetric padding, or a target that isn't shifted
by one sample, lets the model see its answer. Training loss collapses and
generation fails. The impulse test catches the first, and checking that
column $t$ is scored against sample $t + 1$ catches the second.

**Errors compound.** In generation the model reads its own samples. A small
early error, such as a glitch or drift in pitch, is then input that the model
never saw in training. Lowering the sampling temperature trades variety for
stability.

**Silence and noise.** An unconditional WaveNet trained on speech babbles:
it produces speech-like sounds with no words. Conditioning on a mel
spectrogram is what makes it a vocoder.

**It is slow.** Real-time WaveNet needs custom GPU kernels or a much smaller
recurrent model such as WaveRNN. For most uses a GAN vocoder is simpler.

## The exercise

You implement `receptive_field`, `mu_law_encode`, `mu_law_decode`,
`causal_conv1d`, `gated_activation`, `WaveNet.forward`, `WaveNet.step`, and
`generate`. The checks confirm the 3,070-sample receptive field and the
hand-worked mu-law codes, and that mu-law beats 8-bit linear quantization by
at least 15 dB on a quiet tone. They compare your causal convolution with a
loop, your forward pass with the architecture written out from its weights,
and run the impulse test. Two last checks require your cached steps to match
the forward pass, and greedy generation to match re-running the full model on
every prefix.

Press **Run** to train a 12-layer WaveNet with a 127-sample receptive field
on three seconds of a sung vowel at 8 kHz, for about 15 seconds. The run then
generates half a second one sample at a time, prints how many samples per
second your cache produced, plays the training audio and the generated audio,
and draws the generated audio's spectrogram.
