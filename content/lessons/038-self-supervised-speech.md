---
title: Self-supervised speech
slug: 038-self-supervised-speech
part: "Part 4 — Neural speech recognition"
summary: Learn speech from audio alone with wav2vec 2.0, HuBERT, BEST-RQ, and WavLM, compare what they predict, and score them the SUPERB way.
minutes: 80
gpu: true
objectives:
  - Compute the frame count, hop, and receptive field of the wav2vec 2.0 feature encoder, and the fraction of frames that span masking hides.
  - Implement the contrastive loss with distractors from masked frames, and the diversity loss that keeps a codebook in use.
  - Implement HuBERT's masked prediction of k-means targets and BEST-RQ's frozen random-projection targets.
  - Build WavLM's two changes, the denoising mix and the gated relative position bias, and compare the targets of wav2vec 2.0, HuBERT, WavLM, data2vec, and w2v-BERT.
  - Mix layers with SUPERB's learnable weighted sum, and read a SUPERB table across tasks.
  - Fit a closed-form linear probe to every layer of wav2vec2-base and hubert-base, and read where speaker and content information lives.
prereqs:
  - 028-neural-acoustic-models
  - 030-ctc-forward-backward
  - 036-conformers-and-specaugment
exercise: 038-self-supervised-speech
---

# Self-supervised speech

A transcribed hour of speech costs hours of careful listening and typing. An untranscribed hour costs almost nothing, and broadcasts supply millions of them. How can a model learn from the cheap kind?

Self-supervised learning trains an encoder on audio alone: hide some frames, and ask the model to recognize them from the context. To do that, it must learn what speech sounds are and how they follow each other. [Fine-tuning speech models](/lessons/044-fine-tuning-speech-models) turns such an encoder into a recognizer. Here you build the pretraining designs and see how a benchmark ranks them.

## What you already know

- A strided convolution shortens a sequence, from [Neural acoustic models](/lessons/028-neural-acoustic-models).
- CTC trains an encoder against a transcript, from [Computing CTC efficiently](/lessons/030-ctc-forward-backward).
- SpecAugment hides spans of the input, from [Conformers and SpecAugment](/lessons/036-conformers-and-specaugment).

## From raw samples to frames

wav2vec 2.0 reads the waveform at 16,000 samples per second. A **feature encoder** of seven unpadded 1-D convolutions, with kernel widths (10, 3, 3, 3, 3, 2, 2) and strides (5, 2, 2, 2, 2, 2, 2), turns the samples into 512-dimensional vectors, one per **frame**. Each layer maps a length $L$ to

$$
L' = \left\lfloor \frac{L - k}{s} \right\rfloor + 1
$$

In numpy, `length = (length - k) // s + 1`. From 16,000 samples the layers give 3199, 1599, 799, 399, 199, 99, and 49. One second becomes 49 frames, not 50.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Length of one second of audio after each of the seven convolution layers: 16000, 3199, 1599, 799, 399, 199, 99, then 49 frames">
<text x="20" y="22" class="fig-bold">length after each layer (bar length is the log of the count)</text>
<text x="20" y="56" class="fig-small">input samples</text>
<rect x="170" y="40" width="330.0" height="22" class="fill-a"/>
<text x="508.0" y="56" class="fig-small fig-mono">16,000</text>
<text x="20" y="90" class="fig-small">layer 1: k=10, s=5</text>
<rect x="170" y="74" width="275.1" height="22" class="fill-soft-a"/>
<text x="453.1" y="90" class="fig-small fig-mono">3,199</text>
<text x="20" y="124" class="fig-small">layer 2: k=3, s=2</text>
<rect x="170" y="108" width="251.5" height="22" class="fill-soft-a"/>
<text x="429.5" y="124" class="fig-small fig-mono">1,599</text>
<text x="20" y="158" class="fig-small">layer 3: k=3, s=2</text>
<rect x="170" y="142" width="227.8" height="22" class="fill-soft-a"/>
<text x="405.8" y="158" class="fig-small fig-mono">799</text>
<text x="20" y="192" class="fig-small">layer 4: k=3, s=2</text>
<rect x="170" y="176" width="204.2" height="22" class="fill-soft-a"/>
<text x="382.2" y="192" class="fig-small fig-mono">399</text>
<text x="20" y="226" class="fig-small">layer 5: k=3, s=2</text>
<rect x="170" y="210" width="180.4" height="22" class="fill-soft-a"/>
<text x="358.4" y="226" class="fig-small fig-mono">199</text>
<text x="20" y="260" class="fig-small">layer 6: k=2, s=2</text>
<rect x="170" y="244" width="156.6" height="22" class="fill-soft-a"/>
<text x="334.6" y="260" class="fig-small fig-mono">99</text>
<text x="20" y="294" class="fig-small">layer 7: k=2, s=2</text>
<rect x="170" y="278" width="132.7" height="22" class="fill-b"/>
<text x="310.7" y="294" class="fig-small fig-mono">49</text>
</svg>
Figure 1. One second at 16 kHz shrinks layer by layer: each layer maps L to floor((L - k) / s) + 1. Notice that the result is 49 frames, not 50, because every unpadded layer loses a little at the edge.
```

The **hop** is the product of the strides, 320 samples or 20 ms. The **receptive field**, the input samples one frame depends on, adds each layer's extra kernel taps, counted in the step size of its own input:

$$
R = 1 + \sum_{i=1}^{7} (k_i - 1) \prod_{j<i} s_j = 400
$$

That is 25 ms, like a classical spectrogram frame. A Transformer, the **context network**, turns the frames into outputs $c_t$ that see the whole utterance.

## Hide spans

One hidden frame is easy to fill in from its neighbors, so wav2vec 2.0 hides **spans**. Each frame starts a span with probability $p = 0.065$, and it and the $M - 1 = 9$ frames after it become one learned vector. A frame stays visible only if none of the $M$ frames at or before it started a span, so the masked fraction is

$$
P(\text{masked}) = 1 - (1 - p)^M
$$

With $p = 0.065$ and $M = 10$, $1 - 0.935^{10} = 0.489$: about 49% of frames are hidden. In numpy, loop over `np.flatnonzero(starts)` and set `mask[s : s + span] = True`.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Fraction of frames masked against the span start probability p for spans of 10 frames, passing 49 percent at p = 0.065">
<line x1="70.0" y1="30" x2="70.0" y2="220" class="stroke-grid"/>
<text x="70.0" y="236" text-anchor="middle" class="fig-small">0</text>
<line x1="145.0" y1="30" x2="145.0" y2="220" class="stroke-grid"/>
<text x="145.0" y="236" text-anchor="middle" class="fig-small">0.02</text>
<line x1="220.0" y1="30" x2="220.0" y2="220" class="stroke-grid"/>
<text x="220.0" y="236" text-anchor="middle" class="fig-small">0.04</text>
<line x1="295.0" y1="30" x2="295.0" y2="220" class="stroke-grid"/>
<text x="295.0" y="236" text-anchor="middle" class="fig-small">0.06</text>
<line x1="370.0" y1="30" x2="370.0" y2="220" class="stroke-grid"/>
<text x="370.0" y="236" text-anchor="middle" class="fig-small">0.08</text>
<line x1="445.0" y1="30" x2="445.0" y2="220" class="stroke-grid"/>
<text x="445.0" y="236" text-anchor="middle" class="fig-small">0.1</text>
<line x1="520.0" y1="30" x2="520.0" y2="220" class="stroke-grid"/>
<text x="520.0" y="236" text-anchor="middle" class="fig-small">0.12</text>
<line x1="70" y1="220.0" x2="520" y2="220.0" class="stroke-grid"/>
<text x="62" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="172.5" x2="520" y2="172.5" class="stroke-grid"/>
<text x="62" y="176.5" text-anchor="end" class="fig-small">0.2</text>
<line x1="70" y1="125.0" x2="520" y2="125.0" class="stroke-grid"/>
<text x="62" y="129.0" text-anchor="end" class="fig-small">0.4</text>
<line x1="70" y1="77.5" x2="520" y2="77.5" class="stroke-grid"/>
<text x="62" y="81.5" text-anchor="end" class="fig-small">0.6</text>
<line x1="70" y1="30.0" x2="520" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">0.8</text>
<polyline points="70.0,220.0 77.5,215.3 85.0,210.7 92.5,206.1 100.0,201.7 107.5,197.3 115.0,193.0 122.5,188.8 130.0,184.6 137.5,180.6 145.0,176.6 152.5,172.6 160.0,168.8 167.5,165.0 175.0,161.3 182.5,157.6 190.0,154.1 197.5,150.5 205.0,147.1 212.5,143.7 220.0,140.4 227.5,137.1 235.0,133.9 242.5,130.8 250.0,127.7 257.5,124.7 265.0,121.7 272.5,118.8 280.0,116.0 287.5,113.2 295.0,110.4 302.5,107.7 310.0,105.1 317.5,102.5 325.0,99.9 332.5,97.4 340.0,95.0 347.5,92.6 355.0,90.2 362.5,87.9 370.0,85.7 377.5,83.4 385.0,81.3 392.5,79.1 400.0,77.0 407.5,75.0 415.0,73.0 422.5,71.0 430.0,69.1 437.5,67.2 445.0,65.3 452.5,63.5 460.0,61.7 467.5,60.0 475.0,58.2 482.5,56.6 490.0,54.9 497.5,53.3 505.0,51.7 512.5,50.2 520.0,48.6" class="stroke-b" fill="none"/>
<circle cx="313.8" cy="103.8" r="5" class="fill-a"/>
<text x="301.8" y="95.8" text-anchor="end" class="fig-small">wav2vec 2.0: p = 0.065: 48.9%</text>
<circle cx="370.0" cy="85.7" r="5" class="fill-a"/>
<text x="382.0" y="107.7" class="fig-small">HuBERT: p = 0.08: 56.6%</text>
<text x="295.0" y="258" text-anchor="middle">probability p that a frame starts a span</text>
<text x="18" y="125.0" transform="rotate(-90 18 125.0)" text-anchor="middle">fraction of frames masked</text>
</svg>
Figure 2. The masked fraction 1 - (1 - p)^10 for spans of M = 10 frames. Notice that a start probability of only 6.5% hides 48.9% of the frames, because each start covers ten of them and the spans overlap.
```

## The contrastive loss

The wav2vec 2.0 task is multiple choice. For a masked frame $t$, the output $c_t$ must pick the true target $q_t$ out of a line-up that also holds $K$ **distractors**, targets from other masked frames of the same utterance. The targets come from quantizing the *unmasked* features, so the answer is not in the model's input:

$$
\mathcal{L}_m = -\log \frac{\exp\left(\operatorname{sim}(c_t, q_t)/\kappa\right)}{\sum_{\tilde q \in Q_t} \exp\left(\operatorname{sim}(c_t, \tilde q)/\kappa\right)}
$$

Here $\operatorname{sim}$ is cosine similarity, $\kappa$ is a **temperature**, and $Q_t$ holds $q_t$ and the $K$ distractors, so the loss is cross-entropy over the line-up. wav2vec 2.0 uses $\kappa = 0.1$ and $K = 100$, drawn uniformly with replacement from the other $M - 1$ masked frames. In numpy, put the positive in column 0 and average `-log_softmax(logits, axis=1)[:, 0]` over unit vectors.

For $c_t = (1, 0)$, the target $(0.8, 0.6)$, and distractors $(0, 1)$ and $(-0.6, 0.8)$, the cosines are 0.8, 0, and $-0.6$. At $\kappa = 0.1$ the loss is $\log(1 + e^{-8} + e^{-14}) = 0.000336$ nats, and at $\kappa = 1$ it is 0.528.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Probability the contrastive softmax gives the true target and two distractors at temperature 0.1 and at temperature 1">
<line x1="50" y1="220" x2="290" y2="220" class="stroke-muted"/>
<text x="60" y="280" class="fig-bold">kappa = 0.1: loss 0.000336</text>
<rect x="60" y="50.1" width="56" height="169.9" class="fill-b"/>
<text x="88" y="44.1" text-anchor="middle" class="fig-small fig-bold">0.9997</text>
<text x="88" y="236" text-anchor="middle" class="fig-small">true</text>
<rect x="138" y="218.5" width="56" height="1.5" class="fill-soft-a"/>
<text x="166" y="212.5" text-anchor="middle" class="fig-small fig-bold">0.0003</text>
<text x="166" y="236" text-anchor="middle" class="fig-small">d1</text>
<rect x="216" y="218.5" width="56" height="1.5" class="fill-soft-a"/>
<text x="244" y="212.5" text-anchor="middle" class="fig-small fig-bold">0.0000</text>
<text x="244" y="236" text-anchor="middle" class="fig-small">d2</text>
<line x1="330" y1="220" x2="570" y2="220" class="stroke-muted"/>
<text x="340" y="280" class="fig-bold">kappa = 1: loss 0.528</text>
<rect x="340" y="119.8" width="56" height="100.2" class="fill-b"/>
<text x="368" y="113.8" text-anchor="middle" class="fig-small fig-bold">0.5896</text>
<text x="368" y="236" text-anchor="middle" class="fig-small">true</text>
<rect x="418" y="175.0" width="56" height="45.0" class="fill-soft-a"/>
<text x="446" y="169.0" text-anchor="middle" class="fig-small fig-bold">0.2649</text>
<text x="446" y="236" text-anchor="middle" class="fig-small">d1</text>
<rect x="496" y="195.3" width="56" height="24.7" class="fill-soft-a"/>
<text x="524" y="189.3" text-anchor="middle" class="fig-small fig-bold">0.1454</text>
<text x="524" y="236" text-anchor="middle" class="fig-small">d2</text>
<text x="20" y="22" class="fig-bold">softmax probability of each candidate</text>
</svg>
Figure 3. The same three cosines, 0.8, 0, and -0.6, scored at two temperatures. At kappa = 0.1 the true target takes 99.97% of the probability and the loss is 0.000336; at kappa = 1 it takes 59% and the loss is 0.528. Notice that a low temperature makes small angle differences count a lot.
```

## Keep the codebook in use

wav2vec 2.0 makes $q_t$ with a **product quantizer**: $G = 2$ codebooks of $V = 320$ entries. A Gumbel softmax picks one entry per codebook, and the two picks are concatenated and projected. Left alone, a quantizer collapses onto a few entries. The **diversity loss** pushes each codebook's average use toward uniform:

$$
\mathcal{L}_d = \frac{1}{GV} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar p_{g,v} \log \bar p_{g,v}
$$

Here $\bar p_{g,v}$ is the softmax probability of entry $v$ in codebook $g$, without Gumbel noise, averaged over the batch. The sum is minus each codebook's entropy, so minimizing it spreads use out. Equal use scores $-\log V / V = -0.0180$ for $V = 320$, and a collapsed codebook scores 0. The objective is $\mathcal{L}_m + 0.1\,\mathcal{L}_d$. For two entries, frames (0.9, 0.1) and (0.1, 0.9) give $-0.347$, and two frames of (0.9, 0.1) give $-0.163$.

> [!WARNING]
> Average the probabilities over frames first, then take $p \log p$. Averaging each frame's entropy instead rewards every frame for being unsure.

## Predict labels instead

**HuBERT** replaces the learned quantizer with labels made before training. It runs k-means on features of the whole training set and gives every frame the index of its nearest cluster. The model predicts the label at masked frames:

$$
p(c \mid t) = \frac{\exp\left(\operatorname{sim}(A o_t, e_c)/\tau\right)}{\sum_{c'} \exp\left(\operatorname{sim}(A o_t, e_{c'})/\tau\right)}
$$

Project the Transformer output $o_t$ with $A$, take its cosine with each cluster's learned embedding $e_c$, divide by $\tau = 0.1$, and normalize. The loss is cross-entropy over masked frames only, because at an unmasked frame the answer is in the input. **k-means** alternates two steps until nothing changes: label each point with its nearest center, then move each center to the mean of its points.

**BEST-RQ** asks whether targets need training at all. It stacks 4 frames of normalized log-mel features into $x$, projects with a frozen random matrix $A$, and picks the nearest entry of a frozen random codebook, which for unit vectors is the largest cosine. If $Ax = (3, 4)$ and the codebook holds $(1, 0)$, $(0, 2)$, and $(-1, 1)$, the cosines are 0.6, 0.8, and 0.141, so the label is 1.

## Denoise while you predict

HuBERT trains on clean audiobooks, so it seldom hears two voices at once. **WavLM** (Chen et al., 2022) keeps masked label prediction and changes two things.

First, the input. For some utterances, WavLM adds a second utterance or a noise clip into one span of the **primary** utterance, and the targets stay the cluster labels of the clean primary. The model must find the main voice and ignore the intruder. The secondary is scaled so that the primary span's energy $E_\text{pri}$, a sum of squares, sits $r$ dB above the scaled secondary's:

$$
s = \sqrt{\frac{E_\text{pri}}{10^{r/10}\, E_\text{sec}}}
$$

An energy ratio of $r$ dB is a factor of $10^{r/10}$, and amplitude is the square root of energy. In numpy, `mixed[span] += s * secondary[span]`. The paper measures energy over whole utterances. The exercise measures it inside the span, so the ratio in the overlap is exact.

Take six ones, a secondary $(0, 0, 3, -3, 0, 0)$, a span of 2 samples at sample 2, and $r = 10 \log_{10} 4 = 6.02$ dB. The span energies are 2 and 18, so $s = \sqrt{2 / (4 \cdot 18)} = 1/6$ and the mix is $(1, 1, 1.5, 0.5, 1, 1)$.

Second, the position signal. Attention scores gain a bias, $q_i \cdot k_j / \sqrt{d_k} + r_{i-j}$, that depends on the offset $i - j$ between query frame $i$ and key frame $j$. WavLM sorts offsets into $n = 320$ **buckets**, each with one learned number $d$. The first $n/2$ buckets hold $i - j \le 0$ and the rest hold $i - j > 0$, so the sign survives. On each side, offsets below $n/4 = 80$ get their own bucket, offsets up to $m = 800$ share buckets spaced evenly in $\log |i - j|$, and farther ones share the last.

With $n = 8$ and $m = 16$, each side has 4 buckets and 2 are exact. The offsets $-20, -1, 0, +1, +4, +20$ land in buckets $3, 1, 0, 5, 6, 7$.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Relative position bucket against the signed offset i minus j for 320 buckets and a maximum distance of 800">
<line x1="95.0" y1="30" x2="95.0" y2="220" class="stroke-grid"/>
<text x="95.0" y="236" text-anchor="middle" class="fig-small">-800</text>
<line x1="195.0" y1="30" x2="195.0" y2="220" class="stroke-grid"/>
<text x="195.0" y="236" text-anchor="middle" class="fig-small">-400</text>
<line x1="295.0" y1="30" x2="295.0" y2="220" class="stroke-grid"/>
<text x="295.0" y="236" text-anchor="middle" class="fig-small">0</text>
<line x1="395.0" y1="30" x2="395.0" y2="220" class="stroke-grid"/>
<text x="395.0" y="236" text-anchor="middle" class="fig-small">400</text>
<line x1="495.0" y1="30" x2="495.0" y2="220" class="stroke-grid"/>
<text x="495.0" y="236" text-anchor="middle" class="fig-small">800</text>
<line x1="70" y1="220.0" x2="520" y2="220.0" class="stroke-grid"/>
<text x="62" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="172.4" x2="520" y2="172.4" class="stroke-grid"/>
<text x="62" y="176.4" text-anchor="end" class="fig-small">80</text>
<line x1="70" y1="124.7" x2="520" y2="124.7" class="stroke-grid"/>
<text x="62" y="128.7" text-anchor="end" class="fig-small">160</text>
<line x1="70" y1="77.1" x2="520" y2="77.1" class="stroke-grid"/>
<text x="62" y="81.1" text-anchor="end" class="fig-small">240</text>
<line x1="70" y1="30.0" x2="520" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">319</text>
<polyline points="70.0,125.3 71.5,125.3 73.0,125.3 74.5,125.3 76.0,125.3 77.5,125.3 79.0,125.3 80.5,125.3 82.0,125.3 83.5,125.3 85.0,125.3 86.5,125.3 88.0,125.3 89.5,125.3 91.0,125.3 92.5,125.3 94.0,125.3 95.5,125.3 97.0,125.3 98.5,125.3 100.0,125.3 101.5,125.9 103.0,125.9 104.5,125.9 106.0,125.9 107.5,126.5 109.0,126.5 110.5,126.5 112.0,127.1 113.5,127.1 115.0,127.1 116.5,127.1 118.0,127.7 119.5,127.7 121.0,127.7 122.5,128.3 124.0,128.3 125.5,128.3 127.0,128.9 128.5,128.9 130.0,128.9 131.5,129.5 133.0,129.5 134.5,129.5 136.0,129.5 137.5,130.1 139.0,130.1 140.5,130.1 142.0,130.7 143.5,130.7 145.0,130.7 146.5,131.3 148.0,131.3 149.5,131.8 151.0,131.8 152.5,131.8 154.0,132.4 155.5,132.4 157.0,132.4 158.5,133.0 160.0,133.0 161.5,133.6 163.0,133.6 164.5,133.6 166.0,134.2 167.5,134.2 169.0,134.8 170.5,134.8 172.0,134.8 173.5,135.4 175.0,135.4 176.5,136.0 178.0,136.0 179.5,136.6 181.0,136.6 182.5,136.6 184.0,137.2 185.5,137.2 187.0,137.8 188.5,137.8 190.0,138.4 191.5,138.4 193.0,139.0 194.5,139.0 196.0,139.6 197.5,139.6 199.0,140.2 200.5,140.8 202.0,140.8 203.5,141.4 205.0,141.4 206.5,142.0 208.0,142.0 209.5,142.6 211.0,143.2 212.5,143.2 214.0,143.8 215.5,144.4 217.0,144.4 218.5,145.0 220.0,145.5 221.5,145.5 223.0,146.1 224.5,146.7 226.0,146.7 227.5,147.3 229.0,147.9 230.5,148.5 232.0,149.1 233.5,149.1 235.0,149.7 236.5,150.3 238.0,150.9 239.5,151.5 241.0,152.1 242.5,152.7 244.0,153.3 245.5,153.9 247.0,154.5 248.5,155.1 250.0,155.7 251.5,156.9 253.0,157.5 254.5,158.1 256.0,158.7 257.5,159.8 259.0,160.4 260.5,161.6 262.0,162.2 263.5,163.4 265.0,164.0 266.5,165.2 268.0,166.4 269.5,167.6 271.0,168.8 272.5,170.0 274.0,171.8 275.5,173.5 277.0,177.1 278.5,180.7 280.0,184.3 281.5,187.8 283.0,191.4 284.5,195.0 286.0,198.6 287.5,202.1 289.0,205.7 290.5,209.3 292.0,212.9 293.5,216.4 295.0,220.0" class="stroke-b" fill="none"/>
<polyline points="295.2,124.1 296.8,120.5 298.2,117.0 299.8,113.4 301.2,109.8 302.8,106.2 304.2,102.7 305.8,99.1 307.2,95.5 308.8,91.9 310.2,88.4 311.8,84.8 313.2,81.2 314.8,77.6 316.2,75.9 317.8,74.7 319.2,73.5 320.8,72.3 322.2,71.1 323.8,69.9 325.2,68.7 326.8,67.5 328.2,66.9 329.8,65.7 331.2,65.1 332.8,63.9 334.2,63.4 335.8,62.8 337.2,62.2 338.8,61.0 340.2,60.4 341.8,59.8 343.2,59.2 344.8,58.6 346.2,58.0 347.8,57.4 349.2,56.8 350.8,56.2 352.2,55.6 353.8,55.0 355.2,54.4 356.8,53.8 358.2,53.2 359.8,53.2 361.2,52.6 362.8,52.0 364.2,51.4 365.8,51.4 367.2,50.8 368.8,50.3 370.2,49.7 371.8,49.7 373.2,49.1 374.8,48.5 376.2,48.5 377.8,47.9 379.2,47.9 380.8,47.3 382.2,46.7 383.8,46.7 385.2,46.1 386.8,46.1 388.2,45.5 389.8,44.9 391.2,44.9 392.8,44.3 394.2,44.3 395.8,43.7 397.2,43.7 398.8,43.1 400.2,43.1 401.8,42.5 403.2,42.5 404.8,41.9 406.2,41.9 407.8,41.3 409.2,41.3 410.8,41.3 412.2,40.7 413.8,40.7 415.2,40.1 416.8,40.1 418.2,39.5 419.8,39.5 421.2,38.9 422.8,38.9 424.2,38.9 425.8,38.3 427.2,38.3 428.8,37.7 430.2,37.7 431.8,37.7 433.2,37.1 434.8,37.1 436.2,37.1 437.8,36.6 439.2,36.6 440.8,36.0 442.2,36.0 443.8,36.0 445.2,35.4 446.8,35.4 448.2,35.4 449.8,34.8 451.2,34.8 452.8,34.8 454.2,34.2 455.8,34.2 457.2,34.2 458.8,33.6 460.2,33.6 461.8,33.6 463.2,33.6 464.8,33.0 466.2,33.0 467.8,33.0 469.2,32.4 470.8,32.4 472.2,32.4 473.8,31.8 475.2,31.8 476.8,31.8 478.2,31.8 479.8,31.2 481.2,31.2 482.8,31.2 484.2,30.6 485.8,30.6 487.2,30.6 488.8,30.6 490.2,30.0 491.8,30.0 493.2,30.0 494.8,30.0 496.2,30.0 497.8,30.0 499.2,30.0 500.8,30.0 502.2,30.0 503.8,30.0 505.2,30.0 506.8,30.0 508.2,30.0 509.8,30.0 511.2,30.0 512.8,30.0 514.2,30.0 515.8,30.0 517.2,30.0 518.8,30.0" class="stroke-b" fill="none"/>
<circle cx="294.2" cy="218.2" r="5" class="fill-a"/>
<text x="284.2" y="208.2" text-anchor="end" class="fig-small">offset -3: bucket 3</text>
<circle cx="295.8" cy="122.9" r="5" class="fill-a"/>
<text x="305.8" y="126.9" text-anchor="start" class="fig-small">offset +3: bucket 163</text>
<circle cx="495.0" cy="30.0" r="5" class="fill-a"/>
<text x="487.0" y="46.0" text-anchor="end" class="fig-small">offset +800: bucket 319</text>
<text x="295" y="258" text-anchor="middle">offset i - j (frames)</text>
<text x="18" y="125" transform="rotate(-90 18 125)" text-anchor="middle">bucket index</text>
</svg>
Figure 4. The bucket for each offset i - j, with 320 buckets and a maximum distance of 800. Notice that the two signs use separate halves, offsets near zero get one bucket each, and the buckets widen as the distance grows.
```

A **gate** then scales the bias by what the query holds. Query $i$ computes $g_u = \sigma(q_i \cdot u)$ and $g_r = \sigma(q_i \cdot v)$, with learned vectors $u$ and $v$, a learned scalar $w$, and sigmoid $\sigma$:

$$
r_{i-j} = d_{i-j}\,\bigl(1 + g_u + (1 - g_u)\, w\, g_r\bigr)
$$

In numpy, `bias * scale[:, None]`. With $d = 2$ and $w = 1$, a query with both gates at 0.5 has scale 1.75 and $r = 3.5$, and a query with gates 0.8808 and 0.1192 has scale 1.8950 and $r = 3.79$, for the same offset.

## Compare what the models predict

The table lists five pretraining designs by what the model produces at a masked frame: a learned code, an offline code, or a continuous vector.

| Model | Target at a masked frame | Loss |
|---|---|---|
| wav2vec 2.0 | Quantized vector $q_t$, from a learned quantizer | Contrastive plus diversity |
| HuBERT | k-means cluster id, offline | Cross-entropy |
| WavLM | Cluster id of the clean primary | Cross-entropy, noisy input |
| data2vec | Mean of the top $K = 8$ block outputs of a moving-average teacher | L2 regression |
| w2v-BERT | The quantizer's token id | Contrastive plus cross-entropy |

## What the layers learn

A BASE encoder holds 13 representations: the feature encoder's output, called layer 0 here, and one after each Transformer block. A **linear probe** asks how much of a property a layer holds in a form a straight line can read. Average each clip's frames per layer, and fit a ridge regression onto one-hot labels $Y$, with a column of ones appended to the features $X$:

$$
W = \left(X^\top X + \lambda P\right)^{-1} X^\top Y
$$

Here $\lambda$ sets how hard to shrink the weights, and $P$ is the identity with its last diagonal entry set to 0, so the bias is not shrunk. In numpy, `np.linalg.solve(X.T @ X + l2 * P, X.T @ Y)`.

The **Run** block probes six speakers saying 0 to 9. The **content** probe trains on five speakers and tests on the sixth, and the **speaker** probe trains on digits 0 to 4 and tests on 5 to 9. On `wav2vec2-base`, digits peak at layer 4 (95%) and fall to 78% at layer 12, while speakers peak at layer 1 (96%) and fall to 73%.

> [!KEY]
> Self-supervised encoders keep who is speaking in their lower layers and what is said in their middle and upper layers. Pick the layer for the task.

Run k-means on one layer's frames and each 20 ms frame becomes a **unit**. A histogram of 100 layer-6 units names the digit of an unseen speaker 87% of the time, but the speaker of an unseen digit only 40%: units keep phones and drop most of the voice.

## Score a model the SUPERB way

Ranking models needs one protocol for all of them. **SUPERB** (Yang et al., Interspeech 2021) freezes the upstream model and gives each task a **learnable weighted sum of all layers**, because the last layer is not always the best. Only a light downstream model trains, such as a linear layer with CTC for phoneme recognition or a 2-layer BLSTM with CTC for ASR. With layer scores $a_l$, the mix is

$$
h_t = \sum_{l=0}^{L} \operatorname{softmax}(a)_l \, h_t^{(l)}
$$

The softmax makes the weights positive and sum to 1, and they train with the downstream model. In numpy, `np.tensordot(softmax(a), layers, axes=1)`. For layer values $(4, 8, 12)$ and scores $(0, 0, \ln 2)$, the weights are $(1/4, 1/4, 1/2)$ and the mix is $1 + 2 + 6 = 9$. Raw scores as weights give 8.32.

The table copies rows from Table I of Chen et al., "WavLM," 2022. The first three rows equal Table 2 of Yang et al., 2021. PR is phoneme error rate, ASR is word error rate without a language model, and SD is diarization error rate, in percent, lower is better. SID is speaker accuracy and ER is emotion accuracy, in percent, higher is better.

| Upstream | PR | ASR | SID | ER | SD |
|---|---|---|---|---|---|
| FBANK | 82.01 | 23.18 | 8.5E-4 | 35.39 | 10.05 |
| wav2vec 2.0 Base | 5.74 | 6.43 | 75.18 | 63.43 | 6.08 |
| HuBERT Base | 5.41 | 6.42 | 81.42 | 64.92 | 5.88 |
| WavLM Base | 4.84 | 6.21 | 84.51 | 65.94 | 4.55 |

The filterbank baseline does poorly on PR, SID, and ER, so the gains come from pretraining. WavLM Base gains most over HuBERT Base on SID (3.09 points) and SD, where speaker identity and overlap matter, which fits its denoising task. Each cell is one published run without error bars.

> [!KEY]
> What you now know: you can learn speech from audio alone by hiding spans and predicting what was hidden, whether a quantized frame, a k-means label, or a clean utterance's label under noise. SUPERB then scores each encoder across tasks.

> [!RECAP]
> - One second at 16 kHz gives 49 frames, with a 400-sample receptive field and a 320-sample hop.
> - The contrastive loss is cross-entropy over cosine logits divided by $\kappa$, and HuBERT and BEST-RQ predict discrete labels at masked frames.
> - WavLM mixes a scaled second signal into a span, and gates a sign-keeping bucketed bias per query. SUPERB feeds a softmax-weighted sum of frozen layers to a light downstream model.

## Build it step by step

The checks test the functions in this order.

1. **`conv_output_length`.** Returns an int. Common mistake: a negative length for a short input instead of 0.
2. **`receptive_field`.** Returns `(field, hop)`. Add `(k - 1) * product` before updating the product.
3. **`span_mask`.** Boolean `(T,)` in and out. Common mistake: masking `span + 1` frames.
4. **`sample_distractors`.** Returns `(M, K)` indices from the other `M - 1` frames. Common mistake: drawing from all frames.
5. **`contrastive_loss`.** Common mistake: not normalizing the context.
6. **`diversity_loss`.** Common mistake: the order in the warning.
7. **`masked_prediction_loss`.** Common mistake: counting unmasked frames.
8. **`best_rq_targets`.** Common mistake: skipping the unit-length scaling.
9. **`kmeans`.** Returns `(centroids, labels)`. Common mistake: labels from before the last update.
10. **`fit_linear_probe` and `probe_accuracy`.** Common mistake: penalizing the bias.
11. **`mix_for_denoising(primary, secondary, start, length, gain_db)`.** Returns a new array with only the span changed. Common mistake: dropping the square root.
12. **`bucket_relative_position(rel, n_buckets, max_distance)`.** Bucket `abs(rel)`, then add `n_buckets // 2` where `rel > 0`. Common mistake: losing the sign.
13. **`gated_relative_bias(queries, u, v, w, bias)`.** One scale per query row. Common mistake: scaling columns, so the gate follows the key.
14. **`superb_weighted_sum(layers, weights)`.** Softmax over the layer axis, then `np.tensordot`. Common mistake: using the raw weights.

## The exercise

You build the fourteen functions above. The checks compare real `torch` convolutions, a gradient, `torch` cross-entropy, and least squares with your code, and confirm the worked WavLM mix, the 8-bucket table, the gated biases 3.5 and 3.79, and the SUPERB mix of 9. The **Run** probes all 13 layers of `wav2vec2-base` and `hubert-base`.

## Further reading

- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP, 2022 (https://arxiv.org/abs/2110.13900). Algorithm 1 is the mixing.
- Yang et al., "SUPERB: Speech processing Universal PERformance Benchmark," Interspeech, 2021 (https://arxiv.org/abs/2105.01051).
- Baevski et al., "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language," ICML, 2022 (https://arxiv.org/abs/2202.03555).
- Chung et al., "W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training," ASRU, 2021 (https://arxiv.org/abs/2108.06209).
