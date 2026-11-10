---
title: Self-supervised speech
slug: 039-self-supervised-speech
part: "Part 4 — Neural speech recognition"
summary: Learn speech from audio alone with wav2vec 2.0, HuBERT, and BEST-RQ, build their losses and targets, and probe what each layer learned.
minutes: 80
gpu: true
objectives:
  - Compute the frame count, hop, and receptive field of the wav2vec 2.0 feature encoder, and the fraction of frames that span masking hides.
  - Implement the contrastive loss with distractors from masked frames, and the diversity loss that keeps a codebook in use.
  - Implement HuBERT's masked prediction of k-means targets and BEST-RQ's frozen random-projection targets.
  - Fit a closed-form linear probe to every layer of wav2vec2-base and hubert-base, and read where speaker and content information lives.
prereqs:
  - 028-neural-acoustic-models
  - 030-ctc-forward-backward
  - 037-conformers-and-specaugment
exercise: 039-self-supervised-speech
---

# Self-supervised speech

A transcribed hour of speech costs hours of careful listening and typing. An untranscribed hour costs almost nothing, and broadcasts supply millions of them. How can a model learn from the cheap kind?

Self-supervised learning trains an encoder on audio alone: hide some frames, and ask the model to recognize them from the context. To do that, it must learn what speech sounds are and how they follow each other. [Fine-tuning speech models](/lessons/047-fine-tuning-speech-models) turns such an encoder into a recognizer. Here you build the pretraining losses and targets and probe what the layers learn. [WavLM and benchmarking self-supervised models](/lessons/040-wavlm-and-ssl-benchmarks) then adds noise to the input and ranks the encoders.

## What you already know

- A strided convolution shortens a sequence, from [Neural acoustic models](/lessons/028-neural-acoustic-models).
- CTC trains an encoder against a transcript, from [Computing CTC efficiently](/lessons/030-ctc-forward-backward).
- SpecAugment hides spans of the input, from [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment).

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

> [!KEY]
> What you now know: you can learn speech from audio alone by hiding spans and predicting what was hidden, whether a quantized frame or a k-means label, and you can read from a probe which layers hold the speaker and which hold the content.

> [!RECAP]
> - One second at 16 kHz gives 49 frames, with a 400-sample receptive field and a 320-sample hop.
> - The contrastive loss is cross-entropy over cosine logits divided by $\kappa$, and HuBERT and BEST-RQ predict discrete labels at masked frames.
> - A closed-form linear probe shows that speaker identity peaks in the lower layers and content in the middle ones, and k-means units keep the phones and drop most of the voice.

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

## The exercise

You build the ten functions above. The checks compare real `torch` convolutions, `torch` cross-entropy, and least squares with your code, and confirm the hand-worked contrastive and diversity losses. The **Run** probes all 13 layers of `wav2vec2-base` and `hubert-base`.

## Further reading

- Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020 (https://arxiv.org/abs/2006.11477).
- Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," IEEE/ACM TASLP, 2021 (https://arxiv.org/abs/2106.07447).
- Chiu et al., "Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition," ICML, 2022 (https://arxiv.org/abs/2202.01855).
