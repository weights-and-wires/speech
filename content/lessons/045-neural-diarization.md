---
title: Neural diarization
slug: 045-neural-diarization
part: "Part 4 — Neural speech recognition"
summary: Train a network to output per-speaker activity with a permutation-invariant loss, count speakers with attractors, and pick one voice out of an overlap.
minutes: 45
gpu: false
objectives:
  - Compute the permutation-invariant binary cross-entropy loss over per-speaker activity, and its gradient.
  - Break an overlap-aware diarization error rate into missed speech, false alarm, and confusion, counting an overlapped frame once per speaker.
  - Read the number of speakers from encoder-decoder attractor probabilities, and compute speaker activity from attractors.
  - Select one speaker's frames with a cosine against an enrollment embedding, and say why target-speaker VAD handles overlap.
prereqs:
  - 042-speaker-verification
  - 044-diarization
exercise: 045-neural-diarization
---

# Neural diarization

Two people on a call interrupt each other. For about a second, both talk. A clustering diarizer cuts the call into windows and gives each window one speaker label, so in that second it names one person and loses the other.

This chapter changes the output instead. A network reads the whole recording and, for every frame, says how likely each speaker is to be talking. Several speakers can be active at once, so overlap is no longer a special case. Two problems follow. The speaker slots have no fixed meaning, which the permutation-invariant loss solves, and the number of speakers is unknown, which attractors solve. You also score the result by speaker-frame, and see how a speaker embedding can ask the network about one voice.

## What you already know

- Diarization labels time with speaker turns, and the DER counts missed speech, false alarm, and confusion after the best speaker mapping (see [Diarization](/lessons/044-diarization)).
- A speaker embedding places clips of one voice close together by cosine (see [Speaker embeddings and verification](/lessons/042-speaker-verification)).
- A sigmoid turns a score into a probability, and binary cross-entropy (BCE) scores a probability against a 0 or 1 target.
- A **frame** is a short slice of time, and this chapter uses 10 ms frames.
- A loss's gradient with respect to the network's outputs drives training.

## The picture: one activity track per speaker

Figure 1 shows the target. The reference has one row per speaker, and each cell says whether that speaker talks in that frame. Speaker A talks in frames 0 to 7 and speaker B in frames 5 to 11, so frames 5, 6, and 7 hold both voices. The system has two output slots, also one row each.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Reference activity for two speakers over 14 frames and a system output with a fixed order, with the overlap frames 5 to 7 marked">
<text x="20" y="38" class="fig-small fig-bold">reference</text>
<text x="70" y="66" text-anchor="end" class="fig-small">A</text>
<rect x="90" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="122" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="154" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="186" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="218" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="250" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="282" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="314" y="50" width="30" height="22" rx="3" class="fill-a"/>
<rect x="346" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="378" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="410" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="442" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="474" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="506" y="50" width="30" height="22" rx="3" class="fill-sunk"/>
<text x="70" y="96" text-anchor="end" class="fig-small">B</text>
<rect x="90" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="122" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="154" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="186" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="218" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="250" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="282" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="314" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="346" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="378" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="410" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="442" y="80" width="30" height="22" rx="3" class="fill-c"/>
<rect x="474" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="506" y="80" width="30" height="22" rx="3" class="fill-sunk"/>
<text x="20" y="148" class="fig-small fig-bold">system slots</text>
<text x="70" y="176" text-anchor="end" class="fig-small">slot 0</text>
<rect x="90" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="122" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="154" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="186" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="218" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="250" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="282" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="314" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="346" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="378" y="160" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="410" y="160" width="30" height="22" rx="3" class="fill-c"/>
<rect x="442" y="160" width="30" height="22" rx="3" class="fill-c"/>
<rect x="474" y="160" width="30" height="22" rx="3" class="fill-c"/>
<rect x="506" y="160" width="30" height="22" rx="3" class="fill-c"/>
<text x="70" y="206" text-anchor="end" class="fig-small">slot 1</text>
<rect x="90" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="122" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="154" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="186" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="218" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="250" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="282" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="314" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="346" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="378" y="190" width="30" height="22" rx="3" class="fill-a"/>
<rect x="410" y="190" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="442" y="190" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="474" y="190" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="506" y="190" width="30" height="22" rx="3" class="fill-sunk"/>
<rect x="248" y="46" width="96" height="60" rx="4" class="stroke-bad" fill="none"/>
<text x="296" y="124" text-anchor="middle" class="fig-small fig-bold">overlap</text>
<text x="105" y="232" text-anchor="middle" class="fig-small">0</text>
<text x="169" y="232" text-anchor="middle" class="fig-small">2</text>
<text x="233" y="232" text-anchor="middle" class="fig-small">4</text>
<text x="297" y="232" text-anchor="middle" class="fig-small">6</text>
<text x="361" y="232" text-anchor="middle" class="fig-small">8</text>
<text x="425" y="232" text-anchor="middle" class="fig-small">10</text>
<text x="489" y="232" text-anchor="middle" class="fig-small">12</text>
<text x="300" y="248" text-anchor="middle" class="fig-small">frame index</text>
</svg>
Figure 1. Speaker A and speaker B overlap in frames 5 to 7 (red outline). The system's slots are in the opposite order from the reference, so slot 1 is A, and a fixed-order loss would score it against the wrong rows.
```

The network is a stack of layers that reads frame features, often a transformer encoder, and ends in $C$ sigmoids per frame, one per speaker slot. Call the matrix of raw scores `logits`, with shape `(T, C)` for $T$ frames. Each sigmoid is independent, so a frame can have zero, one, or several slots above 0.5. That makes diarization a multi-label problem, and the target is the same boolean matrix the overlap DER scores in the previous chapter. This is the design of end-to-end neural diarization (EEND; Fujita et al., 2019).

## The problem with slots: permutation-invariant loss

The obvious loss is binary cross-entropy between slot $c$ and speaker $c$. It fails, because nothing tells the network which slot holds speaker A. In Figure 1 the system put A in slot 1 and B in slot 0. A loss that insists on slot 0 for A scores a system that found the right people as badly wrong, and training pushes the network to unlearn the right answer.

The fix is the one used for separating voices in [Separating voices](/lessons/071-source-separation), applied to activity instead of waveforms. **Permutation-invariant training** (PIT) scores every assignment of slots to speakers and keeps the cheapest, so the network is judged on whether it separated the speakers and not on which slot it used.

Work a small case. Three frames, two speakers. The reference is A in frames 0 and 1, and B in frames 1 and 2, so frame 1 is an overlap. The network's probabilities, with one row per frame and one column per slot, are $\begin{bmatrix} 0.1 & 0.9 \\ 0.8 & 0.9 \\ 0.9 & 0.1 \end{bmatrix}$. Slot 1 follows A and slot 0 follows B, so the picture is right with the order swapped. BCE for one cell is $-\log p$ when the target is 1 and $-\log(1-p)$ when it is 0. Keeping the order, six cells give $\{2.303,\ 2.303,\ 0.223,\ 0.105,\ 2.303,\ 2.303\}$, a mean of 1.590. Swapping, they give $\{0.105,\ 0.105,\ 0.223,\ 0.105,\ 0.105,\ 0.105\}$, a mean of 0.125.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Two bars: the fixed-order loss is 1.590 and the loss under the swapped assignment is 0.125">
<line x1="150" y1="30" x2="150" y2="140" class="stroke-muted"/>
<text x="142" y="60" text-anchor="end" class="fig-small">fixed order (0, 1)</text>
<rect x="150" y="40" width="95.4" height="30" rx="3" class="fill-bad"/>
<text x="253.4" y="60" class="fig-small fig-bold">1.590</text>
<text x="142" y="110" text-anchor="end" class="fig-small">swapped (1, 0)</text>
<rect x="150" y="90" width="7.5" height="30" rx="3" class="fill-b"/>
<text x="165.5" y="110" class="fig-small fig-bold">0.125</text>
<text x="150" y="170" class="fig-small">mean BCE per frame and speaker, in nats; PIT keeps the smaller bar</text>
</svg>
Figure 2. On the three-frame example, keeping the slot order costs 1.590 nats and the swapped order costs 0.125. PIT keeps the swapped order, because the network separated the speakers correctly.
```

The formula, for a clip of $T$ frames and $C$ slots with probabilities $p_{t,c}$ and targets $y_{t,s}$:

$$
\mathcal{L}_{\text{PIT}} = \min_{\pi \in \mathcal{S}_C} \frac{1}{TC} \sum_{t=1}^{T} \sum_{c=1}^{C} \text{BCE}\left(p_{t,c},\ y_{t,\pi(c)}\right)
$$

Read it aloud: $\mathcal{S}_C$ is the set of all orderings of the $C$ speakers. For each ordering $\pi$, slot $c$ is scored against speaker $\pi(c)$, over every frame and every slot, and the loss is the smallest of those averages. One ordering covers the whole clip, so a speaker cannot hop between slots halfway through. In numpy, with `perm` an ordering from `itertools.permutations`:

```python
loss = min(bce_with_logits(logits, labels[:, list(perm)]) for perm in permutations(range(C)))
```

The gradient is short, because the winning ordering is a fixed choice while you differentiate. With $\hat\pi$ the winner, $\partial \mathcal{L}/\partial z_{t,c} = \left(\sigma(z_{t,c}) - y_{t,\hat\pi(c)}\right)/(TC)$, where $z$ is the logit and $\sigma$ the sigmoid. In the exercise, `finite_difference` confirms it.

> [!KEY]
> A slot has no fixed meaning, so PIT scores every assignment of slots to speakers over the whole clip and keeps the smallest BCE. Training then rewards separating the speakers and ignores which slot each one lands in.

The cost is $C!$ orderings. That is cheap for two to four speakers and grows fast after that, which is why EEND systems fix a maximum slot count.

## Scoring overlap by speaker-frame

You met the overlap DER in the previous chapter. Here you break it into its three error types, because they tell you what to fix. On frame $t$, let $n^{\text{ref}}_t$ and $n^{\text{hyp}}_t$ count the active speakers in the reference and hypothesis, and let $n^{\text{ok}}_t$ count the hypothesis speakers that match a reference speaker under the best mapping. Then:

| Error | Per frame |
|---|---|
| Missed speech | $\max(n^{\text{ref}}_t - n^{\text{hyp}}_t,\ 0)$ |
| False alarm | $\max(n^{\text{hyp}}_t - n^{\text{ref}}_t,\ 0)$ |
| Confusion | $\min(n^{\text{ref}}_t, n^{\text{hyp}}_t) - n^{\text{ok}}_t$ |

The denominator is $\sum_t n^{\text{ref}}_t$, so an overlapped frame counts once per speaker. Sum the three rows and you get $\max(n^{\text{ref}}_t, n^{\text{hyp}}_t) - n^{\text{ok}}_t$ again, the numerator from the previous chapter.

Work Figure 1, 14 frames. The reference has 8 frames of A and 7 of B, so 15 speaker-frames. The system maps slot 1 to A and slot 0 to B, which matches 10 speaker-frames, more than any other mapping. Frames 5, 6, and 7 have two reference speakers but only slot 1, so 3 are missed. Frames 8 and 9 have B in the reference, but only slot 1, which means A, so 2 are confusion. Frames 12 and 13 are silent in the reference, but slot 0 talks, so 2 are false alarms. The DER is $(3 + 2 + 2)/15 = 46.7\%$. A scorer that counted frames with any speech would divide by 12 and report 58.3%, which is too high and no longer a percentage of the reference speech.

Clustering cannot lower the 3 missed frames, because it has one label per frame. An end-to-end network targets them.

## An unknown number of speakers: attractors

A network with $C$ fixed slots handles at most $C$ speakers, and a meeting does not announce its size. **Encoder-decoder attractors** (EEND-EDA; Horiguchi et al., 2020) generate the slots instead of fixing them. An **attractor** is a vector in the same space as the frame embeddings, one per speaker, and it acts like that speaker's template.

The network first produces a frame embedding $e_t$, a vector of dimension $D$ for each frame. An LSTM encoder-decoder reads the sequence of embeddings and emits attractors $a_1, a_2, \dots$, one per step, each with an existence probability that says whether this attractor is a real speaker. Speaker activity for attractor $s$ is the sigmoid of a dot product:

$$
\hat y_{t,s} = \sigma\left(e_t^\top a_s\right)
$$

Read it aloud: a frame that points in the direction of an attractor scores high for that speaker. Because each speaker has its own sigmoid, two attractors can both exceed 0.5 in one frame. Work a tiny case with $D = 2$, frame embeddings $(1, 0)$, $(1, 1)$, $(0, 1)$, and attractors $a_1 = (3, -1)$ and $a_2 = (-1, 3)$. Frame 0 gives logits $3$ and $-1$, so probabilities 0.953 and 0.269. Frame 1 gives 2 and 2, so 0.881 for both: an overlap. Frame 2 gives 0.269 and 0.953. In numpy this is `1 / (1 + np.exp(-(E @ A.T)))`.

The speaker count comes from the existence probabilities. At inference, generate attractors in order and stop at the first probability below a threshold, such as 0.5. With probabilities $0.93, 0.81, 0.20$ the count is 2. If a fourth attractor came out at 0.74, it would not count, because the decoder stops at the third. Training teaches this by labeling the real speakers' attractors as 1 and one extra attractor as 0.

> [!WARNING]
> The count is the number of attractors before the first low probability, not the number of probabilities above the threshold. Counting every probability above 0.5 turns a stray late attractor into a phantom speaker.

## Asking about one speaker: target-speaker VAD

Sometimes you already know who to look for, such as a customer on a call. **Target-speaker voice activity detection** (TS-VAD; Medennikov et al., 2020) gives the network that voice as an input. For each frame it predicts whether each target speaker is active, using an embedding of each target as extra input. The paper uses i-vectors, estimated from a first clustering pass and then refined. The network sees the mixture and the target together, so it can say the target is talking even when someone louder is also talking.

The simplest version of the selection is a cosine test. Given an enrollment embedding $e$ and one embedding per frame $f_t$, the target is active when:

$$
\frac{f_t^\top e}{\lVert f_t \rVert\, \lVert e \rVert} \ge \tau
$$

Read it aloud: the frame and the enrollment point the same way, within a margin. Take enrollment $(1, 0)$, frames $(3, 0)$, $(2, 2)$, and $(0, 5)$, and a threshold of 0.5. The cosines are 1.0, 0.707, and 0.0, so the mask is `[True, True, False]`. Dividing by the norms is the point: a quiet frame of the target, such as $(0.03, 0.04)$ against $(3, 4)$, has a tiny dot product and a cosine of 1.0. A TS-VAD network learns this comparison jointly with the audio instead of thresholding it.

> [!KEY]
> End-to-end diarization outputs one sigmoid per speaker per frame, so overlap is two ones in a row. PIT removes the slot-order problem, attractors remove the fixed speaker count, and a target embedding lets the network answer for one named voice.

> [!RECAP]
> - EEND outputs `(T, C)` speaker activity as independent sigmoids, so a frame can hold several speakers.
> - PIT-BCE takes the smallest mean BCE over all slot orderings for the whole clip, and its gradient is the winning ordering's BCE gradient.
> - The overlap DER splits into miss, false alarm, and confusion, divided by reference speaker-frames.
> - Attractors give an unknown speaker count: stop at the first existence probability below the threshold.
> - Target-speaker VAD conditions on a speaker embedding, and a cosine mask is its simplest form.

## Build it step by step

The checks test the functions in this order. All take numpy arrays.

1. **`bce_with_logits(logits, labels)`** returns the mean BCE as a float. Use the stable form $\max(z, 0) - zy + \log(1 + e^{-|z|})$. The common mistake is to take the sigmoid first and then the log, which overflows for large logits.
2. **`pit_bce_loss(logits, labels)`** returns `(loss, perm)`. Loop over `itertools.permutations`, index the label columns with `list(perm)`, and keep the smallest. The common mistake is to compare slot $c$ with speaker $c$ only.
3. **`pit_bce_grad(logits, labels)`** returns `(T, C)`. Reuse the winning ordering. The common mistake is to divide by $T$ instead of $TC$.
4. **`overlap_frames(activity)`** takes a boolean `(speakers, frames)` matrix and returns a boolean `(frames,)` mask. Sum over speakers and compare with 2.
5. **`der_components(reference, hypothesis)`** returns a dict with `miss`, `false_alarm`, `confusion`, `total`, and `der`. Build the overlap table with an integer matrix product and map with `scipy.optimize.linear_sum_assignment`. The common mistake is dividing by the number of frames that have any speech.
6. **`attractor_count(attractor_probs, threshold)`** returns an int. Find the first index below the threshold. The common mistake is `np.sum(probs >= threshold)`.
7. **`attractor_activity(embeddings, attractors)`** returns `(T, S)`, a sigmoid of a matrix product.
8. **`target_speaker_mask(frame_embeddings, enrollment, threshold)`** returns a boolean `(T,)`. Normalize both sides before the dot product. The common mistake is thresholding the raw dot product, so loud voices pass and quiet ones fail.

## Where this breaks

**Slot limits.** A network with a fixed number of slots, or $C!$ orderings, caps the number of speakers. EEND-EDA lifts the cap in principle, but accuracy still drops as speakers are added.

**Arrival-time ordering.** PIT searches $C!$ orderings on every step. NVIDIA's NeMo Sortformer instead sorts the targets by the time each speaker first talks and compares the outputs in that order, which its documentation calls the Sort Loss, with permutation-invariant loss as the alternative it describes.

**Domain shift.** Fujita et al. train on simulated mixtures and then adapt with real recordings, because simulated overlap differs from real conversation.

## The exercise

You build `bce_with_logits`, `pit_bce_loss`, and `pit_bce_grad` for the loss, `overlap_frames` and `der_components` for scoring, `attractor_count` and `attractor_activity` for the speaker count, and `target_speaker_mask` for the target voice.

The checks confirm the loss against a brute-force search over three speakers, its gradient against `finite_difference` when the best ordering is a swap, and the DER components on the planted timeline and on random timelines against a slow reference. One check catches a fixed-slot BCE that never searches the orderings, and another catches a denominator that counts an overlapped frame once. The **Run** builds the planted timeline, prints the components, and shows the reference and system activity side by side.

## Further reading

- Fujita et al., "End-to-End Neural Speaker Diarization with Permutation-Free Objectives," Interspeech 2019 (https://arxiv.org/abs/1909.05952). The multi-label formulation and the permutation-free loss.
- Fujita et al., "End-to-End Neural Speaker Diarization with Self-attention," ASRU 2019 (https://arxiv.org/abs/1909.06247). Replaces the recurrent layers with self-attention.
- Horiguchi et al., "End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors," Interspeech 2020 (https://arxiv.org/abs/2005.09921).
- Medennikov et al., "Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario," Interspeech 2020 (https://arxiv.org/abs/2005.07272).
- NVIDIA NeMo Framework User Guide, "Models," speaker diarization section, for Sortformer and its Sort Loss (https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html).
