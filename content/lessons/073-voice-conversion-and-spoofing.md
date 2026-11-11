---
title: Voice conversion and spoofing
slug: 073-voice-conversion-and-spoofing
part: "Part 8 — Beyond recognition and synthesis"
summary: Convert a voice by matching self-supervised frames to a target speaker, then score spoof detectors by their error rates and equal error rate.
minutes: 55
gpu: false
objectives:
  - Convert a voice with kNN-VC, and explain why matching self-supervised features changes the speaker but keeps the words.
  - Say what k trades off, and what the vocoder adds after the matching step.
  - Compute false rejection and false acceptance rates for a spoofing countermeasure, and the equal error rate where they cross, with interpolation.
  - Explain why a countermeasure is judged together with a speaker verification system in the tandem detection cost.
  - State the consent rules that apply before you convert anyone's voice.
prereqs:
  - 008-phase-and-griffin-lim
  - 039-self-supervised-speech
  - 042-speaker-verification
exercise: 073-voice-conversion-and-spoofing
---

# Voice conversion and spoofing

If a model separates *what* is said from *who* says it, you can keep the first and swap the second. That is *voice conversion*: the words and timing of one speaker, in the voice of another. It works well enough to cause a problem. A caller who sounds like your colleague asks you to approve a payment, and the voice is not your colleague's.

The bank on the other end has the opposite task. It must decide whether a recording is *bona fide*, a real person speaking, or a *spoof*, speech made by a synthesizer or a converter. This chapter builds both halves. You convert a voice by replacing each frame with its nearest neighbors in the target's speech. Then you score a spoofing detector by the two ways it can be wrong, and by the rate where those errors are equal.

## What you already know

- Self-supervised models such as WavLM give one vector per 20 ms, and their middle layers carry phonetic content.
- Cosine similarity compares directions, not lengths: $x \cdot r / (\lVert x \rVert \lVert r \rVert)$.
- The equal error rate (EER) is where false acceptances equal false rejections, from [Speaker embeddings and verification](/lessons/042-speaker-verification).
- Griffin-Lim turns a magnitude spectrogram back into audio by estimating a phase, from [Phase and Griffin-Lim](/lessons/008-phase-and-griffin-lim).

## Voice conversion by nearest neighbors

Every voice conversion method splits speech into *content*, the phones and their timing, and *speaker identity*, the timbre of one vocal tract. kNN-VC (Baas et al., 2023) needs no training for the conversion itself. It relies on what [Self-supervised speech](/lessons/039-self-supervised-speech) measured: the middle layers of WavLM encode phones nearly the same way for every speaker, while each frame still carries the speaker's timbre.

That gives a recipe in three steps. First, extract features: the paper takes layer 6 of WavLM-Large, one 1,024-dimensional vector per 20 ms, for the source clip and for all the target's reference audio, which is the *matching set*. Second, replace each source frame $x_t$ with the average of its $k$ nearest matching frames:

$$
y_t = \frac{1}{k} \sum_{r \in \text{kNN}_k(x_t)} r, \qquad d(x, r) = 1 - \frac{x \cdot r}{\lVert x \rVert\, \lVert r \rVert}
$$

Read it aloud: find the $k$ reference frames with the smallest cosine distance to the source frame, and average them. The paper uses $k = 4$. Third, a vocoder, a HiFi-GAN trained to turn WavLM features back into a waveform, renders the averaged features as audio. The vocoder is trained once on many speakers, never on the target, so the only target-specific data is the matching set.

```python
q = source / np.linalg.norm(source, axis=1, keepdims=True)
r = matching / np.linalg.norm(matching, axis=1, keepdims=True)
nearest = np.argpartition(-(q @ r.T), k - 1, axis=1)[:, :k]
converted = matching[nearest].mean(axis=1)
```

### Work kNN-VC by hand

Take one source frame $x = (1, 0)$ and four matching frames $r_1 = (0.9, 0.1)$, $r_2 = (0.2, 1)$, $r_3 = (1, -0.2)$, and $r_4 = (-1, 0)$. The cosines are 0.994, 0.196, 0.981, and $-1$. With $k = 2$, the nearest are $r_1$ and $r_3$, and the converted frame is their mean, $(0.95, -0.05)$. With $k = 1$, it would be an exact copy of $r_1$. The query is always the source and the keys are the matching set: matching on the other axis would find source frames for each target frame and return a clip of the wrong length.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A source frame at (1, 0), four matching frames, and the converted frame at (0.95, -0.05), the mean of the two nearest by angle">
  <defs><marker id="c065-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-a"/></marker></defs>
  <line x1="210" y1="125" x2="500" y2="125" class="stroke-grid"/><line x1="300" y1="20" x2="300" y2="235" class="stroke-grid"/>
  <line x1="300" y1="125" x2="388" y2="125" class="stroke-a" marker-end="url(#c065-arrow)"/>
  <circle cx="381" cy="116" r="5" class="fill-b"/><circle cx="318" cy="35" r="5" class="fill-muted"/>
  <circle cx="390" cy="143" r="5" class="fill-b"/><circle cx="210" cy="125" r="5" class="fill-muted"/>
  <circle cx="385" cy="130" r="5" class="fill-c"/>
  <text x="400" y="104" class="fig-small">r1 (0.9, 0.1)</text><text x="330" y="40" class="fig-small">r2 (0.2, 1)</text>
  <text x="400" y="160" class="fig-small">r3 (1, -0.2)</text><text x="216" y="115" class="fig-small">r4 (-1, 0)</text>
  <text x="405" y="135" class="fig-small fig-bold">converted (0.95, -0.05)</text>
  <text x="330" y="150" class="fig-small">x (1, 0)</text>
</svg>
Figure 3. The source frame x points along the horizontal axis. The two nearest frames by angle, r1 and r3 (orange), average to the converted frame (green), which lies in the target's region of feature space. The far frames r2 and r4 play no part.
```

### Why matching keeps the words and changes the voice

The output is built only from target frames, so it can only sound like the target: every averaged vector lies among vectors that the target's vocal tract produced. The words survive because the *choice* of which frames to use comes from the source. Two frames of the same phone are close in layer 6 whichever speaker said them, so the nearest target frame to a source "ah" is a target "ah". The output also has exactly one frame per source frame, so the source's timing and rhythm carry over unchanged.

You can see the argument on synthetic features, where you control the phones. Give each frame a phone vector, plus a speaker offset, plus noise. The source is speaker 0 and the matching set is 400 frames of speaker 1, about 50 per phone. Figure 2 measures how many converted frames still sit nearest the source's phone, for different $k$.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Fraction of converted frames that keep the source phone for k from 1 to 400">
  <line x1="70" y1="180" x2="560" y2="180" class="stroke-muted"/><line x1="70" y1="30" x2="70" y2="180" class="stroke-muted"/>
  <text x="16" y="105" class="fig-small" transform="rotate(-90 16 105)" text-anchor="middle">frames keeping the phone</text>
  <rect x="90" y="30.0" width="50" height="150.0" class="fill-soft-b stroke-b"/>
  <text x="115" y="24.0" text-anchor="middle" class="fig-small fig-bold">100%</text>
  <text x="115" y="198" text-anchor="middle" class="fig-small">1</text>
  <rect x="168" y="30.0" width="50" height="150.0" class="fill-soft-b stroke-b"/>
  <text x="193" y="24.0" text-anchor="middle" class="fig-small fig-bold">100%</text>
  <text x="193" y="198" text-anchor="middle" class="fig-small">4</text>
  <rect x="246" y="30.0" width="50" height="150.0" class="fill-soft-b stroke-b"/>
  <text x="271" y="24.0" text-anchor="middle" class="fig-small fig-bold">100%</text>
  <text x="271" y="198" text-anchor="middle" class="fig-small">16</text>
  <rect x="324" y="30.0" width="50" height="150.0" class="fill-soft-b stroke-b"/>
  <text x="349" y="24.0" text-anchor="middle" class="fig-small fig-bold">100%</text>
  <text x="349" y="198" text-anchor="middle" class="fig-small">64</text>
  <rect x="402" y="96.2" width="50" height="83.8" class="fill-soft-b stroke-b"/>
  <text x="427" y="90.2" text-anchor="middle" class="fig-small fig-bold">56%</text>
  <text x="427" y="198" text-anchor="middle" class="fig-small">128</text>
  <rect x="480" y="161.2" width="50" height="18.8" class="fill-soft-b stroke-b"/>
  <text x="505" y="155.2" text-anchor="middle" class="fig-small fig-bold">12%</text>
  <text x="505" y="198" text-anchor="middle" class="fig-small">400</text>
  <text x="315" y="222" text-anchor="middle" class="fig-small">k, the number of neighbors averaged (matching set: 400 frames)</text>
</svg>
Figure 2. Content survives while k stays below the roughly 50 matching frames per phone, then falls once the average reaches across phones. At k = 400 every output frame is the mean of the whole matching set.
```

At $k = 1$ through 64, every frame keeps its phone, and the speaker offset is the target's. At $k = 128$ the average starts to reach into other phones, and only 56% of frames keep the right one. At $k = 400$ the neighbors are the whole matching set, so every output frame equals the same average and the content is gone. The rule is that $k$ must stay below the number of matching frames per phone. Below the cliff, a small $k$ copies single frames and their noise, while averaging a few smooths it. The paper's $k = 4$ sits well below the cliff, with a matching set of minutes, not 400 frames.

The paper reports that below about 30 seconds of reference audio, kNN-VC falls behind other methods, though even 5 seconds keeps moderate quality. The Run can't load WavLM or a vocoder on the CPU, so it keeps the method and swaps the parts: two synthetic voices, a 110 Hz source and a 210 Hz target with formants raised 18%, mel cepstra with each speaker's mean removed as the features, and Griffin-Lim as the vocoder. On its seeded voices, 96% of matched frames are the same vowel as the source frame, and the output's pitch is 211 Hz, against the target's 208 Hz and the source's 109 Hz.

> [!KEY]
> kNN-VC changes the speaker by replacing every source frame with the average of its nearest neighbors in the target's own speech. It keeps the words because the source decides which frames to pick, and it keeps $k$ small enough that the average stays inside one phone.

## Spoofing countermeasures

The ASVspoof challenges give the detection side a shared setup. A *countermeasure* is a classifier that outputs one score per recording, where a higher score means "bona fide". Its test set mixes bona fide recordings with spoofs from many attack systems, and the *logical access* condition feeds synthetic or converted speech straight to the system, while the *physical access* condition replays recordings through a speaker. The test set includes attacks that the training set never contained, because a detector must handle generators it has not seen.

You score a countermeasure with the same two errors as speaker verification, renamed for this job. At a threshold $\theta$, a recording is accepted as bona fide when its score is at or above $\theta$. The *false rejection rate* (FRR) is the fraction of bona fide scores below $\theta$, and the *false acceptance rate* (FAR) is the fraction of spoof scores at or above $\theta$. Raising $\theta$ lowers FAR and raises FRR, so you can't minimize both. The *equal error rate* is the value where they are equal.

### Work an EER by hand

Take 3 bona fide scores $(0.9, 0.5, 0.4)$ and 4 spoof scores $(0.6, 0.3, 0.1, 0.0)$. Use each distinct score as a threshold, plus one value above the largest:

| Threshold | FRR | FAR |
|---|---|---|
| 0.00 | 0 | 1.00 |
| 0.10 | 0 | 0.75 |
| 0.30 | 0 | 0.50 |
| 0.40 | 0 | 0.25 |
| 0.50 | 0.333 | 0.25 |
| 0.60 | 0.667 | 0.25 |
| 0.90 | 0.667 | 0 |

With only seven trials, the rates move in steps, and no threshold makes them equal. FRR is below FAR at 0.40 and above it at 0.50, so the crossing is between. Join the two points with straight lines and solve for where they meet. The gap $\mathrm{FAR} - \mathrm{FRR}$ is $d_0 = 0.25$ at 0.40 and $d_1 = -0.083$ at 0.50, so the crossing is a fraction $w = d_0 / (d_0 - d_1) = 0.75$ of the way along. The threshold is $0.40 + 0.75 \times 0.10 = 0.475$, and the EER is $0 + 0.75 \times 0.333 = 0.25$ (Figure 3).

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="False rejection and false acceptance rates against threshold for three bona fide and four spoof scores, crossing at 0.25 near threshold 0.475">
  <line x1="70" y1="190" x2="550" y2="190" class="stroke-muted"/><line x1="70" y1="40" x2="70" y2="190" class="stroke-muted"/>
  <line x1="70" y1="190.0" x2="550" y2="190.0" class="stroke-grid"/><text x="62" y="194.0" text-anchor="end" class="fig-small">0.00</text>
  <line x1="70" y1="152.5" x2="550" y2="152.5" class="stroke-grid"/><text x="62" y="156.5" text-anchor="end" class="fig-small">0.25</text>
  <line x1="70" y1="115.0" x2="550" y2="115.0" class="stroke-grid"/><text x="62" y="119.0" text-anchor="end" class="fig-small">0.50</text>
  <line x1="70" y1="77.5" x2="550" y2="77.5" class="stroke-grid"/><text x="62" y="81.5" text-anchor="end" class="fig-small">0.75</text>
  <line x1="70" y1="40.0" x2="550" y2="40.0" class="stroke-grid"/><text x="62" y="44.0" text-anchor="end" class="fig-small">1.00</text>
  <text x="70.0" y="208" text-anchor="middle" class="fig-small">0.0</text>
  <text x="166.0" y="208" text-anchor="middle" class="fig-small">0.2</text>
  <text x="262.0" y="208" text-anchor="middle" class="fig-small">0.4</text>
  <text x="358.0" y="208" text-anchor="middle" class="fig-small">0.6</text>
  <text x="454.0" y="208" text-anchor="middle" class="fig-small">0.8</text>
  <text x="550.0" y="208" text-anchor="middle" class="fig-small">1.0</text>
  <path d="M70.0 190.0 L118.0 190.0 L214.0 190.0 L262.0 190.0 L310.0 140.0 L358.0 90.0 L502.0 90.0 L550.0 40.0" fill="none" class="stroke-b" stroke-width="2"/>
  <path d="M70.0 40.0 L118.0 77.5 L214.0 115.0 L262.0 152.5 L310.0 152.5 L358.0 152.5 L502.0 190.0 L550.0 190.0" fill="none" class="stroke-a" stroke-width="2"/>
  <circle cx="70.0" cy="190.0" r="3" class="fill-b"/>
  <circle cx="118.0" cy="190.0" r="3" class="fill-b"/>
  <circle cx="214.0" cy="190.0" r="3" class="fill-b"/>
  <circle cx="262.0" cy="190.0" r="3" class="fill-b"/>
  <circle cx="310.0" cy="140.0" r="3" class="fill-b"/>
  <circle cx="358.0" cy="90.0" r="3" class="fill-b"/>
  <circle cx="502.0" cy="90.0" r="3" class="fill-b"/>
  <circle cx="550.0" cy="40.0" r="3" class="fill-b"/>
  <circle cx="70.0" cy="40.0" r="3" class="fill-a"/>
  <circle cx="118.0" cy="77.5" r="3" class="fill-a"/>
  <circle cx="214.0" cy="115.0" r="3" class="fill-a"/>
  <circle cx="262.0" cy="152.5" r="3" class="fill-a"/>
  <circle cx="310.0" cy="152.5" r="3" class="fill-a"/>
  <circle cx="358.0" cy="152.5" r="3" class="fill-a"/>
  <circle cx="502.0" cy="190.0" r="3" class="fill-a"/>
  <circle cx="550.0" cy="190.0" r="3" class="fill-a"/>
  <circle cx="298.0" cy="152.5" r="5" class="fill-c"/>
  <line x1="302.0" y1="156.5" x2="319.6" y2="167.0" class="stroke-muted"/>
  <text x="319.6" y="175.0" class="fig-small fig-bold">EER 0.25 at threshold 0.475</text>
  <text x="367.6" y="55.0" class="fig-small">false rejection (bona fide below)</text>
  <text x="94.0" y="70.0" class="fig-small">false acceptance (spoof at or above)</text>
  <text x="310" y="232" text-anchor="middle" class="fig-small">threshold (countermeasure score)</text>
  <text x="16" y="115" class="fig-small" transform="rotate(-90 16 115)" text-anchor="middle">error rate</text>
</svg>
Figure 3. Each dot is the pair of error rates at one score used as a threshold, and the lines join them. The rates cross between the dots at 0.40 and 0.50, at a rate of 0.25 and a threshold of 0.475.
```

Reporting the first threshold where FRR reaches FAR, with no interpolation, gives 0.333 at 0.50, a third too high. In code, `np.searchsorted` on the sorted scores counts how many fall below each threshold, so one call gives a rate for every threshold at once:

```python
frr = np.searchsorted(np.sort(bonafide), th, side="left") / len(bonafide)
far = 1 - np.searchsorted(np.sort(spoof), th, side="left") / len(spoof)
```

The `side="left"` counts only scores strictly below the threshold, so a score equal to the threshold is accepted, bona fide or spoof. The Run draws both rate curves for a toy detector whose scores are two unit Gaussians 2.5 apart. Theory puts that EER at $\Phi(-1.25) = 10.6\%$, and the 400 seeded samples per class measure 11.0%.

### The tandem idea

A countermeasure rarely runs alone. It sits in front of a speaker verification system, and a trial passes only when both say yes. That creates a cost the EER can't see. A countermeasure with a low EER that sets its threshold high rejects many real, enrolled users, and one that sets it low lets spoofs through to a verifier that may accept them. The *tandem detection cost function* (t-DCF; Kinnunen et al., 2018) combines the verifier's errors and the countermeasure's errors into one cost, weighted by how often target, non-target, and spoof trials occur and how much each error costs. It judges a countermeasure by the system it protects. ASVspoof 2019 used the minimum t-DCF as its main metric and the EER as a second one. This course grades only the EER.

Detectors overfit to the attacks they trained on, and an unseen vocoder can break them. The complementary approach is *provenance*: mark generated audio when it is made, as AudioSeal (2024) does with an imperceptible watermark.

> [!KEY]
> A countermeasure is scored by two errors, rejecting real speech and accepting spoofs, and the EER is where they cross. Interpolate the crossing between thresholds, and remember that a deployed countermeasure is judged together with the verifier it protects.

## Consent and misuse

A voice identifies a person. The EU's GDPR treats biometric data used to identify someone as a special category, and Illinois's BIPA regulates voiceprints. A converted voice can impersonate someone to their bank, their family, or their employer. The rules for this course follow: convert only your own voice, with your own opt-in, or voices from public corpora whose licenses allow it, such as LibriSpeech and VCTK. Label converted audio as synthetic when you share it, and watermark it when you can.

> [!WARNING]
> "It's only a demo" is not consent. The pipeline that converts a synthetic voice in this exercise converts a real one, so the rules apply the moment the input is a recording of a person.

> [!RECAP]
> - kNN-VC replaces each source frame with the mean of its $k = 4$ nearest target frames by cosine distance, then a vocoder renders the result.
> - It keeps the words because the source picks the frames, and it changes the voice because every picked frame is the target's. A $k$ larger than the matching frames per phone washes the content out.
> - A countermeasure's FRR is the fraction of bona fide scores below the threshold, and its FAR is the fraction of spoof scores at or above it.
> - The EER is where they cross, found by interpolating between the two thresholds that straddle the crossing.
> - The t-DCF judges a countermeasure together with the speaker verification system behind it.

## Build it step by step

Build the functions in the order the checks test them.

1. **`cosine_knn(query, keys, k)`** returns `(T, k)` indices, nearest first. Normalize, multiply, take `np.argpartition` on the negated similarities, then sort those $k$. The common mistake is passing the arguments the wrong way round, so the matching set becomes the query.
2. **`knn_vc(source, matching, k)`** indexes the matching set with those indices and averages over axis 1. The common mistake is averaging source frames, which keeps the source voice.
3. **`error_rates(bonafide, spoof, thresholds)`** returns `(frr, far)`, one value per threshold, with `np.searchsorted` on sorted scores. The common mistake is a strict inequality on the wrong side, which moves one trial at every tied threshold.
4. **`equal_error_rate(bonafide, spoof)`** returns `(eer, threshold)`. Use the distinct scores plus one larger value as thresholds, find the first index where FRR reaches FAR, and interpolate with the previous index. The common mistakes are skipping the interpolation, and swapping the classes, which reports a value above 0.5 for a good detector.

## Where this breaks

- **A matching set that is too small.** With a few seconds of target speech, some phones have no good neighbor, and kNN-VC copies the wrong one.
- **Countermeasures that don't generalize.** A detector with a low EER on known attacks can fail on a vocoder it never saw. Report results on unseen attacks.
- **An EER on few trials.** With 50 spoof trials, one trial moves FAR by 2 points, so an EER of 4% and 6% can't be told apart.

## The exercise

You build `cosine_knn` and `knn_vc` for voice conversion, and `error_rates` and `equal_error_rate` for a spoofing countermeasure. Press **Run** to hear a synthetic source voice, the target voice, and the source converted into the target, and to see a toy countermeasure's error curves with its EER.

The checks verify your neighbors against a full sort, `knn_vc` against the formula with $k = 1$ and $k = 4$, and kNN-VC on synthetic features, which must keep the source phone and move to the target speaker for at least 95% of frames. They check that the content washes out when $k$ covers the whole matching set. They check `error_rates` at a tied threshold, the EER on crafted scores and on two Gaussians, and the interpolated threshold of 0.475 from this chapter.

## Further reading

- Shinji Watanabe, Speech Recognition and Understanding (11-751/18-781), Carnegie Mellon University, fall 2022. The schedule has no lecture on voice conversion or spoofing, so it is not cited for this chapter (https://www.wavlab.org/activities/2022/11751-2022f/).
- Neil Zhang, "Speech Anti-Spoofing," guest lecture in Computer Audition (ECE 277/477), University of Rochester, fall 2023. It covers the ASVspoof challenge series, EER, and joint optimization with speaker verification (https://hajim.rochester.edu/ece/sites/zduan/teaching/ece477/lectures/GuestLecture_SpeechAntiSpoofing_NeilZhang.pdf).
- Baas, van Niekerk, and Kamper, "Voice Conversion With Just Nearest Neighbors," Interspeech 2023 (https://arxiv.org/abs/2305.18975).
- Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification," Odyssey 2018 (https://arxiv.org/abs/1804.09618).
- Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019 (https://arxiv.org/abs/1904.05441).
