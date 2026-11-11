---
title: Evaluating synthetic speech
slug: 055-evaluating-synthetic-speech
part: "Part 5 — Speech synthesis"
summary: Score synthetic speech with MCD-DTW, F0 error, speaker similarity, ASR WER bias, MOS predictors, and listening-test statistics.
minutes: 45
gpu: false
objectives:
  - Compute mel cepstral distortion between two utterances of different lengths after aligning them with DTW.
  - Measure F0 RMSE over frames voiced in both contours, and the voicing error rate.
  - Score speaker similarity by cosine and intelligibility by the WER gap of an ASR model, and name the choices that change those numbers.
  - "Judge a MOS predictor with LCC and SRCC at the utterance and system levels, using average ranks for ties."
  - "Put a bootstrap confidence interval on a MOS that respects raters and items, and explain why the naive interval is too narrow."
  - "Screen MUSHRA listeners, read CMOS, and compare two systems with the Wilcoxon signed-rank test."
  - Explain consent and watermarking as the minimum practice for voice cloning.
prereqs:
  - 014-mfccs
  - 018-pitch-tracking
  - 021-dynamic-time-warping
  - 026-word-error-rate
  - 042-speaker-verification
exercise: 055-evaluating-synthetic-speech
---

# Evaluating synthetic speech

A recognizer is easy to score: compare its transcript with the right one and count the errors. A synthesizer has no right answer, because one sentence can be read well in many ways, and the judge that finally matters is a person listening. People are slow, expensive, and noisy, so research mixes listening tests with automatic stand-ins. How do you tell whether one voice is better than another, and how sure can you be?

Every automatic measure answers one narrow question: does the spectrum match a recording, is the pitch right, does it sound like the target, can a recognizer understand it? None answers "is it good?", and each can be gamed. A careful report gives several and adds a listening test with an honest interval.

You build the toolkit, from MCD to the Wilcoxon test, and end with whether a voice should be cloned at all.

## What you already know

- DTW aligns two sequences of different lengths by the cheapest monotone path through a cost matrix (see [Dynamic time warping](/lessons/021-dynamic-time-warping)).
- MFCCs are a DCT of log-mel energies, with coefficient 0 tracking loudness (see [MFCCs](/lessons/014-mfccs)).
- WER counts substitutions, deletions, and insertions over the reference's words (see [Word error rate](/lessons/026-word-error-rate)).
- A speaker embedding maps an utterance to a vector, compared by cosine (see [Speaker embeddings and verification](/lessons/042-speaker-verification)).
- YIN gives an F0 per frame, with 0 for unvoiced (see [Pitch tracking](/lessons/018-pitch-tracking)).

## Mel cepstral distortion

*Mel cepstral distortion* (MCD) measures how far a synthesized spectrum is from a recording of the same sentence. It compares the frames' mel cepstra, the low-order coefficients that describe the spectral envelope, and skips coefficient 0, which only says how loud the frame is:

$$
\operatorname{MCD}(c, \hat c) = \frac{10}{\ln 10}\sqrt{2 \sum_{d=1}^{D} (c_d - \hat c_d)^2}\ \text{dB}.
$$

Read it aloud: take the Euclidean distance between the two cepstra from coefficient 1 to $D$ (often 13 or 24), scale it by $\sqrt 2$, and multiply by $10 / \ln 10 = 4.3429$, which turns a natural-log amplitude difference into decibels.

For one frame with $c = (1.0, 0.5, -0.2)$ and $\hat c = (0.9, 0.3, -0.1)$ in coefficients 1 to 3, the squared differences are $0.01 + 0.04 + 0.01 = 0.06$. Doubled that is $0.12$, its square root $0.3464$, and the MCD is $4.3429 \times 0.3464 = 1.504$ dB.

## Align before you compare

A synthesized sentence rarely matches the recording's length, and pairing frame $t$ with frame $t$ compares the start of one vowel with the middle of another. Take $c_1 = (0, 1, 2, 3)$ and a copy with two repeated frames, $c_1 = (0, 0, 1, 2, 2, 3)$. Pairing by position gives a mean MCD of 4.61 dB for the same sentence.

So MCD is computed after DTW: find the cheapest monotone path through the matrix of frame-to-frame MCDs, and average the MCD along it (Figure 1):

$$
\operatorname{MCD}_{\text{DTW}} = \frac{1}{|\pi^*|} \sum_{(n, m) \in \pi^*} \operatorname{MCD}\left(c^{(n)}, \hat c^{(m)}\right).
$$

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="A four-by-six grid of frame-to-frame MCD in dB with the DTW path highlighted; the path has total cost zero">
<text x="296.0" y="28.0" class="fig-small" text-anchor="middle">synthesized frames (c1 = 0, 0, 1, 2, 2, 3)</text>
<text x="141.0" y="52.0" class="fig-small" text-anchor="middle">0</text>
<text x="203.0" y="52.0" class="fig-small" text-anchor="middle">0</text>
<text x="265.0" y="52.0" class="fig-small" text-anchor="middle">1</text>
<text x="327.0" y="52.0" class="fig-small" text-anchor="middle">2</text>
<text x="389.0" y="52.0" class="fig-small" text-anchor="middle">2</text>
<text x="451.0" y="52.0" class="fig-small" text-anchor="middle">3</text>
<text x="100.0" y="85.0" class="fig-small" text-anchor="end">c1 = 0</text>
<text x="100.0" y="127.0" class="fig-small" text-anchor="end">c1 = 1</text>
<text x="100.0" y="169.0" class="fig-small" text-anchor="end">c1 = 2</text>
<text x="100.0" y="211.0" class="fig-small" text-anchor="end">c1 = 3</text>
<text x="20.0" y="250.0" class="fig-small" text-anchor="start">reference frames, one per row</text>
<rect x="110.0" y="60.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="110.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="139.0" y="83.0" class="fig-bold" text-anchor="middle">0.0</text>
<rect x="172.0" y="60.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="172.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="201.0" y="83.0" class="fig-bold" text-anchor="middle">0.0</text>
<rect x="234.0" y="60.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="234.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="263.0" y="83.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="296.0" y="60.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="296.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="325.0" y="83.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="358.0" y="60.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="358.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="387.0" y="83.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="420.0" y="60.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="420.0" y="60.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="449.0" y="83.0" class="fig-small" text-anchor="middle">18.4</text>
<rect x="110.0" y="102.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="110.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="139.0" y="125.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="172.0" y="102.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="172.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="201.0" y="125.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="234.0" y="102.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="234.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="263.0" y="125.0" class="fig-bold" text-anchor="middle">0.0</text>
<rect x="296.0" y="102.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="296.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="325.0" y="125.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="358.0" y="102.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="358.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="387.0" y="125.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="420.0" y="102.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="420.0" y="102.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="449.0" y="125.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="110.0" y="144.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="110.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="139.0" y="167.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="172.0" y="144.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="172.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="201.0" y="167.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="234.0" y="144.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="234.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="263.0" y="167.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="296.0" y="144.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="296.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="325.0" y="167.0" class="fig-bold" text-anchor="middle">0.0</text>
<rect x="358.0" y="144.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="358.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="387.0" y="167.0" class="fig-bold" text-anchor="middle">0.0</text>
<rect x="420.0" y="144.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="420.0" y="144.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="449.0" y="167.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="110.0" y="186.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="110.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="139.0" y="209.0" class="fig-small" text-anchor="middle">18.4</text>
<rect x="172.0" y="186.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="172.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="201.0" y="209.0" class="fig-small" text-anchor="middle">18.4</text>
<rect x="234.0" y="186.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="234.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="263.0" y="209.0" class="fig-small" text-anchor="middle">12.3</text>
<rect x="296.0" y="186.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="296.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="325.0" y="209.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="358.0" y="186.0" width="58.0" height="38.0" class="fill-soft-a"/>
<rect x="358.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="387.0" y="209.0" class="fig-small" text-anchor="middle">6.1</text>
<rect x="420.0" y="186.0" width="58.0" height="38.0" class="fill-b"/>
<rect x="420.0" y="186.0" width="58.0" height="38.0" class="stroke-muted" fill="none"/>
<text x="449.0" y="209.0" class="fig-bold" text-anchor="middle">0.0</text>
<text x="20.0" y="268.0" class="fig-small" text-anchor="start">Pairing by position, (0,0) (1,1) (2,2) (3,3), costs 0, 6.1, 6.1, 6.1 dB: mean 4.61 dB.</text>
<text x="20.0" y="288.0" class="fig-small" text-anchor="start">The DTW path, (0,0) (0,1) (1,2) (2,3) (2,4) (3,5), costs 0 dB.</text>
</svg>
Figure 1. The grid shows the MCD in dB between every reference frame and every synthesized frame. Notice that the synthesized sentence repeats two frames, and that the DTW path absorbs the repeats at zero cost where pairing by position does not.
```

Here $\pi^*$ is the DTW path, a list of frame pairs, and the score is the mean distortion over its $|\pi^*|$ steps. In the figure the path pairs each repeated frame with its original, so the copy scores 0.

> [!KEY]
> Align first, then compare. MCD-DTW averages the frame distortion along the DTW path, so a difference in timing doesn't count as a difference in spectrum.

## Pitch error

MCD mostly ignores pitch, because low cepstral coefficients describe the envelope, not the harmonics. Pitch gets two numbers. The *F0 RMSE* is the root mean squared difference in F0 over frames voiced in both contours. The *V/UV error* is the fraction of frames where one contour is voiced and the other isn't:

$$
\text{F0 RMSE} = \sqrt{\frac{1}{|V|} \sum_{t \in V} \left(f_t - \hat f_t\right)^2}, \qquad
V = \{t : f_t > 0 \text{ and } \hat f_t > 0\}.
$$

Read it aloud: square the hertz differences over frames voiced in both, average, and take the root. When the lengths differ, pair frames along the DTW path first.

Take $f = [200, 210, 0, 220]$ and $\hat f = [190, 215, 180, 0]$ Hz. Frames 0 and 1 are voiced in both, with errors 10 and $-5$ Hz, so the RMSE is $\sqrt{(100 + 25)/2} = 7.91$ Hz. Frames 2 and 3 disagree on voicing, so the V/UV error is $2/4 = 50\%$. Skipping the mask would add a fake 200 Hz error for each unvoiced frame, which stores 0.

## Speaker similarity and intelligibility

**Speaker similarity** runs a speaker verification model on the synthesized clip and on a real clip of the target speaker, and takes the cosine of the two embeddings, from $-1$ to 1. Many zero-shot TTS papers use a fine-tuned WavLM-large verifier. Scores from different verifiers can't be compared, and papers must say which real clip they used, the original prompt or the prompt after a codec.

**Intelligibility** is scored by transcribing the synthesized audio with a fixed ASR model and computing the WER against the text the system was asked to read. The Seed-TTS evaluation uses Whisper-large-v3 for English, and other work uses a HuBERT-Large CTC model. Three effects change the numbers:

- **Normalization.** `Dr. Smith paid $5` and `doctor smith paid five dollars` differ by several edits, but by none after one normalizer writes both the same way. Report the normalizer.
- **The recognizer's language model.** A strong recognizer can guess a word the synthesizer mispronounced, which hides the error. A CTC model with no language model is stricter.
- **The recognizer's own errors.** Every model errs on clean human speech, and more outside its training domain, such as accents and names.

The WER of synthetic speech therefore mixes the synthesizer's faults with the recognizer's. Transcribe the ground-truth recordings of the same sentences with the same model, and report the gap, $\text{WER}_{\text{synth}} - \text{WER}_{\text{real}}$. If a system scores 3.1% and its recordings score 2.4%, the gap is 0.7 points, and that is the number to compare.

> [!TIP]
> Report every automatic score with the tool and settings behind it, such as "WER by Whisper-large-v3 with the Whisper English normalizer". MCD differs by cepstrum type, $D$, and whether $c_0$ is dropped. A score without its tool can't be compared with anything.

## Listening tests

The *mean opinion score* (MOS) is the mean of raters' scores from 1 (bad) to 5 (excellent), following ITU-T P.800. A MOS needs a confidence interval, and the textbook interval, $\bar s \pm 1.96\,\hat\sigma/\sqrt{n}$, is too narrow because it treats all $n$ ratings as independent.

But a harsh rater scores every clip low, and a hard sentence gets low scores from everyone, so a rerun with other raters and sentences moves the mean by far more. A *bootstrap over raters and items* measures the movement directly. Resample whole raters with replacement, resample whole items with replacement, recompute the mean of the resampled table, and repeat a few thousand times. The 2.5th and 97.5th percentiles of those means are a 95% interval.

Figure 2 shows the exercise's simulated test, where 24 raters with personal offsets score 30 items. The bootstrap half-widths are 0.16 to 0.22, against 0.06 or 0.07 from the naive formula.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Mean opinion scores for three systems with a 95 percent bootstrap interval over raters and items, and the much narrower naive interval">
<line x1="120.0" y1="50.0" x2="120.0" y2="205.0" class="stroke-grid"/>
<text x="120.0" y="224.0" class="fig-small" text-anchor="middle">2.5</text>
<line x1="230.0" y1="50.0" x2="230.0" y2="205.0" class="stroke-grid"/>
<text x="230.0" y="224.0" class="fig-small" text-anchor="middle">3</text>
<line x1="340.0" y1="50.0" x2="340.0" y2="205.0" class="stroke-grid"/>
<text x="340.0" y="224.0" class="fig-small" text-anchor="middle">3.5</text>
<line x1="450.0" y1="50.0" x2="450.0" y2="205.0" class="stroke-grid"/>
<text x="450.0" y="224.0" class="fig-small" text-anchor="middle">4</text>
<line x1="560.0" y1="50.0" x2="560.0" y2="205.0" class="stroke-grid"/>
<text x="560.0" y="224.0" class="fig-small" text-anchor="middle">4.5</text>
<text x="340.0" y="248.0" class="fig-small" text-anchor="middle">mean opinion score (1 to 5)</text>
<text x="108.0" y="74.0" class="fig-small" text-anchor="end">reference</text>
<line x1="452.2" y1="62.0" x2="522.6" y2="62.0" class="stroke-b" stroke-width="5"/>
<line x1="474.2" y1="80.0" x2="500.6" y2="80.0" class="stroke-a" stroke-width="5"/>
<circle cx="487.4" cy="62.0" r="5" class="fill-ink"/>
<circle cx="487.4" cy="80.0" r="5" class="fill-ink"/>
<text x="108.0" y="129.0" class="fig-small" text-anchor="end">system A</text>
<line x1="366.4" y1="117.0" x2="454.4" y2="117.0" class="stroke-b" stroke-width="5"/>
<line x1="397.2" y1="135.0" x2="423.6" y2="135.0" class="stroke-a" stroke-width="5"/>
<circle cx="410.4" cy="117.0" r="5" class="fill-ink"/>
<circle cx="410.4" cy="135.0" r="5" class="fill-ink"/>
<text x="108.0" y="184.0" class="fig-small" text-anchor="end">system B</text>
<line x1="175.0" y1="172.0" x2="271.8" y2="172.0" class="stroke-b" stroke-width="5"/>
<line x1="205.8" y1="190.0" x2="236.6" y2="190.0" class="stroke-a" stroke-width="5"/>
<circle cx="221.2" cy="172.0" r="5" class="fill-ink"/>
<circle cx="221.2" cy="190.0" r="5" class="fill-ink"/>
<line x1="120.0" y1="24.0" x2="150.0" y2="24.0" class="stroke-b" stroke-width="5"/>
<text x="158.0" y="28.0" class="fig-small" text-anchor="start">bootstrap over raters and items</text>
<line x1="370.0" y1="24.0" x2="400.0" y2="24.0" class="stroke-a" stroke-width="5"/>
<text x="408.0" y="28.0" class="fig-small" text-anchor="start">naive, all ratings independent</text>
</svg>
Figure 2. Each system's simulated MOS with its bootstrap interval, above the naive interval for the same 720 ratings. Notice that the bootstrap intervals are about three times wider, and that the intervals of the reference and system A just touch.
```

> [!KEY]
> Resample whole raters and whole items, not single ratings. The interval then reflects what would change if you ran the test again with other people and other sentences.

## Compare systems with MUSHRA and CMOS

*MUSHRA* (ITU-R BS.1534-3) shows a listener every system's version of one sentence at once, each with a slider from 0 to 100. Every trial also hides the reference among the clips, plus *anchors*: low-pass copies of the reference with 3.5 kHz and 7 kHz cutoffs. These expose careless listeners. The recommendation excludes a listener who rates the hidden reference below 90 on more than 15% of the items, or the mid-range anchor above 90 on more than 15%.

*CMOS* plays two systems' versions of a sentence in random order, and the listener rates the second against the first from $-3$ (much worse) to $+3$ (much better). Flip the sign of trials that played B first, so positive means A is better.

Both give *paired* ratings. Average each item over listeners, then take $d_i = a_i - b_i$. The *Wilcoxon signed-rank test* uses only the signs and ranks of the differences, so one extreme rating can't dominate. Drop every $d_i = 0$, because a tie has no sign, and let $n$ count the rest. Rank $|d_i|$ from 1 upward, with average ranks for ties. $W$ is the sum of the ranks of the positive $d_i$.

If the systems are equal, $W$ has mean $n(n+1)/4$ and variance $n(n+1)(2n+1)/24 - \sum_g (t_g^3 - t_g)/48$, where $t_g$ is the size of the $g$-th group of tied $|d_i|$. Then $z = (W - \text{mean})/\sqrt{\text{variance}}$.

Take $d = (2, -1, 0, 1, 3, -1, 0, 2)$. Dropping the zeros leaves $n = 6$ and $|d| = (2, 1, 1, 3, 1, 2)$. The three 1s get rank 2 each, the two 2s get 4.5 each, and the 3 gets 6. The positive differences are 2, 1, 3, and 2, so $W = 4.5 + 2 + 6 + 4.5 = 17$. With mean 10.5 and variance $22.75 - 30/48 = 22.125$, $z = 1.38$. Keeping the zeros would push $W$ to 25.

## Predict MOS and judge the predictor

A *MOS predictor* learns to guess a listener's score without a listening test. UTMOS (Saeki et al., 2022) builds on self-supervised speech encoders: the encoder turns a clip into frame features, pooling averages them into one vector $\bar h$, and a small regression head, such as ridge regression $\hat s = w^\top \bar h + b$, trains on listener scores. The exercise builds the scoring.

You judge a predictor at two levels. At the *utterance level*, you compare predicted and listener scores clip by clip. At the *system level*, you first average each system's clips, then compare those few means. The VoiceMOS Challenge (Huang et al., 2022) ranked entries by system-level SRCC, because it wanted to rank systems.

The *linear correlation coefficient* (LCC, or Pearson) measures how well a straight line fits the pairs $(x_i, y_i)$: the sum of $(x_i - \bar x)(y_i - \bar y)$ over the root of the product of the two sums of squares. The *Spearman rank correlation coefficient* (SRCC) is the LCC of the ranks, so it checks only whether the orderings agree. Tied values get the average of their ranks, so `[3, 1, 3, 2]` has ranks `[3.5, 1, 3.5, 2]`. Otherwise a tied value's rank would depend on its input position.

The two can disagree. With listener scores $y = (1, 2, 3, 4, 5)$ and predictions $x = (1, 2, 3, 4, 20)$, the orders agree, so SRCC is 1.00, but the outlier bends the line and LCC is 0.80. Next, take three systems with four clips each, where predicted system means are 2.75, 3.75, and 1.975 against listener means of 3, 4, and 2. At the system level, SRCC is 1.00 and LCC is 0.997, but over all 12 clips they are 0.55 and 0.60. A predictor can rank systems perfectly and still miss clips, so report the level with every number.

## Consent and watermarking

Later chapters build systems that imitate a voice from a few seconds of audio. That helps dubbing and people losing their voice to illness, and it is also the toolkit of a phone scam that sounds like a relative. The minimum practice is:

- **Consent.** Clone only voices whose owners agreed, for the use at hand. A voice in an online video is not consent.
- **Watermarking.** Mark generated audio so it can be recognized later. AudioSeal (2024) embeds an imperceptible watermark that a detector finds in a short excerpt.
- **Detection and disclosure.** Spoofing countermeasures, scored in [Voice conversion and spoofing](/lessons/071-voice-conversion-and-spoofing), detect synthetic speech without a watermark but often miss unseen generators. Labeling synthetic audio depends on neither.

Neither a watermark nor a detector replaces consent. They make careless misuse harder and accidental misuse traceable.

> [!RECAP]
> - MCD-DTW averages the frame MCD, without $c_0$, along the DTW path, and F0 RMSE uses frames voiced in both contours.
> - Report the WER gap against the same ASR on real recordings.
> - A MOS interval resamples whole raters and items, and MUSHRA screening and Wilcoxon handle comparisons.
> - SRCC uses average ranks, and a MOS predictor is judged at the utterance and system levels.

## Build it step by step

Build the functions in the order the checks test them.

1. **`mel_cepstral_distortion(c_ref, c_syn)`** drops coefficient 0 and reduces over the last axis, so broadcasting gives the whole cost matrix. The common mistake is keeping $c_0$.
2. **`dtw_path(cost)`** fills the DTW table and backtracks from the last cell, returning `(i, j)` pairs from `(0, 0)`. The common mistake is stopping one cell early, which drops `(0, 0)`.
3. **`mcd_dtw(c_ref, c_syn)`** averages the cost over the path's cells.
4. **`f0_rmse(f0_ref, f0_syn)`** masks the frames voiced in both and returns `(rmse, vuv_error)`. The common mistake is including frames where one contour is 0.
5. **`speaker_similarity(e1, e2)`** normalizes each row and takes row-wise dot products.
6. **`bootstrap_mos(ratings, n_boot, level, seed)`** indexes with `np.ix_` and takes percentiles. The common mistake is `ratings[r, i]`, which picks single ratings, not a table.
7. **`normalize_text(text)`** and **`word_error_rate(references, transcripts)`** lowercase, strip punctuation, run the Levenshtein recursion over words, and divide total edits by reference words.
8. **`rank_with_ties(x)`**, **`lcc(x, y)`**, **`srcc(x, y)`**, and **`system_level(scores, system_ids)`** give average ranks, the correlations, and one mean per system. The common mistake is ordinal ranks (`argsort` of `argsort`), which don't average ties.
9. **`wilcoxon_signed_rank(a, b)`** drops zero differences, ranks $|d|$ with `rank_with_ties`, and returns `(W, z)`. The common mistake is keeping the zeros.

## Where this breaks

**Reference-based scores punish variety.** MCD and F0 RMSE compare against one recording, so natural but different emphasis scores worse. **Metrics can be gamed.** A system tuned on WER can learn to over-articulate, and one tuned on a MOS predictor can learn its blind spots.

## The exercise

You implement the functions above. The checks reproduce the hand-worked 1.504 dB and 7.91 Hz, compare your DTW path with brute force, and check your bootstrap interval against the spread that rater and item effects cause. They also check ranks with ties, SRCC and LCC against `scipy.stats`, system-level averaging, and the hand-worked $W = 17$.

Press **Run** to synthesize a reference voice, two imitations, and a different speaker. The run prints their scores and bootstrap MOS intervals, scores a stand-in MOS predictor at both levels, runs a Wilcoxon test, and plays all four voices.

## Further reading

- Saeki et al., "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022," Interspeech 2022 (https://arxiv.org/abs/2204.02152).
- Huang et al., "The VoiceMOS Challenge 2022," Interspeech 2022 (https://arxiv.org/abs/2203.11389).
- ITU-R Recommendation BS.1534-3, "Method for the subjective assessment of intermediate quality level of audio systems" (MUSHRA), 2015 (https://www.itu.int/rec/R-REC-BS.1534-3-201510-I/en).
- ITU-T Recommendation P.800, "Methods for subjective determination of transmission quality," 1996, Annex E (CCR) (https://www.itu.int/rec/T-REC-P.800-199608-I/en).
