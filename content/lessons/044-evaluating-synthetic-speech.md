---
title: Evaluating synthetic speech
slug: 044-evaluating-synthetic-speech
part: "Part 5 — Speech synthesis"
summary: Score synthetic speech with MCD-DTW, F0 error, speaker similarity, ASR word error, and MOS with honest intervals, and weigh the ethics of cloning.
minutes: 45
gpu: false
objectives:
  - Compute mel cepstral distortion between two utterances of different lengths after aligning them with DTW.
  - Measure F0 RMSE over frames voiced in both contours, and the voicing error rate.
  - Score speaker similarity by cosine and intelligibility by the WER of an ASR model, and name the choices that change those numbers.
  - "Put a bootstrap confidence interval on a MOS that respects raters and items, and explain why the naive interval is too narrow."
  - Explain consent and watermarking as the minimum practice for voice cloning.
prereqs:
  - 013-mfccs
  - 017-pitch-tracking
  - 020-dynamic-time-warping
  - 024-word-error-rate
  - 037-speaker-verification
exercise: 044-evaluating-synthetic-speech
---

# Evaluating synthetic speech

A recognizer is easy to score: compare its transcript with the right one and count the errors. A synthesizer has no right answer. The same sentence can be read well in many ways, and the only judge that finally matters is a person listening. People are slow, expensive, and noisy, so synthesis research runs on a mix of listening tests and automatic measures that stand in for them. So how do you tell whether one voice is better than another, and how sure can you be?

Every automatic measure answers one narrow question. Does the spectrum match a recording of the same sentence? Is the pitch right? Does it sound like the target speaker? Can a recognizer understand it? None of them answers "is it good?", and each can be gamed. A careful report gives several, names the tools and settings behind each, and adds a listening test with an honest confidence interval.

You build the standard toolkit: mel cepstral distortion with alignment, F0 error, speaker similarity, the word error rate of an ASR model, and a bootstrap interval for a mean opinion score. The chapter ends with the part of evaluation that isn't a number: whether a voice should be cloned at all.

## What you already know

- Dynamic time warping aligns two sequences of different lengths by the cheapest monotone path through a cost matrix (see [Dynamic time warping](/lessons/020-dynamic-time-warping)).
- MFCCs are a DCT of log-mel energies, and coefficient 0 tracks loudness (see [MFCCs](/lessons/013-mfccs)).
- Word error rate counts substitutions, deletions, and insertions against the reference's word count (see [Word error rate](/lessons/024-word-error-rate)).
- A speaker embedding maps an utterance to a vector, compared by cosine (see [Speaker embeddings and verification](/lessons/037-speaker-verification)).
- YIN gives an F0 per frame, and 0 marks an unvoiced frame (see [Pitch tracking](/lessons/017-pitch-tracking)).

## Mel cepstral distortion

*Mel cepstral distortion* (MCD) measures how far a synthesized spectrum is from a recording of the same sentence. It compares the frames' mel cepstra, the low-order coefficients that describe the spectral envelope, and skips coefficient 0, which only says how loud the frame is:

$$
\operatorname{MCD}(c, \hat c) = \frac{10}{\ln 10}\sqrt{2 \sum_{d=1}^{D} (c_d - \hat c_d)^2}\ \text{dB}.
$$

Read it aloud: take the Euclidean distance between the two cepstra from coefficient 1 to $D$, scale it by $\sqrt 2$, and multiply by $10 / \ln 10 = 4.3429$. That constant turns a natural-log amplitude difference into decibels. Common choices of $D$ are 13 and 24. In numpy, for arrays of frames:

```python
diff = c_ref[..., 1:] - c_syn[..., 1:]
mcd = 10 / np.log(10) * np.sqrt(2 * np.sum(diff**2, axis=-1))
```

For one frame with $c = (1.0, 0.5, -0.2)$ and $\hat c = (0.9, 0.3, -0.1)$ in coefficients 1 to 3, the squared differences are $0.01 + 0.04 + 0.01 = 0.06$. Doubled that is $0.12$, its square root $0.3464$, and the MCD is $4.3429 \times 0.3464 = 1.504$ dB.

## Align before you compare

A synthesized sentence rarely has the same length as the recording. Pairing frame $t$ with frame $t$ compares the start of one vowel with the middle of another, and it can't even pair frames past the shorter clip's end. Take a reference with first cepstral coefficients $c_1 = (0, 1, 2, 3)$ and a copy that holds two frames a little longer, $c_1 = (0, 0, 1, 2, 2, 3)$. It is the same sentence, but pairing by position gives errors of $0, 1, 1, 1$ in $c_1$, which is a mean MCD of 4.61 dB.

So MCD is computed after DTW: build the matrix of frame-to-frame MCDs, find the cheapest monotone path through it, and average the MCD along that path (Figure 1):

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

Read it aloud: $\pi^*$ is the DTW path, a list of frame pairs, and the score is the mean distortion over its $|\pi^*|$ steps. In the figure the path pairs each repeated frame with its original, so the copy scores exactly 0.

> [!KEY]
> Align first, then compare. MCD-DTW averages the frame distortion along the DTW path, so a difference in timing doesn't count as a difference in spectrum.

> [!WARNING]
> MCD numbers are comparable only within one implementation. Toolkits differ in how they compute the cepstrum (SPTK mel-cepstra with a warping of 0.42 at 16 kHz, or MFCCs), in $D$, in whether they drop $c_0$, and in whether they divide by the path length or the reference length. Report the tool.

## Pitch error

MCD mostly ignores pitch, because the low cepstral coefficients describe the envelope, not the harmonics. Pitch gets its own two numbers. The *F0 RMSE* is the root mean squared difference in F0 over frames voiced in both contours. The *V/UV error* is the fraction of frames where one contour is voiced and the other isn't:

$$
\text{F0 RMSE} = \sqrt{\frac{1}{|V|} \sum_{t \in V} \left(f_t - \hat f_t\right)^2}, \qquad
V = \{t : f_t > 0 \text{ and } \hat f_t > 0\}.
$$

Read it aloud: keep only the frames where both have a pitch, square the differences in hertz, average, and take the square root. When the lengths differ, pair the frames along the DTW path first.

Take $f = [200, 210, 0, 220]$ and $\hat f = [190, 215, 180, 0]$ Hz. Frames 0 and 1 are voiced in both, with errors 10 and $-5$ Hz, so the RMSE is $\sqrt{(100 + 25)/2} = 7.91$ Hz. Frames 2 and 3 disagree on voicing, so the V/UV error is $2/4 = 50\%$. Skipping the mask would be a bug, because an unvoiced frame stores 0 and would add a fake error of 200 Hz. Because pitch is heard as ratios, many papers report the error in cents, $1200 \log_2(f / \hat f)$, instead of hertz.

## Speaker similarity and intelligibility

**Speaker similarity** runs a speaker verification model on the synthesized clip and on a real clip of the target speaker, and takes the cosine of the two embeddings. It ranges from $-1$ to 1. Many zero-shot TTS papers use a WavLM-large model fine-tuned for verification. The number depends entirely on the verifier, so scores from different verifiers can't be compared. Papers also say which real clip they compared against, the original prompt or the prompt after passing through a codec, because the two give different numbers.

**Intelligibility** is scored by transcribing the synthesized audio with a fixed ASR model and computing the WER against the text the system was asked to read. The Seed-TTS evaluation uses Whisper-large-v3 for English, and other work uses a HuBERT-Large CTC model fine-tuned on LibriSpeech. Two choices change the numbers:

- **Normalization.** `Dr. Smith paid $5` and `doctor smith paid five dollars` differ by several word edits, but by none once one normalizer writes both the same way. Normalize both sides with the same normalizer, and report which.
- **The recognizer's language model.** Whisper can guess a word the synthesizer mispronounced, which hides errors. A CTC model with no language model is stricter.

> [!TIP]
> Report every automatic score with the model and settings that produced it, such as "WER by Whisper-large-v3 with the Whisper English normalizer" or "SIM by WavLM-large". A score without its tool can't be compared with anything.

## Listening tests

The *mean opinion score* (MOS) is the listening test most papers report. Raters hear each clip and score it from 1 (bad) to 5 (excellent), following ITU-T P.800, and the MOS is the mean. *CMOS* plays two systems' versions of the same sentence and asks for a preference from $-3$ to $+3$. *MUSHRA* plays all systems side by side, with a hidden reference and a low-pass anchor, on a 0 to 100 scale, and it is more sensitive for high-quality systems. Automatic MOS predictors such as UTMOS are useful for regression tests, but they inherit the biases of their training data.

A MOS without a confidence interval says little, and the textbook interval is too narrow. It treats all $n$ ratings as independent:

$$
\bar s \pm 1.96\,\frac{\hat\sigma}{\sqrt{n}}.
$$

Read it aloud: the mean, plus or minus 1.96 standard errors, where the standard error is the standard deviation of the ratings over the square root of their number. With 200 ratings and $\hat\sigma = 0.8$, the half-width is $1.96 \times 0.8 / \sqrt{200} = 0.111$.

But ratings aren't independent. A harsh rater scores every clip low, and a hard sentence gets low scores from everyone. Rerun the test with other raters and other sentences, and the mean moves by far more than $\hat\sigma / \sqrt{n}$ predicts. A *bootstrap over raters and items* measures that movement directly. Resample whole raters with replacement, resample whole items with replacement, recompute the mean of the resampled table, and repeat a few thousand times. The 2.5th and 97.5th percentiles of those means are a 95% interval:

```python
r = rng.integers(0, n_raters, n_raters)
i = rng.integers(0, n_items, n_items)
means[b] = ratings[np.ix_(r, i)].mean()
```

Figure 2 shows the exercise's simulated test, where 24 raters with personal offsets each score 30 items. The bootstrap half-widths come out between 0.16 and 0.22, while the naive formula over all 720 ratings claims 0.06 or 0.07. A MOS also depends on the raters and on the other systems in the same test, so compare MOS values only within one test.

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

## Consent and watermarking

Later chapters build zero-shot and voice-conversion systems that imitate a voice from a few seconds of audio. That is useful for dubbing, for accessibility, and for people who are losing their voice to illness. It is also the whole toolkit of a phone scam that sounds like a relative, or a fake recording of a public figure. Evaluation is where a project decides what it is willing to ship, so the minimum practice belongs here:

- **Consent.** Clone only voices whose owners agreed to it, for the use at hand. Research datasets such as LJSpeech and LibriTTS come from recordings released for reuse, and your own voice is always available. A voice in a video online is not consent.
- **Watermarking.** Mark generated audio so it can be recognized later. AudioSeal, from 2024, embeds an imperceptible watermark and ships a detector that finds it even in a short excerpt of a longer clip. Resemble AI's open Chatterbox models embed their own watermark, called Perth, in every output.
- **Detection and disclosure.** Spoofing countermeasures, which [Audio classification and voice conversion](/lessons/059-classification-and-voice-conversion) scores by equal error rate, try to detect synthetic speech without a watermark. They often miss generators they weren't trained on. Labeling synthetic audio as synthetic doesn't depend on either.

A watermark can be removed by a determined attacker, and a detector can be fooled. Neither replaces consent. They make careless misuse harder and accidental misuse traceable.

> [!RECAP]
> - MCD is $(10/\ln 10)\sqrt{2\sum_{d \ge 1}(c_d - \hat c_d)^2}$ dB per frame, without $c_0$, and MCD-DTW averages it along the DTW path.
> - F0 RMSE uses frames voiced in both contours, and V/UV error is the fraction of frames whose voicing disagrees.
> - Speaker similarity is a cosine of embeddings, and WER comes from a fixed ASR model after normalizing both texts.
> - A MOS interval comes from resampling whole raters and whole items.
> - Consent comes first, and watermarks and detectors only help afterward.

## Build it step by step

Build the functions in the order the checks test them.

1. **`mel_cepstral_distortion(c_ref, c_syn)`** slices off coefficient 0 and reduces over the last axis, so broadcasting `c_ref[:, None, :]` against `c_syn[None, :, :]` gives the whole cost matrix. The common mistake is keeping $c_0$, which turns MCD into a loudness difference.
2. **`dtw_path(cost)`** fills the DTW table with the diagonal, down, and right moves and backtracks from the last cell. Return the path as a list of `(i, j)` pairs from `(0, 0)`. The common mistake is stopping the backtrack one cell early, which drops `(0, 0)`.
3. **`mcd_dtw(c_ref, c_syn)`** builds the matrix, finds the path, and averages over the path's cells.
4. **`f0_rmse(f0_ref, f0_syn)`** masks the frames voiced in both and returns `(rmse, vuv_error)`. The common mistake is including frames where one contour is 0.
5. **`speaker_similarity(e1, e2)`** normalizes each row and takes the row-wise dot product.
6. **`bootstrap_mos(ratings, n_boot, level, seed)`** resamples rater and item indices, indexes with `np.ix_`, and takes percentiles. The common mistake is `ratings[r, i]`, which picks a diagonal of single ratings instead of a table.
7. **`normalize_text(text)`** and **`word_error_rate(references, transcripts)`** lowercase, strip punctuation with a regular expression, run the Levenshtein recursion over words, and divide the total edits by the total reference words.

## Where this breaks

**Reference-based scores punish good variety.** MCD and F0 RMSE compare against one recording. A system that reads the sentence with different, but natural, emphasis scores worse than one that averages. These scores suit vocoders and codecs, which aim to reproduce the input, better than TTS.

**Metrics can be gamed.** A system tuned on WER can learn to speak slowly and over-articulate. One tuned on a MOS predictor can learn that predictor's blind spots. Keep a listening test in the loop.

**Rater quality and item count.** Crowd raters vary, and some don't listen, so include hidden references and deliberately degraded anchors. Thirty sentences can all be easy, and the item resampling in the bootstrap makes that visible as a wide interval.

## The exercise

You implement `mel_cepstral_distortion`, `dtw_path`, `mcd_dtw`, `f0_rmse`, `speaker_similarity`, `bootstrap_mos`, `normalize_text`, and `word_error_rate`. The checks reproduce the hand-worked 1.504 dB and 7.91 Hz, compare your DTW path's cost with brute-force enumeration of every path, confirm that MCD-DTW is zero for a copy with repeated frames, and check that your bootstrap interval matches the spread that rater and item effects cause, far wider than the naive formula.

Press **Run** to synthesize a reference voice from a source and formant filters, two imitations, and a different speaker. The run prints each one's MCD-DTW, F0 RMSE, V/UV error, and speaker similarity, draws the MCD matrix that DTW searches, reports bootstrap MOS intervals from a simulated listening test, and plays all four voices.
