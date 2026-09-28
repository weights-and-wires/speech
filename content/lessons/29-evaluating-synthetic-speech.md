---
title: Evaluating synthetic speech
slug: 29-evaluating-synthetic-speech
part: "Part 5 — Speech synthesis"
summary: Score synthetic speech with MCD-DTW, F0 error, speaker similarity, ASR word error, and MOS with honest intervals, and weigh the ethics of cloning.
minutes: 60
gpu: false
objectives:
  - Compute mel cepstral distortion between two utterances of different lengths after aligning them with DTW.
  - Measure F0 RMSE over frames voiced in both contours, and the voicing error rate.
  - Score speaker similarity by cosine and intelligibility by the WER of an ASR model, and name the choices that change those numbers.
  - "Put a bootstrap confidence interval on a MOS that respects raters and items, and explain why the naive interval is too narrow."
  - Explain consent and watermarking as the minimum practice for voice cloning.
prereqs:
  - 12-dynamic-time-warping
  - 14-language-models-and-word-error-rate
  - 23-speaker-recognition-and-diarization
  - 28-normalizing-flows-and-vits
exercise: 29-evaluating-synthetic-speech
---

# Evaluating synthetic speech

A recognizer is easy to score: compare its transcript with the right one and
count the errors. A synthesizer has no right answer. The same sentence can
be read well in many ways, and the only judge that finally matters is a
person listening. People are slow, expensive, and noisy, so synthesis
research runs on a mix of listening tests and automatic measures that stand
in for them.

Every automatic measure answers one narrow question. Does the spectrum match
a recording of the same sentence? Is the pitch right? Does it sound like the
target speaker? Can a recognizer understand it? None of them answers "is it
good?", and each can be gamed. A careful report gives several, names the
tools and settings behind each, and gives a listening test with an honest
confidence interval.

In this lesson you build the standard toolkit: mel cepstral distortion with
alignment, F0 error, speaker similarity, the word error rate of an ASR model,
and a bootstrap interval for a mean opinion score. The lesson ends with the
part of evaluation that isn't a number: whether a voice should be cloned at
all.

## What you already know

- Dynamic time warping aligns two sequences of different lengths by the cheapest monotone path through a cost matrix (lesson 12, *Dynamic time warping*).
- MFCCs are a DCT of log-mel energies, and coefficient 0 tracks loudness (lesson 8, *MFCCs and normalization*).
- Word error rate counts substitutions, deletions, and insertions against the reference's word count (lesson 14, *Language models and word error rate*).
- A speaker embedding maps an utterance to a vector, compared by cosine (lesson 23, *Speaker recognition and diarization*).
- YIN gives an F0 per frame, and 0 marks an unvoiced frame (lesson 10, *Pitch tracking*).

## Mel cepstral distortion

*Mel cepstral distortion* (MCD) measures how far a synthesized spectrum is
from a recording of the same sentence. It compares the frames' mel cepstra,
the low-order coefficients that describe the spectral envelope, and skips
coefficient 0, which only says how loud the frame is:

$$
\operatorname{MCD}(c, \hat c) = \frac{10}{\ln 10}\sqrt{2 \sum_{d=1}^{D} (c_d - \hat c_d)^2}\ \text{dB}.
$$

Read it aloud: take the Euclidean distance between the two cepstra from
coefficient 1 to $D$, scale it by $\sqrt 2$, and multiply by
$10 / \ln 10 = 4.3429$. That constant turns a natural-log amplitude
difference into decibels, as in lesson 1's $20 \log_{10}$. Common choices of
$D$ are 13 and 24. In numpy, for arrays of frames:

```python
diff = c_ref[..., 1:] - c_syn[..., 1:]
mcd = 10 / np.log(10) * np.sqrt(2 * np.sum(diff**2, axis=-1))
```

For one frame with $c = (1.0, 0.5, -0.2)$ and $\hat c = (0.9, 0.3, -0.1)$ in
coefficients 1 to 3, the squared differences are $0.01 + 0.04 + 0.01 = 0.06$.
Doubled that is $0.12$, its square root $0.3464$, and the MCD is
$4.3429 \times 0.3464 = 1.504$ dB.

### Align before you compare

A synthesized sentence rarely has the same length as the recording. Pairing
frame $t$ with frame $t$ compares the start of one vowel with the middle of
another. So MCD is computed after DTW: build the matrix of frame-to-frame
MCDs, find the cheapest monotone path through it, and average the MCD along
that path:

$$
\operatorname{MCD}_{\text{DTW}} = \frac{1}{|\pi^*|} \sum_{(n, m) \in \pi^*} \operatorname{MCD}\left(c^{(n)}, \hat c^{(m)}\right).
$$

Read it aloud: $\pi^*$ is the DTW path, a list of frame pairs, and the score
is the mean distortion over its $|\pi^*|$ steps. A copy of the reference with
some frames repeated scores exactly 0, because the path can pair every copy
with its original.

> [!WARNING]
> MCD numbers are comparable only within one implementation. Toolkits differ
> in how they compute the cepstrum (SPTK mel-cepstra with a warping of 0.42 at
> 16 kHz, or MFCCs), in $D$, in whether $c_0$ is dropped, and in whether they
> divide by the path length or the reference length. Report the tool.

## Pitch error

MCD mostly ignores pitch, because the low cepstral coefficients describe the
envelope, not the harmonics. Pitch gets its own two numbers. The *F0 RMSE* is
the root mean squared difference in F0 over frames voiced in both contours.
The *V/UV error* is the fraction of frames where one contour is voiced and
the other isn't:

$$
\text{F0 RMSE} = \sqrt{\frac{1}{|V|} \sum_{t \in V} \left(f_t - \hat f_t\right)^2}, \qquad
V = \{t : f_t > 0 \text{ and } \hat f_t > 0\}.
$$

Read it aloud: keep only the frames where both have a pitch, square the
differences in hertz, average, and take the square root. When the lengths
differ, pair the frames along the DTW path first.

For example, take $f = [200, 210, 0, 220]$ and $\hat f = [190, 215, 180, 0]$ Hz.
Frames 0 and 1 are voiced in both, with errors 10 and $-5$ Hz. The RMSE is
$\sqrt{(100 + 25)/2} = 7.91$ Hz. Frames 2 and 3 disagree on voicing, so the
V/UV error is $2/4 = 50\%$. Because pitch is heard as ratios, many papers
report the error in cents, $1200 \log_2(f / \hat f)$, instead of hertz.

## Speaker similarity and intelligibility

**Speaker similarity** runs a speaker verification model, such as the ones
in lesson 23, on the synthesized clip and on a real clip of the target
speaker, and takes the cosine of the two embeddings. It ranges from $-1$ to
1. Most papers since 2024 use a WavLM-large model fine-tuned for
verification. The number depends entirely on the verifier, so scores from
different verifiers can't be compared. For zero-shot cloning, papers also
say which real clip they compared against: the original prompt, or the
prompt after passing through the codec. The two give different numbers.

**Intelligibility** is scored by transcribing the synthesized audio with a
fixed ASR model and computing the WER against the text the system was asked
to read. The Seed-TTS evaluation uses Whisper-large-v3 for English; other
work uses a HuBERT-Large CTC model fine-tuned on LibriSpeech. Two choices
change the numbers:

- **Normalization.** "Dr. Smith paid $5" and "doctor smith paid five dollars" differ by three word edits, but by none once one normalizer writes both the same way. Normalize both sides with the same normalizer, and report which.
- **The recognizer's language model.** Whisper can guess a word the synthesizer mispronounced, which hides errors. A CTC model with no language model is stricter.

> [!TIP]
> Report every automatic score with the model and settings that produced
> it, such as "WER by Whisper-large-v3 with the Whisper English normalizer" or
> "SIM by WavLM-large". A score without its tool can't be compared with
> anything.

## Listening tests

The *mean opinion score* (MOS) is the listening test most papers report.
Raters hear each clip and score it from 1 (bad) to 5 (excellent), following
ITU-T P.800, and the MOS is the mean. Variants ask other questions. *CMOS*
plays two systems' versions of the same sentence and asks for a preference
from $-3$ to $+3$. *MUSHRA* plays all systems side by side, with a hidden
reference and a low-pass anchor, on a 0 to 100 scale. It is more sensitive
for high-quality systems. Automatic MOS predictors such as UTMOS, a
wav2vec 2.0 model trained to predict human scores, are useful for regression
tests, but they inherit the biases of their training data.

### An honest interval

A MOS without a confidence interval says little, and the textbook interval
is too narrow. It treats all $n$ ratings as independent:

$$
\bar s \pm 1.96\,\frac{\hat\sigma}{\sqrt{n}}.
$$

Read it aloud: the mean, plus or minus 1.96 standard errors, where the
standard error is the standard deviation of the ratings over the square root
of their number. With 200 ratings and $\hat\sigma = 0.8$, the half-width is
$1.96 \times 0.8 / \sqrt{200} = 0.111$.

But ratings aren't independent. A harsh rater scores every clip low, and a
hard sentence gets low scores from everyone. Rerun the test with other raters
and other sentences, and the mean moves by far more than $\hat\sigma /
\sqrt{n}$ predicts. A *bootstrap over raters and items* measures that
movement directly. Resample whole raters with replacement, resample whole
items with replacement, recompute the mean of the resampled table, and repeat
a few thousand times. The 2.5th and 97.5th percentiles of those means are a
95% interval. In numpy:

```python
r = rng.integers(0, n_raters, n_raters)
i = rng.integers(0, n_items, n_items)
means[b] = ratings[np.ix_(r, i)].mean()
```

In the Run's simulated test, 24 raters with personal offsets each score 30
items. The bootstrap half-widths come out between 0.16 and 0.22, while
the naive formula over all 720 ratings claims 0.06 or 0.07. Two systems whose naive intervals
don't overlap can still be indistinguishable. And a MOS depends on the
raters and on the other systems in the same test, so compare MOS values only
within one test.

> [!KEY]
> Resample whole raters and whole items, not single ratings. The interval
> then reflects what would change if you ran the test again with other people
> and other sentences.

## Consent and watermarking

The zero-shot and voice-conversion systems later in this course can imitate a
voice from a few seconds of audio. That is useful for dubbing, for
accessibility, and for people who are losing their voice to illness. It is
also the whole toolkit of a phone scam that sounds like a relative, or a fake
recording of a public figure. Evaluation is where a project decides what it
is willing to ship, so the minimum practice belongs here:

- **Consent.** Clone only voices whose owners agreed to it, for the use at hand. Research datasets such as LJSpeech and LibriTTS come from recordings released for reuse, and your own voice is always available. A voice in a video online is not consent.
- **Watermarking.** Mark generated audio so it can be recognized later. AudioSeal, from 2024, embeds an imperceptible watermark and ships a detector that finds it even in a short excerpt of a longer clip. Resemble AI's open Chatterbox models embed their own watermark, called Perth, in every output.
- **Detection and disclosure.** Spoofing countermeasures, which lesson 40 scores by equal error rate, try to detect synthetic speech without a watermark. They lag behind new generators. Labeling synthetic audio as synthetic doesn't depend on either.

A watermark can be removed by a determined attacker, and a detector can be
fooled. Neither replaces consent. They make careless misuse harder and
accidental misuse traceable.

> [!RECAP]
> - MCD is $(10/\ln 10)\sqrt{2\sum_{d \ge 1}(c_d - \hat c_d)^2}$ dB per frame, without $c_0$.
> - MCD-DTW averages frame MCD along the cheapest monotone path through the MCD matrix.
> - F0 RMSE uses frames voiced in both contours; V/UV error is the fraction of frames whose voicing disagrees.
> - Speaker similarity is a cosine of embeddings; WER comes from a fixed ASR model after normalizing both texts.
> - A MOS interval comes from resampling whole raters and whole items.

## Build it step by step

Build the functions in the order the checks test them.

1. **`mel_cepstral_distortion(c_ref, c_syn)`** slices off coefficient 0 and
   reduces over the last axis, so broadcasting `c_ref[:, None, :]` against
   `c_syn[None, :, :]` gives the whole cost matrix. The common mistake is
   keeping $c_0$, which turns MCD into a loudness difference.
2. **`dtw_path(cost)`** fills lesson 12's table with the diagonal, down, and
   right moves and backtracks from the last cell. Return the path as a list
   of `(i, j)` pairs from `(0, 0)`. The common mistake is stopping the
   backtrack one cell early, which drops `(0, 0)`.
3. **`mcd_dtw(c_ref, c_syn)`** builds the matrix, finds the path, and
   averages over the path's cells.
4. **`f0_rmse(f0_ref, f0_syn)`** masks the frames voiced in both and returns
   `(rmse, vuv_error)`. The common mistake is including frames where one
   contour is 0, which adds errors of 200 Hz.
5. **`speaker_similarity(e1, e2)`** normalizes each row and takes the row-wise
   dot product.
6. **`bootstrap_mos(ratings, n_boot, level, seed)`** resamples rater and item
   indices, indexes with `np.ix_`, and takes percentiles. The common mistake
   is `ratings[r, i]`, which picks a diagonal of single ratings instead of a
   table.
7. **`normalize_text(text)`** and **`word_error_rate(references, transcripts)`**:
   lowercase, strip punctuation with a regular expression, run lesson 14's
   Levenshtein recursion over words, and divide the total edits by the total
   reference words.

## Cost and scaling

The following table shows the cost of each score for utterances of $N$ and
$M$ frames with $D$ coefficients, $R$ raters, $I$ items, and $B$ bootstrap
replicates.

| Score | Time | Notes |
|---|---|---|
| MCD matrix | $O(NMD)$ | One broadcast subtraction |
| DTW path | $O(NM)$ | Sequential; a band cuts it to $O(N w)$ |
| F0 RMSE, speaker cosine | $O(N)$, $O(E)$ | Plus the pitch tracker and the verifier |
| ASR WER | one ASR pass per clip | The recognizer dominates everything |
| Bootstrap MOS | $O(B R I)$ | 2,000 replicates of a $24 \times 30$ table take well under a second |
| A listening test | people and days | The only direct measure of quality |

## Where this breaks

**Reference-based scores punish good variety.** MCD and F0 RMSE compare
against one recording. A system that reads the sentence with different, but
natural, emphasis scores worse than one that averages. These scores suit
vocoders and codecs, which should reproduce the input, better than TTS.

**Metrics can be gamed.** A system tuned on WER can learn to speak slowly
and over-articulate. One tuned on a MOS predictor can learn that predictor's
blind spots. Keep a listening test in the loop.

**Rater quality.** Crowd raters vary, and some don't listen. Include hidden
references and deliberately degraded anchors, and exclude raters who score
them implausibly.

**Too few items.** Thirty sentences can all be easy. The item resampling in
the bootstrap makes that visible as a wide interval.

## The exercise

You implement `mel_cepstral_distortion`, `dtw_path`, `mcd_dtw`, `f0_rmse`,
`speaker_similarity`, `bootstrap_mos`, `normalize_text`, and
`word_error_rate`. The checks reproduce the hand-worked 1.504 dB and 7.91 Hz,
compare your DTW path's cost with brute-force enumeration of every path,
confirm that MCD-DTW is zero for a copy with repeated frames, and check that
your bootstrap interval matches the spread that rater and item effects cause,
far wider than the naive formula.

Press **Run** to synthesize a reference voice from a source and formant
filters, two imitations, and a different speaker. The run prints each one's
MCD-DTW, F0 RMSE, V/UV error, and speaker similarity, draws the MCD matrix
that DTW searches, reports bootstrap MOS intervals from a simulated listening
test, and plays all four voices.
