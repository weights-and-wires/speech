---
title: Long-form and trustworthy ASR
slug: 046-long-form-and-trustworthy-asr
part: "Part 4 — Neural speech recognition"
summary: Transcribe an hour of audio with a 30-second model, time each word, and report how far to trust the words and the score.
minutes: 45
gpu: false
objectives:
  - Stitch the transcripts of overlapping windows into one text by their longest common run of words.
  - Explain how previous-text conditioning helps context and starts repetition loops.
  - Read word start and end times from cross-attention with dynamic time warping.
  - Compute expected calibration error, and say what it does and does not tell you.
  - Flag repetition loops, and report WER with a bootstrap interval for each speaker group.
prereqs:
  - 021-dynamic-time-warping
  - 026-word-error-rate
  - 041-whisper
  - 045-forced-alignment-and-word-timestamps
exercise: 046-long-form-and-trustworthy-asr
---

# Long-form and trustworthy ASR

You give a recognizer a one-hour meeting, and it returns text. Three questions follow at once. Is the text complete, with no missing seams and no sentence said twice? When was each word said, so a reader can jump to it? And how far can you trust it, for every kind of speaker, not only on average?

The model from [Whisper and large-scale recognition](/lessons/041-whisper) reads exactly 30 seconds at a time. An hour is 120 windows, and every boundary can cut a word in half. A fluent model also writes fluent text for silence, so the transcript can hold sentences nobody said. And one WER for the whole test set hides who the system fails.

This chapter handles those problems with small tools you can check by hand.

## What you already know

- Whisper reads a 30-second window and writes tokens, including timestamp tokens (see [Whisper and large-scale recognition](/lessons/041-whisper)).
- Forced alignment turns a known transcript into word times with a trellis and a frame hop (see [Forced alignment and word timestamps](/lessons/045-forced-alignment-and-word-timestamps)).
- Dynamic time warping finds the cheapest monotone path through a cost matrix (see [Dynamic time warping](/lessons/021-dynamic-time-warping)).
- WER is summed errors over summed reference words, and a bootstrap over utterances puts an interval around it (see [Word error rate](/lessons/026-word-error-rate)).
- A cross-attention weight says how much a decoder token listens to an encoder frame.

## Cut a long recording into windows

The obvious plan is to cut the audio every 30 seconds and transcribe each piece. It fails at the cuts, because a word that straddles a cut is heard in two halves and recognized in neither. Whisper itself avoids fixed cuts. Radford et al. (2022) transcribe long audio sequentially: decode a window, read the last timestamp it predicted, and start the next window there, so a cut falls between segments the model finished.

Other systems run windows in parallel, which is much faster, and keep the cuts away from words in two ways. WhisperX (Bain et al., 2023) runs a voice activity detector and merges speech regions into chunks of at most 30 seconds, so cuts fall in pauses. Other pipelines overlap the windows, as in Figure 1, so every word sits well inside at least one window.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Three 30-second windows over a 70-second recording, with the 5-second overlaps shaded">
<line x1="70" y1="150" x2="550" y2="150" class="stroke-muted"/>
<line x1="70" y1="146" x2="70" y2="154" class="stroke-muted"/>
<text x="70" y="171" text-anchor="middle" class="fig-small">0</text>
<line x1="139" y1="146" x2="139" y2="154" class="stroke-muted"/>
<text x="139" y="171" text-anchor="middle" class="fig-small">10</text>
<line x1="207" y1="146" x2="207" y2="154" class="stroke-muted"/>
<text x="207" y="171" text-anchor="middle" class="fig-small">20</text>
<line x1="276" y1="146" x2="276" y2="154" class="stroke-muted"/>
<text x="276" y="171" text-anchor="middle" class="fig-small">30</text>
<line x1="344" y1="146" x2="344" y2="154" class="stroke-muted"/>
<text x="344" y="171" text-anchor="middle" class="fig-small">40</text>
<line x1="413" y1="146" x2="413" y2="154" class="stroke-muted"/>
<text x="413" y="171" text-anchor="middle" class="fig-small">50</text>
<line x1="481" y1="146" x2="481" y2="154" class="stroke-muted"/>
<text x="481" y="171" text-anchor="middle" class="fig-small">60</text>
<line x1="550" y1="146" x2="550" y2="154" class="stroke-muted"/>
<text x="550" y="171" text-anchor="middle" class="fig-small">70</text>
<rect x="70" y="24" width="206" height="26" rx="4" class="fill-soft-a stroke-a"/>
<text x="78" y="42" class="fig-small">window 1: 0 to 30 s</text>
<rect x="241" y="62" width="206" height="26" rx="4" class="fill-soft-a stroke-a"/>
<text x="249" y="80" class="fig-small">window 2: 25 to 55 s</text>
<rect x="413" y="100" width="137" height="26" rx="4" class="fill-soft-a stroke-a"/>
<text x="421" y="118" class="fig-small">window 3: 50 to 70 s</text>
<rect x="241" y="16" width="34" height="122" class="fill-soft-b" opacity="0.55"/>
<text x="259" y="12" text-anchor="middle" class="fig-small">5 s overlap</text>
<rect x="413" y="16" width="34" height="122" class="fill-soft-b" opacity="0.55"/>
<text x="430" y="12" text-anchor="middle" class="fig-small">5 s overlap</text>
<text x="310" y="188" text-anchor="middle" class="fig-small">time in the recording (seconds)</text>
</svg>
Figure 1. A 70-second recording in 30-second windows that start every 25 seconds. Each seam has 5 seconds of overlap, about 12 words at 2.5 words per second, and a word in the overlap is heard twice.
```

Overlap fixes the cut and creates a duplicate. The text of window 1 ends with the words of the first overlap, and the text of window 2 begins with them again. Plain concatenation says each of those words twice. Dropping a fixed number of words from window 2 fails too, because the recognizer has the least context at both edges, so it spells them differently in each window.

## Stitch the overlap by its longest common run

The edges are unreliable, and the middle of the overlap is not. So match the middle. Look at the last `overlap` words of the text so far (the tail), and the first `overlap` words of the next window (the head). Find the longest run of words that both contain. Keep the first window's text up to the end of that run, and continue with the second window's text after it.

Work one seam by hand with `overlap = 6`. Window 1 says `we hold these truths to be self evident`, and window 2 says `to be self evident that all men`.

- The tail is `these truths to be self evident`. The head is `to be self evident that all`.
- The longest common run is `to be self evident`, four words. It ends at the last word of window 1.
- Keep window 1 whole, because the run ends at its last word. Continue with window 2 after the run: `that all men`.
- The result is `we hold these truths to be self evident that all men`.

If an edge word is garbled, say `evid#` in window 1, the longest common run is the shorter `to be self`, and the garbled word falls after the run, so it is dropped. That is the point of cutting at the run, not at the seam.

A single shared word such as `the` appears everywhere, so a run shorter than `min_run = 2` words does not count, and the windows are joined as they are.

> [!KEY]
> Overlap trades a duplicate for safety at the cut. Stitch by the longest common run, because the edges of each window are the least reliable part, and plain concatenation repeats every overlapped word.

## Previous text helps, and then loops

The second fix for long audio is context. When Whisper decodes window $k$, it can condition on the text of window $k - 1$ as a prompt. This keeps names, spelling, and style consistent across the hour. The cost is a feedback loop: if the previous text contains an error or a repeat, the decoder is more likely to copy it, and the copy becomes the next prompt. A phrase such as `thanks for watching` can then repeat until the window is full.

Whisper's own mitigation, from Radford et al. (2022), is to retry. The decoder first runs at temperature 0. It tries again at a higher temperature, up to 1.0 in steps of 0.2, when the result looks wrong: its average log-probability is below $-1$, or its gzip compression ratio is above 2.4. Repeated text compresses well, so a high ratio means a loop. The same paper marks a window as silence when the no-speech probability is above 0.6 and the average log-probability is below $-1$. Setting `condition_on_previous_text` to false removes the feedback at the price of context.

## Word times from cross-attention

Whisper's encoder emits one frame per 20 ms, so 30 seconds is 1,500 frames. When the decoder writes a token, its cross-attention weights say which frames it listened to. Stack those weights for all tokens and you get a matrix `attn` of shape (tokens, frames). The tokens come in order and the frames come in order, so a good alignment is a monotone path through this matrix. That is a job for the DTW of chapter 021, as in the `openai/whisper` implementation.

Raw attention fails as a cost, and it is worth seeing why. Every weight is positive, so with cost $-\text{attn}$ a longer path collects more weight and wins, even through the wrong cells. Take three tokens over six frames:

| | f0 | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|---|
| token 0 | 0.8 | 0.7 | 0.1 | 0.1 | 0.1 | 0.1 |
| token 1 | 0.1 | 0.2 | 0.8 | 0.7 | 0.2 | 0.1 |
| token 2 | 0.1 | 0.1 | 0.1 | 0.2 | 0.7 | 0.8 |

On the raw cost, the best path steps from token 0 at frame 1 to token 1 at frame 1, where the weight is only 0.2, and then moves right. It picks up an extra cell, and token 1 starts at frame 1 instead of frame 2. The fix is to normalize each frame across the tokens to zero mean and unit standard deviation, as the `openai/whisper` implementation does (it also filters and averages several heads). A weak cell has a negative score, so a detour costs something. For frame 0 the values become 1.41, $-0.71$, $-0.71$. The path on the normalized matrix is $(0,0), (0,1), (1,2), (1,3), (2,4), (2,5)$.

A token's **jump time** is the first frame at which the path is in its row: frames 0, 2, and 4. A word starts at the jump time of its first token, and it ends where the next word starts. The last word ends at the end of the audio. At 20 ms per frame, the three words run $0$ to $0.04$ s, $0.04$ to $0.08$ s, and $0.08$ to $0.12$ s.

Unlike [forced alignment](/lessons/045-forced-alignment-and-word-timestamps), this needs no second model, but it follows where the decoder listened, which is not always where the sound is. WhisperX instead runs forced alignment with a separate phoneme model (Bain et al., 2023).

## Confidence, and whether to believe it

A word's confidence usually comes from the token probabilities the decoder already computed: the product or the minimum over its tokens. Does 0.9 mean right nine times in ten? Group the words by confidence into $M$ equal bins, and compare the accuracy $\text{acc}(B_m)$ with the mean confidence $\text{conf}(B_m)$ in each bin. Guo et al. (2017) define the **expected calibration error** as

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n}\,\bigl|\text{acc}(B_m) - \text{conf}(B_m)\bigr|
$$

Read it aloud: for each bin, take the gap between how right the words were and how right they claimed to be, weight it by the share of words in the bin, and add up. In numpy, `np.searchsorted` assigns each word to a bin, and a loop over bins sums the gaps.

Work it with $M = 2$ and four words. Confidences $0.9, 0.8, 0.7, 0.2$ and correctness $1, 0, 1, 0$. Bin $(0.5, 1]$ holds three words with mean confidence 0.8 and accuracy $2/3$, a gap of $0.133$. Bin $(0, 0.5]$ holds one word, confidence 0.2 and accuracy 0, a gap of 0.2. So $\text{ECE} = \tfrac{3}{4}(0.133) + \tfrac{1}{4}(0.2) = 0.15$. Averaging the two gaps without weights gives 0.167, which lets one rare word count as much as three.

```figure
<svg viewBox="0 0 600 320" role="img" aria-label="A reliability diagram: accuracy in each confidence bin sits below the diagonal">
<line x1="70" y1="280" x2="330" y2="280" class="stroke-grid"/>
<text x="62" y="284" text-anchor="end" class="fig-small">0.0</text>
<text x="70" y="298" text-anchor="middle" class="fig-small">0.0</text>
<line x1="70" y1="150" x2="330" y2="150" class="stroke-grid"/>
<text x="62" y="154" text-anchor="end" class="fig-small">0.5</text>
<text x="200" y="298" text-anchor="middle" class="fig-small">0.5</text>
<line x1="70" y1="20" x2="330" y2="20" class="stroke-grid"/>
<text x="62" y="24" text-anchor="end" class="fig-small">1.0</text>
<text x="330" y="298" text-anchor="middle" class="fig-small">1.0</text>
<rect x="124.0" y="280.0" width="22.0" height="0.0" class="fill-soft-b stroke-b"/>
<rect x="150.0" y="251.1" width="22.0" height="28.9" class="fill-soft-b stroke-b"/>
<rect x="176.0" y="230.6" width="22.0" height="49.4" class="fill-soft-b stroke-b"/>
<rect x="202.0" y="186.6" width="22.0" height="93.4" class="fill-soft-b stroke-b"/>
<rect x="228.0" y="166.6" width="22.0" height="113.4" class="fill-soft-b stroke-b"/>
<rect x="254.0" y="137.3" width="22.0" height="142.7" class="fill-soft-b stroke-b"/>
<rect x="280.0" y="94.0" width="22.0" height="186.0" class="fill-soft-b stroke-b"/>
<rect x="306.0" y="50.0" width="22.0" height="230.0" class="fill-soft-b stroke-b"/>
<line x1="70" y1="280" x2="330" y2="20" class="stroke-ink dashed"/>
<text x="200.0" y="320" text-anchor="middle" class="fig-small">confidence (bin of width 0.1)</text>
<text x="14" y="150.0" class="fig-small" transform="rotate(-90 14 150.0)" text-anchor="middle">accuracy</text>
<text x="370" y="50" class="fig-bold">What the bars show</text>
<text x="370" y="72" class="fig-small">ECE = 0.156</text>
<text x="370" y="94" class="fig-small">fullest bin: 597 of 2000 words</text>
<text x="370" y="116" class="fig-small">at confidence 0.85, accuracy 0.72</text>
<text x="370" y="150" class="fig-small">dashed line: perfect calibration</text>
</svg>
Figure 2. A reliability diagram for 2,000 synthetic words from an overconfident recognizer. Every bar sits below the dashed diagonal, so accuracy falls short of confidence in each bin, and the weighted gap is an ECE of 0.156.
```

Figure 2 is the usual picture: bars for the accuracy in each bin against the diagonal. The recognizer here says 0.94 about words it gets right 88% of the time. Temperature scaling, which divides the logits by one learned number before the softmax, often shrinks this gap (Guo et al., 2017).

> [!WARNING]
> A low ECE does not make a recognizer accurate. A model that says 0.63 about everything, with 63% of its words right, has an ECE near 0 and cannot tell its good words from its bad ones. Calibration and ranking are separate properties.

## Hallucination and loops

A recognizer is trained to always produce text, so silence, music, and noise can get a fluent transcript. Two cheap detectors help. The first is the compression ratio above. The second counts **loop coverage**: the share of words inside a block of 1 to 6 words that repeats back to back at least three times.

Take `the meeting starts at noon see you see you see you`, 11 words. The block `see you` repeats three times and covers 6 words, so the score is $6/11 = 0.55$. In `she said that that was what she had had in mind`, `that that` and `had had` each repeat only twice, and the score is 0. The threshold of three repeats is what spares real stutters and emphasis.

## WER with error bars, for each group

One corpus WER averages over speakers, so a system that fails one group can still look good. Koenecke et al. (2020) measured five commercial systems and found an average WER of 0.35 for Black speakers and 0.19 for white speakers. A single number would have hidden it.

Report WER per subgroup, such as accent or noise level, and give each an interval. Reuse the bootstrap from [Word error rate](/lessons/026-word-error-rate), but resample within one group. For a group of $n_g$ utterances, draw $n_g$ utterances with replacement, recompute summed errors over summed reference words, repeat, and take the 2.5th and 97.5th percentiles. Two rules matter. Pool the errors and words, not the per-utterance rates, or a two-word utterance counts as much as a thirty-word one. And expect a wide interval for a small group, which is the honest answer: with ten utterances, you cannot yet say.

> [!RECAP]
> - Overlap moves a window cut away from words, and stitching by the longest common run removes the duplicates.
> - Conditioning on previous text keeps context and can start a loop, so detect loops and retry.
> - Cross-attention plus DTW gives word times, after each frame is normalized across tokens.
> - ECE weights the accuracy-versus-confidence gap of each bin by its share of the words.
> - Per-group WER with a bootstrap interval shows who the system fails and how sure you are.

## Build it step by step

The checks test the functions in this order. The starter gives you `dtw_path`.

1. **`stitch_chunks(chunk_outputs, overlap, min_run=2)`.** Takes word lists and returns one list. Cut the tail and head, find the longest common run with two loops, and splice at its ends. Common mistake: concatenating, or cutting at the seam instead of at the run.
2. **`word_timestamps(attn, word_token_spans, hop)`.** Normalize each column with `mean(axis=0)` and `std(axis=0)`, call `dtw_path(-z)`, record the first column of each row, and build `(n_words, 2)` seconds. Common mistake: skipping the normalization.
3. **`ece(conf, correct, n_bins=10)`.** Assign bins with `np.searchsorted` on the interior edges, then sum the weighted gaps. Common mistake: averaging the bin gaps without the weights.
4. **`repetition_score(words, min_repeats=3, max_block=6)`.** For each block length and start, count back-to-back repeats and mark the words. Return the share marked. Common mistake: counting `that that` as a loop.
5. **`subgroup_wer_with_ci(errors, ref_lens, groups, n_boot, seed, alpha)`.** For each group in sorted order, resample indices from one shared generator and take percentiles of the pooled WER. Common mistake: averaging per-utterance rates.

## Where this breaks

**Overlap that is too short.** With no common run of `min_run` words, the stitcher appends the window, and a duplicate or a gap can follow. Raise the overlap.

**Attention is not alignment.** Treat attention times as good to a few frames, and check them against a forced aligner when exact times matter.

**Calibration shifts with the data.** An ECE measured on read speech does not carry over to noisy phone calls. Measure it on data like the deployment, and per group.

## The exercise

You write `stitch_chunks`, `word_timestamps`, `ece`, `repetition_score`, and `subgroup_wer_with_ci`. The checks stitch the worked example and a planted transcript with garbled edges and repeated words, recover planted token boundaries from noisy attention, reproduce the ECE of 0.15, flag a planted loop and spare clean text, and match a reference bootstrap per group. The **Run** stitches the worked seam, times three words, and prints the ECE of an overconfident recognizer and the per-group WER.

## Further reading

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," 2022. Section 4.5 covers long-form transcription, the temperature fallback, and its thresholds (https://arxiv.org/abs/2212.04356).
- Bain et al., "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio," Interspeech 2023 (https://arxiv.org/abs/2303.00747).
- Koenecke et al., "Racial disparities in automated speech recognition," PNAS, 2020 (https://doi.org/10.1073/pnas.1915768117).
- Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 (https://arxiv.org/abs/1706.04599).
