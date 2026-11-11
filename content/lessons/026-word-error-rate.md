---
title: Word error rate
slug: 026-word-error-rate
part: "Part 3 — Classical recognition"
summary: Score a transcript with word error rate, test whether a gain is real, and see what normalization and unspaced text do to it.
minutes: 40
gpu: false
objectives:
  - Explain why comparing a transcript with its reference position by position overcounts errors.
  - Fill a Levenshtein table, backtrack it, and count substitutions, deletions, and insertions.
  - Compute corpus-level WER and character error rate, and explain why WER can pass 100%.
  - Test whether one system beats another with a paired bootstrap over utterances, and name the MAPSSWE test.
  - Show how normalization changes a score, and use CER or TER where words have no spaces.
prereqs:
  - 021-dynamic-time-warping
  - 025-n-gram-language-models
exercise: 026-word-error-rate
---

# Word error rate

A recognizer hears "set a timer for two minutes" and writes "set the timer four to minutes please". How wrong is that? You can see it is wrong, but to compare two recognizers, or one recognizer before and after a change, you need a number, and every recognizer in this course is judged by the same one: **word error rate** (WER).

The number has to count mistakes in a transcript whose length differs from the reference. In this chapter you build it from a minimum-edit alignment. Then you ask two questions that follow any score: is a gain real, and does the score depend on how the text is written? This chapter is the course's one reference for evaluation. For a classical recognizer's acoustic scores and search graph, see [Hybrid acoustic models](/lessons/024-hybrid-acoustic-models) and [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs).

## What you already know

- DTW fills a table in which each cell extends the cheapest path to one of its three neighbors. See [Dynamic time warping](/lessons/021-dynamic-time-warping).
- A language model picks between transcripts, and a recognizer proposes several. See [n-gram language models](/lessons/025-n-gram-language-models).

## Why position by position fails

The obvious metric compares the two transcripts word by word and counts the places where they differ. It fails as soon as a word is added. Take the reference "i like green tea" and the hypothesis "i like the green tea". The recognizer made one mistake, the extra "the". But the third and fourth words are now shifted by one, so a position-by-position comparison finds "green" against "the", "tea" against "green", and a leftover "tea", and reports three errors for one slip. The right count first pairs up the words that match, wherever they sit, and then counts what is left over.

## Count the edits

Aligning the two transcripts as well as possible, with the fewest changes, gives three kinds of edit. A **substitution** (S) replaces a word, a **deletion** (D) drops a reference word, and an **insertion** (I) adds a word that is not in the reference. Then

$$
\text{WER} = \frac{S + D + I}{N}
$$

where $N$ is the number of reference words. The smallest possible number of edits is the **Levenshtein distance**, and you find it with a dynamic program like DTW's. Let $E(i, j)$ be the distance between the first $i$ reference words $r$ and the first $j$ hypothesis words $h$:

$$
\begin{aligned}
E(i, 0) &= i \\
E(0, j) &= j \\
E(i, j) &= \min\big\{E(i-1, j) + 1,\ E(i, j-1) + 1,\ E(i-1, j-1) + [r_i \ne h_j]\big\}
\end{aligned}
$$

Read it aloud: an empty hypothesis needs $i$ deletions, and an empty reference needs $j$ insertions. Otherwise, take the cheapest of deleting $r_i$, inserting $h_j$, or pairing them, which costs 1 if they differ and 0 if they match. The whole table costs $O(nm)$ for $n$ reference words and $m$ hypothesis words.

### Fill a small table by hand

Take the reference "a b c" and the hypothesis "a c". The first row and column count deletions and insertions. Then each cell follows the recurrence, for example $E(1, 1) = \min\{2, 2, 0 + 0\} = 0$, because "a" matches "a":

| $E(i, j)$ | $j = 0$ | $a$ | $c$ |
|---|---|---|---|
| $i = 0$ | 0 | 1 | 2 |
| $a$ | 1 | 0 | 1 |
| $b$ | 2 | 1 | 1 |
| $c$ | 3 | 2 | 1 |

The bottom-right cell is 1, so the distance is 1: "c" matches "c", and the cheapest way in is the diagonal from $E(2, 1)$.

### Walk back to count S, D, and I

WER also wants the breakdown. Start at $E(n, m)$ and walk back to $(0, 0)$, each time moving to the neighbor that explains the value: a diagonal step is a hit when the words match and a substitution when they differ, a step up is a deletion, and a step left is an insertion. In the small table the walk goes diagonal to $E(2,1)$ (hit: c), up to $E(1,1)$ (delete "b"), and diagonal to $E(0,0)$ (hit: a), so $S = 0$, $D = 1$, $I = 0$, and WER is $1/3$.

Figure 1 shows the full table for the example in the opening. The shaded cells are the walk back from the corner.

```figure
<svg viewBox="0 0 620 329" role="img" aria-label="The edit distance table for a six-word reference against a seven-word hypothesis with the backtrace path shaded">
<text x="178.0" y="50.0" class="fig-small" text-anchor="middle">-</text>
<text x="234.0" y="50.0" class="fig-small" text-anchor="middle">set</text>
<text x="290.0" y="50.0" class="fig-small" text-anchor="middle">the</text>
<text x="346.0" y="50.0" class="fig-small" text-anchor="middle">timer</text>
<text x="402.0" y="50.0" class="fig-small" text-anchor="middle">four</text>
<text x="458.0" y="50.0" class="fig-small" text-anchor="middle">to</text>
<text x="514.0" y="50.0" class="fig-small" text-anchor="middle">minutes</text>
<text x="570.0" y="50.0" class="fig-small" text-anchor="middle">please</text>
<text x="140.0" y="81.0" class="fig-small" text-anchor="end">-</text>
<text x="140.0" y="113.0" class="fig-small" text-anchor="end">set</text>
<text x="140.0" y="145.0" class="fig-small" text-anchor="end">a</text>
<text x="140.0" y="177.0" class="fig-small" text-anchor="end">timer</text>
<text x="140.0" y="209.0" class="fig-small" text-anchor="end">for</text>
<text x="140.0" y="241.0" class="fig-small" text-anchor="end">two</text>
<text x="140.0" y="273.0" class="fig-small" text-anchor="end">minutes</text>
<rect x="150.0" y="60.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="178.0" y="81.0" class="fig-bold fig-small" text-anchor="middle">0</text>
<rect x="206.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="81.0" class="fig-small" text-anchor="middle">1</text>
<rect x="262.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="81.0" class="fig-small" text-anchor="middle">2</text>
<rect x="318.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="81.0" class="fig-small" text-anchor="middle">3</text>
<rect x="374.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="81.0" class="fig-small" text-anchor="middle">4</text>
<rect x="430.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="81.0" class="fig-small" text-anchor="middle">5</text>
<rect x="486.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="81.0" class="fig-small" text-anchor="middle">6</text>
<rect x="542.0" y="60.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="81.0" class="fig-small" text-anchor="middle">7</text>
<rect x="150.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="113.0" class="fig-small" text-anchor="middle">1</text>
<rect x="206.0" y="92.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="234.0" y="113.0" class="fig-bold fig-small" text-anchor="middle">0</text>
<rect x="262.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="113.0" class="fig-small" text-anchor="middle">1</text>
<rect x="318.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="113.0" class="fig-small" text-anchor="middle">2</text>
<rect x="374.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="113.0" class="fig-small" text-anchor="middle">3</text>
<rect x="430.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="113.0" class="fig-small" text-anchor="middle">4</text>
<rect x="486.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="113.0" class="fig-small" text-anchor="middle">5</text>
<rect x="542.0" y="92.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="113.0" class="fig-small" text-anchor="middle">6</text>
<rect x="150.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="145.0" class="fig-small" text-anchor="middle">2</text>
<rect x="206.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="145.0" class="fig-small" text-anchor="middle">1</text>
<rect x="262.0" y="124.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="290.0" y="145.0" class="fig-bold fig-small" text-anchor="middle">1</text>
<rect x="318.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="145.0" class="fig-small" text-anchor="middle">2</text>
<rect x="374.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="145.0" class="fig-small" text-anchor="middle">3</text>
<rect x="430.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="145.0" class="fig-small" text-anchor="middle">4</text>
<rect x="486.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="145.0" class="fig-small" text-anchor="middle">5</text>
<rect x="542.0" y="124.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="145.0" class="fig-small" text-anchor="middle">6</text>
<rect x="150.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="177.0" class="fig-small" text-anchor="middle">3</text>
<rect x="206.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="177.0" class="fig-small" text-anchor="middle">2</text>
<rect x="262.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="177.0" class="fig-small" text-anchor="middle">2</text>
<rect x="318.0" y="156.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="346.0" y="177.0" class="fig-bold fig-small" text-anchor="middle">1</text>
<rect x="374.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="177.0" class="fig-small" text-anchor="middle">2</text>
<rect x="430.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="177.0" class="fig-small" text-anchor="middle">3</text>
<rect x="486.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="177.0" class="fig-small" text-anchor="middle">4</text>
<rect x="542.0" y="156.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="177.0" class="fig-small" text-anchor="middle">5</text>
<rect x="150.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="209.0" class="fig-small" text-anchor="middle">4</text>
<rect x="206.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="209.0" class="fig-small" text-anchor="middle">3</text>
<rect x="262.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="209.0" class="fig-small" text-anchor="middle">3</text>
<rect x="318.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="209.0" class="fig-small" text-anchor="middle">2</text>
<rect x="374.0" y="188.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="402.0" y="209.0" class="fig-bold fig-small" text-anchor="middle">2</text>
<rect x="430.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="209.0" class="fig-small" text-anchor="middle">3</text>
<rect x="486.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="209.0" class="fig-small" text-anchor="middle">4</text>
<rect x="542.0" y="188.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="209.0" class="fig-small" text-anchor="middle">5</text>
<rect x="150.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="241.0" class="fig-small" text-anchor="middle">5</text>
<rect x="206.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="241.0" class="fig-small" text-anchor="middle">4</text>
<rect x="262.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="241.0" class="fig-small" text-anchor="middle">4</text>
<rect x="318.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="241.0" class="fig-small" text-anchor="middle">3</text>
<rect x="374.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="241.0" class="fig-small" text-anchor="middle">3</text>
<rect x="430.0" y="220.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="458.0" y="241.0" class="fig-bold fig-small" text-anchor="middle">3</text>
<rect x="486.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="514.0" y="241.0" class="fig-small" text-anchor="middle">4</text>
<rect x="542.0" y="220.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="570.0" y="241.0" class="fig-small" text-anchor="middle">5</text>
<rect x="150.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="178.0" y="273.0" class="fig-small" text-anchor="middle">6</text>
<rect x="206.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="234.0" y="273.0" class="fig-small" text-anchor="middle">5</text>
<rect x="262.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="290.0" y="273.0" class="fig-small" text-anchor="middle">5</text>
<rect x="318.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="346.0" y="273.0" class="fig-small" text-anchor="middle">4</text>
<rect x="374.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="402.0" y="273.0" class="fig-small" text-anchor="middle">4</text>
<rect x="430.0" y="252.0" width="56.0" height="32.0" class="fill-sunk stroke-grid" />
<text x="458.0" y="273.0" class="fig-small" text-anchor="middle">4</text>
<rect x="486.0" y="252.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="514.0" y="273.0" class="fig-bold fig-small" text-anchor="middle">3</text>
<rect x="542.0" y="252.0" width="56.0" height="32.0" class="fill-soft-b stroke-b" />
<text x="570.0" y="273.0" class="fig-bold fig-small" text-anchor="middle">4</text>
<text x="374.0" y="22.0" class="fig-small" text-anchor="middle">hypothesis</text>
<text x="60.0" y="160.0" class="fig-small" text-anchor="middle">reference</text>
<text x="300.0" y="312.0" class="fig-small" text-anchor="middle">Shaded cells: the backtrace from E(6, 7) = 4 back to E(0, 0)</text>
</svg>
Figure 1. The edit-distance table for "set a timer for two minutes" against "set the timer four to minutes please". Notice that the shaded path runs along the diagonal with one step to the left at the end, so the 4 edits are 3 substitutions (a to the, for to four, two to to) and 1 insertion (please).
```

So the opening example has $S = 3$, $D = 0$, $I = 1$, and WER $= 4/6 = 66.7\%$. In code, one `min` over three neighbors fills each cell, as `E[i, j] = min(E[i-1, j] + 1, E[i, j-1] + 1, E[i-1, j-1] + (ref[i-1] != hyp[j-1]))`.

> [!KEY]
> WER is the Levenshtein distance between the reference and the hypothesis, divided by the number of reference words. The alignment finds the fewest edits, so a single inserted word counts once, not once for every word it shifts.

## Count characters instead

**Character error rate** (CER) is the same computation over characters, spaces included. For "kitten" against "sitting", two substitutions and one insertion make 3 edits over 6 reference characters, so CER is $3/6 = 50\%$. CER also shows how bad a word error is: the homophone "write to mom" against "right two mom" costs 2 of 3 words, a WER of 66.7%, but 5 of 12 characters, a CER of 41.7%.

CER is the standard metric for languages written without spaces between words, such as Chinese, Japanese, and Thai. There `split()` returns a whole sentence as one word, so any mistake makes the WER 100%. The reference `我想订一张机票` ("I want to book a plane ticket") against `我想定一张机票` has one wrong character, so its WER is 100% and its CER is $1/7 = 14.3\%$. A word segmenter would make WER depend on the segmenter, and CER does not. Your `levenshtein_align` accepts any list, so CER is the same function over `list(text)`.

A **token error rate** (TER) is the same again over the tokens a model emits, such as subwords. Report the tokenizer, because a different vocabulary gives a different token count.

## Average over the corpus, not the sentences

To score a test set, add up the edits and the reference words over all sentences, and divide once:

$$
\text{WER}_{\text{corpus}} = \frac{\sum_k (S_k + D_k + I_k)}{\sum_k N_k}
$$

Read it aloud: total errors over total reference words. Averaging per-sentence rates gives a different, worse answer, because it counts a one-word sentence as much as a long one. Figure 2 works an example.

```figure
<svg viewBox="0 0 620 240" role="img" aria-label="Bars comparing two per-sentence word error rates, their mean of 166.7 percent, and the corpus word error rate of 71.4 percent">
<line x1="190.0" y1="30.0" x2="190.0" y2="200.0" class="stroke-muted"/>
<text x="180.0" y="60.0" class="fig-small" text-anchor="end">sentence 1: 2 errors / 6 words</text>
<rect x="190.0" y="42.0" width="36.6" height="26.0" class="fill-a" />
<text x="234.6" y="60.0" class="fig-small" text-anchor="start">33.3%</text>
<text x="180.0" y="100.0" class="fig-small" text-anchor="end">sentence 2: 3 errors / 1 word</text>
<rect x="190.0" y="82.0" width="330.0" height="26.0" class="fill-a" />
<text x="528.0" y="100.0" class="fig-small" text-anchor="start">300.0%</text>
<text x="180.0" y="140.0" class="fig-small" text-anchor="end">mean of the two</text>
<rect x="190.0" y="122.0" width="183.4" height="26.0" class="fill-d" />
<text x="381.4" y="140.0" class="fig-small" text-anchor="start">166.7%</text>
<text x="180.0" y="180.0" class="fig-small" text-anchor="end">corpus WER: 5 errors / 7 words</text>
<rect x="190.0" y="162.0" width="78.5" height="26.0" class="fill-b" />
<text x="276.5" y="180.0" class="fig-small" text-anchor="start">71.4%</text>
<text x="310.0" y="225.0" class="fig-small" text-anchor="middle">Averaging sentence rates lets one short sentence outweigh a long one.</text>
</svg>
Figure 2. Two sentences, one with 2 errors in 6 words and one with 3 insertions against a single word. Notice that the mean of the two sentence rates is 166.7%, while the corpus rate of 5 errors in 7 words is 71.4%.
```

> [!WARNING]
> Corpus WER is total edits over total reference words, not the mean of per-utterance WERs. A one-word reference with three insertions scores 300% on its own, and averaging would let it swamp every long utterance.

That 300% also shows why WER can pass 100%: a recognizer can insert more words than the reference holds.

## Normalize before you score

WER compares strings, so two transcripts that a person reads as identical can score badly. Each pair is a correct transcription in a different style.

| Reference | Hypothesis |
|---|---|
| Set a timer for 10 minutes, please. | set a timer for ten minutes please |
| Call Mom at 6 o'clock. | call mom at six o'clock |
| Thanks, that's all. | thanks that's all |

The references hold 15 words. Scored as written, 10 of them count as errors, a WER of 66.7%. Lowercasing leaves 7 errors, 46.7%, because `minutes,` is not `minutes`. Stripping punctuation as well leaves 2 errors, 13.3%: `10` against `ten`, and `6` against `six`. Writing the number words as digits leaves none. Figure 3 shows the four scores.

```figure
<svg viewBox="0 0 620 230" role="img" aria-label="Bars of corpus word error rate for the same three transcript pairs: raw 66.7%, lowercase 46.7%, + punctuation 13.3%, + numbers (normalize) 0.0%">
<line x1="210.0" y1="20.0" x2="210.0" y2="180.0" class="stroke-muted"/>
<text x="200.0" y="48" class="fig-small" text-anchor="end">raw</text>
<rect x="210.0" y="30" width="340.0" height="26.0" class="fill-d" />
<text x="558.0" y="48" class="fig-small" text-anchor="start">66.7%</text>
<text x="200.0" y="88" class="fig-small" text-anchor="end">lowercase</text>
<rect x="210.0" y="70" width="238.0" height="26.0" class="fill-b" />
<text x="456.0" y="88" class="fig-small" text-anchor="start">46.7%</text>
<text x="200.0" y="128" class="fig-small" text-anchor="end">+ punctuation</text>
<rect x="210.0" y="110" width="68.0" height="26.0" class="fill-b" />
<text x="286.0" y="128" class="fig-small" text-anchor="start">13.3%</text>
<text x="200.0" y="168" class="fig-small" text-anchor="end">+ numbers (normalize)</text>
<text x="218.0" y="168" class="fig-small" text-anchor="start">0.0%</text>
<text x="310.0" y="215.0" class="fig-small" text-anchor="middle">Corpus WER of the same 15 reference words under each treatment.</text>
</svg>
Figure 3. Corpus WER of three correct transcripts under four treatments. Notice that case and punctuation account for most of the drop, and the last 13.3% is the number words.
```

The exercise's `normalize(text)` lowercases, deletes apostrophes, turns other punctuation into spaces, and writes "zero" to "ten" as digits. It is deliberately small. The [Whisper chapter](/lessons/041-whisper) teaches a fuller normalizer and shows the same effect at scale: 99.8% WER on raw LibriSpeech references and 8.3% once both sides are normalized.

> [!KEY]
> A WER describes a recognizer and a normalizer together. Run the same function over the reference and the hypothesis, and report it beside the number.

## Check whether a gain is real

System A scores 35.0% on your test set and system B scores 25.0%. Is B better? A test set is a sample of the speech you could have recorded, so another sample gives other scores. The **bootstrap** estimates how much they would move. Treat your utterances as the whole population, draw as many as you have with replacement, score both systems, and repeat. The spread of the results is the spread a new sample would show.

Two choices matter. Resample **utterances**, not words: words in one utterance share a speaker and a microphone, so resampling words gives intervals that are too narrow. And make it **paired**: use one draw for both systems and take the difference inside each resample, so a hard utterance hurts both and cancels.

Take four utterances with 6, 2, 8, and 4 reference words. System A makes 2, 1, 3, and 1 edits on them, and system B makes 1, 2, 2, and 0. The totals are 20 words, 7 edits for A, and 5 for B: 35.0% against 25.0%, a gap of 10.0 points. Now draw four utterances with replacement, say 4, 2, 4, and 1. That draw has $4 + 2 + 4 + 6 = 16$ words, A makes $1 + 1 + 1 + 2 = 5$ edits (31.25%), and B makes $0 + 2 + 0 + 1 = 3$ edits (18.75%). A minus B is $+12.5$ points, so B wins this resample. Draw 2, 2, 1, and 2 instead: 12 words, 5 edits for A (41.7%), 7 for B (58.3%), and $-16.7$ points, so B loses. Each resample divides summed edits by summed words, never averaging per-utterance rates.

Repeat many times and keep two numbers: the fraction of resamples where B wins, and the 2.5th and 97.5th percentiles of A minus B, a 95% interval. With 10,000 resamples and seed 0, B wins in 73.7% of draws and the interval is $[-16.7, +20.0]$ points. Ties, 21% of the draws, are not wins. The interval contains 0, so four utterances cannot separate the systems. Repeat the same four 100 times, as if 400 utterances showed the same pattern, and B wins every resample with the interval $[+8.6, +11.3]$. Figure 4 compares the two.

```figure
<svg viewBox="0 0 620 220" role="img" aria-label="95 percent intervals of the WER difference, in percentage points, for 4 utterances and for 400 utterances">
<line x1="305.6" y1="30.0" x2="305.6" y2="150.0" class="stroke-ink dashed"/>
<text x="305.6" y="22.0" class="fig-small" text-anchor="middle">no difference</text>
<text x="140.0" y="75" class="fig-small" text-anchor="end">4 utterances</text>
<rect x="175.9" y="65" width="285.2" height="10.0" class="fill-b" />
<circle cx="383.3" cy="70" r="6" class="fill-ink"/>
<text x="469.1" y="75" class="fig-small" text-anchor="start">[-16.7, +20.0]</text>
<text x="140.0" y="125" class="fig-small" text-anchor="end">400 utterances</text>
<rect x="372.1" y="115" width="21.6" height="10.0" class="fill-b" />
<circle cx="383.3" cy="120" r="6" class="fill-ink"/>
<text x="401.7" y="125" class="fig-small" text-anchor="start">[+8.6, +11.3]</text>
<text x="150.0" y="172.0" class="fig-small" text-anchor="middle">-20</text>
<text x="227.8" y="172.0" class="fig-small" text-anchor="middle">-10</text>
<text x="305.6" y="172.0" class="fig-small" text-anchor="middle">0</text>
<text x="383.3" y="172.0" class="fig-small" text-anchor="middle">+10</text>
<text x="461.1" y="172.0" class="fig-small" text-anchor="middle">+20</text>
<text x="365.0" y="200.0" class="fig-small" text-anchor="middle">WER(A) minus WER(B), in percentage points; the dot is the observed gap.</text>
</svg>
Figure 4. The 95% interval of the WER gap on 4 utterances and on 400, with the same 10-point gap in both. Notice that the short test set's interval crosses zero, and the long one's does not.
```

The older standard in speech is NIST's **matched-pair sentence-segment word error** (MAPSSWE) test (Gillick and Cox, 1989). It cuts the transcripts into segments bounded by words both systems got right, and tests with a normal approximation whether the mean difference in errors per segment is zero. The bootstrap needs no such approximation (Bisani and Ney, 2004).

> [!KEY]
> Report the gap with an interval from a paired bootstrap over utterances. If the interval contains 0, the data do not show that one system is better.

> [!RECAP]
> - WER is $(S + D + I) / N$ from a Levenshtein alignment, summed over the whole corpus before dividing.
> - $E(i, j)$ is the minimum of a deletion, an insertion, and a diagonal step that costs 1 only when the words differ.
> - CER is the same over characters, the usable score for text without spaces. TER is the same over a model's tokens.
> - Normalize both sides with one function and report it, because case, punctuation, and numbers move WER by tens of points.
> - A paired bootstrap over utterances reports how often B wins and a 95% interval of the gap.

## Build it step by step

The checks test the functions in this order.

1. **`levenshtein_align(ref, hyp)`.** Fill the `(n + 1, m + 1)` table, then backtrack, preferring the diagonal. Return `(S, D, I, alignment)`, where `alignment` lists `(op, ref_token, hyp_token)`, with `op` one of `"C"`, `"S"`, `"D"`, `"I"`. The usual mistake is calling a matched pair a substitution because you tested the cost before the equality.
2. **`wer(refs, hyps)`.** Align each pair after `split()`, sum the errors and the reference lengths across pairs, then divide once. The usual mistake is averaging per-utterance rates.
3. **`cer(ref, hyp)`.** Align `list(ref)` against `list(hyp)` and divide by `len(ref)`. The usual mistake is splitting on whitespace first.
4. **`normalize(text)`.** Lowercase, delete apostrophes, replace other punctuation with spaces using `re.sub`, then split, map number words to digits, and join. The usual mistake is deleting punctuation instead of replacing it with a space.
5. **`paired_bootstrap(errors_a, errors_b, ref_lengths, n_resamples, seed)`.** Take per-utterance edit counts. Draw `picks = rng.integers(0, n, size=(n_resamples, n))` once and index both error arrays and the lengths with it. Divide the difference of summed errors by the summed lengths, then return the fraction above 0 and `np.percentile(diff, [2.5, 97.5])`. The usual mistakes are averaging per-utterance WERs and resampling words.

## Where this breaks

**Normalization can hide errors you care about.** A normalizer that merges `10` and `100` or drops a negation flatters a recognizer. Check what yours removes.

**WER treats every word alike.** Dropping "not" costs the same as dropping "the". [Sequence-level training](/lessons/035-sequence-level-training) trains a model on WER directly.

**Alignments are not unique.** The distance is unique, but the split into S, D, and I is not: "the cat sat on the mat" against "the cat sit on mat today" has distance 3 as one of each edit or as three substitutions. Compare totals, not breakdowns.

**Utterances are not always independent.** Utterances from one speaker share conditions, so a bootstrap over utterances reports an interval that is too narrow. Resample whole speakers instead (Liu et al., 2023). [Long-form and trustworthy ASR](/lessons/047-long-form-and-trustworthy-asr) puts a bootstrap interval on the WER of each subgroup.

## The exercise

You build `levenshtein_align`, `wer`, `cer`, `normalize`, and `paired_bootstrap`. Press **Run** to see the edit-distance table, the WER and CER of two homophone errors, corpus WER against a mean of sentence rates, the normalization scores, the bootstrap on 4 and on 400 utterances, and a Mandarin pair where WER and CER disagree.

The checks verify the counts on four worked pairs, the alignment itself, and `levenshtein_align` against a brute-force edit distance on 200 random pairs. They check that corpus `wer` is 5/7 on the example above and can pass 100%, and that `cer("kitten", "sitting")` is 0.5. The new checks run `paired_bootstrap` on cases with one possible answer, swap the systems to test its sign, and compare its interval with a reference bootstrap on 40 utterances. They also pin `normalize`, the normalization scores, and the Mandarin CER.

## Further reading

The numbers and claims in this chapter come from these sources:

- Laurent Gillick and Stephen Cox, "Some statistical issues in the comparison of speech recognition algorithms," ICASSP, 1989.
- [Bootstrap estimates for confidence intervals in ASR performance evaluation](http://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/Bisani_BootstrapEstimatesForConfidenceIntervalsInASRPerformanceEvaluation_ICASSP_2004.pdf), Maximilian Bisani and Hermann Ney, ICASSP, 2004.
- [Modeling dependent structure for utterances in ASR evaluation](https://www.isca-archive.org/interspeech_2023/liu23_interspeech.pdf), Liu et al., Interspeech, 2023.
- [Automatic Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/16.pdf), chapter 16 of *Speech and Language Processing*, Daniel Jurafsky and James H. Martin, third edition draft of August 19, 2026. It describes MAPSSWE.
- [Speech Recognition and Understanding (11-751)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, 2022. The 31 August lecture, "Introduction of speech recognition," lists "Evaluation metric."
