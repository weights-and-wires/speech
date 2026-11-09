---
title: Word error rate
slug: 026-word-error-rate
part: "Part 3 — Classical recognition"
summary: Score a transcript with word error rate from a Levenshtein alignment, and see how HMM-GMM systems composed four transducers into one search graph.
minutes: 35
gpu: false
objectives:
  - Explain why comparing a transcript with its reference position by position overcounts errors.
  - Fill a Levenshtein table, backtrack it, and count substitutions, deletions, and insertions.
  - Compute corpus-level WER and character error rate, and explain why WER can pass 100%.
  - Describe how HMM-GMM systems composed four transducers, H, C, L, and G.
prereqs:
  - 021-dynamic-time-warping
  - 025-n-gram-language-models
exercise: 026-word-error-rate
---

# Word error rate

A recognizer hears "set a timer for two minutes" and writes "set the timer four to minutes please". How wrong is that? You can see it is wrong, but to compare two recognizers, or one recognizer before and after a change, you need a number, and every recognizer in this course is judged by the same one: **word error rate** (WER).

The number has to count mistakes in a transcript whose length differs from the reference. In this chapter you build it from a minimum-edit alignment. The chapter ends with a short look at how classical recognizers glued an acoustic model, a dictionary, and a language model into a single search.

## What you already know

- DTW fills a table in which each cell extends the cheapest path to one of its three neighbors. See [Dynamic time warping](/lessons/021-dynamic-time-warping).
- A language model picks between transcripts, and a recognizer proposes several. See [n-gram language models](/lessons/025-n-gram-language-models).
- A pronunciation dictionary maps each word to a sequence of phones.

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

Read it aloud: an empty hypothesis needs $i$ deletions, and an empty reference needs $j$ insertions. Otherwise, take the cheapest of deleting $r_i$, inserting $h_j$, or pairing them, which costs 1 if they differ and 0 if they match. Because each cell uses only three earlier cells, the whole table costs $O(nm)$ for a reference of $n$ words and a hypothesis of $m$.

### Fill a small table by hand

Take the reference "a b c" and the hypothesis "a c". The first row and column count deletions and insertions. Then each cell follows the recurrence, for example $E(1, 1) = \min\{2, 2, 0 + 0\} = 0$, because "a" matches "a":

| $E(i, j)$ | $j = 0$ | $a$ | $c$ |
|---|---|---|---|
| $i = 0$ | 0 | 1 | 2 |
| $a$ | 1 | 0 | 1 |
| $b$ | 2 | 1 | 1 |
| $c$ | 3 | 2 | 1 |

The bottom-right cell is 1, so the distance is 1. Here $E(3, 2) = \min\{E(2,2) + 1, E(3,1) + 1, E(2,1) + 0\} = \min\{2, 3, 1\} = 1$, because "c" matches "c" and the cheapest way to reach the cell is the diagonal from $E(2, 1)$.

### Walk back to count S, D, and I

The table holds a total, and WER also wants the breakdown. Start at $E(n, m)$ and walk back to $(0, 0)$, each time moving to the neighbor that explains the value: a diagonal step is a hit when the words match and a substitution when they differ, a step up is a deletion, and a step left is an insertion. In the small table the walk goes diagonal from $E(3,2)$ to $E(2,1)$ (hit: c with c), up to $E(1,1)$ (delete "b"), and diagonal to $E(0,0)$ (hit: a with a), so $S = 0$, $D = 1$, $I = 0$, and WER is $1/3$.

The distance is unique, but the breakdown is not. "the cat sat on the mat" against "the cat sit on mat today" has distance 3, and it aligns as one of each edit or as three substitutions. The exercise's backtrace tries the diagonal first, so it finds the three substitutions.

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

So the opening example has $S = 3$, $D = 0$, $I = 1$, and WER $= 4/6 = 66.7\%$. In code, one `min` over three neighbors fills each cell:

```python
E[i, j] = min(E[i-1, j] + 1, E[i, j-1] + 1, E[i-1, j-1] + (ref[i-1] != hyp[j-1]))
```

> [!KEY]
> WER is the Levenshtein distance between the reference and the hypothesis, divided by the number of reference words. The alignment finds the fewest edits, so a single inserted word counts once, not once for every word it shifts.

## Count characters instead

**Character error rate** (CER) is the same computation over characters, spaces included. For "kitten" against "sitting", two substitutions and one insertion make 3 edits over 6 reference characters, so CER is $3/6 = 50\%$. CER is the standard metric for languages written without spaces between words, such as Chinese and Japanese. It also shows how bad a word error is. The homophone "write to mom" against "right two mom" costs 2 of 3 words, a WER of 66.7%, but 5 of 12 characters, a CER of 41.7%.

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

That 300% also shows why WER can pass 100%: nothing stops a recognizer from inserting more words than the reference holds. WER is a count of edits relative to the reference length, not a percentage of anything.

## How HMM-GMM systems put it together

A large-vocabulary recognizer does not train one HMM per word. It trains HMMs for sounds and composes them with everything else. HMM-GMM systems expressed each knowledge source as a **weighted finite-state transducer** (WFST), a graph whose arcs read one symbol, write another, and carry a weight. The weights are negative log probabilities, so a path's weight is the sum along it, and the best path is the one with the smallest sum. Figure 3 shows the four transducers in a row, and the following list says what each one holds:

- **H** reads HMM states and writes context-dependent phones. It holds the HMM topology of each phone.
- **C** reads context-dependent phones and writes phones. It says which triphone each phone in context is.
- **L** reads phones and writes words. It is the pronunciation dictionary.
- **G** reads words and writes words. It is the n-gram language model.

```figure
<svg viewBox="0 0 600 205" role="img" aria-label="Four boxes in a row, H, C, L, and G, each turning one kind of symbol into the next, from HMM states to words">
<defs><marker id="c024-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="80.0" y="70.0" width="80.0" height="60.0" class="fill-soft-a stroke-a" />
<text x="120.0" y="98.0" class="fig-bold" text-anchor="middle">H</text>
<text x="120.0" y="118.0" class="fig-small" text-anchor="middle">topology</text>
<line x1="45.0" y1="100.0" x2="78.0" y2="100.0" class="stroke-ink" marker-end="url(#c024-arrow)"/>
<text x="60.0" y="150.0" class="fig-small" text-anchor="middle">HMM states</text>
<rect x="210.0" y="70.0" width="80.0" height="60.0" class="fill-soft-a stroke-a" />
<text x="250.0" y="98.0" class="fig-bold" text-anchor="middle">C</text>
<text x="250.0" y="118.0" class="fig-small" text-anchor="middle">context</text>
<line x1="175.0" y1="100.0" x2="208.0" y2="100.0" class="stroke-ink" marker-end="url(#c024-arrow)"/>
<text x="190.0" y="150.0" class="fig-small" text-anchor="middle">triphones</text>
<rect x="340.0" y="70.0" width="80.0" height="60.0" class="fill-soft-a stroke-a" />
<text x="380.0" y="98.0" class="fig-bold" text-anchor="middle">L</text>
<text x="380.0" y="118.0" class="fig-small" text-anchor="middle">dictionary</text>
<line x1="305.0" y1="100.0" x2="338.0" y2="100.0" class="stroke-ink" marker-end="url(#c024-arrow)"/>
<text x="320.0" y="150.0" class="fig-small" text-anchor="middle">phones</text>
<rect x="470.0" y="70.0" width="80.0" height="60.0" class="fill-soft-a stroke-a" />
<text x="510.0" y="98.0" class="fig-bold" text-anchor="middle">G</text>
<text x="510.0" y="118.0" class="fig-small" text-anchor="middle">n-gram LM</text>
<line x1="435.0" y1="100.0" x2="468.0" y2="100.0" class="stroke-ink" marker-end="url(#c024-arrow)"/>
<text x="450.0" y="150.0" class="fig-small" text-anchor="middle">words</text>
<line x1="552.0" y1="100.0" x2="580.0" y2="100.0" class="stroke-ink" marker-end="url(#c024-arrow)"/>
<text x="562.0" y="150.0" class="fig-small" text-anchor="middle">words</text>
<text x="300.0" y="32.0" class="fig-bold fig-small" text-anchor="middle">Composed, the four become one graph, HCLG: HMM states in, words out</text>
<text x="300.0" y="190.0" class="fig-small" text-anchor="middle">Each box reads the symbols on its left and writes the symbols on its right.</text>
</svg>
Figure 3. The four transducers that classical recognizers composed. Notice that each one reads the kind of symbol the previous one writes, so the chain turns HMM states into words.
```

A **triphone** is a phone in the context of its neighbors: `k-ae+t` is "ae" after "k" and before "t". With about 40 phones there are up to 64,000 triphones, most never seen in training. Decision trees that ask questions such as "is the left neighbor a nasal?" cluster their states into a few thousand tied states, called **senones**, each with its own GMM.

Composing the four (Mohri, Pereira, and Riley, 2002) gives one graph, HCLG, that reads HMM states and writes words. Kaldi (Povey et al., 2011) builds it with composition, determinization, and minimization at each stage. Decoding is then a Viterbi beam search through HCLG, with the GMMs supplying a score for every state at every frame. For the composition, the two semirings, and a CTC topology on the same kind of graph, see [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs).

> [!RECAP]
> - WER is $(S + D + I) / N$ from a Levenshtein alignment, summed over the whole corpus before dividing.
> - $E(i, j)$ is the minimum of a deletion, an insertion, and a diagonal step that costs 1 only when the words differ.
> - The backtrace prefers the diagonal and returns the counts and the alignment.
> - CER is the same over characters, spaces included.

## Build it step by step

The checks test the functions in this order.

1. **`levenshtein_align(ref, hyp)`.** Fill the `(n + 1, m + 1)` table, then backtrack, preferring the diagonal. Return `(S, D, I, alignment)`, where `alignment` lists `(op, ref_token, hyp_token)` with `op` one of `"C"`, `"S"`, `"D"`, `"I"`. The usual mistake is calling a matched pair a substitution because you tested the cost before the equality.
2. **`wer(refs, hyps)`.** Align each pair after `split()`, sum the errors and the reference lengths across pairs, then divide once. The usual mistake is averaging per-utterance rates.
3. **`cer(ref, hyp)`.** Align `list(ref)` against `list(hyp)` and divide by `len(ref)`.

## Where this breaks

**Text normalization decides WER.** "Mr." against "mister", "10" against "ten", and "won't" against "will not" all count as errors unless both sides are normalized the same way. Whisper's English normalizer lowercases, removes bracketed text and fillers such as "um", expands contractions, and writes numbers as digits. Always report which normalizer you used.

**WER treats every word alike.** Dropping "not" reverses a sentence but costs the same as dropping "the".

**Alignments are not unique.** Two alignments with the same distance can split it into different counts of S, D, and I, so compare the total, not the breakdown, across systems that break ties differently.

## The exercise

You build `levenshtein_align`, `wer`, and `cer`. Press **Run** to see the edit-distance table for the opening example, the WER and CER of two homophone errors, and the gap between corpus WER and a mean of sentence rates.

The checks verify the counts on four worked pairs, that the alignment spells out both sequences in order, and that `levenshtein_align` matches a brute-force edit distance on 200 random pairs. They check that corpus `wer` is 5/7 on the example above, that it can pass 100%, and that `cer("kitten", "sitting")` is 0.5.
