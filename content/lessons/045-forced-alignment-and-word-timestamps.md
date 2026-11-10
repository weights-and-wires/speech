---
title: Forced alignment and word timestamps
slug: 045-forced-alignment-and-word-timestamps
part: "Part 4 — Neural speech recognition"
summary: Find the best CTC path that spells a known transcript, then turn its frames into token and word start and end times in seconds.
minutes: 45
gpu: false
objectives:
  - Fill a CTC Viterbi trellis over the blank-extended transcript, and say when a state may skip the blank.
  - Explain why two equal tokens in a row need a blank between them.
  - Trace the best path back through the backpointers.
  - Turn the frame path into per-token and per-word start and end times, given the frame hop.
prereqs:
  - 022-hidden-markov-models
  - 029-ctc-many-paths-one-label
  - 030-ctc-forward-backward
exercise: 045-forced-alignment-and-word-timestamps
---

# Forced alignment and word timestamps

A subtitle tool already has the words. A dataset builder already has the transcript of a long recording. Both need one thing the words do not carry: when each one was said, to the hundredth of a second, so that a caption appears with the speech or a clip is cut at a word boundary.

The obvious attempt is to split the audio evenly among the words. But words differ in length, so the split drifts off after the first long word, and every later time is wrong. A second attempt is to decode the audio and read times off the result. But a recognizer may spell the words differently than your transcript does, and then there is nothing to match.

This chapter fixes both. You keep the transcript fixed, and you ask the acoustic model for the single most likely path through its CTC lattice that spells exactly that transcript. Each frame on the path belongs to one token, and a frame has a known duration, so the path gives times. This is **forced alignment**.

## What you already know

- A CTC model gives one log-probability per symbol at every frame, and a blank symbol means "no token here" (see [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label)).
- The CTC forward recursion adds up all the paths through the blank-extended sequence (see [CTC forward and backward](/lessons/030-ctc-forward-backward)).
- Viterbi finds the best path through a trellis by replacing a sum with a max and keeping backpointers (see [Hidden Markov models](/lessons/022-hidden-markov-models)).
- A path collapses to a transcript by merging repeated symbols and then deleting blanks.
- A frame is a fixed slice of time. The Run uses frames of 20 ms, so 50 frames make one second.

## Pin a transcript to the frames

You have the emissions, the log-probabilities $\log y^t_k$ of symbol $k$ at frame $t$, and a transcript $l$ of $L$ tokens. Build the blank-extended sequence $l' = (\varnothing, l_1, \varnothing, l_2, \ldots, l_L, \varnothing)$ of length $S = 2L + 1$, as for the CTC loss. A path through the lattice picks one state $s$ of this sequence at every frame. Odd states are tokens, and even states are blanks.

Run the forward recursion with a maximum in place of the sum. Let $\delta_t(s)$ be the best score of any path that is in state $s$ at frame $t$:

$$
\delta_t(s) = \log y^t_{l'_s} + \max\left(\delta_{t-1}(s),\ \delta_{t-1}(s-1),\ \delta_{t-1}(s-2)\right)
$$

Read it aloud: the best path into state $s$ at frame $t$ scores this frame's log-probability of that state's symbol, plus the best of three ways to arrive. It can **stay** in the same state, **step** forward one state, or **skip** forward two. Start in state 0 or 1, end in state $S - 1$ or $S - 2$, and keep a **backpointer** at every cell, which records which of the three won. Then walk back from the better end. In numpy, `delta` is a `(T, S)` table that starts at `-np.inf`, and the backpointer table is an integer array of the same shape.

## Blank handling: when a state may skip

The skip is the only move that needs a rule. A path may skip from state $s - 2$ to state $s$ only when state $s$ is a token, not a blank, and $l'_s$ differs from $l'_{s-2}$. That rule has two parts, and each has a reason.

**Never skip into a blank.** A skip jumps over the state in between. Jumping over a blank is harmless, but jumping over a token would delete it from the transcript. From state $s - 2$ the state between is $s - 1$. When $s$ is a blank, $s - 1$ is a token, so the skip would drop that token. When $s$ is a token, $s - 1$ is a blank, which a path may drop.

**Never skip between equal tokens.** The collapse rule merges repeated symbols before it deletes blanks. So the path `a a` collapses to `a`, and only `a _ a` collapses to `a a`. For the transcript `a a`, the states are blank, `a`, blank, `a`, blank. A skip from the first `a` straight to the second would put two equal tokens next to each other, and the path would spell a single `a`. The blank between them is not optional, which is also why a repeated letter needs one extra frame.

For the transcript `a b`, the rule lets a path go from `a` straight to `b`, because `a b` collapses to `a b` without a blank. A short transcript can then fit in few frames.

### Work a four-frame trellis by hand

Align the transcript `a b` over four frames. The model's probabilities of (blank, `a`, `b`) at each frame are the following. This example works in probabilities, so a path's score is a product, and the recursion multiplies where the formula adds.

| Frame | blank | `a` | `b` |
|---|---|---|---|
| 0 | 0.6 | 0.3 | 0.1 |
| 1 | 0.1 | 0.7 | 0.2 |
| 2 | 0.1 | 0.2 | 0.7 |
| 3 | 0.5 | 0.1 | 0.4 |

The states are 0 blank, 1 `a`, 2 blank, 3 `b`, and 4 blank. At frame 0 only states 0 and 1 are open, with scores 0.6 and 0.3. Every other cell is the best of its allowed arrivals, times that frame's probability of its symbol. Three cells show all three moves:

- Frame 1, state 1 (`a`): stay from 0.3, or step from state 0 at 0.6. The step wins, so the cell is $0.6 \times 0.7 = 0.42$.
- Frame 2, state 3 (`b`): stay gives 0.06, step from state 2 gives 0.03, and the skip from state 1 gives 0.42. The skip is legal because `b` is a token and differs from `a`. The cell is $0.42 \times 0.7 = 0.294$.
- Frame 3, state 4 (blank): stay gives 0.006, and step from state 3 gives 0.294, so the cell is $0.294 \times 0.5 = 0.147$.

```figure
<svg viewBox="0 0 600 340" role="img" aria-label="A Viterbi trellis for the transcript a b over four frames, with the best path blank, a, b, blank highlighted">
<defs><marker id="c041-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="20" y="85" class="fig-small">state 0: blank</text>
<text x="20" y="137" class="fig-small">state 1: a</text>
<text x="20" y="189" class="fig-small">state 2: blank</text>
<text x="20" y="241" class="fig-small">state 3: b</text>
<text x="20" y="293" class="fig-small">state 4: blank</text>
<text x="190" y="46" text-anchor="middle" class="fig-small fig-bold">frame 0</text>
<text x="290" y="46" text-anchor="middle" class="fig-small fig-bold">frame 1</text>
<text x="390" y="46" text-anchor="middle" class="fig-small fig-bold">frame 2</text>
<text x="490" y="46" text-anchor="middle" class="fig-small fig-bold">frame 3</text>
<rect x="150" y="60" width="80" height="40" rx="6" class="fill-soft-b stroke-b"/>
<text x="190" y="85" text-anchor="middle" class="fig-bold">0.6</text>
<rect x="150" y="112" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="190" y="137" text-anchor="middle" class="fig-small">0.3</text>
<rect x="150" y="164" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="190" y="189" text-anchor="middle" class="fig-small">0</text>
<rect x="150" y="216" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="190" y="241" text-anchor="middle" class="fig-small">0</text>
<rect x="150" y="268" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="190" y="293" text-anchor="middle" class="fig-small">0</text>
<rect x="250" y="60" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="290" y="85" text-anchor="middle" class="fig-small">0.06</text>
<rect x="250" y="112" width="80" height="40" rx="6" class="fill-soft-b stroke-b"/>
<text x="290" y="137" text-anchor="middle" class="fig-bold">0.42</text>
<rect x="250" y="164" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="290" y="189" text-anchor="middle" class="fig-small">0.03</text>
<rect x="250" y="216" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="290" y="241" text-anchor="middle" class="fig-small">0.06</text>
<rect x="250" y="268" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="290" y="293" text-anchor="middle" class="fig-small">0</text>
<rect x="350" y="60" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="390" y="85" text-anchor="middle" class="fig-small">0.006</text>
<rect x="350" y="112" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="390" y="137" text-anchor="middle" class="fig-small">0.084</text>
<rect x="350" y="164" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="390" y="189" text-anchor="middle" class="fig-small">0.042</text>
<rect x="350" y="216" width="80" height="40" rx="6" class="fill-soft-b stroke-b"/>
<text x="390" y="241" text-anchor="middle" class="fig-bold">0.294</text>
<rect x="350" y="268" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="390" y="293" text-anchor="middle" class="fig-small">0.006</text>
<rect x="450" y="60" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="490" y="85" text-anchor="middle" class="fig-small">0.003</text>
<rect x="450" y="112" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="490" y="137" text-anchor="middle" class="fig-small">0.0084</text>
<rect x="450" y="164" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="490" y="189" text-anchor="middle" class="fig-small">0.042</text>
<rect x="450" y="216" width="80" height="40" rx="6" class="fill-sunk stroke-muted"/>
<text x="490" y="241" text-anchor="middle" class="fig-small">0.1176</text>
<rect x="450" y="268" width="80" height="40" rx="6" class="fill-soft-b stroke-b"/>
<text x="490" y="293" text-anchor="middle" class="fig-bold">0.147</text>
<line x1="230" y1="80" x2="248" y2="132" class="stroke-b" marker-end="url(#c041-arrow)"/>
<line x1="330" y1="132" x2="348" y2="236" class="stroke-b" marker-end="url(#c041-arrow)"/>
<line x1="430" y1="236" x2="448" y2="288" class="stroke-b" marker-end="url(#c041-arrow)"/>
<text x="300" y="338" text-anchor="middle" class="fig-small">frame (1 frame = 20 ms); each cell is the best probability of any path that ends there</text>
</svg>
Figure 1. The Viterbi table for the transcript a b over four frames, with the probability of the best path into each cell. The highlighted path takes the skip from state 1 to state 3 at frame 2, so a and b sit on frames 1 and 2 with no blank between them. State 4 at frame 3 wins at 0.147.
```

Figure 1 holds every cell. The better end is state 4 at 0.147, because state 3 ends at 0.1176. Walk back through the backpointers: state 4 came from 3, state 3 came from 1 by the skip, and state 1 came from 0. The path is states $(0, 1, 3, 4)$, which reads blank, `a`, `b`, blank, with probability $0.6 \times 0.7 \times 0.7 \times 0.5 = 0.147$.

Next, feed the same four frames to the transcript `a a`, with the column for `a` used twice. The skip from state 1 to state 3 is blocked, so frame 1 has score 0 in state 3, where a skip would have given $0.3 \times 0.7 = 0.21$. The best legal path must pass the blank state: states $(0, 1, 2, 3)$ with probability $0.6 \times 0.7 \times 0.1 \times 0.1 = 0.0042$. An implementation that allows the illegal skip finds states $(0, 1, 3, 4)$ at 0.042, ten times higher, and that path spells only one `a`.

> [!KEY]
> Forced alignment is Viterbi on the CTC lattice of one known transcript. A state may stay, step, or skip, and the skip is legal only onto a token that differs from the one it jumps over.

## From a path to times

The path has one state per frame, so reading times is a bookkeeping step. State $2k + 1$ is token $k$, counting from 0. The frames whose state is $2k + 1$ are the **span** of token $k$. Store a span as a start frame and an end frame that is one past the last frame, so a token on frame 1 alone has the span $(1, 2)$.

A frame has a duration, the **hop**, in seconds. Frame $t$ covers the time from $t \times \text{hop}$ to $(t + 1) \times \text{hop}$. A span $(a, b)$ in frames is then the interval from $a \times \text{hop}$ to $b \times \text{hop}$ in seconds. In the example, `a` has the span $(1, 2)$ and `b` has $(2, 3)$. At 20 ms per frame, `a` runs from 0.02 s to 0.04 s and `b` from 0.04 s to 0.06 s.

A **word** is a run of tokens. Its start is the start of its first token, and its end is the end of its last token. The blanks around a word belong to no word, so a word does not stretch over the silence that follows it.

Work one path by hand. Take three tokens in two words, the first word with two tokens and the second with one. The path is the states $(0, 0, 1, 2, 3, 3, 4, 5, 5, 6)$ at 20 ms per frame:

- Token 0 is state 1, on frame 2 alone: span $(2, 3)$.
- Token 1 is state 3, on frames 4 and 5: span $(4, 6)$.
- Token 2 is state 5, on frames 7 and 8: span $(7, 9)$.
- Word 0 runs from token 0's start to token 1's end: $2 \times 0.02 = 0.04$ s to $6 \times 0.02 = 0.12$ s.
- Word 1 is token 2 alone: 0.14 s to 0.18 s.

The common slip is the end time. Using the last frame's start, $5 \times 0.02 = 0.10$ s for word 0, loses a whole frame. Using the next token's start, 0.14 s, stretches the word across the blanks.

Some vocabularies include a word delimiter such as `|`, which is a token in the transcript and gets its own span. This chapter passes the number of tokens in each word instead, which keeps the path logic the same.

## Cost and scaling

The table has $T \times (2L + 1)$ cells and each cell looks at three neighbors, so the work grows as $T \cdot L$. For a 30 s clip at 20 ms per frame, $T = 1500$, and a 400-character transcript gives about 800 states, a table of 1.2 million cells. Align long audio in chapters, because the table grows with both.

## Where this breaks

**Peaky timing.** CTC models emit each token as a spike a frame or two wide, and the spike may sit a frame or two from where a phonetician would mark the sound. Expect offsets of 20 to 40 ms at 20 ms per frame, and report the hop with the times. HMM aligners such as the Montreal Forced Aligner model durations with states, so they are often tighter at phone boundaries.

**Text the model cannot spell.** The transcript must use the model's vocabulary. Normalize case, digits, and punctuation first, or a symbol has no id in the vocabulary and the transcript cannot be turned into tokens.

**Too few frames.** Each token needs one frame, and each repeated pair needs a blank frame too. When the frames cannot hold the transcript, no legal path exists and the exercise raises `ValueError`.

**A transcript that does not match the audio.** Viterbi always returns the best path, even for the wrong words, so the times look plausible and are wrong. The path's score is the warning, and CTC segmentation (Kürzinger et al., 2020) uses it to filter bad segments.

> [!RECAP]
> - The trellis has $2L + 1$ states for $L$ tokens, with a backpointer at every cell.
> - A state may stay, step, or skip, and a skip lands only on a token that differs from the one it jumps over.
> - A repeated token needs a blank between copies, or the path collapses to one.
> - A token's span is its frames on the path, and time is frame index times the hop.
> - A word runs from its first token's start to its last token's end.

## Build it step by step

The checks test the functions in this order.

1. **`viterbi_trellis(log_probs, tokens, blank)`.** Returns the score table and backpointers, both `(T, S)`. Build the extended list, fill frame 0 in states 0 and 1, then take the best of stay, step, and skip per cell. Common mistake: allowing the skip between two identical tokens.
2. **`forced_align(log_probs, tokens, blank)`.** Returns `(states, score)`. Call `viterbi_trellis`, pick the better of the last two states, and walk back. Raise `ValueError` when both ends stay at `-np.inf`. Common mistake: starting the walk back from the last state only.
3. **`token_spans(states, n_tokens)`.** Returns an `(n_tokens, 2)` integer array of start and end frames. Use `np.flatnonzero(states == 2 * k + 1)`. Common mistake: an end frame that is the last frame, not one past it.
4. **`word_timestamps(states, word_lengths, hop_seconds)`.** Returns an `(n_words, 2)` array in seconds. Use `token_spans` and group tokens by `word_lengths`. Common mistake: ending a word where the next token starts.

## The exercise

You build `viterbi_trellis`, `forced_align`, `token_spans`, and `word_timestamps`.

The checks confirm every cell of the four-frame table above, that the transcript `a a` blocks the skip, `forced_align` against enumeration of every path with repeated letters included, and the token spans and word times of a hand-worked path. One check plants word times in synthetic emissions and recovers them. The **Run** aligns `hello world` on emissions from a make-believe peaky model and prints each word's start and end in seconds.

## Further reading

- Shinji Watanabe, Speech Recognition and Understanding (11-751/18-781), Carnegie Mellon University, fall 2022. The 26 September lecture "Alignment" covers the left-to-right HMM, CTC, and transducer alignments, and the 9 November lecture "End-to-end ASR: CTC" covers the CTC lattice (https://www.wavlab.org/activities/2022/11751-2022f/).
- PyTorch, "CTC forced alignment API tutorial," torchaudio documentation (https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html).
- Kürzinger et al., "CTC-Segmentation of Large Corpora for German End-to-end Speech Recognition," SPECOM 2020 (https://arxiv.org/abs/2007.09127).
- McAuliffe et al., "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi," Interspeech 2017 (https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html).
- Pratap et al., "Scaling Speech Technology to 1,000+ Languages," 2023 (https://arxiv.org/abs/2305.13516). The multilingual forced aligner is in the MMS release.
