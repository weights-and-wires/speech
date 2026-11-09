---
title: Whisper and large-scale recognition
slug: 037-whisper
part: "Part 4 — Neural speech recognition"
summary: Drive whisper-tiny with your own prompt, decoding loops, and timestamp rules, then score it fairly with a normalizer and WER.
minutes: 60
gpu: true
objectives:
  - Build Whisper's forced prompt from its special tokens, and convert a timestamp token to seconds.
  - Write greedy and beam-search decoding loops that work over any decoder's next-token logits, with Whisper's suppression lists.
  - Implement Whisper's timestamp rules as a logit filter, and explain what each rule prevents.
  - Normalize text the way Whisper's evaluation does, and compute corpus WER on LibriSpeech with and without it.
prereqs:
  - 012-log-mel-spectrograms
  - 025-word-error-rate
  - 032-attention-encoder-decoders
exercise: 037-whisper
---

# Whisper and large-scale recognition

You decode the same 20 LibriSpeech utterances with the same `whisper-tiny` weights and score the result twice. The first score is a word error rate of 99.8%. The second is 8.3%. The transcripts did not change, so something other than the model decides what "correct" means.

Whisper took a different bet from the recognizers you have built so far: 680,000 hours of web audio paired with whatever transcripts came with it, and a plain Transformer encoder-decoder trained to predict them. The labels are noisy, but there are so many that the model copes with accents, noise, and dozens of languages. Even so, the trained model is only half of a recognizer. A short prompt of special tokens sets what you ask for, and the decoding loop and the scoring decide what you get.

This chapter has one idea: **the weights are fixed, and everything around them is yours to build.** You write the prompt, a greedy loop, Whisper's beam search, the timestamp rules, and the scoring that makes a WER fair.

## What you already know

- Whisper's input is an 80-band log-mel spectrogram of a 30 s window (see [Log-mel spectrograms](/lessons/012-log-mel-spectrograms)).
- WER is substitutions, deletions, and insertions over the reference words, from a Levenshtein alignment (see [Word error rate](/lessons/025-word-error-rate)).
- An attention encoder-decoder reads the encoder's output and predicts one token at a time (see [Attention encoder-decoders](/lessons/032-attention-encoder-decoders)).
- Beam search keeps several hypotheses and ranks finished ones with a length correction.

## The model in one pass

Whisper cuts audio into 30 s windows at 16 kHz, which is 480,000 samples, and pads short clips with zeros. The front end gives 3,000 log-mel frames, one every 10 ms. Two convolutions, the second with stride 2, halve that to 1,500 encoder positions, one every 20 ms, and the encoder runs once per window.

The decoder is a Transformer language model that attends to the encoder output. It reads a token sequence and returns, at every position, logits over the next token. `whisper-tiny`, the model in the Run, has 4 encoder and 4 decoder blocks of width 384, 39 million parameters, a vocabulary of 51,865 tokens, and a decoder context of 448 tokens.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Whisper's pipeline: 480,000 samples become 3,000 log-mel frames, then 1,500 encoder positions, then a decoder that your loop drives one token at a time">
<defs><marker id="c035-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<rect x="15.0" y="25.0" width="120.0" height="70.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="75.0" y="52.0" text-anchor="middle" class="fig-bold">30 s audio</text>
<text x="75.0" y="74.0" text-anchor="middle" class="fig-small">480,000 samples</text>
<rect x="165.0" y="25.0" width="120.0" height="70.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="225.0" y="52.0" text-anchor="middle" class="fig-bold">log-mel</text>
<text x="225.0" y="74.0" text-anchor="middle" class="fig-small">80 x 3,000</text>
<rect x="315.0" y="25.0" width="120.0" height="70.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="375.0" y="52.0" text-anchor="middle" class="fig-bold">encoder</text>
<text x="375.0" y="74.0" text-anchor="middle" class="fig-small">1,500 x 384</text>
<rect x="465.0" y="25.0" width="120.0" height="70.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="525.0" y="52.0" text-anchor="middle" class="fig-bold">decoder</text>
<text x="525.0" y="74.0" text-anchor="middle" class="fig-small">51,865 logits</text>
<line x1="137.0" y1="60.0" x2="163.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="287.0" y1="60.0" x2="313.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="437.0" y1="60.0" x2="463.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<text x="225.0" y="112.0" text-anchor="middle" class="fig-small">one frame per 10 ms</text>
<text x="375.0" y="112.0" text-anchor="middle" class="fig-small">one position per 20 ms</text>
<rect x="15.0" y="150.0" width="280.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="155.0" y="172.0" text-anchor="middle" class="fig-bold">prompt, chosen by you</text>
<text x="155.0" y="191.0" text-anchor="middle" class="fig-mono fig-small">[50258, 50259, 50359, 50363]</text>
<rect x="325.0" y="150.0" width="260.0" height="52.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="455.0" y="172.0" text-anchor="middle" class="fig-bold">your loop</text>
<text x="455.0" y="191.0" text-anchor="middle" class="fig-small">filter, log-softmax, choose, append</text>
<line x1="525.0" y1="148.0" x2="525.0" y2="98.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="300.0" y1="176.0" x2="323.0" y2="176.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<text x="300.0" y="232.0" text-anchor="middle" class="fig-small">the chosen token joins the sequence, and the decoder runs again</text>
</svg>
Figure 1. The encoder runs once per 30 s window. The decoder runs once per generated token, and everything you control, the prompt, the filter, and the choice rule, sits around that loop.
```

In Figure 1, everything left of the decoder is fixed. The decoder runs once per generated token, so the loop around it decides what comes out.

## The prompt is the task

One multitask model has to know which task you want, so every decode starts from tokens the decoder did not choose. The following table lists the ones you use, with the ids from the tokenizer in `assets/whisper-tiny`.

| Token | Id | Meaning |
|---|---|---|
| `<\|endoftext\|>` | 50257 | End of the transcript |
| `<\|startoftranscript\|>` | 50258 | Start of every prompt |
| `<\|en\|>` | 50259 | English; the 99 language tokens run from 50259 to 50357 |
| `<\|translate\|>` | 50358 | Translate into English |
| `<\|transcribe\|>` | 50359 | Transcribe in the spoken language |
| `<\|notimestamps\|>` | 50363 | Text only, no timestamps |
| `<\|0.00\|>` | 50364 | The first of 1,501 timestamp tokens |

The English transcription prompt without timestamps is `[50258, 50259, 50359, 50363]`. With timestamps, drop the last id.

Timestamp tokens count time in steps of 20 ms, the encoder's resolution:

$$
\text{id} = 50364 + \frac{t}{0.02}, \qquad t = 0.02\,(\text{id} - 50364)
$$

Read it aloud: a time $t$ in seconds becomes the id of `<|0.00|>` plus the number of 20 ms steps. In the Run, `whisper-tiny` ends the first LibriSpeech utterance at 5.44 s. That is $5.44 / 0.02 = 272$ steps, so the closing token is $50364 + 272 = 50636$. The last timestamp, 51864, is `<|30.00|>`.

## A decoder is a function

To keep the loops independent of Whisper, the exercise treats a decoder as a function `step_fn(tokens) -> logits`. It takes the whole sequence so far, prompt included, and returns a 1-D array of next-token logits. The Run wraps `whisper-tiny` in one, and the checks use fake decoders.

**Greedy decoding** takes the likeliest token until it picks `<|endoftext|>` or reaches a length limit. It also sums the log-probability of every choice, which Whisper uses to judge a decode:

$$
\log P(y \mid x) = \sum_{u=1}^{U} \log \operatorname{softmax}\left(\tilde z_u\right)_{y_u}
$$

Read it aloud: $\tilde z_u$ is the logit vector at step $u$ after filtering, $y_u$ is the chosen token, and the sum runs over the $U$ tokens generated. In numpy, one step is three lines:

```python
logits = logit_filter(step_fn(tokens), tokens[len(prompt):])
logp = log_softmax(logits)
token = int(np.argmax(logp))
```

The **filter** sets forbidden tokens to $-\infty$ before the softmax, which gives them probability zero. Whisper's generation config names two lists. `suppress_tokens` holds 88 ids of non-speech symbols, such as brackets and musical notes, and applies at every step. `begin_suppress_tokens` is `[220, 50257]`, a lone space and `<|endoftext|>`, and applies only at the first step, so the model cannot return an empty transcript. With both lists, the Run's greedy tokens on the first utterance match `model.generate` exactly.

> [!TIP]
> Filter before the log-softmax. The summed log-probability then measures the distribution the decoder actually chose from, which is the one Whisper reports.

## Beam search: why not always take the best token?

Greedy decoding cannot take back a choice, and a token that looks best can lead to a poor sentence. Beam search keeps $B$ unfinished hypotheses, so a weaker start gets a second chance.

Whisper's version works like this. At each step, extend every beam by its $B + 1$ likeliest tokens, one more than the beam size, so a beam whose best token is `<|endoftext|>` still leaves $B$ live candidates. Sort all extensions by summed log-probability and walk down the list. An extension ending in `<|endoftext|>` joins the finished set, and anything else joins the next beams until $B$ are kept. Stop once $B$ hypotheses have finished.

Finished hypotheses are ranked by log-probability per token. With no length penalty, Whisper divides by the number $n$ of tokens before `<|endoftext|>`:

$$
\operatorname{score}(y) = \frac{\log P(y \mid x)}{\max(n, 1)}
$$

Read it aloud: the average log-probability of each generated token. Pruning during the search uses the raw sum, and the division applies only at the end.

### Work a greedy trap by hand

Take a decoder over four tokens: `<eot>`, `a`, `cat`, and `the`. At the first step its logits are $(-5, 1.0, -5, 0.8)$, so the log-probabilities are $-6.601$, $-0.601$, $-6.601$, and $-0.801$. After `a` it has no preference, so every token gets $\log 0.25 = -1.386$. After `the`, `cat` gets $-0.003$, and after `the cat`, `<eot>` gets $-0.020$.

- Greedy takes `a` ($-0.601$), then `<eot>` ($-1.386$). The total is $-1.987$ over one token, so the score is $-1.987$.
- Beam search with $B = 3$ keeps `a`, `the`, and `cat` after step 1, and finishes `<eot>` alone at $-6.601$. At step 2, `the cat` leads at $-0.804$ and `a <eot>` finishes at $-1.987$. At step 3, `the cat <eot>` finishes at $-0.824$, the third finished hypothesis, so the search stops.
- The scores are $-6.601$, $-1.987$, and $-0.824 / 2 = -0.412$. Beam search returns `the cat`.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A search tree for the greedy trap: greedy takes a then ends with score -1.987, while beam search finds the cat with score -0.412">
<defs><marker id="c035-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<rect x="10.0" y="110.0" width="80.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="50.0" y="131.0" text-anchor="middle" class="fig-bold">prompt</text>
<text x="50.0" y="149.0" text-anchor="middle" class="fig-small">start</text>
<rect x="170.0" y="30.0" width="90.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="215.0" y="51.0" text-anchor="middle" class="fig-bold">a</text>
<text x="215.0" y="69.0" text-anchor="middle" class="fig-small">-0.601</text>
<rect x="170.0" y="110.0" width="90.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="215.0" y="131.0" text-anchor="middle" class="fig-bold">the</text>
<text x="215.0" y="149.0" text-anchor="middle" class="fig-small">-0.801</text>
<rect x="170.0" y="200.0" width="90.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="215.0" y="221.0" text-anchor="middle" class="fig-bold">&lt;eot&gt;</text>
<text x="215.0" y="239.0" text-anchor="middle" class="fig-small">-6.601</text>
<rect x="340.0" y="30.0" width="100.0" height="50.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="390.0" y="51.0" text-anchor="middle" class="fig-bold">a &lt;eot&gt;</text>
<text x="390.0" y="69.0" text-anchor="middle" class="fig-small">total -1.987</text>
<rect x="340.0" y="110.0" width="100.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="390.0" y="131.0" text-anchor="middle" class="fig-bold">the cat</text>
<text x="390.0" y="149.0" text-anchor="middle" class="fig-small">total -0.804</text>
<rect x="490.0" y="110.0" width="100.0" height="50.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="540.0" y="131.0" text-anchor="middle" class="fig-bold">the cat &lt;eot&gt;</text>
<text x="540.0" y="149.0" text-anchor="middle" class="fig-small">total -0.824</text>
<line x1="92.0" y1="135.0" x2="168.0" y2="55.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="92.0" y1="135.0" x2="168.0" y2="135.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="92.0" y1="135.0" x2="168.0" y2="225.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="262.0" y1="55.0" x2="338.0" y2="55.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="262.0" y1="135.0" x2="338.0" y2="135.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="442.0" y1="135.0" x2="488.0" y2="135.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<text x="300.0" y="48.0" text-anchor="middle" class="fig-small">-1.386</text>
<text x="301.0" y="128.0" text-anchor="middle" class="fig-small">-0.003</text>
<text x="465.0" y="128.0" text-anchor="middle" class="fig-small">-0.020</text>
<text x="390.0" y="98.0" text-anchor="middle" class="fig-small">score -1.987 (greedy)</text>
<text x="540.0" y="178.0" text-anchor="middle" class="fig-small fig-bold">score -0.412</text>
<text x="540.0" y="194.0" text-anchor="middle" class="fig-small">(-0.824 / 2, wins)</text>
<text x="215.0" y="266.0" text-anchor="middle" class="fig-small">score -6.601 (n = 0)</text>
</svg>
Figure 2. Greedy commits to a, which is 0.2 nats likelier than the, and then ends with a score of -1.987. Beam search keeps the word the alive, finds the cat, and its per-token score of -0.412 beats both finished rivals.
```

Greedy committed to `a` because it was likelier by 0.2 nats, and it never saw that nothing good followed.

> [!KEY]
> Beam search keeps a hypothesis alive after a less likely first token, so it can find a sequence that is likelier as a whole. Greedy decoding cannot take back a choice.

## Timestamp rules

With timestamps on, the decoder writes segments as `<|t0|> text <|t1|><|t1|> text <|t2|>`: an opening time, the words, a closing time, and the next segment's opening time. Nothing in the network enforces that grammar, so Whisper applies it as a filter on the logits, in this order:

1. Forbid `<|notimestamps|>`.
2. After two timestamps in a row, or one that starts the output, forbid all timestamps, because text comes next. After text and one timestamp, forbid all text, because a timestamp or `<|endoftext|>` comes next.
3. Forbid timestamps that go backward. The one right after a closing timestamp may repeat it, and any other must be strictly later than the last.
4. At the first step, forbid all text, and forbid timestamps after `<|1.00|>`, id 50414.
5. If the total probability of all timestamp tokens exceeds the probability of the likeliest text token, forbid all text.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="The timestamp grammar on one segment: an opening timestamp, text, a closing timestamp, then a timestamp or end of text">
<defs><marker id="c035-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<rect x="10.0" y="30.0" width="120.0" height="60.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="70.0" y="56.0" text-anchor="middle" class="fig-bold fig-small">&lt;|0.00|&gt;</text>
<text x="70.0" y="76.0" text-anchor="middle" class="fig-small">id 50364</text>
<rect x="150.0" y="30.0" width="160.0" height="60.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="230.0" y="56.0" text-anchor="middle" class="fig-bold fig-small">Mr. Quilter ... gospel.</text>
<text x="230.0" y="76.0" text-anchor="middle" class="fig-small">text tokens</text>
<rect x="330.0" y="30.0" width="120.0" height="60.0" rx="6" class="fill-soft-a stroke-a"/>
<text x="390.0" y="56.0" text-anchor="middle" class="fig-bold fig-small">&lt;|5.44|&gt;</text>
<text x="390.0" y="76.0" text-anchor="middle" class="fig-small">id 50636</text>
<rect x="470.0" y="30.0" width="120.0" height="60.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="530.0" y="56.0" text-anchor="middle" class="fig-bold fig-small">&lt;|endoftext|&gt;</text>
<text x="530.0" y="76.0" text-anchor="middle" class="fig-small">id 50257</text>
<line x1="130.0" y1="60.0" x2="148.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="310.0" y1="60.0" x2="328.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<line x1="450.0" y1="60.0" x2="468.0" y2="60.0" class="stroke-muted" marker-end="url(#c035-arrow)"/>
<text x="70.0" y="118.0" text-anchor="middle" class="fig-small fig-bold">rules 1 and 4</text>
<text x="70.0" y="136.0" text-anchor="middle" class="fig-small">first token is a</text>
<text x="70.0" y="154.0" text-anchor="middle" class="fig-small">timestamp, at or</text>
<text x="70.0" y="172.0" text-anchor="middle" class="fig-small">before 1.00 s</text>
<text x="230.0" y="118.0" text-anchor="middle" class="fig-small fig-bold">rule 2</text>
<text x="230.0" y="136.0" text-anchor="middle" class="fig-small">after an opening</text>
<text x="230.0" y="154.0" text-anchor="middle" class="fig-small">timestamp, no more</text>
<text x="230.0" y="172.0" text-anchor="middle" class="fig-small">timestamps</text>
<text x="390.0" y="118.0" text-anchor="middle" class="fig-small fig-bold">rules 2 and 3</text>
<text x="390.0" y="136.0" text-anchor="middle" class="fig-small">after text, only a</text>
<text x="390.0" y="154.0" text-anchor="middle" class="fig-small">timestamp or eot;</text>
<text x="390.0" y="172.0" text-anchor="middle" class="fig-small">never go backward</text>
<text x="530.0" y="118.0" text-anchor="middle" class="fig-small fig-bold">end</text>
<text x="530.0" y="136.0" text-anchor="middle" class="fig-small">the decode stops</text>
<text x="530.0" y="154.0" text-anchor="middle" class="fig-small">at &lt;|endoftext|&gt;</text>
<text x="300.0" y="208.0" text-anchor="middle" class="fig-small">rule 5: when the timestamps together outweigh every text token, text is forbidden</text>
<text x="300.0" y="226.0" text-anchor="middle" class="fig-small">time = 0.02 s x (id - 50364): 5.44 s is 272 steps, id 50636</text>
</svg>
Figure 3. One segment is a timestamp, text, and a timestamp. The rules are masks on the logits, applied in order, so the network never has to learn the grammar exactly.
```

Rule 5 compares one token against a crowd. If each of the 1,501 timestamps has probability 0.0004, together they hold 0.60, which beats a single text token at 0.30. In log space the test is `np.logaddexp.reduce(logp[50364:]) > logp[:50364].max()`. It closes a segment when the decoder is unsure which time to pick but sure that a time comes next.

> [!KEY]
> The prompt, the suppression lists, and the timestamp rules are part of the recognizer. The same weights give different transcripts under a different decoding loop.

## Normalize before you score

Return to the opening question. LibriSpeech references are uppercase, unpunctuated, and spell titles out: `MISTER QUILTER IS THE APOSTLE`. Whisper writes `Mr. Quilter is the apostle.` Scored word by word, almost every word is wrong, because `Mr.` is not `MISTER` and `apostle.` is not `APOSTLE`. On the Run's 20 utterances, greedy output scores 99.8% WER against the raw references, and 8.3% once both sides are normalized.

Whisper's evaluation applies an English normalizer to reference and hypothesis alike. The exercise implements its core, in this order:

1. Lowercase.
2. Delete text inside brackets `[]`, angle brackets `<>`, and parentheses.
3. Delete the fillers hmm, mm, mhm, mmm, uh, and um as whole words.
4. Expand contractions and titles: `won't` to `will not`, `n't` to ` not`, `'re` to ` are`, `mr` to `mister`, and the others in `REPLACEMENTS`.
5. Turn every remaining punctuation mark or symbol into a space, and collapse runs of spaces.

For example, `Um, I won't go [laughs] (quietly)!` becomes `i will not go`. The full normalizer also turns spoken numbers into digits and British spellings into American ones, so the Run still counts `west 10th street` against `west tenth street` as an error.

WER itself is the one from [Word error rate](/lessons/025-word-error-rate): $(S + D + I) / N$, summed over the whole corpus before dividing, so a long utterance counts for more than a short one.

> [!WARNING]
> Normalize both sides with the same function. Normalizing only the hypothesis leaves the uppercase reference unmatched, and every word counts as wrong.

> [!RECAP]
> - The English transcription prompt is `[50258, 50259, 50359, 50363]`, and timestamp id $50364 + k$ means $0.02k$ seconds.
> - A decoder is `step_fn(tokens) -> logits`. Filter, then log-softmax, then choose.
> - Beam search extends each beam by $B + 1$ tokens, stops after $B$ finish, and ranks by log-probability per token.
> - The timestamp rules are five masks, applied in order.
> - Normalize both sides the same way, then compute corpus WER.

## Build it step by step

The checks test the functions in this order. `word_errors` is provided, because you built it in [Word error rate](/lessons/025-word-error-rate).

1. **`build_prompt(language, task, timestamps)`.** Returns a list of ints from `SPECIAL` and `LANGUAGES`. Common mistake: adding `<|notimestamps|>` when timestamps are wanted.
2. **`greedy_decode(step_fn, prompt, eot, max_new_tokens, logit_filter)`.** Returns `(generated, sum_logprob)`, with `eot` included when chosen. Common mistake: taking the log-softmax before filtering.
3. **`beam_search_decode(step_fn, prompt, eot, beam_size, max_new_tokens, logit_filter)`.** Same return shape. Key candidates by `tuple(tokens)`, and close live beams with `eot` if the length limit arrives first. Common mistake: counting `eot` in the length.
4. **`apply_timestamp_rules(logits, generated, ...)`.** Returns a filtered copy of shape `(V,)`. Common mistake: applying rule 5 before rules 1 to 4.
5. **`normalize_text(text)`.** Returns a string. Use `re.sub` for each step. Common mistake: stripping apostrophes before expanding contractions.
6. **`corpus_wer(references, hypotheses)`.** Normalizes both sides, calls `word_errors` on each pair, and sums the edits before dividing. Common mistake: averaging per-utterance WERs.

## Where this breaks

**Long audio.** One window holds 30 s. Whisper's `transcribe` decodes one window, moves forward to the last timestamp, and passes the previous text as a prompt. When a decode repeats itself (gzip compression ratio above 2.4) or is unsure (average log-probability below $-1.0$), it retries at higher temperatures.

**Normalizer mismatch.** Two papers with different normalizers report incomparable WERs. State the normalizer, the split, and whether you used an external language model.

**Small models.** Beam 5 brought 8.3% down to 7.1% in the Run, but a larger model helps more than a wider beam.

## The exercise

You build `build_prompt`, the loops `greedy_decode` and `beam_search_decode`, the filter `apply_timestamp_rules`, and the scoring functions `normalize_text` and `corpus_wer`.

None of the checks needs Whisper. They drive your loops with fake decoders: greedy against step-by-step argmax, beam size 1 against greedy, a wide beam against brute force, and the greedy trap from this chapter. They check the timestamp rules on crafted logits, the normalizer on strings, and `corpus_wer` against an independent edit distance.

The **Run** is the GPU experiment. It loads `whisper-tiny`, verifies the special-token ids against its tokenizer, and transcribes LibriSpeech dev-clean with your loops, printing WER before and after normalization. It picks `cuda` when a GPU is present and decodes 20 utterances with beam size 5. Without a GPU, it says it is running the small version, 4 utterances and beam size 3, so it finishes in under a minute. Without the assets, it names what it needs and decodes a toy decoder instead.
