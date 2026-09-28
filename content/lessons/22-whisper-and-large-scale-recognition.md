---
title: Whisper and large-scale recognition
slug: 22-whisper-and-large-scale-recognition
part: "Part 4 — Neural speech recognition"
summary: Drive whisper-tiny with your own prompt, greedy and beam-search loops, and timestamp rules, then score it fairly with a normalizer and WER.
minutes: 75
gpu: true
objectives:
  - Build Whisper's forced prompt from its special tokens, and convert a timestamp token to seconds.
  - Write greedy and beam-search decoding loops that work over any decoder's next-token logits, with Whisper's suppression lists.
  - Implement Whisper's timestamp rules as a logit filter, and explain what each rule prevents.
  - Normalize text the way Whisper's evaluation does, and compute corpus WER on LibriSpeech with and without it.
prereqs:
  - 07-hearing-and-the-mel-scale
  - 14-language-models-and-word-error-rate
  - 18-attention-encoder-decoders
exercise: 22-whisper-and-large-scale-recognition
---

# Whisper and large-scale recognition

The recognizers so far learned to transcribe from a few hundred hours of
carefully labeled speech. Even self-supervised pretraining, from the previous
lesson, ends with fine-tuning on a small labeled set. Whisper took the opposite
bet: 680,000 hours of audio from
the web, paired with whatever transcripts came with it, and a plain
Transformer encoder-decoder trained to predict them. The labels are noisy, but
there are so many of them that the model learns to handle accents, noise, and
dozens of languages.

The model is only half the system. What you ask for is set by a short prompt of
special tokens, and what you get depends on the decoding loop: which tokens it
forbids, how many hypotheses it keeps, and when it inserts timestamps. Those
choices are yours, not the model's.

In this lesson you write that half yourself. You build the prompt, a greedy
loop and Whisper's beam search over any decoder's logits, and the timestamp
rules. Then you run them on `whisper-tiny` over LibriSpeech and learn why a WER
of 99.8% and a WER of 8.3% can describe the same transcripts.

## What you already know

- Whisper's input is an 80-band log-mel spectrogram of a 30 s window, which
  you built in Hearing and the mel scale.
- WER is substitutions, deletions, and insertions over reference words, from a
  Levenshtein alignment, as in Language models and word error rate.
- An attention encoder-decoder reads the encoder's output and predicts one
  token at a time, as in Attention encoder-decoders.
- Beam search keeps several hypotheses and ranks finished ones with a length
  correction, from the same lesson.
- A log-softmax turns logits into log-probabilities that sum, after `exp`, to
  one.

## The model in one pass

Whisper cuts audio into 30 s windows at 16 kHz, 480,000 samples, padding short
clips with zeros. The front end gives 3,000 log-mel frames, one every 10 ms.
Two convolutions, the second with stride 2, halve that to 1,500 encoder
positions, one per 20 ms. The encoder is a stack of Transformer blocks over
those positions.

The decoder is a Transformer language model that attends to the encoder
output. It reads a token sequence and returns, at every position, logits over
the next token. `whisper-tiny`, the model in the Run, has 4 encoder and 4
decoder blocks of width 384 with 6 heads, 39 million parameters, a vocabulary
of 51,865 tokens, and a decoder context of 448 tokens. You run the encoder
once per window, then call the decoder once per generated token.

## The prompt is the task

Every decode starts from tokens the decoder didn't choose. They tell one
multitask model which task to do. The following table lists the ones you use,
with the ids from the tokenizer in `assets/whisper-tiny`.

| Token | Id | Meaning |
|---|---|---|
| `<\|endoftext\|>` | 50257 | End of the transcript |
| `<\|startoftranscript\|>` | 50258 | Start of every prompt |
| `<\|en\|>` | 50259 | English; the 99 language tokens run from 50259 to 50357 |
| `<\|translate\|>` | 50358 | Translate into English |
| `<\|transcribe\|>` | 50359 | Transcribe in the spoken language |
| `<\|notimestamps\|>` | 50363 | Text only, no timestamps |
| `<\|0.00\|>` | 50364 | The first of 1,501 timestamp tokens |

The English transcription prompt without timestamps is
`[50258, 50259, 50359, 50363]`. With timestamps, drop the last token. To
detect the language instead of stating it, run one decoder step after
`<|startoftranscript|>` and compare the logits of the language tokens only.

Timestamp tokens count time in steps of 20 ms, the encoder's resolution:

$$
\text{id} = 50364 + \frac{t}{0.02}, \qquad t = 0.02\,(\text{id} - 50364)
$$

Read it aloud: a time $t$ in seconds becomes the id of `<|0.00|>` plus the
number of 20 ms steps. The last one, 51864, is `<|30.00|>`. In the Run,
`whisper-tiny` transcribes the first LibriSpeech utterance as
`<|0.00|> Mr. Quilter is the apostle ... gospel.<|5.44|>`, and 5.44 s is
$5.44/0.02 = 272$ steps, so the closing token is 50636.

## A decoder is a function

To keep the loops independent of Whisper, the exercise treats a decoder as a
function `step_fn(tokens) -> logits`. It takes the whole sequence so far,
prompt included, and returns a 1-D array of next-token logits. The Run wraps
`whisper-tiny` in one; the checks use small fake decoders whose logits they
choose.

**Greedy decoding** takes the most likely token until it picks
`<|endoftext|>` or reaches a length limit. It also sums the log-probability of
each choice, which Whisper uses to judge a decode:

$$
\log P(y \mid x) = \sum_{u=1}^{U} \log \operatorname{softmax}\left(\tilde z_u\right)_{y_u}
$$

Read it aloud: $\tilde z_u$ is the logit vector at step $u$ after filtering,
$y_u$ is the chosen token, and the sum runs over the $U$ tokens generated. In
numpy:

```python
logits = logit_filter(step_fn(tokens), tokens[len(prompt):])
logp = log_softmax(logits)
token = int(np.argmax(logp))
```

The **filter** sets forbidden tokens to $-\infty$ before the softmax. Whisper's
generation config names two lists. `suppress_tokens` holds 88 ids of symbols
that mark non-speech, such as quotation marks, brackets, and musical notes,
and applies at every step. `begin_suppress_tokens` is `[220, 50257]`, a lone
space and `<|endoftext|>`, and applies only at the first step, so the model
can't return an empty transcript. With both lists, the Run's greedy tokens on
the first utterance match `model.generate` exactly.

> [!TIP]
> Filter before the log-softmax. The summed log-probability then measures the
> distribution the decoder chose from, which is the one Whisper reports.

## Whisper's beam search

Beam search follows the recipe from Attention encoder-decoders, with Whisper's
details. Keep $B$ unfinished beams. At each step, extend each beam by its
$B + 1$ most likely tokens, one more than the beam size, so a beam whose best
token is `<|endoftext|>` still leaves $B$ live candidates. Sort all extensions
by summed log-probability and walk down the list: an extension ending in
`<|endoftext|>` joins the finished set, and anything else joins the next
beams, until $B$ beams are kept. Stop once $B$ hypotheses have finished.

Rank the finished hypotheses by log-probability per token. With no length
penalty set, Whisper divides by the number $n$ of tokens before
`<|endoftext|>`:

$$
\operatorname{score}(y) = \frac{\log P(y \mid x)}{\max(n, 1)}
$$

Read it aloud: the average log-probability of each generated token. Pruning
during the search uses the raw sum; the division applies only at the end.

### Work a greedy trap by hand

Take a decoder over four tokens, `<eot>`, `a`, `cat`, and `the`. At the first
step its logits are $(-5, 1.0, -5, 0.8)$, so the log-probabilities are
$-6.601$, $-0.601$, $-6.601$, and $-0.801$. After `a` it has no preference, so
every token gets $\log 0.25 = -1.386$. After `the`, `cat` gets $-0.003$, and
after `the cat`, `<eot>` gets $-0.020$.

- Greedy takes `a` ($-0.601$), then `<eot>` ($-1.386$): a total of $-1.987$
  over one token, a score of $-1.987$.
- Beam search with $B = 3$ keeps `a`, `the`, and `cat` after step 1, and
  finishes `<eot>` alone at $-6.601$. At step 2, `the cat` leads at $-0.804$
  and `a <eot>` finishes at $-1.987$. At step 3, `the cat <eot>` finishes at
  $-0.824$, the third finished hypothesis, and the search stops.
- The scores are $-6.601$, $-1.987$, and $-0.824 / 2 = -0.412$. Beam search
  returns `the cat`.

Greedy committed to `a` because it was likelier by 0.2 nats, and never saw that
nothing good followed it.

> [!KEY]
> Beam search keeps a hypothesis alive after a less likely first token, so it
> can find a sequence that is likelier as a whole. Greedy decoding can't take
> back a choice.

## Timestamp rules

With timestamps on, the decoder writes segments as
`<|t0|> text <|t1|><|t1|> text <|t2|>`: an opening time, the words, a closing
time, and the next segment's opening time. Nothing in the network enforces
that grammar. Whisper applies it as a filter, in this order:

1. Forbid `<|notimestamps|>`.
2. After two timestamps in a row, or one that starts the output, forbid all
   timestamps: text comes next. After text and one timestamp, forbid all text:
   another timestamp or `<|endoftext|>` comes next.
3. Forbid timestamps that go backward. The timestamp right after a closing
   one may repeat it; any other must be strictly later than the last one.
4. At the first step, forbid all text, and forbid timestamps after `<|1.00|>`,
   id 50414.
5. If the total probability of all timestamp tokens exceeds the probability of
   the likeliest text token, forbid all text.

Rule 5 compares one token against a crowd. In log space it's
`np.logaddexp.reduce(logp[50364:]) > logp[:50364].max()`. It makes the decoder
close a segment when it's unsure which time to pick but sure that a time comes
next.

> [!KEY]
> The prompt, the suppression lists, and the timestamp rules are part of the
> recognizer. The same weights give different transcripts under a different
> decoding loop.

## Normalize before you score

LibriSpeech references are uppercase, unpunctuated, and spell numbers out:
`MISTER QUILTER IS THE APOSTLE`. Whisper writes `Mr. Quilter is the apostle.`
Scored word by word, almost every word is wrong. On the Run's 20 utterances,
greedy output scores 99.8% WER against the raw references, and 8.3% once both
sides are normalized.

Whisper's evaluation applies an English normalizer to both reference and
hypothesis. The exercise implements its core, in this order:

1. Lowercase.
2. Delete text inside brackets `[]`, angle brackets `<>`, and parentheses.
3. Delete the fillers hmm, mm, mhm, mmm, uh, and um as whole words.
4. Expand contractions and titles: `won't` to `will not`, `n't` to ` not`,
   `'re` to ` are`, `mr` to `mister`, and the others in `REPLACEMENTS`.
5. Turn every remaining punctuation mark or symbol into a space, and collapse
   runs of spaces.

For example, `Um, I won't go [laughs] (quietly)!` becomes `i will not go`. The
full normalizer also turns spoken numbers into digits and British spellings
into American ones. Without that, the Run counts `west 10th street` against
`west tenth street` as an error.

WER itself is the one from Language models and word error rate:
$\text{WER} = (S + D + I)/N$, summed over the corpus before dividing.

> [!WARNING]
> Normalize both sides with the same function. Normalizing only the hypothesis
> leaves the uppercase reference unmatched, and every word counts as wrong.

> [!RECAP]
> - The English transcription prompt is `[50258, 50259, 50359, 50363]`, and
>   timestamp id $50364 + k$ means $0.02k$ seconds.
> - A decoder is `step_fn(tokens) -> logits`; filter, then log-softmax, then
>   choose.
> - Whisper's beam extends each beam by $B + 1$ tokens, stops after $B$ finish,
>   and ranks by log-probability per token.
> - The timestamp rules are five masks, applied in order, the last one
>   comparing total timestamp mass with the best text token.
> - Normalize reference and hypothesis the same way, then compute corpus WER.

## Build it step by step

The checks test the functions in this order.

1. **`build_prompt(language, task, timestamps)`.** Returns a list of ints from
   `SPECIAL` and `LANGUAGES`. Common mistake: adding `<|notimestamps|>` when
   timestamps are wanted.
2. **`greedy_decode(step_fn, prompt, eot, max_new_tokens, logit_filter)`.**
   Returns `(generated, sum_logprob)`, with `eot` included when chosen. Pass
   `tokens[len(prompt):]` to the filter. Common mistake: taking the log-softmax
   before filtering.
3. **`beam_search_decode(step_fn, prompt, eot, beam_size, max_new_tokens, logit_filter)`.**
   Same return shape. Key candidates by `tuple(tokens)`, sort with
   `sorted(candidates, key=candidates.get, reverse=True)`, and close live beams
   with `eot` if the length limit arrives first. Common mistake: counting
   `eot` in the length.
4. **`apply_timestamp_rules(logits, generated, ...)`.** Returns a filtered copy
   of shape `(V,)`. Common mistake: applying rule 5 before rules 1 to 4.
5. **`normalize_text(text)`.** Returns a string. Use `re.sub` for each step.
   Common mistake: stripping apostrophes before expanding contractions.
6. **`word_errors(reference, hypothesis)` and `corpus_wer(references, hypotheses)`.**
   A Levenshtein table and backtrace for `(S, D, I)`, then a corpus sum.
   Common mistake: averaging per-utterance WERs.

## Cost and scaling

The following table lists costs for a 30 s window, $U$ generated tokens, and
beam size $B$.

| Piece | Time | Memory |
|---|---|---|
| Encoder | once per window, $1500^2$ attention per layer | 1,500 positions |
| Decoder without a cache | $U^2$ positions in all, times $B$ | one sequence per beam |
| Decoder with a key-value cache | $U$ positions in all, times $B$ | cached keys and values |
| Beam bookkeeping | sorting $B(B+1)$ candidates per step | $B$ token lists |

The exercise's loops call the decoder on the whole sequence every step. That's
quadratic in $U$ but simple, and $U$ is short here. Production decoders keep
each layer's keys and values, which makes each step cost one position. On a
laptop CPU, the Run's full version, 20 utterances with greedy and beam 5, took
about a minute.

## Where this breaks

**Long audio.** One window holds 30 s. Whisper's `transcribe` decodes one
window, moves forward to the last timestamp, and passes the previous text as a
prompt. When a decode repeats itself (gzip compression ratio above 2.4) or is
unsure (average log-probability below $-1.0$), it retries at temperatures 0.2,
0.4, up to 1.0.

**Hallucination.** On silence or music, Whisper can write fluent text that was
never said. One study found fabricated phrases in about 1% of transcriptions.
Run voice activity detection first, and treat a window as silent when
`<|nocaptions|>` is likely and the log-probability is low.

**Normalizer mismatch.** Two papers with different normalizers report
incomparable WERs. State the normalizer, the split, and whether you used an
external language model.

**Greedy loops.** A greedy decode can repeat a phrase until the length limit.
Beam search, the compression-ratio check, and the length limit all guard
against it.

**Small models.** `whisper-tiny` mishears rare words: `cyril` becomes `carol`,
and `novatians` becomes `innovations`. Beam 5 fixed some errors in the Run,
bringing 8.3% down to 7.1%, but a larger model helps more than a wider beam.

## The exercise

You build `build_prompt`, the loops `greedy_decode` and `beam_search_decode`,
the filter `apply_timestamp_rules`, and the scoring functions
`normalize_text`, `word_errors`, and `corpus_wer`.

None of the checks needs Whisper. They drive your loops with fake decoders:
greedy against step-by-step argmax, beam size 1 against greedy, a wide beam
against brute-force enumeration of every sequence, and the greedy trap from
this lesson. They check each timestamp rule on crafted logits, the normalizer
on strings, and WER against an independent edit distance. The **Run** loads
`whisper-tiny`, verifies the special-token ids against its tokenizer, and
transcribes LibriSpeech dev-clean with your loops, printing WER before and
after normalization.
