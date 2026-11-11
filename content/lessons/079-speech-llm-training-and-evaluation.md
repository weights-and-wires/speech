---
title: Speech LLM training and evaluation
slug: 079-speech-llm-training-and-evaluation
part: "Part 8 — Beyond recognition and synthesis"
summary: Build multi-task audio instruction data with a response-only loss mask, interleave text and speech, and score spoken answers with confidence intervals.
minutes: 40
gpu: false
objectives:
  - Build an audio instruction sample whose loss mask trains the response and the end token, and never the task tag, the audio placeholders, or the prompt.
  - Describe word-level text-speech interleaving, and check that switches fall on word boundaries and that the speech share follows the schedule.
  - Compute the modality gap per item with a paired bootstrap confidence interval.
  - Compute a pairwise win rate with ties counted as half, and attach a seeded bootstrap confidence interval.
prereqs:
  - 026-word-error-rate
  - 078-spoken-language-models
exercise: 079-speech-llm-training-and-evaluation
---

# Speech LLM training and evaluation

The projector from the previous chapter learned one job: write down the digit it hears. A voice assistant has to do more. It transcribes, translates, answers a question about what a speaker said, and describes a barking dog. Training one projector and one language model for all of that raises two questions. What does a training sample look like, so that one model can tell these jobs apart? And once it is trained, how do you say, with a straight face, that it is any good?

The second question is harder than it looks. Ask the same question as text and as speech, and a model can score differently. Rank two systems with a judge, and a few ties on a small test set can flip the order. This chapter builds the data, then the arithmetic that keeps the evaluation honest.

## What you already know

- A projector turns encoder frames into soft tokens that sit in front of a text prompt, and the language model writes the answer.
- The loss is cross-entropy on the next token, and a label of $-100$ removes a position from it.
- Word error rate compares a hypothesis with a reference, and a paired bootstrap over utterances gives it a confidence interval.

## One sample, many tasks

An audio instruction sample is an audio clip, an instruction, and the answer you want. The *task* decides what the instruction and the answer look like:

| Task | Instruction | Response |
|---|---|---|
| `asr` | Transcribe the audio. | The words spoken. |
| `translate` | Translate the speech into French. | The French sentence. |
| `qa` | What did the speaker ask for? | A short answer. |
| `caption` | Describe the sound. | A sentence about the scene. |

Mixing tasks causes interference, because the labels differ in language, detail, and structure. Qwen-Audio (Chu et al., 2023) conditions the decoder on a sequence of hierarchical tags that name the task, and the authors say shared tags encourage knowledge sharing while specific tags avoid interference. SALMONN (Tang et al., 2023) trains one model on many speech and audio tasks, then adds few-shot activation tuning so that abilities it never trained on, such as translating into an untrained language, switch on. The exercise uses one task tag at the front.

### Lay out one sample

Each sample becomes one row of token ids: the task tag, one placeholder id for each audio soft token, the prompt, the response, and an end token. The placeholder marks where the projector's vectors are spliced in, as in [Spoken language models](/lessons/078-spoken-language-models).

Take the task `asr`, three audio soft tokens, the prompt ids `7 8`, and the response ids `20 21`. With tag id 10, placeholder id 1, and end id 2, the sequence has 9 positions:

| Position | 0 | 1 to 3 | 4 to 5 | 6 to 7 | 8 |
|---|---|---|---|---|---|
| Content | tag | audio | prompt | response | end |
| Id | 10 | 1 1 1 | 7 8 | 20 21 | 2 |
| Takes loss | no | no | no | yes | yes |

```figure
<svg viewBox="0 0 600 170" role="img" aria-label="Nine token positions of one sample. The tag, three audio placeholders, and two prompt tokens are masked out. The two response tokens and the end token, 3 of 9 positions, take loss.">
  <g class="fig-small">
    <rect x="30" y="50" width="54" height="44" class="fill-sunk stroke-muted"/><text x="40" y="76">tag 10</text>
    <rect x="94" y="50" width="54" height="44" class="fill-soft-a stroke-a"/><text x="100" y="76">aud 1</text>
    <rect x="148" y="50" width="54" height="44" class="fill-soft-a stroke-a"/><text x="154" y="76">aud 1</text>
    <rect x="202" y="50" width="54" height="44" class="fill-soft-a stroke-a"/><text x="208" y="76">aud 1</text>
    <rect x="266" y="50" width="54" height="44" class="fill-sunk stroke-muted"/><text x="274" y="76">prm 7</text>
    <rect x="320" y="50" width="54" height="44" class="fill-sunk stroke-muted"/><text x="328" y="76">prm 8</text>
    <rect x="384" y="50" width="54" height="44" class="fill-soft-b stroke-b"/><text x="388" y="76">res 20</text>
    <rect x="438" y="50" width="54" height="44" class="fill-soft-b stroke-b"/><text x="442" y="76">res 21</text>
    <rect x="502" y="50" width="54" height="44" class="fill-soft-b stroke-b"/><text x="514" y="76">end 2</text>
  </g>
  <text x="30" y="125" class="fig-small">mask 0 0 0 0 0 0 1 1 1</text>
  <text x="30" y="30" class="fig-bold">Loss mask of one asr sample</text>
  <text x="30" y="150" class="fig-small">aud = audio placeholder, prm = prompt, res = response. Orange: 3 of 9 positions train the model.</text>
</svg>
Figure 1. The mask is 1 only from the first response token to the end token. The tag, the audio placeholders, and the prompt are inputs, not targets.
```

### Mask only what the model must write

Masked loss, the label shift, and packing are covered in [Supervised fine-tuning and the post-training pipeline](https://rl.ifkash.dev/lessons/053-sft-and-the-post-training-pipeline). The audio-specific part is short. The placeholder positions never take loss, because their ids are filler and the real content is the projector's vectors. Those frames still receive gradients, because the loss on the response flows back through them. The prompt is masked because it is one fixed string per task, so training on it teaches a constant.

In code, build the mask from the length of everything before the response:

```python
head = [TASK_TAGS[task]] + [AUDIO_ID] * audio_len + list(prompt_ids)
ids = head + list(response_ids) + [EOS_ID]
mask = np.zeros(len(ids), dtype=bool)
mask[len(head):] = True
```

> [!WARNING]
> A mask that is True from position 1, or one that forgets the prompt, still gives a falling loss. The model learns the placeholder id and the fixed prompt, which are free, so the average looks healthy while the response gets a smaller share. Count the masked positions in a test.

> [!KEY]
> One sequence layout serves every task. A task tag tells the model which job it has, and the mask trains only the response and the end token.

## Interleaving text and speech

A model trained only on speech units knows little about the world, because transcribed speech is rare next to text. A model trained only on text can't listen. Spirit LM (Nguyen et al., 2024) starts from a pretrained 7B text language model and trains it on one stream that mixes both, with *word-level interleaving* on a small, automatically curated speech-text parallel corpus. In a sentence the model reads some words as text and others as speech units, and the switch happens between two words, never inside one. The aligned words come from forced alignment, as in [Forced alignment and word timestamps](/lessons/046-forced-alignment-and-word-timestamps).

The exercise models the schedule as a two-state chain: the first word is text or speech by a coin flip, and before every later word the modality switches with probability `p_switch`. Because the switch is symmetric, any word is speech with probability one half, for every `p_switch`. A small `p_switch` gives long spans and a large one gives rapid alternation.

### Work one stream by hand

Take four words with their unit counts, `the` with 2, `cat` with 3, `sat` with 2, and `down` with 3, and the schedule text, text, speech, speech. A text word is its string and a speech word is one token per unit. A marker, `[TEXT]` or `[SPEECH]`, appears before the first word and at each change:

```text
[TEXT] the cat [SPEECH] u:sat:0 u:sat:1 u:down:0 u:down:1 u:down:2
```

There are 2 runs and 2 markers. With the schedule speech, text, speech, text, there are 4 runs and 4 markers. Counting markers against runs finds a marker misplaced inside a word.

```python
if modality != previous:
    tokens.append("[TEXT]" if modality == "text" else "[SPEECH]")
    previous = modality
```

> [!KEY]
> Interleaving lets the text model's knowledge reach speech, because the same sentence appears in both forms. Switching at word boundaries keeps each word whole in one modality.

## The modality gap

Interleaving is meant to close a gap. Ask a model the same question as typed text and as a recording, and the scores can differ. That difference is the *modality gap*.

Measure it per item. Take four questions with text scores $0.82, 0.74, 0.90, 0.66$ and speech scores $0.70, 0.71, 0.78, 0.50$. The gap, text minus speech, is $0.12, 0.03, 0.12, 0.16$ and its mean is $0.1075$. Pairing cancels item difficulty: an easy question scores high in both modalities.

A mean from four items needs an interval. The *paired bootstrap* resamples item indices with replacement and recomputes the mean gap, many times. The 2.5th and 97.5th percentiles form a 95% interval, as in [Word error rate](/lessons/026-word-error-rate):

$$
\hat{g}^{(b)} = \frac{1}{n} \sum_{i=1}^{n} g_{k_i^{(b)}}, \qquad k_i^{(b)} \sim \text{Uniform}\{1, \dots, n\}
$$

Read it aloud: draw $n$ item numbers $k$ at random with replacement, average the gaps $g$ at those items, and repeat for draw $b$. The same $k$ picks the text score and the speech score, which is why you resample the gaps and not the two score lists on their own.

```python
gaps = text - speech
picks = rng.integers(0, n, size=(n_boot, n))
means = gaps[picks].mean(axis=1)
low, high = np.percentile(means, [2.5, 97.5])
```

> [!WARNING]
> Resampling the text scores and the speech scores independently breaks the pairing and inflates the interval. If every item has the same gap, the correct interval has zero width, and an unpaired one does not.

## Benchmarks for audio in and audio out

Three benchmarks cover this ground. The table names them without quoting scores.

| Benchmark | What it tests | How it scores |
|---|---|---|
| AIR-Bench (Yang et al., 2024) | Understanding of speech, natural sound, and music. A foundation set of 19 tasks with about 19,000 single-choice questions, and a chat set of about 2,000 open-ended instances. | Choice accuracy for the foundation set. A GPT-4-based judge scores the chat set, using the audio's metadata. |
| VoiceBench (Chen et al., 2024) | LLM-based voice assistants given spoken instructions that vary in speaker, environment, and content. | Described in the paper as a multi-faceted evaluation. |
| SD-Eval (Ao et al., 2024) | Spoken dialogue beyond the words: emotion, accent, age, and background sound. About 7,300 utterances. | BLEU and ROUGE, human ratings, and LLM-based metrics. |

These test audio in: does the model understand what it heard? When a system also answers in speech, you can transcribe its audio with an ASR model and score the transcript, which adds the recognizer's errors to every score.

## Win rate with ties

Open-ended answers have no single right string, so a judge, human or model, sees two answers and picks one. System A beats B, loses to B, or ties. The *win rate* of A counts a tie as half a win:

$$
\text{win rate} = \frac{W + \tfrac{1}{2} T}{W + T + L}
$$

Read it aloud: wins plus half the ties, over all judgments. Suppose a judge compares A and B on 20 prompts and gives 9 wins, 6 ties, and 5 losses. The win rate is $(9 + 3) / 20 = 0.6$. Dropping the ties gives $9 / 14 = 0.643$, which overstates A, because six prompts where B did as well as A disappear.

Map each judgment to 1, 0.5, or 0, resample the 20 scores 2,000 times with seed 0, and take percentiles of the means. The 95% interval is $[0.425, 0.775]$. It contains 0.5, so 20 prompts don't separate A from B.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="Histogram of 2000 bootstrap win rates, peaked near 0.6. The 95 percent interval runs from 0.425 to 0.775. A dashed mark at 0.643 shows the win rate with ties dropped.">
  <line x1="60" y1="190" x2="540" y2="190" class="stroke-muted"/>
  <g>
    <rect x="60" y="189" width="40" height="1"class="fill-soft-a stroke-a"/><rect x="100" y="188" width="40" height="2"class="fill-soft-a stroke-a"/>
    <rect x="140" y="177" width="40" height="13"class="fill-soft-a stroke-a"/><rect x="180" y="157" width="40" height="33"class="fill-soft-a stroke-a"/>
    <rect x="220" y="125" width="40" height="65"class="fill-soft-a stroke-a"/><rect x="260" y="40" width="40" height="150"class="fill-soft-a stroke-a"/>
    <rect x="300" y="98" width="40" height="92"class="fill-soft-a stroke-a"/><rect x="340" y="123" width="40" height="67"class="fill-soft-a stroke-a"/>
    <rect x="380" y="167" width="40" height="23"class="fill-soft-a stroke-a"/><rect x="420" y="170" width="40" height="20"class="fill-soft-a stroke-a"/>
    <rect x="460" y="181" width="40" height="9"class="fill-soft-a stroke-a"/><rect x="500" y="189" width="40" height="1"class="fill-soft-a stroke-a"/>
  </g>
  <line x1="160" y1="30" x2="160" y2="190" class="stroke-b"/><line x1="440" y1="30" x2="440" y2="190" class="stroke-b"/>
  <line x1="414" y1="30" x2="414" y2="190" class="stroke-bad dashed"/>
  <text x="95" y="24" class="fig-small">low 0.425</text><text x="400" y="24" class="fig-small">high 0.775</text>
  <text x="418" y="46" class="fig-small">ties dropped: 0.643</text>
  <text x="60" y="210" class="fig-small">0.30</text><text x="180" y="210" class="fig-small">0.45</text>
  <text x="300" y="210" class="fig-small">0.60</text><text x="420" y="210" class="fig-small">0.75</text>
  <text x="200" y="232" class="fig-small">bootstrap win rate of A over B, 2000 resamples</text>
</svg>
Figure 2. With 9 wins, 6 ties, and 5 losses, the win rate is 0.6 and its 95% interval is wide. Dropping the ties moves the point estimate to 0.643, a value the honest procedure doesn't produce.
```

> [!TIP]
> A model judge often favors the answer shown first. Run each pair in both orders and score a preference only when the two verdicts agree, or count disagreement as a tie. Then the swap test in the exercise holds: swapping A and B maps the rate to one minus itself.

Preference optimization trains a model on pairs like these. For the loss and its variants, see [Direct preference optimization](https://rl.ifkash.dev/lessons/058-direct-preference-optimization). For speech, every pair needs generated audio and a judge calibrated against humans.

> [!RECAP]
> - A sample is a task tag, audio placeholders, a prompt, a response, and an end token, in that order.
> - The loss mask is 1 only on the response and the end token. The placeholders and the prompt are inputs.
> - Word-level interleaving switches modality between words, with a marker at each switch.
> - The modality gap is a per-item difference, and its bootstrap resamples items in pairs.
> - A win rate counts ties as half and keeps them in the denominator, and a bootstrap gives its interval.

## Build it step by step

Build the functions in the order the checks test them.

1. **`build_audio_instruction(task, audio_len, prompt_ids, response_ids)`** returns `(ids, mask)`. Build the head, then the response, then the end id. The common mistake is to mask from position 1, which trains the audio placeholders and the prompt.
2. **`make_schedule(n_words, p_switch, seed)`** returns one modality per word. Draw one coin flip for the first word, then one number per later word, and switch when it is below `p_switch`.
3. **`interleave(words, schedule)`** returns the token stream. Emit a marker only when the modality differs from the previous word's. The common mistake is a marker before every word.
4. **`speech_fraction(schedule)`** returns the share of speech words.
5. **`modality_gap(text_scores, speech_scores, n_boot, seed)`** returns `(gaps, mean, low, high)`. Draw the index matrix once, and index the gaps with it.
6. **`win_rate(judgments, n_boot, seed)`** returns `(rate, low, high)`. Keep every judgment, and map a tie to 0.5.

## Where this breaks

- **A judge that disagrees with people.** A model judge has its own preferences, such as longer answers. Compare its verdicts with human verdicts on a sample before trusting a win rate.
- **A gap that hides in the average.** Break the gap down by speaker, noise, and task.
- **Too few prompts.** The interval in this chapter is wide at 20 judgments. Add prompts before you claim a ranking.

## The exercise

You write `build_audio_instruction`, `make_schedule`, `interleave`, `speech_fraction`, `modality_gap`, and `win_rate`. Press **Run** to print the worked sample and its mask, one interleaved stream, a gap with its interval, and the win rate of 9 wins, 6 ties, and 5 losses.

The checks verify the layout and the mask, which must exclude the tag, the audio placeholders, and the prompt. They test that `interleave` marks every switch at a word boundary, that `make_schedule` is deterministic for a seed and follows `p_switch`, and that the speech share is about one half over many seeds. They check the gap on the worked scores and that its bootstrap is paired, and that ties count as half a win. The last check tests the win-rate interval's width and that swapping the two systems flips it.

## Further reading

- [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/abs/2311.07919), Yunfei Chu and colleagues, 2023. The abstract describes hierarchical tags for multi-task training.
- [SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/abs/2310.13289), Changli Tang and colleagues, 2023. The abstract introduces few-shot activation tuning.
- [Spirit LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755), Tu Anh Nguyen and colleagues, 2024. The abstract describes word-level interleaving.
- [AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension](https://arxiv.org/abs/2402.07729), Qian Yang and colleagues, ACL 2024. It defines the foundation and chat benchmarks.
- [VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196), Yiming Chen and colleagues, 2024.
- [SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words](https://arxiv.org/abs/2406.13340), Junyi Ao and colleagues, NeurIPS 2024.
