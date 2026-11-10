---
title: Spoken language understanding
slug: 065-spoken-language-understanding
part: "Part 8 — Beyond recognition and synthesis"
summary: Score intents and slots, merge a dialogue state, and compute sentence BLEU and a character F-score.
minutes: 45
gpu: false
objectives:
  - Compute intent accuracy and micro slot F1 when a one-best hypothesis drops a slot.
  - Choose an intent by summing n-best probability mass rather than trusting the first hypothesis.
  - Compute sentence BLEU with smoothing and the brevity penalty, and a character unigram F-score.
  - Merge dialogue-state slots across two turns so an earlier value survives.
prereqs:
  - 026-word-error-rate
  - 025-n-gram-language-models
  - 038-whisper
exercise: 065-spoken-language-understanding
---

# Spoken language understanding

You tell a speaker to set an alarm for 07:00 in the kitchen. A recognizer writes down the words. One missing phrase, "in the kitchen", still leaves a sentence that looks almost right. The thing you asked for was not the sentence. It was an alarm, at that time, in that room.

Spoken language understanding turns the audio, or the transcript, into that request. The *intent* is the act, here `SetAlarm`. The *slots* are the details, a time and a room. A dialogue does not clear the form on every turn: the second answer fills the room and leaves the time in place. Speech translation has a different output, a sentence in another language, and you score that sentence by how its words and characters overlap a reference.

A pipeline does the work in stages, so an early mistake changes the intent, the slots, or the translation. One model can also map speech straight to the intent, or straight to the translation. This chapter measures both kinds of damage on one alarm request.

## What you already know

- [Word error rate](/lessons/026-word-error-rate) counts insertions, deletions, and substitutions against a reference transcript.
- An [n-best list](/lessons/025-n-gram-language-models) is a few transcripts ranked by score, the list a decoder keeps after search.
- Whisper's `<|translate|>` token, from [Whisper and large-scale recognition](/lessons/038-whisper), asks one model for English text instead of a transcript in the spoken language.
- A probability is the exponential of a log probability. Adding log probabilities multiplies the probabilities.
- The F1 score is the harmonic mean of precision and recall, $2PR/(P+R)$.

## An intent and the slots that fill it

Picture the request as a short form. The intent is the title. Each slot is a blank with a name and a value: `time` is `07:00`, and `room` is `kitchen`.

SLURP labels home-assistant commands this way. Bastianelli, Vanzo, Swietojanski, and Rieser (2020) release 72,277 recordings across 18 scenarios, with 46 actions and 56 entity types. An *action* is the intent. An *entity* is a slot name plus the words that fill it. A pipeline runs a recognizer and then a text classifier, so a deleted word never becomes a slot. An end-to-end model reads the waveform and emits the form directly.

Take the reference "set an alarm for 07:00 in the kitchen", intent `SetAlarm`, slots `{time: 07:00, room: kitchen}`. The one-best transcript drops "in the kitchen". The parser still returns `SetAlarm`, with only `{time: 07:00}` left.

*Intent accuracy* is the fraction of utterances whose predicted intent equals the reference. Here that fraction is 1. The room is gone, and accuracy cannot say so, because it never looks at the slots.

Slot F1 does. A *slot-value pair* is one name with one value, such as `(room, kitchen)`. *Micro* averaging pools true positives, false positives, and false negatives over every utterance, then computes one precision, one recall, and one F1. This utterance has two reference pairs and one predicted pair. The time matches:

$$
P = \frac{1}{1+0} = 1, \qquad
R = \frac{1}{1+1} = \frac{1}{2}, \qquad
F_1 = \frac{2 \cdot 1 \cdot 1/2}{1 + 1/2} = \frac{2}{3}
$$

Precision is the share of predicted pairs that are right, recall is the share of reference pairs you found, and F1 is their harmonic mean. $2/3$ is about 0.667. Figure 1 draws the four scores. The tall bars ignore the missing room.

```figure
<svg viewBox="0 0 620 246" role="img" aria-label="Four bars: intent accuracy 1, slot precision 1, slot recall 1/2, and slot F1 2/3">
  <line x1="96" y1="50" x2="540" y2="50" class="stroke-grid"/>
  <line x1="96" y1="120" x2="540" y2="120" class="stroke-grid"/>
  <line x1="96" y1="50" x2="96" y2="190" class="stroke-muted"/>
  <line x1="96" y1="190" x2="560" y2="190" class="stroke-muted"/>
  <rect x="108" y="50" width="78" height="140" class="fill-a"/>
  <rect x="222" y="50" width="78" height="140" class="fill-a"/>
  <rect x="336" y="120" width="78" height="70" class="fill-bad"/>
  <rect x="450" y="96.667" width="78" height="93.333" class="fill-b"/>
  <text x="147" y="42" text-anchor="middle">1</text>
  <text x="261" y="42" text-anchor="middle">1</text>
  <text x="375" y="112" text-anchor="middle">1/2</text>
  <text x="489" y="88" text-anchor="middle">2/3</text>
  <text x="88" y="54" text-anchor="end" class="fig-small">1</text>
  <text x="88" y="124" text-anchor="end" class="fig-small">0.5</text>
  <text x="88" y="194" text-anchor="end" class="fig-small">0</text>
  <text x="147" y="210" text-anchor="middle" class="fig-small">intent</text>
  <text x="261" y="210" text-anchor="middle" class="fig-small">precision</text>
  <text x="375" y="210" text-anchor="middle" class="fig-small">recall</text>
  <text x="489" y="210" text-anchor="middle" class="fig-small">slot F1</text>
  <text x="300" y="236" text-anchor="middle">score (0 to 1)</text>
</svg>
Figure 1. Intent accuracy and slot precision stay at 1. Recall falls to 1/2, so slot F1 is 2/3. Bar heights are those four scores.
```

A second utterance that gets both pairs right adds 2 true positives. The pooled counts are then 3, 0, and 1, so precision stays 1, recall is $3/4$, and F1 is $6/7$. Averaging the two utterance scores, $2/3$ and $1$, gives $5/6$ instead. Pool the counts, then divide once:

```python
tp += len(set(pred.items()) & set(gold.items()))
fp += len(set(pred.items()) - set(gold.items()))
fn += len(set(gold.items()) - set(pred.items()))
```

> [!KEY]
> Intent accuracy can stay at 1 while a slot is missing. Slot F1 counts slot-value pairs, so the deleted room pulls this utterance from 1 down to $2/3$.

## Add the hypotheses that share an intent

The one-best line is the transcript with the highest probability, and it can name the wrong intent. The following table lists three hypotheses for the same recording, best first:

| Hypothesis | Intent | Slots | Probability |
|---|---|---|---|
| play an alarm | `PlayMusic` | empty | 0.40 |
| set an alarm for 07:00 | `SetAlarm` | time 07:00 | 0.35 |
| set an alarm for 07:00 in the kitchen | `SetAlarm` | time 07:00, room kitchen | 0.25 |

A scorer that trusts the first row returns `PlayMusic`. The natural logs are $\ln 0.40 \approx -0.9163$, $\ln 0.35 \approx -1.0498$, and $\ln 0.25 \approx -1.3863$. Sum probabilities, not logs. Adding logs multiplies, and $0.35 \times 0.25 = 0.0875$ is less than $0.40$, so the product still picks `PlayMusic`.

The masses are $0.40$ for `PlayMusic` and $0.35 + 0.25 = 0.60$ for `SetAlarm`. Figure 2 puts a dashed line at the one-best probability. The summed `SetAlarm` bar rises past it.

```figure
<svg viewBox="0 0 640 268" role="img" aria-label="Hypothesis probabilities 0.40, 0.35, and 0.25, then intent masses 0.40 for PlayMusic and 0.60 for SetAlarm">
  <line x1="78" y1="68" x2="560" y2="68" class="stroke-grid"/>
  <line x1="78" y1="108" x2="560" y2="108" class="stroke-grid"/>
  <line x1="78" y1="148" x2="560" y2="148" class="stroke-grid"/>
  <line x1="78" y1="48" x2="78" y2="188" class="stroke-muted"/>
  <line x1="78" y1="188" x2="560" y2="188" class="stroke-muted"/>
  <line x1="92" y1="108" x2="516" y2="108" class="stroke-bad dashed"/>
  <rect x="92" y="108" width="52" height="80" class="fill-bad"/>
  <rect x="162" y="118" width="52" height="70" class="fill-c"/>
  <rect x="232" y="138" width="52" height="50" class="fill-c"/>
  <rect x="360" y="108" width="64" height="80" class="fill-bad"/>
  <rect x="452" y="68" width="64" height="120" class="fill-c"/>
  <text x="118" y="100" text-anchor="middle" class="fig-small">0.40</text>
  <text x="188" y="110" text-anchor="middle" class="fig-small">0.35</text>
  <text x="258" y="130" text-anchor="middle" class="fig-small">0.25</text>
  <text x="392" y="100" text-anchor="middle" class="fig-small">0.40</text>
  <text x="484" y="60" text-anchor="middle" class="fig-small">0.60</text>
  <text x="70" y="52" text-anchor="end" class="fig-small">0.7</text>
  <text x="70" y="72" text-anchor="end" class="fig-small">0.6</text>
  <text x="70" y="112" text-anchor="end" class="fig-small">0.4</text>
  <text x="70" y="152" text-anchor="end" class="fig-small">0.2</text>
  <text x="70" y="192" text-anchor="end" class="fig-small">0</text>
  <text x="118" y="206" text-anchor="middle" class="fig-small">H1</text>
  <text x="188" y="206" text-anchor="middle" class="fig-small">H2</text>
  <text x="258" y="206" text-anchor="middle" class="fig-small">H3</text>
  <text x="392" y="206" text-anchor="middle" class="fig-small">PlayMusic</text>
  <text x="484" y="206" text-anchor="middle" class="fig-small">SetAlarm</text>
  <text x="188" y="228" text-anchor="middle" class="fig-small">one hypothesis</text>
  <text x="438" y="228" text-anchor="middle" class="fig-small">summed by intent</text>
  <text x="320" y="254" text-anchor="middle">probability</text>
</svg>
Figure 2. Each hypothesis bar is one probability. H1 is the tallest of those three and the wrong intent. SetAlarm's two bars sum to 0.60 and pass the dashed line at 0.40. Heights are the probabilities divided by 0.70.
```

$$
M(y) = \sum_{h : \operatorname{intent}(h) = y} \exp(\ell_h), \qquad
\hat{y} = \operatorname*{arg\,max}_{y} M(y)
$$

$\ell_h$ is the log probability of hypothesis $h$. $\exp(\ell_h)$ turns it back into a probability, $M(y)$ adds those probabilities inside one intent, and you pick the intent with the largest sum:

```python
mass[intent] = mass.get(intent, 0.0) + math.exp(log_p)
```

> [!WARNING]
> These three logs sit near $-1$, so `math.exp` is safe. A decoder score can be a large negative number. Subtract the largest log before you exponentiate. The shared factor does not change the winning intent.

> [!KEY]
> The one-best intent can be the wrong one. Sum the probability of every hypothesis that names an intent, and take the larger mass.

## Merge the next turn into the form

The same request can arrive in two turns. The user first says "set an alarm for 07:00". The system asks which room. The user says "in the kitchen", and the system confirms the room.

The *dialogue state* is the form so far: the intent, the slots filled so far, and the system act from this turn. A system act is a small record, `request` or `confirm`, plus the slot it talks about. The update takes the previous state, one user act, and one system act, and returns the next state.

Start empty. After the first turn the state holds `SetAlarm`, `{time: 07:00}`, and `{act: request, slot: room}`. The second user act only names the room. The time has to stay. $s$ is the slot dict you already have, and $u$ is the user act's slots:

$$
s'[k] =
\begin{cases}
u[k] & \text{when the user act names slot } k \\
s[k] & \text{otherwise}
\end{cases}
$$

A named slot takes the user's value. A slot the user left out stays. When the user act carries an intent, that intent replaces the stored one, and the system act is stored on the state. Copy the dict before you update it, or you rewrite the previous turn:

```python
slots = dict(state["slots"])
slots.update(user_act["slots"])
```

> [!TIP]
> Return a fresh dict. The caller may keep the previous turn and compare the two.

A tracker that keeps only this turn's slots leaves `{room: kitchen}`. The alarm no longer has a time. That mistake shows on the second turn. The first turn started empty, so a replacement and a merge look the same.

## Score one translated sentence

The Spanish source is "pon la alarma de la cocina". The reference translation is "set the kitchen alarm", and the hypothesis is "set an alarm". The hypothesis is shorter, and it drops the kitchen. BLEU and chrF compare the hypothesis with the reference. Neither score reads the Spanish.

Papineni, Roukos, Ward, and Zhu (2002) define BLEU as a brevity penalty times the geometric mean of *modified n-gram precisions*. Clipping counts each hypothesis n-gram at most as often as it occurs in the reference. Their baseline uses orders 1 through 4 with equal weights, on a whole test set. Here the test set is one sentence, the orders are 1 and 2, and each weight is $1/2$.

Split on spaces. The hypothesis is `set`, `an`, `alarm`, so $c = 3$ words. The reference is `set`, `the`, `kitchen`, `alarm`, so $r = 4$ words. Clipped unigram hits are `set` and `alarm`, so $p_1 = 2/3$. Both bigrams, `(set, an)` and `(an, alarm)`, miss. A geometric mean with a zero in it is zero, so one sentence needs smoothing. Chen and Cherry (2014) call the next rule smoothing 2, attributed to Lin and Och. For every order $n \ge 2$, add 1 to the clipped hits and add 1 to the hypothesis n-gram count. Unigrams stay raw:

$$
p_1 = \frac{2}{3}, \qquad p_2 = \frac{0+1}{2+1} = \frac{1}{3}
$$

The geometric mean is $\sqrt{(2/3)\cdot(1/3)} = \sqrt{2}/3 \approx 0.4714$. Because $c < r$, the penalty is $\exp(1 - r/c)$. Papineni uses 1 only when the hypothesis is longer than the reference:

$$
BP = \exp\left(-\frac{1}{3}\right) \approx 0.7165, \qquad
\mathrm{BLEU} = BP \cdot \frac{\sqrt{2}}{3} \approx 0.3378
$$

Dropping the penalty reports $0.4714$. That is the geometric mean alone, and it is the wrong BLEU on this shorter hypothesis:

```python
bp = 1.0 if c > r else math.exp(1.0 - r / c)
bleu = bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))
```

chrF, from Popović (2015), is an F-score on character n-grams. The published score averages orders 1 through 6. Counting the space as a character did not improve agreement with human rankings, so spaces stay out. With $\beta = 1$ the score is the ordinary harmonic mean. This exercise uses order 1 so you can count every character.

"set an alarm" without spaces is 10 characters, with `a` three times. "set the kitchen alarm" without spaces is 18 characters. Clipping by the reference leaves 9 shared characters, because `a` can match only twice:

$$
P_{\mathrm{char}} = \frac{9}{10}, \qquad
R_{\mathrm{char}} = \frac{1}{2}, \qquad
\mathrm{chrF} = \frac{9}{14} \approx 0.6429
$$

## A cascade puts text in the middle

A *cascade* recognizes the speech, translates the text, and, for speech-to-speech, synthesizes the result. SeamlessM4T is one model that can translate speech to text, translate speech to speech, and recognize speech, for up to 100 languages (Seamless Communication et al., 2023). Its introduction sets that single model against a cascade of recognition, text translation, and synthesis. Whisper's translate task is the smaller end-to-end case: one model, speech in, English text out.

Text between those stages stores neither pitch nor timing. The SeamlessM4T introduction says speech carries expressive detail beyond the words, so a synthesizer that sees only the sentence has to choose both. An end-to-end speech-to-speech model can use the audio, and it can also carry a mistake through with no transcript where the missing room would show.

> [!WARNING]
> On one sentence a zero n-gram count is common, and an unsmoothed geometric mean becomes 0. Use smoothing 2, and do not rank two systems by a single sentence.

## Where this breaks

String match is brittle: `7:00` and `07:00` are one time to a person and two values to `slot_f1`, so normalize before you compare. An n-best list can agree on a wrong intent when every hypothesis shares an early mistake, because the mass then piles onto that intent. BLEU and chrF reward overlap, not whether the alarm would be set in the kitchen, and character unigrams ignore order. The published chrF keeps orders through 6 so that order can change the score.

> [!KEY]
> Intent accuracy misses a deleted room that slot F1 catches. The n-best mass can overturn a wrong one-best intent, and a dialogue state keeps the time when the next turn adds the room. Score a translation of that request with BLEU, brevity penalty included, and with a character F-score.

> [!RECAP]
> - Pool slot-value pairs, then compute one F1. The missing room scores $2/3$, not $1$.
> - Sum $\exp(\ell_h)$ inside each intent. The totals here are $0.60$ for `SetAlarm` and $0.40$ for `PlayMusic`.
> - Sentence BLEU uses orders 1 and 2, smoothing 2 on bigrams, and $\exp(1-r/c)$ because $c = 3$ and $r = 4$.
> - chrF here is character unigram F1 with spaces removed, $9/14$ on this pair.
> - Copy the slot dict and update it. Replacing the state on the second turn drops `time`.

## Build it step by step

1. **`intent_accuracy(pred, gold)`** takes two equal-length lists of intent strings and returns the fraction that match. `["SetAlarm"]` against `["SetAlarm"]` returns `1.0`.

2. **`slot_f1(pred, gold)`** takes two lists of slot dicts. Pool `(name, value)` pairs into one precision, one recall, and one F1. The missing-room utterance returns $2/3$. Returning $1$ because the intent matched is the wrong score. When every count is zero, return `1.0`.

3. **`nbest_intent(hypotheses, log_probs)`** takes dicts with an `"intent"` key and natural logs of the same length. Add `math.exp(log_p)` per intent and return the largest total. The planted list returns `"SetAlarm"`. `hypotheses[0]["intent"]` returns `"PlayMusic"`.

4. **`bleu(hypothesis, reference)`** splits on whitespace, clips n-gram counts, applies smoothing 2 at order 2, uses weights $1/2$, and multiplies by the brevity penalty. `"set an alarm"` against `"set the kitchen alarm"` returns about `0.3378`. Leaving the penalty off returns about `0.4714`.

5. **`chrf(hypothesis, reference)`** drops spaces, counts character unigrams, clips by the reference, and returns the F1 at $\beta = 1$. This pair returns $9/14$.

6. **`update_dialogue_state(state, user_act, system_act)`** returns a fresh dict and does not edit `state`. Copy the slots, update them from the user act, replace the intent when the user act has one, and store `system_act`. After the kitchen turn the slots still contain `time`. Keeping only this turn drops it.

## The exercise

You write `intent_accuracy`, `slot_f1`, `nbest_intent`, `bleu`, `chrf`, and `update_dialogue_state`. The checks use this chapter's alarm request: intent accuracy `1` on the one-best pair, slot F1 $2/3$ with the room missing and $6/7$ pooled with a perfect utterance, n-best intent `SetAlarm`, sentence BLEU with the brevity penalty, character F1 $9/14$, and a two-turn state that still holds `time`. Press **Run** to print the scores and plot the three hypothesis probabilities.

## Further reading

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/), Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu, ACL 2002. Sections 2.1 and 2.3 define modified precision and the brevity penalty.
- [A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU](https://aclanthology.org/W14-3346/), Boxing Chen and Colin Cherry, WMT 2014. Smoothing 2 adds one count for orders 2 and up.
- [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/), Maja Popović, WMT 2015. This paper defines the character F-score, drops spaces, and averages orders 1 through 6.
- [SLURP: A Spoken Language Understanding Resource Package](https://arxiv.org/abs/2011.13205), Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser, EMNLP 2020. Tables 2 and 4 list the recordings, scenarios, actions, and entity types cited here.
- [SeamlessM4T: Massively Multilingual and Multimodal Machine Translation](https://arxiv.org/abs/2308.11596), Seamless Communication and colleagues, 2023. The introduction contrasts one multilingual model with a cascade.
- [CS224S: Spoken Language Processing](https://web.stanford.edu/class/cs224s/), Andrew Maas, Stanford University, Spring 2025. The course page describes spoken language technology with an emphasis on dialog and conversational systems.
