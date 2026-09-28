---
title: Language models and word error rate
slug: 14-language-models-and-word-error-rate
part: "Part 3 — Classical recognition"
summary: Combine acoustic scores with a Kneser-Ney language model, and score transcripts with word error rate from a Levenshtein alignment.
minutes: 70
gpu: false
objectives:
  - Explain the noisy-channel view of recognition, with its language model scale and word insertion penalty.
  - Build an interpolated Kneser-Ney bigram model and measure its perplexity on held-out text.
  - Align a hypothesis against a reference with Levenshtein distance, and compute WER and CER at the corpus level.
  - "Describe how HMM-GMM systems composed four transducers: H, C, L, and G."
prereqs:
  - 11-phonetics-text-and-voice-activity
  - 13-hidden-markov-models
exercise: 14-language-models-and-word-error-rate
---

# Language models and word error rate

"Write to mom" and "right two mom" sound the same. No acoustic model, however
good, can tell them apart, because the difference isn't in the sound. It's in
what people are likely to say. A recognizer needs a second model, one that
knows which word sequences are plausible. That's a **language model** (LM).

This lesson covers both ends of a recognizer that the acoustic models of the
last two lessons don't. At the input side, you build a bigram language model
with Kneser-Ney smoothing, the method behind the n-gram models that decoded
speech for decades, and combine it with acoustic scores. At the output side,
you build the metric every recognizer is judged by: **word error rate** (WER),
computed from a minimum-edit alignment.

Last, you see how HMM-GMM systems glued the pieces into one search graph.

## What you already know

- An HMM gives $\log P(O \mid W)$, the log-likelihood of the frames $O$ under
  the model of a word sequence $W$, as *Hidden Markov models* showed.
- A pronunciation dictionary maps each word to a sequence of phones, such as
  ARPAbet `R AY T` for "write", as *Phonetics, text, and voice activity*
  showed.
- Dynamic time warping fills a table where each cell is the cheapest way to
  reach it from a neighbor.
- $\log_2 x = \ln x / \ln 2$, and $2^{\log_2 x} = x$.

## The noisy-channel view

Think of speech as a message sent through a noisy channel. Someone chose a
word sequence $W$, and the channel turned it into the sound $O$ you received.
To decode, pick the $W$ that was most likely sent:

$$
\hat W = \operatorname*{arg\,max}_W P(W \mid O) = \operatorname*{arg\,max}_W P(O \mid W)\, P(W)
$$

Read it aloud: the best transcript is the one that makes the sound likely,
$P(O \mid W)$, the **acoustic model**, and that is itself likely,
$P(W)$, the **language model**. Bayes' rule turns one into the other, and
$P(O)$ drops out because it's the same for every $W$.

In practice the two scores are badly mismatched. An HMM multiplies one
probability per 10 ms frame and treats frames as independent, so its log
scores are overconfident and huge. The language model contributes one term
per word. Systems scale the language model, and add a per-word term:

$$
\operatorname{score}(W) = \log P(O \mid W) + \lambda \log P(W) + \beta\, |W|
$$

Read it aloud: the acoustic log score, plus the language model log score
times a scale $\lambda$, plus $\beta$ for each of the $|W|$ words. A negative
$\beta$ is a **word insertion penalty**, which stops the decoder from filling
gaps with short words. Both $\lambda$ and $\beta$ are tuned on held-out data.
In the exercise, `rescore` applies this score to an **n-best list**, the few
best transcripts a recognizer proposes.

> [!KEY]
> A recognizer picks the transcript with the best acoustic score plus a scaled
> language model score. The scale $\lambda$ and the insertion penalty $\beta$
> are tuned on held-out data, not derived.

## N-gram language models

A language model assigns a probability to a whole sentence. The chain rule
splits it into one prediction per word, and an **n-gram** model assumes each
word depends only on the $n - 1$ words before it. A **bigram** model looks at
one previous word, the **history** $h$:

$$
P(w_1 \dots w_N) \approx \prod_{i=1}^{N} P(w_i \mid w_{i-1})
$$

Read it aloud: the probability of the sentence is the product of the
probability of each word given the one before it. Pad every sentence with a
start marker `<s>`, which is only ever a history, and an end marker `</s>`,
which is predicted like a word, so the model also learns how sentences end.

The obvious estimate is a ratio of counts,
$P(w \mid h) = C(h\,w) / C(h)$, where $C(h\,w)$ counts the bigram and $C(h)$
counts every bigram that starts with $h$. That's the **maximum-likelihood**
estimate, and it fails on any pair it never saw: probability 0, so any test
sentence containing that pair gets probability 0.

## Kneser-Ney smoothing

**Smoothing** moves a little probability from seen bigrams to unseen ones.
**Absolute discounting** subtracts a fixed amount $d$, usually about 0.75,
from every seen count. The freed mass goes to a backup distribution over
single words.

Kneser and Ney's insight was about that backup. A word like "Francisco" is
common, but only after "San". As a backup, it should be rare, because an unseen
context is unlikely to be followed by it. So the backup counts how many
*different* words a word follows, not how often it occurs. Interpolated
Kneser-Ney for bigrams is the following:

$$
\begin{aligned}
P_{\text{KN}}(w \mid h) &= \frac{\max\big(C(h\,w) - d,\ 0\big)}{C(h)} + \lambda(h)\, P_{\text{cont}}(w) \\
\lambda(h) &= \frac{d\, N_{1+}(h\,\bullet)}{C(h)}
\end{aligned}
$$

$$
P_{\text{cont}}(w) = \frac{N_{1+}(\bullet\, w)}{N_{1+}(\bullet\,\bullet)}
$$

Read it aloud: take the discounted count of the bigram as a share of the
history's count. Add a backup term, where $\lambda(h)$ is exactly the mass the
discount removed: $d$ times $N_{1+}(h\,\bullet)$, the number of distinct words
seen after $h$, over $C(h)$. The **continuation probability**
$P_{\text{cont}}(w)$ is the number of distinct words $w$ follows,
$N_{1+}(\bullet\, w)$, over the number of distinct bigram types. Summed over
every word, the first term gives $1 - \lambda(h)$ and the second gives
$\lambda(h)$, so the distribution sums to 1. In code, all three counts come
from the bigram `Counter`:

```python
followers = sum(1 for (h, _) in counts if h == history)   # N1+(h .)
predecessors = Counter(w for (_, w) in counts)             # N1+(. w), for every w
```

### Work the example by hand

Take a three-sentence corpus: "a b", "a c", and "b c". With markers, the
bigram types are (`<s>` a) twice, (a b), (b `</s>`), (a c), (c `</s>`) twice,
(`<s>` b), and (b c). That's 7 distinct types.

Continuation probabilities come from counting distinct predecessors: "a"
follows only `<s>`, so $P_{\text{cont}}(a) = 1/7$. Each of "b", "c", and
`</s>` follows two different words, so each gets $2/7$.

Now condition on the history "a". It occurs twice, $C(a) = 2$, followed by
two distinct words, so $\lambda(a) = 0.75 \times 2 / 2 = 0.75$:

- $P(b \mid a) = (1 - 0.75)/2 + 0.75 \times 2/7 = 0.125 + 0.2143 = 0.3393$.
- $P(c \mid a) = 0.3393$, by the same arithmetic.
- $P(\texttt{</s>} \mid a) = 0 + 0.75 \times 2/7 = 0.2143$.
- $P(a \mid a) = 0 + 0.75 \times 1/7 = 0.1071$.

The four sum to $0.3393 + 0.3393 + 0.2143 + 0.1071 = 1.0$. The unseen
"`</s>` after a" gets twice the probability of the unseen "a after a",
because `</s>` ends more kinds of context.

Modified Kneser-Ney (Chen and Goodman, 1998) uses three discounts, for counts
of 1, 2, and 3 or more. KenLM (Heafield, 2011) builds such models at scale.

## Perplexity

To compare language models, measure how surprised each one is by held-out
text. **Perplexity** is the inverse probability of the text, normalized per
token:

$$
\begin{aligned}
\text{PP} &= P(w_1 \dots w_N)^{-1/N} = 2^{H} \\
H &= -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_{i-1})
\end{aligned}
$$

Read it aloud: $H$ is the average number of bits the model needs per token,
and perplexity is 2 to that power. A model that picks uniformly among 10
digits has perplexity 10, so perplexity reads as "the number of equally likely
choices the model is torn between". Lower is better. Count every `</s>` in
$N$, but no `<s>`, since it's never predicted.

For the sentence "a c" under the worked model, three tokens are predicted:
$P(a \mid \texttt{<s>}) = 0.4881$, $P(c \mid a) = 0.3393$, and
$P(\texttt{</s>} \mid c) = 0.7321$. Their product is 0.1212, and
$0.1212^{-1/3} = 2.020$.

On the exercise's voice-assistant corpus, a Kneser-Ney bigram trained on 400
sentences has held-out perplexity 2.25. Add-one smoothing, which adds 1 to
every count, scores 3.02.

> [!KEY]
> Kneser-Ney backs off to how many contexts a word completes, not how often
> it occurs. Perplexity compares models only when the vocabulary and the
> tokenization are the same.

## Word error rate

To score a transcript, align it with the reference so that the number of
edits is as small as possible. There are three kinds of edit: a
**substitution** (S) replaces a word, a **deletion** (D) drops a reference
word, and an **insertion** (I) adds a word that isn't in the reference. Then

$$
\text{WER} = \frac{S + D + I}{N}
$$

where $N$ is the number of reference words. The minimum number of edits is
the **Levenshtein distance**, and it's a dynamic program like DTW. With
$E(i, j)$ the distance between the first $i$ reference words and the first
$j$ hypothesis words:

$$
\begin{aligned}
E(i, 0) &= i \\
E(0, j) &= j \\
E(i, j) &= \min\big\{E(i-1, j) + 1,\ E(i, j-1) + 1,\ E(i-1, j-1) + [r_i \ne h_j]\big\}
\end{aligned}
$$

Read it aloud: an empty hypothesis needs $i$ deletions, and an empty
reference needs $j$ insertions. Otherwise, take the cheapest of deleting
$r_i$, inserting $h_j$, or pairing them, which costs 1 if they differ and 0 if
they match. Walk back from $E(n, m)$, recording which move you took, to count
S, D, and I.

For example, the reference "i like green tea" against "i like the green tee"
aligns with one insertion ("the") and one substitution ("tea" to "tee"), so
WER is $2/4 = 50\%$. The Run block aligns "set a timer for two minutes"
against "set the timer four to minutes please": three substitutions and one
insertion, WER $4/6 = 66.7\%$.

The distance is unique, but the breakdown isn't. "the cat sat on the mat"
against "the cat sit on mat today" has distance 3, and it aligns as one of
each edit, or as three substitutions. The exercise's backtrace prefers the
diagonal and finds the three substitutions.

**Character error rate** (CER) is the same computation over characters,
spaces included. "kitten" against "sitting" needs two substitutions and one
insertion, so CER is $3/6 = 50\%$. CER is the standard metric for languages
written without spaces between words, such as Chinese and Japanese.

> [!WARNING]
> Corpus WER is total edits over total reference words, not the mean of
> per-utterance WERs. A one-word reference with three insertions scores 300%
> on its own, and averaging would let it swamp every long utterance.

## How HMM-GMM systems put it together

A large-vocabulary recognizer doesn't train one HMM per word. It trains HMMs
for sounds and composes them with everything else. HMM-GMM systems expressed
each knowledge source as a **weighted finite-state transducer** (WFST), a
graph whose arcs read one symbol, write another, and carry a weight. The
weights are negative log probabilities, so a path's weight is the sum along
it, and the best path is the smallest sum. The following table lists the four
transducers:

| Transducer | Reads | Writes | Holds |
|---|---|---|---|
| H | HMM states | context-dependent phones | the HMM topology of each phone |
| C | context-dependent phones | phones | which triphone each phone in context is |
| L | phones | words | the pronunciation dictionary |
| G | words | words | the n-gram language model |

A **triphone** is a phone in the context of its neighbors: `k-ae+t` is "ae"
after "k" and before "t". With about 40 phones there are up to 64,000
triphones, most never seen in training. Decision trees that ask questions
such as "is the left neighbor a nasal?" cluster their states into a few
thousand tied states, called **senones**, each with its own GMM.

Composing the four (Mohri, Pereira, and Riley, 2002) gives one graph, HCLG,
that reads HMM states and writes words. Kaldi (Povey et al., 2011) builds it
with composition, determinization, and minimization at each stage. Decoding
is then a Viterbi beam search through HCLG, with the GMMs supplying a score
for every state at every frame.

> [!RECAP]
> - Score a hypothesis with $\log P(O \mid W) + \lambda \log P(W) + \beta |W|$.
> - Interpolated Kneser-Ney discounts each seen bigram by $d$ and gives the
>   mass to $P_{\text{cont}}$, which counts distinct predecessors.
> - Perplexity is $2^H$, with `</s>` counted and `<s>` not.
> - WER is $(S + D + I) / N$ from a Levenshtein alignment, summed over the
>   whole corpus before dividing.
> - CER is the same over characters, spaces included.

## Build it step by step

The checks test the functions in this order. The file provides the corpus
grammar and the simulated n-best lists for the Run block.

1. **`bigram_counts(sentences)`.** Takes a list of word lists and returns a
   `Counter` of `(history, word)` pairs. Pad with `BOS` and `EOS`, then
   `zip(padded[:-1], padded[1:])`. Common mistake: forgetting `</s>`, which
   leaves the model unable to end a sentence.
2. **`continuation_probs(counts)`.** Count distinct predecessors of each word
   and divide by `len(counts)`. Common mistake: summing counts instead of
   counting types.
3. **`kn_prob(word, history, counts, d)`.** Compute $C(h)$, the number of
   followers, and the discounted term. Return `P_cont` for an unseen history.
   Common mistake: taking $C(h)$ from unigram counts of `h` anywhere in the
   corpus, including as the last word, where it isn't a history.
4. **`perplexity(sentences, prob)`.** Sum `math.log2(prob(w, h))` over every
   predicted token, divide by the count, and return `2 ** (-mean)`. Common
   mistake: counting `<s>` in $N$.
5. **`levenshtein_align(ref, hyp)`.** Fill the `(n + 1, m + 1)` table, then
   backtrack, preferring the diagonal. Return `(S, D, I, alignment)`. Common
   mistake: calling a matched pair a substitution because you tested the
   cost before the equality.
6. **`wer(refs, hyps)` and `cer(ref, hyp)`.** Sum errors and reference
   lengths across pairs, then divide once. For CER, align `list(ref)` against
   `list(hyp)`. Common mistake: averaging per-utterance rates.
7. **`rescore(nbest, prob, lm_weight, insertion_penalty)`.** Score each
   hypothesis with the noisy-channel formula, using natural logs, and return
   the best text. Common mistake: leaving out the `</s>` term.

## Cost and scaling

The following table uses $n$ and $m$ for the reference and hypothesis
lengths, $V$ for the vocabulary, and $B$ for the number of bigram types.

| Operation | Time | Memory |
|---|---|---|
| Counting bigrams | $O(\text{corpus tokens})$ | $O(B)$ |
| One Kneser-Ney lookup, precomputed | $O(1)$ | $O(B + V)$ |
| Perplexity | $O(N)$ lookups | $O(1)$ |
| Levenshtein alignment | $O(nm)$ | $O(nm)$, or $O(m)$ for the distance alone |
| Rescoring an n-best list of size $K$ | $O(K \times \text{length})$ | $O(K)$ |

The exercise's `kn_prob` rescans the counts on every call, which is fine for
68 bigram types. A real model precomputes $C(h)$, the follower
counts, and $P_{\text{cont}}$ once. An n-gram model's size grows with its
order: a 4-gram model trained on a large English corpus holds many millions
of n-grams, which is why toolkits prune rare ones.

## Where this breaks

**Out-of-vocabulary words.** A word never seen in training has
$P_{\text{cont}} = 0$, so perplexity is infinite. Real models map rare words
to an `<unk>` token and reserve probability for it.

**Short context.** A bigram can't know that "the dog that chased the cats
*was*" needs a singular verb. Higher orders help a little. Neural language
models, later in the course, see much further back.

**Text normalization decides WER.** "Mr." against "mister", "10" against
"ten", and "won't" against "will not" all count as errors unless both sides
are normalized the same way. Whisper's English normalizer lowercases, removes
bracketed text and fillers such as "um", expands contractions, and writes
numbers as digits. Always report which normalizer you used.

**WER treats every word alike.** Dropping "not" reverses a sentence but costs
the same as dropping "the". It also exceeds 100% when a recognizer inserts
many words, so it isn't a percentage of anything.

**Tuning the scales.** $\lambda$ and $\beta$ interact with each other and
with the beam. Tune them together on held-out data, never on the test set.

## The exercise

You build `bigram_counts`, `continuation_probs`, and `kn_prob` for an
interpolated Kneser-Ney bigram model, `perplexity`, `levenshtein_align`,
`wer`, `cer`, and the noisy-channel `rescore`. Press **Run** to see held-out
perplexity against the discount, an edit-distance table, and the WER of a
simulated recognizer before and after rescoring, which falls from 5.7% to 0%
as the language model settles which homophone was meant.

The checks use the worked corpus for the counts, continuation probabilities,
and Kneser-Ney probabilities. They confirm that every distribution sums to 1,
that $d = 0$ gives the maximum-likelihood bigram, that a uniform model's
perplexity equals its vocabulary size, and that Kneser-Ney beats add-one on
held-out text. They check the alignment on the worked pairs and against a
brute-force edit distance on 200 random pairs, then check corpus WER, CER, and
how `rescore` responds to its two weights.
