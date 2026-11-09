---
title: n-gram language models
slug: 024-n-gram-language-models
part: "Part 3 — Classical recognition"
summary: Choose between transcripts that sound alike with a Kneser-Ney bigram language model, measure it with perplexity, and combine it with acoustic scores.
minutes: 40
gpu: false
objectives:
  - Explain the noisy-channel view of recognition, with its language model scale and word insertion penalty.
  - Build an interpolated Kneser-Ney bigram model and verify that its probabilities sum to 1.
  - Measure perplexity on held-out text and compare two smoothing methods.
  - Rescore an n-best list by adding scaled language model scores to acoustic scores.
prereqs:
  - 018-phonemes-and-text-normalization
  - 021-hidden-markov-models
exercise: 024-n-gram-language-models
---

# n-gram language models

"Write to mom" and "right two mom" sound the same. No acoustic model, however good, can tell them apart, because the difference is not in the sound. It is in what people are likely to say. A recognizer needs a second model, one that knows which word sequences are plausible. That is a **language model** (LM).

In this chapter you build a bigram language model with Kneser-Ney smoothing, the method behind the n-gram models that decoded speech for decades. You measure it with perplexity, and you combine its scores with acoustic scores to pick the right transcript from a list of candidates.

## What you already know

- An HMM gives $\log P(O \mid W)$, the log-likelihood of the frames $O$ under the model of a word sequence $W$. See [Hidden Markov models](/lessons/021-hidden-markov-models).
- A pronunciation dictionary maps each word to phones, such as ARPAbet `R AY T` for "write". See [Phonemes and text normalization](/lessons/018-phonemes-and-text-normalization).
- A probability of 0 for something that can happen is dangerous, because a product containing it stays 0.
- $\log_2 x = \ln x / \ln 2$, and $2^{\log_2 x} = x$.

## Add a second opinion to the acoustic score

Think of speech as a message sent through a noisy channel. Someone chose a word sequence $W$, and the channel turned it into the sound $O$ you received. To decode, pick the $W$ that was most likely sent:

$$
\hat W = \operatorname*{arg\,max}_W P(W \mid O) = \operatorname*{arg\,max}_W P(O \mid W)\, P(W)
$$

Read it aloud: the best transcript is the one that makes the sound likely, $P(O \mid W)$, the **acoustic model**, and that is itself likely, $P(W)$, the **language model**. Bayes' rule turns one into the other, and $P(O)$ drops out because it is the same for every $W$.

Figure 1 shows what the second factor buys. Four transcripts sound identical, so the acoustic model scores them the same. A bigram model trained on 400 sentences from the exercise's voice-assistant grammar scores them very differently.

```figure
<svg viewBox="0 0 620 235" role="img" aria-label="Four homophone transcripts with the Kneser-Ney log probability of each as a horizontal bar, the correct one shortest">
<line x1="190.0" y1="40.0" x2="190.0" y2="190.0" class="stroke-muted"/>
<text x="180.0" y="65.0" class="fig-small" text-anchor="end">write to mom</text>
<rect x="190.0" y="48.0" width="62.6" height="24.0" class="fill-b" />
<text x="260.6" y="65.0" class="fig-small" text-anchor="start">ln P = -4.5</text>
<text x="180.0" y="99.0" class="fig-small" text-anchor="end">right to mom</text>
<rect x="190.0" y="82.0" width="273.8" height="24.0" class="fill-soft-b stroke-b" />
<text x="471.8" y="99.0" class="fig-small" text-anchor="start">ln P = -19.6</text>
<text x="180.0" y="133.0" class="fig-small" text-anchor="end">write two mom</text>
<rect x="190.0" y="116.0" width="243.6" height="24.0" class="fill-soft-b stroke-b" />
<text x="441.6" y="133.0" class="fig-small" text-anchor="start">ln P = -17.4</text>
<text x="180.0" y="167.0" class="fig-small" text-anchor="end">right two mom</text>
<rect x="190.0" y="150.0" width="333.4" height="24.0" class="fill-soft-b stroke-b" />
<text x="531.4" y="167.0" class="fig-small" text-anchor="start">ln P = -23.8</text>
<text x="310.0" y="30.0" class="fig-small" text-anchor="middle">bar length = minus ln P(W): a longer bar is less likely</text>
<text x="310.0" y="215.0" class="fig-small" text-anchor="middle">All four sound the same, so the acoustic score cannot choose between them.</text>
</svg>
Figure 1. The language model log probability of four homophone transcripts, negated so a longer bar means less likely. Notice that "write to mom" is far more likely than "right two mom", a gap of about 19 nats, which is a factor of roughly 200 million.
```

In practice the two scores are badly mismatched. An HMM multiplies one probability per 10 ms frame and treats frames as independent, so its log scores are overconfident and huge. The language model contributes one term per word. Systems therefore scale the language model and add a per-word term:

$$
\operatorname{score}(W) = \log P(O \mid W) + \lambda \log P(W) + \beta\, |W|
$$

Read it aloud: the acoustic log score, plus the language model log score times a scale $\lambda$, plus $\beta$ for each of the $|W|$ words. A positive $\beta$ rewards every word, and a negative $\beta$ is a **word insertion penalty** that stops the decoder from filling gaps with short words. You tune both on held-out data. In the exercise, `rescore` applies this score to an **n-best list**, the few best transcripts a recognizer proposes.

Take two candidates for a command, with a language model that gives every word probability 0.5, so each of the $|W|$ words plus the end marker costs $\ln 0.5 = -0.693$. "Call mom" scores acoustic $0.0$, and "call mom now" scores $-0.5$. With $\lambda = 1$ and $\beta = 0$ the totals are $-2.08$ and $-3.27$, so the short one wins. With $\beta = 2$ the totals become $1.92$ and $2.73$, and the long one wins, so $\beta$ trades length against the other two scores.

## Predict each word from the one before

A language model assigns a probability to a whole sentence. The chain rule splits it into one prediction per word, and an **n-gram** model assumes each word depends only on the $n - 1$ words before it. A **bigram** model looks at one previous word, the **history** $h$:

$$
P(w_1 \dots w_N) \approx \prod_{i=1}^{N} P(w_i \mid w_{i-1})
$$

Read it aloud: the probability of the sentence is the product of the probability of each word given the one before it. Pad every sentence with a start marker `<s>`, which is only ever a history, and an end marker `</s>`, which is predicted like a word, so the model also learns how sentences end.

The obvious estimate is a ratio of counts, $P(w \mid h) = C(h\,w) / C(h)$, where $C(h\,w)$ counts the bigram and $C(h)$ counts every bigram that starts with $h$. That is the **maximum-likelihood** estimate, and it fails on any pair it never saw. The pair gets probability 0, so any test sentence containing it gets probability 0, however sensible the rest is.

## Smooth with Kneser-Ney

**Smoothing** moves a little probability from seen bigrams to unseen ones. **Absolute discounting** subtracts a fixed amount $d$, usually about 0.75, from every seen count, and the freed mass goes to a backup distribution over single words.

Kneser and Ney's insight was about that backup. A word like "Francisco" is common, but only after "San". As a backup it should be rare, because an unseen context is unlikely to be followed by it. So the backup counts how many *different* words a word follows, not how often it occurs. Interpolated Kneser-Ney for bigrams is:

$$
\begin{aligned}
P_{\text{KN}}(w \mid h) &= \frac{\max\big(C(h\,w) - d,\ 0\big)}{C(h)} + \lambda(h)\, P_{\text{cont}}(w) \\
\lambda(h) &= \frac{d\, N_{1+}(h\,\bullet)}{C(h)}
\end{aligned}
$$

$$
P_{\text{cont}}(w) = \frac{N_{1+}(\bullet\, w)}{N_{1+}(\bullet\,\bullet)}
$$

Read it aloud: take the discounted count of the bigram as a share of the history's count. Add a backup term, where $\lambda(h)$ is exactly the mass the discount removed: $d$ times $N_{1+}(h\,\bullet)$, the number of distinct words seen after $h$, over $C(h)$. The **continuation probability** $P_{\text{cont}}(w)$ is the number of distinct words $w$ follows, $N_{1+}(\bullet\, w)$, over the number of distinct bigram types. Summed over every word, the first term gives $1 - \lambda(h)$ and the second gives $\lambda(h)$, so the distribution sums to 1. In code, all three counts come from the bigram `Counter`:

```python
followers = sum(1 for (h, _) in counts if h == history)   # N1+(h .)
predecessors = Counter(w for (_, w) in counts)             # N1+(. w), for every w
```

### Work the example by hand

Take a three-sentence corpus: "a b", "a c", and "b c". With markers, the bigram types are (`<s>` a) twice, (a b), (b `</s>`), (a c), (c `</s>`) twice, (`<s>` b), and (b c). That is 7 distinct types.

Continuation probabilities come from counting distinct predecessors. The word "a" follows only `<s>`, so $P_{\text{cont}}(a) = 1/7$. Each of "b", "c", and `</s>` follows two different words, so each gets $2/7$. Now condition on the history "a". It occurs twice, $C(a) = 2$, followed by two distinct words, so $\lambda(a) = 0.75 \times 2 / 2 = 0.75$:

- $P(b \mid a) = (1 - 0.75)/2 + 0.75 \times 2/7 = 0.125 + 0.2143 = 0.3393$, because "a b" occurred once.
- $P(c \mid a) = 0.3393$, by the same arithmetic.
- $P(\texttt{</s>} \mid a) = 0 + 0.75 \times 2/7 = 0.2143$.
- $P(a \mid a) = 0 + 0.75 \times 1/7 = 0.1071$.

The four sum to 1. The unseen "`</s>` after a" gets twice the probability of the unseen "a after a", because `</s>` ends more kinds of context. Figure 2 compares this with counting alone.

```figure
<svg viewBox="0 0 640 245" role="img" aria-label="Grouped bars for the next-word probability after the word a: counting gives zero to unseen words, Kneser-Ney gives them a small share">
<line x1="100.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-muted"/>
<line x1="100.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-grid"/>
<text x="94.0" y="204.0" class="fig-small" text-anchor="end">0</text>
<line x1="100.0" y1="125.0" x2="540.0" y2="125.0" class="stroke-grid"/>
<text x="94.0" y="129.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="100.0" y1="50.0" x2="540.0" y2="50.0" class="stroke-grid"/>
<text x="94.0" y="54.0" class="fig-small" text-anchor="end">0.5</text>
<rect x="130.0" y="50.0" width="32.0" height="150.0" class="fill-a" />
<rect x="166.0" y="98.2" width="32.0" height="101.8" class="fill-b" />
<text x="146.0" y="45.0" class="fig-small" text-anchor="middle">0.500</text>
<text x="182.0" y="93.2" class="fig-small" text-anchor="middle">0.339</text>
<text x="164.0" y="218.0" class="fig-small" text-anchor="middle">b</text>
<rect x="235.0" y="50.0" width="32.0" height="150.0" class="fill-a" />
<rect x="271.0" y="98.2" width="32.0" height="101.8" class="fill-b" />
<text x="251.0" y="45.0" class="fig-small" text-anchor="middle">0.500</text>
<text x="287.0" y="93.2" class="fig-small" text-anchor="middle">0.339</text>
<text x="269.0" y="218.0" class="fig-small" text-anchor="middle">c</text>
<rect x="340.0" y="200.0" width="32.0" height="0.0" class="fill-a" />
<rect x="376.0" y="135.7" width="32.0" height="64.3" class="fill-b" />
<text x="356.0" y="195.0" class="fig-small" text-anchor="middle">0.000</text>
<text x="392.0" y="130.7" class="fig-small" text-anchor="middle">0.214</text>
<text x="374.0" y="218.0" class="fig-small" text-anchor="middle">&lt;/s&gt;</text>
<rect x="445.0" y="200.0" width="32.0" height="0.0" class="fill-a" />
<rect x="481.0" y="167.9" width="32.0" height="32.1" class="fill-b" />
<text x="461.0" y="195.0" class="fig-small" text-anchor="middle">0.000</text>
<text x="497.0" y="162.9" class="fig-small" text-anchor="middle">0.107</text>
<text x="479.0" y="218.0" class="fig-small" text-anchor="middle">a</text>
<rect x="490.0" y="40.0" width="14.0" height="14.0" class="fill-a" />
<text x="510.0" y="52.0" class="fig-small" text-anchor="start">counts only</text>
<rect x="490.0" y="66.0" width="14.0" height="14.0" class="fill-b" />
<text x="510.0" y="78.0" class="fig-small" text-anchor="start">Kneser-Ney</text>
<text x="300.0" y="232.0" class="fig-small" text-anchor="middle">next word after a</text>
<text x="100.0" y="24.0" class="fig-small" text-anchor="start">P(next word | a)</text>
</svg>
Figure 2. The probability of each next word after "a" from counts alone and from Kneser-Ney. Notice that counting gives 0 to the two words never seen after "a", while Kneser-Ney gives them a small share, larger for the word that follows more kinds of word.
```

## Measure a model with perplexity

To compare language models, measure how surprised each one is by held-out text. **Perplexity** is the inverse probability of the text, normalized per token:

$$
\begin{aligned}
\text{PP} &= P(w_1 \dots w_N)^{-1/N} = 2^{H} \\
H &= -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_{i-1})
\end{aligned}
$$

Read it aloud: $H$ is the average number of bits the model needs per token, and perplexity is 2 to that power. A model that picks uniformly among 10 digits has perplexity 10, so perplexity reads as the number of equally likely choices the model is torn between. Lower is better. Count every `</s>` in $N$, but no `<s>`, since it is never predicted.

For the sentence "a c" under the worked model, three tokens are predicted: $P(a \mid \texttt{<s>}) = 0.4881$, $P(c \mid a) = 0.3393$, and $P(\texttt{</s>} \mid c) = 0.7321$. Their product is 0.1212, and $0.1212^{-1/3} = 2.020$.

On the exercise's voice-assistant corpus, a Kneser-Ney bigram trained on 400 sentences has held-out perplexity 2.25 at $d = 0.75$. Add-one smoothing, which adds 1 to every count, scores 3.02. Figure 3 shows that the discount hardly matters on this regular grammar, while the choice of method does.

```figure
<svg viewBox="0 0 600 275" role="img" aria-label="Held-out perplexity of Kneser-Ney against the discount d, always below the flat add-one line">
<line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-muted"/>
<line x1="80.0" y1="30.0" x2="80.0" y2="220.0" class="stroke-muted"/>
<line x1="80.0" y1="220.0" x2="560.0" y2="220.0" class="stroke-grid"/>
<text x="74.0" y="224.0" class="fig-small" text-anchor="end">2.0</text>
<line x1="80.0" y1="156.7" x2="560.0" y2="156.7" class="stroke-grid"/>
<text x="74.0" y="160.7" class="fig-small" text-anchor="end">2.4</text>
<line x1="80.0" y1="93.3" x2="560.0" y2="93.3" class="stroke-grid"/>
<text x="74.0" y="97.3" class="fig-small" text-anchor="end">2.8</text>
<line x1="80.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="74.0" y="34.0" class="fig-small" text-anchor="end">3.2</text>
<text x="80.0" y="238.0" class="fig-small" text-anchor="middle">0</text>
<text x="200.0" y="238.0" class="fig-small" text-anchor="middle">0.25</text>
<text x="320.0" y="238.0" class="fig-small" text-anchor="middle">0.5</text>
<text x="440.0" y="238.0" class="fig-small" text-anchor="middle">0.75</text>
<text x="560.0" y="238.0" class="fig-small" text-anchor="middle">1</text>
<polyline points="104.0,186.4 128.0,186.0 152.0,185.5 176.0,185.1 200.0,184.6 224.0,184.1 248.0,183.7 272.0,183.2 296.0,182.7 320.0,182.3 344.0,181.8 368.0,181.3 392.0,180.9 416.0,180.4 440.0,179.9 464.0,179.4 488.0,178.9 512.0,178.5 536.0,178.0" class="stroke-b" fill="none"/>
<line x1="80.0" y1="58.7" x2="560.0" y2="58.7" class="stroke-a dashed"/>
<text x="560.0" y="52.7" class="fig-small" text-anchor="end">add-one: 3.02</text>
<text x="368.0" y="201.9" class="fig-small" text-anchor="middle">Kneser-Ney at d = 0.75: 2.25</text>
<text x="320.0" y="260.0" class="fig-small" text-anchor="middle">discount d</text>
<text x="74.0" y="20.0" class="fig-small" text-anchor="start">held-out perplexity</text>
</svg>
Figure 3. Held-out perplexity of the Kneser-Ney bigram as the discount d varies from 0.05 to 0.95, against add-one smoothing. Notice that every Kneser-Ney value sits near 2.2 and well below 3.02.
```

> [!KEY]
> A language model scores how plausible a word sequence is, so the recognizer can choose between transcripts that sound alike. Kneser-Ney backs off to how many contexts a word completes, not how often it occurs. Perplexity compares models only when the vocabulary and the tokenization are the same.

> [!RECAP]
> - Score a hypothesis with $\log P(O \mid W) + \lambda \log P(W) + \beta |W|$.
> - Interpolated Kneser-Ney discounts each seen bigram by $d$ and gives the mass to $P_{\text{cont}}$, which counts distinct predecessors.
> - Every KN distribution over the predicted tokens sums to 1, and $d = 0$ gives the maximum-likelihood bigram.
> - Perplexity is $2^H$, with `</s>` counted and `<s>` not.

## Build it step by step

The checks test the functions in this order. The file provides the corpus grammar and the simulated n-best lists for the Run block.

1. **`bigram_counts(sentences)`.** Takes a list of word lists and returns a `Counter` of `(history, word)` pairs. Pad with `BOS` and `EOS`, then `zip(padded[:-1], padded[1:])`. The usual mistake is forgetting `</s>`, which leaves the model unable to end a sentence.
2. **`continuation_probs(counts)`.** Count distinct predecessors of each word and divide by `len(counts)`. The usual mistake is summing counts instead of counting types.
3. **`kn_prob(word, history, counts, d)`.** Compute $C(h)$, the number of followers, and the discounted term. Return `P_cont` for an unseen history. The usual mistake is taking $C(h)$ from unigram counts of `h` anywhere in the corpus, including as the last word, where it is not a history.
4. **`perplexity(sentences, prob)`.** Sum `math.log2(prob(w, h))` over every predicted token, divide by the count, and return `2 ** (-mean)`. The usual mistake is counting `<s>` in $N$.
5. **`rescore(nbest, prob, lm_weight, insertion_penalty)`.** Score each hypothesis with the noisy-channel formula, using natural logs, and return the best text. The usual mistake is leaving out the `</s>` term.

## Where this breaks

**Out-of-vocabulary words.** A word never seen in training has $P_{\text{cont}} = 0$, so perplexity is infinite. Real models map rare words to an `<unk>` token and reserve probability for it.

**Short context.** A bigram cannot know that "the dog that chased the cats *was*" needs a singular verb. Higher orders help a little. Neural language models, later in the course, see much further back.

**Scaling the model.** The exercise's `kn_prob` rescans the counts on every call, which is fine for 68 bigram types. A real model precomputes the counts once, and prunes rare n-grams because the model's size grows with its order. Modified Kneser-Ney (Chen and Goodman, 1998) uses three discounts instead of one, and KenLM (Heafield, 2011) builds such models at scale.

**Tuning the scales.** $\lambda$ and $\beta$ interact with each other and with the beam. Tune them together on held-out data, never on the test set. The next chapter, [Word error rate](/lessons/025-word-error-rate), builds the metric you tune against.

## The exercise

You build `bigram_counts`, `continuation_probs`, and `kn_prob` for an interpolated Kneser-Ney bigram model, then `perplexity` and the noisy-channel `rescore`. Press **Run** to see held-out perplexity against the discount, and to watch rescoring fix homophones in a simulated recognizer's 4-best lists, picking the right sentence 60 times out of 60 where acoustics alone pick it 45 times.

The checks use the worked corpus for the counts, continuation probabilities, and Kneser-Ney probabilities. They confirm that every distribution sums to 1, that $d = 0$ gives the maximum-likelihood bigram, that a uniform model's perplexity equals its vocabulary size, and that Kneser-Ney beats add-one on held-out text. The last check shows how `rescore` responds to its two weights.
