---
title: Language model fusion and hotwords
slug: 032-language-model-fusion
part: "Part 4 — Neural speech recognition"
summary: Add a character language model and a hotword bonus to the beam's score, so the decoder fixes spelling and names the audio left unclear.
minutes: 40
gpu: false
objectives:
  - Explain why a CTC model cannot know that cat is a word and kat is not, and how a language model adds that knowledge.
  - Compute the shallow-fusion score of a prefix from its acoustic probability, its language model probability, and an insertion bonus.
  - Train an add-k smoothed character bigram and score prefixes with it.
  - Add a hotword bonus that rewards partial matches and disappears when the match breaks.
prereqs:
  - 025-n-gram-language-models
  - 031-decoding-ctc
exercise: 032-language-model-fusion
---

# Language model fusion and hotwords

A voice assistant hears a muffled "cat". The acoustic model is unsure between `c` and `k`, and gives them 42% and 50%. A beam search that sees only the acoustics picks `k`, and the transcript reads "the kat". A person would never make that choice, because "kat" is not a word and "cat" is. The acoustic model has no way to know that. It scores sounds, and it has never been told how English is spelled.

The mirror problem is a name. You ask the assistant to call Zoe. The acoustics lean toward `s`, and the language the assistant has seen before has few Zoes, so the name drifts to "soe" and out of the beam.

This chapter fixes both with a score added to the beam of the previous chapter. A **language model** (LM) supplies the spelling knowledge, and a **hotword bonus** supplies the names you know matter. Both are added during the search, so the decoder can still change its mind about a letter when later letters make a word.

## What you already know

- Prefix beam search keeps the best $K$ prefixes, each with a blank-ending probability $p_b$ and a label-ending probability $p_{nb}$, as [Decoding CTC](/lessons/031-decoding-ctc) showed.
- Its `prefix_score` argument adds a function of the prefix to the score used to rank prefixes.
- A bigram language model gives $P(c \mid \text{previous symbol})$, and a noisy-channel decoder multiplies it with the acoustic probability, as [n-gram language models](/lessons/025-n-gram-language-models) showed.
- Add-$k$ smoothing gives an unseen pair a small count instead of zero.

## Add the language model to the score

The obvious plan is to decode first and fix the text afterward: run the beam, then ask the LM which of the candidates reads best. It fails because the beam has already thrown candidates away. If the acoustics put `kat` ahead of `cat` at frame 12 by a small margin, and the beam is narrow, `cat` may be gone before the LM sees it. So the LM has to take part in the ranking, frame by frame.

**Shallow fusion** does that by adding the LM's opinion to each prefix's score, the same noisy-channel idea as in the n-gram chapter:

$$
\operatorname{score}(\ell) = \log\big(p_b(\ell) + p_{nb}(\ell)\big) + \alpha \log P_{\text{LM}}(\ell) + \beta\,|\ell|
$$

Read it aloud: the score of prefix $\ell$ is its acoustic log probability, plus $\alpha$ times the LM's log probability of the same text, plus a bonus $\beta$ for every symbol it holds. The **LM weight** $\alpha$ says how much to trust the LM, and the **insertion bonus** $\beta$ pays for length. Every extra symbol multiplies $P_{\text{LM}}$ by a number below 1, so without $\beta$ the fused score prefers short transcripts. The beam ranks and prunes with this score, but $p_b$ and $p_{nb}$ stay purely acoustic. In code, the term is a function you hand to the beam:

```python
score = np.logaddexp(p_b, p_nb) + alpha * lm_log_prob(prefix) + beta * len(prefix)
```

## A character bigram

The LM in this chapter is a character bigram with **add-$k$ smoothing**. Count how often each symbol $c$ follows each symbol $p$ in a text corpus, with a start symbol before the first character. Then

$$
P(c \mid p) = \frac{n(p, c) + k}{\sum_{c'} n(p, c') + k\,(C - 1)}
$$

Read it aloud: the probability of $c$ after $p$ is the count of that pair plus $k$, divided by the count of all pairs that start with $p$ plus $k$ for each of the $C - 1$ labels. The $k$ gives unseen pairs a small probability instead of zero. The blank never appears in text, so the LM never predicts it, and the start context reuses index 0. In numpy:

```python
counts[:, 1:] += k
lm = np.log(counts / counts.sum(axis=1, keepdims=True))   # lm[p, c], column 0 is -inf
```

Why a character model and not a word model? A character model works on any prefix, even half a word, so it can join the beam at every frame. A word model scores a word only when the beam emits a word boundary, which is why production systems that use one add a boundary symbol and wait for it. Composing a lexicon with a word language model puts that constraint in one graph. [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs) builds that graph for a toy vocabulary. This chapter adds the language model to each prefix's score, which is how a CTC beam uses spelling knowledge before a word boundary arrives.

### Work a fusion by hand

Train the bigram on the two texts `ab` and `abb` with $k = 1$. Counting pairs, with `^` for the start: `^a` twice, `ab` twice, and `bb` once. Each row adds $k = 1$ to both labels:

- $P(a \mid \text{start}) = (2 + 1)/(2 + 2) = 0.75$, and $P(b \mid \text{start}) = 0.25$.
- $P(b \mid a) = (2 + 1)/(2 + 2) = 0.75$, and $P(a \mid a) = 0.25$.
- $P(b \mid b) = (1 + 1)/(1 + 2) = 2/3$, and $P(a \mid b) = 1/3$.

Now decode three frames with probabilities (blank, `a`, `b`) of $(0.2, 0.35, 0.45)$, $(0.8, 0.1, 0.1)$, and $(0.2, 0.45, 0.35)$. Greedy reads `b _ a` and returns `ba`. The following table lists three candidate prefixes, their exact acoustic log probabilities from the beam, and their fused scores with $\alpha = 1$:

| Prefix | $\log p_{\text{CTC}}$ | $\log P_{\text{LM}}$ | Score, $\beta = 0$ | Score, $\beta = 1$ |
|---|---|---|---|---|
| `ba` | $-1.5119$ | $\log 0.25 + \log \tfrac13 = -2.4849$ | $-3.9968$ | $-1.9968$ |
| `ab` | $-1.9914$ | $2 \log 0.75 = -0.5754$ | $-2.5668$ | $-0.5668$ |
| `a` | $-1.8094$ | $\log 0.75 = -0.2877$ | $-2.0971$ | $-1.0971$ |

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Scores of the prefixes ba, ab, and a on a number line: acoustic only, with the language model at insertion bonus 0, and with insertion bonus 1; the winner changes from ba to a to ab">
  <line x1="60" y1="210" x2="540" y2="210" class="stroke-muted"/>
  <line x1="82.9" y1="210" x2="82.9" y2="215" class="stroke-muted"/>
  <text x="82.9" y="230" text-anchor="middle" class="fig-small">-4</text>
  <line x1="197.1" y1="210" x2="197.1" y2="215" class="stroke-muted"/>
  <text x="197.1" y="230" text-anchor="middle" class="fig-small">-3</text>
  <line x1="311.4" y1="210" x2="311.4" y2="215" class="stroke-muted"/>
  <text x="311.4" y="230" text-anchor="middle" class="fig-small">-2</text>
  <line x1="425.7" y1="210" x2="425.7" y2="215" class="stroke-muted"/>
  <text x="425.7" y="230" text-anchor="middle" class="fig-small">-1</text>
  <line x1="540.0" y1="210" x2="540.0" y2="215" class="stroke-muted"/>
  <text x="540.0" y="230" text-anchor="middle" class="fig-small">0</text>
  <text x="300" y="256" text-anchor="middle" class="fig-small">score (natural log; higher is better)</text>
  <circle cx="64" cy="28" r="6" class="fill-d"/><text x="76" y="32" class="fig-small">acoustic only</text>
  <circle cx="190" cy="28" r="6" class="fill-a"/><text x="202" y="32" class="fig-small">LM, bonus 0</text>
  <circle cx="310" cy="28" r="6" class="fill-b"/><text x="322" y="32" class="fig-small">LM, bonus 1</text>
  <circle cx="430" cy="28" r="10" fill="none" class="stroke-ink"/><text x="446" y="32" class="fig-small">best in its column</text>
  <line x1="60" y1="80" x2="540" y2="80" class="stroke-grid"/>
  <text x="14" y="85" class="fig-mono fig-bold">ba</text>
  <circle cx="367.2" cy="80" r="6" class="fill-d"/>
  <circle cx="367.2" cy="80" r="11" fill="none" class="stroke-ink"/>
  <circle cx="83.2" cy="80" r="6" class="fill-a"/>
  <circle cx="311.8" cy="80" r="6" class="fill-b"/>
  <line x1="60" y1="125" x2="540" y2="125" class="stroke-grid"/>
  <text x="14" y="130" class="fig-mono fig-bold">ab</text>
  <circle cx="312.4" cy="125" r="6" class="fill-d"/>
  <circle cx="246.7" cy="125" r="6" class="fill-a"/>
  <circle cx="475.2" cy="125" r="6" class="fill-b"/>
  <circle cx="475.2" cy="125" r="11" fill="none" class="stroke-ink"/>
  <line x1="60" y1="170" x2="540" y2="170" class="stroke-grid"/>
  <text x="14" y="175" class="fig-mono fig-bold">a</text>
  <circle cx="333.2" cy="170" r="6" class="fill-d"/>
  <circle cx="300.3" cy="170" r="6" class="fill-a"/>
  <circle cx="300.3" cy="170" r="11" fill="none" class="stroke-ink"/>
  <circle cx="414.6" cy="170" r="6" class="fill-b"/>
</svg>
Figure 1. The same three prefixes under three scores. The acoustics alone pick ba. Adding the LM with no insertion bonus picks the one-letter a, which pays the LM for only one symbol. Adding a bonus of 1 per symbol picks ab.
```

The acoustics alone pick `ba`, with $p = 0.2205$. The LM makes `ab` far more likely than `ba`, but with $\beta = 0$ the winner is the one-letter `a`, because it pays the LM for only one symbol. With $\beta = 1$, each prefix earns one point per symbol, and `ab` wins. That is the job of the insertion bonus.

Real systems tune $\alpha$ and $\beta$ by grid search on held-out data. For example, for its Libri-light experiments, wav2vec 2.0 used a 4-gram LM with a weight of 2, a word insertion penalty of $-1$, and a beam of 50, fixed in advance.

> [!KEY]
> Shallow fusion adds $\alpha \log P_{\text{LM}}$ and $\beta$ per symbol to each prefix's score. The LM fixes spelling the acoustics cannot hear, and $\beta$ stops it from favoring short output.

## Hotwords

Some words matter more than their LM probability suggests: a contact's name, a product, a drug. The LM has rarely or never seen them, so fusion pushes them out of the beam. A **hotword bonus** gives the decoder a reason to keep them.

The obvious version pays a bonus only when the whole word is finished. That is too late, because the beam has dropped `tell zo` long before the `e` arrives. So the rule is a reward for progress. A prefix earns a bonus $\lambda$ for every symbol of a hotword it has completed, and for every symbol of a hotword it is partway through at its end. If the partial match breaks, the bonus for it disappears:

$$
\operatorname{hot}(\ell) = \lambda \,\big(\text{complete}(\ell) + \text{partial}(\ell)\big)
$$

Read it aloud: $\text{complete}(\ell)$ is the total length of every hotword occurrence inside $\ell$, and $\text{partial}(\ell)$ is the longest $k$ below a hotword's length such that $\ell$ ends with that hotword's first $k$ symbols. For the hotword `zoe` with $\lambda = 2$:

- `tell z` ends with `z`, a partial match of 1, so it earns $2 \times 1 = 2$.
- `tell zo` has a partial match of 2 and earns 4.
- `tell zoe` completes the word and earns $2 \times 3 = 6$.
- `tell zoa` breaks the match and earns 0, so the bonus it had is taken back.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Hotword bonus for the hotword zoe with lambda 2: 2 for a partial match of one symbol, 4 for two, 6 for the full word, and 0 when the match breaks">
  <line x1="60" y1="190" x2="560" y2="190" class="stroke-muted"/>
  <text x="14" y="24" class="fig-small">bonus added to the score</text>
  <rect x="90" y="150" width="70" height="40" class="fill-c"/>
  <text x="125" y="142" text-anchor="middle" class="fig-mono">2</text>
  <text x="125" y="212" text-anchor="middle" class="fig-mono fig-small">tell z</text>
  <rect x="210" y="110" width="70" height="80" class="fill-c"/>
  <text x="245" y="102" text-anchor="middle" class="fig-mono">4</text>
  <text x="245" y="212" text-anchor="middle" class="fig-mono fig-small">tell zo</text>
  <rect x="330" y="70" width="70" height="120" class="fill-c"/>
  <text x="365" y="62" text-anchor="middle" class="fig-mono">6</text>
  <text x="365" y="212" text-anchor="middle" class="fig-mono fig-small">tell zoe</text>
  <rect x="450" y="186" width="70" height="4" class="fill-bad"/>
  <text x="485" y="182" text-anchor="middle" class="fig-mono">0</text>
  <text x="485" y="212" text-anchor="middle" class="fig-mono fig-small">tell zoa</text>
  <text x="60" y="240" class="fig-small">Hotword zoe, lambda = 2. The last prefix breaks the match, so the bonus it had is gone.</text>
</svg>
Figure 2. The hotword bonus grows with each matching symbol and vanishes when the match breaks, so a prefix that heads toward zoe stays in the beam, and one that turns aside does not keep a reward it did not earn.
```

The bonus depends on the prefix alone, so it adds to the fused score as one more term. Libraries track it with a **trie**, a tree of hotword prefixes, so each extension updates it in constant time.

> [!TIP]
> Test your decoder with the LM and hotword terms set to zero first. It must then reproduce plain beam search exactly. Add one term at a time.

> [!RECAP]
> - Fusion adds $\alpha \log P_{\text{LM}} + \beta|\ell|$ to the acoustic score of each prefix, during the search.
> - The bigram's rows are add-$k$ smoothed counts, with column 0, the blank, left at $-\infty$.
> - `lm_score` starts from the start context, so the first letter is not free.
> - The hotword bonus counts complete occurrences plus the longest partial match at the end, and vanishes when the match breaks.

## Build it step by step

The checks test the functions in this order. The file gives you `greedy_decode` and `prefix_beam_search` in full from the previous chapter.

1. **`train_char_bigram(texts, vocab, smoothing)`.** Returns a `(C, C)` array of log probabilities. Count `counts[previous, current]` starting from `previous = 0`, add the smoothing to columns 1 and up, normalize each row, and take `np.log`. The common mistake is adding the smoothing to the blank column, which leaks probability to a symbol the LM can never emit.
2. **`lm_score(prefix, lm, alpha, beta)`.** Returns a float. Sum `lm[a, b]` over consecutive pairs of `(0,) + prefix`, multiply by `alpha`, and add `beta * len(prefix)`. The common mistake is forgetting the start context, so the first letter is free.
3. **`hotword_score(prefix, hotwords, bonus)`.** Returns a float. Count complete occurrences with a sliding window and find the longest partial match with `prefix[-k:] == word[:k]`. The common mistake is letting `k` reach the full word length, which counts a completed word twice.

## Where this breaks

**The weights are tuned for one domain.** An $\alpha$ tuned on audiobooks over-corrects medical dictation, where the right words look unlikely to a general LM. Tune $\alpha$ and $\beta$ on data from the domain you serve.

**A character bigram sees one letter of context.** It knows that `c` often precedes `a`, but not that "the cat" is more likely than "the cap". Production systems use word n-grams or neural LMs. KenLM, a common n-gram library, returns $\log_{10}$ scores, so convert them before you add them to natural-log acoustic scores.

**Hotwords can fire where they do not belong.** A large bonus pulls similar-sounding words toward the hotword. Keep the list short and the bonus modest.

**Cost.** Scoring a bigram prefix from scratch costs $O(L)$ for a prefix of length $L$, so a beam of $K$ over $C$ symbols costs $O(TKCL)$ overall. Real decoders update the LM state one symbol at a time, so the cost does not grow with $L$.

## The exercise

You build `train_char_bigram`, `lm_score`, and `hotword_score`, the pieces that plug into the beam's `prefix_score` hook. Press **Run** to decode a synthetic sentence four ways: greedy, plain beam, beam with the bigram, and beam with the bigram and a hotword. You see the posteriorgram and the blank's probability per frame, and you hear the greedy path as one beep per non-blank frame.

The checks compare the bigram with the hand-counted probabilities, and `lm_score` and `hotword_score` with the numbers above. They confirm that with every weight at zero the fused beam equals the plain beam, and that with a wide beam every fused score equals the brute-force acoustic score plus the LM terms. The last check decodes a sentence in which the LM must turn "kat" into "cat" and the hotword must turn "soe" into "zoe".

> [!KEY]
> What you now know: a CTC beam scores sounds only, so you add a language model and a hotword bonus to each prefix's score during the search. That answers the opening question: the LM prefers "cat" because it has seen that spelling, and the bonus keeps "zoe" alive until the last letter arrives.
