---
title: Language model fusion and hotwords
slug: 032-language-model-fusion
part: "Part 4 — Neural speech recognition"
summary: Add a character LM and hotword bonus to the beam, compare fusion types, subtract a transducer's internal LM, and rerank an n-best list.
minutes: 50
gpu: false
objectives:
  - Explain why a CTC model cannot know that cat is a word and kat is not, and how a language model adds that knowledge.
  - Compute the shallow-fusion score of a prefix from its acoustic probability, its language model probability, and an insertion bonus.
  - Train an add-k smoothed character bigram and score prefixes with it.
  - Add a hotword bonus that rewards partial matches and disappears when the match breaks.
  - Compare shallow, deep, and cold fusion, and say where each one needs training.
  - Subtract an end-to-end model's internal language model, so a target-domain LM can change the ranking.
  - Rerank an n-best list with an interpolated score, and say when a large language model can do the rescoring.
prereqs:
  - 025-n-gram-language-models
  - 031-decoding-ctc
exercise: 032-language-model-fusion
---

# Language model fusion and hotwords

A voice assistant hears a muffled "cat". The acoustic model gives `c` 42% and `k` 50%, so a beam search that sees only the acoustics writes "kat". A person would never choose that, because "kat" is not a word. The acoustic model scores sounds, and has never been told how English is spelled.

The mirror problem is a name. You ask the assistant to call Zoe. The acoustics lean toward `s`, and few Zoes appear in its training text, so the name drifts to "soe".

This chapter fixes both with a score added to the previous chapter's beam. A **language model** (LM) supplies spelling, and a **hotword bonus** supplies the names that matter. Both act during the search, so the decoder can change its mind about a letter when later letters make a word.

## What you already know

- Prefix beam search keeps the best $K$ prefixes, each with a blank-ending probability $p_b$ and a label-ending probability $p_{nb}$, as [Decoding CTC](/lessons/031-decoding-ctc) showed.
- Its `prefix_score` argument adds a function of the prefix to the ranking score.
- A bigram gives $P(c \mid \text{previous symbol})$, and add-$k$ smoothing gives an unseen pair a small count, as [n-gram language models](/lessons/025-n-gram-language-models) showed.

## Add the language model to the score

The obvious plan is to decode first and fix the text afterward. It fails because the beam has already thrown candidates away: if the acoustics put `kat` narrowly ahead of `cat` and the beam is narrow, `cat` may be gone before the LM sees it. So the LM has to take part in the ranking, frame by frame.

**Shallow fusion** adds the LM's opinion to each prefix's score, the noisy-channel idea from the n-gram chapter:

$$
\operatorname{score}(\ell) = \log\big(p_b(\ell) + p_{nb}(\ell)\big) + \alpha \log P_{\text{LM}}(\ell) + \beta\,|\ell|
$$

Read it aloud: the prefix's acoustic log probability, plus $\alpha$ times the LM's log probability of the same text, plus a bonus $\beta$ per symbol. The **LM weight** $\alpha$ says how much to trust the LM, and the **insertion bonus** $\beta$ pays for length. Every extra symbol multiplies $P_{\text{LM}}$ by a number below 1, so without $\beta$ the score prefers short transcripts. The beam ranks with this score, but $p_b$ and $p_{nb}$ stay purely acoustic. In code:

```python
score = np.logaddexp(p_b, p_nb) + alpha * lm_log_prob(prefix) + beta * len(prefix)
```

## A character bigram

The LM in this chapter is a character bigram with **add-$k$ smoothing**. Count how often each symbol $c$ follows each symbol $p$ in a text corpus, with a start symbol before the first character. Then

$$
P(c \mid p) = \frac{n(p, c) + k}{\sum_{c'} n(p, c') + k\,(C - 1)}
$$

Read it aloud: the count of the pair plus $k$, divided by the count of all pairs starting with $p$ plus $k$ for each of the $C - 1$ labels. The blank never appears in text, so the LM never predicts it, and the start context reuses index 0. In numpy:

```python
counts[:, 1:] += k
lm = np.log(counts / counts.sum(axis=1, keepdims=True))   # lm[p, c], column 0 is -inf
```

Why characters? A character model scores any prefix, even half a word, so it joins the beam at every frame. A word model scores a word only at a boundary. [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs) builds the graph that handles that.

### Work a fusion by hand

Train the bigram on `ab` and `abb` with $k = 1$. Counting pairs, with `^` for the start: `^a` twice, `ab` twice, `bb` once. Each row adds $k = 1$ to both labels:

- $P(a \mid \text{start}) = (2 + 1)/(2 + 2) = 0.75$, and $P(b \mid \text{start}) = 0.25$.
- $P(b \mid a) = (2 + 1)/(2 + 2) = 0.75$, and $P(a \mid a) = 0.25$.
- $P(b \mid b) = (1 + 1)/(1 + 2) = 2/3$, and $P(a \mid b) = 1/3$.

Decode three frames with probabilities (blank, `a`, `b`) of $(0.2, 0.35, 0.45)$, $(0.8, 0.1, 0.1)$, and $(0.2, 0.45, 0.35)$. Greedy returns `ba`. The table lists three prefixes with their acoustic log probabilities and fused scores at $\alpha = 1$:

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

The acoustics alone pick `ba`. The LM makes `ab` far likelier than `ba`, but with $\beta = 0$ the one-letter `a` wins, because it pays the LM for only one symbol. With $\beta = 1$, each symbol earns a point and `ab` wins.

Real systems tune $\alpha$ and $\beta$ by grid search on held-out data.

> [!KEY]
> Shallow fusion adds $\alpha \log P_{\text{LM}}$ and $\beta$ per symbol to each prefix's score. The LM fixes spelling the acoustics cannot hear, and $\beta$ stops it from favoring short output.

## Hotwords

Some words matter more than their LM probability suggests: a contact's name, a product, a drug. Fusion pushes them out of the beam. A **hotword bonus** gives the decoder a reason to keep them.

Paying a bonus only for a finished word is too late, because the beam has dropped `tell zo` before the `e` arrives. So the rule rewards progress. A prefix earns a bonus $\lambda$ for every symbol of a hotword it has completed, and for every symbol of a hotword it is partway through at its end. If the partial match breaks, the bonus for it disappears:

$$
\operatorname{hot}(\ell) = \lambda \,\big(\text{complete}(\ell) + \text{partial}(\ell)\big)
$$

Read it aloud: $\text{complete}(\ell)$ is the total length of hotword occurrences inside $\ell$, and $\text{partial}(\ell)$ is the longest $k$ below a hotword's length such that $\ell$ ends with its first $k$ symbols. For the hotword `zoe` with $\lambda = 2$:

- `tell z` ends with `z`, a partial match of 1, so it earns $2 \times 1 = 2$.
- `tell zo` has a partial match of 2 and earns 4.
- `tell zoe` completes the word and earns $2 \times 3 = 6$.
- `tell zoa` breaks the match and earns 0.

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

The bonus depends on the prefix alone, so it adds to the fused score as one more term. Libraries track it with a **trie**, a tree of hotword prefixes.

> [!TIP]
> Set the LM and hotword terms to zero first. The decoder must then reproduce plain beam search exactly. Add one term at a time.

## Three ways to fuse, and one to subtract

Three ways to bring in a separately trained LM differ in when the two models meet.

- **Shallow fusion** combines them at decode time, as above. Nothing is retrained, so you can swap the LM per domain.
- **Deep fusion** (Gulcehre et al., 2015, in translation) joins them inside the network. It concatenates the frozen LM's hidden state to the decoder's, with a learned sigmoid gate on the LM part, and fine-tunes only the output layer.
- **Cold fusion** (Sriram et al., 2018) trains the decoder from scratch with a fixed, pretrained LM wired in through a gate on its logits, so the decoder leaves language to the LM.

Deep and cold fusion need a training run for each new LM. Toshniwal et al. (2018) found shallow fusion best for first-pass decoding.

### Subtract the internal language model

An end-to-end model conditions on previous tokens, so it learns an **internal language model** (ILM) of its training transcripts, already inside $p_{\text{e2e}}$. Add a new-domain LM by shallow fusion and the old domain's preferences still vote. Subtract the ILM with weight $\mu$, and add the external LM with weight $\lambda$:

$$
\operatorname{score}(y) = \log p_{\text{e2e}}(y \mid x) - \mu \log p_{\text{ilm}}(y) + \lambda \log p_{\text{ext}}(y)
$$

Read it aloud: take the model's score, remove what it would say without audio, and add the new LM's view. Meng et al. (2021) call this **internal LM estimation** (ILME). For a transducer, set the encoder's contribution to the joint network to zero, so only the prediction network drives it. Drop the blank's logit, and take the softmax over the remaining tokens. The **density ratio** method (McDermott et al., 2019) instead subtracts a separate LM trained on the source-domain transcripts.

Take two names in a contacts app. The model trained on general speech, so its ILM favors `joe`. The new LM, trained on this user's contacts, favors `zoe`. With natural-log scores and $\mu = \lambda = 0.5$:

| Hypothesis | $\log p_{\text{e2e}}$ | $\log p_{\text{ilm}}$ | $\log p_{\text{ext}}$ | Shallow ($\mu = 0$) | ILME |
|---|---|---|---|---|---|
| `joe` | $-2.0$ | $-1.0$ | $-3.0$ | $-3.5$ | $-2.0 + 0.5 - 1.5 = -3.0$ |
| `zoe` | $-2.6$ | $-3.0$ | $-2.0$ | $-3.6$ | $-2.6 + 1.5 - 1.0 = -2.1$ |

Shallow fusion picks `joe` by 0.1. Subtraction credits `zoe` with $1.5$ and `joe` with only $0.5$, so `zoe` wins by 0.9. Add the ILM by mistake and the scores become $-4.0$ and $-5.1$, so `joe` wins.

Zeroing the encoder is exact for the exercise's additive toy joint, $\operatorname{log\_softmax}(\text{enc} + \text{pred})$. A real joint has a nonlinearity, so it is approximate, and you tune $\mu$ on held-out data.

### Rescore the n-best list

You can also fuse after the search. Rescore the beam's final **n-best list** with a model too slow for the beam, such as a neural LM:

$$
\operatorname{score}(y) = \log p_{\text{am}}(y) + \lambda \log p_{\text{lm}}(y) + \gamma\,|y|
$$

Read it aloud: the acoustic score, plus $\lambda$ times the LM score, plus $\gamma$ per token, which offsets the LM's pull toward short output, like $\beta$. It cannot recover a hypothesis the beam dropped, so the list must be wide enough. [Sequence-level training](/lessons/035-sequence-level-training) turns the list's scores into word posteriors.

A large language model (LLM) can do this in two ways. Prompt it with the n-best list and ask for the likeliest transcript, which can return a sentence not on the list. Or score each hypothesis's log-likelihood under the LLM and use it as $\log p_{\text{lm}}$, which keeps the output on the list. Both add an LLM call per utterance.

> [!RECAP]
> - Fusion adds $\alpha \log P_{\text{LM}} + \beta|\ell|$ to the acoustic score of each prefix, during the search.
> - The bigram's rows are add-$k$ smoothed counts, with the blank column at $-\infty$, and `lm_score` starts from the start context.
> - The hotword bonus counts complete occurrences plus the longest partial match at the end, and vanishes when the match breaks.
> - ILME scores $\log p_{\text{e2e}} - \mu \log p_{\text{ilm}} + \lambda \log p_{\text{ext}}$. The ILM comes from the joint network with the encoder zeroed and the blank removed.
> - N-best rescoring reranks the final list with $\log p_{\text{am}} + \lambda \log p_{\text{lm}} + \gamma|y|$.

## Build it step by step

The checks test the functions in this order. The file gives you `greedy_decode`, `prefix_beam_search`, and the toy transducer's `joint` in full.

1. **`train_char_bigram(texts, vocab, smoothing)`.** Returns a `(C, C)` array. Count `counts[previous, current]` from `previous = 0`, add the smoothing to columns 1 and up, normalize each row, and take `np.log`. The common mistake is smoothing the blank column.
2. **`lm_score(prefix, lm, alpha, beta)`.** Sum `lm[a, b]` over consecutive pairs of `(0,) + prefix`, multiply by `alpha`, and add `beta * len(prefix)`. The common mistake is forgetting the start context.
3. **`hotword_score(prefix, hotwords, bonus)`.** Count complete occurrences with a sliding window, and find the longest partial match with `prefix[-k:] == word[:k]`. The common mistake is letting `k` reach the full word length.
4. **`ilme_subtract(log_p_e2e, log_p_ilm, log_p_ext, mu, lam)`.** Returns `log_p_e2e - mu * log_p_ilm + lam * log_p_ext`, elementwise. The common mistake is adding the ILM.
5. **`internal_lm_logprobs(pred_vectors)`.** Takes an array of shape `(U, C)` and returns shape `(U, C - 1)`. The encoder term is zero, so the logits are the rows. Drop column 0, the blank, then apply `log_softmax`. The common mistakes are keeping the blank in the softmax, and leaving the encoder term in.
6. **`rescore_nbest(hyps, am_scores, lm_scores, lam, length_bonus)`.** Returns `(hypothesis, score)` pairs, best first, with the score `am + lam * lm + length_bonus * len(hyp)`.

## Where this breaks

**The weights are tuned for one domain.** An $\alpha$ tuned on audiobooks over-corrects medical dictation. Tune $\alpha$ and $\beta$ on the domain you serve.

**A character bigram sees one letter of context.** It cannot know that "the cat" beats "the cap". Production systems use word n-grams or neural LMs. KenLM returns $\log_{10}$ scores, so convert them to natural log before adding them to acoustic scores.

**The internal LM is an estimate.** A $\mu$ that is too large removes real evidence. Tune $\mu$ and $\lambda$ together on held-out audio.

**Hotwords can fire where they do not belong.** A large bonus pulls similar-sounding words toward the hotword. Keep the list short.

**Cost.** Scoring a prefix from scratch costs $O(L)$, so the beam costs $O(TKCL)$. Real decoders update the LM state one symbol at a time.

## The exercise

You build the three pieces that plug into the beam's `prefix_score` hook, then `ilme_subtract`, `internal_lm_logprobs`, and `rescore_nbest`. Press **Run** to decode a synthetic sentence four ways: greedy, plain beam, beam with the bigram, and beam with the bigram and a hotword. You see the posteriorgram, hear the greedy path as one beep per non-blank frame, and read the `joe` and `zoe` scores.

The checks compare the bigram, `lm_score`, and `hotword_score` with the hand-computed numbers above. With every weight at zero, the fused beam must equal the plain beam, and with a wide beam every fused score must equal the brute-force acoustic score plus the LM terms. One check turns "kat" into "cat" and "soe" into "zoe". The last three check the `joe` and `zoe` scores, the internal LM (with the blank logit set high to show it is ignored), and an n-best reranking where dropping the LM or the length term changes the order.

> [!KEY]
> What you now know: a CTC beam scores sounds only, so you add a language model and a hotword bonus to each prefix's score during the search. The LM prefers "cat" because it has seen that spelling, and the bonus keeps "zoe" alive. An end-to-end model carries its own LM, so for a new domain you subtract that one, and you can rerank the final list with a slower model.

## Further reading

- *Speech Recognition and Understanding (11-751/18-781)*, Shinji Watanabe, Carnegie Mellon University, fall 2022 ([syllabus](https://www.wavlab.org/activities/2022/11751-2022f/)): "Language model" (12 October), "End-to-end ASR: RNN-T" (14 November), and "Search" (21 November, N-best and rescoring). It has no lecture on LM fusion.
- Kannan et al., ["An analysis of incorporating an external language model into a sequence-to-sequence model"](https://arxiv.org/abs/1712.01996), arXiv, 2017.
- Gulcehre et al., ["On Using Monolingual Corpora in Neural Machine Translation"](https://arxiv.org/abs/1503.03535), arXiv, 2015.
- Sriram et al., ["Cold Fusion: Training Seq2Seq Models Together with Language Models"](https://arxiv.org/abs/1708.06426), arXiv, 2017.
- Toshniwal et al., ["A Comparison of Techniques for Language Model Integration in Encoder-Decoder Speech Recognition"](https://arxiv.org/abs/1807.10857), SLT 2018.
- Meng et al., ["Internal Language Model Estimation for Domain-Adaptive End-to-End Speech Recognition"](https://arxiv.org/abs/2011.01991), SLT 2021.
- McDermott, Sak, and Variani, ["A Density Ratio Approach to Language Model Fusion in End-To-End Automatic Speech Recognition"](https://arxiv.org/abs/2002.11268), ASRU 2019.
