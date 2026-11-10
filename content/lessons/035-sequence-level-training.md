---
title: Sequence-level training
slug: 035-sequence-level-training
part: "Part 4 — Neural speech recognition"
summary: Train on whole transcripts with MMI and expected word errors on an n-best list, and read word posteriors from a confusion network.
minutes: 45
gpu: false
objectives:
  - Write the MMI loss as the log of a numerator over a denominator, and explain why the denominator sums over competitors weighted by a language model.
  - Compute a denominator with the forward algorithm, and its gradient as state occupancy minus the reference.
  - Compute expected word errors over an n-best list, renormalized, and derive its gradient with a mean-error baseline.
  - Build word posteriors from an aligned n-best list, and decode a consensus transcript that can beat the 1-best.
prereqs:
  - 026-word-error-rate
  - 027-wfsts-and-decoding-graphs
  - 034-transducers
exercise: 035-sequence-level-training
---

# Sequence-level training

A voice assistant hears "call mom back". Its model gives that transcript a probability of 0.30, and "call tom home" 0.25. The loss you have met so far asks how probable the reference is. It never asks which wrong answers sit just below, or how many words each gets wrong.

This chapter trains on whole transcripts. **Maximum mutual information** (MMI) compares the reference with every competitor, so training pushes the competitors down. **Minimum word error rate** (MWER) training minimizes the average word error of the model's own best guesses, which is the number you are scored on. The last section reuses the same n-best list to ask how sure the decoder is of each word.

## What you already know

- Word error rate counts substitutions, deletions, and insertions on an alignment, as [Word error rate](/lessons/026-word-error-rate) showed.
- A weighted graph scores every path, and the log semiring sums them, as [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs) showed.
- A forward recursion sums over every alignment without listing them, as in [Transducers](/lessons/034-transducers).
- A beam search returns an n-best list of scored hypotheses, as [Decoding CTC](/lessons/031-decoding-ctc) showed.

## Maximum mutual information

**Maximum likelihood** training raises $p(x \mid w_{\text{ref}})$, the chance of the audio $x$ given the reference words, and never looks at the other transcripts $w$. Decoding picks the $w$ with the highest $p(x \mid w)\,P(w)$, where $P(w)$ is the language model. The fix is to train on the quantity the decoder uses: the reference's share of the total. The **numerator** is the score of the reference, $p(x \mid w_{\text{ref}})\,P(w_{\text{ref}})$. The **denominator** is the sum of the same score over every transcript. MMI maximizes the ratio, so it raises the reference and lowers every rival at the same time:

$$
\mathcal{L}_{\text{MMI}} = -\log \frac{p(x \mid w_{\text{ref}})\,P(w_{\text{ref}})}{\sum_{w} p(x \mid w)\,P(w)}
$$

Read it aloud: minus the log of the reference's share of all the mass. The share is at most 1, so the loss is at least 0.

### A tiny example with real numbers

Use two states, `a` and `b`, over two frames. The acoustic scores $p(x_t \mid s)$ are $0.6$ and $0.4$ in frame 1, and $0.3$ and $0.7$ in frame 2. The language model gives each state $0.5$ as the first, then $0.2$ for `a` to `a`, $0.8$ for `a` to `b`, $0.6$ for `b` to `a`, and $0.4$ for `b` to `b`. The reference path is `ab`. Each path's score multiplies four factors:

| Path | Score | Working |
|---|---|---|
| `aa` | 0.018 | $0.6 \cdot 0.5 \cdot 0.3 \cdot 0.2$ |
| `ab` | 0.168 | $0.6 \cdot 0.5 \cdot 0.7 \cdot 0.8$ |
| `ba` | 0.036 | $0.4 \cdot 0.5 \cdot 0.3 \cdot 0.6$ |
| `bb` | 0.056 | $0.4 \cdot 0.5 \cdot 0.7 \cdot 0.4$ |

The denominator is $0.018 + 0.168 + 0.036 + 0.056 = 0.278$. The reference holds $0.168 / 0.278 = 0.604$ of it, so the loss is $-\ln 0.604 = 0.5037$.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="Scores of the four state paths aa, ab, ba, and bb; the reference path ab holds 0.168 of the 0.278 total">
<text x="70" y="57" text-anchor="end" class="fig-mono">aa</text>
<rect x="90" y="40" width="30.6" height="24" class="fill-soft-a"/>
<text x="128.6" y="57" class="fig-small">0.018</text>
<text x="70" y="91" text-anchor="end" class="fig-mono">ab</text>
<rect x="90" y="74" width="285.6" height="24" class="fill-b"/>
<text x="383.6" y="91" class="fig-small">0.168</text>
<text x="70" y="125" text-anchor="end" class="fig-mono">ba</text>
<rect x="90" y="108" width="61.2" height="24" class="fill-soft-a"/>
<text x="159.2" y="125" class="fig-small">0.036</text>
<text x="70" y="159" text-anchor="end" class="fig-mono">bb</text>
<rect x="90" y="142" width="95.2" height="24" class="fill-soft-a"/>
<text x="193.2" y="159" class="fig-small">0.056</text>
<line x1="90" y1="34" x2="90" y2="176" class="stroke-muted"/>
<text x="90" y="205" class="fig-small">score = P(x | path) times P(path); the four scores sum to the denominator, 0.278</text>
<text x="470" y="92" class="fig-small fig-bold">numerator</text>
<text x="470" y="110" class="fig-small">0.168 / 0.278 = 0.604</text>
</svg>
Figure 1. Training pushes up the share of the reference path ab among all four paths, 0.604 here. The loss is minus the log of that share, so it falls as ab gains share and the others lose it.
```

There are $S^T$ paths for $T$ frames and $S$ states, so use the forward recursion: keep one log score per state, and at each frame combine the previous scores through the transitions with a log-sum-exp.

```python
alpha = log_start + loglik[0]
for t in range(1, T):
    alpha = logsumexp(alpha[:, None] + log_trans, axis=0) + loglik[t]
log_den = logsumexp(alpha)
```

### The gradient is occupancy minus the reference

Differentiate the loss with respect to the acoustic log score $\ell_{t,s} = \log p(x_t \mid s)$. The numerator contributes $-1$ at the reference state of each frame. The denominator contributes the **occupancy** $\gamma_{t,s}$, the probability that frame $t$ is in state $s$ across all paths, which a forward-backward pass gives:

$$
\frac{\partial \mathcal{L}_{\text{MMI}}}{\partial \ell_{t,s}} = \gamma_{t,s} - \mathbb{1}[\,s = s_t^{\text{ref}}\,]
$$

Read it aloud: at every frame, push up the reference state and push down whatever the model thinks is occupied. Here $\gamma$ is $(0.669, 0.331)$ in frame 1 and $(0.194, 0.806)$ in frame 2, so the gradient is $(-0.331, +0.331)$ and $(+0.194, -0.194)$. In numpy, `gamma - reference_onehot`.

> [!KEY]
> MMI trains the reference against its competitors. The gradient at each frame is what the model believes minus what the reference says, so competitors that carry probability get pushed down in proportion.

### Why the denominator needs a language model

Drop $P(w)$ from the denominator and the competitors become every state sequence the acoustics allow, equally welcome. In the example the denominator becomes $(0.6 + 0.4)(0.3 + 0.7) = 1$, and the loss becomes $-\ln 0.42 = 0.8675$.

That is the wrong fight. A decoder with a language model never meets a sequence the LM rules out, so a denominator without the LM wastes capacity on competitors that never win. The denominator must contain what the decoder searches, the acoustic model combined with the LM, and the numerator carries $P(w_{\text{ref}})$ for the same reason.

### Lattice-free MMI

The classic recipe built the denominator from word lattices, which costs a decoding pass. **Lattice-free MMI** (LF-MMI), from Povey et al. (2016), skips the lattices. It builds one fixed denominator graph from a phone n-gram language model, as in [Weighted finite-state transducers and decoding graphs](/lessons/027-wfsts-and-decoding-graphs), and runs the forward-backward pass over the whole graph on a GPU. The numerator sums over the alignments of the reference words, where the toy uses one path. The network outputs at one third the usual frame rate, which keeps the graph small. Povey et al. report about 11.5% relative WER reduction against cross-entropy training across five tasks, with no cross-entropy pretraining. Kaldi's chain models use this idea.

## Minimum word error rate

MMI treats every wrong transcript alike, but WER does not: one wrong word is better than five. **MWER** training minimizes the expected word errors of the model's own hypotheses:

$$
\mathcal{L}_{\text{MWER}} = \sum_{i=1}^{N} \hat{P}_i \, e_i, \qquad \hat{P}_i = \frac{P_i}{\sum_{j=1}^{N} P_j}
$$

Here $e_i$ is the word errors of hypothesis $i$ against the reference, and $P_i$ is the model's probability of it. Summing over all transcripts is out of reach, so Prabhavalkar et al. (2018) sum over an **n-best list**, the $N$ hypotheses a beam search returns. Read it aloud: the list's average error, weighted by how likely the model thinks each hypothesis is.

The renormalization matters. The list covers only part of the probability mass, so the raw $P_i$ sum to less than 1, and a loss built from them would shrink whenever the list lost mass. Dividing by the total makes the weights a distribution over the list.

### Worked example and gradient

Take three hypotheses with raw probabilities $0.30$, $0.18$, and $0.12$, which sum to $0.6$, and errors $2$, $0$, and $3$ words. The renormalized weights are $0.5$, $0.3$, and $0.2$, so the expected errors are $0.5 \cdot 2 + 0.3 \cdot 0 + 0.2 \cdot 3 = 1.6$.

To train, differentiate with respect to each hypothesis's log probability $\log P_j$. Because $\hat{P}$ is a softmax of the log probabilities, $\partial \hat{P}_i / \partial \log P_j = \hat{P}_i(\delta_{ij} - \hat{P}_j)$, and so

$$
\frac{\partial \mathcal{L}_{\text{MWER}}}{\partial \log P_j} = \sum_i e_i \hat{P}_i (\delta_{ij} - \hat{P}_j) = \hat{P}_j \,(e_j - \bar{e}), \qquad \bar{e} = \sum_i \hat{P}_i e_i
$$

Read it aloud: a hypothesis is pushed down if its errors exceed the list's average and up if they fall below, in proportion to its weight. Here $\bar{e} = 1.6$, so the gradients are $0.5 \cdot 0.4 = +0.2$, $0.3 \cdot (-1.6) = -0.48$, and $0.2 \cdot 1.4 = +0.28$. They sum to zero, which has to hold because the weights always sum to 1. In numpy:

```python
p = np.exp(logps - logsumexp(logps))
grad = p * (errors - np.sum(p * errors))
```

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Three hypotheses with renormalized probabilities 0.5, 0.3, and 0.2, word errors 2, 0, and 3, and gradients 0.2, minus 0.48, and 0.28">
<text x="14" y="26" class="fig-small fig-bold">hypothesis</text><text x="120" y="26" class="fig-small fig-bold">P over the list</text>
<text x="235" y="26" class="fig-small fig-bold">word errors</text>
<text x="460" y="26" text-anchor="middle" class="fig-small fig-bold">gradient on log p</text>
<line x1="460" y1="36" x2="460" y2="176" class="stroke-muted"/>
<text x="14" y="61" class="fig-mono">h1</text>
<rect x="120" y="44" width="60.0" height="24" class="fill-a"/>
<text x="188.0" y="61" class="fig-small">0.5</text>
<text x="262" y="61" class="fig-small">2</text>
<rect x="460.0" y="44" width="36.0" height="24" class="fill-bad"/>
<text x="504.0" y="61" text-anchor="start" class="fig-small">+0.20</text>
<text x="14" y="107" class="fig-mono">h2</text>
<rect x="120" y="90" width="36.0" height="24" class="fill-a"/>
<text x="164.0" y="107" class="fig-small">0.3</text>
<text x="262" y="107" class="fig-small">0</text>
<rect x="373.6" y="90" width="86.4" height="24" class="fill-c"/>
<text x="365.6" y="107" text-anchor="end" class="fig-small">-0.48</text>
<text x="14" y="153" class="fig-mono">h3</text>
<rect x="120" y="136" width="24.0" height="24" class="fill-a"/>
<text x="152.0" y="153" class="fig-small">0.2</text>
<text x="262" y="153" class="fig-small">3</text>
<rect x="460.0" y="136" width="50.4" height="24" class="fill-bad"/>
<text x="518.4" y="153" text-anchor="start" class="fig-small">+0.28</text>
<text x="14" y="206" class="fig-small">expected word errors = 1.6. Positive gradient: lowering that log probability lowers the loss.</text>
</svg>
Figure 2. The gradient is p times the gap between a hypothesis's errors and the list's mean, 1.6. Hypotheses worse than the mean get pushed down, h2 gets pushed up, and the three values sum to zero.
```

The subtracted $\bar{e}$ is a **baseline**. Leave it out and every hypothesis with errors gets pushed down. The update has the shape of a policy gradient with a baseline. For more information, see [Baselines and actor-critic](https://rl.ifkash.dev/lessons/031-baselines-and-actor-critic) and [RL for vision, speech, and robots](https://rl.ifkash.dev/lessons/063-rl-for-vision-speech-and-robots) in the RL course.

## Word posteriors and confusion networks

The same list shows how sure the decoder is of each word. The **posterior** of a word is the probability that it appears in a given place: the total weight of the hypotheses that contain it there. A word at 0.95 can be trusted, and one at 0.4 might trigger a clarifying question.

Comparing hypotheses place by place needs them aligned. Mangu, Brill, and Stolcke (2000) cluster the arcs of a word lattice into a chain of **slots** called a **confusion network**: each slot holds the competing words with their posteriors, and a deletion appears as `<eps>`. This chapter takes a shortcut: align every hypothesis to the 1-best with the Levenshtein alignment from the word error rate chapter, and pad gaps with `<eps>`.

Take three aligned hypotheses with probabilities $0.4$, $0.3$, and $0.3$: "call mom back", "call tom home", and "all mom home". The slot posteriors add up each word's hypotheses:

| Slot | Posteriors |
|---|---|
| 1 | `call` $0.4 + 0.3 = 0.7$, `all` $0.3$ |
| 2 | `mom` $0.4 + 0.3 = 0.7$, `tom` $0.3$ |
| 3 | `back` $0.4$, `home` $0.3 + 0.3 = 0.6$ |

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A confusion network with three slots. Slot 1: call 0.7, all 0.3. Slot 2: mom 0.7, tom 0.3. Slot 3: home 0.6, back 0.4. The consensus is call mom home.">
<text x="40" y="24" class="fig-small fig-bold">slot 1</text>
<rect x="40" y="40" width="105.0" height="28" class="fill-b"/>
<text x="46" y="59" class="fig-mono">call</text>
<text x="151.0" y="59" class="fig-small">0.7</text>
<rect x="40" y="80" width="45.0" height="28" class="fill-soft-a"/>
<text x="46" y="99" class="fig-mono">all</text>
<text x="91.0" y="99" class="fig-small">0.3</text>
<text x="230" y="24" class="fig-small fig-bold">slot 2</text>
<rect x="230" y="40" width="105.0" height="28" class="fill-b"/>
<text x="236" y="59" class="fig-mono">mom</text>
<text x="341.0" y="59" class="fig-small">0.7</text>
<rect x="230" y="80" width="45.0" height="28" class="fill-soft-a"/>
<text x="236" y="99" class="fig-mono">tom</text>
<text x="281.0" y="99" class="fig-small">0.3</text>
<text x="420" y="24" class="fig-small fig-bold">slot 3</text>
<rect x="420" y="40" width="90.0" height="28" class="fill-b"/>
<text x="426" y="59" class="fig-mono">home</text>
<text x="516.0" y="59" class="fig-small">0.6</text>
<rect x="420" y="80" width="60.0" height="28" class="fill-soft-a"/>
<text x="426" y="99" class="fig-mono">back</text>
<text x="486.0" y="99" class="fig-small">0.4</text>
<text x="40" y="150" class="fig-small">The 1-best hypothesis is call mom back, with 1.2 expected word errors.</text>
<text x="40" y="172" class="fig-small fig-bold">The consensus, call mom home, appears in no hypothesis and has 1.0.</text>
<text x="40" y="194" class="fig-small">Bar length is the slot posterior. The orange bar is the winner of each slot.</text>
</svg>
Figure 3. Each slot's bars sum to 1. Choosing the tallest bar in every slot gives a transcript that none of the three hypotheses wrote, and its expected errors are lower than the 1-best's.
```

The 1-best is "call mom back". The likeliest word in each slot gives "call mom home", which none of the three hypotheses wrote. Its expected word errors are $0.3 + 0.3 + 0.4 = 1.0$ against $0.3 + 0.3 + 0.6 = 1.2$ for the 1-best. That is the consensus idea: the most probable transcript is not always the one with the fewest expected errors, because WER counts words, not sentences. The posteriors $0.7$, $0.7$, and $0.6$ are the confidence scores. A slot whose likeliest entry is `<eps>` drops the word.

> [!WARNING]
> The slot totals sum to 1 because of the renormalization, however little of the real mass the list covers. An overconfident model makes overconfident posteriors, so calibrate them on held-out data before you act on a threshold.

> [!RECAP]
> - MMI is $-\log$ of the reference's score over the sum of all scores, each an acoustic likelihood times the language model.
> - The denominator comes from the forward algorithm. Its gradient is the occupancy $\gamma$ minus the reference one-hot.
> - The denominator needs the LM because it must match what the decoder searches. LF-MMI builds it from a phone n-gram graph.
> - MWER is the expected word errors over a renormalized n-best list, with gradient $\hat{P}_j(e_j - \bar{e})$.
> - Word posteriors sum the weights of the hypotheses that share a word in a slot. The tallest word per slot is the consensus.

## Build it step by step

The checks test the functions in this order. The file gives you `logsumexp`.

1. **`path_logscore(loglik, log_start, log_trans, path)`.** Returns a float: frame 0's start and emission scores, then a transition and an emission per later frame. The common mistake is leaving out the transitions.
2. **`mmi_denominator(loglik, log_start, log_trans)`.** The forward recursion above, in log space. The common mistake is summing probabilities, which underflow.
3. **`mmi_loss(loglik, log_start, log_trans, path)`.** The denominator minus the numerator. The common mistake is a denominator without the LM.
4. **`state_posteriors(loglik, log_start, log_trans)`.** A forward pass, a backward pass, and a softmax over states per frame, shape `(T, S)`. The common mistake is leaving the emission out of the backward step.
5. **`mmi_grad(loglik, log_start, log_trans, path)`.** `state_posteriors` minus a one-hot reference. The common mistake is flipping the sign.
6. **`mwer_loss(nbest_logps, nbest_errors)`.** A softmax over the list, then a weighted sum of errors. The common mistake is exponentiating without renormalizing.
7. **`mwer_grad(nbest_logps, nbest_errors)`.** Returns `p * (errors - mean)`, shape `(N,)`. The common mistake is `p * errors`.
8. **`word_posteriors(nbest, logps)`.** Softmax the scores, then add each hypothesis's weight to its word in each slot's dict. The common mistake is counting hypotheses instead of weighing them.
9. **`consensus_decode(posteriors)`.** The largest entry per slot, skipping `<eps>`, with its posterior.

## Where this breaks

**The list is the model's own opinion.** A hypothesis the beam missed gets no gradient. A wider beam or sampling helps.

**Alignments are approximate.** Aligning to the 1-best biases the slots toward its words, and Mangu et al. cluster lattice arcs by time and phonetic similarity to do better.

## The exercise

You build the denominator, the loss, and the gradient for the two-state toy, then MWER's loss and gradient, then word posteriors and a consensus decoder. Press **Run** to print this chapter's numbers and plot the expected errors as the error-free hypothesis gains probability.

The checks compare the denominator with a brute-force sum over all 27 paths of a three-state, three-frame problem, and both gradients with finite differences. They fail a denominator without the LM, an MWER loss without renormalization, and an MWER gradient without the mean error. The last two compare word posteriors with an enumerated oracle and require the consensus to beat the 1-best.

> [!KEY]
> What you now know: a model trained on single frames or tokens never learns which wrong answers matter. MMI compares the reference with every competitor, MWER weighs competitors by their word errors, and an n-best list gives each word a posterior you can use as confidence.

## Further reading

- Povey et al., ["Purely Sequence-Trained Neural Networks for ASR Based on Lattice-Free MMI"](https://www.isca-archive.org/interspeech_2016/povey16_interspeech.html), Interspeech 2016.
- Prabhavalkar et al., ["Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models"](https://arxiv.org/abs/1712.01818), arXiv, 2017.
- Mangu, Brill, and Stolcke, ["Finding consensus in speech recognition: word error minimization and other applications of confusion networks"](https://arxiv.org/abs/cs/0010012), *Computer Speech and Language* 14(4), 2000.
- [Speech Recognition and Understanding (11-751/18-781)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, fall 2022. The 21 November lecture, "Search", covers N-best lists and lattices. The schedule has no lecture on discriminative training.
- [ASR lectures](https://opencourse.inf.ed.ac.uk/asr/lectures), University of Edinburgh, 2026. The list has no lecture on discriminative training. Lecture 15, "Lightly supervised and semi-supervised training", lists Manohar et al. (2018), on semi-supervised training with lattice-free MMI, as background reading.
