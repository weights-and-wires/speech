---
title: Decoding CTC
slug: 029-decoding-ctc
part: "Part 4 — Neural speech recognition"
summary: Turn CTC emissions into text with greedy decoding, then with prefix beam search, which adds up paths the way the CTC loss does.
minutes: 40
gpu: false
objectives:
  - Decode CTC emissions greedily, and explain why the best path is not always the most likely transcript.
  - Track two probabilities per prefix, one for paths ending in a blank and one for paths ending in a label.
  - Build CTC prefix beam search and check it against brute-force enumeration of every path.
  - Explain why a narrow beam returns lower bounds and a wide beam returns exact probabilities.
prereqs:
  - 027-ctc-many-paths-one-label
  - 028-ctc-forward-backward
exercise: 029-decoding-ctc
---

# Decoding CTC

A trained CTC model hands you a table, not a sentence. For every frame of audio it gives a probability for every symbol, plus the blank. Training only needed the probability of the right transcript. At test time you do not know the transcript, so you have to search for the one the table likes best.

The obvious search reads the table one frame at a time and takes the winner in each. That works surprisingly often, and fails in a specific, instructive way: thousands of frame-level paths collapse to the same text, and the text with the best single path is not always the most likely text.

In this chapter you build the obvious decoder, see exactly where it fails, and then build **prefix beam search**, which adds up paths the way the CTC loss does but keeps only the most promising few.

## What you already know

- A CTC model emits $y^t_k$, the probability of symbol $k$ at frame $t$, with the blank $\varnothing$ at index 0.
- The collapse function $\mathcal{B}$ merges repeated symbols, then deletes blanks, so $\mathcal{B}(\texttt{aa\_ab\_\_}) = \texttt{aab}$.
- The probability of a transcript is the sum, over every path that collapses to it, of the product of the per-frame probabilities, from [CTC: many paths, one label](/lessons/027-ctc-many-paths-one-label).
- `np.logaddexp(a, b)` computes $\log(e^a + e^b)$ without overflow, and handles $-\infty$.

## Greedy decoding

The simplest decoder picks the most likely symbol in each frame, then collapses the resulting path. This is **greedy** or **best-path** decoding:

$$
\hat{\pi}_t = \operatorname*{arg\,max}_k \; y^t_k, \qquad \hat{\mathbf{l}} = \mathcal{B}(\hat{\pi}_1, \dots, \hat{\pi}_T)
$$

Read it aloud: for each frame $t$, the chosen symbol $\hat\pi_t$ is the index $k$ with the largest probability, and the transcript $\hat{\mathbf{l}}$ is the collapse of that path. In numpy it is two lines:

```python
path = np.argmax(log_probs, axis=1)          # (T,) symbol indices
labels = [s for i, s in enumerate(path) if s != 0 and (i == 0 or s != path[i - 1])]
```

Take the frame winners `a a _ a b b _ b`. Merging runs gives `a _ a b _ b`, and deleting blanks gives `aabb`. The blank between the two `a` runs keeps both of them.

Greedy decoding works well on a trained model because trained CTC models are **peaky**. They predict blank with probability near 1 in most frames, and each label appears as a spike one or two frames wide. When every frame is nearly one-hot, the best path carries almost all of the probability, and nothing else could win.

## Where the best path goes wrong

Peakiness is not a guarantee. Take two frames, each with $y_\varnothing = 0.6$ and $y_a = 0.4$. Greedy picks blank twice and returns the empty string. The path `__` has probability $0.6 \times 0.6 = 0.36$.

But three paths collapse to `a`: `a_`, `_a`, and `aa`. Their probabilities are $0.4 \times 0.6 = 0.24$, $0.6 \times 0.4 = 0.24$, and $0.4 \times 0.4 = 0.16$, which add to $0.64$. Figure 1 draws both. The transcript `a` is almost twice as likely as the empty one, though no single path to it beats `__`.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Probability of the empty transcript, 0.36 from one path, against the transcript a, 0.64 from three paths, drawn as stacked bars">
  <text x="14" y="28" class="fig-small">Two frames, each with P(blank) = 0.6 and P(a) = 0.4. One bar unit is probability, 500 wide for 1.0.</text>
  <text x="14" y="82" class="fig-bold">empty</text>
  <rect x="110" y="60" width="180" height="34" rx="3" class="fill-soft-a stroke-a"/>
  <text x="200" y="82" text-anchor="middle" class="fig-mono">__ 0.36</text>
  <text x="14" y="142" class="fig-bold">a</text>
  <rect x="110" y="120" width="120" height="34" rx="3" class="fill-soft-b stroke-b"/>
  <text x="170" y="142" text-anchor="middle" class="fig-mono">a_ 0.24</text>
  <rect x="230" y="120" width="120" height="34" rx="3" class="fill-soft-b stroke-b"/>
  <text x="290" y="142" text-anchor="middle" class="fig-mono">_a 0.24</text>
  <rect x="350" y="120" width="80" height="34" rx="3" class="fill-soft-b stroke-b"/>
  <text x="390" y="142" text-anchor="middle" class="fig-mono">aa 0.16</text>
  <text x="440" y="142" class="fig-bold">= 0.64</text>
  <text x="14" y="196">Greedy decoding follows the tallest single bar, __, and returns the empty string.</text>
  <text x="14" y="218">The transcript with the most total probability is a.</text>
</svg>
Figure 1. The tallest single bar, the path __, spells the empty transcript. The transcript a is split across three shorter bars that add to more.
```

> [!KEY]
> The best path is not the best transcript. A transcript's probability is spread over many paths, and greedy decoding looks at only one of them.

## Search over prefixes, not paths

The fix is to search over transcripts. You cannot score every transcript, because there are too many, but you can build them left to right. **Prefix beam search** walks the frames in order and keeps a short list, the **beam**, of the most likely **prefixes**: collapsed partial transcripts such as `th` or `the c`. Each prefix carries the total probability of every path so far that collapses to it. After each frame, it keeps only the best $K$ prefixes, where $K$ is the **beam width**.

One number per prefix is not enough. Suppose the prefix is `a` and the next frame emits `a`. If the last frame was also `a`, the two merge and the prefix stays `a`. If a blank came in between, the next `a` starts a second letter and the prefix becomes `aa`. The prefix alone does not say which case you are in, so each prefix $\ell$ keeps two probabilities:

- $p_b(\ell)$, the probability of the paths that collapse to $\ell$ and end in a blank.
- $p_{nb}(\ell)$, the probability of the paths that collapse to $\ell$ and end in its last label.

The search starts with the empty prefix, where $p_b(\varnothing) = 1$ and $p_{nb}(\varnothing) = 0$. At frame $t$, every prefix $\ell$ in the beam sends probability forward in four ways:

$$
\begin{aligned}
p^{t}_b(\ell) &\mathrel{+}= \big(p^{t-1}_b(\ell) + p^{t-1}_{nb}(\ell)\big)\, y^t_\varnothing \\
p^{t}_{nb}(\ell + c) &\mathrel{+}= \big(p^{t-1}_b(\ell) + p^{t-1}_{nb}(\ell)\big)\, y^t_c && c \ne \text{last}(\ell) \\
p^{t}_{nb}(\ell + c) &\mathrel{+}= p^{t-1}_b(\ell)\, y^t_c && c = \text{last}(\ell) \\
p^{t}_{nb}(\ell) &\mathrel{+}= p^{t-1}_{nb}(\ell)\, y^t_c && c = \text{last}(\ell)
\end{aligned}
$$

Read it aloud, one line at a time:

1. A blank leaves the prefix unchanged, and every path to $\ell$ can take it, so all the old probability moves into the ends-in-blank bucket.
2. A label $c$ different from the prefix's last label extends the prefix to $\ell + c$, from either bucket.
3. Repeating the last label after a blank starts another copy, so only $p_b$ can make $\ell + c$.
4. Repeating it straight after the label merges into the same prefix $\ell$, so only $p_{nb}$ takes that route.

The score of a prefix is $p_b(\ell) + p_{nb}(\ell)$. Real decoders work in log space, where every `+=` becomes a `logaddexp` and every product becomes a sum. For one extension it looks like this:

```python
p_all = np.logaddexp(p_b, p_nb)                  # log(p_b + p_nb)
b, nb = nxt[prefix + (c,)]                       # defaults to (-inf, -inf)
nxt[prefix + (c,)] = (b, np.logaddexp(nb, p_all + frame[c]))
```

### Work the two-frame example by hand

Use the two frames from Figure 1, with a beam wide enough to keep everything. After frame 1, starting from $p_b(\varnothing) = 1$:

- A blank keeps the empty prefix: $p_b(\varnothing) = 1 \times 0.6 = 0.6$.
- The label `a` makes the prefix `a`: $p_{nb}(a) = 1 \times 0.4 = 0.4$.

After frame 2:

- The empty prefix takes another blank: $p_b(\varnothing) = 0.6 \times 0.6 = 0.36$.
- The empty prefix emits `a`: $p_{nb}(a) \mathrel{+}= 0.6 \times 0.4 = 0.24$.
- The prefix `a` takes a blank: $p_b(a) = (0 + 0.4) \times 0.6 = 0.24$.
- The prefix `a` repeats `a` with no blank between, which merges: $p_{nb}(a) \mathrel{+}= 0.4 \times 0.4 = 0.16$.
- The prefix `a` repeats `a` after a blank, which would make `aa`: $p_{nb}(aa) = p_b(a) \times 0.4 = 0 \times 0.4 = 0$.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Prefix beam search on two frames: prefixes empty, a, and aa with their blank-ending and label-ending probabilities after each frame">
  <defs><marker id="c028-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
  <text x="20" y="22" class="fig-small">start</text>
  <text x="240" y="22" class="fig-small">after frame 1</text>
  <text x="460" y="22" class="fig-small">after frame 2</text>
  <rect x="20" y="40" width="120" height="60" rx="5" class="fill-soft-a stroke-a"/>
  <text x="80" y="58" text-anchor="middle" class="fig-bold">empty</text>
  <text x="80" y="75" text-anchor="middle" class="fig-small fig-mono">p_b = 1</text>
  <text x="80" y="91" text-anchor="middle" class="fig-small fig-mono">p_nb = 0</text>
  <rect x="240" y="40" width="120" height="60" rx="5" class="fill-soft-a stroke-a"/>
  <text x="300" y="58" text-anchor="middle" class="fig-bold">empty</text>
  <text x="300" y="75" text-anchor="middle" class="fig-small fig-mono">p_b = 0.6</text>
  <text x="300" y="91" text-anchor="middle" class="fig-small fig-mono">p_nb = 0</text>
  <rect x="240" y="130" width="120" height="60" rx="5" class="fill-soft-a stroke-a"/>
  <text x="300" y="148" text-anchor="middle" class="fig-bold">a</text>
  <text x="300" y="165" text-anchor="middle" class="fig-small fig-mono">p_b = 0</text>
  <text x="300" y="181" text-anchor="middle" class="fig-small fig-mono">p_nb = 0.4</text>
  <rect x="460" y="40" width="120" height="60" rx="5" class="fill-soft-a stroke-a"/>
  <text x="520" y="58" text-anchor="middle" class="fig-bold">empty</text>
  <text x="520" y="75" text-anchor="middle" class="fig-small fig-mono">p_b = 0.36</text>
  <text x="520" y="91" text-anchor="middle" class="fig-small fig-mono">p_nb = 0</text>
  <rect x="460" y="130" width="120" height="60" rx="5" class="fill-soft-a stroke-a"/>
  <text x="520" y="148" text-anchor="middle" class="fig-bold">a</text>
  <text x="520" y="165" text-anchor="middle" class="fig-small fig-mono">p_b = 0.24</text>
  <text x="520" y="181" text-anchor="middle" class="fig-small fig-mono">p_nb = 0.40</text>
  <rect x="460" y="220" width="120" height="60" rx="5" class="fill-sunk stroke-muted dashed"/>
  <text x="520" y="238" text-anchor="middle" class="fig-bold">aa</text>
  <text x="520" y="255" text-anchor="middle" class="fig-small fig-mono">p_b = 0</text>
  <text x="520" y="271" text-anchor="middle" class="fig-small fig-mono">p_nb = 0</text>
  <line x1="140" y1="64" x2="238" y2="64" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="150" y="56" class="fig-small">blank</text>
  <line x1="140" y1="86" x2="238" y2="140" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="150" y="126" class="fig-small">a</text>
  <line x1="360" y1="64" x2="458" y2="64" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="370" y="56" class="fig-small">blank</text>
  <line x1="360" y1="86" x2="458" y2="138" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="372" y="120" class="fig-small">a</text>
  <line x1="360" y1="166" x2="458" y2="166" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="366" y="160" class="fig-small">blank</text>
  <line x1="360" y1="186" x2="458" y2="236" class="stroke-ink" marker-end="url(#c028-arrow)"/>
  <text x="240" y="222" class="fig-small">a after a blank</text>
  <text x="20" y="306" class="fig-small">The merge a to a (no blank between) adds 0.4 × 0.4 = 0.16 to p_nb of a, with no arrow.</text>
  <text x="20" y="324" class="fig-small">The aa box has probability 0, so the decoder drops it.</text>
</svg>
Figure 2. The beam for the two-frame example. Following the arrows, the prefix a ends with 0.24 + 0.40 = 0.64, and the empty prefix ends with 0.36, the exact answers from the brute-force sum.
```

The beam ends with `a` at $0.24 + 0.24 + 0.16 = 0.64$ and the empty prefix at $0.36$, which is the brute-force answer. The prefix `aa` has probability 0, because two frames cannot hold `a_a`, so the decoder drops it. With an unlimited beam, prefix beam search gives the exact CTC probability of every transcript. With a beam of $K$, it drops prefixes whose paths might have come back later, so every score it returns is a lower bound on the true probability. The following demo shows the beam at work on a short table.

```demo
{"id": "beam-search", "caption": "Shrink the beam width to 1 and watch cat get pruned down to c, then widen the beam until its answer matches the beam of 256."}
```

> [!KEY]
> Prefix beam search adds up paths the way the CTC loss does, but keeps only the best few prefixes. Two probabilities per prefix, ending in a blank or not, are what make repeated letters come out right.

> [!WARNING]
> A repeated label is the classic bug. Extending `a` by `a` from $p_{nb}$ must stay `a`, and only $p_b$ may make `aa`. Get this wrong and "hello" decodes as "helo" or "hellllo".

> [!RECAP]
> - Greedy decoding is `argmax` per frame, then merge repeats, then drop blanks.
> - Prefix beam search keeps $p_b$ and $p_{nb}$ per prefix, updates them with `np.logaddexp`, and keeps the top $K$ by score after every frame.
> - A repeated label extends the prefix only from $p_b$. From $p_{nb}$ it merges.
> - With a wide beam the scores are exact, and with a narrow beam they are lower bounds.

## Build it step by step

The checks test the functions in this order.

1. **`greedy_decode(log_probs, blank)`.** Takes a `(T, C)` array and returns a list of ints. Use `np.argmax(..., axis=1)`, then walk the path once. The common mistake is dropping blanks before merging, which turns `a _ a` into `a`.
2. **`prefix_beam_search(log_probs, beam_width, blank, prefix_score)`.** Returns a list of `(prefix, score)` pairs, best first. Start from `{(): (0.0, -inf)}`. For each frame, build a `defaultdict` of next-frame `(p_b, p_nb)` pairs from the four rules, rank by `np.logaddexp(p_b, p_nb) + prefix_score(prefix)`, and keep the top `beam_width`. Drop prefixes whose score is `-inf`. The common mistake is letting the third rule extend from `p_nb`.

The optional `prefix_score` is a function from a prefix to a float that the beam adds to each prefix's score. It is `None` here, which means zero. The next chapter plugs a language model into it.

## Cost and scaling

The following table uses $T$ frames, $C$ symbols, and a beam of $K$.

| Decoder | Work | Memory |
|---|---|---|
| Brute force over paths | $C^T$ paths, $3^5 = 243$ in the check, about $28^{100}$ for two seconds of characters | $O(T)$ |
| Greedy | $O(TC)$ | $O(T)$ |
| Prefix beam search | $O(TKC)$ updates, plus sorting $KC$ candidates per frame | $O(KC)$ prefixes |

A wav2vec 2.0 model emits 50 frames per second over 32 symbols, so a beam of 100 extends 3,200 candidates per frame. Real decoders extend only symbols above a floor, and they skip frames where blank is almost certain.

## Where this breaks

**The beam loses the right answer early.** A prefix pruned at frame 10 can never come back. Widen the beam until the output stops changing on held-out data. The cost grows linearly with $K$.

**The emissions are wrong, not the search.** No decoder recovers a word the model gave no probability. When beam search and greedy agree on a wrong answer, look at the model.

**The model has no idea what is a word.** The beam scores spelling only by the acoustics, so "the kat" can outscore "the cat" when the audio is unclear. [Language model fusion and hotwords](/lessons/030-language-model-fusion) adds that knowledge.

## The exercise

You build `greedy_decode` and `prefix_beam_search`. Press **Run** to decode a synthetic sentence both ways. You see the posteriorgram and the blank's probability per frame, and you hear the greedy path as one beep per non-blank frame.

The checks confirm greedy decoding on a hand-built path and on the two-frame counterexample. They compare every score of a wide beam against brute-force enumeration of all $3^5$ paths, and confirm that a narrow beam returns lower bounds, best first. They check that an extra prefix score is added to every prefix, and that on peaky emissions the beam and greedy decoding agree.

> [!KEY]
> What you now know: greedy decoding follows one path, but a transcript's probability is spread over many, so you search over prefixes and track blank-ending and label-ending probability separately. That answers the opening question: the best table of probabilities does not give you the best sentence until you add the paths up.
