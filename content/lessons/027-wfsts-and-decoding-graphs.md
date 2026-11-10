---
title: Weighted finite-state transducers and decoding graphs
slug: 027-wfsts-and-decoding-graphs
part: "Part 3 — Classical recognition"
summary: Compose H, C, L, and G, score every path in two semirings, and build the CTC blank topology.
minutes: 45
gpu: false
objectives:
  - Define a weighted transducer and say what the tropical and log semirings compute.
  - Compose two transducers, including an epsilon arc, and match the paths to an enumeration.
  - Find the cheapest path, the n best paths, and the log-semiring total of one graph.
  - Build toy H and C transducers and compose them with L and G in the order H, C, L, G.
  - Build a CTC topology whose accepted strings match the collapse function.
prereqs:
  - 022-hidden-markov-models
  - 025-n-gram-language-models
  - 024-hybrid-acoustic-models
exercise: 027-wfsts-and-decoding-graphs
---

# Weighted finite-state transducers and decoding graphs

A recognizer has heard something like "cat", and three pieces of knowledge still have to agree. The acoustics score phones, the dictionary says which phones spell a word, and the language model says which words are likely. 
Classical recognizers put each piece in a **weighted finite-state transducer** (WFST): a graph whose arrows read one symbol, write another, and carry a cost. Compose the graphs and a path reads phones and writes words. The transcript you want is the cheapest path. The full graph is called HCLG, after four transducers. This chapter builds all four on the two words `a` and `cat`: a lexicon $L$, a bigram $G$, and the two layers of acoustic detail $C$ and $H$ that sit in front of them.

The CTC alignment is the same kind of graph. Its strings are exactly the strings the collapse function maps to a target.

## What you already know

- Viterbi keeps the path with the smallest sum of frame costs, from [Hidden Markov models](/lessons/022-hidden-markov-models).
- A bigram scores the next word from the previous one, from [n-gram language models](/lessons/025-n-gram-language-models).
- The full graph chains four transducers: H reads senones, C reads triphones, L reads phones and writes words, and G is the language model. [Hybrid acoustic models](/lessons/024-hybrid-acoustic-models) defines senones and triphones.
- A negative log probability turns a product of probabilities into a sum of costs.
- A transducer maps an input string to an output string.

## A graph that reads one alphabet and writes another

Picture circles joined by arrows. Each arrow is one step: what it reads, what it writes, and what it costs.

A transducer here is a dict with a start state, a list of arcs, and a dict of final costs. One arc is `(src, dst, ilabel, olabel, weight)`. The weight is a **cost**, a negative log probability, so a cheap arc is a likely one. Stopping in a **final** state adds that state's final cost.

The label `"<eps>"` is **epsilon**. It reads nothing or writes nothing. The word `cat` is written on its last phone, so `k` writes epsilon. Epsilon is a string. In the CTC graph, the integer `0` is the blank, which is a real symbol.

A **semiring** is the pair of operations that combine costs (Mohri, Pereira, and Riley, 2002). $\otimes$ extends one path by one more arc. $\oplus$ combines alternative paths. Both semirings here add costs for $\otimes$ and differ in $\oplus$.

The **tropical** semiring is Viterbi. Alternatives keep the smaller cost:

$$
x \oplus y = \min(x, y), \qquad x \otimes y = x + y
$$

Read it aloud: the better of two costs is the minimum, and two steps in a row add.

The **log** semiring adds the probabilities back up, which is what the forward algorithm does. Costs stay negative log probabilities:

$$
x \oplus_{\log} y = -\log(e^{-x} + e^{-y}), \qquad x \otimes y = x + y
$$

Read it aloud: turn each cost back into a probability, add, and take the negative log again. In numpy:

```python
total = -np.logaddexp.reduce(-np.array(costs))
```

> [!KEY]
> Tropical $\oplus$ is the minimum, so the winner is the cheapest path. Log $\oplus$ folds every path's probability into one cost. Both semirings add costs along a single path.

## Compose a lexicon with a bigram

Composition joins two transducers on a shared alphabet. $L$ writes words and $G$ reads them. When an output of $L$ equals an input of $G$, and that symbol is not epsilon, you take both arcs. The composed arc reads what $L$ read, writes what $G$ wrote, and costs the sum.

Epsilon is the case a naive zipper drops. Delete every arc that touches `"<eps>"` and `k` vanishes, so `cat` can no longer be written. The rule is three lines:

- An output epsilon on $L$ moves only $L$. $G$ stays put, and the composed arc writes epsilon.
- An input epsilon on $G$ moves only $G$. $L$ stays put, and the composed arc reads epsilon.
- Take each epsilon step once, outside the loop over the other transducer's arcs.

### Work the composition by hand

$L$ has two pronunciations. `ah` writes `a` at cost $0.2$. `k` writes epsilon at cost $0.1$, then `t` writes `cat` at cost $0.3$. State 2 is final at cost $0$.

$G$ has a start state and one state per word. From the start, `a` costs $0.5$ and ends at final cost $0.4$. `cat` costs $0.8$ and ends at final cost $0.2$. One more arc, `cat` after `a`, costs $0.3$. This lexicon writes one word and stops, so that arc never appears in the composition.

Two paths survive, and Figure 1 draws them.

`ah` matches `a`. The arc costs $0.2 + 0.5 = 0.7$ and the final cost is $0.4$, so the path costs $1.1$.

`k t` takes the epsilon first. `k:eps` costs $0.1$ and leaves $G$ at the start. `t:cat` matches at $0.3 + 0.8 = 1.1$, and the final cost is $0.2$. The path costs $0.1 + 1.1 + 0.2 = 1.4$.

```figure
<svg viewBox="0 0 640 320" role="img" aria-label="Composed lexicon and bigram. The path ah to a costs 1.1. The path k t to cat costs 1.4.">
  <defs>
    <marker id="c025-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" class="fill-ink"/>
    </marker>
  </defs>
  <text x="320" y="22" text-anchor="middle" class="fig-bold">L composed with G, phones in and words out</text>
  <line x1="111.5" y1="163.1" x2="496.5" y2="82.9" class="stroke-b" marker-end="url(#c025-arrow)"/>
  <text x="270" y="86" text-anchor="middle" class="fig-mono">ah:a  0.7</text>
  <text x="270" y="104" text-anchor="middle" class="fig-small">path cost 1.1</text>
  <line x1="110.4" y1="176.7" x2="277.6" y2="241.3" class="stroke-a" marker-end="url(#c025-arrow)"/>
  <text x="176" y="236" text-anchor="middle" class="fig-mono fig-small">k:eps  0.1</text>
  <line x1="324.0" y1="250.0" x2="496.0" y2="250.0" class="stroke-b" marker-end="url(#c025-arrow)"/>
  <text x="408" y="236" text-anchor="middle" class="fig-mono">t:cat  1.1</text>
  <text x="408" y="300" text-anchor="middle" class="fig-small">path cost 1.4</text>
  <circle cx="88" cy="168" r="24" class="fill-soft-a stroke-a"/>
  <text x="88" y="173" text-anchor="middle" class="fig-bold">0</text>
  <text x="88" y="210" text-anchor="middle" class="fig-small">start</text>
  <circle cx="300" cy="250" r="24" class="fill-soft-b stroke-b"/>
  <text x="300" y="255" text-anchor="middle" class="fig-bold">1</text>
  <text x="300" y="292" text-anchor="middle" class="fig-small">after k</text>
  <circle cx="520" cy="78" r="24" class="fill-soft-c stroke-c"/>
  <circle cx="520" cy="78" r="18" class="stroke-c" fill="none"/>
  <text x="520" y="83" text-anchor="middle" class="fig-bold">a</text>
  <text x="520" y="44" text-anchor="middle" class="fig-small">final +0.4</text>
  <circle cx="520" cy="250" r="24" class="fill-soft-c stroke-c"/>
  <circle cx="520" cy="250" r="18" class="stroke-c" fill="none"/>
  <text x="520" y="255" text-anchor="middle" class="fig-bold">cat</text>
  <text x="520" y="216" text-anchor="middle" class="fig-small">final +0.2</text>
</svg>
Figure 1. The lexicon composed with the bigram. Notice that k:eps writes nothing and leaves the bigram at its start, which is what makes cat reachable. The path that writes a costs 1.1, and the path that writes cat costs 1.4.
```

The checks compare paths, not state ids. The match is one condition, and the epsilon steps sit outside that loop:

```python
if out_a == in_b and out_a != "<eps>":
    add(next_a, next_b, in_a, out_b, w_a + w_b)
```

A product state is final when both states are final, and the final cost is the sum.

> [!WARNING]
> If both transducers have an epsilon at the same pair of states, either order builds a path, and the two paths share labels and cost. The log semiring then counts that pronunciation twice. $G$ here has no input epsilons, so each pronunciation is one path.

## The cheapest path is Viterbi

Figure 1 costs $1.1$ and $1.4$. Tropical search keeps $1.1$, the transcript `a`. That is Viterbi on a graph that already holds the dictionary and the language model.

**n-best** asks for more than the winner: `a` at $1.1$, then `cat` at $1.4$. CMU 11-751's search lecture on November 21, 2022, covers n-best and lattices.

The log-semiring total puts both paths back together:

$$
1.1 \oplus_{\log} 1.4 = -\log(e^{-1.1} + e^{-1.4}) = 0.5456
$$

The total, $0.5456$, is smaller than either path because two ways to finish hold more probability than one. The forward algorithm is this total on paths of a fixed length.

> [!WARNING]
> Keeping the largest sum of arc weights treats costs as scores. Paths $0.2 + 0.2$ and $0.9$ then prefer $0.9$, but the tropical path costs $0.4$. Forgetting the final cost fails the other way: a cheap arc into an expensive final state can beat a dearer arc into a cheap one.

> [!KEY]
> `shortest_path` is the minimum sum, final cost included. `total_weight` is the log-semiring sum of every accepting path. `n_best` lists those paths, cheapest first.

## Add context and HMM states with C and H

A hybrid network scores a *senone* on every frame, the tied triphone HMM state from [Hybrid acoustic models](/lessons/024-hybrid-acoustic-models), not a phone. Two more transducers close the gap. **C** reads triphones and writes the plain phones $L$ reads. **H** reads senones and writes triphones. Mohri, Pereira, and Riley (section 3.1) define C as a map from context-dependent to context-independent phones. The whole graph is

$$
\mathrm{HCLG} = H \circ C \circ L \circ G
$$

Read it right to left: $G$ allows word sequences, $L$ spells words as phones, $C$ spells phones as triphones, and $H$ spells triphones as senones. Each output alphabet is the next input alphabet. Compose $C$ with $H$ and C's phone outputs never meet H's senone inputs, so no path survives.

### C remembers a window of three phones

A triphone `a-b+c` is phone `b` with left neighbor `a` and right neighbor `c`. Neighboring triphones overlap: after `sil-k+t` the next one starts `k-t+`. State $(a, b)$ in C means the next triphone is `a-b+c` for some $c$. Reading it writes `b` and moves to $(b, c)$. The first triphone has `sil` on its left, and the string ends when the right neighbor is `sil`.

For the phones `k t`, the path reads `sil-k+t` and writes `k`, reaches state $(k, t)$, reads `k-t+sil` and writes `t`, and ends. The string `sil-k+t`, `ah-t+sil` has no path, because $(k, t)$ reads only `k-t+c`. A context-free C, one looping state that writes each triphone's middle phone, accepts that string. It also accepts a first triphone `ah-k+t`, which puts `ah` where silence stands. C sees the whole phone string, so a word's last phone takes the next word's first phone as its right neighbor.

### H reads senones

Give each triphone two HMM states with these senone ids:

| Triphone | Senones |
|---|---|
| `sil-ah+sil` | 0, 1 |
| `sil-k+t` | 2, 3 |
| `k-t+sil` | 4, 1 |

The two triphones with `sil` on the right share senone 1. H is one chain per triphone. The chain for `sil-k+t` reads 2 and writes epsilon, then reads 3 and writes `sil-k+t`, as `k` and `t` did in the lexicon. The last arc returns to the start, so triphones follow one another. Self-loops stay out of H, and the decoder simulates them (Mohri, Pereira, and Riley, section 3.1).

### Read senones as words

The senones $2, 3, 4, 1$ become `sil-k+t`, `k-t+sil` through H, `k`, `t` through C, and `cat` through L and G. The path costs $1.4$, as in Figure 1, because H and C arcs cost $0$ here. The senones $0, 1$ read `a` at $1.1$. The senones $2, 3, 0, 1$ have no path, because `sil-k+t` and `sil-ah+sil` do not overlap.

## Determinization and minimization shrink the search

A transducer is **deterministic** when each state has at most one arc for a given input label, and no input epsilon (Mohri, Pereira, and Riley, 2002). The search then holds one hypothesis per symbol.

Two arcs labeled `a` cost $0.4$ and $1.0$. The cheap one is followed by `x` at $1.0$, and the dear one by `y` at $0.2$. Determinization keeps one `a` arc at $0.4$ and pushes the leftover $0.6$ onto `y`, which costs $0.8$. The paths `ax` at $1.4$ and `ay` at $1.2$ cost the same as before. **Minimization** merges states with the same future.

Homophones, and a pronunciation that is a prefix of another word, do not determinize. Mohri, Pereira, and Riley end each pronunciation with an auxiliary symbol, and Kaldi uses `#1` and `#2`.

## A CTC topology is a transducer

[CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label) derives the loss. Its collapse rule is enough to see that the alignment graph is a transducer.

**Collapse** merges each run of equal symbols into one, then drops the blank. With blank `0`, `(1, 1)` collapses to `[1]` and `(1, 0, 1)` to `[1, 1]`. The check calls that chapter's `collapse`.

The **CTC topology** for a target accepts a string exactly when collapse sends it to that target. Every arc costs $0$, and the input equals the output, so it is an acceptor drawn as a transducer. For `[1, 1]` the states are a leading blank, the first `1`, a blank between the copies, the second `1`, and a trailing blank. Each symbol has a self-loop, because a label or a blank can last more than one frame. There is no arc from the first `1` to the second `1`, so `(1, 1)` is rejected.

When neighbors differ, as in `[1, 2]`, both routes are legal: a direct step, and a blank between them. Both strings collapse to `[1, 2]`.
> [!RECAP]
> - An arc reads a label, writes a label, and adds a cost. Epsilon is the string `"<eps>"`.
> - Tropical $\oplus$ is $\min$, log $\oplus$ is $-\log(e^{-x}+e^{-y})$, and $\otimes$ is $+$.
> - Composition pairs equal non-epsilon labels and steps over an epsilon on one side.
> - C reads triphones and writes phones through an overlapping window, H reads senones and writes triphones, and HCLG composes $H \circ C \circ L \circ G$.
> - The CTC topology accepts a string exactly when the CTC chapter's `collapse` returns the target.

## Build it step by step

The checks call the functions in this order. The first check only calls `compose`.

1. **`compose(a, b)`.** Pair arcs whose labels match and are not `"<eps>"`, and add the costs. Step an output epsilon of `a`, and an input epsilon of `b`, once each, outside the other loop. A product state is final when both states are. The usual mistake is dropping epsilon arcs, which loses `k t` → `cat`.
2. **`shortest_path(fst)`.** Return `(cost, inputs, outputs)` for the least cost. Add the final cost, and drop `"<eps>"` from the tuples. The usual mistake is keeping the maximum sum, or forgetting the final cost. No path returns `(math.inf, (), ())`.
3. **`n_best(fst, n)`.** List the accepting paths of an acyclic graph, sort by increasing cost, and keep `n`. The usual mistake is the right paths in the wrong order.
4. **`total_weight(fst)`.** Return `-np.logaddexp.reduce(-costs)`. The usual mistake is calling `logaddexp` on the costs themselves. No path returns `math.inf`.
5. **`context_transducer(phones, boundary="sil")`.** Build C. The start state reads `sil-b+c` for each phone `b`. State `(a, b)` reads `a-b+c`, writes `b`, and goes to `(b, c)`, or to a final end state when `c` is the boundary. The usual mistake is one looping state that ignores the window.
6. **`hmm_transducer(senones)`.** Build H from a dict of triphone to senone ids. Each triphone is a chain of arcs. The last arc writes the triphone and returns to state 0, and the earlier arcs write epsilon. The usual mistake is one state per senone, which merges the shared senone 1.
7. **`hclg(h, c, l, g)`.** Compose the four, `h` first. The usual mistake is the reverse order, which leaves no path.
8. **`ctc_topology(target, blank=0)`.** Alternate blank states and label states. Self-loops repeat a symbol. Skip the blank between two labels only when the labels differ. Every arc costs $0$ and carries one symbol on both sides. The usual mistake is an arc between two copies of the same label, which accepts `(1, 1)` for `[1, 1]`.

## Where this breaks

**Graph size.** A naive $L \circ G$ can hold a state for every pair of states, so Kaldi determinizes and minimizes after each composition.

**Cycles.** `n_best` and `total_weight` list paths, so the graph is acyclic. The CTC topology has self-loops, so you test it by which strings it accepts.

## The exercise

You build `compose`, `shortest_path`, `n_best`, `total_weight`, `context_transducer`, `hmm_transducer`, `hclg`, and `ctc_topology`. Press **Run** to compose the lexicon with the bigram, print both paths, and plot their costs. It also reads the senones $2, 3, 4, 1$ through HCLG.

The checks enumerate path pairs for an epsilon-free pair and for the lexicon, including the epsilon on `k`. They compare `shortest_path` with the minimum sum, including a final cost that changes the winner, and `total_weight` with the log-semiring sum of those paths. `context_transducer` must reject triphone strings whose windows do not overlap, and `hclg` must read senones $2, 3, 4, 1$ as `cat` at $1.4$. `ctc_topology` is checked on every short string by the CTC chapter's `collapse`.

## Further reading

- [Weighted Finite-State Transducers in Speech Recognition](https://cs.nyu.edu/~mohri/pub/csl01.pdf), Mehryar Mohri, Fernando Pereira, and Michael Riley, Computer Speech and Language. Section 3.1 builds $H \circ C \circ L \circ G$.
- [ASR with WFSTs](https://opencourse.inf.ed.ac.uk/asr/lectures), Peter Bell, ASR lecture 9, University of Edinburgh, 9 February 2026.
- [Speech Recognition and Understanding (11-751)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, 2022. The 21 November lecture is "Search."
