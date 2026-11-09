---
title: Weighted finite-state transducers and decoding graphs
slug: 026-wfsts-and-decoding-graphs
part: "Part 3 — Classical recognition"
summary: Compose a lexicon with a bigram, score every path in two semirings, and build the CTC blank topology.
minutes: 45
gpu: false
objectives:
  - Define a weighted transducer and say what the tropical and log semirings compute.
  - Compose two transducers, including an epsilon arc, and match the paths to an enumeration.
  - Find the cheapest path, the n best paths, and the log-semiring total of one graph.
  - Build a CTC topology whose accepted strings match the collapse function.
prereqs:
  - 021-hidden-markov-models
  - 024-n-gram-language-models
  - 025-word-error-rate
exercise: 026-wfsts-and-decoding-graphs
---

# Weighted finite-state transducers and decoding graphs

A recognizer has heard something like "cat", and three pieces of knowledge still have to agree. The acoustics score phones, the dictionary says which phones spell a word, and the language model says which words are likely. Separate searches can discard a good phone string before the dictionary votes.

Classical recognizers put each piece in a **weighted finite-state transducer** (WFST): a graph whose arrows read one symbol, write another, and carry a cost. Compose the graphs and a path reads phones and writes words. The transcript you want is the cheapest path. The word-error chapter names the full graph HCLG. This chapter builds the middle of it, a lexicon $L$ composed with a bigram $G$, on the two words `a` and `cat`.

The same kind of graph is the CTC alignment. The blank, and the extra blank a repeated label needs, are a transducer. Its strings are exactly the strings the collapse function maps to a target.

## What you already know

- Viterbi keeps the path with the smallest sum of frame costs, from [Hidden Markov models](/lessons/021-hidden-markov-models).
- A bigram scores the next word from the previous one, from [n-gram language models](/lessons/024-n-gram-language-models).
- H reads HMM states, C reads context-dependent phones, L reads phones and writes words, and G is the language model, from [Word error rate](/lessons/025-word-error-rate).
- A negative log probability turns a product of probabilities into a sum of costs.
- A transducer maps an input string to an output string (Jurafsky and Martin, 2009, section 3.4).

## A graph that reads one alphabet and writes another

Picture a few circles joined by arrows. Each arrow is one step the recognizer might take: what it reads, what it writes, and what that step costs.

A transducer here is a dict with a start state, a list of arcs, and a dict of final costs. One arc is `(src, dst, ilabel, olabel, weight)`. The weight is a **cost**, a negative log probability, so a cheap arc is a likely one. These decimals are small so you can add them by hand. Stopping in a **final** state adds that state's final cost.

The label `"<eps>"` is **epsilon**. It reads nothing or writes nothing. The phone `k` in `cat` does not itself write the word. The word is written on the last phone, and `k` writes epsilon. Epsilon is a string. Later, in the CTC graph, the integer `0` is the blank, which is a real symbol.

A **semiring** is the pair of operations that combine costs (Mohri, Pereira, and Riley, 2002). $\otimes$ extends one path by one more arc. $\oplus$ combines alternative paths. Both semirings here use addition as $\otimes$, because a negative log turns a product into a sum. They differ in $\oplus$.

The **tropical** semiring is Viterbi. Alternatives keep the smaller cost:

$$
x \oplus y = \min(x, y), \qquad x \otimes y = x + y
$$

Read it aloud: the better of two costs is the minimum, and two steps in a row add. In Python the winner is `min(costs)`.

The **log** semiring adds the probabilities back up, which is what the forward algorithm does. Costs stay negative log probabilities:

$$
x \oplus_{\log} y = -\log(e^{-x} + e^{-y}), \qquad x \otimes y = x + y
$$

Read it aloud: turn each cost back into a probability, add those probabilities, and take the negative log again. One path is still a sum. Many paths combine with $\oplus_{\log}$. In numpy:

```python
total = -np.logaddexp.reduce(-np.array(costs))
```

> [!KEY]
> Tropical $\oplus$ is the minimum, so the winner is the cheapest path. Log $\oplus$ folds every path's probability into one cost. Both semirings add costs along a single path.

k2 stores the opposite sign. Its scores are log probabilities, so tropical takes a max and the log semiring is $\log(e^{a} + e^{b})$ (k2 core concepts). Negate those scores and you get the costs in this chapter.

## Compose a lexicon with a bigram

Composition joins two transducers on a shared alphabet. $L$ writes words and $G$ reads them. When an output of $L$ equals an input of $G$, and that symbol is not epsilon, you take both arcs. The composed arc reads what $L$ read, writes what $G$ wrote, and costs the sum (Mohri, Pereira, and Riley, 2002).

Epsilon is the case a naive zipper drops. Delete every arc that touches `"<eps>"` and the phone `k` vanishes, so `cat` can no longer be written. The rule is three lines:

- An output epsilon on $L$ moves only $L$. $G$ stays put, and the composed arc writes epsilon.
- An input epsilon on $G$ moves only $G$. $L$ stays put, and the composed arc reads epsilon.
- Take each epsilon step once. Placing it inside the loop over the other transducer's arcs copies the path once per arc.

### Work the composition by hand

$L$ has two pronunciations. `ah` writes `a` at cost $0.2$. `k` writes epsilon at cost $0.1$, then `t` writes `cat` at cost $0.3$. State 2 is final at cost $0$.

$G$ has a start state and one state per word. From the start, `a` costs $0.5$ and ends at final cost $0.4$. `cat` costs $0.8$ and ends at final cost $0.2$. One more arc, `cat` after `a`, costs $0.3$. This lexicon writes one word and stops, so that arc has no partner and never appears in the composition. A lexicon that returned to its start would walk it, and the graph would contain a cycle.

Two paths survive. Figure 1 draws the costs composition produces.

`ah` matches `a`. The arc costs $0.2 + 0.5 = 0.7$, the final cost is $0.4$, and the path costs $1.1$. It reads `ah` and writes `a`.

`k t` takes the epsilon first. `k:eps` costs $0.1$ and leaves $G$ at the start. `t:cat` matches at $0.3 + 0.8 = 1.1$. The final cost is $0.2$. The path costs $0.1 + 1.1 + 0.2 = 1.4$. It reads `k t` and writes `cat`.

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

State ids in the product can be anything. The checks compare paths, not ids. The match itself is one condition, and the epsilon steps sit outside that loop:

```python
if out_a == in_b and out_a != "<eps>":
    add(next_a, next_b, in_a, out_b, w_a + w_b)
```

A product state is final when both states are final, and the final cost is the sum.

> [!WARNING]
> If both transducers have an epsilon at the same pair of states, either order builds a path, and the two paths share labels and cost. The log semiring then counts that pronunciation twice. Mohri, Pereira, and Riley (2002) treat epsilon as a special case for this reason. k2's `compose` treats epsilon as special unless `treat_epsilons_specially` is false. $G$ here has no input epsilons, so each pronunciation is one path.

> [!TIP]
> Give $G$ no input epsilons on the first test, and compare `compose` with a listing of every path pair. A compose that drops epsilon arcs fails that test by losing `cat`.

## The cheapest path is Viterbi

Figure 1 costs $1.1$ and $1.4$. Tropical search keeps $1.1$, the transcript `a`. That is Viterbi on a graph that already holds the dictionary and the language model.

**n-best** asks for more than the winner. Sorted by cost, the paths are `a` at $1.1$, then `cat` at $1.4$. CMU 11-751's search lecture on November 21, 2022, is this list, and the lattice a beam keeps.

The log-semiring total puts both paths back together:

$$
1.1 \oplus_{\log} 1.4 = -\log(e^{-1.1} + e^{-1.4}) = 0.5456
$$

The $0.5456$ is that expression to four decimal places. It is smaller than either path, because two ways to finish hold more probability than one. The forward algorithm is this total on paths of a fixed length.

> [!WARNING]
> Keeping the largest sum of arc weights treats costs as scores to pile up. Paths $0.2 + 0.2$ and $0.9$ then prefer $0.9$. The tropical path is the minimum sum, $0.4$. Forgetting the final cost fails the other way: a cheap arc into an expensive final state can beat a dearer arc into a cheap one.

> [!KEY]
> `shortest_path` is the minimum sum, final cost included. `total_weight` is the log-semiring sum of every accepting path. `n_best` lists those paths, cheapest first.

## Determinization and minimization shrink the search

A transducer is **deterministic** when each state has at most one arc for a given input label, and no input epsilon (Mohri, Pereira, and Riley, 2002). The search then holds one hypothesis per symbol.

From a start state, two arcs labeled `a` cost $0.4$ and $1.0$. The cheap arc is followed by `x` at $1.0$. The dear arc is followed by `y` at $0.2$. Path costs are $1.4$ for `ax` and $1.2$ for `ay`.

Determinization keeps one `a` arc at the minimum, $0.4$, and stores the leftover $1.0 - 0.4 = 0.6$ on the state the dear arc reached. The following `y` pays that leftover, so it costs $0.2 + 0.6 = 0.8$. The `x` arc stays $1.0$. The path costs do not change:

| Path | Before | After determinization |
|---|---|---|
| `ax` | $0.4 + 1.0 = 1.4$ | $0.4 + 1.0 = 1.4$ |
| `ay` | $1.0 + 0.2 = 1.2$ | $0.4 + 0.8 = 1.2$ |

**Minimization** merges states with the same future. If `p` at cost $0.5$ and `q` at cost $0.4$ each reach a state whose only arc is `z` at $0.2$, those states become one. Path costs stay $0.7$ and $0.6$.

Some transducers do not determinize. Homophones, and a pronunciation that is a prefix of another word, can leave the output waiting with no bound. Mohri, Pereira, and Riley mark the end of a pronunciation with an auxiliary symbol so the word is fixed when the phones end. Kaldi's decoding-graph notes use symbols such as `#1` and `#2` for that, and determinize and minimize after each composition. The University of Edinburgh's ASR course teaches the construction as lecture 9, "ASR with WFSTs."

## A CTC topology is a transducer

The CTC loss is derived in [CTC: many paths, one label](/lessons/028-ctc-many-paths-one-label). The collapse rule is enough to see that the alignment graph is a transducer.

**Collapse** walks integer symbols and keeps a symbol when it differs from the previous one and is not the blank. Runs become one symbol, then blanks drop. With blank `0`, `(1, 1)` collapses to `[1]` and `(1, 0, 1)` collapses to `[1, 1]`. The check calls that chapter's `collapse`. Dropping blanks before merging runs turns `(1, 0, 1)` into `[1]`, which is a different rule.

The **CTC topology** for a target accepts a string exactly when collapse sends it to that target. Every arc costs $0$, and the input equals the output, so it is an acceptor drawn as a transducer. For `[1, 1]` the states are a leading blank, the first `1`, a blank between the copies, the second `1`, and a trailing blank. Each symbol has a self-loop, because a label or a blank can last more than one frame. There is no arc from the first `1` to the second `1`, so `(1, 1)` is rejected.

When neighbors differ, as in `[1, 2]`, both routes are legal: a direct step, and a blank between them. Both strings collapse to `[1, 2]`.

k2 builds this filter with `k2.ctc_topo` and composes it with a token string. The exercise lists every short string on its own.

> [!KEY]
> The CTC blank-and-repeat graph is a transducer. A string is accepted exactly when collapse, merge runs and then drop blanks, returns the target. A repeated label has a blank state between its copies.

> [!RECAP]
> - An arc reads a label, writes a label, and adds a cost. Epsilon is the string `"<eps>"`.
> - Tropical $\oplus$ is $\min$ and $\otimes$ is $+$. Log $\oplus$ is $-\log(e^{-x}+e^{-y})$, and $\otimes$ is still $+$.
> - Composition pairs equal non-epsilon labels and steps over an epsilon on one side.
> - `shortest_path` is the minimum sum, `n_best` lists the paths, and `total_weight` is the log-semiring total.
> - The CTC topology accepts a string exactly when the CTC chapter's `collapse` returns the target.

## Build it step by step

The checks call the functions in this order. The first check only calls `compose`.

1. **`compose(a, b)`.** Pair arcs whose labels match and are not `"<eps>"`, and add the costs. Step an output epsilon of `a`, and an input epsilon of `b`, once each, outside the other loop. A product state is final when both states are. The usual mistake is dropping epsilon arcs, which loses `k t` → `cat`, or emitting the epsilon step once per arc of the other transducer.
2. **`shortest_path(fst)`.** Return `(cost, inputs, outputs)` for the least cost. `heapq` settles each state once when arc costs are non-negative. Add the final cost, and drop `"<eps>"` from the tuples. The usual mistake is keeping the maximum sum, or forgetting the final cost. No path returns `(math.inf, (), ())`.
3. **`n_best(fst, n)`.** List the accepting paths of an acyclic graph, sort by increasing cost, and keep `n`. The usual mistake is the right paths in the wrong order.
4. **`total_weight(fst)`.** Return `-np.logaddexp.reduce(-costs)`. The usual mistake is adding the costs, or calling `logaddexp` on the costs themselves, which treats them as log probabilities. No path returns `math.inf`.
5. **`ctc_topology(target, blank=0)`.** Alternate blank states and label states. Self-loops repeat a symbol. Skip the blank between two labels only when the labels differ. Every arc costs $0$ and carries one symbol on both sides. The usual mistake is an arc between two copies of the same label, which accepts `(1, 1)` for `[1, 1]`.

## Cost and scaling

A naive $L \circ G$ can hold a state for every pair of states, and determinization can grow further before minimization merges them. Kaldi determinizes and minimizes after each stage so the graph stays near the size of the language model. The exercise lists paths because its graphs are tiny and acyclic. A decoder runs a beam, Viterbi with a cap on live states.

## Where this breaks

**Epsilons on both tapes.** The one-sided rule fits this $L \circ G$ because $G$ has no input epsilons. When both sides are epsilon at once, both orders appear and the log semiring double-counts. A production filter keeps one order. These checks leave that filter out.

**Graphs that do not determinize.** Homophones, and a pronunciation that is a prefix of another word, need a disambiguation symbol. Kaldi also marks backoff arcs. Without them the HCLG recipe stops midway.

**Cycles.** `n_best` and `total_weight` list paths, so the graph is acyclic. The CTC topology has zero-cost self-loops, so you test it by which finite strings it accepts.

## The exercise

You build `compose`, `shortest_path`, `n_best`, `total_weight`, and `ctc_topology`. Press **Run** to compose the lexicon with the bigram, print both paths, and plot their tropical costs. The log-semiring total is $0.5456$.

The checks enumerate path pairs for an epsilon-free pair and for the lexicon, including the epsilon on `k`. They compare `shortest_path` with the minimum sum, including a final cost that changes the winner, and `total_weight` with the log-semiring sum of those paths. `ctc_topology` is checked on every short string by the CTC chapter's `collapse`.
