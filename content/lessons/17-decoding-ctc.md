---
title: Decoding CTC
slug: 17-decoding-ctc
part: "Part 4 — Neural speech recognition"
summary: Turn CTC emissions into text with greedy decoding, prefix beam search, a character language model, and a hotword bonus.
minutes: 60
gpu: false
objectives:
  - Decode CTC emissions greedily, and explain why the best path isn't always the most likely transcript.
  - Build CTC prefix beam search with separate blank and non-blank probabilities, and check it against brute-force enumeration.
  - Fuse a character bigram language model into the beam with a weight and an insertion bonus.
  - Boost rare words with a hotword bonus that disappears when a partial match fails.
prereqs:
  - 14-language-models-and-word-error-rate
  - 16-connectionist-temporal-classification
exercise: 17-decoding-ctc
---

# Decoding CTC

A CTC model hands you a table, not a sentence. For every frame of audio it
gives a probability for every symbol, plus the blank. Training only needed the
total probability of the right transcript. At test time you don't know the
transcript, so you have to search for the one the table likes best.

That search is harder than it looks. Thousands of frame-level paths collapse
to the same text, and the text with the best single path isn't always the most
likely text. The acoustic model also knows nothing about
spelling or grammar, so "the kat" can outscore "the cat" when the audio is
unclear.

In this lesson you build greedy decoding, then prefix beam search, which adds
up paths the way the CTC loss does. Then you add a character language model and
a hotword bonus, two tools that let a decoder fix what the acoustics got wrong.

## What you already know

- A CTC model emits $y^t_k$, the probability of symbol $k$ at frame $t$, with the blank $\varnothing$ at index 0.
- The collapse function $\mathcal{B}$ merges repeated symbols, then deletes blanks, so $\mathcal{B}(\texttt{aa\_ab\_\_}) = \texttt{aab}$.
- The probability of a labeling $\mathbf{l}$ is the sum over every path that collapses to it, from the lesson "Connectionist temporal classification."
- A bigram language model gives $P(c \mid \text{previous symbol})$, from the lesson "Language models and word error rate."
- `np.logaddexp(a, b)` computes $\log(e^a + e^b)$ without overflow, and handles $-\infty$.

## Greedy decoding

The simplest decoder reads the table one frame at a time. In each frame it
picks the most likely symbol. Then it collapses the resulting path. This is
called **greedy** or **best-path** decoding:

$$
\hat{\pi}_t = \operatorname*{arg\,max}_k \; y^t_k, \qquad \hat{\mathbf{l}} = \mathcal{B}(\hat{\pi}_1, \dots, \hat{\pi}_T)
$$

Read it aloud: for each frame $t$, the chosen symbol $\hat\pi_t$ is the index
$k$ with the largest probability. The transcript $\hat{\mathbf{l}}$ is the
collapse of that path. In numpy it's two lines:

```python
path = np.argmax(log_probs, axis=1)          # (T,) symbol indices
labels = [s for i, s in enumerate(path) if s != 0 and (i == 0 or s != path[i - 1])]
```

Take the frame winners `a a _ a b b _ b`. Merging runs gives `a _ a b _ b`,
and deleting blanks gives `aabb`. Each blank keeps two copies of a letter
apart.

Greedy decoding works surprisingly well on a trained model. CTC models are
**peaky**: they predict blank with probability near 1 in most frames, and each
label appears as a spike one or two frames wide. When every frame is nearly
one-hot, the best path carries almost all the probability.

### Where the best path goes wrong

Now take two frames, each with $y_\varnothing = 0.6$ and $y_a = 0.4$. Greedy
picks blank twice and returns the empty string. The path `__` has probability
$0.6 \times 0.6 = 0.36$.

But three paths collapse to `a`: `a_`, `_a`, and `aa`. Their probabilities are
$0.4 \times 0.6 = 0.24$, $0.6 \times 0.4 = 0.24$, and $0.4 \times 0.4 = 0.16$.
Together they give $p(a) = 0.64$. The transcript `a` is almost twice as likely
as the empty one, but no single path to it beats `__`.

> [!KEY]
> The best path isn't the best transcript. A transcript's probability is spread
> over many paths, and greedy decoding looks at only one of them.

## Prefix beam search

The fix is to search over transcripts, not paths. **Prefix beam search** walks
the frames left to right and keeps a short list, the **beam**, of the most
likely **prefixes**: collapsed partial transcripts such as `th` or `the c`.
Each prefix carries the total probability of every path so far that collapses
to it. After each frame, it keeps only the best $K$ prefixes, where $K$ is the
**beam width**.

One number per prefix isn't enough. Suppose the prefix is `a` and the next
frame emits `a`. If the last frame was also `a`, the two merge and the prefix
stays `a`. If a blank came in between, the new `a` starts a second letter and
the prefix becomes `aa`. So each prefix $\ell$ keeps two probabilities:

- $p_b(\ell)$, the probability of the paths that collapse to $\ell$ and end in a blank.
- $p_{nb}(\ell)$, the probability of the paths that collapse to $\ell$ and end in its last label.

The search starts with the empty prefix: $p_b(\varnothing) = 1$ and
$p_{nb}(\varnothing) = 0$. At frame $t$, every prefix $\ell$ in the beam sends
probability forward in three ways:

$$
\begin{aligned}
p^{t}_b(\ell) &\mathrel{+}= \big(p^{t-1}_b(\ell) + p^{t-1}_{nb}(\ell)\big)\, y^t_\varnothing \\
p^{t}_{nb}(\ell + c) &\mathrel{+}= \big(p^{t-1}_b(\ell) + p^{t-1}_{nb}(\ell)\big)\, y^t_c && c \ne \text{last}(\ell) \\
p^{t}_{nb}(\ell + c) &\mathrel{+}= p^{t-1}_b(\ell)\, y^t_c,\quad p^{t}_{nb}(\ell) \mathrel{+}= p^{t-1}_{nb}(\ell)\, y^t_c && c = \text{last}(\ell)
\end{aligned}
$$

Read it aloud. First line: a blank at frame $t$ leaves the prefix unchanged,
and every path to $\ell$ can take it, so the whole old probability moves into
the ends-in-blank bucket. Second line: a new label $c$, different from the
prefix's last label, extends the prefix to $\ell + c$ from either bucket.
Third line: repeating the last label splits in two. Only paths that ended in a
blank start a new copy, $\ell + c$. Paths that ended in the label itself merge
into the same prefix $\ell$. The score of a prefix is
$p_b(\ell) + p_{nb}(\ell)$.

Real decoders do this in log space, where every $+=$ becomes a `logaddexp` and
every product becomes a sum. For one extension it looks like this:

```python
p_all = np.logaddexp(p_b, p_nb)                  # log(p_b + p_nb)
b, nb = nxt[prefix + (c,)]                       # defaults to (-inf, -inf)
nxt[prefix + (c,)] = (b, np.logaddexp(nb, p_all + frame[c]))
```

### Work the two-frame example by hand

Use the two frames from the previous section, $y_\varnothing = 0.6$ and
$y_a = 0.4$, with a beam wide enough to keep everything.

After frame 1, starting from $p_b(\varnothing) = 1$:

- A blank keeps the empty prefix: $p_b(\varnothing) = 1 \times 0.6 = 0.6$.
- The label `a` makes a new prefix: $p_{nb}(a) = 1 \times 0.4 = 0.4$.

After frame 2:

- The empty prefix takes another blank: $p_b(\varnothing) = 0.6 \times 0.6 = 0.36$.
- The empty prefix emits `a`: $p_{nb}(a) \mathrel{+}= 0.6 \times 0.4 = 0.24$.
- The prefix `a` takes a blank: $p_b(a) = (0 + 0.4) \times 0.6 = 0.24$.
- The prefix `a` repeats `a` with no blank between, which merges: $p_{nb}(a) \mathrel{+}= 0.4 \times 0.4 = 0.16$.
- The prefix `a` repeats `a` after a blank, which would make `aa`: $p_{nb}(aa) = 0 \times 0.4 = 0$.

The beam ends with `a` at $0.24 + 0.24 + 0.16 = 0.64$ and the empty prefix at
$0.36$. That's exactly the brute-force answer. The prefix `aa` has probability
0, because two frames can't hold `a_a`, so the decoder drops it.

With an unlimited beam, prefix beam search gives the exact CTC probability of
every labeling. With a beam of $K$, it drops prefixes whose paths might have
come back later. So every score it returns is a lower bound on the true
probability. The following demo shows the beam at work on a short table.

```demo
{"id": "beam-search", "caption": "Shrink the beam to one and watch the right transcript fall out of it, then widen it until the best prefix stops changing."}
```

> [!KEY]
> Prefix beam search adds up paths the way the CTC loss does, but keeps only
> the best few prefixes. Two probabilities per prefix, ending in blank or not,
> are what make repeated letters come out right.

> [!WARNING]
> A repeated label is the classic bug. Extending `a` by `a` from $p_{nb}$ must
> stay `a`, and only $p_b$ may make `aa`. Get this wrong and "hello" decodes as
> "helo" or "hellllo".

## Shallow fusion with a character language model

When the acoustic model can't tell a `c` from a `k`, it has no way to know
that "cat" is a word and "kat" usually isn't. A **language model** (LM)
does. **Shallow fusion** adds the LM's opinion
to the beam's score, the same noisy-channel idea as the lesson "Language
models and word error rate":

$$
\operatorname{score}(\ell) = \log\big(p_b(\ell) + p_{nb}(\ell)\big) + \alpha \log P_{\text{LM}}(\ell) + \beta\,|\ell|
$$

Read it aloud: the score of prefix $\ell$ is its acoustic log probability,
plus $\alpha$ times the LM's log probability of the same text, plus a bonus
$\beta$ for every symbol it holds. The **LM weight** $\alpha$ says how much to
trust the LM. The **insertion bonus** $\beta$ pays for length. Every extra
symbol multiplies $P_{\text{LM}}$ by a number below 1, so without $\beta$ the
fused score prefers short transcripts. The beam still ranks and prunes with
this score, but $p_b$ and $p_{nb}$ stay purely acoustic:

```python
score = np.logaddexp(p_b, p_nb) + alpha * lm_log_prob(prefix) + beta * len(prefix)
```

The LM in this lesson is a character bigram with **add-$k$ smoothing**. Count
how often each symbol $c$ follows each symbol $p$ in a text corpus, with a
start symbol before the first character. Then

$$
P(c \mid p) = \frac{n(p, c) + k}{\sum_{c'} n(p, c') + k\,(C - 1)}
$$

Read it aloud: the probability of $c$ after $p$ is the count of that pair plus
$k$, divided by the count of all pairs that start with $p$ plus $k$ for each
of the $C - 1$ labels. The $k$ gives unseen pairs a small probability instead
of zero. The blank never appears in text, so the LM never predicts it, and the
start context reuses index 0. In numpy:

```python
counts[:, 1:] += k
lm = np.log(counts / counts.sum(axis=1, keepdims=True))   # lm[p, c], column 0 is -inf
```

### Work a fusion by hand

Train the bigram on the two texts `ab` and `abb` with $k = 1$. Counting pairs,
with `^` for the start: `^a` twice, `ab` twice, and `bb` once. Each row adds
$k = 1$ to both labels:

- $P(a \mid \text{start}) = (2 + 1)/(2 + 2) = 0.75$, and $P(b \mid \text{start}) = 0.25$.
- $P(b \mid a) = (2 + 1)/(2 + 2) = 0.75$, and $P(a \mid a) = 0.25$.
- $P(b \mid b) = (1 + 1)/(1 + 2) = 2/3$, and $P(a \mid b) = 1/3$.

Now decode three frames with probabilities (blank, `a`, `b`) of
$(0.2, 0.35, 0.45)$, $(0.8, 0.1, 0.1)$, and $(0.2, 0.45, 0.35)$. Greedy reads
`b _ a` and returns `ba`. The following table lists the three strongest
prefixes, their exact acoustic log probabilities from the beam, and their
fused scores with $\alpha = 1$:

| Prefix | $\log p_{\text{CTC}}$ | $\log P_{\text{LM}}$ | Score, $\beta = 0$ | Score, $\beta = 1$ |
|---|---|---|---|---|
| `ba` | $-1.5119$ | $\log 0.25 + \log \tfrac13 = -2.4849$ | $-3.9968$ | $-1.9968$ |
| `ab` | $-1.9914$ | $2 \log 0.75 = -0.5754$ | $-2.5668$ | $-0.5668$ |
| `a` | $-1.8094$ | $\log 0.75 = -0.2877$ | $-2.0971$ | $-1.0971$ |

The acoustics alone pick `ba`, with $p = 0.2205$. The LM makes `ab` far more
likely than `ba`, but with $\beta = 0$ the winner is the one-letter `a`: it pays
the LM for only one symbol. With $\beta = 1$, each prefix earns one point per
symbol, and `ab` wins. That's the job of the insertion bonus.

Real systems tune $\alpha$ and $\beta$ by grid search on held-out data. The
`pyctcdecode` library, which fuses a word-level KenLM model into CTC beam
search, defaults to $\alpha = 0.5$, $\beta = 1.5$, and a beam of 100.

> [!KEY]
> Shallow fusion adds $\alpha \log P_{\text{LM}}$ and $\beta$ per symbol to each
> prefix's score. The LM fixes spelling the acoustics can't hear, and $\beta$
> stops it from favoring short output.

## Hotwords

Some words matter more than their LM probability suggests: a contact's name,
a product, a drug. The LM has rarely or never seen them, so fusion pushes
them out of the beam. A **hotword bonus** gives the decoder a reason to keep
them.

The rule is a reward for progress. A prefix earns a bonus $\lambda$ for every
symbol of a hotword it has completed, and for every symbol of a hotword it is
partway through at its end. If the partial match breaks, the bonus for it
disappears:

$$
\operatorname{hot}(\ell) = \lambda \,\big(\text{complete}(\ell) + \text{partial}(\ell)\big)
$$

Read it aloud: $\text{complete}(\ell)$ is the total length of every hotword
occurrence inside $\ell$, and $\text{partial}(\ell)$ is the longest $k$ below
a hotword's length such that $\ell$ ends with that hotword's first $k$
symbols. For the hotword `zoe` with $\lambda = 2$:

- `tell z` ends with `z`, a partial match of 1, so it earns $2 \times 1 = 2$.
- `tell zo` has a partial match of 2 and earns 4.
- `tell zoe` completes the word and earns $2 \times 3 = 6$.
- `tell zoa` breaks the match and earns 0, so the bonus it had is taken back.

The bonus depends on the prefix alone, so it adds to the fused score as one
more term. Libraries track it with a **trie**, a tree of hotword prefixes, so
each extension updates it in constant time. `pyctcdecode` gives hotwords a
default weight of 10.

> [!TIP]
> Test your decoder with the LM and hotword terms set to zero first. It must
> then reproduce plain beam search exactly. Add one term at a time.

> [!RECAP]
> - Greedy decoding is `argmax` per frame, then merge repeats, then drop blanks.
> - Prefix beam search keeps $p_b$ and $p_{nb}$ per prefix, updates them with
>   `np.logaddexp`, and keeps the top $K$ by score after every frame.
> - A repeated label extends the prefix only from $p_b$; from $p_{nb}$ it merges.
> - The fused score is $\log(p_b + p_{nb}) + \alpha \log P_{\text{LM}} + \beta|\ell|$,
>   plus a hotword bonus that vanishes when a partial match fails.

## Build it step by step

The checks test the functions in this order.

1. **`greedy_decode(log_probs, blank)`.** Takes a `(T, C)` array and returns a
   list of ints. Use `np.argmax(..., axis=1)`, then walk the path once. Common
   mistake: dropping blanks before merging, which turns `a _ a` into `a`.
2. **`prefix_beam_search(log_probs, beam_width, blank, prefix_score)`.**
   Returns a list of `(prefix, score)` pairs, best first. Start from
   `{(): (0.0, -inf)}`. For each frame, build a `defaultdict` of new
   `(p_b, p_nb)` pairs from the three rules, rank by
   `np.logaddexp(p_b, p_nb) + prefix_score(prefix)`, and keep the top
   `beam_width`. Drop prefixes whose score is `-inf`. Common mistake: letting
   the third rule extend from `p_nb`.
3. **`train_char_bigram(texts, vocab, smoothing)`.** Returns a `(C, C)` array
   of log probabilities. Count `counts[previous, current]` starting from
   `previous = 0`, add the smoothing to columns 1 and up, normalize each row,
   and take `np.log`. Common mistake: adding the smoothing to the blank column,
   which leaks probability to a symbol the LM can never emit.
4. **`lm_score(prefix, lm, alpha, beta)`.** Returns a float. Sum
   `lm[a, b]` over consecutive pairs of `(0,) + prefix`, multiply by `alpha`,
   and add `beta * len(prefix)`. Common mistake: forgetting the start context,
   so the first letter is free.
5. **`hotword_score(prefix, hotwords, bonus)`.** Returns a float. Count
   complete occurrences with a sliding window and find the longest partial
   match with `prefix[-k:] == word[:k]`. Common mistake: letting `k` reach the
   full word length, which counts a completed word twice.

## Cost and scaling

The table uses $T$ frames, $C$ symbols, a beam of $K$, and a prefix length of
$L$.

| Decoder | Work | Memory |
|---|---|---|
| Brute force over paths | $C^T$ paths, $3^5 = 243$ for the check, $28^{100}$ for two seconds of characters | $O(T)$ |
| Greedy | $O(TC)$ | $O(T)$ |
| Prefix beam search | $O(TKC)$ updates, plus sorting $KC$ candidates per frame | $O(KC)$ prefixes |
| With a bigram LM, scored from scratch | $O(TKCL)$ | a $C \times C$ table |

A wav2vec 2.0 model emits 50 frames per second over 32 symbols, so a beam of
100 extends 3,200 candidates per frame. Real decoders extend only symbols above
a floor, such as `pyctcdecode`'s default minimum log probability of $-5$. They
skip frames where blank is almost certain, and they update the LM state
incrementally, so the cost doesn't grow with $L$.

## Where this breaks

**The beam loses the right answer early.** A prefix pruned at frame 10 can
never come back. Widen the beam until the output stops changing on held-out
data; the cost grows linearly with $K$.

**The weights are tuned for one domain.** An $\alpha$ tuned on audiobooks
over-corrects medical dictation, where the right words look unlikely to a
general LM. Tune $\alpha$ and $\beta$ on data from the domain you serve.

**A character bigram sees one letter of context.** It knows that `c` often
precedes `a`, but not that "the cat" is more likely than "the cap". Production
systems use word n-grams or neural LMs, which need a word-boundary symbol, and
score a word only when the boundary is emitted. KenLM, a common n-gram
library, returns $\log_{10}$ scores, so convert them before you add them to
natural-log acoustic scores.

**Hotwords can fire where they don't belong.** A large bonus pulls
similar-sounding words toward the hotword. Keep the list short and the bonus
modest.

**The emissions are wrong, not the search.** No decoder recovers a word the
model gave no probability. When beam search and greedy agree on a wrong
answer, look at the model.

## The exercise

You build `greedy_decode` and `prefix_beam_search`, then the pieces that plug
into the beam's `prefix_score` hook: `train_char_bigram`, `lm_score`, and
`hotword_score`. Press **Run** to decode a synthetic sentence four ways. You
see the posteriorgram and the blank's probability per frame, and you hear the
greedy path as one beep per non-blank frame.

The checks confirm greedy decoding on a hand-built path and on the two-frame
counterexample. They compare every score of a wide beam against brute-force
enumeration of all $3^5$ paths, and confirm that a narrow beam returns lower
bounds, best first. They check the bigram against the hand-counted
probabilities, `lm_score` and `hotword_score` against the lesson's numbers,
and the fused beam against brute force. The last check decodes a sentence in
which the LM must turn "kat" into "cat" and the hotword must turn "soe" into
"zoe".
