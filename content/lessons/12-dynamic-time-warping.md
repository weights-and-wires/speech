---
title: Dynamic time warping
slug: 12-dynamic-time-warping
part: "Part 3 — Classical recognition"
summary: Align two utterances of different speeds with dynamic programming, and recognize words by their nearest template.
minutes: 60
gpu: false
objectives:
  - Build the local cost matrix between two feature sequences and fill the DTW table with its recurrence.
  - Read the warping path back from the table, and check it against a brute-force search.
  - Constrain the warp with a Sakoe-Chiba band, and explain what it saves and what it risks.
  - Recognize isolated words by their nearest template, with distances normalized by path length.
prereqs:
  - 08-mfccs-and-normalization
  - 11-phonetics-text-and-voice-activity
exercise: 12-dynamic-time-warping
---

# Dynamic time warping

Say "sofa" twice. The second time, you might draw out the "o" and clip the
"a". The two recordings hold the same word, but no frame of one lines up with
the same frame of the other. Compare them frame by frame and you compare an
"o" against an "f", and the distance says they're different words.

**Dynamic time warping** (DTW) fixes the timing before it compares. It finds
the way to stretch and squeeze one sequence onto the other that makes them
match best, and reports how good that best match is. It's like two people
reading the same poem at different speeds: DTW works out which line of one
reader goes with which line of the other.

DTW powered the word recognizers of the 1970s. Before hidden Markov models
took over, isolated-word recognizers stored one or more
recordings of each word, called **templates**, and named an unknown word after
its nearest template under DTW. In this lesson you build that recognizer. The
same dynamic program comes back later in the course: in the forward algorithm
of HMMs, in CTC, and in the alignment search of text-to-speech models.

## What you already know

- A **frame** is a short slice of audio. In this course a frame is 25 ms long,
  and a new one starts every 10 ms, so one second holds about 100 frames.
- MFCCs turn each frame into 13 numbers that describe the shape of its
  spectrum, and removing the utterance mean cancels a fixed channel. For more
  information, see *MFCCs and normalization*.
- The Euclidean distance between two vectors is the length of their
  difference, `np.linalg.norm(u - v)`.
- A vowel shows up as a steady spectral shape and a fricative such as "s" as a
  burst of high-frequency noise, as *Phonetics, text, and voice activity*
  described.
- numpy broadcasting turns an `(n, 1, d)` array minus a `(1, m, d)` array into
  an `(n, m, d)` array.

## Why a straight comparison fails

Two takes of one word differ in total length, and they also differ unevenly
inside. One speaker holds the vowel; another holds the fricative. Stretching
the shorter take linearly to the length of the longer one fixes the total, but
the segments still land in the wrong places.

The synthetic words in this exercise show the size of the effect. Each word is
a sequence of phone-like segments whose durations vary independently, from
60 ms to 300 ms for a vowel. With two templates per word and 10 dB of noise,
comparing after a linear stretch named between 35 and 38 of 40 test words
across four seeded trials. DTW named all 40 in every trial.

A warping path is the right way to think about the fix. Write sequence $a$ as
rows and sequence $b$ as columns of a grid. A **warping path** is a chain of
cells from the top-left corner to the bottom-right corner that pairs frames
up. Each step moves one row down, one column right, or both. A step that moves
only down holds frame $j$ of $b$ while $a$ advances, which is how a long vowel
in $a$ matches a short one in $b$.

## The local cost matrix

Start with every possible pairing. The **local cost** $c(i, j)$ is the
distance between frame $i$ of $a$ and frame $j$ of $b$:

$$
c(i, j) = \lVert a_i - b_j \rVert_2 = \sqrt{\sum_{k=1}^{d} (a_{ik} - b_{jk})^2}
$$

Read it aloud: subtract the two feature vectors, square each of the $d$
differences, add them up, and take the square root. The sequence $a$ has $n$
frames and $b$ has $m$, so the local costs form an $n$ by $m$ matrix. In
numpy, broadcasting builds all of it at once:

```python
cost = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
```

The cost of a path is the sum of the local costs of the cells it visits. DTW
wants the cheapest path.

## The DTW recurrence

A word of 50 frames against a template of 50 frames has more than
$10^{36}$ paths, so you can't try them all. Dynamic programming makes it
cheap. The cheapest path to any cell must arrive from one of its three
neighbors, and it must reach that neighbor by the cheapest path to *it*. So
you fill a table $D$ where $D(i, j)$ is the cost of the cheapest path from
$(0, 0)$ to $(i, j)$:

$$
D(i, j) = c(i, j) + \min\big\{D(i-1, j-1),\ D(i-1, j),\ D(i, j-1)\big\}
$$

Read it aloud: the best total to reach cell $(i, j)$ is this cell's own cost
plus the best total among the cell diagonally above-left, the cell above, and
the cell to the left. Indices here are 0-based, as in the code. The corner
starts the path, so $D(0, 0) = c(0, 0)$. A neighbor outside the table, such as
row $-1$, counts as $+\infty$, so no path can start anywhere except the
corner. The **DTW distance** is $D(n-1, m-1)$.

Fill the table row by row, left to right, so every neighbor is ready before
you need it:

```python
for i in range(n):
    for j in range(m):
        best = 0.0 if i == j == 0 else min(diag, up, left)  # each np.inf when outside
        D[i, j] = cost[i, j] + best
```

> [!KEY]
> The cheapest path to a cell extends the cheapest path to one of its three
> neighbors. That one fact turns an exponential search into an $n$ by $m$
> table.

### Work a 3 by 3 example by hand

Take one feature per frame: $a = [0, 1, 2]$ and $b = [0, 2, 2]$. The local
cost is $|a_i - b_j|$. Rows are frames of $a$ and columns are frames of $b$.

| $c(i, j)$ | $b_0 = 0$ | $b_1 = 2$ | $b_2 = 2$ |
|---|---|---|---|
| $a_0 = 0$ | 0 | 2 | 2 |
| $a_1 = 1$ | 1 | 1 | 1 |
| $a_2 = 2$ | 2 | 0 | 0 |

Now fill $D$ one cell at a time:

- Row 0 can only come from the left: $D(0,0) = 0$, $D(0,1) = 2 + 0 = 2$, and
  $D(0,2) = 2 + 2 = 4$.
- $D(1,0) = 1 + 0 = 1$, from above.
- $D(1,1) = 1 + \min\{0, 2, 1\} = 1$, from the diagonal.
- $D(1,2) = 1 + \min\{2, 4, 1\} = 2$, from the left.
- $D(2,0) = 2 + 1 = 3$, from above.
- $D(2,1) = 0 + \min\{1, 1, 3\} = 1$.
- $D(2,2) = 0 + \min\{1, 2, 1\} = 1$.

The completed table is the following:

| $D(i, j)$ | $j = 0$ | $j = 1$ | $j = 2$ |
|---|---|---|---|
| $i = 0$ | 0 | 2 | 4 |
| $i = 1$ | 1 | 1 | 2 |
| $i = 2$ | 3 | 1 | 1 |

The DTW distance is 1. A second small case shows what warping buys:
$a = [1, 2, 3]$ against $b = [1, 1, 2, 3]$ has distance 0, because the path
pairs $a_0$ with both copies of 1.

## Read the path back

The table holds only totals. To recover the path, start at the last cell and
walk backward. At each cell, step to whichever of the three predecessors has
the smallest $D$, until you reach $(0, 0)$. Then reverse the list.

Paths can tie. In the example, $D(2,2) = 1$ can come from $D(1,1) = 1$ or
$D(2,1) = 1$. Both routes cost 1: the path $(0,0), (1,1), (2,2)$ visits costs
$0 + 1 + 0$, and $(0,0), (1,0), (2,1), (2,2)$ visits $0 + 1 + 0 + 0$. The
exercise breaks ties in the order diagonal, up, left, so it returns the first
one. Either answer is correct; what must hold is that the path's cells add up
to $D(n-1, m-1)$.

The following demo draws two sequences, the cost matrix between them, and the
path.

```demo
{"id": "dtw-path", "caption": "Slow down the second segment and watch the path bend to absorb it while the normalized distance barely moves, then narrow the Sakoe-Chiba band and watch the share of cells the table must fill fall."}
```

## Constrain the warp with a band

Unconstrained DTW can do silly things. It can map the whole of one word onto
the first frame of another, when that one frame happens to be close to
everything. Real timing differences are moderate, so Sakoe and Chiba (1978)
restricted the path to a **band** around the diagonal. With equal lengths, the
band allows cell $(i, j)$ only when $|i - j| \le w$, for a half-width $w$ in
frames.

Utterances rarely have equal lengths, so the exercise centers the band on the
straight line from the first corner to the last:

$$
\left| j - i \, \frac{m - 1}{n - 1} \right| \le w
$$

Read it aloud: at row $i$, the straight line sits at column
$i (m-1)/(n-1)$, and the cell is allowed when its column is within $w$ of
that. With $n = m$ the slope is 1 and this is the classic band. Forbidden
cells hold $+\infty$ in $D$, so the recurrence steers around them without any
special case:

```python
allowed = np.abs(np.arange(m)[None, :] - np.arange(n)[:, None] * slope) <= w
```

The band does two jobs. It forbids absurd warps, and it cuts the work from
$nm$ cells to about $n(2w + 1)$. The Itakura parallelogram (1975) is the other
classic constraint; it limits the path's slope instead, so the allowed region
narrows toward both corners.

> [!WARNING]
> A band too narrow for the real timing difference forces a bad path, and the
> distance comes out large for two takes of the same word. With $w = 0$ and
> equal lengths, the only path left is the diagonal, and DTW becomes the plain
> frame-by-frame distance it was meant to replace.

## Recognize words by nearest template

A template recognizer stores a few MFCC sequences per word. To recognize an
unknown utterance, compute its DTW distance to every template and answer with
the label of the nearest one.

One adjustment matters. The DTW distance is a sum over the path, and a longer
path has more terms, so a long template looks far from everything. Divide by
the number of cells on the path, $L$, to get the **average cost per step**:

$$
\bar{D} = \frac{D(n-1, m-1)}{L}
$$

Read it aloud: the total cost of the best path, shared out over its steps.
In the exercise's own test set, raw distances name 36 of the 40 words, and
per-step averages name all 40.

> [!KEY]
> A template recognizer is nearest-neighbor classification with DTW as the
> distance. Normalize the distance by path length before you compare
> templates of different lengths.

> [!RECAP]
> - `local_cost` is `np.sqrt(np.sum((a[:, None] - b[None]) ** 2, axis=-1))`,
>   shape `(n, m)`.
> - $D(i, j) = c(i, j) + \min$ of the diagonal, upper, and left neighbors, with
>   $+\infty$ outside the table and $D(0,0) = c(0,0)$.
> - Backtracking walks from the last cell to $(0, 0)$ through the cheapest
>   predecessor, then reverses.
> - The band allows $|j - i(m-1)/(n-1)| \le w$; forbidden cells hold
>   `np.inf`.
> - Compare templates by $D(n-1, m-1) / L$, the average cost per step.

## Build it step by step

The checks test the functions in this order. The file provides `in_band`, the
synthetic words (`synth_word` and `WORDS`), and the `mfcc` front end.

1. **`local_cost(a, b)`.** Takes `(n, d)` and `(m, d)` arrays and returns the
   `(n, m)` Euclidean distances. Broadcast, square, sum over the last axis,
   and take the square root. Common mistake: summing over the wrong axis,
   which gives an `(n, d)` or `(m, d)` result.
2. **`accumulated_cost(cost, band=None)`.** Returns `D`, shape `(n, m)`.
   Start from `np.full((n, m), np.inf)`, get the mask from `in_band`, and fill
   the allowed cells row by row with the recurrence. Common mistake:
   initializing the first row and column to zero, which lets a path start
   anywhere along an edge.
3. **`backtrack(D)`.** Returns a list of `(i, j)` tuples from `(0, 0)` to
   `(n - 1, m - 1)`. Collect the predecessors that exist, take the smallest
   with `min`, and append. Reverse at the end. Common mistake: stepping to a
   neighbor at index `-1`, which numpy silently reads from the far end of the
   table.
4. **`dtw(a, b, band=None)`.** Chains the three and returns
   `(float(D[-1, -1]), path)`, or `(float("inf"), [])` when no path fits the
   band. Common mistake: returning the whole table instead of its last cell.
5. **`classify(query, templates, labels, band=None)`.** Loops over the
   templates, scores each with `distance / len(path)`, and returns the label
   of the smallest score. Common mistake: comparing raw distances.

## Cost and scaling

The following table shows how the cost grows with the query length $n$, the
template length $m$, the band half-width $w$, and the number of templates $K$.

| Operation | Time | Memory |
|---|---|---|
| Local cost matrix | $O(nmd)$ | $O(nm)$ |
| DTW table, no band | $O(nm)$ | $O(nm)$ |
| DTW table, band $w$ | $O(nw)$ | $O(nm)$ as written, $O(nw)$ if stored by band |
| Distance only, no path | $O(nm)$ | $O(m)$, two rows at a time |
| Recognizing one word | $K$ times one DTW | one table at a time |

A 0.5-second word is about 50 frames, so one DTW fills 2,500 cells, and 16
templates cost 40,000 cells per query. That's fine for a vocabulary of ten
digits. It isn't fine for 100,000 words, which is one reason recognizers moved
to models that share parts between words.

## Where this breaks

**Speaker and channel changes.** DTW compares raw features. A template from
one speaker matches another speaker's voice poorly, because formants move with
vocal tract length. People stored several templates per word from several
speakers, and normalized features with the mean removal from *MFCCs and
normalization*.

**Silence and noise at the edges.** Both corners of the path are fixed, so
leading silence in the query must match something in the template. Trim the
edges with a voice activity detector first, or allow the path to start and end
anywhere along an edge, a variant called subsequence DTW.

**Pathological warps.** Without a band, one frame can absorb a whole segment.
Bands, slope limits, and weighted step patterns help. Sakoe and Chiba's
symmetric pattern weights the diagonal step by 2 and the others by 1, so every
path has total weight $n + m$, and dividing by $n + m$ normalizes exactly.

**It isn't a true distance.** DTW is symmetric with these steps, but it breaks
the triangle inequality, so indexing tricks that assume a metric can miss the
nearest template.

**It doesn't scale to large vocabularies or continuous speech.** Each word
needs its own templates, and nothing is shared between words. Hidden Markov
models, the next lesson, replace templates with statistical models of each
sound.

## The exercise

You build `local_cost`, `accumulated_cost` with an optional Sakoe-Chiba band,
`backtrack`, `dtw`, and the template recognizer `classify`. Press **Run** to
hear two takes of the synthetic word "osa", see the warping path drawn on
their cost matrix, and see the confusion matrix of a recognizer that uses two
templates per word on 40 words at 10 dB SNR.

The checks compare `local_cost` with `scipy.spatial.distance.cdist`, and
`accumulated_cost` with the 3 by 3 table worked earlier. They compare the last
cell of the table with a brute-force search over every warping path, with and
without a band, and confirm that forbidden cells hold `np.inf`. They confirm
that `backtrack` returns a valid path whose cells add up to the distance, and
that `dtw` absorbs a repeated frame, is symmetric, and reduces to the
frame-by-frame distance with a band of 0. The last check requires `classify`
to name at least 38 of 40 noisy synthetic words.
