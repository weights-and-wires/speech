---
title: Dynamic time warping
slug: 020-dynamic-time-warping
part: "Part 3 — Classical recognition"
summary: Align two utterances spoken at different speeds with dynamic programming, and recognize words by their nearest template.
minutes: 40
gpu: false
objectives:
  - Build the local cost matrix between two feature sequences and fill the DTW table with its recurrence.
  - Read the warping path back from the table, and check it against a brute-force search.
  - Constrain the warp with a Sakoe-Chiba band, and explain what it saves and what it risks.
  - Recognize isolated words by their nearest template, with distances normalized by path length.
prereqs:
  - 013-mfccs
  - 014-deltas-and-normalization
  - 018-phonemes-and-text-normalization
exercise: 020-dynamic-time-warping
---

# Dynamic time warping

Say "sofa" twice. The second time, you might draw out the "o" and clip the "a". The two recordings hold the same word, but no frame of one lines up with the same frame of the other. Compare them frame by frame and you compare an "o" against an "f", so the distance says they are different words.

**Dynamic time warping** (DTW) fixes the timing before it compares. It finds the way to stretch and squeeze one sequence onto the other that makes them match best, and reports how good that best match is. Think of two people reading the same poem at different speeds: DTW works out which line of one reader goes with which line of the other.

DTW powered the word recognizers of the 1970s. They stored recordings of each word, called **templates**, and named an unknown word after its nearest template under DTW. You build that recognizer here. The same dynamic program returns later in the course, in HMMs, in CTC, and in text-to-speech alignment.

## What you already know

- A **frame** is a 25 ms slice of audio, and another starts every 10 ms, so one second holds about 100 frames.
- MFCCs turn each frame into 13 numbers that describe its spectrum, and removing the utterance mean cancels a fixed channel. See [MFCCs](/lessons/013-mfccs) and [Deltas and normalization](/lessons/014-deltas-and-normalization).
- The Euclidean distance between two vectors is `np.linalg.norm(u - v)`.
- A vowel is a steady spectral shape and a fricative such as "s" is a burst of high-frequency noise. See [Phonemes and text normalization](/lessons/018-phonemes-and-text-normalization).
- numpy broadcasting turns an `(n, 1, d)` array minus a `(1, m, d)` array into an `(n, m, d)` array.

## Why a straight comparison fails

Two takes of one word differ in total length, and they also differ unevenly inside: one speaker holds the vowel while another holds the fricative. The obvious fix is to stretch the shorter take linearly to the length of the longer one. That corrects the total, but the segments still land in the wrong places, because the speaker did not stretch every segment by the same factor.

The exercise's synthetic words show the size of the effect. Each segment's duration varies independently, from 60 ms to 300 ms for a vowel. With two templates per word and 10 dB of noise, a linear stretch named between 35 and 38 of 40 test words across four seeded trials. DTW named all 40 in every trial.

The fix is to let the pairing of frames bend. Put sequence $a$ down the rows of a grid and sequence $b$ across its columns. A **warping path** is a chain of cells from the top-left corner to the bottom-right corner, and each cell $(i, j)$ pairs frame $i$ of $a$ with frame $j$ of $b$. Each step of the chain moves one row down, one column right, or both. A step that moves only down holds frame $j$ of $b$ while $a$ advances, which is how a long vowel in $a$ matches a short one in $b$.

```figure
<svg viewBox="0 0 600 255" role="img" aria-label="Two sequences of different length with lines pairing each frame of the short one to frames of the long one">
<line x1="100.0" y1="90.0" x2="100.0" y2="150.0" class="stroke-b"/>
<line x1="100.0" y1="90.0" x2="233.3" y2="150.0" class="stroke-b"/>
<line x1="300.0" y1="90.0" x2="366.7" y2="150.0" class="stroke-b"/>
<line x1="500.0" y1="90.0" x2="500.0" y2="150.0" class="stroke-b"/>
<circle cx="100.0" cy="70.0" r="20" class="fill-soft-a stroke-a"/>
<text x="100.0" y="75.0" class="fig-bold" text-anchor="middle">1</text>
<text x="100.0" y="40.0" class="fig-small" text-anchor="middle">a[0]</text>
<circle cx="300.0" cy="70.0" r="20" class="fill-soft-a stroke-a"/>
<text x="300.0" y="75.0" class="fig-bold" text-anchor="middle">2</text>
<text x="300.0" y="40.0" class="fig-small" text-anchor="middle">a[1]</text>
<circle cx="500.0" cy="70.0" r="20" class="fill-soft-a stroke-a"/>
<text x="500.0" y="75.0" class="fig-bold" text-anchor="middle">3</text>
<text x="500.0" y="40.0" class="fig-small" text-anchor="middle">a[2]</text>
<circle cx="100.0" cy="170.0" r="20" class="fill-soft-c stroke-c"/>
<text x="100.0" y="175.0" class="fig-bold" text-anchor="middle">1</text>
<text x="100.0" y="212.0" class="fig-small" text-anchor="middle">b[0]</text>
<circle cx="233.3" cy="170.0" r="20" class="fill-soft-c stroke-c"/>
<text x="233.3" y="175.0" class="fig-bold" text-anchor="middle">1</text>
<text x="233.3" y="212.0" class="fig-small" text-anchor="middle">b[1]</text>
<circle cx="366.7" cy="170.0" r="20" class="fill-soft-c stroke-c"/>
<text x="366.7" y="175.0" class="fig-bold" text-anchor="middle">2</text>
<text x="366.7" y="212.0" class="fig-small" text-anchor="middle">b[2]</text>
<circle cx="500.0" cy="170.0" r="20" class="fill-soft-c stroke-c"/>
<text x="500.0" y="175.0" class="fig-bold" text-anchor="middle">3</text>
<text x="500.0" y="212.0" class="fig-small" text-anchor="middle">b[3]</text>
<text x="300.0" y="238.0" class="fig-small" text-anchor="middle">Each line is one cell of the warping path; a[0] appears in two of them.</text>
</svg>
Figure 1. The sequence a = [1, 2, 3] is paired with b = [1, 1, 2, 3]. Notice that a[0] is paired with both copies of 1, which is what lets the two sequences match at a cost of 0 although they differ in length.
```

## Compare every pair of frames

Start with every possible pairing. The **local cost** $c(i, j)$ is the distance between frame $i$ of $a$ and frame $j$ of $b$:

$$
c(i, j) = \lVert a_i - b_j \rVert_2 = \sqrt{\sum_{k=1}^{d} (a_{ik} - b_{jk})^2}
$$

Read it aloud: subtract the two feature vectors, square each of the $d$ differences, add them up, and take the square root. Sequence $a$ has $n$ frames and $b$ has $m$, so the costs form an $n$ by $m$ matrix, which broadcasting builds at once:

```python
cost = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))
```

The cost of a path is the sum of the local costs of the cells it visits, and DTW wants the cheapest path.

## Fill a table instead of trying every path

A word of 50 frames against a template of 50 frames has more than $10^{36}$ paths, so you cannot try them all. Dynamic programming makes the search cheap with one observation: the cheapest path to any cell must arrive from one of its three neighbors, and it must reach that neighbor by the cheapest path to *it*. So you fill a table $D$ where $D(i, j)$ is the cost of the cheapest path from $(0, 0)$ to $(i, j)$:

$$
D(i, j) = c(i, j) + \min\big\{D(i-1, j-1),\ D(i-1, j),\ D(i, j-1)\big\}
$$

Read it aloud: the best total to reach cell $(i, j)$ is this cell's own cost plus the best total among the cell diagonally above-left, the cell above, and the cell to the left. Indices are 0-based, as in the code. The corner starts the path, so $D(0, 0) = c(0, 0)$, and a neighbor outside the table counts as $+\infty$. The **DTW distance** is $D(n-1, m-1)$.

Fill the table row by row, left to right, so every neighbor is ready before you need it:

```python
for i in range(n):
    for j in range(m):
        best = 0.0 if i == j == 0 else min(diag, up, left)  # each np.inf when outside
        D[i, j] = cost[i, j] + best
```

> [!KEY]
> The cheapest path to a cell extends the cheapest path to one of its three neighbors. That one fact turns an exponential search into an $n$ by $m$ table.

### Work a 3 by 3 example by hand

Take one feature per frame: $a = [0, 1, 2]$ and $b = [0, 2, 2]$, so the local cost is $|a_i - b_j|$. Fill $D$ one cell at a time:

- Row 0 can only come from the left: $D(0,0) = 0$, $D(0,1) = 2 + 0 = 2$, and $D(0,2) = 2 + 2 = 4$.
- $D(1,0) = 1 + 0 = 1$, from above. $D(1,1) = 1 + \min\{0, 2, 1\} = 1$. $D(1,2) = 1 + \min\{2, 4, 1\} = 2$.
- $D(2,0) = 2 + 1 = 3$, from above. $D(2,1) = 0 + \min\{1, 1, 3\} = 1$. $D(2,2) = 0 + \min\{1, 2, 1\} = 1$.

The DTW distance is 1. Figure 2 shows the finished table with the cheapest path shaded.

```figure
<svg viewBox="0 0 620 330" role="img" aria-label="The three by three DTW table with the local cost in each cell and the cheapest path shaded">
<defs><marker id="c020-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="270.0" y="30.0" class="fig-small" text-anchor="middle">b[j]</text>
<text x="190.0" y="52.0" class="fig-small" text-anchor="middle">0  (j = 0)</text>
<text x="270.0" y="52.0" class="fig-small" text-anchor="middle">2  (j = 1)</text>
<text x="350.0" y="52.0" class="fig-small" text-anchor="middle">2  (j = 2)</text>
<text x="140.0" y="105.0" class="fig-small" text-anchor="end">a[0] = 0</text>
<text x="140.0" y="185.0" class="fig-small" text-anchor="end">a[1] = 1</text>
<text x="140.0" y="265.0" class="fig-small" text-anchor="end">a[2] = 2</text>
<rect x="150.0" y="60.0" width="80.0" height="80.0" class="fill-soft-b stroke-b" />
<text x="190.0" y="107.0" class="fig-bold" text-anchor="middle">0</text>
<text x="224.0" y="75.0" class="fig-small" text-anchor="end">c=0</text>
<rect x="230.0" y="60.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="270.0" y="107.0" class="fig-bold" text-anchor="middle">2</text>
<text x="304.0" y="75.0" class="fig-small" text-anchor="end">c=2</text>
<rect x="310.0" y="60.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="350.0" y="107.0" class="fig-bold" text-anchor="middle">4</text>
<text x="384.0" y="75.0" class="fig-small" text-anchor="end">c=2</text>
<rect x="150.0" y="140.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="190.0" y="187.0" class="fig-bold" text-anchor="middle">1</text>
<text x="224.0" y="155.0" class="fig-small" text-anchor="end">c=1</text>
<rect x="230.0" y="140.0" width="80.0" height="80.0" class="fill-soft-b stroke-b" />
<text x="270.0" y="187.0" class="fig-bold" text-anchor="middle">1</text>
<text x="304.0" y="155.0" class="fig-small" text-anchor="end">c=1</text>
<rect x="310.0" y="140.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="350.0" y="187.0" class="fig-bold" text-anchor="middle">2</text>
<text x="384.0" y="155.0" class="fig-small" text-anchor="end">c=1</text>
<rect x="150.0" y="220.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="190.0" y="267.0" class="fig-bold" text-anchor="middle">3</text>
<text x="224.0" y="235.0" class="fig-small" text-anchor="end">c=2</text>
<rect x="230.0" y="220.0" width="80.0" height="80.0" class="fill-sunk stroke-grid" />
<text x="270.0" y="267.0" class="fig-bold" text-anchor="middle">1</text>
<text x="304.0" y="235.0" class="fig-small" text-anchor="end">c=0</text>
<rect x="310.0" y="220.0" width="80.0" height="80.0" class="fill-soft-b stroke-b" />
<text x="350.0" y="267.0" class="fig-bold" text-anchor="middle">1</text>
<text x="384.0" y="235.0" class="fig-small" text-anchor="end">c=0</text>
<line x1="214.0" y1="124.0" x2="250.0" y2="160.0" class="stroke-ink" marker-end="url(#c020-arrow)"/>
<line x1="294.0" y1="204.0" x2="330.0" y2="240.0" class="stroke-ink" marker-end="url(#c020-arrow)"/>
<text x="420.0" y="100.0" class="fig-small" text-anchor="start">Big number: D(i, j).</text>
<text x="420.0" y="122.0" class="fig-small" text-anchor="start">Small number: local cost.</text>
<text x="420.0" y="144.0" class="fig-small" text-anchor="start">Shaded cells: the path.</text>
<text x="420.0" y="180.0" class="fig-bold fig-small" text-anchor="start">DTW distance = D(2, 2) = 1</text>
</svg>
Figure 2. The completed table for a = [0, 1, 2] and b = [0, 2, 2]. Notice that each big number is its small local cost plus the smallest big number among its three neighbors, and that the shaded path ends at D(2, 2) = 1.
```

## Read the path back

The table holds only totals. To recover the path, start at the last cell and walk backward. At each cell, step to whichever of the three predecessors has the smallest $D$, until you reach $(0, 0)$. Then reverse the list.

Paths can tie. In the example, $D(2,2) = 1$ can come from $D(1,1) = 1$ or from $D(2,1) = 1$, and both routes cost 1. The exercise breaks ties in the order diagonal, up, left. Either answer is correct, as long as the cells of the path add up to $D(n-1, m-1)$.

The following demo draws two sequences, the cost matrix between them, and the path.

```demo
{"id": "dtw-path", "caption": "Slow down the second segment and watch the path bend to absorb it while the normalized distance barely moves, then narrow the Sakoe-Chiba band and watch the share of cells the table must fill fall."}
```

## Constrain the warp with a band

Unconstrained DTW can map the whole of one word onto the first frame of another, when that one frame happens to be close to everything. Real timing differences are moderate, so Sakoe and Chiba (1978) restricted the path to a **band** around the diagonal. With equal lengths, the band allows cell $(i, j)$ only when $|i - j| \le w$, for a half-width $w$ in frames.

Utterances rarely have equal lengths, so the exercise centers the band on the line from the first corner to the last:

$$
\left| j - i \, \frac{m - 1}{n - 1} \right| \le w
$$

Read it aloud: at row $i$ the line sits at column $i (m-1)/(n-1)$, and a cell is allowed when its column is within $w$ of that. With $n = m$ this is the classic band. Forbidden cells hold $+\infty$ in $D$, so the recurrence steers around them:

```python
allowed = np.abs(np.arange(m)[None, :] - np.arange(n)[:, None] * slope) <= w
```

```figure
<svg viewBox="0 0 620 370" role="img" aria-label="A six by nine grid where only the cells within one column of the corner-to-corner line are shaded">
<text x="110.0" y="77.0" class="fig-small" text-anchor="end">i = 0</text>
<rect x="120.0" y="50.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="164.0" y="50.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="208.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="252.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="296.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="340.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="384.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="428.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="472.0" y="50.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<text x="110.0" y="121.0" class="fig-small" text-anchor="end">i = 1</text>
<rect x="120.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="164.0" y="94.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="208.0" y="94.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="252.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="296.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="340.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="384.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="428.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="472.0" y="94.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<text x="110.0" y="165.0" class="fig-small" text-anchor="end">i = 2</text>
<rect x="120.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="164.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="208.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="252.0" y="138.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="296.0" y="138.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="340.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="384.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="428.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="472.0" y="138.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<text x="110.0" y="209.0" class="fig-small" text-anchor="end">i = 3</text>
<rect x="120.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="164.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="208.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="252.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="296.0" y="182.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="340.0" y="182.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="384.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="428.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="472.0" y="182.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<text x="110.0" y="253.0" class="fig-small" text-anchor="end">i = 4</text>
<rect x="120.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="164.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="208.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="252.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="296.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="340.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="384.0" y="226.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="428.0" y="226.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="472.0" y="226.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<text x="110.0" y="297.0" class="fig-small" text-anchor="end">i = 5</text>
<rect x="120.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="164.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="208.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="252.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="296.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="340.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="384.0" y="270.0" width="44.0" height="44.0" class="fill-sunk stroke-grid" />
<rect x="428.0" y="270.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<rect x="472.0" y="270.0" width="44.0" height="44.0" class="fill-soft-a stroke-grid" />
<text x="142.0" y="42.0" class="fig-small" text-anchor="middle">0</text>
<text x="186.0" y="42.0" class="fig-small" text-anchor="middle">1</text>
<text x="230.0" y="42.0" class="fig-small" text-anchor="middle">2</text>
<text x="274.0" y="42.0" class="fig-small" text-anchor="middle">3</text>
<text x="318.0" y="42.0" class="fig-small" text-anchor="middle">4</text>
<text x="362.0" y="42.0" class="fig-small" text-anchor="middle">5</text>
<text x="406.0" y="42.0" class="fig-small" text-anchor="middle">6</text>
<text x="450.0" y="42.0" class="fig-small" text-anchor="middle">7</text>
<text x="494.0" y="42.0" class="fig-small" text-anchor="middle">8</text>
<line x1="142.0" y1="72.0" x2="494.0" y2="292.0" class="stroke-b dashed"/>
<text x="318.0" y="342.0" class="fig-small" text-anchor="middle">column j (frames of b); allowed cells: |j - i (m-1)/(n-1)| &lt;= 1</text>
<text x="528.0" y="80.0" class="fig-small" text-anchor="start">12 of 54 cells</text>
<text x="528.0" y="100.0" class="fig-small" text-anchor="start">are allowed</text>
</svg>
Figure 3. A band of half-width 1 on a 6 by 9 grid allows 12 of 54 cells. Notice that the allowed region follows the dashed corner-to-corner line, not the main diagonal, because the two sequences have different lengths.
```

The band does two jobs. It forbids absurd warps, and it cuts the work from $nm$ cells to about $n(2w + 1)$. Even so, a 50-frame word against 16 templates fills 40,000 cells per query without a band, which is fine for ten digits and not for 100,000 words.

> [!WARNING]
> A band too narrow for the real timing difference forces a bad path. With $w = 0$ and equal lengths, the only path left is the diagonal, and DTW becomes the plain frame-by-frame distance it was meant to replace.

## Recognize words by nearest template

A template recognizer stores a few MFCC sequences per word. To recognize an unknown utterance, compute its DTW distance to every template and answer with the label of the nearest one.

One adjustment matters. The DTW distance sums over the path, so a longer path has more terms and a long template looks far from everything. Divide by the number of cells on the path, $L$, to get the **average cost per step**:

$$
\bar{D} = \frac{D(n-1, m-1)}{L}
$$

Read it aloud: the total cost of the best path, shared out over its steps. In the exercise's own test set, raw distances name 36 of the 40 words, and per-step averages name all 40.

> [!KEY]
> A template recognizer is nearest-neighbor classification with DTW as the distance. Normalize the distance by path length before you compare templates of different lengths, and you can answer the opening question: two takes of "sofa" are close under DTW even when their timing differs.

> [!RECAP]
> - `local_cost` is `np.sqrt(np.sum((a[:, None] - b[None]) ** 2, axis=-1))`, shape `(n, m)`.
> - $D(i, j) = c(i, j) + \min$ of the diagonal, upper, and left neighbors, with $+\infty$ outside the table and $D(0,0) = c(0,0)$.
> - Backtracking walks from the last cell to $(0, 0)$ through the cheapest predecessor, then reverses.
> - The band allows $|j - i(m-1)/(n-1)| \le w$, and forbidden cells hold `np.inf`.
> - Compare templates by $D(n-1, m-1) / L$, the average cost per step.

## Build it step by step

The checks test the functions in this order. The file provides `in_band`, the synthetic words (`synth_word` and `WORDS`), and the `mfcc` front end.

1. **`local_cost(a, b)`.** Takes `(n, d)` and `(m, d)` arrays and returns the `(n, m)` distances. Broadcast, square, sum over the last axis, and take the square root. The usual mistake is summing over the wrong axis.
2. **`accumulated_cost(cost, band=None)`.** Returns `D`, shape `(n, m)`. Start from `np.full((n, m), np.inf)`, get the mask from `in_band`, and fill the allowed cells row by row. The usual mistake is setting the first row and column to zero, which lets a path start anywhere along an edge.
3. **`backtrack(D)`.** Returns `(i, j)` tuples from `(0, 0)` to `(n - 1, m - 1)`. Take the smallest existing predecessor with `min`, then reverse. The usual mistake is stepping to index `-1`, which numpy reads from the far end of the table.
4. **`dtw(a, b, band=None)`.** Chains the three and returns `(float(D[-1, -1]), path)`, or `(float("inf"), [])` when no path fits the band.
5. **`classify(query, templates, labels, band=None)`.** Scores each template with `distance / len(path)` and returns the label of the smallest. The usual mistake is comparing raw distances.

## Where this breaks

**Speaker changes.** DTW compares raw features, and formants move with vocal tract length, so people stored several templates per word from several speakers.

**Silence at the edges.** Both corners of the path are fixed, so leading silence in the query must match something in the template. Trim the edges with a voice activity detector first.

**No sharing between words.** Each word needs its own templates, and nothing is learned about what varies. The next chapter, [Hidden Markov models](/lessons/021-hidden-markov-models), replaces templates with statistical models of each sound.

## The exercise

You build `local_cost`, `accumulated_cost` with an optional Sakoe-Chiba band, `backtrack`, `dtw`, and the template recognizer `classify`. Press **Run** to hear two takes of the synthetic word "osa", see the warping path drawn on their cost matrix, and see the confusion matrix of a two-templates-per-word recognizer on 40 words at 10 dB SNR.

The checks compare `local_cost` with `scipy.spatial.distance.cdist`, `accumulated_cost` with the 3 by 3 table above, and its last cell with a brute-force search over every path, with and without a band. They confirm that `backtrack` returns a valid path that adds up to the distance, that `dtw` absorbs a repeated frame and reduces to the frame-by-frame distance with a band of 0, and that `classify` names at least 38 of 40 noisy words.
