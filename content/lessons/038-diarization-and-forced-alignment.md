---
title: Diarization and forced alignment
slug: 038-diarization-and-forced-alignment
part: "Part 4 — Neural speech recognition"
summary: Cluster speaker embeddings to find who spoke when, score it with DER, and align a known transcript to CTC frames by Viterbi.
minutes: 55
gpu: false
objectives:
  - Cluster segment embeddings spectrally, with the number of speakers read off the eigengap of the normalized Laplacian.
  - Compute the diarization error rate with the best one-to-one speaker mapping.
  - Align a known transcript to CTC emissions by Viterbi over the blank-extended sequence.
  - Read token times off the alignment path, at 20 ms per frame.
prereqs:
  - 021-hidden-markov-models
  - 027-ctc-many-paths-one-label
  - 028-ctc-forward-backward
  - 037-speaker-verification
exercise: 038-diarization-and-forced-alignment
---

# Diarization and forced alignment

A meeting recorder has to answer "who said which part?" A subtitle tool that already has the words has to answer "when was each one said?" Neither question is about what was said. Both put labels on time.

The obvious attempts fail in a useful way. For the first question, you could cut the recording into windows, embed each, and run k-means. But k-means needs the number of clusters first, and you do not know how many people are in the room. For the second, you could split the audio evenly among the words. But words differ in length, so the split drifts off after the first long word.

This chapter fixes both. For the first, you read the number of speakers off the eigenvalues of a graph built from the embeddings, and you score the result with the diarization error rate. For the second, you find the single best path through the CTC lattice that spells the known transcript, and you read each token's frames off it.

## What you already know

- Speaker embeddings put clips of one voice close together by cosine (see [Speaker embeddings and verification](/lessons/037-speaker-verification)).
- k-means groups points by nearest center (see [Self-supervised speech](/lessons/035-self-supervised-speech)).
- CTC scores a transcript by summing over paths through the blank-extended sequence (see [CTC: many paths, one label](/lessons/027-ctc-many-paths-one-label)).
- Viterbi finds the best path through a trellis by replacing a sum with a max and keeping backpointers (see [Hidden Markov models](/lessons/021-hidden-markov-models)).
- An eigenvector $v$ of a symmetric matrix $L$ satisfies $L v = \lambda v$.

## Who spoke when: cluster a graph

**Diarization** labels a recording with speaker turns, without knowing the speakers in advance. The classic pipeline cuts speech into short overlapping windows, 0.5 s every 0.25 s in the Run, embeds each window, and clusters the embeddings.

Spectral clustering turns the clustering into a graph problem, which solves the "how many speakers" problem. Build an **affinity matrix** $A$ with one row and column per segment: the cosine between their embeddings, with negative values set to 0. Keep only each row's largest entries (the exercise keeps the top 30%), which cuts weak links between speakers. Then symmetrize with $\max(A, A^\top)$, so that a link kept by either segment counts.

Now ask how many separate groups the graph has. Form the **normalized Laplacian**:

$$
L = I - D^{-1/2} A D^{-1/2}
$$

Read it aloud: $D$ is the diagonal matrix of the row sums of $A$, so each entry of $A$ is divided by the square root of both its row's and its column's total. The eigenvalues of $L$ lie between 0 and 2, and a graph that falls into $k$ disconnected groups has exactly $k$ eigenvalues equal to 0. The reason is that each group is an island, and the vector that holds the square roots of the row sums on one island and zeros elsewhere satisfies $Lv = 0$. With noise, you get $k$ small eigenvalues followed by a jump, the **eigengap**. So pick the $k$ that maximizes $\lambda_{k+1} - \lambda_k$. In numpy:

```python
d = 1 / np.sqrt(A.sum(axis=1))
L = np.eye(len(A)) - d[:, None] * A * d[None, :]
eigenvalues, vectors = np.linalg.eigh(L)
```

Work a tiny case. Four segments in two pairs, with affinity 1 inside each pair and 0 across, give row sums of 2, so $D^{-1/2} A D^{-1/2} = A/2$. Its eigenvalues are 1, 1, 0, and 0, so $L$'s are 0, 0, 1, and 1. The gaps are 0, 1, and 0, and the largest is at $k = 2$: two speakers.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The eight smallest eigenvalues of the normalized Laplacian for 43 segments: three near zero, then a jump from 0.098 to 0.507">
<defs><marker id="c037-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<line x1="80.0" y1="240.0" x2="560.0" y2="240.0" class="stroke-muted"/>
<line x1="80.0" y1="240.0" x2="560.0" y2="240.0" class="stroke-grid"/>
<text x="72.0" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="80.0" y1="190.0" x2="560.0" y2="190.0" class="stroke-grid"/>
<text x="72.0" y="194.0" text-anchor="end" class="fig-small">0.25</text>
<line x1="80.0" y1="140.0" x2="560.0" y2="140.0" class="stroke-grid"/>
<text x="72.0" y="144.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="80.0" y1="90.0" x2="560.0" y2="90.0" class="stroke-grid"/>
<text x="72.0" y="94.0" text-anchor="end" class="fig-small">0.75</text>
<line x1="80.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="72.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<rect x="100.0" y="238.5" width="40.0" height="1.5" rx="2" class="fill-b"/>
<text x="120.0" y="232.5" text-anchor="middle" class="fig-small">0.000</text>
<text x="120.0" y="258.0" text-anchor="middle" class="fig-small">1</text>
<rect x="158.0" y="226.2" width="40.0" height="13.8" rx="2" class="fill-b"/>
<text x="178.0" y="220.2" text-anchor="middle" class="fig-small">0.069</text>
<text x="178.0" y="258.0" text-anchor="middle" class="fig-small">2</text>
<rect x="216.0" y="220.4" width="40.0" height="19.6" rx="2" class="fill-b"/>
<text x="236.0" y="214.4" text-anchor="middle" class="fig-small">0.098</text>
<text x="236.0" y="258.0" text-anchor="middle" class="fig-small">3</text>
<rect x="274.0" y="138.7" width="40.0" height="101.3" rx="2" class="fill-a"/>
<text x="294.0" y="132.7" text-anchor="middle" class="fig-small">0.507</text>
<text x="294.0" y="258.0" text-anchor="middle" class="fig-small">4</text>
<rect x="332.0" y="75.0" width="40.0" height="165.0" rx="2" class="fill-a"/>
<text x="352.0" y="69.0" text-anchor="middle" class="fig-small">0.825</text>
<text x="352.0" y="258.0" text-anchor="middle" class="fig-small">5</text>
<rect x="390.0" y="66.0" width="40.0" height="174.0" rx="2" class="fill-a"/>
<text x="410.0" y="60.0" text-anchor="middle" class="fig-small">0.870</text>
<text x="410.0" y="258.0" text-anchor="middle" class="fig-small">6</text>
<rect x="448.0" y="61.4" width="40.0" height="178.6" rx="2" class="fill-a"/>
<text x="468.0" y="55.4" text-anchor="middle" class="fig-small">0.893</text>
<text x="468.0" y="258.0" text-anchor="middle" class="fig-small">7</text>
<rect x="506.0" y="55.1" width="40.0" height="184.9" rx="2" class="fill-a"/>
<text x="526.0" y="49.1" text-anchor="middle" class="fig-small">0.925</text>
<text x="526.0" y="258.0" text-anchor="middle" class="fig-small">8</text>
<text x="315.0" y="278.0" text-anchor="middle" class="fig-small">eigenvalue index k, smallest first</text>
<line x1="236.0" y1="206.4" x2="236.0" y2="138.6" class="stroke-muted dashed"/>
<line x1="236.0" y1="138.6" x2="272.0" y2="138.6" class="stroke-muted dashed"/>
<text x="228.0" y="180.0" text-anchor="end" class="fig-small fig-bold">gap 0.409</text>
<text x="100.0" y="62.0" text-anchor="start" class="fig-small">the largest gap comes after k = 3,</text>
<text x="100.0" y="78.0" text-anchor="start" class="fig-small">so the system reads three speakers</text>
</svg>
Figure 1. The three smallest eigenvalues sit near zero, one per speaker, and the fourth jumps to 0.507. The largest gap is after k = 3, so the system reads three speakers off the eigenvalues without being told.
```

Figure 1 shows the same pattern in the Run's 43 windows of three unseen voices. To cluster, take the eigenvectors of the $k$ smallest eigenvalues as the columns of an $N \times k$ matrix, scale each row to unit length, and run k-means on the rows. Segments from one speaker have nearly the same row, so k-means separates them cleanly.

> [!KEY]
> Spectral clustering reads the number of speakers from the eigengap of a graph Laplacian, then runs k-means on eigenvector rows, so you never have to know $k$ in advance.

## Score it: the diarization error rate

The **diarization error rate** (DER) counts three kinds of error over time:

$$
\text{DER} = \frac{\text{missed speech} + \text{false alarm} + \text{speaker confusion}}{\text{total reference speech}}
$$

Read it aloud: missed speech is reference speech the system called silence, a false alarm is the reverse, and confusion is speech given to the wrong speaker. Cluster labels are arbitrary, so before counting confusion, map each system speaker to at most one reference speaker, choosing the mapping that matches the most time. `scipy.optimize.linear_sum_assignment` finds it from the table of overlaps. The exercise scores 10 ms frames with no forgiveness collar at boundaries, one speaker per frame.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Twelve 10 ms frames of reference and system labels: frame 3 is missed speech, frame 4 is speaker confusion, frame 10 is a false alarm">
<defs><marker id="c037-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="127.0" y="30.0" text-anchor="middle" class="fig-small">0</text>
<rect x="110.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="127.0" y="66.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="110.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="127.0" y="126.0" text-anchor="middle" class="fig-bold">5</text>
<text x="165.0" y="30.0" text-anchor="middle" class="fig-small">1</text>
<rect x="148.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="165.0" y="66.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="148.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="165.0" y="126.0" text-anchor="middle" class="fig-bold">5</text>
<text x="203.0" y="30.0" text-anchor="middle" class="fig-small">2</text>
<rect x="186.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="203.0" y="66.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="186.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="203.0" y="126.0" text-anchor="middle" class="fig-bold">5</text>
<text x="241.0" y="30.0" text-anchor="middle" class="fig-small">3</text>
<rect x="224.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="241.0" y="66.0" text-anchor="middle" class="fig-bold">0</text>
<rect x="224.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="241.0" y="126.0" text-anchor="middle" class="fig-bold">-</text>
<text x="279.0" y="30.0" text-anchor="middle" class="fig-small">4</text>
<rect x="262.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="279.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="262.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="279.0" y="126.0" text-anchor="middle" class="fig-bold">5</text>
<text x="317.0" y="30.0" text-anchor="middle" class="fig-small">5</text>
<rect x="300.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="317.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="300.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="317.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="355.0" y="30.0" text-anchor="middle" class="fig-small">6</text>
<rect x="338.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="355.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="338.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="355.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="393.0" y="30.0" text-anchor="middle" class="fig-small">7</text>
<rect x="376.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="393.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="376.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="393.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="431.0" y="30.0" text-anchor="middle" class="fig-small">8</text>
<rect x="414.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="431.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="414.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="431.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="469.0" y="30.0" text-anchor="middle" class="fig-small">9</text>
<rect x="452.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="469.0" y="66.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="452.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="469.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="507.0" y="30.0" text-anchor="middle" class="fig-small">10</text>
<rect x="490.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="507.0" y="66.0" text-anchor="middle" class="fig-bold">-</text>
<rect x="490.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-soft-c stroke-c"/>
<text x="507.0" y="126.0" text-anchor="middle" class="fig-bold">7</text>
<text x="545.0" y="30.0" text-anchor="middle" class="fig-small">11</text>
<rect x="528.0" y="40.0" width="34.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="545.0" y="66.0" text-anchor="middle" class="fig-bold">-</text>
<rect x="528.0" y="100.0" width="34.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="545.0" y="126.0" text-anchor="middle" class="fig-bold">-</text>
<text x="20.0" y="66.0" text-anchor="start" class="fig-bold fig-small">reference</text>
<text x="20.0" y="126.0" text-anchor="start" class="fig-bold fig-small">system</text>
<text x="20.0" y="18.0" text-anchor="start" class="fig-small">frame</text>
<rect x="222" y="38" width="38" height="104" rx="5" class="stroke-bad" fill="none"/>
<text x="255.0" y="162.0" text-anchor="end" class="fig-small fig-bold">missed</text>
<rect x="260" y="38" width="38" height="104" rx="5" class="stroke-bad" fill="none"/>
<text x="265.0" y="162.0" text-anchor="start" class="fig-small fig-bold">confusion</text>
<rect x="488" y="38" width="38" height="104" rx="5" class="stroke-bad" fill="none"/>
<text x="507.0" y="162.0" text-anchor="middle" class="fig-small fig-bold">false alarm</text>
<text x="300.0" y="198.0" text-anchor="middle" class="fig-small">mapping 5 to 0 and 7 to 1 matches 8 of the 9 frames where both speak</text>
<text x="300.0" y="218.0" text-anchor="middle" class="fig-small">DER = (1 missed + 1 false alarm + 1 confusion) / 10 reference speech frames = 30%</text>
</svg>
Figure 2. Cluster ids such as 5 and 7 are arbitrary, so the scorer first maps each system speaker to a reference speaker. The three outlined frames are the only errors, and the reference has 10 speech frames, so the DER is 30%.
```

Work 12 frames by hand, as in Figure 2. The reference is `0 0 0 0 1 1 1 1 1 1 - -` and the system says `5 5 5 - 5 7 7 7 7 7 7 -`, where `-` is silence. Frame 3 is missed, and frame 10 is a false alarm. Mapping 5 to 0 and 7 to 1 matches 8 of the 9 frames where both have speech, so frame 4 is confused. The DER is $(1 + 1 + 1)/10 = 30\%$.

The Run plays three unseen synthetic voices taking nine turns, embeds the windows with a margin-trained embedding, and clusters them. The eigengap finds 3 speakers in 43 windows, and the DER is 2.1%. Every error lies within 0.12 s of a turn boundary, where a window straddles two voices.

## When was each word said: forced alignment

Now the second question. You have CTC emissions, the per-frame log-probabilities $\log y^t_k$ from an acoustic model, and a known transcript $l$. **Forced alignment** asks for the single most likely path that spells $l$. Build the blank-extended sequence $l' = (\varnothing, l_1, \varnothing, l_2, \ldots, l_L, \varnothing)$ of length $S = 2L + 1$, as for the CTC loss. Run the forward recursion with a maximum in place of the sum:

$$
\delta_t(s) = \log y^t_{l'_s} + \max\left(\delta_{t-1}(s),\ \delta_{t-1}(s-1),\ \delta_{t-1}(s-2)\right)
$$

Read it aloud: the best path into state $s$ at frame $t$ scores this frame's log-probability of that state's symbol, plus the best of three ways to arrive: stay, step forward one state, or skip a blank. The skip from $s - 2$ is allowed only when $l'_s$ is not blank and differs from $l'_{s-2}$, the same rule as the CTC loss, because two identical letters need a blank between them. Start in state 0 or 1, end in state $S-1$ or $S-2$, keep a backpointer to the choice at every cell, and walk back from the better end.

Odd states belong to tokens: state $2k + 1$ is token $k$. The frames spent in that state are the token's span, and at 20 ms per frame they give times.

### Work three frames by hand

Align `a` over three frames whose probabilities of (blank, `a`) are (0.6, 0.4), (0.3, 0.7), and (0.8, 0.2). Six paths spell `a`: `a__` (0.096), `_a_` (0.336), `__a` (0.036), `aa_` (0.224), `_aa` (0.084), and `aaa` (0.056). The CTC loss sums them, 0.832. Forced alignment keeps the best, `_a_`, with probability $0.6 \times 0.7 \times 0.8 = 0.336$ and states $(0, 1, 2)$: `a` occupies frame 1 alone.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="A CTC lattice for the transcript a over three frames: states blank, a, blank. The best path takes blank, a, blank with probability 0.336">
<defs><marker id="c037-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="87.0" text-anchor="start" class="fig-small">state 0: blank</text>
<text x="20.0" y="147.0" text-anchor="start" class="fig-small">state 1: a</text>
<text x="20.0" y="207.0" text-anchor="start" class="fig-small">state 2: blank</text>
<text x="195.0" y="40.0" text-anchor="middle" class="fig-small fig-bold">frame 0</text>
<text x="305.0" y="40.0" text-anchor="middle" class="fig-small fig-bold">frame 1</text>
<text x="415.0" y="40.0" text-anchor="middle" class="fig-small fig-bold">frame 2</text>
<rect x="150.0" y="60.0" width="90.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="195.0" y="87.0" text-anchor="middle" class="fig-bold">0.6</text>
<rect x="150.0" y="120.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="195.0" y="147.0" text-anchor="middle">0.4</text>
<rect x="150.0" y="180.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="195.0" y="207.0" text-anchor="middle">0.6</text>
<rect x="260.0" y="60.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="305.0" y="87.0" text-anchor="middle">0.3</text>
<rect x="260.0" y="120.0" width="90.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="305.0" y="147.0" text-anchor="middle" class="fig-bold">0.7</text>
<rect x="260.0" y="180.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="305.0" y="207.0" text-anchor="middle">0.3</text>
<rect x="370.0" y="60.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="415.0" y="87.0" text-anchor="middle">0.8</text>
<rect x="370.0" y="120.0" width="90.0" height="44.0" rx="6" class="fill-sunk stroke-muted"/>
<text x="415.0" y="147.0" text-anchor="middle">0.2</text>
<rect x="370.0" y="180.0" width="90.0" height="44.0" rx="6" class="fill-soft-b stroke-b"/>
<text x="415.0" y="207.0" text-anchor="middle" class="fig-bold">0.8</text>
<line x1="240.0" y1="82.0" x2="258.0" y2="130.0" class="stroke-b" marker-end="url(#c037-arrow)"/>
<line x1="350.0" y1="142.0" x2="368.0" y2="190.0" class="stroke-b" marker-end="url(#c037-arrow)"/>
<text x="530.0" y="92.0" text-anchor="middle" class="fig-small fig-bold">best path</text>
<text x="530.0" y="110.0" text-anchor="middle" class="fig-small">0.6 x 0.7 x 0.8</text>
<text x="530.0" y="128.0" text-anchor="middle" class="fig-small fig-bold">= 0.336</text>
<text x="300.0" y="248.0" text-anchor="middle" class="fig-small">the 3-frame sum over all six paths that spell a is 0.832; Viterbi keeps only the best</text>
</svg>
Figure 3. Each cell shows the probability of that state's symbol at that frame. The highlighted path spends frame 1 alone on a, so the token's span is frame 1, or 20 ms to 40 ms at 20 ms per frame.
```

> [!KEY]
> Forced alignment is Viterbi on the CTC lattice of one known transcript. The path's odd states give each token's frames.

> [!RECAP]
> - Spectral clustering prunes and symmetrizes cosine affinities, reads $k$ off the Laplacian's eigengap, and clusters eigenvector rows.
> - DER counts misses, false alarms, and confusion after the best one-to-one speaker mapping.
> - Forced alignment is the CTC forward recursion with max, backpointers, and the same skip rule.

## Build it step by step

The checks test the functions in this order. `kmeans_rows` and `aam_softmax_loss` are provided, because the Run uses them and you built the loss in [Speaker embeddings and verification](/lessons/037-speaker-verification).

1. **`affinity_matrix(embeddings, keep)`.** Returns `(N, N)`. Use `np.argsort(-sim, axis=1)` to find each row's top `ceil(keep * N)` entries. Common mistake: symmetrizing before pruning.
2. **`speaker_count(affinity, max_speakers)`.** Returns an int from `np.linalg.eigvalsh`. Common mistake: an off-by-one, since the gap after the first eigenvalue means one speaker.
3. **`spectral_cluster(embeddings, max_speakers, keep)`.** Returns `(N,)` labels. Use `np.linalg.eigh` and the provided `kmeans_rows`. Common mistake: clustering the eigenvector columns instead of the rows.
4. **`diarization_error_rate(reference, hypothesis)`.** Returns a float. Common mistake: mapping two system speakers to one reference speaker.
5. **`forced_align(log_probs, tokens, blank)`.** Returns `(states, score)`. Fill a `(T, S)` table of `-np.inf`, and raise `ValueError` when both end states stay at `-np.inf`. Common mistake: allowing the skip between two identical letters.

## Where this breaks

**Overlapping speech.** A clustering pipeline gives each window one speaker, so two people talking at once count as missed speech. End-to-end diarization models, such as EEND and pyannote's segmentation model, predict several speakers per frame.

**The wrong number of speakers.** When a speaker talks only briefly, the eigengap can miss them. Cap `max_speakers` sensibly, or pass the count when you know it.

**Peaky CTC timing.** CTC models emit each token as a spike a frame or two from where a phonetician would put it. Decide whether a word's end extends into the following blanks, and report the frame rate. Normalize the transcript to the model's vocabulary before alignment, and align long audio chapter by chapter, because the table grows as $T(2L + 1)$.

## The exercise

You build `affinity_matrix`, `speaker_count`, `spectral_cluster`, and `diarization_error_rate` for diarization, and `forced_align` for alignment.

The checks confirm the eigengap on block matrices, exact recovery of four synthetic speakers, DER against a brute-force search over mappings, and `forced_align` against enumeration of every path, repeated letters included. The **Run** trains embeddings on synthetic voices, diarizes a conversation you can hear, and aligns `hello` to emissions from a make-believe acoustic model, printing each letter's span in seconds.
