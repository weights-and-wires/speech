---
title: Diarization
slug: 044-diarization
part: "Part 4 — Neural speech recognition"
summary: Cluster speaker embeddings to find who spoke when, then score it with a DER that handles overlapped speech and a collar.
minutes: 50
gpu: false
objectives:
  - Cluster segment embeddings spectrally, with the number of speakers read off the eigengap of the normalized Laplacian.
  - Compute the diarization error rate with the best one-to-one speaker mapping.
  - "Score overlapped speech and apply a forgiveness collar around reference boundaries, and say what NIST and DIHARD do."
  - Explain why clustering cannot label overlapped speech, and how end-to-end neural diarization does.
prereqs:
  - 039-self-supervised-speech
  - 042-speaker-verification
exercise: 044-diarization
---

# Diarization

A meeting recorder has to answer "who said which part?" Nobody tells it how many people are in the room, who they are, or when they will start. The answer is a set of labels on time: speaker A from 0 s to 12 s, speaker B from 12 s to 20 s, and so on. Transcripts, subtitles, and meeting summaries all need it.

The obvious attempt is to cut the recording into windows, embed each window, and run k-means. It fails in two ways. First, k-means needs the number of clusters before it starts, and you do not know how many people are in the room. Second, it gives every window one label, but people interrupt each other, and a window with two voices has no single right answer.

This chapter fixes the first problem and measures the second. You read the number of speakers off the eigenvalues of a graph built from the embeddings, you score the result with the diarization error rate, and you see how the scoring rules treat overlapped speech. You also see why clustering cannot label overlap at all, and what end-to-end models do about it. The next chapter covers the other labeling question: when each word was said, once you know the words.

## What you already know

- Speaker embeddings put clips of one voice close together by cosine (see [Speaker embeddings and verification](/lessons/042-speaker-verification)).
- k-means groups points by nearest center (see [Self-supervised speech](/lessons/039-self-supervised-speech)).
- An eigenvector $v$ of a symmetric matrix $L$ satisfies $L v = \lambda v$, and its eigenvalue is $\lambda$.
- A **frame** is a short slice of time. This chapter uses 10 ms frames, so 100 frames make one second.
- A **hypothesis** is the system's output, and the **reference** is the human annotation it is scored against.

## Who spoke when: cluster a graph

**Diarization** labels a recording with speaker turns, without knowing the speakers in advance. The classic pipeline cuts speech into short overlapping windows, 0.5 s every 0.25 s in the Run, embeds each window, and clusters the embeddings.

Spectral clustering turns the clustering into a graph problem, which solves the "how many speakers" problem. Build an **affinity matrix** $A$ with one row and column per segment: the cosine between their embeddings, with negative values set to 0. Keep only each row's largest entries (the exercise keeps the top 30%), which cuts weak links between speakers. Then symmetrize with $\max(A, A^\top)$, so that a link kept by either segment counts.

Next, ask how many separate groups the graph has. Form the **normalized Laplacian**:

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
<defs><marker id="c040-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
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

## Other ways to cluster

Spectral clustering is one back end among several, and the other two you meet in practice change only how the segments are grouped.

**Agglomerative hierarchical clustering** (AHC) starts with every segment alone and repeatedly merges the two closest clusters. It stops when the best remaining score falls below a threshold, so the threshold, not $k$, decides the speaker count. That threshold needs tuning on data like yours.

**Probabilistic linear discriminant analysis** (PLDA) replaces the cosine as the score. It models an embedding as a speaker's mean plus within-speaker noise, both Gaussian, and scores a pair of segments by the log ratio of "same speaker" to "different speakers". It learns which embedding directions carry speaker identity, which a plain cosine treats alike.

**VBx** (Landini et al., 2022) goes further. It runs a hidden Markov model whose states are speakers, with PLDA scoring each segment against each state, and it refines the assignments by variational Bayes. The model prefers to stay with one speaker from segment to segment, so a lone noisy window does not flip the label. pyannote's `speaker-diarization-community-1` pipeline uses it.

## Score it: the diarization error rate

The **diarization error rate** (DER) counts three kinds of error over time:

$$
\text{DER} = \frac{\text{missed speech} + \text{false alarm} + \text{speaker confusion}}{\text{total reference speech}}
$$

Read it aloud: missed speech is reference speech the system called silence, a false alarm is the reverse, and confusion is speech given to the wrong speaker. Cluster labels are arbitrary, so before counting confusion, map each system speaker to at most one reference speaker, choosing the mapping that matches the most time. `scipy.optimize.linear_sum_assignment` finds it from the table of overlaps. `diarization_error_rate` scores 10 ms frames with no forgiveness collar at boundaries and one speaker per frame.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Twelve 10 ms frames of reference and system labels: frame 3 is missed speech, frame 4 is speaker confusion, frame 10 is a false alarm">
<defs><marker id="c040-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
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

## Scoring conventions: the collar and overlap

Two choices change a DER by several points without changing the system, so every reported number needs both stated.

**The forgiveness collar.** Human annotators disagree by a few hundred milliseconds about where a turn starts, so frames right at a boundary are noisy ground truth. A **collar** removes them from the score: no frame within the collar of a reference boundary counts, in either the errors or the denominator. The NIST scoring used in the Rich Transcription evaluations uses a collar of 0.25 s on each side, which is 25 frames at 10 ms. The collar goes around *reference* boundaries only. If it followed the hypothesis boundaries, a system that chops speech into many short segments would forgive more of its own mistakes.

**Overlap.** Two people sometimes talk at once. NIST scoring can leave those frames out, and the DIHARD challenges score them, with no collar. Leaving overlap out is kinder, because a clustering system gives each frame one speaker and so cannot get an overlapped frame fully right.

Scoring overlap means a frame can have several reference speakers, so the labels become a boolean matrix with one row per speaker. On frame $t$, let $n^{\text{ref}}_t$ and $n^{\text{hyp}}_t$ count the active speakers in the reference and hypothesis, and $n^{\text{ok}}_t$ count the speakers that match under the speaker mapping. Then:

$$
\text{DER} = \frac{\sum_t \left(\max(n^{\text{ref}}_t, n^{\text{hyp}}_t) - n^{\text{ok}}_t\right)}{\sum_t n^{\text{ref}}_t}
$$

Read it aloud: each frame loses one point for every speaker that is missed, invented, or confused, and the total is divided by all the reference speaker-frames. With one speaker per frame, the numerator is exactly missed speech plus false alarm plus confusion, so it reduces to the DER above.

Work an overlap by hand, as in Figure 3. Speaker A talks in frames 0 to 7 and speaker B in frames 5 to 11. The system finds A correctly but hears B only from frame 8, so frames 5, 6, and 7 each lose a speaker. The reference holds $8 + 7 = 15$ speaker-frames, so the DER is $3/15 = 20\%$. A scorer that gave each frame one label would see the right first speaker everywhere and report 0%.

Add a collar of 1 frame. The reference boundaries sit before frames 0, 5, and 8, and after frame 11. A boundary between frames $t - 1$ and $t$ forgives frames $t - 1$ and $t$ here, so frames 0, 4, 5, 7, 8, and 11 leave the score. They hold 8 of the 15 speaker-frames. Of the three errors, frames 5 and 7 are forgiven, and only frame 6 stays, so the DER is $1/7 = 14.3\%$. With overlap left out instead, frames 5 to 7 vanish and the DER is $0/9 = 0\%$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Two reference speakers that overlap in frames 5 to 7 and a system that hears the second speaker from frame 8: frames 5, 6, and 7 are missed, and frames 5 and 7 fall inside the collar">
<text x="128" y="46" text-anchor="middle" class="fig-small">0</text>
<text x="168" y="46" text-anchor="middle" class="fig-small">1</text>
<text x="208" y="46" text-anchor="middle" class="fig-small">2</text>
<text x="248" y="46" text-anchor="middle" class="fig-small">3</text>
<text x="288" y="46" text-anchor="middle" class="fig-small">4</text>
<text x="328" y="46" text-anchor="middle" class="fig-small">5</text>
<text x="368" y="46" text-anchor="middle" class="fig-small">6</text>
<text x="408" y="46" text-anchor="middle" class="fig-small">7</text>
<text x="448" y="46" text-anchor="middle" class="fig-small">8</text>
<text x="488" y="46" text-anchor="middle" class="fig-small">9</text>
<text x="528" y="46" text-anchor="middle" class="fig-small">10</text>
<text x="568" y="46" text-anchor="middle" class="fig-small">11</text>
<text x="20" y="46" class="fig-small">frame</text>
<text x="20" y="84" class="fig-small fig-bold">reference A</text>
<rect x="110" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="150" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="190" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="230" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="270" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="310" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="350" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="390" y="60" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="430" y="60" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="470" y="60" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="510" y="60" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="550" y="60" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<text x="20" y="128" class="fig-small fig-bold">reference B</text>
<rect x="110" y="104" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="150" y="104" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="190" y="104" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="230" y="104" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="270" y="104" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="310" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="350" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="390" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="430" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="470" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="510" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<rect x="550" y="104" width="36" height="32" rx="4" class="fill-soft-a stroke-a"/>
<text x="20" y="188" class="fig-small fig-bold">system X</text>
<rect x="110" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="150" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="190" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="230" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="270" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="310" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="350" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="390" y="164" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="430" y="164" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="470" y="164" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="510" y="164" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="550" y="164" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<text x="20" y="232" class="fig-small fig-bold">system Y</text>
<rect x="110" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="150" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="190" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="230" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="270" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="310" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="350" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="390" y="208" width="36" height="32" rx="4" class="fill-sunk stroke-muted"/>
<rect x="430" y="208" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="470" y="208" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="510" y="208" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="550" y="208" width="36" height="32" rx="4" class="fill-soft-c stroke-c"/>
<rect x="108" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<rect x="268" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<rect x="308" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<rect x="308" y="98" width="40" height="44" rx="5" class="stroke-bad" fill="none"/>
<rect x="348" y="98" width="40" height="44" rx="5" class="stroke-bad" fill="none"/>
<rect x="388" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<rect x="388" y="98" width="40" height="44" rx="5" class="stroke-bad" fill="none"/>
<rect x="428" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<rect x="548" y="54" width="40" height="192" rx="5" class="stroke-muted dashed" fill="none"/>
<text x="300" y="274" text-anchor="middle" class="fig-small">no collar: 3 errors / 15 speaker-frames = 20%   |   collar of 1 frame: 1 / 7 = 14.3%</text>
<text x="300" y="292" text-anchor="middle" class="fig-small">overlap not scored: 0 / 9 = 0%</text>
</svg>
Figure 3. Speaker B talks over speaker A in frames 5 to 7, and the system hears B only from frame 8, so those three frames each lose a speaker (red outlines on the reference). The dashed outlines mark the frames a collar of 1 frame forgives: they surround the reference boundaries only, and they remove frames 5 and 7 from the score.
```

> [!WARNING]
> A DER without its collar and overlap setting cannot be compared with another. The same system can score several points worse with no collar and overlap counted than with a 0.25 s collar and overlap left out.

## Overlap needs a different model

Clustering has a built-in limit. It gives each window one speaker, so overlapped speech always costs at least one missed speaker. Pipelines patch this by detecting overlap regions and giving those frames a second speaker, the next closest cluster, but the second label is a guess.

**End-to-end neural diarization** (EEND; Fujita et al., 2019) removes the limit by changing the output. A network reads the whole recording and outputs, for every frame, a probability that each of $C$ speakers is active. Each speaker has its own sigmoid, so several can be active at once. That makes the task multi-label, and it is the same boolean matrix the overlap DER scores. Training raises one more problem: the network's speaker slots have no fixed meaning, so slot 1 might be speaker B. **Permutation-invariant training** (PIT) fixes it. The loss tries every ordering of the slots against the reference and keeps the smallest. Variants add attractors to handle an unknown number of speakers (EEND-EDA). [Neural diarization](/lessons/045-neural-diarization) builds the PIT loss and the attractor count.

> [!KEY]
> Diarization is clustering plus a scoring contract. Spectral clustering answers "how many speakers" from an eigengap, the DER measures the labels after the best speaker mapping, and the collar and overlap setting decide what that number means.

> [!RECAP]
> - Spectral clustering prunes and symmetrizes cosine affinities, reads $k$ off the Laplacian's eigengap, and clusters eigenvector rows.
> - AHC with a threshold and VBx with PLDA scoring are the other common back ends.
> - DER counts misses, false alarms, and confusion after the best one-to-one speaker mapping.
> - A collar forgives frames around reference boundaries, and overlap can be scored or left out.
> - EEND outputs frame-wise multi-label speaker activity and trains with a permutation-invariant loss.

## Build it step by step

The checks test the functions in this order. `kmeans_rows` and `aam_softmax_loss` are provided, because the Run uses them and you built the loss in [Speaker embeddings and verification](/lessons/042-speaker-verification).

1. **`affinity_matrix(embeddings, keep)`.** Returns `(N, N)`. Use `np.argsort(-sim, axis=1)` to find each row's top `ceil(keep * N)` entries. Common mistake: symmetrizing before pruning.
2. **`speaker_count(affinity, max_speakers)`.** Returns an int from `np.linalg.eigvalsh`. Common mistake: an off-by-one, since the gap after the first eigenvalue means one speaker.
3. **`spectral_cluster(embeddings, max_speakers, keep)`.** Returns `(N,)` labels. Use `np.linalg.eigh` and the provided `kmeans_rows`. Common mistake: clustering the eigenvector columns instead of the rows.
4. **`diarization_error_rate(reference, hypothesis)`.** Returns a float. Common mistake: mapping two system speakers to one reference speaker.
5. **`der_collar_overlap(reference, hypothesis, collar, score_overlap)`.** Takes boolean `(speakers, frames)` matrices and returns a float. Find boundaries with `np.pad` and a difference along time, mark the forgiven frames, then score `max(n_ref, n_hyp) - matched` on the rest. Common mistake: taking the boundaries from the hypothesis, or counting an overlapped frame once.

## Where this breaks

**The wrong number of speakers.** When a speaker talks only briefly, the eigengap can miss them. Cap `max_speakers` sensibly, or pass the count when you know it.

**Short turns.** A 0.5 s window that straddles two speakers matches neither, so turns shorter than the window blur into a neighbor. The Run's errors all sit at turn boundaries for this reason.

**Domain mismatch.** Embeddings trained on clean read speech cluster poorly on far-field meetings, where room acoustics shift every voice. Evaluate on audio like the audio you will serve, and report the collar and overlap setting with the DER.

## The exercise

You build `affinity_matrix`, `speaker_count`, `spectral_cluster`, and `diarization_error_rate` for the clustering pipeline, and `der_collar_overlap` for the scoring conventions.

The checks confirm the eigengap on block matrices, exact recovery of four synthetic speakers, DER against a brute-force search over mappings, and the overlap DER against a slow reference on random overlapping speakers, with and without a collar. One check catches a scorer that counts an overlapped frame once, and another catches a collar placed on the hypothesis boundaries. The **Run** trains embeddings on synthetic voices, scores the overlap example by hand, and diarizes a conversation you can hear.

## Further reading

- Wang et al., "Speaker Diarization with LSTM," ICASSP 2018 (https://arxiv.org/abs/1710.10468). Spectral clustering of embeddings with the eigengap speaker count.
- Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks," Computer Speech & Language, 2022 (https://arxiv.org/abs/2012.14952).
- Fujita et al., "End-to-End Neural Speaker Diarization with Permutation-Free Objectives," Interspeech 2019 (https://arxiv.org/abs/1909.05952).
- Horiguchi et al., "End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors," Interspeech 2020 (https://arxiv.org/abs/2005.09921).
- Park et al., "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap," IEEE Signal Processing Letters, 2020 (https://arxiv.org/abs/2003.02405).
