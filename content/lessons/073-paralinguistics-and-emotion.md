---
title: Paralinguistics and emotion
slug: 073-paralinguistics-and-emotion
part: "Part 8 — Beyond recognition and synthesis"
summary: Measure how a voice sounds with F0, jitter, and shimmer, classify emotion scored by UAR, and judge noisy labels with kappa and CCC.
minutes: 45
gpu: false
objectives:
  - Compute local jitter and shimmer from per-cycle periods and amplitudes, and summarize an F0 track over its voiced frames.
  - Train a linear emotion classifier with class-balanced weights, and score it by unweighted average recall (UAR) from a confusion matrix.
  - Explain why accuracy misleads on imbalanced emotion data and why a majority-class guesser scores 1/3 UAR on three classes.
  - Measure annotator agreement with Cohen's kappa for categories and the concordance correlation coefficient (CCC) for arousal and valence.
  - Relate F0, energy, and rate to the controls a text-to-speech model exposes.
prereqs:
  - 018-pitch-tracking
  - 072-audio-classification-and-clap
exercise: 073-paralinguistics-and-emotion
---

# Paralinguistics and emotion

A caller says "fine" in a flat, slow voice. A transcript shows one word and no sign that the caller is upset. A support system that wants to route this call to a person needs something the words do not carry: *how* the caller sounds.

That information is *paralinguistics*: emotion, stress, and speaker traits. It also drives synthesis, because a text-to-speech system that reads "fine" must decide how high, how loud, and how fast, and an emotion recognizer measures the same three things from the other direction.

This chapter builds the measuring tools: descriptors of a voice, a classifier, and a way to ask how good a score can be when the human labels disagree. Figure 1 shows the two descriptors that this chapter introduces.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Period and peak amplitude of 20 glottal cycles for a steady voice, nearly flat, and a rough voice, scattered">
  <line x1="70" y1="200" x2="300" y2="200" class="stroke-muted"/>
  <line x1="70" y1="50" x2="70" y2="200" class="stroke-muted"/>
  <text x="70" y="28" class="fig-bold">Period of each cycle</text>
  <line x1="70" y1="200.0" x2="300" y2="200.0" class="stroke-grid"/>
  <text x="64" y="204.0" class="fig-small" text-anchor="end">120</text>
  <line x1="70" y1="125.0" x2="300" y2="125.0" class="stroke-grid"/>
  <text x="64" y="129.0" class="fig-small" text-anchor="end">134</text>
  <line x1="70" y1="50.0" x2="300" y2="50.0" class="stroke-grid"/>
  <text x="64" y="54.0" class="fig-small" text-anchor="end">148</text>
  <polyline points="75.8,128.4 87.2,128.8 98.8,127.7 110.2,128.4 121.8,129.3 133.2,128.1 144.8,126.7 156.2,127.2 167.8,129.6 179.2,130.4 190.8,129.5 202.2,128.5 213.8,131.9 225.2,128.9 236.8,130.4 248.2,129.6 259.8,129.3 271.2,129.0 282.8,128.0 294.2,127.1" fill="none" class="stroke-a"/>
  <polyline points="75.8,151.1 87.2,101.5 98.8,104.5 110.2,114.6 121.8,123.8 133.2,134.2 144.8,102.5 156.2,93.6 167.8,96.4 179.2,105.1 190.8,122.2 202.2,150.1 213.8,128.7 225.2,116.8 236.8,151.6 248.2,121.5 259.8,120.9 271.2,116.1 282.8,149.7 294.2,140.4" fill="none" class="stroke-b"/>
  <text x="185.0" y="220" class="fig-small" text-anchor="middle">cycle number</text>
  <text x="26" y="125.0" class="fig-small" text-anchor="middle" transform="rotate(-90 26 125.0)">period (samples)</text>
  <line x1="350" y1="200" x2="580" y2="200" class="stroke-muted"/>
  <line x1="350" y1="50" x2="350" y2="200" class="stroke-muted"/>
  <text x="350" y="28" class="fig-bold">Peak amplitude of each cycle</text>
  <line x1="350" y1="200.0" x2="580" y2="200.0" class="stroke-grid"/>
  <text x="344" y="204.0" class="fig-small" text-anchor="end">0.4</text>
  <line x1="350" y1="125.0" x2="580" y2="125.0" class="stroke-grid"/>
  <text x="344" y="129.0" class="fig-small" text-anchor="end">1.0</text>
  <line x1="350" y1="50.0" x2="580" y2="50.0" class="stroke-grid"/>
  <text x="344" y="54.0" class="fig-small" text-anchor="end">1.6</text>
  <polyline points="355.8,125.2 367.2,123.3 378.8,125.8 390.2,124.6 401.8,123.9 413.2,124.9 424.8,125.9 436.2,126.2 447.8,125.6 459.2,124.7 470.8,126.3 482.2,125.3 493.8,125.2 505.2,124.3 516.8,124.7 528.2,124.6 539.8,125.8 551.2,125.2 562.8,124.0 574.2,123.1" fill="none" class="stroke-a"/>
  <polyline points="355.8,133.2 367.2,146.9 378.8,92.4 390.2,134.3 401.8,118.8 413.2,129.8 424.8,95.3 436.2,100.2 447.8,113.1 459.2,166.3 470.8,124.0 482.2,112.2 493.8,106.2 505.2,136.6 516.8,90.8 528.2,149.8 539.8,137.4 551.2,107.5 562.8,124.1 574.2,87.5" fill="none" class="stroke-b"/>
  <text x="465.0" y="220" class="fig-small" text-anchor="middle">cycle number</text>
  <text x="306" y="125.0" class="fig-small" text-anchor="middle" transform="rotate(-90 306 125.0)">amplitude (relative)</text>
  <text x="70" y="256" class="fig-small fig-bold">steady: jitter 0.2%, shimmer 0.8%</text>
  <text x="350" y="256" class="fig-small fig-bold">rough: jitter 2.4%, shimmer 21.6%</text>
</svg>
Figure 1. Twenty glottal cycles of a steady voice and a rough one, both at 120 Hz. Notice that the steady voice has nearly constant periods and amplitudes, while the rough one wanders in both. Jitter measures the wander in the period, shimmer the wander in the amplitude.
```

## What you already know

- A voiced frame repeats every $T_0 = 1/F_0$ seconds, and [Pitch tracking](/lessons/018-pitch-tracking) estimates $F_0$ per frame and marks unvoiced frames with 0.
- A classifier turns a feature vector into class scores, and a clip can be scored per class, as in [Audio classification and CLAP](/lessons/072-audio-classification-and-clap).
- Mean, variance, and correlation summarize one or two sequences of numbers.
- A confusion matrix counts how often each true class gets each predicted label.
- A semitone is a frequency ratio of $2^{1/12}$, so 12 semitones make an octave.

## What a voice carries besides words

*Prosody* is the melody, loudness, and timing of speech: $F_0$, energy, and speaking rate. *Voice quality* is how regular the vocal folds are, because a tense or breathy voice has uneven cycles. 
Eyben and colleagues proposed a small, fixed set of such descriptors for voice research and affect, the Geneva Minimalistic Acoustic Parameter Set (GeMAPS, with 62 features, and 88 in the extended eGeMAPS). A fixed set makes results from different labs comparable. This chapter builds six relatives of those features, an "eGeMAPS-lite": mean $F_0$, $F_0$ range, jitter, shimmer, energy, and speaking rate.

Anger and joy tend to raise $F_0$, widen its range, and raise energy and rate. Sadness tends to lower and flatten all of them. A voice that shakes with strain adds jitter and shimmer. None of these rules holds for every speaker, which the "Where this breaks" section returns to.

## Jitter and shimmer

Mark each glottal cycle in a voiced stretch. Cycle $i$ has a period $T_i$ and a peak amplitude $A_i$. In a perfectly steady voice, all $T_i$ are equal. *Jitter* is how much the period changes from one cycle to the next, and *shimmer* is the same for the amplitude.

The obvious measure is the average change, $\frac{1}{N-1}\sum_i |T_i - T_{i+1}|$. It carries the unit of the period, so a 100 Hz voice and a 200 Hz voice with the same relative wobble get different numbers. Divide by the mean period and the unit cancels:

$$
\text{jitter} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1} |T_i - T_{i+1}|}{\frac{1}{N}\sum_{i=1}^{N} T_i}
$$

Read it aloud: the average step between neighboring cycles, as a fraction of the average cycle. This is the *local* version. The numerator averages $N-1$ pairs and the denominator averages all $N$ cycles. Shimmer is the same formula with $A_i$ in place of $T_i$.

Work one example. Four periods, in samples at 16 kHz: $100, 102, 98, 101$.

1. The steps are 2, 4, and 3, with mean 3.
2. The mean of the four periods is $401/4 = 100.25$.
3. Jitter is $3 / 100.25 = 0.0299$, or 2.99%.

Dividing by the mean of the first three periods, 100, gives 3.00%. That is close, but it normalizes by a different set of cycles, and the checks catch it. For amplitudes $0.50, 0.55, 0.45, 0.52$, the steps are $0.05, 0.10, 0.07$, the mean step is $0.0733$, the mean amplitude is $0.505$, and shimmer is $0.145$.

```python
steps = np.abs(np.diff(periods))
jitter = steps.mean() / periods.mean()   # mean of N-1 steps over the mean of N periods
```

Healthy sustained vowels typically sit around or below one percent jitter and a few percent shimmer, and Figure 1's rough voice is far beyond that on purpose. A tracker that jumps an octave for one cycle produces a step of 50% to 100% of a period, so jitter inherits every error of the tracker from the pitch chapter.

> [!KEY]
> Jitter and shimmer are relative: the average step between neighboring cycles over the average cycle. A low voice and a high voice, or a loud and a quiet recording, become comparable.

## The other descriptors

$F_0$ comes from a tracker as one value per frame, with 0 for unvoiced frames. Those zeros are not 0 Hz, so every statistic must skip them. For the track $[0, 100, 100, 200, 200, 0, 150, 150, 0, 0]$ Hz, the voiced values are $100, 100, 150, 150, 200, 200$. Their mean is 150 Hz, and including the zeros would give 90 Hz.

Report the range in semitones, because an octave is the same musical distance at 100 Hz and at 300 Hz. A robust choice is the 20th to 80th percentile range, which ignores a few tracker outliers: $12 \log_2(p_{80}/p_{20})$. Here $p_{20} = 100$ and $p_{80} = 200$, so the range is 12 semitones, an octave. *Energy* is the mean frame RMS in decibels, and *speaking rate* is syllables per second, from a transcript with an alignment or from peaks in the energy envelope.

## Classify, and score by UAR

With six numbers per utterance, a linear classifier is a fair first model. Scoring it is subtle, because emotion sets are imbalanced: neutral speech is common and some emotions are rare.

Use ridge regression onto one-hot labels, which has a closed form. With $X_b$ as the features plus a column of ones, $Y$ the one-hot labels, and per-clip weights $w_i$ on a diagonal matrix $D$:

$$
W = \left(X_b^\top D X_b + \lambda P\right)^{-1} X_b^\top D Y
$$

Read it aloud: solve least squares for every class at once, count clip $i$ with weight $w_i$, and penalize every weight except the bias, which is what $P$ (the identity with a 0 for the bias) does. The prediction is the class with the largest score in $X_b W$. Setting $w_i = 1$ treats every clip equally. Setting $w_i = N / (K\, n_{c(i)})$, with $n_{c(i)}$ the size of clip $i$'s class and $K$ the number of classes, gives every class the same total weight.

Next, the score. *Accuracy* is the share of clips labeled correctly. Suppose a test set has 80 neutral, 15 angry, and 5 sad clips. A model that always says "neutral" scores 80% accuracy and detects no emotion at all. *Unweighted average recall* (UAR) fixes this. The recall of a class is the share of its clips the model found, and UAR averages the recalls with equal weight:

$$
\text{UAR} = \frac{1}{K} \sum_{k=1}^{K} \frac{C_{kk}}{\sum_j C_{kj}}
$$

Read it aloud: for each true class (a row of the confusion matrix $C$), divide the diagonal count by the row total, then average. The always-neutral model gets recalls $1, 0, 0$ and UAR $1/3$, which is chance. The Interspeech computational paralinguistics challenges report UAR for this reason. Emotion benchmarks such as IEMOCAP, MSP-Podcast, and the emotion task of SUPERB raise the same imbalance question.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Confusion matrix of 100 clips with recalls 0.875, 0.533, and 0.600; accuracy 0.81 and unweighted average recall 0.669">
  <text x="243.0" y="22" class="fig-small" text-anchor="middle">predicted</text>
  <text x="40" y="143.0" class="fig-small" text-anchor="middle" transform="rotate(-90 40 143.0)">true</text>
  <text x="181.0" y="42" class="fig-small" text-anchor="middle">neutral</text>
  <text x="243.0" y="42" class="fig-small" text-anchor="middle">angry</text>
  <text x="305.0" y="42" class="fig-small" text-anchor="middle">sad</text>
  <text x="142" y="85.0" class="fig-small" text-anchor="end">neutral</text>
  <rect x="150" y="50" width="62" height="62" class="fill-soft-b stroke-grid"/>
  <text x="181.0" y="86.0" text-anchor="middle" class="fig-bold">70</text>
  <rect x="212" y="50" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="86.0" text-anchor="middle" class="">8</text>
  <rect x="274" y="50" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="86.0" text-anchor="middle" class="">2</text>
  <text x="352" y="85.0" class="fig-small">recall 0.875 (80 clips)</text>
  <text x="142" y="147.0" class="fig-small" text-anchor="end">angry</text>
  <rect x="150" y="112" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="181.0" y="148.0" text-anchor="middle" class="">5</text>
  <rect x="212" y="112" width="62" height="62" class="fill-soft-b stroke-grid"/>
  <text x="243.0" y="148.0" text-anchor="middle" class="fig-bold">8</text>
  <rect x="274" y="112" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="305.0" y="148.0" text-anchor="middle" class="">2</text>
  <text x="352" y="147.0" class="fig-small">recall 0.533 (15 clips)</text>
  <text x="142" y="209.0" class="fig-small" text-anchor="end">sad</text>
  <rect x="150" y="174" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="181.0" y="210.0" text-anchor="middle" class="">1</text>
  <rect x="212" y="174" width="62" height="62" class="fill-sunk stroke-grid"/>
  <text x="243.0" y="210.0" text-anchor="middle" class="">1</text>
  <rect x="274" y="174" width="62" height="62" class="fill-soft-b stroke-grid"/>
  <text x="305.0" y="210.0" text-anchor="middle" class="fig-bold">3</text>
  <text x="352" y="209.0" class="fig-small">recall 0.600 (5 clips)</text>
  <text x="150" y="266" class="fig-small fig-bold">accuracy = 81 / 100 = 0.81</text>
  <text x="150" y="286" class="fig-small fig-bold">UAR = (0.875 + 0.533 + 0.600) / 3 = 0.669</text>
</svg>
Figure 2. A confusion matrix of 100 clips, with each row a true class. Notice that the 70 correct neutral clips dominate accuracy, 0.81, while UAR, 0.669, counts the weak angry and sad recalls as much as the strong neutral one.
```

In Figure 2, the recalls are $70/80 = 0.875$, $8/15 = 0.533$, and $3/5 = 0.600$, and their mean is $0.669$. Normalizing by column sums would answer a different question, precision, so the row direction matters.

Balanced weights trade accuracy for recall. On a synthetic set with 300 neutral, 60 angry, and 20 sad utterances, the unweighted model scores accuracy 0.937 and UAR 0.676, because it almost never says "sad". With balanced weights, accuracy falls to 0.771 and UAR rises to 0.890: the model finds every sad clip, at the price of calling some neutral clips emotional.

> [!KEY]
> When classes are imbalanced, report UAR, and train with balanced weights if UAR is the goal. A high accuracy can come entirely from the majority class.

## Labels are noisy

Emotion labels come from people, and people disagree, so measure the labels before blaming the model. For categories, use *Cohen's kappa*. Two annotators label the same 50 clips as "calm" or "tense", and the table counts their pairs, rows for annotator A and columns for B: $\begin{pmatrix}20 & 5\\10 & 15\end{pmatrix}$.

1. They agree on $20 + 15 = 35$ clips, so the observed agreement is $p_o = 35/50 = 0.70$.
2. Annotator A says "calm" 25 times and B says it 30 times. If both labeled at random with those habits, they would agree on "calm" with probability $0.5 \times 0.6$ and on "tense" with $0.5 \times 0.4$, so the chance agreement is $p_e = 0.30 + 0.20 = 0.50$.
3. Kappa is the agreement beyond chance, as a share of the room above chance: $\kappa = (p_o - p_e)/(1 - p_e) = 0.2/0.5 = 0.4$.

Raw agreement misleads when one label dominates. If both annotators say "neutral" for 90% of clips and agree on 82% of all clips, then $p_e = 0.9 \times 0.9 + 0.1 \times 0.1 = 0.82$, so $\kappa = 0$: chance alone gives that agreement.

For *dimensional* labels, annotators rate each clip on a scale, usually arousal (calm to excited) and valence (negative to positive). Averaging a few raters gives a gold value, and the model predicts a number. Pearson correlation ignores a constant offset and a wrong scale, so a model that is always 1 too high, or twice too large, still scores 1. Lin's *concordance correlation coefficient* (CCC) charges for both:

$$
\text{CCC} = \frac{2\,\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}
$$

Read it aloud: twice the covariance, over the two variances plus the squared gap between the means. It equals 1 only when the points fall on the line $y = x$. Figure 3 uses gold values $-1, 0, 1$, with $\mu = 0$ and $\sigma^2 = 2/3$. A prediction shifted by 1 has $\sigma_{xy} = 2/3$, so $\text{CCC} = (4/3)/(2/3 + 2/3 + 1) = 4/7 = 0.571$. A prediction doubled has $\sigma_y^2 = 8/3$ and $\sigma_{xy} = 4/3$, so $\text{CCC} = (8/3)/(10/3) = 0.8$. Pearson gives 1 for both.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Three predictions plotted against the gold values -1, 0, 1: a perfect one, one shifted by 1, and one doubled, with CCC 1, 0.571, and 0.8, all with Pearson 1">
  <line x1="170" y1="140" x2="430" y2="140" class="stroke-grid"/><line x1="300" y1="15" x2="300" y2="265" class="stroke-grid"/>
  <text x="430" y="158" class="fig-small" text-anchor="end">gold</text><text x="308" y="28" class="fig-small">predicted</text>
  <polyline points="250.0,190.0 300.0,140.0 350.0,90.0" fill="none" class="stroke-ink"/>
  <circle cx="250.0" cy="190.0" r="4" class="fill-ink"/>
  <circle cx="300.0" cy="140.0" r="4" class="fill-ink"/>
  <circle cx="350.0" cy="90.0" r="4" class="fill-ink"/>
  <line x1="20" y1="36" x2="44" y2="36" class="stroke-ink"/>
  <text x="52" y="40" class="fig-small">perfect, y = x: Pearson 1.00, CCC 1.000</text>
  <polyline points="250.0,140.0 300.0,90.0 350.0,40.0" fill="none" class="stroke-b"/>
  <circle cx="250.0" cy="140.0" r="4" class="fill-b"/>
  <circle cx="300.0" cy="90.0" r="4" class="fill-b"/>
  <circle cx="350.0" cy="40.0" r="4" class="fill-b"/>
  <line x1="20" y1="58" x2="44" y2="58" class="stroke-b"/>
  <text x="52" y="62" class="fig-small">shifted, y = x + 1: Pearson 1.00, CCC 0.571</text>
  <polyline points="250.0,240.0 300.0,140.0 350.0,40.0" fill="none" class="stroke-a"/>
  <circle cx="250.0" cy="240.0" r="4" class="fill-a"/>
  <circle cx="300.0" cy="140.0" r="4" class="fill-a"/>
  <circle cx="350.0" cy="40.0" r="4" class="fill-a"/>
  <line x1="20" y1="80" x2="44" y2="80" class="stroke-a"/>
  <text x="52" y="84" class="fig-small">doubled, y = 2x: Pearson 1.00, CCC 0.800</text>
</svg>
Figure 3. One gold sequence and three predictions. Notice that all three lines are perfectly correlated with the gold values, but only the line on y = x has a CCC of 1.
```

Use one convention for every moment. `np.cov` divides by $n-1$ and `np.var` divides by $n$, so combining them gives a CCC above 1 for identical inputs.

> [!WARNING]
> A model that tracks arousal but is compressed and offset has a high Pearson and a low CCC. The CCC is the one that says the numbers cannot be used as they are.

Annotator disagreement sets a ceiling. When humans reach only moderate kappa, a model that matches the majority label is about as good as the labels allow. A majority vote hides the disagreement, so some systems keep the vote shares as soft labels. Acoustics also carries arousal better than valence, so valence scores are usually lower.

## From measurement to TTS control

The same descriptors are the controls of expressive synthesis. FastSpeech 2, from [Durations and alignment search](/lessons/052-durations-and-alignment-search), predicts a duration, a pitch value, and an energy value per phone. A rate change scales the durations, a pitch change shifts the $F_0$ contour, and a loudness change scales the energy.

To make "fine" sound more agitated, shift the contour up by 2 semitones, which multiplies every $F_0$ by $2^{2/12} = 1.1225$, so 200 Hz becomes 224.5 Hz. Then widen the range by moving each point away from the contour mean by a factor of 1.5 in the semitone domain. A recognizer from this chapter can then check that the output has the arousal you intended.

## Where this breaks

**Speakers differ more than emotions.** An angry man can still have a lower $F_0$ than a calm woman. Normalize each descriptor per speaker, by that speaker's neutral speech, before you compare across speakers.

**Acted data is not natural data.** IEMOCAP has actors in scripted and improvised scenes, and MSP-Podcast has natural podcast speech. A model trained on one often drops on the other. Features from [Self-supervised speech](/lessons/039-self-supervised-speech) replace the six descriptors in most systems, and UAR and agreement stay the same.

**Short stretches are noisy.** Jitter from 20 cycles has a wide spread, and recording noise raises it.

> [!RECAP]
> - Prosody (F0, energy, rate) and voice quality (jitter, shimmer) describe how something is said.
> - Local jitter and shimmer are the mean step between neighbors over the mean value, so they do not depend on pitch or loudness.
> - F0 statistics use voiced frames only, and the range is best in semitones.
> - UAR averages per-class recall from the confusion matrix, and balanced weights raise it on imbalanced data.
> - Cohen's kappa removes chance agreement for categories, and CCC penalizes bias and scale for dimensional labels.

## Build it step by step

The checks test the functions in this order.

1. **`jitter(periods)`** and **`shimmer(amplitudes)`** return the mean absolute neighbor difference over the mean of all values. The usual mistake is normalizing by the mean of only the first $N-1$ values.
2. **`f0_stats(f0)`** drops zeros, then returns the mean in hertz and the 20th to 80th percentile range in semitones. The usual mistake is averaging the zeros in.
3. **`ccc(pred, gold)`** applies the formula with population moments. The usual mistakes are returning `np.corrcoef`, which ignores bias and scale, and mixing `np.cov` with `np.var`.
4. **`cohen_kappa(table)`** takes the trace over the total as $p_o$, and sums products of row and column shares as $p_e$.
5. **`confusion_matrix(y_true, y_pred, n_classes)`** counts with `np.add.at`, rows true.
6. **`accuracy(confusion)`** and **`uar(confusion)`** divide by the total and by row sums. The usual mistake is dividing by column sums.
7. **`fit_linear(X, y, n_classes, l2, balanced)`** builds the weighted normal equations and calls `np.linalg.solve`. Leave the bias out of the penalty.
8. **`predict_linear(W, X)`** appends the ones column and takes `argmax`.

## The exercise

You build `jitter`, `shimmer`, `f0_stats`, `ccc`, `cohen_kappa`, `confusion_matrix`, `accuracy`, `uar`, `fit_linear`, and `predict_linear`. Press **Run** to hear a steady and a rough synthetic voice, train the classifier on an imbalanced three-emotion set with and without balanced weights, and compare Pearson with CCC for a compressed and offset arousal predictor.

The checks verify the hand examples and planted perturbations of known size, such as periods that alternate 1% above and below a base. They require jitter to ignore a rescaling of the periods, CCC to fall below Pearson under a shift or a compression, kappa to be 0 at 82% raw agreement, and UAR to separate a majority-class guesser from a real model. They compare `fit_linear` with a least-squares solve on rescaled rows, and require balanced weights to win on UAR.

## Further reading

- Eyben et al., "The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," IEEE Transactions on Affective Computing, 2016.
- Busso et al., "IEMOCAP: Interactive emotional dyadic motion capture database," Language Resources and Evaluation, 2008.
- Lin, "A concordance correlation coefficient to evaluate reproducibility," Biometrics, 1989.
- Cohen, "A coefficient of agreement for nominal scales," Educational and Psychological Measurement, 1960.
- Schuller et al., the Interspeech computational paralinguistics challenge overview papers.
- Yang et al., "SUPERB: Speech processing Universal PERformance Benchmark," Interspeech 2021.
