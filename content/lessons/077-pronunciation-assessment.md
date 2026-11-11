---
title: Pronunciation assessment
slug: 077-pronunciation-assessment
part: "Part 8 — Beyond recognition and synthesis"
summary: Score each phone with goodness of pronunciation, flag mispronunciations with a threshold, and check the scores against human ratings.
minutes: 40
gpu: false
objectives:
  - Compute goodness of pronunciation for each aligned phone as a per-frame mean of log posteriors, and explain why you subtract the best phone.
  - Flag mispronounced phones with a threshold, and report precision, recall, and F1.
  - Correlate scores with human ratings by Pearson and by Spearman with tie-averaged ranks.
  - Build a phone confusion matrix from aligned canonical and recognized phones.
prereqs:
  - 024-hybrid-acoustic-models
  - 046-forced-alignment-and-word-timestamps
exercise: 077-pronunciation-assessment
---

# Pronunciation assessment

A learner reads "cat" into a language-learning app and says "cas". A recognizer with a strong language model may still write "cat", because the context and the vocabulary pull it there. For a transcript that is a feature. For a teacher it is the wrong answer: the learner needs to hear that the last sound was off, and which sound it was.

Pronunciation assessment asks how well each sound of a known word was said. The text is given, so the answer is a score per phone. The score has to point at a bad phone, and it has to rank whole utterances the way a human teacher would.

You already built the machinery. [Forced alignment](/lessons/046-forced-alignment-and-word-timestamps) tells you which frames belong to each phone, and an acoustic model gives a posterior over phones at each frame. This chapter adds the score, then the tools to judge it: a threshold, precision and recall, correlation with human ratings, and a confusion matrix.

## What you already know

- A [hybrid acoustic model](/lessons/024-hybrid-acoustic-models) gives a posterior probability over phones or states at each frame.
- [Forced alignment](/lessons/046-forced-alignment-and-word-timestamps) pins a known transcript to frames, and a segment is a start frame plus an end frame that is one past the last frame.
- A phone is one speech sound, and the canonical phone is the one the dictionary pronunciation expects at that place.
- Precision is the share of flagged items that are right, and recall is the share of true items that got flagged.
- A correlation is a number between -1 and 1 that says how closely two lists of scores move together.

## Goodness of pronunciation

Picture the frames of one aligned phone. At each frame the model says how probable each phone is. If the learner said the canonical phone, the model puts most of its probability on it. If the learner said another sound, the model puts it elsewhere, and the canonical phone's probability drops.

The first score is therefore the average of that probability, in the log domain, over the phone's frames. This is the **log posterior** (LPP) score. For a segment of $N$ frames with canonical phone $q$, where $y_t(q)$ is the model's posterior for $q$ at frame $t$:

$$
\mathrm{LPP}(q) = \frac{1}{N} \sum_{t=1}^{N} \log y_t(q)
$$

Read it aloud: take the log probability that the model gave the canonical phone at every frame of the segment, and average. In numpy it is `np.log(post[start:end, phone]).mean()`. The unit is nats per frame, because the log is natural.

### Why not just use the posterior?

LPP has a problem. Some phones are hard for a model even when said perfectly: a short, reduced vowel may never get a posterior above 0.5. Such a phone scores badly for every speaker, and the score blames the learner for the model's uncertainty.

The fix is to compare the canonical phone to the best competitor at each frame. This is **goodness of pronunciation** (GOP). Witt and Young (2000) defined it as a log posterior score normalized by the number of frames, and with frame posteriors from a neural network (the setting of Hu and others, 2015) it takes this form:

$$
\mathrm{GOP}(q) = \frac{1}{N} \sum_{t=1}^{N} \left[ \log y_t(q) - \log \max_{p} y_t(p) \right]
$$

Read it aloud: at every frame, compare the log probability of the canonical phone with the log probability of the model's favorite phone, and average the gap over the segment. The gap is 0 when the canonical phone is the favorite and negative otherwise, so GOP is at most 0, and a hard phone said correctly still scores 0. In numpy, with `block = np.log(post[start:end])`, it is `(block[:, phone] - block.max(axis=1)).mean()`.

A third variant skips the log. The **posterior ratio** is the mean of $y_t(q) / \max_p y_t(p)$, which lies between 0 and 1.

### Work one word by hand

Take the word "cat" with canonical phones k, ae, t and a four-phone inventory of k, ae, t, and s. The learner says "cas". Seven frames of 10 ms are aligned as k on frames 0-1, ae on frames 2-4, and t on frames 5-6. Figure 1 shows the posterior of the canonical phone beside the best posterior at each frame.

```figure
<svg viewBox="0 0 560 320" role="img" aria-label="Per-frame posterior of the canonical phone and of the best phone over seven frames, grouped into k, ae, and t, with GOP 0.00, 0.00, and -0.88">
<line x1="64" y1="230" x2="518" y2="230" class="stroke-muted"/>
<line x1="64" y1="60" x2="64" y2="230" class="stroke-muted"/>
<text x="58" y="234" text-anchor="end" class="fig-small">0</text>
<text x="58" y="149.0" text-anchor="end" class="fig-small">0.5</text>
<text x="58" y="64.0" text-anchor="end" class="fig-small">1</text>
<text x="16" y="145.0" class="fig-small" transform="rotate(-90 16 145.0)" text-anchor="middle">posterior</text>
<rect x="76" y="94.0" width="22" height="136.0" class="fill-a"/>
<rect x="102" y="94.0" width="22" height="136.0" class="fill-b"/>
<text x="87" y="90.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="113" y="90.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="100" y="246" text-anchor="middle" class="fig-small">0</text>
<rect x="140" y="111.0" width="22" height="119.0" class="fill-a"/>
<rect x="166" y="111.0" width="22" height="119.0" class="fill-b"/>
<text x="151" y="107.0" text-anchor="middle" class="fig-small">0.7</text>
<text x="177" y="107.0" text-anchor="middle" class="fig-small">0.7</text>
<text x="164" y="246" text-anchor="middle" class="fig-small">1</text>
<rect x="204" y="94.0" width="22" height="136.0" class="fill-a"/>
<rect x="230" y="94.0" width="22" height="136.0" class="fill-b"/>
<text x="215" y="90.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="241" y="90.0" text-anchor="middle" class="fig-small">0.8</text>
<text x="228" y="246" text-anchor="middle" class="fig-small">2</text>
<rect x="268" y="111.0" width="22" height="119.0" class="fill-a"/>
<rect x="294" y="111.0" width="22" height="119.0" class="fill-b"/>
<text x="279" y="107.0" text-anchor="middle" class="fig-small">0.7</text>
<text x="305" y="107.0" text-anchor="middle" class="fig-small">0.7</text>
<text x="292" y="246" text-anchor="middle" class="fig-small">3</text>
<rect x="332" y="77.0" width="22" height="153.0" class="fill-a"/>
<rect x="358" y="77.0" width="22" height="153.0" class="fill-b"/>
<text x="343" y="73.0" text-anchor="middle" class="fig-small">0.9</text>
<text x="369" y="73.0" text-anchor="middle" class="fig-small">0.9</text>
<text x="356" y="246" text-anchor="middle" class="fig-small">4</text>
<rect x="396" y="196.0" width="22" height="34.0" class="fill-a"/>
<rect x="422" y="111.0" width="22" height="119.0" class="fill-b"/>
<text x="407" y="192.0" text-anchor="middle" class="fig-small">0.2</text>
<text x="433" y="107.0" text-anchor="middle" class="fig-small">0.7</text>
<text x="420" y="246" text-anchor="middle" class="fig-small">5</text>
<rect x="460" y="179.0" width="22" height="51.0" class="fill-a"/>
<rect x="486" y="145.0" width="22" height="85.0" class="fill-b"/>
<text x="471" y="175.0" text-anchor="middle" class="fig-small">0.3</text>
<text x="497" y="141.0" text-anchor="middle" class="fig-small">0.5</text>
<text x="484" y="246" text-anchor="middle" class="fig-small">6</text>
<text x="294.0" y="264" text-anchor="middle" class="fig-small">frame (1 frame = 10 ms)</text>
<line x1="72" y1="274" x2="196" y2="274" class="stroke-ink"/>
<text x="134.0" y="292" text-anchor="middle" class="fig-bold">k: GOP 0.00</text>
<line x1="200" y1="274" x2="388" y2="274" class="stroke-ink"/>
<text x="294.0" y="292" text-anchor="middle" class="fig-bold">ae: GOP 0.00</text>
<line x1="392" y1="274" x2="516" y2="274" class="stroke-ink"/>
<text x="454.0" y="292" text-anchor="middle" class="fig-bold">t: GOP -0.88</text>
<rect x="70" y="14" width="12" height="12" class="fill-a"/><text x="88" y="25" class="fig-small">canonical phone</text>
<rect x="220" y="14" width="12" height="12" class="fill-b"/><text x="238" y="25" class="fig-small">best phone</text>
</svg>
Figure 1. The canonical phone and the best phone agree on k and ae, so their gaps are zero. On the last two frames, s beats t by a wide margin, so the gap is large and GOP for t is -0.88.
```

Work the three segments:

- k has canonical posteriors 0.80 and 0.70, and they are also the best. The log gaps are 0 and 0, so GOP is 0. LPP is $(\ln 0.8 + \ln 0.7)/2 = -0.2899$.
- ae gets the same treatment over three frames. GOP is 0, and LPP is -0.2284.
- t has canonical posteriors 0.20 and 0.30 against best posteriors 0.70 (that is s) and 0.50. The gaps are $\ln(0.2/0.7) = -1.2528$ and $\ln(0.3/0.5) = -0.5108$. Their mean is -0.8818. LPP is -1.4067.

The posterior ratio for t is $(0.2/0.7 + 0.3/0.5)/2 = 0.4429$. The planted error shows up in all three scores, and GOP is the one that leaves the two correct phones at exactly 0.

### Average, do not add

The sum of the per-frame gaps for t is -1.7636, twice the mean. A longer phone collects more frames, so a sum punishes slow speech even when each frame is as good. Stretch the same two-frame pattern to six frames and the sum becomes -5.2908, while the mean stays at -0.8818.

> [!KEY]
> GOP is a per-frame average of how far the canonical phone's log posterior sits below the best phone's. It is 0 for a clean phone and negative for a bad one, and the average makes it independent of how long the phone lasts.

## From scores to decisions

The app needs to say "check your t", not show negative numbers. The simplest rule is a threshold: flag a phone as mispronounced when its GOP is below $\tau$. In code, `gop_scores < threshold`. The comparison is strict, so a phone that sits exactly on the threshold is accepted.

You judge the rule against labels from human annotators. Treat a flagged phone as a positive. The counts are true positives (TP, flagged and truly wrong), false positives (FP, flagged but fine), and false negatives (FN, wrong but not flagged):

$$
P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}
$$

Eight phones show the trade-off. Their GOP scores are $-0.1, -0.4, -2.0, -0.3, -0.8, -1.5, -0.2, -2.5$, and annotators marked phones 1, 2, 4, and 7 (counting from 0) as wrong. At $\tau = -1.0$ the rule flags phones 2, 5, and 7. Two are right, so $TP = 2$, $FP = 1$, and $FN = 2$. That gives $P = 0.667$, $R = 0.5$, and $F_1 = 0.571$.

Raise the threshold to $\tau = -0.35$ and the rule flags phones 1, 2, 4, 5, and 7. Now $TP = 4$, $FP = 1$, $FN = 0$, so $P = 0.8$, $R = 1.0$, and $F_1 = 0.889$. A higher threshold catches more errors at the price of more false alarms.

> [!TIP]
> A language-learning app usually prefers precision, because a false alarm tells a correct learner they are wrong. Pick the threshold on labeled data for the precision you want, and check it per phone, because some phones need their own threshold.

## Agreeing with human raters

The second use of GOP is a score for a whole utterance. Average the phone GOPs and you get one number, which you compare with ratings from human experts. The speechocean762 corpus (Zhang and others, 2021) has 5,000 utterances from 250 non-native English speakers, half of them children, and five experts scored each one at the sentence, word, and phoneme levels.

**Pearson correlation** asks whether the two lists move together in a straight line. Center each list by subtracting its mean, then divide the dot product by the product of the lengths:

$$
r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \, \sum_i (y_i - \bar{y})^2}}
$$

Rescaling or shifting either list leaves $r$ unchanged, so nats per frame and a 0-to-10 rating are comparable. Take five utterances with model scores $-1.2, -0.9, -0.6, -0.5, -0.1$ and human ratings $3, 5, 5, 8, 9$. Pearson gives $r = 0.9324$.

Ratings are not measurements on a ruler, though: a step from 5 to 8 need not match a step from 8 to 11. **Spearman correlation** replaces each value by its rank and takes the Pearson correlation of the ranks. It asks only whether the order agrees.

Ties need care. The two utterances rated 5 share ranks 2 and 3, so each gets the average, 2.5. The ratings become ranks $1, 2.5, 2.5, 4, 5$, and the model scores become $1, 2, 3, 4, 5$. Their Pearson correlation is $0.9747$. If you break the tie by position, the ranks are $1, 2, 3, 4, 5$ for both lists, and the correlation is exactly 1.0, which claims that the model orders the utterances perfectly when the humans did not.

> [!WARNING]
> A tie in the human ratings is common, because raters use a coarse scale. Rank with ties averaged, or Spearman overstates the agreement.

## Which phones get confused

A score says how wrong. A confusion matrix says in what way. Run a phone recognizer on the learner's audio and align its phone string with the canonical string, with the edit-distance alignment from [word error rate](/lessons/026-word-error-rate) applied to phones. Each aligned pair is one canonical phone and the phone that the recognizer heard.

Count the pairs into a matrix $C$ with canonical phones on the rows and recognized phones on the columns, so $C[i, j]$ is how often phone $i$ came out as phone $j$. Row sums equal the canonical counts, and the diagonal holds correct phones. With the four-phone inventory, the canonical string t, t, k, ae, t, s and the recognized string s, t, k, ae, s, s give a row for t of $[0, 0, 1, 2]$: one t stayed t, and two became s. In numpy that is `np.add.at(C, (canonical, recognized), 1)`. A large off-diagonal cell is a teachable pattern, such as a learner whose first language has no th replacing it with s.

> [!RECAP]
> - GOP averages, over a phone's frames, the log posterior of the canonical phone minus that of the best phone.
> - It is at most 0, and dividing by the frame count makes long and short phones comparable.
> - A threshold turns GOP into a flag, and precision, recall, and F1 measure the flags against human labels.
> - Pearson measures a straight-line fit, and Spearman measures rank agreement with ties averaged.
> - A confusion matrix counts aligned canonical and recognized phones, canonical on the rows.

## Build it step by step

The checks test the functions in this order. Every `alignment` is an `(S, 3)` integer array of `[phone, start, end]`, and `posteriors` is `(T, P)`.

1. **`gop_log_posterior(posteriors, alignment)`** returns `(S,)`. Clip the posteriors at 1e-10, take logs, and average `block[:, phone]` over each segment. The common mistake is to treat `end` as inclusive and drop or add a frame.
2. **`gop(posteriors, alignment)`** returns `(S,)`. Subtract `block.max(axis=1)` from the canonical column before averaging. The common mistake is a sum in place of a mean, so long phones look worse.
3. **`detect_mispronunciations(gop_scores, threshold)`** returns a boolean array that is true where the score is below the threshold.
4. **`precision_recall_f1(predicted, gold)`** returns three floats. Count TP, FP, and FN, and return 0.0 for a ratio with a zero denominator. The common mistake is a division by zero when nothing is flagged.
5. **`average_ranks(x)`** returns float ranks from 1. Sort with `np.argsort`, assign positions, then replace each group of equal values by its mean rank.
6. **`pearson(x, y)`** returns a float. Center both arrays and use the formula above.
7. **`spearman(x, y)`** returns a float. It is `pearson` on `average_ranks` of each list. The common mistake is ranking by `argsort` of `argsort`, which orders ties by position.
8. **`phone_confusion(canonical, recognized, n_phones)`** returns an `(n_phones, n_phones)` integer matrix. Use `np.add.at` so repeated pairs all count.

## Where this breaks

**The alignment assumes the canonical phones.** Forced alignment forces the canonical phones onto the frames, so for a phone that was replaced, the segment boundaries are a guess. GOP over bad boundaries mixes frames of the neighbor into the score.

**Accents are not errors.** A model trained mostly on native speech scores a regional accent as wrong. Training on data that matches the learners helps more than moving the threshold.

**Human labels disagree.** Five experts do not always give the same score to one phone, which caps how well a model can correlate with one rater. Compare the model to the average of the raters, and report how well the raters agree with each other.

**A global threshold hides differences.** Each phone has its own distribution of GOP when said correctly, so one threshold over-flags some phones and under-flags others. Per-phone thresholds, or a classifier trained on GOP features from the human labels, fix most of that.

## The exercise

You build `gop_log_posterior`, `gop`, `detect_mispronunciations`, `precision_recall_f1`, `average_ranks`, `pearson`, `spearman`, and `phone_confusion`.

The checks confirm the log posterior and GOP on the seven-frame example above, including the planted t that scores -0.8818, and that a stretched phone keeps its GOP. They confirm precision, recall, and F1 at both thresholds, ranks with ties, the Pearson value 0.9324, the Spearman value 0.9747 that tie-averaged ranks give, and the confusion counts. The **Run** plants one replaced phone among five, prints the GOP per phone and the detection scores, and draws a GOP plot and a confusion heatmap.

## Further reading

- Witt and Young, "Phone-level pronunciation scoring and assessment for interactive language learning," *Speech Communication*, 2000. The paper that defines goodness of pronunciation.
- Hu, Qian, Soong, and Wang, "Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers," *Speech Communication*, 2015. GOP-style features from neural network posteriors.
- Zhang and others, "speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment," Interspeech 2021 (https://arxiv.org/abs/2104.01378). The corpus with sentence, word, and phoneme scores from five experts.
