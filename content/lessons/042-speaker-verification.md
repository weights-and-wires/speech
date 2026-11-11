---
title: Speaker embeddings and verification
slug: 042-speaker-verification
part: "Part 4 — Neural speech recognition"
summary: Pool frames into embeddings, train them with an angular margin, score trials with AS-norm and multi-clip enrollment, and report EER and minDCF.
minutes: 60
gpu: false
objectives:
  - Implement the additive angular margin softmax, including its fallback past pi minus the margin, and explain why the margin tightens clusters.
  - Pool frame features with statistics pooling and attentive statistics pooling, and say what ECAPA-TDNN adds to an x-vector.
  - Score a pair of clips by the cosine of their embeddings, and split trial scores into targets and non-targets.
  - Enroll a speaker from several clips, and normalize a trial score with AS-norm over the top-K cohort scores.
  - Compute the equal error rate from trial scores, and explain what it trades.
  - Compute the normalized minimum detection cost, and explain why it depends on the prior of a target trial.
prereqs:
  - 013-log-mel-spectrograms
  - 039-self-supervised-speech
exercise: 042-speaker-verification
---

# Speaker embeddings and verification

A recognizer asks "what was said?" A bank's phone line asks "is this the account holder?", from a few seconds of speech.

A classifier with one output per customer fails the day a new customer signs up. The standard answer maps any stretch of speech to a fixed-size vector, a **speaker embedding**, so that clips of one person land close together and clips of different people land far apart. A new customer needs only one enrollment clip.

That hides a puzzle: you train the embedding as a classifier over known speakers, but you use it on speakers it has never seen. This chapter builds the pooling, the loss that solves the puzzle, and the back end and two numbers that score the result.

## What you already know

- Cross-entropy over a softmax trains a classifier.
- Cosine similarity compares directions: the cosine of the angle between two vectors, from the same chapter.
- A log-mel spectrogram turns a clip into frames of features (see [Log-mel spectrograms](/lessons/013-log-mel-spectrograms)).
- A threshold turns a score into a yes or no, and a wrong answer can be wrong in two ways.
- $\cos(\theta + m) = \cos\theta\cos m - \sin\theta\sin m$.

## From a voice to a vector

A clip has a different number of frames every time, so the network needs a step that turns $T$ frames into one fixed-size vector. That step is **pooling**.

The **x-vector** (Snyder et al., 2018) has three parts. Frame-level layers, dilated 1-D convolutions called a TDNN, read each frame with a few frames of context. A **statistics pooling** layer takes, for every channel, the mean and standard deviation over all $T$ frames and concatenates them, so $C$ channels give $2C$ numbers. Segment-level dense layers follow, and the activation of one of them is the embedding.

Statistics pooling weights every frame equally, silence included. **ECAPA-TDNN** (Desplanques et al., 2020) adds squeeze-and-excitation blocks, which rescale each channel using a summary of the whole clip, and **attentive statistics pooling**: a softmax over the $T$ frames turns per-frame scores into weights $\alpha_t$ that sum to 1. The pooled statistics are

$$
\tilde\mu = \sum_t \alpha_t h_t, \qquad \tilde\sigma = \sqrt{\max\left(\sum_t \alpha_t h_t^2 - \tilde\mu^2,\ \epsilon\right)}
$$

Read it aloud: $h_t$ is one channel's activation at frame $t$, $\tilde\mu$ is the weighted mean, and $\tilde\sigma$ is the weighted standard deviation. The floor $\epsilon$, about $10^{-9}$, keeps the square root finite when every frame is alike.

### Work the pooling by hand

Take one channel with frames $h = (1, 2, 6)$.

- Statistics pooling gives mean $3$. The mean of squares is $41/3 = 13.667$, so the variance is $4.667$ and the standard deviation is 2.160.
- Attentive pooling with scores $(0, 0, \ln 2)$ gives weights $(0.25, 0.25, 0.5)$. The weighted mean is $0.25 + 0.5 + 3 = 3.75$. The weighted mean of squares is $0.25 + 1 + 18 = 19.25$, so the variance is $19.25 - 3.75^2 = 5.1875$ and the standard deviation is 2.278.

Subtracting the plain mean 3 instead gives variance $19.25 - 9 = 10.25$ and a standard deviation of 3.202, which is wrong: the variance must use the same weights as the mean.

## Why plain softmax is not enough

Normalize each embedding $x$ and each class center $W_j$ to unit length, so the score of class $j$ is $\cos\theta_j$, the cosine of the angle between them. A softmax over these scores is satisfied as soon as the true class scores highest, so each speaker's embeddings can stay spread over a wide cone, and a new speaker's embeddings fall into the gaps. The fix is to make the true class win by a fixed angle, which shrinks the cones and widens the gaps.

## The additive angular margin

The **additive angular margin** softmax (ArcFace) adds a margin $m$ to the true class's angle before the softmax:

$$
\mathcal{L} = -\log \frac{e^{s\cos(\theta_y + m)}}{e^{s\cos(\theta_y + m)} + \sum_{j \ne y} e^{s\cos\theta_j}}
$$

Read it aloud: $\theta_y$ is the angle to the true speaker's center, $m$ is an extra angle in radians, and $s$ is a scale that stretches cosines into logits large enough for a sharp softmax. The true class is scored as if the embedding were $m$ radians farther away than it is.

Compute the shifted cosine as $\cos\theta\cos m - \sin\theta\sin m$ with $\sin\theta = \sqrt{1 - \cos^2\theta}$, so you never call `arccos`.

```figure
<svg viewBox="0 0 600 310" role="img" aria-label="Cosine of the true-class angle against the angle, with and without a 0.2 rad margin, including the fallback past pi minus the margin">
<defs><marker id="c036-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<line x1="70.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">1</text>
<line x1="70.0" y1="95.0" x2="560.0" y2="95.0" class="stroke-grid"/>
<text x="62.0" y="99.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70.0" y1="150.0" x2="560.0" y2="150.0" class="stroke-grid"/>
<text x="62.0" y="154.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="205.0" x2="560.0" y2="205.0" class="stroke-grid"/>
<text x="62.0" y="209.0" text-anchor="end" class="fig-small">-0.5</text>
<line x1="70.0" y1="260.0" x2="560.0" y2="260.0" class="stroke-grid"/>
<text x="62.0" y="264.0" text-anchor="end" class="fig-small">-1</text>
<line x1="70.0" y1="260.0" x2="70.0" y2="265.0" class="stroke-muted"/>
<text x="70.0" y="280.0" text-anchor="middle" class="fig-small">0</text>
<line x1="226.0" y1="260.0" x2="226.0" y2="265.0" class="stroke-muted"/>
<text x="226.0" y="280.0" text-anchor="middle" class="fig-small">1</text>
<line x1="381.9" y1="260.0" x2="381.9" y2="265.0" class="stroke-muted"/>
<text x="381.9" y="280.0" text-anchor="middle" class="fig-small">2</text>
<line x1="537.9" y1="260.0" x2="537.9" y2="265.0" class="stroke-muted"/>
<text x="537.9" y="280.0" text-anchor="middle" class="fig-small">3</text>
<text x="560.0" y="280.0" text-anchor="middle" class="fig-small">pi</text>
<polyline points="70.0,40.0 72.5,40.0 74.9,40.1 77.4,40.1 79.8,40.2 82.3,40.3 84.8,40.5 87.2,40.7 89.7,40.9 92.2,41.1 94.6,41.4 97.1,41.7 99.5,42.0 102.0,42.3 104.5,42.7 106.9,43.1 109.4,43.5 111.9,43.9 114.3,44.4 116.8,44.9 119.2,45.4 121.7,46.0 124.2,46.6 126.6,47.2 129.1,47.8 131.6,48.5 134.0,49.1 136.5,49.8 138.9,50.6 141.4,51.3 143.9,52.1 146.3,52.9 148.8,53.7 151.3,54.6 153.7,55.5 156.2,56.4 158.6,57.3 161.1,58.2 163.6,59.2 166.0,60.2 168.5,61.2 171.0,62.2 173.4,63.3 175.9,64.4 178.3,65.5 180.8,66.6 183.3,67.8 185.7,68.9 188.2,70.1 190.7,71.3 193.1,72.5 195.6,73.8 198.0,75.0 200.5,76.3 203.0,77.6 205.4,78.9 207.9,80.3 210.4,81.6 212.8,83.0 215.3,84.4 217.7,85.8 220.2,87.2 222.7,88.6 225.1,90.1 227.6,91.5 230.1,93.0 232.5,94.5 235.0,96.0 237.4,97.5 239.9,99.1 242.4,100.6 244.8,102.2 247.3,103.7 249.7,105.3 252.2,106.9 254.7,108.5 257.1,110.1 259.6,111.7 262.1,113.4 264.5,115.0 267.0,116.7 269.4,118.3 271.9,120.0 274.4,121.7 276.8,123.4 279.3,125.0 281.8,126.7 284.2,128.4 286.7,130.1 289.1,131.8 291.6,133.6 294.1,135.3 296.5,137.0 299.0,138.7 301.5,140.5 303.9,142.2 306.4,143.9 308.8,145.7 311.3,147.4 313.8,149.1 316.2,150.9 318.7,152.6 321.2,154.3 323.6,156.1 326.1,157.8 328.5,159.5 331.0,161.3 333.5,163.0 335.9,164.7 338.4,166.4 340.9,168.2 343.3,169.9 345.8,171.6 348.2,173.3 350.7,175.0 353.2,176.6 355.6,178.3 358.1,180.0 360.6,181.7 363.0,183.3 365.5,185.0 367.9,186.6 370.4,188.3 372.9,189.9 375.3,191.5 377.8,193.1 380.3,194.7 382.7,196.3 385.2,197.8 387.6,199.4 390.1,200.9 392.6,202.5 395.0,204.0 397.5,205.5 399.9,207.0 402.4,208.5 404.9,209.9 407.3,211.4 409.8,212.8 412.3,214.2 414.7,215.6 417.2,217.0 419.6,218.4 422.1,219.7 424.6,221.1 427.0,222.4 429.5,223.7 432.0,225.0 434.4,226.2 436.9,227.5 439.3,228.7 441.8,229.9 444.3,231.1 446.7,232.2 449.2,233.4 451.7,234.5 454.1,235.6 456.6,236.7 459.0,237.8 461.5,238.8 464.0,239.8 466.4,240.8 468.9,241.8 471.4,242.7 473.8,243.6 476.3,244.5 478.7,245.4 481.2,246.3 483.7,247.1 486.1,247.9 488.6,248.7 491.1,249.4 493.5,250.2 496.0,250.9 498.4,251.5 500.9,252.2 503.4,252.8 505.8,253.4 508.3,254.0 510.8,254.6 513.2,255.1 515.7,255.6 518.1,256.1 520.6,256.5 523.1,256.9 525.5,257.3 528.0,257.7 530.5,258.0 532.9,258.3 535.4,258.6 537.8,258.9 540.3,259.1 542.8,259.3 545.2,259.5 547.7,259.7 550.2,259.8 552.6,259.9 555.1,259.9 557.5,260.0 560.0,260.0" class="stroke-a" fill="none"/>
<polyline points="70.0,42.2 72.5,42.6 74.9,42.9 77.4,43.3 79.8,43.8 82.3,44.3 84.8,44.7 87.2,45.3 89.7,45.8 92.2,46.4 94.6,47.0 97.1,47.6 99.5,48.2 102.0,48.9 104.5,49.6 106.9,50.3 109.4,51.1 111.9,51.8 114.3,52.6 116.8,53.5 119.2,54.3 121.7,55.2 124.2,56.1 126.6,57.0 129.1,57.9 131.6,58.9 134.0,59.9 136.5,60.9 138.9,61.9 141.4,63.0 143.9,64.0 146.3,65.1 148.8,66.2 151.3,67.4 153.7,68.5 156.2,69.7 158.6,70.9 161.1,72.1 163.6,73.4 166.0,74.6 168.5,75.9 171.0,77.2 173.4,78.5 175.9,79.8 178.3,81.2 180.8,82.5 183.3,83.9 185.7,85.3 188.2,86.7 190.7,88.1 193.1,89.6 195.6,91.0 198.0,92.5 200.5,94.0 203.0,95.5 205.4,97.0 207.9,98.5 210.4,100.1 212.8,101.6 215.3,103.2 217.7,104.8 220.2,106.4 222.7,108.0 225.1,109.6 227.6,111.2 230.1,112.8 232.5,114.5 235.0,116.1 237.4,117.8 239.9,119.4 242.4,121.1 244.8,122.8 247.3,124.5 249.7,126.2 252.2,127.9 254.7,129.6 257.1,131.3 259.6,133.0 262.1,134.7 264.5,136.4 267.0,138.2 269.4,139.9 271.9,141.6 274.4,143.4 276.8,145.1 279.3,146.8 281.8,148.6 284.2,150.3 286.7,152.0 289.1,153.8 291.6,155.5 294.1,157.2 296.5,159.0 299.0,160.7 301.5,162.4 303.9,164.1 306.4,165.9 308.8,167.6 311.3,169.3 313.8,171.0 316.2,172.7 318.7,174.4 321.2,176.1 323.6,177.8 326.1,179.5 328.5,181.1 331.0,182.8 333.5,184.4 335.9,186.1 338.4,187.7 340.9,189.3 343.3,191.0 345.8,192.6 348.2,194.2 350.7,195.7 353.2,197.3 355.6,198.9 358.1,200.4 360.6,202.0 363.0,203.5 365.5,205.0 367.9,206.5 370.4,208.0 372.9,209.5 375.3,210.9 377.8,212.3 380.3,213.8 382.7,215.2 385.2,216.6 387.6,217.9 390.1,219.3 392.6,220.6 395.0,222.0 397.5,223.3 399.9,224.5 402.4,225.8 404.9,227.1 407.3,228.3 409.8,229.5 412.3,230.7 414.7,231.9 417.2,233.0 419.6,234.1 422.1,235.3 424.6,236.3 427.0,237.4 429.5,238.4 432.0,239.5 434.4,240.5 436.9,241.4 439.3,242.4 441.8,243.3 444.3,244.2 446.7,245.1 449.2,246.0 451.7,246.8 454.1,247.6 456.6,248.4 459.0,249.2 461.5,249.9 464.0,250.6 466.4,251.3 468.9,252.0 471.4,252.6 473.8,253.2 476.3,253.8 478.7,254.4 481.2,254.9 483.7,255.4 486.1,255.9 488.6,256.4 491.1,256.8 493.5,257.2 496.0,257.6 498.4,257.9 500.9,258.2 503.4,258.5 505.8,258.8 508.3,259.0 510.8,259.3 513.2,259.5 515.7,259.6 518.1,259.7 520.6,259.8 523.1,259.9 525.5,260.0 528.0,260.0 530.5,262.4 532.9,262.7 535.4,263.0 537.8,263.3 540.3,263.5 542.8,263.7 545.2,263.9 547.7,264.0 550.2,264.2 552.6,264.2 555.1,264.3 557.5,264.4 560.0,264.4" class="stroke-b" fill="none"/>
<line x1="233.3" y1="95.0" x2="233.3" y2="260.0" class="stroke-muted dashed"/>
<circle cx="233.3" cy="95.0" r="5" class="fill-a"/>
<circle cx="233.3" cy="115.0" r="5" class="fill-b"/>
<text x="243.3" y="87.0" text-anchor="start" class="fig-small">cos 1.047 = 0.500</text>
<text x="223.3" y="135.0" text-anchor="end" class="fig-small">cos(1.047 + 0.2) = 0.318</text>
<line x1="528.8" y1="40.0" x2="528.8" y2="260.0" class="stroke-muted dashed"/>
<text x="522.8" y="52.0" text-anchor="end" class="fig-small">fallback starts at pi - m</text>
<line x1="330.0" y1="100.0" x2="360.0" y2="100.0" class="stroke-a"/>
<text x="366.0" y="104.0" text-anchor="start" class="fig-small">no margin: cos(theta)</text>
<line x1="330.0" y1="120.0" x2="360.0" y2="120.0" class="stroke-b"/>
<text x="366.0" y="124.0" text-anchor="start" class="fig-small">margin 0.2: cos(theta + m)</text>
<text x="315.0" y="22.0" text-anchor="middle" class="fig-small">logit before scaling by s</text>
<text x="315.0" y="298.0" text-anchor="middle" class="fig-small">angle to the true class center, theta (radians)</text>
</svg>
Figure 1. The margin shifts the true class's cosine down: the embedding at 60 degrees scores 0.318 instead of 0.500, so it must come closer to its center to win. Past pi minus m the shifted curve would rise again, so the loss switches to a straight fallback.
```

Figure 1 shows a trap. Once $\theta_y + m$ passes $\pi$, $\cos(\theta_y + m)$ starts rising again, so the loss would reward moving away from the true center. Where $\cos\theta_y \le \cos(\pi - m)$, use $\cos\theta_y - m\sin m$ instead, which keeps falling as the angle grows.

### Work the margin by hand

Take two speakers, $s = 30$, and an embedding with cosine 0.5 to its own center and 0.4 to the other.

- Without a margin, the logits are 15 and 12. The softmax gives the true class 0.9526, and the loss is 0.0486 nats.
- With $m = 0.2$, the true cosine becomes $0.5 \times 0.9801 - 0.8660 \times 0.1987 = 0.3180$, and its logit is 9.54. The softmax gives it 0.0787, and the loss is 2.54 nats.

An embedding that plain softmax calls finished still pays 2.54 nats. Its loss falls toward zero only when its angle to its own center is at least 0.2 rad smaller than to any other.

> [!KEY]
> The angular margin makes the true class win by an angle $m$, not merely win. That packs each training speaker's embeddings tightly, which is what lets unseen speakers separate at test time.

## Score a trial

A **trial** is a pair of clips and a question: same speaker or not? The score is the cosine between their embeddings. A **target** trial is a same-speaker pair, and a **non-target** trial is a different-speaker pair.

Accept a trial when its score is at least a threshold $\tau$. Two errors trade against each other as $\tau$ moves:

- A **miss**: a target trial scored below $\tau$. $P_{\text{miss}}(\tau)$ is the fraction of target scores below it.
- A **false alarm**: a non-target trial scored at or above $\tau$. $P_{\text{fa}}(\tau)$ is the fraction of non-target scores at or above it.

Raising $\tau$ rejects more impostors and more customers. No threshold removes both errors, so a report has to say where on that trade it sits.

## One number: the equal error rate

The **equal error rate** (EER) is the error where the two rates meet. The exercise tries every distinct score as $\tau$, plus $+\infty$, and returns $(P_{\text{miss}} + P_{\text{fa}})/2$ where the two are closest.

Take target scores 0.9, 0.8, 0.7, and 0.4, and non-target scores 0.6, 0.3, 0.2, and 0.1. At $\tau = 0.6$, one target (0.4) is missed and one non-target (0.6) is accepted, so both rates are 25% and the EER is 25%.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Miss rate rises and false alarm rate falls as the threshold rises, and they meet at 0.25 near a threshold of 0.6">
<defs><marker id="c036-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<line x1="70.0" y1="240.0" x2="560.0" y2="240.0" class="stroke-grid"/>
<text x="62.0" y="244.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="187.5" x2="560.0" y2="187.5" class="stroke-grid"/>
<text x="62.0" y="191.5" text-anchor="end" class="fig-small">0.25</text>
<line x1="70.0" y1="135.0" x2="560.0" y2="135.0" class="stroke-grid"/>
<text x="62.0" y="139.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="70.0" y1="82.5" x2="560.0" y2="82.5" class="stroke-grid"/>
<text x="62.0" y="86.5" text-anchor="end" class="fig-small">0.75</text>
<line x1="70.0" y1="30.0" x2="560.0" y2="30.0" class="stroke-grid"/>
<text x="62.0" y="34.0" text-anchor="end" class="fig-small">1</text>
<line x1="70.0" y1="240.0" x2="70.0" y2="245.0" class="stroke-muted"/>
<text x="70.0" y="260.0" text-anchor="middle" class="fig-small">0</text>
<line x1="168.0" y1="240.0" x2="168.0" y2="245.0" class="stroke-muted"/>
<text x="168.0" y="260.0" text-anchor="middle" class="fig-small">0.2</text>
<line x1="266.0" y1="240.0" x2="266.0" y2="245.0" class="stroke-muted"/>
<text x="266.0" y="260.0" text-anchor="middle" class="fig-small">0.4</text>
<line x1="364.0" y1="240.0" x2="364.0" y2="245.0" class="stroke-muted"/>
<text x="364.0" y="260.0" text-anchor="middle" class="fig-small">0.6</text>
<line x1="462.0" y1="240.0" x2="462.0" y2="245.0" class="stroke-muted"/>
<text x="462.0" y="260.0" text-anchor="middle" class="fig-small">0.8</text>
<line x1="560.0" y1="240.0" x2="560.0" y2="245.0" class="stroke-muted"/>
<text x="560.0" y="260.0" text-anchor="middle" class="fig-small">1</text>
<polyline points="70.0,240.0 119.0,240.0 119.0,240.0 168.0,240.0 168.0,240.0 217.0,240.0 217.0,240.0 266.0,240.0 266.0,187.5 364.0,187.5 364.0,187.5 413.0,187.5 413.0,135.0 462.0,135.0 462.0,82.5 511.0,82.5 511.0,30.0 560.0,30.0" class="stroke-b" fill="none"/>
<polyline points="70.0,30.0 119.0,30.0 119.0,82.5 168.0,82.5 168.0,135.0 217.0,135.0 217.0,187.5 266.0,187.5 266.0,187.5 364.0,187.5 364.0,240.0 413.0,240.0 413.0,240.0 462.0,240.0 462.0,240.0 511.0,240.0 511.0,240.0 560.0,240.0" class="stroke-a" fill="none"/>
<circle cx="364.0" cy="187.5" r="6" class="fill-c"/>
<text x="373.8" y="219.0" text-anchor="start" class="fig-small">EER = 0.25 at threshold 0.6</text>
<text x="511.0" y="22.0" text-anchor="middle" class="fig-small fig-bold">P_miss</text>
<text x="138.6" y="74.5" text-anchor="middle" class="fig-small fig-bold">P_fa</text>
<text x="315.0" y="282.0" text-anchor="middle" class="fig-small">threshold on the cosine score</text>
</svg>
Figure 2. For the four target and four non-target scores in the text, every threshold trades misses for false alarms. The two rates are both 25% at a threshold of 0.6, which is the equal error rate.
```

The EER treats a miss and a false alarm as equally bad, and a bank does not: a false alarm lets an impostor in. The **detection cost** weights the errors by cost and frequency:

$$
C(\tau) = \frac{C_{\text{miss}}\,P_{\text{miss}}(\tau)\,P_{\text{tar}} + C_{\text{fa}}\,P_{\text{fa}}(\tau)\,(1 - P_{\text{tar}})}{\min\left(C_{\text{miss}} P_{\text{tar}},\ C_{\text{fa}} (1 - P_{\text{tar}})\right)}
$$

Read it aloud: $P_{\text{tar}}$ is the prior probability that a trial is a target, and $C_{\text{miss}}$ and $C_{\text{fa}}$ are the costs of each error. The denominator is the cost of the better of two trivial systems, accept everything or reject everything, so a cost of 1.0 means no better than guessing. **minDCF** is the minimum over thresholds. VoxSRC uses $P_{\text{tar}} = 0.05$ with both costs 1, so the formula becomes $P_{\text{miss}} + 19\,P_{\text{fa}}$. A detector that listens all day is scored the same way, in false accepts per hour, in [Keyword spotting, endpointing, and language ID](/lessons/043-keyword-spotting-and-language-id).

For the same eight scores, $\tau = 0.7$ gives miss rate 25% and false-alarm rate 0, so the cost is 0.25, and no threshold does better.

## What the margin buys

The Run makes 70 synthetic voices, each a buzz at its own pitch through formant filters scaled to its own vocal tract, trains a linear embedding on 60, and scores trials on the other 10.

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Equal error rate on 10 unseen synthetic voices: 11.4% for raw features, 2.2% with a plain softmax, and 0.7% with a 0.2 rad margin">
<defs><marker id="c036-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="20.0" y="28.0" text-anchor="start" class="fig-bold fig-small">Training objective</text>
<text x="190.0" y="28.0" text-anchor="start" class="fig-bold fig-small">EER on unseen voices</text>
<text x="20.0" y="74.0" text-anchor="start">raw spectra</text>
<rect x="190.0" y="50.0" width="237.5" height="36.0" rx="3" class="fill-sunk stroke-muted"/>
<text x="437.5" y="67.0" text-anchor="start" class="fig-bold">11.4%</text>
<text x="437.5" y="83.0" text-anchor="start" class="fig-small">minDCF 0.522</text>
<text x="20.0" y="132.0" text-anchor="start">softmax, m = 0</text>
<rect x="190.0" y="108.0" width="45.8" height="36.0" rx="3" class="fill-a"/>
<text x="245.8" y="125.0" text-anchor="start" class="fig-bold">2.2%</text>
<text x="245.8" y="141.0" text-anchor="start" class="fig-small">minDCF 0.209</text>
<text x="20.0" y="190.0" text-anchor="start">AAM softmax, m = 0.2</text>
<rect x="190.0" y="166.0" width="14.6" height="36.0" rx="3" class="fill-b"/>
<text x="214.6" y="183.0" text-anchor="start" class="fig-bold">0.7%</text>
<text x="214.6" y="199.0" text-anchor="start" class="fig-small">minDCF 0.071</text>
<line x1="190.0" y1="40.0" x2="190.0" y2="230.0" class="stroke-muted"/>
</svg>
Figure 3. On voices the embedding never saw in training, the angular margin cuts the equal error rate from 2.2% to 0.7%, and cuts minDCF at P_target = 0.05 from 0.209 to 0.071.
```

The margin embedding separates voices it never trained on, which answers the puzzle.

> [!KEY]
> A verification system is an embedding plus a threshold. The margin loss shapes the embedding, and EER and minDCF report the two ways the threshold can be wrong.

## Enroll with several clips

To combine $N$ enrollment clips, normalize each embedding to unit length, average the unit vectors, and normalize the average. Normalizing first stops a clip with a large norm from dominating. For two clips at right angles with lengths 1 and 10, averaging first gives $(0.0995, 0.9950)$, nearly all the long clip. Normalizing first gives $(0.7071, 0.7071)$.

The alternative is to average the cosines of the test clip with each enrollment clip. For unit vectors, that average equals $t \cdot \bar e$, where $t$ is the test embedding and $\bar e$ is the mean of the enrollment embeddings, which is $\lVert\bar e\rVert$ times the cosine to the normalized average. Both rank test clips alike for one speaker, but score averaging shrinks every score by $\lVert\bar e\rVert$, which is 1 only when the clips agree exactly. A speaker with scattered clips gets lower scores, and one global threshold treats them unfairly.

## Normalize the scores

A cosine of 0.45 means different things for different people, because some voices resemble many others. **Score normalization** judges a score against what impostors score.

**Adaptive symmetric normalization** (AS-norm; Matějka et al., 2017) uses a **cohort**, embeddings from speakers outside the trial. For enrollment $e$, test $t$, and raw score $s$:

1. Score $e$ against every cohort embedding, keep the $K$ highest, and call their mean and standard deviation $\mu_e$ and $\sigma_e$.
2. Do the same for $t$ to get $\mu_t$ and $\sigma_t$.
3. Return $0.5\left(\frac{s - \mu_e}{\sigma_e} + \frac{s - \mu_t}{\sigma_t}\right)$.

Each term is a z-score against the closest impostors.

Take $s = 0.45$ and $K = 3$. The enrollment cohort scores are $(0.30, 0.20, 0.10, 0.05, 0.0, -0.10)$, and the test cohort scores are $(0.20, 0.15, 0.10, 0.0, -0.05, -0.10)$. The top three on each side are $(0.30, 0.20, 0.10)$ with mean 0.200 and standard deviation 0.0816, and $(0.20, 0.15, 0.10)$ with mean 0.150 and standard deviation 0.0408. The result is $0.5\,(3.06 + 7.35) = 5.205$, against 3.225 for the whole cohort.

Many x-vector systems used **PLDA**, a generative back end that models between-speaker and within-speaker covariance and scores a pair by the likelihood ratio of same against different speaker. An angular-margin embedding already clusters by angle, so cosine on normalized embeddings, with AS-norm on top, is the common modern back end.

## VoxCeleb and mismatch

**VoxCeleb1** holds 1,251 speakers and 153,516 utterances from interview videos. Three trial lists set the protocols: O is the original list, with 40 test speakers. E draws trials from all of VoxCeleb1. H draws pairs only from speakers of the same gender and nationality, which removes easy cues. ECAPA-TDNN with 1024 channels reports 0.87%, 1.12%, and 2.12% EER on O, E, and H, in that order.

Two mismatches hurt in practice:

- **Duration.** Xie et al. (2019) cut VoxCeleb1 clips to fixed lengths, and their EER fell from 7.97% at 2 s to 3.39% at 6 s. A short clip may hold mostly silence or noise.
- **Domain.** Embeddings trained on interviews meet telephone audio, other languages, and other microphones. Scores shift, so a threshold from VoxCeleb misfires. Build the AS-norm cohort and the threshold from audio like your deployment audio.

The Run applies both steps to the margin-0.2 embedding. Enrolling from three clips instead of one lowers the EER from 0.3% to 0.0%. AS-norm over 360 training clips raises it from 0.7% to 3.2%: the embedding is already nearly perfect on clean voices, and a small cohort adds noise. AS-norm pays off under mismatch.

> [!RECAP]
> - Statistics pooling concatenates each channel's mean and standard deviation over frames. Attentive pooling weights frames with a softmax and uses the weighted mean inside the weighted variance.
> - AAM softmax replaces the true cosine with $\cos\theta\cos m - \sin\theta\sin m$, or $\cos\theta - m\sin m$ past $\pi - m$, then applies cross-entropy to $s$ times the cosines.
> - A trial's score is the cosine of two embeddings. A miss is a target below the threshold, and a false alarm is a non-target at or above it. EER is where the two rates meet, and minDCF is the minimum weighted cost over the cost of the better trivial system.
> - Enroll by averaging unit-length embeddings and renormalizing. AS-norm z-scores a trial against the top-$K$ cohort scores on each side.

## Build it step by step

The checks test the functions in this order.

1. **`cosine_scores(a, b)`.** Takes two `(N, D)` arrays and returns `(N,)`. Common mistake: dividing by the norm of the whole matrix instead of each row.
2. **`aam_softmax_loss(embeddings, weights, labels, scale, margin)`.** Takes torch tensors and returns a scalar tensor. Clone the cosine matrix and write the shifted value into `logits[rows, labels]`. Common mistake: applying the margin to every class, not only the true one.
3. **`equal_error_rate(target_scores, nontarget_scores)`.** Returns a float. Use the distinct scores plus `np.inf` as thresholds. Common mistake: counting a score equal to the threshold as a miss.
4. **`min_dcf(target_scores, nontarget_scores, p_target, c_miss, c_fa)`.** Returns a float. Common mistake: forgetting the normalization.
5. **`stats_pooling(frames)`.** Takes `(T, C)` and returns `(2 * C,)`: means, then standard deviations, over axis 0.
6. **`attentive_stats_pooling(frames, attn_logits, floor)`.** Softmax over axis 0, then the weighted mean and the floored square root of the weighted second moment minus the squared weighted mean. Common mistake: subtracting the unweighted mean.
7. **`enroll(embeddings)`.** Returns a unit `(D,)` vector. Common mistake: averaging before normalizing each embedding.
8. **`as_norm(score, enroll_cohort_scores, test_cohort_scores, top_k)`.** Keep the last `top_k` of each sorted cohort. Common mistake: using the whole cohort.

## Where this breaks

**Short clips and a shifted domain.** A threshold tuned on long clips from one microphone can fail on short clips from another. Re-estimate it, and draw the AS-norm cohort, from audio like your deployment audio. To hide whose voice a recording holds, see the anonymization section of [Anti-spoofing and watermarking](/lessons/075-anti-spoofing-and-watermarking).

**A margin from step 0.** A large margin at the start can stall the loss. The Run warms it up over 100 steps.

## The exercise

You build `cosine_scores`, `aam_softmax_loss`, `equal_error_rate`, and `min_dcf`, then `stats_pooling`, `attentive_stats_pooling`, `enroll`, and `as_norm`. The checks confirm the worked margin example, the fallback past $\pi - m$, the Gaussian EER $\Phi(-d/2)$, a brute-force minDCF, and the pooling and AS-norm examples. Separate checks catch attentive pooling that subtracts the unweighted mean, enrollment that averages before normalizing, and AS-norm over the whole cohort.

The **Run** plays two synthetic voices, prints EER and minDCF on 10 unseen voices for raw spectra, $m = 0$, and $m = 0.2$, then tries one and three enrollment clips and AS-norm.

## Further reading

- Snyder et al., "X-vectors: Robust DNN Embeddings for Speaker Recognition," ICASSP 2018 (https://www.danielpovey.com/files/2018_icassp_xvectors.pdf).
- Desplanques, Thienpondt, and Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech 2020 (https://arxiv.org/abs/2005.07143).
- Matějka et al., "Analysis of Score Normalization in Multilingual Speaker Recognition," Interspeech 2017 (https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html).
- Xie, Nagrani, Chung, and Zisserman, "Utterance-Level Aggregation for Speaker Recognition in the Wild," ICASSP 2019 (https://arxiv.org/abs/1902.10107).
