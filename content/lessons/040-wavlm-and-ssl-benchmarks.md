---
title: WavLM and benchmarking self-supervised models
slug: 040-wavlm-and-ssl-benchmarks
part: "Part 4 — Neural speech recognition"
summary: Train an encoder to ignore a second voice with WavLM, compare five pretraining targets, and rank encoders the SUPERB way.
minutes: 45
gpu: false
objectives:
  - Mix a second signal into a span of the primary at a chosen energy ratio, and say why the targets stay those of the clean primary.
  - Bucket signed frame offsets into WavLM's relative position bias, and gate that bias for each query.
  - Compare what wav2vec 2.0, HuBERT, WavLM, data2vec, and w2v-BERT predict at a masked frame.
  - Mix frozen layers with SUPERB's learnable weighted sum, and read a SUPERB table across tasks.
prereqs:
  - 037-conformers-and-specaugment
  - 039-self-supervised-speech
exercise: 040-wavlm-and-ssl-benchmarks
---

# WavLM and benchmarking self-supervised models

HuBERT learns from audiobooks, where one person reads in a quiet room. A meeting recording is different: two people talk at once, a fan hums, and someone coughs. An encoder that has never heard an intruding voice has never been asked to ignore one. How do you teach it to?

A second question follows once you have several encoders. wav2vec 2.0, HuBERT, and WavLM are built differently, so a single training curve can't rank them. How do you compare them on equal terms? This chapter builds WavLM's answer to the first question and the SUPERB benchmark's answer to the second.

## What you already know

- Masked prediction of k-means labels trains an encoder from audio alone, from [Self-supervised speech](/lessons/039-self-supervised-speech).
- A softmax turns any scores into positive weights that sum to 1, as in the contrastive loss of the same chapter.
- Attention scores can take a bias that depends on the offset between two frames, from the relative positions in [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment).

## Denoise while you predict

**WavLM** (Chen et al., 2022) keeps HuBERT's masked label prediction and changes two things. The first is the input. For some utterances, WavLM adds a second utterance or a noise clip into one span of the **primary** utterance, and the targets stay the cluster labels of the clean primary. The model must find the main voice and ignore the intruder, because the labels it is scored on describe only the main voice.

The secondary signal needs a volume. If it is too loud, the primary is buried and the task is impossible. If it is too quiet, the task is trivial. So WavLM scales the secondary until the primary span's energy $E_\text{pri}$, a sum of squares, sits $r$ dB above the scaled secondary's:

$$
s = \sqrt{\frac{E_\text{pri}}{10^{r/10}\, E_\text{sec}}}
$$

An energy ratio of $r$ dB is a factor of $10^{r/10}$, and amplitude is the square root of energy, which is why the square root appears. In numpy, `mixed[span] += s * secondary[span]`. The paper measures energy over whole utterances. The exercise measures it inside the span, so the ratio in the overlap is exact.

Take six ones, a secondary $(0, 0, 3, -3, 0, 0)$, a span of 2 samples at sample 2, and $r = 10 \log_{10} 4 = 6.02$ dB. The span energies are 2 and 18, so $s = \sqrt{2 / (4 \cdot 18)} = 1/6$ and the mix is $(1, 1, 1.5, 0.5, 1, 1)$. Figure 1 draws those numbers.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="Six samples of a primary signal, the scaled secondary signal in a two-sample span, and their sum">
<rect x="293" y="30" width="129" height="285" class="fill-soft-d"/>
<text x="357" y="24" text-anchor="middle" class="fig-small">span</text>
<text x="20" y="84" class="fig-small">primary</text>
<line x1="170" y1="80" x2="580" y2="80" class="stroke-muted"/>
<rect x="180" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="195" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="245" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="260" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="310" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="325" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="375" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="390" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="440" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="455" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="505" y="44.0" width="30" height="36.0" class="fill-a"/>
<text x="520" y="38.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<text x="20" y="179" class="fig-small">secondary x 0.1667</text>
<line x1="170" y1="175" x2="580" y2="175" class="stroke-muted"/>
<text x="195" y="169.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<text x="260" y="169.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<rect x="310" y="157.0" width="30" height="18.0" class="fill-b"/>
<text x="325" y="151.0" text-anchor="middle" class="fig-small fig-mono">0.5</text>
<rect x="375" y="175.0" width="30" height="18.0" class="fill-b"/>
<text x="390" y="207.0" text-anchor="middle" class="fig-small fig-mono">-0.5</text>
<text x="455" y="169.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<text x="520" y="169.0" text-anchor="middle" class="fig-small fig-mono">0</text>
<text x="20" y="274" class="fig-small">mixed</text>
<line x1="170" y1="270" x2="580" y2="270" class="stroke-muted"/>
<rect x="180" y="234.0" width="30" height="36.0" class="fill-c"/>
<text x="195" y="228.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="245" y="234.0" width="30" height="36.0" class="fill-c"/>
<text x="260" y="228.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="310" y="216.0" width="30" height="54.0" class="fill-c"/>
<text x="325" y="210.0" text-anchor="middle" class="fig-small fig-mono">1.5</text>
<rect x="375" y="252.0" width="30" height="18.0" class="fill-c"/>
<text x="390" y="246.0" text-anchor="middle" class="fig-small fig-mono">0.5</text>
<rect x="440" y="234.0" width="30" height="36.0" class="fill-c"/>
<text x="455" y="228.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<rect x="505" y="234.0" width="30" height="36.0" class="fill-c"/>
<text x="520" y="228.0" text-anchor="middle" class="fig-small fig-mono">1</text>
<text x="380" y="326" text-anchor="middle" class="fig-small">sample index 0 to 5 (left to right)</text>
</svg>
Figure 1. The worked mix. The secondary (3, -3) shrinks to (0.5, -0.5) and is added only inside the span, so the four samples outside it keep their value of 1. Notice that the added energy 0.5 is a quarter of the primary's 2, which is 6.02 dB.
```

The second change is the position signal. Attention scores gain a bias, $q_i \cdot k_j / \sqrt{d_k} + r_{i-j}$, that depends on the offset $i - j$ between query frame $i$ and key frame $j$. A table with one number for every possible offset would be huge and would see few training examples at long distances. So WavLM sorts offsets into $n = 320$ **buckets**, each with one learned number $d$. The first $n/2$ buckets hold $i - j \le 0$ and the rest hold $i - j > 0$, so the sign survives. On each side, offsets below $n/4 = 80$ get their own bucket, offsets up to $m = 800$ share buckets spaced evenly in $\log |i - j|$, and farther ones share the last.

With $n = 8$ and $m = 16$, each side has 4 buckets and 2 are exact. The offsets $-20, -1, 0, +1, +4, +20$ land in buckets $3, 1, 0, 5, 6, 7$. Figure 2 plots the real 320-bucket map.

```figure
<svg viewBox="0 0 560 280" role="img" aria-label="Relative position bucket against the signed offset i minus j for 320 buckets and a maximum distance of 800">
<line x1="95.0" y1="30" x2="95.0" y2="220" class="stroke-grid"/>
<text x="95.0" y="236" text-anchor="middle" class="fig-small">-800</text>
<line x1="195.0" y1="30" x2="195.0" y2="220" class="stroke-grid"/>
<text x="195.0" y="236" text-anchor="middle" class="fig-small">-400</text>
<line x1="295.0" y1="30" x2="295.0" y2="220" class="stroke-grid"/>
<text x="295.0" y="236" text-anchor="middle" class="fig-small">0</text>
<line x1="395.0" y1="30" x2="395.0" y2="220" class="stroke-grid"/>
<text x="395.0" y="236" text-anchor="middle" class="fig-small">400</text>
<line x1="495.0" y1="30" x2="495.0" y2="220" class="stroke-grid"/>
<text x="495.0" y="236" text-anchor="middle" class="fig-small">800</text>
<line x1="70" y1="220.0" x2="520" y2="220.0" class="stroke-grid"/>
<text x="62" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="172.4" x2="520" y2="172.4" class="stroke-grid"/>
<text x="62" y="176.4" text-anchor="end" class="fig-small">80</text>
<line x1="70" y1="124.7" x2="520" y2="124.7" class="stroke-grid"/>
<text x="62" y="128.7" text-anchor="end" class="fig-small">160</text>
<line x1="70" y1="77.1" x2="520" y2="77.1" class="stroke-grid"/>
<text x="62" y="81.1" text-anchor="end" class="fig-small">240</text>
<line x1="70" y1="30.0" x2="520" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">319</text>
<polyline points="70.0,125.3 71.5,125.3 73.0,125.3 74.5,125.3 76.0,125.3 77.5,125.3 79.0,125.3 80.5,125.3 82.0,125.3 83.5,125.3 85.0,125.3 86.5,125.3 88.0,125.3 89.5,125.3 91.0,125.3 92.5,125.3 94.0,125.3 95.5,125.3 97.0,125.3 98.5,125.3 100.0,125.3 101.5,125.9 103.0,125.9 104.5,125.9 106.0,125.9 107.5,126.5 109.0,126.5 110.5,126.5 112.0,127.1 113.5,127.1 115.0,127.1 116.5,127.1 118.0,127.7 119.5,127.7 121.0,127.7 122.5,128.3 124.0,128.3 125.5,128.3 127.0,128.9 128.5,128.9 130.0,128.9 131.5,129.5 133.0,129.5 134.5,129.5 136.0,129.5 137.5,130.1 139.0,130.1 140.5,130.1 142.0,130.7 143.5,130.7 145.0,130.7 146.5,131.3 148.0,131.3 149.5,131.8 151.0,131.8 152.5,131.8 154.0,132.4 155.5,132.4 157.0,132.4 158.5,133.0 160.0,133.0 161.5,133.6 163.0,133.6 164.5,133.6 166.0,134.2 167.5,134.2 169.0,134.8 170.5,134.8 172.0,134.8 173.5,135.4 175.0,135.4 176.5,136.0 178.0,136.0 179.5,136.6 181.0,136.6 182.5,136.6 184.0,137.2 185.5,137.2 187.0,137.8 188.5,137.8 190.0,138.4 191.5,138.4 193.0,139.0 194.5,139.0 196.0,139.6 197.5,139.6 199.0,140.2 200.5,140.8 202.0,140.8 203.5,141.4 205.0,141.4 206.5,142.0 208.0,142.0 209.5,142.6 211.0,143.2 212.5,143.2 214.0,143.8 215.5,144.4 217.0,144.4 218.5,145.0 220.0,145.5 221.5,145.5 223.0,146.1 224.5,146.7 226.0,146.7 227.5,147.3 229.0,147.9 230.5,148.5 232.0,149.1 233.5,149.1 235.0,149.7 236.5,150.3 238.0,150.9 239.5,151.5 241.0,152.1 242.5,152.7 244.0,153.3 245.5,153.9 247.0,154.5 248.5,155.1 250.0,155.7 251.5,156.9 253.0,157.5 254.5,158.1 256.0,158.7 257.5,159.8 259.0,160.4 260.5,161.6 262.0,162.2 263.5,163.4 265.0,164.0 266.5,165.2 268.0,166.4 269.5,167.6 271.0,168.8 272.5,170.0 274.0,171.8 275.5,173.5 277.0,177.1 278.5,180.7 280.0,184.3 281.5,187.8 283.0,191.4 284.5,195.0 286.0,198.6 287.5,202.1 289.0,205.7 290.5,209.3 292.0,212.9 293.5,216.4 295.0,220.0" class="stroke-b" fill="none"/>
<polyline points="295.2,124.1 296.8,120.5 298.2,117.0 299.8,113.4 301.2,109.8 302.8,106.2 304.2,102.7 305.8,99.1 307.2,95.5 308.8,91.9 310.2,88.4 311.8,84.8 313.2,81.2 314.8,77.6 316.2,75.9 317.8,74.7 319.2,73.5 320.8,72.3 322.2,71.1 323.8,69.9 325.2,68.7 326.8,67.5 328.2,66.9 329.8,65.7 331.2,65.1 332.8,63.9 334.2,63.4 335.8,62.8 337.2,62.2 338.8,61.0 340.2,60.4 341.8,59.8 343.2,59.2 344.8,58.6 346.2,58.0 347.8,57.4 349.2,56.8 350.8,56.2 352.2,55.6 353.8,55.0 355.2,54.4 356.8,53.8 358.2,53.2 359.8,53.2 361.2,52.6 362.8,52.0 364.2,51.4 365.8,51.4 367.2,50.8 368.8,50.3 370.2,49.7 371.8,49.7 373.2,49.1 374.8,48.5 376.2,48.5 377.8,47.9 379.2,47.9 380.8,47.3 382.2,46.7 383.8,46.7 385.2,46.1 386.8,46.1 388.2,45.5 389.8,44.9 391.2,44.9 392.8,44.3 394.2,44.3 395.8,43.7 397.2,43.7 398.8,43.1 400.2,43.1 401.8,42.5 403.2,42.5 404.8,41.9 406.2,41.9 407.8,41.3 409.2,41.3 410.8,41.3 412.2,40.7 413.8,40.7 415.2,40.1 416.8,40.1 418.2,39.5 419.8,39.5 421.2,38.9 422.8,38.9 424.2,38.9 425.8,38.3 427.2,38.3 428.8,37.7 430.2,37.7 431.8,37.7 433.2,37.1 434.8,37.1 436.2,37.1 437.8,36.6 439.2,36.6 440.8,36.0 442.2,36.0 443.8,36.0 445.2,35.4 446.8,35.4 448.2,35.4 449.8,34.8 451.2,34.8 452.8,34.8 454.2,34.2 455.8,34.2 457.2,34.2 458.8,33.6 460.2,33.6 461.8,33.6 463.2,33.6 464.8,33.0 466.2,33.0 467.8,33.0 469.2,32.4 470.8,32.4 472.2,32.4 473.8,31.8 475.2,31.8 476.8,31.8 478.2,31.8 479.8,31.2 481.2,31.2 482.8,31.2 484.2,30.6 485.8,30.6 487.2,30.6 488.8,30.6 490.2,30.0 491.8,30.0 493.2,30.0 494.8,30.0 496.2,30.0 497.8,30.0 499.2,30.0 500.8,30.0 502.2,30.0 503.8,30.0 505.2,30.0 506.8,30.0 508.2,30.0 509.8,30.0 511.2,30.0 512.8,30.0 514.2,30.0 515.8,30.0 517.2,30.0 518.8,30.0" class="stroke-b" fill="none"/>
<circle cx="294.2" cy="218.2" r="5" class="fill-a"/>
<text x="284.2" y="208.2" text-anchor="end" class="fig-small">offset -3: bucket 3</text>
<circle cx="295.8" cy="122.9" r="5" class="fill-a"/>
<text x="305.8" y="126.9" text-anchor="start" class="fig-small">offset +3: bucket 163</text>
<circle cx="495.0" cy="30.0" r="5" class="fill-a"/>
<text x="487.0" y="46.0" text-anchor="end" class="fig-small">offset +800: bucket 319</text>
<text x="295" y="258" text-anchor="middle">offset i - j (frames)</text>
<text x="18" y="125" transform="rotate(-90 18 125)" text-anchor="middle">bucket index</text>
</svg>
Figure 2. The bucket for each offset i - j, with 320 buckets and a maximum distance of 800. Notice that the two signs use separate halves, offsets near zero get one bucket each, and the buckets widen as the distance grows.
```

A **gate** then scales the bias by what the query holds, so the bias can change with what the query holds. Query $i$ computes $g_u = \sigma(q_i \cdot u)$ and $g_r = \sigma(q_i \cdot v)$, with learned vectors $u$ and $v$, a learned scalar $w$, and sigmoid $\sigma$:

$$
r_{i-j} = d_{i-j}\,\bigl(1 + g_u + (1 - g_u)\, w\, g_r\bigr)
$$

Read it aloud: the bias $d$ is multiplied by a scale between 1 and 2 that the query chooses. In numpy, `bias * scale[:, None]`. With $d = 2$ and $w = 1$, a query with both gates at 0.5 has scale 1.75 and $r = 3.5$, and a query with gates 0.8808 and 0.1192 has scale 1.8950 and $r = 3.79$, for the same offset.

> [!WARNING]
> The gates come from the query, so every key in a row shares one scale. Scaling columns instead makes the gate follow the key, which is a different model.

## Compare what the models predict

The following table lists five pretraining designs by what the model produces at a masked frame: a learned code, an offline code, or a continuous vector.

| Model | Target at a masked frame | Loss |
|---|---|---|
| wav2vec 2.0 | Quantized vector $q_t$, from a learned quantizer | Contrastive plus diversity |
| HuBERT | k-means cluster id, offline | Cross-entropy |
| WavLM | Cluster id of the clean primary | Cross-entropy, noisy input |
| data2vec | Mean of the top $K = 8$ block outputs of a moving-average teacher | L2 regression |
| w2v-BERT | The quantizer's token id | Contrastive plus cross-entropy |

Two columns separate the designs. The target is either learned together with the model, as in wav2vec 2.0 and data2vec, or fixed beforehand, as in HuBERT. The loss is either a choice among discrete codes or a regression onto a vector. WavLM changes neither column. It changes what the encoder hears and how it reads position.

## Score a model the SUPERB way

Ranking models needs one protocol for all of them. **SUPERB** (Yang et al., Interspeech 2021) freezes the upstream model and gives each task a **learnable weighted sum of all layers**, because the last layer is not always the best. The previous chapter showed why: speaker information peaks in the lower layers and content in the middle ones. Only a light downstream model trains, such as a linear layer with CTC for phoneme recognition or a 2-layer BLSTM with CTC for ASR. With layer scores $a_l$, the mix is

$$
h_t = \sum_{l=0}^{L} \operatorname{softmax}(a)_l \, h_t^{(l)}
$$

The softmax makes the weights positive and sum to 1, and they train with the downstream model. In numpy, `np.tensordot(softmax(a), layers, axes=1)`. For layer values $(4, 8, 12)$ and scores $(0, 0, \ln 2)$, the weights are $(1/4, 1/4, 1/2)$ and the mix is $1 + 2 + 6 = 9$. Raw scores as weights give 8.32, as Figure 3 shows.

```figure
<svg viewBox="0 0 600 290" role="img" aria-label="Weights on three layers with values 4, 8, and 12, as a softmax and as raw scores, and the mix each gives">
<text x="40" y="22" class="fig-bold">softmax weights: mix 9.0000</text>
<line x1="40" y1="220" x2="270" y2="220" class="stroke-muted"/>
<rect x="48" y="180.0" width="44" height="40.0" class="fill-b"/>
<text x="70" y="174.0" text-anchor="middle" class="fig-small fig-mono">0.2500</text>
<text x="70" y="238" text-anchor="middle" class="fig-small">layer value 4</text>
<rect x="128" y="180.0" width="44" height="40.0" class="fill-b"/>
<text x="150" y="174.0" text-anchor="middle" class="fig-small fig-mono">0.2500</text>
<text x="150" y="238" text-anchor="middle" class="fig-small">layer value 8</text>
<rect x="208" y="140.0" width="44" height="80.0" class="fill-b"/>
<text x="230" y="134.0" text-anchor="middle" class="fig-small fig-mono">0.5000</text>
<text x="230" y="238" text-anchor="middle" class="fig-small">layer value 12</text>
<text x="340" y="22" class="fig-bold">raw scores as weights: mix 8.3178</text>
<line x1="340" y1="220" x2="570" y2="220" class="stroke-muted"/>
<rect x="348" y="220.0" width="44" height="0.0" class="fill-bad"/>
<text x="370" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.0000</text>
<text x="370" y="238" text-anchor="middle" class="fig-small">layer value 4</text>
<rect x="428" y="220.0" width="44" height="0.0" class="fill-bad"/>
<text x="450" y="214.0" text-anchor="middle" class="fig-small fig-mono">0.0000</text>
<text x="450" y="238" text-anchor="middle" class="fig-small">layer value 8</text>
<rect x="508" y="109.1" width="44" height="110.9" class="fill-bad"/>
<text x="530" y="103.1" text-anchor="middle" class="fig-small fig-mono">0.6931</text>
<text x="530" y="238" text-anchor="middle" class="fig-small">layer value 12</text>
<text x="300" y="276" text-anchor="middle" class="fig-small">bar height is the weight on the layer; the weights of the softmax sum to 1</text>
</svg>
Figure 3. The same three layers mixed two ways. The softmax weights are 0.25, 0.25, and 0.5 and give 9. The raw scores 0, 0, and 0.6931 give 8.3178, and they don't sum to 1, so the mix is not an average of the layers.
```

The following table copies rows from Table I of Chen et al., "WavLM," 2022. The first three rows equal Table 2 of Yang et al., 2021. PR is phoneme error rate, ASR is word error rate without a language model, and SD is diarization error rate, in percent, lower is better. SID is speaker accuracy and ER is emotion accuracy, in percent, higher is better.

| Upstream | PR | ASR | SID | ER | SD |
|---|---|---|---|---|---|
| FBANK | 82.01 | 23.18 | 8.5E-4 | 35.39 | 10.05 |
| wav2vec 2.0 Base | 5.74 | 6.43 | 75.18 | 63.43 | 6.08 |
| HuBERT Base | 5.41 | 6.42 | 81.42 | 64.92 | 5.88 |
| WavLM Base | 4.84 | 6.21 | 84.51 | 65.94 | 4.55 |

The filterbank baseline does poorly on PR, SID, and ER, so the gains come from pretraining. WavLM Base gains most over HuBERT Base on SID (3.09 points) and SD, where speaker identity and overlap matter, which fits its denoising task. Each cell is one published run without error bars.

> [!KEY]
> What you now know: you can teach an encoder to ignore a second voice by mixing one into a span and keeping the clean utterance's labels, and you can rank encoders by freezing them and training only a softmax-weighted layer sum and a light head.

> [!RECAP]
> - WavLM mixes a scaled second signal into a span so that the span's energy sits a chosen number of decibels below the primary's, and scores it against the clean primary's labels.
> - Its relative bias uses 320 sign-keeping buckets, and a gate computed from each query scales the whole row.
> - SUPERB feeds a softmax-weighted sum of frozen layers to a light downstream model, and its tables compare encoders across tasks.

## Build it step by step

The checks test the functions in this order.

1. **`mix_for_denoising(primary, secondary, start, length, gain_db)`.** Returns a new array with only the span changed. Common mistake: dropping the square root.
2. **`bucket_relative_position(rel, n_buckets, max_distance)`.** Bucket `abs(rel)`, then add `n_buckets // 2` where `rel > 0`. Common mistake: losing the sign.
3. **`gated_relative_bias(queries, u, v, w, bias)`.** One scale per query row. Common mistake: scaling columns, so the gate follows the key.
4. **`superb_weighted_sum(layers, weights)`.** Softmax over the layer axis, then `np.tensordot`. Common mistake: using the raw weights.

## The exercise

You build the four functions above, on small arrays, on the CPU. The checks confirm the worked WavLM mix, the energy ratio inside the span for several gains, the 8-bucket table and the full 320-bucket map, the gated biases 3.5 and 3.79, and the SUPERB mix of 9, and they compare the mix of five random layers with `torch`. The **Run** prints the same worked examples.

## Further reading

- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP, 2022 (https://arxiv.org/abs/2110.13900). Algorithm 1 is the mixing.
- Yang et al., "SUPERB: Speech processing Universal PERformance Benchmark," Interspeech, 2021 (https://arxiv.org/abs/2105.01051).
- Baevski et al., "data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language," ICML, 2022 (https://arxiv.org/abs/2202.03555).
- Chung et al., "W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training," ASRU, 2021 (https://arxiv.org/abs/2108.06209).
