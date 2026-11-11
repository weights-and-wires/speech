---
title: An ASR training recipe
slug: 047-asr-recipe
part: "Part 4 — Neural speech recognition"
summary: Train byte-pair merges, perturb speed, mix noise at a chosen SNR, bucket batches, and set a warmup learning rate.
minutes: 45
gpu: false
objectives:
  - Train a byte-pair encoder and explain how vocabulary size, frame rate, and CTC's length limit fit together.
  - Perturb speed by resampling and mix noise at a chosen signal-to-noise ratio.
  - Pack utterances into length-bucketed batches and compute the padding waste.
  - Compute a warmup plus inverse-square-root learning rate, and average checkpoints.
prereqs:
  - 002-sampling-and-aliasing
  - 028-neural-acoustic-models
  - 029-ctc-many-paths-one-label
  - 037-conformers-and-specaugment
exercise: 047-asr-recipe
---

# An ASR training recipe

A Conformer and a CTC loss are not yet a training run. You fit them on slow, clean readings. A caller then talks fast over a bus, and the transcript comes back empty. The run never included a fast voice, a noisy one, or a learning rate the weights could follow.

This chapter is that recipe. You build a vocabulary that fits the frames, make speed and noise copies of each clip, pack batches so padding stays small, and warm the learning rate up before it falls. [CS224S](https://web.stanford.edu/class/cs224s/) homework 3 is a neural recognizer on conversational speech, which is the job these pieces are for.

## What you already know

- Resampling by an integer ratio `up/down` is the polyphase filter in [Sampling and aliasing](/lessons/002-sampling-and-aliasing), and it matches `scipy.signal.resample_poly`.
- A 4× front end turns 100 frames, one per 10 ms, into 24 frames, from [Neural acoustic models](/lessons/028-neural-acoustic-models).
- CTC needs at least one frame per label, plus one blank between identical neighbors, from [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label).
- SpecAugment hides bands and time spans of a log-mel spectrogram, from [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment).

## Merge the most frequent symbol pair

Picture each word as a row of letter tiles. The two tiles that sit side by side most often fuse into one tile, and that tile counts as one symbol the next time you look.

Byte-pair encoding is that process ([Sennrich, Haddow, and Birch](https://arxiv.org/abs/1508.07909)). Start from characters. Count every adjacent pair of the symbols you have now, and replace the most frequent pair with their concatenation. The vocabulary size is the character count you started from, plus the number of merges.

Take the corpus `cat`, `cat`, `car`, `car`, `card`, each string once. The first count is the following:

| Pair | Count |
|---|---|
| `(c, a)` | 5 |
| `(a, r)` | 3 |
| `(a, t)` | 2 |
| `(r, d)` | 1 |

You merge `(c, a)`. The strings are now `ca t`, `ca t`, `ca r`, `ca r`, and `ca r d`. The second count is `(ca, r)` three times, `(ca, t)` twice, and `(r, d)` once, so you merge `(ca, r)`. The third count is `(ca, t)` twice and `(car, d)` once, so you merge `(ca, t)`. The merge table is `(c, a)`, then `(ca, r)`, then `(ca, t)`.

A count that stays on the original characters still ranks `(a, r)` second. `a` already sits inside `ca`, so that list encodes `card` as `ca r d` and stops there.

To encode, apply the merges in order, from the left, and join a pair only when both symbols are still separate. `c a r d` becomes `ca r d`, then `car d`. Decoding joins the symbols and restores `card`. icefall stores its 500-piece vocabulary as the SentencePiece model `data/lang_bpe_500/bpe.model`. These strings have no spaces, so joining is the inverse.

When two pairs tie, keep the lexicographically smaller `(left, right)`. This corpus has no tie. The count, in code, is the following:

```python
counts = Counter()
for seq in sequences:
    for left, right in zip(seq, seq[1:]):
        counts[(left, right)] += 1
pair = min(counts, key=lambda item: (-counts[item], item))
```

> [!KEY]
> Count pairs of the symbols you have now. A pair taken from the original character tokens is a different merge, and it encodes a different string.

## Fit the vocabulary to the frame rate

A larger vocabulary stores longer pieces, so the label string gets shorter, and CTC can spend the saved frames on the audio. A transcript with $L$ labels and $r$ adjacent repeats needs

$$
T \ge L + r
$$

and $T$ is the length after the front end. [CS224S lecture 10](https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus) covers CTC and end-to-end recognition. The limit is the one from [CTC: many paths, one label](/lessons/029-ctc-many-paths-one-label).

Take 200 ms of audio, which is 20 frames at 10 ms. The 4× front end does `(20 - 3) // 2 + 1 = 9`, then `(9 - 3) // 2 + 1 = 4`. You have 4 encoder frames.

The characters of `speech` are `s p e e c h`. That is $L = 6$, and the double `e` is one repeat, so the transcript needs 7 frames. Four frames are not enough, and the CTC loss is infinite. The three symbols `sp`, `ee`, and `ch` have $L = 3$ and no identical neighbors, so they need 3 frames. Four frames cover them.

The [icefall Conformer CTC recipe](https://icefall.readthedocs.io/en/latest/recipes/Non-streaming-ASR/librispeech/conformer_ctc.html) uses that 500-piece vocabulary. The [ESPnet LibriSpeech results](https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/README.md) use 5,000 pieces (`bpe5000` in the run names). A few hundred to a few thousand pieces keep an English label inside the frames. A character vocabulary leaves a long string, so a fast talker can overflow the 4× front end.

## Change the speed and mix noise

[Ko, Peddinti, Povey, and Khudanpur](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html) keep three copies of each recording, at speed factors 0.9, 1.0, and 1.1, and report an average relative improvement of 4.3% across four tasks. They resample with SoX `speed`. The icefall recipe uses 0.9 and 1.1, so 960 hours become 2,880 hours. A factor $\alpha$ replaces $x(t)$ with $x(\alpha t)$. The Fourier transform of that warp is

$$
x(\alpha t) \quad \longleftrightarrow \quad \frac{1}{|\alpha|} \hat{x}(\omega / \alpha)
$$

For $\alpha > 1$ the clip is shorter and every frequency is multiplied by $\alpha$. For $\alpha < 1$ the clip is longer and the frequencies fall. Write $\alpha$ as a reduced ratio $p/q$, so 1.1 is $11/10$ and 0.9 is $9/10$. The new length is $\lceil N q / p \rceil$ samples, which is what `scipy.signal.resample_poly(x, q, p)` returns. That is the resampler from [Sampling and aliasing](/lessons/002-sampling-and-aliasing). [NeMo's speed perturbation](https://docs.nvidia.com/nemo/speech/latest/_modules/nemo/collections/asr/parts/preprocessing/perturb.html) resamples too, does not preserve pitch, and defaults to rates from 0.9 to 1.1.

A 400 Hz sine, one second at 16 kHz, shows the two factors. At 1.1 the length is $\lceil 16000 \times 10 / 11 \rceil = 14546$ samples and the peak moves to 440 Hz. At 0.9 the length is $\lceil 16000 \times 10 / 9 \rceil = 17778$ samples and the peak moves to 360 Hz.

Noise is the same kind of choice. Let $P_s$ be the mean square of the clean clip and $P_n$ the mean square of the noise. The gain that hits a chosen signal-to-noise ratio, in decibels, is

$$
g = \sqrt{\frac{P_s}{P_n \, 10^{\mathrm{SNR}/10}}}
$$

The mixture is the clean clip plus $g$ times the noise. Ten decibels is a tenfold power ratio, because a decibel is ten times a base-10 log of power. At 10 dB with both powers equal to 1, $g = 1/\sqrt{10} \approx 0.3162$. [Speech enhancement](/lessons/069-speech-enhancement) uses this same gain.

> [!WARNING]
> The 10 in the exponent is the power rule. `10 ** (snr_db / 20)` treats the decibel value as an amplitude and returns the wrong gain.

The recipe also masks the spectrogram, with the mask you built in [Conformers and SpecAugment](/lessons/037-conformers-and-specaugment). The [ESPnet Conformer config](https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_conformer.yaml) turns it on as `specaug: specaug`, two frequency masks up to 30 bins and two time masks up to 40 frames, and only while training.

## Pack similar lengths into one batch

A batch pads every utterance out to the longest one in it, and that padded count is what the step stores. With $n_b$ utterances and a longest length $L_b$, you store $n_b L_b$ frames. The waste fraction is

$$
W = 1 - \frac{\sum_i \ell_i}{\sum_b n_b L_b}
$$

Sort by length, then keep adding the next utterance while $n L$ is at most `max_frames`. An utterance longer than the budget is its own batch. [NeMo](https://docs.nvidia.com/nemo/speech/latest/asr/datasets.html) uses dynamic bucketing. icefall's `--max-duration` budgets the sum of true durations before padding, and padding still makes the step larger. The cap in this function is the padded size. ESPnet writes the same idea as `batch_bins: 30000000`.

Take lengths 40, 10, 42, 11, 100, and 12, with `max_frames = 120`. Sorted, they are 10, 11, 12, 40, 42, 100.

- 10, 11, and 12 pad to $3 \times 12 = 36$. Adding 40 would pad to $4 \times 40 = 160$, which is over 120.
- 40 and 42 pad to $2 \times 42 = 84$. Adding 100 would pad to $3 \times 100 = 300$.
- 100 stands alone.

Real frames sum to 215. Padded frames sum to $36 + 84 + 100 = 220$. The waste is $5/220$. Packing in the given order instead, without sorting, uses four batches and pads 276 frames, so the waste is $61/276$.

## Warm up, then decay

A large rate on the first step kicks randomly initialized weights away from a useful region. A rate that never falls keeps stepping past the minimum. The schedule in [Vaswani et al.](https://arxiv.org/html/1706.03762) rises in a straight line for `warmup` steps, then falls as the inverse square root of the step. They use 4,000 warmup steps. Figure 1 is that curve for $d_{\mathrm{model}} = 256$.

```figure
<svg viewBox="0 0 640 280" role="img" aria-label="Learning rate versus step. The curve rises through a warmup to a peak at step 4000, then falls to half that peak at step 16000.">
  <rect x="88.0" y="28.0" width="131.0" height="204.0" class="fill-soft-a"/>
  <line x1="88.0" y1="232.0" x2="612.0" y2="232.0" class="stroke-grid"/>
  <text x="80.0" y="236.0" class="fig-small" text-anchor="end">0</text>
  <line x1="88.0" y1="147.0" x2="612.0" y2="147.0" class="stroke-grid"/>
  <text x="80.0" y="151.0" class="fig-small" text-anchor="end">0.0005</text>
  <line x1="88.0" y1="62.0" x2="612.0" y2="62.0" class="stroke-grid"/>
  <text x="80.0" y="66.0" class="fig-small" text-anchor="end">0.001</text>
  <line x1="88.0" y1="28.0" x2="88.0" y2="232.0" class="stroke-muted"/>
  <line x1="88.0" y1="232.0" x2="612.0" y2="232.0" class="stroke-muted"/>
  <line x1="219.0" y1="28.0" x2="219.0" y2="232.0" class="stroke-muted dashed"/>
  <polyline points="88.0,232.0 94.6,223.6 101.1,215.2 107.7,206.8 114.2,198.4 120.8,190.0 127.3,181.6 133.9,173.2 140.4,164.8 147.0,156.4 153.5,148.0 160.1,139.6 166.6,131.2 173.2,122.8 179.7,114.4 186.3,106.0 192.8,97.6 199.4,89.2 205.9,80.8 212.5,72.4 219.0,64.0 225.6,68.1 232.1,71.8 238.7,75.4 245.2,78.7 251.8,81.8 258.3,84.7 264.9,87.4 271.4,90.0 278.0,92.5 284.5,94.8 291.1,97.1 297.6,99.2 304.2,101.2 310.7,103.2 317.3,105.0 323.8,106.8 330.4,108.5 336.9,110.1 343.5,111.7 350.0,113.2 356.6,114.7 363.1,116.1 369.7,117.4 376.2,118.7 382.8,120.0 389.3,121.2 395.9,122.4 402.4,123.6 409.0,124.7 415.5,125.8 422.1,126.8 428.6,127.8 435.2,128.8 441.7,129.8 448.3,130.7 454.8,131.6 461.4,132.5 467.9,133.4 474.5,134.2 481.0,135.0 487.6,135.8 494.1,136.6 500.7,137.3 507.2,138.1 513.8,138.8 520.3,139.5 526.9,140.2 533.4,140.9 540.0,141.6 546.5,142.2 553.1,142.8 559.6,143.5 566.2,144.1 572.7,144.7 579.3,145.2 585.8,145.8 592.4,146.4 598.9,146.9 605.5,147.5 612.0,148.0" class="stroke-b" fill="none"/>
  <circle cx="219.0" cy="64.0" r="4.5" class="fill-b"/>
  <circle cx="612.0" cy="148.0" r="4.5" class="fill-a"/>
  <text x="88.0" y="250.0" class="fig-small" text-anchor="middle">0</text>
  <text x="219.0" y="250.0" class="fig-small" text-anchor="middle">4000</text>
  <text x="350.0" y="250.0" class="fig-small" text-anchor="middle">8000</text>
  <text x="481.0" y="250.0" class="fig-small" text-anchor="middle">12000</text>
  <text x="612.0" y="250.0" class="fig-small" text-anchor="middle">16000</text>
  <text x="153.5" y="44.0" class="fig-small">warmup</text>
  <text x="382.8" y="54.0" class="fig-small">inverse square root</text>
  <text x="16" y="16" class="fig-small">learning rate</text>
  <text x="350.0" y="268" class="fig-small" text-anchor="middle">step</text>
</svg>
Figure 1. The rate climbs in a straight line through the warmup and peaks at step 4,000. At step 16,000, four times the warmup, the inverse-square-root branch has cut the rate in half.
```

For $d_{\mathrm{model}} = 256$ and a warmup of 4,000, step 100 is still on the ramp, at $2.471 \times 10^{-5}$. The peak, at step 4,000, is $9.882 \times 10^{-4}$. Step 16,000 is $4.941 \times 10^{-4}$, half the peak, because $\sqrt{4} = 2$. Keeping only $s^{-0.5}$ gives 0.00625 at step 100, about 250 times the ramp.

The closed form, with step $s \ge 1$ and warmup $w$, is

$$
\mathrm{lr}(s) = d_{\mathrm{model}}^{-0.5} \cdot \min\left(s^{-0.5},\, s \cdot w^{-1.5}\right)
$$

The first term inside the minimum is the decay. The second is the ramp, and it is the smaller one until $s = w$. In code:

```python
d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)
```

The ESPnet config names its scheduler `warmuplr` and sets `warmup_steps: 25000` with `lr: 0.001`. A cosine schedule is the other shape in wide use: a linear warmup, then a cosine fall toward a small floor. The checks use the inverse-square-root form.

> [!WARNING]
> Skipping the warmup leaves the inverse-square-root term in force at step 1. That rate is far above the ramp, and the first updates are too large.

## Average the checkpoints

Late checkpoints wander around a minimum, and their coordinate-wise mean sits in the middle of that wander. For one parameter the average of $W_1, \ldots, W_K$ is

$$
\bar{W} = \frac{1}{K} \sum_{k=1}^{K} W_k
$$

Two vectors $[1, 3]$ and $[5, 7]$ average to $[3, 5]$. Each key of the state dict has its own mean, and the keys have to match. [icefall's streaming decode](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/streaming_decode.py) loads that mean over `--avg` consecutive checkpoints ending at `--epoch`. The Conformer recipe's example passes `--avg 1`, one checkpoint. ESPnet decode names that contain `valid.acc.ave` average the checkpoints `keep_nbest_models: 10` retains.

## Read the training curve

Read the two losses together. The following table is a toy log for practice. It is a made-up run, separate from any published word error.

| Step | Training loss | Validation loss |
|---|---|---|
| 200 | 3.8 | 3.9 |
| 800 | 1.6 | 1.9 |
| 3000 | 0.4 | 2.7 |

From step 200 to step 800 both losses fall, so the model is fitting. From step 800 to step 3000 the training loss falls and the validation loss rises, so the model is overfitting. That is when the speed copies, the noise, and the SpecAugment mask matter, and when you average checkpoints from the valley. If both losses had stayed near 3.8, the run would be underfitting, because the learning rate is too small or the model is too small.

## Cost and scaling

The step's time and memory follow $\sum_b n_b L_b$, the padded frame count. Sorting by length is what keeps that close to the sum of the true lengths. One utterance per batch wastes no padding and also leaves the device idle.

## Where this breaks

Speed perturbation moves pitch on purpose, so leave it out when pitch is the label. One harsh signal-to-noise ratio can bury the speech, so draw the ratio from a range and keep this gain. Averaging an early checkpoint with a late one pulls the weights toward the initialization. Average the late checkpoints `--avg` selects.

> [!RECAP]
> - Merge the most frequent pair of current symbols, then recount.
> - A shorter label string is how a large vocabulary fits CTC's frame limit.
> - Speed perturbation resamples. The noise gain uses $10^{\mathrm{SNR}/10}$.
> - Sort by length, and cap $n L$ by the frame budget.
> - Warm the learning rate up, then decay it, and average the late checkpoints.

## Build it step by step

Build the functions in the order the checks call them.

1. **`train_bpe(corpus, n_merges)`** returns the merges in order. Split each string into characters, count adjacent pairs, and break ties with `min` on `(-count, pair)`. Ranking the original characters merges `(a, r)` second on the worked corpus.
2. **`encode(text, merges)`** returns the symbols, applying each merge from the left without overlap. **`decode(symbols)`** joins them. Applying a later merge before its symbol exists leaves `card` split.
3. **`speed_perturb(x, factor)`** returns a 1-D array. For `factor = p/q` in lowest terms, call `resample_poly(x, q, p)`. The length is `ceil(len(x) * q / p)`. A factor above 1 shortens the clip and multiplies frequencies by that factor. Multiplying the length by the factor slows the clip down.
4. **`mix_at_snr(clean, noise, snr_db)`** returns the mixture and the scaled noise. Both inputs are 1-D and the same length. The exponent in the gain is 10.
5. **`bucket_batches(lengths, max_frames)`** returns lists of indices into `lengths`. Sort by length, and start a batch when the next utterance would push `n * longest` past the budget. Packing the worked lengths in the given order wastes $61/276$, above $5/220$.
6. **`noam_lr(step, d_model, warmup)`** returns one float for `step >= 1`. Returning only `d_model ** -0.5 * step ** -0.5` skips the warmup.
7. **`average_checkpoints(state_dicts)`** returns one dict. Average each key with `sum(arrays) / len(state_dicts)`. Forgetting the division, or averaging only the first and last checkpoint, misses the mean.

## The exercise

You build `train_bpe`, `encode`, `decode`, `speed_perturb`, `mix_at_snr`, `bucket_batches`, `noam_lr`, and `average_checkpoints`. Press **Run** to hear a 400 Hz sine, the same sine at speed 1.1, and the sine mixed at 10 dB, and to plot the learning-rate curve.

The checks match the merge table and the encoding of `card`, the perturbed length and peak frequency, and the mixture's decibel ratio. They hold the buckets to a waste of at most $5/220$, the learning rate to the closed form, and the averaged checkpoint to the exact mean. Every check runs on the CPU. The ESPnet config asks for four 32 GB GPUs to train the full Conformer. The checks pass without one.

> [!KEY]
> The fast, noisy call fails when the training run never built a vocabulary that fits the frames, a batch that fits memory, or a learning rate the weights can follow. The recipe is those choices, and the curve tells you whether they held.
