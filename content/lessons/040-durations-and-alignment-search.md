---
title: Durations and alignment search
slug: 040-durations-and-alignment-search
part: "Part 5 — Speech synthesis"
summary: Replace attention with durations, expand phones to frames, add pitch and energy targets, and find alignments with a dynamic program.
minutes: 55
gpu: false
objectives:
  - Expand a padded batch of phone vectors to frames with the FastSpeech length regulator, and change speaking rate with a speed factor.
  - Compute per-phone pitch and energy targets, and quantize them into the 256 bins FastSpeech 2 embeds.
  - Derive monotonic alignment search as a dynamic program, and verify it against brute-force enumeration.
  - Explain where durations come from in FastSpeech, FastSpeech 2, and Glow-TTS.
prereqs:
  - 039-text-to-speech-and-tacotron
  - 020-dynamic-time-warping
  - 017-pitch-tracking
  - 007-the-short-time-fourier-transform
exercise: 040-durations-and-alignment-search
---

# Durations and alignment search

Tacotron learns when to read each symbol with attention, and attention is fragile. It skips words, repeats them, and stops early, most of all on long sentences. It also forces decoding into a loop of one frame at a time.

FastSpeech removes attention by making timing explicit. A small network predicts how many frames each phone lasts, its *duration*. Then each phone's vector is copied that many times, and every frame is predicted at once, in parallel. The reported mel-spectrogram speedup over an autoregressive teacher was 270 times.

That moves the hard problem to training: where do the true durations come from? This chapter builds the length regulator that uses durations, the pitch and energy targets that FastSpeech 2 adds beside them, and monotonic alignment search, the dynamic program with which Glow-TTS and VITS find durations on their own.

## What you already know

- Tacotron's attention decides which symbol each output frame reads, and it can skip or repeat (see [Text to speech and Tacotron](/lessons/039-text-to-speech-and-tacotron)).
- Dynamic time warping fills a table of best partial costs and backtracks the best path (see [Dynamic time warping](/lessons/020-dynamic-time-warping)).
- YIN gives an F0 estimate and a voicing decision for each frame (see [Pitch tracking](/lessons/017-pitch-tracking)).
- A diagonal Gaussian's log-density is a sum over dimensions of $-\tfrac12 \log 2\pi - \log\sigma - (z - \mu)^2 / 2\sigma^2$.
- An STFT magnitude column describes one frame's spectrum (see [The short-time Fourier transform](/lessons/007-the-short-time-fourier-transform)).

## The length regulator

The length regulator is a photocopier. Given phone vectors $h_1, \ldots, h_L$ and whole-number durations $d_1, \ldots, d_L$, it outputs $h_1$ repeated $d_1$ times, then $h_2$ repeated $d_2$ times, and so on. The output has $N = \sum_i d_i$ frames.

It helps to write the copy as a matrix. The *alignment matrix* $A \in \{0, 1\}^{L \times N}$ has a 1 where frame $n$ belongs to phone $i$:

$$
A_{i n} = \mathbb{1}\left[\,e_i - d_i \le n < e_i\,\right], \qquad
e_i = \sum_{k=1}^{i} d_k, \qquad
\tilde H = A^\top H.
$$

Read it aloud: $e_i$ is the cumulative sum of durations, the frame where phone $i$ ends. Phone $i$ owns the frames from its start, $e_i - d_i$, up to but not including its end. Multiplying by $A^\top$ copies each phone's row of $H$ into the frames it owns. In torch, for a batch:

```python
ends = torch.cumsum(durations, dim=1)
A = ((ends - durations)[..., None] <= n) & (n < ends[..., None])
frames = torch.einsum("bln,bld->bnd", A.float(), h)
```

### Work one example by hand

Take three phones with durations $d = [2, 1, 3]$. The cumulative sums are $e = [2, 3, 6]$, so the starts are $[0, 2, 3]$ and $N = 6$. Phone 1 owns frames 0 and 1, phone 2 owns frame 2, and phone 3 owns frames 3 to 5, so $\tilde H = [h_1, h_1, h_2, h_3, h_3, h_3]$.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="The alignment matrix for three phones with durations 2, 1, 3: phone 1 owns frames 0 and 1, phone 2 owns frame 2, phone 3 owns frames 3 to 5">
<defs><marker id="c039-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="156.0" y="36.0" text-anchor="middle" class="fig-small">0</text>
<text x="212.0" y="36.0" text-anchor="middle" class="fig-small">1</text>
<text x="268.0" y="36.0" text-anchor="middle" class="fig-small">2</text>
<text x="324.0" y="36.0" text-anchor="middle" class="fig-small">3</text>
<text x="380.0" y="36.0" text-anchor="middle" class="fig-small">4</text>
<text x="436.0" y="36.0" text-anchor="middle" class="fig-small">5</text>
<text x="120.0" y="22.0" text-anchor="end" class="fig-small">frame n</text>
<text x="118.0" y="75.0" text-anchor="end" class="fig-small">phone 1, d = 2</text>
<rect x="130.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="156.0" y="75.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="186.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="212.0" y="75.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="242.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="268.0" y="75.0" text-anchor="middle">0</text>
<rect x="298.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="324.0" y="75.0" text-anchor="middle">0</text>
<rect x="354.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="380.0" y="75.0" text-anchor="middle">0</text>
<rect x="410.0" y="50.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="436.0" y="75.0" text-anchor="middle">0</text>
<text x="118.0" y="119.0" text-anchor="end" class="fig-small">phone 2, d = 1</text>
<rect x="130.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="156.0" y="119.0" text-anchor="middle">0</text>
<rect x="186.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="212.0" y="119.0" text-anchor="middle">0</text>
<rect x="242.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="268.0" y="119.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="298.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="324.0" y="119.0" text-anchor="middle">0</text>
<rect x="354.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="380.0" y="119.0" text-anchor="middle">0</text>
<rect x="410.0" y="94.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="436.0" y="119.0" text-anchor="middle">0</text>
<text x="118.0" y="163.0" text-anchor="end" class="fig-small">phone 3, d = 3</text>
<rect x="130.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="156.0" y="163.0" text-anchor="middle">0</text>
<rect x="186.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="212.0" y="163.0" text-anchor="middle">0</text>
<rect x="242.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="268.0" y="163.0" text-anchor="middle">0</text>
<rect x="298.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="324.0" y="163.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="354.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="380.0" y="163.0" text-anchor="middle" class="fig-bold">1</text>
<rect x="410.0" y="138.0" width="52.0" height="40.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="436.0" y="163.0" text-anchor="middle" class="fig-bold">1</text>
<text x="118.0" y="208.0" text-anchor="end" class="fig-small fig-bold">output</text>
<rect x="130.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="156.0" y="212.0" text-anchor="middle" class="fig-bold">h1</text>
<rect x="186.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="212.0" y="212.0" text-anchor="middle" class="fig-bold">h1</text>
<rect x="242.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="268.0" y="212.0" text-anchor="middle" class="fig-bold">h2</text>
<rect x="298.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="324.0" y="212.0" text-anchor="middle" class="fig-bold">h3</text>
<rect x="354.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="380.0" y="212.0" text-anchor="middle" class="fig-bold">h3</text>
<rect x="410.0" y="190.0" width="52.0" height="34.0" rx="4" class="fill-soft-a stroke-a"/>
<text x="436.0" y="212.0" text-anchor="middle" class="fig-bold">h3</text>
<text x="300.0" y="246.0" text-anchor="middle" class="fig-small">cumulative ends e = [2, 3, 6]; phone i owns frames e_i - d_i up to, not including, e_i</text>
</svg>
Figure 1. Row i of A has ones exactly where phone i owns a frame, so A transposed times H copies each phone's vector into its frames. A padded phone has duration 0 and a row of zeros.
```

The matrix form has two advantages over a loop of `repeat_interleave` calls. It works on a padded batch in one call, because a padded phone has duration 0 and owns no frames. And gradients flow back to $H$: phone $i$ receives the sum of the gradients of the $d_i$ frames that copied it.

> [!KEY]
> A duration model turns alignment into bookkeeping: cumulative sums of the durations say which frames each phone owns, and every frame can then be predicted in parallel.

### Durations at inference, and speed

A *duration predictor*, two 1-D convolutions and a linear layer, predicts one number per phone. It is trained with mean squared error on $\log(d + 1)$, because durations are skewed: most phones last a few frames, and a few last dozens. At inference you invert the log, scale by a speed factor $\alpha$, and round:

$$
\hat d_i = \max\left(1,\ \operatorname{round}\left(\alpha\,(e^{\hat \ell_i} - 1)\right)\right).
$$

Read it aloud: exponentiate the predicted log duration $\hat\ell_i$, subtract the 1 that the target added, and multiply by $\alpha$. Round to whole frames, and give every phone at least one frame so no phone vanishes. A factor $\alpha > 1$ slows speech down. For example, $d = [4, 1, 6]$ with $\alpha = 1.3$ gives $5.2, 1.3, 7.8$, which rounds to $[5, 1, 8]$.

## Where durations come from

The length regulator needs true durations during training. Three systems got them in three ways:

- **FastSpeech** trained an autoregressive Transformer TTS teacher first, picked its sharpest attention head, and counted how many decoder steps chose each phone as their argmax.
- **FastSpeech 2** took durations from the Montreal Forced Aligner, an HMM-based aligner that runs Viterbi over the known transcript, as in [Diarization and forced alignment](/lessons/038-diarization-and-forced-alignment). The paper found these more accurate than teacher attention, and dropping the teacher made training about three times faster.
- **Glow-TTS** and later **VITS** search for the alignment inside the model, during training, with monotonic alignment search.

## Pitch and energy targets

Durations remove one source of the one-to-many problem, but not all of them. The same phones with the same durations can still be said on a high or a low pitch, loud or soft, and a model trained with squared error on such targets predicts their average, which sounds flat. FastSpeech 2's *variance adaptor* gives the decoder a pitch value and an energy value per position instead, true ones in training and predicted ones at inference.

The targets come from the recording. For **pitch**, estimate F0 per frame with a pitch tracker, skip the frames without voicing, and average the voiced frames of each phone. **Energy** of frame $t$ is the L2 norm of its STFT magnitude column, $E_t = \sqrt{\sum_k |X_{k t}|^2}$, averaged per phone over all its frames.

Before the decoder sees a pitch or energy value, FastSpeech 2 turns it into one of 256 bins and looks up a learned embedding. Pitch bins are spaced evenly on a log scale, because pitch is heard as ratios, and energy bins are spaced evenly. With $K$ bins there are $K - 1$ edges $b_0 < \cdots < b_{K-2}$ from $v_{\min}$ to $v_{\max}$, and a value's bin is the number of edges below it:

$$
b_k = \exp\left(\log v_{\min} + \frac{k}{K-2}\left(\log v_{\max} - \log v_{\min}\right)\right), \qquad
\operatorname{bin}(v) = \left|\{k : b_k < v\}\right|.
$$

Read it aloud: the edges run from $v_{\min}$ to $v_{\max}$ in equal steps of the log. A value at or below the first edge lands in bin 0, and one above the last edge in bin $K - 1$. In numpy:

```python
edges = np.exp(np.linspace(np.log(v_min), np.log(v_max), n_bins - 1))
bins = np.searchsorted(edges, values, side="left")
```

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="Pitch bins between 71 Hz and 795 Hz are evenly spaced on a log scale; 200 Hz falls in bin 109">
<defs><marker id="c039-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<line x1="40.0" y1="120.0" x2="560.0" y2="120.0" class="stroke-muted"/>
<line x1="40.0" y1="115.0" x2="40.0" y2="125.0" class="stroke-grid"/>
<line x1="44.1" y1="115.0" x2="44.1" y2="125.0" class="stroke-grid"/>
<line x1="48.2" y1="115.0" x2="48.2" y2="125.0" class="stroke-grid"/>
<line x1="52.3" y1="115.0" x2="52.3" y2="125.0" class="stroke-grid"/>
<line x1="56.4" y1="115.0" x2="56.4" y2="125.0" class="stroke-grid"/>
<line x1="60.5" y1="115.0" x2="60.5" y2="125.0" class="stroke-grid"/>
<line x1="64.6" y1="115.0" x2="64.6" y2="125.0" class="stroke-grid"/>
<line x1="68.7" y1="115.0" x2="68.7" y2="125.0" class="stroke-grid"/>
<line x1="72.8" y1="115.0" x2="72.8" y2="125.0" class="stroke-grid"/>
<line x1="76.9" y1="115.0" x2="76.9" y2="125.0" class="stroke-grid"/>
<line x1="80.9" y1="115.0" x2="80.9" y2="125.0" class="stroke-grid"/>
<line x1="85.0" y1="115.0" x2="85.0" y2="125.0" class="stroke-grid"/>
<line x1="89.1" y1="115.0" x2="89.1" y2="125.0" class="stroke-grid"/>
<line x1="93.2" y1="115.0" x2="93.2" y2="125.0" class="stroke-grid"/>
<line x1="97.3" y1="115.0" x2="97.3" y2="125.0" class="stroke-grid"/>
<line x1="101.4" y1="115.0" x2="101.4" y2="125.0" class="stroke-grid"/>
<line x1="105.5" y1="115.0" x2="105.5" y2="125.0" class="stroke-grid"/>
<line x1="109.6" y1="115.0" x2="109.6" y2="125.0" class="stroke-grid"/>
<line x1="113.7" y1="115.0" x2="113.7" y2="125.0" class="stroke-grid"/>
<line x1="117.8" y1="115.0" x2="117.8" y2="125.0" class="stroke-grid"/>
<line x1="121.9" y1="115.0" x2="121.9" y2="125.0" class="stroke-grid"/>
<line x1="126.0" y1="115.0" x2="126.0" y2="125.0" class="stroke-grid"/>
<line x1="130.1" y1="115.0" x2="130.1" y2="125.0" class="stroke-grid"/>
<line x1="134.2" y1="115.0" x2="134.2" y2="125.0" class="stroke-grid"/>
<line x1="138.3" y1="115.0" x2="138.3" y2="125.0" class="stroke-grid"/>
<line x1="142.4" y1="115.0" x2="142.4" y2="125.0" class="stroke-grid"/>
<line x1="146.5" y1="115.0" x2="146.5" y2="125.0" class="stroke-grid"/>
<line x1="150.6" y1="115.0" x2="150.6" y2="125.0" class="stroke-grid"/>
<line x1="154.6" y1="115.0" x2="154.6" y2="125.0" class="stroke-grid"/>
<line x1="158.7" y1="115.0" x2="158.7" y2="125.0" class="stroke-grid"/>
<line x1="162.8" y1="115.0" x2="162.8" y2="125.0" class="stroke-grid"/>
<line x1="166.9" y1="115.0" x2="166.9" y2="125.0" class="stroke-grid"/>
<line x1="171.0" y1="115.0" x2="171.0" y2="125.0" class="stroke-grid"/>
<line x1="175.1" y1="115.0" x2="175.1" y2="125.0" class="stroke-grid"/>
<line x1="179.2" y1="115.0" x2="179.2" y2="125.0" class="stroke-grid"/>
<line x1="183.3" y1="115.0" x2="183.3" y2="125.0" class="stroke-grid"/>
<line x1="187.4" y1="115.0" x2="187.4" y2="125.0" class="stroke-grid"/>
<line x1="191.5" y1="115.0" x2="191.5" y2="125.0" class="stroke-grid"/>
<line x1="195.6" y1="115.0" x2="195.6" y2="125.0" class="stroke-grid"/>
<line x1="199.7" y1="115.0" x2="199.7" y2="125.0" class="stroke-grid"/>
<line x1="203.8" y1="115.0" x2="203.8" y2="125.0" class="stroke-grid"/>
<line x1="207.9" y1="115.0" x2="207.9" y2="125.0" class="stroke-grid"/>
<line x1="212.0" y1="115.0" x2="212.0" y2="125.0" class="stroke-grid"/>
<line x1="216.1" y1="115.0" x2="216.1" y2="125.0" class="stroke-grid"/>
<line x1="220.2" y1="115.0" x2="220.2" y2="125.0" class="stroke-grid"/>
<line x1="224.3" y1="115.0" x2="224.3" y2="125.0" class="stroke-grid"/>
<line x1="228.3" y1="115.0" x2="228.3" y2="125.0" class="stroke-grid"/>
<line x1="232.4" y1="115.0" x2="232.4" y2="125.0" class="stroke-grid"/>
<line x1="236.5" y1="115.0" x2="236.5" y2="125.0" class="stroke-grid"/>
<line x1="240.6" y1="115.0" x2="240.6" y2="125.0" class="stroke-grid"/>
<line x1="244.7" y1="115.0" x2="244.7" y2="125.0" class="stroke-grid"/>
<line x1="248.8" y1="115.0" x2="248.8" y2="125.0" class="stroke-grid"/>
<line x1="252.9" y1="115.0" x2="252.9" y2="125.0" class="stroke-grid"/>
<line x1="257.0" y1="115.0" x2="257.0" y2="125.0" class="stroke-grid"/>
<line x1="261.1" y1="115.0" x2="261.1" y2="125.0" class="stroke-grid"/>
<line x1="265.2" y1="115.0" x2="265.2" y2="125.0" class="stroke-grid"/>
<line x1="269.3" y1="115.0" x2="269.3" y2="125.0" class="stroke-grid"/>
<line x1="273.4" y1="115.0" x2="273.4" y2="125.0" class="stroke-grid"/>
<line x1="277.5" y1="115.0" x2="277.5" y2="125.0" class="stroke-grid"/>
<line x1="281.6" y1="115.0" x2="281.6" y2="125.0" class="stroke-grid"/>
<line x1="285.7" y1="115.0" x2="285.7" y2="125.0" class="stroke-grid"/>
<line x1="289.8" y1="115.0" x2="289.8" y2="125.0" class="stroke-grid"/>
<line x1="293.9" y1="115.0" x2="293.9" y2="125.0" class="stroke-grid"/>
<line x1="298.0" y1="115.0" x2="298.0" y2="125.0" class="stroke-grid"/>
<line x1="302.0" y1="115.0" x2="302.0" y2="125.0" class="stroke-grid"/>
<line x1="306.1" y1="115.0" x2="306.1" y2="125.0" class="stroke-grid"/>
<line x1="310.2" y1="115.0" x2="310.2" y2="125.0" class="stroke-grid"/>
<line x1="314.3" y1="115.0" x2="314.3" y2="125.0" class="stroke-grid"/>
<line x1="318.4" y1="115.0" x2="318.4" y2="125.0" class="stroke-grid"/>
<line x1="322.5" y1="115.0" x2="322.5" y2="125.0" class="stroke-grid"/>
<line x1="326.6" y1="115.0" x2="326.6" y2="125.0" class="stroke-grid"/>
<line x1="330.7" y1="115.0" x2="330.7" y2="125.0" class="stroke-grid"/>
<line x1="334.8" y1="115.0" x2="334.8" y2="125.0" class="stroke-grid"/>
<line x1="338.9" y1="115.0" x2="338.9" y2="125.0" class="stroke-grid"/>
<line x1="343.0" y1="115.0" x2="343.0" y2="125.0" class="stroke-grid"/>
<line x1="347.1" y1="115.0" x2="347.1" y2="125.0" class="stroke-grid"/>
<line x1="351.2" y1="115.0" x2="351.2" y2="125.0" class="stroke-grid"/>
<line x1="355.3" y1="115.0" x2="355.3" y2="125.0" class="stroke-grid"/>
<line x1="359.4" y1="115.0" x2="359.4" y2="125.0" class="stroke-grid"/>
<line x1="363.5" y1="115.0" x2="363.5" y2="125.0" class="stroke-grid"/>
<line x1="367.6" y1="115.0" x2="367.6" y2="125.0" class="stroke-grid"/>
<line x1="371.7" y1="115.0" x2="371.7" y2="125.0" class="stroke-grid"/>
<line x1="375.7" y1="115.0" x2="375.7" y2="125.0" class="stroke-grid"/>
<line x1="379.8" y1="115.0" x2="379.8" y2="125.0" class="stroke-grid"/>
<line x1="383.9" y1="115.0" x2="383.9" y2="125.0" class="stroke-grid"/>
<line x1="388.0" y1="115.0" x2="388.0" y2="125.0" class="stroke-grid"/>
<line x1="392.1" y1="115.0" x2="392.1" y2="125.0" class="stroke-grid"/>
<line x1="396.2" y1="115.0" x2="396.2" y2="125.0" class="stroke-grid"/>
<line x1="400.3" y1="115.0" x2="400.3" y2="125.0" class="stroke-grid"/>
<line x1="404.4" y1="115.0" x2="404.4" y2="125.0" class="stroke-grid"/>
<line x1="408.5" y1="115.0" x2="408.5" y2="125.0" class="stroke-grid"/>
<line x1="412.6" y1="115.0" x2="412.6" y2="125.0" class="stroke-grid"/>
<line x1="416.7" y1="115.0" x2="416.7" y2="125.0" class="stroke-grid"/>
<line x1="420.8" y1="115.0" x2="420.8" y2="125.0" class="stroke-grid"/>
<line x1="424.9" y1="115.0" x2="424.9" y2="125.0" class="stroke-grid"/>
<line x1="429.0" y1="115.0" x2="429.0" y2="125.0" class="stroke-grid"/>
<line x1="433.1" y1="115.0" x2="433.1" y2="125.0" class="stroke-grid"/>
<line x1="437.2" y1="115.0" x2="437.2" y2="125.0" class="stroke-grid"/>
<line x1="441.3" y1="115.0" x2="441.3" y2="125.0" class="stroke-grid"/>
<line x1="445.4" y1="115.0" x2="445.4" y2="125.0" class="stroke-grid"/>
<line x1="449.4" y1="115.0" x2="449.4" y2="125.0" class="stroke-grid"/>
<line x1="453.5" y1="115.0" x2="453.5" y2="125.0" class="stroke-grid"/>
<line x1="457.6" y1="115.0" x2="457.6" y2="125.0" class="stroke-grid"/>
<line x1="461.7" y1="115.0" x2="461.7" y2="125.0" class="stroke-grid"/>
<line x1="465.8" y1="115.0" x2="465.8" y2="125.0" class="stroke-grid"/>
<line x1="469.9" y1="115.0" x2="469.9" y2="125.0" class="stroke-grid"/>
<line x1="474.0" y1="115.0" x2="474.0" y2="125.0" class="stroke-grid"/>
<line x1="478.1" y1="115.0" x2="478.1" y2="125.0" class="stroke-grid"/>
<line x1="482.2" y1="115.0" x2="482.2" y2="125.0" class="stroke-grid"/>
<line x1="486.3" y1="115.0" x2="486.3" y2="125.0" class="stroke-grid"/>
<line x1="490.4" y1="115.0" x2="490.4" y2="125.0" class="stroke-grid"/>
<line x1="494.5" y1="115.0" x2="494.5" y2="125.0" class="stroke-grid"/>
<line x1="498.6" y1="115.0" x2="498.6" y2="125.0" class="stroke-grid"/>
<line x1="502.7" y1="115.0" x2="502.7" y2="125.0" class="stroke-grid"/>
<line x1="506.8" y1="115.0" x2="506.8" y2="125.0" class="stroke-grid"/>
<line x1="510.9" y1="115.0" x2="510.9" y2="125.0" class="stroke-grid"/>
<line x1="515.0" y1="115.0" x2="515.0" y2="125.0" class="stroke-grid"/>
<line x1="519.1" y1="115.0" x2="519.1" y2="125.0" class="stroke-grid"/>
<line x1="523.1" y1="115.0" x2="523.1" y2="125.0" class="stroke-grid"/>
<line x1="527.2" y1="115.0" x2="527.2" y2="125.0" class="stroke-grid"/>
<line x1="531.3" y1="115.0" x2="531.3" y2="125.0" class="stroke-grid"/>
<line x1="535.4" y1="115.0" x2="535.4" y2="125.0" class="stroke-grid"/>
<line x1="539.5" y1="115.0" x2="539.5" y2="125.0" class="stroke-grid"/>
<line x1="543.6" y1="115.0" x2="543.6" y2="125.0" class="stroke-grid"/>
<line x1="547.7" y1="115.0" x2="547.7" y2="125.0" class="stroke-grid"/>
<line x1="551.8" y1="115.0" x2="551.8" y2="125.0" class="stroke-grid"/>
<line x1="555.9" y1="115.0" x2="555.9" y2="125.0" class="stroke-grid"/>
<line x1="560.0" y1="115.0" x2="560.0" y2="125.0" class="stroke-grid"/>
<line x1="40.0" y1="108.0" x2="40.0" y2="132.0" class="stroke-muted"/>
<text x="40.0" y="150.0" text-anchor="middle" class="fig-small">edge 0</text>
<text x="40.0" y="166.0" text-anchor="middle" class="fig-small">71 Hz</text>
<line x1="142.4" y1="108.0" x2="142.4" y2="132.0" class="stroke-muted"/>
<text x="142.4" y="150.0" text-anchor="middle" class="fig-small">edge 50</text>
<text x="142.4" y="166.0" text-anchor="middle" class="fig-small">114 Hz</text>
<line x1="244.7" y1="108.0" x2="244.7" y2="132.0" class="stroke-muted"/>
<text x="244.7" y="150.0" text-anchor="middle" class="fig-small">edge 100</text>
<text x="244.7" y="166.0" text-anchor="middle" class="fig-small">184 Hz</text>
<line x1="347.1" y1="108.0" x2="347.1" y2="132.0" class="stroke-muted"/>
<text x="347.1" y="150.0" text-anchor="middle" class="fig-small">edge 150</text>
<text x="347.1" y="166.0" text-anchor="middle" class="fig-small">296 Hz</text>
<line x1="449.4" y1="108.0" x2="449.4" y2="132.0" class="stroke-muted"/>
<text x="449.4" y="150.0" text-anchor="middle" class="fig-small">edge 200</text>
<text x="449.4" y="166.0" text-anchor="middle" class="fig-small">476 Hz</text>
<line x1="560.0" y1="108.0" x2="560.0" y2="132.0" class="stroke-muted"/>
<text x="560.0" y="150.0" text-anchor="middle" class="fig-small">edge 254</text>
<text x="560.0" y="166.0" text-anchor="middle" class="fig-small">795 Hz</text>
<line x1="262.9" y1="60.0" x2="262.9" y2="120.0" class="stroke-b"/>
<circle cx="262.9" cy="120.0" r="6" class="fill-b"/>
<text x="262.9" y="50.0" text-anchor="middle" class="fig-small fig-bold">200 Hz: 109 edges below it, so bin 109</text>
<text x="300.0" y="215.0" text-anchor="middle" class="fig-small">neighboring edges differ by a ratio of (795/71)^(1/254) = 1.00956, about 16.5 cents</text>
<text x="300.0" y="236.0" text-anchor="middle" class="fig-small">the bin of a value is the number of edges below it</text>
</svg>
Figure 2. Equal steps on a log axis are equal pitch ratios, which is how pitch is heard. Evenly spaced ticks here are the 255 edges, and 200 Hz sits after the 109th.
```

For pitch from 71 Hz to 795 Hz with 256 bins, the ratio between neighboring edges is $(795/71)^{1/254} = 1.00956$, about 16.5 cents. An F0 of 200 Hz sits $\ln(200/71) / \ln(1.00956) = 108.9$ steps above the first edge. So edges 0 to 108 lie below it, 109 of them, and 200 Hz falls in bin 109.

> [!TIP]
> Compute pitch and energy statistics on the training set only, and save the bin edges with the model. A model trained with one set of edges and run with another hears every pitch as a different note.

## Monotonic alignment search

Glow-TTS learns durations with no aligner and no teacher. Its decoder is a normalizing flow (see [Normalizing flows and VITS](/lessons/043-normalizing-flows-and-vits)) that maps each mel frame to a latent vector $z_j$, and its text encoder predicts a Gaussian, a mean $\mu_i$ and a scale $\sigma_i$, for each phone $i$. Given the alignment, the log-likelihood of the frames is a sum of Gaussian log-densities. So training alternates two steps, like hard EM: with the model fixed, find the alignment that makes the frames most likely, then with that alignment fixed, take a gradient step.

The first step is a search over *monotonic* alignments. The first frame goes to the first phone, the last frame to the last phone, and from one frame to the next the phone either stays or moves forward by one. Every phone gets at least one frame, and none is skipped. Let $P_{ij} = \log \mathcal{N}(z_j; \mu_i, \sigma_i)$ be the log-likelihood of frame $j$ under phone $i$. Let $Q_{ij}$ be the best total log-likelihood of any alignment of frames $1..j$ that ends on phone $i$:

$$
Q_{i j} = \max\left(Q_{i, j-1},\ Q_{i-1, j-1}\right) + P_{i j}, \qquad
Q_{1 1} = P_{1 1}, \qquad Q_{i 1} = -\infty \text{ for } i > 1.
$$

Read it aloud: to end frame $j$ on phone $i$, frame $j-1$ was either on the same phone or on the phone before it. Take the better of the two, and add this frame's own log-likelihood. The first frame can only be on phone 1. The best total is $Q_{L T}$. To recover the path, start at phone $L$ on the last frame, and walk back, stepping down a phone whenever $Q_{i-1, j-1} > Q_{i, j-1}$. The durations are the number of frames on each phone. This is the DTW table from [Dynamic time warping](/lessons/020-dynamic-time-warping) with a different set of allowed moves. In numpy, one column at a time:

```python
advance = np.concatenate([[-np.inf], Q[:-1, j - 1]])
Q[:, j] = np.maximum(Q[:, j - 1], advance) + P[:, j]
```

### Work one search by hand

Take three phones and four frames with these log-likelihoods, and fill $Q$ column by column:

$$
P = \begin{bmatrix} -1 & -2 & -6 & -8 \\ -5 & -1 & -1 & -7 \\ -9 & -6 & -3 & -1 \end{bmatrix}
\quad\Rightarrow\quad
Q = \begin{bmatrix} -1 & -3 & -9 & -17 \\ -\infty & -2 & -3 & -10 \\ -\infty & -\infty & -5 & -4 \end{bmatrix}.
$$

For example, $Q_{2,3} = \max(Q_{2,2}, Q_{1,2}) + P_{2,3} = \max(-2, -3) - 1 = -3$, and $Q_{3,4} = \max(Q_{3,3}, Q_{2,3}) + P_{3,4} = \max(-5, -3) - 1 = -4$. Backtracking from $(3, 4)$: $Q_{2,3} = -3$ beats $Q_{3,3} = -5$, so frame 3 was on phone 2, and $Q_{2,2} = -2$ beats $Q_{1,2} = -3$, so frame 2 was too. Frame 1 is on phone 1. The path is $[1, 2, 2, 3]$, the durations are $[1, 2, 1]$, and the total is $-4$.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="The log-likelihood table P and the best-score table Q for three phones and four frames, with the best path 1, 2, 2, 3 highlighted">
<defs><marker id="c039-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="179.0" y="30.0" text-anchor="middle" class="fig-bold fig-small">P: log-likelihood of frame j, phone i</text>
<text x="75.0" y="62.0" text-anchor="end" class="fig-small">frame</text>
<text x="107.0" y="62.0" text-anchor="middle" class="fig-small">1</text>
<text x="155.0" y="62.0" text-anchor="middle" class="fig-small">2</text>
<text x="203.0" y="62.0" text-anchor="middle" class="fig-small">3</text>
<text x="251.0" y="62.0" text-anchor="middle" class="fig-small">4</text>
<text x="75.0" y="100.0" text-anchor="end" class="fig-small">phone 1</text>
<rect x="85.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="107.0" y="92.0" text-anchor="middle" class="fig-bold">-1</text>
<rect x="133.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="155.0" y="92.0" text-anchor="middle">-2</text>
<rect x="181.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="203.0" y="92.0" text-anchor="middle">-6</text>
<rect x="229.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="251.0" y="92.0" text-anchor="middle">-8</text>
<text x="75.0" y="140.0" text-anchor="end" class="fig-small">phone 2</text>
<rect x="85.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="107.0" y="132.0" text-anchor="middle">-5</text>
<rect x="133.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="155.0" y="132.0" text-anchor="middle" class="fig-bold">-1</text>
<rect x="181.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="203.0" y="132.0" text-anchor="middle" class="fig-bold">-1</text>
<rect x="229.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="251.0" y="132.0" text-anchor="middle">-7</text>
<text x="75.0" y="180.0" text-anchor="end" class="fig-small">phone 3</text>
<rect x="85.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="107.0" y="172.0" text-anchor="middle">-9</text>
<rect x="133.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="155.0" y="172.0" text-anchor="middle">-6</text>
<rect x="181.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="203.0" y="172.0" text-anchor="middle">-3</text>
<rect x="229.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="251.0" y="172.0" text-anchor="middle" class="fig-bold">-1</text>
<text x="469.0" y="30.0" text-anchor="middle" class="fig-bold fig-small">Q: best total ending on phone i at frame j</text>
<text x="365.0" y="62.0" text-anchor="end" class="fig-small">frame</text>
<text x="397.0" y="62.0" text-anchor="middle" class="fig-small">1</text>
<text x="445.0" y="62.0" text-anchor="middle" class="fig-small">2</text>
<text x="493.0" y="62.0" text-anchor="middle" class="fig-small">3</text>
<text x="541.0" y="62.0" text-anchor="middle" class="fig-small">4</text>
<text x="365.0" y="100.0" text-anchor="end" class="fig-small">phone 1</text>
<rect x="375.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="397.0" y="92.0" text-anchor="middle" class="fig-bold">-1</text>
<rect x="423.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="445.0" y="92.0" text-anchor="middle">-3</text>
<rect x="471.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="493.0" y="92.0" text-anchor="middle">-9</text>
<rect x="519.0" y="70.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="541.0" y="92.0" text-anchor="middle">-17</text>
<text x="365.0" y="140.0" text-anchor="end" class="fig-small">phone 2</text>
<rect x="375.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="397.0" y="132.0" text-anchor="middle">-inf</text>
<rect x="423.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="445.0" y="132.0" text-anchor="middle" class="fig-bold">-2</text>
<rect x="471.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="493.0" y="132.0" text-anchor="middle" class="fig-bold">-3</text>
<rect x="519.0" y="110.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="541.0" y="132.0" text-anchor="middle">-10</text>
<text x="365.0" y="180.0" text-anchor="end" class="fig-small">phone 3</text>
<rect x="375.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="397.0" y="172.0" text-anchor="middle">-inf</text>
<rect x="423.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="445.0" y="172.0" text-anchor="middle">-inf</text>
<rect x="471.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-sunk stroke-muted"/>
<text x="493.0" y="172.0" text-anchor="middle">-5</text>
<rect x="519.0" y="150.0" width="44.0" height="36.0" rx="4" class="fill-soft-b stroke-b"/>
<text x="541.0" y="172.0" text-anchor="middle" class="fig-bold">-4</text>
<text x="300.0" y="222.0" text-anchor="middle" class="fig-small">Q(2,3) = max(Q(2,2), Q(1,2)) + P(2,3) = max(-2, -3) - 1 = -3</text>
<text x="300.0" y="242.0" text-anchor="middle" class="fig-small">path 1, 2, 2, 3 gives durations [1, 2, 1] and total -1 - 1 - 1 - 1 = -4</text>
</svg>
Figure 3. Each Q cell takes the better of staying on the same phone or advancing from the previous one, then adds the frame's own log-likelihood. The highlighted cells trace the best alignment back from the last cell.
```

Brute force agrees. Four frames split into three nonempty runs in $\binom{3}{2} = 3$ ways, with totals $-6$, $-4$, and $-5$. In general there are $\binom{T-1}{L-1}$ alignments, which is about $10^{20}$ for 20 phones and 100 frames. The dynamic program takes $LT = 2{,}000$ steps.

> [!KEY]
> Monotonic alignment search is Viterbi over "stay or advance by one": the best alignment of the first $j$ frames ending on phone $i$ extends the better of the two ways to reach the previous frame.

MAS runs without gradients, and gradients flow only through $P$ evaluated on the chosen path. The Glow-TTS paper reports that MAS took under 20 ms per iteration on a CPU, less than 2% of training time. At inference there is no mel spectrogram to align, so a duration predictor trained on the durations MAS found supplies them.

> [!RECAP]
> - The length regulator builds $A_{in} = \mathbb{1}[e_i - d_i \le n < e_i]$ from cumulative sums and computes $A^\top H$; padded phones have duration 0.
> - Predicted durations come from $\max(1, \operatorname{round}(\alpha(e^{\hat\ell} - 1)))$.
> - Energy is the L2 norm of each STFT column; pitch and energy are averaged per phone and quantized into 256 bins.
> - $P_{ij}$ is a sum of Gaussian log-densities over dimensions, computed with matrix products.
> - MAS fills $Q_{ij} = \max(Q_{i,j-1}, Q_{i-1,j-1}) + P_{ij}$ and backtracks from the last phone at the last frame.

## Build it step by step

Build the functions in the order the checks test them.

1. **`length_regulate(h, durations)`** takes `(B, L, D)` vectors and `(B, L)` durations and returns `(frames, mask)`. Build `ends` with `torch.cumsum`, compare against `torch.arange(N_max)`, and apply the alignment with `torch.einsum`. Common mistake: using `ends` as the start.
2. **`durations_from_log(log_durations, mask, alpha)`** exponentiates, subtracts 1, scales, rounds, clamps at 1, and multiplies by the mask. Common mistake: clamping after masking, which gives padded phones a frame.
3. **`frame_energy(magnitude)`** and **`phone_average(values, durations, voiced)`** use `np.repeat` to map frames to phones and `np.bincount` to sum. Common mistake: dividing by the duration instead of by the number of voiced frames.
4. **`quantize(values, v_min, v_max, n_bins, log_scale)`** builds `n_bins - 1` edges and calls `np.searchsorted(..., side="left")`. Common mistake: building `n_bins` edges, which makes 257 bins.
5. **`gaussian_log_likelihood(z, mu, log_sigma)`** returns `P` of shape `(L, T)`. Expand the square into three matrix products. Common mistake: forgetting the $-\log\sigma$ term.
6. **`monotonic_alignment_search(P)`** and **`path_to_durations(path, n_text)`** fill `Q` one column at a time, backtrack from `(L - 1, T - 1)`, and count with `np.bincount`.

## Where this breaks

**Rounding drift.** Rounding every phone's duration separately can add or lose several frames over a long sentence. When a total length matters, as in dubbing, round the cumulative sums instead and take differences.

**Averaged prosody.** A duration predictor trained with squared error predicts the mean duration, which gives flat, metronome-like rhythm. VITS replaces it with a stochastic duration predictor that samples durations from a flow.

> [!WARNING]
> MAS needs at least as many frames as phones. A clip shorter than its phone string has no monotonic alignment, and the table's last cell stays $-\infty$. Filter such clips out of the training set.

## The exercise

You implement `length_regulate`, `durations_from_log`, `frame_energy`, `phone_average`, `quantize`, `gaussian_log_likelihood`, `monotonic_alignment_search`, and `path_to_durations`. The checks run the hand-worked expansion, compare a padded batch with `torch.repeat_interleave`, check that each phone's gradient equals its duration and that 200 Hz lands in pitch bin 109, and compare your alignment's score with brute-force enumeration. The last check hides 12 durations in noisy frames and requires MAS to find all of them.

Press **Run** to hide eight phones' durations in 109 noisy frames and find them again with your search. The run prints per-phone pitch, pitch bins, and energy, draws the likelihood table and the chosen path, and plays the phones expanded by your length regulator at $\alpha = 1$ and $\alpha = 1.6$.
