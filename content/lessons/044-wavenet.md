---
title: WaveNet
slug: 044-wavenet
part: "Part 5 — Speech synthesis"
summary: Build a WaveNet from dilated causal convolutions and mu-law classes, and generate audio one sample at a time with a cache.
minutes: 45
gpu: false
objectives:
  - Build dilated causal convolutions, and compute and verify a stack's receptive field with an impulse test.
  - Encode audio with mu-law companding into 256 classes, and explain why that beats linear quantization for quiet sounds.
  - Implement the gated residual block with skip connections, and run it as a next-sample classifier.
  - Generate audio sample by sample with a per-layer cache that matches the parallel forward pass exactly.
prereqs:
  - 004-mu-law-and-wav-files
  - 009-convolution-and-filters
  - 042-text-to-speech-and-tacotron
exercise: 044-wavenet
---

# WaveNet

A vocoder turns a mel spectrogram into a waveform. The mel spectrogram keeps the loudness of each band in each frame and throws away the phase, the fine frequency detail, and every one of the 256 samples inside a frame. Griffin-Lim fills them in by iteration, and it sounds metallic. A neural vocoder learns how real speech fills them in.

WaveNet, published in 2016, was the first neural model to generate raw audio that listeners could barely tell from recordings. Its idea is blunt: predict each sample from the samples before it, one at a time, 16,000 or 24,000 times per second. That raises a question this chapter answers. To predict a sample well, the network must see several pitch periods back, hundreds of samples. How can it look that far back without hundreds of layers?

You build a small WaveNet: the dilated convolutions that see far back cheaply, the mu-law coding that turns each sample into one of 256 classes, the gated residual stack, and a generator with a cache that makes one-sample steps fast.

## What you already know

- Mu-law companding spends more of its 256 levels on quiet sounds (see [Companding and audio files](/lessons/004-mu-law-and-wav-files)).
- A convolution slides a small kernel along a signal, and a causal filter uses only past inputs (see [Convolution and filters](/lessons/009-convolution-and-filters)).
- A classifier outputs logits, and cross-entropy scores the probability it gave the right class.
- An autoregressive model factors a sequence's probability into one prediction per step, each given the steps before (see [Text to speech and Tacotron](/lessons/042-text-to-speech-and-tacotron)).

## One sample at a time

WaveNet writes the probability of a waveform $x_1, \ldots, x_T$ as a product of next-sample predictions:

$$
p(x) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}).
$$

Read it aloud: the probability of the whole clip is the probability of the first sample, times the probability of the second given the first, and so on. A network outputs each factor. During training every factor is computed at once, because the whole clip is known, which is *teacher forcing*, as in Tacotron. During generation each sample must exist before the next can be predicted, so the loop runs $T$ times. A vocoder also conditions every factor on the mel spectrogram. This chapter's exercise leaves that out, so you hear what the model learns about the waveform alone.

## Mu-law classes

WaveNet treats each sample as a class, not a number. A softmax over classes can express any distribution, including one with two peaks, where a squared error on a number would pick the average of the peaks. But 16-bit audio has 65,536 levels, too many for one softmax. WaveNet uses 256 classes and spends them where the ear needs them. Mu-law companding stretches small amplitudes before quantizing:

$$
f(x) = \operatorname{sign}(x)\,\frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \qquad
q = \left\lfloor \frac{f(x) + 1}{2}\,\mu + \frac12 \right\rfloor, \qquad \mu = 255.
$$

Read it aloud: $f$ keeps the sign and takes a logarithm of the magnitude, scaled so that $\pm 1$ maps to $\pm 1$. Then the range $[-1, 1]$ splits into 256 equal steps, and $q$ is the nearest step, an integer from 0 to 255. Decoding inverts both steps, $y = 2q/\mu - 1$ and $x = \operatorname{sign}(y)\,((1 + \mu)^{|y|} - 1)/\mu$. In numpy:

```python
f = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
q = np.floor((f + 1) / 2 * mu + 0.5).astype(np.int64)
```

Work three samples by hand, with $\ln 256 = 5.5452$:

- $x = 0.5$: $f = \ln(128.5)/5.5452 = 0.8757$, and $(0.8757 + 1)/2 \times 255 + 0.5 = 239.66$, so $q = 239$.
- $x = 0.001$: $f = \ln(1.255)/5.5452 = 0.0410$, so $q = \lfloor 133.22 \rfloor = 133$.
- $x = -0.01$: $f = -\ln(3.55)/5.5452 = -0.2285$, so $q = \lfloor 98.87 \rfloor = 98$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="The mu-law curve f(x) rises steeply near zero, so a small amplitude already reaches a large fraction of the 256 codes">
<line x1="70.0" y1="250.0" x2="470.0" y2="250.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="50.0" class="stroke-muted"/>
<line x1="70.0" y1="250.0" x2="70.0" y2="255.0" class="stroke-muted"/>
<text x="70.0" y="270.0" class="fig-small" text-anchor="middle">0</text>
<line x1="65.0" y1="250.0" x2="70.0" y2="250.0" class="stroke-muted"/>
<text x="61.0" y="254.0" class="fig-small" text-anchor="end">0</text>
<line x1="170.0" y1="250.0" x2="170.0" y2="255.0" class="stroke-muted"/>
<text x="170.0" y="270.0" class="fig-small" text-anchor="middle">0.25</text>
<line x1="65.0" y1="200.0" x2="70.0" y2="200.0" class="stroke-muted"/>
<text x="61.0" y="204.0" class="fig-small" text-anchor="end">0.25</text>
<line x1="270.0" y1="250.0" x2="270.0" y2="255.0" class="stroke-muted"/>
<text x="270.0" y="270.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="65.0" y1="150.0" x2="70.0" y2="150.0" class="stroke-muted"/>
<text x="61.0" y="154.0" class="fig-small" text-anchor="end">0.5</text>
<line x1="370.0" y1="250.0" x2="370.0" y2="255.0" class="stroke-muted"/>
<text x="370.0" y="270.0" class="fig-small" text-anchor="middle">0.75</text>
<line x1="65.0" y1="100.0" x2="70.0" y2="100.0" class="stroke-muted"/>
<text x="61.0" y="104.0" class="fig-small" text-anchor="end">0.75</text>
<line x1="470.0" y1="250.0" x2="470.0" y2="255.0" class="stroke-muted"/>
<text x="470.0" y="270.0" class="fig-small" text-anchor="middle">1</text>
<line x1="65.0" y1="50.0" x2="70.0" y2="50.0" class="stroke-muted"/>
<text x="61.0" y="54.0" class="fig-small" text-anchor="end">1</text>
<line x1="70.0" y1="250.0" x2="470.0" y2="50.0" class="stroke-muted" stroke-dasharray="5 4"/>
<polyline points="70.0,250.0 72.0,220.4 74.0,204.3 76.0,193.2 78.0,184.8 80.0,177.9 82.0,172.2 84.0,167.2 86.0,162.9 88.0,159.0 90.0,155.5 92.0,152.3 94.0,149.3 96.0,146.6 98.0,144.1 100.0,141.7 102.0,139.5 104.0,137.4 106.0,135.5 108.0,133.6 110.0,131.8 112.0,130.1 114.0,128.5 116.0,126.9 118.0,125.5 120.0,124.0 122.0,122.7 124.0,121.3 126.0,120.1 128.0,118.8 130.0,117.6 132.0,116.5 134.0,115.4 136.0,114.3 138.0,113.2 140.0,112.2 142.0,111.2 144.0,110.2 146.0,109.3 148.0,108.4 150.0,107.5 152.0,106.6 154.0,105.8 156.0,104.9 158.0,104.1 160.0,103.3 162.0,102.5 164.0,101.8 166.0,101.0 168.0,100.3 170.0,99.6 172.0,98.9 174.0,98.2 176.0,97.5 178.0,96.8 180.0,96.2 182.0,95.6 184.0,94.9 186.0,94.3 188.0,93.7 190.0,93.1 192.0,92.5 194.0,91.9 196.0,91.4 198.0,90.8 200.0,90.2 202.0,89.7 204.0,89.2 206.0,88.6 208.0,88.1 210.0,87.6 212.0,87.1 214.0,86.6 216.0,86.1 218.0,85.6 220.0,85.1 222.0,84.7 224.0,84.2 226.0,83.7 228.0,83.3 230.0,82.8 232.0,82.4 234.0,82.0 236.0,81.5 238.0,81.1 240.0,80.7 242.0,80.3 244.0,79.8 246.0,79.4 248.0,79.0 250.0,78.6 252.0,78.2 254.0,77.8 256.0,77.5 258.0,77.1 260.0,76.7 262.0,76.3 264.0,75.9 266.0,75.6 268.0,75.2 270.0,74.9 272.0,74.5 274.0,74.2 276.0,73.8 278.0,73.5 280.0,73.1 282.0,72.8 284.0,72.4 286.0,72.1 288.0,71.8 290.0,71.4 292.0,71.1 294.0,70.8 296.0,70.5 298.0,70.2 300.0,69.9 302.0,69.5 304.0,69.2 306.0,68.9 308.0,68.6 310.0,68.3 312.0,68.0 314.0,67.7 316.0,67.4 318.0,67.2 320.0,66.9 322.0,66.6 324.0,66.3 326.0,66.0 328.0,65.7 330.0,65.5 332.0,65.2 334.0,64.9 336.0,64.6 338.0,64.4 340.0,64.1 342.0,63.8 344.0,63.6 346.0,63.3 348.0,63.1 350.0,62.8 352.0,62.5 354.0,62.3 356.0,62.0 358.0,61.8 360.0,61.5 362.0,61.3 364.0,61.1 366.0,60.8 368.0,60.6 370.0,60.3 372.0,60.1 374.0,59.9 376.0,59.6 378.0,59.4 380.0,59.2 382.0,58.9 384.0,58.7 386.0,58.5 388.0,58.2 390.0,58.0 392.0,57.8 394.0,57.6 396.0,57.3 398.0,57.1 400.0,56.9 402.0,56.7 404.0,56.5 406.0,56.3 408.0,56.0 410.0,55.8 412.0,55.6 414.0,55.4 416.0,55.2 418.0,55.0 420.0,54.8 422.0,54.6 424.0,54.4 426.0,54.2 428.0,54.0 430.0,53.8 432.0,53.6 434.0,53.4 436.0,53.2 438.0,53.0 440.0,52.8 442.0,52.6 444.0,52.4 446.0,52.2 448.0,52.0 450.0,51.8 452.0,51.7 454.0,51.5 456.0,51.3 458.0,51.1 460.0,50.9 462.0,50.7 464.0,50.5 466.0,50.4 468.0,50.2 470.0,50.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="74.0" cy="204.3" r="5" class="fill-a"/>
<circle cx="270.0" cy="74.9" r="5" class="fill-a"/>
<text x="86.0" y="222.3" class="fig-small" text-anchor="start">x = 0.01: f = 0.228, code 157</text>
<text x="280.0" y="98.9" class="fig-small" text-anchor="start">x = 0.5: f = 0.876, code 239</text>
<text x="318.0" y="180.0" class="fig-small" text-anchor="start">dashed: linear</text>
<text x="270.0" y="285.0" class="fig-small" text-anchor="middle">amplitude x (full scale 1)</text>
<text x="61.0" y="40.0" class="fig-small" text-anchor="end">f(x)</text>
</svg>
Figure 2. The mu-law curve with mu = 255. Notice that x = 0.01, one percent of full scale, already reaches f = 0.228, so quiet samples spread over many codes where a linear quantizer gives them about one.
```

Between $x = 0$ and $x = 0.01$ lie codes 128 to 157, 30 of them. A linear 8-bit quantizer, with steps of $2/255 = 0.0078$, gives that range little more than one. On a tone of amplitude 0.01, the exercise measures a signal-to-noise ratio of 33.7 dB with mu-law against 9.3 dB with 8-bit linear steps.

> [!KEY]
> WaveNet is a classifier: at every sample it predicts a softmax over 256 mu-law codes, trained with cross-entropy on the next sample.

## See far back with dilated convolutions

The obvious network is a stack of causal convolutions with kernel size 2. Each layer sees one more sample than the layer below it, so $L$ layers see $L + 1$ samples. To see 3,000 samples, you need about 3,000 layers. That is far too deep to train or run.

A *dilated* convolution skips inputs. With dilation $d$ and kernel size $k$, output $t$ combines inputs $t, t - d, \ldots, t - (k-1)d$. Double the dilation at every layer, 1, 2, 4, 8, and each layer doubles how far back the stack sees. Figure 1 draws four such layers. The number of input samples one output depends on is the *receptive field*:

$$
R = 1 + (k - 1) \sum_{\ell} d_\ell.
$$

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A stack of four dilated causal layers with dilations 1, 2, 4, and 8: the output at the last time step depends on all 16 input samples">
<line x1="70.0" y1="290.0" x2="70.0" y2="225.0" class="stroke-grid"/>
<line x1="132.0" y1="290.0" x2="132.0" y2="225.0" class="stroke-grid"/>
<line x1="101.0" y1="290.0" x2="132.0" y2="225.0" class="stroke-grid"/>
<line x1="194.0" y1="290.0" x2="194.0" y2="225.0" class="stroke-grid"/>
<line x1="163.0" y1="290.0" x2="194.0" y2="225.0" class="stroke-grid"/>
<line x1="256.0" y1="290.0" x2="256.0" y2="225.0" class="stroke-grid"/>
<line x1="225.0" y1="290.0" x2="256.0" y2="225.0" class="stroke-grid"/>
<line x1="318.0" y1="290.0" x2="318.0" y2="225.0" class="stroke-grid"/>
<line x1="287.0" y1="290.0" x2="318.0" y2="225.0" class="stroke-grid"/>
<line x1="380.0" y1="290.0" x2="380.0" y2="225.0" class="stroke-grid"/>
<line x1="349.0" y1="290.0" x2="380.0" y2="225.0" class="stroke-grid"/>
<line x1="442.0" y1="290.0" x2="442.0" y2="225.0" class="stroke-grid"/>
<line x1="411.0" y1="290.0" x2="442.0" y2="225.0" class="stroke-grid"/>
<line x1="504.0" y1="290.0" x2="504.0" y2="225.0" class="stroke-grid"/>
<line x1="473.0" y1="290.0" x2="504.0" y2="225.0" class="stroke-grid"/>
<line x1="70.0" y1="225.0" x2="70.0" y2="160.0" class="stroke-grid"/>
<line x1="101.0" y1="225.0" x2="101.0" y2="160.0" class="stroke-grid"/>
<line x1="132.0" y1="225.0" x2="132.0" y2="160.0" class="stroke-grid"/>
<line x1="70.0" y1="225.0" x2="132.0" y2="160.0" class="stroke-grid"/>
<line x1="194.0" y1="225.0" x2="194.0" y2="160.0" class="stroke-grid"/>
<line x1="132.0" y1="225.0" x2="194.0" y2="160.0" class="stroke-grid"/>
<line x1="225.0" y1="225.0" x2="225.0" y2="160.0" class="stroke-grid"/>
<line x1="163.0" y1="225.0" x2="225.0" y2="160.0" class="stroke-grid"/>
<line x1="256.0" y1="225.0" x2="256.0" y2="160.0" class="stroke-grid"/>
<line x1="194.0" y1="225.0" x2="256.0" y2="160.0" class="stroke-grid"/>
<line x1="318.0" y1="225.0" x2="318.0" y2="160.0" class="stroke-grid"/>
<line x1="256.0" y1="225.0" x2="318.0" y2="160.0" class="stroke-grid"/>
<line x1="349.0" y1="225.0" x2="349.0" y2="160.0" class="stroke-grid"/>
<line x1="287.0" y1="225.0" x2="349.0" y2="160.0" class="stroke-grid"/>
<line x1="380.0" y1="225.0" x2="380.0" y2="160.0" class="stroke-grid"/>
<line x1="318.0" y1="225.0" x2="380.0" y2="160.0" class="stroke-grid"/>
<line x1="442.0" y1="225.0" x2="442.0" y2="160.0" class="stroke-grid"/>
<line x1="380.0" y1="225.0" x2="442.0" y2="160.0" class="stroke-grid"/>
<line x1="473.0" y1="225.0" x2="473.0" y2="160.0" class="stroke-grid"/>
<line x1="411.0" y1="225.0" x2="473.0" y2="160.0" class="stroke-grid"/>
<line x1="504.0" y1="225.0" x2="504.0" y2="160.0" class="stroke-grid"/>
<line x1="442.0" y1="225.0" x2="504.0" y2="160.0" class="stroke-grid"/>
<line x1="70.0" y1="160.0" x2="70.0" y2="95.0" class="stroke-grid"/>
<line x1="101.0" y1="160.0" x2="101.0" y2="95.0" class="stroke-grid"/>
<line x1="132.0" y1="160.0" x2="132.0" y2="95.0" class="stroke-grid"/>
<line x1="163.0" y1="160.0" x2="163.0" y2="95.0" class="stroke-grid"/>
<line x1="194.0" y1="160.0" x2="194.0" y2="95.0" class="stroke-grid"/>
<line x1="70.0" y1="160.0" x2="194.0" y2="95.0" class="stroke-grid"/>
<line x1="225.0" y1="160.0" x2="225.0" y2="95.0" class="stroke-grid"/>
<line x1="101.0" y1="160.0" x2="225.0" y2="95.0" class="stroke-grid"/>
<line x1="256.0" y1="160.0" x2="256.0" y2="95.0" class="stroke-grid"/>
<line x1="132.0" y1="160.0" x2="256.0" y2="95.0" class="stroke-grid"/>
<line x1="318.0" y1="160.0" x2="318.0" y2="95.0" class="stroke-grid"/>
<line x1="194.0" y1="160.0" x2="318.0" y2="95.0" class="stroke-grid"/>
<line x1="349.0" y1="160.0" x2="349.0" y2="95.0" class="stroke-grid"/>
<line x1="225.0" y1="160.0" x2="349.0" y2="95.0" class="stroke-grid"/>
<line x1="380.0" y1="160.0" x2="380.0" y2="95.0" class="stroke-grid"/>
<line x1="256.0" y1="160.0" x2="380.0" y2="95.0" class="stroke-grid"/>
<line x1="411.0" y1="160.0" x2="411.0" y2="95.0" class="stroke-grid"/>
<line x1="287.0" y1="160.0" x2="411.0" y2="95.0" class="stroke-grid"/>
<line x1="442.0" y1="160.0" x2="442.0" y2="95.0" class="stroke-grid"/>
<line x1="318.0" y1="160.0" x2="442.0" y2="95.0" class="stroke-grid"/>
<line x1="473.0" y1="160.0" x2="473.0" y2="95.0" class="stroke-grid"/>
<line x1="349.0" y1="160.0" x2="473.0" y2="95.0" class="stroke-grid"/>
<line x1="504.0" y1="160.0" x2="504.0" y2="95.0" class="stroke-grid"/>
<line x1="380.0" y1="160.0" x2="504.0" y2="95.0" class="stroke-grid"/>
<line x1="70.0" y1="95.0" x2="70.0" y2="40.0" class="stroke-grid"/>
<line x1="101.0" y1="95.0" x2="101.0" y2="40.0" class="stroke-grid"/>
<line x1="132.0" y1="95.0" x2="132.0" y2="40.0" class="stroke-grid"/>
<line x1="163.0" y1="95.0" x2="163.0" y2="40.0" class="stroke-grid"/>
<line x1="194.0" y1="95.0" x2="194.0" y2="40.0" class="stroke-grid"/>
<line x1="225.0" y1="95.0" x2="225.0" y2="40.0" class="stroke-grid"/>
<line x1="256.0" y1="95.0" x2="256.0" y2="40.0" class="stroke-grid"/>
<line x1="287.0" y1="95.0" x2="287.0" y2="40.0" class="stroke-grid"/>
<line x1="318.0" y1="95.0" x2="318.0" y2="40.0" class="stroke-grid"/>
<line x1="70.0" y1="95.0" x2="318.0" y2="40.0" class="stroke-grid"/>
<line x1="349.0" y1="95.0" x2="349.0" y2="40.0" class="stroke-grid"/>
<line x1="101.0" y1="95.0" x2="349.0" y2="40.0" class="stroke-grid"/>
<line x1="380.0" y1="95.0" x2="380.0" y2="40.0" class="stroke-grid"/>
<line x1="132.0" y1="95.0" x2="380.0" y2="40.0" class="stroke-grid"/>
<line x1="411.0" y1="95.0" x2="411.0" y2="40.0" class="stroke-grid"/>
<line x1="163.0" y1="95.0" x2="411.0" y2="40.0" class="stroke-grid"/>
<line x1="442.0" y1="95.0" x2="442.0" y2="40.0" class="stroke-grid"/>
<line x1="194.0" y1="95.0" x2="442.0" y2="40.0" class="stroke-grid"/>
<line x1="473.0" y1="95.0" x2="473.0" y2="40.0" class="stroke-grid"/>
<line x1="225.0" y1="95.0" x2="473.0" y2="40.0" class="stroke-grid"/>
<line x1="504.0" y1="95.0" x2="504.0" y2="40.0" class="stroke-grid"/>
<line x1="256.0" y1="95.0" x2="504.0" y2="40.0" class="stroke-grid"/>
<line x1="101.0" y1="290.0" x2="101.0" y2="225.0" class="stroke-b"/>
<line x1="70.0" y1="290.0" x2="101.0" y2="225.0" class="stroke-b"/>
<line x1="163.0" y1="290.0" x2="163.0" y2="225.0" class="stroke-b"/>
<line x1="132.0" y1="290.0" x2="163.0" y2="225.0" class="stroke-b"/>
<line x1="225.0" y1="290.0" x2="225.0" y2="225.0" class="stroke-b"/>
<line x1="194.0" y1="290.0" x2="225.0" y2="225.0" class="stroke-b"/>
<line x1="287.0" y1="290.0" x2="287.0" y2="225.0" class="stroke-b"/>
<line x1="256.0" y1="290.0" x2="287.0" y2="225.0" class="stroke-b"/>
<line x1="349.0" y1="290.0" x2="349.0" y2="225.0" class="stroke-b"/>
<line x1="318.0" y1="290.0" x2="349.0" y2="225.0" class="stroke-b"/>
<line x1="411.0" y1="290.0" x2="411.0" y2="225.0" class="stroke-b"/>
<line x1="380.0" y1="290.0" x2="411.0" y2="225.0" class="stroke-b"/>
<line x1="473.0" y1="290.0" x2="473.0" y2="225.0" class="stroke-b"/>
<line x1="442.0" y1="290.0" x2="473.0" y2="225.0" class="stroke-b"/>
<line x1="535.0" y1="290.0" x2="535.0" y2="225.0" class="stroke-b"/>
<line x1="504.0" y1="290.0" x2="535.0" y2="225.0" class="stroke-b"/>
<line x1="163.0" y1="225.0" x2="163.0" y2="160.0" class="stroke-b"/>
<line x1="101.0" y1="225.0" x2="163.0" y2="160.0" class="stroke-b"/>
<line x1="287.0" y1="225.0" x2="287.0" y2="160.0" class="stroke-b"/>
<line x1="225.0" y1="225.0" x2="287.0" y2="160.0" class="stroke-b"/>
<line x1="411.0" y1="225.0" x2="411.0" y2="160.0" class="stroke-b"/>
<line x1="349.0" y1="225.0" x2="411.0" y2="160.0" class="stroke-b"/>
<line x1="535.0" y1="225.0" x2="535.0" y2="160.0" class="stroke-b"/>
<line x1="473.0" y1="225.0" x2="535.0" y2="160.0" class="stroke-b"/>
<line x1="287.0" y1="160.0" x2="287.0" y2="95.0" class="stroke-b"/>
<line x1="163.0" y1="160.0" x2="287.0" y2="95.0" class="stroke-b"/>
<line x1="535.0" y1="160.0" x2="535.0" y2="95.0" class="stroke-b"/>
<line x1="411.0" y1="160.0" x2="535.0" y2="95.0" class="stroke-b"/>
<line x1="535.0" y1="95.0" x2="535.0" y2="40.0" class="stroke-b"/>
<line x1="287.0" y1="95.0" x2="535.0" y2="40.0" class="stroke-b"/>
<circle cx="70.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="101.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="132.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="163.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="194.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="225.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="256.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="287.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="318.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="349.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="380.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="411.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="442.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="473.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="504.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="535.0" cy="290.0" r="5" class="fill-b"/>
<circle cx="70.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="101.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="132.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="163.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="194.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="225.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="256.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="287.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="318.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="349.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="380.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="411.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="442.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="473.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="504.0" cy="225.0" r="5" class="fill-sunk"/>
<circle cx="535.0" cy="225.0" r="5" class="fill-b"/>
<circle cx="70.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="101.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="132.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="163.0" cy="160.0" r="5" class="fill-b"/>
<circle cx="194.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="225.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="256.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="287.0" cy="160.0" r="5" class="fill-b"/>
<circle cx="318.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="349.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="380.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="411.0" cy="160.0" r="5" class="fill-b"/>
<circle cx="442.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="473.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="504.0" cy="160.0" r="5" class="fill-sunk"/>
<circle cx="535.0" cy="160.0" r="5" class="fill-b"/>
<circle cx="70.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="101.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="132.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="163.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="194.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="225.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="256.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="287.0" cy="95.0" r="5" class="fill-b"/>
<circle cx="318.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="349.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="380.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="411.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="442.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="473.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="504.0" cy="95.0" r="5" class="fill-sunk"/>
<circle cx="535.0" cy="95.0" r="5" class="fill-b"/>
<circle cx="70.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="101.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="132.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="163.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="194.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="225.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="256.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="287.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="318.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="349.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="380.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="411.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="442.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="473.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="504.0" cy="40.0" r="5" class="fill-sunk"/>
<circle cx="535.0" cy="40.0" r="5" class="fill-b"/>
<text x="70.0" y="312.0" class="fig-small" text-anchor="middle">0</text>
<text x="163.0" y="312.0" class="fig-small" text-anchor="middle">3</text>
<text x="256.0" y="312.0" class="fig-small" text-anchor="middle">6</text>
<text x="349.0" y="312.0" class="fig-small" text-anchor="middle">9</text>
<text x="442.0" y="312.0" class="fig-small" text-anchor="middle">12</text>
<text x="535.0" y="312.0" class="fig-small" text-anchor="middle">15</text>
<text x="302.5" y="328.0" class="fig-small" text-anchor="middle">time step t (one sample each)</text>
<text x="10.0" y="294.0" class="fig-small" text-anchor="start">input</text>
<text x="10.0" y="229.0" class="fig-small" text-anchor="start">d = 1</text>
<text x="10.0" y="164.0" class="fig-small" text-anchor="start">d = 2</text>
<text x="10.0" y="99.0" class="fig-small" text-anchor="start">d = 4</text>
<text x="10.0" y="44.0" class="fig-small" text-anchor="start">d = 8</text>
<text x="549.0" y="44.0" class="fig-small" text-anchor="start">output</text>
</svg>
Figure 1. Each layer reaches back by its dilation, so four layers with dilations 1, 2, 4, and 8 connect the last output to all 16 input samples (receptive field R = 1 + 15 = 16). Notice that four undilated layers would see only 5.
```

Read it aloud: each layer with dilation $d_\ell$ reaches $(k-1)\,d_\ell$ samples further back, and the reaches add up. The 1 is the current sample itself. For Figure 1, $R = 1 + (1 + 2 + 4 + 8) = 16$. WaveNet used $k = 2$ and dilations $1, 2, 4, \ldots, 512$, repeated three times, so $R = 1 + 3 \times 1023 = 3070$ samples, about 192 ms at 16 kHz. One block of $1..512$ alone gives $R = 1024$ from only 10 layers, where 10 undilated layers see 11.

*Causal* means the output at $t$ never sees $t + 1$. The simplest way to guarantee that is to pad $(k-1)d$ zeros on the left and none on the right:

```python
y = F.conv1d(F.pad(x, ((k - 1) * d, 0)), weight, bias, dilation=d)
```

For example, take $x = [1, 2, 3, 4]$, a kernel with weight 1 on the current sample and 0.5 on the one two back, and $d = 2$. Then $y_t = x_t + 0.5\,x_{t-2}$, with zeros before the start, so $y = [1,\ 2,\ 3.5,\ 5]$.

You can also measure the receptive field. Change one input sample at time $t_0$, and watch which outputs change. Exactly the outputs at $t_0, \ldots, t_0 + R - 1$ must change: none earlier, because the stack is causal, and none later, because they can't see that far back. The following demo draws that cone of dependence for a stack you choose.

```demo
{"id": "wavenet-receptive-field", "caption": "Add layers per stack and then stacks, and compare the receptive field with the same layers without dilation: each layer whose dilation doubles also doubles how far back the output sees."}
```

> [!WARNING]
> `padding="same"` or symmetric padding lets output $t$ see the future. The training loss then drops fast, because the model copies the answer, and generation produces noise. Run the impulse test on every stack.

## The gated residual stack

Each layer applies a dilated causal convolution with $2C$ output channels and splits the result into a *filter* half and a *gate* half:

$$
z = \tanh(W_f * x) \odot \sigma(W_g * x).
$$

Read it aloud: the tanh half proposes a value between $-1$ and 1, and the sigmoid half decides, per channel and time, how much of it passes. The $\odot$ is elementwise multiplication. Two $1 \times 1$ convolutions then read $z$. One adds a *residual* back to the layer's input, so the next layer sees $x + R z$. The other adds to a running *skip* sum. After the last layer, the skip sum passes through ReLU, a $1 \times 1$ convolution, ReLU, and a final $1 \times 1$ convolution to 256 logits per time step.

```python
filt, gate = conv_out.chunk(2, dim=1)
z = torch.tanh(filt) * torch.sigmoid(gate)
x, skips = x + residual(z), skips + skip(z)
```

The residual path keeps gradients flowing through dozens of layers, and the skip sum lets the output read every layer directly, short range and long range alike.

## Generate with a cache

To generate, feed the model its own samples. Pick $x_1$, compute $p(x_2 \mid x_1)$, sample $x_2$, compute $p(x_3 \mid x_1, x_2)$, and so on. The obvious loop runs the whole network over the last $R$ samples at every step. Almost all of that work repeats the step before.

A cache removes the repetition. With kernel size 2, layer $\ell$ at time $t$ needs exactly two inputs: its current input, and its input $d_\ell$ steps ago. Keep, for each layer, a queue of its last $d_\ell$ inputs. One step then costs one pass through the layers at a single time position:

1. Embed the newest code.
2. For each layer, combine the current input with the oldest entry in that layer's queue, apply the gate, and update the residual and skip.
3. Push the current input onto the queue and drop the oldest.
4. Run the output head, and sample the next code from the softmax.

The queue starts full of zeros, which is exactly what left padding gave the parallel forward pass. So the cached logits must equal the forward pass's logits to rounding error, and the exercise checks that they do. This trick is known as *Fast WaveNet*.

> [!TIP]
> Test a cache against the parallel forward pass before you trust a single generated sample. A cache that is off by one step still produces audio, and it sounds like a model that trained badly.

> [!KEY]
> A dilated causal stack sees thousands of past samples with ten layers, mu-law codes make each sample a 256-way choice, and a per-layer cache turns generation into one cheap step per sample. The cost that remains is the $T$ sequential steps.

> [!RECAP]
> - The receptive field of a dilated stack is $1 + (k-1)\sum_\ell d_\ell$, and an impulse test must change exactly that many outputs.
> - Mu-law maps $x$ to $\lfloor (f(x) + 1)/2 \cdot 255 + 0.5 \rfloor$, giving quiet samples many more of the 256 codes.
> - A causal convolution pads $(k-1)d$ zeros on the left only.
> - Each layer gates with $\tanh \odot \sigma$ and feeds a residual and a skip sum.
> - Cached generation keeps each layer's last $d$ inputs and must match the forward pass exactly.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`receptive_field(dilations, kernel_size)`** returns an `int`. The common mistake is forgetting the 1 for the current sample.
2. **`mu_law_encode(x, mu)`** clips to $[-1, 1]$, compands with `np.log1p`, and quantizes with `np.floor(... + 0.5)`. Return `int64`. The common mistake is `np.round`, which rounds halves to even and moves some codes.
3. **`mu_law_decode(q, mu)`** maps codes to $[-1, 1]$ and inverts the logarithm. The common mistake is inverting with `np.exp` instead of raising $1 + \mu$ to $|y|$.
4. **`causal_conv1d(x, weight, bias, dilation)`** pads on the left with `F.pad` and calls `F.conv1d`. The common mistake is padding `kernel_size - 1` without multiplying by the dilation.
5. **`gated_activation(x)`** splits the channels with `chunk(2, dim=1)`.
6. **`WaveNet.forward(codes)`** embeds, transposes to `(B, C, T)`, runs each layer, and applies the head to the skip sum. Its column $t$ predicts sample $t + 1$. The common mistake is feeding the head the wrong layer's output instead of the skip sum.
7. **`WaveNet.step(code, cache)`** computes one time position. Tap 1 of each dilated kernel multiplies the current input and tap 0 the oldest cached one. A $1 \times 1$ convolution at one position is a matrix product with `weight[:, :, 0]`. The common mistake is appending to the cache before reading its oldest entry.
8. **`generate(model, n_samples, temperature, seed, first)`** loops over `model.step`, samples with `torch.multinomial` and one seeded `torch.Generator`, and uses the argmax when the temperature is 0.

## Cost and scaling

With $L$ layers of $C$ channels, receptive field $R$, and $T$ samples, training with teacher forcing costs $O(T L C^2)$ in one parallel pass. Naive generation costs $O(T R L C^2)$ in $T$ sequential steps, and cached generation costs $O(T L C^2)$, still in $T$ steps. The cache makes generation as cheap in arithmetic as training, but it can't remove the sequential steps.

One second of 24 kHz audio needs 24,000 sequential steps. In the HiFi-GAN paper's comparison, a WaveNet vocoder ran at 0.07 kHz on a V100 GPU, about 0.003 times real time, because one time position gives the GPU too little work.

## Where this breaks

**The future leaks in.** Symmetric padding, or a target that isn't shifted by one sample, lets the model see its answer. Training loss collapses and generation fails. The impulse test catches the first, and checking that column $t$ is scored against sample $t + 1$ catches the second.

**Errors compound.** In generation the model reads its own samples. A small early error is input the model never saw in training. Lowering the sampling temperature trades variety for stability.

**Babble.** An unconditional WaveNet trained on speech produces speech-like sounds with no words. Conditioning on a mel spectrogram is what makes it a vocoder. The mel runs at the frame rate, 86 frames per second at 22,050 Hz with a hop of 256, so it is upsampled to the sample rate and added inside both halves of every gate.

**It is slow.** Real-time WaveNet needs custom GPU kernels or a smaller recurrent model. WaveRNN used a single GRU and ran on mobile CPUs. LPCNet put a linear predictor back in and let a small network model only the residual. Parallel WaveNet distilled a WaveNet into a flow, which [Normalizing flows and VITS](/lessons/046-normalizing-flows-and-vits) explains. For most uses a GAN vocoder is simpler, and the next chapter, [GAN vocoders](/lessons/045-gan-vocoders), builds one.

## The exercise

You implement `receptive_field`, `mu_law_encode`, `mu_law_decode`, `causal_conv1d`, `gated_activation`, `WaveNet.forward`, `WaveNet.step`, and `generate`. The checks confirm the 3,070-sample receptive field and the hand-worked mu-law codes, and that mu-law beats 8-bit linear quantization by at least 15 dB on a quiet tone. They compare your causal convolution with a loop, your forward pass with the architecture written out from its weights, and run the impulse test. Two last checks require your cached steps to match the forward pass, and greedy generation to match re-running the full model on every prefix.

Press **Run** to train a 12-layer WaveNet with a 127-sample receptive field on three seconds of a sung vowel at 8 kHz, for about 15 seconds. The run then generates half a second one sample at a time, prints how many samples per second your cache produced, plays the training audio and the generated audio, and draws the generated audio's spectrogram.
