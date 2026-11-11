---
title: Text to speech and Tacotron
slug: 050-text-to-speech-and-tacotron
part: "Part 5 — Speech synthesis"
summary: Follow text to waveform through a TTS pipeline, and build the attention and losses that let Tacotron learn to read in order.
minutes: 55
gpu: false
objectives:
  - Trace a sentence through text normalization, G2P, an acoustic model, and a vocoder, with the frame and sample counts at each stage.
  - Compute one step of location-sensitive attention, including the convolution over cumulative attention.
  - Write the stop-token loss with its single positive frame and the guided attention loss that pulls attention toward the diagonal.
  - "Diagnose an alignment plot: tell a clean diagonal from skipping, repeating, and an early stop."
prereqs:
  - 019-phonemes-and-text-normalization
  - 033-attention-encoder-decoders
  - 008-phase-and-griffin-lim
  - 013-log-mel-spectrograms
exercise: 050-text-to-speech-and-tacotron
---

# Text to speech and Tacotron

Text to speech, or *TTS*, turns written text into a waveform you can play. It is recognition run backwards, and it is harder in one specific way: the mapping is one-to-many. The sentence "I never said she stole it" has one transcript, but a speaker can stress any of its seven words and pause anywhere. A TTS model has to pick one good reading and commit to it.

The hardest part is timing. The text has about 15 characters per second of speech, and the spectrogram has about 86 frames per second. Nothing in the text says how many frames each letter gets, and you cannot give every letter the same number, because letters and sounds differ in length. Tacotron, the model that made neural TTS practical in 2017, learns the timing with attention: at every output frame, the decoder decides which piece of text it is reading.

You build the parts that make that attention work: one step of location-sensitive attention, the stop token that ends the utterance, and the guided attention loss that teaches the model to read left to right. Then you train a toy Tacotron and hear it read.

## What you already know

- Text normalization turns "Dr. Lee paid $3.50" into words, and G2P turns words into phones (see [Phonemes and text normalization](/lessons/019-phonemes-and-text-normalization)).
- A log-mel spectrogram has one column per frame and 80 rows, one per mel band (see [Log-mel spectrograms](/lessons/013-log-mel-spectrograms)).
- Griffin-Lim turns a magnitude spectrogram back into a waveform by guessing phase (see [Phase and Griffin-Lim](/lessons/008-phase-and-griffin-lim)).
- An attention decoder scores every encoder output, takes a softmax, and averages the outputs with those weights (see [Attention encoder-decoders](/lessons/033-attention-encoder-decoders)).
- The sigmoid $\sigma(x) = 1 / (1 + e^{-x})$ turns a logit into a probability, and binary cross-entropy scores it.

## The pipeline from text to waveform

A classic TTS system is a chain of stages, and each one turns a representation into a longer, more detailed one. Text normalization and grapheme-to-phoneme (G2P) conversion turn text into phones, using a lexicon such as CMUdict (about 134,000 entries) with a fallback for unknown words. The **acoustic model** predicts a mel spectrogram from the phones, and Tacotron is an acoustic model. The **vocoder** turns the mel spectrogram into samples.

The lengths grow fast. Take a 2-second sentence in a system with LJSpeech settings: 22,050 Hz audio, a hop of 256 samples (11.6 ms), and 80 mel bands. It has about 30 characters and 24 phones, $\lceil 44{,}100 / 256 \rceil = 173$ mel frames, and 44,100 samples.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="Units in a 2 second sentence on a log scale: about 30 characters, 24 phones, 173 mel frames, and 44,100 samples">
<defs><marker id="c038-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="130.0" y="24.0" text-anchor="start" class="fig-small">number of units, log scale</text>
<text x="120.0" y="66.0" text-anchor="end" class="fig-bold">characters</text>
<rect x="130.0" y="44.0" width="94.3" height="32.0" rx="3" class="fill-a"/>
<text x="232.3" y="58.0" text-anchor="start" class="fig-bold">about 30</text>
<text x="232.3" y="74.0" text-anchor="start" class="fig-small">about 15 per second</text>
<text x="120.0" y="118.0" text-anchor="end" class="fig-bold">phones</text>
<rect x="130.0" y="96.0" width="88.1" height="32.0" rx="3" class="fill-a"/>
<text x="226.1" y="110.0" text-anchor="start" class="fig-bold">about 24</text>
<text x="226.1" y="126.0" text-anchor="start" class="fig-small">about 12 per second</text>
<text x="120.0" y="170.0" text-anchor="end" class="fig-bold">mel frames</text>
<rect x="130.0" y="148.0" width="142.9" height="32.0" rx="3" class="fill-b"/>
<text x="280.9" y="162.0" text-anchor="start" class="fig-bold">173</text>
<text x="280.9" y="178.0" text-anchor="start" class="fig-small">86 per second, 256 samples each</text>
<text x="120.0" y="222.0" text-anchor="end" class="fig-bold">samples</text>
<rect x="130.0" y="200.0" width="296.5" height="32.0" rx="3" class="fill-b"/>
<text x="434.5" y="214.0" text-anchor="start" class="fig-bold">44,100</text>
<text x="434.5" y="230.0" text-anchor="start" class="fig-small">22,050 per second</text>
<line x1="130.0" y1="38.0" x2="130.0" y2="254.0" class="stroke-muted"/>
<text x="300.0" y="262.0" text-anchor="middle" class="fig-small">acoustic model: 30 symbols to 173 frames; vocoder: 173 frames to 44,100 samples</text>
</svg>
Figure 1. The text has about 15 symbols per second, but the spectrogram has 86 frames per second. Nothing in the text says how many frames each symbol gets, so the acoustic model must learn it.
```

Figure 1 shows the counts on a log scale. The acoustic model does the stretch from about 30 symbols to 173 frames, and the vocoder does the 256-fold stretch from frames to samples. Neural models took over the chain one stage at a time: WaveNet (2016) the vocoder, Tacotron (2017) the acoustic model, FastSpeech (2019) removed attention, and VITS (2021) trained the whole chain as one model.

> [!KEY]
> TTS goes from about 15 symbols per second to about 86 frames per second to 22,050 samples per second. The acoustic model must learn how long each symbol lasts, because nothing in the text says.

## Tacotron 2

Tacotron 2 is a sequence-to-sequence model with attention, like the listen-attend-spell recognizer with its input and output swapped. A convolutional and bidirectional-LSTM encoder outputs one vector $h_n$ per character. A pre-net of two 256-unit layers with dropout 0.5 processes the previous mel frame, two LSTM layers of 1024 units form the decoder, and a linear layer predicts the next 80-band mel frame. The first Tacotron predicted $r = 2$ frames per step, its *reduction factor*. A second linear layer with a sigmoid is the stop token, and a post-net of five convolutions sharpens the spectrogram.

With a WaveNet vocoder, the paper reported a mean opinion score of 4.53 against 4.58 for recordings. During training the decoder uses *teacher forcing*: the input at step $t$ is the true frame $t-1$, not the model's own prediction, so early mistakes do not compound. At inference the decoder feeds itself, and the pre-net's dropout stays on. The authors found that essential, and a reimplementation that switches it off with `model.eval()` often produces noise.

## Location-sensitive attention

Location-aware attention, from the chapter on attention encoder-decoders, convolves the previous step's weights with a few learned filters and adds the result to the content energy, so the decoder knows where it looked last. TTS needs that memory even more. The sentence "the cat and the dog" holds "the" twice, and a content score cannot tell the two apart.

Tacotron 2's *location-sensitive attention* changes one thing. It convolves the *cumulative attention* $c$, the running sum of all earlier weights, in place of the last step's weights alone. The sum records how much of the text has been read, not only where the last step looked:

$$
f = F * c, \qquad
e_n = v^\top \tanh\left(W s + V h_n + U f_n + b\right), \qquad
\alpha_n = \frac{\exp(e_n)}{\sum_{m} \exp(e_m)}.
$$

Read it aloud: $F * c$ convolves the cumulative attention with the filters $F$, giving a short feature vector $f_n$ at every text position $n$. The energy $e_n$ adds three projections, of the decoder state $s$, the encoder output $h_n$, and the location features, then applies tanh and a dot product with a learned vector $v$. The weights $\alpha_n$ are the softmax of the energies, the context vector $\sum_n \alpha_n h_n$ goes to the decoder, and after the step, $c \leftarrow c + \alpha$. In torch, with memory of shape `(B, N, D)`:

```python
f = F.conv1d(cum[:, None, :], filters, padding=(k - 1) // 2).transpose(1, 2)
e = torch.tanh((s @ W.T)[:, None] + memory @ V.T + f @ U.T + b) @ v
```

### Work one step by hand

Take three text positions, one location filter, and a one-dimensional attention space. The filter is $[1, -1, 0]$, so with one zero of padding on each side, $f_n = c_{n-1} - c_n$. It is positive right after the covered text. Say the decoder spent the last step entirely on position 0, so $c = [1, 0, 0]$, and the content terms $Ws + Vh_n + b$ come to $[0.2, 0.1, 0.3]$. Take $U = 2$ and $v = 1$:

- Location features: $f = [0 - 1,\ 1 - 0,\ 0 - 0] = [-1, 1, 0]$.
- Inside the tanh: $[0.2 - 2,\ 0.1 + 2,\ 0.3 + 0] = [-1.8, 2.1, 0.3]$.
- Energies: $\tanh$ gives $[-0.9468, 0.9705, 0.2913]$.
- Exponentials: $[0.3880, 2.6391, 1.3382]$, which sum to $4.3653$.
- Weights: $\alpha = [0.0889, 0.6046, 0.3066]$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Attention over three text positions: content alone gives 0.333, 0.302, 0.366, and adding location features gives 0.089, 0.605, 0.307">
<defs><marker id="c038-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<line x1="90.0" y1="230.0" x2="420.0" y2="230.0" class="stroke-grid"/>
<text x="82.0" y="234.0" text-anchor="end" class="fig-small">0</text>
<line x1="90.0" y1="187.5" x2="420.0" y2="187.5" class="stroke-grid"/>
<text x="82.0" y="191.5" text-anchor="end" class="fig-small">0.25</text>
<line x1="90.0" y1="145.0" x2="420.0" y2="145.0" class="stroke-grid"/>
<text x="82.0" y="149.0" text-anchor="end" class="fig-small">0.5</text>
<line x1="90.0" y1="102.5" x2="420.0" y2="102.5" class="stroke-grid"/>
<text x="82.0" y="106.5" text-anchor="end" class="fig-small">0.75</text>
<rect x="120.0" y="173.4" width="50.0" height="56.6" rx="2" class="fill-soft-a stroke-a"/>
<text x="145.0" y="167.4" text-anchor="middle" class="fig-small">0.333</text>
<rect x="174.0" y="214.9" width="50.0" height="15.1" rx="2" class="fill-b"/>
<text x="199.0" y="208.9" text-anchor="middle" class="fig-small fig-bold">0.089</text>
<text x="172.0" y="248.0" text-anchor="middle" class="fig-small">position 0</text>
<rect x="225.0" y="178.7" width="50.0" height="51.3" rx="2" class="fill-soft-a stroke-a"/>
<text x="250.0" y="172.7" text-anchor="middle" class="fig-small">0.302</text>
<rect x="279.0" y="127.2" width="50.0" height="102.8" rx="2" class="fill-b"/>
<text x="304.0" y="121.2" text-anchor="middle" class="fig-small fig-bold">0.605</text>
<text x="277.0" y="248.0" text-anchor="middle" class="fig-small">position 1</text>
<rect x="330.0" y="167.9" width="50.0" height="62.1" rx="2" class="fill-soft-a stroke-a"/>
<text x="355.0" y="161.9" text-anchor="middle" class="fig-small">0.365</text>
<rect x="384.0" y="177.9" width="50.0" height="52.1" rx="2" class="fill-b"/>
<text x="409.0" y="171.9" text-anchor="middle" class="fig-small fig-bold">0.307</text>
<text x="382.0" y="248.0" text-anchor="middle" class="fig-small">position 2</text>
<text x="255.0" y="268.0" text-anchor="middle" class="fig-small">text position n</text>
<rect x="440.0" y="60.0" width="16.0" height="14.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="462.0" y="72.0" text-anchor="start" class="fig-small">content only</text>
<rect x="440.0" y="84.0" width="16.0" height="14.0" rx="2" class="fill-b"/>
<text x="462.0" y="96.0" text-anchor="start" class="fig-small">content + location</text>
<text x="440.0" y="130.0" text-anchor="start" class="fig-small">c = [1, 0, 0]</text>
<text x="440.0" y="148.0" text-anchor="start" class="fig-small">f = [-1, 1, 0]</text>
<text x="440.0" y="170.0" text-anchor="start" class="fig-small">the first symbol is</text>
<text x="440.0" y="186.0" text-anchor="start" class="fig-small">already covered, so</text>
<text x="440.0" y="202.0" text-anchor="start" class="fig-small">attention moves on</text>
</svg>
Figure 2. Content alone prefers position 2 by a thin margin. The location features, computed from what has already been read, move most of the weight to position 1, the next symbol.
```

Content alone preferred position 2, giving the three positions 0.333, 0.302, and 0.366. The location term moved attention to position 1, the next symbol, as Figure 2 shows. The following demo shows whole alignments for "the cat sat on the mat": a clean one, the skipping and repeating failures, and one that has not formed yet.

```demo
{"id": "attention-alignment", "caption": "Choose each alignment and read its largest jump and where it reaches the end, then pick the one that isn't aligned yet and turn on location-sensitive attention to watch the blur sharpen toward the diagonal.", "props": {"task": "tts"}}
```

> [!KEY]
> The location features let the decoder read "where have I already been?" from the cumulative attention, which is what makes it move forward one symbol at a time instead of jumping to any symbol that looks similar.

## The stop token

An autoregressive decoder has to decide when to stop. Tacotron 2 predicts a stop probability $p_t = \sigma(z_t)$ at every frame, from a logit $z_t$, and generation stops at the first frame where it passes 0.5. The target $y_t$ is 1 only at the last real frame. The loss is binary cross-entropy averaged over the $T$ real frames:

$$
\mathcal{L}_{\text{stop}} = -\frac{1}{T} \sum_{t=0}^{T-1}
\left[ w_+ \, y_t \log p_t + (1 - y_t) \log (1 - p_t) \right].
$$

Read it aloud: at every real frame, score the predicted probability of stopping against the target. The positive weight $w_+$ scales up the one frame that says "stop", because an utterance of 500 frames has 499 frames saying "continue". Padding frames after the end do not count at all.

For example, take one utterance of 4 frames with logits $[-3, -2, -1, 2]$ and $w_+ = 1$. The probabilities are $[0.0474, 0.1192, 0.2689, 0.8808]$, the per-frame losses are $[0.0486, 0.1269, 0.3133, 0.1269]$, and their mean is $0.1539$.

> [!WARNING]
> Put the positive at frame `length - 1` of each utterance, not at the last column of the padded batch. A stop target on a padding frame teaches the model that every utterance is as long as the longest one in its batch.

## Guided attention

Text and speech run in the same order, at a roughly steady rate. So the attention matrix of a good alignment lies near the diagonal from the top left to the bottom right. Early in training, attention is a diffuse blur, and the model can take tens of thousands of steps to find the diagonal on its own. The *guided attention loss* of DC-TTS speeds that up by charging a price for attention mass far from the diagonal:

$$
W_{t n} = 1 - \exp\left(-\frac{(n/N - t/T)^2}{2 g^2}\right), \qquad
\mathcal{L}_{\text{guide}} = \frac{1}{TN} \sum_{t=0}^{T-1} \sum_{n=0}^{N-1} A_{t n} W_{t n}.
$$

Read it aloud: $A_{tn}$ is the attention weight on text position $n$ at output frame $t$, for $T$ frames and $N$ text positions. The penalty $W_{tn}$ is 0 when the fraction of text read, $n/N$, equals the fraction of time elapsed, $t/T$, and rises toward 1 as they drift apart, with $g = 0.2$ setting the width of the free band. The loss is the average penalty paid. In torch:

```python
t = torch.arange(T)[:, None] / T
n = torch.arange(N)[None, :] / N
W = 1 - torch.exp(-(n - t) ** 2 / (2 * g**2))
```

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="The guided attention penalty grid for 8 frames and 4 symbols: zero on the diagonal, near one in the far corners">
<defs><marker id="c038-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="185.0" y="26.0" text-anchor="middle" class="fig-small fig-bold">n = 0</text>
<text x="255.0" y="26.0" text-anchor="middle" class="fig-small fig-bold">n = 1</text>
<text x="325.0" y="26.0" text-anchor="middle" class="fig-small fig-bold">n = 2</text>
<text x="395.0" y="26.0" text-anchor="middle" class="fig-small fig-bold">n = 3</text>
<text x="140.0" y="55.0" text-anchor="end" class="fig-small">t = 0</text>
<rect x="150.0" y="36.0" width="67.0" height="25.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="183.5" y="54.0" text-anchor="middle">0.00</text>
<rect x="220.0" y="36.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="253.5" y="54.0" text-anchor="middle">0.54</text>
<rect x="290.0" y="36.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="323.5" y="54.0" text-anchor="middle">0.96</text>
<rect x="360.0" y="36.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="393.5" y="54.0" text-anchor="middle">1.00</text>
<text x="140.0" y="83.0" text-anchor="end" class="fig-small">t = 1</text>
<rect x="150.0" y="64.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="183.5" y="82.0" text-anchor="middle">0.18</text>
<rect x="220.0" y="64.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="253.5" y="82.0" text-anchor="middle">0.18</text>
<rect x="290.0" y="64.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="323.5" y="82.0" text-anchor="middle">0.83</text>
<rect x="360.0" y="64.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="393.5" y="82.0" text-anchor="middle">0.99</text>
<text x="140.0" y="111.0" text-anchor="end" class="fig-small">t = 2</text>
<rect x="150.0" y="92.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="183.5" y="110.0" text-anchor="middle">0.54</text>
<rect x="220.0" y="92.0" width="67.0" height="25.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="253.5" y="110.0" text-anchor="middle">0.00</text>
<rect x="290.0" y="92.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="323.5" y="110.0" text-anchor="middle">0.54</text>
<rect x="360.0" y="92.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="393.5" y="110.0" text-anchor="middle">0.96</text>
<text x="140.0" y="139.0" text-anchor="end" class="fig-small">t = 3</text>
<rect x="150.0" y="120.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="183.5" y="138.0" text-anchor="middle">0.83</text>
<rect x="220.0" y="120.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="253.5" y="138.0" text-anchor="middle">0.18</text>
<rect x="290.0" y="120.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="323.5" y="138.0" text-anchor="middle">0.18</text>
<rect x="360.0" y="120.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="393.5" y="138.0" text-anchor="middle">0.83</text>
<text x="140.0" y="167.0" text-anchor="end" class="fig-small">t = 4</text>
<rect x="150.0" y="148.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="183.5" y="166.0" text-anchor="middle">0.96</text>
<rect x="220.0" y="148.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="253.5" y="166.0" text-anchor="middle">0.54</text>
<rect x="290.0" y="148.0" width="67.0" height="25.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="323.5" y="166.0" text-anchor="middle" class="fig-bold">0.00</text>
<rect x="360.0" y="148.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="393.5" y="166.0" text-anchor="middle">0.54</text>
<text x="140.0" y="195.0" text-anchor="end" class="fig-small">t = 5</text>
<rect x="150.0" y="176.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="183.5" y="194.0" text-anchor="middle">0.99</text>
<rect x="220.0" y="176.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="253.5" y="194.0" text-anchor="middle">0.83</text>
<rect x="290.0" y="176.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="323.5" y="194.0" text-anchor="middle">0.18</text>
<rect x="360.0" y="176.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="393.5" y="194.0" text-anchor="middle">0.18</text>
<text x="140.0" y="223.0" text-anchor="end" class="fig-small">t = 6</text>
<rect x="150.0" y="204.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="183.5" y="222.0" text-anchor="middle">1.00</text>
<rect x="220.0" y="204.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="253.5" y="222.0" text-anchor="middle">0.96</text>
<rect x="290.0" y="204.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="323.5" y="222.0" text-anchor="middle">0.54</text>
<rect x="360.0" y="204.0" width="67.0" height="25.0" rx="3" class="fill-soft-c stroke-c"/>
<text x="393.5" y="222.0" text-anchor="middle">0.00</text>
<text x="140.0" y="251.0" text-anchor="end" class="fig-small">t = 7</text>
<rect x="150.0" y="232.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="183.5" y="250.0" text-anchor="middle" class="fig-bold">1.00</text>
<rect x="220.0" y="232.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="253.5" y="250.0" text-anchor="middle">0.99</text>
<rect x="290.0" y="232.0" width="67.0" height="25.0" rx="3" class="fill-soft-b stroke-b"/>
<text x="323.5" y="250.0" text-anchor="middle">0.83</text>
<rect x="360.0" y="232.0" width="67.0" height="25.0" rx="3" class="fill-soft-a stroke-a"/>
<text x="393.5" y="250.0" text-anchor="middle">0.18</text>
<text x="450.0" y="166.0" text-anchor="start" class="fig-small fig-bold">&lt;- W = 0 at (4, 2)</text>
<text x="450.0" y="250.0" text-anchor="start" class="fig-small fig-bold">&lt;- W = 0.99993 at (7, 0)</text>
<text x="300.0" y="282.0" text-anchor="middle" class="fig-small">penalty W(t, n) = 1 - exp(-(n/N - t/T)^2 / (2 g^2)), g = 0.2</text>
<rect x="150.0" y="294.0" width="16.0" height="14.0" rx="2" class="fill-soft-c stroke-c"/>
<text x="172.0" y="306.0" text-anchor="start" class="fig-small">below 0.1</text>
<rect x="260.0" y="294.0" width="16.0" height="14.0" rx="2" class="fill-soft-a stroke-a"/>
<text x="282.0" y="306.0" text-anchor="start" class="fig-small">0.1 to 0.6</text>
<rect x="380.0" y="294.0" width="16.0" height="14.0" rx="2" class="fill-soft-b stroke-b"/>
<text x="402.0" y="306.0" text-anchor="start" class="fig-small">above 0.6</text>
</svg>
Figure 3. Each cell is the price of putting attention there. Cells on the diagonal cost nothing and the far corners cost almost the full price, so the loss pulls a blurry alignment toward the diagonal.
```

For $T = 8$ frames and $N = 4$ symbols, the cell $(t, n) = (4, 2)$ has $n/N - t/T = 0.5 - 0.5 = 0$, so $W = 0$. The far corner $(7, 0)$ has $0 - 0.875 = -0.875$, and $W = 1 - e^{-0.7656/0.08} = 0.99993$. Attention in that corner costs almost the full price.

The same grid gives a quick score. The *diagonality* is the fraction of attention mass with $|n/N - t/T| \le 0.1$. A clean alignment scores near 1, and a blur scores near the band's share of the grid.

### Four ways attention fails

The alignment plot, text position against decoder step, is the best debugging tool for an attention TTS model. It shows four failures:

- **Skipping.** The bright band jumps forward, and the output drops words.
- **Repeating.** The band stalls or moves backward, and a word repeats. The stop token may never fire.
- **Early stop.** The stop token fires before the band reaches the last symbol.
- **No alignment.** The weights stay a blur, and the output has no words.

All four get worse on long sentences, which is why the next chapter removes attention.

> [!RECAP]
> - Location-sensitive attention convolves the cumulative weights and adds those features to a content score before the softmax.
> - The stop target is 1 at each utterance's last real frame, and padded frames do not count.
> - The guided attention penalty is $1 - \exp(-(n/N - t/T)^2 / 2g^2)$, averaged over the valid cells.
> - Diagonality is the share of attention mass within a band around the diagonal.

## Build it step by step

Build the functions in the order the checks test them. Every attention tensor is laid out `(batch, decoder steps, text positions)`.

1. **`guided_attention_weights(n_frames, n_text, g)`** returns the `(T, N)` grid $W$. Build `t` as a column and `n` as a row, and let broadcasting do the rest. Common mistake: dividing by `T - 1` instead of `T`.
2. **`guided_attention_loss(attention, frame_lengths, text_lengths, g)`** builds a weight tensor utterance by utterance, each from its own two lengths, and a boolean mask of valid cells. Return the masked sum over the mask's count. Common mistake: building one grid from the padded sizes.
3. **`stop_token_loss(logits, lengths, pos_weight)`** builds the target with `arange(T)[None, :] == lengths[:, None] - 1`. Use `F.binary_cross_entropy_with_logits` with `reduction="none"`, and average over the real frames only.
4. **`location_sensitive_attention(query, memory, cumulative, mask, params)`** convolves the cumulative weights with `F.conv1d`, padded by `(K - 1) // 2`, adds the projections, applies tanh, and dots with `v`. Fill masked energies with `-inf` before the softmax. Common mistake: forgetting to transpose the convolution's `(B, F, N)` output to `(B, N, F)`.
5. **`diagonality(attention, width)`** builds the band mask `abs(n / N - t / T) <= width` and returns the share of the mass inside it as a Python `float`.

## Where this breaks

**The alignment never forms.** Without guided attention, attention can stay a blur for tens of thousands of steps. Add the guided loss, and plot the alignment every few hundred steps.

**Teacher forcing hides free-running failure.** The training loss can look fine while inference babbles, because at inference the decoder sees its own imperfect frames. Look at free-running alignments on held-out text.

**The vocoder hears a different mel.** If the acoustic model's mel settings, such as the mel range and the log floor, differ from the vocoder's, the output is noise.

## The exercise

You implement `guided_attention_weights`, `guided_attention_loss`, `stop_token_loss`, `location_sensitive_attention`, and `diagonality`. The checks confirm the guided grid's hand-worked values, that the guided loss prefers a diagonal and ignores padding, that the stop-token loss matches binary cross-entropy, and that your attention step matches a loop-by-loop reference.

Press **Run** to train a toy Tacotron for about 15 seconds on strings of eight made-up phones, each a pair of tones with its own length, from 4 to 8 frames. Like the first Tacotron, it predicts two frames per decoder step. The run decodes an unseen string with no teacher forcing, stops on your stop token, draws the alignment, and plays the target and the model's reading through Griffin-Lim.
