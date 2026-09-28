---
title: Text to speech and Tacotron
slug: 24-text-to-speech-and-tacotron
part: "Part 5 — Speech synthesis"
summary: Follow text to waveform through a TTS pipeline, and build the attention and losses that let Tacotron learn to read in order.
minutes: 70
gpu: false
objectives:
  - Trace a sentence through text normalization, G2P, an acoustic model, and a vocoder, with the frame and sample counts at each stage.
  - Compute one step of location-sensitive attention, including the convolution over cumulative attention.
  - Write the stop-token loss with its single positive frame and the guided attention loss that pulls attention toward the diagonal.
  - "Diagnose an alignment plot: tell a clean diagonal from skipping, repeating, and an early stop."
prereqs:
  - 11-phonetics-text-and-voice-activity
  - 18-attention-encoder-decoders
  - 05-phase-and-griffin-lim
  - 07-hearing-and-the-mel-scale
exercise: 24-text-to-speech-and-tacotron
---

# Text to speech and Tacotron

Text to speech, or *TTS*, turns written text into a waveform you can play. It
is recognition run backwards, and it is harder in one specific way: the
mapping is one-to-many. The sentence "I never said she stole it" has one
correct transcript, but a speaker can say it fast or slow, stress any of its
seven words, and pause anywhere. A TTS model has to pick one good reading and
commit to it.

The hardest part is timing. The text has about 15 characters per second of
speech, and the spectrogram has about 86 frames per second. Nothing in the
text says how many frames each letter gets. Tacotron, the model that made
neural TTS practical in 2017, learns that timing with attention: at every
output frame, the decoder decides which piece of text it is reading.

In this lesson you build the parts that make that attention work: one step of
location-sensitive attention, the stop token that ends the utterance, and the
guided attention loss that teaches the model to read left to right. Then you
train a toy Tacotron and hear it read.

## What you already know

- Text normalization turns "Dr. Lee paid $3.50" into words, and G2P turns words into phones (lesson 11, *Phonetics, text, and voice activity*).
- A log-mel spectrogram has one column per frame and 80 rows, one per mel band (lesson 7, *Hearing and the mel scale*).
- Griffin-Lim turns a magnitude spectrogram back into a waveform by guessing phase (lesson 5, *Phase and Griffin-Lim*).
- An attention decoder scores every encoder output, takes a softmax, and averages the outputs with those weights (lesson 18, *Attention encoder-decoders*).
- The sigmoid $\sigma(x) = 1 / (1 + e^{-x})$ turns a logit into a probability, and binary cross-entropy scores it.

## The pipeline from text to waveform

A classic TTS system is a chain of five stages. Each stage turns one
representation into a longer, more detailed one:

1. **Text normalization** writes out numbers, dates, and abbreviations as words.
2. **Grapheme-to-phoneme (G2P) conversion** turns each word into phones, from a lexicon such as CMUdict (about 134,000 entries) with a fallback for unknown words.
3. **Linguistic features** add stress and phrase breaks. Neural systems keep little more than the phones and punctuation.
4. **The acoustic model** predicts a mel spectrogram from the phones. Tacotron is an acoustic model.
5. **The vocoder** turns the mel spectrogram into samples. Lessons 26 and 27 build vocoders.

The lengths grow fast along the chain. The following table follows a
2-second sentence through a system with LJSpeech settings: 22,050 Hz audio,
a hop of 256 samples, and 80 mel bands.

| Stage | What one unit is | Units for 2 s |
|---|---|---|
| Characters | a letter or space | about 30 |
| Phones | one speech sound | about 24 |
| Mel frames | 256 samples, 11.6 ms | $\lceil 44{,}100 / 256 \rceil = 173$ |
| Samples | $1/22{,}050$ s | 44,100 |

The acoustic model does the stretch from about 30 symbols to 173 frames. The
vocoder does the 256-fold stretch from frames to samples.

Before neural models, unit-selection systems glued recorded snippets
together, and HMM-based systems predicted vocoder parameters that sounded
smooth but buzzy. WaveNet (2016) replaced the vocoder, Tacotron (2017) the
acoustic model, FastSpeech (2019) removed attention, and VITS (2021) trained
the whole chain as one model. Lessons 25 to 28 follow that path.

> [!KEY]
> TTS goes from about 15 symbols per second to about 86 frames per second to
> 22,050 samples per second. The acoustic model must learn how long each symbol
> lasts; nothing in the text says.

## Tacotron 2

Tacotron 2 is a sequence-to-sequence model with attention, like the
listen-attend-spell recognizer of lesson 18 with its input and output swapped.
Its exact parts are the following:

- **Encoder.** A 512-dimensional character embedding, three 1-D convolutions of 512 filters that each span 5 characters, and one bidirectional LSTM with 256 units per direction. It outputs one vector $h_n$ per character.
- **Attention.** Location-sensitive attention with 128-dimensional hidden vectors, and 32 location filters of length 31.
- **Pre-net.** Two fully connected layers of 256 ReLU units with dropout 0.5. The previous mel frame passes through it before it enters the decoder.
- **Decoder.** Two LSTM layers of 1024 units. A linear layer predicts the next 80-band mel frame, one frame per step. The first Tacotron predicted $r = 2$ frames per step, its *reduction factor*.
- **Stop token.** A second linear layer and a sigmoid predict whether this frame is the last. Generation stops at the first frame where the probability passes 0.5.
- **Post-net.** Five convolutions of 512 filters predict a residual that sharpens the whole spectrogram.

With a WaveNet vocoder, the paper reported a mean opinion score of 4.53
against 4.58 for recordings. Lesson 29 explains such scores.

During training the decoder uses *teacher forcing*: the input at step $t$ is
the true frame $t-1$, not the model's own prediction, so early mistakes don't
compound. At inference the decoder feeds itself. The pre-net's dropout stays
on at inference too. The authors found it essential, and a reimplementation
that switches it off with `model.eval()` often produces noise.

## Location-sensitive attention

Content-based attention scores each encoder output $h_n$ against the decoder
state $s$. That fails for speech in a specific way. The sentence "the cat and
the dog" holds "the" twice, and a content score can't tell the two apart. The
decoder needs to know where it was, so it can move forward from there.

Location-sensitive attention adds that memory. Keep a running sum of all
earlier attention weights, the *cumulative attention* $c$. Slide a small set
of learned filters over it to get location features $f_n$, and add them to
the score:

$$
f = K * c, \qquad
e_n = v^\top \tanh\left(W s + V h_n + U f_n + b\right), \qquad
\alpha_n = \frac{\exp(e_n)}{\sum_{m} \exp(e_m)}.
$$

Read it aloud: $K * c$ convolves the cumulative attention with the filters
$K$, giving a short feature vector $f_n$ at every text position $n$. The
energy $e_n$ adds three projections, one of the decoder state, one of the
encoder output, and one of the location features, passes them through tanh,
and dots the result with a learned vector $v$. The weights $\alpha_n$ are the
softmax of the energies. The context vector $\sum_n \alpha_n h_n$ goes to the
decoder. After the step, $c \leftarrow c + \alpha$.

In torch, with memory of shape `(B, N, D)`:

```python
f = F.conv1d(cum[:, None, :], kernel, padding=(k - 1) // 2).transpose(1, 2)
e = torch.tanh((s @ W.T)[:, None] + memory @ V.T + f @ U.T + b) @ v
```

Chorowski and colleagues introduced this attention for speech recognition
in 2015. Tacotron 2 convolves the cumulative weights. The widely used NVIDIA
implementation also stacks the previous step's weights as a second input
channel.

### Work one step by hand

Take three text positions, one location filter, and a one-dimensional
attention space. The filter is $[1, -1, 0]$, so with one zero of padding on
each side, $f_n = c_{n-1} - c_n$. It is positive just past the covered text.
Say the decoder spent the last step entirely on position 0, so
$c = [1, 0, 0]$, and the content terms $Ws + Vh_n + b$ come to
$[0.2, 0.1, 0.3]$. Take $U = 2$ and $v = 1$:

- Location features: $f = [0 - 1,\ 1 - 0,\ 0 - 0] = [-1, 1, 0]$.
- Inside the tanh: $[0.2 - 2,\ 0.1 + 2,\ 0.3 + 0] = [-1.8, 2.1, 0.3]$.
- Energies: $\tanh$ gives $[-0.9468, 0.9705, 0.2913]$.
- Exponentials: $[0.3880, 2.6391, 1.3382]$, which sum to $4.3653$.
- Weights: $\alpha = [0.0889, 0.6046, 0.3066]$.

Content alone preferred position 2, with the largest score of 0.3. The
location term moved attention to position 1, the next symbol. The following
demo shows whole alignments for "the cat sat on the mat", including the two
failures that location features prevent. Its diagonality readout is 1 minus
the guided attention loss defined later in this lesson.

```demo
{"id": "attention-alignment", "caption": "Choose the repeating and the skipping alignments, then turn on location-sensitive attention and watch each one straighten into a single diagonal.", "props": {"task": "tts"}}
```

> [!KEY]
> The location features let the decoder read "where have I already been?"
> from the cumulative attention, which is what makes it move forward one symbol
> at a time instead of jumping to any symbol that looks similar.

## The stop token

An autoregressive decoder has to decide when to stop. Tacotron 2 predicts a
stop probability $p_t = \sigma(z_t)$ at every frame, from a logit $z_t$. The
target $y_t$ is 1 only at the last real frame. The loss is binary
cross-entropy averaged over the $T$ real frames:

$$
\mathcal{L}_{\text{stop}} = -\frac{1}{T} \sum_{t=0}^{T-1}
\left[ w_+ \, y_t \log p_t + (1 - y_t) \log (1 - p_t) \right].
$$

Read it aloud: at every real frame, score the predicted probability of
stopping against the target. The positive weight $w_+$ scales up the one
frame that says "stop", because an utterance of 500 frames has 499 frames
saying "continue". Padding frames after the end don't count at all.

For example, take one utterance of 4 frames with logits $[-3, -2, -1, 2]$
and $w_+ = 1$. The probabilities are $[0.0474, 0.1192, 0.2689, 0.8808]$. The
per-frame losses are $-\log(1 - p_t)$ for the first three and $-\log p_3$
for the last: $[0.0486, 0.1269, 0.3133, 0.1269]$. Their mean is
$0.1539$.

> [!WARNING]
> Put the positive at frame `length - 1` of each utterance, not at the last
> column of the padded batch. A stop target on a padding frame teaches the model
> that every utterance is as long as the longest one in its batch.

## Guided attention

Text and speech run in the same order, and at a roughly steady rate. So the
attention matrix of a good alignment lies near the diagonal from the top left
to the bottom right. Early in training, attention is a diffuse blur, and the
model can take tens of thousands of steps to find the diagonal on its own.
The *guided attention loss* of DC-TTS speeds that up. It charges a price for
attention mass far from the diagonal:

$$
W_{t n} = 1 - \exp\left(-\frac{(n/N - t/T)^2}{2 g^2}\right), \qquad
\mathcal{L}_{\text{guide}} = \frac{1}{TN} \sum_{t=0}^{T-1} \sum_{n=0}^{N-1} A_{t n} W_{t n}.
$$

Read it aloud: $A_{tn}$ is the attention weight on text position $n$ at
output frame $t$, for $T$ frames and $N$ text positions. The penalty
$W_{tn}$ is 0 when the fraction of text read, $n/N$, equals the fraction of
time elapsed, $t/T$. It rises toward 1 as they drift apart, with $g = 0.2$
setting the width of the free band. The loss is the average penalty paid. In
torch:

```python
t = torch.arange(T)[:, None] / T
n = torch.arange(N)[None, :] / N
W = 1 - torch.exp(-(n - t) ** 2 / (2 * g**2))
```

For $T = 8$ frames and $N = 4$ symbols, the cell $(t, n) = (4, 2)$ has
$n/N - t/T = 0.5 - 0.5 = 0$, so $W = 0$. The far corner $(7, 0)$ has
$0 - 0.875 = -0.875$. Its square is $0.765625$, divided by $2g^2 = 0.08$ that
is $9.5703$, and $W = 1 - e^{-9.5703} = 0.99993$. Attention in that corner
costs almost the full price.

The same grid gives a quick score for any alignment. The *diagonality* is the
fraction of attention mass with $|n/N - t/T| \le 0.1$. A clean alignment
scores near 1. A blur scores near the band's share of the grid.

### Four ways attention fails

The alignment plot, text position against decoder step, is the most useful
debugging tool for an attention TTS model. It shows four failures:

- **Skipping.** The bright band jumps forward, and the output drops words.
- **Repeating.** The band stalls or moves backward, and a word repeats or turns into babble. The stop token may never fire, so decoding runs to its length cap.
- **Early stop.** The stop token fires before the band reaches the last symbol.
- **No alignment.** The weights stay a blur, and the output is an average sound with no words.

All four get worse on long or unusual sentences, which is why the next
lesson removes attention altogether.

> [!RECAP]
> - The acoustic model stretches symbols into mel frames; the vocoder stretches frames into samples.
> - Location-sensitive attention convolves the cumulative weights and adds those features to a content score before the softmax.
> - The stop target is 1 at each utterance's last real frame, and padded frames don't count.
> - The guided attention penalty is $1 - \exp(-(n/N - t/T)^2 / 2g^2)$, averaged over the valid cells.
> - Diagonality is the share of attention mass within a band around the diagonal.

## Build it step by step

Build the functions in the order the checks test them. Every attention tensor
is laid out `(batch, decoder steps, text positions)`.

1. **`guided_attention_weights(n_frames, n_text, g)`** returns the `(T, N)`
   grid $W$. Build `t` as a column and `n` as a row, and let broadcasting do
   the rest. The common mistake is dividing by `T - 1` instead of `T`, which
   moves the zero line off the grid the checks expect.
2. **`guided_attention_loss(attention, frame_lengths, text_lengths, g)`**
   builds a weight tensor utterance by utterance, each from its own two
   lengths, and a boolean mask of valid cells. Return the masked sum over the
   mask's count. The common mistake is building one grid from the padded
   sizes, which puts the diagonal in the wrong place for every short
   utterance.
3. **`stop_token_loss(logits, lengths, pos_weight)`** builds the target
   with `arange(T)[None, :] == lengths[:, None] - 1` and the mask with `<`.
   `F.binary_cross_entropy_with_logits` with `reduction="none"` and
   `pos_weight` gives per-frame values. Average them over the real frames
   only.
4. **`location_sensitive_attention(query, memory, cumulative, mask, params)`**
   convolves the cumulative weights with `F.conv1d`, padded by
   `(K - 1) // 2`, adds the three projections and the bias, applies tanh, and
   dots with `v`. Fill masked energies with `-inf` before `torch.softmax`.
   The common mistake is forgetting to transpose the convolution's
   `(B, F, N)` output to `(B, N, F)`.
5. **`diagonality(attention, width)`** builds the band mask
   `abs(n / N - t / T) <= width` and returns the share of the mass inside
   it as a Python `float`.

## Cost and scaling

An attention decoder is sequential: frame $t$ needs frame $t - 1$. The
following table shows how the cost grows with $T$ output frames and $N$ text
symbols.

| Part | Time | Memory |
|---|---|---|
| Encoder | $O(N)$, parallel | $O(N)$ |
| Attention, per decoder step | $O(N)$ | $O(N)$ |
| Decoding, all steps | $O(TN)$, $T$ sequential steps | $O(TN)$ for the alignment |
| Guided attention loss | $O(TN)$ | $O(TN)$ |

A 10-second sentence at 86 frames per second takes 861 sequential decoder
steps, each through two LSTM layers of 1024 units. Teacher forcing doesn't
remove the loop, because the LSTM state still flows from step to step. That
sequential loop, at training and at inference, is the second reason
FastSpeech replaced attention with durations.

## Where this breaks

**The alignment never forms.** Without guided attention or the pre-net
bottleneck, attention can stay a blur for tens of thousands of steps. Add the
guided loss, and plot the alignment every few hundred steps.

**Teacher forcing hides free-running failure.** The training loss can look
fine while inference babbles, because at inference the decoder sees its own
imperfect frames. Always look at free-running alignments on held-out text.

**Long sentences.** Attention trained on 5-second clips often skips or loops
on a 30-second paragraph. Split long text at punctuation, or use a duration
model (lesson 25).

**The vocoder hears a different mel.** If the acoustic model's mel settings,
such as the mel range and the log floor, differ from the vocoder's, the
output is noise. This mismatch is the most common reason a student pipeline
fails.

## The exercise

You implement `guided_attention_weights`, `guided_attention_loss`,
`stop_token_loss`, `location_sensitive_attention`, and `diagonality`. The
checks confirm the guided grid's hand-worked values, that the guided loss
prefers a diagonal and ignores padding, and that the stop-token loss matches
binary cross-entropy. They compare your attention step with a loop-by-loop
reference, and confirm that the location term alone moves attention to the
first uncovered symbol.

Press **Run** to train a toy Tacotron for about 15 seconds on strings of
eight made-up phones, each a pair of tones with its own length. Like the
first Tacotron, it predicts two frames per decoder step, a *reduction
factor* of 2, which halves the sequential loop. The run
decodes a new string with no teacher forcing, stops on your stop token, draws
the alignment, and plays the target and the model's reading through
Griffin-Lim.
