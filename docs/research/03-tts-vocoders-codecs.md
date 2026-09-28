# Research notes 03: Text-to-speech, neural vocoders, neural audio codecs, and codec language models

These notes are source material for the course part on speech synthesis. They cover the
classical text-to-speech (TTS) pipeline, autoregressive and non-autoregressive acoustic models,
neural vocoders, neural audio codecs, codec language models, voice cloning, evaluation, and
datasets. Later agents turn these notes into lessons and auto-checked labs. Where it makes
sense, each section follows the same pattern: intuition, exact equations, a hand-worked
example, references, student gotchas, and a lab idea with a compute budget.

Research date: 2026-09-28. Equations and hyperparameters for the classic papers were checked
against the full text of the arXiv PDFs. Facts about 2024-2026 models were checked against
Hugging Face model cards, the Hugging Face model API (license and parameter-count fields), and
GitHub READMEs. Items that couldn't be verified are listed in
[Unverified or uncertain facts](#unverified-or-uncertain-facts).

Flow matching, diffusion, and GAN theory are covered in depth in
`04-flows-diffusion-generative.md`. This file mentions them only where a TTS or codec system
depends on them.

Compute labels used in lab ideas:

- **CPU-60s**: runs in the browser editor's CPU checker in under 60 seconds with numpy or
  torch-cpu. Any audio, mel spectrograms, or codes come from a small fixture shipped with the lab.
- **GPU-L4** or **GPU-A100**: runs on a RunPod serverless worker. The label gives rough wall
  time and VRAM. Wall times are estimates from model sizes and published throughput, not
  measurements.

## Contents

1. [Conventions](#conventions)
2. [The TTS pipeline and its history](#the-tts-pipeline-and-its-history)
3. [Autoregressive acoustic models](#autoregressive-acoustic-models)
4. [Non-autoregressive acoustic models](#non-autoregressive-acoustic-models)
5. [End-to-end models: VITS family, StyleTTS 2, Kokoro](#end-to-end-models-vits-family-styletts-2-kokoro)
6. [Neural vocoders](#neural-vocoders)
7. [Vector quantization fundamentals](#vector-quantization-fundamentals)
8. [Neural audio codecs](#neural-audio-codecs)
9. [Codec language models](#codec-language-models)
10. [Voice cloning, prosody, emotion, and singing](#voice-cloning-prosody-emotion-and-singing)
11. [Evaluation](#evaluation)
12. [Datasets](#datasets)
13. [Checkpoints and packages for serverless labs](#checkpoints-and-packages-for-serverless-labs)
14. [Proposed lesson ordering](#proposed-lesson-ordering)
15. [Unverified or uncertain facts](#unverified-or-uncertain-facts)
16. [Bibliography](#bibliography)

---

## Conventions

- $x \in [-1, 1]^{T}$ is a waveform with $T$ samples at sample rate $f_s$.
- $X \in \mathbb{R}^{F \times N}$ is a (log-)mel spectrogram with $F$ mel bins and $N$ frames.
  The frame rate is $f_s / h$ where $h$ is the hop length in samples.
- $c_{1:L}$ is the input symbol sequence (characters or phonemes) of length $L$.
- $d_i \in \mathbb{Z}_{\ge 1}$ is the duration of symbol $i$ in frames, with
  $\sum_i d_i = N$.
- For codecs, $z_e \in \mathbb{R}^{D \times S}$ is the encoder output with $S$ frames per
  clip, $N_q$ is the number of quantizers, and $K$ is the codebook size.
- $\operatorname{sg}[\cdot]$ is the stop-gradient operator (`tensor.detach()` in PyTorch).
- Log means natural log unless the text writes $\log_2$ or $\log_{10}$.

Typical front-end settings the labs can standardize on:

| Setting | LJSpeech-style (Tacotron 2, HiFi-GAN V1) | LibriTTS-style (Vocos, BigVGAN-v2 24 kHz) |
|---|---|---|
| Sample rate | 22,050 Hz | 24,000 Hz |
| FFT size, window | 1024, 1024 (Hann) | 1024, 1024 (Hann) |
| Hop | 256 samples (about 11.6 ms, 86.1 frames per second) | 256 samples (93.75 frames per second) |
| Mel bins | 80 | 100 |
| Mel range | 0-8,000 Hz (HiFi-GAN); 125-7,600 Hz (Tacotron 2 paper) | 0-12,000 Hz |
| Compression | $\log(\max(X, 10^{-5}))$ | $\log(\max(X, 10^{-7}))$ |

The Tacotron 2 paper itself uses a 50 ms window and 12.5 ms hop at 24 kHz. The widely used
NVIDIA reimplementation switched to 1024/256 at 22,050 Hz, and HiFi-GAN followed it. A
mismatch between the acoustic model's mel settings and the vocoder's mel settings is the most
common reason student pipelines produce noise.

---

## The TTS pipeline and its history

### Intuition

TTS maps text to a waveform. The mapping is one-to-many: a sentence has countless valid
prosodies, speakers, and recording conditions. Every architecture in this file is a way to
split that hard problem into easier pieces or to model the one-to-many distribution directly.

### Pipeline stages

The classic modular pipeline contains the following stages:

1. **Text normalization.** Converts written forms to spoken forms: `Dr.` to "doctor" or
   "drive," `$3.50` to "three dollars fifty," `1984` to "nineteen eighty-four" (year) or "one
   thousand nine hundred eighty-four" (number). Rule-based weighted finite-state transducers
   (Sparrowhawk, NeMo `nemo_text_processing`) dominate production. LLM-based TTS systems often
   skip explicit normalization and learn it from data, which is a common source of errors on
   numbers and abbreviations.
2. **Grapheme-to-phoneme (G2P) conversion.** Maps words to phonemes (ARPAbet or IPA) with a
   pronunciation lexicon (CMUdict has about 134k entries) plus a neural or rule-based fallback
   for out-of-vocabulary words. Heteronyms ("read," "lead," "live") need context. Tools:
   `phonemizer` with the espeak-ng backend, `g2p_en`, `misaki` (Kokoro), `gruut`.
3. **Linguistic features.** Classical systems add stress, syllable and word boundaries,
   part-of-speech tags, phrase breaks, and positional features (the HTS "full-context label"
   has dozens of fields). Neural systems keep only phonemes or characters plus optional
   punctuation and prosody tokens.
4. **Acoustic model.** Predicts an intermediate acoustic representation, usually an 80- or
   100-bin log-mel spectrogram, or discrete codec tokens in codec language models.
5. **Vocoder.** Converts the acoustic representation to a waveform.

### History in brief

- **Concatenative and unit-selection synthesis (1990s-2010s).** A large single-speaker
  database is cut into units (diphones, half-phones). At synthesis time a Viterbi search
  minimizes the sum of a *target cost* (how well a unit matches the desired linguistic and
  prosodic specification) and a *join cost* (spectral, F0, and energy discontinuity at unit
  boundaries) [Hunt and Black 1996]. Quality is excellent when good units exist and glitchy
  when they don't. It can't change voice or style without recording a new database.
- **HMM-based statistical parametric synthesis (HTS, 2000s).** Context-dependent hidden
  Markov models with decision-tree state clustering generate vocoder parameters (mel-cepstrum,
  log F0, aperiodicity) with explicit state durations. Maximum-likelihood parameter generation
  with dynamic (delta) features produces smooth trajectories [Zen, Tokuda, and Black 2009].
  The output is flexible and small but "buzzy" and over-smoothed, because the model predicts
  the conditional mean. DNN-based parametric synthesis (2013) replaced the decision trees with
  neural networks but kept the vocoder.
- **WORLD vocoder.** A classical source-filter vocoder that decomposes speech into F0
  (DIO or Harvest estimators), spectral envelope (CheapTrick), and band aperiodicity (D4C),
  and resynthesizes from them [Morise et al. 2016]. FastSpeech 2 uses `pyworld` for F0
  extraction. WORLD is still a good CPU baseline for "what does a non-neural vocoder sound
  like."
- **Neural era (2016 onward).** WaveNet (2016) replaced the vocoder, Tacotron (2017) replaced
  the front end and acoustic model, FastSpeech (2019) made acoustic models parallel, VITS
  (2021) made the whole thing end to end, and VALL-E (2023) recast TTS as language modeling
  over codec tokens.

### Gotchas

- Students often assume TTS input is raw text. Most research systems use phonemes, and
  quality on LJSpeech drops noticeably with characters on rare words.
- The same symbol inventory and the same normalization must be used at training and inference.

### Lab idea

- **Text normalization and G2P warm-up (CPU-60s).** Implement a tiny normalizer for cardinal
  numbers 0-9,999, years 1000-2099, and currency, then look up phonemes in a shipped CMUdict
  subset with a fallback letter-to-sound rule table. The checker compares outputs against a
  hidden list of 50 strings.

---

## Autoregressive acoustic models

### Tacotron (2017)

**Intuition.** A sequence-to-sequence model with attention reads characters and writes
spectrogram frames, removing the hand-built linguistic front end.

**Architecture facts.** The encoder is a character embedding, a pre-net, and a CBHG module
(1-D convolution bank, highway network, bidirectional GRU). The decoder uses GRUs with
content-based attention and predicts $r$ frames per step (the *reduction factor*; the paper
reports MOS with $r = 2$ and notes $r = 5$ also works). A post-processing CBHG predicts a
linear spectrogram, and Griffin-Lim inverts it. MOS was 3.82 on US English
[Wang et al. 2017, arXiv:1703.10135].

### Tacotron 2 (2018)

**Intuition.** Keep Tacotron's attention-based sequence-to-sequence idea, simplify the building
blocks, predict mel spectrograms, and replace Griffin-Lim with a WaveNet vocoder.

**Exact architecture** [Shen et al. 2018, arXiv:1712.05884, section 2.2]:

- **Input.** Characters, learned 512-dimensional embedding.
- **Encoder.** Three 1-D convolution layers, each with 512 filters of shape $5 \times 1$
  (each filter spans five characters), batch normalization, and ReLU. Then one bidirectional
  LSTM with 512 units total (256 per direction).
- **Attention.** Location-sensitive attention [Chorowski et al. 2015]. Inputs and location
  features are projected to 128-dimensional hidden representations. Location features come
  from 32 1-D convolution filters of length 31 applied to the cumulative attention weights.
- **Pre-net.** Two fully connected layers of 256 ReLU units. Dropout with $p = 0.5$ is applied
  in the pre-net **at inference time as well as training time**. The paper states the pre-net
  bottleneck "was essential for learning attention."
- **Decoder.** The pre-net output and the previous attention context vector are concatenated
  and fed to two unidirectional LSTM layers with 1024 units. The LSTM output concatenated with
  the attention context goes through a linear projection to predict one 80-bin mel frame (no
  reduction factor: $r = 1$).
- **Stop token.** The same concatenation is projected to a scalar and passed through a sigmoid.
  Generation stops at the first frame where $p_{\text{stop}} > 0.5$.
- **Post-net.** Five 1-D convolution layers with 512 filters of shape $5 \times 1$, batch
  normalization, tanh on all but the last layer. It predicts a residual that's added to the
  decoder output.
- **Regularization.** Dropout 0.5 on convolutions, zoneout 0.1 on LSTMs.
- **Loss.** Summed mean squared error before and after the post-net, plus binary cross-entropy
  on the stop token (the stop-token loss is in the reference implementations; the paper
  describes the stop prediction but lists only the MSE terms explicitly):

$$
\mathcal{L} = \lVert X - \hat X_{\text{pre}} \rVert_2^2 + \lVert X - \hat X_{\text{post}} \rVert_2^2
+ \sum_{n} \operatorname{BCE}(s_n, \hat s_n),
$$

where $\hat X_{\text{post}} = \hat X_{\text{pre}} + \operatorname{PostNet}(\hat X_{\text{pre}})$
and $s_n = 1$ only for the final frame.

- **Training.** Teacher forcing, batch size 64 on one GPU, Adam with
  $\beta = (0.9, 0.999)$, $\epsilon = 10^{-6}$, learning rate $10^{-3}$ decaying exponentially to
  $10^{-5}$ after 50,000 iterations, L2 weight $10^{-6}$. The WaveNet vocoder was trained on
  ground-truth-aligned (teacher-forced) predicted mels. MOS was 4.53 versus 4.58 for
  professional recordings, on an internal 24.6-hour US English dataset.

**Location-sensitive attention equations** [Chorowski et al. 2015, arXiv:1506.07503]. With
decoder state $s_{i-1}$, encoder outputs $h_j$, and previous alignment $\alpha_{i-1}$:

$$
f_i = F * \alpha_{i-1}, \qquad
e_{i,j} = w^\top \tanh(W s_{i-1} + V h_j + U f_{i,j} + b), \qquad
\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k} \exp(e_{i,k})}.
$$

Tacotron 2 convolves the *cumulative* attention $\sum_{k<i} \alpha_k$ (concatenated with
$\alpha_{i-1}$ in the NVIDIA implementation) rather than only $\alpha_{i-1}$.

### Attention failure modes

- **Skipping.** Attention jumps forward past symbols, so words are dropped.
- **Repeating or babbling.** Attention moves backward or stalls, so words repeat or the model
  produces endless noise. The stop token never fires, so inference hits a maximum-length cap.
- **Early stop.** The stop token fires too early, truncating the utterance.
- **Failure to align at all.** A diffuse or constant alignment early in training. The model
  outputs an average-sounding blur. Symptoms appear on long or out-of-domain sentences, and
  they're the main motivation for duration-based models.

Mitigations include location-sensitive attention, forward attention, monotonic or stepwise
attention, GMM (location-based) attention, the reduction factor, the pre-net bottleneck, and
guided attention loss.

### Guided attention loss

**Intuition.** Text and speech are roughly linearly aligned, so penalize attention mass far
from the diagonal [Tachibana et al. 2018, DC-TTS, arXiv:1710.08969]:

$$
\mathcal{L}_{\text{att}}(A) = \mathbb{E}_{n,t}\left[A_{nt} W_{nt}\right], \qquad
W_{nt} = 1 - \exp\left(-\frac{(n/N - t/T)^2}{2 g^2}\right), \quad g = 0.2,
$$

where $A \in \mathbb{R}^{N \times T}$ is the attention matrix over $N$ text positions and $T$
output frames.

**Hand-worked example.** With $N = 4$, $T = 8$, $g = 0.2$: at $(n, t) = (2, 4)$,
$n/N - t/T = 0$, so $W = 0$. At $(n, t) = (0, 7)$, $n/N - t/T = -0.875$, so
$W = 1 - \exp(-0.765625/0.08) = 1 - e^{-9.57} \approx 0.99993$. Attention in the far corner
costs almost the full weight.

### Transformer TTS (2019)

Replaces the RNNs with Transformer encoder and decoder blocks and multi-head attention, keeps
the pre-net, post-net, and stop token, and uses scaled sinusoidal positional encodings with a
trainable scale. It trained about 4.25 times faster than Tacotron 2 and reached MOS 4.39
versus 4.44 for recordings [Li et al. 2019, arXiv:1809.08895]. Inference is still
autoregressive and attention robustness is worse on long sentences, which motivated FastSpeech.

### Gotchas for autoregressive models

- Turning pre-net dropout off at inference (for example, by calling `model.eval()` in a naive
  reimplementation) often collapses the output. The dropout must stay on.
- Teacher forcing makes training loss look good while free-running inference fails. Always
  check free-running alignments.
- Padding frames must be masked out of the MSE, and the stop-token target must mark the last
  real frame, not the last padded frame. Class imbalance (one positive per utterance) usually
  needs a positive weight in the BCE.
- The alignment plot (encoder steps versus decoder steps) is the single most useful debugging
  tool. A clean diagonal usually appears after about 10k-20k steps on LJSpeech with a working
  implementation.

### Lab ideas

- **Location-sensitive attention step (CPU-60s).** Implement one attention step with the
  location convolution. The checker verifies that the weights sum to 1, that the result matches
  a reference implementation on a fixture, and that the cumulative attention updates correctly.
- **Guided attention loss (CPU-60s).** Implement $W_{nt}$ and the loss. Check that a perfect
  diagonal alignment scores lower than an anti-diagonal one and matches a reference value to
  $10^{-6}$.
- **Tiny Tacotron overfit (GPU-L4, about 10 minutes, under 4 GB).** Train a shrunken Tacotron 2
  (embedding 128, one LSTM layer) on 50 LJSpeech clips with guided attention loss. The checker
  measures the diagonality of the alignment (fraction of attention mass within a band around
  the diagonal) and the mel L1 on training clips.

---

## Non-autoregressive acoustic models

### FastSpeech (2019)

**Intuition.** If you know how many frames each phoneme lasts, you can expand the phoneme
sequence to frame length and predict all frames in parallel with no attention between text and
speech at inference.

**Components** [Ren et al. 2019, arXiv:1905.09263]:

- Feed-forward Transformer (FFT) blocks: self-attention plus 1-D convolutions instead of the
  position-wise MLP.
- **Duration from a teacher.** An autoregressive Transformer TTS teacher is trained first. The
  encoder-decoder attention head that's most diagonal is chosen by the *focus rate*
  $F = \frac{1}{S}\sum_{s=1}^{S} \max_{1 \le t \le T} a_{s,t}$ (average over decoder steps $s$
  of the maximum attention weight). Durations are $d_i = \sum_{s} \mathbb{1}[\arg\max_t a_{s,t} = i]$.
- **Sequence-level knowledge distillation.** Targets are the teacher's generated mels, not
  ground truth, which simplifies the output distribution.
- **Duration predictor.** Two 1-D convolution layers with ReLU, layer norm, dropout, and a
  linear output. It predicts log durations with MSE loss.
- **Length regulator** $\mathcal{H}_{\text{mel}} = \operatorname{LR}(\mathcal{H}_{\text{pho}}, \mathcal{D}, \alpha)$,
  where $\alpha$ scales durations for speed control.

Reported speedups were 270 times for mel generation and 38 times end to end with WaveGlow.

### Length regulator

**Definition.** Given hidden states $h_1, \ldots, h_L$ and integer durations $d_1, \ldots, d_L$,
output $\tilde h_{1:N}$ with $h_i$ repeated $d_i$ times, $N = \sum_i d_i$. With speed factor
$\alpha$, use $d_i' = \operatorname{round}(\alpha d_i)$ (the paper uses $\alpha > 1$ for slower
speech).

**Hand-worked example.** $h = [h_1, h_2, h_3]$, $d = [2, 1, 3]$. The output is
$[h_1, h_1, h_2, h_3, h_3, h_3]$, $N = 6$. The equivalent alignment matrix
$A \in \{0,1\}^{3 \times 6}$ is

$$
A = \begin{bmatrix} 1 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 1 & 1 \end{bmatrix},
\qquad \tilde H = A^\top H.
$$

Cumulative sums $[2, 3, 6]$ give the boundaries. The paper's own example: $d = [2, 2, 3, 1]$
gives $[h_1, h_1, h_2, h_2, h_3, h_3, h_3, h_4]$.

**Implementation detail.** In PyTorch, use
`torch.repeat_interleave(h, d, dim=0)` per utterance, or build $A$ from cumulative sums for a
batched, differentiable-in-$h$ version: $A_{i,n} = \mathbb{1}[\,\text{cum}_{i-1} \le n < \text{cum}_i\,]$.
Pad each expanded sequence to the batch maximum and build a frame mask. Durations must be
integers and at least 1 at inference: $\hat d_i = \max(1, \operatorname{round}(\exp(\hat\ell_i) - 1))$
when the predictor targets $\log(d + 1)$.

### FastSpeech 2 (2020)

**Intuition.** Remove the teacher: take durations from a forced aligner, and feed the model the
extra "variance" information (pitch, energy) that makes the one-to-many mapping less ambiguous.

**Variance adaptor** [Ren et al. 2020, arXiv:2006.04558]:

- **Duration predictor.** Same structure as FastSpeech: two 1-D conv layers with ReLU, each
  followed by layer norm and dropout, then a linear layer. Targets are log-domain durations
  extracted with the Montreal Forced Aligner (MFA), trained on the training set only. Loss is
  MSE. MFA durations proved more accurate than teacher attention.
- **Pitch predictor.** F0 is extracted with PyWorld. Unvoiced frames are filled by linear
  interpolation, the contour is converted to log scale, normalized per utterance to zero mean
  and unit variance, and decomposed by a continuous wavelet transform (CWT) with a Mexican-hat
  mother wavelet into 10 scales:

$$
W(\tau, t) = \tau^{-1/2}\int F_0(x)\,\psi\!\left(\frac{x - t}{\tau}\right)dx, \qquad
W_i(t) = W(2^{i+1}\tau_0, t)\,(i + 2.5)^{-5/2}, \quad i = 1..10,\ \tau_0 = 5\text{ ms},
$$

$$
\hat F_0(t) = \sum_{i=1}^{10} \hat W_i(t)\,(i + 2.5)^{-5/2}.
$$

  The predictor outputs the 10-scale pitch spectrogram plus the utterance mean and variance
  (from a pooled hidden vector), all with MSE. For conditioning, F0 (ground truth in training,
  predicted in inference) is **quantized to 256 values in log scale** and looked up in an
  embedding table that's added to the expanded hidden sequence.
- **Energy predictor.** Energy is the L2 norm of each STFT frame's magnitude, quantized
  **uniformly** to 256 values and embedded. The predictor regresses the unquantized value with
  MSE. CWT isn't used for energy.
- **Loss.** $\mathcal{L} = \lVert X - \hat X \rVert_1$ (mel, L1 in common implementations) plus
  the three MSE terms.
- Training is about 3 times faster than FastSpeech because there's no teacher.

**Implementation note.** Most open implementations (for example `ming024/FastSpeech2`) skip
CWT and predict phoneme-averaged or frame-level F0 directly, which is simpler and works well.
Teach CWT as optional depth. Pitch bins are typically `np.exp(np.linspace(log(f_min), log(f_max), 255))`
with `torch.bucketize`.

**Hand-worked quantization example.** With $f_{\min} = 71$ Hz, $f_{\max} = 795$ Hz, and 256
log-spaced bins, the bin width ratio is $(795/71)^{1/255} \approx 1.00951$, about 16 cents. An
F0 of 200 Hz falls in bin $\lfloor \ln(200/71)/\ln(1.00951) \rfloor = \lfloor 1.0357/0.009465 \rfloor = 109$.

### Glow-TTS and monotonic alignment search (2020)

**Intuition.** Learn the alignment inside the model by maximum likelihood. A normalizing flow
maps mels to a latent $z$ that should look Gaussian with mean and scale predicted per text
token. For a fixed model, find the monotonic alignment that maximizes likelihood (a Viterbi
step), then update the model with that alignment fixed (a hard EM step).

**Objective** [Kim et al. 2020, arXiv:2005.11129]. With prior statistics $(\mu_i, \sigma_i)$
for text position $i$ and flow $f_{\text{dec}}$, $z = f_{\text{dec}}^{-1}(X)$:

$$
\log p_X(X \mid c) = \log p_Z(z \mid c, A) + \log\left|\det \frac{\partial f_{\text{dec}}^{-1}(X)}{\partial X}\right|,
\qquad
\log p_Z(z \mid c, A) = \sum_{j=1}^{T_{\text{mel}}} \log \mathcal{N}(z_j; \mu_{A(j)}, \sigma_{A(j)}).
$$

**Monotonic alignment search (MAS).** Let $P_{i,j} = \log \mathcal{N}(z_j; \mu_i, \sigma_i)$
and let $Q_{i,j}$ be the maximum log-likelihood of aligning the first $j$ frames to the first
$i$ tokens, with the alignment monotonic and surjective (every token gets at least one frame,
no skips):

$$
Q_{i,j} = \max(Q_{i-1,j-1},\ Q_{i,j-1}) + P_{i,j}, \qquad
Q_{1,j} = \sum_{k=1}^{j} P_{1,k}, \qquad Q_{i,j} = -\infty \text{ for } i > j.
$$

Backtrack from $A^*(T_{\text{mel}}) = T_{\text{text}}$:
$A^*(j) = \arg\max_{i \in \{A^*(j+1) - 1,\ A^*(j+1)\}} Q_{i,j}$. Complexity is
$O(T_{\text{text}} \times T_{\text{mel}})$. It runs on CPU (under 20 ms per iteration in the
paper, less than 2% of training time) and is not needed at inference.

Durations are $d_i = \sum_j \mathbb{1}[A^*(j) = i]$. The duration predictor is trained with
$\mathcal{L}_{\text{dur}} = \operatorname{MSE}(f_{\text{dur}}(\operatorname{sg}[f_{\text{enc}}(c)]), d)$
(log durations in the code). At inference, sampling uses temperature $T$ on $\sigma$; 0.333 gave
the best MOS. Glow-TTS synthesized 15.7 times faster than Tacotron 2.

**Hand-worked MAS example.** Three tokens, four frames, with log-likelihoods

$$
P = \begin{bmatrix} -1 & -2 & -6 & -8 \\ -5 & -1 & -1 & -7 \\ -9 & -6 & -3 & -1 \end{bmatrix}
\quad\Rightarrow\quad
Q = \begin{bmatrix} -1 & -3 & -9 & -17 \\ -\infty & -2 & -3 & -10 \\ -\infty & -\infty & -5 & -4 \end{bmatrix}.
$$

For example, $Q_{2,3} = \max(Q_{1,2}, Q_{2,2}) + P_{2,3} = \max(-3, -2) - 1 = -3$. Backtracking
from $(3, 4)$ gives the path $[1, 2, 2, 3]$, durations $[1, 2, 1]$, and total log-likelihood
$-4$. Brute-force enumeration of all $\binom{3}{2} = 3$ segmentations confirms $-4$ is the
maximum. (Computed and checked with a script during research.)

**Reference code pattern** (the VITS Cython `maximum_path`), which the lab can mirror:

```python
def maximum_path(value, t_x, t_y):
    # value: [t_x, t_y] log-likelihoods, modified in place into Q
    NEG = -1e9
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = NEG if x == y else value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else NEG
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] += max(v_prev, v_cur)
    path = [[0] * t_y for _ in range(t_x)]
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index][y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index -= 1
    return path
```

The loop bound `max(0, t_x + y - t_y)` also forbids states from which the last token can't be
reached. Fast variants: `monotonic_align` (Cython), numba versions, and Super-MAS, a Triton
kernel reported 19 to 72 times faster than Cython [Lee et al. 2024, arXiv:2409.07704].

**Gotchas.**

- Off-by-one errors in the initial column and forgetting the "can't be ahead of the diagonal"
  constraint ($Q_{i,j} = -\infty$ for $i > j$).
- Computing $P$ naively in a loop. Vectorize: $P_{i,j} = -\tfrac12 \sum_d \left[\log(2\pi) + 2\log\sigma_{i,d} + (z_{j,d} - \mu_{i,d})^2 / \sigma_{i,d}^2\right]$
  expands into matrix products.
- MAS must run under `torch.no_grad()`; gradients flow only through $P$ evaluated on the chosen
  path.

### Matcha-TTS (2023, mention only)

An encoder with a duration predictor and MAS (as in Glow-TTS), plus a 1-D U-Net decoder
trained with optimal-transport conditional flow matching (OT-CFM), which synthesizes good mels
in few ODE steps. It has 18.2M parameters, the smallest memory footprint in its comparison
[Mehta et al. 2024, arXiv:2309.03199]. Package `matcha-tts`, MIT license. Flow matching itself
is covered in file 04.

### Lab ideas

- **Length regulator (CPU-60s).** Implement batched length regulation with padding masks and
  the speed factor. Check `[2, 1, 3]` expands to 6 frames, check against
  `torch.repeat_interleave`, and check the mask shape.
- **MAS versus brute force (CPU-60s).** Implement MAS in numpy. The checker compares against
  exhaustive enumeration on random $3 \times 5$ to $5 \times 9$ matrices, and against the
  reference Cython output on a $40 \times 200$ fixture.
- **Pitch and energy features (CPU-60s).** Given a waveform fixture, compute frame energy as
  the STFT L2 norm, F0 with `librosa.pyin`, linear interpolation over unvoiced frames, and
  256-bin log quantization. Check against reference arrays.
- **Tiny FastSpeech 2 (GPU-L4, about 15 minutes, under 6 GB).** Train a 4-layer
  FastSpeech 2 with MFA durations shipped as a fixture on 500 LJSpeech clips. Check that
  validation mel L1 drops below a threshold and that the duration predictor's log MSE is below a
  threshold. Vocode with a pretrained HiFi-GAN for listening only.

---

## End-to-end models: VITS family, StyleTTS 2, Kokoro

### VITS (2021)

**Intuition.** Train the acoustic model and the vocoder jointly as a conditional VAE. The
posterior encoder sees the real linear spectrogram, the prior comes from text through a
normalizing flow, the decoder is a HiFi-GAN generator, and MAS aligns text and latents. A
stochastic duration predictor adds rhythm diversity.

**Components** [Kim, Kong, and Son 2021, arXiv:2106.06103]:

- **Posterior encoder** $q_\phi(z \mid x_{\text{lin}})$: non-causal WaveNet residual blocks on
  the linear spectrogram (higher resolution than mel), producing a factorized Gaussian.
- **Prior** $p_\theta(z \mid c_{\text{text}}, A)$: a Transformer text encoder gives
  $(\mu_\theta, \sigma_\theta)$ per phoneme, expanded by the alignment $A$, and a normalizing
  flow $f_\theta$ (affine coupling layers built from WaveNet blocks, volume-preserving) makes
  the prior more expressive: $p_\theta(z \mid c) = \mathcal{N}(f_\theta(z); \mu_\theta(c), \sigma_\theta(c))\left|\det \partial f_\theta(z)/\partial z\right|$.
- **Decoder** $G$: HiFi-GAN V1 generator operating on random windows (segments) of $z$, which
  keeps memory manageable.
- **Alignment.** MAS on the ELBO-derived likelihood of $f_\theta(z)$ under the prior.
- **Stochastic duration predictor (SDP).** A flow-based model of the duration distribution.
  Because durations are discrete scalars, it uses *variational dequantization* (a noise
  $u \in [0, 1)$ subtracted from $d$) and *variational data augmentation* (an extra channel
  $\nu$ concatenated to $d$), trained by the bound

$$
\log p_\theta(d \mid c) \ge \mathbb{E}_{q_\phi(u,\nu \mid d, c)}\left[\log \frac{p_\theta(d - u, \nu \mid c)}{q_\phi(u, \nu \mid d, c)}\right].
$$

  Inputs are stop-gradiented. At inference, sample noise, invert the flow, and round.
- **Losses.**

$$
\mathcal{L}_{\text{recon}} = \lVert x_{\text{mel}} - \hat x_{\text{mel}} \rVert_1, \qquad
\mathcal{L}_{\text{kl}} = \log q_\phi(z \mid x_{\text{lin}}) - \log p_\theta(z \mid c_{\text{text}}, A),
$$

$$
\mathcal{L}_{\text{vae}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{dur}} + \mathcal{L}_{\text{adv}}(G) + \mathcal{L}_{\text{fm}}(G),
$$

  with LSGAN adversarial and feature-matching losses identical to HiFi-GAN. The paper writes the
  sum unweighted; the official config weights the mel term by $c_{\text{mel}} = 45$ and the KL by
  $c_{\text{kl}} = 1$ (see [Unverified or uncertain facts](#unverified-or-uncertain-facts)).
- **Training.** 4 V100 GPUs, batch 64 per GPU, up to 800k steps, mixed precision. MOS on
  LJSpeech: VITS 4.43, ground truth 4.46, Glow-TTS + HiFi-GAN fine-tuned 4.32, Tacotron 2 +
  HiFi-GAN fine-tuned 4.25. At inference, the SDP noise scale was 0.8 and the prior noise scale
  0.667.

**Gotchas.** VITS is the most complex model in this part of the course. Students confuse the
direction of the flow (the prior flow maps posterior $z$ toward the text-conditioned Gaussian
during training and runs inverted at inference), and forget that the decoder only sees random
segments during training.

### VITS2 (2023)

Adds adversarial training of the duration predictor, a Transformer block inside the normalizing
flow for long-range dependencies, a speaker-conditioned text encoder, and MAS with Gaussian noise
added to $Q$ early in training to encourage exploration:
$Q_{i,j} = \max(Q_{i-1,j-1}, Q_{i,j-1}) + P_{i,j} + \epsilon$. It also reduces dependence on
phoneme conversion (trains on normalized text) [Kong et al. 2023, arXiv:2307.16430].

### YourTTS (2022)

A multilingual, multi-speaker VITS for zero-shot cloning. It conditions on an external H/ASP
speaker encoder and adds a speaker consistency loss

$$
\mathcal{L}_{\text{SCL}} = -\frac{\alpha}{n}\sum_{i=1}^{n} \cos(\phi(g_i), \phi(h_i)),
$$

where $\phi$ is the frozen speaker encoder, $g_i$ the ground truth, and $h_i$ the generated
audio. It was trained on English, Portuguese, and French [Casanova et al. 2022,
arXiv:2112.02418]. Coqui TTS implementation, MPL-2.0 code.

### StyleTTS 2 (2023)

Models style as a fixed-length vector sampled by a diffusion model conditioned on text (so no
reference audio is required), injects style through adaptive instance normalization (AdaIN)
into the decoder and prosody predictors, and trains end to end with a large speech language
model (WavLM) as a discriminator plus differentiable duration modeling. It reported CMOS +0.28
over recordings on LJSpeech and +1.07 over NaturalSpeech [Li et al. 2023, arXiv:2306.07691].
MIT code.

### Kokoro-82M (2024-2025)

- 82M parameters, Apache-2.0 weights, StyleTTS 2 architecture without the diffusion style
  sampler and without an encoder release ("decoder only"), with an iSTFTNet vocoder at 24 kHz.
- Voices are precomputed style vectors (about 0.5 MB files per voice), not arbitrary zero-shot
  cloning.
- G2P uses the `misaki` library. v1.0 (January 27, 2025) supports 8 languages and 54 voices.
  The model card states training took about 1,000 A100 80GB hours (about $1,000) on a few
  hundred hours of permissively licensed and synthetic audio.
- Package `kokoro>=0.9.2`. The checkpoint `kokoro-v1_0.pth` is about 327 MB and runs faster than
  real time on a laptop CPU, which makes it a good "inference on CPU" demo.

### Lab ideas

- **VITS KL term (CPU-60s).** Given fixture tensors for posterior mean and log-scale, a sampled
  $z$, prior statistics, and the flow log-determinant, implement the VITS KL term (the
  single-sample estimate used in the official code) and match a reference value.
- **Kokoro inference and speed (CPU-60s if weights are pre-cached, otherwise GPU-L4 under 2
  minutes).** Synthesize three sentences, measure real-time factor, and check duration scales
  with the `speed` parameter.

---

## Neural vocoders

A vocoder maps a mel spectrogram $X \in \mathbb{R}^{F \times N}$ to a waveform of $N h$ samples.
The core difficulty is phase: the mel discards phase and fine frequency detail, so the vocoder
must generate them plausibly. Families: autoregressive (WaveNet, WaveRNN, LPCNet), flow-based
(Parallel WaveNet, ClariNet, WaveGlow), GAN-based (MelGAN, Parallel WaveGAN, HiFi-GAN, UnivNet,
BigVGAN), Fourier-domain GAN (iSTFTNet, Vocos), and diffusion (DiffWave, WaveGrad; see file 04).

### WaveNet (2016)

**Intuition.** Model $p(x) = \prod_t p(x_t \mid x_{<t}, h)$ with a deep stack of dilated causal
convolutions, so the receptive field grows exponentially with depth while the network stays
fully convolutional [van den Oord et al. 2016, arXiv:1609.03499].

**Exact components.**

- **Dilated causal convolution.** Kernel size $k = 2$ with dilations doubling up to a limit and
  repeating: $1, 2, 4, \ldots, 512, 1, 2, \ldots, 512, 1, 2, \ldots, 512$.
- **Receptive field.** For $B$ stacks of dilations $1..2^{M-1}$ and kernel size $k$,

$$
R = 1 + (k - 1)\,B \sum_{m=0}^{M-1} 2^m = 1 + (k-1)\,B\,(2^M - 1).
$$

  Each $1..512$ block alone has $R = 1024$ (the paper's "counterpart of a $1 \times 1024$
  convolution").
- **Gated activation unit.** $z = \tanh(W_{f,k} * x) \odot \sigma(W_{g,k} * x)$.
- **Residual and skip connections.** Each layer outputs a residual ($1 \times 1$ conv added to
  its input) and a skip ($1 \times 1$ conv). The skips are summed, then ReLU, $1 \times 1$,
  ReLU, $1 \times 1$, softmax.
- **Mu-law companding and 256-way softmax.**

$$
f(x_t) = \operatorname{sign}(x_t)\,\frac{\ln(1 + \mu |x_t|)}{\ln(1 + \mu)}, \quad -1 < x_t < 1,\ \mu = 255,
$$

  then quantize $f(x_t) \in (-1, 1)$ to 256 levels:
  $q = \lfloor (f + 1)/2 \cdot 255 + 0.5 \rfloor$.
- **Conditioning.** Global (speaker embedding $h$):
  $z = \tanh(W_{f,k} * x + V_{f,k}^\top h) \odot \sigma(W_{g,k} * x + V_{g,k}^\top h)$. Local
  (linguistic features or mel $h_t$ at a lower frame rate): upsample $h$ to the sample rate with
  transposed convolutions (or repetition) to get $y = f(h)$, then
  $z = \tanh(W_{f,k} * x + V_{f,k} * y) \odot \sigma(W_{g,k} * x + V_{g,k} * y)$.

**Hand-worked examples.**

- Receptive field for dilations $1..512$ ($M = 10$), 3 stacks, $k = 2$:
  $R = 1 + 1 \cdot 3 \cdot 1023 = 3070$ samples. Adding the initial causal convolution with
  $k = 2$ gives 3071. At 16 kHz that's about 192 ms; at 24 kHz, 128 ms.
- Mu-law: $x = 0.5$ gives $f = \ln(128.5)/\ln(256) = 4.856/5.545 = 0.8757$, bin $239$.
  $x = 0.001$ gives $f = 0.0410$, bin 133. $x = -0.01$ gives $f = -0.2285$, bin 98. Small
  amplitudes get many more bins than a linear quantizer would give them.

**Cost.** Generation is one network pass per sample: 24,000 sequential passes per second of
24 kHz audio. The Tacotron 2 WaveNet used a mixture of logistics (MoL) output at 24 kHz
instead of the 256-way softmax. In HiFi-GAN's comparison, WaveNet (MoL, 24.73M parameters)
ran at 0.07 kHz on a V100 (0.003 times real time).

**Gotchas.** Causality leaks: forgetting to left-pad by $(k-1)d$ or using `padding='same'`. Off
by one in the target shift (predict $x_t$ from $x_{<t}$). Mixing up mu-law encoding (before
quantization) and decoding $f^{-1}(y) = \operatorname{sign}(y)\,((1+\mu)^{|y|} - 1)/\mu$.

### Parallel WaveNet and ClariNet (brief)

Parallel WaveNet distills an autoregressive WaveNet teacher into an inverse autoregressive flow
(IAF) student that generates all samples in parallel from logistic noise, trained with
*probability density distillation* (reverse KL from student to teacher) plus power, perceptual,
and contrastive losses [van den Oord et al. 2018, arXiv:1711.10433]. ClariNet simplifies this
with a single Gaussian output distribution for the teacher, which makes the KL closed form, and
trains text-to-wave end to end [Ping, Peng, and Chen 2019, arXiv:1807.07281]. Historically
important; too complex for a lab.

### WaveRNN (2018)

**Intuition.** Make autoregressive generation cheap: a single GRU layer with 896 units instead
of a deep convolution stack, plus a *dual softmax* that splits each 16-bit sample into 8 coarse
bits and 8 fine bits, each predicted with a 256-way softmax (the fine softmax conditions on the
coarse bits of the current sample). Weight sparsification (pruning to over 96% sparsity) keeps
quality while shrinking compute enough for mobile CPUs. Subscale WaveRNN generates 16 samples
per step by folding the sequence [Kalchbrenner et al. 2018, arXiv:1802.08435]. Sample value
$v = 256\,c + f$ with $c, f \in \{0..255\}$.

### LPCNet (2019)

Combines classical linear prediction with a small WaveRNN: the linear predictor
$p_t = \sum_{k=1}^{16} a_k s_{t-k}$ (coefficients computed from the input cepstrum) handles the
spectral envelope, so the network only models the spectrally flat excitation $e_t = s_t - p_t$
in 8-bit mu-law. It uses pre-emphasis, a frame-rate network and a sample-rate network at 16 kHz,
and runs in under 3 GFLOPS [Valin and Skoglund 2019, arXiv:1810.11846]. A good reading
assignment on "put signal processing back in."

### WaveGlow (2018, brief)

A Glow-style flow (affine coupling with WaveNet-like transforms, invertible $1 \times 1$
convolutions) trained by exact maximum likelihood on waveforms conditioned on mels
[Prenger, Valle, and Catanzaro 2019, arXiv:1811.00002]. About 88M parameters (87.73M in the
HiFi-GAN comparison), MOS 3.81 in that comparison. Parallel at inference but heavy. Flows are
covered in file 04.

### MelGAN (2019)

The first widely used GAN vocoder: a fully convolutional generator (transposed-convolution
upsampling with dilated residual stacks), a multi-scale discriminator on raw, 2x, and 4x
average-pooled audio, hinge adversarial loss, and a feature-matching loss with weight 10
[Kumar et al. 2019, arXiv:1910.06711]. 4.26M parameters. MOS 3.79 in the HiFi-GAN comparison but
very fast.

### Parallel WaveGAN (2020) and the multi-resolution STFT loss

**Intuition.** A non-causal WaveNet generator that maps Gaussian noise plus upsampled mel to
waveform in one pass, trained with an adversarial loss plus a spectral reconstruction loss at
several STFT resolutions. The multi-resolution STFT loss is the workhorse auxiliary loss for
vocoders and codecs [Yamamoto, Song, and Kim 2020, arXiv:1910.11480].

**Exact formula.** For one STFT resolution $m$:

$$
L_{\text{sc}}(x, \hat x) = \frac{\lVert\, |\operatorname{STFT}(x)| - |\operatorname{STFT}(\hat x)|\,\rVert_F}{\lVert\, |\operatorname{STFT}(x)|\,\rVert_F},
\qquad
L_{\text{mag}}(x, \hat x) = \frac{1}{N}\,\lVert \log|\operatorname{STFT}(x)| - \log|\operatorname{STFT}(\hat x)| \rVert_1,
$$

$$
L_s^{(m)} = L_{\text{sc}} + L_{\text{mag}}, \qquad
L_{\text{aux}} = \frac{1}{M}\sum_{m=1}^{M} L_s^{(m)}, \qquad
L_G = L_{\text{aux}} + \lambda_{\text{adv}} L_{\text{adv}},\ \lambda_{\text{adv}} = 4.0,
$$

where $\lVert\cdot\rVert_F$ is the Frobenius norm and $N$ is the number of magnitude elements.
The three resolutions (Hann window, 24 kHz audio) are:

| Loss | FFT size | Window | Hop |
|---|---|---|---|
| $L^{(1)}$ | 1024 | 600 (25 ms) | 120 (5 ms) |
| $L^{(2)}$ | 2048 | 1200 (50 ms) | 240 (10 ms) |
| $L^{(3)}$ | 512 | 240 (10 ms) | 50 (about 2 ms) |

**Other facts.** Generator: 30 dilated residual conv layers, three dilation cycles, 64 residual
and skip channels, kernel 3. Discriminator: 10 non-causal dilated conv layers. 1.44M parameters,
400k steps with RAdam, discriminator frozen for the first 100k steps, batch 8 of 1-second
clips. Training took 2.8 days on two V100 GPUs (versus 13.5 days for the ClariNet baseline).
MOS 4.16, 28.68 times faster than real time on one GPU.

**Hand-worked example.** If $|\operatorname{STFT}(x)|$ has Frobenius norm 10 and the error
matrix has Frobenius norm 2, $L_{\text{sc}} = 0.2$. If every magnitude is off by a factor of
$e^{0.1}$, $L_{\text{mag}} = 0.1$ regardless of scale. $L_{\text{sc}}$ emphasizes loud bins;
$L_{\text{mag}}$ treats quiet and loud bins equally.

**Gotchas.** Taking $\log$ of zero magnitudes (clamp at $10^{-7}$ first). Forgetting to average
(not sum) over resolutions. Using `torch.stft` without `return_complex=True` and then taking
the wrong norm.

### HiFi-GAN (2020)

**Intuition.** Speech is a sum of periodic signals. Give the discriminator sub-networks that
look at the waveform reshaped by different periods, and give the generator parallel residual
blocks with different receptive fields [Kong, Kim, and Bae 2020, arXiv:2010.05646].

**Generator.** An initial conv from 80 mel channels to $h_u$ channels, then transposed
convolutions that upsample by the hop length (V1: kernel sizes $k_u = [16, 16, 4, 4]$, strides
$[8, 8, 2, 2]$, product 256), halving channels each time. After each upsampling layer, a
*multi-receptive field fusion* (MRF) module sums the outputs of $|k_r|$ residual blocks with
kernel sizes $k_r$ and dilations $D_r$. Leaky ReLU, final conv to 1 channel, tanh.

| Variant | $h_u$ | $k_u$ | $k_r$ | $D_r$ | Params | MOS (LJSpeech) |
|---|---|---|---|---|---|---|
| V1 | 512 | [16, 16, 4, 4] | [3, 7, 11] | [[1,1], [3,1], [5,1]] for each $k_r$ | 13.92M | 4.36 |
| V2 | 128 | [16, 16, 4, 4] | [3, 7, 11] | same as V1 | 0.92M | 4.23 |
| V3 | 256 | [16, 16, 8] | [3, 5, 7] | [[1], [2]], [[2], [6]], [[3], [12]] | 1.46M | 4.05 |

Ground truth MOS was 4.45. V1 runs at 3.7 MHz (167.9 times real time) on a V100; V3 runs 13.44
times real time on a laptop CPU. In the V1 and V2 residual blocks, each dilation pair
$[d, 1]$ is conv (dilation $d$) then conv (dilation 1) with one residual connection, repeated
three times.

**Discriminators.**

- **Multi-period discriminator (MPD).** Five sub-discriminators with periods
  $p \in \{2, 3, 5, 7, 11\}$ (primes, to avoid overlap). Each reshapes the 1-D waveform of length
  $T$ into 2-D of shape $(T/p, p)$ (reflect-padding $T$ to a multiple of $p$) and applies 2-D
  convolutions with kernel width 1 along the period axis, so each column is processed
  independently. Weight normalization.
- **Multi-scale discriminator (MSD).** From MelGAN: three sub-discriminators on raw, 2x, and 4x
  average-pooled audio; strided grouped 1-D convs with leaky ReLU. Spectral normalization on
  the raw-scale one, weight normalization on the others.

**Losses (exact).**

$$
\mathcal{L}_{\text{Adv}}(D; G) = \mathbb{E}_{(x,s)}\left[(D(x) - 1)^2 + (D(G(s)))^2\right], \qquad
\mathcal{L}_{\text{Adv}}(G; D) = \mathbb{E}_{s}\left[(D(G(s)) - 1)^2\right],
$$

$$
\mathcal{L}_{\text{Mel}}(G) = \mathbb{E}_{(x,s)}\left[\lVert \phi(x) - \phi(G(s)) \rVert_1\right], \qquad
\mathcal{L}_{\text{FM}}(G; D) = \mathbb{E}_{(x,s)}\left[\sum_{i=1}^{T} \frac{1}{N_i}\lVert D^i(x) - D^i(G(s)) \rVert_1\right],
$$

$$
\mathcal{L}_G = \sum_{k=1}^{K}\left[\mathcal{L}_{\text{Adv}}(G; D_k) + \lambda_{\text{fm}}\mathcal{L}_{\text{FM}}(G; D_k)\right] + \lambda_{\text{mel}}\mathcal{L}_{\text{Mel}}(G), \qquad
\mathcal{L}_D = \sum_{k=1}^{K}\mathcal{L}_{\text{Adv}}(D_k; G),
$$

with $\lambda_{\text{fm}} = 2$ and $\lambda_{\text{mel}} = 45$ (verified in the paper, equation
5). $\phi$ is the mel transform with the same 80-band, 1024/1024/256 settings as the input.
$K = 8$ sub-discriminators (5 MPD + 3 MSD). LSGAN, not hinge.

**Training.** AdamW, $\beta_1 = 0.8$, $\beta_2 = 0.99$, weight decay 0.01, learning rate
$2 \times 10^{-4}$ decayed by 0.999 every epoch. All models were trained to 2.5M steps; the
official config uses batch 16 of 8,192-sample segments. This is several days on one GPU; a lab
can only show early training. Official checkpoints (LJSpeech V1-V3, VCTK, universal) are MIT
licensed in `jik876/hifi-gan`. SpeechBrain hosts `speechbrain/tts-hifigan-ljspeech` (Apache-2.0,
about 56 MB).

**Gotchas.**

- The mel loss mel function must exactly match the input features (including `fmax`); the
  official code uses `fmax=None` (full band) for the loss mel but 8 kHz for input features.
- Detach $G(s)$ for the discriminator step; don't detach for the generator step.
- The discriminator must see real and fake segments at the same positions.
- Transposed convolutions with kernel not divisible by stride cause checkerboard artifacts; HiFi-GAN
  uses kernel $= 2 \times$ stride and padding $(k - s)/2$.

### UnivNet (2021)

A universal vocoder with a generator built from *location-variable convolutions* (LVC), whose
kernels are predicted from the mel by a kernel predictor, plus a multi-resolution spectrogram
discriminator (MRD) on linear magnitudes at several STFT settings, alongside MPD
[Jang et al. 2021, arXiv:2106.07889]. BigVGAN and Vocos adopted MRD.

### BigVGAN (2022) and BigVGAN-v2 (2024)

**Intuition.** Make a GAN vocoder universal (unseen speakers, languages, singing, music,
environments) by giving the generator a periodic inductive bias and anti-aliasing, then scale
it up [Lee et al. 2023, arXiv:2206.04658].

**Snake activation.**

$$
f_\alpha(x) = x + \frac{1}{\alpha}\sin^2(\alpha x),
$$

with a trainable $\alpha \in \mathbb{R}^h$ per channel. Larger $\alpha$ gives a higher-frequency
periodic component; $\sin^2$ keeps the function monotonic in its trend. BigVGAN-v2 and DAC use
the "SnakeBeta" or log-parameterized variants in code
($x + \frac{1}{\beta + \epsilon}\sin^2(\alpha x)$ with $\alpha, \beta$ stored as logs).

**Hand-worked example.** $x = 0.5$, $\alpha = 1$: $0.5 + \sin^2(0.5) = 0.5 + 0.2298 = 0.7298$.
$\alpha = 5$: $0.5 + 0.2\sin^2(2.5) = 0.5 + 0.2 \times 0.3582 = 0.5716$.

**Anti-aliased multi-periodicity composition (AMP).** Each residual dilated conv layer applies
the Snake nonlinearity to a 2x upsampled signal, then downsamples by 2. Both resampling steps use
a windowed sinc low-pass filter with a Kaiser window. The nonlinearity creates harmonics above
Nyquist; the filter suppresses aliasing of them. Anti-aliased *upsampling layers* were tried and
hurt, so only the activations are filtered.

**Sizes and training.** BigVGAN-base mirrors HiFi-GAN V1 (14M parameters; upsampling
$[8, 8, 2, 2]$). BigVGAN splits 256x upsampling into 6 blocks with 1536 initial channels, 112M
parameters. MSD is replaced by MRD. Batch 32 (doubled from 16 to reduce collapse), segment 8,192,
1M steps, gradient clipping for the 112M model, trained on LibriTTS at 24 kHz with 100 mel bands
up to 12 kHz.

**BigVGAN-v2 (July 2024).** MIT-licensed checkpoints on Hugging Face trained for 5M steps (3M-step
checkpoints also provided) on large diverse data (multilingual speech, environmental sounds,
instruments), with an optional fused CUDA kernel for the anti-aliased activation (1.5-3 times
faster on an A100). Checkpoints: `nvidia/bigvgan_v2_24khz_100band_256x`,
`nvidia/bigvgan_v2_22khz_80band_256x`, `nvidia/bigvgan_v2_22khz_80band_fmax8k_256x`,
`nvidia/bigvgan_v2_44khz_128band_256x`, `nvidia/bigvgan_v2_44khz_128band_512x`. The generator file
is about 450 MB (112M parameters, fp32). The discriminator and optimizer states (about 1.4 GB)
are also published, which enables fine-tuning labs.

### iSTFTNet (2022)

Replaces the last upsampling stages of HiFi-GAN with an inverse STFT: the network upsamples mel
frames only partway (for example, two 8x stages, "C8C8I") and predicts magnitude and phase for a
tiny STFT (FFT size 16, hop 4), then an iSTFT produces the waveform. It's faster with similar
quality [Kaneko et al. 2022, arXiv:2203.02395]. StyleTTS 2 and Kokoro use iSTFTNet-style decoders.

### Vocos (2023)

**Intuition.** Don't upsample to the sample rate at all. Keep the network at the mel frame rate
with ConvNeXt blocks and predict full-resolution STFT coefficients; an iSTFT does the
upsampling for free [Siuzdak 2024, arXiv:2306.00814].

**Head.** Project hidden activations to $n_{\text{fft}} + 2$ channels and split into $m$ and $p$
(each $n_{\text{fft}}/2 + 1$). Then

$$
M = \exp(m), \qquad x = \cos(p),\ y = \sin(p), \qquad \operatorname{STFT} = M\,(x + j y),
$$

and $\hat x = \operatorname{iSTFT}(\operatorname{STFT})$. Mapping $p$ through $\cos$ and $\sin$
wraps phase implicitly, so any real $p$ is a valid phase. (The code clips $M$ at $10^2$ for
stability.)

**Losses.** Hinge adversarial loss with MPD and MRD, L1 mel loss, and feature matching. Training:
LibriTTS train-clean and train-other at 24 kHz, 100 mels, $n_{\text{fft}} = 1024$, hop 256, 2M
iterations total (1M each for generator and discriminator), batch 16 of 16,384-sample crops,
AdamW $2 \times 10^{-4}$ with cosine decay.

**Speed and size.** 13.5M parameters. In the paper's table: Vocos 6,696 times real time on GPU and
169.6 on CPU, versus HiFi-GAN 495.5 and 5.84, BigVGAN (base, 14M) 98.6 and 0.40, iSTFTNet 1,046
and 14.4. That's about 13 times faster than HiFi-GAN and 70 times faster than BigVGAN.

**Checkpoints.** `charactr/vocos-mel-24khz` (MIT, about 54 MB) and `charactr/vocos-encodec-24khz`
(reconstructs audio from EnCodec tokens, used by Bark-style pipelines). Package `vocos`. F5-TTS
uses the mel variant as its default vocoder.

**Gotchas.** The iSTFT must use the same window and `center` convention as the analysis STFT,
or the output is misaligned by half a window. Predicting $\log M$ unbounded can overflow; clip.

### Vocoder comparison

The following table summarizes vocoders a lab is likely to load or train.

| Vocoder | Params | Speed (paper, relative) | Open checkpoint | License |
|---|---|---|---|---|
| WaveNet (MoL) | 24.7M | 0.003x real time (V100) | NVIDIA/torchaudio examples | Various |
| WaveGlow | 87.7M | 22.8x (V100) | NVIDIA NGC | BSD-3 |
| MelGAN | 4.3M | 646x (V100) | `descriptinc/melgan-neurips` | MIT |
| Parallel WaveGAN | 1.44M | 28.7x (single GPU) | `kan-bayashi/ParallelWaveGAN` | MIT |
| HiFi-GAN V1 | 13.9M | 168x (V100) | `jik876/hifi-gan`, SpeechBrain | MIT, Apache-2.0 |
| BigVGAN-v2 | 112M | slower than HiFi-GAN; CUDA kernel helps | `nvidia/bigvgan_v2_*` | MIT |
| Vocos | 13.5M | 6,697x (GPU), 170x (CPU) | `charactr/vocos-mel-24khz` | MIT |

### Lab ideas for vocoders

- **Mu-law and WaveNet receptive field (CPU-60s).** Implement mu-law encode, quantize, and
  decode; check that round-trip SNR on a speech fixture exceeds a threshold calibrated from the
  reference implementation, and that it beats 8-bit linear quantization on the same quiet clip.
  Compute receptive fields for given
  dilation schedules and check against the formula and against an empirical impulse test on a
  causal conv stack.
- **Causal dilated conv stack (CPU-60s).** Build a 10-layer dilated causal conv stack in PyTorch.
  The checker perturbs input sample $t$ and verifies that outputs before $t$ don't change and
  that outputs up to $t + R - 1$ can change.
- **Multi-resolution STFT loss (CPU-60s).** Implement $L_{\text{sc}} + L_{\text{mag}}$ over three
  resolutions. Check against a reference value, check that the loss is 0 for identical inputs,
  and check that $L_{\text{mag}}$ is unchanged when both signals are scaled by the same gain
  while $L_{\text{sc}}$ is also unchanged (both are scale invariant) but the time-domain L1 isn't.
- **MPD reshape and Snake (CPU-60s).** Implement the period reshape with reflect padding (check
  shapes for $T = 8{,}192$ and $p \in \{2,3,5,7,11\}$) and the Snake activation and its gradient.
- **Tiny HiFi-GAN (GPU-A100, 5-10 minutes, under 10 GB).** Train a HiFi-GAN V2-sized generator
  (0.92M parameters) with MPD and mel loss on 100 LJSpeech clips at 22,050 Hz for about 5
  minutes. The checker computes mel L1 on held-out clips and requires it to drop below a
  threshold calibrated from a reference run (early HiFi-GAN training typically reaches mel L1
  around 0.5-0.6 within a few thousand steps; calibrate before shipping). On an L4 budget 10-15
  minutes.
- **Vocoder bake-off (GPU-L4, under 3 minutes, under 3 GB).** Vocode ground-truth mels with
  Griffin-Lim, HiFi-GAN, Vocos, and BigVGAN-v2; compute mel L1, multi-resolution STFT loss, and
  UTMOS; report real-time factor. The checker verifies each metric is within tolerance of a
  reference and that Griffin-Lim scores worst on UTMOS.

---

## Vector quantization fundamentals

Neural codecs and codec language models depend on vector quantization (VQ). Teach this section
before codecs.

### VQ-VAE and the straight-through estimator

**Intuition.** Replace each encoder vector with its nearest codebook entry. The argmin has no
gradient, so copy the decoder's gradient straight through to the encoder, and add losses that
pull codebook entries and encoder outputs toward each other [van den Oord, Vinyals, and
Kavukcuoglu 2017, arXiv:1711.00937].

$$
k^* = \arg\min_k \lVert z_e - e_k \rVert_2, \qquad z_q = e_{k^*}, \qquad
z_q^{\text{ST}} = z_e + \operatorname{sg}[z_q - z_e],
$$

$$
\mathcal{L} = -\log p(x \mid z_q) + \lVert \operatorname{sg}[z_e] - e \rVert_2^2 + \beta\,\lVert z_e - \operatorname{sg}[e] \rVert_2^2, \qquad \beta = 0.25.
$$

The second term is the *codebook loss* (moves codes toward encoder outputs); the third is the
*commitment loss* (keeps the encoder near its chosen code). The paper found results robust for
$\beta \in [0.1, 2.0]$.

### EMA codebook updates

Instead of the codebook loss, update codes with exponential moving averages of assignment counts
$N_k$ and assigned-vector sums $m_k$ (VQ-VAE-2 style), decay $\gamma = 0.99$:

$$
N_k \leftarrow \gamma N_k + (1-\gamma)\,n_k, \qquad
m_k \leftarrow \gamma m_k + (1-\gamma)\sum_{i: k_i^* = k} z_{e,i}, \qquad
e_k = \frac{m_k}{N_k}.
$$

Here $n_k$ is the number of batch vectors assigned to code $k$. In practice, apply Laplace
smoothing to $N_k$ before dividing, so unused codes don't divide by zero:
$N_k \leftarrow \frac{N_k + \epsilon}{\sum_j N_j + K\epsilon}\sum_j N_j$.

### Codebook collapse and its fixes

*Codebook collapse* means only a few codes are ever selected, wasting bitrate. Fixes:

- **k-means initialization** on the first batch (SoundStream).
- **Dead-code restart**: replace any code whose EMA usage falls below a threshold with a random
  encoder output from the current batch. SoundStream tracks EMA assignment counts with decay 0.99
  and replaces codes whose statistic falls below 2.
- **Factorized, low-dimensional lookup** (DAC, following ViT-VQGAN): project $z_e$ to 8 or 32
  dimensions before lookup, then project back.
- **L2-normalized codes**: normalize both $z_e$ and codes, which turns Euclidean lookup into cosine
  similarity.
- **Replace VQ with FSQ** (see later in this section).

Monitor codebook usage with perplexity $\exp(-\sum_k \bar p_k \log \bar p_k)$ where $\bar p_k$ is
the batch frequency of code $k$. Perplexity equal to $K$ means uniform usage.

### Residual vector quantization (RVQ)

**Intuition.** One codebook of size $2^{80}$ is impossible; eight codebooks of size $2^{10}$
applied in sequence to successive residuals give the same bitrate with manageable tables. The
first quantizer captures the coarse structure; each later one refines the leftover error.

**Algorithm** [Zeghidour et al. 2021, SoundStream, Algorithm 1]:

```text
y_hat <- 0; residual <- y
for i = 1 .. N_q:
    q_i <- Q_i(residual)          # nearest code in codebook i
    y_hat <- y_hat + q_i
    residual <- residual - q_i
return y_hat, codes
```

Commitment loss as used in EnCodec, summed over stages with residuals $z_c$:
$\ell_w = \sum_{c=1}^{C} \lVert z_c - q_c(z_c) \rVert_2^2$, with the gradient flowing only into
$z_c$ (the quantized value is stop-gradiented). The straight-through estimator applies to the
total: $\hat y^{\text{ST}} = y + \operatorname{sg}[\hat y - y]$.

**Quantizer dropout.** For each training example, sample $n_q \sim \mathcal{U}\{1, \ldots, N_q\}$
and use only the first $n_q$ quantizers, so one model supports all bitrates. DAC applies quantizer
dropout with probability $p = 0.5$ per example (otherwise all quantizers are used), because full
dropout degrades quality at the maximum bitrate. EnCodec samples the number of codebooks as a
multiple of 4.

**Bitrate formula.**

$$
\text{bitrate (bits/s)} = \frac{f_s}{M} \times N_q \times \log_2 K,
$$

where $M$ is the total encoder stride. The per-quantizer budget is $r / N_q = \log_2 K$; for
SoundStream at 6 kbps and 75 Hz, $r = 6000/75 = 80$ bits per frame, and $N_q = 8$ gives
$K = 2^{80/8} = 1024$.

**Hand-worked RVQ example (2-D, two codebooks).** Input $x = (0.9, -0.3)$.
Codebook 1: $\{(1, 0), (0, 1), (-1, 0), (0, -1)\}$. Squared distances
$[0.10, 2.50, 3.70, 1.30]$, so pick index 0, $q_1 = (1, 0)$, residual $r_1 = (-0.1, -0.3)$.
Codebook 2: $\{(0.1, 0.1), (-0.1, -0.2), (0.2, -0.2), (-0.2, 0.2)\}$. Squared distances from
$r_1$: $[0.20, 0.01, 0.10, 0.26]$, so pick index 1, $q_2 = (-0.1, -0.2)$, residual
$r_2 = (0, -0.1)$. Reconstruction $\hat x = (0.9, -0.2)$. Residual norms go
$\lVert x\rVert = 0.949 \to \lVert r_1 \rVert = 0.316 \to \lVert r_2 \rVert = 0.100$. Codes are
$(0, 1)$, which cost $2 \times \log_2 4 = 4$ bits. Commitment loss
$\ell_w = \lVert r_1 \rVert^2 + \lVert r_2 \rVert^2 = 0.10 + 0.01 = 0.11$.

### Finite scalar quantization (FSQ)

**Intuition.** Drop the learned codebook. Project to a few dimensions, bound each dimension to $L_i$
levels with tanh, and round. The implicit codebook is the grid, of size $\prod_i L_i$, and it's
almost fully used without auxiliary losses [Mentzer et al. 2024, arXiv:2309.15505].

$$
\hat z = \operatorname{round\_ste}\left(\lfloor L/2 \rfloor \tanh(z)\right), \qquad
\operatorname{round\_ste}(x) = x + \operatorname{sg}[\operatorname{round}(x) - x].
$$

For even $L$ the reference code adds an offset so the grid is symmetric
(`bound(z) = tanh(z + shift) * half_l - offset` with `half_l = (L - 1)(1 - eps)/2` and
`offset = 0.5` for even $L$). Index: $\text{idx} = \sum_i (\hat z_i + \lfloor L_i/2 \rfloor)\prod_{j<i} L_j$.

Recommended levels: $|C| = 2^8$: $[8, 6, 5]$; $2^{10}$: $[8, 5, 5, 5]$; $2^{12}$: $[7, 5, 5, 5, 5]$;
$2^{14}$: $[8, 8, 8, 6, 5]$; $2^{16}$: $[8, 8, 8, 5, 5, 5]$. Heuristic: use $L_i \ge 5$.

**Hand-worked FSQ example.** $L = [5, 5, 5]$, $z = (0.3, -2.0, 1.2)$. Bounded
$2\tanh(z) = (0.583, -1.928, 1.667)$, rounded $(1, -2, 2)$. Shifted to $\{0..4\}$: $(3, 0, 4)$,
index $3 + 0 \cdot 5 + 4 \cdot 25 = 103$ out of 125.

**Where FSQ appears.** CosyVoice 2 and 3 tokenizers ($K = 1$, $D = 8$, so $3^8 = 6{,}561$ codes),
XCodec2 (65,536 codes, $4^8$), Stable Codec (residual FSQ), Fish-Speech (grouped FSQ), Spark-TTS
global tokens.

### Gotchas for VQ

- Distances computed as `((z[:, None] - C[None]) ** 2).sum(-1)` blow up memory; use
  $\lVert z\rVert^2 - 2 z C^\top + \lVert C\rVert^2$.
- Applying straight-through per stage instead of once at the end still works but changes
  gradients; be consistent with the reference.
- Commitment loss on the wrong side of the stop-gradient trains the codebook toward a moving
  target and collapses.
- Quantizer dropout must be off at evaluation; evaluate each $n_q$ separately.

### Lab ideas for VQ

- **RVQ encode, decode, and losses (CPU-60s).** Implement RVQ with $N_q$ codebooks, return codes,
  reconstruction, commitment loss, and per-stage residual norms. The checker verifies residual
  norms are non-increasing for k-means-initialized codebooks, that decode(encode(x)) matches the
  returned reconstruction, and that the hand-worked example reproduces exactly.
- **Codebook collapse and restart (CPU-60s).** Train a 64-code VQ on 2-D Gaussian-mixture data
  with random init (usage collapses), then with k-means init plus dead-code restart. The checker
  requires perplexity above a threshold with the fixes.
- **FSQ (CPU-60s).** Implement FSQ bound, quantize, index, and inverse index; the checker verifies
  a bijection over the full implicit codebook and 100% usage on uniform inputs.
- **Bitrate calculator (CPU-60s).** Given codec configs, compute bitrate and tokens per second;
  checked against the table in the next section.

---

## Neural audio codecs

A neural codec is an autoencoder with a quantized bottleneck: encoder (strided convolutions,
sometimes Transformers), quantizer (RVQ, FSQ, or single VQ), decoder (mirror image or Vocos-style
iSTFT head), trained with reconstruction, adversarial, and feature-matching losses. For TTS the
codec matters twice: its tokens are the language model's vocabulary, and its decoder is the
vocoder. Lower frame rates and fewer codebooks make language modeling easier; more bits make
reconstruction better.

### SoundStream (2021)

[Zeghidour et al. 2021, arXiv:2107.03312]

- **Encoder.** 1-D conv with $C$ channels, then $B_{\text{enc}} = 4$ blocks, each with three
  residual units (dilations 1, 3, 9) and a strided conv (kernel $2S$, stride $S$), strides
  $(2, 4, 5, 8)$, so 320x downsampling: 24 kHz becomes 75 frames per second. Channels double at
  each downsampling. Fully causal for streaming.
- **Decoder.** Mirrors the encoder with transposed convs in reverse stride order.
- **Quantizer.** RVQ with EMA codebooks, k-means init on the first batch, dead-code replacement,
  quantizer dropout for 3-18 kbps from one model.
- **Discriminators.** A multi-resolution waveform discriminator (MelGAN-style) plus an STFT
  discriminator on the complex STFT (window 1024, hop 256).
- **Losses.** Hinge adversarial, feature matching normalized by layer length, and a multi-scale
  mel reconstruction loss:

$$
\mathcal{L}^{\text{rec}} = \sum_{s \in 2^6, \ldots, 2^{11}} \sum_t \lVert S_t^s(x) - S_t^s(G(x)) \rVert_1 + \alpha_s \sum_t \lVert \log S_t^s(x) - \log S_t^s(G(x)) \rVert_2,
\quad \alpha_s = \sqrt{s/2},
$$

  where $S_t^s$ is a 64-bin mel frame with window $s$ and hop $s/4$.
  $\mathcal{L}_G = \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}} + \lambda_{\text{rec}}\mathcal{L}_{\text{rec}}$
  with $\lambda_{\text{adv}} = 1$, $\lambda_{\text{feat}} = 100$, $\lambda_{\text{rec}} = 1$.
- **Joint enhancement.** A FiLM-conditioned flag switches denoising on or off.
- One reported configuration has 8.4M parameters. SoundStream at 3 kbps beat Opus at
  12 kbps in listening tests. Weights for the full model weren't released (Lyra v2 is a
  SoundStream-based codec with released inference binaries).

### EnCodec (2022)

[Défossez et al. 2022, arXiv:2210.13438]

- **Architecture.** SoundStream-like conv encoder ($C = 32$, $B = 4$, strides $(2, 4, 5, 8)$,
  ELU), followed by a two-layer LSTM and a final conv to $D = 128$. Decoder mirrors it. 24 kHz mono
  (causal, streamable) and 48 kHz stereo (non-causal) models; 75 and 150 frames per second.
- **RVQ.** Up to 32 codebooks (16 at 48 kHz) of 1024 entries (10 bits). EMA updates with decay
  0.99, dead-code replacement, straight-through estimator, commitment loss. Variable bandwidth
  training picks the number of codebooks as a multiple of 4: 1.5, 3, 6, 12, 24 kbps at 24 kHz.
- **Losses.** Time-domain L1 $\ell_t = \lVert x - \hat x \rVert_1$; frequency-domain
  $\ell_f = \frac{1}{|\alpha||s|}\sum_{i \in e}\lVert S_i(x) - S_i(\hat x)\rVert_1 + \alpha_i \lVert S_i(x) - S_i(\hat x)\rVert_2$
  over 64-bin mels with windows $2^i$, $i = 5..11$, and $\alpha_i = 1$; hinge adversarial
  $\ell_g = \frac1K\sum_k \max(0, 1 - D_k(\hat x))$ from a multi-scale STFT discriminator (MS-STFTD,
  windows 2048, 1024, 512, 256, 128, complex STFT input); relative feature matching
  $\ell_{\text{feat}} = \frac{1}{KL}\sum_{k,l} \frac{\lVert D_k^l(x) - D_k^l(\hat x)\rVert_1}{\operatorname{mean}(\lVert D_k^l(x)\rVert_1)}$;
  and commitment $\ell_w$. Weights at 24 kHz: $\lambda_t = 0.1$, $\lambda_f = 1$, $\lambda_g = 3$,
  $\lambda_{\text{feat}} = 3$. The discriminator updates with probability 2/3 at 24 kHz.
- **Loss balancer.** With $g_i = \partial \ell_i / \partial \hat x$ and
  $\langle \lVert g_i \rVert_2 \rangle_\beta$ its EMA over batches ($\beta = 0.999$), backpropagate

$$
\tilde g_i = R\,\frac{\lambda_i}{\sum_j \lambda_j}\,\frac{g_i}{\langle \lVert g_i \rVert_2 \rangle_\beta}, \qquad R = 1,
$$

  instead of $\sum_i \lambda_i g_i$. Each $\lambda_i / \sum_j \lambda_j$ becomes the fraction of
  the gradient on $\hat x$ that comes from loss $i$, independent of its scale. The commitment loss
  is outside the balancer.
- **Entropy coding.** A small causal Transformer (5 layers, 8 heads, 200 channels, FFN 800) predicts
  the distribution of each codebook's next code from the summed embeddings of the previous step; an
  arithmetic coder then saves up to 40% of bandwidth while staying real-time on one CPU core.
- **Training.** 300 epochs of 2,000 updates, batch 64 of 1-second clips, Adam $3 \times 10^{-4}$,
  $\beta = (0.5, 0.9)$, 8 A100 GPUs.
- **Checkpoints.** `pip install encodec`; Hugging Face `facebook/encodec_24khz` (23.3M parameters)
  and `facebook/encodec_48khz`. MIT license (code and weights). MusicGen uses a separate 32 kHz
  EnCodec with 4 codebooks of 2048 at 50 Hz.

**Hand-worked bitrate.** 24 kHz, stride 320: $24{,}000 / 320 = 75$ Hz. At 6 kbps:
$75 \times 8 \times 10 = 6{,}000$ bits/s, 8 codebooks, 600 tokens per second. At 1.5 kbps: 2
codebooks, 150 tokens per second. At 24 kbps: 32 codebooks, 2,400 tokens per second. Compression
versus 16-bit PCM at 24 kHz (384 kbps): 64x at 6 kbps.

### DAC: Descript Audio Codec, improved RVQGAN (2023)

[Kumar et al. 2023, arXiv:2306.06546]

- **Architecture.** Conv encoder with Snake activations, strides $[2, 4, 8, 8]$ (512x) at
  44.1 kHz, so 86.13 frames per second; decoder rates $[8, 8, 4, 2]$, decoder dimension 1536.
  76M parameters total (22M encoder, 54M decoder).
- **Improved RVQ.** Factorized codes (lookup in 8 dimensions, embeddings in 1024) and L2-normalized
  codes, with the plain VQ-VAE codebook loss (weight 1.0) and commitment loss (weight 0.25),
  without k-means init or restarts. This fixes codebook under-use that EMA alone leaves.
- **Quantizer dropout** with probability 0.5.
- **Discriminators and losses.** MPD (periods 2, 3, 5, 7, 11) plus a complex multi-scale,
  multi-band STFT discriminator (windows 2048, 1024, 512; band limits 0, 0.1, 0.25, 0.5, 0.75, 1.0
  of Nyquist). Multi-scale log-mel L1 with windows 32 to 2048 and 5 to 320 mels. Weights: mel 15.0,
  feature matching 2.0, adversarial 1.0, codebook 1.0, commitment 0.25.
- **Training.** Final model: batch 72 of 0.38-second excerpts for 400k iterations; ablations
  (batch 12, 250k iterations) took about 30 hours on one GPU. AdamW $10^{-4}$, $\beta = (0.8, 0.9)$.
- **Checkpoints.** `pip install descript-audio-codec`; Hugging Face `descript/dac_44khz` (9 codebooks,
  76.6M), `descript/dac_24khz` (32 codebooks, strides $[2, 4, 5, 8]$), `descript/dac_16khz`
  (12 codebooks, strides $[2, 4, 5, 8]$). MIT license. Dia uses DAC 44 kHz tokens.

**Hand-worked bitrate.** $44{,}100 / 512 = 86.13$ Hz; $86.13 \times 9 \times 10 = 7{,}752$ bits/s,
about 8 kbps, which is about 91x compression of 16-bit PCM (705.6 kbps). For the 24 kHz model:
$75 \times 32 \times 10 = 24$ kbps at full depth; for 16 kHz: $50 \times 12 \times 10 = 6$ kbps
(computed from the Hugging Face configs; see the uncertain-facts section).

### Mimi (Kyutai, 2024)

[Défossez et al. 2024, Moshi, arXiv:2410.00037]

- **Architecture.** SEANet-style causal conv encoder with strides $(4, 5, 6, 8)$ plus a final conv
  with stride 2 (1920x), so 24 kHz becomes **12.5 frames per second**, latent dimension 512.
  Causal Transformers (8 layers, 8 heads, RoPE, 250-frame context, LayerScale init 0.01) before and
  after quantization. Streaming with 80 ms frames.
- **Split RVQ with semantic distillation.** WavLM embeddings (16 kHz, 50 Hz, 1024-d) are average
  pooled non-causally (stride 4, kernel 8) to 12.5 Hz, and a cosine distance distills them into the
  output of a *separate* plain VQ (the semantic token) through a linear projection. A 7-level RVQ
  runs in parallel on the same latent for acoustics, and the outputs are summed. Splitting avoids
  forcing acoustic detail into the residual of the semantic quantizer.
- **Quantization.** $Q = 8$ codebooks of 2048 entries in the Moshi setup (the released model has 32
  levels; Sesame CSM and Kyutai TTS use all 32). Embeddings are projected to 256 dimensions for RVQ.
  Quantization is applied only 50% of the time during training (otherwise the unquantized latent
  goes to the decoder).
- **Training.** AdamW $8 \times 10^{-4}$, batch 128 of 12-second windows, 4M steps, weight EMA
  0.99. Adversarial-only training (feature and discriminator losses, no reconstruction loss)
  sounded better in listening tests despite worse objective metrics.
- **Checkpoint.** `kyutai/mimi` in Transformers (`MimiModel`), CC-BY-4.0, 96.2M parameters.

**Hand-worked bitrate.** $12.5 \times 8 \times \log_2 2048 = 12.5 \times 8 \times 11 = 1{,}100$ bits/s
(1.1 kbps), 100 tokens per second. With all 32 codebooks: 4.4 kbps, 400 tokens per second.

### SNAC (2024)

Multi-scale neural audio codec: RVQ levels run at *different temporal resolutions*. The 24 kHz
speech model uses encoder hop 512 and VQ strides $[4, 2, 1]$, giving token rates of about 12, 23,
and 47 Hz with 4096-entry codebooks (12 bits), 19.8M parameters, and 0.98 kbps. The 32 kHz and
44 kHz general-audio models use four levels and reach 1.9 and 2.6 kbps [Siuzdak, Grötschla, and
Lanzendörfer 2024, arXiv:2410.14411]. `pip install snac`, `hubertsiuzdak/snac_24khz` (MIT, about
79 MB). Orpheus uses it.

**Hand-worked bitrate.** $24{,}000 / 512 = 46.875$ Hz fine level, 23.44 Hz middle, 11.72 Hz
coarse. Tokens per second $= 82.03$; bitrate $82.03 \times 12 = 984$ bits/s. Per coarse frame there
are $1 + 2 + 4 = 7$ tokens, which is why Orpheus emits 7 tokens per 85 ms frame.

### SpeechTokenizer (2023)

EnCodec-style RVQ codec (16 kHz, 8 levels, 1024 entries, 50 Hz) whose first RVQ level is distilled
toward HuBERT (layer 9 or the average of all layers) with a "D-axis" loss that maximizes cosine
similarity per feature dimension across time:
$\mathcal{L}_{\text{distill}} = -\frac{1}{D}\sum_{d=1}^{D}\log\sigma\left(\cos(A Q_1^{(:,d)}, S^{(:,d)})\right)$.
RVQ-1 then carries content and RVQ-2:8 carry timbre and detail, which is exactly what a VALL-E
style AR-plus-NAR model wants [Zhang et al. 2024, arXiv:2308.16692]. Hugging Face
`fnlp/SpeechTokenizer`.

### WavTokenizer (2024)

Single-codebook codec: one VQ with 4096 codes, 40 or 75 tokens per second at 24 kHz (strides
$(4,5,5,6)$ or $(2,4,5,8)$), with an attention-augmented Vocos-style iSTFT decoder. The authors
found speech codes concentrate below $2^{12}$ even when the codebook is expanded to $2^{14}$
[Ji et al. 2025, arXiv:2408.16532]. Bitrate at 75 Hz: $75 \times 12 = 900$ bits/s; at 40 Hz:
480 bits/s. MIT license, `novateur/WavTokenizer-*` checkpoints.

### X-Codec and XCodec2 (2024-2025)

- **X-Codec** feeds features from a frozen semantic model (HuBERT or w2v-BERT) alongside acoustic
  features into one RVQ and adds a semantic reconstruction loss after quantization, which lowers
  WER for codec LMs [Ye et al. 2025, arXiv:2408.17175].
- **XCodec2** (from the Llasa paper): a w2v-BERT 2.0 semantic encoder and a Snake-based acoustic
  encoder are concatenated and quantized with **single-codebook FSQ of size 65,536** at 16 kHz with
  total stride 320 (50 tokens per second, 800 bits/s). The decoder is a Transformer with a
  Vocos-style iSTFT head; a semantic decoder reconstructs semantic features as an auxiliary loss
  [Ye et al. 2025, arXiv:2502.04128]. `HKUSTAudio/xcodec2`, CC-BY-NC-4.0, 822.7M parameters
  including the w2v-BERT encoder. Package `xcodec2`.

### BigCodec (2024)

Scales a single-codebook VQ-VAE codec to 159M parameters: 16 kHz, 8192 codes, 80 Hz, **1.04 kbps**
($80 \times 13$), with plain VQ-VAE quantization and low-dimensional projection. It beat
multi-codebook codecs at similar bitrate [Xin et al. 2024, arXiv:2409.05377].

### Stable Codec (TAAE, 2024)

A Transformer audio autoencoder (about 950M parameters) with an FSQ bottleneck (6 dimensions,
25 Hz) and a post-hoc residual decomposition of FSQ levels, reaching 400 and 700 bits/s for 16 kHz
speech [Parker et al. 2025, arXiv:2411.19842]. Bitrate formula for residual FSQ:
$f_r \sum_i \lceil \log_2 k_i \rceil$ with $k_i = L^d$. The paper's example: $L = 17$, $d = 6$,
25 Hz gives $25 \times \lceil \log_2 17^6 \rceil = 25 \times 25 = 625$ bps; two residual stages with
5 levels give $25 \times 2 \times \lceil \log_2 5^6 \rceil = 25 \times 2 \times 14 = 700$ bps.
`stabilityai/stable-codec-speech-16k`, Stability AI Community License, gated.

### Codec comparison

The following table gives the numbers the bitrate lab checks against.

| Codec | $f_s$ | Frame rate | Codebooks $\times$ size | Max bitrate | Tokens/s | Params | License |
|---|---|---|---|---|---|---|---|
| SoundStream | 24 kHz | 75 Hz | $\le 24 \times 1024$ (6 kbps: 8) | 18 kbps (paper range) | 600 at 6 kbps | not released | n/a |
| EnCodec 24k | 24 kHz | 75 Hz | $\le 32 \times 1024$ | 24 kbps | 150-2,400 | 23.3M | MIT |
| DAC 44k | 44.1 kHz | 86.1 Hz | $9 \times 1024$ | 7.75 kbps | 775 | 76.6M | MIT |
| Mimi | 24 kHz | 12.5 Hz | $8$ (up to 32) $\times 2048$ | 1.1 kbps (4.4) | 100 (400) | 96.2M | CC-BY-4.0 |
| SNAC 24k | 24 kHz | 11.7/23.4/46.9 Hz | $3 \times 4096$ | 0.98 kbps | 82 | 19.8M | MIT |
| SpeechTokenizer | 16 kHz | 50 Hz | $8 \times 1024$ | 4 kbps | 400 | about 120M (482 MB file) | not checked |
| WavTokenizer | 24 kHz | 40 or 75 Hz | $1 \times 4096$ | 0.48 or 0.9 kbps | 40 or 75 | not checked | MIT |
| XCodec2 | 16 kHz | 50 Hz | $1 \times 65{,}536$ (FSQ) | 0.8 kbps | 50 | 822.7M (with w2v-BERT) | CC-BY-NC-4.0 |
| BigCodec | 16 kHz | 80 Hz | $1 \times 8192$ | 1.04 kbps | 80 | 159M | not checked |
| BiCodec (Spark-TTS) | 16 kHz | 50 Hz + 32 global | $1 \times 8192$ | 0.65 kbps + global | 50 | about 156M (file 626 MB) | CC-BY-NC-SA-4.0 |
| CosyVoice 2 tokenizer | 16 kHz in | 25 Hz | $1 \times 6{,}561$ (FSQ) | 0.32 kbps | 25 | ASR-encoder based | Apache-2.0 |
| Qwen3-TTS tokenizer | 24 kHz | 12.5 Hz | $16 \times 2048$ | 2.2 kbps | 200 | not checked | Apache-2.0 |
| Higgs Audio v2 tokenizer | 24 kHz | 25 Hz | $8 \times$ (1024 inferred) | 2 kbps | 200 | not checked | Custom |
| VibeVoice acoustic | 24 kHz | 7.5 Hz | continuous VAE latent | n/a | 7.5 frames | about 340M enc + 340M dec | MIT |

"About" entries for SpeechTokenizer and BiCodec are estimated from fp32 checkpoint file sizes
(4 bytes per parameter), so treat them as approximate.

### Codec metrics

- **Bitrate** from the formula; **tokens per second** $= $ frame rate $\times$ codebooks.
- **Mel distance** (DAC reports multi-scale log-mel L1), **STFT distance**, **SI-SDR**.
- **ViSQOL** (Virtual Speech Quality Objective Listener, v3): full-reference, outputs MOS-LQO on
  1-5; speech mode at 16 kHz, audio mode at 48 kHz [Chinen et al. 2020, arXiv:2004.09584].
- **PESQ** and **STOI** for speech (full reference, narrowband and wideband PESQ need 8 kHz or
  16 kHz input).
- **UTMOS** on reconstructions (non-intrusive, speech only), plus WER and speaker similarity of
  reconstructions for TTS-relevant codecs.
- **Codebook utilization**: perplexity or entropy per codebook, as in the DAC paper's Figure 1.

### Lab ideas for codecs

- **Bitrate versus quality sweep (GPU-L4, about 5 minutes, under 4 GB; also feasible on CPU in
  about 2 minutes for 10 short clips).** Load EnCodec 24 kHz, DAC 44 kHz, Mimi, and SNAC. For 10
  LibriTTS clips, encode and decode at every supported bandwidth (EnCodec 1.5-24 kbps; DAC and Mimi
  by truncating codebooks), compute bitrate from token shapes, multi-scale mel distance, and
  optionally ViSQOL or UTMOS. The checker verifies the computed bitrates exactly, verifies the mel
  distance decreases monotonically with the number of codebooks for each codec, and checks values
  are within tolerance of a reference run.
- **Codebook contribution (CPU-60s with cached codes).** Given cached Mimi or EnCodec codes for a
  clip, decode with the first $k$ codebooks for each $k$ and plot mel distance; separately, zero out
  only the semantic codebook of Mimi and show WER rises sharply while mel distance moves little
  (WER from a shipped small ASR or precomputed).
- **Train a tiny RVQ autoencoder (GPU-A100, about 10 minutes, under 8 GB).** A 1-2M parameter
  conv encoder and decoder (strides $[2, 4, 5, 8]$) with 4-level RVQ (EMA, k-means init, restarts)
  and quantizer dropout on 16 kHz LibriTTS clips, mel L1 plus multi-resolution STFT loss, no GAN.
  The checker requires codebook perplexity above a threshold for every level and mel distance to
  drop as more levels are used.

---

## Codec language models

**Intuition.** Once audio is a grid of discrete tokens, TTS becomes conditional language modeling:
given text (and a short audio prompt of the target speaker), predict the codec tokens, then decode
them with the codec. In-context learning from the prompt gives zero-shot voice cloning. The design
questions are (1) which tokens (semantic, acoustic, both), (2) how to handle multiple codebooks per
frame, and (3) autoregressive versus masked or diffusion generation.

### Token types

- **Semantic tokens**: k-means clusters or VQ codes of self-supervised features (HuBERT, w2v-BERT,
  WavLM) or of ASR-encoder features (CosyVoice's "supervised semantic tokens"). They capture content
  and some prosody, discard most speaker and channel detail, and are easy to predict from text.
- **Acoustic tokens**: RVQ codes from a codec (SoundStream, EnCodec, DAC, SNAC). They reconstruct
  audio but are harder to predict.
- **Hybrid**: codecs with a semantically distilled first codebook (SpeechTokenizer, Mimi, X-Codec,
  XCodec2, BiCodec).

### Multi-codebook sequence layouts

With $K$ codebooks per frame and $T$ frames, the options are the following (MusicGen formalizes them
as a *pattern* $P = (P_0, \ldots, P_S)$, a partition of all $(t, k)$ positions into steps, where all
positions in $P_s$ are predicted in parallel given $P_{<s}$) [Copet et al. 2023, arXiv:2306.05284]:

- **Flattening**: one token per step, row-major. Exact factorization; $S = K T$ steps. Orpheus
  (SNAC's 7 tokens per frame) and AudioLM's acoustic stages do this.
- **Parallel**: $P_s = \{(s, k) : k = 1..K\}$. $S = T$ steps; ignores dependencies between codebooks in
  the same frame, which hurts quality.
- **Delay**: $P_s = \{(s - k + 1, k) : k = 1..K,\ s - k \ge 0\}$. Codebook $k$ is shifted right by
  $k - 1$ steps, so when codebook $k$ of frame $t$ is predicted, codebooks $< k$ of frame $t$ are
  already known. $S = T + K - 1$ steps. MusicGen, Dia, Parler-TTS, and Higgs Audio v2 use it.
- **VALL-E pattern**: autoregressive over codebook 1, then non-autoregressive passes for codebooks
  2..K.
- **Hierarchical (RQ-Transformer, depth transformer)**: a large *temporal* Transformer produces one
  context vector per frame, and a small *depth* Transformer autoregresses over the $K$ codebooks
  within the frame. Moshi, Sesame CSM, Kyutai TTS, and Fish-Speech's Dual-AR (slow and fast
  Transformers) use it. Cost is $T$ big steps plus $TK$ tiny steps.

**Hand-worked delay example.** $K = 3$, $T = 4$. The steps are
$P_1 = \{(1,1)\}$, $P_2 = \{(2,1),(1,2)\}$, $P_3 = \{(3,1),(2,2),(1,3)\}$,
$P_4 = \{(4,1),(3,2),(2,3)\}$, $P_5 = \{(4,2),(3,3)\}$, $P_6 = \{(4,3)\}$. $S = 6 = T + K - 1$,
versus 12 for flattening and 4 for parallel. Empty positions get a special pad token. As a token grid
(rows are codebooks, columns are steps, `P` is pad):

```text
k=1: a1 a2 a3 a4 P  P
k=2: P  b1 b2 b3 b4 P
k=3: P  P  c1 c2 c3 c4
```

**Gotchas.** Forgetting to undelay before decoding produces garbled audio. Loss must mask pad
positions. Sampling temperature and top-k are usually set per codebook.

### AudioLM (2022)

Three decoder-only Transformer stages on 16 kHz audio [Borsos et al. 2023, arXiv:2209.03143]:

1. **Semantic modeling**: w2v-BERT XL MLM-layer embeddings quantized by k-means with $K = 1024$ at
   25 Hz. Models long-term structure.
2. **Coarse acoustic modeling**: predicts the first $Q'$ SoundStream RVQ levels (flattened row-major)
   conditioned on the semantic tokens.
3. **Fine acoustic modeling**: predicts the remaining levels $q > Q'$ conditioned on the coarse
   tokens, using only short local context.

SoundStream at 50 Hz with $N = 1024$: each quantizer costs $50 \times 10 = 500$ bits/s, so $Q = 4$
gives 2 kbps; the paper's setup uses 12 levels (6 kbps) with $Q' = 4$ coarse. Trained on 60k hours of
Libri-Light. It's speech continuation, not TTS, but it established the semantic-then-acoustic
hierarchy.

### VALL-E (2023)

[Wang et al. 2023, arXiv:2301.02111]

- **Tokens.** EnCodec 24 kHz at 6 kbps: 75 Hz, 8 codebooks of 1024. Code matrix
  $C \in \{1..1024\}^{T \times 8}$.
- **Factorization.**

$$
p(C \mid x, \tilde C) = \underbrace{\prod_{t} p(c_{t,1} \mid c_{<t,1}, \tilde c_{:,1}, x; \theta_{\text{AR}})}_{\text{AR over codebook 1}}
\ \prod_{j=2}^{8} \underbrace{p(c_{:,j} \mid c_{:,<j}, x, \tilde C; \theta_{\text{NAR}})}_{\text{NAR, one pass per codebook}},
$$

  where $x$ is the phoneme sequence and $\tilde C$ the prompt's codes. The NAR model sums the
  embeddings of codebooks $1..j-1$, gets a stage embedding for $j$, and shares input and output
  embeddings per stage.
- **Model.** AR and NAR are each 12 layers, 16 heads, $d = 1024$, FFN 4096, dropout 0.1. Trained on
  60k hours of Libri-Light with transcripts from a hybrid ASR, 16 V100 GPUs, 800k steps.
- **Zero-shot.** A 3-second enrolled recording as acoustic prompt. Sampling (not beam search) in the AR
  stage avoids infinite loops; greedy in the NAR stages.
- **Metrics.** Speaker similarity with WavLM-TDNN, WER with HuBERT-Large CTC. It beat YourTTS by
  +0.12 CMOS and +0.93 SMOS.

### VALL-E X and VALL-E 2

- **VALL-E X** (2023): multilingual, with language ID tokens, for cross-lingual zero-shot TTS and
  speech-to-speech translation that keeps the speaker's voice [Zhang et al. 2023, arXiv:2303.03926].
- **VALL-E 2** (2024): *repetition-aware sampling* (if a sampled token's repetition ratio within a
  window $K$ of recent tokens exceeds threshold $t_r$, resample from the full distribution instead of
  nucleus sampling) and *grouped code modeling* (predict groups of codes per step to shorten
  sequences). It claimed human parity on LibriSpeech and VCTK [Chen et al. 2024, arXiv:2406.05370].

### SPEAR-TTS (2023)

Splits TTS into "reading" (text to w2v-BERT semantic tokens, a sequence-to-sequence problem) and
"speaking" (semantic to SoundStream acoustic tokens, trained on audio-only data). Back-translation and
pretraining let the reading model work with as little as 15 minutes of parallel data from one speaker;
voice comes from a 3-second prompt in the speaking stage [Kharitonov et al. 2023, arXiv:2302.03540].

### SoundStorm (2023)

Replaces AudioLM's acoustic stages with a MaskGIT-style parallel decoder over RVQ tokens
[Borsos et al. 2023, arXiv:2305.09636]:

- **Training mask.** Sample a prompt boundary $t \sim \mathcal{U}\{0, T-1\}$ and an RVQ level
  $q \sim \mathcal{U}\{1, Q\}$; sample a mask for level $q$ with ratio $p = \cos(u)$,
  $u \sim \mathcal{U}[0, \pi/2]$; mask selected non-prompt tokens at level $q$ and all non-prompt
  tokens at levels $> q$. Cross-entropy only on masked level-$q$ tokens.
- **Decoding.** Level by level, coarse to fine. Within a level, MaskGIT confidence-based iterative
  sampling with a cosine schedule, greedy on the last iteration. With (16, 1, 1, ..., 1) iterations per
  level for $Q = 12$, 30 seconds of audio (18,000 tokens) takes 27 forward passes; 0.5 seconds on a
  TPU-v4.
- **Model.** 350M-parameter Conformer (12 layers, 16 heads, $d = 1024$), conditioned on AudioLM
  semantic tokens (25 Hz, duplicated to 50 Hz). SoundStream: 50 Hz, $Q = 12$, 1024 codes, 6 kbps.

### MaskGCT (2024)

Fully non-autoregressive, two masked generative stages, no alignment or duration model
[Wang et al. 2025, arXiv:2409.00750]:

- **Text-to-semantic (T2S, 695M)**: predicts masked semantic tokens with in-context learning. Semantic
  tokens come from a VQ-VAE on layer-17 w2v-BERT 2.0 features (8,192 codes of dimension 8). Total
  length is set up front (from a simple rule or user input), which gives duration control.
- **Semantic-to-acoustic (S2A, 353M)**: predicts 12 RVQ levels (1024 codes each) of an acoustic codec
  with a Vocos-style decoder.
- Trained on 100k hours of Emilia. `amphion/MaskGCT`, CC-BY-NC-4.0. IndexTTS2 reuses its semantic
  codec.

### Bark (2023)

Suno's three GPT-style 80M-parameter stages: text to 10,000 semantic tokens (causal), semantic to the
first 2 EnCodec codebooks (causal), coarse to the remaining 6 codebooks (non-causal). EnCodec 24 kHz
at 6 kbps. It generates nonverbal sounds and music-like audio from text prompts. MIT license
(relicensed from CC-BY-NC in 2023). Good as a historical "GPT on codec tokens" example.

### Tortoise (2022) and XTTS (2023-2024)

- **Tortoise** [Betker 2023, arXiv:2305.07243]: an autoregressive GPT-2-style model predicts discrete
  VQ-VAE codes of mels (8,192 codes) conditioned on text and reference clips; many candidates are
  sampled and reranked by CLVP (a CLIP-like text-speech contrastive model); a diffusion decoder turns
  the chosen latent into a mel; UnivNet vocodes. Very slow but influential (the "DALL-E for speech"
  recipe). Apache-2.0.
- **XTTS** [Casanova et al. 2024, arXiv:2406.04904]: Tortoise-derived and streamlined. A 13M VQ-VAE
  encodes mels to 8,192 codes at 21.53 Hz, filtered to the 1,024 most frequent. A 443M GPT-2 encoder
  predicts codes from text plus 32 conditioning latents (1024-d) from a Perceiver Resampler over the
  reference mel. A 26M HiFi-GAN decoder generates audio directly from GPT latents (not codes), with a
  speaker embedding. 16 languages in the paper; XTTS-v2 supports 17. `coqui/XTTS-v2` under the Coqui
  Public Model License (non-commercial). Coqui the company shut down in early 2024; the maintained fork
  of the library is `coqui-tts` (Idiap).

### CosyVoice 1, 2, and 3

- **CosyVoice (2024)** [Du et al. 2024, arXiv:2407.05407]: *supervised semantic tokens* from a VQ
  layer inserted into the encoder of the SenseVoice-Large ASR model (4,096 codes), so tokens are
  trained to be ASR-discriminative rather than self-supervised. A text-speech LM (300M) predicts the
  tokens conditioned on an x-vector speaker embedding and the prompt; a conditional flow matching
  model converts tokens to mels; a HiFTNet (HiFi-GAN plus iSTFT) vocoder makes audio.
- **CosyVoice 2 (2024)** [Du et al. 2024, arXiv:2412.10117]: FSQ replaces VQ in the tokenizer
  (project to $D = 8$, quantize each dimension to $[-K, K]$ with $K = 1$, so $3^8 = 6{,}561$ codes with
  100% utilization; index $\mu_i = \sum_{j=0}^{D-1} \bar h_{i,j}(2K+1)^j$ after shifting to
  non-negative), 25 tokens per second. The LM is initialized from Qwen2.5-0.5B, with no separate text
  encoder or speaker embedding in the LM. A chunk-aware causal flow matching model (mel at 50 Hz,
  tokens upsampled 2x) supports streaming and non-streaming with one model. `FunAudioLLM/CosyVoice2-0.5B`,
  Apache-2.0.
- **CosyVoice 3 (2025)** [Du et al. 2025, arXiv:2505.17589]: tokenizer built on the MinMo speech LLM's
  voice encoder with FSQ, trained with multiple supervised tasks; differentiable reward optimization
  (DiffRO) post-training; data scaled from 10k to 1M hours in 9 languages and 18 Chinese dialects;
  model scaled from 0.5B to 1.5B; new CV3-Eval benchmark. Open release: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
  (Apache-2.0, December 2025).

**Token-rate example.** CosyVoice 2 at 25 Hz with 6,561 codes: $25 \times \log_2 6561 = 25 \times 12.68 = 317$
bits/s. A 10-second utterance is only 250 LM tokens, versus 7,500 for EnCodec at 6 kbps.

### Fish-Speech, OpenAudio, and Fish Audio S2

- **Fish-Speech (2024)** [Liao et al. 2024, arXiv:2411.01156]: an LLM (no G2P, text is read directly)
  with a serial *Dual-AR* design: a slow Transformer over frames and a fast Transformer over the
  codebooks within each frame. The codec is Firefly-GAN with grouped FSQ (GFSQ), reporting 100%
  codebook use; trained on 720k hours.
- **OpenAudio S1 (June 2025)**: 4B closed model; **S1-mini** (0.5B) is open as
  `fishaudio/openaudio-s1-mini` under CC-BY-NC-SA-4.0.
- **Fish Audio S2 Pro (March 9, 2026)** [Fish Audio 2026, arXiv:2603.08823]: 4.4B parameters (4B
  slow AR plus 400M fast AR), RVQ codec with 10 codebooks at about 21 Hz, over 10M hours in 80+
  languages, RL alignment, natural-language instruction control, SGLang-based serving (RTF 0.195 and
  about 100 ms time-to-first-audio on one H200). Weights `fishaudio/s2-pro` under the Fish Audio
  Research License (free for research and non-commercial use; commercial by agreement).

### Llasa (2025)

Single-stream "just use Llama" TTS [Ye et al. 2025, arXiv:2502.04128]. XCodec2's single codebook of
65,536 FSQ codes at 50 Hz is added to the Llama vocabulary, and a standard causal LM is trained on
text followed by speech tokens. Llasa-1B, 3B, and 8B are initialized from Llama 3.2 1B and 3B and
Llama 3.1 8B and trained on up to 250k hours. The paper studies train-time scaling and inference-time
scaling (sampling many candidates and selecting with speech-understanding verifiers). `HKUSTAudio/Llasa-1B`
(1.37B parameters with the enlarged embedding table), CC-BY-NC-4.0.

### Spark-TTS (2025)

Qwen2.5-0.5B predicts BiCodec tokens [Wang et al. 2025, arXiv:2503.01710]. BiCodec splits speech into
*semantic tokens* (single VQ with 8,192 codes at 50 tokens per second, 650 bits/s, computed from the
average of wav2vec 2.0 XLSR-53 layers 11, 14, and 16) and 32 fixed-length *global tokens* (FSQ, from
an ECAPA-TDNN-style encoder over mels with learnable cross-attention queries) that carry speaker
attributes. A chain-of-thought-style scheme first predicts coarse attribute tokens (gender, pitch
level, speed), then fine attribute values, then global and semantic tokens, which enables voice
creation without a reference. VoxBox dataset of 100k hours. `SparkAudio/Spark-TTS-0.5B`,
CC-BY-NC-SA-4.0.

### Orpheus (2025)

Canopy Labs, March 2025. Llama-3.2-3B fine-tuned on about 100k hours of English to emit SNAC 24 kHz
tokens flattened as 7 tokens per coarse frame (about 83 tokens per second), with emotion tags such as
`<laugh>` and `<sigh>`. Streaming latency about 200 ms. Apache-2.0; `canopylabs/orpheus-3b-0.1-ft`
(3.78B parameters counting the enlarged embeddings, gated). A clean example of "an unmodified LLM plus
a small codec."

### Sesame CSM (2025)

Conversational Speech Model, 1B variant released March 2025 under Apache-2.0 (`sesame/csm-1b`, gated,
1.55B total parameters). A Llama-3.2-1B backbone runs once per 12.5 Hz frame over interleaved text and
audio from the whole conversation and predicts Mimi codebook 0; a small (about 100M) audio decoder
predicts codebooks 1-31. Native in Transformers (`CsmForConditionalGeneration`) since v4.52.1.
English-focused.

### Dia and Dia2 (2025)

- **Dia-1.6B** (Nari Labs, April 2025): encoder-decoder Transformer generating DAC 44 kHz tokens with
  the delay pattern for multi-speaker dialogue from a transcript with `[S1]` and `[S2]` tags, plus
  nonverbal tags. English only. Apache-2.0, 1.61B parameters.
- **Dia2** (1B and 2B): streaming dialogue TTS using the Kyutai Mimi codec, up to 2 minutes of English,
  Apache-2.0, `nari-labs/Dia2-1B` and `nari-labs/Dia2-2B`.

### Higgs Audio v2 (2025) and Higgs TTS 3 (2026)

- **Higgs Audio v2** (Boson AI): 5.8B parameters (3.6B Llama-3.2-3B-based LLM plus 2.2B "DualFFN"
  audio-specific feed-forward layers), a unified 25 Hz tokenizer for speech, music, and sounds with
  semantic distillation (2 kbps), multi-codebook delay pattern, trained on about 10M hours. License is a
  Llama-derivative community license. Hugging Face `bosonai/higgs-audio-v2-generation-3B-base` and
  `bosonai/higgs-audio-v2-tokenizer`.
- **Higgs TTS 3** (2026): about 4B parameters, 36-layer decoder, 8 codebooks at 25 fps with a delay
  pattern, 24 kHz, 102 languages, inline control tokens, research and non-commercial license with a
  creator-use grant. `bosonai/higgs-tts-3-4b`.

### VibeVoice (2025)

Microsoft's long-form, multi-speaker TTS based on *next-token diffusion* [Peng et al. 2025,
arXiv:2508.19205]:

- Two continuous tokenizers at **7.5 Hz** (3,200x downsampling of 24 kHz): an acoustic tokenizer, a
  $\sigma$-VAE with about 340M parameters each for encoder and decoder, and a semantic tokenizer with
  the same encoder architecture trained with an ASR proxy task.
- A Qwen2.5 LLM (1.5B or 7B) reads the voice prompts and the script; its hidden state at each position
  conditions a 4-layer diffusion head that denoises the next continuous acoustic latent (DDPM training
  with noise prediction, DPM-Solver++ sampling, classifier-free guidance). Tokenizers are frozen during
  LLM training.
- Up to 90 minutes in a 64k context with up to 4 speakers.
- **Release history.** Released August 25, 2025 under MIT. On September 5, 2025, Microsoft removed the
  TTS code from `microsoft/VibeVoice` after misuse; `microsoft/VibeVoice-1.5B` is marked disabled.
  `microsoft/VibeVoice-Realtime-0.5B` (December 3, 2025; MIT; about 1.0B parameters including
  tokenizers) remains available, and the repository has since shipped ASR models (VibeVoice-ASR,
  January 2026) and Transformers integration (March 2026).

### IndexTTS 2 and IndexTTS-2.5 (2025-2026)

Bilibili's autoregressive zero-shot TTS [Zhou et al. 2025, arXiv:2506.21619]:

- A text-to-semantic AR Transformer predicts MaskGCT-style semantic tokens. **Duration control**: the
  user can specify the exact number of output tokens (an explicit token-count embedding), or let the
  model choose freely.
- **Emotion and timbre disentanglement**: separate speaker and emotion prompts, with a gradient reversal
  layer to strip emotion from the speaker embedding. A Qwen3-based model (a 0.6B merge in the
  checkpoint) maps natural-language emotion descriptions to emotion vectors.
- A semantic-to-mel flow matching stage ("S2M") conditioned on GPT latent representations for
  stability in emotional speech, then BigVGAN-v2.
- Released September 8, 2025 (`IndexTeam/IndexTTS-2`). IndexTTS-2.5 (August 10, 2026, 0.8B,
  multilingual: Chinese, English, Japanese, Spanish, Arabic). License: the bilibili Model Use License
  Agreement (commercial use needs a separate agreement), not Apache-2.0 despite some third-party
  claims.

### Kyutai TTS and Pocket TTS (2025-2026)

- **Kyutai TTS 1.6B** (July 2025) [Zeghidour et al. 2025, DSM, arXiv:2509.08753]: *delayed streams
  modeling* treats text and audio as time-aligned parallel streams at 12.5 Hz; for TTS the audio stream
  is delayed relative to text (16 steps, 1.28 s), so audio starts after the first few words arrive and
  the model streams text in and audio out. A 1B temporal backbone plus a 600M depth Transformer over 32
  Mimi codebooks. English and French. CC-BY-4.0; voice conditioning is limited to precomputed voice
  embeddings. `pip install moshi`, `kyutai/tts-1.6b-en_fr`. The same framework gives Kyutai's streaming
  ASR by delaying text instead.
- **Pocket TTS** (January 2026): 100M parameters, runs about 6 times faster than real time on a MacBook
  Air M4 CPU, 6 languages, CC-BY-4.0 (`kyutai/pocket-tts`, gated), `pip install pocket-tts`. It's based
  on *continuous audio language models* (CALM): a Transformer backbone conditions a small MLP that
  generates the next continuous VAE latent frame by consistency modeling, avoiding lossy tokens
  [Rouard et al. 2025, arXiv:2509.06926].

### Chatterbox family (2025-2026)

Resemble AI, MIT license, `pip install chatterbox-tts`. A 0.5B Llama-based "T3" model predicts
CosyVoice-style S3 speech tokens from text and a reference; "S3Gen" (flow matching plus a HiFT
vocoder) makes audio. Controls: classifier-free guidance weight and an emotion "exaggeration" scalar.
All outputs carry a Perth perceptual watermark. Variants: original English (0.5B), Multilingual
(0.5B; V3 released June 10, 2026 with 23+ languages), Turbo (350M, English, distilled single-step
decoder instead of 10 steps, paralinguistic tags), and Nano (110M, CPU-oriented).

### Qwen3-TTS (2026)

Alibaba Qwen, released January 22, 2026, Apache-2.0, `pip install qwen-tts`. Discrete multi-codebook
LM with a Qwen3-TTS-Tokenizer-12Hz (12.5 Hz, 16 codebooks of 2,048, so $12.5 \times 16 \times 11 = 2.2$
kbps). Sizes 0.6B and 1.7B; variants Base (3-second voice cloning), CustomVoice (9 preset speakers
with instructions), and VoiceDesign (1.7B, voice from a text description). "Dual-track" streaming with
about 97 ms first-packet latency. 10 languages.

### Other notable 2026 open-weight releases

- **Voxtral TTS** (Mistral, March 26, 2026): 4B, Ministral-3B backbone, autoregressive plus flow
  matching, 9 languages, CC-BY-NC-4.0 weights (`mistralai/Voxtral-4B-TTS-2603`, about 8 GB in BF16).
- **Breeze TTS 2** (BreezeBlue, August 25, 2026): 3B, English and Chinese, codec based on Qwen3-TTS's,
  research and non-commercial weights license. Artificial Analysis reported it as the top open-weights
  model in its Speech Arena at release (aggregator ranking, not a paper result).

### Summary table of codec language models

The following table collects the numbers the course cites most. Dates are first open release.

| System | Date | Params | Tokens | Generation | License (weights) |
|---|---|---|---|---|---|
| VALL-E | 2023-01 | 2 x ~0.2B (12L, d=1024) | EnCodec 75 Hz x 8 | AR codebook 1 + NAR | Not released |
| SoundStorm | 2023-05 | 350M | SoundStream 50 Hz x 12 | Masked parallel | Not released |
| Bark | 2023-04 | 3 x 80M | Semantic 10k + EnCodec 8 | 3-stage AR | MIT |
| XTTS-v2 | 2023-11 | ~0.5B | VQ-VAE 21.5 Hz, 1024 codes | AR + HiFi-GAN | CPML (non-commercial) |
| MaskGCT | 2024-09 | 695M + 353M | Semantic 8192 + 12 RVQ | Masked, 2 stages | CC-BY-NC-4.0 |
| CosyVoice 2 | 2024-12 | 0.5B LM | FSQ 25 Hz, 6,561 | AR + flow matching | Apache-2.0 |
| Fun-CosyVoice3 | 2025-12 | 0.5B LM | MinMo FSQ 25 Hz | AR + flow matching | Apache-2.0 |
| Llasa | 2025-02 | 1B, 3B, 8B | XCodec2 50 Hz, 65,536 | Single-stream AR | CC-BY-NC-4.0 |
| Spark-TTS | 2025-03 | 0.5B | BiCodec 50 Hz + 32 global | Single-stream AR | CC-BY-NC-SA-4.0 |
| Orpheus | 2025-03 | 3B | SNAC 7 per frame, ~83/s | Flattened AR | Apache-2.0 |
| Sesame CSM | 2025-03 | 1B + 0.1B | Mimi 12.5 Hz x 32 | Backbone + depth decoder | Apache-2.0 |
| Dia | 2025-04 | 1.6B | DAC 44 kHz x 9 | Delay pattern | Apache-2.0 |
| OpenAudio S1-mini | 2025-06 | 0.5B | GFSQ codec | Dual-AR | CC-BY-NC-SA-4.0 |
| Kyutai TTS | 2025-07 | 1B + 0.6B | Mimi 12.5 Hz x 32 | Delayed streams | CC-BY-4.0 |
| Higgs Audio v2 | 2025-07 | 5.8B | 25 Hz x 8 | Delay pattern, DualFFN | Llama-derived custom |
| VibeVoice | 2025-08 | 1.5B, 7B | Continuous 7.5 Hz | Next-token diffusion | MIT (TTS code pulled) |
| IndexTTS2 | 2025-09 | Not checked (about 5.9 GB of checkpoints) | Semantic codec + S2M | AR + flow matching | bilibili license |
| Chatterbox | 2025-05 | 0.5B (Turbo 350M) | S3 tokens | AR + flow matching | MIT |
| Qwen3-TTS | 2026-01 | 0.6B, 1.7B | 12.5 Hz x 16 | Multi-codebook AR | Apache-2.0 |
| Pocket TTS | 2026-01 | 100M | Continuous VAE latents | CALM, consistency head | CC-BY-4.0 |
| Fish Audio S2 Pro | 2026-03 | 4.4B | RVQ ~21 Hz x 10 | Dual-AR + RL | Fish Audio Research License |
| Voxtral TTS | 2026-03 | 4B | Not checked | AR + flow matching | CC-BY-NC-4.0 |
| Breeze TTS 2 | 2026-08 | 3B | Qwen3-TTS-based codec | Backbone + depth decoder | Research, non-commercial |

### Lab ideas for codec LMs

- **Delay pattern round trip (CPU-60s).** Implement `apply_delay(codes, pad)` and `undo_delay` for
  $[K, T]$ grids; check the round trip is identity, that $S = T + K - 1$, and that a causal model reading
  the delayed grid never sees codebook $k$ of frame $t$ before codebooks $< k$ of frame $t$ (the checker
  verifies with a dependency test).
- **VALL-E-style NAR input construction (CPU-60s).** Given codes $C \in \{0..1023\}^{T \times 8}$ and
  embedding tables, build the NAR input for stage $j$ (sum of embeddings for codebooks $< j$) and the
  loss mask; match a reference tensor.
- **Tiny codec LM (GPU-A100, about 15-20 minutes, under 16 GB).** Encode 2-3 hours of LJSpeech with
  Mimi (first 8 codebooks) or EnCodec 1.5 kbps. Train a 6-layer, 20-40M parameter decoder-only
  Transformer over phonemes plus delayed codes (or over codebook 1 only, plus copy codebooks 2..K from a
  NAR baseline). The checker measures validation cross-entropy per codebook against thresholds and
  decodes one sample for listening. Intelligibility won't be high at this scale; don't check WER.
- **MaskGIT decoding schedule (CPU-60s).** Implement the cosine unmasking schedule and
  confidence-based selection over a given logits oracle; check the number of tokens revealed per
  iteration and that the final grid has no masks.
- **Zero-shot cloning measurement (GPU-L4, about 5 minutes, under 8 GB).** Run CosyVoice2-0.5B or
  Chatterbox (MIT) on 20 prompt-target pairs, compute WER with Whisper and speaker similarity with an
  ECAPA or WavLM verifier. The checker verifies metric code against precomputed references rather than
  asserting absolute quality.

---

## Voice cloning, prosody, emotion, and singing

### Speaker embeddings and zero-shot adaptation

- **Speaker-encoder transfer (SV2TTS).** Train a speaker verification encoder (GE2E d-vector) on
  thousands of untranscribed speakers, freeze it, and condition a Tacotron 2 on its embedding
  [Jia et al. 2018, arXiv:1806.04558]. Unseen speakers work from a few seconds of audio, with
  similarity limited by the embedding's capacity.
- **Embedding types.** d-vectors (GE2E), x-vectors (TDNN with statistics pooling), ECAPA-TDNN (192-d,
  SpeechBrain `speechbrain/spkrec-ecapa-voxceleb`, Apache-2.0), WavLM-based verifiers. YourTTS and
  XTTS use H/ASP and Perceiver latents respectively; CosyVoice 1 uses x-vectors.
- **Conditioning mechanisms.** Concatenation or addition to encoder outputs, FiLM or AdaIN in the
  decoder (StyleTTS 2), cross-attention to a set of prompt latents (XTTS Perceiver, 32 latents), and
  in-context prompting (VALL-E and all codec LMs: the prompt's tokens precede the target's).
- **Fine-tuning adaptation.** Fine-tune all or part of a model (speaker embedding only, adapters,
  LoRA) on minutes of target speech. Higher similarity than zero-shot but needs training per voice.
- **Prompt handling gotchas.** The prompt transcript must match the prompt audio in codec LMs, or the
  model reads the wrong text. Long prompts leak prompt content into the output. Similarity is
  usually reported against the original prompt (SIM-o) and sometimes against the codec-reconstructed
  prompt (SIM-r); they aren't comparable.

### Prosody and style control

- **Explicit variance features**: FastSpeech 2 pitch, energy, and duration can be scaled or edited
  per phoneme at inference.
- **Reference encoders and global style tokens (GST)**: a reference encoder compresses a mel to a
  prosody embedding [Skerry-Ryan et al. 2018, arXiv:1803.09047]; GST learns a bank of style tokens and
  attends over them, so styles can be picked without reference audio [Wang et al. 2018,
  arXiv:1803.09017].
- **Sampled style**: StyleTTS 2 samples a style vector with diffusion; VITS's stochastic duration
  predictor samples rhythm.
- **Natural-language descriptions**: Parler-TTS style captioning ("a calm female voice, slightly fast,
  close microphone") [Lyth and King 2024, arXiv:2402.01912]; Qwen3-TTS VoiceDesign, Fish Audio S2
  instructions, and Spark-TTS attribute tokens are later versions of this idea.
- **Inline tags**: Orpheus (`<laugh>`), Dia (`(laughs)`), Chatterbox Turbo, Higgs TTS 3.

### Emotion

Approaches: emotion labels as conditioning, emotion embeddings from an emotion recognizer
(emotion2vec), separate emotion prompts disentangled from timbre (IndexTTS2 uses a gradient reversal
layer so the speaker encoder can't encode emotion), an "exaggeration" scalar (Chatterbox), and RL
post-training toward emotion rewards (CosyVoice 3 DiffRO, Fish Audio S2). Evaluation uses emotion
classifiers plus human preference tests.

### Safety and consent

Zero-shot cloning enables impersonation. Course labs should use consented voices (LJSpeech, LibriTTS
speakers, the student's own voice) and can demonstrate watermarking: Chatterbox embeds Perth
watermarks, and AudioSeal is a localized neural watermark with a detector [San Roman et al. 2024,
arXiv:2401.17264]. VibeVoice's withdrawal (September 2025) is a concrete case study.

### Singing voice synthesis: DiffSinger (brief)

Singing needs note pitch, long vowels, and wide F0 range. DiffSinger's *shallow diffusion mechanism*
first predicts a coarse mel with a simple auxiliary decoder, diffuses it forward to a shallow step
$k < T$ (where the noised coarse and noised ground-truth distributions intersect), and runs the reverse
process only from $k$, which cuts inference steps and improves quality over a plain L1 decoder. A
boundary predictor chooses $k$ adaptively. Evaluated on the PopCS Mandarin pop dataset
[Liu et al. 2022, arXiv:2105.02446]. VISinger applies the VITS recipe to singing.

### Lab idea

- **Speaker similarity with ECAPA (CPU-60s with cached embeddings, or GPU-L4 under 1 minute to embed
  30 clips).** Compute ECAPA-TDNN embeddings for same-speaker and different-speaker pairs from
  LibriTTS, compute cosine similarity, and estimate the equal error rate. The checker verifies EER
  below a threshold and that cosine similarity is computed on L2-normalized embeddings.

---

## Evaluation

### Subjective tests

- **MOS (mean opinion score)**: absolute category rating 1 (bad) to 5 (excellent), ITU-T P.800.
  Report the mean with a 95% confidence interval
  $\bar s \pm 1.96\,\hat\sigma / \sqrt{n}$ over $n$ ratings (better: a mixed model or bootstrap over
  raters and items). Example: 200 ratings with $\hat\sigma = 0.8$ gives a half-width of
  $1.96 \times 0.8 / \sqrt{200} = 0.111$. MOS depends heavily on the rater pool and the other systems
  in the test, so compare only within a test.
- **CMOS (comparative MOS)**: raters compare pairs on $-3$ to $+3$; a positive score favors the system
  under test. SMOS rates speaker similarity on 1-5.
- **MUSHRA** (ITU-R BS.1534-3): raters score 0-100 for all systems side by side, including a hidden
  reference and a low-pass anchor (3.5 kHz; often also 7 kHz). Post-screening excludes raters who
  score the hidden reference below 90 for more than 15% of items. More sensitive than MOS for
  high-quality codecs and vocoders.
- **Preference (AB) tests** and arena-style Elo (Artificial Analysis Speech Arena, TTS Arena) are
  common for recent systems but aren't controlled experiments.

### Automatic MOS predictors

- **UTMOS** (UTMOS22 "strong learner"): SSL (wav2vec 2.0) features fine-tuned to predict MOS, the
  VoiceMOS Challenge 2022 winner [Saeki et al. 2022, arXiv:2204.02152]. Commonly loaded through the
  SpeechMOS torch hub (`torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")`), 16 kHz input.
  Range roughly 1-5; LJSpeech ground truth scores about 4.3-4.4.
- **UTMOSv2**: wav2vec 2.0 features fused with an EfficientNetV2 spectrogram model, VoiceMOS 2024
  Track 1 system; `pip install git+https://github.com/sarulab-speech/UTMOSv2.git`.
- **DNSMOS**: P.808 overall and P.835 SIG (speech), BAK (background), OVRL (overall), ONNX models from
  the Microsoft DNS Challenge; available in torchmetrics as
  `DeepNoiseSuppressionMeanOpinionScore` (needs `onnxruntime`, `librosa`). Designed for noise
  suppression, less sensitive to TTS artifacts [Reddy et al. 2021, arXiv:2010.15258].
- **NISQA** is another non-intrusive predictor. All predictors are biased toward their training data;
  treat them as regression tests, not absolute truth.

### Intelligibility: WER and CER

Transcribe synthesized audio with a fixed ASR model and compute WER against the input text.

- Seed-TTS-eval: Whisper-large-v3 for English, Paraformer-zh for Mandarin.
- VALL-E and LibriSpeech-PC evaluations: HuBERT-Large fine-tuned on LibriSpeech 960 h (CTC, no LM),
  `facebook/hubert-large-ls960-ft`.
- **Gotchas.** Text normalization dominates small WER differences: apply the Whisper English text
  normalizer to both hypothesis and reference. Whisper can "correct" mispronunciations using its
  language model, so it underestimates errors; CTC models are stricter. Report the ASR model and the
  normalizer.

### Speaker similarity

Cosine similarity between speaker embeddings of the synthesized audio and the prompt (or reference):
$\text{SIM} = \frac{e_1^\top e_2}{\lVert e_1\rVert \lVert e_2\rVert} \in [-1, 1]$.

- Seed-TTS-eval and most 2024-2026 papers: WavLM-large fine-tuned for speaker verification (an
  ECAPA-TDNN head, from the UniSpeech repository).
- VALL-E: WavLM-TDNN.
- Lightweight lab options: SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) and
  `microsoft/wavlm-base-plus-sv` in Transformers. Scores from different verifiers aren't comparable.

### Mel cepstral distortion with DTW

**Intuition.** Distance between mel-cepstra of reference and synthesized speech, in dB, after aligning
frames with dynamic time warping because durations differ.

$$
\operatorname{MCD}(c, \hat c) = \frac{10}{\ln 10}\sqrt{2\sum_{d=1}^{D}(c_d - \hat c_d)^2}\ \text{dB},
\qquad
\operatorname{MCD}_{\text{DTW}} = \frac{1}{|\pi^*|}\sum_{(n, m) \in \pi^*} \operatorname{MCD}(c^{(n)}, \hat c^{(m)}),
$$

where $c_d$ are mel-cepstral coefficients (order $D$, commonly 13 or 24, from SPTK/`pysptk` mcep or
WORLD's spectral envelope with frequency warping $\alpha$ set by sample rate, 0.42 at 16 kHz), $c_0$
(energy) is excluded, and $\pi^*$ is the DTW path minimizing total distance. Conventions differ between
toolkits (MFCC versus mcep, $D$, whether to average over path length or reference length), so MCD
values are only comparable within one implementation (`pymcd` and `mcd` on PyPI are common).

**Hand-worked example.** $c = (1.0, 0.5, -0.2)$, $\hat c = (0.9, 0.3, -0.1)$: squared differences sum to
$0.06$, $\sqrt{0.12} = 0.3464$, times $10/\ln 10 = 4.3429$ gives $1.504$ dB.

### F0 metrics

- **F0 RMSE** over frames voiced in both signals (after DTW if lengths differ), in Hz or in cents
  $1200\log_2(f/\hat f)$.
- **V/UV error rate**: fraction of frames where voicing decisions disagree. Also F0 Pearson
  correlation, and FastSpeech 2's moments (standard deviation, skewness, kurtosis) of the F0
  distribution.
- **Example.** $f = [200, 210, 0, 220]$, $\hat f = [190, 215, 180, 0]$ (0 means unvoiced). Both voiced in
  frames 1-2: errors $10, -5$, RMSE $\sqrt{(100 + 25)/2} = 7.91$ Hz. V/UV error $2/4 = 50\%$.

### Codec metrics

See [Codec metrics](#codec-metrics): bitrate, tokens per second, mel distance, STFT distance, SI-SDR,
ViSQOL, PESQ, STOI, UTMOS, WER and SIM of reconstructions, and codebook utilization.

### Benchmarks

- **Seed-TTS-eval** [Anastassiou et al. 2024, arXiv:2406.02430]: test-en (Common Voice; 1,000 in the
  paper, 1,088 in the released meta file), test-zh (DiDiSpeech-2; 2,000 in the paper, 2,020 released),
  and test-hard (difficult Mandarin sentences). Meta format:
  `filename|prompt text|prompt audio|synthesis text|reference audio`. Scripts `cal_wer.sh` and
  `cal_sim.sh`. Seed-TTS weights were not released.
- **LibriSpeech-PC test-clean 4-10 s subset** (1,127 utterances in F5-TTS's setup).
- **CV3-Eval** (CosyVoice 3) for multilingual and in-the-wild conditions.

### Lab ideas for evaluation

- **MCD-DTW and F0 RMSE (CPU-60s).** Implement DTW over cepstral frames and the MCD formula; the
  checker compares against a reference on fixture cepstra, checks MCD is 0 for identical inputs, and
  checks symmetry under swapping. Implement F0 RMSE and V/UV error on fixture contours.
- **TTS evaluation harness (GPU-L4, about 5 minutes, under 8 GB).** For 30 synthesized and reference
  clips, compute WER with Whisper (large-v3 or a smaller model for speed) using the Whisper normalizer,
  SIM with ECAPA, and UTMOS. The checker validates each metric against precomputed values within
  tolerance.
- **MOS confidence intervals (CPU-60s).** Given a ratings table (raters by items by systems), compute
  MOS with a bootstrap over raters and items, run MUSHRA post-screening, and report which system pairs
  differ significantly.

---

## Datasets

| Dataset | Size | Speakers | Sample rate | License | Notes |
|---|---|---|---|---|---|
| LJSpeech 1.1 | 13,100 clips, 23:55:17 (about 24 h), 2.6 GB | 1 (female, US English) | 22,050 Hz, 16-bit | Public domain | Clips 1.11-10.10 s; LibriVox non-fiction readings; normalized transcripts provided |
| VCTK 0.92 | about 44 h, 10.9 GB | 110 (English accents) | 48 kHz (recorded at 96 kHz) | CC-BY-4.0 | About 400 sentences per speaker; two microphones (`mic1` DPA 4035, `mic2` MKH 800); long leading and trailing silences need trimming |
| LibriTTS | 585 h | 2,456 | 24 kHz | CC-BY-4.0 | Derived from LibriSpeech with sentence splits and original-case text; train-clean-100, train-clean-360, train-other-500, dev, and test splits |
| LibriTTS-R | 585 h | 2,456 | 24 kHz | CC-BY-4.0 | LibriTTS restored with Miipher (OpenSLR 141); better for vocoder and codec training |
| Hi-Fi TTS | about 292 h | 10 (at least 17 h each) | 44.1 kHz | CC-BY-4.0 | LibriVox-based (OpenSLR 109) |
| Emilia | 101.7k h | many | 24 kHz | CC-BY-NC-4.0 | In-the-wild, 6 languages (EN, ZH, DE, FR, JA, KO), WebDataset tar shards |
| Emilia-YODAS | 113.9k h | many | 24 kHz | CC-BY-4.0 | Together with Emilia forms Emilia-Large (215.6k h) |

**Small subsets suited to labs.**

- **LJSpeech first 100-500 clips** (8-40 minutes of audio, under 100 MB): single speaker, public domain,
  the default for overfitting tests and tiny vocoder or acoustic-model training.
- **LibriTTS or LibriTTS-R dev-clean and test-clean** (each about 9 h, a few hundred MB compressed):
  multi-speaker evaluation, codec sweeps, speaker similarity.
- **VCTK, 5-10 speakers** at 16 or 24 kHz after trimming: cloning and speaker embedding labs.
- **Emilia, one or two language shards** only for GPU labs; note the non-commercial license.
- Precompute and ship fixtures (mels, MFA durations, codes, embeddings) for all CPU-60s labs so the
  checker never downloads large files.

**Gotchas.** Resample consistently (LJSpeech is 22,050 Hz, most codecs want 16 or 24 kHz). Trim VCTK
silence. LibriTTS text is original-case with punctuation; LJSpeech's `normalized_text` column already
expands numbers. MFA alignments for LJSpeech are widely mirrored; ship them rather than running MFA in a
lab.

---

## Checkpoints and packages for serverless labs

Sizes are from Hugging Face file listings or the model API. VRAM estimates for inference in fp16 or
bf16 at batch 1 are rough (weights plus activations for short clips).

| Purpose | Hugging Face name | Size | License | Package | Fits |
|---|---|---|---|---|---|
| Vocoder | `speechbrain/tts-hifigan-ljspeech` | 56 MB (about 14M params) | Apache-2.0 | `speechbrain` | CPU |
| Acoustic model | `speechbrain/tts-tacotron2-ljspeech` | 113 MB | Apache-2.0 | `speechbrain` | CPU |
| Vocoder | `charactr/vocos-mel-24khz` | 54 MB (13.5M) | MIT | `vocos` | CPU |
| Vocoder | `nvidia/bigvgan_v2_24khz_100band_256x` | 450 MB generator (112M) | MIT | `bigvgan` (repo) | CPU slow, L4 |
| Codec | `facebook/encodec_24khz` | 23.3M | MIT | `transformers` or `encodec` | CPU |
| Codec | `descript/dac_44khz` (also 24k, 16k) | 76.6M | MIT | `transformers` or `descript-audio-codec` | CPU, L4 |
| Codec | `kyutai/mimi` | 96.2M | CC-BY-4.0 | `transformers` (`MimiModel`) or `moshi` | CPU, L4 |
| Codec | `hubertsiuzdak/snac_24khz` | 79 MB (19.8M) | MIT | `snac` | CPU |
| Codec | `novateur/WavTokenizer-large-speech-75token` | 1.75 GB checkpoint (includes training state) | MIT | repo code | L4 |
| Codec | `HKUSTAudio/xcodec2` | 822.7M | CC-BY-NC-4.0 | `xcodec2` | L4 |
| Codec | `fnlp/SpeechTokenizer` | 482 MB | Not checked | `speechtokenizer` | CPU, L4 |
| TTS, CPU demo | `hexgrad/Kokoro-82M` | 327 MB | Apache-2.0 | `kokoro` (needs espeak-ng for some languages) | CPU |
| TTS, CPU demo | `kyutai/pocket-tts` | 100M (gated) | CC-BY-4.0 | `pocket-tts` | CPU |
| TTS, zero-shot | `FunAudioLLM/CosyVoice2-0.5B` | about 4 GB of files | Apache-2.0 | repo code | L4 (about 4-6 GB) |
| TTS, zero-shot | `ResembleAI/chatterbox` | about 2 GB per T3 variant | MIT | `chatterbox-tts` | L4 |
| TTS, zero-shot | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 2.5 GB (914M incl. tokenizer) | Apache-2.0 | `qwen-tts` | L4 |
| TTS, streaming | `kyutai/tts-1.6b-en_fr` | 3.7 GB + 385 MB Mimi | CC-BY-4.0 | `moshi` | L4 |
| TTS, codec LM | `sesame/csm-1b` | 1.55B (gated) | Apache-2.0 | `transformers` | L4 |
| TTS, dialogue | `nari-labs/Dia-1.6B` | 1.61B | Apache-2.0 | `dia` repo | L4 (about 10 GB reported by the authors; not checked) |
| TTS, LLM + SNAC | `canopylabs/orpheus-3b-0.1-ft` | 3.78B (gated) | Apache-2.0 | `orpheus-speech` or `vllm` | A100 or L4 in bf16 |
| TTS, research only | `HKUSTAudio/Llasa-1B`, `SparkAudio/Spark-TTS-0.5B`, `amphion/MaskGCT`, `fishaudio/openaudio-s1-mini`, `coqui/XTTS-v2` | 0.5-1.4B | Non-commercial (CC-BY-NC, CC-BY-NC-SA, CPML) | various | L4 |
| Speaker verification | `speechbrain/spkrec-ecapa-voxceleb` | about 80 MB | Apache-2.0 | `speechbrain` | CPU |
| Speaker verification | `microsoft/wavlm-base-plus-sv` | about 100M (405 MB file) | Not checked on the card | `transformers` | CPU |
| MOS predictor | `tarepan/SpeechMOS` (`utmos22_strong`) | about 400 MB | Not checked | torch hub | CPU, L4 |
| ASR for WER | `openai/whisper-large-v3` (or `whisper-small` for CPU) | 1.55B | Apache-2.0 | `transformers` | L4 |
| ASR for WER | `facebook/hubert-large-ls960-ft` | 316M | Apache-2.0 | `transformers` | CPU, L4 |

**Package notes.**

- `torchaudio` entered maintenance mode in 2025: APIs deprecated in 2.8 were removed in 2.9, and
  decoding and encoding moved to TorchCodec. `transforms`, `functional`, `models`, and `pipelines`
  (including `Tacotron2TTSBundle`) remain. Use `soundfile` or TorchCodec for I/O in labs, and pin
  versions.
- MAS: `monotonic_align` (Cython, needs a build step), numba ports, or Super-MAS (Triton). For CPU labs,
  the student writes the numpy DP themselves.
- G2P: `phonemizer` needs the system package `espeak-ng`; `g2p_en` doesn't. Prefer shipping phoneme
  fixtures for CPU labs.
- Coqui TTS: use the maintained `coqui-tts` fork (Idiap) rather than the archived `TTS` package.

---

## Proposed lesson ordering

1. **The TTS pipeline and text front end.** History from unit selection to neural TTS, text
   normalization, G2P, mel settings. *Lab (CPU-60s):* number and currency normalizer plus CMUdict G2P
   with fallback, checked against 50 hidden strings.
2. **Autoregressive acoustic models and attention.** Tacotron 2 architecture, location-sensitive
   attention, stop token, failure modes, guided attention. *Lab (CPU-60s):* implement one
   location-sensitive attention step and the guided attention loss; optional GPU-L4 extension overfits
   a tiny Tacotron on 50 clips and checks alignment diagonality.
3. **Durations, length regulation, and alignment search.** FastSpeech, FastSpeech 2 variance adaptor,
   Glow-TTS MAS. *Lab (CPU-60s):* batched length regulator plus MAS verified against brute force on
   small matrices.
4. **Neural vocoders I: WaveNet to GANs.** Dilated causal convs, mu-law, receptive field, WaveRNN and
   LPCNet ideas, MR-STFT loss. *Lab (CPU-60s):* causal dilated stack with a receptive-field impulse
   test, mu-law round trip, and multi-resolution STFT loss.
5. **Neural vocoders II: HiFi-GAN, BigVGAN, Vocos.** MPD and MSD, LSGAN, feature matching, Snake and
   anti-aliasing, iSTFT heads. *Lab (GPU-A100, about 5-10 minutes):* train a HiFi-GAN V2-sized
   generator on 100 LJSpeech clips and pass a mel L1 threshold; CPU part implements MPD reshaping,
   Snake, and the Vocos head.
6. **End-to-end TTS: VITS and its descendants.** CVAE plus flow plus GAN, stochastic duration
   predictor, VITS2, YourTTS, StyleTTS 2, Kokoro. *Lab (CPU-60s):* implement the VITS KL term on
   fixtures; demo Kokoro on CPU and measure real-time factor.
7. **Quantization and neural audio codecs.** VQ-VAE, straight-through, commitment loss, EMA,
   collapse, RVQ, quantizer dropout, FSQ; SoundStream, EnCodec, DAC, Mimi, SNAC, single-codebook
   codecs. *Lab (CPU-60s):* RVQ and FSQ from scratch with residual-norm and utilization checks, plus a
   bitrate calculator; *GPU-L4 (about 5 minutes):* bitrate-versus-quality sweep over EnCodec, DAC,
   Mimi, and SNAC.
8. **Codec language models.** Semantic versus acoustic tokens, AudioLM, VALL-E, delay and depth
   patterns, SoundStorm and MaskGCT, CosyVoice, Llasa, CSM, VibeVoice, and the 2025-2026 landscape.
   *Lab (CPU-60s):* delay-pattern apply and undo plus VALL-E NAR input construction; *GPU-A100 (about
   15-20 minutes):* train a tiny codec LM on Mimi tokens and pass per-codebook cross-entropy thresholds.
9. **Evaluation, cloning, and responsible use.** MOS, CMOS, MUSHRA, UTMOS, DNSMOS, WER, SIM, MCD-DTW,
   F0 RMSE, Seed-TTS-eval; speaker embeddings, prosody and emotion control, watermarking. *Lab
   (CPU-60s):* MCD-DTW, F0 RMSE, and MOS bootstrap intervals; *GPU-L4 (about 5 minutes):* a WER, SIM,
   and UTMOS harness over zero-shot clones from an MIT or Apache-2.0 model.

---

## Unverified or uncertain facts

The following items couldn't be fully verified or rest on secondary sources:

- **VITS loss weights.** The paper writes the loss unweighted; the weights $c_{\text{mel}} = 45$ and
  $c_{\text{kl}} = 1$ are from memory of the official `configs/ljs_base.json`, not rechecked.
- **Vocos mel loss weight.** Recalled as 45 in the reference training config; the paper doesn't state a
  weight.
- **Tacotron 2 stop-token loss.** The paper describes stop prediction but lists only the MSE terms; the
  BCE term is from reference implementations.
- **DAC 24 kHz and 16 kHz bitrates.** The DAC README lists 8 kbps for all three models, but the Hugging
  Face configs give 32 codebooks at 75 Hz (24 kbps) and 12 codebooks at 50 Hz (6 kbps). The
  configs' `hop_length: 512` field contradicts their stride products (320), so compute from the strides
  and verify by encoding a clip.
- **Higgs Audio v2 tokenizer codebook size.** 1024 is inferred from "8 codebooks, 25 Hz, 2 kbps"; the
  training-data figure of about 10M hours is from memory of the release blog.
- **Higgs Audio v2 license name.** Described on the model card as a Llama-derivative license; the exact
  name wasn't rechecked.
- **Chatterbox internals.** The Llama-based 0.5B T3, S3 tokens, and HiFT vocoder come from the README's
  acknowledgements and prior knowledge, not a paper.
- **Kokoro "decoder only".** From the model card; how the style vectors were produced isn't documented.
- **Breeze TTS 2 and arena rankings.** From the vendor's model card and Artificial Analysis posts;
  aggregator Elo values differ between pages (1,125 versus 1,215 in different categories).
- **Fish Audio S2 license.** The model card says Fish Audio Research License; some third-party pages
  claim MIT. Trust the model card.
- **IndexTTS2 license.** The repository says bilibili Model Use License Agreement; some third-party
  pages claim Apache-2.0. Trust the repository. Parameter count wasn't found.
- **Pocket TTS internals.** The CALM paper describes the approach; the exact Pocket TTS architecture
  (VAE frame rate, consistency head size) wasn't checked.
- **Qwen3-TTS release month.** A summary of the model card read "January 2025"; Simon Willison's
  post and the Qwen blog date the release January 22, 2026, which is consistent with the Hugging Face
  last-modified dates. Use January 22, 2026.
- **Seed-TTS test-hard size** wasn't found in the repository README.
- **Lab wall times, VRAM figures, and metric thresholds** are estimates and must be calibrated with a
  reference run before shipping.
- **MUSHRA post-screening numbers** (hidden reference below 90 for more than 15% of items) are from
  memory of BS.1534-3.
- **Bark license history** (CC-BY-NC relicensed to MIT in 2023) and the **Chatterbox first-release
  month** (May 2025) are from memory.
- **UTMOS on LJSpeech ground truth** (about 4.3-4.4) is from memory of published tables; calibrate on
  the lab's own fixture.
- **SpeechMOS checkpoint size** (about 400 MB) wasn't rechecked; the `v1.2.0` release tag exists.

---

## Bibliography

Classic pipeline and acoustic models:

- Hunt, A. and Black, A. (1996). Unit selection in a concatenative speech synthesis system using a large speech database. ICASSP. https://doi.org/10.1109/ICASSP.1996.541110
- Zen, H., Tokuda, K., and Black, A. (2009). Statistical parametric speech synthesis. Speech Communication. https://doi.org/10.1016/j.specom.2009.04.004
- Morise, M., Yokomori, F., and Ozawa, K. (2016). WORLD: a vocoder-based high-quality speech synthesis system. IEICE. https://doi.org/10.1587/transinf.2015EDP7457
- Wang, Y. et al. (2017). Tacotron. https://arxiv.org/abs/1703.10135
- Shen, J. et al. (2018). Natural TTS synthesis by conditioning WaveNet on mel spectrogram predictions (Tacotron 2). https://arxiv.org/abs/1712.05884
- Chorowski, J. et al. (2015). Attention-based models for speech recognition (location-aware attention). https://arxiv.org/abs/1506.07503
- Tachibana, H. et al. (2018). DC-TTS with guided attention. https://arxiv.org/abs/1710.08969
- Li, N. et al. (2019). Neural speech synthesis with Transformer network. https://arxiv.org/abs/1809.08895
- Ren, Y. et al. (2019). FastSpeech. https://arxiv.org/abs/1905.09263
- Ren, Y. et al. (2021). FastSpeech 2. https://arxiv.org/abs/2006.04558
- Kim, J. et al. (2020). Glow-TTS. https://arxiv.org/abs/2005.11129
- Lee, J. et al. (2024). Super Monotonic Alignment Search. https://arxiv.org/abs/2409.07704
- Kim, J., Kong, J., and Son, J. (2021). VITS. https://arxiv.org/abs/2106.06103
- Kong, J. et al. (2023). VITS2. https://arxiv.org/abs/2307.16430
- Casanova, E. et al. (2022). YourTTS. https://arxiv.org/abs/2112.02418
- Mehta, S. et al. (2024). Matcha-TTS. https://arxiv.org/abs/2309.03199
- Li, Y. A. et al. (2023). StyleTTS 2. https://arxiv.org/abs/2306.07691
- Kokoro-82M model card. https://huggingface.co/hexgrad/Kokoro-82M

Vocoders:

- van den Oord, A. et al. (2016). WaveNet. https://arxiv.org/abs/1609.03499
- van den Oord, A. et al. (2018). Parallel WaveNet. https://arxiv.org/abs/1711.10433
- Ping, W., Peng, K., and Chen, J. (2019). ClariNet. https://arxiv.org/abs/1807.07281
- Kalchbrenner, N. et al. (2018). Efficient neural audio synthesis (WaveRNN). https://arxiv.org/abs/1802.08435
- Valin, J.-M. and Skoglund, J. (2019). LPCNet. https://arxiv.org/abs/1810.11846
- Prenger, R., Valle, R., and Catanzaro, B. (2019). WaveGlow. https://arxiv.org/abs/1811.00002
- Kumar, K. et al. (2019). MelGAN. https://arxiv.org/abs/1910.06711
- Yamamoto, R., Song, E., and Kim, J.-M. (2020). Parallel WaveGAN. https://arxiv.org/abs/1910.11480
- Kong, J., Kim, J., and Bae, J. (2020). HiFi-GAN. https://arxiv.org/abs/2010.05646 (code: https://github.com/jik876/hifi-gan)
- Jang, W. et al. (2021). UnivNet. https://arxiv.org/abs/2106.07889
- Lee, S.-g. et al. (2023). BigVGAN. https://arxiv.org/abs/2206.04658 (v2 checkpoints: https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
- Kaneko, T. et al. (2022). iSTFTNet. https://arxiv.org/abs/2203.02395
- Siuzdak, H. (2024). Vocos. https://arxiv.org/abs/2306.00814 (checkpoint: https://huggingface.co/charactr/vocos-mel-24khz)

Quantization and codecs:

- van den Oord, A., Vinyals, O., and Kavukcuoglu, K. (2017). VQ-VAE. https://arxiv.org/abs/1711.00937
- Mentzer, F. et al. (2024). Finite scalar quantization. https://arxiv.org/abs/2309.15505
- Zeghidour, N. et al. (2021). SoundStream. https://arxiv.org/abs/2107.03312
- Défossez, A. et al. (2022). EnCodec. https://arxiv.org/abs/2210.13438 (code: https://github.com/facebookresearch/encodec)
- Kumar, R. et al. (2023). High-fidelity audio compression with improved RVQGAN (DAC). https://arxiv.org/abs/2306.06546 (code: https://github.com/descriptinc/descript-audio-codec)
- Défossez, A. et al. (2024). Moshi and Mimi. https://arxiv.org/abs/2410.00037
- Siuzdak, H., Grötschla, F., and Lanzendörfer, L. (2024). SNAC. https://arxiv.org/abs/2410.14411
- Zhang, X. et al. (2024). SpeechTokenizer. https://arxiv.org/abs/2308.16692
- Ji, S. et al. (2025). WavTokenizer. https://arxiv.org/abs/2408.16532
- Ye, Z. et al. (2025). X-Codec: Codec does matter. https://arxiv.org/abs/2408.17175
- Xin, D. et al. (2024). BigCodec. https://arxiv.org/abs/2409.05377
- Parker, J. et al. (2025). Scaling Transformers for low-bitrate speech coding (Stable Codec). https://arxiv.org/abs/2411.19842
- Chinen, M. et al. (2020). ViSQOL v3. https://arxiv.org/abs/2004.09584

Codec language models:

- Borsos, Z. et al. (2023). AudioLM. https://arxiv.org/abs/2209.03143
- Wang, C. et al. (2023). VALL-E. https://arxiv.org/abs/2301.02111
- Zhang, Z. et al. (2023). VALL-E X. https://arxiv.org/abs/2303.03926
- Chen, S. et al. (2024). VALL-E 2. https://arxiv.org/abs/2406.05370
- Kharitonov, E. et al. (2023). SPEAR-TTS. https://arxiv.org/abs/2302.03540
- Borsos, Z. et al. (2023). SoundStorm. https://arxiv.org/abs/2305.09636
- Copet, J. et al. (2023). MusicGen (codebook interleaving patterns). https://arxiv.org/abs/2306.05284
- Wang, Y. et al. (2025). MaskGCT. https://arxiv.org/abs/2409.00750
- Suno Bark model card. https://github.com/suno-ai/bark/blob/main/model-card.md
- Betker, J. (2023). Tortoise: Better speech synthesis through scaling. https://arxiv.org/abs/2305.07243
- Casanova, E. et al. (2024). XTTS. https://arxiv.org/abs/2406.04904
- Du, Z. et al. (2024). CosyVoice. https://arxiv.org/abs/2407.05407
- Du, Z. et al. (2024). CosyVoice 2. https://arxiv.org/abs/2412.10117
- Du, Z. et al. (2025). CosyVoice 3. https://arxiv.org/abs/2505.17589 (weights: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
- Liao, S. et al. (2024). Fish-Speech. https://arxiv.org/abs/2411.01156
- Fish Audio (2026). Fish Audio S2 technical report. https://arxiv.org/abs/2603.08823 (weights: https://huggingface.co/fishaudio/s2-pro)
- Ye, Z. et al. (2025). Llasa and XCodec2. https://arxiv.org/abs/2502.04128
- Wang, X. et al. (2025). Spark-TTS. https://arxiv.org/abs/2503.01710
- Canopy Labs. Orpheus TTS. https://github.com/canopyai/Orpheus-TTS
- Sesame. CSM-1B. https://huggingface.co/sesame/csm-1b and https://github.com/SesameAILabs/csm
- Nari Labs. Dia and Dia2. https://github.com/nari-labs/dia and https://github.com/nari-labs/dia2
- Boson AI. Higgs Audio v2. https://github.com/boson-ai/higgs-audio ; Higgs TTS 3: https://huggingface.co/bosonai/higgs-tts-3-4b
- Peng, Z. et al. (2025). VibeVoice technical report. https://arxiv.org/abs/2508.19205 (repository: https://github.com/microsoft/VibeVoice)
- Zhou, S. et al. (2025). IndexTTS2. https://arxiv.org/abs/2506.21619 (repository: https://github.com/index-tts/index-tts)
- Zeghidour, N. et al. (2025). Delayed streams modeling (Kyutai TTS). https://arxiv.org/abs/2509.08753 (model: https://huggingface.co/kyutai/tts-1.6b-en_fr)
- Rouard, S. et al. (2025). Continuous audio language models (Pocket TTS). https://arxiv.org/abs/2509.06926 (model: https://huggingface.co/kyutai/pocket-tts)
- Resemble AI. Chatterbox. https://github.com/resemble-ai/chatterbox
- Qwen team (2026). Qwen3-TTS. https://github.com/QwenLM/Qwen3-TTS and https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- Mistral AI (2026). Voxtral TTS. https://mistral.ai/news/voxtral-tts/ and https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
- BreezeBlue (2026). Breeze TTS 2. https://huggingface.co/BreezeBlue/Breeze-TTS-2

Cloning, style, singing, and safety:

- Jia, Y. et al. (2018). Transfer learning from speaker verification to multispeaker TTS (SV2TTS). https://arxiv.org/abs/1806.04558
- Skerry-Ryan, R. J. et al. (2018). Towards end-to-end prosody transfer. https://arxiv.org/abs/1803.09047
- Wang, Y. et al. (2018). Style tokens (GST). https://arxiv.org/abs/1803.09017
- Lyth, D. and King, S. (2024). Natural language guidance of high-fidelity TTS (Parler-TTS). https://arxiv.org/abs/2402.01912
- Liu, J. et al. (2022). DiffSinger. https://arxiv.org/abs/2105.02446
- San Roman, R. et al. (2024). AudioSeal. https://arxiv.org/abs/2401.17264

Evaluation and datasets:

- Saeki, T. et al. (2022). UTMOS. https://arxiv.org/abs/2204.02152 ; UTMOSv2: https://github.com/sarulab-speech/UTMOSv2
- Reddy, C. K. A. et al. (2021). DNSMOS. https://arxiv.org/abs/2010.15258 ; DNSMOS P.835: https://arxiv.org/abs/2110.01763
- Kubichek, R. (1993). Mel-cepstral distance measure for objective speech quality assessment. IEEE PACRIM. https://doi.org/10.1109/PACRIM.1993.407206
- ITU-R BS.1534-3 (MUSHRA). https://www.itu.int/rec/R-REC-BS.1534
- ITU-T P.800 (MOS). https://www.itu.int/rec/T-REC-P.800
- Anastassiou, P. et al. (2024). Seed-TTS. https://arxiv.org/abs/2406.02430 ; Seed-TTS-eval: https://github.com/BytedanceSpeech/seed-tts-eval
- Chen, Y. et al. (2024). F5-TTS (evaluation protocol for LibriSpeech-PC and Seed-TTS sets). https://arxiv.org/abs/2410.06885
- Ito, K. and Johnson, L. (2017). The LJ Speech Dataset. https://keithito.com/LJ-Speech-Dataset/
- Veaux, C., Yamagishi, J., and MacDonald, K. (2019). CSTR VCTK Corpus 0.92. https://datashare.ed.ac.uk/handle/10283/3443
- Zen, H. et al. (2019). LibriTTS. https://arxiv.org/abs/1904.02882
- Koizumi, Y. et al. (2023). LibriTTS-R. https://arxiv.org/abs/2305.18802
- Bakhturina, E. et al. (2021). Hi-Fi multi-speaker English TTS dataset. https://arxiv.org/abs/2104.01497
- He, H. et al. (2024). Emilia. https://arxiv.org/abs/2407.05361 ; dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset
- TorchAudio maintenance plan. https://github.com/pytorch/audio/issues/3902
