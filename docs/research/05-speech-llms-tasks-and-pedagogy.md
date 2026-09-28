# Research notes 05: Speech LLMs, remaining audio tasks, and course pedagogy

These notes are source material for lesson writers. They cover two halves:

- **Half A** covers spoken language models (speech LLMs), the textless line of generative
  spoken language modeling, speech enhancement and separation, voice conversion, audio
  classification, anti-spoofing and watermarking, and real-time streaming systems.
- **Half B** covers pedagogy and lab feasibility: existing syllabi, small datasets,
  synthetic data, the Python audio tooling landscape as of September 2026, RunPod
  serverless practicalities, and in-browser audio visualization.

Research date: 2026-09-28. Facts marked **(verified)** were checked against a primary
source on that date. Facts marked **(unverified)** come from memory or secondary sources,
and lesson writers must check them before stating them as fact. Numbers without a mark
come from the cited paper's abstract or main results table as the author recalls them;
treat them as approximate.

Conventions used in the lab ideas:

- **CPU** means the automated check must run in 60-120 seconds on a CPU container with
  NumPy, SciPy, and CPU PyTorch, with no network access.
- **GPU** means a RunPod serverless job (L4 24 GB unless stated, A100 80 GB for 7B+
  models).
- "Check" describes what the auto-grader asserts.

---

## Part A1: Speech and audio LLMs

### A1.1 The adapter pattern

**Intuition.** A pretrained audio encoder (usually Whisper's encoder) turns audio into a
sequence of continuous frame vectors. A small *projector* maps those vectors into the
embedding space of a pretrained text LLM, after reducing the frame rate so the sequence
fits the context window. The LLM then treats the projected frames as "soft tokens"
alongside text tokens. Because the LLM only has to read audio, not produce it, the output
is text. Producing speech needs extra machinery, covered in later sections.

**Pipeline and shapes.** Whisper computes 80 or 128 log-mel bins with a 25 ms window and
10 ms hop (100 frames per second). Two stride-2 convolutions halve this to 50 Hz, so a
30 s window yields 1,500 encoder frames of width $d_{enc}$ (1,280 for Whisper large).

The projector options are as follows:

- **Frame stacking plus MLP.** Concatenate $k$ consecutive frames and project:
  $$z_j = W_2\,\sigma\!\left(W_1\,[h_{jk}; h_{jk+1}; \dots; h_{jk+k-1}] + b_1\right) + b_2,
  \qquad z_j \in \mathbb{R}^{d_{LLM}}$$
  With $k=4$, 50 Hz becomes 12.5 Hz (Voxtral). With $k=5$, it becomes 10 Hz (LLaMA-Omni).
- **Strided 1-D convolution or average pooling.** Qwen2-Audio pools with stride 2, giving
  roughly 40 ms per frame (25 Hz).
- **Q-Former.** A set of $M$ learned queries cross-attends to encoder frames and emits $M$
  vectors. SALMONN uses a *window-level* Q-Former: windows of $L=17$ frames (about
  0.33 s) with one query each, so the output rate is about 3 tokens per second **(verified)**.
- **Cross-attention into the LLM** (Flamingo-style). This avoids lengthening the LLM input
  but changes the LLM's architecture. It's less common in speech LLMs.

**Token-rate arithmetic (good for a quiz).** A 10-minute recording at 12.5 Hz yields
7,500 audio tokens. Voxtral's 32k context holds about 40 minutes of audio at 12.5 Hz
**(verified)**.

**Training recipe (typical).**

1. **Alignment stage.** Freeze the encoder and the LLM. Train only the projector on ASR
   and audio captioning pairs. The loss is next-token cross-entropy on the text target
   only:
   $$\mathcal{L} = -\sum_{i \in \text{target}} \log p_\theta\!\left(y_i \mid
   z_{1:J}, x^{\text{prompt}}, y_{<i}\right)$$
   Mask prompt and audio positions out of the loss.
2. **Instruction tuning.** Unfreeze the LLM partially (LoRA) and train on diverse audio
   instruction data (question answering about sounds, emotion, speaker attributes,
   speech translation).
3. **Optional preference stage.** Qwen2-Audio uses DPO after SFT **(verified)**.

**Common mistakes.**

- Forgetting to mask the loss on audio and prompt positions, which wastes capacity and
  can destabilize training.
- Mismatched normalization. The projector output norm must be comparable to text
  embedding norms. A LayerNorm or RMSNorm at the projector output helps.
- *Task overfitting*. SALMONN reports that instruction tuning on ASR and captioning makes
  the model ignore instructions and always transcribe. SALMONN fixes this with a short
  "activation tuning" stage **(verified)**.
- Assuming Whisper's encoder handles arbitrary lengths. It's trained on 30 s padded
  windows. Longer audio must be chunked, and shorter audio is padded (Whisper) unless the
  model was retrained with variable lengths.

**Lab idea A1-L1 (CPU): "Toy adapter."** Provide precomputed frozen features for a tiny
synthetic task. For example, formant-synthesized vowel sequences (see Part B3) encoded
by a fixed random conv encoder, and a frozen 2-layer character-level LM pretrained on
the vowel names. Students write `stack_frames(h, k)` and a 2-layer MLP projector, splice
projected vectors before the text prompt, and train only the projector for 200 steps.
Checks:

- `stack_frames` output shape is `(T // k, k * d)` and drops or pads the remainder as
  specified.
- The loss mask excludes audio positions (the grader inspects gradients on a probe
  batch).
- Held-out "transcription" accuracy exceeds a threshold after training in under 60 s.

**Lab idea A1-L2 (GPU, L4).** Load Whisper-small's encoder and a 0.5B instruct LLM (for
example, Qwen2.5-0.5B-Instruct). Train a stacking projector on 1-2 hours of LibriSpeech
dev-clean for 10 minutes. Check that WER on 50 held-out utterances is below a loose
threshold (for example, 40%). This shows the adapter works with less than one GPU-hour.

### A1.2 Canonical systems (chronological)

The following entries give architecture, scale, token rates, and training recipe. Each
entry includes the arXiv link.

#### SpeechGPT (Zhang et al., 2023)

- arXiv: https://arxiv.org/abs/2305.11000
- LLaMA-13B with its vocabulary extended by 1,000 discrete units from k-means on mHuBERT
  features (50 Hz, deduplicated).
- Three stages: (1) modality-adaptation pretraining on unit sequences; (2) cross-modal
  instruction fine-tuning on SpeechInstruct; (3) *chain-of-modality* instruction tuning
  with LoRA, where the model emits the transcript, then a text response, then the
  response's units.
- A unit-based HiFi-GAN vocoder produces speech.
- **Interleaving strategy:** sequential ("think in text, then speak"). High latency,
  because the full text response precedes speech.

#### AudioPaLM (Rubenstein et al., 2023)

- arXiv: https://arxiv.org/abs/2306.12925
- Initializes from PaLM-2 (8B in the main experiments) and extends the embedding matrix
  with audio tokens from a USM-derived k-means tokenizer. Audio tokens decode through
  AudioLM-style stages or SoundStorm.
- Tasks: ASR, AST, S2ST, TTS, all as token-to-token.
- Lesson: initializing from a strong text LLM matters more than the audio tokenizer
  details for translation quality.

#### SALMONN (Tang et al., 2023)

- arXiv: https://arxiv.org/abs/2310.13289
- Dual encoders: Whisper (speech) and BEATs (general audio), concatenated frame-wise at
  50 Hz **(verified)**.
- Window-level Q-Former, Vicuna-13B with LoRA rank 8 (about 33M trainable parameters)
  **(verified)**.
- Three stages: pretraining (ASR and captioning), instruction tuning (15 tasks, about
  2.3M samples), and activation tuning (12 steps on story-style samples to undo task
  overfitting) **(verified)**.

#### Qwen-Audio (Chu et al., 2023) and Qwen2-Audio (Chu et al., 2024)

- Qwen-Audio: https://arxiv.org/abs/2311.07919. Whisper-large-v2 encoder plus Qwen-7B,
  trained on 30+ tasks using hierarchical task tags.
- Qwen2-Audio: https://arxiv.org/abs/2407.10759. Whisper-large-v3-initialized encoder,
  16 kHz input, 128-channel mel, Qwen-7B LLM, 8.2B parameters total **(verified)**.
  Pooling of stride 2 gives about 40 ms per frame. Natural-language prompts replace the
  tags; SFT covers "voice chat" and "audio analysis" modes; DPO follows **(verified)**.

#### LLaMA-Omni (Fang et al., 2024)

- arXiv: https://arxiv.org/abs/2409.06666
- Whisper-large-v3 encoder, a speech adaptor that concatenates every $k=5$ frames plus an
  MLP, and Llama-3.1-8B-Instruct.
- A *streaming speech decoder*: a small non-autoregressive Transformer reads the LLM's
  output hidden states, upsamples each by a factor $\lambda$ (25), and predicts discrete
  HuBERT units with a CTC loss. A unit HiFi-GAN vocoder synthesizes each chunk.
- Response latency as low as 226 ms; trained in under 3 days on 4 GPUs on InstructS2S-200K
  **(verified)**.
- **Why CTC here:** the decoder doesn't need an alignment between text tokens and units.
  CTC marginalizes over alignments, so the lesson reuses the CTC loss from the ASR unit.

#### Mini-Omni (Xie and Wu, 2024) and Mini-Omni2

- Mini-Omni: https://arxiv.org/abs/2408.16725. Mini-Omni2: https://arxiv.org/abs/2410.11190
- Qwen2-0.5B backbone with a Whisper-small encoder **(unverified size detail)**.
- Audio output uses the SNAC codec, whose 24 kHz model has three codebook levels at
  different rates. One coarse frame expands to $1 + 2 + 4 = 7$ tokens. The LM has eight
  output heads (text plus seven audio layers) and emits them in parallel with a
  *text-first delay*: text leads, and audio layer $k$ lags by $k$ steps **(verified:
  "text-delay parallel decoding", eight sub-LM heads)**.
- "Batch parallel decoding": run a text-only copy and an audio copy in the same batch,
  and inject the text-only copy's tokens into the audio copy. This transfers the
  backbone's reasoning into speech mode.

#### Moshi and Mimi (Défossez et al., Kyutai, 2024)

- arXiv: https://arxiv.org/abs/2410.00037. Code: https://github.com/kyutai-labs/moshi
- **Helium** text LM: 7B parameters, 2.1T tokens of English text, 32k SentencePiece vocab,
  context 4,096 **(verified)**.
- **Mimi codec** **(verified)**:
  - 24 kHz mono in, 12.5 Hz frames, 8 codebooks of 2,048 entries.
  - Bitrate: $12.5 \times 8 \times \log_2 2048 = 12.5 \times 8 \times 11 = 1100$ bps.
  - *Split RVQ*: one plain VQ for a "semantic" codebook plus a 7-level RVQ for acoustics.
    The semantic level is distilled from WavLM with a cosine-distance loss.
  - Causal Transformer bottlenecks (8 layers, 250-frame context) in the encoder and
    decoder. Trained with adversarial and feature-matching losses only (no reconstruction
    loss), which the paper reports improves perceived quality.
- **RQ-Transformer.** A large *Temporal Transformer* (initialized from Helium) runs once
  per 80 ms frame $s$ and outputs a context vector $z_s$. A small *Depth Transformer*
  (6 layers, width 1,024, 16 heads) runs autoregressively over the $K$ sub-tokens of that
  frame **(verified)**:
  $$p(V_s \mid V_{<s}) = \prod_{k=1}^{K} p\!\left(V_{s,k} \mid z_s, V_{s,<k}\right),
  \qquad z_s = \text{TemporalTransformer}(V_{<s})$$
  The factorization turns $T \times K$ sequential steps into $T$ large steps plus
  $T \times K$ tiny steps.
- **Multi-stream.** Per frame, $K = 17$ tokens: 1 text token, 8 Moshi audio tokens, and
  8 user audio tokens **(verified)**. There are no turn boundaries. Both speakers can talk
  at once, which is what "full duplex" means here.
- **Acoustic delay.** Acoustic codebooks lag the semantic codebook by $\tau = 1$ or 2
  frames **(verified)**, so the model commits to content before acoustic detail.
- **Inner monologue.** Moshi predicts its own text transcript, time-aligned to its audio
  using Whisper word timestamps, with `PAD` and `EPAD` (end-of-padding) tokens filling gaps
  **(verified)**. Shifting text earlier than audio turns the same model into TTS. Shifting
  it later turns it into streaming ASR.
- **Latency:** theoretical 160 ms (80 ms frame plus 80 ms acoustic delay), 200 ms in
  practice **(verified)**.
- **Training stages** **(verified)**: Helium pretraining; audio pretraining on 7M hours
  **(hours unverified)** with a single stream; multi-stream training with diarized data;
  Fisher fine-tuning on real two-channel conversations; instruction tuning on 20k+ hours
  of synthetic dialogue with a consistent voice.

**Lab idea A1-L3 (CPU): "Moshi's token grid."** Students write `build_multistream(text,
sem, acoustic, user, tau)` that lays out a `(T, 17)` integer grid with the text stream, the
semantic stream, acoustic codebooks delayed by `tau`, and the user stream, using
`PAD`/`EPAD`. They also write the inverse. Checks: round-trip identity; correct placement
of delays; the count of `PAD` tokens equals frames without a word onset.

**Lab idea A1-L4 (CPU): "Depth transformer factorization."** Implement a tiny
RQ-Transformer (temporal: 2 layers, width 64; depth: 1 layer, width 32) on synthetic RVQ
codes of a toy 2-codebook codec. Check that the log-likelihood computed via the factored
form equals the brute-force joint on a 3-frame example, and that sampling respects
causality (changing future frames doesn't change earlier logits).

#### GLM-4-Voice (Zeng et al., 2024)

- arXiv: https://arxiv.org/abs/2412.02612
- GLM-4-9B base, continually pretrained to 1T tokens **(verified)**.
- **Tokenizer:** Whisper-large-v3 encoder with a pooling layer and a VQ bottleneck
  inserted mid-encoder, fine-tuned on ASR. 12.5 Hz, single codebook, 175 bps
  **(verified)**. The bitrate implies $175 / 12.5 = 14$ bits, so a 16,384-entry codebook.
- **Synthetic interleaved data:** a text-to-token model converts text pretraining data into
  speech tokens, which yields 455B tokens of interleaved speech-text data. This is the key
  trick for transferring text knowledge **(verified)**.
- **Decoding:** CosyVoice-style decoder (speech token encoder, conditional flow matching,
  HiFi-GAN). It needs at least 10 speech tokens (0.8 s) to start **(verified)**.
- **Interleaving:** "streaming thoughts" alternates 13 text tokens and 26 speech tokens
  **(verified)**. Text stays ahead of speech, so speech is conditioned on text without
  waiting for the full response.

#### Qwen2.5-Omni (Xu et al., 2025)

- arXiv: https://arxiv.org/abs/2503.20215
- **Thinker-Talker.** The *Thinker* is the multimodal LLM (7B). The *Talker* is a
  separate autoregressive decoder that receives the Thinker's hidden states and the
  sampled text token embeddings, and emits speech codec tokens **(verified)**. This
  decouples "what to say" from "how to say it" while keeping one end-to-end model.
- **TMRoPE:** time-aligned multimodal RoPE where one temporal position ID equals 40 ms.
  Audio and video interleave in 2 s chunks **(verified)**.
- **Streaming speech decoding:** a sliding-window block-attention flow-matching DiT
  (limited lookback and lookahead blocks) produces mel, and BigVGAN produces waveform
  **(verified)**.

#### Qwen3-Omni (Qwen team, September 2025)

- arXiv: https://arxiv.org/abs/2509.17765
- **Thinker:** MoE, 30B total with 3B active. **Talker:** MoE, 3B total with 0.3B active
  **(verified)**.
- **AuT audio encoder:** about 0.6B parameters, trained from scratch on 20M hours of
  supervised audio, 12.5 Hz output, block-wise window attention (1-8 s windows)
  **(verified)**.
- **Speech output:** multi-codebook codec at 12.5 Hz. The Talker predicts codebook 0, an
  80M-parameter multi-token prediction (MTP) module predicts the residual codebooks, and a
  200M causal ConvNet ("Code2Wav") replaces diffusion for waveform synthesis
  **(verified)**. First-packet latency: 234 ms theoretical for audio **(verified)**.
- **Recipe:** encoder alignment with a frozen LLM, a general stage of about 2T multimodal
  tokens, then a long-context stage to 32,768 tokens. Thinker post-training uses SFT,
  strong-to-weak distillation, and GSPO; Talker post-training uses DPO and speaker
  fine-tuning **(verified)**.
- Variants: Instruct, Thinking, and Captioner. Speech input in 19 languages; speech output
  in 10 languages **(verified)**.
- **Qwen3.5-Omni** (April 2026, https://arxiv.org/abs/2604.15804): hybrid-attention MoE at
  hundreds of billions of parameters, 256k context, over 100M hours of audio-visual data,
  and an "ARIA" mechanism that aligns text and speech units for stable streaming
  synthesis **(verified from abstract)**. Weights status: **(unverified)**.
- **Qwen-Audio-3.0-Gen-Preview** (July 2026, https://arxiv.org/abs/2607.27011) exists;
  details **(unverified)**.

#### Kimi-Audio (Moonshot AI, April 2025)

- arXiv: https://arxiv.org/abs/2504.18425. Code: https://github.com/MoonshotAI/Kimi-Audio
- Initialized from Qwen2.5-7B **(verified)**.
- **Hybrid input:** discrete 12.5 Hz semantic tokens (GLM-4-Voice's tokenizer) *plus*
  continuous Whisper features downsampled to 12.5 Hz, added to the token embeddings
  **(verified)**.
- **Parallel heads:** shared Transformer layers, then a text head and an audio head that
  generate simultaneously. Text is padded to align with audio **(verified)**.
- **Detokenizer:** chunk-wise streaming flow matching (12.5 Hz tokens to 50 Hz mel) with 4
  tokens of look-ahead, then BigVGAN **(verified)**.
- **Data:** about 13M hours of audio; 585B audio tokens and 585B text tokens in
  pretraining; about 300k hours of SFT data **(verified)**.

#### Step-Audio 2 (StepFun, July 2025)

- arXiv: https://arxiv.org/abs/2507.16632
- Latent audio encoder plus adaptor feeding a text LLM whose vocabulary is extended with
  6.6k audio tokens (CosyVoice 2-style tokenizer). The model outputs interleaved text and
  audio tokens **(verified)**.
- Embedding the new tokens uses 128B text tokens plus 128B audio tokens of training
  **(verified)**. Reasoning-centric RL, RAG, and tool calls (web search, timbre search)
  are part of the system **(verified)**.
- An open "mini" variant exists; size about 8B **(unverified)**.
- Follow-ups: Step-Audio-R1 (https://arxiv.org/abs/2511.15848), an audio reasoning model
  trained with "modality-grounded reasoning distillation", and Step-Audio-R1.5
  (https://arxiv.org/abs/2604.25719) **(verified to exist)**.

#### Voxtral and Voxtral Realtime (Mistral AI)

- Voxtral: https://arxiv.org/abs/2507.13264. Encoder, a 4x downsampling adapter (12.5 Hz),
  and a decoder LM. Voxtral Mini (3B, from Ministral 3B) and Voxtral Small (24B, from
  Mistral Small 3.1). 32k context, about 40 minutes of audio **(verified)**. Apache 2.0.
- Voxtral Realtime: https://arxiv.org/abs/2602.11298, February 2026. A streaming ASR
  model built on the delayed streams modeling framework, with a new causal audio encoder
  and Ada RMS-Norm delay conditioning. At 480 ms delay it matches Whisper quality.
  Pretrained on 13 languages; weights Apache 2.0 (checkpoint
  `Voxtral-Mini-4B-Realtime-2602`) **(verified)**.

#### MiniCPM-o (OpenBMB)

- MiniCPM-o 2.6 (January 2025): about 8B, built from SigLIP, Whisper-medium, ChatTTS, and
  Qwen2.5-7B **(unverified exact composition)**.
- MiniCPM-o 4.5 (arXiv https://arxiv.org/abs/2604.27393, April 2026): 9B, built from
  SigLIP2, Whisper-medium, CosyVoice2, and Qwen3-8B. The "Omni-Flow" framework aligns all
  input and output streams on one time axis for full-duplex behavior. Runs on devices with
  less than 12 GB RAM **(verified)**.

#### Sesame CSM (Sesame, February 2025)

- Blog: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice.
  Code: https://github.com/SesameAILabs/csm
- A Llama-architecture backbone predicts Mimi codebook 0 per frame; a small decoder
  predicts codebooks 1 through $N-1$ (the same split as Moshi's depth transformer).
  Sizes: Tiny 1B+100M, Small 3B+250M, Medium 8B+300M **(verified)**.
- **Compute amortization:** train the decoder on a random 1/16 of frames, while codebook 0
  trains on every frame **(verified)**. About 1M hours of mostly English audio, 2,048
  sequence length (about 2 minutes) **(verified)**.
- CSM-1B is open (Apache 2.0) and needs Llama-3.2-1B access **(verified)**. It generates
  speech only, not text.

#### Kyutai delayed streams modeling (DSM), Unmute, and later work

- DSM paper: https://arxiv.org/abs/2509.08753. Code:
  https://github.com/kyutai-labs/delayed-streams-modeling
- **Idea:** train one decoder-only model on time-aligned text and audio streams, and delay
  the *output* stream relative to the *input* stream by a fixed lag. Delaying text behind
  audio gives streaming ASR. Delaying audio behind text gives streaming TTS
  **(verified)**. The delay is the only difference between the two tasks.
- Kyutai STT transcribes about 400 real-time streams on one H100 **(verified)**. Model
  variants (for example, a 1B English and French model with 0.5 s delay and a 2.6B English
  model with 2.5 s delay) **(unverified)**. Kyutai TTS is 1.6B **(verified from blog
  title)**.
- **Unmute** (https://unmute.sh) wraps a text LLM with Kyutai STT and TTS to make a
  cascaded but low-latency voice agent **(verified)**.
- Later Kyutai releases **(verified from the Kyutai blog)**:
  - Hibiki (February 2025): simultaneous speech-to-speech translation.
  - Hibiki-Zero (February 2026): real-time speech translation across four languages.
  - **Pocket TTS** (January 2026): 100M parameters and runs faster than real time on a
    laptop CPU. A flow-matching LM over continuous 32-dim Mimi latents at 12.5 Hz, with a
    6-layer causal Transformer and an AdaLN MLP flow head **(architecture detail from a
    secondary source; unverified)**. Training code released in August 2026. A
    "drifting" sampler-head variant appeared on 2026-09-28.
  - MoshiRAG (April 2026): knowledge retrieval for speech models.
  - "Voice of Reason" (September 2026): speech-to-speech math models fine-tuned from
    GLM-4-Voice-9B with SFT and RL. Spoken GSM8K rises from 27.3% to 77.1%
    **(secondary source)**.
- Pocket TTS is the most relevant new release for this course: it's a real, open,
  CPU-runnable model whose architecture (a continuous-latent flow head on an AR
  Transformer) ties the codec, AR LM, and flow matching units together.

#### Other 2025-2026 systems worth one line each

- LongCat-Flash-Omni (Meituan, https://arxiv.org/abs/2511.00279): 560B total, 27B active,
  128k context, real-time audio-visual interaction **(verified)**.
- Fun-Audio-Chat (Alibaba Tongyi, https://arxiv.org/abs/2512.20156): multi-stage
  post-training on existing LLMs without large-scale audio-text pretraining
  **(verified)**.
- Covo-Audio (https://arxiv.org/abs/2602.09823) and PersonaPlex (full-duplex with voice
  and role control) **(existence verified; details unverified)**.
- **MiniMind-O** (https://arxiv.org/abs/2605.03937): about 0.1B parameters. A MiniMind
  Thinker, a 4-layer Talker, frozen SenseVoice-Small and SigLIP2 encoders, and 8-layer
  Mimi code output. Code, weights, and data are open **(verified)**. This is the best
  "whole omni model at toy scale" reference for a capstone.
- Full-duplex research in 2026 is active: the ICASSP 2026 HumDial challenge and
  HumDial-FDBench (https://arxiv.org/abs/2604.21406), and a 2026 full-duplex survey
  (https://arxiv.org/abs/2606.19453).
- Surveys: "On the landscape of spoken language models" (TMLR 2025,
  https://arxiv.org/abs/2504.08528) and "Recent advances in speech language models"
  (ACL 2025, https://aclanthology.org/2025.acl-long.682.pdf). The Interspeech 2026 SLM
  tutorial (https://interspeech2026slm.github.io/) follows the first survey.

### A1.3 Interleaving and multi-stream strategies

The following taxonomy is the core conceptual content for a speech-LLM lesson.

| Strategy | How text and speech relate | Examples | Latency | Main risk |
|---|---|---|---|---|
| Sequential (chain of modality) | Full text response, then speech tokens | SpeechGPT | High | Waits for full text |
| Fixed-ratio interleave | Alternate $a$ text tokens and $b$ speech tokens | GLM-4-Voice (13:26) | Medium | Text may run out or lag |
| Parallel heads with text delay | Text and audio emitted in the same step; audio lags | Mini-Omni, Kimi-Audio | Low | Alignment by padding |
| Time-aligned inner monologue | Text aligned to audio time with PAD tokens | Moshi, DSM | Very low | Needs word timestamps |
| Thinker-Talker | Separate speech decoder conditioned on LLM hidden states | Qwen2.5/3-Omni, MiniMind-O | Low | Two models to train |
| Word-level interleaved pretraining | Random switches between text and speech at word boundaries | SpiritLM | N/A (pretraining) | Needs aligned corpora |

**Codebook delay patterns.** With $K$ RVQ codebooks per frame, a model must either
flatten ($T \cdot K$ steps), predict all codebooks in parallel (fast but ignores
dependencies), or use a *delay pattern*. MusicGen (https://arxiv.org/abs/2306.05284)
delays codebook $k$ by $k$ steps, so at step $t$ the model emits
$(c_{t,1}, c_{t-1,2}, \dots, c_{t-K+1,K})$. Generation takes $T + K - 1$ steps, and each
codebook $k$ is conditioned on coarser codebooks of the same frame from earlier steps.
Moshi's acoustic delay and Mini-Omni's text-first delay are variants.

**Lab idea A1-L5 (CPU): "Delay pattern."** Implement `apply_delay(codes, pad_id)` and
`revert_delay(delayed)` for a `(K, T)` array. Check round-trip identity, output shape
`(K, T + K - 1)`, and that `delayed[k, t] == codes[k, t - k]`.

**Lab idea A1-L6 (CPU): "Interleave builder."** Given word-aligned text tokens and
12.5 Hz speech tokens, build (a) GLM-style 13:26 interleaving, (b) SpiritLM-style
word-level random switching with `[TEXT]` and `[SPEECH]` markers, and (c) Moshi-style
aligned text with `PAD`/`EPAD`. Checks on counts and ordering.

### A1.4 Evaluation benchmarks

- **VoiceBench** (https://arxiv.org/abs/2410.17196). Spoken instructions, both synthetic
  and real, for LLM-based voice assistants. Subsets: AlpacaEval, CommonEval, WildVoice,
  SD-QA, MMSU, OpenBookQA, BBH, IFEval, AdvBench, and MTBench **(verified)**. Evaluated
  mostly on text output (a GPT judge for open-ended subsets).
- **AIR-Bench** (https://arxiv.org/abs/2402.07729). Audio instruction following:
  foundation (single-task multiple choice) and chat (open-ended, GPT-4 judged) tracks
  over speech, sound, and music **(unverified details)**.
- **MMAU** (https://arxiv.org/abs/2410.19168). About 10k audio clips with
  human-annotated multiple-choice questions across speech, sound, and music, testing
  expert-level reasoning **(unverified count)**. MMAU-Pro extends it **(unverified)**.
- **Big Bench Audio** (https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio).
  1,000 spoken questions from four Big Bench Hard categories (formal fallacies, navigate,
  object counting, web of lies), 250 each, spoken by 23 synthetic voices. Evaluated in
  speech-to-speech and speech-to-text modes **(verified)**.
- **URO-Bench** (https://arxiv.org/abs/2502.17810). Speech-to-speech benchmark covering
  understanding, reasoning, and oral conversation; basic (16 datasets) and pro (20
  datasets) tracks, including multilingual, multi-round, and paralinguistic tests
  **(verified)**.
- Also: WavBench (https://arxiv.org/abs/2602.12135), Audio MultiChallenge
  (https://arxiv.org/abs/2512.14865), and HumDial-FDBench for full duplex.

**Evaluation pitfalls to teach.** LLM-as-judge variance; ASR-transcribing the model's
speech output before judging (which hides prosody); benchmarks built with TTS voices
(which inflate scores relative to real speech); and data contamination.

**Lab idea A1-L7 (CPU): "Mini benchmark harness."** Given a JSON of model outputs for 40
multiple-choice audio questions (precomputed), compute accuracy per category, a bootstrap
95% CI, and a McNemar test between two systems. Checks on exact values.

---

## Part A2: Generative spoken language modeling with discrete units (textless NLP)

**Intuition.** Language models don't need text. Cluster self-supervised speech features
into discrete "pseudo-phone" units, train a standard LM on unit sequences, and
resynthesize waveform from generated units. This proved speech alone carries enough
structure for a language model to learn lexical and some syntactic regularities.

### GSLM (Lakhotia et al., 2021)

- arXiv: https://arxiv.org/abs/2102.01192
- Three parts:
  1. **Speech-to-unit (S2u).** CPC, wav2vec 2.0, or HuBERT features, then k-means with
     $K \in \{50, 100, 200\}$. For HuBERT at 50 Hz, a frame $x_t$ maps to
     $u_t = \arg\min_k \lVert x_t - \mu_k \rVert_2^2$.
  2. **Unit LM (uLM).** A Transformer LM on *deduplicated* units (runs collapsed:
     `5 5 5 12 12 7` becomes `5 12 7`).
  3. **Unit-to-speech (u2S).** Tacotron 2 on units, then WaveGlow.
- **Metrics:** ABX (unit discriminability), sWUGGY (word vs. nonword preference), sBLIMP
  (grammatical vs. ungrammatical pair preference) from the Zero Resource Speech Challenge
  2021 (https://arxiv.org/abs/2011.11588), plus ASR-based PER on resynthesis and
  "VERT" and perplexity for generation diversity.
- **Finding:** HuBERT units with $K=100$ or $200$ worked best overall.

### pGSLM (Kharitonov et al., 2021)

- arXiv: https://arxiv.org/abs/2109.03264
- Adds prosody: a multi-stream Transformer jointly models units, their durations (since
  deduplication discards duration), and quantized log-F0. A HiFi-GAN vocoder conditions on
  units and F0.

### dGSLM (Nguyen et al., 2022)

- arXiv: https://arxiv.org/abs/2203.16502
- Two-channel dialogue from Fisher, modeled by a dual-tower Transformer with
  cross-attention between channels. Turn-taking, overlaps, and backchannels emerge. A
  direct ancestor of Moshi's multi-stream design.

### TWIST (Hassid et al., 2023)

- arXiv: https://arxiv.org/abs/2305.13009
- Warm-start the unit LM from a pretrained text LM (OPT or LLaMA). Consistently better
  than cold start; scales to 7B **(verified)**. Uses HuBERT units at 25 Hz with 500
  clusters **(unverified)**. Introduces spoken StoryCloze and topic StoryCloze
  **(verified)**.

### SpiritLM (Nguyen et al., 2024)

- arXiv: https://arxiv.org/abs/2402.05755
- Llama 2 7B continued on text, speech units, and word-level interleaved text-speech
  sequences **(verified)**. Base uses HuBERT phonetic units; Expressive adds pitch and style
  tokens **(verified)**. Unit rates and vocabulary sizes: HuBERT about 25 Hz with 501
  units; pitch tokens about 12.5 Hz; style tokens about 1 Hz **(unverified)**.
- Few-shot ASR, TTS, and classification emerge from interleaving **(verified)**.

**Common mistakes (textless).**

- Evaluating LM perplexity across different unit vocabularies. Perplexities aren't
  comparable across $K$ or tokenizers. Use sWUGGY and sBLIMP instead.
- Forgetting deduplication at LM time but not at vocoder time (the vocoder needs
  durations).
- Fitting k-means on too little data or on unnormalized features.

**Lab idea A2-L1 (CPU): "Units from scratch."** Provide 2,000 precomputed 50 Hz feature
frames (for example, MFCC+deltas from a synthetic vowel corpus, or cached HuBERT layer-6
features for 20 LibriSpeech utterances stored as a `float16` `.npy` under 10 MB).
Students implement k-means (k-means++ init, Lloyd iterations), unit assignment, run-length
deduplication, and a bigram unit LM with add-one smoothing. Checks: inertia decreases
monotonically; dedup is correct; the unit LM assigns higher probability to held-out "real"
unit strings than to shuffled ones (a toy sWUGGY).

**Lab idea A2-L2 (GPU, L4): "Textless continuation."** Precompute HuBERT units on
LibriSpeech train-clean-5 (Mini LibriSpeech), train a 20M-parameter unit Transformer for
15 minutes, and continue prompts. Resynthesize with a pretrained unit vocoder (for
example, fairseq's unit HiFi-GAN, if a license-compatible checkpoint is available) or show
units as a token image only. Check that validation unit perplexity falls below a
threshold.

---

## Part A3: Speech enhancement and separation

### A3.1 Signal model and classical enhancement

Additive noise model in the STFT domain:
$$Y(k, t) = S(k, t) + N(k, t)$$

**Spectral subtraction** (Boll, 1979, https://doi.org/10.1109/TASSP.1979.1163209). Estimate
the noise power $\hat{P}_N(k)$ from speech-free frames (for example, the first 0.25 s or
VAD-negative frames), subtract it with over-subtraction factor $\alpha$ and spectral floor
$\beta$, and reuse the noisy phase:
$$|\hat{S}(k,t)|^2 = \max\!\left(|Y(k,t)|^2 - \alpha \hat{P}_N(k),\ \beta\,|Y(k,t)|^2\right),
\qquad \hat{S}(k,t) = |\hat{S}(k,t)|\, e^{j \angle Y(k,t)}$$
The floor prevents negative power. Without it, isolated spectral peaks survive and cause
*musical noise*.

**Wiener filter (exact gain).** For uncorrelated zero-mean $S$ and $N$, the linear
MMSE estimate $\hat{S} = H Y$ minimizes $\mathbb{E}|S - HY|^2$. Setting the derivative to
zero gives
$$H(k) = \frac{\mathbb{E}[S Y^*]}{\mathbb{E}[|Y|^2]} = \frac{P_S(k)}{P_S(k) + P_N(k)}
= \frac{\xi(k)}{1 + \xi(k)}, \qquad \xi(k) = \frac{P_S(k)}{P_N(k)}$$
where $\xi$ is the *a priori SNR*. The *a posteriori SNR* is
$\gamma(k,t) = |Y(k,t)|^2 / P_N(k)$. The practical estimator is the decision-directed
approach (Ephraim and Malah, 1984, https://doi.org/10.1109/TASSP.1984.1164453):
$$\hat{\xi}(k,t) = a\,\frac{|\hat{S}(k,t-1)|^2}{\hat{P}_N(k)} + (1-a)\,\max\!\left(\gamma(k,t)
- 1, 0\right), \qquad a \approx 0.98$$

**Ideal masks** (oracle targets for supervised enhancement; Wang et al., 2014,
https://doi.org/10.1109/TASLP.2014.2352935):

- Ideal binary mask: $\text{IBM}(k,t) = \mathbb{1}\left[10\log_{10}\frac{|S|^2}{|N|^2} > \text{LC}\right]$,
  with local criterion LC often 0 dB or -5 dB.
- Ideal ratio mask: $\text{IRM}(k,t) = \left(\frac{|S|^2}{|S|^2 + |N|^2}\right)^{\beta}$, with
  $\beta = 0.5$ typical. With $\beta=1$ it equals the Wiener gain using instantaneous
  powers.
- Phase-sensitive mask: $\text{PSM} = \frac{|S|}{|Y|}\cos(\angle S - \angle Y)$, truncated to
  $[0, 1]$.
- Complex ideal ratio mask: $M = S / Y$ (complex division), usually compressed with tanh.

**Common mistakes.** Computing masks on dB spectrograms; forgetting that the IRM uses
powers; resynthesizing with the clean phase and then claiming the mask alone achieves that
quality; and noise estimation that includes speech frames.

**Lab idea A3-L1 (CPU): "Classical enhancement bake-off."** Using synthetic signals (a
formant-synthesized sentence plus white or pink noise at 0 dB), implement spectral
subtraction, the Wiener gain with decision-directed $\xi$, and the IBM and IRM oracles.
Checks: `wiener_gain(xi)` matches $\xi/(1+\xi)$ exactly; oracle IRM output improves SI-SDR
by more than 8 dB; decision-directed Wiener improves SI-SDR by more than 2 dB; the floor
prevents negative values.

### A3.2 Scale-invariant SDR (SI-SDR)

Le Roux et al., 2019, "SDR - half-baked or well done?" (https://arxiv.org/abs/1811.02508).
First remove the mean from both the estimate $\hat{s}$ and reference $s$. Then project the
estimate onto the reference:
$$s_{\text{target}} = \frac{\langle \hat{s}, s \rangle}{\lVert s \rVert^2}\, s, \qquad
e = \hat{s} - s_{\text{target}}, \qquad
\text{SI-SDR}(\hat{s}, s) = 10 \log_{10} \frac{\lVert s_{\text{target}} \rVert^2}{\lVert e
\rVert^2}$$
Equivalently, with $\alpha^* = \langle \hat{s}, s\rangle / \lVert s\rVert^2$:
$\text{SI-SDR} = 10\log_{10}\frac{\lVert \alpha^* s\rVert^2}{\lVert \alpha^* s - \hat{s}\rVert^2}$.

Properties to check: invariant to rescaling $\hat{s}$ by any $c > 0$ (a negative $c$
leaves the value unchanged too, because $\alpha^*$ flips sign); *not* symmetric in its
arguments; SI-SDR improvement (SI-SDRi) is SI-SDR of the estimate minus SI-SDR of the
mixture. SI-SNR in Conv-TasNet is the same quantity.

**Common mistakes.** Swapping arguments (projecting the reference onto the estimate);
forgetting zero-mean; no epsilon in the log (NaN on silent references); using $+$SI-SDR as
a loss instead of $-$SI-SDR; averaging SI-SDR in the linear domain.

### A3.3 Permutation-invariant training (PIT)

Yu et al., 2017 (https://arxiv.org/abs/1607.00325); utterance-level uPIT, Kolbæk et al.,
2017 (https://arxiv.org/abs/1703.06284). A separator with $C$ outputs has no natural order.
The loss takes the minimum over all permutations $\pi \in \mathcal{S}_C$:
$$\mathcal{L}_{\text{uPIT}} = \min_{\pi \in \mathcal{S}_C} \frac{1}{C}\sum_{c=1}^{C}
\ell\!\left(\hat{s}_c, s_{\pi(c)}\right), \qquad \ell(\hat{s}, s) = -\text{SI-SDR}(\hat{s}, s)$$
Compute a $C \times C$ pairwise loss matrix $L_{ij} = \ell(\hat{s}_i, s_j)$, then minimize
over permutations. Brute force costs $O(C!)$; for $C > 4$ use the Hungarian algorithm on $L$
(`scipy.optimize.linear_sum_assignment`), which gives the same minimum for sum-decomposable
losses. Frame-level PIT (per-frame permutations) causes speaker swaps across frames;
utterance-level PIT fixes the permutation for the whole utterance.

**Lab idea A3-L2 (CPU): "PIT SI-SDR loss."** Implement `si_sdr(est, ref)` (batched,
PyTorch, differentiable) and `pit_loss(est, ref)` for `(B, C, T)` tensors, returning the
loss and the best permutation. Checks:

- Scale invariance: `si_sdr(3 * est, ref) == si_sdr(est, ref)`.
- Permutation invariance: shuffling `ref` channels doesn't change the loss, and the
  returned permutation undoes the shuffle.
- Matches a brute-force reference for $C = 3$ and Hungarian for $C = 5$.
- Gradients flow (`loss.backward()` gives finite gradients).
- Known value: `si_sdr(ref + 0.1 * noise_orthogonal, ref)` equals the closed form.

### A3.4 Conv-TasNet

Luo and Mesgarani, 2019 (https://arxiv.org/abs/1809.07454).

- **Encoder:** a learned 1-D conv filterbank of $N$ filters, length $L$, stride $L/2$,
  followed by ReLU: $w = \text{ReLU}(x \circledast U)$, $w \in \mathbb{R}^{N \times T'}$.
  Best config $N = 512$, $L = 16$ samples (2 ms at 8 kHz).
- **Separator (TCN):** LayerNorm, then a 1x1 conv bottleneck to $B = 128$ channels. Then
  $R = 3$ repeats of $X = 8$ 1-D conv blocks with dilations $1, 2, 4, \dots, 128$. Each
  block: 1x1 conv to $H = 512$, PReLU, normalization, depthwise conv (kernel $P = 3$,
  dilation $d$), PReLU, normalization, then a 1x1 conv to a residual path ($B$) and a skip
  path ($S_c$). Sum skips, apply PReLU and a 1x1 conv to $C \cdot N$, then sigmoid to get
  masks $m_c \in [0,1]^{N \times T'}$.
- **Decoder:** transposed conv (overlap-add of learned basis): $\hat{s}_c = \text{Decoder}(w \odot m_c)$.
- **Normalization:** global LayerNorm (gLN) for non-causal; cumulative LayerNorm (cLN) for
  causal.
- **Receptive field** in encoder frames: $1 + R \cdot (P-1) \sum_{i=0}^{X-1} 2^i =
  1 + 3 \cdot 2 \cdot 255 = 1531$ frames, which is about 1.53 s at 8 kHz with stride 8.
- **Result:** 15.3 dB SI-SNRi on WSJ0-2mix **(unverified exact number)**.
- **Training:** uPIT with negative SI-SNR, 4 s segments, 8 kHz.

**Common mistakes.** Mismatched encoder and decoder stride and padding (length errors at
the end); not trimming the output to the input length; using BatchNorm with small batches;
a causal model that uses gLN (which leaks future statistics).

**Lab idea A3-L3 (CPU): "Tiny Conv-TasNet."** Students implement the encoder, one TCN
block (with dilation and causal padding option), the mask head, and the decoder. The
grader checks: output length equals input length for arbitrary lengths; the receptive
field function matches the formula; causal mode doesn't change past outputs when future
input changes. Training: separate two synthetic sources (a chirp and a formant vowel
train, 8 kHz, 1 s) for 300 steps on CPU; check SI-SDRi above 6 dB.

### A3.5 Beyond Conv-TasNet

- **DPRNN** (Luo et al., 2020, https://arxiv.org/abs/1910.06379). Split the long encoder
  sequence into overlapping chunks of length $K$ (50% hop), making a 3-D tensor
  (features x chunk length x number of chunks). Alternate an intra-chunk BiLSTM (local)
  and an inter-chunk BiLSTM (global). With $K \approx \sqrt{2T'}$, each RNN sees
  $O(\sqrt{T'})$ steps. Enables very short windows (2 samples) and reported 18.8 dB
  SI-SNRi **(unverified exact number)**.
- **SepFormer** (Subakan et al., 2021, https://arxiv.org/abs/2010.13154). DPRNN's
  dual-path structure with Transformer blocks; 22.3 dB SI-SNRi on WSJ0-2mix with dynamic
  mixing **(unverified exact number)**.
- **TF-GridNet** (Wang et al., 2023, https://arxiv.org/abs/2211.12433). Time-frequency
  domain complex spectral mapping (stack real and imaginary parts), with blocks of an
  intra-frame full-band module, a sub-band temporal module, and cross-frame self-attention.
  23.5 dB SI-SDRi on WSJ0-2mix **(verified)**.
- **Demucs** for music (Défossez et al., 2019, https://arxiv.org/abs/1911.13254): a
  waveform U-Net with a BiLSTM bottleneck. **Hybrid Demucs** adds a parallel spectrogram
  branch. **HT Demucs** (Rouard et al., 2022, https://arxiv.org/abs/2211.08553) replaces
  the innermost layers with a cross-domain Transformer (self-attention within each domain,
  cross-attention across). 9.20 dB SDR on MUSDB18-HQ with sparse attention and per-source
  fine-tuning, trained with 800 extra songs **(verified)**.
  torchaudio still ships a Hybrid Demucs tutorial and pipeline (see Part B4).
- **Demucs for speech denoising** (Défossez et al., 2020, https://arxiv.org/abs/2006.12847):
  causal waveform U-Net that runs in real time on CPU.
- **DeepFilterNet** (Schröter et al., 2022, https://arxiv.org/abs/2110.05588; v2
  https://arxiv.org/abs/2205.05474; v3 https://arxiv.org/abs/2305.08227). Two stages at
  48 kHz: (1) predict real gains on about 32 ERB bands for the spectral envelope; (2)
  *deep filtering* on low frequencies (up to about 5 kHz), predicting a complex FIR filter
  per bin over the last $N$ frames:
  $$\hat{S}(k,t) = \sum_{i=0}^{N-1} C_{k,i,t}\; X(k, t - i + l)$$
  where $l$ is a small lookahead. Deep filtering recovers periodic harmonics that a
  single-frame mask can't, because the STFT resolution is too coarse for pitch
  harmonics at 20 ms windows **(band counts unverified)**.
- **RNNoise** (Valin, 2018, https://arxiv.org/abs/1709.08243). A hybrid DSP and deep
  learning design: 20 ms windows with 10 ms hop, 22 Bark-like bands, 42 hand-designed input
  features (band energies as cepstrum, derivatives, and pitch features), three GRU layers
  predicting 22 band gains plus a VAD output, and a pitch comb filter for harmonics. It's
  small enough for embedded real time **(feature counts from memory; unverified)**. The
  lesson point: a network predicting a few band gains is enough for useful denoising.

**Lab idea A3-L4 (CPU): "Band-gain denoiser (RNNoise-lite)."** Compute 22 triangular band
energies, train a 1-layer GRU (hidden 48) to predict band gains (IRM targets per band) on
synthetic speech-like signals plus noise, and interpolate band gains to bins. Check that
SI-SDR improves by more than 3 dB on held-out mixtures and the model has fewer than 20k
parameters.

**Lab idea A3-L5 (GPU, L4): "Pretrained separation and music demixing."** Run torchaudio's
Hybrid Demucs bundle (or `demucs` package) on a 30 s Creative Commons clip, compute SDR
against provided stems, and return the stems as base64 WAV for browser playback. Also runs
on CPU in about 1-2 minutes for 30 s, so a CPU fallback exists **(runtime unverified)**.

### A3.6 Augmentation: noise, reverb, and room simulation

**Mixing at a target SNR.** For clean $s$ and noise $n$ (same length):
$$y = s + g\, n, \qquad g = \sqrt{\frac{P_s}{P_n \cdot 10^{\text{SNR}/10}}},
\qquad P_x = \frac{1}{T}\sum_t x_t^2$$
Common mistake: computing power over the padded or silent regions, or using amplitude
ratios with $10^{\text{SNR}/20}$ inconsistently.

**Reverb.** Convolve with a room impulse response: $y = h * s$. Use FFT convolution
(`scipy.signal.fftconvolve`) and then trim to the original length, keeping the direct-path
delay consistent with the target (a common source of label misalignment).

**Image source method** (Allen and Berkley, 1979, https://doi.org/10.1121/1.382599). In a
shoebox room with walls reflecting with coefficient $\beta$, reflections equal free-field
propagation from mirrored "image" sources. The RIR is a sum of delayed, attenuated
impulses:
$$h(t) = \sum_{i} \frac{\beta^{\,n_i}}{4\pi d_i}\, \delta\!\left(t - \frac{d_i}{c}\right)$$
where $d_i$ is the distance from image $i$ to the microphone, $n_i$ is the number of
reflections, and $c \approx 343$ m/s. Fractional delays are rendered with a windowed sinc.
**Sabine's formula** relates absorption to reverberation time:
$$RT_{60} = \frac{0.161\, V}{\sum_j S_j \alpha_j}$$
with $V$ in m³ and surface areas $S_j$ in m². pyroomacoustics (MIT license, v0.10.1, May
2026, wheels for Python 3.9-3.14) **(verified)** exposes `pra.ShoeBox`,
`pra.inverse_sabine(rt60, room_dim)` to get absorption and max order, and C++ image-source
and ray-tracing backends. Paper: https://arxiv.org/abs/1710.04196.

**Datasets.** MUSAN (music, speech, and noise; 11 GB; CC BY 4.0; OpenSLR 17)
**(verified)** and RIRS_NOISES (real and simulated RIRs plus point-source noises; 1.3 GB;
Apache 2.0; 16 kHz; OpenSLR 28) **(verified)**. Neither fits comfortably in a small Docker
image; ship a curated 20-50 MB subset or simulate.

**Lab idea A3-L6 (CPU): "Image-method RIR from scratch."** Implement a 2-D or 3-D
shoebox image-source RIR up to order 6 with fractional-delay sinc. Checks: the direct-path
peak lands at sample `round(fs * d / c)`; energy decay slope gives an RT60 within 20% of
Sabine's prediction (via Schroeder backward integration,
$E(t) = \int_t^\infty h^2(\tau)\,d\tau$); results match pyroomacoustics within tolerance
when it's installed (optional check).

---

## Part A4: Voice conversion

**Task.** Change *who* is speaking while keeping *what* is said (and ideally prosody).
Equivalently, disentangle content from speaker identity.

### AutoVC (Qian et al., 2019)

- arXiv: https://arxiv.org/abs/1905.05879
- A content encoder $E_c$ with a carefully sized bottleneck (reduced channels and temporal
  downsampling), a pretrained speaker encoder $E_s$ (d-vector), and a decoder $D$. Train
  only on self-reconstruction:
  $$\hat{X} = D\!\left(E_c(X), E_s(X')\right), \qquad
  \mathcal{L} = \lVert \hat{X} - X \rVert_2^2 + \mu \lVert \hat{X}_0 - X \rVert_2^2 +
  \lambda \lVert E_c(\hat{X}) - E_c(X) \rVert_1$$
  where $X'$ is another utterance of the same speaker and $\hat{X}_0$ is the output before
  the postnet.
- **Key argument:** if the bottleneck is just narrow enough, the decoder can't recover
  speaker information from $E_c$ and must use $E_s$. Too wide leaks speaker identity;
  too narrow loses content. At conversion time, swap in the target's $E_s$.

### kNN-VC (Baas et al., 2023)

- arXiv: https://arxiv.org/abs/2305.18975. Code: https://github.com/bshall/knn-vc
- **Method** **(verified at high level)**:
  1. Extract WavLM-Large features (layer 6 in the paper **(unverified layer)**) at 50 Hz
     for the source utterance and for all of the target speaker's reference audio (the
     *matching set*).
  2. Replace each source frame $x_t$ with the mean of its $k$ nearest matching-set frames
     under cosine distance:
     $$y_t = \frac{1}{k}\sum_{r \in \text{kNN}_k(x_t)} r, \qquad
     d(x, r) = 1 - \frac{x \cdot r}{\lVert x\rVert \lVert r\rVert}$$
     The paper uses $k = 4$ **(unverified)**.
  3. Vocode $y$ with a HiFi-GAN trained on WavLM features. A *prematched* vocoder, trained
     on kNN-regressed features of the training set, is more robust to kNN artifacts.
- Why it works: early-middle WavLM layers encode phonetic content linearly, and nearby
  frames of the same phone from the target speaker carry the target's timbre.
- Quality improves with more reference audio (a few minutes work well; below about 30 s
  intelligibility drops) **(unverified thresholds)**.

**Lab idea A4-L1 (CPU, precomputed): "kNN-VC core."** Ship precomputed WavLM layer-6
features (`float16`) for 3 LibriSpeech speakers (about 2 minutes each, about 6k frames x
1,024 dims, about 12 MB each). Students implement batched cosine kNN regression with
NumPy (matrix multiply of L2-normalized features, `argpartition` for top-k). Checks:
matches a reference implementation; $k=1$ output frames are exact copies of matching-set
frames; a linear speaker-ID probe (provided) classifies converted frames as the target
speaker more than 80% of the time; a phone probe keeps content accuracy above a
threshold.

**Lab idea A4-L2 (GPU, L4): "Full kNN-VC."** Load WavLM-Large and the prematched HiFi-GAN
through `torch.hub` (the kNN-VC repo exposes hub entry points **(unverified)**), convert
student-uploaded audio (with consent checkbox), and return base64 WAV. Grade with a
speaker-verification cosine (for example, an ECAPA model) between output and target above a
threshold, and ASR WER of output below a threshold.

### RVC and Seed-VC

- **RVC** (Retrieval-based Voice Conversion, https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI):
  content features from HuBERT or ContentVec; a faiss index over the target's training
  features lets inference blend each frame with its retrieved neighbors (an index ratio,
  conceptually the same as kNN-VC); an F0 extractor (RMVPE is common); and a VITS-style
  decoder with an NSF-HiFi-GAN vocoder. It's popular for singing and requires per-voice
  training **(architecture details from the repo; unverified specifics)**.
- **Seed-VC** (Liu, 2024, https://arxiv.org/abs/2411.09943): zero-shot VC with a diffusion
  Transformer that reads the full reference as in-context prompt. An *external timbre
  shifter* perturbs the source's timbre during training so the model can't copy it,
  reducing leakage and matching the train and inference conditions. F0 conditioning
  extends it to singing **(verified)**.

### Ethics and consent (must be in the lesson)

- Voice is biometric data in many jurisdictions (for example, the GDPR treats biometric
  data for identification as a special category; Illinois BIPA). Cloning a voice without
  consent can enable fraud and impersonation.
- Practical course policy: labs convert only (a) the student's own voice with explicit
  opt-in, or (b) public corpora whose license allows it (LibriSpeech, VCTK, both CC BY 4.0).
  Don't allow uploads of third-party voices. Watermark outputs (see AudioSeal in Part A6)
  and show a disclosure label.
- Discuss detection (anti-spoofing) and provenance (watermarking, C2PA) as the counterpart
  to generation.

---

## Part A5: Audio classification and representation learning

### A5.1 Datasets and metrics

- **ESC-50**: 2,000 clips of 5 s, 50 classes in 5 groups, 44.1 kHz mono, 5 folds, about
  600 MB, CC BY-NC 3.0 **(verified)**. Report 5-fold cross-validation accuracy; never mix
  folds (clips from the same source recording share a fold).
- **UrbanSound8K**: 8,732 clips of up to 4 s, 10 classes, 10 predefined folds, about 6 GB,
  CC BY-NC 3.0 **(unverified size)**. Using random splits instead of the folds inflates
  accuracy.
- **AudioSet**: about 2M 10 s YouTube clips, 527 labels, weakly labeled and multi-label
  (https://research.google.com/audioset/). Distributed as YouTube IDs and embeddings, not
  audio, so it isn't lab-friendly. Metric: mean average precision (mAP) over classes.
- **Speech Commands v2**: 105,829 one-second 16 kHz utterances of 35 words, CC BY 4.0,
  2.37 GiB download **(verified)**. Standard 12-class task: 10 command words plus
  "unknown" and "silence".

### A5.2 Models

- **PANNs** (Kong et al., 2020, https://arxiv.org/abs/1912.10211). CNNs trained on full
  AudioSet with log-mel input at 32 kHz. CNN14 reaches 0.431 mAP **(unverified)**.
  Wavegram-Logmel variants learn a front end.
- **AST** (Gong et al., 2021, https://arxiv.org/abs/2104.01778). A pure Transformer on
  spectrogram patches: 128 mel bins, 25 ms window, 10 ms hop; 16x16 patches with 6-pixel
  overlap (stride 10) in both axes, so a 10 s clip (1,024 frames) gives
  $12 \times 101 = 1212$ patches **(verified numbers from the paper as recalled)**.
  Initialized from ImageNet DeiT with positional-embedding interpolation. Results: 0.485
  mAP AudioSet, 95.6% ESC-50, 98.1% Speech Commands V2 **(verified)**.
  Patch count formula (lab check):
  $$N_f = \left\lfloor \frac{F - 16}{s} \right\rfloor + 1, \quad
  N_t = \left\lfloor \frac{T - 16}{s} \right\rfloor + 1$$
- **Audio-MAE** (Huang et al., 2022, https://arxiv.org/abs/2207.06405). Mask 80% of
  spectrogram patches, encode only visible patches, reconstruct masked patches with a
  decoder using local attention. About 47.3 mAP **(unverified)**.
- **BEATs** (Chen et al., 2022, https://arxiv.org/abs/2212.09058). Iterative
  self-supervised pretraining: an acoustic tokenizer produces discrete labels for masked
  patch prediction, and each iteration distills a better tokenizer from the last model.
  About 48.6 mAP (AS-2M, single model) **(unverified)**.
- **CLAP** (Elizalde et al., 2022, https://arxiv.org/abs/2206.04769; LAION-CLAP, Wu et
  al., 2022, https://arxiv.org/abs/2211.06687). Contrastive audio-text pretraining. For a
  batch of $N$ pairs with L2-normalized embeddings $a_i$ and $t_i$ and learned temperature
  $\tau$:
  $$S_{ij} = \frac{a_i^\top t_j}{\tau}, \qquad
  \mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}\left[-\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}}
  - \log\frac{e^{S_{ii}}}{\sum_j e^{S_{ji}}}\right]$$
  Zero-shot classification: embed prompts such as "the sound of a dog barking" and take
  the argmax cosine.
  Common mistakes: forgetting to normalize, using one direction only, fixing $\tau$ too
  high, and letting $\log \tau$ go unbounded (clamp it).
- **Keyword spotting.** Small CNNs on 40-dim log-mel or MFCC (Sainath and Parada, 2015;
  "Hello Edge," https://arxiv.org/abs/1711.07128). **BC-ResNet** (Kim et al., 2021,
  https://arxiv.org/abs/2106.04140) uses *broadcasted residual learning*: most residual
  functions are 1-D temporal convolutions on frequency-averaged features, broadcast back
  to 2-D. 98.0% (v1) and 98.7% (v2) accuracy with small parameter counts **(verified)**.

**Augmentation and training tricks.** SpecAugment (time and frequency masking,
https://arxiv.org/abs/1904.08779), mixup (for multi-label, mix labels too), random time
shift, background noise from the dataset's `_background_noise_` folder, and balanced
sampling for AudioSet.

**Lab idea A5-L1 (CPU): "Keyword spotting on synthetic or small data."** Two options:

- Offline: synthetic 10-"word" dataset of DTMF-like tone patterns or formant vowel
  sequences (see Part B3), with noise.
- Download at image build: a 10-word subset of Speech Commands v2 test set as
  precomputed 40x98 log-mel `float16` arrays (about 4,000 clips, about 30 MB).

Students implement log-mel (from the earlier unit), a 4-layer CNN or a BC-ResNet-1 block
(with SubSpectral Norm optional), and train for 60 s. Check accuracy above threshold and
parameter count below 30k.

**Lab idea A5-L2 (CPU): "AST patch embedding."** Implement patchify with overlap,
positional embedding interpolation (bilinear from a 14x14 grid to 12x101), and a
[CLS]-token classifier. Check patch counts against the formula, and that a 1-layer model
learns a synthetic tone-vs-noise task.

**Lab idea A5-L3 (CPU): "CLAP loss."** Implement symmetric InfoNCE with a learnable
temperature. Check against a reference value on fixed tensors, invariance to embedding
scale after normalization, and that the loss equals $\log N$ when all embeddings are
identical.

**Lab idea A5-L4 (GPU, L4): "Zero-shot ESC-50 with CLAP."** Evaluate a LAION-CLAP
checkpoint on ESC-50 fold 1 with student-written prompts; check accuracy above 80%
**(threshold unverified; published zero-shot ESC-50 is around 90%)**.

---

## Part A6: Anti-spoofing, deepfake detection, and watermarking (brief)

- **ASVspoof challenges** (https://www.asvspoof.org/). 2019 introduced logical access (LA:
  TTS and VC attacks) and physical access (PA: replay). 2021 added a deepfake (DF) track
  with codecs. ASVspoof 5 (2024) moved to crowdsourced data and adversarial attacks
  **(unverified details)**.
- **Metrics.** Equal error rate (EER): the threshold where false acceptance rate equals
  false rejection rate. Tandem detection cost function (t-DCF) combines the spoofing
  countermeasure with the speaker verification system.
- **AASIST** (Jung et al., 2022, https://arxiv.org/abs/2110.01200). Raw-waveform sinc
  front end, then a heterogeneous stacking graph attention layer across spectral and
  temporal graphs, max graph operation, and extended readout. 20% relative improvement
  over the prior state of the art on ASVspoof 2019 LA; AASIST-L has only 85k parameters
  **(verified)**. EER about 0.83% and min t-DCF about 0.0275 **(unverified)**.
- **Generalization is the open problem.** Detectors overfit to the attacks and codecs seen
  in training.
- **AudioSeal** (San Roman et al., 2024, https://arxiv.org/abs/2401.17264). A generator adds
  an imperceptible additive watermark; a detector outputs a per-sample probability that the
  watermark is present, so it can localize watermarked segments in edited audio. It uses a
  perceptual loss inspired by auditory masking, and detection is up to two orders of
  magnitude faster than prior methods **(verified)**. It can also embed a 16-bit message
  **(unverified)**.

**Lab idea A6-L1 (CPU): "EER and a toy watermark."** Implement EER from scores (ROC sweep
with interpolation) and check against known values. Then implement a spread-spectrum
watermark (add a keyed pseudorandom sequence at -30 dB relative to the signal; detect by
normalized correlation), and check detection AUC above 0.99 at 0 dB added noise and that
SI-SDR of the watermarked signal exceeds 25 dB.

---

## Part A7: Real-time and streaming systems

### A7.1 Latency budget

Human turn-taking gaps average about 200 ms across languages (Stivers et al., 2009,
https://doi.org/10.1073/pnas.0903616106). Moshi cites 230 ms as the natural-conversation
average **(verified from Moshi summary)**. A voice agent's response latency decomposes as
follows:
$$T_{\text{resp}} = T_{\text{endpoint}} + T_{\text{ASR final}} + T_{\text{LLM TTFT}} +
T_{\text{TTS first chunk}} + T_{\text{network}} + T_{\text{playback buffer}}$$
For streaming models, *algorithmic latency* is the lookahead the model needs (window size
plus future context plus decoding delay). Examples:

- STFT with 25 ms window: at least 25 ms algorithmic latency.
- Moshi: 80 ms frame plus 80 ms acoustic delay = 160 ms.
- DSM ASR: the text delay (0.5-2.5 s) is the latency.
- GLM-4-Voice: at least 10 speech tokens (0.8 s) before decoding starts.

### A7.2 Causal and streaming building blocks

- **Causal convolution.** Pad only on the left by $(K-1)\,d$ for kernel $K$ and dilation
  $d$. Test: output at time $t$ must not change if input after $t$ changes.
- **Streaming convolution with state.** Keep the last $(K-1)\,d$ input samples as a buffer
  between chunks. Chunked output must equal full-sequence output exactly (a great
  auto-check).
- **Chunked attention.** Attention mask allows full attention within a chunk and limited
  left context (for example, Emformer https://arxiv.org/abs/2010.10759; the
  cache-aware streaming FastConformer in NeMo). The KV cache holds left context.
- **Causal normalization.** Cumulative LayerNorm or per-frame LayerNorm, not global
  statistics.
- **Overlap-add output.** Streaming vocoders and codecs emit chunks that overlap-add, and
  window-sum normalization must hold (COLA condition).

### A7.3 Endpointing and turn-taking

- **Energy VAD with hangover.** Frame energy in dB above an adaptive noise floor, with
  hangover (keep "speech" state for $H$ frames after energy drops) to avoid cutting word
  endings, and minimum speech duration.
- **Neural VAD.** Silero VAD (MIT license, https://github.com/snakers4/silero-vad) is the
  common default.
- **Semantic endpointing.** Pipecat Smart Turn v3 (BSD-2-Clause, open weights and data,
  23 languages, about 12 ms CPU inference) classifies whether the speaker finished, from
  raw audio rather than transcripts **(verified)**
  (https://github.com/pipecat-ai/smart-turn).
- **Full-duplex behaviors.** Barge-in (user interrupts; the system must stop TTS and flush
  buffers), backchannels ("mm-hm") that shouldn't trigger a turn, and echo cancellation so
  the system doesn't hear itself. Cascaded systems need explicit state machines; Moshi-style
  models learn these behaviors from two-channel data.

### A7.4 On-device and quantization

- Dynamic int8 quantization of Linear and LSTM layers
  (`torch.ao.quantization.quantize_dynamic`) is the easiest CPU speedup
  **(API status in torch 2.14 unverified; torch.ao quantization has been migrating to
  torchao)**.
- Weight-only int8 and int4 (torchao, GGML/GGUF in whisper.cpp and llama.cpp), and ONNX
  Runtime for deployment.
- Real-time factor (RTF) = processing time / audio duration. RTF < 1 is necessary but not
  sufficient; chunk latency matters.

**Lab idea A7-L1 (CPU): "Streaming equivalence."** Implement a stateful causal conv stack
and a streaming log-mel front end that processes 20 ms chunks. Check that chunked output
equals offline output to 1e-6, and measure RTF.

**Lab idea A7-L2 (CPU): "Endpointing state machine."** Implement an energy VAD with
hysteresis and hangover, and a turn-taking state machine (LISTENING, USER_SPEAKING,
ENDPOINT_PENDING, AGENT_SPEAKING, BARGE_IN). Check event timestamps against labeled
synthetic sessions within 50 ms tolerance.

**Lab idea A7-L3 (CPU): "Latency budget calculator."** A pure-function lab: given
component latencies and chunk sizes, compute time-to-first-audio and check against
expected values; include the Moshi 160 ms decomposition.

---

## Part B1: Existing course syllabi

Each entry summarizes structure and what to borrow.

### Stanford CS224S, Spoken Language Processing (Spring 2025)

- URL: https://web.stanford.edu/class/cs224s/ and syllabus
  https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus **(verified)**.
  The Spring 2025 offering is the most recent one found; no 2026 syllabus was located.
- Instructors: Andrew Maas and Tolúlọpẹ́ Ògúnrẹ̀mí.
- Order: introduction; phonetics; TTS (history, front end, concatenative and parametric,
  then deep TTS with Tacotron, WaveNet, FastSpeech, CosyVoice 2, and VITS2); spoken dialog
  and system design (frames, GUS, industry case studies); ASR (noisy channel, WER, HMMs,
  HMM-DNN, CTC, LAS, then Conformer and Whisper); paralinguistics and ethics;
  self-supervised foundation models (wav2vec 2.0, HuBERT); multilingual and low-resource;
  LLM-based dialog; full-duplex LLM dialog in the final lecture.
- Four homeworks: audio tools and phonetic transcription; speech tools and synthesis;
  neural ASR on conversational speech; fine-tuning ASR base models and non-English speech
  (or a project).
- Borrow: TTS before ASR is unusual and works for motivation; dialog-systems framing; an
  ethics lecture embedded mid-course.

### CMU 11-751 / 18-781, Speech Recognition and Understanding

- URL (Fall 2022, Shinji Watanabe): https://www.wavlab.org/activities/2022/11751-2022f/
  **(verified)**. Course hub: https://sites.google.com/a/is.cs.cmu.edu/lti-speech-classes/
- Order: overview; formulations; feature extraction; two ESPnet hands-on sessions;
  alignment; HMM (three parts); advanced acoustic modeling; language models; deep learning
  for ASR; advanced architectures; neural LMs; end-to-end ASR (attention, CTC, RNN-T);
  advanced end-to-end topics; search; guest lectures on robustness and enhancement.
- Borrow: explicit "formulations" lecture that relates HMM, CTC, attention, and RNN-T as
  different factorizations of $p(W \mid X)$; search as its own lecture.

### CMU 11-785, Introduction to Deep Learning (speech-heavy homeworks)

- URL: https://deeplearning.cs.cmu.edu/ (for example, F24 at
  https://deeplearning.cs.cmu.edu/F24/index.html).
- Speech parts **(verified from homework descriptions)**: HW1P2 frame-level phoneme
  classification from MFCC; HW3P2 utterance-to-phoneme mapping with CTC loss and beam
  search on unaligned labels; HW4P2 attention-based ASR (Listen, Attend and Spell with a
  pyramidal BiLSTM listener). The Autolab parts have students implement layers (including
  RNN and GRU cells, CTC forward-backward, and beam search) from scratch in NumPy.
- Borrow: the "implement it in NumPy, then use it in PyTorch" pairing, and autograder
  tolerance design. This is the closest existing model to the planned browser labs.

### University of Edinburgh, Speech Processing (speech.zone)

- URL: https://speech.zone/courses/speech-processing/ **(verified)**. Organizer: Catherine
  Lai (with Simon King's materials).
- Modules: 0 getting started; 1 phonetics and representations; 2 acoustic phonetics;
  3 digital speech signals (spectrogram, DSP, DFT); 4 source-filter model; 5 TTS front end
  (pronunciation, letter-to-sound, prosody with CART); 6 waveform generation; 7 ASR by
  pattern matching (DTW); 8 feature engineering (MFCC); 9 HMM; 10 connected speech and HMM
  training.
- Assignments: build a Festival TTS voice and a digit recognizer (HTK).
- Borrow: the DTW-before-HMM progression, and the digit recognizer as the first complete
  ASR system.

### University of Edinburgh, Automatic Speech Recognition (ASR), 2024-25

- URL: https://www.inf.ed.ac.uk/teaching/courses/asr/index-2025.html; lectures
  https://www.inf.ed.ac.uk/teaching/courses/asr/lectures-2025.html **(verified)**. Peter
  Bell.
- 18 lectures: introduction; signal analysis 1 and 2; HMMs; HMM algorithms; GMMs;
  context-dependent phone models; large-vocabulary ASR; WFSTs; hybrid NN acoustic models;
  NN architectures; speaker adaptation; CTC; RNN-T; multilingual and low-resource;
  attention encoder-decoder; self-supervised learning; ASR with LLMs.
- Labs and coursework use OpenFst with Python, then Kaldi.
- Borrow: WFST lecture placement (after HMMs, before hybrid NN), and closing on SSL and
  LLM-based ASR.

### Johns Hopkins University (CLSP)

- EN.520.666 *Information Extraction from Speech and Text* (Sanjeev Khudanpur), the
  classic JHU ASR course in the HLT master's program **(verified to exist; syllabus not
  retrieved)**. CLSP course list: https://www.clsp.jhu.edu/courses/ (returned 403 during
  research).
- JHU's Kaldi, k2, and Icefall lineage (lattice-free MMI, WFSTs) informs the "classical"
  path. The JSALT summer workshops (https://www.clsp.jhu.edu/workshops/) produce research
  tutorials.

### MIT 6.345, Automatic Speech Recognition (Spring 2003, OCW)

- URL: https://ocw.mit.edu/courses/6-345-automatic-speech-recognition-spring-2003/
  **(verified)**. Jim Glass and Victor Zue.
- Topics: acoustic theory of speech production, speech sounds, signal representation,
  pattern classification, HMMs, acoustic modeling, language modeling, graphical models,
  segment-based ASR, and search. Has classic problem sets and spectrogram reading.
- Borrow: spectrogram reading exercises and the acoustic-phonetic depth.

### Hugging Face Audio Course

- URL: https://huggingface.co/learn/audio-course/chapter0/introduction **(verified)**.
- Units: 0 welcome; 1 working with audio data; 2 pipelines; 3 Transformer architectures
  for audio (CTC and seq2seq); 4 music genre classifier; 5 ASR (fine-tune Whisper); 6 TTS;
  7 putting it together (speech-to-speech translation, voice assistant, meeting
  transcription); 8 finish line.
- Borrow: nothing from-scratch, but excellent for "use the pretrained model" capstones and
  for the `datasets` streaming patterns.

### NVIDIA NeMo tutorials

- Speech code moved to https://github.com/NVIDIA-NeMo/Speech **(verified)**. Tutorials:
  https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html
- Notebooks include ASR with NeMo, ASR for telephony, cache-aware streaming and buffered
  streaming ASR, hybrid ASR-TTS, and TTS (FastPitch, HiFi-GAN). 2026 releases include
  Nemotron-3.5-ASR-Streaming-0.6B (controllable latency 80 ms to 1 s) and
  Parakeet-unified-en-0.6b **(from search result; unverified)**.
- Borrow: cache-aware streaming as a reference design for the streaming lesson.

### Jurafsky and Martin, Speech and Language Processing (3rd ed. draft)

- URL: https://web.stanford.edu/~jurafsky/slp3/ **(verified)**. The draft dated
  2026-08-19 is reorganized into volumes. Speech chapters: 15 *Phonetics and Speech
  Feature Extraction*, 16 *Automatic Speech Recognition*, 17 *Text-to-Speech*. Appendix A
  covers HMMs.
- Borrow: free, current reading for the phonetics, features, ASR, and TTS lessons. Chapter
  numbers changed between drafts, so cite chapter titles, not numbers.

### Speech LLM courses and tutorials (2025-2026)

- No full university course dedicated to speech LLMs was found. The closest resources:
  - Interspeech 2026 tutorial "On the landscape of spoken language models"
    (https://interspeech2026slm.github.io/), with presenters from MIT, TTIC, CMU, and NTU
    (Hung-yi Lee, Karen Livescu, Shinji Watanabe, and others) and hands-on coding on
    GitHub **(verified)**.
  - Hung-yi Lee's NTU lectures on speech foundation models and speech LLMs (YouTube)
    **(unverified specific 2026 offering)**.
  - Kyutai's "Neural audio codecs" blog post (2025-10-21, https://kyutai.org/blog/) as an
    explainer on how codecs plug into LLMs **(verified to exist)**.

**Synthesis.** Every traditional course uses the same spine: phonetics, signals, features,
pattern matching or HMM, neural ASR, TTS. None covers codecs, diffusion, flow matching,
and speech LLMs from scratch, which is this course's gap to fill.

---

## Part B2: Small datasets for labs

The following table lists license, size, and sample rate. "Bake" means the data (or a
derived subset) fits in the CPU Docker image budget of about 500 MB.

| Dataset | License | Size (download) | Rate | Contents | Bake? | Stable URL |
|---|---|---|---|---|---|---|
| yesno (OpenSLR 1) | "No formal license but free to use for any purpose" **(verified)** | 4.7 MB | 8 kHz | 60 files, 8 Hebrew yes/no words each, 1 speaker | Yes | https://openslr.trmal.net/resources/1/waves_yesno.tar.gz |
| Free Spoken Digit Dataset | CC BY-SA 4.0 **(verified)** | About 10-20 MB **(unverified)** | 8 kHz | 3,000 clips, 6 speakers, 50 per digit per speaker | Yes | https://github.com/Jakobovski/free-spoken-digit-dataset (Zenodo DOI tags) |
| Speech Commands v2 | CC BY 4.0 **(verified)** | 2.37 GiB (8.17 GiB extracted) **(verified)** | 16 kHz | 105,829 1 s clips, 35 words | Subset only (precomputed log-mels) | http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz |
| LibriSpeech dev-clean | CC BY 4.0 **(verified)** | 337 MB | 16 kHz | About 5.4 h, 40 speakers | Borderline; subset | https://openslr.trmal.net/resources/12/dev-clean.tar.gz |
| LibriSpeech test-clean | CC BY 4.0 | 346 MB **(verified)** | 16 kHz | About 5.4 h, 40 speakers | Borderline; subset | https://openslr.trmal.net/resources/12/test-clean.tar.gz |
| Mini LibriSpeech (OpenSLR 31) | CC BY 4.0 **(verified)** | dev-clean-2 126 MB; train-clean-5 332 MB | 16 kHz | About 2 h dev, 5 h train | dev-clean-2 yes | https://openslr.trmal.net/resources/31/ |
| LJSpeech 1.1 | Public domain **(verified)** | 2.6 GB | 22.05 kHz | 13,100 clips, about 24 h, 1 speaker | No (subset of 300 clips is about 60 MB) | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 |
| VCTK 0.92 | CC BY 4.0 **(license unverified on page; widely cited)** | 10.94 GB **(verified)** | 48 kHz | 110 speakers, about 400 sentences each | No | https://doi.org/10.7488/ds/2645 |
| LibriTTS-R | CC BY 4.0 **(verified)** | dev_clean 1.3 GB; test_clean 1.2 GB | 24 kHz | Restored LibriTTS | No (subset) | https://openslr.trmal.net/resources/141/ |
| ESC-50 | CC BY-NC 3.0 (ESC-10 subset CC BY) **(verified)** | About 600 MB | 44.1 kHz | 2,000 5 s clips, 50 classes | Resampled 16 kHz subset (ESC-10: 400 clips, about 13 MB at 16 kHz int16) | https://github.com/karoldvl/ESC-50/archive/master.zip |
| MUSAN (OpenSLR 17) | CC BY 4.0 **(verified)** | 11 GB | 16 kHz | Music, speech, noise | Curated noise subset only | https://openslr.trmal.net/resources/17/musan.tar.gz |
| RIRS_NOISES (OpenSLR 28) | Apache 2.0 **(verified)** | 1.3 GB | 16 kHz | Real and simulated RIRs, point noises | Subset of 200 RIRs (about 10 MB) | https://openslr.trmal.net/resources/28/rirs_noises.zip |
| TIMIT (LDC93S1) | LDC user agreement; paid for non-members **(verified; fee not shown without login)** | About 440 MB **(unverified)** | 16 kHz | 630 speakers, time-aligned phones | Not redistributable | https://catalog.ldc.upenn.edu/LDC93S1 |
| CMUdict | Unrestricted use (BSD-style license file) **(verified)** | About 3.5 MB **(unverified)** | N/A | About 134k entries, 39 ARPAbet phones with stress **(unverified count)** | Yes | https://github.com/cmusphinx/cmudict |

Notes:

- OpenSLR's primary download host is `openslr.trmal.net`, with EU (`openslr.elda.org`) and
  CN (`openslr.magicdatatech.com`) mirrors **(verified)**. Older scripts use
  `www.openslr.org/resources/...`; pin checksums from each resource's `md5sum.txt`.
- **TIMIT alternatives** for phone-level labs: LibriSpeech or Mini LibriSpeech with Montreal
  Forced Aligner phone alignments (MFA is MIT-licensed; precomputed LibriSpeech alignments
  exist from several authors **(unverified licensing of each)**); CMU ARCTIC (free, BSD-style
  license, with phone labels **(unverified)**); or fully synthetic formant speech with
  exact phone boundaries (see Part B3).
- **Attribution and ShareAlike.** CC BY requires attribution in the course; FSDD's CC BY-SA
  requires derived datasets (for example, precomputed features) to carry the same license.
  ESC-50's NC clause is fine for a free course but not for a paid product.
- **Budget guideline for the CPU image:** about 50-150 MB of data total, stored as
  precomputed `float16` features or 16 kHz `int16` `.npy` arrays rather than FLAC, so labs
  don't need a decoder.

---

## Part B3: Synthetic data strategies (offline, deterministic)

Synthetic signals let checks run in seconds with exact ground truth. Seed every generator.

1. **Sines, chirps, and noise.** `scipy.signal.chirp`, white noise, pink noise (1/f by
   spectral shaping), and band-limited noise. Ground truth for aliasing, windowing, STFT
   resolution, and pitch tracking.
2. **Glottal source plus formant filter (source-filter).** Impulse train at $F_0$ (or a
   Rosenberg or LF glottal pulse), passed through a cascade of second-order resonators
   (Klatt, 1980, https://doi.org/10.1121/1.383940). One resonator with center frequency $F$,
   bandwidth $B$, and sample period $T$:
   $$y[n] = A\,x[n] + B_1\,y[n-1] + C\,y[n-2]$$
   $$C = -e^{-2\pi B T}, \quad B_1 = 2 e^{-\pi B T}\cos(2\pi F T), \quad A = 1 - B_1 - C$$
   Typical adult male formants (Peterson and Barney, 1952,
   https://doi.org/10.1121/1.1906875): /i/ 270, 2290, 3010 Hz; /a/ 730, 1090, 2440 Hz;
   /u/ 300, 870, 2240 Hz **(values as commonly tabulated; verify against the paper)**.
   Bandwidths of 60-120 Hz. Add jitter, shimmer, and a slight $F_0$ contour for realism.
   Ground truth for LPC, formant tracking, pitch, and vowel classification.
3. **Synthetic "words" and "sentences."** Concatenate vowel segments and
   noise-burst "consonants" with known durations to make a vocabulary of 10-20 words, with
   exact phone boundaries. Good for DTW, HMM, CTC, and forced-alignment labs. Randomize
   $F_0$, formant scaling (speaker "vocal tract length"), and speaking rate to make it
   non-trivial.
4. **DTMF digits.** Row tones 697, 770, 852, 941 Hz and column tones 1209, 1336, 1477 Hz
   (ITU-T Q.23). A 10-class "spoken digit" substitute for classification and Goertzel
   algorithm labs.
5. **Simulated rooms.** Image-method RIRs (Part A3.6), with random room sizes and RT60s for
   dereverberation and augmentation.
6. **Two-source mixtures.** Chirp plus vowel train, or two formant "speakers" with different
   $F_0$ ranges, for separation and PIT labs.
7. **Toy codecs and tokens.** Random 2-D Gaussian mixtures for VQ and RVQ labs; synthetic
   RVQ code sequences from a known Markov chain for AR-LM and delay-pattern labs, where the
   true entropy is known and the check is "model NLL within 5% of the true entropy."
8. **Toy diffusion and flow targets.** 1-D mixtures of Gaussians and short sinusoid
   "waveforms" with random phase for DDPM and flow-matching labs, checking a moment match
   or a 1-D Wasserstein distance.
9. **Pretrained-model stand-ins.** For adapter and kNN-VC labs, a fixed random conv encoder
   with a seeded init gives deterministic "features" with no downloads, or small cached
   feature files baked into the image.

---

## Part B4: Python audio tooling landscape (September 2026)

### torchaudio

- **Status (verified):** torchaudio entered maintenance mode starting with 2.9. APIs
  deprecated in 2.8 were removed in 2.9. The I/O stack (`torchaudio.io`, `StreamReader`,
  `StreamWriter`, sox backend, backend dispatch such as `list_audio_backends`, and
  `sox_effects`) was deprecated in 2.8 and removed in 2.9 **(removal of specific
  submodules partly from secondary sources)**. The CTC decoder API (flashlight-based) is
  listed among removals in secondary sources **(unverified; the docs index still shows
  a CUDA CTC decoder tutorial)**.
- `torchaudio.load()` and `torchaudio.save()` still exist as aliases of
  `load_with_torchcodec()` and `save_with_torchcodec()`, which call TorchCodec. The docs
  recommend porting to native TorchCodec APIs **(verified)**.
- **Kept (verified):** "most of the APIs in `transforms`, `functional`,
  `compliance.kaldi`, `models` and `pipelines`" remain. C++ and CUDA extensions
  initially slated for removal were preserved: `lfilter`, `RNNTLoss`, CUDA CTC decoder
  (`CUCTC`), `forced_align`, and `overdrive` (issue https://github.com/pytorch/audio/issues/3902).
  The docs index still lists `datasets` and tutorials for resampling, augmentation, feature
  extraction, forced alignment, wav2vec 2.0, Tacotron 2, MVDR, Hybrid Demucs, and SQUIM.
- **Final release (verified):** torchaudio 2.11.0 (2026-03-23) is built against PyTorch's
  stable ABI and "works with PyTorch 2.11 and with every future PyTorch release." There is
  no torchaudio 2.12, 2.13, or 2.14, while torch itself is at 2.14.0 (2026-09-02). Install
  `torchaudio==2.11.*` alongside any torch 2.11 or later.
- **CTC loss is in torch core:** `torch.nn.CTCLoss` and `torch.nn.functional.ctc_loss`.
  RNN-T loss remains in torchaudio (`torchaudio.functional.rnnt_loss`).
- **Course implication:** implement STFT, mel, MFCC, resampling, and CTC decoding from
  scratch in NumPy (the pedagogy wants this anyway), use `torch.stft` and
  `torch.nn.functional.ctc_loss` for PyTorch labs, and treat torchaudio as optional.

### Decoding and I/O

- **TorchCodec** 0.16.0 (2026-08-13) **(verified)**: `AudioDecoder` and `AudioEncoder`;
  requires torch 2.11 or later and a system FFmpeg (versions 4-9 supported). Linux wheels
  install CUDA variants by default; use `--index-url https://download.pytorch.org/whl/cpu`
  for CPU. FFmpeg adds about 50-100 MB to a slim image **(unverified)**.
- **soundfile** 0.14.0 (2026-06-06) **(verified)**: bundles libsndfile in wheels; reads WAV,
  FLAC, OGG, and MP3 (since 0.11). No FFmpeg needed. Recommended default for the CPU image.
- **scipy.io.wavfile**: zero extra dependencies for WAV only.
- **Python `wave`** module: stdlib, PCM WAV only, good for writing base64 WAV bytes.

### Signal processing and features

- **scipy.signal**: `stft`, `istft`, `get_window`, `lfilter`, `resample_poly`, `firwin`,
  `fftconvolve`, `chirp`, `find_peaks`. Enough for all DSP labs.
- **librosa** 1.0.0 (2026-08-11) **(verified)**: requires Python 3.12 or later; depends on
  numba (which pulls llvmlite, about 100-150 MB combined **(unverified size)**), soxr,
  audioread, and lazy_loader. Useful as a *reference* implementation in graders (for
  example, compare student mel filterbanks to `librosa.filters.mel`), but heavy for the
  student image. Consider installing it only in the grader image or precomputing reference
  outputs.
- **julius** (fast resampling and filters in PyTorch, from Alexandre Défossez) and
  **audiotools** (Descript) are optional; both are small **(maintenance status
  unverified)**.
- **pyroomacoustics** 0.10.1 (MIT) for RIR simulation **(verified)**.

### Loading pretrained models without transformers

- **transformers** 5.17.0 (2026-09-09) **(verified)** loads Whisper, wav2vec 2.0, HuBERT,
  WavLM, EnCodec, Mimi, and many speech LLMs. It's convenient but large and changes often.
- **Alternatives:**
  - `safetensors` plus hand-written model code. For small models (Whisper-tiny/base,
    wav2vec 2.0 base, EnCodec 24 kHz, Mimi), writing the forward pass is itself a good
    capstone lab. Download weights with `huggingface_hub.hf_hub_download` at image build
    time.
  - Kyutai's `moshi` pip package provides Mimi; Meta's `encodec` package provides EnCodec;
    `openai-whisper` provides Whisper (depends on torch and tiktoken).
  - ONNX exports with `onnxruntime` (CPU wheel about 15-20 MB **(unverified)**).
- Recommendation: CPU image uses no transformers; GPU image includes transformers pinned to
  one version.

### Image sizes (rough)

- `python:3.12-slim`: about 45 MB compressed, 125 MB uncompressed **(unverified)**.
- NumPy plus SciPy: about 100-120 MB installed **(unverified)**.
- CPU torch: about 616 MB installed for the torch package (vs. 913 MB torch plus about
  2.8 GB NVIDIA libraries for the default CUDA build) **(secondary source)**. The
  PyPI Linux wheel for torch 2.14.0 cp312 is 554.6 MB and is a CUDA build **(verified)**.
  Community CPU images with Python and torch are about 245-625 MB compressed
  **(secondary source)**.
- **Estimate for the CPU lab image:** python:3.12-slim plus NumPy, SciPy, soundfile,
  CPU torch, and about 100 MB of data is about 1.0-1.2 GB uncompressed, about
  400-500 MB compressed.
- **Estimate for the CUDA worker image:** a CUDA runtime base (or `runpod/pytorch`) plus
  torch CUDA, transformers, and dependencies is about 6-10 GB uncompressed. Keep model
  weights out of the image (see Part B5).

---

## Part B5: RunPod serverless practicalities (September 2026)

All items in this part were checked against docs.runpod.io and runpod.io/pricing on
2026-09-28 unless marked.

### GPU type IDs and pools

The REST `POST /endpoints` schema accepts `gpuTypeIds` with exact strings, including the
following **(verified)**:

- `NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB`
- `NVIDIA L4`, `NVIDIA L40S`, `NVIDIA L40`
- `NVIDIA GeForce RTX 4090`, `NVIDIA GeForce RTX 5090`, `NVIDIA GeForce RTX 3090`
- `NVIDIA RTX A5000`, `NVIDIA RTX A4000`, `NVIDIA RTX A6000`, `NVIDIA A40`
- `NVIDIA H100 80GB HBM3`, `NVIDIA H100 PCIe`, `NVIDIA H100 NVL`, `NVIDIA H200`, `NVIDIA B200`
- `NVIDIA RTX 6000 Ada Generation`, `NVIDIA RTX 4000 Ada Generation`,
  `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `Tesla T4`, `AMD Instinct MI300X OAM`

Serverless GPU pools (console and GraphQL `gpuIds`) **(verified)**:

| Pool | GPUs | VRAM |
|---|---|---|
| `AMPERE_16` | A4000, A4500, RTX 4000, RTX 2000 | 16 GB |
| `AMPERE_24` | L4, A5000, 3090 | 24 GB |
| `ADA_24` | 4090 | 24 GB |
| `AMPERE_48` | A6000, A40 | 48 GB |
| `ADA_48_PRO` | L40, L40S, 6000 Ada | 48 GB |
| `AMPERE_80` | A100 | 80 GB |
| `ADA_80_PRO` | H100 | 80 GB |
| `HOPPER_141` | H200 | 141 GB |

Note that L4 isn't a standalone pool: `AMPERE_24` mixes L4, A5000, and 3090. To force L4,
pass `gpuTypeIds: ["NVIDIA L4"]` through REST. Multiple GPU categories form a prioritized
fallback list.

### Prices (serverless flex, per hour as listed; per-second derived)

| Tier | $/hour | $/second (derived) |
|---|---|---|
| 80 GB A100 | 2.72 | 0.000756 |
| 24 GB (L4, A5000, 3090) | 0.69 | 0.000192 |
| 24 GB RTX 4090 | 1.10 | 0.000306 |
| 48 GB (L40, L40S, 6000 Ada) | 1.75 | 0.000486 |
| 80 GB H100 | 4.79 | 0.001331 |
| 16 GB tier | 0.58 | 0.000161 |

These are flex-worker rates **(verified hourly figures; per-second is hourly/3600)**.
Active (always-on) workers are discounted (secondary sources cite 20-40%
**(unverified)**). Billing runs from worker start until it fully stops, rounded up to the
second **(verified)**, which includes cold-start time and the idle timeout. Network
volumes cost $0.07/GB/month under 1 TB **(verified)**.

Cost example: a 45 s kNN-VC job on L4 costs about $0.009 plus cold start. A class of 200
students running 10 such jobs each costs about $20-40 including cold starts.

### Endpoint settings

- Execution timeout: default 600 s, min 5 s, max 7 days **(verified)**. REST field
  `executionTimeoutMs`.
- Idle timeout: default 5 s (REST `idleTimeout`, 1-3600 s) **(verified)**.
- Job TTL: default 24 h, from submission **(verified)**.
- `workersMin` (active workers, always billed) and `workersMax`; scaler `QUEUE_DELAY`
  (default) or `REQUEST_COUNT` **(verified)**.
- **FlashBoot:** enabled by default on new endpoints; retains worker state after spin-down
  for faster revival **(verified)**. Effectiveness depends on traffic; cold starts still
  happen after long idle periods.
- **Model weights:**
  - *Cached models* (platform feature): specify a Hugging Face repo; files mount at
    `/runpod-volume/huggingface-cache/hub/` in HF cache layout; you aren't billed while the
    model downloads; one cached model per endpoint **(verified)**.
  - *Network volumes*: mounted at `/runpod-volume`; persist across workers but pin the
    endpoint to the volume's data center and add read latency **(verified)**. The SDK's
    `VolumeCache` helper syncs directories such as `/root/.cache/huggingface` **(verified)**.
  - Baking weights into the image makes cold starts depend on image pull time; acceptable
    for weights under about 1-2 GB.

### Request operations and limits

Endpoint base URL `https://api.runpod.ai/v2/{ENDPOINT_ID}` with header
`Authorization: Bearer RUNPOD_API_KEY` **(URL pattern from memory; unverified in this
session)**. Operations **(verified)**:

- `/run` (async): payload up to 10 MB; results kept 30 minutes.
- `/runsync`: payload up to 20 MB; waits 90 s by default (`?wait=` 1,000-300,000 ms);
  results kept 1 minute (up to 5 with `wait`).
- `/status/{job_id}`, `/stream/{job_id}` (each streamed chunk up to 1 MB), `/cancel`,
  `/retry`, `/purge-queue`, `/health`.

Implication for audio: 10 s of 24 kHz 16-bit mono is 480 KB raw, 640 KB as base64. Fine
for `/run`. For longer audio, upload to object storage and pass a URL, or stream results
in chunks under 1 MB.

### Handler and streaming

The following handler streams chunks and also returns an aggregate for `/run` clients
**(verified pattern)**:

```python
import runpod

def handler(job):
    params = job["input"]
    for chunk in synthesize_chunks(params):  # user function
        yield {"audio_b64": chunk}           # each yield <= 1 MB

runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,  # /run and /runsync get the list
})
```

Clients poll `/stream/{job_id}` to receive yielded items as they arrive. Concurrent
handlers (async functions with a concurrency modifier) are documented at
https://docs.runpod.io/serverless/workers/concurrent-handler.

### REST vs. GraphQL

- REST API: `https://rest.runpod.io/v1` with resources `/endpoints`, `/templates`,
  `/networkvolumes`, and others. `POST /endpoints` requires `templateId` and accepts
  `gpuTypeIds`, `gpuCount`, `workersMin`, `workersMax`, `idleTimeout`, `executionTimeoutMs`,
  `flashboot`, `scalerType`, `scalerValue`, `networkVolumeId`, `dataCenterIds`,
  `allowedCudaVersions`, and `computeType` (`GPU` or `CPU`) **(verified)**. Reference:
  https://docs.runpod.io/api-reference/endpoints/POST/endpoints
- GraphQL (`https://api.runpod.io/graphql`) remains for legacy automation; the pool names
  such as `AMPERE_24` appear there **(GraphQL status unverified in this session)**.
- The docs also expose an "api-reference-v2" section **(verified to exist)**; check which
  version the course tooling targets.

### Recommendations for the course

- One CUDA worker image for all GPU labs, with a `lab_id` dispatch in the handler. Pin
  torch, transformers, and model revisions.
- Default GPU: L4 through `gpuTypeIds: ["NVIDIA L4"]` with `AMPERE_24` fallback. Use A100
  80 GB only for 7B+ models (Qwen2-Audio 7B in bf16 is about 16 GB and fits on L4;
  Qwen3-Omni-30B-A3B in bf16 is about 60-70 GB and needs A100 80 GB **(memory estimates
  unverified)**).
- Set `executionTimeoutMs` to 300,000 for labs, `idleTimeout` to 30-60 s during class
  sessions to reduce cold starts, and use cached models or a network volume for weights.
- CPU serverless endpoints exist (`computeType: "CPU"`), which could host heavier CPU
  graders outside the browser environment **(pricing unverified)**.

---

## Part B6: Rendering audio and plots in a browser learning app

### Playing audio returned by Python

1. In Python, write 16-bit PCM WAV into memory (stdlib `wave` or `soundfile.write(buf, x,
   sr, format="WAV", subtype="PCM_16")`), then base64-encode.
2. In the browser: `audio.src = "data:audio/wav;base64," + b64`. Base64 inflates size by
   4/3; 10 s of 16 kHz mono PCM is 320 KB raw and about 427 KB encoded.
3. For long or streamed audio, decode chunks with `AudioContext.decodeAudioData` or push
   raw `Float32Array` PCM into an `AudioWorkletNode` ring buffer.
4. Clip before converting to int16 (`np.clip(x, -1, 1)`), because wraparound on overflow
   produces loud clicks.
5. Browsers block autoplay with sound until a user gesture. Create or `resume()` the
   `AudioContext` in a click handler.

### Spectrograms on canvas

- Compute the STFT magnitude in dB (in Python or JS), normalize to [0, 1] with a fixed
  dynamic range (for example, max minus 80 dB), map through a 256-entry colormap lookup
  table (viridis or magma, precomputed RGB arrays), and write into an `ImageData` buffer
  with `putImageData`. Draw frequency on the vertical axis with low frequencies at the
  bottom (flip rows).
- For large spectrograms, render to an offscreen canvas once and scale with
  `drawImage` (set `imageSmoothingEnabled = false` for crisp bins).
- Use a log or mel frequency axis by resampling rows, or draw the mel spectrogram directly.

### Live microphone analysis

- `navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false,
  noiseSuppression: false, autoGainControl: false}})`. Disable processing for analysis
  labs, because browser noise suppression alters spectra.
- Requires a secure context: HTTPS, `localhost`, or `file://` **(verified)**. Inside an
  iframe, the parent must grant `allow="microphone"`; it can't be called from `srcdoc` or
  sandboxed frames without `allow-same-origin` **(verified)**.
- `AnalyserNode`: `fftSize` a power of two from 32 to 32768; `getFloatFrequencyData` returns
  dB values; `smoothingTimeConstant` defaults to 0.8 (set it to 0 for honest spectrograms);
  it applies a Blackman window internally **(from the Web Audio spec; unverified defaults
  in this session)**. For raw samples, use `getFloatTimeDomainData` or an `AudioWorklet`.
- The context sample rate is usually 44.1 or 48 kHz; resample to 16 kHz in JS before
  sending audio to Python models.

### JavaScript libraries

- **wavesurfer.js** (BSD-3-Clause): waveform, regions, timeline, record, and spectrogram
  plugins **(verified)**. Version 7 or 8 **(the repo references version 8; unverified)**.
- **Meyda** (MIT): real-time feature extraction (RMS, spectral centroid, MFCC, chroma)
  **(unverified license)**.
- **pitchy** (McLeod pitch method) and **pitchfinder** (YIN, AMDF) for pitch
  **(unverified licenses)**.
- **essentia.js**: WebAssembly port of Essentia; AGPL-3.0 **(unverified)**, so check
  licensing before bundling.
- **fft.js** or a hand-written radix-2 FFT (about 60 lines) for in-browser STFT.
- **Plotly.js** or **D3** for interactive plots; **KaTeX** for equations.
- **onnxruntime-web** and **transformers.js** to run small models (Whisper-tiny, Silero
  VAD) directly in the browser.

### Interactive visualization ideas

Each item states what it teaches and what the JS must compute.

1. **Sampling and aliasing slider.** Teaches Nyquist and folding. JS: generate a sine at
   frequency $f$, sample at $f_s$, show sample points and the reconstructed (sinc or
   linear) signal; display the alias $|f - k f_s|$. Play both.
2. **Quantization and dither.** Teaches bit depth, SNR of about $6.02 b + 1.76$ dB, and
   dither. JS: quantize to $b$ bits, show the error signal and its spectrum.
3. **Window and leakage explorer.** Teaches main-lobe width vs. side-lobe level. JS:
   rectangular, Hann, Hamming, and Blackman windows; zero-padded FFT of a windowed sine at
   a non-bin frequency; dB plot.
4. **STFT time-frequency tradeoff.** Teaches resolution. JS: STFT of a chirp plus clicks
   with window length and hop sliders; canvas spectrogram.
5. **Mel filterbank designer.** Teaches the mel scale
   $m = 2595 \log_{10}(1 + f/700)$ and triangular filters. JS: filter edges from equally
   spaced mel points, draw filters, apply to a live spectrum.
6. **MFCC pipeline stepper.** Teaches pre-emphasis, framing, window, power spectrum, mel,
   log, and DCT-II with liftering. JS: each stage as a small plot; toggle stages.
7. **Pitch tracker on microphone input.** Teaches autocorrelation, YIN, and octave errors.
   JS: autocorrelation or YIN difference function per 40 ms frame; plot the $F_0$ track.
8. **LPC envelope fitter.** Teaches the source-filter model and the all-pole spectrum. JS:
   autocorrelation method with Levinson-Durbin for order $p$ slider; overlay
   $1/|A(e^{j\omega})|$ on the FFT spectrum; show formants as pole angles.
9. **Formant vowel synthesizer (F1/F2 pad).** Teaches the vowel space. JS: a 2-D pad
   choosing $F_1, F_2$; cascade resonators on an impulse train in an `AudioWorklet`.
10. **DTW path explorer.** Teaches dynamic programming alignment. JS: cost matrix of two
    feature sequences, cumulative cost with step patterns, backtrace; animate the path.
11. **HMM Viterbi trellis.** Teaches Viterbi vs. forward. JS: a 3-state left-to-right HMM
    with editable transition and emission probabilities; animate max vs. sum per cell in
    log space.
12. **CTC alignment lattice.** Teaches the blank-extended label sequence, valid paths, and
    the forward variable. JS: build the $2L+1$ state graph, compute $\alpha$ over $T$
    frames from a synthetic posteriorgram; highlight the best path and show the collapse
    function.
13. **Beam search tree.** Teaches beam width, pruning, and length normalization. JS: expand
    a toy LM's top-$k$ at each step, keep $B$ beams, show pruned branches.
14. **Attention alignment map.** Teaches monotonic alignment in seq2seq ASR and TTS. JS:
    render a precomputed attention matrix as a heatmap; scrub decoder steps.
15. **WaveNet receptive field visualizer.** Teaches dilated causal convolutions. JS:
    draw layers with dilation $2^i$, highlight the input samples that affect one output;
    compute $R = 1 + (K-1)\sum_i d_i$.
16. **VQ and RVQ in 2-D.** Teaches codebooks, residual quantization, and bitrate. JS:
    k-means on 2-D points, then quantize residuals with a second codebook; show
    reconstruction error vs. number of stages; the bitrate is
    $\text{frame rate} \times N_q \times \log_2 K$.
17. **Codec bitrate calculator.** Teaches the token-rate and bitrate tradeoff. JS:
    arithmetic for EnCodec, DAC, Mimi, SNAC, and GLM-4-Voice tokenizers; tokens per second
    and context-length usage for an LLM.
18. **Diffusion on 1-D signals.** Teaches forward noising and reverse denoising. JS: DDPM
    schedule $\bar\alpha_t$, show $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
    for a sine; for reverse, use the analytic score of a Gaussian mixture.
19. **Flow matching trajectories.** Teaches straight paths and ODE integration. JS: 2-D
    Gaussian to two-moons with the conditional linear path
    $x_t = (1-t)x_0 + t x_1$; Euler steps with the analytic marginal velocity for a
    mixture target; step-count slider.
20. **Delay pattern and multi-stream grid.** Teaches MusicGen delays and Moshi streams. JS:
    a grid of tokens per codebook and time; toggle delay; highlight what is available
    when predicting a cell.
21. **Separation and PIT playground.** Teaches masks and permutations. JS: mix two
    synthetic sources, apply oracle IBM or IRM, compute SI-SDR, and shuffle outputs to show
    the permutation problem.
22. **Latency timeline.** Teaches voice-agent latency budgets. JS: a Gantt chart of VAD,
    ASR, LLM, and TTS stages with sliders; show time to first audio.

---
## Part C1: Proposed ordering of Half A lessons

The following order puts the classical and signal-level tasks first (they reuse STFT and
loss skills), then representation learning, then the generative and LLM topics that depend
on codecs and AR models from earlier units.

1. **Enhancement: from spectral subtraction to masks.** Lab: implement spectral
   subtraction, the Wiener gain with decision-directed SNR, and IBM and IRM oracles on
   synthetic noisy vowels; check SI-SDR gains. **CPU.**
2. **Separation: SI-SDR, PIT, and Conv-TasNet.** Lab: batched differentiable SI-SDR and
   uPIT (brute force and Hungarian), then a tiny Conv-TasNet separating a chirp from a
   vowel train; check invariances and SI-SDRi above 6 dB. **CPU.**
3. **Rooms, reverb, and augmentation.** Lab: image-method RIR, Schroeder decay and RT60
   vs. Sabine, SNR-controlled mixing; optional music demixing with pretrained Hybrid
   Demucs. **CPU** (demixing: **GPU, L4**, or slow CPU).
4. **Audio classification and keyword spotting.** Lab: AST-style patchify and a small
   CNN or BC-ResNet block on precomputed Speech Commands log-mels; symmetric CLAP loss.
   **CPU** (zero-shot CLAP on ESC-50: **GPU, L4**).
5. **Voice conversion and ethics.** Lab: kNN-VC regression on precomputed WavLM features
   with speaker and phone probes; full kNN-VC with vocoder on the student's own voice.
   **CPU** (core) plus **GPU, L4** (full pipeline).
6. **Textless spoken language modeling.** Lab: k-means units, deduplication, and a unit
   LM with a toy sWUGGY; optional unit Transformer on Mini LibriSpeech units. **CPU**
   (optional part **GPU, L4**).
7. **Speech LLMs I: the adapter pattern.** Lab: frame stacking, projector, and loss
   masking to plug frozen audio features into a tiny frozen LM; then Whisper-small encoder
   plus a 0.5B LLM on LibriSpeech. **CPU** plus **GPU, L4**.
8. **Speech LLMs II: speaking, interleaving, and full duplex.** Lab: delay patterns,
   GLM-style and Moshi-style stream builders, and a tiny RQ-Transformer checked against
   the brute-force joint likelihood; optional inference with a real model (Moshi or
   Qwen2.5-Omni) returning audio. **CPU** plus optional **GPU** (L4 for Moshi 7B in bf16
   **(memory unverified)**, A100 80 GB for Qwen3-Omni).
9. **Streaming systems, anti-spoofing, and watermarking.** Lab: stateful causal conv
   with chunked-equals-offline check, VAD and turn-taking state machine, EER, and a
   spread-spectrum watermark. **CPU.**

---

## Part C2: Recommended overall course structure (38 lessons, compressible to 36)

The structure runs from signals to speech LLMs. Units 1-3 are CPU only. GPU labs appear
from Unit 4 onward, always with a CPU-checkable core. Each lesson has one auto-graded lab.

### Unit 1: Sound and signals (lessons 1-6)

1. **Sound, sampling, and aliasing.** Lab: sample and reconstruct sines; alias
   frequency prediction. CPU.
2. **Quantization, WAV files, and loudness.** Lab: write a WAV encoder with `wave` and
   base64; dB and RMS; mu-law companding. CPU.
3. **The DFT and FFT from scratch.** Lab: radix-2 FFT matching `numpy.fft`; windows and
   leakage. CPU.
4. **The STFT and its inverse.** Lab: STFT and ISTFT with COLA check; resolution tradeoff.
   CPU.
5. **Speech production and the source-filter model.** Lab: Klatt resonator vowel
   synthesis; LPC with Levinson-Durbin; formant estimation. CPU.
6. **Pitch.** Lab: autocorrelation and YIN pitch trackers on synthetic and recorded speech;
   gross pitch error metric. CPU.

### Unit 2: Features and classical recognition (lessons 7-12)

7. **Mel scale, filterbanks, and log-mel.** Lab: mel filterbank matching a reference
   within tolerance. CPU.
8. **MFCCs, deltas, and normalization.** Lab: DCT-II, liftering, CMVN. CPU.
9. **Template matching with DTW.** Lab: DTW digit recognizer on FSDD or synthetic words.
   CPU.
10. **GMMs and EM.** Lab: diagonal GMM with EM; log-likelihood monotonicity check. CPU.
11. **HMMs: forward, Viterbi, and Baum-Welch.** Lab: log-space forward and Viterbi on a
    toy isolated-word recognizer; yesno dataset. CPU.
12. **Language models, WER, and decoding graphs.** Lab: n-gram LM with smoothing,
    edit-distance WER, and a tiny WFST-style lexicon composition. CPU.

### Unit 3: Neural ASR (lessons 13-19)

13. **Neural acoustic models.** Lab: frame-level phone classifier (MLP, then CNN) on
    synthetic or aligned Mini LibriSpeech features. CPU.
14. **CTC: loss and forward-backward from scratch.** Lab: NumPy CTC loss matching
    `torch.nn.functional.ctc_loss`. CPU.
15. **CTC decoding: greedy, beam search, and LM fusion.** Lab: prefix beam search with a
    character n-gram. CPU.
16. **Attention encoder-decoder ASR.** Lab: tiny LAS or Transformer on synthetic "word"
    speech; attention map plot. CPU.
17. **RNN-T and streaming ASR.** Lab: RNN-T loss lattice (NumPy) checked against
    `torchaudio.functional.rnnt_loss`; chunked attention masks. CPU.
18. **Conformer, SpecAugment, and training at scale.** Lab: Conformer block, SpecAugment;
    then fine-tune a pretrained CTC model. CPU plus GPU (L4).
19. **Self-supervised speech and Whisper.** Lab: wav2vec 2.0 contrastive loss and
    HuBERT-style masked prediction on toy data; Whisper inference with log-mel from scratch.
    CPU plus GPU (L4).

### Unit 4: Speech synthesis (lessons 20-25)

20. **TTS front end.** Lab: text normalization rules and a G2P using CMUdict lookup plus a
    fallback model. CPU.
21. **Acoustic models: Tacotron 2 and FastSpeech 2.** Lab: duration expansion (length
    regulator) and monotonic alignment search. CPU.
22. **Vocoders I: WaveNet and autoregressive waveform models.** Lab: dilated causal conv
    stack, receptive field, mu-law softmax on a tiny dataset. CPU.
23. **Vocoders II: GAN vocoders (HiFi-GAN).** Lab: multi-resolution STFT loss,
    multi-period discriminator shapes; inference with a pretrained HiFi-GAN. CPU plus GPU
    (L4).
24. **End-to-end TTS: VITS.** Lab: the KL term, monotonic alignment search, and a
    normalizing-flow coupling layer with exact log-determinant check. CPU.
25. **Evaluation of speech generation.** Lab: MCD, F0 RMSE, WER with ASR, speaker
    similarity; MOS study design and confidence intervals. CPU.

### Unit 5: Neural codecs and token models (lessons 26-29)

26. **Vector quantization and RVQ.** Lab: k-means VQ with straight-through estimator,
    commitment loss, RVQ; bitrate calculation. CPU.
27. **Neural audio codecs: SoundStream, EnCodec, DAC, and Mimi.** Lab: toy conv codec with
    RVQ on synthetic audio; run pretrained EnCodec or Mimi and compare bitrates. CPU plus
    GPU (L4).
28. **Codec language models: AudioLM, VALL-E, and delay patterns.** Lab: AR model over
    synthetic RVQ codes with delay pattern; NLL near the true entropy. CPU.
29. **Zero-shot TTS with codec LMs.** Lab: prompt-conditioned generation with a small
    pretrained model (for example, Pocket TTS on CPU or CosyVoice on GPU). CPU or GPU.

### Unit 6: Diffusion and flow matching for audio (lessons 30-32)

30. **Diffusion models from scratch.** Lab: DDPM on 1-D signals with analytic checks of
    $q(x_t \mid x_0)$ and the simplified loss. CPU.
31. **Audio diffusion: DiffWave, Grad-TTS, and latent diffusion.** Lab: tiny DiffWave-style
    vocoder on synthetic sinusoids; classifier-free guidance. CPU plus GPU (L4).
32. **Flow matching and its audio uses (Voicebox, F5-TTS, CosyVoice, Pocket TTS).**
    Lab: conditional flow matching on 2-D and 1-D targets, Euler vs. midpoint solvers,
    step count vs. quality; masked infilling. CPU.

### Unit 7: Beyond ASR and TTS (lessons 33-34)

33. **Enhancement, separation, and augmentation.** Combines Half A lessons 1-3. CPU plus
    optional GPU.
34. **Classification, voice conversion, and security.** Combines Half A lessons 4, 5, and
    9's security part. CPU plus GPU (L4).

### Unit 8: Spoken language models (lessons 35-38)

35. **Textless spoken language modeling.** Half A lesson 6. CPU.
36. **Speech LLMs I: the adapter pattern.** Half A lesson 7. CPU plus GPU (L4).
37. **Speech LLMs II: interleaving, Thinker-Talker, and full duplex.** Half A lesson 8.
    CPU plus optional GPU (A100 80 GB for Qwen3-Omni).
38. **Streaming voice agents (capstone).** Half A lesson 9's streaming part, plus a
    cascaded agent (VAD, streaming ASR, text LLM, streaming TTS) with a latency report;
    compare to an end-to-end model. CPU plus GPU (L4).

If the course must stay at 36 lessons, merge lessons 33 and 34, and merge 35 into 36.

**Design notes.**

- Put the codec unit before diffusion and flow matching, because modern TTS decoders
  (CosyVoice, Kimi-Audio, GLM-4-Voice) use flow matching on top of codec or semantic
  tokens, and speech LLMs need both.
- Reuse components deliberately: the log-mel from lesson 7 feeds every later lab; the CTC
  loss from lesson 14 reappears in LLaMA-Omni's speech decoder; the RQ and delay code from
  Unit 5 reappears in Moshi; SI-SDR from Unit 7 serves as a metric in codec labs.
- Keep a "reference implementation" oracle for each lab in the grader image (librosa,
  torchaudio 2.11, scipy), not in the student image.

---

## Part C3: Bibliography

### Speech and audio LLMs

- SpeechGPT: https://arxiv.org/abs/2305.11000
- AudioPaLM: https://arxiv.org/abs/2306.12925
- SALMONN: https://arxiv.org/abs/2310.13289
- Qwen-Audio: https://arxiv.org/abs/2311.07919
- Qwen2-Audio: https://arxiv.org/abs/2407.10759
- Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
- Qwen3-Omni: https://arxiv.org/abs/2509.17765
- Qwen3.5-Omni: https://arxiv.org/abs/2604.15804
- Qwen-Audio-3.0-Gen-Preview: https://arxiv.org/abs/2607.27011
- LLaMA-Omni: https://arxiv.org/abs/2409.06666
- Mini-Omni: https://arxiv.org/abs/2408.16725
- Mini-Omni2: https://arxiv.org/abs/2410.11190
- GLM-4-Voice: https://arxiv.org/abs/2412.02612
- Moshi: https://arxiv.org/abs/2410.00037
- Delayed streams modeling (Kyutai STT and TTS): https://arxiv.org/abs/2509.08753
- Continuous audio language models (CALM): https://arxiv.org/abs/2509.06926
- Kyutai blog (Pocket TTS, Hibiki-Zero, Unmute, codecs explainer): https://kyutai.org/blog/
- Kimi-Audio: https://arxiv.org/abs/2504.18425
- Step-Audio: https://arxiv.org/abs/2502.11946
- Step-Audio 2: https://arxiv.org/abs/2507.16632
- Step-Audio-R1: https://arxiv.org/abs/2511.15848
- Step-Audio-R1.5: https://arxiv.org/abs/2604.25719
- Voxtral: https://arxiv.org/abs/2507.13264
- Voxtral Realtime: https://arxiv.org/abs/2602.11298
- MiniCPM-o 4.5: https://arxiv.org/abs/2604.27393
- Sesame CSM: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice and https://github.com/SesameAILabs/csm
- LongCat-Flash-Omni: https://arxiv.org/abs/2511.00279
- Fun-Audio-Chat: https://arxiv.org/abs/2512.20156
- Covo-Audio: https://arxiv.org/abs/2602.09823
- MiniMind-O: https://arxiv.org/abs/2605.03937
- MusicGen (delay pattern): https://arxiv.org/abs/2306.05284
- Survey, landscape of SLMs: https://arxiv.org/abs/2504.08528
- Survey, recent advances in SpeechLMs: https://aclanthology.org/2025.acl-long.682.pdf
- Full-duplex survey (2026): https://arxiv.org/abs/2606.19453
- HumDial challenge study: https://arxiv.org/abs/2604.21406

### Benchmarks

- VoiceBench: https://arxiv.org/abs/2410.17196
- AIR-Bench: https://arxiv.org/abs/2402.07729
- MMAU: https://arxiv.org/abs/2410.19168
- Big Bench Audio: https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio
- URO-Bench: https://arxiv.org/abs/2502.17810
- WavBench: https://arxiv.org/abs/2602.12135
- Audio MultiChallenge: https://arxiv.org/abs/2512.14865

### Textless NLP

- GSLM: https://arxiv.org/abs/2102.01192
- pGSLM: https://arxiv.org/abs/2109.03264
- dGSLM: https://arxiv.org/abs/2203.16502
- TWIST: https://arxiv.org/abs/2305.13009
- SpiritLM: https://arxiv.org/abs/2402.05755
- Zero Resource Speech Challenge 2021 (sWUGGY, sBLIMP): https://arxiv.org/abs/2011.11588

### Enhancement, separation, and rooms

- Boll, spectral subtraction (1979): https://doi.org/10.1109/TASSP.1979.1163209
- Ephraim and Malah (1984): https://doi.org/10.1109/TASSP.1984.1164453
- Wang, Narayanan, and Wang, training targets (2014): https://doi.org/10.1109/TASLP.2014.2352935
- SI-SDR: https://arxiv.org/abs/1811.02508
- PIT: https://arxiv.org/abs/1607.00325
- uPIT: https://arxiv.org/abs/1703.06284
- Conv-TasNet: https://arxiv.org/abs/1809.07454
- DPRNN: https://arxiv.org/abs/1910.06379
- SepFormer: https://arxiv.org/abs/2010.13154
- TF-GridNet: https://arxiv.org/abs/2211.12433
- Demucs (music): https://arxiv.org/abs/1911.13254
- Demucs (speech denoising): https://arxiv.org/abs/2006.12847
- HT Demucs: https://arxiv.org/abs/2211.08553
- DeepFilterNet: https://arxiv.org/abs/2110.05588; v2 https://arxiv.org/abs/2205.05474; v3 https://arxiv.org/abs/2305.08227
- RNNoise: https://arxiv.org/abs/1709.08243
- MUSAN: https://arxiv.org/abs/1510.08484
- Allen and Berkley, image method (1979): https://doi.org/10.1121/1.382599
- pyroomacoustics: https://arxiv.org/abs/1710.04196 and https://github.com/LCAV/pyroomacoustics

### Voice conversion

- AutoVC: https://arxiv.org/abs/1905.05879
- kNN-VC: https://arxiv.org/abs/2305.18975 and https://github.com/bshall/knn-vc
- WavLM: https://arxiv.org/abs/2110.13900
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Seed-VC: https://arxiv.org/abs/2411.09943

### Classification and representation

- PANNs: https://arxiv.org/abs/1912.10211
- AST: https://arxiv.org/abs/2104.01778
- Audio-MAE: https://arxiv.org/abs/2207.06405
- BEATs: https://arxiv.org/abs/2212.09058
- CLAP (Microsoft): https://arxiv.org/abs/2206.04769
- LAION-CLAP: https://arxiv.org/abs/2211.06687
- Speech Commands: https://arxiv.org/abs/1804.03209
- Hello Edge (KWS): https://arxiv.org/abs/1711.07128
- BC-ResNet: https://arxiv.org/abs/2106.04140
- SpecAugment: https://arxiv.org/abs/1904.08779
- ESC-50: https://github.com/karolpiczak/ESC-50

### Anti-spoofing and watermarking

- ASVspoof: https://www.asvspoof.org/
- AASIST: https://arxiv.org/abs/2110.01200
- AudioSeal: https://arxiv.org/abs/2401.17264

### Streaming and turn-taking

- Stivers et al., turn-taking universals (2009): https://doi.org/10.1073/pnas.0903616106
- Emformer: https://arxiv.org/abs/2010.10759
- Silero VAD: https://github.com/snakers4/silero-vad
- Pipecat Smart Turn: https://github.com/pipecat-ai/smart-turn

### Synthesis references for synthetic data

- Klatt (1980), cascade/parallel formant synthesizer: https://doi.org/10.1121/1.383940
- Peterson and Barney (1952), vowel formants: https://doi.org/10.1121/1.1906875

### Courses

- Stanford CS224S (Spring 2025): https://web.stanford.edu/class/cs224s/semesters/2025-spring/syllabus
- CMU 11-751 (Fall 2022): https://www.wavlab.org/activities/2022/11751-2022f/
- CMU 11-785: https://deeplearning.cs.cmu.edu/
- Edinburgh Speech Processing: https://speech.zone/courses/speech-processing/
- Edinburgh ASR (2024-25): https://www.inf.ed.ac.uk/teaching/courses/asr/lectures-2025.html
- JHU CLSP courses: https://www.clsp.jhu.edu/courses/
- MIT 6.345 (2003): https://ocw.mit.edu/courses/6-345-automatic-speech-recognition-spring-2003/
- Hugging Face Audio Course: https://huggingface.co/learn/audio-course/chapter0/introduction
- NVIDIA NeMo Speech: https://github.com/NVIDIA-NeMo/Speech
- Jurafsky and Martin, SLP3: https://web.stanford.edu/~jurafsky/slp3/
- Interspeech 2026 SLM tutorial: https://interspeech2026slm.github.io/

### Datasets

- OpenSLR yesno: https://www.openslr.org/1/
- LibriSpeech: https://www.openslr.org/12/
- Mini LibriSpeech: https://www.openslr.org/31/
- MUSAN: https://www.openslr.org/17/
- RIRS_NOISES: https://www.openslr.org/28/
- LibriTTS-R: https://www.openslr.org/141/
- FSDD: https://github.com/Jakobovski/free-spoken-digit-dataset
- Speech Commands (TFDS card): https://www.tensorflow.org/datasets/catalog/speech_commands
- LJSpeech: https://keithito.com/LJ-Speech-Dataset/
- VCTK 0.92: https://datashare.ed.ac.uk/handle/10283/3443
- TIMIT: https://catalog.ldc.upenn.edu/LDC93S1
- CMUdict: https://github.com/cmusphinx/cmudict

### Tooling

- torchaudio docs (2.11): https://docs.pytorch.org/audio/stable/torchaudio.html
- torchaudio installation (stable ABI note): https://docs.pytorch.org/audio/stable/installation.html
- torchaudio future (issue 3902): https://github.com/pytorch/audio/issues/3902
- torchaudio releases: https://github.com/pytorch/audio/releases
- TorchCodec: https://pypi.org/project/torchcodec/
- torch: https://pypi.org/project/torch/
- librosa: https://pypi.org/project/librosa/
- soundfile: https://pypi.org/project/soundfile/
- transformers: https://pypi.org/project/transformers/
- pyroomacoustics: https://pypi.org/project/pyroomacoustics/

### RunPod

- Pricing: https://www.runpod.io/pricing
- GPU types and pools: https://docs.runpod.io/references/gpu-types
- Endpoint settings: https://docs.runpod.io/serverless/endpoints/endpoint-configurations
- Operation reference: https://docs.runpod.io/serverless/endpoints/operation-reference
- Streaming and aggregate outputs: https://docs.runpod.io/serverless/development/aggregate-outputs
- Cached models: https://docs.runpod.io/serverless/endpoints/model-caching
- Volume cache: https://docs.runpod.io/serverless/development/volume-cache
- REST create endpoint: https://docs.runpod.io/api-reference/endpoints/POST/endpoints
- Serverless billing: https://docs.runpod.io/serverless/pricing

### Browser

- getUserMedia (MDN): https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- AnalyserNode (MDN): https://developer.mozilla.org/en-US/docs/Web/API/AnalyserNode
- wavesurfer.js: https://github.com/katspaugh/wavesurfer.js
