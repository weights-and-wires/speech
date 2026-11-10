---
title: Full-duplex models and voice agents
slug: 069-full-duplex-and-voice-agents
part: "Part 8 — Beyond recognition and synthesis"
summary: Build a voice agent's streams, depth step, streaming convolution, and turn-taking, then score its backchannels, echo, gaps, overlaps, and latency budget.
minutes: 80
gpu: false
objectives:
  - Break a voice agent's response time into its parts, and compare a cascade's latency with a full-duplex model's.
  - Lay out Moshi's parallel text and audio streams, with the inner monologue and the acoustic delay.
  - Interleave text and speech tokens at a fixed ratio, and name the other ways speech LLMs combine the two.
  - Score and sample frames with an RQ-Transformer's depth step, and check the factorization against the joint probability.
  - Stream a causal convolution chunk by chunk with the same output as offline, and run a turn-taking state machine with barge-in.
  - Tell a backchannel from a barge-in, say why echo breaks a voice agent, and compute gaps, overlaps, and takeover rates from a two-channel timeline.
prereqs:
  - 011-microphone-arrays
  - 035-streaming-encoders
  - 054-codec-language-models
  - 068-spoken-language-models
exercise: 069-full-duplex-and-voice-agents
---

# Full-duplex models and voice agents

People take turns fast. Across languages, the gap between one person stopping and the next starting is about 200 ms. A voice assistant that waits a second feels slow, and one that can't be interrupted feels rude. Conversation is mostly a problem of time.

A voice agent can chain a VAD, a recognizer, a language model, and a synthesizer. Or it can be one model that listens and speaks at once, which is what *full duplex* means; Moshi, from Kyutai, is the clearest example. This final chapter builds parts of both, and the metrics that show whether a conversation flows.

## What you already know

- A codec turns 80 ms of audio into one frame of $Q$ codebook tokens at 12.5 Hz, as Mimi does, and a delay pattern shifts codebook $k$ right by $k$ steps.
- A causal convolution pads only on the left, so output $t$ depends on inputs up to $t$.
- An energy VAD marks each 10 ms frame as speech or silence.
- NLMS echo cancellation subtracts a loudspeaker's echo from a microphone signal.

## Where the time goes

A *cascaded* agent's response time is a sum:

$$
T_{\text{resp}} = T_{\text{endpoint}} + T_{\text{ASR}} + T_{\text{LLM}} + T_{\text{TTS}} + T_{\text{network}} + T_{\text{buffer}}
$$

Read it aloud: the silence the agent waits for before it decides you're done, plus the recognizer's final result, the language model's time to its first token, the synthesizer's time to its first audio, the network, and the playback buffer.

```figure
<svg viewBox="0 0 600 170" role="img" aria-label="A stacked bar of 1,000 milliseconds: endpoint silence 500, final transcript 50, first LLM token 250, first TTS chunk 120, network 40, playback buffer 40">
  <rect x="40" y="50" width="250" height="40" class="fill-b"/>
  <rect x="290" y="50" width="25" height="40" class="fill-a"/>
  <rect x="315" y="50" width="125" height="40" class="fill-c"/>
  <rect x="440" y="50" width="60" height="40" class="fill-d"/>
  <rect x="500" y="50" width="20" height="40" class="fill-muted"/>
  <rect x="520" y="50" width="20" height="40" class="fill-sunk stroke-muted"/>
  <text x="130" y="76" class="fig-bold">endpoint 500</text>
  <text x="350" y="76" class="fig-bold">LLM 250</text>
  <text x="446" y="76" class="fig-small">TTS 120</text>
  <text x="270" y="112" class="fig-small">ASR 50</text>
  <text x="470" y="112" class="fig-small">network 40 + buffer 40</text>
  <text x="40" y="140" class="fig-small">0 ms</text><text x="505" y="140" class="fig-small">1,000 ms</text>
  <text x="200" y="160">time since you stopped speaking (ms)</text>
</svg>
Figure 1. Half of a cascade's 1,000 ms is the endpoint wait, the time the agent needs to be sure you are finished.
```

Streaming helps every stage: a streaming recognizer, from [Streaming encoders](/lessons/035-streaming-encoders), has the transcript ready at the endpoint. The endpoint wait is harder. Cut it to 200 ms and the agent interrupts every pause for breath, because silence alone can't separate a pause from the end of a turn.

The sum also prices a target. To answer in 600 ms, the other five terms cost $50 + 250 + 120 + 40 + 40 = 500$ ms, which leaves 100 ms for the endpoint wait, too short for a speaker to breathe.

```demo
{"id": "latency-timeline", "caption": "Lower the VAD silence and turn off streaming ASR, and compare the cascade's first audio with the full-duplex row and the 200 ms human gap."}
```

> [!KEY]
> A cascade's latency is the sum of its stages, and the endpoint wait usually dominates. You can shrink every stage but that one, because the agent can't know you're done until you stay quiet.

### Cascade or native speech-to-speech

A native speech-to-speech model such as Moshi maps the user's audio tokens to the agent's, with no endpoint wait and no transcript between stages. It answers in 160 ms in theory and 200 ms measured, against 1,000 ms in Figure 1, and learns interruptions from two-channel data instead of VAD rules. The price is control: there is no transcript between stages to read.

## Listen and speak at once

Moshi (Défossez et al., 2024) has no turns. Every 80 ms, it reads one frame of the user's audio and writes one frame of its own, whether anyone is talking or not. Silence, overlaps, and interruptions are ordinary frames, learned from two-channel recordings. Each frame holds $K = 2Q + 1 = 17$ tokens with $Q = 8$: one text token, 8 Mimi tokens for Moshi's voice, and 8 for the user's. Mimi's first codebook is semantic, and the other 7 are acoustic.

### The inner monologue

The text token is Moshi's own transcript, aligned in time with its speech. When a word starts in the audio, the stream holds that word's tokens, then `PAD` until the next word, with an `EPAD` ("end of padding") right before the next word. For three words that start at frames 2, 7, and 13, with 2, 1, and 3 tokens, the 20-frame text stream is the following, with 0 for `PAD` and 1 for `EPAD`.

```figure
<svg viewBox="0 0 640 130" role="img" aria-label="Twenty cells holding 0 1 101 102 0 0 1 103 0 0 0 0 1 104 105 106 0 0 0 0, with word tokens in orange, EPAD in blue, and PAD in gray">
  <g class="fig-small">
    <rect x="20" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="30" y="59">0</text>
    <rect x="48" y="40" width="28" height="28" class="fill-soft-a stroke-a"/><text x="58" y="59">1</text>
    <rect x="76" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="80" y="59">101</text>
    <rect x="104" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="108" y="59">102</text>
    <rect x="132" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="142" y="59">0</text>
    <rect x="160" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="170" y="59">0</text>
    <rect x="188" y="40" width="28" height="28" class="fill-soft-a stroke-a"/><text x="198" y="59">1</text>
    <rect x="216" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="220" y="59">103</text>
    <rect x="244" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="254" y="59">0</text>
    <rect x="272" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="282" y="59">0</text>
    <rect x="300" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="310" y="59">0</text>
    <rect x="328" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="338" y="59">0</text>
    <rect x="356" y="40" width="28" height="28" class="fill-soft-a stroke-a"/><text x="366" y="59">1</text>
    <rect x="384" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="388" y="59">104</text>
    <rect x="412" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="416" y="59">105</text>
    <rect x="440" y="40" width="28" height="28" class="fill-soft-b stroke-b"/><text x="444" y="59">106</text>
    <rect x="468" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="478" y="59">0</text>
    <rect x="496" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="506" y="59">0</text>
    <rect x="524" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="534" y="59">0</text>
    <rect x="552" y="40" width="28" height="28" class="fill-sunk stroke-grid"/><text x="562" y="59">0</text>
  </g>
  <text x="76" y="96" class="fig-small">word 1 starts, frame 2</text>
  <text x="216" y="96" class="fig-small">word 2, frame 7</text>
  <text x="384" y="96" class="fig-small">word 3, frame 13</text>
  <text x="20" y="28" class="fig-small">frame 0</text><text x="540" y="28" class="fig-small">frame 19</text>
  <text x="200" y="120">frame index (80 ms each)</text>
</svg>
Figure 2. Each word's tokens start at its frame (orange), an EPAD (1, blue) sits just before each word, and PAD (0, gray) fills the rest.
```

### Delays across streams

Moshi's acoustic codebooks lag its semantic codebook by $\tau$ frames ($\tau = 1$ after pretraining), so the model commits to what it says before it renders how it sounds. Stream $k$ of the grid shifts right by its delay $\delta_k$:

$$
D[k, t + \delta_k] = C[k, t], \qquad \delta = (\underbrace{0}_{\text{text}},\ \underbrace{0, \tau, \dots, \tau}_{\text{Moshi}},\ \underbrace{0, \tau, \dots, \tau}_{\text{user}})
$$

Read it aloud: the text and the two semantic streams stay in place, and all 14 acoustic streams move $\tau$ frames later, the delay pattern from [Codec language models](/lessons/054-codec-language-models). The latency this costs is one frame plus the delay, $T_{\text{duplex}} = (1 + \tau) / f_{\text{frame}}$. With $f_{\text{frame}} = 12.5$ Hz and $\tau = 1$, that is $2 / 12.5 = 160$ ms, the theoretical latency the paper states.

```python
out = np.full((K, T + max(delays)), pad)
for k, d in enumerate(delays):
    out[k, d : d + T] = codes[k]
```

```demo
{"id": "delay-pattern", "caption": "Choose Moshi streams and set the acoustic delay to 1 and then 2 frames, and notice that each frame's semantic token is always produced before its acoustic tokens."}
```

## Interleave text and speech

Moshi aligns text to audio frames. Other systems use one sequence. SpeechGPT writes the full text answer, then its speech tokens. GLM-4-Voice alternates 13 text and 26 speech tokens. Mini-Omni and Kimi-Audio predict both in one step with the audio lagging, and Qwen2.5-Omni's speech decoder reads the language model's hidden states.

At 12.5 Hz, 26 speech tokens are about 2 seconds of audio, and 13 text tokens are several words. With 20 text tokens and 60 speech tokens, the sequence is text 0 to 12, speech 0 to 25, text 13 to 19, speech 26 to 51, and the last 8 speech tokens.

## The depth step

A 10-second reply flattened into one sequence takes $125 \times 17 = 2{,}125$ steps of a 7-billion-parameter model. Moshi's *RQ-Transformer* runs the large *Temporal Transformer* once per frame to get a context $z_s$, and a small *Depth Transformer*, 6 layers of width 1,024, predicts that frame's tokens one by one:

$$
p(V_s \mid V_{<s}) = \prod_{k=1}^{K} p\!\left( V_{s,k} \mid z_s,\ V_{s,<k} \right), \qquad z_s = \text{Temporal}(V_{<s})
$$

Read it aloud: the probability of frame $s$ is the product, over its $K$ tokens, of each token's probability given the frame's context and the tokens already chosen. The context sees only earlier frames, so 10 seconds costs 125 large steps plus 2,125 tiny ones.

Take $K = 2$ binary tokens with $p(V_1 = 0) = 0.7$, $p(V_2 = 0 \mid V_1 = 0) = 0.9$, and $p(V_2 = 0 \mid V_1 = 1) = 0.2$. The four joint probabilities are $0.63$, $0.07$, $0.06$, and $0.24$, and they sum to 1. The checks confirm that your factored probabilities sum to 1.

> [!KEY]
> The RQ-Transformer is the chain rule applied twice: over frames with a large model, and over the tokens inside a frame with a small one. It is exact, and it makes the large model's step count equal to the number of frames.

## Stream a causal convolution

A streaming codec or vocoder gets audio in chunks. A causal convolution with kernel $K$ and dilation $d$ needs the last $(K - 1) d$ input samples to compute its next output. Keep them as *state* between chunks, starting from zeros, and chunked output equals offline output exactly:

```python
padded = np.concatenate([state, chunk], axis=1)
out = bias[:, None] + sum(w[:, :, j] @ padded[:, j * d : j * d + n] for j in range(K))
state = padded[:, -(K - 1) * d :]
```

Take $K = 2$, $d = 1$, and $y_t = 0.5\, x_{t-1} + x_t$ with input $(2, 4, 6, 8)$. Offline, the output is $(2, 5, 8, 11)$. In two chunks, the first gives $(2, 5)$ and stores the state $(4)$. The second computes $0.5 \cdot 4 + 6 = 8$ and $0.5 \cdot 6 + 8 = 11$, equal to the offline output.

> [!WARNING]
> Padding each chunk with zeros instead of the stored state corrupts the first $(K - 1) d$ outputs of every chunk. At 20 ms chunks, that is a 50 Hz buzz.

## Take turns

A cascaded agent needs rules for when to speak. The exercise's state machine has five states: `LISTENING`, `USER_SPEAKING`, `ENDPOINT_PENDING`, `RESPONDING`, and `AGENT_SPEAKING`. Two counters drive it: consecutive speech and silent frames.

- Speech counts only after `min_speech` frames in a row, so a short "mm-hm" doesn't take the turn.
- The endpoint fires after `endpoint` silent frames, and a speech frame resets the wait.
- The agent starts `response` frames after the endpoint.
- Detected speech while the agent talks is a *barge-in*: stop at once and listen.

With 10 ms frames, `min_speech` 10, `endpoint` 30, and `response` 20, trace a user who speaks from frame 10 to 59, clicks at frames 62 and 63, says "mm-hm" for 5 frames at 150, and interrupts at 200. Speech is detected at frame 19. The click restarts the silence count, so the endpoint fires at 93 and the agent starts at 113. The "mm-hm" is too short to count, and the interruption reaches 10 frames at 209, where the agent stops.

```figure
<svg viewBox="0 0 640 190" role="img" aria-label="A timeline from frame 0 to 220 with the user speaking from 10 to 59, clicks at 62 and 63, speech detected at 19, endpoint at 93, agent starting at 113, a short mm-hm at 150, an interruption at 200, and the agent stopping at 209">
  <line x1="30" y1="120" x2="610" y2="120" class="stroke-muted"/>
  <rect x="30" y="80" width="90" height="30" class="fill-soft-a stroke-a"/><text x="40" y="100" class="fig-small">user speaks</text>
  <rect x="127" y="85" width="6" height="20" class="fill-muted"/>
  <rect x="165" y="80" width="130" height="30" class="fill-soft-c stroke-c"/><text x="176" y="100" class="fig-small">wait 30, then 20</text>
  <rect x="295" y="80" width="130" height="30" class="fill-soft-b stroke-b"/><text x="305" y="100" class="fig-small">agent speaks</text>
  <rect x="470" y="90" width="14" height="20" class="fill-sunk stroke-grid"/>
  <rect x="520" y="80" width="40" height="30" class="fill-soft-a stroke-a"/>
  <text x="26" y="140" class="fig-small">0</text>
  <text x="40" y="60" class="fig-small">19 speech detected</text>
  <text x="160" y="140" class="fig-small">93 endpoint</text>
  <text x="285" y="140" class="fig-small">113 agent starts</text>
  <text x="450" y="140" class="fig-small">150 mm-hm: ignored</text>
  <text x="500" y="60" class="fig-small">200 barge-in, 209 stop</text>
  <text x="220" y="182">time (frames, 10 ms each)</text>
</svg>
Figure 3. The click at frames 62 and 63 restarts the silence count, so the endpoint fires at 93 instead of 89. The five-frame "mm-hm" never reaches 10 frames, so the agent keeps talking, and the real interruption stops it at 209.
```

### Backchannel or barge-in?

The `min_speech` rule uses duration alone, so a quiet "yeah" of 12 frames stops the agent and a sharp "stop" of 5 doesn't. Full-Duplex-Bench (Lin et al., 2025) calls speech a *backchannel* when it lasts under 1 second and holds fewer than two words. Adding level, in dBFS (0 is full scale), gives the exercise's `classify_overlap`:

```python
if duration_s >= 1.0: return "barge_in"                # holding the floor
if has_words and level_db >= -35.0: return "barge_in"  # a short, loud "stop"
return "backchannel"                                   # "mm-hm", a quiet "yeah"
```

A full-duplex model needs no such rule. It reads the user's stream beside its own every 80 ms, and two-channel training taught it that an "mm-hm" leaves its words flowing and a "wait" is followed by silence.

### Hear your own voice

The agent's audio reaches the user's microphone through their loudspeaker. Without cancellation, the VAD hears speech while the agent talks, reads it as a barge-in, and the agent stops itself mid-sentence. A full-duplex model has the same problem, because its user stream would hold its own voice. Echo cancellation fixes it: an adaptive filter that knows what the agent played subtracts the predicted echo. [Microphone arrays](/lessons/011-microphone-arrays) builds it with NLMS, and its double-talk rule applies: freeze the filter while the user speaks over the agent.

## Measure turn-taking

Timing shows whether a conversation flows. Heldner and Edlund (2010) and dGSLM (Nguyen et al., 2023) measure it from two channels, each a list of speech segments `(start, end)` in seconds. A *gap* is silence between one speaker stopping and the other starting, an *overlap* is both talking at a speaker change, and a *pause* is silence inside one speaker's turn. dGSLM's floor transfer offset merges both into one signed number, the next turn's start minus the previous turn's end, positive for a gap.

Work one timeline. The user speaks at 0.0 to 2.0, 2.9 to 3.4, and 5.8 to 7.0 s. The agent speaks at 2.3 to 3.0, 3.3 to 5.0, and 6.8 to 8.0 s.

1. The user's 0.5 s segment starts inside the agent's speech and lasts under 1 s, so it is a backchannel and drops out. The agent's two segments become one turn: U[0, 2], A[2.3, 5.0], U[5.8, 7.0], A[6.8, 8.0].
2. The offsets are $0.3$, $0.8$, and $-0.2$ s: gaps of 0.3 and 0.8 s, and an overlap of 0.2 s.

Report the median and a percentile, because gap durations skew long (Heldner and Edlund, 2010): here 0.55 s and 0.75 s at the 90th percentile. The *turn-change rate* is 3 changes in 8 s, or 22.5 per minute. Counting the backchannel as a turn gives 5 changes, 37.5 per minute, and measuring from the previous start gives a first gap of 2.3 s.

Full-Duplex-Bench defines a different *takeover rate* (TOR): per sample, $TO_i$ is 1 when the model produces speech that is neither silence nor a backchannel, and TOR is the mean of $TO_i$ over $N$ samples. It should be low when the user pauses or backchannels, and near 1 after an interruption.

> [!RECAP]
> - A cascade's response time is a sum of stages; Moshi's algorithmic latency is $(1 + \tau) / 12.5$ Hz, 160 ms at $\tau = 1$.
> - Moshi's 17 streams are text, 8 Moshi codebooks, and 8 user codebooks, with acoustic codebooks delayed by $\tau$.
> - The RQ-Transformer multiplies per-token probabilities inside a frame, given a context from earlier frames.
> - A streaming convolution carries $(K - 1) d$ samples of state; a turn-taking machine counts speech and silence runs, and a backchannel rule adds words and level.
> - A gap runs from one turn's end to the next turn's start, and TOR is the share of samples where the model takes the turn.

## Build it step by step

Build the functions in the order the checks test them.

1. **`inner_monologue`** fills with `pad`, writes the tokens, then places `epad` before each start that has a free frame. Writing `epad` first overwrites a token.
2. **`moshi_delays`**, **`delay_streams`**, and **`undelay_streams`** build the 17 delays and shift each row by its delay and back. **`interleave_fixed_ratio`** alternates slices until both run out, with speech ids offset.
3. **`frame_log_prob`** sums one log softmax per codebook, with the prefix `frame_codes[:k]`. **`sequence_log_prob`** sums frames with `temporal_fn(codes[:s])`; passing `codes[:s + 1]` lets a frame see itself. **`sample_frame`** samples each codebook with `rng.choice`.
4. **`causal_conv1d`** pads $(K - 1) d$ zeros on the left and sums $K$ shifted matrix products. **`StreamingConv1d`** stores the state in `__init__` and updates it in `step`. Handle a chunk of length 0.
5. **`response_latency_ms`**, **`full_duplex_latency_ms`**, and **`turn_taking`** follow the formulas and the transition list. In `turn_taking`, update both run lengths first, then apply one transition.
6. **`classify_overlap`** tests duration first, then words and level together.
7. **`turn_taking_stats`** drops backchannels, merges consecutive segments of one speaker into turns, and subtracts the previous end from the next start. **`takeover_rate`** averages a per-sample flag.

## Where this breaks

- **The endpoint guess.** A fixed silence wait either interrupts slow speakers or makes everyone wait.
- **Echo.** Cancellation fails when the room changes, and the agent then interrupts itself again.

## The exercise

You write the stream builders, the depth-step functions, the two convolutions, the latency functions, `turn_taking`, `classify_overlap`, `turn_taking_stats`, and `takeover_rate`. Press **Run** to hear a stereo conversation, the user on the left and the agent on the right, with an "mm-hm" the agent ignores and a barge-in that cuts it off. The checks cover every piece, including this chapter's trace and its timeline (gaps 0.3 and 0.8 s, overlap 0.2 s, 22.5 changes per minute).

> [!KEY]
> A cascade adds its stages and waits for silence to end a turn, while a full-duplex model runs every frame, so interruptions and backchannels become things to measure rather than rules to tune.

## Further reading

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer, Patrick Pérez, Hervé Jégou, Edouard Grave, and Neil Zeghidour, 2024.
- [Generative Spoken Dialogue Language Modeling](https://aclanthology.org/2023.tacl-1.15/), Tu Anh Nguyen and colleagues, Transactions of the Association for Computational Linguistics 11, 2023.
- [Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities](https://arxiv.org/abs/2503.04721), Guan-Ting Lin and colleagues, 2025.
- [Pauses, gaps and overlaps in conversations](https://doi.org/10.1016/j.wocn.2010.08.002), Mattias Heldner and Jens Edlund, Journal of Phonetics 38(4), 2010.
