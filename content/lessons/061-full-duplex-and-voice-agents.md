---
title: Full-duplex models and voice agents
slug: 061-full-duplex-and-voice-agents
part: "Part 8 — Beyond recognition and synthesis"
summary: Build a voice agent's streams, depth step, streaming convolution, and turn-taking, and see where its latency goes.
minutes: 80
gpu: false
objectives:
  - Break a voice agent's response time into its parts, and compare a cascade's latency with a full-duplex model's.
  - Lay out Moshi's parallel text and audio streams, with the inner monologue and the acoustic delay.
  - Interleave text and speech tokens at a fixed ratio, and name the other ways speech LLMs combine the two.
  - Score and sample frames with an RQ-Transformer's depth step, and check the factorization against the joint probability.
  - Stream a causal convolution chunk by chunk with the same output as offline, and run a turn-taking state machine with barge-in.
prereqs:
  - 033-streaming-encoders
  - 048-codec-language-models
  - 060-spoken-language-models
exercise: 061-full-duplex-and-voice-agents
---

# Full-duplex models and voice agents

People take turns fast. Across languages, the gap between one person stopping and the next starting is about 200 ms. A voice assistant that waits a second before it answers feels slow, and one that can't be interrupted feels rude. Building a machine that converses, rather than one that answers, is mostly a problem of time.

A voice agent can run a VAD, a recognizer, a language model, a synthesizer, and a codec, and each adds delay. You can chain them and fight for every millisecond, or train one model that listens and speaks at once, which is what *full duplex* means. Moshi, from Kyutai, is the clearest example of the second kind. In this final chapter you build the parts of both: Moshi's parallel token streams and its depth step, a text-and-speech interleaver, a convolution that streams without changing its output, and the state machine that decides when a cascaded agent speaks.

## What you already know

- A codec turns 80 ms of audio into one frame of $Q$ codebook tokens at 12.5 Hz, as Mimi does.
- The delay pattern shifts codebook $k$ right by $k$ steps so a model predicts a whole column per step.
- A causal convolution pads only on the left, so output $t$ depends on inputs up to $t$.
- An energy VAD marks each 10 ms frame as speech or silence.
- An adapter lets a text language model read speech but not produce it.

## Where the time goes

A *cascaded* agent chains separate models, and its response time is a sum:

$$
T_{\text{resp}} = T_{\text{endpoint}} + T_{\text{ASR}} + T_{\text{LLM}} + T_{\text{TTS}} + T_{\text{network}} + T_{\text{buffer}}
$$

Read it aloud: the silence the agent waits for before it decides you're done, plus the recognizer's final result, the language model's time to its first token, the synthesizer's time to its first chunk of audio, the network, and the playback buffer. Figure 1 draws the budget the Run uses.

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

Streaming every stage helps: a streaming recognizer, from [Streaming encoders](/lessons/033-streaming-encoders), has the transcript ready when the endpoint fires, and the synthesizer starts on the first few words. The endpoint wait is harder. Cut it to 200 ms and the agent interrupts every pause for breath, because from silence alone it can't tell a pause from the end of a turn.

```demo
{"id": "latency-timeline", "caption": "Lower the VAD silence and turn off streaming ASR, and compare the cascade's first audio with the full-duplex row and the 200 ms human gap."}
```

> [!KEY]
> A cascade's latency is the sum of its stages, and the endpoint wait usually dominates. You can shrink every stage but that one, because the agent can't know you're done until you stay quiet.

## Listen and speak at once

Moshi (Défossez et al., 2024) has no turns. Every 80 ms, it reads one frame of the user's audio and writes one frame of its own, whether anyone is talking or not. Silence, overlaps, backchannels, and interruptions are all ordinary frames, learned from two-channel recordings of real conversations. Each frame holds $K = 2Q + 1 = 17$ tokens with $Q = 8$: one text token, 8 Mimi tokens for Moshi's voice, and 8 for the user's. Mimi's first codebook is semantic, distilled from WavLM, and the other 7 are acoustic.

### The inner monologue

The text token is Moshi's own transcript, aligned in time with its speech. When a word starts in the audio, the stream holds that word's tokens, then `PAD` until the next word, with an `EPAD` ("end of padding") right before the next word. Predicting text first gives the model a plan for what it says. For three words that start at frames 2, 7, and 13, with 2, 1, and 3 tokens, the 20-frame text stream is the following, with 0 for `PAD` and 1 for `EPAD`.

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

Moshi's acoustic codebooks lag its semantic codebook by $\tau$ frames, so the model commits to what it says before it renders how it sounds. The paper uses $\tau = 2$ in pretraining and $\tau = 1$ afterward. Stream $k$ of the grid is shifted right by its delay $\delta_k$:

$$
D[k, t + \delta_k] = C[k, t], \qquad \delta = (\underbrace{0}_{\text{text}},\ \underbrace{0, \tau, \dots, \tau}_{\text{Moshi}},\ \underbrace{0, \tau, \dots, \tau}_{\text{user}})
$$

Read it aloud: the text and the two semantic streams stay in place, and all 14 acoustic streams move $\tau$ frames later. It is the delay pattern from [Codec language models](/lessons/048-codec-language-models), with one delay per stream. The latency this costs is one frame plus the delay, $T_{\text{duplex}} = (1 + \tau) / f_{\text{frame}}$. With $f_{\text{frame}} = 12.5$ Hz and $\tau = 1$, that is $2 / 12.5 = 160$ ms, the theoretical latency the paper states. It measures 200 ms in practice.

```python
out = np.full((K, T + max(delays)), pad)
for k, d in enumerate(delays):
    out[k, d : d + T] = codes[k]
```

```demo
{"id": "delay-pattern", "caption": "Choose Moshi streams and set the acoustic delay to 1 and then 2 frames, and notice that each frame's semantic token is always produced before its acoustic tokens."}
```

## Interleave text and speech

Moshi keeps text and audio in parallel streams. Other systems put them in one sequence. The following table lists the main strategies.

| Strategy | How text and speech relate | Example |
|---|---|---|
| Sequential | Full text answer, then its speech tokens | SpeechGPT |
| Fixed ratio | Alternate $a$ text tokens and $b$ speech tokens | GLM-4-Voice, 13 and 26 |
| Parallel heads | Text and audio in one step, audio lagging | Mini-Omni, Kimi-Audio |
| Time-aligned | Text aligned to audio frames with padding | Moshi, DSM |
| Thinker-Talker | A separate speech decoder reads the LLM's hidden states | Qwen2.5-Omni |

GLM-4-Voice's tokenizer runs at 12.5 Hz, so 26 speech tokens are about 2 seconds of audio, while 13 text tokens are several words. Text stays ahead, and speech is generated with its words already written. With 20 text tokens and 60 speech tokens, the sequence is text 0 to 12, speech 0 to 25, text 13 to 19, speech 26 to 51, and the last 8 speech tokens.

## The depth step

With 17 tokens per frame, flattening a 10-second reply into one sequence takes $125 \times 17 = 2{,}125$ steps of a 7-billion-parameter model. Moshi's *RQ-Transformer* runs the large *Temporal Transformer* once per frame to get a context $z_s$, and a small *Depth Transformer*, 6 layers of width 1,024, predicts that frame's tokens one by one:

$$
p(V_s \mid V_{<s}) = \prod_{k=1}^{K} p\!\left( V_{s,k} \mid z_s,\ V_{s,<k} \right), \qquad z_s = \text{Temporal}(V_{<s})
$$

Read it aloud: the probability of frame $s$ is the product, over its $K$ tokens, of each token's probability given the frame's context and the tokens already chosen in the frame. The context sees only earlier frames. With it, 10 seconds costs 125 large steps plus 2,125 tiny ones.

```python
log_p = sum(log_softmax(depth_fn(z, frame[:k]))[frame[k]] for k in range(K))
```

Take $K = 2$ binary tokens with $p(V_1 = 0) = 0.7$, $p(V_2 = 0 \mid V_1 = 0) = 0.9$, and $p(V_2 = 0 \mid V_1 = 1) = 0.2$. The four joint probabilities are $0.63$, $0.07$, $0.06$, and $0.24$, and they sum to 1. The checks verify exactly that for your code: summed over every possible frame, and every possible two-frame sequence, the factored probabilities give 1.

> [!KEY]
> The RQ-Transformer is the chain rule applied twice: over frames with a large model, and over the tokens inside a frame with a small one. It is exact, and it makes the large model's step count equal to the number of frames.

## Stream a causal convolution

Mimi and every streaming vocoder run convolutions on audio that arrives in chunks. A causal convolution with kernel $K$ and dilation $d$ needs the last $(K - 1) d$ input samples to compute its next output. Keep them as *state* between chunks, starting from zeros, and chunked output equals offline output exactly:

```python
padded = np.concatenate([state, chunk], axis=1)
out = bias[:, None] + sum(w[:, :, j] @ padded[:, j * d : j * d + n] for j in range(K))
state = padded[:, -(K - 1) * d :]
```

Take one channel, $K = 2$, $d = 1$, and $y_t = 0.5\, x_{t-1} + x_t$, with input $(2, 4, 6, 8)$. Offline, the output is $(2, 5, 8, 11)$. In two chunks, the first gives $(2, 5)$ and stores the state $(4)$. The second computes $0.5 \cdot 4 + 6 = 8$ and $0.5 \cdot 6 + 8 = 11$. The two outputs are equal.

> [!WARNING]
> Padding each chunk with zeros instead of the stored state corrupts the first $(K - 1) d$ outputs of every chunk. At 20 ms chunks, that error repeats 50 times a second, and the ear hears it as a buzz.

## Take turns

A cascaded agent needs rules for when to speak. The exercise's state machine has five states: `LISTENING`, `USER_SPEAKING`, `ENDPOINT_PENDING`, `RESPONDING`, and `AGENT_SPEAKING`. Two counters drive it: consecutive speech frames and consecutive silent frames.

- Speech counts only after `min_speech` frames in a row, so a short "mm-hm" doesn't take the turn.
- The endpoint fires after `endpoint` silent frames. A speech frame in between resets the wait.
- The agent starts speaking `response` frames after the endpoint.
- Detected speech while the agent talks is a *barge-in*: stop the audio at once and listen.

With 10 ms frames, `min_speech` 10, `endpoint` 30, and `response` 20, trace a user who speaks from frame 10 to 59, clicks at frames 62 and 63, says "mm-hm" for 5 frames at 150, and interrupts at 200. Speech is detected at frame 19. The click restarts the silence count, so the endpoint fires at frame 93, and the agent starts at 113. The "mm-hm" is too short to count. The interruption reaches 10 frames at 209, and the agent stops there.

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

## Where to go next

This is the last chapter of the course. You built every piece such a model is made of: the spectrogram and the mel front end, the recognizer and its decoder, the vocoder, the codec, the token model, and the flow that turns noise into speech. The following systems are good places to continue, each with open code, weights, or papers:

- **Moshi and Mimi** (Kyutai, 2024): the full-duplex model this chapter took apart.
- **Delayed streams modeling** (Kyutai, 2025): one framework for streaming ASR and TTS.
- **Qwen2.5-Omni** (2025) and **GLM-4-Voice** (2024): Thinker-Talker and fixed-ratio interleaving.

> [!RECAP]
> - A cascade's response time is a sum of stages; Moshi's algorithmic latency is $(1 + \tau) / 12.5$ Hz, 160 ms at $\tau = 1$.
> - Moshi's 17 streams are text, 8 Moshi codebooks, and 8 user codebooks, with acoustic codebooks delayed by $\tau$.
> - The inner monologue holds each word's tokens from its start frame, `PAD` elsewhere, and `EPAD` before each word.
> - The RQ-Transformer multiplies per-token probabilities inside a frame, given a context from earlier frames.
> - A streaming convolution carries $(K - 1) d$ samples of state; a turn-taking machine counts speech and silence runs.

## Build it step by step

Build the functions in the order the checks test them.

1. **`inner_monologue(word_starts, word_tokens, n_frames, pad, epad)`** returns `(n_frames,)`. Fill with `pad`, write the tokens, then place `epad` before each start that has a free frame. The common mistake is writing `epad` first and overwriting a token.
2. **`moshi_delays(n_q, tau)`** returns the 17 delays for `n_q = 8`.
3. **`delay_streams`** and **`undelay_streams`** shift each row by its delay and back.
4. **`interleave_fixed_ratio`** alternates slices until both run out, with speech ids offset.
5. **`frame_log_prob(z, frame_codes, depth_fn)`** sums one log softmax per codebook, with the prefix `frame_codes[:k]`.
6. **`sequence_log_prob`** sums frames with `temporal_fn(codes[:s])`. The common mistake is passing `codes[:s + 1]`, which lets a frame see itself.
7. **`sample_frame`** samples each codebook with `rng.choice` and appends it to the prefix.
8. **`causal_conv1d`** pads $(K - 1) d$ zeros on the left and sums $K$ shifted matrix products. **`StreamingConv1d`** stores the state in `__init__` and updates it in `step`. Handle a chunk of length 0.
9. **`response_latency_ms`** and **`full_duplex_latency_ms`** implement the two latency formulas.
10. **`turn_taking`** returns the state after each frame and the events. Update both run lengths first, then apply one transition.

## Where this breaks

- **The endpoint guess.** A fixed silence wait either interrupts slow speakers or makes everyone wait. Semantic endpointing and full-duplex models attack it from both sides.

## The exercise

You build `inner_monologue`, `moshi_delays`, `delay_streams`, `undelay_streams`, and `interleave_fixed_ratio` for the token streams; `frame_log_prob`, `sequence_log_prob`, and `sample_frame` for the depth step; `causal_conv1d` and `StreamingConv1d` for streaming; and `response_latency_ms`, `full_duplex_latency_ms`, and `turn_taking` for the agent. Press **Run** to hear a simulated conversation in stereo, the user on the left and the agent on the right, with an "mm-hm" the agent ignores and a barge-in that cuts it off.

The checks test the monologue and delays on hand-built grids and their round trip, interleaving at 13 to 26 and 2 to 1, and that your factored probabilities sum to 1 over every frame and every two-frame sequence. Your sampler must match the model's distribution, your streaming convolution must match the offline one for chunks of 0, 1, and uneven sizes, and the last checks confirm the latency formulas and the exact events and states of the trace in this chapter.

> [!KEY]
> A voice agent that converses is a problem of time. A cascade adds its stages and waits for silence to end a turn, while a full-duplex model runs every frame, with text, delayed audio streams, and a small depth step that keep each step cheap enough to stay in step with the speaker. That is the course's last piece: a model that listens and speaks at once, built from the sampled waveform up.
