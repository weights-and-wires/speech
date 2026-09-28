---
title: Full-duplex models and voice agents
slug: 42-full-duplex-and-voice-agents
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
  - 19-transducers-and-streaming
  - 32-codec-language-models
  - 41-spoken-language-models
exercise: 42-full-duplex-and-voice-agents
---

# Full-duplex models and voice agents

People take turns fast. Across languages, the gap between one person
stopping and the next starting is about 200 ms. A voice assistant that
waits a second before it answers feels slow, and one that can't be
interrupted feels rude. Building a machine that converses, rather than one
that answers, is mostly a problem of time.

Every piece this course built appears here. A voice agent runs a VAD, a
recognizer, a language model, a synthesizer, and a codec, and each adds
delay. You can chain them and fight for every millisecond, or train one
model that listens and speaks at once, which is what *full duplex* means.
Moshi, from Kyutai, is the clearest example of the second kind.

In this final lesson you build the parts of both: Moshi's parallel token
streams and its depth step, a text-and-speech interleaver, a convolution that
streams without changing its output, and the state machine that decides when
a cascaded agent speaks.

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

Read it aloud: the silence the agent waits for before it decides you're done,
plus the recognizer's final result, the language model's time to its first
token, the synthesizer's time to its first chunk of audio, the network, and
the playback buffer. Each term is measured from the end of the previous one.

The following budget is the one the Run uses. The endpoint wait is the
largest term.

| Stage | Time |
|---|---|
| Endpoint silence | 500 ms |
| Final transcript | 50 ms |
| First LLM token | 250 ms |
| First TTS chunk | 120 ms |
| Network | 40 ms |
| Playback buffer | 40 ms |
| Total | 1,000 ms |

Streaming every stage helps: a streaming recognizer, from lesson 19,
*Transducers and streaming*, has the transcript ready when the endpoint fires, and the
synthesizer starts on the first few words. The endpoint wait is harder. Cut
it to 200 ms and the agent interrupts every pause for breath.

The following demo lays these stages on a timeline.

```demo
{"id": "latency-timeline", "caption": "Lower the VAD silence and turn off streaming ASR, and compare the cascade's first audio with the full-duplex row and the 200 ms human gap."}
```

> [!KEY]
> A cascade's latency is the sum of its stages, and the endpoint wait usually dominates. You can shrink every stage but that one, because the agent can't know you're done until you stay quiet.

## Listen and speak at once

Moshi (Défossez et al., 2024) has no turns. Every 80 ms, it reads one frame
of the user's audio and writes one frame of its own, whether anyone is
talking or not. Silence, overlaps, backchannels, and interruptions are all
ordinary frames, learned from two-channel recordings of real conversations.
An earlier model, dGSLM, showed that turn-taking emerges this way from
unit models trained on the Fisher corpus.

Each frame holds $K = 2Q + 1 = 17$ tokens with $Q = 8$: one text token, 8
Mimi tokens for Moshi's voice, and 8 for the user's. Mimi's first codebook
is semantic, distilled from WavLM, and the other 7 are acoustic.

### The inner monologue

The text token is Moshi's own transcript, aligned in time with its speech.
The paper builds it so that "when a word starts" in the audio, the stream
holds that word's tokens, then `PAD` until the next word, with an `EPAD`
("end of padding") right before the next word. The word timings come from
Whisper's timestamps. Predicting text first gives the model a plan for what
it says, which the paper reports improves its content.

For three words that start at frames 2, 7, and 13, with 2, 1, and 3 tokens,
the 20-frame text stream is the following, with 0 for `PAD` and 1 for
`EPAD`:

```text
0 1 101 102 0 0 1 103 0 0 0 0 1 104 105 106 0 0 0 0
```

Kyutai's later *delayed streams modeling* (DSM) turned the alignment into a
switch. Train one model on time-aligned text and audio. Delay the text
behind the audio and the model becomes streaming ASR. Delay the audio behind
the text and it becomes streaming TTS.

### Delays across streams

Moshi's acoustic codebooks lag its semantic codebook by $\tau$ frames, so the
model commits to what it says before it renders how it sounds. The paper uses
$\tau = 2$ in pretraining and $\tau = 1$ afterward. Stream $k$ of the grid is
shifted right by its delay $\delta_k$:

$$
D[k, t + \delta_k] = C[k, t], \qquad \delta = (\underbrace{0}_{\text{text}},\ \underbrace{0, \tau, \dots, \tau}_{\text{Moshi}},\ \underbrace{0, \tau, \dots, \tau}_{\text{user}})
$$

Read it aloud: the text and the two semantic streams stay in place, and all
14 acoustic streams move $\tau$ frames later. It is the delay pattern of
lesson 32, *Codec language models*, with one delay per stream instead of one
per codebook index. The latency this costs is one frame plus the delay:

$$
T_{\text{duplex}} = \frac{1 + \tau}{f_{\text{frame}}}
$$

With $f_{\text{frame}} = 12.5$ Hz and $\tau = 1$, that is
$2 / 12.5 = 160$ ms, the theoretical latency the paper states. It measures
200 ms in practice.

```python
out = np.full((K, T + max(delays)), pad)
for k, d in enumerate(delays):
    out[k, d : d + T] = codes[k]
```

The following demo builds a delayed grid and shows what each step can see.

```demo
{"id": "delay-pattern", "caption": "Choose Moshi streams and set the acoustic delay to 1 and then 2 frames, and notice that each frame's semantic token is always produced before its acoustic tokens."}
```

## Interleave text and speech

Moshi keeps text and audio in parallel streams. Other systems put them in
one sequence. The following table lists the main strategies.

| Strategy | How text and speech relate | Example |
|---|---|---|
| Sequential | Full text answer, then its speech tokens | SpeechGPT |
| Fixed ratio | Alternate $a$ text tokens and $b$ speech tokens | GLM-4-Voice, 13 and 26 |
| Parallel heads | Text and audio in one step, audio lagging | Mini-Omni, Kimi-Audio |
| Time-aligned | Text aligned to audio frames with padding | Moshi, DSM |
| Thinker-Talker | A separate speech decoder reads the LLM's hidden states | Qwen2.5-Omni |

GLM-4-Voice's tokenizer runs at 12.5 Hz, so 26 speech tokens are about 2
seconds of audio, while 13 text tokens are several words. Text stays ahead,
and speech is always generated with its words already written. With 20 text
tokens and 60 speech tokens, the sequence is text 0 to 12, speech 0 to 25,
text 13 to 19, speech 26 to 51, and the last 8 speech tokens.

## The depth step

With 17 tokens per frame, flattening a 10-second reply into one sequence
takes $125 \times 17 = 2{,}125$ steps of a 7-billion-parameter model. Moshi's
*RQ-Transformer* runs the large *Temporal Transformer* once per frame to get
a context $z_s$, and a small *Depth Transformer*, 6 layers of width 1,024,
predicts that frame's tokens one by one:

$$
p(V_s \mid V_{<s}) = \prod_{k=1}^{K} p\!\left( V_{s,k} \mid z_s,\ V_{s,<k} \right), \qquad z_s = \text{Temporal}(V_{<s})
$$

Read it aloud: the probability of frame $s$ is the product, over its $K$
tokens, of each token's probability given the frame's context and the tokens
already chosen in the frame. The context sees only earlier frames. With it, 10
seconds costs 125 large steps plus 2,125 tiny ones. Sesame CSM and Kyutai
TTS use the same split.

```python
log_p = sum(log_softmax(depth_fn(z, frame[:k]))[frame[k]] for k in range(K))
```

Take $K = 2$ binary tokens with $p(V_1 = 0) = 0.7$, $p(V_2 = 0 \mid V_1 = 0) = 0.9$,
and $p(V_2 = 0 \mid V_1 = 1) = 0.2$. The four joint probabilities are
$0.63$, $0.07$, $0.06$, and $0.24$, and they sum to 1. The checks verify
exactly that for your code: summed over every possible frame, and every
possible two-frame sequence, the factored probabilities give 1.

> [!KEY]
> The RQ-Transformer is the chain rule applied twice: over frames with a large model, and over the tokens inside a frame with a small one. It is exact, and it makes the large model's step count equal to the number of frames.

## Stream a causal convolution

Mimi and every streaming vocoder run convolutions on audio that arrives in
chunks. A causal convolution with kernel $K$ and dilation $d$ needs the last
$(K - 1) d$ input samples to compute its next output. Keep them as *state*
between chunks, starting from zeros, and chunked output equals offline
output exactly:

```python
padded = np.concatenate([state, chunk], axis=1)
out = bias[:, None] + sum(w[:, :, j] @ padded[:, j * d : j * d + n] for j in range(K))
state = padded[:, -(K - 1) * d :]
```

Take one channel, $K = 2$, $d = 1$, and $y_t = 0.5\, x_{t-1} + x_t$, with
input $(2, 4, 6, 8)$. Offline, the output is $(2, 5, 8, 11)$. In two chunks,
the first gives $(2, 5)$ and stores the state $(4)$. The second computes
$0.5 \cdot 4 + 6 = 8$ and $0.5 \cdot 6 + 8 = 11$. The two outputs are equal.

> [!WARNING]
> Padding each chunk with zeros instead of the stored state corrupts the first $(K - 1) d$ outputs of every chunk. At 20 ms chunks, that error repeats 50 times a second, and the ear hears it as a buzz.

## Take turns

A cascaded agent needs rules for when to speak. The exercise's state machine
has five states: `LISTENING`, `USER_SPEAKING`, `ENDPOINT_PENDING`,
`RESPONDING`, and `AGENT_SPEAKING`. Two counters drive it: consecutive
speech frames and consecutive silent frames.

- Speech counts only after `min_speech` frames in a row, so a short "mm-hm" doesn't take the turn.
- The endpoint fires after `endpoint` silent frames. A speech frame in between resets the wait.
- The agent starts speaking `response` frames after the endpoint, the rest of the latency budget.
- Detected speech while the agent talks is a *barge-in*: stop the audio at once and listen.

With 10 ms frames, `min_speech` 10, `endpoint` 30, and `response` 20, trace a
user who speaks from frame 10 to 59, clicks at frames 62 and 63, says "mm-hm"
for 5 frames at 150, and interrupts at 200. Speech is detected at frame 19.
The click restarts the silence count, so the endpoint fires at frame 93, and
the agent starts at 113. The "mm-hm" is too short to count. The interruption
reaches 10 frames at 209, and the agent stops there.

Real agents replace the energy VAD with a neural one, such as the
MIT-licensed Silero VAD, and add *semantic endpointing*: Pipecat's Smart Turn
v3 classifies from the audio whether the speaker has finished, which lets the
endpoint wait shrink without cutting people off.

## Where to go next

This is the last lesson of the course. It began with a tone sampled at a
fixed rate, and it ends with a model that listens and speaks at once. Between
the two, you built every piece such a model is made of: the spectrogram and
the mel front end, the recognizer and its decoder, the vocoder, the codec,
the token model, and the flow that turns noise into speech. The following
systems and resources are good places to continue, each with open code,
weights, or papers:

- **Moshi and Mimi** (Kyutai, 2024): the full-duplex model this lesson took apart, with open code and weights.
- **Delayed streams modeling** (Kyutai, 2025): one framework for streaming ASR and TTS, and Unmute, which wraps a text LLM with Kyutai's STT and TTS into a low-latency cascade.
- **Voxtral Realtime** (Mistral AI, 2026): streaming ASR built on delayed streams modeling, which matches Whisper's quality at a 480 ms delay, with Apache 2.0 weights.
- **Sesame CSM** (2025): a Llama backbone predicts Mimi's first codebook and a small decoder the rest, the depth split in another form. CSM-1B is Apache 2.0.
- **Qwen2.5-Omni and Qwen3-Omni** (2025): Thinker-Talker models. Qwen3-Omni's Thinker is a mixture of experts with 30B parameters, 3B active.
- **GLM-4-Voice** (2024) and **LLaMA-Omni** (2024): fixed-ratio interleaving, and a speech decoder trained with CTC that answers in as little as 226 ms.
- **MiniMind-O** (2026): a whole omni model at about 0.1B parameters with open code, weights, and data, small enough to read end to end.
- **Surveys and a tutorial**: "On the landscape of spoken language models" (TMLR 2025), and the Interspeech 2026 tutorial on spoken language models that follows it.
- **Benchmarks**: VoiceBench for spoken instructions, Big Bench Audio's 1,000 spoken reasoning questions, URO-Bench for speech-to-speech, and HumDial-FDBench for full-duplex dialogue.

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
3. **`delay_streams(codes, delays, pad)`** and **`undelay_streams(delayed, delays, n_frames)`** shift each row by its delay and back.
4. **`interleave_fixed_ratio(text, speech, n_text, n_speech, speech_offset)`** alternates slices until both run out, with speech ids offset.
5. **`frame_log_prob(z, frame_codes, depth_fn)`** sums one log softmax per codebook, with the prefix `frame_codes[:k]`.
6. **`sequence_log_prob(codes, temporal_fn, depth_fn)`** sums frames with `temporal_fn(codes[:s])`. The common mistake is passing `codes[:s + 1]`, which lets a frame see itself.
7. **`sample_frame(z, depth_fn, n_codebooks, rng)`** samples each codebook with `rng.choice` and appends it to the prefix.
8. **`causal_conv1d(x, weight, bias, dilation)`** pads $(K - 1) d$ zeros on the left and sums $K$ shifted matrix products.
9. **`StreamingConv1d`** stores the state in `__init__` and updates it in `step`. Handle a chunk of length 0.
10. **`response_latency_ms`** and **`full_duplex_latency_ms`** implement the two latency formulas.
11. **`turn_taking(...)`** returns the state after each frame and the events. Update both run lengths first, then apply one transition.

## Cost and scaling

The following table compares generation costs for $T$ frames of $K$ tokens.

| Layout | Large-model steps | Small-model steps |
|---|---|---|
| Flattened | $T K$ | none |
| Parallel heads | $T$ | none, but tokens in a frame ignore each other |
| RQ-Transformer | $T$ | $T K$ |
| Fixed-ratio interleave | text plus speech tokens | none |

A full-duplex model must finish each step in less than one frame, 80 ms at
12.5 Hz, or it falls behind the conversation, so that real-time constraint
decides model size. A streaming convolution's state costs
$(K - 1) d$ samples per channel per layer, which is tiny, and a Transformer's
key-value cache grows with the context, which is why Mimi's Transformers use
a 250-frame window.

## Where this breaks

- **The endpoint guess.** A fixed silence wait either interrupts slow speakers or makes everyone wait. Semantic endpointing and full-duplex models attack it from both sides.
- **Echo.** Without echo cancellation, the agent's own voice reaches the microphone and triggers a barge-in.
- **Word timestamps.** The inner monologue needs words aligned to audio, and alignment errors teach the model to speak words at the wrong time.
- **Evaluation.** Benchmarks often transcribe the model's speech and judge the text, which hides prosody and timing, and synthetic test voices flatter models compared with real speech.
- **A duplex model that talks over you.** Learned turn-taking follows its training conversations. When the data's style differs from the user's, the model backchannels too much or yields too little.

## The exercise

You build `inner_monologue`, `moshi_delays`, `delay_streams`,
`undelay_streams`, and `interleave_fixed_ratio` for the token streams;
`frame_log_prob`, `sequence_log_prob`, and `sample_frame` for the depth step;
`causal_conv1d` and `StreamingConv1d` for streaming; and
`response_latency_ms`, `full_duplex_latency_ms`, and `turn_taking` for the
agent. Press **Run** to hear a simulated conversation in stereo: the user on
the left, the agent on the right, with an "mm-hm" the agent ignores and a
barge-in that cuts it off. The Run also draws Moshi's delayed 17-stream grid
and confirms that a streamed convolution matches the offline one exactly.

The checks test the monologue and delays on hand-built grids and their
round trip, interleaving at 13 to 26 and 2 to 1, and that your factored
probabilities sum to 1 over every frame and every two-frame sequence. Your
sampler must match the model's distribution, your causal convolution must
match torch and ignore the future, and your streaming convolution must match
it exactly for chunks of 0, 1, and uneven sizes. The last checks confirm the
latency formulas and the exact events and states of the trace in this
lesson.
