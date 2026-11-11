---
title: Streaming latency and on-device speech
slug: 050-streaming-latency-and-on-device
part: "Part 4 — Neural speech recognition"
summary: Measure real-time factor, the delay of a chunked encoder, first-partial and final latency, partial stability, and an on-device memory budget.
minutes: 40
gpu: false
objectives:
  - Compute the real-time factor and say why RTF below 1 is necessary but not sufficient for streaming.
  - Split a streaming recognizer's latency into algorithmic delay and compute time, per position in a chunk.
  - Read first-partial, finalization, and endpoint-to-final latency from a timeline, measured from when each word ended.
  - Score partial results with the unstable-prefix rate.
  - Check whether a quantized model fits a device's memory and keeps up with real time.
prereqs:
  - 034-transducers
  - 036-streaming-encoders
exercise: 050-streaming-latency-and-on-device
---

# Streaming latency and on-device speech

You say "turn on the lights" to a phone with no network. The words should appear as you speak, the last one should settle soon after you stop, and the whole model must fit in the memory the phone gives one app. A recognizer can be accurate and still fail every one of those tests.

"Fast" hides several different numbers. How much audio can the model process per second of compute? How long does a word sit unseen after you say it? How often does a word on screen change after it appeared? How many megabytes does the model need? This chapter names each number, measures it from a timeline, and checks it against a device.

## What you already know

- A streaming transducer writes tokens as audio arrives, if its encoder does not need the future, from [Transducers](/lessons/034-transducers).
- A chunked encoder makes frame $p$ of a $C$-frame chunk wait $(C - p)$ frames, from [Streaming encoders](/lessons/036-streaming-encoders).
- A voice activity detector decides when speech ends, and its hangover is a wait, from [Voice activity detection](/lessons/020-voice-activity-detection).
- A weight stored at $b$ bits takes $b / 8$ bytes.
- Word error rate scores the final text but says nothing about when the text arrived.

## Two clocks: compute and waiting

Latency has two sources, and they respond to different fixes. **Compute latency** is the time the processor needs to run the model on a chunk. **Algorithmic latency** is the time the model must wait for audio it needs before it can answer, and no faster chip removes it. A chunked encoder with a 160 ms chunk cannot emit the chunk's first frame until 160 ms of audio exists.

Compute is summarized by the **real-time factor**, RTF: seconds of compute per second of audio.

$$
\mathrm{RTF} = \frac{\text{compute seconds}}{\text{audio seconds}}
$$

Read it aloud: an RTF of 0.3 means 3 s of compute handles 10 s of audio. The inverse, $1/\mathrm{RTF}$, is the throughput in seconds of audio per second of compute. In code it is `compute_seconds / audio_seconds`.

An RTF of 1 or more breaks streaming, because the model falls further behind with every chunk and the backlog grows without bound. Below 1, a chunk finishes before the next one arrives. But RTF below 1 is not enough. It says the model keeps up on average and says nothing about how long a single word waits.

> [!WARNING]
> RTF is an average over a whole file. A model with an RTF of 0.5 can still add a half second to one word, if that word lands in a chunk that is slow to compute. Measure the delay of each word, not only the average.

## The algorithmic delay of a chunk

Take chunks of $C$ encoder frames, $L$ extra look-ahead frames, and a hop of `hop_ms` per frame. The first frame of a chunk waits for the other $C - 1$ frames, then for $L$ more, and its own frame lasts one hop. Position $p$, counted from 0, therefore has the delay

$$
\mathrm{delay}(p) = (C - p + L) \cdot \text{hop\_ms}
$$

Work $C = 4$, $L = 2$, and a 40 ms hop. The delays are $(4 + 2) \cdot 40 = 240$, then 200, 160, and 120 ms. The worst is 240 ms and the mean is 180 ms, which equals $((C + 1)/2 + L) \cdot 40$. This is [Streaming encoders](/lessons/036-streaming-encoders) with one extra term, so the shape is the same: a bigger chunk or more look-ahead raises the delay.

Compute adds to it. The model can start only after the chunk and its look-ahead have arrived, and it then runs for $\mathrm{RTF} \cdot C \cdot \text{hop\_ms}$ ms. At $\mathrm{RTF} = 0.5$ the chunk above costs $0.5 \cdot 160 = 80$ ms, so every position pays 80 ms more: 320, 280, 240, and 200 ms. Figure 1 sweeps the chunk size at an RTF of 0.3.

```figure
<svg viewBox="0 0 580 270" role="img" aria-label="Mean latency in milliseconds for chunk sizes 1 to 16 frames, split into algorithmic delay and compute">
<text x="100" y="47" text-anchor="end" class="fig-small">chunk 1</text>
<rect x="110" y="30" width="74.5" height="24" class="fill-a"/>
<rect x="184.5" y="30" width="7.5" height="24" class="fill-b"/>
<text x="198.0" y="47" class="fig-small">132 ms</text>
<text x="100" y="87" text-anchor="end" class="fig-small">chunk 2</text>
<rect x="110" y="70" width="86.9" height="24" class="fill-a"/>
<rect x="196.9" y="70" width="14.9" height="24" class="fill-b"/>
<text x="217.8" y="87" class="fig-small">164 ms</text>
<text x="100" y="127" text-anchor="end" class="fig-small">chunk 4</text>
<rect x="110" y="110" width="111.8" height="24" class="fill-a"/>
<rect x="221.8" y="110" width="29.8" height="24" class="fill-b"/>
<text x="257.6" y="127" class="fig-small">228 ms</text>
<text x="100" y="167" text-anchor="end" class="fig-small">chunk 8</text>
<rect x="110" y="150" width="161.4" height="24" class="fill-a"/>
<rect x="271.4" y="150" width="59.6" height="24" class="fill-b"/>
<text x="337.0" y="167" class="fig-small">356 ms</text>
<text x="100" y="207" text-anchor="end" class="fig-small">chunk 16</text>
<rect x="110" y="190" width="260.8" height="24" class="fill-a"/>
<rect x="370.8" y="190" width="119.2" height="24" class="fill-b"/>
<text x="496.0" y="207" class="fig-small">612 ms</text>
<text x="110" y="250" class="fig-small">blue is algorithmic delay, orange is compute at RTF 0.3 (hop 40 ms, 2 look-ahead frames)</text>
</svg>
Figure 1. Mean latency for chunks of 1 to 16 frames, with a 40 ms hop, 2 look-ahead frames, and RTF 0.3. A chunk of 16 costs 612 ms, 420 of it waiting for audio. Notice that compute grows with the chunk too, because a chunk's compute time is a share of its length.
```

> [!KEY]
> Algorithmic latency comes from the model's design: chunk size, look-ahead, and hop. Compute latency comes from the hardware, as RTF times the chunk length. Report both, because the first needs a different model and the second needs a faster one.

## Latency from a timeline

A product feels slow when a word appears late. So measure each latency from the moment the word was spoken, which is the word's end, and not from the start of a chunk or of the audio. A recognizer cannot output a word before it has heard most of it, so the end of the word is the earliest fair starting line.

Take the utterance "turn on the lights". A forced alignment, from [Forced alignment and word timestamps](/lessons/046-forced-alignment-and-word-timestamps), gives the word ends at 480, 640, 760, and 1,240 ms. The recognizer's log, and an endpointer that fires at 1,640 ms, give Figure 2.

```figure
<svg viewBox="0 0 580 270" role="img" aria-label="Timeline of an utterance: word ends, partial results, the endpoint, and the final result, with the latencies marked">
<line x1="50" y1="150" x2="530" y2="150" class="stroke-muted"/>
<line x1="50.0" y1="146" x2="50.0" y2="154" class="stroke-muted"/>
<text x="50.0" y="168" text-anchor="middle" class="fig-small">0</text>
<line x1="170.0" y1="146" x2="170.0" y2="154" class="stroke-muted"/>
<text x="170.0" y="168" text-anchor="middle" class="fig-small">500</text>
<line x1="290.0" y1="146" x2="290.0" y2="154" class="stroke-muted"/>
<text x="290.0" y="168" text-anchor="middle" class="fig-small">1000</text>
<line x1="410.0" y1="146" x2="410.0" y2="154" class="stroke-muted"/>
<text x="410.0" y="168" text-anchor="middle" class="fig-small">1500</text>
<line x1="530.0" y1="146" x2="530.0" y2="154" class="stroke-muted"/>
<text x="530.0" y="168" text-anchor="middle" class="fig-small">2000</text>
<text x="290" y="188" text-anchor="middle">time since the audio started (ms)</text>
<text x="50" y="16" class="fig-bold fig-small">word ends (what was said)</text>
<circle cx="165.2" cy="150" r="5" class="fill-a"/>
<line x1="165.2" y1="150" x2="165.2" y2="38" class="stroke-a dashed"/>
<text x="165.2" y="34" text-anchor="middle" class="fig-small">turn 480</text>
<circle cx="203.6" cy="150" r="5" class="fill-a"/>
<line x1="203.6" y1="150" x2="203.6" y2="56" class="stroke-a dashed"/>
<text x="203.6" y="52" text-anchor="middle" class="fig-small">on 640</text>
<circle cx="232.4" cy="150" r="5" class="fill-a"/>
<line x1="232.4" y1="150" x2="232.4" y2="38" class="stroke-a dashed"/>
<text x="232.4" y="34" text-anchor="middle" class="fig-small">the 760</text>
<circle cx="347.6" cy="150" r="5" class="fill-a"/>
<line x1="347.6" y1="150" x2="347.6" y2="56" class="stroke-a dashed"/>
<text x="347.6" y="52" text-anchor="middle" class="fig-small">lights 1240</text>
<rect x="179.4" y="205" width="10" height="10" class="fill-b"/>
<rect x="237.0" y="205" width="10" height="10" class="fill-b"/>
<rect x="285.0" y="205" width="10" height="10" class="fill-b"/>
<rect x="400.2" y="205" width="10" height="10" class="fill-b"/>
<rect x="501.0" y="205" width="10" height="10" class="fill-d"/>
<text x="184.4" y="232" text-anchor="middle" class="fig-small">560</text>
<text x="290.0" y="232" text-anchor="middle" class="fig-small">1000</text>
<text x="405.2" y="232" text-anchor="middle" class="fig-small">1480</text>
<text x="506.0" y="232" text-anchor="middle" class="fig-small">final 1900</text>
<line x1="443.6" y1="100" x2="443.6" y2="150" class="stroke-c"/>
<text x="443.6" y="94" text-anchor="middle" class="fig-small">endpoint 1640</text>
<text x="50" y="256" class="fig-small">orange squares are partials, purple is the final. First partial latency 80 ms, finalization 660 ms.</text>
</svg>
Figure 2. Word ends (blue), partial results (orange), the endpoint, and the final (purple). Notice the gap between the last word end at 1,240 ms and the final at 1,900 ms: the endpoint wait is 400 ms of it, and the recognizer needs 260 ms more.
```

Three numbers come from this log:

- **First-partial latency** is the time of the first non-empty partial minus the end of the first word: $560 - 480 = 80$ ms.
- **Finalization latency** is the time of the final minus the end of the last word: $1900 - 1240 = 660$ ms.
- **Endpoint-to-final latency** is the time of the final minus the time the endpointer fired: $1900 - 1640 = 260$ ms.

The last two are linked. Finalization is the endpoint wait, $1640 - 1240 = 400$ ms, plus endpoint-to-final, 260 ms. The endpoint wait belongs to the endpointer: it is the hangover from [Voice activity detection](/lessons/020-voice-activity-detection), and [Full-duplex models and voice agents](/lessons/079-full-duplex-and-voice-agents) prices it in a voice agent's budget. Endpoint-to-final belongs to the recognizer, which must flush the last chunk and rescore.

> [!WARNING]
> Subtracting the chunk start, or the start of the audio, from an event time makes every latency look larger and hides which stage is slow. The log above gives 560 ms for the first partial that way, not 80 ms.

## Partial results change their mind

The orange squares in Figure 2 are **partials**: the best guess so far. They are useful only if they mostly stay put. The log shows "turn on a" at 1,000 ms and "turn on the light" at 1,480 ms, then "turn on the lights". A reader saw "a" and then "the", and saw "light" gain an "s".

The **unstable-prefix rate** counts that. A partial is a prefix of the final text when it agrees with the final from the first word onward. The stable part of a partial is its longest such prefix, and every word after the first mismatch is unstable, because the display rewrites everything from the first changed word.

$$
\text{rate} = \frac{\sum_k \left( |p_k| - \mathrm{lcp}(p_k, f) \right)}{\sum_k |p_k|}
$$

Here $p_k$ is the $k$-th partial as a list of words, $f$ is the final, $|\cdot|$ is a word count, and $\mathrm{lcp}$ is the length of the common prefix. Read it aloud: of all the words shown, what fraction sat after the first mistake? For the log, the partials have 1, 2, 3, and 4 words. "turn on a" has a common prefix of 2, so 1 word is unstable. "turn on the light" has a common prefix of 3, so 1 word is unstable. The rate is $2/10 = 0.2$.

> [!WARNING]
> Comparing lengths is wrong. A partial "turn of" has the same length as the final "turn on", but its second word never becomes correct. Compare words from the front.

Instability trades against delay. A recognizer that waits for more look-ahead before it shows a word is more stable and slower, which is the same trade as the chunk size.

## What the inference course owns

Three techniques make a model cheaper to run. The [inference course](https://qwen.ifkash.dev/) owns them, and this chapter links to it and doesn't repeat them.

- **Quantization:** storing weights in fewer bits, covered in its "Part 6 — Scaling". The only part used here is the byte arithmetic below.
- **KV caching:** a streaming encoder keeps the keys and values of its left context so each chunk doesn't recompute them, and the inference course covers the cache in "Part 3 — Making it fast".
- **Speculative decoding:** a small draft model proposes tokens and the large model verifies them, covered in "Part 6 — Scaling". It speeds up autoregressive decoders such as Whisper's, not a streaming transducer.

## An on-device budget

A phone gives one app a fixed budget, and you can check it with arithmetic before you build anything. The weights take $\lceil \text{params} \cdot \text{bits} / 8 \rceil$ bytes. A 30 million parameter model takes 60 MB at 16 bits, 30 MB at 8 bits, and 15 MB at 4 bits.

Activations add to it, but only one chunk at a time. For one layer on a chunk of $C$ frames with $N_L$ frames of left context, model dimension $d$, $h$ heads, and $b$ bytes per value, count the layer's input and output, the keys and values for the whole context, and the attention scores:

$$
\text{bytes} = b \cdot \left( 2Cd + 2(N_L + C)d + h\,C\,(N_L + C) \right)
$$

Layers run one after another and free their activations, so the peak is one layer, not the sum. This is a simplified count that leaves out the cached left context of the other layers. For $C = 16$, $N_L = 64$, $d = 256$, $h = 4$, and 2 bytes per value (fp16), the three terms are $8{,}192 + 40{,}960 + 5{,}120 = 54{,}272$ values, which is 108,544 bytes, about 0.1 MB. The weights dominate. Figure 3 checks a 40 MB budget.

```figure
<svg viewBox="0 0 580 230" role="img" aria-label="Weight bytes of a 30 million parameter model at 16, 8, and 4 bits against a 40 MB budget">
<text x="100" y="47" text-anchor="end" class="fig-small">16 bits</text>
<rect x="110" y="30" width="380.0" height="24" class="fill-bad"/>
<rect x="490.0" y="30" width="1.5" height="24" class="fill-a"/>
<text x="498.7" y="47" class="fig-small">60.1 MB</text>
<text x="100" y="93" text-anchor="end" class="fig-small">8 bits</text>
<rect x="110" y="76" width="190.0" height="24" class="fill-b"/>
<rect x="300.0" y="76" width="1.5" height="24" class="fill-a"/>
<text x="308.7" y="93" class="fig-small">30.1 MB</text>
<text x="100" y="139" text-anchor="end" class="fig-small">4 bits</text>
<rect x="110" y="122" width="95.0" height="24" class="fill-b"/>
<rect x="205.0" y="122" width="1.5" height="24" class="fill-a"/>
<text x="213.7" y="139" class="fig-small">15.1 MB</text>
<line x1="363.3" y1="20" x2="363.3" y2="170" class="stroke-ink dashed"/>
<text x="363.3" y="14" text-anchor="middle" class="fig-small">budget 40 MB</text>
<text x="110" y="205" class="fig-small">orange fits, red does not. The thin blue slice is one chunk of activations, 0.1 MB.</text>
</svg>
Figure 3. Weights plus one chunk of activations for a 30 million parameter model. At 16 bits the model needs 60.1 MB and misses the 40 MB budget. At 8 and 4 bits it needs 30.1 and 15.1 MB and fits. Notice that the activations are a sliver compared with the weights.
```

Memory is half the check. Speed is the other half. Rule of thumb, an encoder uses about one multiply-accumulate (MAC) per parameter per frame. At 25 frames per second, 30 million parameters need $30 \times 10^6 \times 25 = 750 \times 10^6$ MAC per audio second. On a core that sustains $2 \times 10^9$ MAC per second, the RTF is $750 / 2000 = 0.375$, so the model keeps up. On a core at $0.5 \times 10^9$ it is 1.5, and it doesn't.

> [!KEY]
> What you now know: a streaming recognizer has two clocks, waiting and compute, and RTF captures only the second. You measure first-partial, finalization, and endpoint-to-final latency from each word's end, score partials by the unstable-prefix rate, and decide a device with two sums: weights plus one chunk must fit the memory, and the RTF must stay below 1.

> [!RECAP]
> - RTF is compute seconds per audio second. At 1 or more, a stream falls behind forever.
> - Position $p$ in a chunk has an algorithmic delay of $(C - p + L)$ hops, and compute adds $\mathrm{RTF} \cdot C$ hops.
> - Measure each latency from the word end. Finalization is the endpoint wait plus endpoint-to-final.
> - The unstable-prefix rate counts shown words after the first mismatch with the final.
> - A device needs weights plus one chunk of activations within its memory, and an RTF below 1.

## Build it step by step

The checks test the functions in this order.

1. **`rtf(audio_seconds, compute_seconds)`.** Returns compute divided by audio. Common mistake: dividing the other way, which gives throughput.
2. **`algorithmic_latency(chunk_frames, lookahead_frames, hop_ms)`.** Returns an array of `chunk_frames` delays in ms. Use `np.arange(chunk_frames)` for $p$. Common mistake: forgetting the frame's own hop, so the last position has no delay at zero look-ahead.
3. **`streaming_latency(chunk_frames, lookahead_frames, hop_ms, rtf_value)`.** Adds `rtf_value * chunk_frames * hop_ms` to every position, and returns infinity when the RTF is 1 or more.
4. **`partial_latencies(events, word_ends, endpoint_ms)`.** Takes `(time_ms, kind, text)` events. Returns a dict with `first_partial_ms`, `finalization_ms`, and `endpoint_to_final_ms`. Skip partials with no words. Common mistake: returning event times instead of subtracting the word end.
5. **`unstable_prefix_rate(partials, final)`.** Splits each string into words and walks from the front until a mismatch. Common mistake: comparing lengths.
6. **`model_bytes(params, bits)` and `chunk_activation_bytes(...)`.** Use `math.ceil` for the weights. Common mistake: using bits as bytes.
7. **`fits_on_device(params, bits, chunk_activation_bytes, budget_bytes)` and `device_rtf(...)`.** Weights plus activations, compared with `<=`. Common mistake: leaving the activations out.

## Where this breaks

**Compute is not steady.** A phone throttles when it heats up, so an RTF measured in the first minute may not hold in the tenth. Measure over a long run and look at the slowest chunks.

**Memory is not only weights.** The runtime, the audio buffer, and the decoder's search state take space too. Leave headroom, and treat the budget above as a lower bound on the need.

**Partials are a product choice.** Some apps hide partials until they are stable, trading first-partial latency for fewer rewrites. The metrics here let you compare the options.

## The exercise

You write `rtf`, `algorithmic_latency`, `streaming_latency`, `partial_latencies`, `unstable_prefix_rate`, `model_bytes`, `chunk_activation_bytes`, `fits_on_device`, and `device_rtf`. Press **Run** to print the chunk-size sweep from Figure 1, the latencies of the "turn on the lights" log, and whether the 30 million parameter model fits at 16, 8, and 4 bits.

The checks compare your delays with the worked $240, 200, 160, 120$ ms, and compute with $320, 280, 240, 200$ ms. They confirm that an RTF of 1 gives infinite latency, that latencies run from the word end and not the audio start, and that finalization splits into the endpoint wait and endpoint-to-final. They score the log at 0.2, reject a partial that matches only in length, and test the memory budget at its exact boundary.

## Further reading

- Shangguan et al., 2021, "Dissecting user-perceived latency of on-device E2E speech recognition," on how to measure the latency a user sees.
- He et al., 2019, "Streaming end-to-end speech recognition for mobile devices," on a streaming transducer small enough for a phone.
- The sherpa-onnx documentation, whose pages on online transducer models and endpoint detection show these settings in a deployed toolkit.
- [Inference course home page](https://qwen.ifkash.dev/), for quantization, the KV cache, and speculative decoding.
