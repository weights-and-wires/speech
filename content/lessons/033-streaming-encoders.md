---
title: Streaming encoders
slug: 033-streaming-encoders
part: "Part 4 — Neural speech recognition"
summary: Limit what each frame may attend to, with a chunked mask, so an encoder can stream, and measure the delay and context that each chunk size buys.
minutes: 40
gpu: false
objectives:
  - Explain why a self-attention encoder cannot stream, and why a causal mask alone loses context.
  - Build a chunked attention mask with a limit on left chunks.
  - Compute the look-ahead delay of each frame in a chunk, and the context each frame sees.
  - Show that a masked encoder's early outputs never depend on future audio.
prereqs:
  - 031-attention-encoder-decoders
  - 032-transducers
exercise: 033-streaming-encoders
---

# Streaming encoders

You dictate a message, and the words must appear while you talk, not a minute after you stop. A transducer already writes words as audio arrives, but only if its encoder does not need the whole recording first. A self-attention encoder does need it: every frame attends to every other frame, so frame 1's output depends on the last frame of the utterance, and nothing can be computed until the audio ends.

The fix is to limit what each frame may look at. The question for this chapter is how much of the future a frame should see, and what that costs. Looking at more future gives better accuracy, because a sound often becomes clear only after what follows it. But every frame you wait for is a frame of delay the user feels.

## What you already know

- Self-attention scores every pair of frames, and setting a score to $-\infty$ gives that pair a weight of exactly 0, from [Attention encoder-decoders](/lessons/031-attention-encoder-decoders).
- A transducer streams only when its encoder does, from [Transducers](/lessons/032-transducers).
- An encoder frame is a fixed number of milliseconds long, such as 40 ms after the front end subsamples.
- A unidirectional GRU streams by construction, because each output depends only on the past.
- A mask is a grid of true and false values, one per query and key pair.

## Why not look only at the past?

The simplest streaming mask is **causal**: frame $i$ may attend to frames $j \le i$. Its delay is zero, since each output needs only audio that has already arrived. But it gives up something real. A phone's sound is shaped by the phone that follows it, so the frame where an "s" turns into an "h" is easier to label if you can see a few frames beyond it. A causal encoder must commit before the evidence is in.

At the other end, full attention sees everything and has a delay equal to the whole utterance. A **chunked** mask sits between them. It splits the frames into chunks of $C$ frames, and a frame may see its whole chunk, including frames after it, plus earlier chunks. The wait is then at most one chunk.

## The chunked mask

Frame $i$ belongs to chunk $\lfloor i / C \rfloor$. Frame $i$ may attend to frame $j$ when

$$
\left\lfloor j / C \right\rfloor \le \left\lfloor i / C \right\rfloor
\quad\text{and}\quad
\left\lfloor j / C \right\rfloor \ge \left\lfloor i / C \right\rfloor - N_L
$$

Read it aloud: a frame sees every frame in its own chunk and in earlier chunks, but not in later ones, and at most $N_L$ chunks back. With no limit, drop the second condition. Work $T = 6$ and $C = 2$. The chunks are frames $\{0, 1\}$, $\{2, 3\}$, and $\{4, 5\}$. Frames 0 and 1 see frames 0 and 1. Frames 2 and 3 see frames 0 to 3. Frames 4 and 5 see all six. So the number of frames each query sees is $(2, 2, 4, 4, 6, 6)$. With $N_L = 1$, frames 4 and 5 lose the first chunk and see only 2 to 5, so the counts are $(2, 2, 4, 4, 4, 4)$.

```figure
<svg viewBox="0 0 560 330" role="img" aria-label="A 12 by 12 chunked attention mask with chunks of 4 frames and one chunk of left context; allowed cells are orange">
<rect x="70" y="30" width="21" height="21" class="fill-b"/>
<rect x="92" y="30" width="21" height="21" class="fill-b"/>
<rect x="114" y="30" width="21" height="21" class="fill-b"/>
<rect x="136" y="30" width="21" height="21" class="fill-b"/>
<rect x="158" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="180" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="202" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="224" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="246" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="30" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="52" width="21" height="21" class="fill-b"/>
<rect x="92" y="52" width="21" height="21" class="fill-b"/>
<rect x="114" y="52" width="21" height="21" class="fill-b"/>
<rect x="136" y="52" width="21" height="21" class="fill-b"/>
<rect x="158" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="180" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="202" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="224" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="246" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="52" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="74" width="21" height="21" class="fill-b"/>
<rect x="92" y="74" width="21" height="21" class="fill-b"/>
<rect x="114" y="74" width="21" height="21" class="fill-b"/>
<rect x="136" y="74" width="21" height="21" class="fill-b"/>
<rect x="158" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="180" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="202" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="224" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="246" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="74" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="96" width="21" height="21" class="fill-b"/>
<rect x="92" y="96" width="21" height="21" class="fill-b"/>
<rect x="114" y="96" width="21" height="21" class="fill-b"/>
<rect x="136" y="96" width="21" height="21" class="fill-b"/>
<rect x="158" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="180" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="202" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="224" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="246" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="96" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="118" width="21" height="21" class="fill-b"/>
<rect x="92" y="118" width="21" height="21" class="fill-b"/>
<rect x="114" y="118" width="21" height="21" class="fill-b"/>
<rect x="136" y="118" width="21" height="21" class="fill-b"/>
<rect x="158" y="118" width="21" height="21" class="fill-b"/>
<rect x="180" y="118" width="21" height="21" class="fill-b"/>
<rect x="202" y="118" width="21" height="21" class="fill-b"/>
<rect x="224" y="118" width="21" height="21" class="fill-b"/>
<rect x="246" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="118" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="140" width="21" height="21" class="fill-b"/>
<rect x="92" y="140" width="21" height="21" class="fill-b"/>
<rect x="114" y="140" width="21" height="21" class="fill-b"/>
<rect x="136" y="140" width="21" height="21" class="fill-b"/>
<rect x="158" y="140" width="21" height="21" class="fill-b"/>
<rect x="180" y="140" width="21" height="21" class="fill-b"/>
<rect x="202" y="140" width="21" height="21" class="fill-b"/>
<rect x="224" y="140" width="21" height="21" class="fill-b"/>
<rect x="246" y="140" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="140" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="140" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="140" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="162" width="21" height="21" class="fill-b"/>
<rect x="92" y="162" width="21" height="21" class="fill-b"/>
<rect x="114" y="162" width="21" height="21" class="fill-b"/>
<rect x="136" y="162" width="21" height="21" class="fill-b"/>
<rect x="158" y="162" width="21" height="21" class="fill-b"/>
<rect x="180" y="162" width="21" height="21" class="fill-b"/>
<rect x="202" y="162" width="21" height="21" class="fill-b"/>
<rect x="224" y="162" width="21" height="21" class="fill-b"/>
<rect x="246" y="162" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="162" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="162" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="162" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="184" width="21" height="21" class="fill-b"/>
<rect x="92" y="184" width="21" height="21" class="fill-b"/>
<rect x="114" y="184" width="21" height="21" class="fill-b"/>
<rect x="136" y="184" width="21" height="21" class="fill-b"/>
<rect x="158" y="184" width="21" height="21" class="fill-b"/>
<rect x="180" y="184" width="21" height="21" class="fill-b"/>
<rect x="202" y="184" width="21" height="21" class="fill-b"/>
<rect x="224" y="184" width="21" height="21" class="fill-b"/>
<rect x="246" y="184" width="21" height="21" class="fill-sunk"/>
<rect x="268" y="184" width="21" height="21" class="fill-sunk"/>
<rect x="290" y="184" width="21" height="21" class="fill-sunk"/>
<rect x="312" y="184" width="21" height="21" class="fill-sunk"/>
<rect x="70" y="206" width="21" height="21" class="fill-sunk"/>
<rect x="92" y="206" width="21" height="21" class="fill-sunk"/>
<rect x="114" y="206" width="21" height="21" class="fill-sunk"/>
<rect x="136" y="206" width="21" height="21" class="fill-sunk"/>
<rect x="158" y="206" width="21" height="21" class="fill-b"/>
<rect x="180" y="206" width="21" height="21" class="fill-b"/>
<rect x="202" y="206" width="21" height="21" class="fill-b"/>
<rect x="224" y="206" width="21" height="21" class="fill-b"/>
<rect x="246" y="206" width="21" height="21" class="fill-b"/>
<rect x="268" y="206" width="21" height="21" class="fill-b"/>
<rect x="290" y="206" width="21" height="21" class="fill-b"/>
<rect x="312" y="206" width="21" height="21" class="fill-b"/>
<rect x="70" y="228" width="21" height="21" class="fill-sunk"/>
<rect x="92" y="228" width="21" height="21" class="fill-sunk"/>
<rect x="114" y="228" width="21" height="21" class="fill-sunk"/>
<rect x="136" y="228" width="21" height="21" class="fill-sunk"/>
<rect x="158" y="228" width="21" height="21" class="fill-b"/>
<rect x="180" y="228" width="21" height="21" class="fill-b"/>
<rect x="202" y="228" width="21" height="21" class="fill-b"/>
<rect x="224" y="228" width="21" height="21" class="fill-b"/>
<rect x="246" y="228" width="21" height="21" class="fill-b"/>
<rect x="268" y="228" width="21" height="21" class="fill-b"/>
<rect x="290" y="228" width="21" height="21" class="fill-b"/>
<rect x="312" y="228" width="21" height="21" class="fill-b"/>
<rect x="70" y="250" width="21" height="21" class="fill-sunk"/>
<rect x="92" y="250" width="21" height="21" class="fill-sunk"/>
<rect x="114" y="250" width="21" height="21" class="fill-sunk"/>
<rect x="136" y="250" width="21" height="21" class="fill-sunk"/>
<rect x="158" y="250" width="21" height="21" class="fill-b"/>
<rect x="180" y="250" width="21" height="21" class="fill-b"/>
<rect x="202" y="250" width="21" height="21" class="fill-b"/>
<rect x="224" y="250" width="21" height="21" class="fill-b"/>
<rect x="246" y="250" width="21" height="21" class="fill-b"/>
<rect x="268" y="250" width="21" height="21" class="fill-b"/>
<rect x="290" y="250" width="21" height="21" class="fill-b"/>
<rect x="312" y="250" width="21" height="21" class="fill-b"/>
<rect x="70" y="272" width="21" height="21" class="fill-sunk"/>
<rect x="92" y="272" width="21" height="21" class="fill-sunk"/>
<rect x="114" y="272" width="21" height="21" class="fill-sunk"/>
<rect x="136" y="272" width="21" height="21" class="fill-sunk"/>
<rect x="158" y="272" width="21" height="21" class="fill-b"/>
<rect x="180" y="272" width="21" height="21" class="fill-b"/>
<rect x="202" y="272" width="21" height="21" class="fill-b"/>
<rect x="224" y="272" width="21" height="21" class="fill-b"/>
<rect x="246" y="272" width="21" height="21" class="fill-b"/>
<rect x="268" y="272" width="21" height="21" class="fill-b"/>
<rect x="290" y="272" width="21" height="21" class="fill-b"/>
<rect x="312" y="272" width="21" height="21" class="fill-b"/>
<line x1="69.5" y1="26" x2="69.5" y2="294" class="stroke-ink"/>
<line x1="66" y1="29.5" x2="334" y2="29.5" class="stroke-ink"/>
<line x1="157.5" y1="26" x2="157.5" y2="294" class="stroke-ink"/>
<line x1="66" y1="117.5" x2="334" y2="117.5" class="stroke-ink"/>
<line x1="245.5" y1="26" x2="245.5" y2="294" class="stroke-ink"/>
<line x1="66" y1="205.5" x2="334" y2="205.5" class="stroke-ink"/>
<line x1="333.5" y1="26" x2="333.5" y2="294" class="stroke-ink"/>
<line x1="66" y1="293.5" x2="334" y2="293.5" class="stroke-ink"/>
<text x="80.5" y="22" text-anchor="middle" class="fig-small">0</text>
<text x="62" y="45.0" text-anchor="end" class="fig-small">0</text>
<text x="102.5" y="22" text-anchor="middle" class="fig-small">1</text>
<text x="62" y="67.0" text-anchor="end" class="fig-small">1</text>
<text x="124.5" y="22" text-anchor="middle" class="fig-small">2</text>
<text x="62" y="89.0" text-anchor="end" class="fig-small">2</text>
<text x="146.5" y="22" text-anchor="middle" class="fig-small">3</text>
<text x="62" y="111.0" text-anchor="end" class="fig-small">3</text>
<text x="168.5" y="22" text-anchor="middle" class="fig-small">4</text>
<text x="62" y="133.0" text-anchor="end" class="fig-small">4</text>
<text x="190.5" y="22" text-anchor="middle" class="fig-small">5</text>
<text x="62" y="155.0" text-anchor="end" class="fig-small">5</text>
<text x="212.5" y="22" text-anchor="middle" class="fig-small">6</text>
<text x="62" y="177.0" text-anchor="end" class="fig-small">6</text>
<text x="234.5" y="22" text-anchor="middle" class="fig-small">7</text>
<text x="62" y="199.0" text-anchor="end" class="fig-small">7</text>
<text x="256.5" y="22" text-anchor="middle" class="fig-small">8</text>
<text x="62" y="221.0" text-anchor="end" class="fig-small">8</text>
<text x="278.5" y="22" text-anchor="middle" class="fig-small">9</text>
<text x="62" y="243.0" text-anchor="end" class="fig-small">9</text>
<text x="300.5" y="22" text-anchor="middle" class="fig-small">10</text>
<text x="62" y="265.0" text-anchor="end" class="fig-small">10</text>
<text x="322.5" y="22" text-anchor="middle" class="fig-small">11</text>
<text x="62" y="287.0" text-anchor="end" class="fig-small">11</text>
<text x="202.0" y="320" text-anchor="middle">key frame j (what it looks at)</text>
<text x="20" y="162.0" transform="rotate(-90 20 162.0)" text-anchor="middle">query frame i</text>
<text x="364" y="50" class="fig-bold">chunks of 4</text>
<text x="364" y="70" class="fig-small">frames 0-3, 4-7, 8-11</text>
<text x="364" y="100" class="fig-small">frame 5 sees 0 to 7:</text>
<text x="364" y="116" class="fig-small">its chunk and one</text>
<text x="364" y="132" class="fig-small">chunk to the left</text>
<text x="364" y="162" class="fig-small">frame 5 cannot see</text>
<text x="364" y="178" class="fig-small">frames 8 to 11</text>
</svg>
Figure 1. The mask for 12 frames, 4 per chunk, with one chunk of left context. Row i shows which keys frame i may read. Notice the blocks along the diagonal: a frame sees the whole of its own chunk, including frames after it, but nothing in a later chunk.
```

In torch, the chunk index of every frame is one `arange`, and broadcasting compares all pairs at once:

```python
chunk = torch.arange(T) // chunk_size
allowed = chunk[None, :] <= chunk[:, None]               # (T, T), True where i may see j
if left_chunks is not None:
    allowed &= chunk[None, :] >= chunk[:, None] - left_chunks
```

> [!WARNING]
> The sense of a boolean mask depends on the function. `nn.MultiheadAttention` takes `attn_mask` where `True` means blocked, so pass `~allowed`. `F.scaled_dot_product_attention` takes `True` to mean allowed. Mixing them up blocks exactly the frames you meant to allow.

## The delay each frame pays

Put the first frame of a chunk and the last frame of a chunk side by side. The last frame sees its whole chunk the moment it arrives. The first frame must wait for the other $C - 1$ frames before its output is ready. So position $p$ (counting from 0) waits for $C - 1 - p$ frames, its **look-ahead**. Its delay, from the start of its own frame to its output, is the look-ahead plus the frame itself:

$$
\text{delay}(p) = (C - 1 - p + 1) \cdot \text{frame\_ms} = (C - p) \cdot \text{frame\_ms}
$$

For $C = 4$ and 40 ms frames, the delays are 160, 120, 80, and 40 ms, as Figure 2 shows. The worst delay is $C \cdot \text{frame\_ms}$ and the mean is $(C + 1)/2 \cdot \text{frame\_ms}$, which is 100 ms here. This is also how real systems count: NVIDIA's cache-aware streaming FastConformer counts context in 80 ms frames, and a right context of 13 frames gives a latency of $(13 + 1) \times 80 = 1{,}120$ ms.

```figure
<svg viewBox="0 0 560 250" role="img" aria-label="Delay from the start of each frame to its output for a chunk of four 40 ms frames: 160, 120, 80, and 40 ms">
<line x1="40" y1="190" x2="500" y2="190" class="stroke-muted"/>
<rect x="70" y="46.0" width="70" height="144.0" class="fill-b"/>
<text x="105" y="40.0" text-anchor="middle" class="fig-bold fig-small">160 ms</text>
<text x="105" y="208" text-anchor="middle" class="fig-small">position 0</text>
<text x="105" y="224" text-anchor="middle" class="fig-small">waits for 3 more</text>
<rect x="180" y="82.0" width="70" height="108.0" class="fill-b"/>
<text x="215" y="76.0" text-anchor="middle" class="fig-bold fig-small">120 ms</text>
<text x="215" y="208" text-anchor="middle" class="fig-small">position 1</text>
<text x="215" y="224" text-anchor="middle" class="fig-small">waits for 2 more</text>
<rect x="290" y="118.0" width="70" height="72.0" class="fill-b"/>
<text x="325" y="112.0" text-anchor="middle" class="fig-bold fig-small">80 ms</text>
<text x="325" y="208" text-anchor="middle" class="fig-small">position 2</text>
<text x="325" y="224" text-anchor="middle" class="fig-small">waits for 1 more</text>
<rect x="400" y="154.0" width="70" height="36.0" class="fill-b"/>
<text x="435" y="148.0" text-anchor="middle" class="fig-bold fig-small">40 ms</text>
<text x="435" y="208" text-anchor="middle" class="fig-small">position 3</text>
<text x="435" y="224" text-anchor="middle" class="fig-small">waits for 0 more</text>
<text x="40" y="24" class="fig-bold">delay from the start of a frame to its output</text>
</svg>
Figure 2. In a chunk of four 40 ms frames, the first frame waits for the other three and has a delay of 160 ms, and the last frame has a delay of 40 ms, just its own length. Notice that the mean delay, 100 ms, is (C + 1) / 2 frames.
```

## What the delay buys

A bigger chunk gives each frame more context. Take 100 frames of 40 ms and 2 left chunks. A chunk of 1 gives a mean of 2.97 visible frames, and a chunk of 8 gives 21.92. Figure 3 plots every chunk size from 1 to 32: the mean context grows roughly in step with the mean delay, so you pay a steady rate of milliseconds per frame of context. The left limit $N_L$ does something different. It does not cost delay, but it caps memory, because no frame ever sees more than $(N_L + 1) \cdot C$ frames, however long the audio is.

```figure
<svg viewBox="0 0 560 270" role="img" aria-label="Mean context in frames against mean delay in milliseconds for chunk sizes 1 to 32, rising steadily">
<line x1="70.0" y1="30" x2="70.0" y2="220" class="stroke-grid"/>
<text x="70.0" y="236" text-anchor="middle" class="fig-small">0</text>
<line x1="134.3" y1="30" x2="134.3" y2="220" class="stroke-grid"/>
<text x="134.3" y="236" text-anchor="middle" class="fig-small">100</text>
<line x1="198.6" y1="30" x2="198.6" y2="220" class="stroke-grid"/>
<text x="198.6" y="236" text-anchor="middle" class="fig-small">200</text>
<line x1="262.9" y1="30" x2="262.9" y2="220" class="stroke-grid"/>
<text x="262.9" y="236" text-anchor="middle" class="fig-small">300</text>
<line x1="327.1" y1="30" x2="327.1" y2="220" class="stroke-grid"/>
<text x="327.1" y="236" text-anchor="middle" class="fig-small">400</text>
<line x1="391.4" y1="30" x2="391.4" y2="220" class="stroke-grid"/>
<text x="391.4" y="236" text-anchor="middle" class="fig-small">500</text>
<line x1="455.7" y1="30" x2="455.7" y2="220" class="stroke-grid"/>
<text x="455.7" y="236" text-anchor="middle" class="fig-small">600</text>
<line x1="520.0" y1="30" x2="520.0" y2="220" class="stroke-grid"/>
<text x="520.0" y="236" text-anchor="middle" class="fig-small">700</text>
<line x1="70" y1="220.0" x2="520" y2="220.0" class="stroke-grid"/>
<text x="62" y="224.0" text-anchor="end" class="fig-small">0</text>
<line x1="70" y1="192.9" x2="520" y2="192.9" class="stroke-grid"/>
<text x="62" y="196.9" text-anchor="end" class="fig-small">10</text>
<line x1="70" y1="165.7" x2="520" y2="165.7" class="stroke-grid"/>
<text x="62" y="169.7" text-anchor="end" class="fig-small">20</text>
<line x1="70" y1="138.6" x2="520" y2="138.6" class="stroke-grid"/>
<text x="62" y="142.6" text-anchor="end" class="fig-small">30</text>
<line x1="70" y1="111.4" x2="520" y2="111.4" class="stroke-grid"/>
<text x="62" y="115.4" text-anchor="end" class="fig-small">40</text>
<line x1="70" y1="84.3" x2="520" y2="84.3" class="stroke-grid"/>
<text x="62" y="88.3" text-anchor="end" class="fig-small">50</text>
<line x1="70" y1="57.1" x2="520" y2="57.1" class="stroke-grid"/>
<text x="62" y="61.1" text-anchor="end" class="fig-small">60</text>
<line x1="70" y1="30.0" x2="520" y2="30.0" class="stroke-grid"/>
<text x="62" y="34.0" text-anchor="end" class="fig-small">70</text>
<polyline points="95.7,211.9 108.6,204.0 134.3,188.7 185.7,160.5 288.6,111.9 494.3,45.9" class="stroke-b" fill="none"/>
<circle cx="95.7" cy="211.9" r="5" class="fill-b"/>
<text x="103.7" y="231.9" class="fig-small">C = 1</text>
<circle cx="108.6" cy="204.0" r="5" class="fill-b"/>
<text x="116.6" y="224.0" class="fig-small">C = 2</text>
<circle cx="134.3" cy="188.7" r="5" class="fill-b"/>
<text x="142.3" y="208.7" class="fig-small">C = 4</text>
<circle cx="185.7" cy="160.5" r="5" class="fill-b"/>
<text x="193.7" y="180.5" class="fig-small">C = 8</text>
<circle cx="288.6" cy="111.9" r="5" class="fill-b"/>
<text x="296.6" y="131.9" class="fig-small">C = 16</text>
<circle cx="494.3" cy="45.9" r="5" class="fill-b"/>
<text x="502.3" y="65.9" class="fig-small">C = 32</text>
<text x="295.0" y="258" text-anchor="middle">mean delay (ms), frames of 40 ms</text>
<text x="18" y="125.0" transform="rotate(-90 18 125.0)" text-anchor="middle">mean frames each frame sees</text>
</svg>
Figure 3. Each point is a chunk size C with 2 left chunks, over 100 frames of 40 ms. Context rises almost in step with delay, so you buy it with latency at a steady rate. Notice that the delay for C = 32 is 660 ms, already a noticeable pause.
```

Training with random chunk sizes, as WeNet's dynamic chunk training does, lets one model serve several latencies, so you can choose the trade-off at deployment.

> [!KEY]
> A causal mask streams with zero look-ahead but loses the evidence that follows a sound. A chunked mask lets each frame see to the end of its chunk, so the chunk size is the look-ahead you pay in delay, and the left-chunk limit bounds the memory each step needs.

## Why the early outputs never change

An encoder streams only if no output depends on future chunks. The chunked mask guarantees it by construction: a weight of 0 on a future key removes that key from the weighted sum, so the output is the same whatever the key holds. You can test this directly. Compute the outputs, change the frames in the last chunk, and compute them again. Every output in earlier chunks must be identical, while the last chunk's own outputs must change. Look-ahead also does not add up across layers: because each layer lets a frame see only to the end of its own chunk, stacking layers does not push the frame's view further into the future, unlike a fixed right context that grows with every layer.

> [!KEY]
> What you now know: an encoder streams when its mask hides future chunks. A chunk of $C$ frames costs $(C + 1)/2$ frames of mean delay, `left_chunks` caps the context, and you can verify the whole idea by changing the future and seeing the past stay the same.

> [!RECAP]
> - A frame in chunk $\lfloor i / C \rfloor$ may attend to frame $j$ when $\lfloor j / C \rfloor \le \lfloor i / C \rfloor$ and, with a limit, $\lfloor j / C \rfloor \ge \lfloor i / C \rfloor - N_L$.
> - Position $p$ in a chunk has a look-ahead of $C - 1 - p$ frames and a delay of $(C - p) \cdot \text{frame\_ms}$.
> - A frame sees at most $(N_L + 1) \cdot C$ frames, so the context is bounded.
> - `nn.MultiheadAttention` wants the inverted mask, `~allowed`.

## Build it step by step

The checks test the functions in this order.

1. **`chunked_attention_mask(T, chunk_size, left_chunks)`.** Returns a `(T, T)` bool tensor. Compute `chunk = torch.arange(T) // chunk_size` and compare by broadcasting. Common mistake: using `j <= i` instead of chunk indices, which gives a causal mask with no look-ahead.
2. **`context_frames(mask)`.** Returns a `(T,)` int tensor: the number of `True` entries in each row. Common mistake: summing the wrong axis, which counts how many queries see each key.
3. **`lookahead_frames(chunk_size)` and `delay_ms(chunk_size, frame_ms)`.** The first returns `[C - 1, ..., 0]`, the second `(lookahead + 1) * frame_ms`. Common mistake: forgetting the frame's own length, which gives 0 ms for the last frame.
4. **`masked_self_attention(x, mask)`.** Takes `(T, d)` and a bool mask, with `x` as query, key, and value, and returns `(output, weights)`. Use `masked_fill(~mask, float("-inf"))` before the softmax. Common mistake: masking after the softmax, so the rows no longer sum to 1.

## Where this breaks

**BatchNorm breaks chunked inference.** Statistics computed over a full utterance differ from those over one chunk. Many streaming models use LayerNorm in their convolution modules for this reason.

**Pseudo-streaming is not streaming.** Rerunning a full-context model on overlapping windows gives text while audio arrives, but each window waits for its whole length and repeats work. Measure latency from a word's end to its emission, not from the chunk size alone.

**Convolutions look ahead too.** A convolution with a symmetric kernel reads future frames. A streaming encoder pads it on one side only, or the "causal" encoder is not causal.

## The exercise

You build `chunked_attention_mask`, `context_frames`, `lookahead_frames`, `delay_ms`, and `masked_self_attention`. Press **Run** to compare chunk sizes from 1 to 32 frames: it prints each one's mean delay and mean context, plots context against delay as Figure 3 does, draws a mask, and confirms that changing the last chunk leaves the earlier outputs alone.

The checks compare your mask with the rule, with and without a left-chunk limit, and compare the context counts with the hand-worked $(2, 2, 4, 4, 6, 6)$. They check the delays $160, 120, 80, 40$ ms, run your attention against a frame-by-frame loop and confirm that masked keys get weight exactly 0. The last two check that changing future chunks leaves earlier outputs unchanged, and that a bigger chunk buys more context at more delay while `left_chunks` caps the context.
