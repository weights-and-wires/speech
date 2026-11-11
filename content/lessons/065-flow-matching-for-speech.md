---
title: Flow matching for speech
slug: 065-flow-matching-for-speech
part: "Part 7 — Diffusion and flow matching"
summary: Train a flow-matching mel generator that says the digits, with Voicebox's masked loss, classifier-free guidance, F5-TTS's sway sampling, and a chunk mask.
minutes: 70
gpu: true
objectives:
  - Build the masked infilling loss that Voicebox, E2 TTS, and F5-TTS train with, and explain why text-to-speech is a special case of infilling.
  - Apply classifier-free guidance to a velocity, convert between the two guidance conventions, and batch both passes into one call.
  - Warp the sampling time grid with sway sampling and explain why the early steps near noise matter most.
  - Build CosyVoice 2's chunk-causal attention mask and explain how one model serves streaming and offline synthesis.
  - Turn generated log-mels into sound with a pseudo-inverse filterbank and Griffin-Lim.
prereqs:
  - 063-flow-matching
  - 061-classifier-free-guidance
  - 012-hearing-and-the-mel-scale
  - 008-phase-and-griffin-lim
exercise: 065-flow-matching-for-speech
---

# Flow matching for speech

You want one model that can read a sentence in a stranger's voice after hearing three seconds of them, replace a single wrong word in a recording without redoing the rest, and plain read text aloud. The previous chapter gave you the engine, a flow that carries noise to data. Replace each 2-D point with a mel spectrogram and add a condition such as a transcript, and you have the acoustic model of a text-to-speech (TTS) system. Voicebox, E2 TTS, F5-TTS, Matcha-TTS, and the decoder of CosyVoice 2 all work this way, and a separate vocoder turns the mel into a waveform.

Four tools turn the plain flow into a speech generator. A masked loss teaches the model to fill in missing frames, which covers all three jobs above at once. Guidance makes the output follow its condition more strongly. A warped time grid spends the sampling steps where they matter. A chunk mask lets the same model stream. You build all four, then train a small model that says the digits zero to nine, and you hear it. The convention stays as it was: $t = 0$ is noise, $t = 1$ is data, and $\sigma_{\min} = 0$, as in F5-TTS.

## What you already know

- A [flow-matching](/lessons/063-flow-matching) model regresses $v_\theta(x_t, t)$ onto $x_1 - x_0$ along the straight path $x_t = (1 - t)\,x_0 + t\,x_1$, and samples by Euler steps.
- A log-mel spectrogram is a matrix of bands by frames, and the frame rate is the sample rate divided by the hop.
- [Griffin-Lim](/lessons/008-phase-and-griffin-lim) recovers a waveform from a magnitude spectrogram by alternating between the STFT and its inverse.
- Classifier-free guidance combines a conditional and an unconditional prediction to follow the condition more closely.
- An attention mask sets which positions each query may look at.

## Speech as a picture to fill in

Picture a mel spectrogram as a painting with a strip cut out. Voicebox (Le et al., 2023) trains one model to repaint the strip, given the rest of the painting and the text. A *mask* $m$ marks each frame 1 when the model must generate it and 0 when it can see it. The visible frames form the *context* $x_{\text{ctx}} = (1 - m) \odot x_1$, which the network receives next to the noisy input $x_t$. The loss counts only the frames the model had to generate. Counting the visible ones would reward copying the context, which teaches nothing:

$$
\mathcal{L} = \frac{\sum \left[ m \odot \left(v_\theta(x_t, x_{\text{ctx}}, c, t) - (x_1 - x_0)\right)^2 \right]}{F \sum m}
$$

Read it aloud: take the squared velocity error at every band and frame, keep only the masked frames, and divide by the number of masked values, $F$ bands times the number of masked frames. Frames the model could see add nothing, whatever it outputs there. The symbol $c$ is the condition, the transcript in Voicebox and the digit label in this chapter's run. In torch, with mels shaped `(B, F, N)` and a mask shaped `(B, N)`:

```python
m = mask[:, None, :]
context = (1 - m) * x1
loss = (((v - (x1 - x0)) ** 2) * m).sum() / (m.sum() * v.shape[1])
```

Voicebox masks 70 to 100 percent of the frames of each training clip. At inference, *everything* is infilling. For zero-shot TTS, the context is a few seconds of a reference speaker, and the masked part is the sentence to synthesize in that voice. For editing, the mask covers one word.

### Work one masked loss by hand

Take one band and four frames (Figure 1): data $x_1 = (2, 1, -1, 0)$, noise $x_0 = (0.5, -1, 1, 0.2)$, mask $m = (0, 1, 1, 0)$, and $t = 0.5$. Then:

- $x_t = 0.5\,x_0 + 0.5\,x_1 = (1.25, 0, 0, 0.1)$.
- The context is $(2, 0, 0, 0)$: frame 1 is visible, frames 2 and 3 are masked, and frame 4 is visible but happens to be 0.
- The target is $x_1 - x_0 = (1.5, 2, -2, -0.2)$.
- A prediction $v = (0, 1, -1, 0)$ has squared errors $(2.25, 1, 1, 0.04)$. The masked ones sum to 2, over $1 \times 2$ masked values, so the loss is 1.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="A one-band, four-frame example of the masked loss: frames 2 and 3 are masked, and only their squared errors, 1 and 1, count, giving a loss of 1">
<defs><marker id="c052-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="282.0" y="28.0" class="fig-small" text-anchor="middle">frame 1</text>
<text x="362.0" y="28.0" class="fig-small" text-anchor="middle">frame 2</text>
<rect x="326.0" y="36.0" width="72.0" height="230.0" class="fill-soft-b"/>
<text x="442.0" y="28.0" class="fig-small" text-anchor="middle">frame 3</text>
<rect x="406.0" y="36.0" width="72.0" height="230.0" class="fill-soft-b"/>
<text x="522.0" y="28.0" class="fig-small" text-anchor="middle">frame 4</text>
<text x="40.0" y="28.0" class="fig-small" text-anchor="start">mask m (1 = generate)</text>
<text x="40.0" y="62.0" class="fig-small" text-anchor="start">data x1</text>
<text x="282.0" y="62.0" class="fig-mono" text-anchor="middle">2</text>
<text x="362.0" y="62.0" class="fig-mono" text-anchor="middle">1</text>
<text x="442.0" y="62.0" class="fig-mono" text-anchor="middle">-1</text>
<text x="522.0" y="62.0" class="fig-mono" text-anchor="middle">0</text>
<text x="40.0" y="94.0" class="fig-small" text-anchor="start">noise x0</text>
<text x="282.0" y="94.0" class="fig-mono" text-anchor="middle">0.5</text>
<text x="362.0" y="94.0" class="fig-mono" text-anchor="middle">-1</text>
<text x="442.0" y="94.0" class="fig-mono" text-anchor="middle">1</text>
<text x="522.0" y="94.0" class="fig-mono" text-anchor="middle">0.2</text>
<text x="40.0" y="126.0" class="fig-small" text-anchor="start">x_t at t = 0.5</text>
<text x="282.0" y="126.0" class="fig-mono" text-anchor="middle">1.25</text>
<text x="362.0" y="126.0" class="fig-mono" text-anchor="middle">0</text>
<text x="442.0" y="126.0" class="fig-mono" text-anchor="middle">0</text>
<text x="522.0" y="126.0" class="fig-mono" text-anchor="middle">0.1</text>
<text x="40.0" y="158.0" class="fig-small" text-anchor="start">context (1-m) x1</text>
<text x="282.0" y="158.0" class="fig-mono" text-anchor="middle">2</text>
<text x="362.0" y="158.0" class="fig-mono" text-anchor="middle">0</text>
<text x="442.0" y="158.0" class="fig-mono" text-anchor="middle">0</text>
<text x="522.0" y="158.0" class="fig-mono" text-anchor="middle">0</text>
<text x="40.0" y="190.0" class="fig-small" text-anchor="start">target x1 - x0</text>
<text x="282.0" y="190.0" class="fig-mono" text-anchor="middle">1.5</text>
<text x="362.0" y="190.0" class="fig-mono" text-anchor="middle">2</text>
<text x="442.0" y="190.0" class="fig-mono" text-anchor="middle">-2</text>
<text x="522.0" y="190.0" class="fig-mono" text-anchor="middle">-0.2</text>
<text x="40.0" y="222.0" class="fig-small" text-anchor="start">prediction v</text>
<text x="282.0" y="222.0" class="fig-mono" text-anchor="middle">0</text>
<text x="362.0" y="222.0" class="fig-mono" text-anchor="middle">1</text>
<text x="442.0" y="222.0" class="fig-mono" text-anchor="middle">-1</text>
<text x="522.0" y="222.0" class="fig-mono" text-anchor="middle">0</text>
<text x="40.0" y="254.0" class="fig-small" text-anchor="start">squared error</text>
<text x="282.0" y="254.0" class="fig-mono" text-anchor="middle">2.25</text>
<text x="362.0" y="254.0" class="fig-mono" text-anchor="middle">1</text>
<text x="442.0" y="254.0" class="fig-mono" text-anchor="middle">1</text>
<text x="522.0" y="254.0" class="fig-mono" text-anchor="middle">0.04</text>
<text x="40.0" y="286.0" class="fig-bold" text-anchor="start">loss = (1 + 1) / (1 x 2 masked values) = 1</text>
<text x="282.0" y="16.0" class="fig-small" text-anchor="middle">visible</text>
<text x="362.0" y="16.0" class="fig-small" text-anchor="middle">masked</text>
<text x="442.0" y="16.0" class="fig-small" text-anchor="middle">masked</text>
<text x="522.0" y="16.0" class="fig-small" text-anchor="middle">visible</text>
</svg>
Figure 1. The shaded columns are the frames the model must generate. Notice that the 2.25 error on frame 1 and the 0.04 error on frame 4 never enter the loss, because the model could see those frames.
```

> [!KEY]
> Masked infilling turns text-to-speech, editing, and voice cloning into one task: generate the masked frames, given the visible ones and the condition.

## Guidance for a flow

A model trained on a transcript may follow it loosely. Classifier-free guidance, from [the guidance chapter](/lessons/061-classifier-free-guidance), pushes it to follow the condition harder, and it works on velocities as well as on noise predictions. During training, replace the condition with a *null* condition some of the time, with probability 0.2 in Voicebox. At each sampling step, predict once with the condition and once without, and extrapolate past the conditional velocity:

$$
\tilde v = v_c + \alpha\,(v_c - v_u)
$$

Read it aloud: start from the conditional velocity $v_c$ and push further in the direction that separates it from the unconditional velocity $v_u$, by a factor $\alpha$. With $v_c = 1.0$, $v_u = 0.4$, and $\alpha = 2$, the guided velocity is $1.0 + 2 \times 0.6 = 2.2$ (Figure 2). In code:

```python
both = model(torch.cat([x, x]), t, torch.cat([cond, null_cond]))
v = both[:B] + alpha * (both[:B] - both[B:])
```

```figure
<svg viewBox="0 0 600 170" role="img" aria-label="A number line showing the unconditional velocity 0.4, the conditional velocity 1.0, and the guided velocity 2.2 for alpha equal to 2">
<defs><marker id="c052b-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="60.0" y1="100.0" x2="560.0" y2="100.0" class="stroke-ink"/>
<line x1="60.0" y1="96.0" x2="60.0" y2="104.0" class="stroke-ink"/>
<text x="60.0" y="122.0" class="fig-small" text-anchor="middle">0</text>
<line x1="160.0" y1="96.0" x2="160.0" y2="104.0" class="stroke-ink"/>
<text x="160.0" y="122.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="260.0" y1="96.0" x2="260.0" y2="104.0" class="stroke-ink"/>
<text x="260.0" y="122.0" class="fig-small" text-anchor="middle">1</text>
<line x1="360.0" y1="96.0" x2="360.0" y2="104.0" class="stroke-ink"/>
<text x="360.0" y="122.0" class="fig-small" text-anchor="middle">1.5</text>
<line x1="460.0" y1="96.0" x2="460.0" y2="104.0" class="stroke-ink"/>
<text x="460.0" y="122.0" class="fig-small" text-anchor="middle">2</text>
<line x1="560.0" y1="96.0" x2="560.0" y2="104.0" class="stroke-ink"/>
<text x="560.0" y="122.0" class="fig-small" text-anchor="middle">2.5</text>
<circle cx="140.0" cy="100.0" r="6" class="fill-a"/>
<text x="140.0" y="84.0" class="fig-small" text-anchor="middle">v_u = 0.4</text>
<circle cx="260.0" cy="100.0" r="6" class="fill-b"/>
<text x="260.0" y="84.0" class="fig-small" text-anchor="middle">v_c = 1.0</text>
<circle cx="500.0" cy="100.0" r="6" class="fill-c"/>
<text x="500.0" y="84.0" class="fig-small" text-anchor="middle">guided = 2.2</text>
<line x1="268.0" y1="60.0" x2="492.0" y2="60.0" class="stroke-ink" marker-end="url(#c052b-arrow)"/>
<text x="380.0" y="50.0" class="fig-small" text-anchor="middle">alpha (v_c - v_u) = 2 x 0.6</text>
<text x="310.0" y="152.0" class="fig-small" text-anchor="middle">velocity</text>
</svg>
Figure 2. Guidance starts at the conditional velocity and keeps going in the direction that separates it from the unconditional one. With alpha = 2 it travels twice the gap, 1.2, past v_c.
```

This is the w-convention, where $\alpha = 0$ means no guidance, used by Voicebox and F5-TTS. In the s-convention of diffusers and AudioLDM, the same guidance is $s = 1 + \alpha$. F5-TTS's default $\alpha = 2$ is $s = 3$, and CosyVoice 2's 0.7 is $s = 1.7$. Guidance doubles the cost of each step, so stacking the two passes into one batch of $2B$ rows, as the code does, keeps it to one network call.

> [!WARNING]
> "CFG 2" means twice as much extra push in the w-convention as in the s-convention. Report which one you use.

## Spend the steps where they matter

Near $t = 0$ the input is almost pure noise, and the model decides the large-scale layout: where each sound goes and roughly how loud it is. Near $t = 1$ it only polishes detail. So an even time grid wastes steps on polish. F5-TTS's *sway sampling* moves the step boundaries toward the noise end:

$$
f_{\text{sway}}(u; s) = u + s\left(\cos\!\left(\tfrac{\pi}{2}u\right) - 1 + u\right)
$$

Read it aloud: take evenly spaced times $u$ and bend them by $s$ times a curve that is 0 at both ends. For $s < 0$ the times crowd toward 0. At $s = -1$ the formula simplifies to $1 - \cos(\pi u / 2)$, the cosine time scheduler CosyVoice 2 uses. In numpy:

```python
u = np.linspace(0, 1, nfe + 1)
t = u + s * (np.cos(np.pi / 2 * u) - 1 + u)
```

For $s = -1$ and 8 steps (Figure 3), the grid $u = i / 8$ becomes 0, 0.0192, 0.0761, 0.1685, 0.2929, 0.4444, 0.6173, 0.8049, and 1. The first step is 0.0192 long and the last is 0.1951, against 0.125 each for an even grid. The map stays increasing for $s$ between $-1$ and $2 / (\pi - 2) \approx 1.752$. Sway needs no retraining, and F5-TTS reports 2.53 percent word error rate on LibriSpeech-PC test-clean at NFE 16 with it, against 2.42 at NFE 32.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Two number lines for 8 sampling steps: evenly spaced times, and times warped by sway sampling with s = -1, which crowd toward 0">
<defs><marker id="c052c-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="60.0" y1="60.0" x2="540.0" y2="60.0" class="stroke-ink"/>
<text x="60.0" y="34.0" class="fig-bold" text-anchor="start">even grid</text>
<circle cx="60.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="120.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="180.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="240.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="300.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="360.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="420.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="480.0" cy="60.0" r="5" class="fill-a"/>
<circle cx="540.0" cy="60.0" r="5" class="fill-a"/>
<line x1="60.0" y1="120.0" x2="540.0" y2="120.0" class="stroke-ink"/>
<text x="60.0" y="94.0" class="fig-bold" text-anchor="start">sway, s = -1</text>
<circle cx="60.0" cy="120.0" r="5" class="fill-b"/>
<circle cx="69.2" cy="120.0" r="5" class="fill-b"/>
<circle cx="96.5" cy="120.0" r="5" class="fill-b"/>
<circle cx="140.9" cy="120.0" r="5" class="fill-b"/>
<circle cx="200.6" cy="120.0" r="5" class="fill-b"/>
<circle cx="273.3" cy="120.0" r="5" class="fill-b"/>
<circle cx="356.3" cy="120.0" r="5" class="fill-b"/>
<circle cx="446.4" cy="120.0" r="5" class="fill-b"/>
<circle cx="540.0" cy="120.0" r="5" class="fill-b"/>
<text x="60.0" y="150.0" class="fig-small" text-anchor="start">t = 0 (noise)</text>
<text x="540.0" y="150.0" class="fig-small" text-anchor="end">t = 1 (data)</text>
<text x="300.0" y="176.0" class="fig-small" text-anchor="middle">first sway step 0.0192 long, last 0.1951; even steps 0.125</text>
</svg>
Figure 3. Both rows run from noise at 0 to data at 1 in 8 steps. Notice that the sway row puts six of its nine times before 0.45, where the layout of the utterance is decided.
```

> [!KEY]
> The steps near noise decide the layout of the utterance, and the steps near data only polish it. Sway sampling spends more, smaller steps early.

## Stream with a chunk mask

A streaming TTS system must play the first words before it has generated the last ones. CosyVoice 2 (Du et al., 2024) runs a language model that emits speech tokens at 25 Hz, upsamples them to 50 Hz, and uses a flow-matching decoder to turn them into mel frames. The decoder's attention mask decides what each frame may see. With *chunks* of $M$ frames, a frame may attend to every frame in its own chunk and in earlier chunks, never later ones:

$$
\text{allowed}[i, j] = \left( \lfloor j / M \rfloor \le \lfloor i / M \rfloor \right)
$$

Read it aloud: query frame $i$ may look at key frame $j$ when the chunk of $j$ isn't later than the chunk of $i$. For six frames in chunks of two (Figure 4), frames 0 and 1 see frames 0 and 1, frames 2 and 3 see frames 0 to 3, and frames 4 and 5 see everything. Chunks of 1 give a fully causal mask, and one chunk as long as the clip gives full attention. In numpy:

```python
chunk = np.arange(n) // M
allowed = chunk[None, :] <= chunk[:, None]
```

```figure
<svg viewBox="0 0 600 250" role="img" aria-label="A six by six attention mask with chunks of two frames: frames 0 and 1 see frames 0 and 1, frames 2 and 3 see frames 0 to 3, frames 4 and 5 see all six">
<defs><marker id="c052d-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="180.0" y="60.0" class="fig-small" text-anchor="end">0</text>
<text x="205.0" y="32.0" class="fig-small" text-anchor="middle">0</text>
<rect x="191.0" y="41.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="41.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="41.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="281.0" y="41.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="311.0" y="41.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="341.0" y="41.0" width="28.0" height="28.0" class="fill-sunk"/>
<text x="180.0" y="90.0" class="fig-small" text-anchor="end">1</text>
<text x="235.0" y="32.0" class="fig-small" text-anchor="middle">1</text>
<rect x="191.0" y="71.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="71.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="71.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="281.0" y="71.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="311.0" y="71.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="341.0" y="71.0" width="28.0" height="28.0" class="fill-sunk"/>
<text x="180.0" y="120.0" class="fig-small" text-anchor="end">2</text>
<text x="265.0" y="32.0" class="fig-small" text-anchor="middle">2</text>
<rect x="191.0" y="101.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="101.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="101.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="281.0" y="101.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="311.0" y="101.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="341.0" y="101.0" width="28.0" height="28.0" class="fill-sunk"/>
<text x="180.0" y="150.0" class="fig-small" text-anchor="end">3</text>
<text x="295.0" y="32.0" class="fig-small" text-anchor="middle">3</text>
<rect x="191.0" y="131.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="131.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="131.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="281.0" y="131.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="311.0" y="131.0" width="28.0" height="28.0" class="fill-sunk"/>
<rect x="341.0" y="131.0" width="28.0" height="28.0" class="fill-sunk"/>
<text x="180.0" y="180.0" class="fig-small" text-anchor="end">4</text>
<text x="325.0" y="32.0" class="fig-small" text-anchor="middle">4</text>
<rect x="191.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="281.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="311.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="341.0" y="161.0" width="28.0" height="28.0" class="fill-b"/>
<text x="180.0" y="210.0" class="fig-small" text-anchor="end">5</text>
<text x="355.0" y="32.0" class="fig-small" text-anchor="middle">5</text>
<rect x="191.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="221.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="251.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="281.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="311.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<rect x="341.0" y="191.0" width="28.0" height="28.0" class="fill-b"/>
<text x="280.0" y="14.0" class="fig-small" text-anchor="middle">key frame j</text>
<text x="40.0" y="130.0" class="fig-small" text-anchor="start">query frame i</text>
<text x="390.0" y="100.0" class="fig-small" text-anchor="start">orange: i may look at j</text>
<text x="390.0" y="120.0" class="fig-small" text-anchor="start">gray: blocked (j is in a later chunk)</text>
</svg>
Figure 4. The mask is a staircase of 2 by 2 blocks. Notice that the allowed region includes the block on the diagonal, so frames see their own chunk, and nothing to its right.
```

During training, CosyVoice 2 samples one of four masks for each batch: non-causal, fully causal, chunks of $M$, and chunks of $2M$. One set of weights then serves offline synthesis, which sees everything, and streaming, which waits for only one chunk. The flow's ten sampling steps run on each chunk as it arrives.

## From mel back to sound

The model generates 80-band log-mel frames. A real system hands them to a neural vocoder such as HiFi-GAN, Vocos, or BigVGAN. This chapter reuses the tools you built earlier, so the chain is exact and inspectable. First undo the mel filterbank $W$, which maps 513 FFT bins to 80 bands, so many spectra share each mel frame. The pseudo-inverse $W^+$ picks the one with the smallest total energy. Negative values can appear between filters, so clip them to 0:

$$
\hat{S} = \sqrt{\max\left(W^+ \exp(\text{log-mel}),\ 0\right)}
$$

Read it aloud: exponentiate the log-mel back to power, solve for a power spectrum by least squares, clip, and take the square root. Then Griffin-Lim invents a phase that fits the magnitude. The smoothing blurs the harmonics, so the result sounds whispery even for a real recording. The **Run** block plays real mels through the same chain so you can hear that ceiling, apart from the model's own errors.

> [!RECAP]
> - The masked loss averages the squared velocity error over masked frames only, and the context is $(1 - m) \odot x_1$.
> - Guidance is $v_c + \alpha(v_c - v_u)$ in the w-convention, which equals the s-convention with $s = 1 + \alpha$.
> - Sway sampling with $s = -1$ is $t = 1 - \cos(\pi u / 2)$, with small steps near noise.
> - A chunk mask lets frame $i$ attend to frame $j$ when $\lfloor j/M \rfloor \le \lfloor i/M \rfloor$.
> - The pseudo-inverse filterbank and Griffin-Lim turn a log-mel into sound.

## Build it step by step

Build the functions in the order the checks test them.

1. **`sway(u, s)`** returns the warped times, same type as `u`. Pick `torch.cos` for a tensor and `np.cos` otherwise. At `u = 0.5`, $s = -1$ gives 0.2929.
2. **`infilling_batch(x0, x1, t, mask)`** returns `(x_t, context, target)`, each `(B, F, N)`. Reshape `t` to `(B, 1, 1)` and `mask` to `(B, 1, N)`. Mistake: building the context from `x_t` instead of the clean `x1`.
3. **`masked_flow_loss(v_pred, target, mask)`** multiplies the squared error by the mask, sums, and divides by `mask.sum() * F`. Taking `.mean()` over all elements makes the loss shrink when fewer frames are masked.
4. **`cfg_velocity(v_cond, v_uncond, alpha)`** is one line. `alpha = 0` returns `v_cond`.
5. **`sample(velocity, x0, cond, null_cond, nfe, sway_s, cfg_alpha)`** builds the grid with `sway(torch.linspace(0, 1, nfe + 1), sway_s)`, then loops over steps. With guidance, call `velocity` once on `torch.cat([x, x])` with `torch.cat([cond, null_cond])` and a `(2B,)` time; without it, call once on `x`. Mistake: adding a fixed `dt`, which ignores the warp.
6. **`chunk_causal_mask(n_frames, chunk_size)`** is one broadcast comparison of chunk indices.
7. **`mel_to_magnitude(mel_power, filterbank)`** uses `np.linalg.pinv`, `np.maximum`, and `np.sqrt`. Feed it power, not log-mel.

## Where this breaks

- **The null condition drifts.** If training zeroes the label embedding and sampling uses a learned null token, the unconditional branch is meaningless and guidance amplifies it. Use one null, in one place, for both.
- **Guidance over-saturates.** Large $\alpha$ makes every sample the most typical one: loud, flat, and all alike. The run's classifier accuracy rises with guidance while the variety falls.
- **The reference frames are left in.** An infilling model regenerates the whole tensor. Discard the frames that belonged to the reference before you vocode, and compute the loss only on masked frames.

## The exercise

You build `sway`, `infilling_batch`, `masked_flow_loss`, `cfg_velocity`, `sample`, `chunk_causal_mask`, and `mel_to_magnitude`. The checks confirm the sway grid, its identity with $1 - \cos(\pi u / 2)$, and its range of monotonicity. They verify the infilling inputs and that unmasked frames get exactly zero gradient, the two guidance conventions, and that `sample` calls the field once per step with $2B$ rows on the sway grid. On an exact two-class Gaussian field, `sample` must reach the class and guidance must push past it. The last checks compare your chunk mask with a brute-force rule, your mel inverse with the pseudo-inverse, and train a tiny conditional flow with your pieces.

Press **Run** to train on `fsdd`, resampled to 16 kHz and cut to 0.6 s, with digit labels as the condition, a null label 20 percent of the time, and whole or 70 to 100 percent masks. On a laptop CPU, the small version trained for 700 steps in about 30 s. A linear classifier that scores 97 percent on real held-out digits read 89 percent of its unguided samples and 100 percent of its guided ones as the right digit. A 5,000-step run with a 1.2M-parameter model read 97 percent unguided, and its mels show formants and fricatives.
