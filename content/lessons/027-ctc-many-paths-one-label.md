---
title: "CTC: many paths, one label"
slug: 027-ctc-many-paths-one-label
part: "Part 4 — Neural speech recognition"
summary: Train a recognizer from transcripts alone by letting the network emit a blank, then summing the probability of every path that spells the label.
minutes: 40
gpu: false
objectives:
  - Explain why a frame classifier cannot train on a transcript alone, and why one fixed alignment is not an answer.
  - Apply the CTC collapse function, and say why a repeated label needs a blank between its copies.
  - Compute the probability of a label as a sum over paths, by hand for three frames and by brute force in code.
  - State how many frames a transcript needs, and what the loss does when it does not fit.
prereqs:
  - 026-neural-acoustic-models
exercise: 027-ctc-many-paths-one-label
---

# CTC: many paths, one label

A frame classifier needs a label for every frame, and real speech does not come with them. LibriSpeech gives you an audio file and the sentence that was read, nothing more. Somewhere in those 1,500 frames is the "h" of "hello", but nobody wrote down where.

The obvious fix is to guess the labels. Spread the five letters of "hello" evenly over the frames, so 300 frames each, and train on that. It fails the moment the speaker pauses before the first word or stretches the "o". The guessed boundaries are wrong, so the network learns from wrong labels, and the errors are not random. Every utterance gets the same wrong timing.

**Connectionist temporal classification** (CTC), from Graves and colleagues in 2006, takes a different route. It does not pick an alignment. It asks the network for a probability at every frame, then adds up the probability of every alignment that spells the right transcript. This is like grading an essay on whether it says the right thing, without caring which line each sentence starts on. This chapter builds that sum on tiny inputs, where you can list every term. The next chapter makes it fast.

## What you already know

- A neural acoustic model turns each frame into a probability over symbols, and a strided front end sets how many frames $T$ there are, as [Neural acoustic models](/lessons/026-neural-acoustic-models) showed.
- Probabilities of independent events multiply.
- Probabilities of events that cannot both happen add.
- A log probability turns a product into a sum, and `np.logaddexp(a, b)` computes $\log(e^a + e^b)$ without overflow.

## A symbol for "nothing here"

At each of $T$ frames, the network outputs a distribution $y^t$ over $C$ symbols. CTC adds one symbol to the alphabet, the **blank**, written `_` or $\varnothing$, at index 0. The labels sit at indices 1 to $C - 1$. A **path** $\pi$ picks one symbol for each frame, so a path is a $T$-long string such as `aa_ab__`.

The blank earns its place by answering two questions. What does the network say during silence, when no letter is being spoken? It says blank. And how do you spell a double letter, such as the "ll" in "hello", when the network sees the same sound in a row for several frames? A blank between the two letters says that they are separate.

A path becomes a transcript through the **collapse function** $\mathcal{B}$, in two steps. First merge each run of the same symbol into one. Then delete the blanks. Figure 1 applies it to one path.

```figure
<svg viewBox="0 0 600 270" role="img" aria-label="The path a a blank a b blank blank is shortened by merging runs to a blank a b blank, then by dropping blanks to a a b">
  <defs><marker id="c026-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="fill-ink"/></marker></defs>
  <rect x="150" y="40" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="169" y="60" text-anchor="middle" class="fig-mono">a</text>
  <rect x="194" y="40" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="213" y="60" text-anchor="middle" class="fig-mono">a</text>
  <rect x="238" y="40" width="38" height="30" rx="4" class="fill-sunk stroke-muted"/>
  <text x="257" y="60" text-anchor="middle" class="fig-mono">_</text>
  <rect x="282" y="40" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="301" y="60" text-anchor="middle" class="fig-mono">a</text>
  <rect x="326" y="40" width="38" height="30" rx="4" class="fill-soft-c stroke-c"/>
  <text x="345" y="60" text-anchor="middle" class="fig-mono">b</text>
  <rect x="370" y="40" width="38" height="30" rx="4" class="fill-sunk stroke-muted"/>
  <text x="389" y="60" text-anchor="middle" class="fig-mono">_</text>
  <rect x="414" y="40" width="38" height="30" rx="4" class="fill-sunk stroke-muted"/>
  <text x="433" y="60" text-anchor="middle" class="fig-mono">_</text>
  <rect x="150" y="120" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="169" y="140" text-anchor="middle" class="fig-mono">a</text>
  <rect x="194" y="120" width="38" height="30" rx="4" class="fill-sunk stroke-muted"/>
  <text x="213" y="140" text-anchor="middle" class="fig-mono">_</text>
  <rect x="238" y="120" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="257" y="140" text-anchor="middle" class="fig-mono">a</text>
  <rect x="282" y="120" width="38" height="30" rx="4" class="fill-soft-c stroke-c"/>
  <text x="301" y="140" text-anchor="middle" class="fig-mono">b</text>
  <rect x="326" y="120" width="38" height="30" rx="4" class="fill-sunk stroke-muted"/>
  <text x="345" y="140" text-anchor="middle" class="fig-mono">_</text>
  <rect x="150" y="200" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="169" y="220" text-anchor="middle" class="fig-mono">a</text>
  <rect x="194" y="200" width="38" height="30" rx="4" class="fill-soft-a stroke-a"/>
  <text x="213" y="220" text-anchor="middle" class="fig-mono">a</text>
  <rect x="238" y="200" width="38" height="30" rx="4" class="fill-soft-c stroke-c"/>
  <text x="257" y="220" text-anchor="middle" class="fig-mono">b</text>
  <text x="14" y="60" class="fig-bold">path</text>
  <text x="14" y="140" class="fig-bold">merged</text>
  <text x="14" y="220" class="fig-bold">transcript</text>
  <line x1="300" y1="76" x2="300" y2="114" class="stroke-ink" marker-end="url(#c026-arrow)"/>
  <text x="312" y="100">merge each run into one symbol</text>
  <line x1="300" y1="156" x2="300" y2="194" class="stroke-ink" marker-end="url(#c026-arrow)"/>
  <text x="312" y="180">delete the blanks</text>
  <text x="14" y="256" class="fig-small">The blank between the two a's in the path survives the merge, so both a's reach the transcript.</text>
</svg>
Figure 1. The collapse function shortens a path in two steps. Notice that the blank between the third and fourth frames keeps the two a's apart, so the transcript keeps both.
```

The order of the two steps is the whole design. Merging first lets one letter last several frames: `aaa` becomes `a`. The blank lets a letter repeat: $\mathcal{B}(\texttt{a\_a}) = \texttt{aa}$, but $\mathcal{B}(\texttt{aa}) = \texttt{a}$. If you deleted the blanks first, `a_a` would become `aa` and then merge into `a`, so no path could ever spell "ll". A path of only blanks, `___`, collapses to the empty transcript.

## Many paths, one transcript

Now ask the question the loss needs: how likely is the transcript "a", given the audio? Many different paths collapse to it, and each is a different story about which frames the "a" occupied. The network treats frames as independent, so the probability of one path is the product of one probability per frame. A transcript can come from any of its paths, and the paths exclude each other, so the transcript's probability is the sum:

$$
p(\mathbf{l} \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y^t_{\pi_t}, \qquad \mathcal{L}_{\text{CTC}} = -\log p(\mathbf{l} \mid x)
$$

Read it aloud: for every path in $\mathcal{B}^{-1}(\mathbf{l})$, the set of paths that collapse to $\mathbf{l}$, multiply the probability of its symbol at each frame. Add up all those products. The loss is minus the log of the total, so it is small when the network gives the right transcript most of its probability.

Work a case small enough to list every path. The vocabulary is blank and `a`. Three frames give $P(a)$ of 0.6, 0.3, and 0.8, so the blank probabilities are 0.4, 0.7, and 0.2. The transcript is `a`. Exactly six paths collapse to it: `a__`, `_a_`, `__a`, `aa_`, `_aa`, and `aaa`. Figure 2 multiplies out each one.

```figure
<svg viewBox="0 0 600 322" role="img" aria-label="Six paths that collapse to the label a in three frames, each with its probability as a product and a bar; the probabilities sum to 0.608">
  <text x="14" y="22">P(a) in frames 1, 2, 3 is 0.6, 0.3, 0.8, so P(blank) is 0.4, 0.7, 0.2.</text>
  <text x="30" y="52" class="fig-small">frame 1 2 3</text>
  <text x="170" y="52" class="fig-small">product over frames</text>
  <text x="310" y="52" class="fig-small">probability</text>
  <rect x="30" y="64" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="48" y="82" text-anchor="middle" class="fig-mono">a</text>
  <rect x="70" y="64" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="88" y="82" text-anchor="middle" class="fig-mono">_</text>
  <rect x="110" y="64" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="128" y="82" text-anchor="middle" class="fig-mono">_</text>
  <text x="170" y="82" class="fig-mono">0.6 × 0.7 × 0.2</text>
  <text x="310" y="82" class="fig-mono">0.084</text>
  <rect x="380" y="68" width="67.2" height="18" class="fill-b"/>
  <rect x="30" y="98" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="48" y="116" text-anchor="middle" class="fig-mono">_</text>
  <rect x="70" y="98" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="88" y="116" text-anchor="middle" class="fig-mono">a</text>
  <rect x="110" y="98" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="128" y="116" text-anchor="middle" class="fig-mono">_</text>
  <text x="170" y="116" class="fig-mono">0.4 × 0.3 × 0.2</text>
  <text x="310" y="116" class="fig-mono">0.024</text>
  <rect x="380" y="102" width="19.2" height="18" class="fill-b"/>
  <rect x="30" y="132" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="48" y="150" text-anchor="middle" class="fig-mono">_</text>
  <rect x="70" y="132" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="88" y="150" text-anchor="middle" class="fig-mono">_</text>
  <rect x="110" y="132" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="128" y="150" text-anchor="middle" class="fig-mono">a</text>
  <text x="170" y="150" class="fig-mono">0.4 × 0.7 × 0.8</text>
  <text x="310" y="150" class="fig-mono">0.224</text>
  <rect x="380" y="136" width="179.2" height="18" class="fill-b"/>
  <rect x="30" y="166" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="48" y="184" text-anchor="middle" class="fig-mono">a</text>
  <rect x="70" y="166" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="88" y="184" text-anchor="middle" class="fig-mono">a</text>
  <rect x="110" y="166" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="128" y="184" text-anchor="middle" class="fig-mono">_</text>
  <text x="170" y="184" class="fig-mono">0.6 × 0.3 × 0.2</text>
  <text x="310" y="184" class="fig-mono">0.036</text>
  <rect x="380" y="170" width="28.8" height="18" class="fill-b"/>
  <rect x="30" y="200" width="36" height="26" rx="4" class="fill-sunk stroke-muted"/>
  <text x="48" y="218" text-anchor="middle" class="fig-mono">_</text>
  <rect x="70" y="200" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="88" y="218" text-anchor="middle" class="fig-mono">a</text>
  <rect x="110" y="200" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="128" y="218" text-anchor="middle" class="fig-mono">a</text>
  <text x="170" y="218" class="fig-mono">0.4 × 0.3 × 0.8</text>
  <text x="310" y="218" class="fig-mono">0.096</text>
  <rect x="380" y="204" width="76.8" height="18" class="fill-b"/>
  <rect x="30" y="234" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="48" y="252" text-anchor="middle" class="fig-mono">a</text>
  <rect x="70" y="234" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="88" y="252" text-anchor="middle" class="fig-mono">a</text>
  <rect x="110" y="234" width="36" height="26" rx="4" class="fill-soft-a stroke-a"/>
  <text x="128" y="252" text-anchor="middle" class="fig-mono">a</text>
  <text x="170" y="252" class="fig-mono">0.6 × 0.3 × 0.8</text>
  <text x="310" y="252" class="fig-mono">0.144</text>
  <rect x="380" y="238" width="115.2" height="18" class="fill-b"/>
  <line x1="30" y1="272" x2="570" y2="272" class="stroke-muted"/>
  <text x="30" y="294" class="fig-bold">sum = 0.608, so the loss is −ln 0.608 = 0.4976</text>
  <text x="30" y="312" class="fig-small">The likeliest single path, __a, has probability 0.224, well under the total.</text>
</svg>
Figure 2. Six paths spell "a", and their probabilities add to 0.608. The likeliest path alone, __a, carries only 0.224 of that.
```

So $p(a \mid x) = 0.608$ and the loss is $-\ln 0.608 = 0.49758$, which is exactly what `F.ctc_loss(..., reduction="sum")` returns for this input. With every probability at 0.5, each of the six paths has probability $1/8$, so $p = 0.75$.

### Why the sum, and not the best path

Why not train on the single best path? Because the best path can lose to a transcript whose probability is spread thinner. Take two frames, each with blank at 0.6 and `a` at 0.4. The best path is `__`, with probability $0.6 \times 0.6 = 0.36$, and it spells the empty transcript. Three paths spell `a`: `a_` at $0.4 \times 0.6 = 0.24$, `_a` at $0.6 \times 0.4 = 0.24$, and `aa` at $0.4 \times 0.4 = 0.16$. Their sum is $0.64$, so "a" is almost twice as likely as nothing, though no single path to it beats `__`. The loss follows the sum, and the sum is the right thing to follow.

> [!KEY]
> CTC does not choose an alignment. The probability of a transcript is the sum, over every path that collapses to it, of the product of the per-frame probabilities.

## How many frames does a transcript need?

A path spends at least one frame on every label. And it spends one extra frame, a blank, between every pair of identical neighbors, because the merge step would otherwise fuse them. So a transcript with $L$ labels and $r$ adjacent repeats needs

$$
T \ge L + r
$$

Read it aloud: one frame per label, plus one per repeat. In numpy that is `len(target) + sum(a == b for a, b in zip(target, target[1:]))`. With fewer frames there is no path, $p = 0$, and the loss is $+\infty$.

You can check it on the example. The transcript `aa` in three frames has one path, `a_a`, with probability $0.6 \times 0.7 \times 0.8 = 0.336$. In two frames it has none, because `aa` collapses to `a`. The demo below draws the paths for any label as a lattice, with states down and frames across. You use that lattice in the next chapter to add the paths up quickly.

```demo
{"id": "ctc-lattice", "caption": "Type the label eee and lower T until the loss becomes infinite: it needs five frames, one per letter plus a blank between each pair of copies."}
```

> [!WARNING]
> $T$ is the number of frames after the subsampling front end. A 4x front end turns one second into 24 frames, so a fast speaker's character transcript can need more frames than the model has. torch then returns `inf`, unless you pass `zero_infinity=True`.

## Why you cannot stop at brute force

The sum over paths is easy to write and hopeless to run. For a transcript with no adjacent repeats, the number of paths is $\binom{T + L}{2L}$. For 20 letters in 150 frames that is $\binom{170}{40}$, about $10^{39}$ paths, and a real recognizer has a larger alphabet and longer sentences. The exercise's brute-force loss enumerates `itertools.product(range(C), repeat=T)`, which has $C^T$ terms, so it works only on tiny inputs. It is still valuable: it is the reference the fast algorithm in [Computing CTC efficiently](/lessons/028-ctc-forward-backward) must match to the last digit.

> [!RECAP]
> - The blank is a symbol for "no label here". It is index 0 in torch and Hugging Face.
> - The collapse function merges runs, then deletes blanks. A repeated label needs a blank between its copies.
> - $p(\mathbf{l} \mid x)$ is the sum over paths that collapse to $\mathbf{l}$ of the product of per-frame probabilities.
> - A transcript needs $T \ge L + r$ frames, or its loss is infinite.

## Build it step by step

The checks test the functions in this order.

1. **`collapse(path, blank)`.** Takes a sequence of ints and returns a list. Walk the path, keep a symbol when it differs from the previous one and is not blank. The common mistake is removing blanks first, which merges `a_a` into `a`.
2. **`path_log_prob(log_probs, path)`.** Takes `(T, C)` log probabilities and a path of $T$ symbols. `log_probs[np.arange(T), path].sum()` picks one entry per frame. The common mistake is multiplying probabilities, which underflows for long paths.
3. **`paths_for(target, n_frames, n_symbols, blank)`.** Enumerate `itertools.product(range(n_symbols), repeat=n_frames)` and keep the paths whose collapse equals the target. The common mistake is comparing a tuple with a list, which is never equal.
4. **`ctc_loss_brute_force(log_probs, target, blank)`.** Negate the log-sum-exp of every path's log probability, with `np.logaddexp.reduce`. The common mistake is returning `nan` instead of `inf` when no path fits.
5. **`min_frames(target)`.** The target's length plus the number of equal adjacent pairs. The common mistake is counting every repeated label instead of every adjacent pair.

## Where this breaks

**Independence between frames.** CTC treats each frame's output as independent given the audio, so it cannot learn that "their" and "there" depend on the next word. An external language model, fused during decoding, fixes much of this, as [Language model fusion and hotwords](/lessons/030-language-model-fusion) shows.

**Infeasible targets.** Too much subsampling, or a long character transcript, makes $T < L + r$. The loss is infinite and its gradient is undefined. Filter such utterances, use subword units, or set `zero_infinity=True`.

**Blank conventions.** torch and Hugging Face's `Wav2Vec2ForCTC` put blank at index 0. torchaudio's `rnnt_loss` and many NeMo models put it last. Mixing them trains a model to emit a letter as silence.

## The exercise

You build `collapse`, `path_log_prob`, `paths_for`, `ctc_loss_brute_force`, and `min_frames`. Press **Run** to list the six paths of the worked example with their probabilities, compare your loss with torch, and plot how the number of paths grows with the number of frames.

The checks confirm `collapse` on repeats and blanks, and the probability of single paths. They require exactly the six paths that spell `a` in three frames and the one path that spells `aa`. They check the loss against the worked $-\ln 0.608$ and against `torch.nn.functional.ctc_loss` on targets with repeated labels. They confirm that `a` at 0.64 beats the empty transcript at 0.36, and that a loss is finite at `min_frames` and infinite one frame below it.

> [!KEY]
> What you now know: you can train on a transcript without alignments, because the loss adds up every alignment that spells it. That answers the opening question: nobody has to say where the "h" is, because every place it could be is already in the sum.
