---
title: Few-step generation and audio metrics
slug: 38-few-step-generation-and-audio-metrics
part: "Part 7 — Diffusion and flow matching"
summary: Teach a flow to jump from noise to data in one step with shortcut models and MeanFlow, then score generators with FAD and CLAP.
minutes: 75
gpu: false
objectives:
  - Explain why one step of an ordinary flow returns the data mean, and what a one-step model must learn instead.
  - Build the shortcut-model loss, whose target is two half steps of the model itself.
  - Derive and compute the MeanFlow target for the average velocity with one Jacobian-vector product.
  - Compute the Fréchet audio distance from embeddings, and correct its sample-size bias with FAD-infinity.
  - Compute a CLAP score, and say what FAD and CLAP can't tell you, for speech and for music.
prereqs:
  - 36-flow-matching
  - 37-flow-matching-for-speech
  - 34-samplers-scores-and-guidance
exercise: 38-few-step-generation-and-audio-metrics
---

# Few-step generation and audio metrics

A flow-matching TTS model such as F5-TTS spends 16 to 32 network evaluations
on each utterance, and guidance doubles that. A voice agent that must answer
in a fraction of a second, or a music tool that renders while you drag a
slider, wants one to eight. This lesson builds two ways to get there without
a separate teacher: shortcut models and MeanFlow.

The second half is about measuring. Once a generator is fast, you need a
number that says whether it still sounds right. The Fréchet audio distance
(FAD) compares the statistics of generated and real audio, and the CLAP score
checks that sound matches its text prompt. Both are used everywhere, and both
have biases you must know before you trust a table of results.

The time convention stays the same as in the last two lessons: $t = 0$ is
noise and $t = 1$ is data.

## What you already know

- A flow's velocity field $v(x, t)$ moves noise at $t = 0$ to data at $t = 1$,
  and Euler integrates it one step at a time.
- One Euler step from $t = 0$ lands on the data mean, because the field there
  is the average of every straight path through the noise sample.
- Reflow straightens a flow by retraining it on its own noise-and-sample pairs.
- A multivariate Gaussian is described by a mean vector and a covariance
  matrix.
- Cosine similarity measures the angle between two vectors, whatever their
  lengths.

## What a one-step model must learn

The velocity $v(x, t)$ is the direction the sample moves right now. To jump
from time $s$ to a later time $e$ in one step, you need the *average* velocity
over the whole interval, the chord of the trajectory rather than its tangent:

$$
u(x_s, s, e) = \frac{1}{e - s} \int_s^e v(x_\tau, \tau)\, d\tau,
\qquad x_e = x_s + (e - s)\, u(x_s, s, e)
$$

Read it aloud: the average velocity is the total displacement from $s$ to $e$
divided by the time it took, so one step with it lands exactly where the
trajectory lands. When $e = s$, it is the ordinary velocity.

Take the Gaussian flow of lesson 36: data $\mathcal{N}(2, 0.5^2)$ and a noise
sample $x_0 = 1.3$, whose trajectory ends at 2.65. The velocity at the start
is $v = 0.7$, so one Euler step stops at 2.0, the mean. The average velocity
over $[0, 1]$ is $2.65 - 1.3 = 1.35$. The trajectory passes through 1.726722 at
$t = 0.5$, so the two halves have average velocities
$(1.726722 - 1.3) / 0.5 = 0.853444$ and $(2.65 - 1.726722) / 0.5 = 1.846556$.
Their mean is exactly 1.35. Averages over consecutive halves compose into the
average over the whole, and both methods in this lesson are built on that
fact.

> [!KEY]
> A one-step generator must learn the average velocity over an interval, not
> the velocity at a point. The average over a long interval is the mean of the
> averages over its halves.

## Distillation, in brief

The oldest route to few steps is a teacher. *Progressive distillation*
(Salimans and Ho) trains a student to match two DDIM steps of the teacher in
one, then halves again, and introduced v-prediction along the way.
*Consistency models* (Song et al.) learn a function that maps every point of a
trajectory to its end. In this course's convention, the natural form is
$f(x_t, t) = x_t + (1 - t)\,\hat{v}$, which equals $x_t$ at $t = 1$ by
construction. CoMoSpeech uses consistency distillation for one-step TTS, and
WaveFM for a one-step vocoder. *Distribution matching distillation* trains a
one-step generator so its outputs match the teacher's distribution, and
DMOSpeech uses it to bring an F5-style model to 4 steps. Stable Audio Open
Small skips the teacher and fine-tunes with an adversarial loss, then samples
in 8 steps.

Shortcut models and MeanFlow need no teacher. They train from scratch and
learn the jump alongside the flow.

## Shortcut models

A *shortcut model* (Frans et al.) adds the step size $d$ as a third input,
$s_\theta(x_t, t, d)$. At $d = 0$ it is an ordinary flow, trained on
$x_1 - x_0$. For $d > 0$ it learns from itself: one step of size $2d$ must
equal two steps of size $d$:

$$
s_\theta(x_t, t, 2d) \leftarrow \tfrac{1}{2}\, s_\theta(x_t, t, d) + \tfrac{1}{2}\, s_\theta(x'_{t+d}, t + d, d),
\qquad x'_{t+d} = x_t + d\, s_\theta(x_t, t, d)
$$

Read it aloud: take a step of size $d$, then another from where it lands, and
teach the big step to point along the average of the two. The arrow means the
right side is a fixed target, computed without gradient. Small steps are
accurate because they are nearly flow matching. Each doubling passes that
accuracy up, so the largest step, $d = 1$, becomes a one-step sampler. In
torch:

```python
with torch.no_grad():
    first = model(x_t, t, d)
    second = model(x_t + d[:, None] * first, t + d, d)
target = 0.5 * (first + second)
```

The **Run** block trains a quarter of each batch on self-consistency, with
$d$ from $1/32$ to $1/2$, and starts each of those rows at a multiple of $2d$
so that $t + 2d \le 1$. The paper uses 128 base steps, so $d$ takes 8 values
from $1/128$ to 1.

## MeanFlow

*MeanFlow* (Geng et al.) learns the average velocity $u_\theta(x, s, e)$
directly. There's no integral in the training data, so it uses an identity
instead. Multiply the definition by $e - s$ and differentiate with respect to
the start time $s$, holding $e$ fixed and letting $x_s$ move along the flow:

$$
u(x_s, s, e) = v(x_s, s) + (e - s)\,\frac{d}{ds} u(x_s, s, e),
\qquad \frac{d}{ds} u = v \cdot \nabla_x u + \partial_s u
$$

Read it aloud: the average velocity equals the velocity at the start, plus a
correction for how the average changes as the start slides forward along the
trajectory. The total derivative has two parts: $u$ changes because $x$ moves
with velocity $v$, and because $s$ itself moves. In the Gaussian example,
$u = 1.35$ and $v = 0.7$, so $du/ds$ must be 0.65, which a finite difference
along the trajectory confirms.

Training plugs the conditional velocity $x_1 - x_0$ in for $v$, as flow
matching does, and regresses $u_\theta$ onto the right side with the gradient
stopped. One forward-mode *Jacobian-vector product* (JVP) returns $u_\theta$
and its derivative along the tangent $(v, 1, 0)$ for $(x, s, e)$:

```python
_, du_ds = torch.func.jvp(u_fn, (x, s, e), (v, torch.ones_like(s), torch.zeros_like(e)))
target = (v + (e - s)[:, None] * du_ds).detach()
```

To sample in one step, compute $x_1 = x_0 + u_\theta(x_0, 0, 1)$.

> [!WARNING]
> The MeanFlow paper puts data at $t = 0$ and writes
> $u = v - (t - r)\,\tfrac{d}{dt}u$. Its $t$ is $1 - s$ here, its $r$ is
> $1 - e$, and its velocities point the other way, which turns its minus sign
> into this lesson's plus. Copy the paper's formula into a noise-at-0 code
> base and the model learns to run backward.

On the eight-blob ring of the **Run** block, one step from a plain flow gives
an FAD of 96.7 on note embeddings, against 0.38 for the shortcut model and
0.39 for MeanFlow. The plain flow needs 8 steps to reach 0.16.

## Measure a generator with FAD

The *Fréchet audio distance* (Kilgour et al.) runs real and generated clips
through a fixed audio embedding model, fits a Gaussian to each cloud of
embeddings, and measures how far apart the two Gaussians are:

$$
\text{FAD} = \lVert \mu_r - \mu_g \rVert^2 + \operatorname{tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Read it aloud: the squared distance between the two means, plus a term that
is zero when the covariances match and grows as their shapes differ. In one
dimension it is $(\mu_r - \mu_g)^2 + (\sigma_r - \sigma_g)^2$, so
$\mathcal{N}(0, 1)$ against $\mathcal{N}(1, 2^2)$ gives $1 + 1 = 2$. With
diagonal covariances it is $\lVert \Delta\mu \rVert^2 + \sum_i (\sigma_{r,i} - \sigma_{g,i})^2$.
In numpy:

```python
root = scipy.linalg.sqrtm(cov_r @ cov_g).real
fad = np.sum((mu_r - mu_g) ** 2) + np.trace(cov_r + cov_g - 2 * root)
```

The original embedding is VGGish: 128 numbers per 0.96 s of 64-band log-mel
at 16 kHz. Stable Audio reports it with OpenL3, and `fadtk` (Gui et al.)
supports CLAP, EnCodec, MERT, and others.

### What biases FAD

FAD looks like one objective number. It isn't, for the following reasons:

- **Sample size.** An estimated covariance is noisy, and the noise always
  adds distance, so FAD falls as the sets grow. In the **Run** block, two
  independent sets of 2,000 real notes score 0.030, and one generator scores
  0.461 with 100 samples and 0.139 with 1,600. Compare only at equal sizes.
  *FAD-infinity* (Gui et al.) computes FAD at several sizes $N$, fits a line
  against $1/N$, and reports its value at $1/N = 0$, which is 0.102 for that
  generator.
- **The embedding.** Change the embedding model and every number changes. On
  raw 2-D points instead of note embeddings, the one-step flow scores 3.13
  rather than 96.7. Gui et al. found that the embedding and the reference set
  both decide how well FAD tracks perceived quality. A FAD without its
  embedding's name means nothing.
- **The Gaussian.** FAD sees only a mean and a covariance. Two sets with the
  same mean and covariance score 0 even when one clusters in a few tight modes
  and the other spreads smoothly.

Per-song FAD, one clip against the reference statistics, helps you find
outliers in a generated set.

> [!TIP]
> `scipy.linalg.sqrtm` can return complex values from rounding. Take the real
> part, and check that the imaginary part you dropped is tiny.

## Check the prompt with a CLAP score

FAD says whether generated audio looks like real audio, not whether it
matches its prompt. The *CLAP score* embeds each text prompt and its generated
clip with a contrastive language-audio model and averages their cosine
similarity:

$$
\text{CLAP} = \frac{1}{n}\sum_{i=1}^{n} \frac{a_i \cdot b_i}{\lVert a_i \rVert\, \lVert b_i \rVert}
$$

Read it aloud: for each prompt, the cosine of the angle between the text
embedding $a_i$ and the audio embedding $b_i$, averaged over prompts. For
example, text $(1, 0)$ with audio $(0.6, 0.8)$ scores 0.6, and text $(0, 1)$
with audio $(0, 2)$ scores 1, so the CLAP score is 0.8. Some papers clip at 0
or multiply by 100, and the score depends on the CLAP checkpoint, so report
both. For speech, lesson 29's measures fit better: word error rate from an
ASR model and speaker similarity.

## A note on music

Music is where these metrics are weakest and the models are largest. MusicGen
predicts EnCodec tokens at 50 Hz with 4 codebooks and the delay pattern from
lesson 32, at 300M to 3.3B parameters. Most other systems generate a
continuous latent with diffusion or flow matching. Stable Audio Open runs a
diffusion transformer on a 64-channel VAE latent at 21.5 Hz for up to 47 s of
44.1 kHz stereo. ACE-Step and DiffRhythm use flow matching for whole songs,
DiffRhythm with 32 Euler steps for up to 4 min 45 s. Few-step methods matter
most here, because a song is thousands of latent frames.

The honest part is evaluation. FAD fits a Gaussian to embeddings of short
windows, so it can't tell whether the lyrics are right or a chorus returns. A
CLAP score checks the words of the prompt, not the structure of a song.
Listening tests remain the reference. The data matters as much as the model:
MusicGen trained on 20,000 hours of licensed music, and Stable Audio Open only
on Creative Commons audio, which shapes what each can make.

> [!RECAP]
> - A one-step model learns the average velocity $u(x, s, e)$, and averages
>   over halves compose into the whole.
> - Shortcut models teach the step $2d$ to match two stop-gradient steps of
>   size $d$.
> - MeanFlow regresses onto $v + (e - s)\,du/ds$, computed with one JVP with
>   tangent $(v, 1, 0)$, in the noise-at-0 convention.
> - FAD is the Fréchet distance between Gaussians fitted to embeddings. It is
>   biased upward at small $N$, and it depends on the embedding.
> - The CLAP score is the mean cosine similarity of matching text and audio
>   embeddings.

## Build it step by step

1. **`shortcut_target(model, x_t, t, d)`** returns the two-step average,
   shape `(B, D)`, under `torch.no_grad()`. The common mistake is starting
   the second step at time `t` instead of `t + d`.
2. **`shortcut_loss(model, x0, x1, t, d)`** builds `x_t`, fills the target
   with `x1 - x0` where `d == 0` and with `shortcut_target` elsewhere, and
   asks the model for step `torch.where(d == 0, d, 2 * d)`. Return the mean
   squared error.
3. **`meanflow_target(u_fn, x, s, e, v)`** calls `torch.func.jvp` once and
   returns the detached target. Check it: at `e == s` it returns `v`.
4. **`frechet_distance(mu1, cov1, mu2, cov2)`** uses `scipy.linalg.sqrtm` and
   `np.trace`. Accept 1-D inputs with `np.atleast_1d` and `np.atleast_2d`.
5. **`fad(emb_a, emb_b)`** fits `mean(axis=0)` and
   `np.cov(emb, rowvar=False)`. Forgetting `rowvar=False` gives an
   `(n, n)` covariance of the samples, not of the features.
6. **`fad_infinity(emb_ref, emb_gen, sizes, seed)`** draws subsets with
   `rng.choice(..., replace=False)`, scores each, and returns the intercept of
   `np.polyfit(1 / N, scores, 1)`.
7. **`clap_score(text_emb, audio_emb)`** normalizes rows and averages the
   row-wise dot products.

## Cost and scaling

The following table compares what each method costs to train and to sample:

| Method | Training step | Sampling |
|---|---|---|
| Flow matching | 1 forward and backward pass | 10 to 32 NFE, doubled by guidance |
| Shortcut model | Adds 2 forward passes for a quarter of the batch | 1 to 8 NFE, any step size it was trained on |
| MeanFlow | A JVP, about the cost of 1 more forward and backward | 1 NFE, or a few |
| Distillation | A trained teacher, then a second training run | 1 to 8 NFE |

FAD costs one embedding pass per clip, then a $D \times D$ covariance and a
matrix square root, which takes time proportional to $D^3$: trivial for
VGGish's 128 dimensions and noticeable for embeddings of several thousand.
FAD-infinity multiplies the covariance work by the number of sizes.

## Where this breaks

- **The convention flips.** MeanFlow and Stable Audio put data at 0. Port
  their equations without converting and the model generates noise.
- **Few steps lose detail first.** One-step samples tend to lose fine texture
  and variety before they lose their overall shape. Listen, and compare FAD at
  equal sample sizes.
- **Self-consistency drifts.** A shortcut model's large steps learn from its
  own small steps, so an error at small $d$ spreads upward. Keep enough of
  each batch on plain flow matching.
- **Small evaluation sets flatter no one.** FAD from 100 clips is inflated
  and noisy. Use thousands, equal sizes, or FAD-infinity.
- **The metric is gamed.** A generator tuned against one embedding can lower
  FAD without sounding better. Report more than one metric, and listen.

## The exercise

You build `shortcut_target`, `shortcut_loss`, `meanflow_target`,
`frechet_distance`, `fad`, `fad_infinity`, and `clap_score`. The checks
confirm the two-step target and its missing gradient, the loss on flow rows
and on self-consistency rows, including its gradient against a hand-built
reference. They check that the exact average velocity of a Gaussian flow is a
fixed point of your MeanFlow target, which a sign error breaks. They test FAD
against its 1-D, diagonal, and commuting closed forms, its sample-size bias,
FAD-infinity, and the CLAP score. The last check trains a small shortcut model
and requires its one-step samples to land on the ring.

Press **Run** to train a plain flow, a shortcut model, and a MeanFlow model on
the eight-blob ring and hear twelve one-step samples from each as notes, in
tune only when they land on the data. The run prints FAD against the number of
steps for all three, and shows FAD's sample-size bias and its dependence on
the embedding.
