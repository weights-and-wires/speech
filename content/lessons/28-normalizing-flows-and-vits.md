---
title: Normalizing flows and VITS
slug: 28-normalizing-flows-and-vits
part: "Part 5 — Speech synthesis"
summary: Build invertible layers with exact log-determinants, train a flow by maximum likelihood, and write the KL term that trains VITS.
minutes: 75
gpu: false
objectives:
  - Apply the change-of-variables formula, and say which direction's log-determinant it adds.
  - Build affine coupling, ActNorm, and an invertible 1x1 convolution, and verify each log-determinant against an autograd Jacobian.
  - Train a RealNVP on 2-D data by maximum likelihood, and sample from it by running the layers backward.
  - Explain how VITS joins a VAE, a flow prior, monotonic alignment search, and a HiFi-GAN decoder, and derive its single-sample KL term.
prereqs:
  - 25-durations-and-alignment-search
  - 27-gan-vocoders
exercise: 28-normalizing-flows-and-vits
---

# Normalizing flows and VITS

A GAN vocoder generates fast but gives you no likelihood: it can't say how
probable a clip is, so you can't train it by maximum likelihood. WaveNet gives
an exact likelihood but generates one sample at a time. A *normalizing flow*
gets both. It is an invertible network that maps data to Gaussian noise. Run
it forward and you get an exact log-likelihood to train on. Run it backward
and you turn fresh noise into data, all positions at once.

Flows carried speech synthesis for several years. WaveGlow was a flow
vocoder. Glow-TTS used a flow decoder and found its own alignment with the
search from lesson 25. VITS put a flow inside a variational autoencoder,
added a HiFi-GAN decoder, and became a widely used end-to-end TTS model,
trained from text and audio in one piece.

In this lesson you build the three invertible layers flows are made of, check
each log-determinant against brute force, train a small flow on 2-D data,
and write the KL term that VITS trains with.

## What you already know

- Glow-TTS aligns frames to phones by monotonic alignment search over Gaussian log-likelihoods (lesson 25, *Durations and alignment search*).
- A HiFi-GAN generator turns latent frames into a waveform, trained with mel, adversarial, and feature-matching losses (lesson 27, *GAN vocoders*).
- The Jacobian of a map is the matrix of its partial derivatives, and a triangular matrix's determinant is the product of its diagonal.
- A diagonal Gaussian's log-density is $\sum_d \left[-\tfrac12 \log 2\pi - \log\sigma_d - (z_d - \mu_d)^2 / 2\sigma_d^2\right]$.
- A variational autoencoder trains an encoder $q(z \mid x)$ and a decoder with a reconstruction loss plus a KL divergence to a prior.

## The change of variables

A flow is an invertible map $f$ from data $x$ to a latent $z = f(x)$ with a
simple density, a standard Gaussian. Where $f$ stretches space, density
thins; where it squeezes, density piles up. The change-of-variables formula
does that accounting exactly:

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|.
$$

Read it aloud: the log-density of a data point is the log-density of its
latent, plus the log of how much $f$ expands volume near that point. The
determinant of the Jacobian measures that expansion. The Jacobian belongs to
the data-to-latent direction, and the term is added.

Take one dimension and $z = 2x$. At $x = 1$, $z = 2$ and
$\log p_Z(2) = -\tfrac12 \log 2\pi - 2 = -2.9189$. The map doubles lengths, so
$\log |dz/dx| = \ln 2 = 0.6931$, and $\log p_X(1) = -2.9189 + 0.6931 =
-2.2258$. That is the log-density of $\mathcal{N}(0, 1/4)$ at 1, as it must
be: $x = z / 2$ has standard deviation $1/2$.

A flow stacks $K$ invertible layers, $z = f_K(\cdots f_1(x))$, and their
log-determinants add. Sampling draws $z \sim \mathcal{N}(0, I)$ and runs the
inverses in reverse order. The whole design problem is to build layers that
are expressive, invertible, and have a log-determinant that is cheap to
compute. A general $D \times D$ Jacobian costs $O(D^3)$ to take a
determinant of, which is hopeless for a mel spectrogram.

## Affine coupling

The *affine coupling* layer of RealNVP and Glow solves that problem with a
trick. Split the input into two halves, $x_a$ and $x_b$. Leave $x_a$ alone,
and use it to scale and shift $x_b$:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp\left(s(x_a)\right) + t(x_a), \qquad
\log \left| \det \frac{\partial y}{\partial x} \right| = \sum_i s(x_a)_i.
$$

Read it aloud: the first half passes through. The second half is multiplied,
entry by entry, by the exponential of a scale $s$, and shifted by $t$. Both
$s$ and $t$ are outputs of any network you like, fed only the first half.
The Jacobian is triangular: $y_a$ doesn't depend on $x_b$, and each entry of
$y_b$ depends on its own $x_b$ entry through the factor $e^{s_i}$. So the
determinant is the product of those factors, and its log is the sum of $s$.
In torch:

```python
x_a, x_b = x[:, :half], x[:, half:]
s, t = net(x_a)
y = torch.cat([x_a, x_b * torch.exp(s) + t], dim=1)
logdet = s.sum(dim=1)
```

The inverse is as cheap as the forward pass. Since $y_a = x_a$, you can
recompute $s$ and $t$ from $y_a$ and undo the step:
$x_b = (y_b - t(y_a)) \odot \exp(-s(y_a))$. The network $s$ itself never has
to be inverted. Alternate which half is transformed from layer to layer, or
the first half never changes.

### Work one coupling by hand

Take $x = (1, 2)$, $s(x_a) = 0.5\,x_a$, and $t(x_a) = x_a$:

- $y_a = 1$, and $y_b = 2 e^{0.5} + 1 = 4.297443$.
- $\log |\det J| = s(1) = 0.5$.
- $\log \mathcal{N}(y; 0, I) = -\log 2\pi - (1^2 + 4.297443^2)/2 = -1.837877 - 9.734006 = -11.571883$.
- $\log p_X(x) = -11.571883 + 0.5 = -11.071883$.
- The inverse gives back $x_b = (4.297443 - 1)\,e^{-0.5} = 2$.

> [!KEY]
> A coupling layer's Jacobian is triangular, so its log-determinant is the sum
> of the predicted log-scales, and its inverse reuses the same network on the
> unchanged half.

## ActNorm and the invertible 1x1 convolution

Glow adds two more layers between couplings, both acting on a sequence
$x \in \mathbb{R}^{C \times T}$ of $C$ channels and $T$ time steps.

**ActNorm** is a per-channel scale and shift,
$y_{ct} = e^{\ell_c} x_{ct} + b_c$. Its scale and bias are initialized from
the first batch so that every channel's output has zero mean and unit
variance, then trained freely. Every one of the $T$ time steps gets the same
scale, so the log-determinant is $T \sum_c \ell_c$.

**The invertible 1x1 convolution** mixes channels with one learned
$C \times C$ matrix, $y_t = W x_t$ at every time step. It generalizes the fixed
swap of halves between couplings. Its log-determinant is $T \log |\det W|$:

$$
\log \left| \det J_{\text{ActNorm}} \right| = T \sum_{c=1}^{C} \ell_c, \qquad
\log \left| \det J_{1 \times 1} \right| = T \log \left| \det W \right|.
$$

Read it aloud: both layers apply the same small map at each of $T$ time
steps, so the whole Jacobian is block diagonal with $T$ identical blocks, and
the log-determinant is $T$ times one block's. For example, with $T = 4$ and
log-scales $[0.3, -0.2, 0.5]$, ActNorm's log-determinant is
$4 \times 0.6 = 2.4$. A $W$ with $|\det W| = 2.1$ gives
$4 \ln 2.1 = 2.968$. In torch, `torch.linalg.slogdet(W).logabsdet` gives
$\log|\det W|$ without overflow.

> [!WARNING]
> Forgetting the factor $T$ is a common flow bug. The loss still falls,
> but the likelihood is wrong, and the model learns to shrink or inflate
> channels for free.

## Training by maximum likelihood

With exact log-densities, training is ordinary maximum likelihood: minimize
the mean of $-\log p_X(x)$ over the data. No discriminator, no
bound. Report the loss in nats per dimension, so clips of different lengths
compare. Sampling runs the flow backward from Gaussian noise.

For audio, WaveGlow did exactly that on raw waveforms. It stacked 12
couplings and 12 invertible 1x1 convolutions, with WaveNet-like networks
predicting $s$ and $t$ from the mel spectrogram and half the samples. It
trained with a Gaussian of standard deviation $\sqrt{0.5}$ and sampled with
0.6, a lower *temperature* that trades variety for cleaner audio. It needed
about 88 million parameters, and in the HiFi-GAN paper's listening test a
GAN vocoder with under 1 million parameters beat it. GAN vocoders replaced it. Flows survived inside TTS models, where they model a latent space instead of raw audio.

## VITS

VITS, from 2021, is a *conditional variational autoencoder* for speech, with
four parts:

- **The posterior encoder** reads the real linear spectrogram of a clip and outputs a Gaussian $q(z \mid x)$ for each frame, a mean $m_q$ and log standard deviation $\log\sigma_q$. It exists only in training.
- **The prior** comes from text. A Transformer text encoder gives each phone a Gaussian $(m_p, \log\sigma_p)$. Monotonic alignment search (lesson 25) expands the phones to frames. A flow $f$ of coupling layers maps the posterior's $z$ to $z_p = f(z)$, where it is scored against that Gaussian. The flow lets a simple per-phone Gaussian describe a complicated distribution of frames.
- **The decoder** is a HiFi-GAN V1 generator from lesson 27. It turns $z$ into a waveform, trained on random windows of frames to save memory.
- **A stochastic duration predictor**, itself a flow, learns a distribution of durations so rhythm can vary.

At inference there is no audio. VITS samples $z_p$ from the text prior, runs
the flow backward to get $z$, and decodes it, all in one parallel pass. The
paper reported a MOS of 4.43 on LJSpeech against 4.46 for recordings. Its
training loss is the sum of a mel L1 term, the KL term, the duration loss,
and the HiFi-GAN adversarial and feature-matching losses. The official
configuration weights the mel term by 45 and the KL term by 1.

### The KL term

The KL term pulls the posterior toward the prior. With one sample
$z = m_q + \sigma_q \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$, it is
$\log q(z \mid x) - \log p(z \mid \text{text})$. The VITS flow is additive
($s = 0$), so its log-determinant is 0. Write out both Gaussians, cancel the
$\log 2\pi$ terms, and replace $\epsilon^2$ by its expected value 1:

$$
\mathcal{L}_{\text{kl}} = \frac{1}{|\mathcal{M}|} \sum_{(c,t) \in \mathcal{M}}
\left[ \log\sigma_p - \log\sigma_q - \frac12 + \frac{(z_p - m_p)^2}{2\sigma_p^2} \right].
$$

Read it aloud: at every real frame and channel, the term is the prior's log
standard deviation minus the posterior's, minus one half, plus the squared
distance of the flowed sample from the prior mean in units of the prior's
variance. The $-\tfrac12$ is $\log q$'s $-\epsilon^2/2$, averaged. The sum
runs over the mask $\mathcal{M}$ of real entries and divides by its size.
For one entry with $z_p = 0.5$, $\log\sigma_q = -1$, $m_p = 0$, and
$\log\sigma_p = 0$, the term is $0 + 1 - 0.5 + 0.125 = 0.625$. In torch:

```python
kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
loss = (kl * mask).sum() / mask.sum()
```

Averaged over many samples of $z$, with an identity flow, this equals the
closed-form KL divergence between two Gaussians. The exercise checks that.

> [!KEY]
> VITS's KL term compares the posterior sample, pushed through the flow, with
> the text prior's Gaussian. The flow is what lets a per-phone Gaussian explain
> frames that are far from Gaussian.

> [!RECAP]
> - The change of variables adds the log-determinant of the data-to-latent map.
> - A coupling layer's log-determinant is $\sum s(x_a)$, and its inverse is $x_b = (y_b - t)\,e^{-s}$.
> - ActNorm's log-determinant is $T \sum_c \ell_c$, and a 1x1 convolution's is $T \log|\det W|$.
> - Train by minimizing $-\log p_X$; sample by inverting the layers in reverse order.
> - The VITS KL term is $\log\sigma_p - \log\sigma_q - \tfrac12 + (z_p - m_p)^2 / 2\sigma_p^2$, averaged over real entries.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`coupling_forward(x, net, flip)`** splits `x` into halves, swaps the
   roles when `flip` is true, and returns `(y, logdet)`. The common mistake
   is concatenating the halves in the wrong order when `flip` is true.
2. **`coupling_inverse(y, net, flip)`** recomputes `(s, t)` from the
   unchanged half and undoes the step.
3. **`actnorm_init(x)`** and **`actnorm_forward(x, log_scale, bias)`**:
   take the mean and biased standard deviation over dims `(0, 2)`, and
   return a log-determinant of shape `(B,)`. The common mistake is omitting
   the factor $T$.
4. **`invconv_forward(x, W)`** and **`invconv_inverse(y, W)`** use
   `torch.einsum("ij,bjt->bit", ...)` and `torch.linalg.slogdet`.
5. **`RealNVP.forward`**, **`RealNVP.log_prob`**, and **`RealNVP.sample`**:
   loop over `self.nets` with `flip = i % 2 == 1` and sum the
   log-determinants. `sample` draws `torch.randn` with a seeded
   `torch.Generator` and runs the inverses in reverse. The common mistake is
   running the inverses in forward order.
6. **`train_flow(data, steps, lr, batch, seed)`** runs Adam on the mean of
   `-log_prob` over random batches.
7. **`vits_kl(z_p, logs_q, m_p, logs_p, mask)`** is two lines. The common
   mistake is dividing by the number of frames instead of the number of real
   entries.

## Cost and scaling

The following table compares the layers for a sequence of $C$ channels and
$T$ steps, with coupling networks of cost $N$ per step.

| Layer | Forward and inverse | Log-determinant | Parameters |
|---|---|---|---|
| Affine coupling | $O(TN)$ each | $O(TC)$, a sum | the network's |
| ActNorm | $O(TC)$ | $O(C)$ | $2C$ |
| 1x1 convolution | $O(TC^2)$ | $O(C^3)$ once, or $O(C)$ with Glow's LU form | $C^2$ |
| General invertible layer | $O(TC^2)$ or more | $O((TC)^3)$ | not practical |

A flow keeps the data's full dimension at every layer, since it can't throw
information away. WaveGlow needed about 88 million parameters and a large
GPU for that reason. Sampling is parallel, so a flow is fast at inference,
but every layer runs at full resolution.

## Where this breaks

**Direction mix-ups.** Papers write the flow in both directions. The
log-determinant you add is always that of the data-to-latent map. Adding the
other one's, or subtracting, gives a model that trains to nonsense.

**Mass between modes.** A flow is a smooth bijection, so it can't tear
Gaussian noise into separate islands. It thins the bridges between modes but
leaves some mass there. The Run shows it: some sampled notes land between the
notes of the scale.

**Discrete data.** Audio samples and durations are integers. A continuous
density can pile up on a single point, so the likelihood grows without bound.
Add uniform noise first, *dequantization*, as the VITS duration predictor does.

**Unstable scales.** An unbounded $s$ can reach $e^{50}$ in a few steps.
Bound it with a tanh, as the exercise's networks do, or start couplings at
the identity by initializing the last layer to zero.

## The exercise

You implement `coupling_forward`, `coupling_inverse`, `actnorm_init`,
`actnorm_forward`, `invconv_forward`, `invconv_inverse`, the `RealNVP`
methods `forward`, `log_prob`, and `sample`, `train_flow`, and `vits_kl`.
The checks reproduce the hand-worked $-11.071883$, compare every
log-determinant with `slogdet` of an autograd Jacobian, confirm each inverse,
and integrate your density over a grid to 1. They train your flow on two
moons and require it to beat the best single Gaussian by 0.4 nats, and
compare your KL term, averaged over 200,000 posterior samples, with the
closed-form Gaussian KL.

Press **Run** to train an 8-layer RealNVP for about 10 seconds on 4,000
notes of a pentatonic tune, each a pitch and a log length. The run prints the
held-out likelihood against a single Gaussian, draws the learned density,
and plays a 16-note melody sampled from your flow.
