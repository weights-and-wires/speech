---
title: Flow matching
slug: 36-flow-matching
part: "Part 7 — Diffusion and flow matching"
summary: Learn a velocity field that carries noise to data along straight lines, solve its ODE in a few steps, and straighten it with reflow.
minutes: 70
gpu: false
objectives:
  - Build the straight conditional path from a noise sample to a data sample, its target velocity, and the flow-matching loss.
  - "Convert between the two time conventions: flow matching's noise at t = 0 and the diffusion lessons' data at t = 0."
  - Integrate a velocity field with Euler, midpoint, and RK4, and measure each solver's order of convergence.
  - Explain why straight training paths still give curved sampling paths, and straighten them with minibatch optimal transport and reflow.
prereqs:
  - 28-normalizing-flows-and-vits
  - 33-diffusion-from-scratch
  - 34-samplers-scores-and-guidance
exercise: 36-flow-matching
---

# Flow matching

Diffusion adds noise step by step and trains a network to take it away. It
works, but it carries machinery: a noise schedule, cumulative products, and a
choice of targets. Flow matching gets the same kind of generator with less.
You draw a straight line from a noise sample to a data sample, and you teach a
network how fast to move along it.

At sampling time, the network is a velocity field, like the wind map in a
weather report. Drop a noise sample into the wind, let it drift for one unit
of time, and it lands on something that looks like data. Matcha-TTS,
Voicebox, E2 TTS, F5-TTS, and the acoustic decoder of CosyVoice 2 all generate
speech this way, and the next lesson builds one.

One warning comes first, because it causes more bugs than anything else in
this part of the course. Flow-matching papers run the clock in the opposite
direction from the diffusion papers. In this lesson and the next two, time 0
is noise and time 1 is data.

## What you already know

- A normalizing flow maps noise to data with an invertible function, and a
  continuous flow does the same by solving an ordinary differential equation.
- A diffusion model's noisy sample is $\sqrt{\bar\alpha}$ times the data plus
  $\sqrt{1 - \bar\alpha}$ times Gaussian noise.
- The v-prediction target mixes noise and data as $a_t \epsilon - \sigma_t x_0$.
- The probability-flow ODE turns a diffusion model into a deterministic sampler,
  and DDIM is a first-order step along it.
- An optimizer can fit a network to any regression target you can compute.

## A straight path for every pair

> [!WARNING]
> In this lesson, $x_0$ is **noise** and $x_1$ is **data**, as in Lipman et
> al., Matcha-TTS, Voicebox, and F5-TTS. In the diffusion lessons, $x_0$ is
> data. The same symbol means opposite things, so restate the convention at the
> top of every file you write.

Pick one noise sample $x_0 \sim \mathcal{N}(0, I)$ and one data sample $x_1$,
such as one frame of a mel spectrogram. Walk from the first to the second in a
straight line at constant speed. The point you reach at time $t$ between 0 and
1 is the *conditional path*, and the speed you walk at is its *target
velocity*:

$$
x_t = \bigl(1 - (1 - \sigma_{\min})\,t\bigr)\,x_0 + t\,x_1, \qquad
u = x_1 - (1 - \sigma_{\min})\,x_0
$$

Read it aloud: at time $t$, keep a fraction of the noise that shrinks from 1
to $\sigma_{\min}$, and add a fraction $t$ of the data. The velocity is the
derivative of $x_t$ with respect to $t$, and it doesn't depend on $t$, because
the path is straight. The small constant $\sigma_{\min}$ leaves a trace of
noise at $t = 1$. Matcha-TTS uses $10^{-4}$. F5-TTS and CosyVoice 2 use 0,
which gives the rectified-flow target $u = x_1 - x_0$, "data minus noise." In
numpy, with one time per example:

```python
t = t.reshape(-1, *([1] * (x0.ndim - 1)))      # (B,) -> (B, 1, 1)
x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
u = x1 - (1 - sigma_min) * x0
```

### Work one path by hand

Take $\sigma_{\min} = 0$, noise $x_0 = -1$, data $x_1 = 2$, and $t = 0.25$.
Then $x_t = 0.75 \times (-1) + 0.25 \times 2 = -0.25$, and the target is
$u = 2 - (-1) = 3$. With $\sigma_{\min} = 10^{-4}$, the point moves to
$(1 - 0.9999 \times 0.25)(-1) + 0.5 = -0.250025$ and the target to
$2 + 0.9999 = 2.9999$.

## The loss, and the field it learns

Training is plain regression. Draw a batch of data, a batch of noise, and a
time $t \sim U[0, 1]$ for each example. Build $x_t$ and $u$, and ask the
network $v_\theta(x_t, t)$ to output $u$:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,x_1}\left\lVert v_\theta(x_t, t) - u \right\rVert^2
$$

Read it aloud: the loss is the average squared distance between the network's
velocity and the velocity of the straight path through $x_t$. Training solves
no ODE and needs no noise schedule. The exercise averages over every element:

```python
x_t, u = ot_path(x0, x1, t)
loss = ((model(x_t, t) - u) ** 2).mean()
```

Many straight paths pass through the same point $x_t$, from different noise
toward different data. The network can't know which one it's on, so the best
it can do is output their average. Lipman et al. prove that this average, the
*marginal velocity field*, carries the whole noise distribution onto the whole
data distribution. Each conditional path is straight, but the average of many
crossing paths bends.

> [!KEY]
> Flow matching trains on straight lines from each noise sample to each data
> sample. The network learns their average at each point, which is a curved
> field that still moves all of the noise onto all of the data.

### A field you can write down

When the data is one-dimensional and Gaussian, $\mathcal{N}(\mu, s^2)$, the
average has a closed form. The point $x_t$ is then Gaussian with mean $t\mu$
and variance $(1 - t)^2 + t^2 s^2$, and the marginal field is:

$$
v(x, t) = \mu + \frac{t s^2 - (1 - t)}{(1 - t)^2 + t^2 s^2}\,(x - t\mu)
$$

Read it aloud: the field pushes every point toward the data mean, plus a
correction proportional to the point's distance from the moving mean $t\mu$.
Solving this ODE exactly sends $x_0$ to $\mu + s\,x_0$, so the exercise uses
it as an oracle whose right answer you know.

At $t = 0$ the fraction is $-1$, so $v(x_0, 0) = \mu - x_0$. One Euler step of
size 1 lands on $x_0 + (\mu - x_0) = \mu$, the data mean, whatever $x_0$ was.
That is why one-step sampling from an undistilled flow is blurry: every sample
becomes the average.

Take $\mu = 2$, $s = 0.5$, and $x_0 = 1.3$, so the exact answer is
$2 + 0.5 \times 1.3 = 2.65$. Two Euler steps of size 0.5 go as follows:

1. At $t = 0$: $v = 2 - 1.3 = 0.7$, so $x = 1.3 + 0.5 \times 0.7 = 1.65$.
2. At $t = 0.5$: the variance is $0.25 + 0.0625 = 0.3125$, the fraction is
   $(0.125 - 0.5) / 0.3125 = -1.2$, and $x - t\mu = 0.65$. So
   $v = 2 - 1.2 \times 0.65 = 1.22$, and $x = 1.65 + 0.5 \times 1.22 = 2.26$.

## The same state in the diffusion convention

Every Gaussian path, the straight one included, is a diffusion process with
some schedule. So you can move any flow-matching state into the language of
lesson 33, where data sits at time 0 and a noisy sample is
$\sqrt{\bar\alpha}\,x_{\text{data}} + \sqrt{1 - \bar\alpha}\,\epsilon$. The
following table lists how the two conventions write the same objects:

| Object | Flow matching (this lesson) | DDPM (lessons 33 to 35) |
|---|---|---|
| Noise | $x_0$, at $t = 0$ | $\epsilon$, or $x_T$ at the last step |
| Data | $x_1$, at $t = 1$ | $x_0$, at step 0 |
| Noisy point | $t\,x_1 + (1 - t)\,x_0$ | $\sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,\epsilon$ |
| Signal-to-noise ratio | $t^2 / (1 - t)^2$ | $\bar\alpha / (1 - \bar\alpha)$ |
| Network target | $u = x_1 - x_0$, toward data | $v = a_t\epsilon - \sigma_t x_0$, toward noise |
| Sampling direction | $t$ from 0 up to 1 | step from $T$ down to 0 |

Two points match when their signal-to-noise ratios match. Setting
$t^2 / (1 - t)^2 = \bar\alpha / (1 - \bar\alpha)$ and writing
$r = \sqrt{t^2 + (1 - t)^2}$ gives the exact mapping:

$$
\bar\alpha = \frac{t^2}{t^2 + (1 - t)^2}, \qquad
t = \frac{\sqrt{\bar\alpha}}{\sqrt{\bar\alpha} + \sqrt{1 - \bar\alpha}}, \qquad
x_{\text{vp}} = \frac{x_t}{r}
$$

Read it aloud: the matching flow time is the diffusion step's signal weight
divided by the sum of both weights. The flow point is the diffusion point
shrunk by $r$, because the squares of $t$ and $1 - t$ don't add to one.

To convert the targets, recover the data and the noise from the flow state
first: $x_{\text{data}} = x_t + (1 - t)\,u$ and $\epsilon = x_t - t\,u$. Then
combine them into $v = \sqrt{\bar\alpha}\,\epsilon - \sqrt{1 - \bar\alpha}\,x_{\text{data}}$.
At $t = 0.5$ the two weights are equal, and $v = -u / \sqrt{2}$: the same
arrow, pointing the other way and shorter.

Take the path from before: data 2, noise $-1$, $t = 0.25$, so $x_t = -0.25$
and $u = 3$. Then $r = \sqrt{0.0625 + 0.5625} = 0.790569$ and
$\bar\alpha = 0.0625 / 0.625 = 0.1$. The diffusion point is
$-0.25 / 0.790569 = -0.316228$, which equals
$\sqrt{0.1} \times 2 + \sqrt{0.9} \times (-1)$. The v target is
$0.316228 \times (-1) - 0.948683 \times 2 = -2.213594$.

The mapping also relates the losses. Because $x_{\text{data}} = x_t + (1 - t)u$,
an error $\delta$ in the velocity is an error $(1 - t)\delta$ in the predicted
data. So the flow-matching loss is a data-prediction loss weighted by
$1 / (1 - t)^2$, and a noise-prediction loss weighted by $1 / t^2$. The
parameterizations differ only in how much each noise level counts. The same
algebra shows that one DDIM step on the straight path, from $t$ to $t'$, gives
$t'\hat{x}_{\text{data}} + (1 - t')\hat\epsilon$, which is exactly one Euler
step. DDIM is Euler in disguise.

> [!KEY]
> A flow time $t$ and a diffusion step with the same signal-to-noise ratio
> describe the same noisy sample up to a scale. Convert through the data and
> noise estimates, never by swapping symbols.

## Solving the ODE

Sampling integrates $dx/dt = v_\theta(x, t)$ from a noise sample at $t = 0$ to
$t = 1$. Each call to the network is one *function evaluation*, and the count
of them, the NFE, is the price of a sample. With $n$ equal steps of size
$h = 1/n$, the three solvers the exercise builds are the following:

$$
\begin{aligned}
\text{Euler:}\quad & x \leftarrow x + h\,v(x, t) \\
\text{Midpoint:}\quad & x \leftarrow x + h\,v\!\left(x + \tfrac{h}{2} v(x, t),\ t + \tfrac{h}{2}\right) \\
\text{RK4:}\quad & x \leftarrow x + \tfrac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
$$

Read it aloud: Euler moves along the velocity at the start of the step.
Midpoint takes a trial half step and uses the velocity there for the whole
step. RK4 averages four slopes, at the start, at two trial midpoints, and at a
trial end, weighted 1, 2, 2, and 1. They cost 1, 2, and 4 NFE per step.

A solver's *order* $p$ says how its error shrinks: halving $h$ divides the
error by about $2^p$. Euler has order 1, midpoint 2, and RK4 4. The exercise
measures the orders on $dx/dt = t\,x$, whose exact answer at $t = 1$ is
$x_0 e^{1/2}$. On the Gaussian flow, midpoint does better than its guaranteed
order, with its error falling by about 8 per halving, which is a property of
that field, not of midpoint.

The following table compares the solvers on the Gaussian flow at equal cost,
where the exact answer is 2.65:

| NFE | Euler | Midpoint | RK4 |
|---|---|---|---|
| 1 | 2.0000 | not possible | not possible |
| 2 | 2.2600 | 2.5200 | not possible |
| 4 | 2.4427 | 2.6427 | 2.4940 |
| 8 | 2.5392 | 2.6478 | 2.6458 |
| 32 | 2.6206 | 2.6500 | 2.6500 |

At tiny budgets a higher order doesn't pay: one RK4 step is worse than two
midpoint steps. Real systems pick cheap, low-order solvers. Matcha-TTS uses
Euler with 2 to 10 steps, F5-TTS Euler with 16 or 32, CosyVoice 2 Euler with
10, and Voicebox and E2 TTS midpoint with NFE 32.

The following demo draws the straight training lines and the paths a sampler
actually follows.

```demo
{"id": "flow-trajectories", "caption": "Turn on the training pairs to see the straight conditional paths, set the steps to 1 and watch every Euler sample land near the data mean although the exact paths bend, then turn on minibatch OT and watch the path bend readout fall toward 1."}
```

## Straighter paths: minibatch OT and reflow

A curved field needs many small steps. Two tricks make the field straighter,
so that a few large steps land on the data.

*Minibatch optimal transport* (Tong et al., 2024) changes which noise sample each
data sample is paired with. Within each batch, it finds the permutation that
minimizes the total squared distance, with the Hungarian algorithm in
`scipy.optimize.linear_sum_assignment`. Paired this way, paths rarely cross,
so the average at each point is close to a single straight path. For example,
take noise $\{-1, 1\}$ and data $\{1.5, -2\}$. Pairing in order costs
$2.5^2 + 3^2 = 15.25$, and the two paths cross. The swapped pairing costs
$1^2 + 0.5^2 = 1.25$, and neither crosses the other.

> [!TIP]
> Matcha-TTS and CosyVoice call their loss "OT-CFM," but they mean the
> straight OT *path* with random pairing. Tong's minibatch OT *coupling* is a
> separate step that reorders the pairs.

*Reflow* (Liu et al., 2022) uses the trained model itself. Draw noise $z_0$,
solve the ODE carefully to get $z_1$, and train a second flow on the fixed
pairs $(z_0, z_1)$. The first flow's ODE paths never cross, so straight lines
between its pairs cross far less often than lines between random pairs, and
the second flow comes out much straighter. To measure it, compare
each Euler step's velocity with the chord $z_1 - z_0$:

$$
S = \frac{1}{n}\sum_{k=0}^{n-1} \mathbb{E}\left\lVert (z_1 - z_0) - v(z_k, t_k) \right\rVert^2
$$

Read it aloud: straightness is the average squared gap between the velocity
along the way and the one velocity that makes the trip in a straight line. It
is zero exactly when every trajectory is straight. On the eight-blob ring in
the **Run** block, one reflow cut $S$ from 1.51 to 0.009, and the mean radius
of one-step samples went from 0.23 to 1.96, against 2.00 for the data.

> [!RECAP]
> - The path is $(1 - (1 - \sigma_{\min})t)\,x_0 + t\,x_1$, noise at 0 and
>   data at 1, and the target is $x_1 - (1 - \sigma_{\min})\,x_0$.
> - The loss is the mean squared error between $v_\theta(x_t, t)$ and that
>   target, and the network learns the curved average of all such paths.
> - Matching the signal-to-noise ratio converts a flow time into a DDPM
>   $\bar\alpha$: $\bar\alpha = t^2 / (t^2 + (1 - t)^2)$.
> - Euler, midpoint, and RK4 have orders 1, 2, and 4 and cost 1, 2, and 4
>   NFE per step.
> - Minibatch OT pairing and reflow straighten the field so that few steps
>   suffice.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ot_path(x0, x1, t, sigma_min)`** returns `(x_t, target)`, both with the
   shape of `x0`. Reshape a `(B,)` time to `(B, 1, ..., 1)` before
   multiplying. The common mistake is multiplying a `(B,)` time into a
   `(B, C, T)` tensor, which fails, or worse, broadcasts over the last axis
   when `T == B`.
2. **`flow_matching_loss(model, x0, x1, t, sigma_min)`** calls `ot_path`,
   then `model(x_t, t)`, and returns the mean of the squared difference. Use
   the same `sigma_min` in the path and the target.
3. **`gaussian_velocity(x, t, mu, s)`** is the closed-form field, one line of
   arithmetic. Check it by hand: at `t = 0` it returns `mu - x`.
4. **`integrate(velocity, x0, n_steps, method)`** loops over `k` with
   `t = k / n_steps` and applies the Euler, midpoint, or RK4 update. Compute
   each time from `k`, not by adding `h` repeatedly, so the last step ends on
   1.
5. **`to_diffusion(x_t, t, u)`** returns `(alpha_bar, x_vp, v)`. Recover the
   data and the noise, then combine them. The mistake to avoid is writing
   $v$ with the flow's weights $t$ and $1 - t$ instead of
   $\sqrt{\bar\alpha}$ and $\sqrt{1 - \bar\alpha}$.
6. **`minibatch_ot_pairing(x0, x1)`** flattens each sample, builds the
   `(n, n)` matrix of squared distances by broadcasting, and returns the
   column indices from `linear_sum_assignment`.
7. **`straightness(velocity, x0, n_steps)`** runs Euler, keeps every
   velocity, and averages the squared norm of `chord - v` over steps and
   samples. Sum over the feature axis before averaging over samples.

## Cost and scaling

Training costs one forward and one backward pass per batch, whatever sampler
you use later. Sampling costs the NFE times one forward pass. The following
table lists what each piece costs for a batch of $n$ samples of dimension
$D$:

| Piece | Time | Notes |
|---|---|---|
| One training step | one network pass, forward and backward | No ODE is solved |
| Euler, midpoint, RK4 sample | 1, 2, or 4 passes per step | Classifier-free guidance doubles each pass |
| Minibatch OT pairing | $O(n^2 D)$ for distances, $O(n^3)$ to assign | Negligible for $n = 256$, heavy past a few thousand |
| One reflow round | one full ODE solve per training pair, then a second training run | Often a large set of pairs |

## Where this breaks

- **The clock runs backward.** A model sampled in the opposite convention
  from its training integrates the wrong way and outputs noise. The SD3
  family and Stable Audio Open Small put data at 0 and integrate from 1 down
  to 0. Name the convention in every file.
- **One step returns the mean.** An undistilled flow sampled in one step
  gives the blurred average. Use several steps, reflow, or the few-step
  methods of lesson 38.
- **The endpoint divides by zero.** Converting a velocity to a noise estimate
  divides by $t$, and converting it to a score divides by $1 - t$. Keep $t$
  away from the ends, or never convert there.
- **OT pairing fades in high dimension.** For a batch of 256 mel
  spectrograms, the best pairing is barely better than a random one. Larger
  batches help, at a cubic cost.
- **Reflow compounds errors.** The second model learns the first model's
  mistakes along with its pairs, so people stop after one or two rounds.

## The exercise

You build `ot_path`, `flow_matching_loss`, `gaussian_velocity`, `integrate`,
`to_diffusion`, `minibatch_ot_pairing`, and `straightness`. The checks confirm
the worked values, broadcasting, the Gaussian field against a
finite-difference derivative of its flow map, the solver table and orders, and
`to_diffusion` against the DDPM formulas. They compare your pairing with all
5,040 pairings of seven points. The last check trains a small flow with your
loss, reflows it once, and requires straightness to fall by more than 70
percent.

Press **Run** to race the solvers, train and reflow a flow on an eight-blob
ring, and hear its samples as notes: each sample's angle picks a pitch on the
C major scale, and its radius sets the loudness. One Euler step plays quiet,
out-of-tune notes, and eight steps, or one step after reflow, play the scale.
