---
title: Straighter paths and better solvers
slug: 057-straighter-paths-and-better-solvers
part: "Part 7 — Diffusion and flow matching"
summary: Cut the number of network calls a flow needs, by using a higher-order solver, by pairing noise with nearby data, and by retraining on the flow's own samples.
minutes: 50
gpu: false
objectives:
  - Integrate a velocity field with Euler, midpoint, and RK4, and measure each solver's order of convergence.
  - Explain why a straight trajectory needs one step and a curved one needs many.
  - Pair noise with data by minibatch optimal transport and explain why it reduces crossings.
  - Straighten a flow with one round of reflow, and measure straightness.
prereqs:
  - 053-fast-samplers
  - 056-flow-matching
exercise: 057-straighter-paths-and-better-solvers
---

# Straighter paths and better solvers

A flow-matching model gives you good samples at 32 steps and a blur at one. Every step is a call to the network, so on a phone or in a live voice agent you want far fewer. There are two ways to get there. You can walk the curved field more cleverly, with a better solver, or you can change the field so that it is nearly straight and needs no cleverness. This chapter does both, and shows why the second one is the bigger win.

As in [the flow-matching chapter](/lessons/056-flow-matching), time 0 is noise and time 1 is data, $x_0$ is noise, and $x_1$ is data.

## What you already know

- A flow's velocity field $v(x, t)$ carries noise at $t = 0$ to data at $t = 1$, and Euler integrates it with steps $x \leftarrow x + h\,v(x, t)$.
- The network learns the average of all straight paths through each point, a curved field, so one Euler step lands on the data mean.
- On the Gaussian flow $\mathcal{N}(2, 0.5^2)$ with $x_0 = 1.3$, the exact endpoint is 2.65 and Euler with 2 steps ends at 2.26.
- Higher-order solvers such as DDIM's successors take fewer steps on a smooth ODE, as in [the samplers chapter](/lessons/053-fast-samplers).

## Better solvers

Euler moves along the velocity at the start of a step, so on a curved path it overshoots to the outside of the bend. The fix is to peek at where you are heading. With $n$ equal steps of size $h = 1/n$, the three solvers the exercise builds are the following:

$$
\begin{aligned}
\text{Euler:}\quad & x \leftarrow x + h\,v(x, t) \\
\text{Midpoint:}\quad & x \leftarrow x + h\,v\!\left(x + \tfrac{h}{2} v(x, t),\ t + \tfrac{h}{2}\right) \\
\text{RK4:}\quad & x \leftarrow x + \tfrac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)
\end{aligned}
$$

Read it aloud: Euler moves along the velocity at the start of the step. Midpoint takes a trial half step and uses the velocity there for the whole step. RK4 averages four slopes, at the start, at two trial midpoints, and at a trial end, weighted 1, 2, 2, and 1. They cost 1, 2, and 4 network evaluations (NFE) per step.

Work midpoint by hand on the Gaussian flow from the last chapter ($\mu = 2$, $s = 0.5$, $x_0 = 1.3$) with one step. The velocity at $t = 0$ is 0.7, so the trial half step reaches $1.3 + 0.5 \times 0.7 = 1.65$. The velocity there at $t = 0.5$ is 1.22, so the full step is $1.3 + 1 \times 1.22 = 2.52$. One midpoint step costs two evaluations and lands at 2.52, closer to 2.65 than two Euler steps at 2.26.

A solver's *order* $p$ says how its error shrinks: halving $h$ divides the error by about $2^p$. Euler has order 1, midpoint 2, and RK4 4. The exercise measures them on $dx/dt = t\,x$, whose exact answer at $t = 1$ is $x_0 e^{1/2}$. The following table compares the solvers at equal cost on the Gaussian flow, where the exact answer is 2.65, and Figure 1 plots the errors.

| NFE | Euler | Midpoint | RK4 |
|---|---|---|---|
| 1 | 2.0000 | not possible | not possible |
| 2 | 2.2600 | 2.5200 | not possible |
| 4 | 2.4427 | 2.6427 | 2.4940 |
| 8 | 2.5392 | 2.6478 | 2.6458 |
| 32 | 2.6206 | 2.6500 | 2.6500 |

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Error of Euler, midpoint, and RK4 against the number of network evaluations on the Gaussian flow, falling with order 1, 2, and 4 per step halving">
<defs><marker id="c053-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="70.0" y1="240.0" x2="500.0" y2="240.0" class="stroke-ink"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-ink"/>
<line x1="70.0" y1="30.0" x2="70.0" y2="240.0" class="stroke-grid"/>
<text x="70.0" y="256.0" class="fig-small" text-anchor="middle">1</text>
<line x1="156.0" y1="30.0" x2="156.0" y2="240.0" class="stroke-grid"/>
<text x="156.0" y="256.0" class="fig-small" text-anchor="middle">2</text>
<line x1="242.0" y1="30.0" x2="242.0" y2="240.0" class="stroke-grid"/>
<text x="242.0" y="256.0" class="fig-small" text-anchor="middle">4</text>
<line x1="328.0" y1="30.0" x2="328.0" y2="240.0" class="stroke-grid"/>
<text x="328.0" y="256.0" class="fig-small" text-anchor="middle">8</text>
<line x1="414.0" y1="30.0" x2="414.0" y2="240.0" class="stroke-grid"/>
<text x="414.0" y="256.0" class="fig-small" text-anchor="middle">16</text>
<line x1="500.0" y1="30.0" x2="500.0" y2="240.0" class="stroke-grid"/>
<text x="500.0" y="256.0" class="fig-small" text-anchor="middle">32</text>
<line x1="70.0" y1="240.0" x2="500.0" y2="240.0" class="stroke-grid"/>
<text x="64.0" y="244.0" class="fig-small" text-anchor="end">1e-7</text>
<line x1="70.0" y1="180.0" x2="500.0" y2="180.0" class="stroke-grid"/>
<text x="64.0" y="184.0" class="fig-small" text-anchor="end">1e-5</text>
<line x1="70.0" y1="120.0" x2="500.0" y2="120.0" class="stroke-grid"/>
<text x="64.0" y="124.0" class="fig-small" text-anchor="end">1e-3</text>
<line x1="70.0" y1="60.0" x2="500.0" y2="60.0" class="stroke-grid"/>
<text x="64.0" y="64.0" class="fig-small" text-anchor="end">1e-1</text>
<line x1="70.0" y1="30.0" x2="500.0" y2="30.0" class="stroke-grid"/>
<text x="64.0" y="34.0" class="fig-small" text-anchor="end">1</text>
<path d="M70.0 35.6 L156.0 42.3 L242.0 50.5 L328.0 58.7 L414.0 67.2 L500.0 75.9" class="stroke-a" fill="none"/>
<circle cx="70.0" cy="35.6" r="4" class="fill-a"/>
<circle cx="156.0" cy="42.3" r="4" class="fill-a"/>
<circle cx="242.0" cy="50.5" r="4" class="fill-a"/>
<circle cx="328.0" cy="58.7" r="4" class="fill-a"/>
<circle cx="414.0" cy="67.2" r="4" class="fill-a"/>
<circle cx="500.0" cy="75.9" r="4" class="fill-a"/>
<text x="506.0" y="79.9" class="fig-small" text-anchor="start">Euler</text>
<path d="M156.0 56.6 L242.0 94.1 L328.0 109.6 L414.0 136.1 L500.0 163.0" class="stroke-b" fill="none"/>
<circle cx="156.0" cy="56.6" r="4" class="fill-b"/>
<circle cx="242.0" cy="94.1" r="4" class="fill-b"/>
<circle cx="328.0" cy="109.6" r="4" class="fill-b"/>
<circle cx="414.0" cy="136.1" r="4" class="fill-b"/>
<circle cx="500.0" cy="163.0" r="4" class="fill-b"/>
<text x="506.0" y="167.0" class="fig-small" text-anchor="start">Midpoint</text>
<path d="M242.0 54.2 L328.0 101.4 L414.0 183.1 L500.0 241.2" class="stroke-c" fill="none"/>
<circle cx="242.0" cy="54.2" r="4" class="fill-c"/>
<circle cx="328.0" cy="101.4" r="4" class="fill-c"/>
<circle cx="414.0" cy="183.1" r="4" class="fill-c"/>
<circle cx="500.0" cy="240.0" r="4" class="fill-c"/>
<text x="506.0" y="244.0" class="fig-small" text-anchor="start">RK4</text>
<text x="285.0" y="282.0" class="fig-small" text-anchor="middle">network evaluations (NFE)</text>
<text x="70.0" y="20.0" class="fig-small" text-anchor="start">|x(1) - 2.65|</text>
</svg>
Figure 1. Each curve falls faster the higher the order, but at small budgets the order does not pay: RK4 is worse than midpoint at 4 and at 8 evaluations, and better only from 16.
```

At tiny budgets a higher order doesn't pay: one RK4 step is worse than two midpoint steps. The order describes how fast the error falls once $h$ is small, not how good the solver is at 4 evaluations. Real systems pick cheap solvers: Matcha-TTS uses Euler with 2 to 10 steps, F5-TTS Euler with 16 or 32, and Voicebox midpoint with NFE 32.

No solver gets you to one evaluation, because the field itself is the problem: its first value points at the data mean, not at your endpoint. To fix that, you have to change the field.

## Why straight paths need fewer steps

Suppose a trajectory is exactly straight, so its velocity is the same at every time. Then Euler's step $x + h\,v$ moves exactly along the path, and one step of size 1 lands on the endpoint with no error. A curved trajectory has a velocity that changes, so each Euler step is wrong by an amount that grows with the curvature, and you need small steps to keep up with it.

So the number of steps a flow needs depends on how straight its trajectories are, and that depends on which noise sample is paired with which data sample during training. Paths from random pairs cross, the network averages them, and the average bends. Two tricks change the pairing so that paths cross less.

## Minibatch optimal transport pairing

*Minibatch optimal transport* (Tong et al., 2024) changes which noise sample each data sample is paired with. Within each batch, it finds the permutation that minimizes the total squared distance, with `scipy.optimize.linear_sum_assignment`. Paired this way, paths rarely cross, so the average at each point is close to one straight path.

Take noise $\{-1, 1\}$ and data $\{1.5, -2\}$ (Figure 2). Pairing in order costs $2.5^2 + 3^2 = 15.25$, and the two paths cross. The swapped pairing costs $1^2 + 0.5^2 = 1.25$, and neither crosses the other. Because paths that cross are exactly the pairs where each could have moved less by trading partners, the cheapest pairing has no crossings.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Left, noise -1 and 1 paired in order with data 1.5 and -2, so the paths cross and cost 15.25. Right, the swapped pairing, where the paths don't cross and cost 1.25.">
<defs><marker id="c053b-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="50.0" y1="180.0" x2="270.0" y2="180.0" class="stroke-ink"/>
<line x1="50.0" y1="40.0" x2="50.0" y2="180.0" class="stroke-ink"/>
<line x1="50.0" y1="40.0" x2="50.0" y2="180.0" class="stroke-grid"/>
<text x="50.0" y="196.0" class="fig-small" text-anchor="middle">0</text>
<line x1="270.0" y1="40.0" x2="270.0" y2="180.0" class="stroke-grid"/>
<text x="270.0" y="196.0" class="fig-small" text-anchor="middle">1</text>
<line x1="50.0" y1="164.4" x2="270.0" y2="164.4" class="stroke-grid"/>
<text x="44.0" y="168.4" class="fig-small" text-anchor="end">-2</text>
<line x1="50.0" y1="102.2" x2="270.0" y2="102.2" class="stroke-grid"/>
<text x="44.0" y="106.2" class="fig-small" text-anchor="end">0</text>
<line x1="50.0" y1="40.0" x2="270.0" y2="40.0" class="stroke-grid"/>
<text x="44.0" y="44.0" class="fig-small" text-anchor="end">2</text>
<text x="160.0" y="22.0" class="fig-bold" text-anchor="middle">Paired in order</text>
<path d="M50.0 133.3 L270.0 55.6" class="stroke-a" fill="none"/>
<path d="M50.0 71.1 L270.0 164.4" class="stroke-b" fill="none"/>
<text x="160.0" y="218.0" class="fig-small" text-anchor="middle">total squared distance 15.25</text>
<line x1="340.0" y1="180.0" x2="560.0" y2="180.0" class="stroke-ink"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="180.0" class="stroke-ink"/>
<line x1="340.0" y1="40.0" x2="340.0" y2="180.0" class="stroke-grid"/>
<text x="340.0" y="196.0" class="fig-small" text-anchor="middle">0</text>
<line x1="560.0" y1="40.0" x2="560.0" y2="180.0" class="stroke-grid"/>
<text x="560.0" y="196.0" class="fig-small" text-anchor="middle">1</text>
<line x1="340.0" y1="164.4" x2="560.0" y2="164.4" class="stroke-grid"/>
<text x="334.0" y="168.4" class="fig-small" text-anchor="end">-2</text>
<line x1="340.0" y1="102.2" x2="560.0" y2="102.2" class="stroke-grid"/>
<text x="334.0" y="106.2" class="fig-small" text-anchor="end">0</text>
<line x1="340.0" y1="40.0" x2="560.0" y2="40.0" class="stroke-grid"/>
<text x="334.0" y="44.0" class="fig-small" text-anchor="end">2</text>
<text x="450.0" y="22.0" class="fig-bold" text-anchor="middle">Swapped pairing</text>
<path d="M340.0 133.3 L560.0 164.4" class="stroke-a" fill="none"/>
<path d="M340.0 71.1 L560.0 55.6" class="stroke-b" fill="none"/>
<text x="450.0" y="218.0" class="fig-small" text-anchor="middle">total squared distance 1.25</text>
</svg>
Figure 2. Minibatch optimal transport picks the pairing with the smallest total squared distance. Notice that the cheaper pairing is also the one whose paths don't cross.
```

> [!TIP]
> Matcha-TTS and CosyVoice call their loss "OT-CFM," but they mean the straight path with random pairing, not Tong's minibatch OT coupling.

The following demo lets you turn the pairing on and watch the paths straighten.

```demo
{"id": "flow-trajectories", "caption": "Turn on minibatch OT and watch the path bend readout fall toward 1, then lower the steps and compare the Euler samples with and without it."}
```

## One round of reflow

Minibatch OT only reorders pairs inside a small batch. *Reflow* (Liu et al., 2022) uses the trained model itself to build better pairs. Draw noise $z_0$, solve the ODE carefully to get $z_1$, and train a second flow on the fixed pairs $(z_0, z_1)$. The first flow's ODE paths never cross, because a velocity field has one direction at each point and time. So lines between its pairs cross far less than lines between random pairs, and the second flow comes out much straighter.

To measure straightness, compare each Euler step's velocity with the chord $z_1 - z_0$:

$$
S = \frac{1}{n}\sum_{k=0}^{n-1} \mathbb{E}\left\lVert (z_1 - z_0) - v(z_k, t_k) \right\rVert^2
$$

Read it aloud: straightness is the average squared gap between the velocity along the way and the one velocity that makes the trip in a straight line, so it is zero exactly when every trajectory is straight. A constant field gives $S = 0$, and the curved Gaussian flow gives $S > 0$.

On the eight-blob ring in the **Run** block, one reflow cut $S$ from 1.51 to 0.009, and the mean radius of one-step samples went from 0.23 to 1.96, against 2.00 for the data (Figure 3). One Euler step after reflow lands on the ring, where before it landed near the middle.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Left, straightness of the ring flow falls from 1.51 to 0.009 after one reflow. Right, the mean radius of one-step samples rises from 0.23 to 1.96, against 2.00 for the data.">
<text x="160" y="24" class="fig-bold" text-anchor="middle">Straightness S</text>
<line x1="50" y1="200" x2="270" y2="200" class="stroke-ink"/>
<rect x="75" y="58.4" width="70" height="141.6" class="fill-a"/>
<text x="110" y="52.4" class="fig-small" text-anchor="middle">1.51</text>
<text x="110" y="218" class="fig-small" text-anchor="middle">flow</text>
<rect x="170" y="199.2" width="70" height="0.8" class="fill-b"/>
<text x="205" y="193.2" class="fig-small" text-anchor="middle">0.009</text>
<text x="205" y="218" class="fig-small" text-anchor="middle">after reflow</text>
<text x="450" y="24" class="fig-bold" text-anchor="middle">Mean radius after 1 Euler step</text>
<line x1="340" y1="200" x2="560" y2="200" class="stroke-ink"/>
<rect x="365" y="185.6" width="70" height="14.4" class="fill-a"/>
<text x="400" y="179.6" class="fig-small" text-anchor="middle">0.23</text>
<text x="400" y="218" class="fig-small" text-anchor="middle">flow</text>
<rect x="460" y="77.5" width="70" height="122.5" class="fill-b"/>
<text x="495" y="71.5" class="fig-small" text-anchor="middle">1.96</text>
<text x="495" y="218" class="fig-small" text-anchor="middle">after reflow</text>
<line x1="340" y1="75.0" x2="560" y2="75.0" class="stroke-muted dashed"/><text x="560" y="70.0" class="fig-small" text-anchor="end">data 2.00</text>
</svg>
Figure 3. One round of reflow on the eight-blob ring. The flow's trajectories become nearly straight, and one Euler step now lands on the ring instead of near its middle.
```

> [!KEY]
> You can lower the cost of a sample in two ways. A higher-order solver follows a curved field with fewer, costlier steps, but its first step still points at the data mean. Straightening the field with minibatch OT or reflow makes one Euler step nearly exact, which is how few-step speech generators work.

> [!RECAP]
> - Euler, midpoint, and RK4 have orders 1, 2, and 4 and cost 1, 2, and 4 NFE per step. At small budgets a lower order can win.
> - A straight trajectory has constant velocity, so one Euler step is exact. A curved one needs many.
> - Minibatch OT pairs each batch by the permutation with the smallest total squared distance, so the paths cross less.
> - Reflow retrains on the flow's own noise-and-sample pairs. On the ring it cut straightness from 1.51 to 0.009.

## Build it step by step

The exercise gives you `ot_path`, `flow_matching_loss`, and `gaussian_velocity` from the last chapter in full. Build the rest in the order the checks test them.

1. **`integrate(velocity, x0, n_steps, method)`** loops over `k` with `t = k / n_steps` and applies the Euler, midpoint, or RK4 update. Compute each time from `k`, so the last step ends on 1.
2. **`minibatch_ot_pairing(x0, x1)`** flattens each sample, builds the `(n, n)` matrix of squared distances by broadcasting, and returns the column indices from `linear_sum_assignment`.
3. **`straightness(velocity, x0, n_steps)`** runs Euler, keeps every velocity, and averages the squared norm of `chord - v` over steps and samples. Sum over the feature axis before averaging over samples.

## Where this breaks

- **OT pairing fades in high dimension.** For 256 mel spectrograms, the best pairing is barely better than a random one.
- **Reflow compounds errors.** The second flow learns from the first flow's imperfect samples, so people stop after one or two rounds.
- **A straight field is not a free lunch.** A one-step reflowed flow still inherits any blur or artifact in the pairs it was trained on.

## The exercise

You build `integrate`, `minibatch_ot_pairing`, and `straightness`. The checks confirm the Euler and midpoint numbers in the table, RK4 at 16 steps, the three convergence orders on $dx/dt = t\,x$, and your pairing against all 5,040 pairings of seven points. A straightness check requires zero for straight flows and a positive value for the curved one. The last check trains a small flow, reflows it once, and requires straightness to fall by more than 70 percent.

Press **Run** to race the solvers, train and reflow a flow on the ring, and hear its samples as notes. One Euler step plays quiet, out-of-tune notes, and eight steps, or one step after reflow, play the scale.
