---
title: Flow matching
slug: 060-flow-matching
part: "Part 7 — Diffusion and flow matching"
summary: Teach a network the velocity that carries noise to data along straight paths, and sample by taking Euler steps along it.
minutes: 45
gpu: false
objectives:
  - Build the straight path from a noise sample to a data sample, its target velocity, and the flow-matching loss.
  - Explain why straight training paths still give a curved velocity field, and why one Euler step returns the data mean.
  - Sample from a velocity field with Euler steps and check the result against a closed-form Gaussian flow.
  - "Convert between the two time conventions: flow matching's noise at t = 0 and the diffusion chapters' data at t = 0."
prereqs:
  - 050-normalizing-flows-and-vits
  - 056-diffusion-from-scratch
  - 057-fast-samplers
exercise: 060-flow-matching
---

# Flow matching

Diffusion reaches a sample by a long detour. It adds noise in many small steps, trains a network to undo each one, and needs a schedule and a choice of target to keep it consistent. If you designed the trip from a noise sample to a data sample yourself, the simplest trip would be a straight line at constant speed.

Flow matching builds a generator from exactly that. You draw a straight line from a noise sample to a data sample, and you teach a network how fast to move along it. At sampling time the network is a velocity field, like the wind map in a weather report: drop a noise sample into the wind, let it drift for one unit of time, and it lands on something that looks like data. Matcha-TTS, Voicebox, F5-TTS, and CosyVoice 2's decoder all generate speech this way.

A puzzle drives the chapter: if you train on straight lines, why doesn't sampling follow them, and why does one step give a blur? One warning comes first, because it causes more bugs than anything else in this part. Flow-matching papers run the clock in the opposite direction from the diffusion papers. Here, time 0 is noise and time 1 is data. [The next chapter](/lessons/061-straighter-paths-and-better-solvers) makes the paths straighter so that few steps suffice, and [the chapter after it](/lessons/062-flow-matching-for-speech) builds a speech model on this one.

## What you already know

- A continuous [normalizing flow](/lessons/050-normalizing-flows-and-vits) maps noise to data by solving an ordinary differential equation (ODE).
- A diffusion model's noisy sample is $\sqrt{\bar\alpha}$ times the data plus $\sqrt{1 - \bar\alpha}$ times Gaussian noise.
- [DDIM](/lessons/057-fast-samplers) is a first-order step along the probability-flow ODE.
- An optimizer can fit a network to any regression target you can compute.

## A straight path for every pair

> [!WARNING]
> In this chapter, $x_0$ is **noise** and $x_1$ is **data**, as in Lipman et al., Matcha-TTS, Voicebox, and F5-TTS. In the diffusion chapters, $x_0$ is data. The same symbol means opposite things, so restate the convention at the top of every file you write.

Pick one noise sample $x_0 \sim \mathcal{N}(0, I)$ and one data sample $x_1$, such as one mel frame, and walk from the first to the second in a straight line at constant speed. The point you reach at time $t$ between 0 and 1 is the *conditional path*, and the speed you walk at is its *target velocity*:

$$
x_t = \bigl(1 - (1 - \sigma_{\min})\,t\bigr)\,x_0 + t\,x_1, \qquad
u = x_1 - (1 - \sigma_{\min})\,x_0
$$

Read it aloud: at time $t$, keep a fraction of the noise that shrinks from 1 to $\sigma_{\min}$, and add a fraction $t$ of the data. The velocity is the derivative of $x_t$ with respect to $t$, and it doesn't depend on $t$, because the path is straight. The constant $\sigma_{\min}$ leaves a trace of noise at $t = 1$. Matcha-TTS uses $10^{-4}$. F5-TTS and CosyVoice 2 use 0, which gives the target $u = x_1 - x_0$, "data minus noise." In numpy:

```python
t = t.reshape(-1, *([1] * (x0.ndim - 1)))      # (B,) -> (B, 1, 1)
x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
u = x1 - (1 - sigma_min) * x0
```

### Work one path by hand

Take $\sigma_{\min} = 0$, noise $x_0 = -1$, data $x_1 = 2$, and $t = 0.25$. Then $x_t = 0.75 \times (-1) + 0.25 \times 2 = -0.25$, and the target is $u = 2 - (-1) = 3$.

## The loss, and the field it learns

Training is plain regression. Draw a batch of data, a batch of noise, and a time $t \sim U[0, 1]$ for each example. Build $x_t$ and $u$, and ask the network $v_\theta(x_t, t)$ to output $u$:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\,x_0,\,x_1}\left\lVert v_\theta(x_t, t) - u \right\rVert^2
$$

Read it aloud: the loss is the average squared distance between the network's velocity and the straight path's velocity. Training solves no ODE and needs no schedule.

Now the puzzle. The network sees only the point $x_t$ and the time, not which pair produced it. Many straight paths pass through the same point, as the left panel of Figure 1 shows, so the best the network can do is output their average. Lipman et al. prove that this average, the *marginal velocity field*, still carries the noise distribution onto the data distribution. Each conditional path is straight, but the average of many crossing paths bends.

```figure
<svg viewBox="0 0 620 300" role="img" aria-label="Left, four straight lines from noise samples to randomly paired data samples, which cross. Right, the four exact flow paths from the same noise samples, which bend and never cross.">
<defs><marker id="c052-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<line x1="50.0" y1="240.0" x2="280.0" y2="240.0" class="stroke-ink"/>
<line x1="50.0" y1="50.0" x2="50.0" y2="240.0" class="stroke-ink"/>
<line x1="50.0" y1="50.0" x2="50.0" y2="240.0" class="stroke-grid"/>
<text x="50.0" y="256.0" class="fig-small" text-anchor="middle">0</text>
<line x1="165.0" y1="50.0" x2="165.0" y2="240.0" class="stroke-grid"/>
<text x="165.0" y="256.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="280.0" y1="50.0" x2="280.0" y2="240.0" class="stroke-grid"/>
<text x="280.0" y="256.0" class="fig-small" text-anchor="middle">1</text>
<line x1="50.0" y1="240.0" x2="280.0" y2="240.0" class="stroke-grid"/>
<text x="44.0" y="244.0" class="fig-small" text-anchor="end">-2</text>
<line x1="50.0" y1="203.5" x2="280.0" y2="203.5" class="stroke-grid"/>
<text x="44.0" y="207.5" class="fig-small" text-anchor="end">-1</text>
<line x1="50.0" y1="166.9" x2="280.0" y2="166.9" class="stroke-grid"/>
<text x="44.0" y="170.9" class="fig-small" text-anchor="end">0</text>
<line x1="50.0" y1="130.4" x2="280.0" y2="130.4" class="stroke-grid"/>
<text x="44.0" y="134.4" class="fig-small" text-anchor="end">1</text>
<line x1="50.0" y1="93.8" x2="280.0" y2="93.8" class="stroke-grid"/>
<text x="44.0" y="97.8" class="fig-small" text-anchor="end">2</text>
<line x1="50.0" y1="57.3" x2="280.0" y2="57.3" class="stroke-grid"/>
<text x="44.0" y="61.3" class="fig-small" text-anchor="end">3</text>
<text x="165.0" y="32.0" class="fig-bold" text-anchor="middle">Straight training paths</text>
<text x="165.0" y="272.0" class="fig-small" text-anchor="middle">time t (0 is noise, 1 is data)</text>
<path d="M50.0 210.8 L280.0 79.2" class="stroke-a" fill="none"/>
<circle cx="280.0" cy="79.2" r="4" class="fill-ink"/>
<path d="M50.0 181.5 L280.0 108.5" class="stroke-b" fill="none"/>
<circle cx="280.0" cy="108.5" r="4" class="fill-ink"/>
<path d="M50.0 152.3 L280.0 75.6" class="stroke-c" fill="none"/>
<circle cx="280.0" cy="75.6" r="4" class="fill-ink"/>
<path d="M50.0 123.1 L280.0 112.1" class="stroke-d" fill="none"/>
<circle cx="280.0" cy="112.1" r="4" class="fill-ink"/>
<line x1="350.0" y1="240.0" x2="580.0" y2="240.0" class="stroke-ink"/>
<line x1="350.0" y1="50.0" x2="350.0" y2="240.0" class="stroke-ink"/>
<line x1="350.0" y1="50.0" x2="350.0" y2="240.0" class="stroke-grid"/>
<text x="350.0" y="256.0" class="fig-small" text-anchor="middle">0</text>
<line x1="465.0" y1="50.0" x2="465.0" y2="240.0" class="stroke-grid"/>
<text x="465.0" y="256.0" class="fig-small" text-anchor="middle">0.5</text>
<line x1="580.0" y1="50.0" x2="580.0" y2="240.0" class="stroke-grid"/>
<text x="580.0" y="256.0" class="fig-small" text-anchor="middle">1</text>
<line x1="350.0" y1="240.0" x2="580.0" y2="240.0" class="stroke-grid"/>
<text x="344.0" y="244.0" class="fig-small" text-anchor="end">-2</text>
<line x1="350.0" y1="203.5" x2="580.0" y2="203.5" class="stroke-grid"/>
<text x="344.0" y="207.5" class="fig-small" text-anchor="end">-1</text>
<line x1="350.0" y1="166.9" x2="580.0" y2="166.9" class="stroke-grid"/>
<text x="344.0" y="170.9" class="fig-small" text-anchor="end">0</text>
<line x1="350.0" y1="130.4" x2="580.0" y2="130.4" class="stroke-grid"/>
<text x="344.0" y="134.4" class="fig-small" text-anchor="end">1</text>
<line x1="350.0" y1="93.8" x2="580.0" y2="93.8" class="stroke-grid"/>
<text x="344.0" y="97.8" class="fig-small" text-anchor="end">2</text>
<line x1="350.0" y1="57.3" x2="580.0" y2="57.3" class="stroke-grid"/>
<text x="344.0" y="61.3" class="fig-small" text-anchor="end">3</text>
<text x="465.0" y="32.0" class="fig-bold" text-anchor="middle">Exact flow paths</text>
<text x="465.0" y="272.0" class="fig-small" text-anchor="middle">time t (0 is noise, 1 is data)</text>
<path d="M350.0 210.8 L353.9 208.8 L357.8 206.8 L361.7 204.8 L365.6 202.9 L369.5 200.9 L373.4 198.9 L377.3 197.0 L381.2 195.0 L385.1 193.1 L389.0 191.1 L392.9 189.2 L396.8 187.3 L400.7 185.3 L404.6 183.4 L408.5 181.5 L412.4 179.6 L416.3 177.7 L420.2 175.8 L424.1 173.9 L428.0 172.1 L431.9 170.2 L435.8 168.4 L439.7 166.5 L443.6 164.7 L447.5 162.9 L451.4 161.1 L455.3 159.3 L459.2 157.5 L463.1 155.8 L466.9 154.0 L470.8 152.3 L474.7 150.6 L478.6 148.9 L482.5 147.3 L486.4 145.6 L490.3 144.0 L494.2 142.5 L498.1 140.9 L502.0 139.4 L505.9 137.9 L509.8 136.4 L513.7 135.0 L517.6 133.6 L521.5 132.2 L525.4 130.9 L529.3 129.6 L533.2 128.3 L537.1 127.1 L541.0 125.9 L544.9 124.7 L548.8 123.6 L552.7 122.5 L556.6 121.5 L560.5 120.4 L564.4 119.5 L568.3 118.5 L572.2 117.6 L576.1 116.6 L580.0 115.8" class="stroke-a" fill="none"/>
<circle cx="580.0" cy="115.8" r="4" class="fill-ink"/>
<path d="M350.0 181.5 L353.9 180.1 L357.8 178.6 L361.7 177.1 L365.6 175.6 L369.5 174.1 L373.4 172.6 L377.3 171.2 L381.2 169.7 L385.1 168.2 L389.0 166.7 L392.9 165.3 L396.8 163.8 L400.7 162.3 L404.6 160.9 L408.5 159.4 L412.4 157.9 L416.3 156.5 L420.2 155.0 L424.1 153.6 L428.0 152.1 L431.9 150.7 L435.8 149.2 L439.7 147.8 L443.6 146.4 L447.5 144.9 L451.4 143.5 L455.3 142.1 L459.2 140.7 L463.1 139.3 L466.9 137.9 L470.8 136.5 L474.7 135.1 L478.6 133.7 L482.5 132.3 L486.4 130.9 L490.3 129.6 L494.2 128.2 L498.1 126.9 L502.0 125.5 L505.9 124.2 L509.8 122.9 L513.7 121.6 L517.6 120.3 L521.5 119.0 L525.4 117.8 L529.3 116.5 L533.2 115.2 L537.1 114.0 L541.0 112.8 L544.9 111.6 L548.8 110.4 L552.7 109.2 L556.6 108.0 L560.5 106.8 L564.4 105.7 L568.3 104.5 L572.2 103.4 L576.1 102.3 L580.0 101.2" class="stroke-b" fill="none"/>
<circle cx="580.0" cy="101.2" r="4" class="fill-ink"/>
<path d="M350.0 152.3 L353.9 151.3 L357.8 150.3 L361.7 149.3 L365.6 148.3 L369.5 147.3 L373.4 146.3 L377.3 145.3 L381.2 144.3 L385.1 143.3 L389.0 142.3 L392.9 141.3 L396.8 140.3 L400.7 139.3 L404.6 138.3 L408.5 137.3 L412.4 136.3 L416.3 135.3 L420.2 134.2 L424.1 133.2 L428.0 132.2 L431.9 131.1 L435.8 130.1 L439.7 129.1 L443.6 128.0 L447.5 127.0 L451.4 125.9 L455.3 124.9 L459.2 123.8 L463.1 122.7 L466.9 121.7 L470.8 120.6 L474.7 119.5 L478.6 118.4 L482.5 117.3 L486.4 116.2 L490.3 115.1 L494.2 114.0 L498.1 112.8 L502.0 111.7 L505.9 110.5 L509.8 109.4 L513.7 108.2 L517.6 107.0 L521.5 105.8 L525.4 104.6 L529.3 103.4 L533.2 102.2 L537.1 100.9 L541.0 99.7 L544.9 98.4 L548.8 97.1 L552.7 95.8 L556.6 94.5 L560.5 93.2 L564.4 91.9 L568.3 90.6 L572.2 89.2 L576.1 87.9 L580.0 86.5" class="stroke-c" fill="none"/>
<circle cx="580.0" cy="86.5" r="4" class="fill-ink"/>
<path d="M350.0 123.1 L353.9 122.6 L357.8 122.1 L361.7 121.6 L365.6 121.1 L369.5 120.6 L373.4 120.0 L377.3 119.5 L381.2 119.0 L385.1 118.5 L389.0 117.9 L392.9 117.4 L396.8 116.8 L400.7 116.3 L404.6 115.7 L408.5 115.2 L412.4 114.6 L416.3 114.0 L420.2 113.4 L424.1 112.8 L428.0 112.2 L431.9 111.6 L435.8 111.0 L439.7 110.3 L443.6 109.7 L447.5 109.0 L451.4 108.4 L455.3 107.7 L459.2 107.0 L463.1 106.2 L466.9 105.5 L470.8 104.7 L474.7 104.0 L478.6 103.2 L482.5 102.3 L486.4 101.5 L490.3 100.6 L494.2 99.7 L498.1 98.8 L502.0 97.9 L505.9 96.9 L509.8 95.9 L513.7 94.8 L517.6 93.7 L521.5 92.6 L525.4 91.5 L529.3 90.3 L533.2 89.1 L537.1 87.9 L541.0 86.6 L544.9 85.2 L548.8 83.9 L552.7 82.5 L556.6 81.1 L560.5 79.6 L564.4 78.1 L568.3 76.6 L572.2 75.1 L576.1 73.5 L580.0 71.9" class="stroke-d" fill="none"/>
<circle cx="580.0" cy="71.9" r="4" class="fill-ink"/>
</svg>
Figure 1. Straight paths from random pairs cross each other, so at a crossing point a network cannot know which line it is on. The field it learns is their average. Its paths, drawn here for data from N(2, 0.5^2), bend and end at 2 + 0.5 times the starting noise.
```

> [!KEY]
> Flow matching trains on straight lines from each noise sample to each data sample. The network learns their average at each point, which is a curved field that still moves all of the noise onto all of the data.

## A field you can write down

When the data is one-dimensional and Gaussian, $\mathcal{N}(\mu, s^2)$, the average has a closed form. The point $x_t$ is then Gaussian with mean $t\mu$ and variance $(1 - t)^2 + t^2 s^2$, and the marginal field is:

$$
v(x, t) = \mu + \frac{t s^2 - (1 - t)}{(1 - t)^2 + t^2 s^2}\,(x - t\mu)
$$

Read it aloud: the field pushes every point toward the data mean, plus a correction proportional to its distance from the moving mean $t\mu$. Solving it exactly sends $x_0$ to $\mu + s\,x_0$, an oracle with a known answer.

At $t = 0$ the fraction is $-1$, so $v(x_0, 0) = \mu - x_0$. That is the whole puzzle in one line: at the start, every noise sample sees the average of every path through it, and that average points at the data mean $\mu$, not at the sample's own endpoint.

## Sampling with Euler steps

To sample, you integrate $dx/dt = v_\theta(x, t)$ from a noise sample at $t = 0$ to $t = 1$. The simplest way is *Euler's method*: split the unit of time into $n$ steps of size $h = 1/n$, and at each step move along the velocity at the start of the step:

$$
x \leftarrow x + h\,v(x, t)
$$

Read it aloud: ask the field which way to go, then walk that way for time $h$. Each call to the network is one *function evaluation*, and the count of them, the NFE, is the price of a sample.

With one step ($n = 1$, $h = 1$), the update from $t = 0$ is $x_0 + (\mu - x_0) = \mu$. The result is the data mean whatever $x_0$ was, so every noise sample collapses to the same blur. That answers the puzzle. The training paths are straight, but the field at the start is the average of all of them, and one big step follows that average instead of your own path.

Take $\mu = 2$, $s = 0.5$, and $x_0 = 1.3$, so the exact answer is $2 + 0.5 \times 1.3 = 2.65$. Two Euler steps of size 0.5 go as follows:

1. At $t = 0$: $v = 2 - 1.3 = 0.7$, so $x = 1.3 + 0.5 \times 0.7 = 1.65$.
2. At $t = 0.5$: the variance is $0.25 + 0.0625 = 0.3125$ and the fraction is $(0.125 - 0.5) / 0.3125 = -1.2$. So $v = 2 - 1.2 \times 0.65 = 1.22$, and $x = 1.65 + 0.5 \times 1.22 = 2.26$.

Two steps land at 2.26, closer than one step's 2.0 but short of 2.65 (Figure 2). Each step follows a tangent line, and the exact path curves away from it, so more steps hug the curve better. With 4, 8, and 32 steps, Euler ends at 2.4427, 2.5392, and 2.6206. Matcha-TTS uses 2 to 10 steps, and F5-TTS uses 16 or 32. [The next chapter](/lessons/061-straighter-paths-and-better-solvers) compares Euler with better solvers and straightens the field so fewer steps are enough.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Flow time against position for the Gaussian flow. The exact path rises from 1.3 to 2.65. One Euler step jumps straight to 2.0, the data mean. Two Euler steps reach 2.26.">
<line x1="70" y1="230" x2="540" y2="230" class="stroke-ink"/><line x1="70" y1="40" x2="70" y2="230" class="stroke-ink"/>
<line x1="70.0" y1="40" x2="70.0" y2="230" class="stroke-grid"/><text x="70.0" y="246" class="fig-small" text-anchor="middle">0</text>
<line x1="187.5" y1="40" x2="187.5" y2="230" class="stroke-grid"/><text x="187.5" y="246" class="fig-small" text-anchor="middle">0.25</text>
<line x1="305.0" y1="40" x2="305.0" y2="230" class="stroke-grid"/><text x="305.0" y="246" class="fig-small" text-anchor="middle">0.5</text>
<line x1="422.5" y1="40" x2="422.5" y2="230" class="stroke-grid"/><text x="422.5" y="246" class="fig-small" text-anchor="middle">0.75</text>
<line x1="540.0" y1="40" x2="540.0" y2="230" class="stroke-grid"/><text x="540.0" y="246" class="fig-small" text-anchor="middle">1</text>
<line x1="70" y1="230.0" x2="540" y2="230.0" class="stroke-grid"/><text x="64" y="234.0" class="fig-small" text-anchor="end">1</text>
<line x1="70" y1="182.5" x2="540" y2="182.5" class="stroke-grid"/><text x="64" y="186.5" class="fig-small" text-anchor="end">1.5</text>
<line x1="70" y1="135.0" x2="540" y2="135.0" class="stroke-grid"/><text x="64" y="139.0" class="fig-small" text-anchor="end">2</text>
<line x1="70" y1="87.5" x2="540" y2="87.5" class="stroke-grid"/><text x="64" y="91.5" class="fig-small" text-anchor="end">2.5</text>
<line x1="70" y1="40.0" x2="540" y2="40.0" class="stroke-grid"/><text x="64" y="44.0" class="fig-small" text-anchor="end">3</text>
<line x1="70" y1="135.0" x2="540" y2="135.0" class="stroke-muted dashed"/><text x="536" y="129.0" class="fig-small" text-anchor="end">data mean 2.0</text>
<path d="M70.0 201.5 L79.4 200.2 L88.8 198.8 L98.2 197.5 L107.6 196.1 L117.0 194.7 L126.4 193.3 L135.8 191.8 L145.2 190.4 L154.6 188.9 L164.0 187.4 L173.4 185.9 L182.8 184.4 L192.2 182.8 L201.6 181.2 L211.0 179.6 L220.4 177.9 L229.8 176.2 L239.2 174.5 L248.6 172.7 L258.0 170.9 L267.4 169.0 L276.8 167.1 L286.2 165.1 L295.6 163.1 L305.0 161.0 L314.4 158.8 L323.8 156.5 L333.2 154.2 L342.6 151.8 L352.0 149.2 L361.4 146.6 L370.8 143.9 L380.2 141.1 L389.6 138.1 L399.0 135.1 L408.4 131.9 L417.8 128.6 L427.2 125.1 L436.6 121.5 L446.0 117.8 L455.4 113.9 L464.8 109.9 L474.2 105.8 L483.6 101.5 L493.0 97.1 L502.4 92.5 L511.8 87.9 L521.2 83.1 L530.6 78.2 L540.0 73.2" class="stroke-a" fill="none"/>
<path d="M70.0 201.5 L540.0 135.0" class="stroke-b" fill="none"/>
<circle cx="70.0" cy="201.5" r="4" class="fill-b"/>
<circle cx="540.0" cy="135.0" r="4" class="fill-b"/>
<path d="M70.0 201.5 L305.0 168.2 L540.0 110.3" class="stroke-c" fill="none"/>
<circle cx="70.0" cy="201.5" r="4" class="fill-c"/>
<circle cx="305.0" cy="168.2" r="4" class="fill-c"/>
<circle cx="540.0" cy="110.3" r="4" class="fill-c"/>
<text x="534.0" y="65.2" class="fig-small" text-anchor="end">exact 2.65</text>
<text x="534.0" y="102.3" class="fig-small" text-anchor="end">2 steps 2.26</text>
<text x="534.0" y="151.0" class="fig-small" text-anchor="end">1 step 2.0</text>
<text x="305.0" y="285" class="fig-small" text-anchor="middle">flow time t (0 is noise, 1 is data)</text>
<text x="70" y="26" class="fig-small" text-anchor="start">position x(t)</text>
</svg>
Figure 2. Each step follows a tangent, so Euler undershoots the curved exact path. One step lands on the data mean, 2.0, for any noise sample, and two steps reach 2.26 against the exact 2.65.
```

```demo
{"id": "flow-trajectories", "caption": "Turn on the training pairs to see the straight conditional paths, then set the steps to 1 and watch every Euler sample land near the data mean although the exact paths bend."}
```

## The same state in the diffusion convention

Every Gaussian path is a diffusion process with some schedule, so you can move any flow state into the diffusion chapters' language, where data sits at time 0. Two points match when their signal-to-noise ratios match: $t^2 / (1 - t)^2 = \bar\alpha / (1 - \bar\alpha)$. With $r = \sqrt{t^2 + (1 - t)^2}$, that gives:

$$
\bar\alpha = \frac{t^2}{t^2 + (1 - t)^2}, \qquad
x_{\text{vp}} = \frac{x_t}{r}
$$

Read it aloud: the flow point is the diffusion point shrunk by $r$, because the squares of $t$ and $1 - t$ don't add to one. To convert the target, recover the data and the noise first: $x_{\text{data}} = x_t + (1 - t)\,u$ and $\epsilon = x_t - t\,u$. Then combine them into $v = \sqrt{\bar\alpha}\,\epsilon - \sqrt{1 - \bar\alpha}\,x_{\text{data}}$.

For the path from before (data 2, noise $-1$, $t = 0.25$), $\bar\alpha = 0.0625 / 0.625 = 0.1$, the diffusion point is $-0.25 / 0.790569 = -0.316228$, and the v target is $\sqrt{0.1} \times (-1) - \sqrt{0.9} \times 2 = -2.213594$. The same algebra shows that one DDIM step on the straight path is exactly one Euler step: DDIM is Euler in disguise.

> [!KEY]
> A flow time $t$ and a diffusion step with the same signal-to-noise ratio describe the same noisy sample up to a scale. Convert through the data and noise estimates, never by swapping symbols.

> [!RECAP]
> - The path runs from noise at 0 to data at 1, and its target velocity is $x_1 - (1 - \sigma_{\min})\,x_0$.
> - The loss regresses $v_\theta(x_t, t)$ onto that target, so the network learns the curved average of all such paths.
> - One Euler step from noise lands on the data mean, because the field at $t = 0$ is the average over every path. More steps follow the curve.
> - Matching the signal-to-noise ratio converts a flow time into a DDPM $\bar\alpha = t^2 / (t^2 + (1 - t)^2)$.

## Build it step by step

Build the functions in the order the checks test them.

1. **`ot_path(x0, x1, t, sigma_min)`** returns `(x_t, target)`, both with the shape of `x0`. Reshape a `(B,)` time to `(B, 1, ..., 1)` first. Mistake: multiplying a `(B,)` time into a `(B, C, T)` tensor, which broadcasts over the wrong axis when `T == B`.
2. **`flow_matching_loss(model, x0, x1, t, sigma_min)`** calls `ot_path`, then `model(x_t, t)`, and returns the mean squared difference.
3. **`gaussian_velocity(x, t, mu, s)`** is the closed-form field. At `t = 0` it returns `mu - x`.
4. **`euler(velocity, x0, n_steps)`** loops over `k` with `t = k / n_steps` and applies `x + h * velocity(x, t)`. Compute each time from `k`, so the last step ends on 1.
5. **`to_diffusion(x_t, t, u)`** returns `(alpha_bar, x_vp, v)`. Recover the data and the noise, then combine them. Mistake: writing $v$ with the flow's weights $t$ and $1 - t$ instead of $\sqrt{\bar\alpha}$ and $\sqrt{1 - \bar\alpha}$.

## Where this breaks

- **The clock runs backward.** A model sampled in the opposite convention from its training integrates the wrong way and outputs noise. The SD3 family and Stable Audio Open Small put data at 0.
- **The endpoint divides by zero.** Converting a velocity to a noise estimate divides by $t$, and to a score by $1 - t$.

## The exercise

You build `ot_path`, `flow_matching_loss`, `gaussian_velocity`, `euler`, and `to_diffusion`. The checks confirm the worked values, the Gaussian field against a finite-difference derivative of its flow map, the Euler numbers above including the one-step collapse to the mean, and `to_diffusion` against the DDPM formulas.

Press **Run** to train a flow on an eight-blob ring and hear its samples as notes: each sample's angle picks a pitch on the C major scale, and its radius sets the loudness. One Euler step plays quiet, out-of-tune notes, and eight steps play the scale.
