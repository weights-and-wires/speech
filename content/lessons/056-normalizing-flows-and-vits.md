---
title: Normalizing flows and VITS
slug: 056-normalizing-flows-and-vits
part: "Part 5 — Speech synthesis"
summary: Build invertible layers with exact log-determinants, train a flow by maximum likelihood, and write the KL term that trains VITS.
minutes: 50
gpu: false
objectives:
  - Apply the change-of-variables formula, and say which direction's log-determinant it adds.
  - Build affine coupling, ActNorm, and an invertible 1x1 convolution, and verify each log-determinant against an autograd Jacobian.
  - Train a RealNVP on 2-D data by maximum likelihood, and sample from it by running the layers backward.
  - Explain how VITS joins a VAE, a flow prior, monotonic alignment search, and a HiFi-GAN decoder, and write its single-sample KL term.
prereqs:
  - 052-durations-and-alignment-search
  - 055-gan-vocoders
exercise: 056-normalizing-flows-and-vits
---

# Normalizing flows and VITS

A GAN vocoder generates fast but gives you no likelihood: it can't say how probable a clip is, so you can't train it by maximum likelihood. WaveNet gives an exact likelihood but generates one sample at a time. Is there a model with both, an exact likelihood to train on and a single parallel pass to sample from?

A *normalizing flow* is one. It is an invertible network that maps data to Gaussian noise. Run it forward and you get an exact log-likelihood. Run it backward and you turn fresh noise into data, all positions at once. Flows carried speech synthesis for several years: WaveGlow was a flow vocoder, Glow-TTS used a flow decoder, and VITS put a flow inside a variational autoencoder to make an end-to-end TTS model.

You build the invertible layers flows are made of, check each log-determinant against brute force, train a small flow on 2-D data, and write the KL term that VITS trains with.

## What you already know

- Glow-TTS aligns frames to phones by monotonic alignment search over Gaussian log-likelihoods (see [Durations and alignment search](/lessons/052-durations-and-alignment-search)).
- A HiFi-GAN generator turns latent frames into a waveform (see [GAN vocoders](/lessons/055-gan-vocoders)).
- The Jacobian of a map is the matrix of its partial derivatives, and a triangular matrix's determinant is the product of its diagonal.
- A diagonal Gaussian's log-density is $\sum_d \left[-\tfrac12 \log 2\pi - \log\sigma_d - (z_d - \mu_d)^2 / 2\sigma_d^2\right]$.
- A variational autoencoder trains an encoder $q(z \mid x)$ and a decoder with a reconstruction loss plus a KL divergence to a prior.

## Count the stretch

A flow is an invertible map $f$ from data $x$ to a latent $z = f(x)$ with a simple density, a standard Gaussian. You know the Gaussian's density, so can you read off the data's density from it? Not quite. Where $f$ stretches space, probability thins out, and where it squeezes, probability piles up. The change-of-variables formula does that accounting exactly:

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|.
$$

Read it aloud: the log-density of a data point is the log-density of its latent, plus the log of how much $f$ expands volume near that point. The determinant of the Jacobian measures that expansion. The Jacobian belongs to the data-to-latent direction, and the term is added.

Take one dimension and $z = 2x$. At $x = 1$, $z = 2$ and $\log p_Z(2) = -\tfrac12 \log 2\pi - 2 = -2.9189$. The map doubles lengths, so $\log |dz/dx| = \ln 2 = 0.6931$, and $\log p_X(1) = -2.9189 + 0.6931 = -2.2258$. That is the log-density of $\mathcal{N}(0, 1/4)$ at 1, as it must be, because $x = z / 2$ has standard deviation $1/2$. Figure 1 shows the two densities.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="Density of the latent z and of the data x when z = 2x: the data density is twice as tall and half as wide">
<line x1="70.0" y1="245.0" x2="530.0" y2="245.0" class="stroke-muted"/>
<line x1="70.0" y1="245.0" x2="70.0" y2="250.0" class="stroke-muted"/>
<text x="70.0" y="265.0" class="fig-small" text-anchor="middle">-3</text>
<line x1="146.7" y1="245.0" x2="146.7" y2="250.0" class="stroke-muted"/>
<text x="146.7" y="265.0" class="fig-small" text-anchor="middle">-2</text>
<line x1="223.3" y1="245.0" x2="223.3" y2="250.0" class="stroke-muted"/>
<text x="223.3" y="265.0" class="fig-small" text-anchor="middle">-1</text>
<line x1="300.0" y1="245.0" x2="300.0" y2="250.0" class="stroke-muted"/>
<text x="300.0" y="265.0" class="fig-small" text-anchor="middle">0</text>
<line x1="376.7" y1="245.0" x2="376.7" y2="250.0" class="stroke-muted"/>
<text x="376.7" y="265.0" class="fig-small" text-anchor="middle">1</text>
<line x1="453.3" y1="245.0" x2="453.3" y2="250.0" class="stroke-muted"/>
<text x="453.3" y="265.0" class="fig-small" text-anchor="middle">2</text>
<line x1="530.0" y1="245.0" x2="530.0" y2="250.0" class="stroke-muted"/>
<text x="530.0" y="265.0" class="fig-small" text-anchor="middle">3</text>
<line x1="70.0" y1="245.0" x2="530.0" y2="245.0" class="stroke-grid"/>
<text x="62.0" y="249.0" class="fig-small" text-anchor="end">0</text>
<line x1="70.0" y1="174.4" x2="530.0" y2="174.4" class="stroke-grid"/>
<text x="62.0" y="178.4" class="fig-small" text-anchor="end">0.4</text>
<line x1="70.0" y1="103.8" x2="530.0" y2="103.8" class="stroke-grid"/>
<text x="62.0" y="107.8" class="fig-small" text-anchor="end">0.8</text>
<polyline points="70.0,245.0 71.5,245.0 73.1,245.0 74.6,245.0 76.1,245.0 77.7,245.0 79.2,245.0 80.7,245.0 82.3,245.0 83.8,245.0 85.3,245.0 86.9,245.0 88.4,245.0 89.9,245.0 91.5,245.0 93.0,245.0 94.5,245.0 96.1,245.0 97.6,245.0 99.1,245.0 100.7,245.0 102.2,245.0 103.7,245.0 105.3,245.0 106.8,245.0 108.3,245.0 109.9,245.0 111.4,245.0 112.9,245.0 114.5,245.0 116.0,245.0 117.5,245.0 119.1,245.0 120.6,245.0 122.1,245.0 123.7,245.0 125.2,245.0 126.7,245.0 128.3,245.0 129.8,245.0 131.3,245.0 132.9,245.0 134.4,245.0 135.9,245.0 137.5,245.0 139.0,245.0 140.5,245.0 142.1,245.0 143.6,245.0 145.1,245.0 146.7,245.0 148.2,245.0 149.7,245.0 151.3,245.0 152.8,245.0 154.3,244.9 155.9,244.9 157.4,244.9 158.9,244.9 160.5,244.9 162.0,244.9 163.5,244.9 165.1,244.9 166.6,244.8 168.1,244.8 169.7,244.8 171.2,244.8 172.7,244.7 174.3,244.7 175.8,244.6 177.3,244.6 178.9,244.5 180.4,244.5 181.9,244.4 183.5,244.3 185.0,244.2 186.5,244.1 188.1,244.0 189.6,243.9 191.1,243.8 192.7,243.6 194.2,243.4 195.7,243.3 197.3,243.1 198.8,242.8 200.3,242.6 201.9,242.3 203.4,242.1 204.9,241.7 206.5,241.4 208.0,241.0 209.5,240.7 211.1,240.2 212.6,239.8 214.1,239.3 215.7,238.7 217.2,238.2 218.7,237.6 220.3,236.9 221.8,236.2 223.3,235.5 224.9,234.7 226.4,233.9 227.9,233.0 229.5,232.0 231.0,231.1 232.5,230.0 234.1,229.0 235.6,227.8 237.1,226.7 238.7,225.4 240.2,224.1 241.7,222.8 243.3,221.5 244.8,220.0 246.3,218.6 247.9,217.1 249.4,215.5 250.9,214.0 252.5,212.4 254.0,210.7 255.5,209.1 257.1,207.4 258.6,205.7 260.1,204.0 261.7,202.3 263.2,200.6 264.7,198.9 266.3,197.2 267.8,195.5 269.3,193.9 270.9,192.3 272.4,190.7 273.9,189.1 275.5,187.6 277.0,186.2 278.5,184.8 280.1,183.5 281.6,182.3 283.1,181.1 284.7,180.0 286.2,179.0 287.7,178.1 289.3,177.3 290.8,176.6 292.3,176.0 293.9,175.5 295.4,175.1 296.9,174.8 298.5,174.7 300.0,174.6 301.5,174.7 303.1,174.8 304.6,175.1 306.1,175.5 307.7,176.0 309.2,176.6 310.7,177.3 312.3,178.1 313.8,179.0 315.3,180.0 316.9,181.1 318.4,182.3 319.9,183.5 321.5,184.8 323.0,186.2 324.5,187.6 326.1,189.1 327.6,190.7 329.1,192.3 330.7,193.9 332.2,195.5 333.7,197.2 335.3,198.9 336.8,200.6 338.3,202.3 339.9,204.0 341.4,205.7 342.9,207.4 344.5,209.1 346.0,210.7 347.5,212.4 349.1,214.0 350.6,215.5 352.1,217.1 353.7,218.6 355.2,220.0 356.7,221.5 358.3,222.8 359.8,224.1 361.3,225.4 362.9,226.7 364.4,227.8 365.9,229.0 367.5,230.0 369.0,231.1 370.5,232.0 372.1,233.0 373.6,233.9 375.1,234.7 376.7,235.5 378.2,236.2 379.7,236.9 381.3,237.6 382.8,238.2 384.3,238.7 385.9,239.3 387.4,239.8 388.9,240.2 390.5,240.7 392.0,241.0 393.5,241.4 395.1,241.7 396.6,242.1 398.1,242.3 399.7,242.6 401.2,242.8 402.7,243.1 404.3,243.3 405.8,243.4 407.3,243.6 408.9,243.8 410.4,243.9 411.9,244.0 413.5,244.1 415.0,244.2 416.5,244.3 418.1,244.4 419.6,244.5 421.1,244.5 422.7,244.6 424.2,244.6 425.7,244.7 427.3,244.7 428.8,244.8 430.3,244.8 431.9,244.8 433.4,244.8 434.9,244.9 436.5,244.9 438.0,244.9 439.5,244.9 441.1,244.9 442.6,244.9 444.1,244.9 445.7,244.9 447.2,245.0 448.7,245.0 450.3,245.0 451.8,245.0 453.3,245.0 454.9,245.0 456.4,245.0 457.9,245.0 459.5,245.0 461.0,245.0 462.5,245.0 464.1,245.0 465.6,245.0 467.1,245.0 468.7,245.0 470.2,245.0 471.7,245.0 473.3,245.0 474.8,245.0 476.3,245.0 477.9,245.0 479.4,245.0 480.9,245.0 482.5,245.0 484.0,245.0 485.5,245.0 487.1,245.0 488.6,245.0 490.1,245.0 491.7,245.0 493.2,245.0 494.7,245.0 496.3,245.0 497.8,245.0 499.3,245.0 500.9,245.0 502.4,245.0 503.9,245.0 505.5,245.0 507.0,245.0 508.5,245.0 510.1,245.0 511.6,245.0 513.1,245.0 514.7,245.0 516.2,245.0 517.7,245.0 519.3,245.0 520.8,245.0 522.3,245.0 523.9,245.0 525.4,245.0 526.9,245.0 528.5,245.0 530.0,245.0" class="stroke-a" fill="none" stroke-width="2.5"/>
<polyline points="70.0,245.0 71.5,245.0 73.1,245.0 74.6,245.0 76.1,245.0 77.7,245.0 79.2,245.0 80.7,245.0 82.3,245.0 83.8,245.0 85.3,245.0 86.9,245.0 88.4,245.0 89.9,245.0 91.5,245.0 93.0,245.0 94.5,245.0 96.1,245.0 97.6,245.0 99.1,245.0 100.7,245.0 102.2,245.0 103.7,245.0 105.3,245.0 106.8,245.0 108.3,245.0 109.9,245.0 111.4,245.0 112.9,245.0 114.5,245.0 116.0,245.0 117.5,245.0 119.1,245.0 120.6,245.0 122.1,245.0 123.7,245.0 125.2,245.0 126.7,245.0 128.3,245.0 129.8,245.0 131.3,245.0 132.9,245.0 134.4,245.0 135.9,245.0 137.5,245.0 139.0,245.0 140.5,245.0 142.1,245.0 143.6,245.0 145.1,245.0 146.7,245.0 148.2,244.9 149.7,244.9 151.3,244.9 152.8,244.9 154.3,244.9 155.9,244.9 157.4,244.9 158.9,244.8 160.5,244.8 162.0,244.8 163.5,244.8 165.1,244.7 166.6,244.7 168.1,244.6 169.7,244.6 171.2,244.5 172.7,244.4 174.3,244.4 175.8,244.3 177.3,244.2 178.9,244.0 180.4,243.9 181.9,243.8 183.5,243.6 185.0,243.4 186.5,243.2 188.1,243.0 189.6,242.8 191.1,242.5 192.7,242.2 194.2,241.9 195.7,241.5 197.3,241.1 198.8,240.7 200.3,240.2 201.9,239.7 203.4,239.1 204.9,238.5 206.5,237.8 208.0,237.1 209.5,236.3 211.1,235.5 212.6,234.5 214.1,233.5 215.7,232.5 217.2,231.3 218.7,230.1 220.3,228.8 221.8,227.4 223.3,225.9 224.9,224.4 226.4,222.7 227.9,220.9 229.5,219.1 231.0,217.1 232.5,215.1 234.1,212.9 235.6,210.7 237.1,208.3 238.7,205.9 240.2,203.3 241.7,200.6 243.3,197.9 244.8,195.1 246.3,192.2 247.9,189.2 249.4,186.1 250.9,182.9 252.5,179.7 254.0,176.5 255.5,173.2 257.1,169.8 258.6,166.4 260.1,163.0 261.7,159.6 263.2,156.2 264.7,152.8 266.3,149.4 267.8,146.1 269.3,142.8 270.9,139.5 272.4,136.3 273.9,133.3 275.5,130.3 277.0,127.4 278.5,124.6 280.1,122.0 281.6,119.5 283.1,117.2 284.7,115.0 286.2,113.0 287.7,111.2 289.3,109.6 290.8,108.2 292.3,107.0 293.9,106.0 295.4,105.2 296.9,104.6 298.5,104.3 300.0,104.2 301.5,104.3 303.1,104.6 304.6,105.2 306.1,106.0 307.7,107.0 309.2,108.2 310.7,109.6 312.3,111.2 313.8,113.0 315.3,115.0 316.9,117.2 318.4,119.5 319.9,122.0 321.5,124.6 323.0,127.4 324.5,130.3 326.1,133.3 327.6,136.3 329.1,139.5 330.7,142.8 332.2,146.1 333.7,149.4 335.3,152.8 336.8,156.2 338.3,159.6 339.9,163.0 341.4,166.4 342.9,169.8 344.5,173.2 346.0,176.5 347.5,179.7 349.1,182.9 350.6,186.1 352.1,189.2 353.7,192.2 355.2,195.1 356.7,197.9 358.3,200.6 359.8,203.3 361.3,205.9 362.9,208.3 364.4,210.7 365.9,212.9 367.5,215.1 369.0,217.1 370.5,219.1 372.1,220.9 373.6,222.7 375.1,224.4 376.7,225.9 378.2,227.4 379.7,228.8 381.3,230.1 382.8,231.3 384.3,232.5 385.9,233.5 387.4,234.5 388.9,235.5 390.5,236.3 392.0,237.1 393.5,237.8 395.1,238.5 396.6,239.1 398.1,239.7 399.7,240.2 401.2,240.7 402.7,241.1 404.3,241.5 405.8,241.9 407.3,242.2 408.9,242.5 410.4,242.8 411.9,243.0 413.5,243.2 415.0,243.4 416.5,243.6 418.1,243.8 419.6,243.9 421.1,244.0 422.7,244.2 424.2,244.3 425.7,244.4 427.3,244.4 428.8,244.5 430.3,244.6 431.9,244.6 433.4,244.7 434.9,244.7 436.5,244.8 438.0,244.8 439.5,244.8 441.1,244.8 442.6,244.9 444.1,244.9 445.7,244.9 447.2,244.9 448.7,244.9 450.3,244.9 451.8,244.9 453.3,245.0 454.9,245.0 456.4,245.0 457.9,245.0 459.5,245.0 461.0,245.0 462.5,245.0 464.1,245.0 465.6,245.0 467.1,245.0 468.7,245.0 470.2,245.0 471.7,245.0 473.3,245.0 474.8,245.0 476.3,245.0 477.9,245.0 479.4,245.0 480.9,245.0 482.5,245.0 484.0,245.0 485.5,245.0 487.1,245.0 488.6,245.0 490.1,245.0 491.7,245.0 493.2,245.0 494.7,245.0 496.3,245.0 497.8,245.0 499.3,245.0 500.9,245.0 502.4,245.0 503.9,245.0 505.5,245.0 507.0,245.0 508.5,245.0 510.1,245.0 511.6,245.0 513.1,245.0 514.7,245.0 516.2,245.0 517.7,245.0 519.3,245.0 520.8,245.0 522.3,245.0 523.9,245.0 525.4,245.0 526.9,245.0 528.5,245.0 530.0,245.0" class="stroke-b" fill="none" stroke-width="2.5"/>
<circle cx="376.7" cy="235.5" r="5" class="fill-a"/>
<circle cx="376.7" cy="225.9" r="5" class="fill-b"/>
<text x="386.7" y="227.5" class="fig-small" text-anchor="start">p_Z(f(1)) = 0.0540</text>
<text x="386.7" y="213.9" class="fig-small" text-anchor="start">p_X(1) = 0.1080</text>
<line x1="80.0" y1="40.0" x2="120.0" y2="40.0" class="stroke-a" stroke-width="2.5"/>
<text x="128.0" y="44.0" class="fig-small" text-anchor="start">p_Z(f(x)): the latent's density at z = 2x</text>
<line x1="80.0" y1="64.0" x2="120.0" y2="64.0" class="stroke-b" stroke-width="2.5"/>
<text x="128.0" y="68.0" class="fig-small" text-anchor="start">p_X(x) = 2 p_Z(2x): add the log-determinant</text>
<text x="300.0" y="290.0" class="fig-small" text-anchor="middle">data value x</text>
</svg>
Figure 1. The map z = 2x halves the width of the density, so the data density is twice as tall. Notice that the extra factor of 2 at every x is exactly the log-determinant term, ln 2 = 0.6931, added in log space.
```

A flow stacks $K$ invertible layers, $z = f_K(\cdots f_1(x))$, and their log-determinants add. Sampling draws $z \sim \mathcal{N}(0, I)$ and runs the inverses in reverse order. The whole design problem is to build layers that are expressive, invertible, and have a log-determinant that is cheap to compute. A general $D \times D$ Jacobian costs $O(D^3)$ to take a determinant of, which is hopeless when $D$ is every sample of a clip.

## Affine coupling

The *affine coupling* layer of RealNVP and Glow solves that with a trick. Split the input into two halves, $x_a$ and $x_b$. Leave $x_a$ alone, and use it to scale and shift $x_b$:

$$
y_a = x_a, \qquad y_b = x_b \odot \exp\left(s(x_a)\right) + t(x_a), \qquad
\log \left| \det \frac{\partial y}{\partial x} \right| = \sum_i s(x_a)_i.
$$

Read it aloud: the first half passes through. The second half is multiplied, entry by entry, by the exponential of a scale $s$, and shifted by $t$. Both $s$ and $t$ are outputs of any network you like, fed only the first half. The Jacobian is triangular: $y_a$ doesn't depend on $x_b$, and each entry of $y_b$ depends on its own $x_b$ entry through the factor $e^{s_i}$. So the determinant is the product of those factors, and its log is the sum of $s$. In torch:

```python
x_a, x_b = x[:, :half], x[:, half:]
s, t = net(x_a)
y = torch.cat([x_a, x_b * torch.exp(s) + t], dim=1)
logdet = s.sum(dim=1)
```

The inverse is as cheap as the forward pass. Since $y_a = x_a$, you can recompute $s$ and $t$ from $y_a$ and undo the step: $x_b = (y_b - t(y_a)) \odot \exp(-s(y_a))$. The network itself never has to be inverted. Alternate which half is transformed from layer to layer, or the first half never changes.

Work one coupling by hand (Figure 2). Take $x = (1, 2)$, $s(x_a) = 0.5\,x_a$, and $t(x_a) = x_a$:

- $y_a = 1$, and $y_b = 2 e^{0.5} + 1 = 4.297443$.
- $\log |\det J| = s(1) = 0.5$.
- $\log \mathcal{N}(y; 0, I) = -\log 2\pi - (1^2 + 4.297443^2)/2 = -1.837877 - 9.734006 = -11.571883$.
- $\log p_X(x) = -11.571883 + 0.5 = -11.071883$.
- The inverse gives back $x_b = (4.297443 - 1)\,e^{-0.5} = 2$.

```figure
<svg viewBox="0 0 600 300" role="img" aria-label="An affine coupling layer on x = (1, 2): the first half passes through and sets the scale and shift of the second half, giving y = (1, 4.2974)">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<rect x="20.0" y="40.0" width="90.0" height="44.0" class="fill-soft-a" rx="6"/>
<rect x="20.0" y="40.0" width="90.0" height="44.0" class="stroke-a" fill="none" rx="6"/>
<text x="65.0" y="68.0" text-anchor="middle">x_a = 1</text>
<rect x="20.0" y="190.0" width="90.0" height="44.0" class="fill-soft-a" rx="6"/>
<rect x="20.0" y="190.0" width="90.0" height="44.0" class="stroke-a" fill="none" rx="6"/>
<text x="65.0" y="218.0" text-anchor="middle">x_b = 2</text>
<line x1="110.0" y1="62.0" x2="470.0" y2="62.0" class="stroke-a" marker-end="url(#c042-arrow)"/>
<text x="290.0" y="52.0" class="fig-small" text-anchor="middle">y_a = x_a = 1</text>
<rect x="190.0" y="110.0" width="130.0" height="50.0" class="fill-soft-b" rx="6"/>
<rect x="190.0" y="110.0" width="130.0" height="50.0" class="stroke-b" fill="none" rx="6"/>
<text x="255.0" y="132.0" text-anchor="middle">network s, t</text>
<text x="255.0" y="150.0" class="fig-small" text-anchor="middle">s = 0.5 x_a, t = x_a</text>
<line x1="150.0" y1="62.0" x2="150.0" y2="135.0" class="stroke-a"/>
<line x1="150.0" y1="135.0" x2="190.0" y2="135.0" class="stroke-a" marker-end="url(#c042-arrow)"/>
<line x1="255.0" y1="160.0" x2="255.0" y2="200.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<text x="262.0" y="182.0" class="fig-small" text-anchor="start">s = 0.5, t = 1</text>
<rect x="210.0" y="190.0" width="90.0" height="44.0" class="fill-soft-c" rx="6"/>
<rect x="210.0" y="190.0" width="90.0" height="44.0" class="stroke-c" fill="none" rx="6"/>
<text x="255.0" y="218.0" class="fig-small" text-anchor="middle">x_b e^s + t</text>
<line x1="110.0" y1="212.0" x2="210.0" y2="212.0" class="stroke-a" marker-end="url(#c042-arrow)"/>
<line x1="300.0" y1="212.0" x2="470.0" y2="212.0" class="stroke-b" marker-end="url(#c042-arrow)"/>
<text x="385.0" y="202.0" class="fig-small" text-anchor="middle">y_b = 2 e^0.5 + 1 = 4.2974</text>
<rect x="470.0" y="40.0" width="110.0" height="44.0" class="fill-soft-a" rx="6"/>
<rect x="470.0" y="40.0" width="110.0" height="44.0" class="stroke-a" fill="none" rx="6"/>
<text x="525.0" y="68.0" text-anchor="middle">y_a = 1</text>
<rect x="470.0" y="190.0" width="110.0" height="44.0" class="fill-soft-b" rx="6"/>
<rect x="470.0" y="190.0" width="110.0" height="44.0" class="stroke-b" fill="none" rx="6"/>
<text x="525.0" y="218.0" text-anchor="middle">y_b = 4.2974</text>
<text x="20.0" y="275.0" class="fig-small" text-anchor="start">log |det J| = s = 0.5.  Inverse: x_b = (y_b - t) e^(-s) = (4.2974 - 1) e^(-0.5) = 2.</text>
</svg>
Figure 2. A coupling layer for x = (1, 2). Notice that the network only ever reads the half that passes through, so the layer is easy to invert and its log-determinant is the sum of the scales.
```

> [!KEY]
> A coupling layer's Jacobian is triangular, so its log-determinant is the sum of the predicted log-scales, and its inverse reuses the same network on the unchanged half.

## ActNorm and the invertible 1x1 convolution

Glow adds two more layers between couplings, both acting on a sequence $x \in \mathbb{R}^{C \times T}$ of $C$ channels and $T$ time steps. *ActNorm* is a per-channel scale and shift, $y_{ct} = e^{\ell_c} x_{ct} + b_c$. Its scale and bias are initialized from the first batch so every channel's output has zero mean and unit variance, then trained freely. The *invertible 1x1 convolution* mixes channels with one learned $C \times C$ matrix, $y_t = W x_t$ at every time step. It generalizes the fixed swap of halves between couplings.

$$
\log \left| \det J_{\text{ActNorm}} \right| = T \sum_{c=1}^{C} \ell_c, \qquad
\log \left| \det J_{1 \times 1} \right| = T \log \left| \det W \right|.
$$

Read it aloud: both layers apply the same small map at each of $T$ time steps, so the whole Jacobian is block diagonal with $T$ identical blocks, and the log-determinant is $T$ times one block's. With $T = 4$ and log-scales $[0.3, -0.2, 0.5]$, ActNorm's log-determinant is $4 \times 0.6 = 2.4$. A $W$ with $|\det W| = 2.1$ gives $4 \ln 2.1 = 2.968$. In torch, `torch.linalg.slogdet(W).logabsdet` gives $\log|\det W|$ without overflow.

> [!WARNING]
> Forgetting the factor $T$ is a common flow bug. The loss still falls, but the likelihood is wrong, and the model learns to shrink or inflate channels for free.

## Train by maximum likelihood

With exact log-densities, training is ordinary maximum likelihood: minimize the mean of $-\log p_X(x)$ over the data. There is no discriminator and no bound. Report the loss in nats per dimension, so clips of different lengths compare. Sampling runs the flow backward from Gaussian noise.

WaveGlow did exactly that on raw waveforms. It stacked 12 couplings and 12 invertible 1x1 convolutions, with WaveNet-like networks predicting $s$ and $t$ from the mel spectrogram. It needed about 88 million parameters, and in the HiFi-GAN paper's listening test a GAN vocoder with under 1 million parameters beat it. A flow can't throw information away, so every layer runs at the data's full size. GAN vocoders replaced flows for waveforms, but flows survived inside TTS models, where they model a smaller latent space instead of raw audio.

## VITS

VITS, from 2021, is a *conditional variational autoencoder* for speech (Figure 3). It has four parts:

- **The posterior encoder** reads the real linear spectrogram and outputs a Gaussian $q(z \mid x)$ per frame, with mean $m_q$ and log standard deviation $\log\sigma_q$. It exists only in training.
- **The prior** comes from text. A Transformer text encoder gives each phone a Gaussian $(m_p, \log\sigma_p)$, and monotonic alignment search expands the phones to frames. A flow $f$ of coupling layers maps the posterior's $z$ to $z_p = f(z)$, where it is scored against that Gaussian. The flow lets a simple per-phone Gaussian describe a complicated distribution of frames.
- **The decoder** is a HiFi-GAN V1 generator that turns $z$ into a waveform, trained on random windows of frames to save memory.
- **A stochastic duration predictor**, itself a flow, learns a distribution of durations so rhythm can vary.

```figure
<svg viewBox="0 0 600 330" role="img" aria-label="VITS during training and at inference: training encodes real audio and aligns it with text; inference samples from the text prior, runs the flow backward, and decodes">
<defs><marker id="c042-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" class="fill-ink"/></marker></defs>
<text x="20.0" y="28.0" class="fig-bold" text-anchor="start">Training</text>
<rect x="20.0" y="40.0" width="100.0" height="44.0" class="fill-soft-a" rx="6"/>
<rect x="20.0" y="40.0" width="100.0" height="44.0" class="stroke-a" fill="none" rx="6"/>
<text x="70.0" y="60.0" class="fig-small" text-anchor="middle">audio</text>
<text x="70.0" y="76.0" class="fig-small" text-anchor="middle">spectrogram</text>
<rect x="160.0" y="40.0" width="100.0" height="44.0" class="fill-soft-a" rx="6"/>
<rect x="160.0" y="40.0" width="100.0" height="44.0" class="stroke-a" fill="none" rx="6"/>
<text x="210.0" y="60.0" class="fig-small" text-anchor="middle">posterior</text>
<text x="210.0" y="76.0" class="fig-small" text-anchor="middle">encoder q(z|x)</text>
<rect x="300.0" y="40.0" width="70.0" height="44.0" class="fill-sunk" rx="6"/>
<rect x="300.0" y="40.0" width="70.0" height="44.0" class="stroke-muted" fill="none" rx="6"/>
<text x="335.0" y="67.0" class="fig-small" text-anchor="middle">z</text>
<rect x="410.0" y="40.0" width="80.0" height="44.0" class="fill-soft-c" rx="6"/>
<rect x="410.0" y="40.0" width="80.0" height="44.0" class="stroke-c" fill="none" rx="6"/>
<text x="450.0" y="60.0" class="fig-small" text-anchor="middle">decoder</text>
<text x="450.0" y="76.0" class="fig-small" text-anchor="middle">HiFi-GAN</text>
<rect x="520.0" y="40.0" width="70.0" height="44.0" class="fill-soft-c" rx="6"/>
<rect x="520.0" y="40.0" width="70.0" height="44.0" class="stroke-c" fill="none" rx="6"/>
<text x="555.0" y="67.0" class="fig-small" text-anchor="middle">audio</text>
<line x1="120.0" y1="62.0" x2="160.0" y2="62.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="260.0" y1="62.0" x2="300.0" y2="62.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="370.0" y1="62.0" x2="410.0" y2="62.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="490.0" y1="62.0" x2="520.0" y2="62.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<rect x="300.0" y="120.0" width="70.0" height="44.0" class="fill-soft-b" rx="6"/>
<rect x="300.0" y="120.0" width="70.0" height="44.0" class="stroke-b" fill="none" rx="6"/>
<text x="335.0" y="147.0" class="fig-small" text-anchor="middle">flow f</text>
<rect x="140.0" y="120.0" width="130.0" height="44.0" class="fill-soft-d" rx="6"/>
<rect x="140.0" y="120.0" width="130.0" height="44.0" class="stroke-d" fill="none" rx="6"/>
<text x="205.0" y="140.0" class="fig-small" text-anchor="middle">text prior</text>
<text x="205.0" y="156.0" class="fig-small" text-anchor="middle">(m_p, log s_p)</text>
<line x1="335.0" y1="84.0" x2="335.0" y2="120.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="300.0" y1="142.0" x2="270.0" y2="142.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<text x="400.0" y="138.0" class="fig-small" text-anchor="start">KL term: z_p = f(z)</text>
<text x="400.0" y="156.0" class="fig-small" text-anchor="start">against the text prior</text>
<line x1="20.0" y1="190.0" x2="580.0" y2="190.0" class="stroke-grid"/>
<text x="20.0" y="218.0" class="fig-bold" text-anchor="start">Inference</text>
<rect x="20.0" y="232.0" width="110.0" height="44.0" class="fill-soft-d" rx="6"/>
<rect x="20.0" y="232.0" width="110.0" height="44.0" class="stroke-d" fill="none" rx="6"/>
<text x="75.0" y="252.0" class="fig-small" text-anchor="middle">text prior</text>
<text x="75.0" y="268.0" class="fig-small" text-anchor="middle">sample z_p</text>
<rect x="180.0" y="232.0" width="100.0" height="44.0" class="fill-soft-b" rx="6"/>
<rect x="180.0" y="232.0" width="100.0" height="44.0" class="stroke-b" fill="none" rx="6"/>
<text x="230.0" y="252.0" class="fig-small" text-anchor="middle">flow f</text>
<text x="230.0" y="268.0" class="fig-small" text-anchor="middle">run backward</text>
<rect x="330.0" y="232.0" width="70.0" height="44.0" class="fill-sunk" rx="6"/>
<rect x="330.0" y="232.0" width="70.0" height="44.0" class="stroke-muted" fill="none" rx="6"/>
<text x="365.0" y="259.0" class="fig-small" text-anchor="middle">z</text>
<rect x="440.0" y="232.0" width="100.0" height="44.0" class="fill-soft-c" rx="6"/>
<rect x="440.0" y="232.0" width="100.0" height="44.0" class="stroke-c" fill="none" rx="6"/>
<text x="490.0" y="252.0" class="fig-small" text-anchor="middle">decoder</text>
<text x="490.0" y="268.0" class="fig-small" text-anchor="middle">HiFi-GAN</text>
<line x1="130.0" y1="254.0" x2="180.0" y2="254.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="280.0" y1="254.0" x2="330.0" y2="254.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<line x1="400.0" y1="254.0" x2="440.0" y2="254.0" class="stroke-ink" marker-end="url(#c042-arrow)"/>
<text x="20.0" y="312.0" class="fig-small" text-anchor="start">No audio goes in at inference: the text prior replaces the posterior encoder.</text>
</svg>
Figure 3. VITS has a training path that uses real audio and an inference path that doesn't. Notice that the flow connects the two: it maps the posterior's z to the space where the text prior lives.
```

At inference there is no audio. VITS samples $z_p$ from the text prior, runs the flow backward to get $z$, and decodes it, all in one parallel pass. The paper reported a MOS of 4.43 on LJSpeech against 4.46 for recordings. Its training loss adds a mel L1 term, the KL term, the duration loss, and the HiFi-GAN adversarial and feature-matching losses. The official configuration weights the mel term by 45 and the KL term by 1.

### The KL term

The KL term pulls the posterior toward the prior. With one sample $z = m_q + \sigma_q \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$, it is $\log q(z \mid x) - \log p(z \mid \text{text})$. The VITS flow is additive ($s = 0$), so its log-determinant is 0. Write out both Gaussians, cancel the $\log 2\pi$ terms, and replace $\epsilon^2$ by its expected value 1:

$$
\mathcal{L}_{\text{kl}} = \frac{1}{|\mathcal{M}|} \sum_{(c,t) \in \mathcal{M}}
\left[ \log\sigma_p - \log\sigma_q - \frac12 + \frac{(z_p - m_p)^2}{2\sigma_p^2} \right].
$$

Read it aloud: at every real frame and channel, the term is the prior's log standard deviation minus the posterior's, minus one half, plus the squared distance of the flowed sample from the prior mean in units of the prior's variance. The $-\tfrac12$ is $\log q$'s $-\epsilon^2/2$, averaged. The sum runs over the mask $\mathcal{M}$ of real entries and divides by its size. For one entry with $z_p = 0.5$, $\log\sigma_q = -1$, $m_p = 0$, and $\log\sigma_p = 0$, the term is $0 + 1 - 0.5 + 0.125 = 0.625$. In torch:

```python
kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
loss = (kl * mask).sum() / mask.sum()
```

> [!KEY]
> A flow gives an exact likelihood and one-pass sampling because every layer is invertible with a cheap log-determinant. VITS uses that to compare a posterior sample, pushed through the flow, with the text prior's Gaussian, so a per-phone Gaussian can explain frames that are far from Gaussian.

> [!RECAP]
> - The change of variables adds the log-determinant of the data-to-latent map.
> - A coupling layer's log-determinant is $\sum s(x_a)$, and its inverse is $x_b = (y_b - t)\,e^{-s}$.
> - ActNorm's log-determinant is $T \sum_c \ell_c$, and a 1x1 convolution's is $T \log|\det W|$.
> - Train by minimizing $-\log p_X$, and sample by inverting the layers in reverse order.
> - The VITS KL term is $\log\sigma_p - \log\sigma_q - \tfrac12 + (z_p - m_p)^2 / 2\sigma_p^2$, averaged over real entries.

## Build it step by step

Build the pieces in the order the checks test them.

1. **`coupling_forward(x, net, flip)`** splits `x` into halves, swaps the roles when `flip` is true, and returns `(y, logdet)`. The common mistake is concatenating the halves in the wrong order when `flip` is true.
2. **`coupling_inverse(y, net, flip)`** recomputes `(s, t)` from the unchanged half and undoes the step.
3. **`actnorm_init(x)`** and **`actnorm_forward(x, log_scale, bias)`** take the mean and biased standard deviation over dims `(0, 2)`, and return a log-determinant of shape `(B,)`. The common mistake is omitting the factor $T$.
4. **`invconv_forward(x, W)`** and **`invconv_inverse(y, W)`** use `torch.einsum("ij,bjt->bit", ...)` and `torch.linalg.slogdet`.
5. **`RealNVP.forward`**, **`RealNVP.log_prob`**, and **`RealNVP.sample`** loop over `self.nets` with `flip = i % 2 == 1` and sum the log-determinants. `sample` draws `torch.randn` with a seeded `torch.Generator` and runs the inverses in reverse. The common mistake is running the inverses in forward order.
6. **`train_flow(data, steps, lr, batch, seed)`** runs Adam on the mean of `-log_prob` over random batches.
7. **`vits_kl(z_p, logs_q, m_p, logs_p, mask)`** is two lines. The common mistake is dividing by the number of frames instead of the number of real entries.

## Where this breaks

**Direction mix-ups.** Papers write the flow in both directions. The log-determinant you add is always that of the data-to-latent map. Adding the other one's, or subtracting, gives a model that trains to nonsense.

**Mass between modes.** A flow is a smooth bijection, so it can't tear Gaussian noise into separate islands. It thins the bridges between modes but leaves some mass there. The Run shows it: some sampled notes land between the notes of the scale.

**Discrete data.** Audio samples and durations are integers. A continuous density can pile up on a single point, so the likelihood grows without bound. Add uniform noise first, which is called *dequantization*, as the VITS duration predictor does.

**Unstable scales.** An unbounded $s$ can reach $e^{50}$ in a few steps. Bound it with a tanh, as the exercise's networks do, or start couplings at the identity by initializing the last layer to zero.

## The exercise

You implement `coupling_forward`, `coupling_inverse`, `actnorm_init`, `actnorm_forward`, `invconv_forward`, `invconv_inverse`, the `RealNVP` methods `forward`, `log_prob`, and `sample`, `train_flow`, and `vits_kl`. The checks reproduce the hand-worked $-11.071883$, compare every log-determinant with `slogdet` of an autograd Jacobian, confirm each inverse, and integrate your density over a grid to 1. They train your flow on two moons and require it to beat the best single Gaussian by 0.4 nats, and compare your KL term, averaged over 200,000 posterior samples, with the closed-form Gaussian KL.

Press **Run** to train an 8-layer RealNVP for about 10 seconds on 4,000 notes of a pentatonic tune, each a pitch and a log length. The run prints the held-out likelihood against a single Gaussian, draws the learned density, and plays a 16-note melody sampled from your flow.
