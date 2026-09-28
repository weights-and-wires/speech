---
title: Enhancement and separation
slug: 39-enhancement-and-separation
part: "Part 8 — Beyond recognition and synthesis"
summary: Clean noise out of speech with subtraction, Wiener gains, and masks, then pull two voices apart with a Conv-TasNet and PIT.
minutes: 75
gpu: false
objectives:
  - Mix speech and noise at an exact SNR, and enhance the mixture with spectral subtraction and a decision-directed Wiener filter.
  - Compute the ideal binary and ratio masks from powers, and explain why they are upper bounds rather than methods.
  - Score an estimate with SI-SDR, and train a separator with the permutation-invariant loss.
  - Build Conv-TasNet's encoder, dilated TCN blocks, masks, and decoder, and compute the TCN's receptive field.
prereqs:
  - 04-the-short-time-fourier-transform
  - 31-neural-audio-codecs
exercise: 39-enhancement-and-separation
---

# Enhancement and separation

Most real speech arrives with something else on top of it: a train, a fan,
a room full of people. *Speech enhancement* removes the something else and keeps the
voice. *Source separation* goes further. Two people talk at once, and you
want each voice on its own track.

The two problems share one idea. Look at the sound as a spectrogram, decide
how much of each time-frequency cell belongs to the voice you want, and turn
the rest down. Classical enhancement makes that decision from a noise
estimate and a formula. Neural separation learns it from examples, and it
has to solve a puzzle along the way: when a network outputs two voices, which
output is which voice?

You build the classical enhancers, the oracle masks that cap them, and a
small Conv-TasNet trained with the loss that solves the puzzle.

## What you already know

- The STFT turns a signal into complex cells $X(k, t)$, one per frequency bin $k$ and frame $t$, and the inverse STFT turns them back.
- Power is the squared magnitude $|X|^2$, and a decibel is $10 \log_{10}$ of a power ratio.
- SI-SDR from the lesson on neural audio codecs scores an estimate against a reference, ignoring its gain.
- A 1-D convolution with dilation $d$ skips $d - 1$ samples between its taps.
- PyTorch's autograd gives the gradient of any loss you can write with tensor operations.

## Noisy speech in the STFT domain

Noise adds to speech in the air, so it adds in the samples, and because the
STFT is linear, it adds in every cell too:

$$
Y(k, t) = S(k, t) + N(k, t)
$$

Read it aloud: the noisy cell $Y$ is the speech cell $S$ plus the noise cell
$N$. Enhancement estimates $S$ from $Y$ alone. To test an enhancer, build
mixtures at a known *signal-to-noise ratio* (SNR), the speech power over the
noise power in decibels, by scaling the noise by a gain $g$:

$$
y = s + g\,n, \qquad g = \sqrt{\frac{P_s}{P_n \cdot 10^{\text{SNR}/10}}}, \qquad P_x = \frac{1}{T}\sum_t x_t^2
$$

Read it aloud: $P_s$ and $P_n$ are the mean squares of the speech and the
noise over the clip. Dividing by $10^{\text{SNR}/10}$ turns the SNR from
decibels back into a power ratio, and the square root turns a power gain into
an amplitude gain. For speech with $P_s = 0.04$, noise with $P_n = 1$, and a
target of 5 dB, the gain is $\sqrt{0.04 / 3.162} = 0.1125$.

```python
gain = np.sqrt(np.mean(s**2) / (np.mean(n**2) * 10 ** (snr_db / 10)))
```

## Spectral subtraction

The oldest trick assumes the noise is steady. Measure its power in each bin
while nobody talks, then subtract that power from every frame. Boll published
it in 1979:

$$
|\hat S(k, t)|^2 = \max\left(|Y(k, t)|^2 - \alpha \hat P_N(k),\ \beta\, |Y(k, t)|^2\right), \qquad \hat S = |\hat S|\, e^{j \angle Y}
$$

Read it aloud: take the noisy power, subtract $\alpha$ times the estimated
noise power $\hat P_N(k)$, and never go below a floor of $\beta$ times the
noisy power. Keep the noisy phase $\angle Y$, because nothing better is
available. The *over-subtraction factor* $\alpha$, often 2, removes more
than the average noise. The *floor* $\beta$, often 0.01, keeps the power
positive.

For one cell with $|Y|^2 = 25$, $\hat P_N = 4$, $\alpha = 2$, and
$\beta = 0.01$, the estimate is $\max(25 - 8, 0.25) = 17$, so the magnitude
drops from 5 to 4.12.

```python
power = np.abs(Y) ** 2
clean = np.maximum(power - alpha * noise_power[:, None], beta * power)
S_hat = np.sqrt(clean) * np.exp(1j * np.angle(Y))
```

Noise power is an average, and each frame of noise fluctuates around it.
Where a frame exceeds the average, a lone peak survives. Peaks that come and
go frame by frame sound like tinkling tones, called *musical noise*. The
floor and over-subtraction hide it without removing it.

## The Wiener gain

A better enhancer multiplies each cell by a gain $H$ between 0 and 1. The
*Wiener filter* picks the $H$ that minimizes $\mathbb{E}|S - H Y|^2$. For
uncorrelated speech and noise, setting the derivative to zero gives

$$
H(k) = \frac{P_S(k)}{P_S(k) + P_N(k)} = \frac{\xi(k)}{1 + \xi(k)}, \qquad \xi(k) = \frac{P_S(k)}{P_N(k)}
$$

Read it aloud: the gain is the speech's share of the power in that bin,
written with the *a priori SNR* $\xi$, the speech-to-noise power ratio. At
$\xi = 3$ the gain is 0.75. Bins where speech dominates pass almost
untouched, and bins of pure noise go to zero.

The formula needs $\xi$, which depends on the clean speech. The measurable
*a posteriori SNR* $\gamma = |Y|^2 / \hat P_N$ gives the estimate
$\gamma - 1$, but it jumps from frame to frame and brings back musical noise. Ephraim and Malah's *decision-directed*
estimate (1984) blends it with the previous frame's cleaned output:

$$
\hat\xi(k, t) = a\,\frac{|\hat S(k, t - 1)|^2}{\hat P_N(k)} + (1 - a) \max\left(\gamma(k, t) - 1,\ 0\right)
$$

Read it aloud: 98 percent of the estimate, with $a = 0.98$, comes from the
SNR of what you kept in the previous frame, and 2 percent from what this
frame measures. A floor $\xi_{\min}$, usually $-25$ dB, keeps the gain from
reaching exactly zero. Each frame then sets $G = \hat\xi / (1 + \hat\xi)$ and
$\hat S = G Y$, and the next frame uses that $\hat S$.

```python
for t in range(n_frames):
    gamma = np.abs(Y[:, t]) ** 2 / noise_power
    xi = np.maximum(a * prev / noise_power + (1 - a) * np.maximum(gamma - 1, 0), xi_min)
    gains[:, t] = xi / (1 + xi)
    prev = gains[:, t] ** 2 * np.abs(Y[:, t]) ** 2
```

### Work the recursion by hand

Follow one bin with $\hat P_N = 1$, $a = 0.98$, and $\xi_{\min} = 0.00316$,
starting from $|\hat S|^2 = 0$. The noisy powers are 5 and 2 while only noise
plays, then 30 and 30 when a vowel starts.

| Frame | $\lvert Y \rvert^2$ | $\hat\xi$ | $G$ | $\lvert \hat S \rvert^2 = G^2 \lvert Y \rvert^2$ |
|---|---|---|---|---|
| 0 | 5 | $0.02 \times 4 = 0.080$ | 0.074 | 0.027 |
| 1 | 2 | $0.98 \times 0.027 + 0.02 \times 1 = 0.047$ | 0.045 | 0.004 |
| 2 | 30 | $0.98 \times 0.004 + 0.02 \times 29 = 0.584$ | 0.369 | 4.08 |
| 3 | 30 | $0.98 \times 4.08 + 0.02 \times 29 = 4.576$ | 0.821 | 20.2 |

The noise frames stay suppressed although frame 0 measured $\gamma = 5$. The
price is a lag: the gain needs two frames to open for the vowel.

> [!KEY]
> The Wiener gain $\xi / (1 + \xi)$ is optimal for the true a priori SNR. Everything practical about it is in how you estimate $\xi$, and the decision-directed rule trades a little attack time for far less musical noise.

## Ideal masks

Suppose an oracle told you $S$ and $N$ for every cell. The best gain is then
exact, and it shows how far any real method could go. Supervised enhancers
train networks to predict these *ideal masks*.

The *ideal binary mask* keeps a cell when speech is louder than the noise by
a local criterion LC, often 0 dB:

$$
\text{IBM}(k, t) = \mathbb{1}\left[ 10 \log_{10} \frac{|S|^2}{|N|^2} > \text{LC} \right]
$$

The *ideal ratio mask* is soft:

$$
\text{IRM}(k, t) = \left( \frac{|S|^2}{|S|^2 + |N|^2} \right)^{\beta}
$$

Read them aloud: the IBM is 1 where the cell's local SNR beats the
criterion and 0 elsewhere. The IRM is the speech's share of the cell's
power, raised to $\beta$, typically 0.5. With $\beta = 1$, it is the Wiener
gain computed from the true powers of that one cell. For a cell with
$|S|^2 = 4$ and $|N|^2 = 1$, the local SNR is 6.02 dB, so the IBM is 1 and
the IRM with $\beta = 0.5$ is $\sqrt{0.8} = 0.894$.

```python
ibm = (10 * np.log10(ps / pn) > lc_db).astype(float)
irm = (ps / (ps + pn)) ** beta
```

On the Run's vowels at 5 dB SNR, spectral subtraction scores 12.5 dB SI-SDR,
the decision-directed Wiener filter 15.9 dB, and the IBM and IRM oracles
18.6 dB and 18.1 dB. The oracles keep the noisy phase, so they stop short of
perfect.

> [!WARNING]
> Compute masks from powers, never from dB spectrograms, and never report an oracle mask as if it were a method. An oracle needs the clean signal, so its score is a ceiling, not a result.

## Score it with SI-SDR

You met SI-SDR in the lesson on neural audio codecs. Remove the mean from
the estimate $\hat s$ and the reference $s$, find the scaled copy of the
reference that best explains the estimate, and compare the energies:

$$
s_{\text{target}} = \frac{\langle \hat s, s \rangle}{\lVert s \rVert^2}\, s, \qquad
\text{SI-SDR}(\hat s, s) = 10 \log_{10} \frac{\lVert s_{\text{target}} \rVert^2}{\lVert \hat s - s_{\text{target}} \rVert^2}
$$

Read it aloud: the target is the reference scaled by the projection
coefficient, and everything else in the estimate is distortion. Separation
papers report the *improvement*, SI-SDRi: the estimate's SI-SDR minus the
mixture's. Conv-TasNet calls the same quantity SI-SNR.

For $s = (1, -1, 2, -2)$ and $\hat s = (2, -1, 3, -4)$, both zero-mean, the
coefficient is $17 / 10 = 1.7$. The target is $(1.7, -1.7, 3.4, -3.4)$ with
energy 28.9, the error is $(0.3, 0.7, -0.4, -0.6)$ with energy 1.1, and the
SI-SDR is $10 \log_{10} 26.27 = 14.19$ dB.

Here you write it in torch, batched over the last axis, so that
$-\text{SI-SDR}$ can be a training loss:

```python
scale = (est * ref).sum(-1, keepdim=True) / (ref * ref).sum(-1, keepdim=True)
target = scale * ref
si_sdr = 10 * torch.log10((target**2).sum(-1) / ((est - target) ** 2).sum(-1))
```

## Separation and the permutation problem

A separator takes one mixture and outputs $C$ signals, one per source.
Trained against references in a fixed order, it fails: nothing about two
voices says which one is "source 1". A network can split them perfectly and
still score badly because it put them in the other order.

*Permutation-invariant training* (PIT) scores every assignment of outputs to
references and keeps the best:

$$
\mathcal{L}_{\text{uPIT}} = \min_{\pi \in \mathcal{S}_C} \frac{1}{C} \sum_{c=1}^{C} -\text{SI-SDR}\left(\hat s_c,\ s_{\pi(c)}\right)
$$

Read it aloud: for each permutation $\pi$ of the $C$ references, average
the negative SI-SDR of output $c$ against reference $\pi(c)$. The loss is
the smallest of those averages. The "u" is for *utterance-level*: one
permutation covers the whole clip. Choosing a permutation per frame lets the
voices swap tracks in the middle of a sentence.

Compute the $C \times C$ matrix of pairwise losses once, then score each
permutation by indexing it. With two outputs and a loss matrix
$L = \begin{bmatrix} -3 & -12 \\ -10 & -2 \end{bmatrix}$, where $L_{ij}$ is
output $i$ against reference $j$, keeping the order costs
$(-3 - 2) / 2 = -2.5$ and swapping costs $(-12 - 10) / 2 = -11$. PIT keeps
the swap. Output 0 is reference 1, at 12 dB.

```python
pair = -si_sdr(est[:, :, None, :], ref[:, None, :, :])  # (B, C, C)
cost = pair[:, torch.arange(C), perms].mean(-1)          # (B, number of permutations)
loss = cost.min(dim=1).values.mean()
```

There are $C!$ permutations, which is fine for 2 or 3 sources. Past 4, the
Hungarian algorithm in `scipy.optimize.linear_sum_assignment` finds the same
minimum in polynomial time.

The following demo separates a chirp from a vowel with an oracle mask and scores the outputs in both orders.

```demo
{"id": "separation-pit", "caption": "Press Shuffle the outputs and watch the fixed-order loss jump while the PIT loss stays put, then lower the chirp level to see where the oracle masks start to fail."}
```

> [!KEY]
> A separator's outputs have no natural order. PIT takes the minimum loss over all orders, so the network is judged on how well it separates, not on which track it picked.

## Conv-TasNet

Luo and Mesgarani's Conv-TasNet (2019) drops the STFT and learns its own
filterbank. It has three parts:

- **Encoder.** A 1-D convolution with $N$ filters of length $L$ and stride $L / 2$, followed by ReLU, turns the waveform into a nonnegative representation $w$ with $N$ channels and $T'$ frames. The paper's best setting is $N = 512$ filters of $L = 16$ samples, 2 ms at 8 kHz.
- **Separator.** A temporal convolutional network (TCN) reads $w$ and outputs one mask $m_c \in [0, 1]^{N \times T'}$ per source, through a sigmoid.
- **Decoder.** A transposed convolution with the same length and stride overlap-adds a learned basis: $\hat s_c = \text{Decoder}(w \odot m_c)$.

Each separator block applies a 1x1 convolution, PReLU, and a normalization,
then a *depthwise* convolution, which filters each channel on its own, with
kernel $P$ and dilation $d$. Another PReLU, normalization, and 1x1
convolution follow, and the result is added to the block's input. The
dilations double, $1, 2, 4, \dots, 2^{X - 1}$, and the stack repeats $R$
times.

Each block with dilation $d$ widens the view by $(P - 1) d$ frames, so the
*receptive field*, the number of input frames that can affect one output
frame, is

$$
\text{RF} = 1 + R\,(P - 1) \sum_{i=0}^{X - 1} 2^i = 1 + R\,(P - 1)(2^X - 1)
$$

Read it aloud: one frame, plus $P - 1$ times the sum of the dilations,
$2^X - 1$, for each of the $R$ repeats. With the paper's $R = 3$, $X = 8$, and $P = 3$, that is $1 + 3 \cdot 2 \cdot 255 =
1531$ frames. At a hop of 8 samples, 1 ms at 8 kHz, the network sees about
1.5 seconds of context.

The exercise's model is smaller: $N = 64$, $L = 16$, $R = 2$, $X = 5$, and
$P = 3$, with 55,541 parameters. Its receptive field is
$1 + 2 \cdot 2 \cdot 31 = 125$ frames, which spans
$124 \times 8 + 16 = 1008$ samples, 126 ms at 8 kHz.

The paper's global layer normalization averages over the whole utterance.
The exercise normalizes each frame over its channels instead, so a change at
one frame reaches only the frames in the receptive field, which a check
measures.

The encoder only covers whole frames, so pad the input on the right before
encoding and trim the decoder's output back to the input length:

```python
frames = -(-(n - L) // stride) + 1           # ceiling division
pad = L + stride * (frames - 1) - n
w = F.relu(encoder(F.pad(x, (0, pad))[:, None]))
```

The Run trains this model with PIT for 300 steps, about 20 seconds on a
laptop CPU, on mixtures of a synthetic vowel train and a random chirp.
Held-out mixtures go from 0 dB SI-SDR to about 14 dB.

Later separators keep the masks and PIT and change the middle. DPRNN and
SepFormer split the long frame sequence into chunks and alternate models
inside and across chunks. TF-GridNet (2023) works on the complex STFT and
reports 23.5 dB SI-SDRi on the WSJ0-2mix benchmark. For music, HT Demucs
(2022) joins a waveform U-Net and a spectrogram branch with a Transformer.
Real-time enhancers such as RNNoise and DeepFilterNet predict gains for a few
dozen frequency bands, a direct descendant of the Wiener gain.

> [!RECAP]
> - Mix at a known SNR with $g = \sqrt{P_s / (P_n 10^{\text{SNR}/10})}$.
> - Spectral subtraction floors the subtracted power; the Wiener gain is $\xi / (1 + \xi)$ with a decision-directed $\xi$.
> - The IBM and IRM come from true powers and set a ceiling for any mask.
> - SI-SDR projects the estimate onto the reference; PIT takes the minimum loss over output orders.
> - Conv-TasNet's TCN sees $1 + R(P - 1)(2^X - 1)$ frames.

## Build it step by step

1. **`si_sdr(est, ref, eps)`** maps torch tensors `(..., T)` to `(...,)`. Subtract the means with `keepdim=True`, project, and add `eps` to both energies. The common mistake is projecting the reference onto the estimate.
2. **`mix_at_snr(clean, noise, snr_db)`** returns the mixture and the scaled noise. The common mistake is `10 ** (snr_db / 20)` inside a power ratio.
3. **`spectral_subtraction(Y, noise_power, alpha, beta)`** returns the complex estimate. Broadcast `noise_power[:, None]` and rebuild with `np.exp(1j * np.angle(Y))`. The common mistake is subtracting magnitudes instead of powers.
4. **`wiener_gain(xi)`** is one line. **`decision_directed_gains`** loops over frames and returns `(F, T)` gains, carrying the kept power $G^2 |Y|^2$ from zeros. The common mistake is carrying the gain instead.
5. **`ideal_masks(S, N, lc_db, beta)`** returns the IBM and IRM from powers. Add a tiny constant before dividing, so silent cells don't give NaN.
6. **`pit_loss(est, ref)`** returns the loss and a `(B, C)` permutation. Build the pairwise matrix by broadcasting, list permutations with `itertools.permutations`, and take `argmin`. The common mistake is returning references-to-outputs instead of outputs-to-references.
7. **`tcn_receptive_field(repeats, blocks, kernel)`** returns the formula's integer.
8. **`TCNBlock`** builds its layers in `__init__` and adds the input in `forward`. Use `groups=hidden` and `padding=dilation * (kernel - 1) // 2`.
9. **`TinyConvTasNet.forward(x)`** pads, encodes, masks with a sigmoid, reshapes to `(B, C, N, frames)`, decodes all sources in one batched call, and trims to `T`. The common mistake is forgetting the trim.

## Cost and scaling

The following table shows how each method's cost grows with the number of
frames $T$, bins $F$, sources $C$, and, for Conv-TasNet, blocks $RX$ with
bottleneck $B$ and hidden width $H$.

| Method | Time | Notes |
|---|---|---|
| Spectral subtraction, Wiener | $O(F T)$ after the STFT | Runs in real time on anything |
| Ideal masks | $O(F T)$ | Needs the clean signal |
| PIT loss | $O(C^2 T + C!\,C)$ | The $C!$ term matters only past 4 sources |
| Conv-TasNet | $O(T' \cdot R X \cdot B H)$ | Linear in length; memory grows with $T'$ for training |

Conv-TasNet's hop of 8 samples gives 1000 frames per second at 8 kHz, ten
times the rate of a 10 ms STFT. Short filters buy quality at the cost of
sequence length, which is what the dual-path models attack.

## Where this breaks

- **Noise that changes.** The classical enhancers assume the noise measured before speech holds for the whole clip. A door slam breaks that. Real systems update the estimate in pauses found by a VAD, or track each bin's minimum power over a second or two.
- **Reverberation.** Echoes of the voice are not additive noise, and these masks can't remove them. Training mixes in simulated room impulse responses, and the target must keep the direct-path delay, or the labels shift in time.
- **The noisy phase.** Every mask here reuses the phase of $Y$, which at low SNR is mostly the noise's. Complex masks and time-domain models such as Conv-TasNet avoid that cap.
- **An unknown number of speakers.** A two-output separator has nowhere to put a third voice. Systems train for a maximum count and let extra outputs go silent, or separate one speaker at a time.
- **Mismatched conditions.** WSJ0-2mix is read speech mixed digitally, and models trained on it degrade on real meetings. Test on data like your use case.

## The exercise

You build the scoring and mixing helpers `si_sdr` and `mix_at_snr`, the
classical enhancers `spectral_subtraction`, `wiener_gain`, and
`decision_directed_gains`, the oracles in `ideal_masks`, the separation loss
`pit_loss`, and Conv-TasNet's `tcn_receptive_field`, `TCNBlock`, and
`TinyConvTasNet.forward`. Press **Run** to hear each enhancer and to train a
separator that pulls a chirp out of a vowel train.

The checks test SI-SDR against the projection formula and an exact 10 dB
case, the SNR of your mixtures, and subtraction and masks on cells worked by
hand. They compare your decision-directed recursion with a reference loop,
and require gains of 8 dB for the oracle IRM and 5 dB for your Wiener filter
on noisy vowels. PIT must undo known shuffles. A change at one frame must
reach exactly the receptive field your formula predicts. Your Conv-TasNet
must keep any input length, and 60 PIT steps on a fixed batch must raise its
SI-SDR by at least 10 dB.
