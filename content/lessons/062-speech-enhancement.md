---
title: Speech enhancement
slug: 062-speech-enhancement
part: "Part 8 — Beyond recognition and synthesis"
summary: Turn down the time-frequency cells that belong to noise, with spectral subtraction, a decision-directed Wiener gain, and ideal masks, scored by SI-SDR.
minutes: 45
gpu: false
objectives:
  - Mix speech and noise at an exact SNR, and enhance the mixture with spectral subtraction and a decision-directed Wiener filter.
  - Compute the ideal binary and ratio masks from powers, and explain why they are upper bounds rather than methods.
  - Score an estimate with SI-SDR, which ignores the estimate's gain.
prereqs:
  - 007-the-short-time-fourier-transform
  - 052-neural-audio-codecs
exercise: 062-speech-enhancement
---

# Speech enhancement

You record a voice memo next to a fan. A listener understands the words, but a recognizer stumbles, and the hiss tires your ears. *Speech enhancement* removes the fan and keeps the voice. How can a program take apart a sum when it sees only the sum?

It can't take it apart exactly, but one trick comes close. Draw the sound as a spectrogram, decide for every cell how much of it belongs to the voice, and turn the rest down. Classical enhancement makes that decision from a noise estimate and a formula. This chapter builds the formulas, and the ceiling any such decision can reach. The [next chapter](/lessons/063-source-separation) asks what changes when the unwanted sound is another voice.

## What you already know

- The STFT turns a signal into complex cells $X(k, t)$, one per frequency bin $k$ and frame $t$, and the inverse STFT turns them back.
- Power is the squared magnitude $|X|^2$, and a decibel is $10 \log_{10}$ of a power ratio.
- SI-SDR, from [Neural audio codecs](/lessons/052-neural-audio-codecs), scores an estimate against a reference, ignoring its gain.

## One idea: a mask per cell

Noise adds to speech in the air, so it adds in the samples. The STFT is linear, so it adds in every cell:

$$
Y(k, t) = S(k, t) + N(k, t)
$$

Read it aloud: the noisy cell $Y$ is the speech cell $S$ plus the noise cell $N$. An enhancer sees only $Y$ and must estimate $S$. The simplest estimate multiplies each cell by a *mask* or *gain* between 0 and 1: 1 keeps the cell, 0 deletes it. Figure 1 shows the idea.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="A noisy 4 by 6 spectrogram grid, a mask grid with values near 1 where speech is strong and near 0 elsewhere, and their product">
  <defs>
    <marker id="c056-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker>
  </defs>
  <text x="25" y="22" class="fig-bold">Noisy Y</text>
  <text x="225" y="22" class="fig-bold">Mask H</text>
  <text x="425" y="22" class="fig-bold">Estimate H · Y</text>
  <g>
    <rect x="25" y="35" width="40" height="40" class="fill-soft-a"/><rect x="65" y="35" width="40" height="40" class="fill-soft-a"/><rect x="105" y="35" width="40" height="40" class="fill-soft-a"/><rect x="145" y="35" width="40" height="40" class="fill-soft-a"/>
    <rect x="25" y="75" width="40" height="40" class="fill-soft-a"/><rect x="65" y="75" width="40" height="40" class="fill-a"/><rect x="105" y="75" width="40" height="40" class="fill-a"/><rect x="145" y="75" width="40" height="40" class="fill-soft-a"/>
    <rect x="25" y="115" width="40" height="40" class="fill-soft-a"/><rect x="65" y="115" width="40" height="40" class="fill-a"/><rect x="105" y="115" width="40" height="40" class="fill-a"/><rect x="145" y="115" width="40" height="40" class="fill-soft-a"/>
    <rect x="25" y="155" width="40" height="40" class="fill-soft-a"/><rect x="65" y="155" width="40" height="40" class="fill-soft-a"/><rect x="105" y="155" width="40" height="40" class="fill-soft-a"/><rect x="145" y="155" width="40" height="40" class="fill-soft-a"/>
  </g>
  <g class="fig-small">
    <rect x="225" y="35" width="160" height="160" class="fill-paper stroke-grid"/>
    <rect x="265" y="75" width="80" height="80" class="fill-soft-b stroke-b"/>
    <text x="231" y="60">0.05</text><text x="271" y="60">0.1</text><text x="311" y="60">0.1</text><text x="351" y="60">0.05</text>
    <text x="231" y="100">0.1</text><text x="271" y="100">0.9</text><text x="311" y="100">0.9</text><text x="351" y="100">0.1</text>
    <text x="231" y="140">0.1</text><text x="271" y="140">0.9</text><text x="311" y="140">0.9</text><text x="351" y="140">0.1</text>
    <text x="231" y="180">0.05</text><text x="271" y="180">0.1</text><text x="311" y="180">0.1</text><text x="351" y="180">0.05</text>
  </g>
  <rect x="425" y="75" width="80" height="80" class="fill-b"/>
  <rect x="425" y="35" width="160" height="160" class="stroke-grid" fill="none"/>
  <line x1="190" y1="115" x2="220" y2="115" class="stroke-muted" marker-end="url(#c056-arrow)"/>
  <line x1="390" y1="115" x2="420" y2="115" class="stroke-muted" marker-end="url(#c056-arrow)"/>
  <text x="25" y="217" class="fig-small">Dark cells are speech, pale cells are noise. Rows are frequency, columns are time.</text>
</svg>
Figure 1. A mask is one number per cell. It keeps the dark speech cells (gain near 0.9) and turns the pale noise cells down to a tenth or less.
```

Every method in this chapter fills in that middle grid differently. The two practical methods use a noise power estimate per bin, measured while nobody talks. The oracle masks cheat, to show the best a mask could do. To test an enhancer, build mixtures at a known *signal-to-noise ratio* (SNR), the speech power over the noise power in decibels. Scale the noise by a gain $g$:

$$
y = s + g\,n, \qquad g = \sqrt{\frac{P_s}{P_n \cdot 10^{\text{SNR}/10}}}, \qquad P_x = \frac{1}{T}\sum_t x_t^2
$$

Read it aloud: $P_s$ and $P_n$ are mean squares. Dividing by $10^{\text{SNR}/10}$ turns decibels back into a power ratio, and the square root turns a power gain into an amplitude gain. For $P_s = 0.04$, $P_n = 1$, and a target of 5 dB, the gain is $\sqrt{0.04 / 3.162} = 0.1125$.

## Spectral subtraction

The obvious approach: measure the noise power in each bin while nobody talks, then subtract it from every frame. Boll published this in 1979:

$$
|\hat S(k, t)|^2 = \max\left(|Y(k, t)|^2 - \alpha \hat P_N(k),\ \beta\, |Y(k, t)|^2\right), \qquad \hat S = |\hat S|\, e^{j \angle Y}
$$

Read it aloud: take the noisy power, subtract $\alpha$ times the estimated noise power $\hat P_N(k)$, and never go below a floor of $\beta$ times the noisy power. Keep the noisy phase $\angle Y$, because nothing better is available. The *over-subtraction factor* $\alpha$, often 2, removes more than the average noise. The *floor* $\beta$, often 0.01, keeps the power positive.

For one cell with $|Y|^2 = 25$, $\hat P_N = 4$, $\alpha = 2$, and $\beta = 0.01$, the estimate is $\max(25 - 8, 0.25) = 17$, so the magnitude drops from 5 to 4.12.

```python
power = np.abs(Y) ** 2
clean = np.maximum(power - alpha * noise_power[:, None], beta * power)
S_hat = np.sqrt(clean) * np.exp(1j * np.angle(Y))
```

Where does this fail? Noise power is an average, and each frame fluctuates around it, so some cells survive as lone peaks that come and go. They sound like tinkling tones, called *musical noise*. The fix is a gain that changes smoothly.

## The Wiener gain

The *Wiener filter* picks the gain $H$ that minimizes $\mathbb{E}|S - H Y|^2$. For uncorrelated speech and noise, setting the derivative to zero gives

$$
H(k) = \frac{P_S(k)}{P_S(k) + P_N(k)} = \frac{\xi(k)}{1 + \xi(k)}, \qquad \xi(k) = \frac{P_S(k)}{P_N(k)}
$$

Read it aloud: the gain is the speech's share of the power in that bin, written with the *a priori SNR* $\xi$, the speech-to-noise power ratio. At $\xi = 3$ the gain is 0.75. Figure 2 plots the curve: bins where speech dominates pass almost untouched, and bins of pure noise go toward zero.

```figure
<svg viewBox="0 0 600 240" role="img" aria-label="The Wiener gain xi over one plus xi plotted against a priori SNR in decibels, rising from 0.01 at minus 20 dB to 0.99 at plus 20 dB">
  <line x1="60" y1="190" x2="540" y2="190" class="stroke-muted"/>
  <line x1="60" y1="30" x2="60" y2="190" class="stroke-muted"/>
  <line x1="60" y1="110" x2="540" y2="110" class="stroke-grid dashed"/>
  <path d="M60 188 L84 188 L108 186 L132 184 L156 181 L180 175 L204 168 L228 158 L252 144 L276 128 L300 110 L324 92 L348 76 L372 62 L396 52 L420 45 L444 39 L468 36 L492 34 L516 32 L540 32" class="stroke-b" fill="none"/>
  <circle cx="300" cy="110" r="5" class="fill-b"/>
  <circle cx="357" cy="70" r="5" class="fill-a"/>
  <text x="308" y="130" class="fig-small">xi = 1 (0 dB): gain 0.5</text>
  <text x="366" y="88" class="fig-small">xi = 3 (4.8 dB): gain 0.75</text>
  <text x="20" y="194" class="fig-small">0</text><text x="20" y="114" class="fig-small">0.5</text><text x="20" y="38" class="fig-small">1</text>
  <text x="52" y="210" class="fig-small">-20</text><text x="172" y="210" class="fig-small">-10</text><text x="296" y="210" class="fig-small">0</text><text x="412" y="210" class="fig-small">10</text><text x="528" y="210" class="fig-small">20</text>
  <text x="250" y="232">a priori SNR xi (dB)</text>
  <text x="68" y="24" class="fig-small">gain H</text>
</svg>
Figure 2. The Wiener gain is a soft switch. It passes cells above 10 dB almost whole (gain above 0.9) and mutes cells below -10 dB (gain below 0.1).
```

The formula needs $\xi$, which depends on the clean speech you don't have. The obvious stand-in is the measurable *a posteriori SNR* $\gamma = |Y|^2 / \hat P_N$, minus one, but it jumps between frames and brings back musical noise. Ephraim and Malah's *decision-directed* estimate (1984) blends it with the previous frame's cleaned output:

$$
\hat\xi(k, t) = a\,\frac{|\hat S(k, t - 1)|^2}{\hat P_N(k)} + (1 - a) \max\left(\gamma(k, t) - 1,\ 0\right)
$$

Read it aloud: 98 percent of the estimate, with $a = 0.98$, comes from the SNR of what you kept in the previous frame, and 2 percent from what this frame measures. A floor $\xi_{\min}$, usually $-25$ dB, keeps the gain from reaching exactly zero. Each frame then sets $G = \hat\xi / (1 + \hat\xi)$ and $\hat S = G Y$, and the next frame reuses that $\hat S$.

### Work the recursion by hand

Follow one bin with $\hat P_N = 1$, $a = 0.98$, and $\xi_{\min} = 0.00316$, starting from $|\hat S|^2 = 0$. The noisy powers are 5 and 2 while only noise plays, then 30 and 30 when a vowel starts.

| Frame | $\lvert Y \rvert^2$ | $\hat\xi$ | $G$ | $\lvert \hat S \rvert^2 = G^2 \lvert Y \rvert^2$ |
|---|---|---|---|---|
| 0 | 5 | $0.02 \times 4 = 0.080$ | 0.074 | 0.027 |
| 1 | 2 | $0.98 \times 0.027 + 0.02 \times 1 = 0.047$ | 0.045 | 0.004 |
| 2 | 30 | $0.98 \times 0.004 + 0.02 \times 29 = 0.584$ | 0.369 | 4.08 |
| 3 | 30 | $0.98 \times 4.08 + 0.02 \times 29 = 4.576$ | 0.821 | 20.2 |

Figure 3 compares this with the one-frame estimate $\max(\gamma - 1, 0)$ alone.

```figure
<svg viewBox="0 0 600 230" role="img" aria-label="Bar chart of gains over four frames. The one-frame estimate gives 0.80, 0.50, 0.97, 0.97. The decision-directed estimate gives 0.07, 0.04, 0.37, 0.82.">
  <line x1="60" y1="190" x2="540" y2="190" class="stroke-muted"/>
  <line x1="60" y1="30" x2="60" y2="190" class="stroke-muted"/>
  <rect x="90" y="62" width="40" height="128" class="fill-a"/><rect x="134" y="178" width="40" height="12" class="fill-b"/>
  <rect x="210" y="110" width="40" height="80" class="fill-a"/><rect x="254" y="183" width="40" height="7" class="fill-b"/>
  <rect x="330" y="35" width="40" height="155" class="fill-a"/><rect x="374" y="131" width="40" height="59" class="fill-b"/>
  <rect x="450" y="35" width="40" height="155" class="fill-a"/><rect x="494" y="59" width="40" height="131" class="fill-b"/>
  <text x="20" y="194" class="fig-small">0</text><text x="20" y="38" class="fig-small">1</text>
  <text x="108" y="210" class="fig-small">frame 0</text><text x="228" y="210" class="fig-small">frame 1</text><text x="348" y="210" class="fig-small">frame 2</text><text x="468" y="210" class="fig-small">frame 3</text>
  <text x="70" y="24" class="fig-small">gain (dimensionless)</text>
  <rect x="350" y="8" width="12" height="12" class="fill-a"/><text x="366" y="19" class="fig-small">one frame</text>
  <rect x="440" y="8" width="12" height="12" class="fill-b"/><text x="456" y="19" class="fig-small">decision-directed</text>
</svg>
Figure 3. In the two noise-only frames, the one-frame estimate opens the gain to 0.80 and 0.50, while the decision-directed gain stays below 0.08. The price is that it needs two frames to open for the vowel.
```

> [!KEY]
> The Wiener gain $\xi / (1 + \xi)$ is optimal for the true a priori SNR. Everything practical about it is in how you estimate $\xi$. The decision-directed rule trades a little attack time for far less musical noise.

## Ideal masks: the ceiling

How good could any mask be? Suppose an oracle told you $S$ and $N$ for every cell. The gain is then exact, and it shows how far a real method could go. Supervised enhancers train networks to predict these *ideal masks*.

The *ideal binary mask* (IBM) keeps a cell when speech is louder than the noise by a local criterion LC, often 0 dB. The *ideal ratio mask* (IRM) is soft:

$$
\text{IBM}(k, t) = \mathbb{1}\left[ 10 \log_{10} \frac{|S|^2}{|N|^2} > \text{LC} \right], \qquad
\text{IRM}(k, t) = \left( \frac{|S|^2}{|S|^2 + |N|^2} \right)^{\beta}
$$

Read them aloud: the IBM is 1 where the cell's local SNR beats the criterion and 0 elsewhere. The IRM is the speech's share of the cell's power, raised to $\beta$, typically 0.5. With $\beta = 1$, it is the Wiener gain computed from the true powers of that one cell. For a cell with $|S|^2 = 4$ and $|N|^2 = 1$, the local SNR is 6.02 dB, so the IBM is 1 and the IRM with $\beta = 0.5$ is $\sqrt{0.8} = 0.894$.

```python
ibm = (10 * np.log10(ps / pn) > lc_db).astype(float)
irm = (ps / (ps + pn)) ** beta
```

On the Run's vowels at 5 dB SNR, spectral subtraction scores 12.5 dB SI-SDR, the decision-directed Wiener filter 15.9 dB, and the IBM and IRM oracles 18.6 dB and 18.1 dB. The oracles stop short of perfect because they keep the noisy phase.

> [!WARNING]
> Compute masks from powers, never from dB spectrograms, and never report an oracle mask as if it were a method. An oracle needs the clean signal, so its score is a ceiling, not a result.

## Score it with SI-SDR

Plain SNR punishes an estimate for being quieter, though the sound is the same. *Scale-invariant SDR* (SI-SDR) fixes that. It removes the mean from both signals, scales the reference by the projection coefficient $\langle \hat s, s \rangle / \lVert s \rVert^2$ to get the part of the estimate that matches the reference, and compares the energy of that part with the energy of what is left over, in decibels. Papers report the *improvement*, SI-SDRi: the estimate's SI-SDR minus the mixture's.

Work it on four samples. Let the reference be $s = [1, -1, 1, -1]$ and the estimate $\hat s = [1.5, -0.5, 0.5, -1.5]$. The dot product is $1.5 + 0.5 + 0.5 + 1.5 = 4$ and $\lVert s \rVert^2 = 4$, so the projection coefficient is 1 and the matching part is $s$ itself. What is left is $[0.5, 0.5, -0.5, -0.5]$, with energy 1. The score is $10 \log_{10}(4 / 1) = 6.02$ dB. Double the estimate to $[3, -1, 1, -3]$: the coefficient becomes 2, the left-over is $[1, 1, -1, -1]$ with energy 4, the matching part has energy 16, and the score is still 6.02 dB.

Write it in torch, batched over the last axis, so $-\text{SI-SDR}$ can be a loss in the next chapter:

```python
target = ((est * ref).sum(-1, keepdim=True) / (ref * ref).sum(-1, keepdim=True)) * ref
si_sdr = 10 * torch.log10((target**2).sum(-1) / ((est - target) ** 2).sum(-1))
```

> [!RECAP]
> - Mix at a known SNR with $g = \sqrt{P_s / (P_n 10^{\text{SNR}/10})}$.
> - Spectral subtraction floors the subtracted power, and keeps the noisy phase.
> - The Wiener gain is $\xi / (1 + \xi)$, with a decision-directed $\xi$ that smooths across frames.
> - The IBM and IRM come from true powers and set a ceiling for any mask.
> - SI-SDR projects the estimate onto the reference, so it ignores gain.

## Build it step by step

Build the functions in the checks' order.

1. **`si_sdr(est, ref, eps)`** maps `(..., T)` tensors to `(...,)`. Subtract the means with `keepdim=True`, project, and add `eps` to both energies. The common mistake is skipping the mean.
2. **`mix_at_snr(clean, noise, snr_db)`** returns the mixture and the scaled noise. The common mistake is `10 ** (snr_db / 20)` inside a power ratio.
3. **`spectral_subtraction`** broadcasts `noise_power[:, None]` and rebuilds with `np.exp(1j * np.angle(Y))`. The common mistake is subtracting magnitudes instead of powers.
4. **`wiener_gain`** is one line. **`decision_directed_gains`** loops over frames, carrying the kept power $G^2 |Y|^2$ from zeros. The common mistake is carrying the gain instead.
5. **`ideal_masks`** returns the IBM and IRM from powers. Add a tiny constant before dividing.

## Where this breaks

- **Noise that changes.** The classical enhancers assume the noise measured before speech holds for the whole clip. A door slam breaks that, so real systems update the estimate in pauses found by a VAD.
- **Reverberation and phase.** Echoes are not additive noise, and every mask here reuses the noisy phase, which caps the score. Time-domain models avoid that cap, as [Separating voices](/lessons/063-source-separation) shows.

## The exercise

You build `si_sdr`, `mix_at_snr`, `spectral_subtraction`, `wiener_gain`, `decision_directed_gains`, and `ideal_masks`. Press **Run** to hear each enhancer on noisy vowels and compare SI-SDR scores.

The checks test your functions on cells worked by hand, confirm that SI-SDR ignores scale, and require gains of 8 dB for the oracle IRM and 5 dB for your Wiener filter on noisy vowels at 0 dB.

> [!KEY]
> To clean noise from speech, you decide per time-frequency cell how much to keep. A formula (subtraction, Wiener gain) fills in that mask from a noise estimate, an oracle mask shows how good any mask could be, and SI-SDR scores the result without caring about gain. When the unwanted sound is another voice, there is no quiet stretch to measure, which is where the next chapter starts.
