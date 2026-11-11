---
title: Microphone arrays
slug: 011-microphone-arrays
part: "Part 1 — Sound and signals"
summary: Align a microphone array, estimate the delay with GCC-PHAT, and cancel loudspeaker echo with NLMS.
minutes: 40
gpu: false
objectives:
  - Compute the sample delay of a plane wave along a linear microphone array.
  - Align the channels with delay-and-sum, and state the SNR gain for uncorrelated noise.
  - Estimate a time difference of arrival with GCC-PHAT, including a fraction of a sample.
  - Compute MVDR weights from a noise covariance so the look direction keeps gain 1.
  - Cancel a synthetic echo with NLMS and read the echo return loss enhancement.
prereqs:
  - 009-convolution-and-filters
  - 010-rooms-and-reverb
exercise: 011-microphone-arrays
---

# Microphone arrays

A smart speaker sits across the room. The person is a few meters away, the walls add copies of that voice, and the loudspeaker plays the assistant's reply into the same microphones that have to hear the person. One microphone records all of that as a single mixture. Several microphones, a few centimeters apart, hear the voice at slightly different times. That time difference is how you aim the array at the person.

The loudspeaker is a separate handle, because you already have the signal that was played. The room's copy of it is an impulse response, so you can subtract an estimate of that copy. CHiME-6 records dinner-party speech on linear arrays of four sample-synchronized microphones, and one of its baselines is a weighted delay-and-sum. This chapter builds that delay, the weights that beat it when noise has a direction, and the canceler. The agent that talks and listens at once is [Full-duplex models and voice agents](/lessons/074-full-duplex-and-voice-agents).

## What you already know

- Convolution with an impulse response is how a room, and a loudspeaker path, changes a signal.
- A synthetic room impulse response starts with a direct-path sample of 1 and then decays.
- A power ratio in decibels is $10 \log_{10}$ of the ratio.
- A delay of $D$ samples is a linear phase $e^{-j 2\pi f D}$ across the FFT.
- Uncorrelated noise gets quieter when you average it. An aligned voice does not.

## A far voice arrives as a plane wave

Far from a short row of microphones, the wavefront is almost flat. The distance to the person is much larger than the gap between microphones, so the amplitude is nearly the same at each one. The difference that matters is the arrival time. Call $\theta = 0$ *broadside*: the wave comes straight at the array, perpendicular to the line of microphones. From an angle, microphone $m$ (counting from 0) sits an extra $m \, d \sin\theta$ meters back, with $d$ the spacing in meters.

Divide by the speed of sound. In dry air at $20\,^\circ\mathrm{C}$ that speed is $343$ m/s, so

$$
\tau_m = \frac{m \, d \sin\theta}{c}
$$

In samples at rate $f_s$, the delay is $\delta_m = f_s \tau_m$. Take four microphones, $d = 5$ cm, $\theta = 30^\circ$, and $f_s = 16$ kHz. Then $\sin 30^\circ = 1/2$, the extra path from one microphone to the next is $2.5$ cm, and

$$
\delta = 16000 \times \frac{0.025}{343} = 1.166 \text{ samples}
$$

The four delays are $0$, $1.166$, $2.332$, and $3.499$ samples. Figure 1 plots them. They are not whole samples, so a shift is an interpolation, not an index.

```figure
<svg viewBox="0 0 640 280" role="img" aria-label="Four microphones in a line. A 30 degree plane wave arrives 0, 1.166, 2.332, and 3.499 samples later.">
  <text x="320" y="28" text-anchor="middle" class="fig-bold">Arrival delay along a 5 cm array</text>
  <text x="24" y="120" class="fig-small">samples</text>
  <line x1="78" y1="228.0" x2="580" y2="228.0" class="stroke-grid"/>
  <text x="70" y="232.0" text-anchor="end" class="fig-small">0</text>
  <line x1="78" y1="182.0" x2="580" y2="182.0" class="stroke-grid"/>
  <text x="70" y="186.0" text-anchor="end" class="fig-small">1</text>
  <line x1="78" y1="136.0" x2="580" y2="136.0" class="stroke-grid"/>
  <text x="70" y="140.0" text-anchor="end" class="fig-small">2</text>
  <line x1="78" y1="90.0" x2="580" y2="90.0" class="stroke-grid"/>
  <text x="70" y="94.0" text-anchor="end" class="fig-small">3</text>
  <rect x="108.0" y="228.0" width="56" height="0.0" class="fill-a"/>
  <text x="136.0" y="220.0" text-anchor="middle" class="fig-mono fig-small">0.000</text>
  <text x="136.0" y="250.0" text-anchor="middle" class="fig-small">mic 0</text>
  <rect x="228.0" y="174.4" width="56" height="53.6" class="fill-a"/>
  <text x="256.0" y="166.4" text-anchor="middle" class="fig-mono fig-small">1.166</text>
  <text x="256.0" y="250.0" text-anchor="middle" class="fig-small">mic 1</text>
  <rect x="348.0" y="120.7" width="56" height="107.3" class="fill-a"/>
  <text x="376.0" y="112.7" text-anchor="middle" class="fig-mono fig-small">2.332</text>
  <text x="376.0" y="250.0" text-anchor="middle" class="fig-small">mic 2</text>
  <rect x="468.0" y="67.1" width="56" height="160.9" class="fill-a"/>
  <text x="496.0" y="59.1" text-anchor="middle" class="fig-mono fig-small">3.499</text>
  <text x="496.0" y="250.0" text-anchor="middle" class="fig-small">mic 3</text>
  <line x1="78" y1="228" x2="580" y2="228" class="stroke-muted"/>
  <text x="320" y="268" text-anchor="middle" class="fig-small">30 degrees from broadside, 343 m/s, 16 kHz</text>
</svg>
Figure 1. Each step along the array adds 1.166 samples. The heights are $16000 \times 0.025 / 343$ times the microphone index.
```

> [!KEY]
> A far source reaches a line of microphones as one wave with delay $\tau_m = m d \sin\theta / c$ at microphone $m$. Close to the array the wavefront curves, and each microphone also hears a different amplitude.

## Delay-and-sum lines the voice up

Undo the delays and average. If $\delta_m$ is how many samples late microphone $m$ is, reading that channel at time $n + \delta_m$ lines it up with microphone 0:

$$
y[n] = \frac{1}{M} \sum_{m=0}^{M-1} x_m[n + \delta_m]
$$

Samples off the end of a channel are zeros. The factor $1/M$ keeps the voice at its original level. Dropping it does not change a signal-to-noise ratio, because the voice and the noise scale together, but it does change the waveform.

A click arrives at samples 2, 3, and 4 on three microphones, so the delays are $0$, $1$, and $2$:

$$
\begin{aligned}
x_0 &= [0, 0, 1, 0, 0, 0, 0] \\
x_1 &= [0, 0, 0, 1, 0, 0, 0] \\
x_2 &= [0, 0, 0, 0, 1, 0, 0]
\end{aligned}
$$

Advance channel 1 by one sample and channel 2 by two. Each click lands on sample 2, and the average is $[0, 0, 1, 0, 0, 0, 0]$. Average without those shifts and the click smears into $[0, 0, 1/3, 1/3, 1/3, 0, 0]$. For a whole-sample delay, `np.interp(n + delay, n, channel, left=0, right=0)` is an exact shift. The same call interpolates a fraction of a sample.

After a correct shift, every channel is the same voice $s[n]$ plus its own noise $v_m[n]$. When the noises are uncorrelated with each other and with $s$, and each has power $\sigma^2$, the average is $s[n]$ plus a noise of power $\sigma^2 / M$. Variances add, and the $1/M$ is squared. The signal-to-noise ratio, a power ratio, grows by $M$:

$$
G = 10 \log_{10} M
$$

Four microphones are $6.02$ dB, since $10 \log_{10} 4 = 20 \log_{10} 2$. This is the gain for uncorrelated noise after a perfect alignment. It is not a measurement from a real array. A noise from one direction is correlated across the microphones, and the formula does not apply to it.

> [!KEY]
> Delay-and-sum raises the SNR by $10 \log_{10} M$ decibels when the noises are uncorrelated and the voice is aligned. The voice adds in step. The noise does not.

## GCC-PHAT reads the delay

The delays so far were given. A real array measures the time difference of arrival between a pair of channels. Ordinary cross-correlation does that, and then a vowel's spectrum pulls the peak around, because a loud formant dominates the sum. The phase transform drops the magnitude and keeps the phase, which is where the delay lives:

$$
G[k] = \frac{X_1[k] \, X_2[k]^{*}}{\left|X_1[k] \, X_2[k]^{*}\right|}
$$

The inverse FFT of $G$ peaks as a sinc on the delay. Figure 2 is that curve for a delay of $2.5$ samples. Bins where the product is zero have no phase. Leave them out of the division.

```figure
<svg viewBox="0 0 640 280" role="img" aria-label="GCC-PHAT against lag in samples. The curve peaks halfway between lag 2 and lag 3, at the planted delay of 2.5 samples.">
  <text x="320" y="28" text-anchor="middle" class="fig-bold">GCC-PHAT of a 2.5-sample delay</text>
  <text x="22" y="130" class="fig-small">correlation</text>
  <line x1="143.2" y1="214" x2="143.2" y2="220" class="stroke-muted"/>
  <text x="143.2" y="236" text-anchor="middle" class="fig-small">0</text>
  <line x1="208.5" y1="214" x2="208.5" y2="220" class="stroke-muted"/>
  <text x="208.5" y="236" text-anchor="middle" class="fig-small">1</text>
  <line x1="273.8" y1="214" x2="273.8" y2="220" class="stroke-muted"/>
  <text x="273.8" y="236" text-anchor="middle" class="fig-small">2</text>
  <line x1="339.0" y1="214" x2="339.0" y2="220" class="stroke-muted"/>
  <text x="339.0" y="236" text-anchor="middle" class="fig-small">3</text>
  <line x1="404.2" y1="214" x2="404.2" y2="220" class="stroke-muted"/>
  <text x="404.2" y="236" text-anchor="middle" class="fig-small">4</text>
  <line x1="469.5" y1="214" x2="469.5" y2="220" class="stroke-muted"/>
  <text x="469.5" y="236" text-anchor="middle" class="fig-small">5</text>
  <line x1="534.8" y1="214" x2="534.8" y2="220" class="stroke-muted"/>
  <text x="534.8" y="236" text-anchor="middle" class="fig-small">6</text>
  <line x1="600.0" y1="214" x2="600.0" y2="220" class="stroke-muted"/>
  <text x="600.0" y="236" text-anchor="middle" class="fig-small">7</text>
  <line x1="78" y1="166.6" x2="600" y2="166.6" class="stroke-grid"/>
  <text x="70" y="170.6" text-anchor="end" class="fig-small">0</text>
  <line x1="78" y1="87.5" x2="600" y2="87.5" class="stroke-grid"/>
  <text x="70" y="91.5" text-anchor="end" class="fig-small">0.5</text>
  <line x1="78" y1="214" x2="600" y2="214" class="stroke-muted"/>
  <line x1="78" y1="40" x2="78" y2="214" class="stroke-muted"/>
  <polyline points="78.0,180.9 143.2,146.4 208.5,200.1 273.8,65.9 339.0,65.9 404.2,200.1 469.5,146.4 534.8,180.9 600.0,155.4" class="stroke-a" fill="none"/>
  <line x1="306.4" y1="48" x2="306.4" y2="214" class="stroke-b dashed"/>
  <circle cx="306.4" cy="65.9" r="5" class="fill-b"/>
  <text x="314.4" y="55.9" class="fig-small">2.5 samples</text>
  <text x="340" y="268" text-anchor="middle" class="fig-small">lag of microphone 2 behind microphone 1 (samples)</text>
</svg>
Figure 2. The bins at lags 2 and 3 are equal, so the delay sits halfway between them. The curve is the inverse FFT of the phase transform of a 2.5-sample circular delay.
```

Numpy puts lag 0 at index 0 and negative lags at the end. The product $X_1 X_2^{*}$ peaks at the lag of channel 1 behind channel 2, so negate it to report how late channel 2 is. A fraction of a sample comes from the peak bin $k$ and its neighbors $y_{-}$, $y_{k}$, $y_{+}$:

$$
\hat{\delta} = k + \frac{1}{2} \frac{y_{-} - y_{+}}{y_{-} - 2 y_{k} + y_{+}}
$$

On a pure delay this parabola misses the true lag by at most $0.119$ samples. The integer bin is $0.5$ samples off when the delay lands halfway between bins. FFT at the length of the two signals. Zero-padding a circular phase ramp moves the peak.

## MVDR places a null

Delay-and-sum treats every direction the same apart from the delay you steered, so a loud interferer from another angle still leaks through. At one frequency, let $\mathbf{a}$ be the steering vector of the direction you want to keep. Entry $m$ is $e^{-j 2\pi f \tau_m}$. From broadside, $\mathbf{a}$ is a vector of ones.

Minimum-variance distortionless response picks weights $\mathbf{w}$ that minimize the output noise power $\mathbf{w}^{H} \mathbf{R} \mathbf{w}$, with $\mathbf{R}$ the noise covariance, while the look direction keeps gain 1:

$$
\mathbf{w} = \frac{\mathbf{R}^{-1} \mathbf{a}}{\mathbf{a}^{H} \mathbf{R}^{-1} \mathbf{a}}, \qquad \mathbf{w}^{H} \mathbf{a} = 1
$$

The denominator is the constraint. Drop it and $\mathbf{w}^{H} \mathbf{a}$ is whatever $\mathbf{a}^{H} \mathbf{R}^{-1} \mathbf{a}$ happens to be. Delay-and-sum is $\mathbf{a} / (\mathbf{a}^{H} \mathbf{a})$. It meets the constraint and ignores $\mathbf{R}$.

Two microphones, with the second one noisier: $\mathbf{R} = \mathrm{diag}(1, 4)$ and $\mathbf{a} = [1, 1]^{\mathsf T}$. Then $\mathbf{R}^{-1}\mathbf{a} = [1, 0.25]^{\mathsf T}$, the denominator is $1.25$, and $\mathbf{w} = [0.8, 0.2]^{\mathsf T}$. The constraint holds because $0.8 + 0.2 = 1$. Output noise power is $0.8$. Delay-and-sum uses $[0.5, 0.5]$ and gets $1.25$. Same voice, less noise.

`np.linalg.solve(R, a)` is $\mathbf{R}^{-1}\mathbf{a}$, and `np.vdot(a, that)` is $\mathbf{a}^{H}\mathbf{R}^{-1}\mathbf{a}$, because `np.vdot` conjugates its first argument. Against a directional interferer the same weights null that direction, and the output power drops further than delay-and-sum's.

> [!WARNING]
> $\mathbf{R}$ has to be the noise. If the person's voice is inside $\mathbf{R}$, the minimizer turns that voice down, and the constraint cannot save a direction you left out of $\mathbf{a}$.

## NLMS subtracts the echo you already have

Beamforming never sees the loudspeaker waveform. Echo cancellation does. Let $x[n]$ be what you played, and let the microphone be the echo $d[n] = (h * x)[n]$ plus, sometimes, the near-end person. A filter of $L$ taps predicts the echo from recent samples of $x$:

$$
e[n] = d[n] - \mathbf{w}^{\mathsf T} \mathbf{x}[n]
$$

with $\mathbf{x}[n] = [x[n], x[n-1], \ldots, x[n-L+1]]^{\mathsf T}$ and $x[k] = 0$ for $k < 0$. Normalized least mean squares steps along $\mathbf{x}[n]$, scaled so a loud frame and a quiet frame move by similar amounts:

$$
\mathbf{w} \leftarrow \mathbf{w} + \mu \frac{e[n] \, \mathbf{x}[n]}{\lVert \mathbf{x}[n] \rVert^{2} + \delta}
$$

$\delta > 0$ keeps a silent frame from dividing by zero. One tap shows the division. Let $x = [2, 2]$, the microphone be $[1, 1]$ (the true gain is $0.5$), $\mu = 1$, and $\delta = 0$, with $\mathbf{w}$ starting at 0. The first error is $1$ and the energy of $\mathbf{x}$ is $4$, so the tap becomes $1 \cdot 1 \cdot 2 / 4 = 0.5$. The second prediction is $1$ and the error is $0$. The error signal is $[1, 0]$. Skip the division and the first update adds $2$, the tap overshoots, and the second error is $-3$.

Echo return loss enhancement is how much echo power the subtraction removed. WebRTC defines it as

$$
\mathrm{ERLE} = 10 \log_{10} \frac{P_{\mathrm{echo}}}{P_{\mathrm{out}}}
$$

A filter that never moves scores $0$ dB, because the output is the echo.

Double-talk is the stretch when the loudspeaker and the person are both active. Then $e[n]$ contains the person's voice, and the update walks the taps away from the true room. Hold $\mathbf{w}$ still during that stretch. The exercise is echo alone, on the synthetic impulse response from the rooms chapter, so $e[n]$ really is leftover echo.

> [!KEY]
> Delays across the microphones aim the array. The known loudspeaker signal lets NLMS subtract the room's copy of it, which is what keeps a full-duplex agent from recognizing its own voice.

> [!RECAP]
> - A plane wave is $\delta_m = f_s m d \sin\theta / c$ samples late at microphone $m$.
> - Delay-and-sum shifts by $\delta_m$ and averages. Uncorrelated noise falls by $10 \log_{10} M$ decibels.
> - GCC-PHAT whitens the cross-spectrum. The peak lag, plus a parabola, is the delay.
> - MVDR weights are $\mathbf{R}^{-1}\mathbf{a} / (\mathbf{a}^{H}\mathbf{R}^{-1}\mathbf{a})$, so $\mathbf{w}^{H}\mathbf{a} = 1$.
> - NLMS steps by $\mu e[n] \mathbf{x}[n] / (\lVert\mathbf{x}[n]\rVert^2 + \delta)$. Freeze it during double-talk.

## Build it step by step

A positive delay means that channel arrives later. The checks call the functions in this order.

1. **`gcc_phat(x1, x2)`** takes two equal-length 1-D arrays and returns how many samples $x_2$ lags $x_1$, as a Python float. FFT at that length, divide $X_1 \overline{X_2}$ by its magnitude, inverse-FFT, negate the lag, and add the parabolic offset. Returning the integer bin is $0.5$ samples off for a delay of $2.5$.
2. **`delay_and_sum(signals, delays)`** takes shape `(M, N)` and delays of shape `(M,)`. Advance each channel with `np.interp` and zeros outside the buffer, then average. Return shape `(N,)`. Averaging in place, with no shift, smears the worked click into three samples of height $1/3$.
3. **`mvdr_weights(noise_cov, steering)`** takes an $(M, M)$ covariance and a steering vector of length $M$. Solve $\mathbf{R}\mathbf{u} = \mathbf{a}$ and divide $\mathbf{u}$ by `np.vdot(a, u)`. Then `np.vdot(w, a)` is 1. Returning $\mathbf{R}^{-1}\mathbf{a}$ without the division breaks the constraint.
4. **`nlms_echo_cancel(far, mic, n_taps, mu, delta)`** takes two length-$N$ signals and returns the error, shape `(N,)`. The tap vector at time $n$ is `far[n], far[n-1], ...`, with zeros before the start. Subtract the prediction, then update. A step of $\mu e \mathbf{x}$ with no division by the energy overshoots the one-tap example.

## Where this breaks

One delay lines up the direct path and leaves the reflections, each of which has its own delay. On the CHiME-3 real recordings, the Interspeech 2019 beamforming tutorial lists word error of $9.0\%$ and $15.6\%$ with no beamforming, $9.4\%$ and $16.2\%$ for delay-and-sum with a direction estimate, and $4.8\%$ and $8.9\%$ for a spatial-mixture beamformer, on the real development set and the real test set. Delay-and-sum aimed at a direction did not beat the raw channels.

The $10 \log_{10} M$ gain needs uncorrelated noise. A directional interferer is correlated, and MVDR is the method that nulls it, provided $\mathbf{R}$ was estimated while the person was quiet.

A spacing wider than half a wavelength lets a second angle imitate the same delays. At $5$ cm and $343$ m/s that starts at $c / (2d) = 3430$ Hz.

During double-talk the error is mostly the near-end voice. Keep adapting and the filter spends its taps on the person. WebRTC's AEC3 freezes or slows the update for that stretch. The [full-duplex chapter](/lessons/074-full-duplex-and-voice-agents) is where an agent talks over the person on purpose.

## The exercise

You write `gcc_phat`, `delay_and_sum`, `mvdr_weights`, and `nlms_echo_cancel`. Press **Run** to hear a noisy channel against the delay-and-sum mix, and a loudspeaker through the room from the last chapter against the NLMS residual.

The checks recover two fractional delays, the worked click, and an SNR gain within five standard deviations of $10 \log_{10} M$ on seeded uncorrelated noise. They match the weights $[0.8, 0.2]$, require the distortionless constraint, and require less noise power than delay-and-sum on a directional interferer. They match the one-tap error $[1, 0]$, and they require $20$ dB of ERLE on a synthetic room. A filter that never updates scores $0$ dB there.

## Further reading

- [Microphone array signal processing and deep learning for speech enhancement](https://www.interspeech2019.org/program/tutorials/), tutorial T8, Reinhold Haeb-Umbach and Tomohiro Nakatani, Interspeech 2019. The [beamforming slides](https://ei.uni-paderborn.de/fileadmin-eim/elektrotechnik/fg/nth/Forschung/Publications/Interspeech_2019_Tutorial/IS2019_beamforming.pdf) cover the plane wave, delay-and-sum, and MVDR.
- [CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings](https://arxiv.org/html/2004.09249v2), Shinji Watanabe and colleagues, 2020. Four-channel linear arrays, with weighted delay-and-sum as one baseline.
- [Speech Recognition and Understanding (11-751/18-781)](https://www.wavlab.org/activities/2022/11751-2022f/), Shinji Watanabe, Carnegie Mellon University, fall 2022. The 28 November lecture by Zhong-Qiu Wang covers multichannel enhancement.
- *Microphone Array Signal Processing*, Jacob Benesty, Jingdong Chen, and Yiteng Huang, Springer, 2008. Chapter 3 is delay-and-sum and MVDR. Chapter 9 is time-difference estimation.
- [Linear and Parametric Microphone Array Processing](https://www.audiolabs-erlangen.de/fau/professor/habets/activities/ICASSP-2013/), Emanuel Habets and Sharon Gannot, ICASSP 2013 tutorial.
- [Defeating Ambient Noise](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ivantash-tashev_icassp06_tutorial_defeating_ambient_noise.pdf), Ivan Tashev, ICASSP 2006 tutorial. Delay-and-sum, and GCC-PHAT.
- [Sound: Speed of Sound](http://english.ioa.cas.cn/psk/201410/t20141027_130148.html), Institute of Acoustics, 27 October 2014. The speed in air at $20\,^\circ\mathrm{C}$ is $343$ m/s.
- [audio_processing_statistics.h](https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing_statistics.h), WebRTC. ERLE is $10 \log_{10}(P_{\mathrm{echo}} / P_{\mathrm{out}})$.
