---
title: Anti-spoofing and watermarking
slug: 075-anti-spoofing-and-watermarking
part: "Part 8 — Beyond recognition and synthesis"
summary: Build LFCC features, score a spoof detector with min t-DCF, mark audio with a spread-spectrum watermark, and test anonymization with an attacker.
minutes: 55
gpu: false
objectives:
  - Build a linear-frequency filterbank and LFCC features, and say why a spoof detector wants resolution at high frequencies.
  - Compute the ASVspoof 2019 minimum t-DCF from stated costs and priors, and say why its best threshold differs from the EER threshold.
  - Embed and detect a spread-spectrum watermark, and measure its bit error rate under noise and a low-pass filter.
  - Explain why a watermark proves where audio came from, and what a missing watermark does not prove.
  - Evaluate speaker anonymization by the equal error rate of an attacker, for an ignorant and an informed attacker.
prereqs:
  - 014-mfccs
  - 042-speaker-verification
  - 074-voice-conversion-and-spoofing
exercise: 075-anti-spoofing-and-watermarking
---

# Anti-spoofing and watermarking

A synthetic voice can sound like a specific person after a few seconds of their speech. That helps a voice assistant and endangers a bank that confirms identity by voice. The previous chapter scored a spoof detector by its error rates. This chapter covers the defenses: features a detector can use, a cost that judges it together with the verifier it protects, a mark that generated audio carries from the start, and a test of whether anonymizing a voice hides the speaker. Each defense has a limit, and the chapter states it.

## What you already know

- The mel filterbank, a log, and a DCT turn a spectrum into cepstral coefficients, from [MFCCs](/lessons/014-mfccs).
- A spoof countermeasure outputs a score per recording, and its EER is where false rejection equals false acceptance, from [Voice conversion and spoofing](/lessons/074-voice-conversion-and-spoofing).
- A speaker verification system scores trials by cosine similarity, from [Speaker embeddings and verification](/lessons/042-speaker-verification).

## Linear filters for spoof detection

MFCCs spend their filters on low frequencies, where speech carries its words. With 20 mel filters, only 5 centers sit above 4 kHz. Vocoder artifacts, such as a slightly wrong high band, live where those filters are wide and few. *LFCCs*, linear-frequency cepstral coefficients, use triangular filters spaced equally in hertz, and they served as a baseline in the ASVspoof challenges. With 20 filters, 10 centers sit above 4 kHz.

```figure
<svg viewBox="0 0 600 190" role="img" aria-label="Centers of 20 linear and 20 mel filters from 0 to 8 kilohertz; the linear filters cover 5.5 to 6.5 kilohertz with three centers, the mel filters with one">
  <rect x="400" y="30" width="60" height="105" class="fill-soft-b"/>
  <line x1="70" y1="135" x2="550" y2="135" class="stroke-muted"/>
  <text x="70" y="153" text-anchor="middle" class="fig-small">0</text>
  <text x="190" y="153" text-anchor="middle" class="fig-small">2</text>
  <text x="310" y="153" text-anchor="middle" class="fig-small">4</text>
  <text x="430" y="153" text-anchor="middle" class="fig-small">6</text>
  <text x="550" y="153" text-anchor="middle" class="fig-small">8</text>
  <text x="70" y="51" class="fig-small fig-bold">linear (LFCC)</text>
  <circle cx="92.9" cy="65" r="4" class="fill-a"/>
  <circle cx="115.7" cy="65" r="4" class="fill-a"/>
  <circle cx="138.6" cy="65" r="4" class="fill-a"/>
  <circle cx="161.4" cy="65" r="4" class="fill-a"/>
  <circle cx="184.3" cy="65" r="4" class="fill-a"/>
  <circle cx="207.1" cy="65" r="4" class="fill-a"/>
  <circle cx="230.0" cy="65" r="4" class="fill-a"/>
  <circle cx="252.9" cy="65" r="4" class="fill-a"/>
  <circle cx="275.7" cy="65" r="4" class="fill-a"/>
  <circle cx="298.6" cy="65" r="4" class="fill-a"/>
  <circle cx="321.4" cy="65" r="4" class="fill-a"/>
  <circle cx="344.3" cy="65" r="4" class="fill-a"/>
  <circle cx="367.1" cy="65" r="4" class="fill-a"/>
  <circle cx="390.0" cy="65" r="4" class="fill-a"/>
  <circle cx="412.9" cy="65" r="4" class="fill-a"/>
  <circle cx="435.7" cy="65" r="4" class="fill-a"/>
  <circle cx="458.6" cy="65" r="4" class="fill-a"/>
  <circle cx="481.4" cy="65" r="4" class="fill-a"/>
  <circle cx="504.3" cy="65" r="4" class="fill-a"/>
  <circle cx="527.1" cy="65" r="4" class="fill-a"/>
  <text x="70" y="96" class="fig-small fig-bold">mel (MFCC)</text>
  <circle cx="75.4" cy="110" r="4" class="fill-c"/>
  <circle cx="81.4" cy="110" r="4" class="fill-c"/>
  <circle cx="88.2" cy="110" r="4" class="fill-c"/>
  <circle cx="95.9" cy="110" r="4" class="fill-c"/>
  <circle cx="104.5" cy="110" r="4" class="fill-c"/>
  <circle cx="114.3" cy="110" r="4" class="fill-c"/>
  <circle cx="125.3" cy="110" r="4" class="fill-c"/>
  <circle cx="137.7" cy="110" r="4" class="fill-c"/>
  <circle cx="151.7" cy="110" r="4" class="fill-c"/>
  <circle cx="167.4" cy="110" r="4" class="fill-c"/>
  <circle cx="185.2" cy="110" r="4" class="fill-c"/>
  <circle cx="205.3" cy="110" r="4" class="fill-c"/>
  <circle cx="227.9" cy="110" r="4" class="fill-c"/>
  <circle cx="253.4" cy="110" r="4" class="fill-c"/>
  <circle cx="282.1" cy="110" r="4" class="fill-c"/>
  <circle cx="314.5" cy="110" r="4" class="fill-c"/>
  <circle cx="351.0" cy="110" r="4" class="fill-c"/>
  <circle cx="392.2" cy="110" r="4" class="fill-c"/>
  <circle cx="438.6" cy="110" r="4" class="fill-c"/>
  <circle cx="491.0" cy="110" r="4" class="fill-c"/>
  <text x="310" y="176" text-anchor="middle" class="fig-small">filter center frequency (kHz)</text>
</svg>
Figure 1. Centers of 20 filters from 0 to 8 kHz. The shaded band is 5.5 to 6.5 kHz, which three linear filters cover and one mel filter covers.
```

Only the edges change. For $M$ filters, take $M + 2$ edges from 0 Hz to $f_s/2$ in equal steps. Filter $m$ rises from edge $m$ to edge $m+1$ and falls to edge $m+2$, with a peak of 1. The rest is the MFCC recipe: filter energies, a natural log, and an orthonormal DCT-II down the filter axis.

```python
edges = np.linspace(0.0, rate / 2, n_filters + 2)
lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
fb = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
ceps = dct(np.log(fb @ power + 1e-10), axis=0, norm="ortho")[:n_ceps]
```

### Work a three-filter bank by hand

Take a 16-point FFT at 16 kHz, so the 9 bins are 1 kHz apart, and $M = 3$ filters with edges at 0, 2, 4, 6, and 8 kHz. Filter 0 peaks at 2 kHz, so its weights over bins 0 to 8 are $(0, \tfrac12, 1, \tfrac12, 0, 0, 0, 0, 0)$. Filters 1 and 2 are the same triangle shifted by two and four bins. Every filter sums to 2, so a flat power spectrum of ones gives three energies of 2. The DCT of a constant vector has one nonzero coefficient, $c_0 = \sqrt{3}\ln 2 = 1.201$.

### A toy detector

To see the resolution argument, draw 160 synthetic voices: harmonic tones with random pitch, tilt, and noise level. The spoof of each voice is the same voice with its 5.5 to 6.5 kHz band lowered by 4% in amplitude, a 0.35 dB dip. Average 19 coefficients (without $c_0$) over each recording and fit a logistic regression on 80 pairs. On the other 80 pairs, the EER is 6.2% with 20 LFCCs and 21.2% with 20 mel filters. This is one seed on synthetic signals: an illustration, not a benchmark.

Real systems score with a Gaussian mixture model per class, or a network such as AASIST (Jung et al., 2022). All of them learn the attacks in their training data, so report results on held-out attacks.

## Scoring a detector with its verifier

The ASVspoof 2019 protocol splits recordings into training, development, and evaluation sets, and the evaluation set contains attack systems the training set lacks. The organizers supply the scores of a fixed speaker verification system (ASV), so every team is judged against the same verifier. The metric is the minimum *tandem detection cost function* (t-DCF), with EER as a secondary number (Kinnunen et al., 2020).

The t-DCF prices three errors in a pipeline where a trial passes only if the countermeasure (CM) and the ASV both accept it: rejecting a real customer costs $C_\text{miss}$, and accepting an impostor or a spoof costs $C_\text{fa}$. As a function of the CM threshold, the cost is a constant plus two weighted CM errors:

$$
\text{t-DCF} = C_0 + C_1 P^\text{cm}_\text{miss} + C_2 P^\text{cm}_\text{fa}
$$

Here $P^\text{cm}_\text{miss}$ is the false rejection rate of bona fide speech, and $P^\text{cm}_\text{fa}$ is the false acceptance rate of spoofs. The weights bring in the verifier's behavior:

$$
C_1 = \pi_\text{tar} C_\text{miss}(1 - P^\text{asv}_\text{miss}) - \pi_\text{non} C_\text{fa} P^\text{asv}_\text{fa}, \qquad C_2 = \pi_\text{spoof} C_\text{fa} (1 - P^\text{asv}_\text{miss,spoof})
$$

Read it aloud: $C_1$ is the cost of rejecting a target the verifier would have accepted, less a small credit for the impostors that rejection blocks, and $C_2$ is the cost of passing a spoof that the verifier then accepts. $C_0$ does not depend on the CM, so this chapter drops it and divides by $\min(C_1, C_2)$, the cost of the better trivial CM, which accepts or rejects everything. A score of 1 means the CM adds nothing. The Kinnunen paper also discusses a default cost that keeps $C_0$, so state your normalization.

### Work a minimum t-DCF by hand

Use the ASVspoof 2019 priors and costs: $\pi_\text{spoof} = 0.05$, $\pi_\text{tar} = 0.95 \times 0.99 = 0.9405$, $\pi_\text{non} = 0.95 \times 0.01 = 0.0095$, $C_\text{miss} = 1$, and $C_\text{fa} = 10$. Give the verifier a miss rate of 0.05, a false alarm rate of 0.01, and a spoof miss rate of 0.3, chosen for this chapter. Then $C_1 = 0.9405 \times 0.95 - 0.0095 \times 10 \times 0.01 = 0.892525$ and $C_2 = 10 \times 0.05 \times 0.7 = 0.35$. The normalizer is $\min(C_1, C_2) = 0.35$, so the cost is $2.550\,P^\text{cm}_\text{miss} + P^\text{cm}_\text{fa}$.

Reuse the scores from the previous chapter: bona fide $(0.9, 0.5, 0.4)$ and spoof $(0.6, 0.3, 0.1, 0.0)$.

| Threshold | FRR | FAR | Normalized cost |
|---|---|---|---|
| 0.00 | 0 | 1.00 | 1.000 |
| 0.10 | 0 | 0.75 | 0.750 |
| 0.30 | 0 | 0.50 | 0.500 |
| 0.40 | 0 | 0.25 | 0.250 |
| 0.50 | 0.333 | 0.25 | 1.100 |
| 0.60 | 0.667 | 0.25 | 1.950 |
| 0.90 | 0.667 | 0 | 1.700 |
| 0.91 | 1 | 0 | 2.550 |

The minimum is 0.25 at threshold 0.40, where the previous chapter's EER threshold was 0.475. They differ because rejecting a real customer costs 2.55 times as much per unit of error as admitting a spoof, so the best threshold stays lenient. The first row, accept-everything, costs exactly 1.

> [!KEY]
> The EER treats both errors as equally bad. The t-DCF prices them by how often each trial occurs and what each costs, with the verifier's errors folded in. The minimum also assumes an oracle threshold chosen on the evaluation data, which a deployed system can't do. State the priors, costs, and normalization.

## Marking audio at the source

Detection starts after the audio exists. A *watermark* starts earlier: the generator adds a hidden, keyed pattern to everything it produces, and a detector with the key checks for it later. This is provenance, and it works only for generators that cooperate.

A classical design is *spread spectrum*. Give each bit a random sequence of $+1$ and $-1$ values, called *chips*, that only the key holder can regenerate. Add the chips to the host at a small amplitude $a$, with the sign set by the bit. To read a bit, correlate the received segment with the same chips and take the sign. Host and noise look random to the chips, so their correlation grows like $\sqrt{L}$ for $L$ chips, while the watermark's grows like $L$. That is the processing gain: more chips per bit buy reliability and cost capacity.

### Work one bit by hand

Take host samples $x = (0.8, 0.7, 0.9, 0.8)$ and chips $c = (+1, +1, -1, +1)$. Embed the bit 0 with $a = 0.1$, so $y = x - 0.1\,c = (0.7, 0.6, 1.0, 0.7)$. A detector that correlates directly gets $\sum y_i c_i = 0.7 + 0.6 - 1.0 + 0.7 = 1.0$, which is positive, and reads a 1. That is wrong: the host's mean of 0.8 multiplied the chip sum of 2 and drowned the mark. Subtract the mean of $y$, 0.75, to get $(-0.05, -0.15, 0.25, -0.05)$, and the correlation is $-0.5$, which reads 0.

### Attack it and measure

The *bit error rate* (BER) is the fraction of decoded bits that are wrong. The Run block embeds 32 bits, 500 chips each, at amplitude 0.01 into a 1-second host, 17.6 dB below it. Figure 2 averages the BER over 20 keys.

```figure
<svg viewBox="0 0 620 270" role="img" aria-label="Bit error rate of a spread-spectrum watermark against added noise and against a low-pass filter">
  <line x1="60" y1="200" x2="290" y2="200" class="stroke-muted"/><line x1="60" y1="50" x2="60" y2="200" class="stroke-muted"/>
  <line x1="60" y1="200.0" x2="290" y2="200.0" class="stroke-grid"/><text x="54" y="204.0" text-anchor="end" class="fig-small">0.00</text>
  <line x1="60" y1="125.0" x2="290" y2="125.0" class="stroke-grid"/><text x="54" y="129.0" text-anchor="end" class="fig-small">0.25</text>
  <line x1="60" y1="50.0" x2="290" y2="50.0" class="stroke-grid"/><text x="54" y="54.0" text-anchor="end" class="fig-small">0.50</text>
  <path d="M70.0 200.0 L105.0 200.0 L140.0 193.9 L175.0 171.9 L210.0 141.9 L245.0 107.7 L280.0 84.7" fill="none" class="stroke-b" stroke-width="2"/>
  <circle cx="70.0" cy="200.0" r="3.5" class="fill-b"/><text x="70.0" y="218" text-anchor="middle" class="fig-small">20</text>
  <circle cx="105.0" cy="200.0" r="3.5" class="fill-b"/><text x="105.0" y="218" text-anchor="middle" class="fig-small">10</text>
  <circle cx="140.0" cy="193.9" r="3.5" class="fill-b"/><text x="140.0" y="218" text-anchor="middle" class="fig-small">0</text>
  <circle cx="175.0" cy="171.9" r="3.5" class="fill-b"/><text x="175.0" y="218" text-anchor="middle" class="fig-small">-5</text>
  <circle cx="210.0" cy="141.9" r="3.5" class="fill-b"/><text x="210.0" y="218" text-anchor="middle" class="fig-small">-10</text>
  <circle cx="245.0" cy="107.7" r="3.5" class="fill-b"/><text x="245.0" y="218" text-anchor="middle" class="fig-small">-15</text>
  <circle cx="280.0" cy="84.7" r="3.5" class="fill-b"/><text x="280.0" y="218" text-anchor="middle" class="fig-small">-20</text>
  <text x="175" y="244" text-anchor="middle" class="fig-small">added noise, SNR (dB)</text>
  <line x1="360" y1="200" x2="590" y2="200" class="stroke-muted"/><line x1="360" y1="50" x2="360" y2="200" class="stroke-muted"/>
  <line x1="360" y1="200.0" x2="590" y2="200.0" class="stroke-grid"/><text x="354" y="204.0" text-anchor="end" class="fig-small">0.00</text>
  <line x1="360" y1="125.0" x2="590" y2="125.0" class="stroke-grid"/><text x="354" y="129.0" text-anchor="end" class="fig-small">0.25</text>
  <line x1="360" y1="50.0" x2="590" y2="50.0" class="stroke-grid"/><text x="354" y="54.0" text-anchor="end" class="fig-small">0.50</text>
  <path d="M370.0 199.1 L422.5 177.5 L475.0 125.9 L527.5 93.6 L580.0 68.8" fill="none" class="stroke-b" stroke-width="2"/>
  <circle cx="370.0" cy="199.1" r="3.5" class="fill-b"/><text x="370.0" y="218" text-anchor="middle" class="fig-small">7</text>
  <circle cx="422.5" cy="177.5" r="3.5" class="fill-b"/><text x="422.5" y="218" text-anchor="middle" class="fig-small">4</text>
  <circle cx="475.0" cy="125.9" r="3.5" class="fill-b"/><text x="475.0" y="218" text-anchor="middle" class="fig-small">2</text>
  <circle cx="527.5" cy="93.6" r="3.5" class="fill-b"/><text x="527.5" y="218" text-anchor="middle" class="fig-small">1</text>
  <circle cx="580.0" cy="68.8" r="3.5" class="fill-b"/><text x="580.0" y="218" text-anchor="middle" class="fig-small">0.5</text>
  <text x="475" y="244" text-anchor="middle" class="fig-small">low-pass cutoff (kHz)</text>
  <text x="16" y="125" class="fig-small" transform="rotate(-90 16 125)" text-anchor="middle">bit error rate</text></svg>
Figure 2. Bit error rate of the watermark. Noise, at the SNR of the marked clip, leaves it intact down to 0 dB. A low-pass filter hurts sooner, because it removes watermark energy but not host energy.
```

With no attack, the BER is 0, and noise at 20 dB and 10 dB leaves it at 0. At 0 dB it is 0.02. A low-pass at 4 kHz, the kind of band limit a lossy codec applies, raises it to 0.075, and one at 1 kHz gives 0.355. Down-sampling to 8 kHz and back by linear interpolation gives 0.077. Chips are white, so a 4 kHz cutoff removes about half the watermark's energy but none of the host's low-frequency energy, which is the interference.

Neural watermarks learn that shaping. AudioSeal (San Roman et al., 2024) trains a generator that adds the mark and a detector that reads it, with a loss for localizing the mark to individual samples and a perceptual loss that keeps it masked. WavMark (Chen et al., 2023) hides up to 32 bits in each second of audio. Both are more robust than this sketch, but they need their trained detector, and an attacker with access to it can try to remove the mark.

> [!WARNING]
> A watermark shows that a cooperating generator made the audio. A missing watermark proves nothing: the generator might not mark, or the mark might be degraded. Never treat a failed detection as proof that a recording is real.

## Anonymization and the attacker's EER

Anonymization hides who is speaking while keeping the words. A common design swaps the speaker embedding for a *pseudo-speaker*, the mean of several vectors from a pool of other speakers, and synthesizes speech from it. The VoicePrivacy challenges (Tomashenko et al., 2020, 2022) define the test: an *attacker* tries to link anonymized trials to enrolled speakers, and you report the attacker's EER, where 0 means perfect linking and 0.5 means guessing. The challenges also measure utility, such as ASR WER, because replacing every voice with noise scores perfect privacy.

An *ignorant* attacker enrolls speakers from original speech. A *lazy-informed* attacker knows about the anonymization and enrolls from anonymized speech. Take 60 synthetic speakers with 16-dimensional vectors and two noisy utterances each. With no anonymization, the attacker's EER is 2.2%. Swapping each trial for the mean of 20 random pool vectors raises it to 51.7% against an ignorant attacker.

If the system gives each person one fixed pseudo-speaker, the lazy-informed attacker links them again: the EER falls to 1.7%. A fresh pseudo-speaker for every utterance keeps it at 54.5%, but the voices no longer share an identity across a conversation, which breaks diarization.

> [!KEY]
> A privacy claim names its attacker. An EER measured against an ignorant attacker says little about an attacker who knows how the voice was changed, and a score of 0.5 bought by destroying the speech is not a result.

> [!RECAP]
> - LFCCs are MFCCs with equally spaced triangular filters, which give a spoof detector more resolution above 4 kHz.
> - The min t-DCF prices CM errors by priors, costs, and the verifier's errors, normalized by $\min(C_1, C_2)$. Its best threshold can differ from the EER threshold.
> - A spread-spectrum watermark adds a keyed chip sequence per bit and reads it by correlation after removing the segment mean.
> - Anonymization is evaluated by an attacker's EER, and the result depends on what the attacker knows.

## Build it step by step

Build the functions in the order the checks test them.

1. **`lfcc_filterbank(n_filters, n_fft, sample_rate)`** returns `(n_filters, 1 + n_fft // 2)`: equally spaced edges, then triangles with `np.minimum` and `np.maximum`. The common mistake is mel-spaced edges.
2. **`lfcc(power, filterbank, n_ceps)`** returns `(n_ceps, frames)`: the bank, a natural log of the energy plus `1e-10`, and an orthonormal DCT-II down axis 0.
3. **`tdcf_costs(asv_pmiss, asv_pfa, asv_pmiss_spoof, ...)`** returns `(c1, c2)`. The common mistake is using the spoof miss rate where $1 - P^\text{asv}_\text{miss,spoof}$ belongs.
4. **`min_tdcf(bonafide, spoof, c1, c2)`** returns `(cost, threshold)` over the distinct scores plus one larger value. The common mistake is the wrong normalizer, which makes a useless detector look excellent.
5. **`embed_watermark(x, bits, key, strength)`** adds `strength * (2 * bit - 1) * chips` to each segment. The common mistake is using the bit as 0 or 1, which adds no mark for a 0.
6. **`detect_watermark(y, n_bits, key)`** subtracts each segment's mean, correlates with the chips, and compares to 0. The common mistake is skipping the mean.
7. **`anonymize_embeddings(vectors, pool, k, rng)`** averages `k` pool vectors, drawn without replacement, for every row. The common mistake is mixing in the input, which leaks the speaker.
8. **`attacker_eer(enrolled, trials)`** scores every pair by cosine, takes the diagonal as target trials and the rest as non-target, and returns the EER. The common mistake is leaving the diagonal among the non-target scores.

## The exercise

You build `lfcc_filterbank` and `lfcc`, `tdcf_costs` and `min_tdcf`, `embed_watermark` and `detect_watermark`, and `anonymize_embeddings` and `attacker_eer`. Press **Run** to embed 32 bits, attack them, and print the bit errors, then the attacker's EER before and after anonymization.

The checks compare the filterbank with the closed-form triangles, `lfcc` with a flat-spectrum answer, the t-DCF weights and this chapter's table, and the watermark on a host with a DC offset, under 20 dB noise and a 4 kHz low-pass. The last checks need the attacker's EER low before the swap and near 0.5 after it.

## Further reading

- Kinnunen et al., "Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals," IEEE/ACM TASLP, 2020 (https://arxiv.org/abs/2007.05979).
- Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019 (https://arxiv.org/abs/1904.05441).
- Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP 2022 (https://arxiv.org/abs/2110.01200).
- San Roman et al., "Proactive Detection of Voice Cloning with Localized Watermarking," ICML 2024 (https://arxiv.org/abs/2401.17264).
- Chen et al., "WavMark: Watermarking for Audio Generation," 2023 (https://arxiv.org/abs/2308.12770).
- Tomashenko et al., "Introducing the VoicePrivacy Initiative," 2020 (https://arxiv.org/abs/2005.01387), and "The VoicePrivacy 2020 Challenge: Results and findings," 2022 (https://arxiv.org/abs/2109.00648).
