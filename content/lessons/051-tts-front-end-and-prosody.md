---
title: The TTS front end and prosody
slug: 051-tts-front-end-and-prosody
part: "Part 5 — Speech synthesis"
summary: Pick the right pronunciation from context, add pitch and energy embeddings, and steer style with tokens, rate, and speaker prompts.
minutes: 50
gpu: false
objectives:
  - Choose a homograph's pronunciation from its neighbors with a rule layer on top of a lexicon.
  - Build FastSpeech 2's variance adaptor, with log-spaced pitch bins and embeddings added before length regulation.
  - Predict how a pitch scale and a speaking rate change bin indices and frame counts.
  - "Mix a bank of style tokens by attention from a reference embedding."
  - Explain when a speaker embedding is enough and when zero-shot cloning needs a speaker prompt.
prereqs:
  - 019-phonemes-and-text-normalization
  - 049-text-to-speech-and-tacotron
  - 050-durations-and-alignment-search
exercise: 051-tts-front-end-and-prosody
---

# The TTS front end and prosody

A voice assistant reads "I have read the record" and says "reed" for the first verb. A second assistant reads it with the right sounds, but the pitch is flat and every word takes the same time, so it sounds like a bored announcer. A third clones a voice from three seconds of audio and loses the speaker's accent. These are three different failures, and a text-to-speech system fixes each one at a different place.

The first failure belongs to the *front end*, the code that turns text into phones before any neural network runs. The second belongs to *prosody*: the pitch, loudness, and timing that carry meaning and mood beyond the phones. The third belongs to *conditioning*, the way a model learns whose voice and which style to use.

This chapter builds a rule layer for the first, the full FastSpeech 2 variance adaptor with pitch and rate knobs for the second, and style tokens for the third. It then compares speaker embeddings with speaker prompts.

## What you already know

- A normalizer turns "$5" into words, and a dictionary with a letter-to-sound fallback turns words into phones (see [Phonemes and text normalization](/lessons/019-phonemes-and-text-normalization)).
- Tacotron reads text and predicts mel frames, and a TTS system is a pipeline of text, phones, mel, and waveform (see [Text to speech and Tacotron](/lessons/049-text-to-speech-and-tacotron)).
- A length regulator copies each phone's vector for its duration, and pitch is quantized into 256 log-spaced bins (see [Durations and alignment search](/lessons/050-durations-and-alignment-search)).
- A softmax turns scores into weights that are positive and sum to one.

## The front end: choosing a pronunciation in context

A pronunciation dictionary maps a spelling to phones, and that fails for a *homograph*, one spelling with two pronunciations. "Read" is /riːd/ in "I will read" and /rɛd/ in "I have read". "Lead" is a verb in "lead the way" and a metal in "lead pipe". "Record" is stressed on the first syllable as a noun and on the second as a verb. Lookup alone can't choose between the entries.

The choice depends on grammar: a word after "have" is a past participle, and a word after "to" is a verb. A production front end runs a part-of-speech tagger. This chapter uses a cheaper stand-in, a *rule layer*: each homograph has a default pronunciation and an ordered list of rules, and each rule names cue words that must sit before it, after it, or both.

For "I have read the record", the rule for "read" asks whether the previous word is in {have, had, was}. It is "have", so the answer is "R EH1 D". For "record", the previous word "the" matches no rule, so the default noun pronunciation wins. In "I will read to record", "will" is not a past cue, so "read" keeps its present default, and "to" fires the verb rule for "record".

The front end runs in a fixed order: normalize the text, check the homograph table, then the lexicon, and then fall back to letter-to-sound for words in neither. Production systems make the fallback a neural grapheme-to-phoneme model, a small sequence-to-sequence network. They keep the homograph layer, because a model that sees one word can't know its neighbors.

*SSML*, the Speech Synthesis Markup Language, lets the author skip the guessing. Tags such as `<say-as>` say how to read a number, `<break>` inserts a pause, `<phoneme>` gives phones directly, and `<prosody rate pitch>` sets speed and pitch. The front end turns each tag into a control this chapter builds: fixed phones, a pause token, a duration scale, and a pitch scale.

> [!KEY]
> A dictionary answers "how is this word spelled", and context answers "which word is this". A front end needs both, and a cheap rule layer over neighboring words fixes the common homographs.

## The variance adaptor in full

A phone sequence leaves much unsaid. The same phones can be fast or slow, high or low, loud or soft, and a decoder trained with squared error on all those variants learns their average, which sounds flat. FastSpeech 2 therefore predicts three signals per phone, durations plus *pitch* and *energy*, and feeds them back in. These predictors form the *variance adaptor*.

Picture dials on a mixing desk. Each phone arrives with a hidden vector $h_i$. The adaptor turns that phone's pitch and energy into bin numbers, looks up a learned embedding for each bin, and adds both to $h_i$. Then the length regulator copies the result into frames.

During training, the recording's true pitch and energy set the dials. At inference, small convolutional predictors guess them from $h_i$. Because you can overwrite a guess, the dials become controls.

### Work one example by hand

Take three phones with durations $[2, 1, 3]$, pitches $[210, 300, 450]$ Hz, and energies $[1, 3, 7]$. Use 8 pitch bins between 100 and 800 Hz, spaced evenly in log frequency, so there are 7 edges. Each edge is half an octave above the previous one: 100, 141.4, 200, 282.8, 400, 565.7, 800 Hz. The bin number counts the edges below the value, so 210, 300, and 450 Hz land in bins 3, 4, and 5.

Energy uses 7 evenly spaced edges from 0 to 8, which are 0, 1.33, 2.67, 4, 5.33, 6.67, and 8. Energies 1, 3, and 7 land in bins 1, 3, and 6. Say the pitch table row for bin $k$ is $[k, 0]$ and the energy row is $[0, k]$, and take $h = [[1,0],[0,1],[1,1]]$. Phone 1 becomes $[1,0] + [3,0] + [0,1] = [4, 1]$, phone 2 becomes $[4, 4]$, and phone 3 becomes $[6, 7]$. Length regulation copies them 2, 1, and 3 times, giving six frames.

The formula says the same thing in symbols:

$$
\tilde h_i = h_i + E_p\big[\operatorname{bin}_p(s \cdot p_i)\big] + E_e\big[\operatorname{bin}_e(e_i)\big],
\qquad
\text{frames} = \operatorname{LR}\big(\tilde h,\ \operatorname{round}(d / r)\big).
$$

Read it aloud: scale the pitch $p_i$ by $s$, find its bin, and add that bin's row of the embedding table $E_p$ to $h_i$. Do the same for energy with table $E_e$. Then length-regulate, using durations $d$ divided by a rate $r$. In torch:

```python
p_edges = torch.exp(torch.linspace(np.log(v_min), np.log(v_max), K - 1))
bins = torch.bucketize(pitch * pitch_scale, p_edges)   # (B, L), long
h = h + pitch_table[bins] + energy_table[energy_bins]  # still (B, L, D)
frames, mask = length_regulate(h, scaled_durations)
```

> [!WARNING]
> Add the embeddings before length regulation when pitch and energy are per phone, as here. After regulation the sequence has $N$ frames, but $L$ phone-level indices don't fit it: the shapes mismatch, or with padding they misalign. A frame-level variant adds after regulation, but it needs frame-level targets.

## Pitch and rate as knobs

Because the pitch enters through a lookup, scaling it is one multiplication before the bucket step. The bin spacing decides what that does. With log-spaced bins, multiplying every pitch by $s$ adds $\ln s$ to every log pitch, so every bin index moves by the same number of bins. The bin width is $\ln(795/71)/254 = 0.00951$, so $s = 1.2$ moves each pitch by $\ln 1.2 / 0.00951 = 19.2$ bins, which is 19 or 20 after rounding. A semitone is about 6.07 bins.

Linear bins fail this test. A 1.2 scale adds $0.2p$ hertz, which shifts a 100 Hz voice by 7 bins and a 500 Hz voice by 35, as Figure 1 shows.

```figure
<svg viewBox="0 0 600 260" role="img" aria-label="Bins moved by a 1.2 pitch scale: about 19 for log-spaced bins at every pitch, but growing from 7 to 35 for linear bins">
<line x1="70.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-muted"/>
<line x1="70.0" y1="200.0" x2="70.0" y2="40.0" class="stroke-muted"/>
<line x1="70.0" y1="200.0" x2="540.0" y2="200.0" class="stroke-grid"/>
<text x="62.0" y="204.0" text-anchor="end" class="fig-small">0</text>
<line x1="70.0" y1="160.0" x2="540.0" y2="160.0" class="stroke-grid"/>
<text x="62.0" y="164.0" text-anchor="end" class="fig-small">10</text>
<line x1="70.0" y1="120.0" x2="540.0" y2="120.0" class="stroke-grid"/>
<text x="62.0" y="124.0" text-anchor="end" class="fig-small">20</text>
<line x1="70.0" y1="80.0" x2="540.0" y2="80.0" class="stroke-grid"/>
<text x="62.0" y="84.0" text-anchor="end" class="fig-small">30</text>
<line x1="70.0" y1="40.0" x2="540.0" y2="40.0" class="stroke-grid"/>
<text x="62.0" y="44.0" text-anchor="end" class="fig-small">40</text>
<text x="70.0" y="218.0" text-anchor="middle" class="fig-small">100</text>
<text x="187.5" y="218.0" text-anchor="middle" class="fig-small">200</text>
<text x="305.0" y="218.0" text-anchor="middle" class="fig-small">300</text>
<text x="422.5" y="218.0" text-anchor="middle" class="fig-small">400</text>
<text x="540.0" y="218.0" text-anchor="middle" class="fig-small">500</text>
<text x="305.0" y="240.0" text-anchor="middle" class="fig-small">pitch before scaling (Hz)</text>
<text x="14" y="120.0" text-anchor="middle" class="fig-small" transform="rotate(-90 14 120.0)">bins moved</text>
<polyline points="70.0,124.0 128.8,124.0 187.5,120.0 246.2,124.0 305.0,124.0 363.8,124.0 422.5,124.0 481.2,124.0 540.0,124.0" fill="none" class="stroke-b" stroke-width="2.5"/>
<circle cx="70.0" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="128.8" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="187.5" cy="120.0" r="3.5" class="fill-b"/>
<circle cx="246.2" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="305.0" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="363.8" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="422.5" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="481.2" cy="124.0" r="3.5" class="fill-b"/>
<circle cx="540.0" cy="124.0" r="3.5" class="fill-b"/>
<polyline points="70.0,172.0 128.8,156.0 187.5,144.0 246.2,128.0 305.0,116.0 363.8,100.0 422.5,88.0 481.2,72.0 540.0,60.0" fill="none" class="stroke-a" stroke-width="2.5"/>
<circle cx="70.0" cy="172.0" r="3.5" class="fill-a"/>
<circle cx="128.8" cy="156.0" r="3.5" class="fill-a"/>
<circle cx="187.5" cy="144.0" r="3.5" class="fill-a"/>
<circle cx="246.2" cy="128.0" r="3.5" class="fill-a"/>
<circle cx="305.0" cy="116.0" r="3.5" class="fill-a"/>
<circle cx="363.8" cy="100.0" r="3.5" class="fill-a"/>
<circle cx="422.5" cy="88.0" r="3.5" class="fill-a"/>
<circle cx="481.2" cy="72.0" r="3.5" class="fill-a"/>
<circle cx="540.0" cy="60.0" r="3.5" class="fill-a"/>
<text x="305.0" y="110.0" text-anchor="middle" class="fig-small fig-bold">log bins: 19 or 20 everywhere</text>
<text x="534.0" y="52.0" text-anchor="end" class="fig-small fig-bold">linear bins</text>
</svg>
Figure 1. Moving every pitch up by a factor of 1.2 shifts log-spaced bins by 19 or 20 whatever the pitch, but linear bins by an amount proportional to the pitch. The values come from scripts/figures/051-tts-front-end-and-prosody.py.
```

To convert a musical shift into a scale, use $s = 2^{k/12}$ for $k$ semitones: 12 gives 2, and 7 gives 1.498. A pitch of 0 means unvoiced and stays 0 after scaling, so unvoiced phones stay in bin 0.

Rate works on durations. A rate $r$ means $r$ times faster, so each duration becomes $\operatorname{round}(d/r)$, never below 1. With $d = [4, 1, 6]$ and $r = 2$ the raw values are $[2, 0.5, 3]$, and the middle one rounds to 0 and clamps to 1, giving $[2, 1, 3]$. Total frames never rise as $r$ rises.

> [!TIP]
> Test a control with a number it must produce. "Raise pitch by 1.2" implies "every bin moves by 19 or 20", which a wrong bin spacing breaks at once.

A knob works only inside the range the model saw. A pitch past the last edge clips into bin 255, and the output stops following the knob.

## Style: from one average to a bank of tokens

Pitch and energy are two named dials, but style has more. A cheerful voice and a tired voice differ in rhythm, pitch range, and breathiness together, and no label says which moved. Skerry-Ryan and colleagues fed a *reference encoder* a clip, compressed it to one vector, and gave that vector to Tacotron, so the output copied the clip's prosody. It copies everything in the clip, including the speaker.

Wang and colleagues added a bank of $T$ learned vectors, the *global style tokens* (GST). The reference vector doesn't enter the decoder directly. It attends over the bank, and the decoder gets the weighted mix. This forces the model to describe any style as a blend of a few learned directions, and the tokens often settle on things like speaking rate or pitch range without labels. The paper's scoring uses tanh keys:

$$
w = \operatorname{softmax}\!\left(\frac{\tanh(T)\, r}{\sqrt{D}}\right), \qquad \text{style} = w^\top T.
$$

Read it aloud: score each token's tanh against the reference $r$, scale by $\sqrt D$ so large vectors don't saturate the softmax, and use the weights to mix the raw tokens. The following numpy lines do it, with a max-subtraction so that large scores don't overflow:

```python
logits = np.tanh(tokens) @ ref / np.sqrt(tokens.shape[1])
w = np.exp(logits - logits.max()); w /= w.sum()
```

Take four tokens $3 e_0, \ldots, 3 e_3$ and a reference $[0, 0, 4, 0]$. Token 2's logit is $4 \tanh(3)/2 = 1.990$ and the others are 0. The weights are $e^{1.990} / (e^{1.990} + 3) = 0.709$ for token 2 and $0.097$ for each other token, and the style vector is $[0.291, 0.291, 2.128, 0.291]$.

```figure
<svg viewBox="0 0 600 220" role="img" aria-label="Attention weights over four style tokens: 0.709 on token 2 and 0.097 on each of the others">
<line x1="90" y1="170.0" x2="510" y2="170.0" class="stroke-muted"/>
<rect x="120" y="154.2" width="60" height="15.8" rx="3" class="fill-soft-a stroke-a"/>
<text x="150" y="148.2" text-anchor="middle" class="fig-small fig-bold">0.097</text>
<text x="150" y="188.0" text-anchor="middle" class="fig-small">token 0</text>
<rect x="220" y="154.2" width="60" height="15.8" rx="3" class="fill-soft-a stroke-a"/>
<text x="250" y="148.2" text-anchor="middle" class="fig-small fig-bold">0.097</text>
<text x="250" y="188.0" text-anchor="middle" class="fig-small">token 1</text>
<rect x="320" y="54.8" width="60" height="115.2" rx="3" class="fill-b"/>
<text x="350" y="48.8" text-anchor="middle" class="fig-small fig-bold">0.709</text>
<text x="350" y="188.0" text-anchor="middle" class="fig-small">token 2</text>
<rect x="420" y="154.2" width="60" height="15.8" rx="3" class="fill-soft-a stroke-a"/>
<text x="450" y="148.2" text-anchor="middle" class="fig-small fig-bold">0.097</text>
<text x="450" y="188.0" text-anchor="middle" class="fig-small">token 3</text>
<text x="300" y="206" text-anchor="middle" class="fig-small">attention weight (sums to 1)</text>
</svg>
Figure 2. One reference vector attends over four style tokens and mostly selects token 2. The weights sum to one. The values come from scripts/figures/051-tts-front-end-and-prosody.py.
```

At inference you can skip the reference and set the weights by hand, which turns a token into a control. StyleTTS 2 takes the opposite route: it treats the style vector as a latent variable and generates it from the text with a diffusion model, so it needs no reference audio.

> [!KEY]
> A reference encoder copies prosody, and a token bank turns that copy into a small set of shared, adjustable directions. Attention is the mixer.

## Speaker embeddings versus speaker prompts

To clone a voice, the model needs the speaker's identity. Two designs exist.

| | Speaker embedding | Speaker prompt |
|---|---|---|
| What it is | One fixed-size vector from a speaker encoder, or a lookup row per training speaker | A few seconds of the speaker's audio, as mel frames or codec tokens, placed in the model's context |
| How the model uses it | Adds or concatenates it with the hidden states, like the pitch embedding above | Reads it as a prefix and continues the same voice |
| What it preserves | The overall timbre the encoder was trained to keep | Timbre, accent, recording channel, and speaking habits in the clip |
| Zero-shot limit | An unseen voice must lie in the span of voices the encoder learned | Works for any voice the model can imitate in context |

A vector of a few hundred numbers is a bottleneck that drops accent, room sound, and rhythm. A prompt keeps them, because the model sees the actual frames. That is why zero-shot systems moved to prompts. VALL-E continues a prompt of codec tokens as a language model (see [Codec language models](/lessons/059-codec-language-models)), and F5-style models fill in masked mel frames around a prompt (see [Flow matching for speech](/lessons/066-flow-matching-for-speech)). Prompting is the same idea as the reference encoder, with the compression removed.

A prompt clones a voice from a few seconds of audio, so cloning needs consent. For detection and risk, see [Voice conversion and spoofing](/lessons/071-voice-conversion-and-spoofing).

## Build it step by step

1. **`semitones_to_scale(semitones)`** returns $2^{k/12}$. The usual mistake is dividing by 12 after the power instead of inside it.
2. **`scale_durations(durations, rate)`** takes integer frames and returns integers: `np.maximum(1, np.round(d / rate))`. The usual mistake is multiplying by the rate, which makes fast speech long.
3. **`choose_pronunciation(word, prev, nxt, entry)`** walks the rules in order and returns the first whose every cue matches, comparing lower case. A cue never matches a missing neighbor. The usual mistake is firing a rule when only one of its two cues matches.
4. **`g2p_with_homographs(words, lexicon, homographs, fallback)`** checks the homograph table, then the lexicon, then `fallback`, using each word's neighbors in the list. The usual mistake is looking up the capitalized word, which misses the table.
5. **`style_attention(ref, tokens)`** returns the style vector and the weights, with tanh keys and the $\sqrt D$ scale. The usual mistake is a softmax without the max subtraction, which overflows.
6. **`variance_adaptor(...)`** buckets the scaled pitch with log edges and the energy with linear edges, adds both embeddings to `h`, scales durations, and length-regulates. The previous chapter's regulator is provided. The usual mistake is adding after regulation, or using linear pitch bins.

## The exercise

You write the six functions above in `starter.py`. The checks cover the rate and semitone arithmetic, the style weights on the hand-worked example and on huge logits, the homograph table and the rule-matching logic, and the variance adaptor on the three-phone example. They also check that a 1.2 pitch scale shifts every log bin by 19 or 20, and that gradients reach `h` and the used table rows, using finite differences. **Run** plays a six-phone pitch contour twice, once as given and once four semitones up and 1.25 times faster.

## Further reading

- Ren and colleagues, "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020), for the variance adaptor.
- Wang and colleagues, "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis" (2018), for global style tokens.
- Skerry-Ryan and colleagues, "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron" (2018), for the reference encoder.
- Li and colleagues, "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models" (2023), for style as a latent variable.
- Jurafsky and Martin, *Speech and Language Processing*, third edition draft, the chapter "Text-to-Speech".
- Stanford CS224S, "Spoken Language Processing", for the course's treatment of speech synthesis.
