---
title: Phonetics, text, and voice activity
slug: 11-phonetics-text-and-voice-activity
part: "Part 2 — Hearing and speech production"
summary: Write sounds as phones, spell out numbers and money, look words up with a rule fallback, and find the speech in a noisy recording.
minutes: 65
gpu: false
objectives:
  - Explain phonemes, allophones, and coarticulation, and convert ARPAbet with stress to IPA.
  - Normalize text for speech by spelling out cardinals, ordinals, years, decimals, percentages, and money.
  - Convert words to phones by dictionary lookup with a longest-match letter-to-sound fallback, and measure its phone error rate.
  - Build an energy voice activity detector with a noise floor, hysteresis, and hangover, and score it by frame F1.
prereqs:
  - 01-sound-and-sampling
  - 04-the-short-time-fourier-transform
exercise: 11-phonetics-text-and-voice-activity
---

# Phonetics, text, and voice activity

A speech system meets two kinds of messy input. Text arrives with digits,
symbols, and abbreviations: "The 21st voice costs $3.50." Nobody says "dollar
sign three point five zero". Before a TTS model can speak it, the text must
become the words a person would say, and the words must become the sounds that
make them up.

Audio arrives with silence and noise around the speech. A recognizer that
transcribes the silence wastes compute and invents words. A phone call that
sends the silence wastes bandwidth. A *voice activity detector* (VAD) marks
which stretches of a recording hold speech.

This lesson covers both ends: the units linguists use to write speech sounds,
a text normalizer, a grapheme-to-phoneme converter, and a VAD built from frame
energy.

## What you already know

- A decibel is $10 \log_{10}$ of a power ratio, and the mean of $x^2$ over a frame is its power.
- A centered frame is padded by half its length at each end by reflection, so a signal of $N$ samples has $1 + N / H$ frames at hop $H$.
- A vowel's identity comes from its formants, and its pitch from the source.
- Voiced sounds are periodic; sounds like "s" and "f" are turbulent noise.
- Python's `re` module finds tokens that match a regular expression.

## Phones and phonemes

A *phoneme* is a sound category that distinguishes words in a language. English
/p/ and /b/ are different phonemes, because "pat" and "bat" are different
words. A *phone* is an actual sound as spoken. One phoneme can come out as
several phones, called its *allophones*, depending on context. English /p/ is
aspirated, with a puff of air, in "pin" and unaspirated in "spin". American /t/
becomes a quick tap in "butter". Speakers rarely notice allophones, but a
spectrogram shows them plainly.

Sounds also blend into their neighbors. Your lips round for the "oo" in "soon"
while you are still saying the "s". This *coarticulation* means one phoneme
has different acoustics in different words, which is why recognizers in Part 3
model phones in context.

Linguists describe consonants by three properties:

- *Place*: where the airflow is blocked, from the lips (bilabial /p/) to the
  throat (glottal /h/).
- *Manner*: how it is blocked. Stops close fully (/t/), fricatives leave a
  narrow gap (/s/), nasals send air through the nose (/m/), and approximants
  barely narrow the tract (/w/).
- *Voicing*: whether the vocal folds vibrate. /z/ is voiced, /s/ is not.

Vowels are described by the tongue's height and backness and by lip rounding,
which you met as $F_1$ and $F_2$.

## ARPAbet and IPA

The *International Phonetic Alphabet* (IPA) writes the sounds of every
language, with symbols such as ʃ for the "sh" in "ship" and θ for the "th" in
"thin". Those symbols are awkward to type, so American English speech
technology uses *ARPAbet*, an ASCII phone set. The CMU Pronouncing Dictionary
(CMUdict) uses 39 ARPAbet phonemes and lists about 126,000 words with 135,000
pronunciations. Each vowel carries a stress digit: 0 for unstressed, 1 for
primary stress, 2 for secondary. The following line is its entry for "speech":

```text
SPEECH  S P IY1 CH
```

Most consonants map to the IPA letter you expect. The following table lists
the vowels and the consonants whose symbols change.

| ARPAbet | IPA | Example | ARPAbet | IPA | Example |
|---|---|---|---|---|---|
| AA | ɑ | father | CH | tʃ | chin |
| AE | æ | cat | DH | ð | this |
| AH | ʌ, or ə unstressed | but, about | HH | h | hat |
| AO | ɔ | dog | JH | dʒ | jump |
| AW | aʊ | sound | NG | ŋ | sing |
| AY | aɪ | night | R | ɹ | red |
| EH | ɛ | red | SH | ʃ | ship |
| ER | ɝ, or ɚ unstressed | bird, butter | TH | θ | thin |
| EY | eɪ | cake | Y | j | yes |
| IH | ɪ | pin | ZH | ʒ | measure |
| IY | i | sheep | G | ɡ | go |
| OW | oʊ | boat | | | |
| OY | ɔɪ | coin | | | |
| UH | ʊ | book | | | |
| UW | u | moon | | | |

Two vowels change symbol when unstressed: AH0 is the *schwa* ə, and ER0 is ɚ.
This lesson writes a stress mark, ˈ for primary and ˌ for secondary, right
before the stressed vowel, which keeps each phone's symbols together. So
`HH AH0 L OW1` becomes *həlˈoʊ*. Dictionaries put the mark at the start of
the syllable instead: /həˈloʊ/.

## Text normalization

*Text normalization* rewrites non-standard words, such as numbers, symbols,
and abbreviations, as the words people say. It is harder than it looks because
the same token reads differently in context. "1995" is "nineteen ninety five"
as a year and "one thousand nine hundred ninety five" as a count. "Dr." is
"doctor" or "drive". Production systems use hand-written grammars compiled to
weighted finite-state transducers, such as NVIDIA NeMo's text processing, or
neural models constrained by such grammars. This lesson uses a few rules
that cover the common cases:

- **Cardinals.** Build words from ones, tens, hundreds, thousands, and millions,
  with no hyphens and no "and": 12,345 is "twelve thousand three hundred forty
  five".
- **Ordinals.** Change only the last word: one to first, two to second, three
  to third, five to fifth, eight to eighth, nine to ninth, twelve to twelfth,
  a final "y" to "ieth", and anything else gains "th". So 21st is "twenty
  first", and 40th is "fortieth".
- **Years.** A plain four-digit number from 1100 to 2099 reads in pairs:
  "nineteen ninety five", "nineteen oh five", "nineteen hundred". The years
  2000 to 2009 read as cardinals, "two thousand five".
- **Decimals, percentages, and money.** "3.14" is "three point one four".
  "50%" is "fifty percent". "$3.50" is "three dollars fifty cents", with
  "dollar" and "cent" in the singular for exactly one.

### Normalize a sentence by hand

Take "The 21st voice costs $3.50."

1. Lowercase: "the 21st voice costs $3.50."
2. Tokens, left to right: `the`, `21st`, `voice`, `costs`, `$3.50`. The final
   period matches no pattern and is dropped.
3. `21st` has an ordinal suffix. The cardinal of 21 is "twenty one", and its
   last word becomes "first".
4. `$3.50` is money: 3 dollars, and the first two decimal digits, 50, are
   cents.

The result is "the twenty first voice costs three dollars fifty cents".
Speech recognition needs the reverse direction too: it scores a transcript
only after both it and the reference are normalized the same way. Whisper
ships an `EnglishTextNormalizer` for that purpose; for more information, see
*Whisper and large-scale recognition*.

## Grapheme-to-phoneme conversion

*Grapheme-to-phoneme* (G2P) conversion turns spelling into phones. A
*grapheme* is a letter or a group of letters that writes one sound, such as
"sh". English spelling is irregular, so G2P starts with a dictionary: look the
word up in CMUdict, and use the listed pronunciation.

Words missing from the dictionary, such as names and new words, need a
fallback. The oldest is *letter-to-sound rules*: a table from graphemes to
phones, applied left to right. At each position, take the longest grapheme
that matches, emit its phones, and move past it. Some rules apply only at the
end of a word, such as a silent final "e", written `e$`.

Take "nation" with the rules in the exercise. The longest match at "n" is
`n`, giving N. At "a", the longest match is `a`, giving AE1. At "tion", the
four-letter rule `tion` beats `t`, giving SH AH0 N. The result is
`N AE1 SH AH0 N`. CMUdict says `N EY1 SH AH0 N`: the rules cannot see that
the "a" is long.

The *phone error rate* (PER) scores a G2P system: the edit distance between the
predicted and true phone sequences, divided by the length of the true one.
Part 3 builds that edit distance. Here, "nation" has one substitution in five
phones, a PER of 20%. On 46 held-out CMUdict words, the exercise's rules
reach 11.6% PER with stress removed. Their errors are the ones that need
more than one letter of context: the silent "e" in "cake" that lengthens the
"a", the "gh" in "laugh", and the "kn" in "knife".

Better fallbacks learn from the dictionary itself: joint-sequence models such
as Sequitur G2P, and neural sequence-to-sequence models that read the whole
word before they write any phone. None of them solves *homographs*, words spelled alike but said
differently, such as "read", "live", and "record". Those need the sentence.

> [!KEY]
> Text reaches a speech model in three steps: normalize non-standard words,
> look each word up in a pronunciation dictionary, and fall back to rules or a
> learned model for words the dictionary lacks.

## Voice activity detection

The simplest VAD uses loudness. Speech is louder than the background, so
compute each frame's energy in decibels:

$$
E_t = 10 \log_{10}\left(\frac{1}{N} \sum_{n=0}^{N-1} x_t[n]^2 + 10^{-10}\right)
$$

Read it aloud: square each of the $N$ samples in frame $t$, average them, add a
tiny constant so that silence gives $-100$ dB rather than minus infinity, and
convert to decibels. A sine of amplitude 0.5 has a mean power of 0.125, or
$-9.03$ dB.

```python
energy = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
```

A fixed threshold fails as soon as the microphone gain changes, so measure the
threshold against the recording's own *noise floor*. The 10th percentile of
the frame energies works when at least a tenth of the recording is silence.
Speech is then energy some margin above the floor.

One threshold makes a VAD flicker whenever the energy hovers near it.
*Hysteresis* uses two: switch on above a high threshold, and switch off only
below a lower one. A *hangover* keeps speech on for a few frames after the
energy drops, so the quiet end of a word is not cut off. This exercise switches
on at 9 dB above the floor, off at 5 dB above it, with a hangover of 50 ms.
Real speech, with its weak final consonants, usually needs 100 ms to 300 ms.

### Run the hysteresis by hand

Take energies $[0, 12, 8, 3, 3, 3, 9, 2, 2]$ dB, an on threshold of 10, an off
threshold of 5, and a hangover of 2 frames.

1. Frame 0 is 0 dB, not above 10: silence.
2. Frame 1 is 12 dB, above 10: speech starts.
3. Frame 2 is 8 dB. That is below 10 but not below 5, so speech continues.
4. Frames 3 and 4 are below 5: the quiet count reaches 1, then 2. That is not
   more than the hangover, so speech continues.
5. Frame 5 makes the count 3, more than 2: silence.
6. Frame 6 is 9 dB, above the off threshold but not the on threshold, so
   silence continues. Frames 7 and 8 stay silent.

The decisions are $[0, 1, 1, 1, 1, 0, 0, 0, 0]$. With no hysteresis, frame 6
would have switched speech back on.

### Score a VAD

Treat speech frames as the positive class. A *true positive* (TP) is a speech
frame marked speech, a *false positive* (FP) is a non-speech frame marked
speech, and a *false negative* (FN) is a speech frame missed. The frame F1
score combines precision and recall:

$$
F_1 = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}
$$

Read it aloud: twice the hits, over twice the hits plus both kinds of mistake.
It is 1 for a perfect detector and 0 for one that finds no speech. For
decisions $[1, 1, 0, 0, 1, 0]$ against the truth $[1, 0, 0, 1, 1, 0]$,
TP = 2, FP = 1, and FN = 1, so $F_1 = 4 / 6 = 0.667$.

Two widely used detectors do better than energy. WebRTC's VAD, from the
browser real-time audio stack, classifies frames of 10, 20, or 30 ms with
Gaussian mixture models over the energies of six bands between 80 Hz and
4 kHz. Silero VAD is a small neural
network that outputs a speech probability for each 512-sample chunk at
16 kHz, 32 ms, and marks speech above 0.5 by default.

> [!KEY]
> An energy VAD compares each frame with the recording's own noise floor, and
> hysteresis with a hangover keeps its decisions from flickering.

> [!RECAP]
> - CMUdict writes 39 ARPAbet phonemes with stress digits; AH0 and ER0 are the unstressed ə and ɚ in IPA.
> - Normalize text into spoken words: cardinals, ordinals from the last word, years in pairs, decimals, percentages, and money.
> - G2P looks words up first and falls back to longest-match letter-to-sound rules.
> - Frame energy is $10 \log_{10}$ of mean power; thresholds sit above a percentile noise floor.
> - Hysteresis switches on high and off low after a hangover; frame F1 is $2\text{TP} / (2\text{TP} + \text{FP} + \text{FN})$.

## Build it step by step

1. **`strip_stress(phones)`** strips the digits 0, 1, and 2 from every phone
   with `str.rstrip`.
2. **`arpabet_to_ipa(phones)`** splits each phone into its base and stress,
   handles AH0 and ER0, adds the stress mark, and joins. The usual mistake is
   looking up `IY1` in the table, which is keyed by `IY`.
3. **`number_to_words(n)`** recurses on under 20, under 100, under 1000, and
   then thousands and millions with `divmod`. The usual mistake is "twenty
   zero" for 20.
4. **`ordinal_to_words(n)`** and **`year_to_words(n)`** rewrite
   `number_to_words`. The usual mistake is making "twelve" into "twelveth".
5. **`normalize_text(text)`** finds tokens with one regular expression, money
   first, and dispatches each. The usual mistake is letting the comma in
   "1995," join the number; match thousands commas only as `,\d{3}`.
6. **`letter_to_sound(word, rules)`** sorts the rule keys by grapheme length,
   with `$` rules first on a tie, and scans the word. **`g2p(word, lexicon)`**
   tries the lexicon first. **`text_to_phones(text, lexicon)`** chains them.
7. **`frame_energy_db(x, frame_length, hop)`** frames with reflection padding
   and averages the squares. The usual mistake is $20 \log_{10}$ of a power.
8. **`hysteresis(energy_db, on_db, off_db, hangover)`** loops once over the
   frames with two state variables.
9. **`energy_vad(x, sample_rate, ...)`** converts milliseconds to samples and
   frames, finds the floor with `np.percentile`, and calls `hysteresis`.
10. **`frame_f1(pred, ref)`** counts with boolean masks.

## Cost and scaling

For a text of $L$ characters, a lexicon of $V$ words, $R$ rules, and an audio
clip of $T$ frames of $N$ samples, the steps cost the following.

| Step | Time | Memory |
|---|---|---|
| Normalization | $O(L)$ | $O(L)$ |
| Dictionary lookup | $O(1)$ per word in a hash table | $O(V)$; CMUdict is a few MB |
| Letter-to-sound rules | $O(R)$ per letter | $O(R)$ |
| Frame energies | $O(T N)$ | $O(T N)$ framed, or $O(N)$ streaming |
| Hysteresis and F1 | $O(T)$ | $O(1)$ state |

An energy VAD costs a few operations per sample. It also works one frame at a
time, except for the percentile floor. A live VAD tracks the floor with a slowly updated minimum instead.

## Where this breaks

- **Low SNR.** At 20 dB and 10 dB SNR, the exercise's VAD reaches a frame F1
  of 0.92 and 0.94 on the Run block's clip. At 5 dB it falls to 0.77, because
  quiet speech sinks into the noise floor.
- **Noise that changes.** A door slam or music is loud but not speech. Energy
  cannot tell them apart; spectral shape, periodicity, or a trained model can.
- **Quiet speech sounds.** Fricatives such as "f" and "th" have little energy.
  A hangover protects word endings, and a zero-crossing rate, the rate at
  which the signal changes sign, catches noisy sounds that energy misses.
- **Ambiguous text.** "St." is "saint" or "street", and "2/3" is a date or a
  fraction. Rules need context, and production normalizers carry large
  grammars for it.
- **Names and borrowed words.** English letter-to-sound rules fail on words
  from other languages. Extend the lexicon by hand for the names you need.

> [!WARNING]
> Normalize the reference and the hypothesis the same way before you score
> speech recognition. "Twenty one" against "21" counts as two errors, and a
> word error rate computed on raw text measures the normalizer, not the
> recognizer.

## The exercise

You build `strip_stress`, `arpabet_to_ipa`, `number_to_words`,
`ordinal_to_words`, `year_to_words`, `normalize_text`, `letter_to_sound`,
`g2p`, `text_to_phones`, `frame_energy_db`, `hysteresis`, `energy_vad`, and
`frame_f1`. The lexicon of CMUdict entries, the IPA table, the letter-to-sound
rules, and the number words are provided. Press **Run** to see three
sentences normalized and converted to IPA, see the rules on four words, and
hear and see the VAD on noisy synthetic speech at three SNRs.

The checks verify IPA with stress, tables of cardinals, ordinals, and years,
and eleven normalization cases. They test dictionary lookup and the fallback,
exact rule outputs, a phone error rate of at most 15% on held-out CMUdict
words, and a whole sentence to phones. They check frame energy against its
formula, work the hysteresis example, compute F1 by hand, and require a frame
F1 of at least 0.9 at 20 dB and 10 dB SNR on three noisy clips.
