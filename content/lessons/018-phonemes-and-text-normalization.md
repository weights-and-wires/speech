---
title: Phonemes and text normalization
slug: 018-phonemes-and-text-normalization
part: "Part 2 — Hearing and speech production"
summary: Turn written text into the sounds a speaker makes, by spelling out numbers, then looking words up in a dictionary with a rule fallback.
minutes: 40
gpu: false
objectives:
  - Explain the difference between a phoneme and a phone, and convert ARPAbet with stress marks to IPA.
  - Normalize text for speech by spelling out cardinals, ordinals, years, decimals, percentages, and money.
  - Convert words to phones by dictionary lookup with a longest-match letter-to-sound fallback.
  - Say why a rule-based fallback gets some words wrong, and measure it by phone error rate.
prereqs:
  - 015-the-source-filter-model
exercise: 018-phonemes-and-text-normalization
---

# Phonemes and text normalization

A text-to-speech system receives "The 21st voice costs $3.50." Nobody says "dollar sign three point five zero". People say "the twenty first voice costs three dollars fifty cents", and a model that makes sound needs those words, and then the sounds that make them up. A speech recognizer meets the same problem in reverse: it can only be scored fairly when its transcript and the reference are written the same way.

So how does a machine get from characters to sounds? Spelling is no guide on its own: "cough", "though", and "through" share four letters and no ending sound. The answer takes three steps. Normalize the text into the words a person would say, look each word up in a pronunciation dictionary, and, for a word the dictionary lacks, guess its sounds from its letters.

Figure 1 shows the first two steps on one sentence. This chapter builds all three, and shows exactly where the third one goes wrong.

```figure
<svg viewBox="0 0 600 288" role="img" aria-label="The sentence The 21st voice costs $3.50 split into five tokens, each rewritten as spoken words, with the IPA of each word">
<defs><marker id="c018-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" class="fill-muted"/></marker></defs>
<text x="14" y="22" class="fig-bold">1. Text</text>
<rect x="14" y="32" width="112" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="70.0" y="52" text-anchor="middle" class="fig-mono">The</text>
<line x1="70.0" y1="63" x2="70.0" y2="100" class="stroke-muted" marker-end="url(#c018-arrow)"/>
<rect x="14" y="104" width="112" height="46" rx="4" class="fill-soft-b stroke-b"/>
<text x="22" y="130" class="fig-small">the</text>
<text x="22" y="145" class="fig-small fig-mono">/ðə/</text>
<rect x="130" y="32" width="112" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="186.0" y="52" text-anchor="middle" class="fig-mono">21st</text>
<line x1="186.0" y1="63" x2="186.0" y2="100" class="stroke-muted" marker-end="url(#c018-arrow)"/>
<rect x="130" y="104" width="112" height="80" rx="4" class="fill-soft-b stroke-b"/>
<text x="138" y="130" class="fig-small">twenty</text>
<text x="138" y="145" class="fig-small fig-mono">/twˈɛnti/</text>
<text x="138" y="164" class="fig-small">first</text>
<text x="138" y="179" class="fig-small fig-mono">/fˈɝst/</text>
<rect x="246" y="32" width="112" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="302.0" y="52" text-anchor="middle" class="fig-mono">voice</text>
<line x1="302.0" y1="63" x2="302.0" y2="100" class="stroke-muted" marker-end="url(#c018-arrow)"/>
<rect x="246" y="104" width="112" height="46" rx="4" class="fill-soft-b stroke-b"/>
<text x="254" y="130" class="fig-small">voice</text>
<text x="254" y="145" class="fig-small fig-mono">/vˈɔɪs/</text>
<rect x="362" y="32" width="112" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="418.0" y="52" text-anchor="middle" class="fig-mono">costs</text>
<line x1="418.0" y1="63" x2="418.0" y2="100" class="stroke-muted" marker-end="url(#c018-arrow)"/>
<rect x="362" y="104" width="112" height="46" rx="4" class="fill-soft-b stroke-b"/>
<text x="370" y="130" class="fig-small">costs</text>
<text x="370" y="145" class="fig-small fig-mono">/kˈɑsts/</text>
<rect x="478" y="32" width="112" height="30" rx="4" class="fill-soft-a stroke-a"/>
<text x="534.0" y="52" text-anchor="middle" class="fig-mono">$3.50</text>
<line x1="534.0" y1="63" x2="534.0" y2="100" class="stroke-muted" marker-end="url(#c018-arrow)"/>
<rect x="478" y="104" width="112" height="148" rx="4" class="fill-soft-b stroke-b"/>
<text x="486" y="130" class="fig-small">three</text>
<text x="486" y="145" class="fig-small fig-mono">/θɹˈi/</text>
<text x="486" y="164" class="fig-small">dollars</text>
<text x="486" y="179" class="fig-small fig-mono">/dˈɑlɚz/</text>
<text x="486" y="198" class="fig-small">fifty</text>
<text x="486" y="213" class="fig-small fig-mono">/fˈɪfti/</text>
<text x="486" y="232" class="fig-small">cents</text>
<text x="486" y="247" class="fig-small fig-mono">/sˈɛnts/</text>
<text x="14" y="278" class="fig-bold">2. Normalize, then look up each word: its spelling and its IPA</text>
</svg>
Figure 1. Normalization and dictionary lookup on one sentence. Notice that two tokens each become several words: 21st becomes twenty first, and $3.50 becomes four words. Every one of the nine words is in the dictionary here, so no spelling rules are needed.
```

## What you already know

- A vowel is identified by its formants, which come from the shape of the vocal tract.
- Python's `re` module finds the tokens in a string that match a regular expression.
- A Python `dict` looks a key up in constant time.
- Recursion solves a number by solving a smaller number, as in `divmod`.

## Phones and phonemes

A *phoneme* is a sound category that distinguishes words in a language. English /p/ and /b/ are different phonemes, because "pat" and "bat" are different words. A *phone* is the actual sound a speaker makes. One phoneme can come out as several phones, called its *allophones*, depending on context: English /p/ has a puff of air in "pin" and none in "spin", and American /t/ becomes a quick tap in "butter". Speakers rarely notice allophones, but a spectrogram shows them plainly.

Sounds also blend into their neighbors. Your lips round for the "oo" in "soon" while you are still saying the "s". This *coarticulation* means one phoneme has different acoustics in different words, which is why recognizers later in the course model phones in context.

## ARPAbet and IPA

The *International Phonetic Alphabet* (IPA) writes the sounds of every language, with symbols such as ʃ for the "sh" in "ship". Those symbols are awkward to type, so American English speech technology uses *ARPAbet*, an ASCII phone set. The CMU Pronouncing Dictionary (CMUdict) uses 39 ARPAbet phonemes and has about 135,000 entries, some of them variant pronunciations of the same word. Each vowel carries a stress digit: 0 for unstressed, 1 for primary stress, 2 for secondary. The entry for "speech" is:

```text
SPEECH  S P IY1 CH
```

Most consonants map to the IPA letter you expect. The following table lists the vowels and the consonants whose symbols change.

| ARPAbet | IPA | Example | ARPAbet | IPA | Example |
|---|---|---|---|---|---|
| AA | ɑ | father | CH | tʃ | chin |
| AE | æ | cat | DH | ð | this |
| AH | ʌ, or ə unstressed | but, about | JH | dʒ | jump |
| AO | ɔ | dog | NG | ŋ | sing |
| AW | aʊ | sound | R | ɹ | red |
| AY | aɪ | night | SH | ʃ | ship |
| EH | ɛ | red | TH | θ | thin |
| ER | ɝ, or ɚ unstressed | bird, butter | Y | j | yes |
| EY | eɪ | cake | ZH | ʒ | measure |
| IH | ɪ | pin | G | ɡ | go |
| IY | i | sheep | HH | h | hat |
| OW | oʊ | boat | | | |
| OY | ɔɪ | coin | | | |
| UH | ʊ | book | | | |
| UW | u | moon | | | |

Two vowels change symbol when unstressed: AH0 is the *schwa* ə, and ER0 is ɚ. This chapter writes a stress mark, ˈ for primary and ˌ for secondary, right before the stressed vowel, which keeps each phone's symbols together. So `HH AH0 L OW1` becomes *həlˈoʊ*. Dictionaries put the mark at the start of the syllable instead: /həˈloʊ/.

## Normalize the text first

The obvious approach is to replace each digit with its word. It fails at once, because the same token reads differently in context. "1995" is "nineteen ninety five" as a year and "one thousand nine hundred ninety five" as a count, and "$3.50" has a dollar sign in front and a decimal point in the middle that no digit-by-digit replacement can read. So *text normalization* rewrites each non-standard token by a rule that looks at the whole token. Production systems use hand-written grammars compiled to finite-state transducers, or neural models constrained by such grammars. This chapter uses a few rules that cover the common cases:

- **Cardinals.** Build words from ones, tens, hundreds, thousands, and millions, with no hyphens and no "and": 12,345 is "twelve thousand three hundred forty five".
- **Ordinals.** Change only the last word: one to first, two to second, three to third, five to fifth, eight to eighth, nine to ninth, twelve to twelfth, a final "y" to "ieth", and anything else gains "th". So 21st is "twenty first", and 40th is "fortieth".
- **Years.** A plain four-digit number from 1100 to 2099 reads in pairs: "nineteen ninety five", "nineteen oh five", "nineteen hundred". The years 2000 to 2009 read as cardinals, "two thousand five".
- **Decimals, percentages, and money.** "3.14" is "three point one four", and "50%" is "fifty percent". "$3.50" is "three dollars fifty cents", with "dollar" and "cent" in the singular for exactly one.

Work the sentence by hand, left to right:

1. Lowercase it: "the 21st voice costs $3.50." Then find the tokens: `the`, `21st`, `voice`, `costs`, `$3.50`. The final period matches no pattern and is dropped.
2. `21st` has an ordinal suffix. The cardinal of 21 is "twenty one", so its last word becomes "first".
3. `$3.50` is money. The whole part, 3, gives "three dollars", and the first two decimal digits, 50, give "fifty cents".

The result is "the twenty first voice costs three dollars fifty cents". One regular expression finds all the tokens, with money tried first, so the `$` is not left behind. Whisper uses the inverse direction for scoring: it ships an `EnglishTextNormalizer` so that "twenty one" and "21" count as the same word, as [Whisper and large-scale recognition](/lessons/035-whisper) explains.

## Look the words up, then guess

*Grapheme-to-phoneme* (G2P) conversion turns spelling into phones. A *grapheme* is a letter or group of letters that writes one sound, such as "sh". English spelling is irregular, so G2P starts with a dictionary: look the word up in CMUdict and use the listed pronunciation. That is correct whenever the word is in it.

Words the dictionary lacks, such as names and coined words, need a fallback. The oldest is *letter-to-sound rules*, a table from graphemes to phones. The obvious version maps one letter to one sound, and it fails on "ship": it says S, H, I, P, when "sh" is one sound. So at each position the rules take the *longest* grapheme that matches, emit its phones, and move past it. Some rules apply only at the end of a word, such as a silent final "e", written `e$`.

Take "nation" with the exercise's rules, as in Figure 2. At "n" the longest match is `n`, giving N. At "a" it is `a`, giving AE1. At "tion" the four-letter rule `tion` beats `t`, giving SH AH0 N. The result is `N AE1 SH AH0 N`, but CMUdict says `N EY1 SH AH0 N`. The rules cannot see that the "a" is long.

```figure
<svg viewBox="0 0 600 318" role="img" aria-label="Longest-match letter-to-sound scans of ship, nation, and cake, compared with the dictionary pronunciation">
<text x="10" y="18" class="fig-small">Each box is the longest rule that matches at that point. Below it, the phones the rules emit are compared with the dictionary.</text>
<text x="10" y="64" class="fig-bold">ship</text>
<rect x="100" y="44" width="56" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="128.0" y="62" text-anchor="middle" class="fig-mono">sh</text>
<rect x="164" y="44" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="181.0" y="62" text-anchor="middle" class="fig-mono">i</text>
<rect x="206" y="44" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="223.0" y="62" text-anchor="middle" class="fig-mono">p</text>
<text x="10" y="88" class="fig-small">rules emit</text>
<text x="10" y="106" class="fig-small">dictionary</text>
<text x="100" y="88" class="fig-small fig-mono">SH</text>
<text x="100" y="106" class="fig-small fig-mono fill-c">SH</text>
<text x="160" y="88" class="fig-small fig-mono">IH1</text>
<text x="160" y="106" class="fig-small fig-mono fill-c">IH1</text>
<text x="220" y="88" class="fig-small fig-mono">P</text>
<text x="220" y="106" class="fig-small fig-mono fill-c">P</text>
<text x="10" y="156" class="fig-bold">nation</text>
<rect x="100" y="136" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="117.0" y="154" text-anchor="middle" class="fig-mono">n</text>
<rect x="142" y="136" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="159.0" y="154" text-anchor="middle" class="fig-mono">a</text>
<rect x="184" y="136" width="100" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="234.0" y="154" text-anchor="middle" class="fig-mono">tion</text>
<text x="10" y="180" class="fig-small">rules emit</text>
<text x="10" y="198" class="fig-small">dictionary</text>
<text x="100" y="180" class="fig-small fig-mono">N</text>
<text x="100" y="198" class="fig-small fig-mono fill-c">N</text>
<text x="160" y="180" class="fig-small fig-mono">AE1</text>
<text x="160" y="198" class="fig-small fig-mono fill-bad">EY1</text>
<text x="220" y="180" class="fig-small fig-mono">SH</text>
<text x="220" y="198" class="fig-small fig-mono fill-c">SH</text>
<text x="280" y="180" class="fig-small fig-mono">AH0</text>
<text x="280" y="198" class="fig-small fig-mono fill-c">AH0</text>
<text x="340" y="180" class="fig-small fig-mono">N</text>
<text x="340" y="198" class="fig-small fig-mono fill-c">N</text>
<text x="10" y="248" class="fig-bold">cake</text>
<rect x="100" y="228" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="117.0" y="246" text-anchor="middle" class="fig-mono">c</text>
<rect x="142" y="228" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="159.0" y="246" text-anchor="middle" class="fig-mono">a</text>
<rect x="184" y="228" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="201.0" y="246" text-anchor="middle" class="fig-mono">k</text>
<rect x="226" y="228" width="34" height="26" rx="3" class="fill-soft-a stroke-a"/>
<text x="243.0" y="246" text-anchor="middle" class="fig-mono">e$</text>
<text x="10" y="272" class="fig-small">rules emit</text>
<text x="10" y="290" class="fig-small">dictionary</text>
<text x="100" y="272" class="fig-small fig-mono">K</text>
<text x="100" y="290" class="fig-small fig-mono fill-c">K</text>
<text x="160" y="272" class="fig-small fig-mono">AE1</text>
<text x="160" y="290" class="fig-small fig-mono fill-bad">EY1</text>
<text x="220" y="272" class="fig-small fig-mono">K</text>
<text x="220" y="290" class="fig-small fig-mono fill-c">K</text>
</svg>
Figure 2. The rules get ship right, but they put a short AE1 in nation and cake where the dictionary has a long EY1. Notice that the rules never look past the letter group in front of them, so they cannot see that the final e or the following letters change the vowel.
```

The *phone error rate* (PER) scores a G2P system: the edit distance between the predicted and true phone sequences, divided by the length of the true one. Edit distance counts the fewest insertions, deletions, and substitutions that turn one sequence into the other, which [Word error rate](/lessons/024-word-error-rate) builds in full. Here "nation" has one substitution in five phones, a PER of 20%. On 46 held-out CMUdict words, with stress digits removed, the rules make 18 phone errors in 155 phones, a PER of 11.6%. Twelve of the 46 words have an error, and they are the ones that need more than one letter of context: the silent "e" that lengthens the vowel in "cake", "time", "home", and "phone", the "gh" in "laugh", and the "kn" in "knife".

Better fallbacks learn from the dictionary itself. Joint-sequence models such as Sequitur G2P and neural sequence-to-sequence models read the whole word before they write any phone. None of them solves *homographs*, words spelled alike but said differently, such as "read", "live", and "record". Those need the sentence.

> [!KEY]
> Text reaches a speech model in three steps: normalize non-standard words into spoken words, look each word up in a pronunciation dictionary, and fall back to rules or a learned model for words the dictionary lacks. The fallback is the weak step, because it cannot see beyond a few letters.

> [!RECAP]
> - CMUdict writes 39 ARPAbet phonemes with stress digits, and AH0 and ER0 are the unstressed ə and ɚ in IPA.
> - Normalize text into spoken words: cardinals, ordinals from the last word, years in pairs, decimals, percentages, and money.
> - G2P looks words up first and falls back to longest-match letter-to-sound rules.
> - Phone error rate is the edit distance over the true length, and the rules reach 11.6% on 46 held-out words.

## Build it step by step

The checks test the functions in this order. The lexicon, the IPA table, the rules, and the number words are provided.

1. **`strip_stress(phones)`** strips the digits 0, 1, and 2 from every phone with `str.rstrip`.
2. **`arpabet_to_ipa(phones)`** splits each phone into its base and stress, handles AH0 and ER0, adds the stress mark, and joins. The usual mistake is looking up `IY1` in the table, which is keyed by `IY`.
3. **`number_to_words(n)`** recurses on under 20, under 100, under 1000, and then millions and thousands with `divmod`. The usual mistake is "twenty zero" for 20.
4. **`ordinal_to_words(n)`** and **`year_to_words(n)`** rewrite the result of `number_to_words`. The usual mistake is making "twelve" into "twelveth".
5. **`normalize_text(text)`** finds tokens with one regular expression, money first, and dispatches each. The usual mistake is letting the comma in "1995," join the number; match thousands commas only as `,\d{3}`.
6. **`letter_to_sound(word, rules)`** sorts the rule keys by grapheme length, with `$` rules first on a tie, and scans the word.
7. **`g2p(word, lexicon)`** tries the lexicon first and then the rules, and **`text_to_phones(text, lexicon)`** chains normalization and `g2p`.

## Where this breaks

**Ambiguous text.** "St." is "saint" or "street", and "2/3" is a date or a fraction. Rules need context, and production normalizers carry large grammars for it.

**Names and borrowed words.** English letter-to-sound rules fail on words from other languages. Extend the lexicon by hand for the names you need.

> [!WARNING]
> Normalize the reference and the hypothesis the same way before you score speech recognition. "Twenty one" against "21" counts as two errors, and a word error rate computed on raw text measures the normalizer, not the recognizer.

## The exercise

You build `strip_stress`, `arpabet_to_ipa`, `number_to_words`, `ordinal_to_words`, `year_to_words`, `normalize_text`, `letter_to_sound`, `g2p`, and `text_to_phones`. The lexicon of CMUdict entries, the IPA table, the letter-to-sound rules, and the number words are provided. Press **Run** to see three sentences normalized and converted to IPA, and the rules applied to five words.

The checks verify IPA with stress, tables of cardinals, ordinals, and years, and eleven normalization cases. They test dictionary lookup and the fallback, exact rule outputs, a phone error rate of at most 15% on held-out CMUdict words, and a whole sentence turned into phones.
