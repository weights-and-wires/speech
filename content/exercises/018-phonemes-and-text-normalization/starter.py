"""Phonemes and text normalization.

Turn text into phones. Build IPA conversion, a text normalizer for numbers,
ordinals, years, and money, and dictionary G2P with a letter-to-sound
fallback. The lexicon, the IPA table, the letter-to-sound rules, and the number
words are provided as data. Fill in the functions below, then press Run to see
sentences turned into words and phones, and Check to test your code.
"""

import re

# LEXICON holds a few dozen entries from the CMU Pronouncing Dictionary
# (the cmusphinx/cmudict data in the `cmudict` Python package 1.1.3, first
# pronunciation of each word), used under its BSD-style license, which asks
# that the following notice be kept with the entries:
#
#   Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer. The contents of
#      this file are deemed to be source code.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   This work was supported in part by funding from the Defense Advanced
#   Research Projects Agency, the Office of Naval Research and the National
#   Science Foundation of the United States of America, and by member companies
#   of the Carnegie Mellon Sphinx Speech Consortium.
#   THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY "AS IS" AND ANY
#   EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY NOR ITS EMPLOYEES
#   BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
LEXICON = {
    word: phones.split()
    for word, phones in {
        "a": "AH0", "the": "DH AH0", "and": "AH0 N D", "of": "AH1 V", "is": "IH1 Z",
        "it": "IH1 T", "to": "T UW1", "in": "IH0 N", "this": "DH IH1 S", "costs": "K AA1 S T S",
        "speech": "S P IY1 CH", "sound": "S AW1 N D", "voice": "V OY1 S", "pitch": "P IH1 CH",
        "hello": "HH AH0 L OW1", "world": "W ER1 L D",
        "zero": "Z IH1 R OW0", "one": "W AH1 N", "two": "T UW1", "three": "TH R IY1",
        "four": "F AO1 R", "five": "F AY1 V", "six": "S IH1 K S", "seven": "S EH1 V AH0 N",
        "eight": "EY1 T", "nine": "N AY1 N", "ten": "T EH1 N", "eleven": "IH0 L EH1 V AH0 N",
        "twelve": "T W EH1 L V", "thirteen": "TH ER1 T IY1 N", "fourteen": "F AO1 R T IY1 N",
        "fifteen": "F IH0 F T IY1 N", "sixteen": "S IH0 K S T IY1 N",
        "seventeen": "S EH1 V AH0 N T IY1 N", "eighteen": "EY0 T IY1 N",
        "nineteen": "N AY1 N T IY1 N", "twenty": "T W EH1 N T IY0", "thirty": "TH ER1 D IY2",
        "forty": "F AO1 R T IY0", "fifty": "F IH1 F T IY0", "sixty": "S IH1 K S T IY0",
        "seventy": "S EH1 V AH0 N T IY0", "eighty": "EY1 T IY0", "ninety": "N AY1 N T IY0",
        "hundred": "HH AH1 N D R AH0 D", "thousand": "TH AW1 Z AH0 N D", "million": "M IH1 L Y AH0 N",
        "dollar": "D AA1 L ER0", "dollars": "D AA1 L ER0 Z", "cent": "S EH1 N T",
        "cents": "S EH1 N T S", "point": "P OY1 N T", "percent": "P ER0 S EH1 N T", "oh": "OW1",
        "first": "F ER1 S T", "second": "S EH1 K AH0 N D", "third": "TH ER1 D",
        "fourth": "F AO1 R TH", "fifth": "F IH1 F TH", "sixth": "S IH1 K S TH",
        "seventh": "S EH1 V AH0 N TH", "eighth": "EY1 T TH", "ninth": "N AY1 N TH",
        "tenth": "T EH1 N TH", "eleventh": "IH0 L EH1 V AH0 N TH", "twelfth": "T W EH1 L F TH",
    }.items()
}

# The 39 CMUdict phonemes in IPA. AH and ER change when unstressed; see arpabet_to_ipa.
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ",
    "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s",
    "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Letter-to-sound rules for the fallback: grapheme -> phones. A key ending in
# "$" matches only at the end of a word.
RULES = {
    "tion": ["SH", "AH0", "N"], "tch": ["CH"], "igh": ["AY1"],
    "ch": ["CH"], "sh": ["SH"], "th": ["TH"], "ph": ["F"], "wh": ["W"], "ck": ["K"],
    "ng": ["NG"], "qu": ["K", "W"], "ee": ["IY1"], "ea": ["IY1"], "oo": ["UW1"],
    "ou": ["AW1"], "ow": ["OW1"], "oa": ["OW1"], "ai": ["EY1"], "ay": ["EY1"],
    "oi": ["OY1"], "oy": ["OY1"], "au": ["AO1"], "aw": ["AO1"], "ew": ["UW1"],
    "ar": ["AA1", "R"], "er": ["ER0"], "ir": ["ER1"], "ur": ["ER1"], "or": ["AO1", "R"],
    "ll": ["L"], "ss": ["S"], "tt": ["T"], "ff": ["F"], "pp": ["P"], "mm": ["M"],
    "nn": ["N"], "dd": ["D"], "bb": ["B"], "gg": ["G"], "rr": ["R"], "zz": ["Z"],
    "e$": [], "y$": ["IY0"],
    "a": ["AE1"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH1"], "f": ["F"], "g": ["G"],
    "h": ["HH"], "i": ["IH1"], "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"],
    "o": ["AA1"], "p": ["P"], "q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"], "u": ["AH1"],
    "v": ["V"], "w": ["W"], "x": ["K", "S"], "y": ["Y"], "z": ["Z"],
}

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


# Phones --------------------------------------------------------------------

def strip_stress(phones: list[str]) -> list[str]:
    """Remove the stress digit (0, 1, or 2) from every phone: ["IY1"] -> ["IY"]."""
    raise NotImplementedError


def arpabet_to_ipa(phones: list[str]) -> str:
    """Convert ARPAbet phones to one IPA string.

    Look each phone up in ARPABET_TO_IPA without its stress digit, except that
    AH0 becomes "ə" and ER0 becomes "ɚ". Put "ˈ" before a vowel with stress 1
    and "ˌ" before a vowel with stress 2. Join everything with no spaces:
    ["S", "P", "IY1", "CH"] -> "spˈitʃ".
    """
    raise NotImplementedError


# Numbers -------------------------------------------------------------------

def number_to_words(n: int) -> str:
    """A cardinal from 0 to 999,999,999 in lowercase words separated by spaces.

    Use ONES below 20 and TENS for the tens, with no hyphens and no "and":
    21 -> "twenty one", 105 -> "one hundred five",
    12345 -> "twelve thousand three hundred forty five", 2000000 -> "two million".
    """
    raise NotImplementedError


def ordinal_to_words(n: int) -> str:
    """An ordinal in words: change only the last word of number_to_words(n).

    one -> first, two -> second, three -> third, five -> fifth, eight -> eighth,
    nine -> ninth, twelve -> twelfth; a word ending in "y" swaps it for "ieth"
    (twenty -> twentieth); any other word adds "th" (four -> fourth,
    hundred -> hundredth). 21 -> "twenty first".
    """
    raise NotImplementedError


def year_to_words(n: int) -> str:
    """A year from 1000 to 2999 the way people say it.

    2000 to 2009, and any multiple of 1000, read as cardinals ("two thousand
    five"). Otherwise split into n // 100 and n % 100: a remainder of 0 gives
    "nineteen hundred", a remainder under 10 gives "nineteen oh five", and any
    other gives "nineteen ninety five" or "twenty twenty four".
    """
    raise NotImplementedError


def normalize_text(text: str) -> str:
    """Lowercase words separated by single spaces, ready for G2P.

    Lowercase the text, then read it as tokens, left to right, dropping
    everything else (punctuation, hyphens, spaces):
    - money: "$" and a number, which may have thousands commas and a decimal
      part. Whole dollars and cents (the first two decimal digits) in words,
      "dollar" or "dollars", "cent" or "cents", singular for exactly 1;
      leave out a part that is 0: "$3.50" -> "three dollars fifty cents",
      "$0.99" -> "ninety nine cents".
    - a number with thousands commas ("12,000"), a decimal part ("3.14" ->
      "three point one four"), or a suffix: "%" -> "... percent", and
      "st", "nd", "rd", or "th" -> the ordinal.
    - a plain four-digit number from 1100 to 2099 is a year; any other
      number is a cardinal.
    - a word: letters, which may hold apostrophes inside ("don't").
    """
    raise NotImplementedError


# Grapheme to phoneme ------------------------------------------------------------

def letter_to_sound(word: str, rules: dict = RULES) -> list[str]:
    """Phones for a word from letter-to-sound rules, a list of ARPAbet strings.

    The lowercased word is split, left to right, into the longest rule
    graphemes that match at each position, and the phones are the matched
    rules' phones in order. A key ending in "$" matches only if its grapheme
    ends the word, and it wins a tie with the plain key of the same grapheme.
    A character that no rule matches, such as an apostrophe, is skipped.
    """
    raise NotImplementedError


def g2p(word: str, lexicon: dict = LEXICON) -> list[str]:
    """The phones of one word: its entry in `lexicon` (keyed by lowercase word), else letter_to_sound."""
    raise NotImplementedError


def text_to_phones(text: str, lexicon: dict = LEXICON) -> list[list[str]]:
    """Phones for every word of `text` after text normalization: one list of phones per word."""
    raise NotImplementedError


if __name__ == "__main__":
    for text in ["The 21st voice costs $3.50.", "In 1995, 12,000 people heard it.", "Pitch rose 40% in 2024."]:
        words = normalize_text(text)
        phones = text_to_phones(text)
        ipa = " ".join(arpabet_to_ipa(p) for p in phones)
        print(f"{text!r}\n  -> {words}\n  -> /{ipa}/")
    for word in ["ship", "night", "cake", "phone", "nation"]:
        print(f"{word}: the rules give {' '.join(letter_to_sound(word))}")
