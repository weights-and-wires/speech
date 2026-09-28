"""Phonetics, text, and voice activity.

Turn text into phones and find the speech in a noisy recording. Build IPA
conversion, a text normalizer for numbers, ordinals, years, and money,
dictionary G2P with a letter-to-sound fallback, and an energy voice activity
detector with hysteresis, scored by frame F1. The lexicon, the IPA table, the
letter-to-sound rules, and the number words are provided as data. Fill in the
functions below, then press Run to see sentences turned into phones and hear
the detector at work, and Check to test your code.
"""

import re

import numpy as np
from scipy.signal import lfilter

from speechlab import play, show_plot

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

    Lowercase the word and scan it left to right. At each position, use the
    rule whose grapheme is longest among those that match there; a key ending
    in "$" matches only if its grapheme ends the word, and wins a tie with the
    plain key of the same grapheme. Emit that rule's phones and move past the
    grapheme. Skip a character that no rule matches, such as an apostrophe.
    """
    raise NotImplementedError


def g2p(word: str, lexicon: dict = LEXICON) -> list[str]:
    """The phones of one word: its entry in `lexicon` (keyed by lowercase word), else letter_to_sound."""
    raise NotImplementedError


def text_to_phones(text: str, lexicon: dict = LEXICON) -> list[list[str]]:
    """normalize_text, then g2p on each word: one list of phones per word."""
    raise NotImplementedError


# Voice activity -------------------------------------------------------------------

def frame_energy_db(x: np.ndarray, frame_length: int = 400, hop: int = 160) -> np.ndarray:
    """Frame log energy in dB, shape (1 + len(x) // hop,).

    Pad x by frame_length // 2 samples at each end by reflection, take frame t
    as padded[t * hop : t * hop + frame_length], and return
    10 * log10(mean(frame ** 2) + 1e-10) for each frame.
    """
    raise NotImplementedError


def hysteresis(energy_db: np.ndarray, on_db: float, off_db: float, hangover: int) -> np.ndarray:
    """Speech decisions with two thresholds and a hangover, a bool array like `energy_db`.

    Start in silence. In silence, switch to speech at a frame whose energy is
    above on_db. In speech, count consecutive frames below off_db; a frame at
    or above off_db resets the count to 0; when the count exceeds `hangover`,
    switch to silence at that frame. Each output is the state after its frame.
    """
    raise NotImplementedError


def energy_vad(x, sample_rate=16000, frame_ms=25.0, hop_ms=10.0, on_margin_db=9.0, off_margin_db=5.0,
               hangover_ms=50.0, floor_percentile=10.0) -> np.ndarray:
    """An energy voice activity detector, one bool per hop, shape (1 + len(x) // hop,).

    Frame lengths are round(ms * sample_rate / 1000) samples. Compute
    frame_energy_db, take the noise floor as its `floor_percentile` percentile
    (np.percentile), and run hysteresis with on_db = floor + on_margin_db,
    off_db = floor + off_margin_db, and hangover round(hangover_ms / hop_ms) frames.
    """
    raise NotImplementedError


def frame_f1(pred: np.ndarray, ref: np.ndarray) -> float:
    """The F1 score of speech frames: 2 TP / (2 TP + FP + FN), or 1.0 when that denominator is 0."""
    raise NotImplementedError


if __name__ == "__main__":
    for text in ["The 21st voice costs $3.50.", "In 1995, 12,000 people heard it.", "Pitch rose 40% in 2024."]:
        words = normalize_text(text)
        phones = text_to_phones(text)
        ipa = " ".join(arpabet_to_ipa(p) for p in phones)
        print(f"{text!r}\n  -> {words}\n  -> /{ipa}/")
    for word in ["ship", "night", "cake", "phone"]:
        print(f"{word}: rules give {' '.join(letter_to_sound(word))}")

    rate = 16000

    def noisy_speech(seed, snr_db, seconds=3.0):
        """Voiced bursts through formant filters, in pink noise at snr_db; truth is one flag per 10 ms frame."""
        rng = np.random.default_rng(seed)
        n = int(seconds * rate)
        t = np.arange(n) / rate
        truth, speech, start = np.zeros(n, dtype=bool), np.zeros(n), 0.2
        while True:
            length = rng.uniform(0.2, 0.6)
            if start + length > seconds - 0.2:
                break
            span = (t >= start) & (t < start + length)
            truth |= span
            envelope = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0.0, 1.0) * span * rng.uniform(0.5, 1.0)
            f0 = rng.uniform(90.0, 250.0)
            phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))) / rate
            speech += envelope * sum(np.sin(k * phase) / k for k in range(1, int(4000 // f0)))
            start += length + rng.uniform(0.15, 0.5)
        for f, bw in [(rng.uniform(400, 800), 80), (rng.uniform(1000, 2000), 100), (2500, 150)]:
            r = np.exp(-np.pi * bw / rate)
            a = [1.0, -2 * r * np.cos(2 * np.pi * f / rate), r * r]
            speech = lfilter([sum(a)], a, speech)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(np.arange(len(spectrum)), 1)), n)
        noise = pink * np.sqrt(np.mean(speech[truth] ** 2) / np.mean(pink**2) / 10 ** (snr_db / 10))
        mix = speech + noise
        return 0.5 * mix / np.max(np.abs(mix)), truth[np.minimum(np.arange(1 + n // 160) * 160, n - 1)]

    for snr in (20.0, 10.0, 5.0):
        mix, ref = noisy_speech(0, snr)
        decisions = energy_vad(mix, rate)
        print(f"SNR {snr:4.0f} dB: frame F1 {frame_f1(decisions, ref):.3f}")
        if snr == 10.0:
            play(mix, rate, label="Voiced bursts in pink noise at 10 dB SNR")
            energy = frame_energy_db(mix)
            times = np.arange(len(energy)) * 160 / rate
            floor = np.percentile(energy, 10)
            show_plot((times, energy), (times, floor + 20 * decisions), (times, floor + 20 * ref),
                      labels=["frame energy (dB)", "VAD decision", "truth"], label="Energy VAD at 10 dB SNR",
                      xlabel="Time (s)", ylabel="dB")
