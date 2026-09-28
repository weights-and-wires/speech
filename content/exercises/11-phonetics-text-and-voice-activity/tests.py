"""Checks for the phonetics, text, and voice activity exercise.

HELD_OUT holds pronunciations from the CMU Pronouncing Dictionary (the
cmusphinx/cmudict data in the `cmudict` Python package 1.1.3, first
pronunciation), Copyright (C) 1993-2015 Carnegie Mellon University,
used under its BSD-style license; the full notice is in the exercise's
starter.py and solution.py.
"""

import numpy as np
from scipy.signal import lfilter

from speechcheck import check, close, equal, fail, has, shape

RATE = 16000

HELD_OUT = {
    "cat": "K AE1 T", "dog": "D AO1 G", "ship": "SH IH1 P", "chat": "CH AE1 T", "thin": "TH IH1 N",
    "shop": "SH AA1 P", "fish": "F IH1 SH", "bat": "B AE1 T", "pat": "P AE1 T", "pin": "P IH1 N",
    "spin": "S P IH1 N", "tree": "T R IY1", "green": "G R IY1 N", "red": "R EH1 D", "blue": "B L UW1",
    "night": "N AY1 T", "sheep": "SH IY1 P", "moon": "M UW1 N", "rain": "R EY1 N", "boat": "B OW1 T",
    "coin": "K OY1 N", "sharp": "SH AA1 R P", "bird": "B ER1 D", "turn": "T ER1 N", "fork": "F AO1 R K",
    "chin": "CH IH1 N", "bench": "B EH1 N CH", "much": "M AH1 CH", "duck": "D AH1 K", "black": "B L AE1 K",
    "milk": "M IH1 L K", "stop": "S T AA1 P", "jump": "JH AH1 M P", "sand": "S AE1 N D",
    "hand": "HH AE1 N D", "wind": "W AY1 N D", "lamp": "L AE1 M P", "cake": "K EY1 K", "phone": "F OW1 N",
    "time": "T AY1 M", "home": "HH OW1 M", "cute": "K Y UW1 T", "laugh": "L AE1 F", "knife": "N AY1 F",
    "people": "P IY1 P AH0 L", "water": "W AO1 T ER0",
}


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (x != y))
    return row[-1]


def noisy_speech(seed, snr_db, seconds=3.0):
    """Voiced bursts through formant filters, in pink noise at snr_db; truth is one flag per 10 ms frame."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    truth, speech, start = np.zeros(n, dtype=bool), np.zeros(n), 0.2
    while True:
        length = rng.uniform(0.2, 0.6)
        if start + length > seconds - 0.2:
            break
        span = (t >= start) & (t < start + length)
        truth |= span
        envelope = np.clip(np.minimum(t - start, start + length - t) / 0.02, 0.0, 1.0) * span * rng.uniform(0.5, 1.0)
        f0 = rng.uniform(90.0, 250.0)
        phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))) / RATE
        speech += envelope * sum(np.sin(k * phase) / k for k in range(1, int(4000 // f0)))
        start += length + rng.uniform(0.15, 0.5)
    for f, bw in [(rng.uniform(400, 800), 80), (rng.uniform(1000, 2000), 100), (2500, 150)]:
        r = np.exp(-np.pi * bw / RATE)
        a = [1.0, -2 * r * np.cos(2 * np.pi * f / RATE), r * r]
        speech = lfilter([sum(a)], a, speech)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(np.arange(len(spectrum)), 1)), n)
    noise = pink * np.sqrt(np.mean(speech[truth] ** 2) / np.mean(pink**2) / 10 ** (snr_db / 10))
    mix = speech + noise
    return 0.5 * mix / np.max(np.abs(mix)), truth[np.minimum(np.arange(1 + n // 160) * 160, n - 1)]


def same(got, want, what):
    if got != want:
        fail(f"{what}: expected {want!r}, got {got!r}")


@check("strip_stress drops stress digits, and arpabet_to_ipa writes IPA with stress marks before vowels")
def test_phones(sub):
    strip_stress, arpabet_to_ipa = has(sub, "strip_stress"), has(sub, "arpabet_to_ipa")
    same(list(strip_stress(["S", "P", "IY1", "CH"])), ["S", "P", "IY", "CH"], "strip_stress of SPEECH")
    same(list(strip_stress(["TH", "ER1", "D", "IY2"])), ["TH", "ER", "D", "IY"], "strip_stress of THIRTY")
    same(arpabet_to_ipa(["S", "P", "IY1", "CH"]), "spˈitʃ", "arpabet_to_ipa of S P IY1 CH")
    same(arpabet_to_ipa(["HH", "AH0", "L", "OW1"]), "həlˈoʊ", "arpabet_to_ipa of HH AH0 L OW1 (unstressed AH is schwa)")
    same(arpabet_to_ipa(["B", "AH1", "T", "ER0"]), "bˈʌtɚ", "arpabet_to_ipa of B AH1 T ER0")
    same(arpabet_to_ipa(["TH", "ER1", "D", "IY2"]), "θˈɝdˌi", "arpabet_to_ipa of TH ER1 D IY2 (secondary stress)")
    same(arpabet_to_ipa(["M", "EH1", "ZH", "ER0", "NG"]), "mˈɛʒɚŋ", "arpabet_to_ipa of M EH1 ZH ER0 NG")


@check("number_to_words spells cardinals from zero to hundreds of millions")
def test_cardinals(sub):
    number_to_words = has(sub, "number_to_words")
    table = {
        0: "zero", 7: "seven", 13: "thirteen", 20: "twenty", 21: "twenty one", 99: "ninety nine",
        100: "one hundred", 105: "one hundred five", 110: "one hundred ten", 999: "nine hundred ninety nine",
        1000: "one thousand", 1001: "one thousand one", 12345: "twelve thousand three hundred forty five",
        100000: "one hundred thousand", 2000000: "two million",
        123456789: "one hundred twenty three million four hundred fifty six thousand seven hundred eighty nine",
    }
    for n, words in table.items():
        same(number_to_words(n), words, f"number_to_words({n})")


@check("ordinal_to_words and year_to_words follow the lesson's rules")
def test_ordinals_years(sub):
    ordinal_to_words, year_to_words = has(sub, "ordinal_to_words"), has(sub, "year_to_words")
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 8: "eighth", 9: "ninth",
                12: "twelfth", 20: "twentieth", 21: "twenty first", 40: "fortieth", 100: "one hundredth",
                112: "one hundred twelfth"}
    for n, words in ordinals.items():
        same(ordinal_to_words(n), words, f"ordinal_to_words({n})")
    years = {1995: "nineteen ninety five", 1905: "nineteen oh five", 1900: "nineteen hundred",
             2000: "two thousand", 2005: "two thousand five", 2010: "twenty ten", 2024: "twenty twenty four",
             1066: "ten sixty six"}
    for n, words in years.items():
        same(year_to_words(n), words, f"year_to_words({n})")


@check("normalize_text spells out money, ordinals, years, decimals, and percentages, and drops punctuation")
def test_normalize(sub):
    normalize_text = has(sub, "normalize_text")
    cases = {
        "$3.50": "three dollars fifty cents",
        "$1": "one dollar",
        "$0.99": "ninety nine cents",
        "$1.01": "one dollar one cent",
        "$2,500": "two thousand five hundred dollars",
        "It is 3.14 and 50%.": "it is three point one four and fifty percent",
        "Call me at 1024!": "call me at one thousand twenty four",
        "On May 3rd, 1776, twenty-one people": "on may third seventeen seventy six twenty one people",
        "In 1995, 12,000 people heard it.": "in nineteen ninety five twelve thousand people heard it",
        "I don't know.": "i don't know",
        "The 21st voice costs $3.50.": "the twenty first voice costs three dollars fifty cents",
    }
    for text, words in cases.items():
        same(normalize_text(text), words, f"normalize_text({text!r})")


@check("g2p looks words up in the lexicon, ignoring case, before it falls back to the rules")
def test_lookup(sub):
    g2p, lexicon = has(sub, "g2p"), has(sub, "LEXICON")
    same(list(g2p("speech")), ["S", "P", "IY1", "CH"], "g2p('speech')")
    same(list(g2p("Hello")), ["HH", "AH0", "L", "OW1"], "g2p('Hello')")
    same(list(g2p("eight")), ["EY1", "T"], "g2p('eight'), which the rules alone would get wrong")
    same(list(g2p("ship", {"ship": ["X"]})), ["X"], "g2p('ship') with a lexicon that holds it")
    same(list(g2p("ship", {})), ["SH", "IH1", "P"], "g2p('ship') with an empty lexicon, which uses the rules")
    if "ship" in lexicon:
        fail("LEXICON should not be changed; 'ship' is one of the held-out words")


@check("letter_to_sound applies the longest matching rule, with end-of-word rules, and gets most held-out words right")
def test_rules(sub):
    letter_to_sound, strip_stress = has(sub, "letter_to_sound"), has(sub, "strip_stress")
    cases = {
        "ship": ["SH", "IH1", "P"], "night": ["N", "AY1", "T"], "sharp": ["SH", "AA1", "R", "P"],
        "costly": ["K", "AA1", "S", "T", "L", "IY0"], "yes": ["Y", "EH1", "S"], "quick": ["K", "W", "IH1", "K"],
        "nation": ["N", "AE1", "SH", "AH0", "N"], "cake": ["K", "AE1", "K"], "don't": ["D", "AA1", "N", "T"],
    }
    for word, phones in cases.items():
        same(list(letter_to_sound(word)), phones, f"letter_to_sound({word!r})")
    errors = total = 0
    for word, pron in HELD_OUT.items():
        ref = pron.split()
        errors += levenshtein(list(strip_stress(letter_to_sound(word))), list(strip_stress(ref)))
        total += len(ref)
    if errors / total > 0.15:
        fail(f"the phone error rate on {len(HELD_OUT)} held-out CMUdict words is {100 * errors / total:.1f}%, expected at most 15%")


@check("text_to_phones normalizes a sentence and returns one phone list per word")
def test_text_to_phones(sub):
    text_to_phones = has(sub, "text_to_phones")
    got = [list(p) for p in text_to_phones("It costs $21.")]
    want = [["IH1", "T"], ["K", "AA1", "S", "T", "S"], ["T", "W", "EH1", "N", "T", "IY0"], ["W", "AH1", "N"],
            ["D", "AA1", "L", "ER0", "Z"]]
    same(got, want, "text_to_phones('It costs $21.')")


@check("frame_energy_db is the log mean power of centered frames")
def test_energy(sub):
    frame_energy_db = has(sub, "frame_energy_db")
    x = np.random.default_rng(0).standard_normal(4000) * 0.1
    got = np.asarray(frame_energy_db(x, 400, 160), dtype=float)
    shape(got, (26,), label="energies of 4000 samples at hop 160")
    padded = np.pad(x, 200, mode="reflect")
    want = [10 * np.log10(np.mean(padded[t * 160 : t * 160 + 400] ** 2) + 1e-10) for t in range(26)]
    close(got, want, atol=1e-9, label="10 log10(mean power + 1e-10)")
    tone = 0.5 * np.sin(2 * np.pi * 200 * np.arange(8000) / RATE)
    close(np.asarray(frame_energy_db(tone))[5:-5], np.full(41, 10 * np.log10(0.125)), atol=0.01,
          label="a sine of amplitude 0.5, whose mean power is 0.125, or -9.03 dB")
    close(np.asarray(frame_energy_db(np.zeros(1600))), np.full(11, -100.0), atol=1e-9, label="silence, floored at -100 dB")


@check("hysteresis switches on above on_db and off only after the hangover")
def test_hysteresis(sub):
    hysteresis = has(sub, "hysteresis")
    e = np.array([0, 12, 8, 3, 3, 3, 9, 2, 2], dtype=float)
    equal(np.asarray(hysteresis(e, 10.0, 5.0, 2), dtype=bool),
          np.array([0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool), label="the worked example, hangover 2")
    equal(np.asarray(hysteresis(e, 10.0, 5.0, 0), dtype=bool),
          np.array([0, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool), label="the same energies with no hangover")
    e2 = np.array([11, 3, 6, 3, 3, 3, 11], dtype=float)
    equal(np.asarray(hysteresis(e2, 10.0, 5.0, 2), dtype=bool),
          np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool), label="a frame back above off_db resets the hangover count")


@check("frame_f1 scores decisions, and energy_vad reaches F1 0.9 at 20 dB and 10 dB SNR")
def test_vad(sub):
    frame_f1, energy_vad = has(sub, "frame_f1"), has(sub, "energy_vad")
    pred = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    ref = np.array([1, 0, 0, 1, 1, 0], dtype=bool)
    close(frame_f1(pred, ref), 2 * 2 / (2 * 2 + 1 + 1), atol=1e-12, label="F1 with TP 2, FP 1, FN 1")
    close(frame_f1(np.zeros(5, bool), np.zeros(5, bool)), 1.0, label="F1 when neither track has speech")
    for snr in (20.0, 10.0):
        for seed in range(3):
            mix, truth = noisy_speech(seed, snr)
            decisions = np.asarray(energy_vad(mix, RATE), dtype=bool)
            shape(decisions, truth.shape, label="the VAD decisions, one per 10 ms frame")
            score = frame_f1(decisions, truth)
            if score < 0.9:
                fail(f"energy_vad's frame F1 is {score:.3f} on noisy speech at {snr:.0f} dB SNR (seed {seed}); expected at least 0.9")
