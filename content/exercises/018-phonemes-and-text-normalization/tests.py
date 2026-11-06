"""Checks for the phonemes and text normalization exercise.

HELD_OUT holds pronunciations from the CMU Pronouncing Dictionary (the
cmusphinx/cmudict data in the `cmudict` Python package 1.1.3, first
pronunciation), Copyright (C) 1993-2015 Carnegie Mellon University,
used under its BSD-style license; the full notice is in the exercise's
starter.py and solution.py.
"""

from speechcheck import check, fail, has

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
