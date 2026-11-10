"""Checks for the Whisper decoding exercise. None of them needs Whisper."""

import itertools

import numpy as np

from speechcheck import almost, check, equal, fail, has

TS = 50364
EOT = 50257
NO_TS = 50363
V = 51865


def log_softmax(z):
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return z - m - np.log(np.sum(np.exp(z - m)))


def table_decoder(table, vocab, default):
    """A fake decoder: logits depend on the tokens after the one-token prompt."""

    def step(tokens):
        return np.array(table.get(tuple(tokens[1:]), default), dtype=float)[:vocab]

    return step


def random_decoder(vocab, seed):
    """A fake decoder with fixed random logits for every prefix."""
    cache = {}
    rng = np.random.default_rng(seed)

    def step(tokens):
        key = tuple(tokens)
        if key not in cache:
            cache[key] = rng.normal(size=vocab) * 1.5
        return cache[key].copy()

    return step


@check("build_prompt gives the forced tokens <|startoftranscript|><|en|><|transcribe|><|notimestamps|>")
def test_prompt(sub):
    got = has(sub, "build_prompt")
    equal(list(got()), [50258, 50259, 50359, 50363], label="English transcription without timestamps")
    equal(list(got("en", "transcribe", True)), [50258, 50259, 50359], label="with timestamps")
    equal(list(got("de", "translate", False)), [50258, 50261, 50358, 50363], label="German to English translation")


@check("greedy_decode takes the argmax, stops at eot, sums log-probabilities, and applies logit_filter")
def test_greedy(sub):
    got = has(sub, "greedy_decode")
    # vocab: 0 = eot, 1, 2, 3
    table = {(): [0, 3, 1, 0], (1,): [0, 0, 2, 1], (1, 2): [4, 0, 0, 1]}
    step = table_decoder(table, 4, [5, 0, 0, 0])
    tokens, lp = got(step, [9], 0, 10)
    equal(list(tokens), [1, 2, 0], label="greedy tokens")
    want = log_softmax(table[()])[1] + log_softmax(table[(1,)])[2] + log_softmax(table[(1, 2)])[0]
    almost(float(lp), float(want), 1e-9, label="summed log-probability")
    tokens, _ = got(step, [9], 0, 2)
    equal(list(tokens), [1, 2], label="tokens with max_new_tokens = 2, which stops before eot")

    table = {(): [0, 3, 1, 0], (1,): [0, 0, 2, 1], (3,): [2, 0, 0, 0]}
    step = table_decoder(table, 4, [5, 0, 0, 0])
    seen = []

    def ban_first(logits, generated):
        seen.append(list(generated))
        logits = np.array(logits, dtype=float)
        if not generated:
            logits[1] = -np.inf
        return logits

    tokens, lp = got(step, [9], 0, 10, ban_first)
    equal(list(tokens), [2, 0], label="tokens when token 1 is banned at the first step")
    if seen[:2] != [[], [2]]:
        fail(f"logit_filter should see generated = [] and then [2], got {seen[:2]}")
    filtered = np.array([0, -np.inf, 1, 0.0])
    want = log_softmax(filtered)[2] + log_softmax([5, 0, 0, 0])[0]
    almost(float(lp), float(want), 1e-9, label="log-probability under the filtered logits")


@check("beam_search_decode with beam size 1 returns the greedy answer")
def test_beam_one(sub):
    greedy = has(sub, "greedy_decode")
    beam = has(sub, "beam_search_decode")
    for seed in range(5):
        step = random_decoder(5, seed)
        g, g_lp = greedy(step, [7], 0, 6)
        b, b_lp = beam(step, [7], 0, 1, 6)
        # Beam search closes a beam that hits the length limit with eot; greedy doesn't.
        g, b = list(g), list(b)
        if len(b) == 7 and b[-1] == 0:
            b = b[:-1]
        if g != b:
            fail(f"seed {seed}: greedy gave {list(g)} but beam size 1 gave {list(b)}")
        almost(float(b_lp), float(g_lp), 1e-9, label=f"seed {seed} log-probability")


def brute_force_best(step, prompt, eot, vocab, max_new):
    best, best_score, best_lp = None, -np.inf, None
    for n in range(1, max_new + 1):
        for seq in itertools.product(range(vocab), repeat=n):
            if eot in seq[:-1]:
                continue
            lp, tokens = 0.0, list(prompt)
            for t in seq:
                lp += log_softmax(step(tokens))[t]
                tokens.append(t)
            if seq[-1] == eot:
                words = n - 1
            elif n == max_new:
                words, seq = n, seq + (eot,)
            else:
                continue
            score = lp / max(words, 1)
            if score > best_score:
                best, best_score, best_lp = list(seq), score, lp
    return best, best_lp


@check("beam_search_decode with a wide beam finds the brute-force best per-token log-probability")
def test_beam_brute_force(sub):
    beam = has(sub, "beam_search_decode")
    for seed in range(4):
        step = random_decoder(3, 100 + seed)
        want, want_lp = brute_force_best(step, [5], 0, 3, 4)
        got, got_lp = beam(step, [5], 0, 40, 4)
        if list(got) != want:
            fail(f"seed {seed}: expected {want} (log-prob {want_lp:.4f}), got {list(got)} (log-prob {got_lp:.4f})")
        almost(float(got_lp), float(want_lp), 1e-9, label=f"seed {seed} log-probability")


@check("beam_search_decode escapes a greedy trap that greedy_decode falls into")
def test_beam_trap(sub):
    greedy = has(sub, "greedy_decode")
    beam = has(sub, "beam_search_decode")
    # vocab: 0 = eot, 1 = "a", 2 = "cat", 3 = "the"
    table = {(): [-5, 1.0, -5, 0.8], (1,): [0, 0, 0, 0], (3,): [-3, -3, 4, -3]}
    step = table_decoder(table, 4, [5, 0, 0, 0])
    g, _ = greedy(step, [9], 0, 5)
    equal(list(g), [1, 0], label="greedy picks 'a' and then gives up")
    b, lp = beam(step, [9], 0, 3, 5)
    equal(list(b), [3, 2, 0], label="beam size 3 finds 'the cat'")
    want = log_softmax(table[()])[3] + log_softmax(table[(3,)])[2] + log_softmax([5, 0, 0, 0])[0]
    almost(float(lp), float(want), 1e-9, label="log-probability of 'the cat <eot>'")


def allowed(sub, generated, logits=None):
    rules = has(sub, "apply_timestamp_rules")
    if logits is None:
        # Text at logit 10 outweighs all 1,501 timestamps at logit 0 together,
        # so rule 5 stays out of the way unless a check sets its own logits.
        logits = np.zeros(V)
        logits[:TS] = 10.0
    z = logits
    out = np.asarray(rules(z.copy(), list(generated), TS, EOT, NO_TS, 50))
    if out.shape != (V,):
        fail(f"apply_timestamp_rules should return shape ({V},), got {out.shape}")
    return np.isfinite(out)


def _timestamp_pairs(sub):
    ok = allowed(sub, [TS, 100, 200])  # <|0.00|> text text
    if ok[NO_TS]:
        fail("<|notimestamps|> must always be suppressed")
    if ok[TS]:
        fail("after <|0.00|> and text, <|0.00|> again would go backward; it must be suppressed")
    if not (ok[300] and ok[TS + 1] and ok[EOT]):
        fail("after <|0.00|> and text, text, a later timestamp, and eot must all stay allowed")
    ok = allowed(sub, [TS, 100, TS + 60])  # text, then <|1.20|> closes the segment
    if ok[300] or ok[250]:
        fail("after text and one timestamp, text is forbidden: a timestamp or eot must come next")
    if not (ok[TS + 60] and ok[TS + 90] and ok[EOT]):
        fail("after text and <|1.20|>, <|1.20|> itself, a later timestamp, and eot must stay allowed")
    if ok[TS + 59]:
        fail("after text and <|1.20|>, the earlier <|1.18|> must be suppressed")
    ok = allowed(sub, [TS, 100, TS + 60, TS + 60])  # a closed pair, then a new segment opens
    if ok[TS + 60] or ok[TS + 70]:
        fail("after two timestamps in a row, the next token must be text, not another timestamp")
    if not ok[300]:
        fail("after two timestamps in a row, text must stay allowed")
    ok = allowed(sub, [TS])
    if ok[TS + 10] or not ok[300]:
        fail("after the opening <|0.00|> alone, text must come next")


def _timestamp_start_and_mass(sub):
    ok = allowed(sub, [])
    if ok[:TS].any():
        fail(f"the first token must be a timestamp, but ids {np.flatnonzero(ok[:TS])[:5]} are still allowed")
    if not (ok[TS] and ok[TS + 50]) or ok[TS + 51]:
        fail("the first timestamp must lie in <|0.00|> to <|1.00|>, ids 50364 to 50414")
    logits = np.zeros(V)
    logits[:TS] = -10.0
    logits[300] = 2.0
    ok = allowed(sub, [TS, 100], logits)
    # The 1,501 timestamps at logit 0 hold far more mass than the one text token at 2.
    if ok[300]:
        fail("when the timestamps' total probability beats the best text token, text must be suppressed")
    logits[300] = 12.0
    ok = allowed(sub, [TS, 100], logits)
    if not ok[300]:
        fail("when one text token beats the timestamps' total probability, text must stay allowed")


@check("apply_timestamp_rules keeps timestamps paired and forward, starts at or before 1.00 s, and forces one when they hold more mass")
def test_timestamp_rules(sub):
    _timestamp_pairs(sub)
    _timestamp_start_and_mass(sub)


@check("normalize_text lowercases, drops brackets and fillers, expands contractions, and strips punctuation")
def test_normalize(sub):
    got = has(sub, "normalize_text")
    cases = {
        "Mr. Quilter is the apostle.": "mister quilter is the apostle",
        "MISTER QUILTER IS THE APOSTLE": "mister quilter is the apostle",
        "Um, I won't go [laughs] (quietly)!": "i will not go",
        "Well-known  \"facts\" -- you're right": "well known facts you are right",
        "They'll say it's ours.": "they will say it is ours",
    }
    for text, want in cases.items():
        out = got(text)
        if out != want:
            fail(f"normalize_text({text!r}) should give {want!r}, got {out!r}")


def edit_distance(a, b):
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[-1])


@check("corpus_wer normalizes both sides and sums edits before dividing, matching an independent edit distance")
def test_wer(sub):
    wer = has(sub, "corpus_wer")
    refs = ["the cat sat on the mat", "HELLO THERE"]
    hyps = ["A cat sat on mat.", "Hello, there!"]
    almost(float(wer(refs, hyps)), 2 / 8, 1e-12, label="corpus WER, 2 edits over 8 reference words")
    rng = np.random.default_rng(0)
    words = ["a", "b", "c", "d"]
    for trial in range(20):
        ref = [list(rng.choice(words, rng.integers(1, 7))) for _ in range(3)]
        hyp = [list(rng.choice(words, rng.integers(0, 7))) for _ in range(3)]
        want = sum(edit_distance(r, h) for r, h in zip(ref, hyp)) / sum(len(r) for r in ref)
        got = float(wer([" ".join(r) for r in ref], [" ".join(h) for h in hyp]))
        almost(got, want, 1e-12, label=f"corpus WER of random trial {trial}")
    # Averaging per-utterance WERs would give 0.625 here; the corpus rate is 2 / 5.
    almost(float(wer(["a", "a b c d"], ["b", "a b c e"])), 2 / 5, 1e-12, label="corpus WER weights long utterances more")


LANGS = {"en": 50259, "zh": 50260, "de": 50261, "es": 50262, "fr": 50265, "ja": 50266}


def _language_logits(seed, top_code, top_id=None):
    """Logits for a decoder step after <|startoftranscript|>.

    The language tokens get small random logits, with `top_code` clearly ahead.
    A text token, id 11, gets the largest logit of all, as an unsure decoder
    might give a word it would write if it were allowed to.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=V)
    for code, idx in LANGS.items():
        z[idx] = rng.normal() - 1.0
    z[LANGS[top_code]] = 2.0
    z[11 if top_id is None else top_id] = 9.0
    return z


def same(got, want, label):
    if got != want:
        fail(f"{label}: expected {want!r}, got {got!r}")


@check("detect_language takes the softmax over the language tokens only and returns the likeliest language")
def test_detect_language(sub):
    detect = has(sub, "detect_language")
    # A wrong implementation takes the softmax over the whole vocabulary of 51,865
    # logits. Its language probabilities then sum to far less than 1, and an
    # argmax over every token picks the text token (id 11) instead of a language.
    for seed, top in [(0, "de"), (1, "ja"), (2, "en")]:
        z = _language_logits(seed, top)
        code, probs = detect(z, dict(LANGS))
        same(code, top, f"detected language for seed {seed}")
        if set(probs) != set(LANGS):
            fail(f"probs should have the keys {sorted(LANGS)}, got {sorted(probs)}")
        total = sum(probs.values())
        almost(total, 1.0, 1e-9, label=f"language probabilities sum for seed {seed}")
        want = np.exp(log_softmax([z[i] for i in LANGS.values()]))
        got = np.array([probs[c] for c in LANGS])
        almost(np.max(np.abs(got - want)), 0.0, 1e-9, label=f"largest probability error for seed {seed}")
    # Language order and the subset passed in must not matter: only these ids compete.
    subset = {"fr": LANGS["fr"], "es": LANGS["es"]}
    z = _language_logits(3, "fr")
    z[LANGS["es"]] = 1.0
    code, probs = detect(z, subset)
    same(code, "fr", "detected language among two candidates")
    almost(sum(probs.values()), 1.0, 1e-9, label="probabilities sum over two candidates")
    almost(probs["fr"], 1 / (1 + np.exp(-1.0)), 1e-9, label="P(fr) among two candidates, logits 2.0 and 1.0")


@check("build_prompt uses the detected language when language is None, and an explicit language wins")
def test_prompt_detected(sub):
    prompt = has(sub, "build_prompt")
    detect = has(sub, "detect_language")
    equal(list(prompt(None, "transcribe", False, detected="fr")), [50258, 50265, 50359, 50363], label="prompt for detected French")
    equal(list(prompt("de", "transcribe", True, detected="fr")), [50258, 50261, 50359], label="explicit German beats detected French")
    z = _language_logits(4, "ja")
    code, _ = detect(z, dict(LANGS))
    equal(list(prompt(None, "translate", False, detected=code)), [50258, 50266, 50358, 50363], label="prompt from detect_language's output")
    equal(list(prompt()), [50258, 50259, 50359, 50363], label="the default prompt is still English")
