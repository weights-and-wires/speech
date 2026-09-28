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


@check("greedy_decode takes the argmax at each step, stops at eot, and sums log-probabilities")
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


@check("greedy_decode applies logit_filter to the logits, passing the tokens generated so far")
def test_greedy_filter(sub):
    got = has(sub, "greedy_decode")
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


@check("apply_timestamp_rules makes timestamps come in pairs and never go backward")
def test_timestamp_pairs(sub):
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


@check("apply_timestamp_rules starts with a timestamp at or before 1.00 s, and forces one when timestamps hold more mass")
def test_timestamp_start_and_mass(sub):
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


@check("word_errors and corpus_wer match the worked example and an independent edit distance")
def test_wer(sub):
    errors = has(sub, "word_errors")
    wer = has(sub, "corpus_wer")
    s, d, i = errors("the cat sat on the mat".split(), "a cat sat on mat".split())
    equal((int(s), int(d), int(i)), (1, 1, 0), label="(S, D, I) for 'the cat sat on the mat' against 'a cat sat on mat'")
    rng = np.random.default_rng(0)
    words = ["a", "b", "c", "d"]
    for trial in range(40):
        ref = list(rng.choice(words, rng.integers(0, 7)))
        hyp = list(rng.choice(words, rng.integers(0, 7)))
        s, d, i = (int(v) for v in errors(ref, hyp))
        if s + d + i != edit_distance(ref, hyp):
            fail(f"{ref} against {hyp}: S + D + I = {s + d + i}, but the edit distance is {edit_distance(ref, hyp)}")
        if len(hyp) != len(ref) - d + i:
            fail(f"{ref} against {hyp}: counts ({s}, {d}, {i}) break len(hyp) = len(ref) - D + I")
    refs = ["the cat sat on the mat", "HELLO THERE"]
    hyps = ["A cat sat on mat.", "Hello, there!"]
    almost(float(wer(refs, hyps)), 2 / 8, 1e-12, label="corpus WER, 2 edits over 8 reference words")
