"""Whisper's prompt, your own decoding loops, timestamp rules, and a fair WER.

A decoder here is any function `step_fn(tokens) -> logits`: it takes the whole
token sequence so far, a list of ints that starts with the prompt, and returns
the next-token logits as a 1-D array over the vocabulary. The Run block wraps
whisper-tiny's decoder in one; the checks use small fake decoders.
"""

import re

import numpy as np

from speechlab import has_asset, play, show_spectrogram

# Special-token ids of the multilingual Whisper tokenizer, read from
# assets/whisper-tiny with WhisperTokenizer.convert_tokens_to_ids. The Run
# block checks them against the tokenizer again.
SPECIAL = {
    "<|endoftext|>": 50257,
    "<|startoftranscript|>": 50258,
    "<|translate|>": 50358,
    "<|transcribe|>": 50359,
    "<|startoflm|>": 50360,
    "<|startofprev|>": 50361,
    "<|nocaptions|>": 50362,
    "<|notimestamps|>": 50363,
    "<|0.00|>": 50364,
}
LANGUAGES = {"en": 50259, "zh": 50260, "de": 50261, "es": 50262, "fr": 50265, "ja": 50266}
EOT = SPECIAL["<|endoftext|>"]
TIMESTAMP_BEGIN = SPECIAL["<|0.00|>"]


def build_prompt(language: str = "en", task: str = "transcribe", timestamps: bool = False) -> list[int]:
    """The forced start of every Whisper decode, as token ids.

    <|startoftranscript|>, the language token from LANGUAGES, <|transcribe|>
    or <|translate|>, and <|notimestamps|> unless `timestamps` is true.
    """
    prompt = [SPECIAL["<|startoftranscript|>"], LANGUAGES[language], SPECIAL[f"<|{task}|>"]]
    if not timestamps:
        prompt.append(SPECIAL["<|notimestamps|>"])
    return prompt


def _log_softmax(z: np.ndarray) -> np.ndarray:
    finite = np.max(z)
    shifted = z - finite
    return shifted - np.log(np.sum(np.exp(shifted)))


def greedy_decode(step_fn, prompt: list[int], eot: int, max_new_tokens: int, logit_filter=None) -> tuple[list[int], float]:
    """Pick the most likely token at every step until `eot` or the length limit.

    Args:
        step_fn: Maps the token list so far to next-token logits, shape (V,).
        prompt: The forced tokens the decode starts from.
        eot: The end-of-text id, which stops the decode once it is chosen.
        max_new_tokens: The most tokens to generate after the prompt.
        logit_filter: Optional `f(logits, generated) -> logits`, applied before
            each choice; `generated` is the list of tokens after the prompt.

    Returns:
        (generated, sum_logprob): the tokens after the prompt, ending with
        `eot` if it was chosen, and the sum of their log-probabilities under
        the log-softmax of the filtered logits.
    """
    tokens = list(prompt)
    total = 0.0
    for _ in range(max_new_tokens):
        logits = np.array(step_fn(tokens), dtype=np.float64)
        if logit_filter is not None:
            logits = logit_filter(logits, tokens[len(prompt) :])
        logp = _log_softmax(logits)
        choice = int(np.argmax(logp))
        total += float(logp[choice])
        tokens.append(choice)
        if choice == eot:
            break
    return tokens[len(prompt) :], total


def beam_search_decode(
    step_fn, prompt: list[int], eot: int, beam_size: int, max_new_tokens: int, logit_filter=None
) -> tuple[list[int], float]:
    """Whisper's beam search, ranked by log-probability per token.

    Keep `beam_size` unfinished hypotheses. At each step, extend every one by
    its `beam_size + 1` most likely next tokens, sort all the extensions by
    their summed log-probability, and walk down that list: an extension ending
    in `eot` joins the finished set, anything else joins the next beams, until
    `beam_size` beams are kept. Stop once `beam_size` hypotheses have finished
    or after `max_new_tokens` steps; then close any live beams with `eot` at no
    extra cost until there are `beam_size` finished ones.

    Among the finished hypotheses, return the one with the largest
    sum_logprob / max(n, 1), where n counts the tokens before `eot`.

    Returns:
        (generated, sum_logprob) for the winner, as in greedy_decode.
    """
    start = len(prompt)
    beams = [(list(prompt), 0.0)]
    finished: dict[tuple[int, ...], float] = {}
    for _ in range(max_new_tokens):
        candidates: dict[tuple[int, ...], float] = {}
        for tokens, score in beams:
            logits = np.array(step_fn(tokens), dtype=np.float64)
            if logit_filter is not None:
                logits = logit_filter(logits, tokens[start:])
            logp = _log_softmax(logits)
            for token in np.argsort(-logp, kind="stable")[: beam_size + 1]:
                if np.isfinite(logp[token]):
                    candidates[tuple(tokens) + (int(token),)] = score + float(logp[token])
        beams = []
        for seq in sorted(candidates, key=candidates.get, reverse=True):
            if seq[-1] == eot:
                if len(finished) < beam_size:
                    finished[seq] = candidates[seq]
            else:
                beams.append((list(seq), candidates[seq]))
                if len(beams) == beam_size:
                    break
        if len(finished) >= beam_size or not beams:
            break
    for tokens, score in sorted(beams, key=lambda b: b[1], reverse=True):
        if len(finished) >= beam_size:
            break
        finished[tuple(tokens) + (eot,)] = score

    def normalized(seq):
        n = len(seq) - start - 1
        return finished[seq] / max(n, 1)

    best = max(finished, key=normalized)
    return list(best[start:]), finished[best]


def apply_timestamp_rules(
    logits: np.ndarray,
    generated: list[int],
    timestamp_begin: int = TIMESTAMP_BEGIN,
    eot: int = EOT,
    no_timestamps: int = SPECIAL["<|notimestamps|>"],
    max_initial_index: int = 50,
) -> np.ndarray:
    """Whisper's timestamp rules, as a filtered copy of `logits`, shape (V,).

    Ids from `timestamp_begin` up are timestamps; everything below is text or
    a special token. Set to -inf, in this order:

    1. `no_timestamps`, always.
    2. After a timestamp that follows a timestamp (or starts the output), all
       timestamps: text must come next. After a timestamp that follows text,
       every id below `eot`: a timestamp or `eot` must come next.
    3. Timestamps that go backward: below the last timestamp when it opens a
       pair's end (the case in rule 2 that forbids text), otherwise below the
       last timestamp plus one.
    4. With nothing generated yet, every id below `timestamp_begin`, and every
       timestamp above `timestamp_begin + max_initial_index`.
    5. Then, if the log-sum-exp of the timestamp log-probabilities beats the
       largest log-probability below `timestamp_begin`, every id below it.
    """
    out = np.array(logits, dtype=np.float64)
    out[no_timestamps] = -np.inf
    last_ts = len(generated) >= 1 and generated[-1] >= timestamp_begin
    penultimate_ts = len(generated) < 2 or generated[-2] >= timestamp_begin
    if last_ts:
        if penultimate_ts:
            out[timestamp_begin:] = -np.inf
        else:
            out[:eot] = -np.inf
    stamps = [t for t in generated if t >= timestamp_begin]
    if stamps:
        floor = stamps[-1] if (last_ts and not penultimate_ts) else stamps[-1] + 1
        out[timestamp_begin:floor] = -np.inf
    if not generated:
        out[:timestamp_begin] = -np.inf
        out[timestamp_begin + max_initial_index + 1 :] = -np.inf
    logp = _log_softmax(out)
    stamp_mass = np.logaddexp.reduce(logp[timestamp_begin:])
    if stamp_mass > np.max(logp[:timestamp_begin]):
        out[:timestamp_begin] = -np.inf
    return out


FILLERS = re.compile(r"\b(hmm|mm|mhm|mmm|uh|um)\b")
REPLACEMENTS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\blet's\b", "let us"),
    (r"\bmr\b", "mister "),
    (r"\bmrs\b", "missus "),
    (r"\bdr\b", "doctor "),
    (r"\bst\b", "saint "),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'s\b", " is"),
    (r"'d\b", " would"),
    (r"'ll\b", " will"),
    (r"'ve\b", " have"),
    (r"'m\b", " am"),
]


def normalize_text(text: str) -> str:
    """A reduced Whisper English normalizer, applied to both sides before WER.

    Lowercase; delete text inside [], <>, and (); delete the fillers hmm, mm,
    mhm, mmm, uh, and um as whole words; join a space before an apostrophe;
    apply the REPLACEMENTS in order as regular expressions; turn every other
    punctuation mark or symbol into a space; and collapse runs of whitespace
    into one space, with none at either end.
    """
    s = text.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
    s = re.sub(r"\(([^)]+?)\)", "", s)
    s = FILLERS.sub("", s)
    s = re.sub(r"\s+'", "'", s)
    for pattern, replacement in REPLACEMENTS:
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Provided in full: you built this in the word error rate chapter.

    (substitutions, deletions, insertions) of a minimum-edit alignment.

    Any alignment with the fewest total edits is fine; the three counts
    always satisfy len(hypothesis) = len(reference) - deletions + insertions.
    """
    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=int)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1])
            cost[i, j] = min(diagonal, cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            s += reference[i - 1] != hypothesis[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return int(s), int(d), int(ins)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus word error rate: total edits over total reference words.

    Normalize both sides with normalize_text, split on spaces, and sum
    substitutions, deletions, and insertions over every pair before dividing
    by the summed reference length. Use the provided word_errors for each pair.
    """
    edits = words = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = normalize_text(ref).split(), normalize_text(hyp).split()
        edits += sum(word_errors(r, h))
        words += len(r)
    return edits / max(words, 1)


# The Run experiment -----------------------------------------------------------


def slaney_mel_filterbank(sample_rate: int = 16000, n_fft: int = 400, n_mels: int = 80) -> np.ndarray:
    """librosa's default mel filters (Slaney scale and area normalization), shape (n_mels, n_fft // 2 + 1)."""

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), f / (200.0 / 3))

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * (200.0 / 3))

    freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2), n_mels + 2))
    rising = (freqs[None, :] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
    falling = (edges[2:, None] - freqs[None, :]) / (edges[2:] - edges[1:-1])[:, None]
    weights = np.maximum(0.0, np.minimum(rising, falling))
    return weights * (2.0 / (edges[2:] - edges[:-2]))[:, None]


def whisper_log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Whisper's log-mel input for one 30 s window, shape (80, 3000), as in the log-mel spectrograms chapter."""
    audio = np.asarray(audio, dtype=np.float64)[:480000]
    audio = np.pad(audio, (0, 480000 - audio.size))
    padded = np.pad(audio, (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    log_spec = np.log10(np.maximum(filters @ power.T, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def librispeech_utterances(count: int):
    """The first `count` dev-clean utterances, taking one per chapter, as (path, transcript)."""
    from speechlab import asset_path

    root = asset_path("librispeech-dev-clean") / "LibriSpeech" / "dev-clean"
    found = []
    for listing in sorted(root.glob("*/*/*.trans.txt")):
        uid, text = listing.read_text().splitlines()[0].split(" ", 1)
        found.append((listing.parent / f"{uid}.flac", text))
        if len(found) == count:
            break
    return found


def suppression_filter(suppress: list[int], at_begin: list[int], rules=None):
    """A logit filter that applies Whisper's suppress lists, then optional timestamp rules."""

    def apply(logits, generated):
        logits = logits.copy()
        logits[suppress] = -np.inf
        if not generated:
            logits[at_begin] = -np.inf
        return rules(logits, generated) if rules else logits

    return apply


if __name__ == "__main__":
    print("Prompt without timestamps:", build_prompt())
    ok = normalize_text("Mr. Quilter's (the apostle) gospel -- uh, WON'T wait!")
    print("Normalized:", repr(ok))

    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperTokenizer
        from transformers.utils import logging

        logging.set_verbosity_error()
        logging.disable_progress_bar()
        have = has_asset("whisper-tiny") and has_asset("librispeech-dev-clean")
    except ImportError:
        have = False

    if not have:
        print("This run needs the whisper-tiny and librispeech-dev-clean assets and transformers.")
        print("Without them, it decodes a toy three-word decoder instead.")
        vocab = ["<eot>", "a", "cat", "the"]
        # "a" is the likelier first word, but nothing confident follows it; "the" leads to "cat".
        table = {(): [-5, 1.0, -5, 0.8], (1,): [0, 0, 0, 0], (3,): [-3, -3, 4, -3]}
        toy = lambda tokens: np.array(table.get(tuple(tokens[1:]), [5, 0, 0, 0]), dtype=float)  # noqa: E731
        for name, (gen, lp) in (("greedy", greedy_decode(toy, [9], 0, 6)), ("beam 3", beam_search_decode(toy, [9], 0, 3, 6))):
            print(f"{name}: {' '.join(vocab[t] for t in gen)}  (log-prob {lp:.3f})")
    else:
        from speechlab import asset_path, read_audio

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        n_utts, beam = (20, 5) if device == "cuda" else (4, 3)
        if device == "cpu":
            print(f"No GPU, so this is the small version: {n_utts} utterances and beam size {beam} instead of 20 and 5.")

        folder = str(asset_path("whisper-tiny"))
        tokenizer = WhisperTokenizer.from_pretrained(folder)
        model = WhisperForConditionalGeneration.from_pretrained(folder).to(device).eval()
        for name, idx in list(SPECIAL.items()) + [(f"<|{c}|>", i) for c, i in LANGUAGES.items()]:
            if tokenizer.convert_tokens_to_ids(name) != idx:
                raise SystemExit(f"{name} is {tokenizer.convert_tokens_to_ids(name)} in the tokenizer, not {idx}")
        print("All special-token ids match assets/whisper-tiny's tokenizer.")

        suppress = list(model.generation_config.suppress_tokens)
        at_begin = list(model.generation_config.begin_suppress_tokens)
        filters = slaney_mel_filterbank()
        refs, greedy_hyps, beam_hyps = [], [], []
        for k, (path, text) in enumerate(librispeech_utterances(n_utts)):
            audio, rate = read_audio(path)
            mel = torch.from_numpy(whisper_log_mel(audio, filters))[None].to(device)
            with torch.inference_mode():
                encoded = model.model.encoder(mel).last_hidden_state  # (1, 1500, 384), run once

                def step_fn(tokens):
                    with torch.inference_mode():
                        ids = torch.tensor([tokens], device=device)
                        hidden = model.model.decoder(input_ids=ids, encoder_hidden_states=encoded).last_hidden_state
                        return model.proj_out(hidden)[0, -1].float().cpu().numpy()

                plain = suppression_filter(suppress, at_begin)
                g_tokens, g_lp = greedy_decode(step_fn, build_prompt(), EOT, 120, plain)
                b_tokens, b_lp = beam_search_decode(step_fn, build_prompt(), EOT, beam, 120, plain)
                if k == 0:
                    play(audio[: 6 * rate], rate, label="The first utterance")
                    show_spectrogram(mel[0, :, : int(len(audio) / 160)].cpu().numpy(), 16000, 160,
                                     label="Its Whisper log-mel input", db=False, y_axis="mel",
                                     fmax=8000.0, value_label="(log10 + 4) / 4")
                    reference = model.generate(mel, language="en", task="transcribe", num_beams=1, do_sample=False)
                    ref_ids = [t for t in reference[0].tolist() if t < EOT]
                    print(f"Your greedy tokens match model.generate: {[t for t in g_tokens if t < EOT] == ref_ids}")
                    stamped, _ = greedy_decode(step_fn, build_prompt(timestamps=True), EOT, 120,
                                               suppression_filter(suppress, at_begin, apply_timestamp_rules))
                    pieces = []
                    for t in stamped:
                        pieces.append(f"[{(t - TIMESTAMP_BEGIN) * 0.02:.2f}]" if t >= TIMESTAMP_BEGIN else tokenizer.decode([t]))
                    print("With timestamps:", "".join(pieces).replace("<|endoftext|>", ""))
            greedy_text = tokenizer.decode([t for t in g_tokens if t < EOT])
            beam_text = tokenizer.decode([t for t in b_tokens if t < EOT])
            refs.append(text)
            greedy_hyps.append(greedy_text)
            beam_hyps.append(beam_text)
            print(f"{k + 1:2d}. ref:    {normalize_text(text)}")
            print(f"    greedy: {normalize_text(greedy_text)}")
            if beam_text != greedy_text:
                print(f"    beam:   {normalize_text(beam_text)}")

        raw = sum(sum(word_errors(r.split(), h.split())) for r, h in zip(refs, greedy_hyps))
        raw /= sum(len(r.split()) for r in refs)
        print(f"Greedy WER without normalization: {raw:.1%}")
        print(f"Greedy WER, normalized: {corpus_wer(refs, greedy_hyps):.1%}")
        print(f"Beam-{beam} WER, normalized: {corpus_wer(refs, beam_hyps):.1%}")
