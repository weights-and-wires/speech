"""Checks for the ASR training recipe.

Wrong implementation `merge_most_frequent_token_pair`: it ranks pairs by
their counts in the original character tokenization and never recounts
after a merge. On the worked corpus that list is
[("c", "a"), ("a", "r"), ("a", "t")]. `train_bpe` must merge the most
frequent pair of current symbols, so the second merge is ("ca", "r").

Wrong implementation `inverse_sqrt_without_warmup`: it returns
d_model**-0.5 * step**-0.5 and skips the linear warmup. At step 100,
with d_model 256 and warmup 4000, that value is 0.00625. The closed form
is 2.470529422e-05.
"""

import numpy as np

from speechcheck import almost, check, close, equal, fail, near_frequency, tone


CORPUS = ["cat", "cat", "car", "car", "card"]
MERGES = [("c", "a"), ("ca", "r"), ("ca", "t")]
RATE = 16000


def _fn(sub, name, expected):
    if not hasattr(sub, name):
        fail(
            f"your module defines no `{name}`. expected {expected}, got no function"
        )
    return getattr(sub, name)


def _call(fn, args, expected):
    try:
        return fn(*args)
    except NotImplementedError:
        fail(f"{fn.__name__} is not implemented. expected {expected}, got no result")


def _closed_form(step, d_model, warmup):
    step_f = float(step)
    return d_model**-0.5 * min(step_f**-0.5, step_f * warmup**-1.5)


@check("train_bpe matches the worked merge table")
def test_train_bpe(sub):
    # `merge_most_frequent_token_pair` fails here: it returns
    # [("c", "a"), ("a", "r"), ("a", "t")] instead of MERGES.
    train = _fn(sub, "train_bpe", MERGES)
    got = _call(train, (CORPUS, 3), MERGES)
    equal(got, MERGES, label="merges")


@check("encode and decode round-trip the worked merges")
def test_round_trip(sub):
    encode = _fn(sub, "encode", ["car", "d"])
    decode = _fn(sub, "decode", "card")
    encoded = _call(encode, ("card", MERGES), ["car", "d"])
    equal(encoded, ["car", "d"], label="encode('card')")
    for word in ("cat", "car", "card"):
        symbols = _call(encode, (word, MERGES), f"symbols of {word}")
        restored = _call(decode, (symbols,), word)
        if restored != word:
            fail(f"decode(encode({word!r})) is {restored!r}, expected {word!r}")
    cat = _call(encode, ("cat", MERGES), ["cat"])
    equal(cat, ["cat"], label="encode('cat')")


@check("speed_perturb shortens or lengthens and moves the peak")
def test_speed(sub):
    speed = _fn(sub, "speed_perturb", "14546 samples near 440 Hz")
    tone_1s = tone(400, RATE, 1.0)
    # factor = numer/denom. Length is ceil(N * denom / numer).
    cases = ((1.1, 11, 10, 440.0), (0.9, 9, 10, 360.0))
    for factor, numer, denom, hz in cases:
        got = _call(speed, (tone_1s, factor), f"length ceil(16000 * {denom}/{numer})")
        want_len = -(-(len(tone_1s) * denom) // numer)
        if len(np.asarray(got)) != want_len:
            fail(
                f"factor {factor}: length is {len(np.asarray(got))}, "
                f"expected {want_len}"
            )
        near_frequency(got, RATE, hz, label=f"factor {factor}")


@check("mix_at_snr hits the requested SNR in decibels")
def test_mix(sub):
    mix = _fn(sub, "mix_at_snr", "SNR equal to the requested decibel value")
    rng = np.random.default_rng(0)
    clean = tone(400, RATE, 0.25)
    noise = rng.standard_normal(clean.shape)
    for snr in (-5.0, 0.0, 12.0):
        mixed, scaled = _call(mix, (clean, noise, snr), f"SNR {snr} dB")
        measured = 10 * np.log10(np.mean(clean**2) / np.mean(np.asarray(scaled) ** 2))
        almost(measured, snr, 1e-8, label=f"SNR at {snr} dB")
        close(np.asarray(mixed), clean + np.asarray(scaled), label=f"mixture at {snr} dB")


@check("bucket_batches stays under the padding-waste bound")
def test_buckets(sub):
    # `pack_in_given_order` never sorts. On these lengths it pads 276
    # frames and wastes 61/276, above the 5/220 bound, and it uses 4 batches.
    bucket = _fn(sub, "bucket_batches", "3 batches and waste at most 5/220")
    lengths = [40, 10, 42, 11, 100, 12]
    max_frames = 120
    batches = _call(bucket, (lengths, max_frames), "waste at most 5/220")
    groups = [[int(index) for index in batch] for batch in batches]
    flat = [index for batch in groups for index in batch]
    if sorted(flat) != list(range(len(lengths))):
        fail(f"indices are {sorted(flat)}, expected {list(range(len(lengths)))}")
    real = sum(lengths)
    padded = 0
    for batch in groups:
        if not batch:
            fail(f"a batch is empty. expected non-empty batches, got {groups}")
        longest = max(lengths[index] for index in batch)
        padded_batch = len(batch) * longest
        if len(batch) > 1 and padded_batch > max_frames:
            fail(
                f"batch {batch} pads to {padded_batch} frames, "
                f"expected at most {max_frames}"
            )
        padded += padded_batch
    waste = 1 - real / padded
    bound = 5 / 220
    if waste > bound + 1e-12:
        fail(f"padding waste is {waste:.6f}, expected at most {bound:.6f} (5/220)")
    if len(groups) != 3:
        fail(f"{len(groups)} batches, expected 3")
    alone = _call(bucket, ([50], 10), "[[0]]")
    alone_groups = [[int(index) for index in batch] for batch in alone]
    if alone_groups != [[0]]:
        fail(
            f"a 50-frame utterance with max_frames 10 returned {alone_groups}, "
            "expected [[0]]"
        )


@check("noam_lr matches the closed form, including the warmup")
def test_noam(sub):
    # `inverse_sqrt_without_warmup` fails at step 100: it returns 0.00625
    # where the closed form, still on the ramp, is 2.470529422e-05.
    lr = _fn(sub, "noam_lr", "the warmup ramp at step 100")
    d_model, warmup = 256, 4000
    for step in (1, 100, 4000, 16000):
        want = _closed_form(step, d_model, warmup)
        got = _call(lr, (step, d_model, warmup), f"{want:.6e} at step {step}")
        almost(got, want, 1e-12, label=f"step {step}")


@check("average_checkpoints matches the exact mean")
def test_average(sub):
    average = _fn(sub, "average_checkpoints", "w = [3, 3] and b = [3]")
    states = [
        {"w": np.array([0.0, 0.0]), "b": np.array([0.0])},
        {"w": np.array([0.0, 9.0]), "b": np.array([3.0])},
        {"w": np.array([9.0, 0.0]), "b": np.array([6.0])},
    ]
    got = _call(average, (states,), "w = [3, 3] and b = [3]")
    if set(got) != {"w", "b"}:
        fail(f"keys are {sorted(got)}, expected ['b', 'w']")
    close(got["w"], np.array([3.0, 3.0]), atol=1e-12, rtol=0, label="w")
    close(got["b"], np.array([3.0]), atol=1e-12, rtol=0, label="b")
