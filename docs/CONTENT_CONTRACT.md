# Content contract

Every lesson in this course is two things on disk: one Markdown file that the
reader reads, and one exercise directory that the reader's code is checked
against. This document is the exact contract both must satisfy. The loader in
`backend/app/curriculum.py` and the harness in `backend/app/runner/harness.py`
depend on it.

## Audience

Readers are undergraduates and working engineers who want to understand how
machines hear and speak, from a sampled waveform to a speech language model.
They can write Python and have used numpy. They have met calculus, linear
algebra, and probability once, and remember some of it. They know what a
neural network, gradient descent, backpropagation, and a transformer are, at
the level of the companion machine learning course: they can train a small
model in PyTorch, but they have not trained one on audio. They know nothing
about signal processing, acoustics, phonetics, or speech, and the course
teaches all of it from scratch. Write for them:

- Start with the intuition in plain words. Before any formula, say what the
  method does, in one or two sentences a first-year student could repeat to a
  friend. Use an everyday analogy when one fits: a spectrogram is sheet music,
  a codec is a very short vocabulary for sound.
- Make every formula earn its place. Include a formula only when the exercise
  implements it or the reader needs it to follow the next step. Everything else
  is stated in words.
- Read each formula aloud. After every display equation, say in plain English
  what each piece is and what the whole thing computes. Then show the same
  formula as one or two lines of numpy or torch.
- Work one small example by hand. Use tiny numbers, a four-sample signal, a
  two-state HMM, a three-frame CTC lattice, and show every intermediate value,
  so the reader can check their code against it.
- Define every symbol and every term the first time it appears, including ones
  that feel basic, such as "sample rate", "frame", "bin", or "decibel". Prefer
  an ordinary word to jargon when both mean the same thing.
- Say the units. A number in this course is in samples, frames, seconds, hertz,
  mel, or decibels, and the reader must always know which. Most bugs in audio
  code are unit bugs: frames confused with samples, power confused with
  magnitude, natural log confused with `log10`.
- Make it audible. When a lesson is about sound, the reader should hear it: the
  alias, the leakage, the reconstruction, the generated clip. The **Run** block
  plays what the reader built.
- Keep the conditions honest. When a result needs an assumption, such as a
  stationary signal inside one frame, say so in plain words and say what goes
  wrong without it.
- Connect ideas to code. The reader should finish every section knowing which
  numpy or torch operations they will write.
- Name the systems. When an idea lives on in a real model, such as the mel
  front end in Whisper or residual vector quantization in EnCodec, say so, with
  the concrete numbers that model uses. The research notes in `docs/research/`
  have them, with sources. Never state a model's size, date, or score that the
  notes mark as unverified without checking it yourself.

Every result on the page must still be correct. Simpler wording must never
change what a result says.

Use American spelling: "normalization", "color", "modeling".

## Prose style

Follow the Google developer documentation style guide:

- Address the reader as "you". Use active voice and present tense.
- One idea per sentence. Split any sentence that needs a second comma-spliced
  clause to land. Long subordinate chains are the main thing that made these
  lessons hard to read.
- Sentence case for every heading. Start task headings with a bare infinitive.
- Serial commas. Em dashes with no surrounding spaces.
- Code, filenames, and identifiers in code font. No `simply`, `just`, `easily`,
  `obviously`, `note that`, `e.g.`, `i.e.`, `etc.`
- Don't write timelessly-rotting words: `currently`, `now`, `new`, `latest`,
  `recently`, `state of the art`.
- Introduce every list, table, and code block with a complete sentence.

## Lesson file

Path: `content/lessons/NN-slug.md`, where `NN` is the two-digit order prefix and
`slug` matches the `slug` field.

The file starts with YAML front matter:

```yaml
---
title: The short-time Fourier transform
slug: 05-the-stft
part: "Part 1 — Signals"
summary: One sentence, under 140 characters, that says what the lesson delivers.
minutes: 55
gpu: false
objectives:
  - Cut a signal into overlapping windowed frames, and take the FFT of each.
  - "Explain the trade-off between time and frequency resolution: a longer window sees finer frequency and coarser time."
  - Draw a spectrogram in decibels and read a chirp, a tone, and a click from it.
prereqs:
  - 03-sampling
  - 04-the-dft
exercise: 05-the-stft
---
```

Every value that holds a colon must be wrapped in double quotes, or YAML reads
the item as a mapping rather than a string. A mapping that reaches the client
renders as nothing at all, so `scripts/check_content.py` rejects one.

Every field is required except `prereqs`, which can be omitted. `gpu` is `true`
only when the exercise genuinely needs a GPU. `minutes` is an honest reading and
working estimate.

The body is Markdown with these conventions:

- One `#` heading, matching `title`, as the first line after the front matter.
- `##` for sections, `###` for subsections. Never skip a level.
- Math in KaTeX: `$inline$` and `$$display$$`. Put each `$$` on its own line
  above and below the equation. A one-line `$$...$$` parses as inline math and
  renders at text size, so `scripts/check_content.py` rejects it. The renderer loads
  `remark-math` and `rehype-katex`, so use KaTeX-supported commands only.
  `\operatorname`, `\mathbb`, `\mathcal`, `\begin{bmatrix}`, `\begin{aligned}`,
  `\top`, `\lVert`, and `\argmin` via `\operatorname*{arg\,min}` all work.
- Fenced code blocks tagged with a language. Python blocks show the shape of an
  implementation, not a full solution to the exercise.
- Tables for genuinely two-dimensional data.
- Callouts, written as a blockquote whose first line is a type marker. The
  reader renders each type as a colored box with its own label:

  ```markdown
  > [!KEY]
  > A trained model's error on its own training data is lower than its true
  > error, on average.
  ```

  | Marker | Label | Use it for |
  |---|---|---|
  | `[!KEY]` | Key idea | The one sentence to remember from a section. Two to four per lesson. |
  | `[!TIP]` | Tip | A practical shortcut or habit. At most two per lesson. |
  | `[!WARNING]` | Watch out | A trap the reader is about to fall into. At most two per lesson. |
  | `[!RECAP]` | Before you build | Three to five bullets, placed right before `## Build it step by step`. One per lesson. |

  `scripts/check_content.py` rejects any other marker, and a second recap.
- Interactive demos, written as a fenced block tagged `demo` that holds one JSON
  object. For more information, see [Demos](#demos).

Every lesson body includes, in this order:

1. An opening that says, in plain words, what problem the lesson solves and why
   it matters. Two or three short paragraphs. No heading, and no formulas.
2. `## What you already know` — a short list of the ideas the lesson builds on,
   each in one plain sentence. Keep it to five bullets.
3. The substance, in as many `##` sections as the material needs. Each section
   leads with intuition, then gives the formula, reads it aloud, and shows it
   as numpy. Include at least one small example worked by hand with real
   numbers.
4. `## Build it step by step` — a numbered walkthrough of the exercise, one
   step per function in the order the checks test them. Each step says what the
   function takes and returns, the numpy operations it needs, and the mistake
   people most often make. It gives the shape of the code, never the full
   answer.
5. `## Cost and scaling` — how the running time and memory grow as the data or
   the model gets bigger, in plain words. A short table works well.
6. `## Where this breaks` — the common ways the method fails in practice, and
   what people do about each one.
7. `## The exercise` — one or two paragraphs that say what the reader builds
   and what the checks verify. Name the functions by their exact identifiers.

Aim for 1200 to 2200 words of body, plus math. Short sentences and short
paragraphs matter more than the word count.

## Exercise directory

Path: `content/exercises/<id>/`, where `<id>` matches the lesson's `exercise`
field. It holds exactly four files.

### `meta.json`

```json
{
  "id": "05-the-stft",
  "title": "Build a spectrogram from frames",
  "brief": "Frame a signal, window it, and take the magnitude of each frame's FFT.",
  "needs_gpu": false,
  "timeout": 60,
  "packages": ["numpy"],
  "hints": [
    "Build the frame indices with broadcasting: np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None].",
    "np.fft.rfft(frames * window, axis=1) gives one row per frame; transpose it so rows are frequencies."
  ]
}
```

`packages` is documentation only. The CPU runtime carries `numpy` 2.3.3,
`scipy` 1.16.2, `torch` 2.9.1 (CPU build), and `soundfile` 0.14.0. The GPU
worker carries the same `numpy` and `scipy`, `torch` 2.9.1 with CUDA 12.8,
`soundfile` 0.14.0, `transformers` 5.17.0, `huggingface_hub` 1.33.0, and
`safetensors` 0.8.0. Anything else is unavailable; don't list it. `timeout` is in seconds and must leave headroom:
set it to about four times the reference solution's runtime, and never above 120
for a CPU exercise.

### `starter.py`

What the reader sees when the lesson opens. It must:

- Import only from the standard library, `numpy`, `scipy`, `soundfile`,
  `speechlab`, and, when the lesson needs it, `torch`. A `gpu: true` lesson can
  also import `transformers`, `huggingface_hub`, and `safetensors`, but its
  checks must pass on the CPU without them.
- Define every function the checks call, with the exact signature, a docstring
  that states the contract including array shapes, and a body that raises
  `NotImplementedError`.
- Carry a `if __name__ == "__main__":` block that builds a small example and
  prints something informative. The **Run** button executes this block, so it must
  work once the reader fills in the functions, and it must fail with a clear
  `NotImplementedError` before that. `scripts/check_content.py` presses **Run**
  on both files to confirm it. When the lesson is about sound, the block plays
  what the reader built with `speechlab.play` and draws it with
  `speechlab.show_spectrogram` or `show_plot`. For more information, see
  [Output: speechlab](#output-speechlab).
- Never contain the answer, in a comment or anywhere else.

### `solution.py`

A correct, idiomatic reference implementation with the same public surface as
`starter.py`. It must pass every check in `tests.py`. Keep it clean enough to read
as the worked answer, because the reader can reveal it.

### `tests.py`

The checks. Import the harness helpers and register each check with the
`@check` decorator:

```python
"""Checks for the short-time Fourier transform exercise."""

import numpy as np

from speechcheck import check, close, fail, near_frequency, shape, tone

RATE = 16000


@check("frame_signal keeps only whole frames")
def test_frames(sub):
    x = np.arange(20, dtype=float)
    frames = sub.frame_signal(x, 8, 4)
    shape(frames, (4, 8), label="frames")
    close(frames[1], x[4:12], label="frame 1")


@check("A pure tone peaks in its own bin")
def test_tone(sub):
    x = tone(40 * RATE / 512, RATE, 0.25, amplitude=1.0)
    spec = sub.magnitude_spectrogram(x, 512, 128)
    if not np.all(np.argmax(spec, axis=0) == 40):
        fail(f"the strongest bins are {np.unique(np.argmax(spec, axis=0))}, expected 40")
```

The `speechcheck` module is on the path at run time. Its API:

| Name | Signature | Behavior |
|---|---|---|
| `check` | `@check("description")` | Registers a check. The description is what the reader sees. |
| `close` | `close(got, want, rtol=1e-5, atol=1e-8, label=None)` | Fails with a readable diff when the arrays differ. |
| `equal` | `equal(got, want, label=None)` | Exact equality, for shapes, dtypes, and integers. |
| `fail` | `fail("message")` | Fails the check with that message. |
| `shape` | `shape(array, (3, 4), label=None)` | Asserts a shape. |
| `almost` | `almost(got, want, tol, label=None)` | Scalar comparison within `tol`. |
| `timed` | `with timed("matmul"):` | Records elapsed seconds, reported to the reader. |
| `finite` | `finite(array, label=None)` | Fails when any element is NaN or infinite. |
| `in_range` | `in_range(array, low, high, label=None)` | Fails unless every element lies in `[low, high]`. |
| `has` | `has(module, "name")` | Returns the attribute, failing readably when it is missing. |
| `finite_difference` | `finite_difference(func, x, eps=1e-6)` | Central-difference gradient of a scalar function. |
| `tone` | `tone(frequency, sample_rate, seconds, amplitude=0.5, phase=0.0)` | A sine wave, for a deterministic test signal. |
| `chirp` | `chirp(f0, f1, sample_rate, seconds, amplitude=0.5, method="linear")` | A sweep, linear or exponential (`"log"`) in frequency. |
| `white_noise` | `white_noise(sample_rate, seconds, amplitude=0.1, seed=0)` | Seeded Gaussian noise. |
| `snr_db` | `snr_db(got, want)` | The signal-to-noise ratio of `got` against the reference `want`, in dB. |
| `min_snr` | `min_snr(got, want, db, label=None)` | Fails unless that ratio is at least `db`. |
| `peak_frequency` | `peak_frequency(signal, sample_rate)` | The frequency of the largest bin of the Hann-windowed spectrum. |
| `near_frequency` | `near_frequency(signal, sample_rate, hz, tol_hz=None, label=None)` | Fails unless the strongest frequency is within `tol_hz`, two bins by default. |

Rules for checks:

- Each check receives `sub`, the reader's module, as its only argument.
- Seed every random number generator. `np.random.default_rng(0)` for numpy,
  `torch.manual_seed(0)` for torch. A check that passes intermittently is a bug.
- Write six to ten checks per exercise, ordered from the smallest piece to the
  whole. The first check must pass as soon as the reader gets one function right,
  so the pane shows progress early.
- Check the mathematics, not the implementation. Verify a gradient against a
  finite-difference approximation rather than against a particular loop.
- Never import `solution` from `tests.py`.
- Compare audio against a reference with `min_snr` or a frequency check, not
  sample by sample, unless the operation is exact, such as framing.
- A check can draw with `speechlab`, and the figure appears in the console
  marked with the check that drew it. Draw only what helps the reader see a
  failure, at most one figure per check.
- Keep the whole suite under the `timeout` in `meta.json`, with margin. On CPU
  that means matrices no larger than about 2000 by 200, and signals no longer
  than a few seconds at 16 kHz. The CPU runner has about 700 MB of memory for
  each of its two workers.
- Failure messages must say what was expected and what arrived. The reader cannot
  see your code.

## Output: speechlab

`speechlab` sits next to the harness, so a starter file, a solution, and a check
can all import it. Each call sends one event to the page, which renders it in
the console in the order it happened, between the prints around it.

The following table lists the helpers.

| Name | Signature | What the page shows |
|---|---|---|
| `play` | `play(samples, sample_rate, label=None)` | An audio player with the label and duration, and a **Download WAV** link. |
| `show_spectrogram` | `show_spectrogram(matrix, sample_rate=None, hop_length=None, label=None, db=True, y_axis="linear", *, fmin=0.0, fmax=None, htk=False, power=False, top_db=80.0, colormap="magma", value_label=None)` | A heatmap with time in seconds and frequency in Hz, a color bar, and a tooltip. |
| `show_image` | `show_image(matrix, label=None, *, xlabel=None, ylabel=None, x_labels=None, y_labels=None, colormap="viridis", vmin=None, vmax=None, origin="upper", value_label=None)` | A generic heatmap, such as an attention or alignment map, with named rows and columns. |
| `show_plot` | `show_plot(*series, label=None, xlabel=None, ylabel=None, labels=None, x_scale="linear", y_scale="linear", xlim=None, ylim=None)` | Line plots of up to six series, each a 1-D array of y values or an `(x, y)` tuple. |
| `show_waveform` | `show_waveform(samples, sample_rate, label=None)` | A clip against time in seconds. |
| `read_audio` | `read_audio(path)` | Nothing; returns `(samples, sample_rate)` as float32, the layout `play` takes. |
| `asset_path` | `asset_path(name)` | Nothing; returns the path of a baked-in asset. See [Assets](#assets). |
| `has_asset` | `has_asset(name)` | Nothing; returns whether the asset is on this runner. |

The following rules apply to the arguments:

- `play` takes a 1-D array, or a `(channels, samples)` array with one or two
  channels. Floats must lie in `[-1, 1]`; anything outside is clipped, with a
  note. `int16` is sent as it is.
- `show_spectrogram` takes a `(frequency bins, frames)` matrix with row 0 the
  lowest frequency, the layout `np.abs(np.fft.rfft(frames, axis=1)).T` gives.
  With `db=True` it converts magnitudes to decibels itself; pass `db=False`
  for a matrix that is already in dB, and `power=True` for a power spectrogram.
  `y_axis="mel"` labels mel bands between `fmin` and `fmax`.
- Torch tensors on any device are accepted wherever an array is.

Every payload is bounded. The limits are the following:

| Limit | Value | What happens past it |
|---|---|---|
| One clip | 30 seconds, or about 2 MB of 16-bit WAV | The clip is cut, and a note says so. |
| One heatmap | 256 rows by 512 columns | Cells are pooled: the maximum for a spectrogram, the mean for `show_image`. |
| One series | 2000 points | The lowest and highest point of each of 1000 buckets are kept. |
| Tick labels | 64 an axis | The labels are dropped, and a note says so. |
| One run | 32 clips and figures, 8 MB in all | Later calls send nothing, and one note says so. |

`scripts/check_content.py` fails an exercise whose solution plays a clip that
gets cut or passes the run limit. Keep clips to a few seconds at 16 kHz, which
is also what keeps a run fast.

## Demos

A demo is an interactive figure inside a lesson: sliders, a plot, and often a
button that plays sound. A lesson places one with a fenced block tagged `demo`
that holds one JSON object:

````markdown
```demo
{"id": "stft-explorer", "caption": "Lengthen the window until the two tones separate."}
```
````

The following rules apply:

- `id` must name a demo in `frontend/src/demos/index.ts`.
  `scripts/check_content.py` rejects an unknown id.
- `caption` is required. It is one sentence that says what to do and what to
  notice, shown under the demo's title.
- `props` is optional: an object passed to the demo as extra properties, for a
  demo that documents the options it takes. No other key is allowed.
- Use at most three demos in a lesson. Introduce each with a sentence in the
  text, and never make a demo the only place a result appears: the PDF book
  prints it as a still figure.
- Place a demo after the idea it illustrates, not before it.

To build a demo, see `docs/DEMOS.md`.

## Assets

A lab that needs a pretrained model, a tokenizer, or a recording loads it from
disk with `speechlab.asset_path`, never from the network at run time.
`gpu/assets.json` lists the assets, and `gpu/fetch_assets.py` downloads them
into the GPU worker's image at `/assets/<name>` when the image is built. Each
entry is one of the following:

```json
{"name": "whisper-tiny", "type": "huggingface", "repo": "openai/whisper-tiny",
 "revision": "COMMIT_HASH", "repo_type": "model",
 "files": ["config.json", "model.safetensors", "*.json"]}
```

```json
{"name": "hello-clip", "type": "url", "url": "https://example.com/hello.wav",
 "sha256": "SHA256_HEX", "filename": "hello.wav", "extract": null}
```

Replace `COMMIT_HASH` with the repository's 40-character commit, and
`SHA256_HEX` with the file's 64-character digest. The following rules apply:

- Pin every asset: a Hugging Face `revision` is a commit hash, and a URL has its
  `sha256`. The fetcher rejects anything else.
- List only the files the lab loads. The whole image must stay small enough to
  start quickly; the fetcher fails past 12 GiB in all.
- Only a `gpu: true` lesson can depend on an asset. On the CPU runner,
  `SPEECH_ASSETS` points at the repository's Git-ignored `assets/`, which is
  usually empty, so the lab checks `has_asset(name)` first and falls back to
  something smaller, or says what it would have done. Its checks never need an
  asset.

## GPU lessons

A lesson is `gpu: true`, and its exercise `needs_gpu: true`, when its **Run**
block does something that is worth a GPU: training a model for a few minutes,
or running a pretrained model over real recordings. The following rules apply:

- The checks never need the GPU. They run on the CPU, on synthetic data, with
  no asset and no `transformers`, inside the usual CPU limits. The reader's
  core functions are what the checks verify. `scripts/check_content.py` runs
  them on the CPU in CI.
- The **Run** block is the experiment. It picks
  `device = "cuda" if torch.cuda.is_available() else "cpu"` and prints which
  one it got.
- On a GPU, the experiment finishes in under ten minutes on an A100, and the
  exercise's `timeout` covers it with margin, up to 900 seconds.
- Without a GPU, the same block shrinks itself: fewer steps, fewer clips, a
  smaller model, so it finishes in under 60 seconds on the CPU. It says, in one
  printed line, that it is running the small version and why. The CPU runner
  caps a run at 180 seconds, and `scripts/check_content.py` presses **Run** on
  the CPU.
- Without its assets, the block says which asset it needs and falls back to
  synthetic data, or stops cleanly with that message. It never downloads.
- The assets available are the ones in `gpu/assets.json`. Load models with
  `transformers` from `speechlab.asset_path(name)`, and read audio with
  `speechlab.read_audio`. The following table lists them.

| Name | What it is | License | Load it with |
|---|---|---|---|
| `whisper-tiny` | `openai/whisper-tiny`, 39M parameters, with its tokenizer and feature extractor configs | Apache-2.0 | `WhisperForConditionalGeneration`, `WhisperTokenizer` |
| `wav2vec2-base` | `facebook/wav2vec2-base`, 95M, self-supervised on LibriSpeech 960 h, not fine-tuned | Apache-2.0 | `Wav2Vec2Model` |
| `hubert-base` | `facebook/hubert-base-ls960`, 95M, self-supervised, not fine-tuned | Apache-2.0 | `HubertModel` |
| `encodec-24khz` | `facebook/encodec_24khz`, 15M, 1.5 to 24 kbps | Code MIT, weights CC-BY-NC-4.0 | `EncodecModel` |
| `mimi` | `kyutai/mimi`, 79M, 12.5 Hz frames, 8 codebooks in `transformers` | CC-BY-4.0 | `MimiModel` |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M`, a 135M-parameter base language model | Apache-2.0 | `AutoModelForCausalLM`, `AutoTokenizer` |
| `fsdd` | Free Spoken Digit Dataset v1.0.10: 3,000 recordings of the digits 0 to 9 by six speakers, 8 kHz WAV, under `free-spoken-digit-dataset-1.0.10/recordings/` as `{digit}_{speaker}_{index}.wav` | CC-BY-SA-4.0 | `speechlab.read_audio` |
| `librispeech-dev-clean` | LibriSpeech dev-clean: 5.4 hours, 40 speakers, 16 kHz FLAC, under `LibriSpeech/dev-clean/{speaker}/{chapter}/`, with one `*.trans.txt` transcript file per chapter | CC-BY-4.0 | `speechlab.read_audio` |

To add an asset, add it to `gpu/assets.json` with a pinned revision or
checksum, and say why in the lesson's pull request. The worker image grows with
every asset, and a larger image starts slower.

## Mathematical accuracy

Every result, gradient, and complexity claim in a lesson must be correct. A
lesson that states a result without deriving it carries the same burden as one
that derives it: if it is on the page, it is right. When you write a gradient in
prose, the check that verifies it in `tests.py` must use finite differences, so
the prose and the code confirm each other. If those two disagree, the lesson is
wrong, not the reader.

Cutting a derivation must never change what a result says. Shortening the path
to a formula is editing; changing the formula is breaking the lesson.
