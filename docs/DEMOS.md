# Demos

A demo is an interactive figure inside a lesson: a plot, a handful of controls,
and often a button that plays sound. It lets a reader push on an idea, such as
aliasing or the resolution of a spectrogram, before writing the code for it.
This guide explains how to build one.

Two demos exist as worked examples. Read them before you write a third:

- `frontend/src/demos/SamplingAndAliasing.tsx` samples a sine at an adjustable
  rate, draws the samples and the alias they reconstruct, and plays both the
  sine and the samples.
- `frontend/src/demos/StftExplorer.tsx` draws the spectrogram of a test signal
  under adjustable window length, hop, and window type, and records from the
  microphone on request.

## Add a demo

To add a demo, complete the following steps:

1. Create `frontend/src/demos/YourDemo.tsx`, whose default export is the
   component. It takes `DemoProps` from `./index`, which carries the block's
   `caption`, and renders a `DemoFrame` from `./kit`:

   ```tsx
   import { useState } from "react";

   import type { DemoProps } from "./index";
   import { Curve, DemoFrame, Plot, Slider } from "./kit";

   export default function YourDemo({ caption }: DemoProps) {
     const [hz, setHz] = useState(3);
     const points = Array.from({ length: 200 }, (_, i): [number, number] => {
       const t = i / 199;
       return [t, Math.sin(2 * Math.PI * hz * t)];
     });
     return (
       <DemoFrame
         title="A sine you can retune"
         caption={caption}
         controls={<Slider label="Frequency" value={hz} min={1} max={10} onChange={setHz} />}
       >
         <Plot xDomain={[0, 1]} yDomain={[-1.1, 1.1]} xLabel="time (s)">
           {() => <Curve points={points} />}
         </Plot>
       </DemoFrame>
     );
   }
   ```

2. Register it in `frontend/src/demos/index.ts`. The id is kebab-case, quoted,
   and on a line of its own, because `scripts/check_content.py` reads the ids
   from this file with a regular expression:

   ```ts
   "your-demo": {
     title: "A sine you can retune",
     Component: lazy(() => import("./YourDemo")),
   },
   ```

3. Place it in a lesson with a fenced block tagged `demo`:

   ````markdown
   ```demo
   {"id": "your-demo", "caption": "Raise the frequency and count the cycles."}
   ```
   ````

4. Check the lesson and build the site:

   ```bash
   .venv/bin/python scripts/check_content.py SLUG
   cd frontend && npm run build
   ```

   Replace `SLUG` with the lesson's slug.

5. Build the book and look at the demo's page, because the book prints the demo
   as a still figure:

   ```bash
   node frontend/scripts/build-book.mjs --force
   ```

The `title` in the registry and the `title` passed to `DemoFrame` must match.
The contract's [Demos](CONTENT_CONTRACT.md#demos) section has the rules for
using a demo in a lesson: a caption is required, and a lesson holds at most
three.

## Props

A demo block can carry a `props` object, which is spread into the component
beside `caption`. Use it for a demo that more than one lesson needs, configured
differently in each. Declare the props the demo takes on top of `DemoProps`,
and give every one a default, because a block without `props` must still work:

```tsx
type Props = DemoProps & { sampleRate?: number };

export default function YourDemo({ caption, sampleRate = 16000 }: Props) {
  // ...
}
```

A block then passes them as follows:

````markdown
```demo
{"id": "your-demo", "caption": "...", "props": {"sampleRate": 8000}}
```
````

Validate a prop's value inside the demo, and fall back to the default when it
is out of range. The content checker confirms only that `props` is an object.

## The kit

`frontend/src/demos/kit.tsx` holds the frame, the controls, and the plots every
demo is built from, so that all of them look like one set. Every colour comes
from a theme token, such as `var(--demo)`, `var(--ink-3)`, or `var(--bad)`,
so a demo works in the light and dark themes without extra work. Never write a
raw colour in a demo.

### The frame

The frame components are the following:

| Name | Signature | Use |
|---|---|---|
| `DemoFrame` | `DemoFrame({ title, caption?, controls?, children })` | The card: title, caption, the figure, and a control strip that is hidden in print. |
| `PrintedFigure` | React context, `{ url } \| null` | Set by the PDF book. A demo reads it to know it is printed. |
| `useWhenPrinted` | `useWhenPrinted(run: () => void)` | Runs `run` once on mount in the book, for a demo that starts empty. |

### Controls

The control components are the following:

| Name | Signature | Use |
|---|---|---|
| `Slider` | `Slider({ label, value, min, max, step?, format?, onChange })` | A range input with its value beside the label. |
| `Choice` | `Choice<T>({ label, value, options: { value, label }[], onChange })` | A row of radio buttons. |
| `Toggle` | `Toggle({ label, checked, onChange })` | A checkbox. |
| `Action` | `Action({ children, onClick, tone?: "primary" \| "quiet" })` | A button. |
| `Actions` | `Actions({ children })` | A row of buttons. |
| `Readout` | `Readout({ label, value, hint?, tone?: "plain" \| "demo" \| "good" \| "bad" })` | One labelled number. |
| `Readouts` | `Readouts({ children })` | A responsive strip of readouts. |

### SVG plots

`Plot` draws axes in SVG and passes its scales to its children, which draw in
data coordinates. It scales to its column and keeps its aspect ratio.

| Name | Signature | Use |
|---|---|---|
| `Plot` | `Plot({ width?, height?, xDomain, yDomain, xLabel?, yLabel?, xTicks?, yTicks?, tickFormat?, onPointer?, children: (scales) => ReactNode })` | The axes. `onPointer` reports pointer events in data coordinates. |
| `Curve` | `Curve({ points, stroke?, width?, dashed?, fill?, opacity? })` | A polyline, optionally filled to the axis. |
| `Bars` | `Bars({ bins: { x0, x1, y }[], fill?, stroke?, gap? })` | A histogram. |
| `Rule` | `Rule({ x, label?, colour?, dashed? })` | A vertical line, such as the Nyquist frequency. |
| `Points` | `Points({ points, r?, fill?, stroke?, opacity? })` | Dots. |
| `Stems` | `Stems({ points, colour?, r? })` | Samples drawn as stems from zero, the way a textbook draws a sampled signal. |
| `Waveform` | `Waveform({ samples, sampleRate, colour?, width? })` | A clip against time in seconds, drawn from its envelope. |
| `Legend` | `Legend({ items: { label, colour, dashed? }[] })` | A key under the plot. |
| `useScales` | `useScales(): Scales` | The enclosing plot's scales, for a child component. |
| `envelope` | `envelope(samples, sampleRate, buckets = 600, start?, end?)` | The lowest and highest sample of each bucket, as points. |

### Canvas figures

A spectrogram has too many cells for SVG, so it draws on a canvas. The run
console uses the same components for `speechlab` figures.

| Name | Signature | Use |
|---|---|---|
| `Spectrogram` | `Spectrogram({ spec, sampleRate, floorDb = -80, rows?, colormap = "magma", height?, description? })` | A spectrogram in decibels, from `stft` and `toDb`, with time and frequency axes. `rows` keeps a band `[first, last)` for a zoom. |
| `HeatmapCanvas` | `HeatmapCanvas({ levels, rows, cols, origin?, colormap?, x, y, value?, height?, colorbar?, description? })` | Any heatmap of 8-bit levels, with axes, a colour bar, and a tooltip. A `mel` y axis spaces rows in mels. |
| `LineCanvas` | `LineCanvas({ series, xlabel?, ylabel?, xScale?, yScale?, xlim?, ylim?, height?, description? })` | Line plots with log axes, a legend, and a crosshair. |
| `quantize` | `quantize(values, min, max): Uint8Array` | Values to levels for `HeatmapCanvas`. |

The colour maps are `magma`, `viridis`, `inferno`, and `gray`, from
`frontend/src/lib/colormap.ts`. They run from dark to light, so quiet cells
stay dark in both themes.

### Sound

`frontend/src/demos/audio.ts` plays and records. The kit re-exports its hooks.

| Name | Signature | Use |
|---|---|---|
| `useAudioPlayer` | `useAudioPlayer(): { play(name, samples, sampleRate, gain?), stop(), playing }` | One sound at a time; a new one stops the last, and leaving the page stops it. `playing` names what is playing. |
| `useMicrophone` | `useMicrophone(): { state, record(seconds, sampleRate) }` | Records from the microphone on request. See [The microphone](#the-microphone). |
| `playSamples` | `playSamples(samples, sampleRate, { gain?, onEnded? }): () => void` | The function under `useAudioPlayer`. |
| `audioContext` | `audioContext(): AudioContext` | The page's one audio context. |

`playSamples` builds an `AudioBuffer` at the samples' own rate, from 3 kHz to
384 kHz, and the browser resamples it to the output with its own band-limited
filter. So a buffer at 8 kHz plays what those samples encode, aliases
included, which is how the sampling demo lets a reader hear an alias.

Keep playback quiet and short. Use a gain of 0.25 for a full-scale sine, and
fade the ends with `fadeEdges` so a clip doesn't start or stop with a click.

### Signal processing

`frontend/src/demos/dsp.ts` matches numpy, scipy, and librosa, so a demo shows
the numbers the reader's own code computes. `npm test` in `frontend/` checks it
against reference values from those libraries.

| Name | Signature | Matches |
|---|---|---|
| `fft` | `fft(re: Float64Array, im: Float64Array, inverse = false): void` | `numpy.fft.fft` and `ifft`, in place, radix 2. |
| `rfftMagnitude` | `rfftMagnitude(frame, n = nextPowerOfTwo(frame.length)): Float64Array` | `abs(numpy.fft.rfft(frame, n))`. |
| `hann`, `hamming` | `hann(n, periodic = true): Float64Array` | `scipy.signal.get_window`. |
| `makeWindow` | `makeWindow(name: "rectangular" \| "hann" \| "hamming", n, periodic = true)` | The same, by name. |
| `stft` | `stft(signal, { nFft, hop, window?, center = true, padMode = "reflect" }): Spectrogram` | `abs(librosa.stft(..., pad_mode="reflect"))`: `bins` rows by `frames` columns, row-major. |
| `toDb` | `toDb(values, { floorDb = -80, power = false, ref? }): Float32Array` | `librosa.amplitude_to_db(ref=max, top_db=-floorDb)`. |
| `hzToMel`, `melToHz` | `hzToMel(hz, formula: "slaney" \| "htk" = "slaney")` | `librosa.hz_to_mel`, with `htk=True` for `"htk"`. |
| `melFilterbank` | `melFilterbank({ sampleRate, nFft, nMels, fmin = 0, fmax = sampleRate / 2, formula = "slaney", norm = "slaney" }): Float64Array[]` | `librosa.filters.mel`. |
| `applyFilterbank` | `applyFilterbank(bank, spectrogram): Spectrogram` | `bank @ spectrogram`. |
| `dct2` | `dct2(x, norm: "ortho" \| "backward" = "ortho"): Float64Array` | `scipy.fft.dct(x, type=2, norm=...)`. |
| `sine`, `linearChirp` | `sine(hz, sampleRate, seconds, amplitude = 0.5, phase = 0): Float32Array` | Test signals. |
| `fadeEdges` | `fadeEdges(samples, sampleRate, seconds = 0.02): Float32Array` | A raised-cosine fade in and out, in place. |
| `linspace`, `isPowerOfTwo`, `nextPowerOfTwo` | | Helpers. |

When you add a function to `dsp.ts`, add a test to `frontend/tests/dsp.test.ts`,
and when the test needs reference values, add them to
`frontend/tests/make_dsp_reference.py` and run it. Keep `dsp.ts` free of
browser APIs and of TypeScript features Node can't strip, such as `enum`, so
the tests run on it without a build step.

## The still figure in the book

The PDF book renders every demo inside `PrintedFigure`, hides its controls and
buttons, and adds a line that names the lesson online. What prints is the demo's
first render, so the following rules apply:

- The starting state must be worth printing. Choose defaults that show the
  point of the demo, the way the sampling demo starts on an aliased sine.
- A demo that starts empty and waits for a button prints an empty plot. Call
  `useWhenPrinted` to run its simulation once in the book.
- Never depend on hover, animation, or sound for the figure to make sense.
- Draw within the column. A canvas sizes itself to its container; an SVG
  `Plot` scales to it. The book build fails on anything wider than the page.
- Use seeded randomness, so the figure is the same in every build.

## Performance

A lesson can hold three demos, and a reader drags sliders on a phone, so keep
each one light:

- Recompute inside `useMemo`, keyed on the controls, never on every render.
- Keep each update under about 16 ms on a laptop. The STFT demo transforms
  16,000 samples with windows up to 2048 samples, which is well inside that.
  Past about a million multiply-adds for each update, compute on a coarser
  grid, or debounce the slider.
- Keep signals to a few seconds at 16 kHz, and plots to a few thousand points.
  `Waveform` draws the envelope, so a long clip costs no more than a short one.
- Keep a demo's own code small. Each demo is its own lazy chunk and loads only
  when a lesson uses it. Import from `./kit` and `./dsp`, not a charting or
  audio library.
- Start no timers, audio, or animation on load. `useAudioPlayer` stops its sound
  when the demo leaves the page; anything else you start, stop in an effect's
  cleanup.

## The microphone

A demo can record from the microphone, but only when the reader asks. The
following rules apply:

- Open the microphone only from a click on a button that says what it does,
  such as **Record from the microphone**. Never on load, never on scroll, and
  never from a slider.
- The demo must work in full without it. The microphone replaces a built-in
  signal; it is never the only signal.
- Handle every outcome. `useMicrophone().state` is `asking`, `recording`,
  `denied`, `unsupported`, or `failed`; show a sentence for each, and carry on
  with the built-in signal.
- Record a fixed, short length, a few seconds, and close the stream as soon as
  the recording ends. `record` does both.
- Offer a way back to the built-in signal, such as **Back to the test signal**.
- Never send a recording anywhere. It stays in the page's memory.

The browser allows the microphone only on HTTPS or `localhost`, and asks the
reader's permission the first time.

The following code sample shows the pattern from `StftExplorer.tsx`:

```tsx
const microphone = useMicrophone();
const [recorded, setRecorded] = useState<Float32Array | null>(null);
const signal = recorded ?? TEST_SIGNAL;

async function listenToMicrophone() {
  const samples = await microphone.record(2, 16000);
  if (samples) setRecorded(samples);
}
```
