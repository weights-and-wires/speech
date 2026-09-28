/**
 * The demo registry.
 *
 * A lesson places a demo with a fenced ```demo block holding JSON, such as
 * `{"id": "stft-explorer", "caption": "..."}`, and `components/Markdown.tsx`
 * looks the id up here. Every demo is loaded lazily, so a lesson that uses
 * none of them pays nothing for the ones it does not use.
 *
 * `scripts/check_content.py` reads the ids out of this file, so a lesson can
 * never reference a demo that does not exist. Keep the `DEMOS` literal in a
 * shape the script can parse: one `"id": {` entry per demo, with the id
 * quoted, on a line of its own.
 *
 * `docs/DEMOS.md` explains how to add one.
 */

import { lazy, type ComponentType, type LazyExoticComponent } from "react";

/**
 * What every demo receives. `caption` is the block's sentence about what to
 * notice. A block's optional `props` object is spread in beside it, so a demo
 * that takes options declares them on top of these.
 */
export type DemoProps = { caption?: string };

type Entry = {
  /** The heading shown on the demo's frame. */
  title: string;
  Component: LazyExoticComponent<ComponentType<DemoProps & Record<string, unknown>>>;
};

export const DEMOS: Record<string, Entry> = {
  "sampling-and-aliasing": {
    title: "Sampling a sine, and the alias it can turn into",
    Component: lazy(() => import("./SamplingAndAliasing")),
  },
  "stft-explorer": {
    title: "Trading time for frequency in a spectrogram",
    Component: lazy(() => import("./StftExplorer")),
  },
  "quantization-and-dither": {
    title: "Quantizing a signal, with and without dither",
    Component: lazy(() => import("./Placeholder")),
  },
  "window-leakage": {
    title: "How a window trades main-lobe width for side-lobe leakage",
    Component: lazy(() => import("./Placeholder")),
  },
  "griffin-lim-iterations": {
    title: "Recovering phase from magnitude, one iteration at a time",
    Component: lazy(() => import("./Placeholder")),
  },
  "room-reverb": {
    title: "A room as an impulse response",
    Component: lazy(() => import("./Placeholder")),
  },
  "mel-filterbank": {
    title: "Laying triangles along the mel scale",
    Component: lazy(() => import("./Placeholder")),
  },
  "mfcc-pipeline": {
    title: "From a frame of speech to thirteen numbers",
    Component: lazy(() => import("./Placeholder")),
  },
  "vowel-formants": {
    title: "Vowels as a source and a filter",
    Component: lazy(() => import("./Placeholder")),
  },
  "pitch-tracker": {
    title: "Finding the pitch with YIN",
    Component: lazy(() => import("./Placeholder")),
  },
  "dtw-path": {
    title: "Warping one sequence onto another",
    Component: lazy(() => import("./Placeholder")),
  },
  "viterbi-trellis": {
    title: "The best path through an HMM trellis",
    Component: lazy(() => import("./Placeholder")),
  },
  "ctc-lattice": {
    title: "Every alignment CTC adds up",
    Component: lazy(() => import("./Placeholder")),
  },
  "beam-search": {
    title: "Keeping the best few hypotheses",
    Component: lazy(() => import("./Placeholder")),
  },
  "attention-alignment": {
    title: "Where the decoder looks at each step",
    Component: lazy(() => import("./Placeholder")),
  },
  "wavenet-receptive-field": {
    title: "How far back a dilated stack can see",
    Component: lazy(() => import("./Placeholder")),
  },
  "rvq-2d": {
    title: "Quantizing the residual, stage by stage",
    Component: lazy(() => import("./Placeholder")),
  },
  "codec-bitrate": {
    title: "Frames, codebooks, and bits per second",
    Component: lazy(() => import("./Placeholder")),
  },
  "delay-pattern": {
    title: "Interleaving codebooks with a delay",
    Component: lazy(() => import("./Placeholder")),
  },
  "diffusion-1d": {
    title: "Noising a signal and walking it back",
    Component: lazy(() => import("./Placeholder")),
  },
  "flow-trajectories": {
    title: "Straight paths from noise to data",
    Component: lazy(() => import("./Placeholder")),
  },
  "separation-pit": {
    title: "Masks, mixtures, and the permutation problem",
    Component: lazy(() => import("./Placeholder")),
  },
  "latency-timeline": {
    title: "Where the time goes in a voice agent",
    Component: lazy(() => import("./Placeholder")),
  },
};

export function getDemo(id: string): Entry | undefined {
  return DEMOS[id];
}
