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
};

export function getDemo(id: string): Entry | undefined {
  return DEMOS[id];
}
