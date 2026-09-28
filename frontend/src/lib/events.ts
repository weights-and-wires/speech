/**
 * The reading pane and the editor live in different subtrees, and the lesson
 * text is rendered from Markdown, so they talk through one window event rather
 * than threading callbacks through every component in between.
 */

export const REVEAL_EVENT = "speech:reveal";

/** Which symbol to show, or none for the first thing left to write. */
export type RevealDetail = { symbol?: string };

export function revealInEditor(symbol?: string) {
  window.dispatchEvent(new CustomEvent<RevealDetail>(REVEAL_EVENT, { detail: { symbol } }));
}
