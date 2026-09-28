/**
 * What the canvas figures share: tick placement, theme colours, and sizing.
 *
 * A canvas cannot read CSS variables as it paints, so a figure reads the
 * theme's colours once per draw with `themeColours`, and redraws when the
 * theme changes, which `useThemeVersion` reports. `useWidth` tracks the width
 * of the figure's column, so a figure is never wider than its container and
 * behaves on a phone without thinking about it.
 */

import { useEffect, useState, type RefObject } from "react";

import { hzToMel, melToHz, type MelFormula } from "../demos/dsp";

// Ticks -----------------------------------------------------------------

/** Round tick values between `lo` and `hi`: 1, 2, or 5 times a power of ten apart. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi < lo) [lo, hi] = [hi, lo];
  if (hi === lo) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power;
  const out: number[] = [];
  const first = Math.ceil(lo / step - 1e-9) * step;
  for (let v = first; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Ticks at 1, 2, and 5 times each power of ten in range, for a log axis. */
export function logTicks(lo: number, hi: number): number[] {
  if (!(lo > 0) || !(hi > lo)) return [];
  const out: number[] = [];
  const decades = Math.log10(hi / lo);
  const mantissas = decades > 4 ? [1] : decades > 2 ? [1, 3] : [1, 2, 5];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
    for (const m of mantissas) {
      const v = m * 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}

/** Hz ticks for a mel axis, spread evenly in mels rather than in Hz. */
export function melTicks(lo: number, hi: number, formula: MelFormula, count = 5): number[] {
  const candidates = [0, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 16000, 20000];
  const inRange = candidates.filter((hz) => hz >= lo && hz <= hi);
  const span = hzToMel(hi, formula) - hzToMel(lo, formula);
  const out: number[] = [];
  let last = -Infinity;
  for (const hz of inRange) {
    const mel = hzToMel(hz, formula);
    if (mel - last >= span / (count + 1)) {
      out.push(hz);
      last = mel;
    }
  }
  return out;
}

export { hzToMel, melToHz };

/** A tick label with no more digits than the spacing needs. */
export function formatTick(value: number, step?: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 10000) return `${+(value / 1000).toFixed(abs >= 100000 ? 0 : 1)}k`;
  // A log axis has no single step, so each tick keeps its own leading digits.
  if (!step) return String(+value.toPrecision(3));
  const digits = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
  return value.toFixed(digits);
}

/** A readout value with four significant digits. */
export function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return value.toExponential(2);
  return String(+value.toPrecision(4));
}

/** An axis title with its unit, such as "Frequency (Hz)". */
export function axisTitle(label: string | null | undefined, unit: string | null | undefined): string {
  if (label && unit) return `${label} (${unit})`;
  return label ?? unit ?? "";
}

// Theme -----------------------------------------------------------------

export type ThemeColours = {
  canvas: string;
  raise: string;
  ink: string;
  ink2: string;
  ink3: string;
  ink4: string;
  rule: string;
  ruleStrong: string;
  series: string[];
};

/** The theme's colours, read from the CSS variables on `<html>`. */
export function themeColours(): ThemeColours {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    canvas: read("--canvas", "#fcfbf8"),
    raise: read("--raise", "#ffffff"),
    ink: read("--ink", "#1c1b18"),
    ink2: read("--ink-2", "#37352f"),
    ink3: read("--ink-3", "#6b675d"),
    ink4: read("--ink-4", "#9a9589"),
    rule: read("--rule", "#e6e3da"),
    ruleStrong: read("--rule-strong", "#d4d0c4"),
    series: [1, 2, 3, 4, 5, 6].map((i) => read(`--series-${i}`, "#2b59c3")),
  };
}

/** Changes whenever `<html data-theme>` does, so a canvas knows to repaint. */
export function useThemeVersion(): number {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const observer = new MutationObserver(() => setVersion((v) => v + 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return version;
}

/** The content width of `ref`'s element, kept current as it resizes. */
export function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    setWidth(Math.floor(node.clientWidth));
    const observer = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0]?.contentRect.width ?? node.clientWidth);
      setWidth((current) => (current === next ? current : next));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

/**
 * Sizes a canvas for its CSS box and returns a context drawing in CSS pixels.
 * It renders at least at twice the CSS resolution, so a figure stays sharp in
 * the PDF book, where headless Chrome prints at one device pixel per CSS pixel.
 */
export function prepareCanvas(canvas: HTMLCanvasElement, width: number, height: number): CanvasRenderingContext2D | null {
  const ratio = Math.max(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  return context;
}

export const TICK_FONT = `10px "JetBrains Mono Variable", ui-monospace, Menlo, monospace`;
export const LABEL_FONT = `11px "Inter Variable", ui-sans-serif, system-ui, sans-serif`;

/** Decodes base64 into bytes. */
export function decodeBase64(text: string): Uint8Array {
  const binary = atob(text);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}
