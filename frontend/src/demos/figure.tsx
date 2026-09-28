/**
 * Pieces for demos that draw their own figure rather than a `Plot`: a canvas
 * that fills its column and repaints with the theme, a hook for the width of
 * a column, and colour lookups that read the theme's tokens.
 *
 * `Plot` suits a curve in data coordinates. A lattice, a trellis, a Gantt
 * chart, or a heatmap with marks on top needs its own layout in pixels, and
 * these keep that layout inside the column and in the theme's colours.
 */

import { useEffect, useId, useRef, useState, type PointerEvent, type RefObject } from "react";

import { colormap, type ColormapName } from "../lib/colormap";
import { prepareCanvas, useThemeVersion, useWidth } from "../lib/plotting";


export { TICK_FONT, LABEL_FONT } from "../lib/plotting";

/** A theme token's current value, such as `token("--demo")`. */
export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** The width of `ref`'s element in CSS pixels, and the ref to attach. */
export function useColumnWidth<T extends HTMLElement = HTMLDivElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const width = useWidth(ref);
  return [ref, width];
}

/** A colour map as a function from a value in [0, 1] to a CSS colour. */
export function colourScale(name: ColormapName = "magma"): (value: number) => string {
  const lut = colormap(name);
  return (value) => {
    const level = Math.max(0, Math.min(255, Math.round(value * 255))) * 3;
    return `rgb(${lut[level]} ${lut[level + 1]} ${lut[level + 2]})`;
  };
}

export type Paint = {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  /** A theme token's value, read once per paint. */
  colour: (name: string) => string;
};

/**
 * A canvas that fills its column. `draw` runs on every change of `deps`, of
 * the width, and of the theme; `height` may depend on the width. Pointer
 * events report CSS pixels inside the canvas.
 */
export function CanvasFigure({
  height,
  draw,
  deps,
  onPointer,
  description,
}: {
  height: number | ((width: number) => number);
  draw: (paint: Paint) => void;
  deps: readonly unknown[];
  onPointer?: (x: number, y: number, width: number, type: "down" | "move" | "leave") => void;
  description: string;
}) {
  const [box, width] = useColumnWidth();
  const canvas = useRef<HTMLCanvasElement>(null);
  const theme = useThemeVersion();
  const cssHeight = typeof height === "function" ? height(width) : height;

  useEffect(() => {
    const node = canvas.current;
    if (!node || width <= 0) return;
    const ctx = prepareCanvas(node, width, cssHeight);
    if (!ctx) return;
    const cache = new Map<string, string>();
    const colour = (name: string) => {
      let value = cache.get(name);
      if (value === undefined) {
        value = token(name);
        cache.set(name, value);
      }
      return value;
    };
    draw({ ctx, width, height: cssHeight, colour });
    // `draw` is a fresh closure each render; `deps` says when it changed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, cssHeight, theme, ...deps]);

  function pointer(event: PointerEvent<HTMLCanvasElement>, type: "down" | "move") {
    if (!onPointer) return;
    const rect = event.currentTarget.getBoundingClientRect();
    onPointer(event.clientX - rect.left, event.clientY - rect.top, width, type);
  }

  return (
    <div ref={box} className="relative w-full min-w-0" style={{ height: cssHeight }}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={description}
        className={`block max-w-full ${onPointer ? "cursor-pointer touch-pan-y" : ""}`}
        onPointerDown={onPointer ? (event) => pointer(event, "down") : undefined}
        onPointerMove={onPointer ? (event) => pointer(event, "move") : undefined}
        onPointerLeave={onPointer ? () => onPointer(-1, -1, width, "leave") : undefined}
      />
    </div>
  );
}

/**
 * The value after `delay` ms without a change, for recomputation too heavy
 * to run on every step of a slider. The first value is immediate, so the
 * printed figure never waits.
 */
export function useDebounced<T>(value: T, delay = 120): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return settled;
}

/** A short text field in the style of the kit's controls. */
export function TextField({
  label,
  value,
  onChange,
  maxLength,
  hint,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  maxLength?: number;
  hint?: string;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="flex items-baseline justify-between gap-2 text-[12.5px] text-ink-2">
        <span>{label}</span>
        {hint && <span className="text-[11.5px] text-ink-3">{hint}</span>}
      </label>
      <input
        id={id}
        type="text"
        value={value}
        maxLength={maxLength}
        spellCheck={false}
        autoCapitalize="off"
        autoComplete="off"
        onChange={(event) => onChange(event.target.value)}
        className="mt-1.5 min-h-8 w-full rounded-md bg-raise px-2.5 font-mono text-[13px] text-ink shadow-[inset_0_0_0_1px_var(--rule-strong)] outline-none focus:shadow-[inset_0_0_0_1px_var(--demo)]"
      />
    </div>
  );
}

/**
 * Values in [0, 1] as an image tinted with `colour`, transparent where the
 * value is 0, as a data URL for an SVG `<image>` inside a `Plot`. Row 0 is
 * the top of the image. It reads the colour as the canvas resolves it, so
 * any theme token's value works.
 */
export function tintImage(values: ArrayLike<number>, rows: number, cols: number, colour: string, strength = 0.9): string {
  if (typeof document === "undefined" || rows <= 0 || cols <= 0) return "";
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = colour;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  const image = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i += 1) {
    const v = Math.max(0, Math.min(1, values[i]));
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = Math.round(255 * strength * v);
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL();
}

export { useThemeVersion };

/** Relative luminance of a CSS colour as the canvas resolves it, 0 to 1. */
function luminance(ctx: CanvasRenderingContext2D, colour: string): number {
  ctx.save();
  ctx.fillStyle = colour;
  const resolved = String(ctx.fillStyle);
  ctx.restore();
  const hex = /^#([0-9a-f]{6})$/i.exec(resolved);
  const rgb = hex
    ? [0, 2, 4].map((i) => parseInt(hex[1].slice(i, i + 2), 16))
    : (resolved.match(/[\d.]+/g) ?? ["0", "0", "0"]).slice(0, 3).map(Number);
  const [r, g, b] = rgb.map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * Whichever of the theme's ink and canvas colours reads better on `background`.
 * A heatmap cell's number needs this, since the colour map is the same in
 * both themes and the text tokens are not.
 */
export function textOn(ctx: CanvasRenderingContext2D, background: string, colour: (name: string) => string): string {
  const ink = colour("--ink");
  const canvas = colour("--canvas");
  const lb = luminance(ctx, background);
  const contrast = (c: string) => {
    const l = luminance(ctx, c);
    return (Math.max(l, lb) + 0.05) / (Math.min(l, lb) + 0.05);
  };
  return contrast(ink) >= contrast(canvas) ? ink : canvas;
}

/**
 * A posteriorgram, one row per symbol and one column per frame, in the magma
 * colour map with each cell's probability written in when it fits.
 * `columns` places frame i at `[x0 + i * cellW, x0 + (i + 1) * cellW]`, so the
 * frames can line up with another figure's columns.
 */
export function Posteriorgram({
  probs,
  vocab,
  columns,
  description,
}: {
  /** `probs[frame][symbol]`. */
  probs: number[][];
  vocab: string[];
  columns: (width: number) => { x0: number; cellW: number };
  description: string;
}) {
  const heat = colourScale("magma");
  const rowH = 20;
  return (
    <CanvasFigure
      height={vocab.length * rowH + 16}
      deps={[probs, vocab]}
      description={description}
      draw={({ ctx, width, colour }) => {
        const { x0, cellW } = columns(width);
        ctx.font = `10px "JetBrains Mono Variable", ui-monospace, monospace`;
        ctx.textBaseline = "middle";
        probs.forEach((row, t) => {
          ctx.textAlign = "center";
          ctx.fillStyle = colour("--ink-3");
          if (cellW >= 18 || t % 2 === 0) ctx.fillText(String(t + 1), x0 + (t + 0.5) * cellW, 7);
          row.forEach((p, v) => {
            const fill = heat(p);
            const left = x0 + t * cellW;
            const top = 14 + v * rowH;
            ctx.fillStyle = fill;
            ctx.fillRect(left + 0.5, top + 0.5, cellW - 1, rowH - 1);
            if (cellW >= 30) {
              ctx.fillStyle = textOn(ctx, fill, colour);
              ctx.font = `9.5px "JetBrains Mono Variable", ui-monospace, monospace`;
              ctx.fillText(p.toFixed(2).replace(/^0/, ""), left + cellW / 2, top + rowH / 2 + 0.5);
              ctx.font = `10px "JetBrains Mono Variable", ui-monospace, monospace`;
            }
          });
        });
        ctx.textAlign = "right";
        vocab.forEach((symbol, v) => {
          ctx.fillStyle = colour("--ink");
          ctx.fillText(symbol, x0 - 8, 14 + (v + 0.5) * rowH);
        });
      }}
    />
  );
}
