import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";

import { colormap, type ColormapName } from "../../lib/colormap";
import {
  LABEL_FONT,
  TICK_FONT,
  axisTitle,
  formatTick,
  formatValue,
  hzToMel,
  melTicks,
  melToHz,
  niceTicks,
  prepareCanvas,
  themeColours,
  useThemeVersion,
  useWidth,
} from "../../lib/plotting";
import type { MelFormula } from "../../demos/dsp";

export type HeatmapAxis = {
  label?: string | null;
  unit?: string | null;
  /** The value at the low edge of the first row or column. */
  min: number;
  /** The value at the high edge of the last row or column. */
  max: number;
  /** A mel axis spaces its rows evenly in mels; its ticks are still in Hz. */
  scale?: "linear" | "mel";
  mel?: MelFormula;
  /** One name per row or column, such as an attention map's tokens. */
  ticks?: string[] | null;
};

export type HeatmapProps = {
  /** `rows * cols` levels from 0 to 255, row by row. */
  levels: Uint8Array;
  rows: number;
  cols: number;
  /** `lower` draws row 0 at the bottom, as a spectrogram does. */
  origin?: "lower" | "upper";
  colormap?: ColormapName;
  x: HeatmapAxis;
  y: HeatmapAxis;
  /** What levels 0 and 255 stand for, for the colour bar and the tooltip. */
  value?: { label?: string | null; min: number; max: number };
  /** The CSS height. Defaults to a height that follows the width. */
  height?: number;
  colorbar?: boolean;
  /** A sentence for screen readers. */
  description?: string;
};

type Layout = { left: number; right: number; top: number; bottom: number; width: number; height: number };

/**
 * A heatmap on a canvas, with axes, an optional colour bar, and a tooltip.
 *
 * The run console draws `speechlab` spectrograms and attention maps with it,
 * and the demos draw their spectrograms with it, so both look the same. The
 * canvas fills its column and never overflows it.
 */
export default function HeatmapCanvas({
  levels,
  rows,
  cols,
  origin = "lower",
  colormap: cmap = "magma",
  x,
  y,
  value,
  height,
  colorbar = true,
  description,
}: HeatmapProps) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const width = useWidth(box);
  const theme = useThemeVersion();
  const [hover, setHover] = useState<{ left: number; top: number; text: string } | null>(null);

  const categoricalY = !!y.ticks && y.ticks.length === rows;
  const categoricalX = !!x.ticks && x.ticks.length === cols;
  const cssHeight = height ?? Math.round(Math.min(320, Math.max(150, width * 0.42)));

  const layout = useMemo<Layout>(() => {
    const longestY = categoricalY ? Math.max(...y.ticks!.map((t) => t.length)) : 5;
    const left = Math.min(110, Math.max(40, 8 + longestY * 6.2)) + (y.label || y.unit ? 14 : 0);
    // Room for the colour bar and its longest label, at about 6 px a character.
    const barLabel = value
      ? Math.max(formatValue(value.min).length, formatValue(value.max).length, (value.label ?? "").slice(0, 6).length)
      : 0;
    const right = colorbar && value ? 22 + Math.ceil(barLabel * 6.2) : 10;
    const xRotated = categoricalX && cols > 6;
    const longestX = categoricalX ? Math.max(...x.ticks!.map((t) => t.length)) : 0;
    const bottom = (xRotated ? Math.min(70, 14 + longestX * 4.6) : 20) + (x.label || x.unit ? 16 : 0);
    return { left, right, top: 8, bottom, width, height: cssHeight };
  }, [width, cssHeight, categoricalX, categoricalY, colorbar, cols, x, y, value]);

  // The image itself, one pixel per cell, rebuilt only when the data changes.
  const image = useMemo(() => {
    if (rows <= 0 || cols <= 0 || levels.length < rows * cols) return null;
    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    const context = offscreen.getContext("2d");
    if (!context) return null;
    const pixels = context.createImageData(cols, rows);
    const lut = colormap(cmap);
    for (let r = 0; r < rows; r += 1) {
      const source = (origin === "lower" ? rows - 1 - r : r) * cols;
      for (let c = 0; c < cols; c += 1) {
        const level = levels[source + c] * 3;
        const at = (r * cols + c) * 4;
        pixels.data[at] = lut[level];
        pixels.data[at + 1] = lut[level + 1];
        pixels.data[at + 2] = lut[level + 2];
        pixels.data[at + 3] = 255;
      }
    }
    context.putImageData(pixels, 0, 0);
    return offscreen;
  }, [levels, rows, cols, origin, cmap]);

  useEffect(() => {
    const node = canvas.current;
    if (!node || !image || width <= 0) return;
    const ctx = prepareCanvas(node, width, cssHeight);
    if (!ctx) return;
    const colours = themeColours(node);
    const { left, right, top, bottom } = layout;
    const plotW = Math.max(10, width - left - right);
    const plotH = Math.max(10, cssHeight - top - bottom);

    // Smoothing helps when there are more cells than pixels; when each cell is
    // several pixels wide, hard edges are the honest picture.
    ctx.imageSmoothingEnabled = cols > plotW || rows > plotH;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, left, top, plotW, plotH);
    ctx.strokeStyle = colours.ruleStrong;
    ctx.lineWidth = 1;
    ctx.strokeRect(left - 0.5, top - 0.5, plotW + 1, plotH + 1);

    ctx.font = TICK_FONT;
    ctx.fillStyle = colours.ink3;
    ctx.strokeStyle = colours.ruleStrong;

    // The x axis.
    if (categoricalX) {
      const every = Math.max(1, Math.ceil(cols / Math.max(1, plotW / 14)));
      const rotated = cols > 6;
      for (let c = 0; c < cols; c += every) {
        const px = left + ((c + 0.5) / cols) * plotW;
        ctx.save();
        ctx.translate(px, top + plotH + 6);
        if (rotated) {
          ctx.rotate(-Math.PI / 4);
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
        } else {
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
        }
        ctx.fillText(x.ticks![c], 0, 0);
        ctx.restore();
      }
    } else {
      const ticks = niceTicks(x.min, x.max, Math.max(3, Math.floor(plotW / 55)));
      const step = ticks.length > 1 ? ticks[1] - ticks[0] : undefined;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (const t of ticks) {
        const px = left + ((t - x.min) / (x.max - x.min || 1)) * plotW;
        ctx.beginPath();
        ctx.moveTo(px, top + plotH);
        ctx.lineTo(px, top + plotH + 3);
        ctx.stroke();
        ctx.fillText(formatTick(t, step), px, top + plotH + 5);
      }
    }

    // The y axis.
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    if (categoricalY) {
      const every = Math.max(1, Math.ceil(rows / Math.max(1, plotH / 13)));
      for (let r = 0; r < rows; r += every) {
        const fromTop = origin === "lower" ? rows - 1 - r : r;
        const py = top + ((fromTop + 0.5) / rows) * plotH;
        ctx.fillText(y.ticks![r], left - 5, py);
      }
    } else {
      const mel = y.scale === "mel";
      const formula = y.mel ?? "slaney";
      const ticks = mel
        ? melTicks(y.min, y.max, formula, Math.max(3, Math.floor(plotH / 28)))
        : niceTicks(y.min, y.max, Math.max(3, Math.floor(plotH / 28)));
      const step = !mel && ticks.length > 1 ? ticks[1] - ticks[0] : undefined;
      const lo = mel ? hzToMel(y.min, formula) : y.min;
      const hi = mel ? hzToMel(y.max, formula) : y.max;
      for (const t of ticks) {
        const v = mel ? hzToMel(t, formula) : t;
        let fraction = (v - lo) / (hi - lo || 1);
        if (origin === "upper") fraction = 1 - fraction;
        const py = top + plotH - fraction * plotH;
        ctx.beginPath();
        ctx.moveTo(left - 3, py);
        ctx.lineTo(left, py);
        ctx.stroke();
        ctx.fillText(formatTick(t, step), left - 5, py);
      }
    }

    // Axis titles.
    ctx.font = LABEL_FONT;
    ctx.fillStyle = colours.ink3;
    const xTitle = axisTitle(x.label, x.unit);
    if (xTitle) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(xTitle, left + plotW / 2, cssHeight - 1);
    }
    const yTitle = axisTitle(y.label, y.unit);
    if (yTitle) {
      ctx.save();
      ctx.translate(10, top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(yTitle, 0, 0);
      ctx.restore();
    }

    // The colour bar.
    if (colorbar && value) {
      const barX = left + plotW + 8;
      const barW = 8;
      const lut = colormap(cmap);
      for (let i = 0; i < plotH; i += 1) {
        const level = Math.round((1 - i / Math.max(1, plotH - 1)) * 255) * 3;
        ctx.fillStyle = `rgb(${lut[level]} ${lut[level + 1]} ${lut[level + 2]})`;
        ctx.fillRect(barX, top + i, barW, 1);
      }
      ctx.strokeStyle = colours.ruleStrong;
      ctx.strokeRect(barX - 0.5, top - 0.5, barW + 1, plotH + 1);
      ctx.font = TICK_FONT;
      ctx.fillStyle = colours.ink3;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(formatValue(value.max), barX + barW + 3, top);
      ctx.textBaseline = "bottom";
      ctx.fillText(formatValue(value.min), barX + barW + 3, top + plotH);
      if (value.label) {
        ctx.textBaseline = "middle";
        ctx.fillText(value.label.slice(0, 6), barX + barW + 3, top + plotH / 2);
      }
    }
  }, [image, width, cssHeight, layout, theme, x, y, value, colorbar, categoricalX, categoricalY, cols, rows, origin, cmap]);

  function onPointer(event: PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const { left, right, top, bottom } = layout;
    const plotW = width - left - right;
    const plotH = cssHeight - top - bottom;
    const fx = (px - left) / plotW;
    const fy = (py - top) / plotH;
    if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) {
      setHover(null);
      return;
    }
    const col = Math.min(cols - 1, Math.floor(fx * cols));
    const fromTop = Math.min(rows - 1, Math.floor(fy * rows));
    const row = origin === "lower" ? rows - 1 - fromTop : fromTop;
    const parts: string[] = [];

    if (categoricalX) parts.push(x.ticks![col]);
    else parts.push(`${formatValue(x.min + ((col + 0.5) / cols) * (x.max - x.min))}${x.unit ? ` ${x.unit}` : ""}`);

    if (categoricalY) {
      parts.push(y.ticks![row]);
    } else {
      const f = (row + 0.5) / rows;
      let v: number;
      if (y.scale === "mel") {
        const formula = y.mel ?? "slaney";
        const lo = hzToMel(y.min, formula);
        const hi = hzToMel(y.max, formula);
        v = melToHz(lo + f * (hi - lo), formula);
      } else {
        v = y.min + f * (y.max - y.min);
      }
      parts.push(`${formatValue(v)}${y.unit ? ` ${y.unit}` : ""}`);
    }

    if (value) {
      const level = levels[row * cols + col];
      const v = value.min + (level / 255) * (value.max - value.min);
      parts.push(`${formatValue(v)}${value.label ? ` ${value.label}` : ""}`);
    }
    setHover({ left: px, top: py, text: parts.join(" · ") });
  }

  return (
    <div ref={box} className="relative w-full min-w-0" style={{ height: cssHeight }}>
      <canvas
        ref={canvas}
        role="img"
        aria-label={description ?? "A heatmap"}
        className="block max-w-full touch-pan-y"
        onPointerMove={onPointer}
        onPointerDown={onPointer}
        onPointerLeave={() => setHover(null)}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 whitespace-nowrap rounded-md bg-raise px-2 py-1 font-mono text-[11px] text-ink shadow-[var(--shadow)] ring-1 ring-rule"
          style={{
            left: Math.min(Math.max(0, hover.left + 10), Math.max(0, width - 190)),
            top: Math.max(0, hover.top - 30),
          }}
        >
          {hover.text}
        </div>
      )}
    </div>
  );
}
