import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";

import {
  LABEL_FONT,
  TICK_FONT,
  formatTick,
  formatValue,
  logTicks,
  niceTicks,
  prepareCanvas,
  themeColours,
  useThemeVersion,
  useWidth,
} from "../../lib/plotting";

export type LineSeries = {
  label?: string | null;
  x: ArrayLike<number | null>;
  y: ArrayLike<number | null>;
};

export type LineProps = {
  series: LineSeries[];
  xlabel?: string | null;
  ylabel?: string | null;
  xScale?: "linear" | "log";
  yScale?: "linear" | "log";
  xlim?: [number | null, number | null] | null;
  ylim?: [number | null, number | null] | null;
  height?: number;
  description?: string;
};

type Domain = [number, number];

function usable(v: number | null | undefined, log: boolean): v is number {
  return v != null && Number.isFinite(v) && (!log || v > 0);
}

function extent(values: ArrayLike<number | null>[], log: boolean): Domain | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const list of values) {
    for (let i = 0; i < list.length; i += 1) {
      const v = list[i];
      if (!usable(v, log)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

/**
 * Line plots on a canvas: up to six series, linear or log axes, a legend, and
 * a crosshair that reads every series at the pointer. The run console draws
 * `speechlab.show_plot` with it.
 */
export default function LineCanvas({
  series,
  xlabel,
  ylabel,
  xScale = "linear",
  yScale = "linear",
  xlim,
  ylim,
  height,
  description,
}: LineProps) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const width = useWidth(box);
  const theme = useThemeVersion();
  const [hover, setHover] = useState<{ px: number; lines: string[] } | null>(null);
  const cssHeight = height ?? Math.round(Math.min(280, Math.max(150, width * 0.38)));
  const logX = xScale === "log";
  const logY = yScale === "log";

  const domains = useMemo(() => {
    let xd = extent(series.map((s) => s.x), logX) ?? [0, 1];
    let yd = extent(series.map((s) => s.y), logY) ?? [0, 1];
    if (xlim) xd = [xlim[0] ?? xd[0], xlim[1] ?? xd[1]];
    if (xd[0] === xd[1]) xd = logX ? [xd[0] / 2, xd[0] * 2] : [xd[0] - 0.5, xd[1] + 0.5];
    if (ylim) {
      yd = [ylim[0] ?? yd[0], ylim[1] ?? yd[1]];
    } else if (logY) {
      yd = [yd[0] / 1.2, yd[1] * 1.2];
    } else {
      const pad = (yd[1] - yd[0]) * 0.06 || 0.5;
      yd = [yd[0] - pad, yd[1] + pad];
    }
    return { x: xd as Domain, y: yd as Domain };
  }, [series, xlim, ylim, logX, logY]);

  const layout = { left: 50 + (ylabel ? 12 : 0), right: 12, top: 10, bottom: 22 + (xlabel ? 16 : 0) };
  const plotW = Math.max(10, width - layout.left - layout.right);
  const plotH = Math.max(10, cssHeight - layout.top - layout.bottom);

  const toPx = (v: number, [lo, hi]: Domain, log: boolean, size: number) => {
    const f = log ? Math.log10(v / lo) / Math.log10(hi / lo) : (v - lo) / (hi - lo || 1);
    return f * size;
  };
  const sx = (v: number) => layout.left + toPx(v, domains.x, logX, plotW);
  const sy = (v: number) => layout.top + plotH - toPx(v, domains.y, logY, plotH);

  useEffect(() => {
    const node = canvas.current;
    if (!node || width <= 0) return;
    const ctx = prepareCanvas(node, width, cssHeight);
    if (!ctx) return;
    const colours = themeColours(node);

    const xTicks = logX ? logTicks(...domains.x) : niceTicks(...domains.x, Math.max(3, Math.floor(plotW / 55)));
    const yTicks = logY ? logTicks(...domains.y) : niceTicks(...domains.y, Math.max(3, Math.floor(plotH / 28)));
    const xStep = !logX && xTicks.length > 1 ? xTicks[1] - xTicks[0] : undefined;
    const yStep = !logY && yTicks.length > 1 ? yTicks[1] - yTicks[0] : undefined;

    ctx.font = TICK_FONT;
    ctx.lineWidth = 1;
    ctx.strokeStyle = colours.rule;
    ctx.fillStyle = colours.ink3;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of yTicks) {
      const py = Math.round(sy(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(layout.left, py);
      ctx.lineTo(layout.left + plotW, py);
      ctx.stroke();
      ctx.fillText(formatTick(t, yStep), layout.left - 6, py);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of xTicks) {
      ctx.fillText(formatTick(t, xStep), sx(t), layout.top + plotH + 5);
    }
    ctx.strokeStyle = colours.ruleStrong;
    ctx.beginPath();
    ctx.moveTo(layout.left, layout.top + plotH + 0.5);
    ctx.lineTo(layout.left + plotW, layout.top + plotH + 0.5);
    ctx.stroke();

    ctx.save();
    ctx.beginPath();
    ctx.rect(layout.left, layout.top, plotW, plotH);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    series.forEach((s, index) => {
      ctx.strokeStyle = colours.series[index % colours.series.length];
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < s.y.length; i += 1) {
        const xv = s.x[i];
        const yv = s.y[i];
        if (!usable(xv, logX) || !usable(yv, logY)) {
          pen = false;
          continue;
        }
        if (pen) ctx.lineTo(sx(xv), sy(yv));
        else ctx.moveTo(sx(xv), sy(yv));
        pen = true;
      }
      ctx.stroke();
    });
    ctx.restore();

    ctx.font = LABEL_FONT;
    ctx.fillStyle = colours.ink3;
    if (xlabel) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(xlabel, layout.left + plotW / 2, cssHeight - 1);
    }
    if (ylabel) {
      ctx.save();
      ctx.translate(10, layout.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(ylabel, 0, 0);
      ctx.restore();
    }

    if (hover) {
      ctx.strokeStyle = colours.ink4;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hover.px + 0.5, layout.top);
      ctx.lineTo(hover.px + 0.5, layout.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // `sx` and `sy` are rebuilt from the values listed here on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, width, cssHeight, theme, domains, hover, xlabel, ylabel, logX, logY]);

  function onPointer(event: PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    if (px < layout.left || px > layout.left + plotW) {
      setHover(null);
      return;
    }
    const f = (px - layout.left) / plotW;
    const [lo, hi] = domains.x;
    const xv = logX ? lo * (hi / lo) ** f : lo + f * (hi - lo);
    const lines = [`x = ${formatValue(xv)}`];
    series.forEach((s, index) => {
      // The nearest point in x. Series are short, so a scan is fast enough.
      let best = -1;
      let gap = Infinity;
      for (let i = 0; i < s.x.length; i += 1) {
        const v = s.x[i];
        if (v == null) continue;
        const d = Math.abs(v - xv);
        if (d < gap) {
          gap = d;
          best = i;
        }
      }
      const yv = best >= 0 ? s.y[best] : null;
      const name = s.label ?? (series.length > 1 ? `series ${index + 1}` : "y");
      lines.push(`${name} = ${yv == null ? "—" : formatValue(yv)}`);
    });
    setHover({ px, lines });
  }

  const named = series.some((s) => s.label);

  return (
    <div className="w-full min-w-0">
      <div ref={box} className="relative w-full" style={{ height: cssHeight }}>
        <canvas
          ref={canvas}
          role="img"
          aria-label={description ?? "A line plot"}
          className="block max-w-full touch-pan-y"
          onPointerMove={onPointer}
          onPointerDown={onPointer}
          onPointerLeave={() => setHover(null)}
        />
        {hover && (
          <div
            className="pointer-events-none absolute z-10 rounded-md bg-raise px-2 py-1 font-mono text-[11px] leading-snug text-ink shadow-[var(--shadow)] ring-1 ring-rule"
            style={{ left: Math.min(hover.px + 10, Math.max(0, width - 170)), top: layout.top }}
          >
            {hover.lines.map((line, index) => (
              <div key={index} className="whitespace-nowrap">
                {line}
              </div>
            ))}
          </div>
        )}
      </div>
      {named && (
        <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 font-sans text-[12px] text-ink-3">
          {series.map((s, index) => (
            <span key={index} className="flex items-center gap-1.5">
              <span
                className="inline-block h-0 w-4 border-t-2"
                style={{ borderColor: `var(--series-${(index % 6) + 1})` }}
              />
              {s.label ?? `series ${index + 1}`}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
