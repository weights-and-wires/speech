/**
 * The controls, the plotting frame, and the audio pieces every demo is built from.
 *
 * A demo in a lesson is a figure a reader can push on, not an app. It gets a
 * frame, a plot, and a handful of controls, all in the colour `--demo`.
 * Keeping all of that here is what makes the demos look like one set.
 *
 * `docs/DEMOS.md` is the guide to writing one. The pieces:
 *
 * - The frame: `DemoFrame`, `PrintedFigure`, `useWhenPrinted`.
 * - Controls: `Slider`, `Choice`, `Toggle`, `Action`, `Actions`, `Readout`, `Readouts`.
 * - SVG plots in data coordinates: `Plot`, `Curve`, `Bars`, `Rule`, `Points`,
 *   `Stems`, `Waveform`, `Legend`, `useScales`.
 * - Canvas figures: `Spectrogram`, and the `HeatmapCanvas` and `LineCanvas`
 *   the run console uses, with `quantize` to turn values into levels.
 * - Sound: `useAudioPlayer` and `useMicrophone`, from `./audio`.
 * - Signal processing: `./dsp`.
 */

import {
  createContext,
  useContext,
  useEffect,
  useId,
  useMemo,
  type ReactNode,
} from "react";

import HeatmapCanvas from "../components/figures/HeatmapCanvas";
import LineCanvas from "../components/figures/LineCanvas";
import type { ColormapName } from "../lib/colormap";
import type { Spectrogram as SpectrogramData } from "./dsp";

export { HeatmapCanvas, LineCanvas };
export { useAudioPlayer, useMicrophone } from "./audio";

// The frame --------------------------------------------------------------

/**
 * The card a demo sits in.
 *
 * `caption` is the sentence from the Markdown block that says what to notice.
 * It is rendered under the title rather than under the figure, because a reader
 * needs it before they start pushing buttons.
 */
/**
 * Set by the PDF book, where a demo is a still figure. `url` is the chapter
 * online, where the reader can push on it.
 */
export const PrintedFigure = createContext<{ url: string } | null>(null);

/**
 * Runs a demo's simulation once when it is printed. A demo that starts empty
 * and waits for a button would otherwise print as an empty plot.
 */
export function useWhenPrinted(run: () => void) {
  const printed = useContext(PrintedFigure);
  useEffect(() => {
    if (printed) run();
    // Once, on mount: a printed figure never changes afterwards.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

export function DemoFrame({
  title,
  caption,
  controls,
  children,
}: {
  title: string;
  caption?: string;
  controls?: ReactNode;
  children: ReactNode;
}) {
  const printed = useContext(PrintedFigure);
  return (
    <figure className="demo my-8 overflow-hidden rounded-[10px] bg-raise shadow-[inset_0_0_0_1px_var(--rule)]">
      <figcaption className="border-b border-rule px-5 pb-3.5 pt-4">
        <span className="demo-kicker flex items-center gap-2 text-[12px] font-semibold text-demo">
          <span className="size-1.5 rounded-full bg-demo" aria-hidden />
          {printed ? "Figure" : "Try it"}
        </span>
        <span className="mt-1 block font-serif text-[1.12rem] font-semibold leading-snug text-ink">
          {title}
        </span>
        {caption && (
          <span className="mt-1 block text-[13.5px] leading-snug text-ink-3">{caption}</span>
        )}
      </figcaption>

      <div className="px-5 py-4">{children}</div>

      {controls && !printed && (
        <div className="grid gap-x-6 gap-y-4 border-t border-rule bg-chrome px-5 py-4 sm:grid-cols-2">
          {controls}
        </div>
      )}
      {printed && (
        <p className="demo-online border-t border-rule bg-chrome px-5 py-2.5 text-[12px] text-ink-3">
          The interactive version is at{" "}
          <a href={`https://${printed.url}`} className="text-accent">
            {printed.url}
          </a>
          .
        </p>
      )}
    </figure>
  );
}

// Controls ---------------------------------------------------------------

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  format,
  onChange,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  /** How the value reads beside the label. Defaults to the number itself. */
  format?: (value: number) => string;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div>
      <label
        htmlFor={id}
        className="flex items-baseline justify-between gap-2 text-[12.5px] text-ink-2"
      >
        <span>{label}</span>
        <span className="font-mono text-[12px] tabular-nums text-ink">
          {format ? format(value) : value}
        </span>
      </label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full cursor-pointer accent-[var(--demo)]"
      />
    </div>
  );
}

export function Choice<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: ReactNode;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div>
      <span className="text-[12.5px] text-ink-2">{label}</span>
      <div
        role="radiogroup"
        aria-label={typeof label === "string" ? label : undefined}
        className="mt-1.5 flex flex-wrap gap-1"
      >
        {options.map((option) => {
          const active = option.value === value;
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(option.value)}
              className={`min-h-8 rounded-md px-2.5 text-[12.5px] transition-colors ${
                active
                  ? "bg-raise font-medium text-ink shadow-[inset_0_0_0_1px_var(--ink-4)]"
                  : "text-ink-3 shadow-[inset_0_0_0_1px_var(--rule)] hover:text-ink hover:shadow-[inset_0_0_0_1px_var(--rule-strong)]"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <label
      htmlFor={id}
      className="flex cursor-pointer select-none items-center gap-2 text-[12.5px] text-ink-2"
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="size-3.5 cursor-pointer accent-[var(--demo)]"
      />
      {label}
    </label>
  );
}

export function Action({
  children,
  onClick,
  tone = "primary",
}: {
  children: ReactNode;
  onClick: () => void;
  tone?: "primary" | "quiet";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`btn ${tone === "primary" ? "btn-primary" : "btn-secondary"} min-h-8! text-[12.5px]`}
    >
      {children}
    </button>
  );
}

/** A row of buttons, which is how most demos drive their simulation. */
export function Actions({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}

/** One labelled number, for whatever the demo wants the reader to watch. */
export function Readout({
  label,
  value,
  hint,
  tone = "plain",
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "plain" | "demo" | "good" | "bad";
}) {
  const colour =
    tone === "demo"
      ? "text-demo"
      : tone === "good"
        ? "text-good"
        : tone === "bad"
          ? "text-bad"
          : "text-ink";
  // A readout is built for a number, and a demo occasionally puts a short
  // sentence in one. Setting a sentence in 17px tabular mono makes it a wall,
  // so anything past a few characters drops to body type and wraps.
  const wordy = typeof value === "string" && value.length > 12;
  const valueClass = wordy
    ? `mt-0.5 text-[12.5px] leading-snug ${colour}`
    : `mt-0.5 font-mono text-[17px] tabular-nums ${colour}`;
  return (
    <div className="rounded-lg bg-chrome px-3 py-2 shadow-[inset_0_0_0_1px_var(--rule)]">
      <div className="text-[11.5px] font-medium text-ink-3">
        {label}
      </div>
      <div className={valueClass}>{value}</div>
      {hint && <div className="mt-0.5 text-[11px] leading-snug text-ink-3">{hint}</div>}
    </div>
  );
}

/** A responsive strip of readouts. */
export function Readouts({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">{children}</div>
  );
}

// The plot ---------------------------------------------------------------

export type Scales = {
  /** Data x to pixels. */
  sx: (x: number) => number;
  /** Data y to pixels. */
  sy: (y: number) => number;
  /** Pixels back to data x, for pointer handling. */
  ix: (px: number) => number;
  /** Pixels back to data y, for pointer handling. */
  iy: (py: number) => number;
  xDomain: [number, number];
  yDomain: [number, number];
  /** The drawing area in pixels, inside the margins. */
  area: { left: number; right: number; top: number; bottom: number };
};

const ScaleContext = createContext<Scales | null>(null);

/** The scales of the enclosing `Plot`, for a child that needs them. */
export function useScales(): Scales {
  const scales = useContext(ScaleContext);
  if (!scales) throw new Error("useScales must be used inside a Plot");
  return scales;
}

const MARGIN = { left: 44, right: 14, top: 12, bottom: 30 };

/**
 * An SVG plot with linear axes.
 *
 * The plot scales to the width of its column and keeps its aspect ratio, so a
 * demo behaves on a phone without the demo having to think about it. Children
 * arrive as a function of the scales, which is what lets a demo draw in data
 * coordinates.
 */
export function Plot({
  width = 560,
  height = 240,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  xTicks = 5,
  yTicks = 4,
  tickFormat,
  onPointer,
  children,
}: {
  width?: number;
  height?: number;
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel?: string;
  yLabel?: string;
  xTicks?: number;
  yTicks?: number;
  tickFormat?: (value: number, axis: "x" | "y") => string;
  /** Pointer events in data coordinates, for a demo the reader drags on. */
  onPointer?: (
    event: { x: number; y: number; type: "down" | "move" | "up" },
    native: React.PointerEvent<SVGSVGElement>,
  ) => void;
  children: (scales: Scales) => ReactNode;
}) {
  const scales = useMemo<Scales>(() => {
    const area = {
      left: MARGIN.left,
      right: width - MARGIN.right,
      top: MARGIN.top,
      bottom: height - MARGIN.bottom,
    };
    const [x0, x1] = xDomain;
    const [y0, y1] = yDomain;
    const spanX = x1 - x0 || 1;
    const spanY = y1 - y0 || 1;
    return {
      sx: (x) => area.left + ((x - x0) / spanX) * (area.right - area.left),
      sy: (y) => area.bottom - ((y - y0) / spanY) * (area.bottom - area.top),
      ix: (px) => x0 + ((px - area.left) / (area.right - area.left)) * spanX,
      iy: (py) => y0 + ((area.bottom - py) / (area.bottom - area.top)) * spanY,
      xDomain,
      yDomain,
      area,
    };
  }, [width, height, xDomain, yDomain]);

  const format = tickFormat ?? ((value: number) => shortNumber(value));

  function toData(event: React.PointerEvent<SVGSVGElement>) {
    const svg = event.currentTarget;
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    const py = ((event.clientY - box.top) / box.height) * height;
    return { x: scales.ix(px), y: scales.iy(py) };
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full touch-none select-none"
      style={{ aspectRatio: `${width} / ${height}` }}
      role="img"
      onPointerDown={
        onPointer
          ? (event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              onPointer({ ...toData(event), type: "down" }, event);
            }
          : undefined
      }
      onPointerMove={
        onPointer ? (event) => onPointer({ ...toData(event), type: "move" }, event) : undefined
      }
      onPointerUp={
        onPointer
          ? (event) => {
              event.currentTarget.releasePointerCapture(event.pointerId);
              onPointer({ ...toData(event), type: "up" }, event);
            }
          : undefined
      }
    >
      <g aria-hidden>
        {gridValues(yDomain, yTicks).map((value) => (
          <g key={`y${value}`}>
            <line
              x1={scales.area.left}
              x2={scales.area.right}
              y1={scales.sy(value)}
              y2={scales.sy(value)}
              stroke="var(--rule)"
              strokeWidth={1}
            />
            <text
              x={scales.area.left - 7}
              y={scales.sy(value)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize={10}
              fontFamily="var(--font-mono)"
              fill="var(--ink-3)"
            >
              {format(value, "y")}
            </text>
          </g>
        ))}
        {gridValues(xDomain, xTicks).map((value, index, all) => (
          <text
            key={`x${value}`}
            x={scales.sx(value)}
            y={scales.area.bottom + 15}
            // The last label ends at its tick, so a long one stays inside the plot.
            textAnchor={index === all.length - 1 && all.length > 1 ? "end" : "middle"}
            fontSize={10}
            fontFamily="var(--font-mono)"
            fill="var(--ink-3)"
          >
            {format(value, "x")}
          </text>
        ))}
        <line
          x1={scales.area.left}
          x2={scales.area.right}
          y1={scales.area.bottom}
          y2={scales.area.bottom}
          stroke="var(--rule-strong)"
          strokeWidth={1}
        />
        {xLabel && (
          <text
            x={(scales.area.left + scales.area.right) / 2}
            y={height - 2}
            textAnchor="middle"
            fontSize={10}
            fill="var(--ink-3)"
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={10}
            y={(scales.area.top + scales.area.bottom) / 2}
            textAnchor="middle"
            fontSize={10}
            fill="var(--ink-3)"
            transform={`rotate(-90 10 ${(scales.area.top + scales.area.bottom) / 2})`}
          >
            {yLabel}
          </text>
        )}
      </g>

      <ScaleContext.Provider value={scales}>{children(scales)}</ScaleContext.Provider>
    </svg>
  );
}

/** A polyline through `points`, in data coordinates. */
export function Curve({
  points,
  stroke = "var(--demo)",
  width = 1.8,
  dashed = false,
  fill,
  opacity = 1,
}: {
  points: readonly (readonly [number, number])[];
  stroke?: string;
  width?: number;
  dashed?: boolean;
  /** When set, the area under the curve is filled down to the axis. */
  fill?: string;
  opacity?: number;
}) {
  const { sx, sy, yDomain } = useScales();
  if (points.length === 0) return null;
  const line = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`)
    .join(" ");
  const base = sy(Math.max(0, yDomain[0]));
  return (
    <>
      {fill && (
        <path
          d={`${line} L${sx(points[points.length - 1][0]).toFixed(2)},${base.toFixed(2)} L${sx(points[0][0]).toFixed(2)},${base.toFixed(2)} Z`}
          fill={fill}
          opacity={opacity}
        />
      )}
      <path
        d={line}
        fill="none"
        stroke={stroke}
        strokeWidth={width}
        strokeLinejoin="round"
        strokeLinecap="round"
        strokeDasharray={dashed ? "4 3" : undefined}
        opacity={opacity}
      />
    </>
  );
}

/** Bars from a histogram or a probability mass function, in data coordinates. */
export function Bars({
  bins,
  fill = "color-mix(in oklab, var(--demo) 55%, transparent)",
  stroke = "var(--demo)",
  gap = 1,
}: {
  bins: readonly { x0: number; x1: number; y: number }[];
  fill?: string;
  stroke?: string;
  gap?: number;
}) {
  const { sx, sy, yDomain } = useScales();
  const base = sy(Math.max(0, yDomain[0]));
  return (
    <>
      {bins.map((bin, i) => {
        const left = sx(bin.x0);
        const right = sx(bin.x1);
        const top = sy(bin.y);
        const w = Math.max(0, right - left - gap);
        const h = Math.max(0, base - top);
        if (h <= 0 || w <= 0) return null;
        return (
          <rect
            key={i}
            x={left + gap / 2}
            y={top}
            width={w}
            height={h}
            fill={fill}
            stroke={stroke}
            strokeWidth={0.6}
          />
        );
      })}
    </>
  );
}

/** A vertical rule at `x`, with an optional label at the top. */
export function Rule({
  x,
  label,
  colour = "var(--ink-3)",
  dashed = true,
}: {
  x: number;
  label?: string;
  colour?: string;
  dashed?: boolean;
}) {
  const { sx, area } = useScales();
  const px = sx(x);
  return (
    <>
      <line
        x1={px}
        x2={px}
        y1={area.top}
        y2={area.bottom}
        stroke={colour}
        strokeWidth={1.2}
        strokeDasharray={dashed ? "3 3" : undefined}
      />
      {label && (
        <text
          x={px}
          y={area.top - 1}
          textAnchor="middle"
          fontSize={10}
          fontFamily="var(--font-mono)"
          fill={colour}
        >
          {label}
        </text>
      )}
    </>
  );
}

/** Scattered points, in data coordinates. */
export function Points({
  points,
  r = 3.2,
  fill = "var(--demo)",
  stroke = "var(--canvas)",
  opacity = 1,
}: {
  points: readonly (readonly [number, number])[];
  r?: number;
  fill?: string;
  stroke?: string;
  opacity?: number;
}) {
  const { sx, sy } = useScales();
  return (
    <>
      {points.map(([x, y], i) => (
        <circle
          key={i}
          cx={sx(x)}
          cy={sy(y)}
          r={r}
          fill={fill}
          stroke={stroke}
          strokeWidth={0.8}
          opacity={opacity}
        />
      ))}
    </>
  );
}

/** A small key, for a plot showing more than one thing. */
export function Legend({
  items,
}: {
  items: readonly { label: string; colour: string; dashed?: boolean }[];
}) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5 text-[12px] text-ink-3">
          <span
            className="inline-block h-0 w-4 border-t-2"
            style={{
              borderColor: item.colour,
              borderTopStyle: item.dashed ? "dashed" : "solid",
            }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// Helpers ----------------------------------------------------------------

function gridValues([lo, hi]: [number, number], count: number): number[] {
  // No ticks asked for means no grid, rather than one line at 0/0.
  if (count <= 0) return [];
  const out: number[] = [];
  for (let i = 0; i <= count; i += 1) out.push(lo + ((hi - lo) * i) / count);
  return out;
}

function shortNumber(value: number): string {
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(2);
}


// Audio figures ------------------------------------------------------------

/**
 * Samples, drawn as stems from zero with a dot on top, the way a textbook
 * draws a sampled signal.
 */
export function Stems({
  points,
  colour = "var(--demo)",
  r = 3,
}: {
  points: readonly (readonly [number, number])[];
  colour?: string;
  r?: number;
}) {
  const { sx, sy } = useScales();
  const base = sy(0);
  return (
    <>
      {points.map(([x, y], i) => (
        <g key={i}>
          <line x1={sx(x)} x2={sx(x)} y1={base} y2={sy(y)} stroke={colour} strokeWidth={1.2} />
          <circle cx={sx(x)} cy={sy(y)} r={r} fill={colour} stroke="var(--raise)" strokeWidth={0.8} />
        </g>
      ))}
    </>
  );
}

/**
 * The lowest and highest sample in each of `buckets` equal slices, as points
 * to draw, so a long clip keeps its shape at any width instead of aliasing.
 */
export function envelope(
  samples: ArrayLike<number>,
  sampleRate: number,
  buckets = 600,
  start = 0,
  end = samples.length,
): [number, number][] {
  const count = Math.max(0, end - start);
  if (count <= buckets * 2) {
    const out: [number, number][] = [];
    for (let i = start; i < end; i += 1) out.push([i / sampleRate, samples[i]]);
    return out;
  }
  const out: [number, number][] = [];
  for (let b = 0; b < buckets; b += 1) {
    const from = start + Math.floor((b * count) / buckets);
    const to = start + Math.floor(((b + 1) * count) / buckets);
    let lo = Infinity;
    let hi = -Infinity;
    let loAt = from;
    let hiAt = from;
    for (let i = from; i < to; i += 1) {
      const v = samples[i];
      if (v < lo) {
        lo = v;
        loAt = i;
      }
      if (v > hi) {
        hi = v;
        hiAt = i;
      }
    }
    if (loAt <= hiAt) out.push([loAt / sampleRate, lo], [hiAt / sampleRate, hi]);
    else out.push([hiAt / sampleRate, hi], [loAt / sampleRate, lo]);
  }
  return out;
}

/** A clip against time in seconds, inside a `Plot`. */
export function Waveform({
  samples,
  sampleRate,
  colour = "var(--demo)",
  width = 1.1,
}: {
  samples: ArrayLike<number>;
  sampleRate: number;
  colour?: string;
  width?: number;
}) {
  const points = useMemo(() => envelope(samples, sampleRate), [samples, sampleRate]);
  return <Curve points={points} stroke={colour} width={width} />;
}

/** Values to heatmap levels: `min` becomes 0 and `max` becomes 255. */
export function quantize(values: ArrayLike<number>, min: number, max: number): Uint8Array {
  const out = new Uint8Array(values.length);
  const span = max - min || 1;
  for (let i = 0; i < values.length; i += 1) {
    const t = (values[i] - min) / span;
    out[i] = t <= 0 ? 0 : t >= 1 ? 255 : Math.round(t * 255);
  }
  return out;
}

/**
 * A spectrogram in decibels, from `dsp.stft` and `dsp.toDb`, with time in
 * seconds and frequency in Hz. `rows` keeps a band of rows, `[first, last)`,
 * for a demo that zooms in on part of the spectrum.
 */
export function Spectrogram({
  spec,
  sampleRate,
  floorDb = -80,
  rows,
  colormap = "magma",
  height,
  description,
}: {
  /** Row-major decibels, `bins` rows by `frames` columns, row 0 at 0 Hz. */
  spec: SpectrogramData;
  sampleRate: number;
  floorDb?: number;
  rows?: [number, number];
  colormap?: ColormapName;
  height?: number;
  description?: string;
}) {
  const [first, last] = rows ?? [0, spec.bins];
  const levels = useMemo(
    () => quantize(spec.data.subarray(first * spec.frames, last * spec.frames), floorDb, 0),
    [spec, first, last, floorDb],
  );
  const binHz = sampleRate / spec.nFft;
  return (
    <HeatmapCanvas
      levels={levels}
      rows={last - first}
      cols={spec.frames}
      origin="lower"
      colormap={colormap}
      x={{ label: "Time", unit: "s", min: 0, max: (spec.frames * spec.hop) / sampleRate }}
      y={{ label: "Frequency", unit: "Hz", min: Math.max(0, (first - 0.5) * binHz), max: (last - 0.5) * binHz }}
      value={{ label: "dB", min: floorDb, max: 0 }}
      height={height}
      description={description ?? "A spectrogram"}
    />
  );
}
