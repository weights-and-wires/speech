import { useContext, useEffect, useId, useMemo, useRef, useState } from "react";

import { colormap } from "../lib/colormap";
import { dtw, gaussian, seededRandom, type DtwStepPattern } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  PrintedFigure,
  Readout,
  Readouts,
  Slider,
} from "./kit";

/**
 * A "word" as a feature track: formant targets (F1, F2 in kHz) held for a
 * number of frames each, with smooth transitions between them. Saying it at
 * another speed stretches each segment by its own factor.
 */
type Segment = { f1: number; f2: number; frames: number };

const WORD: Segment[] = [
  { f1: 0.3, f2: 2.2, frames: 6 },
  { f1: 0.55, f2: 1.8, frames: 9 },
  { f1: 0.7, f2: 1.2, frames: 8 },
  { f1: 0.35, f2: 0.9, frames: 7 },
];
/** How much more slowly the second saying takes each segment, before the reader's stretch. */
const SLOWER = [1.2, 1, 1.3, 1.2];
const OTHER: Segment[] = [
  { f1: 0.7, f2: 1.1, frames: 10 },
  { f1: 0.3, f2: 2.3, frames: 14 },
  { f1: 0.45, f2: 1.9, frames: 12 },
  { f1: 0.3, f2: 2.2, frames: 9 },
];

function track(segments: Segment[], stretch: number[], seed: number): [number, number][] {
  const random = seededRandom(seed);
  const targets: { at: number; f1: number; f2: number }[] = [];
  let total = 0;
  segments.forEach((s, k) => {
    const frames = Math.round(s.frames * (stretch[k] ?? 1));
    targets.push({ at: total + frames / 2, f1: s.f1, f2: s.f2 });
    total += frames;
  });
  const out: [number, number][] = [];
  for (let t = 0; t < total; t += 1) {
    // Interpolate between the targets either side, with a smooth ease.
    let k = 0;
    while (k < targets.length - 1 && targets[k + 1].at <= t) k += 1;
    const a = targets[k];
    const b = targets[Math.min(targets.length - 1, k + 1)];
    const x = b.at > a.at ? Math.min(1, Math.max(0, (t - a.at) / (b.at - a.at))) : 0;
    const e = x * x * (3 - 2 * x);
    out.push([
      a.f1 + (b.f1 - a.f1) * e + 0.02 * gaussian(random),
      a.f2 + (b.f2 - a.f2) * e + 0.04 * gaussian(random),
    ]);
  }
  return out;
}

const FIRST = track(WORD, [1, 1, 1, 1], 1);
const DIFFERENT = track(OTHER, [1, 1, 1, 1], 3);

/** Values as an image, row 0 at the bottom, cells that are infinite left clear. */
function heatmapUrl(values: Float64Array, rows: number, cols: number): string | null {
  if (typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  if (!context) return null;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  const lut = colormap("viridis");
  const image = context.createImageData(cols, rows);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const v = values[r * cols + c];
      if (!Number.isFinite(v)) continue;
      const at = ((rows - 1 - r) * cols + c) * 4;
      const level = Math.round(((v - lo) / (hi - lo || 1)) * 255) * 3;
      image.data[at] = lut[level];
      image.data[at + 1] = lut[level + 1];
      image.data[at + 2] = lut[level + 2];
      image.data[at + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  return canvas.toDataURL();
}

const PATTERNS: { value: DtwStepPattern; label: string; note: string }[] = [
  { value: "symmetric1", label: "Symmetric, all steps 1", note: "D / path length" },
  { value: "symmetric2", label: "Symmetric, diagonal 2", note: "D / (n + m)" },
  { value: "asymmetric", label: "Asymmetric, slope 0 to 2", note: "D / m" },
];

/** A round tick count for an axis of `frames` frames. */
const ticksFor = (frames: number) => (frames % 10 === 0 ? frames / 10 : frames % 5 === 0 ? frames / 5 : 4);

export default function DtwPath({ caption }: DemoProps) {
  const [other, setOther] = useState<"same" | "other">("same");
  const [stretch, setStretch] = useState(2.2);
  const [pattern, setPattern] = useState<DtwStepPattern>("symmetric1");
  const [show, setShow] = useState<"local" | "cumulative">("local");
  /** The Sakoe-Chiba radius in frames, or null for no band. */
  const [band, setBand] = useState<number | null>(null);
  const [revealed, setRevealed] = useState<number | null>(null);
  const timer = useRef<number | null>(null);
  const clip = `dtw-clip-${useId().replace(/:/g, "")}`;
  // A printed page is shorter than a screen, so the matrix is too.
  const printed = useContext(PrintedFigure) !== null;

  const second = useMemo(
    () => (other === "same" ? track(WORD, SLOWER.map((s, k) => (k === 1 ? s * stretch : s)), 2) : DIFFERENT),
    [other, stretch],
  );
  // The second sequence runs up the rows, the first along the columns.
  const n = FIRST.length;
  const m = second.length;

  const cost = useMemo(() => {
    const out = new Float64Array(m * n);
    for (let i = 0; i < m; i += 1) {
      for (let j = 0; j < n; j += 1) out[i * n + j] = Math.hypot(second[i][0] - FIRST[j][0], second[i][1] - FIRST[j][1]);
    }
    return out;
  }, [second, n, m]);

  const radius = band === null ? Infinity : band;
  const result = useMemo(() => dtw(cost, m, n, { band: radius, pattern }), [cost, n, m, radius, pattern]);

  const slope = (n - 1) / (m - 1);
  const inBand = (k: number) => Math.abs((k % n) - Math.floor(k / n) * slope) <= radius + 1e-9;
  const shownValues = useMemo(() => {
    if (show === "cumulative") return result.cumulative;
    if (radius === Infinity) return cost;
    // Outside the band, the local cost is never looked at.
    return cost.map((v, k) => (inBand(k) ? v : Infinity));
    // `inBand` depends only on n, m, and radius.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [show, result, cost, radius, n, m]);
  const url = useMemo(() => heatmapUrl(shownValues, m, n), [shownValues, n, m]);

  useEffect(() => () => {
    if (timer.current !== null) window.clearInterval(timer.current);
  }, []);
  useEffect(() => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
    setRevealed(null);
  }, [result]);

  function trace() {
    if (timer.current !== null) window.clearInterval(timer.current);
    let count = 1;
    setRevealed(1);
    timer.current = window.setInterval(() => {
      count += 1;
      setRevealed(count);
      if (count >= result.path.length && timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
        setRevealed(null);
      }
    }, 45);
  }

  const path = result.path;
  const visible = revealed === null ? path : path.slice(Math.max(0, path.length - revealed));
  const pathPoints = visible.map(([i, j]): [number, number] => [j + 0.5, i + 0.5]);
  const head = revealed !== null ? visible[0] : null;
  const banded = radius !== Infinity;
  let cells = n * m;
  if (banded) {
    cells = 0;
    for (let k = 0; k < n * m; k += 1) if (inBand(k)) cells += 1;
  }
  const found = Number.isFinite(result.distance);
  const normaliser = pattern === "symmetric1" ? path.length : pattern === "symmetric2" ? n + m : m;
  const note = PATTERNS.find((p) => p.value === pattern)?.note ?? "";

  // The second sequence's frames averaged onto each frame of the first, along the path.
  const warped = useMemo(() => {
    const sums = Array.from({ length: n }, () => [0, 0, 0]);
    for (const [i, j] of path) {
      sums[j][0] += second[i][1];
      sums[j][1] += second[i][0];
      sums[j][2] += 1;
    }
    return sums
      .map(([f2, f1, count], j) => (count ? { j, f2: f2 / count, f1: f1 / count } : null))
      .filter((v): v is { j: number; f2: number; f1: number } => v !== null);
  }, [path, second, n]);

  const series = (x: [number, number][], which: 0 | 1) => x.map((v, t): [number, number] => [t, v[which]]);
  const longest = Math.ceil(Math.max(n, m) / 10) * 10;

  return (
    <DemoFrame
      title="Warping one sequence onto another"
      caption={caption}
      controls={
        <>
          <Choice
            label="Second sequence"
            value={other}
            options={[
              { value: "same", label: "The same word, slower" },
              { value: "other", label: "A different word" },
            ]}
            onChange={setOther}
          />
          {other === "same" && (
            <Slider
              label="Slow down its second segment"
              value={stretch}
              min={1}
              max={3.5}
              step={0.1}
              format={(v) => `${v.toFixed(1)}x`}
              onChange={setStretch}
            />
          )}
          <Choice label="Step pattern" value={pattern} options={PATTERNS} onChange={setPattern} />
          <Slider
            label="Sakoe-Chiba band"
            value={band === null ? n : Math.min(band, n)}
            min={1}
            max={n}
            format={(v) => (band === null ? "none" : `±${v} frames`)}
            onChange={(v) => setBand(v >= n ? null : v)}
          />
          <Choice
            label="Shade the cells by"
            value={show}
            options={[
              { value: "local", label: "Local cost" },
              { value: "cumulative", label: "Cumulative cost" },
            ]}
            onChange={setShow}
          />
          <Actions>
            <Action onClick={trace}>Trace the path back</Action>
          </Actions>
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="min-w-0">
          <p className="text-[12.5px] font-medium text-ink-2">The two sequences, frame by frame</p>
          <Plot width={360} height={170} xDomain={[0, longest]} yDomain={[0, 2.5]} xLabel="frame" yLabel="kHz" yTicks={5} xTicks={ticksFor(longest)} tickFormat={(v, axis) => v.toFixed(axis === "y" ? 1 : 0)}>
            {() => (
              <>
                <Curve points={series(FIRST, 1)} width={1.6} />
                <Curve points={series(FIRST, 0)} width={1.2} dashed />
                <Curve points={series(second, 1)} stroke="var(--series-2)" width={1.6} />
                <Curve points={series(second, 0)} stroke="var(--series-2)" width={1.2} dashed />
              </>
            )}
          </Plot>
        </div>
        <div className="min-w-0">
          <p className="text-[12.5px] font-medium text-ink-2">The second, warped onto the first</p>
          <Plot width={360} height={170} xDomain={[0, n]} yDomain={[0, 2.5]} xLabel="frame of the first" yLabel="kHz" yTicks={5} xTicks={ticksFor(n)} tickFormat={(v, axis) => v.toFixed(axis === "y" ? 1 : 0)}>
            {() => (
              <>
                <Curve points={series(FIRST, 1)} width={1.6} />
                <Curve points={series(FIRST, 0)} width={1.2} dashed />
                {found && (
                  <>
                    <Curve points={warped.map((w) => [w.j, w.f2] as [number, number])} stroke="var(--series-2)" width={1.6} />
                    <Curve points={warped.map((w) => [w.j, w.f1] as [number, number])} stroke="var(--series-2)" width={1.2} dashed />
                  </>
                )}
              </>
            )}
          </Plot>
        </div>
      </div>
      <Legend
        items={[
          { label: `first, n = ${n} frames`, colour: "var(--demo)" },
          { label: `second, m = ${m} frames`, colour: "var(--series-2)" },
          { label: "F2 solid, F1 dashed", colour: "var(--ink-4)" },
        ]}
      />

      <div className="mt-4">
        <Plot
          width={560}
          height={printed ? 270 : 380}
          xDomain={[0, n]}
          yDomain={[0, m]}
          xLabel="frame of the first sequence"
          yLabel="frame of the second"
          xTicks={ticksFor(n)}
          yTicks={ticksFor(m)}
          tickFormat={(v) => v.toFixed(0)}
        >
          {({ sx, sy }) => (
            <>
              {url && (
                <image
                  href={url}
                  x={sx(0)}
                  y={sy(m)}
                  width={sx(n) - sx(0)}
                  height={sy(0) - sy(m)}
                  preserveAspectRatio="none"
                  style={{ imageRendering: "pixelated" }}
                />
              )}
              <clipPath id={clip}>
                <rect x={sx(0)} y={sy(m)} width={sx(n) - sx(0)} height={sy(0) - sy(m)} />
              </clipPath>
              {banded && (
                <g clipPath={`url(#${clip})`}>
                  {[radius, -radius].map((offset) => (
                    <Curve
                      key={offset}
                      points={[
                        [0.5 + offset, 0.5],
                        [n - 0.5 + offset, m - 0.5],
                      ]}
                      stroke="var(--ink-3)"
                      width={1}
                      dashed
                    />
                  ))}
                </g>
              )}
              {pathPoints.length > 1 && (
                <>
                  <Curve points={pathPoints} stroke="var(--raise)" width={4.5} />
                  <Curve points={pathPoints} stroke="var(--bad)" width={2.2} />
                </>
              )}
              {head && (
                <circle cx={sx(head[1] + 0.5)} cy={sy(head[0] + 0.5)} r={5} fill="var(--bad)" stroke="var(--raise)" strokeWidth={1.5} />
              )}
            </>
          )}
        </Plot>
      </div>
      <p className="mt-1 text-[12.5px] leading-snug text-ink-3">
        {show === "local"
          ? "Each cell is the distance between a frame of the first and a frame of the second: dark is close."
          : "Each cell is the cheapest total cost of any path from the corner to it: the path follows the dark valley."}{" "}
        Where the second lingers, the path climbs straight up.
        {banded && " Cells outside the band are never computed, so they are left blank."}
        {!found && " No path from corner to corner fits in this band."}
      </p>

      <Readouts>
        <Readout label="DTW distance" value={found ? result.distance.toFixed(2) : "none"} tone={found ? "demo" : "bad"} hint="D at the far corner" />
        <Readout label="Normalized" value={found ? (result.distance / normaliser).toFixed(3) : "n/a"} hint={note} />
        <Readout label="Path length" value={found ? path.length : 0} hint={`${n} by ${m} frames`} />
        <Readout label="Cells in the band" value={`${Math.round((cells / (n * m)) * 100)}%`} hint={`${cells} of ${n * m}`} />
      </Readouts>
    </DemoFrame>
  );
}
