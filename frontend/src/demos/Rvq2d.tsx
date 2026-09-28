import { useMemo, useState } from "react";

import { gaussian, seededRandom } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  Readout,
  Readouts,
  Slider,
  useScales,
} from "./kit";

type Point = [number, number];
type Quantizer = "rvq" | "fsq";

const POINTS = 400;
const MAX_STAGES = 6;
const SIZES = ["4", "9", "16", "25", "64"] as const;
type Size = (typeof SIZES)[number];
/** The codec frame rate the bitrate readout assumes: EnCodec's 75 Hz at 24 kHz. */
const FRAME_RATE = 75;

/** Points from a mixture of five tilted Gaussians. */
function mixture(seed: number): Point[] {
  const random = seededRandom(seed);
  const components = Array.from({ length: 5 }, () => ({
    centre: [4 * random() - 2, 4 * random() - 2] as Point,
    spread: [0.15 + 0.35 * random(), 0.1 + 0.2 * random()] as Point,
    angle: Math.PI * random(),
  }));
  return Array.from({ length: POINTS }, () => {
    const c = components[Math.floor(random() * components.length)];
    const u = gaussian(random) * c.spread[0];
    const v = gaussian(random) * c.spread[1];
    return [
      c.centre[0] + u * Math.cos(c.angle) - v * Math.sin(c.angle),
      c.centre[1] + u * Math.sin(c.angle) + v * Math.cos(c.angle),
    ];
  });
}

const dist2 = (a: Point, b: Point) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

function nearest(p: Point, codebook: Point[]): number {
  let best = 0;
  for (let k = 1; k < codebook.length; k += 1) if (dist2(p, codebook[k]) < dist2(p, codebook[best])) best = k;
  return best;
}

/** k-means with k-means++ seeding and Lloyd's iterations. */
function kmeans(points: Point[], k: number, random: () => number, iterations = 25): Point[] {
  const centres: Point[] = [points[Math.floor(random() * points.length)]];
  while (centres.length < k) {
    const d = points.map((p) => Math.min(...centres.map((c) => dist2(p, c))));
    const total = d.reduce((a, b) => a + b, 0);
    let target = random() * total;
    let pick = 0;
    while (pick < points.length - 1 && target > d[pick]) target -= d[pick++];
    centres.push(points[pick]);
  }
  for (let it = 0; it < iterations; it += 1) {
    const sums = centres.map(() => [0, 0, 0]);
    for (const p of points) {
      const j = nearest(p, centres);
      sums[j][0] += p[0];
      sums[j][1] += p[1];
      sums[j][2] += 1;
    }
    sums.forEach(([x, y, n], j) => {
      // An empty cluster restarts on the point worst served, as dead-code restarts do.
      if (n === 0) {
        let worst = points[0];
        for (const p of points) if (dist2(p, centres[nearest(p, centres)]) > dist2(worst, centres[nearest(worst, centres)])) worst = p;
        centres[j] = worst;
      } else centres[j] = [x / n, y / n];
    });
  }
  return centres;
}

/**
 * Finite scalar quantization of 2-D residuals (Mentzer et al., 2024): each
 * coordinate is bounded with tanh to L levels and rounded, with the half-step
 * offset for even L, so the grid is fixed and nothing is learned but its
 * scale, which here is fitted to the residuals the way a projection would be.
 */
function fsqGrid(points: Point[], levels: number): Point[] {
  const half = (levels - 1) / 2;
  const offset = levels % 2 === 0 ? 0.5 : 0;
  const reach = Math.max(1e-6, ...points.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
  let best: { scale: number; error: number } = { scale: reach, error: Infinity };
  for (let i = 0; i < 40; i += 1) {
    const scale = reach * (0.05 + (2.5 * i) / 39);
    const q = (z: number) => {
      const code = Math.max(-half, Math.min(half, Math.round(half * Math.tanh(z / scale) + offset) - offset));
      return (code * scale) / Math.max(half, 0.5);
    };
    const error = points.reduce((s, p) => s + (p[0] - q(p[0])) ** 2 + (p[1] - q(p[1])) ** 2, 0);
    if (error < best.error) best = { scale, error };
  }
  const grid: Point[] = [];
  for (let i = 0; i < levels; i += 1) {
    for (let j = 0; j < levels; j += 1) {
      const a = ((i - half) * best.scale) / Math.max(half, 0.5);
      const b = ((j - half) * best.scale) / Math.max(half, 0.5);
      grid.push([a, b]);
    }
  }
  return grid;
}

type Stage = { input: Point[]; codebook: Point[]; codes: number[]; reconstruction: Point[]; mse: number };

/** Residual quantization, SoundStream's Algorithm 1: quantize, subtract, repeat. */
function residualQuantize(points: Point[], size: number, kind: Quantizer, seed: number): Stage[] {
  const random = seededRandom(seed * 7 + size);
  const stages: Stage[] = [];
  let residual = points;
  let reconstruction: Point[] = points.map(() => [0, 0]);
  for (let s = 0; s < MAX_STAGES; s += 1) {
    const codebook = kind === "rvq" ? kmeans(residual, size, random) : fsqGrid(residual, Math.round(Math.sqrt(size)));
    const codes = residual.map((p) => nearest(p, codebook));
    const input = residual;
    reconstruction = reconstruction.map((r, i) => [r[0] + codebook[codes[i]][0], r[1] + codebook[codes[i]][1]]);
    residual = residual.map((p, i) => [p[0] - codebook[codes[i]][0], p[1] - codebook[codes[i]][1]]);
    const mse = residual.reduce((sum, p) => sum + p[0] ** 2 + p[1] ** 2, 0) / points.length;
    stages.push({ input, codebook, codes, reconstruction, mse });
  }
  return stages;
}

function variance(points: Point[]): number {
  const mx = points.reduce((s, p) => s + p[0], 0) / points.length;
  const my = points.reduce((s, p) => s + p[1], 0) / points.length;
  return points.reduce((s, p) => s + (p[0] - mx) ** 2 + (p[1] - my) ** 2, 0) / points.length;
}

const NICE = [0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.6, 2, 2.4, 3.2, 4];

const db = (value: number) => 10 * Math.log10(Math.max(value, 1e-12));

export default function Rvq2d({ caption }: DemoProps) {
  const [stage, setStage] = useState(2);
  const [size, setSize] = useState<Size>("4");
  const [kind, setKind] = useState<Quantizer>("rvq");
  const [seed, setSeed] = useState(5);

  const points = useMemo(() => mixture(seed), [seed]);
  const spread = useMemo(() => variance(points), [points]);
  const both = useMemo(
    () => ({
      rvq: residualQuantize(points, Number(size), "rvq", seed),
      fsq: residualQuantize(points, Number(size), "fsq", seed),
    }),
    [points, size, seed],
  );
  const stages = both[kind];
  const current = stages[stage - 1];
  const bitsPerStage = Math.log2(Number(size));
  const bits = stage * bitsPerStage;
  // The residual's extent, rounded up to a span whose quarters are round numbers.
  const extent = Math.max(0.01, ...current.input.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.04;
  const reach = NICE.find((v) => v >= extent) ?? Math.ceil(extent);
  const errorDb = db(current.mse / spread);

  const curve = (which: Quantizer): [number, number][] =>
    both[which].map((s, i) => [i + 1, db(s.mse / spread)]);
  const lowest = Math.min(...curve("rvq").map((p) => p[1]), ...curve("fsq").map((p) => p[1]));
  const floorDb = -10 * Math.max(1, Math.ceil(-lowest / 10));

  return (
    <DemoFrame
      title="Quantizing the residual, stage by stage"
      caption={caption}
      controls={
        <>
          <Slider label="Stages used" value={stage} min={1} max={MAX_STAGES} onChange={setStage} />
          <Choice
            label="Codebook size, K"
            value={size}
            options={SIZES.map((k) => ({ value: k, label: k }))}
            onChange={setSize}
          />
          <Choice
            label="Quantizer"
            value={kind}
            options={[
              { value: "rvq", label: "RVQ, k-means codebooks" },
              { value: "fsq", label: "Residual FSQ, a fixed grid" },
            ]}
            onChange={setKind}
          />
          <Actions>
            <Action tone="quiet" onClick={() => setSeed((s) => s + 1)}>
              Draw new points
            </Action>
          </Actions>
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">
            Stage {stage}: {stage === 1 ? "the data" : `the residual after ${stage - 1}`}, and codebook {stage}
          </p>
          <Plot width={300} height={280} xDomain={[-reach, reach]} yDomain={[-reach, reach]} xTicks={4} yTicks={4}>
            {() => (
              <>
                <Links from={current.input} to={current.codes.map((c) => current.codebook[c])} colour="var(--ink-4)" />
                <Points points={current.input} r={1.8} fill="var(--ink-3)" stroke="none" opacity={0.7} />
                <Codewords points={current.codebook} />
              </>
            )}
          </Plot>
        </div>
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">
            The data, and its reconstruction from {stage} {stage === 1 ? "code" : "codes"}
          </p>
          <Plot width={300} height={280} xDomain={[-4, 4]} yDomain={[-4, 4]} xTicks={4} yTicks={4}>
            {() => (
              <>
                <Links from={points} to={current.reconstruction} colour="var(--bad)" />
                <Points points={points} r={1.8} fill="var(--ink-4)" stroke="none" opacity={0.8} />
                <Points points={current.reconstruction} r={2.2} fill="var(--demo)" stroke="none" />
              </>
            )}
          </Plot>
        </div>
      </div>
      <Legend
        items={[
          { label: "points", colour: "var(--ink-3)" },
          { label: "codewords", colour: "var(--series-2)" },
          { label: "reconstruction", colour: "var(--demo)" },
          { label: "error left over", colour: "var(--bad)" },
        ]}
      />
      <p className="mt-1 text-[12px] leading-snug text-ink-3">
        The left plot zooms in as the residual shrinks: read its ticks.
      </p>

      <div className="mt-4">
        <p className="mb-1 text-[12px] font-medium text-ink-3">
          Error left over, relative to the data's variance, as stages add bits
        </p>
        <Plot
          height={170}
          xDomain={[0.7, MAX_STAGES + 0.3]}
          yDomain={[floorDb, 0]}
          xTicks={0}
          yTicks={-floorDb / 10}
          xLabel={`stages, ${bitsPerStage.toFixed(bitsPerStage % 1 ? 2 : 0)} bits each`}
          yLabel="dB"
          tickFormat={(v) => v.toFixed(0)}
        >
          {({ sx, area }) => (
            <>
              {Array.from({ length: MAX_STAGES }, (_, i) => (
                <text
                  key={i}
                  x={sx(i + 1)}
                  y={area.bottom + 15}
                  textAnchor="middle"
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                  fill="var(--ink-3)"
                >
                  {i + 1}
                </text>
              ))}
              <Curve points={curve("fsq")} stroke="var(--series-3)" dashed={kind !== "fsq"} opacity={kind === "fsq" ? 1 : 0.7} />
              <Curve points={curve("rvq")} stroke="var(--demo)" dashed={kind !== "rvq"} opacity={kind === "rvq" ? 1 : 0.7} />
              <Points points={curve(kind)} fill={kind === "rvq" ? "var(--demo)" : "var(--series-3)"} />
              <Points points={[[stage, errorDb]]} r={5.5} fill="var(--series-2)" />
            </>
          )}
        </Plot>
        <Legend
          items={[
            { label: "RVQ, k-means", colour: "var(--demo)", dashed: kind !== "rvq" },
            { label: "residual FSQ", colour: "var(--series-3)", dashed: kind !== "fsq" },
          ]}
        />
      </div>

      <Readouts>
        <Readout label="Bits per vector" value={bits.toFixed(bits % 1 ? 1 : 0)} hint={`${stage} × log2 ${size}`} tone="demo" />
        <Readout label="Error" value={`${errorDb.toFixed(1)} dB`} hint="mean squared, over the variance" />
        <Readout
          label="Entries stored"
          value={(stage * Number(size)).toLocaleString("en-US")}
          hint={`one flat codebook: ${Number(size) ** stage > 1e6 ? (Number(size) ** stage).toExponential(1) : (Number(size) ** stage).toLocaleString("en-US")}`}
        />
        <Readout
          label={`At ${FRAME_RATE} frames/s`}
          value={`${Math.round(bits * FRAME_RATE).toLocaleString("en-US")} bps`}
          hint="frame rate × stages × log2 K"
        />
      </Readouts>
    </DemoFrame>
  );
}

/** A thin line from each point to its partner. */
function Links({ from, to, colour }: { from: Point[]; to: Point[]; colour: string }) {
  const { sx, sy } = useScales();
  const d = from
    .map((p, i) => `M${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}L${sx(to[i][0]).toFixed(1)},${sy(to[i][1]).toFixed(1)}`)
    .join("");
  return <path d={d} stroke={colour} strokeWidth={0.7} opacity={0.45} fill="none" />;
}

/** Codewords as rings, so the points under them stay visible. */
function Codewords({ points }: { points: Point[] }) {
  const { sx, sy, area } = useScales();
  return (
    <>
      {points.map(([x, y], i) => {
        const cx = sx(x);
        const cy = sy(y);
        if (cx < area.left || cx > area.right || cy < area.top || cy > area.bottom) return null;
        return <circle key={i} cx={cx} cy={cy} r={4.6} fill="none" stroke="var(--series-2)" strokeWidth={2} />;
      })}
    </>
  );
}
