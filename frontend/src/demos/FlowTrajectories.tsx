import { useMemo, useState } from "react";

import { gaussian, seededRandom } from "./dsp";
import { useDebounced } from "./figure";
import type { DemoProps } from "./index";
import {
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  Readout,
  Readouts,
  Slider,
  Toggle,
  useScales,
} from "./kit";

type Vec = [number, number];
type Target = "moons" | "eight";
type Solver = "euler" | "midpoint" | "rk4";
type Field = (x: Vec, t: number) => Vec;

const PARTICLES = 150;
const PAIRS = 256;
const TRACKS = 60;
const STEP_CHOICES = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64];
const DOMAIN: [number, number] = [-3.2, 3.2];
const NFE: Record<Solver, number> = { euler: 1, midpoint: 2, rk4: 4 };

type Component = { mean: Vec; sd: number; weight: number };

/**
 * The target, as a mixture of small isotropic Gaussians so that its marginal
 * velocity has a closed form: two moons, drawn as twenty blobs along two
 * arcs, or eight blobs on a circle.
 */
function mixture(target: Target): Component[] {
  if (target === "eight") {
    return Array.from({ length: 8 }, (_, i) => ({
      mean: [2.2 * Math.cos((i * Math.PI) / 4), 2.2 * Math.sin((i * Math.PI) / 4)] as Vec,
      sd: 0.18,
      weight: 1 / 8,
    }));
  }
  const out: Component[] = [];
  for (let i = 0; i < 10; i += 1) {
    const a = (Math.PI * i) / 9;
    out.push({ mean: [1.5 * (Math.cos(a) - 0.5), 1.5 * (Math.sin(a) - 0.25)], sd: 0.13, weight: 1 / 20 });
    out.push({ mean: [1.5 * (0.5 - Math.cos(a)), 1.5 * (0.25 - Math.sin(a))], sd: 0.13, weight: 1 / 20 });
  }
  return out;
}

/**
 * The marginal velocity of the straight path x_t = (1 - t) x_0 + t x_1 with
 * independent pairs and sigma_min = 0 (Lipman convention: x_0 is noise at
 * t = 0, x_1 is data at t = 1). For one component N(mu, s^2 I), x_t is
 * N(t mu, ((1 - t)^2 + t^2 s^2) I) and the velocity is
 * mu + (t s^2 - (1 - t)) / ((1 - t)^2 + t^2 s^2) (x - t mu); the mixture's
 * is the posterior-weighted sum over components.
 */
function exactField(parts: Component[]): Field {
  return ([x, y], t) => {
    let top = -Infinity;
    const logs = parts.map((c) => {
      const v = (1 - t) ** 2 + t * t * c.sd * c.sd;
      const d2 = (x - t * c.mean[0]) ** 2 + (y - t * c.mean[1]) ** 2;
      const l = Math.log(c.weight) - Math.log(v) - d2 / (2 * v);
      top = Math.max(top, l);
      return l;
    });
    let total = 0;
    let vx = 0;
    let vy = 0;
    parts.forEach((c, k) => {
      const r = Math.exp(logs[k] - top);
      const v = (1 - t) ** 2 + t * t * c.sd * c.sd;
      const gain = (t * c.sd * c.sd - (1 - t)) / v;
      total += r;
      vx += r * (c.mean[0] + gain * (x - t * c.mean[0]));
      vy += r * (c.mean[1] + gain * (y - t * c.mean[1]));
    });
    return [vx / total, vy / total];
  };
}

/**
 * With minibatch OT pairs there is no closed form, so the field is the
 * kernel regression of the pairs' velocities x_1 - x_0 on their positions
 * at time t: what a flow trained on these pairs would learn, smoothed. The
 * kernel narrows toward t = 1, so the samples land on the data.
 */
function pairField(x0: Vec[], x1: Vec[]): Field {
  return ([x, y], t) => {
    // Wide where the pairs' positions are spread like noise, narrow near the data.
    const bandwidth = 0.35 * (1 - t) + 0.06;
    let total = 0;
    let vx = 0;
    let vy = 0;
    for (let i = 0; i < x0.length; i += 1) {
      const px = (1 - t) * x0[i][0] + t * x1[i][0];
      const py = (1 - t) * x0[i][1] + t * x1[i][1];
      const w = Math.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * bandwidth * bandwidth)) + 1e-300;
      total += w;
      vx += w * (x1[i][0] - x0[i][0]);
      vy += w * (x1[i][1] - x0[i][1]);
    }
    return total > 1e-200 ? [vx / total, vy / total] : [0, 0];
  };
}

/** One step of Euler, the midpoint rule, or classical Runge-Kutta. */
function step(field: Field, solver: Solver, x: Vec, t: number, h: number): Vec {
  const add = (a: Vec, b: Vec, s: number): Vec => [a[0] + s * b[0], a[1] + s * b[1]];
  const k1 = field(x, t);
  if (solver === "euler") return add(x, k1, h);
  const k2 = field(add(x, k1, h / 2), t + h / 2);
  if (solver === "midpoint") return add(x, k2, h);
  const k3 = field(add(x, k2, h / 2), t + h / 2);
  const k4 = field(add(x, k3, h), t + h);
  return [x[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), x[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])];
}

function integrate(field: Field, solver: Solver, steps: number, starts: Vec[]) {
  const h = 1 / steps;
  const paths = starts.map((s) => [s]);
  for (let n = 0; n < steps; n += 1) {
    // t from the step count, never accumulated, so the last step lands on 1.
    const t = n / steps;
    paths.forEach((path) => path.push(step(field, solver, path[path.length - 1], t, h)));
  }
  return paths;
}

/** Minimum-cost assignment by the Hungarian algorithm, O(n^3); returns the column for each row. */
function hungarian(cost: number[][]): number[] {
  const n = cost.length;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1);
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Infinity);
    const used = new Uint8Array(n + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j += 1) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const assignment = new Array<number>(n);
  for (let j = 1; j <= n; j += 1) assignment[p[j] - 1] = j - 1;
  return assignment;
}

function crossings(a: Vec[], b: Vec[]): number {
  const orient = (p: Vec, q: Vec, r: Vec) => Math.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]));
  let count = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = i + 1; j < a.length; j += 1) {
      if (orient(a[i], b[i], a[j]) !== orient(a[i], b[i], b[j]) && orient(a[j], b[j], a[i]) !== orient(a[j], b[j], b[i])) count += 1;
    }
  }
  return count;
}

/** Path length over the straight distance, averaged: 1 for straight paths. */
function straightness(paths: Vec[][]): number {
  let sum = 0;
  for (const path of paths) {
    let length = 0;
    for (let i = 1; i < path.length; i += 1) length += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    const chord = Math.hypot(path[path.length - 1][0] - path[0][0], path[path.length - 1][1] - path[0][1]);
    sum += length / Math.max(chord, 1e-9);
  }
  return sum / paths.length;
}

export default function FlowTrajectories({ caption }: DemoProps) {
  const [target, setTarget] = useState<Target>("moons");
  const [solver, setSolver] = useState<Solver>("euler");
  const [stepIndex, setStepIndex] = useState(3);
  const [fieldTime, setFieldTime] = useState(0.5);
  const [showPairs, setShowPairs] = useState(false);
  const [ot, setOt] = useState(false);
  const steps = useDebounced(STEP_CHOICES[stepIndex], 120);

  const world = useMemo(() => {
    const parts = mixture(target);
    const random = seededRandom(target === "moons" ? 3 : 4);
    const noise = (): Vec => [gaussian(random), gaussian(random)];
    const data = (): Vec => {
      let u = random();
      const c = parts.find((p) => (u -= p.weight) < 0) ?? parts[parts.length - 1];
      return [c.mean[0] + c.sd * gaussian(random), c.mean[1] + c.sd * gaussian(random)];
    };
    const starts = Array.from({ length: PARTICLES }, noise);
    const x0 = Array.from({ length: PAIRS }, noise);
    const x1 = Array.from({ length: PAIRS }, data);
    const cost = x0.map((a) => x1.map((b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
    const matched = hungarian(cost).map((j) => x1[j]);
    return { parts, starts, x0, x1, matched, exact: exactField(parts) };
  }, [target]);

  const field = useMemo(
    () => (ot ? pairField(world.x0, world.matched) : world.exact),
    [ot, world],
  );
  const reference = useMemo(() => integrate(field, "rk4", ot ? 48 : 128, world.starts), [field, world, ot]);
  const paths = useMemo(() => integrate(field, solver, steps, world.starts), [field, solver, steps, world]);
  const error =
    paths.reduce((s, path, i) => {
      const a = path[path.length - 1];
      const b = reference[i][reference[i].length - 1];
      return s + Math.hypot(a[0] - b[0], a[1] - b[1]);
    }, 0) / PARTICLES;

  // The convergence of each solver on the exact field, against a fine RK4.
  const convergence = useMemo(() => {
    const exactRef = integrate(world.exact, "rk4", 256, world.starts.slice(0, 50));
    const counts = [1, 2, 4, 8, 16, 32];
    const run = (s: Solver): [number, number][] =>
      counts.map((n) => {
        const ends = integrate(world.exact, s, n, world.starts.slice(0, 50));
        const e =
          ends.reduce((sum, path, i) => {
            const a = path[path.length - 1];
            const b = exactRef[i][exactRef[i].length - 1];
            return sum + Math.hypot(a[0] - b[0], a[1] - b[1]);
          }, 0) / 50;
        return [Math.log10(n * NFE[s]), Math.log10(Math.max(e, 1e-8))];
      });
    return { euler: run("euler"), midpoint: run("midpoint"), rk4: run("rk4") };
  }, [world]);

  const pairs = (ot ? world.matched : world.x1).slice(0, 90);
  const pairCost = (targets: Vec[]) =>
    targets.reduce((s, b, i) => s + (b[0] - world.x0[i][0]) ** 2 + (b[1] - world.x0[i][1]) ** 2, 0) / targets.length;
  const cross = useMemo(
    () => ({ random: crossings(world.x0.slice(0, 90), world.x1.slice(0, 90)), ot: crossings(world.x0.slice(0, 90), world.matched.slice(0, 90)) }),
    [world],
  );
  const bend = straightness(reference.slice(0, TRACKS));

  const grid = useMemo(() => {
    const out: { at: Vec; v: Vec }[] = [];
    const n = 13;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const at: Vec = [DOMAIN[0] + ((i + 0.5) * (DOMAIN[1] - DOMAIN[0])) / n, DOMAIN[0] + ((j + 0.5) * (DOMAIN[1] - DOMAIN[0])) / n];
        out.push({ at, v: field(at, Math.min(fieldTime, 0.999)) });
      }
    }
    return out;
  }, [field, fieldTime]);

  const lowestError = Math.min(...Object.values(convergence).flatMap((c) => c.map((p) => p[1])));

  return (
    <DemoFrame
      title="Straight paths from noise to data"
      caption={caption}
      controls={
        <>
          <Choice
            label="Target"
            value={target}
            options={[
              { value: "moons", label: "Two moons" },
              { value: "eight", label: "Eight Gaussians" },
            ]}
            onChange={setTarget}
          />
          <Choice
            label="Solver"
            value={solver}
            options={[
              { value: "euler", label: "Euler" },
              { value: "midpoint", label: "Midpoint" },
              { value: "rk4", label: "RK4" },
            ]}
            onChange={setSolver}
          />
          <Slider
            label="Steps"
            value={stepIndex}
            min={0}
            max={STEP_CHOICES.length - 1}
            format={(i) => `${STEP_CHOICES[i]}, ${STEP_CHOICES[i] * NFE[solver]} NFE`}
            onChange={setStepIndex}
          />
          <Slider
            label="Velocity field at t"
            value={fieldTime}
            min={0}
            max={1}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={setFieldTime}
          />
          <Toggle label="Show training pairs, the conditional straight paths" checked={showPairs} onChange={setShowPairs} />
          <Toggle label="Pair noise with data by minibatch OT" checked={ot} onChange={setOt} />
        </>
      }
    >
      <p className="mb-1 text-[12px] font-medium text-ink-3">
        t = 0 is noise and t = 1 is data (the Lipman convention); arrows are the velocity field at t = {fieldTime.toFixed(2)}
      </p>
      <div className="mx-auto max-w-[520px]">
        <Plot width={400} height={400} xDomain={DOMAIN} yDomain={DOMAIN} xTicks={4} yTicks={4} tickFormat={(v) => v.toFixed(1)}>
          {() => (
            <>
              <Points points={world.x1} r={1.6} fill="var(--ink-4)" stroke="none" opacity={0.6} />
              {showPairs && <Segments from={world.x0.slice(0, pairs.length)} to={pairs} colour="var(--series-4)" />}
              <Arrows grid={grid} />
              {reference.slice(0, TRACKS).map((path, i) => (
                <Curve key={`r${i}`} points={path} stroke="var(--ink-3)" width={0.8} dashed opacity={0.5} />
              ))}
              {paths.slice(0, TRACKS).map((path, i) => (
                <Curve key={i} points={path} stroke="var(--series-2)" width={1.3} opacity={0.85} />
              ))}
              <Points points={paths.map((p) => p[p.length - 1])} r={2.1} fill="var(--demo)" stroke="none" />
            </>
          )}
        </Plot>
      </div>
      <Legend
        items={[
          { label: `${solver === "rk4" ? "RK4" : solver === "midpoint" ? "midpoint" : "Euler"}, ${steps} ${steps === 1 ? "step" : "steps"}`, colour: "var(--series-2)" },
          { label: "exact path", colour: "var(--ink-3)", dashed: true },
          { label: "samples at t = 1", colour: "var(--demo)" },
          { label: "data", colour: "var(--ink-4)" },
          ...(showPairs ? [{ label: ot ? "OT pairs" : "random pairs", colour: "var(--series-4)" }] : []),
        ]}
      />
      {ot && (
        <p className="mt-1 text-[12px] leading-snug text-ink-3">
          With OT pairs there is no closed form, so the field is a kernel regression of {PAIRS} pairs' velocities.
        </p>
      )}

      <div className="mt-4">
        <p className="mb-1 text-[12px] font-medium text-ink-3">
          Endpoint error against network evaluations, on the exact field, both axes logarithmic
        </p>
        <Plot
          height={160}
          xDomain={[0, Math.log10(160)]}
          yDomain={[Math.floor(lowestError), 1]}
          xLabel="network evaluations"
          yLabel="error"
          xTicks={0}
          yTicks={1 - Math.floor(lowestError)}
          tickFormat={(v) => (v === 0 ? "1" : `1e${v.toFixed(0)}`)}
        >
          {({ sx, area }) => (
            <>
              {[1, 2, 4, 8, 16, 32, 64, 128].map((n) => (
                <text
                  key={n}
                  x={sx(Math.log10(n))}
                  y={area.bottom + 15}
                  textAnchor="middle"
                  fontSize={10}
                  fontFamily="var(--font-mono)"
                  fill="var(--ink-3)"
                >
                  {n}
                </text>
              ))}
              <Curve points={convergence.euler} stroke="var(--series-2)" width={solver === "euler" ? 2.2 : 1.2} />
              <Curve points={convergence.midpoint} stroke="var(--series-3)" width={solver === "midpoint" ? 2.2 : 1.2} />
              <Curve points={convergence.rk4} stroke="var(--series-4)" width={solver === "rk4" ? 2.2 : 1.2} />
              <Points points={[...convergence.euler, ...convergence.midpoint, ...convergence.rk4]} r={2} fill="var(--ink-3)" stroke="none" />
            </>
          )}
        </Plot>
        <Legend
          items={[
            { label: "Euler, order 1", colour: "var(--series-2)" },
            { label: "midpoint, order 2", colour: "var(--series-3)" },
            { label: "RK4, order 4", colour: "var(--series-4)" },
          ]}
        />
      </div>

      <Readouts>
        <Readout label="Evaluations" value={steps * NFE[solver]} hint={`${steps} × ${NFE[solver]} per step`} />
        <Readout label="Endpoint error" value={error.toFixed(3)} hint="mean distance to the exact end" tone="demo" />
        <Readout label="Path bend" value={bend.toFixed(3)} hint="length over straight distance; 1 is straight" tone={bend < 1.02 ? "good" : "plain"} />
        <Readout
          label={ot ? "OT pairs" : "Random pairs"}
          value={`${(ot ? cross.ot : cross.random).toLocaleString("en-US")} crossings`}
          hint={`cost ${pairCost(ot ? world.matched : world.x1).toFixed(2)}, the other ${pairCost(ot ? world.x1 : world.matched).toFixed(2)}`}
        />
      </Readouts>
    </DemoFrame>
  );
}

function Segments({ from, to, colour }: { from: Vec[]; to: Vec[]; colour: string }) {
  const { sx, sy } = useScales();
  const d = from.map((a, i) => `M${sx(a[0]).toFixed(1)},${sy(a[1]).toFixed(1)}L${sx(to[i][0]).toFixed(1)},${sy(to[i][1]).toFixed(1)}`).join("");
  return <path d={d} stroke={colour} strokeWidth={0.8} opacity={0.55} fill="none" />;
}

function Arrows({ grid }: { grid: { at: Vec; v: Vec }[] }) {
  const { sx, sy } = useScales();
  const longest = Math.max(1e-9, ...grid.map(({ v }) => Math.hypot(v[0], v[1])));
  const cell = Math.abs(sx(DOMAIN[0] + (DOMAIN[1] - DOMAIN[0]) / 13) - sx(DOMAIN[0]));
  return (
    <g stroke="var(--ink-3)" strokeWidth={0.9} opacity={0.7} fill="none">
      {grid.map(({ at, v }, i) => {
        const scale = (0.85 * cell) / longest;
        const x1 = sx(at[0]);
        const y1 = sy(at[1]);
        const dx = v[0] * scale;
        const dy = -v[1] * scale;
        const x2 = x1 + dx;
        const y2 = y1 + dy;
        const len = Math.hypot(dx, dy);
        if (len < 0.8) return null;
        const ux = dx / len;
        const uy = dy / len;
        const head = Math.min(3, len * 0.4);
        return (
          <path
            key={i}
            d={`M${x1.toFixed(1)},${y1.toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}M${(x2 - head * (ux - uy * 0.6)).toFixed(1)},${(y2 - head * (uy + ux * 0.6)).toFixed(1)}L${x2.toFixed(1)},${y2.toFixed(1)}L${(x2 - head * (ux + uy * 0.6)).toFixed(1)},${(y2 - head * (uy - ux * 0.6)).toFixed(1)}`}
          />
        );
      })}
    </g>
  );
}
