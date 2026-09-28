import { useMemo, useState } from "react";

import { gaussian, seededRandom } from "./dsp";
import { tintImage, token, useDebounced, useThemeVersion } from "./figure";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Bars,
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Readout,
  Readouts,
  Rule,
  Slider,
} from "./kit";

const T = 1000;
const PARTICLES = 800;
const TRACKS = 28;
const X: [number, number] = [-4, 4];
const STEP_CHOICES = [1, 2, 5, 10, 20, 50, 100, 250, 1000];

/**
 * The data: a mixture of three Gaussians. Its noised marginals are Gaussian
 * mixtures too, so the score at every t is exact, and the reverse process
 * needs no network. Convention: DDPM's, with the data at t = 0 and noise at
 * t = T (Ho et al., 2020).
 */
const DATA = [
  { weight: 0.3, mean: -2, sd: 0.35 },
  { weight: 0.5, mean: 0.5, sd: 0.25 },
  { weight: 0.2, mean: 2.2, sd: 0.3 },
];

type Schedule = "linear" | "cosine";

/** ᾱ_t for t = 0..T, with ᾱ_0 = 1. */
function alphaBar(schedule: Schedule): Float64Array {
  const out = new Float64Array(T + 1);
  out[0] = 1;
  if (schedule === "linear") {
    // DDPM: β linear from 1e-4 to 0.02 over T = 1000 steps.
    for (let t = 1; t <= T; t += 1) {
      const beta = 1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1);
      out[t] = out[t - 1] * (1 - beta);
    }
  } else {
    // Nichol and Dhariwal (2021), s = 0.008, with β clipped at 0.999.
    const f = (t: number) => Math.cos(((t / T + 0.008) / 1.008) * (Math.PI / 2)) ** 2;
    for (let t = 1; t <= T; t += 1) {
      const beta = Math.min(1 - f(t) / f(t - 1), 0.999);
      out[t] = out[t - 1] * (1 - beta);
    }
  }
  return out;
}

/** The marginal q(x_t) = sum_k w_k N(sqrt(ᾱ) mu_k, ᾱ s_k^2 + 1 - ᾱ). */
function components(ab: number) {
  return DATA.map((c) => ({ weight: c.weight, mean: Math.sqrt(ab) * c.mean, variance: ab * c.sd ** 2 + 1 - ab }));
}

function density(x: number, ab: number): number {
  return components(ab).reduce(
    (sum, c) => sum + (c.weight * Math.exp(-((x - c.mean) ** 2) / (2 * c.variance))) / Math.sqrt(2 * Math.PI * c.variance),
    0,
  );
}

type Part = { mean: number; variance: number; offset: number };

/** The noised mixture's components at ᾱ, with each one's log weight and normalizer folded in. */
function parts(ab: number): Part[] {
  return components(ab).map((c) => ({
    mean: c.mean,
    variance: c.variance,
    offset: Math.log(c.weight) - 0.5 * Math.log(c.variance),
  }));
}

/** The exact score, the gradient of log q(x_t), as a responsibility-weighted sum. */
function score(x: number, mixture: Part[]): number {
  let top = -Infinity;
  for (const c of mixture) top = Math.max(top, c.offset - (x - c.mean) ** 2 / (2 * c.variance));
  let total = 0;
  let weighted = 0;
  for (const c of mixture) {
    const r = Math.exp(c.offset - (x - c.mean) ** 2 / (2 * c.variance) - top);
    total += r;
    weighted += (r * -(x - c.mean)) / c.variance;
  }
  return weighted / total;
}

/**
 * Sampling from x_T ~ N(0, 1) in `steps` steps with the DDIM update, where
 * eta = 1 is DDPM's ancestral sampling and eta = 0 is deterministic DDIM
 * (Song, Meng, and Ermon, 2021). The noise prediction is the exact one,
 * eps = -sqrt(1 - ᾱ_t) * score.
 */
function sample(ab: Float64Array, steps: number, eta: number, seed: number) {
  const random = seededRandom(seed);
  const start = Array.from({ length: PARTICLES }, () => gaussian(random));
  const zs = seededRandom(seed + 99);
  const times = Array.from({ length: steps + 1 }, (_, j) => Math.round((T * (steps - j)) / steps));
  const x = start.slice();
  const every = Math.max(1, Math.floor(steps / 120));
  const tracks: [number, number][][] = Array.from({ length: TRACKS }, (_, i) => [[T, x[i]]]);
  for (let j = 0; j < steps; j += 1) {
    const t = times[j];
    const next = times[j + 1];
    const a = ab[t];
    const b = ab[next];
    const sigma = eta * Math.sqrt((1 - b) / (1 - a)) * Math.sqrt(Math.max(0, 1 - a / b));
    const mixture = parts(a);
    for (let i = 0; i < PARTICLES; i += 1) {
      const eps = -Math.sqrt(1 - a) * score(x[i], mixture);
      const x0 = (x[i] - Math.sqrt(1 - a) * eps) / Math.sqrt(a);
      const noise = sigma > 0 ? gaussian(zs) : 0;
      x[i] = Math.sqrt(b) * x0 + Math.sqrt(Math.max(0, 1 - b - sigma * sigma)) * eps + sigma * noise;
    }
    if ((j + 1) % every === 0 || j === steps - 1) for (let i = 0; i < TRACKS; i += 1) tracks[i].push([next, x[i]]);
  }
  return { samples: x, tracks };
}

/** The data's quantile function, by inverting its CDF on a fine grid. */
const QUANTILES = (() => {
  const grid = 4000;
  const xs = Array.from({ length: grid }, (_, i) => -5 + (10 * i) / (grid - 1));
  const cdf: number[] = [];
  let run = 0;
  xs.forEach((x) => {
    run += density(x, 1) * (10 / (grid - 1));
    cdf.push(run);
  });
  return (p: number) => {
    const target = p * run;
    let lo = 0;
    let hi = grid - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return xs[lo];
  };
})();

/** The 1-Wasserstein distance from the samples to the data, in the units of x. */
function distanceToData(samples: number[]): number {
  const sorted = samples.slice().sort((a, b) => a - b);
  return sorted.reduce((sum, v, i) => sum + Math.abs(v - QUANTILES((i + 0.5) / sorted.length)), 0) / sorted.length;
}

function histogram(values: number[], bins = 60) {
  const width = (X[1] - X[0]) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const b = Math.floor((v - X[0]) / width);
    if (b >= 0 && b < bins) counts[b] += 1;
  }
  return counts.map((c, b) => ({ x0: X[0] + b * width, x1: X[0] + (b + 1) * width, y: c / (values.length * width) }));
}

const curvePoints = (ab: number): [number, number][] =>
  Array.from({ length: 241 }, (_, i) => {
    const x = X[0] + ((X[1] - X[0]) * i) / 240;
    return [x, density(x, ab)];
  });

export default function Diffusion1d({ caption }: DemoProps) {
  const [schedule, setSchedule] = useState<Schedule>("linear");
  const [t, setT] = useState(300);
  const [sampler, setSampler] = useState<"ddpm" | "ddim">("ddim");
  const [stepIndex, setStepIndex] = useState(3);
  const [seed, setSeed] = useState(1);
  const theme = useThemeVersion();
  const steps = STEP_CHOICES[stepIndex];
  // A thousand steps for two samplers takes a moment, so wait for the slider to settle.
  const settledSteps = useDebounced(steps, 150);

  const ab = useMemo(() => alphaBar(schedule), [schedule]);
  const run = useMemo(() => sample(ab, settledSteps, sampler === "ddpm" ? 1 : 0, seed), [ab, settledSteps, sampler, seed]);
  const other = useMemo(() => sample(ab, settledSteps, sampler === "ddpm" ? 0 : 1, seed), [ab, settledSteps, sampler, seed]);
  const distance = distanceToData(run.samples);
  const otherDistance = distanceToData(other.samples);

  // Forward samples at t, from data samples and fixed noise.
  const forward = useMemo(() => {
    const random = seededRandom(seed + 7);
    const a = ab[t];
    return Array.from({ length: PARTICLES }, () => {
      let u = random();
      const c = DATA.find((d) => (u -= d.weight) < 0) ?? DATA[DATA.length - 1];
      const x0 = c.mean + c.sd * gaussian(random);
      return Math.sqrt(a) * x0 + Math.sqrt(1 - a) * gaussian(random);
    });
  }, [ab, t, seed]);

  // q(x_t) over (t, x), each column scaled to its own peak, as a tinted image.
  const background = useMemo(() => {
    const cols = 125;
    const rows = 100;
    const values = new Float32Array(rows * cols);
    for (let c = 0; c < cols; c += 1) {
      const a = ab[Math.round(((c + 0.5) / cols) * T)];
      const column = Array.from({ length: rows }, (_, r) => density(X[1] - ((r + 0.5) / rows) * (X[1] - X[0]), a));
      const peak = Math.max(...column);
      column.forEach((v, r) => (values[r * cols + c] = v / peak));
    }
    return tintImage(values, rows, cols, token("--demo"), 0.6);
    // `theme` is here so the tint follows the theme.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ab, theme]);

  const a = ab[t];
  const snr = a / (1 - a);
  const snrDb = t === 0 ? Infinity : 10 * Math.log10(snr);

  // A short tone, noised at this t with one fixed draw of noise.
  const tone = useMemo(() => {
    const random = seededRandom(42);
    return Array.from({ length: 96 }, (_, i) => {
      const clean = Math.sin((2 * Math.PI * 3 * i) / 96);
      return { i, clean, noisy: Math.sqrt(a) * clean + Math.sqrt(1 - a) * gaussian(random) };
    });
  }, [a]);

  const both = { linear: alphaBar("linear"), cosine: alphaBar("cosine") };
  const scheduleCurve = (s: Schedule): [number, number][] =>
    Array.from({ length: 101 }, (_, i) => [i * 10, both[s][i * 10]]);

  return (
    <DemoFrame
      title="Noising a signal and walking it back"
      caption={caption}
      controls={
        <>
          <Choice
            label="Noise schedule"
            value={schedule}
            options={[
              { value: "linear", label: "Linear" },
              { value: "cosine", label: "Cosine" },
            ]}
            onChange={setSchedule}
          />
          <Slider label="Time step, t" value={t} min={0} max={T} step={10} format={(v) => `${v} of ${T}`} onChange={setT} />
          <Choice
            label="Sampler"
            value={sampler}
            options={[
              { value: "ddpm", label: "DDPM, stochastic" },
              { value: "ddim", label: "DDIM, deterministic" },
            ]}
            onChange={setSampler}
          />
          <Slider
            label="Sampling steps"
            value={stepIndex}
            min={0}
            max={STEP_CHOICES.length - 1}
            format={(i) => String(STEP_CHOICES[i])}
            onChange={setStepIndex}
          />
          <Actions>
            <Action tone="quiet" onClick={() => setSeed((s) => s + 1)}>
              New noise
            </Action>
          </Actions>
        </>
      }
    >
      <p className="mb-1 text-[12px] font-medium text-ink-3">
        Sampling paths from noise at t = {T} back to data at t = 0, over q(x_t) shaded behind
      </p>
      <Plot
        height={230}
        xDomain={[0, T]}
        yDomain={X}
        xLabel="t   (DDPM convention: t = 0 is data, t = T is noise)"
        yLabel="x"
        xTicks={5}
        yTicks={4}
        tickFormat={(v) => v.toFixed(0)}
      >
        {({ sx, sy }) => (
          <>
            {background && (
              <image
                href={background}
                x={sx(0)}
                y={sy(X[1])}
                width={sx(T) - sx(0)}
                height={sy(X[0]) - sy(X[1])}
                preserveAspectRatio="none"
              />
            )}
            {run.tracks.map((track, i) => (
              <Curve key={i} points={track} stroke="var(--series-2)" width={1} opacity={0.75} />
            ))}
            <Rule x={t} label={`t = ${t}`} colour="var(--ink-2)" />
          </>
        )}
      </Plot>

      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">Forward: q(x_t) at t = {t}, and noised samples</p>
          <Plot width={300} height={170} xDomain={X} yDomain={[0, 1.3]} yTicks={2} xTicks={4} tickFormat={(v) => v.toFixed(v % 1 ? 1 : 0)}>
            {() => (
              <>
                <Bars bins={histogram(forward)} gap={0.5} />
                <Curve points={curvePoints(a)} stroke="var(--ink-2)" width={1.4} />
              </>
            )}
          </Plot>
        </div>
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">
            Reverse: {settledSteps} {sampler.toUpperCase()} {settledSteps === 1 ? "step" : "steps"} against the data
          </p>
          <Plot width={300} height={170} xDomain={X} yDomain={[0, 1.3]} yTicks={2} xTicks={4} tickFormat={(v) => v.toFixed(v % 1 ? 1 : 0)}>
            {() => (
              <>
                <Bars
                  bins={histogram(run.samples)}
                  gap={0.5}
                  fill="color-mix(in oklab, var(--series-2) 50%, transparent)"
                  stroke="var(--series-2)"
                />
                <Curve points={curvePoints(1)} stroke="var(--ink-2)" width={1.4} />
              </>
            )}
          </Plot>
        </div>
      </div>
      <Legend
        items={[
          { label: "exact density", colour: "var(--ink-2)" },
          { label: "forward samples", colour: "var(--demo)" },
          { label: "generated samples", colour: "var(--series-2)" },
        ]}
      />

      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">ᾱ_t for both schedules</p>
          <Plot width={300} height={140} xDomain={[0, T]} yDomain={[0, 1]} yTicks={2} xTicks={4} tickFormat={(v, axis) => (axis === "x" ? v.toFixed(0) : v.toFixed(1))}>
            {() => (
              <>
                <Curve points={scheduleCurve("linear")} stroke="var(--demo)" width={schedule === "linear" ? 2 : 1} dashed={schedule !== "linear"} />
                <Curve points={scheduleCurve("cosine")} stroke="var(--series-4)" width={schedule === "cosine" ? 2 : 1} dashed={schedule !== "cosine"} />
                <Rule x={t} colour="var(--ink-2)" />
              </>
            )}
          </Plot>
          <Legend
            items={[
              { label: "linear", colour: "var(--demo)", dashed: schedule !== "linear" },
              { label: "cosine", colour: "var(--series-4)", dashed: schedule !== "cosine" },
            ]}
          />
        </div>
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">A tone, noised to t = {t}</p>
          <Plot width={300} height={140} xDomain={[0, 95]} yDomain={[-3, 3]} yTicks={2} xTicks={0} tickFormat={(v) => v.toFixed(0)}>
            {() => (
              <>
                <Curve points={tone.map((p) => [p.i, p.clean])} stroke="var(--ink-4)" width={1} />
                <Curve points={tone.map((p) => [p.i, p.noisy])} stroke="var(--demo)" width={1.3} />
              </>
            )}
          </Plot>
          <p className="mt-1 text-[12px] text-ink-3">x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε</p>
        </div>
      </div>

      <Readouts>
        <Readout label="ᾱ_t" value={a >= 0.001 ? a.toFixed(3) : a.toExponential(2)} hint={`√ᾱ_t = ${Math.sqrt(a).toFixed(3)}`} />
        <Readout label="SNR" value={Number.isFinite(snrDb) ? `${snrDb.toFixed(1)} dB` : "∞"} hint="ᾱ_t / (1 − ᾱ_t)" />
        <Readout
          label={`${sampler.toUpperCase()}, ${settledSteps} ${settledSteps === 1 ? "step" : "steps"}`}
          value={distance.toFixed(3)}
          hint="W1 distance to the data"
          tone="demo"
        />
        <Readout
          label={`${sampler === "ddpm" ? "DDIM" : "DDPM"}, same steps`}
          value={otherDistance.toFixed(3)}
          hint="for comparison"
        />
      </Readouts>
    </DemoFrame>
  );
}
