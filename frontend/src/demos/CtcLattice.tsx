import { useMemo, useState } from "react";

import { seededRandom, gaussian } from "./dsp";
import { CanvasFigure, Posteriorgram, TextField, type Paint } from "./figure";
import type { DemoProps } from "./index";
import { Choice, DemoFrame, Legend, Readout, Readouts, Slider } from "./kit";

const BLANK = "∅";
const DEFAULT_LABEL = "bee";
const MAX_LETTERS = 5;

/**
 * A synthetic posteriorgram: one spike per letter of `label`, spread evenly
 * over `frames`, on a floor of blank. `peakiness` runs from 0, where each
 * letter smears over several frames, to 1, where it is a one-frame spike and
 * blank takes the rest, as a trained CTC model's output looks.
 */
function posteriorgram(label: string, frames: number, peakiness: number) {
  const letters = [...new Set(label)];
  // One letter the label does not use, so the model has something to confuse.
  const distractor = [..."aeiostnrl"].find((c) => !letters.includes(c)) ?? "z";
  const vocab = [BLANK, ...letters, distractor];
  const random = seededRandom(1234 + label.length * 31 + label.charCodeAt(0));
  const spike = 2.5 + 4 * peakiness;
  const width = Math.max(0.5, 1.4 - 0.9 * peakiness);
  const floor = 1.5 + 2 * peakiness;
  const jitter = 0.35 * (1 - 0.5 * peakiness);
  const centres = [...label].map((_, i) => ((i + 0.5) * frames) / label.length - 0.5);

  const probs: number[][] = [];
  for (let t = 0; t < frames; t += 1) {
    const logits = vocab.map((symbol, v) => {
      let z: number;
      if (v === 0) z = floor;
      else if (symbol === distractor) z = 0.4;
      else {
        z = -1;
        [...label].forEach((c, i) => {
          if (c === symbol) z = Math.max(z, spike * Math.exp(-((t - centres[i]) ** 2) / (2 * width * width)));
        });
      }
      return z + jitter * gaussian(random);
    });
    const top = Math.max(...logits);
    const exps = logits.map((z) => Math.exp(z - top));
    const sum = exps.reduce((a, b) => a + b, 0);
    probs.push(exps.map((e) => e / sum));
  }
  return { vocab, probs };
}

type Lattice = {
  /** The blank-extended label, as vocabulary indices and as symbols. */
  ext: number[];
  symbols: string[];
  allowSkip: boolean[];
  /** `alpha[t][s]`, which includes the emission at frame t. */
  alpha: number[][];
  /** `beta[t][s]`, the probability of frames t+1..T-1 from state s; it excludes frame t. */
  beta: number[][];
  total: number;
  paths: number;
  best: { states: number[]; probability: number };
};

/** The CTC forward and backward recursions over the 2L + 1 extended states. */
function ctcLattice(label: string, vocab: string[], probs: number[][]): Lattice {
  const ext: number[] = [0];
  for (const c of label) ext.push(vocab.indexOf(c), 0);
  const S = ext.length;
  const T = probs.length;
  const symbols = ext.map((v) => vocab[v]);
  const allowSkip = ext.map((v, s) => s >= 2 && v !== 0 && v !== ext[s - 2]);

  const forward = (emit: (t: number, s: number) => number, combine: "sum" | "max") => {
    const table = Array.from({ length: T }, () => new Array<number>(S).fill(0));
    const from = Array.from({ length: T }, () => new Array<number>(S).fill(-1));
    table[0][0] = emit(0, 0);
    if (S > 1) table[0][1] = emit(0, 1);
    for (let t = 1; t < T; t += 1) {
      for (let s = 0; s < S; s += 1) {
        const candidates: [number, number][] = [[table[t - 1][s], s]];
        if (s >= 1) candidates.push([table[t - 1][s - 1], s - 1]);
        if (allowSkip[s]) candidates.push([table[t - 1][s - 2], s - 2]);
        if (combine === "sum") {
          table[t][s] = candidates.reduce((sum, [v]) => sum + v, 0) * emit(t, s);
        } else {
          const [value, at] = candidates.reduce((a, b) => (b[0] > a[0] ? b : a));
          table[t][s] = value * emit(t, s);
          from[t][s] = at;
        }
      }
    }
    return { table, from };
  };

  const alpha = forward((t, s) => probs[t][ext[s]], "sum").table;
  const counts = forward(() => 1, "sum").table;

  const beta = Array.from({ length: T }, () => new Array<number>(S).fill(0));
  beta[T - 1][S - 1] = 1;
  if (S > 1) beta[T - 1][S - 2] = 1;
  for (let t = T - 2; t >= 0; t -= 1) {
    for (let s = 0; s < S; s += 1) {
      let sum = beta[t + 1][s] * probs[t + 1][ext[s]];
      if (s + 1 < S) sum += beta[t + 1][s + 1] * probs[t + 1][ext[s + 1]];
      if (s + 2 < S && allowSkip[s + 2]) sum += beta[t + 1][s + 2] * probs[t + 1][ext[s + 2]];
      beta[t][s] = sum;
    }
  }

  const ends = S > 1 ? [S - 1, S - 2] : [S - 1];
  const total = ends.reduce((sum, s) => sum + alpha[T - 1][s], 0);
  const paths = ends.reduce((sum, s) => sum + counts[T - 1][s], 0);

  // The single most probable alignment, by the same recursion with max for sum.
  const viterbi = forward((t, s) => probs[t][ext[s]], "max");
  let end = ends[0];
  for (const s of ends) if (viterbi.table[T - 1][s] > viterbi.table[T - 1][end]) end = s;
  const states = new Array<number>(T);
  states[T - 1] = end;
  for (let t = T - 1; t > 0; t -= 1) states[t - 1] = viterbi.from[t][states[t]];
  const probability = viterbi.table[T - 1][end];

  return { ext, symbols, allowSkip, alpha, beta, total, paths, best: { states, probability } };
}

/** The collapse function B: merge repeated symbols, then drop blanks. */
function collapseSteps(path: string[]) {
  const merged = path.filter((c, i) => i === 0 || c !== path[i - 1]);
  const dropped = merged.filter((c) => c !== BLANK);
  return { merged, dropped };
}

function compact(value: number): string {
  if (value === 0) return "0";
  if (value >= 0.01) return value.toFixed(2).replace(/^0/, "");
  const exponent = Math.floor(Math.log10(value));
  return `${(value / 10 ** exponent).toFixed(1)}e${exponent}`;
}

function scientific(value: number): string {
  if (value === 0) return "0";
  if (value >= 0.001) return value.toPrecision(3);
  return value.toExponential(2);
}

const LEFT = 30;
const TOP = 16;
const RIGHT = 4;

function layout(width: number, frames: number, states: number) {
  const cellW = Math.max(12, (width - LEFT - RIGHT) / frames);
  const cellH = Math.max(20, Math.min(32, cellW * 0.7));
  return { cellW, cellH, height: TOP + states * cellH + 6 };
}

export default function CtcLattice({ caption }: DemoProps) {
  const [draft, setDraft] = useState(DEFAULT_LABEL);
  const [frames, setFrames] = useState(10);
  const [peakiness, setPeakiness] = useState(0.5);
  const [shade, setShade] = useState<"alpha" | "occupancy">("alpha");
  const [picked, setPicked] = useState<{ t: number; s: number } | null>(null);

  const label = draft.toLowerCase().replace(/[^a-z]/g, "").slice(0, MAX_LETTERS) || DEFAULT_LABEL;
  const repeats = [...label].filter((c, i) => i > 0 && c === label[i - 1]).length;

  const { vocab, probs } = useMemo(() => posteriorgram(label, frames, peakiness), [label, frames, peakiness]);
  const lattice = useMemo(() => ctcLattice(label, vocab, probs), [label, vocab, probs]);
  const S = lattice.ext.length;
  const feasible = lattice.total > 0;

  // By default, the second copy of a repeated letter, where the skip rule bites.
  const defaultPick = useMemo(() => {
    const repeated = lattice.ext.findIndex((v, s) => s >= 2 && v !== 0 && v === lattice.ext[s - 2]);
    const s = repeated >= 0 ? repeated : Math.min(S - 2, 3);
    let t = frames - 3;
    while (t > 0 && lattice.alpha[t][s] * lattice.beta[t][s] === 0) t -= 1;
    return { t: Math.max(1, t), s };
  }, [lattice, S, frames]);
  const pick = picked && picked.t < frames && picked.s < S ? picked : defaultPick;

  const greedy = probs.map((row) => vocab[row.indexOf(Math.max(...row))]);
  const bestPath = feasible ? lattice.best.states.map((s) => lattice.symbols[s]) : ["(no path)"];
  const bestCollapse = collapseSteps(bestPath);
  const greedyCollapse = collapseSteps(greedy);
  const greedyText = greedyCollapse.dropped.join("");

  const live = (t: number, s: number) => lattice.alpha[t][s] * lattice.beta[t][s] > 0;

  function draw({ ctx, width, colour }: Paint) {
    const { cellW, cellH } = layout(width, frames, S);
    const x = (t: number) => LEFT + (t + 0.5) * cellW;
    const y = (s: number) => TOP + (s + 0.5) * cellH;
    const demo = colour("--demo");
    const path = colour("--series-2");
    const skip = colour("--series-4");

    ctx.font = `10px "JetBrains Mono Variable", ui-monospace, monospace`;
    ctx.fillStyle = colour("--ink-3");
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const every = cellW < 18 ? 2 : 1;
    for (let t = 0; t < frames; t += every) ctx.fillText(String(t + 1), x(t), TOP / 2);
    ctx.textAlign = "right";
    lattice.symbols.forEach((symbol, s) => {
      ctx.fillStyle = symbol === BLANK ? colour("--ink-4") : colour("--ink");
      ctx.fillText(symbol, LEFT - 8, y(s));
    });

    // The cells, shaded by their share of the frame's total.
    for (let t = 0; t < frames; t += 1) {
      const column = lattice.alpha[t].map((a, s) => (shade === "alpha" ? a : a * lattice.beta[t][s]));
      const sum = column.reduce((a, b) => a + b, 0) || 1;
      for (let s = 0; s < S; s += 1) {
        const share = column[s] / sum;
        const left = LEFT + t * cellW + 1.5;
        const top = TOP + s * cellH + 1.5;
        ctx.beginPath();
        ctx.roundRect(left, top, cellW - 3, cellH - 3, 3);
        if (live(t, s)) {
          ctx.globalAlpha = 0.1 + 0.75 * share;
          ctx.fillStyle = demo;
          ctx.fill();
          ctx.globalAlpha = 1;
        } else {
          ctx.strokeStyle = colour("--rule");
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        if (live(t, s) && cellW >= 40) {
          ctx.fillStyle = share > 0.55 ? colour("--raise") : colour("--ink-2");
          ctx.textAlign = "center";
          ctx.font = `9.5px "JetBrains Mono Variable", ui-monospace, monospace`;
          ctx.fillText(compact(lattice.alpha[t][s]), x(t), y(s) + 0.5);
        }
      }
    }

    // Every transition some complete path uses: stay, advance, or skip.
    const edge = (t: number, from: number, to: number, stroke: string, lineWidth: number, dashed = false) => {
      ctx.beginPath();
      ctx.moveTo(x(t - 1) + cellW * 0.22, y(from));
      ctx.lineTo(x(t) - cellW * 0.22, y(to));
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dashed ? [3, 3] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    for (let t = 1; t < frames; t += 1) {
      for (let s = 0; s < S; s += 1) {
        if (lattice.beta[t][s] === 0 || lattice.alpha[t][s] === 0) continue;
        const sources = [s, s - 1, ...(lattice.allowSkip[s] ? [s - 2] : [])];
        for (const from of sources) {
          if (from < 0 || lattice.alpha[t - 1][from] === 0) continue;
          const isSkip = from === s - 2;
          ctx.globalAlpha = 0.55;
          edge(t, from, s, isSkip ? skip : colour("--ink-4"), 0.9, isSkip);
          ctx.globalAlpha = 1;
        }
      }
    }

    // The best path.
    ctx.beginPath();
    lattice.best.states.forEach((s, t) => (t === 0 ? ctx.moveTo(x(t), y(s)) : ctx.lineTo(x(t), y(s))));
    if (feasible) {
      ctx.strokeStyle = path;
      ctx.lineWidth = 2.4;
      ctx.lineJoin = "round";
      ctx.stroke();
      lattice.best.states.forEach((s, t) => {
        ctx.beginPath();
        ctx.arc(x(t), y(s), 3.2, 0, 2 * Math.PI);
        ctx.fillStyle = path;
        ctx.fill();
      });
    }

    // The picked cell and the transitions into it, the forbidden skip crossed out.
    const { t, s } = pick;
    ctx.beginPath();
    ctx.roundRect(LEFT + t * cellW + 0.5, TOP + s * cellH + 0.5, cellW - 1, cellH - 1, 4);
    ctx.strokeStyle = colour("--ink");
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (t > 0) {
      edge(t, s, s, colour("--ink"), 1.8);
      if (s >= 1) edge(t, s - 1, s, colour("--ink"), 1.8);
      if (s >= 2) {
        if (lattice.allowSkip[s]) edge(t, s - 2, s, skip, 2.2, true);
        else {
          edge(t, s - 2, s, colour("--bad"), 1.4, true);
          const mx = (x(t - 1) + x(t)) / 2;
          const my = (y(s - 2) + y(s)) / 2;
          ctx.strokeStyle = colour("--bad");
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(mx - 4, my - 4);
          ctx.lineTo(mx + 4, my + 4);
          ctx.moveTo(mx + 4, my - 4);
          ctx.lineTo(mx - 4, my + 4);
          ctx.stroke();
        }
      }
    }
  }

  function onPointer(px: number, py: number, width: number, type: "down" | "move" | "leave") {
    if (type !== "down") return;
    const { cellW, cellH } = layout(width, frames, S);
    const t = Math.floor((px - LEFT) / cellW);
    const s = Math.floor((py - TOP) / cellH);
    if (t >= 0 && t < frames && s >= 0 && s < S) setPicked({ t, s });
  }

  // The recursion for the picked cell, written out with its numbers.
  const { t: pt, s: ps } = pick;
  const name = (s: number) => lattice.symbols[s];
  const terms: string[] = [];
  if (pt > 0) {
    terms.push(`α${pt}(${name(ps)}) ${scientific(lattice.alpha[pt - 1][ps])}`);
    if (ps >= 1) terms.push(`α${pt}(${name(ps - 1)}) ${scientific(lattice.alpha[pt - 1][ps - 1])}`);
    if (lattice.allowSkip[ps]) terms.push(`α${pt}(${name(ps - 2)}) ${scientific(lattice.alpha[pt - 1][ps - 2])}, the skip`);
  }
  const why =
    ps >= 2 && !lattice.allowSkip[ps]
      ? lattice.symbols[ps] === BLANK
        ? "A blank state has no skip."
        : `No skip from the previous "${lattice.symbols[ps]}": the two copies would merge.`
      : "";

  return (
    <DemoFrame
      title="Every alignment CTC adds up"
      caption={caption}
      controls={
        <>
          <TextField
            label="Label"
            value={draft}
            maxLength={MAX_LETTERS}
            hint="up to five letters"
            onChange={(value) => {
              setDraft(value);
              setPicked(null);
            }}
          />
          <Slider label="Frames, T" value={frames} min={4} max={14} onChange={(v) => { setFrames(v); setPicked(null); }} />
          <Slider
            label="Peakiness"
            value={peakiness}
            min={0}
            max={1}
            step={0.05}
            format={(v) => (v < 0.3 ? `${v.toFixed(2)}, smeared` : v > 0.7 ? `${v.toFixed(2)}, spiky` : v.toFixed(2))}
            onChange={setPeakiness}
          />
          <Choice
            label="Shade the cells by"
            value={shade}
            options={[
              { value: "alpha", label: "forward α" },
              { value: "occupancy", label: "α·β, the paths through" },
            ]}
            onChange={setShade}
          />
          <p className="text-[12.5px] leading-snug text-ink-3 sm:col-span-2">
            Click a cell to see the sum that fills it.
          </p>
        </>
      }
    >
      <p className="mb-1.5 text-[12px] font-medium text-ink-3">The model's output y, one column per frame, brighter for more probable</p>
      <Posteriorgram
        probs={probs}
        vocab={vocab}
        columns={(width) => ({ x0: LEFT, cellW: layout(width, frames, S).cellW })}
        description={`A posteriorgram over ${vocab.join(", ")} for ${frames} frames`}
      />

      <p className="mb-1.5 mt-4 text-[12px] font-medium text-ink-3">
        The lattice: {S} states for "{label}", the 2L + 1 = {S} of the blank-extended label, by {frames} frames
      </p>
      <CanvasFigure
        height={(width) => layout(width, frames, S).height}
        deps={[lattice, shade, pick, frames, S, feasible]}
        draw={draw}
        onPointer={onPointer}
        description={`The CTC lattice for "${label}" with its best path`}
      />
      <Legend
        items={[
          { label: "stay or advance", colour: "var(--ink-4)" },
          { label: "skip a blank", colour: "var(--series-4)", dashed: true },
          { label: "best path", colour: "var(--series-2)" },
        ]}
      />
      <p className="mt-2 font-mono text-[12px] leading-relaxed text-ink-2">
        {pt === 0
          ? `α1(${name(ps)}) = y1(${name(ps)}) = ${scientific(lattice.alpha[0][ps])}${ps > 1 ? ", since a path starts on the first blank or the first letter" : ""}`
          : `α${pt + 1}(${name(ps)}) = (${terms.join(" + ")}) × y${pt + 1}(${name(ps)}) ${scientific(
              probs[pt][lattice.ext[ps]],
            )} = ${scientific(lattice.alpha[pt][ps])}`}
      </p>
      {why && <p className="mt-1 text-[12.5px] text-bad">{why}</p>}

      <div className="mt-4 grid gap-2 text-[12px]">
        <PathRow title="Best path" symbols={bestPath} />
        <PathRow title="Merge repeats" symbols={bestCollapse.merged} />
        <PathRow title="Drop blanks" symbols={bestCollapse.dropped} tone="good" />
        <PathRow title="Greedy, per frame" symbols={greedy} />
        <PathRow
          title="Greedy, collapsed"
          symbols={greedyCollapse.dropped.length ? greedyCollapse.dropped : ["(empty)"]}
          tone={greedyText === label ? "good" : "bad"}
        />
      </div>

      <Readouts>
        <Readout
          label="p(label | x)"
          value={feasible ? scientific(lattice.total) : "0"}
          hint="α_T(S) + α_T(S − 1)"
          tone="demo"
        />
        <Readout label="CTC loss" value={feasible ? (-Math.log(lattice.total)).toFixed(3) : "∞"} hint="−ln p, in nats" />
        <Readout label="Valid paths" value={lattice.paths.toLocaleString("en-US")} hint={`of ${vocab.length}^${frames} in all`} />
        <Readout
          label="Best path's share"
          value={feasible ? `${((100 * lattice.best.probability) / lattice.total).toFixed(1)}%` : "—"}
          hint="of p(label | x)"
        />
      </Readouts>
      {!feasible && (
        <p className="mt-3 text-[13px] leading-snug text-bad">
          No path fits: "{label}" needs at least L + repeats = {label.length + repeats} frames, one per letter and a
          blank between each repeated pair.
        </p>
      )}
    </DemoFrame>
  );
}

function PathRow({ title, symbols, tone }: { title: string; symbols: string[]; tone?: "good" | "bad" }) {
  const colour = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-ink";
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span className="w-32 shrink-0 text-ink-3">{title}</span>
      <span className={`flex flex-wrap gap-1 font-mono ${colour}`}>
        {symbols.map((symbol, i) => (
          <span
            key={i}
            className={`inline-flex min-w-5 justify-center rounded px-1 shadow-[inset_0_0_0_1px_var(--rule)] ${
              symbol === BLANK ? "text-ink-4" : ""
            }`}
          >
            {symbol}
          </span>
        ))}
      </span>
    </div>
  );
}
