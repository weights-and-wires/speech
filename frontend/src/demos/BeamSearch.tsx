import { useMemo, useState } from "react";

import { Posteriorgram, useColumnWidth } from "./figure";
import type { DemoProps } from "./index";
import { DemoFrame, Legend, Readout, Readouts, Slider, Toggle } from "./kit";

const BLANK = "∅";
const VOCAB = [BLANK, "c", "a", "o", "t"];

/**
 * The model's output for eight frames, lined up with the frames below of someone saying "cat". The vowel is
 * split between "a" and "o", and "o" wins each frame it is ahead in, so the
 * best path reads "cot" while "a" holds more probability over all paths.
 * Rows are frames, columns follow `VOCAB`.
 */
const POSTERIORS: number[][] = [
  [0.3, 0.6, 0.04, 0.03, 0.03],
  [0.55, 0.35, 0.04, 0.03, 0.03],
  [0.3, 0.02, 0.3, 0.36, 0.02],
  [0.36, 0.02, 0.34, 0.26, 0.02],
  [0.6, 0.02, 0.18, 0.18, 0.02],
  [0.4, 0.02, 0.02, 0.02, 0.54],
  [0.62, 0.02, 0.02, 0.02, 0.32],
  [0.9, 0.02, 0.03, 0.02, 0.03],
];
const T = POSTERIORS.length;

/**
 * A character bigram model, `P(next | previous)`, with `^` for the start of
 * the word and `$` for its end. It has seen "cat" far more than "cot".
 */
const BIGRAM: Record<string, Record<string, number>> = {
  "^": { c: 0.5, a: 0.25, o: 0.05, t: 0.2 },
  c: { a: 0.6, o: 0.2, t: 0.05, c: 0.01, $: 0.14 },
  a: { t: 0.6, c: 0.15, o: 0.01, a: 0.01, $: 0.23 },
  o: { t: 0.4, c: 0.1, a: 0.1, o: 0.1, $: 0.3 },
  t: { $: 0.6, a: 0.2, c: 0.05, o: 0.1, t: 0.05 },
};

function lmLog(prefix: string, end = false): number {
  let previous = "^";
  let sum = 0;
  for (const c of prefix) {
    sum += Math.log(BIGRAM[previous][c] ?? 1e-3);
    previous = c;
  }
  if (end) sum += Math.log(BIGRAM[previous].$ ?? 1e-3);
  return sum;
}

type Candidate = {
  prefix: string;
  /** Probability of the prefix over paths ending in blank, and in a letter. */
  pb: number;
  pnb: number;
  acoustic: number;
  lm: number;
  bonus: number;
  score: number;
  kept: boolean;
  /** The prefixes at the previous step that sent probability here. */
  parents: string[];
};

/**
 * CTC prefix beam search (Hannun et al., 2014) with shallow fusion:
 * a prefix scores log(p_b + p_nb) + alpha log P_LM(prefix) + beta |prefix|,
 * and the beam keeps the `width` best at each frame. The final ranking adds
 * the LM's probability of ending the word.
 */
function prefixBeamSearch(width: number, alpha: number, beta: number) {
  let beam = new Map<string, { pb: number; pnb: number }>([["", { pb: 1, pnb: 0 }]]);
  const steps: Candidate[][] = [];
  const rank = (prefix: string, pb: number, pnb: number, end = false) => {
    const acoustic = Math.log(pb + pnb);
    const lm = alpha * lmLog(prefix, end);
    const bonus = beta * prefix.length;
    return { acoustic, lm, bonus, score: acoustic + lm + bonus };
  };

  for (let t = 0; t < T; t += 1) {
    const y = POSTERIORS[t];
    const next = new Map<string, { pb: number; pnb: number; parents: Set<string> }>();
    const entry = (prefix: string, parent: string) => {
      let value = next.get(prefix);
      if (!value) {
        value = { pb: 0, pnb: 0, parents: new Set() };
        next.set(prefix, value);
      }
      value.parents.add(parent);
      return value;
    };
    for (const [prefix, { pb, pnb }] of beam) {
      // A blank leaves the prefix as it is, now ending in blank.
      entry(prefix, prefix).pb += (pb + pnb) * y[0];
      for (let v = 1; v < VOCAB.length; v += 1) {
        const c = VOCAB[v];
        const extended = prefix + c;
        if (prefix.endsWith(c)) {
          // A repeat after a blank starts a new copy; without one it merges.
          entry(extended, prefix).pnb += pb * y[v];
          entry(prefix, prefix).pnb += pnb * y[v];
        } else {
          entry(extended, prefix).pnb += (pb + pnb) * y[v];
        }
      }
    }
    const ranked = [...next.entries()]
      .map(([prefix, { pb, pnb, parents }]) => ({ prefix, pb, pnb, parents: [...parents], ...rank(prefix, pb, pnb), kept: false }))
      .sort((a, b) => b.score - a.score);
    ranked.forEach((candidate, i) => (candidate.kept = i < width));
    steps.push(ranked);
    beam = new Map(ranked.slice(0, width).map((c) => [c.prefix, { pb: c.pb, pnb: c.pnb }]));
  }

  const final = [...beam.entries()]
    .map(([prefix, { pb, pnb }]) => ({ prefix, ...rank(prefix, pb, pnb, true) }))
    .sort((a, b) => b.score - a.score);
  return { steps, final };
}

const show = (prefix: string) => prefix || "ε";

export default function BeamSearch({ caption }: DemoProps) {
  const [width, setWidth] = useState(3);
  const [alpha, setAlpha] = useState(0.5);
  const [bonusOn, setBonusOn] = useState(false);
  const [step, setStep] = useState(T);
  const beta = bonusOn ? 1 : 0;

  const search = useMemo(() => prefixBeamSearch(width, alpha, beta), [width, alpha, beta]);
  const wide = useMemo(() => prefixBeamSearch(256, alpha, beta), [alpha, beta]);
  const greedyPath = POSTERIORS.map((row) => VOCAB[row.indexOf(Math.max(...row))]);
  const greedy = greedyPath
    .filter((c, i) => i === 0 || c !== greedyPath[i - 1])
    .filter((c) => c !== BLANK)
    .join("");
  const answer = search.final[0]?.prefix ?? "";
  const best = wide.final[0]?.prefix ?? "";
  const current = search.steps[step - 1];

  return (
    <DemoFrame
      title="Keeping the best few hypotheses"
      caption={caption}
      controls={
        <>
          <Slider label="Beam width, B" value={width} min={1} max={6} onChange={setWidth} />
          <Slider
            label="LM weight, α"
            value={alpha}
            min={0}
            max={2}
            step={0.25}
            format={(v) => v.toFixed(2)}
            onChange={setAlpha}
          />
          <Slider label="Frame" value={step} min={1} max={T} format={(v) => `${v} of ${T}`} onChange={setStep} />
          <Toggle
            label="Insertion bonus, β = 1 per letter, against the LM's pull toward short outputs"
            checked={bonusOn}
            onChange={setBonusOn}
          />
        </>
      }
    >
      <p className="mb-1.5 text-[12px] font-medium text-ink-3">The model's output for eight frames, lined up with the frames below</p>
      <Posteriorgram
        probs={POSTERIORS}
        vocab={VOCAB}
        columns={(width) => {
          const colW = trellisColumn(width);
          return { x0: TRELLIS_LEFT + colW / 2, cellW: colW };
        }}
        description="A posteriorgram over blank, c, a, o, and t for eight frames"
      />

      <p className="mb-1 mt-4 text-[12px] font-medium text-ink-3">
        Prefixes at each frame, best first: the beam keeps {width}, and the rest are pruned
      </p>
      <Trellis steps={search.steps} width={width} upTo={step} />
      <Legend
        items={[
          { label: "kept in the beam", colour: "var(--demo)" },
          { label: "pruned", colour: "var(--ink-4)", dashed: true },
        ]}
      />

      <div className="mt-4 min-w-0">
        <p className="mb-1 text-[12px] font-medium text-ink-3">
          Frame {step}: score = log p_CTC + α log P_LM + β·length
        </p>
        <table className="w-full table-fixed border-collapse font-mono text-[11.5px] tabular-nums">
          <thead>
            <tr className="text-left text-[11px] text-ink-3">
              <th className="w-[22%] py-1 font-medium">prefix</th>
              <th className="py-1 text-right font-medium">log p</th>
              <th className="py-1 text-right font-medium">α log P</th>
              <th className="py-1 text-right font-medium">β·len</th>
              <th className="py-1 text-right font-medium">score</th>
            </tr>
          </thead>
          <tbody>
            {current.slice(0, width + 4).map((c) => (
              <tr key={c.prefix} className={`border-t border-rule ${c.kept ? "text-ink" : "text-ink-4"}`}>
                <td className={`py-1 ${c.kept ? "font-semibold" : "line-through"}`}>{show(c.prefix)}</td>
                <td className="py-1 text-right">{c.acoustic.toFixed(2)}</td>
                <td className="py-1 text-right">{c.lm.toFixed(2)}</td>
                <td className="py-1 text-right">{c.bonus.toFixed(1)}</td>
                <td className="py-1 text-right">{c.score.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {current.length > width + 4 && (
          <p className="mt-1 text-[11.5px] text-ink-3">and {current.length - width - 4} more, pruned</p>
        )}
      </div>

      <Readouts>
        <Readout
          label="Greedy"
          value={greedy}
          hint={`best path ${greedyPath.join("")}`}
          tone={greedy === "cat" ? "good" : "bad"}
        />
        <Readout
          label={`Beam of ${width}`}
          value={show(answer)}
          hint="after the end-of-word LM term"
          tone={answer === "cat" ? "good" : "bad"}
        />
        <Readout label="A beam of 256" value={show(best)} hint="as good as exhaustive here" />
        <Readout
          label="The spoken word"
          value="cat"
          hint={alpha === 0 ? "acoustics alone" : `with α = ${alpha.toFixed(2)}${beta ? ", β = 1" : ""}`}
        />
      </Readouts>
    </DemoFrame>
  );
}

const ROW = 24;
const TRELLIS_LEFT = 16;
/** The trellis's column spacing: the start, then one column per frame. */
const trellisColumn = (width: number) => Math.max(34, (width - TRELLIS_LEFT - 24) / T);
const TOP = 18;
const pillWidth = (label: string) => Math.max(18, label.length * 6.2 + 8);

/** The prefixes of every frame as columns, with an edge from each prefix that fed them. */
function Trellis({ steps, width, upTo }: { steps: Candidate[][]; width: number; upTo: number }) {
  const [box, pixels] = useColumnWidth();
  const shownPerStep = width + 3;
  const height = TOP + (shownPerStep + 0.6) * ROW;
  const columns = T + 1;
  const colW = trellisColumn(pixels);
  const x = (t: number) => TRELLIS_LEFT + t * colW;
  const y = (row: number) => TOP + row * ROW + ROW / 2;

  // Rows: the root at step 0, then each step's candidates in rank order.
  const layout = steps.map((candidates) => candidates.slice(0, shownPerStep));
  const rowOf = (t: number, prefix: string) =>
    t === 0 ? (prefix === "" ? 0 : -1) : layout[t - 1].findIndex((c) => c.prefix === prefix);

  return (
    <div ref={box} className="w-full min-w-0">
      {pixels > 0 && (
        <svg width={pixels} height={height} role="img" aria-label="The beam's prefixes at each frame">
          {Array.from({ length: columns }, (_, t) => (
            <text
              key={t}
              x={x(t)}
              y={10}
              textAnchor="middle"
              fontSize={10}
              fontFamily="var(--font-mono)"
              fill={t === upTo ? "var(--ink)" : "var(--ink-3)"}
            >
              {t === 0 ? "start" : t}
            </text>
          ))}
          {layout.slice(0, upTo).map((candidates, i) =>
            candidates.map((c, row) =>
              c.parents.map((parent) => {
                const from = rowOf(i, parent);
                if (from < 0) return null;
                const parentLabel = show(parent);
                return (
                  <line
                    key={`${i}-${c.prefix}-${parent}`}
                    x1={x(i) + pillWidth(parentLabel) / 2}
                    y1={y(from)}
                    x2={x(i + 1) - pillWidth(show(c.prefix)) / 2}
                    y2={y(row)}
                    stroke={c.kept ? "var(--demo)" : "var(--ink-4)"}
                    strokeWidth={c.kept ? 1.3 : 0.8}
                    strokeDasharray={c.kept ? undefined : "3 3"}
                    opacity={c.kept ? 0.85 : 0.45}
                  />
                );
              }),
            ),
          )}
          <Node x={x(0)} y={y(0)} label="ε" kept />
          {layout.slice(0, upTo).map((candidates, i) =>
            candidates.map((c, row) => (
              <Node key={`${i}-${c.prefix}`} x={x(i + 1)} y={y(row)} label={show(c.prefix)} kept={c.kept} />
            )),
          )}
          {upTo < T && (
            <rect
              x={x(upTo) + colW * 0.5}
              y={TOP - 4}
              width={Math.max(0, pixels - x(upTo) - colW * 0.5)}
              height={height - TOP + 4}
              fill="var(--raise)"
              opacity={0.6}
            />
          )}
        </svg>
      )}
    </div>
  );
}

function Node({ x, y, label, kept }: { x: number; y: number; label: string; kept: boolean }) {
  const w = pillWidth(label);
  return (
    <g>
      <rect
        x={x - w / 2}
        y={y - 8}
        width={w}
        height={16}
        rx={8}
        fill={kept ? "color-mix(in oklab, var(--demo) 16%, var(--raise))" : "var(--raise)"}
        stroke={kept ? "var(--demo)" : "var(--ink-4)"}
        strokeWidth={kept ? 1.2 : 0.8}
        strokeDasharray={kept ? undefined : "2.5 2"}
      />
      <text
        x={x}
        y={y + 3.5}
        textAnchor="middle"
        fontSize={10}
        fontFamily="var(--font-mono)"
        fill={kept ? "var(--ink)" : "var(--ink-3)"}
      >
        {label}
      </text>
    </g>
  );
}
