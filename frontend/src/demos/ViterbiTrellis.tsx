import { useEffect, useMemo, useRef, useState } from "react";

import { useMediaQuery } from "../lib/useMediaQuery";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  DemoFrame,
  Plot,
  Readout,
  Readouts,
  Slider,
  Toggle,
} from "./kit";

const STATES = 3;
const SYMBOLS = ["A", "B", "C"] as const;
type Symbol = 0 | 1 | 2;
/** Each state mostly emits its own symbol: b_j(o) for state j and symbol o. */
const EMISSION = [
  [0.7, 0.2, 0.1],
  [0.15, 0.7, 0.15],
  [0.1, 0.2, 0.7],
];
const START: Symbol[] = [0, 0, 1, 0, 1, 1, 2, 2];
const STEP_MS = 320;

const log = (p: number) => (p > 0 ? Math.log(p) : -Infinity);

function logSumExp(values: number[]): number {
  const top = Math.max(...values);
  if (top === -Infinity) return -Infinity;
  return top + Math.log(values.reduce((sum, v) => sum + Math.exp(v - top), 0));
}

type Mode = "viterbi" | "forward";

type Cell = {
  value: number;
  /** For each previous state, its score plus the log transition into this one. */
  terms: number[];
  /** Viterbi's back-pointer: the previous state the max came from. */
  from: number;
  emission: number;
};

/** Forward (log-sum-exp) or Viterbi (max) over the trellis, in natural logs. */
function trellis(mode: Mode, transitions: number[][], observations: Symbol[]): Cell[][] {
  const table: Cell[][] = [];
  observations.forEach((o, t) => {
    const column: Cell[] = [];
    for (let j = 0; j < STATES; j += 1) {
      const emission = log(EMISSION[j][o]);
      if (t === 0) {
        // The model starts in its first state.
        column.push({ value: (j === 0 ? 0 : -Infinity) + emission, terms: [], from: -1, emission });
        continue;
      }
      const terms = table[t - 1].map((cell, i) => cell.value + log(transitions[i][j]));
      let from = 0;
      for (let i = 1; i < STATES; i += 1) if (terms[i] > terms[from]) from = i;
      const combined = mode === "viterbi" ? terms[from] : logSumExp(terms);
      column.push({ value: combined + emission, terms, from: terms[from] === -Infinity ? -1 : from, emission });
    }
    table.push(column);
  });
  return table;
}

const fmt = (v: number, digits = 2) => (v === -Infinity ? "−∞" : v.toFixed(digits).replace("-", "−"));

export default function ViterbiTrellis({ caption }: DemoProps) {
  const [stay1, setStay1] = useState(0.6);
  const [skip, setSkip] = useState(0);
  const [stay2, setStay2] = useState(0.6);
  const [observations, setObservations] = useState<Symbol[]>(START);
  const [mode, setMode] = useState<Mode>("viterbi");
  const [endInLast, setEndInLast] = useState(true);
  const T = observations.length;
  const total = T * STATES;
  const [cursor, setCursor] = useState(total);
  const [playingSteps, setPlayingSteps] = useState(false);
  const timer = useRef<number | null>(null);

  const skipProbability = Math.min(skip, 1 - stay1);
  const transitions = useMemo(
    () => [
      [stay1, 1 - stay1 - skipProbability, skipProbability],
      [0, stay2, 1 - stay2],
      [0, 0, 1],
    ],
    [stay1, stay2, skipProbability],
  );

  const viterbi = useMemo(() => trellis("viterbi", transitions, observations), [transitions, observations]);
  const forward = useMemo(() => trellis("forward", transitions, observations), [transitions, observations]);
  const table = mode === "viterbi" ? viterbi : forward;

  const ends = endInLast ? [STATES - 1] : [0, 1, 2];
  const logLikelihood = logSumExp(ends.map((j) => forward[T - 1][j].value));
  const bestEnd = ends.reduce((a, b) => (viterbi[T - 1][b].value > viterbi[T - 1][a].value ? b : a));
  const bestScore = viterbi[T - 1][bestEnd].value;
  const bestPath = useMemo(() => {
    if (bestScore === -Infinity) return [];
    const path = [bestEnd];
    for (let t = T - 1; t > 0; t -= 1) path.unshift(viterbi[t][path[0]].from);
    return path;
  }, [viterbi, bestEnd, bestScore, T]);

  function stopTimer() {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
    setPlayingSteps(false);
  }
  useEffect(() => () => {
    if (timer.current !== null) window.clearInterval(timer.current);
  }, []);
  // Any change to the model shows the finished trellis again.
  useEffect(() => {
    stopTimer();
    setCursor(total);
  }, [transitions, observations, mode, total]);

  function playSteps() {
    stopTimer();
    let at = cursor >= total ? 1 : cursor + 1;
    setCursor(at);
    setPlayingSteps(true);
    timer.current = window.setInterval(() => {
      at += 1;
      setCursor(at);
      if (at >= total) stopTimer();
    }, STEP_MS);
  }

  function cycle(t: number) {
    setObservations((current) => current.map((o, k) => (k === t ? (((o + 1) % 3) as Symbol) : o)));
  }

  const revealed = (t: number, j: number) => t * STATES + j < cursor;
  const currentIndex = cursor > 0 && cursor <= total ? cursor - 1 : -1;
  const focus = currentIndex >= 0 && cursor < total ? { t: Math.floor(currentIndex / STATES), j: currentIndex % STATES } : null;
  const finished = cursor >= total;
  const onBest = (t: number, j: number) => finished && mode === "viterbi" && bestPath[t] === j;

  let working: string | null = null;
  if (focus) {
    const cell = table[focus.t][focus.j];
    const name = mode === "viterbi" ? "δ" : "α";
    const combine = mode === "viterbi" ? "max" : "logsumexp";
    if (focus.t === 0) {
      working = `${name}1(S${focus.j + 1}) = log π(S${focus.j + 1}) + log b(${SYMBOLS[observations[0]]}) = ${fmt(focus.j === 0 ? 0 : -Infinity)} + ${fmt(cell.emission)} = ${fmt(cell.value)}`;
    } else {
      const parts = cell.terms.map((_, i) => `${fmt(table[focus.t - 1][i].value)} + ${fmt(log(transitions[i][focus.j]))}`);
      working = `${name}${focus.t + 1}(S${focus.j + 1}) = ${combine}(${parts.join(",  ")}) + log b(${SYMBOLS[observations[focus.t]]}) = ${fmt(cell.value)}`;
    }
  }

  // A phone gets a narrower drawing, so its numbers stay readable when scaled.
  const narrow = useMediaQuery("(max-width: 480px)");
  const width = narrow ? 340 : 440;
  const height = 250;
  const radius = narrow ? 16 : 18;

  return (
    <DemoFrame
      title="The best path through an HMM trellis"
      caption={caption}
      controls={
        <>
          <Choice
            label="Compute"
            value={mode}
            options={[
              { value: "viterbi", label: "Viterbi: max" },
              { value: "forward", label: "Forward: sum" },
            ]}
            onChange={setMode}
          />
          <Slider label="S1 stays, a11" value={stay1} min={0.05} max={0.95} step={0.05} format={(v) => v.toFixed(2)} onChange={setStay1} />
          <Slider
            label="S1 skips to S3, a13"
            value={skipProbability}
            min={0}
            max={Math.max(0.05, Math.round((1 - stay1) * 100) / 100)}
            step={0.05}
            format={(v) => v.toFixed(2)}
            onChange={setSkip}
          />
          <Slider label="S2 stays, a22" value={stay2} min={0.05} max={0.95} step={0.05} format={(v) => v.toFixed(2)} onChange={setStay2} />
          <Toggle label="The path must end in S3" checked={endInLast} onChange={setEndInLast} />
          <Actions>
            {playingSteps ? (
              <Action onClick={stopTimer}>Pause</Action>
            ) : (
              <Action onClick={playSteps}>{cursor >= total ? "Fill cell by cell" : "Continue"}</Action>
            )}
            <Action
              tone="quiet"
              onClick={() => {
                stopTimer();
                setCursor((c) => (c >= total ? 1 : c + 1));
              }}
            >
              One cell
            </Action>
            <Action
              tone="quiet"
              onClick={() => {
                stopTimer();
                setCursor(total);
              }}
            >
              Fill all
            </Action>
          </Actions>
        </>
      }
    >
      <Plot width={width} height={height} xDomain={[0.5, T + 0.5]} yDomain={[3.9, 0.2]} xTicks={0} yTicks={0}>
        {({ sx, sy }) => (
          <>
            {/* Transitions between columns. */}
            {observations.slice(1).map((_, k) => {
              const t = k + 1;
              return Array.from({ length: STATES }, (_, j) =>
                Array.from({ length: STATES }, (_, i) => {
                  if (!(transitions[i][j] > 0)) return null;
                  const cellShown = revealed(t, j);
                  const isFocus = focus && focus.t === t && focus.j === j;
                  const cell = table[t][j];
                  const back = cellShown && mode === "viterbi" && cell.from === i && cell.value > -Infinity;
                  const best = onBest(t, j) && bestPath[t - 1] === i;
                  let stroke = "var(--rule-strong)";
                  let strokeWidth = 1;
                  let opacity = 1;
                  if (mode === "forward" && (cellShown || isFocus) && cell.terms[i] > -Infinity) {
                    // Each edge drawn in proportion to its share of the sum.
                    const share = Math.exp(cell.terms[i] - logSumExp(cell.terms));
                    stroke = "var(--demo)";
                    strokeWidth = 0.8 + 3 * share;
                    opacity = 0.35 + 0.65 * share;
                  }
                  if (back) {
                    stroke = "var(--demo)";
                    strokeWidth = 1.6;
                  }
                  if (best) {
                    stroke = "var(--bad)";
                    strokeWidth = 3.2;
                  }
                  return (
                    <line
                      key={`${t}-${i}-${j}`}
                      x1={sx(t)}
                      y1={sy(i + 1)}
                      x2={sx(t + 1)}
                      y2={sy(j + 1)}
                      stroke={stroke}
                      strokeWidth={strokeWidth}
                      opacity={opacity}
                    />
                  );
                }),
              );
            })}

            {/* The cells. */}
            {observations.map((_, t) =>
              Array.from({ length: STATES }, (_, j) => {
                const shown = revealed(t, j);
                const cell = table[t][j];
                const isFocus = focus && focus.t === t && focus.j === j;
                const best = onBest(t, j);
                return (
                  <g key={`c${t}-${j}`}>
                    <circle
                      cx={sx(t + 1)}
                      cy={sy(j + 1)}
                      r={radius}
                      fill="var(--raise)"
                      stroke={best ? "var(--bad)" : isFocus ? "var(--demo)" : shown ? "var(--ink-4)" : "var(--rule-strong)"}
                      strokeWidth={best || isFocus ? 2.4 : 1}
                      strokeDasharray={shown ? undefined : "3 3"}
                    />
                    {shown && (
                      <text
                        x={sx(t + 1)}
                        y={sy(j + 1)}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize={narrow ? 10 : 11}
                        fontFamily="var(--font-mono)"
                        fill={cell.value === -Infinity ? "var(--ink-4)" : "var(--ink)"}
                      >
                        {fmt(cell.value, 1)}
                      </text>
                    )}
                  </g>
                );
              }),
            )}

            {/* State names and the observations, which a click changes. */}
            {Array.from({ length: STATES }, (_, j) => (
              <text key={`s${j}`} x={sx(0.5) - 20} y={sy(j + 1)} dominantBaseline="central" fontSize={11} fill="var(--ink-3)">
                S{j + 1}
              </text>
            ))}
            {observations.map((o, t) => (
              <g
                key={`o${t}`}
                role="button"
                tabIndex={0}
                aria-label={`Observation ${t + 1} is ${SYMBOLS[o]}; change it`}
                className="cursor-pointer"
                onClick={() => cycle(t)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    cycle(t);
                  }
                }}
              >
                <rect x={sx(t + 1) - 13} y={sy(0.2) + 1} width={26} height={20} rx={4} fill="var(--chrome)" stroke="var(--rule-strong)" />
                <text x={sx(t + 1)} y={sy(0.2) + 11} textAnchor="middle" dominantBaseline="central" fontSize={12} fontWeight={600} fill="var(--ink)">
                  {SYMBOLS[o]}
                </text>
                <text x={sx(t + 1)} y={sy(3.9) - 2} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink-4)">
                  t={t + 1}
                </text>
              </g>
            ))}
          </>
        )}
      </Plot>

      <p className="mt-2 min-h-[2.6em] break-words font-mono text-[12px] leading-snug text-ink-2">
        {working ??
          (mode === "viterbi"
            ? "Each cell holds log δ: the score of the best path into it. The red path is the best of all, traced back through the pointers."
            : "Each cell holds log α: every path into it, summed with log-sum-exp. Edge width is each predecessor's share of the sum.")}
      </p>
      <p className="mt-1 text-[12.5px] leading-snug text-ink-3">
        Click an observation to change it. S1 mostly emits A, S2 mostly B, and S3 mostly C, each with probability 0.7.
      </p>

      <Readouts>
        <Readout label="log P(O), forward" value={fmt(logLikelihood)} hint="all paths, summed" />
        <Readout label="log P(best path)" value={fmt(bestScore)} hint="Viterbi" tone="demo" />
        <Readout
          label="Best path's share"
          value={bestScore === -Infinity ? "0%" : `${(Math.exp(bestScore - logLikelihood) * 100).toFixed(1)}%`}
          hint="of the total probability"
        />
        <Readout label="Best state sequence" value={bestPath.length ? bestPath.map((j) => j + 1).join(" ") : "none"} />
      </Readouts>
    </DemoFrame>
  );
}
