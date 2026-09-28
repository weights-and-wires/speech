import { useMemo, useState } from "react";

import { gaussian, seededRandom } from "./dsp";
import { CanvasFigure, colourScale, type Paint } from "./figure";
import type { DemoProps } from "./index";
import { Bars, Choice, DemoFrame, Plot, Readout, Readouts, Slider, Toggle } from "./kit";

const TEXT = "the cat sat on the mat";
/** Frames each character lasts: vowels longest, spaces shortest. */
const DURATIONS = [...TEXT].map((c) => (c === " " ? 1 : "aeiou".includes(c) ? 4 : 2));
const STARTS = DURATIONS.map((_, i) => DURATIONS.slice(0, i).reduce((a, b) => a + b, 0));
const FRAMES = DURATIONS.reduce((a, b) => a + b, 0);
const TOKENS = TEXT.length;
/** The guided-attention width from DC-TTS (Tachibana et al., 2018). */
const G = 0.2;

type Case = "monotonic" | "skip" | "repeat" | "diffuse";
type Task = "tts" | "asr";

/** The character a frame belongs to, as a continuous position. */
function tokenAt(frame: number): number {
  let j = 0;
  while (j + 1 < TOKENS && STARTS[j + 1] <= frame) j += 1;
  return j + (frame - STARTS[j] + 0.5) / DURATIONS[j] - 0.5;
}

const centreFrame = (token: number) => {
  const j = Math.max(0, Math.min(TOKENS - 1, Math.round(token)));
  return STARTS[j] + DURATIONS[j] / 2 - 0.5;
};

/**
 * Where the failing model's content attention goes instead, in characters.
 * Skipping: after the "a" of "cat", the "at" of "sat" looks the same, so it
 * jumps four characters ahead. Repeating: the second "the" looks like the
 * first, so it jumps back and reads the sentence again.
 */
function failure(kind: Case, token: number): number | null {
  if (kind === "skip" && token > 5.5) return Math.min(TOKENS - 1, token + 4);
  if (kind === "repeat" && token > 14.5) return token - 15;
  return null;
}

/**
 * An attention matrix, `steps` rows of weights over `positions`, each row
 * summing to 1. The content part peaks where the model thinks it should
 * look, plus the failure's distractor; with `location`, each row is also
 * multiplied by the previous row convolved with a kernel that allows only
 * small forward moves, a stand-in for location-sensitive attention's
 * `F * alpha_{i-1}` term (Chorowski et al., 2015).
 */
function attention(task: Task, kind: Case, location: boolean) {
  const steps = task === "tts" ? FRAMES : TOKENS;
  const positions = task === "tts" ? TOKENS : FRAMES;
  const random = seededRandom(task === "tts" ? 11 : 12);
  const gauss = (u: number, centre: number, width: number) => Math.exp(-((u - centre) ** 2) / (2 * width * width));

  const content: number[][] = [];
  for (let i = 0; i < steps; i += 1) {
    const token = task === "tts" ? tokenAt(i) : i;
    const wrong = failure(kind, token);
    const centre = task === "tts" ? token : centreFrame(token);
    const width =
      kind === "diffuse" ? (task === "tts" ? 4.5 : 11) : task === "tts" ? 0.7 : DURATIONS[i] / 2 + 0.6;
    const row = Array.from({ length: positions }, (_, u) => {
      let value = (wrong === null ? 1 : 0.45) * gauss(u, centre, width);
      if (wrong !== null) value += 1.8 * gauss(u, task === "tts" ? wrong : centreFrame(wrong), width);
      if (kind === "diffuse") value += 0.35;
      return Math.max(1e-4, value * (1 + 0.18 * gaussian(random))) + 1e-3;
    });
    content.push(row);
  }

  // Location: moves of 0, 1, or 2 characters per step for TTS, and of up to
  // about two characters' worth of frames per character for ASR.
  const kernel =
    task === "tts" ? [0.45, 0.4, 0.15] : Array.from({ length: 9 }, (_, k) => 1 - k / 10);
  const out: number[][] = [];
  for (let i = 0; i < steps; i += 1) {
    let row = content[i].slice();
    if (location) {
      const previous = i === 0 ? row.map((_, u) => (u === 0 ? 1 : 0)) : out[i - 1];
      row = row.map((value, u) => {
        let prior = 0;
        kernel.forEach((w, k) => {
          if (u - k >= 0) prior += w * previous[u - k];
        });
        return value * (prior + 1e-4);
      });
    }
    const sum = row.reduce((a, b) => a + b, 0);
    out.push(row.map((v) => v / sum));
  }
  return { steps, positions, weights: out };
}

/**
 * The attention mass near the diagonal, averaged over decoder steps: each
 * weight counts by a Gaussian of width G in its distance from the diagonal.
 * It is 1 when every step attends on the diagonal. The guided attention loss
 * penalizes the complement of the same Gaussian, but sums over every cell.
 */
function diagonality(weights: number[][], task: Task) {
  const N = weights.length;
  const U = weights[0].length;
  let sum = 0;
  weights.forEach((row, i) =>
    row.forEach((a, u) => {
      const [n, t] = task === "tts" ? [u / U, i / N] : [i / N, u / U];
      sum += a * Math.exp(-((n - t) ** 2) / (2 * G * G));
    }),
  );
  return sum / N;
}

export default function AttentionAlignment({ caption, task: taskProp }: DemoProps & { task?: string }) {
  const task: Task = taskProp === "asr" ? "asr" : "tts";
  const [kind, setKind] = useState<Case>("repeat");
  const [location, setLocation] = useState(false);
  const { steps, positions, weights } = useMemo(() => attention(task, kind, location), [task, kind, location]);
  const [scrub, setScrub] = useState(task === "tts" ? 44 : 17);
  const step = Math.min(scrub, steps) - 1;

  const peaks = weights.map((row) => row.indexOf(Math.max(...row)));
  const focus = weights.reduce((s, row) => s + Math.max(...row), 0) / steps;
  // Jumps in characters beyond the expected move: 0 or 1 a step for TTS,
  // where a character lasts several steps, and exactly 1 for ASR.
  const charPeaks = peaks.map((p) => (task === "tts" ? p : Math.round(tokenAt(p))));
  let worstBack = 0;
  let worstForward = 0;
  for (let i = 1; i < steps; i += 1) {
    const move = charPeaks[i] - charPeaks[i - 1] - (task === "asr" ? 1 : 0);
    worstBack = Math.min(worstBack, move);
    worstForward = Math.max(worstForward, move);
  }
  const score = diagonality(weights, task);
  const lastToken = TOKENS - 1;
  const reachedEnd = charPeaks.findIndex((p) => p >= lastToken);

  const heat = colourScale("magma");
  const LEFT = 28;
  const BOTTOM = 26;
  const TOP = 6;

  function draw({ ctx, width, height, colour }: Paint) {
    // Frames run along x in both tasks; characters run up y.
    const cols = FRAMES;
    const rows = TOKENS;
    const plotW = width - LEFT - 8;
    const plotH = height - TOP - BOTTOM;
    const cw = plotW / cols;
    const rh = plotH / rows;
    const max = Math.max(...weights.flat());
    for (let i = 0; i < steps; i += 1) {
      for (let u = 0; u < positions; u += 1) {
        const [c, r] = task === "tts" ? [i, u] : [u, i];
        ctx.fillStyle = heat(Math.sqrt(weights[i][u] / max));
        ctx.fillRect(LEFT + c * cw, TOP + plotH - (r + 1) * rh, cw + 0.6, rh + 0.6);
      }
    }
    ctx.strokeStyle = colour("--rule-strong");
    ctx.lineWidth = 1;
    ctx.strokeRect(LEFT - 0.5, TOP - 0.5, plotW + 1, plotH + 1);

    // The diagonal the guided attention loss pulls toward.
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = colour("--ink-4");
    ctx.beginPath();
    ctx.moveTo(LEFT, TOP + plotH);
    ctx.lineTo(LEFT + plotW, TOP);
    ctx.stroke();
    ctx.setLineDash([]);

    // The scrubbed decoder step.
    ctx.strokeStyle = colour("--demo");
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (task === "tts") {
      const px = LEFT + (step + 0.5) * cw;
      ctx.moveTo(px, TOP);
      ctx.lineTo(px, TOP + plotH);
    } else {
      const py = TOP + plotH - (step + 0.5) * rh;
      ctx.moveTo(LEFT, py);
      ctx.lineTo(LEFT + plotW, py);
    }
    ctx.stroke();

    ctx.font = `10px "JetBrains Mono Variable", ui-monospace, monospace`;
    ctx.fillStyle = colour("--ink-3");
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const every = rh < 11 ? 2 : 1;
    for (let r = 0; r < rows; r += every) {
      ctx.fillText(TEXT[r] === " " ? "␣" : TEXT[r], LEFT - 5, TOP + plotH - (r + 0.5) * rh);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let f = 0; f <= cols; f += 10) ctx.fillText(String(f), LEFT + f * cw, TOP + plotH + 4);
    ctx.font = `11px "Inter Variable", ui-sans-serif, system-ui, sans-serif`;
    ctx.textBaseline = "bottom";
    ctx.fillText(
      task === "tts" ? "decoder step, one mel frame each" : "encoder frame",
      LEFT + plotW / 2,
      height - 1,
    );
  }

  const row = weights[step];
  const bins = row.map((y, u) => ({ x0: u - 0.45, x1: u + 0.45, y }));
  const attended = charPeaks[step];

  return (
    <DemoFrame
      title="Where the decoder looks at each step"
      caption={caption}
      controls={
        <>
          <Choice
            label="Alignment"
            value={kind}
            options={[
              { value: "monotonic", label: "Monotonic" },
              { value: "skip", label: "Skipping" },
              { value: "repeat", label: "Repeating" },
              { value: "diffuse", label: "Not aligned yet" },
            ]}
            onChange={setKind}
          />
          <Slider
            label={task === "tts" ? "Decoder step" : "Output character"}
            value={Math.min(scrub, steps)}
            min={1}
            max={steps}
            onChange={setScrub}
          />
          <Toggle
            label="Location-sensitive: each step also sees where the last one looked"
            checked={location}
            onChange={setLocation}
          />
        </>
      }
    >
      <CanvasFigure
        height={(w) => Math.round(Math.min(300, Math.max(190, w * 0.42)))}
        deps={[weights, step, task]}
        draw={draw}
        description={`An attention map for "${TEXT}", ${kind}`}
      />
      <p className="mt-1 text-[12px] leading-snug text-ink-3">
        {task === "tts"
          ? "Each column is one decoder step's attention over the characters, brighter for more weight. The dashed line is the diagonal."
          : "Each row is one output character's attention over the encoder frames, brighter for more weight. The dashed line is the diagonal."}
      </p>

      <div className="mt-4">
        <p className="mb-1 text-[12px] font-medium text-ink-3">
          {task === "tts" ? `Step ${step + 1}` : `Character ${step + 1}, "${TEXT[step]}"`}: the attention weights
        </p>
        {task === "tts" && (
          <div className="mb-2 flex flex-wrap gap-[3px] font-mono text-[13px]">
            {[...TEXT].map((c, u) => (
              <span
                key={u}
                className={`relative inline-flex h-6 w-[1.15rem] items-center justify-center overflow-hidden rounded ${
                  u === attended ? "text-ink shadow-[inset_0_0_0_1.5px_var(--ink)]" : "text-ink-2"
                }`}
              >
                <span
                  aria-hidden
                  className="absolute inset-0 bg-demo"
                  style={{ opacity: Math.min(0.85, row[u] * 1.2) }}
                />
                <span className="relative">{c === " " ? "␣" : c}</span>
              </span>
            ))}
          </div>
        )}
        <Plot
          height={110}
          xDomain={[-0.5, positions - 0.5]}
          yDomain={[0, 1]}
          yTicks={2}
          xTicks={task === "tts" ? 0 : 5}
          xLabel={task === "tts" ? undefined : "encoder frame"}
          tickFormat={(v, axis) => (axis === "x" ? v.toFixed(0) : v.toFixed(1))}
        >
          {() => <Bars bins={bins} />}
        </Plot>
      </div>

      <Readouts>
        <Readout label="Diagonality" value={score.toFixed(2)} hint={`mass within a Gaussian of width ${G} of the diagonal`} tone={score > 0.8 ? "good" : "bad"} />
        <Readout label="Focus" value={focus.toFixed(2)} hint="mean of each step's largest weight" />
        <Readout
          label="Largest jump"
          value={`${worstBack < -1.5 ? worstBack.toFixed(0) : `+${worstForward.toFixed(0)}`} chars`}
          hint={worstBack < -1.5 ? "backward: it repeats" : worstForward > 2.5 ? "forward: it skips" : "small steps only"}
          tone={worstBack < -1.5 || worstForward > 2.5 ? "bad" : "good"}
        />
        <Readout
          label="Reaches the end"
          value={reachedEnd < 0 ? "never" : `step ${reachedEnd + 1}`}
          hint={reachedEnd < 0 ? "so the stop token never fires" : `of ${steps}`}
          tone={reachedEnd < 0 ? "bad" : "plain"}
        />
      </Readouts>
    </DemoFrame>
  );
}
