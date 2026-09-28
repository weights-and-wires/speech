import { useState, type ReactNode } from "react";

import type { DemoProps } from "./index";
import { Choice, DemoFrame, Readout, Readouts, Slider } from "./kit";

type Layout = "flatten" | "parallel" | "delay" | "moshi";

/** A token's place in generation: its step, then its order within the step. */
type Slot = { step: number; depth: number };
type Stream = { id: string; label: string; kind: "text" | "audio" | "user"; k: number; slots: Slot[] };

/**
 * When each token is generated, for `K` codebooks over `T` frames. MusicGen
 * formalizes a layout as a partition of the (frame, codebook) positions into
 * steps (Copet et al., 2023):
 *
 * - flattening: one token a step, frame by frame, K T steps;
 * - parallel: a frame's K tokens in one step, T steps;
 * - delay: codebook k shifted right by k steps, T + K - 1 steps;
 * - Moshi: a text token and the semantic codebook at frame t, the acoustic
 *   codebooks delayed by tau, and a depth transformer that emits a step's
 *   tokens one after another, so T + tau big steps (Défossez et al., 2024).
 *   The user's audio arrives the same way and is heard, not predicted.
 */
function schedule(layout: Layout, K: number, T: number, tau: number): Stream[] {
  const frames = (slot: (t: number) => Slot) => Array.from({ length: T }, (_, t) => slot(t));
  const audio = (k: number, slot: (t: number) => Slot): Stream => ({
    id: `q${k}`,
    label: layout === "moshi" ? (k === 0 ? "semantic" : `acoustic ${k + 1}`) : `codebook ${k + 1}`,
    kind: "audio",
    k,
    slots: frames(slot),
  });
  const ks = Array.from({ length: K }, (_, k) => k);
  if (layout === "flatten") return ks.map((k) => audio(k, (t) => ({ step: t * K + k, depth: 0 })));
  if (layout === "parallel") return ks.map((k) => audio(k, (t) => ({ step: t, depth: 0 })));
  if (layout === "delay") return ks.map((k) => audio(k, (t) => ({ step: t + k, depth: 0 })));
  const moshiSlot = (k: number) => (t: number) => ({ step: k === 0 ? t : t + tau, depth: 1 + k });
  return [
    { id: "text", label: "text", kind: "text", k: -1, slots: frames((t) => ({ step: t, depth: 0 })) },
    ...ks.map((k) => audio(k, moshiSlot(k))),
    ...ks.map((k) => ({
      id: `u${k}`,
      label: k === 0 ? "user sem." : `user ac. ${k + 1}`,
      kind: "user" as const,
      k,
      slots: frames(moshiSlot(k)),
    })),
  ];
}

type State = "picked" | "known" | "parallel" | "later" | "heard";

function relation(slot: Slot, picked: Slot, stream: Stream): State {
  if (stream.kind === "user") return slot.step < picked.step ? "heard" : "later";
  if (slot.step < picked.step || (slot.step === picked.step && slot.depth < picked.depth)) return "known";
  if (slot.step === picked.step && slot.depth === picked.depth) return "parallel";
  return "later";
}

const CELL: Record<State, string> = {
  picked: "bg-demo text-raise font-semibold",
  known: "bg-[color-mix(in_oklab,var(--demo)_24%,transparent)] text-ink",
  parallel: "bg-[color-mix(in_oklab,var(--warn)_30%,transparent)] text-ink",
  later: "text-ink-4 shadow-[inset_0_0_0_1px_var(--rule)]",
  heard: "bg-[color-mix(in_oklab,var(--series-3)_22%,transparent)] text-ink",
};

export default function DelayPattern({ caption, layout: layoutProp }: DemoProps & { layout?: string }) {
  const initial: Layout =
    layoutProp === "flatten" || layoutProp === "parallel" || layoutProp === "moshi" ? layoutProp : "delay";
  const [layout, setLayout] = useState<Layout>(initial);
  const [K, setK] = useState(4);
  const [T, setT] = useState(6);
  const [tau, setTau] = useState(1);
  const [pick, setPick] = useState<{ stream: string; t: number }>({ stream: "q2", t: 2 });

  const streams = schedule(layout, K, T, tau);
  const predicted = streams.filter((s) => s.kind !== "user");
  const pickedStream = predicted.find((s) => s.id === pick.stream) ?? predicted[Math.min(2, predicted.length - 1)];
  const pickedT = Math.min(pick.t, T - 1);
  const picked = pickedStream.slots[pickedT];
  const steps = Math.max(...streams.flatMap((s) => s.slots.map((slot) => slot.step))) + 1;
  const audioTokens = K * T;

  const knownCount = predicted.reduce(
    (n, s) => n + s.slots.filter((slot) => relation(slot, picked, s) === "known").length,
    0,
  );
  const parallelCount =
    predicted.reduce((n, s) => n + s.slots.filter((slot) => relation(slot, picked, s) === "parallel").length, 0) - 1;
  const sameFrameCoarser = pickedStream.kind === "audio" && pickedStream.k > 0
    ? predicted
        .filter((s) => s.kind === "audio" && s.k < pickedStream.k)
        .every((s) => relation(s.slots[pickedT], picked, s) === "known")
    : null;

  const perStep = layout === "flatten" ? 1 : layout === "moshi" ? 1 + 2 * K : K;
  const positions = layout === "flatten" ? audioTokens : steps;
  const sequential = layout === "moshi" ? `${steps} + ${steps * (K + 1)}` : String(steps);

  const name = (s: Stream) => (s.kind === "text" ? "text" : `${layout === "moshi" && s.k === 0 ? "the semantic codebook" : `codebook ${s.k + 1}`}`);

  return (
    <DemoFrame
      title="Interleaving codebooks with a delay"
      caption={caption}
      controls={
        <>
          <Choice
            label="Layout"
            value={layout}
            options={[
              { value: "flatten", label: "Flatten" },
              { value: "parallel", label: "Parallel" },
              { value: "delay", label: "Delay" },
              { value: "moshi", label: "Moshi streams" },
            ]}
            onChange={setLayout}
          />
          <Slider label="Codebooks, K" value={K} min={2} max={8} onChange={setK} />
          <Slider label="Frames, T" value={T} min={3} max={10} onChange={setT} />
          {layout === "moshi" && (
            <Slider
              label="Acoustic delay, τ"
              value={tau}
              min={1}
              max={2}
              format={(v) => `${v} ${v === 1 ? "frame" : "frames"}, ${80 * v} ms`}
              onChange={setTau}
            />
          )}
          <p className="text-[12.5px] leading-snug text-ink-3 sm:col-span-2">Click a token to predict it.</p>
        </>
      }
    >
      <p className="mb-1.5 text-[12px] font-medium text-ink-3">
        The codes, one column per frame; each cell holds the step that generates it
      </p>
      <Grid
        columns={T}
        header={(t) => `t${t + 1}`}
        rows={predicted.map((stream) => ({
          key: stream.id,
          label: stream.label,
          cells: stream.slots.map((slot, t) => {
            const state = stream.id === pickedStream.id && t === pickedT ? "picked" : relation(slot, picked, stream);
            return {
              key: `${stream.id}-${t}`,
              text: String(slot.step + 1) + (layout === "moshi" ? `.${slot.depth + 1}` : ""),
              state,
              onClick: () => setPick({ stream: stream.id, t }),
            };
          }),
        }))}
      />

      <p className="mb-1.5 mt-4 text-[12px] font-medium text-ink-3">
        {layout === "flatten"
          ? `The sequence the model reads: ${audioTokens} tokens, one a step`
          : `The sequence the model reads: ${steps} steps, ${perStep} tokens each`}
      </p>
      {layout === "flatten" ? (
        <div className="flex flex-wrap gap-[3px]">
          {Array.from({ length: audioTokens }, (_, i) => {
            const t = Math.floor(i / K);
            const k = i % K;
            const stream = predicted[k];
            const state = stream.id === pickedStream.id && t === pickedT ? "picked" : relation(stream.slots[t], picked, stream);
            return (
              <Clickable
                key={i}
                onClick={() => setPick({ stream: stream.id, t })}
                className={`inline-flex h-7 min-w-9 items-center justify-center rounded px-1 font-mono text-[10.5px] ${CELL[state]}`}
              >
                {k + 1}·t{t + 1}
              </Clickable>
            );
          })}
        </div>
      ) : (
        <Grid
          columns={steps}
          header={(s) => String(s + 1)}
          rows={streams.map((stream) => ({
            key: stream.id,
            label: stream.label,
            cells: Array.from({ length: steps }, (_, s) => {
              const t = stream.slots.findIndex((slot) => slot.step === s);
              if (t < 0) return { key: `${stream.id}-${s}`, text: "", state: null };
              const state =
                stream.id === pickedStream.id && t === pickedT ? "picked" : relation(stream.slots[t], picked, stream);
              return {
                key: `${stream.id}-${s}`,
                text: `t${t + 1}`,
                state,
                onClick: stream.kind === "user" ? undefined : () => setPick({ stream: stream.id, t }),
              };
            }),
          }))}
        />
      )}

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[12px] text-ink-3">
        <Key state="picked">being predicted</Key>
        <Key state="known">already generated, visible</Key>
        <Key state="parallel">same step, not visible</Key>
        {layout === "moshi" && <Key state="heard">user audio heard so far</Key>}
        <Key state="later">later</Key>
        {layout !== "flatten" && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3.5 w-4 rounded-sm shadow-[inset_0_0_0_1px_var(--rule-strong)] [background:repeating-linear-gradient(135deg,transparent_0_3px,var(--rule)_3px_4px)]" />
            pad
          </span>
        )}
      </div>

      <p className="mt-3 text-[13px] leading-snug text-ink-2">
        Predicting {name(pickedStream)} of frame {pickedT + 1} at step {picked.step + 1}
        {layout === "moshi" ? `, place ${picked.depth + 1} in the depth transformer's order` : ""}: the model sees{" "}
        {knownCount} {knownCount === 1 ? "token" : "tokens"} of this sequence
        {parallelCount > 0 ? `, and ${parallelCount} more come out in the same step without seeing each other` : ""}.
        {sameFrameCoarser === true && " It sees the coarser codebooks of its own frame."}
        {sameFrameCoarser === false && " It cannot see the coarser codebooks of its own frame."}
      </p>

      <Readouts>
        <Readout label="Sequence length" value={positions} hint={layout === "flatten" ? "K × T positions" : "positions"} tone="demo" />
        <Readout
          label="Sequential steps"
          value={sequential}
          hint={
            layout === "flatten"
              ? "K × T"
              : layout === "parallel"
                ? "T"
                : layout === "delay"
                  ? "T + K − 1"
                  : "big steps + depth steps"
          }
        />
        <Readout label="Tokens a step" value={perStep} hint={layout === "moshi" ? "text + Moshi + user" : undefined} />
        <Readout
          label="Own frame's coarser codes"
          value={sameFrameCoarser === null ? "n/a" : sameFrameCoarser ? "visible" : "hidden"}
          tone={sameFrameCoarser === null ? "plain" : sameFrameCoarser ? "good" : "bad"}
          hint={sameFrameCoarser === null ? "pick a finer codebook" : undefined}
        />
      </Readouts>
    </DemoFrame>
  );
}

type Cell = { key: string; text: string; state: State | null; onClick?: () => void };

function Grid({
  columns,
  header,
  rows,
}: {
  columns: number;
  header: (i: number) => string;
  rows: { key: string; label: string; cells: Cell[] }[];
}) {
  return (
    <div
      className="grid gap-[3px]"
      style={{ gridTemplateColumns: `minmax(3.2rem, 5.4rem) repeat(${columns}, minmax(0, 1fr))` }}
    >
      <span />
      {Array.from({ length: columns }, (_, i) => (
        <span key={i} className="text-center font-mono text-[10px] text-ink-3">
          {header(i)}
        </span>
      ))}
      {rows.map((row) => (
        <Row key={row.key} label={row.label}>
          {row.cells.map((cell) =>
            cell.state === null ? (
              <span
                key={cell.key}
                className="h-7 rounded shadow-[inset_0_0_0_1px_var(--rule)] [background:repeating-linear-gradient(135deg,transparent_0_3px,var(--rule)_3px_4px)]"
              />
            ) : (
              <Clickable
                key={cell.key}
                onClick={cell.onClick}
                className={`flex h-7 min-w-0 items-center justify-center overflow-hidden rounded font-mono text-[10.5px] ${CELL[cell.state]}`}
              >
                {cell.text}
              </Clickable>
            ),
          )}
        </Row>
      ))}
    </div>
  );
}

/**
 * A grid cell that picks a token. It is a span with a button's role rather
 * than a button, because the book hides every button in a printed demo and
 * the grid is the figure.
 */
function Clickable({ onClick, className, children }: { onClick?: () => void; className: string; children: ReactNode }) {
  if (!onClick) return <span className={className}>{children}</span>;
  return (
    <span
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick();
        }
      }}
      className={`${className} cursor-pointer select-none focus-visible:outline-2 focus-visible:outline-[var(--demo)]`}
    >
      {children}
    </span>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <span className="self-center truncate pr-1 text-[11px] text-ink-3">{label}</span>
      {children}
    </>
  );
}

function Key({ state, children }: { state: State; children: ReactNode }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`inline-block h-3.5 w-4 rounded-sm ${CELL[state]}`} />
      {children}
    </span>
  );
}
