import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import type { MediaEvent, RunEvent } from "../lib/api";
import RunMedia from "./figures/RunMedia";

export type ConsoleLine =
  | { kind: "stdout" | "stderr" | "status"; text: string }
  | { kind: "error"; text: string; detail?: string | null }
  | { kind: "media"; event: MediaEvent };

export type CheckResult = Extract<RunEvent, { type: "check" }>;

type Props = {
  lines: ConsoleLine[];
  checks: CheckResult[];
  planned: number;
  running: boolean;
  mode: "run" | "check" | null;
  backend: string | null;
  summary: Extract<RunEvent, { type: "done" }> | null;
  /** The next lesson, offered once every check passes. */
  next?: { slug: string; title: string } | null;
  /** The modifier key's name on this platform, for the empty-state hint. */
  mod: string;
};

/**
 * The bottom half of the exercise pane: a transcript for **Run**, and a list of
 * checks for **Check**. Both scroll to the newest line while a run is going.
 *
 * Clips and figures from `speechlab` sit in the transcript in the order they
 * arrived, between the prints around them; in check mode they follow the list
 * of checks and say which check produced them.
 */
export default function RunConsole({
  lines,
  checks,
  planned,
  running,
  mode,
  backend,
  summary,
  next,
  mod,
}: Props) {
  const bottom = useRef<HTMLDivElement>(null);
  const top = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(true);
  const solved = mode === "check" && !running && !!summary?.ok;
  // The first failure is the one to read, so it opens by itself.
  const firstFailed = checks
    .filter((c) => c.status === "failed")
    .reduce<number | null>((low, c) => (low === null || c.index < low ? c.index : low), null);

  // While checks stream in, follow the newest one. Once they finish, go back
  // to the top: the result and the first failure are what the reader wants,
  // not the tail of the list.
  const finished = mode === "check" && !running && !!summary && planned > 0;

  useEffect(() => {
    if (finished) top.current?.scrollIntoView({ block: "start" });
  }, [finished]);

  useEffect(() => {
    if (stuck && !finished) bottom.current?.scrollIntoView({ block: "end" });
  }, [lines.length, checks.length, stuck, summary, finished]);

  const idle = !running && lines.length === 0 && checks.length === 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-canvas">
      <header className="flex h-9 shrink-0 items-center gap-2.5 border-b border-rule px-3">
        <span className="font-mono text-[10px] font-semibold uppercase tracking-[0.12em] text-ink-2">
          {mode === "check" ? "Checks" : "Output"}
        </span>
        {backend && (
          <span className="text-[12px] text-ink-4">
            {backend === "runpod" ? "on a RunPod GPU" : "on the CPU"}
          </span>
        )}
        {running && (
          <span className="flex items-center gap-1.5 text-[12px] text-ink-3">
            <Spinner />
            Running
          </span>
        )}
        <div className="flex-1" />
        {summary && !running && <Summary summary={summary} mode={mode} />}
      </header>

      <div
        className="min-h-0 min-w-0 flex-1 overflow-y-auto overscroll-contain px-3 py-2.5 font-mono text-[12.5px] leading-relaxed"
        onScroll={(event) => {
          const el = event.currentTarget;
          setStuck(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
        }}
      >
        {idle && (
          <div className="px-1 py-5 font-sans text-[13px] leading-relaxed text-ink-3">
            <p>
              <span className="font-medium text-ink-2">Run</span> executes your file and
              shows what it prints. <span className="font-medium text-ink-2">Check</span>{" "}
              tests it against the exercise and explains any failure.
            </p>
            <p className="mt-1.5 hidden text-ink-4 lg:block">
              Shortcuts: <span className="kbd">{mod}↵</span> to run,{" "}
              <span className="kbd">{mod}⇧↵</span> to check.
            </p>
          </div>
        )}

        <div ref={top} />
        {mode === "check" && planned > 0 && (
          <Scoreboard
            passed={checks.filter((c) => c.status !== "failed").length}
            done={checks.length}
            planned={planned}
            solved={solved}
            next={next}
          />
        )}

        {mode === "check" && planned > 0 && (
          <ol className="mb-3 space-y-1">
            {Array.from({ length: planned }).map((_, index) => {
              const result = checks.find((c) => c.index === index);
              return (
                <CheckRow
                  key={index}
                  index={index}
                  result={result}
                  running={running}
                  openFirst={index === firstFailed}
                />
              );
            })}
          </ol>
        )}

        {lines.map((line, index) => (
          <Line key={index} line={line} />
        ))}

        <div ref={bottom} />
      </div>
    </div>
  );
}

/** How many checks pass so far, and a way forward once they all do. */
function Scoreboard({
  passed,
  done,
  planned,
  solved,
  next,
}: {
  passed: number;
  done: number;
  planned: number;
  solved: boolean;
  next?: { slug: string; title: string } | null;
}) {
  if (solved) {
    return (
      <div className="fade-in mb-3 rounded-lg border border-[color-mix(in_oklab,var(--good)_35%,var(--rule))] bg-[color-mix(in_oklab,var(--good)_7%,var(--canvas))] px-3.5 py-3 font-sans">
        <p className="flex items-center gap-2 text-[14px] font-semibold text-ink">
          <svg viewBox="0 0 16 16" className="size-4 text-good" aria-hidden>
            <circle cx="8" cy="8" r="7.5" fill="currentColor" />
            <path d="M4.8 8.3 7 10.4l4.2-4.6" stroke="var(--canvas)" strokeWidth="1.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          All {planned} checks pass
        </p>
        <p className="mt-1 text-[13px] leading-snug text-ink-3">
          The lesson is marked solved.
          {next ? " The next one builds on what you just wrote." : " That was the last lesson."}
        </p>
        {next && (
          <Link to={`/lessons/${next.slug}`} className="btn btn-primary mt-3">
            Next: {next.title}
            <span aria-hidden>→</span>
          </Link>
        )}
      </div>
    );
  }
  const failing = done - passed;
  return (
    <div className="mb-3 px-0.5 font-sans">
      <div className="mb-1.5 flex items-center justify-between gap-3 text-[12.5px]">
        <span className="tabular-nums text-ink-2">
          {passed} of {planned} passing
        </span>
        {failing > 0 && done === planned && (
          <span className="text-ink-3">
            {failing === 1 ? "One check fails" : `${failing} checks fail`}. The first is open below.
          </span>
        )}
      </div>
      <div className="flex h-1 gap-0.5 overflow-hidden rounded-full">
        {Array.from({ length: planned }).map((_, i) => (
          <span
            key={i}
            className={`flex-1 transition-colors duration-300 ${
              i < passed ? "bg-good" : i < done ? "bg-bad" : "bg-rule"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

function CheckRow({
  index,
  result,
  running,
  openFirst,
}: {
  index: number;
  result?: CheckResult;
  running: boolean;
  openFirst: boolean;
}) {
  const [open, setOpen] = useState<boolean | null>(null);
  // A new run clears the results, and with them the reader's last choice.
  useEffect(() => {
    if (!result) setOpen(null);
  }, [result]);
  const failed = result?.status === "failed";
  const pending = !result;
  const expandable = !!(result?.message || result?.output);
  // Until the reader chooses, the first failure is open and the rest are shut.
  const shown = open ?? (openFirst && expandable);

  return (
    <li
      className={`fade-in rounded-md ${
        failed ? "bg-[color-mix(in_oklab,var(--bad)_6%,transparent)]" : ""
      }`}
    >
      <button
        type="button"
        disabled={!expandable}
        aria-expanded={expandable ? shown : undefined}
        onClick={() => setOpen(!shown)}
        className="flex w-full items-start gap-2.5 rounded-md px-2 py-1.5 text-left enabled:hover:bg-sunk"
      >
        <Glyph pending={pending} failed={failed} running={running} />
        <span
          className={`min-w-0 flex-1 font-sans text-[13px] leading-snug ${
            pending ? "text-ink-4" : failed ? "text-ink" : "text-ink-2"
          }`}
        >
          {result?.description ?? `Check ${index + 1}`}
        </span>
        {result && (
          <span className="shrink-0 pt-px text-[11px] tabular-nums text-ink-4">
            {result.seconds < 0.01 ? "<10 ms" : `${(result.seconds * 1000).toFixed(0)} ms`}
          </span>
        )}
      </button>

      {shown && result?.message && (
        <pre className="mx-2 mb-2 ml-8 overflow-x-auto whitespace-pre-wrap border-l-2 border-bad pl-3 text-[12px] leading-relaxed text-bad">
          {result.message}
        </pre>
      )}
      {shown && result?.output && (
        <pre className="mx-2 mb-2 ml-8 overflow-x-auto whitespace-pre-wrap border-l-2 border-rule-strong pl-3 text-[12px] leading-relaxed text-ink-3">
          {result.output}
        </pre>
      )}
      {!shown && failed && result?.message && (
        <p className="-mt-1 truncate pb-1.5 pl-8 pr-2 font-sans text-[12px] text-bad">
          {result.message.split("\n")[0]}
        </p>
      )}
    </li>
  );
}

function Glyph({
  pending,
  failed,
  running,
}: {
  pending: boolean;
  failed: boolean;
  running: boolean;
}) {
  if (pending) {
    return running ? (
      <span className="mt-0.5 flex size-3.5 shrink-0 items-center justify-center text-ink-4">
        <Spinner />
      </span>
    ) : (
      <span className="mt-[3px] size-3 shrink-0 rounded-full border border-rule-strong" />
    );
  }
  return failed ? (
    <svg viewBox="0 0 16 16" className="mt-0.5 size-3.5 shrink-0 text-bad" role="img" aria-label="Failed">
      <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  ) : (
    <svg viewBox="0 0 16 16" className="mt-0.5 size-3.5 shrink-0 text-good" role="img" aria-label="Passed">
      <path d="M3.5 8.5l3 3 6-7" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function Line({ line }: { line: ConsoleLine }) {
  if (line.kind === "media") return <RunMedia event={line.event} />;
  if (line.kind === "status") {
    return <p className="fade-in py-px font-sans text-[12px] text-ink-4">{line.text}</p>;
  }
  if (line.kind === "error") {
    return (
      <div className="fade-in my-2 rounded-md bg-[color-mix(in_oklab,var(--bad)_7%,transparent)] px-3 py-2">
        <p className="font-sans text-[13px] font-medium text-bad">{line.text}</p>
        {line.detail && (
          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[12px] text-bad">{line.detail}</pre>
        )}
      </div>
    );
  }
  return (
    <pre
      className={`whitespace-pre-wrap break-words ${
        line.kind === "stderr" ? "text-warn" : "text-ink"
      }`}
    >
      {line.text || " "}
    </pre>
  );
}

function Summary({
  summary,
  mode,
}: {
  summary: Extract<RunEvent, { type: "done" }>;
  mode: "run" | "check" | null;
}) {
  const seconds = summary.seconds ? ` in ${summary.seconds.toFixed(1)}s` : "";
  if (mode === "check" && summary.total != null) {
    return (
      <span className={`text-[12px] tabular-nums ${summary.ok ? "text-good" : "text-bad"}`}>
        {summary.passed} of {summary.total} passed{seconds}
      </span>
    );
  }
  return (
    <span className={`text-[12px] tabular-nums ${summary.ok ? "text-ink-3" : "text-bad"}`}>
      {summary.timeout ? "Timed out" : summary.ok ? "Finished" : "Failed"}
      {seconds}
    </span>
  );
}

function Spinner() {
  return (
    <svg viewBox="0 0 16 16" className="size-3 animate-spin" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" fill="none" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" />
    </svg>
  );
}
