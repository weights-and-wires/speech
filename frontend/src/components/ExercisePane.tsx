import { useCallback, useEffect, useRef, useState } from "react";

import { api, ApiError, runStream, type ExerciseDetail, type RunEvent } from "../lib/api";
import CodeEditor from "./CodeEditor";
import Markdown from "./Markdown";
import RunConsole, { type CheckResult, type ConsoleLine } from "./RunConsole";
import SplitPane from "./SplitPane";

type Props = {
  exercise: ExerciseDetail;
  initialCode: string;
  /** Whether a RunPod endpoint is configured at all. */
  gpuAvailable: boolean;
  /** Whether this lesson's work is worth sending to a GPU. */
  gpuLesson: boolean;
  /** The next lesson, offered once every check passes. */
  next?: { slug: string; title: string } | null;
  onSolved: () => void;
  /** Folds the editor away so the lesson takes the whole width. */
  onCollapse?: () => void;
};

const AUTOSAVE_MS = 1200;

export default function ExercisePane({
  exercise,
  initialCode,
  gpuAvailable,
  gpuLesson,
  next,
  onSolved,
  onCollapse,
}: Props) {
  const [code, setCode] = useState(initialCode);
  const [lines, setLines] = useState<ConsoleLine[]>([]);
  const [checks, setChecks] = useState<CheckResult[]>([]);
  const [planned, setPlanned] = useState(0);
  const [running, setRunning] = useState(false);
  const [mode, setMode] = useState<"run" | "check" | null>(null);
  const [backend, setBackend] = useState<string | null>(null);
  const [summary, setSummary] = useState<Extract<RunEvent, { type: "done" }> | null>(
    null,
  );
  // How many hints are showing. They come one at a time, so a reader who
  // needs only a nudge is not handed the whole answer.
  const [hintsShown, setHintsShown] = useState(0);
  // Checks are sized for the CPU and a cold serverless worker costs more than
  // the whole suite, so the GPU is opt-in and only where a lesson asks for it.
  const [useGpu, setUseGpu] = useState(false);
  const [revealed, setRevealed] = useState<string | null>(null);
  const [saved, setSaved] = useState<"idle" | "saving" | "saved" | "failed">("idle");

  const abort = useRef<AbortController | null>(null);
  const codeRef = useRef(code);
  codeRef.current = code;
  const gpuRef = useRef(useGpu);
  gpuRef.current = useGpu;

  // A new exercise resets everything the previous one left behind.
  useEffect(() => {
    setCode(initialCode);
    setLines([]);
    setChecks([]);
    setPlanned(0);
    setSummary(null);
    setMode(null);
    setBackend(null);
    setHintsShown(0);
    setUseGpu(false);
    setRevealed(null);
    setSaved("idle");
  }, [exercise.id, initialCode]);

  // Autosave the draft, so a closed tab never loses work.
  useEffect(() => {
    if (code === initialCode) return;
    setSaved("saving");
    const timer = window.setTimeout(() => {
      api
        .saveDraft(exercise.id, code)
        .then(() => setSaved("saved"))
        .catch(() => setSaved("failed"));
    }, AUTOSAVE_MS);
    return () => window.clearTimeout(timer);
  }, [code, exercise.id, initialCode]);

  useEffect(() => () => abort.current?.abort(), []);

  const start = useCallback(
    async (next: "run" | "check") => {
      abort.current?.abort();
      const controller = new AbortController();
      abort.current = controller;

      setRunning(true);
      setMode(next);
      setLines([]);
      setChecks([]);
      setPlanned(0);
      setSummary(null);
      setBackend(null);

      // The server records a solve after it sends `done` and before it closes
      // the stream, so progress is refreshed once the stream has closed.
      let solvedNow = false;
      try {
        for await (const event of runStream(
          {
            exercise_id: exercise.id,
            mode: next,
            code: codeRef.current,
            prefer_gpu: gpuRef.current,
          },
          controller.signal,
        )) {
          switch (event.type) {
            case "begin":
              setBackend(event.backend);
              break;
            case "plan":
              setPlanned(event.total);
              break;
            case "check":
              setChecks((current) => [...current, event]);
              break;
            case "status":
              setLines((current) => [...current, { kind: "status", text: event.text }]);
              break;
            case "stdout":
            case "stderr":
              setLines((current) => [
                ...current,
                { kind: event.type, text: event.text },
              ]);
              break;
            case "audio":
            case "figure":
              setLines((current) => [...current, { kind: "media", event }]);
              break;
            case "error":
              setLines((current) => [
                ...current,
                { kind: "error", text: event.message, detail: event.detail },
              ]);
              break;
            case "done":
              setSummary(event);
              if (next === "check" && event.ok) solvedNow = true;
              break;
          }
        }
        if (solvedNow) onSolved();
      } catch (exc) {
        if (controller.signal.aborted) return;
        setLines((current) => [
          ...current,
          {
            kind: "error",
            text:
              exc instanceof ApiError
                ? exc.message
                : "Lost the connection to the runner.",
          },
        ]);
      } finally {
        if (!controller.signal.aborted) setRunning(false);
      }
    },
    [exercise.id, onSolved],
  );

  function stop() {
    abort.current?.abort();
    setRunning(false);
    setLines((current) => [...current, { kind: "status", text: "Stopped." }]);
  }

  async function reset() {
    if (!window.confirm("Replace your code with the starter file?")) return;
    await api.resetDraft(exercise.id).catch(() => undefined);
    setCode(exercise.starter);
  }

  async function reveal() {
    if (revealed) {
      setRevealed(null);
      return;
    }
    if (
      !window.confirm(
        "Show the reference solution? Your own code stays where it is.",
      )
    ) {
      return;
    }
    try {
      const body = await api.solution(exercise.id);
      setRevealed(body.solution);
    } catch {
      setRevealed("# The solution could not be loaded.");
    }
  }

  const mac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);
  const mod = mac ? "⌘" : "Ctrl";

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-canvas">
      <header className="shrink-0 border-b border-rule bg-chrome px-3 pb-2.5 pt-2.5">
        <div className="flex items-baseline gap-2">
          <h2 className="min-w-0 flex-1 truncate text-[14px] font-semibold text-ink">
            {exercise.title}
          </h2>
          <span className="shrink-0 text-[12px] text-ink-4">
            {exercise.timeout}s limit
          </span>
          {onCollapse && (
            <button
              type="button"
              onClick={onCollapse}
              aria-label="Hide the code editor"
              title={`Hide the code editor (${mod} .)`}
              className="icon-btn -my-1 -mr-1.5 self-center"
            >
              <FoldIcon />
            </button>
          )}
        </div>
        <p className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3" title={exercise.brief}>
          {exercise.brief}
        </p>

        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
          {running ? (
            <button type="button" onClick={stop} className="btn btn-danger">
              <span className="size-2 rounded-[2px] bg-current" aria-hidden />
              Stop
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => start("run")}
                className="btn btn-secondary"
                title={`Run the file (${mod} Return)`}
              >
                Run
                <span className="kbd hidden lg:inline-flex">{mod}↵</span>
              </button>
              <button
                type="button"
                onClick={() => start("check")}
                className="btn btn-primary"
                title={`Run the checks (${mod} Shift Return)`}
              >
                Check
                <span className="kbd hidden lg:inline-flex">{mac ? "⇧⌘↵" : "Ctrl⇧↵"}</span>
              </button>

              {gpuLesson ? (
                <label
                  title={
                    gpuAvailable
                      ? "Send this run to a RunPod GPU worker. A cold worker takes " +
                        "a few seconds to start, so the checks are quicker on the CPU."
                      : "No GPU endpoint is configured, so every run stays on the CPU."
                  }
                  className={`ml-1 flex min-h-8 items-center gap-2 rounded-md px-2 text-[13px] transition-colors ${
                    gpuAvailable ? "cursor-pointer text-ink-2 hover:text-ink" : "cursor-not-allowed text-ink-4"
                  }`}
                >
                  <input
                    type="checkbox"
                    className="size-3.5 accent-[var(--primary)]"
                    checked={useGpu && gpuAvailable}
                    disabled={!gpuAvailable}
                    onChange={(event) => setUseGpu(event.target.checked)}
                  />
                  Use a GPU
                </label>
              ) : null}
            </>
          )}

          <div className="flex-1" />

          <span
            className={`mr-1 text-[12px] ${saved === "failed" ? "text-bad" : "text-ink-4"}`}
            aria-live="polite"
          >
            {saved === "saving"
              ? "Saving…"
              : saved === "saved"
                ? "Saved"
                : saved === "failed"
                  ? "Not saved"
                  : ""}
          </span>
          {exercise.hints.length > 0 && (
            <button
              type="button"
              onClick={() => setHintsShown((n) => (n ? 0 : 1))}
              aria-expanded={hintsShown > 0}
              className="btn btn-ghost px-2.5!"
            >
              {hintsShown ? "Hide hints" : "Hint"}
            </button>
          )}
          <button type="button" onClick={reset} className="btn btn-ghost px-2.5!" title="Replace your code with the starter file">
            Reset
          </button>
          <button
            type="button"
            onClick={reveal}
            aria-pressed={revealed !== null}
            className="btn btn-ghost px-2.5!"
          >
            Solution
          </button>
        </div>

        {hintsShown > 0 && (
          <div className="fade-in mt-2.5 rounded-lg bg-raise p-3 shadow-[inset_0_0_0_1px_var(--rule)]">
            <ol className="space-y-2">
              {exercise.hints.slice(0, hintsShown).map((hint, index) => (
                <li key={index} className="fade-in flex gap-2.5 text-[13.5px] leading-snug text-ink-2">
                  <span className="w-12 shrink-0 text-[12px] font-medium text-ink-4">Hint {index + 1}</span>
                  <span>{hint}</span>
                </li>
              ))}
            </ol>
            {hintsShown < exercise.hints.length && (
              <button
                type="button"
                onClick={() => setHintsShown((n) => n + 1)}
                className="mt-2.5 text-[12.5px] font-medium text-accent hover:underline"
              >
                Show hint {hintsShown + 1} of {exercise.hints.length}
              </button>
            )}
          </div>
        )}
      </header>

      {revealed !== null ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center gap-2 border-b border-rule px-3 py-1.5">
            <span className="flex-1 text-[13px] font-medium text-ink-2">Reference solution</span>
            <button type="button" onClick={() => setRevealed(null)} className="btn btn-secondary min-h-7!">
              Back to your code
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-auto overscroll-contain p-3">
            <Markdown inline>{"```python\n" + revealed + "\n```"}</Markdown>
          </div>
        </div>
      ) : (
        <SplitPane
          direction="column"
          storageKey={`speech.split.exercise`}
          initial={62}
          min={25}
          max={85}
          className="min-h-0 flex-1"
          label="Resize the editor and the output"
          first={
            <div className="min-h-0 flex-1 overflow-hidden">
              <CodeEditor
                value={code}
                onChange={setCode}
                onRun={() => start("run")}
                onCheck={() => start("check")}
              />
            </div>
          }
          second={
            <RunConsole
              lines={lines}
              checks={checks}
              planned={planned}
              running={running}
              mode={mode}
              backend={backend}
              summary={summary}
              next={next}
              mod={mod}
            />
          }
        />
      )}
    </div>
  );
}

/** A pane with an arrow pushing it off to the right. */
function FoldIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      <rect x="1.75" y="2.75" width="12.5" height="10.5" rx="2" stroke="currentColor" strokeWidth="1.3" />
      <path d="M9.75 3v10" stroke="currentColor" strokeWidth="1.3" />
      <path d="M5 6l2 2-2 2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
