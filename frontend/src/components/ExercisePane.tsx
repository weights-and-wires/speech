import { useCallback, useEffect, useId, useRef, useState } from "react";

import { api, ApiError, runStream, type ExerciseDetail, type RunEvent } from "../lib/api";
import { useEditorLook } from "../lib/editorPrefs";
import { useTheme } from "../lib/theme";
import CodeEditor from "./CodeEditor";
import { useOutputCollapse } from "./CollapsibleOutput";
import EditorToolbar, { ToolbarPanel, type ToolbarAction } from "./EditorToolbar";
import Markdown from "./Markdown";
import RunConsole, { ConsoleHeader, type CheckResult, type ConsoleLine } from "./RunConsole";
import { modKey } from "./shellIcons";
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
  // Counts runs, so the output panel opens when one starts.
  const [runId, setRunId] = useState(0);

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
      setRunId((n) => n + 1);
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
      setHintsShown(0);
      setRevealed(body.solution);
    } catch {
      setRevealed("# The solution could not be loaded.");
    }
  }

  const mod = modKey();
  const look = useEditorLook(useTheme().resolved);
  const output = useOutputCollapse(runId);
  const consoleId = useId();
  const panelId = useId();

  const status =
    saved === "saving" ? "Saving…" : saved === "saved" ? "Saved" : saved === "failed" ? "Not saved" : "";

  const actions: ToolbarAction[] = [
    ...(exercise.hints.length > 0
      ? [
          {
            id: "hint",
            label: hintsShown ? "Hide hints" : "Show a hint",
            icon: "bulb" as const,
            pressed: hintsShown > 0,
            controls: panelId,
            onClick: () => {
              setRevealed(null);
              setHintsShown((n) => (n ? 0 : 1));
            },
          },
        ]
      : []),
    {
      id: "reset",
      label: "Reset to the starter file",
      icon: "reset",
      priority: "low",
      onClick: () => void reset(),
    },
    {
      id: "solution",
      label: revealed !== null ? "Hide the solution" : "Show the solution",
      icon: "key",
      priority: "low",
      pressed: revealed !== null,
      controls: panelId,
      onClick: () => void reveal(),
    },
    ...(gpuLesson
      ? [
          {
            id: "gpu",
            label: gpuAvailable ? (useGpu ? "Run on the CPU" : "Run on a GPU") : "No GPU endpoint is configured",
            tip: gpuAvailable
              ? "Send runs to a RunPod GPU worker. A cold worker takes a few seconds to start."
              : "No GPU endpoint is configured, so every run stays on the CPU.",
            icon: "gpu" as const,
            priority: "low" as const,
            pressed: useGpu && gpuAvailable,
            disabled: !gpuAvailable,
            onClick: () => setUseGpu((v) => !v),
          },
        ]
      : []),
  ];

  const header = (collapsed: boolean) => (
    <ConsoleHeader
      mode={mode}
      backend={backend}
      running={running}
      summary={summary}
      collapsed={collapsed}
      onToggle={output.toggle}
      controls={consoleId}
    />
  );

  return (
    <div className={`slab flex min-h-0 flex-1 flex-col ${look === "paper" ? "code-light sh-slab-paper" : ""}`}>
      <EditorToolbar
        title={exercise.title}
        info={<Markdown inline>{exercise.brief}</Markdown>}
        meta={`${exercise.timeout}s limit`}
        status={status}
        running={running}
        onRun={() => start("run")}
        onCheck={() => start("check")}
        onStop={stop}
        actions={actions}
        onCollapse={onCollapse}
        collapseShortcut={`${mod} .`}
      />

      {hintsShown > 0 && (
        <ToolbarPanel id={panelId} title="Hints" onClose={() => setHintsShown(0)}>
          <ol className="space-y-2">
            {exercise.hints.slice(0, hintsShown).map((hint, index) => (
              <li key={index} className="fade-in flex gap-2.5 text-[13.5px] leading-snug text-ink-2">
                <span className="w-12 shrink-0 font-mono text-[10px] font-medium uppercase tracking-[0.08em] text-ink-3">
                  Hint {index + 1}
                </span>
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
        </ToolbarPanel>
      )}

      {revealed !== null && (
        <ToolbarPanel id={panelId} title="Reference solution" onClose={() => setRevealed(null)}>
          <Markdown inline>{"```python\n" + revealed + "\n```"}</Markdown>
        </ToolbarPanel>
      )}

      <SplitPane
        direction="column"
        storageKey="speech.split.exercise"
        initial={62}
        min={25}
        max={85}
        className="min-h-0 flex-1"
        label="Resize the editor and the output"
        collapsed={output.collapsed}
        rail={header(true)}
        first={
          <div className="min-h-0 flex-1 overflow-hidden">
            <CodeEditor value={code} onChange={setCode} onRun={() => start("run")} onCheck={() => start("check")} />
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
            onToggle={output.toggle}
            bodyId={consoleId}
          />
        }
      />
    </div>
  );
}
