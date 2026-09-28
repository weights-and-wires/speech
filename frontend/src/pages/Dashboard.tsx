import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import StatusMark from "../components/StatusMark";
import { api, type Part, type ProgressMap } from "../lib/api";
import { splitPart, titleOf } from "../lib/parts";

type Props = { parts: Part[]; progress: ProgressMap };

type Me = Awaited<ReturnType<typeof api.me>>;

export default function Dashboard({ parts, progress }: Props) {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    api.me().then(setMe).catch(() => undefined);
  }, []);

  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);
  const solved = lessons.filter((l) => progress[l.slug]?.status === "solved").length;
  // The lesson touched most recently that is not yet solved, then the first
  // one never opened, then the first lesson.
  const next = useMemo(() => {
    const started = lessons
      .filter((l) => progress[l.slug]?.status === "started")
      .sort((a, b) => (progress[b.slug]?.updated_at ?? 0) - (progress[a.slug]?.updated_at ?? 0));
    return started[0] ?? lessons.find((l) => !progress[l.slug]) ?? lessons[0];
  }, [lessons, progress]);
  const resuming = next ? !!progress[next.slug] : false;
  const nextNumber = next ? lessons.indexOf(next) + 1 : 0;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-3xl px-5 pb-16 pt-10 sm:px-8 lg:pt-14">
        <header>
          <h1 className="font-serif text-[2.1rem] font-semibold leading-[1.15] tracking-[-0.015em] text-ink sm:text-[2.5rem]">
            Speech and audio, one build at a time
          </h1>
          <p className="mt-3 max-w-[38rem] font-serif text-[1.08rem] leading-relaxed text-ink-3">
            Each lesson explains one idea in plain words, works a small example
            by hand, and then walks you through building it. Your code runs
            against checks that say exactly what is right and what is not, and
            what it plays or plots appears beside its output.
          </p>

          {lessons.length > 0 && (
            <div className="mt-7 max-w-md">
              <div className="flex items-baseline justify-between gap-3 text-[13px]">
                <span className="text-ink-2">
                  <span className="font-semibold tabular-nums text-ink">{solved}</span> of{" "}
                  {lessons.length} lessons solved
                </span>
                {me && me.stats.runs > 0 && (
                  <span className="text-ink-4 tabular-nums">
                    {me.stats.runs} runs · {formatSeconds(me.stats.compute_seconds)} of compute
                  </span>
                )}
              </div>
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-rule">
                <div
                  className="h-full rounded-full bg-good transition-[width] duration-500"
                  style={{ width: `${(solved / lessons.length) * 100}%` }}
                />
              </div>
            </div>
          )}
        </header>

        {next && (
          <Link
            to={`/lessons/${next.slug}`}
            className="group mt-9 flex flex-col gap-4 rounded-xl border border-rule bg-raise p-5 transition-colors hover:border-rule-strong sm:flex-row sm:items-center sm:gap-6"
          >
            <span className="min-w-0 flex-1">
              <span className="block text-[12.5px] font-medium text-ink-3">
                {resuming ? "Continue where you left off" : "Start here"} · Lesson {nextNumber}
              </span>
              <span className="mt-1 block font-serif text-[1.35rem] font-semibold leading-snug text-ink">
                {next.title}
              </span>
              <span className="mt-1 block text-[14px] leading-snug text-ink-3">{next.summary}</span>
            </span>
            <span className="btn btn-primary self-start sm:self-center">
              {resuming ? "Continue" : "Start lesson"}
              <span aria-hidden className="transition-transform group-hover:translate-x-0.5">
                →
              </span>
            </span>
          </Link>
        )}

        <div className="mt-12 space-y-11">
          {parts.map((part) => {
            const { number, name } = splitPart(part.part);
            const done = part.lessons.filter((l) => progress[l.slug]?.status === "solved").length;
            return (
              <section key={part.part} aria-labelledby={`part-${number || name}`}>
                <div className="flex items-baseline gap-3 border-b border-rule-strong pb-2">
                  <h2 id={`part-${number || name}`} className="min-w-0 flex-1 font-serif text-[1.2rem] font-semibold text-ink">
                    {number && <span className="mr-2 text-ink-4">Part {number}</span>}
                    {name}
                  </h2>
                  <span className="text-[12.5px] tabular-nums text-ink-4">
                    {done} of {part.lessons.length}
                  </span>
                </div>
                <ol>
                  {part.lessons.map((lesson) => {
                    const state = progress[lesson.slug]?.status;
                    return (
                      <li key={lesson.slug} className="border-b border-rule">
                        <Link
                          to={`/lessons/${lesson.slug}`}
                          className="group -mx-3 flex items-start gap-3 rounded-md px-3 py-3.5 transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_4%,transparent)]"
                        >
                          <span className="w-6 shrink-0 pt-0.5 text-[13px] tabular-nums text-ink-4">
                            {lessons.indexOf(lesson) + 1}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-[15px] font-medium leading-snug text-ink group-hover:text-accent">
                              {lesson.title}
                            </span>
                            <span className="mt-0.5 block text-[13.5px] leading-snug text-ink-3">
                              {lesson.summary}
                            </span>
                          </span>
                          <span className="flex shrink-0 items-center gap-3 pt-0.5">
                            {lesson.minutes ? (
                              <span className="hidden text-[12.5px] tabular-nums text-ink-4 sm:inline">
                                {lesson.minutes} min
                              </span>
                            ) : null}
                            <StatusMark state={state} />
                          </span>
                        </Link>
                      </li>
                    );
                  })}
                </ol>
              </section>
            );
          })}
        </div>

        {me && me.runs.length > 0 && (
          <section className="mt-14">
            <h2 className="border-b border-rule-strong pb-2 font-serif text-[1.2rem] font-semibold text-ink">
              Recent runs
            </h2>
            <ul>
              {me.runs.slice(0, 8).map((run, index) => (
                <li key={index} className="border-b border-rule">
                  <Link
                    to={`/lessons/${run.exercise_id}`}
                    className="-mx-3 flex items-center gap-3 rounded-md px-3 py-2.5 text-[13.5px] transition-colors hover:bg-[color-mix(in_oklab,var(--ink)_4%,transparent)]"
                  >
                    <span
                      className={`size-1.5 shrink-0 rounded-full ${run.ok ? "bg-good" : "bg-bad"}`}
                      aria-label={run.ok ? "Succeeded" : "Failed"}
                    />
                    <span className="min-w-0 flex-1 truncate text-ink-2">
                      {titleOf(run.exercise_id, lessons)}
                    </span>
                    <span className="shrink-0 tabular-nums text-ink-3">
                      {run.mode === "check" && run.total != null
                        ? `${run.passed ?? 0}/${run.total} checks`
                        : run.ok
                          ? "Ran"
                          : "Run failed"}
                    </span>
                    {run.backend === "runpod" && (
                      <span className="hidden shrink-0 text-[12px] text-ink-4 sm:inline">GPU</span>
                    )}
                    <span className="w-16 shrink-0 text-right text-[12.5px] tabular-nums text-ink-4">
                      {ago(run.started_at)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}

function formatSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(0)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

/** Seconds since the epoch → "just now", "5 min ago", "3 h ago", "2 d ago". */
function ago(epochSeconds: number): string {
  const seconds = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}
