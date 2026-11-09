import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Link } from "react-router-dom";

import { api, type LessonMeta, type Part, type ProgressMap } from "../lib/api";
import { lessonNumbers, splitPart, titleOf } from "../lib/parts";

type Props = {
  parts: Part[];
  progress: ProgressMap;
};

type Me = Awaited<ReturnType<typeof api.me>>;

/**
 * The course overview: a hero with the reader's next lesson, then one module
 * card per part, each with its own progress strip and a way back in.
 */
export default function Dashboard({ parts, progress }: Props) {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    api.me().then(setMe).catch(() => undefined);
  }, []);

  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);
  const solved = lessons.filter((l) => progress[l.slug]?.status === "solved").length;
  const minutes = lessons.reduce((sum, l) => sum + (l.minutes ?? 0), 0);
  // The lesson touched most recently that is not yet solved, then the first
  // one never opened, then the first lesson.
  const next = useMemo(() => nextLesson(lessons, progress), [lessons, progress]);
  const resuming = next ? !!progress[next.slug] : false;
  const numbers = useMemo(() => lessonNumbers(lessons), [lessons]);
  const share = lessons.length ? (solved / lessons.length) * 100 : 0;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
      <div className="mx-auto max-w-[1320px] px-4 pb-16 pt-8 sm:px-8 lg:px-12 lg:pt-11">
        <section className="hero">
          <div className="min-w-0">
            <span className="kicker mb-4">Weights and wires · Speech and audio</span>
            <h1>
              Speech and audio, <em>one build</em> at a time
            </h1>
            <p className="lede">
              Each lesson explains one idea in plain words, works a small example by hand, and then
              walks you through building it. Your code runs against checks that say exactly what is
              right and what is not, and what it plays or plots appears beside its output.
            </p>
            <div className="hero-actions">
              {next && (
                <Link to={`/lessons/${next.slug}`} className="button button-primary">
                  {resuming ? "Continue" : "Start the course"}
                  <span aria-hidden>→</span>
                </Link>
              )}
              <Link to="/demos" className="button">
                All demos
              </Link>
            </div>
            <div className="hero-meta">
              <span>{lessons.length} lessons</span>
              <span>{parts.length} parts</span>
              {minutes > 0 && <span>About {Math.round(minutes / 60)} hours of reading</span>}
            </div>
          </div>

          <div className="min-w-0 rounded-[7px] border border-line bg-surface p-5 sm:p-6">
            <div className="flex items-baseline justify-between gap-3">
              <span className="kicker">Your progress</span>
              <span className="meta tabular-nums">
                {solved} / {lessons.length} solved
              </span>
            </div>
            <div className="progress-track mt-3">
              <div className="progress-fill" style={{ width: `${share}%` }} />
            </div>
            {me && me.stats.runs > 0 && (
              <p className="meta mt-2.5">
                {me.stats.runs} runs · {formatSeconds(me.stats.compute_seconds)} of compute
              </p>
            )}

            {next && (
              <Link
                to={`/lessons/${next.slug}`}
                className="group mt-5 block rounded-[6px] border border-line bg-paper p-4 transition-colors hover:border-forest hover:bg-forest-soft"
              >
                <span className="meta block">
                  {resuming ? "Continue where you left off" : "Start here"} · Lesson{" "}
                  {numbers.get(next.slug)}
                </span>
                <span className="mt-1.5 block font-serif text-[22px] font-medium leading-tight tracking-[-0.025em] text-ink">
                  {next.title}
                </span>
                <span className="mt-1.5 block text-[13.5px] leading-relaxed text-muted">{next.summary}</span>
                <span className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-semibold text-forest">
                  {resuming ? "Continue" : "Start lesson"}
                  <span aria-hidden className="transition-transform group-hover:translate-x-0.5">
                    →
                  </span>
                </span>
              </Link>
            )}
          </div>
        </section>

        <div className="section-heading">
          <div>
            <span className="kicker">Modules</span>
            <h2 className="mt-1.5">The course, part by part</h2>
            <p>
              Pick up any part where you left off.
            </p>
          </div>
        </div>

        <div className="module-grid">
          {parts.map((part, index) => (
            <ModuleCard
              key={part.part}
              part={part}
              index={index}
              progress={progress}
              numbers={numbers}
            />
          ))}
        </div>

        {me && me.runs.length > 0 && (
          <>
            <div className="section-heading">
              <div>
                <span className="kicker">Activity</span>
                <h2 className="mt-1.5">Recent runs</h2>
              </div>
            </div>
            <ul className="overflow-hidden rounded-[7px] border border-line bg-surface">
              {me.runs.slice(0, 8).map((run, index) => (
                <li key={index} className="border-b border-line last:border-b-0">
                  <Link
                    to={`/lessons/${run.exercise_id}`}
                    className="flex items-center gap-3 px-4 py-2.5 text-[13.5px] transition-colors hover:bg-forest-soft"
                  >
                    <span
                      className={`size-1.5 shrink-0 rounded-full ${run.ok ? "bg-good" : "bg-bad"}`}
                      aria-label={run.ok ? "Succeeded" : "Failed"}
                    />
                    <span className="min-w-0 flex-1 truncate text-ink">
                      {titleOf(run.exercise_id, lessons)}
                    </span>
                    <span className="meta shrink-0 tabular-nums">
                      {run.mode === "check" && run.total != null
                        ? `${run.passed ?? 0}/${run.total} checks`
                        : run.ok
                          ? "Ran"
                          : "Run failed"}
                    </span>
                    {run.backend === "runpod" && (
                      <span className="chip chip-forest hidden shrink-0 sm:inline-flex">GPU</span>
                    )}
                    <span className="meta hidden w-16 shrink-0 text-right tabular-nums sm:inline">
                      {ago(run.started_at)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

/** One part as a card: its lessons, how many are solved, and where to resume. */
function ModuleCard({
  part,
  index,
  progress,
  numbers,
}: {
  part: Part;
  index: number;
  progress: ProgressMap;
  numbers: Map<string, string>;
}) {
  const { number, name } = splitPart(part.part);
  const label = number || String(index + 1);
  const done = part.lessons.filter((l) => progress[l.slug]?.status === "solved").length;
  const total = part.lessons.length;
  const share = total ? Math.round((done / total) * 100) : 0;
  const resume = nextLesson(part.lessons, progress);
  const first = part.lessons[0];
  const last = part.lessons[part.lessons.length - 1];
  const range =
    first && last && first !== last
      ? `Lessons ${numbers.get(first.slug)} to ${numbers.get(last.slug)}`
      : first
        ? `Lesson ${numbers.get(first.slug)}`
        : "";
  const started = part.lessons.some((l) => progress[l.slug]);
  const complete = total > 0 && done === total;

  return (
    <article className="module-card" style={{ "--progress": `${share}%` } as CSSProperties}>
      <div className="module-card-header">
        <span className="module-number">{number ? `Part ${label.padStart(2, "0")}` : name}</span>
        <span className="module-icon" aria-hidden>
          {label.padStart(2, "0")}
        </span>
      </div>
      <h3>
        <Link to={`/lessons/${(resume ?? first)?.slug ?? ""}`} className="hover:text-forest">
          {name}
        </Link>
      </h3>
      <p className="line-clamp-3">
        {part.lessons
          .slice(0, 4)
          .map((lesson) => lesson.title)
          .join(" · ")}
        {part.lessons.length > 4 ? " · …" : ""}
      </p>
      <div className="module-card-footer">
        <span className="module-count">
          {range} · {done}/{total} solved
        </span>
        {resume && (
          <Link
            to={`/lessons/${resume.slug}`}
            className="button min-h-8! px-3! py-1! text-[12px]!"
            aria-label={`${complete ? "Review" : started ? "Continue" : "Start"} ${name}`}
          >
            {complete ? "Review" : started ? "Continue" : "Start"}
            <span aria-hidden>→</span>
          </Link>
        )}
      </div>
      <div
        className="module-progress"
        role="progressbar"
        aria-label={`${name}: lessons solved`}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
      />
    </article>
  );
}

/**
 * The lesson to open next: the one touched most recently that is not yet
 * solved, then the first one never opened, then the first lesson.
 */
function nextLesson(lessons: LessonMeta[], progress: ProgressMap): LessonMeta | undefined {
  const started = lessons
    .filter((l) => progress[l.slug]?.status === "started")
    .sort((a, b) => (progress[b.slug]?.updated_at ?? 0) - (progress[a.slug]?.updated_at ?? 0));
  return started[0] ?? lessons.find((l) => !progress[l.slug]) ?? lessons[0];
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
