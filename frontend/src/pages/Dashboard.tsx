import { useEffect, useMemo, useState } from "react";

import { api, type Part, type ProgressMap } from "../lib/api";
import { DEMOS } from "../demos";
import { lessonNumbers, splitPart, titleOf } from "../lib/parts";
import { nextLesson, partNumber, solvedCount } from "../lib/shellData";
import Overview, { OverviewBlock, OverviewList } from "./Overview";

type Props = {
  parts: Part[];
  progress: ProgressMap;
};

type Me = Awaited<ReturnType<typeof api.me>>;

/**
 * The speech course's overview at "/": the shared Overview page filled from
 * the lesson index, with the reader's recent runs after the part cards.
 */
export default function Dashboard({ parts, progress }: Props) {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    api.me().then(setMe).catch(() => undefined);
  }, []);

  const lessons = useMemo(() => parts.flatMap((part) => part.lessons), [parts]);
  const numbers = useMemo(() => lessonNumbers(lessons), [lessons]);
  const demos = Object.keys(DEMOS).length;
  const solved = solvedCount(lessons, progress);
  const minutes = lessons.reduce((sum, l) => sum + (l.minutes ?? 0), 0);
  const next = useMemo(() => nextLesson(lessons, progress), [lessons, progress]);
  const resuming = next ? !!progress[next.slug] : false;

  const runs = me?.stats.runs ?? 0;
  const progressDetail =
    runs > 0 ? `${runs} runs · ${formatSeconds(me?.stats.compute_seconds ?? 0)} of compute` : undefined;

  return (
    <Overview
      eyebrow="Weights and wires · Speech and audio"
      pill="Read, build, and check"
      title={
        <>
          Speech and audio, <em>one build</em> at a time
        </>
      }
      intro="Each lesson explains one idea in plain words, works a small example by hand, and then walks you through building it. Your code runs against checks that say exactly what is right and what is not, and what it plays or plots appears beside its output."
      primary={
        next
          ? {
              href: `/lessons/${next.slug}`,
              label: solved === lessons.length && lessons.length > 0 ? "Review the course" : resuming ? `Continue: ${next.title}` : "Start the course",
            }
          : undefined
      }
      secondary={demos > 0 ? [{ href: "/demos", label: "All demos" }] : []}
      next={
        next
          ? {
              href: `/lessons/${next.slug}`,
              kicker: `${resuming ? "Continue where you left off" : "Start here"} · Lesson ${numbers.get(next.slug) ?? ""}`,
              title: next.title,
              summary: next.summary,
              action: resuming ? "Continue" : "Start lesson",
            }
          : undefined
      }
      progress={{ done: solved, total: lessons.length, label: "Lessons solved", detail: progressDetail }}
      facts={[
        { icon: "book", value: String(lessons.length), label: "lessons with checks" },
        { icon: "layers", value: String(parts.length), label: "parts" },
        ...(demos > 0 ? [{ icon: "spark" as const, value: String(demos), label: "interactive demos" }] : []),
        ...(minutes > 0 ? [{ icon: "clock" as const, value: `~${Math.round(minutes / 60)} h`, label: "of reading" }] : []),
      ]}
      path={{
        eyebrow: "The learning path",
        title: "The course, part by part",
        aside: `${parts.length} parts · ${lessons.length} lessons`,
      }}
      sections={parts.map((part, index) => {
        const partMinutes = part.lessons.reduce((sum, l) => sum + (l.minutes ?? 0), 0);
        const first = part.lessons[0];
        const last = part.lessons[part.lessons.length - 1];
        return {
          id: part.part,
          number: partNumber(part.part, index),
          title: splitPart(part.part).name,
          meta: [
            first && last && first !== last ? `Lessons ${numbers.get(first.slug)} to ${numbers.get(last.slug)}` : `${part.lessons.length} lesson`,
            partMinutes > 0 ? `${Math.round(partMinutes / 6) / 10} h` : "",
          ]
            .filter(Boolean)
            .join(" · "),
          lessons: part.lessons.map((lesson) => ({
            id: lesson.slug,
            href: `/lessons/${lesson.slug}`,
            title: lesson.title,
            number: numbers.get(lesson.slug),
            done: progress[lesson.slug]?.status === "solved",
            tag: lesson.gpu ? "GPU" : undefined,
          })),
        };
      })}
    >
      {me && me.runs.length > 0 && (
        <OverviewBlock eyebrow="Activity" title="Recent runs" id="sh-runs-heading">
          <OverviewList
            items={me.runs.slice(0, 8).map((run, index) => ({
              id: `${run.exercise_id}-${index}`,
              href: `/lessons/${run.exercise_id}`,
              kicker: `${ago(run.started_at)}${run.backend === "runpod" ? " · GPU" : ""}`,
              title: titleOf(run.exercise_id, lessons),
              detail:
                run.mode === "check" && run.total != null
                  ? `${run.passed ?? 0} of ${run.total} checks passed`
                  : run.ok
                    ? "Ran without errors"
                    : "Run failed",
              trailing: (
                <span
                  className={`size-1.5 shrink-0 rounded-full ${run.ok ? "bg-good" : "bg-bad"}`}
                  role="img"
                  aria-label={run.ok ? "Succeeded" : "Failed"}
                />
              ),
            }))}
          />
        </OverviewBlock>
      )}
    </Overview>
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
