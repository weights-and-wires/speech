import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import ExercisePane from "../components/ExercisePane";
import LessonReader from "../components/LessonReader";
import SplitPane from "../components/SplitPane";
import { api, ApiError, type Lesson, type LessonMeta } from "../lib/api";
import { REVEAL_EVENT, revealInEditor } from "../lib/events";
import { titleOf } from "../lib/parts";
import { WIDE, useMediaQuery } from "../lib/useMediaQuery";

type Props = {
  gpuAvailable: boolean;
  /** Every lesson, in order, so neighbours and prerequisites show real titles. */
  lessons: LessonMeta[];
  onProgress: () => void;
};

export default function LessonPage({ gpuAvailable, lessons, onProgress }: Props) {
  const { slug = "" } = useParams();
  const navigate = useNavigate();
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Narrow screens show the lesson and the exercise one at a time.
  const [tab, setTab] = useState<"first" | "second">("first");
  const wide = useMediaQuery(WIDE);

  useEffect(() => {
    let live = true;
    setLesson(null);
    setError(null);
    setTab("first");
    api
      .lesson(slug)
      .then((body) => {
        if (!live) return;
        setLesson(body);
        onProgress();
      })
      .catch((exc) => {
        if (!live) return;
        setError(
          exc instanceof ApiError && exc.status === 404
            ? "There is no lesson at that address."
            : "Could not load the lesson.",
        );
      });
    return () => {
      live = false;
    };
  }, [slug, onProgress]);

  // Left and right arrows move between lessons when the focus is not in an
  // input or the editor.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
          target.closest(".monaco-editor"))
      ) {
        return;
      }
      if (event.key === "[" && lesson?.prev) navigate(`/lessons/${lesson.prev}`);
      if (event.key === "]" && lesson?.next) navigate(`/lessons/${lesson.next}`);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lesson, navigate]);

  // On a narrow screen, a jump to the editor has to show the exercise tab
  // first. The editor itself handles the jump.
  useEffect(() => {
    const onReveal = () => setTab("second");
    window.addEventListener(REVEAL_EVENT, onReveal);
    return () => window.removeEventListener(REVEAL_EVENT, onReveal);
  }, []);

  const onSolved = useCallback(() => {
    onProgress();
  }, [onProgress]);

  if (error) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div>
          <p className="font-serif text-lg text-ink">{error}</p>
          <Link to="/" className="btn btn-secondary mt-4">
            Back to the contents
          </Link>
        </div>
      </div>
    );
  }

  if (!lesson) {
    return (
      <div className="flex flex-1 items-center justify-center text-[13px] text-ink-4" aria-busy="true">
        Loading the lesson…
      </div>
    );
  }

  const reading = (
    <LessonReader
      lesson={lesson}
      lessons={lessons}
      onStartExercise={lesson.exercise_detail ? () => revealInEditor() : undefined}
    />
  );

  if (!lesson.exercise_detail) {
    return <div className="flex min-h-0 flex-1 flex-col">{reading}</div>;
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      {!wide && (
        <div
          role="tablist"
          aria-label="Lesson and exercise"
          className="flex shrink-0 gap-1 border-b border-rule bg-chrome p-1.5"
        >
          <Tab
            selected={tab === "first"}
            onSelect={() => setTab("first")}
            label="Lesson"
          />
          <Tab
            selected={tab === "second"}
            onSelect={() => setTab("second")}
            label="Exercise"
          />
        </div>
      )}

      <SplitPane
        storageKey="speech.split.lesson"
        initial={52}
        min={28}
        max={72}
        className="min-h-0 flex-1"
        label="Resize the lesson and the editor"
        stacked={!wide}
        active={tab}
        first={reading}
        second={
          <ExercisePane
            key={lesson.exercise_detail.id}
            exercise={lesson.exercise_detail}
            initialCode={lesson.draft ?? lesson.exercise_detail.starter}
            gpuAvailable={gpuAvailable}
            gpuLesson={lesson.gpu}
            next={lesson.next ? { slug: lesson.next, title: titleOf(lesson.next, lessons) } : null}
            onSolved={onSolved}
          />
        }
      />
    </div>
  );
}

/** One of the two panes a narrow screen shows at a time. */
function Tab({
  selected,
  onSelect,
  label,
}: {
  selected: boolean;
  onSelect: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={selected}
      onClick={onSelect}
      className={`min-h-9 flex-1 rounded-md px-3 text-[13.5px] font-medium transition-colors ${
        selected
          ? "bg-raise text-ink shadow-[inset_0_0_0_1px_var(--rule-strong)]"
          : "text-ink-3 hover:text-ink"
      }`}
    >
      {label}
    </button>
  );
}
